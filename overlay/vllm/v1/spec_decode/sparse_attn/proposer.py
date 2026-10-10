# SPDX-License-Identifier: Apache-2.0
# Modified for sparse_atten_spec: StreamingLLM and Vegas port; see NOTICE.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse self-drafting with target weights and full-KV verification.

Adapted from npz7yyk/vegas commit 79085a31 (Apache-2.0).
"""

from dataclasses import replace
from math import ceil
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from vllm import envs
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.config import (
    CompilationMode,
    CUDAGraphMode,
    VllmConfig,
    get_layers_from_vllm_config,
)
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import apply_sampling_constraints
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.spec_decode.sparse_attn.streamingllm import (
    draft_slot_mapping,
    gather_draft_hidden_states,
    prepare_draft_buffers,
    streamingllm_block_table,
)
from vllm.v1.utils import CpuGpuBuffer

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class SparseAttnProposer:
    # Reuse the current vLLM kernels for discarded requests and rejection counts.
    prepare_next_token_ids_padded = SpecDecodeBaseProposer.prepare_next_token_ids_padded
    prepare_inputs_padded = SpecDecodeBaseProposer.prepare_inputs_padded

    def __init__(
        self, vllm_config: VllmConfig, device: torch.device, runner: "GPUModelRunner"
    ):
        self.vllm_config = vllm_config
        self.runner = runner
        spec = vllm_config.speculative_config
        assert spec is not None
        self.speculative_config = spec
        self.vegas = None
        if spec.sparse_attn_algorithm == "vegas":
            from vllm.vllm_flash_attn.flash_attn_interface import vegas_fa3_supported

            supported, reason = vegas_fa3_supported()
            if not supported:
                raise ValueError(f"Vegas requires patched FA3 on SM90: {reason}")
            if vllm_config.model_config.dtype != torch.bfloat16:
                raise ValueError("Vegas requires a BF16 model and BF16 KV cache.")
            if vllm_config.cache_config.cache_dtype not in ("auto", "bfloat16"):
                raise ValueError("Vegas requires a BF16 KV cache.")
        parallel = vllm_config.parallel_config
        if (
            vllm_config.use_v2_model_runner
            or parallel.tensor_parallel_size != 1
            or parallel.pipeline_parallel_size != 1
            or parallel.data_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or vllm_config.model_config.is_multimodal_model
            or vllm_config.lora_config is not None
            or spec.disable_padded_drafter_batch
            or spec.parallel_drafting
            or spec.uses_dynamic_speculative_decoding()
            or spec.rejection_sample_method != "standard"
            or vllm_config.compilation_config.mode
            == CompilationMode.STOCK_TORCH_COMPILE
        ):
            raise ValueError(
                "sparse_attn currently requires single-GPU, text-only V1 "
                "decoding with fixed sequential drafts, standard rejection, "
                "padded batches, no LoRA and no stock torch.compile."
            )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.num_speculative_tokens = spec.num_speculative_tokens
        max_batch = vllm_config.scheduler_config.max_num_seqs
        self.input_ids = torch.zeros(max_batch, dtype=torch.int32, device=device)
        self.positions = torch.full(
            (max_batch,), self.max_model_len, dtype=torch.int64, device=device
        )
        self.draft_tokens = torch.empty(
            (max_batch, self.num_speculative_tokens), dtype=torch.int32, device=device
        )
        self.graph_runner: CUDAGraphWrapper | None = None
        self.random_drafting = False
        self.draft_hidden_states = torch.empty(
            (
                max_batch,
                self.num_speculative_tokens,
                vllm_config.model_config.get_hidden_size(),
            ),
            dtype=vllm_config.model_config.dtype,
            device=device,
        )
        self.draft_req_ids: list[str] = []
        self.draft_sampling_metadata: SamplingMetadata | None = None
        self.probabilistic_proposals = 0
        self.graph_sizes: list[int] = []
        self.graph_replays = 0
        self.padded_graph_replays = 0
        self.backup_next_token_ids = CpuGpuBuffer(
            max_batch,
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
            device=device,
            with_numpy=True,
        )
        self.query_start_loc = torch.arange(
            max_batch + 1, dtype=torch.int32, device=device
        )
        self.attn_layer_names: list[str] = []
        self.block_size = 0  # Resolved from the actual attention kernel layout.

    def load_model(self, target_model: nn.Module) -> None:
        self.model = target_model
        self.attn_layer_names = list(
            get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase)
        )
        if self.speculative_config.sparse_attn_algorithm == "vegas":
            from vllm.v1.spec_decode.sparse_attn.vegas import VegasState

            # Reserve persistent buffers before the runner profiles its KV budget.
            self.vegas = VegasState(
                self.vllm_config, self.input_ids.device, self.attn_layer_names, 0
            )
            self.vegas.warmup()

    def validate_same_kv_cache_group(self, kv_cache_config: KVCacheConfig) -> None:
        groups = kv_cache_config.kv_cache_groups
        if len(groups) != 1 or not isinstance(
            groups[0].kv_cache_spec, FullAttentionSpec
        ):
            raise ValueError("sparse_attn requires one full-attention KV cache group.")
        spec = groups[0].kv_cache_spec
        if spec.sliding_window or spec.attention_chunk_size or spec.non_causal:
            raise ValueError("sparse_attn requires causal global attention layers.")
        if not set(self.attn_layer_names).issubset(groups[0].layer_names):
            raise ValueError("All target attention layers must share the KV group.")
        attn_groups = self.runner.attn_groups[0]
        if any(group.backend.get_name() != "FLASH_ATTN" for group in attn_groups):
            raise ValueError(
                "sparse_attn requires attention_config.backend=FLASH_ATTN."
            )
        self.block_size = self.runner._kernel_block_sizes[0]
        table = self.runner.input_batch.block_table.block_tables[0].block_table.gpu
        self.full_table = torch.zeros_like(table)
        if self.speculative_config.sparse_attn_algorithm == "vegas":
            assert self.vegas is not None
            self.vegas.block_size = self.block_size
            return
        max_tokens = max(
            ceil(self.max_model_len * self.speculative_config.sparse_attn_ratio),
            self.speculative_config.sparse_attn_min_tokens,
        )
        max_blocks = (max_tokens + self.block_size - 1) // self.block_size
        if max_blocks < 2:
            raise ValueError("StreamingLLM needs capacity for sink and tail blocks.")
        self.sparse_table = torch.empty(
            (table.shape[0], max_blocks), dtype=table.dtype, device=table.device
        )
        self.sparse_lens = torch.empty_like(self.input_ids)

    @torch.inference_mode()
    def capture_model(self) -> None:
        """Called inside the runner's graph capture lifecycle, after KV allocation."""
        if (
            self.vllm_config.model_config.enforce_eager
            or self.speculative_config.enforce_eager
        ):
            return
        config = self.vllm_config.compilation_config
        max_batch = self.input_ids.shape[0]
        self.graph_sizes = sorted(
            {1, max_batch}
            | {
                size
                for size in config.cudagraph_capture_sizes or []
                if 0 < size <= max_batch
            }
        )
        self.graph_runner = CUDAGraphWrapper(
            self._forward, self.vllm_config, runtime_mode=CUDAGraphMode.FULL
        )
        # Invalid positions keep warmup/capture from overwriting any real KV.
        self.input_ids.zero_()
        self.positions.fill_(self.max_model_len)
        self.full_table.zero_()
        for size in reversed(self.graph_sizes):
            for _ in range(max(1, config.cudagraph_num_of_warmups)):
                self._execute(size, CUDAGraphMode.NONE)
            torch.cuda.synchronize()
            self._execute(size, CUDAGraphMode.FULL)
            torch.cuda.synchronize()

    def _execute(self, size: int, mode: CUDAGraphMode) -> torch.Tensor:
        with set_forward_context(
            None,
            self.vllm_config,
            num_tokens=size,
            cudagraph_runtime_mode=mode,
            batch_descriptor=BatchDescriptor(
                num_tokens=size, num_reqs=size, uniform=True
            ),
        ):
            if self.graph_runner is None:
                return self._forward(size)
            return self.graph_runner(size)

    def propose(
        self,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs=None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
        slot_mappings=None,
    ) -> torch.Tensor:
        self.validate_sampling_metadata(sampling_metadata)
        assert self.block_size > 0
        batch = next_token_ids.shape[0]
        size = next((size for size in self.graph_sizes if size >= batch), batch)
        self.random_drafting = not sampling_metadata.all_greedy
        self.draft_req_ids = self.runner.input_batch.req_ids[:batch].copy()
        self.draft_sampling_metadata = replace(
            sampling_metadata,
            temperature=(
                sampling_metadata.temperature.clone()
                if sampling_metadata.temperature is not None
                else None
            ),
            top_p=(
                sampling_metadata.top_p.clone()
                if sampling_metadata.top_p is not None
                else None
            ),
            top_k=(
                sampling_metadata.top_k.clone()
                if sampling_metadata.top_k is not None
                else None
            ),
            max_num_logprobs=None,
            logprob_token_ids=None,
        )
        prepare_draft_buffers(
            self.input_ids[:size],
            self.positions[:size],
            self.full_table[:size],
            next_token_ids,
            common_attn_metadata.seq_lens,
            common_attn_metadata.block_table_tensor,
            self.max_model_len,
            num_rejected_tokens_gpu,
        )
        if self.vegas is not None:
            # Unfinished prefill rows must neither draft nor write their KV.
            self.positions[:batch].masked_fill_(
                self.runner.discard_request_mask.gpu[:batch], self.max_model_len
            )
        descriptor = BatchDescriptor(num_tokens=size, num_reqs=size, uniform=True)
        wrapper = self.graph_runner
        entry = (
            wrapper.concrete_cudagraph_entries.get(descriptor)
            if wrapper is not None
            else None
        )
        # No runtime capture: only replay entries prepared during startup.
        mode = CUDAGraphMode.NONE
        if entry is not None and entry.cudagraph is not None:
            mode = CUDAGraphMode.FULL
        for step in range(self.num_speculative_tokens):
            if mode == CUDAGraphMode.FULL:
                self.graph_replays += 1
                self.padded_graph_replays += int(size > batch)
            hidden = self._execute(size, mode)[:batch]
            self.draft_hidden_states[:batch, step].copy_(hidden)
            logits = self.model.compute_logits(hidden)
            sampled = self.runner.sampler(
                logits,
                replace(
                    sampling_metadata, max_num_logprobs=None, logprob_token_ids=None
                ),
            ).sampled_token_ids.flatten()
            self.draft_tokens[:batch, step].copy_(sampled)
            self.input_ids[:batch].copy_(sampled)
            self.positions[:batch].add_(1)
        if self.random_drafting:
            self.probabilistic_proposals += 1
        return self.draft_tokens[:batch]

    def validate_sampling_metadata(self, metadata: SamplingMetadata) -> None:
        # Rebuilding q must reproduce every transform used by the sampler.
        if (
            not metadata.no_penalties
            or metadata.bad_words_token_ids
            or metadata.allowed_token_ids_mask is not None
            or (
                metadata.thinking_budget_state_holder is not None
                and metadata.thinking_budget_state_holder.has_tracked_requests()
            )
        ):
            raise ValueError(
                "sparse_attn q reconstruction supports temperature/top-k/top-p only."
            )
        for req_id in self.runner.input_batch.req_ids:
            params = self.runner.requests[req_id].sampling_params
            if params is not None and (
                params.min_p or params.min_tokens or params.logit_bias
            ):
                raise ValueError(
                    "sparse_attn q reconstruction does not support "
                    "min_p/min_tokens/logit_bias."
                )

    def get_draft_probs(self, spec_decode_metadata, req_ids: list[str]):
        metadata = self.draft_sampling_metadata
        if metadata is None or metadata.all_greedy:
            return None
        counts = spec_decode_metadata.num_draft_tokens
        if len(counts) != len(req_ids):
            raise ValueError("Draft counts and request IDs are misaligned.")
        if not any(counts):
            return None
        row_by_id = {req_id: row for row, req_id in enumerate(self.draft_req_ids)}
        rows = []
        for req_id, count in zip(req_ids, counts):
            if count < 0 or count > self.num_speculative_tokens:
                raise ValueError("Invalid draft count for cached hidden states.")
            row = row_by_id.get(req_id)
            if count and row is None:
                raise RuntimeError(f"Missing draft hidden states for request {req_id}")
            rows.append(row if row is not None else 0)
        indices = None
        if req_ids != self.draft_req_ids:
            indices = torch.tensor(
                rows, dtype=torch.int64, device=self.draft_hidden_states.device
            )
        if req_ids == self.draft_req_ids and all(
            count == self.num_speculative_tokens for count in counts
        ):
            hidden = self.draft_hidden_states[: len(req_ids)].reshape(
                -1, self.draft_hidden_states.shape[-1]
            )
        else:
            hidden = gather_draft_hidden_states(
                self.draft_hidden_states,
                spec_decode_metadata.cu_num_draft_tokens,
                sum(counts),
                len(req_ids),
                indices,
            )
        if indices is not None:
            metadata = replace(
                metadata,
                temperature=metadata.temperature[indices],
                top_p=metadata.top_p[indices] if metadata.top_p is not None else None,
                top_k=metadata.top_k[indices] if metadata.top_k is not None else None,
            )
        logits = self.model.compute_logits(hidden).float()
        logits = apply_sampling_constraints(
            logits, spec_decode_metadata.cu_num_draft_tokens, metadata
        )
        # Native rejection ignores q on greedy rows and uses argmax verification.
        return logits.softmax(-1, dtype=torch.float32)

    def _forward(self, batch: int) -> torch.Tensor:
        positions = self.positions[:batch]
        full_table = self.full_table[:batch]
        input_ids = self.input_ids[:batch]
        safe_positions, slots = draft_slot_mapping(
            positions, full_table, self.block_size, self.max_model_len
        )
        vegas = getattr(self, "vegas", None)
        if vegas is None:
            sparse_table, sparse_lens = streamingllm_block_table(
                full_table,
                (safe_positions + 1).to(torch.int32),
                self.block_size,
                self.speculative_config.sparse_attn_ratio,
                self.speculative_config.sparse_attn_min_tokens,
                out=(self.sparse_table[:batch], self.sparse_lens[:batch]),
            )
            max_seq_len = min(
                self.max_model_len, sparse_table.shape[1] * self.block_size
            )
        else:
            sparse_table = full_table
            sparse_lens = vegas.lengths[0, :batch]
            max_seq_len = vegas.slot_width
        metadata = FlashAttentionMetadata(
            num_actual_tokens=batch,
            max_query_len=1,
            query_start_loc=self.query_start_loc[: batch + 1],
            max_seq_len=max_seq_len,
            seq_lens=sparse_lens,
            block_table=sparse_table,
            slot_mapping=slots,
            use_cascade=False,
            common_prefix_len=0,
            cu_prefix_query_lens=None,
            prefix_kv_lens=None,
            suffix_kv_lens=None,
        )
        # Reuse native FA scheduling, without overwriting target graph buffers.
        attn_metadata = {}
        for group in self.runner.attn_groups[0]:
            if vegas is not None:
                for name in group.layer_names:
                    token_slots, lengths = vegas.draft_slots(
                        name, full_table, positions
                    )
                    attn_metadata[name] = replace(
                        metadata,
                        seq_lens=lengths,
                        sparse_token_slots=token_slots,
                        max_num_splits=1,
                    )
                continue
            builder = group.get_metadata_builder()
            group_metadata = replace(metadata)
            if self.graph_runner is not None and builder.aot_schedule:
                attention_config = self.vllm_config.attention_config
                group_metadata.max_num_splits = (
                    attention_config.flash_attn_max_num_splits_for_cuda_graph
                )
            if builder.fa4_hd256 or envs.VLLM_BATCH_INVARIANT:
                group_metadata.max_num_splits = 1
            group_metadata.scheduler_metadata = builder._get_scheduler_metadata(
                aot_schedule=builder.aot_schedule,
                batch_size=batch,
                cu_query_lens=metadata.query_start_loc,
                max_query_len=1,
                seqlens=sparse_lens,
                max_seq_len=metadata.max_seq_len,
                causal=True,
                max_num_splits=group_metadata.max_num_splits,
            )
            attn_metadata.update(dict.fromkeys(group.layer_names, group_metadata))
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=batch,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            slot_mapping=dict.fromkeys(self.attn_layer_names, slots),
            # Compile the model, but let the outer per-step wrapper own graphs.
            skip_compiled=False,
        ):
            return self.model(input_ids=input_ids, positions=safe_positions)

    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs=False,
        is_graph_capturing=False,
        attn_metadata=None,
        slot_mappings=None,
    ) -> None:
        with set_forward_context(
            None,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            skip_compiled=False,
        ):
            self.model(
                input_ids=self.runner.input_ids.gpu[:num_tokens],
                positions=self.runner.positions[:num_tokens],
            )

# SPDX-License-Identifier: Apache-2.0
# Modified for sparse_atten_spec: StreamingLLM and Vegas port; see NOTICE.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verification-guided, per-layer KV selection (Vegas weight mode).

Selections are refreshed for every scheduled row by the target forward, so
request churn/reordering cannot retain another request's selection. Drafts
translate logical indices against the current full block table, not old pages.
"""

from math import ceil

import torch
import triton
import triton.language as tl

from vllm.v1.spec_decode.sparse_attn import varlen_reduce, varlen_topk


@triton.jit
def _slots_kernel(
    indices,
    table,
    budgets,
    history,
    positions,
    slots,
    lengths,
    index_stride: tl.constexpr,
    table_stride: tl.constexpr,
    slot_stride: tl.constexpr,
    block_size: tl.constexpr,
    max_model_len: tl.constexpr,
    max_budget: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    position = tl.load(positions + row)
    valid = tl.load(history + row)
    active = (position >= valid - 1) & (position < max_model_len) & (valid > 0)
    budget = tl.minimum(tl.load(budgets + row), valid)
    budget = tl.where(active, budget, 0)
    recent = tl.where(active, tl.maximum(position + 1 - valid, 0), 0)
    length = budget + recent
    tl.store(lengths + row, tl.minimum(length, WIDTH))
    for start in range(0, WIDTH, BLOCK):
        col = start + tl.arange(0, BLOCK)
        selected = tl.load(
            indices + row * index_stride + col,
            mask=(col < budget) & (col < max_budget),
            other=-1,
        )
        logical = tl.where(col < budget, selected, valid + col - budget)
        mask = (col < length) & (col < WIDTH) & (logical >= 0)
        page = tl.load(
            table + row * table_stride + logical // block_size, mask=mask, other=0
        )
        slot = page * block_size + logical % block_size
        tl.store(
            slots + row * slot_stride + col, tl.where(mask, slot, -1), mask=col < WIDTH
        )


class VegasState:
    def __init__(self, config, device, layer_names: list[str], block_size: int):
        model = config.model_config
        spec = config.speculative_config
        self.layer_rows = {name: i for i, name in enumerate(layer_names)}
        self.block_size = block_size
        self.max_model_len = model.max_model_len
        batch = config.scheduler_config.max_num_seqs
        heads = model.get_num_attention_heads(config.parallel_config)
        self.ratio = spec.sparse_attn_ratio
        self.min_tokens = spec.sparse_attn_min_tokens
        self.max_budget = min(
            self.max_model_len,
            max(ceil(self.max_model_len * self.ratio), self.min_tokens),
        )
        self.slot_width = self.max_budget + 2 * spec.num_speculative_tokens + 1
        score_width = triton.cdiv(self.max_model_len, 8) * 8
        self.scores = torch.empty(
            (batch, heads, 2, score_width), dtype=torch.bfloat16, device=device
        )
        self.metric = torch.empty(
            (batch, score_width), dtype=torch.bfloat16, device=device
        )
        self.indices = torch.full(
            (len(layer_names), batch, self.max_budget),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.slots = torch.full(
            (len(layer_names), batch, self.slot_width),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.lengths = torch.zeros(
            (len(layer_names), batch), dtype=torch.int32, device=device
        )
        self.prefilling = torch.zeros(batch, dtype=torch.bool, device=device)
        self.valid_lens = torch.zeros(batch, dtype=torch.int32, device=device)
        self.budgets = torch.zeros_like(self.valid_lens)
        self.entries = torch.zeros_like(self.valid_lens)
        self.workspace = torch.empty(
            varlen_topk.calc_topk_workspace_size(batch, score_width, self.max_budget),
            dtype=torch.uint8,
            device=device,
        )

    def warmup(self) -> None:
        varlen_reduce.warmup()
        varlen_topk.warmup()

    def prepare_target(self, prefilling: torch.Tensor, num_reqs: int) -> None:
        """Refresh fixed-address phase input before target capture/replay."""
        self.prefilling.zero_()
        self.prefilling[:num_reqs].copy_(prefilling[:num_reqs])

    def scores_for(self, layer_name: str, batch: int) -> torch.Tensor:
        # Scratch is consumed before the next attention layer reuses it.
        assert layer_name in self.layer_rows
        return self.scores[:batch]

    def select(self, layer_name, scores, lse, metadata, softmax_scale) -> None:
        batch = metadata.seq_lens.shape[0]
        qlens = metadata.query_start_loc[1:] - metadata.query_start_loc[:-1]
        seq_lens = metadata.seq_lens
        prefill = self.prefilling[:batch]
        active = (qlens > 0) & (seq_lens > 0)
        valid = torch.where(prefill, seq_lens, seq_lens + 1 - qlens)
        self.valid_lens[:batch].copy_(
            torch.where(active, valid, 0).clamp(min=0, max=self.max_model_len)
        )
        self.entries[:batch].copy_(
            torch.where(prefill & (qlens > 1), 2, torch.where(qlens == 1, 1, 0))
        )
        budget = torch.ceil(seq_lens.float() * self.ratio).to(torch.int32)
        self.budgets[:batch].copy_(
            torch.minimum(
                budget.clamp(min=self.min_tokens, max=self.max_budget),
                self.valid_lens[:batch],
            )
        )
        varlen_reduce.varlen_reduce(
            scores,
            self.valid_lens[:batch],
            self.entries[:batch],
            self.metric[:batch],
            lse,
            metadata.query_start_loc,
            softmax_scale,
            True,
        )
        varlen_topk.varlen_topk(
            self.metric[:batch],
            self.budgets[:batch],
            self.valid_lens[:batch],
            self.indices[self.layer_rows[layer_name], :batch],
            self.workspace,
        )

    def draft_slots(self, layer_name, full_table, positions):
        batch = positions.shape[0]
        row = self.layer_rows[layer_name]
        slots = self.slots[row, :batch]
        lengths = self.lengths[row, :batch]
        _slots_kernel[(batch,)](
            self.indices[row, :batch],
            full_table,
            self.budgets,
            self.valid_lens,
            positions,
            slots,
            lengths,
            self.indices.stride(1),
            full_table.stride(0),
            slots.stride(0),
            self.block_size,
            self.max_model_len,
            self.max_budget,
            self.slot_width,
            256,
        )
        return slots, lengths

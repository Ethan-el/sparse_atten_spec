# SPDX-License-Identifier: Apache-2.0
# Added or modified for sparse_atten_spec validation; see NOTICE.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU window checks with optional production runner and CUDA regressions."""

import argparse
import importlib.util
from math import ceil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


def check_vegas_wrapper():
    """Check the additive FA3 ABI without invoking FlashAttention kernels."""
    source = (
        Path(__file__).resolve().parents[1]
        / "vllm/vllm_flash_attn/flash_attn_interface.py"
    )
    spec = importlib.util.spec_from_file_location("vegas_fa_wrapper", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    q = torch.empty((2, 2, 8), dtype=torch.bfloat16)
    cache = torch.empty((3, 2, 16, 16), dtype=torch.bfloat16)
    k, v = cache.transpose(1, 2).split(8, dim=-1)
    out = torch.empty_like(q)
    lse = torch.empty((2, 2), dtype=torch.float32)
    kwargs = dict(
        q=q,
        k=k,
        v=v,
        out=out,
        max_seqlen_q=1,
        cu_seqlens_q=torch.arange(3, dtype=torch.int32),
        max_seqlen_k=16,
        seqused_k=torch.tensor([16, 16], dtype=torch.int32),
        block_table=torch.tensor([[1], [2]], dtype=torch.int32),
        causal=True,
        num_splits=1,
        fa_version=3,
    )
    scores = torch.empty((2, 2, 2, 16), dtype=torch.bfloat16)
    slots = torch.empty((2, 4), dtype=torch.int32)
    with patch.object(torch.ops._vllm_fa3_C, "fwd", create=True) as fwd:
        fwd.return_value = (out, lse, None, None)
        assert module.flash_attn_varlen_func(**kwargs) is out
        args, extras = fwd.call_args
        assert len(args) == 37 and not extras, "Legacy FA3 ABI changed"
        assert args[1].data_ptr() == k.data_ptr() and args[1].stride() == k.stride()
        assert args[2].data_ptr() == v.data_ptr() and args[2].stride() == v.stride()
        result = module.flash_attn_varlen_func(
            **kwargs, scores=scores, return_softmax_lse=True
        )
        assert result[0] is out and result[1] is lse
        assert fwd.call_args.kwargs.keys() == {"scores"}
        assert fwd.call_args.kwargs["scores"] is scores
        assert module.flash_attn_varlen_func(**kwargs, sparse_token_slots=slots) is out
        assert fwd.call_args.kwargs.keys() == {"sparse_token_slots"}
        assert fwd.call_args.kwargs["sparse_token_slots"] is slots
        calls = fwd.call_count
        for version in (2, 4):
            try:
                module.flash_attn_varlen_func(
                    **dict(kwargs, fa_version=version), scores=scores
                )
            except NotImplementedError:
                pass
            else:
                raise AssertionError(f"FA{version} accepted Vegas arguments")
        try:
            module.flash_attn_varlen_func(
                **kwargs, scores=scores, sparse_token_slots=slots
            )
        except ValueError:
            pass
        else:
            raise AssertionError("Score collection and sparse draft were combined")
        assert fwd.call_count == calls, "Unsupported call reached a CUDA operator"
        with patch.object(module, "_is_fa3_supported", return_value=(True, None)):
            fwd.default._schema.arguments = [
                SimpleNamespace(name=name) for name in ("scores", "sparse_token_slots")
            ]
            assert module.vegas_fa3_supported() == (True, None)
            fwd.default._schema.arguments = [SimpleNamespace(name="q")]
            supported, reason = module.vegas_fa3_supported()
            assert not supported and "scores" in reason
        with (
            patch.object(module, "FA3_AVAILABLE", True),
            patch.dict(
                "sys.modules",
                {
                    "vllm.platforms": SimpleNamespace(
                        current_platform=SimpleNamespace(
                            is_device_capability_family=lambda capability: False
                        )
                    )
                },
            ),
        ):
            supported, reason = module.vegas_fa3_supported()
            assert not supported and "9.x" in reason, "SM89 enabled FA3"
    supported, reason = module.vegas_fa3_supported()
    assert not supported and reason, "CPU-only wrapper unexpectedly enabled Vegas"
    print("PASS: Vegas wrapper ABI, strided KV, schema and unsupported-device gates")


def check_runner_cache():
    """Use the production cache path for reorder, truncation and batch churn."""
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    cached = torch.arange(24, dtype=torch.float32).view(2, 3, 4)
    original = cached.clone()
    runner = SimpleNamespace(
        _draft_probs=cached,
        _draft_prob_req_ids=["a", "b"],
        input_batch=SimpleNamespace(req_ids=["a", "b"]),
    )
    flatten = GPUModelRunner._get_spec_decode_draft_probs
    result = flatten(runner, SimpleNamespace(num_draft_tokens=[3, 3]))
    assert torch.equal(result, cached.reshape(6, 4))
    assert result.data_ptr() == cached.data_ptr(), "Fast path copied probabilities"
    for ids, counts, expected in (
        (["b", "a"], [2, 1], torch.cat([cached[1, :2], cached[0, :1]])),
        (["b", "c"], [3, 0], cached[1]),
        (["b"], [2], cached[1, :2]),
        (["a", "b"], [0, 0], None),
        (["c"], [1], None),
    ):
        runner.input_batch.req_ids = ids
        result = flatten(runner, SimpleNamespace(num_draft_tokens=counts))
        if expected is None:
            assert result is None
        else:
            assert torch.equal(result, expected)
    assert torch.equal(cached, original)
    print("PASS: production q cache fast path, reorder, truncation and batch churn")
    check_proposer_flow()
    check_compilation_context()


def check_compilation_context():
    """Exercise real forward contexts with a CPU Dynamo eager-backend probe."""
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context, is_forward_context_available
    from vllm.v1.spec_decode.sparse_attn.proposer import SparseAttnProposer

    drafter = SparseAttnProposer.__new__(SparseAttnProposer)
    drafter.vegas = None
    drafter.graph_runner = None
    drafter.vllm_config = SimpleNamespace(
        attention_config=SimpleNamespace(flash_attn_max_num_splits_for_cuda_graph=8),
        compilation_config=SimpleNamespace(
            fast_moe_cold_start=False, static_forward_context={}
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=1, use_sequence_parallel_moe=False, is_moe_model=False
        ),
    )
    drafter.block_size = 16
    drafter.max_model_len = 128
    drafter.input_ids = torch.tensor([3, 4], dtype=torch.int32)
    drafter.positions = torch.tensor([31, 63], dtype=torch.int64)
    drafter.full_table = torch.arange(16, dtype=torch.int32).reshape(2, 8)
    drafter.sparse_table = torch.empty((2, 4), dtype=torch.int32)
    drafter.sparse_lens = torch.empty(2, dtype=torch.int32)
    drafter.query_start_loc = torch.arange(3, dtype=torch.int32)
    drafter.speculative_config = SimpleNamespace(
        sparse_attn_ratio=0.07, sparse_attn_min_tokens=32
    )
    drafter.attn_layer_names = ["probe"]
    schedules = []

    def schedule(**kwargs):
        schedules.append(kwargs)
        return kwargs["seqlens"].clone() if kwargs["aot_schedule"] else None

    builder = SimpleNamespace(
        aot_schedule=True, fa4_hd256=False, _get_scheduler_metadata=schedule
    )
    group = SimpleNamespace(layer_names=["probe"], get_metadata_builder=lambda: builder)
    drafter.runner = SimpleNamespace(attn_groups=[[group]])
    contexts = []
    compilations = []

    def backend(graph, inputs):
        compilations.append(graph)
        return graph.forward

    compiled = torch.compile(
        lambda ids, pos: ids.float() + pos.float(), backend=backend, dynamic=True
    )

    def model(input_ids, positions):
        context = get_forward_context()
        assert not context.skip_compiled, "Draft bypassed the compiled model"
        assert context.cudagraph_runtime_mode == CUDAGraphMode.NONE
        if context.attn_metadata is not None:
            attn = context.attn_metadata["probe"]
            assert attn.max_seq_len == 64
            if builder.aot_schedule:
                assert torch.equal(attn.scheduler_metadata, attn.seq_lens)
            else:
                assert attn.scheduler_metadata is None
        contexts.append(context)
        return compiled(input_ids, positions)

    drafter.model = model
    outer_modes = []

    def outer_graph(batch):
        outer_modes.append(get_forward_context().cudagraph_runtime_mode)
        return drafter._forward(batch)

    drafter.graph_runner = outer_graph
    assert torch.equal(
        drafter._execute(2, CUDAGraphMode.FULL), torch.tensor([34.0, 67.0])
    )
    assert outer_modes == [CUDAGraphMode.FULL]
    assert schedules[-1]["max_num_splits"] == 8
    drafter.graph_runner = None
    drafter.positions.add_(1)
    assert torch.equal(
        drafter._execute(2, CUDAGraphMode.NONE), torch.tensor([35.0, 68.0])
    )
    assert schedules[-1]["max_num_splits"] == 0
    assert not torch.equal(
        contexts[0].attn_metadata["probe"].scheduler_metadata,
        contexts[1].attn_metadata["probe"].scheduler_metadata,
    )
    builder.aot_schedule = False
    drafter._execute(2, CUDAGraphMode.NONE)
    builder.fa4_hd256 = True
    drafter._execute(2, CUDAGraphMode.NONE)
    assert schedules[-1]["max_num_splits"] == 1
    drafter.runner = SimpleNamespace(
        input_ids=SimpleNamespace(gpu=drafter.input_ids), positions=drafter.positions
    )
    drafter.dummy_run(2)
    assert len(contexts) == 5 and len(compilations) >= 1
    assert not is_forward_context_available(), "Forward context leaked"
    print(
        "PASS: CPU Dynamo probe; draft/warmup allow compilation, no nested graph mode"
    )


def check_proposer_flow():
    """Check production control flow with CPU doubles, not CUDA kernels."""
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import BatchDescriptor
    from vllm.v1.sample.logits_processor import LogitsProcessors
    from vllm.v1.sample.metadata import SamplingMetadata
    from vllm.v1.spec_decode.sparse_attn.proposer import SparseAttnProposer
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    empty = torch.empty(0)
    metadata = SamplingMetadata(
        temperature=torch.tensor([0.6, 1.2]),
        all_greedy=False,
        all_random=True,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=empty,
        presence_penalties=empty,
        repetition_penalties=empty,
        output_token_ids=[[], []],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )
    drafter = SparseAttnProposer.__new__(SparseAttnProposer)
    drafter.vegas = None
    drafter.block_size = 16
    drafter.max_model_len = 128
    drafter.num_speculative_tokens = 3
    drafter.input_ids = torch.zeros(4, dtype=torch.int32)
    drafter.positions = torch.full((4,), 128, dtype=torch.int64)
    drafter.full_table = torch.zeros((4, 8), dtype=torch.int32)
    drafter.draft_tokens = torch.empty((4, 3), dtype=torch.int32)
    drafter.draft_hidden_states = torch.empty((4, 3, 2))
    drafter.graph_sizes = [4]
    drafter.graph_replays = drafter.padded_graph_replays = 0
    drafter.probabilistic_proposals = 0
    descriptor = BatchDescriptor(num_tokens=4, num_reqs=4, uniform=True)
    drafter.graph_runner = SimpleNamespace(
        concrete_cudagraph_entries={descriptor: SimpleNamespace(cudagraph=object())}
    )
    calls = []

    def execute(size, mode):
        calls.append((size, mode, drafter.positions.clone()))
        return torch.stack([drafter.input_ids.float(), drafter.positions.float()], -1)

    drafter._execute = execute
    samples = []

    def sampler(logits, params):
        samples.append(logits.clone())
        return SimpleNamespace(sampled_token_ids=logits.argmax(-1)[:, None].int())

    drafter.model = SimpleNamespace(compute_logits=lambda hidden: hidden)
    params = SimpleNamespace(min_p=0, min_tokens=0, logit_bias=None)
    drafter.runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["a", "b"], sampling_metadata=metadata),
        requests={key: SimpleNamespace(sampling_params=params) for key in ("a", "b")},
        sampler=sampler,
    )
    common = SimpleNamespace(
        seq_lens=torch.tensor([31, 63], dtype=torch.int32),
        block_table_tensor=torch.arange(16, dtype=torch.int32).reshape(2, 8),
    )
    drafter.propose(
        torch.tensor([7, 8], dtype=torch.int32),
        None,
        common,
        metadata,
        num_rejected_tokens_gpu=torch.tensor([1, 2], dtype=torch.int32),
    )
    assert len(calls) == len(samples) == 3
    assert all(size == 4 and mode == CUDAGraphMode.FULL for size, mode, _ in calls)
    assert [positions[:2].tolist() for _, _, positions in calls] == [
        [30, 61],
        [31, 62],
        [32, 63],
    ]
    assert all(positions[2:].tolist() == [128, 128] for _, _, positions in calls)
    assert drafter.graph_replays == drafter.padded_graph_replays == 3
    assert drafter.draft_req_ids == ["a", "b"]
    metadata.temperature.fill_(9)
    assert drafter.draft_sampling_metadata.temperature.tolist() != [9, 9]
    expected_hidden = torch.cat(
        [drafter.draft_hidden_states[1, :2], drafter.draft_hidden_states[0, :1]]
    )
    seen = []
    drafter.model.compute_logits = lambda hidden: seen.append(hidden.clone()) or hidden
    spec = SimpleNamespace(
        num_draft_tokens=[2, 1],
        cu_num_draft_tokens=torch.tensor([2, 3], dtype=torch.int32),
    )

    def constraints(logits, cu_counts, params):
        counts = torch.diff(
            torch.cat([torch.zeros(1, dtype=cu_counts.dtype), cu_counts])
        )
        temperatures = params.temperature.repeat_interleave(counts.long())
        assert torch.allclose(temperatures, torch.tensor([1.2, 1.2, 0.6]))
        return logits / temperatures[:, None]

    with patch(
        "vllm.v1.spec_decode.sparse_attn.proposer.apply_sampling_constraints",
        constraints,
    ):
        worker = SimpleNamespace(
            drafter=drafter, input_batch=SimpleNamespace(req_ids=["b", "a"])
        )
        probs = GPUModelRunner._get_spec_decode_draft_probs(worker, spec)
    assert len(seen) == 1 and torch.equal(seen[0], expected_hidden)
    assert torch.allclose(probs.sum(-1), torch.ones(3))
    assert not hasattr(drafter, "draft_probs")
    same_order = SimpleNamespace(
        num_draft_tokens=[3, 3],
        cu_num_draft_tokens=torch.tensor([3, 6], dtype=torch.int32),
    )

    def unchanged_constraints(logits, cu_counts, params):
        assert params is drafter.draft_sampling_metadata
        return logits

    with (
        patch(
            "vllm.v1.spec_decode.sparse_attn.proposer.apply_sampling_constraints",
            unchanged_constraints,
        ),
        patch(
            "torch.tensor",
            side_effect=AssertionError(
                "Unchanged requests should not upload row indices"
            ),
        ),
    ):
        assert drafter.get_draft_probs(same_order, ["a", "b"]).shape == (6, 2)
        same_order.num_draft_tokens = [2, 1]
        same_order.cu_num_draft_tokens = spec.cu_num_draft_tokens
        assert drafter.get_draft_probs(same_order, ["a", "b"]).shape == (3, 2)
    try:
        drafter.get_draft_probs(spec, ["missing", "a"])
    except RuntimeError:
        pass
    else:
        raise AssertionError("Missing hidden states silently accepted")
    assert (
        drafter.get_draft_probs(SimpleNamespace(num_draft_tokens=[0, 0]), ["new", "a"])
        is None
    )
    metadata.all_random = False
    metadata.temperature.copy_(torch.tensor([0.0, 0.6]))
    drafter.propose(torch.tensor([7, 8], dtype=torch.int32), None, common, metadata)
    assert not drafter.draft_sampling_metadata.all_random
    assert not drafter.draft_sampling_metadata.all_greedy

    def mixed_constraints(logits, cu_counts, params):
        assert params.temperature.tolist()[0] == 0
        assert not params.all_random and not params.all_greedy
        counts = torch.diff(torch.cat([cu_counts.new_zeros(1), cu_counts]))
        temperatures = params.temperature.repeat_interleave(counts.long())
        return logits / torch.where(temperatures == 0, 1.0, temperatures)[:, None]

    with patch(
        "vllm.v1.spec_decode.sparse_attn.proposer.apply_sampling_constraints",
        mixed_constraints,
    ):
        mixed_probs = drafter.get_draft_probs(same_order, ["a", "b"])
    assert torch.allclose(mixed_probs.sum(-1), torch.ones(3))
    drafter.draft_sampling_metadata.all_greedy = True
    assert drafter.get_draft_probs(spec, ["b", "a"]) is None
    print(
        "PASS: CPU doubles: per-step forward/sampler, hidden cache, "
        "q reorder, parameter snapshot and mixed-batch routing"
    )


def check_hidden_gather(module, device="cpu", use_kernel=False):
    """Match compact request mapping, truncation, zero counts and strided inputs."""
    checks = 0
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for hidden_size in (4, 127, 512, 1025):
            src = torch.arange(3 * 4 * hidden_size * 2, device=device).to(dtype)
            src = src.reshape(3, 4, hidden_size * 2)[:, :, ::2]
            original = src.clone()
            for counts, rows in (
                ([4, 4, 4], None),
                ([2, 0, 3], [2, 0, 1]),
                ([0, 1, 0], [0, 2, 0]),
                ([0, 0, 0], None),
            ):
                cu = torch.tensor(counts, dtype=torch.int32, device=device).cumsum(
                    0, dtype=torch.int32
                )
                mapping = (
                    None
                    if rows is None
                    else torch.tensor(rows, dtype=torch.int64, device=device)
                )
                actual = module.gather_draft_hidden_states(
                    src, cu, sum(counts), 3, mapping
                )
                expected = torch.cat(
                    [
                        src[i if rows is None else rows[i], :count]
                        for i, count in enumerate(counts)
                    ]
                )
                assert torch.equal(actual, expected)
                if use_kernel and sum(counts):
                    actual.fill_(float("nan"))
                    module.gather_hidden_kernel[
                        (3, 4, module.triton.cdiv(hidden_size, 512))
                    ](
                        src,
                        cu,
                        mapping,
                        actual,
                        *src.stride(),
                        HIDDEN_SIZE=hidden_size,
                        HAS_MAPPING=mapping is not None,
                        BLOCK_SIZE=512,
                    )
                    assert torch.equal(actual, expected)
                assert torch.equal(src, original)
                checks += 1
    print(f"PASS: {device} {checks} hidden gather cases; kernel={use_kernel}")


def check_cuda(module):
    """Match the CPU window and refresh fixed outputs on graph replay."""
    if not torch.cuda.is_available():
        raise RuntimeError("--cuda requires an available GPU")
    check_hidden_gather(module, "cuda")
    hidden = torch.arange(3 * 4 * 512, device="cuda").reshape(3, 4, 512).float()
    cu = torch.tensor([2, 2, 5], dtype=torch.int32, device="cuda")
    mapping = torch.tensor([2, 0, 1], dtype=torch.int64, device="cuda")
    module.gather_draft_hidden_states(hidden, cu, 5, 3, mapping)
    gather_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gather_graph):
        gathered = module.gather_draft_hidden_states(hidden, cu, 5, 3, mapping)
    pointer = gathered.data_ptr()
    cu.copy_(torch.tensor([0, 4, 5], dtype=torch.int32, device="cuda"))
    mapping.copy_(torch.tensor([1, 2, 0], dtype=torch.int64, device="cuda"))
    hidden.add_(1)
    gather_graph.replay()
    assert gathered.data_ptr() == pointer
    assert torch.equal(gathered, torch.cat([hidden[2, :4], hidden[0, :1]]))
    print("PASS: CUDA gather replay with changing mapping, counts and source")
    lengths_cpu = torch.tensor(
        [1, 15, 16, 17, 255, 4096, 32769, 40960], dtype=torch.int32
    )
    # Noncontiguous row stride, as in sliced runner block tables.
    table_base = torch.arange(8 * 2564, dtype=torch.int32).reshape(8, 2564)
    table_cpu = table_base[:, :2560]
    table = table_base.to("cuda")[:, :2560]
    lengths = lengths_cpu.to("cuda")
    original = table.clone()
    out = (torch.full_like(table, -99), torch.empty_like(lengths))
    pointers = tuple(t.data_ptr() for t in out)

    def compare(batch, block_size, ratio, minimum):
        expected, expected_lens = module.streamingllm_block_table(
            table_cpu[:batch],
            lengths.cpu()[:batch],
            block_size,
            ratio,
            minimum,
            out=(
                torch.empty_like(out[0][:batch], device="cpu"),
                torch.empty_like(out[1][:batch], device="cpu"),
            ),
        )
        actual, actual_lens = (t[:batch].cpu() for t in out)
        assert torch.equal(actual_lens, expected_lens)
        columns = torch.arange(actual.shape[1])[None, :]
        valid = columns < ((expected_lens - 1) // block_size + 1)[:, None]
        assert torch.equal(actual[valid], expected[valid])
        assert pointers == tuple(t.data_ptr() for t in out)
        assert torch.equal(table, original), "Target table was mutated"

    checks = 0
    for batch in (1, 3, 8):
        for block_size in (16, 64):
            for ratio, minimum in ((0.05, 32), (0.07, 256), (0.9, 65536)):
                capacity = (
                    max(ceil(40960 * ratio), minimum) + block_size - 1
                ) // block_size
                out = (
                    torch.full((8, capacity), -99, dtype=torch.int32, device="cuda"),
                    torch.empty_like(lengths),
                )
                pointers = tuple(t.data_ptr() for t in out)
                returned = module.streamingllm_block_table(
                    table[:batch],
                    lengths[:batch],
                    block_size,
                    ratio,
                    minimum,
                    out=tuple(t[:batch] for t in out),
                )
                assert tuple(t.data_ptr() for t in returned) == pointers
                compare(batch, block_size, ratio, minimum)
                checks += batch
    torch.cuda.synchronize()
    out = (
        torch.empty((8, 128), dtype=torch.int32, device="cuda"),
        torch.empty_like(lengths),
    )
    pointers = tuple(t.data_ptr() for t in out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        module.streamingllm_block_table(table, lengths, 16, 0.05, 32, out=out)
    for offset in (0, 17, 2048):
        lengths.copy_((lengths_cpu + offset).clamp_max(40960))
        graph.replay()
        compare(8, 16, 0.05, 32)
    print(f"PASS: CUDA {checks} windows; fixed buffers and changing-window graphs")


def main():
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--runner", action="store_true")
    args = parser.parse_args()
    check_vegas_wrapper()
    source = (
        Path(__file__).resolve().parents[1]
        / "vllm/v1/spec_decode/sparse_attn/streamingllm.py"
    )
    spec = importlib.util.spec_from_file_location("streamingllm", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checks = 0
    for block_size in (16, 64):
        lengths = torch.arange(1, 1025, dtype=torch.int32)
        table = torch.arange(1024 * 64, dtype=torch.int32).reshape(1024, 64)
        original = table.clone()
        for ratio, minimum in ((0.05, 32), (0.25, 128), (0.9, 2048)):
            sparse, sparse_lengths = module.streamingllm_block_table(
                table, lengths, block_size, ratio, minimum
            )
            for row, length in enumerate(lengths.tolist()):
                last = (length - 1) // block_size
                budget = max(length * ratio, minimum)
                blocks = int(torch.ceil(torch.tensor(budget / block_size)).item())
                blocks = min(blocks, table.shape[1] - 1)
                start = max(last - blocks + 1, 1)
                expected = [0, *range(start, last + 1)]
                assert sparse[row, : len(expected)].tolist() == (
                    table[row, expected].tolist()
                )
                assert sparse_lengths[row].item() == length - (start - 1) * block_size
                assert 1 <= sparse_lengths[row] <= length
                checks += 1
            assert torch.equal(table, original), "Target block table was mutated"
            assert torch.equal(lengths, torch.arange(1, 1025, dtype=torch.int32))
    print(f"PASS: {checks} sink/window cases; target table and lengths unchanged")
    # The author's global capacity includes sink, unlike the per-row tail budget.
    for block_size in (16, 64):
        for max_len in (1024, 8192, 40960):
            lengths = torch.arange(1, max_len + 1, dtype=torch.int32)
            full = torch.arange(
                (max_len + block_size - 1) // block_size, dtype=torch.int32
            )[None, :].expand(max_len, -1)
            max_tokens = max(ceil(max_len * 0.07), 256)
            capacity = (max_tokens + block_size - 1) // block_size
            out = (
                torch.empty((max_len, capacity), dtype=torch.int32),
                torch.empty_like(lengths),
            )
            pointers = tuple(t.data_ptr() for t in out)
            result, sparse_lens = module.streamingllm_block_table(
                full, lengths, block_size, 0.07, 256, out=out
            )
            budget = torch.ceil((lengths * 0.07).clamp_min(256) / block_size)
            budget.clamp_max_(capacity - 1)
            last = (lengths - 1) // block_size
            start = (last - budget.to(torch.int32) + 1).clamp_min(1)
            assert torch.equal(sparse_lens, lengths - (start - 1) * block_size)
            columns = torch.arange(capacity)[None, :]
            expected = start[:, None] + columns - 1
            expected[:, 0] = 0
            valid = columns < ((sparse_lens - 1) // block_size + 1)[:, None]
            assert torch.equal(result[valid], expected[valid])
            assert pointers == tuple(t.data_ptr() for t in out)
            assert result.shape[1] == capacity
            if max_len == 8192 and block_size == 16:
                assert sparse_lens[-1].item() == 576
    print("PASS: 100352 author-capacity cases; compact fixed outputs; 8192 -> 576")
    # Growing/shrinking requests must refresh the same graph input addresses.
    ids = torch.empty(4, dtype=torch.int32)
    positions = torch.empty(4, dtype=torch.int64)
    table_buffer = torch.empty((4, 64), dtype=torch.int32)
    pointers = (ids.data_ptr(), positions.data_ptr(), table_buffer.data_ptr())
    for batch in (4, 1, 3, 2, 4):
        source_ids = torch.arange(batch, dtype=torch.int32) + 10
        source_lens = torch.arange(batch, dtype=torch.int32) + 60
        rejected = torch.arange(batch, dtype=torch.int32)
        source_table = torch.arange(batch * 64, dtype=torch.int32).reshape(batch, 64)
        source_copy = source_table.clone()
        module.prepare_draft_buffers(
            ids,
            positions,
            table_buffer,
            source_ids,
            source_lens,
            source_table,
            1024,
            rejected,
        )
        assert pointers == (
            ids.data_ptr(),
            positions.data_ptr(),
            table_buffer.data_ptr(),
        )
        assert torch.equal(ids[:batch], source_ids)
        assert positions[:batch].tolist() == [60] * batch
        assert torch.equal(table_buffer[:batch], source_table)
        assert positions[batch:].tolist() == [1024] * (4 - batch)
        assert not torch.any(ids[batch:])
        assert not torch.any(table_buffer[batch:])
        assert torch.equal(source_table, source_copy)
        assert torch.equal(source_lens, torch.arange(batch, dtype=torch.int32) + 60)
    module.prepare_draft_buffers(
        ids, positions, table_buffer, source_ids, source_lens, source_table, 1024, None
    )
    assert torch.equal(positions.to(torch.int32), source_lens)
    print("PASS: fixed-address buffers, rejected suffix and batch padding refresh")
    positions = torch.tensor([1023, 1024, 1025, 64], dtype=torch.int64)
    full_table = torch.arange(4 * 64, dtype=torch.int32).reshape(4, 64)
    for step in range(3):
        logical_positions = positions + step
        original_positions = logical_positions.clone()
        safe, slots = module.draft_slot_mapping(logical_positions, full_table, 16, 1024)
        for row, position in enumerate(logical_positions.tolist()):
            if position >= 1024:
                assert safe[row].item() == 0 and slots[row].item() == -1
            else:
                assert safe[row].item() == position
                assert (
                    slots[row].item()
                    == full_table[row, position // 16] * 16 + position % 16
                )
        assert torch.equal(logical_positions, original_positions)
    print(
        "PASS: original KV write slots and no padded/boundary writes across draft steps"
    )
    check_hidden_gather(module)
    if args.cuda:
        check_cuda(module)
    if args.runner:
        check_runner_cache()


if __name__ == "__main__":
    main()

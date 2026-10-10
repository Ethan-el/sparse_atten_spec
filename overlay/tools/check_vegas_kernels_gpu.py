# SPDX-License-Identifier: Apache-2.0
# Added or modified for sparse_atten_spec validation; see NOTICE.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check Vegas selection kernels on CUDA without FlashAttention 3.

Run from the vLLM checkout with its Python environment:
    .venv/bin/python tools/check_vegas_kernels_gpu.py
"""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm.v1.spec_decode.sparse_attn.varlen_reduce import (
    varlen_reduce,
)
from vllm.v1.spec_decode.sparse_attn.varlen_reduce import (
    warmup as warmup_reduce,
)
from vllm.v1.spec_decode.sparse_attn.varlen_topk import (
    calc_topk_workspace_size,
    varlen_topk,
)
from vllm.v1.spec_decode.sparse_attn.varlen_topk import (
    warmup as warmup_topk,
)
from vllm.v1.spec_decode.sparse_attn.vegas import VegasState


def _ints(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32, device="cuda")


def _cumulative(lengths: list[int]) -> torch.Tensor:
    result = [0]
    for length in lengths:
        result.append(result[-1] + length)
    return _ints(result)


def _reduce_reference(
    scores: torch.Tensor,
    lengths: torch.Tensor,
    entries: torch.Tensor,
    lse: torch.Tensor,
    cumulative: torch.Tensor,
    scale: float,
    use_weight: bool,
) -> torch.Tensor:
    scores = scores.cpu().float()
    lengths = lengths.cpu().tolist()
    entries = entries.cpu().tolist()
    lse = lse.cpu()
    cumulative = cumulative.cpu().tolist()
    result = torch.full(
        (scores.shape[0], scores.shape[-1]), -float("inf"), dtype=torch.bfloat16
    )
    for row, (length, entry) in enumerate(zip(lengths, entries)):
        length = max(0, min(length, scores.shape[-1]))
        if length == 0:
            continue
        slots = (0, 1) if entry == 0 else (entry - 1,)
        first, last = cumulative[row], cumulative[row + 1] - 1
        if use_weight and (first < 0 or first > last or last >= lse.shape[1]):
            continue
        values = scores[row][:, list(slots), :length]
        if use_weight:
            tokens = [first if slot == 0 else last for slot in slots]
            values = torch.exp(scale * values - lse[:, tokens, None])
        result[row, :length] = values.mean(dim=(0, 1)).to(torch.bfloat16)
    return result


def _check_reduce(
    scores: torch.Tensor,
    lengths: torch.Tensor,
    entries: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    cumulative: torch.Tensor,
    scale: float,
    use_weight: bool,
) -> None:
    reference = _reduce_reference(
        scores, lengths, entries, lse, cumulative, scale, use_weight
    )
    actual = output.cpu()
    assert torch.equal(torch.isneginf(actual), torch.isneginf(reference))
    torch.testing.assert_close(actual, reference, rtol=0.016, atol=0.001)


def check_reduce() -> None:
    """Check scalar/vector dispatch, masking, ranking modes and graph replay."""
    scale = 0.37
    for batch_size in (6, 256):
        qlens = [([0, 1, 3, 5, 1, 0])[i % 6] for i in range(batch_size)]
        cumulative = _cumulative(qlens)
        lengths = _ints([([0, 7, 13, 24, 1, 7])[i % 6] for i in range(batch_size)])
        scores = torch.randn(batch_size, 3, 2, 24, device="cuda", dtype=torch.bfloat16)
        lse = torch.randn(3, sum(qlens), device="cuda", dtype=torch.float32)
        output = torch.empty(batch_size, 24, device="cuda", dtype=torch.bfloat16)
        for entry in (0, 1, 2):
            entries = _ints([entry] * batch_size)
            for use_weight in (False, True):
                varlen_reduce(
                    scores,
                    lengths,
                    entries,
                    output,
                    lse,
                    cumulative,
                    scale,
                    use_weight,
                )
                _check_reduce(
                    scores,
                    lengths,
                    entries,
                    output,
                    lse,
                    cumulative,
                    scale,
                    use_weight,
                )

    for batch_size, max_len in ((0, 24), (3, 0)):
        scores = torch.empty(
            batch_size, 3, 2, max_len, device="cuda", dtype=torch.bfloat16
        )
        lengths = _ints([0] * batch_size)
        output = torch.empty(batch_size, max_len, device="cuda", dtype=torch.bfloat16)
        varlen_reduce(
            scores,
            lengths,
            lengths,
            output,
            torch.empty(3, 0, device="cuda"),
            _ints([0] * (batch_size + 1)),
            scale,
            False,
        )
        assert output.numel() == 0

    for use_weight in (False, True):
        scores = torch.randn(6, 3, 2, 24, device="cuda", dtype=torch.bfloat16)
        lengths = _ints([0, 7, 13, 24, 1, 7])
        entries = _ints([0, 1, 2, 0, 1, 2])
        cumulative = _cumulative([0, 1, 3, 5, 1, 0])
        lse = torch.randn(3, 10, device="cuda")
        output = torch.empty(6, 24, device="cuda", dtype=torch.bfloat16)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            varlen_reduce(
                scores,
                lengths,
                entries,
                output,
                lse,
                cumulative,
                scale,
                use_weight,
            )
        for _ in range(3):
            scores.copy_(torch.randn_like(scores))
            lse.copy_(torch.randn_like(lse))
            lengths.copy_(_ints([7, 0, 13, 1, 24, 0]))
            entries.copy_(_ints([2, 0, 1, 2, 0, 1]))
            cumulative.copy_(_cumulative([1, 0, 5, 1, 0, 3]))
            graph.replay()
            _check_reduce(
                scores,
                lengths,
                entries,
                output,
                lse,
                cumulative,
                scale,
                use_weight,
            )


def _metric(batch_size: int, max_len: int, dtype: torch.dtype) -> torch.Tensor:
    """Make exact unique float values; bf16 deliberately introduces ties."""
    rows = [torch.randperm(max_len, device="cuda") for _ in range(batch_size)]
    return torch.stack(rows).to(dtype=dtype)


def _check_topk(
    metric: torch.Tensor,
    lengths: torch.Tensor,
    ks: torch.Tensor,
    output: torch.Tensor,
    select_min: bool,
) -> None:
    metric = metric.cpu().float()
    output = output.cpu()
    for row, (length, k) in enumerate(zip(lengths.cpu().tolist(), ks.cpu().tolist())):
        length = max(0, min(length, metric.shape[1]))
        k = max(0, min(k, length, output.shape[1]))
        indices = output[row, :k].long()
        assert torch.all(output[row, k:] == -1), (row, "stale padding")
        if k == 0:
            continue
        assert torch.all((indices >= 0) & (indices < length))
        assert indices.unique().numel() == k, (row, "duplicate indices")
        selected = metric[row, indices].sort().values
        expected = torch.topk(metric[row, :length], k, largest=not select_min).values
        # Comparing values accepts any legal subset of ties at the threshold.
        assert torch.equal(selected, expected.sort().values), (row, length, k)


def check_topk() -> None:
    """Check both paths, clamped lengths/k, padding and mutable graph inputs."""
    for dtype in (torch.float32, torch.bfloat16):
        for max_len in (4096, 65536):
            metric = _metric(7, max_len, dtype)
            lengths = _ints([0, 7, 13, max_len, max_len + 17, -1, 4096])
            ks = _ints([7, 0, 9999, 64, max_len, 3, 17])
            output = torch.empty(7, 64, dtype=torch.int32, device="cuda")
            workspace = torch.empty(
                calc_topk_workspace_size(7, max_len, 64),
                dtype=torch.uint8,
                device="cuda",
            )
            for select_min in (False, True):
                for path in (0, 1, -1):
                    varlen_topk(
                        metric,
                        ks,
                        lengths,
                        output,
                        workspace,
                        select_min=select_min,
                        force_path=path,
                    )
                    _check_topk(metric, lengths, ks, output, select_min)

        metric = _metric(7, 4096, dtype)
        lengths = _ints([0, 7, 13, 4096, 4096, 0, 100])
        ks = _ints([0, 5, 64, 9999, 17, 0, 100])
        output = torch.empty(7, 64, dtype=torch.int32, device="cuda")
        workspace = torch.empty(
            calc_topk_workspace_size(7, 4096, 64), dtype=torch.uint8, device="cuda"
        )
        for path in (0, 1):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                varlen_topk(metric, ks, lengths, output, workspace, force_path=path)
            for iteration in range(3):
                metric.copy_(_metric(7, 4096, dtype))
                lengths.copy_(_ints([4096, 0, 7, 13, 100, 4096, 0]))
                ks.copy_(_ints([iteration, 64, 9999, 5, 64, 17, 0]))
                graph.replay()
                _check_topk(metric, lengths, ks, output, False)

    for batch_size, max_len, max_k in ((0, 4096, 64), (3, 0, 64), (3, 13, 0)):
        metric = torch.empty(batch_size, max_len, device="cuda")
        output = torch.full((batch_size, max_k), 123, dtype=torch.int32, device="cuda")
        zeros = _ints([0] * batch_size)
        varlen_topk(metric, zeros, zeros, output)
        assert torch.all(output == -1)


def _check_state_selection(state, metadata, source_scores, lse, scale) -> None:
    seq_lens = metadata.seq_lens.cpu()
    qlens = metadata.query_start_loc.cpu().diff()
    prefill = state.prefilling.cpu()
    active = (qlens > 0) & (seq_lens > 0)
    valid = torch.where(prefill, seq_lens, seq_lens + 1 - qlens)
    valid = torch.where(active, valid, 0).clamp(0, state.max_model_len)
    entries = torch.where(prefill & (qlens > 1), 2, torch.where(qlens == 1, 1, 0))
    budget = torch.ceil(seq_lens.float() * state.ratio).to(torch.int32)
    budget = budget.clamp(state.min_tokens, state.max_budget).minimum(valid)
    assert torch.equal(state.valid_lens.cpu(), valid)
    assert torch.equal(state.entries.cpu(), entries)
    assert torch.equal(state.budgets.cpu(), budget)
    for name, scores in source_scores.items():
        reference = _reduce_reference(
            scores, valid, entries, lse, metadata.query_start_loc, scale, True
        )
        _check_topk(
            reference, valid, budget, state.indices[state.layer_rows[name]], False
        )
    assert (
        state.indices[state.layer_rows["a"]].data_ptr()
        != state.indices[state.layer_rows["b"]].data_ptr()
    ), "layers reused the same selection storage"


def _check_state_slots(state, table, positions) -> None:
    table = table.cpu()
    positions = positions.cpu().tolist()
    history = state.valid_lens.cpu().tolist()
    budgets = state.budgets.cpu().tolist()
    for name, layer in state.layer_rows.items():
        selected = state.indices[layer].cpu()
        actual = state.slots[layer].cpu()
        lengths = state.lengths[layer].cpu()
        for row, position in enumerate(positions):
            logical = []
            if history[row] > 0 and history[row] - 1 <= position < state.max_model_len:
                valid = min(history[row], position + 1)
                count = min(budgets[row], valid)
                logical = selected[row, :count].tolist() + list(
                    range(valid, position + 1)
                )
                logical = logical[: state.slot_width]
                assert len(logical) == len(set(logical)), (name, row, "duplicate KV")
                assert all(0 <= index <= position for index in logical), (
                    name,
                    row,
                    "rejected future KV escaped into the draft",
                )
            expected = torch.full((state.slot_width,), -1, dtype=torch.int32)
            for col, index in enumerate(logical):
                expected[col] = table[row, index // state.block_size] * state.block_size
                expected[col] += index % state.block_size
            assert int(lengths[row]) == len(logical), (name, row, "wrong sparse length")
            assert torch.equal(actual[row], expected), (name, row, position)


def _full_table() -> torch.Tensor:
    return torch.stack(
        [
            torch.randperm(8, device="cuda", dtype=torch.int32) + 32 + row * 8
            for row in range(4)
        ]
    )


def check_state() -> None:
    """Check real state, selection kernels and physical slots without FA3."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm.v1.spec_decode.sparse_attn.proposer import SparseAttnProposer

    config = SimpleNamespace(
        model_config=SimpleNamespace(
            max_model_len=128,
            dtype=torch.bfloat16,
            get_num_attention_heads=lambda parallel: 2,
        ),
        parallel_config=SimpleNamespace(),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        speculative_config=SimpleNamespace(
            sparse_attn_algorithm="vegas",
            num_speculative_tokens=4,
            sparse_attn_ratio=0.07,
            sparse_attn_min_tokens=16,
        ),
    )
    drafter = SparseAttnProposer.__new__(SparseAttnProposer)
    drafter.vllm_config = config
    drafter.speculative_config = config.speculative_config
    drafter.input_ids = _ints([0] * 4)
    with patch(
        "vllm.v1.spec_decode.sparse_attn.proposer.get_layers_from_vllm_config",
        return_value={"a": object(), "b": object()},
    ):
        drafter.load_model(object())
    state = drafter.vegas
    assert isinstance(state, VegasState) and state.block_size == 0
    pointers = {
        name: tensor.data_ptr()
        for name, tensor in vars(state).items()
        if isinstance(tensor, torch.Tensor)
    }
    cache = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=["a", "b"],
                kv_cache_spec=FullAttentionSpec(
                    block_size=16, num_kv_heads=1, head_size=64, dtype=torch.bfloat16
                ),
            )
        ]
    )
    drafter.runner = SimpleNamespace(
        _kernel_block_sizes=[16],
        attn_groups=[
            [SimpleNamespace(backend=SimpleNamespace(get_name=lambda: "FLASH_ATTN"))]
        ],
    )
    # Profiling and real KV initialization must reuse the pre-profile allocation.
    for table in (torch.zeros_like(_full_table()), _full_table()):
        drafter.runner.input_batch = SimpleNamespace(
            block_table=SimpleNamespace(
                block_tables=[SimpleNamespace(block_table=SimpleNamespace(gpu=table))]
            )
        )
        drafter.validate_same_kv_cache_group(cache)
        assert drafter.vegas is state and state.block_size == 16
        assert all(
            getattr(state, name).data_ptr() == ptr for name, ptr in pointers.items()
        )
    metadata = SimpleNamespace(
        seq_lens=_ints([7, 80, 105, 0]),
        query_start_loc=_cumulative([7, 1, 5, 0]),
    )
    logical = torch.arange(128, device="cuda", dtype=torch.float32)
    # Different rankings make wrong query/head/LSE wiring observable in top-k.
    ranks = torch.stack(
        [logical, logical.flip(0), logical.roll(43), logical.roll(71)]
    ).view(1, 2, 2, 128)
    source_scores = {
        name: (values / 8).expand(4, -1, -1, -1).to(torch.bfloat16).contiguous()
        for name, values in (("a", ranks), ("b", ranks.flip(-1)))
    }
    tokens = torch.arange(13, device="cuda", dtype=torch.float32)
    lse = torch.stack([tokens / 4, tokens.flip(0) / 3])
    scale = 0.25
    table = _full_table()

    def select_all() -> None:
        for name, scores in source_scores.items():
            scratch = state.scores_for(name, 4)
            scratch.copy_(scores)
            state.select(name, scratch, lse, metadata, scale)

    def slots_all(positions) -> None:
        for name in source_scores:
            state.draft_slots(name, table, positions)

    state.prepare_target(torch.tensor([True, False, False, False]), 4)
    select_all()
    _check_state_selection(state, metadata, source_scores, lse, scale)
    assert state.valid_lens.tolist() == [7, 80, 101, 0]
    assert state.budgets.tolist() == [7, 16, 16, 0]
    assert state.entries.tolist() == [2, 1, 0, 0]
    # Corrected positions for first rejection, partial acceptance and all accepted.
    for accepted in (0, 2, 4):
        for draft_step in range(4):
            positions = _ints(
                [
                    7 + accepted + draft_step,
                    80 + accepted + draft_step,
                    101 + accepted + draft_step,
                    0,
                ]
            )
            slots_all(positions)
            _check_state_slots(state, table, positions)
    for values in ([128, -1, 100, 0], [5, 78, 99, 0]):
        positions = _ints(values)
        slots_all(positions)
        _check_state_slots(state, table, positions)

    # New rows represent reordered/new/finished requests; old selections must go.
    metadata.seq_lens.copy_(_ints([57, 6, 0, 96]))
    metadata.query_start_loc.copy_(_cumulative([5, 6, 0, 2]))
    state.prepare_target(torch.tensor([False, True, False, False]), 4)
    select_all()
    _check_state_selection(state, metadata, source_scores, lse, scale)
    assert state.valid_lens.tolist() == [53, 6, 0, 95]
    assert state.budgets.tolist() == [16, 6, 0, 16]
    assert torch.all(state.indices[:, 2] == -1)
    assert torch.all(state.indices[:, 1, 6:] == -1)
    positions = _ints([55, 7, 0, 99])
    slots_all(positions)
    _check_state_slots(state, table, positions)

    ratio_config = SimpleNamespace(**vars(config))
    ratio_config.speculative_config = SimpleNamespace(
        **{**vars(config.speculative_config), "sparse_attn_min_tokens": 2}
    )
    ratio_state = VegasState(ratio_config, "cuda", ["a", "b"], 16)
    assert ratio_state.max_budget == 9  # ceil(128 * 0.07), not the minimum.
    for seq_lens, qlens, phases, valid, budgets, entries in (
        (
            [1, 29, 53, 128],
            [1, 2, 3, 5],
            [True, True, False, True],
            [1, 29, 51, 128],
            [1, 3, 4, 9],
            [1, 2, 0, 2],
        ),
        (
            [14, 57, 127, 0],
            [3, 4, 1, 0],
            [True, True, False, False],
            [14, 57, 127, 0],
            [2, 4, 9, 0],
            [2, 2, 1, 0],
        ),
    ):
        boundary_metadata = SimpleNamespace(
            seq_lens=_ints(seq_lens), query_start_loc=_cumulative(qlens)
        )
        ratio_state.prepare_target(torch.tensor(phases), 4)
        for name, scores in source_scores.items():
            ratio_state.select(name, scores, lse, boundary_metadata, scale)
        _check_state_selection(
            ratio_state, boundary_metadata, source_scores, lse, scale
        )
        assert ratio_state.valid_lens.tolist() == valid
        assert ratio_state.budgets.tolist() == budgets
        assert ratio_state.entries.tolist() == entries

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        select_all()
        slots_all(positions)
    for iteration in range(3):
        if iteration == 2:
            seq_lens, qlens = [13, 31, 60, 0], [2, 4, 1, 0]
            phases, current_positions = [True, True, False, False], [14, 32, 61, 0]
        elif iteration % 2:
            seq_lens, qlens = [57, 6, 0, 96], [5, 6, 0, 2]
            phases, current_positions = [False, True, False, False], [55, 7, 0, 99]
        else:
            seq_lens, qlens = [7, 80, 105, 0], [7, 1, 5, 0]
            phases, current_positions = [True, False, False, False], [9, 82, 103, 0]
        metadata.seq_lens.copy_(_ints(seq_lens))
        metadata.query_start_loc.copy_(_cumulative(qlens))
        state.prepare_target(torch.tensor(phases), 4)
        positions.copy_(_ints(current_positions))
        table.copy_(_full_table())
        for scores in source_scores.values():
            scores.copy_(scores.flip(-1))
            scores.add_(0.125)
        lse.add_(0.0625)
        graph.replay()
        _check_state_selection(state, metadata, source_scores, lse, scale)
        _check_state_slots(state, table, positions)
    check_proposer_forward(state)


def check_proposer_forward(state) -> None:
    """Exercise real Vegas draft metadata with a GPU model probe."""
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context, is_forward_context_available
    from vllm.v1.spec_decode.sparse_attn.proposer import SparseAttnProposer

    drafter = SparseAttnProposer.__new__(SparseAttnProposer)
    drafter.vegas = state
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
    drafter.block_size = state.block_size
    drafter.max_model_len = state.max_model_len
    drafter.input_ids = _ints([3, 4, 5, 0])
    drafter.positions = torch.empty(4, dtype=torch.int64, device="cuda")
    drafter.full_table = _full_table()
    drafter.query_start_loc = torch.arange(5, dtype=torch.int32, device="cuda")
    drafter.attn_layer_names = ["a", "b"]
    drafter.runner = SimpleNamespace(
        attn_groups=[[SimpleNamespace(layer_names=["a", "b"])]]
    )
    weights = torch.arange(1, state.slot_width + 1, device="cuda").float()

    def model(input_ids, positions):
        context = get_forward_context()
        assert not context.skip_compiled
        assert context.cudagraph_runtime_mode == CUDAGraphMode.NONE
        a, b = (context.attn_metadata[name] for name in ("a", "b"))
        assert a is not b
        assert a.sparse_token_slots.data_ptr() != b.sparse_token_slots.data_ptr()
        assert a.seq_lens.data_ptr() != b.seq_lens.data_ptr()
        for name, metadata in (("a", a), ("b", b)):
            layer = state.layer_rows[name]
            assert (
                metadata.sparse_token_slots.data_ptr() == state.slots[layer].data_ptr()
            )
            assert metadata.seq_lens.data_ptr() == state.lengths[layer].data_ptr()
            assert metadata.block_table.data_ptr() == drafter.full_table.data_ptr()
            assert metadata.max_seq_len == state.slot_width
            assert metadata.max_num_splits == 1
            assert metadata.num_actual_tokens == 4
        assert a.slot_mapping.data_ptr() == b.slot_mapping.data_ptr()
        # These GPU operations consume the same metadata during graph replay.
        return torch.stack(
            (
                (a.sparse_token_slots.float() * weights).sum(dim=1),
                (b.sparse_token_slots.float() * weights).sum(dim=1),
                a.seq_lens.float(),
                b.seq_lens.float(),
                a.slot_mapping.float(),
                positions.float(),
                input_ids.float(),
                a.block_table[:, 0].float(),
            ),
            dim=1,
        )

    drafter.model = model

    def update(iteration):
        history = [20 + iteration, 40 + iteration, 60 + iteration, 0]
        budget = [4 + iteration, 6, 8 - iteration, 0]
        state.valid_lens.copy_(_ints(history))
        state.budgets.copy_(_ints(budget))
        state.indices.fill_(-1)
        for row in range(3):
            count = budget[row]
            indices = torch.arange(count, dtype=torch.int32, device="cuda")
            state.indices[0, row, :count].copy_(indices + iteration)
            state.indices[1, row, :count].copy_(history[row] - 1 - indices)
        drafter.positions.copy_(
            _ints(
                [
                    history[0] + iteration,
                    history[1] + 2,
                    history[2] + 1,
                    state.max_model_len,
                ]
            )
        )
        drafter.full_table.copy_(_full_table())
        drafter.input_ids.copy_(_ints([3 + iteration, 4, 5, 0]))

    def check(hidden):
        _check_state_slots(state, drafter.full_table, drafter.positions)
        positions = drafter.positions.cpu()
        table = drafter.full_table.cpu()
        write_slots = torch.full((4,), -1, dtype=torch.int64)
        safe_positions = positions.clone()
        for row, position in enumerate(positions.tolist()):
            if 0 <= position < state.max_model_len:
                write_slots[row] = (
                    table[row, position // state.block_size] * state.block_size
                )
                write_slots[row] += position % state.block_size
            else:
                safe_positions[row] = 0
        cpu_weights = weights.cpu()
        expected = torch.stack(
            (
                (state.slots[0].cpu().float() * cpu_weights).sum(dim=1),
                (state.slots[1].cpu().float() * cpu_weights).sum(dim=1),
                state.lengths[0].cpu().float(),
                state.lengths[1].cpu().float(),
                write_slots.float(),
                safe_positions.float(),
                drafter.input_ids.cpu().float(),
                table[:, 0].float(),
            ),
            dim=1,
        )
        torch.testing.assert_close(hidden.cpu(), expected, rtol=0, atol=0)
        assert int(write_slots[-1]) == -1
        assert torch.all(state.slots[:, -1] == -1)
        assert not is_forward_context_available(), "Draft context leaked"

    update(0)
    check(drafter._forward(4))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        hidden = drafter._forward(4)
    for iteration in (1, 2, 3):
        update(iteration)
        graph.replay()
        check(hidden)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Vegas kernel checks require a CUDA GPU")
    torch.manual_seed(7)
    print(f"Compiling Vegas kernels for {torch.cuda.get_device_name()}", flush=True)
    warmup_reduce()
    warmup_topk()
    check_reduce()
    print("PASS Vegas reduce: logits/weights, scalar/vector, masking, CUDA graphs")
    check_topk()
    print("PASS Vegas top-k: both paths, float32/bf16 ties, padding, CUDA graphs")
    check_state()
    torch.cuda.synchronize()
    print("PASS Vegas state: ratio budgets, short prefill, query/LSE ranking")
    print("PASS Vegas state: layers, request churn, rejection, slots, CUDA graphs")
    print("PASS Vegas lifecycle: pre-profile allocation, stable KV reinitialization")
    print("PASS Vegas proposer: layer metadata, KV writes, mutable graph replay")


if __name__ == "__main__":
    main()

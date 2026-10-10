# SPDX-License-Identifier: Apache-2.0
# Modified for sparse_atten_spec: StreamingLLM and Vegas port; see NOTICE.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import triton
import triton.language as tl


@triton.jit
def gather_hidden_kernel(
    src,
    cu_counts,
    mapping,
    out,
    batch_stride,
    token_stride,
    hidden_stride,
    HIDDEN_SIZE: tl.constexpr,
    HAS_MAPPING: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # platformxlab/vegas gather_hidden_kernel, adapted to existing Triton runtime.
    req = tl.program_id(0)
    token = tl.program_id(1)
    start = tl.load(cu_counts + req - 1, mask=req > 0, other=0)
    count = tl.load(cu_counts + req) - start
    if token < count:
        row = tl.load(mapping + req) if HAS_MAPPING else req
        columns = tl.program_id(2) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        values = tl.load(
            src + row * batch_stride + token * token_stride + columns * hidden_stride,
            mask=columns < HIDDEN_SIZE,
            other=0,
        )
        tl.store(
            out + (start + token) * HIDDEN_SIZE + columns,
            values,
            mask=columns < HIDDEN_SIZE,
        )


def gather_draft_hidden_states(
    src: torch.Tensor,
    cu_num_draft_tokens: torch.Tensor,
    total_tokens: int,
    batch_size: int,
    reorder_mapping: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compact valid drafts using the author's request/token mapping semantics."""
    output = src.new_empty((total_tokens, src.shape[-1]))
    if total_tokens == 0:
        return output
    if src.is_cuda:
        gather_hidden_kernel[
            (batch_size, src.shape[1], triton.cdiv(src.shape[-1], 512))
        ](
            src,
            cu_num_draft_tokens,
            reorder_mapping,
            output,
            *src.stride(),
            HIDDEN_SIZE=src.shape[-1],
            HAS_MAPPING=reorder_mapping is not None,
            BLOCK_SIZE=512,
        )
    else:
        # CPU reference only; no host reads occur on the CUDA path.
        start = 0
        for req, end in enumerate(cu_num_draft_tokens.tolist()):
            row = req if reorder_mapping is None else int(reorder_mapping[req])
            output[start:end].copy_(src[row, : end - start])
            start = end
    return output


@triton.jit
def copy_blocks_kernel(
    src_ptr,
    dst_ptr,
    seq_lens_ptr,
    sparse_lens_ptr,
    src_stride,
    dst_stride,
    block_size: tl.constexpr,
    ratio: tl.constexpr,
    min_tokens: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_SLIDING_BLOCKS: tl.constexpr,
):
    # Adapted from platformxlab/vegas: sink + valid tail, not the full table.
    row = tl.program_id(0)
    length = tl.load(seq_lens_ptr + row)
    last = (length - 1) // block_size
    budget = tl.ceil(
        tl.maximum(length.to(tl.float32) * ratio, min_tokens) / block_size
    ).to(tl.int32)
    budget = tl.minimum(budget, MAX_SLIDING_BLOCKS)
    start = tl.maximum(last - budget + 1, 1)
    count = tl.maximum(last - start + 1, 0)
    src = src_ptr + row * src_stride
    dst = dst_ptr + row * dst_stride
    tl.store(dst, tl.load(src))
    for off in range(0, count, BLOCK_SIZE):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        values = tl.load(src + start + offsets, mask=offsets < count, other=0)
        tl.store(dst + 1 + offsets, values, mask=offsets < count)
    tl.store(sparse_lens_ptr + row, length - (start - 1) * block_size)


def draft_slot_mapping(
    positions: torch.Tensor,
    full_table: torch.Tensor,
    block_size: int,
    max_model_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep original write positions; invalid/padded rows never write KV."""
    valid = positions < max_model_len
    safe_positions = torch.where(valid, positions, 0)
    block_ids = full_table.gather(1, (safe_positions // block_size)[:, None]).flatten()
    slots = block_ids * block_size + safe_positions % block_size
    return safe_positions, slots.masked_fill(~valid, -1)


def prepare_draft_buffers(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    next_token_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    full_table: torch.Tensor,
    max_model_len: int,
    num_rejected_tokens: torch.Tensor | None,
) -> None:
    """Refresh fixed-address graph inputs; padded rows must never write KV."""
    batch = next_token_ids.shape[0]
    input_ids[batch:].zero_()
    positions[batch:].fill_(max_model_len)
    block_table[batch:].zero_()
    input_ids[:batch].copy_(next_token_ids)
    positions[:batch].copy_(seq_lens)
    if num_rejected_tokens is not None:
        positions[:batch].sub_(num_rejected_tokens)
    block_table[:batch].copy_(full_table)


def streamingllm_block_table(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    ratio: float,
    min_tokens: int,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep one sink block and a block-rounded recent window, without mutation."""
    capacity = out[0].shape[1] if out is not None else block_table.shape[1]
    if block_table.is_cuda:
        sparse_table, sparse_lens = out or (
            torch.empty_like(block_table),
            torch.empty_like(seq_lens),
        )
        copy_blocks_kernel[(seq_lens.shape[0],)](
            block_table,
            sparse_table,
            seq_lens,
            sparse_lens,
            block_table.stride(0),
            sparse_table.stride(0),
            block_size,
            ratio,
            min_tokens,
            BLOCK_SIZE=128,
            MAX_SLIDING_BLOCKS=capacity - 1,
            enable_fp_fusion=False,
        )
        return sparse_table, sparse_lens
    last = (seq_lens - 1) // block_size
    budget = torch.ceil((seq_lens * ratio).clamp_min(min_tokens) / block_size).to(
        torch.int64
    )
    budget.clamp_max_(capacity - 1)
    start = (last - budget + 1).clamp_min(1)
    # CPU reference for regression checks; CUDA copies only the valid window.
    columns = torch.arange(capacity, device=block_table.device)
    indices = start[:, None] + columns[None, :] - 1
    indices[:, 0] = 0
    indices = indices.clamp_max(block_table.shape[1] - 1).long()
    sparse_table = block_table.gather(1, indices)
    sparse_lens = seq_lens - (start - 1) * block_size
    sparse_lens = sparse_lens.to(seq_lens.dtype)
    if out is not None:
        out[0].copy_(sparse_table)
        out[1].copy_(sparse_lens)
        return out
    return sparse_table, sparse_lens

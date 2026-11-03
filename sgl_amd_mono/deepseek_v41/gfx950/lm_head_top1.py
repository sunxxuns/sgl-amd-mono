"""Skinny bf16 lm_head GEMM with a per-CTA top-1 epilogue."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_BLOCK_M = 64
_BLOCK_N = 128
_BLOCK_K = 128
_partial_cache = {}
_out_cache = {}


@triton.jit
def _lm_head_partial_kernel(
    hidden,
    weight,
    part_v,
    part_i,
    M,
    N,
    K,
    stride_hm,
    stride_hk,
    stride_wn,
    stride_wk,
    index_offset,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile = tl.program_id(0)
    offs_n = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        h = tl.load(
            hidden + offs_m[:, None] * stride_hm + offs_k[None, :] * stride_hk,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        w = tl.load(
            weight + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
            mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
            other=0.0,
        )
        acc += tl.dot(h, tl.trans(w))

    vals = acc.to(tl.bfloat16).to(tl.float32)
    vals = tl.where(offs_n[None, :] < N, vals, -float("inf"))
    best_v, best_i = tl.max(
        vals, axis=1, return_indices=True, return_indices_tie_break_left=True
    )
    tl.store(part_v + tile * M + offs_m, best_v, mask=offs_m < M)
    tl.store(
        part_i + tile * M + offs_m,
        (tile * BLOCK_N + best_i + index_offset).to(tl.int32),
        mask=offs_m < M,
    )


@triton.jit
def _lm_head_reduce_kernel(
    part_v,
    part_i,
    out,
    M,
    NTILES,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < NTILES
    vals = tl.load(part_v + offs * M + row, mask=mask, other=-float("inf"))
    idx = tl.load(part_i + offs * M + row, mask=mask, other=0x7FFFFFFF)
    best_v = tl.max(vals, axis=0)
    best_i = tl.min(tl.where(vals == best_v, idx, 0x7FFFFFFF), axis=0)
    tl.store(out + row * 2, best_v)
    tl.store(out + row * 2 + 1, best_i.to(tl.float32))


def _buffers(hidden: torch.Tensor, n: int):
    rows = hidden.shape[0]
    ntiles = triton.cdiv(n, _BLOCK_N)
    key = (hidden.device, rows, ntiles)
    part = _partial_cache.get(key)
    if part is None:
        part = (
            torch.empty((ntiles, rows), dtype=torch.float32, device=hidden.device),
            torch.empty((ntiles, rows), dtype=torch.int32, device=hidden.device),
        )
        _partial_cache[key] = part
    out = _out_cache.get(key)
    if out is None:
        out = torch.empty((rows, 2), dtype=torch.float32, device=hidden.device)
        _out_cache[key] = out
    return part[0], part[1], out, ntiles


def lm_head_top1(
    hidden: torch.Tensor, weight: torch.Tensor, index_offset: int = 0
) -> torch.Tensor:
    """Return this rank's ``(max, lowest_index)`` for every row."""
    assert hidden.dim() == 2 and weight.dim() == 2
    assert hidden.dtype == weight.dtype == torch.bfloat16
    assert hidden.shape[1] == weight.shape[1]
    rows, k = hidden.shape
    n = weight.shape[0]
    assert rows <= _BLOCK_M
    part_v, part_i, out, ntiles = _buffers(hidden, n)
    _lm_head_partial_kernel[(ntiles,)](
        hidden,
        weight,
        part_v,
        part_i,
        rows,
        n,
        k,
        hidden.stride(0),
        hidden.stride(1),
        weight.stride(0),
        weight.stride(1),
        index_offset,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_K,
        num_warps=8,
        num_stages=3,
    )
    _lm_head_reduce_kernel[(rows,)](
        part_v,
        part_i,
        out,
        rows,
        ntiles,
        BLOCK=triton.next_power_of_2(ntiles),
        num_warps=4,
    )
    return out


def warmup(weight: torch.Tensor, rows=(6, 12, 24, 48)) -> None:
    if weight.dtype != torch.bfloat16 or weight.dim() != 2:
        return
    for m in rows:
        dummy = torch.zeros(
            (m, weight.shape[1]), dtype=weight.dtype, device=weight.device
        )
        lm_head_top1(dummy, weight)

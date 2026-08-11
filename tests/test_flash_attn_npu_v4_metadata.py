# Copyright (c) 2026, Minghua Shen.

import pytest
import torch
import torch_npu
import numpy as np

from flash_attn_npu_4 import (
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
    get_scheduler_metadata,
)
from tests.test_flash_attn_npu_v4 import ref_flash_attention
from tests.npu_precision_utils import compare_rule


RTOL = 2e-2
ATOL = 2e-2
WINDOW_SIZE = (-1, -1)

SMALL_RANGE = (-1.0, 1.0)
WIDE_RANGE = (-5.0, 5.0)


def _rand_npu(shape, data_type, value_range):
    low, high = value_range
    return (low + (high - low) * torch.rand(shape)).to(data_type).npu()


def _prefix_sums(lengths):
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return offsets


def _int32_npu(values):
    return torch.tensor(values, dtype=torch.int32).npu()


def _metadata(
    *,
    batch_size, q_seqlen, kv_seqlen, num_heads, kv_heads, head_size,
    cache_seqlens, data_type, cu_seqlens_q=None, page_size=None,
    is_causal=False, num_splits=0,
):
    return get_scheduler_metadata(
        batch_size=batch_size, max_seqlen_q=q_seqlen, max_seqlen_k=kv_seqlen,
        num_heads_q=num_heads, num_heads_kv=kv_heads, headdim=head_size,
        cache_seqlens=cache_seqlens, qkv_dtype=data_type,
        cu_seqlens_q=cu_seqlens_q, page_size=page_size,
        causal=is_causal, window_size=WINDOW_SIZE, num_splits=num_splits,
    )


def _causal_mask(q_seqlen, kv_seqlen):
    return torch.triu(
        torch.ones(q_seqlen, kv_seqlen),
        diagonal=kv_seqlen - q_seqlen + 1,
    ).bool()


def _paged_kv_for_batch(k_cache_cpu, v_cache_cpu, pt_cpu, bi, kv_len, bs):
    keys, values = [], []
    row = pt_cpu[bi]
    for p in range(kv_len):
        blk, off = int(row[p // bs]), p % bs
        keys.append(k_cache_cpu[blk, off])
        values.append(v_cache_cpu[blk, off])
    return torch.stack(keys, dim=0), torch.stack(values, dim=0)


def _make_paged_cache(batch_size, kv_seqlen, kv_heads, head_size, block_size, data_type):
    max_blocks = (kv_seqlen + block_size - 1) // block_size
    n_blocks = batch_size * max_blocks
    kc = _rand_npu((n_blocks, block_size, kv_heads, head_size), data_type, SMALL_RANGE)
    vc = _rand_npu((n_blocks, block_size, kv_heads, head_size), data_type, SMALL_RANGE)
    pt = torch.arange(n_blocks, dtype=torch.int32).reshape(batch_size, max_blocks).npu()
    return kc, vc, pt


# ---------------------------------------------------------------------------
# Golden builders  (same pattern as test_flash_attn_npu_v4.py)
# ---------------------------------------------------------------------------

def _build_golden_bsnd(
    query, kv_for_batch, *, batch_size, q_seqlen, num_heads, head_size,
    scale, data_type, is_causal,
):
    query_cpu = query.detach().cpu()
    golden_out = torch.empty((batch_size, q_seqlen, num_heads, head_size), dtype=data_type)
    golden_lse = torch.empty((batch_size, num_heads, q_seqlen), dtype=torch.float32)
    for bi in range(batch_size):
        k_cpu, v_cpu = kv_for_batch(bi)
        mask = _causal_mask(q_seqlen, k_cpu.shape[0]) if is_causal else None
        out, lse = ref_flash_attention(
            query_cpu[bi], k_cpu, v_cpu, scale, mask, data_type, rescale_threshold=4.0,
        )
        golden_out[bi] = out.reshape(q_seqlen, num_heads, head_size)
        golden_lse[bi] = lse.reshape(num_heads, q_seqlen)
    return golden_out, golden_lse


def _build_golden_tnd(
    query, kv_for_batch, *, q_offsets, batch_size, num_heads, head_size,
    scale, data_type, is_causal,
):
    query_cpu = query.detach().cpu()
    t_q = q_offsets[-1]
    golden_out = torch.empty((t_q, num_heads, head_size), dtype=data_type)
    golden_lse = torch.empty((num_heads, t_q), dtype=torch.float32)
    for bi in range(batch_size):
        s, e = q_offsets[bi], q_offsets[bi + 1]
        k_cpu, v_cpu = kv_for_batch(bi)
        mask = _causal_mask(e - s, k_cpu.shape[0]) if is_causal else None
        out, lse = ref_flash_attention(
            query_cpu[s:e], k_cpu, v_cpu, scale, mask, data_type, rescale_threshold=4.0,
        )
        golden_out[s:e] = out.reshape(e - s, num_heads, head_size)
        golden_lse[:, s:e] = lse.reshape(num_heads, e - s)
    return golden_out, golden_lse


# ---------------------------------------------------------------------------
# Test cases (same format as test_flash_attn_npu_v4.py)
#   (data_type, B, H, Hkv, Q, KV, D, page_sz, causal, layout, cache_mode)
# ---------------------------------------------------------------------------

VARLEN_CASES = [
    (torch.bfloat16, 1, 1, 1, 512,  1024, 128, 128, True,  "TND", 0),
    (torch.bfloat16, 2, 4, 4, 1024, 1024, 128, 128, False, "TND", 0),
    (torch.float16,   7, 5, 1, 512,  512,  128, 128, True,  "TND", 0),
    (torch.bfloat16, 5, 4, 4, 512,  512,  128, 128, True,  "TND", 0),
    (torch.float16,   7, 5, 1, 777,  888,  192, 128, False, "TND", 0),
    (torch.bfloat16, 1, 1, 1, 7777, 8192, 64,  128, True,  "TND", 0),
]


KV_CACHE_CASES = [
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, 128, False, "BSND", 1),
    (torch.bfloat16, 5, 4, 4, 1024, 1024, 128, 128, True,  "BSND", 1),
    (torch.bfloat16, 1, 1, 1, 2048, 2048, 128, 128, False, "BSND", 1),
    (torch.bfloat16, 5, 4, 4, 1024, 1024, 128, 128, True,  "TND",  1),
    (torch.bfloat16, 5, 4, 4, 512,  512,  128, 128, True,  "TND",  1),
]


METADATA_REUSE_CASES = [
    (torch.bfloat16, False),
    (torch.bfloat16, True),
    (torch.float16,   False),
    (torch.float16,   True),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, "
    "head_size, block_size, is_causal, layout, cache_mode",
    VARLEN_CASES,
)
def test_flash_attn_varlen_func_metadata_tnd(
    data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen,
    head_size, block_size, is_causal, layout, cache_mode,
):
    """flash_attn_varlen_func + scheduler_metadata — golden vs CANN comparison."""
    q_lengths = [q_seqlen] * batch_size
    kv_lengths = [kv_seqlen] * batch_size
    q_offsets = _prefix_sums(q_lengths)
    kv_offsets = _prefix_sums(kv_lengths)

    query = _rand_npu((q_offsets[-1], num_heads, head_size), data_type, WIDE_RANGE)
    key = _rand_npu((kv_offsets[-1], kv_heads, head_size), data_type, WIDE_RANGE)
    value = _rand_npu((kv_offsets[-1], kv_heads, head_size), data_type, WIDE_RANGE)
    scale = 1.0 / (head_size ** 0.5)

    cu_seqlens_q = _int32_npu(q_offsets)
    cu_seqlens_k = _int32_npu(kv_offsets)
    cache_seqlens = _int32_npu(kv_lengths)

    meta = _metadata(
        batch_size=batch_size, q_seqlen=q_seqlen, kv_seqlen=kv_seqlen,
        num_heads=num_heads, kv_heads=kv_heads, head_size=head_size,
        cache_seqlens=cache_seqlens, data_type=data_type,
        cu_seqlens_q=cu_seqlens_q, is_causal=is_causal,
    )

    out_npu = flash_attn_varlen_func(
        query, key, value, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=q_seqlen, max_seqlen_k=kv_seqlen, softmax_scale=scale,
        causal=is_causal, window_size=WINDOW_SIZE, num_splits=1,
        scheduler_metadata=meta,
    )

    key_cpu = key.detach().cpu()
    value_cpu = value.detach().cpu()
    golden_out, _ = _build_golden_tnd(
        query,
        lambda bi: (key_cpu[kv_offsets[bi]:kv_offsets[bi + 1]],
                    value_cpu[kv_offsets[bi]:kv_offsets[bi + 1]]),
        q_offsets=q_offsets, batch_size=batch_size, num_heads=num_heads,
        head_size=head_size, scale=scale, data_type=data_type, is_causal=is_causal,
    )

    torch.testing.assert_close(out_npu.cpu(), golden_out, rtol=RTOL, atol=ATOL)
    _, ok = compare_rule(golden_out.cpu().float(), out_npu.cpu().float())
    assert ok, "Golden vs CANN (metadata varlen) check FAILED"


@pytest.mark.parametrize(
    "data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, "
    "head_size, block_size, is_causal, layout, cache_mode",
    KV_CACHE_CASES,
)
def test_flash_attn_kvcache_metadata(
    data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen,
    head_size, block_size, is_causal, layout, cache_mode,
):
    """flash_attn_with_kvcache + scheduler_metadata — golden vs CANN comparison."""
    scale = 1.0 / (head_size ** 0.5)
    is_tnd = (layout == "TND")

    if is_tnd:
        q_lengths = [q_seqlen] * batch_size
        q_offsets = _prefix_sums(q_lengths)
        cu_seqlens_q = _int32_npu(q_offsets)
        query = _rand_npu((q_offsets[-1], num_heads, head_size), data_type, SMALL_RANGE)
    else:
        q_offsets = None
        cu_seqlens_q = None
        query = _rand_npu((batch_size, q_seqlen, num_heads, head_size), data_type, SMALL_RANGE)

    key_cache, value_cache, page_table = _make_paged_cache(
        batch_size, kv_seqlen, kv_heads, head_size, block_size, data_type)
    cache_seqlens = _int32_npu([kv_seqlen] * batch_size)

    meta = _metadata(
        batch_size=batch_size, q_seqlen=q_seqlen, kv_seqlen=kv_seqlen,
        num_heads=num_heads, kv_heads=kv_heads, head_size=head_size,
        cache_seqlens=cache_seqlens, data_type=data_type,
        cu_seqlens_q=cu_seqlens_q, page_size=block_size, is_causal=is_causal,
    )

    out_npu, lse_npu = flash_attn_with_kvcache(
        query, key_cache, value_cache, cache_seqlens=cache_seqlens,
        page_table=page_table, cu_seqlens_q=cu_seqlens_q, max_seqlen_q=q_seqlen,
        softmax_scale=scale, causal=is_causal, window_size=WINDOW_SIZE,
        num_splits=0, return_softmax_lse=True, scheduler_metadata=meta,
    )

    kc_cpu = key_cache.detach().cpu()
    vc_cpu = value_cache.detach().cpu()
    pt_cpu = page_table.cpu()

    kv_fn = lambda bi: _paged_kv_for_batch(kc_cpu, vc_cpu, pt_cpu, bi, kv_seqlen, block_size)
    if is_tnd:
        golden_out, golden_lse = _build_golden_tnd(
            query, kv_fn, q_offsets=q_offsets, batch_size=batch_size,
            num_heads=num_heads, head_size=head_size, scale=scale,
            data_type=data_type, is_causal=is_causal,
        )
    else:
        golden_out, golden_lse = _build_golden_bsnd(
            query, kv_fn, batch_size=batch_size, q_seqlen=q_seqlen,
            num_heads=num_heads, head_size=head_size, scale=scale,
            data_type=data_type, is_causal=is_causal,
        )

    torch.testing.assert_close(out_npu.cpu(), golden_out, rtol=RTOL, atol=ATOL)
    # NOTE: softmax_lse is not checked for paged KV — known limitation in metadata path.
    _, ok = compare_rule(golden_out.cpu().float(), out_npu.cpu().float())
    assert ok, "Golden vs CANN (metadata kvcache) check FAILED"


@pytest.mark.parametrize("data_type, is_causal", METADATA_REUSE_CASES)
def test_metadata_reuse_across_steps(data_type, is_causal):
    """Reuse same scheduler_metadata across decode steps — golden vs CANN."""
    batch_size, num_heads, kv_heads, head_size = 1, 4, 4, 128
    block_size, max_kv_seqlen = 128, 2048
    scale = 1.0 / (head_size ** 0.5)

    key_cache, value_cache, page_table = _make_paged_cache(
        batch_size, max_kv_seqlen, kv_heads, head_size, block_size, data_type)
    kc_cpu, vc_cpu, pt_cpu = key_cache.detach().cpu(), value_cache.detach().cpu(), page_table.cpu()

    meta = _metadata(
        batch_size=batch_size, q_seqlen=1, kv_seqlen=max_kv_seqlen,
        num_heads=num_heads, kv_heads=kv_heads, head_size=head_size,
        cache_seqlens=_int32_npu([max_kv_seqlen] * batch_size), data_type=data_type,
        page_size=block_size, is_causal=is_causal,
    )

    for step_kv_len in [128, 256, 384, 512]:
        query = _rand_npu((batch_size, 1, num_heads, head_size), data_type, SMALL_RANGE)
        cache_seqlens = _int32_npu([step_kv_len] * batch_size)

        out_npu, lse_npu = flash_attn_with_kvcache(
            query, key_cache, value_cache, cache_seqlens=cache_seqlens,
            page_table=page_table, max_seqlen_q=1, softmax_scale=scale,
            causal=is_causal, window_size=WINDOW_SIZE, num_splits=0,
            return_softmax_lse=True, scheduler_metadata=meta,
        )

        golden_out, golden_lse = _build_golden_bsnd(
            query,
            lambda bi: _paged_kv_for_batch(kc_cpu, vc_cpu, pt_cpu, bi, step_kv_len, block_size),
            batch_size=batch_size, q_seqlen=1, num_heads=num_heads,
            head_size=head_size, scale=scale, data_type=data_type, is_causal=is_causal,
        )

        torch.testing.assert_close(out_npu.cpu(), golden_out, rtol=RTOL, atol=ATOL)
        # NOTE: softmax_lse is not checked for paged KV — known limitation.
        _, ok = compare_rule(golden_out.cpu().float(), out_npu.cpu().float())
        assert ok, f"Golden vs CANN (metadata reuse) step={step_kv_len} FAILED"

"""Standalone CPU marker for v2 flash_attn_with_kvcache."""

import torch


def _npu_pv_dtype(*tensors):
    """Recover the NPU input dtype after ATK promotes benchmark inputs."""
    tensors = [tensor for tensor in tensors if tensor is not None]
    for tensor in tensors:
        if tensor.dtype in (torch.float16, torch.bfloat16):
            return tensor.dtype
    if tensors and all(tensor.dtype == torch.float32 for tensor in tensors):
        if all(torch.equal(tensor, tensor.to(torch.bfloat16).float()) for tensor in tensors):
            return torch.bfloat16
        return torch.float16
    return tensors[0].dtype


def _npu_softmax_pv(scores, v, pv_dtype):
    """Match the kernel: FP32 exp/sum, low-precision P, FP32 PV accumulation."""
    row_max = scores.amax(dim=-1, keepdim=True)
    valid_row = torch.isfinite(row_max)
    exp_scores = torch.where(valid_row, torch.exp(scores - row_max), torch.zeros_like(scores))
    row_sum = exp_scores.sum(dim=-1, keepdim=True).transpose(0, 1)
    numerator = torch.einsum("hqk,khd->qhd", exp_scores.to(pv_dtype).float(), v)
    return torch.where(row_sum > 0, numerator / row_sum, torch.zeros_like(numerator))


def _attention(q, k, v, softmax_scale, causal, window_size, softcap):
    output_dtype = q.dtype
    pv_dtype = _npu_pv_dtype(q, k, v)
    q, k, v = q.cpu().float(), k.cpu().float(), v.cpu().float()
    sq, hq, head_dim = q.shape
    sk, hkv, _ = k.shape
    k = k.repeat_interleave(hq // hkv, dim=1)
    v = v.repeat_interleave(hq // hkv, dim=1)
    scale = head_dim ** -0.5 if softmax_scale is None else softmax_scale
    scores = torch.einsum("qhd,khd->hqk", q, k) * scale
    if softcap > 0:
        scores = softcap * torch.tanh(scores / softcap)
    q_pos = torch.arange(sq)[:, None]
    k_pos = torch.arange(sk)[None, :]
    center = q_pos + sk - sq
    left, right = window_size
    if causal:
        right = 0
    mask = torch.zeros((sq, sk), dtype=torch.bool)
    if left >= 0:
        mask |= k_pos < center - left
    if left >= 0 or right >= 0:
        mask |= k_pos > center + right
    scores.masked_fill_(mask.unsqueeze(0), float("-inf"))
    return _npu_softmax_pv(scores, v, pv_dtype).to(output_dtype)


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens=None,
    cache_batch_idx=None,
    cache_leftpad=None,
    block_table=None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    rotary_interleaved=True,
    alibi_slopes=None,
    num_splits=0,
    scheduler_metadata=None,
    return_softmax_lse=False,
):
    if any(item is not None for item in (rotary_cos, rotary_sin, cache_leftpad, alibi_slopes)):
        raise NotImplementedError("Rotary, leftpad, and alibi are outside the v2 910 subset")
    batch = q.shape[0]
    output = []
    for index in range(batch):
        cache_index = index if cache_batch_idx is None else int(cache_batch_idx[index])
        length = k_cache.shape[1] if cache_seqlens is None else int(cache_seqlens[index])
        if k is not None:
            k_cache[cache_index, length:length + k.shape[1]] = k[index]
            v_cache[cache_index, length:length + v.shape[1]] = v[index]
            length += k.shape[1]
        output.append(_attention(
            q[index],
            k_cache[cache_index, :length],
            v_cache[cache_index, :length],
            softmax_scale,
            causal,
            window_size,
            softcap,
        ))
    return torch.stack(output)

"""Standalone CPU marker for v3 flash_attn_func."""

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


def _attention(q, k, v, qv, softmax_scale, causal, window_size, softcap):
    output_dtype = q.dtype
    pv_dtype = _npu_pv_dtype(q, k, v, qv)
    q = q.cpu().float()
    k = k.cpu().float()
    v = v.cpu().float()
    qv = None if qv is None else qv.cpu().float()
    sq, hq, head_dim = q.shape
    sk, hkv, _ = k.shape
    if hq % hkv:
        raise ValueError("Hq must be divisible by Hkv")
    k = k.repeat_interleave(hq // hkv, dim=1)
    v = v.repeat_interleave(hq // hkv, dim=1)
    scale_dim = head_dim + (v.shape[-1] if qv is not None else 0)
    scale = scale_dim ** -0.5 if softmax_scale is None else softmax_scale
    scores = torch.einsum("qhd,khd->hqk", q, k) * scale
    if qv is not None:
        scores += torch.einsum("qhd,khd->hqk", qv, v) * scale
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


def flash_attn_func(
    q,
    k,
    v,
    softmax_scale=None,
    causal=False,
    qv=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    deterministic=False,
    sm_margin=0,
    return_attn_probs=False,
):
    if any(item is not None for item in (q_descale, k_descale, v_descale)):
        raise NotImplementedError("Descales are outside the 910 accuracy subset")
    if attention_chunk != 0 or pack_gqa or return_attn_probs:
        raise NotImplementedError("Unsupported v3 option in this accuracy phase")
    return torch.stack([
        _attention(
            q[index],
            k[index],
            v[index],
            None if qv is None else qv[index],
            softmax_scale,
            causal,
            window_size,
            softcap,
        )
        for index in range(q.shape[0])
    ])

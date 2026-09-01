"""ATK CPU/GPU/NPU adapter for v4 flash_attn_varlen_func."""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from benchmark_flash_attn_varlen_func_v4 import flash_attn_varlen_func as cpu_flash_attn_varlen_func


def _balanced_cu(total, batch, device):
    base, remainder = divmod(total, batch)
    lengths = [base + (index < remainder) for index in range(batch)]
    values = [0]
    for length in lengths:
        values.append(values[-1] + length)
    return torch.tensor(values, dtype=torch.int32, device=device)


def _normalize(kwargs):
    batch = max(kwargs["cu_seqlens_q"].numel() - 1, 1)
    kwargs["cu_seqlens_q"] = _balanced_cu(kwargs["q"].shape[0], batch, kwargs["q"].device)
    kwargs["cu_seqlens_k"] = _balanced_cu(kwargs["k"].shape[0], batch, kwargs["k"].device)
    delta_q = kwargs["cu_seqlens_q"][1:] - kwargs["cu_seqlens_q"][:-1]
    delta_k = kwargs["cu_seqlens_k"][1:] - kwargs["cu_seqlens_k"][:-1]
    kwargs["max_seqlen_q"] = int(delta_q.max())
    kwargs["max_seqlen_k"] = int(delta_k.max())


@register("flash_attn_varlen_func_v4_accuracy")
class FlashAttnVarlenFuncV4Api(BaseApi):
    def __call__(self, input_data, with_output=False):
        kwargs = dict(input_data.kwargs)
        _normalize(kwargs)
        if self.device == "cpu":
            return cpu_flash_attn_varlen_func(**kwargs)
        if self.device == "gpu":
            # TriDao official FA3 (hopper) benchmark; imported lazily because
            # flash_attn_interface is only importable on Hopper GPU hosts.
            try:
                import flash_attn_interface as gpu_fa3
            except ImportError:
                from flash_attn_3 import flash_attn_interface as gpu_fa3

            kwargs = {
                name: value.to(f"cuda:{self.device_id}") if isinstance(value, torch.Tensor) else value
                for name, value in kwargs.items()
            }
            return gpu_fa3.flash_attn_varlen_func(**kwargs)
        if self.device == "npu":
            from flash_attn_npu_4 import flash_attn_varlen_func

            kwargs = {
                name: value.to(f"npu:{self.device_id}") if isinstance(value, torch.Tensor) else value
                for name, value in kwargs.items()
            }
            # scheduler_metadata is computed automatically on the AICPU when
            # omitted (v4 behavior), so no explicit metadata call is required.
            return flash_attn_varlen_func(**kwargs)
        raise RuntimeError(f"Unsupported backend: {self.device}")
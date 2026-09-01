"""ATK CPU/GPU/NPU adapter for v2 flash_attn_varlen_func."""

import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from benchmark_flash_attn_varlen_func_v2 import flash_attn_varlen_func as cpu_flash_attn_varlen_func


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


@register("flash_attn_varlen_func_v2_accuracy")
class FlashAttnVarlenFuncV2Api(BaseApi):
    def __call__(self, input_data, with_output=False):
        kwargs = dict(input_data.kwargs)
        _normalize(kwargs)
        if self.device == "cpu":
            return cpu_flash_attn_varlen_func(**kwargs)
        if self.device == "gpu":
            # TriDao official FA2 benchmark; imported lazily because the
            # flash_attn CUDA extension is only importable on GPU hosts.
            from flash_attn import flash_attn_varlen_func as gpu_flash_attn_varlen_func

            kwargs = {
                name: value.to(f"cuda:{self.device_id}") if isinstance(value, torch.Tensor) else value
                for name, value in kwargs.items()
            }
            return gpu_flash_attn_varlen_func(**kwargs)
        if self.device == "npu":
            import flash_attn_npu.flash_attn_npu_interface as interface
            from flash_attn_npu import flash_attn_varlen_func

            kwargs = {
                name: value.to(f"npu:{self.device_id}") if isinstance(value, torch.Tensor) else value
                for name, value in kwargs.items()
            }
            if os.environ.get("FA_ATK_USE_SCHEDULER_METADATA", "1") != "0":
                return flash_attn_varlen_func(**kwargs)
            original = interface.get_scheduler_metadata
            interface.get_scheduler_metadata = lambda *args, **params: None
            try:
                return flash_attn_varlen_func(**kwargs)
            finally:
                interface.get_scheduler_metadata = original
        raise RuntimeError(f"Unsupported backend: {self.device}")

"""Cross-input constraints for v2 ``flash_attn_varlen_func``."""

from math import ceil

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY


BATCHES = (1, 2, 4, 8)
HEADS = (1, 2, 4, 8, 16)


def _nearest(value, choices):
    return min(choices, key=lambda candidate: abs(candidate - max(int(value), 1)))


def _inputs(case_config):
    return {item.name: item for item in case_config.inputs if not isinstance(item, list)}


@GENERATOR_REGISTRY.register("flash_attn_varlen_func_v2_constraints")
class FlashAttnVarlenFuncV2Constraints(CaseGenerator):
    def after_case_config(self, case_config):
        inputs = _inputs(case_config)
        total_q, heads_q, head_dim = inputs["q"].shape
        total_k = inputs["k"].shape[0]
        valid_kv_heads = tuple(head for head in HEADS if head <= heads_q and heads_q % head == 0)
        heads_kv = _nearest(inputs["k"].shape[1], valid_kv_heads)
        dtype = inputs["q"].dtype

        if inputs["causal"].range_values is True:
            total_q = min(total_q, total_k)
        valid_batches = tuple(batch for batch in BATCHES if batch <= total_q and batch <= total_k)
        batch_candidate = max(inputs["cu_seqlens_q"].shape[0] - 1, 1)
        batch = _nearest(batch_candidate, valid_batches)

        inputs["q"].shape = [total_q, heads_q, head_dim]
        for name, shape in (
            ("k", [total_k, heads_kv, head_dim]),
            ("v", [total_k, heads_kv, head_dim]),
        ):
            inputs[name].dtype = dtype
            inputs[name].shape = shape
        for name, total in (("cu_seqlens_q", total_q), ("cu_seqlens_k", total_k)):
            inputs[name].dtype = "int32"
            inputs[name].shape = [batch + 1]
            inputs[name].range_values = [0, total]
        inputs["max_seqlen_q"].range_values = ceil(total_q / batch)
        inputs["max_seqlen_k"].range_values = ceil(total_k / batch)
        inputs["alibi_slopes"].range_values = "default"
        inputs["block_table"].range_values = "default"
        return case_config

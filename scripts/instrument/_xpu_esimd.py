# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guarded adapter for the optional ESIMD GDN decode core."""

import importlib
import os
from types import ModuleType
from typing import Any, NamedTuple

import torch

from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

logger = init_logger(__name__)

_MAX_DECODE_BATCH = 128
_HEAD_DIM = 128
_SPEC_GDN_DISABLED = os.environ.get("DISABLE_ESIMD_GDN_SPEC", "0") == "1"


# W57 probe hook: one log line per distinct (comp, detail) so an arm's
# engagement is provable from the engine log instead of inferred from tok/s.
# The set lives host-side and is updated at graph-capture time, so a captured
# kernel is recorded exactly once and graph replays add no log volume.
_XPU_W57_SEEN = set()


def _xpu_w57_note(comp: str, detail: str = "") -> None:
    key = (comp, detail)
    if key in _XPU_W57_SEEN:
        return
    _XPU_W57_SEEN.add(key)
    logger.info("[w57-census] comp=%s %s", comp, detail)

_PARAM_ATTRS = (
    "_esimd_gdn_conv_weight_fp16",
    "_esimd_gdn_conv_bias_fp16",
    "_esimd_gdn_A_log_fp16",
    "_esimd_gdn_dt_bias_fp16",
    "_esimd_gdn_attn_scale",
)


class _GdnInputPlan(NamedTuple):
    op: Any
    qkvz_weight: torch.Tensor
    qkvz_scale: torch.Tensor
    qkvz: torch.Tensor
    ba_weight: torch.Tensor
    ba_scale: torch.Tensor | None
    ba: torch.Tensor
    block_size: tuple[int, int] | None
    ba_is_fp16: bool


class _GdnOutputPlan(NamedTuple):
    op: Any
    norm_weight: torch.Tensor
    weight: torch.Tensor
    scale: torch.Tensor
    output: torch.Tensor
    local_heads: int
    head_dim: int
    eps: float
    reduce_results: bool
    block_size: tuple[int, int] | None


class _GdnOutputNormPlan(NamedTuple):
    op: Any
    norm_weight: torch.Tensor
    local_heads: int
    head_dim: int
    eps: float


class _GdnCoreGuardPlan(NamedTuple):
    op: Any
    local_k_heads: int
    local_v_heads: int
    head_k_dim: int
    head_v_dim: int
    qkvz_dim: int
    ba_dim: int


def _gdn_projection_disabled() -> bool:
    return os.environ.get("VLLM_XPU_DISABLE_ESIMD_GDN_PROJ", "0") == "1"


def _int4_esimd_disabled() -> bool:
    return (
        os.environ.get("DISABLE_ESIMD_INT4", "0") == "1"
        or os.environ.get("VLLM_XPU_DISABLE_ESIMD_INT4", "0") == "1"
    )


def _load_esimd_module() -> ModuleType:
    """Load the optional Torch 2.12 ESIMD package lazily."""
    return importlib.import_module("custom_esimd_kernels_vllm")


def _is_xpu_tensor(value: object) -> bool:
    return isinstance(value, torch.Tensor) and value.device.type == "xpu"


def _is_unadapted_linear(layer: object) -> bool:
    from vllm.lora.layers.base import BaseLayerWithLoRA

    return not isinstance(layer, BaseLayerWithLoRA)


def _get_base_layer_for_no_lora(layer: object) -> object | None:
    base_layer = getattr(layer, "base_layer", None)
    if base_layer is None:
        return layer
    punica_wrapper = getattr(layer, "punica_wrapper", None)
    if not getattr(punica_wrapper, "no_lora", False):
        return None
    return base_layer


def _fp8_linear_parts(
    layer: object,
    input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int] | None] | None:
    weight = getattr(layer, "weight", None)
    scale = getattr(layer, "weight_scale", None)
    block_scale = getattr(layer, "weight_scale_inv", None)
    block_size = tuple(getattr(layer, "weight_block_size", ()) or ())
    scalar_scale_ready = (
        isinstance(scale, torch.Tensor)
        and _is_contiguous_tensor(scale)
        and scale.numel() == 1
        and scale.dtype == torch.float32
    )
    block_scale_ready = (
        isinstance(block_scale, torch.Tensor)
        and _is_contiguous_tensor(block_scale, dtype=torch.float32, ndim=2)
        and block_size == (128, 128)
        and isinstance(weight, torch.Tensor)
        and tuple(block_scale.shape)
        == (
            (weight.shape[0] + block_size[0] - 1) // block_size[0],
            (weight.shape[1] + block_size[1] - 1) // block_size[1],
        )
    )
    selected_scale = block_scale if block_scale_ready else scale
    if not isinstance(weight, torch.Tensor) or not isinstance(
        selected_scale, torch.Tensor
    ):
        return None
    if not (
        _is_unadapted_linear(layer)
        and getattr(layer, "bias", None) is None
        and not getattr(layer, "gather_output", False)
        and _is_contiguous_tensor(weight, ndim=2)
        and weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        and (scalar_scale_ready or block_scale_ready)
        and weight.device == input.device
        and selected_scale.device == input.device
        and weight.shape[1] == input.shape[1]
    ):
        return None
    if block_scale_ready and isinstance(block_scale, torch.Tensor):
        return weight, block_scale, (128, 128)
    if not isinstance(scale, torch.Tensor):
        return None
    return weight, scale, None


def _int4_linear_parts(
    layer: object,
    input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    quant_method = getattr(layer, "quant_method", None)
    weight = getattr(layer, "weight_esimd", None)
    scale = getattr(layer, "scale_esimd", None)
    if not (
        not _int4_esimd_disabled()
        and _is_unadapted_linear(layer)
        and getattr(quant_method, "is_sym_int4", False)
        and getattr(layer, "bias", None) is None
        and not getattr(layer, "gather_output", False)
        and _is_contiguous_tensor(weight, dtype=torch.uint8, ndim=2)
        and _is_contiguous_tensor(scale, dtype=torch.float16, ndim=2)
        and weight.device == input.device
        and scale.device == input.device
        and weight.shape[1] * 2 == input.shape[1]
        and scale.shape[0] == weight.shape[0]
        and scale.shape[1] == input.shape[1] // 128
    ):
        return None
    return weight.view(torch.int32), scale


def _fp16_linear_weight(layer: object, input: torch.Tensor) -> torch.Tensor | None:
    weight = getattr(layer, "weight", None)
    if not (
        _is_unadapted_linear(layer)
        and getattr(layer, "bias", None) is None
        and not getattr(layer, "gather_output", False)
        and isinstance(weight, torch.Tensor)
        and _is_contiguous_tensor(weight, dtype=torch.float16, ndim=2)
        and weight.device == input.device
        and weight.shape[1] == input.shape[1]
    ):
        return None
    return weight


def try_esimd_gdn_input_projections(
    hidden_states: torch.Tensor,
    qkvz_proj: Any,
    ba_proj: Any,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Fuse the two scalar-scale FP8 GDN input projections for M=1 decode."""
    qkvz_proj = _get_base_layer_for_no_lora(qkvz_proj)
    ba_proj = _get_base_layer_for_no_lora(ba_proj)
    if qkvz_proj is None or ba_proj is None:
        return None
    plan = getattr(qkvz_proj, "_esimd_gdn_input_plan", None)
    if not isinstance(plan, _GdnInputPlan):
        plan = None
    if hidden_states.ndim != 2 or hidden_states.shape[0] != 1:
        return None

    if plan is None:
        if (
            _gdn_projection_disabled()
            or getattr(torch.compiler, "is_compiling", lambda: False)()
            or not _is_contiguous_tensor(hidden_states, dtype=torch.float16, ndim=2)
        ):
            return None
        qkvz_parts = _fp8_linear_parts(qkvz_proj, hidden_states)
        ba_parts = _fp8_linear_parts(ba_proj, hidden_states)
        ba_fp16_weight = (
            None
            if ba_parts is not None
            else _fp16_linear_weight(ba_proj, hidden_states)
        )
        if qkvz_parts is None or (ba_parts is None and ba_fp16_weight is None):
            return None

        try:
            module = _load_esimd_module()
        except (ImportError, OSError, AttributeError, RuntimeError):
            return None
        qkvz_weight, qkvz_scale, qkvz_block_size = qkvz_parts
        if ba_parts is not None:
            ba_weight, ba_scale, ba_block_size = ba_parts
        else:
            ba_weight, ba_scale, ba_block_size = ba_fp16_weight, None, None
        if qkvz_block_size is None and ba_block_size is None:
            op_name = "esimd_gemv_fp8_pert_fused2"
        elif qkvz_block_size == (128, 128) and ba_block_size == (128, 128):
            op_name = "esimd_gemv_fp8_blockscale_fused2"
        elif qkvz_block_size == (128, 128) and ba_fp16_weight is not None:
            op_name = "esimd_gemv_fp8_blockscale_fp16_fused2"
        else:
            return None
        op = getattr(module, op_name, None)
        if not callable(op):
            return None

        qkvz = torch.empty(
            (1, qkvz_weight.shape[0]),
            dtype=torch.float16,
            device=hidden_states.device,
        )
        ba = torch.empty(
            (1, ba_weight.shape[0]),
            dtype=torch.float16,
            device=hidden_states.device,
        )
        qkvz_proj._esimd_gdn_input_buffers = qkvz, ba
        plan = _GdnInputPlan(
            op=op,
            qkvz_weight=qkvz_weight,
            qkvz_scale=qkvz_scale,
            qkvz=qkvz,
            ba_weight=ba_weight,
            ba_scale=ba_scale,
            ba=ba,
            block_size=qkvz_block_size,
            ba_is_fp16=ba_fp16_weight is not None,
        )
        qkvz_proj._esimd_gdn_input_plan = plan

    if plan.block_size is None:
        plan.op(
            hidden_states,
            plan.qkvz_weight,
            plan.qkvz_scale,
            plan.qkvz,
            plan.ba_weight,
            plan.ba_scale,
            plan.ba,
        )
    elif plan.ba_is_fp16:
        plan.op(
            hidden_states,
            plan.qkvz_weight,
            plan.qkvz_scale,
            plan.qkvz,
            plan.ba_weight,
            plan.ba,
            plan.block_size[0],
            plan.block_size[1],
        )
    else:
        plan.op(
            hidden_states,
            plan.qkvz_weight,
            plan.qkvz_scale,
            plan.qkvz,
            plan.ba_weight,
            plan.ba_scale,
            plan.ba,
            plan.block_size[0],
            plan.block_size[1],
        )
    return plan.qkvz, plan.ba


def try_esimd_gdn_fp16_ba_projection(
    hidden_states: torch.Tensor,
    qkvz_proj: Any,
    ba_proj: Any,
) -> torch.Tensor | None:
    """Use the exact FP16 a/b weight with ESIMD for 2..8-token decode.

    Offline block checkpoints keep ``in_proj_ba`` in FP16 while qkvz is block
    FP8. The M=1 pair is handled by the fused2 kernel above; this helper avoids
    oneDNN setup for the small batched FP16 projection.
    """
    qkvz_proj = _get_base_layer_for_no_lora(qkvz_proj)
    ba_proj = _get_base_layer_for_no_lora(ba_proj)
    if qkvz_proj is None or ba_proj is None:
        return None
    rows = hidden_states.shape[0] if hidden_states.ndim == 2 else 0
    if not 2 <= rows <= 8:
        return None
    if (
        _gdn_projection_disabled()
        or getattr(torch.compiler, "is_compiling", lambda: False)()
        or not _is_contiguous_tensor(hidden_states, dtype=torch.float16, ndim=2)
        or _fp8_linear_parts(qkvz_proj, hidden_states) is None
    ):
        return None
    qkvz_scale = getattr(qkvz_proj, "weight_scale_inv", None)
    if qkvz_scale is None:
        return None
    weight = _fp16_linear_weight(ba_proj, hidden_states)
    if weight is None:
        return None
    op = getattr(ba_proj, "_esimd_gdn_ba_op", None)
    if op is None:
        try:
            module = _load_esimd_module()
        except (ImportError, OSError, AttributeError, RuntimeError):
            return None
        op = getattr(module, "esimd_gemv_fp16", None)
        if not callable(op):
            return None
        ba_proj._esimd_gdn_ba_op = op
    buffers = getattr(ba_proj, "_esimd_gdn_ba_buffers", None)
    if buffers is None:
        buffers = {}
        ba_proj._esimd_gdn_ba_buffers = buffers
    output = buffers.get(rows)
    if output is None:
        output = torch.empty(
            rows, weight.shape[0], dtype=torch.float16, device=hidden_states.device
        )
        buffers[rows] = output
    op(hidden_states, weight, output)
    return output


def try_esimd_gdn_output_projection(
    layer: Any,
    core_attn_out: torch.Tensor,
    z: torch.Tensor,
) -> torch.Tensor | None:
    """Fuse gated RMSNorm and FP8 or symmetric INT4 out projection for M=1."""
    base_out_proj = _get_base_layer_for_no_lora(layer.out_proj)
    if base_out_proj is None:
        return None
    plan = getattr(layer, "_esimd_gdn_output_plan", None)
    if not isinstance(plan, _GdnOutputPlan):
        plan = None
    if (
        core_attn_out.ndim != 3
        or z.ndim != 3
        or core_attn_out.shape != z.shape
        or core_attn_out.shape[0] != 1
    ):
        return None

    if plan is None:
        if (
            _gdn_projection_disabled()
            or getattr(torch.compiler, "is_compiling", lambda: False)()
            or not _is_contiguous_tensor(core_attn_out, dtype=torch.float16, ndim=3)
            or not _is_contiguous_tensor(z, dtype=torch.float16, ndim=3)
        ):
            return None
        norm = layer.norm
        out_proj = base_out_proj
        flattened = core_attn_out.flatten(1)
        fp8_parts = _fp8_linear_parts(out_proj, flattened)
        int4_parts = (
            None if fp8_parts is not None else _int4_linear_parts(out_proj, flattened)
        )
        if (
            (fp8_parts is None and int4_parts is None)
            or str(norm.activation).lower() not in {"silu", "swish"}
            or norm.group_size is not None
            or not norm.norm_before_gate
        ):
            return None

        local_heads = int(layer.num_v_heads) // int(layer.tp_size)
        head_dim = int(layer.head_v_dim)
        if core_attn_out.shape[1:] != (local_heads, head_dim):
            return None

        try:
            module = _load_esimd_module()
        except (ImportError, OSError, AttributeError, RuntimeError):
            return None
        if fp8_parts is not None:
            weight, scale, block_size = fp8_parts
            op_name = (
                "esimd_norm_gemv_fp8_blockscale"
                if block_size is not None
                else "esimd_norm_gemv_fp8_pert"
            )
        else:
            assert int4_parts is not None
            weight, scale = int4_parts
            block_size = None
            op_name = "esimd_norm_gemv_int4_pert"
        op = getattr(module, op_name, None)
        if not callable(op):
            return None

        norm_weight_fp16 = norm.weight.detach().to(dtype=torch.float16).contiguous()
        if not (
            _is_contiguous_tensor(norm_weight_fp16, dtype=torch.float16, ndim=1)
            and norm_weight_fp16.shape == (head_dim,)
            and norm_weight_fp16.device == core_attn_out.device
        ):
            return None
        layer._esimd_gdn_norm_weight_fp16 = norm_weight_fp16

        output = torch.empty(
            (1, weight.shape[0]), dtype=torch.float16, device=core_attn_out.device
        )
        layer._esimd_gdn_output_buffer = output
        plan = _GdnOutputPlan(
            op=op,
            norm_weight=norm_weight_fp16,
            weight=weight,
            scale=scale,
            output=output,
            local_heads=local_heads,
            head_dim=head_dim,
            eps=norm.eps,
            reduce_results=int(layer.tp_size) > 1 and out_proj.reduce_results,
            block_size=block_size,
        )
        layer._esimd_gdn_output_plan = plan
    elif core_attn_out.shape[1:] != (plan.local_heads, plan.head_dim):
        return None

    plan.op(
        core_attn_out.view(plan.local_heads, plan.head_dim),
        z.view(plan.local_heads, plan.head_dim),
        plan.norm_weight,
        plan.weight,
        plan.scale,
        plan.output,
        plan.local_heads,
        plan.head_dim,
        plan.eps,
    )

    output = plan.output
    if plan.reduce_results:
        from vllm.distributed import tensor_model_parallel_all_reduce

        output = tensor_model_parallel_all_reduce(output)
    return output


def try_esimd_gdn_output_norm_batched(
    layer: Any,
    core_attn_out: torch.Tensor,
    z: torch.Tensor,
) -> torch.Tensor | None:
    """Replace RMSNormGated (Triton layer_norm_fwd) with the ESIMD gated
    RMSNorm for the 2 <= M <= 128 pure-decode batch, matching v0.21.0.

    Only the norm is fused here (unlike ``try_esimd_gdn_output_projection``,
    which fuses norm + FP8 GEMV for the M==1 tail): out_proj for M>1 runs a
    regular GEMM and already hits its own scaled_mm fast path, so there is
    nothing to gain from fusing it into this kernel. Caller still reshapes
    the result and calls ``layer.out_proj`` itself.
    """
    plan = getattr(layer, "_esimd_gdn_output_norm_plan", None)
    if not isinstance(plan, _GdnOutputNormPlan):
        plan = None
    if (
        core_attn_out.ndim != 3
        or z.ndim != 3
        or core_attn_out.shape != z.shape
        or not (2 <= core_attn_out.shape[0] <= _MAX_DECODE_BATCH)
    ):
        return None

    if plan is None:
        if (
            _gdn_projection_disabled()
            or getattr(torch.compiler, "is_compiling", lambda: False)()
            or not _is_contiguous_tensor(core_attn_out, dtype=torch.float16, ndim=3)
            or not _is_contiguous_tensor(z, dtype=torch.float16, ndim=3)
        ):
            return None
        norm = layer.norm
        if (
            str(norm.activation).lower() not in {"silu", "swish"}
            or norm.group_size is not None
            or not norm.norm_before_gate
        ):
            return None

        local_heads = int(layer.num_v_heads) // int(layer.tp_size)
        head_dim = int(layer.head_v_dim)
        if core_attn_out.shape[1:] != (local_heads, head_dim):
            return None

        try:
            module = _load_esimd_module()
        except (ImportError, OSError, AttributeError, RuntimeError):
            return None
        op = getattr(module, "esimd_rms_norm_gated", None)
        if not callable(op):
            return None

        # Private contiguous fp16 copy: the norm weight is stored fp32 and
        # the XPU caching allocator may otherwise reuse a shared half()
        # result mid-run, matching the v0.21.0 comment/workaround.
        norm_weight_fp16 = norm.weight.detach().to(dtype=torch.float16).contiguous()
        if not (
            _is_contiguous_tensor(norm_weight_fp16, dtype=torch.float16, ndim=1)
            and norm_weight_fp16.shape == (head_dim,)
            and norm_weight_fp16.device == core_attn_out.device
        ):
            return None
        layer._esimd_gdn_output_norm_weight_fp16 = norm_weight_fp16

        plan = _GdnOutputNormPlan(
            op=op,
            norm_weight=norm_weight_fp16,
            local_heads=local_heads,
            head_dim=head_dim,
            eps=norm.eps,
        )
        layer._esimd_gdn_output_norm_plan = plan
    elif core_attn_out.shape[1:] != (plan.local_heads, plan.head_dim):
        return None

    # RMSNormGated normalizes per head over head_v_dim (the norm weight is
    # [head_v_dim], not [local_heads*head_dim]); flatten to
    # [M*local_heads, head_dim] so each row is one head, matching the M==1
    # fused tail, then reshape to [M, local_heads*head_dim] for out_proj.
    num_tokens = core_attn_out.shape[0]
    x_rows = core_attn_out.reshape(num_tokens * plan.local_heads, plan.head_dim)
    z_rows = z.reshape(num_tokens * plan.local_heads, plan.head_dim)
    normed_rows = torch.empty_like(x_rows)
    plan.op(x_rows, z_rows, plan.norm_weight, normed_rows, plan.eps)
    return normed_rows.reshape(num_tokens, plan.local_heads * plan.head_dim)


def _metadata_for_layer(layer: Any):
    try:
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict):
            return None
        return metadata.get(layer.prefix)
    except (AttributeError, TypeError):
        return None


def _is_contiguous_tensor(
    value: object,
    *,
    dtype: torch.dtype | None = None,
    ndim: int | None = None,
) -> bool:
    if not isinstance(value, torch.Tensor) or not _is_xpu_tensor(value):
        return False
    if dtype is not None and value.dtype != dtype:
        return False
    if ndim is not None and value.ndim != ndim:
        return False
    return value.is_contiguous()


def _cached_decode_guard(
    layer: Any,
    plan: _GdnCoreGuardPlan,
    projected_qkvz: torch.Tensor,
    projected_ba: torch.Tensor,
    core_attn_out: torch.Tensor,
    z: torch.Tensor,
):
    metadata = _metadata_for_layer(layer)
    if metadata is None:
        return None
    try:
        num_decodes = int(metadata.num_decodes)
        if not (
            int(metadata.num_prefills) == 0
            and int(metadata.num_prefill_tokens) == 0
            and int(metadata.num_spec_decodes) == 0
            and int(metadata.num_spec_decode_tokens) == 0
            and num_decodes
            == int(metadata.num_decode_tokens)
            == int(metadata.num_actual_tokens)
            and 0 < num_decodes <= _MAX_DECODE_BATCH
            and metadata.spec_sequence_masks is None
        ):
            return None

        expected_output_shape = (
            num_decodes,
            plan.local_v_heads,
            plan.head_v_dim,
        )
        if not (
            projected_qkvz.shape == (num_decodes, plan.qkvz_dim)
            and projected_ba.shape == (num_decodes, plan.ba_dim)
            and core_attn_out.shape == expected_output_shape
            and z.shape == expected_output_shape
        ):
            return None

        state_indices = metadata.non_spec_state_indices_tensor
        if not (
            isinstance(state_indices, torch.Tensor)
            and state_indices.dtype == torch.int32
            and state_indices.ndim == 1
            and state_indices.is_contiguous()
            and state_indices.numel() >= num_decodes
        ):
            return None

        conv_state, ssm_state = layer.kv_cache
        if state_indices.device.type == "cpu":
            selected_indices = state_indices[:num_decodes]
            if bool((selected_indices < 0).any()) or bool(
                (selected_indices >= conv_state.shape[0]).any()
            ):
                return None
        conv_weight = layer.conv1d.weight
        conv_bias = layer.conv1d.bias
    except (AttributeError, IndexError, TypeError, ValueError):
        return None

    return (
        plan.op,
        metadata,
        state_indices[:num_decodes],
        conv_state,
        ssm_state,
        conv_weight,
        conv_bias,
        num_decodes,
        plan.local_k_heads,
        plan.local_v_heads,
        plan.head_k_dim,
        plan.head_v_dim,
    )


def _decode_guard(
    layer: Any,
    projected_qkvz: torch.Tensor,
    projected_ba: torch.Tensor,
    core_attn_out: torch.Tensor,
    z: torch.Tensor,
):
    """Return validated decode metadata, dimensions, and state tensors."""
    cached_plan = getattr(layer, "_esimd_gdn_core_guard_plan", None)
    if isinstance(cached_plan, _GdnCoreGuardPlan):
        return _cached_decode_guard(
            layer,
            cached_plan,
            projected_qkvz,
            projected_ba,
            core_attn_out,
            z,
        )

    metadata = _metadata_for_layer(layer)
    if metadata is None:
        return None

    try:
        num_decodes = int(metadata.num_decodes)
        num_actual_tokens = int(metadata.num_actual_tokens)
        num_prefills = int(metadata.num_prefills)
        num_prefill_tokens = int(metadata.num_prefill_tokens)
        num_spec_decodes = int(metadata.num_spec_decodes)
        num_spec_decode_tokens = int(metadata.num_spec_decode_tokens)
        num_decode_tokens = int(metadata.num_decode_tokens)
        if not (
            num_prefills == 0
            and num_prefill_tokens == 0
            and num_spec_decodes == 0
            and num_spec_decode_tokens == 0
            and num_decodes == num_decode_tokens == num_actual_tokens
            and 0 < num_decodes <= _MAX_DECODE_BATCH
            and metadata.spec_sequence_masks is None
        ):
            return None

        tp_size = int(layer.tp_size)
        num_k_heads = int(layer.num_k_heads)
        num_v_heads = int(layer.num_v_heads)
        head_k_dim = int(layer.head_k_dim)
        head_v_dim = int(layer.head_v_dim)
        if (
            tp_size <= 0
            or num_k_heads % tp_size != 0
            or num_v_heads % tp_size != 0
            or head_k_dim != _HEAD_DIM
            or head_v_dim != _HEAD_DIM
            or int(layer.conv_kernel_size) != 4
            or num_v_heads % num_k_heads != 0
            or str(layer.activation).lower() not in {"silu", "swish"}
            or is_conv_state_dim_first()
            or not isinstance(layer.gqa_interleaved_layout, bool)
        ):
            return None

        local_k_heads = num_k_heads // tp_size
        local_v_heads = num_v_heads // tp_size
        if layer.gqa_interleaved_layout:
            remaining_threads = 32 - 4 * local_k_heads
            if remaining_threads <= 0:
                return None
            double_v = local_v_heads > remaining_threads // 2
            useful_threads = 4 * local_k_heads + (
                local_v_heads if double_v else 2 * local_v_heads
            )
            if useful_threads > 32:
                return None
        conv_dim = 2 * local_k_heads * head_k_dim + local_v_heads * head_v_dim
        qkvz_dim = 2 * local_k_heads * head_k_dim + 2 * local_v_heads * head_v_dim
        ba_dim = 2 * local_v_heads
        expected_output_shape = (num_decodes, local_v_heads, head_v_dim)

        if not (
            _is_contiguous_tensor(projected_qkvz, dtype=torch.float16, ndim=2)
            and projected_qkvz.shape == (num_decodes, qkvz_dim)
            and _is_contiguous_tensor(projected_ba, dtype=torch.float16, ndim=2)
            and projected_ba.shape == (num_decodes, ba_dim)
            and _is_contiguous_tensor(core_attn_out, dtype=torch.float16, ndim=3)
            and core_attn_out.shape == expected_output_shape
            and _is_contiguous_tensor(z, dtype=torch.float16, ndim=3)
            and z.shape == expected_output_shape
        ):
            return None

        state_indices = metadata.non_spec_state_indices_tensor
        if not (
            _is_contiguous_tensor(state_indices, dtype=torch.int32, ndim=1)
            and state_indices.numel() >= num_decodes
        ):
            return None

        conv_state, ssm_state = layer.kv_cache
        if not (
            _is_xpu_tensor(conv_state)
            and conv_state.dtype == torch.float16
            and conv_state.ndim == 3
            and conv_state.shape[0] > 0
            and conv_state.shape[1] == int(layer.conv_kernel_size) - 1
            and conv_state.shape[2] == conv_dim
            and conv_state.stride()[1:] == (conv_dim, 1)
            and _is_xpu_tensor(ssm_state)
            and ssm_state.dtype == torch.float16
            and ssm_state.ndim == 4
            and ssm_state.shape[0] > 0
            and ssm_state.shape[1:] == (local_v_heads, head_v_dim, head_k_dim)
            and ssm_state.stride()[1:] == (head_v_dim * head_k_dim, head_k_dim, 1)
            and state_indices.device == conv_state.device
            and conv_state.device == ssm_state.device
        ):
            return None

        conv_weight = layer.conv1d.weight
        if not (
            _is_xpu_tensor(conv_weight)
            and conv_weight.ndim == 3
            and conv_weight.shape == (conv_dim, 1, int(layer.conv_kernel_size))
            and conv_weight.is_contiguous()
            and conv_weight.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and _is_xpu_tensor(layer.A_log)
            and layer.A_log.ndim == 1
            and layer.A_log.shape == (local_v_heads,)
            and _is_xpu_tensor(layer.dt_bias)
            and layer.dt_bias.ndim == 1
            and layer.dt_bias.shape == (local_v_heads,)
            and conv_weight.device == layer.A_log.device
            and conv_weight.device == layer.dt_bias.device
            and projected_qkvz.device == conv_weight.device
            and projected_ba.device == projected_qkvz.device
            and core_attn_out.device == projected_qkvz.device
            and z.device == projected_qkvz.device
        ):
            return None

        # Metadata builders produce in-range XPU indices. Validate CPU/mock
        # indices here as well without introducing a per-layer device sync on
        # the real eager XPU path.
        if state_indices.device.type == "cpu":
            selected_indices = state_indices[:num_decodes]
            if bool((selected_indices < 0).any()) or bool(
                (selected_indices >= conv_state.shape[0]).any()
            ):
                return None

        conv_bias = layer.conv1d.bias
        if conv_bias is not None and (
            not _is_xpu_tensor(conv_bias)
            or conv_bias.ndim != 1
            or conv_bias.shape != (conv_dim,)
            or conv_bias.device != conv_weight.device
        ):
            return None
    except (AttributeError, IndexError, TypeError, ValueError):
        return None

    try:
        module = _load_esimd_module()
    except (ImportError, OSError, AttributeError, RuntimeError):
        return None
    op_name = (
        "esimd_gdn_conv_fused"
        if layer.gqa_interleaved_layout
        else "esimd_gdn_conv_fused_seq"
    )
    op = getattr(module, op_name, None)
    if not callable(op):
        return None

    layer._esimd_gdn_core_guard_plan = _GdnCoreGuardPlan(
        op=op,
        local_k_heads=local_k_heads,
        local_v_heads=local_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        qkvz_dim=qkvz_dim,
        ba_dim=ba_dim,
    )
    return (
        op,
        metadata,
        state_indices[:num_decodes],
        conv_state,
        ssm_state,
        conv_weight,
        conv_bias,
        num_decodes,
        local_k_heads,
        local_v_heads,
        head_k_dim,
        head_v_dim,
    )


def try_esimd_gdn_core_decode(
    layer: Any,
    projected_qkvz: torch.Tensor,
    projected_ba: torch.Tensor,
    core_attn_out: torch.Tensor,
    z: torch.Tensor,
) -> bool:
    """Run the ESIMD GDN core when all decode-only guards pass.

    The function returns ``False`` without mutating state when the optional
    package, metadata, layout, dtype, or shape is not eligible. Callers must
    invoke the existing native GDN op when this returns ``False``.
    """
    cached_plan = getattr(layer, "_esimd_gdn_core_guard_plan", None)
    if (
        not isinstance(cached_plan, _GdnCoreGuardPlan)
        and getattr(torch.compiler, "is_compiling", lambda: False)()
    ):
        return False

    guard = _decode_guard(layer, projected_qkvz, projected_ba, core_attn_out, z)
    if guard is None:
        return False

    (
        op,
        _metadata,
        state_indices,
        conv_state,
        ssm_state,
        conv_weight,
        conv_bias,
        num_decodes,
        local_k_heads,
        local_v_heads,
        head_k_dim,
        head_v_dim,
    ) = guard

    # The optional kernel consumes fp16 parameter buffers. Build them only
    # after every fallback guard above has passed, so rejection is side-effect
    # free with respect to the layer and both state pools.
    if not all(hasattr(layer, name) for name in _PARAM_ATTRS):
        try:
            conv_weight_fp16 = (
                conv_weight.view(conv_weight.shape[0], conv_weight.shape[2])
                .detach()
                .to(dtype=torch.float16)
                .contiguous()
            )
            conv_bias_fp16 = (
                torch.zeros(
                    conv_weight.shape[0],
                    dtype=torch.float16,
                    device=conv_weight.device,
                )
                if conv_bias is None
                else conv_bias.detach().to(dtype=torch.float16).contiguous()
            )
            a_log_fp16 = layer.A_log.detach().to(dtype=torch.float16).contiguous()
            dt_bias_fp16 = layer.dt_bias.detach().to(dtype=torch.float16).contiguous()
            attn_scale = float(head_k_dim**-0.5)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return False
        layer._esimd_gdn_conv_weight_fp16 = conv_weight_fp16
        layer._esimd_gdn_conv_bias_fp16 = conv_bias_fp16
        layer._esimd_gdn_A_log_fp16 = a_log_fp16
        layer._esimd_gdn_dt_bias_fp16 = dt_bias_fp16
        layer._esimd_gdn_attn_scale = attn_scale

    # Do not catch execution errors here. The kernel mutates the recurrent
    # state in place, so falling back after a partial launch could corrupt it.
    _xpu_w57_note(
        "gdn_decode_esimd",
        f"num_decodes={num_decodes} conv={conv_state.dtype} "
        f"ssm={ssm_state.dtype} qkvz={projected_qkvz.dtype}",
    )
    op(
        projected_qkvz,
        conv_state,
        layer._esimd_gdn_conv_weight_fp16,
        layer._esimd_gdn_conv_bias_fp16,
        state_indices,
        layer._esimd_gdn_A_log_fp16,
        layer._esimd_gdn_dt_bias_fp16,
        projected_ba,
        ssm_state,
        state_indices,
        core_attn_out,
        z,
        num_decodes,
        local_k_heads,
        local_v_heads,
        head_k_dim,
        head_v_dim,
        layer._esimd_gdn_attn_scale,
    )
    return True


def _active_prefix(value: torch.Tensor, size: int) -> torch.Tensor | None:
    if value.numel() < size:
        return None
    if value.numel() == size:
        return value
    return value.narrow(0, 0, size)


def _active_spec_state_prefix(
    value: torch.Tensor, num_spec_tokens: int
) -> torch.Tensor | None:
    if value.numel() < num_spec_tokens:
        return None
    if value.numel() == num_spec_tokens:
        return value
    if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] != num_spec_tokens:
        return None
    return value.narrow(0, 0, 1)


def try_esimd_gdn_core_spec_decode(
    layer: Any,
    projected_qkvz: torch.Tensor,
    projected_ba: torch.Tensor,
    core_attn_out: torch.Tensor,
    z: torch.Tensor,
) -> bool:
    """Run the sequential ESIMD GDN core for one speculative request."""
    if _SPEC_GDN_DISABLED or getattr(
        torch.compiler, "is_compiling", lambda: False
    )():
        return False

    metadata = _metadata_for_layer(layer)
    if metadata is None:
        return False

    try:
        num_spec_decodes = int(metadata.num_spec_decodes)
        num_spec_tokens = int(metadata.num_spec_decode_tokens)
        tp_size = int(layer.tp_size)
        if not (
            int(metadata.num_prefills) == 0
            and int(metadata.num_prefill_tokens) == 0
            and int(metadata.num_decodes) == 0
            and int(metadata.num_decode_tokens) == 0
            and num_spec_decodes == 1
            and num_spec_tokens == int(metadata.num_actual_tokens)
            and num_spec_tokens > 1
            and metadata.spec_sequence_masks is not None
            and not layer.gqa_interleaved_layout
            and tp_size > 0
            and int(layer.num_k_heads) % tp_size == 0
            and int(layer.num_v_heads) % tp_size == 0
            and int(layer.head_k_dim) == _HEAD_DIM
            and int(layer.head_v_dim) == _HEAD_DIM
            and int(layer.conv_kernel_size) == 4
            and str(layer.activation).lower() in {"silu", "swish"}
            and not is_conv_state_dim_first()
        ):
            return False

        local_k_heads = int(layer.num_k_heads) // tp_size
        local_v_heads = int(layer.num_v_heads) // tp_size
        if local_k_heads != 8 or local_v_heads not in (16, 24):
            return False

        qkvz_dim = 2 * local_k_heads * _HEAD_DIM + 2 * local_v_heads * _HEAD_DIM
        ba_dim = 2 * local_v_heads
        output_shape = (num_spec_tokens, local_v_heads, _HEAD_DIM)
        if not (
            _is_contiguous_tensor(projected_qkvz, dtype=torch.float16, ndim=2)
            and projected_qkvz.shape == (num_spec_tokens, qkvz_dim)
            and _is_contiguous_tensor(projected_ba, dtype=torch.float16, ndim=2)
            and projected_ba.shape == (num_spec_tokens, ba_dim)
            and _is_contiguous_tensor(core_attn_out, dtype=torch.float16, ndim=3)
            and core_attn_out.shape == output_shape
            and _is_contiguous_tensor(z, dtype=torch.float16, ndim=3)
            and z.shape == output_shape
        ):
            return False

        state_indices = metadata.spec_state_indices_tensor
        spec_token_indices = metadata.spec_token_indx
        num_accepted_tokens = metadata.num_accepted_tokens
        if not (
            _is_contiguous_tensor(state_indices, dtype=torch.int32)
            and state_indices.ndim in (1, 2)
            and _is_contiguous_tensor(
                spec_token_indices, dtype=torch.int32, ndim=1
            )
            and _is_contiguous_tensor(
                num_accepted_tokens, dtype=torch.int32, ndim=1
            )
        ):
            return False
        state_indices = _active_spec_state_prefix(state_indices, num_spec_tokens)
        spec_token_indices = _active_prefix(spec_token_indices, num_spec_tokens)
        num_accepted_tokens = _active_prefix(
            num_accepted_tokens, num_spec_decodes
        )
        if (
            state_indices is None
            or spec_token_indices is None
            or num_accepted_tokens is None
        ):
            return False

        conv_state, ssm_state = layer.kv_cache
        conv_dim = 2 * local_k_heads * _HEAD_DIM + local_v_heads * _HEAD_DIM
        conv_weight = layer.conv1d.weight
        conv_bias = layer.conv1d.bias
        if not (
            _is_xpu_tensor(conv_state)
            and conv_state.dtype == torch.float16
            and conv_state.ndim == 3
            and conv_state.shape[0] > 0
            and (
                conv_state.shape[1] == 3
                or conv_state.shape[1] >= num_spec_tokens + 2
            )
            and conv_state.shape[2] == conv_dim
            and conv_state.stride()[1:] == (conv_dim, 1)
            and _is_xpu_tensor(ssm_state)
            and ssm_state.dtype == torch.float16
            and ssm_state.shape[1:] == (
                local_v_heads,
                _HEAD_DIM,
                _HEAD_DIM,
            )
            and _is_xpu_tensor(conv_weight)
            and conv_weight.shape == (conv_dim, 1, 4)
            and conv_weight.is_contiguous()
            and conv_weight.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and _is_xpu_tensor(layer.A_log)
            and layer.A_log.shape == (local_v_heads,)
            and _is_xpu_tensor(layer.dt_bias)
            and layer.dt_bias.shape == (local_v_heads,)
            and state_indices.device == conv_state.device
            and spec_token_indices.device == conv_state.device
            and num_accepted_tokens.device == conv_state.device
            and ssm_state.device == conv_state.device
            and conv_weight.device == conv_state.device
            and layer.A_log.device == conv_state.device
            and layer.dt_bias.device == conv_state.device
            and projected_qkvz.device == conv_state.device
            and projected_ba.device == conv_state.device
            and core_attn_out.device == conv_state.device
            and z.device == conv_state.device
        ):
            return False
        if state_indices.device.type == "cpu" and (
            bool((state_indices < 0).any())
            or bool((state_indices >= conv_state.shape[0]).any())
        ):
            return False
        if conv_bias is not None and (
            not _is_xpu_tensor(conv_bias)
            or conv_bias.shape != (conv_dim,)
            or conv_bias.device != conv_weight.device
        ):
            return False
    except (AttributeError, IndexError, TypeError, ValueError):
        return False

    try:
        op = _load_esimd_module().esimd_gdn_conv_fused_seq_spec
    except (ImportError, OSError, AttributeError, RuntimeError):
        return False
    if not callable(op):
        return False

    if not all(hasattr(layer, name) for name in _PARAM_ATTRS):
        try:
            layer._esimd_gdn_conv_weight_fp16 = (
                conv_weight.view(conv_dim, 4)
                .detach()
                .to(dtype=torch.float16)
                .contiguous()
            )
            layer._esimd_gdn_conv_bias_fp16 = (
                torch.zeros(
                    conv_dim, dtype=torch.float16, device=conv_weight.device
                )
                if conv_bias is None
                else conv_bias.detach().to(dtype=torch.float16).contiguous()
            )
            layer._esimd_gdn_A_log_fp16 = (
                layer.A_log.detach().to(dtype=torch.float16).contiguous()
            )
            layer._esimd_gdn_dt_bias_fp16 = (
                layer.dt_bias.detach().to(dtype=torch.float16).contiguous()
            )
            layer._esimd_gdn_attn_scale = float(_HEAD_DIM**-0.5)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return False

    _xpu_w57_note(
        "gdn_spec_esimd",
        f"num_spec_tokens={num_spec_tokens} conv={conv_state.dtype} "
        f"ssm={ssm_state.dtype} qkvz={projected_qkvz.dtype}",
    )
    op(
        projected_qkvz,
        conv_state,
        layer._esimd_gdn_conv_weight_fp16,
        layer._esimd_gdn_conv_bias_fp16,
        state_indices,
        layer._esimd_gdn_A_log_fp16,
        layer._esimd_gdn_dt_bias_fp16,
        projected_ba,
        ssm_state,
        core_attn_out,
        z,
        spec_token_indices,
        num_accepted_tokens,
        num_spec_decodes,
        num_spec_tokens,
        local_k_heads,
        local_v_heads,
        _HEAD_DIM,
        _HEAD_DIM,
        layer._esimd_gdn_attn_scale,
    )
    return True

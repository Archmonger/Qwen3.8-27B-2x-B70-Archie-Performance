# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Sequence
from functools import cache

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.esimd_utils import get_esimd_op
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
    kFp8StaticTokenSym,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

from .BlockScaledMMLinearKernel import Fp8BlockScaledMMLinearKernel
from .ScaledMMLinearKernel import FP8ScaledMMLinearKernel, FP8ScaledMMLinearLayerConfig

logger = init_logger(__name__)

_ESIMD_FP8_MAX_M = 32
_ESIMD_FP8_ALIGNMENT = 128
# The BMG FP8 GEMV kernel supports the wider decode projections used by Gemma4.
# Keep an override for shape-specific A/B runs and for disabling the path.
_ESIMD_FP8_GEMV_MAX_N = int(os.environ.get("VLLM_XPU_FP8_GEMV_MAX_N", "32768"))
_DISABLE_ESIMD_FP8 = os.environ.get("VLLM_XPU_DISABLE_FP8_ESIMD", "0") == "1"
_DISABLE_ESIMD_FP8_BATCH = (
    os.environ.get("VLLM_XPU_DISABLE_FP8_ESIMD_BATCH", "0") == "1"
)

# Qwen3.6-27B TP2 regions where ESIMD beat native fp8_gemm_w8a16 in the
# synchronized microbenchmark. Untested or losing large shapes stay native.
_ESIMD_FP8_BATCH_CAPABILITIES = {
    (5120, 3072): frozenset({2, 4, 8, 12, 16, 32, 64}),
}
# Keep decode and small prefills on the existing W8A8 path. Across the Qwen3.6
# 27B and 35B-A3B TP2 projection shapes, grouped W8A16 wins consistently from
# M=128 while one shape still favors W8A8 at M=64.
_XPU_BLOCK_W8A16_MIN_M = int(os.environ.get("VLLM_XPU_BLOCK_W8A16_MIN_M", "128"))
_XPU_BLOCK_ONEDNN_MIN_M = int(os.environ.get("VLLM_XPU_BLOCK_ONEDNN_MIN_M", "8"))
_XPU_BLOCK_QWEN36_27B_ONEDNN_MIN_M = int(
    os.environ.get("VLLM_XPU_BLOCK_QWEN36_27B_ONEDNN_MIN_M", "4")
)
_XPU_BLOCK_QWEN36_27B_SHAPES = {
    (7168, 5120),
    (5120, 3072),
    (17408, 5120),
    (5120, 8704),
    (8192, 5120),
}
# W57 probe hook: an A/B arm can add (n, k) shapes to the ESIMD-blockscale
# allowlist through the environment instead of a rebuild, so an arm is one env
# var and stays reversible. Format "5120x17408,10240x5120".
for _w57_pair in filter(
    None,
    (_p.strip() for _p in os.environ.get("VLLM_XPU_BLOCK_EXTRA_SHAPES", "").split(",")),
):
    _w57_n, _w57_k = _w57_pair.split("x")
    _XPU_BLOCK_QWEN36_27B_SHAPES.add((int(_w57_n), int(_w57_k)))

# W57 probe hook: census of which branch each (path, n, k, m) actually took.
# Logs once per distinct key so an arm's engagement is provable from the engine
# log rather than inferred from tok/s -- a silently inert knob is the failure
# mode this sweep has been burned by before.
_XPU_W57_CENSUS = {}


def _xpu_w57_note(path: str, n: int, k: int, m: int) -> None:
    key = (path, n, k, m)
    count = _XPU_W57_CENSUS.get(key)
    if count is None:
        _XPU_W57_CENSUS[key] = 1
        logger.info("[w57-census] path=%s n=%d k=%d m=%d", path, n, k, m)
    else:
        _XPU_W57_CENSUS[key] = count + 1


@cache
def _load_esimd_op(is_decode: bool):
    name = "esimd_gemv_fp8_pert" if is_decode else "esimd_gemm_fp8_pert"
    return get_esimd_op(name)


@cache
def _load_esimd_block_op():
    return get_esimd_op("esimd_gemm_fp8_blockscale")


def _is_xpu_tensor(value: object) -> bool:
    return isinstance(value, torch.Tensor) and value.device.type == "xpu"


def _prepare_xpu_w8a16_weight(
    weight: torch.Tensor, out_features: int, in_features: int
) -> torch.Tensor:
    """Store W8A16 weights as contiguous checkpoint-order ``[N, K]``."""
    if weight.shape not in {(out_features, in_features), (in_features, out_features)}:
        raise ValueError(
            "XPU W8A16 FP8 expects weight shape "
            f"({out_features}, {in_features}) or ({in_features}, {out_features}), "
            f"got {tuple(weight.shape)}"
        )
    loader_transposed = weight.shape == (in_features, out_features) and (
        in_features != out_features or not weight.is_contiguous()
    )
    return (weight.t() if loader_transposed else weight).contiguous()


def _try_esimd_fp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    allow_qwen_spec_n48: bool = False,
) -> torch.Tensor | None:
    """Use ESIMD FP8 GEMV/GEMM for decode and small/measured batches."""
    if (
        _DISABLE_ESIMD_FP8
        or getattr(torch.compiler, "is_compiling", lambda: False)()
        or bias is not None
        or x.ndim != 2
        or weight.ndim != 2
        or not 1 <= x.shape[0] <= 64
        or x.shape[1] != weight.shape[1]
        or (
            weight.shape[0] % _ESIMD_FP8_ALIGNMENT != 0
            and not (allow_qwen_spec_n48 and weight.shape[0] == 48)
        )
        or weight.shape[1] % _ESIMD_FP8_ALIGNMENT != 0
        or x.dtype != torch.float16
        or weight.dtype != torch.float8_e4m3fn
        or weight_scale.dtype != torch.float32
        or weight_scale.numel() != 1
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or not weight_scale.is_contiguous()
        or not all(_is_xpu_tensor(t) for t in (x, weight, weight_scale))
        or x.device != weight.device
        or x.device != weight_scale.device
    ):
        return None

    is_decode = x.shape[0] == 1 and weight.shape[0] <= _ESIMD_FP8_GEMV_MAX_N
    if not is_decode:
        if _DISABLE_ESIMD_FP8_BATCH:
            return None
        is_small_batch_gemm = 2 <= x.shape[0] <= _ESIMD_FP8_MAX_M
        measured_region = x.shape[0] in _ESIMD_FP8_BATCH_CAPABILITIES.get(
            (weight.shape[0], weight.shape[1]), ()
        )
        if not (is_small_batch_gemm or measured_region):
            return None
    try:
        op = _load_esimd_op(is_decode)
    except (AttributeError, ImportError, OSError):
        return None

    output = torch.empty(
        (x.shape[0], weight.shape[0]), dtype=torch.float16, device=x.device
    )
    op(x, weight, weight_scale, output)
    logger.info_once(
        "XPU runtime-FP8 linear is using the ESIMD %s path.",
        "GEMV" if is_decode else "GEMM",
        scope="local",
    )
    return output


class XPUW8A8FP8LinearKernel(FP8ScaledMMLinearKernel):
    _SUPPORTED_ACT_QUANT_KEYS = {
        kFp8DynamicTensorSym,
        kFp8DynamicTokenSym,
        kFp8StaticTensorSym,
        kFp8StaticTokenSym,
    }
    _SUPPORTED_WEIGHT_QUANT_KEYS = {
        kFp8StaticChannelSym,
        kFp8StaticTensorSym,
    }

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "XPUW8A8FP8Linear only support on XPU"
        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if c.weight_quant_key not in cls._SUPPORTED_WEIGHT_QUANT_KEYS:
            return (
                False,
                "XPUW8A8FP8Linear only support per-channel and per-tensor quantization",
            )
        if c.activation_quant_key not in cls._SUPPORTED_ACT_QUANT_KEYS:
            return (
                False,
                (
                    "XPUW8A8FP8Linear only support per-tensor and per-token "
                    "activation quantization"
                ),
            )
        if c.weight_quant_key.dtype not in {torch.float8_e5m2, torch.float8_e4m3fn}:
            return False, "XPUW8A8FP8Linear only support FP8 weight dtype"
        if c.activation_quant_key.dtype not in {
            torch.float8_e5m2,
            torch.float8_e4m3fn,
        }:
            return False, "XPUW8A8FP8Linear only support FP8 activation dtype"
        return True, None

    def __init__(
        self, c: FP8ScaledMMLinearLayerConfig, layer_param_names: Sequence[str]
    ) -> None:
        super().__init__(c, layer_param_names)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Ensure weight is stored as C-contiguous [K, N] (KN layout).

        Checkpoints store weight as [N, K]; fp8_gemm requires [K, N],
        C-contiguous.  Three incoming layouts are possible:
          • [N, K] C-contiguous   ← direct checkpoint   → .t().contiguous()
          • [K, N] Fortran-order  ← fp8.py's weight.t() → .contiguous()
          • [K, N] C-contiguous   ← already correct     → no-op

        For square weights (K == N) the shape is ambiguous; contiguity is used
        as a proxy: C-contiguous ≡ checkpoint [N, K] (needs transpose);
        Fortran-order ≡ fp8.py already transposed (needs only contiguous).
        """
        K = getattr(layer, "input_size_per_partition", self.config.weight_shape[1])
        N = getattr(layer, "output_size_per_partition", self.config.weight_shape[0])
        w = layer.weight

        if w.shape not in {(K, N), (N, K)}:
            raise ValueError(
                f"XPUFP8ScaledMM expects weight shape ({K},{N}) or ({N},{K}), "
                f"but got {tuple(w.shape)}"
            )

        needs_transpose = w.shape == (N, K) if K != N else w.is_contiguous()
        layer_weight = w.t() if needs_transpose else w
        replace_parameter(layer, "weight", layer_weight)
        ws = layer.weight_scale
        if ws.numel() == 1:
            replace_parameter(layer, "weight_scale", ws.reshape(1))

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        # B is C-contiguous [K, N] from process_weights_after_loading.
        # fp8_gemm routes on scale dtype (float32) and numel:
        #   As [1]   → per-tensor  (numel==1 branch)
        #   As [M,1] → per-token   (group={1,K} branch, broadcast across K)
        #   Bs [1]   → per-tensor
        #   Bs [N]   → per-channel (mask=bit1 branch)
        # No shape manipulation needed here.
        output = torch.ops._xpu_C.fp8_gemm(A, B, out_dtype, As, Bs, bias)
        return output.view(*output_shape)


class XPUW8A16FP8LinearKernel(FP8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "XPUW8A16FP8Linear only support on XPU"
        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if c.weight_quant_key not in {kFp8StaticChannelSym, kFp8StaticTensorSym}:
            return (
                False,
                (
                    "XPUW8A16FP8Linear only support per-channel and per-tensor "
                    "quantization"
                ),
            )
        if c.weight_quant_key.dtype not in {torch.float8_e5m2, torch.float8_e4m3fn}:
            return False, "XPUW8A16FP8Linear only support FP8 weight dtype"
        return True, None

    def __init__(
        self, c: FP8ScaledMMLinearLayerConfig, layer_param_names: Sequence[str]
    ) -> None:
        assert self.can_implement(c)[0]
        assert self.is_supported()[0]
        self.config = c
        self.layer_param_names = layer_param_names

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight
        out_features, in_features = self.config.weight_shape
        weight = _prepare_xpu_w8a16_weight(weight, out_features, in_features)
        replace_parameter(layer, "weight", weight.data)

        weight_scale = layer.weight_scale.t().contiguous()
        replace_parameter(layer, "weight_scale", weight_scale.data)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight
        weight_scale = layer.weight_scale
        if getattr(layer, "_force_native_fp8", False):
            return torch.ops._xpu_C.fp8_gemm_w8a16(
                x, weight.t(), weight_scale, bias
            )
        esimd_output = _try_esimd_fp8_linear(
            x,
            weight,
            weight_scale,
            bias,
            allow_qwen_spec_n48=getattr(
                layer, "_allow_qwen_spec_esimd_fp8_n48", False
            ),
        )
        if esimd_output is not None:
            return esimd_output
        return torch.ops._xpu_C.fp8_gemm_w8a16(x, weight.t(), weight_scale, bias)

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        pass


class XPUFp8BlockScaledMMKernel(Fp8BlockScaledMMLinearKernel):
    # The optimized XPU paths consume the original fp16 activation and exact
    # checkpoint 128x128 scales. Avoid the generic W8A8 input-quant step.
    apply_input_quant = False

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "XPUFp8BlockScaledMM only support on XPU"
        return True, None

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        params = self._get_layer_params(layer, **kwargs)
        weight = params.weight
        weight_scale = (
            params.weight_scale
            if params.weight_scale_inv is None
            else params.weight_scale_inv
        )
        if weight_scale is None:
            return super().apply_weights(layer, x, bias, **kwargs)
        output = self.apply_block_scaled_mm(
            A=x.view(-1, x.shape[-1]),
            B=weight,
            As=weight_scale,
            Bs=weight_scale,
        )
        if bias is not None:
            output = output + bias
        return output.to(dtype=self.config.out_dtype).view(
            *x.shape[:-1], weight.shape[0]
        )

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        block_n, block_k = map(int, self.weight_group_shape)
        a16 = (
            A
            if A.dtype == torch.float16 and A.is_contiguous()
            else A.to(torch.float16).contiguous()
        )
        m = a16.shape[0]
        n, k = B.shape
        scale = (
            Bs
            if Bs.dtype == torch.float32 and Bs.is_contiguous()
            else Bs.to(torch.float32).contiguous()
        )
        expected_scale_shape = (
            (n + block_n - 1) // block_n,
            (k + block_k - 1) // block_k,
        )
        if (
            B.dtype != torch.float8_e4m3fn
            or block_n != 128
            or block_k != 128
            or k % block_k != 0
            or tuple(scale.shape) != expected_scale_shape
        ):
            _xpu_w57_note("fp8_gemm_fallback", n, k, m)
            return torch.ops._xpu_C.fp8_gemm(
                A,
                B.t(),
                self.config.out_dtype,
                As,
                Bs.t().contiguous(),
                torch.Tensor(),
            )

        use_onednn = m >= _XPU_BLOCK_ONEDNN_MIN_M or (
            m >= _XPU_BLOCK_QWEN36_27B_ONEDNN_MIN_M
            and (n, k) in _XPU_BLOCK_QWEN36_27B_SHAPES
        )
        if (
            use_onednn
            and n % block_n == 0
            and hasattr(torch.ops._xpu_C, "fp8_gemm_w8a16")
        ):
            _xpu_w57_note("onednn_w8a16", n, k, m)
            return torch.ops._xpu_C.fp8_gemm_w8a16(
                a16,
                B.t(),
                scale.t().contiguous(),
                None,
            )

        try:
            op = _load_esimd_block_op()
        except (AttributeError, ImportError, OSError):
            _xpu_w57_note("fp8_gemm_esimd_unavailable", n, k, m)
            return torch.ops._xpu_C.fp8_gemm(
                A,
                B.t(),
                self.config.out_dtype,
                As,
                Bs.t().contiguous(),
                torch.Tensor(),
            )
        output = torch.empty(m, n, dtype=torch.float16, device=a16.device)
        _xpu_w57_note("esimd_blockscale", n, k, m)
        op(a16, B, scale, output, block_n, block_k)
        return output

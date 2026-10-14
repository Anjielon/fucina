"""2026-10-02 fast TT6 — GDN causal convolution WITHOUT transposes (E2E_CONV_DIRECT=1; off = the usual transformers).

Why (GPU profile of 2026-10-02, layers 0 and 3 of R2B3m, TT6 configuration, 8192 tokens): transformers passes
`causal_conv1d_fn` the TRANSPOSED view (B, C, T) of `in_proj_qkv(h)` (B, T, 8192) fp32; F.conv1d (MIOpen) wants it
contiguous and copies it, and copying a transposed 2 x 8192 x 4096 fp32 (256 MiB) runs at ~0.85 GB/s: 4 copies of
~300 ms + 1 of ~430 ms per GDN layer (forward, recomputation, backward) = ~1.6 s per 8192 tokens = ~0.2 ms/token per
layer, x 30 GDN layers = ~6 ms/token, a third of the TT5 step.

Here the same convolution is done in the contiguous (B, T, C) layout, with a Triton kernel (CPU/tests: torch):
    y[b, t, c] = fp32( sum_k fp64(w[c, k]) * fp64(x[b, t - (K-1) + k, c]) )          (zero outside the sequence)
    dx[b, s, c] = fp32( sum_k fp64(w[c, k]) * fp64(g[b, s + (K-1) - k, c]) )
BIT-EXACT with MIOpen on the GPU (probe `tools/sonda_conv_miopen.py`, 2026-10-02: MIOpen accumulates in fp64 — kernel
`naive_conv_ab_nonpacked_*_float_double_float` — and over 67 M elements, forward and backward: 0 different; fp32
accumulation gets 47 % of them wrong). fp32 x fp32 products are exact in fp64 (48 bits): the only rounding is that of
the 4-term sum, identical to MIOpen's.
If the convolution weight requires a gradient (--allena-f32 with ssm_conv1d) we fall back to the transformers path
(MIOpen's dL/dw is a reduction over B x T in an order not reproduced here): no change in numbers, ever.
The output is a CONTIGUOUS (B, C, T) like that of F.conv1d (2026-10-02 evening: with the output in (B, T, C) the
values were the same but the different strides made torch.compile compile a different GDN preparation, with a
different order of the sums in the backward: dL/dtheta differing by 4.6e-5 relative over the 40 layers). The kernel
reads and writes with arbitrary strides: no copy either on the way in or on the way out."""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _TRITON = True
except Exception:  # pragma: no cover
    _TRITON = False

COUNTS = {"direct": 0, "fallback": 0}

if _TRITON:
    @triton.jit
    def _k_conv(x_ptr, sxb, sxt, sxc, w_ptr, y_ptr, syb, syt, syc, T, C, K: tl.constexpr, DIRECTION: tl.constexpr,
                BT: tl.constexpr, BC: tl.constexpr):
        """DIRECTION 0: y[t] = sum_k w[k] x[t-(K-1)+k] (forward); DIRECTION 1: y[s] = sum_k w[k] x[s+(K-1)-k] (dX).
        x, y with arbitrary strides (b, t, c); w (C, K) contiguous. fp64 accumulation, fp32 output."""
        b = tl.program_id(0).to(tl.int64)
        t = tl.program_id(1) * BT + tl.arange(0, BT)
        c = tl.program_id(2) * BC + tl.arange(0, BC)
        mc = c < C
        c64 = c.to(tl.int64)
        acc = tl.zeros((BT, BC), dtype=tl.float64)
        for k in tl.static_range(K):
            if DIRECTION == 0:
                ts = t - (K - 1) + k
            else:
                ts = t + (K - 1) - k
            m = (ts >= 0) & (ts < T)
            xv = tl.load(x_ptr + b * sxb + ts[:, None].to(tl.int64) * sxt + c64[None, :] * sxc,
                         mask=m[:, None] & mc[None, :], other=0.0)
            wv = tl.load(w_ptr + c * K + k, mask=mc, other=0.0)
            acc += xv.to(tl.float64) * wv.to(tl.float64)[None, :]
        mt = t < T
        tl.store(y_ptr + b * syb + t[:, None].to(tl.int64) * syt + c64[None, :] * syc, acc.to(tl.float32),
                 mask=mt[:, None] & mc[None, :])


if _TRITON:
    @triton.jit
    def _k_conv32(x_ptr, sxb, sxt, sxc, w_ptr, y_ptr, syb, syt, syc, T, C, K: tl.constexpr, DIRECTION: tl.constexpr,
                  BT: tl.constexpr, BC: tl.constexpr):
        """Same convolution as _k_conv WITHOUT fp64 (2026-10-09 E2E_KQAT=conv32; gfx1151 fp64 is ~1/32 of fp32: the fp64 kernel
        ran at ~6 GB/s, ~0.4 ms/token over 30 GDN layers). Compensated dot product (Ogita-Rump-Oishi Dot2): every
        product is split exactly (p = fl(w x), e = fma(w, x, -p)), the running sum by TwoSum, the errors summed apart;
        the result fl(s + c) is the exact 4-term sum rounded to fp32 except when it lies within ~2^-46 of a rounding
        midpoint: the same value as the fp64 accumulation of MIOpen / _k_conv but for a fraction ~2^-22 of the elements
        (1 ulp fp32), counted by the GPU test against _k_conv."""
        b = tl.program_id(0).to(tl.int64)
        t = tl.program_id(1) * BT + tl.arange(0, BT)
        c = tl.program_id(2) * BC + tl.arange(0, BC)
        mc = c < C
        c64 = c.to(tl.int64)
        s = tl.zeros((BT, BC), dtype=tl.float32)
        cc = tl.zeros((BT, BC), dtype=tl.float32)
        for k in tl.static_range(K):
            if DIRECTION == 0:
                ts = t - (K - 1) + k
            else:
                ts = t + (K - 1) - k
            m = (ts >= 0) & (ts < T)
            xv = tl.load(x_ptr + b * sxb + ts[:, None].to(tl.int64) * sxt + c64[None, :] * sxc,
                         mask=m[:, None] & mc[None, :], other=0.0)
            wv = tl.load(w_ptr + c * K + k, mask=mc, other=0.0)[None, :]
            p = xv * wv
            e = tl.fma(xv, wv, -p)
            sn = s + p
            bb = sn - s
            err = (s - (sn - bb)) + (p - bb)
            s = sn
            cc = cc + (err + e)
        mt = t < T
        tl.store(y_ptr + b * syb + t[:, None].to(tl.int64) * syt + c64[None, :] * syc, s + cc,
                 mask=mt[:, None] & mc[None, :])


def _conv_torch(x: torch.Tensor, w: torch.Tensor, direction: int) -> torch.Tensor:
    """Same arithmetic as the kernel, in torch (CPU and tests): fp64, k in increasing order, then fp32. x (B, T, C)."""
    B, T, C = x.shape
    K = w.shape[1]
    x64 = x.to(torch.float64)
    w64 = w.to(torch.float64)
    xp = F.pad(x64, (0, 0, K - 1, 0)) if direction == 0 else F.pad(x64, (0, 0, 0, K - 1))
    y = torch.zeros(B, T, C, dtype=torch.float64, device=x.device)
    for k in range(K):
        sh = k if direction == 0 else (K - 1 - k)
        y = y + xp[:, sh:sh + T] * w64[:, k]
    return y.to(torch.float32)


def conv_btc(x: torch.Tensor, w: torch.Tensor, direction: int = 0, output_bct: bool = False) -> torch.Tensor:
    """x: view (B, T, C) with arbitrary strides, fp32; w (C, K). -> (B, T, C) fp32: contiguous, or (output_bct)
    the (B, T, C) view of a CONTIGUOUS (B, C, T) tensor (the output layout of F.conv1d: downstream code sees the
    usual strides). Triton on GPU, torch elsewhere."""
    w = w.contiguous()
    B, T, C = x.shape
    if _TRITON and x.is_cuda and os.environ.get("E2E_CONV_DIRECT_TORCH", "") != "1":
        if output_bct:
            y = torch.empty(B, C, T, dtype=torch.float32, device=x.device).transpose(1, 2)
        else:
            y = torch.empty(B, T, C, dtype=torch.float32, device=x.device)
        BT, BC = 32, 128
        from fucina.kernels import qat as _KQ
        if _KQ.active("conv32"):                     # no mul+add contraction: the error-free transforms need it
            _k_conv32[(B, triton.cdiv(T, BT), triton.cdiv(C, BC))](x, x.stride(0), x.stride(1), x.stride(2), w,
                                                                  y, y.stride(0), y.stride(1), y.stride(2), T, C,
                                                                  w.shape[1], direction, BT, BC, num_warps=4,
                                                                  enable_fp_fusion=False)
        else:
            _k_conv[(B, triton.cdiv(T, BT), triton.cdiv(C, BC))](x, x.stride(0), x.stride(1), x.stride(2), w,
                                                                y, y.stride(0), y.stride(1), y.stride(2), T, C,
                                                                w.shape[1], direction, BT, BC, num_warps=4)
        return y
    y = _conv_torch(x, w, direction)
    return y.transpose(1, 2).contiguous().transpose(1, 2) if output_bct else y


class CausalConvBTC(torch.autograd.Function):
    """Input: view (B, T, C) (any strides). Output: (B, T, C) view of a contiguous (B, C, T), like F.conv1d.
    dX: contiguous (B, T, C) (the values do not depend on the layout)."""
    @staticmethod
    def forward(ctx, x_btc, w, output_bct=True):
        ctx.save_for_backward(w)
        return conv_btc(x_btc, w, 0, output_bct=output_bct)

    @staticmethod
    def backward(ctx, g):
        (w,) = ctx.saved_tensors
        dx = conv_btc(g, w, 1)
        if VERIFY is not None:                   # diagnosis only: comparison with MIOpen's dX on the same g
            VERIFY.append(_verify_dx(g, w, dx))
        return dx, None, None


VERIFY: Optional[list] = None


def _verify_dx(g, w, dx) -> dict:
    B, T, C = g.shape
    K = w.shape[1]
    xt = torch.zeros(B, C, T, device=g.device, dtype=g.dtype, requires_grad=True)
    with torch.enable_grad():
        y = F.conv1d(xt, w.unsqueeze(1), None, padding=K - 1, groups=C)[:, :, :T]
        (rif,) = torch.autograd.grad(y, xt, g.transpose(1, 2).contiguous())
    rif = rif.transpose(1, 2)
    d = rif != dx
    return {"different": int(d.sum()), "n": dx.numel(), "max_abs": float((rif - dx).abs().max()),
            "g_contiguous": bool(g.is_contiguous()), "g_strides": tuple(g.stride())}


def causal_conv1d_fn_direct(hidden_states: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
                             activation: Optional[str] = None, **kwargs):
    """Same signature and same numbers as `transformers...causal_conv1d_fn` (fp32, bias None, weight without gradient)."""
    ok = (bias is None and hidden_states.dtype == torch.float32 and weight.dtype == torch.float32
          and hidden_states.dim() == 3 and weight.dim() == 2 and not weight.requires_grad and not kwargs
          and (hidden_states.is_cuda or os.environ.get("E2E_CONV_DIRECT_CPU", "") == "1"))
    # on the CPU F.conv1d accumulates in fp32 (it is not MIOpen): there the usual path stays, except in tests (..._CPU=1)
    if not ok:
        COUNTS["fallback"] += 1
        return _ORIGINAL(hidden_states, weight, bias, activation=activation, **kwargs)
    COUNTS["direct"] += 1
    x_btc = hidden_states.transpose(1, 2)            # (B, T, C): read with its own strides, no copy
    # E2E_CONV_DIRECT=1: CONTIGUOUS (B, C, T) output like F.conv1d -> everything downstream BIT-EXACT.
    # E2E_CONV_DIRECT=btc: contiguous (B, T, C) output -> no copies afterwards either (faster), same h values,
    # but the compiled GDN (torch.compile, different strides) sums in a different order in the backward: dL/dtheta
    # differs at rounding level (4.6e-5 relative over the 40 layers, gate of 2026-10-02). The coordinator decides.
    bct = os.environ.get("E2E_CONV_DIRECT", "1").strip() != "btc"
    y = CausalConvBTC.apply(x_btc, weight, bct).transpose(1, 2)
    if activation is not None:
        from fucina.kernels import qat as _KQ               # 2026-10-09 E2E_KQAT=silu: silu_backward on the same layout
        if activation == "silu" and _KQ.active("silu"):
            y = SiluSameLayout.apply(y)
        else:
            from transformers.activations import ACT2FN
            y = ACT2FN[activation](y)
    return y.to(hidden_states.dtype)


class SiluSameLayout(torch.autograd.Function):
    """F.silu with a backward that first brings the incoming gradient into the memory layout of the saved input.
    2026-10-09 profile (DI1 config, E2E_CONV_DIRECT=1): aten::silu_backward on [2, 8192, 4096] with the gradient laid out
    (B, T, C) and the input (B, C, T) ran the non-contiguous `elementwise_kernel_manual_unroll` at ~2 GB/s: 400 ms
    per GDN layer and micro-batch (~1.5 ms/token over 30 layers). Here: one transposing copy of the gradient, then the
    SAME aten::silu_backward on two tensors with the same strides (vectorized kernel): the same fp32 operations on the
    same values, element by element -> the same result bit for bit (the layout does not enter the arithmetic)."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return F.silu(x)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        if g.stride() != x.stride():
            # 2026-10-09 profile: torch's copy_ between the two layouts ran at ~1.8 GB/s (290 ms per call): a tiled
            # transposing copy in Triton (pure data movement: exact) instead
            g = copy_layout(g, x)
        return torch.ops.aten.silu_backward(g, x)


if _TRITON:
    @triton.jit
    def _k_transpose(src, ss_b, ss_t, ss_c, dst, sd_b, sd_t, sd_c, T, C, BT: tl.constexpr, BC: tl.constexpr):
        """dst[b, t, c] = src[b, t, c] for two (B, T, C) views with different strides, tile by tile (one side is
        contiguous along c, the other along t: the tile is transposed in registers so that both accesses coalesce)."""
        b = tl.program_id(0).to(tl.int64)
        t = tl.program_id(1) * BT + tl.arange(0, BT)
        c = tl.program_id(2) * BC + tl.arange(0, BC)
        m = (t < T)[:, None] & (c < C)[None, :]
        v = tl.load(src + b * ss_b + t[:, None].to(tl.int64) * ss_t + c[None, :].to(tl.int64) * ss_c, mask=m)
        tl.store(dst + b * sd_b + t[:, None].to(tl.int64) * sd_t + c[None, :].to(tl.int64) * sd_c, v, mask=m)


def copy_layout(g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """A copy of g (B, T, C) laid out with the strides of x (exact: data movement only)."""
    out = torch.empty_strided(x.shape, x.stride(), dtype=g.dtype, device=g.device)
    if not (_TRITON and g.is_cuda and g.dim() == 3):
        return out.copy_(g)
    B, T, C = g.shape
    BT, BC = 64, 64
    _k_transpose[(B, triton.cdiv(T, BT), triton.cdiv(C, BC))](g, g.stride(0), g.stride(1), g.stride(2),
                                                             out, out.stride(0), out.stride(1), out.stride(2),
                                                             T, C, BT, BC, num_warps=4)
    return out


_ORIGINAL = None


def install(module=None) -> None:
    global _ORIGINAL
    if module is None:
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as module
    if module.causal_conv1d_fn is causal_conv1d_fn_direct:
        return
    _ORIGINAL = module.causal_conv1d_fn
    module.causal_conv1d_fn = causal_conv1d_fn_direct


def uninstall(module=None) -> None:
    if module is None:
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as module
    if _ORIGINAL is not None and module.causal_conv1d_fn is causal_conv1d_fn_direct:
        module.causal_conv1d_fn = _ORIGINAL

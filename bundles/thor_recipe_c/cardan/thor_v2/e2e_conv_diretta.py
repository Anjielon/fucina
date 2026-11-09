"""2/10 TT6-veloce — convoluzione causale del GDN SENZA trasposte (E2E_CONV_DIRETTA=1; spento = transformers di sempre).

Perche' (profilo GPU del 2/10, strati 0 e 3 di R2B3m, configurazione di TT6, 8192 token): transformers passa a
`causal_conv1d_fn` la vista TRASPOSTA (B, C, T) di `in_proj_qkv(h)` (B, T, 8192) fp32; F.conv1d (MIOpen) la vuole
contigua e la copia, e la copia di una trasposta 2 x 8192 x 4096 fp32 (256 MiB) gira a ~0,85 GB/s: 4 copie da ~300 ms
+ 1 da ~430 ms per strato GDN (avanti, ricalcolo, indietro) = ~1,6 s ogni 8192 token = ~0,2 ms/token per strato,
x 30 strati GDN = ~6 ms/token, un terzo del passo di TT5.

Qui la stessa convoluzione si fa nella disposizione (B, T, C) contigua, con un kernel Triton (CPU/prove: torch):
    y[b, t, c] = fp32( sum_k fp64(w[c, k]) * fp64(x[b, t - (K-1) + k, c]) )          (zero fuori dalla sequenza)
    dx[b, s, c] = fp32( sum_k fp64(w[c, k]) * fp64(g[b, s + (K-1) - k, c]) )
AL BIT come MIOpen sulla GPU (sonda `strumenti/sonda_conv_miopen.py`, 2/10: MIOpen accumula in fp64 — kernel
`naive_conv_ab_nonpacked_*_float_double_float` — e su 67 M elementi avanti e indietro: 0 diversi; l'accumulo fp32
ne sbaglia il 47 %). I prodotti fp32 x fp32 sono esatti in fp64 (48 bit): l'unico arrotondamento e' quello della
somma di 4 termini, identico a quello di MIOpen.
Se il peso della convoluzione chiede gradiente (--allena-f32 con ssm_conv1d) si torna alla via di transformers (il
dL/dw di MIOpen e' una riduzione su B x T in un ordine che qui non si riproduce): nessun cambio di numeri, mai.
L'uscita e' un (B, C, T) CONTIGUO come quella di F.conv1d (2/10 sera: con l'uscita in (B, T, C) i valori erano
gli stessi ma i passi diversi facevano compilare a torch.compile un'altra preparazione del GDN, con un altro ordine
delle somme nell'indietro: dL/dtheta diverso nel 4,6e-5 relativo sui 40 strati). Il kernel legge e scrive con passi
qualsiasi: nessuna copia ne' all'andata ne' al ritorno."""
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

CONTI = {"diretta": 0, "ripiego": 0}

if _TRITON:
    @triton.jit
    def _k_conv(x_ptr, sxb, sxt, sxc, w_ptr, y_ptr, syb, syt, syc, T, C, K: tl.constexpr, VERSO: tl.constexpr,
                BT: tl.constexpr, BC: tl.constexpr):
        """VERSO 0: y[t] = sum_k w[k] x[t-(K-1)+k] (avanti); VERSO 1: y[s] = sum_k w[k] x[s+(K-1)-k] (dX).
        x, y con passi qualsiasi (b, t, c); w (C, K) contiguo. Accumulo fp64, uscita fp32."""
        b = tl.program_id(0).to(tl.int64)
        t = tl.program_id(1) * BT + tl.arange(0, BT)
        c = tl.program_id(2) * BC + tl.arange(0, BC)
        mc = c < C
        c64 = c.to(tl.int64)
        acc = tl.zeros((BT, BC), dtype=tl.float64)
        for k in tl.static_range(K):
            if VERSO == 0:
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
    def _k_conv32(x_ptr, sxb, sxt, sxc, w_ptr, y_ptr, syb, syt, syc, T, C, K: tl.constexpr, VERSO: tl.constexpr,
                  BT: tl.constexpr, BC: tl.constexpr):
        """Same convolution as _k_conv WITHOUT fp64 (9/10 E2E_KQAT=conv32; gfx1151 fp64 is ~1/32 of fp32: the fp64 kernel
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
            if VERSO == 0:
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


def _conv_torch(x: torch.Tensor, w: torch.Tensor, verso: int) -> torch.Tensor:
    """Stessa aritmetica del kernel, in torch (CPU e prove): fp64, k in ordine crescente, poi fp32. x (B, T, C)."""
    B, T, C = x.shape
    K = w.shape[1]
    x64 = x.to(torch.float64)
    w64 = w.to(torch.float64)
    xp = F.pad(x64, (0, 0, K - 1, 0)) if verso == 0 else F.pad(x64, (0, 0, 0, K - 1))
    y = torch.zeros(B, T, C, dtype=torch.float64, device=x.device)
    for k in range(K):
        sh = k if verso == 0 else (K - 1 - k)
        y = y + xp[:, sh:sh + T] * w64[:, k]
    return y.to(torch.float32)


def conv_btc(x: torch.Tensor, w: torch.Tensor, verso: int = 0, uscita_bct: bool = False) -> torch.Tensor:
    """x: vista (B, T, C) con passi qualsiasi, fp32; w (C, K). -> (B, T, C) fp32: contigua, oppure (uscita_bct)
    la vista (B, T, C) di un tensore (B, C, T) CONTIGUO (la disposizione dell'uscita di F.conv1d: chi segue vede
    gli stessi passi di sempre). Triton su GPU, torch altrove."""
    w = w.contiguous()
    B, T, C = x.shape
    if _TRITON and x.is_cuda and os.environ.get("E2E_CONV_DIRETTA_TORCH", "") != "1":
        if uscita_bct:
            y = torch.empty(B, C, T, dtype=torch.float32, device=x.device).transpose(1, 2)
        else:
            y = torch.empty(B, T, C, dtype=torch.float32, device=x.device)
        BT, BC = 32, 128
        import e2e_kernel_qat as _KQ
        if _KQ.attivo("conv32"):                     # no mul+add contraction: the error-free transforms need it
            _k_conv32[(B, triton.cdiv(T, BT), triton.cdiv(C, BC))](x, x.stride(0), x.stride(1), x.stride(2), w,
                                                                  y, y.stride(0), y.stride(1), y.stride(2), T, C,
                                                                  w.shape[1], verso, BT, BC, num_warps=4,
                                                                  enable_fp_fusion=False)
        else:
            _k_conv[(B, triton.cdiv(T, BT), triton.cdiv(C, BC))](x, x.stride(0), x.stride(1), x.stride(2), w,
                                                                y, y.stride(0), y.stride(1), y.stride(2), T, C,
                                                                w.shape[1], verso, BT, BC, num_warps=4)
        return y
    y = _conv_torch(x, w, verso)
    return y.transpose(1, 2).contiguous().transpose(1, 2) if uscita_bct else y


class ConvCausaleBTC(torch.autograd.Function):
    """Ingresso: vista (B, T, C) (qualsiasi passo). Uscita: vista (B, T, C) di un (B, C, T) contiguo, come
    F.conv1d. dX: (B, T, C) contiguo (i valori non dipendono dalla disposizione)."""
    @staticmethod
    def forward(ctx, x_btc, w, uscita_bct=True):
        ctx.save_for_backward(w)
        return conv_btc(x_btc, w, 0, uscita_bct=uscita_bct)

    @staticmethod
    def backward(ctx, g):
        (w,) = ctx.saved_tensors
        dx = conv_btc(g, w, 1)
        if VERIFICA is not None:                   # solo diagnosi: confronto col dX di MIOpen sullo stesso g
            VERIFICA.append(_verifica_dx(g, w, dx))
        return dx, None, None


VERIFICA: Optional[list] = None


def _verifica_dx(g, w, dx) -> dict:
    B, T, C = g.shape
    K = w.shape[1]
    xt = torch.zeros(B, C, T, device=g.device, dtype=g.dtype, requires_grad=True)
    with torch.enable_grad():
        y = F.conv1d(xt, w.unsqueeze(1), None, padding=K - 1, groups=C)[:, :, :T]
        (rif,) = torch.autograd.grad(y, xt, g.transpose(1, 2).contiguous())
    rif = rif.transpose(1, 2)
    d = rif != dx
    return {"diversi": int(d.sum()), "n": dx.numel(), "max_abs": float((rif - dx).abs().max()),
            "g_contigua": bool(g.is_contiguous()), "g_strides": tuple(g.stride())}


def causal_conv1d_fn_diretta(hidden_states: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
                             activation: Optional[str] = None, **kwargs):
    """Stessa firma e stessi numeri di `transformers...causal_conv1d_fn` (fp32, bias None, peso senza gradiente)."""
    ok = (bias is None and hidden_states.dtype == torch.float32 and weight.dtype == torch.float32
          and hidden_states.dim() == 3 and weight.dim() == 2 and not weight.requires_grad and not kwargs
          and (hidden_states.is_cuda or os.environ.get("E2E_CONV_DIRETTA_CPU", "") == "1"))
    # sulla CPU F.conv1d accumula in fp32 (non e' MIOpen): li' resta la via di sempre, salvo le prove (..._CPU=1)
    if not ok:
        CONTI["ripiego"] += 1
        return _ORIGINALE(hidden_states, weight, bias, activation=activation, **kwargs)
    CONTI["diretta"] += 1
    x_btc = hidden_states.transpose(1, 2)            # (B, T, C): letta coi suoi passi, nessuna copia
    # E2E_CONV_DIRETTA=1: uscita (B, C, T) CONTIGUA come F.conv1d -> tutto il resto identico AL BIT.
    # E2E_CONV_DIRETTA=btc: uscita (B, T, C) contigua -> niente copie neanche dopo (piu' veloce), stessi valori di h,
    # ma il GDN compilato (torch.compile, passi diversi) somma in un altro ordine nell'indietro: dL/dtheta diverso
    # all'arrotondamento (4,6e-5 relativo sui 40 strati, cancello del 2/10). Decisione di chi coordina.
    bct = os.environ.get("E2E_CONV_DIRETTA", "1").strip() != "btc"
    y = ConvCausaleBTC.apply(x_btc, weight, bct).transpose(1, 2)
    if activation is not None:
        import e2e_kernel_qat as _KQ               # 9/10 E2E_KQAT=silu: silu_backward sulla stessa disposizione
        if activation == "silu" and _KQ.attivo("silu"):
            y = SiluStessaDisposizione.apply(y)
        else:
            from transformers.activations import ACT2FN
            y = ACT2FN[activation](y)
    return y.to(hidden_states.dtype)


class SiluStessaDisposizione(torch.autograd.Function):
    """F.silu with a backward that first brings the incoming gradient into the memory layout of the saved input.
    9/10 profile (DI1 config, E2E_CONV_DIRETTA=1): aten::silu_backward on [2, 8192, 4096] with the gradient laid out
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
            # 9/10 profile: torch's copy_ between the two layouts ran at ~1.8 GB/s (290 ms per call): a tiled
            # transposing copy in Triton (pure data movement: exact) instead
            g = copia_disposizione(g, x)
        return torch.ops.aten.silu_backward(g, x)


if _TRITON:
    @triton.jit
    def _k_trasponi(src, ss_b, ss_t, ss_c, dst, sd_b, sd_t, sd_c, T, C, BT: tl.constexpr, BC: tl.constexpr):
        """dst[b, t, c] = src[b, t, c] for two (B, T, C) views with different strides, tile by tile (one side is
        contiguous along c, the other along t: the tile is transposed in registers so that both accesses coalesce)."""
        b = tl.program_id(0).to(tl.int64)
        t = tl.program_id(1) * BT + tl.arange(0, BT)
        c = tl.program_id(2) * BC + tl.arange(0, BC)
        m = (t < T)[:, None] & (c < C)[None, :]
        v = tl.load(src + b * ss_b + t[:, None].to(tl.int64) * ss_t + c[None, :].to(tl.int64) * ss_c, mask=m)
        tl.store(dst + b * sd_b + t[:, None].to(tl.int64) * sd_t + c[None, :].to(tl.int64) * sd_c, v, mask=m)


def copia_disposizione(g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """A copy of g (B, T, C) laid out with the strides of x (exact: data movement only)."""
    out = torch.empty_strided(x.shape, x.stride(), dtype=g.dtype, device=g.device)
    if not (_TRITON and g.is_cuda and g.dim() == 3):
        return out.copy_(g)
    B, T, C = g.shape
    BT, BC = 64, 64
    _k_trasponi[(B, triton.cdiv(T, BT), triton.cdiv(C, BC))](g, g.stride(0), g.stride(1), g.stride(2),
                                                             out, out.stride(0), out.stride(1), out.stride(2),
                                                             T, C, BT, BC, num_warps=4)
    return out


_ORIGINALE = None


def installa(modulo=None) -> None:
    global _ORIGINALE
    if modulo is None:
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as modulo
    if modulo.causal_conv1d_fn is causal_conv1d_fn_diretta:
        return
    _ORIGINALE = modulo.causal_conv1d_fn
    modulo.causal_conv1d_fn = causal_conv1d_fn_diretta


def disinstalla(modulo=None) -> None:
    if modulo is None:
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as modulo
    if _ORIGINALE is not None and modulo.causal_conv1d_fn is causal_conv1d_fn_diretta:
        modulo.causal_conv1d_fn = _ORIGINALE

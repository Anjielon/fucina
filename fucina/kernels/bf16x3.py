#!/usr/bin/env python3
"""bf16x3 — fp32 products computed with three bf16 GEMMs (2026-09-27, recipe «B3» of `fucina.student.student` and
`--head-kl bf16x3` of the training driver).

Why: on the run's GPU (torch 2.11 rocm 7.13, gfx1151) the fp32 GEMM reaches 1.3-2.2 TFLOPS and the bf16 one 20-27
(measured 2026-09-27). In recipe B the NON-expert projections (attention, GDN in_proj/out_proj, shared expert) and
the KL head (student logits, dL/dh) are fp32 GEMMs: ~56% of the step's FLOPs.

Scheme (Henry, Tang, Heinecke, arXiv 1904.06376; JAX `Precision.HIGH` / `DotAlgorithmPreset.BF16_BF16_F32_X3`):
    x = hi + lo,  hi = bf16(x),  lo = bf16(x - hi)          (x - hi is exact in fp32)
    a b ~= hi_a hi_b + hi_a lo_b + lo_a hi_b                 (lo_a lo_b ~ 2^-16 |a b| is dropped)
Each bf16 x bf16 product is EXACT in fp32 (8 + 8 mantissa bits <= 24); the sums are fp32. The two small terms are
summed first, then the large one. Expected error ~1e-5 relative (trial X1 of `e2e_test_bf16x3.py`, with the
measured number), versus ~1e-6/1e-7 for the fp32 GEMM.

bf16 -> fp32 GEMM path (`via_mm`, probed once per device):
- "triton" (2026-09-27 afternoon, the GPU path after the gate rejected at 11:04): ONE Triton kernel computes the three
  products (`_k_x3`). The fp32 A is split IN REGISTERS (hi/lo with round-to-nearest-even, the same bits as
  `decompose`), the three bf16 `tl.dot` go to the WMMA units, and each K tile (BK) is accumulated in its OWN
  accumulator, added to the total with an fp32 vector addition (rounded to nearest even). Why this way (measurements
  of 2026-09-27 on gfx1151, techniques note §bf16x3):
    * `mm(out_dtype=float32)` is EXACT but does not use the WMMA units: 2.4 TFLOPS like the fp32 GEMM (5.4 with
      hipBLASLt), versus 16-34 for bf16 with bf16 output. Three of those = 2-3 times SLOWER than fp32: the gate
      saw it (x1.06-1.88).
    * accumulation INSIDE the WMMA (v_wmma_f32_16x16x16_bf16, and `acc += tl.dot(...)` which Triton fuses into it)
      TRUNCATES toward zero: a sum of 16384 positive products drifts by -2.2e-5 relative (fp32: 8e-9). With the
      tile in its own accumulator (`z = acc * 0.0` as the initial accumulator: Triton does not recognize it as
      zero and does not fuse the addition) the drift disappears: 2.6e-9.
- "out_dtype": `torch.mm(a, b, out_dtype=torch.float32)` and `torch.addmm(c, a, b, out_dtype=torch.float32)`
  (aten::mm.dtype / addmm.dtype, CUDA/ROCm only in torch 2.11): bf16 inputs, fp32 accumulation and output; the sum
  of the three terms happens in the GEMM epilogue (beta = 1), no extra passes over memory.
- "output_bf16" (ONLY if the previous one is missing): bf16 `torch.mm` (fp32 accumulation inside hipBLASLt, output
  rounded to bf16) converted to fp32 and summed in fp32. The output rounding (2^-9 relative of the large term)
  dominates: error ~1e-3, like a plain bf16 GEMM (trial X1, stated number): bf16x3 on this path is NOT useful.
  Recorded in `state()`; the B3 gate writes it out.
- "emulate" (CPU, tests): a.float() @ b.float() — bf16 values are exact in fp32 and so are the products: the same
  arithmetic as "out_dtype" except for the order of the fp32 sums.
`E2E_BF16X3_VIA` = triton | out_dtype | output_bf16 | emulate forces the path (tests).

`BF16x3Linear`: replaces a FROZEN `nn.Linear`. The weight is split ONCE (hi, lo in bf16: same memory as the fp32
weight, which is freed). Autograd: forward y = x W^T in bf16x3; backward dX = dY W with the same split of dY; no dW
(the non-expert weights are frozen) and no x saved for the backward.

`K2Linear` (2026-09-27, ONLY for the "noise" arm of the B3 gate, recipe `Bk2_ind32`): the same projection in fp32
but with the sum over K split into two halves (y = x_1 W_1^T + x_2 W_2^T; dX likewise). fp32 arithmetic as in B,
only a different order of the sums: how much B itself moves under a different fp32 rounding (the "floor").
"""
from __future__ import annotations

import os
from typing import Optional

import torch

VIA_FORCED = os.getenv("E2E_BF16X3_VIA", "")
_VIA: dict = {}          # device -> {"via": ..., "addmm": bool, ...}


def decompose(x: torch.Tensor) -> tuple:
    """x fp32 -> (hi, lo) bf16 with x ~= hi + lo (error <= about 2^-17 |x|)."""
    if x.dtype != torch.float32:
        raise TypeError(f"bf16x3: input {x.dtype}, expected float32")
    hi = x.to(torch.bfloat16)
    lo = (x - hi.float()).to(torch.bfloat16)
    return hi, lo


# ═══════════════════════════════════ "triton" path ═══════════════════════════════════
try:  # Triton is in the forge venv; without it the triton path is missing (the trial discards it, fallback)
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None

if triton is not None:
    @triton.jit
    def _k_x3(a_ptr, sa_m, sa_k, bh_ptr, bl_ptr, sb_k, sb_n, c_ptr, sc_s, sc_m, sc_n, M, N, K, KS,
              A_FP32: tl.constexpr, HA_LO: tl.constexpr, ADD_C: tl.constexpr, K_EVEN: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr):
        """C (+)= A B with B = hi (+ lo) bf16. A fp32 (A_FP32: split in registers) or bf16. Products of the BK
        tile: lo_a hi_b + hi_a lo_b + hi_a hi_b in the same tile accumulator (small ones first), then
        acc = acc + tile with an fp32 vector addition (no truncated WMMA accumulation).
        Grid axis 1 = K chunk (split-K, KS elements, multiple of BK): partial output at c + s sc_s."""
        pid = tl.program_id(0)
        pid_s = tl.program_id(1)
        npm = tl.cdiv(M, BM)
        npn = tl.cdiv(N, BN)
        ning = GM * npn
        fm = (pid // ning) * GM
        gsm = min(npm - fm, GM)
        pid_m = fm + ((pid % ning) % gsm)
        pid_n = (pid % ning) // gsm
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        m_m = rm < M
        m_n = rn < N
        rm64 = rm.to(tl.int64)
        rn64 = rn.to(tl.int64)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        k_ini = pid_s * KS
        k_end = tl.minimum(K, k_ini + KS)
        for k0 in range(k_ini, k_end, BK):
            rk = k0 + tl.arange(0, BK)
            if K_EVEN:
                ma = m_m[:, None]
                mb = m_n[None, :]
            else:
                ma = m_m[:, None] & (rk < K)[None, :]
                mb = (rk < K)[:, None] & m_n[None, :]
            a = tl.load(a_ptr + rm64[:, None] * sa_m + rk[None, :].to(tl.int64) * sa_k, mask=ma, other=0.0)
            boff = rk[:, None].to(tl.int64) * sb_k + rn64[None, :] * sb_n
            bh = tl.load(bh_ptr + boff, mask=mb, other=0.0)
            z = acc * 0.0                      # tile accumulator: a zero that Triton does not fuse with acc
            if A_FP32:
                ah = a.to(tl.bfloat16, fp_downcast_rounding="rtne")
                al = (a - ah.to(tl.float32)).to(tl.bfloat16, fp_downcast_rounding="rtne")
                if HA_LO:
                    bl = tl.load(bl_ptr + boff, mask=mb, other=0.0)
                    r = tl.dot(al, bh, z)
                    r = tl.dot(ah, bl, r)
                    r = tl.dot(ah, bh, r)
                else:
                    r = tl.dot(al, bh, z)
                    r = tl.dot(ah, bh, r)
            else:
                if HA_LO:
                    bl = tl.load(bl_ptr + boff, mask=mb, other=0.0)
                    r = tl.dot(a, bl, z)
                    r = tl.dot(a, bh, r)
                else:
                    r = tl.dot(a, bh, z)
            acc = acc + r
        coff = pid_s.to(tl.int64) * sc_s + rm64[:, None] * sc_m + rn64[None, :] * sc_n
        cm = m_m[:, None] & m_n[None, :]
        if ADD_C:
            acc = tl.load(c_ptr + coff, mask=cm, other=0.0) + acc
        tl.store(c_ptr + coff, acc, mask=cm)


def _kernel():
    if triton is None:
        raise RuntimeError("bf16x3: triton missing, triton path impossible")
    return _k_x3, triton


# FIXED configuration (no autotune: same shape -> same tile -> same bits on every resume). Chosen from the
# 2026-09-27 sweep on gfx1151 (8192x2048x8192, alongside the run): 64x64x32, 4 warps, 1 stage = 46-48 ms
# versus 112-124 for the fp32 GEMM; 128x128 tiles overflow the registers (0.3 TFLOPS).
CONF_TRITON = {"BM": 64, "BN": 64, "BK": 32, "GM": 8, "num_warps": 4, "num_stages": 1}
# split-K (also deterministic: depends only on the shape): when the output tiles are few and K is long
# (head dL/dh: 512 x 248320 x 2048 = 256 tiles, 7760 rounds each: 351 ms versus 307 for fp32)
# K is split into S parts, partial fp32 outputs (S, M, N) summed by torch in fp32. Measured 2026-09-27 (alongside
# the run): S = 1, 2, 4, 8, 16 -> 180, 164, 145, 123, 98 ms (fp32 260), error 4.5e-6 in all cases.
SPLITK_PROGRAMS = 4096      # split while tiles x S <= this
SPLITK_K_MIN = 16384         # ... and only if K is at least this
SPLITK_MAX = 16


def chunks_k(M: int, N: int, K: int) -> int:
    """How many K parts (1 = no split-K) for the shape (M, N, K), with CONF_TRITON."""
    cf = CONF_TRITON
    prog = -(-M // cf["BM"]) * -(-N // cf["BN"])
    if K < SPLITK_K_MIN or prog >= SPLITK_PROGRAMS:
        return 1
    return int(max(1, min(SPLITK_MAX, SPLITK_PROGRAMS // prog, K // (4 * cf["BK"]))))


def mm_triton(a: torch.Tensor, b_hi: torch.Tensor, b_lo: Optional[torch.Tensor] = None,
              c: Optional[torch.Tensor] = None) -> torch.Tensor:
    """(c +) a @ (b_hi + b_lo) fp32, 2D. a fp32 (bf16x3: three products, or two without b_lo) or bf16 (one or two
    products: its values are exact). b_hi/b_lo bf16 (K, N), same strides (transposed views too). c fp32 (M, N):
    accumulated into and written in place (like addmm with beta = 1)."""
    k, triton = _kernel()
    M, K = a.shape
    if b_hi.shape[0] != K:
        raise ValueError(f"mm_triton: a {tuple(a.shape)} versus b {tuple(b_hi.shape)}")
    if a.dtype not in (torch.float32, torch.bfloat16) or b_hi.dtype != torch.bfloat16:
        raise TypeError(f"mm_triton: a {a.dtype}, b {b_hi.dtype} (expected float32|bfloat16 and bfloat16)")
    if b_lo is not None and (b_lo.stride() != b_hi.stride() or b_lo.shape != b_hi.shape
                             or b_lo.dtype != torch.bfloat16):
        raise ValueError("mm_triton: b_hi and b_lo with different shape/strides/type")
    N = b_hi.shape[1]
    if c is None:
        out = torch.empty(M, N, device=a.device, dtype=torch.float32)
    else:
        if c.dtype != torch.float32 or tuple(c.shape) != (M, N):
            raise ValueError(f"mm_triton: c {c.dtype} {tuple(c.shape)}")
        out = c
    if M == 0 or N == 0 or K == 0:
        return out if c is not None else out.zero_()
    cf = CONF_TRITON
    S = chunks_k(M, N, K)
    KS = -(-K // S)
    KS = -(-KS // cf["BK"]) * cf["BK"]            # multiple of BK: every chunk starts on a whole tile
    S = -(-K // KS)
    dest = out if S == 1 else torch.empty(S, M, N, device=a.device, dtype=torch.float32)
    sd = (0, dest.stride(0), dest.stride(1)) if S == 1 else dest.stride()
    grid = (triton.cdiv(M, cf["BM"]) * triton.cdiv(N, cf["BN"]), S)
    k[grid](a, a.stride(0), a.stride(1), b_hi, b_hi if b_lo is None else b_lo, b_hi.stride(0), b_hi.stride(1),
            dest, sd[0], sd[1], sd[2], M, N, K, KS,
            A_FP32=a.dtype == torch.float32, HA_LO=b_lo is not None, ADD_C=c is not None and S == 1,
            K_EVEN=K % cf["BK"] == 0, BM=cf["BM"], BN=cf["BN"], BK=cf["BK"], GM=cf["GM"],
            num_warps=cf["num_warps"], num_stages=cf["num_stages"])
    if S > 1:
        if c is None:
            torch.sum(dest, 0, out=out)
        else:
            out.add_(dest.sum(0))
    return out


def _trial_triton(device: torch.device) -> dict:
    """The triton path on the real device: bf16x3 versus fp64 (<= 1e-5 relative in norm), the bf16 A (exact except
    for order: <= 1e-6) and the accumulation drift (long sum of positive products: <= 1e-6; the truncating WMMA
    would give ~2e-5)."""
    out = {}
    try:
        g = torch.Generator(device=device).manual_seed(0)
        a = torch.randn(300, 2080, device=device, generator=g)
        w = torch.randn(200, 2080, device=device, generator=g) * 0.02
        h, lo = decompose(w)
        rif = a.double() @ w.double().t()
        y = mm_triton(a, h.t(), lo.t())
        out["err_x3"] = float((y.double() - rif).norm() / rif.norm())
        ab = a.to(torch.bfloat16)
        ref1 = ab.double() @ h.double().t()
        out["err_bf16"] = float((mm_triton(ab, h.t()).double() - ref1).norm() / ref1.norm())
        ap = (torch.rand(32, 16384, device=device, generator=g) + 1).to(torch.bfloat16)
        bp = (torch.rand(16384, 64, device=device, generator=g) + 1).to(torch.bfloat16)
        rifp = ap.double() @ bp.double()
        out["drift_accumulator"] = float(((mm_triton(ap, bp).double() - rifp) / rifp).mean())
        out["ok"] = bool(out["err_x3"] <= 1e-5 and out["err_bf16"] <= 1e-6 and abs(out["drift_accumulator"]) <= 1e-6)
    except Exception as ex:                  # triton missing or broken: fallback (recorded)
        out["ok"] = False
        out["error"] = repr(ex)[:300]
    return out


def _trial_via(device: torch.device) -> dict:
    """On the real device: does mm.dtype (bf16 -> fp32) exist? And addmm.dtype? Is the output that of an fp32
    accumulation (comparison with the fp32 emulation, same arithmetic except for order)? Without mm.dtype: does the
    bf16 GEMM accumulate in fp32 before rounding the output? (fraction of outputs == bf16(emulation))."""
    g = torch.Generator(device=device).manual_seed(0)
    a = torch.randn(256, 512, device=device, generator=g).to(torch.bfloat16)
    b = torch.randn(512, 128, device=device, generator=g).to(torch.bfloat16)
    rif = a.float() @ b.float()
    scale = float(rif.abs().max())
    out = {"device": str(device)}
    try:
        y = torch.mm(a, b, out_dtype=torch.float32)
        out["mm_out_dtype"] = y.dtype == torch.float32
        out["err_mm_out_dtype"] = float((y - rif).abs().max()) / scale
    except Exception as ex:
        out["mm_out_dtype"] = False
        out["error_mm_out_dtype"] = repr(ex)[:200]
    try:
        c = torch.randn(256, 128, device=device, generator=g)
        y = torch.addmm(c, a, b, out_dtype=torch.float32)
        out["addmm_out_dtype"] = y.dtype == torch.float32
        out["err_addmm_out_dtype"] = float((y - (c + rif)).abs().max()) / scale
    except Exception as ex:
        out["addmm_out_dtype"] = False
        out["error_addmm_out_dtype"] = repr(ex)[:200]
    y16 = torch.mm(a, b)
    out["output_bf16_equal_a_bf16_accumulator_fp32"] = float((y16 == rif.to(torch.bfloat16)).float().mean())
    ok = out["mm_out_dtype"] and out.get("err_mm_out_dtype", 1.0) < 1e-5
    out["via"] = "out_dtype" if ok else "output_bf16"
    out["addmm"] = bool(ok and out["addmm_out_dtype"] and out.get("err_addmm_out_dtype", 1.0) < 1e-5)
    out["triton"] = _trial_triton(device)
    if out["triton"]["ok"]:
        out["via"] = "triton"
    return out


def via_mm(device) -> dict:
    device = torch.device(device)
    k = str(device)
    if k not in _VIA:
        if VIA_FORCED:
            _VIA[k] = {"device": k, "via": VIA_FORCED, "addmm": VIA_FORCED == "out_dtype", "forced": True}
            if VIA_FORCED not in ("triton", "out_dtype", "output_bf16", "emulate"):
                raise ValueError(f"E2E_BF16X3_VIA={VIA_FORCED!r}: triton | out_dtype | output_bf16 | emulate")
        elif device.type == "cpu":
            _VIA[k] = {"device": k, "via": "emulate", "addmm": False}
        else:
            _VIA[k] = _trial_via(device)
    return _VIA[k]


def state() -> dict:
    """The paths chosen so far (for the gate outcome and the registry)."""
    return dict(_VIA)


def mm_bf16(a: torch.Tensor, b: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
    """(c +) a @ b with a, b bf16 2D: exact products, fp32 accumulation, fp32 output (except on the "output_bf16" path)."""
    v = via_mm(a.device)
    via = v["via"]
    if via == "triton":
        return mm_triton(a, b, None, c)
    if via == "out_dtype":
        if c is None:
            return torch.mm(a, b, out_dtype=torch.float32)
        if v["addmm"]:
            return torch.addmm(c, a, b, out_dtype=torch.float32)
        return c.add_(torch.mm(a, b, out_dtype=torch.float32))
    if via == "output_bf16":
        y = torch.mm(a, b).float()
    elif via == "emulate":
        y = a.float() @ b.float()
    else:
        raise ValueError(f"E2E_BF16X3_VIA={via!r}: triton | out_dtype | output_bf16 | emulate")
    return y if c is None else c.add_(y)


def mm_x3(a_hi, a_lo, b_hi, b_lo) -> torch.Tensor:
    """a @ b in bf16x3 from the splits: (lo_a hi_b + hi_a lo_b) + hi_a hi_b, fp32."""
    y = mm_bf16(a_lo, b_hi)
    y = mm_bf16(a_hi, b_lo, y)
    return mm_bf16(a_hi, b_hi, y)


def matmul_x3(a: torch.Tensor, b_hi: torch.Tensor, b_lo: torch.Tensor) -> torch.Tensor:
    """a (..., K) fp32 @ b (K, N) already split -> (..., N) fp32."""
    shape = a.shape
    if via_mm(a.device)["via"] == "triton":
        if a.dtype != torch.float32:
            raise TypeError(f"bf16x3: input {a.dtype}, expected float32")
        return mm_triton(a.reshape(-1, shape[-1]), b_hi, b_lo).reshape(*shape[:-1], b_hi.shape[1])
    ah, al = decompose(a.reshape(-1, shape[-1]))
    y = mm_x3(ah, al, b_hi, b_lo)
    del ah, al
    return y.reshape(*shape[:-1], b_hi.shape[1])


class _X3Linear(torch.autograd.Function):
    """y = x W^T (+ b) in bf16x3; dX = dY W in bf16x3. No gradient for W (frozen); x is not saved."""

    @staticmethod
    def forward(ctx, x, lin: "BF16x3Linear"):
        ctx.lin = lin
        y = matmul_x3(x, lin.hi.t(), lin.lo.t())
        if lin.bias is not None:
            y = y + lin.bias
        return y

    @staticmethod
    def backward(ctx, gy):
        lin = ctx.lin
        dx = None
        if ctx.needs_input_grad[0]:
            dx = matmul_x3(gy.to(torch.float32), lin.hi, lin.lo)
        return dx, None


class BF16x3Linear(torch.nn.Module):
    """Frozen nn.Linear with weight (out, in) split once into (hi, lo) bf16. hi/lo are plain attributes (NOT
    buffers: a `.to(dtype)` of the module must not convert them); build it AFTER moving the layer to the device.
    fp32 input and output."""

    def __init__(self, lin: torch.nn.Linear):
        super().__init__()
        W = lin.weight.detach().to(torch.float32)
        self.hi, self.lo = decompose(W)
        self.bias = None if lin.bias is None else lin.bias.detach().to(torch.float32).clone()
        self.in_features, self.out_features = lin.in_features, lin.out_features

    def forward(self, x):
        if x.dtype != torch.float32:
            raise TypeError(f"BF16x3Linear: input {x.dtype}, expected float32 (recipe B3: fp32 residual)")
        return _X3Linear.apply(x, self)

    @torch.no_grad()
    def set_weight(self, W: torch.Tensor) -> None:
        """2026-09-27: new fp32 weight (out, in) (reload of the requantized bytes, `fucina.student.lora` install)."""
        if tuple(W.shape) != (self.out_features, self.in_features):
            raise ValueError(f"BF16x3Linear: weight {tuple(W.shape)}, expected {(self.out_features, self.in_features)}")
        self.hi, self.lo = decompose(W.to(device=self.hi.device, dtype=torch.float32))

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bf16x3"


# Projections with fewer than MIN_OUTPUTS rows stay exact fp32: GDN in_proj_b / in_proj_a (32 rows, the beta gates
# and the decay), shared_expert_gate (1 row). They are < 0.3% of the non-expert FLOPs; the router is not an
# nn.Linear (the student MoE router_w) and stays fp32 as in B (top-k as in llama.cpp).
MIN_OUTPUTS = 256


class _K2Linear(torch.autograd.Function):
    """y = x_1 W_1^T + x_2 W_2^T (K in two halves, fp32); dX = [dY W]: dY_1 W_1 + dY_2 W_2 (outputs in two halves)."""

    @staticmethod
    def forward(ctx, x, lin: "K2Linear"):
        ctx.lin = lin
        h = lin.W.shape[1] // 2
        y = x[..., :h] @ lin.W[:, :h].t()
        y = y + x[..., h:] @ lin.W[:, h:].t()
        if lin.bias is not None:
            y = y + lin.bias
        return y

    @staticmethod
    def backward(ctx, gy):
        lin = ctx.lin
        dx = None
        if ctx.needs_input_grad[0]:
            h = lin.W.shape[0] // 2
            gy = gy.to(torch.float32)
            dx = gy[..., :h] @ lin.W[:h]
            dx = dx + gy[..., h:] @ lin.W[h:]
        return dx, None


class K2Linear(torch.nn.Module):
    """FROZEN fp32 nn.Linear with the sum over K in two halves ("noise" arm of the B3 gate: same fp32 arithmetic as
    B, a different order of the sums). Like `BF16x3Linear`: no dW, no saved x, fp32 input."""

    def __init__(self, lin: torch.nn.Linear):
        super().__init__()
        self.W = lin.weight.detach().to(torch.float32).clone()
        self.bias = None if lin.bias is None else lin.bias.detach().to(torch.float32).clone()
        self.in_features, self.out_features = lin.in_features, lin.out_features

    def forward(self, x):
        if x.dtype != torch.float32:
            raise TypeError(f"K2Linear: input {x.dtype}, expected float32")
        return _K2Linear.apply(x, self)

    @torch.no_grad()
    def set_weight(self, W: torch.Tensor) -> None:
        if tuple(W.shape) != tuple(self.W.shape):
            raise ValueError(f"K2Linear: weight {tuple(W.shape)}, expected {tuple(self.W.shape)}")
        self.W = W.to(device=self.W.device, dtype=torch.float32).clone()

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, fp32 K/2"


def replace_linear(module: torch.nn.Module, min_outputs: int = MIN_OUTPUTS, class_=None) -> list:
    """Replaces in place every nn.Linear with >= min_outputs rows by `class_` (default `BF16x3Linear`; the gate's
    noise arm: `K2Linear`). Returns the names."""
    class_ = BF16x3Linear if class_ is None else class_
    done = []
    for name, m in list(module.named_modules()):
        for child_name, child in list(m.named_children()):
            if isinstance(child, torch.nn.Linear) and child.out_features >= min_outputs:
                setattr(m, child_name, class_(child))
                done.append(f"{name}.{child_name}" if name else child_name)
    return done


class X3Head:
    """Student head (V, H) split ONCE (hi, lo bf16): training driver `--head-kl bf16x3`. Exposes shape / device /
    dtype like the fp32 tensor it replaces (whoever counts the targets outside the head reads shape)."""

    def __init__(self, W: torch.Tensor):
        self.shape = W.shape
        self.device = W.device
        self.dtype = torch.float32
        self.hi, self.lo = decompose(W.to(torch.float32))

    def logit(self, h: torch.Tensor) -> torch.Tensor:
        """h (P, H) fp32 -> h W^T (P, V) fp32."""
        return matmul_x3(h, self.hi.t(), self.lo.t())

    def per(self, p: torch.Tensor) -> torch.Tensor:
        """p (P, V) fp32 -> p W (P, H) fp32 (dL/dh of the KL)."""
        return matmul_x3(p, self.hi, self.lo)

#!/usr/bin/env python3
# FROZEN COPY v5 (2026-09-27) of the fast GDN module at commit 9b99f34 (in-place loop + compiled preparation, 21.9x
# faster than the fallback; error against true fp64 <= 1.16x that of the fallback). Selected with E2E_GDN=veloce.
"""Fast version of the transformers PyTorch fallback for the gated delta rule (torch_chunk_gated_delta_rule), with the
SAME mathematics in fp32 (2026-09-27).

Why it exists: an E2E run spends ~75% of its time in the 30 Gated DeltaNet layers, which are served by the transformers
fallback (fla does not run on gfx1151: page fault in fp32; torch.compile of the fallback: compilation takes > 1 hour).
The fallback is slow because of how it is WRITTEN, not because of its mathematics:
  1. the triangular solve (I + L)^-1 is a Python loop of 63 steps, each with two .clone() calls and one slice
     assignment: ~190 kernels in the forward, and in the backward every CopySlices clones the gradient of the whole
     tensor (63 times);
  2. the recurrence across chunks indexes query[:, :, i] and similar: in the backward every SelectBackward allocates a
     FULL tensor of zeros and copies the slice into it (N^2 memory traffic), and inside the loop it recomputes, for
     every chunk, products that do not depend on the state (q_i k_i^T, q_i * exp(g_i), k_i * exp(g_last - g_i)).
The new version (the levers add up; VERSION below selects which one runs):
  v1  (I + L)^-1 with ONE torch.linalg.solve_triangular (unitriangular, forward substitution: same algorithm as the
      loop); everything that does not depend on the state S is computed for ALL chunks at once, outside the loop; the
      recurrence on the state is a torch.autograd.Function with a hand-written backward (only the two truly
      sequential dependencies stay in the loop: v_new_i = u_i - w_i S_i ; S_{i+1} = S_i e^{g_last} + kw_i^T v_new_i,
      and their adjoint); the gradients of u, w, kw, e^{g_last} are computed afterwards, batched over all chunks.
  v2  PER-CHUNK layout (N, B*H, C, .): every slice in the loop is contiguous; the output o = qg S + attn v_new is
      computed inside the Function; hand-written backward for (I + L)^-1 too (dM = -X^T dX X^T: two products instead
      of two trsm).
  v3  the loop works IN PLACE (baddbmm_, mul out=, addcmul_): no device-to-device copies per step.
  v4  the recurrence in two Triton kernels: MEASURED SLOWER than v3 on gfx1151 (fp32 tl.dot without matrix units):
      kept as a documented alternative, not in use.
  v5  v3 + torch.compile (dynamic=False) of the loop-free preparation only (_prep): fuses the ~40 element-wise
      operations of the forward and of their backward. If compilation fails, it falls back to v3 automatically.
No dtype change: the computation is fp32 as in the fallback, TF32 never enabled (fp64 for the tests via
_dtype_compute).
Test: the GDN test script (forward + dq dk dv dg dbeta [+ initial/final state] against a TRUE fp64 reference — a copy
of the fallback with a parametric dtype — in 4 scenarios: normal, strong decay, weak decay, correlated keys).

Usage in a run (BEFORE the first forward; the transformers module looks up the name on every call):
    from fucina.kernels.gdn_fast_v5 import install
    install()          # M.torch_chunk_gated_delta_rule = chunk_gated_delta_rule_fast
In the E2E training script: E2E_GDN=veloce. Version: GDN_FAST_VERSION (default 5; 3 = without torch.compile).
"""
import os
from typing import Optional

import torch
import torch.nn.functional as F


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """Identical to transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.l2norm (aligned with fla)."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


class _Recurrence(torch.autograd.Function):
    """Cross-chunk recurrence of the gated delta rule, on the state S (B, H, Dk, Dv).

    Inputs (B, H, N, C, .): u = X v_beta, w = X (k_beta e^g), kw = k e^{g_last - g}; egl = e^{g_last} (B, H, N);
    S0 = initial state (B, H, Dk, Dv).
    For i = 0..N-1:  Sp_i = S_i ;  vn_i = u_i - w_i S_i ;  S_{i+1} = S_i egl_i + kw_i^T vn_i.
    Outputs: vn (B, H, N, C, Dv), Sp (B, H, N, Dk, Dv) = state BEFORE each chunk, S_N (B, H, Dk, Dv)."""

    @staticmethod
    def forward(ctx, u, w, kw, egl, S0):
        B, H, N, C, Dv = u.shape
        Dk = w.shape[-1]
        vn = torch.empty_like(u)
        Sp = torch.empty(B, H, N, Dk, Dv, dtype=u.dtype, device=u.device)
        S = S0
        for i in range(N):
            Sp[:, :, i] = S
            vn_i = u[:, :, i] - torch.matmul(w[:, :, i], S)                 # vn_i = u_i - w_i S_i
            vn[:, :, i] = vn_i
            S = S * egl[:, :, i, None, None] + torch.matmul(kw[:, :, i].transpose(-1, -2), vn_i)
        ctx.save_for_backward(w, kw, egl, vn, Sp)
        return vn, Sp, S

    @staticmethod
    def backward(ctx, d_vn, d_Sp, d_SN):
        w, kw, egl, vn, Sp = ctx.saved_tensors
        B, H, N, C, Dv = vn.shape
        Dk = w.shape[-1]
        dt, dev = vn.dtype, vn.device
        d_vn = torch.zeros_like(vn) if d_vn is None else d_vn
        dS = torch.zeros(B, H, Dk, Dv, dtype=dt, device=dev) if d_SN is None else d_SN.contiguous()
        dvn_tot = torch.empty_like(vn)                       # total gradient of vn_i
        dS_after = torch.empty_like(Sp)                       # dL/dS_{i+1}, for the batched gradients of kw and egl
        for i in range(N - 1, -1, -1):
            dS_after[:, :, i] = dS
            # S_{i+1} = S_i egl_i + kw_i^T vn_i  ->  dvn_i += kw_i dS_{i+1}
            dvn_i = d_vn[:, :, i] + torch.matmul(kw[:, :, i], dS)
            dvn_tot[:, :, i] = dvn_i
            # dS_i = dS_{i+1} egl_i - w_i^T dvn_i (+ d_Sp_i, because Sp_i = S_i is an output)
            dS = dS * egl[:, :, i, None, None] - torch.matmul(w[:, :, i].transpose(-1, -2), dvn_i)
            if d_Sp is not None:
                dS = dS + d_Sp[:, :, i]
        du = dvn_tot
        dw = -torch.matmul(dvn_tot, Sp.transpose(-1, -2))                    # vn_i = u_i - w_i S_i
        dkw = torch.matmul(vn, dS_after.transpose(-1, -2))                     # (kw_i^T vn_i): d kw_i = vn_i dS^T
        degl = (Sp * dS_after).sum((-1, -2))                                   # S_i egl_i
        return du, dw, dkw, degl, dS


def _core(q, k, v, g, beta, chunk_size: int, initial_state: Optional[torch.Tensor], dt: torch.dtype):
    """q, k already normalized if requested; all in (B, T, H, .) as transformers passes them."""
    q, k, v, beta, g = [x.transpose(1, 2).contiguous().to(dt) for x in (q, k, v, beta, g)]
    B, H, T, Dk = k.shape
    Dv = v.shape[-1]
    C = chunk_size
    pad = (C - T % C) % C
    if pad:
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
        beta = F.pad(beta, (0, pad))
        g = F.pad(g, (0, pad))
    L = T + pad
    N = L // C
    q = q * (1 / (q.shape[-1] ** 0.5))
    k_beta = k * beta.unsqueeze(-1)
    v_beta = v * beta.unsqueeze(-1)
    q, k, k_beta, v_beta = [x.reshape(B, H, N, C, x.shape[-1]) for x in (q, k, k_beta, v_beta)]
    g = g.reshape(B, H, N, C).cumsum(-1)

    triu0 = torch.triu(torch.ones(C, C, dtype=torch.bool, device=q.device), diagonal=0)    # diagonal included
    triu1 = torch.triu(torch.ones(C, C, dtype=torch.bool, device=q.device), diagonal=1)
    # intra-chunk decay (i >= j): exp(g_i - g_j); 0 above the diagonal (like .tril().exp().tril() in the fallback)
    decay = (g.unsqueeze(-1) - g.unsqueeze(-2)).masked_fill(triu1, float("-inf")).exp()
    Lm = ((k_beta @ k.transpose(-1, -2)) * decay).masked_fill(triu0, 0)       # strictly lower triangular
    eye = torch.eye(C, dtype=dt, device=q.device)
    # X = (I + Lm)^-1 : unitriangular forward substitution (the fallback does the same, row by row in Python)
    X = torch.linalg.solve_triangular(Lm, eye.expand(B, H, N, C, C), upper=False, unitriangular=True)
    eg = g.exp()
    u = X @ v_beta                                   # "value" in the fallback
    w = X @ (k_beta * eg.unsqueeze(-1))              # "k_cumdecay" in the fallback
    attn = (q @ k.transpose(-1, -2)) * decay         # intra-chunk, lower triangular including the diagonal
    qg = q * eg.unsqueeze(-1)
    gl = g[..., -1]                                  # (B, H, N)
    kw = k * (gl.unsqueeze(-1) - g).exp().unsqueeze(-1)
    egl = gl.exp()
    S0 = (torch.zeros(B, H, Dk, Dv, dtype=dt, device=q.device) if initial_state is None
          else initial_state.to(device=q.device, dtype=dt))
    vn, Sp, SN = _Recurrence.apply(u.contiguous(), w.contiguous(), kw.contiguous(), egl.contiguous(), S0)
    o = qg @ Sp + attn @ vn                          # (B, H, N, C, Dv)
    o = o.reshape(B, H, L, Dv)[:, :, :T]
    return o, SN


# ───────────────────────────── version 2: per-chunk layout, hand-written backward for (I + L)^-1 too ─────────────
class _Inverse(torch.autograd.Function):
    """X = (I + Lm)^-1, Lm strictly lower triangular (C, C), batched. Forward: unitriangular forward substitution
    (solve_triangular). Backward: X = M^-1 -> dM = -X^T dX X^T, restricted to the strictly lower part (the only one
    Lm carries): two products instead of the two triangular solves of the generic backward."""

    @staticmethod
    def forward(ctx, Lm):
        C = Lm.shape[-1]
        eye = torch.eye(C, dtype=Lm.dtype, device=Lm.device).expand(Lm.shape)
        X = torch.linalg.solve_triangular(Lm, eye, upper=False, unitriangular=True)
        ctx.save_for_backward(X)
        return X

    @staticmethod
    def backward(ctx, dX):
        (X,) = ctx.saved_tensors
        Xt = X.transpose(-1, -2)
        return torch.matmul(torch.matmul(Xt, dX), Xt).neg_().tril_(-1)


class _RecurrenceOutput(torch.autograd.Function):
    """Cross-chunk recurrence + output, in PER-CHUNK layout (N, BH, C, .): every slice in the loop is contiguous.
    For i = 0..N-1:  vn_i = u_i - w_i S_i ;  S_{i+1} = S_i egl_i + kw_i^T vn_i ;  o_i = qg_i S_i + attn_i vn_i.
    Inputs: u, w, kw, qg, attn (N, BH, C, .), egl (N, BH), S0 (BH, Dk, Dv). Outputs: o (N, BH, C, Dv), S_N."""

    @staticmethod
    def forward(ctx, u, w, kw, egl, qg, attn, S0):
        N, BH, C, Dv = u.shape
        Dk = w.shape[-1]
        Sa = torch.empty(N + 1, BH, Dk, Dv, dtype=u.dtype, device=u.device)    # Sa[i] = S_i
        Sa[0].copy_(S0)
        vn = torch.empty_like(u)
        kwT = kw.transpose(-1, -2)
        eglv = egl.unsqueeze(-1).unsqueeze(-1)
        for i in range(N):
            torch.baddbmm(u[i], w[i], Sa[i], alpha=-1, out=vn[i])                      # u_i - w_i S_i
            torch.baddbmm(Sa[i] * eglv[i], kwT[i], vn[i], out=Sa[i + 1])               # S_i egl_i + kw_i^T vn_i
        Sp = Sa[:N]
        o = torch.matmul(qg, Sp)
        o.add_(torch.matmul(attn, vn))                                                 # qg_i S_i + attn_i vn_i
        ctx.save_for_backward(w, kw, egl, qg, attn, vn, Sa)
        return o, Sa[N].clone()

    @staticmethod
    def backward(ctx, d_o, d_SN):
        w, kw, egl, qg, attn, vn, Sa = ctx.saved_tensors
        N = vn.shape[0]
        Sp = Sa[:N]
        d_o = d_o.contiguous()
        d_attn = torch.matmul(d_o, vn.transpose(-1, -2))
        d_qg = torch.matmul(d_o, Sp.transpose(-1, -2))
        d_vn = torch.matmul(attn.transpose(-1, -2), d_o)                                # direct part (o)
        d_Sp = torch.matmul(qg.transpose(-1, -2), d_o)
        dSa = torch.empty_like(Sa)                                                       # dSa[i] = total dL/dS_i
        if d_SN is None:
            dSa[N].zero_()
        else:
            dSa[N].copy_(d_SN)
        dvn = torch.empty_like(vn)
        wT = w.transpose(-1, -2)
        eglv = egl.unsqueeze(-1).unsqueeze(-1)
        for i in range(N - 1, -1, -1):
            torch.baddbmm(d_vn[i], kw[i], dSa[i + 1], out=dvn[i])                       # + kw_i dS_{i+1}
            torch.baddbmm(torch.addcmul(d_Sp[i], dSa[i + 1], eglv[i]), wT[i], dvn[i], alpha=-1, out=dSa[i])
        dS_after = dSa[1:]
        dw = torch.matmul(dvn, Sp.transpose(-1, -2)).neg_()
        dkw = torch.matmul(vn, dS_after.transpose(-1, -2))
        degl = (Sp * dS_after).sum((-1, -2))
        return dvn, dw, dkw, degl, d_qg, d_attn, dSa[0]


class _RecurrenceOutput3(torch.autograd.Function):
    """Like _RecurrenceOutput, but the loop works IN PLACE (baddbmm_ / mul out= / addcmul_): no device-to-device
    copy per step (the out= form of baddbmm first copies the input into the output: 256 copies per call in the v2
    profile) and no temporary for the state."""

    @staticmethod
    def forward(ctx, u, w, kw, egl, qg, attn, S0):
        N, BH, C, Dv = u.shape
        Dk = w.shape[-1]
        Sa = torch.empty(N + 1, BH, Dk, Dv, dtype=u.dtype, device=u.device)    # Sa[i] = S_i
        Sa[0].copy_(S0)
        vn = u.clone()
        kwT = kw.transpose(-1, -2)
        eglv = egl.unsqueeze(-1).unsqueeze(-1)
        for i in range(N):
            vn[i].baddbmm_(w[i], Sa[i], alpha=-1)                                   # vn_i = u_i - w_i S_i
            torch.mul(Sa[i], eglv[i], out=Sa[i + 1])                                # S_i egl_i
            Sa[i + 1].baddbmm_(kwT[i], vn[i])                                       # + kw_i^T vn_i
        Sp = Sa[:N]
        o = torch.matmul(qg, Sp)
        o.view(N * BH, C, Dv).baddbmm_(attn.view(N * BH, C, C), vn.view(N * BH, C, Dv))   # + attn_i vn_i
        ctx.save_for_backward(w, kw, egl, qg, attn, vn, Sa)
        return o, Sa[N].clone()

    @staticmethod
    def backward(ctx, d_o, d_SN):
        w, kw, egl, qg, attn, vn, Sa = ctx.saved_tensors
        N, BH, C, Dv = vn.shape
        Dk = w.shape[-1]
        Sp = Sa[:N]
        d_o = d_o.contiguous()
        d_attn = torch.matmul(d_o, vn.transpose(-1, -2))
        d_qg = torch.matmul(d_o, Sp.transpose(-1, -2))
        dvn = torch.matmul(attn.transpose(-1, -2), d_o)                                 # direct part, then in place
        dSa = torch.empty_like(Sa)                                                        # dSa[i] = total dL/dS_i
        torch.matmul(qg.transpose(-1, -2), d_o, out=dSa[:N])                             # direct part (o_i = qg_i S_i)
        if d_SN is None:
            dSa[N].zero_()
        else:
            dSa[N].copy_(d_SN)
        wT = w.transpose(-1, -2)
        eglv = egl.unsqueeze(-1).unsqueeze(-1)
        for i in range(N - 1, -1, -1):
            dvn[i].baddbmm_(kw[i], dSa[i + 1])                                           # + kw_i dS_{i+1}
            dSa[i].addcmul_(dSa[i + 1], eglv[i])                                         # + dS_{i+1} egl_i
            dSa[i].baddbmm_(wT[i], dvn[i], alpha=-1)                                     # - w_i^T dvn_i
        dS_after = dSa[1:]
        dw = torch.matmul(dvn, Sp.transpose(-1, -2)).neg_()
        dkw = torch.matmul(vn, dS_after.transpose(-1, -2))
        degl = (Sp * dS_after).sum((-1, -2))
        return dvn, dw, dkw, degl, d_qg, d_attn, dSa[0]


# ───────────────────── version 4: the recurrence in TWO Triton kernels (forward, backward) ─────────────────────
# The columns of S (value index) are INDEPENDENT in the recurrence: vn_i[:, c] = u_i[:, c] - w_i S_i[:, c],
# S_{i+1}[:, c] = S_i[:, c] egl_i + kw_i^T vn_i[:, c]; the same holds for the adjoint. One program = (head bh, block of
# BV columns) and walks the 64 chunks IN SEQUENCE keeping its piece of S in registers: 1 launch instead of 192
# (forward) and of 192 (backward), and the products stay in fp32 (tl.dot input_precision="ieee": no tf32).
try:
    import triton
    import triton.language as tl
    _TRITON = True
except Exception:                      # pragma: no cover - without Triton v3 remains
    _TRITON = False

if _TRITON:
    @triton.jit
    def _k_rec_forward(u_ptr, w_ptr, kw_ptr, egl_ptr, s0_ptr, vn_ptr, sa_ptr,
                      BH, N: tl.constexpr,
                      C: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr):
        """For i = 0..N-1: Sa[i] = S; vn_i = u_i - w_i S; S = S egl_i + kw_i^T vn_i. At the end Sa[N] = S.
        Per-chunk layout, contiguous: u/vn (N, BH, C, DV), w/kw (N, BH, C, DK), egl (N, BH),
        s0 (BH, DK, DV), sa (N + 1, BH, DK, DV)."""
        bh = tl.program_id(0).to(tl.int64)
        iv = tl.program_id(1)
        rc = tl.arange(0, C)
        rk = tl.arange(0, DK)
        rv = iv * BV + tl.arange(0, BV)
        off_s = rk[:, None] * DV + rv[None, :]                         # (DK, BV) within one state matrix
        off_ck = rc[:, None] * DK + rk[None, :]                        # (C, DK)
        off_cv = rc[:, None] * DV + rv[None, :]                        # (C, BV)
        S = tl.load(s0_ptr + bh * DK * DV + off_s)
        for i in range(N):
            i64 = i
            base_s = (i64 * BH + bh) * DK * DV
            tl.store(sa_ptr + base_s + off_s, S)
            wi = tl.load(w_ptr + (i64 * BH + bh) * C * DK + off_ck)
            ui = tl.load(u_ptr + (i64 * BH + bh) * C * DV + off_cv)
            vni = ui - tl.dot(wi, S, input_precision="ieee")
            tl.store(vn_ptr + (i64 * BH + bh) * C * DV + off_cv, vni)
            kwi = tl.load(kw_ptr + (i64 * BH + bh) * C * DK + off_ck)
            e = tl.load(egl_ptr + i64 * BH + bh)
            S = S * e + tl.dot(tl.trans(kwi), vni, input_precision="ieee")
        tl.store(sa_ptr + (N * BH + bh) * DK * DV + off_s, S)

    @triton.jit
    def _k_rec_backward(dvn_ptr, dsa_ptr, w_ptr, kw_ptr, egl_ptr,
                        BH, N: tl.constexpr,
                        C: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr):
        """Adjoint, IN PLACE. Input: dvn = direct part of dL/dvn (attn^T d_o), dsa[:N] = direct part of
        dL/dS_i (qg^T d_o), dsa[N] = dL/dS_N. For i = N-1..0: dvn_i += kw_i dS_{i+1};
        dS_i = dS_{i+1} egl_i + dsa[i] - w_i^T dvn_i. Output: total dvn, dsa[i] = total dL/dS_i."""
        bh = tl.program_id(0).to(tl.int64)
        iv = tl.program_id(1)
        rc = tl.arange(0, C)
        rk = tl.arange(0, DK)
        rv = iv * BV + tl.arange(0, BV)
        off_s = rk[:, None] * DV + rv[None, :]
        off_ck = rc[:, None] * DK + rk[None, :]
        off_cv = rc[:, None] * DV + rv[None, :]
        dS = tl.load(dsa_ptr + (N * BH + bh) * DK * DV + off_s)
        for j in range(N):
            i64 = N - 1 - j
            kwi = tl.load(kw_ptr + (i64 * BH + bh) * C * DK + off_ck)
            p_dvn = dvn_ptr + (i64 * BH + bh) * C * DV + off_cv
            dvni = tl.load(p_dvn) + tl.dot(kwi, dS, input_precision="ieee")
            tl.store(p_dvn, dvni)
            wi = tl.load(w_ptr + (i64 * BH + bh) * C * DK + off_ck)
            e = tl.load(egl_ptr + i64 * BH + bh)
            p_ds = dsa_ptr + (i64 * BH + bh) * DK * DV + off_s
            dS = dS * e + tl.load(p_ds) - tl.dot(tl.trans(wi), dvni, input_precision="ieee")
            tl.store(p_ds, dS)


_INTERPRETER = os.environ.get("TRITON_INTERPRET") == "1"      # kernel tests on CPU (Triton interpreter)
_BV = 32          # state columns per program: grid (B*H, Dv/_BV) = 64 x 4 at the Thor dimensions
_NW = 8           # warps per program


class _RecurrenceTriton(torch.autograd.Function):
    """Same mathematics as _RecurrenceOutput3; the loop over the chunks is inside two Triton kernels."""

    @staticmethod
    def forward(ctx, u, w, kw, egl, qg, attn, S0):
        N, BH, C, Dv = u.shape
        Dk = w.shape[-1]
        assert Dv % _BV == 0 and u.dtype == torch.float32
        u, w, kw, egl = (t.contiguous() for t in (u, w, kw, egl))
        Sa = torch.empty(N + 1, BH, Dk, Dv, dtype=u.dtype, device=u.device)
        vn = torch.empty_like(u)
        _k_rec_forward[(BH, Dv // _BV)](u, w, kw, egl, S0.contiguous(), vn, Sa, BH, N=N, C=C, DK=Dk, DV=Dv, BV=_BV,
                                       num_warps=_NW)
        Sp = Sa[:N]
        o = torch.matmul(qg, Sp)
        o.view(N * BH, C, Dv).baddbmm_(attn.reshape(N * BH, C, C), vn.view(N * BH, C, Dv))
        ctx.save_for_backward(w, kw, egl, qg, attn, vn, Sa)
        return o, Sa[N].clone()

    @staticmethod
    def backward(ctx, d_o, d_SN):
        w, kw, egl, qg, attn, vn, Sa = ctx.saved_tensors
        N, BH, C, Dv = vn.shape
        Dk = w.shape[-1]
        Sp = Sa[:N]
        d_o = d_o.contiguous()
        d_attn = torch.matmul(d_o, vn.transpose(-1, -2))
        d_qg = torch.matmul(d_o, Sp.transpose(-1, -2))
        dvn = torch.matmul(attn.transpose(-1, -2), d_o)
        dSa = torch.empty_like(Sa)
        torch.matmul(qg.transpose(-1, -2), d_o, out=dSa[:N])
        if d_SN is None:
            dSa[N].zero_()
        else:
            dSa[N].copy_(d_SN)
        _k_rec_backward[(BH, Dv // _BV)](dvn, dSa, w, kw, egl, BH, N=N, C=C, DK=Dk, DV=Dv, BV=_BV, num_warps=_NW)
        dS_after = dSa[1:]
        dw = torch.matmul(dvn, Sp.transpose(-1, -2)).neg_()
        dkw = torch.matmul(vn, dS_after.transpose(-1, -2))
        degl = (Sp * dS_after).sum((-1, -2))
        return dvn, dw, dkw, degl, d_qg, d_attn, dSa[0]


def _in_chunk(x: torch.Tensor, N: int, C: int) -> torch.Tensor:
    """(B, L, H, ...) -> (N, B*H, C, ...), contiguous."""
    B, L, H = x.shape[:3]
    remainder = x.shape[3:]
    x = x.reshape(B, N, C, H, *remainder)
    perm = (1, 0, 3, 2) + tuple(range(4, x.dim()))
    return x.permute(*perm).reshape(N, B * H, C, *remainder)


def _prep(q, k, v, g, beta, C: int, N: int, l2: bool):
    """Everything that does not depend on the state, for ALL chunks at once (per-chunk layout (N, B*H, C, .)).
    Pure, loop-free function: v5 compiles it with torch.compile (which fuses the element-wise operations)."""
    if l2:
        q = l2norm(q, dim=-1, eps=1e-6)
        k = l2norm(k, dim=-1, eps=1e-6)
    q = q * (1 / (q.shape[-1] ** 0.5))
    k_beta = k * beta.unsqueeze(-1)
    v_beta = v * beta.unsqueeze(-1)
    q, k, k_beta, v_beta, g = [_in_chunk(x, N, C) for x in (q, k, k_beta, v_beta, g)]    # (N, BH, C, .)
    g = g.cumsum(-1)
    triu0 = torch.triu(torch.ones(C, C, dtype=torch.bool, device=q.device), diagonal=0)
    triu1 = torch.triu(torch.ones(C, C, dtype=torch.bool, device=q.device), diagonal=1)
    decay = (g.unsqueeze(-1) - g.unsqueeze(-2)).masked_fill(triu1, float("-inf")).exp()
    Lm = ((k_beta @ k.transpose(-1, -2)) * decay).masked_fill(triu0, 0)
    attn = (q @ k.transpose(-1, -2)) * decay
    eg = g.exp()
    kbeg = k_beta * eg.unsqueeze(-1)
    qg = q * eg.unsqueeze(-1)
    gl = g[..., -1]                                   # (N, BH)
    kw = k * (gl.unsqueeze(-1) - g).exp().unsqueeze(-1)
    egl = gl.exp()
    return Lm, v_beta, kbeg, attn, qg, kw, egl


_PREP_COMPILED = None
_COMPILATION_FAILED = False


def _prep_compiled():
    global _PREP_COMPILED
    if _PREP_COMPILED is None:
        _PREP_COMPILED = torch.compile(_prep, dynamic=False)
    return _PREP_COMPILED


def _prep_choice(compile_: bool):
    """The compiled preparation (v5) if requested and if compilation has not already failed in this process;
    otherwise the same function in eager mode (v3): same mathematics, just more kernels."""
    if not compile_ or _COMPILATION_FAILED:
        return _prep
    comp = _prep_compiled()

    def f(*a):
        global _COMPILATION_FAILED
        try:
            return comp(*a)
        except Exception as e:                  # compilation error (inductor/Triton): stay on v3, with a loud warning
            _COMPILATION_FAILED = True
            import warnings
            warnings.warn(f"gdn_fast_v5: torch.compile of the preparation failed ({type(e).__name__}: {e}); "
                          f"using the eager preparation (v3)")
            return _prep(*a)
    return f


def _core_v2(q, k, v, g, beta, chunk_size: int, initial_state: Optional[torch.Tensor], dt: torch.dtype,
               recurrence=_RecurrenceOutput, l2: bool = False, compile_: bool = False):
    """Like _core, but in per-chunk layout (N, B*H, C, .) and with the hand-written-backward Functions."""
    B, T, H, Dk = k.shape
    Dv = v.shape[-1]
    C = chunk_size
    pad = (C - T % C) % C
    q, k, v, beta, g = [x.to(dt) for x in (q, k, v, beta, g)]
    if pad:
        q = F.pad(q, (0, 0, 0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, 0, 0, pad))
        beta = F.pad(beta, (0, 0, 0, pad))
        g = F.pad(g, (0, 0, 0, pad))
    L = T + pad
    N = L // C
    Lm, v_beta, kbeg, attn, qg, kw, egl = _prep_choice(compile_)(q, k, v, g, beta, C, N, l2)
    X = _Inverse.apply(Lm)
    u = X @ v_beta
    w = X @ kbeg
    S0 = (torch.zeros(B * H, Dk, Dv, dtype=dt, device=q.device) if initial_state is None
          else initial_state.to(device=q.device, dtype=dt).reshape(B * H, Dk, Dv))
    o, SN = recurrence.apply(u, w, kw, egl, qg, attn, S0)
    # (N, BH, C, Dv) -> (B, L, H, Dv)
    o = o.reshape(N, B, H, C, Dv).permute(1, 0, 3, 2, 4).reshape(B, L, H, Dv)[:, :T]
    return o, SN.reshape(B, H, Dk, Dv)


# Version in use (measured 2026-09-27, B=2 T=4096 32 heads dk=dv=128, fp32, alongside a training run, alternating
# repetitions):
#   fallback 2568 ms forward+backward · v3 174 ms (x14.7) · v5 117 ms (x21.9) · v4/v6 (Triton recurrence) slower than v3.
# Selected without touching the code via GDN_FAST_VERSION (e.g. 3 = no torch.compile).
VERSION = int(os.environ.get("GDN_FAST_VERSION", "5"))


def chunk_gated_delta_rule_fast(query, key, value, g, beta, chunk_size=64, initial_state=None,
                                  output_final_state=False, use_qk_l2norm_in_kernel=False, cu_seqlens=None,
                                  _dtype_compute: torch.dtype = torch.float32, _version: Optional[int] = None,
                                  **kwargs):
    """Same signature and same result as transformers torch_chunk_gated_delta_rule. `_dtype_compute` is only for the
    tests (fp64 as the algebraic reference); runs use the default fp32, like the fallback."""
    if cu_seqlens is not None:
        raise NotImplementedError("gdn_fast_v5: cu_seqlens is not supported (runs use row mode: one sequence per batch row)")
    initial_dtype = query.dtype
    ver = _version or VERSION
    if ver == 1:
        if use_qk_l2norm_in_kernel:
            query = l2norm(query, dim=-1, eps=1e-6)
            key = l2norm(key, dim=-1, eps=1e-6)
        o, SN = _core(query, key, value, g, beta, chunk_size, initial_state, _dtype_compute)
        o = o.transpose(1, 2)
    else:
        # v2: hand-written recurrence; v3: in-place loop; v4: Triton recurrence; v5 = v3 + compiled preparation;
        # v6 = v4 + compiled preparation
        if ver == 2:
            recipe = _RecurrenceOutput
        elif ver in (4, 6) and _TRITON and (query.is_cuda or _INTERPRETER) and _dtype_compute == torch.float32 \
                and value.shape[-1] % _BV == 0:
            recipe = _RecurrenceTriton
        else:                                   # v3/v5, or v4/v6 outside the Triton case (fp64 tests, CPU)
            recipe = _RecurrenceOutput3
        o, SN = _core_v2(query, key, value, g, beta, chunk_size, initial_state, _dtype_compute, recipe,
                           l2=use_qk_l2norm_in_kernel, compile_=ver >= 5)
    o = o.contiguous().to(initial_dtype)
    return o, (SN if output_final_state else None)


def install(module=None) -> None:
    """Replaces the fallback in the transformers module (looked up by name on every GatedDeltaNet call)."""
    if module is None:
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as module
    module.torch_chunk_gated_delta_rule = chunk_gated_delta_rule_fast

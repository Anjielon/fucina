#!/usr/bin/env python3
"""GDN di Thor-T: versione veloce del ripiego PyTorch di transformers (torch_chunk_gated_delta_rule), STESSA
matematica in fp32 (27/9/2026).

Perche' e' nato: la corsa E2E passa ~75% del tempo nei 30 strati Gated DeltaNet, serviti dal ripiego di transformers
(fla non gira su gfx1151: page fault in fp32; torch.compile del ripiego: compilazione > 1 ora). Il ripiego e' lento per
ragioni di SCRITTURA, non di matematica:
  1. la soluzione triangolare (I + L)^-1 e' un ciclo Python di 63 passi con due .clone() e un'assegnazione a fetta
     ciascuno: in avanti ~190 kernel, all'indietro ogni CopySlices clona il gradiente dell'intero tensore (63 volte);
  2. la ricorrenza fra chunk indicizza query[:, :, i] & co.: all'indietro ogni SelectBackward alloca un tensore
     PIENO di zeri e ci copia la fetta (N^2 di traffico di memoria), e dentro il ciclo si rifanno per ogni chunk
     prodotti che non dipendono dallo stato (q_i k_i^T, q_i * exp(g_i), k_i * exp(g_last - g_i)).
La versione nuova (le leve si sommano; VERSIONE sotto sceglie quale gira):
  v1  (I + L)^-1 con UNA torch.linalg.solve_triangular (unitriangolare, sostituzione in avanti: stesso algoritmo del
      ciclo); tutto cio' che non dipende dallo stato S per TUTTI i chunk insieme, fuori dal ciclo; la ricorrenza sullo
      stato come torch.autograd.Function con indietro a mano (nel ciclo restano solo le due dipendenze sequenziali
      vere: v_new_i = u_i - w_i S_i ; S_{i+1} = S_i e^{g_last} + kw_i^T v_new_i, e il loro aggiunto); i gradienti di
      u, w, kw, e^{g_last} dopo, batched su tutti i chunk.
  v2  disposizione PER CHUNK (N, B*H, C, .): ogni fetta del ciclo e' contigua; l'uscita o = qg S + attn v_new dentro
      la Function; indietro a mano anche per (I + L)^-1 (dM = -X^T dX X^T: due prodotti invece di due trsm).
  v3  il ciclo IN POSTO (baddbmm_, mul out=, addcmul_): niente copie dispositivo-dispositivo per passo.
  v4  la ricorrenza in due kernel Triton: MISURATA PIU' LENTA di v3 su gfx1151 (tl.dot fp32 senza unita' matriciali):
      resta come alternativa documentata, non in uso.
  v5  v3 + torch.compile (dynamic=False) della sola preparazione senza cicli (_prep): fonde le ~40 operazioni
      elemento per elemento dell'avanti e del loro indietro. Se la compilazione fallisce si torna da soli alla v3.
Nessun cambio di dtype: il calcolo e' fp32 come nel ripiego, TF32 mai acceso (per le prove fp64 con _dtype_calcolo).
Prova: strumenti/prova_gdn_veloce.py (avanti + dq dk dv dg dbeta [+ stato iniziale/finale] contro il riferimento fp64
VERO — copia del ripiego col dtype parametrico — in 4 scenari: normale, decadimento forte, debole, chiavi correlate).

Uso nella corsa (PRIMA del primo avanti; il modulo di transformers cerca il nome a ogni chiamata):
    from strumenti.gdn_veloce import installa
    installa()          # M.torch_chunk_gated_delta_rule = chunk_gated_delta_rule_veloce
In e2e_allena.py: E2E_GDN=veloce. Versione: GDN_VELOCE_VERSIONE (default 5; 3 = senza torch.compile).
"""
import os
from typing import Optional

import torch
import torch.nn.functional as F


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """Identica a transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.l2norm (allineata a fla)."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


class _Ricorrenza(torch.autograd.Function):
    """Ricorrenza fra chunk del gated delta rule, sullo stato S (B, H, Dk, Dv).

    Ingressi (B, H, N, C, .): u = X v_beta, w = X (k_beta e^g), kw = k e^{g_last - g}; egl = e^{g_last} (B, H, N);
    S0 = stato iniziale (B, H, Dk, Dv).
    Per i = 0..N-1:  Sp_i = S_i ;  vn_i = u_i - w_i S_i ;  S_{i+1} = S_i egl_i + kw_i^T vn_i.
    Uscite: vn (B, H, N, C, Dv), Sp (B, H, N, Dk, Dv) = stato PRIMA di ogni chunk, S_N (B, H, Dk, Dv)."""

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
        dvn_tot = torch.empty_like(vn)                       # gradiente totale di vn_i
        dS_dopo = torch.empty_like(Sp)                       # dL/dS_{i+1}, per i gradienti batched di kw ed egl
        for i in range(N - 1, -1, -1):
            dS_dopo[:, :, i] = dS
            # S_{i+1} = S_i egl_i + kw_i^T vn_i  ->  dvn_i += kw_i dS_{i+1}
            dvn_i = d_vn[:, :, i] + torch.matmul(kw[:, :, i], dS)
            dvn_tot[:, :, i] = dvn_i
            # dS_i = dS_{i+1} egl_i - w_i^T dvn_i (+ d_Sp_i, perche' Sp_i = S_i e' un'uscita)
            dS = dS * egl[:, :, i, None, None] - torch.matmul(w[:, :, i].transpose(-1, -2), dvn_i)
            if d_Sp is not None:
                dS = dS + d_Sp[:, :, i]
        du = dvn_tot
        dw = -torch.matmul(dvn_tot, Sp.transpose(-1, -2))                    # vn_i = u_i - w_i S_i
        dkw = torch.matmul(vn, dS_dopo.transpose(-1, -2))                     # (kw_i^T vn_i): d kw_i = vn_i dS^T
        degl = (Sp * dS_dopo).sum((-1, -2))                                   # S_i egl_i
        return du, dw, dkw, degl, dS


def _nucleo(q, k, v, g, beta, chunk_size: int, initial_state: Optional[torch.Tensor], dt: torch.dtype):
    """q, k gia' normalizzati se richiesto; tutti in (B, T, H, .) come li passa transformers."""
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

    triu0 = torch.triu(torch.ones(C, C, dtype=torch.bool, device=q.device), diagonal=0)    # diagonale compresa
    triu1 = torch.triu(torch.ones(C, C, dtype=torch.bool, device=q.device), diagonal=1)
    # decadimento intra-chunk (i >= j): exp(g_i - g_j); sopra la diagonale 0 (come .tril().exp().tril() del ripiego)
    decay = (g.unsqueeze(-1) - g.unsqueeze(-2)).masked_fill(triu1, float("-inf")).exp()
    Lm = ((k_beta @ k.transpose(-1, -2)) * decay).masked_fill(triu0, 0)       # strettamente triangolare inferiore
    eye = torch.eye(C, dtype=dt, device=q.device)
    # X = (I + Lm)^-1 : sostituzione in avanti unitriangolare (il ripiego fa la stessa, riga per riga in Python)
    X = torch.linalg.solve_triangular(Lm, eye.expand(B, H, N, C, C), upper=False, unitriangular=True)
    eg = g.exp()
    u = X @ v_beta                                   # "value" del ripiego
    w = X @ (k_beta * eg.unsqueeze(-1))              # "k_cumdecay" del ripiego
    attn = (q @ k.transpose(-1, -2)) * decay         # intra-chunk, triangolare inferiore con diagonale
    qg = q * eg.unsqueeze(-1)
    gl = g[..., -1]                                  # (B, H, N)
    kw = k * (gl.unsqueeze(-1) - g).exp().unsqueeze(-1)
    egl = gl.exp()
    S0 = (torch.zeros(B, H, Dk, Dv, dtype=dt, device=q.device) if initial_state is None
          else initial_state.to(device=q.device, dtype=dt))
    vn, Sp, SN = _Ricorrenza.apply(u.contiguous(), w.contiguous(), kw.contiguous(), egl.contiguous(), S0)
    o = qg @ Sp + attn @ vn                          # (B, H, N, C, Dv)
    o = o.reshape(B, H, L, Dv)[:, :, :T]
    return o, SN


# ───────────────────────────── versione 2: disposizione per chunk, indietro a mano anche per (I + L)^-1 ─────────────
class _Inversa(torch.autograd.Function):
    """X = (I + Lm)^-1, Lm strettamente triangolare inferiore (C, C) in lotto. Avanti: sostituzione in avanti
    unitriangolare (solve_triangular). Indietro: X = M^-1 -> dM = -X^T dX X^T, ristretto alla parte strettamente
    inferiore (l'unica che Lm porta): due prodotti invece delle due risoluzioni triangolari dell'indietro generico."""

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


class _RicorrenzaUscita(torch.autograd.Function):
    """Ricorrenza fra chunk + uscita, in disposizione PER CHUNK (N, BH, C, .): ogni fetta del ciclo e' contigua.
    Per i = 0..N-1:  vn_i = u_i - w_i S_i ;  S_{i+1} = S_i egl_i + kw_i^T vn_i ;  o_i = qg_i S_i + attn_i vn_i.
    Ingressi: u, w, kw, qg, attn (N, BH, C, .), egl (N, BH), S0 (BH, Dk, Dv). Uscite: o (N, BH, C, Dv), S_N."""

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
        d_vn = torch.matmul(attn.transpose(-1, -2), d_o)                                # parte diretta (o)
        d_Sp = torch.matmul(qg.transpose(-1, -2), d_o)
        dSa = torch.empty_like(Sa)                                                       # dSa[i] = dL/dS_i totale
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
        dS_dopo = dSa[1:]
        dw = torch.matmul(dvn, Sp.transpose(-1, -2)).neg_()
        dkw = torch.matmul(vn, dS_dopo.transpose(-1, -2))
        degl = (Sp * dS_dopo).sum((-1, -2))
        return dvn, dw, dkw, degl, d_qg, d_attn, dSa[0]


class _RicorrenzaUscita3(torch.autograd.Function):
    """Come _RicorrenzaUscita, ma il ciclo lavora IN POSTO (baddbmm_ / mul out= / addcmul_): nessuna copia
    dispositivo-dispositivo per passo (la forma out= di baddbmm copia prima l'ingresso nell'uscita: 256 copie
    per chiamata nel profilo della v2) e nessun temporaneo per lo stato."""

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
        dvn = torch.matmul(attn.transpose(-1, -2), d_o)                                 # parte diretta, poi in posto
        dSa = torch.empty_like(Sa)                                                        # dSa[i] = dL/dS_i totale
        torch.matmul(qg.transpose(-1, -2), d_o, out=dSa[:N])                             # parte diretta (o_i = qg_i S_i)
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
        dS_dopo = dSa[1:]
        dw = torch.matmul(dvn, Sp.transpose(-1, -2)).neg_()
        dkw = torch.matmul(vn, dS_dopo.transpose(-1, -2))
        degl = (Sp * dS_dopo).sum((-1, -2))
        return dvn, dw, dkw, degl, d_qg, d_attn, dSa[0]


# ───────────────────── versione 4: la ricorrenza in DUE kernel Triton (avanti, indietro) ─────────────────────
# Le colonne di S (indice di valore) sono INDIPENDENTI nella ricorrenza: vn_i[:, c] = u_i[:, c] - w_i S_i[:, c],
# S_{i+1}[:, c] = S_i[:, c] egl_i + kw_i^T vn_i[:, c]; lo stesso nell'aggiunto. Un programma = (testa bh, blocco di
# BV colonne) e percorre i 64 chunk IN SEQUENZA tenendo il suo pezzo di S nei registri: 1 lancio invece di 192
# (avanti) e di 192 (indietro), e i prodotti restano in fp32 (tl.dot input_precision="ieee": niente tf32).
try:
    import triton
    import triton.language as tl
    _TRITON = True
except Exception:                      # pragma: no cover - senza Triton resta la v3
    _TRITON = False

if _TRITON:
    @triton.jit
    def _k_ric_avanti(u_ptr, w_ptr, kw_ptr, egl_ptr, s0_ptr, vn_ptr, sa_ptr,
                      BH, N: tl.constexpr,
                      C: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr):
        """Per i = 0..N-1: Sa[i] = S; vn_i = u_i - w_i S; S = S egl_i + kw_i^T vn_i. Alla fine Sa[N] = S.
        Disposizione per chunk, contigua: u/vn (N, BH, C, DV), w/kw (N, BH, C, DK), egl (N, BH),
        s0 (BH, DK, DV), sa (N + 1, BH, DK, DV)."""
        bh = tl.program_id(0).to(tl.int64)
        iv = tl.program_id(1)
        rc = tl.arange(0, C)
        rk = tl.arange(0, DK)
        rv = iv * BV + tl.arange(0, BV)
        off_s = rk[:, None] * DV + rv[None, :]                         # (DK, BV) dentro una matrice di stato
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
    def _k_ric_indietro(dvn_ptr, dsa_ptr, w_ptr, kw_ptr, egl_ptr,
                        BH, N: tl.constexpr,
                        C: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr):
        """Aggiunto, IN POSTO. Ingresso: dvn = parte diretta di dL/dvn (attn^T d_o), dsa[:N] = parte diretta di
        dL/dS_i (qg^T d_o), dsa[N] = dL/dS_N. Per i = N-1..0: dvn_i += kw_i dS_{i+1};
        dS_i = dS_{i+1} egl_i + dsa[i] - w_i^T dvn_i. Uscita: dvn totale, dsa[i] = dL/dS_i totale."""
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


_INTERPRETE = os.environ.get("TRITON_INTERPRET") == "1"      # prove dei kernel su CPU (interprete di Triton)
_BV = 32          # colonne di stato per programma: griglia (B*H, Dv/_BV) = 64 x 4 alle dimensioni di Thor
_NW = 8           # warp per programma


class _RicorrenzaTriton(torch.autograd.Function):
    """Stessa matematica di _RicorrenzaUscita3; il ciclo sui chunk e' dentro due kernel Triton."""

    @staticmethod
    def forward(ctx, u, w, kw, egl, qg, attn, S0):
        N, BH, C, Dv = u.shape
        Dk = w.shape[-1]
        assert Dv % _BV == 0 and u.dtype == torch.float32
        u, w, kw, egl = (t.contiguous() for t in (u, w, kw, egl))
        Sa = torch.empty(N + 1, BH, Dk, Dv, dtype=u.dtype, device=u.device)
        vn = torch.empty_like(u)
        _k_ric_avanti[(BH, Dv // _BV)](u, w, kw, egl, S0.contiguous(), vn, Sa, BH, N=N, C=C, DK=Dk, DV=Dv, BV=_BV,
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
        _k_ric_indietro[(BH, Dv // _BV)](dvn, dSa, w, kw, egl, BH, N=N, C=C, DK=Dk, DV=Dv, BV=_BV, num_warps=_NW)
        dS_dopo = dSa[1:]
        dw = torch.matmul(dvn, Sp.transpose(-1, -2)).neg_()
        dkw = torch.matmul(vn, dS_dopo.transpose(-1, -2))
        degl = (Sp * dS_dopo).sum((-1, -2))
        return dvn, dw, dkw, degl, d_qg, d_attn, dSa[0]


def _in_chunk(x: torch.Tensor, N: int, C: int) -> torch.Tensor:
    """(B, L, H, ...) -> (N, B*H, C, ...) contiguo."""
    B, L, H = x.shape[:3]
    resto = x.shape[3:]
    x = x.reshape(B, N, C, H, *resto)
    perm = (1, 0, 3, 2) + tuple(range(4, x.dim()))
    return x.permute(*perm).reshape(N, B * H, C, *resto)


def _prep(q, k, v, g, beta, C: int, N: int, l2: bool):
    """Tutto cio' che non dipende dallo stato, per TUTTI i chunk insieme (disposizione per chunk (N, B*H, C, .)).
    Funzione pura senza cicli: nella v5 la compila torch.compile (fonde le operazioni elemento per elemento)."""
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


_PREP_COMPILATA = None
_COMPILAZIONE_FALLITA = False


def _prep_compilata():
    global _PREP_COMPILATA
    if _PREP_COMPILATA is None:
        _PREP_COMPILATA = torch.compile(_prep, dynamic=False)
    return _PREP_COMPILATA


def _prep_scelta(compila: bool):
    """La preparazione compilata (v5) se richiesta e se la compilazione non e' gia' fallita in questo processo;
    altrimenti la stessa funzione in eager (v3): stessa matematica, solo piu' kernel."""
    if not compila or _COMPILAZIONE_FALLITA:
        return _prep
    comp = _prep_compilata()

    def f(*a):
        global _COMPILAZIONE_FALLITA
        try:
            return comp(*a)
        except Exception as e:                  # errore di compilazione (inductor/Triton): si resta sulla v3, a voce alta
            _COMPILAZIONE_FALLITA = True
            import warnings
            warnings.warn(f"gdn_veloce: torch.compile della preparazione fallita ({type(e).__name__}: {e}); "
                          f"uso la preparazione eager (v3)")
            return _prep(*a)
    return f


def _nucleo_v2(q, k, v, g, beta, chunk_size: int, initial_state: Optional[torch.Tensor], dt: torch.dtype,
               ricorrenza=_RicorrenzaUscita, l2: bool = False, compila: bool = False):
    """Come _nucleo, ma in disposizione per chunk (N, B*H, C, .) e con le Function dall'indietro a mano."""
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
    Lm, v_beta, kbeg, attn, qg, kw, egl = _prep_scelta(compila)(q, k, v, g, beta, C, N, l2)
    X = _Inversa.apply(Lm)
    u = X @ v_beta
    w = X @ kbeg
    S0 = (torch.zeros(B * H, Dk, Dv, dtype=dt, device=q.device) if initial_state is None
          else initial_state.to(device=q.device, dtype=dt).reshape(B * H, Dk, Dv))
    o, SN = ricorrenza.apply(u, w, kw, egl, qg, attn, S0)
    # (N, BH, C, Dv) -> (B, L, H, Dv)
    o = o.reshape(N, B, H, C, Dv).permute(1, 0, 3, 2, 4).reshape(B, L, H, Dv)[:, :T]
    return o, SN.reshape(B, H, Dk, Dv)


# Versione in uso (misure del 27/9, B=2 T=4096 32 teste dk=dv=128, fp32, accanto alla corsa EB2, ripetizioni alternate):
#   ripiego 2568 ms avanti+indietro · v3 174 ms (x14.7) · v5 117 ms (x21.9) · v4/v6 (ricorrenza Triton) piu' lente di v3.
# Si sceglie senza toccare il codice con GDN_VELOCE_VERSIONE (es. 3 = niente torch.compile).
VERSIONE = int(os.environ.get("GDN_VELOCE_VERSIONE", "5"))


def chunk_gated_delta_rule_veloce(query, key, value, g, beta, chunk_size=64, initial_state=None,
                                  output_final_state=False, use_qk_l2norm_in_kernel=False, cu_seqlens=None,
                                  _dtype_calcolo: torch.dtype = torch.float32, _versione: Optional[int] = None,
                                  **kwargs):
    """Stessa firma e stesso risultato di transformers torch_chunk_gated_delta_rule. `_dtype_calcolo` solo per le prove
    (fp64 come riferimento algebrico); la corsa usa il default fp32, come il ripiego."""
    if cu_seqlens is not None:
        raise NotImplementedError("gdn_veloce: cu_seqlens non supportato (la corsa usa il modo riga)")
    initial_dtype = query.dtype
    ver = _versione or VERSIONE
    if ver == 1:
        if use_qk_l2norm_in_kernel:
            query = l2norm(query, dim=-1, eps=1e-6)
            key = l2norm(key, dim=-1, eps=1e-6)
        o, SN = _nucleo(query, key, value, g, beta, chunk_size, initial_state, _dtype_calcolo)
        o = o.transpose(1, 2)
    else:
        # v2: ricorrenza a mano; v3: ciclo in posto; v4: ricorrenza Triton; v5 = v3 + preparazione compilata;
        # v6 = v4 + preparazione compilata
        if ver == 2:
            ric = _RicorrenzaUscita
        elif ver in (4, 6) and _TRITON and (query.is_cuda or _INTERPRETE) and _dtype_calcolo == torch.float32 \
                and value.shape[-1] % _BV == 0:
            ric = _RicorrenzaTriton
        else:                                   # v3/v5, oppure v4/v6 fuori dal caso Triton (fp64 delle prove, CPU)
            ric = _RicorrenzaUscita3
        o, SN = _nucleo_v2(query, key, value, g, beta, chunk_size, initial_state, _dtype_calcolo, ric,
                           l2=use_qk_l2norm_in_kernel, compila=ver >= 5)
    o = o.contiguous().to(initial_dtype)
    return o, (SN if output_final_state else None)


def installa(modulo=None) -> None:
    """Sostituisce il ripiego nel modulo di transformers (cercato per nome a ogni chiamata del GatedDeltaNet)."""
    if modulo is None:
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as modulo
    modulo.torch_chunk_gated_delta_rule = chunk_gated_delta_rule_veloce

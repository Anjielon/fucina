"""⛔ SCARTATA (2/10, misura GPU): NON cablata nel percorso (e2e_indietro_hilo e' quello di sempre). Sulla GPU, a
8192 token e 256 esperti, il lancio fuso e' 3-4x PIU' LENTO dei due lanci (gate 434 -> 1659 ms dX+dd, down 279 -> 935)
e il dL/dd NON e' al bit (relativo 1e-7: col riquadro di righe 32 invece di 64 il compilatore accumula diversamente);
il dX invece e' al bit. Resta come prova negativa (e2e_test_hilo_fuso.py).

2/10 TT6-veloce — indietro DIVISO fp16x2 con le due meta' (hi, lo) NELLO STESSO kernel (E2E_HILO_FUSO=1; spento =
`e2e_indietro_hilo` di sempre, due lanci per prodotto).

Oggi (`e2e_indietro_hilo.indietro_famiglia`, via fp16x2, motore triton2, fuori dal passo V) per OGNI famiglia:
  dX:  s = 2^-e(max|g| per riga); hi, lo = fp16(g s), fp16(g s - hi);  y = dx2(lo); y += dx2(hi); y *= 2^e; y -> fp16
  dd:  s = 2^-e(max|g| per esperto x riga); hi, lo come sopra;          d = dd2(lo); d += dd2(hi); d *= 2^e
cioe' 4 lanci (2 dX + 2 dd), 2 decodifiche dei trit per il dX, X letto due volte per il dd, e in memoria g s, hi, lo,
la raccolta s[eid] (S x R fp32), le somme y += / d += e le moltiplicazioni per 2^e.
Qui: UN lancio per il dX e UNO per il dd. Il riquadro carica g (fp32) e la sua scala, divide hi/lo nei registri
(stesse operazioni: fp32 * 2^-e esatto, RNE a fp16, sottrazione fp32, RNE a fp16), e porta DUE serie di
accumulatori (lo, hi) che vedono gli STESSI prodotti nello STESSO ordine lungo K (stesso BK = 32 per il dX, stesso
BS = 32 di token per il dd) dei due lanci di oggi; l'epilogo fa (lo + hi) * 2^e come oggi (stessa somma fp32, stesso
prodotto esatto) e per il dd le somme per gruppo di trit (.) acc separate per lo e hi come i due lanci di oggi.
Per tenere i registri di oggi il riquadro dei token del dX scende da 32 a 16 e quello delle righe del dd da 64 a 32:
non toccano l'ordine delle somme di nessun elemento (righe/token indipendenti).
=> AL BIT con la via di oggi (prove: e2e_test_hilo_fuso.py, CPU interprete + GPU sullo strato vero)."""
from __future__ import annotations

import os

import torch

import e2e_studente as S

try:
    import triton
    import triton.language as tl
    from e2e_studente import _w_pezzo, _t_pezzo, _schema_sottoblocco, _ordine_gruppi
    _TRITON = True
except Exception:  # pragma: no cover
    _TRITON = False

CONTI = {"fuso": 0}
BS_DX = 16
BR_DD = 32


def attivo() -> bool:
    return os.environ.get("E2E_HILO_FUSO", "").strip() == "1"


if _TRITON:
    @triton.jit
    def _k_dx2_hl(g_ptr, sg_s, sc_ptr, iv_ptr, pb, sb_e, NBR, d_ptr, sd_e, NG, o_ptr, so_s,
                  te_ptr, ts_ptr, offs_ptr, R, C, NT,
                  GS: tl.constexpr, QK: tl.constexpr, NB: tl.constexpr, ROT: tl.constexpr,
                  BS: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr):
        """`_k_dx2` con A = g fp32 diviso nei registri in (hi, lo) fp16 e due serie di accumulatori."""
        pid_t, pid_n = _ordine_gruppi(tl.program_id(0), NT, C // 256, GROUP_M)
        e = tl.load(te_ptr + pid_t)
        if e < 0:
            return
        s0 = tl.load(ts_ptr + pid_t)
        s_fine = tl.load(offs_ptr + e + 1)
        e64 = e.to(tl.int64)
        ls = tl.arange(0, BS)
        ms = (s0 + ls) < s_fine
        lk = tl.arange(0, BK)
        c0 = pid_n * 256
        qs, qh = _schema_sottoblocco(c0, QK, NB, ROT)
        i = tl.arange(0, 32)
        offr = (lk * NBR)[:, None]
        off0 = offr + (qs + i)[None, :]
        off5 = offr + (qs + 32 + (i % 16))[None, :]
        off7 = offr + tl.where(i < 16, qs + 32 + i, qh + (i % 4))[None, :]
        offd = lk * NG
        offa = (ls * sg_s)[:, None] + lk[None, :]
        pw5 = tl.where(i < 16, 1, 3)
        pw6 = tl.where(i < 16, 9, 27)
        pw7 = tl.where(i < 16, 81, tl.where(i < 20, 1, tl.where(i < 24, 3, tl.where(i < 28, 9, 27))))
        g = c0 // GS
        pe = pb + e64 * sb_e
        pde = d_ptr + e64 * sd_e + g
        pa = g_ptr + s0.to(tl.int64) * sg_s
        sc = tl.load(sc_ptr + s0 + ls, mask=ms, other=1.0)[:, None]
        iv = tl.load(iv_ptr + s0 + ls, mask=ms, other=1.0)[:, None]
        h0 = tl.zeros((BS, 32), dtype=tl.float32)
        l0 = tl.zeros((BS, 32), dtype=tl.float32)
        h1 = tl.zeros((BS, 32), dtype=tl.float32)
        l1 = tl.zeros((BS, 32), dtype=tl.float32)
        h2 = tl.zeros((BS, 32), dtype=tl.float32)
        l2 = tl.zeros((BS, 32), dtype=tl.float32)
        h3 = tl.zeros((BS, 32), dtype=tl.float32)
        l3 = tl.zeros((BS, 32), dtype=tl.float32)
        h4 = tl.zeros((BS, 32), dtype=tl.float32)
        l4 = tl.zeros((BS, 32), dtype=tl.float32)
        h5 = tl.zeros((BS, 32), dtype=tl.float32)
        l5 = tl.zeros((BS, 32), dtype=tl.float32)
        h6 = tl.zeros((BS, 32), dtype=tl.float32)
        l6 = tl.zeros((BS, 32), dtype=tl.float32)
        h7 = tl.zeros((BS, 32), dtype=tl.float32)
        l7 = tl.zeros((BS, 32), dtype=tl.float32)
        for r0 in range(0, R, BK):
            mr = (r0 + lk) < R
            prow = pe + r0 * NBR
            m2 = mr[:, None]
            v0 = tl.load(prow + off0, mask=m2, other=0).to(tl.int32)
            v5 = tl.load(prow + off5, mask=m2, other=0).to(tl.int32)
            v7 = tl.load(prow + off7, mask=m2, other=0).to(tl.int32)
            pdr = pde + r0 * NG
            dA = tl.load(pdr + offd, mask=mr, other=0.0).to(tl.float16)
            if GS == 128:
                dB = tl.load(pdr + 1 + offd, mask=mr, other=0.0).to(tl.float16)
            else:
                dB = dA
            v = tl.load(pa + r0 + offa, mask=ms[:, None] & mr[None, :], other=0.0) * sc
            hi = v.to(tl.float16)
            lo = (v - hi.to(tl.float32)).to(tl.float16)
            w = _w_pezzo(v0, v5, v7, dA, dB, pw5, pw6, pw7, 0, GS, 0, tl.float16)
            l0 += tl.dot(lo, w)
            h0 += tl.dot(hi, w)
            w = _w_pezzo(v0, v5, v7, dA, dB, pw5, pw6, pw7, 1, GS, 0, tl.float16)
            l1 += tl.dot(lo, w)
            h1 += tl.dot(hi, w)
            w = _w_pezzo(v0, v5, v7, dA, dB, pw5, pw6, pw7, 2, GS, 0, tl.float16)
            l2 += tl.dot(lo, w)
            h2 += tl.dot(hi, w)
            w = _w_pezzo(v0, v5, v7, dA, dB, pw5, pw6, pw7, 3, GS, 0, tl.float16)
            l3 += tl.dot(lo, w)
            h3 += tl.dot(hi, w)
            w = _w_pezzo(v0, v5, v7, dA, dB, pw5, pw6, pw7, 4, GS, 0, tl.float16)
            l4 += tl.dot(lo, w)
            h4 += tl.dot(hi, w)
            w = _w_pezzo(v0, v5, v7, dA, dB, pw5, pw6, pw7, 5, GS, 0, tl.float16)
            l5 += tl.dot(lo, w)
            h5 += tl.dot(hi, w)
            w = _w_pezzo(v0, v5, v7, dA, dB, pw5, pw6, pw7, 6, GS, 0, tl.float16)
            l6 += tl.dot(lo, w)
            h6 += tl.dot(hi, w)
            w = _w_pezzo(v0, v5, v7, dA, dB, pw5, pw6, pw7, 7, GS, 0, tl.float16)
            l7 += tl.dot(lo, w)
            h7 += tl.dot(hi, w)
        OT = o_ptr.dtype.element_ty
        po = o_ptr + s0.to(tl.int64) * so_s + c0
        offo = (ls * so_s)[:, None] + i[None, :]
        mo = ms[:, None]
        tl.store(po + 0 + offo, ((l0 + h0) * iv).to(OT), mask=mo)
        tl.store(po + 32 + offo, ((l1 + h1) * iv).to(OT), mask=mo)
        tl.store(po + 64 + offo, ((l2 + h2) * iv).to(OT), mask=mo)
        tl.store(po + 96 + offo, ((l3 + h3) * iv).to(OT), mask=mo)
        tl.store(po + 128 + offo, ((l4 + h4) * iv).to(OT), mask=mo)
        tl.store(po + 160 + offo, ((l5 + h5) * iv).to(OT), mask=mo)
        tl.store(po + 192 + offo, ((l6 + h6) * iv).to(OT), mask=mo)
        tl.store(po + 224 + offo, ((l7 + h7) * iv).to(OT), mask=mo)

    @triton.jit
    def _k_dd2_hl(g_ptr, sg_s, sc_ptr, iv_ptr, x_ptr, sx_s, pb, sb_e, NBR, dd_ptr, sdd_e, NG, offs_ptr, R, C,
                  GS: tl.constexpr, QK: tl.constexpr, NB: tl.constexpr, ROT: tl.constexpr,
                  BR: tl.constexpr, BS: tl.constexpr):
        """`_k_dd2` con dY = g fp32 diviso nei registri (scala per esperto x riga) e due serie di accumulatori."""
        e = tl.program_id(0)
        pid_r = tl.program_id(1)
        c0 = tl.program_id(2) * 256
        s_ini = tl.load(offs_ptr + e)
        s_fine = tl.load(offs_ptr + e + 1)
        r0 = pid_r * BR
        lr = tl.arange(0, BR)
        mr = (r0 + lr) < R
        ls = tl.arange(0, BS)
        i = tl.arange(0, 32)
        offy = (ls * sg_s)[:, None] + lr[None, :]
        offx = (ls * sx_s)[:, None] + i[None, :]
        pdy = g_ptr + r0
        px = x_ptr + c0
        e64 = e.to(tl.int64)
        sc = tl.load(sc_ptr + e64 * R + r0 + lr, mask=mr, other=1.0)[None, :]
        h0 = tl.zeros((BR, 32), dtype=tl.float32)
        l0 = tl.zeros((BR, 32), dtype=tl.float32)
        h1 = tl.zeros((BR, 32), dtype=tl.float32)
        l1 = tl.zeros((BR, 32), dtype=tl.float32)
        h2 = tl.zeros((BR, 32), dtype=tl.float32)
        l2 = tl.zeros((BR, 32), dtype=tl.float32)
        h3 = tl.zeros((BR, 32), dtype=tl.float32)
        l3 = tl.zeros((BR, 32), dtype=tl.float32)
        h4 = tl.zeros((BR, 32), dtype=tl.float32)
        l4 = tl.zeros((BR, 32), dtype=tl.float32)
        h5 = tl.zeros((BR, 32), dtype=tl.float32)
        l5 = tl.zeros((BR, 32), dtype=tl.float32)
        h6 = tl.zeros((BR, 32), dtype=tl.float32)
        l6 = tl.zeros((BR, 32), dtype=tl.float32)
        h7 = tl.zeros((BR, 32), dtype=tl.float32)
        l7 = tl.zeros((BR, 32), dtype=tl.float32)
        for s0 in range(s_ini, s_fine, BS):
            ms = (s0 + ls) < s_fine
            s64 = s_ini.to(tl.int64) + (s0 - s_ini)
            v = tl.load(pdy + s64 * sg_s + offy, mask=ms[:, None] & mr[None, :], other=0.0) * sc
            hi = v.to(tl.float16)
            lo = (v - hi.to(tl.float32)).to(tl.float16)
            hit = tl.trans(hi)
            lot = tl.trans(lo)
            pxs = px + s64 * sx_s + offx
            mx = ms[:, None]
            xx = tl.load(pxs + 0, mask=mx, other=0.0)
            l0 += tl.dot(lot, xx)
            h0 += tl.dot(hit, xx)
            xx = tl.load(pxs + 32, mask=mx, other=0.0)
            l1 += tl.dot(lot, xx)
            h1 += tl.dot(hit, xx)
            xx = tl.load(pxs + 64, mask=mx, other=0.0)
            l2 += tl.dot(lot, xx)
            h2 += tl.dot(hit, xx)
            xx = tl.load(pxs + 96, mask=mx, other=0.0)
            l3 += tl.dot(lot, xx)
            h3 += tl.dot(hit, xx)
            xx = tl.load(pxs + 128, mask=mx, other=0.0)
            l4 += tl.dot(lot, xx)
            h4 += tl.dot(hit, xx)
            xx = tl.load(pxs + 160, mask=mx, other=0.0)
            l5 += tl.dot(lot, xx)
            h5 += tl.dot(hit, xx)
            xx = tl.load(pxs + 192, mask=mx, other=0.0)
            l6 += tl.dot(lot, xx)
            h6 += tl.dot(hit, xx)
            xx = tl.load(pxs + 224, mask=mx, other=0.0)
            l7 += tl.dot(lot, xx)
            h7 += tl.dot(hit, xx)
        qs, qh = _schema_sottoblocco(c0, QK, NB, ROT)
        pw5 = tl.where(i < 16, 1, 3)
        pw6 = tl.where(i < 16, 9, 27)
        pw7 = tl.where(i < 16, 81, tl.where(i < 20, 1, tl.where(i < 24, 3, tl.where(i < 28, 9, 27))))
        prow = pb + e64 * sb_e + r0.to(tl.int64) * NBR
        offr = (lr * NBR)[:, None]
        m2 = mr[:, None]
        v0 = tl.load(prow + offr + (qs + i)[None, :], mask=m2, other=0).to(tl.int32)
        v5 = tl.load(prow + offr + (qs + 32 + (i % 16))[None, :], mask=m2, other=0).to(tl.int32)
        v7 = tl.load(prow + offr + tl.where(i < 16, qs + 32 + i, qh + (i % 4))[None, :], mask=m2,
                     other=0).to(tl.int32)
        L0 = l0 * _t_pezzo(v0, v5, v7, pw5, pw6, pw7, 0, 0).to(tl.float32)
        L0 += l1 * _t_pezzo(v0, v5, v7, pw5, pw6, pw7, 1, 0).to(tl.float32)
        L0 += l2 * _t_pezzo(v0, v5, v7, pw5, pw6, pw7, 2, 0).to(tl.float32)
        L0 += l3 * _t_pezzo(v0, v5, v7, pw5, pw6, pw7, 3, 0).to(tl.float32)
        L1 = l4 * _t_pezzo(v0, v5, v7, pw5, pw6, pw7, 4, 0).to(tl.float32)
        L1 += l5 * _t_pezzo(v0, v5, v7, pw5, pw6, pw7, 5, 0).to(tl.float32)
        L1 += l6 * _t_pezzo(v0, v5, v7, pw5, pw6, pw7, 6, 0).to(tl.float32)
        L1 += l7 * _t_pezzo(v0, v5, v7, pw5, pw6, pw7, 7, 0).to(tl.float32)
        H0 = h0 * _t_pezzo(v0, v5, v7, pw5, pw6, pw7, 0, 0).to(tl.float32)
        H0 += h1 * _t_pezzo(v0, v5, v7, pw5, pw6, pw7, 1, 0).to(tl.float32)
        H0 += h2 * _t_pezzo(v0, v5, v7, pw5, pw6, pw7, 2, 0).to(tl.float32)
        H0 += h3 * _t_pezzo(v0, v5, v7, pw5, pw6, pw7, 3, 0).to(tl.float32)
        H1 = h4 * _t_pezzo(v0, v5, v7, pw5, pw6, pw7, 4, 0).to(tl.float32)
        H1 += h5 * _t_pezzo(v0, v5, v7, pw5, pw6, pw7, 5, 0).to(tl.float32)
        H1 += h6 * _t_pezzo(v0, v5, v7, pw5, pw6, pw7, 6, 0).to(tl.float32)
        H1 += h7 * _t_pezzo(v0, v5, v7, pw5, pw6, pw7, 7, 0).to(tl.float32)
        g = c0 // GS
        iv = tl.load(iv_ptr + e64 * R + r0 + lr, mask=mr, other=1.0)
        pdd = dd_ptr + e64 * sdd_e + r0.to(tl.int64) * NG + g + lr * NG
        if GS == 128:
            tl.store(pdd, (tl.sum(L0, axis=1) + tl.sum(H0, axis=1)) * iv, mask=mr)
            tl.store(pdd + 1, (tl.sum(L1, axis=1) + tl.sum(H1, axis=1)) * iv, mask=mr)
        else:
            tl.store(pdd, (tl.sum(L0 + L1, axis=1) + tl.sum(H0 + H1, axis=1)) * iv, mask=mr)


def dx_fuso(g32: torch.Tensor, d: torch.Tensor, fam, piano, blk, out_dtype) -> torch.Tensor:
    """dX = fp((dx2(lo) + dx2(hi)) * 2^e) con scala per riga, in UN lancio. g32 (S, R) fp32 contiguo."""
    import e2e_indietro_hilo as HL
    s, inv = HL.scala_pot2(g32.abs().amax(1))
    S._controlla_v2(fam)
    d = d.contiguous()
    out = torch.empty(g32.shape[0], fam.C, dtype=out_dtype, device=g32.device)
    te, ts, n_t = piano.tabella(BS_DX)
    _k_dx2_hl[(n_t * (fam.C // 256),)](g32, g32.stride(0), s.contiguous(), inv.contiguous(), fam.blocchi,
                                       fam.blocchi.stride(0), fam.NBR, d, d.stride(0), fam.NG, out, out.stride(0),
                                       te, ts, piano.offs, fam.R, fam.C, n_t, fam.gs, fam.qk, fam.nb,
                                       int(fam.ruotata), BS_DX, blk.BK_dx2, blk.GROUP_M, num_warps=blk.num_warps2,
                                       **S._stadi(blk.stadi_dx2))
    CONTI["fuso"] += 1
    return out


def dd_fuso(g32: torch.Tensor, x: torch.Tensor, fam, piano, blk) -> torch.Tensor:
    """dL/dd = (dd2(lo) + dd2(hi)) * 2^e con scala per esperto x riga, in UN lancio. -> (E, R, NG) fp32."""
    import e2e_indietro_hilo as HL
    eid = HL.esperto_di_riga(piano, g32.shape[0])
    s_er, inv_er = HL.scala_pot2(HL.massimi_segmento(g32.abs(), eid, fam.E))
    S._controlla_v2(fam)
    x = x.contiguous()
    dd = torch.empty(fam.E, fam.R, fam.NG, dtype=torch.float32, device=g32.device)
    griglia = (fam.E, triton.cdiv(fam.R, BR_DD), fam.C // 256)
    _k_dd2_hl[griglia](g32, g32.stride(0), s_er.contiguous(), inv_er.contiguous(), x, x.stride(0), fam.blocchi,
                       fam.blocchi.stride(0), fam.NBR, dd, dd.stride(0), fam.NG, piano.offs, fam.R, fam.C,
                       fam.gs, fam.qk, fam.nb, int(fam.ruotata), BR_DD, blk.BS_dd2, num_warps=blk.num_warps2,
                       **S._stadi(blk.stadi_dd2))
    CONTI["fuso"] += 1
    return dd


def ammesso(mot, fam, x) -> bool:
    """Solo dove la via di oggi e' quella dei due lanci triton2 fuori dal passo V, con gli stessi riquadri lungo K."""
    import e2e_indietro_hilo as HL
    blk = getattr(mot, "blk", None)
    return (_TRITON and attivo() and getattr(mot, "v2", False) and not getattr(mot, "denso", False)
            and HL._modo_v(mot) == 0 and blk is not None and blk.BK_dx2 == 32 and blk.BS_dd2 == 32
            and x.dtype == torch.float16 and (x.is_cuda or os.environ.get("TRITON_INTERPRET") == "1"))

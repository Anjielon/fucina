#!/usr/bin/env python3
"""e2e_densi_piegati — densi ternari a PIU' piani «piegati» in un solo peso effettivo (1/10/2026, ramo densi-piegati
da TT5-ottimizzato c333d5e). File NUOVO. Interruttore: E2E_DENSI_PIEGATI = "" (default: SPENTO, nulla cambia),
"x2" (peso effettivo in coppia fp16 hi+lo, prodotti fp16 con accumulo fp32) oppure "f32" (peso effettivo fp32,
prodotti fp32). Progetto e conti: ~/tony-forgia/thor_v2/ottimizzazioni_tt5/PIEGATI.md (cura #3 dell'AUDIT, idea 2
della LETTERATURA).

## Perche'
Ogni denso di TT5 (`LineareTernaria` con nome2/nome3) e' W = T1*d1 + T2*d2 + T3*d3 (trit T_k in {-1,0,1}, scale per
gruppo d_k fp16). Oggi lo si calcola come TRE `FamigliaEsperti` sullo stesso ingresso: tre prodotti in avanti, tre dX,
tre dL/dd (e in B3_ind32 i sei dell'indietro sono fp32 = FMA, niente WMMA su gfx1151). Ma
    Y   = X W^T                      con W = somma_k W_k            -> UN prodotto
    dX  = dY W                                                      -> UN prodotto
    G   = dY^T X       (uguale per tutti i piani: stesso X, stesso dY) -> UN prodotto
    dL/dd_k[r, g] = somma_{c in g} T_k[r, c] G[r, c]                -> epilogo elementare, nessun prodotto
e le statistiche del passo V (proposta, verifica: funzioni di G = dY^T X; curvatura: di (dY^2)^T (X^2), anch'essa
uguale per tutti i piani) si ricavano dallo STESSO G con le funzioni di riferimento di e2e_pv_codici.

## Rappresentazione «x2» (fp32-accurata con prodotti fp16)
Un SOLO peso fp16 non basta: W ha fino a ~11 + log2(d1/d3) bit e fp16(W) sbaglia di 2^-11 relativo (≈ 5e-4), cento
volte l'errore di un prodotto fp32. Si usa la coppia (Henry et al. 1904.06376 / Ootomo-Yokota 2203.03341, come
`e2e_indietro_hilo`):
    Wf = fl32(fl32(W1 + W2) + W3)                      (|Wf - W| <= 2^-23 * somma_k |W_k|, due arrotondamenti fp32)
    riga r scalata per potenza di due: a_r = 2^-e_r, max_c |Wf[r,c]| a_r in [1/2, 1)   (esatto)
    hi = fp16(Wf a_r), lo = fp16(Wf a_r - hi)           (|Wf a_r - hi - lo| <= 2^-24 assoluto)
    => |hi + lo - Wf a_r| <= 2^-23 * max_c |W a_r|      (cioe' 2^-23 del massimo della riga)
X (attivazione ruotata, fp16) e' ESATTA; dY (fp32) va diviso come in `e2e_indietro_hilo` (potenza di due per riga di
token nel dX, per colonna r nel G). Prodotti (fp16 x fp16 esatti in fp32, accumulo fp32):
    avanti: Y   = 2^e_r (X lo^T + X hi^T)                                    2 prodotti (oggi 3)
    dX    : dX  = 2^f_s (h_lo hi + h_hi lo + h_hi hi),  h = dY 2^e_r 2^-f_s   3 prodotti (oggi 3 fp32 / 6 fp16)
    G     : G   = 2^g_r (dY_lo^T X + dY_hi^T X),  dY scalato 2^-g_r            2 prodotti (oggi 3 fp32 / 6 fp16)
Si scarta solo h_lo lo (<= 2^-24 * 2^-24 relativo). «f32»: Wf, dY e X in fp32, UN prodotto fp32 ciascuno.

## Limiti d'errore (dimostrazione in PIEGATI.md §3; u = 2^-24, gamma_n = n u / (1 - n u))
Per ogni elemento, con B la somma dei valori assoluti dei termini (B_Y[s,r] = somma_c |x_sc| somma_k |W_k[r,c]|;
B_dd[r,g] = somma_{s, c in g} |dY_sr| |x_sc|; Bbar_dd[r,g] = max_s |dY_sr| somma_{s, c in g} |x_sc|):
    |Y_piegato  - Y_oggi |  <= (2 gamma_{C+2}   + 2^-22) B_Y
    |dd_piegato - dd_oggi|  <= (2 gamma_{S+gs}) B_dd + 2^-23 Bbar_dd    (x2; in f32 senza il secondo termine)
    dX: oggi e' fl16(fl16(fl16(dx1) + fl16(dx2)) + fl16(dx3)) (accumulo di autograd in fp16), piegato fl16(dX32):
        |dX_piegato - dX_oggi| <= 5 * 2^-11 * somma_k |dx_k| + 2 gamma_{R} B_dX  (domina l'arrotondamento fp16 di OGGI)
Sono limiti deterministici di caso peggiore (Higham, cap. 3); l'errore tipico e' ~sqrt(n) u ed e' misurato dalle prove
(`e2e_test_densi_piegati.py`).

## Quando si piega (altrimenti: via di oggi, identica al bit)
Interruttore acceso, >= 2 piani, tutti dello stesso tipo/forma ed E = 1, ricetta con dtype_gemm fp16 e (ruota o famiglia
non ruotata), ingresso fp16, motore torch o Triton (anche +V). Il passo V con la curvatura delle scale (newton-theta,
`BufferV.HT`) usa per HT la funzione per piano di oggi (UN piano alla volta): TT5 non la usa (adamw).
"""
from __future__ import annotations

import os
import weakref
from typing import Optional

import torch

MODI = ("", "x2", "f32", "riordina")   # "riordina": SOLO braccio di rumore del cancello (via di oggi, piani sommati al contrario)
# gemm: "triton" (default dal 2/10) = kernel Triton `_k_mm16` (fp16 -> fp32); "auto" = torch.mm(out_dtype=fp32), cioe'
# hipBLASLt: sul cancello GPU del 1/10 23 863 ms contro 221 per attn_qkv, INUTILIZZABILE; "f32" = sempre matmul fp32.
# Sulla CPU (fuori da E2E_PIEGA_KERNEL_CPU) i prodotti fp16 sono matmul fp32 sugli ingressi esatti (stessi prodotti).
PIEGA = {"modo": os.getenv("E2E_DENSI_PIEGATI", "").strip(), "gemm": os.getenv("E2E_PIEGA_GEMM", "triton").strip(),
         "cache_trit": os.getenv("E2E_PIEGA_CACHE_TRIT", "0") == "1",
         # SOLO prove CPU (TRITON_INTERPRET=1): percorre i rami della GPU (decodifica Triton dei trit, gemm `_k_mm16`)
         "kernel_cpu": os.getenv("E2E_PIEGA_KERNEL_CPU", "0") == "1"}
if PIEGA["modo"] not in MODI:
    raise ValueError(f"E2E_DENSI_PIEGATI={PIEGA['modo']!r}: {MODI}")
CONTI = {"piegati": 0, "ripiego": 0}          # chiamate dell'avanti (diagnostica, nessun effetto)
_TRIT_CACHE: dict = {}


def imposta(modo: Optional[str] = None, gemm: Optional[str] = None, cache_trit: Optional[bool] = None) -> None:
    if modo is not None:
        if modo not in MODI:
            raise ValueError(f"modo {modo!r}: {MODI}")
        PIEGA["modo"] = modo
    if gemm is not None:
        if gemm not in ("auto", "triton", "f32"):
            raise ValueError(f"gemm {gemm!r}")
        PIEGA["gemm"] = gemm
    if cache_trit is not None:
        PIEGA["cache_trit"] = bool(cache_trit)
        _TRIT_CACHE.clear()


def attivo() -> bool:
    return PIEGA["modo"] in ("x2", "f32", "riordina")


def applica_riordina(xg: torch.Tensor, ds: list, fams: list, ric, mot, oss) -> torch.Tensor:
    """Braccio di RUMORE del cancello (2/10): la via di OGGI (n FamigliaEsperti, stessi kernel) con le uscite dei piani
    sommate in ordine INVERSO (oggi (y1 + y2) + y3, qui (y3 + y2) + y1) e quindi anche il dX accumulato da autograd in
    un altro ordine. Cambia SOLO arrotondamenti fp32/fp16: misura quanto i 40 strati amplificano una perturbazione di
    arrotondamento dei densi (il braccio Bk2_ind32 coi densi ternari non perturba nulla: KL ~0, h al bit)."""
    import e2e_studente as S
    CONTI["piegati"] += 1
    piano = S.piano_unico(xg.shape[0], xg.device)
    y = None
    for f, d in reversed(list(zip(fams, ds))):
        yk = S.FamigliaEsperti.apply(xg.contiguous(), d, f, piano, ric, mot, oss)
        y = yk if y is None else y + yk
    return y


def ammesso(fams: list, ric, mot, xg: torch.Tensor) -> bool:
    import e2e_studente as S
    if len(fams) < 2 or xg.dtype != torch.float16 or ric.dtype_gemm != torch.float16:
        return False
    if not isinstance(mot, (S.MotoreTorch, S.MotoreTriton)) or getattr(mot, "denso", False):
        return False
    f = fams[0]
    if f.ruotata and not ric.ruota:
        return False
    return all((g.tipo, g.E, g.R, g.C, g.NBR) == (f.tipo, 1, f.R, f.C, f.NBR) for g in fams)


# ═══════════════════════════════════ peso effettivo ═══════════════════════════════════
def trit(fam, mot) -> torch.Tensor:
    """Trit (R, C) del piano (dominio dei byte: ruotato per TQ1_0R_1024, come il prodotto della ricetta B). Su GPU col
    kernel di decodifica fuso (d = 1: trit x 1 esatto in fp16); su CPU col decoder torch. int8."""
    import e2e_studente as S
    B = fam.blocchi
    if PIEGA["cache_trit"]:
        # chiave = IDENTITA' del tensore dei byte (riferimento debole: un tensore liberato non si confonde con uno nuovo
        # allo stesso indirizzo) + contatore di versione (i salti del passo V scrivono i byte in place: _version sale)
        v = _TRIT_CACHE.get(id(B))
        if v is not None and v[0]() is B and v[1] == B._version:
            return v[2]
    if (isinstance(mot, S.MotoreTriton) and B.is_cuda) or PIEGA["kernel_cpu"]:
        t = S.decodifica_pesi(fam, torch.ones_like(fam.d0), torch.float16, getattr(mot, "blk", None) or S.Blocchi())[0] \
            .to(torch.int8)
    else:
        t = S.trit_esperto(fam, 0)
    if PIEGA["cache_trit"]:
        _TRIT_CACHE[id(B)] = (weakref.ref(B), B._version, t)
    return t


def peso_effettivo_f32(trits: list, ds: list, gs: int) -> torch.Tensor:
    """Wf (R, C) fp32 = fl(fl(T1 d1 + T2 d2) + T3 d3): ogni T_k d_k e' esatto (d_k fp16), le somme nell'ordine dei piani
    (lo stesso ordine delle uscite di oggi)."""
    R, C = trits[0].shape
    W = None
    for t, d in zip(trits, ds):
        w = (t.view(R, C // gs, gs).to(torch.float32) * d.reshape(R, C // gs, 1).to(torch.float32)).view(R, C)
        W = w if W is None else W.add_(w)
    return W


def scala_pot2(m: torch.Tensor) -> tuple:
    import e2e_indietro_hilo as HL
    return HL.scala_pot2(m)


def dividi_fp16(v: torch.Tensor) -> tuple:
    import e2e_indietro_hilo as HL
    return HL.dividi_fp16(v)


def peso_x2(Wf: torch.Tensor) -> tuple:
    """(hi, lo, inv_r): Wf[r] = inv_r[r] (hi[r] + lo[r]) entro 2^-23 del massimo della riga; hi, lo fp16 (R, C)."""
    a, inv = scala_pot2(__import__("e2e_kernel_qat").amax_abs(Wf, 1))
    hi, lo = dividi_fp16(Wf * a[:, None])
    return hi, lo, inv


# ═══════════════════════════════════ prodotti ═══════════════════════════════════
try:
    import triton
    import triton.language as tl
    _TRITON = True
except Exception:  # pragma: no cover
    _TRITON = False

if _TRITON:
    @triton.jit
    def _k_mm16(a_ptr, sa_m, sa_k, b_ptr, sb_k, sb_n, c_ptr, sc_m, sc_n, M, N, K,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr):
        """C (M, N) fp32 = A (M, K) @ B (K, N), A e B fp16 (passi qualunque: le trasposte sono viste), accumulo fp32."""
        pid = tl.program_id(0)
        npm = tl.cdiv(M, BM)
        npn = tl.cdiv(N, BN)
        per_gruppo = GROUP_M * npn
        gid = pid // per_gruppo
        prima = gid * GROUP_M
        gm = tl.minimum(npm - prima, GROUP_M)
        pm = prima + (pid % per_gruppo) % gm
        pn = (pid % per_gruppo) // gm
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            kk = k0 + rk
            a = tl.load(a_ptr + rm[:, None].to(tl.int64) * sa_m + kk[None, :].to(tl.int64) * sa_k,
                        mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.0)
            b = tl.load(b_ptr + kk[:, None].to(tl.int64) * sb_k + rn[None, :].to(tl.int64) * sb_n,
                        mask=(kk[:, None] < K) & (rn[None, :] < N), other=0.0)
            acc += tl.dot(a, b)
        tl.store(c_ptr + rm[:, None].to(tl.int64) * sc_m + rn[None, :].to(tl.int64) * sc_n, acc,
                 mask=(rm[:, None] < M) & (rn[None, :] < N))

    @triton.jit
    def _k_mm16b(a_ptr, sa_m, sa_k, b_ptr, sb_k, sb_n, c_ptr, sc_m, sc_n, M, N, K,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr, PARI: tl.constexpr):
        """Same product as _k_mm16 (fp16 operands, fp32 accumulation chained along K in the same order), written with
        pointer increments and 32-bit offsets inside the tile (K multiple of BK): fewer registers and VALU per step,
        so that 128 x 128 tiles fit (9/10 bench: plain fp16 Triton GEMM 25-29 TFLOPS at these tiles on gfx1151)."""
        pid = tl.program_id(0)
        npm = tl.cdiv(M, BM)
        npn = tl.cdiv(N, BN)
        per_gruppo = GROUP_M * npn
        gid = pid // per_gruppo
        prima = gid * GROUP_M
        gm = tl.minimum(npm - prima, GROUP_M)
        pm = prima + (pid % per_gruppo) % gm
        pn = (pid % per_gruppo) // gm
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        mm_ = (rm < M)[:, None]
        mn = (rn < N)[None, :]
        pa = a_ptr + (pm * BM).to(tl.int64) * sa_m + (tl.arange(0, BM) * sa_m)[:, None] + (rk * sa_k)[None, :]
        pb = b_ptr + (pn * BN).to(tl.int64) * sb_n + (rk * sb_k)[:, None] + (tl.arange(0, BN) * sb_n)[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            if PARI:                                      # M, N multiples of the tile: no masks (fewer registers)
                a = tl.load(pa)
                b = tl.load(pb)
            else:
                a = tl.load(pa, mask=mm_, other=0.0)
                b = tl.load(pb, mask=mn, other=0.0)
            acc += tl.dot(a, b)
            pa += BK * sa_k
            pb += BK * sb_k
        pc = c_ptr + (pm * BM).to(tl.int64) * sc_m + (pn * BN).to(tl.int64) * sc_n \
            + (tl.arange(0, BM) * sc_m)[:, None] + (tl.arange(0, BN) * sc_n)[None, :]
        if PARI:
            tl.store(pc, acc)
        else:
            tl.store(pc, acc, mask=mm_ & mn)

    def mm16_triton(a: torch.Tensor, b: torch.Tensor, BM: int = 64, BN: int = 64, BK: int = 32, num_warps: int = 4,
                    GROUP_M: int = 8) -> torch.Tensor:
        assert a.dtype == b.dtype == torch.float16 and a.shape[1] == b.shape[0]
        import e2e_kernel_qat as KQ                 # 9/10 E2E_KQAT=mm16_grande: _k_mm16b a riquadri grandi
        if KQ.attivo("mm16_grande") and a.shape[1] % KQ.MM16_GRANDE[2] == 0:
            BM2, BN2, BK2, w2 = KQ.MM16_GRANDE       # stessa catena WMMA lungo K per elemento -> al bit (prova K7)
            M, K = a.shape
            N = b.shape[1]
            c = torch.empty(M, N, dtype=torch.float32, device=a.device)
            _k_mm16b[(triton.cdiv(M, BM2) * triton.cdiv(N, BN2),)](
                a, a.stride(0), a.stride(1), b, b.stride(0), b.stride(1), c, c.stride(0), c.stride(1), M, N, K,
                BM2, BN2, BK2, GROUP_M, int(M % BM2 == 0 and N % BN2 == 0), num_warps=w2, num_stages=2)
            return c
        M, K = a.shape
        N = b.shape[1]
        c = torch.empty(M, N, dtype=torch.float32, device=a.device)
        griglia = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
        _k_mm16[griglia](a, a.stride(0), a.stride(1), b, b.stride(0), b.stride(1), c, c.stride(0), c.stride(1),
                         M, N, K, BM, BN, BK, GROUP_M, num_warps=num_warps)
        return c


def mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a @ b in fp32. Ingressi fp16 (valori esatti) -> prodotti esatti, accumulo fp32; ingressi fp32 -> GEMM fp32."""
    if a.dtype == torch.float32 or PIEGA["gemm"] == "f32" or not (a.is_cuda or PIEGA["kernel_cpu"]):
        return torch.matmul(a.to(torch.float32), b.to(torch.float32))
    if PIEGA["gemm"] == "triton":
        return mm16_triton(a, b)
    return torch.mm(a, b, out_dtype=torch.float32)


# ═══════════════════════════════════ nodo autograd ═══════════════════════════════════
def _modo_v(mot) -> int:
    st = getattr(mot, "stato", None)
    return int(getattr(st, "modo", 0) or 0) if st is not None else 0


def avanti(xg: torch.Tensor, ds: list, fams: list, mot, modo: str) -> tuple:
    """-> (Y fp32 (S, R), pesi per l'indietro). pesi = (trits, hi, lo, inv) in x2, (trits, Wf) in f32."""
    gs = fams[0].gs
    import e2e_kernel_qat as KQ                 # 9/10 E2E_KQAT=cache_densi: peso piegato memorizzato per passo
    pesi = KQ.cache_densi_prendi(fams, modo) if modo == "x2" else None
    if pesi is not None:
        trits, hi, lo, inv = pesi
    else:
        trits = [trit(f, mot) for f in fams]
        Wf = peso_effettivo_f32(trits, ds, gs)
        if modo == "f32":
            return torch.matmul(xg.to(torch.float32), Wf.T), (trits, Wf)
        hi, lo, inv = peso_x2(Wf)
        del Wf
        if modo == "x2":                          # trits NOT kept in the cache (4.4 GB on 40 layers): re-decoded
            KQ.cache_densi_metti(fams, modo, (None, hi, lo, inv))      # by the backward when it needs them
    if KQ.attivo("densi_blas") and KQ.attivo("piega1") and xg.is_cuda:   # 9/10 livello fp16 su hipBLASLt
        return torch.mm(xg, hi.T, out_dtype=torch.float32).mul_(inv[None, :]), (trits, hi, lo, inv)
    if KQ.attivo("densi_k") and not KQ.attivo("mm16_grande") and xg.is_cuda:       # 9/10: i prodotti in UN lancio (_k_mmx), stessi valori al bit
        return KQ.mmx(xg, hi.T, lo.T, 1 if KQ.attivo("piega1") else 3, ocscale=inv), (trits, hi, lo, inv)
    if KQ.attivo("piega1"):                       # 9/10 livello fp16: peso piegato fp16 UNICO (hi), vedi e2e_kernel_qat
        return mm(xg, hi.T).mul_(inv[None, :]), (trits, hi, lo, inv)
    y = mm(xg, lo.T)
    y += mm(xg, hi.T)
    return y.mul_(inv[None, :]), (trits, hi, lo, inv)


def indietro_x(g32: torch.Tensor, pesi: tuple, modo: str) -> torch.Tensor:
    if modo == "f32":
        return torch.matmul(g32, pesi[1])
    _, hi, lo, inv = pesi
    import e2e_kernel_qat as KQ
    if KQ.attivo("densi_blas") and KQ.attivo("piega1") and KQ.attivo("x1") and g32.is_cuda:   # fp16 su hipBLASLt
        s, inv_s = scala_pot2((g32.abs() * inv[None, :]).amax(1, keepdim=True))
        return torch.mm((g32 * inv[None, :] * s).to(torch.float16), hi, out_dtype=torch.float32).mul_(inv_s)
    if KQ.attivo("densi_k") and not KQ.attivo("mm16_grande") and g32.is_cuda:     # 9/10: scale e divisione nei registri, i prodotti in UN lancio
        s, inv_s = scala_pot2((g32.abs() * inv[None, :]).amax(1, keepdim=True))
        t = (0 if KQ.attivo("x1") else 4) | (0 if KQ.attivo("piega1") else 2) | 1
        return KQ.mmx(g32, hi, lo, t, rscale=s, cscale=inv, oscale=inv_s)
    import e2e_kernel_qat as KQ                 # 9/10 livello fp16: x1 = dY fp16 unico, piega1 = peso fp16 unico
    x1, p1 = KQ.attivo("x1"), KQ.attivo("piega1")
    if KQ.attivo("prep") and g32.is_cuda and g32.stride(1) == 1:      # 9/10: preparazione fusa, stessi byte
        hh, hl, s, inv_s = KQ.prep_righe(g32, inv.contiguous(), lo=not x1)
    else:
        h = g32 * inv[None, :]                                   # potenze di due: esatto
        s, inv_s = scala_pot2(__import__("e2e_kernel_qat").amax_abs(h, 1, keepdim=True))
        hh, hl = dividi_fp16(h * s)
        del h
    dx = None
    for a_, b_, serve in ((hl, hi, not x1), (hh, lo, not p1), (hh, hi, True)):
        if serve:
            dx = mm(a_, b_) if dx is None else dx.add_(mm(a_, b_))
    return dx.mul_(inv_s)


def prodotto_g(g32: torch.Tensor, xg: torch.Tensor, modo: str) -> torch.Tensor:
    """G (R, C) fp32 = dY^T X."""
    if modo == "f32":
        return torch.matmul(g32.T, xg.to(torch.float32))
    s, inv = scala_pot2(__import__("e2e_kernel_qat").amax_abs(g32, 0))                   # per colonna r di dY (riga r di G)
    import e2e_kernel_qat as KQ
    if KQ.attivo("densi_blas") and KQ.attivo("x1") and g32.is_cuda:     # fp16 su hipBLASLt
        return torch.mm((g32 * s[None, :]).to(torch.float16).T, xg, out_dtype=torch.float32).mul_(inv[:, None])
    if KQ.attivo("densi_k") and not KQ.attivo("mm16_grande") and g32.is_cuda:     # 9/10: divisione nei registri, i due prodotti in UN lancio
        return KQ.mmx(g32.T, xg, None, 1 if KQ.attivo("x1") else 5, rscale=s, oscale=inv)
    if KQ.attivo("prep") and g32.is_cuda and g32.stride(1) == 1:      # 9/10: preparazione fusa, stessi byte
        gh, gl, s, inv = KQ.prep_colonne(g32, lo=not KQ.attivo("x1"))
    else:
        gh, gl = dividi_fp16(g32 * s[None, :])
    if KQ.attivo("x1"):                           # 9/10 livello fp16: dY fp16 unico (scalato per potenza di due)
        return mm(gh.T, xg).mul_(inv[:, None])
    G = mm(gl.T, xg)
    G += mm(gh.T, xg)
    return G.mul_(inv[:, None])


def dd_da_g(G: torch.Tensor, t: torch.Tensor, gs: int) -> torch.Tensor:
    R, C = G.shape
    return (t.view(R, C // gs, gs).to(torch.float32) * G.view(R, C // gs, gs)).sum(-1)


class DensoPiegato(torch.autograd.Function):
    """y = somma_k X W_k(d_k)^T con UN peso effettivo (vedi testa del modulo). Ingressi: xg (S, C) fp16, poi le scale
    d_1..d_n (1, R, NG) dei piani. Stesse uscite di n `FamigliaEsperti` sommati (dtype, forme), entro i limiti dichiarati."""

    @staticmethod
    def forward(ctx, xg, fams, ric, mot, oss, modo, *ds):
        y, pesi = avanti(xg, list(ds), list(fams), mot, modo)
        ctx.save_for_backward(xg, *ds)
        ctx.pesi, ctx.fams, ctx.ric, ctx.mot, ctx.oss, ctx.modo = pesi, list(fams), ric, mot, oss, modo
        return y

    @staticmethod
    def backward(ctx, gy):
        import e2e_pv_codici as PV
        xg, *ds = ctx.saved_tensors
        fams, mot, oss, modo, pesi = ctx.fams, ctx.mot, ctx.oss, ctx.modo, ctx.pesi
        serve_x = ctx.needs_input_grad[0]
        serve_d = [ctx.needs_input_grad[6 + i] for i in range(len(ds))]
        g32 = gy.to(torch.float32).contiguous()
        if oss is not None:
            for f in fams:                                    # stesse righe di diagnostica di oggi (una per piano)
                oss.gradiente(f, gy, g32)
        if any(serve_d):                                      # 2/10 --pv-dd-solo-salti: piani senza salti preparati
            serve_d = [sd and not PV.dd_superfluo(mot, f) for sd, f in zip(serve_d, fams)]
        dx = None
        if serve_x:
            dx = indietro_x(g32, pesi, modo).to(xg.dtype)
            if oss is not None:
                oss.guarda(dx)
        dds = [None] * len(ds)
        if any(serve_d):
            trits = pesi[0] if pesi[0] is not None else [trit(f, mot) for f in fams]   # cache_densi: on demand
            gs = fams[0].gs
            mv = _modo_v(mot)
            bufs = [mot.stato.buffer(f) if mv else None for f in fams]
            if mv == 3:                                       # curvatura: dL/dd = 0, H dal (dY^2)^T (X^2) (UNO per tutti)
                G2 = torch.matmul((g32 * g32).T, xg.to(torch.float32).pow(2))
                for i, f in enumerate(fams):
                    if not serve_d[i]:
                        continue
                    dds[i] = torch.zeros(1, f.R, f.NG, dtype=ds[i].dtype, device=g32.device)
                    PV.curvatura_da_dw2(bufs[i], 0, G2, gs)
                    if bufs[i].HT is not None:                # newton-theta: per piano (la via di oggi)
                        piano = _piano(g32.shape[0], g32.device)
                        if type(mot).__name__ == "MotoreTritonV":
                            PV.ht_triton(g32, xg.to(torch.float32).contiguous(), f, piano, bufs[i].HT)
                        else:
                            PV.ht_torch(g32, xg.to(torch.float32), f, piano, bufs[i].HT)
            else:
                G = prodotto_g(g32, xg, modo)
                for i, f in enumerate(fams):
                    if not serve_d[i]:
                        continue
                    dds[i] = dd_da_g(G, trits[i], gs).unsqueeze(0).to(ds[i].dtype)
                    if mv == 1:
                        PV.proposta_da_dw(bufs[i], 0, G, trits[i], f.d0[0], gs)
                    elif mv == 2:
                        PV.verifica_da_dw(bufs[i], 0, G, gs)
        ctx.pesi = None
        return (dx, None, None, None, None, None, *dds)


def _piano(T: int, device):
    import e2e_studente as S
    return S.piano_unico(T, device)


def applica(xg: torch.Tensor, ds: list, fams: list, ric, mot, oss) -> torch.Tensor:
    if PIEGA["modo"] == "riordina":
        return applica_riordina(xg, ds, fams, ric, mot, oss)
    CONTI["piegati"] += 1
    return DensoPiegato.apply(xg.contiguous(), tuple(fams), ric, mot, oss, PIEGA["modo"], *ds)


def byte_transitori(R: int, C: int, S_: int, modo: str = "x2") -> dict:
    """Memoria TRANSITORIA in piu' rispetto a oggi per UN denso (oggi: nessun peso materializzato, decodifica nel kernel).
    trits int8 n=3 piani, Wf fp32 (durante la costruzione), hi+lo fp16 (salvati fino all'indietro dello strato), G fp32,
    divisioni di dY (S, R) e h (S, R)."""
    RC = R * C
    t = 3 * RC
    if modo == "f32":
        return {"trit": t, "W": 4 * RC, "G": 4 * RC, "dY_diviso": 0, "picco": t + 8 * RC}
    return {"trit": t, "W_costruzione_f32": 4 * RC, "hi_lo": 4 * RC, "G": 4 * RC, "dY_diviso": 4 * S_ * R + 4 * S_ * R,
            "picco": t + 4 * RC + 4 * RC + 8 * S_ * R}

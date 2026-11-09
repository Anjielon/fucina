#!/usr/bin/env python3
"""e2e_indietro_hilo — indietro degli ESPERTI a prodotti divisi hi/lo (27/9/2026, ramo indietro-bf16x3). File NUOVO.

## Perche'
La corsa (EB2, e cosi' EB3) usa la ricetta `B_ind32`: l'avanti degli esperti in fp16 esatto (attivazione ruotata fp16,
peso trit x d fp16), ma l'INDIETRO (dX = dY W e dL/dd = somma_gruppo trit (.) dY^T X) con ingressi fp32. Sul
gfx1151 un `tl.dot` fp32 non ha la via WMMA (FMA, 1,3-3 TFLOPS contro 20-34 della fp16) e `_k_dd2` in fp32 riversa
438 VGPR (commit f5b1143). `B_ind32` e' nata perche' con l'indietro fp16 (ricetta B) NESSUNA scala globale K tiene
insieme trabocco e sottoflusso dei gradienti (G1, famiglia gate): e' un problema di GAMMA dinamica, non di bit.

Qui gli stessi prodotti si fanno con i kernel fp16/bf16 GIA' esistenti (quelli della ricetta B), dividendo dY in
due o tre parti (Henry-Tang-Heinecke 1904.06376 per bf16x3; Ootomo-Yokota 2203.03341 per la divisione fp16 con
scala contro il sottoflusso), e sommando le uscite fp32:

* **"fp16x2"** (ricette `B_ind2`, `B3_ind2`): il peso W = trit x d e' ESATTO in fp16 (d e' fp16, trit in {-1,0,1});
  X (attivazione ruotata, silu(g)u) e' ESATTO in fp16. Solo dY (fp32) va diviso. Per togliere il problema di gamma
  si scala con una potenza di due (esatta) prima di dividere:
    dX:   per RIGA (token) s:            v_s = dY_s * 2^-e_s,  max|v_s| in [0,5, 1)   (dX_s e' lineare in dY_s)
    dL/dd per (esperto e, riga r di W):  v_{s,r} = dY_{s,r} * 2^-f_{e,r} per s nel segmento di e   (dL/dd[e,r,:]
          e' lineare nella colonna r di dY ristretta ai token di e)
    hi = fp16(v), lo = fp16(v - hi)                     (v - hi esatto in fp32)
    dX   = 2^e_s  (lo W + hi W)                          due GEMM fp16 -> fp32 (prima il termine piccolo)
    dL/dd = 2^f_er (D(lo, X) + D(hi, X))                 idem con il kernel di dL/dd
  Errore di rappresentazione di dY: |v - hi - lo| <= 2^-24 in assoluto con max|v| >= 1/2, cioe' <= 2^-23 del
  massimo della riga (del segmento): il livello di un prodotto fp32 (2^-24), MEGLIO di bf16x3 (2^-16 per termine).
  Il termine lo x W manca di niente: W non ha parte bassa. DUE prodotti fp16 per GEMM, non tre.
* **"bf16x3"** (ricette `B_ind3`, `B3_ind3`): dY = hi + lo in bf16 (gamma di fp32: niente scala), d = d_hi + d_lo
  in bf16 (d fp16 ha 11 bit: la parte bassa ha <= 3 bit, ESATTA in bf16), X fp16 = x_hi + x_lo in bf16 (esatto);
  tre prodotti (lo hi + hi lo + hi hi, si scarta lo lo ~2^-16). Sulla CPU si prova col motore torch (emulazione
  esatta); con Triton va provata SOLO sulla GPU (l'interprete di Triton non ha bf16: numeri senza senso, gia'
  visto in e2e_test_triton2 K1) -> microprova nel cancello (`e2e_cancello_indietro.py`).

Accumulo: i kernel fondono `acc += tl.dot(...)` nella WMMA, che su gfx1151 TRONCA (deriva ~2e-8 per passo da 16,
TECNICHE §17 del ramo bf16x3fix): dX ha contrazione R = 512 (gate/up) o 2048 (down) -> <= 3e-6 di deriva; dL/dd
contrae sui token dell'esperto (media ~256). E' la stessa aritmetica della ricetta B (G1 superato); la misura vera
e' la microprova GPU del cancello (errore per prodotto contro fp64).

dX finisce comunque arrotondato a fp16 (`.to(x.dtype)`, come in B_ind32: x_ord e silu(g)u sono fp16): 2^-11 per
elemento, 100 volte l'errore dei prodotti divisi. Su dX dunque fp16x2/bf16x3 e fp32 danno lo STESSO tensore salvo
~1% di elementi a 1 ulp fp16; la precisione conta per dL/dd (dtheta), che resta fp32.

## Costo (stima [S], la misura e' il cancello)
Kernel della ricetta B (triton2, fp16, cancello_kernel del 27/9 11:28, accanto a EB2): indietro esperti ~0,037-0,043
ms/token per strato. Indietro fp32 (B_ind32 contro B, M0 26/9, motore triton): +0,197 ms/token per strato. fp16x2 =
2 x fp16 + divisioni (qualche passata elementare su dY) ~0,08-0,09; B_ind32 ~0,23-0,24 -> ~-0,15 ms/token per
strato, ~-6 ms/token su 40 strati. bf16x3 = 3 x ~0,04 = ~0,12.

## Uso
Ricette in `e2e_studente.RICETTE` (`B_ind2`, `B3_ind2`, `B_ind3`, `B3_ind3`); il gancio e' in
`e2e_studente._indietro_famiglia` e scatta SOLO con `Ricetta.indietro_diviso` non vuoto e `dtype_indietro` None
(la ricetta fp32 di riferimento di G1, `e2e_allena.ricetta_indietro_fp32`, mette dtype_indietro = fp32 e torna
all'indietro fp32 esatto). Default: nessuna ricetta esistente cambia (prova H0 di `e2e_test_indietro_hilo.py`).
"""
from __future__ import annotations

from typing import Optional

import torch

VIE = ("fp16x2", "bf16x3")
_E_MAX = 125          # |esponente| della scala: 2^+-125 e' finito ed esatto in fp32


# ═══════════════════════════════════ divisioni ═══════════════════════════════════
def scala_pot2(m: torch.Tensor) -> tuple:
    """m >= 0 (massimi). -> (s, inv) fp32 con s = 2^-e ESATTO, m*s in [0,5, 1) (frexp), s = 1 dove m == 0 o non
    finito; inv = 2^e (l'inversa, esatta). |e| <= 125."""
    _, e = torch.frexp(m)
    ok = (m > 0) & torch.isfinite(m)
    e = torch.where(ok, e, torch.zeros_like(e)).clamp_(-_E_MAX, _E_MAX)
    uno = torch.ones_like(m, dtype=torch.float32)
    return torch.ldexp(uno, -e), torch.ldexp(uno, e)


def dividi_fp16(v: torch.Tensor) -> tuple:
    """v fp32 (gia' scalato: |v| < 1) -> (hi, lo) fp16 con |v - hi - lo| <= 2^-24."""
    if v.dtype != torch.float32:
        raise TypeError(f"dividi_fp16: {v.dtype}, atteso float32")
    hi = v.to(torch.float16)
    lo = (v - hi.float()).to(torch.float16)
    return hi, lo


def dividi_bf16(v: torch.Tensor) -> tuple:
    """v fp32 -> (hi, lo) bf16 (e2e_bf16x3.scomponi, qui senza dipendere dal modulo)."""
    if v.dtype != torch.float32:
        raise TypeError(f"dividi_bf16: {v.dtype}, atteso float32")
    hi = v.to(torch.bfloat16)
    lo = (v - hi.float()).to(torch.bfloat16)
    return hi, lo


def esperto_di_riga(piano, S_: int) -> torch.Tensor:
    """(S,) indice dell'esperto di ogni riga ordinata (senza sincronizzazioni)."""
    righe = torch.arange(S_, device=piano.offs.device, dtype=piano.offs.dtype)
    return torch.searchsorted(piano.offs[1:], righe, right=True).clamp_(max=piano.E - 1)


def massimi_segmento(a: torch.Tensor, eid: torch.Tensor, E: int) -> torch.Tensor:
    """a (S, R) >= 0, eid (S,) -> (E, R): massimo per (esperto, colonna) sui token dell'esperto (0 se vuoto)."""
    M = torch.zeros(E, a.shape[1], dtype=a.dtype, device=a.device)
    return M.scatter_reduce_(0, eid.unsqueeze(1).expand(-1, a.shape[1]), a, reduce="amax", include_self=True)


# ═══════════════════════════════════ prodotti ═══════════════════════════════════
def _W_esperto(fam, e: int, d: torch.Tensor) -> torch.Tensor:
    """Peso (R, C) fp32 = trit x d (esatto: d ha <= 11 bit), per il motore torch."""
    import e2e_studente as S
    q = S.trit_esperto(fam, e).to(torch.float32)
    return (q.view(fam.R, fam.NG, fam.gs) * d[e].to(torch.float32).unsqueeze(-1)).view(fam.R, fam.C)


def prodotto_dx(mot, g: torch.Tensor, d: torch.Tensor, fam, piano) -> torch.Tensor:
    """g (S, R) fp16|bf16 @ W_e (trit x d) per esperto -> (S, C) FP32 (accumulo fp32, uscita NON arrotondata).
    d: scale (E, R, NG) i cui valori sono esatti nel dtype di g (il kernel le converte a quel dtype)."""
    import e2e_studente as S
    if isinstance(mot, S.MotoreTorch):
        out = torch.zeros(g.shape[0], fam.C, dtype=torch.float32, device=g.device)
        o = piano.offs.tolist()
        for e in range(fam.E):
            if o[e + 1] > o[e]:
                out[o[e]:o[e + 1]] = torch.matmul(g[o[e]:o[e + 1]].float(), _W_esperto(fam, e, d))
        return out
    if getattr(mot, "denso", False):
        raise NotImplementedError("indietro diviso: motore triton_denso non supportato (peso denso nel dtype della "
                                  "ricetta, la parte bassa di d in bf16 non ci sta)")
    if getattr(mot, "v2", False):
        return S._dx2_triton(g, fam, d.contiguous(), piano, torch.float32, mot.blk)
    return S._gemm_triton(g, fam, d.contiguous(), None, piano, True, torch.float32, mot.blk)


def prodotto_dd(mot, g: torch.Tensor, x: torch.Tensor, fam, piano) -> torch.Tensor:
    """dL/dd (E, R, NG) fp32 = somma_gruppo trit (.) (g^T x) per esperto; g, x stesso dtype (fp16|bf16)."""
    return mot.dd(g, x, fam, piano, torch.float32)


# ═══════════════════════════════════ indietro di UNA famiglia ═══════════════════════════════════
def _modo_v(mot) -> int:
    """Modo del passo V (e2e_pv_codici) del motore: 0 = nessuno. Nei modi 1-3 il dL/dd del motore V raccoglie
    anche statistiche NON lineari in dY (proposta: argmax |acc|; verifica: Q += v^2; curvatura: dY^2): due chiamate
    con dY diviso le sbaglierebbero. Il modo 1/2 col motore Triton va nel kernel V diviso (acc intero prima delle
    statistiche); il modo 3 e il motore torch V restano sulla via fp32 di B_ind32 (stesse statistiche al bit)."""
    st = getattr(mot, "stato", None)
    return int(getattr(st, "modo", 0) or 0) if st is not None else 0


def dx_diviso(mot, g32: torch.Tensor, d: torch.Tensor, fam, piano, via: str) -> torch.Tensor:
    """dX = dY W in FP32 (prima dell'arrotondamento al dtype di x) coi prodotti divisi di `via`."""
    if via == "fp16x2":
        s, inv = scala_pot2(__import__("e2e_kernel_qat").amax_abs(g32, 1, keepdim=True))
        import e2e_kernel_qat as KQ               # 9/10 E2E_KQAT=dx: divisione nei registri, hi e lo in UN kernel
        if KQ.attivo("dx") and getattr(mot, "v2", False) and not getattr(mot, "denso", False) \
                and type(mot).__name__ == "MotoreTriton":
            return KQ.dx_s_triton(g32, s, inv, fam, d.contiguous(), piano, mot.blk, 1 if KQ.attivo("x1") else 3)
        hi, lo = dividi_fp16(g32 * s)
        if KQ.attivo("x1"):                       # 9/10 livello fp16: dY fp16 unico (scala per riga, potenza di 2)
            return prodotto_dx(mot, hi, d, fam, piano).mul_(inv)
        y = prodotto_dx(mot, lo, d, fam, piano)
        y += prodotto_dx(mot, hi, d, fam, piano)
        del hi, lo
        return y.mul_(inv)
    if via == "bf16x3":
        gh, gl = dividi_bf16(g32)
        dh, dl = dividi_bf16(d.to(torch.float32))
        y = prodotto_dx(mot, gl, dh, fam, piano)
        y += prodotto_dx(mot, gh, dl, fam, piano)
        y += prodotto_dx(mot, gh, dh, fam, piano)
        return y
    raise ValueError(f"via {via!r}: {VIE}")


def _dd_fp32(mot, g32, x, fam, piano, d):
    """dL/dd come la via fp32 di `_indietro_famiglia` con dtype_indietro fp32 (B_ind32): x (fp16 esatto) in fp32."""
    return mot.dd(g32, x.to(torch.float32).contiguous(), fam, piano, d.dtype)


def indietro_famiglia(gy, x, d, fam, piano, ric, mot, oss, serve_x: bool, serve_d: bool):
    """Corpo di `e2e_studente._indietro_famiglia` per `ric.indietro_diviso` in VIE. Stesse uscite della via fp32
    (dx nel dtype di x, dL/dd nel dtype di d), prodotti divisi al posto dei prodotti fp32."""
    via = ric.indietro_diviso
    if via not in VIE:
        raise ValueError(f"indietro_diviso {via!r}: {VIE}")
    if not (ric.ruota or not fam.ruotata):
        raise ValueError("indietro diviso: solo ricette con l'attivazione ruotata (B...), X deve essere quella del "
                         "prodotto")
    if x.dtype != torch.float16:
        raise TypeError(f"indietro diviso: ingresso degli esperti {x.dtype}, atteso float16 (ricette B)")
    g_in = gy
    g32 = gy.to(torch.float32).contiguous()
    if oss is not None:
        oss.gradiente(fam, g_in, g32)             # come la via fp32: niente sottoflusso fp16 da contare
    if serve_d:                                   # 2/10 --pv-dd-solo-salti: dL/dd che nessuno legge (e2e_pv_codici)
        import e2e_pv_codici as PV
        if PV.dd_superfluo(mot, fam):
            serve_d = False
    dx = dd = None
    if serve_x:
        dx = dx_diviso(mot, g32, d, fam, piano, via).to(x.dtype)
    if via == "fp16x2":
        if serve_d:
            modo_v = _modo_v(mot)
            import e2e_kernel_qat as KQ           # 9/10 E2E_KQAT=dd: dL/dd dalle somme parziali, dY fp32 esatto
            if modo_v == 0 and KQ.attivo("dd") and type(mot).__name__ == "MotoreTriton" \
                    and getattr(mot, "v2", False) and fam.C % fam.gs == 0 and fam.gs in (128, 256):
                dd = KQ.dd_p_triton(g32, x.contiguous(), fam, piano).to(d.dtype)
            elif modo_v == 0 or (modo_v in (1, 2) and type(mot).__name__ == "MotoreTritonV"):
                eid = esperto_di_riga(piano, g32.shape[0])
                s_er, inv_er = scala_pot2(massimi_segmento(g32.abs(), eid, fam.E))
                hi, lo = dividi_fp16(g32 * s_er[eid])
                if modo_v == 0:
                    dd = prodotto_dd(mot, lo, x, fam, piano)
                    dd += prodotto_dd(mot, hi, x, fam, piano)
                    dd = dd * inv_er.unsqueeze(-1)
                else:                              # passo V: dY^T X intero nel kernel, poi le statistiche
                    import e2e_pv_codici as PV
                    dd = PV.dd_v_triton(hi, x, fam, piano, mot.blk_v, mot.stato.buffer(fam), modo_v, dy_lo=lo,
                                        inv=inv_er)
                del hi, lo
                dd = dd.to(d.dtype)
            else:
                dd = _dd_fp32(mot, g32, x, fam, piano, d)
    else:                                         # bf16x3
        if serve_d and _modo_v(mot) != 0:
            dd = _dd_fp32(mot, g32, x, fam, piano, d)
        elif serve_d:
            gh, gl = dividi_bf16(g32)
            xh, xl = dividi_bf16(x.to(torch.float32))
            dd = prodotto_dd(mot, gl, xh, fam, piano)
            dd += prodotto_dd(mot, gh, xl, fam, piano)
            dd += prodotto_dd(mot, gh, xh, fam, piano)
            dd = dd.to(d.dtype)
            del xh, xl, gh, gl
    if dx is not None and oss is not None:
        oss.guarda(dx)
    return dx, dd


def errore_rappresentazione(g: torch.Tensor, via: str, piano=None, E: Optional[int] = None) -> dict:
    """Solo prove/cancello: quanto la divisione di g (fp32) perde, rispetto al massimo della riga (dX) e del segmento
    (dL/dd), in fp64."""
    g32 = g.to(torch.float32)
    out = {}
    if via == "fp16x2":
        s, inv = scala_pot2(__import__("e2e_kernel_qat").amax_abs(g32, 1, keepdim=True))
        hi, lo = dividi_fp16(g32 * s)
        r = (hi.double() + lo.double()) * inv.double()
        den = g32.abs().amax(1, keepdim=True).double().clamp_min(1e-300)
        out["righe_max_rel"] = float(((r - g32.double()).abs() / den).max())
        if piano is not None:
            eid = esperto_di_riga(piano, g32.shape[0])
            s_er, inv_er = scala_pot2(massimi_segmento(g32.abs(), eid, E))
            hi, lo = dividi_fp16(g32 * s_er[eid])
            r = (hi.double() + lo.double()) * inv_er[eid].double()
            den = massimi_segmento(g32.abs(), eid, E)[eid].double().clamp_min(1e-300)
            out["segmenti_max_rel"] = float(((r - g32.double()).abs() / den).max())
    else:
        hi, lo = dividi_bf16(g32)
        r = hi.double() + lo.double()
        a = g32.double().abs()
        v = a >= 2.0 ** -100                  # sotto, lo e' subnormale in bf16: conta solo il relativo alla riga
        out["elementi_max_rel"] = float(((r - g32.double()).abs()[v] / a[v]).max()) if bool(v.any()) else 0.0
        den = g32.abs().amax(1, keepdim=True).double().clamp_min(1e-300)
        out["righe_max_rel"] = float(((r - g32.double()).abs() / den).max())
    return out

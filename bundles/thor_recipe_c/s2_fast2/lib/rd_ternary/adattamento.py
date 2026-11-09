"""Matematica batched: fit di un piano ternario sul RESIDUO nella metrica di
Fisher, rifit esatto delle scale, distorsione K-FAC.

Tutte le funzioni lavorano su lotti [B, out, in] (B = esperti x opzioni): e' il
punto che rende fattibili 40x256 esperti — con un ciclo Python per tensore si
resterebbe bloccati sui lanci di kernel, non sui FLOP.

Metrica:  d(D) = 0.5 tr(G D A D^T),  A = H_I [in,in], G = H_O [out,out].
  * fit del piano (GPTQ/OBQ ternario lungo le colonne con Hessiana A): ottimo
    per riga nella metrica di A; le righe sono pesate da G_ii ma accoppiate
    da G fuori diagonale solo in fase di VALUTAZIONE (d esatta con G piena);
  * rifit delle scale: minimi quadrati esatti per riga su tutti i (piano, gruppo)
    nella metrica di A (sistema K*ng x K*ng per riga, risolto in batch).
"""
from __future__ import annotations

import os
import warnings
from typing import Callable, Optional

import torch

from .formato import GROUP, dequant


def damp(a: torch.Tensor, rel: float) -> torch.Tensor:
    """A + rel * media(diag A) * I  (stabilita' di Cholesky, come in GPTQ)."""
    d = a.diagonal(dim1=-2, dim2=-1).contiguous().mean(-1).clamp_min(1e-12)  # = damp_diag, bit per bit
    eye = torch.eye(a.shape[-1], dtype=a.dtype, device=a.device)
    return a + rel * d[..., None, None] * eye


# R015 (4/10): su questa APU i risolutori a lotti di rocBLAS/hipSOLVER (potrs, potri/TRSM) cadono con lotti grandi
# (YAQA, fix fec2b6a: inversa a lotti di 8; dkl2: HIPSOLVER_STATUS_INTERNAL_ERROR in hipsolverDnSpotrsBatched).
# Ogni elemento del lotto e' indipendente, quindi spezzare il lotto NON cambia il risultato.
def _su_cpu(t: torch.Tensor) -> bool:
    """Punto unico (e sostituibile nei test) della domanda «questo tensore vive sulla CPU?»."""
    return t.device.type == "cpu"


GPTQ_LOTTO_GPU = int(os.environ.get("RD_GPTQ_LOTTO", "8"))     # matrici [i,i] per chiamata, solo su GPU
SOLVE_LOTTO_GPU = int(os.environ.get("RD_SOLVE_LOTTO", "64"))  # sistemi piccoli per chiamata (modo "gpu")
SOLVE_SU = os.environ.get("RD_SOLVE_SU", "cpu")                # "cpu" (default su GPU) | "gpu" (a lotti, ripiego CPU)


def gptq_factor(ad: torch.Tensor, max_tries: int = 6, lotto: Optional[int] = None) -> torch.Tensor:
    """Fattore triangolare superiore U di chol(A^-1) (convenzione GPTQ).
    Su GPU il lotto si spezza in gruppi di `lotto` matrici (default 8, come YAQA): stesso risultato,
    nessuna chiamata batched grande a rocBLAS."""
    lotto = GPTQ_LOTTO_GPU if lotto is None else lotto
    if not _su_cpu(ad) and lotto > 0 and ad.shape[0] > lotto:
        return torch.cat([_gptq_factor(ad[b:b + lotto], max_tries) for b in range(0, ad.shape[0], lotto)])
    return _gptq_factor(ad, max_tries)


def _gptq_factor(ad: torch.Tensor, max_tries: int = 6) -> torch.Tensor:
    a = ad
    for _ in range(max_tries):
        L, info = torch.linalg.cholesky_ex(a)
        fail = info > 0
        if not bool(fail.any()):
            break
        # piu' smorzamento SOLO agli elementi falliti (il risultato non dipende dal lotto)
        a = torch.where(fail.view(*fail.shape, 1, 1), damp(a, 0.1), a)
    hinv = torch.cholesky_inverse(L)
    U, info = torch.linalg.cholesky_ex(hinv, upper=True)
    if bool((info > 0).any()):
        raise RuntimeError("Cholesky di A^-1 fallita: aumentare lo smorzamento")
    return U


def ternary_scale(w: torch.Tensor, c: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Scala ottima di un gruppo ternario pesato.

    Con supporto = top-k per |w| e segni = sign(w), l'errore pesato e'
      sum c w^2 - (sum_sel c|w|)^2 / sum_sel c
    -> si sceglie k che massimizza (cumsum c|w|)^2 / cumsum c (ricerca esatta su k).
    w [...,GS], c broadcastabile a w, mask bool opzionale (posizioni ammesse).
    """
    a = w.abs()
    c = c.expand_as(a)
    if mask is not None:
        a = a * mask
        c = c * mask + (~mask) * 1e30  # una posizione vietata non conviene mai includerla
    # ordinamento STABILE: a parita' di |w| l'ordine non dipende dal lotto ne' dal dispositivo
    a_s, order = torch.sort(a, dim=-1, descending=True, stable=True)
    c_s = c.gather(-1, order)
    num = (c_s * a_s).cumsum(-1)
    den = c_s.cumsum(-1).clamp_min(1e-30)
    k = (num * num / den).argmax(-1, keepdim=True)
    s = (num.gather(-1, k) / den.gather(-1, k)).squeeze(-1)
    return torch.nan_to_num(s, nan=0.0).clamp_min(0.0)


@torch.no_grad()
def gptq_ternary(r: torch.Tensor, U: torch.Tensor, cdiag: torch.Tensor,
                 mask: Optional[torch.Tensor] = None, group: int = GROUP):
    """Quantizza il residuo r [B,o,i] in UN piano ternario con propagazione
    dell'errore (OBQ) nella metrica di A. Blocco = gruppo = 256: la scala del
    gruppo si calcola sulle colonne GIA' aggiornate dall'errore dei gruppi precedenti.

    mask [B,o,i] bool: posizioni ammesse (piano sparso); fuori maschera il trit e' 0
    e l'errore viene compensato sulle colonne successive.
    Ritorna (T int8 [B,o,i], S fp16 [B,o,ng]).
    """
    B, o, i = r.shape
    GS = group
    ng = i // GS
    W = r.clone()
    T = torch.zeros(B, o, i, dtype=torch.int8, device=r.device)
    S = torch.zeros(B, o, ng, dtype=torch.float16, device=r.device)
    for g in range(ng):
        i1, i2 = g * GS, (g + 1) * GS
        W1 = W[:, :, i1:i2].clone()
        m1 = mask[:, :, i1:i2] if mask is not None else None
        s = ternary_scale(W1, cdiag[:, None, i1:i2], m1)
        S[:, :, g] = s.half()
        s = S[:, :, g].float()  # si quantizza con la scala come verra' memorizzata
        safe = s.clamp_min(1e-30)
        Ub = U[:, i1:i2, i1:i2]
        Err = torch.zeros(B, o, GS, dtype=r.dtype, device=r.device)
        for j in range(GS):
            w = W1[:, :, j]
            t = torch.clamp(torch.round(w / safe), -1, 1)
            t = torch.where(s > 0, t, torch.zeros_like(t))
            if m1 is not None:
                t = t * m1[:, :, j]
            err = (w - t * s) / Ub[:, j, j][:, None]
            W1[:, :, j:] -= err[:, :, None] * Ub[:, None, j, j:]
            Err[:, :, j] = err
            T[:, :, i1 + j] = t.to(torch.int8)
        if i2 < i:
            W[:, :, i2:] -= torch.bmm(Err, U[:, i1:i2, i2:])
    return T, S


def _mm(x: torch.Tensor, y: torch.Tensor, mm_dtype: torch.dtype) -> torch.Tensor:
    """Prodotto in precisione ridotta (WMMA su gfx1151) con normalizzazione
    per non uscire dal range fp16; risultato in fp32."""
    if mm_dtype == torch.float32:
        return x @ y
    # normalizzazione per matrice (non per lotto): il risultato non dipende dal lotto
    sx = x.abs().amax(dim=(-1, -2), keepdim=True).clamp_min(1e-30)
    sy = y.abs().amax(dim=(-1, -2), keepdim=True).clamp_min(1e-30)
    return ((x / sx).to(mm_dtype) @ (y / sy).to(mm_dtype)).float() * (sx * sy)


@torch.no_grad()
def refit_scales(planes: torch.Tensor, target: torch.Tensor, ad: torch.Tensor,
                 chunk_rows: int = 128, mm_dtype: torch.dtype = torch.float32,
                 group: int = GROUP) -> torch.Tensor:
    """Minimi quadrati esatti delle scale (tutti i piani insieme), riga per riga,
    nella metrica di A:  min_s || w_r - sum_{k,g} s_{kg} v_{kg} ||_A^2.

    planes [B,K,o,i] int8, target [B,o,i], ad [B,i,i]  ->  scales fp16 [B,K,o,ng].
    Costo dominante: K*o*i^2 per il prodotto T A (come una passata GPTQ).
    """
    B, K, o, i = planes.shape
    GS = group
    ng = i // GS
    P = K * ng
    Ar = ad.reshape(B, ng, GS, i)
    out = torch.empty(B, K, o, ng, dtype=torch.float16, device=planes.device)
    for r0 in range(0, o, chunk_rows):
        r1 = min(o, r0 + chunk_rows)
        r = r1 - r0
        Tg = planes[:, :, r0:r1].float().reshape(B, K, r, ng, GS)
        # Y[b,k,r,g,:] = (T_k[r] ristretto al gruppo g) @ A
        Y = _mm(Tg.permute(0, 3, 1, 2, 4).reshape(B, ng, K * r, GS),
                Ar, mm_dtype).reshape(B, ng, K, r, i).permute(0, 2, 3, 1, 4)
        Yg = Y.reshape(B, K, r, ng, ng, GS)
        M = torch.einsum("bkrghs,blrhs->brkglh", Yg, Tg).reshape(B, r, P, P)
        rhs = torch.einsum("bkrgj,brj->brkg", Y, target[:, r0:r1].float()).reshape(B, r, P)
        diag = M.diagonal(dim1=-2, dim2=-1)
        ridge = 1e-7 * diag.mean(-1, keepdim=True).clamp_min(1e-30) + (diag <= 0).float()
        M = M + torch.diag_embed(ridge)
        s = risolvi_spd(M, rhs)
        s = torch.where(diag > 0, s, torch.zeros_like(s))
        out[:, :, r0:r1] = s.reshape(B, r, K, ng).permute(0, 2, 1, 3).half()
    return out


def _spd_cpu(M: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """M [N,P,P], rhs [N,P] sul dispositivo di M (qui CPU). Cholesky dove riesce; SOLO gli elementi
    la cui Cholesky fallisce (rango deficiente in fp32) vanno al minimi-quadrati 'gelsd'. (Prima, UN solo
    elemento fallito mandava TUTTO il lotto al lstsq: e' il diluvio di righe LAPACK SGELSD del 4/10 —
    sistemi QUADRATI, non sottodeterminati: LAPACK chiama 'overdetermined' ogni m >= n.)"""
    L, info = torch.linalg.cholesky_ex(M)
    bad = info > 0
    if not bool(bad.any()):
        return torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)
    ok = ~bad
    s = torch.empty_like(rhs)
    if bool(ok.any()):
        s[ok] = torch.cholesky_solve(rhs[ok].unsqueeze(-1), L[ok]).squeeze(-1)
    s[bad] = _lstsq_any_device(M[bad], rhs[bad])
    return s


def _spd_a_lotti(M: torch.Tensor, rhs: torch.Tensor, lotto: int,
                 risolvi: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """Sistemi a gruppi di `lotto`; se `risolvi` solleva RuntimeError su un gruppo, quel gruppo (e solo quello)
    si risolve su CPU. Indipendente dal lotto: ogni sistema e' risolto per conto suo."""
    out = []
    for b in range(0, M.shape[0], lotto):
        m, r = M[b:b + lotto], rhs[b:b + lotto]
        try:
            out.append(risolvi(m, r))
        except RuntimeError as e:  # hipSOLVER/rocBLAS: INTERNAL_ERROR, area di lavoro nulla...
            warnings.warn(f"solve batched su GPU fallito ({str(e)[:80]}): gruppo di {m.shape[0]} su CPU")
            out.append(_spd_cpu(m.cpu(), r.cpu()).to(M.device))
    return torch.cat(out)


def risolvi_spd(M: torch.Tensor, rhs: torch.Tensor, modo: Optional[str] = None,
                lotto: Optional[int] = None) -> torch.Tensor:
    """Risolve M s = rhs per ogni elemento del lotto. M [...,P,P] simmetrica, rhs [...,P] -> [...,P].
      * M su CPU: Cholesky (stesse operazioni di prima, bit per bit se nessun elemento fallisce);
      * M su GPU, modo "cpu" (default): i sistemi sono PICCOLI (P = K*ng, 1..~100) e a milioni: si portano
        su CPU (decine di MB), si risolvono lì e tornano. Nessuna chiamata batched hipSOLVER/rocBLAS;
      * M su GPU, modo "gpu": gruppi di `lotto` (default 64) sistemi, ripiego CPU per gruppo se solleva."""
    modo = SOLVE_SU if modo is None else modo
    lotto = SOLVE_LOTTO_GPU if lotto is None else lotto
    shape = rhs.shape
    P = shape[-1]
    Mf, rf = M.reshape(-1, P, P), rhs.reshape(-1, P)
    if _su_cpu(M):
        s = _spd_cpu(Mf, rf)
    elif modo == "gpu":
        def _gpu(m, r):
            L, info = torch.linalg.cholesky_ex(m)
            if bool((info > 0).any()):
                return _spd_cpu(m.cpu(), r.cpu()).to(m.device)
            return torch.cholesky_solve(r.unsqueeze(-1), L).squeeze(-1)
        s = _spd_a_lotti(Mf, rf, max(1, lotto), _gpu)
    elif modo == "cpu":
        s = _spd_cpu(Mf.cpu(), rf.cpu()).to(M.device)
    else:
        raise ValueError(f"RD_SOLVE_SU={modo!r}: ammessi 'cpu' e 'gpu'")
    return s.reshape(shape)


@torch.no_grad()
def kfac_distortion(D: torch.Tensor, A: torch.Tensor, G: torch.Tensor,
                    mm_dtype: torch.dtype = torch.float32, chunk: int = 64) -> torch.Tensor:
    """d = 0.5 tr(G D A D^T) = 0.5 sum((G D) * (D A)), per elemento del lotto -> [B]."""
    res = []
    for b0 in range(0, D.shape[0], chunk):
        d = D[b0: b0 + chunk].float()
        da = _mm(d, A[b0: b0 + chunk], mm_dtype)
        gd = _mm(G[b0: b0 + chunk], d, mm_dtype)
        res.append(0.5 * (gd * da).sum((-1, -2)))
    return torch.cat(res)


def error_of(planes: torch.Tensor, scales: torch.Tensor, w_teacher: torch.Tensor) -> torch.Tensor:
    """D = W_q - W_teacher (base memorizzata)."""
    return dequant(planes, scales) - w_teacher


def _lstsq_any_device(M: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Ripiego per sistemi singolari. Su GPU (CUDA/ROCm) torch offre solo 'gels', che
    vuole rango pieno: il ripiego gira su CPU con 'gelsd' e torna sul dispositivo."""
    if M.device.type == "cpu":
        return torch.linalg.lstsq(M, rhs.unsqueeze(-1)).solution.squeeze(-1)
    sol = torch.linalg.lstsq(M.cpu(), rhs.unsqueeze(-1).cpu(), driver="gelsd").solution
    return sol.squeeze(-1).to(M.device)


# =========================================================================== percorsi strutturati
# Le stesse funzioni quando un fattore di Kronecker e' DIAGONALE (proxy imatrix:
# H_I = diag, H_O = I/out). Matematicamente identiche alle versioni dense, ma
# O(o*i) invece di O(o*i^2) / O(o^2*i), senza Cholesky da i^3 e senza il ciclo
# per colonna di GPTQ (con U diagonale la propagazione dell'errore e' nulla).
# Equivalenza verificata in tests/test_percorso_veloce.py.

def diag_if_diagonal(m: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """La diagonale [...,n] se OGNI matrice del lotto e' esattamente diagonale, altrimenti None."""
    if m is None:
        return None
    d = m.diagonal(dim1=-2, dim2=-1)
    if int(torch.count_nonzero(m)) == int(torch.count_nonzero(d)):
        return d.contiguous()
    return None


def damp_diag(a_diag: torch.Tensor, rel: float) -> torch.Tensor:
    """damp() sulla sola diagonale (stessa formula, stesso risultato elemento per elemento)."""
    d = a_diag.contiguous().mean(-1).clamp_min(1e-12)
    return a_diag + rel * d[..., None]


@torch.no_grad()
def gptq_ternary_diag(r: torch.Tensor, cdiag: torch.Tensor, mask: Optional[torch.Tensor] = None,
                      group: int = GROUP):
    """gptq_ternary con Hessiana DIAGONALE: U = chol(A^-1) e' diagonale, l'errore di
    una colonna non tocca le successive, quindi trit e scale sono quelli di gptq_ternary
    calcolati in un colpo solo (stesse operazioni elemento per elemento -> stessi bit)."""
    B, o, i = r.shape
    ng = i // group
    W = r.reshape(B, o, ng, group)
    c = cdiag.reshape(B, 1, ng, group)
    m = mask.reshape(B, o, ng, group) if mask is not None else None
    S = ternary_scale(W, c, m).half()                     # [B,o,ng]
    s = S.float()[..., None]
    t = torch.clamp(torch.round(W / s.clamp_min(1e-30)), -1, 1)
    t = torch.where(s > 0, t, torch.zeros_like(t))
    if m is not None:
        t = t * m
    return t.reshape(B, o, i).to(torch.int8), S


@torch.no_grad()
def refit_scales_diag(planes: torch.Tensor, target: torch.Tensor, a_diag: torch.Tensor,
                      chunk_rows: int = 128, group: int = GROUP) -> torch.Tensor:
    """refit_scales con A diagonale: il sistema K*ng x K*ng di ogni riga e' diagonale a
    blocchi (un blocco K x K per gruppo). Stessa regolarizzazione (ridge relativo alla
    media della diagonale INTERA della riga) e stesse scale nulle per le colonne vuote."""
    B, K, o, i = planes.shape
    ng = i // group
    out = torch.empty(B, K, o, ng, dtype=torch.float16, device=planes.device)
    aw = a_diag.float().reshape(B, 1, 1, ng, group)
    for r0 in range(0, o, chunk_rows):
        r1 = min(o, r0 + chunk_rows)
        r = r1 - r0
        Tg = planes[:, :, r0:r1].float().reshape(B, K, r, ng, group)
        TA = Tg * aw
        M = torch.einsum("bkrgs,blrgs->brgkl", TA, Tg)                     # [B,r,ng,K,K]
        rhs = torch.einsum("bkrgs,brgs->brgk", TA,
                           target[:, r0:r1].float().reshape(B, r, ng, group))
        diag = M.diagonal(dim1=-2, dim2=-1)                                 # [B,r,ng,K]
        ridge = 1e-7 * diag.mean((-1, -2), keepdim=True).clamp_min(1e-30) + (diag <= 0).float()
        M = M + torch.diag_embed(ridge)
        s = risolvi_spd(M, rhs)
        s = torch.where(diag > 0, s, torch.zeros_like(s))
        out[:, :, r0:r1] = s.permute(0, 3, 1, 2).half()                    # [B,K,r,ng]
    return out


def _blocks_mm(d: torch.Tensor, a_blocks: torch.Tensor, mm_dtype: torch.dtype) -> torch.Tensor:
    """d [B,o,i] @ blockdiag(a_blocks [B,nb,bs,bs]) -> [B,o,i] senza toccare gli zeri fuori blocco."""
    B, o, i = d.shape
    nb, bs = a_blocks.shape[1], a_blocks.shape[2]
    x = d.reshape(B, o, nb, bs).transpose(1, 2)                      # [B,nb,o,bs]
    return _mm(x, a_blocks, mm_dtype).transpose(1, 2).reshape(B, o, i)


@torch.no_grad()
def kfac_distortion_struct(D: torch.Tensor, A: Optional[torch.Tensor], G: Optional[torch.Tensor],
                           a_diag: Optional[torch.Tensor] = None, g_diag: Optional[torch.Tensor] = None,
                           mm_dtype: torch.dtype = torch.float32, chunk: int = 64,
                           a_blocks: Optional[torch.Tensor] = None) -> torch.Tensor:
    """d = 0.5 tr(G D A D^T) usando la forma diagonale (o diagonale a blocchi, a_blocks
    [B,nb,bs,bs]) dei fattori quando c'e'. Senza strutture ricade esattamente su kfac_distortion."""
    if a_diag is None and g_diag is None and a_blocks is None:
        return kfac_distortion(D, A, G, mm_dtype, chunk)
    res = []
    for b0 in range(0, D.shape[0], chunk):
        b1 = b0 + chunk
        d = D[b0:b1].float()
        if a_diag is not None:
            da = d * a_diag[b0:b1, None, :].float()
        elif a_blocks is not None:
            da = _blocks_mm(d, a_blocks[b0:b1], mm_dtype)
        else:
            da = _mm(d, A[b0:b1], mm_dtype)
        gd = g_diag[b0:b1, :, None].float() * d if g_diag is not None else _mm(G[b0:b1], d, mm_dtype)
        res.append(0.5 * (gd * da).sum((-1, -2)))
    return torch.cat(res)


# --------------------------------------------------------------------------- diagonale a blocchi
# Una H_I diagonale espressa nella base ruotata (Hadamard a blocchi da 1024: H S diag(a) S H)
# e' DIAGONALE A BLOCCHI: Cholesky, rifit e distorsione si fanno blocco per blocco
# (Cholesky (n/bs)^2 volte piu' economica, rifit e distorsione n/bs volte).

def diag_blocks(a: torch.Tensor, bs: int) -> torch.Tensor:
    """[B,n,n] -> blocchi diagonali contigui [B,n/bs,bs,bs]."""
    B, n, _ = a.shape
    nb = n // bs
    return a.reshape(B, nb, bs, nb, bs).diagonal(dim1=1, dim2=3).permute(0, 3, 1, 2).contiguous()


def block_diag_size(a: Optional[torch.Tensor], min_size: int) -> Optional[int]:
    """Il piu' piccolo bs (min_size, 2 min_size, ... < n) tale che OGNI matrice del lotto sia
    esattamente diagonale a blocchi bs x bs; None se non lo e' (o se a e' None)."""
    if a is None:
        return None
    n = a.shape[-1]
    total = int(torch.count_nonzero(a))
    bs = min_size
    while bs < n:
        if n % bs == 0 and int(torch.count_nonzero(diag_blocks(a, bs))) == total:
            return bs
        bs *= 2
    return None


def gptq_factor_blocks(ad_blocks: torch.Tensor) -> torch.Tensor:
    """U di gptq_factor per una matrice diagonale a blocchi, calcolato blocco per blocco
    (chol(A^-1) di una diagonale a blocchi e' diagonale a blocchi). -> U pieno [B,n,n]."""
    B, nb, bs, _ = ad_blocks.shape
    ub = gptq_factor(ad_blocks.reshape(B * nb, bs, bs)).reshape(B, nb, bs, bs)
    U = torch.zeros(B, nb * bs, nb * bs, dtype=ub.dtype, device=ub.device)
    for b in range(nb):
        U[:, b * bs:(b + 1) * bs, b * bs:(b + 1) * bs] = ub[:, b]
    return U


@torch.no_grad()
def refit_scales_blocks(planes: torch.Tensor, target: torch.Tensor, ad_blocks: torch.Tensor,
                        chunk_rows: int = 128, mm_dtype: torch.dtype = torch.float32,
                        group: int = GROUP) -> torch.Tensor:
    """refit_scales con A diagonale a blocchi: i gruppi di blocchi diversi non si
    accoppiano, quindi ogni blocco e' un rifit indipendente con ingresso bs.
    (Unica differenza numerica: il ridge relativo 1e-7 si calcola per blocco.)"""
    B, K, o, i = planes.shape
    nb, bs = ad_blocks.shape[1], ad_blocks.shape[2]
    p = planes.reshape(B, K, o, nb, bs).permute(0, 3, 1, 2, 4).reshape(B * nb, K, o, bs)
    t = target.reshape(B, o, nb, bs).permute(0, 2, 1, 3).reshape(B * nb, o, bs)
    s = refit_scales(p, t, ad_blocks.reshape(B * nb, bs, bs), chunk_rows, mm_dtype, group)
    ngb = bs // group
    return s.reshape(B, nb, K, o, ngb).permute(0, 2, 3, 1, 4).reshape(B, K, o, nb * ngb)

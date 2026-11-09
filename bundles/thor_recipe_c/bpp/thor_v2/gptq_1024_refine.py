"""Raffinamento POST-GPTQ per il tipo TQ1_0R_1024 (20/9/2026) — coordinate-
descent congiunta sulle SOLE scale per gruppo (128 pesi), a trit ternari
GIA' FISSATI da gptq_1024_lib.gptq_1024().

Motivazione (compito di Angelo): GPTQ e' greedy/sequenziale — quantizza il
gruppo g usando una scala LS "ingenua" (`_scala_un_gruppo`, minimizza l'MSE
grezzo, IGNORA la Hessiana) e poi propaga l'errore SOLO ai gruppi successivi
non ancora quantizzati, mai indietro. La scala di ogni gruppo non e' mai
quella che minimizza l'errore pesato dalla Hessiana VERA date le scale di
TUTTI gli altri gruppi.

Qui si risolve, per ogni riga e con i trit fissi, il problema ai minimi
quadrati pesato dalla Hessiana su TUTTI i gruppi della riga INSIEME (non
sequenziale):

    d* = argmin_d  (w - T(q) d)^T H (w - T(q) d)

dove T(q) e' la matrice (n_in x ng) le cui colonne sono i trit di ciascun
gruppo, imbottiti di zeri fuori dal gruppo. La soluzione in forma chiusa e':

    (T^T H T) d = T^T H w

Questo NON PUO' peggiorare l'errore pesato-H rispetto alla scala GPTQ
originale (che e' solo UNA scelta particolare di d): e' un minimo su un
insieme che la contiene. Se il verificatore trova un peggioramento, e' un
bug nell'implementazione, non un limite del metodo — vedi verifica_
raffinamento_scale.py.

GROUP=128 (scala) puo' attraversare piu' blocchi di rotazione da 1024: la
matrice H usata QUI e' la Hessiana ROTATA INTERA del tensore (H' = R H R^T,
la stessa passata a GPTQ), quindi il joint-solve cattura anche le
correlazioni FRA i blocchi di rotazione (per in_feat=2048, nb=2 blocchi,
ng=16 gruppi totali) — non solo dentro un blocco da 1024.
"""
from __future__ import annotations

import torch

from gptq_1024_lib import GROUP


@torch.no_grad()
def raffina_scale_joint(
    Wrot: torch.Tensor,
    Q: torch.Tensor,
    Hrot: torch.Tensor,
    device: str = "cuda",
    ridge: float = 1e-6,
) -> torch.Tensor:
    """Wrot: (rows, n_in) target ruotato (float). Q: (rows, n_in) trit fissi
    in {-1,0,1} (float o int, viene castato). Hrot: (n_in, n_in) Hessiana
    ROTATA (NON il fattore di Cholesky — la matrice vera, simmetrica).
    Ritorna D_new: (rows, ng) float32 — una scala per gruppo di GROUP pesi,
    ottima congiuntamente dato Q fisso.
    """
    rows, n_in = Wrot.shape
    assert n_in % GROUP == 0
    ng = n_in // GROUP

    W = Wrot.to(device=device, dtype=torch.float64)
    Qd = Q.to(device=device, dtype=torch.float64)
    H = Hrot.to(device=device, dtype=torch.float64)
    assert H.shape == (n_in, n_in)

    # b[r,g] = q_r[g] . (H w_r)[gruppo g]  — un solo matmul grande (H simmetrica)
    U = W @ H  # (rows, n_in)
    Qg = Qd.reshape(rows, ng, GROUP)
    Ug = U.reshape(rows, ng, GROUP)
    b = (Qg * Ug).sum(-1)  # (rows, ng)

    # M[r,g,g'] = q_r[g] . H[gruppo g, gruppo g'] . q_r[g']  — 136 coppie (ng=16), simmetrica
    Hb = H.reshape(ng, GROUP, ng, GROUP).permute(0, 2, 1, 3)  # (ng, ng, GROUP, GROUP)
    M = torch.empty(rows, ng, ng, device=device, dtype=torch.float64)
    for g in range(ng):
        tmp_gg = torch.einsum("ri,ij->rj", Qg[:, g, :], Hb[g, g])
        M[:, g, g] = (tmp_gg * Qg[:, g, :]).sum(-1)
        for gp in range(g + 1, ng):
            tmp = torch.einsum("ri,ij->rj", Qg[:, g, :], Hb[g, gp])
            val = (tmp * Qg[:, gp, :]).sum(-1)
            M[:, g, gp] = val
            M[:, gp, g] = val

    tr_medio = torch.diagonal(M, dim1=-2, dim2=-1).mean().clamp(min=1e-12)
    eye = torch.eye(ng, device=device, dtype=torch.float64).unsqueeze(0)
    M_reg = M + eye * (ridge * tr_medio)

    d = torch.linalg.solve(M_reg, b.unsqueeze(-1)).squeeze(-1)  # (rows, ng)
    return d.to(torch.float32).cpu()


@torch.no_grad()
def raffina_scale_joint_batch(
    Wrot: torch.Tensor,
    Q: torch.Tensor,
    Hrot: torch.Tensor,
    device: str = "cuda",
    ridge: float = 1e-6,
) -> torch.Tensor:
    """⭐ 22/9 22:40 — STESSO identico algoritmo di raffina_scale_joint, con
    una dimensione BATCH in piu' per una Hessiana DIVERSA per elemento del
    batch (un esperto ciascuno) — esattamente come raffina_trit_sweep_batch
    fa per il trit sweep. Nato perche' raffina_scale_joint chiamata 256 volte
    per round (Angelo: «deve essere veloce!!») e' un ciclo Python di
    chiamate CPU<->GPU che a scala e' diventato il vero collo di bottiglia
    (misurato: 1.1x di accelerazione totale nonostante il trit_sweep batchato
    fosse gia' perfetto). Il doppio ciclo su (g,gp) resta piccolo (ng*(ng+1)/2
    iterazioni, es. 136 per ng=16) e NON cresce con B: ogni iterazione diventa
    un einsum piu' grande (tutti gli esperti insieme), non 256 chiamate.

    ⚠️ `tr_medio` per il ridge e' calcolato PER-ESPERTO (non globale su tutto
    il batch): esperti con H di scala diversa avrebbero un ridge sballato se
    condividessero una sola media — replica esattamente cosa succederebbe
    chiamando la versione non-batch una volta per esperto.

    Wrot,Q: (B,rows,n_in) · Hrot: (B,n_in,n_in). Ritorna D_new: (B,rows,ng)
    float32 sulla CPU, stesso contratto di raffina_scale_joint.
    """
    B, rows, n_in = Wrot.shape
    assert n_in % GROUP == 0
    ng = n_in // GROUP

    W = Wrot.to(device=device, dtype=torch.float64)
    Qd = Q.to(device=device, dtype=torch.float64)
    H = Hrot.to(device=device, dtype=torch.float64)
    assert H.shape == (B, n_in, n_in)

    U = torch.matmul(W, H)  # (B, rows, n_in) — batch matmul, una H per esperto
    Qg = Qd.reshape(B, rows, ng, GROUP)
    Ug = U.reshape(B, rows, ng, GROUP)
    b = (Qg * Ug).sum(-1)  # (B, rows, ng)

    Hb = H.reshape(B, ng, GROUP, ng, GROUP).permute(0, 1, 3, 2, 4)  # (B, ng, ng, GROUP, GROUP)
    M = torch.empty(B, rows, ng, ng, device=device, dtype=torch.float64)
    for g in range(ng):
        tmp_gg = torch.einsum("bri,bij->brj", Qg[:, :, g, :], Hb[:, g, g])
        M[:, :, g, g] = (tmp_gg * Qg[:, :, g, :]).sum(-1)
        for gp in range(g + 1, ng):
            tmp = torch.einsum("bri,bij->brj", Qg[:, :, g, :], Hb[:, g, gp])
            val = (tmp * Qg[:, :, gp, :]).sum(-1)
            M[:, :, g, gp] = val
            M[:, :, gp, g] = val

    tr_medio = torch.diagonal(M, dim1=-2, dim2=-1).mean(dim=(1, 2)).clamp(min=1e-12)  # (B,)
    eye = torch.eye(ng, device=device, dtype=torch.float64).view(1, 1, ng, ng)
    M_reg = M + eye * (ridge * tr_medio).view(B, 1, 1, 1)

    d = torch.linalg.solve(M_reg, b.unsqueeze(-1)).squeeze(-1)  # (B, rows, ng)
    return d.to(torch.float32).cpu()


def ricostruisci(Q: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """Q: (rows, n_in) trit · D: (rows, ng) scala per gruppo -> (rows, n_in)."""
    rows, n_in = Q.shape
    ng = D.shape[1]
    group = n_in // ng
    Dexp = D.repeat_interleave(group, dim=1)
    return Q.to(Dexp.dtype) * Dexp


def errore_pesato_relativo(Derr: torch.Tensor, H: torch.Tensor, Wrif: torch.Tensor, device: str = "cuda") -> float:
    """sqrt(tr(Derr H Derrᵀ) / tr(Wrif H Wrifᵀ)) — stessa metrica relativa
    usata in verifica_gptq_1024.py (err_*_w), utile per confrontare run diverse.

    ⚠️ tr(D H Dᵀ) = Σ_riga ((D @ H) ⊙ D) — MAI materializzare D @ H @ Dᵀ
    (rows x rows): su un tensore con molte righe quell'intermedio esplode
    (LEVERS.md, incidente 20/9: 248.320 righe -> 493 GB, OOM globale, ucciso
    mogavis-brain-server). Qui le righe sono al piu' poche migliaia, ma la
    forma sicura costa uguale ed evita di ripetere l'incidente se qualcuno
    riusa questa funzione su un tensore piu' grande."""
    D = Derr.to(device=device, dtype=torch.float64)
    Hh = H.to(device=device, dtype=torch.float64)
    W = Wrif.to(device=device, dtype=torch.float64)
    num = (D @ Hh * D).sum()
    den = (W @ Hh * W).sum()
    return float(torch.sqrt(torch.abs(num) / torch.abs(den)))

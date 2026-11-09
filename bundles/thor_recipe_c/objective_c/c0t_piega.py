"""Obiettivo C / C0T (9/10): nucleo della PIEGA di un guadagno per colonna g (norma RMSNorm) in un consumatore gia'
ternario a UN piano, e requantizzazione ternaria. Matematica (test_c0t_matematica.py T1/T2/T4, esatta in fp64):
  y = W (x^ * g) = (W diag(g)) x^   =>   bersaglio T = Q_C0 diag(g)  (riproduce AL BIT la funzione di C0),
  metrica = Hessiana dell'ingresso SENZA guadagno: H_x^ = D H_x D, D = diag(1/g).
In base ruotata (TQ1_0R_1024): T' = Q' R diag(g) R^T (= ruota(deruota(Q') * g)), H' = R H_x^ R^T.
Vie (scelta MISURATA per tensore, vince l'errore H piu' basso):
  J = JARQ dai codici di C0 (cardan_piega.jarq_lotto: scale ai minimi quadrati + Babai, righe peggiori tornano a C0)
  F = quantizzazione da zero di T (tt_quant.quantizza_1024 / quantizza_tq1_0, ricetta 20set, la stessa di C0)
Errore riportato: sqrt(sum (T-Q) H (T-Q)^T / sum T H T^T) = errore d'uscita AGGIUNTO rispetto a C0, relativo."""
import os, sys
sys.path.insert(0, "@@ROOT:models@@/build/thor-rexppiani2/gguf-py")
sys.path.insert(0, "@@ROOT:models@@/build/leve/raffina")
sys.path.insert(0, "@@ROOT:models@@/gguf/tony-tern/CARDAN/codice")
sys.path.insert(0, "@@ROOT:depot@@/tutto_ternario/codice")
os.environ.setdefault("TT_DEVICE", "cpu")
import numpy as np
import torch
import comune as CM
import cardan_piega as CP
import tt_quant as TQ

gp = CM.gp


def norm_H(H):
    H = 0.5 * (H + H.T)
    d = torch.diagonal(H).clone()
    morti = d <= 0
    if morti.any():
        H[morti, morti] = d[~morti].mean()
    return H / torch.diagonal(H).mean()


def H_senza_guadagno(H_x: np.ndarray, g: np.ndarray) -> torch.Tensor:
    Hx = torch.from_numpy(np.asarray(H_x, np.float64))
    gi = torch.from_numpy(1.0 / np.asarray(g, np.float64))
    return norm_H(Hx * gi[:, None] * gi[None, :])


def err_rel(T, Q, H):
    E = T - Q
    return float((((E @ H) * E).sum() / ((T @ H) * T).sum()) ** 0.5)


def piega_J(T0, S0, g, Hx_hat, ruotato, righe=8192):
    """T0 int8 [R,C], S0 fp16-valori [R,ng] (base memorizzata). Ritorna (T int8, S fp16-valori, misure)."""
    R, C = T0.shape
    gt = torch.from_numpy(np.asarray(g, np.float64))
    Hm = CM.ruota_H(Hx_hat) if ruotato else Hx_hat
    M = CM.ruota_H(torch.diag(gt)) if ruotato else None
    Tn = torch.empty_like(T0); Sn = torch.empty_like(S0, dtype=torch.float64)
    num = den = num0 = 0.0
    for a in range(0, R, righe):
        b = min(R, a + righe)
        Q0 = (T0[a:b].double().reshape(b - a, S0.shape[1], -1) * S0[a:b].double().unsqueeze(-1)).reshape(b - a, C)
        W = Q0 @ M if ruotato else Q0 * gt
        T2, S2, cat = CP.jarq_lotto(W[None], Hm[None], T0[None, None, a:b], S0[None, None, a:b].double(), giri=3, nu2=1e-4)
        Tn[a:b] = T2[0, 0]; Sn[a:b] = S2[0, 0]
        Q = CP.ricostruisci_lotto(T2, S2)[0]
        E = W - Q; num += float(((E @ Hm) * E).sum()); den += float(((W @ Hm) * W).sum())
        E0 = W - Q0; num0 += float(((E0 @ Hm) * E0).sum())
    return Tn, Sn, {"via": "J", "err_rel_H": (num / den) ** 0.5, "err_rel_H_senza_piega": (num0 / den) ** 0.5}


def piega_F(T0, S0, g, Hx_hat_raw, ruotato, tid):
    """Quantizzazione da zero del bersaglio (base ORIGINALE) con la ricetta di C0; ritorna (T, S, misure) in base memorizzata."""
    R, C = T0.shape
    Q0 = (T0.double().reshape(R, S0.shape[1], -1) * S0.double().unsqueeze(-1)).reshape(R, C)
    Wo = (CM.deruota(Q0) if ruotato else Q0) * torch.from_numpy(np.asarray(g, np.float64))
    W32 = Wo.float().numpy()
    if ruotato:
        packed, mis = TQ.quantizza_1024(W32, Hx_hat_raw, "")
    else:
        packed, mis = TQ.quantizza_tq1_0(W32, Hx_hat_raw, "")
    T, S = gp.decode_rows(packed.reshape(R, -1), tid, C)
    return torch.from_numpy(T), torch.from_numpy(S.astype(np.float32)).double(), {"via": "F", **{k: mis[k] for k in mis if k.startswith("err") or k == "secondi"}}


def err_in_base(T, S, Tb, Sb, g, Hx_hat, ruotato):
    """errore relativo H del candidato (T,S) rispetto al bersaglio (codici C0 Tb,Sb piegati per g), base memorizzata."""
    R, C = Tb.shape
    gt = torch.from_numpy(np.asarray(g, np.float64))
    Q0 = (Tb.double().reshape(R, Sb.shape[1], -1) * Sb.double().unsqueeze(-1)).reshape(R, C)
    W = Q0 @ CM.ruota_H(torch.diag(gt)) if ruotato else Q0 * gt
    Q = (T.double().reshape(R, S.shape[1], -1) * S.double().unsqueeze(-1)).reshape(R, C)
    Hm = CM.ruota_H(Hx_hat) if ruotato else Hx_hat
    return err_rel(W, Q, Hm)


def codifica(T, S, tid):
    return gp.encode_rows(T.numpy().astype(np.int8), S.to(torch.float16).numpy(), tid)

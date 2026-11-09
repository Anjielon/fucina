#!/usr/bin/env python3
"""SchurQuant (arXiv:2608.15567) — il pezzo matematico del componente
«teacher reconstruction su prefisso quantizzato», VERIFICATO nella convenzione
della pipeline vera (v2, 22/9 07:00, dopo 5 revisioni avversarie della v1).

Il paper, §3.1:
  Eq.7  L(W) = ½‖(W X − Y) Ω^½‖²_F + (λ/2)‖W − W_ref‖²_F        (Ω diagonale, pesi per token)
  Eq.8  G = X Ω Xᵀ + λI          C = Y Ω Xᵀ + λ W_ref            (statistiche sufficienti; λ DENTRO)
  Eq.9  L(W) = ½ tr(W G Wᵀ) − tr(C Wᵀ) + const
Teacher-reconstruction = Eq.7-9 con X = X_q (ingresso del PREFISSO QUANTIZZATO)
e Y = W_ref X_ref (uscita del maestro PULITO). Il minimo non vincolato risolve
W G = C. Per ogni Wq: L(Wq) − L(W*) = ½ tr((Wq − W*) G (Wq − W*)ᵀ) — la forma
che `gptq_1024` (Cholesky di H⁻¹, eliminazione colonna per colonna = Schur a
chunk 1) e `raffina_coorddesc` minimizzano gia', passando Wrot := W*, Hrot := G.

CONVENZIONE DELLA PIPELINE (cattura_hessiane_*_thor.py:151,280): i TOKEN
stanno sulle RIGHE. X_q, X_ref: (n_tok, in_feat); Y: (n_tok, out_rows);
G = X_qᵀ Ω X_q; C = Yᵀ Ω X_q = W_ref (X_refᵀ Ω X_q). La rotazione R agisce
sull'ULTIMO asse (in_feat) di X_q e W; Y non si ruota mai.
SMORZAMENTO come `gptq_1024_lib.prepara_hessiana`: diagonali morte → 1, poi
+ 0.01·mean(diag)·I; risoluzione via Cholesky, MAI pinv/inv esplicita.

Cosa hanno trovato le 5 revisioni della v1 (tutto corretto qui):
- paper: loss = Eq.9 (la v1 la chiamava Eq.8 ed era il DOPPIO); λ dentro G,C;
  smorzamento Tikhonov, non pinv; «Schur» = eliminazione del suffisso, che il
  nostro gptq_1024 gia' fa; MoE mai trattato dal paper.
- numerica: allclose senza rtol → soglia effettiva 0.59 su ~6e4; ora rtol=0 e
  errore RELATIVO esplicito. Test negativi aggiunti (devono FALLIRE).
- rotazione: la v1 aveva i token sulle COLONNE: ruotare quell'asse e' un bug
  silenzioso se n_tok % 1024 == 0. Qui la rotazione si prova con le funzioni
  VERE del motore.
- audit: con G mal condizionata (cond ~1e13) pinv sbaglia di 0.72 e l'identita'
  numerica salta; con lo smorzamento del motore regge. Il vecchio «test 3» era
  circolare: qui e' etichettato come sanity check della risoluzione.
"""
from __future__ import annotations
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernel-rotazione"))
from gptq_1024_lib import QK, R_1024_matrice, ruota_hessiana_1024, ruota_righe_1024  # noqa: E402

torch.manual_seed(7)
DT = torch.float64
SMORZ = 0.01          # identico a prepara_hessiana
ESITI: list[tuple[str, bool, str]] = []


def esito(nome: str, ok: bool, dettaglio: str):
    ESITI.append((nome, ok, dettaglio))
    print(f"{'✅' if ok else '⛔'} {nome}: {dettaglio}")


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / max(float(b.abs().max()), 1e-300))


def statistiche(X_q: torch.Tensor, X_ref: torch.Tensor, W_ref: torch.Tensor, omega: torch.Tensor, lam: float):
    """Eq.8 nella convenzione token-sulle-righe. omega: (n_tok,) ≥ 0, normalizzata a media 1
    (come nel paper, cosi' λ e' confrontabile SUL PESO relativo dei token). Ritorna G (in,in),
    C (out,in), const.
    ⛔⛔ 22/9 07:35 — trovato sul primo caso reale (esperto 124, strato 10): senza
    dividere per n_tok, G ha scala O(n_tok · varianza) — misurato 9.03e3 su 14303
    token — e λ=1 diventa TRASCURABILE, ancora piu' trascurabile con un corpus piu'
    grande. λ deve avere un peso comparabile a un G "per-token-medio" (come la
    convenzione H_sum/n_tot gia' in uso in tutta la pipeline, gptq_1024_lib) perche'
    il suo significato NON dipenda dalla taglia del corpus. Divido per n_tok: W* =
    C G⁻¹ e' INVARIANTE a un riscalamento uniforme di (G,C) — verificato dal
    revisore "rotazione" — quindi questo non cambia nulla SENZA λ; CON λ, ora
    λ=1 e' comparabile alla scala naturale di un G normalizzato (~O(1) per feature
    con attivazioni a varianza unitaria), non piu' trascurabile per costruzione."""
    assert (omega >= 0).all(), "Ω con pesi negativi: G non sarebbe PSD — rifiutato"
    n_tok = X_q.shape[0]
    om = (omega / omega.mean()) / n_tok
    Xw = X_q * om[:, None]
    G = X_q.T @ Xw                                   # (1/n) X_qᵀ Ω X_q
    C = W_ref @ (X_ref.T @ Xw)                       # (1/n) W_ref (X_refᵀ Ω X_q)
    Y = X_ref @ W_ref.T                              # (n_tok, out)
    const = 0.5 * float((Y * Y * om[:, None]).sum())  # ½ (1/n) tr(Yᵀ Ω Y)
    n_in = G.shape[0]
    return G + lam * torch.eye(n_in, dtype=DT), C + lam * W_ref, const


def smorza(G: torch.Tensor) -> torch.Tensor:
    """prepara_hessiana: diagonali morte → 1, poi + SMORZ·mean(diag)·I."""
    Gd = G.clone()
    d = torch.diagonal(Gd)
    morti = d == 0
    d[morti] = 1.0
    # ⛔ 22/9 18:45 — `device=G.device`: senza, l'identita' nasceva sempre su
    # CPU e la funzione esplodeva appena G stava su GPU. Non si era mai visto
    # perche' finora `smorza` era chiamata solo dal controllo su CPU; la forgia
    # per-strato la chiama con G in VRAM. Una funzione pura deve seguire il
    # device del suo ingresso, non imporne uno.
    Gd += torch.eye(G.shape[0], dtype=DT, device=G.device) * (SMORZ * d.mean())
    return Gd


def w_star(G_d: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """Risolve W G_d = C  ⇔  G_d Wᵀ = Cᵀ (G_d simmetrica) via Cholesky. Mai inversa esplicita."""
    L = torch.linalg.cholesky(G_d)
    return torch.cholesky_solve(C.T.contiguous(), L).T


def loss9(W: torch.Tensor, G: torch.Tensor, C: torch.Tensor, const: float) -> torch.Tensor:
    return 0.5 * torch.trace(W @ G @ W.T) - torch.trace(C @ W.T) + const


def caso_base(n_tok=4096, in_feat=QK, out=8, lam=1.0, seme=7):
    g = torch.Generator().manual_seed(seme)
    X_q = torch.randn(n_tok, in_feat, dtype=DT, generator=g)
    X_ref = X_q + 0.3 * torch.randn(n_tok, in_feat, dtype=DT, generator=g)   # prefisso quantizzato ≠ pulito
    W_ref = 0.3 * torch.randn(out, in_feat, dtype=DT, generator=g)
    omega = torch.rand(n_tok, dtype=DT, generator=g) + 0.1
    return X_q, X_ref, W_ref, omega, lam


def main():
    X_q, X_ref, W_ref, omega, lam = caso_base()
    G, C, const = statistiche(X_q, X_ref, W_ref, omega, lam)
    Gd = smorza(G)
    Ws = w_star(Gd, C)

    # T1 — W* e' stazionario per la loss SMORZATA (quella che il motore vede).
    #      Prova ESATTA: gradiente analitico ∇L = W G_d − C (non una differenza finita:
    #      la v2a usava ε=1e-6 su una loss ~1e5 e leggeva come "gradiente" il rumore di
    #      arrotondamento, che cresce come 1/ε — misurato 22/9 07:05). La differenza
    #      finita resta come controllo incrociato con ε=1: per una QUADRATICA la
    #      differenza centrale e' esatta per ogni ε, e con ε grande il rumore e' ~1e-11.
    grad = Ws @ Gd - C
    res = float(grad.abs().max() / C.abs().max())
    esito("T1a gradiente analitico W*G_d−C = 0", res <= 1e-12, f"residuo relativo {res:.3e}")
    eps = 1.0
    dirz = torch.randn_like(Ws)
    der = (loss9(Ws + eps * dirz, Gd, C, const) - loss9(Ws - eps * dirz, Gd, C, const)) / (2 * eps)
    scala = float(loss9(Ws, Gd, C, const).abs())
    rumore = 4 * 2.2e-16 * scala / (2 * eps)
    esito("T1b differenza finita (ε=1, esatta su quadratica)", abs(float(der)) <= 100 * rumore + 1e-9,
          f"derivata {float(der):.3e}, rumore atteso {rumore:.1e}, analitica {float((grad * dirz).sum()):.3e}")

    # T2 — identita' L(Wq)−L(W*) = ½ tr((Wq−W*) G_d (Wq−W*)ᵀ) per Wq ARBITRARI, errore RELATIVO, rtol=0
    peggiore = 0.0
    for k, s in enumerate((1.0, 10.0, 1e3, 1e-4, -5.0)):
        Wq = s * torch.randn_like(Ws)
        lhs = loss9(Wq, Gd, C, const) - loss9(Ws, Gd, C, const)
        D = Wq - Ws
        rhs = 0.5 * torch.trace(D @ Gd @ D.T)
        peggiore = max(peggiore, float((lhs - rhs).abs() / rhs.abs()))
    esito("T2 identita' (5 scale)", peggiore <= 1e-10, f"errore relativo max {peggiore:.3e}")

    # T3 — SANITY CHECK della risoluzione (non prova il paper): λ=0, X_q=X_ref, smorzamento→0 ⇒ W* ≈ W_ref
    G0, C0, _ = statistiche(X_ref, X_ref, W_ref, omega, 0.0)
    Ws0 = torch.cholesky_solve(C0.T.contiguous(), torch.linalg.cholesky(G0)).T
    esito("T3 sanity (X_q=X_ref, λ=0 → W*=W_ref)", rel(Ws0, W_ref) <= 1e-9, f"errore relativo {rel(Ws0, W_ref):.3e}")

    # T4 — C economica (mai materializzare Y) == C diretta (stessa normalizzazione 1/n_tok di statistiche())
    om = (omega / omega.mean()) / X_q.shape[0]
    Y = X_ref @ W_ref.T
    C_dir = (Y.T @ (X_q * om[:, None])) + lam * W_ref
    esito("T4 C economica == C diretta", rel(C, C_dir) <= 1e-12, f"errore relativo {rel(C, C_dir):.3e}")

    # T5 — ROTAZIONE con le funzioni VERE del motore: R sull'ULTIMO asse di X_q (in_feat), Y mai ruotata.
    #      G_rot deve essere ruota_hessiana_1024(G), W*_rot deve essere W* Rᵀ (= ruota_righe_1024(W*)).
    R = torch.from_numpy(R_1024_matrice())                         # float64, (1024,1024), R Rᵀ = I
    esito("T5a R ortogonale", rel(R @ R.T, torch.eye(QK, dtype=DT)) <= 1e-12, f"max|RRᵀ−I| = {float((R @ R.T - torch.eye(QK, dtype=DT)).abs().max()):.3e}")
    X_q_rot = X_q @ R.T                                            # ruota l'asse in_feat (righe = token)
    X_ref_rot = X_ref @ R.T
    W_ref_rot = W_ref @ R.T
    G_rot, C_rot, _ = statistiche(X_q_rot, X_ref_rot, W_ref_rot, omega, lam)
    Ws_rot = w_star(smorza(G_rot), C_rot)
    G_rot_atteso = torch.from_numpy(ruota_hessiana_1024(G.numpy()))
    esito("T5b G_rot == ruota_hessiana_1024(G)", rel(G_rot, G_rot_atteso) <= 1e-9, f"errore relativo {rel(G_rot, G_rot_atteso):.3e}")
    esito("T5c W*_rot == W* Rᵀ", rel(Ws_rot, Ws @ R.T) <= 1e-8, f"errore relativo {rel(Ws_rot, Ws @ R.T):.3e}")
    # la funzione di produzione (float32) deve concordare con R esplicita entro la precisione float32
    Ws_rot_prod = torch.from_numpy(ruota_righe_1024(Ws.numpy())).to(DT)
    esito("T5d ruota_righe_1024(W*) == W* Rᵀ (float32)", rel(Ws_rot_prod, Ws @ R.T) <= 1e-5, f"errore relativo {rel(Ws_rot_prod, Ws @ R.T):.3e}")
    # ⛔ controllo dell'ASSE: ruotare i TOKEN (asse 0) invece delle feature deve dare un G DIVERSO
    if X_q.shape[0] % QK == 0:
        X_q_asse_sbagliato = torch.from_numpy(ruota_righe_1024(X_q.T.numpy())).to(DT).T   # ruota l'asse token
        G_sb, _, _ = statistiche(X_q_asse_sbagliato, X_ref, W_ref, omega, lam)
        esito("T5e asse sbagliato (token) rilevato", rel(G_sb, G_rot_atteso) > 1e-2, f"errore relativo {rel(G_sb, G_rot_atteso):.3e} (deve essere GRANDE)")

    # T6 — G MAL CONDIZIONATA (feature quasi collineari): lo smorzamento del motore regge, pinv no
    g = torch.Generator().manual_seed(11)
    base = torch.randn(4096, 8, dtype=DT, generator=g)
    mix = torch.randn(8, QK, dtype=DT, generator=g)
    X_bad = base @ mix + 1e-7 * torch.randn(4096, QK, dtype=DT, generator=g)   # rango ~8 su 1024
    Gb, Cb, cb = statistiche(X_bad, X_bad + 0.3 * torch.randn_like(X_bad), W_ref, omega, lam)
    cond = float(torch.linalg.cond(Gb))
    Gbd = smorza(Gb); Wsb = w_star(Gbd, Cb)
    Wq = torch.randn_like(Wsb)
    lhs = loss9(Wq, Gbd, Cb, cb) - loss9(Wsb, Gbd, Cb, cb); D = Wq - Wsb; rhs = 0.5 * torch.trace(D @ Gbd @ D.T)
    esito("T6 G mal condizionata + smorzamento", float((lhs - rhs).abs() / rhs.abs()) <= 1e-9,
          f"cond(G)={cond:.2e}, errore relativo identita' {float((lhs - rhs).abs() / rhs.abs()):.3e}")

    # T7 — NEGATIVO: C costruita da un X_q DIVERSO da quello di G (bug di allineamento cache) → deve FALLIRE
    g2 = torch.Generator().manual_seed(99)
    X_altro = torch.randn(X_q.shape, dtype=DT, generator=g2)
    _, C_dis, _ = statistiche(X_altro, X_ref, W_ref, omega, lam)   # C con X_altro, G resta quella di X_q
    Ws_dis = w_star(Gd, C_dis)
    der_dis = (loss9(Ws_dis + eps * dirz, Gd, C, const) - loss9(Ws_dis - eps * dirz, Gd, C, const)) / (2 * eps)
    esito("T7 NEGATIVO X_q disallineato fra G e C → non stazionario", abs(float(der_dis)) > 1e-3 * max(scala, 1.0),
          f"derivata {float(der_dis):.3e} (deve essere GRANDE)")

    # T8 — NEGATIVO: segno sbagliato in W* → l'identita' deve FALLIRE
    Wq = torch.randn_like(Ws)
    lhs = loss9(Wq, Gd, C, const) - loss9(-Ws, Gd, C, const); D = Wq + Ws; rhs = 0.5 * torch.trace(D @ Gd @ D.T)
    esito("T8 NEGATIVO segno sbagliato → identita' rotta", float((lhs - rhs).abs() / rhs.abs()) > 1e-3,
          f"errore relativo {float((lhs - rhs).abs() / rhs.abs()):.3e} (deve essere GRANDE)")

    # T9 — Ω con zeri (token mascherati, come un esperto raro) e G singolare (n_tok < in_feat)
    Xs, Xr, Wr, om9, _ = caso_base(n_tok=512, seme=5)
    om9[::2] = 0.0
    G9, C9, c9 = statistiche(Xs, Xr, Wr, om9, lam)
    G9d = smorza(G9); Ws9 = w_star(G9d, C9)
    Wq = torch.randn_like(Ws9)
    lhs = loss9(Wq, G9d, C9, c9) - loss9(Ws9, G9d, C9, c9); D = Wq - Ws9; rhs = 0.5 * torch.trace(D @ G9d @ D.T)
    esito("T9 Ω con zeri + n_tok<in_feat", float((lhs - rhs).abs() / rhs.abs()) <= 1e-9,
          f"rango(G)={int(torch.linalg.matrix_rank(G9))}/{QK}, errore relativo {float((lhs - rhs).abs() / rhs.abs()):.3e}")

    # T10 — Ω negativa deve essere RIFIUTATA
    try:
        om_neg = omega.clone(); om_neg[0] = -1.0
        statistiche(X_q, X_ref, W_ref, om_neg, lam)
        esito("T10 Ω negativa rifiutata", False, "NON rifiutata")
    except AssertionError as e:
        esito("T10 Ω negativa rifiutata", True, str(e)[:40])

    tutti = all(ok for _, ok, _ in ESITI)
    print("\n" + (f"✅ {len(ESITI)}/{len(ESITI)} — riduzione verificata nella convenzione della pipeline; si puo' costruire sopra"
                  if tutti else f"⛔ {sum(1 for _, ok, _ in ESITI if not ok)} controlli falliti — NON PROCEDERE"))
    return tutti


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)

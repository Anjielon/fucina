#!/usr/bin/env python3
"""e1_perdita_legame.py — perdita E1 per il LEGAME lettera <-> contenuto (3/10, RICERCA_CLOUD_1820.md).

Sui 4 logit lettera z (ultimo token di ogni prompt) di gruppi INTERI di 4 rotazioni:

  L = w_kd  * KL( p_maestro(rotazione r) || p_studente(rotazione r) ) * T^2      maestro alla STESSA rotazione
    + lam   * mean_r KL( q_r || m )        q_r riportata al CONTENUTO (perm), m = media_r q_r  (coerenza fra rotazioni)
    + w_pif * PIF                                                               negativi puntuali (arXiv 2406.01026)

PIF (Point-wise Intelligent Feedback): ogni coppia (lettera, contenuto) e' un esempio a se', giudicato da solo
(perdita PUNTUALE, non a coppie come DPO). Il positivo e' la lettera del contenuto giusto; i negativi sono i contenuti
SBAGLIATI, e con le 4 rotazioni ogni contenuto sbagliato compare legato a TUTTE e 4 le lettere: e' esattamente il
negativo che insegna "la lettera non conta, conta cio' che c'e' scritto accanto".
  PIF = mean_righe [ -log p(oro)  +  (1/3) sum_{sbagliate} -log(1 - p(sbagliata)) ]
(p = softmax sulle 4 lettere). Il termine positivo da solo e' la SFT; quello negativo e' l'unlikelihood puntuale.
`pif_negativi_soli=True` tiene solo i negativi (se il maestro deve restare l'unico riferimento per il positivo).

Le funzioni di riallineamento e la KL sono quelle di e1_letter_kd (stesse convenzioni: perm[p] = contenuto in
posizione p). Autoverifica: test_residui_ternari.py (sezione perdita) e `python3 e1_perdita_legame.py`.
"""
from __future__ import annotations

import re

import torch
import torch.nn.functional as F

from e1_letter_kd import debias_teacher, letter_kd_loss, rotation_consistency_loss

# fonti che NON entrano mai (ne' in allenamento ne' in selezione): MMLU e il suo auxiliary_train
FONTI_VIETATE = ("mmlu", "arc", "openbookqa", "obqa", "race", "mctest")
FONTI_SELEZIONE = ("logiqa", "medmcqa")


def fonte_vietata(source: str) -> bool:
    """MMLU ovunque nel nome (sottostringa); le altre come parola intera ("ai2_arc", "allenai/openbookqa", "race")."""
    s = str(source).lower()
    toks = set(re.split(r"[^a-z0-9]+", s))
    return "mmlu" in s or "openbookqa" in s or "mctest" in s or bool(toks & set(FONTI_VIETATE))


def pif_loss(z: torch.Tensor, gold_pos: torch.Tensor, solo_negativi: bool = False) -> torch.Tensor:
    """Perdita puntuale PIF sui 4 logit [B,4]; gold_pos [B]."""
    lp = F.log_softmax(z.float(), -1)
    p = lp.exp()
    oro = F.one_hot(gold_pos, 4).bool()
    neg = -torch.log1p(-p.clamp(max=1 - 1e-6))                     # -log(1 - p)
    l_neg = (neg * (~oro)).sum(-1) / 3.0
    if solo_negativi:
        return l_neg.mean()
    return (-lp[oro] + l_neg).mean()


def perdita_legame(z: torch.Tensor, batch: dict, w_kd: float = 1.0, lam: float = 0.5, w_pif: float = 0.3,
                   T: float = 1.0, target: str = "raw", pif_negativi_soli: bool = False) -> tuple:
    perm, gi, G = batch["perm"], batch["group_index"], batch["n_groups"]
    tgt = batch["teacher_logprobs"]
    if target == "debiased":
        tgt = debias_teacher(tgt, perm, gi, G)
    zero = z.new_zeros(()).float()
    kd = letter_kd_loss(z, tgt, T) if w_kd > 0 else zero
    rc = rotation_consistency_loss(z, perm, gi, G) if lam > 0 else zero
    pf = pif_loss(z, batch["gold_pos"], pif_negativi_soli) if w_pif > 0 else zero
    loss = w_kd * kd + lam * rc + w_pif * pf
    with torch.no_grad():
        pred = z.argmax(-1)
        st = {"loss": float(loss), "kd": float(kd), "rot": float(rc), "pif": float(pf),
              "acc": float((pred == batch["gold_pos"]).float().mean()),
              "agree_teacher": float((pred == batch["teacher_logprobs"].argmax(-1)).float().mean()),
              "pred_hist": torch.bincount(pred, minlength=4).tolist()}
    return loss, st


def metriche_selezione(z: torch.Tensor, batch: dict, sources: list) -> dict:
    """Metriche di SELEZIONE (held-out LogiQA/MedMCQA): accuratezza, RStd (dev. std. del richiamo per lettera, in %),
    coerenza fra rotazioni (frazione di gruppi con la stessa scelta di contenuto in tutte e 4), per fonte e totale."""
    pred = z.argmax(-1)
    gold = batch["gold_pos"]
    perm, gi, G = batch["perm"], batch["group_index"], batch["n_groups"]
    pc = perm.gather(1, pred[:, None]).squeeze(1)                  # contenuto scelto
    out = {}
    for nome in sorted(set(sources)) + ["tutte"]:
        m = torch.tensor([nome == "tutte" or s == nome for s in sources], device=z.device)
        if not m.any():
            continue
        rec = [float((pred[m & (gold == k)] == k).float().mean()) for k in range(4) if (m & (gold == k)).any()]
        gsel = gi[m].unique()
        coer = sum(1 for g in gsel.tolist() if pc[gi == g].unique().numel() == 1) / max(1, gsel.numel())
        out[nome] = {"acc": float((pred[m] == gold[m]).float().mean()), "n": int(m.sum()),
                     "RStd": float(torch.tensor(rec).std(unbiased=False) * 100) if len(rec) > 1 else 0.0,
                     "coerenza_rot": coer}
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    perms = torch.tensor([[(p + r) % 4 for p in range(4)] for _ in range(2) for r in range(4)])
    gi = torch.tensor([0] * 4 + [1] * 4)
    gold = torch.tensor([perms[i].tolist().index(0) for i in range(8)])
    b = {"perm": perms, "group_index": gi, "n_groups": 2, "gold_pos": gold,
         "teacher_logprobs": F.log_softmax(torch.randn(8, 4), -1)}
    w = torch.zeros(8, 4, requires_grad=True)
    opt = torch.optim.SGD([w], lr=0.5)
    for i in range(60):
        loss, st = perdita_legame(w, b)
        opt.zero_grad(); loss.backward(); opt.step()
        if i == 0:
            l0 = st["loss"]
    print(f"autoverifica: {l0:.4f} -> {st['loss']:.4f}", metriche_selezione(w.detach(), b, ["logiqa"] * 8))

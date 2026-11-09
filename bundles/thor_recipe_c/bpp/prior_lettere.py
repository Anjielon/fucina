#!/usr/bin/env python3
"""prior_lettere.py — lever L6 of SINTESI_1025 (26/10): batch-level answer-letter PRIOR term for the joint CARDAN trainer.

Why (~/tony-forgia/ricerche/SINTESI_1025/R6_kl24_vs_mmlu.md §2-§3, §7.4a): the Hessian re-ternarised routed-expert downs
shift the model's answer-letter prior (MMLU-2000: A chosen 345 times by TRE-KLP, 412 by T400, 502 by the teacher; KL of the
letter marginal vs the teacher 0.0104 against 0.0046). The per-question KL4 in the loss kept the prior in place for T400,
but it is memorised question by question (train KL4 -> 0.01, held-out flat). The prior has only 3 degrees of freedom, so a
term on the MEAN letter distribution alone can fix it without any per-question target.

Term (one step, a batch of Q calibration questions = the questions of N KL4 packs):
    pbar_S = (1/Q) sum_q softmax(student letter logits of q)      (4-vector, differentiable)
    pbar_T = (1/Q) sum_q pT_q                                      (teacher 4-letter distributions of the same questions)
    L_prior = W * KL(pbar_T || pbar_S) = W * sum_j pbar_T[j] (log pbar_T[j] - log pbar_S[j])
Gradient w.r.t. the letter logits z_q of question q:  dL/dz_q = -(W/Q) (diag(p_q) - p_q p_q^T) r,  r = pbar_T / pbar_S,
i.e. every question receives the push of ONE shared 4-vector r: the term is invariant to any permutation of the student
distributions across questions, so it cannot store a per-question answer.

Running estimate (--prior-lettere-ema BETA > 0): rS = BETA sg(rS_prev) + (1 - BETA) pbar_S, rT = BETA rT_prev +
(1 - BETA) pbar_T (the teacher mean with the SAME weights, so rT and rS always describe the same weighted question set), loss
= KL(rT || rS); the gradient flows through the current batch only and is the literal derivative of that loss, i.e. (1 - BETA)
x the batch-form gradient evaluated at the running point (effective weight W (1 - BETA) on the batch). Measured on a toy
(L6_prior_lettere.md §4): the straight-through variant (value = running estimate, unit-scale gradient, = this form with
W / (1 - BETA)) is a delayed-feedback loop that oscillates and distorts per-question distributions for BETA >= 0.8 under Adam,
so it is NOT offered. The first step starts the EMA at the batch itself (value = batch KL, gain (1 - BETA)).
The running state is committed only when the optimiser step is taken (conferma), never on a skipped step (scarta).

Used by cardan_congiunto_scale.py (--prior-lettere-peso / --prior-lettere-pacchi / --prior-lettere-ema). Pure torch, no I/O.
"""
from __future__ import annotations

from typing import Optional

import torch


def offsets(n: int, P: int) -> list:
    """Offsets of the N packs of a step inside the cycle of P packs: evenly spread, distinct for N <= P, first = 0."""
    return [(k * P) // n for k in range(n)]


def kl_marginale(pT: torch.Tensor, pS: torch.Tensor) -> torch.Tensor:
    """KL(pT || pS) of two 4-letter marginals (same clamp as the trainer's KL4)."""
    return (pT * (torch.log(pT.clamp_min(1e-30)) - torch.log(pS.clamp_min(1e-30)))).sum()


class PriorLettere:
    """State and maths of the term; the trainer supplies the student log-probabilities and the teacher distributions."""

    def __init__(self, peso: float, pacchi: int = 1, beta: float = 0.0) -> None:
        self.peso, self.n_pacchi, self.beta = float(peso), int(pacchi), float(beta)
        self.S: Optional[torch.Tensor] = None           # running student mean (EMA only)
        self.T: Optional[torch.Tensor] = None           # running teacher mean (EMA only)
        self.n = 0                                      # committed EMA updates
        self._prossimo = None

    @staticmethod
    def errore_config(peso: float, pacchi: int, beta: float) -> Optional[str]:
        if peso < 0:
            return "--prior-lettere-peso non puo' essere negativo"
        if pacchi < 1:
            return "--prior-lettere-pacchi vuole N >= 1"
        if not 0.0 <= beta < 1.0:
            return "--prior-lettere-ema vuole 0 <= beta < 1"
        return None

    def pacchi(self, passo: int, pacco_del_passo, P: int) -> list:
        """The N packs of a step: pacco_del_passo(passo + offset); offset 0 = the step's own KL4 pack."""
        return [pacco_del_passo(passo + o) for o in offsets(self.n_pacchi, P)]

    def termine(self, lS_list: list, pT_list: list) -> tuple:
        """(KL tensor, info) for one batch. lS_list: [Q_k, 4] student letter log-softmax per pack (with grad);
        pT_list: [Q_k, 4] teacher distributions. The weight W is applied by the caller."""
        pS_b = torch.cat([l.exp() for l in lS_list]).mean(0)
        pT_b = torch.cat(pT_list).to(device=pS_b.device, dtype=pS_b.dtype).mean(0)
        info = {}
        if self.beta > 0:
            S0 = pS_b.detach() if self.S is None else self.S   # first step: the EMA starts at the batch itself, so the
            T0 = pT_b if self.T is None else self.T             # value is the batch KL and the gain is (1 - BETA) from step 1
            rT = self.beta * T0 + (1.0 - self.beta) * pT_b
            rS = self.beta * S0 + (1.0 - self.beta) * pS_b      # S0 detached: gradient through the current batch only
            self._prossimo = (rS.detach().clone(), rT.detach().clone())
            pS_eff = rS
            info["prior_batch"] = float(kl_marginale(pT_b, pS_b.detach()))
        else:
            pS_eff, rT = pS_b, pT_b
        return kl_marginale(rT, pS_eff), info

    def conferma(self) -> None:
        """The optimiser step was taken: commit the running estimate of this step."""
        if self._prossimo is not None:
            self.S, self.T = self._prossimo
            self.n += 1
        self._prossimo = None

    def scarta(self) -> None:
        """Skipped step: the running estimate is not updated."""
        self._prossimo = None

    def stato(self) -> dict:
        return {"S": None if self.S is None else self.S.detach().cpu().clone(),
                "T": None if self.T is None else self.T.detach().cpu().clone(), "n": self.n, "beta": self.beta}

    def carica(self, d: dict, dev) -> None:
        self.S = None if d.get("S") is None else d["S"].to(dev)
        self.T = None if d.get("T") is None else d["T"].to(dev)
        self.n = int(d.get("n", 0))
        self._prossimo = None


class Accumulo:
    """Dataset-level letter marginals over many packs (held-out diagnostic): sums of student and teacher distributions."""

    def __init__(self) -> None:
        self.sS = None
        self.sT = None
        self.n = 0

    def aggiungi(self, lS: torch.Tensor, pT: torch.Tensor) -> None:
        pS = lS.detach().double().exp().sum(0)
        pT_ = pT.detach().double().sum(0)
        self.sS = pS if self.sS is None else self.sS + pS
        self.sT = pT_ if self.sT is None else self.sT + pT_
        self.n += int(lS.shape[0])

    def risultato(self) -> dict:
        if not self.n:
            return {}
        mS, mT = self.sS / self.n, self.sT / self.n
        return {"prior": float(kl_marginale(mT, mS)), "pS": [round(float(x), 5) for x in mS.cpu()],
                "pT": [round(float(x), 5) for x in mT.cpu()], "n": self.n}

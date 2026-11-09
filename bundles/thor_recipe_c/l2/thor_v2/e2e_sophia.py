#!/usr/bin/env python3
"""e2e_sophia — Newton diagonale stile Sophia sulle scale theta (27/9/2026, ramo newton-theta; registro
TECNICHE_E2E_PV.md §19 e §36). File NUOVO: `e2e_allena` lo importa solo con `--ottimizzatore-theta sophia`.

Sophia (Liu, Li, Hall, Liang, Ma 2023, arXiv 2305.14342, Alg. 3), sul SOLO gruppo theta:
    m   <- b1 m + (1 - b1) g                       (g = theta.grad, unita' K come in AdamW: m e' lo stesso exp_avg)
    h   <- b2 h + (1 - b2) h_misurata              (quando la curvatura e' misurata: `aggiorna_curvatura`)
    theta <- theta - lr * clip(m / max(gamma h^, eps), -1, 1),   h^ = h / (1 - b2^n)   (n = misure di h)
Scostamenti DICHIARATI dall'articolo: (1) h^ corretto come Adam (l'articolo parte da h = 0 senza correzione: con
h = 0.01 h_misurata al primo passo il clip scatterebbe ovunque per ~100 passi, e noi ne facciamo ~128);
(2) weight decay 0 (le scale non vanno tirate verso theta = 0 = la forgia); (3) la curvatura non e' lo stimatore
GNB (B g^2, un campione) ma la Gauss-Newton PER TOKEN del passo V, gia' pagata:
    h_misurata = (N / n_c) d^2 HT,  HT = sum_{micro di curvatura} sum_s dY_{s,r}^2 u_{s,r,g}^2,  u = t_{r,g} . X_{s,g}
(e2e_pv_codici._k_ht / ht_torch; dY dall'indietro di sqrt(K / token) (p - e_y), y ~ p dello studente: unita' K come
m). Misurato sugli strati 0/3 veri (strumenti/diagnosi_newton_theta.py strato): E_y[(sum_s dY u)^2] / E_y[HT] = 1,00
(i termini fra token non contano: HT e' la Gauss-Newton piena).
Senza misura (n = 0) il denominatore e' eps: passo = lr * segno(m), cioe' il primo passo di Adam (il cancello del
primo passo di G1 resta valido).

Il resto (F32, LoRA) resta AdamW: `OttimizzatoreSophia` compone SophiaTheta (gruppo 0) + torch AdamW (gruppi 1..)
con la stessa interfaccia che `e2e_allena` usa (param_groups, state, step, zero_grad, state_dict, load_state_dict).
"""
from __future__ import annotations

import math
from collections import ChainMap
from typing import Callable, Optional

import torch


class SophiaTheta(torch.optim.Optimizer):
    """Sophia sul gruppo theta. Stato per parametro: step, exp_avg (m), hess (h, EMA senza correzione), n_hess (int,
    misure di h). `ultimo_conteggio`: (non clippati, vivi) del passo, tensori sul dispositivo (nessuna sincronizzazione:
    li legge chi scrive il registro dopo la sincronizzazione del passo). Vivi = m != 0."""

    def __init__(self, params, lr: float, beta1: float, gamma: float, beta_h: float, eps: float,
                 ombra_adam: Optional[tuple] = None):
        if not 0.0 <= beta1 < 1.0 or not 0.0 < beta_h < 1.0 or not gamma > 0 or not eps > 0:
            raise ValueError(f"Sophia: beta1 {beta1}, beta_h {beta_h}, gamma {gamma}, eps {eps}")
        super().__init__(params, dict(lr=lr, betas=(beta1, beta_h), gamma=gamma, eps=eps, weight_decay=0.0))
        self.ultimo_conteggio: Optional[tuple] = None
        # 27/9 integrata (attribuzione): OMBRA di AdamW = (beta2, eps) di Adam. Si tiene anche v = EMA di g^2
        # ("exp_avg_sq", con il suo contatore "step_ombra") per sapere quale passo avrebbe fatto AdamW dallo STESSO stato
        # (`theta_adam`). Non entra nel passo di Sophia: None = stato e passo identici a prima.
        self.ombra = None if ombra_adam is None else (float(ombra_adam[0]), float(ombra_adam[1]))

    def _stato(self, p) -> dict:
        s = self.state[p]
        if not s:
            s["step"] = torch.zeros((), dtype=torch.float32)
            s["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            s["hess"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            s["n_hess"] = 0
        return s

    @torch.no_grad()
    def aggiorna_curvatura(self, p, h: torch.Tensor) -> None:
        """h <- b2 h + (1 - b2) h_misurata (h_misurata >= 0, forma di p, unita' di m)."""
        gr = next(g for g in self.param_groups if any(q is p for q in g["params"]))
        bh = gr["betas"][1]
        s = self._stato(p)
        s["hess"].mul_(bh).add_(h.reshape(p.shape).to(s["hess"].dtype), alpha=1.0 - bh)
        s["n_hess"] = int(s["n_hess"]) + 1

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        if closure is not None:
            raise ValueError("SophiaTheta: niente closure")
        nc = vivi = None
        for gr in self.param_groups:
            if gr["weight_decay"] != 0.0:
                raise ValueError("SophiaTheta: weight_decay deve restare 0")
            lr, (b1, bh), gam, eps = gr["lr"], gr["betas"], gr["gamma"], gr["eps"]
            for p in gr["params"]:
                if p.grad is None:
                    continue
                s = self._stato(p)
                s["step"] += 1
                m = s["exp_avg"]
                m.mul_(b1).add_(p.grad, alpha=1.0 - b1)
                if self.ombra is not None:                       # integrata: v di AdamW, solo registro
                    if "exp_avg_sq" not in s:
                        s["exp_avg_sq"] = torch.zeros_like(m)
                        s["step_ombra"] = torch.zeros((), dtype=torch.float32)
                    s["step_ombra"] += 1
                    s["exp_avg_sq"].mul_(self.ombra[0]).addcmul_(p.grad, p.grad, value=1.0 - self.ombra[0])
                n = int(s["n_hess"])
                if n > 0:
                    den = (s["hess"] * (gam / (1.0 - bh ** n))).clamp_min_(eps)
                else:
                    den = torch.full_like(m, eps)
                r = torch.div(m, den).clamp_(-1.0, 1.0)
                p.add_(r, alpha=-lr)
                viv = m != 0
                c1 = (viv & (r.abs() < 1.0)).sum()
                c2 = viv.sum()
                nc = c1 if nc is None else nc + c1
                vivi = c2 if vivi is None else vivi + c2
        self.ultimo_conteggio = (nc, vivi)
        return None


    @torch.no_grad()
    def theta_adam(self, p, theta0: torch.Tensor) -> Optional[torch.Tensor]:
        """27/9 integrata: theta che AdamW (torch, weight decay 0) avrebbe dato da theta0 con lo STESSO m (exp_avg,
        beta1) e la v dell'ombra: theta0 - lr / (1 - b1^t) * m / (sqrt(v) / sqrt(1 - b2^t_v) + eps_adam). None senza
        ombra o senza v."""
        s = self.state.get(p)
        if self.ombra is None or not s or "exp_avg_sq" not in s:
            return None
        gr = next(g for g in self.param_groups if any(q is p for q in g["params"]))
        b1 = gr["betas"][0]
        b2, eps = self.ombra
        t, tv = float(s["step"]), float(s["step_ombra"])
        den = (s["exp_avg_sq"].sqrt() / math.sqrt(1.0 - b2 ** tv)).add_(eps)
        return theta0 - (gr["lr"] / (1.0 - b1 ** t)) * s["exp_avg"] / den


class OttimizzatoreSophia:
    """SophiaTheta (theta, gruppo 0) + torch AdamW (F32, LoRA: gruppi 1..; None se non ce ne sono). `param_groups` e'
    la lista dei gruppi di entrambi (gli STESSI dizionari: cambiare lr/betas alla ripresa arriva agli ottimizzatori);
    `state` e' la vista unita dei due stati."""

    tipo = "sophia"

    def __init__(self, sophia: SophiaTheta, adam: Optional[torch.optim.Optimizer]):
        self.sophia = sophia
        self.adam = adam

    @property
    def param_groups(self) -> list:
        return self.sophia.param_groups + (self.adam.param_groups if self.adam is not None else [])

    @property
    def state(self):
        return ChainMap(self.sophia.state, self.adam.state) if self.adam is not None else self.sophia.state

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.sophia.zero_grad(set_to_none=set_to_none)
        if self.adam is not None:
            self.adam.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self.sophia.step()
        if self.adam is not None:
            self.adam.step()

    def state_dict(self) -> dict:
        return {"tipo": self.tipo, "sophia": self.sophia.state_dict(),
                "adam": self.adam.state_dict() if self.adam is not None else None}

    def load_state_dict(self, sd: dict) -> None:
        if sd.get("tipo") != self.tipo:
            raise ValueError(f"stato di ottimizzatore {sd.get('tipo')!r}, atteso {self.tipo!r}")
        self.sophia.load_state_dict(sd["sophia"])
        if (sd["adam"] is None) != (self.adam is None):
            raise ValueError("stato Sophia salvato con/senza gruppi AdamW diversi da ora: usare la ripresa a famiglie")
        if self.adam is not None:
            self.adam.load_state_dict(sd["adam"])


def e_sophia(sd) -> bool:
    return isinstance(sd, dict) and sd.get("tipo") == OttimizzatoreSophia.tipo


def stato_sophia_da_adam(sd_adam: dict, n_theta: int, sophia: SophiaTheta) -> dict:
    """Stato di SophiaTheta dai primi `n_theta` parametri di uno stato di AdamW (passaggio adamw -> sophia alla
    ripresa): m = exp_avg (stessa EMA con lo stesso beta1), step; h vuota (n_hess 0). Iperparametri: quelli di ORA."""
    ora = sophia.state_dict()
    stato = {}
    for i in range(n_theta):
        s = sd_adam["state"].get(i)
        if s is None:
            continue
        stato[i] = {"step": s["step"].clone().to(torch.float32), "exp_avg": s["exp_avg"].clone(),
                    "hess": torch.zeros_like(s["exp_avg"]), "n_hess": 0}
        if sophia.ombra is not None and "exp_avg_sq" in s:   # integrata: l'ombra di AdamW parte dalla v di Adam
            stato[i]["exp_avg_sq"] = s["exp_avg_sq"].clone()
            stato[i]["step_ombra"] = s["step"].clone().to(torch.float32)
    return {"state": stato, "param_groups": ora["param_groups"]}


def parte_adam(sd_adam: dict, n_theta: int) -> Optional[dict]:
    """Stato di AdamW senza i primi n_theta parametri e senza il gruppo 0 (theta): gli indici ripartono da 0. None se
    lo stato salvato non ha altri gruppi."""
    if len(sd_adam["param_groups"]) <= 1:
        return None
    gruppi = []
    for g in sd_adam["param_groups"][1:]:
        g2 = dict(g)
        g2["params"] = [i - n_theta for i in g["params"]]
        gruppi.append(g2)
    return {"state": {i - n_theta: v for i, v in sd_adam["state"].items() if i >= n_theta}, "param_groups": gruppi}


def adam_da_parte(sd_parte: Optional[dict], n_theta: int, gruppo_theta: dict, sd_sophia: Optional[dict] = None) -> dict:
    """Inverso di `parte_adam` (passaggio sophia -> adamw): stato di AdamW coi gruppi salvati spostati di n_theta;
    `gruppo_theta` = il gruppo 0 dell'AdamW di ORA. Theta: se lo stato di Sophia porta l'OMBRA di AdamW (exp_avg_sq +
    step_ombra, integrata), Adam RIPRENDE da m = exp_avg e v = ombra (27/9 22:0x: EB3 passo 19, Sophia peggio di AdamW
    nell'attribuzione; ripartire da zero rifarebbe un passo ~lr*segno su tutti i gruppi); altrimenti theta VUOTO."""
    gruppi = [dict(gruppo_theta)]
    stato = {}
    if sd_sophia is not None:
        for i in range(n_theta):
            s = sd_sophia["state"].get(i)
            if s and "exp_avg_sq" in s and "step_ombra" in s:
                stato[i] = {"step": s["step_ombra"].clone().to(torch.float32), "exp_avg": s["exp_avg"].clone(),
                            "exp_avg_sq": s["exp_avg_sq"].clone()}
    if sd_parte is not None:
        for g in sd_parte["param_groups"]:
            g2 = dict(g)
            g2["params"] = [i + n_theta for i in g["params"]]
            gruppi.append(g2)
        stato.update({i + n_theta: v for i, v in sd_parte["state"].items()})   # 27/9 22:4x: update, non riassegnare (cancellava theta)
    return {"state": stato, "param_groups": gruppi}

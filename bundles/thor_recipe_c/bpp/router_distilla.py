#!/usr/bin/env python3
"""router_distilla.py — 26/10 arm B++: lever L1 "R-CAMMINO" of ~/tony-forgia/ricerche/SINTESI_1025/R7_instradamento.md
(--router-cammino W of cardan_congiunto_scale.py) and the "prova 0" oracle (--forza-scelte-maestro).

Why (R7 §0): at layer 12 the T400 router agrees 97.4 % with the teacher router ON THE SAME INPUT; the 29 % of top-8
choices it loses come from INPUT DRIFT. A router-only correction is capped at +0.02 overlap; the fix needs a router-side
target whose gradient flows UPSTREAM through the student's own residual stream, on every layer (the path). Full-softmax
KL puts ~60 % of its weight on ranks 9-256, irrelevant to the choice; the 8th/9th teacher margin is tiny (median 0.043
logit), so the exact set is not a target: the gate MASS is.

Term (off by default: W = 0 installs nothing, the trainer is the B+ trainer at the bit), on the first n tokens of the paper
row, l over the router layers (default: every cached layer the student trains, i.e. all 40), lambda_l (sum 1):
    L_R = W * sum_l lambda_l * (1/n) sum_t [ KL_U(pT_lt || pS_lt) + beta * M_lt ]                       (R7 §5 L1)
  KL_U   KL on the decision zone U = teacher top-K_T ∪ student top-K_S (default 16 ∪ 8) plus ONE "rest" bucket:
         pbar = (p_U, 1 - sum_U p) for both, KL_U = sum_{U ∪ rest} pbarT log(pbarT / pbarS)          (R7 §4.5)
  M      soft mass at risk on the teacher ranks [a, b] (default 4-9, the ranks that flip: R7 §0.2):
         M = sum_{e in ranks a..b of the teacher} wT_e * s_e,  wT_e = pT_e / sum_{teacher top-8} pT  (= the teacher gate
         pi^T for ranks <= 8, same normaliser beyond), s_e = 1 - sigmoid((zS_e - thS)/tau) for teacher ranks <= 8 (must be
         chosen: R7's M_l exactly when [a, b] = [1, 8]) and s_e = sigmoid((zS_e - thS)/tau) for ranks > 8 (must stay out:
         the 9th is the first intruder of the boundary), thS = sg((zS_(8) + zS_(9)) / 2) the student's own boundary,
         tau = --router-cammino-tau, or 0.25 * rms(zS - zT) of the layer (detached) when 0 (R7 §4.5)
  pS from the student's router on the student's OWN propagated state (forward hook of every MoEStudente, the very
  expression of MoEStudente.forward), so the gradient reaches router_w (the F32 master of ffn_gate_inp under
  --allena-f32) and, through x, every trained block upstream (R7 §4.3). --router-cammino-perdita kl_piena = full-softmax
  KL instead (kept for comparison; R7: nearly useless).
Teacher targets: router logits of the q8 teacher (llama.cpp "ffn_moe_logits-<l>", pre-softmax, F32 GEMM) cached by
  cattura_router_maestro.py in fp16 (default) or fp32 — NEVER bf16 (R7 §1.0: bf16 rounding alone flips the top-8 set on
  11 % of tokens): a bf16 cache is refused. 40 x 256 x 2 B = 20 KiB/token.
Injection: the hook returns Inietta(out, k), k = W lambda_l term_l (0-dim, with its graph): same forward value; when the
  backward reaches the layer output it also backpropagates k with gradient 1. No extra forward/backward; works in the
  checkpointed recompute (hook fires when the layer is recomputed with grad) and in a held graph. One injection per layer
  per backward (counted).
Diagnostics per layer (no gradient): ov8 = |S^T ∩ S^S| / 8, massa = mu = sum_{S^T ∩ S^S} pi^T (R7 §4.1).

"Prova 0" (R7 §5): ForzaScelte imposes the teacher top-8 at every MoE layer through MoEStudente.instr (the trainer's own
  replay hook: gates = the student's softmax renormalised over S^T, "Rollout Routing Replay"), on the tokens covered by the
  cache, the student's own choices elsewhere; the trainer measures KL24 / paper held-out free and forced.
Prove CPU: test_bpp/test_bpp.py R* and prova_reale_bpp.py RR1.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as Fnn

VERSIONE = "router_distilla 2 R-CAMMINO (26/10/2026)"
NOME_TENSORE = "ffn_moe_logits-{}"
_NEG = -1e30                                               # finite "minus infinity" (masked logsumexp, no NaN gradients)


def _lettore_dump():
    try:
        import e2e_dump_logit_leggi as M
    except ImportError:
        import sys
        sys.path.append(str(Path(__file__).resolve().parent / "recupero_7dabd6f" / "thor_v2"))
        import e2e_dump_logit_leggi as M
    return M


def _a_float32(a: np.ndarray) -> torch.Tensor:
    a = np.asarray(a)
    if a.dtype == np.uint16:                              # bf16 bits (only for reading: training refuses bf16 caches)
        return torch.from_numpy(np.ascontiguousarray(a).view(np.int16)).view(torch.bfloat16).float()
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


class SorgenteRouter:
    """Reader of a cattura_router_maestro.py cache (layout of cattura_stati_maestro.py, piano_router.json):
        <lato>/blocco_NNNN/blocco.json  SEAL; ids.i32 [rows, n_token]; dump/seq_KKKK/ffn_moe_logits-<l>.f16 [n_token, E]
    `logit(lato, i, strati, device)` -> (float32 [len(strati), n_token, E], ids int64 numpy)."""

    def __init__(self, cartella, bf16_ammesso: bool = False):
        Uscita = _lettore_dump().Uscita
        self.dir = Path(cartella)
        self.piano = json.loads((self.dir / "piano_router.json").read_text())
        if self.piano.get("salva_tipo") == "bf16" and not bf16_ammesso:
            raise ValueError(f"{cartella}: router logits in bf16 (R7 §1.0: 11 % of top-8 sets flip by rounding alone); "
                             f"recapture with --salva-tipo f16 or f32")
        self.strati = [int(x) for x in self.piano["strati"]]
        self.n_token = int(self.piano["n_token"])
        self.n_expert = int(self.piano["n_expert"])
        self._idx, self._usc = {}, {}
        for b in self.piano["blocchi"]:
            d = self.dir / b["nome"]
            if not (d / "blocco.json").exists():
                continue
            s = json.loads((d / "blocco.json").read_text())
            ids = np.fromfile(d / "ids.i32", dtype=np.int32).reshape(len(s["righe"]), self.n_token)
            if hashlib.sha256(ids.tobytes()).hexdigest() != s["sha256_ids"]:
                raise RuntimeError(f"{d}: ids.i32 does not match its seal")
            self._usc[b["nome"]] = Uscita(d / "dump")
            for k, i in enumerate(s["righe"]):
                self._idx[(b["lato"], int(i))] = (b["nome"], k, ids[k])

    def disponibile(self, lato: str, i: int) -> bool:
        return (lato, int(i)) in self._idx

    def righe(self, lato: str) -> list:
        return sorted(i for (lt, i) in self._idx if lt == lato)

    def logit(self, lato: str, i: int, strati, device) -> tuple:
        nome, k, ids = self._idx[(lato, int(i))]
        u = self._usc[nome]
        out = []
        for L in strati:
            if L not in self.strati:
                raise KeyError(f"layer {L} not in the router cache (layers {self.strati})")
            a = u.tensore(k, NOME_TENSORE.format(L))
            if tuple(a.shape) != (self.n_token, self.n_expert):
                raise ValueError(f"{nome} seq {k} {NOME_TENSORE.format(L)}: shape {tuple(a.shape)}")
            out.append(_a_float32(a))
        return torch.stack(out).to(device), ids.astype(np.int64)


class _Inietta(torch.autograd.Function):
    """out unchanged in the forward; in the backward the gradient of out passes through and the 0-dim k (the weighted
    term of the layer, with its own graph) receives gradient 1: its backward reaches router_w and the layer input."""

    @staticmethod
    def forward(ctx, out, k, contatore):
        ctx.contatore = contatore
        ctx.kd = (k.dtype, k.device)
        return out.clone()

    @staticmethod
    def backward(ctx, g):
        ctx.contatore[0] += 1
        return g, torch.ones((), dtype=ctx.kd[0], device=ctx.kd[1]), None


def router_logit(mod, x: torch.Tensor) -> torch.Tensor:
    """= the router expression of e2e_studente.MoEStudente.forward on x [N, H]."""
    if getattr(mod, "router_w32", None) is not None and x.dtype not in (torch.float32, torch.float64):
        return Fnn.linear(x.float(), mod.router_w32)
    return Fnn.linear(x, mod.router_w)


def _acc(z: torch.Tensor):
    return torch.float64 if z.dtype == torch.float64 else torch.float32


def kl_piena(z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
    """mean over the tokens of KL(softmax(z_t) || softmax(z_s)), all experts."""
    acc = _acc(z_s)
    ls = torch.log_softmax(z_s.to(acc), -1)
    lt = torch.log_softmax(z_t.to(device=ls.device, dtype=acc), -1)
    return (lt.exp() * (lt - ls)).sum(-1).mean()


def zona(z_s: torch.Tensor, z_t: torch.Tensor, k_t: int, k_s: int) -> torch.Tensor:
    """bool [n, E]: teacher top-k_t ∪ student top-k_s (no gradient)."""
    with torch.no_grad():
        U = torch.zeros(z_t.shape, dtype=torch.bool, device=z_t.device)
        U.scatter_(-1, torch.topk(z_t, min(k_t, z_t.shape[-1]), -1).indices, True)
        U.scatter_(-1, torch.topk(z_s.detach(), min(k_s, z_s.shape[-1]), -1).indices, True)
    return U


def kl_zona(z_s: torch.Tensor, z_t: torch.Tensor, k_t: int = 16, k_s: int = 8) -> torch.Tensor:
    """mean over the tokens of KL_U(pbarT || pbarS), U = zona(...), pbar = (p_U, 1 - sum_U p) (R7 §4.5)."""
    acc = _acc(z_s)
    ls = torch.log_softmax(z_s.to(acc), -1)
    lt = torch.log_softmax(z_t.to(device=ls.device, dtype=acc), -1)
    U = zona(z_s, z_t, k_t, k_s)
    dentro = (lt.exp() * (lt - ls)).masked_fill(~U, 0.0).sum(-1)
    lt_r = torch.logsumexp(lt.masked_fill(U, _NEG), -1)          # log of the teacher's "rest" mass
    ls_r = torch.logsumexp(ls.masked_fill(U, _NEG), -1)
    resto = lt_r.exp() * (lt_r - ls_r)                           # 0 (exactly) when U covers every expert
    return (dentro + resto).mean()


def tau_adattiva(z_s: torch.Tensor, z_t: torch.Tensor) -> float:
    """R7 §4.5: tau = 0.25 * rms(zS - zT) of the layer (detached)."""
    with torch.no_grad():
        return max(0.25 * float((z_s.detach().to(_acc(z_s)) - z_t.to(z_s.device, _acc(z_s))).pow(2).mean().sqrt()), 1e-6)


def massa_a_rischio(z_s: torch.Tensor, z_t: torch.Tensor, ranghi=(4, 9), tau: float = 0.1, k: int = 8) -> torch.Tensor:
    """mean over the tokens of M = sum_{teacher ranks a..b} wT_e s_e (module docstring)."""
    a, b = int(ranghi[0]), int(ranghi[1])
    E = z_t.shape[-1]
    b = min(b, E)
    acc = _acc(z_s)
    zs = z_s.to(acc)
    with torch.no_grad():
        zt = z_t.to(device=zs.device, dtype=acc)
        ordT = torch.argsort(zt, -1, descending=True)
        pT = torch.softmax(zt, -1)
        kk = min(k, E)
        norm_ = pT.gather(-1, ordT[:, :kk]).sum(-1, keepdim=True)
        sel = ordT[:, a - 1:b]
        w = pT.gather(-1, sel) / norm_
        dentro = (torch.arange(a, b + 1, device=zs.device) <= kk).view(1, -1)
        srt = torch.sort(zs.detach(), -1, descending=True).values
        th = 0.5 * (srt[:, kk - 1] + srt[:, kk]) if kk < E else srt[:, kk - 1] - 1.0
    s = torch.sigmoid((zs.gather(-1, sel) - th.view(-1, 1)) / tau)
    rischio = torch.where(dentro, 1.0 - s, s)
    return (w * rischio).sum(-1).mean()


@torch.no_grad()
def diagnostica(z_s: torch.Tensor, z_t: torch.Tensor, k: int = 8) -> tuple:
    """(ov8, massa) means over the tokens: |S^T ∩ S^S|/k and sum_{S^T ∩ S^S} pi^T (R7 §4.1)."""
    k = min(k, z_t.shape[-1])
    zt = z_t.to(z_s.device, torch.float32)
    St = torch.topk(zt, k, -1).indices
    Ss = torch.topk(z_s.detach().float(), k, -1).indices
    inS = torch.zeros(zt.shape, dtype=torch.bool, device=zt.device).scatter_(-1, Ss, True)
    comune = inS.gather(-1, St)
    piT = torch.softmax(zt, -1).gather(-1, St)
    piT = piT / piT.sum(-1, keepdim=True)
    return float(comune.float().mean()), float((piT * comune).sum(-1).mean())


class DistillaRouter:
    """Forward hooks on st.moduli[L].mlp for L in `strati`. Contexts:
       su(lato, i, ids)        training: hooks active during the forward+backward inside; injection only where grad is on;
                               on exit (no exception) checks that every layer injected exactly once (colpi)
       misura(lato, i, ids)    held-out: values only (no grad, no injection)
    `ultimo` = sum_l lambda_l term_l of the last measure (the unweighted-by-W loss), `ultimo_diag` = {ov8, massa, kl, m}."""

    def __init__(self, st, src: SorgenteRouter, strati, peso: float, n_token: int = 0, perdita: str = "cammino",
                 zona_k=(16, 8), beta: float = 1.0, ranghi=(4, 9), tau: float = 0.0, lambdas: Optional[dict] = None):
        if perdita not in ("cammino", "kl_piena"):
            raise ValueError(f"perdita {perdita!r}: cammino | kl_piena")
        self.st, self.src, self.strati, self.peso = st, src, list(strati), float(peso)
        self.n_max = int(n_token) if n_token else src.n_token
        self.perdita, self.zona_k, self.beta, self.ranghi, self.tau = perdita, tuple(zona_k), float(beta), tuple(ranghi), tau
        lam = {L: 1.0 for L in self.strati} if not lambdas else {L: float(lambdas.get(L, 0.0)) for L in self.strati}
        tot = sum(lam.values())
        if not tot > 0:
            raise ValueError("router: lambda_l all zero on the router layers")
        self.lam = {L: v / tot for L, v in lam.items()}
        self.bersagli: dict = {}
        self.n = 0
        self.attivo = False
        self.registra = False
        self.valori: dict = {}
        self.diag: dict = {}
        self.colpi = [0]
        self.ultimo: Optional[float] = None
        self.ultimo_diag: dict = {}
        self._ganci = [st.moduli[L].mlp.register_forward_hook(self._gancio_di(L), with_kwargs=True) for L in self.strati]

    def rimuovi(self) -> None:
        for h in self._ganci:
            h.remove()
        self._ganci = []

    def termine(self, z_s: torch.Tensor, z_t: torch.Tensor, k: int = 8) -> tuple:
        """(term of one layer with its graph, {parts as floats}); k = experts per token of the layer (top_k)."""
        if self.perdita == "kl_piena":
            kl = kl_piena(z_s, z_t)
            return kl, {"kl": float(kl.detach())}
        kl = kl_zona(z_s, z_t, *self.zona_k)
        tau = self.tau if self.tau > 0 else tau_adattiva(z_s, z_t)
        m = massa_a_rischio(z_s, z_t, self.ranghi, tau, k)
        return kl + self.beta * m, {"kl": float(kl.detach()), "m": float(m.detach()), "tau": tau}

    def _gancio_di(self, L: int):
        def gancio(mod, args, kwargs, out):
            if L not in self.bersagli or not (self.attivo or self.registra):
                return None
            if not torch.is_grad_enabled() and not self.registra:
                return None                               # checkpointed forward without grad: the recompute does it
            x = args[0] if args else kwargs["hidden_states"]
            xf = x.reshape(-1, x.shape[-1])[: self.n]
            z_s = router_logit(mod, xf)
            z_t = self.bersagli[L]
            k = int(getattr(mod, "top_k", 8))
            t, parti = self.termine(z_s, z_t, k)
            self.valori[L] = float(t.detach())
            ov, ma = diagnostica(z_s, z_t, k)
            self.diag[L] = {**parti, "ov8": ov, "massa": ma}
            if self.attivo and torch.is_grad_enabled() and t.requires_grad and out.requires_grad:
                return _Inietta.apply(out, t * (self.peso * self.lam[L]), self.colpi)
            return None
        return gancio

    def _prepara(self, lato: str, i: int, ids_riga: torch.Tensor) -> None:
        z, ids_c = self.src.logit(lato, i, self.strati, ids_riga.device)
        n = min(z.shape[1], int(ids_riga.shape[-1]), self.n_max)
        if not np.array_equal(ids_c[:n], ids_riga.reshape(-1)[:n].cpu().numpy()):
            raise RuntimeError(f"router: ids of the router cache differ from the paper row {lato} {i}")
        self.bersagli = {L: z[k, :n].float() for k, L in enumerate(self.strati)}
        self.n, self.valori, self.diag, self.colpi[0] = n, {}, {}, 0

    def _chiudi(self) -> None:
        if self.valori:
            self.ultimo = sum(self.lam[L] * v for L, v in self.valori.items()) / sum(self.lam[L] for L in self.valori)
            chiavi = sorted({k for d in self.diag.values() for k in d})
            self.ultimo_diag = {k: sum(d[k] for d in self.diag.values() if k in d) / len(self.diag) for k in chiavi}
            self.ultimo_diag["strati"] = {L: dict(d) for L, d in sorted(self.diag.items())}
        else:
            self.ultimo, self.ultimo_diag = None, {}
        self.bersagli, self.attivo, self.registra = {}, False, False

    def su(self, lato: str, i: int, ids_riga: torch.Tensor):
        rd = self

        class _Ctx:
            def __enter__(self_):
                rd._prepara(lato, i, ids_riga)
                rd.attivo = True
                return rd

            def __exit__(self_, et, ev, tb):
                atteso = len(rd.bersagli)
                rd._chiudi()
                if et is None and rd.colpi[0] != atteso:
                    raise RuntimeError(f"router: {rd.colpi[0]} iniezioni nell'indietro, attese {atteso}")
                return False
        return _Ctx()

    def misura(self, lato: str, i: int, ids_riga: torch.Tensor):
        rd = self

        class _Ctx:
            def __enter__(self_):
                rd._prepara(lato, i, ids_riga)
                rd.registra = True
                return rd

            def __exit__(self_, et, ev, tb):
                rd._chiudi()
                return False
        return _Ctx()


class ForzaScelte:
    """Prova 0 (R7 §5): forward PRE-hooks on st.moduli[L].mlp that set MoEStudente.instr = the teacher top-k on the first n
    tokens (the student's own top-k beyond n), for the row given to su(); the student's gates are then its own softmax
    renormalised over the forced set (MoEStudente.forward with instr). Diagnostic: fraction of forced token-slots that
    differ from the student's own choice (ov8 of the free student on the same state)."""

    def __init__(self, st, sorgenti: dict, strati):
        self.st, self.sorgenti, self.strati = st, dict(sorgenti), list(strati)
        self.idx: dict = {}
        self.n = 0
        self.attivo = False
        self.ov: list = []
        self._ganci = [st.moduli[L].mlp.register_forward_pre_hook(self._gancio_di(L), with_kwargs=True)
                       for L in self.strati]

    def rimuovi(self) -> None:
        for h in self._ganci:
            h.remove()
        self._ganci = []

    def _gancio_di(self, L: int):
        def gancio(mod, args, kwargs):
            if not self.attivo or L not in self.idx:
                return None
            x = args[0] if args else kwargs["hidden_states"]
            with torch.no_grad():
                z = router_logit(mod, x.reshape(-1, x.shape[-1]))
                k = int(mod.top_k)
                prop = torch.topk(torch.softmax(z.float(), -1), k, -1).indices
                idx = prop.clone()
                n = min(self.n, idx.shape[0])
                idx[:n] = self.idx[L][:n].to(idx.device)
                inS = torch.zeros(z.shape, dtype=torch.bool, device=z.device).scatter_(-1, prop[:n], True)[:n]
                self.ov.append(float(inS.gather(-1, idx[:n]).float().mean()))
            mod.instr = idx
            return None
        return gancio

    def su(self, chiave: str, lato: str, i: int, ids_riga: torch.Tensor):
        fz = self
        src = self.sorgenti[chiave]

        class _Ctx:
            def __enter__(self_):
                z, ids_c = src.logit(lato, i, fz.strati, ids_riga.device)
                n = min(z.shape[1], int(ids_riga.shape[-1]))
                if not np.array_equal(ids_c[:n], ids_riga.reshape(-1)[:n].cpu().numpy()):
                    raise RuntimeError(f"forza: ids of the router cache differ from row {lato} {i}")
                fz.idx = {L: torch.topk(z[j, :n], int(fz.st.moduli[L].mlp.top_k), -1).indices
                          for j, L in enumerate(fz.strati)}
                fz.n, fz.attivo = n, True
                return fz

            def __exit__(self_, et, ev, tb):
                fz.attivo, fz.idx = False, {}
                return False
        return _Ctx()


def leggi_lambda(p) -> dict:
    """--router-cammino-lambda FILE: JSON {layer: weight} (e.g. the mass lost per layer, R4 a2); normalised later."""
    d = json.loads(Path(p).read_text())
    d = d.get("lambda", d)
    out = {int(k): float(v) for k, v in d.items()}
    if any(v < 0 or not math.isfinite(v) for v in out.values()):
        raise ValueError(f"{p}: lambda must be finite and >= 0")
    return out

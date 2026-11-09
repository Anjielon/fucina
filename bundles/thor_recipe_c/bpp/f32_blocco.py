#!/usr/bin/env python3
"""f32_blocco.py — F32-E2E (SINTESI_1025, 26/10/2026; ~/tony-forgia/ricerche/SINTESI_1025/ANTAGONISTA_1026.md §2).

Train ONLY the F32 tensors of the GGUF (norms, routers ffn_gate_inp(_shexp), ssm_a / ssm_dt / ssm_conv1d, output_norm:
301 tensors, 22 213 248 values on C) with every trit and every fp16 scale frozen. Trained == deployed: the F32 bytes of
the GGUF are written back as they are (f32_esporta.py + patch_gguf.py), no fold, no rounding.

Nothing of the student is rewritten: the masters are e2e_studente.TensoreF32 (`Studente(allena_f32=...)`, the 27/9
--allena-f32 branch of e2e_allena, proved by e2e_test_f32 A23-A28: value recomputed from the master and checked AT THE BIT
against the loader at construction, installed before every layer call, gradient through the recompute of the checkpointed
backward). This module only adds the optimiser side:

  classes (lr, parametrisation; CLI of cardan_congiunto_scale.py)
    norme   attn_norm, post_attention_norm, attn_q_norm, attn_k_norm, ssm_norm, output_norm   --lr-f32-norme  1e-4  log
    router  ffn_gate_inp, ffn_gate_inp_shexp                                                    --lr-f32-router 3e-5  linear
    ssm_a   ssm_a  (file value = -exp(A_log): log|w| IS A_log)                                  --lr-f32-ssm-a  1e-4  log
    ssm     ssm_dt (bias), ssm_conv1d                                                            --lr-f32-ssm    3e-5  linear
  log parametrisation (--f32-log, default "norme,ssm_a"): w = s * exp(phi), s = sign(w0) fixed, phi = log|w0| fp32 (dtype
    of the master), Adam on phi; dL/dphi = dL/dw * w (chain rule, exact at the current point); after the step
    w = s * exp(phi) ONLY where phi moved (elements whose phi did not move keep the master at the bit: an untouched element
    is never re-rounded through exp(log(.))). A zero value in a log class is a configuration error (none in C).
  blocks (each its own Adam, betas (0.9, 0.999), weight_decay 0, clipped on its own norm like every block of the trainer):
    f32_norme [norme] · f32_router [router] · f32_ssm [ssm_a, ssm] (two param groups) — only the classes present.

Frozen check (--solo-f32): every theta of the student (expert and dense-ternary log-scales, st.parametri()) has no gradient
and is exactly 0 (theta = 0 is the host GGUF at the bit: scale = fp16(d0 exp(0)) = d0).
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

CLASSI = {
    "norme": ("attn_norm", "post_attention_norm", "attn_q_norm", "attn_k_norm", "ssm_norm", "output_norm"),
    "router": ("ffn_gate_inp", "ffn_gate_inp_shexp"),
    "ssm_a": ("ssm_a",),
    "ssm": ("ssm_dt", "ssm_conv1d"),
}
BLOCCHI = {"f32_norme": ("norme",), "f32_router": ("router",), "f32_ssm": ("ssm_a", "ssm")}
# the 11 families that are F32 in the GGUF of C (ssm_alpha / ssm_beta are TQ1_0R_1024 there: not F32)
TUTTE_C = ("attn_norm", "post_attention_norm", "attn_q_norm", "attn_k_norm", "ssm_norm", "ffn_gate_inp",
           "ffn_gate_inp_shexp", "ssm_a", "ssm_dt", "ssm_conv1d", "output_norm")


def famiglie(voce: str) -> tuple:
    """"tutte" -> the 11 F32 families of C; otherwise the e2e_studente parser (canonical order, unknown -> ValueError)."""
    import e2e_studente as S
    if voce.strip() == "tutte":
        return S.famiglie_f32(",".join(TUTTE_C))
    return S.famiglie_f32(voce)


def classe_di(fam: str) -> str:
    for c, fams in CLASSI.items():
        if fam in fams:
            return c
    raise ValueError(f"famiglia F32 {fam!r} senza classe in f32_blocco.CLASSI (ssm_alpha/ssm_beta non sono F32 in C)")


def log_classi(voce: str) -> tuple:
    out = tuple(c.strip() for c in voce.split(",") if c.strip())
    for c in out:
        if c not in CLASSI:
            raise ValueError(f"--f32-log: classe {c!r} sconosciuta ({sorted(CLASSI)})")
    if "router" in out or "ssm" in out:
        raise ValueError("--f32-log: router e ssm (dt, conv1d) hanno segni misti, solo parametrizzazione lineare")
    return out


def errore_config(a, fams: tuple) -> Optional[str]:
    """Configuration errors of the F32 switches (the trainer exits with code 5)."""
    if a.modo != "congiunto":
        return "--allena-f32 vuole --modo congiunto"
    # 26/10 BPP (copy BPP/codice): --fuso and --allinea-peso now ALLOWED with --allena-f32 (B++ = B+ recipe, which is fused
    # and aligned, + F32). Nothing to change in the trainer: the masters are installed at every layer call
    # (Studente._strato), in the fused [paper row | KL4 pack] forward and in the recompute of its backward exactly as in the
    # separate path, and the alignment hooks only add dL/dH terms. Proved on CPU by test_bpp/test_bpp.py S5 (F32 gradients
    # fused == separate, with and without the alignment, on a segment-aware synthetic student). --prova-fuso stays refused
    # (it compares the original blocks only).
    if a.prova_fuso:
        return "--allena-f32 non si combina con --prova-fuso"
    for k in ("lr_f32_norme", "lr_f32_router", "lr_f32_ssm_a", "lr_f32_ssm"):
        if not getattr(a, k) > 0:
            return f"--{k.replace('_', '-')} deve essere > 0"
    try:
        for f in fams:
            classe_di(f)
        log_classi(a.f32_log)
    except ValueError as e:
        return str(e)
    if a.solo_f32:
        altri = [n for n, v in (("--rango-esperti", a.rango_esperti), ("--rango-densi", a.rango_densi),
                                ("--scale-down-lr", a.scale_down_lr), ("--scale-gateup-lr", a.scale_gateup_lr),
                                ("--down-lowrank-rango", a.down_lowrank_rango)) if v != 0]
        if altri:
            return f"--solo-f32: devono essere 0 {', '.join(altri)} (solo i tensori F32 si allenano)"
    return None


class _Voce:
    """One trained F32 tensor: the Studente's TensoreF32 + its optimiser-side variable (phi in log space, or the master)."""

    def __init__(self, tf, classe: str, log: bool):
        self.tf, self.classe, self.log = tf, classe, log
        m = tf.master.detach()
        self.m0 = m.clone()
        if log:
            if bool((m == 0).any()):
                raise ValueError(f"{tf.nome}: valori zero nella classe log {classe} (|w| = exp(phi) impossibile)")
            self.segno = torch.sign(m)
            self.phi = torch.log(m.abs()).clone()
            self.phi.requires_grad_(True)              # leaf; never in a graph: its .grad is set by gradienti_log()
            self.phi_prima = None
        else:
            self.segno = self.phi = self.phi_prima = None

    @property
    def var(self) -> torch.Tensor:
        """The tensor Adam steps."""
        return self.phi if self.log else self.tf.master


class BloccoF32:
    def __init__(self, st, lr: dict, logc: tuple, betas=(0.9, 0.999)):
        if not getattr(st, "f32", None):
            raise ValueError("lo studente non ha tensori F32 allenati (Studente(allena_f32=...))")
        self.st = st
        self.lr = dict(lr)
        self.logc = tuple(logc)
        self.voci = [_Voce(tf, classe_di(tf.famiglia), classe_di(tf.famiglia) in self.logc) for tf in st.f32.values()]
        self._blocchi = {}
        for nb, classi in BLOCCHI.items():
            gruppi, lrs = [], []
            for c in classi:
                ps = [v.var for v in self.voci if v.classe == c]
                if ps:
                    gruppi.append({"params": ps, "lr": self.lr[c]}); lrs.append(self.lr[c])
            if gruppi:
                opt = torch.optim.Adam(gruppi, betas=betas, weight_decay=0.0)
                self._blocchi[nb] = (opt, [p for g in gruppi for p in g["params"]], lrs)
        self.copertura = None                           # {famiglia: [tensori con gradiente non nullo, tensori]} del passo

    # ── blocks for the trainer (same shape as its own: (optimizer, params to clip, base lr per param group))
    def nomi(self) -> list:
        return list(self._blocchi)

    def blocchi(self) -> dict:
        return dict(self._blocchi)

    def conteggi(self) -> dict:
        out = {}
        for v in self.voci:
            c = out.setdefault(v.tf.famiglia, [0, 0, v.classe, v.log])
            c[0] += 1; c[1] += v.tf.master.numel()
        return out

    def descrizione(self) -> str:
        c = self.conteggi()
        n, nv = sum(x[0] for x in c.values()), sum(x[1] for x in c.values())
        fam = ", ".join(f"{f} {x[0]}" for f, x in c.items())
        cl = ", ".join(f"{k} lr {self.lr[k]:g}{' log' if k in self.logc else ''}" for k in CLASSI
                       if any(v.classe == k for v in self.voci))
        return f"F32 allenati: {n} tensori, {nv} valori ({fam}); classi: {cl}; blocchi {self.nomi()}"

    # ── around the step
    def azzera_grad_master(self) -> None:
        """Masters of log classes are not in any optimiser: their accumulated grad is cleared here (start of a step)."""
        for v in self.voci:
            if v.log:
                v.tf.master.grad = None

    @torch.no_grad()
    def gradienti_log(self) -> dict:
        """After the backward, before clipping: dL/dphi = dL/dw * w for log classes (master grad then released);
        records the gradient coverage per family. Returns the coverage."""
        cop = {}
        for v in self.voci:
            g = v.tf.master.grad
            nz = g is not None and bool((g != 0).any())
            c = cop.setdefault(v.tf.famiglia, [0, 0]); c[0] += int(nz); c[1] += 1
            if v.log:
                v.phi.grad = None if g is None else (g * v.tf.master.detach()).to(v.phi.dtype)
                v.tf.master.grad = None
                v.phi_prima = v.phi.detach().clone()
        self.copertura = cop
        return cop

    @torch.no_grad()
    def dopo_passo(self) -> None:
        """After the optimiser step of the F32 blocks: w = s * exp(phi) where phi moved; elsewhere the master is untouched."""
        for v in self.voci:
            if not v.log or v.phi_prima is None:
                continue
            mosso = v.phi != v.phi_prima
            if bool(mosso.any()):
                nuovo = (v.segno * torch.exp(v.phi)).to(v.tf.master.dtype)
                v.tf.master.copy_(torch.where(mosso, nuovo, v.tf.master))
            v.phi_prima = None

    # ── diagnostics
    @torch.no_grad()
    def statistiche(self) -> dict:
        """Relative change per class (||w - w0|| / ||w0||), max |delta log| over the log classes, values changed."""
        acc = {}
        mx, cambiati, tot = 0.0, 0, 0
        for v in self.voci:
            m = v.tf.master.detach()
            d = (m - v.m0).double()
            a = acc.setdefault(v.classe, [0.0, 0.0])
            a[0] += float(d.pow(2).sum()); a[1] += float(v.m0.double().pow(2).sum())
            cambiati += int((m != v.m0).sum()); tot += m.numel()
            if v.log:
                mx = max(mx, float((torch.log(m.abs()) - torch.log(v.m0.abs())).abs().max()))
        out = {f"f32_rel_{c}": math.sqrt(x[0] / max(x[1], 1e-300)) for c, x in acc.items()}
        out["f32_dlog_max"] = mx
        out["f32_cambiati"] = cambiati / max(1, tot)
        return out

    # ── persistence (tappe, ultimo, migliore, --riprendi)
    @torch.no_grad()
    def pacco(self, meta: dict) -> dict:
        """valori = the F32 bytes to deploy (fp32, shape of the tensor in the GGUF file); phi of the log classes (resume);
        sha256 of each value buffer."""
        valori, phi, sha = {}, {}, {}
        for v in self.voci:
            a = np.ascontiguousarray(v.tf.valore_file().astype(np.float32, copy=False))
            valori[v.tf.nome] = torch.from_numpy(a.copy())
            sha[v.tf.nome] = hashlib.sha256(a.tobytes()).hexdigest()
            if v.log:
                phi[v.tf.nome] = v.phi.detach().cpu().clone()
        return {"valori": valori, "phi": phi, "sha256_fp32": sha, "gguf": str(getattr(self.st, "gguf", "")),
                "classi": {v.tf.nome: v.classe for v in self.voci}, "log": list(self.logc), "lr": dict(self.lr),
                "formato": "valori[nome] = fp32 nella forma del tensore F32 del GGUF (f32_esporta.py -> patch_gguf.py)",
                "meta": meta}

    def salva(self, p: Path, meta: dict) -> None:
        p = Path(p)
        torch.save(self.pacco(meta), str(p) + ".tmp")
        Path(str(p) + ".tmp").replace(p)

    @torch.no_grad()
    def carica(self, p: Path) -> dict:
        d = torch.load(p, map_location="cpu", weights_only=False)
        nomi = [v.tf.nome for v in self.voci]
        if sorted(d["valori"]) != sorted(nomi):
            raise ValueError(f"ripresa F32: tensori diversi ({len(d['valori'])} nel pacco, {len(nomi)} allenati)")
        if tuple(d.get("log", ())) != self.logc:
            raise ValueError(f"ripresa F32: classi log {d.get('log')} != {list(self.logc)}")
        self.st.imposta_f32({n: d["valori"][n] for n in nomi})
        for v in self.voci:
            if v.log:
                v.phi.copy_(d["phi"][v.tf.nome].to(v.phi.device, v.phi.dtype))
                if not torch.equal(v.segno, torch.sign(v.tf.master.detach())):
                    raise ValueError(f"ripresa F32: segno di {v.tf.nome} diverso da quello della partenza")
        return d.get("meta", {})

    def stato_ottimizzatori(self) -> dict:
        return {nb: opt.state_dict() for nb, (opt, _, _) in self._blocchi.items()}

    def carica_ottimizzatori(self, d: dict) -> None:
        if sorted(d) != sorted(self._blocchi):
            raise ValueError(f"ripresa F32: blocchi Adam {sorted(d)} != {sorted(self._blocchi)}")
        for nb, (opt, _, _) in self._blocchi.items():
            opt.load_state_dict(d[nb])


@torch.no_grad()
def cancello_presenza(st, F: BloccoF32, avanti: Callable[[], torch.Tensor], h_rif: torch.Tensor) -> list:
    """One tensor per family: master x (1 + 1e-3) must change the output, putting it back must give h_rif AT THE BIT.
    Returns the list of failures (empty = ok). Same rule as the trainer's presence gates of the other branches."""
    errori, viste = [], set()
    for v in F.voci:
        if v.tf.famiglia in viste:
            continue
        viste.add(v.tf.famiglia)
        m = v.tf.master
        salvo = m.detach().clone()
        m.mul_(1.0 + 1e-3)
        h2 = avanti()
        m.copy_(salvo)
        h3 = avanti()
        if torch.equal(h2, h_rif):
            errori.append(f"{v.tf.nome}: cambiarlo non cambia l'uscita (fuori dal percorso dello studente)")
        if not torch.equal(h3, h_rif):
            errori.append(f"{v.tf.nome}: rimesso al suo valore, l'uscita non torna al bit")
    return errori


@torch.no_grad()
def controllo_congelati(st) -> dict:
    """--solo-f32: theta of every ternary tensor (experts + dense ternary) without gradient and exactly 0."""
    ps = list(st.parametri())
    mx = 0.0
    for p in ps:
        if p.numel():
            mx = max(mx, float(p.detach().abs().max()))
    con_grad = sum(1 for p in ps if p.grad is not None)
    return {"theta_n": len(ps), "theta_max_abs": mx, "theta_con_grad": con_grad, "ok": mx == 0.0 and con_grad == 0}

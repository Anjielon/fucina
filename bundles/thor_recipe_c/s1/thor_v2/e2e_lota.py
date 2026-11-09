#!/usr/bin/env python3
"""e2e_lota — LoTA-QAF (Chen et al., «Lossless Ternary Adaptation for Quantization-Aware Fine-Tuning», NeurIPS 2025,
arXiv 2505.18724v2; codice MIT github.com/KingdalfGoodman/LoTA-QAF, letti per intero il 27/9) sui CODICI ternari degli
esperti di Thor-T, nel formato (TQ1_0R_1024 gate/up, TQ1_0 down). 27/9/2026, ramo lota-qaf. File NUOVO: si importano in
SOLA LETTURA `e2e_studente` (DatiFamiglia, motori), `e2e_campi_scala` (decodifica dei trit) e `e2e_pv_codici`
(`salta_byte`: scrittura canonica dei byte; `Registro` / `esporta_codici`: export al bit).

## Cosa fa (paper §3.2-3.3, codice `LoTA/layer.py`, `t_signSGD.py`)
Per ogni esperto e di ogni famiglia (gate, up, down) di ogni strato, due adattatori TERNARI di rango r:
    A_e in {-1,0,1}^(r x C)   (lato INGRESSO, il loro A_T trasposto: iniziale kaiming normale ternarizzata con soglia
                               0,75 mean|w|, ~55% non nulli)
    B_e in {-1,0,1}^(R x r)   (lato USCITA, il loro B_T trasposto: iniziale ZERO -> al passo 0 il modello e' invariato)
    Delta W_e = B_e A_e  (interi in [-r, r]),   W^_e = sign(Delta W_e) 1[|Delta W_e| > omega]  (Eq. 3)
    T_e = clamp(T0_e + W^_e (.) vivo, -1, 1)                                         (Eq. 5, «boundary check» del codice)
dove T0 = codici BASE (i byte all'accensione di LoTA: GGUF + eventuali registri PV di una fase precedente), vivo = gruppo
con d0 != 0 (una scala nulla resta nulla: spostarne i codici non cambia il modello e sporcherebbe l'export). Da T0 un codice
si sposta di AL PIU' un livello: il registro netto T0 -> T e' sempre di salti di un livello (export con `esporta_codici`).
Il fattore di offset mu del paper (Eq. 4, «residual»: media per gruppo di Delta W - omega W^ assorbita nello ZERO del
formato asimmetrico GPTQ) NON esiste qui: TQ1_0/TQ1_0R sono simmetrici (W = d T, nessuno zero). Scelta dichiarata.

**I byte sono sempre una funzione PURA di (base, A, B)**: `sincronizza` porta i byte a clamp(T0 + W^ vivo) scrivendo solo
le differenze con `PV.salta_byte` (byte canonici, identici al quantizzatore C di llama.cpp; un codice che torna a T0
ritrova il byte di base AL BIT). Annullamento = ripristino di A, B + `sincronizza`; ripresa = A, B dal disco +
`sincronizza`; export = `registro_netto` (T0 -> T) + `PV.esporta_codici`. Nessun registro per passo (a milioni di salti
per passo sarebbe ~9 B x salti x passi di disco per nulla).

## Gradiente (straight-through, come `ThresholdFunction_pack.backward` del codice: identita' da W^ a Delta W)
dL/dT[r, c] = d[r, g(c)] dL/dW_rot[r, c] = G[r, c]  (il peso nel dominio dei codici e' W_rot = d (.) T, esatto),
    dL/dB_e = G_e A_e^T        dL/dA_e = B_e^T G_e
senza materializzare G (= dW per esperto, 120 GiB su Thor): con X_rot (S_e, C) e dY (S_e, R) dei token dell'esperto,
    P[s,g,k] = sum_{c in g} X[s,c] A[k,c]         dL/dB[r,k] = sum_g d[r,g] sum_s dY[s,r] P[s,g,k]
    Q[s,g,k] = sum_r dY[s,r] d[r,g] B[r,k]         dL/dA[k,c] = sum_s X[s,c] Q[s,g(c),k]
costo 2 S r (2C + 2 R NG) FLOP per esperto e famiglia (gate/up di Thor: 41k S r; down: 18k S r) contro 2 S R C
dell'avanti (2,1 M S): +31% / +14% di un avanti degli esperti a r = 16. Il gancio sta nel `dd` del motore (stessa
chiamata che calcola dL/dd, stessi dY e X_rot, nessun indietro in piu'): `avvolgi_motore` (sottoclasse dinamica del
motore di base: `isinstance(..., MotoreTriton)` resta vero) chiama il `dd` di base e poi `raccogli`. Il gradiente del
paper passa anche dove il clamp blocca (codice gia' al bordo): stessa scelta qui (nessuna maschera nel STE).

## t-SignSGD (Eq. 6 + `t_signSGD.py`), con memoria facoltativa
    X <- clip(X - sign(m) 1[|m| > max(tau, sigma_t)], -1, 1),   sigma_t = quantile(|m|, 1 - p_t) (interp. lineare)
per TENSORE: qui il tensore e' (strato, famiglia, A|B) coi 256 esperti insieme (gli esperti ricevono token diversi: il
top-p va agli esperti con segnale). p_t: 5% -> 0,1% linearmente nell'80% dei passi, poi 0,1% -> 0,01% (codice del paper;
il testo §3.3 dice «decays to 0.01%», §4.1 da' la forma a due tratti). tau = 1e-9 sul gradiente VERO (qui il gradiente e'
K volte quello vero: soglia tau K). m = gradiente del passo (lr = 1, il paper: Eq. 6 «does not use a learning rate»; nel
codice `lr` e' un argomento inutilizzato) oppure, con `lr` < 1, la media esponenziale m <- (1 - lr) m + lr g (lo scrive il
paper in App. E-F come lavoro futuro; e' il Bop del registro §21 con soglia top-p): un solo buffer u = m / lr per tensore,
u <- (1 - lr) u prima del passo, poi si accumula g; lr = 1 azzera u, cioe' m = g al bit. Il trucco del codice in fase 2
(|g| sopra il quantile 0,99995 escluso se ||g|| > 10) NON si porta: la soglia 10 e' legata alla loro scala di perdita.

## Regione di fiducia (TECNICHE §31/§32: salti che insieme peggiorano)
`allena` misura la perdita sulla sonda (primo micro-lotto del passo dopo, mai visto) prima e dopo `sincronizza`; se sale,
A e B tornano indietro (e i byte con loro, al bit) e il moltiplicatore di p si dimezza (pavimento `mult_min`). Senza codici
cambiati la sonda non si misura (i passi di «carica» di Delta W sotto omega non costano avanti).

Memoria persistente (Thor-T: 40 strati x 3 famiglie x 256 esperti, R x C = 512 x 2048 o 2048 x 512): A, B int8
= 256 r (R + C) = 655.360 r B per famiglia e strato -> 78,6 M r B = **1,26 GB a r = 16**; u (gradiente/momento fp32)
**5,03 GB a r = 16**; base (copia dei byte dei codici) **6,96 GB** (misura sui metadati del GGUF: `memoria_thor`).
Transitori: un pezzo di esperti per `sincronizza` (32 esperti gate: Delta W fp32 134 MB + trit int8).

Prove: `e2e_test_lota.py` (solo CPU).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import e2e_studente as S  # noqa: F401  (sola lettura)
import e2e_campi_scala as CAMPI  # sola lettura
import e2e_codici_scale as CS  # sola lettura
import e2e_pv_codici as PV  # sola lettura: salta_byte, Registro, famiglia_scrivibile, FAMIGLIE_IDX

VERSIONE = "e2e_lota 1 (27/9/2026)"


# ═══════════════════════════════════ configurazione ═══════════════════════════════════
@dataclass(frozen=True)
class ConfigLoTA:
    rango: int
    omega: Optional[float] = None          # None = 0,75 r (paper §4.1 per Alpaca/GSM8K/SQL)
    top_p: float = 0.05                    # quota iniziale di voci aggiornate per tensore (paper: 5%)
    top_p_fine: float = 1e-3               # alla fine della fase 1 (80% dei passi): 0,1%
    top_p_coda: float = 1e-4               # all'ultimo passo: 0,01%
    lr: float = 1.0                        # tasso della media del gradiente; 1 = t-SignSGD del paper (nessuna memoria)
    tau: float = 1e-9                      # soglia minima sul |gradiente| VERO (paper: 1e-9)
    seme: int = 20260927
    mult_min: float = 1.0 / 64             # pavimento del moltiplicatore della regione di fiducia
    pezzo_esperti: int = 32                # esperti per pezzo in `sincronizza` (transitorio limitato)
    dtype_u: torch.dtype = torch.float32   # buffer del gradiente/momento (float64 nelle prove)
    init: str = "paper"                    # "paper": A kaiming ternarizzata; "gradiente": A = ternario di Omega^T G (sotto)
    tolleranza: float = 0.0                # la sonda tiene se dopo <= prima (1 + tolleranza); 0 = come il passo V
    finestra: int = 0                      # > 0: sonda FISSA a finestra (sotto) al posto della sonda passo per passo

    def __post_init__(self):
        if not isinstance(self.rango, int) or self.rango < 1:
            raise ValueError(f"rango LoTA {self.rango!r}: intero >= 1")
        if not 0.0 < self.omega_eff < self.rango:
            raise ValueError(f"omega {self.omega_eff} fuori da (0, rango = {self.rango}) (paper Eq. 3)")
        if not (0.0 < self.top_p_coda <= self.top_p_fine <= self.top_p <= 1.0):
            raise ValueError(f"top-p: serve 0 < coda {self.top_p_coda} <= fine {self.top_p_fine} <= iniziale {self.top_p} <= 1")
        if not 0.0 < self.lr <= 1.0:
            raise ValueError(f"lr LoTA {self.lr} fuori da (0, 1] (tasso della media del gradiente; 1 = paper)")
        if self.tau < 0:
            raise ValueError(f"tau {self.tau} < 0")
        if not 0.0 < self.mult_min <= 1.0:
            raise ValueError(f"mult_min {self.mult_min} fuori da (0, 1]")
        if self.init not in ("paper", "gradiente"):
            raise ValueError(f"init LoTA {self.init!r}: 'paper' o 'gradiente'")
        if not self.tolleranza >= 0.0:
            raise ValueError(f"tolleranza della sonda {self.tolleranza} < 0")
        if not isinstance(self.finestra, int) or self.finestra < 0:
            raise ValueError(f"finestra della sonda {self.finestra!r}: intero >= 0")

    @property
    def omega_eff(self) -> float:
        return 0.75 * self.rango if self.omega is None else float(self.omega)

    def firma(self) -> dict:
        """Cio' che DEFINISCE il modello (A, B, omega -> codici): cambiarlo con un adattatore allenato cambia i codici."""
        return {"rango": self.rango, "omega": self.omega_eff, "seme": self.seme, "init": self.init}


def top_p_passo(cfg: ConfigLoTA, k: int, n: int) -> float:
    """Quota p del passo LoTA k (1-based) su n: la schedula di `t_signSGD.py` (threshold_ratio 0,95 -> min_grad 0,999 ->
    filter_upper 0,9999 = top 5% -> 0,1% -> 0,01%) con i tre valori della config. n <= 0: costante top_p."""
    if n <= 0:
        return cfg.top_p
    fine1 = n * 0.8
    if k <= fine1:
        prog = k / fine1
        return cfg.top_p + prog * (cfg.top_p_fine - cfg.top_p)
    if k >= n:
        return cfg.top_p_coda
    prog = (k - fine1) / (n - fine1)
    return cfg.top_p_fine + prog * (cfg.top_p_coda - cfg.top_p_fine)


def quantile_lineare(v: torch.Tensor, q: float) -> torch.Tensor:
    """torch.quantile(v, q) (interpolazione lineare, NaN esclusi a monte) senza il suo limite di 2^24 elementi: due
    kthvalue e torch.lerp (lo stesso lerp di torch.quantile: stesso valore al bit sulle prove)."""
    v = v.reshape(-1)
    n = v.numel()
    if n == 0:
        return torch.zeros((), dtype=v.dtype, device=v.device)
    # come ATen (Sorting.cpp, quantile_compute): rango = q * (n - 1) nel dtype di v, sotto = troncamento, sopra = ceil,
    # peso = rango - sotto nel dtype di v, poi lerp
    rango = torch.tensor(q, dtype=v.dtype) * (n - 1)
    lo = int(rango.to(torch.int64))
    hi = min(int(torch.ceil(rango).to(torch.int64)), n - 1)
    a = torch.kthvalue(v, lo + 1).values
    b = a if hi == lo else torch.kthvalue(v, hi + 1).values
    return torch.lerp(a, b, (rango - lo).to(device=v.device))


def _ternario_kaiming(g: torch.Generator, forma: tuple) -> torch.Tensor:
    """`IntLinear.__init__` del paper: kaiming_normal_, soglia 0,75 mean|w| sul tensore, poi {-1,0,1} (la scala di
    kaiming non conta: soglia relativa). Sulla CPU col generatore dato: stesso A su CPU e GPU."""
    w = torch.randn(forma, generator=g, dtype=torch.float32)
    delta = 0.75 * w.abs().mean()
    return torch.where(w > delta, 1, torch.where(w < -delta, -1, 0)).to(torch.int8)


def ternarizza_per_esperto(Z: torch.Tensor) -> torch.Tensor:
    """(E, ...) reale -> {-1,0,1} int8 con la soglia dell'init del paper (0,75 mean|z|) calcolata PER esperto."""
    E = Z.shape[0]
    delta = 0.75 * Z.abs().reshape(E, -1).mean(1).reshape((E,) + (1,) * (Z.dim() - 1))
    return torch.where(Z > delta, 1, torch.where(Z < -delta, -1, 0)).to(torch.int8)


def _seme_famiglia(seme: int, L: int, nome: str) -> int:
    h = hashlib.sha256(f"{seme}|{L}|{nome}".encode()).digest()
    return int.from_bytes(h[:8], "little") & ((1 << 63) - 1)


# ═══════════════════════════════════ impacchettamento 2 bit ═══════════════════════════════════
def impacchetta(t: torch.Tensor) -> np.ndarray:
    """int8 in {-1,0,1} -> uint8, 4 valori per byte ((v+1) << 2i): 1/4 della memoria su disco."""
    v = (t.detach().to("cpu", torch.int16).reshape(-1) + 1).to(torch.uint8)
    pad = (-v.numel()) % 4
    if pad:
        v = torch.cat([v, torch.zeros(pad, dtype=torch.uint8)])
    v = v.view(-1, 4)
    b = v[:, 0] | (v[:, 1] << 2) | (v[:, 2] << 4) | (v[:, 3] << 6)
    return b.numpy().copy()


def spacchetta(b: np.ndarray, forma: tuple) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(b, dtype=np.uint8))
    v = torch.stack([(x >> (2 * i)) & 3 for i in range(4)], dim=1).reshape(-1)
    n = int(np.prod(forma))
    out = v[:n].to(torch.int16) - 1
    if bool((out.abs() > 1).any()):
        raise ValueError("valore 3 in un adattatore impacchettato: file rotto")
    return out.to(torch.int8).reshape(forma)


# ═══════════════════════════════════ adattatore di una famiglia ═══════════════════════════════════
class AdattatoreFamiglia:
    """A (E, r, C), B (E, R, r) int8 ternari; uA, uB (stesse forme, dtype_u): gradiente/momento del passo; base (E, R, NBR)
    uint8 = byte dei codici T0 (copia fatta all'accensione)."""

    def __init__(self, chiave: tuple, fam, cfg: ConfigLoTA):
        self.chiave = (int(chiave[0]), str(chiave[1]))
        self.fam = fam
        self.cfg = cfg
        dev = fam.blocchi.device
        E, R, C, r = fam.E, fam.R, fam.C, cfg.rango
        g = torch.Generator().manual_seed(_seme_famiglia(cfg.seme, self.chiave[0], self.chiave[1]))
        self.A = torch.stack([_ternario_kaiming(g, (r, C)) for _ in range(E)]).to(dev)
        self.B = torch.zeros(E, R, r, dtype=torch.int8, device=dev)
        self.uA = torch.zeros(E, r, C, dtype=cfg.dtype_u, device=dev)
        self.uB = torch.zeros(E, R, r, dtype=cfg.dtype_u, device=dev)
        self.base = fam.blocchi.clone()

    def byte(self) -> dict:
        return {"AB": self.A.numel() + self.B.numel(),
                "u": (self.uA.numel() + self.uB.numel()) * self.uA.element_size(),
                "base": self.base.numel()}

    # ─────────── gradiente (gancio del dd) ───────────
    @torch.no_grad()
    def raccogli(self, gy: torch.Tensor, x: torch.Tensor, piano, d: torch.Tensor) -> None:
        """Accumula in uB, uA dL/dB, dL/dA di questo micro-lotto (forma fattorizzata, niente dW). gy (S, R) e x (S, C)
        sono quelli del `dd` (token ordinati per esperto, x nel dominio dei codici: X_rot per le famiglie ruotate)."""
        fam, r = self.fam, self.cfg.rango
        R, C, NG, gs = fam.R, fam.C, fam.NG, fam.gs
        acc = torch.float64 if self.uA.dtype == torch.float64 else torch.float32
        o = piano.offs.tolist()                                  # una sincronizzazione per famiglia e micro-lotto
        for e in range(fam.E):
            a, b = o[e], o[e + 1]
            if b <= a:
                continue
            n = b - a
            X = x[a:b].to(acc).view(n, NG, gs)
            dY = gy[a:b].to(acc)
            de = d[e].to(acc)                                    # (R, NG)
            Ae = self.A[e].to(acc).view(r, NG, gs)
            P = torch.einsum("ngc,kgc->ngk", X, Ae)             # (n, NG, r)
            M = (dY.T @ P.reshape(n, NG * r)).view(R, NG, r)
            self.uB[e] += (M * de.unsqueeze(-1)).sum(1).to(self.uB.dtype)
            Be = self.B[e].to(acc)                               # (R, r)
            Q = (dY @ (de.unsqueeze(-1) * Be.unsqueeze(1)).reshape(R, NG * r)).view(n, NG, r)
            self.uA[e] += torch.einsum("ngk,ngc->kgc", Q, X).reshape(r, C).to(self.uA.dtype)

    # ─────────── t-SignSGD ───────────
    @torch.no_grad()
    def passo(self, p: float, soglia_abs: float) -> dict:
        """Eq. 6 su A e su B (un tensore ciascuno, esperti insieme). Ritorna per l'annullamento {nome: (idx, vecchi)} e i
        conteggi (voci scelte, voci cambiate: una voce gia' a +-1 spinta oltre non cambia, come col clip del paper)."""
        out = {}
        for nome, X, U in (("A", self.A, self.uA), ("B", self.B, self.uB)):
            a = U.abs().reshape(-1)
            sigma = quantile_lineare(a, 1.0 - p)
            sel = (a > sigma) & (a > soglia_abs)
            idx = torch.nonzero(sel).reshape(-1)
            Xf = X.view(-1)
            vecchi = Xf[idx].clone()
            nuovi = (vecchi.to(torch.int16) - torch.sign(U.reshape(-1)[idx]).to(torch.int16)).clamp(-1, 1).to(torch.int8)
            Xf[idx] = nuovi
            out[nome] = (idx, vecchi, int(idx.numel()), int((nuovi != vecchi).sum()))
        return out

    @torch.no_grad()
    def annulla_passo(self, mosse: dict) -> None:
        for nome, X in (("A", self.A), ("B", self.B)):
            idx, vecchi = mosse[nome][0], mosse[nome][1]
            X.view(-1)[idx] = vecchi

    # ─────────── codici ───────────
    def _trit(self, B: torch.Tensor) -> torch.Tensor:
        fam = self.fam
        return CAMPI.trit_torch(B.reshape(-1, fam.nb), fam.tipo).view(B.shape[0], fam.R, fam.C)

    @torch.no_grad()
    def w_hat(self, e0: int, e1: int) -> torch.Tensor:
        """W^ (e1-e0, R, C) int8 = sign(B A) 1[|B A| > omega], gia' moltiplicato per `vivo` (gruppi con d0 != 0). Il
        prodotto e' esatto in fp32 (interi <= r)."""
        fam = self.fam
        DW = torch.matmul(self.B[e0:e1].to(torch.float32), self.A[e0:e1].to(torch.float32))
        Wh = (torch.sign(DW) * (DW.abs() > self.cfg.omega_eff)).to(torch.int8)
        vivo = (fam.d0[e0:e1] != 0).repeat_interleave(fam.gs, dim=-1)
        return Wh * vivo.to(torch.int8)

    @torch.no_grad()
    def bersaglio(self, e0: int, e1: int) -> torch.Tensor:
        """Codici che i byte devono portare: clamp(T0 + W^ vivo, -1, 1) (e1-e0, R, C) int8."""
        T0 = self._trit(self.base[e0:e1])
        return (T0.to(torch.int16) + self.w_hat(e0, e1).to(torch.int16)).clamp(-1, 1).to(torch.int8)

    @torch.no_grad()
    def sincronizza(self, applica: bool = True) -> dict:
        """Porta i byte ai codici bersaglio (solo le differenze, salti di uno o due livelli in due scritture canoniche).
        applica=False: conta soltanto. Ritorna codici cambiati, codici diversi dalla base, W^ non nulli (vivi)."""
        fam = self.fam
        cambiati = diversi = nonnulli = 0
        for e0 in range(0, fam.E, self.cfg.pezzo_esperti):
            e1 = min(fam.E, e0 + self.cfg.pezzo_esperti)
            T0 = self._trit(self.base[e0:e1])
            Wh = self.w_hat(e0, e1)
            Tn = (T0.to(torch.int16) + Wh.to(torch.int16)).clamp(-1, 1).to(torch.int8)
            Tc = self._trit(fam.blocchi[e0:e1])
            dif = Tn != Tc
            nonnulli += int((Wh != 0).sum())
            diversi += int((Tn != T0).sum())
            n = int(dif.sum())
            cambiati += n
            if applica and n:
                e, r, c = torch.nonzero(dif, as_tuple=True)
                ta, tn = Tc[e, r, c].to(torch.int32), Tn[e, r, c].to(torch.int32)
                meta = ta + torch.sign(tn - ta)
                PV.salta_byte(fam.blocchi, fam.tipo, e + e0, r, c, ta, meta)
                due = (tn - ta).abs() == 2
                if bool(due.any()):
                    PV.salta_byte(fam.blocchi, fam.tipo, e[due] + e0, r[due], c[due], meta[due], tn[due])
            del T0, Wh, Tn, Tc, dif
        return {"cambiati": cambiati, "diversi_da_base": diversi, "w_hat_non_nulli": nonnulli}

    @torch.no_grad()
    def registro_netto(self) -> PV.Registro:
        """Salti T0 -> codici attuali (sempre di UN livello: clamp(T0 + W^) dista al piu' 1 da T0), righe del registro PV."""
        fam = self.fam
        parti = []
        for e0 in range(0, fam.E, self.cfg.pezzo_esperti):
            e1 = min(fam.E, e0 + self.cfg.pezzo_esperti)
            T0, Tc = self._trit(self.base[e0:e1]), self._trit(fam.blocchi[e0:e1])
            e, r, c = torch.nonzero(T0 != Tc, as_tuple=True)
            if e.numel() == 0:
                continue
            tv, tn = T0[e, r, c], Tc[e, r, c]
            if bool(((tn.to(torch.int16) - tv.to(torch.int16)).abs() != 1).any()):
                raise RuntimeError(f"{self.chiave}: codice a due livelli dalla base: byte non allineati a LoTA")
            righe = np.zeros(int(e.numel()), dtype=PV.REG_DTYPE)
            righe["L"], righe["f"] = self.chiave[0], PV.FAMIGLIE_IDX[self.chiave[1]]
            righe["e"], righe["r"], righe["c"] = (e + e0).cpu().numpy(), r.cpu().numpy(), c.cpu().numpy()
            righe["t_vecchio"], righe["t_nuovo"] = tv.cpu().numpy(), tn.cpu().numpy()
            parti.append(righe)
        return PV.Registro(np.concatenate(parti) if parti else np.zeros(0, dtype=PV.REG_DTYPE))

    @torch.no_grad()
    def controlla(self) -> dict:
        """Invariante: byte == bersaglio (per esperto a pezzi), trit in {-1,0,1}. Solleva se no."""
        fam = self.fam
        for e0 in range(0, fam.E, self.cfg.pezzo_esperti):
            e1 = min(fam.E, e0 + self.cfg.pezzo_esperti)
            if not torch.equal(self._trit(fam.blocchi[e0:e1]), self.bersaglio(e0, e1)):
                raise RuntimeError(f"{self.chiave}: byte diversi da clamp(T0 + W^) negli esperti {e0}-{e1}")
        return {"ok": True}


# ═══════════════════════════════════ motore col gancio ═══════════════════════════════════
_CLASSI: dict = {}


def _classe_lota(cls):
    """Sottoclasse di `cls` (motore dello studente) il cui `dd` chiama quello di base e poi il gancio LoTA."""
    if cls not in _CLASSI:
        def dd(self, gy, x, fam, piano, d_dtype):
            out = cls.dd(self, gy, x, fam, piano, d_dtype)
            self._lota.raccogli_famiglia(gy, x, fam, piano)
            return out
        _CLASSI[cls] = type(cls.__name__ + "LoTA", (cls,), {"dd": dd, "__doc__": f"{cls.__name__} + gancio LoTA nel dd"})
    return _CLASSI[cls]


def avvolgi_motore(base, stato: "StatoLoTA"):
    if getattr(base, "_lota", None) is not None:
        raise ValueError("motore gia' avvolto da LoTA")
    sub = _classe_lota(type(base))
    m = object.__new__(sub)
    m.__dict__.update(base.__dict__)
    m._lota = stato
    m.nome = f"{getattr(base, 'nome', type(base).__name__)}+LoTA"
    return m


# ═══════════════════════════════════ stato di tutte le famiglie ═══════════════════════════════════
class StatoLoTA:
    """Adattatori di tutte le famiglie {(L, nome): AdattatoreFamiglia}, raccolta accesa/spenta, moltiplicatore della
    regione di fiducia, contatore dei passi LoTA (1-based nella schedula)."""

    def __init__(self, famiglie: dict, cfg: ConfigLoTA, scale: Optional[dict] = None):
        self.cfg = cfg
        self.ad = {k: AdattatoreFamiglia(k, f, cfg) for k, f in famiglie.items()}
        self._per_id = {id(f): self.ad[k] for k, f in famiglie.items()}
        self.scale = scale or {}                    # {(L, nome): callable -> d corrente (E, R, NG)}
        self.attivo = False
        self.mult = 1.0
        self.passi = 0
        self.inizializzato = cfg.init == "paper"    # "gradiente": il primo passo misura Omega^T G e da' A (niente codici)
        self.ultimo_init = False
        self.migliore: Optional[dict] = None        # sonda a finestra: {"valore", "passo", "A", "B"} (A, B a 2 bit, CPU)
        self.senza_miglioramento = 0

    # ─────────── raccolta ───────────
    def raccogli_famiglia(self, gy, x, fam, piano) -> None:
        if not self.attivo:
            return
        ad = self._per_id.get(id(fam))
        if ad is None:
            raise KeyError(f"famiglia {getattr(fam, 'strato', '?')}.{getattr(fam, 'nome', '?')} senza adattatore LoTA")
        with torch.no_grad():
            d = self.scale[ad.chiave]() if ad.chiave in self.scale else ad.fam.d0
        ad.raccogli(gy, x, piano, d.detach())

    def inizia_passo(self) -> None:
        """u <- (1 - lr) u (lr = 1: zero, cioe' m = gradiente del passo, il paper). Con init "gradiente" non ancora fatta:
        B <- Omega (ternario casuale, seme per famiglia) e u = 0: nel passo il ramo dL/dA raccoglie Z = Omega^T G."""
        for ad in self.ad.values():
            for u in (ad.uA, ad.uB):
                if self.cfg.lr >= 1.0 or not self.inizializzato:
                    u.zero_()
                else:
                    u.mul_(1.0 - self.cfg.lr)
            if not self.inizializzato:
                g = torch.Generator().manual_seed(_seme_famiglia(self.cfg.seme + 1, ad.chiave[0], ad.chiave[1]))
                ad.B.copy_(torch.stack([_ternario_kaiming(g, tuple(ad.B.shape[1:])) for _ in range(ad.B.shape[0])])
                           .to(ad.B.device))

    class _Raccolta:
        def __init__(self, st):
            self.st = st

        def __enter__(self):
            self.st.attivo = True

        def __exit__(self, *a):
            self.st.attivo = False

    def raccolta(self):
        return StatoLoTA._Raccolta(self)

    # ─────────── passo ───────────
    def p_passo(self, k: int, n: int) -> float:
        return top_p_passo(self.cfg, k, n) * self.mult

    def passo(self, p: float, K: float = 1.0) -> dict:
        """t-SignSGD su tutti gli adattatori. K = scala della perdita (u e' K volte il gradiente vero). Ritorna le mosse.
        Init "gradiente" (variante NOSTRA, non del paper: LoRA-GA in ternario con uno schizzo a una passata): al primo passo
        A_e <- ternario(Z_e), Z_e = Omega_e^T G_e (r x C: le righe sono miscele casuali delle righe di G_e, cioe' uno
        schizzo del suo spazio delle righe dominante), soglia 0,75 mean|Z_e| per esperto; B <- 0 (codici invariati),
        u <- 0. Un esperto senza token (Z_e = 0) tiene la A del paper. Nessuna mossa di codice in quel passo."""
        self.ultimo_init = False
        if not self.inizializzato:
            vuoto = torch.zeros(0, dtype=torch.int64)
            for ad in self.ad.values():
                Z = ad.uA
                nuovo = ternarizza_per_esperto(Z)
                vivo = Z.abs().reshape(Z.shape[0], -1).amax(1) > 0
                ad.A[vivo] = nuovo[vivo]
                ad.B.zero_()
                ad.uA.zero_()
                ad.uB.zero_()
            self.inizializzato = True
            self.ultimo_init = True
            return {k: {"A": (vuoto, vuoto.to(torch.int8), 0, 0), "B": (vuoto, vuoto.to(torch.int8), 0, 0)}
                    for k in self.ad}
        soglia = self.cfg.tau * K / self.cfg.lr          # |m| = lr |u| > tau K
        return {k: ad.passo(p, soglia) for k, ad in self.ad.items()}

    def annulla(self, mosse: dict) -> dict:
        for k, m in mosse.items():
            self.ad[k].annulla_passo(m)
        return self.sincronizza()

    def sincronizza(self, applica: bool = True) -> dict:
        tot = {"cambiati": 0, "diversi_da_base": 0, "w_hat_non_nulli": 0}
        for ad in self.ad.values():
            for k2, v in ad.sincronizza(applica).items():
                tot[k2] += v
        return tot

    def esito(self, prima: float, dopo: float, n_cambiati: int) -> bool:
        """Regione di fiducia: tiene se la sonda non sale oltre la tolleranza relativa (o se nessun codice e' cambiato);
        altrimenti mult / 2. Tolleranza 0 = il criterio del passo V."""
        tiene = n_cambiati == 0 or dopo <= prima * (1.0 + self.cfg.tolleranza)
        if not tiene:
            self.mult = max(self.cfg.mult_min, self.mult / 2)
        return tiene

    def _fotografa(self) -> dict:
        return {"A": {k: impacchetta(ad.A) for k, ad in self.ad.items()},
                "B": {k: impacchetta(ad.B) for k, ad in self.ad.items()}}

    def _torna_a(self, foto: dict) -> dict:
        for k, ad in self.ad.items():
            ad.A.copy_(spacchetta(foto["A"][k], tuple(ad.A.shape)).to(ad.A.device))
            ad.B.copy_(spacchetta(foto["B"][k], tuple(ad.B.shape)).to(ad.B.device))
        return self.sincronizza()

    def esito_finestra(self, valore: float, passo: int) -> dict:
        """Sonda a FINESTRA (variante nostra, nata dal §4 del LEGGIMI: LoTA prima sale e poi scende, e la sonda passo per passo
        la tronca al primo gradino). La sonda e' FISSA (le stesse righe a ogni passo: valori confrontabili fra i passi) e si
        misura DOPO aver scritto i codici del passo, che si tengono sempre. Se il valore batte il migliore, A e B si
        fotografano (2 bit, CPU). Se per `finestra` passi di fila non lo batte, o se lo supera di oltre la tolleranza
        (0 = nessun tetto alla salita dentro la finestra), A e B tornano alla foto del migliore (byte con loro, al bit) e p si
        dimezza. Il passo P (theta) intanto continua: il migliore e' quello della sonda con le scale di ALLORA."""
        f = self.cfg.finestra
        if f <= 0:
            raise ValueError("esito_finestra senza finestra > 0")
        if self.migliore is None or valore < self.migliore["valore"]:
            self.migliore = {"valore": float(valore), "passo": int(passo), **self._fotografa()}
            self.senza_miglioramento = 0
            return {"azione": "migliore", "valore": valore, "migliore": valore}
        self.senza_miglioramento += 1
        oltre = self.cfg.tolleranza > 0 and valore > self.migliore["valore"] * (1.0 + self.cfg.tolleranza)
        if self.senza_miglioramento >= f or oltre:
            ris = self._torna_a(self.migliore)
            self.mult = max(self.cfg.mult_min, self.mult / 2)
            self.senza_miglioramento = 0
            return {"azione": "ritorno", "valore": valore, "migliore": self.migliore["valore"],
                    "passo_del_migliore": self.migliore["passo"], "motivo": "tetto" if oltre else "finestra",
                    "diversi_da_base": ris["diversi_da_base"]}
        return {"azione": "attesa", "valore": valore, "migliore": self.migliore["valore"],
                "senza_miglioramento": self.senza_miglioramento}

    def registri_netti(self) -> list:
        """UN registro (salti T0 -> attuale di tutte le famiglie), in una lista: e' cio' che va dopo i registri PV congelati."""
        return [PV.Registro.unisci([ad.registro_netto() for ad in self.ad.values()])]

    def controlla(self) -> dict:
        for ad in self.ad.values():
            ad.controlla()
        return {"ok": True}

    def conteggi(self) -> dict:
        tot_codici = sum(ad.fam.E * ad.fam.R * ad.fam.C for ad in self.ad.values())
        nz = {n: sum(int((getattr(ad, n) != 0).sum()) for ad in self.ad.values()) for n in ("A", "B")}
        nv = {n: sum(getattr(ad, n).numel() for ad in self.ad.values()) for n in ("A", "B")}
        return {"codici": tot_codici, "A_non_nulli": nz["A"], "A_voci": nv["A"], "B_non_nulli": nz["B"],
                "B_voci": nv["B"]}

    def byte(self) -> dict:
        tot = {"AB": 0, "u": 0, "base": 0}
        for ad in self.ad.values():
            for k, v in ad.byte().items():
                tot[k] += v
        return tot

    # ─────────── ripresa ───────────
    def istantanea(self) -> dict:
        """A e B impacchettati a 2 bit; u solo con lr < 1 (con lr = 1 si azzera a ogni passo: non e' stato); mult, passi,
        firma della configurazione. Tutto sulla CPU."""
        ist = {"versione": VERSIONE, "firma": self.cfg.firma(), "lr": self.cfg.lr, "mult": self.mult,
               "passi": self.passi, "inizializzato": self.inizializzato, "chiavi": [list(k) for k in self.ad],
               "A": {}, "B": {}, "u": {}, "senza_miglioramento": self.senza_miglioramento, "migliore": None}
        if self.migliore is not None:
            m = self.migliore
            ist["migliore"] = {"valore": m["valore"], "passo": m["passo"],
                               "A": {f"{k[0]}.{k[1]}": torch.from_numpy(v) for k, v in m["A"].items()},
                               "B": {f"{k[0]}.{k[1]}": torch.from_numpy(v) for k, v in m["B"].items()}}
        for k, ad in self.ad.items():
            s = f"{k[0]}.{k[1]}"
            ist["A"][s] = torch.from_numpy(impacchetta(ad.A))
            ist["B"][s] = torch.from_numpy(impacchetta(ad.B))
            if self.cfg.lr < 1.0:
                ist["u"][s] = (ad.uA.detach().cpu().clone(), ad.uB.detach().cpu().clone())
        return ist

    def ripristina(self, ist: dict, sincronizza: bool = True) -> dict:
        if ist["firma"] != self.cfg.firma():
            raise RuntimeError(f"LoTA: firma salvata {ist['firma']} != configurazione {self.cfg.firma()} (rango/omega/seme "
                               "definiscono i codici: nuova corsa)")
        if [tuple(k) for k in ist["chiavi"]] != list(self.ad):
            raise RuntimeError("LoTA: famiglie salvate diverse da quelle dello studente")
        for k, ad in self.ad.items():
            s = f"{k[0]}.{k[1]}"
            ad.A.copy_(spacchetta(ist["A"][s].numpy(), tuple(ad.A.shape)).to(ad.A.device))
            ad.B.copy_(spacchetta(ist["B"][s].numpy(), tuple(ad.B.shape)).to(ad.B.device))
            if self.cfg.lr < 1.0 and s in ist["u"]:
                ad.uA.copy_(ist["u"][s][0].to(ad.uA.device, ad.uA.dtype))
                ad.uB.copy_(ist["u"][s][1].to(ad.uB.device, ad.uB.dtype))
            elif self.cfg.lr < 1.0:
                ad.uA.zero_()
                ad.uB.zero_()
        self.mult = float(ist["mult"])
        self.passi = int(ist["passi"])
        self.inizializzato = bool(ist.get("inizializzato", True))
        self.senza_miglioramento = int(ist.get("senza_miglioramento", 0))
        m = ist.get("migliore")
        self.migliore = None if m is None else {
            "valore": float(m["valore"]), "passo": int(m["passo"]),
            "A": {k: m["A"][f"{k[0]}.{k[1]}"].numpy() for k in self.ad},
            "B": {k: m["B"][f"{k[0]}.{k[1]}"].numpy() for k in self.ad}}
        return self.sincronizza() if sincronizza else {}


# ═══════════════════════════════════ innesto nello studente ═══════════════════════════════════
def installa(st, cfg: ConfigLoTA, registri_congelati: Optional[list] = None) -> StatoLoTA:
    """Mette LoTA in TUTTI gli strati di uno `S.Studente` (o di uno studente con la stessa interfaccia: strati, esperti(L)
    con fam/scala/mot, mot). Su CPU i byte passano da `PV.famiglia_scrivibile` (copia: il memmap e' in sola lettura).
    `registri_congelati` (registri PV di una fase precedente) si rigiocano PRIMA di copiare la base: T0 = GGUF + PV."""
    fams, scale = {}, {}
    for L in st.strati:
        es = st.esperti(L)
        for n, f in es.fam.items():
            if f.blocchi.device.type == "cpu" and not getattr(f, "_pv_scrivibile", False):
                PV.famiglia_scrivibile(f)
            fams[(int(L), n)] = f
            scale[(int(L), n)] = (lambda es=es, n=n: es.scala(n))
    if registri_congelati:
        PV.rigioca(fams, registri_congelati)
    stato = StatoLoTA(fams, cfg, scale)
    mot = avvolgi_motore(st.mot, stato)
    for L in st.strati:
        st.esperti(L).mot = mot
    st.mot = mot
    return stato


# ═══════════════════════════════════ memoria su Thor-T (metadati del GGUF) ═══════════════════════════════════
def memoria_thor(gguf: Path, ranghi=(8, 16, 32), dtype_u_byte: int = 4, strati=range(40)) -> dict:
    """Byte persistenti per rango, dai metadati del GGUF (forme e taglie dei tensori degli esperti degli `strati` che lo
    studente carica: 0-39, il blocco MTP 40 no): A+B int8 (e a 2 bit su disco), u, base. Nessun dato oltre l'intestazione."""
    from gguf import GGUFReader
    r = GGUFReader(str(gguf))
    fam, saltati = [], []
    for t in r.tensors:
        for nome, pezzo in S.FAMIGLIE.items():
            if t.name.endswith(f".{pezzo}.weight") and t.name.startswith("blk."):
                if t.tensor_type.name not in CS.LAYOUT:          # es. blocco MTP (blk.40) in Q8_0: non e' ternario
                    saltati.append(f"{t.name} {t.tensor_type.name}")
                    continue
                L = int(t.name.split(".")[1])
                if L not in strati:
                    saltati.append(f"{t.name} (strato fuori dallo studente)")
                    continue
                E, R, NBR = (int(x) for x in np.asarray(t.data).shape)
                nb, qk = CS.LAYOUT[t.tensor_type.name][:2]
                fam.append({"L": L, "nome": nome, "tipo": t.tensor_type.name, "E": E, "R": R, "C": NBR // nb * qk,
                            "byte": int(t.n_bytes)})
    del r
    out = {"famiglie": len(fam), "strati": len({f["L"] for f in fam}), "codici": sum(f["E"] * f["R"] * f["C"] for f in fam),
           "base_byte": sum(f["byte"] for f in fam), "saltati": saltati,
           "forme": sorted({(f["tipo"], f["E"], f["R"], f["C"]) for f in fam}), "per_rango": {}}
    for rr in ranghi:
        ab = sum(f["E"] * rr * (f["R"] + f["C"]) for f in fam)
        out["per_rango"][rr] = {"AB_int8": ab, "AB_2bit_disco": (ab + 3) // 4, "u": ab * dtype_u_byte,
                                "totale_vram": ab + ab * dtype_u_byte + out["base_byte"]}
    return out


def flop_gradiente(R: int, C: int, NG: int, r: int) -> int:
    """FLOP del gancio per token instradato: 2 r (2C + 2 R NG) (P, dL/dB, Q, dL/dA)."""
    return 2 * r * (2 * C + 2 * R * NG)


if __name__ == "__main__":
    import json
    import sys
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else S.GGUF_THOR_T
    print(json.dumps(memoria_thor(p), indent=1))

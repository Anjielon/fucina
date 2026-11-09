#!/usr/bin/env python3
"""e2e_onpolicy — stadio di DISTILLAZIONE ON-POLICY (OPD) dopo E2E, obiettivo C (12/10/2026). File NUOVO.

## Braccio FEDELE: arXiv 2609.26708 (Chen, Liu, Wang et al., «Train Where the Quantized Model Goes: On-Policy
Distillation for Low-Bit Reasoning»), letto il 12/10 (abs + html v1). Cosa si prende:
  * partenza: il checkpoint QAD (per noi: l'export scelto di E2E-C2, la nostra QAD teacher-forced);
  * lo studente GENERA attraverso il cammino quantizzato di DEPLOYMENT (per noi: il GGUF sul motore thor-c0t con
    GGML_VK_TQ1R_Q8=1 = attivazioni int8 come nel paper, KV 16 bit come nel paper);
  * maestro CONGELATO a precisione piena che segna i prefissi DELLO STUDENTE (per noi il Q8_0 di Ornith-1.5 con la
    testa bf16 originale: il maestro di tutta la catena);
  * perdita (Eq. 3): L = -E[sum_t A(x, y) log pi(y_t|.)] + beta E[sum_t log pi(y_t|.) / pi_T(y_t|.)], A = vantaggio
    GRPO (DeepSeekMath) della ricompensa del verificatore (risposta finale giusta) fra le G completioni dello stesso
    prompt, KL INVERSA stimata sui token campionati dallo studente, beta = 1;
  * temperatura 1, non-thinking; matematica GSM8K (+ MATH L1-3, DAPO-Math: non in casa), G = 4, 8 prompt a passo;
  * LR 3e-6 contro 2e-5 della QAD (rapporto 0,15); checkpoint ogni 20 passi scelto sulla validazione della fase.
  Stimatore (--opd-kl campionata, default): dL/dz_t = (A_t - beta d_t)(p_S - e_y), d_t = log p_S(y_t) - log p_T(y_t)
  senza gradiente, per token (il paper la scrive per token: nessuna somma sul futuro).

## Varianti DICHIARATE (interruttori, default spenti) — da RICERCA_LEVE_NUOVE_1012.md
  * --opd-kl piena: KL(p_S || p_T) ESATTA a vocabolario intero sui prefissi dello studente (stesso valore atteso del
    termine campionato, varianza nulla sul token). Variante Thor.
  * --opd-kl eakld: EdgeRazor arXiv 2605.04062 Eq. 6, L = lambda KL(T||S) + (1 - lambda) KL(S||T), lambda = media sui
    token-risposta di min(H_T, log K) / log K (entropia del MAESTRO), K = 16 (--opd-eakld-k). Il paper media sul
    lotto; qui sul MICRO-lotto (2 righe ~ 15-30 completioni): dichiarato.
  * --opd-pivot "H=4,gamma=0.9,fasi=4,m=4,sep=2,ancora=0.1": PIVOT-Q arXiv 2610.05745. Per completione (episodio):
    q_t = discrepanza per stato, r_t = sum_{h=1..H} gamma^(h-1) q_{t+h} / sum gamma^(h-1); 4 fasi contigue, in ogni
    fase ranghi percentili, P = 0,5 q~ + 0,5 r~, i primi M per P con distanza minima `sep` (il paper non da' il valore:
    2 dichiarato); pesi w = clip(P / media P dei scelti, 0,5, 2) rinormalizzati a media 1; la KL del maestro SOLO sugli
    stati scelti; ANCORA: lambda = 0,1 x media della KL(T||S) su stati PULITI dei dati originali (per noi: soluzioni
    d'oro di GSM8K impacchettate come segmenti «ancora», 4 per passo dell'ottimizzatore come il paper). Varianti Thor:
    q_t = KL(p_S||p_T) piena (il paper: distanza L2 delle azioni continue); il termine del vantaggio resta su TUTTE le
    posizioni generate (PIVOT non ha verificatore); il maestro si calcola comunque su tutte le posizioni (il dump
    costa ~2 % del giro: il risparmio del 92,6 % del paper e' sulla perdita, non sul calcolo).
  * media dei pesi (arXiv 2610.05329): la fa gia' la macchina (--ema-theta, istantanea passo_N/ema): e' un interruttore
    del DRIVER (esporta passo_N/ema invece di passo_N), non di questo modulo.

Deviazioni comuni dichiarate: parametrizzazione = theta (scale) + codici del passo V di TT2, non pesi master bf16
con STE; completioni rinnovate per GIRO (N passi dell'ottimizzatore), non a ogni passo; sonda del passo V e held-out
= perdita uniforme della corsa (CAKLD gamma 1 = KL inversa su tutti i token delle righe, prompt compresi).
Nella macchina di sempre (`KLFusa._pesata`) i pesi della KL piena del percorso base sono TUTTI 0 con l'OPD: quella
perdita resta solo diagnostica (registro), tutto il gradiente OPD viene da `termine_opd`.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

VERSIONE = "e2e_onpolicy 2 (12/10/2026)"
DOMINIO = "opd_gsm8k"
MODI_KL = ("campionata", "piena", "eakld")
PIVOT_DEFAULT = {"H": 4, "gamma": 0.9, "fasi": 4, "m": 4, "sep": 2, "ancora": 0.1}


def pivot_da_testo(testo: str) -> Optional[dict]:
    """"" -> None (spento); "H=4,gamma=0.9,..." -> dizionario completo (chiavi mancanti = default del paper)."""
    if not testo:
        return None
    out = dict(PIVOT_DEFAULT)
    for voce in (x.strip() for x in testo.split(",") if x.strip()):
        k, _, v = voce.partition("=")
        if k not in out:
            raise ValueError(f"--opd-pivot: chiave {k!r} (note: {sorted(out)})")
        out[k] = type(PIVOT_DEFAULT[k])(float(v)) if isinstance(PIVOT_DEFAULT[k], int) else float(v)
    if out["H"] < 1 or out["fasi"] < 1 or out["m"] < 1 or out["sep"] < 1 or not 0 < out["gamma"] <= 1 \
            or out["ancora"] < 0:
        raise ValueError(f"--opd-pivot fuori dominio: {out}")
    return out


# ═══════════════════════════════════════ verificatore e vantaggi ═══════════════════════════════════════
_NUM = r"-?\d[\d,]*(?:\.\d+)?"


def numero(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", "").rstrip("."))
    except ValueError:
        return None


def oro_gsm8k(answer: str) -> float:
    """Risposta d'oro di GSM8K: il numero dopo '####' (virgole delle migliaia tolte)."""
    m = re.search(r"####\s*(" + _NUM + ")", answer)
    if not m:
        raise ValueError(f"risposta GSM8K senza ####: {answer[-80:]!r}")
    v = numero(m.group(1))
    if v is None:
        raise ValueError(f"oro illeggibile {m.group(1)!r}")
    return v


def estrai_risposta(testo: str) -> Optional[float]:
    """Risposta finale: l'ULTIMO '#### <numero>'; se manca, l'ultimo numero del testo (lm-eval 'flexible-extract')."""
    m = re.findall(r"####\s*\$?\s*(" + _NUM + ")", testo)
    if not m:
        m = re.findall(_NUM, testo)
    if not m:
        return None
    return numero(m[-1])


def ricompensa(testo: str, oro: float) -> float:
    """Ricompensa del verificatore (paper: correttezza della risposta finale): 1 o 0."""
    v = estrai_risposta(testo)
    return 1.0 if v is not None and abs(v - oro) <= 1e-6 * max(1.0, abs(oro)) else 0.0


def vantaggi_grpo(r: list) -> list:
    """Vantaggio di gruppo (DeepSeekMath, supervisione dell'esito): (r_i - media) / std di popolazione del gruppo;
    gruppo a ricompensa costante -> 0 (resta la sola KL)."""
    a = np.asarray(r, dtype=np.float64)
    s = a.std()
    if not np.isfinite(s) or s < 1e-12:
        return [0.0] * len(r)
    return ((a - a.mean()) / s).tolist()


# ═══════════════════════════════════════ impacchettamento nel formato corpus ═══════════════════════════════════════
def impacchetta(seqs: list, L: int, id_pad: int, righe_max: Optional[int] = None) -> dict:
    """seqs: [{"prompt": [id], "gen": [id], "adv": float, "ancora": bool (facoltativo)}] in ordine. Riempimento
    (prima riga con posto, altrimenti una nuova) in righe da L token; ogni sequenza = UN segmento; il resto della riga =
    un segmento di riempimento (id_pad, nessuna posizione). Per ogni riga: pos (t: il logit in t predice il token di
    `gen` in t + 1), adv, seg (indice della sequenza in seqs), anc (1 = segmento ancora: soluzione d'oro). Con
    righe_max le righe oltre si scartano INTERE."""
    righe, usate, scartate = [], [], []
    for k, s in enumerate(seqs):
        n = len(s["prompt"]) + len(s["gen"])
        if len(s["prompt"]) < 1 or len(s["gen"]) < 1 or n > L:
            scartate.append(k)
            continue
        dove = next((r for r in righe if r["n"] + n <= L), None)
        if dove is None:
            dove = {"n": 0, "seg": []}
            righe.append(dove)
        dove["seg"].append(k)
        dove["n"] += n
    if righe_max is not None and len(righe) > righe_max:
        for r in righe[righe_max:]:
            scartate.extend(r["seg"])
        righe = righe[:righe_max]
    ids = np.full((len(righe), L), id_pad, dtype=np.int32)
    cu, pos, adv, seg, anc = [], [], [], [], []
    for i, r in enumerate(righe):
        c, p, a, sg, an, o = [0], [], [], [], [], 0
        for k in r["seg"]:
            s = seqs[k]
            P, G = len(s["prompt"]), len(s["gen"])
            ids[i, o:o + P] = s["prompt"]
            ids[i, o + P:o + P + G] = s["gen"]
            p.append(np.arange(o + P - 1, o + P + G - 1, dtype=np.int64))
            a.append(np.full(G, 0.0 if s.get("ancora") else float(s["adv"]), dtype=np.float32))
            sg.append(np.full(G, k, dtype=np.int64))
            an.append(np.full(G, 1 if s.get("ancora") else 0, dtype=np.int8))
            o += P + G
            c.append(o)
            usate.append(k)
        if o < L:
            c.append(L)
        cu.append(np.asarray(c, dtype=np.int64))
        cat = lambda xs, dt: np.concatenate(xs) if xs else np.zeros(0, dt)  # noqa: E731
        pos.append(cat(p, np.int64)); adv.append(cat(a, np.float32))
        seg.append(cat(sg, np.int64)); anc.append(cat(an, np.int8))
    return {"ids": ids, "cu": cu, "pos": pos, "adv": adv, "seg": seg, "anc": anc, "usate": usate,
            "scartate": sorted(scartate)}


def scrivi_lato(cartella: Path, lato: str, variante: str, pk: dict, meta: dict) -> dict:
    """Scrive il lato nel formato di `e2e_teacher.righe_lato` (`e2e_corpus_4096_<v>.pt` / `e2e_heldout_4096_<v>.pt`:
    l_seq, nomi_domini, domini {dom: int32 [N, L]}, cu {dom: concat}, cu_ptr {dom: [N+1]}, ordine [(0, r)]) e accanto
    `onpolicy_<lato>.pt` (pos [n, 2] = (riga, t), adv [n], seg [n], anc [n], sha256 del corpus, meta)."""
    cartella.mkdir(parents=True, exist_ok=True)
    N, L = pk["ids"].shape
    if N == 0:
        raise ValueError(f"lato {lato}: nessuna riga")
    cu_cat = np.concatenate(pk["cu"])
    ptr = np.zeros(N + 1, dtype=np.int64)
    ptr[1:] = np.cumsum([len(c) for c in pk["cu"]])
    d = {"l_seq": L, "nomi_domini": [DOMINIO], "domini": {DOMINIO: torch.from_numpy(pk["ids"].copy())},
         "cu": {DOMINIO: torch.from_numpy(cu_cat)}, "cu_ptr": {DOMINIO: torch.from_numpy(ptr)},
         "ordine": torch.tensor([[0, r] for r in range(N)], dtype=torch.int64)}
    nome = f"e2e_corpus_4096_{variante}.pt" if lato == "train" else f"e2e_heldout_4096_{variante}.pt"
    f = cartella / nome
    torch.save(d, f)
    sha = hashlib.sha256(f.read_bytes()).hexdigest()
    pos = np.concatenate([np.stack([np.full(len(p), r, np.int64), p], 1) for r, p in enumerate(pk["pos"])])
    g = cartella / f"onpolicy_{lato}.pt"
    torch.save({"pos": torch.from_numpy(pos), "adv": torch.from_numpy(np.concatenate(pk["adv"])),
                "seg": torch.from_numpy(np.concatenate(pk["seg"])), "anc": torch.from_numpy(np.concatenate(pk["anc"])),
                "corpus_sha256": sha, "corpus_file": nome, "versione": VERSIONE, "meta": json.dumps(meta)}, g)
    return {"corpus": str(f), "corpus_sha256": sha, "onpolicy": str(g), "righe": N, "posizioni": int(len(pos))}


# ═══════════════════════════════════════ sorgente del ciclo ═══════════════════════════════════════
class SorgenteOnPolicy:
    """Sorgente con la STESSA interfaccia di `e2e_allena.SorgenteCache` (tutto delegato) piu' `generati(lato, i)` ->
    (t int64 [k], adv float32 [k], seg int64 [k], anc int8 [k]). `dati` = {lato: {riga: (t, adv, seg, anc)}}."""

    def __init__(self, base, dati: dict, firma_op: Optional[dict] = None):
        self.base = base
        self.dati = dati
        self._firma_op = firma_op or {}

    def __getattr__(self, nome):                         # piano, dir, modo, ids, h, inizi, dominio, ... del base
        if nome in ("base", "dati", "_firma_op"):
            raise AttributeError(nome)
        return getattr(self.base, nome)

    def firma(self) -> dict:
        f = dict(self.base.firma())
        f["onpolicy"] = self._firma_op
        return f

    def generati(self, lato: str, i: int) -> tuple:
        v = self.dati.get(lato, {}).get(i)
        if v is None:
            return np.zeros(0, np.int64), np.zeros(0, np.float32), np.zeros(0, np.int64), np.zeros(0, np.int8)
        return v


def leggi_onpolicy(p: Path, corpus_file: Path, L: int, n_righe: int, inizi_di: Callable) -> tuple:
    """Legge e VERIFICA `onpolicy_<lato>.pt`: corpus citato == quello del piano (sha256), 0 <= t < L - 1, righe nel
    lato, coppie uniche, vantaggi finiti, e il token t + 1 NON e' l'inizio di un segmento (nessuna posizione di
    confine fra due sequenze)."""
    d = torch.load(p, map_location="cpu", weights_only=True)
    sha = hashlib.sha256(Path(corpus_file).read_bytes()).hexdigest()
    if d["corpus_sha256"] != sha:
        raise ValueError(f"{p}: corpus {d['corpus_sha256'][:12]} diverso da {Path(corpus_file).name} {sha[:12]}")
    pos, adv, seg, anc = d["pos"].long(), d["adv"].float(), d["seg"].long(), d["anc"].to(torch.int8)
    n = pos.shape[0]
    if pos.ndim != 2 or pos.shape[1] != 2 or adv.shape != (n,) or seg.shape != (n,) or anc.shape != (n,):
        raise ValueError(f"{p}: forme pos {tuple(pos.shape)} adv {tuple(adv.shape)} seg {tuple(seg.shape)}")
    if n == 0:
        raise ValueError(f"{p}: nessuna posizione generata")
    if bool(((pos[:, 1] < 0) | (pos[:, 1] >= L - 1)).any()) or bool(((pos[:, 0] < 0) | (pos[:, 0] >= n_righe)).any()):
        raise ValueError(f"{p}: posizioni fuori dalla riga o dal lato")
    if not bool(torch.isfinite(adv).all()):
        raise ValueError(f"{p}: vantaggi non finiti")
    if len(set(map(tuple, pos.tolist()))) != n:
        raise ValueError(f"{p}: posizioni ripetute")
    out: dict = {}
    for r in sorted(set(pos[:, 0].tolist())):
        sel = (pos[:, 0] == r).nonzero().squeeze(1)
        o = torch.argsort(pos[sel, 1])
        sel = sel[o]
        t = pos[sel, 1].numpy().copy()
        iz = inizi_di(r)
        if iz is not None and bool(np.asarray(iz)[t + 1].any()):
            raise ValueError(f"{p}: riga {r}: una posizione generata predice l'inizio di un segmento")
        out[int(r)] = (t, adv[sel].numpy().copy(), seg[sel].numpy().copy(), anc[sel].numpy().copy())
    return out, {"file": str(p), "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest(), "n": int(n),
                 "ancora": int(anc.sum()), "adv_media": float(adv.mean()),
                 "adv_nonzero": float((adv != 0).float().mean())}


def apri_sorgente_onpolicy(base, log: Callable = print) -> SorgenteOnPolicy:
    """Dal piano del teacher: per ogni lato `onpolicy_<lato>.pt` accanto al file del corpus. Il teacher DEVE essere
    in modo «segmenti» (ogni completione e' una sequenza a se', come lo studente la genera)."""
    if base.modo != "segmenti":
        raise ValueError(f"teacher in modo {base.modo!r}: l'OPD vuole «segmenti» (una completione = una sequenza)")
    dati, firma = {}, {}
    for lato, c in base.piano["corpus"].items():
        f = Path(c["file"])
        p = f.parent / f"onpolicy_{lato}.pt"
        if not p.exists():
            raise FileNotFoundError(f"{p}: posizioni generate del lato {lato} assenti")
        righe = base.righe_previste(lato)
        if len(righe) != int(c["n_righe"]):
            raise ValueError(f"{lato}: {len(righe)} righe nel piano, {c['n_righe']} nel corpus")
        dati[lato], firma[lato] = leggi_onpolicy(p, f, int(c["L"]), int(c["n_righe"]),
                                                 lambda r, lato=lato: base.inizi(lato, r))
        log(f"OPD {lato}: {firma[lato]['n']} posizioni ({firma[lato]['ancora']} ancora) su {len(righe)} righe, "
            f"vantaggio medio {firma[lato]['adv_media']:+.4f}, non nullo {100 * firma[lato]['adv_nonzero']:.1f}%")
    return SorgenteOnPolicy(base, dati, firma)


# ═══════════════════════════════════════ pesi del micro-lotto ═══════════════════════════════════════
def pesi_generati(src, lato: str, m: list, ids: torch.Tensor, ca, device) -> tuple:
    """Per il micro-lotto m (B righe da Tn token, appiattite b * Tn + t come hS): (peso, opd). peso = ("risposta", w,
    marcati) per il percorso pesato di `KLFusa` con w = 0 OVUNQUE (la KL piena di base resta diagnostica) e marcati =
    posizioni generate; opd = tutto cio' che serve a `termine_opd`."""
    B, Tn = ids.shape
    cols = [[], [], [], []]
    for b, i in enumerate(m):
        t, a, s, n = src.generati(lato, i)
        cols[0].append(torch.as_tensor(t, dtype=torch.int64) + b * Tn)
        cols[1].append(torch.as_tensor(a, dtype=torch.float64))
        cols[2].append(torch.as_tensor(s, dtype=torch.int64))
        cols[3].append(torch.as_tensor(n, dtype=torch.bool))
    pos, adv, seg, anc = (torch.cat(c).to(device) for c in cols)
    y = ids.reshape(-1).to(device)[pos + 1]
    marcati = torch.zeros(B * Tn, dtype=torch.bool, device=device)
    marcati[pos] = True
    w = torch.zeros(B * Tn, dtype=torch.float64, device=device)
    return ("risposta", w, marcati), {"pos": pos, "y": y, "adv": adv, "seg": seg, "anc": anc,
                                      "beta": float(ca.opd_beta), "modo": ca.opd_kl, "eakld_k": int(ca.opd_eakld_k),
                                      "pivot": pivot_da_testo(ca.opd_pivot)}


# ═══════════════════════════════════════ PIVOT-Q: scelta degli stati vulnerabili ═══════════════════════════════════════
def _ranghi(x: np.ndarray) -> np.ndarray:
    """Ranghi percentili in [0, 1] (media dei pari); un solo elemento -> 1."""
    n = len(x)
    if n == 1:
        return np.ones(1)
    o = np.argsort(x, kind="stable")
    r = np.empty(n)
    r[o] = np.arange(n, dtype=np.float64)
    for v in np.unique(x):                                # pari: rango medio
        k = x == v
        if k.sum() > 1:
            r[k] = r[k].mean()
    return r / (n - 1)


def scegli_vulnerabili(q: np.ndarray, seg: np.ndarray, cfg: dict) -> np.ndarray:
    """PIVOT-Q (arXiv 2610.05745 §3) sulle posizioni GENERATE (ordinate per riga e t, seg = completione). Ritorna i
    pesi w (0 = non scelto) gia' rinormalizzati a media 1 sui scelti di ogni completione."""
    w = np.zeros(len(q))
    H, g = int(cfg["H"]), float(cfg["gamma"])
    for s in np.unique(seg):
        idx = np.nonzero(seg == s)[0]
        qs = q[idx]
        n = len(qs)
        r = np.zeros(n)
        for t in range(n):                               # sconto sui prossimi H (troncato alla fine, rinormalizzato)
            fut = qs[t + 1:t + 1 + H]
            if len(fut):
                ww = g ** np.arange(len(fut))
                r[t] = (ww * fut).sum() / ww.sum()
            else:
                r[t] = qs[t]
        scelti, P = [], np.zeros(n)
        for fase in np.array_split(np.arange(n), min(int(cfg["fasi"]), n)):
            if not len(fase):
                continue
            P[fase] = 0.5 * _ranghi(qs[fase]) + 0.5 * _ranghi(r[fase])
            presi = []
            for j in fase[np.argsort(-P[fase], kind="stable")]:
                if all(abs(j - k) >= int(cfg["sep"]) for k in presi):
                    presi.append(j)
                if len(presi) == int(cfg["m"]):
                    break
            scelti.extend(presi)
        scelti = np.asarray(sorted(scelti))
        mP = max(P[scelti].mean(), 1e-12)
        ws = np.clip(P[scelti] / mP, 0.5, 2.0)
        w[idx[scelti]] = ws / ws.mean()
    return w


# ═══════════════════════════════════════ termine fuso ═══════════════════════════════════════
def _lp(logit_s, logit_t, hS, WS, hT, WT, p):
    lS = logit_s(hS[p], WS).float()
    lS = lS - torch.logsumexp(lS, -1, keepdim=True)
    lT = logit_t(hT[p], WT, lS.dtype).float()
    lT = lT - torch.logsumexp(lT, -1, keepdim=True)
    return lS, lT


def termine_opd(hS, WS, hT, WT, opd: dict, pezzo: int, dh: torch.Tensor, logit_s: Callable, logit_t: Callable,
                per_testa: Callable, info: Optional[dict] = None) -> torch.Tensor:
    """Termine OPD alle posizioni del micro-lotto, a pezzi, gradiente nel passaggio in avanti (come `KLFusa`): somma a
    dh[pos] (dL/dz) @ WS e ritorna il valore (fp64). Per posizione generata (anc = 0), con w_t = peso PIVOT (1 se
    spento), A_t vantaggio:
      vantaggio (sempre):  -A_t log p_S(y_t)                         dz = A_t (p_S - e_y)
      campionata:          beta w_t d_t (funzione di punteggio)       dz += -beta w_t d_t (p_S - e_y)
      piena:               beta w_t KL(S||T)                         dz += beta w_t p_S (lS - lT - KL(S||T))
      eakld:               beta w_t [lam KL(T||S) + (1-lam) KL(S||T)]  dz += beta w_t [lam (p_S - p_T) + (1-lam) (...)]
    Posizioni ancora (anc = 1, solo con PIVOT): ancora / n_ancora x KL(T||S), dz = (ancora / n_ancora)(p_S - p_T).
    lam (eakld) = media sulle posizioni generate del micro-lotto di min(H_T, log K) / log K (senza gradiente)."""
    pos, y, adv, beta, modo = opd["pos"], opd["y"], opd["adv"], float(opd["beta"]), opd["modo"]
    if modo not in MODI_KL:
        raise ValueError(f"opd modo {modo!r}")
    dev = hS.device
    anc = opd.get("anc")
    anc = torch.zeros_like(pos, dtype=torch.bool) if anc is None else anc.bool()
    pv = opd.get("pivot")
    if anc.any() and pv is None:
        raise ValueError("segmenti ancora senza --opd-pivot")
    gen = ~anc
    w = torch.ones(pos.numel(), dtype=torch.float64, device=dev)
    lam = None
    K = int(opd.get("eakld_k", 16))
    if pv is not None or modo == "eakld":                # passo 1 (senza gradiente): discrepanza e/o entropia del maestro
        q = torch.zeros(pos.numel(), dtype=torch.float64, device=dev)
        hT_ = torch.zeros(pos.numel(), dtype=torch.float64, device=dev)
        with torch.no_grad():
            for i in range(0, pos.numel(), pezzo):
                lS, lT = _lp(logit_s, logit_t, hS, WS, hT, WT, pos[i:i + pezzo])
                pS = lS.exp()
                q[i:i + pezzo] = (pS * (lS - lT)).sum(-1, dtype=torch.float64)
                hT_[i:i + pezzo] = -(lT.exp() * lT).sum(-1, dtype=torch.float64)
                del lS, lT, pS
        if modo == "eakld":
            hg = hT_[gen].clamp(max=math.log(K)) / math.log(K)
            lam = float(hg.mean()) if hg.numel() else 1.0
        if pv is not None:
            wg = scegli_vulnerabili(q[gen].cpu().numpy(), opd["seg"][gen].cpu().numpy(), pv)
            w[gen] = torch.from_numpy(wg).to(dev)
    n_anc = int(anc.sum())
    wa = (float(pv["ancora"]) / n_anc) if (pv is not None and n_anc) else 0.0
    tot = torch.zeros((), dtype=torch.float64, device=dev)
    sd = torch.zeros((), dtype=torch.float64, device=dev)
    skl = torch.zeros((), dtype=torch.float64, device=dev)
    for i in range(0, pos.numel(), pezzo):
        p, yy, a, ww, an = pos[i:i + pezzo], y[i:i + pezzo], adv[i:i + pezzo], w[i:i + pezzo], anc[i:i + pezzo]
        lS, lT = _lp(logit_s, logit_t, hS, WS, hT, WT, p)
        r = torch.arange(p.numel(), device=dev)
        lSy = lS[r, yy].double()
        d = lSy - lT[r, yy].double()
        g_ = (~an).double()                               # 1 sulle generate
        pS = lS.exp()
        N = lS - lT                                       # log p_S - log p_T
        kl_inv = (pS * N).sum(-1, dtype=torch.float64)
        pT = lT.exp()
        kl_fwd = -(pT * N).sum(-1, dtype=torch.float64)
        # gradiente rispetto ai logit dello studente
        oh = torch.zeros_like(pS)
        oh[r, yy] = 1.0
        dz = (pS - oh) * (a * g_).to(pS.dtype).unsqueeze(-1)                       # vantaggio
        val = (-(a * g_) * lSy).sum()
        cb = (beta * ww * g_)
        if modo == "campionata":
            dz -= (pS - oh) * (cb * d).to(pS.dtype).unsqueeze(-1)
            val = val + (cb * d).sum()
        else:
            l_ = 0.0 if modo == "piena" else lam
            g_inv = pS * (N - kl_inv.to(N.dtype).unsqueeze(-1))
            dz += ((1.0 - l_) * g_inv + l_ * (pS - pT)) * cb.to(pS.dtype).unsqueeze(-1)
            val = val + (cb * ((1.0 - l_) * kl_inv + l_ * kl_fwd)).sum()
        if wa:
            ca_ = an.double() * wa
            dz += (pS - pT) * ca_.to(pS.dtype).unsqueeze(-1)
            val = val + (ca_ * kl_fwd).sum()
        tot += val
        sd += (d * g_).sum()
        skl += (kl_inv * g_).sum()
        dh.index_add_(0, p, per_testa(dz, WS).to(dh.dtype))
        del lS, lT, pS, pT, N, oh, dz
    if info is not None:
        ng = int(gen.sum())
        nf = torch.tensor(float(ng), dtype=torch.float64, device=dev)
        info.update({"opd_n": torch.tensor(ng, device=dev), "opd_d": sd, "opd_kl_inv": skl, "opd_valore": tot,
                     "opd_scelti": (w[gen] > 0).sum() if pv is not None else torch.tensor(ng, device=dev),
                     "opd_lambda": torch.tensor(lam if lam is not None else float("nan"), dtype=torch.float64),
                     # i pesi della KL piena di base sono 0: le statistiche si riportano alle posizioni generate
                     "somma_w": nf, "somma_w2": nf.clamp_min(1.0),
                     "max_w": torch.ones((), dtype=torch.float64, device=dev)})
        info["_pesi_opd"] = w                               # per le prove (riferimento autograd)
        info["_lambda_opd"] = lam
    return tot


def termine_opd_riferimento(hS, WS, hT, WT, opd: dict, w: torch.Tensor, lam: Optional[float]) -> torch.Tensor:
    """Riferimento AUTOGRAD (solo prove), dati i pesi PIVOT w e lam gia' decisi: funzione surrogata con lo stesso
    gradiente (per «campionata» la funzione di punteggio del paper)."""
    pos, y, adv, beta, modo = opd["pos"], opd["y"], opd["adv"], float(opd["beta"]), opd["modo"]
    anc = opd.get("anc")
    anc = torch.zeros_like(pos, dtype=torch.bool) if anc is None else anc.bool()
    g_ = (~anc).to(hS.dtype)
    lS = torch.log_softmax(hS[pos] @ WS.T, -1)
    lT = torch.log_softmax(hT[pos].to(lS.dtype) @ WT.T.to(lS.dtype), -1)
    r = torch.arange(pos.numel())
    lSy, lTy = lS[r, y], lT[r, y]
    L = -(adv.to(lS.dtype) * g_ * lSy).sum()
    cb = beta * w.to(lS.dtype) * g_
    kl_inv = (lS.exp() * (lS - lT)).sum(-1)
    kl_fwd = (lT.exp() * (lT - lS)).sum(-1)
    if modo == "campionata":
        L = L + (cb * (lSy - lTy).detach() * lSy).sum()
    elif modo == "piena":
        L = L + (cb * kl_inv).sum()
    else:
        L = L + (cb * ((1 - lam) * kl_inv + lam * kl_fwd)).sum()
    pv = opd.get("pivot")
    n_anc = int(anc.sum())
    if pv is not None and n_anc:
        L = L + (anc.to(lS.dtype) * kl_fwd).sum() * float(pv["ancora"]) / n_anc
    return L

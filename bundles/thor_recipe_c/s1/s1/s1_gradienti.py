#!/usr/bin/env python3
"""S1 tappa 4 (GPU): gradienti della KL di C e fattori di Kronecker per l'obiettivo a DUE LATI (K1 del piano), su righe
di calibrazione e su righe di CONTROLLO disgiunte.

Riuso dichiarato: e' raffina/gradienti_densi.py (G2PTQ 2609.31009 · YAQA 2505.22988 · FastKron 2608.06291) portato
sullo studente di C (e2e_studente ac482f3c, densi ternari «tutte», E2E_DENSI_SSM_AB=1), con il raccoglitore riscritto
qui (niente dipendenza da rd_ternary del ramo r023) e due aggiunte:
  (1) FISHER DI CAMMINO (F1 di MATEMATICA_FUSA_1013/T1, variante Thor dichiarata): per il Teorema 1 la KL esatta e'
      KL(p_T||p_S) = int_0^1 (1 - tau) dz^T Omega(z_T + tau dz) dz dtau. Si campiona per token tau ~ w(tau) normalizzata
      e xi = c (sqrt(pi) * eps - pi <sqrt(pi), eps>), eps ~ N(0, I), pi = softmax(z(tau)), c^2 = int w: allora
      E[xi xi^T] = int w(tau) Omega(z(tau)) dtau (stima senza distorsione, una retropropagazione in piu' per riga).
      B_cam = E[g g^T] con g = J^T xi all'uscita di ogni denso. w: «diretta» = 1 - tau (fedele alla KL che si minimizza,
      default), «inversa» = tau, «centro» = 3 tau (1 - tau) (simmetrica, premia dove la decisione si ribalta).
  (2) RIGHE DI CONTROLLO disgiunte (insieme «ctl»): stesso calcolo su altre 32 righe, per verificare che B (e A, g)
      non siano rumore di campione (primo rischio della linea FRONTIERA §3.3).
Il bersaglio e' la KL a testo pieno sulla cache del maestro q8 WikiText-103 (corpus ricco di conoscenza, 0 13-grammi MMLU),
NON la KL4 sulle domande SELRISP: quelle sono la held-out del cancello di accettazione e non si toccano.
Uscite (base MEMORIZZATA, come gradienti_densi.py): g_<t>.npy (media sulle righe della KL media per token; fp32 in cal,
fp16 in ctl), A_<t>.npy = E[x x^T], B_<t>.npy = E[g g^T] (Fisher empirica del residuo, la B di oggi), Bcam_<t>.npy
(Fisher di cammino), tutte con diagonale media 1 e la scala in scale_kron.tsv; kl_righe.tsv; fatto.txt.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


PESI_CAMMINO = {"diretta": 0.5, "inversa": 0.5, "centro": 0.5}       # int_0^1 w (w normalizzati a 1/2)


def campiona_tau(n, modo, gen, dev):
    u = torch.rand(n, 1, generator=gen, device=dev, dtype=torch.float64)
    if modo == "diretta":                      # densita' 2 (1 - tau): Beta(1, 2)
        return 1.0 - torch.sqrt(u)
    if modo == "inversa":                      # Beta(2, 1)
        return torch.sqrt(u)
    if modo == "centro":                       # Beta(2, 2) per inversione numerica (densita' 6 tau (1 - tau))
        t = u.clone()
        for _ in range(40):                    # Newton su F(t) = 3t^2 - 2t^3 = u
            f = 3 * t * t - 2 * t ** 3 - u
            d = (6 * t * (1 - t)).clamp_min(1e-12)
            t = (t - f / d).clamp(0.0, 1.0)
        return t
    raise ValueError(modo)


def xi_da_logit(lT, lS, modo, gen):
    """lT, lS (P, V) logit (anche non normalizzati). Ritorna xi (P, V) fp32 con E[xi xi^T] = int w Omega(z(tau))."""
    tau = campiona_tau(lT.shape[0], modo, gen, lT.device).to(lT.dtype)
    z = lS.sub_(lT).mul_(tau).add_(lT)                                  # z(tau) = z_T + tau (z_S - z_T), in lS
    z.sub_(torch.logsumexp(z, -1, keepdim=True)).exp_()                # pi
    pi = z
    eps = torch.randn(pi.shape, generator=gen, device=pi.device, dtype=pi.dtype)
    eps.mul_(pi.sqrt())                                                 # sqrt(pi) * eps
    s = eps.sum(-1, keepdim=True)
    eps.sub_(pi * s).mul_(math.sqrt(PESI_CAMMINO[modo]))
    return eps


class Raccoglitore:
    """Per (strato, famiglia densa di primo piano): G = sum g^T x, A = sum x^T x, B = sum g^T g (modo «kl»); Bcam = sum
    g^T g (modo «cam»). Base memorizzata (gli argomenti di mot.dd)."""

    def __init__(self):
        self.modo = "kl"
        self.G, self.A, self.B, self.Bc, self.n, self.nc = {}, {}, {}, {}, {}, {}
        self.righe = 0

    @staticmethod
    def vuole(fam) -> bool:
        return fam.E == 1 and not fam.nome[-1:].isdigit()

    def aggiungi(self, L, nome, g, x):
        k = (L, nome)
        if self.modo == "kl":
            if k not in self.G:
                self.G[k] = torch.zeros(g.shape[1], x.shape[1], dtype=torch.float32, device=g.device)
                self.A[k] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=g.device)
                self.B[k] = torch.zeros(g.shape[1], g.shape[1], dtype=torch.float32, device=g.device)
                self.n[k] = 0
            self.G[k].addmm_(g.T, x)
            self.A[k].addmm_(x.T, x)
            self.B[k].addmm_(g.T, g)
            self.n[k] += x.shape[0]
        else:
            if k not in self.Bc:
                self.Bc[k] = torch.zeros(g.shape[1], g.shape[1], dtype=torch.float32, device=g.device)
                self.nc[k] = 0
            self.Bc[k].addmm_(g.T, g)
            self.nc[k] += x.shape[0]


class MotoreS1:
    def __init__(self, interno, rac):
        self._m, self._r = interno, rac

    def __getattr__(self, nome):
        return getattr(self._m, nome)

    def dd(self, gy, x, fam, piano, d_dtype):
        if self._r.vuole(fam):
            self._r.aggiungi(getattr(fam, "strato", -1), fam.nome, gy.float(), x.float())
        return self._m.dd(gy, x, fam, piano, d_dtype)


def taglia(lot, n):
    """Prova: primi n token della riga (ids, hT, inizi), stessa classe Lotto (dataclass di e2e_allena)."""
    if not n:
        return lot
    import dataclasses
    ini = lot.inizi[..., :n] if lot.inizi is not None else None
    return dataclasses.replace(lot, ids=lot.ids[:, :n], hT=lot.hT[:n], inizi=ini)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ospite", required=True)
    ap.add_argument("--corpus-json", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--insieme", choices=("cal", "ctl"), required=True)
    ap.add_argument("--thor-v2", required=True)
    ap.add_argument("--ricetta", default="B3_ind32")
    ap.add_argument("--motore", default="triton")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cammino", choices=tuple(PESI_CAMMINO), default="diretta")
    ap.add_argument("--senza-cammino", action="store_true")
    ap.add_argument("--pezzo", type=int, default=256)
    ap.add_argument("--seme", type=int, default=7)
    ap.add_argument("--strati", type=int, default=40, help="prova: primi N strati")
    ap.add_argument("--righe", type=int, default=0, help="prova: prime N righe dell'insieme")
    ap.add_argument("--tronca", type=int, default=0, help="prova: primi N token della riga")
    a = ap.parse_args()
    sys.path.insert(0, a.thor_v2)
    import e2e_allena as A
    import e2e_studente as S
    out = Path(a.out) / a.insieme
    if (out / "fatto.txt").exists():
        log(f"{out}: gia' fatto")
        return 0
    tmp = Path(str(out) + ".tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    try:
        import e2e_densi_piegati as DP
        if getattr(DP, "PIEGA", {}).get("modo", ""):
            DP.PIEGA["modo"] = ""
            log("E2E_DENSI_PIEGATI spento per la raccolta (i densi piegati non passano da mot.dd)")
    except ImportError:
        pass
    if hasattr(A, "scegli_gdn"):
        A.scegli_gdn(log)
    st = S.Studente(gguf=Path(a.ospite), strati=range(a.strati), ricetta=a.ricetta, motore_nome=a.motore,
                    device=str(dev), densi_ternari="tutte", log=log)
    if dev.type == "cuda":
        st.carica_embedding()
    WS = A.testa_studente(st, "fp32")
    WT, sha_t = A.testa_maestro(dev, torch.float32)
    src = A.SorgenteCache(a.cache)
    car = A.Caricatore(src, dev)
    cj = json.load(open(a.corpus_json))
    righe = cj["gradienti"]["righe_cal" if a.insieme == "cal" else "righe_ctl"]
    if a.righe:
        righe = righe[: a.righe]
    rac = Raccoglitore()
    for L in st.strati:
        es = st.esperti(L)
        es.mot = MotoreS1(es.mot, rac)
    st.congela_strati(list(st.strati))
    t0 = time.time()
    kl_righe = []
    for n, i in enumerate(righe):
        lot = taglia(car.prendi("train", [i]), a.tronca)
        T = lot.ids.shape[1]
        stato = st.avanti(lot.ids, inizi=lot.inizi)
        # (a) gradiente della KL (come gradienti_densi.py)
        rac.modo = "kl"
        xL = stato["xL"].detach().requires_grad_(True)
        kl = 0.0
        with torch.enable_grad():
            h = st.esce(xL)
            hv = h.detach()
            dh = torch.zeros_like(hv)
            for j in range(0, hv.shape[0], a.pezzo):
                r = A.kl_pezzo(hv[j:j + a.pezzo], WS, lot.hT[j:j + a.pezzo], WT, True)
                dh[j:j + a.pezzo] = r["dh"] / T
                kl += float(r["kl"].sum())
            h.backward(dh)
        st.indietro(stato, xL.grad)
        rac.righe += 1
        # (b) Fisher di cammino: xi al posto di p_S - p_T, stesso indietro (stato riusato: avanti senza grafo)
        if not a.senza_cammino:
            rac.modo = "cam"
            gen = torch.Generator(device=dev).manual_seed(a.seme * 100003 + int(i))
            xL2 = stato["xL"].detach().requires_grad_(True)
            with torch.enable_grad():
                h2 = st.esce(xL2)
                hv2 = h2.detach()
                dh2 = torch.zeros_like(hv2)
                for j in range(0, hv2.shape[0], a.pezzo):
                    lT = A._logit_maestro(lot.hT[j:j + a.pezzo], WT, torch.float32)
                    lS = A._logit_studente(hv2[j:j + a.pezzo].float(), WS)
                    xi = xi_da_logit(lT, lS, a.cammino, gen)
                    del lT, lS
                    dh2[j:j + a.pezzo] = (A._per_testa(xi, WS) / T).to(dh2.dtype)
                    del xi
                h2.backward(dh2)
            st.indietro(stato, xL2.grad)
            del xL2, h2, dh2
        for p in st.parametri():
            p.grad = None
        kl_righe.append((int(i), T, kl / T))
        del stato, xL, h, dh
        msg = f"{a.insieme} riga {n + 1}/{len(righe)} (cache {i}): KL media {kl / T:.5f}; {(time.time() - t0) / (n + 1):.1f} s/riga"
        if dev.type == "cuda":
            msg += f"; picco {torch.cuda.max_memory_allocated(dev) / 2**30:.1f} GiB"
        log(msg)
    with open(tmp / "scale_kron.tsv", "w") as f:
        f.write("tipo\ttensore\tscala_media_diag\ttoken\n")
        for k in sorted(rac.G):
            L, nome = k
            t = f"blk.{L}.{nome}.weight"
            g = rac.G[k] / rac.righe
            np.save(tmp / f"g_{t}.npy", g.cpu().numpy().astype(np.float32 if a.insieme == "cal" else np.float16))
            for tag, M, nn in (("A", rac.A[k], rac.n[k]), ("B", rac.B[k], rac.n[k]),
                               ("Bcam", rac.Bc.get(k), rac.nc.get(k, 0))):
                if M is None:
                    continue
                M = M / max(nn, 1)
                sc = float(torch.diagonal(M).mean().clamp_min(1e-38))
                np.save(tmp / f"{tag}_{t}.npy", (M / sc).cpu().numpy().astype(np.float32))
                f.write(f"{tag}\t{t}\t{sc:.6e}\t{nn}\n")
    with open(tmp / "kl_righe.tsv", "w") as f:
        f.write("riga_cache\ttoken\tkl_media\n")
        for r in kl_righe:
            f.write("\t".join(str(x) for x in r) + "\n")
    (tmp / "fatto.txt").write_text(json.dumps({
        "righe": len(righe), "insieme": a.insieme, "tensori": len(rac.G), "cache": a.cache, "ospite": a.ospite,
        "cammino": None if a.senza_cammino else a.cammino, "testa_maestro_sha256": sha_t, "strati": a.strati,
        "tronca": a.tronca, "kl_media": float(np.mean([r[2] for r in kl_righe])) if kl_righe else None,
        "minuti": round((time.time() - t0) / 60, 1)}) + "\n")
    if out.exists():
        import shutil
        shutil.rmtree(out)
    os.replace(tmp, out)
    log(f"{a.insieme}: {len(rac.G)} tensori, {len(righe)} righe in {(time.time() - t0) / 60:.1f} min -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

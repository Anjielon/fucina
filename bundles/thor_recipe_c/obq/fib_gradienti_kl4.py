#!/usr/bin/env python3
"""OBQ fibrato, tappa GPU: fattori di Kronecker (K-FAC) della KL4 sulle risposte, SUL FLUSSO DI C, in due META'
DISGIUNTE delle domande di calibrazione (adattamento «cal» / guardia «ctl»), entrambe disgiunte dalle 883 held-out
del cancello.

Riuso dichiarato (nessuna modifica agli originali):
  * l'obiettivo KL4 e l'impacchettamento delle domande sono quelli di raffina/gradienti_densi.py (SELRISP 10/10:
    pacchi_kl4 / passo_kl4: softmax a 4 sulle lettere, maestro q8 dalle righe MCLP, segmenti che ripartono a ogni
    domanda come al banco), copiati qui;
  * il raccoglitore per famiglia densa di primo piano (mot.dd: A = sum x^T x, B = sum g^T g nella base MEMORIZZATA)
    e la costruzione dello studente di C (e2e_studente ac482f3c, densi ternari «tutte», E2E_DENSI_SSM_AB=1) sono quelli
    di sorgenti_forgia/obiettivo_c/s1/s1_gradienti.py, copiati qui (senza la Fisher di cammino: la leva usa la Fisher
    empirica di K-FAC, come V; Kunstner et al. 1905.12558, dichiarato).
Divisione delle domande: filtro sulle righe MCLP del maestro, random.Random(7).shuffle (lo stesso di kl4_heldout.py e di
pacchi_kl4), prime 896 = «usate» (le altre 883 sono la held-out del cancello e qui NON si toccano). Delle 896:
posizioni 0..447 = cal, 448..895 = ctl.
Uscite (OUT/<insieme>/, atomiche via .tmp): A_<t>.npy, B_<t>.npy (media diag 1; cal fp32, ctl fp16 — al caricamento
si riproietta sul cono PSD), scale_kron.tsv, kl4_torch.tsv, fatto.txt (json).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def leggi_mclp(p):
    out = {}
    for r in open(p, errors="replace"):
        i = r.find("MCLP\t")
        if i < 0:
            continue
        c = r[i:].rstrip("\n").split("\t")
        n = int(c[4])
        out[int(c[1])] = (int(c[2]), [float(x) for x in c[5:5 + n]])
    return out


def domande(calib_jsonl, maestro_log, n_usate=896, seme=7):
    T4 = leggi_mclp(maestro_log)
    Q = [json.loads(x) for x in open(calib_jsonl)]
    Q = [q for q in Q if q["i"] in T4]
    for q in Q:
        assert T4[q["i"]][0] == q["label"], ("maestro e calibrazione in disaccordo", q["i"])
    random.Random(seme).shuffle(Q)
    usate = Q[:n_usate]
    for q in usate:
        q["pT"] = torch.softmax(torch.tensor(T4[q["i"]][1], dtype=torch.float64), 0)
    h = n_usate // 2
    return {"cal": usate[:h], "ctl": usate[h:n_usate]}, len(Q) - n_usate


def impacchetta(Q, pacco_tok):
    pacchi, cur, nt = [], [], 0
    for q in Q:
        L = len(q["ids"])
        if cur and nt + L > pacco_tok:
            pacchi.append(cur); cur, nt = [], 0
        cur.append(q); nt += L
    if cur:
        pacchi.append(cur)
    return pacchi


class Raccoglitore:
    def __init__(self):
        self.A, self.B, self.n = {}, {}, {}

    @staticmethod
    def vuole(fam) -> bool:
        return fam.E == 1 and not fam.nome[-1:].isdigit()

    def aggiungi(self, L, nome, g, x):
        k = (L, nome)
        if k not in self.A:
            self.A[k] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=x.device)
            self.B[k] = torch.zeros(g.shape[1], g.shape[1], dtype=torch.float32, device=g.device)
            self.n[k] = 0
        self.A[k].addmm_(x.T, x)
        self.B[k].addmm_(g.T, g)
        self.n[k] += x.shape[0]


class Motore:
    def __init__(self, interno, rac):
        self._m, self._r = interno, rac

    def __getattr__(self, nome):
        return getattr(self._m, nome)

    def dd(self, gy, x, fam, piano, d_dtype):
        if self._r.vuole(fam):
            self._r.aggiungi(getattr(fam, "strato", -1), fam.nome, gy.float(), x.float())
        return self._m.dd(gy, x, fam, piano, d_dtype)


def passo_kl4(st, WS, pacco, dev, righe_out):
    ids = torch.tensor([sum((q["ids"] for q in pacco), [])], dtype=torch.long, device=dev)
    T = ids.shape[1]
    inizi = torch.zeros((1, T), dtype=torch.long)
    ultimi, o = [], 0
    for q in pacco:
        inizi[0, o] = 1
        o += len(q["ids"])
        ultimi.append(o - 1)
    stato = st.avanti(ids, inizi=inizi)
    xL = stato["xL"].detach().requires_grad_(True)
    with torch.enable_grad():
        h = st.esce(xL)
        hh = h.reshape(-1, h.shape[-1])
        dh = torch.zeros_like(hh)
        let = torch.tensor(pacco[0]["lettere"], device=dev)
        Wl = WS[let].to(torch.float32)
        idx = torch.tensor(ultimi, device=dev)
        z = hh[idx].detach().to(torch.float32) @ Wl.T
        lS = torch.log_softmax(z.double(), -1)
        pT = torch.stack([q["pT"] for q in pacco]).to(dev)
        kl = (pT * (torch.log(pT.clamp_min(1e-300)) - lS)).sum(-1)
        g = (lS.exp() - pT) / len(pacco)
        dh[idx] = (g.to(torch.float32) @ Wl).to(dh.dtype)
        hh.backward(dh)
    st.indietro(stato, xL.grad)
    pS = lS.exp()
    for q, k, ps, pt in zip(pacco, kl.tolist(), pS.tolist(), pT.tolist()):
        righe_out.append((q["i"], q["label"], k, ps[q["label"]], pt[q["label"]]))
    del stato, xL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ospite", required=True)
    ap.add_argument("--thor-v2", required=True)
    ap.add_argument("--calib-jsonl", required=True)
    ap.add_argument("--calib-maestro", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--insieme", choices=("cal", "ctl"), required=True)
    ap.add_argument("--n-usate", type=int, default=896)
    ap.add_argument("--pacco", type=int, default=4096)
    ap.add_argument("--ricetta", default="B3_ind32")
    ap.add_argument("--motore", default="triton2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--strati", type=int, default=40, help="prova: primi N strati")
    ap.add_argument("--domande", type=int, default=0, help="prova: prime N domande dell'insieme")
    a = ap.parse_args()
    sys.path.insert(0, a.thor_v2)
    import e2e_allena as A
    import e2e_studente as S
    out = Path(a.out) / a.insieme
    if (out / "fatto.txt").exists():
        log(f"{out}: gia' fatto")
        return 0
    tmp = Path(str(out) + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    insiemi, n_held = domande(a.calib_jsonl, a.calib_maestro, a.n_usate)
    Q = insiemi[a.insieme]
    if a.domande:
        Q = Q[: a.domande]
    pacchi = impacchetta(Q, a.pacco)
    log(f"{a.insieme}: {len(Q)} domande in {len(pacchi)} pacchi da <= {a.pacco} token; held-out intatte {n_held}")
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
    rac = Raccoglitore()
    for L in st.strati:
        es = st.esperti(L)
        es.mot = Motore(es.mot, rac)
    st.congela_strati(list(st.strati))
    t0 = time.time()
    righe = []
    for n, p in enumerate(pacchi):
        passo_kl4(st, WS, p, dev, righe)
        for prm in st.parametri():
            prm.grad = None
        msg = (f"{a.insieme} pacco {n + 1}/{len(pacchi)}: KL4 media {np.mean([r[2] for r in righe]):.5f} su {len(righe)} "
               f"domande; {(time.time() - t0) / (n + 1):.1f} s/pacco; densi {len(rac.A)}")
        if dev.type == "cuda":
            msg += f"; picco {torch.cuda.max_memory_allocated(dev) / 2**30:.1f} GiB"
        log(msg)
    dt = np.float32 if a.insieme == "cal" else np.float16
    with open(tmp / "scale_kron.tsv", "w") as f:
        f.write("tipo\ttensore\tscala_media_diag\ttoken\n")
        for k in sorted(rac.A):
            L, nome = k
            t = f"blk.{L}.{nome}.weight"
            for tag, M in (("A", rac.A[k]), ("B", rac.B[k])):
                M = M / max(rac.n[k], 1)
                sc = float(torch.diagonal(M).mean().clamp_min(1e-38))
                np.save(tmp / f"{tag}_{t}.npy", (M / sc).cpu().numpy().astype(dt))
                f.write(f"{tag}\t{t}\t{sc:.6e}\t{rac.n[k]}\n")
    with open(tmp / "kl4_torch.tsv", "w") as f:
        f.write("indice\tgiusta\tkl4\tpS_giusta\tpT_giusta\n")
        for r in righe:
            f.write("\t".join(str(x) for x in r) + "\n")
    (tmp / "fatto.txt").write_text(json.dumps({
        "insieme": a.insieme, "domande": len(Q), "pacchi": len(pacchi), "indici": [q["i"] for q in Q],
        "tensori": len(rac.A), "ospite": a.ospite, "strati": a.strati, "kl4_media": float(np.mean([r[2] for r in righe])),
        "held_out_intatte": n_held, "minuti": round((time.time() - t0) / 60, 1)}) + "\n")
    if out.exists():
        shutil.rmtree(out)
    os.replace(tmp, out)
    log(f"{a.insieme}: {len(rac.A)} densi, {len(Q)} domande in {(time.time() - t0) / 60:.1f} min -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

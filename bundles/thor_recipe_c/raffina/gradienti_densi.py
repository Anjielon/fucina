#!/usr/bin/env python3
"""G2PTQ (arXiv 2609.31009) — gradiente ESATTO della KL del modello intero (maestro || R2B3m) rispetto ai pesi dei
densi multi-piano, nella base memorizzata (ruotata per TQ1_0R_1024), sullo studente vero (e2e_studente, densi ternari).

Riusa il gancio del raccoglitore dKL (rd_ternary/dkl_raccogli.py del ramo r023: MotoreDKL.dd calcola dW = gyᵀ x)
con un Raccoglitore ridotto ai SOLI densi (niente esperti: 4 buffer per tensore -> 1 accumulatore) e tutti i 40
strati allenabili in una passata (un solo indietro per riga). Righe = `train` della cache del maestro (mai held-out,
mai MMLU). Uscita: OUT/g_<nome>.npy float32 [R, C] = media sulle righe della KL media per token.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

RD = os.environ.get("RD_DKL", "@@ROOT:models@@/gguf/tony-tern/R026/_wt/r023/thor_v2/rd_ternary")
sys.path.insert(0, RD)
import dkl_raccogli as DR  # noqa: E402

ESPERTI = {"gate", "up", "down"}


class MotoreRaff:
    """Come DR.MotoreDKL, ma per i soli densi (piano 1) accumula anche i fattori di Kronecker di YAQA/FastKron:
    A = E[x xᵀ] (ingresso, base memorizzata) e B = E[gy gyᵀ] (gradiente d'uscita della KL), per token."""

    def __init__(self, interno, rac):
        self._m, self._r = interno, rac

    def __getattr__(self, nome):
        return getattr(self._m, nome)

    def dd(self, gy, x, fam, piano, d_dtype):
        L, nome = getattr(fam, "strato", -1), fam.nome
        if self._r.vuole(L, nome) and fam.E == 1:
            g32, x32 = gy.float(), x.float()
            self._r.somma(L, nome, g32.T @ x32)
            if self._r.kron:
                self._r.somma_kron(L, nome, x32, g32)
        return self._m.dd(gy, x, fam, piano, d_dtype)


class RaccoglitoreDensi(DR.Raccoglitore):
    kron = True
    A: dict = None
    B: dict = None
    ntok: dict = None

    def somma_kron(self, L, nome, x, g):
        if self.A is None:
            self.A, self.B, self.ntok = {}, {}, {}
        k = (L, nome)
        if k not in self.A:
            self.A[k] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=x.device)
            self.B[k] = torch.zeros(g.shape[1], g.shape[1], dtype=torch.float32, device=g.device)
            self.ntok[k] = 0
        self.A[k].addmm_(x.T, x)
        self.B[k].addmm_(g.T, g)
        self.ntok[k] += x.shape[0]

    def vuole(self, strato, nome):
        return strato in self.strati and self.e_piano_uno(nome) and nome not in ESPERTI

    def fine_passata(self, tag, seconda_meta=False):
        for k, c in self.cur.items():
            self.s1[k] += c
            c.zero_()
        self.n_kl += 1

    def somma(self, strato, nome, dW):
        k = (strato, nome)
        if dW.dim() == 3:
            dW = dW[0]
        if k not in self.cur:
            self.cur[k] = torch.zeros_like(dW, dtype=torch.float32)
            self.s1[k] = torch.zeros_like(self.cur[k])
        self.cur[k] += dW.float()


def leggi_mclp(log_path):
    """MCLP<TAB>task<TAB>right<TAB>chosen<TAB>n<TAB>logprob x n<TAB>token x n (perplexity.cpp, 26/9) -> {task: (right, [lp])}."""
    out = {}
    for r in open(log_path, errors="replace"):
        i = r.find("MCLP\t")
        if i < 0:
            continue
        c = r[i:].rstrip("\n").split("\t")
        n = int(c[4])
        out[int(c[1])] = (int(c[2]), [float(x) for x in c[5:5 + n]])
    return out


def pacchi_kl4(a, log):
    """Calibration questions packed into rows of <= calib_pacco tokens, one segment per question (positions and
    recurrent state restart at each segment: zero-shot like the bench)."""
    import json as _j
    import random as _r
    T4 = leggi_mclp(a.calib_maestro)
    Q = [_j.loads(x) for x in open(a.calib_jsonl)]
    Q = [q for q in Q if q["i"] in T4]
    assert Q, "kl4: no calibration question has a teacher MCLP line"
    for q in Q:
        assert T4[q["i"]][0] == q["label"], ("teacher log and calibration file disagree", q["i"])
    _r.Random(a.seme).shuffle(Q)
    if a.calib_n:
        Q = Q[: a.calib_n]
    pacchi, cur, nt = [], [], 0
    for q in Q:
        L = len(q["ids"])
        if cur and nt + L > a.calib_pacco:
            pacchi.append(cur); cur, nt = [], 0
        cur.append(q); nt += L
    if cur:
        pacchi.append(cur)
    for p in pacchi:
        for q in p:
            lp = torch.tensor(T4[q["i"]][1], dtype=torch.float64)
            q["pT"] = torch.softmax(lp, 0)
    info = {"n": len(Q), "q_medio": len(Q) / len(pacchi), "tok_medio": sum(len(q["ids"]) for q in Q) / len(pacchi)}
    log(f"kl4: {info['n']} domande in {len(pacchi)} pacchi da <= {a.calib_pacco} token "
        f"({info['q_medio']:.1f} domande, {info['tok_medio']:.0f} token per pacco)")
    return pacchi, info


def passo_kl4(st, WS, pacco, dev, rac, righe_out):
    """One packed row: forward with segments, KL(pT4 || pS4) on the letter logits at the last prompt token of each
    question, d(mean KL4 of the pack)/dh only at those positions, backward through the student."""
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
        Wl = WS[let].to(torch.float32)                              # (4, H)
        idx = torch.tensor(ultimi, device=dev)
        z = hh[idx].detach().to(torch.float32) @ Wl.T                # (q, 4)
        lS = torch.log_softmax(z.double(), -1)
        pT = torch.stack([q["pT"] for q in pacco]).to(dev)           # (q, 4) fp64
        kl = (pT * (torch.log(pT.clamp_min(1e-300)) - lS)).sum(-1)    # (q,)
        g = (lS.exp() - pT) / len(pacco)                             # d mean KL4 / dz
        dh[idx] = (g.to(torch.float32) @ Wl).to(dh.dtype)
        hh.backward(dh)
    st.indietro(stato, xL.grad)
    rac.fine_passata("kl4")
    pS = lS.exp()
    for q, k, ps, pt in zip(pacco, kl.tolist(), pS.tolist(), pT.tolist()):
        righe_out.append((q["i"], q["label"], k, ps[q["label"]], pt[q["label"]]))
    del stato, xL


def main():
    DR.preparse_thor_v2(None)
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default="@@ROOT:models@@/gguf/tony-tern/R2B3m/Thor-T-R2B3m.gguf")
    ap.add_argument("--teacher-cache", default="@@ROOT:depot@@/e2e_teacher_tt6/q8-TT6b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--righe", type=int, default=32)
    ap.add_argument("--seme", type=int, default=7)
    ap.add_argument("--ricetta", default="B_ind32")
    ap.add_argument("--pezzo-kl", type=int, default=512)
    ap.add_argument("--thor-v2", default=None)
    # 10/10 SELRISP (answer-guided selection): objective = KL4 on the MMLU answer token (4-way softmax over the letter
    # logprobs of the q8 teacher, MCLP lines of llama-perplexity) on a calibration set DISJOINT from the bench.
    # Default "kl" = the whole-text KL on teacher-cache rows, unchanged.
    ap.add_argument("--obiettivo", choices=("kl", "kl4"), default="kl")
    ap.add_argument("--calib-jsonl", default=None, help="kl4: calib_mmlu.jsonl (ids, label, letter ids)")
    ap.add_argument("--calib-maestro", default=None, help="kl4: teacher llama-perplexity log with MCLP lines")
    ap.add_argument("--calib-n", type=int, default=0, help="kl4: first N questions after the seeded shuffle (0 = all)")
    ap.add_argument("--calib-pacco", type=int, default=4096, help="kl4: max tokens per packed row (segments)")
    a = ap.parse_args()
    import e2e_allena as A
    import e2e_studente as S
    from rd_ternary.dispositivo import resolve_device
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = lambda *m: print(f"[{time.strftime('%H:%M:%S')}]", *m, flush=True)
    dev = resolve_device("auto")
    DR.disattiva_densi_piegati(log)
    if hasattr(A, "scegli_gdn"):
        A.scegli_gdn(log)
    st = DR.costruisci_studente(S, dict(gguf=Path(a.gguf), strati=range(40), ricetta=a.ricetta,
                                        motore_nome=os.environ.get("E2E_MOTORE", "triton"), device=str(dev), log=log,
                                        densi_ternari="tutte"), log)
    st.carica_embedding()
    WS = A.testa_studente(st, "fp32")
    WT, _ = A.testa_maestro(dev, torch.float32)
    src = A.SorgenteCache(a.teacher_cache)
    car = A.Caricatore(src, dev)
    righe = src.righe_previste("train")[:]
    __import__("random").Random(a.seme).shuffle(righe)
    righe = righe[: a.righe]
    if a.obiettivo == "kl4":
        pacchi, info4 = pacchi_kl4(a, log)
        righe = list(range(len(pacchi)))
    blk = list(range(40))
    rac = RaccoglitoreDensi(blk, device=dev)
    for L in st.strati:
        es = st.esperti(L)
        es.mot = MotoreRaff(es.mot, rac)
    st.congela_strati(blk)
    t0 = time.time()
    kl4_righe = []
    for n, i in enumerate(righe):
        if a.obiettivo == "kl4":
            passo_kl4(st, WS, pacchi[i], dev, rac, kl4_righe)
            if n == 0:
                log(f"densi visti dopo il primo pacco: {len(rac.s1)} famiglie; picco {torch.cuda.max_memory_allocated(dev) / 2**30:.1f} GiB")
            if os.environ.get("GRAD_META") == "1" and n + 1 == len(righe) // 2:
                for (L_, nome_), g_ in rac.s1.items():
                    np.save(out / f"gmeta_blk.{L_}.{nome_}.weight.npy", (g_ / rac.n_kl).cpu().numpy().astype(np.float16))
            log(f"pacco {n + 1}/{len(righe)}: {(time.time() - t0) / (n + 1):.1f} s/pacco, KL4 media finora "
                f"{np.mean([r[2] for r in kl4_righe]):.5f} su {len(kl4_righe)} domande")
            continue
        lot = car.prendi("train", [i])
        T = lot.ids.shape[1]
        stato = st.avanti(lot.ids, inizi=lot.inizi)
        xL = stato["xL"].detach().requires_grad_(True)
        with torch.enable_grad():
            h = st.esce(xL)
            hv = h.detach().requires_grad_(True)
            dh = torch.zeros_like(hv)
            for j in range(0, hv.shape[0], a.pezzo_kl):
                r = A.kl_pezzo(hv[j:j + a.pezzo_kl].detach(), WS, lot.hT[j:j + a.pezzo_kl], WT, True)
                dh[j:j + a.pezzo_kl] = r["dh"] / T
            h.backward(dh)
        st.indietro(stato, xL.grad)
        rac.fine_passata("kl")
        if n == 0:
            log(f"densi visti dopo la prima riga: {len(rac.s1)} famiglie; picco {torch.cuda.max_memory_allocated(dev) / 2**30:.1f} GiB")
        del stato, xL
        # 9/10 (PROVA_LEVE_VS_PAPER): GRAD_META=1 salva anche la media della PRIMA META' delle righe (fp16), per misurare
        # quanto del gradiente e' segnale e quanto rumore di campionamento (cos fra le due meta').
        if os.environ.get("GRAD_META") == "1" and n + 1 == len(righe) // 2:
            for (L_, nome_), g_ in rac.s1.items():
                np.save(out / f"gmeta_blk.{L_}.{nome_}.weight.npy", (g_ / rac.n_kl).cpu().numpy().astype(np.float16))
        log(f"riga {n + 1}/{len(righe)}: {(time.time() - t0) / (n + 1):.1f} s/riga")
    for (L, nome), g in rac.s1.items():
        np.save(out / f"g_blk.{L}.{nome}.weight.npy", (g / rac.n_kl).cpu().numpy().astype(np.float32))
    for k in rac.A:
        L, nome = k
        n = max(rac.ntok[k], 1)
        for tag, M in (("A", rac.A[k]), ("B", rac.B[k])):
            M = M / n
            sc = float(torch.diagonal(M).mean().clamp_min(1e-38))
            np.save(out / f"{tag}_blk.{L}.{nome}.weight.npy", (M / sc).cpu().numpy().astype(np.float32))  # mean diag = 1; fp32 (8/10): in fp16 la Gram perde la semidefinitezza
            with open(out / "scale_kron.tsv", "a") as f:
                f.write(f"{tag}\tblk.{L}.{nome}.weight\t{sc:.6e}\t{n}\n")
    if a.obiettivo == "kl4":
        with open(out / "kl4_torch.tsv", "w") as f:
            f.write("indice\tgiusta\tkl4\tpS_giusta\tpT_giusta\n")
            for r in kl4_righe:
                f.write("\t".join(str(x) for x in r) + "\n")
        # K-FAC Hessian of the mean KL4: n_units (questions) x tokens per pack, averaged (see predittore_kfac.py)
        (out / "kfac_fattore.txt").write_text(f"{info4['q_medio'] * info4['tok_medio']:.6e} q_medio {info4['q_medio']:.3f} "
                                              f"tok_medio {info4['tok_medio']:.1f} pacchi {len(pacchi)} domande {info4['n']}\n")
    (out / "fatto.txt").write_text(f"{rac.n_kl} righe, seme {a.seme}, cache {a.teacher_cache}, {len(rac.s1)} tensori\n")
    log(f"scritti {len(rac.s1)} gradienti in {out}")


if __name__ == "__main__":
    main()

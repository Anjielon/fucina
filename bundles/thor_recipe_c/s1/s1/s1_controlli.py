#!/usr/bin/env python3
"""S1 tappa 5 (CPU): controlli d'accettazione delle statistiche (S1 non da' un GGUF).

  (1) PONTE COL MANIFESTO: la H del flusso del MAESTRO sulla sorgente M (stesso corpus, stesso taglio) deve ridare la H
      del manifesto R2B3 su cui e' costruita ogni variante di C (stesso controllo di FAIR-Calib «A_U contro manifesto»).
      Differenza relativa di Frobenius per strato e punto; soglia dichiarata 2e-2 (bf16 su due caricatori diversi; la
      contaminazione toglie 1 delle 96 righe di M, la n. 74: ~1 % di scarto atteso per costruzione).
  (2) STABILITA' A/B: densi, H_q della meta' A contro la B; esperti, mediana per strato di ||Gq_A - Gq_B|| / ||Gq_A||
      sui pesi normalizzati (Gq/peso) e quota di esperti sotto 2.000 / 10.000 token.
  (3) DOVE STA C: H_q (flusso di C) contro H_f (maestro) sulla stessa M, e H_q della conoscenza (Kv, Kw) contro M.
  (4) GRADIENTI: per ogni tensore denso, sottospazio delle prime 8 direzioni di B (e di Bcam) in calibrazione contro
      controllo disgiunto: cos^2 medio degli angoli principali e quota d'energia di B_ctl nelle 8 direzioni di B_cal
      (contro la stessa quota di B_cal); coseno fra g_cal e g_ctl. «Alta» se la mediana del cos^2 >= 0,8 (soglia
      dichiarata; T1: il 99,8 % dell'energia di ffn_up_shexp 39 sta nelle prime 8).
Uscite: controlli.json, USCITE.txt (i percorsi che i passi S1b-S4 devono leggere).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import s1_comune as SC  # noqa: E402

MANIF_R2B3 = "@@ROOT:depot@@/tutto_ternario/tensori_R2B3/manifesto.json"
FAM_MANIF = {"attn": "attn_qkv3", "ffn": "ffn_gate_shexp3", "att_out": ("ssm_out3", "attn_output3"),
             "down": "ffn_down_shexp3"}


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def top_k(M: torch.Tensor, k: int = 8, it: int = 30) -> torch.Tensor:
    """Prime k autodirezioni (iterazione di sottospazio a blocco 2k: veloce anche su 8192 x 8192)."""
    n = M.shape[0]
    g = torch.Generator().manual_seed(0)
    Q = torch.linalg.qr(torch.randn(n, min(2 * k, n), generator=g, dtype=M.dtype))[0]
    for _ in range(it):
        Q = torch.linalg.qr(M @ Q)[0]
    w, V = torch.linalg.eigh(Q.T @ M @ Q)
    return Q @ V[:, -k:]


def controlla_cattura(C: Path, strati: list, ris: dict):
    lett = SC.LettoreS1(C)
    man = None
    try:
        man = json.load(open(MANIF_R2B3))
    except OSError:
        ris["avvisi"].append(f"manifesto {MANIF_R2B3} non leggibile: ponte non controllato")
    ponte, ab, dove, esp = {}, {}, {}, {}
    for L in strati:
        tipo = lett.info("A", L)["tipo"]
        for p in SC.PUNTI:
            HfA = lett.H_denso("A", L, p, ("M",), "f")
            HfB = lett.H_denso("B", L, p, ("M",), "f")
            nA, nB = lett.info("A", L)["token"]["M"], lett.info("B", L)["token"]["M"]
            Hf = (nA * HfA + nB * HfB) / (nA + nB)
            if man is not None:
                k = FAM_MANIF[p]
                if isinstance(k, tuple):
                    k = k[0] if tipo == "linear_attention" else k[1]
                inf = man.get(f"blk.{L}.{k}.weight", {}).get("hessiana")
                if inf:
                    ponte[f"{p}/{L:02d}"] = rel(Hf, np.load(inf["file"]).astype(np.float64))
            HqA = lett.H_denso("A", L, p)
            HqB = lett.H_denso("B", L, p)
            ab[f"{p}/{L:02d}"] = rel(HqB, HqA)
            HqM = (nA * lett.H_denso("A", L, p, ("M",)) + nB * lett.H_denso("B", L, p, ("M",))) / (nA + nB)
            d = {"C_contro_maestro_M": rel(HqM, Hf)}
            for s in ("Kv", "Kw"):
                d[f"{s}_contro_M"] = rel(lett.H_denso("A", L, p, (s,)), lett.H_denso("A", L, p, ("M",)))
            dove[f"{p}/{L:02d}"] = d
        fa = lett.dir("A", L) / "esperti_Gq.npy"
        if fa.exists():
            exA, exB = lett.esperti("A", L), lett.esperti("B", L)
            rr = []
            for e in range(0, exA["Gq"].shape[0], 16):          # campione di 16 esperti per strato
                if exA["conta"][e] == 0 or exB["conta"][e] == 0:
                    continue
                a = np.asarray(exA["Gq"][e], np.float64) * exA["Gq_scala"][e] / exA["peso"][e]
                b = np.asarray(exB["Gq"][e], np.float64) * exB["Gq_scala"][e] / exB["peso"][e]
                rr.append(float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-300)))
            c = np.concatenate([exA["conta"], exB["conta"]])
            esp[f"{L:02d}"] = {"Gq_AB_rel_mediana_campione": float(np.median(rr)) if rr else None,
                               "conta_mediana_A": float(np.median(exA["conta"])),
                               "sotto_2000_A": int((exA["conta"] < 2000).sum()),
                               "sotto_10000_A": int((exA["conta"] < 10000).sum()), "mai_visti": int((c == 0).sum())}
    v = list(ponte.values())
    ris["ponte_manifesto"] = {"per_tensore": ponte, "max": max(v) if v else None,
                              "mediana": float(np.median(v)) if v else None, "soglia": 2e-2,
                              "ok": (max(v) < 2e-2) if v else None}
    va = list(ab.values())
    ris["stabilita_AB_densi"] = {"per_punto": ab, "max": max(va) if va else None, "mediana": float(np.median(va)) if va else None}
    ris["dove_sta_C"] = dove
    ris["esperti"] = esp


def controlla_gradienti(G: Path, ris: dict):
    cal, ctl = G / "cal", G / "ctl"
    out = {}
    for f in sorted(cal.glob("B_blk.*.npy")):
        t = f.name[2:-4]
        d = {}
        for tag in ("B", "Bcam"):
            a, b = cal / f"{tag}_{t}.npy", ctl / f"{tag}_{t}.npy"
            if not (a.exists() and b.exists()):
                continue
            Ma = torch.from_numpy(np.load(a)).double()
            Mb = torch.from_numpy(np.load(b)).double()
            k = min(8, Ma.shape[0])
            Va, Vb = top_k(Ma, k), top_k(Mb, k)
            s = torch.linalg.svdvals(Va.T @ Vb)
            d[tag] = {"cos2_medio_top8": float((s ** 2).mean()),
                      "energia_top8_cal": float(torch.trace(Va.T @ Ma @ Va) / torch.trace(Ma)),
                      "energia_ctl_nelle_top8_cal": float(torch.trace(Va.T @ Mb @ Va) / torch.trace(Mb))}
        ga, gb = cal / f"g_{t}.npy", ctl / f"g_{t}.npy"
        if ga.exists() and gb.exists():
            x, y = np.load(ga).astype(np.float64).ravel(), np.load(gb).astype(np.float64).ravel()
            d["g_coseno_cal_ctl"] = float(x @ y / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-300))
        a2 = ctl / f"A_{t}.npy"
        if a2.exists():
            d["A_rel_cal_ctl"] = rel(np.load(a2).astype(np.float64), np.load(cal / f"A_{t}.npy").astype(np.float64))
        out[t] = d
    sint = {}
    for tag in ("B", "Bcam"):
        v = [d[tag]["cos2_medio_top8"] for d in out.values() if tag in d]
        if v:
            sint[tag] = {"cos2_top8_mediana": float(np.median(v)), "minimo": float(min(v)),
                         "alta": bool(np.median(v) >= 0.8), "tensori": len(v)}
    v = [d["g_coseno_cal_ctl"] for d in out.values() if "g_coseno_cal_ctl" in d]
    if v:
        sint["g_coseno_mediana"] = float(np.median(v))
    ris["gradienti"] = {"per_tensore": out, "sintesi": sint}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radice", required=True)
    ap.add_argument("--strati", default="0-39")
    a = ap.parse_args()
    R = Path(a.radice)
    s = a.strati.split("-")
    strati = list(range(int(s[0]), int(s[-1]) + 1))
    ris = {"quando": time.strftime("%FT%T"), "avvisi": []}
    t0 = time.time()
    torch.set_num_threads(4)
    controlla_cattura(R / "cattura", strati, ris)
    log(f"cattura: ponte col manifesto {ris['ponte_manifesto'].get('max')}, A/B densi mediana "
        f"{ris['stabilita_AB_densi'].get('mediana')}")
    if (R / "gradienti" / "cal").exists() and (R / "gradienti" / "ctl").exists():
        controlla_gradienti(R / "gradienti", ris)
        log(f"gradienti: {ris['gradienti']['sintesi']}")
    else:
        ris["avvisi"].append("gradienti cal/ctl assenti")
    ris["secondi"] = round(time.time() - t0, 1)
    SC.scrivi_json(R / "controlli.json", ris)
    (R / "USCITE.txt").write_text("\n".join([
        f"cattura (statistiche del flusso di C, meta' A costruisce / B guardia): {R / 'cattura'}/{{A,B}}/strato_LL/",
        "  lettura: s1_comune.LettoreS1(<cattura>).H_denso / M_incrociata / H_esperto(k=8000) / C_esperto",
        f"gradienti calibrazione (g, A, B=Fisher empirica, Bcam=Fisher di cammino; base memorizzata): {R / 'gradienti' / 'cal'}",
        f"gradienti controllo disgiunto: {R / 'gradienti' / 'ctl'}",
        f"corpus e contaminazione: {R / 'corpus'}/corpus.json, contaminazione.json",
        f"ospite: {R / 'ospite.json'}",
        f"controlli: {R / 'controlli.json'}", ""]))
    log(f"controlli scritti in {R / 'controlli.json'} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

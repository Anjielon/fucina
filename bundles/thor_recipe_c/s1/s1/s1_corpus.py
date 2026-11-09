#!/usr/bin/env python3
"""S1 tappa 1 (CPU): corpus ricco di conoscenza per le statistiche del flusso di C + righe dei gradienti + contaminazione.

Sorgenti (tutte pubbliche, gia' in casa, nessun account):
  M   calib_auto2_pulito.txt, 96 x 2048 = 196.608 token: il corpus di OGNI H di C (manifesto R2B3), tokenizzato come
      FAIR-Calib/cattura_hessiane_attn_thor (testo intero in un colpo, righe da 2048, numero multiplo di 16). Serve da
      ponte: la sua H sul flusso del MAESTRO deve ridare la H del manifesto (controllo della cattura).
  Kv  calibrazione_corpus_v2.pt, domini wiki + edu (450 + 450 x 1024 = 921.600 token): Wikipedia inglese e testi
      educativi (FineWeb-Edu), gia' depurati da 3 documenti contaminati il 25/9. La linea FRONTIERA ha misurato che la
      H con wiki+edu toglie il 16-18 % dell'errore su testo di conoscenza mai visto (strato 0).
  Kw  WikiText-103 train (arXiv 1609.07843, licenza CC BY-SA), finestre da 256 token della cache del maestro q8
      CARDAN_FEDELE/teacher/q8-maestro-wt103 (gia' verificata: 0 prefissi e 0 13-grammi contro i 14.042 MMLU test).
      Le righe della cache (16 finestre ciascuna) si dividono, con l'ordine di gradienti_densi.py (Random(seme).shuffle):
      [0, n_cal) gradienti di calibrazione, [n_cal, n_cal+n_ctl) gradienti di CONTROLLO (disgiunte), poi le righe della
      cattura. Nessuna riga sta in due insiemi.
Meta' A / B: per ogni sorgente, sequenze di indice pari -> A (costruisce), dispari -> B (guardia held-out).
Contaminazione: 13-grammi di token (come il manifesto wt103 e GPT-3) contro TUTTE le domande MMLU in casa (cais/mmlu
'all': test 14.042, validation 1.531, dev 285; domanda e domanda+scelte), contro i 2000 prompt del banco
(mc_v2/mmlu_00000.bin, testo dopo l'intestazione) e contro le 1.779 domande di SELRISP/calib (dove vive la KL4 held-out
del cancello). Una sequenza con anche un solo 13-gramma in comune si TOGLIE; alla fine si verifica 0 collisioni.
Uscite: <out>/corpus.pt, corpus.json, contaminazione.json, FATTO.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import s1_comune as SC  # noqa: E402

HF = Path("@@ROOT:models@@/gguf/thor-src-hf")
M_TXT = Path("@@ROOT:forge@@/calib_auto2_pulito.txt")
KV_PT = Path("@@ROOT:models@@/gguf/tony-tern/calibrazione/calibrazione_corpus_v2.pt")
WT_CACHE = Path("@@ROOT:models@@/gguf/tony-tern/CARDAN_FEDELE/teacher/q8-maestro-wt103")
BANCO = Path("@@ROOT:forge@@/mc_v2/mmlu_00000.bin")
SELRISP = Path("@@ROOT:models@@/gguf/tony-tern/SELRISP/calib/calib_mmlu.jsonl")


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def leggi_banco(p: Path) -> list:
    d = p.read_bytes()
    n = struct.unpack_from("<I", d, 0)[0]
    pos = struct.unpack_from(f"<{n}I", d, 4)
    out = []
    for o in pos:
        L = struct.unpack_from("<I", d, o)[0]
        out.append(d[o + 4:o + 4 + L].decode())
    return out


def corpo(prompt: str) -> str:
    """Testo del compito senza l'intestazione «The following are multiple choice questions ... about X.»."""
    i = prompt.find("\n\n")
    return prompt[i + 2:] if i >= 0 else prompt


def ngrammi_mmlu(tok, log) -> tuple:
    """(array ordinato degli hash, dizionario hash -> tupla per la verifica esatta, conteggi per fonte)."""
    import os
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from datasets import load_dataset
    testi, conte = [], {}
    for split in ("test", "validation", "dev"):
        d = load_dataset("cais/mmlu", "all", split=split)
        for r in d:
            testi.append(r["question"])
            testi.append(r["question"].strip() + "\n" + "\n".join(f"{c}. {x}" for c, x in zip("ABCD", r["choices"])))
        conte[f"mmlu_{split}"] = len(d)
    b = leggi_banco(BANCO)
    testi += [corpo(p) for p in b]
    conte["banco_mmlu_00000"] = len(b)
    sel = [json.loads(x) for x in open(SELRISP)]
    conte["selrisp_calib"] = len(sel)
    seqs = [np.asarray(tok(t, add_special_tokens=False)["input_ids"], np.int64) for t in testi]
    for q in sel:                                     # ids gia' tokenizzati: tolta l'intestazione (fino al primo \n\n)
        ids = q["ids"]
        dec = tok.decode(ids)
        seqs.append(np.asarray(tok(corpo(dec), add_special_tokens=False)["input_ids"], np.int64))
    tab = {}
    for s in seqs:
        if s.size < SC.NGRAM:
            continue
        h = SC.hash_ngrammi(s)
        for k, v in zip(h.tolist(), np.lib.stride_tricks.sliding_window_view(s, SC.NGRAM)):
            tab.setdefault(k, tuple(int(x) for x in v))
    arr = np.array(sorted(tab), dtype=np.uint64)
    log(f"contaminazione: {len(seqs)} testi MMLU/banco/SELRISP, {arr.size} 13-grammi distinti")
    return arr, tab, conte


def colpi(seqs: np.ndarray, arr: np.ndarray, tab: dict) -> np.ndarray:
    """Per ogni sequenza (righe), numero di 13-grammi in comune VERIFICATI esattamente."""
    out = np.zeros(seqs.shape[0], np.int64)
    for i in range(seqs.shape[0]):
        s = np.asarray(seqs[i], np.int64)
        h = SC.hash_ngrammi(s)
        pos = np.searchsorted(arr, h).clip(max=arr.size - 1)       # arr e' ordinato: niente riordino a ogni chiamata
        m = arr[pos] == h
        if m.any():
            w = np.lib.stride_tricks.sliding_window_view(s, SC.NGRAM)
            out[i] = sum(1 for j in np.nonzero(m)[0] if tab.get(int(h[j])) == tuple(int(x) for x in w[j]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--thor-v2", required=True, help="albero congelato (e2e_teacher per la cache WT103)")
    ap.add_argument("--seme", type=int, default=7)
    ap.add_argument("--righe-cal", type=int, default=32)
    ap.add_argument("--righe-ctl", type=int, default=32)
    ap.add_argument("--righe-wt", type=int, default=64, help="righe della cache WT103 (x16 finestre) per la cattura")
    ap.add_argument("--ridotto", type=int, default=0, help="prova: tieni solo N sequenze per sorgente")
    a = ap.parse_args()
    sys.path.insert(0, a.thor_v2)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "FATTO").exists():
        log(f"corpus gia' FATTO in {out}: esco")
        return 0
    t0 = time.time()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(HF))
    # M
    t = tok(M_TXT.read_text(encoding="utf-8", errors="replace"), return_tensors="pt",
            add_special_tokens=False).input_ids[0]
    n_seq = ((t.numel() // 2048) // 16) * 16
    M = t[: n_seq * 2048].reshape(n_seq, 2048)
    # Kv
    c = torch.load(KV_PT, map_location="cpu", weights_only=False)
    Kv = torch.cat([c["wiki"], c["edu"]])
    Kv_dom = ["wiki"] * c["wiki"].shape[0] + ["edu"] * c["edu"].shape[0]
    # Kw + righe dei gradienti
    import e2e_teacher as T
    cm = T.CacheMaestro(WT_CACHE)
    righe = [i for b in cm.piano["blocchi"] if b["lato"] == "train" for i in range(b["da"], b["a"])]
    righe = list(righe)
    random.Random(a.seme).shuffle(righe)
    arr, tab, conte = ngrammi_mmlu(tok, log)

    def finestre(i):
        ids = np.asarray(cm.ids("train", i), np.int64)
        cu = cm.cu_seqlens("train", i)
        return [ids[s0:s1] for s0, s1 in zip(cu[:-1], cu[1:])]

    def riga_pulita(i):
        return all(int(colpi(w[None], arr, tab)[0]) == 0 for w in finestre(i) if w.size >= SC.NGRAM)

    usate, scartate = [], []
    it = iter(righe)

    def prendi(n):
        sel = []
        while len(sel) < n:
            i = next(it)
            (sel if riga_pulita(i) else scartate).append(i)
        usate.extend(sel)
        return sel
    cal = prendi(a.righe_cal)
    ctl = prendi(a.righe_ctl)
    wt = prendi(a.righe_wt)
    fin = []
    for i in wt:
        for w in finestre(i):
            assert w.size == 256, (i, w.size)
            fin.append(torch.from_numpy(w))
    Kw = torch.stack(fin)
    assert not (set(cal) & set(ctl)) and not (set(cal) & set(wt)) and not (set(ctl) & set(wt))
    sorg = {"M": M.to(torch.int32), "Kv": Kv.to(torch.int32), "Kw": Kw.to(torch.int32)}
    rep = {"tolte": {}, "colpi_iniziali": {}, "fonti_mmlu": conte, "ngram": SC.NGRAM, "righe_wt_scartate": scartate}
    for s in ("M", "Kv"):
        k = colpi(sorg[s].numpy(), arr, tab)
        rep["colpi_iniziali"][s] = int((k > 0).sum())
        tieni = np.nonzero(k == 0)[0]
        rep["tolte"][s] = [int(x) for x in np.nonzero(k > 0)[0]]
        sorg[s] = sorg[s][torch.from_numpy(tieni)]
        if s == "Kv":
            Kv_dom = [Kv_dom[j] for j in tieni]
    k = colpi(sorg["Kw"].numpy(), arr, tab)
    rep["colpi_iniziali"]["Kw"] = int((k > 0).sum())
    assert (k == 0).all(), "Kw: righe pulite con colpi?"
    if a.ridotto:
        sorg = {s: v[: a.ridotto] for s, v in sorg.items()}
        Kv_dom = Kv_dom[: a.ridotto]
    # verifica finale: zero collisioni
    fin_c = {s: int((colpi(v.numpy(), arr, tab) > 0).sum()) for s, v in sorg.items()}
    assert all(v == 0 for v in fin_c.values()), fin_c
    rep["colpi_finali"] = fin_c
    meta = {s: {"A": list(range(0, v.shape[0], 2)), "B": list(range(1, v.shape[0], 2))} for s, v in sorg.items()}
    torch.save({"sorgenti": sorg, "Kv_dom": Kv_dom, "meta": meta, "righe_cal": cal, "righe_ctl": ctl, "righe_wt": wt},
               out / "corpus.pt")
    info = {"quando": time.strftime("%FT%T"), "tokenizzatore": str(HF), "seme": a.seme,
            "sorgenti": {s: {"forma": list(v.shape), "token": int(v.numel()),
                             "token_A": int(v[meta[s]["A"]].numel()), "token_B": int(v[meta[s]["B"]].numel()),
                             "sha256_ids": hashlib.sha256(v.numpy().tobytes()).hexdigest()} for s, v in sorg.items()},
            "descrizione": {"M": f"{M_TXT} (corpus del manifesto R2B3, righe 2048)",
                            "Kv": f"{KV_PT} domini wiki+edu (righe 1024)",
                            "Kw": f"WikiText-103 train, finestre 256 della cache {WT_CACHE}"},
            "Kv_domini": {d: Kv_dom.count(d) for d in sorted(set(Kv_dom))},
            "gradienti": {"cache": str(WT_CACHE), "righe_cal": cal, "righe_ctl": ctl,
                          "regola": f"Random({a.seme}).shuffle(righe train) come gradienti_densi.py; disgiunte"},
            "righe_wt_cattura": wt, "ridotto": a.ridotto, "secondi": round(time.time() - t0, 1)}
    info["token_totali"] = sum(v["token"] for v in info["sorgenti"].values())
    SC.scrivi_json(out / "contaminazione.json", rep)
    SC.scrivi_json(out / "corpus.json", info)
    (out / "FATTO").write_text(time.strftime("%FT%T") + "\n")
    log(f"corpus: {info['token_totali']} token {[(s, v['forma']) for s, v in info['sorgenti'].items()]}; "
        f"collisioni iniziali {rep['colpi_iniziali']} -> finali {fin_c}; righe gradienti cal {len(cal)} ctl {len(ctl)}; "
        f"{time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

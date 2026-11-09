#!/usr/bin/env python3
"""Test di `e2e_corpus_4096.py` (25/9/2026, file NUOVO). Due parti:

1. **Unitari** (senza file, secondi): `riempi`, `quote_domini`, `confini`/`confini_multi` con `C.impacca_fonte` e
   `C.dominio_multi` VERI a lunghezza piccola e un tokenizzatore finto, `ordine_domini`, `messaggi_da_testo`.
2. **Sui file prodotti** (`--dir`, default `calibrazione/e2e`), con controlli INDIPENDENTI dal costruttore: si
   rileggono i token, si decodificano col tokenizzatore di Thor (caricato qui, non via `C.Tok`), si ricavano i messaggi
   dal template e si confrontano con un hash di finestre di 32 token DIVERSO da quello del costruttore:
   - forma, dtype, sha256 del manifesto, somme delle righe, niente dominio wiki, nessuna riga ripetuta;
   - cu_seqlens: 0 ... 4096 crescenti; nei domini a esempi ogni confine interno cade su `<|im_start|>`;
   - `ordine`: ogni (dominio, riga) una volta, prefissi con la composizione;
   - train/held-out: id disgiunti; finestre di 32 token del testo continuo in comune = 0; messaggi normalizzati
     (>= 24 caratteri, utente/assistente/ragionamento, senza system e senza <tool_call>) in comune = 0, e loro
     finestre di 32 token in comune = 0;
   - corpus contro wiki.test.raw, mmlu_test.bin, pannelli (storico e v2): stesse misure = 0;
   - held-out contro i corpus di calibrazione della forgia: stesse misure = 0.

    env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=2 RAYON_NUM_THREADS=2 nice -n 19 ionice -c3 \\
        @@ROOT:venv@@/bin/python3 e2e_test_corpus_4096.py [--dir DIR] [--solo-unitari]
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_k, "2")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import struct  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import unicodedata  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import e2e_corpus_4096 as E  # noqa: E402
from e2e_corpus_4096 import C  # noqa: E402

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
ESITI: list = []


def prova(nome: str, cond: bool, dettaglio="") -> None:
    ESITI.append({"prova": nome, "ok": bool(cond), "dettaglio": dettaglio})
    print(("✅" if cond else "⛔"), nome, dettaglio if dettaglio != "" else "", flush=True)


# ═══════════════════════════ 1. unitari ═══════════════════════════
class TkFinto:
    """Tokenizzatore finto: un token per carattere (codice), cosi' le lunghezze sono note."""

    def ids(self, testi):
        return [[ord(c) for c in t] for t in testi]


def unitari() -> None:
    # riempi: senza tetti = C.ripartisci
    pesi = [("a", 60), ("b", 30), ("c", 35), ("d", 100)]
    cap = {f: 10**6 for f, _ in pesi}
    prova("riempi senza tetti == C.ripartisci", E.riempi(118, pesi, cap) == C.ripartisci(118, tuple(pesi)),
          E.riempi(118, pesi, cap))
    q = E.riempi(118, pesi, {"a": 12, "b": 9, "c": 10**6, "d": 56})
    prova("riempi: tetti rispettati, somma esatta, il resto va a chi ha posto",
          q["a"] == 12 and q["b"] == 9 and q["d"] == 56 and sum(q.values()) == 118, q)
    q = E.riempi(50, pesi, {"a": 5, "b": 5, "c": 5, "d": 5})
    prova("riempi: tutte piene -> somma = capienza totale", q == {"a": 5, "b": 5, "c": 5, "d": 5}, q)
    q = E.riempi(10, [("a", 1), ("b", 1)], {"a": 0, "b": 20})
    prova("riempi: capienza 0 esclusa", q == {"a": 0, "b": 10}, q)
    # quote_domini
    pesi_u = E.quote_u("tutte")
    prova("quote U senza wiki (lette da V2.MISCELE)", pesi_u == {"edu": 2, "cot": 2, "it_web": 2, "it_wiki": 1,
                                                                   "it_chat": 3, "it_pensiero": 1, "strumenti": 2}, pesi_u)
    grande = {d: 10**6 for d in pesi_u}
    q, rap = E.quote_domini(512, pesi_u, grande)
    prova("quote_domini senza tetti == C.ripartisci(512)", q == C.ripartisci(512, tuple(pesi_u.items()))
          and rap["mancano_sequenze"] == 0, q)
    cap = dict(grande, cot=30)
    q, rap = E.quote_domini(512, pesi_u, cap)
    base = C.ripartisci(512, tuple(pesi_u.items()))
    prova("quote_domini: cot a 30, deficit a edu+it_web, somma 512",
          q["cot"] == 30 and q["edu"] + q["it_web"] == base["edu"] + base["it_web"] + base["cot"] - 30
          and sum(q.values()) == 512 and all(q[d] == base[d] for d in pesi_u if d not in ("cot", "edu", "it_web")),
          q)
    cap = dict(cap, edu=90, it_web=90)
    q, rap = E.quote_domini(512, pesi_u, cap)
    prova("quote_domini: assorbitori pieni -> mancano dichiarate", rap["mancano_sequenze"] == 512 - sum(q.values())
          and rap["mancano_sequenze"] > 0 and q["edu"] == 90 and q["it_web"] == 90, rap["mancano_sequenze"])
    # confini con impacca_fonte e dominio_multi VERI, L = 16
    tk = TkFinto()
    es = {f: [C.Esempio(f, f"{f}:{i}", chr(65 + i % 26) * (3 + (7 * i) % 11), []) for i in range(40)]
          for f in ("x", "y", "z")}
    with E.lunghezza(16):
        p = C.impacca_fonte("x", es["x"], 5, tk)
        cu, ini = E.confini(p.usati, p.seq, 16)
        ok = all(c[0] == 0 and c[-1] == 16 and all(a < b for a, b in zip(c, c[1:])) for c in cu)
        flusso = [t for e in p.usati for t in e.ids]
        inizi = set(np.cumsum([0] + [len(e.ids) for e in p.usati]).tolist())
        ok2 = all(sorted({j * 16 + x for x in c[:-1]} - {j * 16} | ({j * 16} if ini[j] else set()))
                  == sorted(s for s in inizi if j * 16 <= s < (j + 1) * 16) for j, c in enumerate(cu))
        prova("confini: cu validi e = inizi degli esempi (impacca_fonte vero, L=16)", ok and ok2
              and flusso[:80] == [t for s in p.seq for t in s], cu[:3])
        comp = (("x", 3), ("y", 2), ("z", 4))
        s, et, info, u = C.dominio_multi("d", 9, es, comp, tk)
        cu, ini = E.confini_multi(s, info, u, comp, 16)
        prova("confini_multi: ricostruzione al bit e cu validi (dominio_multi vero)", len(cu) == 9 and all(
            c[0] == 0 and c[-1] == 16 for c in cu), et)
        # una riga alterata deve far fallire la verifica
        s2 = [list(r) for r in s]
        s2[4][3] += 1
        try:
            E.confini_multi(s2, info, u, comp, 16)
            prova("confini_multi rifiuta una riga alterata", False)
        except RuntimeError:
            prova("confini_multi rifiuta una riga alterata", True)
    prova("lunghezza() rimette C.L", C.L == 1024, C.L)
    o = E.ordine_domini({d: n for d, n in zip(E.DOMINI_E2E, (4, 4, 4, 2, 6, 2, 4))})
    visti = {tuple(x) for x in o}
    prova("ordine_domini: ogni riga una volta", len(o) == 26 and len(visti) == 26)
    t = ("<|im_start|>system\n# Tools\n<tools>{...}</tools><|im_end|>\n<|im_start|>user\nChe ore sono a Roma adesso?"
         "<|im_end|>\n<|im_start|>assistant\n<think>\nDevo chiamare lo strumento dell'ora.\n</think>\n\n<tool_call>\n"
         "<function=orologio>\n</function>\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\nSono le 10:15"
         "\n</tool_response><|im_end|>\n")
    m = E.messaggi_da_testo(t)
    prova("messaggi_da_testo: utente, pensiero, risposta strumento; niente system ne' tool_call",
          m == {"che ore sono a roma adesso", "devo chiamare lo strumento dell ora", "sono le 10 15"}, sorted(m))


# ═══════════════════════════ 2. sui file ═══════════════════════════
_P = np.random.default_rng(777).integers(1, 2**61, size=32, dtype=np.int64).astype(np.uint64) | np.uint64(1)


def finestre(ids) -> np.ndarray:
    """Hash di finestre di 32 token, costanti DIVERSE da quelle del costruttore (seme 777, non 20260925)."""
    a = np.asarray(ids, dtype=np.int64)
    if a.size < 32:
        return np.empty(0, dtype=np.uint64)
    w = np.lib.stride_tricks.sliding_window_view(a.astype(np.uint64), 32)
    with np.errstate(over="ignore"):
        h = (w * _P).sum(axis=1, dtype=np.uint64)
        return np.unique(h ^ (h >> np.uint64(29)))


def comuni(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.intersect1d(a, b, assume_unique=True).size)


def nrm(t: str) -> str:
    t = unicodedata.normalize("NFKC", t).lower()
    return " ".join(re.findall(r"\w+", t))


RX_T = re.compile(r"<\|im_start\|>(system|user|assistant|tool)\n(.*?)<\|im_end\|>", re.S)


def messaggi(testo: str) -> list:
    out = []
    for ruolo, t in RX_T.findall(testo):
        if ruolo == "system":
            continue
        if ruolo == "assistant" and "</think>" in t:
            a, b = t.split("</think>", 1)
            parti = [a.replace("<think>", ""), b]
        else:
            parti = [t]
        for p in parti:
            p = re.sub(r"<tool_call>.*?</tool_call>", " ", p, flags=re.S)
            p = p.replace("<tool_response>", " ").replace("</tool_response>", " ")
            n = nrm(p)
            if len(n) >= 24:
                out.append(n)
    return out


class Tk:
    def __init__(self):
        from transformers import AutoTokenizer
        self.t = AutoTokenizer.from_pretrained(str(C.SRC))

    def dec(self, ids) -> str:
        return self.t.decode(list(ids))

    def enc(self, testi) -> list:
        return self.t(list(testi), add_special_tokens=False)["input_ids"] if testi else []


def per_fonte(x: torch.Tensor, etichette: list) -> dict:
    out = {}
    for f in dict.fromkeys(etichette):
        righe = [i for i, e in enumerate(etichette) if e == f]
        out[f] = x[righe].reshape(-1).tolist()
    return out


def impronta(tk: Tk, dominio_seq: dict, etichette: dict, continuo: set) -> dict:
    """Per un insieme di domini: finestre del testo continuo, messaggi normalizzati, finestre dei messaggi."""
    fin_c, mess = [], set()
    for d, x in dominio_seq.items():
        if x.numel() == 0:
            continue
        if d in continuo:
            fin_c.append(finestre(x.reshape(-1).numpy()))
            continue
        for f, flusso in per_fonte(x, etichette[d]).items():
            mess |= set(messaggi(tk.dec(flusso)))
    fin_m = [finestre(ids) for ids in tk.enc(sorted(mess))]
    unisci = lambda v: np.unique(np.concatenate(v)) if v else np.empty(0, dtype=np.uint64)  # noqa: E731
    return {"continuo": unisci(fin_c), "messaggi": mess, "fin_messaggi": unisci(fin_m)}


def sui_file(d: Path) -> None:
    t0 = time.time()
    tk = Tk()
    man = {p.stem.replace("e2e_corpus_4096_", ""): json.loads(p.read_text()) for p in sorted(d.glob("e2e_corpus_4096_*.json"))}
    prova("manifesti presenti", bool(man), sorted(man))
    # riferimenti comuni
    rif: dict = {}
    wt = tk.enc([C.FILE_WIKI_TEST.read_text(encoding="utf-8")])[0]
    rif["wiki.test.raw"] = {"continuo": finestre(wt), "messaggi": set(), "fin_messaggi": np.empty(0, dtype=np.uint64)}
    raw = C.FILE_MMLU.read_bytes()
    n = struct.unpack_from("<I", raw, 0)[0]
    testi = []
    for o in struct.unpack_from(f"<{n}I", raw, 4):
        ln = struct.unpack_from("<I", raw, o)[0]
        testi.append(raw[o + 4:o + 4 + ln].decode("utf-8", errors="replace"))
    fm = [finestre(x) for x in tk.enc(testi)]
    rif["mmlu_test.bin"] = {"continuo": np.unique(np.concatenate(fm)), "messaggi": {nrm(q) for q in testi if len(nrm(q)) >= 24},
                            "fin_messaggi": np.empty(0, dtype=np.uint64)}
    for p in (C.DIR_PAN / "pannello_ids.pt", C.DIR_PAN / "pannello_nuovi_domini_v2.pt"):
        pan = torch.load(p, weights_only=False)["domini"]
        seqs = {k: v.reshape(-1, v.shape[-1]) for k, v in pan.items()}
        # pannello: tutto e' riferimento; il testo continuo come finestre, le chat anche come messaggi
        im = impronta(tk, seqs, {k: ["p"] * v.shape[0] for k, v in seqs.items()},
                      continuo={"wiki", "edu", "it_web", "it_wiki", "storico", "lunghe"})
        allc = np.unique(np.concatenate([finestre(v.reshape(-1).numpy()) for v in seqs.values()]))
        im["continuo"] = allc
        rif[p.name] = im
    forgia = {}
    for p in sorted(C.DIR_CAL.glob("calibrazione_corpus*.pt")):
        m = json.loads(p.with_suffix(".json").read_text())
        x = torch.load(p, weights_only=True)
        et = {k: [f for f, c in m["etichette_per_sequenza"][k] for _ in range(c)] for k in x}
        im = impronta(tk, x, et, continuo=set(x))      # finestre di TUTTE le sequenze (template compreso)
        im2 = impronta(tk, {k: v for k, v in x.items() if k not in E.TESTO_CONTINUO + ("wiki",)}, et, continuo=set())
        forgia[p.name] = {"continuo": im["continuo"], "messaggi": im2["messaggi"], "fin_messaggi": im2["fin_messaggi"]}
    print(f"   riferimenti pronti in {time.time() - t0:.0f} s", flush=True)

    for v, m in man.items():
        lati = {}
        for lato in ("train", "heldout"):
            f = m["file"][lato]
            p = d / f["file"]
            prova(f"{v}/{lato}: sha256 del file = manifesto", C.sha256_file(p) == f["sha256"])
            x = torch.load(p, weights_only=True)
            lati[lato] = x
            dom = x["domini"]
            prova(f"{v}/{lato}: niente wiki, domini = {E.DOMINI_E2E}", tuple(x["nomi_domini"]) == E.DOMINI_E2E
                  and "wiki" not in dom)
            forme = all(t.dtype == torch.int64 and t.dim() == 2 and t.shape[1] == 4096 for t in dom.values())
            righe = sum(t.shape[0] for t in dom.values())
            prova(f"{v}/{lato}: int64 (n, 4096), righe = manifesto", forme and righe == f["righe"], righe)
            prova(f"{v}/{lato}: sha256 dei token per dominio = manifesto",
                  all(C.sha_tensore(dom[k]) == m["domini"][lato][k]["sha256_token"] for k in dom))
            tutte = torch.cat([t for t in dom.values()])
            uniche = torch.unique(tutte, dim=0).shape[0]
            prova(f"{v}/{lato}: nessuna riga ripetuta", uniche == tutte.shape[0], f"{uniche}/{tutte.shape[0]}")
            # cu_seqlens
            ok, bordi, bordi_ok = True, 0, 0
            for k, t in dom.items():
                ptr, cu = x["cu_ptr"][k], x["cu"][k]
                ok &= ptr.shape[0] == t.shape[0] + 1 and int(ptr[-1]) == cu.shape[0]
                for i in range(t.shape[0]):
                    c = cu[int(ptr[i]):int(ptr[i + 1])].tolist()
                    ok &= c[0] == 0 and c[-1] == 4096 and all(a < b for a, b in zip(c, c[1:]))
                    if k in E.TESTO_CONTINUO:
                        ok &= c == [0, 4096]
                    else:
                        for b in c[1:-1]:
                            bordi += 1
                            bordi_ok += int(t[i, b]) == 248045
                        if bool(x["inizio_esempio"][k][i]):
                            bordi += 1
                            bordi_ok += int(t[i, 0]) == 248045
            prova(f"{v}/{lato}: cu_seqlens validi; ogni inizio d'esempio su <|im_start|>", ok and bordi == bordi_ok,
                  f"{bordi_ok}/{bordi} confini")
            o = x["ordine"].tolist()
            n_d = {k: dom[k].shape[0] for k in E.DOMINI_E2E}
            prova(f"{v}/{lato}: ordine = ogni riga una volta", sorted(map(tuple, o)) ==
                  sorted((i, j) for i, k in enumerate(E.DOMINI_E2E) for j in range(n_d[k])))
            if righe:
                pref = o[:righe // 2]
                cnt = {k: sum(1 for a, _ in pref if E.DOMINI_E2E[a] == k) for k in E.DOMINI_E2E}
                dev = max(abs(cnt[k] - n_d[k] / 2) for k in E.DOMINI_E2E)
                prova(f"{v}/{lato}: prima meta' dell'ordine con la composizione (scarto <= 1 riga)", dev <= 1, cnt)
            ids = [i for k in dom for i in x["esempi"][k]]
            prova(f"{v}/{lato}: nessun esempio usato due volte", len(ids) == len(set(ids)), len(ids))
        tr, ho = lati["train"], lati["heldout"]
        prova(f"{v}: righe train/held-out = richieste", sum(t.shape[0] for t in tr["domini"].values()) ==
              m["file"]["train"]["righe"] and sum(t.shape[0] for t in ho["domini"].values()) == m["file"]["heldout"]["righe"])
        id_tr = {i for k in tr["esempi"] for i in tr["esempi"][k]}
        id_ho = {i for k in ho["esempi"] for i in ho["esempi"][k]}
        prova(f"{v}: id train ∩ held-out = 0", not (id_tr & id_ho), len(id_tr & id_ho))
        cont = set(E.TESTO_CONTINUO)
        imp = {lato: impronta(tk, x["domini"], x["fonti"], cont) for lato, x in lati.items()}
        a, b = imp["train"], imp["heldout"]
        prova(f"{v}: testo continuo, finestre 32 train ∩ held-out = 0", comuni(a["continuo"], b["continuo"]) == 0,
              comuni(a["continuo"], b["continuo"]))
        prova(f"{v}: messaggi (>= 24 car.) train ∩ held-out = 0", not (a["messaggi"] & b["messaggi"]),
              sorted(a["messaggi"] & b["messaggi"])[:3])
        prova(f"{v}: finestre 32 dei messaggi train ∩ held-out = 0", comuni(a["fin_messaggi"], b["fin_messaggi"]) == 0,
              comuni(a["fin_messaggi"], b["fin_messaggi"]))
        for nome, r in rif.items():
            for lato, im in imp.items():
                c1 = comuni(im["continuo"], r["continuo"])
                c2 = len(im["messaggi"] & r["messaggi"])
                c3 = comuni(im["fin_messaggi"], np.unique(np.concatenate([r["fin_messaggi"], r["continuo"]])))
                prova(f"{v}/{lato} vs {nome}: finestre continuo / messaggi / finestre messaggi = 0",
                      c1 == 0 and c2 == 0 and c3 == 0, f"{c1} / {c2} / {c3}")
        for nome, r in forgia.items():
            c1 = comuni(b["continuo"], r["continuo"])
            c2 = len(b["messaggi"] & r["messaggi"])
            c3 = comuni(b["fin_messaggi"], np.unique(np.concatenate([r["fin_messaggi"], r["continuo"]])))
            prova(f"{v}/heldout vs forgia {nome}: finestre continuo / messaggi / finestre messaggi = 0",
                  c1 == 0 and c2 == 0 and c3 == 0, f"{c1} / {c2} / {c3}")
        prova(f"{v}: manifesto, disgiunzione dai riferimenti ✅", m["disgiunzione_corpus_vs_riferimenti"]["esito"]
              .startswith("✅"), m["disgiunzione_corpus_vs_riferimenti"]["esito"])
    print(f"   file verificati in {time.time() - t0:.0f} s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(E.DIR_E2E))
    ap.add_argument("--solo-unitari", action="store_true")
    ap.add_argument("--esito", default="")
    a = ap.parse_args()
    t0 = time.time()
    unitari()
    if not a.solo_unitari:
        sui_file(Path(a.dir))
    ko = [e for e in ESITI if not e["ok"]]
    print(f"\n{'TUTTO VERDE' if not ko else f'{len(ko)} ROSSI'}: {len(ESITI)} prove in {time.time() - t0:.0f} s")
    if a.esito:
        Path(a.esito).write_text(json.dumps({"prove": ESITI, "rossi": len(ko), "secondi": round(time.time() - t0, 1)},
                                            ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return 1 if ko else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""e2e_onpolicy_rollout — completioni DELLO STUDENTE per l'OPD (arXiv 2609.26708), 12/10/2026. File NUOVO.

Lo studente genera col suo cammino di DEPLOYMENT: un llama-server (motore thor-c0t, GGML_VK_TQ1R_Q8=1) acceso dal
driver sul GGUF del giro. Qui solo HTTP: template di chat -> id (POST /apply-template, /tokenize), poi /completion con
il prompt come LISTA DI ID e `return_tokens` (gli id campionati ESATTI, EOG compreso: server-context.cpp
process_token), cosi' maestro e studente in allenamento vedono gli stessi token generati, senza ri-tokenizzare.

Campionamento del paper: temperatura 1, nessun troncamento (top_k 0, top_p 1, min_p 0, typical 1, nessuna penalita',
DRY/XTC spenti), modo non-thinking (chat_template_kwargs enable_thinking false), G completioni per prompt (4 per la
matematica), 8 prompt a passo dell'ottimizzatore. Lunghezza massima: il paper non la dichiara («decoding budget»);
--max-token 1024 (8x la lunghezza media di riferimento bf16 su GSM8K, ~130 token): i loop restano visibili e pagano
la ricompensa 0.

Dati: GSM8K train (openai/gsm8k, 7473, cache HF locale), permutazione fissa (seme 2609); i primi --held-prompt
dell'ordine sono il HELD-OUT fisso (mai addestrati), gli altri si consumano a giri senza ripetersi (cursore in
--stato). Il test di GSM8K NON si tocca. Disgiunzione dalla sonda parlata (3 domande italiane) controllata.

Sottocomandi:
  genera  --server URL --stato DIR --giro K --out DIR [--righe 32 --righe-passo 4 --righe-held 8 ...]
          completioni train + held-out -> corpus + onpolicy_<lato>.pt (e2e_onpolicy.scrivi_lato) + riassunto.json
  held    --server URL --stato DIR --out DIR      solo il held-out (misura del cancello: ricompensa media, loop, KL libera)
  prova   --out DIR                               SOLO CPU, nessun server: tokenizzatore finto, impacchettamento,
                                                  rilettura col lettore del teacher, piano del teacher (prepara)
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

import e2e_onpolicy as O

GSM8K = Path("@@ROOT:models@@/hf-cache/datasets/openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866/"
             "gsm8k-train.arrow")
SEME_PERM = 2609
ISTRUZIONE = "\nSolve the problem step by step, then write the final answer on the last line as '#### <number>'."
SONDA = ("Un prodotto costa 23 euro", "Alice ha 3 fratelli", "CANE al contrario")
CAMPIONAMENTO = {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0, "typical_p": 1.0,
                 "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0, "dry_multiplier": 0.0,
                 "xtc_probability": 0.0, "mirostat": 0}


def carica_gsm8k(f: Path = GSM8K) -> list:
    import pyarrow as pa
    t = pa.ipc.open_stream(open(f, "rb")).read_all().to_pylist()
    for r in t:
        if any(s in r["question"] for s in SONDA):
            raise RuntimeError("una domanda di GSM8K coincide con la sonda parlata")
    return t


def ordine(n: int) -> list:
    return np.random.default_rng(SEME_PERM).permutation(n).tolist()


# ═══════════════════════════════════════ HTTP ═══════════════════════════════════════
def _post(url: str, corpo: dict, timeout: float = 900.0) -> dict:
    r = urllib.request.Request(url, data=json.dumps(corpo).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as h:
        return json.loads(h.read())


class Server:
    def __init__(self, url: str):
        self.url = url.rstrip("/")

    def prompt_ids(self, domanda: str) -> list:
        t = _post(self.url + "/apply-template", {"messages": [{"role": "user", "content": domanda + ISTRUZIONE}],
                                                  "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
        return _post(self.url + "/tokenize", {"content": t, "add_special": False, "parse_special": True})["tokens"]

    def testo_ids(self, testo: str) -> list:
        return _post(self.url + "/tokenize", {"content": testo, "add_special": False, "parse_special": True})["tokens"]

    def completa(self, ids: list, seme: int, max_token: int) -> dict:
        d = _post(self.url + "/completion", {"prompt": ids, "n_predict": max_token, "seed": seme, "stream": False,
                                             "return_tokens": True, "cache_prompt": False, **CAMPIONAMENTO})
        return {"gen": list(d.get("tokens") or []), "testo": d.get("content", ""), "stop": d.get("stop_type", "?")}


class TokFinto:
    """Solo `prova`: id = byte + 3 (nessun server)."""
    def prompt_ids(self, domanda):
        return [1] + [b + 3 for b in (domanda + ISTRUZIONE).encode()][:400] + [2]

    def testo_ids(self, testo):
        return [b + 3 for b in testo.encode()][:600]

    def completa(self, ids, seme, max_token):
        rng = np.random.default_rng(seme + len(ids))
        n = int(rng.integers(20, max_token))
        return {"gen": rng.integers(3, 259, n).tolist() + [2], "testo": f"... #### {int(rng.integers(0, 4))}",
                "stop": "eos"}


def rip8(gen: list) -> float:
    """Ripetizione (paper: 8-grammi ripetuti): 1 - 8-grammi distinti / 8-grammi."""
    g = [tuple(gen[i:i + 8]) for i in range(max(0, len(gen) - 7))]
    return 0.0 if not g else 1.0 - len(set(g)) / len(g)


def genera_gruppi(srv, dati: list, indici: list, G: int, max_token: int, paralleli: int, seme0: int) -> list:
    """G completioni per prompt (semi seme0 + 1000 * indice + g), ricompensa, vantaggi GRPO per gruppo."""
    lavori = []
    pr = {i: srv.prompt_ids(dati[i]["question"]) for i in indici}
    with cf.ThreadPoolExecutor(paralleli) as ex:
        for i in indici:
            for g in range(G):
                lavori.append((i, g, ex.submit(srv.completa, pr[i], seme0 + 1000 * i + g, max_token)))
        out = []
        for i in indici:
            oro = O.oro_gsm8k(dati[i]["answer"])
            gr = [(g, f.result()) for ii, g, f in lavori if ii == i]
            r = [O.ricompensa(c["testo"], oro) for _, c in gr]
            a = O.vantaggi_grpo(r)
            for (g, c), rr, aa in zip(gr, r, a):
                out.append({"indice": i, "g": g, "prompt": pr[i], "gen": c["gen"], "testo": c["testo"],
                            "stop": c["stop"], "ricompensa": rr, "adv": aa, "rip8": rip8(c["gen"])})
    return out


def ancore(srv, dati: list, indici: list) -> list:
    """Segmenti ANCORA (PIVOT-Q): stati puliti dei dati originali = prompt + soluzione d'oro di GSM8K (calcoli <<...>>
    tolti, '#### N' in fondo come chiede l'istruzione) + fine turno."""
    import re
    out = []
    for i in indici:
        sol = re.sub(r"<<[^>]*>>", "", dati[i]["answer"]).strip()
        out.append({"indice": i, "prompt": srv.prompt_ids(dati[i]["question"]),
                    "gen": srv.testo_ids(sol + "<|im_end|>"), "adv": 0.0, "ancora": True})
    return out


def riassunto(seqs: list) -> dict:
    g = [s for s in seqs if not s.get("ancora")]
    if not g:
        return {}
    return {"completioni": len(g), "ricompensa_media": float(np.mean([s["ricompensa"] for s in g])),
            "lunghezza_media": float(np.mean([len(s["gen"]) for s in g])),
            "troncate": float(np.mean([s["stop"] == "limit" for s in g])),
            "rip8_media": float(np.mean([s["rip8"] for s in g])),
            "gruppi_con_segnale": float(np.mean([s["adv"] != 0 for s in g]))}


def _scrivi_jsonl(p: Path, seqs: list) -> None:
    with open(p, "w") as f:
        for s in seqs:
            f.write(json.dumps({k: v for k, v in s.items()}) + "\n")


def lato_held(srv, dati, perm, a, out: Path, L: int, id_pad: int) -> dict:
    held = perm[:a.held_prompt]
    seqs = genera_gruppi(srv, dati, held, a.G, a.max_token, a.paralleli, a.seme + 7)
    pk = O.impacchetta(seqs, L, id_pad, righe_max=a.righe_held)
    _scrivi_jsonl(out / "rollout_heldout.jsonl", seqs)
    w = O.scrivi_lato(out, "heldout", a.variante, pk, {"lato": "heldout", "prompt": held, "G": a.G})
    return {**w, **riassunto(seqs), "scartate": len(pk["scartate"])}


def cmd_genera(a, srv, solo_held: bool = False) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dati = carica_gsm8k(Path(a.gsm8k))
    perm = ordine(len(dati))
    L, id_pad = a.L, a.id_pad
    t0 = time.time()
    if id_pad is None:
        fine = srv.testo_ids("<|im_end|>")
        if len(fine) != 1:
            raise RuntimeError(f"<|im_end|> non e' un token solo: {fine}")
        id_pad = fine[0]
    rs = {"versione": O.VERSIONE, "giro": a.giro, "campionamento": CAMPIONAMENTO, "max_token": a.max_token, "G": a.G,
          "id_pad": id_pad, "L": L}
    rs["heldout"] = lato_held(srv, dati, perm, a, out, L, id_pad)
    if not solo_held:
        stato = Path(a.stato)
        stato.mkdir(parents=True, exist_ok=True)
        fg = stato / f"giro_{a.giro}.json"
        if fg.exists():                                   # idempotente: un giro rifatto usa GLI STESSI prompt
            cur0 = json.loads(fg.read_text())["cursore_inizio"]
        else:
            fc = stato / "cursore.json"
            cur0 = json.loads(fc.read_text())["cursore"] if fc.exists() else a.held_prompt
        cur, seqs, pk = cur0, [], None
        n_passi = a.righe // a.righe_passo
        while True:
            if cur + a.prompt_passo > len(perm):
                raise RuntimeError("prompt di GSM8K esauriti")
            seqs += genera_gruppi(srv, dati, perm[cur:cur + a.prompt_passo], a.G, a.max_token, a.paralleli, a.seme)
            cur += a.prompt_passo
            pk = O.impacchetta(seqs, L, id_pad)
            if len(pk["ids"]) > a.righe:                  # le prime `righe` righe sono PIENE
                break
        if a.ancora_per_passo > 0:                        # PIVOT-Q: 4 ancore per passo dell'ottimizzatore
            anc = ancore(srv, dati, perm[cur:cur + a.ancora_per_passo * n_passi])
            cur += a.ancora_per_passo * n_passi
            pk0 = O.impacchetta(seqs, L, id_pad, righe_max=a.righe)       # solo le completioni che entrano nel giro
            seqs = [seqs[j] for j in sorted(pk0["usate"])]
            k = max(1, len(seqs) // max(1, len(anc)))
            misto = []
            for j, s in enumerate(seqs):
                misto.append(s)
                if (j + 1) % k == 0 and anc:
                    misto.append(anc.pop(0))
            seqs = misto + anc
        pk = O.impacchetta(seqs, L, id_pad, righe_max=a.righe)
        if len(pk["ids"]) != a.righe:
            raise RuntimeError(f"{len(pk['ids'])} righe invece di {a.righe}")
        _scrivi_jsonl(out / "rollout_train.jsonl", seqs)
        w = O.scrivi_lato(out, "train", a.variante, pk, {"lato": "train", "cursore": [cur0, cur], "G": a.G})
        rs["train"] = {**w, **riassunto([seqs[k] for k in pk["usate"]]), "scartate": len(pk["scartate"]),
                       "ancore": int(sum(1 for k in pk["usate"] if seqs[k].get("ancora")))}
        fg.write_text(json.dumps({"cursore_inizio": cur0, "cursore_fine": cur}))
        (stato / "cursore.json").write_text(json.dumps({"cursore": cur, "giro": a.giro}))
    rs["secondi"] = round(time.time() - t0, 1)
    (out / "riassunto.json").write_text(json.dumps(rs, indent=1))
    print(json.dumps({k: v for k, v in rs.items() if k in ("heldout", "train", "secondi")}))
    return 0


def cmd_prova(a) -> int:
    """SOLO CPU: GSM8K reale, tokenizzatore finto, due lati, rilettura col lettore del teacher, verifica delle
    posizioni, piano del teacher (prepara) in una radice di prova."""
    import torch
    import e2e_teacher as T
    a.righe, a.righe_held, a.righe_passo, a.held_prompt, a.prompt_passo, a.L = 4, 1, 2, 4, 2, 1024
    a.max_token, a.id_pad = 200, 2
    out = Path(a.out)
    a.stato = str(out / "stato")
    a.ancora_per_passo = 1
    cmd_genera(a, TokFinto())
    for lato in ("train", "heldout"):
        c = T.righe_lato(lato, out, a.variante)
        d = torch.load(out / f"onpolicy_{lato}.pt", weights_only=True)
        f = out / (f"e2e_corpus_4096_{a.variante}.pt" if lato == "train" else f"e2e_heldout_4096_{a.variante}.pt")
        dati, firma = O.leggi_onpolicy(out / f"onpolicy_{lato}.pt", f, c["L"], len(c["righe"]), lambda r: c["inizi"][r])
        for r, (t, adv, seg, anc) in dati.items():       # il token in t+1 e' generato: mai un inizio di segmento
            assert not c["inizi"][r][t + 1].any()
        print(f"prova {lato}: {len(c['righe'])} righe da {c['L']}, segmenti {int(c['inizi'].sum())}, posizioni "
              f"{firma['n']} (ancora {firma['ancora']}), sha onpolicy {firma['sha256'][:12]}")
    gg = Path(a.modello)
    T.prepara("prova-opd", ["heldout", "train"], variante=a.variante, modo="segmenti", righe_blocco=2,
              modello=gg, radice=out / "teacher", corpus_dir=out,
              sha_modello=Path(str(gg) + ".sha256").read_text().split()[0])   # sha dichiarato accanto (37 GB: non si rilegge)
    p = json.loads((out / "teacher" / "prova-opd" / "piano.json").read_text())
    print(f"prova piano del teacher: modo {p['modo']}, blocchi {len(p['blocchi'])}, lati {list(p['corpus'])}")
    print("PROVA ROLLOUT OK")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("comando", choices=["genera", "held", "prova"])
    ap.add_argument("--server", default="http://127.0.0.1:11464")
    ap.add_argument("--gsm8k", default=str(GSM8K))
    ap.add_argument("--stato", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--giro", type=int, default=1)
    ap.add_argument("--variante", default="onpolicy")
    ap.add_argument("--righe", type=int, default=32, help="righe da L del lato train (un giro)")
    ap.add_argument("--righe-passo", type=int, default=4, help="righe per passo dell'ottimizzatore (per le ancore)")
    ap.add_argument("--righe-held", type=int, default=8)
    ap.add_argument("--held-prompt", type=int, default=16)
    ap.add_argument("--prompt-passo", type=int, default=8, help="prompt per lotto di generazione (paper: 8)")
    ap.add_argument("--G", type=int, default=4, help="completioni per prompt (paper, matematica: 4)")
    ap.add_argument("--max-token", type=int, default=1024)
    ap.add_argument("--paralleli", type=int, default=8)
    ap.add_argument("--seme", type=int, default=12)
    ap.add_argument("--L", type=int, default=4096)
    ap.add_argument("--id-pad", type=int, default=None)
    ap.add_argument("--ancora-per-passo", type=int, default=0, help="PIVOT-Q: segmenti ancora per passo (paper 4)")
    ap.add_argument("--modello", default="@@ROOT:models@@/gguf/tony-tern/teacher-q8/Ornith-1.5-35B-A3B-Q8_0.gguf",
                    help="prova: GGUF del maestro (solo intestazione: n_embd)")
    a = ap.parse_args(argv)
    if a.comando == "prova":
        return cmd_prova(a)
    srv = Server(a.server)
    if a.comando == "held":
        return cmd_genera(a, srv, solo_held=True)
    if not a.stato:
        ap.error("genera vuole --stato")
    return cmd_genera(a, srv)


if __name__ == "__main__":
    sys.exit(main())

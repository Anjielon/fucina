#!/usr/bin/env python3
"""teacher_client.py — logprob del maestro sulle 4 lettere da un llama-server (llama.cpp).

Per ogni riga del jsonl di e1_build_data.py chiede a /completion UN token con
`n_probs` e legge, fra le probabilita' PRE-campionamento del primo token, quelle di
" A"," B"," C"," D" (identificate per ID di token, ricavati da /tokenize).

Aggiunge alla riga:
  teacher_logprobs : [4] log-softmax rinormalizzato sulle 4 lettere (il bersaglio della KD)
  teacher_raw      : [4] logprob sull'INTERO vocabolario (None se fuori dal top-n)
  teacher_mass     : somma delle 4 probabilita' sull'intero vocabolario (diagnostica)
  teacher_missing  : quante lettere non erano nel top-n (riempite con un pavimento)

Avvio del server (una porta, slot paralleli; ricorda: il ctx si DIVIDE per --parallel):
  llama-server -m teacher.gguf --port 8091 -c 16384 --parallel 4 -ngl 999 --no-webui
Uso:
  python3 teacher_client.py --server http://127.0.0.1:8091 --in data/items.jsonl \
      --out data/items_teacher.jsonl --workers 4
Riprende da dove si era fermato (salta gli id gia' presenti nel file di uscita).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from common import CHOICE_STRINGS, read_jsonl

logger = logging.getLogger("teacher")


class LlamaServerClient:
    """Client minimale (solo libreria standard) per llama-server."""

    def __init__(self, base_url: str, timeout: float = 300.0, api_key: Optional[str] = None):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def _post(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        last: Optional[Exception] = None
        for attempt in range(3):                       # max 3 tentativi
            try:
                req = urllib.request.Request(self.base + path, data=data, headers=self.headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode("utf-8"))
            except Exception as e:  # noqa: BLE001 — 503 (slot occupati), timeout: si ritenta
                last = e
                time.sleep(1.5 * 2 ** attempt)
        raise RuntimeError(f"POST {path} fallito: {last}")

    def tokenize(self, text: str, add_special: bool = False) -> list[int]:
        out = self._post("/tokenize", {"content": text, "add_special": add_special})
        toks = out["tokens"]
        # alcune versioni restituiscono [{"id":..,"piece":..}] con with_pieces
        return [t["id"] if isinstance(t, dict) else int(t) for t in toks]

    def letter_ids(self, prompt: str) -> list[int]:
        """ID dei token " A".." D" ricavati IN CONTESTO: tokenize(prompt + " A") deve essere
        tokenize(prompt) + [UN token]. Tokenizzare " A" da solo e' inaffidabile: con
        SentencePiece (Llama) il prefisso di spazio fittizio da' due token (misurato:
        [259, 29909]) mentre dopo "Answer:" il token vero e' uno solo ("▁A").
        Se la coda non e' un token singolo, lo scoring a token singolo sarebbe falso: errore."""
        base = self.tokenize(prompt, add_special=True)
        ids = []
        for s in CHOICE_STRINGS:
            full = self.tokenize(prompt + s, add_special=True)
            if full[:len(base)] != base or len(full) != len(base) + 1:
                raise RuntimeError(f"confine di token non pulito per {s!r}: coda {full[-4:]} "
                                   f"vs base {base[-3:]}")
            ids.append(full[-1])
        if len(set(ids)) != 4:
            raise RuntimeError(f"id lettere non distinti: {ids}")
        return ids

    def first_token_logprobs(self, prompt: str, n_probs: int) -> dict[int, float]:
        """{id_token: logprob} del primo token generato, dal top-n PRE-campionamento."""
        out = self._post("/completion", {
            "prompt": prompt, "n_predict": 1, "n_probs": n_probs,
            "temperature": 0.0, "cache_prompt": True,
            "post_sampling_probs": False,       # vogliamo il softmax dei logit grezzi
            "return_tokens": False, "stream": False,
        })
        cps = out.get("completion_probabilities") or []
        if not cps:
            raise RuntimeError("risposta senza completion_probabilities (server troppo vecchio?)")
        first = cps[0]
        res: dict[int, float] = {}
        if "top_logprobs" in first:            # formato attuale (2025+)
            for e in first["top_logprobs"]:
                res[int(e["id"])] = float(e["logprob"])
        elif "probs" in first:                 # formato storico: tok_str/prob, senza id
            for e in first["probs"]:
                p = float(e.get("prob", 0.0))
                key = e.get("id")
                if key is None:                # ricaviamo l'id dal testo del token
                    toks = self.tokenize(e["tok_str"])
                    if len(toks) != 1:
                        continue
                    key = toks[0]
                res[int(key)] = math.log(max(p, 1e-30))
        else:
            raise RuntimeError(f"formato completion_probabilities sconosciuto: {list(first)}")
        return res


def letters_from_topn(top: dict[int, float], letter_ids: list[int]) -> tuple[list, list, float, int]:
    """Estrae i 4 logprob; le lettere mancanti prendono un pavimento sotto il minimo visto."""
    raw = [top.get(i) for i in letter_ids]
    missing = sum(r is None for r in raw)
    floor = (min(top.values()) if top else -30.0) - math.log(10.0)   # 10x sotto l'ultimo del top-n
    filled = [r if r is not None else floor for r in raw]
    m = max(filled)
    z = math.log(sum(math.exp(x - m) for x in filled)) + m
    renorm = [x - z for x in filled]
    mass = sum(math.exp(r) for r in raw if r is not None)
    return renorm, raw, mass, missing


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://127.0.0.1:8091")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-probs", type=int, default=100)
    ap.add_argument("--retry-n-probs", type=int, default=1000,
                    help="se manca una lettera si richiede con un top-n piu' largo")
    ap.add_argument("--workers", type=int, default=4, help="= --parallel del server")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cli = LlamaServerClient(args.server, api_key=args.api_key)
    rows = read_jsonl(args.inp)
    if args.limit:
        rows = rows[:args.limit]
    lids = cli.letter_ids(rows[0]["prompt"])
    for r in rows[1:4]:                         # stessi id anche su altri prompt
        if cli.letter_ids(r["prompt"]) != lids:
            raise RuntimeError("gli id delle lettere cambiano fra prompt: tokenizer ambiguo")
    logger.info("id lettere %s — confine di token verificato", lids)

    done = set()
    outp = Path(args.out)
    if outp.exists():
        # un processo interrotto (timeout, kill) puo' lasciare l'ultima riga a meta': si tronca
        # il file all'ultimo a-capo, altrimenti la riga seguente si incollerebbe a quella rotta
        data = outp.read_bytes()
        cut = data.rfind(b"\n") + 1
        if cut < len(data):
            logger.warning("ripresa: tolti %d byte di riga incompleta in coda", len(data) - cut)
            with outp.open("r+b") as fh:
                fh.truncate(cut)
        done = {json.loads(l)["id"] for l in outp.open(encoding="utf-8") if l.strip()}
        logger.info("ripresa: %d righe gia' fatte", len(done))
    todo = [r for r in rows if r["id"] not in done]
    lock = threading.Lock()
    stats = {"n": 0, "missing": 0, "mass": 0.0, "t0": time.time()}

    def work(r: dict) -> dict:
        top = cli.first_token_logprobs(r["prompt"], args.n_probs)
        renorm, raw, mass, miss = letters_from_topn(top, lids)
        if miss and args.retry_n_probs > args.n_probs:
            top = cli.first_token_logprobs(r["prompt"], args.retry_n_probs)
            renorm, raw, mass, miss = letters_from_topn(top, lids)
        return {**r, "teacher_logprobs": renorm, "teacher_raw": raw,
                "teacher_mass": mass, "teacher_missing": miss}

    with outp.open("a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(work, r) for r in todo]
        for fut in as_completed(futs):
            rec = fut.result()
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats["n"] += 1
                stats["missing"] += int(rec["teacher_missing"] > 0)
                stats["mass"] += rec["teacher_mass"]
                if stats["n"] % 500 == 0:
                    f.flush()
                    el = time.time() - stats["t0"]
                    logger.info("%d/%d  %.1f righe/s  massa media lettere %.3f  righe con mancanti %d",
                                stats["n"], len(todo), stats["n"] / el,
                                stats["mass"] / stats["n"], stats["missing"])
    n = max(stats["n"], 1)
    logger.info("fatto: %d righe, massa media sulle 4 lettere %.3f, righe con lettere mancanti %d",
                stats["n"], stats["mass"] / n, stats["missing"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

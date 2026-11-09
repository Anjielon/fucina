#!/usr/bin/env python3
"""e2_collect_hidden.py — raccoglie l'ULTIMO stato nascosto (= ingresso di lm_head, dopo la
norma finale) all'ultima posizione di ogni prompt MMLU-formato degli insiemi ammessi.

Tre motori, stesso output:
  --backend hf        modello transformers; un hook sull'ingresso di lm_head (esatto per
                      costruzione, qualunque architettura). Scrive anche i 4 logprob lettera.
  --backend server    llama-server avviato con `--embeddings --pooling last` -> /embedding
  --backend cli       llama-embedding --pooling last --embd-normalize -1 (nessuna normalizzazione!)

In llama.cpp l'embedding con pooling "last" e' preso da t_embd, che i grafi impostano
DOPO la norma finale e PRIMA di lm_head ("result_norm"): e' proprio h. Va disattivata la
normalizzazione L2 (--embd-normalize -1), altrimenti h viene riscalato.

Uscita:  <out>.npy  float32 [N, d]      <out>.meta.jsonl  (id, group_id, split, gold_pos, perm, ...)

Esempi:
  python3 e2_collect_hidden.py --backend cli --cli-bin llama-embedding -m student.gguf \
      --in data/items.jsonl --out data/hid --max-groups 3000
  python3 e2_collect_hidden.py --backend hf --hf-model /percorso/modello --in data/items.jsonl \
      --out data/hid --max-groups 3000
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

from common import read_jsonl, stable_int, write_jsonl

logger = logging.getLogger("collect")
SEP = "<|§§SEP§§|>"          # separatore che non compare mai nei prompt


def pick_rows(rows: list[dict], max_groups_train: int, max_groups_heldout: int,
              seed: int) -> list[dict]:
    """Sceglie gruppi INTERI (4 rotazioni), separatamente per train e heldout."""
    by_split: dict[str, dict[str, list[dict]]] = {}
    for r in rows:
        by_split.setdefault(r["split"], {}).setdefault(r["group_id"], []).append(r)
    out = []
    for split, cap in (("train", max_groups_train), ("heldout", max_groups_heldout)):
        groups = {g: v for g, v in by_split.get(split, {}).items() if len(v) == 4}
        keys = sorted(groups, key=lambda g: stable_int(f"{seed}/{g}"))
        if cap:
            keys = keys[:cap]
        for k in keys:
            out.extend(sorted(groups[k], key=lambda r: r["rot"]))
    return out


# ── Motore transformers ──────────────────────────────────────────────────────
def collect_hf(rows: list[dict], model_path: str, device: str, batch: int,
               dtype: str) -> tuple[np.ndarray, list[list[float]]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from e1_letter_kd import hf_letter_ids

    tok = AutoTokenizer.from_pretrained(model_path)
    tok.padding_side = "left"                       # ultima posizione = -1 per tutti
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=getattr(torch, dtype))
    model.to(device).eval()
    lids = torch.tensor(hf_letter_ids(tok, rows[0]["prompt"]), device=device)
    captured: list = []
    head = model.get_output_embeddings()
    hook = head.register_forward_pre_hook(lambda m, inp: captured.append(inp[0][:, -1, :].float().cpu()))
    H, L = [], []
    with torch.inference_mode():
        for i in range(0, len(rows), batch):
            enc = tok([r["prompt"] for r in rows[i:i + batch]], return_tensors="pt", padding=True).to(device)
            captured.clear()
            try:
                out = model(**enc, logits_to_keep=1, use_cache=False)
            except TypeError:
                out = model(**enc, use_cache=False)
            H.append(captured[-1].numpy())
            lp = torch.log_softmax(out.logits[:, -1, :].float(), -1).index_select(-1, lids)
            L.extend(lp.cpu().tolist())
            if (i // batch) % 20 == 0:
                logger.info("hf %d/%d", i + len(rows[i:i + batch]), len(rows))
    hook.remove()
    return np.concatenate(H).astype(np.float32), L


# ── Motore llama-server /embedding ───────────────────────────────────────────
def collect_server(rows: list[dict], url: str, workers: int) -> np.ndarray:
    from concurrent.futures import ThreadPoolExecutor

    def one(prompt: str) -> list[float]:
        # embd_normalize=-1: nessuna normalizzazione (il default L2 riscalerebbe h)
        data = json.dumps({"content": prompt, "embd_normalize": -1}).encode()
        req = urllib.request.Request(url.rstrip("/") + "/embedding", data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            out = json.loads(r.read())
        # formati: [{"index":0,"embedding":[[...]]}] oppure {"embedding":[...]}
        if isinstance(out, list):
            out = out[0]
        emb = out["embedding"]
        if emb and isinstance(emb[0], list):       # pooling none -> per token: prendiamo l'ultimo
            emb = emb[-1]
        return emb

    with ThreadPoolExecutor(workers) as ex:
        embs = list(ex.map(one, [r["prompt"] for r in rows]))
    return np.asarray(embs, dtype=np.float32)


# ── Motore llama-embedding (CLI) ─────────────────────────────────────────────
def collect_cli(rows: list[dict], cli_bin: str, model: str, chunk: int, ctx: int,
                ngl: int, extra: list[str]) -> np.ndarray:
    """Un processo per blocco di `chunk` prompt: carica il modello ogni volta, quindi blocchi
    grandi (centinaia). -ub deve contenere il prompt piu' lungo (il pooling lo richiede)."""
    out = []
    env = dict(os.environ)
    bindir = str(Path(cli_bin).resolve().parent)
    env["LD_LIBRARY_PATH"] = bindir + ":" + env.get("LD_LIBRARY_PATH", "")
    for i in range(0, len(rows), chunk):
        part = rows[i:i + chunk]
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(SEP.join(r["prompt"] for r in part))
            fname = f.name
        cmd = [cli_bin, "-m", model, "-f", fname, "--pooling", "last", "--embd-normalize", "-1",
               "--embd-separator", SEP, "--embd-output-format", "json",
               "-c", str(ctx), "-b", str(ctx), "-ub", str(ctx), "-ngl", str(ngl), *extra]
        res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=7200)
        os.unlink(fname)
        if res.returncode != 0:
            raise RuntimeError(f"llama-embedding fallito ({res.returncode}): {res.stderr[-2000:]}")
        js = json.loads(res.stdout[res.stdout.index("{"):])
        data = sorted(js["data"], key=lambda d: d["index"])
        if len(data) != len(part):
            raise RuntimeError(f"attesi {len(part)} embedding, ricevuti {len(data)} "
                               "(un prompt contiene il separatore? -ub troppo piccolo?)")
        out.extend(d["embedding"] for d in data)
        logger.info("cli %d/%d", i + len(part), len(rows))
    return np.asarray(out, dtype=np.float32)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["hf", "server", "cli"], required=True)
    ap.add_argument("--in", dest="inp", required=True, help="jsonl di e1_build_data.py")
    ap.add_argument("--out", required=True, help="prefisso: scrive <out>.npy e <out>.meta.jsonl")
    ap.add_argument("--max-groups", type=int, default=3000, help="gruppi train (4 righe l'uno)")
    ap.add_argument("--max-groups-heldout", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    # hf
    ap.add_argument("--hf-model")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch", type=int, default=8)
    # server
    ap.add_argument("--server", default="http://127.0.0.1:8092")
    ap.add_argument("--workers", type=int, default=4)
    # cli
    ap.add_argument("--cli-bin", default="llama-embedding")
    ap.add_argument("-m", "--model")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--ngl", type=int, default=999)
    ap.add_argument("--cli-extra", default="", help="argomenti aggiuntivi, separati da spazio")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    rows = pick_rows(read_jsonl(args.inp), args.max_groups, args.max_groups_heldout, args.seed)
    if any(SEP in r["prompt"] for r in rows):
        raise SystemExit("un prompt contiene il separatore: cambiarlo")
    logger.info("righe da raccogliere: %d", len(rows))
    letter_lp = None
    if args.backend == "hf":
        H, letter_lp = collect_hf(rows, args.hf_model, args.device, args.batch, args.dtype)
    elif args.backend == "server":
        H = collect_server(rows, args.server, args.workers)
    else:
        H = collect_cli(rows, args.cli_bin, args.model, args.chunk, args.ctx, args.ngl,
                        args.cli_extra.split())
    np.save(args.out + ".npy", H)
    keep = ("id", "group_id", "source", "split", "rot", "perm", "gold_pos", "gold", "teacher_logprobs")
    meta = []
    for i, r in enumerate(rows):
        m = {k: r[k] for k in keep if k in r}
        if letter_lp is not None:
            m["student_letter_logprobs"] = letter_lp[i]
        meta.append(m)
    write_jsonl(args.out + ".meta.jsonl", meta)
    logger.info("salvati %s.npy %s e %s.meta.jsonl; norma media di h %.3f", args.out, H.shape,
                args.out, float(np.linalg.norm(H, axis=1).mean()))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""test_gguf_cpu.py — collaudo (c) del GGUF scritto da e2_scrivi_gguf.py, SOLO CPU.

1. llama-server (thor-velocita) su CPU (-ngl 0, --device none, GGML_VK_VISIBLE_DEVICES vuoto:
   il backend Vulkan non vede nessun dispositivo) col modello ORIGINALE, --embeddings
   --pooling last: h = ingresso di lm_head all'ultima posizione (embd_normalize -1).
2. stesso server senza --embeddings, ORIGINALE: logprob delle 4 lettere (/completion, n_probs).
3. idem col GGUF MODIFICATO.
Predizione (logica di e2_verifica_h.py): z_lettera = W_lettere . h, con W = somma dei piani di
output.* letta dal .modifica.json (W_letters_original / W_letters_gguf). Si confronta
  misurato  d_L = [lp_mod - logsumexp4(lp_mod)] - [lp_orig - logsumexp4(lp_orig)]
  predetto  d_L = log_softmax4(z_orig + (W_gguf - W_orig) h) - log_softmax4(z_orig)
e anche la ricostruzione assoluta log_softmax4(W_orig h) contro il server originale.
I server sono processi lanciati QUI e fermati QUI (terminate del proprio Popen).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

import numpy as np

from common import read_jsonl
from teacher_client import LlamaServerClient

B = "@@ROOT:models@@/build/thor-velocita/build/bin"


def start(model: str, port: int, threads: int, emb: bool, log: str) -> subprocess.Popen:
    env = dict(os.environ, LD_LIBRARY_PATH=B, GGML_VK_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="-1",
               CUDA_VISIBLE_DEVICES="-1")
    cmd = ["nice", "-n", "19", f"{B}/llama-server", "-m", model, "--port", str(port), "-ngl", "0",
           "--device", "none", "-t", str(threads), "-tb", str(threads), "-c", "1024", "-np", "1",
           "-b", "512", "-ub", "512", "--no-webui"]
    if emb:
        cmd += ["--embeddings", "--pooling", "last"]
    p = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env)
    for _ in range(900):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                if b'"ok"' in r.read():
                    return p
        except Exception:  # noqa: BLE001 — server non ancora su
            pass
        if p.poll() is not None:
            raise SystemExit(f"server uscito: {open(log).read()[-800:]}")
        time.sleep(2)
    p.terminate()
    raise SystemExit("server non pronto in 30 min")


def stop(p: subprocess.Popen) -> None:
    if p.poll() is None:
        p.terminate()
        try:
            p.wait(60)
        except subprocess.TimeoutExpired:
            p.kill()


def embed(port: int, prompts: list[str]) -> np.ndarray:
    out = []
    for s in prompts:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/embedding",
                                     data=json.dumps({"content": s, "embd_normalize": -1}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as r:
            e = json.loads(r.read())
        e = e[0]["embedding"] if isinstance(e, list) else e["embedding"]
        out.append(e[-1] if isinstance(e[0], list) else e)
    return np.asarray(out, dtype=np.float64)


def letter_lp(port: int, prompts: list[str], n_probs: int) -> tuple[np.ndarray, list[int]]:
    cli = LlamaServerClient(f"http://127.0.0.1:{port}", timeout=1800)
    lids = cli.letter_ids(prompts[0])
    rows = []
    for s in prompts:
        top = cli.first_token_logprobs(s, n_probs)
        miss = [i for i in lids if i not in top]
        if miss:
            raise SystemExit(f"lettere {miss} fuori dal top-{n_probs}")
        rows.append([top[i] for i in lids])
    return np.asarray(rows), lids


def ls4(z: np.ndarray) -> np.ndarray:
    m = z.max(1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(1, keepdims=True))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orig", required=True)
    ap.add_argument("--mod", required=True, help="GGUF modificato (accanto: .modifica.json)")
    ap.add_argument("--items", required=True)
    ap.add_argument("--n", type=int, default=8, help="prompt (i piu' corti dell'heldout)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--port", type=int, default=11489)
    ap.add_argument("--n-probs", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rep = json.load(open(a.mod + ".modifica.json"))
    Wo, Wg = np.asarray(rep["W_letters_original"]), np.asarray(rep["W_letters_gguf"])
    items = [r for r in read_jsonl(a.items) if r.get("split") == "heldout"]
    items.sort(key=lambda r: len(r["prompt"]))
    prompts = [r["prompt"] for r in items[:a.n]]
    logd = os.path.dirname(a.out)
    res: dict = {"n": len(prompts), "prompt_chars": [len(p) for p in prompts]}
    t0 = time.time()
    p = start(a.orig, a.port, a.threads, True, f"{logd}/srv_emb.log")
    try:
        H = embed(a.port, prompts)
    finally:
        stop(p)
    res["t_emb_s"] = round(time.time() - t0)
    t0 = time.time()
    p = start(a.orig, a.port, a.threads, False, f"{logd}/srv_orig.log")
    try:
        lpo, lids = letter_lp(a.port, prompts, a.n_probs)
    finally:
        stop(p)
    res["t_orig_s"] = round(time.time() - t0)
    t0 = time.time()
    p = start(a.mod, a.port, a.threads, False, f"{logd}/srv_mod.log")
    try:
        lpm, lids2 = letter_lp(a.port, prompts, a.n_probs)
    finally:
        stop(p)
    res["t_mod_s"] = round(time.time() - t0)
    assert lids == lids2 == rep["letter_ids"], (lids, lids2, rep["letter_ids"])
    zo = H @ Wo.T
    meas = ls4(lpm) - ls4(lpo)
    pred = ls4(zo + H @ (Wg - Wo).T) - ls4(zo)
    res.update({
        "letter_ids": lids, "h_norm_mean": float(np.linalg.norm(H, axis=1).mean()),
        "recon_orig_max_abs": float(np.abs(ls4(zo) - ls4(lpo)).max()),
        "recon_orig_argmax_agree": float((zo.argmax(1) == lpo.argmax(1)).mean()),
        "delta_measured": meas.tolist(), "delta_predicted": pred.tolist(),
        "delta_max_abs_err": float(np.abs(meas - pred).max()),
        "delta_measured_max_abs": float(np.abs(meas).max()),
        "delta_corr": float(np.corrcoef(meas.ravel(), pred.ravel())[0, 1]) if np.abs(meas).max() > 0 else None,
    })
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("delta_m") or k.endswith("abs")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

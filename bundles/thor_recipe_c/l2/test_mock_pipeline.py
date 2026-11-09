#!/usr/bin/env python3
"""test_mock_pipeline.py — collaudo SOLO CPU della catena lettera, senza llama-server e senza GPU.

Un finto llama-server (http.server, 127.0.0.1) risponde a /health, /tokenize, /completion
(top_logprobs del primo token) e /embedding (h deterministico, d=2048), usando il VERO
tokenizer di Thor (Qwen, vocabolario 248.320). Poi esegue, come farebbe coda_lettera.sh:
  1. teacher_client.py   su poche righe del jsonl vero
  2. e2_collect_hidden.py --backend server su pochi gruppi (train + heldout)
  3. e2_bnp_edit.py all  con lm_head VERA di R2B3m (tipo TQ1_0R_1024), --rows letters, rank1 e bnp
  4. e1_letter_kd.py     innesto su un modello Qwen MINUSCOLO a pesi casuali (CPU, pochi passi)

Uso: HIP_VISIBLE_DEVICES=-1 python3 test_mock_pipeline.py --items ITEMS.jsonl --work DIR
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

QUI = Path(__file__).resolve().parent
TOK_DIR = "@@ROOT:models@@/gguf/thor-src-hf"
R2B3M = "@@ROOT:models@@/gguf/tony-tern/R2B3m/Thor-T-R2B3m.gguf"
D = 2048


def make_server(tok, port: int) -> ThreadingHTTPServer:
    rng_letters = None

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silenzio
            pass

        def _send(self, obj, code=200):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            self._send({"status": "ok"})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/tokenize":
                self._send({"tokens": tok.encode(req["content"], add_special_tokens=False)})
            elif self.path == "/completion":
                p = req["prompt"]
                seed = abs(hash(p)) % (2 ** 32)
                r = np.random.default_rng(seed)
                ids = r.choice(248320, size=req["n_probs"], replace=False).tolist()
                # le lettere: tre su quattro nel top-n; a volte manca "D" (prova il ripiego)
                letters = [357, 417, 351, 414]
                drop = req["n_probs"] < 200 and seed % 7 == 0
                for i, L in enumerate(letters):
                    if drop and i == 3:
                        continue
                    ids[i] = L
                lp = np.sort(r.normal(-3, 2, len(ids)))[::-1]
                top = [{"id": int(t), "logprob": float(v), "token": "x"} for t, v in zip(ids, lp)]
                self._send({"content": " A", "completion_probabilities": [
                    {"id": int(ids[0]), "logprob": float(lp[0]), "top_logprobs": top}]})
            elif self.path == "/embedding":
                assert req.get("embd_normalize") == -1, "il client deve chiedere embd_normalize=-1"
                r = np.random.default_rng(abs(hash(req["content"])) % (2 ** 32))
                v = (r.standard_normal(D) * 3).astype(np.float32).tolist()
                self._send([{"index": 0, "embedding": [v]}])
            else:
                self._send({"error": "?"}, 404)

    return ThreadingHTTPServer(("127.0.0.1", port), H)


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=QUI)
    if r.returncode != 0:
        raise SystemExit(f"FALLITO rc={r.returncode}: {' '.join(cmd)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--port", type=int, default=18931)
    a = ap.parse_args()
    os.environ.setdefault("PYTHONHASHSEED", "0")
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    srv = make_server(tok, a.port)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{a.port}"
    py = sys.executable

    # campione: 30 gruppi train + 10 heldout (gruppi interi, 4 rotazioni)
    rows = [json.loads(l) for l in open(a.items, encoding="utf-8")]
    want = {"train": 30, "heldout": 10}
    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(r["group_id"], []).append(r)
    small = []
    for g, v in groups.items():
        sp = v[0]["split"]
        if want.get(sp, 0) > 0 and len(v) == 4:
            small.extend(v)
            want[sp] -= 1
        if not any(want.values()):
            break
    items = work / "items_small.jsonl"
    items.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in small))

    # 1) maestro (finto) — e ripresa: seconda passata deve saltare tutto
    teach = work / "items_teacher.jsonl"
    if teach.exists():
        teach.unlink()
    run([py, "teacher_client.py", "--server", url, "--in", str(items), "--out", str(teach), "--workers", "4"])
    run([py, "teacher_client.py", "--server", url, "--in", str(items), "--out", str(teach), "--workers", "4"])
    tr = [json.loads(l) for l in open(teach)]
    assert len(tr) == len(small), (len(tr), len(small))
    assert all(len(r["teacher_logprobs"]) == 4 for r in tr)
    assert abs(sum(np.exp(tr[0]["teacher_logprobs"])) - 1) < 1e-6
    # ripresa dopo un'interruzione a meta' riga (timeout/kill): nessuna riga corrotta
    cut = work / "items_teacher_cut.jsonl"
    raw = teach.read_bytes()
    cut.write_bytes(raw[:raw.find(b"\n", 4000) + 1] + b'{"id": "rotta a meta')
    run([py, "teacher_client.py", "--server", url, "--in", str(items), "--out", str(cut), "--workers", "4"])
    rc = [json.loads(l) for l in open(cut)]
    assert len(rc) == len(small) and len({r["id"] for r in rc}) == len(small), len(rc)

    # 2) stati nascosti (finto server /embedding)
    hid = work / "hid"
    run([py, "e2_collect_hidden.py", "--backend", "server", "--server", url, "--in", str(teach),
         "--out", str(hid), "--max-groups", "30", "--max-groups-heldout", "10", "--workers", "4"])
    H = np.load(str(hid) + ".npy")
    assert H.shape == (160, D), H.shape

    # 3) modifica con lm_head VERA di R2B3m (solo le 4 righe lettera)
    for meth in (["--method", "rank1"], ["--method", "bnp", "--k", "32", "--align", "abs"]):
        out = work / ("e2_" + meth[1])
        run([py, "e2_bnp_edit.py", "all", "--hidden", str(hid), "--w", "gguf:" + R2B3M,
             "--letter-ids", "357,417,351,414", *meth, "--planes", "3", "--rows", "letters",
             "--save-planes", "--out", str(out)])
        rep = json.loads((out / "report.json").read_text())
        assert set(rep["eval_heldout"]) == {"original", "original_ternary", "edited_float", "edited_ternary"}
        if meth[1] == "rank1":
            assert rep["recon_edited"]["bias_leak_ratio"] < 1.0

    # 3b) cancello h-vs-server: col finto server h e logprob sono scorrelati -> deve RIFIUTARE (rc 3)
    r = subprocess.run([py, "e2_verifica_h.py", "--hidden", str(hid), "--check", str(teach),
                        "--w", "gguf:" + R2B3M, "--letter-ids", "357,417,351,414",
                        "--out", str(work / "verifica_h.json")], cwd=QUI)
    assert r.returncode == 3, f"il cancello doveva rifiutare h casuali (rc={r.returncode})"

    # 4) innesto KD su modello minuscolo (CPU)
    tiny = work / "tiny_qwen"
    if not (tiny / "config.json").exists():
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
        cfg = AutoConfig.for_model("qwen2", vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                                   num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                                   max_position_embeddings=4096, tie_word_embeddings=True)
        torch.manual_seed(0)
        m = AutoModelForCausalLM.from_config(cfg)
        m.save_pretrained(tiny)
        tok.save_pretrained(tiny)
    run([py, "e1_letter_kd.py", "--hf-model", str(tiny), "--data", str(teach), "--device", "cpu",
         "--steps", "6", "--last-n", "2", "--groups-per-batch", "2", "--target", "debiased"])
    srv.shutdown()
    print("MOCK PIPELINE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

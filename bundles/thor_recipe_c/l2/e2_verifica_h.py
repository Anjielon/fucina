#!/usr/bin/env python3
"""e2_verifica_h.py — cancello: gli h raccolti sono DAVVERO l'ingresso di lm_head dello studente?

Confronta, sulle stesse righe, i 4 logprob lettera ricostruiti come log_softmax(W_lettere . h)
(h da e2_collect_hidden.py, W dal GGUF) con quelli che lo STESSO modello da' via /completion
(teacher_client.py puntato al server dello studente, rinormalizzati sulle 4 lettere).
Il README avverte: in llama.cpp lm_head lavora su attivazioni quantizzate, quindi la
differenza non e' zero (0,05-0,09 misurati su TinyLlama); un h sbagliato (normalizzato L2,
preso dallo strato sbagliato) da' invece accordo d'argmax vicino al caso (~0,25-0,4).

Uso:
  python3 e2_verifica_h.py --hidden data/hid --check data/student_check.jsonl \
      --w gguf:STUDENT.gguf --letter-ids 357,417,351,414 --out out/verifica_h.json
Esce con 0 se l'accordo d'argmax >= --min-agree, altrimenti 3 (il chiamante decide).
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch

from common import read_jsonl
from e2_bnp_edit import load_hidden, load_matrix


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hidden", required=True)
    ap.add_argument("--check", required=True, help="uscita di teacher_client.py sul server dello studente")
    ap.add_argument("--w", required=True)
    ap.add_argument("--letter-ids", required=True)
    ap.add_argument("--min-agree", type=float, default=0.85)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    H, meta = load_hidden(a.hidden)
    pos = {m["id"]: i for i, m in enumerate(meta)}
    chk = [r for r in read_jsonl(a.check) if r["id"] in pos]
    if not chk:
        raise SystemExit("nessuna riga in comune fra --check e gli h raccolti")
    lids = [int(x) for x in a.letter_ids.split(",")]
    Wl = load_matrix(a.w, rows=lids) if a.w.startswith("gguf:") else load_matrix(a.w)[lids]
    h = torch.from_numpy(H[[pos[r["id"]] for r in chk]])
    z = torch.log_softmax(h @ Wl.T, -1)
    t = torch.tensor([r["teacher_logprobs"] for r in chk])
    res = {"n": len(chk), "argmax_agree": float((z.argmax(-1) == t.argmax(-1)).float().mean()),
           "max_abs_logprob_diff": float((z - t).abs().max()),
           "mean_abs_logprob_diff": float((z - t).abs().mean()),
           "h_mean_norm": float(np.linalg.norm(H, axis=1).mean()),
           "min_agree": a.min_agree}
    res["ok"] = res["argmax_agree"] >= a.min_agree
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res))
    return 0 if res["ok"] else 3


if __name__ == "__main__":
    sys.exit(main())

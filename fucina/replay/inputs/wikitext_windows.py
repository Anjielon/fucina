"""Distillation corpus of the CARDAN paper (arXiv 2609.21137, section 5.1): 10,000 random 256-token windows.

Recorded run: ``CARDAN_FEDELE/corpus`` (10 Oct 2026). Train: 10,000 windows of the WikiText-103 (raw v1) train
split; held-out (step selection): 64 windows of the validation split. Window starts are drawn with
``random.Random(20261010)`` over the token stream of the split (tokenized in ~4 MB chunks cut at line ends).
Contamination check against the full MMLU test split (60-character prefixes and 13-grams of normalized text): the
run stops if any is found.

Outputs, in the corpus format of the teacher-cache tool (rows of 4096 tokens = 16 independent 256-token segments):
  e2e_corpus_4096_wt103.pt   625 rows      recorded sha256 6b70ad22d8b9046dcbf0821c7d0dd85bddb96e4990b0d6e39119ffcfa96fb056
  e2e_heldout_4096_wt103.pt  4 rows        recorded sha256 c26bc437f994306beffcb6bb9328a1994ebddeb4bd83b7fc79575eb7c30869ef
  wt103_manifest.json
The dictionary keys of the ``.pt`` files (``l_seq``, ``nomi_domini``, ``domini``, ``cu``, ``cu_ptr``, ``ordine``) are the
data format read by the teacher-cache and training code, kept as recorded.

Usage: python -m fucina.replay.inputs.wikitext_windows --wikitext DIR --mmlu-test FILE --tokenizer DIR --out OUT
(DIR holds train-0000{0,1}-of-00002.parquet and validation-00000-of-00001.parquet of wikitext-103-raw-v1.)
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

SEQ, N_TRAIN, N_HELD, ROW, SEED = 256, 10000, 64, 4096, 20261010


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


CHUNK = 4_000_000


def text_rows(files):
    """The ``text`` column of parquet files, row by row, without loading a file at once."""
    import pyarrow.parquet as pq
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=4096, columns=["text"]):
            yield from batch.column(0).to_pylist()


def chunks(rows):
    """The chunks of the recorded tokenization: from position a, cut after the first newline at or beyond
    a + 4,000,000 characters of the concatenated text (or at its end). Same cuts as on the whole string."""
    pieces, size, need = [], 0, CHUNK + 1
    for r in rows:
        pieces.append(r)
        size += len(r)
        while size >= need:
            buf = "".join(pieces)
            b = buf.find("\n", CHUNK)
            if b < 0:
                pieces, need = [buf], size + 65536      # no newline yet: read more before looking again
                break
            yield buf[:b + 1]
            rest = buf[b + 1:]
            pieces, size, need = [rest], len(rest), CHUNK + 1
    buf = "".join(pieces)
    while buf:
        b = buf.find("\n", CHUNK)
        b = len(buf) if b < 0 else b + 1
        yield buf[:b]
        buf = buf[b:]


def token_stream(tok, files, store: Path):
    """Tokenize the split chunk by chunk into an int32 file on disk; returns a read-only memmap of the ids."""
    import numpy as np
    n = 0
    with open(store, "wb") as f:
        for c in chunks(text_rows(files)):
            ids = np.asarray(tok(c, add_special_tokens=False)["input_ids"], dtype=np.int32)
            ids.tofile(f)
            n += len(ids)
    return np.memmap(store, dtype=np.int32, mode="r", shape=(n,))


def windows(ids, n: int, rng: random.Random):
    starts = sorted(rng.sample(range(0, len(ids) - SEQ), n))
    return [ids[s:s + SEQ].tolist() for s in starts], int(len(ids))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--wikitext", required=True, type=Path)
    ap.add_argument("--mmlu-test", required=True, type=Path, help="cais/mmlu all/test-00000-of-00001.parquet")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    a.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    store = a.out / "_tokens.i32"                 # on disk, next to the outputs: peak RAM stays well under 2 GB
    ids_train = token_stream(tok, sorted(a.wikitext.glob("train-*.parquet")), store)
    sel_train, n_train = windows(ids_train, N_TRAIN, rng)
    del ids_train
    ids_valid = token_stream(tok, [a.wikitext / "validation-00000-of-00001.parquet"], store)
    sel_held, n_valid = windows(ids_valid, N_HELD, rng)
    del ids_valid
    store.unlink()
    texts = [tok.decode(x) for x in sel_train + sel_held]
    questions = pd.read_parquet(a.mmlu_test)["question"].tolist()
    everything = normalize(" ".join(texts))
    prefixes = [normalize(x)[:60] for x in questions if len(normalize(x)) >= 40]
    hit_prefix = sum(1 for p in prefixes if p in everything)
    grams = set()
    for x in questions:
        w = normalize(x).split()
        grams.update(tuple(w[i:i + 13]) for i in range(len(w) - 12))
    words = everything.split()
    hit_13 = sum(1 for i in range(len(words) - 12) if tuple(words[i:i + 13]) in grams)
    report = {"seq": SEQ, "n_train": N_TRAIN, "n_held": N_HELD, "seed": SEED, "tokens_train_split": n_train,
              "tokens_validation_split": n_valid, "mmlu_test_questions": len(questions),
              "mmlu_60char_prefixes_found": hit_prefix, "mmlu_13grams_found": hit_13,
              "paper": "arXiv 2609.21137 section 5.1: 10,000 256-token windows (WikiText-103)"}
    if hit_prefix or hit_13:
        (a.out / "wt103_manifest.json").write_text(json.dumps(report, indent=1))
        raise SystemExit("MMLU contamination found in the WikiText windows")
    per = ROW // SEQ
    for name, xs in (("corpus", sel_train), ("heldout", sel_held)):
        rows = [sum(xs[i:i + per], []) for i in range(0, len(xs) - len(xs) % per, per)]
        X = torch.tensor(np.array(rows, dtype=np.int32))
        cu_row = torch.arange(0, ROW + 1, SEQ, dtype=torch.int32)
        d = {"l_seq": ROW, "nomi_domini": ["wt103"], "domini": {"wt103": X},
             "cu": {"wt103": cu_row.repeat(X.shape[0])},
             "cu_ptr": {"wt103": torch.arange(0, (per + 1) * (X.shape[0] + 1), per + 1, dtype=torch.int64)},
             "ordine": torch.tensor([[0, r] for r in range(X.shape[0])], dtype=torch.int64)}
        torch.save(d, a.out / f"e2e_{name}_4096_wt103.pt")
        report[f"rows_{name}"] = int(X.shape[0])
    (a.out / "wt103_manifest.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Answer-guided calibration set from MMLU validation + dev, disjoint from the test split and from the bench.

Recorded run: ``SELRISP`` (10 Oct 2026), 1,779 questions. The chain uses it for the answer-token KL (KL4) of the
distillation steps (896 questions for calibration, the rest held out) and for the K-FAC factors of the OBQ step.

Sources: ``cais/mmlu`` config ``all``, splits ``validation`` (1,531) and ``dev`` (285), never ``test``. Each row
uses the bench prompt (lm-eval letters, zero-shot). A question is dropped when it does not have four choices, when
its normalized question+choices hash equals that of ANY of the 14,042 test questions, when its prompt equals a bench
prompt, or when it repeats an earlier question.

Outputs in OUT:
  calib_mmlu.bin      task file for ``llama-perplexity --multiple-choice`` (the teacher's per-question log-probs)
  calib_mmlu.jsonl    one row per question: i, split, subject, label, ids (prompt token ids), lettere (token ids of
                      " A".." D"). The key names are the data format read by the distillation code, kept as recorded.
  calib_manifest.json counts, drops and sha256 of the .bin

Recorded hashes: calib_mmlu.bin 226532c016d227ae656da383f75d2a07cb98582bf52b848eed303f5370c8ffab,
calib_mmlu.jsonl cc773ec667d59bfafb9d2e11ed5af13a49bd050172cec6412a83d20aa78e0dbd.

Usage: python -m fucina.replay.inputs.mmlu_calibration --mmlu DIR --bench mmlu_00000.bin --tokenizer DIR --out OUT
(DIR of --mmlu holds all/{test,validation,dev}-00000-of-00001.parquet of cais/mmlu at the pinned revision.)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import multiple_choice as mc


def normalized_hash(question: str, choices) -> str:
    q = " ".join(question.split())
    c = "\x1f".join(" ".join(x.split()) for x in choices)
    return hashlib.sha256((q + "\x1f" + c).encode()).hexdigest()


def read_split(mmlu_dir: Path, split: str) -> list[dict]:
    import pyarrow.parquet as pq
    t = pq.read_table(Path(mmlu_dir) / "all" / f"{split}-00000-of-00001.parquet").to_pylist()
    return [{"question": r["question"], "subject": r["subject"], "choices": list(r["choices"]),
             "answer": int(r["answer"])} for r in t]


def select(mmlu_dir: Path, bench: Path) -> tuple[list, dict, int, int]:
    test_hashes = {normalized_hash(r["question"], r["choices"]) for r in read_split(mmlu_dir, "test")}
    bench_prompts = mc.read_questions(bench)
    bench_hashes = {hashlib.sha256(p.encode()).hexdigest() for p in bench_prompts}
    rows, drops, seen = [], {"non4": 0, "in_test": 0, "in_bench": 0, "dup": 0}, set()
    for split in ("validation", "dev"):
        for r in read_split(mmlu_dir, split):
            if len(r["choices"]) != 4:
                drops["non4"] += 1
                continue
            h = normalized_hash(r["question"], r["choices"])
            if h in test_hashes:
                drops["in_test"] += 1
                continue
            p = mc.mmlu_prompt(r["subject"], r["question"], r["choices"])
            if hashlib.sha256(p.encode()).hexdigest() in bench_hashes:
                drops["in_bench"] += 1
                continue
            if h in seen:
                drops["dup"] += 1
                continue
            seen.add(h)
            rows.append((split, r["subject"], p, r["answer"]))
    return rows, drops, len(test_hashes), len(bench_prompts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mmlu", required=True, type=Path, help="cais/mmlu snapshot directory (parquet)")
    ap.add_argument("--bench", required=True, type=Path, help="MMLU-2000 bench task file (mmlu_00000.bin)")
    ap.add_argument("--tokenizer", required=True, help="tokenizer directory (the bf16 model)")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    a.out.mkdir(parents=True, exist_ok=True)
    rows, drops, n_test, n_bench = select(a.mmlu, a.bench)
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    letters = [tok(" " + c, add_special_tokens=False)["input_ids"] for c in "ABCD"]
    if not all(len(x) == 1 for x in letters):
        raise SystemExit(f"answer letters are not single tokens: {letters}")
    letters = [x[0] for x in letters]
    tasks, n_tokens = [], 0
    with open(a.out / "calib_mmlu.jsonl", "w") as f:
        for i, (split, subject, prompt, label) in enumerate(rows):
            ids = tok(prompt, add_special_tokens=True)["input_ids"]
            full = tok(prompt + " A", add_special_tokens=True)["input_ids"]
            if full != ids + [letters[0]]:      # the engine scores tokenize(prompt + " " + letter)
                raise SystemExit(f"row {i}: the answer letter is not a clean extra token")
            n_tokens += len(ids)
            f.write(json.dumps({"i": i, "split": split, "subject": subject, "label": label, "ids": ids,
                                "lettere": letters}) + "\n")
            tasks.append(mc.task(prompt, ["A", "B", "C", "D"], label))
    mc.write(tasks, a.out / "calib_mmlu.bin")
    manifest = {"n": len(rows), "dropped": drops, "test_hashes": n_test, "bench": n_bench, "letters": letters,
                "mean_tokens": n_tokens / max(len(rows), 1),
                "sha256_bin": hashlib.sha256((a.out / "calib_mmlu.bin").read_bytes()).hexdigest(),
                "sources": "cais/mmlu all validation+dev (never test); prompt = lm-eval MMLU letters, zero-shot"}
    (a.out / "calib_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

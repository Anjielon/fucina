#!/usr/bin/env python3
"""SAFE REPAIR — the gate that turns a promising idea into a verified one.

Learned the hard way on 2026-08-27. A router recalibration (QESC, EAC-MoE
arXiv 2508.01625) improved its own internal metric — top-10 routing overlap
0.5418 -> 0.5565 across 35 of 60 layers — and made the model measurably
*worse*: wikitext perplexity 5.81 -> 6.88, verifiable-answer tests 4/4 -> 3/4.

The reason is structural, and worth stating plainly: a quantized model's
routers are coherent with *its own* activations. Forcing them to imitate the
full-precision model's logits, starting from different activations, breaks
that coherence. Every layer improves on paper; the model degrades.

Hence the rule, enforced here for every repair:
  1. apply to a COPY, never to the original;
  2. measure the REAL outcome (perplexity) before and after;
  3. keep it only if the outcome improves, otherwise discard the copy;
  4. if the repair has many parts, try it in FRACTIONS (25/50/100%) —
     often some parts help while the rest hurt.

When a repair touches only a few small tensors, prefer `selective_rollback`:
it undoes the change in seconds without copying the model at all.
"""
from __future__ import annotations
import os
import argparse, shutil, subprocess, time
from pathlib import Path


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def perplexity(gguf: str, corpus: str, engine: str, chunks: int = 30) -> tuple[float, float]:
    """The real outcome, with its uncertainty. No repair is judged without it.

    Returns (ppl, sigma). The sigma matters: on 2026-08-27 a repair "won" by
    0.0026 while the measurement's own uncertainty was +/-0.0874 — 33 times
    larger. That comparison proved nothing. Compare only at equal corpus and
    equal chunk count, and require the margin to exceed the uncertainty.
    """
    b = Path(engine) / "bin"
    out = subprocess.run(
        [str(b / "llama-perplexity"), "-m", gguf, "-ngl", "999",
         "-c", "2048", "-b", "2048", "--chunks", str(chunks), "-f", corpus],
        capture_output=True, text=True,
        env={"LD_LIBRARY_PATH": str(b), "ODINO_NO_P2": "1", "HOME": str(Path.home())},
    )
    for line in reversed((out.stdout + out.stderr).splitlines()):
        if "Final estimate: PPL" in line:
            value = float(line.split("PPL =")[1].split("+/-")[0])
            sigma = float(line.split("+/-")[1])
            return value, sigma
    raise RuntimeError("perplexity unavailable:\n" + (out.stderr or "")[-600:])


def repair_with_gate(original: str, repair, corpus: str, engine: str,
                     label: str = "repair") -> bool:
    """Run `repair(copy)` on a copy; keep it only if perplexity improves."""
    src = Path(original)
    copy = src.with_suffix(f".{label}.gguf")
    log("measuring the outcome BEFORE the repair...")
    before, sigma_b = perplexity(str(src), corpus, engine)
    log(f"  PPL before: {before:.4f} +/- {sigma_b:.4f}")
    log(f"copying the model ({src.stat().st_size / 2**30:.1f} GiB)...")
    shutil.copy2(src, copy)
    try:
        repair(str(copy))
        after, sigma_a = perplexity(str(copy), corpus, engine)
        margin, noise = before - after, max(sigma_b, sigma_a)
        log(f"  PPL after:  {after:.4f} +/- {sigma_a:.4f} · margin {margin:+.4f} vs noise {noise:.4f}")
        if margin > noise:
            copy.replace(src)
            log(f"✅ repair KEPT: {before:.4f} -> {after:.4f} (margin beats the noise)")
            return True
        copy.unlink()
        why = "no gain" if margin <= 0 else "gain smaller than the measurement noise"
        log(f"⛔ repair DISCARDED ({why}, original untouched): {before:.4f} -> {after:.4f}")
        return False
    except Exception as e:
        copy.unlink(missing_ok=True)
        log(f"⛔ repair failed, copy removed: {e}")
        raise


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="measure a GGUF's perplexity (the real outcome)")
    ap.add_argument("gguf")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--engine", default=os.path.expanduser("~/build-llamacpp-tq1/build"))
    ap.add_argument("--chunks", type=int, default=30)
    a = ap.parse_args()
    ppl, sigma = perplexity(a.gguf, a.corpus, a.engine, a.chunks)
    print(f"PPL = {ppl:.4f} +/- {sigma:.4f}")

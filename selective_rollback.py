#!/usr/bin/env python3
"""SELECTIVE ROLLBACK — undo a repair without copying the model.

Born on 2026-08-27: a router recalibration had degraded a 397B model
(PPL 5.81 -> 6.88) and rolling back meant re-copying 88 GiB with only 41 GiB
free. But the repair had touched exactly 35 small tensors (8 MB each), so only
those need restoring — seconds instead of an impossible copy.

Method (field-verified):
  1. diff the two files tensor by tensor -> the exact list of differences;
  2. GUARD: sample 40 tensors that should be untouched; if any differs, the
     diagnosis is wrong and the rollback does not run;
  3. copy the good tensors ONE AT A TIME (never more than 8 MB resident);
  4. read back and verify after every write;
  5. final check: zero residual differences.

General rule: a repair that touches few tensors is always reversible in place.
Record *which* tensors a repair writes, not merely that it writes some.
"""
from __future__ import annotations
import argparse, random, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/angelo/build-llamacpp/gguf-py")
sys.path.insert(0, str(Path(__file__).parent))
from gguf import GGUFReader
from patch_gguf import ChirurgoGGUF


def diff_tensors(good: str, current: str, name_filter: str = "") -> list[str]:
    """Tensors where the two files differ (optionally filtered by name)."""
    tb = {t.name: t for t in GGUFReader(good).tensors}
    tc = {t.name: t for t in GGUFReader(current).tensors}
    assert set(tb) == set(tc), "the two files do not share the same tensors"
    div = []
    for n in sorted(tb):
        if name_filter and name_filter not in n:
            continue
        if not np.array_equal(np.asarray(tb[n].data), np.asarray(tc[n].data)):
            div.append(n)
    return div


def guard(good: str, current: str, expected: set[str], sample: int = 40) -> int:
    """How many *unexpected* tensors differ (must be 0 for a safe rollback)."""
    tb = {t.name: t for t in GGUFReader(good).tensors}
    tc = {t.name: t for t in GGUFReader(current).tensors}
    others = [n for n in tb if n not in expected]
    random.seed(0)
    ko = 0
    for n in random.sample(others, min(sample, len(others))):
        if not np.array_equal(np.asarray(tb[n].data)[:200000], np.asarray(tc[n].data)[:200000]):
            ko += 1
    return ko


def restore(good: str, current: str, names: list[str]) -> int:
    """Copy the listed tensors from the good file into the current one, one at a time."""
    tb = {t.name: t for t in GGUFReader(good).tensors}
    c = ChirurgoGGUF(current)
    done = 0
    for n in names:
        expected_v = np.asarray(tb[n].data)
        current_v = c.leggi(n)
        if np.array_equal(expected_v.reshape(-1), current_v.reshape(-1)):
            continue
        c.scrivi(n, expected_v.reshape(current_v.shape))
        assert np.array_equal(c.leggi(n).reshape(-1), expected_v.reshape(-1)), f"verification failed for {n}"
        done += 1
        del expected_v, current_v
    return done


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("good"); ap.add_argument("current")
    ap.add_argument("--filter", dest="name_filter", default="", help="e.g. ffn_gate_inp for routers only")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    div = diff_tensors(a.good, a.current, a.name_filter)
    print(f"differing tensors: {len(div)}")
    ko = guard(a.good, a.current, set(div))
    print(f"guard (unexpected tensors differing): {ko} — {'OK' if ko == 0 else 'ABORT'}")
    if a.apply and ko == 0:
        n = restore(a.good, a.current, div)
        left = diff_tensors(a.good, a.current, a.name_filter)
        print(f"✅ restored {n} · residual differences: {len(left)}")
    elif a.apply:
        print("⛔ guard failed: rollback aborted")

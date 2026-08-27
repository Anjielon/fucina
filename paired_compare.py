#!/usr/bin/env python3
"""PAIRED COMPARISON — decide small differences the absolute number cannot.

`llama-perplexity` reports a figure with an uncertainty that reflects how much
the *corpus* varies from chunk to chunk. That variance is shared by both models
being compared, so it drowns real differences: a repair worth 0.02 is invisible
against a ±0.09 spread, in either direction.

Comparing the same chunks pairwise removes it. What matters then is not the
size of the gap but its consistency: if model A beats model B on 27 chunks out
of 30, that is decisive even when the mean difference is small.

    paired_compare.py run_a.log run_b.log

Prefer llama.cpp's own paired machinery when you can afford the one-time
baseline dump — it propagates the uncertainty from the logits instead of
reconstructing it from the printed running estimates, and it reports the Δp
decomposition that tells noise from damage:

    llama-perplexity --kl-divergence-base baseline.kld
    llama-perplexity --kl-divergence-base baseline.kld --kl-divergence

This script exists for the case where you have only the logs.

The logs are `llama-perplexity` output over the *same* corpus and chunk count.
The script reads the running estimates, recovers the per-chunk values, and
reports the sign test — how often one model wins — alongside the mean gap.
"""
from __future__ import annotations
import re
import sys
from math import comb


def per_chunk(path: str) -> list[float]:
    """Recover per-chunk perplexities from the running estimates in a log."""
    text = open(path, errors="ignore").read()
    # llama-perplexity prints the running estimates on a single line,
    # comma-separated: "[1]5.29,[2]5.41,[3]6.02,". Match them anywhere.
    running = [(int(i), float(v)) for i, v in re.findall(r"\[(\d+)\]([\d.]+)", text)]
    if not running:
        return []
    running.sort()
    import math
    values, prev_sum = [], 0.0
    for n, cumulative in running:
        total = math.log(cumulative) * n       # the estimate is a running geometric mean
        values.append(math.exp(total - prev_sum))
        prev_sum = total
    return values


def sign_test(a: list[float], b: list[float]) -> tuple[int, int, float]:
    """How often A beats B, and the probability of seeing that by chance."""
    wins = sum(1 for x, y in zip(a, b) if x < y)
    n = sum(1 for x, y in zip(a, b) if x != y)
    if n == 0:
        return 0, 0, 1.0
    k = max(wins, n - wins)
    p = 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return wins, n, min(1.0, p)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    a, b = per_chunk(sys.argv[1]), per_chunk(sys.argv[2])
    n = min(len(a), len(b))
    if n < 5:
        sys.exit(f"too few chunks recovered ({n}); need the per-chunk output")
    a, b = a[:n], b[:n]
    wins, decided, p = sign_test(a, b)
    gap = sum(x - y for x, y in zip(a, b)) / n
    print(f"chunks compared: {n}")
    print(f"mean difference (A − B): {gap:+.4f}   [negative means A is better]")
    print(f"A wins on {wins}/{decided} chunks · sign test p = {p:.4f}")
    print("verdict:", "A is better" if p < 0.05 and wins * 2 > decided else
                      "B is better" if p < 0.05 else "not distinguishable")

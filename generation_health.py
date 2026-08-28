#!/usr/bin/env python3
"""Generation-health probe — the number that goes NEXT TO every perplexity.

Born 2026-08-28, the day a ppl of 19.6 coexisted with functionally collapsed
generation (empty chat replies, looping broken text) on a ternarized
GatedDeltaNet hybrid. Teacher-forced perplexity windows never see the error
that free-running generation accumulates in a recurrent state (exposure-bias
line: arXiv 2204.01171; ternary-SSM failure: 2606.18114).

Protocol (fixed, so numbers are comparable across models and days):
  - 6 fixed short prompts (IT/EN mix, chat register — NOT WikiText)
  - free-running, 256 tokens, temperature 0 (deterministic)
  - metric: repetition-4 rate = 1 - unique4grams/total4grams, averaged
  - empty/truncated (<32 tokens) generations count as rate 1.0 (collapsed)

Reading: < 0.15 healthy · 0.15-0.35 degraded · > 0.35 collapsed.

Usage: generation_health.py PORT   → prints one line, exits 0/1/2
"""
from __future__ import annotations
import json, sys, urllib.request

PROMPTS = [
    "Spiega in poche frasi perche' il cielo e' blu.",
    "Scrivi una funzione Python che inverta una stringa e spiega come funziona.",
    "Racconta brevemente la trama dei Promessi Sposi.",
    "What are the main differences between TCP and UDP?",
    "Suggerisci tre idee per una cena veloce con quello che c'e' in dispensa.",
    "Describe the water cycle in simple terms.",
]


def rep4(text: str) -> float:
    toks = text.split()
    if len(toks) < 32:
        return 1.0                      # empty/truncated = collapsed
    grams = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    return 1.0 - len(set(grams)) / len(grams)


def main() -> None:
    port = int(sys.argv[1])
    rates = []
    for p in PROMPTS:
        body = json.dumps({
            "model": "probe", "stream": False, "max_tokens": 256,
            "temperature": 0,
            "messages": [{"role": "user", "content": p}],
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                content = json.load(r)["choices"][0]["message"].get("content") or ""
        except Exception as e:
            print(f"probe error on prompt: {e}", file=sys.stderr)
            content = ""
        rates.append(rep4(content))
    m = sum(rates) / len(rates)
    verdict = "SANO" if m < 0.15 else ("DEGRADATO" if m < 0.35 else "COLLASSATO")
    print(f"salute-generativa rep4 = {m:.3f}  [{verdict}]  "
          f"(per-prompt: {' '.join(f'{r:.2f}' for r in rates)})")
    sys.exit(0 if m < 0.15 else (1 if m < 0.35 else 2))


if __name__ == "__main__":
    main()

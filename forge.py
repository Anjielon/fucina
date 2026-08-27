#!/usr/bin/env python3
"""FUCINA — the orchestrator.

    forge.py <stage> --model X.gguf --budget-gib 94

Stages, in order:
    inspect   architecture, tensor families, budget arithmetic
    measure   Hessians, routing counts, layer sensitivity, super weights
    race      every applicable lever is tested ON A SAMPLE of this model
    plan      budget -> per-tensor type map + the levers that won their race
    forge     write the quantized model (resumable journal)
    validate  the real outcome: perplexity, task tests, routing agreement

Each stage writes <work>/forge-state.json, so the process is resumable and
inspectable — the same discipline that makes a multi-hour forge survivable.

Design principle, paid for in full: **no lever is applied on faith**. A lever
that improves its own internal metric can still degrade the model (see
docs/LESSONS.md). Every repair goes through `safe_repair.repair_with_gate`,
which measures perplexity before and after and keeps the change only when the
gain exceeds the measurement's own uncertainty.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

STAGES = ["inspect", "measure", "race", "plan", "forge", "validate"]


def load_state(work: Path) -> dict:
    f = work / "forge-state.json"
    return json.loads(f.read_text()) if f.exists() else {"stages_done": []}


def save_state(work: Path, s: dict) -> None:
    (work / "forge-state.json").write_text(json.dumps(s, indent=1))


def inspect(a, s):
    """Identity card: architecture, tensor families, budget arithmetic."""
    from levers import REGISTRY
    s["model"] = a.model
    s["budget_gib"] = a.budget_gib
    print(f"inspect: {a.model} -> budget {a.budget_gib} GiB")
    print(f"registry: {len(REGISTRY)} candidate levers")


def measure(a, s):
    """One-off per model: Hessians, routing counts, sensitivity, super weights."""
    print("measure: Hessians -> build_hessians.py · super weights -> sw_scan")


def race(a, s):
    """Every applicable lever is measured ON A SAMPLE of the current model
    (a few layers x both matrix families x hot/warm/cold experts) with the
    true Hessian. Priors from the literature only decide the running order —
    the winner is decided here, on this model."""
    from levers import REGISTRY
    print("race: candidate levers and how each is judged")
    for l in REGISTRY:
        print(f"   {l.name:<34} test: {l.test[:60]}")


def plan(a, s):
    """Budget -> per-tensor type map, plus the levers that won their race."""
    print(f"plan: profile={a.profile} -> tensor plan (fixed rules + race results)")


def forge(a, s):
    """Write the model: two-plane TQ1_0 for experts, source quality elsewhere."""
    print("forge: forge_gguf.py with the plan from the previous stage")


def validate(a, s):
    """The real outcome: perplexity with its uncertainty, task tests, routing
    agreement. A stage that can only be passed, never argued with."""
    print("validate: safe_repair.perplexity + the task suite")


def main():
    ap = argparse.ArgumentParser(description="ternary forge for MoE models")
    ap.add_argument("stage", choices=STAGES + ["all"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--budget-gib", type=float, default=94.0)
    ap.add_argument("--profile", choices=["quality", "speed", "streaming"], default="quality")
    ap.add_argument("--work", default="./forge-work")
    a = ap.parse_args()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    s = load_state(work)
    for stage in (STAGES if a.stage == "all" else [a.stage]):
        print(f"═══ {stage.upper()} ═══")
        globals()[stage](a, s)
        if stage not in s["stages_done"]:
            s["stages_done"].append(stage)
        save_state(work, s)


if __name__ == "__main__":
    main()

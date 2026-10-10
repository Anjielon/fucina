"""Manual mode (SPEC section 1): for one piece, the admissible techniques with the engine's advice (from the
database) and, on request, jobs in SERIES (each on top of the previous product) or in PARALLEL (all on the same
base), each with its own paired measurement."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .candidate import Candidate
from .engine import Engine
from .executor import Executor

ADVICE = {"passed": "enable", "consolidated": "enable", "superseded": "enable (but the group has a better one)",
          "off": "disable", "testing": "try it (no measurement against this base)", "waiting": "try it",
          "to_decide": "your call (AMBIGUOUS)", "suspended": "suspended", "excluded": "not admitted",
          "missing_tool": "tool missing", "deferred": "deferred"}


def manual(model: str, objective: str, run_dir: str, piece: str, series: list, parallel: list,
           mode: str = "simulate", db: Optional[str] = None, base: Optional[str] = None) -> dict:
    e = Engine(model, objective, run_dir, mode, db)
    sb = next((s for s in e.signatures.values() if s.label == (base or "R2B3m")), None)
    if sb is None:
        raise SystemExit(f"unknown base {base!r}: the model's candidates are "
                         f"{', '.join(s.label for s in e.signatures.values())}")
    cand = Candidate(model=e.model["name"], model_type=e.model["model_type"], gguf=sb.gguf,
                     objective=e.objective.as_dict(), label=sb.label)
    real = e.executor
    e.executor = Executor(str(Path(run_dir) / "_advice"), "simulate")      # the advice queues nothing real
    rows, warnings = [], []
    admitted = [t for t in e.techniques.values() if (piece in t.pieces or not t.pieces)]
    for t in sorted(admitted, key=lambda t: (t.phase, t.name)):
        v = e.evaluate(t, cand, t.phase)
        rows.append({"phase": t.phase, "technique": t.name, "state": v.state, "advice": ADVICE.get(v.state, v.state),
                     "numbers": v.numbers, "reason": v.reason})
    e.executor = real
    names = {t.name for t in admitted}
    for n in series + parallel:
        if n not in e.techniques:
            raise SystemExit(f"unknown technique: {n}")
        if n not in names:
            warnings.append(f"{n} is not admitted on piece {piece} (used anyway: manual choice)")
    groups: dict = {}
    for n in series + parallel:
        g = getattr(e.techniques[n], "group", "")
        if g:
            groups.setdefault(g, []).append(n)
    for g, nn in groups.items():
        if len(nn) > 1 and set(nn) & set(series):
            warnings.append(f"{', '.join(nn)} are alternatives of the same group ({g}): in series they overwrite each other")
    series_ids, c, dep = [], cand, []
    for n in series:
        t = e.techniques[n]
        id_m, ids, gguf, _ = e.enqueue_trial(t, c, e.signature(cand), piece, dep, label=f"{n}-series")
        series_ids.append(id_m)
        dep = [i for i in ids if not i.startswith("M_")][-1:] or dep
        c = Candidate(**{**asdict(c), "choices": [], "history": [], "gguf": gguf, "label": f"{c.label}+{n}"})
    parallel_ids = []
    for n in parallel:
        id_m, _, _, _ = e.enqueue_trial(e.techniques[n], cand, e.signature(cand), piece)
        parallel_ids.append(id_m)
    launched = e.launch()
    return {"piece": piece, "base": sb.label, "objective": e.objective.text, "techniques": rows, "series": series_ids,
            "parallel": parallel_ids, "warnings": warnings, "run": run_dir, "launched": [j.id for j in launched]}

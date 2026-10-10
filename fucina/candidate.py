"""Current candidate and resumable on-disk run state (SPEC sections 2 and 6)."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Choice:
    phase: str
    piece: str
    technique: str
    state: str                      # testing | passed | off | to_decide | consolidated
    numbers: dict = field(default_factory=dict)
    reason: str = ""
    switch: str = ""


@dataclass
class Candidate:
    model: str                      # logical name (e.g. Thor-T)
    model_type: str                 # moe | dense | hybrid
    gguf: str                       # current file
    objective: dict                 # e.g. {"mmlu_min": 80.05, "ram_max_gib": 9, "tg_min": 50}
    current_phase: str = "F0"
    choices: list = field(default_factory=list)
    label: str = ""                 # name of the candidate in the historic measurements (R2B3m, PV12P, ...)
    history: list = field(default_factory=list)   # [{phase, label, gguf, reason}] consolidated candidates, in order
    forecast: dict = field(default_factory=dict)

    def enabled(self) -> list:
        return [c for c in self.choices if c.state in ("passed", "consolidated")]

    def to_decide(self) -> list:
        return [c for c in self.choices if c.state == "to_decide"]


class RunState:
    """One JSON file per run, rewritten atomically on every change."""

    def __init__(self, run_dir: str):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.file = self.dir / "state.json"

    def save(self, c: Candidate) -> None:
        d = asdict(c)
        d["saved"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1))
        tmp.replace(self.file)

    def load(self) -> Candidate:
        d = json.loads(self.file.read_text())
        d.pop("saved", None)
        d["choices"] = [Choice(**c) for c in d.get("choices", [])]
        return Candidate(**d)

    def exists(self) -> bool:
        return self.file.exists()

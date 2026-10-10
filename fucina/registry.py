"""Technique contract and self-registering registry (SPEC section 3).

The core never imports concrete techniques: each one lives in ``techniques/<name>.py`` and registers itself with
:func:`register`."""
from __future__ import annotations

import importlib
import pkgutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Thresholds:
    kl_min_pct: float = -0.5         # minimum drop of realised KL (in %) to call it a pass
    sigma: float = 2.0               # difference in standard errors
    tg_max_loss_pct: float = 3.0     # maximum slowdown accepted without asking
    ram_max_bytes: Optional[int] = None


@dataclass
class Outcome:
    verdict: str                     # PASS | FAIL | AMBIGUOUS | INFORMATIVE | UNINFORMATIVE
    numbers: dict = field(default_factory=dict)
    reason: str = ""


@dataclass
class Change:
    gguf: str                        # produced file (or the unchanged candidate, with a switch)
    switch: str                      # how to turn it off without rebuilding
    extra_bytes: int = 0
    job_ids: list = field(default_factory=list)   # queued jobs that build it, in order


class Technique(ABC):
    name: str = ""
    phase: str = ""                  # F0..F6
    pieces: tuple = ()               # admissible pieces (empty = whole model)
    model_types: tuple = ("moe", "hybrid", "dense")
    ternary: bool = True             # does the model stay fully ternary?
    slows_down: bool = False
    gpu_cost_h: float = 0.5
    est_extra_bytes: int = 0
    source: str = ""                 # arXiv id + reported number ("published lever" rule)
    thresholds: Thresholds = Thresholds()

    def admits(self, phase: str, piece: str, model_type: str, constraints: dict) -> bool:
        if phase != self.phase or (self.pieces and piece not in self.pieces):
            return False
        if model_type not in self.model_types:
            return False
        if constraints.get("all_ternary", True) and not self.ternary:
            return False
        return True

    def cheap_test(self, ctx: Any) -> Outcome:        # CPU, minutes; default: no test
        return Outcome("INFORMATIVE", reason="no cheap test defined")

    @abstractmethod
    def apply(self, ctx: Any, piece: str) -> Change: ...

    @abstractmethod
    def switch(self) -> str: ...


_REGISTRY: dict = {}


def register(cls: type) -> type:
    instance = cls()
    if not instance.name:
        raise ValueError(f"{cls.__name__}: missing name")
    _REGISTRY[instance.name] = instance
    return cls


def load_techniques() -> dict:
    from . import techniques
    for m in pkgutil.iter_modules(techniques.__path__):
        importlib.import_module(f"{techniques.__name__}.{m.name}")
    return dict(_REGISTRY)


def admissible(phase: str, piece: str, model_type: str, constraints: dict) -> list:
    return [t for t in load_techniques().values() if t.admits(phase, piece, model_type, constraints)]

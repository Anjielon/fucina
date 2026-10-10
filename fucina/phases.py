"""Pipeline phases. The default order is the one learned on Thor-T (SPEC section 2)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Phase:
    code: str
    name: str
    description: str
    training: bool = False      # True = a training phase: run ONCE, on top of the final structure


PHASES = (
    Phase("F0", "diagnosis", "per-piece sensitivity: measured dKL, importance matrix, kurtosis, routing frequency"),
    Phase("F1", "dense", "quantizer for dense tensors (planes, grids, refinements)"),
    Phase("F2", "experts", "extra capacity on experts (dKL-selected second plane, shared bases, corrections)"),
    Phase("F3", "head", "embedding, output head, MTP head"),
    Phase("F4", "realignment", "whole-model realignment against the teacher", training=True),
    Phase("F5", "runtime", "runtime: int8 activations, MTP (acceptance and tok/s)"),
    Phase("F6", "evaluation", "spoken probe, paired MMLU, reasoning, speed, resident RAM"),
)
BY_CODE = {p.code: p for p in PHASES}


def phase(code: str) -> Phase:
    return BY_CODE[code]

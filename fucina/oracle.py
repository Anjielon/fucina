"""Database oracle: measurements already made (measurement database) for a technique AGAINST the current candidate.
Always consulted (nothing is measured twice) and, with ``--simulate``, used instead of the GPU.

* referred -- the row compares the technique with the current candidate (declared base, or MMLU/KL of the base =
  signature);
* nearby   -- a measurement of the same technique against another candidate (evidence for priority and defaults)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from .db import Measurements
from .decision import BYTE_TOLERANCE
from .importers.labels import technique_of
from .measure import Measurement


@dataclass
class Signature:
    """How a candidate is recognised in historic measurements."""
    label: str
    gguf: str = ""
    mmlu: list = field(default_factory=list)
    kl: Optional[float] = None
    file_bytes: Optional[int] = None
    tg: Optional[float] = None
    kl_pct: Optional[float] = None       # drop relative to the law's reference
    kl_json: str = ""                    # per-row KL already measured (reference of paired measurements)

    def recognises(self, r: dict) -> bool:
        par = r.get("_par", {})
        if par.get("base") and par["base"] == self.label:
            return True
        if par.get("base") and par["base"] not in ("", None) and r.get("mmlu_base") is None:
            return False
        mb = r.get("mmlu_base")
        if mb is not None and any(abs(mb - x) <= 0.06 for x in self.mmlu):
            return True
        kb = r.get("kl_base")
        return kb is not None and self.kl is not None and abs(kb - self.kl) < 0.002 and mb is None


_BYTES_IN_NOTE = re.compile(r"[(,]\s*(\d{10,11}) B")


class Oracle:
    def __init__(self, measurements: Measurements, model: str, techniques: dict):
        self.measurements = measurements
        self.model = model
        self.techniques = techniques
        self._rows: Optional[list] = None

    def rows(self) -> list:
        if self._rows is None:
            out = []
            for r in self.measurements.all():
                if r.get("model") != self.model:
                    continue
                try:
                    r["_par"] = json.loads(r["params"]) if r.get("params") else {}
                except (TypeError, ValueError):
                    r["_par"] = {}
                if str(r.get("source", "")).startswith("fucina") or r.get("technique") in self.techniques:
                    r["_technique"] = r.get("technique")
                else:
                    r["_technique"] = technique_of(r.get("technique") or "", self.techniques)
                if r.get("file_bytes") is None and r.get("note"):
                    g = _BYTES_IN_NOTE.search(r["note"][:300])
                    if g:
                        r["file_bytes"] = int(g.group(1))
                out.append(r)
            self._rows = out
        return self._rows

    def invalidate(self) -> None:
        self._rows = None

    def of(self, technique: str) -> list:
        return [r for r in self.rows() if r["_technique"] == technique]

    # ── row choice and merge ──
    @staticmethod
    def _measurement(r: dict) -> Measurement:
        p = r["_par"]
        return Measurement(kl=r.get("kl"), kl_base=r.get("kl_base"), kl_pct=r.get("kl_pct"), kl_se=p.get("kl_se"),
                           tg=r.get("tg"), tg_base=r.get("tg_base"), file_bytes=r.get("file_bytes"), mmlu=r.get("mmlu"),
                           mmlu_base=r.get("mmlu_base"), p_mcnemar=r.get("p_mcnemar"), wins=p.get("wins"),
                           losses=p.get("losses"),
                           source=f"{r.get('source')} #{r.get('id')} {p.get('variant') or r.get('technique')}",
                           note=(r.get("note") or "")[:200]).complete()

    def _pick(self, rows: list, cap: Optional[int]) -> Optional[Measurement]:
        if cap:
            rows = [r for r in rows if r.get("file_bytes") is None or r["file_bytes"] <= cap * BYTE_TOLERANCE]
        if not rows:
            return None
        largest = max((r["file_bytes"] for r in rows if r.get("file_bytes")), default=None)

        def score(r: dict) -> tuple:
            near_cap = 2 if (r.get("file_bytes") and largest and r["file_bytes"] >= 0.9 * largest) \
                else 1 if r.get("file_bytes") is None else 0
            rich = sum(r.get(k) is not None for k in ("kl_pct", "mmlu", "tg"))
            ours = str(r.get("source") or "").startswith("fucina")      # rows written by Fucina: paired numbers
            return (near_cap, ours, str(r.get("recorded_at") or ""), rich)
        rows = sorted(rows, key=score, reverse=True)
        chosen = rows[0]
        m = self._measurement(chosen)
        var = chosen["_par"].get("variant") or chosen.get("technique")
        # fill missing fields from rows of the SAME experiment (same variant or same size)
        for r in rows[1:]:
            same = ((r["_par"].get("variant") or r.get("technique")) == var or
                    (r.get("file_bytes") and chosen.get("file_bytes")
                     and abs(r["file_bytes"] - chosen["file_bytes"]) < 1e-3 * chosen["file_bytes"]) or
                    (r.get("file_bytes") is None and chosen.get("file_bytes") is None))
            if not same:
                continue
            o = self._measurement(r)
            for k in ("kl", "kl_base", "kl_pct", "kl_se", "kl_delta", "tg", "tg_base", "mmlu", "mmlu_base", "p_mcnemar",
                      "wins", "losses", "file_bytes"):
                if getattr(m, k) is None and getattr(o, k) is not None:
                    if k in ("tg", "tg_base") and (m.tg is not None or m.tg_base is not None) and k == "tg_base" \
                            and m.tg is None:
                        continue
                    setattr(m, k, getattr(o, k))
                    m.source += f" + #{r.get('id')}"
        return m.complete()

    def referred(self, technique: str, sig: Signature, cap: Optional[int] = None) -> Optional[Measurement]:
        return self._pick([r for r in self.of(technique) if sig.recognises(r)], cap)

    def nearby(self, technique: str, sig: Signature, cap: Optional[int] = None,
               signatures: tuple = ()) -> Optional[tuple]:
        """Measurement of the technique against ANOTHER candidate: (measurement, label of that base)."""
        others = [r for r in self.of(technique) if not sig.recognises(r)]
        m = self._pick(others, cap)
        if m is None:
            return None
        base = next((r["_par"].get("base") for r in others if r["_par"].get("base")), None)
        if base is None and m.mmlu_base is not None:
            base = next((f.label for f in signatures
                         if f is not sig and any(abs(m.mmlu_base - x) <= 0.06 for x in f.mmlu)),
                        f"MMLU {m.mmlu_base:.2f}")
        return m, base or "another candidate"

    def best_historic_gain(self, technique: str) -> Optional[float]:
        """Best KL drop (%) ever measured for the technique on this model (for ordering)."""
        v = [r["kl_pct"] for r in self.of(technique) if r.get("kl_pct") is not None]
        return min(v) if v else None

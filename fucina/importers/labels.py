"""Experiment label -> registered technique."""
from __future__ import annotations

import re
from typing import Optional

from ..registry import load_techniques


def technique_of(label: str, techniques: Optional[dict] = None) -> Optional[str]:
    """Name of the registered technique whose ``measurement_aliases`` match the label; if several match, the one
    matching furthest to the left wins («E2 rank1 su pv12» is E2, not the second plane)."""
    best = None
    for t in (techniques or load_techniques()).values():
        for a in getattr(t, "measurement_aliases", ()):
            g = re.search(a, label, re.I)
            if g and (best is None or g.start() < best[0]):
                best = (g.start(), t.name)
    return best[1] if best else None

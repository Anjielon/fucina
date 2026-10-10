"""Pre-load the database with measurements already made on Thor-T: the CHIUSO (closed) lines of the activity log
(``import_sources.cycles``) that carry an MMLU or KL number."""
from __future__ import annotations

import re
import sys
from typing import Optional

from .. import config
from ..db import Measurements

# The patterns read the lab notebook's wording ("vs R2B3m", "partenza" = starting point, decimal commas).
_MMLU = re.compile(r"(\d{2}[.,]\d+)\s*vs\s*(?:R2B3m(?: int8)?|partenza|pv12|Q2_K)?\s*(\d{2}[.,]\d+)")
_P = re.compile(r"p\s*[= ]?\s*(0[.,]\d+|1[.,]0+)")
_KL = re.compile(r"KL[^0-9-]*(-?\d+[.,]\d+)\s*%")
_TG = re.compile(r"tg\D{0,20}?(\d{2,3}[.,]\d+)")


def _f(x: Optional[str]) -> Optional[float]:
    return float(x.replace(",", ".")) if x else None


def cycles_path() -> str:
    p = (config.current().get("import_sources") or {}).get("cycles") or config.current().get("cycle_log")
    if not p:
        raise config.ConfigError("import_sources.cycles is not configured")
    return p


def import_cycles(m: Measurements, path: Optional[str] = None, model: str = "Thor-T") -> int:
    n = 0
    for line in open(path or cycles_path(), encoding="utf-8", errors="replace"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3 or parts[1] != "CHIUSO":
            continue
        text = parts[2]
        mm = _MMLU.search(text)
        if not mm and not _KL.search(text):
            continue
        technique = re.split(r"[:(]", text, maxsplit=1)[0].strip()[:80]
        p = _P.search(text)
        kl = _KL.search(text)
        tg = _TG.search(text)
        m.add(recorded_at=parts[0], model=model, model_type="hybrid", technique=technique,
              mmlu=_f(mm.group(1)) if mm else None, mmlu_base=_f(mm.group(2)) if mm else None,
              p_mcnemar=_f(p.group(1)) if p else None, kl_pct=_f(kl.group(1)) if kl else None,
              tg=_f(tg.group(1)) if tg else None, source="cicli.tsv", note=text[:500])
        n += 1
    return n


if __name__ == "__main__":
    db = Measurements(sys.argv[1] if len(sys.argv) > 1 else None)
    print(import_cycles(db), "measurements imported into", db.path)

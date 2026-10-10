"""Importer of the structured Thor-T lab-notebook sources (complements :mod:`.cycles`, which reads the closed lines of
the activity log):

* the table of ``import_sources.results`` (results of 3-5 October);
* the «KL realizzata» (realised KL) and «tg» lines of the activity log (also ACTIVE ones);
* the measured state of ``import_sources.plan`` (plan of 8 October);
* the RILQ-mini table of ``import_sources.audit`` (reproduction audit).

Every row carries ``params.variant`` (the experiment label) and ``technique`` = the registered name (via
``measurement_aliases``) when recognised. Idempotent: rows whose source starts with ``fucina:`` are deleted and
rewritten. The regular expressions match the notebook's Italian wording on purpose (they parse data)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from .. import config
from ..db import Measurements
from ..registry import load_techniques
from .labels import technique_of

GIB = 2 ** 30
MMLU_R2B3M = 74.60          # reference candidate of Thor-T: MMLU-2000 and realised KL
KL_R2B3M = 0.41539
MODEL = "Thor-T"


def _source(key: str) -> str:
    p = (config.current().get("import_sources") or {}).get(key)
    if not p:
        raise config.ConfigError(f"import_sources.{key} is not configured")
    return p


def _n(x: Optional[str]) -> Optional[float]:
    if x is None:
        return None
    x = x.replace("−", "-").replace(",", ".").replace("*", "").strip()
    try:
        return float(x)
    except ValueError:
        return None


def _row(m: Measurements, source: str, variant: str, techniques: dict, **c) -> int:
    par = c.pop("params", {}) or {}
    par["variant"] = variant
    c.setdefault("model", MODEL)
    c.setdefault("model_type", "hybrid")
    return m.add(source=f"fucina:{source}", technique=c.pop("technique", None) or technique_of(variant, techniques)
                 or variant, params=par, **c)


# ── results table ──
_P = re.compile(r"p\s*(<|=)?\s*([0-9.,e-]+)")


def import_results(m: Measurements, path: Optional[str] = None, techniques: Optional[dict] = None) -> int:
    techniques = techniques or load_techniques()
    n = 0
    for r in Path(path or _source("results")).read_text().splitlines():
        if not r.startswith("|") or "---" in r or r.startswith("| leva"):
            continue
        c = [x.strip() for x in r.strip("|").split("|")]
        if len(c) < 6:
            continue
        lever, size, mmlu_s, against, kl_s, outcome = c[:6]
        variant = re.sub(r"\*|\(.*?\)", "", lever).strip()
        gib = _n((re.search(r"([\d,]+)\s*GiB", size) or [None, None])[1])
        base = MMLU_R2B3M
        if "Dell" in lever:
            base = 74.85
        mm = re.findall(r"\d{2},\d{1,2}", mmlu_s)
        p = _P.search(against)
        kl = re.search(r"([\d,]+)\s*\(([−\-+][\d,]+)\s*%\)", kl_s)
        on = re.search(r"su (pv12)", against)
        if on:
            base = 78.50
        pv = None
        if p:
            pv = 1e-4 if p.group(1) == "<" else _n(p.group(2))
        wl = re.search(r"(\d+)/(\d+)", against)
        for i, x in enumerate(mm or [None]):
            if x is None and not kl:
                continue
            names = [x.strip() for x in variant.split(" / ")]
            vn = names[i] if len(names) == len(mm) and len(mm) > 1 else (variant if len(mm) <= 1 else f"{variant}#{i}")
            _row(m, "RISULTATI_3-5_OTTOBRE", vn, techniques, technique=technique_of(vn, techniques),
                 recorded_at="2026-10-05T12:00", file_bytes=int(gib * GIB) if gib else None, mmlu=_n(x),
                 mmlu_base=base, p_mcnemar=pv, kl=_n(kl.group(1)) if kl else None,
                 kl_base=KL_R2B3M if kl else None, kl_pct=_n(kl.group(2)) if kl else None, verdict=outcome,
                 params={"wins": int(wl.group(1)), "losses": int(wl.group(2))} if wl and len(mm) <= 1 else {},
                 note=r[:400])
            n += 1
    return n


# ── activity log: paired realised KL and speed ──
_KL1 = re.compile(r"([A-Za-z][\w.\-]*) KL (\d\.\d+) ±(\d\.\d+) \([^)]*?vs ([\w.\-]+) ([+-]\d\.\d+) ±(\d\.\d+)")
_KL2 = re.compile(r"([A-Za-z][\w.\-]*) KL (\d\.\d+) \(([+-]\d+\.\d+) % vs ([\w.\-]+) (\d\.\d+), ([+-]\d+\.\d+) se\)")
_TG1 = re.compile(r"(?:^|[;:|]\s*|\s)([A-Za-z][\w.\-]*) tg (\d+\.\d+)")
_TG2 = re.compile(r"tg128 int8 partenza (\d+\.\d+)±[\d.]+ (\w+) (\d+\.\d+)")
_TG3 = re.compile(r"tg R2B3m int8 (\d+\.\d+) ([\w.\-]+) (\d+\.\d+)")
_MM = re.compile(r"MMLU(?:-2000)?(?: int8)?(?: [\w-]+)? (\d{2}\.\d+) vs (?:[\w ]+?)?(\d{2}\.\d+)"
                 r"(?: vinte/perse (\d+) (\d+))? p (\d\.\d+)")
_BYTES = re.compile(r"\((\d{10,11}) B")
_BYTES_IN = re.compile(r"[(,]\s*(\d{10,11}) B")
_TG4 = re.compile(r"tg senza/con: (\d+\.\d+) / (\d+\.\d+)")


def _true_base(base: str, header: str) -> str:
    """«partenza»/«base» (starting point) in the lines of a run = the candidate named in the header
    (e.g. «RILQ-THOR R2B3m»)."""
    if base and base.lower() in ("partenza", "base"):
        g = re.search(r"\b(R2B3m|PV12P|pv12)\b", header or "")
        if g:
            return g.group(1)
    return base


def bytes_from_cycles(variant: str, path: Optional[str] = None) -> Optional[int]:
    """File size of VARIANT from the last activity-log line declaring it («VARIANT (..., NNN B)»)."""
    out = None
    for line in open(path or _source("cycles"), encoding="utf-8", errors="replace"):
        i = line.find(f"\t{variant} (")
        if i >= 0:
            g = _BYTES_IN.search(line[i:i + 200])
            if g:
                out = int(g.group(1))
    return out


def import_kl_cycles(m: Measurements, path: Optional[str] = None, techniques: Optional[dict] = None,
                     since: str = "2026-10-01") -> int:
    techniques = techniques or load_techniques()
    rows: dict = {}

    def put(key, *a, **k):          # same experiment reported by several lines (ACTIVE then CLOSED): the last wins
        rows[key] = (a, k)

    for line in open(path or _source("cycles"), encoding="utf-8", errors="replace"):
        p = line.rstrip("\n").split("\t")
        if len(p) < 3 or p[0] < since or p[1] not in ("ATTIVO", "CHIUSO"):
            continue
        when, text = p[0], p[2]
        header = re.split(r"[:(]", text, maxsplit=1)[0].strip()
        mm = _MM.search(text)
        by = _BYTES.search(text)
        tg = {a: float(b) for a, b in _TG1.findall(text)}
        t3 = _TG3.search(text)
        if t3:
            tg["R2B3m"] = float(t3.group(1))
            tg[t3.group(2)] = float(t3.group(3))
        seen = set()
        for g in _KL1.finditer(text):
            name, kl, se_r, base, d, se = g.groups()
            if float(d) == 0.0 or name in seen:
                continue
            seen.add(name)
            kl_b = float(kl) - float(d)
            var = name if name.lower() not in ("rilq",) else f"{header} rilq"
            put((var, kl), m, "cicli.tsv/KL", var, techniques, technique=technique_of(var, techniques), recorded_at=when,
                kl=float(kl), kl_base=kl_b, kl_pct=100 * float(d) / kl_b, file_bytes=int(by.group(1)) if by else None,
                tg=tg.get(name), tg_base=tg.get(base),
                mmlu=float(mm.group(1)) if mm else None, mmlu_base=float(mm.group(2)) if mm else None,
                p_mcnemar=float(mm.group(5)) if mm else None,
                params={"kl_se": float(se), "base": _true_base(base, header)}
                | ({"wins": int(mm.group(3)), "losses": int(mm.group(4))} if mm and mm.group(3) else {}),
                note=text[:500])
        for g in _KL2.finditer(text):
            name, kl, pct, base, klb, z = g.groups()
            if name in seen:
                continue
            seen.add(name)
            d = float(kl) - float(klb)
            put((name, kl), m, "cicli.tsv/KL", name, techniques, recorded_at=when, kl=float(kl), kl_base=float(klb),
                kl_pct=float(pct), tg=tg.get(name), tg_base=tg.get(base),
                params={"kl_se": abs(d / float(z)) if float(z) else None, "base": _true_base(base, header)},
                note=text[:500])
        t2 = _TG2.search(text)
        if t2 and not seen:
            var = f"{header} {t2.group(2)}"
            put((var, "tg"), m, "cicli.tsv/tg", var, techniques, technique=technique_of(var, techniques),
                recorded_at=when, tg=float(t2.group(3)), tg_base=float(t2.group(1)),
                params={"base": _true_base("partenza", header)}, note=text[:500])
        t4 = _TG4.search(text)
        if t4 and mm:
            put((header, "int8"), m, "cicli.tsv/velocita", header, techniques, recorded_at=when, tg=float(t4.group(2)),
                tg_base=float(t4.group(1)), mmlu=float(mm.group(1)), mmlu_base=float(mm.group(2)),
                p_mcnemar=float(mm.group(5)), params={"base": "R2B3m"}, note=text[:500])
    for a, k in rows.values():
        _row(*a, **k)
    return len(rows)


# ── plan: measured state ──
def import_plan(m: Measurements, path: Optional[str] = None, techniques: Optional[dict] = None,
                cycles: Optional[str] = None) -> int:
    techniques = techniques or load_techniques()
    n = 0
    text = Path(path or _source("plan")).read_text()
    block = text.split("## Stato misurato", 1)[-1].split("\n## ", 1)[0]
    for r in block.splitlines():
        r = r.strip("- ").strip()
        g = re.match(r"(\w+)", r)
        if not g:
            continue
        var = g.group(1)
        t = technique_of(r.split(":")[0], techniques)
        if t is None:
            continue
        mm = re.search(r"MMLU (\d{2},\d+)(?: vs (\d{2},\d+))?", r)
        tgm = re.search(r"tg (\d+,\d+)", r)
        klm = re.search(r"KL corpus maestro ([−\-][\d,]+) %", r) or re.search(r"KL held ([−\-][\d,]+) %", r)
        p = re.search(r"p ([\d,]+)", r)
        if not (mm or tgm or klm):
            continue
        base = _n(mm.group(2)) if mm and mm.group(2) else MMLU_R2B3M
        # the plan's «12,13 GiB» are decimal GB: the true size comes from the activity-log line of the same experiment
        _row(m, "PIANO_2026-10-08", var, techniques, technique=t, recorded_at="2026-10-08T08:00",
             file_bytes=bytes_from_cycles(var, cycles), mmlu=_n(mm.group(1)) if mm else None,
             mmlu_base=base if mm else None, p_mcnemar=_n(p.group(1)) if p else None,
             kl_pct=_n(klm.group(1)) if klm else None, kl_base=KL_R2B3M if klm else None,
             tg=_n(tgm.group(1)) if tgm else None, note=r[:400])
        n += 1
    return n


# ── reproduction audit: RILQ-mini (Qwen3-0.6B) ──
def import_audit_rilq(m: Measurements, path: Optional[str] = None, techniques: Optional[dict] = None) -> int:
    techniques = techniques or load_techniques()
    n = 0
    for r in Path(path or _source("audit")).read_text().splitlines():
        r = r.strip()
        g = re.match(r"\|\s*\((i{1,3}|iv|v|vi)\)\s*([^|]+)\|\s*([^|]+)\|\s*([\d,]+)\s*\|\s*\*\*([−\-][\d,]+) %\*\*", r)
        if not g:
            continue
        tag, var, steps, kl, pct = g.groups()
        ns = re.match(r"\D*(\d+)", steps)
        _row(m, "AUDIT_RIPRODUZIONE/A5", f"RILQ-mini ({tag}) {var.strip()}", techniques, technique="rilq",
             model="Qwen3-0.6B", model_type="dense", recorded_at="2026-10-07T19:33", kl=_n(kl), kl_pct=_n(pct),
             params={"steps": int(ns.group(1)) if ns else None}, note=r[:400])
        n += 1
    return n


def import_all(m: Measurements, cycles: Optional[str] = None) -> dict:
    """Rewrite the «fucina:*» rows (idempotent). Rows written by :mod:`.cycles` (source cicli.tsv) stay."""
    m.con.execute("DELETE FROM measurements WHERE source LIKE 'fucina:%'")
    m.con.commit()
    t = load_techniques()
    cycles = cycles or _source("cycles")
    return {"results": import_results(m, techniques=t), "kl_cycles": import_kl_cycles(m, cycles, techniques=t),
            "plan": import_plan(m, techniques=t, cycles=cycles), "audit_rilq": import_audit_rilq(m, techniques=t)}


if __name__ == "__main__":
    import sys
    db = Measurements(sys.argv[1] if len(sys.argv) > 1 else None)
    print(json.dumps(import_all(db)), "->", db.path)

"""Measurement database (sqlite) and rule base (YAML) — SPEC section 5. Nothing is discarded: every experiment is a
row, failures included."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

from . import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS measurements (
  id INTEGER PRIMARY KEY, recorded_at TEXT, model TEXT, model_type TEXT, phase TEXT, piece TEXT,
  technique TEXT, params TEXT, gguf TEXT, file_bytes INTEGER, ram_gib REAL,
  kl REAL, kl_base REAL, kl_pct REAL, mmlu REAL, mmlu_base REAL, p_mcnemar REAL, tg REAL, tg_base REAL,
  verdict TEXT, source TEXT, note TEXT
);
CREATE INDEX IF NOT EXISTS ix_technique ON measurements(technique, model);
"""


def default_path() -> str:
    return str(config.current()["database"])


class Measurements:
    def __init__(self, path: Optional[str] = None):
        self.path = str(path or default_path())
        self.con = sqlite3.connect(self.path)
        self.con.executescript(_SCHEMA)

    def add(self, **c: Any) -> int:
        c.setdefault("recorded_at", time.strftime("%Y-%m-%dT%H:%M"))
        if isinstance(c.get("params"), dict):
            c["params"] = json.dumps(c["params"], ensure_ascii=False)
        cols = ",".join(c)
        qs = ",".join("?" * len(c))
        cur = self.con.execute(f"INSERT INTO measurements ({cols}) VALUES ({qs})", tuple(c.values()))
        self.con.commit()
        return cur.lastrowid

    def by_technique(self, technique: str, model: Optional[str] = None) -> list:
        q = "SELECT * FROM measurements WHERE technique=?" + (" AND model=?" if model else "") + " ORDER BY recorded_at"
        cur = self.con.execute(q, (technique, model) if model else (technique,))
        names = [d[0] for d in cur.description]
        return [dict(zip(names, r)) for r in cur.fetchall()]

    def all(self) -> list:
        cur = self.con.execute("SELECT * FROM measurements ORDER BY recorded_at")
        names = [d[0] for d in cur.description]
        return [dict(zip(names, r)) for r in cur.fetchall()]


def load_rules(dirs: Optional[Iterable] = None) -> list:
    """Each YAML file is a list of rules {id, statement, scope{model_type, piece, phase}, effect, evidence[],
    confidence, exceptions[]}. Default: the packaged rule base plus ``rules_dirs`` from the configuration."""
    rules = []
    for d in (dirs if dirs is not None else config.current().rules_dirs()):
        for f in sorted(Path(d).glob("*.yaml")):
            for r in yaml.safe_load(f.read_text()) or []:
                r["_file"] = f.name
                rules.append(r)
    return rules


def rules_for(phase: str, piece: str, model_type: str, rules: Optional[list] = None) -> list:
    """Rules whose scope covers (phase, piece, model_type); a missing field matches everything."""
    out = []
    for r in rules if rules is not None else load_rules():
        s = r.get("scope", {})
        if all(not s.get(k) or v in (s[k] if isinstance(s[k], list) else [s[k]])
               for k, v in (("phase", phase), ("piece", piece), ("model_type", model_type))):
            out.append(r)
    return sorted(out, key=lambda r: -float(r.get("confidence", 0)))

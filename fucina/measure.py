"""Measurements (SPEC section 4, step 4): whole-model realised KL on the teacher's held-out rows, paired against the
current candidate; generation speed in the SAME session; file size/RAM; paired MMLU with McNemar.

Readers for the raw outputs of the wrapped, validated scripts (``kl_realizzata.py``, ``mcnemar_mclp.py``,
``llama-bench``) and the measurement job (``combo_measure``: int8 realised KL on the teacher rows + tg128, MMLU when
the KL improves by more than 2 standard errors). The JSON keys read here are those scripts' output format."""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from . import config
from .executor import Job, bash

GIB = 2 ** 30
DEFAULT_FILE_CAP = 9_492_293_427      # see decision.DEFAULT_FILE_CAP_BYTES


@dataclass
class Measurement:
    """A paired candidate-vs-reference measurement. Every field is optional: historic measurements are partial."""
    kl: Optional[float] = None
    kl_base: Optional[float] = None
    kl_delta: Optional[float] = None       # paired per-row mean (candidate - base)
    kl_se: Optional[float] = None          # paired standard error
    kl_pct: Optional[float] = None         # 100 * delta / base
    tg: Optional[float] = None
    tg_base: Optional[float] = None
    file_bytes: Optional[int] = None
    ram_gib: Optional[float] = None
    mmlu: Optional[float] = None
    mmlu_base: Optional[float] = None
    p_mcnemar: Optional[float] = None
    wins: Optional[int] = None
    losses: Optional[int] = None
    probe: Optional[str] = None
    source: str = ""
    note: str = ""
    extra: dict = field(default_factory=dict)

    def complete(self) -> "Measurement":
        if self.kl_delta is None and self.kl is not None and self.kl_base is not None:
            self.kl_delta = self.kl - self.kl_base
        if self.kl_pct is None and self.kl_delta is not None and self.kl_base:
            self.kl_pct = 100.0 * self.kl_delta / self.kl_base
        if self.kl_delta is None and self.kl_pct is not None and self.kl_base:
            self.kl_delta = self.kl_pct * self.kl_base / 100.0
        return self

    @property
    def z_kl(self) -> Optional[float]:
        if self.kl_delta is None or not self.kl_se:
            return None
        return self.kl_delta / self.kl_se

    @property
    def tg_pct(self) -> Optional[float]:
        if self.tg is None or not self.tg_base:
            return None
        return 100.0 * (self.tg - self.tg_base) / self.tg_base

    @property
    def mmlu_delta(self) -> Optional[float]:
        if self.mmlu is None or self.mmlu_base is None:
            return None
        return self.mmlu - self.mmlu_base

    def mmlu_se(self, n: int = 2000, discordant: float = 0.1) -> float:
        """Standard error of the paired MMLU delta (points). Exact with wins/losses; otherwise ~10 % discordant."""
        if self.wins is not None and self.losses is not None:
            return 100.0 * math.sqrt(max(self.wins + self.losses, 1)) / n
        return 100.0 * math.sqrt(discordant / n)

    def short(self) -> str:
        p = []
        if self.kl_pct is not None:
            z = self.z_kl
            p.append(f"KL {self.kl_pct:+.1f} %" + (f" ({z:+.1f} se)" if z is not None else ""))
        if self.mmlu is not None:
            p.append(f"MMLU {self.mmlu:.2f}" + (f" vs {self.mmlu_base:.2f}" if self.mmlu_base is not None else "")
                     + (f" p {self.p_mcnemar:.3g}" if self.p_mcnemar is not None else ""))
        if self.tg is not None:
            p.append(f"tg {self.tg:.1f}" + (f" ({self.tg_pct:+.1f} %)" if self.tg_pct is not None else ""))
        if self.file_bytes:
            p.append(f"{self.file_bytes / GIB:.2f} GiB")
        return ", ".join(p) or "no numbers"

    def as_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if v not in (None, "", {})}
        if self.z_kl is not None:
            d["z_kl"] = round(self.z_kl, 2)
        if self.tg_pct is not None:
            d["tg_pct"] = round(self.tg_pct, 2)
        return d


# ── raw-output readers ──
def read_kl(candidate_json: str, base_json: str) -> Measurement:
    """Per-row paired KL (``kl_realizzata.py`` format: per_riga[k]['kl_media'])."""
    c = json.loads(Path(candidate_json).read_text())
    b = json.loads(Path(base_json).read_text())
    common = [k for k in c["per_riga"] if k in b["per_riga"]]
    dd = [c["per_riga"][k]["kl_media"] - b["per_riga"][k]["kl_media"] for k in common]
    m = sum(dd) / len(dd)
    se = (sum((x - m) ** 2 for x in dd) / max(len(dd) - 1, 1)) ** .5 / len(dd) ** .5
    return Measurement(kl=c["kl_media"], kl_base=b["kl_media"], kl_delta=m, kl_se=se,
                       source=f"{candidate_json} vs {base_json}", extra={"rows": len(common)}).complete()


def read_mcnemar(mcnemar_json: str) -> Measurement:
    """``mcnemar_mclp.py`` output (A = reference, B = candidate)."""
    d = json.loads(Path(mcnemar_json).read_text())
    return Measurement(mmlu=d["acc_B"], mmlu_base=d["acc_A"], p_mcnemar=d["p_mcnemar_esatto"],
                       wins=d.get("B_giusta_A_sbagliata"), losses=d.get("A_giusta_B_sbagliata"), source=mcnemar_json)


_TG = re.compile(r"\|\s*tg128\s*\|\s*([0-9.]+)")
_PP = re.compile(r"\|\s*pp512\s*\|\s*([0-9.]+)")


def read_bench(txt: str) -> dict:
    t = Path(txt).read_text(errors="replace")
    tg = _TG.search(t)
    pp = _PP.search(t)
    return {"tg": float(tg.group(1)) if tg else None, "pp": float(pp.group(1)) if pp else None}


def read_combo_dir(directory: str, name: str, ref_kl: str, ref_name: str = "R2B3m") -> Measurement:
    """Results of the measurement script in DIRECTORY for variant NAME (kl/kl_NAME.json, bench_*.txt,
    mcnemar_NAME.json)."""
    d = Path(directory)
    m = read_kl(str(d / "kl" / f"kl_{name}.json"), ref_kl) if (d / "kl" / f"kl_{name}.json").exists() else Measurement()
    bc, bb = d / f"bench_{name}.txt", d / f"bench_{ref_name}.txt"
    if bc.exists():
        m.tg = read_bench(str(bc))["tg"]
    if bb.exists():
        m.tg_base = read_bench(str(bb))["tg"]
    for f in d.glob(f"mcnemar*{name}*.json"):
        mm = read_mcnemar(str(f))
        m.mmlu, m.mmlu_base, m.p_mcnemar, m.wins, m.losses = mm.mmlu, mm.mmlu_base, mm.p_mcnemar, mm.wins, mm.losses
    return m


# ── measurement job (validated script, one copy per run as its header asks) ──
def measurement_job(id_: str, name: str, gguf: str, directory: str, ref_kl: Optional[str] = None,
                    ref_name: str = "R2B3m", mmlu: bool = True, cap_bytes: Optional[int] = None,
                    engine_bin: Optional[str] = None, ref_mmlu: Optional[str] = None, **meta: Any) -> Job:
    """The script's interface is its environment (C_RIF_KL, C_RIF_NOME, C_MMLU, C_TETTO, C_TETTO_MIN, C_BIN,
    C_RIF_MMLU) and its arguments ``NAME DIR NAME=GGUF``."""
    paths = config.current().paths
    env = {"C_RIF_KL": ref_kl or paths["kl_ref"], "C_RIF_NOME": ref_name, "C_MMLU": "1" if mmlu else "0",
           "C_TETTO": str(cap_bytes or DEFAULT_FILE_CAP), "C_TETTO_MIN": "120"}
    if engine_bin:
        env["C_BIN"] = engine_bin
    if ref_mmlu:
        env["C_RIF_MMLU"] = ref_mmlu
    copy = f"{directory}/measure_{id_}.sh"
    script = (f"mkdir -p {directory} && cp {paths['combo_measure']} {copy} && "
              + " ".join(f"{k}={v}" for k, v in env.items()) + f" bash {copy} {name} {directory} {name}={gguf}")
    return Job(id=id_, name=f"measure {name}: int8 realised KL + tg128" + (" (+MMLU if the KL improves)" if mmlu else ""),
               command=bash(script), gpu=True, est_h=0.6 + (1.0 if mmlu else 0.0), output_dir=directory, **meta)

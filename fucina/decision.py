"""Pure decision rules (SPEC section 4 steps 2, 5, 7 and section 5): objective, threshold verdict, KL->MMLU law,
priority. No I/O: everything is unit-testable."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

from .measure import GIB, Measurement
from .registry import Outcome, Thresholds

# "resident_ram" rule, measured on the reference model: a file <= ~8.84 GiB stays under 9 GiB of RAM at ctx 8192
DEFAULT_FILE_CAP_BYTES = 9_492_293_427
RAM_TO_FILE_RATIO = DEFAULT_FILE_CAP_BYTES / (9 * GIB)
BYTE_TOLERANCE = 1.003                 # historic sizes are recorded in GiB rounded to 2 decimals


# ── objective ──
@dataclass
class Objective:
    mmlu_min: Optional[float] = None
    ram_max_gib: Optional[float] = None
    file_max_bytes: Optional[int] = None
    tg_min: Optional[float] = None
    text: str = ""

    @property
    def cap_bytes(self) -> int:
        if self.file_max_bytes:
            return int(self.file_max_bytes)
        if self.ram_max_gib:
            return int(self.ram_max_gib * GIB * RAM_TO_FILE_RATIO)
        return DEFAULT_FILE_CAP_BYTES

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v not in (None, "")}


def _quantity(v: str) -> float:
    g = re.fullmatch(r"([\d.,_]+)\s*(gib|gb|b)?", v.strip().lower())
    if not g:
        raise ValueError(f"invalid quantity: {v}")
    x = float(g.group(1).replace(",", ".").replace("_", ""))
    return x * GIB if g.group(2) == "gib" else x * 1e9 if g.group(2) == "gb" else x


def parse_objective(text: str, presets: Optional[dict] = None) -> Objective:
    """``mmlu>=80.05,ram<=9GiB,tg>=50,file<=13085779616`` or the name of an objective of the model (e.g. ``A``)."""
    if presets and text in presets:
        o = parse_objective(presets[text])
        o.text = f"{text}: {o.text}"
        return o
    o = Objective(text=text)
    for part in filter(None, (p.strip() for p in text.split(","))):
        g = re.fullmatch(r"(mmlu|ram|file|tg)\s*(>=|<=)\s*(.+)", part, re.I)
        if not g:
            raise ValueError(f"constraint not understood: {part!r} (use: mmlu>=80.05,ram<=9GiB,tg>=50,file<=N)")
        k, v = g.group(1).lower(), g.group(3)
        if k == "mmlu":
            o.mmlu_min = float(v.replace(",", "."))
        elif k == "tg":
            o.tg_min = float(v.replace(",", "."))
        elif k == "ram":
            o.ram_max_gib = _quantity(v) / GIB if re.search(r"[a-z]", v, re.I) else float(v)
        elif k == "file":
            o.file_max_bytes = int(_quantity(v))
    return o


# ── KL -> MMLU law (rule kl_mmlu_law) ──
@dataclass
class Law:
    """MMLU ~ base + coeff * (KL drop %)^2. The defaults are the law fitted on Thor-T (MMLU-2000)."""
    base: float = 74.60
    coeff: float = 0.00186

    def mmlu(self, kl_drop_pct: float) -> float:
        """Negative kl_drop_pct = KL lower than the law's reference."""
        return self.base + self.coeff * min(kl_drop_pct, 0.0) ** 2

    def drop_for(self, mmlu: float) -> float:
        return -math.sqrt(max(mmlu - self.base, 0.0) / self.coeff)


def compose_drops(*drops_pct: float) -> float:
    """Relative KL drops in series: (1+a)(1+b)-1, in %."""
    f = 1.0
    for c in drops_pct:
        f *= 1.0 + c / 100.0
    return 100.0 * (f - 1.0)


# ── threshold verdict (step 5) ──
@dataclass
class Verdict:
    verdict: str                     # PASS | FAIL | AMBIGUOUS
    reason: str
    advice: str = ""                 # enable | disable | remeasure
    to_verify: bool = False          # "published lever" rule
    quality: Optional[float] = None
    numbers: dict = field(default_factory=dict)

    def outcome(self) -> Outcome:
        n = dict(self.numbers)
        n["advice"] = self.advice
        if self.to_verify:
            n["to_verify"] = True
        return Outcome(self.verdict, n, self.reason)


def _kl_quality(m: Measurement, s: Thresholds) -> tuple:
    """+1 clear improvement, 0.5 weak, 0 no effect, -1 clearly worse, None unknown."""
    if m.kl_pct is None:
        return None, ""
    z = m.z_kl
    if z is not None:
        if z <= -s.sigma and m.kl_pct <= s.kl_min_pct:
            return 1.0, f"KL {m.kl_pct:+.2f} % ({z:+.1f} se)"
        if z >= s.sigma:
            return -1.0, f"KL worse {m.kl_pct:+.2f} % ({z:+.1f} se)"
        if z > -1.0:
            return 0.0, f"KL flat {m.kl_pct:+.2f} % ({z:+.1f} se)"
        return 0.5, f"KL weakly lower {m.kl_pct:+.2f} % ({z:+.1f} se, threshold {s.sigma} se and {s.kl_min_pct} %)"
    # without a standard error only large effects are clear (5 % ~ 2 se, typical at 24 rows)
    if m.kl_pct <= min(s.kl_min_pct, -5.0):
        return 1.0, f"KL {m.kl_pct:+.1f} % (no se, large effect)"
    if m.kl_pct >= 5.0:
        return -1.0, f"KL worse {m.kl_pct:+.1f} %"
    return 0.5 if m.kl_pct < 0 else 0.0, f"KL {m.kl_pct:+.1f} % without standard error"


def _mmlu_quality(m: Measurement) -> tuple:
    d = m.mmlu_delta
    if d is None:
        return None, ""
    se = m.mmlu_se()
    sig = (m.p_mcnemar < 0.05) if m.p_mcnemar is not None else abs(d) >= 2 * se
    pt = f"p {m.p_mcnemar:.3g}" if m.p_mcnemar is not None else f"~{abs(d) / se:.1f} se"
    if sig:
        return (1.0 if d > 0 else -1.0), f"MMLU {m.mmlu:.2f} vs {m.mmlu_base:.2f} ({d:+.2f}, {pt})"
    return 0.0, f"MMLU flat {m.mmlu:.2f} vs {m.mmlu_base:.2f} ({d:+.2f}, {pt})"


def judge(m: Measurement, thresholds: Thresholds, objective: Objective, judged_by: str = "kl", source: str = "",
          law: Optional[Law] = None, slows_down: bool = False) -> Verdict:
    law = law or Law()
    num = m.as_dict()
    # size / RAM constraint
    if m.file_bytes and m.file_bytes > objective.cap_bytes * BYTE_TOLERANCE:
        return Verdict("FAIL", f"{m.file_bytes} B above the cap {objective.cap_bytes} B", "disable", numbers=num)
    # speed
    tg_ok, tg_text = True, ""
    if m.tg is not None:
        tp = m.tg_pct
        if tp is not None and tp < -50.0 and judged_by != "speed":
            if slows_down:      # a lever that costs speed by nature: a collapse beyond half is more likely noise
                return Verdict("AMBIGUOUS", f"suspicious tg {m.tg:.1f} vs {m.tg_base:.1f} ({tp:+.1f} %): a lever that "
                               f"slows down is expected around -15 %, remeasure the speed on an idle machine",
                               "remeasure", numbers=num)
            return Verdict("FAIL", f"speed collapse tg {m.tg:.1f} ({tp:+.1f} %)", "disable", numbers=num)
        if objective.tg_min is not None:
            tg_ok = m.tg >= objective.tg_min
            tg_text = f"tg {m.tg:.1f} {'>=' if tg_ok else '<'} minimum {objective.tg_min:g}"
        elif tp is not None:
            tg_ok = tp >= -thresholds.tg_max_loss_pct
            tg_text = f"tg {tp:+.1f} % (tolerance {thresholds.tg_max_loss_pct:g} %)"
    qm, tm = _mmlu_quality(m)
    if judged_by == "speed":
        tp = m.tg_pct
        if qm == -1.0:
            return Verdict("FAIL", f"speed lever that lowers MMLU: {tm}", "disable", numbers=num)
        if tp is None:
            return Verdict("AMBIGUOUS", "speed lever without a paired tg measurement", "remeasure", numbers=num)
        if tp >= 3.0 and qm in (0.0, 1.0, None):
            return Verdict("PASS", f"tg {tp:+.1f} %" + (f", {tm}" if tm else ""), "enable", numbers=num)
        if tp < 0:
            return Verdict("FAIL", f"speed lever that slows down (tg {tp:+.1f} %)", "disable", numbers=num)
        return Verdict("AMBIGUOUS", f"small speed gain (tg {tp:+.1f} %)", "enable", numbers=num)
    qk, tk = _kl_quality(m, thresholds)
    if qm == -1.0:
        return _fail(f"worse: {tm}" + (f"; {tk}" if tk else ""), source, num)
    if qm == 1.0:
        q, reason = 1.0, tm + (f"; {tk}" if tk else "")
    elif qk is None:
        if qm is None:
            return Verdict("AMBIGUOUS", "no quality measurement (neither KL nor MMLU)", "remeasure", numbers=num)
        q, reason = 0.0, tm
    else:
        q, reason = qk, tk + (f"; {tm}" if tm else "")
        d = m.mmlu_delta
        if q == 1.0 and d is not None and d < -m.mmlu_se():
            return Verdict("AMBIGUOUS", f"better KL but non-significant MMLU drop: {reason}", "remeasure", quality=q,
                           numbers=num)
    if q == 1.0:
        if tg_ok:
            return Verdict("PASS", reason + (f"; {tg_text}" if tg_text else ""), "enable", quality=q, numbers=num)
        gain = law.mmlu(m.kl_pct) - law.base if m.kl_pct is not None else (m.mmlu_delta or 0.0)
        adv = "enable" if gain >= 0.5 or qm == 1.0 else "disable"
        return Verdict("AMBIGUOUS", f"improves but slows down: {reason}; {tg_text}; expected MMLU gain {gain:+.2f}",
                       adv, quality=q, numbers=num)
    if q == 0.5:
        return Verdict("AMBIGUOUS", reason, "remeasure", quality=q, numbers=num)
    return _fail(("no measurable effect: " if q == 0.0 else "") + reason, source, num, q)


def _fail(reason: str, source: str, num: dict, q: Optional[float] = None) -> Verdict:
    published = bool(re.search(r"arXiv|\d{4}\.\d{4,5}", source or ""))
    if published:
        reason += " — published lever: «to verify» until the reported number is reproduced or the cause is shown"
    return Verdict("FAIL", reason, "disable", to_verify=published, quality=q, numbers=num)


# ── step 7: is an MMLU run worth it? ──
def needs_mmlu(expected_kl_pct: Optional[float], visible_threshold_pct: float = -25.0) -> bool:
    """MMLU only if the expected KL drop is visible through the law (rule kl_mmlu_law: threshold -25 %)."""
    return expected_kl_pct is not None and expected_kl_pct <= visible_threshold_pct


# ── step 2: priority ──
def priority(kl_gain_pct: float, gpu_cost_h: float, rule_weight: float = 1.0) -> float:
    """Expected gain (KL drop %, positive = better) per GPU hour, weighted by the rules."""
    return max(-kl_gain_pct, 0.0) / max(gpu_cost_h, 0.25) * rule_weight

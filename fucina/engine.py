"""Decision engine (SPEC section 4): per phase and per piece, the admissible techniques in order of expected gain per
GPU hour; measurements come first from the database (oracle), then from the queue (executor); threshold verdict
PASS/FAIL/AMBIGUOUS; ambiguous ones become questions (questions.json, ``fucina decide``); per-byte levers that pass
are combined on top of the structure (second plane) and the realignment (F4) runs ONCE on top of the combination.

The engine is recomputable: every ``fucina auto`` starts again from the database + queue.json + questions.json +
switches.json (state on disk -> resume), recognises the jobs already done and reads their results."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from . import config
from . import decision as D
from .candidate import Candidate, Choice, RunState
from .db import Measurements, load_rules, rules_for
from .executor import Executor, Job, bash, log_cycle
from .measure import Measurement, measurement_job, read_combo_dir
from .oracle import Oracle, Signature
from .phases import PHASES
from .registry import Thresholds, load_techniques

STATES = ("consolidated", "passed", "superseded", "testing", "waiting", "to_decide", "off", "suspended", "excluded",
          "missing_tool", "deferred")
ENABLED = ("consolidated", "passed")
ABLATION_THRESHOLD_PCT = 0.3    # a lever that removes less than this from the combination's KL is not kept
WHOLE_MODEL = "(whole model)"


# ── questions and switches on disk ──
class QuestionQueue:
    """questions.json: ambiguous choices with their numbers and the advice; answers: enable | disable | defer."""
    ANSWERS = ("enable", "disable", "defer")

    def __init__(self, run_dir: Path):
        self.file = Path(run_dir) / "questions.json"
        self.items: list = json.loads(self.file.read_text()) if self.file.exists() else []

    def save(self) -> None:
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.items, ensure_ascii=False, indent=1))
        tmp.replace(self.file)

    def find(self, id_: str) -> Optional[dict]:
        return next((q for q in self.items if q["id"] == id_), None)

    def ask(self, id_: str, **c: Any) -> Optional[str]:
        q = self.find(id_)
        if q is None:
            q = {"id": id_, "asked": time.strftime("%Y-%m-%dT%H:%M"), "answer": None, **c}
            self.items.append(q)
            self.save()
        elif q.get("answer") is None:
            q.update(c)
            self.save()
        return q.get("answer")

    def answer(self, id_: str, answer: str, note: str = "") -> dict:
        if answer not in self.ANSWERS:
            raise ValueError(f"answer {answer!r}: use {', '.join(self.ANSWERS)}")
        q = self.find(id_)
        if q is None:
            raise KeyError(id_)
        q["answer"] = answer
        q["answered"] = time.strftime("%Y-%m-%dT%H:%M")
        if note:
            q["user_note"] = note
        self.save()
        return q

    def open(self) -> list:
        return [q for q in self.items if q.get("answer") is None]


class Switches:
    """switches.json: user choices (TUI/CLI/forge) that override the engine: {technique: enable|disable}."""

    def __init__(self, run_dir: Path):
        self.file = Path(run_dir) / "switches.json"
        self.items: dict = json.loads(self.file.read_text()) if self.file.exists() else {}

    def set(self, technique: str, value: Optional[str]) -> None:
        if value is None:
            self.items.pop(technique, None)
        else:
            assert value in ("enable", "disable"), value
            self.items[technique] = value
        self.file.write_text(json.dumps(self.items, ensure_ascii=False, indent=1))

    def get(self, technique: str) -> Optional[str]:
        return self.items.get(technique)


# ── evaluation of a technique ──
@dataclass
class Evaluation:
    technique: str
    phase: str
    pieces: list
    state: str
    reason: str = ""
    verdict: str = ""
    numbers: dict = field(default_factory=dict)
    gguf: str = ""
    switch: str = ""
    jobs: list = field(default_factory=list)
    base: str = ""                    # label of the candidate it is judged against
    priority: float = 0.0
    to_verify: bool = False
    advice: str = ""


@dataclass
class Context:
    """What the techniques see (ScriptTechnique contract)."""
    candidate: Candidate
    executor: Executor
    outputs_dir: str
    cap_bytes: int
    simulate: bool
    params: dict
    suffix: str = ""
    depends_on: list = field(default_factory=list)
    placeholder: bool = False


def _h(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:6]


def load_model(name_or_file: str) -> dict:
    """A model YAML by path, or ``<name>.yaml`` in one of the configured ``models_dirs``."""
    p = Path(name_or_file)
    if not p.exists():
        dirs = config.current()["models_dirs"]
        p = next((Path(d) / f"{name_or_file.lower()}.yaml" for d in dirs
                  if (Path(d) / f"{name_or_file.lower()}.yaml").exists()),
                 Path(dirs[0] if dirs else ".") / f"{name_or_file.lower()}.yaml")
    if not p.exists():
        raise FileNotFoundError(f"model {name_or_file}: {p} not found (set models_dirs in the fucina config)")
    d = config._expand(yaml.safe_load(p.read_text()))
    d["_file"] = str(p)
    return d


def _copy(c: Candidate, **kw: Any) -> Candidate:
    return Candidate(**{**asdict(c), "choices": [], **kw})


class Engine:
    def __init__(self, model: str, objective: str, run_dir: str, mode: str = "simulate",
                 db: Optional[str] = None, params: Optional[dict] = None, outputs_dir: Optional[str] = None,
                 rules: Optional[list] = None):
        cfg = config.current()
        self.model = load_model(model)
        self.objective_arg = objective
        self.objective = D.parse_objective(objective, self.model.get("objectives"))
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.simulate = mode == "simulate"
        self.params = params or {}
        self.outputs = outputs_dir or (str(self.dir / "outputs") if self.simulate else f"{cfg['outputs']}/{self.dir.name}")
        # only the techniques of techniques/*.py (the registry may hold test classes registered elsewhere)
        self.techniques = {k: t for k, t in load_techniques().items()
                           if type(t).__module__.startswith(f"{__package__}.techniques")}
        self.measurements = Measurements(db)
        self.oracle = Oracle(self.measurements, self.model["name"], self.techniques)
        self.rules = rules if rules is not None else load_rules()
        self.executor = Executor(str(self.dir), mode)
        self.questions = QuestionQueue(self.dir)
        self.switches = Switches(self.dir)
        self.state = RunState(str(self.dir))
        law = self.model.get("kl_mmlu_law") or {}
        self.law = D.Law(**law) if law else D.Law()
        self.signatures: dict = {}
        for label, c in (self.model.get("candidates") or {}).items():
            self.signatures[c["gguf"]] = Signature(label=label, gguf=c["gguf"], mmlu=list(c.get("mmlu") or []),
                                                   kl=c.get("kl"), file_bytes=c.get("file_bytes"), tg=c.get("tg"),
                                                   kl_pct=c.get("kl_pct"), kl_json=cfg.resolve(c.get("kl_json", "")))
        self.evaluations: list = []
        self.events: list = []
        self.measure_architectures = tuple(cfg["measure_architectures"])

    # ── helpers ──
    @property
    def cap(self) -> int:
        return self.objective.cap_bytes

    def signature(self, cand: Candidate) -> Signature:
        s = self.signatures.get(cand.gguf)
        if s is None:
            s = Signature(label=cand.label or Path(cand.gguf).stem, gguf=cand.gguf)
            self.signatures[cand.gguf] = s
        return s

    def ctx(self, cand: Candidate, **kw: Any) -> Context:
        return Context(candidate=cand, executor=self.executor, outputs_dir=self.outputs, cap_bytes=self.cap,
                       simulate=self.simulate, params=self.params, **kw)

    def event(self, text: str) -> None:
        self.events.append(text)

    def rule_weight(self, t: Any, phase: str, piece: str) -> float:
        w = 1.0
        for r in rules_for(phase, piece, self.model["model_type"], self.rules):
            eff = r.get("effect") or {}
            if eff.get("criterion") and eff["criterion"] == getattr(t, "name", None):
                w *= 1.0 + r.get("confidence", 0.5)
            if r.get("id") == "local_without_realignment" and getattr(t, "judged_by", "") == "kl" and phase in ("F1", "F2"):
                w *= 1.0 - 0.3 * r.get("confidence", 0.5)
        return w

    def enabling_rule(self, phase: str) -> Optional[dict]:
        for r in rules_for(phase, "", self.model["model_type"], self.rules):
            if (r.get("effect") or {}).get("enable"):
                return r
        return None

    def phase_techniques(self, phase: str) -> list:
        return [t for t in self.techniques.values() if t.phase == phase]

    def pieces_of(self, t: Any) -> list:
        pz = list(t.pieces) if t.pieces else [WHOLE_MODEL]
        return [p for p in pz if p in self.model.get("pieces", pz) or p == WHOLE_MODEL] or pz

    # ── measurement: from the database or queued ──
    def reference_kl(self, cand: Candidate, sig: Signature, depends: list, placeholder: bool) -> tuple:
        """KL JSON of the candidate (reference of the paired measurements) and the jobs it depends on."""
        if sig.kl_json:
            return sig.kl_json, depends
        d = f"{self.outputs}/measurements/ref_{sig.label}"
        id_ = f"M_ref_{sig.label}_{_h(cand.gguf)}"
        exists = bool(cand.gguf) and Path(cand.gguf).exists()     # candidate already on disk: measure it right away
        self.executor.enqueue(measurement_job(id_, sig.label, cand.gguf, d, mmlu=False, cap_bytes=self.cap,
                                              depends_on=[] if exists else list(depends),
                                              placeholder=False if exists else placeholder, phase=cand.current_phase,
                                              technique="reference"))
        self.executor.find(id_).name = f"reference {sig.label}: int8 realised KL + tg128 (base of the paired measurements)"
        return f"{d}/kl/kl_{sig.label}.json", list(depends) + [id_]

    def read_measurement_job(self, id_: str, technique: str, sig: Signature, ref_kl: str,
                             reader: Optional[Callable[[], Optional[Measurement]]] = None) -> Optional[Measurement]:
        j = self.executor.find(id_)
        if j is None or j.state != "done":
            return None
        source = f"fucina:run:{self.dir.name}:{id_}"
        if not any(r.get("source") == source for r in self.measurements.all()):
            name = j.env.get("FUCINA_NAME", technique)
            m = reader() if reader else read_combo_dir(j.output_dir, name, ref_kl, sig.label)
            if m is None:
                return None
            par = {"base": sig.label, "variant": name, "kl_se": m.kl_se, "wins": m.wins, "losses": m.losses}
            self.measurements.add(model=self.model["name"], technique=technique, phase=j.phase, piece=j.piece, kl=m.kl,
                                  kl_base=m.kl_base, kl_pct=m.kl_pct, mmlu=m.mmlu, mmlu_base=m.mmlu_base,
                                  p_mcnemar=m.p_mcnemar, tg=m.tg, tg_base=m.tg_base, file_bytes=m.file_bytes,
                                  params=json.dumps(par), source=source, note=f"Fucina measurement ({j.name})")
            self.oracle.invalidate()
        return self.oracle.referred(technique, sig, self.cap)

    def enqueue_trial(self, t: Any, cand: Candidate, sig: Signature, piece: str, depends: list = (),
                      placeholder: bool = False, label: str = "") -> tuple:
        """Construction (wrapper) + paired measurement against the candidate. Returns (measurement id, ids, gguf,
        ref_kl)."""
        suffix = _h(cand.gguf)
        c = self.ctx(cand, suffix=suffix, depends_on=list(depends), placeholder=placeholder)
        change = t.apply(c, piece)
        ids = list(change.job_ids)
        if getattr(t, "self_measuring", False):
            return (ids[-1] if ids else ""), ids, change.gguf, ""
        ref, dep = self.reference_kl(cand, sig, list(depends), placeholder)
        name = label or t.name
        mmlu = D.needs_mmlu(getattr(t, "expected_kl_gain_pct", None))
        id_m = f"M_{t.phase}_{t.name}_{suffix}"
        engine = getattr(t, "engine", "bin_planes")
        mj = measurement_job(id_m, name, change.gguf, f"{self.outputs}/measurements/{t.phase}_{t.name}_{suffix}",
                             ref_kl=ref, ref_name=sig.label, mmlu=mmlu, cap_bytes=self.cap,
                             engine_bin=config.current().paths.get(engine) if engine != "bin_planes" else None,
                             depends_on=list(dict.fromkeys(dep + ids)), placeholder=placeholder, phase=t.phase,
                             technique=t.name, piece=piece)
        mj.env["FUCINA_NAME"] = name
        mj = self.executor.enqueue(mj)
        return mj.id, ids + [mj.id], change.gguf, ref

    def exclusion(self, t: Any, phase: str) -> str:
        """Why the technique does not apply to THIS model ("" = admitted)."""
        if not t.admits(phase, t.pieces[0] if t.pieces else "", self.model["model_type"],
                        self.model.get("constraints") or {}):
            return ("not ternary (constraint all_ternary)" if not t.ternary
                    else f"not admitted for a {self.model['model_type']} model")
        arch = self.model.get("architecture")
        if arch and getattr(t, "architectures", None) and arch not in t.architectures:
            return (f"wrapped scripts validated only for {', '.join(t.architectures)} (model {arch}): "
                    f"a wrapper for this architecture is needed")
        if t.pieces and self.model.get("pieces") and not set(t.pieces) & set(self.model["pieces"]):
            return f"the model has none of the pieces {', '.join(t.pieces)}"
        return ""

    # ── judging a technique against the candidate ──
    def evaluate(self, t: Any, cand: Candidate, phase: str, placeholder: bool = False, depends: list = (),
                 wait: bool = False) -> Evaluation:
        """wait=True: the candidate is not the final one (structure or combination still being tested) -> database
        measurements only count as evidence and the technique stays waiting (e.g. the F4 realignment)."""
        sig = self.signature(cand)
        pieces = self.pieces_of(t)
        v = Evaluation(t.name, phase, pieces, "testing", base=sig.label, switch=t.switch())
        gain = self.oracle.best_historic_gain(t.name)
        gain = gain if gain is not None and gain < 0 else getattr(t, "expected_kl_gain_pct", -1.0)
        v.priority = round(D.priority(gain, t.gpu_cost_h or 0.25, self.rule_weight(t, phase, pieces[0])), 2)
        user = self.switches.get(t.name)
        excl = self.exclusion(t, phase)
        if excl:
            v.state, v.reason = "excluded", excl
            return v
        if user == "disable":
            v.state, v.reason = "off", "disabled by the user"
            return v
        if getattr(t, "suspended", "") and user != "enable":
            v.state, v.reason = "suspended", t.suspended
            return v
        if getattr(t, "default_off", "") and user != "enable":
            v.state, v.reason = "off", f"opt-in module ({t.default_off}): `fucina enable {t.name}`"
            return v
        e = t.cheap_test(self.ctx(cand))
        if e.verdict == "UNINFORMATIVE":
            v.state, v.reason, v.numbers = "missing_tool", e.reason, e.numbers
            return v
        # 1) measurement already made against THIS candidate (database or finished job)
        m = None if wait else self.oracle.referred(t.name, sig, self.cap)
        suffix = _h(cand.gguf)
        if getattr(t, "self_measuring", False):
            jj = t.jobs(self.ctx(cand), pieces[0])
            id_m = f"{jj[-1].id}_{suffix}" if jj else ""
            c0 = self.ctx(cand)
            reader = lambda: t.own_measurement(c0, pieces[0])
        else:
            id_m, reader = f"M_{t.phase}_{t.name}_{suffix}", None
        if m is None and not wait and id_m and self.executor.find(id_m):
            ref = sig.kl_json or f"{self.outputs}/measurements/ref_{sig.label}/kl/kl_{sig.label}.json"
            m = self.read_measurement_job(id_m, t.name, sig, ref, reader)
        if m is not None:
            ver = D.judge(m, t.thresholds, self.objective, t.judged_by, t.source, self.law, t.slows_down)
            self._apply_verdict(v, t, ver, m, cand, source=m.source)
            if user == "enable" and v.state != "passed":
                v.state, v.reason = "passed", f"enabled by the user ({v.reason})"
            return v
        if user == "enable":
            v.state, v.reason = "passed", "enabled by the user without a measurement"
            v.gguf = t.product_gguf(self.ctx(cand), pieces[0])
            return v
        # 2) nearby evidence (same technique against another candidate)
        near = self.oracle.nearby(t.name, sig, self.cap, tuple(self.signatures.values()))
        if wait:
            ref_m = self.oracle.referred(t.name, sig, self.cap)
            if ref_m is not None:
                near = (ref_m, f"{sig.label} (provisional candidate)")
        if near is not None and wait:
            mv, base = near
            ver = D.judge(mv, t.thresholds, self.objective, t.judged_by, t.source, self.law, t.slows_down)
            v.numbers = {"nearby_evidence": mv.as_dict(), "nearby_verdict": ver.verdict, "nearby_base": base}
        elif near is not None:
            mv, base = near
            ver = D.judge(mv, t.thresholds, self.objective, t.judged_by, t.source, self.law, t.slows_down)
            rule = self.enabling_rule(phase)
            if ver.verdict == "FAIL":
                v.state, v.verdict, v.numbers = "off", "FAIL", mv.as_dict()
                v.reason = f"measured against {base}: {ver.reason} (can be remeasured against {sig.label})"
                v.to_verify = ver.to_verify
                return v
            if ver.verdict == "PASS" and rule:
                v.state, v.verdict, v.numbers = "consolidated", "PASS", mv.as_dict()
                v.reason = f"rule {rule['id']} (confidence {rule.get('confidence')}): {ver.reason} against {base}"
                v.gguf = t.product_gguf(self.ctx(cand), pieces[0])
                return v
            v.numbers = {"nearby_evidence": mv.as_dict(), "nearby_verdict": ver.verdict, "nearby_base": base}
        # 3) queued: construction + paired measurement
        id_m, ids, gguf, _ = self.enqueue_trial(t, cand, sig, pieces[0], depends, placeholder)
        v.jobs, v.gguf = ids, gguf
        v.state = "waiting" if placeholder else "testing"
        v.reason = (f"queued: construction + measurement against {sig.label} (expected gain {gain:+.1f} % KL)"
                    + (f"; against {v.numbers['nearby_base']}: {v.numbers['nearby_verdict']}"
                       if v.numbers.get("nearby_base") else ""))
        return v

    def _apply_verdict(self, v: Evaluation, t: Any, ver: D.Verdict, m: Measurement, cand: Candidate,
                       source: str = "") -> None:
        v.verdict, v.numbers, v.advice, v.to_verify = ver.verdict, m.as_dict(), ver.advice, ver.to_verify
        v.gguf = t.product_gguf(self.ctx(cand), v.pieces[0])
        if ver.verdict == "PASS":
            v.state, v.reason = "passed", ver.reason
        elif ver.verdict == "FAIL":
            v.state, v.reason = "off", ver.reason + (" [to verify]" if ver.to_verify else "")
        elif ver.advice == "carry":       # 22 Oct 2026: flat-not-worse = branch head kept automatically, no question
            v.state, v.reason = "deferred", f"CARRY (branch head, re-tested with the next lever): {ver.reason}"
        else:
            label = self.signature(cand).label
            id_q = f"{t.name}@{label}"
            a = self.questions.ask(id_q, technique=t.name, phase=t.phase, candidate=label,
                                   text=f"{t.name} against {label}: {ver.reason}",
                                   numbers=m.as_dict(), advice=ver.advice, objective=self.objective.text, source=source)
            if a == "enable":
                v.state, v.reason = "passed", f"enabled by the user (AMBIGUOUS: {ver.reason})"
            elif a == "disable":
                v.state, v.reason = "off", f"disabled by the user (AMBIGUOUS: {ver.reason})"
            elif a == "defer":
                v.state, v.reason = "deferred", f"deferred by the user (AMBIGUOUS: {ver.reason})"
            else:
                v.state = "to_decide"
                v.reason = f"AMBIGUOUS: {ver.reason} — advice: {ver.advice} (`fucina decide`)"

    # ── the full pass ──
    def run(self) -> dict:
        self.evaluations, self.events = [], []
        self.executor.touched = set()
        start = self.model.get("start", "")
        cand = Candidate(model=self.model["name"], model_type=self.model["model_type"], gguf=start,
                         objective=self.objective.as_dict(), label="bf16")
        cand.history.append({"phase": "-", "label": "bf16", "gguf": start, "reason": "start"})
        waiting: list = []               # jobs everything that follows depends on (structure not decided yet)
        concluded: dict = {}

        def add(v: Evaluation) -> None:
            self.evaluations.append(v)
            for p in v.pieces:
                cand.choices.append(Choice(v.phase, p, v.technique, _choice_state(v.state), v.numbers, v.reason, v.switch))

        def consolidate(phase: str, label: str, gguf: str, reason: str) -> None:
            cand.gguf, cand.label = gguf, label
            cand.history.append({"phase": phase, "label": label, "gguf": gguf, "reason": reason})
            self.event(f"{phase}: candidate -> {label} ({reason})")

        # F0 — diagnosis
        for t in self._order(self.phase_techniques("F0"), "F0"):
            e = t.cheap_test(self.ctx(cand))
            v = Evaluation(t.name, "F0", self.pieces_of(t), "testing", switch=t.switch(), base=cand.label)
            if self.exclusion(t, "F0"):
                v.state, v.reason = "excluded", self.exclusion(t, "F0")
            elif getattr(t, "default_off", "") and self.switches.get(t.name) != "enable":
                v.state, v.reason = "off", f"opt-in module ({t.default_off}): `fucina enable {t.name}`"
            elif e.verdict == "PASS":
                v.state, v.reason, v.numbers = "consolidated", e.reason, e.numbers
            elif e.verdict == "UNINFORMATIVE":
                v.state, v.reason = "missing_tool", e.reason
            else:
                change = t.apply(self.ctx(cand), v.pieces[0])
                v.jobs = list(change.job_ids)
                v.reason = "diagnosis queued: " + (e.reason or "")
                waiting += v.jobs
            add(v)
        concluded["F0"] = not waiting

        # F1-F3: alternatives against the candidate at the start of the phase; structures consolidated at the end
        # of the phase; levers -> combination
        levers: list = []                # evaluations that are candidates for the combination (passed or testing)
        structure_jobs: list = []
        for phase in ("F1", "F2", "F3"):
            cand.current_phase = phase
            phase_base = _copy(cand)
            structures = []
            for t in self._order(self.phase_techniques(phase), phase):
                if getattr(t, "judged_by", "") == "structure":
                    v = self._structure(t, cand, phase)
                    add(v)
                    if v.state == "consolidated":
                        consolidate(phase, v.base or t.name, v.gguf, v.reason)
                        phase_base = _copy(cand)
                    elif v.state == "testing":
                        structure_jobs += v.jobs
                        waiting += v.jobs
                    continue
                v = self.evaluate(t, phase_base, phase, placeholder=bool(waiting), depends=list(waiting))
                add(v)
                if getattr(t, "structural", False):
                    structures.append((t, v))
                elif v.state in ("passed", "consolidated", "testing", "waiting", "to_decide"):
                    levers.append((t, v))
            # exclusive groups: among the passed alternatives of one group only the best stays
            self._exclusive(phase)
            for t, v in structures:
                if v.state == "passed":
                    label = self._product_label(v.gguf, t.name, cand)
                    consolidate(phase, label, v.gguf, f"{t.name}: {v.reason}")
                elif v.state == "to_decide":          # ambiguous structure: everything that follows waits for the answer
                    waiting.append(f"QUESTION_{t.name}")
                    self.event(f"{phase}: structure {t.name} AMBIGUOUS — the following phases wait for `fucina decide`")
                elif v.state in ("testing", "waiting"):
                    waiting += list(v.jobs)
                    structure_jobs += v.jobs
                    self.event(f"{phase}: structure {t.name} being tested — the following phases wait")
            concluded[phase] = not any(v.state in ("testing", "waiting", "to_decide") for v in self.evaluations
                                       if v.phase == phase)

        # combination of the levers (per-byte and zero-byte) on top of the structure
        levers = [(t, v) for t, v in levers if v.state in ("passed", "consolidated", "testing", "waiting", "to_decide")]
        comb = self._combination(cand, levers, waiting)
        if comb is not None:
            self.evaluations.append(comb)
            if comb.state == "passed":
                consolidate("COMBO", "COMBO", comb.gguf, comb.reason)
            elif comb.state in ("testing", "waiting"):
                waiting = list(dict.fromkeys(waiting + [i for i in comb.jobs
                                                        if i.startswith("M_COMBO_") and "_without_" not in i]))

        # EXPECTED candidate: the consolidated one, or the product still being tested (combination, then
        # realignment) on top of which the following phases queue their placeholders
        expected = _copy(cand)
        if comb is not None and comb.state in ("testing", "waiting") and comb.gguf:
            expected.gguf, expected.label = comb.gguf, f"{cand.label}+COMBO(expected)"
        # F4 — realignment ONCE on top of the product
        cand.current_phase = "F4"
        for t in self._order(self.phase_techniques("F4"), "F4"):
            v = self.evaluate(t, _copy(expected), "F4", placeholder=bool(waiting), depends=list(waiting),
                              wait=bool(waiting))
            add(v)
            if v.state == "passed" and v.gguf and not waiting:
                consolidate("F4", f"{self.signature(cand).label}+{t.name}", v.gguf, v.reason)
                expected = _copy(cand)
            elif v.state in ("testing", "waiting"):
                waiting = list(dict.fromkeys(waiting + v.jobs[-1:]))
                if v.gguf:
                    expected.gguf, expected.label = v.gguf, f"{expected.label}+{t.name}(expected)"
        # F5 — runtime (engine levers: switched on through the environment)
        cand.current_phase = "F5"
        env = dict((comb.numbers.get("env_disable") or {}) if comb is not None and comb.state in ENABLED else {})
        for t in self._order(self.phase_techniques("F5"), "F5"):
            v = self.evaluate(t, _copy(expected), "F5", placeholder=bool(waiting), depends=list(waiting))
            add(v)
            if v.state in ENABLED and hasattr(t, "env"):
                env.update(t.env())
        # F6 — final evaluation (probe, paired MMLU, tg) of the product
        cand.current_phase = "F6"
        fin = self._final_evaluation(expected, env, waiting)
        self.evaluations.append(fin)
        cand.forecast = self.forecast(cand)
        cand.current_phase = next((p.code for p in PHASES if not concluded.get(p.code, False)), "F6")
        for j in self.executor.jobs:          # placeholders of a previous pass no longer proposed: cancelled
            if j.placeholder and j.id not in self.executor.touched and j.state in ("queued", "simulated", "dry_run"):
                self.executor.cancel(j.id, "superseded by a new engine pass")
        self.cand = cand
        self.env = env
        self.state.save(cand)
        r = self.report()
        (self.dir / "report.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
        (self.dir / "recipe.yaml").write_text(yaml.safe_dump(self.recipe(), allow_unicode=True, sort_keys=False))
        return r

    # ── pieces of the pass ──
    def _order(self, techniques: list, phase: str) -> list:
        def key(t: Any) -> tuple:
            g = self.oracle.best_historic_gain(t.name)
            g = g if g is not None and g < 0 else getattr(t, "expected_kl_gain_pct", -1.0)
            return (getattr(t, "judged_by", "") != "structure",
                    -D.priority(g, t.gpu_cost_h or 0.25, self.rule_weight(t, phase, (t.pieces or ("",))[0])), t.name)
        return sorted(techniques, key=key)

    def _structure(self, t: Any, cand: Candidate, phase: str) -> Evaluation:
        v = Evaluation(t.name, phase, self.pieces_of(t), "testing", switch=t.switch())
        if self.exclusion(t, phase):
            v.state, v.reason = "excluded", self.exclusion(t, phase)
            return v
        if self.switches.get(t.name) == "disable":
            v.state, v.reason = "off", "disabled by the user"
            return v
        g = t.product_gguf(self.ctx(cand), v.pieces[0])
        if g and Path(g).exists():
            v.state, v.gguf = "consolidated", g
            v.base = self._product_label(g, t.name, cand)
            s = self.signatures.get(g)
            v.numbers = {k: getattr(s, k) for k in ("file_bytes", "mmlu", "kl", "tg") if s and getattr(s, k)}
            v.reason = f"structure already built: {Path(g).name}" + (f" ({t.source})" if t.source else "")
            return v
        change = t.apply(self.ctx(cand, suffix=_h(cand.gguf)), v.pieces[0])
        v.jobs, v.gguf = list(change.job_ids), change.gguf
        v.reason = "structure to be built"
        return v

    def _product_label(self, gguf: str, name: str, cand: Optional[Candidate] = None) -> str:
        s = self.signatures.get(gguf)
        if s:
            return s.label
        return f"{cand.label}+{name}" if cand is not None and cand.label else name

    def _exclusive(self, phase: str) -> None:
        groups: dict = {}
        for v in self.evaluations:
            t = self.techniques.get(v.technique)
            if v.phase == phase and t is not None and getattr(t, "group", "") \
                    and getattr(t, "judged_by", "") != "structure" and v.state == "passed":
                groups.setdefault(t.group, []).append(v)
        for g, vv in groups.items():
            if len(vv) < 2:
                continue
            vv.sort(key=lambda v: (v.numbers.get("kl_pct") if v.numbers.get("kl_pct") is not None else 0.0))
            for v in vv[1:]:
                v.state = "superseded"
                v.reason += f" — superseded by {vv[0].technique} in group {g}"

    def _phase_product(self, cand: Candidate, phase: str) -> str:
        """GGUF of the last candidate consolidated in PHASE ("" if none)."""
        return next((s["gguf"] for s in reversed(cand.history) if s["phase"] == phase), "")

    def _combination(self, cand: Candidate, levers: list, waiting: list) -> Optional[Evaluation]:
        """Passed levers (or levers still being tested -> placeholders) on top of the structure: dense tensors of the
        best F1 lever + second plane (combination script, with the K scale) and then the levers that add tensors
        (Prism/CARDAN) rewritten on top; measured with all on and all-but-one (engine switches, no rebuild)."""
        if not levers:
            return None
        cfg = config.current()
        decided = all(v.state in ENABLED for _, v in levers) and not waiting
        sig = self.signature(cand)
        v = Evaluation("combination", "COMBO", ["(per-byte and zero-byte levers)"], "waiting", base=sig.label)
        # different engines cannot live in one binary: the engine of the lever with the best gain stays
        engines: dict = {}
        for t, lv in levers:
            e = getattr(t, "engine", "bin_planes")
            if e != "bin_planes":
                engines.setdefault(e, []).append((t, lv))
        excluded = []
        if len(engines) > 1 and decided:
            gain = lambda t, lv: lv.numbers.get("kl_pct") if lv.numbers.get("kl_pct") is not None else t.expected_kl_gain_pct
            best = min(engines, key=lambda e: min(gain(t, lv) for t, lv in engines[e]))
            excluded = [t.name for e, tt in engines.items() if e != best for t, _ in tt]
        inside = [(t, lv) for t, lv in levers if t.name not in excluded]
        dep = list(dict.fromkeys(waiting + [lv.jobs[-1] for _, lv in levers if lv.jobs]))
        suffix = _h(cand.gguf + ",".join(sorted(t.name for t, _ in inside)))
        out = f"{self.outputs}/combination_{suffix}"
        dense = [(t, lv) for t, lv in inside if t.phase == "F1"]
        dense_base = dense[0][1].gguf if len(dense) == 1 else (
            (self._phase_product(cand, "F1") or cand.gguf) if not dense
            else f"<best of {', '.join(t.name for t, _ in dense)}>")
        plan = self._structure_plan(cand)
        ids, last, gg, done = [], None, cand.gguf, set()
        if plan:
            k = next((str(t.param(self.ctx(cand), "k")) for t, _ in inside if t.name == "k_scale"), "")
            label = f"FUCINA_{suffix}"
            gg = f"{cfg.path('combo_dir')}/{label}/{cand.model}-{label}.gguf"
            copy = f"{out}/combo_experts_fucina.sh"
            s = (f"mkdir -p {out}; cp {cfg.path('combo_experts')} {copy} && C_SOLO_SCRIVI=1 bash {copy} {label} "
                 f"{dense_base} {plan} {self.cap} {k}")
            last = f"COMBO_{suffix}"
            self.executor.enqueue(Job(
                id=last, name=f"combination: dense {Path(dense_base).name} + second plane ({Path(plan).parent.name})"
                + (f" + K scale {k}" if k else "") + ("" if decided else " [placeholder: waiting for outcomes]"),
                command=bash(s), gpu=True, est_h=1.5, depends_on=dep, output_dir=out, phase="COMBO",
                technique="combination", placeholder=not decided))
            ids.append(last)
            done |= {t.name for t, _ in dense} | {"k_scale"}
        elif dense:
            gg = dense_base
            done |= {t.name for t, _ in dense}
        for t, lv in inside:                      # levers that add tensors: rewritten on top of the combined product
            if t.name in done:
                continue
            c2 = Candidate(**{**asdict(cand), "choices": [], "history": [], "gguf": gg})
            c = self.ctx(c2, suffix=suffix, depends_on=[last] if last else list(dep), placeholder=not decided)
            change = t.apply(c, (t.pieces or ("",))[0])
            new = list(change.job_ids)
            if new:
                last = new[-1]
                ids += new
            gg = change.gguf
        ref, dep2 = self.reference_kl(cand, sig, [last] if last else list(dep), not decided)
        engine_bin = next((cfg.paths.get(t.engine) for t, _ in inside if getattr(t, "engine", "bin_planes") != "bin_planes"),
                          None)
        mj = measurement_job(f"M_COMBO_{suffix}", "COMBO", gg, f"{out}/measure", ref_kl=ref, ref_name=sig.label, mmlu=True,
                             cap_bytes=self.cap, engine_bin=engine_bin, depends_on=dep2, placeholder=not decided,
                             phase="COMBO", technique="combination")
        mj.env["FUCINA_NAME"] = "COMBO"
        mj = self.executor.enqueue(mj)
        ids.append(mj.id)
        for t, _ in inside:
            sw = t.switch()
            if sw.startswith(("THOR_SPEGNI", "GGML_")) and "=" in sw:
                kk, vv = sw.split(" ")[0].split("=", 1)
                mo = measurement_job(f"M_COMBO_{suffix}_without_{t.name}", f"COMBO-{t.name}", gg,
                                     f"{out}/measure_without_{t.name}", ref_kl=ref, ref_name=sig.label, mmlu=False,
                                     cap_bytes=self.cap, engine_bin=engine_bin, depends_on=[mj.id],
                                     placeholder=not decided, phase="COMBO", technique="combination")
                mo.env.update({kk: vv, "FUCINA_NAME": f"COMBO-{t.name}"})
                mo.name = f"combination without {t.name} ({kk}={vv})"
                self.executor.enqueue(mo)
                ids.append(mo.id)
        v.jobs, v.gguf = ids, gg
        v.numbers = {"levers": [t.name for t, _ in inside], "excluded_by_engine": excluded, "dense_base": dense_base,
                     "structure_plan": plan}
        if not decided:
            pending = [t.name for t, lv in levers if lv.state not in ENABLED]
            v.reason = (f"placeholder: {', '.join(t.name for t, _ in levers)} on top of {sig.label}; "
                        f"confirmed when the outcomes of {', '.join(pending) or 'the structure'} arrive")
            return v
        m = self.read_measurement_job(mj.id, "combination", sig, ref)
        if m is None:
            v.state, v.reason = "testing", f"queued: {', '.join(t.name for t, _ in inside)} on top of {sig.label}"
            return v
        ver = D.judge(m, Thresholds(), self.objective, "kl", "", self.law)
        v.verdict, v.numbers = ver.verdict, {**v.numbers, **m.as_dict()}
        v.state = "passed" if ver.verdict == "PASS" else "off" if ver.verdict == "FAIL" else "to_decide"
        v.reason = ver.reason
        self._ablations(v, inside, m, out, ref, sig)
        return v

    def _ablations(self, v: Evaluation, inside: list, m: Measurement, out: str, ref: str, sig: Signature) -> None:
        """Outcomes of the all-but-one measurements: a lever whose removal does not worsen the combination's KL
        (contribution < ABLATION_THRESHOLD_PCT) is useless or harmful and is switched off (only the worst one: the
        other suspects must be remeasured afterwards, because removals do not add up)."""
        if not m.kl:
            return
        contributions = {}
        for t, _ in inside:
            j = next((x for x in self.executor.jobs if x.id.endswith(f"_without_{t.name}") and x.id.startswith("M_COMBO_")
                      and x.output_dir == f"{out}/measure_without_{t.name}"), None)
            if j is None or j.state != "done":
                continue
            ms = read_combo_dir(j.output_dir, f"COMBO-{t.name}", ref, sig.label)
            if ms.kl:
                contributions[t.name] = round((ms.kl / m.kl - 1) * 100, 2)   # how much the KL worsens without it
        if not contributions:
            return
        v.numbers["kl_contribution_pct"] = contributions
        useless = sorted((c, n) for n, c in contributions.items() if c < ABLATION_THRESHOLD_PCT)
        if not useless:
            return
        c, n = useless[0]
        t = next(t for t, _ in inside if t.name == n)
        sw = t.switch().split(" ")[0]
        kk, vv = sw.split("=", 1)
        v.numbers["env_disable"] = {kk: vv}
        v.reason += (f"; all-but-one: without {n} the KL changes by {c:+.2f} % (< {ABLATION_THRESHOLD_PCT} %) -> {n} "
                     f"switched off ({kk}={vv})"
                     + (f"; also suspect: {', '.join(x for _, x in useless[1:])} (remeasure)" if useless[1:] else ""))
        self.event(f"COMBO: {n} brings nothing to the combination ({c:+.2f} % KL without it) -> switched off with {kk}={vv}")

    def _structure_plan(self, cand: Candidate) -> str:
        """piano.json of the consolidated second plane (next to its GGUF), or "" if the structure has none."""
        for s in reversed(cand.history):
            if s["phase"] == "F2":
                return str(Path(s["gguf"]).parent / "piano.json")
        return ""

    def _final_evaluation(self, cand: Candidate, env: dict, waiting: list) -> Evaluation:
        sig = self.signature(cand)
        v = Evaluation("final_eval", "F6", [WHOLE_MODEL], "waiting" if waiting else "testing", base=sig.label)
        arch = self.model.get("architecture")
        if arch and arch not in self.measure_architectures:
            v.state, v.reason = "excluded", (f"the measurement script is validated only for "
                                             f"{', '.join(self.measure_architectures)}")
            return v
        if len(cand.history) <= 1:
            v.state, v.reason = "excluded", "no technique applied: nothing to evaluate"
            return v
        id_ = f"F6_eval_{_h(cand.gguf + json.dumps(env, sort_keys=True))}"
        mj = measurement_job(id_, f"FINAL-{sig.label}", cand.gguf, f"{self.outputs}/final_eval_{_h(cand.gguf)}",
                             mmlu=True, cap_bytes=self.cap, depends_on=list(waiting), placeholder=bool(waiting),
                             phase="F6", technique="final_eval")
        mj.env.update(env)
        mj.env["FUCINA_NAME"] = f"FINAL-{sig.label}"
        mj.name = f"final evaluation {sig.label}: probe + KL + paired MMLU + tg" + (f" ({env})" if env else "")
        self.executor.enqueue(mj)
        v.jobs = [id_]
        s = self.signatures.get(cand.gguf)
        if s and s.mmlu and not waiting:
            mm = max(s.mmlu)
            ok = self.objective.mmlu_min is None or mm >= self.objective.mmlu_min
            v.numbers = {"mmlu": mm, "file_bytes": s.file_bytes, "tg": s.tg}
            v.reason = f"{sig.label}: MMLU {mm:.2f}" + (
                f" {'>=' if ok else '<'} {self.objective.mmlu_min}" if self.objective.mmlu_min else "")
        else:
            v.reason = "final evaluation waiting for the product" if waiting else "final evaluation queued"
        return v

    # ── forecast through the KL->MMLU law ──
    def forecast(self, cand: Candidate) -> dict:
        certain, expected, items = [], [], []
        structure = None
        for s in cand.history:
            sig = self.signatures.get(s["gguf"])
            if sig and sig.kl_pct is not None:
                structure = (s["label"], sig.kl_pct)
        if structure:
            certain.append(structure[1])
            items.append(f"{structure[0]} {structure[1]:+.1f} %")
        groups_seen = set()
        for v in self.evaluations:
            t = self.techniques.get(v.technique)
            if t is None or v.phase not in ("F1", "F2", "F3", "F4") or getattr(t, "judged_by", "") == "structure" \
                    or getattr(t, "structural", False):
                continue
            g = getattr(t, "group", "")
            if v.state == "passed" and v.numbers.get("kl_pct") is not None:
                certain.append(v.numbers["kl_pct"])
                items.append(f"{t.name} {v.numbers['kl_pct']:+.1f} % (measured)")
            elif v.state in ("testing", "waiting"):
                if g and g in groups_seen:
                    continue
                if g:
                    groups_seen.add(g)
                gm = v.numbers.get("nearby_evidence", {}).get("kl_pct") if v.phase == "F4" else None
                exp = gm if gm is not None and gm < 0 else t.expected_kl_gain_pct
                expected.append(exp)
                items.append(f"{t.name} {exp:+.1f} % (expected)")
        sure = D.compose_drops(*certain) if certain else 0.0
        total = D.compose_drops(*(certain + expected)) if (certain or expected) else 0.0
        p = {"kl_pct_measured": round(sure, 1), "mmlu_measured": round(self.law.mmlu(sure), 2),
             "kl_pct_expected": round(total, 1), "mmlu_expected": round(self.law.mmlu(total), 2), "items": items,
             "law": f"MMLU ≈ {self.law.base} + {self.law.coeff}·(KL drop %)²"}
        if not self.model.get("kl_mmlu_law"):
            p.update(mmlu_measured=None, mmlu_expected=None, law="no measured KL->MMLU law for this model",
                     comment=f"expected KL {total:+.1f} % (no KL->MMLU law for this model)")
            return p
        if self.objective.mmlu_min:
            needed = self.law.drop_for(self.objective.mmlu_min)
            p["kl_pct_needed"] = round(needed, 1)
            p["reaches"] = p["mmlu_expected"] >= self.objective.mmlu_min
            p["comment"] = (f"forecast {p['mmlu_expected']:.2f} {'>=' if p['reaches'] else '<'} {self.objective.mmlu_min} "
                            f"(needs a KL drop of ~{needed:.0f} %, expected {total:.1f} %)")
        return p

    # ── outputs ──
    def report(self) -> dict:
        queue = [asdict(j) for j in self.executor.jobs]
        for j in queue:
            j.pop("command", None)
        return {"model": self.model["name"], "model_file": self.model["_file"], "objective_arg": self.objective_arg,
                "db": self.measurements.path, "objective": self.objective.as_dict(), "cap_bytes": self.cap,
                "mode": self.mode, "at": time.strftime("%Y-%m-%dT%H:%M"),
                "candidate": {"label": self.cand.label, "gguf": self.cand.gguf, "phase": self.cand.current_phase},
                "history": self.cand.history, "evaluations": [asdict(v) for v in self.evaluations],
                "events": self.events, "forecast": self.cand.forecast, "open_questions": self.questions.open(),
                "engine_env": self.env, "queue": queue}

    def recipe(self) -> dict:
        r = {"model": self.model["name"], "model_file": self.model["_file"], "objective": self.objective_arg,
             "candidate": self.cand.label, "gguf": self.cand.gguf, "history": self.cand.history,
             "engine_env": self.env, "forecast": self.cand.forecast, "phases": {}}
        for v in self.evaluations:
            r["phases"].setdefault(v.phase, []).append({"technique": v.technique, "state": v.state, "switch": v.switch,
                                                        "reason": v.reason})
        r["switches"] = {v.technique: ("enable" if v.state in ENABLED else "disable") for v in self.evaluations
                         if v.technique in self.techniques and v.state in ENABLED + ("off",)}
        return r

    def launch(self) -> list:
        """dry_run/run modes: launch the ready jobs (one at a time on the GPU) and log one activity line."""
        if self.mode == "simulate":
            return []
        launched = self.executor.advance()
        while self.mode == "dry_run":         # a dry run writes the whole queue in order (nothing really starts)
            more = self.executor.advance()
            if not more:
                break
            launched += more
        if launched and self.mode == "run":
            log_cycle(f"FUCINA {self.dir.name}: launched {', '.join(j.id for j in launched)}")
        return launched


def _choice_state(s: str) -> str:
    return {"passed": "passed", "consolidated": "consolidated", "off": "off", "to_decide": "to_decide",
            "superseded": "off", "testing": "testing", "waiting": "testing"}.get(s, s)


def forge(recipe: str, run_dir: str, mode: str = "dry_run", db: Optional[str] = None) -> Engine:
    """Replay a recipe: its choices become user switches, then one engine pass."""
    r = yaml.safe_load(Path(recipe).read_text())
    e = Engine(r["model_file"], r["objective"], run_dir, mode, db)
    for t, val in (r.get("switches") or {}).items():
        e.switches.set(t, val)
    e.run()
    e.launch()
    return e

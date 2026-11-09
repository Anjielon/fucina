"""Declarative replay recipes: inputs, steps, commands, checks.

A replay recipe (YAML) describes a recorded chain of quantization steps so that it can be re-run on another machine
from public inputs. It has four parts:

``roots``
    Named directories that the user provides (``work``, ``inputs``, ``bundle``, ``engine``, ``python``, ...). They are
    the only machine-specific values: every path in the recipe is written as ``{root}/relative/path``.
``inputs``
    Every file or directory the chain reads that no step of the recipe produces: Hugging Face files pinned to a
    revision (``kind: hf``), files shipped inside this package (``kind: package``) or artifacts produced by a
    preparation step of the same recipe (``kind: derived``). Each input carries the hash of the recorded run.
``steps``
    The chain, in execution order. A step lists what it ``needs`` (input or output keys), what it ``produces``
    (paths and how they are verified), the bash commands it runs, the code it uses and an honest ``status``:
    ``ported`` (code in this package), ``bundle`` (frozen research code of the recorded run, see
    :mod:`fucina.replay.bundle`) or ``blocked`` (cannot run outside the recording machine yet; ``blockers`` says why).
``target``
    The recorded result the chain must reach, with the check that decides it.

Placeholders: ``{name}`` for a root, ``{path.KEY}`` for the path of an input or output, ``{gpu_env}`` for the
per-step GPU environment. ``${VAR}`` (bash) is left alone. Literal absolute paths are rejected by :func:`validate`,
except a few system locations (``/usr/bin``, ``/bin``, ``/dev/null``, ``/proc``, ``/sys``).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

PLACEHOLDER = re.compile(r"(?<![$\\]){([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_\-]+)?)}")
ABS_PATH = re.compile(r"(?<![\w.$})\]])(/(?:[\w.+\-]+/)*[\w.+\-]+)")
SYSTEM_PREFIXES = ("/usr/", "/bin/", "/dev/null", "/proc/", "/sys/", "/dev/stdin", "/dev/stdout", "/dev/stderr")
STATUSES = ("ported", "bundle", "blocked")
KINDS = ("fetch", "cpu", "gpu", "measure", "check")
VERIFY = ("sha256", "manifest", "tensors", "mmlu", "exists", "none")


class RecipeError(ValueError):
    pass


@dataclass
class Input:
    key: str
    kind: str                         # hf | package | derived
    path: str                         # template, e.g. {inputs}/bf16
    repo: str = ""
    repo_type: str = "model"          # model | dataset
    revision: str = ""
    files: list = field(default_factory=list)
    sha256: str = ""                  # recorded hash of a single file
    manifest: str = ""                # recorded per-file hashes of a directory (file relative to the recipe)
    size_gib: float = 0.0
    produced_by: str = ""
    published: str = ""               # where the recorded copy is (or will be) published
    note: str = ""


@dataclass
class Output:
    key: str
    path: str
    verify: str = "exists"
    expect: str = ""                  # sha256, tensor digest, manifest file, or "73.15" for mmlu
    tolerance: str = ""


@dataclass
class Step:
    id: str
    title: str
    kind: str
    run: list
    needs: list = field(default_factory=list)
    produces: list = field(default_factory=list)
    code: list = field(default_factory=list)
    env: dict = field(default_factory=dict)
    status: str = "bundle"
    blockers: list = field(default_factory=list)
    determinism: str = "bitwise"      # bitwise | tolerance | stochastic | measurement
    gpu_hours: float = 0.0
    cpu_hours: float = 0.0
    vram_gib: float = 0.0
    disk_gib: float = 0.0
    optional: bool = False
    technique: str = ""
    note: str = ""


@dataclass
class Recipe:
    name: str
    title: str
    roots: dict
    inputs: dict
    steps: list
    target: dict
    source: Optional[Path] = None
    description: str = ""
    deviations: list = field(default_factory=list)

    def step(self, sid: str) -> Step:
        for s in self.steps:
            if s.id == sid:
                return s
        raise RecipeError(f"no step {sid!r}")

    def artifacts(self) -> dict:
        """key -> path template, for inputs and step outputs."""
        out = {k: i.path for k, i in self.inputs.items()}
        for s in self.steps:
            for o in s.produces:
                out[o.key] = o.path
        return out

    def producer(self) -> dict:
        """artifact key -> id of the step that produces it ('' for external inputs)."""
        out = {k: (i.produced_by if i.kind == "derived" else "") for k, i in self.inputs.items()}
        for s in self.steps:
            for o in s.produces:
                out[o.key] = s.id
        return out


def _list(x) -> list:
    if x is None:
        return []
    return list(x) if isinstance(x, (list, tuple)) else [x]


def load(path: str | Path) -> Recipe:
    path = Path(path)
    d = yaml.safe_load(path.read_text())
    if not isinstance(d, dict):
        raise RecipeError(f"{path}: not a mapping")
    inputs = {}
    for k, v in (d.get("inputs") or {}).items():
        v = dict(v or {})
        v["files"] = _list(v.get("files"))
        inputs[k] = Input(key=k, **v)
    steps = []
    for raw in d.get("steps") or []:
        raw = dict(raw)
        raw["id"] = str(raw["id"])
        raw["run"] = _list(raw.get("run"))
        raw["needs"] = _list(raw.get("needs"))
        raw["code"] = _list(raw.get("code"))
        raw["blockers"] = _list(raw.get("blockers"))
        raw["produces"] = [Output(**o) for o in _list(raw.get("produces"))]
        steps.append(Step(**raw))
    r = Recipe(name=d["name"], title=d.get("title", d["name"]), roots=d.get("roots") or {}, inputs=inputs,
               steps=steps, target=d.get("target") or {}, source=path, description=d.get("description", ""),
               deviations=_list(d.get("deviations")))
    validate(r)
    return r


def _templates(r: Recipe):
    for k, i in r.inputs.items():
        yield f"input {k}", i.path
    for s in r.steps:
        for c in s.run:
            yield f"step {s.id}", c
        for o in s.produces:
            yield f"step {s.id} output {o.key}", o.path
        for k, v in s.env.items():
            yield f"step {s.id} env {k}", str(v)
        for c in s.code:
            yield f"step {s.id} code", c


def literal_paths(text: str) -> list[str]:
    """Absolute paths written literally (not produced by a placeholder) that are not system locations."""
    bare = PLACEHOLDER.sub("X", text)
    bare = re.sub(r"\$\{[^}]*\}|\$\w+", "X", bare)
    return [p for p in ABS_PATH.findall(bare)
            if not any(p == s.rstrip("/") or p.startswith(s) for s in SYSTEM_PREFIXES)]


def validate(r: Recipe) -> None:
    errors = []
    ids = [s.id for s in r.steps]
    if len(ids) != len(set(ids)):
        errors.append("duplicate step ids")
    known = set(r.roots) | {"gpu_env", "vk_dev", "threads", "recipe_dir", "package"}
    arts = r.artifacts()
    for where, text in _templates(r):
        for name in PLACEHOLDER.findall(text):
            if name.startswith("path."):
                if name[5:] not in arts:
                    errors.append(f"{where}: unknown artifact {name}")
            elif name not in known:
                errors.append(f"{where}: unknown placeholder {{{name}}}")
        for p in literal_paths(text):
            errors.append(f"{where}: literal absolute path {p} (use a root placeholder)")
    seen = set(k for k, i in r.inputs.items() if i.kind != "derived")
    order = {s.id: n for n, s in enumerate(r.steps)}
    for k, i in r.inputs.items():
        if i.kind not in ("hf", "package", "derived"):
            errors.append(f"input {k}: unknown kind {i.kind}")
        if i.kind == "hf" and not (i.repo and i.revision):
            errors.append(f"input {k}: a Hugging Face input needs repo and a pinned revision")
        if i.kind == "derived" and i.produced_by not in order:
            errors.append(f"input {k}: produced_by {i.produced_by!r} is not a step")
    for s in r.steps:
        if s.kind not in KINDS:
            errors.append(f"step {s.id}: unknown kind {s.kind}")
        if s.status not in STATUSES:
            errors.append(f"step {s.id}: unknown status {s.status}")
        if s.status == "blocked" and not s.blockers:
            errors.append(f"step {s.id}: blocked without blockers")
        for n in s.needs:
            if n in seen:
                continue
            i = r.inputs.get(n)
            if i is not None and i.kind == "derived" and order[i.produced_by] < order[s.id]:
                continue
            errors.append(f"step {s.id}: needs {n!r}, which no earlier step or external input provides")
        for o in s.produces:
            if o.verify not in VERIFY:
                errors.append(f"step {s.id} output {o.key}: unknown verify {o.verify}")
            seen.add(o.key)
        for k, i in r.inputs.items():
            if i.kind == "derived" and i.produced_by == s.id:
                seen.add(k)
    if errors:
        raise RecipeError("invalid recipe:\n  " + "\n  ".join(errors))


def render(text: str, values: dict[str, str]) -> str:
    def sub(m):
        k = m.group(1)
        if k not in values:
            raise RecipeError(f"unresolved placeholder {{{k}}}")
        return str(values[k])
    return PLACEHOLDER.sub(sub, text)


def values_for(r: Recipe, roots: dict[str, str], extra: Optional[dict] = None) -> dict[str, str]:
    """Placeholder values: roots first, then artifact paths (which may use roots)."""
    missing = [k for k in r.roots if k not in roots]
    if missing:
        raise RecipeError("roots not configured: " + ", ".join(missing) +
                          " (use --set NAME=PATH, the `replay:` section of the fucina config, or FUCINA_REPLAY_<NAME>)")
    v: dict[str, Any] = {k: str(p) for k, p in roots.items()}
    v.setdefault("recipe_dir", str(r.source.parent) if r.source else ".")
    v.setdefault("package", str(Path(__file__).resolve().parents[1]))
    v.setdefault("gpu_env", "")
    v.setdefault("vk_dev", "Vulkan0")
    v.setdefault("threads", "8")
    v.update(extra or {})
    for k, t in r.artifacts().items():
        v[f"path.{k}"] = render(t, v)
    return v

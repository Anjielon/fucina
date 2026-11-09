"""Plan, check and run a replay recipe.

``fucina replay RECIPE --dry-run`` resolves every command against the configured roots and checks, without running
anything, that:

* every placeholder resolves and every absolute path in a command lies under a configured root (or a system
  location): the replay reads nothing else on the machine;
* no command contains a forbidden prefix (``--forbid``, e.g. the author's home directory): the proof that a run
  on the author's machine would not silently read private files;
* every input is either external with a pinned locator and a recorded hash, or produced by an earlier step;
* every code reference exists (``package:`` modules of this package, ``bundle:`` files of the code bundle);
* it reports, per step, whether it can run (``ported``/``bundle``) or is still ``blocked``, with the estimated GPU
  and CPU hours.

``--run`` executes the selected steps in order, verifies each output and records the result in
``{work}/state/<step>.json``; a verified step is skipped when the run is resumed.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from . import gguf_hash
from .recipe import ABS_PATH, SYSTEM_PREFIXES, Recipe, RecipeError, Step, render, values_for

FINAL = re.compile(r"Final result:\s*([0-9.]+)")


# ── roots ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def configured_roots(r: Recipe, sets: list[str], config_section: Optional[dict] = None) -> dict[str, str]:
    """Root values from (lowest to highest priority): recipe defaults, the fucina config ``replay:`` section,
    ``FUCINA_REPLAY_<NAME>`` environment variables, ``--set NAME=PATH``."""
    out = {}
    for k, spec in r.roots.items():
        if isinstance(spec, dict) and spec.get("default") is not None:
            out[k] = str(spec["default"])
    for k, v in (config_section or {}).items():
        out[k] = str(v)
    for k in r.roots:
        e = os.environ.get(f"FUCINA_REPLAY_{k.upper()}")
        if e:
            out[k] = e
    for s in sets:
        if "=" not in s:
            raise RecipeError(f"--set {s!r}: expected NAME=PATH")
        k, v = s.split("=", 1)
        out[k] = v
    seed = {"recipe_dir": str(r.source.parent) if r.source else ".",
            "package": str(Path(__file__).resolve().parents[1])}
    for _ in range(len(out) + 1):            # defaults may refer to other roots: "{work}/layout"
        changed = False
        for k, v in list(out.items()):
            new = re.sub(r"\{(\w+)\}", lambda m: out.get(m.group(1), seed.get(m.group(1), m.group(0))), v)
            if new != v:
                out[k], changed = new, True
        if not changed:
            break
    for k, v in list(out.items()):
        if "{" in v:
            raise RecipeError(f"root {k}: cannot resolve {v}")
        out[k] = os.path.normpath(os.path.abspath(os.path.expanduser(v))) if ("/" in v or v.startswith("~")) else v
    return out


# ── selection ───────────────────────────────────────────────────────────────────────────────────────────────────────
def select(r: Recipe, steps: Optional[str], first: Optional[str], last: Optional[str],
           with_optional: bool) -> list[Step]:
    ids = [s.id for s in r.steps]
    if steps:
        want = [x.strip() for x in steps.split(",") if x.strip()]
        for w in want:
            r.step(w)
        return [s for s in r.steps if s.id in want]
    a = ids.index(first) if first else 0
    b = ids.index(last) if last else len(ids) - 1
    return [s for s in r.steps[a:b + 1] if with_optional or not s.optional]


# ── checks ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def path_violations(text: str, allowed: list[str], forbid: list[str], exempt: tuple = ()) -> list[str]:
    bad = []
    for e in exempt:                          # this package itself may live anywhere
        text = re.sub(re.escape(e) + r"[\w./+\-]*", "PACKAGE", text)
    for f in forbid:
        if f and f in text:
            bad.append(f"forbidden prefix {f}")
    for p in ABS_PATH.findall(text):
        if any(p == s.rstrip("/") or p.startswith(s) for s in SYSTEM_PREFIXES):
            continue
        if not any(p == a or p.startswith(a.rstrip("/") + "/") for a in allowed):
            bad.append(f"path outside the configured roots: {p}")
    return bad


def code_status(ref: str, values: dict) -> tuple[bool, str]:
    kind, _, target = ref.partition(":")
    if kind == "package":
        spec = importlib.util.find_spec(target)
        return spec is not None, target
    if kind == "bundle":                     # installed bundle, or the published one before step L0 installs it
        for root in (values["bundle"], values.get("bundle_src", "")):
            if root and (Path(root) / target).exists():
                return True, str(Path(root) / target)
        return False, str(Path(values["bundle"]) / target)
    if kind == "engine":
        p = Path(values["engine"]) / target
        return p.exists(), str(p)
    return False, f"unknown code reference {ref}"


def step_commands(s: Step, values: dict) -> tuple[str, dict]:
    env = {k: render(str(v), values) for k, v in s.env.items()}
    body = "\n".join(render(c, values) for c in s.run)
    return body, env


def plan(r: Recipe, roots: dict, sel: list[Step], forbid: list[str], show: bool = True,
         script: Optional[Path] = None) -> dict:
    values = values_for(r, roots)
    allowed = [v for k, v in roots.items() if isinstance(v, str) and v.startswith("/")]
    allowed += [str(Path(values["package"]).parent), values["recipe_dir"]]
    problems: list[str] = []
    lines = [f"# replay {r.name}: {r.title}", f"# recipe {r.source}"]
    lines += [f"#   root {k} = {roots[k]}" for k in r.roots]
    sh = ["#!/usr/bin/env bash", f"# generated by `fucina replay --dry-run` from {r.source}", "set -euo pipefail"]

    if r.forge.get("release"):
        from .status import forge_commit
        commit, how = forge_commit(r.forge["release"], strict=False)
        lines.append(f"# forge release {r.forge['release']}: " + (f"commit {commit} ({how})" if commit else f"NOT RESOLVED: {how}"))
        if not commit:
            problems.append(f"forge version: {how} (step V could not write REPLAY_STATUS)")
    lines.append("\n## external inputs")
    for k, i in r.inputs.items():
        p = values[f"path.{k}"]
        if i.kind == "derived":
            lines.append(f"  {k:<22} derived by step {i.produced_by:<5} -> {p}")
            continue
        loc = f"hf {i.repo_type}:{i.repo}@{i.revision}" if i.kind == "hf" else "package data"
        pin = (f"sha256 {i.sha256[:16]}" if i.sha256 else f"manifest {i.manifest}" if i.manifest else "NO HASH")
        if not (i.sha256 or i.manifest):
            problems.append(f"input {k}: no recorded hash")
        have = "present" if Path(p).exists() else "to fetch" if i.kind == "hf" else "MISSING"
        if i.kind == "package" and not Path(p).exists():
            problems.append(f"input {k}: package file missing: {p}")
        elif i.kind == "package" and i.sha256 and Path(p).is_file() and sha256(Path(p)) != i.sha256:
            problems.append(f"input {k}: {p} differs from its pinned sha256 {i.sha256[:16]}")
        lines.append(f"  {k:<22} {loc:<60} {pin:<24} {have}")
        problems += [f"input {k}: {v}" for v in path_violations(p, allowed, forbid, (str(Path(values["package"]).parent), values["recipe_dir"]))]

    tot = {"ported": [0.0, 0.0], "bundle": [0.0, 0.0], "blocked": [0.0, 0.0]}
    lines.append("\n## steps")
    for s in sel:
        body, env = step_commands(s, values)
        tot[s.status][0] += s.gpu_hours
        tot[s.status][1] += s.cpu_hours
        head = (f"\n### {s.id} [{s.kind}, {s.status}, {s.determinism}"
                f"{f', GPU {s.gpu_hours:g} h' if s.gpu_hours else ''}{f', CPU {s.cpu_hours:g} h' if s.cpu_hours else ''}"
                f"{f', VRAM {s.vram_gib:g} GiB' if s.vram_gib else ''}] {s.title}")
        lines.append(head)
        if s.technique:
            lines.append(f"    technique: {s.technique}")
        for n in s.needs:
            lines.append(f"    needs    {n} = {values[f'path.{n}']}")
        for o in s.produces:
            lines.append(f"    produces {o.key} = {values[f'path.{o.key}']}  [verify {o.verify}"
                         f"{' ' + o.expect[:16] if o.expect else ''}{'; ' + o.tolerance if o.tolerance else ''}]")
        for c in s.code:
            ok, where = code_status(c, values)
            lines.append(f"    code     {c}  {'ok' if ok else 'NOT FOUND: ' + where}")
            if not ok and not c.startswith("engine:"):
                problems.append(f"step {s.id}: code {c} not found ({where})")
        for b in s.blockers:
            lines.append(f"    BLOCKER  {b}")
        text = body + "\n" + "\n".join(f"{k}={v}" for k, v in env.items())
        for v in path_violations(text, allowed, forbid, (str(Path(values["package"]).parent), values["recipe_dir"])):
            problems.append(f"step {s.id}: {v}")
        if show:
            for k, v in env.items():
                lines.append(f"    env      {k}={v}")
            for c in body.splitlines():
                lines.append(f"    $ {c}")
        sh.append(f"\n# ── step {s.id} [{s.status}] {s.title}")
        if s.status == "blocked":
            sh.append(f"# BLOCKED: {'; '.join(s.blockers)}")
        sh.append("(" + " ".join(f"export {k}={shlex.quote(v)};" for k, v in env.items()))
        sh.append(body + "\n)")

    lines.append("\n## summary")
    for st, (g, c) in tot.items():
        n = sum(1 for s in sel if s.status == st)
        lines.append(f"  {st:<8} {n:>3} steps   GPU {g:6.1f} h   CPU {c:6.1f} h")
    blocked = [s.id for s in sel if s.status == "blocked"]
    lines.append(f"  blocked steps: {', '.join(blocked) or 'none'}")
    lines.append(f"  problems: {len(problems)}")
    lines += [f"    - {p}" for p in problems]
    if show:
        print("\n".join(lines))
    if script:
        script.write_text("\n".join(sh) + "\n")
        script.chmod(0o755)
    return {"problems": problems, "blocked": blocked, "totals": tot}


# ── verification ────────────────────────────────────────────────────────────────────────────────────────────────────
def sha256(path: Path) -> str:
    return gguf_hash.file_sha256(path)


def verify_output(o, values: dict, recipe_dir: Path) -> tuple[bool, str]:
    p = Path(values[f"path.{o.key}"])
    if o.verify == "none":
        return True, "not verified (declared)"
    if not p.exists():
        return False, f"missing {p}"
    if o.verify == "exists":
        return True, "present"
    if o.verify == "sha256":
        got = sha256(p)
        return (not o.expect or got == o.expect), f"sha256 {got}"
    if o.verify == "tensors":
        got = gguf_hash.tensor_digest(gguf_hash.tensor_hashes(p))
        return (not o.expect or got == o.expect), f"tensor digest {got}"
    if o.verify == "manifest":
        man = recipe_dir / o.expect
        bad = []
        for line in man.read_text().splitlines():
            if not line.strip():
                continue
            h, rel = line.split(None, 1)
            f = p / rel.strip()
            if not f.exists() or sha256(f) != h:
                bad.append(rel.strip())
        return not bad, f"{len(bad)} files differ from {man.name}" + (f": {bad[:5]}" if bad else "")
    if o.verify == "mmlu":
        m = FINAL.findall(p.read_text(errors="replace"))
        if not m:
            return False, "no 'Final result' in the log"
        got = float(m[-1])
        band = float(o.tolerance.split("=")[-1]) if o.tolerance.startswith("band=") else 1.0
        return (not o.expect or abs(got - float(o.expect)) <= band), f"MMLU {got:.2f} (recorded {o.expect})"
    return False, f"unknown verify {o.verify}"


# ── run ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def run(r: Recipe, roots: dict, sel: list[Step], allow_blocked: bool = False) -> int:
    values = values_for(r, roots)
    work = Path(roots["work"])
    state = work / "state"
    logs = work / "logs"
    state.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    recipe_dir = r.source.parent if r.source else Path(".")
    for s in sel:
        done = state / f"{s.id}.json"
        if done.exists() and json.loads(done.read_text()).get("ok"):
            print(f"[{s.id}] verified earlier: skipped")
            continue
        if s.status == "blocked" and not allow_blocked:
            print(f"[{s.id}] BLOCKED: {'; '.join(s.blockers)}")
            return 3
        for n in s.needs:
            if not Path(values[f"path.{n}"]).exists():
                print(f"[{s.id}] input {n} missing: {values[f'path.{n}']}")
                return 2
        body, env = step_commands(s, values)
        for o in s.produces:
            Path(values[f"path.{o.key}"]).parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        log = logs / f"{s.id}.log"
        with open(log, "a") as f:
            f.write(f"\n# {time.strftime('%F %T')} step {s.id}\n{body}\n")
            f.flush()
            rc = subprocess.run(["bash", "-euo", "pipefail", "-c", body], env={**os.environ, **env},
                                stdout=f, stderr=subprocess.STDOUT).returncode
        res = {o.key: verify_output(o, values, recipe_dir) for o in s.produces} if rc == 0 else {}
        ok = rc == 0 and all(v[0] for v in res.values())
        rec = {"step": s.id, "ok": ok, "rc": rc, "minutes": round((time.time() - t0) / 60, 2),
               "checks": {k: {"ok": v[0], "detail": v[1]} for k, v in res.items()},
               "when": time.strftime("%FT%T")}
        (state / (f"{s.id}.json" if ok else f"{s.id}.failed.json")).write_text(json.dumps(rec, indent=1))
        print(f"[{s.id}] {'OK' if ok else 'FAILED'} in {rec['minutes']} min  {json.dumps(rec['checks'])}")
        if not ok and s.determinism == "bitwise":
            return 1
        if not ok and s.determinism != "bitwise":
            print(f"[{s.id}] non-bitwise step outside its check: continuing, the final target decides")
    return 0



def bench_list_problems(r: Recipe, tsv: Path, recipe_dir: Path) -> list[str]:
    """Every input of the recipe that lives in the bench dataset must have a row in its file list with the same path
    and the same sha256 (directories: one row per file of the recorded manifest)."""
    repo = (r.bench_list or {}).get("repo")
    if not repo:
        return []
    rows = {}
    for line in Path(tsv).read_text().splitlines()[1:]:
        f = line.split("\t")
        if len(f) >= 3:
            rows[f[0]] = f[2]
    want: dict[str, str] = {}
    for k, i in r.inputs.items():
        if i.kind != "hf" or i.repo != repo:
            continue
        if i.hashes:
            want.update({p: (h, k) for p, h in i.hashes.items()})
        elif i.manifest and len(i.files) == 1 and i.files[0].endswith("/"):
            for line in (recipe_dir / i.manifest).read_text().splitlines():
                if line.strip():
                    h, rel = line.split(None, 1)
                    want[i.files[0] + rel.strip()] = (h, k)
        elif len(i.files) == 1 and i.sha256:
            want[i.files[0]] = (i.sha256, k)
        else:
            want[f"<{k}>"] = ("", k)
    bad = []
    for path, (h, k) in sorted(want.items()):
        if path.startswith("<"):
            bad.append(f"bench list: input {k} names no single file with a hash")
        elif path not in rows:
            bad.append(f"bench list: input {k}: no row for {path}")
        elif rows[path] != h:
            bad.append(f"bench list: input {k}: {path} has sha256 {rows[path][:16]} in the list, {h[:16]} in the recipe")
    return bad

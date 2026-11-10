"""Command-line interface (SPEC section 7)::

  fucina [--config FILE] [--db FILE] COMMAND ...

  fucina auto      [--model M] [--objective A|"mmlu>=80.05,ram<=9GiB,tg>=50"] [--simulate|--dry-run|--run] [--run-dir DIR]
  fucina manual    [--model M] --piece experts_gate_up [--series a,b] [--parallel c,d] [--base LABEL]
  fucina status    [--run-dir DIR] [--refresh]
  fucina decide    [--run-dir DIR] [ID enable|disable|defer]
  fucina enable|disable|release TECHNIQUE [--run-dir DIR]
  fucina rules     [--phase F2] [--piece P] [--model-type T]
  fucina measurements [--technique T]
  fucina import
  fucina techniques
  fucina forge     --recipe runs/X/recipe.yaml [--simulate|--dry-run|--run]
  fucina tui       [--run-dir DIR]

No GPU and no process with ``--simulate`` (the default: measurements come from the database); ``--dry-run`` writes the
commands to ``dry_run_commands.sh``; ``--run`` launches the jobs with systemd-run (GPU jobs take the machine-wide lock
themselves)."""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Optional

from . import config
from .db import Measurements, load_rules, rules_for


def _runs_dir() -> Path:
    return Path(config.current()["runs_dir"])


def _run_dir(a: argparse.Namespace) -> Path:
    if getattr(a, "run_dir", None):
        return Path(a.run_dir)
    runs = sorted((p for p in _runs_dir().glob("*") if (p / "state.json").exists()), key=lambda p: p.stat().st_mtime)
    if not runs:
        sys.exit("no run yet: start one with `fucina auto`")
    return runs[-1]


def _mode(a: argparse.Namespace) -> str:
    return "run" if a.run else "dry_run" if a.dry_run else "simulate"


def _model(a: argparse.Namespace) -> str:
    m = a.model or config.current().get("default_model")
    if not m:
        sys.exit("no model: pass --model NAME (or set default_model in the fucina config)")
    return m


def _objective(a: argparse.Namespace) -> str:
    return a.objective or config.current().get("default_objective") or "A"


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "_", s)[:40].strip("_")


def cmd_auto(a: argparse.Namespace) -> None:
    from .engine import Engine
    from .view import print_report
    model, objective = _model(a), _objective(a)
    run = Path(a.run_dir) if a.run_dir else _runs_dir() / f"{model}-{_slug(objective)}-{_mode(a)}"
    e = Engine(model, objective, str(run), _mode(a), db=a.db, params=json.loads(a.params or "{}"))
    r = e.run()
    launched = e.launch()
    print_report(r, run)
    if launched:
        print(f"\nlaunched: {', '.join(j.id for j in launched)}")
    if _mode(a) == "dry_run" and (run / "dry_run_commands.sh").exists():
        print(f"commands (not launched): {run / 'dry_run_commands.sh'}")


def cmd_status(a: argparse.Namespace) -> None:
    from .view import print_report
    run = _run_dir(a)
    if a.refresh:
        from .executor import Executor
        Executor(str(run), "run").refresh()
    print_report(json.loads((run / "report.json").read_text()), run)


def cmd_decide(a: argparse.Namespace) -> None:
    from .engine import QuestionQueue
    run = _run_dir(a)
    qq = QuestionQueue(run)
    if a.id:
        q = qq.answer(a.id, a.answer, a.note or "")
        print(f"{q['id']}: {q['answer']} (run `fucina auto --run-dir {run}` to recompute)")
        return
    opened = qq.open()
    if not opened:
        print("no open questions")
        return
    interactive = sys.stdin.isatty() and not a.list
    for q in opened:
        print(f"\n[{q['id']}] phase {q['phase']} — {q['text']}\n  advice: {q.get('advice') or '-'}"
              f"   objective: {q.get('objective')}\n  numbers: {json.dumps(q.get('numbers', {}), ensure_ascii=False)[:300]}")
        if interactive:
            r = input("  enable / disable / defer / (enter = skip): ").strip().lower()
            if r in qq.ANSWERS:
                qq.answer(q["id"], r)
    if not interactive:
        print("\nanswer with: fucina decide ID enable|disable|defer")


def cmd_switch(a: argparse.Namespace) -> None:
    from .engine import Switches
    run = _run_dir(a)
    Switches(run).set(a.technique, None if a.action == "release" else a.action)
    print(f"{a.technique}: {a.action} (run {run.name}); run `fucina auto --run-dir {run}` to recompute")


def cmd_rules(a: argparse.Namespace) -> None:
    rr = load_rules()
    if a.phase:
        rr = [r for r in rr if r in rules_for(a.phase, a.piece or "", a.model_type, rr)]
    for r in rr:
        print(f"• {r['id']} (confidence {r.get('confidence')}): {r['statement']}")
        if r.get("effect"):
            print(f"    effect: {r['effect']}")
        if r.get("evidence"):
            print(f"    evidence: {'; '.join(map(str, r['evidence']))[:240]}")


def cmd_measurements(a: argparse.Namespace) -> None:
    m = Measurements(a.db)
    for r in m.all():
        if a.technique and a.technique.lower() not in (r.get("technique") or "").lower():
            continue
        par = json.loads(r["params"]) if r.get("params") else {}
        num = " ".join(f"{k}={r[k]}" for k in ("kl_pct", "mmlu", "mmlu_base", "p_mcnemar", "tg", "tg_base", "file_bytes")
                       if r.get(k) is not None)
        print(f"#{r['id']} {r.get('model')} {r.get('technique')} [{par.get('base', '')}] {num}  ({r.get('source')})")


def cmd_import(a: argparse.Namespace) -> None:
    from .importers.cycles import import_cycles
    from .importers.thor_notes import import_all
    m = Measurements(a.db)
    if Path(m.path).exists() and not os.access(m.path, os.W_OK):
        raise SystemExit(f"{m.path} is not writable by this user: run `fucina import` as its owner "
                         f"or use --db <other.sqlite>")
    try:
        n0 = import_cycles(m)
        n = import_all(m)
    except sqlite3.OperationalError as e:
        raise SystemExit(f"measurement database ({m.path}): {e}")
    print(f"measurement database ({m.path}): {n0} rows from the activity log, structured sources {n}")


def cmd_techniques(a: argparse.Namespace) -> None:
    from .registry import load_techniques
    if getattr(a, "census", False):
        from .census import census_markdown
        print(census_markdown(load_techniques().values(), short=a.short))
        return
    for t in sorted(load_techniques().values(), key=lambda t: (t.phase, t.name)):
        susp = f" [suspended: {t.suspended}]" if getattr(t, "suspended", "") else ""
        miss = t.missing() if hasattr(t, "missing") else []
        print(f"{t.phase} {t.name:18s} {getattr(t, 'judged_by', ''):11s} pieces={','.join(t.pieces) or '-'} "
              f"switch={t.switch()!r} cost={t.gpu_cost_h} h{susp}" + (f" MISSING {miss[0]}" if miss else ""))


def cmd_manual(a: argparse.Namespace) -> None:
    from .manual import manual
    from .view import print_manual
    model = _model(a)
    run = Path(a.run_dir) if a.run_dir else _runs_dir() / f"{model}-manual-{a.piece}-{_mode(a)}"
    print_manual(manual(model, _objective(a), str(run), a.piece, _list(a.series), _list(a.parallel),
                        _mode(a), a.db, a.base, _overrides(a.set)))


def _overrides(items: Optional[list]) -> dict:
    """``--set joint_cardan.steps=550 --set joint_cardan.stop_mmlu=73.8`` -> {technique: {param: value}}."""
    out: dict = {}
    for it in items or []:
        k, _, v = it.partition("=")
        t, _, prm = k.partition(".")
        if not (t and prm and _):
            raise SystemExit(f"--set expects technique.param=value, got {it!r}")
        try:
            val = json.loads(v)
        except ValueError:
            val = v
        out.setdefault(t, {})[prm] = val
    return out


def _list(s: Optional[str]) -> list:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def cmd_forge(a: argparse.Namespace) -> None:
    from .engine import forge
    from .view import print_report
    run = Path(a.run_dir) if a.run_dir else Path(a.recipe).parent.with_name(Path(a.recipe).parent.name + "-forge")
    e = forge(a.recipe, str(run), _mode(a), a.db)
    print_report(e.report(), run)


def cmd_tui(a: argparse.Namespace) -> None:
    from .tui import start
    start(_run_dir(a))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fucina", description="Fucina: measured, per-tensor-family ternary recipes")
    p.add_argument("--config", default=None, help="fucina config YAML (default: $FUCINA_CONFIG or ./fucina.yaml)")
    p.add_argument("--db", default=os.environ.get("FUCINA_DB"), help="measurement database (default: from the config)")
    s = p.add_subparsers(dest="cmd", required=True)

    def modes(q: argparse.ArgumentParser) -> None:
        g = q.add_mutually_exclusive_group()
        g.add_argument("--simulate", action="store_true", help="no process: measurements from the database (default)")
        g.add_argument("--dry-run", action="store_true", help="write the commands without launching them")
        g.add_argument("--run", action="store_true", help="launch the jobs (systemd-run, GPU lock)")

    q = s.add_parser("auto", help="automatic F0-F6 pass")
    q.set_defaults(f=cmd_auto)
    q.add_argument("--model")
    q.add_argument("--objective")
    q.add_argument("--run-dir")
    q.add_argument("--params", help='JSON {technique: {param: value}}')
    modes(q)
    q = s.add_parser("manual", help="per piece: techniques in series/parallel, with advice")
    q.set_defaults(f=cmd_manual)
    q.add_argument("--model")
    q.add_argument("--objective")
    q.add_argument("--piece", required=True)
    q.add_argument("--series")
    q.add_argument("--parallel")
    q.add_argument("--base", help="label of the starting candidate (default: R2B3m)")
    q.add_argument("--set", action="append", metavar="TECH.PARAM=VALUE",
                   help="per-step parameter (repeatable): steps, stop level, rank, ...")
    q.add_argument("--run-dir")
    modes(q)
    q = s.add_parser("status", help="phase/technique tree of a run")
    q.set_defaults(f=cmd_status)
    q.add_argument("--run-dir")
    q.add_argument("--refresh", action="store_true", help="read the state of launched jobs")
    q = s.add_parser("decide", help="answer the ambiguous questions")
    q.set_defaults(f=cmd_decide)
    q.add_argument("id", nargs="?")
    q.add_argument("answer", nargs="?", choices=("enable", "disable", "defer"))
    q.add_argument("--note")
    q.add_argument("--run-dir")
    q.add_argument("--list", action="store_true", help="list the questions without asking")
    for action in ("enable", "disable", "release"):
        q = s.add_parser(action, help=f"{action} a technique in the run (overrides the engine)")
        q.set_defaults(f=cmd_switch, action=action)
        q.add_argument("technique")
        q.add_argument("--run-dir")
    q = s.add_parser("rules", help="rules of the rule base")
    q.set_defaults(f=cmd_rules)
    q.add_argument("--phase")
    q.add_argument("--piece")
    q.add_argument("--model-type", default="hybrid")
    q = s.add_parser("measurements", help="rows of the measurement database")
    q.set_defaults(f=cmd_measurements)
    q.add_argument("--technique")
    q = s.add_parser("import", help="load the measurements already made into the database")
    q.set_defaults(f=cmd_import)
    q = s.add_parser("techniques", help="registered techniques")
    q.set_defaults(f=cmd_techniques)
    q.add_argument("--census", action="store_true", help="markdown lever census (docs/LEVERS.md)")
    q.add_argument("--short", action="store_true", help="with --census: one-line module table (README)")
    q = s.add_parser("forge", help="replay a recipe.yaml")
    q.set_defaults(f=cmd_forge)
    q.add_argument("--recipe", required=True)
    q.add_argument("--run-dir")
    modes(q)
    q = s.add_parser("analyze", help="read any GGUF: layer types, tensor families, applicable levers (serial/parallel)")
    q.set_defaults(f=cmd_analyze)
    q.add_argument("gguf")
    q.add_argument("--yaml", help="also write the analysis + plan as YAML")
    q = s.add_parser("replay", help="replay a recorded chain from public inputs (examples/thor/recipe_c.yaml)")
    q.set_defaults(f=cmd_replay)
    q.add_argument("recipe")
    q.add_argument("--set", action="append", default=[], dest="sets", metavar="ROOT=PATH",
                   help="value of a root of the recipe (also: `replay:` section of the config, FUCINA_REPLAY_<ROOT>)")
    g = q.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", help="resolve and check every step, run nothing (default)")
    g.add_argument("--run", action="store_true", help="run the selected steps, verify and record them")
    q.add_argument("--steps", help="comma-separated step ids")
    q.add_argument("--from", dest="first")
    q.add_argument("--to", dest="last")
    q.add_argument("--forbid", action="append", default=[], metavar="PREFIX",
                   help="fail if a resolved command contains this prefix (e.g. a private home directory)")
    q.add_argument("--script", help="with --dry-run: also write the resolved commands to this bash file")
    q.add_argument("--quiet", action="store_true", help="with --dry-run: summary only, no commands")
    q.add_argument("--allow-blocked", action="store_true", help="with --run: also run steps marked blocked")
    q.add_argument("--with-optional", action="store_true", help="also plan/run the optional steps (recaptures)")
    q.add_argument("--bench-list", help="file list of the bench dataset (BENCH_FILES.tsv): every bench input must "
                   "have a row with the same path and sha256")
    q = s.add_parser("replay-bundle", help="build / install / verify the code bundle of a replay recipe")
    q.set_defaults(f=cmd_replay_bundle)
    q.add_argument("args", nargs=argparse.REMAINDER)
    q = s.add_parser("tui", help="terminal user interface")
    q.set_defaults(f=cmd_tui)
    q.add_argument("--run-dir")
    return p


def cmd_replay(a) -> None:
    from .replay import recipe as RR, runner as RU
    r = RR.load(a.recipe)
    roots = RU.configured_roots(r, a.sets, (config.current().data.get("replay") or {}) if config.is_active() else {})
    sel = RU.select(r, a.steps, a.first, a.last, with_optional=a.with_optional)
    forbid = a.forbid + [x for x in os.environ.get("FUCINA_REPLAY_FORBID", "").split(":") if x]
    if a.run:
        res = RU.plan(r, roots, sel, forbid, show=False)
        if res["problems"]:
            raise SystemExit("refusing to run: " + "; ".join(res["problems"][:10]))
        raise SystemExit(RU.run(r, roots, sel, a.allow_blocked))
    res = RU.plan(r, roots, sel, forbid, show=not a.quiet, script=Path(a.script) if a.script else None)
    if a.bench_list:
        extra = RU.bench_list_problems(r, Path(a.bench_list), r.source.parent)
        print(f"bench list {a.bench_list}: {len(extra)} problems")
        for p in extra:
            print(f"  - {p}")
        res["problems"] += extra
    if a.quiet:
        print(f"{len(sel)} steps; blocked: {', '.join(res['blocked']) or 'none'}; problems: {len(res['problems'])}")
        for p in res["problems"]:
            print(f"  - {p}")
    raise SystemExit(1 if res["problems"] else 0)


def cmd_replay_bundle(a) -> None:
    from .replay import bundle
    raise SystemExit(bundle.main(a.args))


def cmd_analyze(a) -> None:
    from . import analyze as A
    an = A.analyze(a.gguf)
    pl = A.plan(an)
    print(A.report(an, pl))
    if a.yaml:
        import yaml
        Path(a.yaml).write_text(yaml.safe_dump({"analysis": an, "plan": pl}, sort_keys=False))
        print(f"written {a.yaml}")


def main(argv: Optional[list] = None) -> None:
    p = build_parser()
    a = p.parse_args(argv)
    try:
        if a.config or not config.is_active():
            config.activate(a.config)
        if getattr(a, "answer", None) is None and getattr(a, "id", None):
            p.error("decide ID needs an answer: enable | disable | defer")
        a.f(a)
    except config.ConfigError as e:
        p.exit(2, f"fucina: configuration error: {e}\n")


if __name__ == "__main__":
    main()

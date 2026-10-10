"""Printing of a run report with ``rich`` (tree phases -> techniques -> state/numbers, GPU queue, questions,
forecast). Without rich: plain text."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .executor import gpu_lock_owner
from .phases import BY_CODE

COLOURS = {"consolidated": "bold green", "passed": "green", "superseded": "dim green", "testing": "yellow",
           "waiting": "dim yellow", "to_decide": "bold magenta", "off": "red", "suspended": "dim",
           "excluded": "dim", "missing_tool": "dim red", "deferred": "magenta"}
SYMBOLS = {"consolidated": "●", "passed": "✔", "superseded": "≈", "testing": "◐", "waiting": "○", "to_decide": "?",
           "off": "✘", "suspended": "‖", "excluded": "–", "missing_tool": "!", "deferred": "…"}
PHASE_NAMES = {**{c: f"{c} {p.name}" for c, p in BY_CODE.items()}, "COMBO": "lever combination"}
ORDER = ["F0", "F1", "F2", "F3", "COMBO", "F4", "F5", "F6"]


def short_numbers(n: dict) -> str:
    if not n:
        return ""
    if "nearby_evidence" in n:
        e = n["nearby_evidence"]
        return f"[{n.get('nearby_base')}: {n.get('nearby_verdict')}] " + short_numbers(e)
    p = []
    if n.get("kl_pct") is not None:
        p.append(f"KL {n['kl_pct']:+.1f} %" + (f" ({n['z_kl']:+.1f} se)" if n.get("z_kl") is not None else ""))
    if n.get("mmlu") is not None:
        mm = n["mmlu"] if not isinstance(n["mmlu"], list) else max(n["mmlu"])
        p.append(f"MMLU {mm:.2f}" + (f" vs {n['mmlu_base']:.2f}" if n.get("mmlu_base") is not None else ""))
    if n.get("tg") is not None:
        p.append(f"tg {n['tg']:.1f}" + (f" ({n['tg_pct']:+.1f} %)" if n.get("tg_pct") is not None else ""))
    if n.get("file_bytes"):
        p.append(f"{n['file_bytes'] / 2 ** 30:.2f} GiB")
    if n.get("levers"):
        p.append("levers: " + ", ".join(n["levers"]))
    return ", ".join(p)


def by_phase(r: dict) -> dict:
    g: dict = {}
    for v in r["evaluations"]:
        g.setdefault(v["phase"], []).append(v)
    return g


def tree(r: dict) -> Any:
    from rich.text import Text
    from rich.tree import Tree
    ob = ", ".join(f"{k}={v}" for k, v in r["objective"].items() if k != "text")
    t = Tree(Text.assemble((f"{r['model']} ", "bold"), (f"objective {r['objective'].get('text', ob)}", "cyan"),
                           (f"  cap {r['cap_bytes'] / 2 ** 30:.2f} GiB  mode {r['mode']}", "dim")))
    g = by_phase(r)
    for f in ORDER:
        if f not in g:
            continue
        states = [v["state"] for v in g[f]]
        summary = " ".join(f"{SYMBOLS.get(s, '')}{states.count(s)}" for s in dict.fromkeys(states))
        branch = t.add(Text.assemble((PHASE_NAMES.get(f, f), "bold"), (f"  {summary}", "dim")))
        for v in g[f]:
            col = COLOURS.get(v["state"], "")
            pz = ",".join(v["pieces"])
            line = Text.assemble((f"{SYMBOLS.get(v['state'], ' ')} {v['technique']:<18}", col), (f" {v['state']:<15}", col),
                                 (f" {pz[:28]:<28}", "dim"), (f" {short_numbers(v['numbers'])}", "cyan"))
            leaf = branch.add(line)
            if v.get("reason"):
                leaf.add(Text(v["reason"][:220], style="dim"))
    return t


def queue_table(r: dict) -> Any:
    from rich.table import Table
    tb = Table(title="job queue (GPU: one at a time, machine-wide lock)", expand=False, show_lines=False)
    for c in ("id", "state", "gpu", "h", "job"):
        tb.add_column(c, overflow="fold")
    for j in r["queue"]:
        if j["state"] == "cancelled":
            continue
        st = j["state"] + (" ·placeholder" if j.get("placeholder") else "")
        tb.add_row(j["id"], st, "GPU" if j["gpu"] else "cpu", f"{j['est_h']:.1f}", j["name"][:90])
    return tb


def print_report(r: dict, run: Optional[Path] = None) -> None:
    try:
        from rich.console import Console
        from rich.panel import Panel
    except ImportError:
        return _print_text(r)
    c = Console()
    c.print(tree(r))
    c.print(Panel(" → ".join(f"{s['label']}" for s in r["history"]) + f"\n[dim]{r['candidate']['gguf']}[/dim]",
                  title="recipe (consolidated candidates)", expand=False))
    p = r.get("forecast") or {}
    if p:
        mm = lambda x: f" → MMLU {x:.2f}" if x is not None else ""
        c.print(Panel(f"measured: KL {p['kl_pct_measured']:+.1f} %{mm(p['mmlu_measured'])}\n"
                      f"with the levers being tested: KL {p['kl_pct_expected']:+.1f} %{mm(p['mmlu_expected'])}\n"
                      + (f"[bold]{p['comment']}[/bold]\n" if p.get("comment") else "")
                      + f"[dim]{p['law']}; {'; '.join(p['items'])}[/dim]", title="forecast (KL→MMLU law)", expand=False))
    c.print(queue_table(r))
    gpu = [j for j in r["queue"] if j["gpu"] and j["state"] not in ("done", "cancelled")]
    c.print(f"[dim]GPU jobs queued: {len(gpu)} (~{sum(j['est_h'] for j in gpu):.1f} h); "
            f"GPU lock: {gpu_lock_owner() or 'free'}[/dim]")
    if r.get("open_questions"):
        c.print(Panel("\n".join(f"[magenta]{d['id']}[/magenta]: {d['text']}  (advice: {d.get('advice')})"
                                for d in r["open_questions"]) + "\n[dim]fucina decide [ID enable|disable|defer][/dim]",
                      title="questions for you", expand=False))
    if run:
        c.print(f"[dim]run: {run}  (state.json, queue.json, questions.json, recipe.yaml)[/dim]")


def _print_text(r: dict) -> None:
    g = by_phase(r)
    for f in ORDER:
        for v in g.get(f, []):
            print(f"{f:5s} {v['technique']:18s} {v['state']:15s} {short_numbers(v['numbers'])} | {v['reason'][:160]}")
    print("recipe:", " -> ".join(s["label"] for s in r["history"]))
    if r.get("forecast"):
        print("forecast:", r["forecast"].get("comment") or r["forecast"])
    for j in r["queue"]:
        print(f"  {j['id']:36s} {j['state']:9s} {'GPU' if j['gpu'] else 'cpu'} {j['name'][:80]}")


def print_manual(outcome: dict) -> None:
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        print(json.dumps(outcome, ensure_ascii=False, indent=1))
        return
    c = Console()
    tb = Table(title=f"piece {outcome['piece']} — admissible techniques against {outcome['base']} "
                     f"(objective {outcome['objective']})")
    for k in ("phase", "technique", "advice", "numbers", "reason"):
        tb.add_column(k, overflow="fold")
    for v in outcome["techniques"]:
        tb.add_row(v["phase"], v["technique"], f"[{COLOURS.get(v['state'], '')}]{v['advice']}[/]",
                   short_numbers(v["numbers"]), v["reason"][:140])
    c.print(tb)
    if outcome.get("series"):
        c.print("[bold]series[/bold]: " + " → ".join(outcome["series"]))
    if outcome.get("parallel"):
        c.print("[bold]parallel[/bold]: " + " | ".join(outcome["parallel"]))
    for w in outcome.get("warnings") or []:
        c.print(f"[yellow]! {w}[/yellow]")
    c.print(f"[dim]run: {outcome['run']}[/dim]")

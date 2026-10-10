"""Terminal UI (SPEC section 7) built on ``textual``; falls back to ``rich`` (a static view refreshed in place).
Tree of phases -> techniques with state and numbers, details of the selected item, job queue, GPU lock, forecast.

Keys: e enable · d disable · x release (run switches) · 1/2/3 answer the question (enable/disable/defer)
      r recompute (one engine pass) · u refresh launched jobs · q quit."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from .executor import gpu_lock_owner
from .view import COLOURS, ORDER, PHASE_NAMES, SYMBOLS, by_phase, short_numbers


def read(run: Path) -> dict:
    return json.loads((Path(run) / "report.json").read_text())


def recompute(run: Path) -> dict:
    from .engine import Engine
    r = read(run)
    e = Engine(r["model_file"], r["objective_arg"], str(run), r["mode"], db=r.get("db"))
    out = e.run()
    e.launch()
    return out


def start(run: Path) -> None:
    try:
        import textual  # noqa: F401
    except ImportError:
        return start_rich(run)
    App = build_app()
    App(Path(run)).run()


def start_rich(run: Path, passes: int = 0) -> None:
    from rich.console import Console, Group
    from rich.live import Live

    from .view import queue_table, tree
    c = Console()
    n = 0
    with Live(console=c, refresh_per_second=0.5, screen=False) as live:
        while True:
            r = read(run)
            live.update(Group(tree(r), queue_table(r), f"[dim]GPU lock: {gpu_lock_owner() or 'free'} — Ctrl-C to quit[/dim]"))
            n += 1
            if passes and n >= passes:
                return
            time.sleep(5)


def build_app() -> Any:
    from rich.text import Text
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.widgets import DataTable, Footer, Header, Static, Tree

    class Fucina(App):
        TITLE = "fucina"
        CSS = """
        #left { width: 3fr; }
        #right { width: 2fr; border-left: solid $accent; padding: 0 1; }
        #tree { height: 1fr; }
        #queue { height: 12; border-top: solid $accent; }
        #head { height: auto; padding: 0 1; color: $text-muted; }
        """
        BINDINGS = [Binding("e", "switch('enable')", "enable"), Binding("d", "switch('disable')", "disable"),
                    Binding("x", "switch(None)", "release"), Binding("1", "answer('enable')", "yes"),
                    Binding("2", "answer('disable')", "no"), Binding("3", "answer('defer')", "defer"),
                    Binding("r", "recompute", "recompute"), Binding("u", "refresh_jobs", "jobs"),
                    Binding("q", "quit", "quit")]

        def __init__(self, run: Path):
            super().__init__()
            self.run_dir = run
            self.r = read(run)
            self.selected: Optional[dict] = None

        def compose(self) -> ComposeResult:
            yield Header()
            yield Static(id="head")
            with Horizontal():
                with Vertical(id="left"):
                    yield Tree("phases", id="tree")
                    yield DataTable(id="queue", zebra_stripes=True)
                yield Static(id="right")
            yield Footer()

        def on_mount(self) -> None:
            t = self.query_one("#queue", DataTable)
            t.add_columns("id", "state", "gpu", "h", "job")
            t.cursor_type = "row"
            self.draw()
            self.set_interval(5, self.reload)

        def reload(self) -> None:
            try:
                self.r = read(self.run_dir)
            except (OSError, ValueError):
                return
            self.draw(keep=True)

        def draw(self, keep: bool = False) -> None:
            r = self.r
            p = r.get("forecast") or {}
            self.sub_title = f"{r['model']} · {r['objective'].get('text', '')} · {r['mode']}"
            head = (f"recipe: {' → '.join(s['label'] for s in r['history'])}   "
                    f"forecast: {p.get('comment') or p.get('mmlu_expected', '')}\n"
                    f"GPU lock: {gpu_lock_owner() or 'free'}   open questions: {len(r.get('open_questions', []))}"
                    f"   updated {r.get('at', '')}")
            self.query_one("#head", Static).update(head)
            tr = self.query_one("#tree", Tree)
            tr.clear()
            tr.root.expand()
            tr.root.set_label(Text(f"{r['model']} — cap {r['cap_bytes'] / 2 ** 30:.2f} GiB", style="bold"))
            g = by_phase(r)
            for f in ORDER:
                if f not in g:
                    continue
                n = tr.root.add(Text(PHASE_NAMES.get(f, f), style="bold"), expand=True)
                for v in g[f]:
                    col = COLOURS.get(v["state"], "")
                    label = Text.assemble((f"{SYMBOLS.get(v['state'], ' ')} {v['technique']:<18}", col),
                                          (f" {v['state']:<15}", col), (f" {short_numbers(v['numbers'])[:60]}", "cyan"))
                    n.add_leaf(label, data=v)
            t = self.query_one("#queue", DataTable)
            t.clear()
            for j in r["queue"]:
                if j["state"] == "cancelled":
                    continue
                st = j["state"] + (" ·ph" if j.get("placeholder") else "")
                t.add_row(j["id"], st, "GPU" if j["gpu"] else "cpu", f"{j['est_h']:.1f}", j["name"][:80])
            self.show(self.selected if keep else None)

        def show(self, v: Optional[dict]) -> None:
            d = self.query_one("#right", Static)
            if not v:
                d.update(Text("select a technique in the tree\n\n" + (__doc__ or "").split("Keys:")[-1].strip(),
                              style="dim"))
                return
            q = next((q for q in self.r.get("open_questions", []) if q.get("technique") == v["technique"]), None)
            lines = [Text(f"{v['technique']} ({v['phase']})", style="bold"),
                     Text(f"state: {v['state']}", style=COLOURS.get(v["state"], "")),
                     Text(f"against: {v.get('base') or '-'}   pieces: {', '.join(v['pieces'])}"),
                     Text(f"switch: {v.get('switch') or '-'}", style="dim"), Text(""),
                     Text(v.get("reason") or ""), Text(""),
                     Text(json.dumps(v.get("numbers") or {}, ensure_ascii=False, indent=1)[:1500], style="cyan")]
            if v.get("jobs"):
                lines += [Text(""), Text("jobs: " + ", ".join(v["jobs"]), style="dim")]
            if q:
                lines += [Text(""), Text(f"QUESTION: {q['text']}\nadvice: {q.get('advice')} — "
                                         f"1 enable · 2 disable · 3 defer", style="bold magenta")]
            d.update(Text("\n").join(lines))

        def on_tree_node_highlighted(self, ev: Any) -> None:
            self.selected = ev.node.data
            self.show(self.selected)

        def action_switch(self, value: Optional[str]) -> None:
            if not self.selected or self.selected["phase"] in ("COMBO", "F6"):
                return self.notify("select a technique", severity="warning")
            from .engine import Switches
            Switches(self.run_dir).set(self.selected["technique"], value)
            self.notify(f"{self.selected['technique']}: {value or 'released'} — r to recompute")

        def action_answer(self, answer: str) -> None:
            from .engine import QuestionQueue
            if not self.selected:
                return
            qq = QuestionQueue(self.run_dir)
            q = next((x for x in qq.open() if x.get("technique") == self.selected["technique"]), None)
            if q is None:
                return self.notify("no open question on this technique", severity="warning")
            qq.answer(q["id"], answer)
            self.notify(f"{q['id']}: {answer} — recomputing")
            self.action_recompute()

        def action_recompute(self) -> None:
            try:
                self.r = recompute(self.run_dir)
            except Exception as e:          # the UI must not crash on an engine error
                return self.notify(f"engine error: {e}", severity="error")
            self.draw(keep=False)
            self.notify("engine pass done")

        def action_refresh_jobs(self) -> None:
            from .executor import Executor
            if self.r.get("mode") == "run":
                Executor(str(self.run_dir), "run").refresh()
            self.action_recompute()

    return Fucina

"""Job execution (SPEC section 6). Three modes:

* ``simulate`` -- no process at all: jobs stay in the queue as ``simulated`` (measurements come from the database);
* ``dry_run``  -- write the commands to ``dry_run_commands.sh`` without launching them;
* ``run``      -- ``systemd-run --user`` (unit ``fucina-<id>``), CPU at nice 15 and <= 8 threads; GPU jobs take the
  single machine-wide GPU lock themselves (the wrapped scripts call the lock functions of ``lock_script``).

The queue lives on disk (``queue.json`` in the run directory) so that a run can be resumed."""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from . import config

JOB_STATES = ("queued", "running", "done", "error", "simulated", "dry_run", "cancelled")
MODES = ("simulate", "dry_run", "run")


@dataclass
class Job:
    id: str
    name: str                       # human-readable description
    command: list                   # argv (including bash -c ...)
    gpu: bool = False               # True = takes the GPU lock (training or measurement)
    training: bool = False          # never two trainings at once
    est_h: float = 0.0
    depends_on: list = field(default_factory=list)
    output_dir: str = ""            # results directory
    phase: str = ""
    technique: str = ""
    piece: str = ""
    state: str = "queued"
    unit: str = ""
    started: str = ""
    finished: str = ""
    exit_code: Optional[int] = None
    memory_max: str = "14G"
    placeholder: bool = False       # depends on outcomes not known yet: not launched until the engine confirms it
    env: dict = field(default_factory=dict)

    def command_text(self) -> str:
        pre = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in self.env.items())
        return (f"env {pre} " if pre else "") + " ".join(shlex.quote(c) for c in self.command)


def bash(script: str) -> list:
    """A bash command with the common job environment (TMPDIR off /tmp, bounded CPU threads)."""
    cfg = config.current()
    env = ([f"TMPDIR={cfg['tmpdir']}"] if cfg.get("tmpdir") else []) + \
        [f"{k}={v}" for k, v in (cfg.get("job_env") or {}).items()] + ["XDG_RUNTIME_DIR=/run/user/$(id -u)"]
    pre = f"export {' '.join(env)}; set -uo pipefail; "
    return ["bash", "-c", pre + script]


class Executor:
    def __init__(self, run_dir: str, mode: str = "simulate"):
        assert mode in MODES, mode
        self.mode = mode
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.file = self.dir / "queue.json"
        self.jobs: list = []
        self.touched: set = set()       # ids enqueued during the current engine pass
        if self.file.exists():
            self.jobs = [Job(**d) for d in json.loads(self.file.read_text())]

    # ── queue ──
    def save(self) -> None:
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps([asdict(j) for j in self.jobs], ensure_ascii=False, indent=1))
        tmp.replace(self.file)

    def find(self, id_: str) -> Optional[Job]:
        return next((j for j in self.jobs if j.id == id_), None)

    def enqueue(self, j: Job) -> Job:
        """Idempotent: a job whose id is already queued is not duplicated (resume)."""
        self.touched.add(j.id)
        old = self.find(j.id)
        if old is not None:
            if old.placeholder and not j.placeholder:          # the engine confirmed the job: it becomes launchable
                old.placeholder = False
                old.command = j.command
                old.depends_on = j.depends_on
                if old.state == "simulated" and self.mode != "simulate":
                    old.state = "queued"
                self.save()
            return old
        if self.mode == "simulate":
            j.state = "simulated"
        self.jobs.append(j)
        self.save()
        return j

    def gpu_queue(self) -> list:
        return [j for j in self.jobs if j.gpu and j.state in ("queued", "running", "simulated", "dry_run")]

    def cancel(self, id_: str, reason: str = "") -> None:
        j = self.find(id_)
        if j is not None and j.state in ("queued", "simulated", "dry_run"):
            j.state = "cancelled"
            j.name += f" [cancelled: {reason}]" if reason else ""
            self.save()

    # ── execution ──
    def ready(self) -> list:
        done = {j.id for j in self.jobs if j.state == "done" or (self.mode == "dry_run" and j.state == "dry_run")}
        return [j for j in self.jobs
                if j.state == "queued" and not j.placeholder and all(d in done for d in j.depends_on)]

    def advance(self) -> list:
        """Refresh running jobs and launch the ready ones. GPU: one at a time from Fucina (the lock is machine-wide
        anyway); trainings: never two at once. Returns the launched jobs."""
        self.refresh()
        launched = []
        gpu_busy = any(j.gpu and j.state == "running" for j in self.jobs)
        for j in self.ready():
            if j.gpu and gpu_busy:
                continue
            self.launch(j)
            launched.append(j)
            gpu_busy = gpu_busy or j.gpu
        return launched

    def launch(self, j: Job) -> None:
        if self.mode == "simulate":
            j.state = "simulated"
        elif self.mode == "dry_run":
            j.state = "dry_run"
            with open(self.dir / "dry_run_commands.sh", "a") as f:
                f.write(f"# {j.id}: {j.name}\n{j.command_text()}\n")
        else:
            j.unit = f"fucina-{j.id}".replace("_", "-")[:60]
            Path(j.output_dir or self.dir).mkdir(parents=True, exist_ok=True)
            log = Path(j.output_dir or self.dir) / f"{j.id}.log"
            rc = Path(j.output_dir or self.dir) / f"{j.id}.exitcode"
            cmd = ["systemd-run", "--user", "--collect", "--unit", j.unit, "-p", f"MemoryMax={j.memory_max}",
                   "-p", "CPUWeight=20", "nice", "-n", "15", "bash", "-c",
                   f"{j.command_text()} > {shlex.quote(str(log))} 2>&1; echo $? > {shlex.quote(str(rc))}"]
            subprocess.run(cmd, check=True, capture_output=True)
            j.state = "running"
        j.started = time.strftime("%Y-%m-%dT%H:%M")
        self.save()

    def refresh(self) -> None:
        if self.mode != "run":
            return
        for j in self.jobs:
            if j.state != "running":
                continue
            r = subprocess.run(["systemctl", "--user", "is-active", "-q", j.unit])
            if r.returncode == 0:
                continue
            rc = Path(j.output_dir or self.dir) / f"{j.id}.exitcode"
            res = rc.read_text().strip() if rc.exists() else ""
            j.exit_code = int(res) if res.isdigit() else 1          # unit gone without an exit code = error
            j.state = "done" if j.exit_code == 0 else "error"
            j.finished = time.strftime("%Y-%m-%dT%H:%M")
        self.save()


def gpu_lock_owner() -> str:
    """Who holds the machine-wide GPU lock (read-only, for the UI)."""
    f = config.current().get("lock_owner_file")
    if not f:
        return ""
    try:
        return Path(f).read_text().strip()
    except OSError:
        return ""


def log_cycle(text: str, status: Optional[str] = None) -> None:
    """Append one line to the optional activity log (date<TAB>STATUS<TAB>text, no «rc=»)."""
    cfg = config.current()
    text = text.replace("\t", " ").replace("\n", " ").replace("rc=", "exit code ")
    if os.getenv("FUCINA_NO_CYCLE_LOG") or not cfg.get("cycle_log"):
        return
    with open(cfg["cycle_log"], "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M')}\t{status or cfg['cycle_log_status']}\t{text}\n")

"""Recipe C, step 27: S7 on-policy distillation (QAOPD, arXiv 2609.26708) on T550, 10 rounds x 2 steps -> g10.

Re-implementation of the recorded launcher ``c_s7_t550_1020.sh`` (20 Oct 2026) for any machine:

* the student server (``llama-server``) and the 21-seed spoken probe run as **child processes** of this script
  (the record started them as ``systemd-run --user`` units, which only exist on the recording machine);
* no machine-wide GPU lock, no activity log, no export governor, no system-disk guard: those were house rules, not
  part of the computation;
* every location is an argument.

What is kept exactly: the variant ``thor`` (20 steps = 10 rounds x 2 steps x 4 rows, lr 1.5e-4 / F32 1.5e-6,
1536 generated tokens, strict official reward, GSM8K-256 greedy gate every 10 steps), the trainer argv of every round,
the teacher-cache settings, the export sequence and the student server flags. Each round: fresh rollouts of the
current student -> teacher hidden states on them -> 2 training steps -> export ``Thor-T-C-S7-T550-g<k>.gguf``.
As in the record, the final GGUF is g10 whatever the gates say (the record closed with "no checkpoint accepted" and
T400 was built on g10); the gates are measurements, written to ``checkpoint.tsv``.

Usage::

    python -m fucina.replay.thor.s7_onpolicy --host T550.gguf --out S7_DIR --code ONPOLICY_THOR_V2 \\
        --faithful FAITHFUL_ARM_DIR --teacher-q8 TEACHER.gguf --engine ENGINE_BIN --python PYTHON \\
        [--probe-judge sonda_giudice.py] [--port 11476] [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

STEPS, STEPS_PER_ROUND, ROWS_PER_STEP, GATE_EVERY, N_GATE, MAX_TOKENS = 20, 2, 4, 10, 256, 1536
LR, LR_F32, HELD_ROWS = "1.5e-4", "1.5e-6", 2
F32 = ("attn_norm,post_attention_norm,attn_q_norm,attn_k_norm,ssm_norm,ffn_gate_inp,ffn_gate_inp_shexp,ssm_a,ssm_dt,"
       "ssm_conv1d,output_norm")
KQAT = "cache_densi,silu,tile,amax,mm16_grande,guardia"
TRAIN_ENV = {"E2E_SENZA_COESISTENZA": "1", "E2E_GDN": "veloce", "E2E_GDN_MODULO": "gdn_veloce_corsa_v5",
             "E2E_MOTORE": "triton2", "E2E_KQAT": KQAT, "E2E_DENSI_SSM_AB": "1"}
NO_GPU = {"HIP_VISIBLE_DEVICES": "", "CUDA_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": ""}
ENGINE_ENV = {"GGML_VK_TQ1R_Q8": "1", "GGML_VK_DISABLE_GRAPH_OPTIMIZE": "1"}
PROBE_SEEDS = (1, 2, 3, 7, 42, 100, 999)


def trainer_argv(run: str, gguf: Path, teacher: Path, snapshots: Path) -> list[str]:
    """The recorded argv of one round (``argv()`` of the launcher)."""
    return ("allena --corsa {run} --gguf {gguf} --teacher {teacher} --ricetta B3_ind32 --testa-kl bf16x3 "
            "--logit-maestro bf16 --epoche 1 --righe-passo {rp} --righe-micro 1 --heldout-ogni {pg} "
            "--heldout-righe-rapide 0 --heldout-pieno-ogni 1 --non-consumare --K0 524288 --pv-codici --pv-rho 0.03 "
            "--pv-passi-accumulo 1 --pv-modo vbop --pv-rho-adattivo --perdita cakld --gamma 1.0 --peso-token generati "
            "--opd-kl campionata --opd-beta 1.0 --opd-eakld-k 16 --allena-f32 {f32} --lr {lr} --lr-f32 {lrf} "
            "--ema-theta 0.9 --ripresa-ogni {pg} --ottimizzatore-theta adamw --pv-accetta guadagno "
            "--pv-z-guadagno-min 1 --istantanea-ogni-passo --istantanee-dir {snap} --istantanee-tieni 1 "
            "--istantanee-libero-min-gib 1 --pv-sonda-dosi --pv-dosi-frazioni 1/4096,1/1024,1/256,1/64,1/16,1/4,1 "
            "--pv-ricentra --pv-ricentra-giri 3 --densi-ternari tutte").format(
        run=run, gguf=gguf, teacher=teacher, rp=ROWS_PER_STEP, pg=STEPS_PER_ROUND, f32=F32, lr=LR, lrf=LR_F32,
        snap=snapshots).split()


class Runner:
    def __init__(self, a):
        self.a = a
        self.out: Path = a.out
        self.log = a.out / "s7.log"
        self.server: subprocess.Popen | None = None

    # ── processes ──────────────────────────────────────────────────────────────────────────────────────────────
    def say(self, msg: str) -> None:
        line = f"[{time.strftime('%F %T')}] {msg}"
        print(line, flush=True)
        if not self.a.dry_run:
            with open(self.log, "a") as f:
                f.write(line + "\n")

    def run(self, argv: list, cwd: Path | None = None, env: dict | None = None, log: Path | None = None,
            ok=(0,)) -> int:
        shown = " ".join(shlex.quote(str(x)) for x in argv)
        self.say(f"$ {' '.join(f'{k}={v}' for k, v in (env or {}).items())} {shown}" + (f" > {log}" if log else ""))
        if self.a.dry_run:
            return 0
        with open(log, "a") if log else open(os.devnull, "w") as f:
            rc = subprocess.run([str(x) for x in argv], cwd=cwd, env={**os.environ, **(env or {})},
                                stdout=f if log else None, stderr=subprocess.STDOUT if log else None).returncode
        if rc not in ok:
            raise SystemExit(f"command failed ({rc}): {shown[:300]}")
        return rc

    def start_server(self, gguf: Path, probe: bool = False) -> None:
        """llama-server on the student, as a child process. Training server: 8 slots x 2048 tokens, no reasoning
        parsing; probe server: the probe's sampling settings."""
        self.stop_server()
        a = self.a
        if probe:
            args = ["--alias", "sonda", "--ctx-size", "8192", "--parallel", "1", "--flash-attn", "on",
                    "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--temp", "0.6", "--min-p", "0.03",
                    "--top-p", "0.9", "--jinja", "--reasoning-budget", "4096"]
        else:
            args = ["--ctx-size", str(8 * 2048), "--parallel", "8", "--jinja", "--reasoning-format", "none", "--no-webui"]
        argv = [a.engine / "llama-server", "--model", gguf, "-ngl", "999", "--host", "127.0.0.1", "--port",
                str(a.port), "--threads", "8", *args]
        self.say("server: " + " ".join(map(str, argv)))
        if a.dry_run:
            return
        env = {**os.environ, **ENGINE_ENV, "LD_LIBRARY_PATH": str(a.engine)}
        self.server = subprocess.Popen([str(x) for x in argv], env=env, stdout=subprocess.DEVNULL,
                                       stderr=open(self.out / "server.log", "a"), start_new_session=True)
        for _ in range(90):
            if self.server.poll() is not None:
                raise SystemExit(f"llama-server exited ({self.server.returncode}); see {self.out}/server.log")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=2) as r:
                    if b"ok" in r.read():
                        return
            except OSError:
                pass
            time.sleep(5)
        self.stop_server()
        raise SystemExit("llama-server did not become healthy in 7.5 minutes")

    def stop_server(self) -> None:
        if self.server and self.server.poll() is None:
            self.server.terminate()
            try:
                self.server.wait(60)
            except subprocess.TimeoutExpired:
                self.server.kill()
                self.server.wait()
        self.server = None

    def opf(self, *args, log: Path, env: dict | None = None) -> int:
        """The frozen faithful arm (opd_fedele.py) run from the trainer tree, as ``OPF`` in the record."""
        return self.run([self.a.python, self.a.faithful / "opd_fedele.py", *args], cwd=self.a.code,
                        env={"OPD_THOR_V2": str(self.a.code), **(env or {})}, log=log)

    # ── pieces of the round ─────────────────────────────────────────────────────────────────────────────────────
    def teacher(self, run: str, corpus: Path) -> None:
        a = self.a
        sha = a.teacher_sha or (hashlib.sha256(a.teacher_q8.read_bytes()).hexdigest() if not a.dry_run else "SHA")
        log = self.out / f"maestro_{run}.log"
        base = [a.python, "e2e_teacher.py"]
        self.run(base + ["prepara", "--corsa", run, "--radice", self.out / "teacher", "--corpus-dir", corpus,
                         "--variante", "onpolicy", "--lati", "heldout,train", "--modo", "segmenti", "--righe-blocco",
                         "8", "--modello", a.teacher_q8, "--sha-modello", sha], cwd=a.code, log=log)
        self.run(base + ["esegui", "--corsa", run, "--radice", self.out / "teacher"], cwd=a.code, log=log)

    def gate(self, gguf: Path, name: str) -> dict | None:
        """GSM8K-256 greedy, strict official reward (measurement; the record never stopped on it)."""
        d = self.out / "cancelli" / name
        if not self.a.dry_run:
            d.mkdir(parents=True, exist_ok=True)
            if (d / "riassunto.json").exists():
                return json.loads((d / "riassunto.json").read_text())
        self.start_server(gguf)
        try:
            self.opf("cancello", "--server", f"http://127.0.0.1:{self.a.port}", "--out", d, "--n-cancello",
                     str(N_GATE), "--max-token", str(MAX_TOKENS), log=d / "cancello.log")
        finally:
            self.stop_server()
        return None if self.a.dry_run else json.loads((d / "riassunto.json").read_text())

    def probe(self, gguf: Path, name: str) -> None:
        """21-seed spoken probe (3 questions x 7 seeds), answers judged by the probe judge; measurement only."""
        if not self.a.probe_judge:
            return
        d = self.out / "cancelli" / name
        raw = d / "sonda_grezze.jsonl"
        self.start_server(gguf, probe=True)
        try:
            if self.a.dry_run:
                self.say(f"probe: 3 questions x seeds {PROBE_SEEDS} -> {raw}")
                return
            right = 0
            for dom in ("D1", "D2", "D3"):
                q = subprocess.run([sys.executable, str(self.a.probe_judge), "domanda", dom], capture_output=True,
                                   text=True, check=True).stdout.strip()
                for seed in PROBE_SEEDS:
                    body = json.dumps({"model": "s", "stream": False, "max_tokens": 2000, "temperature": 0.6,
                                       "seed": seed, "messages": [{"role": "user", "content": q}],
                                       "chat_template_kwargs": {"enable_thinking": True}}).encode()
                    req = urllib.request.Request(f"http://127.0.0.1:{self.a.port}/v1/chat/completions", body,
                                                 {"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=300) as r:
                        answer = r.read()
                    verdict = subprocess.run([sys.executable, str(self.a.probe_judge), "giudica", "--dominio", dom,
                                              "--seed", str(seed), "--grezzo", str(raw)], input=answer,
                                             capture_output=True, check=True).stdout.decode().split("|")[0]
                    right += verdict.strip() == "OK"
            (d / "sonda.out").write_text(f"giuste {right}/21\n")
            self.say(f"probe {name}: {right}/21")
        finally:
            self.stop_server()

    def round(self, k: int, host: Path) -> Path:
        a, out = self.a, self.out
        d, run, snaps = out / f"giro_{k}", f"C-OPDF-g{k}", out / f"ist_g{k}"
        gk = out / f"Thor-T-C-S7-T550-g{k}.gguf"
        if (d / "FATTO.json").exists():
            return Path(json.loads((d / "FATTO.json").read_text())["gguf"])
        if not a.dry_run:
            d.mkdir(parents=True, exist_ok=True)
        rows = ROWS_PER_STEP * STEPS_PER_ROUND
        if a.dry_run or not (d / "corpus" / "riassunto.json").exists():          # (a) fresh rollouts
            self.start_server(host)
            try:
                self.opf("genera", "--server", f"http://127.0.0.1:{a.port}", "--stato", out / "stato_prompt",
                         "--giro", str(k), "--out", d / "corpus", "--righe", str(rows), "--prompt-primo", str(2 * rows),
                         "--n-cancello", str(N_GATE), "--max-token", str(MAX_TOKENS), log=d / "rollout.log")
            finally:
                self.stop_server()
        for f in ("e2e_heldout_4096_onpolicy.pt", "onpolicy_heldout.pt"):
            self.run(["cp", "-f", out / "heldout_kl" / f, d / "corpus" / f])
        self.teacher(f"opdf-g{k}", d / "corpus")                               # (b) teacher on the rollouts
        snap = snaps / f"passo_{STEPS_PER_ROUND:06d}"
        if a.dry_run or not (snap / "registri_pv.npz").exists():                # (c) 2 training steps
            resume: list = []
            for attempt in (1, 2, 3):
                rc = self.run([a.python, a.code / "e2e_allena.py", *trainer_argv(run, host, out / "teacher" /
                               f"opdf-g{k}", snaps), *resume], cwd=a.code,
                              env={"E2E_ALLENA_DIR": str(out / "allena"), **TRAIN_ENV}, log=d / "allena.out",
                              ok=(0, 75))
                if rc == 75 and attempt < 3:                                    # the trainer's watchdog: resume
                    resume = ["--riprendi"]
                    continue
                break
            if not a.dry_run and not (snap / "registri_pv.npz").exists():
                raise SystemExit(f"round {k}: training ended without the step-{STEPS_PER_ROUND} snapshot")
        if a.dry_run or not Path(f"{gk}.sha256").exists():                      # (d) export
            n = f"passo_{STEPS_PER_ROUND:06d}"
            scaled = out / f"ssd_{gk.stem}.scale"
            self.run(["cp", "-f", out / "allena" / run / "config.json", snaps / "config.json"])
            cpu = {**NO_GPU, "OMP_NUM_THREADS": "8", "E2E_DENSI_SSM_AB": "1"}
            self.run(["nice", "-n", "19", a.python, "e2e_esporta.py", "--sorgente", host, "--corsa", snaps,
                      "--istantanea", n, "--uscita", scaled, "--device", "cpu"], cwd=a.code, env=cpu,
                     log=d / "export.log")
            self.run(["nice", "-n", "19", a.python, "e2e_pv_codici.py", "esporta", "--scalato", scaled, "--dst", gk,
                      "--registri", snaps / n / "registri_pv.npz"], cwd=a.code, env=cpu, log=d / "export.log")
            if not a.dry_run:
                scaled.unlink(missing_ok=True)
                if gk.stat().st_size != host.stat().st_size:
                    raise SystemExit(f"round {k}: exported size differs from the host")
                Path(f"{gk}.sha256").write_text(hashlib.sha256(gk.read_bytes()).hexdigest() + "\n")
        if not a.dry_run:
            (d / "FATTO.json").write_text(json.dumps({"gguf": str(gk), "passi": k * STEPS_PER_ROUND}))
        self.say(f"round {k}: {gk}")
        return gk

    def main(self) -> Path:
        a, out = self.a, self.out
        if not a.dry_run:
            out.mkdir(parents=True, exist_ok=True)
        try:
            # references of the host: GSM8K gate, probe, held-out KL set (its corpus is reused by every round)
            self.gate(a.host, "ospite")
            self.probe(a.host, "ospite")
            hk = out / "heldout_kl"
            if a.dry_run or not (hk / "riassunto_heldout.json").exists():
                self.start_server(a.host)
                try:
                    self.opf("held-kl", "--server", f"http://127.0.0.1:{a.port}", "--out", hk, "--n-cancello",
                             str(N_GATE), "--righe-held", str(HELD_ROWS), "--max-token", str(MAX_TOKENS),
                             log=out / "heldout_kl.log")
                finally:
                    self.stop_server()
            g = a.host
            rounds = (STEPS + STEPS_PER_ROUND - 1) // STEPS_PER_ROUND
            for k in range(1, rounds + 1):
                g = self.round(k, g)
                step = k * STEPS_PER_ROUND
                if step % GATE_EVERY == 0 or k == rounds:
                    r = self.gate(g, f"p{step}")
                    self.probe(g, f"p{step}")
                    if r is not None:
                        with open(out / "checkpoint.tsv", "a") as f:
                            f.write(f"{step}\t{g}\t{r['acc_stretto']:.2f}\t{r['giuste_stretto']}\n")
            return g
        finally:
            self.stop_server()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", type=Path, required=True, help="T550 GGUF")
    ap.add_argument("--out", type=Path, required=True, help="S7 working directory")
    ap.add_argument("--code", type=Path, required=True, help="on-policy trainer tree (ONPOLICY/codice/thor_v2)")
    ap.add_argument("--faithful", type=Path, required=True, help="frozen faithful arm: opd_fedele.py + reward")
    ap.add_argument("--teacher-q8", type=Path, required=True)
    ap.add_argument("--teacher-sha", default="e1425b9cba1c7514989cefe944fa9f4a68c34f480bcded2eec3549f75eea0fe0")
    ap.add_argument("--engine", type=Path, required=True, help="bin/ of the release engine")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--port", type=int, default=11476)
    ap.add_argument("--probe-judge", type=Path, help="sonda_giudice.py (omit to skip the probe)")
    ap.add_argument("--dry-run", action="store_true", help="print every command, start nothing")
    a = ap.parse_args(argv)
    g = Runner(a).main()
    print(f"g10: {g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

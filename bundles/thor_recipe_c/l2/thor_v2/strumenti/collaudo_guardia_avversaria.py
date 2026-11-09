#!/usr/bin/env python3
"""Verifica AVVERSARIA della guardia di applica_patch_tempo.py (25/9 mattina) — solo CPU, niente GPU.

Forme di coda che il collaudo del progettista NON provava. Ogni processo finto dorme e basta: le righe che
"lavorerebbero" stanno DOPO un `exit 0`, quindi non vengono mai eseguite (la guardia legge testo, non esegue).
Radice finta sotto --lavoro (niente file nuovi in thor_v2); per le forme con ~ si usa la radice VERA solo
come TESTO dentro uno script che non la tocca.

    python3 strumenti/collaudo_guardia_avversaria.py --lavoro <scratch> [--modulo <applica_patch_tempo.py>]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ESITI: list = []
VIVI: list = []
LAVORO = ["//nessuna//"]
MARCA = "avversario_guardia_25set"


def ok(nome, cond, dettaglio=""):
    ESITI.append((nome, bool(cond)))
    print(f"  {'✅' if cond else '⛔'} {nome}" + (f" — {dettaglio}" if dettaglio else ""), flush=True)


def carica(modulo: Path):
    spec = importlib.util.spec_from_file_location("applica_in_prova", modulo)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def lancia(argv, cwd, env=None, **kw) -> int:
    p = subprocess.Popen(argv, cwd=cwd, env={**os.environ, **(env or {})}, stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True, **kw)
    VIVI.append(p.pid)
    return p.pid


def ferma_tutti():
    for pid in VIVI:
        try:
            argv = open(f"/proc/{pid}/cmdline", "rb").read()
        except OSError:
            continue
        try:
            fd255 = os.readlink(f"/proc/{pid}/fd/255")
        except OSError:
            fd255 = ""
        try:
            cwd = os.readlink(f"/proc/{pid}/cwd")
        except OSError:
            cwd = ""
        # PID verificato un istante prima: e' uno dei nostri finti (marca, cartella di lavoro del collaudo, o sleep)
        if (MARCA.encode() in argv or argv.startswith(b"sleep\x00") or LAVORO[0].encode() in argv
                or LAVORO[0] in fd255 or cwd.startswith(LAVORO[0])):
            try:
                os.killpg(pid, signal.SIGTERM) if os.getpgid(pid) == pid else os.kill(pid, signal.SIGTERM)
            except OSError:
                pass


def script(p: Path, testo: str) -> Path:
    p.write_text("#!/bin/bash\n# " + MARCA + "\n" + testo)
    p.chmod(0o755)
    return p


def trova_pid(marca: str) -> int:
    for _ in range(50):
        for v in os.listdir("/proc"):
            if v.isdigit():
                try:
                    argv = open(f"/proc/{v}/cmdline", "rb").read().split(b"\0")
                except OSError:
                    continue
                if marca.encode() in argv and not argv[0].endswith(b"sleep"):
                    VIVI.append(int(v))
                    return int(v)
        time.sleep(0.1)
    raise RuntimeError(f"processo con {marca} non trovato")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modulo", type=Path, default=Path(__file__).resolve().parent / "applica_patch_tempo.py")
    ap.add_argument("--lavoro", type=Path, required=True)
    a = ap.parse_args()
    A = carica(a.modulo.resolve())
    lav = a.lavoro.resolve()
    lav.mkdir(parents=True, exist_ok=True)
    LAVORO[0] = str(lav)
    genitore = lav / "tony-forgia-finto"
    radice = genitore / "thor_v2"
    (radice / "strumenti").mkdir(parents=True, exist_ok=True)
    fuori = lav / "fuori"                     # come stato/velocita/: fuori dalla radice
    fuori.mkdir(exist_ok=True)
    VERA = Path(A.THOR_V2)
    rel_casa = VERA.relative_to(Path.home())

    # ── forme che la guardia DEVE vedere ──
    si = {}
    # F1 = stato/velocita/coda_q4.sh (24/9 20:48): script FUORI, aspetta PRIMA di fare cd nella radice
    s = script(fuori / "coda_q4_finta.sh", f"until false; do sleep 5; done\nexit 0\ncd {radice} && ./qualita.sh 0.01 150\n")
    si["coda FUORI dalla radice che aspetta PRIMA del cd (forma di coda_q4.sh)"] = lancia([str(s)], "/")
    # F2 = la stessa lanciata come `bash script` con cwd fuori
    s = script(fuori / "coda_bash_finta.sh", f"while sleep 5; do :; done\nexit 0\ncd {radice}\n./ricetta.sh\n")
    si["coda FUORI lanciata con `bash script` (cwd fuori)"] = lancia(["bash", str(s)], str(fuori))
    # F3 = script fuori con variabile: D=...; cd $D/thor_v2
    s = script(fuori / "coda_variabile_finta.sh", f"D={genitore}\nwhile sleep 5; do :; done\nexit 0\ncd $D/thor_v2\n")
    si["coda FUORI con la radice composta da variabile (D=…; cd $D/thor_v2)"] = lancia([str(s)], "/")
    # F4 = script DENTRO la radice lanciato ./x.sh che poi fa `cd /` (argv relativo, cwd fuori)
    s = script(radice / "coda_che_esce_finta.sh", "cd /\nwhile sleep 5; do :; done\n")
    si["script DENTRO la radice, lanciato ./x.sh, che poi fa cd / (argv relativo + cwd fuori)"] = lancia(
        ["./coda_che_esce_finta.sh"], str(radice))
    # F5 = nohup nello stesso comando dell'applicatore: condivide la pipe di stdout (vedi prova sotto)
    # F6 = argomento via collegamento simbolico a un file che NON esiste ancora
    coll = lav / "collegamento_a_radice"
    if coll.is_symlink() or coll.exists():
        coll.unlink()
    coll.symlink_to(radice)
    si["argomento via collegamento simbolico a un file che NON esiste ancora"] = lancia(
        ["bash", "-c", f"sleep 300; exit 0; {coll}/nuovo_non_esiste.sh  # {MARCA}"], "/")
    # F7 = opzione attaccata al percorso (-C/percorso)
    si["opzione attaccata al percorso (env -C<radice> dentro un bash -c)"] = lancia(
        ["bash", "-c", f"sleep 300; exit 0; env -C{radice} ./x.sh  # {MARCA}"], "/")
    # F8 = script fuori con la radice VERA scritta con ~ (solo testo, mai eseguito) — sulla radice vera
    s = script(fuori / "coda_tilde_finta.sh", f"while sleep 5; do :; done\nexit 0\ncd ~/{rel_casa} && ./x.sh\n")
    pid_tilde = lancia([str(s)], "/")
    # F9 = python lanciato da fuori con uno script fuori che inserisce la radice in sys.path
    drv = fuori / "driver_finto.py"
    drv.write_text(f"# {MARCA}\nimport sys, time\nsys.path.insert(0, {str(radice)!r})\ntime.sleep(300)\n")
    si["python con uno script FUORI che mette la radice in sys.path"] = lancia([sys.executable, str(drv)], "/")
    # F10 = python con PYTHONPATH RELATIVO alla sua cwd
    si["PYTHONPATH relativo (thor_v2/strumenti) con cwd nel genitore"] = lancia(
        [sys.executable, "-c", f"import time; time.sleep(300)  # {MARCA}"], str(genitore),
        env={"PYTHONPATH": "thor_v2/strumenti"})
    # ── forme che NON deve vedere ──
    no = {
        "prefisso trappola thor_v2_altro in uno script fuori": lancia(
            [str(script(fuori / "trappola_finta.sh", f"while sleep 5; do :; done\nexit 0\ncd {radice}_altro\n"))], "/"),
        "script fuori senza nessun riferimento alla radice": lancia(
            [str(script(fuori / "innocuo_finto.sh", "while sleep 5; do :; done\n"))], "/"),
        # come ~/.local/bin/ping-claude.sh (servizio sempre vivo): NOMINA la radice (confronto con la cwd
        # degli altri, commenti) ma non la usa — segnalarlo bloccherebbe l'applicatore per sempre
        "monitor che confronta la cwd altrui con la radice (forma di ping-claude.sh)": lancia(
            [str(script(fuori / "monitor_finto.sh",
                        f"# thor_v2 in corso: {radice} e' la cartella da guardare\nwhile sleep 5; do\n"
                        f"  for p in 1; do [ \"$(readlink /proc/$p/cwd)\" = {radice} ] || continue; done\ndone\n"))],
            "/"),
    }
    time.sleep(1.0)
    trovati = {c["pid"]: c for c in A.processi_in_conflitto(radice, dettagli=True)}
    for nome, pid in si.items():
        ok(f"VISTO: {nome} (pid {pid})", pid in trovati, "; ".join(trovati[pid]["motivi"]) if pid in trovati else "NON visto")
    for nome, pid in no.items():
        ok(f"NON visto (giusto): {nome} (pid {pid})", pid not in trovati,
           "; ".join(trovati[pid]["motivi"]) if pid in trovati else "")
    veri = {c["pid"]: c for c in A.processi_in_conflitto(VERA, dettagli=True)}
    ok(f"VISTO (radice vera, solo testo): coda FUORI con ~/{rel_casa} dopo l'attesa (pid {pid_tilde})",
       pid_tilde in veri, "; ".join(veri[pid_tilde]["motivi"]) if pid_tilde in veri else "NON visto")

    # F5: nohup + applicatore nello STESSO comando: il finto eredita la pipe di stdout dell'applicatore
    coda5 = script(radice / "coda_nohup_finta.sh", "while sleep 5; do :; done\n")
    # la coda tiene aperta la pipe: niente subprocess.run (aspetterebbe la sua chiusura per sempre),
    # si legge la sola riga JSON dell'applicatore
    p5 = subprocess.Popen(["bash", "-c", f"cd / && nohup {coda5} {MARCA}_nohup & sleep 0.5; "
                           f"{sys.executable} {a.modulo.resolve()} --solo-guardia --json --cartella {radice}"],
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, cwd="/",
                          start_new_session=True)
    VIVI.append(p5.pid)
    riga5 = ""
    for riga in p5.stdout:
        if riga.startswith("{"):
            riga5 = riga
            break

    class _R:
        stdout, stderr = riga5, ""
    r = _R()
    try:
        pid5 = trova_pid(f"{MARCA}_nohup")
        d = json.loads(r.stdout.strip().splitlines()[-1])
        visti5 = {c["pid"] for c in d["conflitti"]}
        ok(f"VISTO: `nohup coda &` nello stesso comando dell'applicatore (pipe di stdout condivisa, pid {pid5})",
           pid5 in visti5, f"pipeline dichiarata {d.get('pipeline')}, {'VISTO' if pid5 in visti5 else 'escluso come compagno di pipeline'}")
    except (RuntimeError, ValueError, IndexError) as e:
        ok("F5 eseguito", False, f"{e} · {r.stdout[-300:]} {r.stderr[-300:]}")

    # ── niente auto-blocco, in piu' modi di lancio (radice finta con cwd dentro) ──
    ap_ = a.modulo.resolve()
    modi = {
        "diretto con cwd nella radice": ["bash", "-c", f"cd {radice} && {sys.executable} {ap_} --solo-guardia --json --cartella {radice}"],
        "| tee | cat": ["bash", "-c", f"cd {radice} && {sys.executable} {ap_} --solo-guardia --json --cartella {radice} | tee {lav}/t.json | cat"],
        "2>&1 | tee": ["bash", "-c", f"cd {radice} && {sys.executable} {ap_} --solo-guardia --json --cartella {radice} 2>&1 | tee {lav}/t2.json"],
        "timeout 60": ["bash", "-c", f"cd {radice} && timeout 60 {sys.executable} {ap_} --solo-guardia --json --cartella {radice}"],
        "nohup (stdout su file)": ["bash", "-c", f"cd {radice} && nohup {sys.executable} {ap_} --solo-guardia --json --cartella {radice} > {lav}/n.json 2>/dev/null; cat {lav}/n.json"],
        "setsid --wait": ["bash", "-c", f"cd {radice} && setsid --wait {sys.executable} {ap_} --solo-guardia --json --cartella {radice}"],
        "echo | applicatore (pipe in ingresso)": ["bash", "-c", f"cd {radice} && echo x | {sys.executable} {ap_} --solo-guardia --json --cartella {radice}"],
        "da uno script DENTRO la radice (catena che applica)": [str(script(radice / "catena_che_applica_finta.sh",
            f"cd {radice}\n{sys.executable} {ap_} --solo-guardia --json --cartella {radice}\n"))],
    }
    finti = set(si.values()) | set(no.values())
    for nome, argv in modi.items():
        # i processi VERI della macchina (code vere, altri lavori) non sono del lancio: si tolgono quelli gia'
        # visti un istante prima da questo processo (cwd /, nessuna pipeline con la radice)
        finti |= {c["pid"] for c in A.processi_in_conflitto(radice, dettagli=True) if c["pid"] is not None}
        r = subprocess.run(argv, capture_output=True, text=True, cwd="/")
        try:
            d = json.loads(r.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            ok(f"niente auto-blocco ({nome}): uscita JSON", False, (r.stdout + r.stderr)[-200:])
            continue
        estranei = [c for c in d["conflitti"] if c["pid"] not in finti and MARCA not in c["cmdline"]]
        ok(f"niente auto-blocco ({nome}): nessun processo del lancio stesso fra i conflitti", not estranei,
           "; ".join(f"{c['pid']} {c['comm']} ({c['motivi'][0]})" for c in estranei)[:200])
    # ── code pianificate: uscite di crontab/atq/at/systemctl SIMULATE (nessun timer creato davvero) ──
    coda_timer = script(fuori / "coda_timer_finta.sh", f"exit 0\ncd {radice} && ./forgia.sh\n")

    class Esito:
        def __init__(self, out="", rc=0, err=""):
            self.stdout, self.returncode, self.stderr = out, rc, err

    def finto_esegui(scenario):
        def esegui(argv, timeout=20.0):
            cmd = " ".join(argv)
            if argv[0] == "crontab":
                if scenario == "cron":
                    return Esito(f"# commento {radice}\nMAILTO=x\n30 3 * * * cd {radice} && ./notte.sh\n")
                if scenario == "cron_script":
                    return Esito(f"@reboot /bin/bash {coda_timer}\n")
                return Esito("", 1, "no crontab for angelo")
            if argv[0] == "atq":
                return Esito("7\tFri Sep 26 03:00:00 2026 a angelo\n" if scenario == "at" else "")
            if argv[:2] == ["at", "-c"]:
                return Esito(f"#!/bin/sh\numask 22\nPATH=/usr/bin; export PATH\ncd {radice} || {{ exit 1; }}\n./x.sh\n")
            if "list-units" in cmd:
                if scenario == "timer_rotto" and "--user" in argv:
                    return Esito("", 1, "Failed to connect to bus")
                return Esito("run-r1.timer loaded active waiting x\n" if scenario == "timer" and "--user" in argv else "")
            if "Triggers" in cmd:
                return Esito("Id=run-r1.timer\nTriggers=run-r1.service\n")
            if "ExecStart" in cmd:
                return Esito(f"Id=run-r1.service\nExecStart={{ path={coda_timer} ; argv[]={coda_timer} arg ; "
                             "ignore_errors=no ; start_time=[n/a] }\nWorkingDirectory=!/\nEnvironment=\n")
            return Esito("")
        return esegui
    R = A._Radice(radice)
    for scen, atteso in (("cron", "cron"), ("cron_script", "cron"), ("at", "at"), ("timer", "timer"),
                         ("timer_rotto", "timer"), ("niente", None)):
        pian = A.code_pianificate(R, esegui=finto_esegui(scen))
        tipi = {c["comm"] for c in pian}
        if atteso is None:
            ok("code pianificate (simulate): nessuna → nessun conflitto (commenti e 'no crontab' non contano)",
               not pian, str(pian)[:200])
        else:
            ok(f"code pianificate (simulate): {scen} → conflitto di tipo {atteso}", tipi == {atteso},
               "; ".join(c["motivi"][0] for c in pian)[:200])
    # lettura VERA, sola lettura, sulla radice vera: nessuna coda pianificata la usa oggi (e le fonti rispondono)
    t0 = time.time()
    veri_pian = A.code_pianificate(A._Radice(VERA))
    ok(f"code pianificate VERE (crontab, atq, timer utente e sistema) lette in {time.time() - t0:.1f}s: "
       f"{len(veri_pian)} conflitti", True, "; ".join(f"{c['comm']} {c['motivi'][0]}" for c in veri_pian)[:300])
    ferma_tutti()
    n = sum(v for _, v in ESITI)
    print(f"\n{'✅ TUTTI' if n == len(ESITI) else '⛔ FALLITI ' + str(len(ESITI) - n) + ' di'} {len(ESITI)} controlli")
    return 0 if n == len(ESITI) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        ferma_tutti()

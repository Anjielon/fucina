#!/usr/bin/env python3
"""Esegue prove GPU SOLO in una finestra libera e le interrompe (solo le proprie) se la finestra si chiude.

Finestra = la corsa prioritaria e' sospesa (stato T) o assente, ricostruzioni inattive, <=2 PID KFD prima
del lancio (<=4 durante la prova), smoke test separato con timeout 60 s prima di OGNI prova, le code
ammettono il lavoratore (cardan_code_gpu), MemAvailable >= --ram-gib e swap-in <= 20 MB/s, gpu_busy <= --busy-max.
Durante la prova: ogni 2 s; se la corsa torna R, compare un altro processo GPU, la RAM scende sotto 8 GiB o lo swap
supera 20 MB/s -> SIGTERM poi SIGKILL al SOLO gruppo della prova (start_new_session), esito "interrotta" nel registro.
Nessun segnale ad altri processi. Prove in sequenza da un file: una riga = nome<TAB>comando bash.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import cardan_code_gpu as CQ  # noqa: E402


def stato_pid(pid):
    try:
        return (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return None


def corsa_pids(nome):
    out = []
    for p in Path("/proc").iterdir():
        if p.name.isdigit():
            try:
                argv = [os.fsdecode(x) for x in (p / "cmdline").read_bytes().split(b"\0") if x]
                if not argv:
                    continue
                eseguibile = Path(argv[0]).name
                corrisponde = (eseguibile.startswith(nome) if nome == "llama-" else
                               eseguibile.startswith("python") and
                               any(Path(x).name == nome for x in argv[1:]))
                if corrisponde:
                    out.append(int(p.name))
            except OSError:
                pass
    return out


def ricostruzioni_pids(proc=Path("/proc")):
    out = []
    for p in proc.iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            argv = [os.fsdecode(x) for x in (p / "cmdline").read_bytes().split(b"\0") if x]
            cgroup = (p / "cgroup").read_text()
        except FileNotFoundError:
            continue
        python = bool(argv) and Path(argv[0]).name.startswith("python")
        if ("ricostruisci-" in cgroup or
                any(Path(x).name.startswith(("ricostruisci_", "ricostruisci-"))
                    and x.endswith((".py", ".sh")) for x in argv) or
                python and ("congiunto-s7t550-" in cgroup or
                            any(Path(x).name.startswith("cardan_congiunto") and x.endswith(".py")
                                for x in argv[1:]))):
            out.append(int(p.name))
    return out


def kfd_pids():
    return {int(p.name) for p in (CQ.KFD / "proc").iterdir() if p.name.isdigit()}


def sonda_gpu():
    """Contesto minimo separato, mai una deduzione da busy/code. Nessun retry dopo errore."""
    cmd = [sys.executable, "-c",
           "import torch;torch.ones(1).cuda();torch.cuda.synchronize()"]
    try:
        p = subprocess.run(cmd, timeout=60, capture_output=True, text=True,
                           env=CQ.ambiente_torch(dict(os.environ)))
    except subprocess.TimeoutExpired:
        return "AVVISO: smoke test GPU scaduto dopo 60 s; nessuna prova avviata"
    except OSError as e:
        return f"AVVISO: smoke test GPU non avviabile: {e}"
    if p.returncode:
        return f"AVVISO: smoke test GPU rc={p.returncode}: {p.stderr[-2000:]}"
    return None


def verifica_preavvio(a):
    motivo = finestra(a)
    if motivo:
        return motivo
    motivo = sonda_gpu()
    if motivo:
        return motivo
    return finestra(a)


def swap_s():
    def sw():
        return int(dict(r.split() for r in Path("/proc/vmstat").read_text().splitlines())["pswpin"])
    a = sw(); time.sleep(1); return (sw() - a) * 4096


def mem():
    return int([r for r in Path("/proc/meminfo").read_text().splitlines() if r.startswith("MemAvailable")][0].split()[1]) * 1024


def busy():
    v = [int(f.read_text()) for f in Path("/sys/class/drm").glob("card*/device/gpu_busy_percent")]
    return max(v) if v else 0


def finestra(a, propri=()):
    """None se libera, altrimenti motivo."""
    corse = [p for p in corsa_pids(a.corsa) if p not in propri]
    if any(stato_pid(p) not in ("T", "t", None) for p in corse):
        return "corsa prioritaria attiva"
    if any(stato_pid(p) not in ("T", "t", "Z", None)
           for p in ricostruzioni_pids() if p not in propri):
        return "corsa congiunto/ricostruzione prioritaria attiva"
    n_kfd = len(kfd_pids())
    if n_kfd > (4 if propri else 2):
        return f"AVVISO: processi KFD {n_kfd} > {4 if propri else 2}; nessun nuovo lancio"
    tot, per_pid = CQ.uso()
    estranei = [p for p in per_pid if p not in corse and p not in propri]
    if estranei:
        return f"altri processi GPU {estranei}"
    if any(stato_pid(p) not in ("T", "t", "Z", None) for p in corsa_pids("llama-")):
        return "llama Vulkan attivo"
    if mem() < (a.ram_gib if not propri else 8) * (1 << 30):
        return f"MemAvailable {mem() / (1 << 30):.1f} GiB"
    if swap_s() > 20e6:
        return "swap-in > 20 MB/s"
    if not propri:
        n = CQ.ammissibili(0, a.compute, a.sdma)
        if n is not None and n < 1:
            return "code GPU"
        if busy() > a.busy_max:
            return f"gpu_busy {busy()}"
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prove", required=True, help="file: nome<TAB>comando")
    ap.add_argument("--registro", required=True)
    ap.add_argument("--corsa", default="cardan_congiunto.py")
    ap.add_argument("--ram-gib", type=float, default=14)
    ap.add_argument("--compute", type=int, default=2)
    ap.add_argument("--sdma", type=int, default=0)
    ap.add_argument("--busy-max", type=int, default=10)
    ap.add_argument("--attesa-max-s", type=float, default=6 * 3600)
    a = ap.parse_args(argv)
    if a.attesa_max_s <= 0 or a.ram_gib < 8 or not 0 <= a.busy_max <= 100:
        ap.error("attesa positiva, RAM >= 8 GiB, busy tra 0 e 100")
    def log(**d):
        with open(a.registro, "a") as reg:
            reg.write(json.dumps({"ora": time.strftime("%F %T"), **d}) + "\n")
    prove = [r.split("\t", 1) for r in Path(a.prove).read_text().splitlines() if r.strip() and not r.startswith("#")]
    if not prove:
        raise ValueError("nessuna prova: non e' un collaudo riuscito")
    t0 = time.time()
    for nome, cmd in prove:
        for tentativo in range(1, 4):
            while True:
                m = finestra(a)
                if m is None:
                    break
                if m.startswith("AVVISO:"):
                    log(evento="AVVISO", prova=nome, motivo=m)
                    return 5
                if time.time() - t0 > a.attesa_max_s:
                    log(evento="rinuncia", prova=nome, motivo=m); return 2
                time.sleep(10)
            m = verifica_preavvio(a)
            if m is not None:
                log(evento="AVVISO", prova=nome, motivo=m)
                return 5
            log(evento="avvio", prova=nome, tentativo=tentativo, cmd=cmd)
            p = subprocess.Popen(["bash", "-c", cmd], start_new_session=True,
                                 env=CQ.ambiente_torch(dict(os.environ, HSA_ENABLE_SDMA="0")))
            ts = time.time(); motivo = None
            while p.poll() is None:
                time.sleep(2)
                if p.poll() is not None:
                    break
                gruppo = {int(x.name) for x in Path("/proc").iterdir() if x.name.isdigit() and _pgid(int(x.name)) == p.pid}
                motivo = finestra(a, propri=gruppo)
                if motivo:
                    try:
                        os.killpg(p.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    time.sleep(10)
                    if p.poll() is None:
                        try:
                            os.killpg(p.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    p.wait(); break
            log(evento="fine" if not motivo else "interrotta", prova=nome, rc=p.returncode,
                secondi=round(time.time() - ts, 1), motivo=motivo)
            if not motivo:
                if p.returncode:
                    return 1
                break
        else:
            log(evento="rinuncia", prova=nome, motivo="tre finestre interrotte; nessuna certificazione")
            return 4
    return 0


def _pgid(pid):
    try:
        return os.getpgid(pid)
    except OSError:
        return None


if __name__ == "__main__":
    sys.exit(main())

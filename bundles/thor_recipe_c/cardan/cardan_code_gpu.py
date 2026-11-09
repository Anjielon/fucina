"""Ammissione per CODE hardware GPU nella fucina (priorita' 0 di Angelo, 21/10 13:2x), sopra gpu_queues.py.

gpu_queues.py (di Angelo, non modificato) legge capacita' e code in uso del PRIMO nodo GPU KFD. Qui:
  * per device: ogni coda in /sys/class/kfd/kfd/proc/<pid>/queues/<q>/ ha `gpuid`; il nodo di topologia con lo
    stesso gpu_id da' la capacita' (num_cp_queues, motori SDMA x code). Device torch i = i-esimo nodo con
    simd_count > 0 in ordine di nodo (vale senza HIP_VISIBLE_DEVICES che rimappi; con rimappatura -> errore esplicito).
  * code PER TIPO DI LAVORATORE MISURATE, non supposte: `misura` lancia un comando e campiona le code del suo albero
    di processi (massimo nel tempo); `campiona` osserva in modo passivo tutti i processi su /dev/kfd con la loro riga
    di comando (nessun segnale, solo lettura di /sys e /proc). Le misure finiscono in `code_misurate_gpu.json`.
  * regola di contesa (c): un lavoratore in piu' entra solo se il rendimento complessivo misurato (unita'/minuto)
    cresce almeno di FORGIA_GUADAGNO_MINIMO (0,10). Il fattore (lavoratori ammessi per classe, rendimenti per N,
    sovrascrittura delle code di calcolo) e' salvato in `contesa_gpu_misurata.json`, letto dalla catena (`lavoratori`).
  * (d) TORCHINDUCTOR_COMPILE_THREADS=1: `ambiente_torch()` lo impone prima dell'import di torch.

KFD queue types: 0 compute, 1 SDMA, 2 SDMA xGMI. Niente segnali a processi trovati su /proc.
"""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
import time

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import gpu_queues as GQ  # noqa: E402

KFD = Path(os.environ.get("FORGIA_KFD", str(GQ.KFD)))  # FORGIA_KFD: albero finto per le prove
MISURATE = Path(os.environ.get("FORGIA_CODE_MISURATE", QUI / "code_misurate_gpu.json"))
CONTESA = Path(os.environ.get("FORGIA_CONTESA_GPU", QUI / "contesa_gpu_misurata.json"))
GUADAGNO_MINIMO = float(os.environ.get("FORGIA_GUADAGNO_MINIMO", "0.10"))


def ambiente_torch(env=None):
    """(d) Ogni lancio torch della fucina: un solo worker di compilazione inductor (ognuno apre /dev/kfd e prende
    una coda SDMA: 33 worker + congiunto + 5 pieghe = code SDMA esaurite, GPU bloccata il 21/10)."""
    env = os.environ if env is None else env
    env["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    return env


def nodi_gpu(kfd: Path = KFD):
    """[(gpu_id, Capacity)] dei nodi GPU in ordine di nodo."""
    out = []
    nodi = kfd / "topology" / "nodes"
    if not nodi.is_dir():
        return out
    for n in sorted(nodi.iterdir(), key=lambda p: int(p.name)):
        p = GQ._props(n)
        if p.get("simd_count", 0) > 0:
            per = p.get("num_sdma_queues_per_engine", 0)
            gid = int((n / "gpu_id").read_text().strip())
            out.append((gid, GQ.Capacity(compute=p.get("num_cp_queues", 0),
                                         sdma=(p.get("num_sdma_engines", 0) + p.get("num_sdma_xgmi_engines", 0)) * per)))
    return out


def gpu_id_di(device: int, kfd: Path = KFD):
    if os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get("ROCR_VISIBLE_DEVICES") \
            or os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("code GPU: *_VISIBLE_DEVICES impostato, indice torch -> nodo KFD non determinabile")
    nodi = nodi_gpu(kfd)
    if device >= len(nodi):
        raise LookupError(f"device {device}: solo {len(nodi)} nodi GPU KFD")
    return nodi[device]


def uso(kfd: Path = KFD, pids=None):
    """{gpu_id: {"compute": c, "sdma": s}}, {pid: {gpu_id: {...}}} ora (pids: solo questi)."""
    tot, per_pid = {}, {}
    proc = kfd / "proc"
    if not proc.exists():
        if (kfd / "topology").exists():
            raise RuntimeError(f"telemetria code: directory mancante {proc}")
        return tot, per_pid
    for d in proc.iterdir():
        if pids is not None and int(d.name) not in pids:
            continue
        try:
            code = list((d / "queues").iterdir())
        except FileNotFoundError as e:
            try:
                d.stat()
            except FileNotFoundError:
                continue
            raise RuntimeError(f"telemetria code incompleta: {d}") from e
        except OSError as e:
            raise RuntimeError(f"telemetria code non leggibile: {d}") from e
        for q in code:
            try:
                t = int((q / "type").read_text().strip()); g = int((q / "gpuid").read_text().strip())
            except FileNotFoundError as e:
                # Una coda puo' sparire fra elenco e lettura, non diventare illeggibile.
                try:
                    q.stat()
                except FileNotFoundError:
                    continue
                raise RuntimeError(f"telemetria code incompleta: {q}") from e
            except (OSError, ValueError) as e:
                raise RuntimeError(f"telemetria code non leggibile: {q}") from e
            k = "compute" if t == GQ.COMPUTE else "sdma" if t in (GQ.SDMA, GQ.SDMA_XGMI) else None
            if k is None or g <= 0:
                raise RuntimeError(f"telemetria code non valida: {q}, type={t}, gpuid={g}")
            for dd in (tot.setdefault(g, {"compute": 0, "sdma": 0}),
                       per_pid.setdefault(int(d.name), {}).setdefault(g, {"compute": 0, "sdma": 0})):
                dd[k] += 1
    return tot, per_pid


def fattore_compute():
    """Sovrascrittura delle code di calcolo ammessa: 1,0 salvo misura di rendimento salvata (regola c)."""
    try:
        return float(json.loads(CONTESA.read_text()).get("compute_oversubscription", 1.0))
    except (OSError, ValueError):
        return 1.0


def code_lavoratore(tipo: str):
    """Code misurate di un tipo di lavoratore {"compute": c, "sdma": s}; LookupError se mai misurate."""
    try:
        m = json.loads(MISURATE.read_text())["tipi"][tipo]
    except (OSError, ValueError, KeyError) as e:
        raise LookupError(f"code del lavoratore {tipo!r} NON misurate ({MISURATE})") from e
    if (m.get("misurato") is not True or type(m.get("campioni")) is not int or
            m["campioni"] < 1 or m.get("rc") != 0 or not m.get("cmd") or
            any(type(m.get(k)) is not int or m[k] < 0 for k in ("compute", "sdma", "processi_kfd"))):
        raise LookupError(f"code del lavoratore {tipo!r}: manca una misura reale valida ({MISURATE})")
    if m["compute"] + m["sdma"] and m["processi_kfd"] < 1:
        raise LookupError(f"code del lavoratore {tipo!r}: processi KFD non osservati")
    env = m.get("env", {})
    if (env.get("TORCHINDUCTOR_COMPILE_THREADS") != "1" or
            env.get("HSA_ENABLE_SDMA", "1") != os.environ.get("HSA_ENABLE_SDMA", "1")):
        raise LookupError(f"code del lavoratore {tipo!r}: ambiente SDMA/compilatori diverso dalla misura")
    return {"compute": m["compute"], "sdma": m["sdma"]}


def ammissibili(device: int, per_compute: int, per_sdma: int, sdma_riserva: int = 1, kfd: Path = KFD):
    """Quanti NUOVI lavoratori entrano sul device. None = nessun KFD (nessun limite noto)."""
    if not (kfd / "topology").is_dir():
        return None
    gid, cap = gpu_id_di(device, kfd)
    u = uso(kfd)[0].get(gid, {"compute": 0, "sdma": 0})
    return GQ.admissible(per_compute, per_sdma, sdma_reserve=sdma_riserva,
                         compute_oversubscription=fattore_compute(), cap=cap,
                         used=GQ.Usage(compute=u["compute"], sdma=u["sdma"], per_pid={}))


def stato(kfd: Path = KFD):
    """Per il registro: capacita' e uso per gpu_id, con la riga di comando dei processi."""
    tot, per_pid = uso(kfd)
    def riga(pid):
        try:
            return (Path("/proc") / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ")[:160].decode(errors="replace")
        except OSError:
            return "?"
    return {"capacita": {g: dataclasses.asdict(c) for g, c in nodi_gpu(kfd)}, "uso": tot,
            "processi": {p: {"code": v, "cmd": riga(p)} for p, v in per_pid.items()}}


# ---------------- regola di contesa (c) ----------------

def rendimento_cresce(r_n: float, r_n1: float, guadagno=GUADAGNO_MINIMO):
    """r = unita'/minuto COMPLESSIVE (tutti i processi sulla GPU) con N e N+1 lavoratori."""
    return r_n1 >= (1 + guadagno) * r_n


def lavoratori_ammessi(rendimenti: dict, guadagno=GUADAGNO_MINIMO):
    """rendimenti {N: unita'/min misurate}: il piu' grande N raggiunto aggiungendo uno alla volta finche' cresce >= 10 %."""
    r = {int(n): float(v) for n, v in rendimenti.items()}
    scelto = 1
    while scelto in r and scelto + 1 in r and rendimento_cresce(r[scelto], r[scelto + 1], guadagno):
        scelto += 1
    return scelto


def salva_contesa(classe: str, n: int, unita_min: float, fonte: str, path: Path = None):
    """Aggiorna (tmp + rename) il file dei fattori letto dalla catena."""
    path = path or CONTESA
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError):
        d = {"compute_oversubscription": 1.0, "classi": {}}
    c = d["classi"].setdefault(classe, {"rendimenti": {}, "fonti": {}})
    c["rendimenti"][str(n)] = unita_min
    c["fonti"][str(n)] = fonte
    c["ammessi"] = lavoratori_ammessi(c["rendimenti"])
    c["guadagno_minimo"] = GUADAGNO_MINIMO
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(d, indent=1, sort_keys=True)); os.replace(tmp, path)
    return c["ammessi"]


def lavoratori(classe: str, massimo: int, path: Path = None):
    """Per la catena: lavoratori da lanciare per la classe = min(massimo, ammessi misurati); 1 se mai misurata."""
    try:
        n = int(json.loads((path or CONTESA).read_text())["classi"][classe]["ammessi"])
    except (OSError, ValueError, KeyError):
        n = 1
    return max(1, min(massimo, n))


# ---------------- misure (b) ----------------

def _albero(pid):
    figli, coda = {pid}, [pid]
    while coda:
        p = coda.pop()
        try:
            for t in (Path("/proc") / str(p) / "task").iterdir():
                for c in (t / "children").read_text().split():
                    if int(c) not in figli:
                        figli.add(int(c)); coda.append(int(c))
        except OSError:
            pass
    return figli


def misura(tipo: str, cmd: list, passo_s=0.2, env=None, salva=True):
    """Lancia cmd, campiona le code del suo albero finche' termina: massimo per gpu (compute, sdma) e processi KFD."""
    from finestra_gpu import verifica_preavvio
    from types import SimpleNamespace
    a = SimpleNamespace(corsa="cardan_congiunto.py", ram_gib=14, compute=2, sdma=0, busy_max=10)
    motivo = verifica_preavvio(a)
    if motivo:
        raise RuntimeError(f"AVVISO: misura GPU non avviata: {motivo}")
    env = ambiente_torch(dict(os.environ if env is None else env))
    p = subprocess.Popen(cmd, env=env, start_new_session=True)
    massimo = {"compute": 0, "sdma": 0, "processi_kfd": 0}
    campioni = 0
    try:
        while p.poll() is None:
            alb = _albero(p.pid)
            _, per_pid = uso(pids=alb)
            campioni += 1
            c = sum(v["compute"] for g in per_pid.values() for v in g.values())
            s = sum(v["sdma"] for g in per_pid.values() for v in g.values())
            massimo = {"compute": max(massimo["compute"], c), "sdma": max(massimo["sdma"], s),
                       "processi_kfd": max(massimo["processi_kfd"], len(per_pid))}
            time.sleep(passo_s)
    finally:
        if p.poll() is None:
            import signal
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                p.wait()
    r = {**massimo, "rc": p.returncode, "cmd": cmd, "ora": time.strftime("%F %T"),
         "misurato": True, "campioni": campioni,
         "env": {k: v for k, v in (env or os.environ).items()
                 if k in ("HSA_ENABLE_SDMA", "TORCHINDUCTOR_COMPILE_THREADS")}}
    if p.returncode == 0 and campioni == 0:
        raise RuntimeError(f"misura code {tipo!r}: processo terminato senza alcun campione, profilo non salvato")
    if salva and p.returncode == 0:
        registra_tipo(tipo, r)
    return r


def registra_tipo(tipo, r, path: Path = None):
    path = path or MISURATE
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError):
        d = {"tipi": {}}
    d["tipi"][tipo] = r
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(d, indent=1, sort_keys=True)); os.replace(tmp, path)


def _altri():
    """Lavoro GPU fuori da KFD (llama.cpp Vulkan) e occupazione GPU/RAM, per le analisi di contesa."""
    llama = 0
    for p in Path("/proc").iterdir():
        if p.name.isdigit():
            try:
                if (p / "comm").read_text().startswith("llama-"):
                    llama += 1
            except OSError:
                pass
    busy = {}
    for f in Path("/sys/class/drm").glob("card*/device/gpu_busy_percent"):
        try:
            busy[f.parent.parent.name] = int(f.read_text())
        except (OSError, ValueError):
            pass
    m = {r.split(":")[0]: int(r.split()[1]) for r in Path("/proc/meminfo").read_text().splitlines()}
    sw = int(dict(r.split() for r in Path("/proc/vmstat").read_text().splitlines())["pswpin"])
    return {"llama_vulkan": llama, "gpu_busy": busy, "mem_available_kib": m["MemAvailable"], "pswpin": sw}


def campiona(secondi: float, passo: float, out: Path):
    """Osservazione passiva: una riga JSON per campione con code per processo e riga di comando."""
    fine = time.time() + secondi
    with open(out, "a", buffering=1) as f:
        while time.time() < fine:
            f.write(json.dumps({"ora": time.strftime("%F %T"), **stato(), **_altri()}) + "\n")
            time.sleep(passo)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("stato")
    a1 = sp.add_parser("ammissibili"); a1.add_argument("--device", type=int, default=0)
    a1.add_argument("--tipo"); a1.add_argument("--compute", type=int); a1.add_argument("--sdma", type=int)
    a1.add_argument("--riserva", type=int, default=1)
    a2 = sp.add_parser("misura"); a2.add_argument("tipo"); a2.add_argument("comando", nargs=argparse.REMAINDER)
    a3 = sp.add_parser("campiona"); a3.add_argument("--secondi", type=float, default=3600)
    a3.add_argument("--passo", type=float, default=5); a3.add_argument("--out", required=True)
    a4 = sp.add_parser("lavoratori"); a4.add_argument("classe"); a4.add_argument("--massimo", type=int, required=True)
    a5 = sp.add_parser("contesa"); a5.add_argument("classe"); a5.add_argument("n", type=int)
    a5.add_argument("unita_min", type=float); a5.add_argument("--fonte", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "stato":
        print(json.dumps(stato(), indent=1))
    elif a.cmd == "ammissibili":
        cw = code_lavoratore(a.tipo) if a.tipo else {"compute": a.compute, "sdma": a.sdma}
        n = ammissibili(a.device, cw["compute"], cw["sdma"], a.riserva)
        print("illimitato" if n is None else n)
    elif a.cmd == "misura":
        print(json.dumps(misura(a.tipo, a.comando)))
    elif a.cmd == "campiona":
        campiona(a.secondi, a.passo, Path(a.out))
    elif a.cmd == "lavoratori":
        print(lavoratori(a.classe, a.massimo))
    elif a.cmd == "contesa":
        print(salva_contesa(a.classe, a.n, a.unita_min, a.fonte))
    return 0


if __name__ == "__main__":
    sys.exit(main())

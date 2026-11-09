#!/usr/bin/env python3
"""Coda persistente GPU, senza fork di contesti HIP, con pubblicazioni atomiche.

Il modulo si importa senza torch. La CLI richiede budget espliciti o un profilo
misurato della stessa ricetta; auto non inventa picchi mancanti.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import fcntl
import hashlib
import importlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import re
import resource
import shutil
import signal
import sys
import time
import traceback
import uuid

GIB = 1 << 30
RISERVA_HOST = 8 * GIB
SWAP_MAX = 20_000_000


@dataclasses.dataclass(frozen=True)
class Unita:
    chiave: str
    dati: dict

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.chiave):
            raise ValueError(f"chiave non valida: {self.chiave!r}")


@dataclasses.dataclass(frozen=True)
class Budget:
    gpu: int
    rss: int
    fili: int
    disco: int
    margine: float = 1.25
    fili_io: int = 0

    def __post_init__(self):
        if any(type(x) is not int for x in (self.gpu, self.rss, self.fili, self.disco)):
            raise ValueError("picchi, fili e disco devono essere interi")
        if min(self.gpu, self.rss, self.fili, self.disco) <= 0:
            raise ValueError("picchi, fili e disco devono essere positivi")
        if not math.isfinite(self.margine) or self.margine < 1:
            raise ValueError("margine deve essere finito e >= 1")
        if type(self.fili_io) is not int or self.fili_io < 0:
            raise ValueError("fili_io deve essere un intero non negativo")

    @property
    def gpu_prenotata(self):
        return math.ceil(self.gpu * self.margine)

    def host_prenotato(self, uma):
        return math.ceil((self.rss + (self.gpu if uma else 0)) * self.margine)


@dataclasses.dataclass(frozen=True)
class Risorse:
    disponibile: int
    swap_letta_s: float
    fili: int
    gpu_libera: dict[int, int]


def leggi_memoria(path=Path("/proc/meminfo")):
    return {k: int(v.split()[0]) * 1024
            for r in path.read_text().splitlines() for k, v in [r.split(":", 1)]}


def pagine_swap(path=Path("/proc/vmstat")):
    valori = dict(r.split() for r in path.read_text().splitlines())
    return int(valori["pswpin"])


def misura_host(intervallo=1.0):
    t0, p0 = time.monotonic(), pagine_swap()
    time.sleep(intervallo)
    velocita = max(0, pagine_swap() - p0) * os.sysconf("SC_PAGE_SIZE") / (time.monotonic() - t0)
    return leggi_memoria()["MemAvailable"], velocita


def guardia_host(disponibile, swap):
    if disponibile < RISERVA_HOST or swap > SWAP_MAX:
        raise RuntimeError(f"AMMISSIONE NEGATA: MemAvailable={disponibile / GIB:.2f} GiB "
                           f"(minimo 8); swap-in={swap / 1e6:.2f} MB/s (massimo 20)")


def corse_protette(proc=Path("/proc")):
    trovati = []
    for p in proc.iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            argv = (p / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if not argv:
            continue
        try:
            cgroup = (p / "cgroup").read_text()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            cgroup = ""
        if any(nome in cgroup for nome in ("congiunto-s7t550-1020d", "tappa100-gpu",
                                           "s5-s7t550g5-1020b")):
            trovati.append(int(p.name))
            continue
        eseguibile = Path(os.fsdecode(argv[0])).name
        if not (eseguibile.startswith("python") or eseguibile.startswith("llama-")):
            continue
        args = [os.fsdecode(x) for x in argv[1:]]
        if eseguibile.startswith("python"):
            script = next((Path(x).name for x in args if x.endswith(".py")), "")
            if script not in {"cardan_congiunto.py", "cardan_kd_paper.py", "cardan_kd.py",
                              "cardan_piega_gpu.py", "cardan_piega.py",
                              "cardan_piega_densi_gpu.py", "cardan_piega_veloce.py"}:
                continue
        if any("/CARDAN_CONGIUNTO_S7T550_1020/" in x or "/S5_SU_S7T550G5_1020/" in x for x in args):
            trovati.append(int(p.name))
    return trovati


def guardia_corse():
    pids = corse_protette()
    if pids:
        raise RuntimeError(f"AMMISSIONE NEGATA: corse protette presenti, PID {pids}; "
                           "nessun segnale inviato, riprovare dopo la loro conclusione")


def seleziona_gpu(scelta, numero):
    if scelta == "all":
        ids = list(range(numero))
    else:
        ids = [int(x) for x in scelta.split(",")]
    if not ids or len(ids) != len(set(ids)) or any(x < 0 or x >= numero for x in ids):
        raise ValueError(f"--gpu {scelta!r}: indici visibili validi 0..{numero - 1}, senza duplicati")
    return ids


def tipo_device(torch, d):
    p = torch.cuda.get_device_properties(d)
    return f"{p.name}|{getattr(p, 'gcnArchName', '')}|torch={torch.__version__}|hip={torch.version.hip}"


def limite_code_non_misurate(device):
    """La prima misura sconosciuta non deve contendere code con altri processi."""
    import cardan_code_gpu as CQ
    if not (CQ.KFD / "topology").is_dir():
        return 1
    gid, cap = CQ.gpu_id_di(device)
    uso = CQ.uso()[0].get(gid, {"compute": 0, "sdma": 0})
    return int(cap.compute > 0 and cap.sdma > 1 and not any(uso.values()))


def pianifica(risorse, budget, processi="auto", uma=True, riserva_gpu=GIB, code=None):
    """code: {"compute": c, "sdma": s} MISURATE per lavoratore (cardan_code_gpu): terzo limite per device accanto a
    host/CPU/VRAM = lavoratori che entrano nelle code hardware libere ADESSO (SDMA mai oltre capacita' - riserva,
    calcolo oltre la capacita' solo col fattore misurato salvato). code None = nessun limite di code (host non KFD
    o chiamata storica); code "non_misurate" = 1 lavoratore per device (niente parallelismo su code ignote)."""
    if processi != "auto" and (not str(processi).isdigit() or int(processi) < 1):
        raise ValueError("--processi-per-gpu deve essere auto o un intero positivo")
    guardia_host(risorse.disponibile, risorse.swap_letta_s)
    limite_host = max(0, risorse.disponibile - RISERVA_HOST) // budget.host_prenotato(uma)
    limite_cpu = risorse.fili // (budget.fili + budget.fili_io)
    limiti = {d: max(0, n - riserva_gpu) // budget.gpu_prenotata
              for d, n in risorse.gpu_libera.items()}
    if code == "non_misurate":
        limiti = {d: min(n, limite_code_non_misurate(d)) for d, n in limiti.items()}
    elif code is not None:
        import cardan_code_gpu as CQ
        for d in limiti:
            n = CQ.ammissibili(d, code["compute"], code["sdma"])
            if n is not None:
                limiti[d] = min(limiti[d], n)
    if processi != "auto":
        limiti = {d: min(n, int(processi)) for d, n in limiti.items()}
    posti = {d: 0 for d in limiti}
    restano = min(limite_host, limite_cpu)
    while restano:
        prima = restano
        for d in posti:
            if posti[d] < limiti[d] and restano:
                posti[d] += 1
                restano -= 1
        if restano == prima:
            break
    if not sum(posti.values()):
        raise RuntimeError("AMMISSIONE NEGATA: nessun lavoratore entra nei budget host/device/CPU/code GPU")
    return posti


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def firma_json(dati):
    return hashlib.sha256(json.dumps(dati, sort_keys=True, allow_nan=False).encode()).hexdigest()


def identita_file(path):
    p = Path(path).resolve(strict=True)
    st = p.stat()
    return {"path": str(p), "byte": st.st_size, "mtime_ns": st.st_mtime_ns, "inode": st.st_ino}


def sincronizza_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def scrivi_json(path, dati):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.parziale")
    with open(tmp, "x") as f:
        json.dump(dati, f, sort_keys=True, indent=1, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    sincronizza_dir(path.parent)


def completa(path, firma, unita):
    path = Path(path)
    if not path.exists():
        return False
    m = json.loads((path / "completa.json").read_text())
    if m["firma"] != firma or m["unita"] != dataclasses.asdict(unita):
        raise RuntimeError(f"unita' di altra ricetta/input: {path}")
    effettivi = {str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()
                 and p.name != "completa.json"}
    if effettivi != set(m["file"]):
        raise RuntimeError(f"insieme file incompleto o estraneo: {path}")
    for nome, meta in m["file"].items():
        f = path / nome
        if f.stat().st_size != meta["byte"] or sha_file(f) != meta["sha256"]:
            raise RuntimeError(f"unita' corrotta: {f}; conservata, nessuna cancellazione")
    return True


def pubblica_unita(stage, dest, firma, unita, misure):
    stage, dest = Path(stage), Path(dest)
    if dest.exists():
        raise FileExistsError(f"non sovrascrivo {dest}")
    files = {}
    for p in sorted(stage.rglob("*")):
        if p.is_symlink():
            raise ValueError(f"symlink non ammesso nel risultato: {p}")
        if p.is_file():
            with open(p, "rb") as f:
                os.fsync(f.fileno())
            files[str(p.relative_to(stage))] = {"byte": p.stat().st_size, "sha256": sha_file(p)}
    if not files:
        raise RuntimeError(f"lavoratore senza risultati: {unita.chiave}")
    scrivi_json(stage / "completa.json", {"firma": firma, "unita": dataclasses.asdict(unita),
                                         "file": files, "misure": misure})
    for d in sorted((p for p in stage.rglob("*") if p.is_dir()), reverse=True):
        sincronizza_dir(d)
    sincronizza_dir(stage)
    os.rename(stage, dest)
    sincronizza_dir(dest.parent)


def code_senza_kfd(code, kfd):
    """D19: on a host without a KFD topology the hardware-queue limit does not exist; FORGIA_CODE_SENZA_KFD=nessun_limite
    returns None (no queue limit) there. Any other value, or a KFD host, returns `code` unchanged."""
    modo = os.environ.get("FORGIA_CODE_SENZA_KFD", "")
    if modo not in ("", "nessun_limite"):
        raise ValueError("FORGIA_CODE_SENZA_KFD must be empty or nessun_limite")
    if modo == "nessun_limite" and not (Path(kfd) / "topology").is_dir():
        return None
    return code


def picco_oltre_budget(misure, budget):
    """Post-unit peak check. Default (FORGIA_PICCO_CONTRO unset/"profilo"): the raw profile peak, as before.
    "margine" (D37, 22/10): the RESERVED amount ceil(peak * margine) that admission actually set aside; a re-run that
    exceeds the profiled peak by RSS jitter (+0.3 % measured on 2xT4) no longer drains the queue."""
    modo = os.environ.get("FORGIA_PICCO_CONTRO", "profilo")
    if modo not in ("profilo", "margine"):
        raise ValueError("FORGIA_PICCO_CONTRO must be profilo or margine")
    f = budget.margine if modo == "margine" else 1.0
    return (misure["picco_rss_byte"] > math.ceil(budget.rss * f)
            or misure.get("picco_gpu_byte", 0) > math.ceil(budget.gpu * f))


def risolvi_factory(nome):
    modulo, simbolo = nome.split(":", 1)
    return getattr(importlib.import_module(modulo), simbolo)


def _lavoratore(factory, config, device, root, firma, ingresso, uscita, fili, padre):
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "impossibile impostare PR_SET_PDEATHSIG")
    if os.getppid() != padre:
        raise RuntimeError("coordinatore gia' terminato")
    os.nice(max(0, 19 - os.getpriority(os.PRIO_PROCESS, 0)))
    os.environ.update(OMP_NUM_THREADS=str(fili), MKL_NUM_THREADS=str(fili), LEVE_THREADS=str(fili),
                      PYTHONDONTWRITEBYTECODE="1")
    pid = os.getpid()
    log = Path(root) / f"lavoratore-{device}-{pid}.log"
    with open(log, "x", buffering=1) as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        try:
            for p, sha in config.get("codici", {}).items():
                if sha_file(p) != sha:
                    raise RuntimeError(f"codice cambiato prima dell'avvio: {p}")
            handler = risolvi_factory(factory)(config, device)
            uscita.put(("pronto", pid, None))
            while True:
                unita = ingresso.get()
                if unita is None:
                    return
                stage = Path(root) / f".lavoro-{unita.chiave}-{uuid.uuid4().hex}"
                stage.mkdir()
                t0 = time.monotonic()
                metriche = handler(unita.dati, stage)
                misure = {"secondi": time.monotonic() - t0,
                           "picco_rss_byte": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                           **metriche}
                pubblica_unita(stage, Path(root) / "unita" / unita.chiave, firma, unita, misure)
                uscita.put(("fatta", pid, {"chiave": unita.chiave, **misure}))
        except BaseException:
            errore = traceback.format_exc()
            print(errore, flush=True)
            uscita.put(("errore", pid, errore))
            raise


@contextlib.contextmanager
def lucchetto(path):
    with open(path, "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError(f"altro pianificatore attivo: {path}") from e
        yield


def _esegui_prima_obiettivo(unita, factory, config, root, firma, posti, budget):
    """Un coordinatore, un gruppo per device, una sola unita' in volo per worker."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    chiavi = [u.chiave for u in unita]
    if len(set(chiavi)) != len(chiavi):
        raise ValueError("unita' duplicate")
    (root / "unita").mkdir(exist_ok=True)
    manifeste = root / "piano.json"
    piano = {"firma": firma, "unita": [dataclasses.asdict(u) for u in unita],
             "factory": factory, "config": config}
    if manifeste.exists():
        if json.loads(manifeste.read_text()) != piano:
            raise RuntimeError("ripresa con piano/config/input diversi: scegliere una nuova --out")
    else:
        if any((root / "unita").iterdir()):
            raise RuntimeError("unita' presenti senza piano: non le adotto")
        scrivi_json(manifeste, piano)
    pendenti = [u for u in unita if not completa(root / "unita" / u.chiave, firma, u)]
    if not pendenti:
        return
    if shutil.disk_usage(root).free < sum(budget.disco for _ in pendenti) + GIB:
        raise RuntimeError("spazio insufficiente per risultati + 1 GiB di riserva")
    ctx = mp.get_context("spawn")
    uscita = ctx.Queue()
    figli, code, attivi = {}, {}, {}
    iterator = iter(pendenti)
    esaurita = False
    errore = None
    morti_visti = set()
    try:
        # Avvii seriali: non sovrapporre picchi di import/caricamento dei modelli.
        for device, n in posti.items():
            for _ in range(min(n, len(pendenti) - len(figli))):
                guardia_corse()
                guardia_host(*misura_host())
                q = ctx.Queue()
                p = ctx.Process(target=_lavoratore,
                                args=(factory, config, device, str(root), firma, q, uscita,
                                      budget.fili, os.getpid()))
                p.start()
                figli[p.pid], code[p.pid] = p, q
                while True:
                    try:
                        stato, pid, msg = uscita.get(timeout=1)
                        if stato != "pronto" or pid != p.pid:
                            raise RuntimeError(f"avvio lavoratore fallito: {msg}")
                        break
                    except queue.Empty:
                        if p.exitcode is not None:
                            raise RuntimeError(f"lavoratore {p.pid} morto all'avvio ({p.exitcode})")
        while True:
            if errore is None:
                for pid in figli:
                    if pid in attivi or esaurita:
                        continue
                    try:
                        guardia_corse()
                        guardia_host(*misura_host())
                    except RuntimeError as e:
                        errore = str(e)
                        print(f"{errore}; nessuna nuova assegnazione, completo le unita' in volo",
                              file=sys.stderr, flush=True)
                        break
                    u = next(iterator, None)
                    if u is None:
                        esaurita = True
                        break
                    attivi[pid] = u
                    code[pid].put(u)
            if not attivi:
                if errore is not None:
                    raise RuntimeError(errore)
                break
            try:
                stato, pid, msg = uscita.get(timeout=1)
            except queue.Empty:
                morti = [(pid, p.exitcode) for pid, p in figli.items()
                         if p.exitcode is not None and pid not in morti_visti]
                if morti:
                    errore = f"lavoratori terminati prima del completamento: {morti}"
                    for pid, _ in morti:
                        morti_visti.add(pid)
                        attivi.pop(pid, None)
                continue
            if stato != "fatta":
                errore = f"lavoratore {pid}: {msg}"
                attivi.pop(pid, None)
                continue
            if pid not in attivi or msg["chiave"] != attivi[pid].chiave:
                raise RuntimeError("risultato senza assegnazione coerente")
            print(json.dumps({"pid": pid, **msg}), flush=True)
            del attivi[pid]
            if picco_oltre_budget(msg, budget):
                errore = ("picco misurato oltre budget: nessuna nuova unita', "
                          "rivedere il profilo; completo le unita' gia' in volo")
    finally:
        # Solo figli creati qui; MAI segnali a processi trovati su /proc.
        for pid, p in figli.items():
            if p.is_alive():
                code[pid].put(None)
        for p in figli.values():
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        for q in code.values():
            q.close()
        uscita.close()


def esegui(unita, factory, config, root, firma, posti, budget):
    from pianificatore_misurato_gpu import esegui as misurato
    return misurato(unita, factory, config, root, firma, posti, budget)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--piano", required=True, help="JSON: factory, config, unita, firma")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", default=os.environ.get("FORGIA_GPU", "all"))
    ap.add_argument("--processi-per-gpu", default=os.environ.get("FORGIA_PROCESSI_PER_GPU", "auto"))
    ap.add_argument("--profilo", help="JSON misurato: ricetta_sha256, gpu_byte, rss_byte, fili, disco_byte")
    ap.add_argument("--rendimenti", help="JSON misurato: ricetta_sha256, dispositivi, tempi, pesi/lavori/classi")
    ap.add_argument("--calibra-compresenza", action="store_true",
                    help="solo prova esplicita su 1-2 unita'; vietata con corse protette")
    ap.add_argument("--attesa-massima-s", type=float, default=0,
                    help="attesa pressione risorse: 0 illimitata, altrimenti timeout esplicito")
    ap.add_argument("--priorita", type=float, default=1, help="peso del completamento dell'intera fase")
    ap.add_argument("--budget-gpu-gib", type=float)
    ap.add_argument("--budget-rss-gib", type=float)
    ap.add_argument("--fili", type=int, default=4)
    ap.add_argument("--disco-unita-gib", type=float, default=1)
    ap.add_argument("--discreta", action="store_true", help="non sommare budget device al consumo host UMA")
    ap.add_argument("--solo-piano", action="store_true")
    a = ap.parse_args(argv)
    if not math.isfinite(a.priorita) or a.priorita <= 0:
        ap.error("--priorita deve essere positiva e finita")
    piano = json.loads(Path(a.piano).read_text())
    if a.calibra_compresenza and (not a.profilo or a.processi_per_gpu == "auto" or
                                 len(piano["unita"]) > 2):
        ap.error("calibrazione compresenza: profilo misurato, N esplicito e massimo 2 unita'")
    if not math.isfinite(a.attesa_massima_s) or a.attesa_massima_s < 0:
        ap.error("--attesa-massima-s deve essere finita e >= 0")
    if a.profilo:
        profilo = json.loads(Path(a.profilo).read_text())
        if profilo["ricetta_sha256"] != piano["ricetta_sha256"]:
            ap.error("profilo di altra ricetta/dtype/lotto/codice")
        budget = Budget(profilo["gpu_byte"], profilo["rss_byte"], profilo["fili"], profilo["disco_byte"])
    else:
        if a.processi_per_gpu == "auto" or a.budget_gpu_gib is None or a.budget_rss_gib is None:
            ap.error("auto richiede --profilo misurato; per calibrare: processi-per-gpu 1 e budget espliciti")
        if a.processi_per_gpu != "1":
            ap.error("senza profilo misurato e' ammesso solo 1 processo per GPU per calibrazione")
        budget = Budget(math.ceil(a.budget_gpu_gib * GIB), math.ceil(a.budget_rss_gib * GIB),
                        a.fili, math.ceil(a.disco_unita_gib * GIB))
    budget = dataclasses.replace(budget, fili_io=piano["config"].get("fili_io", 0))
    guardia_corse()
    disponibili, swap = misura_host()
    guardia_host(disponibili, swap)
    import cardan_code_gpu as CQ
    CQ.ambiente_torch()  # (d) un solo worker di compilazione inductor, anche nei lavoratori (ereditano l'ambiente)
    # Code hardware per lavoratore: dal profilo misurato o dalla tabella delle misure per tipo; mai supposte.
    if a.profilo and "code_compute" in profilo:
        code = {"compute": int(profilo["code_compute"]), "sdma": int(profilo["code_sdma"])}
    elif piano["config"].get("tipo_lavoratore"):
        try:
            code = CQ.code_lavoratore(piano["config"]["tipo_lavoratore"])
        except LookupError:
            code = "non_misurate"
    else:
        code = "non_misurate"
    # D19 (22/10): KFD hardware queues exist only on the AMD driver. On a host WITHOUT a KFD topology (e.g. NVIDIA) the
    # queue limit has no meaning; FORGIA_CODE_SENZA_KFD=nessun_limite drops it there (host/VRAM/CPU budgets and the
    # measured contention rules still apply). Default unchanged: 1 worker per device when queues are unknown.
    code = code_senza_kfd(code, CQ.KFD)
    import torch
    ids = seleziona_gpu(a.gpu, torch.cuda.device_count())
    if budget.fili != piano["config"].get("fili", budget.fili):
        ap.error("numero di fili diverso dalla ricetta del piano")
    if a.profilo and any(tipo_device(torch, d) not in profilo["dispositivi"] for d in ids):
        ap.error("profilo non misurato su questo tipo di GPU / versione torch-HIP")
    obiettivo = {}
    if a.rendimenti:
        obiettivo = json.loads(Path(a.rendimenti).read_text())
        if (obiettivo["ricetta_sha256"] != piano["ricetta_sha256"] or
                any(tipo_device(torch, d) not in obiettivo["dispositivi"] or
                    obiettivo.get("device_firme", {}).get(str(d)) != tipo_device(torch, d) for d in ids)):
            ap.error("rendimenti non misurati per questa ricetta/device")
    from cardan_classi_gpu import unisci
    obiettivo = {**unisci(piano["config"], obiettivo),
                 "priorita": a.priorita, "uma": not a.discreta, "gpu_reale": True,
                 "calibra_compresenza": a.calibra_compresenza,
                 "attesa_massima_s": a.attesa_massima_s}
    if "lavori" not in obiettivo:
        obiettivo["lavori"] = {u["chiave"]: piano["config"].get("fase", "fase")
                              for u in piano["unita"]}
    obiettivo["code_lavoratore"] = code
    piano["config"] = {**piano["config"], "pianificazione": obiettivo}
    risorse = Risorse(disponibili, swap, len(os.sched_getaffinity(0)),
                      {d: torch.cuda.mem_get_info(d)[0] for d in ids})
    posti = pianifica(risorse, budget, a.processi_per_gpu, not a.discreta, code=code)
    if a.calibra_compresenza:
        richiesti = min(len(piano["unita"]), len(ids) * int(a.processi_per_gpu))
        if sum(posti.values()) < richiesti:
            ap.error("la compresenza richiesta non entra nei budget; nessuna prova ridotta implicitamente")
    print(json.dumps({"posti": posti, "risorse": dataclasses.asdict(risorse),
                      "budget": dataclasses.asdict(budget), "code_lavoratore": code,
                      "code_gpu": CQ.stato()}), flush=True)
    if a.solo_piano:
        return 0
    root = Path(a.out).resolve()
    def interrotto(signum, frame):
        raise KeyboardInterrupt(f"segnale {signum}, arresto soltanto i miei figli")
    signal.signal(signal.SIGTERM, interrotto)
    # Un solo coordinatore anche fra --out differenti sulla stessa macchina.
    with lucchetto(Path(__file__).with_name(".pianificatore_gpu.lock")):
        esegui([Unita(**u) for u in piano["unita"]], piano["factory"], piano["config"],
               root, piano["firma"], posti, budget)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError, OSError) as e:
        print(f"ERRORE pianificatore: {e}", file=sys.stderr, flush=True)
        sys.exit(2)

"""Campioni continui host/sysfs; nessun contesto HIP o processo esterno."""
from __future__ import annotations

import os
from pathlib import Path
import time
import datetime
import json
import re

import pianificatore_gpu as PG


def progresso_log(path):
    """Coda limitata; delta wall include held-out, salvataggi e attese."""
    p = Path(path)
    with open(p, "rb") as f:
        f.seek(max(0, os.fstat(f.fileno()).st_size - 65536))
        righe = f.read(65536).decode("utf-8", errors="replace").splitlines()
    passi = []
    for riga in righe:
        m = re.match(r"^\[([0-9T:.-]+)\] (\{.*\})$", riga)
        if not m:
            continue
        try:
            record = json.loads(m[2])
            ora = datetime.datetime.fromisoformat(m[1]).timestamp()
        except (json.JSONDecodeError, ValueError):
            continue
        if type(record.get("passo")) is int:
            passi.append((ora, record))
    if not passi:
        return {"registro": str(p), "passo": None, "motivo": "nessun passo completo nella coda 64 KiB"}
    ora, ultimo = passi[-1]
    velocita = None
    if len(passi) >= 2:
        dt, dn = ora - passi[-2][0], ultimo["passo"] - passi[-2][1]["passo"]
        if dt > 0 and dn > 0:
            velocita = dn / dt
    return {"registro": str(p), "passo": ultimo["passo"], "eta_ultimo_passo_s": max(0, time.time() - ora),
            "passi_s_wall": velocita, "secondi_unita_dichiarati": ultimo.get("s")}


def osserva_corse(pids, proc):
    records = []
    for pid in pids:
        rec = {"pid": pid, "progressi": []}
        try:
            args = [os.fsdecode(x) for x in (proc / str(pid) / "cmdline").read_bytes().split(b"\0") if x]
            stato = (proc / str(pid) / "stat").read_text().split(") ", 1)[1].split()[0]
            rec["stato_proc"] = stato
            outs = [Path(args[i + 1]) for i, x in enumerate(args[:-1]) if x in ("--uscita", "--out")]
            for out in outs:
                path = out / "registro.log"
                if path.is_file():
                    rec["progressi"].append(progresso_log(path))
            if not rec["progressi"]:
                rec["motivo"] = "registro passi non individuato; nessuna stima inventata"
        except (FileNotFoundError, ProcessLookupError, PermissionError, OSError) as e:
            rec["errore_osservazione"] = str(e)
        records.append(rec)
    return records


class Telemetria:
    def __init__(self, proc=Path("/proc"), drm=Path("/sys/class/drm")):
        self.proc, self.drm = proc, drm
        self.precedente = self._contatori()
        t, _, totale, idle = self.precedente
        self.precedente_cpu = t, totale, idle
        self.frazione_cpu = None
        self.intervallo_cpu = 0

    def _contatori(self):
        # guest/guest_nice sono gia' inclusi in user/nice.
        cpu = [int(x) for x in (self.proc / "stat").read_text().splitlines()[0].split()[1:9]]
        return time.monotonic(), PG.pagine_swap(self.proc / "vmstat"), sum(cpu), cpu[3] + cpu[4]

    def campiona(self):
        t, sw, totale, idle = self._contatori()
        pt, ps, pc, pi = self.precedente
        self.precedente = t, sw, totale, idle
        dt = t - pt
        ct, cc, ci = self.precedente_cpu
        if t - ct >= 1 or self.frazione_cpu is None:
            dc = totale - cc
            self.frazione_cpu = min(1.0, max(0.0, (idle - ci) / dc)) if dc > 0 else 0.0
            self.intervallo_cpu = t - ct
            self.precedente_cpu = t, totale, idle
        frazione = self.frazione_cpu
        gpu = []
        for card in sorted(self.drm.glob("card[0-9]*")):
            if not card.name[4:].isdigit():
                continue
            dev = card / "device"
            record = {"sysfs": str(dev.resolve()), "card": card.name}
            for file in ("gpu_busy_percent", "mem_info_vram_used", "mem_info_vram_total",
                         "mem_info_gtt_used", "mem_info_gtt_total"):
                p = dev / file
                try:
                    record[file] = int(p.read_text())
                except (FileNotFoundError, PermissionError, OSError) as e:
                    record[file] = None
                    record[file + "_errore"] = str(e)
            gpu.append(record)
        cpu_totali = len(os.sched_getaffinity(0))
        runnable = int((self.proc / "loadavg").read_text().split()[3].split("/")[0])
        protetti = PG.corse_protette(self.proc)
        return {"ora": time.time(), "intervallo_s": dt,
                "mem_available": PG.leggi_memoria(self.proc / "meminfo")["MemAvailable"],
                "swap_in_byte_s": max(0, sw - ps) * os.sysconf("SC_PAGE_SIZE") / max(dt, 1e-9),
                "cpu_occupati_stima": cpu_totali * (1 - frazione),
                "cpu_liberi_stima": cpu_totali * frazione, "cpu_affinity": cpu_totali,
                "cpu_intervallo_s": self.intervallo_cpu,
                "fili_runnable_host": max(0, runnable - 1),
                "gpu": gpu, "corse_protette_pid": protetti,
                "corse_protette": osserva_corse(protetti, self.proc)}

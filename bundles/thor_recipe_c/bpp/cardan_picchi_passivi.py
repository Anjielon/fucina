#!/usr/bin/env python3
"""Passive physical-peak recorder for the forge workers (D3/D10/D16 closure, 22/10).

Read-only: /proc/<pid>/status, /proc/<pid>/cmdline, /sys/class/kfd/kfd/proc/<pid>/vram_*, /proc/meminfo, /proc/vmstat,
/sys/class/drm/card*/device/{gpu_busy_percent,mem_info_vram_used}. It never signals, attaches to or slows a process,
never opens /dev/kfd and imports nothing heavy. Every --passo seconds it appends one JSON line per sample to --out:
  {"t", "mem_avail", "swapin_Bps", "gpu_busy", "vram_used", "kfd": n, "proc": [{"pid","classe","rss","hwm","swap","vram"}]}
and, when a watched process disappears, one {"fine": pid, "classe", "hwm_ultimo", "vram_max", "durata_s"} line.

  summary: python3 cardan_picchi_passivi.py --riassunto FILE.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import time

CLASSI = (("piega_esperti_lotti", "cardan_piega_lotti.py"), ("piega_densi", "cardan_piega_densi_gpu"),
          ("piega_esperti_gpu", "cardan_piega_gpu"), ("trainer_congiunto", "cardan_congiunto"),
          ("audit_energie", "cardan_verifica_energie_gpu"), ("mmlu_vulkan", "llama-perplexity"),
          ("compile_worker", "compile_worker"), ("patch", "patch_gguf.py"))


def classe(cmd):
    for nome, pat in CLASSI:
        if pat in cmd:
            return nome
    return None


def status(pid):
    d = {}
    try:
        for line in open(f"/proc/{pid}/status"):
            k, _, v = line.partition(":")
            if k in ("VmRSS", "VmHWM", "VmSwap", "RssAnon", "RssFile", "RssShmem"):
                d[k] = int(v.split()[0]) * 1024
    except OSError:
        return None
    return d


def vram(pid):
    try:
        return max(int(Path(f).read_text()) for f in glob.glob(f"/sys/class/kfd/kfd/proc/{pid}/vram_*"))
    except (OSError, ValueError):
        return None


def leggi(path, default=None):
    try:
        return int(Path(path).read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return default


def pswpin():
    for line in open("/proc/vmstat"):
        if line.startswith("pswpin "):
            return int(line.split()[1])
    return 0


def memavail():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return None


def registra(out, passo, durata, stop):
    card = sorted(glob.glob("/sys/class/drm/card*/device/gpu_busy_percent"))
    card = os.path.dirname(card[0]) if card else None
    vivi, t_end = {}, time.time() + durata
    s0, t0 = pswpin(), time.time()
    with open(out, "a") as f:
        while time.time() < t_end and not os.path.exists(stop):
            time.sleep(passo)
            s1, t1 = pswpin(), time.time()
            campione = {"t": round(t1, 1), "mem_avail": memavail(), "swapin_Bps": (s1 - s0) * 4096 / max(t1 - t0, 1e-3),
                        "gpu_busy": leggi(f"{card}/gpu_busy_percent") if card else None,
                        "vram_used": leggi(f"{card}/mem_info_vram_used") if card else None,
                        "kfd": len(os.listdir("/sys/class/kfd/kfd/proc")), "proc": []}
            s0, t0 = s1, t1
            visti = set()
            for p in os.listdir("/proc"):
                if not p.isdigit():
                    continue
                try:
                    cmd = Path(f"/proc/{p}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
                except OSError:
                    continue
                c = classe(cmd)
                if c is None or "cardan_picchi_passivi" in cmd:
                    continue
                st = status(p)
                if st is None:
                    continue
                v = vram(p)
                if c == "compile_worker" and v is None:
                    continue                                   # 32 inductor workers without a KFD context: noise
                pid = int(p)
                visti.add(pid)
                info = vivi.setdefault(pid, {"classe": c, "inizio": t1, "vram_max": 0, "hwm": 0, "cmd": cmd[:400]})
                info["vram_max"] = max(info["vram_max"], v or 0)
                info["hwm"] = max(info["hwm"], st.get("VmHWM", 0))
                info["anon_max"] = max(info.get("anon_max", 0), st.get("RssAnon", 0))
                campione["proc"].append({"pid": pid, "classe": c, "rss": st.get("VmRSS"), "hwm": st.get("VmHWM"),
                                         "anon": st.get("RssAnon"), "file": st.get("RssFile"),
                                         "swap": st.get("VmSwap"), "vram": v})
            f.write(json.dumps(campione) + "\n")
            for pid in [p for p in vivi if p not in visti]:
                info = vivi.pop(pid)
                f.write(json.dumps({"fine": pid, "classe": info["classe"], "hwm_ultimo": info["hwm"], "anon_max": info.get("anon_max", 0),
                                    "vram_max": info["vram_max"], "durata_s": round(t1 - info["inizio"], 1),
                                    "cmd": info["cmd"]}) + "\n")
            f.flush()


def riassunto(path):
    per = {}
    conc = {}
    for line in open(path):
        r = json.loads(line)
        if "fine" in r:
            c = per.setdefault(r["classe"], {"processi": 0, "hwm_max": 0, "vram_max": 0, "durate": []})
            c["processi"] += 1
            c["hwm_max"] = max(c["hwm_max"], r["hwm_ultimo"])
            c["anon_max"] = max(c.get("anon_max", 0), r.get("anon_max", 0))
            c["vram_max"] = max(c["vram_max"], r["vram_max"])
            c["durate"].append(r["durata_s"])
        elif "proc" in r:
            k = tuple(sorted(x["classe"] for x in r["proc"] if x["classe"] != "compile_worker"))
            e = conc.setdefault(" + ".join(k) or "-", {"campioni": 0, "rss_somma_max": 0, "swapin_max": 0,
                                                       "mem_avail_min": None, "gpu_busy": []})
            e["campioni"] += 1
            e["rss_somma_max"] = max(e["rss_somma_max"], sum(x["rss"] or 0 for x in r["proc"]))
            e["swapin_max"] = max(e["swapin_max"], r["swapin_Bps"])
            if r["mem_avail"] is not None:
                e["mem_avail_min"] = r["mem_avail"] if e["mem_avail_min"] is None else min(e["mem_avail_min"], r["mem_avail"])
            if r["gpu_busy"] is not None:
                e["gpu_busy"].append(r["gpu_busy"])
    for e in conc.values():
        g = e.pop("gpu_busy")
        e["gpu_busy_medio"] = round(sum(g) / len(g), 1) if g else None
    return {"per_classe": per, "per_compresenza": conc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--passo", type=float, default=5.0)
    ap.add_argument("--durata-h", type=float, default=48.0)
    ap.add_argument("--stop", default="")
    ap.add_argument("--riassunto")
    a = ap.parse_args()
    if a.riassunto:
        print(json.dumps(riassunto(a.riassunto), indent=1))
        return
    registra(a.out, a.passo, a.durata_h * 3600, a.stop or a.out + ".stop")


if __name__ == "__main__":
    main()

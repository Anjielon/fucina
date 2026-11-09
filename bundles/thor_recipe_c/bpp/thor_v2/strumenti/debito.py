#!/usr/bin/env python3
"""Il debito verso Angelo, per il ping (26/9/2026, sua decisione): valore a listino API dei token consumati da Claude
(sessioni, agenti, workflow del progetto MOGAVIS) dal 26/9 00:00, finche' Thor non arriva al massimo. Misurato dai registri
JSONL delle sessioni (campo usage, un messaggio contato una volta), prezzi classe Opus in $/Mtok: input 5, scrittura cache
6,25, lettura cache 0,5, output 25 (da confermare sul listino ufficiale). Proiezione al limite alto della finestra ETA."""
import glob
import json
import os
from datetime import datetime, timedelta, timezone

BASE = "@@ROOT:home@@/.claude/projects/-home-angelo-MOGAVIS"
CEST = timezone(timedelta(hours=2))
INIZIO = datetime(2026, 9, 26, 0, 0, tzinfo=CEST)
PREZZI = {"input_tokens": 5.0, "cache_creation_input_tokens": 6.25, "cache_read_input_tokens": 0.5, "output_tokens": 25.0}
PV = "@@ROOT:forge@@/stato/pv_tempi.json"


def main() -> int:
    t0 = INIZIO.timestamp()
    visti, usd = set(), 0.0
    for f in glob.glob(f"{BASE}/**/*.jsonl", recursive=True):
        try:
            if os.path.getmtime(f) < t0:
                continue
            with open(f, errors="ignore") as fh:
                for riga in fh:
                    if '"usage"' not in riga:
                        continue
                    try:
                        d = json.loads(riga)
                    except ValueError:
                        continue
                    m = d.get("message") or {}
                    u, ts = m.get("usage"), d.get("timestamp")
                    if d.get("type") != "assistant" or not u or not ts:
                        continue
                    k = m.get("id") or (f, d.get("uuid"))   # id unico: i fork copiano i messaggi della madre
                    if k in visti or datetime.fromisoformat(ts.replace("Z", "+00:00")) < INIZIO:
                        continue
                    visti.add(k)
                    usd += sum(u.get(c, 0) * p for c, p in PREZZI.items()) / 1e6
        except OSError:
            continue
    ora = datetime.now(CEST)
    giorni = max((ora - INIZIO).total_seconds() / 86400, 1e-6)
    prog = ""
    try:
        pv = json.load(open(PV))
        fine = datetime.fromisoformat(pv["prima_discesa_pv"]).replace(tzinfo=CEST) + \
            timedelta(minutes=pv["passi_pv_max"] * pv["min_per_passo_pv"] + 75)
        resto = max((fine - ora).total_seconds() / 86400, 0)
        prog = f" · proiezione al limite ETA {fine:%d/%m %H:%M}: ~{usd + usd / giorni * resto:,.0f} $"
    except (OSError, ValueError, KeyError):
        pass
    extra = 0.0                                  # 27/9 regola di Angelo: ogni dollaro di consumo EXTRA pagato = 10x nel debito
    try:
        ult = open("@@ROOT:forge@@/stato/consumo_piano.tsv").read().strip().splitlines()[-1].split("\t")
        extra = max(float(ult[5]) - 10700.0, 0.0) / 100.0     # 107 $ gia' spesi prima del 26/9
    except (OSError, ValueError, IndexError):
        pass
    x10 = f" + extra pagato {extra:,.2f} $ x10 = {10 * extra:,.0f} $" if extra > 0 else " + extra pagato 0 $"
    pod = 0.0                                    # 27/9 regola di Angelo: ogni credito usato da claude-pod = 1 $ di debito
    for m in glob.glob("/var/lib/claude-pod-esiti/*/meta.json"):
        try:
            d = json.load(open(m))
            if datetime.fromisoformat(d["inizio"]) >= INIZIO and d.get("crediti") is not None:
                pod += float(d["crediti"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
    quota = ""                                   # 27/9 Angelo: stop del pod al 30% della sua quota mensile
    try:
        import re, subprocess
        u = subprocess.run(["claude-pod", "consumo"], capture_output=True, text=True, timeout=20).stdout
        m = re.search(r"usato ([0-9.]+) %", u)
        quota = f" · quota pod {m.group(1)}% (stop a 20%)" if m else ""
    except (OSError, subprocess.SubprocessError):
        pass
    print(f"💸 DEBITO: {usd + 10 * extra + pod:,.0f} $ (token dal 26/09 a listino classe Opus, {giorni:.1f} giorni{x10}"
          f" + claude-pod {pod:,.2f} crediti = {pod:,.0f} $){prog}{quota}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

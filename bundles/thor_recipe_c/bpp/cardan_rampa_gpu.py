#!/usr/bin/env python3
"""Rampa di lavoratori MISURATA per la catena (priorita' 0 di Angelo, 21/10 13:2x, punti a-c).

Coda di unita' indipendenti (es. strati della piega esperti: un processo `cardan_piega_gpu_atomica.py --strati u` per
unita'), lavoratori aggiunti UNO ALLA VOLTA:
  * rendimento con N lavoratori = unita'/minuto misurate sulle unita' iniziate E finite mentre i lavoratori erano N
    (N / durata media di quelle unita': tutte hanno girato con la stessa compresenza);
  * si passa a N+1 solo se le code GPU libere ADESSO lo ammettono (cardan_code_gpu, code misurate del tipo), la RAM
    resta sopra la riserva (MemAvailable - picco misurato >= 8 GiB) e lo swap-in <= 20 MB/s;
  * con N+1 misurato: se il rendimento non cresce almeno del 10 % (FORGIA_GUADAGNO_MINIMO) si torna a N e ci si ferma;
  * ogni rendimento e' salvato in contesa_gpu_misurata.json (cardan_code_gpu.salva_contesa) per classe: la tappa
    successiva parte direttamente dal numero ammesso misurato e ricontrolla N+1 soltanto se ancora in salita.
Veglia (D2/D3/stallo): unita' fallite rilanciate fino a --riprove volte; nessuna uscita per --inattivita-s -> TERM/KILL
dei SOLI propri figli, uscita 4. Nessun segnale ad altri processi. Registro JSON per decisione (--registro).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import cardan_code_gpu as CQ  # noqa: E402

GIB = 1 << 30


def host():
    m = {r.split(":")[0]: int(r.split()[1]) * 1024 for r in Path("/proc/meminfo").read_text().splitlines()}
    def sw():
        return int(dict(r.split() for r in Path("/proc/vmstat").read_text().splitlines())["pswpin"])
    s0, t0 = sw(), time.monotonic(); time.sleep(0.5)
    return m["MemAvailable"], (sw() - s0) * os.sysconf("SC_PAGE_SIZE") / (time.monotonic() - t0)


class Rampa:
    def __init__(self, a):
        self.a = a
        self.coda = list(a.unita)
        self.prove = {u: 0 for u in self.coda}
        self.vivi = {}                     # pid -> (Popen, unita, inizio, n_al_lancio, epoca)
        self.n = 1
        self.epoca = 0                     # cambia a ogni cambio di N: un'unita' conta solo se iniziata e finita nella stessa
        self.durate = {}                   # n -> [secondi] di unita' "pure"
        self.rendimenti = {}
        self.fermo = False                 # N deciso: niente piu' salita
        self.reg = open(a.registro, "a", buffering=1)
        try:
            prec = json.loads(CQ.CONTESA.read_text())["classi"][a.classe]
            self.n = max(1, min(a.massimo, int(prec["ammessi"])))
            self.rendimenti = {int(k): float(v) for k, v in prec["rendimenti"].items()}
            self.fermo = str(self.n + 1) in prec["rendimenti"]   # N+1 gia' misurato e respinto
        except (OSError, ValueError, KeyError):
            pass
        self.log("avvio", unita=len(self.coda), n_iniziale=self.n, rendimenti_salvati=self.rendimenti,
                 fermo=self.fermo, code_tipo=self.code_tipo())

    def log(self, evento, **d):
        r = {"evento": evento, "ora": time.strftime("%F %T"), "classe": self.a.classe, "n": self.n, **d}
        self.reg.write(json.dumps(r) + "\n"); print(json.dumps(r), flush=True)

    def code_tipo(self):
        try:
            return CQ.code_lavoratore(self.a.tipo)
        except LookupError:
            return None

    def ammissione(self):
        """(motivo o None) per lanciare un lavoratore in piu' ADESSO."""
        disp, swap = host()
        if swap > 20e6:
            return f"swap-in {swap / 1e6:.1f} MB/s > 20"
        if disp - self.a.picco_gib * GIB < 8 * GIB:
            return f"MemAvailable {disp / GIB:.1f} GiB - picco {self.a.picco_gib} < 8"
        cw = self.code_tipo()
        if cw is None:
            if self.vivi:
                return f"code del tipo {self.a.tipo!r} non misurate: un solo lavoratore"
            from pianificatore_gpu import limite_code_non_misurate
            if limite_code_non_misurate(self.a.device) < 1:
                return f"code del tipo {self.a.tipo!r} non misurate: prima misura solo a GPU vuota"
        else:
            n = CQ.ammissibili(self.a.device, cw["compute"], cw["sdma"])
            if n is not None and n < 1:
                return f"code GPU esaurite ({json.dumps(CQ.uso()[0])})"
        return None

    def lancia(self, u):
        cmd = self.a.comando.replace("{u}", u)
        out = open(self.a.log_unita.replace("{u}", u), "a")
        p = subprocess.Popen(["bash", "-c", cmd], stdout=out, stderr=subprocess.STDOUT, start_new_session=True,
                             env=CQ.ambiente_torch(dict(os.environ)))
        self.vivi[p.pid] = (p, u, time.monotonic(), self.n, self.epoca)
        self.log("lancio", unita=u, pid=p.pid, vivi=len(self.vivi), code=CQ.uso()[0] and {str(k): v for k, v in CQ.uso()[0].items()})

    def misura_e_decidi(self):
        d = self.durate.get(self.n, [])
        if len(d) < self.n or self.fermo and self.n in self.rendimenti:
            return
        r = 60.0 * self.n / (sum(d) / len(d))
        self.rendimenti[self.n] = r
        amm = CQ.salva_contesa(self.a.classe, self.n, r, f"cardan_rampa_gpu {self.a.registro} {time.strftime('%F %T')}")
        self.log("rendimento", unita_min=r, campioni=len(d), ammessi_salvati=amm)
        if self.n > 1 and self.n - 1 in self.rendimenti and not CQ.rendimento_cresce(self.rendimenti[self.n - 1], r):
            self.log("DECISIONE", scelta=self.n - 1, motivo=f"N={self.n} {r:.3f} u/min < 1.1 x N={self.n - 1} "
                     f"{self.rendimenti[self.n - 1]:.3f}: un lavoratore in meno")
            self.n -= 1; self.epoca += 1; self.fermo = True
        elif self.n < self.a.massimo and not self.fermo:
            self.n += 1; self.epoca += 1
            self.log("DECISIONE", scelta=self.n, motivo="salita: misuro N+1")
        else:
            self.fermo = True

    def esegui(self):
        t_ult = time.monotonic()
        while self.coda or self.vivi:
            for pid, (p, u, t0, n0, ep) in list(self.vivi.items()):
                rc = p.poll()
                if rc is None:
                    continue
                del self.vivi[pid]; t_ult = time.monotonic()
                dur = time.monotonic() - t0
                if rc == 0:
                    pura = ep == self.epoca and n0 == self.n and len(self.vivi) + 1 >= self.n
                    self.log("fine", unita=u, rc=0, secondi=round(dur, 1), campione=pura)
                    if pura:
                        self.durate.setdefault(self.n, []).append(dur)
                        self.misura_e_decidi()
                elif self.prove[u] < self.a.riprove:
                    self.prove[u] += 1; self.coda.insert(0, u)
                    self.log("rilancio", unita=u, rc=rc, prova=self.prove[u])
                else:
                    self.log("ERRORE", unita=u, rc=rc); self.arresta(); return 3
            while self.coda and len(self.vivi) < self.n:
                motivo = self.ammissione()
                if motivo:
                    if (motivo, len(self.vivi)) != getattr(self, "_ult_attesa", None) or \
                            time.monotonic() - getattr(self, "_t_attesa", 0) > 60:
                        self._ult_attesa, self._t_attesa = (motivo, len(self.vivi)), time.monotonic()
                        self.log("ATTESA", motivo=motivo, vivi=len(self.vivi))
                    break
                self.lancia(self.coda.pop(0))
            if time.monotonic() - t_ult > self.a.inattivita_s and self.vivi:
                self.log("STALLO", secondi=round(time.monotonic() - t_ult), vivi=list(self.vivi))
                self.arresta(); return 4
            time.sleep(self.a.passo_s)
        self.log("chiuso", rendimenti=self.rendimenti, n_finale=self.n)
        return 0

    def arresta(self):
        for p, *_ in self.vivi.values():
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        time.sleep(self.a.passo_s)
        for p, *_ in self.vivi.values():
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--classe", required=True, help="chiave in contesa_gpu_misurata.json (es. piega_esperti)")
    ap.add_argument("--tipo", required=True, help="tipo in code_misurate_gpu.json (es. piega_esperti_sdma0)")
    ap.add_argument("--unita", required=True, help="lista separata da virgole, o a..b per interi")
    ap.add_argument("--comando", required=True, help="comando bash con {u}")
    ap.add_argument("--log-unita", required=True, help="file di uscita per unita' con {u}")
    ap.add_argument("--registro", required=True)
    ap.add_argument("--massimo", type=int, default=int(os.environ.get("FORGIA_PROCESSI_PER_GPU_MAX", 6)))
    ap.add_argument("--picco-gib", type=float, default=3.0, help="picco RSS misurato per lavoratore")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--riprove", type=int, default=2)
    ap.add_argument("--inattivita-s", type=float, default=1800)
    ap.add_argument("--passo-s", type=float, default=5)
    a = ap.parse_args(argv)
    if ".." in a.unita:
        x, y = a.unita.split("..")
        a.unita = [str(i) for i in range(int(x), int(y) + 1)]
    else:
        a.unita = [u for u in a.unita.split(",") if u]
    r = Rampa(a)
    signal.signal(signal.SIGTERM, lambda s, f: (r.arresta(), sys.exit(143)))
    return r.esegui()


if __name__ == "__main__":
    sys.exit(main())

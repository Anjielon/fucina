#!/usr/bin/env python3
"""Collaudo di applica_patch_tempo.py (25/9 mattina): guardia per CONTENUTO e ricontrollo md5.

Solo CPU, nessun file di thor_v2 scritto: le applicazioni vanno su COPIE nella cartella di lavoro.
I processi finti sono `sleep` dentro script con un nome lungo (comm troncato a 15 caratteri,
come le code vere "confronto_scorr" / "ricetta_base_nu"); si fermano alla fine per PID.

    env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 19 \\
        python3 strumenti/collaudo_guardia_coda.py --lavoro <scratch>/guardia [--originale <copia del vecchio>]

Parte 1 — guardia (ogni caso: il processo finto DEVE comparire, o NON deve):
  shebang con cwd in thor_v2 · setsid -f (orfano, padre = init) · argomento .py assoluto con cwd
  fuori · argomento via collegamento simbolico · coda armata `bash -c "sleep; cd /…/thor_v2 && …"`
  · `cd thor_v2` relativo · PYTHONPATH dentro thor_v2 · negativi (prefisso thor_v2_altro, sleep
  fuori, tail su stato/) · niente auto-blocco (lanciata da un `bash -c "cd thor_v2 && …"`).
  Con --originale: la guardia VECCHIA (per nome) non vede il finto lanciato col suo shebang.
Parte 2 — ricontrollo e ripristino, su copie: percorso buono (file = uscita della prova);
  file cambiato fra prova e applicazione (rifiuto, nulla scritto); patch che fallisce a meta'
  catena (ripristino ai byte originali); uscita diversa dalla prova (ripristino); processo legato
  alla copia comparso fra prova e applicazione (rifiuto).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

QUI = Path(__file__).resolve().parent
THOR_V2 = QUI.parent
sys.path.insert(0, str(QUI))
import applica_patch_tempo as A  # noqa: E402

ESITI: list = []
VIVI: list = []


def ok(nome: str, cond, dettaglio: str = "") -> bool:
    ESITI.append((nome, bool(cond)))
    print(f"  {'✅' if cond else '⛔'} {nome}" + (f" — {dettaglio}" if dettaglio else ""), flush=True)
    return bool(cond)


def lancia(argv, cwd, env=None) -> int:
    # sessione propria: alla fine si ferma l'intero gruppo (bash -c + il suo sleep)
    p = subprocess.Popen(argv, cwd=cwd, env={**os.environ, **(env or {})}, stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    VIVI.append(p.pid)
    return p.pid


def pid_da_setsid(script: Path, cwd, marca: str) -> int:
    """setsid -f: il processo nasce orfano (il setsid esce subito). Il suo PID si trova per
    CONTENUTO (argomento-marca unico), non per nome."""
    subprocess.run(["setsid", "-f", str(script), marca], cwd=cwd, stdin=subprocess.DEVNULL,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    for _ in range(50):
        for voce in os.listdir("/proc"):
            if voce.isdigit():
                try:
                    argv = open(f"/proc/{voce}/cmdline", "rb").read().split(b"\0")
                except OSError:
                    continue
                if marca.encode() in argv and b"sleep" not in argv[0]:
                    VIVI.append(int(voce))
                    return int(voce)
        time.sleep(0.1)
    raise RuntimeError("processo lanciato con setsid non trovato")


def ferma_tutti() -> None:
    for pid in VIVI:
        try:
            # PID verificato un istante prima: e' uno dei nostri finti (argv contiene la marca o sleep)
            argv = open(f"/proc/{pid}/cmdline", "rb").read()
        except OSError:
            continue
        if b"collaudo_guardia" in argv or argv.startswith(b"sleep\x0012"):
            try:
                if os.getpgid(pid) == pid:
                    os.killpg(pid, signal.SIGTERM)
                else:
                    os.kill(pid, signal.SIGTERM)
            except OSError:
                pass


def pids(conflitti: list) -> set:
    return {c["pid"] for c in conflitti}


def parte_guardia(lavoro: Path, originale: "Path | None") -> None:
    print("\n══ parte 1: guardia per contenuto ══", flush=True)
    finto = lavoro / "confronto_scorrevole_finto_collaudo_guardia.sh"
    finto.write_text("#!/bin/bash\n# finto di collaudo_guardia_coda.py: dorme e basta\nsleep 120\n")
    finto.chmod(0o755)
    collegamento = lavoro / "collegamento_a_thor_v2"
    if collegamento.is_symlink() or collegamento.exists():
        collegamento.unlink()
    collegamento.symlink_to(THOR_V2)

    casi_si = {
        "shebang, cwd in thor_v2 (comm troncato)": lancia([str(finto)], THOR_V2),
        "setsid -f, orfano con cwd in thor_v2": pid_da_setsid(finto, THOR_V2, "marca_setsid_collaudo_guardia"),
        "argomento .py assoluto dentro thor_v2, cwd /": lancia(
            [str(finto), str(THOR_V2 / "forgia_globale_40_strati.py")], "/"),
        "argomento via collegamento simbolico, cwd /": lancia(
            [str(finto), str(collegamento / "valida_finestra_8_12.py")], "/"),
        "coda armata bash -c con cd assoluto, cwd /": lancia(
            ["bash", "-c", f"sleep 120; cd {THOR_V2} && ./non_esiste_collaudo_guardia.sh"], "/"),
        "coda armata bash -c con cd RELATIVO, cwd tony-forgia": lancia(
            ["bash", "-c", "sleep 120; cd thor_v2 && ./non_esiste_collaudo_guardia.sh"], THOR_V2.parent),
        "coda armata con ~ nel testo, cwd /": lancia(
            ["bash", "-c", f"sleep 120; cd ~/{THOR_V2.relative_to(Path.home())} && ./finto_collaudo_guardia.sh"], "/"),
        "PYTHONPATH dentro thor_v2, cwd /": lancia(
            [sys.executable, "-c", "import time; time.sleep(120)  # finto collaudo_guardia"], "/",
            env={"PYTHONPATH": str(THOR_V2 / "strumenti")}),
    }
    casi_no = {
        "prefisso trappola thor_v2_altro, cwd /": lancia(
            ["bash", "-c", f"sleep 120; cd {THOR_V2}_altro_collaudo_guardia || true"], "/"),
        "sleep con cwd /": lancia(["sleep", "121"], "/"),
        "tail su stato/, cwd tony-forgia": lancia(
            ["bash", "-c", "sleep 120; tail -n 1 stato/velocita/riepilogo.log  # finto collaudo_guardia"],
            THOR_V2.parent),
    }
    time.sleep(1.0)
    trovati = A.processi_in_conflitto(THOR_V2, dettagli=True)
    per_pid = {c["pid"]: c for c in trovati}
    for nome, pid in casi_si.items():
        ok(f"VISTO: {nome} (pid {pid}, comm {open(f'/proc/{pid}/comm').read().strip()!r})", pid in per_pid,
           "; ".join(per_pid[pid]["motivi"]) if pid in per_pid else "NON visto")
    for nome, pid in casi_no.items():
        ok(f"NON visto (giusto): {nome} (pid {pid})", pid not in per_pid,
           "; ".join(per_pid[pid]["motivi"]) if pid in per_pid else "")
    padre = int(open(f"/proc/{casi_si['setsid -f, orfano con cwd in thor_v2']}/stat").read().rsplit(')', 1)[1].split()[1])
    ok(f"setsid -f: il finto NON e' figlio del collaudo (riadottato dal pid {padre}), visto lo stesso",
       padre != os.getpid() and casi_si['setsid -f, orfano con cwd in thor_v2'] in per_pid)

    if originale is not None:
        spec = importlib.util.spec_from_file_location("applica_vecchio", originale)
        vecchio = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vecchio)
        righe_vecchie = vecchio.processi_in_conflitto()
        pid_s = casi_si["shebang, cwd in thor_v2 (comm troncato)"]
        pid_set = casi_si["setsid -f, orfano con cwd in thor_v2"]
        ok("controprova: la guardia VECCHIA (per nome) NON vede il finto lanciato col suo shebang",
           not any(r.split()[0] == str(pid_s) for r in righe_vecchie))
        ok("controprova: la guardia VECCHIA NON vede il finto lanciato con setsid",
           not any(r.split()[0] == str(pid_set) for r in righe_vecchie))

    # niente auto-blocco: la guardia lanciata da `bash -c "cd thor_v2 && ..."` (come dal terminale di Claude)
    codice = ("import json,sys; sys.path.insert(0,'strumenti'); import applica_patch_tempo as A; "
              "n=A.processi_in_conflitto(dettagli=True); "
              "A._antenati=lambda pid=None: set(); A._mia_pipeline=lambda: set(); "
              "s=A.processi_in_conflitto(dettagli=True); "
              "import os; print(json.dumps({'io': os.getpid(), 'padre': os.getppid(), "
              "'normale': [c['pid'] for c in n], 'senza_esclusione': [c['pid'] for c in s]}))")
    r = subprocess.run(["bash", "-c", f"cd {THOR_V2} && {sys.executable} -c \"{codice}\""],
                       capture_output=True, text=True, cwd="/")
    d = json.loads(r.stdout.strip().splitlines()[-1])
    ok("niente auto-blocco: la guardia non elenca se stessa ne' il bash -c che l'ha lanciata",
       d["io"] not in d["normale"] and d["padre"] not in d["normale"])
    ok("... e senza le esclusioni (antenati, pipeline) li elencherebbe (le esclusioni servono davvero)",
       d["io"] in d["senza_esclusione"] and d["padre"] in d["senza_esclusione"])
    # in una pipeline (`| tee | cat`, come si lancia per tenere il log): i compagni di pipeline hanno la sua cwd
    uscita_tee = lavoro / "guardia_in_pipeline.json"
    r = subprocess.run(["bash", "-c", f"cd {THOR_V2} && {sys.executable} strumenti/applica_patch_tempo.py --solo-guardia "
                        f"--json | tee {uscita_tee} | cat > /dev/null; exit ${{PIPESTATUS[0]}}"], capture_output=True,
                       text=True, cwd="/")
    d = json.loads(uscita_tee.read_text())
    fratelli = [c for c in d["conflitti"] if c["ppid"] == d["padre"]]
    ok(f"in pipeline (| tee | cat): compagni riconosciuti ({len(d['pipeline'])}) e nessuno elencato, "
       "exit 1 per i finti", len(d["pipeline"]) >= 2 and not fratelli and r.returncode == 1
       and set(casi_si.values()) <= pids(d["conflitti"]), "; ".join(f"{c['pid']} {c['comm']}" for c in fratelli))
    r = subprocess.run([sys.executable, str(QUI / "applica_patch_tempo.py"), "--solo-guardia", "--json"],
                       capture_output=True, text=True, cwd=THOR_V2)
    d = json.loads(r.stdout)
    ok("--solo-guardia --json da cwd thor_v2: exit 1 con i finti elencati, se stesso escluso",
       r.returncode == 1 and d["io"] not in pids(d["conflitti"]) and set(casi_si.values()) <= pids(d["conflitti"]))
    vivi_veri = [c for c in d["conflitti"] if c["pid"] not in set(casi_si.values()) | set(casi_no.values())]
    print(f"  (processi VERI legati a thor_v2 in questo momento: {len(vivi_veri)}; primi: "
          + "; ".join(f"{c['pid']} {c['comm']}" for c in vivi_veri[:3]) + ")")

    # guardia su una radice finta: vede SOLO chi ci lavora
    radice = lavoro / "radice_finta"
    radice.mkdir(exist_ok=True)
    pid_r = lancia(["sleep", "122"], radice)
    time.sleep(0.5)
    ok("radice finta: vede esattamente il solo processo che ci lavora",
       pids(A.processi_in_conflitto(radice, dettagli=True)) == {pid_r})


def _copia(dest: Path) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for f in THOR_V2.glob("*.py"):
        shutil.copy2(f, dest / f.name)
    return dest


def _stato(c: Path) -> dict:
    return {f.name: f.read_bytes() for f in sorted(c.glob("*.py"))}


def parte_ricontrollo(lavoro: Path) -> None:
    print("\n══ parte 2: ricontrollo md5, ripristino, rifiuti (su copie) ══", flush=True)
    vero_prima = _stato(THOR_V2)

    # a. percorso buono
    c = _copia(lavoro / "copie" / "buona")
    with tempfile.TemporaryDirectory(prefix="collaudo_guardia_") as d:
        foto, uscita = A.prova_generale(c, Path(d))
        A.applica_vera(c, foto, uscita)
    finali = _stato(c)
    ok(f"percorso buono: i {len(uscita)} bersagli della copia sono IDENTICI all'uscita della prova ({sorted(uscita)})",
       {n: finali[n] for n in uscita} == uscita and len(uscita) == 3)
    marche = ["patch_tempo_prepara_asincrona", "patch_tempo_riuso_teacher", "patch_tempo_prefetch_strato",
              "patch_tempo_cache_attenzione", "patch_tempo_attenzione_confine", "patch_tempo_congelo_senza_moe",
              "patch_tempo_beta_costante"]
    testo = "".join(finali[n].decode() for n in uscita)
    ok("percorso buono: tutte e 7 le marche presenti", all(m in testo for m in marche))

    # b. file cambiato fra prova e applicazione
    c = _copia(lavoro / "copie" / "cambiata")
    with tempfile.TemporaryDirectory(prefix="collaudo_guardia_") as d:
        foto, uscita = A.prova_generale(c, Path(d))
        f = c / "forgia_scorrevole.py"
        f.write_bytes(f.read_bytes() + b"\n# cambiato fra prova e applicazione (collaudo)\n")
        dopo_cambio = _stato(c)
        try:
            A.applica_vera(c, foto, uscita)
            rifiuto, msg = False, ""
        except SystemExit as e:
            rifiuto, msg = True, str(e)
    ok("file cambiato fra prova e applicazione: RIFIUTO con i due md5", rifiuto and "file cambiati" in msg
       and "forgia_scorrevole.py" in msg, msg[:160])
    ok("... e nessun file della copia toccato", _stato(c) == dopo_cambio)

    # c. patch che fallisce a meta' catena
    c = _copia(lavoro / "copie" / "meta_catena")
    originali = _stato(c)

    def catena_rotta(cartella):
        for p in A.ORDINE[:3]:
            subprocess.run([sys.executable, str(QUI / f"patch_tempo_{p}.py"), "--cartella", str(cartella)],
                           check=True, capture_output=True)
        raise SystemExit("⛔ patch finta RIFIUTATA (collaudo): catena fermata")
    with tempfile.TemporaryDirectory(prefix="collaudo_guardia_") as d:
        foto, uscita = A.prova_generale(c, Path(d))
        try:
            A.applica_vera(c, foto, uscita, catena_fn=catena_rotta)
            msg = ""
        except SystemExit as e:
            msg = str(e)
    ok("catena interrotta dopo 3 patch: errore e file RIPORTATI ai byte originali",
       "riportati ai byte originali" in msg and _stato(c) == originali, msg.splitlines()[-1][:150] if msg else "")

    # d. uscita diversa dalla prova (qualcosa ha toccato un file durante la catena)
    c = _copia(lavoro / "copie" / "uscita_diversa")
    originali = _stato(c)

    def catena_sporca(cartella):
        A.catena(cartella)
        g = Path(cartella) / "valida_finestra_8_12.py"
        g.write_bytes(g.read_bytes() + b"\n# sporcato durante la catena (collaudo)\n")
    with tempfile.TemporaryDirectory(prefix="collaudo_guardia_") as d:
        foto, uscita = A.prova_generale(c, Path(d))
        try:
            A.applica_vera(c, foto, uscita, catena_fn=catena_sporca)
            msg = ""
        except SystemExit as e:
            msg = str(e)
    ok("uscita diversa dalla prova: errore e file RIPORTATI ai byte originali",
       "differiscono dall'uscita della prova" in msg and _stato(c) == originali, msg.splitlines()[0][:150] if msg else "")

    # d2. un file NON bersaglio cambia fra prova e applicazione e DURANTE la catena (lavoro altrui),
    #     e la catena poi fallisce: nessun rifiuto per quel file, e il ripristino NON lo tocca
    c = _copia(lavoro / "copie" / "file_altrui")
    altrui = c / "e2e_lavoro_altrui_finto.py"
    altrui.write_text("# versione 1 (collaudo)\n")
    originali = _stato(c)

    def catena_con_altrui(cartella):
        for p in A.ORDINE[:2]:
            subprocess.run([sys.executable, str(QUI / f"patch_tempo_{p}.py"), "--cartella", str(cartella)],
                           check=True, capture_output=True)
        altrui.write_text("# versione 3, scritta DURANTE la catena da un altro lavoro (collaudo)\n")
        raise SystemExit("⛔ patch finta RIFIUTATA (collaudo): catena fermata")
    with tempfile.TemporaryDirectory(prefix="collaudo_guardia_") as d:
        foto, uscita = A.prova_generale(c, Path(d))
        altrui.write_text("# versione 2, fra prova e applicazione (collaudo)\n")
        try:
            A.applica_vera(c, foto, uscita, catena_fn=catena_con_altrui)
            msg = ""
        except SystemExit as e:
            msg = str(e)
    ok("file NON bersaglio cambiato fra prova e applicazione: nessun rifiuto per lui (la catena parte)",
       "file cambiati" not in msg and "patch finta" in msg, msg.splitlines()[0][:100] if msg else "")
    ok("... catena fallita: bersagli ai byte originali, e il file altrui NON ripristinato (resta la versione 3)",
       all(_stato(c)[n] == originali[n] for n in uscita) and "versione 3" in altrui.read_text())

    # e. processo legato alla copia comparso fra prova e applicazione
    c = _copia(lavoro / "copie" / "coda_nata")
    originali = _stato(c)
    with tempfile.TemporaryDirectory(prefix="collaudo_guardia_") as d:
        foto, uscita = A.prova_generale(c, Path(d))
        pid = lancia(["bash", "-c", f"sleep 120; cd {c} && ./finto_collaudo_guardia.sh"], "/")
        time.sleep(0.5)
        try:
            A.applica_vera(c, foto, uscita)
            msg = ""
        except SystemExit as e:
            msg = str(e)
    ok(f"coda comparsa fra prova e applicazione (pid {pid}): RIFIUTO al ricontrollo, nulla scritto",
       "(ricontrollo)" in msg and _stato(c) == originali, msg[:120])

    dopo = _stato(THOR_V2)
    nuovi = sorted(set(dopo) - set(vero_prima))
    ok("i file veri di thor_v2 NON sono stati toccati dal collaudo (ogni .py di prima, identico al byte)",
       all(dopo.get(n) == b for n, b in vero_prima.items()),
       (f"nati durante il collaudo, da ALTRI processi (il collaudo scrive solo in --lavoro): {nuovi}" if nuovi else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lavoro", type=Path, required=True)
    ap.add_argument("--originale", type=Path, default=None, help="copia della versione vecchia (guardia per nome)")
    args = ap.parse_args()
    lavoro = args.lavoro.resolve()
    if lavoro == THOR_V2 or THOR_V2 in lavoro.parents:
        raise SystemExit("la cartella di lavoro non puo' stare dentro thor_v2")
    lavoro.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        parte_guardia(lavoro, args.originale)
        parte_ricontrollo(lavoro)
    finally:
        ferma_tutti()
    n_ok = sum(1 for _, v in ESITI if v)
    print(f"\n{'✅ TUTTI' if n_ok == len(ESITI) else '⛔ FALLITI ' + str(len(ESITI) - n_ok) + ' di'} "
          f"{len(ESITI)} controlli in {time.time() - t0:.0f}s")
    return 0 if n_ok == len(ESITI) else 1


if __name__ == "__main__":
    raise SystemExit(main())

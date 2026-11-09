#!/usr/bin/env python3
"""Applica le 7 patch di velocita' (PIANO_TEMPO_SCORREVOLE.md) in catena — prima su una COPIA, poi sui file veri.

Ogni patch da sola e' gia' atomica (niente scritture parziali), ma una CATENA interrotta a meta'
lascerebbe thor_v2 con alcune leve si' e altre no. Qui:
  1. GUARDIA per CONTENUTO (25/9 mattina, verdetto E5 del verificatore avversario): e' in conflitto
     ogni processo — esclusi questo, i suoi antenati e i processi della sua pipeline (`| tee`) — che
       a. ha la cartella di lavoro dentro la cartella da patchare, oppure
       b. ha fra gli argomenti (anche dentro un testo `bash -c "..."`/`python -c "..."`, spezzato
          sui separatori della shell) un percorso che risolve dentro quella cartella (assoluto,
          relativo alla SUA cartella di lavoro, o con ~ / $HOME), oppure
       c. ha nel PYTHONPATH una voce dentro quella cartella.
     Prima si guardava il NOME del processo (`comm == "bash"` + elenco di script): uno script
     lanciato col suo shebang ha come comm il proprio nome troncato a 15 caratteri
     ("confronto_scorr", "ricetta_base_nu") e non veniva mai visto; le code risultavano bloccate
     solo per caso, dal `bash -c` che le aveva lanciate. Nessun elenco di nomi da tenere aggiornato.
       d. (25/9, verifica avversaria: otto forme sfuggivano) ha un FILE APERTO dentro la cartella
          (bash tiene lo script sul descrittore 255: `./x.sh` che poi fa `cd /`), oppure il TESTO dello
          script che esegue la USA — ci entra (cd/pushd), nomina un percorso dentro, la mette in
          PYTHONPATH/sys.path/chdir/-C (stato/velocita/coda_q4.sh vive fuori e aspetta PRIMA del `cd`;
          anche `D=…; cd $D/thor_v2`, `~/…`). Nominarla soltanto non basta: ~/.local/bin/ping-claude.sh,
          servizio sempre vivo, la confronta con la cwd altrui e bloccherebbe per sempre. Un argomento
          la usa anche come opzione attaccata (`-C/…/thor_v2`); il collegamento simbolico si risolve
          anche verso file che non esistono ancora;
       e. code ARMATE che non sono ancora processi: crontab, lavori `at`, timer systemd utente e
          sistema (anche i transitori di systemd-run). Una fonte non interrogabile = conflitto.
     La pipeline si segue nel VERSO dei tubi (a valle chi legge la mia uscita, a monte chi scrive
     il mio ingresso): un `nohup coda &` nello stesso comando, che eredita la mia stdout, resta visibile.
  2. prova generale: copia dei sorgenti in una cartella temporanea e catena completa li';
  3. subito prima della catena vera: di nuovo la guardia, e l'md5 di ogni file (i BERSAGLI della
     catena, dai PIANO delle patch, + script delle patch) deve essere quello copiato per la prova. Alle 02:10 del 25/9
     forgia_scorrevole.py era cambiata fra una prova e l'altra: senza questo controllo si
     applicherebbe una catena mai provata su quei byte;
  4. la catena sui file veri, con i byte originali tenuti da parte: se una patch fallisce, o se
     alla fine anche un solo bersaglio differisce dall'uscita della prova, TUTTI i bersagli tornano
     come prima (scrittura .tmp + os.replace) e il comando esce con errore. I file che la catena non
     tocca non si leggono, non si confrontano e non si ripristinano (lavori altrui nella cartella).
     (25/9) Il ripristino scatta su QUALUNQUE eccezione (Ctrl-C, OSError) e su SIGTERM/SIGHUP; contro
     SIGKILL o la macchina spenta c'e' il GIORNALE (strumenti/.giornale_applica_patch_tempo: byte
     originali + md5), scritto prima della catena e tolto solo a bersagli coerenti: finche' esiste
     ogni corsa rifiuta, e --ripristina-giornale riporta i bersagli ai byte originali.
Con --con-export la catena comprende in coda la patch dell'export incrementale (leva F).
Con --prova si ferma dopo il punto 2. Con --solo-guardia stampa i processi in conflitto (JSON con
--json) e termina (exit 1 se ce n'e' almeno uno). Nessuna patch viene applicata due volte (la
marca lo impedisce).

    python3 strumenti/applica_patch_tempo.py --prova          # controllo completo, NULLA scritto in thor_v2
    python3 strumenti/applica_patch_tempo.py --solo-guardia   # chi blocca, e perche'
    python3 strumenti/applica_patch_tempo.py                  # applica
Collaudo della guardia e del ricontrollo: strumenti/collaudo_guardia_coda.py (CPU, su copie).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

QUI = Path(__file__).resolve().parent
THOR_V2 = QUI.parent
ORDINE = ["prepara_asincrona", "riuso_teacher", "prefetch_strato", "cache_attenzione", "attenzione_confine",
          "congelo_senza_moe", "beta_costante"]
# separatori della shell (e di liste tipo PYTHONPATH / --opzione=valore) per spezzare i testi in candidati percorso
_SEPARATORI = re.compile(r"[\s;&|()<>'\"`=,:]+")


# ═══════════════════════════ 1. guardia per contenuto ═══════════════════════════
def _antenati(pid: "int | None" = None) -> set:
    """Il processo `pid` (default: questo) e tutti i suoi antenati fino a init."""
    pid, visti = (os.getpid() if pid is None else pid), set()
    while pid > 1 and pid not in visti:
        visti.add(pid)
        try:
            with open(f"/proc/{pid}/stat") as f:
                pid = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return visti


def _tubo(pid: str, fd: int) -> "str | None":
    try:
        dest = os.readlink(f"/proc/{pid}/fd/{fd}")
    except OSError:
        return None
    return dest if dest.startswith("pipe:") else None


def _mia_pipeline() -> set:
    """I compagni di pipeline di questo processo, seguendo la DIREZIONE dei tubi: a valle chi LEGGE
    (stdin) da un tubo in cui la catena scrive (`applica | tee log | cat`), a monte chi SCRIVE
    (stdout/stderr) in un tubo da cui la catena legge (`echo | applica`). Hanno la stessa cartella di
    lavoro ma sono il comando stesso, non una coda.
    ⛔ 25/9 (verifica avversaria): prima bastava CONDIVIDERE un tubo su fd 0/1, in qualunque verso, e
    un `nohup coda.sh &` lanciato nello stesso comando dell'applicatore — che eredita la sua stdout —
    veniva escluso come «compagno»: una coda viva, invisibile. Chi scrive nello STESSO tubo in cui
    scrivo io non e' ne' a valle ne' a monte: resta visibile."""
    valle = {t for t in (_tubo("self", 1), _tubo("self", 2)) if t}
    monte = {t for t in (_tubo("self", 0),) if t}
    io = os.getpid()
    voci = [v for v in os.listdir("/proc") if v.isdigit() and int(v) != io]
    compagni: set = set()
    cambiato = True
    while cambiato and (valle or monte):
        cambiato = False
        for voce in voci:
            if int(voce) in compagni:
                continue
            t0, t1, t2 = _tubo(voce, 0), _tubo(voce, 1), _tubo(voce, 2)
            if t0 is not None and t0 in valle:                         # legge cio' che la catena scrive
                compagni.add(int(voce))
                valle |= {t for t in (t1, t2) if t}
                cambiato = True
            elif (t1 is not None and t1 in monte) or (t2 is not None and t2 in monte):   # scrive per la catena
                compagni.add(int(voce))
                if t0:
                    monte.add(t0)
                cambiato = True
    return compagni


def _dentro(percorso: str, radice: str, radice_reale: str) -> bool:
    """`percorso` (gia' assoluto e normalizzato) e' la radice o sta sotto di lei — anche passando per un
    collegamento simbolico. ⛔ 25/9: realpath SEMPRE (non solo se il percorso esiste): un file che non
    esiste ancora dietro un collegamento alla radice (`coll/nuovo.sh`) sfuggiva."""
    for r in {radice, radice_reale}:
        if percorso == r or percorso.startswith(r + "/"):
            return True
    try:
        reale = os.path.realpath(percorso)
    except (OSError, ValueError):
        return False
    return reale == radice_reale or reale.startswith(radice_reale + "/")


def _candidati(testo: str, cwd: "str | None", casa: str) -> "list[str]":
    """Tutti i pezzi di `testo` che possono essere percorsi, resi assoluti (relativi a `cwd`;
    `cwd=None` = solo i pezzi assoluti). Un'opzione attaccata al percorso (`-C/percorso`,
    `-o~/x`) conta dal primo `/` o `~` (25/9: `env -C<radice>` sfuggiva)."""
    out = []
    for pezzo in _SEPARATORI.split(testo):
        if pezzo.startswith(("-", "+")):
            inizio = [i for i in (pezzo.find("/"), pezzo.find("~")) if i > 0]
            if not inizio:
                continue
            pezzo = pezzo[min(inizio):]
        if not pezzo:
            continue
        for pref in ("${HOME}", "$HOME", "~"):
            if pezzo == pref or pezzo.startswith(pref + "/"):
                pezzo = casa + pezzo[len(pref):]
                break
        if os.path.isabs(pezzo):
            out.append(os.path.normpath(pezzo))
        elif cwd is not None:             # anche `cd thor_v2` da tony-forgia: ogni pezzo relativo alla SUA cwd
            out.append(os.path.normpath(os.path.join(cwd, pezzo)))
    return out


class _Radice:
    """La radice in tutte le forme in cui un testo puo' nominarla."""

    def __init__(self, radice: Path):
        self.s = os.path.normpath(str(Path(radice).absolute()))
        self.r = os.path.realpath(self.s)
        self.casa = os.path.expanduser("~")
        forme = {self.s, self.r}
        for r in (self.s, self.r):
            if r.startswith(self.casa + "/"):
                rel = r[len(self.casa) + 1:]
                forme |= {"~/" + rel, "$HOME/" + rel, "${HOME}/" + rel}
        self.base = os.path.basename(self.s)
        self.genitori = {os.path.dirname(self.s), os.path.dirname(self.r)}
        fine = r"(?![\w.-])"                             # thor_v2_altro, thor_v2-bis, thor_v2.bak: altre cartelle
        assoluta = "|".join(re.escape(f) for f in sorted(forme, key=len, reverse=True))
        variabile = r"\$\{?\w+\}?/" + re.escape(self.base)        # `D=…; cd $D/thor_v2`, `${D}/thor_v2`
        relativa = r"(?<![\w./$}-])" + re.escape(self.base)       # `cd thor_v2` (conta solo dal genitore)
        comandi_cartella = r"(?:^|[\s;&|(`\"'{])(?:cd|pushd)\s+(?:-[LPe]+\s+)?[\"']?"
        # Un testo USA la radice (non basta nominarla: ping-claude.sh, servizio sempre vivo, la confronta
        # con la cwd degli altri processi e non esegue nulla lì) se: ci entra (cd/pushd), nomina un
        # percorso DENTRO di lei, o la mette dove si esegue/importa (PYTHONPATH, sys.path, chdir, -C, …).
        uso = r"(?:PYTHONPATH|sys\.path|chdir|WorkingDirectory|--cartella|runpy|importlib|\s-C\s?)"

        def regole(forma: str) -> "list[re.Pattern]":
            return [re.compile(comandi_cartella + f"(?:{forma})" + fine),
                    re.compile(f"(?:{forma})/[\\w.$-]"),
                    re.compile(uso + r".*?(?:" + forma + ")" + fine + "|(?:" + forma + ")" + fine + ".*?" + uso)]
        self.re_uso = regole(f"{assoluta}|{variabile}")
        self.re_uso_rel = regole(relativa)

    def dentro(self, percorso: str) -> bool:
        return _dentro(percorso, self.s, self.r)

    def usata(self, testo: str, relativa_da: "set | None" = None) -> "str | None":
        """La prima riga (non commento) di `testo` che USA la radice, None se nessuna. `relativa_da`: le
        cartelle da cui chi legge il testo risolve i nomi relativi (cwd del processo, cartella dello
        script): `cd thor_v2` conta solo se una di loro e' il genitore della radice."""
        relativa = bool(relativa_da and relativa_da & self.genitori)
        for riga in testo.splitlines():
            r = riga.strip()
            if not r or r.startswith("#"):
                continue
            for regola in self.re_uso + (self.re_uso_rel if relativa else []):
                if regola.search(r):
                    return r[:120]
        return None


_INTERPRETE = re.compile(r"^(?:ba|da|z|k|)sh$|^python[0-9.]*$")
_OPZIONI_CON_VALORE = {"-O", "+O", "-o", "+o", "-W", "-X", "--rcfile", "--init-file"}
_LIMITE_SCRIPT = 1 << 20
_cache_script: dict = {}


def _script_di(argv: "list[str]", cwd: "str | None", casa: str) -> "str | None":
    """Il file che un interprete (sh/bash/python) sta eseguendo, risolto; None se il codice e' in linea
    (`-c`, `python -m`) o se argv[0] non e' un interprete. Uno script lanciato col suo shebang ha
    argv = [interprete, script, ...]: il kernel lo riscrive cosi'."""
    if not argv or not _INTERPRETE.match(os.path.basename(argv[0])):
        return None
    python = os.path.basename(argv[0]).startswith("python")
    salta = False
    for a in argv[1:]:
        if salta:
            salta = False
            continue
        if a == "--":
            continue
        if a.startswith(("-", "+")) and len(a) > 1:
            if a in _OPZIONI_CON_VALORE:
                salta = True
            elif python and a.startswith(("-W", "-X")):
                pass                                            # valore attaccato (-Werror::..., -Xutf8)
            elif not a.startswith("--") and ("c" in a[1:] or (python and "m" in a[1:])):
                return None                                     # -c / -lc / -m: niente file di script
            continue
        for c in _candidati(a, cwd, casa)[:1]:
            return c if os.path.isfile(c) else None
        return None
    return None


def _testo_script(percorso: str) -> "str | None":
    try:
        st = os.stat(percorso)
        if not os.path.isfile(percorso) or st.st_size > _LIMITE_SCRIPT:
            return None
        chiave = (percorso, st.st_size, st.st_mtime_ns, st.st_ino)
        if chiave not in _cache_script:
            with open(percorso, "rb") as f:
                _cache_script[chiave] = f.read().decode(errors="replace")
        return _cache_script[chiave]
    except OSError:
        return None


def _motivi_processo(voce: str, argv: "list[str]", R: _Radice) -> "list[str]":
    """Perche' il processo `voce` e' legato alla radice (lista vuota = non lo e')."""
    try:
        cwd = os.readlink(f"/proc/{voce}/cwd")
    except OSError:
        cwd = None                        # processo di un altro utente: restano gli argomenti assoluti
    motivi = []
    cwd_dentro = cwd is not None and R.dentro(os.path.normpath(cwd))
    if cwd_dentro:
        motivi.append(f"cartella di lavoro {cwd}")
    # con la cwd gia' dentro ogni pezzo relativo risolverebbe dentro: contano solo gli assoluti
    cwd_rel = None if cwd_dentro else cwd
    for a in argv:
        per_a = [c for c in _candidati(a, cwd_rel, R.casa) if R.dentro(c)]
        if per_a:
            motivi.append(f"argomento che nomina {per_a[0]}")
            break
    else:
        usata = R.usata(" ".join(argv), {cwd} if cwd else None)   # `bash -c "D=…; cd $D/thor_v2"`
        if usata:
            motivi.append(f"argomento che usa la radice: «{usata}»")
    try:
        with open(f"/proc/{voce}/environ", "rb") as f:
            ambiente = f.read().split(b"\0")
        pp = next((v.decode(errors="replace")[len("PYTHONPATH="):] for v in ambiente
                   if v.startswith(b"PYTHONPATH=")), "")
        voci_pp = [c for c in _candidati(pp.replace(":", " "), cwd_rel, R.casa) if R.dentro(c)]
        if voci_pp:
            motivi.append(f"PYTHONPATH con {voci_pp[0]}")
    except OSError:
        pass
    # ⛔ 25/9: file APERTI dentro la radice. Uno script della radice lanciato `./x.sh` che poi fa `cd /`
    # ha argv relativo e cwd fuori: lo tradisce il descrittore 255 con cui bash tiene aperto lo script.
    script_aperti = []
    try:
        fds = os.listdir(f"/proc/{voce}/fd")
    except OSError:
        fds = []
    for fd in fds:
        try:
            dest = os.readlink(f"/proc/{voce}/fd/{fd}")
        except OSError:
            continue
        if not dest.startswith("/") or dest.startswith(("/dev/", "/proc/", "/sys/")):
            continue
        dest = dest[:-len(" (deleted)")] if dest.endswith(" (deleted)") else dest
        if R.dentro(dest):
            motivi.append(f"file aperto dentro la radice: {dest} (fd {fd})")
            break
        if fd == "255":
            script_aperti.append(dest)
    # ⛔ 25/9: il TESTO dello script. stato/velocita/coda_q4.sh (24/9) vive FUORI dalla radice e aspetta
    # PRIMA di fare `cd /…/thor_v2`: argv, cwd e file aperti non la nominano finche' non parte.
    for s in {x for x in [_script_di(argv, cwd, R.casa)] + script_aperti if x}:
        testo = _testo_script(s)
        usata = R.usata(testo, {c for c in (cwd, os.path.dirname(s)) if c}) if testo else None
        if usata:
            motivi.append(f"lo script {s} usa la radice: «{usata}»")
            break
    return motivi


def _esegui(argv: "list[str]", timeout: float = 20.0) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    env.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path=/run/user/{os.getuid()}/bus")
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env)


def _blocchi_show(testo: str) -> "list[dict]":
    """`systemctl show -p A -p B u1 u2`: un blocco per unita', separati da una riga vuota."""
    blocchi, cor = [], {}
    for riga in testo.splitlines() + [""]:
        if not riga.strip():
            if cor:
                blocchi.append(cor)
            cor = {}
        elif "=" in riga:
            k, v = riga.split("=", 1)
            cor[k] = (cor[k] + " " + v) if k in cor else v
    return blocchi


def code_pianificate(R: _Radice, esegui=_esegui) -> list:
    """Code ARMATE che non sono ancora un processo: crontab dell'utente, lavori `at`, timer systemd
    (utente e sistema, anche i transitori di `systemd-run --on-active`). Si legge il comando, la
    cartella di lavoro, l'ambiente e il testo dello script lanciato. Una fonte che non si riesce a
    interrogare e' un conflitto (nel dubbio si rifiuta), non un «nessuno»."""
    out = []

    def conflitto(tipo: str, nome: str, motivo: str, testo: str = "") -> None:
        out.append({"pid": None, "ppid": None, "comm": tipo, "motivi": [motivo],
                    "cmdline": f"{nome}: {' '.join(testo.split())[:140]}"})

    def guarda(tipo: str, nome: str, testo: str, cwd: "str | None" = None) -> None:
        dentro = [c for c in _candidati(testo, cwd, R.casa) if R.dentro(c)]
        usata = R.usata(testo, {cwd} if cwd else None)
        if dentro or usata:
            conflitto(tipo, nome, f"usa la radice ({dentro[0] if dentro else usata})", testo)
            return
        for pezzo in re.split(r"[;&|\n]+", testo):                     # ogni comando: lo script che lancia
            argv = pezzo.split()
            if not argv:
                continue
            script = {_script_di(argv, cwd, R.casa)}                    # `python3 x.py`, `bash x.sh`
            primo = _candidati(argv[0], cwd, R.casa)[:1]                # `/percorso/x.sh` col suo shebang
            if primo and (_testo_script(primo[0]) or "").startswith("#!"):
                script.add(primo[0])
            for s in sorted(x for x in script if x):
                t = _testo_script(s)
                usata = R.usata(t, {c for c in (cwd, os.path.dirname(s)) if c}) if t else None
                if usata:
                    conflitto(tipo, nome, f"lo script {s} usa la radice: «{usata}»", testo)
                    return

    try:                                                               # cron
        r = esegui(["crontab", "-l"])
        if r.returncode == 0:
            for riga in r.stdout.splitlines():
                campi = riga.split()
                if not campi or campi[0].startswith("#") or re.match(r"^[A-Za-z_]\w*=", campi[0]):
                    if campi and not campi[0].startswith("#"):
                        guarda("cron", "crontab (variabile)", riga, R.casa)
                    continue
                comando = " ".join(campi[1:] if campi[0].startswith("@") else campi[5:])
                guarda("cron", "crontab", comando, R.casa)
        elif "no crontab" not in (r.stderr or "").lower():
            conflitto("cron", "crontab -l", f"cron NON verificabile (rc {r.returncode}: {r.stderr.strip()[:80]})")
    except (OSError, subprocess.TimeoutExpired) as e:
        if not isinstance(e, FileNotFoundError):
            conflitto("cron", "crontab -l", f"cron NON verificabile ({e})")
    try:                                                               # at
        r = esegui(["atq"])
        if r.returncode != 0:
            conflitto("at", "atq", f"lavori at NON verificabili (rc {r.returncode}: {r.stderr.strip()[:80]})")
        for riga in r.stdout.splitlines() if r.returncode == 0 else []:
            job = riga.split()[0] if riga.split() else ""
            if job:
                rr = esegui(["at", "-c", job])
                if rr.returncode != 0:
                    conflitto("at", f"at {job}", f"lavoro at {job} NON leggibile (rc {rr.returncode})")
                else:
                    guarda("at", f"at {job}", rr.stdout)
    except (OSError, subprocess.TimeoutExpired) as e:
        if not isinstance(e, FileNotFoundError):
            conflitto("at", "atq", f"lavori at NON verificabili ({e})")
    for utente in (True, False):                                       # timer systemd
        base = ["systemctl"] + (["--user"] if utente else []) + ["--no-pager"]
        dove = "utente" if utente else "sistema"
        try:
            r = esegui(base + ["list-units", "--all", "--type=timer", "--no-legend", "--plain"])
            if r.returncode != 0:
                conflitto("timer", f"timer {dove}", f"timer {dove} NON verificabili (rc {r.returncode}: "
                          f"{r.stderr.strip()[:80]})")
                continue
            timer = [riga.split()[0] for riga in r.stdout.splitlines() if riga.split()]
            if not timer:
                continue
            r = esegui(base + ["show", "-p", "Id", "-p", "Triggers"] + timer)
            servizi = {b.get("Id", "?"): b.get("Triggers", "").split() for b in _blocchi_show(r.stdout)}
            tutti = sorted({s for ss in servizi.values() for s in ss})
            if r.returncode != 0 or not tutti:
                if r.returncode != 0:
                    conflitto("timer", f"timer {dove}", f"timer {dove} NON verificabili (show rc {r.returncode})")
                continue
            r = esegui(base + ["show", "-p", "Id", "-p", "ExecStart", "-p", "WorkingDirectory", "-p", "Environment"]
                       + tutti)
            if r.returncode != 0:
                conflitto("timer", f"timer {dove}", f"servizi dei timer {dove} NON verificabili (rc {r.returncode})")
                continue
            per_servizio = {b.get("Id", "?"): b for b in _blocchi_show(r.stdout)}
            for t, ss in servizi.items():
                for s in ss:
                    b = per_servizio.get(s, {})
                    argv_exec = " ".join(re.findall(r"argv\[\]=(.*?) ;", b.get("ExecStart", ""))) or b.get("ExecStart", "")
                    cwd_s = b.get("WorkingDirectory", "").lstrip("!-") or None
                    guarda("timer", f"{t} → {s} ({dove})",
                           f"{argv_exec} ; WorkingDirectory={cwd_s or ''} ; Environment={b.get('Environment', '')}",
                           cwd_s)
        except (OSError, subprocess.TimeoutExpired) as e:
            if not isinstance(e, FileNotFoundError):
                conflitto("timer", f"timer {dove}", f"timer {dove} NON verificabili ({e})")
    return out


def processi_in_conflitto(radice: Path = THOR_V2, escludi: "set | None" = None, dettagli: bool = False,
                          pianificate: bool = True) -> list:
    """Processi (esclusi questo, i suoi antenati, la sua pipeline e `escludi`) legati per CONTENUTO a `radice`:
    cartella di lavoro, argomenti (anche la radice composta da variabile), PYTHONPATH, file aperti,
    testo dello script che eseguono. Con `pianificate` anche le code armate che non sono ancora un
    processo (cron, at, timer systemd). Con `dettagli` ritorna dizionari {pid, ppid, comm, motivi,
    cmdline} (pid None per le pianificate); altrimenti righe di testo."""
    R = _Radice(radice)
    esclusi = _antenati() | _mia_pipeline() | set(escludi or ())
    trovati = []
    for voce in os.listdir("/proc"):
        if not voce.isdigit() or int(voce) in esclusi:
            continue
        try:
            with open(f"/proc/{voce}/cmdline", "rb") as f:
                argv = [a.decode(errors="replace") for a in f.read().split(b"\0") if a]
            with open(f"/proc/{voce}/comm") as f:
                comm = f.read().strip()
        except OSError:
            continue
        if not argv:                      # thread del kernel o zombie
            continue
        motivi = _motivi_processo(voce, argv, R)
        if motivi:
            try:
                with open(f"/proc/{voce}/stat") as f:
                    ppid = int(f.read().rsplit(")", 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                ppid = None
            trovati.append({"pid": int(voce), "ppid": ppid, "comm": comm, "motivi": motivi,
                            "cmdline": " ".join(argv)[:160]})
    if pianificate:
        trovati += code_pianificate(R)
    if dettagli:
        return trovati
    return [f"{c['pid'] if c['pid'] is not None else '-'} {c['comm']} ({'; '.join(c['motivi'])}) :: {c['cmdline']}"
            for c in trovati]


def testo_conflitti(conflitti: list, massimo: int = 12) -> str:
    """Le prime `massimo` righe (un compilatore di torch ne apre decine, tutti uguali) + il conto."""
    righe = conflitti[:massimo]
    if len(conflitti) > massimo:
        righe.append(f"... e altri {len(conflitti) - massimo} processi (tutti con --solo-guardia --json)")
    return "\n".join(righe)


# ═══════════════════════════ 2. prova generale e catena ═══════════════════════════
def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def bersagli_catena() -> "list[str]":
    """I file che la catena puo' cambiare: l'unione dei PIANO delle patch in ORDINE. Solo questi si
    fotografano, si confrontano e si ripristinano: un file d'altri (25/9 05:34: un altro lavoro
    scriveva e2e_*.py in thor_v2 mentre giravano i collaudi) non si tocca MAI, nemmeno per ripristino."""
    import importlib.util
    nomi: set = set()
    for p in ORDINE:
        spec = importlib.util.spec_from_file_location(f"_patch_{p}", QUI / f"patch_tempo_{p}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        nomi |= set(mod.PIANO)
    return sorted(nomi)


def fotografia(cartella: Path) -> dict:
    """{nome: byte} dei file bersaglio della catena e di ogni script di patch usato qui."""
    foto = {f"cartella/{n}": (Path(cartella) / n).read_bytes() for n in bersagli_catena()}
    for f in [QUI / "patch_tempo_comune.py"] + [QUI / f"patch_tempo_{p}.py" for p in ORDINE]:
        foto[f"strumenti/{f.name}"] = f.read_bytes()
    return foto


def catena(cartella: Path) -> None:
    for p in ORDINE:
        r = subprocess.run([sys.executable, str(QUI / f"patch_tempo_{p}.py"), "--cartella", str(cartella)],
                           capture_output=True, text=True)
        uscita = (r.stdout + r.stderr).strip()
        print(f"   {p}: {uscita.splitlines()[-1][:150] if uscita else ''}")
        if r.returncode != 0:
            raise SystemExit(f"⛔ patch {p} RIFIUTATA su {cartella}: catena fermata")


def prova_generale(cartella: Path, dir_prova: Path) -> "tuple[dict, dict]":
    """Copia i file bersaglio di `cartella` in `dir_prova`/thor_v2 DAI BYTE FOTOGRAFATI e ci applica
    la catena. Ritorna (fotografia d'ingresso, {nome: byte d'uscita} dei file bersaglio della copia)."""
    foto = fotografia(cartella)
    prova = Path(dir_prova) / "thor_v2"
    (prova / "strumenti").mkdir(parents=True)
    for chiave, b in foto.items():
        if chiave.startswith("cartella/"):
            (prova / chiave.split("/", 1)[1]).write_bytes(b)
    print(f"prova generale su {prova}:")
    catena(prova)
    return foto, {k.split("/", 1)[1]: (prova / k.split("/", 1)[1]).read_bytes() for k in foto if k.startswith("cartella/")}


def _scrivi_atomico(p: Path, b: bytes) -> None:
    q = p.with_name(p.name + ".ripristino_tmp")
    q.write_bytes(b)
    os.replace(q, p)


# ⛔ 25/9 (verifica avversaria): il ripristino scattava SOLO su SystemExit. Ctrl-C (KeyboardInterrupt),
# un OSError, SIGTERM, SIGHUP e SIGKILL a meta' catena lasciavano valida e forgia_scorrevole con 3 patch
# su 7 — e la corsa dopo falliva con «patch gia' applicata» senza dire come tornare indietro. Ora:
#   - GIORNALE su disco (byte originali + md5 attesi) scritto PRIMA della catena, tolto solo alla fine;
#     se c'e' ancora (SIGKILL, macchina spenta) ogni corsa dopo RIFIUTA e --ripristina-giornale riporta
#     i bersagli ai byte originali;
#   - SIGTERM/SIGHUP diventano un'eccezione durante la catena, e il ripristino scatta su QUALUNQUE
#     eccezione (BaseException), poi la rilancia.
def dir_giornale(cartella: Path) -> Path:
    return Path(cartella) / "strumenti" / ".giornale_applica_patch_tempo"


def _scrivi_giornale(cartella: Path, originali: dict, uscita_prova: dict) -> None:
    d = dir_giornale(cartella)
    if d.exists():
        raise SystemExit(f"⛔ giornale di una catena interrotta presente in {d}: prima --ripristina-giornale")
    tmp = d.with_name(d.name + ".in_scrittura")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "originali").mkdir(parents=True)
    for n, b in originali.items():
        (tmp / "originali" / n).write_bytes(b)
    (tmp / "giornale.json").write_text(json.dumps({
        "quando": time.strftime("%F %T"), "pid": os.getpid(), "cartella": str(cartella),
        "md5_originali": {n: _md5(b) for n, b in originali.items()},
        "md5_attesi_dopo": {n: _md5(b) for n, b in uscita_prova.items()}, "ordine": list(ORDINE)}, indent=1))
    os.replace(tmp, d)                   # il giornale esiste tutto intero o non esiste


def leggi_giornale(cartella: Path) -> "dict | None":
    d = dir_giornale(cartella)
    if not d.exists():
        return None
    try:
        g = json.loads((d / "giornale.json").read_text())
    except (OSError, ValueError) as e:
        raise SystemExit(f"⛔ giornale {d} illeggibile ({e}): verifica a mano, i bersagli possono essere a meta'")
    g["originali"] = {n: (d / "originali" / n).read_bytes() for n in g["md5_originali"]}
    if any(_md5(b) != g["md5_originali"][n] for n, b in g["originali"].items()):
        raise SystemExit(f"⛔ giornale {d}: i byte originali salvati non corrispondono ai loro md5")
    return g


def _riporta(cartella: Path, originali: dict) -> "tuple[list, Path | None]":
    """Riporta ai byte originali i bersagli che ne differiscono; le versioni scartate in una cartella a parte."""
    dove, riportati = None, []
    for n in sorted(originali):
        f = Path(cartella) / n
        tmp_patch = f.with_name(f.name + ".patch_tmp")          # resto di una patch uccisa fra scrittura e rename
        if tmp_patch.exists():
            tmp_patch.unlink()
        if not f.exists() or f.read_bytes() != originali[n]:
            if dove is None:
                dove = Path(tempfile.mkdtemp(prefix="applica_patch_anomalia_"))
            if f.exists():
                shutil.copy2(f, dove / f.name)
            _scrivi_atomico(f, originali[n])
            riportati.append(n)
    return riportati, dove


class _Segnale(SystemExit):
    pass


def applica_vera(cartella: Path, foto: dict, uscita_prova: dict, catena_fn=catena) -> None:
    """Punti 3-4: ricontrollo (guardia + md5) subito prima, catena, confronto con la prova;
    su qualunque errore i file BERSAGLIO tornano ai byte originali (gli altri non si toccano mai)."""
    if dir_giornale(cartella).exists():
        raise SystemExit(f"⛔ giornale di una catena interrotta presente in {dir_giornale(cartella)}: "
                         "i bersagli possono essere a meta' — prima --ripristina-giornale")
    conflitti = processi_in_conflitto(cartella)
    if conflitti:
        print(testo_conflitti(conflitti))
        raise SystemExit("⛔ (ricontrollo) processi legati a questi file sono vivi o in coda: niente patch adesso")
    ora = fotografia(cartella)
    cambiati = sorted(set(foto) ^ set(ora)) + sorted(k for k in set(foto) & set(ora) if foto[k] != ora[k])
    if cambiati:
        raise SystemExit("⛔ file cambiati fra la prova generale e l'applicazione (md5 prova → ora): "
                         + "; ".join(f"{k} {_md5(foto[k]) if k in foto else '-'} → {_md5(ora[k]) if k in ora else '-'}"
                                     for k in cambiati) + " — rilancia la prova")
    originali = {k.split("/", 1)[1]: b for k, b in foto.items() if k.startswith("cartella/")}
    print(f"ricontrollo md5 superato ({len(foto)} file identici alla prova); applicazione a {cartella}:")

    def al_segnale(num, _frame):
        raise _Segnale(f"⛔ segnale {signal.Signals(num).name} durante la catena")
    vecchi = {s: signal.signal(s, al_segnale) for s in (signal.SIGTERM, signal.SIGHUP)}
    errore, eccezione = None, None
    try:
        _scrivi_giornale(cartella, originali, uscita_prova)
        try:
            catena_fn(cartella)
            finali = {n: (Path(cartella) / n).read_bytes() for n in originali}
            diversi = sorted(set(finali) ^ set(uscita_prova)) + sorted(
                n for n in set(finali) & set(uscita_prova) if finali[n] != uscita_prova[n])
            if diversi:
                errore = ("⛔ dopo la catena i file differiscono dall'uscita della prova: "
                          + ", ".join(diversi))
        except BaseException as e:  # noqa: BLE001 — Ctrl-C, segnali, OSError: si ripristina e si rilancia
            errore, eccezione = (str(e) or type(e).__name__), e
        if errore:
            _, dove = _riporta(cartella, originali)
        shutil.rmtree(dir_giornale(cartella))      # bersagli coerenti (tutti patchati o tutti originali)
    finally:
        for s, h in vecchi.items():
            signal.signal(s, h)
    if errore:
        msg = f"{errore}\n   ↩ file riportati ai byte originali" + (f"; le versioni scartate sono in {dove}" if dove else "")
        if eccezione is not None and not isinstance(eccezione, SystemExit):
            print(msg, file=sys.stderr)
            raise eccezione
        raise SystemExit(msg)


def ripristina_giornale(cartella: Path) -> int:
    g = leggi_giornale(cartella)
    if g is None:
        print(f"nessun giornale in {dir_giornale(cartella)}: niente da ripristinare")
        return 0
    conflitti = processi_in_conflitto(cartella)
    if conflitti:
        print(testo_conflitti(conflitti))
        raise SystemExit("⛔ processi legati a questi file sono vivi o in coda: niente ripristino adesso")
    stato = {n: (_md5((Path(cartella) / n).read_bytes()) if (Path(cartella) / n).exists() else "assente")
             for n in g["originali"]}
    print(f"giornale del {g['quando']} (pid {g['pid']}), catena {g['ordine']}:")
    for n, m in stato.items():
        quale = ("= originale" if m == g["md5_originali"][n] else "= atteso dopo la catena"
                 if m == g["md5_attesi_dopo"].get(n) else "A META' o sconosciuto")
        print(f"   {n}: {m[:8]} {quale}")
    riportati, dove = _riporta(cartella, g["originali"])
    shutil.rmtree(dir_giornale(cartella))
    print(f"✅ bersagli ai byte originali ({len(riportati)} riscritti: {riportati})"
          + (f"; le versioni scartate sono in {dove}" if dove else "") + "; giornale tolto")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prova", action="store_true", help="solo la prova generale su una copia")
    ap.add_argument("--solo-guardia", action="store_true", help="stampa i processi in conflitto e termina")
    ap.add_argument("--json", action="store_true", help="con --solo-guardia: uscita JSON")
    ap.add_argument("--cartella", type=Path, default=THOR_V2)
    ap.add_argument("--con-export", action="store_true",
                    help="in coda alla catena anche patch_tempo_export_incrementale (leva F, 25/9): stessa guardia, "
                         "stessa prova, stesso ricontrollo")
    ap.add_argument("--ripristina-giornale", action="store_true",
                    help="dopo una catena interrotta senza ripristino (SIGKILL, macchina spenta): riporta i bersagli "
                         "ai byte originali salvati nel giornale e lo toglie (con la stessa guardia)")
    args = ap.parse_args()
    if args.con_export:
        ORDINE.append("export_incrementale")
    cartella = args.cartella.resolve()
    if args.solo_guardia:
        c = processi_in_conflitto(cartella, dettagli=args.json)
        print(json.dumps({"io": os.getpid(), "padre": os.getppid(), "antenati": sorted(_antenati()),
                          "pipeline": sorted(_mia_pipeline()), "conflitti": c,
                          "giornale": dir_giornale(cartella).exists()}) if args.json
              else (testo_conflitti(c) if c else "nessun processo in conflitto")
              + (f"\n⛔ giornale di una catena interrotta: {dir_giornale(cartella)}" if dir_giornale(cartella).exists()
                 and not args.json else ""))
        return 1 if c or dir_giornale(cartella).exists() else 0
    if args.ripristina_giornale:
        return ripristina_giornale(cartella)
    if dir_giornale(cartella).exists():
        raise SystemExit(f"⛔ giornale di una catena interrotta presente in {dir_giornale(cartella)}: i bersagli "
                         "possono essere a meta' (alcune patch si', altre no). Prima: "
                         "python3 strumenti/applica_patch_tempo.py --ripristina-giornale")
    conflitti = processi_in_conflitto(cartella)
    if conflitti and not args.prova:
        print(testo_conflitti(conflitti))
        raise SystemExit("⛔ processi legati a questi file sono vivi o in coda: niente patch adesso")
    with tempfile.TemporaryDirectory(prefix="prova_patch_tempo_") as d:
        foto, uscita_prova = prova_generale(cartella, Path(d))
        print(f"✅ prova generale: le {len(ORDINE)} patch si applicano in catena")
        if args.prova:
            if conflitti:
                print("⚠️ (per l'applicazione vera oggi ci sarebbe un rifiuto: processi legati)\n" + testo_conflitti(conflitti))
            return 0
        applica_vera(cartella, foto, uscita_prova)
    print("✅ patch di velocita' applicate: file identici all'uscita della prova. "
          "Collaudo: strumenti/collaudo_patch_tempo.py --lavoro <scratch> --suite")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

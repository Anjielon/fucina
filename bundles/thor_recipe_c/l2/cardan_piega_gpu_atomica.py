"""cardan_piega_gpu.py con uscite TRANSAZIONALI (D6; necessario con la VEGLIA che uccide i lavoratori in stallo).

Stessa matematica e stessi argomenti di cardan_piega_gpu.py (che viene importato e chiamato, non copiato). Cambia solo:
- ogni .npy e' scritto in `<nome>.npy.tmp<pid>`, fsync, poi os.replace: un processo ucciso non lascia mai un .npy
  troncato che la ripresa (`exists()`) prenderebbe per FATTO; i .tmp non finiscono in `ls tensori/*.npy` ne' in patch;
- all'avvio, ogni .npy in tensori/ che non si apre o ha dimensione diversa da quella dichiarata dall'intestazione
  (scritto da una versione non transazionale e interrotto) viene RINOMINATO `*.troncato<ora>` (mai cancellato) e
  quindi rifatto.
- giro 7 (D1/D4, FORGIA_VERIFICA_ALGEBRA=1 di default, 0 = come prima): `cardan_verifica_algebra_gpu.installa`:
  fattori Cholesky GPU controllati con Freivalds fp64 (le matrici sospette o «non SPD» rifatte da LAPACK fp64 CPU,
  le buone restano AL BIT), trasferimento delle Hessiane controllato con somma esatta dei bit, normalizzazione con
  media della diagonale allo stesso passo dell'originale (H identica al bit). Contatori a fine corsa su stderr.
"""
from __future__ import annotations

import os
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"  # priorita' 0 (21/10): ogni worker inductor apre /dev/kfd e prende una coda SDMA
import sys
import time
import builtins
import contextlib
import fcntl
from pathlib import Path

import numpy as np

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import cardan_piega_gpu as P  # noqa: E402
import cardan_babai_triton_gpu as _BT  # noqa: E402
_BT.installa(P)                                                    # FORGIA_BABAI_TRITON=1: ciclo colonne Triton


def salva_atomico(file, arr, allow_pickle=True, fix_imports=True):
    p = Path(file)
    if p.suffix != ".npy":
        p = p.with_name(p.name + ".npy")
    tmp = p.with_name(f"{p.name}.tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        np.lib.format.write_array(f, np.asanyarray(arr), allow_pickle=allow_pickle)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)
    import pianificatore_gpu as PG
    PG.sincronizza_dir(p.parent)


def npy_integro(p: Path) -> bool:
    try:
        with open(p, "rb") as f:
            ver = np.lib.format.read_magic(f)
            leggi = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}[ver]
            forma, fortran, dtype = leggi(f)
            inizio = f.tell()
        attesa = inizio + int(np.prod(forma, dtype=np.int64)) * np.dtype(dtype).itemsize
        return p.stat().st_size == attesa
    except Exception:
        return False


def _recente(p: Path) -> bool:
    """D29: file toccato da poco = forse lo scrive adesso un processo (anche non transazionale): non si tocca."""
    try:
        return time.time() - p.stat().st_mtime < float(os.environ.get("FORGIA_ATOMICA_MTIME_S", "300"))
    except OSError:
        return True


def _opzione(argv, nome):
    for i, arg in enumerate(argv):
        if arg == nome and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(nome + "="):
            return arg[len(nome) + 1:]
    return None


def _scrittore_vivo(out: Path, strati_ok=None, proc=Path("/proc")) -> list:
    """D29: pid vivi (diversi dal mio) che hanno `--out <out>` nella riga di comando."""
    mio, trovati, o = os.getpid(), [], str(out.resolve())
    for d in proc.iterdir():
        if not d.name.isdigit() or int(d.name) == mio:
            continue
        try:
            a = (d / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        a = [x.decode(errors="replace") for x in a]
        destinazione = _opzione(a, "--out")
        if not destinazione or not any(Path(x).name.startswith("cardan_piega") and x.endswith(".py") for x in a):
            continue
        p = Path(destinazione)
        if not p.is_absolute():
            try:
                p = (d / "cwd").resolve(strict=True) / p
            except OSError:
                trovati.append(int(d.name))
                continue
        if str(p.resolve()) != o:
            continue
        intervallo = _opzione(a, "--strati")
        if strati_ok is not None and intervallo:
            try:
                lo, _, hi = intervallo.partition("-")
                altri = set(range(int(lo), int(hi or lo) + 1))
            except ValueError:
                altri = None
            if altri is not None and altri.isdisjoint(strati_ok):
                continue
        trovati.append(int(d.name))
    return trovati


def scarta_troncati(td: Path, strati_ok=None, scrittore_escluso=False) -> list:
    scartati = []
    for p in sorted(td.glob("*.npy")):
        if strati_ok is not None and (not p.name.startswith("blk.") or
                                      int(p.name.split(".")[1]) not in strati_ok):
            continue
        if not scrittore_escluso and _recente(p):
            continue
        if not npy_integro(p):
            q = p.with_name(f"{p.name}.troncato{time.strftime('%H%M%S')}")
            os.replace(p, q)
            scartati.append(q.name)
    return scartati


def pulisci_tmp_orfani(td: Path) -> list:
    """Rimuove solo i `*.npy.tmp<pid>` il cui processo non esiste piu' (residui di scrittura interrotta, non lavoro)."""
    tolti = []
    for p in td.glob("*.npy.tmp*"):
        try:
            pid = int(p.name.rsplit(".tmp", 1)[1])
        except ValueError:
            continue
        if not Path(f"/proc/{pid}").exists():
            try:
                p.unlink()
                tolti.append(p.name)
            except OSError:
                pass
    return tolti


def _righe_registro(reg: Path) -> list:
    import json
    righe = []
    if reg.exists():
        for r in reg.read_text().splitlines():
            try:
                righe.append(json.loads(r))
            except ValueError:
                # D28: riga troncata fusa con la successiva (kill senza newline): si recupera l'ultimo oggetto intero
                i = r.rfind('{"')
                while i > 0:
                    try:
                        righe.append(json.loads(r[i:]))
                        break
                    except ValueError:
                        i = r.rfind('{"', 0, i)
    return righe


def riconcilia_registro(out: Path, strati_ok=None) -> list:
    """D6 registro: un .npy senza riga di registro (kill fra np.save e reg.write) viene rinominato `*.senza_riga<ora>`
    cosi' la ripresa rifa' la famiglia. Solo se nessun altro lavoratore vivo sta scrivendo (lo strato e' dell'altro)."""
    if _scrittore_vivo(out, strati_ok):
        return []
    td = out / "tensori"
    chiavi = {(r["strato"], r["famiglia"]) for r in _righe_registro(out / "registro.jsonl")}
    spostati = []
    import re
    for p in sorted(td.glob("blk.*.ffn_*_exps.weight.npy")):
        m = re.match(r"blk\.(\d+)\.ffn_(gate|up)_exps\.weight\.npy$", p.name)
        if not m:
            continue
        L, f = int(m.group(1)), m.group(2)
        if strati_ok is not None and L not in strati_ok:
            continue
        if (L, f) not in chiavi:
            q = p.with_name(f"{p.name}.senza_riga{time.strftime('%H%M%S')}")
            os.replace(p, q)
            spostati.append(q.name)
            e2 = p.with_name(p.name.replace("_exps.weight", "_exps2.weight"))
            if e2.exists():
                os.replace(e2, e2.with_name(f"{e2.name}.senza_riga{time.strftime('%H%M%S')}"))
    return spostati


def _scrivi_registro_unico(out: Path) -> int:
    """registro_unico.jsonl: una riga per (strato, famiglia), l'ultima vince; scrittura atomica."""
    import json
    uniche = {}
    for r in _righe_registro(out / "registro.jsonl"):
        uniche[(r["strato"], r["famiglia"])] = r
    dest = out / "registro_unico.jsonl"
    tmp = dest.with_name(f"{dest.name}.tmp{os.getpid()}")
    with open(tmp, "w") as f:
        for k in sorted(uniche):
            f.write(json.dumps(uniche[k]) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dest)
    import pianificatore_gpu as PG
    PG.sincronizza_dir(dest.parent)
    return len(uniche)


def scrivi_registro_unico(out: Path) -> int:
    with builtins.open(out / "registro.jsonl", "a+b") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            return _scrivi_registro_unico(out)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def ripara_fine_riga(reg: Path) -> bool:
    """D28: se il registro non finisce con newline (kill a meta' riga) ne scrive uno prima di aprire in append."""
    try:
        if reg.exists() and reg.stat().st_size:
            with open(reg, "rb") as f:
                f.seek(-1, 2)
                if f.read(1) != b"\n":
                    with open(reg, "ab") as g:
                        g.write(b"\n")
                    return True
    except OSError:
        pass
    return False


def _unico(argv):
    out = _opzione(argv, "--out")
    if out:
        P.log(f"registro_unico.jsonl: {scrivi_registro_unico(Path(out))} famiglie")


class RegistroDurevole:
    """Serializza le righe tra processi, inclusa la riparazione del newline."""
    def __init__(self, path):
        self.file = builtins.open(path, "a+b", buffering=0)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.file.close()

    def write(self, testo):
        f = self.file
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            if f.seek(0, os.SEEK_END):
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    f.write(b"\n")
            dati = testo.encode("utf-8")
            while dati:
                n = f.write(dati)
                if not n:
                    raise OSError("scrittura registro incompleta")
                dati = dati[n:]
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
        return len(testo)

    def flush(self):
        os.fsync(self.file.fileno())


def apri_durevole(file, mode="r", *args, **kwargs):
    if mode == "a" and Path(file).name == "registro.jsonl":
        return RegistroDurevole(file)
    return builtins.open(file, mode, *args, **kwargs)


def _main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    out = _opzione(argv, "--out")
    if out:
        td = Path(out) / "tensori"
        lo, _, hi = (_opzione(argv, "--strati") or "0-39").partition("-")
        mio = set(range(int(lo), int(hi or lo) + 1))
        vivi = _scrittore_vivo(Path(out), mio)
        if vivi:
            raise RuntimeError(f"scrittore vivo sugli stessi strati/--out (pid {vivi}): nessun file modificato (D29)")
        if td.is_dir() and not vivi:
            s = scarta_troncati(td, mio, scrittore_escluso=True)
            if s:
                P.log(f"npy troncati rinominati (rifatti): {s}")
            P.log(f"tmp orfani tolti: {pulisci_tmp_orfani(td)}")
            if os.environ.get("FORGIA_RICONCILIA_REGISTRO", "1") == "1":
                sr = riconcilia_registro(Path(out), mio)
                if sr:
                    P.log(f"npy senza riga di registro rinominati (rifatti): {sr}")
    np.save = salva_atomico
    P.open = apri_durevole
    if os.environ.get("FORGIA_VERIFICA_ALGEBRA", "1") == "1":
        import cardan_verifica_algebra_gpu as VA
        P.log(f"verifica algebra installata: {VA.installa(P)}")
        try:
            return P.main(argv)
        finally:
            P.log(f"verifica algebra, contatori: {VA.CONTATORI}")
            _unico(argv)
    try:
        return P.main(argv)
    finally:
        _unico(argv)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    out = _opzione(argv, "--out")
    if out is None:
        return _main(argv)
    lo, _, hi = (_opzione(argv, "--strati") or "0-39").partition("-")
    minimo, massimo = int(lo), int(hi or lo)
    if not 0 <= minimo <= massimo <= 39:
        raise ValueError("--strati deve essere un intervallo crescente tra 0 e 39")
    strati = set(range(minimo, massimo + 1))
    vivi = _scrittore_vivo(Path(out), strati)
    if vivi:
        raise RuntimeError(f"scrittore vivo sugli stessi strati/--out (pid {vivi}): nessun file modificato")
    import pianificatore_gpu as PG
    Path(out).mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as stack:
        for L in sorted(strati):
            stack.enter_context(PG.lucchetto(Path(out) / f".piega_strato_{L}.lock"))
        return _main(argv)


if __name__ == "__main__":
    main()

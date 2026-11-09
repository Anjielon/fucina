#!/usr/bin/env python3
"""MAGAZZINO della cache per lotto (THOR_CACHE_LOTTI): l'I/O di UNA corsa.

25/9/2026, progetto `PROGETTO_CACHE_LOTTI.md` (§3.2, §3.5-§3.8, §8). La faccia che
vedono valida e forgia e' `deposito_lotti.py` (decide DOVE sta un valore); qui si
fanno soltanto i byte (D3 del playbook `spacchettare`: l'I/O isolato da chi decide).

## Perche' esiste
La forgia scorrevole a 150+1 lotti terrebbe sulla GPU ~77,5 GiB riservati, + 33 della
casa = 110 > 95,15: oltre, i byte scivolano in GTT e costano RAM host 1:1. Lo stato per
lotto che non sta nel budget va su disco (NVMe, mai /tmp che e' RAM) in BYTE GREZZI:
nessun cast, nessun ricalcolo, ogni tensore riletto e' il tensore scritto, al bit.

## Cosa fa (una corsa = una cartella)
- cartella `<radice>/<corsa>-<pid>-<uuid>/` con MARCATORE (numero magico, pid, istante di
  avvio del processo) e LUCCHETTO `fcntl.flock` tenuto per tutta la corsa. All'apertura
  si cancellano SOLO le cartelle col marcatore e il lucchetto libero (processo morto: il
  kernel rilascia il flock); una radice con voci senza marcatore, o che contiene/sta
  dentro l'uscita o la ripresa, e' RIFIUTATA (critica 3: con la radice sul genitore di
  OUT una cartella `v2-Thor-150` sarebbe sembrata l'orfana del pid 150). Il nome porta un
  uuid: due corse nello stesso processo non condividono mai la cartella (critica 4).
  Si toglie anche una `.nascita-*` interrotta (senza marcatore, pid del nome morto, dentro al
  piu' lucchetto e marcatore); il lanciatore chiama la stessa pulizia prima dei `df`
  (`pulisci_radice`): revisione risorse del 25/9, difetti 2 e 3.
- un file per tensore: 4096 byte di intestazione JSON + i byte dello storage, riempiti a
  multipli di 4096. `O_DIRECT` su buffer mmap allineati (la page cache sull'UMA e' RAM
  della casa, lezione /tmp del 22/9); se il filesystem lo rifiuta (EINVAL) si ripiega su
  fdatasync + posix_fadvise(DONTNEED), scritto nel log.
- UN SOLO thread di I/O, FIFO (critica 9): scritture, letture (precarica) e cancellazioni
  passano dalla stessa coda, quindi una lettura viene sempre dopo la sua scrittura e una
  cancellazione sempre dopo le letture chieste prima. Nessuna gara fra thread da gestire:
  l'ordine e' per costruzione. Nessuna operazione GPU nel thread di I/O.
- conteggio dei riferimenti per file; il file a zero si cancella in coda (dopo la sua
  scrittura pendente); una precarica rilasciata si annulla o si butta, e i suoi errori
  si scartano.
- PROVENIENZA senza callback (critica 10): registro `id(t) -> (weakref, file, _version)`
  e per file l'ultimo oggetto vivo. Nessuna callback dei weakref (girerebbero nel thread
  che rilascia l'ultimo riferimento, anche dentro una sezione critica): le voci morte si
  riconoscono alla lettura (`ref() is not t`) e si potano a intervalli.
- IMPRONTA intera per tensore, calcolata sul device del tensore a blocchi di dimensione
  fissa (critica 16: temporanei <= 64 MiB, mai 4-6x il tensore) e ricontrollata a ogni
  rilettura. `VERIFICA=completa` tiene anche una copia ombra in RAM host.
- disco pieno: controllo PRIMA di ogni scrittura (`DiscoPienoCache`); ENOSPC/EIO nel
  thread di I/O -> file parziale rimosso, errore conservato e risollevato alla prima
  operazione successiva. MAI un ripiego silenzioso sulla GPU.

## Limiti dichiarati (critica 13)
Una modifica SUL POSTO di un tensore dopo il deposito si vede solo finche' l'oggetto e'
vivo (`_version`, e le scritture via `.data` o `untyped_storage()` non lo cambiano
nemmeno). Un tensore riletto da disco, modificato e poi morto, alla rilettura dopo ridara'
i byte ORIGINALI: il contratto dei chiamanti e' non modificare mai sul posto questi valori
(verificato il 25/9 in valida e finestra_globale). Le prove spenta == accesa lo
controllano per costruzione.
"""
from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import mmap
import os
import shutil
import threading
import time
import uuid
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

MAGIA = "thor-cache-lotti-v1"
MARCATORE = ".marcatore_cache_lotti.json"
LUCCHETTO = ".lucchetto"
NASCITA = ".nascita-"            # cartella in costruzione: marcatore + lucchetto PRIMA del rename atomico
TESTA = 4096                     # intestazione; anche l'allineamento O_DIRECT
RIGA = 1 << 16                   # elementi per riga dell'impronta (nessun overflow int64, vedi impronta)
RIGHE_PER_BLOCCO = 64            # 2^22 elementi per blocco: temporanei int64 <= 64 MiB
CODA_SCRITTURA_MAX = 256 << 20   # byte in coda di scrittura (contropressione: RAM host limitata)
PRECARICA_MAX = 512 << 20        # byte di precarica in volo
POOL_INATTIVO_MAX = 1 << 30      # buffer host inattivi tenuti per il riuso
OMBRA_MAX = 4 << 30              # tetto della copia ombra di VERIFICA=completa
_FS_IN_RAM = {"tmpfs", "ramfs", "devtmpfs"}


def log(*a) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


class DiscoPienoCache(OSError):
    """Spazio insufficiente per la cache per lotto: la forgia si FERMA (il punto di ripresa
    dell'ultima finestra completa e' intatto). Mai un ripiego sulla GPU."""


def _arrotonda(n: int) -> int:
    return (n + TESTA - 1) // TESTA * TESTA


def tipo_fs(percorso: Path) -> str:
    """Tipo del filesystem che contiene `percorso` (il punto di montaggio piu' lungo)."""
    p = os.path.realpath(percorso)
    migliore, tipo = "", "?"
    with open("/proc/self/mounts") as f:
        for riga in f:
            campi = riga.split()
            if len(campi) < 3:
                continue
            m = campi[1].replace("\\040", " ")
            if (p == m or p.startswith(m.rstrip("/") + "/")) and len(m) > len(migliore):
                migliore, tipo = m, campi[2]
    return tipo


def _avvio_processo(pid: int) -> "int | None":
    """Istante di avvio del processo (campo 22 di /proc/pid/stat, in tick): distingue un pid riusato."""
    try:
        testo = Path(f"/proc/{pid}/stat").read_text()
        return int(testo[testo.rindex(")") + 2:].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def _pesi(device, _cache={}) -> torch.Tensor:  # noqa: B006 — cache per device, voluta
    chiave = str(device)
    if chiave not in _cache:
        _cache[chiave] = torch.arange(1, RIGA + 1, dtype=torch.int64, device=device)
    return _cache[chiave]


def impronta(t: torch.Tensor) -> str:
    """Impronta INTERA dei byte di `t` (denso), calcolata sul suo device.

    Parole a 32 bit (a 8 se i byte non sono multipli di 4) in int64, a righe di 2^16:
    per riga somma e somma pesata (peso 1..2^16). |parola| < 2^31 e somma dei pesi < 2^31
    -> ogni somma di riga < 2^62: aritmetica intera ESATTA, in qualunque ordine la GPU
    riduca. Le coppie di tutte le righe vanno in uno sha256 (sul CPU, pochi KiB). Blocchi
    di 64 righe: temporanei <= 64 MiB qualunque sia la taglia del tensore."""
    n_byte = t.numel() * t.element_size()
    h = hashlib.sha256(f"{n_byte}:{t.dtype}:{tuple(t.shape)}".encode())
    if n_byte == 0:
        return h.hexdigest()
    byte = t.detach().reshape(-1).view(torch.uint8)
    parole = byte.view(torch.int32) if n_byte % 4 == 0 else byte
    n = parole.numel()
    w = _pesi(t.device)
    passo = RIGA * RIGHE_PER_BLOCCO
    righe = []
    for i0 in range(0, n, passo):
        pezzo = parole[i0:i0 + passo].to(torch.int64)
        m = pezzo.numel()
        pieno = (m // RIGA) * RIGA
        if pieno:
            mat = pezzo[:pieno].view(-1, RIGA)
            righe.append(torch.stack((mat.sum(1), (mat * w).sum(1)), 1))
            del mat
        if m > pieno:
            coda = pezzo[pieno:]
            righe.append(torch.stack((coda.sum(), (coda * w[:m - pieno]).sum())).view(1, 2))
        del pezzo
    h.update(torch.cat(righe).cpu().numpy().tobytes())
    return h.hexdigest()


def libera_pagine(percorso: Path) -> None:
    """Pagine di `percorso` fuori dalla page cache (fdatasync prima: le pagine sporche non
    si scartano). Per il punto di ripresa a interruttore acceso (P4, critica 15)."""
    fd = os.open(percorso, os.O_RDONLY)
    try:
        os.fdatasync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


# ─────────────── radice e orfani (funzioni del modulo: le usa anche il lanciatore) ───────────────
# 25/9 (revisione risorse, difetto 2): erano metodi del Magazzino, quindi la pulizia girava solo
# DENTRO la forgia, dopo i `df` del lanciatore: una cartella orfana (forgia uccisa con SIGKILL,
# OOM) contava come spazio occupato e poteva far rifiutare una partenza che la forgia stessa
# avrebbe liberato. Ora `pulisci_radice` e' lo stesso giudice, chiamabile prima di contare.
def controlla_radice(radice: Path, protette=()) -> None:
    radice = Path(radice).absolute()
    tipo = tipo_fs(radice if radice.exists() else radice.parent)
    if tipo in _FS_IN_RAM:
        raise ValueError(f"THOR_CACHE_LOTTI_DIR={radice}: filesystem {tipo} (RAM): la cache va su disco "
                         "(@@ROOT:models@@/gguf/tony-tern/...), mai /tmp")
    r = Path(os.path.realpath(radice))
    for p in protette:
        if p is None:
            continue
        q = Path(os.path.realpath(p))
        if r == q or q in r.parents or r in q.parents:
            raise ValueError(f"THOR_CACHE_LOTTI_DIR={radice} e {q} si contengono: la pulizia della cache "
                             "non deve mai poter toccare l'uscita o il punto di ripresa")


def _leggi_marcatore(cartella: Path) -> "dict | None":
    try:
        m = json.loads((cartella / MARCATORE).read_text())
    except (OSError, ValueError):
        return None
    return m if isinstance(m, dict) and m.get("magia") == MAGIA else None


def _pid_vivo(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True                      # esiste, di un altro utente
    return True


def _nascita_interrotta(voce: Path) -> "str | None":
    """25/9 (revisione risorse, difetto 3): `.nascita-<corsa>-<pid>-<uuid>` SENZA marcatore valido
    = processo morto fra il mkdir e la scrittura del marcatore (SIGKILL nella finestra, o marcatore
    scritto a meta'). E' nostra solo se il nome ha la forma di `_nasci` e dentro ci sono al piu' il
    lucchetto e il marcatore (file normali): allora "orfana" se il pid del nome e' morto e il
    lucchetto (se c'e') si lascia prendere, "viva" se no (una nascita in corso). None = non nostra:
    resta il rifiuto rumoroso, la cache non cancella cio' che non riconosce."""
    parti = voce.name[len(NASCITA):].rsplit("-", 2)
    if (len(parti) != 3 or not parti[1].isdigit() or int(parti[1]) <= 0 or len(parti[2]) != 12
            or any(c not in "0123456789abcdef" for c in parti[2])):
        return None
    if any(p.name not in (LUCCHETTO, MARCATORE) or p.is_symlink() or not p.is_file() for p in voce.iterdir()):
        return None
    if _pid_vivo(int(parti[1])):
        return "viva"
    try:
        fd = os.open(voce / LUCCHETTO, os.O_RDWR)
    except FileNotFoundError:
        return "orfana"                  # morto prima di creare il lucchetto
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return "viva"                    # lucchetto ereditato da un figlio ancora vivo
    finally:
        os.close(fd)
    return "orfana"


def pulisci_orfani(radice: Path) -> None:
    """Toglie le cartelle col marcatore e il lucchetto libero (processo morto: il kernel rilascia il
    flock) e le nascite interrotte; rifiuta una radice con voci che non riconosce."""
    sconosciute, vive = [], []
    for voce in sorted(Path(radice).iterdir()):
        if not voce.is_dir() or voce.is_symlink():
            sconosciute.append(voce.name)
            continue
        if _leggi_marcatore(voce) is None:
            esito = _nascita_interrotta(voce) if voce.name.startswith(NASCITA) else None
            if esito == "viva":
                vive.append(voce.name)
            elif esito == "orfana":
                log(f"cache lotti: tolgo la nascita interrotta {voce.name} (senza marcatore, processo morto)")
                shutil.rmtree(voce)
            else:
                sconosciute.append(voce.name)
            continue
        try:
            fd = os.open(voce / LUCCHETTO, os.O_RDWR)
        except FileNotFoundError:
            sconosciute.append(voce.name)
            continue
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                vive.append(voce.name)          # corsa viva (anche di questo processo): non si tocca
                continue
            byte = sum(f.stat().st_size for f in voce.rglob("*") if f.is_file())
            log(f"cache lotti: tolgo la cartella orfana {voce.name} ({byte / 2**30:.2f} GiB, "
                f"marcatore {_leggi_marcatore(voce)})")
            shutil.rmtree(voce)
        finally:
            os.close(fd)
    if sconosciute:
        raise RuntimeError(
            f"THOR_CACHE_LOTTI_DIR={radice} contiene voci che non sono cartelle della cache "
            f"({sconosciute[:8]}{' ...' if len(sconosciute) > 8 else ''}): radice sbagliata o dati d'altro. "
            "La cache non pulisce cio' che non ha il suo marcatore: scegliere una radice dedicata.")
    if vive:
        log(f"⚠️ cache lotti: {len(vive)} corse vive nella stessa radice ({vive[:4]}): non le tocco")


def pulisci_radice(radice, protette=()) -> None:
    """Per il lanciatore, PRIMA dei `df` (difetto 2): stessi controlli e stessa pulizia che la
    forgia fa all'apertura della corsa. Radice assente: nulla da pulire."""
    r = Path(radice).absolute()
    controlla_radice(r, protette)
    if r.is_dir():
        pulisci_orfani(r)


class _Pool:
    """Buffer host mmap anonimi (allineati alla pagina, come vuole O_DIRECT), riusati per taglia."""

    def __init__(self):
        self._liberi: dict = {}
        self._byte = 0
        self._chiuso = False
        self._lock = threading.Lock()

    def prendi(self, n: int) -> mmap.mmap:
        with self._lock:
            lista = self._liberi.get(n)
            if lista:
                self._byte -= n
                return lista.pop()
        return mmap.mmap(-1, n)

    def rendi(self, mm: mmap.mmap) -> None:
        n = len(mm)
        with self._lock:
            if not self._chiuso and self._byte + n <= POOL_INATTIVO_MAX:   # chiuso: una precarica tardiva non resta appesa
                self._liberi.setdefault(n, []).append(mm)
                self._byte += n
                return
        mm.close()

    def chiudi(self) -> None:
        with self._lock:
            self._chiuso = True
            for lista in self._liberi.values():
                for mm in lista:
                    mm.close()
            self._liberi.clear()
            self._byte = 0


class _File:
    __slots__ = ("fid", "percorso", "n_byte", "dimensione", "dtype", "forma", "stride", "device", "impronta",
                 "nome", "lotto", "ref", "scrittura", "precarica", "errore", "vivo", "versione")

    def __init__(self, fid, percorso, t: torch.Tensor, imp: str, nome: str, lotto):
        self.fid, self.percorso = fid, percorso
        self.n_byte = t.numel() * t.element_size()
        self.dimensione = TESTA + _arrotonda(self.n_byte)
        self.dtype, self.forma, self.stride, self.device = t.dtype, tuple(t.shape), tuple(t.stride()), t.device
        self.impronta, self.nome, self.lotto = imp, nome, lotto
        self.ref = 1
        self.scrittura = None     # Future della scrittura
        self.precarica = None     # Future della lettura anticipata (-> mmap)
        self.errore = None        # errore della scrittura (dal thread di I/O)
        self.vivo = None          # weakref all'ultimo oggetto (scritto o riletto) di questo file
        self.versione = None      # suo _version


class Magazzino:
    """I file di una corsa. Metodi pubblici chiamati dal thread principale, tranne
    `leggi(..., registra=False)` che e' sicuro anche dal thread del punto di ripresa."""

    def __init__(self, radice: Path, corsa: str, protette=(), riserva_byte: int = 8 << 30,
                 verifica: str = "impronta"):
        if verifica not in ("impronta", "completa"):
            raise ValueError(f"THOR_CACHE_LOTTI_VERIFICA={verifica!r}: attesi 'impronta' o 'completa'")
        self.radice = Path(radice).absolute()
        self.riserva = int(riserva_byte)
        self.verifica = verifica
        controlla_radice(self.radice, protette)
        self.radice.mkdir(parents=True, exist_ok=True)
        pulisci_orfani(self.radice)
        self.cartella, self._fd_lucchetto = self._nasci(corsa)
        self._diretto = self._prova_o_direct()
        self._lock = threading.RLock()
        self._cond = threading.Condition()
        self._in_coda = 0                   # byte in coda di scrittura
        self._byte_precarica = 0
        self._files: dict = {}
        self._per_id: dict = {}              # id(tensore) -> (weakref, fid, _version)
        self._inserimenti = 0
        self._prossimo = 0
        self._errore = None
        self._chiuso = False
        self._ombra: dict = {}
        self._byte_ombra = 0
        self._pool = _Pool()
        self._io = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cache-lotti-io")
        self.contatori = {"scritti": 0, "letti": 0, "file_scritti": 0, "file_letti": 0, "riusi_provenienza": 0,
                          "oggetti_vivi_riusati": 0, "precariche_usate": 0, "precariche_buttate": 0,
                          "vivi_max": 0, "byte_vivi_max": 0}
        self._byte_vivi = 0
        log(f"cache lotti: corsa {self.cartella.name} in {self.radice} · I/O "
            + ("O_DIRECT" if self._diretto else "BUFFERIZZATO (O_DIRECT rifiutato: fdatasync + fadvise)")
            + f" · riserva {self.riserva / 2**30:.1f} GiB · verifica {self.verifica}")

    # ─────────────── nascita (radice e orfani: funzioni del modulo, sopra) ───────────────
    def _nasci(self, corsa: str) -> "tuple[Path, int]":
        pulito = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(corsa))[:60] or "corsa"
        nome = f"{pulito}-{os.getpid()}-{uuid.uuid4().hex[:12]}"
        tmp = self.radice / f"{NASCITA}{nome}"
        tmp.mkdir()
        fd = None
        try:
            fd = os.open(tmp / LUCCHETTO, os.O_RDWR | os.O_CREAT, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            (tmp / MARCATORE).write_text(json.dumps({
                "magia": MAGIA, "pid": os.getpid(), "avvio_processo": _avvio_processo(os.getpid()),
                "corsa": str(corsa), "creata": time.strftime("%F %T")}))
            finale = self.radice / nome
            os.rename(tmp, finale)
        except BaseException:
            # 25/9 (revisione risorse, difetto 3): una nascita a meta' (ENOSPC/EDQUOT sul marcatore)
            # lasciava `.nascita-*` senza marcatore, che ogni avvio dopo rifiutava come «dati d'altro»
            # fino a una cancellazione a mano. Si toglie qui, lucchetto chiuso; il guasto vero risale.
            if fd is not None:
                os.close(fd)
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return finale, fd

    def _prova_o_direct(self) -> bool:
        """O_DIRECT accettato dal filesystem della cartella? (EINVAL -> ripiego bufferizzato, scritto nel log)"""
        p = self.cartella / ".prova_o_direct"
        mm = mmap.mmap(-1, TESTA)
        fd = None
        try:
            fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o600)
            with memoryview(mm) as mv:
                os.pwritev(fd, [mv], 0)
            return True
        except OSError as e:
            if e.errno != errno.EINVAL:
                raise
            return False
        finally:
            if fd is not None:
                os.close(fd)
            mm.close()
            p.unlink(missing_ok=True)

    # ─────────────── thread di I/O (nessuna operazione torch qui) ───────────────
    def _apri_fd(self, percorso: Path, scrittura: bool) -> int:
        flag = (os.O_WRONLY | os.O_CREAT | os.O_TRUNC) if scrittura else os.O_RDONLY
        return os.open(percorso, flag | (os.O_DIRECT if self._diretto else 0), 0o600)

    def _job_scrivi(self, f: _File, mm: mmap.mmap) -> None:
        try:
            fd = self._apri_fd(f.percorso, True)
            try:
                with memoryview(mm) as mv:
                    fatto = 0
                    while fatto < f.dimensione:
                        k = os.pwritev(fd, [mv[fatto:f.dimensione]], fatto)
                        if k <= 0:
                            raise OSError(errno.EIO, f"scrittura ferma a {fatto}/{f.dimensione} byte")
                        fatto += k
                if not self._diretto:
                    os.fdatasync(fd)
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
            with self._lock:
                self.contatori["scritti"] += f.dimensione
                self.contatori["file_scritti"] += 1
        except BaseException as e:           # ENOSPC/EIO: file parziale via, errore conservato
            f.errore = e
            try:
                f.percorso.unlink()
            except FileNotFoundError:
                pass
            with self._lock:
                if self._errore is None:
                    self._errore = e
        finally:
            self._pool.rendi(mm)
            with self._cond:
                self._in_coda -= f.dimensione
                self._cond.notify_all()

    def _job_leggi(self, f: _File) -> mmap.mmap:
        if f.errore is not None:
            raise RuntimeError(f"cache lotti: la scrittura di {f.percorso.name} ({f.nome}, lotto {f.lotto}) "
                               f"e' fallita: {f.errore!r}") from f.errore
        mm = self._pool.prendi(f.dimensione)
        try:
            fd = self._apri_fd(f.percorso, False)
            try:
                with memoryview(mm) as mv:
                    fatto = 0
                    while fatto < f.dimensione:
                        k = os.preadv(fd, [mv[fatto:f.dimensione]], fatto)
                        if k <= 0:
                            raise OSError(errno.EIO, f"{f.percorso.name}: letti {fatto}/{f.dimensione} byte "
                                                     "(file troncato)")
                        fatto += k
                if not self._diretto:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except BaseException:
            self._pool.rendi(mm)
            raise
        with self._lock:
            self.contatori["letti"] += f.dimensione
            self.contatori["file_letti"] += 1
        return mm

    @staticmethod
    def _job_cancella(percorso: Path) -> None:
        try:
            percorso.unlink()
        except FileNotFoundError:
            pass

    # ─────────────── servizio ───────────────
    def controlla(self) -> None:
        if self._chiuso:
            raise RuntimeError("cache lotti: corsa gia' chiusa")
        if self._errore is not None:
            e = self._errore
            raise OSError(getattr(e, "errno", errno.EIO),
                          f"cache lotti: errore nel thread di I/O ({e!r}) — forgia fermata, il punto di ripresa "
                          "dell'ultima finestra completa e' intatto") from e

    def _controlla_spazio(self, n: int) -> None:
        st = os.statvfs(self.cartella)
        libero = st.f_bavail * st.f_frsize
        if libero - self._in_coda < n + self.riserva:
            raise DiscoPienoCache(
                errno.ENOSPC,
                f"cache lotti: {self.cartella} ha {libero / 2**30:.2f} GiB liberi ({self._in_coda / 2**30:.2f} in coda), "
                f"ne servono {n / 2**30:.3f} + riserva {self.riserva / 2**30:.1f}: forgia fermata, il punto di ripresa "
                "dell'ultima finestra completa e' intatto")

    def spazio_libero(self) -> int:
        st = os.statvfs(self.cartella)
        return st.f_bavail * st.f_frsize

    def _registra(self, t: torch.Tensor, f: _File) -> None:
        ref = weakref.ref(t)
        self._per_id[id(t)] = (ref, f.fid, t._version)
        f.vivo, f.versione = ref, t._version
        self._inserimenti += 1
        if self._inserimenti % 256 == 0:
            morti = [k for k, (r, _, _) in self._per_id.items() if r() is None]
            for k in morti:
                del self._per_id[k]

    def trova(self, t: torch.Tensor) -> "int | None":
        """File che contiene gia' `t` (stesso oggetto, `_version` invariato), o None."""
        with self._lock:
            voce = self._per_id.get(id(t))
            if voce is None or voce[0]() is not t or voce[1] not in self._files:
                return None
            if t._version != voce[2]:
                return None                      # modificato sul posto: si riscrive
            self.contatori["riusi_provenienza"] += 1
            return voce[1]

    def scrivi(self, t: torch.Tensor, nome: str, lotto) -> int:
        """Nuovo file con i byte di `t` (denso, gia' controllato dal chiamante). Ritorna il fid
        con UN riferimento. Copia device->host SINCRONA qui: al ritorno `t` si puo' liberare."""
        self.controlla()
        n_byte = t.numel() * t.element_size()
        dimensione = TESTA + _arrotonda(n_byte)
        self._controlla_spazio(dimensione)
        imp = impronta(t)
        with self._lock:
            fid = self._prossimo
            self._prossimo += 1
        f = _File(fid, self.cartella / f"f{fid:07d}.bin", t, imp, nome, lotto)
        testa = json.dumps({"magia": MAGIA, "fid": fid, "n_byte": n_byte, "dtype": str(t.dtype),
                            "forma": list(f.forma), "stride": list(f.stride), "impronta": imp, "nome": nome,
                            "lotto": lotto}).encode()
        if len(testa) > TESTA:
            raise ValueError(f"intestazione di {len(testa)} byte > {TESTA}")
        mm = self._pool.prendi(dimensione)
        try:
            mm[:TESTA] = testa + b"\0" * (TESTA - len(testa))
            if n_byte:
                vista = torch.frombuffer(mm, dtype=torch.uint8, count=n_byte, offset=TESTA)
                vista.copy_(t.detach().reshape(-1).view(torch.uint8))
                del vista
            if dimensione > TESTA + n_byte:
                mm[TESTA + n_byte:dimensione] = b"\0" * (dimensione - TESTA - n_byte)
            if self.verifica == "completa":
                if self._byte_ombra + n_byte > OMBRA_MAX:
                    raise ValueError(f"THOR_CACHE_LOTTI_VERIFICA=completa: copia ombra oltre "
                                     f"{OMBRA_MAX / 2**30:.0f} GiB (solo collaudo, a pochi lotti)")
                self._ombra[fid] = t.detach().to("cpu", copy=True)
                self._byte_ombra += n_byte
        except BaseException:
            self._pool.rendi(mm)
            raise
        with self._cond:                         # contropressione: RAM host limitata
            while self._in_coda > 0 and self._in_coda + dimensione > CODA_SCRITTURA_MAX:
                self._cond.wait()
            self._in_coda += dimensione
        with self._lock:
            self._files[fid] = f
            self._registra(t, f)
            self._byte_vivi += dimensione
            self.contatori["vivi_max"] = max(self.contatori["vivi_max"], len(self._files))
            self.contatori["byte_vivi_max"] = max(self.contatori["byte_vivi_max"], self._byte_vivi)
            f.scrittura = self._io.submit(self._job_scrivi, f, mm)
        return fid

    def incref(self, fid: int) -> None:
        with self._lock:
            self._files[fid].ref += 1

    def decref(self, fid: int) -> None:
        with self._lock:
            f = self._files.get(fid)
            if f is None:
                return
            f.ref -= 1
            if f.ref > 0:
                return
            del self._files[fid]
            self._byte_vivi -= f.dimensione
            self._scarta_precarica(f)
            if fid in self._ombra:
                del self._ombra[fid]
                self._byte_ombra -= f.n_byte
            f.vivo = None
            if not self._chiuso:
                self._io.submit(self._job_cancella, f.percorso)   # FIFO: dopo scrittura e letture gia' chieste

    def _scarta_precarica(self, f: _File) -> None:
        pf, f.precarica = f.precarica, None
        if pf is None:
            return
        self._byte_precarica -= f.dimensione
        self.contatori["precariche_buttate"] += 1
        if not pf.cancel():
            pf.add_done_callback(self._rendi_da_future)

    def _rendi_da_future(self, fut) -> None:
        try:
            mm = fut.result()
        except BaseException:
            return                               # errore di una precarica rilasciata: scartato (critica 9)
        self._pool.rendi(mm)

    def _vivo(self, f: _File) -> "torch.Tensor | None":
        t = f.vivo() if f.vivo is not None else None
        if t is None:
            return None
        if t._version != f.versione:
            raise RuntimeError(f"cache lotti: {f.nome} lotto {f.lotto} modificato SUL POSTO dopo il deposito "
                               f"(_version {f.versione} -> {t._version}): la cache non sarebbe piu' esatta")
        return t

    def precarica(self, fid: int) -> None:
        with self._lock:
            f = self._files.get(fid)
            if f is None or f.precarica is not None or self._chiuso:
                return
            if self._vivo(f) is not None:
                return                           # l'oggetto e' vivo: la lettura non servira'
            if self._byte_precarica + f.dimensione > PRECARICA_MAX:
                return                           # oltre il tetto: si leggera' al momento
            self._byte_precarica += f.dimensione
            f.precarica = self._io.submit(self._job_leggi, f)

    def leggi(self, fid: int, registra: bool = True) -> torch.Tensor:
        """Il tensore del file `fid`: l'oggetto vivo se c'e' (provenienza), altrimenti riletto
        (dalla precarica o subito) in un tensore NUOVO con dtype/forma/stride/device originali,
        impronta verificata. `registra=False` (thread del punto di ripresa): non tocca le
        precariche del thread principale e non diventa l'oggetto vivo del file."""
        if registra:
            self.controlla()
        with self._lock:
            f = self._files.get(fid)
            if f is None:
                raise RuntimeError(f"cache lotti: file {fid} non piu' nel magazzino (chiuso troppo presto?)")
            t = self._vivo(f)
            if t is not None:
                self.contatori["oggetti_vivi_riusati"] += 1
                if self._ombra:
                    self._confronta_ombra(f, t)
                return t
            pf = None
            if registra and f.precarica is not None:
                pf, f.precarica = f.precarica, None
                self._byte_precarica -= f.dimensione
                self.contatori["precariche_usate"] += 1
            if pf is None:
                pf = self._io.submit(self._job_leggi, f)
        mm = pf.result()
        try:
            t = self._materializza(f, mm)
        finally:
            self._pool.rendi(mm)
        if impronta(t) != f.impronta:
            raise RuntimeError(f"cache lotti: IMPRONTA diversa rileggendo {f.percorso.name} ({f.nome}, lotto "
                               f"{f.lotto}): file corrotto, scambiato o letto prima della fine della scrittura")
        if self._ombra:
            self._confronta_ombra(f, t)
        if registra:
            with self._lock:
                if fid in self._files:
                    self._registra(t, f)
        return t

    def _materializza(self, f: _File, mm: mmap.mmap) -> torch.Tensor:
        try:
            testa = json.loads(bytes(mm[:TESTA]).rstrip(b"\0"))
        except ValueError as e:
            raise RuntimeError(f"cache lotti: intestazione illeggibile in {f.percorso.name}") from e
        attesa = {"magia": MAGIA, "fid": f.fid, "n_byte": f.n_byte, "dtype": str(f.dtype), "forma": list(f.forma),
                  "stride": list(f.stride), "impronta": f.impronta}
        diversi = {k: (testa.get(k), v) for k, v in attesa.items() if testa.get(k) != v}
        if diversi:
            raise RuntimeError(f"cache lotti: {f.percorso.name} non e' il file atteso: {diversi}")
        t = torch.empty_strided(f.forma, f.stride, dtype=f.dtype, device=f.device)
        if f.n_byte:
            vista = torch.frombuffer(mm, dtype=torch.uint8, count=f.n_byte, offset=TESTA)
            t.view(-1).view(torch.uint8).copy_(vista)     # host -> device SINCRONA: al ritorno il buffer e' libero
            del vista
        return t

    def _confronta_ombra(self, f: _File, t: torch.Tensor) -> None:
        o = self._ombra.get(f.fid)
        if o is not None and not torch.equal(t.detach().to("cpu"), o):
            raise RuntimeError(f"cache lotti (VERIFICA=completa): {f.nome} lotto {f.lotto} diverso dalla copia ombra")

    def chiudi(self) -> None:
        """Fine corsa: coda di I/O svuotata, cartella rimossa. Idempotente."""
        if self._chiuso:
            return
        self._chiuso = True
        errori = []
        with self._lock:
            for f in list(self._files.values()):
                self._scarta_precarica(f)
            self._files.clear()
            self._per_id.clear()
            self._ombra.clear()
        try:
            self._io.shutdown(wait=True, cancel_futures=True)   # scritture non iniziate: inutili, si butta tutto
        except BaseException as e:
            errori.append(e)
        try:
            shutil.rmtree(self.cartella)
        except FileNotFoundError:
            pass
        except BaseException as e:
            errori.append(e)
        try:
            os.close(self._fd_lucchetto)
        except OSError:
            pass
        self._pool.chiudi()
        c = self.contatori
        log(f"cache lotti: corsa {self.cartella.name} chiusa · scritti {c['scritti'] / 2**30:.2f} GiB "
            f"({c['file_scritti']} file) · letti {c['letti'] / 2**30:.2f} GiB ({c['file_letti']} file) · "
            f"riusi per provenienza {c['riusi_provenienza']} · oggetti vivi riusati {c['oggetti_vivi_riusati']} · "
            f"precariche usate {c['precariche_usate']} buttate {c['precariche_buttate']} · picco file vivi "
            f"{c['vivi_max']} ({c['byte_vivi_max'] / 2**30:.2f} GiB)")
        if errori:
            raise RuntimeError(f"cache lotti: chiusura con errori: {errori!r}") from errori[0]

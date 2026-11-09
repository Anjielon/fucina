#!/usr/bin/env python3
"""Collaudo CPU della CACHE PER LOTTO (THOR_CACHE_LOTTI, deposito_lotti.py + deposito_io.py) — 25/9/2026.

Uso (regola di sessione: la GPU e' della forgia, qui solo CPU):
    env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 19 \\
        @@ROOT:venv@@/bin/python3 test_cache_lotti.py [--ricetta base|produzione] [--solo-contenitore]

Cartella di lavoro: @@ROOT:models@@/gguf/tony-tern/test_cache_lotti/<uuid>/ (MAI /tmp: e' RAM, e la
cache la rifiuta), rimossa a fine collaudo. File giocattolo: pochi KiB.

(a) IL CONTENITORE (Deposito, budget GPU 0 -> tutto su disco):
  C1 andata e ritorno: bf16/fp16/fp32/int64/int8/bool, 0-dim, None, liste/tuple/dict annidati;
     torch.equal, stessi dtype/forma/stride/device/requires_grad; i file ci sono davvero.
  C2 residenza: budget -> R; b < R lo STESSO oggetto; b >= R il deposito non trattiene nulla.
  C3 provenienza: stesso oggetto in due depositi = un file; chiudere uno lascia il file all'altro;
     set ripetuto sulla stessa chiave non tocca i contatori; modifica sul posto -> riscrive, e il
     vecchio oggetto al get SOLLEVA; riferimento() passa il file senza scriverlo.
  C4 impronta: un byte corrotto su disco fa sollevare il get.
  C5 rifiuti: vista non densa, offset, storage piu' grande, requires_grad, grad_fn; radice su
     tmpfs; radice che contiene l'uscita; THOR_CACHE_LOTTI=2; budget negativo; "" = spento.
  C6 disco pieno: riserva enorme -> DiscoPienoCache PRIMA di scrivere; ENOSPC nel thread di I/O ->
     errore alla prima operazione dopo, file parziale rimosso.
  C7 precarica: stessi valori con e senza; chiave assente = no-op; precarica rilasciata dalla chiusura.
  C8 thread: leggi_per_scrittura da un secondo thread mentre il principale legge.
  C9 pulizia: chiudi idempotente, chiudi_corsa toglie la cartella; orfano di un processo UCCISO
     (SIGKILL) rimosso alla corsa dopo; corsa VIVA di un altro processo non toccata; voce senza
     marcatore -> radice rifiutata; chiudi_corsa non maschera un'eccezione in volo.
  C10 spento: dict vero, identita', None, no-op; corsa implicita (valida da sola) e sua chiusura.
  C11 VERIFICA=completa: copia ombra, una manomissione e' scoperta.
  C12 (revisione risorse 25/9) nascite interrotte: `.nascita-*` di un pid morto tolta, di un pid
     vivo lasciata, estranea rifiutata; nascita fallita (ENOSPC) senza residui; `pulisci_radice`
     del lanciatore toglie l'orfano di un SIGKILL senza aprire una corsa.
(b) FORGIA GIOCATTOLO spenta vs accesa (strati Qwen3.5-MoE veri a scala giocattolo, interfaccia
    VERA: raccordo_scorrevole + ottimizza_finestra_JOINT), 6 lotti (5 + 1 held-out), passi 7 (il
    giro t % n_tr rilegge i lotti): codici su disco, perdite di ogni passo e held-out, semi di ogni
    finestra, sha1 del giornale e impronta dei punti di ripresa IDENTICI (torch.equal / ==, mai
    allclose) con budget 0, R = 2 e infinito; scrittura sincrona e asincrona; teacher catturato e
    ricalcolato; W = 3. File scritti = conto di §7.3 (provenienza: nessuna scrittura doppia).
    Ripresa incrociata (interrotta accesa -> ripresa spenta e viceversa) = corsa intera.
    Interruzione a cache accesa: l'eccezione arriva intatta e la cartella della cache non resta.
    B6 SIGTERM a cache accesa: SystemExit(143), cartella tolta, gestore rimesso; spenta: nessun gestore.
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import gc
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

RADICE_COLLAUDO = Path("@@ROOT:models@@/gguf/tony-tern/test_cache_lotti")
QUI = Path(__file__).resolve().parent
ESITI: list = []


def ok(nome: str, cond, dettaglio: str = "") -> bool:
    ESITI.append((nome, bool(cond)))
    print(f"    {'✅' if cond else '⛔'} {nome}" + (f" — {dettaglio}" if dettaglio else ""), flush=True)
    return bool(cond)


def solleva(nome: str, fn, tipo=Exception, testo: str = "") -> bool:
    try:
        fn()
    except tipo as e:
        return ok(nome, testo in str(e), f"{type(e).__name__}: {str(e)[:110]}")
    except BaseException as e:  # noqa: BLE001
        return ok(nome, False, f"eccezione sbagliata {type(e).__name__}: {str(e)[:110]}")
    return ok(nome, False, "nessuna eccezione")


@contextlib.contextmanager
def env(**kv):
    prima = {k: os.environ.get(k) for k in kv}
    for k, v in kv.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)
    try:
        yield
    finally:
        for k, v in prima.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class _Tee(io.TextIOBase):
    def __init__(self, *f):
        self.f = f

    def write(self, s):
        for x in self.f:
            x.write(s)
        return len(s)

    def flush(self):
        for x in self.f:
            x.flush()


# ═══════════════════════════════════ (a) il contenitore ═══════════════════════════════════
def _file_bin(DL) -> list:
    c = DL.corsa_aperta()
    return sorted(c.mag.cartella.glob("f*.bin")) if c is not None else []


def _svuota_io(DL) -> None:
    """Aspetta che il thread di I/O abbia eseguito tutto cio' che e' in coda (FIFO: un lavoro vuoto dopo)."""
    c = DL.corsa_aperta()
    c.mag._io.submit(lambda: None).result()


def c1_andata_ritorno(DL, torch, radice: Path) -> None:
    print("\n── C1 andata e ritorno (budget 0: tutto su disco) ──")
    with env(THOR_CACHE_LOTTI="1", THOR_CACHE_LOTTI_DIR=radice, THOR_CACHE_LOTTI_GPU_GIB="0"):
        DL.apri_corsa("c1", n_lotti=8)
        try:
            g = torch.Generator().manual_seed(1)
            valori = {
                0: torch.randn(3, 5, generator=g).to(torch.bfloat16),
                1: torch.randn(4, 7, generator=g).to(torch.float16),
                2: torch.randn(2, 3, 4, generator=g),
                3: torch.randint(-2**62, 2**62, (9,), generator=g, dtype=torch.int64),
                4: torch.randint(-128, 127, (13,), generator=g, dtype=torch.int8),      # 13 byte: non multiplo di 4
                5: torch.rand(11, generator=g) > 0.5,
                6: torch.tensor(3.25),
                7: None,
                8: [torch.randn(2, 2, generator=g), None, (torch.arange(5), {"x": torch.ones(3, dtype=torch.bfloat16)})],
                9: torch.empty(0, 4),
            }
            copie = {b: _clona(torch, v) for b, v in valori.items()}
            d = DL.deposito("c1")
            for b, v in valori.items():
                d[b] = v
            n_tensori = sum(1 for v in _foglie(torch, valori))
            _svuota_io(DL)
            ok("C1: un file per tensore sulla radice di collaudo (nessuna copia sulla GPU/CPU trattenuta)",
               len(_file_bin(DL)) == n_tensori, f"{len(_file_bin(DL))} file, {n_tensori} tensori")
            del valori, v
            gc.collect()
            letti = {b: d[b] for b in d}
            ok("C1: stessi valori al bit, stessi dtype/forma/stride/device, requires_grad False",
               _uguali(torch, letti, copie), "")
            ok("C1: len/iter/contains/keys come un dict", len(d) == 10 and list(d) == list(range(10))
               and 3 in d and 42 not in d and sorted(d.keys()) == list(range(10)))
            del d[3]
            ok("C1: del d[b] toglie la voce (e il file a zero riferimenti)", 3 not in d and len(d) == 9)
            d.chiudi()
            _svuota_io(DL)
            ok("C1: chiudi toglie tutti i file", _file_bin(DL) == [])
            cartella = DL.corsa_aperta().mag.cartella
        finally:
            DL.chiudi_corsa()
        ok("C1: chiudi_corsa toglie la cartella della corsa", not cartella.exists(), str(cartella.name))


def _foglie(torch, v):
    if torch.is_tensor(v):
        yield v
    elif isinstance(v, (list, tuple)):
        for x in v:
            yield from _foglie(torch, x)
    elif isinstance(v, dict):
        for x in v.values():
            yield from _foglie(torch, x)


def _clona(torch, v):
    if torch.is_tensor(v):
        return v.clone()
    if isinstance(v, list):
        return [_clona(torch, x) for x in v]
    if isinstance(v, tuple):
        return tuple(_clona(torch, x) for x in v)
    if isinstance(v, dict):
        return {k: _clona(torch, x) for k, x in v.items()}
    return v


def _uguali(torch, a, b) -> bool:
    if torch.is_tensor(a) or torch.is_tensor(b):
        return (torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape
                and a.stride() == b.stride() and a.device == b.device and not a.requires_grad
                and torch.equal(a, b))
    if type(a) is not type(b):
        return False
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_uguali(torch, x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return sorted(a) == sorted(b) and all(_uguali(torch, a[k], b[k]) for k in a)
    return a == b


def c2_c3_residenza_provenienza(DL, torch, radice: Path) -> None:
    import weakref
    print("\n── C2 residenza · C3 provenienza ──")
    ids = torch.zeros(6, 2, 3, dtype=torch.int64)             # 6 lotti, h = 2 x 3 x 4 elementi
    per_lotto = 24 * (4 * 3 + 4)                                # h fp32 x (1 + W=2) + e fp32
    with env(THOR_CACHE_LOTTI="1", THOR_CACHE_LOTTI_DIR=radice,
             THOR_CACHE_LOTTI_GPU_GIB=repr(2.5 * per_lotto / 2 ** 30)):
        DL.apri_corsa("c2", n_lotti=6)
        try:
            DL.fissa_residenti(ids, hidden=4, byte_h=4, w=2)
            c = DL.corsa_aperta()
            ok("C2: R = floor(budget / byte_per_lotto) = 2", c.R == 2, f"R={c.R}")
            d = DL.deposito("h")
            t = [torch.randn(2, 3, 4) for _ in range(6)]
            rif = [weakref.ref(x) for x in t]
            for b in range(6):
                d[b] = t[b]
            ok("C2: b < R -> lo STESSO oggetto (semantica di un dict)", d[0] is t[0] and d[1] is t[1])
            copia3 = t[3].clone()
            del t
            gc.collect()
            ok("C2: b >= R -> il deposito non trattiene l'oggetto (weakref morto)",
               rif[0]() is not None and all(rif[b]() is None for b in range(2, 6)))
            ok("C2: b >= R riletto uguale al bit", torch.equal(d[3], copia3))
            ok("C2: byte residenti <= budget", c.cassa.usati <= c.cassa.budget and c.cassa.usati == 2 * 96,
               f"{c.cassa.usati} byte")
            d.chiudi()
            ok("C2: chiusa, la cassa torna a zero", c.cassa.usati == 0)

            # ── C3 provenienza
            _svuota_io(DL)
            prima = c.mag.contatori["file_scritti"]
            x = torch.randn(2, 3, 4)
            a, b_ = DL.deposito("a"), DL.deposito("b")
            a[4] = x
            b_[4] = [x, None]
            a[4] = x                                            # set ripetuto, stesso oggetto
            _svuota_io(DL)
            ok("C3: stesso oggetto in due depositi (e set ripetuto) = UN file",
               c.mag.contatori["file_scritti"] - prima == 1 and len(_file_bin(DL)) == 1)
            ok("C3: get con l'oggetto vivo = lo stesso oggetto (nessuna rilettura)", a[4] is x and b_[4][0] is x)
            a.chiudi()
            _svuota_io(DL)
            ok("C3: chiuso un deposito, il file resta all'altro", len(_file_bin(DL)) == 1 and b_[4][0] is x)
            copia = x.clone()
            del x
            gc.collect()
            ok("C3: morto l'oggetto, rilettura dal file uguale al bit", torch.equal(b_[4][0], copia))
            a.chiudi()                                          # idempotente
            ok("C3: chiudi ripetuto = no-op", a.chiuso)
            solleva("C3: deposito chiuso -> accesso rifiutato", lambda: a[4], RuntimeError, "chiuso")
            # riferimento: passa il file senza scriverlo ne' leggerlo
            letti_prima, scritti_prima = c.mag.contatori["file_letti"], c.mag.contatori["file_scritti"]
            u = DL.deposito("uscita")
            u[4] = b_.riferimento(4, 0)
            _svuota_io(DL)
            ok("C3: riferimento() = stesso file, zero letture, zero scritture",
               c.mag.contatori["file_letti"] == letti_prima and c.mag.contatori["file_scritti"] == scritti_prima
               and torch.equal(u[4], copia))
            b_.chiudi()
            u.chiudi()
            _svuota_io(DL)
            ok("C3: chiusi tutti, nessun file", _file_bin(DL) == [])
            # modifica sul posto: si riscrive, e il vecchio oggetto al get solleva
            y = torch.randn(2, 3, 4)
            e1, e2 = DL.deposito("e1"), DL.deposito("e2")
            e1[5] = y
            y.add_(1.0)
            e2[5] = y
            _svuota_io(DL)
            ok("C3: dopo una modifica sul posto il set RISCRIVE (2 file)", len(_file_bin(DL)) == 2)
            solleva("C3: get del valore depositato PRIMA della modifica -> solleva (non sarebbe esatto)",
                    lambda: e1[5], RuntimeError, "SUL POSTO")
            e1.chiudi()
            e2.chiudi()
        finally:
            DL.chiudi_corsa()


def c4_impronta(DL, torch, radice: Path) -> None:
    import deposito_io as DI
    print("\n── C4 impronta ──")
    with env(THOR_CACHE_LOTTI="1", THOR_CACHE_LOTTI_DIR=radice):
        DL.apri_corsa("c4", n_lotti=2)
        try:
            d = DL.deposito("x")
            d[0] = torch.randn(64, 33)
            d[1] = torch.randn(64, 33)
            copia = d[0].clone()
            _svuota_io(DL)
            f0, f1 = _file_bin(DL)
            dati = bytearray(f0.read_bytes())
            dati[DI.TESTA + 100] ^= 0x01
            f0.write_bytes(bytes(dati))
            gc.collect()
            solleva("C4: un bit cambiato nel file -> get solleva (impronta)", lambda: d[0], RuntimeError, "IMPRONTA")
            dati[DI.TESTA + 100] ^= 0x01
            f0.write_bytes(bytes(dati))
            ok("C4: file ripristinato -> rilettura uguale", torch.equal(d[0], copia))
            b0, b1 = f0.read_bytes(), f1.read_bytes()
            f0.write_bytes(b1)
            f1.write_bytes(b0)
            gc.collect()
            solleva("C4: due file scambiati -> get solleva", lambda: d[0], RuntimeError, "non e' il file atteso")
            f0.write_bytes(b0)
            f1.write_bytes(b1)
            f0.write_bytes(b0[:len(b0) - DI.TESTA])
            gc.collect()
            solleva("C4: file troncato -> get solleva", lambda: d[0], OSError, "troncato")
            f0.write_bytes(b0)
            d.chiudi()
        finally:
            DL.chiudi_corsa()


def c5_rifiuti(DL, torch, radice: Path, tmpfs: Path) -> None:
    print("\n── C5 rifiuti rumorosi ──")
    with env(THOR_CACHE_LOTTI="1", THOR_CACHE_LOTTI_DIR=radice):
        DL.apri_corsa("c5", n_lotti=1)
        try:
            d = DL.deposito("x")
            base = torch.randn(4, 6)
            solleva("C5: vista non contigua rifiutata", lambda: d.__setitem__(0, base.t()), ValueError, "non contiguo")
            solleva("C5: vista con offset rifiutata", lambda: d.__setitem__(0, base[1:]), ValueError, "offset")
            solleva("C5: vista su storage piu' grande rifiutata", lambda: d.__setitem__(0, base[:2]), ValueError,
                    "storage")
            p = torch.randn(3, requires_grad=True)
            solleva("C5: requires_grad rifiutato", lambda: d.__setitem__(0, p), ValueError, "requires_grad")
            solleva("C5: grad_fn rifiutato", lambda: d.__setitem__(0, p * 2), ValueError, "grad_fn")
            solleva("C5: tipo non ammesso rifiutato", lambda: d.__setitem__(0, 3.0), TypeError, "non ammesso")
            ok("C5: dopo i rifiuti il deposito e' vuoto e senza file", len(d) == 0 and _file_bin(DL) == [])
            d.chiudi()
        finally:
            DL.chiudi_corsa()
    with env(THOR_CACHE_LOTTI="1", THOR_CACHE_LOTTI_DIR=tmpfs / "cache"):
        solleva("C5: radice su tmpfs (/tmp e' RAM) rifiutata", lambda: DL.apri_corsa("c5b", 1), ValueError, "tmpfs")
        ok("C5: e non ha creato nulla", not (tmpfs / "cache").exists())
    with env(THOR_CACHE_LOTTI="1", THOR_CACHE_LOTTI_DIR=radice):
        solleva("C5: radice che CONTIENE l'uscita rifiutata",
                lambda: DL.apri_corsa("c5c", 1, protette=[radice / "out"]), ValueError, "si contengono")
        solleva("C5: radice DENTRO la ripresa rifiutata",
                lambda: DL.apri_corsa("c5d", 1, protette=[radice.parent]), ValueError, "si contengono")
    with env(THOR_CACHE_LOTTI="2"):
        solleva("C5: THOR_CACHE_LOTTI=2 -> ValueError", DL.attiva, ValueError, "attesi")
    with env(THOR_CACHE_LOTTI=""):
        ok("C5: THOR_CACHE_LOTTI vuota = spento (critica 14)", DL.attiva() is False)
    with env(THOR_CACHE_LOTTI="1", THOR_CACHE_LOTTI_GPU_GIB="-1"):
        solleva("C5: budget negativo -> ValueError", DL.budget_byte, ValueError, ">= 0")
    with env(THOR_CACHE_LOTTI="1", THOR_CACHE_LOTTI_VERIFICA="pigra", THOR_CACHE_LOTTI_DIR=radice):
        solleva("C5: VERIFICA sconosciuta -> ValueError", lambda: DL.apri_corsa("c5e", 1), ValueError, "VERIFICA")


def c6_disco_pieno(DL, torch, radice: Path) -> None:
    import errno
    import deposito_io as DI
    print("\n── C6 disco pieno ──")
    with env(THOR_CACHE_LOTTI="1", THOR_CACHE_LOTTI_DIR=radice, THOR_CACHE_LOTTI_RISERVA_GIB="1e7"):
        DL.apri_corsa("c6", n_lotti=1)
        try:
            d = DL.deposito("x")
            solleva("C6: riserva non disponibile -> DiscoPienoCache PRIMA di scrivere",
                    lambda: d.__setitem__(0, torch.randn(8)), DL.DiscoPienoCache, "forgia fermata")
            ok("C6: nessun file scritto, nessuna voce", _file_bin(DL) == [] and len(d) == 0)
            solleva("C6: fissa_residenti rifiuta la partenza se il picco non ci sta",
                    lambda: DL.fissa_residenti(torch.zeros(4, 2, 3, dtype=torch.int64), 4, 4, 2),
                    DL.DiscoPienoCache, "rifiuto la partenza")
            d.chiudi()
        finally:
            DL.chiudi_corsa()
    with env(THOR_CACHE_LOTTI="1", THOR_CACHE_LOTTI_DIR=radice):
        DL.apri_corsa("c6b", n_lotti=2)
        vero = DI.os.pwritev

        def pieno(*a, **k):
            raise OSError(errno.ENOSPC, "No space left on device (simulato)")
        try:
            d = DL.deposito("x")
            DI.os.pwritev = pieno
            try:
                d[0] = torch.randn(8)
                _svuota_io(DL)
            finally:
                DI.os.pwritev = vero
            ok("C6: ENOSPC nel thread di I/O -> file parziale rimosso", _file_bin(DL) == [])
            solleva("C6: ... e l'errore esce alla PRIMA operazione dopo (mai un ripiego)",
                    lambda: d.__setitem__(1, torch.randn(8)), OSError, "thread di I/O")
        finally:
            with contextlib.suppress(BaseException):
                DL.chiudi_corsa()
        ok("C6: la cartella della corsa fallita non resta", not any(radice.glob("c6b-*")))


def c7_c8_precarica_thread(DL, torch, radice: Path) -> None:
    print("\n── C7 precarica · C8 thread ──")
    with env(THOR_CACHE_LOTTI="1", THOR_CACHE_LOTTI_DIR=radice):
        DL.apri_corsa("c7", n_lotti=6)
        try:
            c = DL.corsa_aperta()
            d = DL.deposito("x")
            copie = {}
            for b in range(6):
                t = torch.randn(16, 16)
                copie[b] = t.clone()
                d[b] = [t, torch.randn(4)]
                copie[(b, 1)] = d[b][1].clone()
            del t
            gc.collect()
            senza = [d[b][0] for b in range(6)]
            del senza
            gc.collect()
            usate = c.mag.contatori["precariche_usate"]
            con = []
            for b in range(6):
                DL.precarica((d, None, {}), b, b + 1, 99)          # 99: chiave assente = no-op
                con.append(d[b][0].clone())
            gc.collect()
            ok("C7: con precarica stessi valori al bit", all(torch.equal(con[b], copie[b]) for b in range(6)))
            ok("C7: le precariche sono state usate", c.mag.contatori["precariche_usate"] - usate >= 6,
               f"{c.mag.contatori['precariche_usate'] - usate}")
            # C8: il thread del punto di ripresa legge mentre il principale legge
            fuori, errori = {}, []

            def secondo():
                try:
                    for b in range(6):
                        fuori[b] = DL.leggi_per_scrittura(d, b)[0].clone()
                except BaseException as e:  # noqa: BLE001
                    errori.append(e)
            th = threading.Thread(target=secondo)
            th.start()
            dentro = {b: d[b][0].clone() for b in range(6)}
            th.join()
            ok("C8: leggi_per_scrittura da un secondo thread = valori del principale",
               not errori and all(torch.equal(fuori[b], dentro[b]) and torch.equal(fuori[b], copie[b])
                                  for b in range(6)), str(errori[:1]))
            del fuori, dentro
            gc.collect()
            buttate = c.mag.contatori["precariche_buttate"]
            DL.precarica((d,), 0, 1, 2)                          # 3 lotti x 2 file, mai consumate
            d.chiudi()
            _svuota_io(DL)
            ok("C7: le precariche pendenti rilasciate dalla chiusura (buffer resi), nessun file resta",
               _file_bin(DL) == [] and c.mag.contatori["precariche_buttate"] - buttate == 6,
               f"buttate {c.mag.contatori['precariche_buttate'] - buttate}")
        finally:
            DL.chiudi_corsa()


_FIGLIO = """
import os, sys, time
sys.path.insert(0, {qui!r})
import deposito_io as DI
m = DI.Magazzino(DI.Path({radice!r}), {nome!r})
print("CARTELLA", m.cartella, flush=True)
if {uccidi}:
    os.kill(os.getpid(), 9)
sys.stdin.readline()
m.chiudi()
"""


def _figlio(radice: Path, nome: str, uccidi: bool):
    codice = _FIGLIO.format(qui=str(QUI), radice=str(radice), nome=nome, uccidi=uccidi)
    e = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": ""}
    return subprocess.Popen([sys.executable, "-c", codice], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, env=e)


def _cartella_del_figlio(p) -> Path:
    for riga in p.stdout:
        if riga.startswith("CARTELLA "):
            return Path(riga.split(" ", 1)[1].strip())
    raise RuntimeError(f"il figlio non ha detto la sua cartella: {p.stderr.read()[-300:]}")


def c9_pulizia(DL, torch, radice: Path) -> None:
    import deposito_io as DI
    print("\n── C9 pulizia: orfani, corse vive, radice sporca, eccezione in volo ──")
    morto = _figlio(radice, "morto", True)
    cart_morto = _cartella_del_figlio(morto)
    morto.wait()
    ok("C9: il figlio ucciso con SIGKILL ha lasciato la sua cartella", morto.returncode == -signal.SIGKILL
       and cart_morto.exists(), f"rc {morto.returncode}")
    vivo = _figlio(radice, "vivo", False)              # anche lui pulisce gli orfani alla nascita
    cart_vivo = _cartella_del_figlio(vivo)
    ok("C9: orfano (marcatore + lucchetto libero) RIMOSSO dalla corsa dopo (qui: quella del figlio vivo)",
       not cart_morto.exists())
    with env(THOR_CACHE_LOTTI="1", THOR_CACHE_LOTTI_DIR=radice):
        DL.apri_corsa("c9", n_lotti=1)
        DL.chiudi_corsa()
    ok("C9: corsa VIVA di un altro processo (lucchetto tenuto) NON toccata", cart_vivo.exists())
    vivo.stdin.write("\n")
    vivo.stdin.flush()
    vivo.wait(timeout=60)
    ok("C9: la corsa viva si chiude da se' e toglie la sua cartella", not cart_vivo.exists() and vivo.returncode == 0,
       vivo.stderr.read()[-200:] if vivo.returncode else "")
    (radice / "estranea").mkdir()
    with env(THOR_CACHE_LOTTI="1", THOR_CACHE_LOTTI_DIR=radice):
        solleva("C9: radice con una voce SENZA marcatore rifiutata (mai cancellata)",
                lambda: DL.apri_corsa("c9b", 1), RuntimeError, "marcatore")
    ok("C9: ... e la voce estranea e' intatta", (radice / "estranea").exists())
    (radice / "estranea").rmdir()
    # chiudi_corsa con un'eccezione gia' in volo: la pulizia rotta NON la maschera
    vera = DI.Magazzino.chiudi

    def rotta(self):
        vera(self)
        raise RuntimeError("pulizia rotta (simulata)")
    with env(THOR_CACHE_LOTTI="1", THOR_CACHE_LOTTI_DIR=radice):
        DI.Magazzino.chiudi = rotta
        try:
            esito = None
            try:
                try:
                    DL.apri_corsa("c9c", 1)
                    raise ValueError("guasto vero")
                finally:
                    DL.chiudi_corsa()
            except ValueError:
                esito = "vero"
            except RuntimeError:
                esito = "maschera"
            ok("C9: eccezione in volo + pulizia rotta -> arriva l'eccezione VERA", esito == "vero", str(esito))
            DL.apri_corsa("c9d", 1)
            solleva("C9: senza eccezione in volo la pulizia rotta SOLLEVA", DL.chiudi_corsa, RuntimeError,
                    "pulizia rotta")
        finally:
            DI.Magazzino.chiudi = vera
    ok("C9: nessuna cartella di corsa resta nella radice", not any(radice.iterdir()), str(list(radice.iterdir())))


def _pid_morto() -> int:
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def c12_nascite_e_lanciatore(DL, torch, radice: Path) -> None:
    """25/9 (revisione risorse, difetti 2 e 3)."""
    import errno as _errno
    import deposito_io as DI
    print("\n── C12 nascite interrotte · pulizia dal lanciatore (pulisci_radice) ──")
    morto = _pid_morto()
    solo_lucchetto = radice / f"{DI.NASCITA}c12-{morto}-{uuid.uuid4().hex[:12]}"
    solo_lucchetto.mkdir()
    (solo_lucchetto / DI.LUCCHETTO).touch()
    (solo_lucchetto / DI.MARCATORE).write_text('{"magia": "thor-ca')          # marcatore scritto a meta'
    vuota = radice / f"{DI.NASCITA}c12-{morto}-{uuid.uuid4().hex[:12]}"       # morto subito dopo il mkdir
    vuota.mkdir()
    viva = radice / f"{DI.NASCITA}c12-{os.getpid()}-{uuid.uuid4().hex[:12]}"  # nascita in corso (pid vivo)
    viva.mkdir()
    DI.pulisci_radice(radice)
    ok("C12: nascita interrotta (pid morto, lucchetto + marcatore a meta') TOLTA", not solo_lucchetto.exists())
    ok("C12: nascita interrotta subito dopo il mkdir (pid morto, vuota) TOLTA", not vuota.exists())
    ok("C12: nascita di un pid VIVO non toccata e non rifiutata", viva.exists())
    viva.rmdir()
    for nome, dentro in (("contenuto estraneo", "altro.txt"), ("nome senza la forma di _nasci", None)):
        estranea = (radice / f"{DI.NASCITA}c12-{morto}-{uuid.uuid4().hex[:12]}" if dentro
                    else radice / f"{DI.NASCITA}fatta-a-mano")
        estranea.mkdir()
        if dentro:
            (estranea / dentro).write_text("x")
        solleva(f"C12: .nascita con {nome}: radice RIFIUTATA", lambda: DI.pulisci_radice(radice), RuntimeError,
                "marcatore")
        ok(f"C12: ... e la voce ({nome}) e' intatta", estranea.exists() and (not dentro or (estranea / dentro).exists()))
        shutil.rmtree(estranea)
    # difetto 3 alla sorgente: una nascita che fallisce (ENOSPC simulato) non lascia la .nascita-*
    vera = DI._avvio_processo

    def pieno(_pid):
        raise OSError(_errno.ENOSPC, "disco pieno (simulato)")
    DI._avvio_processo = pieno
    try:
        solleva("C12: nascita fallita (ENOSPC sul marcatore): l'errore VERO risale", lambda: DI.Magazzino(radice, "c12"),
                OSError, "simulato")
    finally:
        DI._avvio_processo = vera
    ok("C12: ... e non resta nessuna .nascita-*", not list(radice.glob(f"{DI.NASCITA}*")), str(list(radice.iterdir())))
    m = DI.Magazzino(radice, "c12b")
    ok("C12: la corsa dopo nasce senza rifiuti", m.cartella.exists())
    m.chiudi()
    # difetto 2: il lanciatore toglie l'orfano PRIMA dei df, senza aprire una corsa
    figlio = _figlio(radice, "morto12", True)
    cart = _cartella_del_figlio(figlio)
    figlio.wait()
    DI.pulisci_radice(radice, protette=[radice.parent / "uscita_finta", radice.parent / "ripresa_finta"])
    ok("C12: pulisci_radice toglie l'orfano di un SIGKILL (nessuna corsa aperta)", not cart.exists()
       and DL.corsa_aperta() is None)
    assente = radice / "assente"
    DI.pulisci_radice(assente)
    ok("C12: pulisci_radice su una radice assente: nulla, e non la crea", not assente.exists())
    solleva("C12: pulisci_radice con l'uscita dentro la radice: rifiutata", lambda: DI.pulisci_radice(
        radice, protette=[radice / "uscita"]), ValueError, "si contengono")
    ok("C12: radice vuota a fine prova", not any(radice.iterdir()), str(list(radice.iterdir())))


def c10_spento_implicita(DL, torch, radice: Path) -> None:
    print("\n── C10 spento = comportamento di prima · corsa implicita ──")
    with env(THOR_CACHE_LOTTI=None):
        h, e = {0: torch.ones(2)}, {0: torch.zeros(2)}
        coppia = (h, e)
        ok("C10: deposito() = un dict vero", type(DL.deposito("x")) is dict)
        ok("C10: in_deposito = identita' (is)", DL.in_deposito(coppia, 0) is coppia)
        ok("C10: contenitori_seme / contenitore_cattura = None",
           DL.contenitori_seme(1) is None and DL.contenitore_cattura(1) is None)
        DL.precarica((h, e, None), 0, 1)
        DL.chiudi(h, e, None)
        ok("C10: precarica/chiudi no-op sui dict", h[0] is coppia[0][0] and len(h) == 1)
        ok("C10: leggi_per_scrittura = mappa[b]", DL.leggi_per_scrittura(h, 0) is h[0])
        chiamate = []

        def seme(ids, emb):
            chiamate.append((ids, emb))
            return h, e
        r = DL.semina(seme, "ids", "emb")
        ok("C10: semina = seme_iniziale(ids, emb), stesso risultato", r[0] is h and r[1] is e
           and chiamate == [("ids", "emb")])
        DL.apri_corsa("spento", 3)
        DL.fissa_residenti(torch.zeros(1, 1, 1), 1, 1, 2)
        DL.chiudi_corsa()
        ok("C10: apri/fissa/chiudi_corsa spenti: nessuna corsa, niente su disco",
           DL.corsa_aperta() is None and not any(radice.iterdir()))
    with env(THOR_CACHE_LOTTI="1", THOR_CACHE_LOTTI_DIR=radice):
        d = DL.deposito("implicita")                           # valida usata da sola: nessun apri_corsa
        c = DL.corsa_aperta()
        d[0] = torch.randn(5)
        _svuota_io(DL)
        ok("C10: corsa IMPLICITA aperta al primo deposito, R = 0 (su disco)",
           c is not None and c.implicita and c.R == 0 and len(_file_bin(DL)) == 1)
        DL.apri_corsa("esplicita", 1)
        ok("C10: apri_corsa esplicito chiude l'implicita (cartella tolta)",
           c.chiusa and not c.mag.cartella.exists() and DL.corsa_aperta() is not c)
        solleva("C10: il deposito della corsa chiusa rifiuta l'accesso", lambda: d[0], RuntimeError, "chiusa")
        solleva("C10: due corse esplicite insieme rifiutate", lambda: DL.apri_corsa("seconda", 1), RuntimeError,
                "gia' aperta")
        DL.chiudi_corsa()


def c11_verifica_completa(DL, torch, radice: Path) -> None:
    print("\n── C11 VERIFICA=completa (copia ombra) ──")
    with env(THOR_CACHE_LOTTI="1", THOR_CACHE_LOTTI_DIR=radice, THOR_CACHE_LOTTI_VERIFICA="completa"):
        DL.apri_corsa("c11", n_lotti=2)
        try:
            c = DL.corsa_aperta()
            d = DL.deposito("x")
            d[0] = torch.randn(10)
            d[1] = torch.randn(10)
            gc.collect()
            ok("C11: rilettura confrontata con l'ombra: passa", d[0].shape == (10,) and d[1].shape == (10,))
            fid = next(iter(c.mag._ombra))
            c.mag._ombra[fid][3] += 1.0                          # manomissione (solo collaudo)
            gc.collect()
            solleva("C11: ombra diversa dal riletto -> solleva", lambda: [d[0], d[1]], RuntimeError, "ombra")
            d.chiudi()
        finally:
            DL.chiudi_corsa()


# ═══════════════════════════════════ (b) forgia giocattolo ═══════════════════════════════════
def _giocattolo(TB, torch, n: int, dtype, n_lotti: int = 6):
    class GiocattoloLotti(TB.Giocattolo):
        """Il Giocattolo di test_forgia_scorrevole con piu' lotti (n_lotti-1 + 1 held-out)."""

        def __init__(self):
            super().__init__(n, dtype)
            g = torch.Generator().manual_seed(11)
            self.ids = torch.randint(0, self.cfg.vocab_size, (n_lotti, 2, 6), generator=g)

        def batch(self):
            return self.ids.clone(), n_lotti - 1
    return GiocattoloLotti()


class Registro:
    """Tutto cio' che deve essere IDENTICO fra spenta e accesa."""

    def __init__(self):
        self.perdite: list = []
        self.contatori: list = []


def forgia(ctx, nome: str, cache: "dict | None", n: int = 5, w: int = 2, n_passi: int = 7, asincrona: bool = True,
           riusa_teacher: bool = True, interrompi=None, cartella: "Path | None" = None, corsa_id: str = "tcl"):
    """Una corsa giocattolo; `cache` = variabili THOR_CACHE_LOTTI* (None = spenta)."""
    TB, FS, R, V, DI, torch = ctx["TB"], ctx["FS"], ctx["R"], ctx["V"], ctx["DI"], ctx["torch"]
    cartella = cartella or (ctx["base"] / f"forgia_{nome}")
    gioc = _giocattolo(TB, torch, n, ctx["dtype"])
    reg = Registro()

    @functools.wraps(R.ottimizza_finestra_JOINT_B)        # la firma (interfaccia A) resta visibile a verifica_interfaccia_A
    def ottimizza(*a, **k):
        diag: list = []
        ris = R.ottimizza_finestra_JOINT_B(*a, raccogli_diagnostica=diag, **k)
        for d in diag:
            reg.perdite.append((list(d["perdite"]), d["heldout"]))
        return ris
    vera_chiudi = DI.Magazzino.chiudi

    def chiudi_e_conta(self):
        reg.contatori.append(dict(self.contatori))
        return vera_chiudi(self)
    variabili = {"THOR_CACHE_LOTTI": None, "THOR_CACHE_LOTTI_GPU_GIB": None, "THOR_CACHE_LOTTI_DIR": None,
                 "THOR_SCORREVOLE_IMPRONTA_RIPRESA": "1"}
    variabili.update(cache or {})
    buf = io.StringIO()
    DI.Magazzino.chiudi = chiudi_e_conta
    try:
        with env(**variabili), contextlib.redirect_stdout(_Tee(sys.stdout, buf) if ctx["verboso"] else buf):
            oss = TB.corsa(cartella, gioc, n_passi=n_passi, w=w, corsa_id=corsa_id, asincrona=asincrona,
                           interrompi=interrompi, ottimizza_fn=ottimizza, riusa_teacher=riusa_teacher)
    finally:
        DI.Magazzino.chiudi = vera_chiudi
        reg.log = buf.getvalue()
    reg.oss = oss
    reg.cartella = cartella
    reg.codici = TB.codici_su_disco(cartella, n, corsa_id)
    reg.giornale = [(g["strato"], g["sha1_Q"]) for g in
                    (json.loads(r) for r in (cartella / "out" / "giornale_scorrevole.jsonl").read_text().splitlines())]
    reg.impronte = re.findall(r"impronta del punto di ripresa finestra (\d+): ([0-9a-f]{64})", reg.log)
    return reg


def _semi_uguali(torch, a, b) -> bool:
    if sorted(a.semi) != sorted(b.semi):
        return False
    for k in a.semi:
        for x, y in zip(a.semi[k], b.semi[k]):
            if (x is None) != (y is None):
                return False
            if x is not None and (sorted(x) != sorted(y) or not all(
                    x[i].dtype == y[i].dtype and torch.equal(x[i], y[i]) for i in x)):
                return False
    return True


def confronta(ctx, etichetta: str, rif: Registro, reg: Registro, file_attesi: "int | None" = None) -> None:
    torch, TB = ctx["torch"], ctx["TB"]
    ok(f"{etichetta}: codici su disco IDENTICI (tutti gli strati, D e Q)",
       sorted(rif.codici) == sorted(reg.codici) and all(TB.uguali(rif.codici[L], reg.codici[L]) for L in rif.codici))
    ok(f"{etichetta}: perdite di OGNI passo e held-out identiche (==)",
       len(rif.perdite) == len(reg.perdite) and rif.perdite == reg.perdite, f"{len(reg.perdite)} finestre")
    ok(f"{etichetta}: semi di ogni finestra identici (torch.equal)", _semi_uguali(torch, rif.oss, reg.oss))
    ok(f"{etichetta}: sha1 dei codici nel giornale identici", rif.giornale == reg.giornale)
    ok(f"{etichetta}: impronta di OGNI punto di ripresa identica (stessi byte di torch.save)",
       bool(rif.impronte) and rif.impronte == reg.impronte, f"{len(reg.impronte)} punti")
    if file_attesi is not None:
        scritti = sum(c["file_scritti"] for c in reg.contatori)
        ok(f"{etichetta}: file scritti = conto di §7.3 ({file_attesi}): nessuna scrittura doppia",
           scritti == file_attesi, f"scritti {scritti} · riusi per provenienza "
                                   f"{sum(c['riusi_provenienza'] for c in reg.contatori)}")


def _radice_pulita(ctx, etichetta: str) -> None:
    r = ctx["radice_b"]
    ok(f"{etichetta}: nessuna cartella di cache rimasta", not r.exists() or not any(r.iterdir()),
       str(list(r.iterdir())[:3]) if r.exists() else "")


def parte_b(ctx) -> None:
    torch = ctx["torch"]
    rb = ctx["radice_b"]
    per_lotto = 2 * 6 * 8 * (ctx["dtype"].itemsize * 3 + 4)       # toy: h 2x6x8, W=2, e fp32
    budget_R2 = repr(2.5 * per_lotto / 2 ** 30)
    acc = {"THOR_CACHE_LOTTI": "1", "THOR_CACHE_LOTTI_DIR": str(rb)}
    print("\n── B1 forgia giocattolo (5 strati, W=2, 6 lotti, 7 passi): spenta vs accesa ──")
    rif = forgia(ctx, "spenta", None)
    ok("B1: riferimento spento: 4 finestre, codici di 5 strati", len(rif.codici) == 5 and len(rif.perdite) == 4)
    # §7.3 a R=0: finestra 0 = 6 file/lotto (seme h, seme e, hq, cattura, teacher[2], e del confine),
    # centrali 4 (hq, cattura, teacher[2], e), ultima 3 (hq, teacher[1], teacher[2]); 6 lotti
    reg = forgia(ctx, "accesa_B0", {**acc, "THOR_CACHE_LOTTI_GPU_GIB": "0"})
    confronta(ctx, "B1 budget 0 (tutto su disco)", rif, reg, file_attesi=6 * (6 + 4 + 4 + 3))
    _radice_pulita(ctx, "B1 budget 0")
    reg = forgia(ctx, "accesa_R2", {**acc, "THOR_CACHE_LOTTI_GPU_GIB": budget_R2})
    ok("B1 R=2: il log dichiara R = 2", "R = 2 residenti" in reg.log)
    confronta(ctx, "B1 R=2", rif, reg, file_attesi=4 * (6 + 4 + 4 + 3) + 2 * 4)   # b<R: solo hq
    reg = forgia(ctx, "accesa_inf", {**acc, "THOR_CACHE_LOTTI_GPU_GIB": "inf"})
    confronta(ctx, "B1 budget infinito", rif, reg, file_attesi=6 * 4)               # solo hq
    _radice_pulita(ctx, "B1")

    print("\n── B2 varianti: scrittura sincrona · teacher ricalcolato · W=3 ──")
    rif_s = forgia(ctx, "spenta_sinc", None, asincrona=False)
    confronta(ctx, "B2 sincrona: spenta = spenta asincrona", rif, rif_s)
    confronta(ctx, "B2 sincrona, budget 0", rif_s, forgia(ctx, "accesa_sinc", {**acc, "THOR_CACHE_LOTTI_GPU_GIB": "0"},
                                                          asincrona=False))
    rif_t = forgia(ctx, "spenta_ricalc", None, riusa_teacher=False)
    confronta(ctx, "B2 teacher ricalcolato, R=2", rif_t,
              forgia(ctx, "accesa_ricalc", {**acc, "THOR_CACHE_LOTTI_GPU_GIB": budget_R2}, riusa_teacher=False))
    rif_w3 = forgia(ctx, "spenta_w3", None, w=3)
    confronta(ctx, "B2 W=3, budget 0", rif_w3, forgia(ctx, "accesa_w3", {**acc, "THOR_CACHE_LOTTI_GPU_GIB": "0"}, w=3))
    _radice_pulita(ctx, "B2")

    print("\n── B3 ripresa incrociata (scrittura sincrona) ──")
    FS = ctx["FS"]
    for nome, prima, dopo in (("accesa->spenta", {**acc, "THOR_CACHE_LOTTI_GPU_GIB": "0"}, None),
                              ("spenta->accesa", None, {**acc, "THOR_CACHE_LOTTI_GPU_GIB": budget_R2})):
        for punto, k in (("dopo_ripresa", 1), ("prima_transizione", 2)):
            cart = ctx["base"] / f"ripresa_{nome.replace('>', '')}_{punto}_{k}"

            def interrompi(p, kk, _p=punto, _k=k):
                if p == _p and kk == _k:
                    raise FS.InterruzioneSimulata(f"{p}@{kk}")
            try:
                forgia(ctx, "x", prima, asincrona=False, interrompi=interrompi, cartella=cart)
                ok(f"B3 {nome} {punto}@{k}: interruzione avvenuta", False)
                continue
            except FS.InterruzioneSimulata:
                pass
            _radice_pulita(ctx, f"B3 {nome} {punto}@{k} dopo l'interruzione")
            reg = forgia(ctx, "x", dopo, asincrona=False, cartella=cart)
            fatte = [k2 for (e, k2, _) in reg.oss.eventi if e == "finestra_inizio"]
            ok(f"B3 {nome} {punto}@{k}: ripresa dalla finestra {fatte[0] if fatte else '?'} -> codici IDENTICI alla "
               f"corsa intera", fatte and fatte[0] > 0 and all(ctx["TB"].uguali(rif_s.codici[L], reg.codici[L])
                                                               for L in rif_s.codici), f"finestre {fatte}")
    _radice_pulita(ctx, "B3")

    print("\n── B4 interruzione a cache accesa: eccezione intatta, cartella tolta ──")

    def rompi(p, k):
        if p == "prima_transizione" and k == 1:
            raise FS.InterruzioneSimulata("guasto simulato")
    try:
        forgia(ctx, "rotta", {**acc, "THOR_CACHE_LOTTI_GPU_GIB": "0"}, interrompi=rompi)
        ok("B4: l'interruzione arriva al chiamante", False)
    except FS.InterruzioneSimulata as e:
        ok("B4: l'interruzione arriva al chiamante INTATTA", "guasto simulato" in str(e))
    _radice_pulita(ctx, "B4")
    ok("B4: nessuna corsa della cache resta aperta", ctx["DL"].corsa_aperta() is None)

    print("\n── B6 SIGTERM a cache accesa (revisione risorse, difetto 2): uscita 143, cartella tolta ──")
    visti: dict = {}

    def guarda(p, k):
        if p == "prima_transizione" and k == 0:
            visti["spenta"] = signal.getsignal(signal.SIGTERM)
    forgia(ctx, "spenta_sigterm", None, interrompi=guarda)
    ok("B6: spenta: nessun gestore di SIGTERM durante la corsa (azione di default)",
       visti.get("spenta") == signal.SIG_DFL, str(visti.get("spenta")))

    def termina(p, k):
        if p == "prima_transizione" and k == 0:
            visti["accesa"] = signal.getsignal(signal.SIGTERM)
        if p == "prima_transizione" and k == 1:
            if signal.getsignal(signal.SIGTERM) == signal.SIG_DFL:
                raise FS.InterruzioneSimulata("gestore di SIGTERM NON armato: il segnale ucciderebbe il collaudo")
            os.kill(os.getpid(), signal.SIGTERM)
    try:
        forgia(ctx, "sigterm", {**acc, "THOR_CACHE_LOTTI_GPU_GIB": "0"}, interrompi=termina)
        esito = "nessuna uscita"
    except SystemExit as e:
        esito = e.code
    except FS.InterruzioneSimulata as e:
        esito = str(e)
    ok("B6: accesa: gestore armato durante la corsa", visti.get("accesa") not in (None, signal.SIG_DFL),
       str(visti.get("accesa")))
    ok("B6: SIGTERM -> SystemExit(143) attraverso il finally della forgia", esito == 143, str(esito))
    _radice_pulita(ctx, "B6")
    ok("B6: nessuna corsa aperta e gestore di SIGTERM rimesso (default)",
       ctx["DL"].corsa_aperta() is None and signal.getsignal(signal.SIGTERM) == signal.SIG_DFL)

    print("\n── B5 valida da sola (corsa implicita): uscita_teacher_fine in un deposito, per provenienza (critiche 5c, 7) ──")
    V, DL = ctx["V"], ctx["DL"]

    def una_finestra(uscita):
        gioc = _giocattolo(ctx["TB"], torch, 2, ctx["dtype"])
        cfg, emb, pul, qua = gioc.carica([0, 1])
        ids, n_tr = gioc.batch()
        diag: list = []
        V.ottimizza_finestra_JOINT(qua, ids, cfg, 5, strati_teacher=pul, emb=emb, n_batch_addestramento=n_tr,
                                   raccogli_diagnostica=diag, uscita_teacher_fine=uscita)
        return diag[0]["perdite"], diag[0]["heldout"]
    with contextlib.redirect_stdout(io.StringIO()):
        u_dict: dict = {}
        p_spenta = una_finestra(u_dict)
        with env(THOR_CACHE_LOTTI="1", THOR_CACHE_LOTTI_DIR=str(rb)):
            u_dep = DL.deposito("uscita_fine")
            c = DL.corsa_aperta()
            p_accesa = una_finestra(u_dep)
            scritti_prima = c.mag.contatori["file_scritti"]
            _svuota_io(DL)
            valori = {b: u_dep[b].clone() for b in u_dep}
            vivi = len(list(c.mag.cartella.glob("f*.bin")))
            try:
                una_finestra({})
                rifiuto = "nessuno"
            except NotImplementedError as e:
                rifiuto = str(e)
            u_dep.chiudi()
            DL.chiudi_corsa()
    ok("B5: corsa implicita (nessun apri_corsa) con R = 0", c is not None and c.implicita and c.R == 0)
    ok("B5: perdite e held-out identici spenta/accesa", p_spenta == p_accesa)
    ok("B5: uscita_teacher_fine nel deposito = quella del dict, al bit (6 lotti)",
       sorted(valori) == sorted(u_dict) and all(torch.equal(valori[b], u_dict[b]) for b in u_dict))
    ok("B5: i file dell'uscita restano vivi (per provenienza) dopo la chiusura dei depositi della finestra",
       vivi == len(u_dict), f"{vivi} file vivi, scritti {scritti_prima}")
    ok("B5: a cache accesa un dict VERO per uscita_teacher_fine e' rifiutato (terrebbe tutti i lotti)",
       "dict vero" in rifiuto, rifiuto[:90])
    ok("B5: nessuna corsa aperta, radice pulita", DL.corsa_aperta() is None and not any(rb.iterdir()))
    del torch


# ═══════════════════════════════════ main ═══════════════════════════════════
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ricetta", choices=("base", "produzione"), default="base")
    ap.add_argument("--solo-contenitore", action="store_true")
    ap.add_argument("--verboso", action="store_true", help="mostra il log delle corse giocattolo")
    args = ap.parse_args()
    for v in [k for k in os.environ if k.startswith("THOR_CACHE_LOTTI")]:
        raise SystemExit(f"{v} impostata nell'ambiente: il collaudo le gestisce da se'")
    sys.path.insert(0, str(QUI))
    import test_raccordo_scorrevole as TR
    os.environ.update(TR.RICETTE[args.ricetta])        # PRIMA di importare valida (costanti di modulo)
    import torch
    assert not torch.cuda.is_available(), "solo CPU (CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES=)"
    import deposito_lotti as DL
    import deposito_io as DI
    base = RADICE_COLLAUDO / f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    base.mkdir(parents=True)
    tmpfs = Path("/tmp") / f"test_cache_lotti_{uuid.uuid4().hex[:8]}"          # mai creata: serve al rifiuto
    t0 = time.time()
    torch.manual_seed(0)
    print(f"cartella di collaudo {base} · ricetta {args.ricetta}")
    try:
        radice_a = base / "cache_a"
        radice_a.mkdir()
        c1_andata_ritorno(DL, torch, radice_a)
        c2_c3_residenza_provenienza(DL, torch, radice_a)
        c4_impronta(DL, torch, radice_a)
        c5_rifiuti(DL, torch, radice_a, tmpfs)
        c6_disco_pieno(DL, torch, radice_a)
        c7_c8_precarica_thread(DL, torch, radice_a)
        c9_pulizia(DL, torch, radice_a)
        c10_spento_implicita(DL, torch, radice_a)
        c11_verifica_completa(DL, torch, radice_a)
        c12_nascite_e_lanciatore(DL, torch, radice_a)
        ok("(a) radice del contenitore vuota a fine prove", not any(radice_a.iterdir()), str(list(radice_a.iterdir())))
        if not args.solo_contenitore:
            import valida_finestra_8_12 as V
            import forgia_scorrevole as FS
            import raccordo_scorrevole as R
            import test_forgia_scorrevole as TB
            ctx = {"TB": TB, "FS": FS, "R": R, "V": V, "DI": DI, "DL": DL, "torch": torch, "base": base,
                   "radice_b": base / "cache_b", "verboso": args.verboso,
                   "dtype": torch.bfloat16 if args.ricetta == "produzione" else torch.float32}
            parte_b(ctx)
    finally:
        with contextlib.suppress(BaseException):
            DL.chiudi_corsa()
        shutil.rmtree(base, ignore_errors=True)
        shutil.rmtree(tmpfs, ignore_errors=True)
    falliti = [n for n, o in ESITI if not o]
    print(f"\n{'✅ TUTTI' if not falliti else '⛔ FALLITI ' + str(len(falliti)) + ' su'} {len(ESITI)} controlli "
          f"(ricetta {args.ricetta}) in {time.time() - t0:.0f}s" + ("" if not falliti else ": " + "; ".join(falliti)))
    return 0 if not falliti else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Sentinella di progresso di `e2e_allena` (28/9/2026, blocco di EB3).

Il fatto: EB3 (commit 4ed4a95) si e' fermata al passo 21 nella curvatura del passo V, in attesa attiva di un segnale GPU
mai arrivato (`rocr::core::BusyWaitSignal::WaitRelaxed` nella prima sincronizzazione dell'avanti), GPU allo 0 %, nessun
errore del kernel (le code KFD di ROCm non hanno il timeout degli anelli del kernel): ~25 minuti persi prima che qualcuno
guardasse. La diagnosi (messaggio del commit) NON trova la causa nel codice della corsa: la stessa curvatura sul modello
minuscolo e' identica, op per op e forma per forma, con e senza --lora-ripristina, e il blocco e' cominciato mentre una
KLD di llama.cpp in Vulkan (-ngl 999, unita' kld-k01 15:37-15:44) girava sulla stessa GPU. Un'attesa di quel tipo non si
ripara da Python; si puo' pero' impedire che resti MUTA:

  * chi lavora chiama `segno()` a ogni unita' di progresso (micro-lotto caricato, lotto della sonda, riga di registro);
  * un filo Python controlla: nessun segno da `secondi` -> una riga nel registro (quanto e' fermo, chi altro tiene la
    GPU secondo `controllo_gpu`), le pile di TUTTI i fili (faulthandler, nel registro) e uscita col codice 75
    (EX_TEMPFAIL: guasto transitorio). L'unita' systemd con RestartForceExitStatus=75 riparte dall'ultima ripresa (la
    guardia GPU all'avvio aspetta che la GPU sia libera); ogni altro errore resta un errore;
  * rete di sicurezza per un blocco col GIL tenuto (il filo Python non girerebbe): `faulthandler.dump_traceback_later`
    in C, `secondi + margine`, pile nel registro e uscita 1.

Default `E2E_SENTINELLA_S` = 1800 s (0 = spenta). Nessun effetto sulla matematica: `segno()` scrive un float."""
from __future__ import annotations

import faulthandler
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional

CODICE_USCITA = 75                    # EX_TEMPFAIL: la corsa si riprende dall'ultima ripresa (RestartForceExitStatus=75)
DEFAULT_S = 1800.0                    # il vuoto di progresso piu' lungo misurato (passo V senza prendi) e' ~12 min
DEFAULT_MARGINE_S = 600.0             # la rete in C scatta dopo il filo Python, solo se il filo non ha potuto girare
RIARMO_MIN_S = 10.0                   # la rete in C si riarma al piu' ogni 10 s (segno() puo' arrivare a raffica)...
                                      # ...e comunque 4 volte per finestra (soglie brevi delle prove)

_ultimo = [time.monotonic()]
_attiva: list = [None]


def segno() -> None:
    """Un'unita' di progresso. Costo: un float (e, al piu' ogni RIARMO_MIN_S, il riarmo della rete in C)."""
    _ultimo[0] = time.monotonic()
    s = _attiva[0]
    if s is not None:
        s._riarma()


def secondi_da_env(env=os.environ) -> tuple:
    """(secondi, margine) da E2E_SENTINELLA_S / E2E_SENTINELLA_MARGINE_S; secondi <= 0 = spenta."""
    return (float(env.get("E2E_SENTINELLA_S", DEFAULT_S)), float(env.get("E2E_SENTINELLA_MARGINE_S", DEFAULT_MARGINE_S)))


class Sentinella:
    def __init__(self, registro: Path, secondi: float, log: Callable[[str], None],
                 controllo_gpu: Optional[Callable[[], str]] = None, margine: float = DEFAULT_MARGINE_S,
                 uscita: Callable[[int], None] = os._exit):
        if not secondi > 0 or not margine > 0:
            raise ValueError(f"sentinella: secondi {secondi}, margine {margine}: servono > 0")
        self.secondi, self.margine, self.log, self.controllo_gpu, self._uscita = secondi, margine, log, controllo_gpu, uscita
        self._f = open(registro, "a")         # resta aperto: faulthandler scrive sul descrittore, anche senza GIL
        self._fine = threading.Event()
        self._riarmato = 0.0
        self._ogni = min(RIARMO_MIN_S, (secondi + margine) / 4)
        self._filo = threading.Thread(target=self._ciclo, name="sentinella", daemon=True)

    def _riarma(self) -> None:
        t = time.monotonic()
        if t - self._riarmato >= self._ogni:
            self._riarmato = t
            faulthandler.dump_traceback_later(self.secondi + self.margine, repeat=False, file=self._f, exit=True)

    def _ciclo(self) -> None:
        passo = max(0.05, min(30.0, self.secondi / 4))
        while not self._fine.wait(passo):
            fermo = time.monotonic() - _ultimo[0]
            if fermo > self.secondi:
                self._scatta(fermo)
                return

    def _scatta(self, fermo: float) -> None:
        try:
            gpu = self.controllo_gpu() if self.controllo_gpu is not None else "(non controllata)"
        except Exception as e:                           # il controllo non deve impedire l'uscita
            gpu = f"(controllo fallito: {type(e).__name__}: {e})"
        try:
            self.log(f"⛔ SENTINELLA: nessun progresso da {fermo:.0f} s (soglia {self.secondi:.0f} s): processo fermo "
                     f"(attesa GPU senza fine?). Pile di tutti i fili qui sotto; esco col codice {CODICE_USCITA} (l'unita' "
                     f"con RestartForceExitStatus={CODICE_USCITA} riparte dall'ultima ripresa). Altri sulla GPU: {gpu}")
        except Exception:
            pass
        try:
            faulthandler.dump_traceback(file=self._f, all_threads=True)
            self._f.flush()
        finally:
            self._uscita(CODICE_USCITA)

    def avvia(self) -> None:
        self._riarmato = 0.0
        self._riarma()
        self._filo.start()

    def ferma(self) -> None:
        self._fine.set()
        faulthandler.cancel_dump_traceback_later()
        if self._filo.is_alive() and self._filo is not threading.current_thread():
            self._filo.join(timeout=5)
        self._f.close()


@contextmanager
def sentinella(registro: Path, secondi: float, log: Callable[[str], None],
               controllo_gpu: Optional[Callable[[], str]] = None, margine: float = DEFAULT_MARGINE_S,
               uscita: Callable[[int], None] = os._exit):
    """Sentinella accesa per la durata del blocco (secondi <= 0: nulla). Una sola alla volta nel processo."""
    if not secondi or secondi <= 0:
        yield None
        return
    if _attiva[0] is not None:
        raise RuntimeError("sentinella gia' accesa in questo processo")
    s = Sentinella(Path(registro), secondi, log, controllo_gpu, margine, uscita)
    segno()
    _attiva[0] = s
    s.avvia()
    log(f"sentinella di progresso accesa: {secondi:.0f} s senza progresso -> pile nel registro e uscita "
        f"{CODICE_USCITA} (rete in C a {secondi + margine:.0f} s); E2E_SENTINELLA_S=0 la spegne")
    try:
        yield s
    finally:
        _attiva[0] = None
        s.ferma()

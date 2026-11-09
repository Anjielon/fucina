#!/usr/bin/env python3
"""CACHE PER LOTTO della forgia scorrevole — la faccia che vedono valida e forgia.

25/9/2026, progetto `PROGETTO_CACHE_LOTTI.md`. Interruttore `THOR_CACHE_LOTTI` (SPENTO di
default): spento, ogni funzione qui e' il comportamento di prima (`deposito()` da' un dict
vero, `in_deposito` l'identita', `contenitori_seme` None, precarica/chiudi non fanno
nulla) e la forgia non cambia di un bit.

## Perche'
La diagnosi del 25/9 ha mostrato sovra-adattamento (32 lotti ripassati ~9 volte per strato);
EfficientQAT (Block-AP) riduce il divario addestramento/validazione con piu' campioni a
parita' di passi. 150 lotti pero' non stanno sulla GPU con la casa accesa (misurato: ~0,19
GiB/lotto -> 77,5 riservati + 33 = 110 > 95,15). Lo stato per lotto (seme h/e, ingresso
quantizzato hq, uscite del teacher) passa in un DEPOSITO: i primi R lotti restano sulla GPU
(lo STESSO oggetto, semantica di un dict), gli altri vanno su disco in byte grezzi
(`deposito_io.Magazzino`). Al passo si rilegge UN lotto, precaricato durante il passo prima.

## Esattezza (regola 1: nessuna scorciatoia)
Nessun valore si ricalcola ne' si converte: byte copiati, tensore NUOVO con dtype, forma,
stride e device originali, impronta intera controllata a ogni rilettura. Un valore gia'
depositato (stesso oggetto, `_version` invariato) non si riscrive: il nuovo nome prende un
riferimento al file che c'e' (PROVENIENZA: seme = h0 locale = teacher[0]; teacher[1] =
cattura = seme h della finestra dopo) e una rilettura di un file il cui oggetto e' ancora
vivo restituisce QUELL'oggetto (come oggi `cache[b][0] is h0_per_b[b]`).

## Ciclo di vita (critiche 1, 4, 7, 8, 9)
- una CORSA (`apri_corsa`/`chiudi_corsa`, la forgia nel suo try/finally) = un Magazzino =
  una cartella con uuid; `chiudi_corsa` non solleva se un'altra eccezione e' gia' in volo
  (la scrive nel log: non deve mascherare il guasto vero);
- valida usata da sola a interruttore acceso: la corsa si apre PIGRA al primo `deposito()`
  (R = 0, tutto su disco) e si chiude con `atexit` o al primo `apri_corsa` esplicito;
- `Deposito.chiudi()` e' idempotente (la seconda volta: no-op scritto nel log); i depositi
  mai chiusi (eccezione) si rilasciano con `weakref.finalize` in CODA, svuotata dal thread
  principale, e dopo `chiudi_corsa` il finalize non fa nulla (legato all'istanza);
- `contenitori_seme(k)` crea SEMPRE depositi nuovi, distinti dal deposito di cattura: la
  cattura si chiude una volta sola (forgia, dopo il confine) e il seme resta vivo coi suoi
  riferimenti ai file (provenienza).

## Regola di residenza (budget `THOR_CACHE_LOTTI_GPU_GIB`, §3.4)
R = floor(budget / byte_per_lotto), byte_per_lotto = h x (1 + W) + e_max (e_max fp32, il
seme B12 della finestra 0): 160 MiB su Thor, R = 25 con 4 GiB. Classe PASSO: lotto b < R
residente (lo stesso oggetto). Classe UNA_VOLTA (hq): sempre su disco. Un valore gia'
residente in un altro deposito resta residente (stesso storage, zero byte in piu': la
Cassa conta per storage). Senza `fissa_residenti` (valida da sola) R = 0.

Variabili (lette a OGNI chiamata; tutte fuori ricetta, `forgia_scorrevole._NON_RICETTA`):
  THOR_CACHE_LOTTI            "" o "0" spento (default), "1" acceso, altro -> ValueError (critica 14)
  THOR_CACHE_LOTTI_GPU_GIB    budget GPU dello stato per lotto residente (default 0 = tutto su disco; "inf" ammesso)
  THOR_CACHE_LOTTI_DIR        radice (default @@ROOT:models@@/gguf/tony-tern/cache_lotti; mai tmpfs)
  THOR_CACHE_LOTTI_RISERVA_GIB spazio che deve restare libero dopo ogni scrittura (default 8)
  THOR_CACHE_LOTTI_VERIFICA   impronta (default) | completa (copia ombra in RAM host, solo collaudo)
"""
from __future__ import annotations

import atexit
import collections.abc
import inspect
import math
import os
import queue
import sys
import threading
import weakref
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import deposito_io  # noqa: E402
from deposito_io import DiscoPienoCache, libera_pagine, log  # noqa: E402,F401 — ri-esportati

PASSO = "passo"            # letto a ogni passo: residente se b < R
UNA_VOLTA = "una_volta"    # letto una volta (hq, al congelamento): sempre su disco
RADICE_PREDEFINITA = "@@ROOT:models@@/gguf/tony-tern/cache_lotti"
BYTE_E_MAX = 4             # l'errore del seme della finestra 0 e' fp32 (B12)


# ═══════════════════════ interruttori ═══════════════════════
def attiva() -> bool:
    """THOR_CACHE_LOTTI: assente, "" o "0" -> spento; "1" -> acceso; altro -> ValueError.
    La stringa vuota vale spento (critica 14): `X="${X:-}"` in un lanciatore non deve rompere nulla."""
    v = os.environ.get("THOR_CACHE_LOTTI", "")
    if v in ("", "0"):
        return False
    if v == "1":
        return True
    raise ValueError(f"THOR_CACHE_LOTTI={v!r}: attesi '1' (acceso), '0' o vuoto (spento)")


def _numero(nome: str, predefinito: str, ammetti_inf: bool = False) -> float:
    grezzo = os.environ.get(nome, "") or predefinito
    try:
        x = float(grezzo)
    except ValueError:
        raise ValueError(f"{nome}={grezzo!r}: atteso un numero") from None
    if math.isnan(x) or x < 0 or (math.isinf(x) and not ammetti_inf):
        raise ValueError(f"{nome}={grezzo!r}: atteso un numero >= 0" + (" (o inf)" if ammetti_inf else ""))
    return x


def budget_byte() -> float:
    return _numero("THOR_CACHE_LOTTI_GPU_GIB", "0", ammetti_inf=True) * 2 ** 30


def riserva_byte() -> int:
    return int(_numero("THOR_CACHE_LOTTI_RISERVA_GIB", "8") * 2 ** 30)


def radice() -> Path:
    return Path(os.environ.get("THOR_CACHE_LOTTI_DIR", "") or RADICE_PREDEFINITA)


def modo_verifica() -> str:
    return os.environ.get("THOR_CACHE_LOTTI_VERIFICA", "") or "impronta"


# ═══════════════════════ cassa (budget per storage) ═══════════════════════
class Cassa:
    """Byte residenti contati PER STORAGE: lo stesso tensore in tre depositi conta una volta.
    Un'ammissione oltre il budget va su disco (scritto nel log una volta sola)."""

    def __init__(self, budget: float = 0.0):
        self.budget = budget
        self.usati = 0
        self.massimo = 0
        self._per_storage: dict = {}
        self._avvisato = False
        self._lock = threading.Lock()

    @staticmethod
    def _chiave(t: torch.Tensor):
        return (t.device.type, t.device.index, t.untyped_storage().data_ptr())

    def gia_residente(self, t: torch.Tensor) -> bool:
        with self._lock:
            k = self._chiave(t)
            if k in self._per_storage:
                self._per_storage[k][0] += 1
                return True
            return False

    def ammetti(self, t: torch.Tensor, nome: str, lotto) -> bool:
        n = t.untyped_storage().nbytes()
        with self._lock:
            k = self._chiave(t)
            if k in self._per_storage:
                self._per_storage[k][0] += 1
                return True
            if self.usati + n > self.budget:
                if not self._avvisato:
                    self._avvisato = True
                    log(f"cache lotti: {nome} lotto {lotto} supererebbe il budget GPU ({self.usati / 2**30:.2f} + "
                        f"{n / 2**30:.3f} > {self.budget / 2**30:.2f} GiB): va su disco (avviso unico)")
                return False
            self._per_storage[k] = [1, n]
            self.usati += n
            self.massimo = max(self.massimo, self.usati)
            return True

    def rilascia(self, t: torch.Tensor) -> None:
        with self._lock:
            k = self._chiave(t)
            voce = self._per_storage.get(k)
            if voce is None:
                return
            voce[0] -= 1
            if voce[0] == 0:
                del self._per_storage[k]
                self.usati -= voce[1]


# ═══════════════════════ la corsa ═══════════════════════
class _Corsa:
    def __init__(self, nome: str, protette, implicita: bool):
        self.mag = deposito_io.Magazzino(radice(), nome, protette=protette, riserva_byte=riserva_byte(),
                                         verifica=modo_verifica())
        self.cassa = Cassa(0.0)
        self.R = 0
        self.nome = nome
        self.implicita = implicita
        self.chiusa = False
        self.n_lotti = None
        self.rilasci = queue.SimpleQueue()     # rilasci dai finalize (sicura dentro una callback del GC)
        self.depositi: "weakref.WeakSet" = weakref.WeakSet()

    def drena(self) -> None:
        """Rilasci accodati dai finalize dei depositi mai chiusi: SOLO dal thread principale."""
        while True:
            try:
                stato = self.rilasci.get_nowait()
            except queue.Empty:
                return
            if not self.chiusa:
                stato.rilascia_tutto(self)

    def chiudi(self) -> None:
        if self.chiusa:
            return
        self.chiusa = True
        for d in list(self.depositi):
            d._invalida()
        self.mag.chiudi()


_LOCK = threading.RLock()
_CORSA: "_Corsa | None" = None


def _protette_predefinite() -> list:
    """Uscita e ripresa note senza chiedere a valida (niente import circolare)."""
    p = []
    V = sys.modules.get("valida_finestra_8_12")
    if V is not None and getattr(V, "OUT_DIR", None):
        p.append(Path(V.OUT_DIR))
    for var in ("THOR_VALIDA_OUT", "THOR_SCORREVOLE_RIPRESA_DIR"):
        if os.environ.get(var):
            p.append(Path(os.environ[var]))
    return p


def _corsa() -> _Corsa:
    """La corsa aperta; se non c'e' (valida usata da sola), una corsa IMPLICITA con R = 0,
    chiusa da atexit o dal primo apri_corsa (critica 7)."""
    global _CORSA
    with _LOCK:
        if _CORSA is None or _CORSA.chiusa:
            _CORSA = _Corsa("valida", _protette_predefinite(), implicita=True)
            atexit.register(_chiudi_implicita, _CORSA)
            log("cache lotti: corsa IMPLICITA aperta (valida senza apri_corsa): R = 0, tutto su disco")
        return _CORSA


def _chiudi_implicita(c: _Corsa) -> None:
    global _CORSA
    with _LOCK:
        if _CORSA is c:
            _CORSA = None
    try:
        c.chiudi()
    except Exception as e:  # noqa: BLE001 — all'uscita del processo si scrive e basta
        log(f"⚠️ cache lotti: chiusura della corsa implicita fallita: {e!r}")


def apri_corsa(corsa: str, n_lotti: int, protette=()) -> None:
    """Forgia: apre la corsa della cache (pulisce gli orfani marcati). Spenta: no-op."""
    global _CORSA
    if not attiva():
        return
    with _LOCK:
        if _CORSA is not None and not _CORSA.chiusa:
            if not _CORSA.implicita:
                raise RuntimeError(f"cache lotti: corsa {_CORSA.nome!r} gia' aperta (chiudi_corsa mancante)")
            _chiudi_implicita(_CORSA)
        _CORSA = _Corsa(corsa, list(protette) + _protette_predefinite(), implicita=False)
        _CORSA.n_lotti = int(n_lotti)


def fissa_residenti(ids_batch: torch.Tensor, hidden: int, byte_h: int, w: int) -> None:
    """Budget -> R (una volta per corsa, alla prima finestra) e controllo dello spazio al picco
    (§7.5: lotti su disco x byte_per_lotto + tutti gli hq). Spenta: no-op."""
    if not attiva():
        return
    c = _corsa()
    n_elem = int(ids_batch.shape[1]) * int(ids_batch.shape[2]) * int(hidden)
    h = n_elem * int(byte_h)
    per_lotto = h * (1 + int(w)) + n_elem * BYTE_E_MAX
    budget = budget_byte()
    n = int(ids_batch.shape[0])
    R = n if math.isinf(budget) else min(n, int(budget // per_lotto)) if per_lotto else n
    c.R, c.cassa.budget = R, budget
    picco = (n - R) * per_lotto + n * h
    libero = c.mag.spazio_libero()
    log(f"cache lotti: {n} lotti · {per_lotto / 2**20:.1f} MiB/lotto (h {h / 2**20:.1f} x {1 + w} + e fp32) · "
        f"budget GPU {budget / 2**30:.2f} GiB -> R = {R} residenti · picco su disco ~{picco / 2**30:.2f} GiB "
        f"(liberi {libero / 2**30:.1f}, riserva {c.mag.riserva / 2**30:.1f})")
    if libero < picco + c.mag.riserva:
        raise DiscoPienoCache(28, f"cache lotti: servono ~{(picco + c.mag.riserva) / 2**30:.1f} GiB su "
                                  f"{c.mag.cartella.parent}, liberi {libero / 2**30:.1f}: rifiuto la partenza")


def chiudi_corsa() -> None:
    """Fine corsa (nel finally della forgia). Se un'altra eccezione e' gia' in volo, un errore
    di pulizia si scrive nel log e NON si solleva (non deve mascherare il guasto vero)."""
    global _CORSA
    with _LOCK:
        c, _CORSA = _CORSA, None
    if c is None:
        return
    in_volo = sys.exc_info()[0] is not None
    try:
        c.chiudi()
    except BaseException as e:
        if in_volo:
            log(f"⚠️ cache lotti: pulizia fallita ({e!r}) mentre un altro errore e' in corso: scritto, non sollevato")
            return
        raise


def corsa_aperta() -> "_Corsa | None":
    return _CORSA if _CORSA is not None and not _CORSA.chiusa else None


# ═══════════════════════ il deposito ═══════════════════════
class _Posto:
    __slots__ = ("i",)

    def __init__(self, i: int):
        self.i = i


class _Foglia:
    """Un tensore del deposito: residente (`t`) oppure su file (`fid`)."""
    __slots__ = ("t", "fid")

    def __init__(self, t=None, fid=None):
        self.t, self.fid = t, fid


class Riferimento:
    """Una foglia di un deposito passata a un altro deposito SENZA leggerla (provenienza pura:
    `uscita_teacher_fine`, critica 5c)."""
    __slots__ = ("foglia",)

    def __init__(self, foglia: _Foglia):
        self.foglia = foglia


def _scomponi(valore, foglie: list):
    if valore is None:
        return None
    if isinstance(valore, (torch.Tensor, Riferimento)):
        foglie.append(valore)
        return _Posto(len(foglie) - 1)
    if type(valore) is list:
        return [_scomponi(v, foglie) for v in valore]
    if type(valore) is tuple:
        return tuple(_scomponi(v, foglie) for v in valore)
    if type(valore) is dict:
        return {k: _scomponi(v, foglie) for k, v in valore.items()}
    raise TypeError(f"deposito: valore {type(valore).__name__} non ammesso (Tensor, None, list/tuple/dict di questi)")


def _ricomponi(modello, valori: list):
    if modello is None:
        return None
    if isinstance(modello, _Posto):
        return valori[modello.i]
    if type(modello) is list:
        return [_ricomponi(m, valori) for m in modello]
    if type(modello) is tuple:
        return tuple(_ricomponi(m, valori) for m in modello)
    return {k: _ricomponi(m, valori) for k, m in modello.items()}


def _controlla_denso(t: torch.Tensor, nome: str, lotto) -> None:
    """§3.7: rifiuto rumoroso, mai un .contiguous() d'ufficio (cambierebbe lo stride e con
    esso, forse, il kernel scelto) ne' un cast."""
    motivo = None
    if t.requires_grad or t.grad_fn is not None:
        motivo = "requires_grad/grad_fn (fuori dal grafo: .detach() a monte)"
    elif not t.is_contiguous():
        motivo = f"non contiguo (stride {tuple(t.stride())})"
    elif t.storage_offset() != 0:
        motivo = f"vista con offset {t.storage_offset()}"
    elif t.untyped_storage().nbytes() != t.numel() * t.element_size():
        motivo = f"vista su uno storage piu' grande ({t.untyped_storage().nbytes()} byte per {t.numel()} elementi)"
    if motivo:
        raise ValueError(f"deposito {nome}, lotto {lotto}: tensore {tuple(t.shape)} {t.dtype} rifiutato: {motivo}")


class _StatoDeposito:
    """Le voci di un deposito, separate dall'oggetto: il finalize le rilascia senza tenerlo vivo."""

    def __init__(self):
        self.voci: dict = {}
        self.chiuso = False

    def rilascia_voce(self, corsa: _Corsa, voce) -> None:
        for fo in voce[1]:
            if fo.t is not None:
                corsa.cassa.rilascia(fo.t)
            else:
                corsa.mag.decref(fo.fid)

    def rilascia_tutto(self, corsa: _Corsa) -> None:
        voci, self.voci = self.voci, {}
        for voce in voci.values():
            self.rilascia_voce(corsa, voce)


def _finalizza(ref_corsa, stato: _StatoDeposito) -> None:
    c = ref_corsa()
    if c is not None and not c.chiusa:
        c.rilasci.put(stato)           # SimpleQueue: sicura dentro una callback del GC


class Deposito(collections.abc.MutableMapping):
    """Dizionario {lotto b: valore} (valore = Tensor, None, list/tuple/dict annidati) con i
    lotti oltre il budget su disco. Stessa struttura, dtype, forma, stride, device e
    requires_grad (False) al ritorno. Vedi il docstring del modulo per le regole.

    ⚠️ Contratto: i valori depositati NON si modificano sul posto (deposito_io, limiti)."""

    def __init__(self, corsa: _Corsa, nome: str, finestra=None, classe: str = PASSO):
        if classe not in (PASSO, UNA_VOLTA):
            raise ValueError(f"classe {classe!r}: attese {PASSO!r} o {UNA_VOLTA!r}")
        self._corsa = corsa
        self.nome = nome if finestra is None else f"{nome}@{finestra}"
        self.classe = classe
        self._stato = _StatoDeposito()
        self._fin = weakref.finalize(self, _finalizza, weakref.ref(corsa), self._stato)
        corsa.depositi.add(self)

    def _viva(self) -> _Corsa:
        if self._corsa.chiusa:
            raise RuntimeError(f"deposito {self.nome}: la corsa della cache e' chiusa")
        if self._stato.chiuso:
            raise RuntimeError(f"deposito {self.nome}: gia' chiuso")
        return self._corsa

    def _ammetti(self, c: _Corsa, x, b):
        if isinstance(x, Riferimento):
            fo = x.foglia
            if fo.t is None:
                c.mag.incref(fo.fid)
                return _Foglia(fid=fo.fid)
            x = fo.t
        _controlla_denso(x, self.nome, b)
        if c.cassa.gia_residente(x):
            return _Foglia(t=x)
        if self.classe == PASSO and isinstance(b, int) and 0 <= b < c.R and c.cassa.ammetti(x, self.nome, b):
            return _Foglia(t=x)
        fid = c.mag.trova(x)
        if fid is not None:
            c.mag.incref(fid)
            return _Foglia(fid=fid)
        return _Foglia(fid=c.mag.scrivi(x, self.nome, b))

    def __setitem__(self, b, valore) -> None:
        c = self._viva()
        c.drena()
        c.mag.controlla()
        foglie: list = []
        modello = _scomponi(valore, foglie)
        nuove: list = []
        try:
            for x in foglie:
                nuove.append(self._ammetti(c, x, b))
        except BaseException:
            self._stato.rilascia_voce(c, (None, nuove))
            raise
        vecchia = self._stato.voci.get(b)
        self._stato.voci[b] = (modello, nuove)
        if vecchia is not None:                # prima i nuovi riferimenti, poi il rilascio: stesso oggetto = saldo 0
            self._stato.rilascia_voce(c, vecchia)

    def _valori(self, b, registra: bool):
        modello, foglie = self._stato.voci[b]
        mag = self._corsa.mag
        return _ricomponi(modello, [fo.t if fo.t is not None else mag.leggi(fo.fid, registra=registra)
                                    for fo in foglie])

    def __getitem__(self, b):
        c = self._viva()
        c.drena()
        return self._valori(b, True)

    def leggi_per_scrittura(self, b):
        """Dal thread del punto di ripresa: stessi valori, stesso device (quindi `torch.save`
        scrive gli stessi byte di oggi); non tocca precariche ne' provenienza del principale."""
        self._viva()
        return self._valori(b, False)

    def riferimento(self, b, indice: int) -> Riferimento:
        """Foglia `indice` (posizione nella lista del valore) del lotto b, per un altro deposito, senza leggerla."""
        self._viva()
        modello, foglie = self._stato.voci[b]
        posto = modello[indice]
        if not isinstance(posto, _Posto):
            raise TypeError(f"deposito {self.nome}: la voce {indice} del lotto {b} non e' un tensore")
        return Riferimento(foglie[posto.i])

    def __delitem__(self, b) -> None:
        c = self._viva()
        c.drena()
        voce = self._stato.voci.pop(b)
        self._stato.rilascia_voce(c, voce)

    # identita', non contenuto: il __eq__ di Mapping rileggerebbe TUTTI i lotti (e renderebbe il
    # deposito non hashabile, che serve al WeakSet della corsa)
    def __eq__(self, altro) -> bool:
        return self is altro

    __hash__ = object.__hash__

    def __iter__(self):
        return iter(sorted(self._stato.voci))

    def __len__(self) -> int:
        return len(self._stato.voci)

    def __contains__(self, b) -> bool:
        return b in self._stato.voci

    def clear(self) -> None:
        c = self._viva()
        c.drena()
        self._stato.rilascia_tutto(c)

    def precarica(self, b) -> None:
        """Lettura anticipata del lotto b (thread di I/O). Chiave assente: no-op."""
        c = self._viva()
        voce = self._stato.voci.get(b)
        if voce is None:
            return
        for fo in voce[1]:
            if fo.t is None:
                c.mag.precarica(fo.fid)

    def chiudi(self) -> None:
        """Rilascia tutto (i file a zero riferimenti si cancellano in coda). Idempotente."""
        if self._stato.chiuso:
            log(f"cache lotti: deposito {self.nome} gia' chiuso: chiusura ripetuta, nessun effetto")
            return
        self._fin.detach()
        self._corsa.depositi.discard(self)
        if not self._corsa.chiusa:
            self._corsa.drena()
            self._stato.rilascia_tutto(self._corsa)
        self._stato.voci = {}
        self._stato.chiuso = True

    def _invalida(self) -> None:
        """Dalla chiusura della corsa: niente I/O (la cartella se ne va intera), solo i riferimenti GPU."""
        self._fin.detach()
        self._stato.voci = {}
        self._stato.chiuso = True

    @property
    def chiuso(self) -> bool:
        return self._stato.chiuso

    def __repr__(self) -> str:
        return f"<Deposito {self.nome} {self.classe} · {len(self)} lotti{' · CHIUSO' if self.chiuso else ''}>"


# ═══════════════════════ funzioni di comodo (spente = comportamento di prima) ═══════════════════════
def deposito(nome: str, finestra=None, classe: str = PASSO) -> "dict | Deposito":
    """Spenta: `{}` (un dict vero). Accesa: un Deposito nella corsa aperta (o implicita)."""
    if not attiva():
        return {}
    return Deposito(_corsa(), nome, finestra, classe)


def contenitori_seme(finestra) -> "tuple | None":
    """Spenta: None. Accesa: (h, e) — depositi NUOVI (mai il deposito di cattura, critica 1)."""
    if not attiva():
        return None
    return deposito("seme_h", finestra), deposito("seme_e", finestra)


def contenitore_cattura(finestra) -> "Deposito | None":
    """Spenta: None (la cattura resta la lista di oggi). Accesa: deposito dell'uscita del teacher."""
    if not attiva():
        return None
    return deposito("cattura", finestra)


def _riversa(m, nome: str, finestra):
    if m is None or isinstance(m, Deposito):
        return m
    d = deposito(nome, finestra)
    for b in sorted(m):
        d[b] = m.pop(b)                    # il dizionario d'origine si svuota lotto per lotto
    return d


def in_deposito(coppia: tuple, finestra) -> tuple:
    """Spenta: `coppia` tale e quale. Accesa: (h, e) riversati in due depositi (i dict
    ricevuti si svuotano: sulla GPU resta al piu' un lotto in piu' alla volta)."""
    if not attiva():
        return coppia
    h, e = coppia
    return _riversa(h, "seme_h", finestra), _riversa(e, "seme_e", finestra)


def semina(seme_iniziale, ids_batch: torch.Tensor, emb: torch.Tensor, finestra: int = 0) -> tuple:
    """Seme della finestra 0. Spenta: `seme_iniziale(ids_batch, emb)`, identico. Accesa: se la
    funzione accetta `contenitori`, li riempie lotto per lotto (critica 2: mai 14 GiB sulla GPU),
    altrimenti il risultato si riversa nei depositi."""
    if not attiva():
        return seme_iniziale(ids_batch, emb)
    try:
        accetta = "contenitori" in inspect.signature(seme_iniziale).parameters
    except (TypeError, ValueError):
        accetta = False
    if accetta:
        cont = contenitori_seme(finestra)
        h, e = seme_iniziale(ids_batch, emb, contenitori=cont)
        if h is not cont[0] or e is not cont[1]:
            chiudi(*cont)
            return in_deposito((h, e), finestra)
        return h, e
    return in_deposito(seme_iniziale(ids_batch, emb), finestra)


def precarica(mappe, *lotti) -> None:
    """Letture anticipate dei `lotti` nelle `mappe` che sono depositi (dict e None: no-op)."""
    for m in mappe:
        if isinstance(m, Deposito):
            for b in lotti:
                m.precarica(b)


def chiudi(*mappe) -> None:
    """Chiude le mappe che sono depositi (dict e None: no-op)."""
    for m in mappe:
        if isinstance(m, Deposito):
            m.chiudi()


def leggi_per_scrittura(mappa, b):
    """Spenta (dict): `mappa[b]`. Deposito: lettura sicura dal thread del punto di ripresa."""
    return mappa.leggi_per_scrittura(b) if isinstance(mappa, Deposito) else mappa[b]


def libera_allocatore() -> None:
    """gc + empty_cache dopo aver riversato lo stato per lotto (critica 2): i segmenti da 32/64 MiB
    liberati non servono alle allocazioni grandi dell'ottimizzatore, e oltre la VRAM su questa
    macchina un'allocazione non fallisce ma scivola in GTT (torch non svuoterebbe da solo)."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

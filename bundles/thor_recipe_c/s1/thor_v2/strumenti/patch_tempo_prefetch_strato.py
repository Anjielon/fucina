"""Patch TEMPO «prefetch dello strato entrante» (leva A / L3 del piano) — 25/9 notte.

Nella forgia scorrevole ogni finestra [k,k+1] carica, all'inizio e in modo sincrono, lo strato
entrante k+1 (ramo pulito bf16 + ramo corrente GGUF + embedding ricaricato e buttato):
misurato 4-7 s in conf6b-S1 ([1,2] 7 s, [2,3] 5 s, [3,4] 4 s, [4,5] 4 s) e 7-11 s in conf6e-S2
([1,2] 8 s, [2,3] 7 s, [3,4] 11 s, [4,5] 12 s: li' il thread di scrittura lavorava in parallelo).
Qui lo strato entrante della finestra DOPO si carica in un thread DURANTE i passi di questa:
all'inizio della finestra dopo e' gia' pronto.

Perche' SOLO durante i passi (e non subito dopo il confine): `costruisci_strato` crea lo strato
dentro `accelerate.init_empty_weights()`, che sostituisce nn.Module.register_parameter per TUTTO
il processo finche' dura. Se in quel tratto il thread principale registrasse un parametro (la
costruzione dei TernarioDiffBatchFuso all'inizio della finestra, la rilegatura del raccordo alla
fine) il parametro finirebbe sul device 'meta'. Dentro il ciclo dei passi il thread principale
non crea ne' registra parametri (i pesi si sostituiscono via `_parameters`, il router via
`__dict__`): per questo ottimizza_finestra_JOINT riceve `durante_i_passi` — una funzione chiamata
subito PRIMA del primo passo, che avvia il caricamento e ritorna quella da chiamare subito DOPO
l'ultimo passo, che lo ASPETTA (il caricamento dura secondi, i passi minuti: l'attesa e' ~0).

  valida_finestra_8_12.py — `durante_i_passi` (default None = invariato).
  forgia_scorrevole.py    — THOR_SCORREVOLE_PREFETCH=1 (default; fuori ricetta): _PrefetchStrato.
    Un prefetch fallito non ferma la forgia: all'inizio della finestra lo strato si ricarica in
    modo sincrono, con il messaggio d'errore nel log (i numeri non dipendono da quando si carica).

NUMERI: IDENTICI AL BIT (stessi pesi, caricati prima). MEMORIA: durante i passi 3 strati
residenti invece di 2 (+4,08 GiB di VRAM: conf6e-S2 01:26:21 -> 01:26:27, "VRAM allocata"
4,09 -> 8,17 GiB caricando il secondo strato); RAM dell'host come oggi (lo stesso caricamento,
in un altro momento). Se il passo si interrompe con un'eccezione il caricamento si aspetta nel
`finally` della forgia (il processo si ferma comunque: lo stato vero e' il punto di ripresa).
DA MISURARE su GPU prima di adottarla: s/passo con il prefetch attivo (il trasferimento e la CPU
del caricamento si sovrappongono ai passi) e picco di RAM dell'host (swap gia' usato 18 GiB).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_tempo_comune import esegui_patch  # noqa: E402

MARCA = "patch_tempo_prefetch_strato"

VALIDA = [
    ('''                              adam_ripartenza: str = "riparti"''',
     '''                              adam_ripartenza: str = "riparti",
                              durante_i_passi: "object | None" = None'''),
    ('''        for t in range(n_passi):
            b = t % n_tr
''',
     '''        # ⭐ 25/9 (patch_tempo_prefetch_strato): lavoro di contorno SOLO durante i passi (il chiamante
        # carica lo strato della finestra dopo); qui dentro non si crea ne' registra alcun parametro
        _fine_durante = durante_i_passi() if durante_i_passi is not None else None
        for t in range(n_passi):
            b = t % n_tr
'''),
    ('''        if batch_ho and ROUTING_VIVO:
''',
     '''        if _fine_durante is not None:
            _fine_durante()             # (patch_tempo_prefetch_strato) aspetta la fine del lavoro di contorno
            _fine_durante = None
        if batch_ho and ROUTING_VIVO:
'''),
]

FORGIA = [
    ('''class InterruzioneSimulata(RuntimeError):''',
     '''# ⭐ 25/9 (patch_tempo_prefetch_strato): lo strato entrante della finestra dopo si carica durante i passi
PREFETCH_STRATO = os.getenv("THOR_SCORREVOLE_PREFETCH", "1") == "1"
_NON_RICETTA |= {"THOR_SCORREVOLE_PREFETCH"}


class _PrefetchStrato:
    """Un thread, uno strato alla volta. `avvia(L)` e' chiamata da ottimizza_finestra_JOINT subito
    prima del primo passo (argomento `durante_i_passi`) e ritorna `aspetta`, che la finestra
    chiama subito dopo l'ultimo passo: il caricamento (dentro init_empty_weights) non si
    sovrappone mai alla creazione di parametri del thread principale (vedi patch)."""

    def __init__(self, carica_strati):
        self._carica = carica_strati
        self._esecutore = None
        self._futuro = None
        self._strato = None

    def avvia(self, L: int):
        if self._futuro is not None:
            if self._strato != L:
                raise RuntimeError(f"prefetch: strato {self._strato} gia' in corso, chiesto {L}")
            return self.aspetta                      # piu' chunk nella stessa finestra: gia' avviato
        if self._esecutore is None:
            from concurrent.futures import ThreadPoolExecutor as _TPE
            self._esecutore = _TPE(max_workers=1, thread_name_prefix="prefetch_strato")
        self._strato = L
        self._futuro = self._esecutore.submit(self._carica, [L])
        log(f"  prefetch: carico lo strato {L} durante i passi")
        return self.aspetta

    def aspetta(self) -> None:
        if self._futuro is not None:
            from concurrent.futures import wait as _wait
            _wait([self._futuro])                    # finito (bene o male); l'errore esce a prendi()

    def ha(self, L: int) -> bool:
        return self._futuro is not None and self._strato == L

    def prendi(self, L: int):
        futuro, self._futuro, self._strato = self._futuro, None, None
        try:
            return futuro.result()
        except Exception as e:              # noqa: BLE001 — si ricarica in modo sincrono, a voce alta
            log(f"⚠️ prefetch dello strato {L} FALLITO ({type(e).__name__}: {e}): lo carico adesso, in modo sincrono")
            return self._carica([L])

    def chiudi(self) -> None:
        self.aspetta()
        self._futuro = self._strato = None
        if self._esecutore is not None:
            self._esecutore.shutdown(wait=True)
            self._esecutore = None


class InterruzioneSimulata(RuntimeError):'''),
    ('''    scrittore = ThreadPoolExecutor(max_workers=1)
    lavoro_prec = None
''',
     '''    scrittore = ThreadPoolExecutor(max_workers=1)
    lavoro_prec = None
    # ⭐ 25/9 (patch_tempo_prefetch_strato)
    prefetch = _PrefetchStrato(carica_strati) if PREFETCH_STRATO else None
    if prefetch is not None and "durante_i_passi" not in inspect.signature(V.ottimizza_finestra_JOINT).parameters:
        raise RuntimeError("THOR_SCORREVOLE_PREFETCH=1 ma valida_finestra_8_12 non ha 'durante_i_passi' "
                           "(patch_tempo_prefetch_strato applicata a meta')")
'''),
    ('''                cfg_n, emb_n, p_n, q_n = carica_strati([L])
''',
     '''                cfg_n, emb_n, p_n, q_n = (prefetch.prendi(L) if prefetch is not None and prefetch.ha(L)
                                          else carica_strati([L]))       # (patch_tempo_prefetch_strato)
'''),
    ('''            kw = {PARAM_INIZIALI: iniziali or None, PARAM_USCITA: uscita}
''',
     '''            kw = {PARAM_INIZIALI: iniziali or None, PARAM_USCITA: uscita}
            if prefetch is not None and k1 < n_strati:
                # (patch_tempo_prefetch_strato) lo strato entrante della finestra dopo, durante i passi
                kw["durante_i_passi"] = (lambda _L=k1: prefetch.avvia(_L))
'''),
    ('''    finally:
        scrittore.shutdown(wait=True)
''',
     '''    finally:
        if prefetch is not None:
            prefetch.chiudi()                 # (patch_tempo_prefetch_strato) nessun thread lasciato vivo
        scrittore.shutdown(wait=True)
'''),
]

PIANO = {"valida_finestra_8_12.py": VALIDA, "forgia_scorrevole.py": FORGIA}

if __name__ == "__main__":
    raise SystemExit(esegui_patch(__doc__.splitlines()[0], PIANO, MARCA))

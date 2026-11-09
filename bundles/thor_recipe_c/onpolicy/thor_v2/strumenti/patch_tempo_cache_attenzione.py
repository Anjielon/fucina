"""Patch TEMPO «cache dell'attenzione del primo strato» (leva L1 del piano) — 25/9 notte.

Nel passo di addestramento il PRIMO strato della finestra riceve h+e del batch: un ingresso
FISSO per tutta la finestra (il seme non cambia), e i pesi della sua attenzione (lineare o
piena, ramo corrente GGUF) sono congelati. Oggi a ogni visita del batch (150 passi / 32 batch
= 4,7 visite) il passo rifa' `input_layernorm` + attenzione di quello strato sullo stesso
ingresso: lo stesso identico conto. Il congelamento del routing all'inizio della finestra lo
calcola gia' (no_grad, stesso modulo, stesso ingresso hq = h + e.to(bf16)): qui un gancio ne
tiene l'uscita, e nei passi quel modulo restituisce l'uscita tenuta (e input_layernorm, la cui
uscita va SOLO all'attenzione, non si calcola). Tutto il resto dello strato — somma residua,
post_attention_layernorm, router congelato, esperti con i pesi che si allenano, esperto
condiviso — gira COME OGGI, con il codice di transformers: nessuna formula riscritta.

Perche' il gradiente non cambia: l'ingresso del primo strato non chiede gradiente e i pesi
dell'attenzione sono congelati (requires_grad False dal 24/9): nell'indietro l'attenzione del
primo strato non c'e' mai stata. Si risparmia solo il suo avanti.

Guadagno STIMATO (NON MISURATO): il blocco norme+attenzione+condiviso vale ~129 ms per strato
in avanti (py-spy della corsa conf6b-S1, rapporto costo_passo.md §2.3, intervallo 95-150), di cui
qui si toglie la parte norma d'ingresso + attenzione. Da misurare con la radiografia per blocco.

VERIFICA AL BIT nella corsa vera: THOR_CACHE_ATTENZIONE_VERIFICA=prima (default) -> al passo 0
di ogni finestra il primo strato gira per intero e la sua uscita d'attenzione (calcolata col
gradiente abilitato) deve essere torch.equal a quella tenuta dal congelamento (no_grad): se no,
la forgia si FERMA con il messaggio per spegnere la leva. =ogni_batch -> lo stesso controllo
alla prima visita di OGNI batch (costa una attenzione in piu' per batch). Solo nel primo chunk
della finestra (in produzione la finestra e' un chunk solo), solo con routing congelato
(THOR_ROUTING_VIVO=0: il congelamento e' il punto di cattura), solo con il gradiente abilitato
(il F̂(h) diagnostico, senza e, fa il percorso normale).

Interruttore THOR_CACHE_ATTENZIONE=0 (default 1; fuori ricetta nella forgia scorrevole).
NUMERI: IDENTICI AL BIT (verificato a ogni finestra). MEMORIA: +1 uscita d'attenzione per batch
(33 x 8 x 1024 x 2048 bf16 = 1,03 GiB) durante la finestra.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_tempo_comune import esegui_patch  # noqa: E402

MARCA = "patch_tempo_cache_attenzione"

VALIDA = [
    # -- 1. interruttori e mattoni (prima della funzione, aggancio condiviso: resta unico)
    ('''def ottimizza_finestra_JOINT(strati_correnti: dict, ids_batch: torch.Tensor, cfg, n_passi: int,''',
     '''# ⭐ 25/9 (patch_tempo_cache_attenzione): uscita d'attenzione del PRIMO strato della finestra, tenuta dal
# congelamento del routing e riusata nei passi (stesso modulo congelato, stesso ingresso fisso h+e)
CACHE_ATTENZIONE = os.getenv("THOR_CACHE_ATTENZIONE", "1") == "1"
CACHE_ATTENZIONE_VERIFICA = os.getenv("THOR_CACHE_ATTENZIONE_VERIFICA", "prima")
if CACHE_ATTENZIONE_VERIFICA not in ("prima", "ogni_batch"):
    raise ValueError(f"THOR_CACHE_ATTENZIONE_VERIFICA={CACHE_ATTENZIONE_VERIFICA!r}: attesi 'prima' o 'ogni_batch'")


def _modulo_attenzione(strato: torch.nn.Module) -> torch.nn.Module:
    tipo = getattr(strato, "block_type", None)
    if tipo == "linear_attention":
        return strato.linear_attn
    if tipo == "full_attention":
        return strato.self_attn
    raise RuntimeError(f"cache attenzione: tipo di strato {tipo!r} sconosciuto")


class _CatturaAttenzione:
    """Gancio sul modulo d'attenzione dello strato: le uscite (riferimenti, nessuna copia) nell'ordine
    delle chiamate. `per_batch(n)` pretende ESATTAMENTE n chiamate (una per batch)."""

    def __init__(self, strato: torch.nn.Module):
        self.modulo = _modulo_attenzione(strato)
        self.uscite: list = []
        self._maniglia = None

    def __enter__(self):
        self._maniglia = self.modulo.register_forward_hook(lambda _m, _a, o: self.uscite.append(o))
        return self

    def __exit__(self, *_):
        self._maniglia.remove()
        return False

    def per_batch(self, n: int) -> dict:
        if len(self.uscite) != n:
            raise RuntimeError(f"cache attenzione: {len(self.uscite)} chiamate all'attenzione del primo strato, "
                               f"attese {n} (una per batch nel congelamento del routing)")
        return {b: self.uscite[b] for b in range(n)}


@contextlib.contextmanager
def _attenzione_da_cache(strato: torch.nn.Module, uscita):
    """Per UN forward dello strato: input_layernorm non si calcola (la sua uscita va solo
    all'attenzione) e il modulo d'attenzione restituisce `uscita` (stessa struttura del suo forward:
    tensore per la lineare, (uscita, pesi) per la piena). Stesso meccanismo di _instrada_congelato."""
    norma, modulo = strato.input_layernorm, _modulo_attenzione(strato)
    if "forward" in norma.__dict__ or "forward" in modulo.__dict__:
        raise RuntimeError("cache attenzione: forward gia' sostituito (sostituzioni annidate?)")
    norma.__dict__["forward"] = lambda hidden_states: hidden_states
    modulo.__dict__["forward"] = lambda *_a, **_k: uscita
    try:
        yield
    finally:
        norma.__dict__.pop("forward", None)
        modulo.__dict__.pop("forward", None)


def _uguali_al_bit(a, b) -> bool:
    if isinstance(a, tuple) or isinstance(b, tuple):
        return (isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b)
                and all(_uguali_al_bit(x, y) for x, y in zip(a, b)))
    if a is None or b is None:
        return a is None and b is None
    return a.dtype == b.dtype and a.shape == b.shape and bool(torch.equal(a, b))


def _stratofn_con_cache_attenzione(fn, strato: torch.nn.Module, uscita_attn, verifica: bool, annuncia: bool,
                                   etichetta: str):
    """StratoFn del primo strato con l'attenzione dalla cache. `verifica`: questo passo fa il percorso
    NORMALE e confronta al bit l'uscita d'attenzione che produce con quella tenuta."""
    def f(h: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            return fn(h)                 # F̂(h) diagnostico: ingresso senza e, percorso normale
        if verifica:
            with _CatturaAttenzione(strato) as c:
                uscita = fn(h)
            vera = c.per_batch(1)[0]
            if not _uguali_al_bit(vera, uscita_attn):
                raise RuntimeError(f"cache attenzione ({etichetta}): l'uscita d'attenzione del passo NON e' identica al "
                                   "bit a quella del congelamento: la cache cambierebbe i numeri. Spegnere con "
                                   "THOR_CACHE_ATTENZIONE=0")
            if annuncia:
                log(f"  cache attenzione ({etichetta}): uscita del passo IDENTICA AL BIT a quella del congelamento")
            return uscita
        with _attenzione_da_cache(strato, uscita_attn):
            return fn(h)
    return f


def ottimizza_finestra_JOINT(strati_correnti: dict, ids_batch: torch.Tensor, cfg, n_passi: int,'''),
    # -- 2. il congelamento del routing tiene le uscite d'attenzione del primo strato
    ('''        if ROUTING_VIVO:
            # (None, None) = nessun instradamento imposto: decide il router vero a ogni forward
            congelati_per_b = {b: {L: (None, None) for L in chunk} for b in range(n_batch_loc)}
        else:
            _pronti = _ricostruisci_pesi_duri(chunk, strati_correnti, ternari_gu, ternari_dw, S)
            for b in range(n_batch_loc):
                congelati_per_b[b] = _congela_routing_chunk(chunk, strati_correnti, hq_per_b[b], _forward_grezzo,
                                                            ternari_gu=ternari_gu, ternari_dw=ternari_dw, S=S,
                                                            pesi_pronti=_pronti)
            del _pronti
''',
     '''        cache_attn = None          # (patch_tempo_cache_attenzione)
        if ROUTING_VIVO:
            # (None, None) = nessun instradamento imposto: decide il router vero a ogni forward
            congelati_per_b = {b: {L: (None, None) for L in chunk} for b in range(n_batch_loc)}
        else:
            _pronti = _ricostruisci_pesi_duri(chunk, strati_correnti, ternari_gu, ternari_dw, S)
            # (patch_tempo_cache_attenzione) solo il primo chunk: il suo ingresso hq e' ESATTAMENTE l'h+e del passo
            _usa_cache_attn = CACHE_ATTENZIONE and i0 == 0
            _cattura_attn = _CatturaAttenzione(strati_correnti[chunk[0]]) if _usa_cache_attn else contextlib.nullcontext()
            with _cattura_attn:
                for b in range(n_batch_loc):
                    congelati_per_b[b] = _congela_routing_chunk(chunk, strati_correnti, hq_per_b[b], _forward_grezzo,
                                                                ternari_gu=ternari_gu, ternari_dw=ternari_dw, S=S,
                                                                pesi_pronti=_pronti)
            del _pronti
            if _usa_cache_attn:
                cache_attn = _cattura_attn.per_batch(n_batch_loc)
                log(f"  chunk {chunk}: cache attenzione dello strato {chunk[0]} "
                    f"({getattr(strati_correnti[chunk[0]], 'block_type', '?')}): {n_batch_loc} uscite tenute dal "
                    f"congelamento, riusate nei passi (verifica al bit: {CACHE_ATTENZIONE_VERIFICA})")
            del _cattura_attn
'''),
    # -- 3. nei passi il primo strato usa la cache (col controllo al bit al passo 0 / alla prima visita)
    ('''                                       raccoglitore=_nuovi_routing, chiave=L)
                for L in chunk
            ]
''',
     '''                                       raccoglitore=_nuovi_routing, chiave=L)
                for L in chunk
            ]
            if cache_attn is not None:
                # (patch_tempo_cache_attenzione)
                strati_quant_fn[0] = _stratofn_con_cache_attenzione(
                    strati_quant_fn[0], strati_correnti[chunk[0]], cache_attn[b],
                    verifica=(t == 0) if CACHE_ATTENZIONE_VERIFICA == "prima" else (t < n_tr),
                    annuncia=(t == 0), etichetta=f"strato {chunk[0]}, batch {b}, passo {t}")
'''),
    # -- 4. finiti i passi la cache si libera
    ('''        if batch_ho and not ROUTING_VIVO:
''',
     '''        cache_attn = None          # (patch_tempo_cache_attenzione) passi finiti: si libera
        if batch_ho and not ROUTING_VIVO:
'''),
]

FORGIA = [
    ('''class InterruzioneSimulata(RuntimeError):''',
     '''# ⭐ 25/9 (patch_tempo_cache_attenzione): leva di velocita' identica al bit, fuori ricetta
_NON_RICETTA |= {"THOR_CACHE_ATTENZIONE", "THOR_CACHE_ATTENZIONE_VERIFICA"}


class InterruzioneSimulata(RuntimeError):'''),
]

PIANO = {"valida_finestra_8_12.py": VALIDA, "forgia_scorrevole.py": FORGIA}

if __name__ == "__main__":
    raise SystemExit(esegui_patch(__doc__.splitlines()[0], PIANO, MARCA))

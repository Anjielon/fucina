"""Patch TEMPO «attenzione del confine dalla cache» (leva L1b del piano) — 25/9 notte. RICHIEDE la
patch patch_tempo_cache_attenzione (L1) gia' applicata a valida_finestra_8_12.py: senza, l'aggancio
in valida manca e questa patch rifiuta senza scrivere nulla.

Alla fine della finestra [k,k+1] il confine a meta' (`calcola_confine_a_meta`) fa passare ogni
batch per lo strato k COMMESSO, sull'ingresso h_seme[b] + e_seme[b] (in bf16): lo STESSO ingresso
del primo strato nella finestra, dove la leva L1 ha gia' tenuto l'uscita dell'attenzione (pesi
d'attenzione congelati: il commit tocca solo gli esperti). Qui quell'uscita esce dalla finestra
(`cache_attenzione_uscita`) e nel confine l'attenzione dello strato k non si ricalcola: il batch 0
fa il percorso normale e si confronta AL BIT con la cache (se differisce la forgia si ferma e dice
come spegnere la leva), i batch 1..n-1 prendono la cache. Il confine chiama lo strato una volta per
batch, in ordine (V.calcola_confine_finestra): il contesto conta le chiamate e pretende n.

Guadagno STIMATO (NON MISURATO): come L1, ~0,09-0,145 s di attenzione per batch (rapporto
costo_passo.md §2.3: ~129 ms/strato per norme+attenzione+condiviso) x 32 batch → ~3-4,5 s a
finestra. NUMERI: IDENTICI AL BIT (verificato a ogni finestra sul batch 0). MEMORIA: la cache
d'attenzione (1,03 GiB) resta viva fino alla fine del confine (~10 s) invece di liberarsi a fine
passi. Interruttore THOR_SCORREVOLE_ATTENZIONE_CONFINE=0 (default 1; fuori ricetta).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_tempo_comune import esegui_patch  # noqa: E402

MARCA = "patch_tempo_attenzione_confine"

VALIDA = [
    ('''                              adam_ripartenza: str = "riparti"''',
     '''                              adam_ripartenza: str = "riparti",
                              cache_attenzione_uscita: "dict | None" = None'''),
    # aggancio scritto dalla patch L1: se L1 non c'e', questa patch rifiuta
    ('''                cache_attn = _cattura_attn.per_batch(n_batch_loc)
''',
     '''                cache_attn = _cattura_attn.per_batch(n_batch_loc)
                if cache_attenzione_uscita is not None:      # (patch_tempo_attenzione_confine)
                    cache_attenzione_uscita.clear()
                    cache_attenzione_uscita.update(cache_attn)
'''),
]

FORGIA = [
    ('''class InterruzioneSimulata(RuntimeError):''',
     '''# ⭐ 25/9 (patch_tempo_attenzione_confine): nel confine l'attenzione dello strato commesso viene dalla
# cache della finestra (stesso ingresso, pesi d'attenzione intatti dal commit); batch 0 verificato al bit
ATTENZIONE_CONFINE = os.getenv("THOR_SCORREVOLE_ATTENZIONE_CONFINE", "1") == "1"
_NON_RICETTA |= {"THOR_SCORREVOLE_ATTENZIONE_CONFINE"}


class _AttenzioneNelConfine:
    """Contesto attorno al confine di UNO strato: la i-esima chiamata dello strato (i = batch, in
    ordine) prende l'attenzione da cache[i]; la chiamata 0 la calcola davvero e la confronta al bit."""

    def __init__(self, strato, cache: dict, n_batch: int, k: int):
        if sorted(cache) != list(range(n_batch)):
            raise RuntimeError(f"confine {k}: cache d'attenzione per le batch {sorted(cache)[:5]}..., attese 0..{n_batch - 1}")
        self.strato, self.cache, self.n, self.k = strato, cache, n_batch, k
        self.norma = strato.input_layernorm
        self.modulo = V._modulo_attenzione(strato)
        self.chiamate = 0

    def __enter__(self):
        if "forward" in self.norma.__dict__ or "forward" in self.modulo.__dict__:
            raise RuntimeError(f"confine {self.k}: forward gia' sostituito (sostituzioni annidate?)")
        norma_vera = self.norma.forward
        modulo_vero = self.modulo.forward

        def norma(hidden_states):
            return norma_vera(hidden_states) if self.chiamate == 0 else hidden_states

        def attenzione(*a, **k):
            i = self.chiamate
            self.chiamate += 1
            if i >= self.n:
                raise RuntimeError(f"confine {self.k}: {i + 1} chiamate all'attenzione, attese {self.n}")
            if i == 0:
                vera = modulo_vero(*a, **k)
                if not V._uguali_al_bit(vera, self.cache[0]):
                    raise RuntimeError(f"confine {self.k}: attenzione del batch 0 NON identica al bit alla cache della "
                                       "finestra: la cache cambierebbe il seme. Spegnere con "
                                       "THOR_SCORREVOLE_ATTENZIONE_CONFINE=0")
                return vera
            return self.cache[i]
        self.norma.__dict__["forward"] = norma
        self.modulo.__dict__["forward"] = attenzione
        return self

    def __exit__(self, tipo, *_):
        self.norma.__dict__.pop("forward", None)
        self.modulo.__dict__.pop("forward", None)
        if tipo is None:
            if self.chiamate != self.n:
                raise RuntimeError(f"confine {self.k}: {self.chiamate} chiamate all'attenzione, attese {self.n}")
            log(f"  confine dello strato {self.k}: attenzione di {self.n - 1} batch dalla cache della finestra "
                "(batch 0 ricalcolato: identico al bit)")
        return False


class InterruzioneSimulata(RuntimeError):'''),
    ('''    giornale = cartella_out / "giornale_scorrevole.jsonl"
''',
     '''    giornale = cartella_out / "giornale_scorrevole.jsonl"
    if ATTENZIONE_CONFINE and "cache_attenzione_uscita" not in inspect.signature(V.ottimizza_finestra_JOINT).parameters:
        raise RuntimeError("THOR_SCORREVOLE_ATTENZIONE_CONFINE=1 ma valida_finestra_8_12 non ha "
                           "cache_attenzione_uscita (patch_tempo_attenzione_confine applicata a meta')")
'''),
    ('''            del iniziali
''',
     '''            del iniziali
            cache_a = {} if (ATTENZIONE_CONFINE and not ultima) else None     # (patch_tempo_attenzione_confine)
            if cache_a is not None:
                kw["cache_attenzione_uscita"] = cache_a
'''),
    ('''                h_seme, e_seme = calcola_confine_a_meta(puliti, quant, [k0], ids_batch, cfg, emb,
                                                        h_seme, e_seme, h_teacher_uscita_per_b=teacher_k)
''',
     '''                with (_AttenzioneNelConfine(quant[k0], cache_a, n_batch, k0) if cache_a
                      else contextlib.nullcontext()):     # (patch_tempo_attenzione_confine)
                    h_seme, e_seme = calcola_confine_a_meta(puliti, quant, [k0], ids_batch, cfg, emb,
                                                            h_seme, e_seme, h_teacher_uscita_per_b=teacher_k)
                cache_a = None
'''),
]

PIANO = {"valida_finestra_8_12.py": VALIDA, "forgia_scorrevole.py": FORGIA}

if __name__ == "__main__":
    raise SystemExit(esegui_patch(__doc__.splitlines()[0], PIANO, MARCA))

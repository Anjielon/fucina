"""Patch TEMPO «congelamento del routing senza il MoE dell'ultimo strato» (leva L6 del piano) — 25/9 notte.

`_congela_routing_chunk` fa passare ogni batch per TUTTI gli strati del chunk con un forward
completo, e di ogni strato tiene solo la scelta del router (calcolata dall'uscita di
`post_attention_layernorm`). L'uscita dell'ULTIMO strato non la usa nessuno (`h = h_next` e il
ciclo finisce): il suo blocco MoE — esperti (gate_up + down, 256 esperti top-8) ed esperto
condiviso — e' calcolato e buttato, per ognuna delle 33 batch a ogni finestra (piu' la batch
held-out del giudizio finale). Qui il gancio che cattura l'ingresso del router dell'ultimo
strato, dopo averlo preso, FERMA il forward con un'eccezione dedicata (`_FermaAlRouter`), che il
chiamante raccoglie: stessa cattura, stesso router, stesse scelte; manca solo il conto buttato.
I contesti di sostituzione (pesi duri, rotazione) si chiudono come sempre (try/finally).

Guadagno STIMATO (NON MISURATO): esperti avanti ~48 + ~27 ms per strato a 8 x 1024 token
(py-spy conf6b-S1, rapporto costo_passo.md §2.3: attese «b»/«c»/«d») → ~2,5 s a finestra sui
33 passaggi del congelamento (14,3 s per 66 passaggi misurati nel pre-passo di [3,4]).

NUMERI: IDENTICI AL BIT (l'uscita non usata non entra in nessun conto). Interruttore
THOR_CONGELO_SENZA_MOE_ULTIMO=0 (default 1; fuori ricetta nella forgia scorrevole).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_tempo_comune import esegui_patch  # noqa: E402

MARCA = "patch_tempo_congelo_senza_moe"

VALIDA = [
    ('''def _congela_routing_chunk(chunk: list, strati_correnti: dict, h0_quant: torch.Tensor, forward_grezzo,''',
     '''# ⭐ 25/9 (patch_tempo_congelo_senza_moe): l'ultimo strato del congelamento si ferma all'ingresso del router
CONGELO_SENZA_MOE_ULTIMO = os.getenv("THOR_CONGELO_SENZA_MOE_ULTIMO", "1") == "1"


class _FermaAlRouter(Exception):
    """Sollevata dal gancio su post_attention_layernorm dell'ULTIMO strato del congelamento, DOPO aver
    preso l'ingresso del router: il resto dello strato (MoE) calcolerebbe un'uscita che nessuno usa."""


def _avanti_fino_al_router(forward_grezzo, strato, h):
    try:
        return forward_grezzo(strato, h)
    except _FermaAlRouter:
        return None


def _congela_routing_chunk(chunk: list, strati_correnti: dict, h0_quant: torch.Tensor, forward_grezzo,'''),
    ('''            hook = strato.post_attention_layernorm.register_forward_hook(
                lambda _m, _i, o: catture.__setitem__("v", o.detach()))
''',
     '''            _fine_qui = CONGELO_SENZA_MOE_ULTIMO and L == chunk[-1]     # (patch_tempo_congelo_senza_moe)

            def _cattura_router(_m, _i, o, _catture=catture, _fine=_fine_qui):
                _catture["v"] = o.detach()
                if _fine:
                    raise _FermaAlRouter()
            hook = strato.post_attention_layernorm.register_forward_hook(_cattura_router)
'''),
    ('''                    h_next = forward_grezzo(strato, h)
                del gu, dw
''',
     '''                    h_next = _avanti_fino_al_router(forward_grezzo, strato, h)
                del gu, dw
'''),
    ('''            else:
                h_next = forward_grezzo(strato, h)
            hook.remove()
''',
     '''            else:
                h_next = _avanti_fino_al_router(forward_grezzo, strato, h)
            hook.remove()
'''),
]

FORGIA = [
    ('''class InterruzioneSimulata(RuntimeError):''',
     '''# ⭐ 25/9 (patch_tempo_congelo_senza_moe): leva di velocita' identica al bit, fuori ricetta
_NON_RICETTA |= {"THOR_CONGELO_SENZA_MOE_ULTIMO"}


class InterruzioneSimulata(RuntimeError):'''),
]

PIANO = {"valida_finestra_8_12.py": VALIDA, "forgia_scorrevole.py": FORGIA}

if __name__ == "__main__":
    raise SystemExit(esegui_patch(__doc__.splitlines()[0], PIANO, MARCA))

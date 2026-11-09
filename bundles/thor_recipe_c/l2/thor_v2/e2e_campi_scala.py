#!/usr/bin/env python3
"""E2E sulle scale (progetto `e2e_progetto_1.md`): i campi `d` dei blocchi ternari
del GGUF, letti e riscritti AL BIT, e i trit decodificati in torch dagli stessi byte.

Scritto il 25/9/2026. File NUOVO: non importa ne' modifica nessun file della forgia
in corsa (vincolo di sessione); dipende solo da gguf-py della build k1024.

Due formati, layout verificati sul sorgente (non ricordati):

| tipo          | byte/blocco | pesi/blocco | scale/blocco | campo d        | gruppo della scala |
|---------------|-------------|-------------|--------------|----------------|--------------------|
| TQ1_0R_1024   | 224         | 1024        | 8 (fp16)     | byte 208..223  | 128 pesi RUOTATI   |
| TQ1_0         | 54          | 256         | 1 (fp16)     | byte 52..53    | 256 pesi           |

Fonti: `kernel-rotazione/tq1_0r_1024_codec.py` (docstring del layout, identico al C
di ggml-quants.c della build k1024) e `pack_tq1_0_standard.py` (qs 48 + qh 4 + d 2).

Ordine dei blocchi di un tensore esperti GGUF di forma [ne0=in, ne1=out, ne2=E]:
blocco = (e*out + r)*nb + b, con nb = in / pesi_blocco. La scala g del blocco b della
riga r dell'esperto e va in D[e, r, b*scale_blocco + g]: e' la forma (E, out, ng) di
`valida_finestra_8_12.pesi_da_DQ` / `_estrai_famiglia` (verificato da
`e2e_test_campi_scala.py` contro `dequant_gguf_torch.dequant_np_slice`).

La decodifica dei trit ripete l'aritmetica intera di `dequantize_row_tq1_0`
((b*3^n) & 255) * 3 >> 8, cosi' i codici in VRAM SONO i byte del GGUF: nessuna
seconda copia dei codici, nessuna possibilita' che i codici addestrati divergano da
quelli esportati.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "@@ROOT:models@@/build/llamacpp-tq1-k1024/gguf-py")

LAYOUT = {
    "TQ1_0R_1024": {"byte_blocco": 224, "pesi_blocco": 1024, "scale_blocco": 8, "off_d": 208, "gruppo": 128,
                    "quarti": 4},
    "TQ1_0": {"byte_blocco": 54, "pesi_blocco": 256, "scale_blocco": 1, "off_d": 52, "gruppo": 256,
              "quarti": 1},
}


def _lay(tipo: str) -> dict:
    if tipo not in LAYOUT:
        raise ValueError(f"tipo {tipo!r} non ternario: attesi {sorted(LAYOUT)}")
    return LAYOUT[tipo]


def blocchi(raw: np.ndarray, tipo: str) -> np.ndarray:
    """Byte del tensore (qualsiasi forma) -> vista (N_blocchi, byte_blocco) uint8."""
    L = _lay(tipo)
    b = np.ascontiguousarray(raw).reshape(-1).view(np.uint8)
    if b.size % L["byte_blocco"]:
        raise ValueError(f"{b.size} byte non multiplo di {L['byte_blocco']} ({tipo})")
    return b.reshape(-1, L["byte_blocco"])


def maschera_codici(tipo: str) -> np.ndarray:
    """(byte_blocco,) bool: True sui byte dei CODICI (qs+qh), False sui byte di d."""
    L = _lay(tipo)
    m = np.ones(L["byte_blocco"], dtype=bool)
    m[L["off_d"]:L["off_d"] + 2 * L["scale_blocco"]] = False
    return m


def leggi_scale(raw: np.ndarray, tipo: str, E: int | None = None, n_out: int | None = None) -> np.ndarray:
    """Scale fp16 ESATTE (bit per bit) del tensore. Con E e n_out: forma (E, n_out, ng)."""
    L = _lay(tipo)
    B = blocchi(raw, tipo)
    d = B[:, L["off_d"]:L["off_d"] + 2 * L["scale_blocco"]].copy().view(np.float16).reshape(-1)
    if E is not None:
        d = d.reshape(E, n_out, -1)
    return d


def scrivi_scale(raw: np.ndarray, tipo: str, d_fp16: np.ndarray) -> np.ndarray:
    """Copia dei byte con i SOLI campi d sostituiti da `d_fp16` (qualsiasi forma con lo
    stesso numero di scale, gia' in float16: l'arrotondamento a fp16 lo decide chi
    addestra, perche' deve essere lo stesso visto nel forward). Codici intatti."""
    L = _lay(tipo)
    d = np.ascontiguousarray(d_fp16)
    if d.dtype != np.float16:
        raise TypeError(f"d deve essere float16 (arrotondato dall'addestramento), non {d.dtype}")
    if not np.isfinite(d.astype(np.float32)).all():
        raise FloatingPointError("scala non finita: il GGUF la scriverebbe, llama.cpp produrrebbe NaN")
    nuovo = blocchi(raw, tipo).copy()
    if d.size != nuovo.shape[0] * L["scale_blocco"]:
        raise ValueError(f"{d.size} scale per {nuovo.shape[0]} blocchi x {L['scale_blocco']}")
    nuovo[:, L["off_d"]:L["off_d"] + 2 * L["scale_blocco"]] = d.reshape(-1, L["scale_blocco"]).view(np.uint8) \
        .reshape(nuovo.shape[0], 2 * L["scale_blocco"])
    return nuovo.reshape(np.asarray(raw).shape) if np.asarray(raw).dtype == np.uint8 else nuovo


def verifica_patch(vecchio: np.ndarray, nuovo: np.ndarray, tipo: str, d_fp16: np.ndarray) -> dict:
    """Controlli d'export (non tautologici: rileggono i BYTE scritti).
    1) byte dei codici identici al bit; 2) campi d = d_fp16 al bit; 3) scale finite."""
    Bv, Bn = blocchi(vecchio, tipo), blocchi(nuovo, tipo)
    if Bv.shape != Bn.shape:
        raise RuntimeError(f"forma dei blocchi cambiata {Bv.shape} -> {Bn.shape}")
    m = maschera_codici(tipo)
    codici_uguali = bool(np.array_equal(Bv[:, m], Bn[:, m]))
    d_letti = leggi_scale(nuovo, tipo)
    d_uguali = bool(np.array_equal(d_letti.view(np.uint16), np.ascontiguousarray(d_fp16).reshape(-1).view(np.uint16)))
    finite = bool(np.isfinite(d_letti.astype(np.float32)).all())
    esito = {"codici_identici": codici_uguali, "d_identiche_al_bit": d_uguali, "d_finite": finite,
             "blocchi": int(Bv.shape[0]),
             "scale_cambiate": int((leggi_scale(vecchio, tipo).view(np.uint16) != d_letti.view(np.uint16)).sum())}
    if not (codici_uguali and d_uguali and finite):
        raise RuntimeError(f"export NON al bit: {esito}")
    return esito


# ───────────────────────────── trit in torch ─────────────────────────────
def _cifre_torch(b, n_cifre: int):
    """(…, nb) uint8 -> (…, n_cifre, nb) int8 in {-1,0,1}: cifra n = ((b*3^n)&255)*3>>8, -1.
    Stessa aritmetica di dequantize_row_tq1_0 (troncamento a 8 bit incluso)."""
    import torch
    b16 = b.to(torch.int16)                    # 255*81 = 20655 < 32767
    out = [((((b16 * (3 ** n)) & 255) * 3) >> 8) - 1 for n in range(n_cifre)]
    return torch.stack(out, dim=-2).to(torch.int8)


def _quarto_torch(qs, qh):
    """qs (…,48), qh (…,4) uint8 -> (…,256) int8, ordine naturale del blocco da 256:
    byte 0..31 -> elementi n*32+m; byte 32..47 -> 160+n*16+m; qh -> 240+n*4+m."""
    import torch
    lead = qs.shape[:-1]
    a = _cifre_torch(qs[..., 0:32], 5).reshape(*lead, 160)
    b = _cifre_torch(qs[..., 32:48], 5).reshape(*lead, 80)
    c = _cifre_torch(qh, 4).reshape(*lead, 16)
    return torch.cat([a, b, c], dim=-1)


def trit_torch(B, tipo: str):
    """B: tensore torch uint8 (N, byte_blocco) -> (N, pesi_blocco) int8 in {-1,0,1}.
    Per TQ1_0R_1024 i trit sono nel dominio RUOTATO (come le scale)."""
    import torch
    L = _lay(tipo)
    if L["quarti"] == 1:
        return _quarto_torch(B[:, 0:48], B[:, 48:52])
    parti = [_quarto_torch(B[:, q * 48:(q + 1) * 48], B[:, 192 + q * 4:192 + (q + 1) * 4]) for q in range(4)]
    return torch.cat(parti, dim=-1)


def scale_torch(B, tipo: str):
    """B: torch uint8 (N, byte_blocco) -> (N, scale_blocco) float16, bit per bit."""
    import torch
    L = _lay(tipo)
    s = B[:, L["off_d"]:L["off_d"] + 2 * L["scale_blocco"]].contiguous()
    return s.view(torch.float16).reshape(B.shape[0], L["scale_blocco"])

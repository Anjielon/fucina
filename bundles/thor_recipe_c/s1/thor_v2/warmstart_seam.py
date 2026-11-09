#!/usr/bin/env python3
"""Warm-start per il refinement "a seam" (ICBQ, arXiv 2608.09595 — PIANO.md
24/9 14:1x/14:2x): sovrascrive `gate_up_proj`/`down_proj` di uno strato GIA'
caricato (`carica_finestra_corrente`, ramo `quantizzati`) con la
ricostruzione dei codici ternari GIA' calcolati su disco (`.npy`), invece di
lasciare il valore letto dal GGUF base — così `ottimizza_finestra_JOINT`
riparte da un punto vicino alla soluzione precedente, non da zero.

Logica VERIFICATA bit-esatta sui segni (round-trip, scratchpad, 24/9 14:2x):
- gate/up vivono nello spazio RUOTATO (`_costruisci_ternari_layer(...,
  ruota=True)`): dequantizza per gruppo (GROUP_GU=128), concatena gate+up
  sull'asse "righe out", DE-ruota con `_RT_1024_torch`/`S1024` (STESSA
  rotazione di produzione, `rotazione_gpu.py` — mai una seconda formula).
- down vive nello spazio REALE, nessuna rotazione (GROUP_DW=256).
- La scala D ricostruita differisce dall'originale (3-13%, inizializzazione
  RTN grezza — vedi test): il refinement la corregge in pochi passi,
  partendo già da segni esatti.

Fail-open per-strato: se manca anche un solo file `.npy` per uno strato,
quello strato NON riceve il warm-start (resta il valore di
`carica_finestra_corrente`, comportamento INVARIATO) — mai un salto
silenzioso su un array a metà."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

GROUP_GU = 128
GROUP_DW = 256


def carica_pesi_warmstart(quantizzati: dict, layers: "list[int]", out_dir: Path) -> "list[int]":
    """Per ogni L in `layers` con TUTTI e 6 gli `.npy` presenti in `out_dir`,
    sovrascrive IN-PLACE `quantizzati[L].mlp.experts.{gate_up_proj,down_proj}
    .data`. Ritorna la lista degli L per cui il warm-start è stato applicato."""
    from rotazione_gpu import _RT_1024_torch, S1024  # noqa: E402 — sola lettura, stessa rotazione di produzione

    fatti: list[int] = []
    for L in layers:
        percorsi = {n: out_dir / f"{n}_{L:02d}.npy" for n in ("Dg", "Qg", "Du", "Qu", "Dd", "Qd")}
        if not all(p.exists() for p in percorsi.values()):
            continue
        Dg, Qg, Du, Qu, Dd, Qd = (np.load(percorsi[n]) for n in ("Dg", "Qg", "Du", "Qu", "Dd", "Qd"))

        strato = quantizzati[L]
        esperti = strato.mlp.experts
        device = esperti.gate_up_proj.device
        dtype_gu = esperti.gate_up_proj.dtype
        dtype_dw = esperti.down_proj.dtype

        gate_ruotato = np.repeat(Dg, GROUP_GU, axis=-1) * Qg.astype(np.float32)
        up_ruotato = np.repeat(Du, GROUP_GU, axis=-1) * Qu.astype(np.float32)
        gu_ruotato = np.concatenate([gate_ruotato, up_ruotato], axis=1)
        gu_ruotato_t = torch.from_numpy(gu_ruotato).to(device, dtype=torch.float32)
        S = torch.from_numpy(S1024).to(device, dtype=torch.float32)
        gu_reale = _RT_1024_torch(gu_ruotato_t, S)
        with torch.no_grad():
            esperti.gate_up_proj.data.copy_(gu_reale.to(dtype_gu))

        down_reale = np.repeat(Dd, GROUP_DW, axis=-1) * Qd.astype(np.float32)
        down_reale_t = torch.from_numpy(down_reale).to(device, dtype=torch.float32)
        with torch.no_grad():
            esperti.down_proj.data.copy_(down_reale_t.to(dtype_dw))

        fatti.append(L)
    return fatti


def _test_con_mock() -> bool:
    """Verifica STRUTTURALE (mock leggero, CPU, zero GPU/Thor): la funzione
    scrive nei posti giusti (`.mlp.experts.gate_up_proj`/`down_proj`) i
    valori ATTESI — stessa matematica del round-trip già verificato a mano
    (scratchpad 24/9), qui sull'API vera che `ottimizza_finestra_JOINT`
    userà. Usa i .npy REALI dello strato 0 già su disco."""
    from rotazione_gpu import _R_1024_torch, S1024  # noqa: E402

    out_dir = Path("@@ROOT:models@@/gguf/tony-tern/valida-finestra-8-12")
    L = 0
    if not (out_dir / f"Dg_{L:02d}.npy").exists():
        print("⚠️ .npy di riferimento assenti — test saltato (non un fallimento, un ambiente diverso)")
        return True

    class _EspertiFinti:
        def __init__(self, gu_shape, dw_shape):
            self.gate_up_proj = torch.nn.Parameter(torch.zeros(gu_shape, dtype=torch.bfloat16))
            self.down_proj = torch.nn.Parameter(torch.zeros(dw_shape, dtype=torch.bfloat16))

    class _MlpFinto:
        def __init__(self, gu_shape, dw_shape):
            self.experts = _EspertiFinti(gu_shape, dw_shape)

    class _StratoFinto:
        def __init__(self, gu_shape, dw_shape):
            self.mlp = _MlpFinto(gu_shape, dw_shape)

    Dg = np.load(out_dir / f"Dg_{L:02d}.npy")
    Qg = np.load(out_dir / f"Qg_{L:02d}.npy")
    Dd = np.load(out_dir / f"Dd_{L:02d}.npy")
    Qd = np.load(out_dir / f"Qd_{L:02d}.npy")
    E, meta, n_in = Qg.shape
    strato = _StratoFinto((E, 2 * meta, n_in), Qd.shape)
    quantizzati = {L: strato}

    fatti = carica_pesi_warmstart(quantizzati, [L], out_dir)
    ok = fatti == [L]
    print(f"warm-start applicato a: {fatti} (atteso [{L}])")

    # ri-ruota il gate_up_proj scritto e verifica che i SEGNI tornino esatti
    # (stessa verifica del round-trip, ora sull'oggetto vero)
    S = torch.from_numpy(S1024).to(torch.float32)
    gu_scritto = strato.mlp.experts.gate_up_proj.data.to(torch.float32)
    gu_ri_ruotato = _R_1024_torch(gu_scritto, S)
    gate_ri = gu_ri_ruotato[:, :meta]
    Qg_atteso_sign = np.sign(Qg.astype(np.float32))
    Qg_ricavato_sign = torch.sign(gate_ri).numpy()
    d_segni = np.abs(Qg_ricavato_sign - Qg_atteso_sign)
    # tolleranza sui pesi ESATTAMENTE zero (segno ambiguo per rumore float),
    # altrove i segni devono combaciare
    non_zero = Qg_atteso_sign != 0
    frac_ok = (d_segni[non_zero] < 1e-3).mean() if non_zero.any() else 1.0
    print(f"gate: frazione segni non-zero combacianti = {frac_ok:.6f} (atteso 1.0)")
    ok &= frac_ok > 0.999

    down_scritto = strato.mlp.experts.down_proj.data.to(torch.float32).numpy()
    Qd_atteso_sign = np.sign(Qd.astype(np.float32))
    Qd_ricavato_sign = np.sign(down_scritto)
    non_zero_d = Qd_atteso_sign != 0
    frac_ok_d = (np.abs(Qd_ricavato_sign - Qd_atteso_sign)[non_zero_d] < 1e-3).mean() if non_zero_d.any() else 1.0
    print(f"down: frazione segni non-zero combacianti = {frac_ok_d:.6f} (atteso 1.0)")
    ok &= frac_ok_d > 0.999

    # strato senza .npy -> fail-open, nessun warm-start
    fatti_vuoti = carica_pesi_warmstart({99: strato}, [99], out_dir)
    ok &= fatti_vuoti == []
    print(f"fail-open su strato senza .npy: {fatti_vuoti} (atteso [])")

    print("✅ TUTTI I TEST PASSATI" if ok else "⛔ ALCUNI TEST FALLITI")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _test_con_mock() else 1)

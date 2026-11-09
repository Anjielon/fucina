#!/usr/bin/env python3
"""Loop ESTERNO VERO sui 40 strati reali di Thor — il pezzo che mancava
(PIANO.md 24/9 08:40): mette insieme, senza duplicarne la logica, i pezzi
già costruiti e verificati singolarmente stanotte:

- pianificazione finestre (stride/overlap): `ottimizzazione_globale_loop.
  pianifica_finestre` (pura, mai toccata, riusata cosi' com'e').
- caricamento/scaricamento per-finestra a memoria limitata:
  `valida_finestra_8_12.carica_finestra_corrente(layers=...)` /
  `libera_finestra` (parametrizzate e verificate su GPU vera oggi).
- nucleo di ottimizzazione veloce: `valida_finestra_8_12.
  ottimizza_finestra_JOINT` (2,5 s/passo misurato, loss converge).
- passaggio del vero punto di partenza fra finestre: `valida_finestra_8_12.
  calcola_confine_finestra` + `h0_esterno_per_b`/`e0_esterno_per_b` di
  `ottimizza_finestra_JOINT` — verificato bit-esatto in
  `test_seed_esterno_equivalenza.py`.
- scrittura codici + esportazione GGUF finale:
  `scrivi_codici_su_disco` (per-finestra) + `esporta_gguf_finestra`
  (UNA sola volta a fine corsa, su tutti gli strati — vedi il suo
  docstring per il perche').

⛔ SOLO no-overlap per ora (stride==W): il caso overlap richiederebbe un
confine a un offset INTERNO alla finestra (non implementato in
`calcola_confine_finestra`, dichiarato li' esplicitamente, non finto).
La finestra 0 parte SEMPRE dal vero strato 0 della rete — li' l'embedding
grezzo E' il bersaglio corretto (nessun bug, nessun seed esterno necessario:
`h0_esterno_per_b=None` per costruzione al primo giro).

Uso:
    THOR_FORGIA_N_STRATI=40 THOR_FORGIA_W=2 THOR_VALIDA_N_PASSI=200 \
        python3 forgia_globale_40_strati.py
Per un fumo veloce (verifica end-to-end, non produzione):
    THOR_FORGIA_N_STRATI=4 THOR_FORGIA_W=2 THOR_VALIDA_N_PASSI=6 \
        python3 forgia_globale_40_strati.py --salta-export
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ottimizzazione_globale_loop import pianifica_finestre  # noqa: E402 — pura, riusata
from valida_finestra_8_12 import (  # noqa: E402 — riusate, mai duplicate
    NUCLEO_LR,
    calcola_confine_finestra,
    carica_finestra_corrente,
    carica_batch_calibrazione,
    seme_embedding_quantizzato,
    esporta_gguf_finestra,
    libera_finestra,
    log,
    ottimizza_finestra_JOINT,
    scrivi_codici_su_disco,
)

N_STRATI_TOTALE = int(os.getenv("THOR_FORGIA_N_STRATI", "40"))
W = int(os.getenv("THOR_FORGIA_W", "2"))
OVERLAP = os.getenv("THOR_FORGIA_OVERLAP", "0") == "1"
N_PASSI = int(os.getenv("THOR_VALIDA_N_PASSI", "200"))
ALPHA_PROPAGAZIONE = float(os.getenv("THOR_FORGIA_ALPHA", "1.0"))
# 24/9 21:0x: il confine riusa l'uscita del teacher gia' calcolata dall'ottimizzazione
CONFINE_RIUSA_TEACHER = os.getenv("THOR_CONFINE_RIUSA_TEACHER", "1") == "1"


def esegui(n_strati_totale: int, w: int, overlap: bool, n_passi: int, salta_export: bool,
           alpha_propagazione: float = 1.0) -> dict:
    """⭐ 24/9 13:0x — `alpha_propagazione` (default 1.0 = comportamento
    INVARIATO): smorza `e0_seme` (l'errore accumulato) prima di passarlo
    alla finestra successiva, moltiplicandolo per questo fattore.

    Perché: trovato leggendo QEP (Fujitsu/RIKEN, NeurIPS 2025, arXiv
    2504.09629, PIANO.md 13:04) — la correzione PIENA dell'errore
    propagato (α=1, quello che facciamo sempre) può causare OVERFITTING
    "particolarmente pronunciato quando il dataset di calibrazione è
    piccolo... o quando il modello include blocchi con un grande numero
    di parametri come i blocchi MLP" — ESATTAMENTE la nostra situazione
    (4096 campioni, 256 esperti/strato). Ipotesi da VERIFICARE, non
    assunta: l'overfitting si amplifica su 20 finestre consecutive fino al
    collasso osservato (v1, 40 strati: ok=0/15 muto=4 guasto=11)."""
    if overlap:
        raise NotImplementedError(
            "overlap=True non ancora supportato qui: calcola_confine_finestra sa solo il "
            "confine di FINE finestra (no-overlap), non un offset interno — dichiarato, non finto.")

    finestre = pianifica_finestre(n_strati_totale, w, overlap)
    log(f"═══ FORGIA GLOBALE {n_strati_totale} strati, W={w}, overlap={overlap} → "
        f"{len(finestre)} finestre, {n_passi} passi/finestra ═══")

    # audit B6: esempi misti math/code in formato chat + <think>, batch held-out disgiunte
    ids_batch, n_addestramento = carica_batch_calibrazione()
    h0_seme, e0_seme = None, None
    t0_totale = time.time()
    strati_visti: list[int] = []

    from concurrent.futures import ThreadPoolExecutor
    _scrittore = ThreadPoolExecutor(max_workers=1)     # scrittura dei codici in parallelo
    scritture: list = []
    for idx, (l0, l1_esclusa) in enumerate(finestre):
        layers = list(range(l0, l1_esclusa))
        t0 = time.time()
        log(f"── finestra {idx+1}/{len(finestre)}: strati {layers} ──")

        cfg, emb, puliti, quantizzati = carica_finestra_corrente(layers=layers)
        if idx == 0 and os.getenv("THOR_E0_EMBEDDING_QUANT", "1") == "1":
            # audit B12: la finestra 0 parte dall'embedding QUANTIZZATO della base
            # (come in produzione), non con errore iniziale zero
            h0_seme, e0_seme = seme_embedding_quantizzato(ids_batch, emb)

        teacher_fine: dict = {}
        risultato = ottimizza_finestra_JOINT(
            quantizzati, ids_batch, cfg, n_passi=n_passi, strati_teacher=puliti, emb=emb,
            lr=NUCLEO_LR, h0_esterno_per_b=h0_seme, e0_esterno_per_b=e0_seme,
            n_batch_addestramento=n_addestramento,
            uscita_teacher_fine=teacher_fine if CONFINE_RIUSA_TEACHER else None)

        # ⭐ 24/9 21:0x: i codici (numpy, gia' su CPU) si scrivono in un thread mentre
        # parte il confine e la finestra dopo (16 s/finestra); si aspetta prima dell'export.
        scritture.append(_scrittore.submit(scrivi_codici_su_disco, risultato))
        strati_visti.extend(layers)

        if idx + 1 < len(finestre):
            # Seme per la finestra SUCCESSIVA: vero output COMMESSO di questa
            # (puliti = teacher, invariato; quantizzati = appena aggiornato
            # in-place con i codici duri dentro ottimizza_finestra_JOINT).
            # ⛔ 24/9 15:1x — il seme IN INGRESSO a questa finestra va passato:
            # senza, il confine ripartiva dall'embedding e attraversava solo
            # questa finestra (bug che rendeva sbagliato l'ingresso di ogni
            # finestra dalla terza in poi — vedi calcola_confine_finestra).
            h_seme_prima = h0_seme
            h0_seme, e0_seme = calcola_confine_finestra(
                puliti, quantizzati, ids_batch, cfg, emb, h0_ingresso_per_b=h0_seme,
                e0_ingresso_per_b=e0_seme, h_fine_teacher_per_b=teacher_fine or None)
            if teacher_fine and idx == 0:
                # controllo una tantum: il teacher riusato = il teacher ricalcolato (batch 0)
                h_t_rif, _ = calcola_confine_finestra(
                    puliti, quantizzati, ids_batch[:1], cfg, emb,
                    h0_ingresso_per_b=({0: h_seme_prima[0]} if h_seme_prima is not None else None),
                    e0_ingresso_per_b=None)
                d = h_t_rif[0].float() - h0_seme[0].float()
                rel = (d.norm() / h_t_rif[0].float().norm()).item()
                log(f"  controllo confine: teacher riusato vs ricalcolato (batch 0): scarto massimo "
                    f"{d.abs().max().item():.3e}, relativo {rel:.3e}")
                if rel > 1e-2:
                    raise RuntimeError(f"confine: teacher riusato diverso dal ricalcolato (rel {rel:.3e})")
            if alpha_propagazione != 1.0:
                e0_seme = {b: e0_seme[b] * alpha_propagazione for b in e0_seme}
                log(f"  e0 smorzato con alpha={alpha_propagazione} prima della finestra successiva")

        libera_finestra(puliti, quantizzati, emb)
        log(f"── finestra {idx+1}/{len(finestre)} FATTA in {time.time()-t0:.1f}s ──")

    log(f"═══ tutte le {len(finestre)} finestre fatte in {time.time()-t0_totale:.1f}s "
        f"({strati_visti[0]}-{strati_visti[-1]}, {len(strati_visti)} strati) ═══")

    for f in scritture:
        f.result()          # solleva qui se una scrittura dei codici e' fallita
    _scrittore.shutdown(wait=True)
    if salta_export:
        log("--salta-export: fermo prima dell'esportazione GGUF")
        return {"finestre": finestre, "strati": strati_visti}

    path = esporta_gguf_finestra(layers=strati_visti)
    log(f"GGUF finale: {path}")
    return {"finestre": finestre, "strati": strati_visti, "gguf": str(path)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--salta-export", action="store_true")
    args = ap.parse_args()
    esegui(N_STRATI_TOTALE, W, OVERLAP, N_PASSI, args.salta_export,
           alpha_propagazione=ALPHA_PROPAGAZIONE)
    return 0


if __name__ == "__main__":
    sys.exit(main())

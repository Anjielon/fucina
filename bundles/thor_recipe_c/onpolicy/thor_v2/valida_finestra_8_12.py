#!/usr/bin/env python3
"""VALIDAZIONE della finestra scorrevole strati 8-12 — ottimizzazione GLOBALE
congiunta cross-strato (SchurQuant/Thor, arXiv 2607.14630 + arXiv 2606.13054).

24/9/2026, ordine del workflow dopo il depth-sweep delle 02:0x (`stato/PIANO.md`).

## Perche' QUESTA finestra

Il depth-sweep di stanotte (`stato/depth_sweep.log`, script gemello ancora nello
scratchpad di sessione: mai stato nel repo, per questo il numero e' riportato
qui LETTERALE) ha sondato profondita' CRESCENTI (0..N-1, cumulative da strato
0 — e' un vincolo del metodo: X_q allo strato L dipende da TUTTI gli strati
precedenti gia' quantizzati, il prefisso non si puo' isolare a meta') con la
sonda-15 (sotto, §SONDA):

    profondita' 0-6   (N= 7): ok= 9/15  muto=0
    profondita' 0-7   (N= 8): ok=10/15  muto=0
    profondita' 0-8   (N= 9): ok= 9/15  muto=0
    profondita' 0-9   (N=10): ok= 9/15  muto=0
    profondita' 0-10  (N=11): ok= 6/15  muto=2   <-- LA TRANSIZIONE

Pulito fino a 10 strati (0-9), rotto a 11 (0-10): la zona 8-12 e' dove il
depth-sweep smette di essere "rumore" e diventa "rotto" (crollo netto fra
N=10 e N=11, coerente col MUTO osservato altrove per composizioni oltre i
~10-12 strati — vedi anche il commento in `cattura_schur_esperti.py` sul
"MUTO su 0-12"). E' la finestra dove l'errore accumulato (Lemma 1,
e_{l+1}=A_l(e_l)+q_l) supera la soglia che fa collassare il routing MoE.

⛔ **Onesta' sul confronto**: la baseline sopra (6/15, muto=2) e' CUMULATIVA
da strato 0 col metodo ATTUALE (sequenziale, un layer alla volta, obiettivo
teacher-reconstruction locale — `schurquant_esperto_reale.py`/
`forgia_schur_esperti.py`). Questo script isola la finestra 8-12: lascia
0-7 e 13-39 esattamente come sono OGGI sulla base (`THOR_SCHUR_BASE`) e
sostituisce SOLO 8-12 con l'esito del metodo nuovo. Non e' lo stesso identico
esperimento (finestra isolata contro cumulativo), ma e' la domanda scientifica
che conta: "a parita' di sfondo, il metodo nuovo tiene la finestra 8-12
meglio del vecchio?" — e resta l'unico confronto disponibile finche' non si
rigenera un depth-sweep con la finestra nuova al posto della vecchia negli
stessi punti (funzione `genera_baseline_vecchio_metodo_stessa_finestra`,
sotto, PREDISPOSTA ma NON eseguita di default: costerebbe GPU extra non
richiesta da questo compito).

## Cosa fa OGGI questo script (eseguibile SUBITO, senza aspettare nessuno)

1. Guardia GPU condivisa VERA (rocm-smi: VRAM+%uso, non un nome di processo —
   lezione [[feedback_guardiano_gpu_libera_per_nome_di_processo]]) + unita'
   systemd --user thor-*/adhoc-* gia' attive.
2. Carica un PUGNO di batch REALI dal corpus di calibrazione (`THOR_CORPUS`,
   lo stesso di `cattura_schur_esperti.py`) — poche batch, non l'intero
   corpus: `THOR_VALIDA_N_BATCH` x `THOR_VALIDA_BATCH_SEQ` sequenze da
   `THOR_VALIDA_L_SEQ` token (default 2x2x1024 = 4096 campioni).
3. Carica i 5 strati 8-12 coi pesi CORRENTI da disco — ramo quantizzato
   (`da_gguf=True`, attenzione/SSM/esperti dequantizzati dalla base GGUF di
   produzione: STESSA identica configurazione che ha dato -68,9% verificato
   in `schurquant_esperto_reale.py`) e ramo pulito (`da_gguf=False`, bf16 HF
   vero) — riusando `cattura_schur_esperti.costruisci_strato`, importato in
   **sola lettura**: questo file non ne cambia una riga.
4. Misura, OGGI, una diagnostica VERA (non un placeholder): la loss di
   corrispondenza statistica Eq.6 (`loss_corrispondenza_statistica.py`, GIA'
   verificata) fra i due rami, strato per strato, sul corpus caricato — un
   numero concreto di "quanto e' rotta" la finestra PRIMA di ottimizzare.
5. Chiama il nucleo di ottimizzazione GLOBALE — ⭐ **IMPLEMENTATO 24/9**
   (§NUCLEO sotto, `ottimizza_finestra_JOINT`): la matematica (fusione
   soft-to-hard ternaria di `quant_ternaria_globale.py` + kernel Lemma-1 di
   `finestra_globale.py`, entrambi SOLA LETTURA, entrambi gia' verificati a
   scala giocattolo) e' collegata a strati REALI (`Qwen3_5MoeDecoderLayer`)
   con TernarioDiff per gate_up_proj (ruotato) E down_proj (senza rotazione),
   routing congelato per chunk, commit progressivo dei codici duri fra chunk.
   Collaudo end-to-end a dimensioni GIOCATTOLO (mai Thor): vedi
   `_test_ottimizza_finestra_JOINT_giocattolo` in fondo al file
   (`--test-giocattolo`).
6. Scrive i codici (D,Q) su disco nel contratto di
   `gptq_1024_lib.pack_da_d_q` (lo stesso di `forgia_schur_esperti.py`),
   esporta un GGUF di prova patchando SOLO `blk.{8..12}.ffn_{gate,up}_exps.
   weight` (mai cumulativo da 0 — vedi sopra) — ⚠️ SOLO gate/up: l'estensione
   di questi due esportatori a `ffn_down_exps` resta fuori perimetro di
   `ottimizza_finestra_JOINT` (che gia' lo calcola e lo ritorna, vedi
   §NUCLEO) — lancia la STESSA sonda-15 di stanotte (reimplementata qui in
   Python: lo script che l'ha generata vive in uno scratchpad di sessione,
   fuori dal repo, e non e' un riferimento stabile) e verdetto
   (§CRITERIO DI SUCCESSO).

⛔ VINCOLO DI SESSIONE (GPU condivisa, dato dal workflow): nessun caricamento
del modello Thor INTERO ne' training reale su di esso — questo file lavora
SOLO sulla finestra a 5 strati dichiarata, e libera SEMPRE la VRAM
(`del()`+`torch.cuda.empty_cache()`) dopo ogni fase pesante.

## §NUCLEO — il contratto che l'altro filone deve rispettare

`ottimizza_finestra_JOINT(strati_correnti, ids_batch, cfg, n_passi)` deve
ritornare `{L: {"ffn_gate_exps": (D, Q), "ffn_up_exps": (D, Q)}}` per OGNI L
della finestra, con D: (256, out=512, ng=16) float32 (scala ogni 128 pesi,
GROUP=128 di `gptq_1024_lib`) e Q: (256, 512, 1024) int8 in {-1,0,1} — ESATTO
contratto di `gptq_1024_lib.pack_da_d_q` per riga, gia' quello che
`forgia_schur_esperti.py` scrive oggi (Dg/Qg gate, Du/Qu up). Qualunque sia
la matematica dentro (soft-to-hard ternario fuso, Lemma 1, AdamW su z/s...),
l'esportatore di questo file non ha bisogno di saperlo: consuma solo quel
dizionario. Oggi la funzione solleva `NotImplementedError` con questo stesso
contratto scritto nel messaggio.

⭐ **AGGIORNATO 24/9 (compito del workflow, dopo PIANO.md 02:56 e
`stato/PIANO_OTTIMIZZAZIONE_GLOBALE.md` §5)**: il dizionario per OGNI L deve
portare ANCHE `"ffn_down_exps": (D, Q)` — down_proj entra nella STESSA
ottimizzazione globale di gate/up, non in una separata (altrimenti si
raddoppierebbe, invece di chiudere, il gap diagnosticato in PIANO.md per
gate/up). Formato **diverso** da gate/up, per costruzione (vedi
`cattura_schur_down.py`, che gia' calcola G_down/C_down con la stessa Eq.8):
TQ1_0 STANDARD, **SENZA rotazione**, blocco **GROUP=256** — D: (256,
out=hid=2048, ng=inter//256=2) float32, Q: (256, 2048, inter=512) int8 in
{-1,0,1} — contratto di `pack_tq1_0_da_d_q`
(`pack_tq1_0_standard.py`), NON di `gptq_1024_lib.pack_da_d_q` (quello e' il
codec ruotato di gate/up, blocco 1024: i due packer non sono
intercambiabili). Il **meccanismo** che rende down_proj una variabile
allenabile nella STESSA loss/AdamW di gate/up e' gia' dimostrato a scala
giocattolo in `thor/finestra_globale.py` (test 4,
`_test_down_proj_variabile_libera`): un secondo `TernarioDiff` per
down_proj, gruppo GROUP=256 (contro 128 di gate/up), parametri concatenati
nella STESSA lista `parametri_ternari` che entra in un SOLO `torch.optim.
AdamW` — nessun lavoro di design nuovo per collegarlo qui, solo l'estensione
a pesi VERI (fuori perimetro di questo file, che resta un contratto). ⚠️
Chi implementa il nucleo VERO deve inoltre estendere `scrivi_codici_su_disco`
ed `esporta_gguf_finestra` (sotto) per scrivere/patchare ANCHE
`blk.{L}.ffn_down_exps.weight` — non fatto qui: questo aggiornamento tocca
SOLO il contratto d'uscita dichiarato, non l'implementazione dei due
esportatori, che oggi gestiscono solo gate/up.

## §CRITERIO DI SUCCESSO (preciso, misurato con lo stesso metro di stanotte)

BASELINE (misurata stanotte, depth-sweep 0-10, N=11, metodo sequenziale attuale):
    ok=6/15  muto=2
OBIETTIVO (livello "pulito" pre-transizione, N<=10 nello stesso depth-sweep):
    ok>=9/15  muto=0
Verdetto a 4 esiti (funzione `giudica`, sotto — nessuna zona grigia):
    SUCCESSO PIENO       muto<=0 e ok>=9   (la finestra torna al livello pulito)
    MIGLIORAMENTO        muto<2, oppure (muto==2 e ok>6)   (meglio della baseline, non ancora pulito)
    NESSUN MIGLIORAMENTO muto==2 e ok<=6   (= baseline, nessun progresso misurato)
    REGRESSIONE          muto>2, oppure (muto==2 e ok<6)   (peggio della baseline)

## File letti in SOLA LETTURA (mai modificati da questo script)
`cattura_schur_esperti.py` (costruisci_strato) · `esporta_schur_esperti.py`
(impacchetta_famiglia, verifica_packer, TIPO_BASE_ATTESO/TIPO_NUOVO) ·
`loss_corrispondenza_statistica.py` · `gptq_1024_lib.py` ·
`finestra_globale.py` (errore_lemma1_finestra, passo_ottimizzazione,
GRUPPO_GATEUP_REALE/GRUPPO_DOWN_REALE/_group_giocattolo) ·
`quant_ternaria_globale.py` (TernarioDiff, schedula_beta) ·
`rotazione_gpu.py` (_R_1024_torch, _RT_1024_torch, S1024).
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getenv("GGUF_PY_DIR", "@@ROOT:models@@/build/llamacpp-tq1-k1024/gguf-py"))

from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as T  # noqa: E402

from gptq_1024_lib import GROUP, QK, BLOCK_BYTES  # noqa: E402,F401  (contratto, non uso diretto qui)
from loss_corrispondenza_statistica import (  # noqa: E402
    crea_proiezione_casuale, perdita_statistiche, K_PROIEZIONE_DEFAULT,
)
# ⛔ §NUCLEO — 24/9, il pezzo che questo file aspettava (vedi PIANO.md 03:11):
# il kernel Lemma-1 a finestra scorrevole (gia' verificato 1/2 strati e
# down_proj-come-variabile-libera GIOCATTOLO) + la parametrizzazione ternaria
# vera (gia' verificata su un "esperto" giocattolo). SOLA LETTURA — nessuna
# riga di questi due moduli tocca ne' finestra_globale.py ne'
# quant_ternaria_globale.py. GRUPPO_GATEUP_REALE/GRUPPO_DOWN_REALE/
# _group_giocattolo sono RIUSATI da li' (mai ridichiarati qui): sono le
# STESSE costanti/lo stesso ripiego usati dal test 4 di finestra_globale.py.
from finestra_globale import (  # noqa: E402
    errore_lemma1_finestra, passo_ottimizzazione, precalcola_teacher,
    GRUPPO_GATEUP_REALE, GRUPPO_DOWN_REALE, _group_giocattolo,
)
from quant_ternaria_globale import schedula_beta  # noqa: E402
# ⭐ 24/9 (integrazione velocita'): `TernarioDiff` per-esperto + ModuleList di
# 256 istanze e' SOSTITUITO da `TernarioDiffBatchFuso` (asse E in testa, UNA
# sola istanza, nucleo compilato) — SOLA LETTURA anche questo modulo, gia'
# verificato BIT-ESATTO (max|diff|<=7.45e-9) contro il loop di riferimento e
# misurato 5.48x piu' veloce end-to-end in `quant_ternaria_globale_batch_v2.py`
# (vedi quel file per diagnosi/misura complete). Non una riga di matematica
# cambia qui: solo il MECCANISMO di costruzione/ricostruzione (§NUCLEO sotto).
import rotazione_gpu as _rotazione_gpu_mod  # noqa: E402  (per il monkeypatch compilato, sotto)
from quant_ternaria_globale_batch_v2 import TernarioDiffBatchFuso  # noqa: E402
# rotazione Hadamard a blocchi di QK=1024 (formato GGUF di destinazione
# TQ1_0R_1024, SOLO gate/up — down_proj e' TQ1_0 STANDARD senza rotazione,
# vedi §NUCLEO) — riuso delle funzioni TORCH (differenziabili, verificato
# round-trip 4.8e-7 su un vettore 2x1024 con grad, 24/9) di rotazione_gpu.py,
# MAI le gemelle numpy (quelle spezzerebbero l'autograd a meta' pipeline).
from rotazione_gpu import _R_1024_torch, _RT_1024_torch, S1024 as _S1024_NP  # noqa: E402
from init_ls import reinit_ls_  # noqa: E402  (audit B3: init LS dai pesi originali)
# 25/9 (PROGETTO_CACHE_LOTTI.md): stato per lotto su disco oltre il budget GPU, interruttore THOR_CACHE_LOTTI
# SPENTO di default (spento: DL.deposito() da' un dict vero e ogni aggancio e' un no-op). Alias DL e non D:
# qui D e' il nome locale delle scale (D, Q = ...) e lo oscurerebbe.
import deposito_lotti as DL  # noqa: E402
# 26/9: ancora ai pesi di riferimento (THOR_ANCORA_RAPPORTO, spento di default = nessun oggetto costruito,
# nessuna riga di calcolo in piu'). Perche' e come: docstring di ancora_forgia.py.
import ancora_forgia  # noqa: E402
# 26/9 sera: Selective Projection Decay dopo il passo (THOR_SPD_LAMBDA, spenta di default = nessun oggetto,
# nessuna fotografia, nessun calcolo in piu'). La cura disaccoppiata dopo il fallimento dell'ancora nella
# perdita. Perche' e come: docstring di spd_forgia.py (Tian et al., NeurIPS 2024, arXiv 2411.01713).
import spd_forgia  # noqa: E402

# ⭐ 24/9 (integrazione velocita', 2/2): `_R_1024_torch`/`_RT_1024_torch` (sopra)
# chiamano `fwht1024_torch` per NOME dentro il modulo `rotazione_gpu` — un
# lookup dinamico nel namespace del modulo ad OGNI chiamata, non un binding
# fissato all'import. Sostituendo l'attributo sul MODULO (non la funzione
# importata qui) con la sua versione compilata, `_R_1024_torch`/`_RT_1024_torch`
# la usano SENZA che una riga di `rotazione_gpu.py` o di questo file (nel loro
# corpo) cambi — "nessun cambio di codice, solo richiamata attraverso una
# versione compilata invece che eager", esattamente come misurato (6.85x,
# bit-esatto) da chi ha verificato il guadagno. Nessun wrapper/cache extra
# serve: `torch.compile` gia' tiene un compilo per FORMA di ingresso, e le due
# forme che qui ricorrono (gate_up ruotato: fissa per tutti gli strati della
# finestra; down_proj: MAI passato di qui, ruota=False sempre) restano stabili
# per l'intera corsa — un solo compilo, riusato su ogni strato/passo/chunk.
# Compilazione LAZY (al primo uso reale, non qui): a scala giocattolo
# (n_in << QK) `_ruota_se_possibile`/`_deruota_se_possibile` ritornano
# l'ingresso invariato e non chiamano MAI `fwht1024_torch` — il monkeypatch
# resta innocuo per `--test-giocattolo`.
_rotazione_gpu_mod.fwht1024_torch = torch.compile(_rotazione_gpu_mod.fwht1024_torch)

# ⭐ 24/9 20:0x (radiografia del passo): su ROCm/gfx1151 F.grouped_mm degli esperti e'
# un ciclo di 4168 `mm` piccoli per passo (~21% del tempo GPU). THOR_GROUPED_BMM=1 lo
# sostituisce con UN bmm batched (grouped_mm_bmm.py, verificato da test_grouped_mm_bmm.py).
if os.getenv("THOR_GROUPED_BMM", "0") == "1":
    import grouped_mm_bmm as _grouped_mm_bmm  # noqa: E402
    _grouped_mm_bmm.installa()

# ────────────────────────────── configurazione ──────────────────────────────
# ⛔ percorsi SOLO da ambiente (questo file viaggia fra casa e pod, come il
# resto della pipeline): STESSI default di cattura_schur_esperti.py.
SRC = Path(os.getenv("THOR_HF_DIR", "@@ROOT:models@@/gguf/thor-src-hf"))
GGUF_BASE = Path(os.getenv("THOR_SCHUR_BASE", "@@ROOT:models@@/gguf/tony-tern/Thor-v20-downaggr3.gguf"))
CORPUS = Path(os.getenv("THOR_CORPUS", "@@ROOT:forge@@/thor/cot_calib/corpus_cot.txt"))

# la finestra oggetto di questa validazione — 5 strati, estremi INCLUSI
FINESTRA_L0 = int(os.getenv("THOR_VALIDA_L0", "8"))
FINESTRA_L1 = int(os.getenv("THOR_VALIDA_L1", "12"))
FINESTRA = list(range(FINESTRA_L0, FINESTRA_L1 + 1))

# corpus di validazione: POCHE batch reali (non l'intero corpus di calibrazione)
DOWN_TERNARIO = os.getenv("THOR_FORGIA_DOWN_TERNARIO", "1") == "1"
HELDOUT_OGNI = int(os.getenv("THOR_HELDOUT_OGNI", "0"))   # 0 = solo inizio/fine   # audit B5: interruttore A/B
RICONGELA_OGNI = int(os.getenv("THOR_RICONGELA_OGNI", "0"))   # 0 = solo all'ingresso della fase dura
PROFILO = os.getenv("THOR_PROFILO", "0") == "1"   # radiografia dei passi 5-9 (torch.profiler)
DIAGNOSTICA_PASSO = os.getenv("THOR_DIAGNOSTICA_PASSO", "1") == "1"   # 0 = niente F̂(h) diagnostico nei passi
# 24/9 20:1x (radiografia: FWHT sui pesi ~15% del passo): 1 = si ruota l'INGRESSO degli
# esperti invece di de-ruotare i pesi. W·x = W_rot·(R·x): stessa funzione, 2048x2048
# numeri per passo invece di 2x537 M. Non identica al bit (cambia dove cade
# l'arrotondamento bf16): va verificata sulla loss prima di adottarla.
ROTAZIONE_INGRESSO = os.getenv("THOR_ROTAZIONE_INGRESSO", "0") == "1"
# 24/9 20:2x (radiografia a 8 seq: copie 22%): 1 = il nucleo ternario scrive bf16
# direttamente (cast fuso). Vale solo dove il peso va nel modulo senza FWHT in fp32
# dopo: down sempre, gate_up solo con ROTAZIONE_INGRESSO. Identico al bit.
NUCLEO_BF16 = os.getenv("THOR_NUCLEO_BF16", "0") == "1"
# 24/9 20:3x: 1 = ROUTING VIVO. Il router sceglie gli esperti DENTRO il passo, sui codici
# correnti, come fara' il modello in produzione: e' il limite del ricongelo (ogni passo),
# a costo zero (niente congelamento iniziale ne' ricongeli su tutte le batch) e allinea
# l'addestramento alla valutazione finale (che usa gia' il router vero). Da validare sulla qualita'.
ROUTING_VIVO = os.getenv("THOR_ROUTING_VIVO", "0") == "1"
# 24/9 20:5x: Q3 ha mostrato che il routing VIVO converge peggio (-12% al passo 75).
# 1 = ROUTING AGGIORNATO GRATIS: resta congelato per batch (come Q1/Q2), ma durante il
# passo si legge l'ingresso VERO del router (post_attention_layernorm dentro F̂(h+e)) e si
# ricalcola la scelta per QUELLA batch, usata al suo prossimo turno (= ricongelo ogni
# n_batch passi) senza nessun forward in piu'.
ROUTING_AGGIORNATO = os.getenv("THOR_ROUTING_AGGIORNATO", "0") == "1"
# 24/9 23:0x: 1 = un passo con loss o gradienti non finiti si SALTA (niente aggiornamento) invece di
# corrompere i parametri; oltre il 5% di passi saltati la forgia si ferma. (NaN sporadico visto in M1.)
SALTA_NON_FINITI = os.getenv("THOR_SALTA_NON_FINITI", "0") == "1"
CRONO_OGNI = int(os.getenv("THOR_CRONO_OGNI", "0"))   # 24/9 19:5x: secondi/passo ogni N passi (0 = spento)
ID_CORSA = os.getenv("THOR_ID_CORSA", time.strftime("%Y%m%d-%H%M%S"))   # audit B10: provenienza
N_BATCH = int(os.getenv("THOR_VALIDA_N_BATCH", "32"))   # audit B6: 2 -> 32 (a rotazione)
BATCH_SEQ = int(os.getenv("THOR_VALIDA_BATCH_SEQ", "2"))
L_SEQ = int(os.getenv("THOR_VALIDA_L_SEQ", "1024"))

# ⭐ 24/9: default alzato 50 → 200 (ordine ricevuto insieme al nucleo,
# `ottimizza_finestra_JOINT` sotto: "un numero di passi configurabile,
# default 200, non 1000 — dichiaralo come scelta di partenza ECONOMICA da
# aumentare se serve"). `main()` passa questo stesso valore come `n_passi`.
N_PASSI = int(os.getenv("THOR_VALIDA_N_PASSI", "200"))

# ── §NUCLEO: iperparametri dell'ottimizzazione globale (ottimizza_finestra_JOINT) ──
# beta_max/frazione_dura: STESSO schema due-fasi (morbida->dura/STE) gia'
# verificato in quant_ternaria_globale._prova_due_fasi_ternaria (80%/20%).
NUCLEO_BETA_MAX = float(os.getenv("THOR_VALIDA_BETA_MAX", "16.0"))
# 24/9 23:5x — FONTE UNICA dei pesi commessi. Prima il commit usava forward_duro (exp/sigmoid
# COMPILATI) e i codici su disco la stessa formula EAGER: su GPU le due differiscono di 1 ULP fp32,
# e i pochi z che cadono ESATTAMENTE sulla soglia delta finiscono dalla parte opposta (misurato su
# strato 0: 3270 pesi = ~3 codici x 1024 dopo la de-rotazione, salto max 1,22e-4 = D/32). La catena
# vedeva un modello, il GGUF ne riceveva un altro. Ora il modulo si scrive DAI CODICI estratti, con
# le scale arrotondate a fp16 come le scrive il GGUF (TQ1_0: `d` e' ggml_half) — cio' che vede la
# catena e' per costruzione cio' che vede llama.cpp (a meno della rotazione dell'ingresso, B9).
COMMIT_DA_CODICI = os.getenv("THOR_COMMIT_DA_CODICI", "1") == "1"
COMMIT_SCALA_FP16 = os.getenv("THOR_COMMIT_SCALA_FP16", "1") == "1"
NUCLEO_FRAZIONE_DURA = float(os.getenv("THOR_VALIDA_FRAZIONE_DURA", "0.2"))
NUCLEO_LR = float(os.getenv("THOR_VALIDA_LR", "0.05"))
# 25/9 00:2x — SCHEDULA del passo. Misurato: S0 ripetuta col solo commit dai codici (ingresso di [4,5]
# spostato dello 0,04%: v20 8,2717e-5 -> 8,2754e-5) finisce 0,842x -> 0,875x della v20 (+4%): con lr
# COSTANTE fino all'ultimo passo si tiene un'iterata rumorosa. "coseno" porta lr a ~0 alla fine
# (0,5(1+cos(pi t/T)), per gruppo, sul lr di ciascun gruppo). CAMBIA-NUMERI: default "costante" (corse
# gia' fatte riproducibili), il coseno si adotta solo se il pannello lo conferma.
LR_SCHEDULA = os.getenv("THOR_VALIDA_LR_SCHEDULA", "costante")
if LR_SCHEDULA not in ("costante", "coseno"):
    raise ValueError(f"THOR_VALIDA_LR_SCHEDULA={LR_SCHEDULA!r}: attese 'costante' o 'coseno'")


def passi_duri(n_passi: int) -> int:
    """Passi della fase dura (STE) di un chunk: la regola di ottimizza_finestra_JOINT, in un posto
    solo (26/9: la legge anche la forgia scorrevole, per rifiutare PRIMA di caricare uno strato
    una THOR_ANCORA_A=init senza passi morbidi, vedi ancora_forgia.verifica_ricetta)."""
    return max(1, int(n_passi * NUCLEO_FRAZIONE_DURA))


def fattore_lr(t: int, n_passi: int, schedula: str = LR_SCHEDULA) -> float:
    """Moltiplicatore del lr al passo t (0-based) di n_passi."""
    if schedula == "costante":
        return 1.0
    import math
    return 0.5 * (1.0 + math.cos(math.pi * t / n_passi))
# 0 = nessuno split: la finestra intera (5 strati) allenata INSIEME in un solo
# chunk — l'opzione economica di default. Il compito ammette esplicitamente
# "sotto-finestre di 4 se la memoria non basta": qui e' un ripiego DISPONIBILE
# (chunking sequenziale coi confini di errore ricalcolati, sotto), non
# un'affermazione che 4 basti — la vera soglia si misura su Thor VERO (fuori
# perimetro di sessione qui: nessun caricamento del modello intero).
NUCLEO_SOTTOFINESTRA_MAX = int(os.getenv("THOR_VALIDA_SOTTOFINESTRA", "0"))

OUT_DIR = Path(os.getenv("THOR_VALIDA_OUT", "@@ROOT:models@@/gguf/tony-tern/valida-finestra-8-12"))
GGUF_OUT = Path(os.getenv(
    "THOR_VALIDA_GGUF_OUT", "@@ROOT:models@@/gguf/tony-tern/Thor-v31-TEST-finestra8-12.gguf"))

# guardia GPU condivisa (§verifica_gpu_libera)
VRAM_MAX_GIB = float(os.getenv("THOR_VALIDA_VRAM_MAX_GIB", "10.0"))   # storico: non piu' usata dalla guardia (25/9)
VRAM_UTILE_GIB = 95.15   # oltre, i byte scivolano in GTT e costano RAM host 1:1 (misurato 5/9)
VRAM_SERVE_GIB = float(os.getenv("THOR_VALIDA_VRAM_SERVE_GIB", "14.0"))   # sonda/banco di un Thor da ~9-12 GiB + contesto
GPU_OCCUPATA = Path(__file__).resolve().parent / "strumenti" / "gpu_occupata.sh"
GPU_PCT_MAX = int(os.getenv("THOR_VALIDA_GPU_PCT_MAX", "15"))

# sonda finale — STESSA di stanotte (3 domande x 5 seed = 15)
LLAMA_BIN_K1024 = Path(os.getenv(
    "THOR_LLAMA_BIN_K1024", "@@ROOT:models@@/build/llamacpp-tq1-k1024/build/bin/llama-server"))
PORTA_SONDA = int(os.getenv("THOR_VALIDA_PORTA_SONDA", "11462"))
UNITA_SONDA = os.getenv("THOR_VALIDA_UNITA_SONDA", "adhoc-sonda-valida-8-12")  # ⚠️ prefisso adhoc- obbligatorio
SEMI_SONDA = (1, 2, 3, 7, 42, 99, 1234)   # audit B8: 7 seed (regola di progetto: sonda multi-seed)
DOMANDE_SONDA = (
    ("Un prodotto costa 23 euro. Sconto del 15 per cento, poi meno altri 2 euro. "
     "Quanto costa alla fine? Rispondi col solo numero.", "NUM:17.55"),
    ("Alice ha 3 fratelli e 2 sorelle. Quante sorelle ha il fratello di Alice? "
     "Rispondi col solo numero.", "NUM:3"),
    ("Scrivi la parola CANE al contrario. Rispondi con la sola parola.", r"enac"),
)

# baseline/obiettivo — misurati stanotte, letterali (§CRITERIO DI SUCCESSO)
BASELINE_OK, BASELINE_MUTO = 6, 2
OBIETTIVO_OK_MIN, OBIETTIVO_MUTO_MAX = 9, 0
# audit B8: soglie tarate su 15 prove (3 domande x 5 seed): riscalate sul
# totale attuale (3 x 7 = 21). Il confronto serio e' con la BASE misurata
# nella stessa sessione, non con queste costanti.
OBIETTIVO_OK_MIN = -(-OBIETTIVO_OK_MIN * len(SEMI_SONDA) * 3 // 15)
BASELINE_OK = round(BASELINE_OK * len(SEMI_SONDA) * 3 / 15)

TIPO_BASE_ATTESO = int(T.TQ1_0)        # 34 — quello che ffn_*_exps ha oggi
TIPO_NUOVO = int(T.TQ1_0R_1024)        # 45 — quello che scriviamo


def log(*a) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ═══════════════════════════ 1. GUARDIA GPU CONDIVISA ═══════════════════════
def _rocm_smi_numeri() -> dict:
    """Due invocazioni SEPARATE di rocm-smi (showuse, showmeminfo) — stesso
    formato gia' letto a mano in questa sessione. Ritorna None sui campi che
    non e' riuscita a leggere: MAI un default ottimistico silenzioso."""
    out = {"vram_used_gib": None, "vram_total_gib": None, "gpu_pct": None}
    try:
        r = subprocess.run(["rocm-smi", "--showuse"], capture_output=True, text=True, timeout=10)
        m = re.search(r"GPU use \(%\):\s*(\d+)", r.stdout)
        if m:
            out["gpu_pct"] = int(m.group(1))
    except Exception as e:
        log(f"⚠️ rocm-smi --showuse fallito: {e}")
    try:
        r = subprocess.run(["rocm-smi", "--showmeminfo", "vram"], capture_output=True, text=True, timeout=10)
        mu = re.search(r"VRAM Total Used Memory \(B\):\s*(\d+)", r.stdout)
        mt = re.search(r"VRAM Total Memory \(B\):\s*(\d+)", r.stdout)
        if mu:
            out["vram_used_gib"] = int(mu.group(1)) / 2**30
        if mt:
            out["vram_total_gib"] = int(mt.group(1)) / 2**30
    except Exception as e:
        log(f"⚠️ rocm-smi --showmeminfo fallito: {e}")
    return out


def verifica_gpu_libera(vram_serve_gib: float = VRAM_SERVE_GIB, gpu_pct_max: int = GPU_PCT_MAX) -> None:
    """⛔⛔ Guardiano VERO, non un `pgrep -f <nome>` (lezione pagata il 22/9,
    [[feedback_guardiano_gpu_libera_per_nome_di_processo]]: un pipeline che
    controllava un nome di processo ha dichiarato "GPU libera" mentre una
    corsa da 46h girava sotto un altro nome). Legge la RISORSA (VRAM/%uso) +
    un secondo segnale indipendente (unita' thor-*/adhoc-* systemd --user
    gia' ATTIVE, non 'failed'). Fail-CLOSED: se rocm-smi non si legge, non
    si procede — un "libera" ottimistico su dato mancante e' peggio di un
    rifiuto rumoroso."""
    n = _rocm_smi_numeri()
    if n["vram_used_gib"] is not None and n["gpu_pct"] is not None:
        log(f"rocm-smi: VRAM usata {n['vram_used_gib']:.2f}/{n['vram_total_gib']:.1f} GiB "
            f"· GPU busy {n['gpu_pct']}%")
    else:
        log(f"rocm-smi: LETTURA INCOMPLETA — {n}")

    guasti = []
    if n["vram_used_gib"] is None or n["gpu_pct"] is None:
        guasti.append("rocm-smi non leggibile: impossibile verificare che la GPU sia libera")
    else:
        # ⛔ 25/9 13:00 — qui c'era «VRAM usata totale > 10 GiB → rifiuto»: con la CASA accesa (cervello+turbo, 33 GiB, per
        #    ordine di Angelo) la sonda della forgia B si e' rifiutata di partire e il banco e' saltato. Era la trappola gia'
        #    scritta nelle regole del banco: la guardia va scritta come «quanto SERVE», non «quanto e' usato». Ora tre segnali:
        #    (1) VRAM LIBERA sufficiente per il lavoro; (2) nessun ALTRO lavoro sulla GPU, misurato processo per processo
        #    (kfd + DRM, casa esente) da strumenti/gpu_occupata.sh; (3) nessuna unita' thor-*/adhoc-* attiva (sotto).
        libera = VRAM_UTILE_GIB - n["vram_used_gib"]
        if libera < vram_serve_gib:
            guasti.append(f"VRAM libera {libera:.2f} GiB < necessaria {vram_serve_gib} GiB")
        if n["gpu_pct"] > gpu_pct_max:
            guasti.append(f"GPU busy {n['gpu_pct']}% > soglia {gpu_pct_max}%")
    try:
        r = subprocess.run([str(GPU_OCCUPATA)], capture_output=True, text=True, timeout=60,
                           env={**os.environ, "GPU_ESCLUDI_PID": f"{os.getpid()} {os.getppid()}"})
        if r.returncode != 0:
            guasti.append(f"GPU occupata da un altro lavoro (server della casa esclusi): {r.stdout.strip()[:300]}")
    except Exception as e:  # fail-closed: senza la misura per processo non si procede
        guasti.append(f"strumenti/gpu_occupata.sh non eseguibile: {e}")

    try:
        r = subprocess.run(
            ["systemctl", "--user", "list-units", "thor-*", "adhoc-*",
             "--no-legend", "--state=running,activating"],
            capture_output=True, text=True, timeout=10)
        vive = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
        if vive:
            guasti.append(f"unita' systemd --user gia' ATTIVE (secondo segnale): {vive}")
    except Exception as e:
        log(f"⚠️ systemctl list-units fallito (non bloccante di per se', ma nessuna conferma): {e}")

    if guasti:
        for g in guasti:
            log(f"⛔ GPU NON libera: {g}")
        raise SystemExit(3)
    log("✅ GPU libera (o quasi) — procedo")


def precondizioni() -> None:
    """Stesso spirito di `cattura_schur_esperti.precondizioni`: muore PRIMA
    di qualunque lavoro se una radice manca (mai un `return` silenzioso —
    [[feedback_percorso_cablato_in_file_che_viaggia]])."""
    guasti = []
    for nome, p in (("THOR_HF_DIR", SRC), ("THOR_SCHUR_BASE", GGUF_BASE), ("THOR_CORPUS", CORPUS)):
        if not p.exists():
            guasti.append(f"{nome}={p} NON ESISTE")
    if not torch.cuda.is_available():
        guasti.append("nessuna GPU: questa validazione richiede CUDA/ROCm")
    if not LLAMA_BIN_K1024.exists():
        guasti.append(f"llama-server K1024 non trovato: {LLAMA_BIN_K1024} "
                       "(serve il fork con supporto TQ1_0R_1024, non il build standard)")
    if guasti:
        for g in guasti:
            log(f"⛔ PRECONDIZIONE VIOLATA: {g}")
        raise SystemExit(4)
    log(f"precondizioni ok — finestra {FINESTRA_L0}-{FINESTRA_L1} ({len(FINESTRA)} strati)")


# ═══════════════════════ 2. CORPUS REALE, POCHE BATCH ═══════════════════════
def carica_poche_batch() -> torch.Tensor:
    """Tokenizza `THOR_CORPUS` (STESSO corpus di calibrazione della cattura a
    scala piena) e ne prende SOLO N_BATCH x BATCH_SEQ sequenze da L_SEQ token
    — 'corpus reale a poche batch', mai l'intero corpus (113 batch nella
    cattura completa). Ritorna (N_BATCH, BATCH_SEQ, L_SEQ) int64 su CPU: si
    sposta su cuda un batch alla volta (mai tutto insieme)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(SRC))
    testo = CORPUS.read_text(encoding="utf-8", errors="replace")
    ids = tok(testo, return_tensors="pt", add_special_tokens=False).input_ids[0]
    servono = N_BATCH * BATCH_SEQ * L_SEQ
    if ids.numel() < servono:
        raise RuntimeError(
            f"corpus {CORPUS} troppo corto per {N_BATCH}x{BATCH_SEQ}x{L_SEQ}: "
            f"{ids.numel()} token, ne servono {servono}")
    ids = ids[:servono].reshape(N_BATCH, BATCH_SEQ, L_SEQ)
    log(f"corpus '{CORPUS.name}': {N_BATCH} batch x {BATCH_SEQ} seq x {L_SEQ} tok "
        f"({servono} campioni totali — 'poche batch reali', non l'intero corpus)")
    return ids


def seme_embedding_quantizzato(ids_batch: torch.Tensor, emb: torch.Tensor,
                               contenitori: "tuple | None" = None) -> "tuple[dict, dict]":
    """audit B12: in produzione l'embedding e' quello del GGUF base (Q5_K), non il
    bf16: la finestra 0 deve partire con e0 = emb_Q5K[ids] - emb_bf16[ids], non con
    e0 = 0. Dequantizza SOLO le righe dei token presenti nella calibrazione.
    Ritorna (h0_per_b, e0_per_b) pronti come seme esterno di ottimizza_finestra_JOINT.

    25/9 (cache per lotto, critica 2): `contenitori` = (h, e) dati dalla forgia a
    THOR_CACHE_LOTTI=1: si riempiono lotto per lotto invece di materializzare sulla GPU
    tutti i lotti insieme (14,2 GiB a 151 lotti, h 32 + e fp32 64 MiB ciascuno). Stessi
    tensori; le norme per `rel` si prendono lotto per lotto sugli STESSI tensori, nello
    stesso ordine: stesso valore del log. None (default) = i due dict di prima."""
    from dequant_gguf_torch import _dequant_raw
    t = next(t for t in GGUFReader(str(GGUF_BASE)).tensors if t.name == "token_embd.weight")
    unici = torch.unique(ids_batch)
    righe = np.asarray(t.data)[unici.numpy()]
    emb_q = torch.from_numpy(_dequant_raw(righe, t)).to(emb.device, torch.float32)   # (n_unici, hidden)
    assert emb_q.shape == (unici.numel(), emb.shape[1]), f"forma embedding quantizzato {tuple(emb_q.shape)}"
    posizione = torch.searchsorted(unici, ids_batch)                                  # indice nella tabella ridotta
    h0, e0 = ({}, {}) if contenitori is None else contenitori
    norme_e, norme_h = [], []
    for b in range(ids_batch.shape[0]):
        ids_b = ids_batch[b].to(emb.device)
        h_b = torch.nn.functional.embedding(ids_b, emb).to(emb.dtype)
        e_b = (emb_q[posizione[b].to(emb.device)] - h_b.float()).detach()
        norme_e.append(e_b.norm())
        norme_h.append(h_b.float().norm())
        h0[b], e0[b] = h_b, e_b
        del h_b, e_b
    rel = (torch.stack(norme_e).norm() / torch.stack(norme_h).norm()).item()
    log(f"B12: seme della finestra 0 dall'embedding {t.tensor_type.name} della base · errore relativo {rel:.4f}")
    return h0, e0


CALIB_DATI = Path(os.getenv("THOR_CALIB_DATI", "@@ROOT:models@@/gguf/tony-tern/pannello"))
CALIB_EDU_FILE = "fineweb-edu.sample-10BT.000_00000.rg0-1.jsonl"
CALIB_EDU_PRIMO_DOC = 173          # 0-172 appartengono al pannello held-out (pannello_ids.json)
CALIB_WIKI_FILE = "wikitext-2-raw-v1.train.raw"   # SOLO train: validation = pannello, test = PPL
# 25/9: corpus multi-dominio GIA' tokenizzato (costruisci_corpus_calibrazione.py): {dominio: (n, L_SEQ) int64}
CALIB_CORPUS = os.getenv("THOR_CALIB_CORPUS", "")


def _carica_corpus_pronto(percorso: str) -> dict:
    """Corpus pronto `{dominio: (n, L_SEQ) int64}`. Se accanto c'e' il manifesto (stesso nome,
    `.json`), lo sha256 dei token di OGNI dominio deve essere quello dichiarato."""
    import hashlib
    p = Path(percorso)
    corpus = torch.load(p, weights_only=True)
    if not isinstance(corpus, dict) or not corpus:
        raise ValueError(f"THOR_CALIB_CORPUS={p}: atteso un dizionario {{dominio: tensore}}")
    for nome, t in corpus.items():
        if not (isinstance(t, torch.Tensor) and t.dtype == torch.int64 and t.dim() == 2 and t.shape[1] == L_SEQ):
            raise ValueError(f"THOR_CALIB_CORPUS={p}: dominio {nome!r} non e' (n, {L_SEQ}) int64")
    man = p.with_suffix(".json")
    if man.exists():
        dichiarati = json.loads(man.read_text(encoding="utf-8")).get("domini", {})
        for nome, t in corpus.items():
            sha = hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
            if nome not in dichiarati or dichiarati[nome].get("sha256_token") != sha:
                raise RuntimeError(f"THOR_CALIB_CORPUS={p}: dominio {nome!r} non combacia col manifesto {man.name}")
    else:
        log(f"⚠️ THOR_CALIB_CORPUS={p}: manifesto {man.name} assente, impronte non verificate")
    return corpus


def _batch_miscela(tok, esempi_cot: list, spec: str) -> torch.Tensor:
    """Batch di calibrazione MISTI (24/9 22:4x): `spec` = "dominio:sequenze_per_batch,...",
    domini cot (ragionamento math/code in formato chat, gli stessi esempi di prima), edu
    (FineWeb-Edu grezzo dal documento 173) e wiki (wikitext-2 TRAIN grezzo). La somma delle
    sequenze deve essere BATCH_SEQ. Ogni batch contiene TUTTI i domini nelle stesse quote.
    Solleva se un dominio non ha testo sufficiente (mai batch corte o ripetute in silenzio)."""
    quote = {}
    for pezzo in spec.split(","):
        nome, n = pezzo.split(":")
        quote[nome.strip()] = int(n)
    corpus = _carica_corpus_pronto(CALIB_CORPUS) if CALIB_CORPUS else None
    ammessi = set(corpus) if corpus is not None else {"cot", "edu", "wiki"}
    somma = sum(quote.values())
    # 25/9: le quote valgono per un CICLO di `giro` batch (somma = giro x BATCH_SEQ; es. sedicesimi
    # con 8 sequenze per batch = ciclo di 2). giro 1 = la costruzione di prima, riga per riga.
    giro = somma // BATCH_SEQ if somma > 0 and somma % BATCH_SEQ == 0 else 0
    if giro < 1 or not set(quote) <= ammessi or min(quote.values()) < 0 or N_BATCH % giro:
        raise ValueError(f"THOR_CALIB_MISCELA={spec!r}: domini ammessi {sorted(ammessi)}, quote >= 0, somma = "
                         f"multiplo di BATCH_SEQ ({BATCH_SEQ}) e N_BATCH ({N_BATCH}) multiplo del ciclo "
                         f"({somma}/{BATCH_SEQ})" + (f" · corpus {CALIB_CORPUS}" if CALIB_CORPUS else ""))

    def _sequenze(ids: list, n: int, nome: str) -> torch.Tensor:
        if len(ids) < n * L_SEQ:
            raise RuntimeError(f"calibrazione mista: {nome} ha {len(ids)} token, ne servono {n * L_SEQ}")
        return torch.tensor(ids[:n * L_SEQ], dtype=torch.int64).view(n, L_SEQ)

    pool = {}
    for nome, q in quote.items():
        n = q * (N_BATCH // giro)
        if q == 0:
            continue
        if corpus is not None:
            if corpus[nome].shape[0] < n:
                raise RuntimeError(f"corpus {CALIB_CORPUS}: dominio {nome} ha {corpus[nome].shape[0]} sequenze, "
                                   f"ne servono {n} (mai sequenze ripetute)")
            pool[nome] = corpus[nome][:n].clone()
        elif nome == "cot":
            flusso = []
            for _tipo, testo in esempi_cot:
                if len(flusso) >= n * L_SEQ:
                    break
                flusso += tok(testo, add_special_tokens=False).input_ids
            pool[nome] = _sequenze(flusso, n, nome)
        elif nome == "edu":
            docs = [json.loads(r)["text"] for i, r in enumerate((CALIB_DATI / CALIB_EDU_FILE).open(encoding="utf-8"))
                    if i >= CALIB_EDU_PRIMO_DOC]
            pool[nome] = _sequenze(tok("\n\n".join(docs), add_special_tokens=False).input_ids, n, nome)
        else:
            testo = (CALIB_DATI / CALIB_WIKI_FILE).read_text(encoding="utf-8")
            pool[nome] = _sequenze(tok(testo, add_special_tokens=False).input_ids, n, nome)
    batch = []
    if giro == 1:
        for b in range(N_BATCH):
            batch.append(torch.cat([pool[k][b * q:(b + 1) * q] for k, q in quote.items() if q > 0], dim=0))
        return torch.stack(batch, dim=0)
    posti = sorted(((i + 0.5) / q, j, k) for j, (k, q) in enumerate(quote.items()) if q > 0 for i in range(q))
    ordine = [k for _, _, k in posti]
    preso = {k: 0 for k in quote}
    for _c in range(N_BATCH // giro):
        for g in range(giro):
            righe = []
            for k in ordine[g * BATCH_SEQ:(g + 1) * BATCH_SEQ]:
                righe.append(pool[k][preso[k]])
                preso[k] += 1
            batch.append(torch.stack(righe, dim=0))
    assert all(preso[k] == pool[k].shape[0] for k in pool), "ogni sequenza del pool va usata una volta"
    return torch.stack(batch, dim=0)


def _guardia_heldout_corpus(miscela: str, seme: int, usati_ho: int) -> None:
    """25/9 (verifica): il corpus pronto esclude dal dominio cot solo gli esempi del SUO held-out storico
    (0..fine del primo intervallo `esclusi`, ordine mescolato col SUO seme). L'held-out di questa corsa deve
    starci dentro, altrimenti le sue sequenze sono anche di addestramento. Solleva, mai in silenzio."""
    import hashlib
    quote = {p.split(":")[0].strip(): int(p.split(":")[1]) for p in miscela.split(",") if p.strip()}
    if quote.get("cot", 0) <= 0:
        return
    man = Path(CALIB_CORPUS).with_suffix(".json")
    if not man.exists():
        raise RuntimeError(f"THOR_CALIB_CORPUS={CALIB_CORPUS}: manifesto {man.name} assente, impossibile verificare "
                           "che l'held-out storico sia fuori dal dominio cot del corpus")
    m = json.loads(man.read_text(encoding="utf-8"))
    cot = m.get("domini", {}).get("cot", {})
    fine = int(cot.get("esclusi", [[0, -1]])[0][1]) + 1          # held-out escluso dal corpus: esempi 0..fine-1
    problemi = []
    if int(m.get("seme", -1)) != seme:
        problemi.append(f"seme {seme} contro {m.get('seme')} del corpus")
    if str(m.get("tokenizzatore")) != str(SRC):
        problemi.append(f"tokenizzatore {SRC} contro {m.get('tokenizzatore')} del corpus")
    if usati_ho > fine:
        problemi.append(f"held-out = esempi CoT 0..{usati_ho - 1}, il corpus esclude solo 0..{fine - 1} "
                        f"(con BATCH_SEQ={BATCH_SEQ} e L_SEQ={L_SEQ}: THOR_VALIDA_N_HELDOUT=1, come i lanciatori, "
                        "o un corpus ricostruito con l'esclusione giusta)")
    locali = (m.get("sorgenti") or {}).get("locali") or {}
    for tipo in ("math", "code"):
        atteso = (locali.get(f"cot_{tipo}") or {}).get("sha256")
        f = CORPUS.parent / f"cot_{tipo}.jsonl"
        if atteso and hashlib.sha256(f.read_bytes()).hexdigest() != atteso:
            problemi.append(f"{f} diverso da quello del corpus")
    if problemi:
        raise RuntimeError(f"THOR_CALIB_CORPUS={CALIB_CORPUS}: held-out storico NON disgiunto dal dominio cot: "
                           + "; ".join(problemi))


def carica_batch_calibrazione() -> "tuple[torch.Tensor, int]":
    """⛔⛔ 24/9 (audit B6): `carica_poche_batch` prendeva i PRIMI 4096 token del
    corpus piatto (~15 problemi di matematica, nessun formato chat, nessun
    <think>) — con l'ottimizzatore finalmente funzionante e' la ricetta del
    sovra-adattamento. Qui: gli esempi di cot_math.jsonl + cot_code.jsonl
    (ragionamenti generati dal modello stesso) nel FORMATO CHAT VERO del
    modello (apply_chat_template, ragionamento in <think>), mescolati con seme
    fisso, impacchettati in sequenze da L_SEQ token. Le ultime N_HELDOUT batch
    sono costruite da esempi DISGIUNTI da quelli di addestramento: servono a
    misurare il sovra-adattamento (loss dura prima/dopo).
    Ritorna (ids (N_BATCH+N_HELDOUT, BATCH_SEQ, L_SEQ) int64, n_addestramento)."""
    import random
    from transformers import AutoTokenizer
    n_ho = int(os.getenv("THOR_VALIDA_N_HELDOUT", "4"))
    seme = int(os.getenv("THOR_CALIB_SEED", "1234"))
    cartella = CORPUS.parent
    tok = AutoTokenizer.from_pretrained(str(SRC))
    esempi = []
    for tipo in ("math", "code"):
        for riga in (cartella / f"cot_{tipo}.jsonl").open(encoding="utf-8"):
            try:
                d = json.loads(riga)
            except json.JSONDecodeError:
                continue
            if d.get("errore") or not d.get("risposta"):
                continue
            testo = tok.apply_chat_template(
                [{"role": "user", "content": d["q"]},
                 {"role": "assistant", "content": d["risposta"], "reasoning_content": d.get("ragionamento", "")}],
                tokenize=False)
            esempi.append((tipo, testo))
    random.Random(seme).shuffle(esempi)
    per_seq = L_SEQ
    per_batch = BATCH_SEQ * per_seq

    def _impacca(sorgente, n_batch):
        flusso, usati, conta = [], 0, {"math": 0, "code": 0}
        while len(flusso) < n_batch * per_batch:
            if usati >= len(sorgente):
                raise RuntimeError(f"esempi di calibrazione finiti: servono {n_batch} batch da {per_batch} token")
            tipo, testo = sorgente[usati]
            flusso += tok(testo, add_special_tokens=False).input_ids
            conta[tipo] += 1
            usati += 1
        ids = torch.tensor(flusso[:n_batch * per_batch], dtype=torch.int64)
        return ids.reshape(n_batch, BATCH_SEQ, per_seq), usati, conta

    ids_ho, usati_ho, conta_ho = _impacca(esempi, n_ho)          # held-out: i PRIMI esempi mescolati
    miscela = os.getenv("THOR_CALIB_MISCELA", "")
    if CALIB_CORPUS and miscela:
        _guardia_heldout_corpus(miscela, seme, usati_ho)
    if CALIB_CORPUS and not miscela:
        raise ValueError(f"THOR_CALIB_CORPUS={CALIB_CORPUS} senza THOR_CALIB_MISCELA: dichiarare le quote "
                         "(es. wiki:3,edu:2,cot:2,it_web:2,it_wiki:1,it_chat:3,it_pensiero:1,strumenti:2)")
    if miscela:
        # ⛔⛔ 24/9 22:4x (pannello A1): con la sola calibrazione math/code in formato chat
        # le ricette battono la v20 SOLO su quel dominio (0,53x) e le sono PEGGIO sul testo
        # generale (wiki 1,70x, edu 1,43x): la PPL wikitext sarebbe peggiorata. Nessun esperto
        # e' a secco (copertura_esperti.py: >= 500 token ciascuno), ma i piu' usati dal testo
        # generale si ottimizzavano su ingressi math/code: DISALLINEAMENTO DI DISTRIBUZIONE.
        # Qui ogni batch mescola i domini ("cot:4,edu:2,wiki:2" = sequenze per batch).
        # L'held-out resta quello storico (confronti validi); il pannello resta FUORI:
        # FineWeb-Edu dal documento 173 (0-172 sono del pannello), wikitext solo TRAIN.
        ids_tr = _batch_miscela(tok, esempi[usati_ho:], miscela)
        log(f"calibrazione MISTA «{miscela}» (seme {seme}): {N_BATCH} batch di addestramento x "
            f"{BATCH_SEQ}x{L_SEQ} token + {n_ho} held-out storico ({usati_ho} esempi math/code chat)"
            + (f" · corpus pronto {CALIB_CORPUS}" if CALIB_CORPUS else ""))
        return torch.cat([ids_tr, ids_ho], dim=0), N_BATCH
    ids_tr, usati_tr, conta_tr = _impacca(esempi[usati_ho:], N_BATCH)   # addestramento: disgiunti
    log(f"calibrazione (formato chat + <think>, seme {seme}): {N_BATCH} batch di addestramento "
        f"({usati_tr} esempi: {conta_tr['math']} math, {conta_tr['code']} code) + {n_ho} held-out "
        f"({usati_ho} esempi disgiunti: {conta_ho['math']} math, {conta_ho['code']} code) · "
        f"{BATCH_SEQ}x{per_seq} token per batch · esempi disponibili {len(esempi)}")
    return torch.cat([ids_tr, ids_ho], dim=0), N_BATCH


# ═══════════════════ 3. FINESTRA CORRENTE (pesi da disco) ═══════════════════
def carica_finestra_corrente(layers: "list[int] | None" = None):
    """Costruisce gli strati della finestra coi pesi CORRENTI da disco.
    Import SOLA LETTURA di `cattura_schur_esperti` — questo file non ne
    tocca una riga, solo importa `costruisci_strato` (§docstring modulo).

    ⭐ 24/9: `layers` esplicito (default None → `FINESTRA`, comportamento
    INVARIATO per questo script) — permette a un chiamante ESTERNO (il loop
    a scorrimento sui 40 strati veri, non ancora scritto: vedi PIANO.md
    08:26) di caricare UNA finestra piccola alla volta invece dell'intera
    `FINESTRA` fissa di questo file. Nessuna chiamata esistente cambia
    comportamento (tutte omettono l'argomento).

    Ritorna (cfg, embed_bf16_su_cuda, {L: strato_pulito}, {L: strato_quantizzato}).
    ⚠️ Chi chiama e' responsabile di liberare (`libera_finestra`) appena finito
    — 5 strati bf16 con esperti MoE non sono gratis (regola 'una corsia
    pesante per volta', vale anche qui: libera prima di aprire il server sonda)."""
    layers = FINESTRA if layers is None else list(layers)
    os.environ.setdefault("THOR_HF_DIR", str(SRC))
    os.environ.setdefault("THOR_GPTQ_BASE", str(GGUF_BASE))
    from cattura_schur_esperti import costruisci_strato  # noqa: E402  (sola lettura)
    from transformers import AutoConfig
    from safetensors import safe_open

    cfg = AutoConfig.from_pretrained(str(SRC)).text_config
    cfg._attn_implementation = "sdpa"
    # ⭐ 24/9 07:xx — LEVA DI VELOCITA' VERIFICATA (agente dedicato, bit-esatto
    # rel_err=0.0): costruire lo strato a mano (init_empty_weights+load_state_
    # dict, mai from_pretrained) lascia cfg._experts_implementation=None per
    # sempre -> Qwen3_5MoeExperts sceglie la via "eager" (loop Python su fino
    # a 256 esperti attivi, 256 nodi autograd separati invece di uno solo:
    # backward 21,3x il forward). "grouped_mm" su ROCm (nessun kernel nativo)
    # ripiega da solo su _grouped_mm_fallback (torch.library.custom_op),
    # STESSA matematica, UN solo nodo autograd. Misurato: backward esperti
    # 246,1ms->63,3ms (3,9x), strato intero 496,1ms->328,0ms (1,51x).
    cfg._experts_implementation = "grouped_mm"
    # ⭐ 24/9 20:4x (radiografia a 8 seq: scatter dell'indicizzazione dei token 7%):
    # THOR_ESPERTI_THOR=1 -> stessa implementazione con gradiente diretto (gather con la
    # permutazione inversa) per raccolta dei token e permutazione finale (esperti_thor.py,
    # verificata da test_esperti_thor.py: identica al bit su CPU).
    if os.getenv("THOR_ESPERTI_THOR", "0") == "1":
        import esperti_thor
        esperti_thor.registra()
        cfg._experts_implementation = esperti_thor.NOME

    puliti, quantizzati = {}, {}
    for L in layers:
        log(f"  carico strato {L}: ramo pulito (bf16 HF) + ramo corrente (GGUF, da_gguf=True)")
        puliti[L] = costruisci_strato(cfg, L, da_gguf=False)
        quantizzati[L] = costruisci_strato(cfg, L, da_gguf=True)
        # ⭐ 24/9 20:0x (radiografia del passo): i parametri di questi moduli NON si
        # addestrano (si addestrano solo z/log_s/delta_raw dei ternari), ma nascevano
        # con requires_grad=True: a ogni passo torch calcolava i gradienti dei PESI di
        # attenzione/SSM/esperto condiviso/router (conv SSM "wrw": 35 ms a chiamata) e
        # li buttava. Il gradiente rispetto agli INGRESSI (quello che serve) resta.
        for p_ in list(puliti[L].parameters()) + list(quantizzati[L].parameters()):
            p_.requires_grad_(False)

    idx = json.load(open(SRC / "model.safetensors.index.json"))["weight_map"]
    kemb = "model.language_model.embed_tokens.weight"
    with safe_open(str(SRC / idx[kemb]), "pt") as sf:
        emb = sf.get_tensor(kemb).to("cuda", torch.bfloat16)

    log(f"finestra {layers[0]}-{layers[-1]} caricata (pulita+corrente) · "
        f"VRAM allocata {torch.cuda.memory_allocated()/2**30:.2f} GiB")
    return cfg, emb, puliti, quantizzati


def libera_finestra(*oggetti) -> None:
    """Libera SEMPRE la VRAM dopo ogni fase pesante (vincolo esplicito del
    workflow). `oggetti` puo' contenere tensori, moduli o dict {L: modulo}:
    si spacchettano i dict prima di cancellare."""
    for o in oggetti:
        if isinstance(o, dict):
            o.clear()
        del o
    gc.collect()
    torch.cuda.empty_cache()
    log(f"VRAM liberata · allocata ora {torch.cuda.memory_allocated()/2**30:.2f} GiB")


def calcola_confine_finestra(puliti: dict, quantizzati: dict, ids_batch: torch.Tensor,
                              cfg, emb: torch.Tensor,
                              h0_ingresso_per_b: "dict[int, torch.Tensor] | None" = None,
                              e0_ingresso_per_b: "dict[int, torch.Tensor] | None" = None,
                              h_fine_teacher_per_b: "dict[int, torch.Tensor] | None" = None,
                              contenitori: "tuple | None" = None) -> tuple:
    """⭐ 24/9 — il pezzo che il loop a scorrimento sui 40 strati usa ad OGNI
    cambio di finestra: dato l'output COMMESSO (pesi già aggiornati con i
    codici duri) di una finestra appena ottimizzata, calcola il vero
    `(h0, e0)` da passare come seme esterno (`h0_esterno_per_b`/
    `e0_esterno_per_b`) a `ottimizza_finestra_JOINT` per la finestra
    SUCCESSIVA — stessa identica idea, mai duplicata concettualmente, di
    `ottimizzazione_globale_loop.py::ottimizza_rete_multi_finestra` (lì su
    strati giocattolo, qui su Thor vero).

    ⚠️ Caso NO-OVERLAP (stride==len(layers), l'unico cablato oggi nel loop
    esterno): il confine è la FINE della finestra — l'ultimo layer di
    `puliti`/`quantizzati` in ordine. Per l'overlap (stride<len(layers))
    servirebbe il confine a un offset INTERNO — non implementato qui,
    dichiarato non ancora, non finto silenziosamente.

    Ritorna (h0_per_b, e0_per_b): due dict {b: tensore}, pronti per essere
    passati TALI E QUALI a `ottimizza_finestra_JOINT` come seme esterno.
    Stesso posizionale/maschera di `ottimizza_finestra_JOINT` — mai una
    seconda formula per lo stesso calcolo (lezione RoPE dtype, PIANO.md
    24/9 06:53: dtype del dummy = dtype del flusso vero, mai il default)."""
    from transformers.masking_utils import create_causal_mask
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding

    layers = sorted(puliti)
    assert sorted(quantizzati) == layers, "puliti e quantizzati devono coprire la stessa finestra"
    device = emb.device
    n_batch_loc, b_loc, l_seq_loc = ids_batch.shape

    hs_dummy = torch.zeros(b_loc, l_seq_loc, cfg.hidden_size, device=device, dtype=emb.dtype)
    rot = Qwen3_5MoeTextRotaryEmbedding(cfg).to(device)
    pos = torch.arange(l_seq_loc, device=device).unsqueeze(0).expand(b_loc, -1)
    pos_emb = rot(hs_dummy, pos)
    maschera = create_causal_mask(config=cfg, inputs_embeds=hs_dummy, attention_mask=None,
                                   past_key_values=None, position_ids=pos)
    del hs_dummy

    def _fwd(strato, h):
        o = strato(hidden_states=h, position_embeddings=pos_emb, attention_mask=maschera, position_ids=pos)
        return o[0] if isinstance(o, tuple) else o

    # ⛔⛔⛔ 24/9 15:1x — BUG TROVATO rileggendo per il refinement: senza
    # seme d'ingresso questa funzione ripartiva SEMPRE dall'embedding e
    # attraversava solo gli strati di QUESTA finestra — quindi dalla terza
    # finestra in poi il "confine vero" era F_{finestra j-1}(embedding), non
    # l'uscita vera di tutti gli strati 0..fine(j-1). Ogni finestra >=2 veniva
    # ottimizzata su un ingresso della distribuzione sbagliata. Ora il
    # chiamante passa il seme con cui QUESTA finestra e' stata ottimizzata,
    # ricostruito ESATTAMENTE come fa ottimizza_finestra_JOINT (h_q = h_t+e0).
    # Default None = comportamento vecchio (corretto SOLO per la finestra che
    # parte dal vero strato 0).
    # 25/9 (cache per lotto): `contenitori` = (h, e) depositi NUOVI della finestra dopo
    # (deposito_lotti.contenitori_seme, mai il deposito di cattura); None = i due dict di prima.
    # La precarica del lotto b+1 (e di b, se non gia' in volo) gira nel thread di I/O mentre si
    # calcola il lotto b; su dict e None e' un no-op.
    h0_per_b, e0_per_b = contenitori if contenitori is not None else ({}, {})
    with torch.no_grad():
        for b in range(n_batch_loc):
            DL.precarica((h0_ingresso_per_b, e0_ingresso_per_b, h_fine_teacher_per_b), b, b + 1)
            if h0_ingresso_per_b is None:
                h = torch.nn.functional.embedding(ids_batch[b].to(device), emb).to(emb.dtype)
                h_t = h_q = h
            else:
                h_t = h0_ingresso_per_b[b]
                e_in = None if e0_ingresso_per_b is None else e0_ingresso_per_b[b]
                h_q = h_t if e_in is None else (h_t + e_in.to(h_t.dtype))
            for L in layers:
                if h_fine_teacher_per_b is None:
                    h_t = _fwd(puliti[L], h_t)
                h_q = _fwd(quantizzati[L], h_q)
            if h_fine_teacher_per_b is not None:
                # ⭐ 24/9 21:0x: uscita del teacher gia' calcolata dall'ottimizzazione
                # (stesso strato, stesso ingresso h0_ingresso): metà dei forward del confine
                h_t = h_fine_teacher_per_b[b]
            h0_per_b[b] = h_t
            e0_per_b[b] = (h_q - h_t).detach()
    return h0_per_b, e0_per_b


# ═══════════════ 4. DIAGNOSTICA DI OGGI (reale, non placeholder) ════════════
def forward_finestra(strati: dict, ids_batch: torch.Tensor, cfg, emb: torch.Tensor) -> dict:
    """Un forward SENZA gradiente della finestra (questa e' la diagnostica di
    OGGI — il forward CON gradiente per l'ottimizzatore vero e' dentro
    §NUCLEO, non qui). Ritorna {L: hidden_state (n_token, d) dopo lo strato L},
    catturato sullo STESSO hook (`post_attention_layernorm`) che
    `cattura_schur_esperti.py` usa per hs_ref/hs_q — cosi' la diagnostica di
    oggi e la loss di domani leggono lo STESSO segnale."""
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding
    from transformers.masking_utils import create_causal_mask

    B, l_seq = ids_batch.shape
    with torch.no_grad():
        hs = torch.nn.functional.embedding(ids_batch.to("cuda"), emb).to(torch.bfloat16)
        rot = Qwen3_5MoeTextRotaryEmbedding(cfg).to("cuda")
        pos = torch.arange(l_seq, device="cuda").unsqueeze(0)
        pos_emb = rot(hs, pos)
        maschera = create_causal_mask(config=cfg, inputs_embeds=hs, attention_mask=None,
                                       past_key_values=None, position_ids=pos)

        uscite = {}
        for L in sorted(strati):
            catture = {}
            h = strati[L].post_attention_layernorm.register_forward_hook(
                lambda _m, _i, o, _L=L: catture.__setitem__(_L, o.detach()))
            o = strati[L](hidden_states=hs, position_embeddings=pos_emb,
                           attention_mask=maschera, position_ids=pos)
            h.remove()
            hs = o[0] if isinstance(o, tuple) else o
            # (n_token, d): appiattisce batch e sequenza, come H_rif/H_hat
            # in loss_corrispondenza_statistica (che lavora su 2D).
            uscite[L] = catture[L].reshape(-1, catture[L].shape[-1]).float()
        return uscite


def misura_diagnostica_oggi(puliti: dict, quantizzati: dict, ids: torch.Tensor, cfg, emb: torch.Tensor) -> dict:
    """Numero VERO, misurato oggi (non un placeholder): quanto la finestra
    8-12 e' "rotta" ORA, per-strato, con la loss di corrispondenza statistica
    Eq.6 (`loss_corrispondenza_statistica.perdita_statistiche`, GIA'
    verificata su tensori sintetici). Qui NON uso `perdita_ge`/`insieme_strati`
    (quelle sono l'aggregato Eq.9 sull'INTERA rete a 40 strati — fuori
    perimetro di una finestra da 5: userebbero un layer-set pensato per 40).

    Concatena TUTTE le N_BATCH batch caricate (poche, per costruzione) prima
    di misurare — piu' token = proiezione casuale (Eq.6) meno rumorosa.
    Ritorna {L: {"d_mu":.., "d_c":.., "d_k":.., "totale":..}} (float python,
    non tensori: questo dizionario si stampa/salva, non entra in un grafo).

    ⭐ 24/9: gli strati iterati sono `sorted(puliti)` (le chiavi REALMENTE
    caricate), non piu' il globale `FINESTRA` — per questo file (chiamato
    sempre con `puliti` = l'intera `FINESTRA`) e' lo stesso identico
    insieme; per un chiamante esterno con una finestra diversa (il loop a
    scorrimento, PIANO.md 08:26) e' l'unico comportamento corretto: usare
    `FINESTRA` avrebbe sollevato KeyError o, peggio, letto silenziosamente
    gli strati SBAGLIATI se le taglie combaciassero per caso."""
    layers = sorted(puliti)
    per_strato_rif: dict[int, list[torch.Tensor]] = {L: [] for L in layers}
    per_strato_hat: dict[int, list[torch.Tensor]] = {L: [] for L in layers}
    for b in range(ids.shape[0]):
        u_rif = forward_finestra(puliti, ids[b], cfg, emb)
        u_hat = forward_finestra(quantizzati, ids[b], cfg, emb)
        for L in layers:
            per_strato_rif[L].append(u_rif[L])
            per_strato_hat[L].append(u_hat[L])

    risultati = {}
    for L in layers:
        H_rif = torch.cat(per_strato_rif[L], dim=0)
        H_hat = torch.cat(per_strato_hat[L], dim=0)
        R = crea_proiezione_casuale(H_rif.shape[-1], K_PROIEZIONE_DEFAULT, seed=L, device=H_rif.device)
        d = perdita_statistiche(H_rif, H_hat, R)
        risultati[L] = {k: float(v) for k, v in d.items()}
        log(f"  strato {L}: D_mu={risultati[L]['d_mu']:.4e}  D_c={risultati[L]['d_c']:.4e}  "
            f"D_k={risultati[L]['d_k']:.4e}  totale={risultati[L]['totale']:.4e}")
    return risultati


# ══════════════ 5.0 §NUCLEO — mattoni dell'aggancio (24/9) ══════════════════
# Tutto cio' che serve a `ottimizza_finestra_JOINT` per lavorare su STRATI
# VERI (Qwen3_5MoeDecoderLayer, reali o giocattolo — la classe e' la stessa,
# solo le dimensioni cambiano): rotazione condizionale, monkeypatch di
# gate_up_proj/down_proj/router, costruzione dei TernarioDiff, StratoFn.
def _ruota_se_possibile(V: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """Rotazione Hadamard a blocchi di QK=1024 (formato GGUF di destinazione
    TQ1_0R_1024, SOLO gate/up — vedi §NUCLEO) — SOLO se l'ultima dimensione e'
    multiplo di QK (vero su Thor reale: hidden=2048). A scala GIOCATTOLO
    (in_feat << 1024, vincolo di sessione) il blocco non esiste: si ritorna
    l'INGRESSO INVARIATO, esplicitamente (non un'approssimazione silenziosa) —
    l'identita' e' una rotazione no-op autoconsistente SOLO perche' applicata
    IDENTICA in init/forward/export, mai mescolata con la rotazione vera."""
    if V.shape[-1] % QK != 0:
        return V
    return _R_1024_torch(V, S.to(V.dtype))


def _deruota_se_possibile(V: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """Inversa di `_ruota_se_possibile` — STESSA condizione, STESSO ripiego
    identita' a scala giocattolo."""
    if V.shape[-1] % QK != 0:
        return V
    return _RT_1024_torch(V, S.to(V.dtype))


def _costruisci_ternari_layer(tensore_corrente: torch.Tensor, S: torch.Tensor,
                               group: int, ruota: bool) -> TernarioDiffBatchFuso:
    """UN'UNICA istanza `TernarioDiffBatchFuso` (asse E in testa), sul tensore
    CORRENTE (gia' forgiato/GGUF di produzione — MAI da zero, come richiesto):
    `tensore_corrente` e' l'intero gate_up_proj (E, n_out, n_in) (ruota=True,
    rotazione TQ1_0R_1024 PRIMA di costruire il modulo — cosi' il latente z
    parte gia' vicino ai codici del formato di DESTINAZIONE, non a quelli
    TQ1_0 non ruotato di oggi) oppure l'intero down_proj (ruota=False, TQ1_0
    standard — §NUCLEO).

    ⭐ 24/9: sostituisce le E istanze `TernarioDiff` in un `ModuleList`
    (ricostruite/lette con un loop Python di `tensore_corrente[e]`) con UNA
    sola costruzione batch — la rotazione (quando `ruota=True`) e' applicata
    UNA volta sul tensore intero (E, n_out, n_in), non E volte su E fette:
    stessa matematica (broadcast sui leading dims, `_ruota_se_possibile` non
    fa alcuna riduzione fra esperti), zero loop Python. Verificato bit-esatto
    contro il loop-di-riferimento in
    `quant_ternaria_globale_batch_v2._verifica_equivalenza_v2`."""
    W = tensore_corrente.detach().float()
    if ruota:
        W = _ruota_se_possibile(W, S)
    return TernarioDiffBatchFuso(W, group=group)


@contextlib.contextmanager
def _sostituisci_parametro(modulo: torch.nn.Module, nome: str, tensore: torch.Tensor):
    """Sostituisce TEMPORANEAMENTE un nn.Parameter del modulo con un tensore
    DIFFERENZIABILE qualunque (qui: la ricostruzione TernarioDiff corrente) —
    cosi' `Qwen3_5MoeExperts.forward`, che legge `self.gate_up_proj`/
    `self.down_proj` FRESCHI ad ogni chiamata (mai cache), usa la ricostruzione
    invece del peso congelato, e il gradiente scorre nei parametri z/log_s/
    delta_raw di TernarioDiff. Verificato (script di prova, 24/9): forward
    senza NaN, gradiente vivo, ripristino ESATTO (stesso oggetto Parameter,
    stesso posto — `_parameters`, non `__dict__` — dopo l'uscita: un forward
    NORMALE post-ripristino da' lo STESSO output bit-esatto di uno mai patchato).

    ⛔ nn.Module.__setattr__ RIFIUTA di assegnare un Tensor non-Parameter a un
    nome gia' registrato come Parameter — va prima tolto da `_parameters` e
    scritto con `object.__setattr__` (bypassa quel controllo); al ripristino
    va tolto da `__dict__` (dove l'abbiamo messo) PRIMA di rimetterlo in
    `_parameters`, o l'attributo resterebbe visibile da DUE posti."""
    originale = modulo._parameters.pop(nome)
    object.__setattr__(modulo, nome, tensore)
    try:
        yield
    finally:
        del modulo.__dict__[nome]
        modulo._parameters[nome] = originale


@contextlib.contextmanager
def _instrada_congelato(blocco_moe: torch.nn.Module, selezionati: torch.Tensor, pesi: torch.Tensor):
    """Instradamento FISSO per il router VERO (`Qwen3_5MoeTopKRouter`), stesso
    principio di `finestra_globale.congela_routing` ma sul modulo REALE (che,
    a differenza dello scaffold giocattolo, e' gia' un nn.Module chiamabile
    direttamente — nessun bisogno di `moe_routing.instradamento`, scritto
    apposta per contesti SENZA un router vero da richiamare). I logit tornati
    sono un placeholder (SparseMoeBlock li scarta, `_, pesi, selezionati =
    self.gate(...)`): l'assert su n_tok e' la vera rete di sicurezza, scopre
    subito un uso scorretto (batch diversa da quella su cui e' stato congelato)."""
    gate = blocco_moe.gate

    def _fisso(hidden_states, _sel=selezionati, _pesi=pesi):
        n_tok = hidden_states.reshape(-1, hidden_states.shape[-1]).shape[0]
        assert n_tok == _sel.shape[0], (
            f"routing congelato per {_sel.shape[0]} token, ricevuti {n_tok} — "
            "batch diversa da quella usata per congelare (bug di chiamata)")
        return hidden_states.new_zeros(n_tok, gate.num_experts), _pesi, _sel

    gate.__dict__["forward"] = _fisso
    try:
        yield
    finally:
        gate.__dict__.pop("forward", None)


@contextlib.contextmanager
def _ingresso_ruotato(esperti: torch.nn.Module, S: torch.Tensor):
    """Ruota l'ingresso del modulo esperti (R·x a blocchi di 1024, la STESSA funzione
    che ruota le righe dei pesi): con gate_up nel dominio ruotato, W_rot·(R·x) = W·x.
    Il router e l'esperto condiviso NON passano di qui (moduli separati)."""
    def _gancio(_m, args, kwargs):
        if args:
            x = args[0]
            return (_ruota_se_possibile(x.float(), S).to(x.dtype), *args[1:]), kwargs
        x = kwargs["hidden_states"]
        return args, {**kwargs, "hidden_states": _ruota_se_possibile(x.float(), S).to(x.dtype)}
    maniglia = esperti.register_forward_pre_hook(_gancio, with_kwargs=True)
    try:
        yield
    finally:
        maniglia.remove()


def _gate_up_per_forward(gu_rot: torch.Tensor, S: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """gate_up da mettere nel modulo: ruotato (se ROTAZIONE_INGRESSO) o de-ruotato."""
    return gu_rot.to(dtype) if ROTAZIONE_INGRESSO else _deruota_se_possibile(gu_rot, S).to(dtype)


def _ctx_rotazione(esperti: torch.nn.Module, S: torch.Tensor):
    return _ingresso_ruotato(esperti, S) if ROTAZIONE_INGRESSO else contextlib.nullcontext()


def _ricostruisci_pesi_duri(chunk: list, strati_correnti: dict, ternari_gu: dict,
                            ternari_dw: "dict | None", S: "torch.Tensor | None") -> dict:
    """{L: (gate_up reale, down reale | None)} coi codici DURI correnti, nel dtype
    del modulo. Dipende solo dai codici, mai dai dati: si calcola UNA volta e si
    riusa per tutte le batch (24/9 19:5x — prima si rifaceva per ognuna delle 36
    batch a ogni ricongelo e in ogni valutazione held-out)."""
    pronti: dict = {}
    with torch.no_grad():
        for L in chunk:
            _dt = strati_correnti[L].mlp.experts.gate_up_proj.dtype
            gu = _gate_up_per_forward(ternari_gu[L].forward_duro(NUCLEO_BETA_MAX), S, _dt)
            dw = (None if ternari_dw is None or ternari_dw[L] is None
                  else ternari_dw[L].forward_duro(NUCLEO_BETA_MAX).to(_dt))
            pronti[L] = (gu, dw)
    return pronti


def _congela_routing_chunk(chunk: list, strati_correnti: dict, h0_quant: torch.Tensor, forward_grezzo,
                           ternari_gu: "dict | None" = None, ternari_dw: "dict | None" = None,
                           S: "torch.Tensor | None" = None, pesi_pronti: "dict | None" = None) -> dict:
    """Un forward NO_GRAD sequenziale attraverso il chunk (nell'ORDINE di
    `chunk`, numeri di strato VERI — mai un indice posizionale/`id(strato)`,
    che si disallineerebbe silenziosamente se l'ordine cambiasse) coi pesi
    CORRENTI (per il primo chunk: quelli di produzione; per i successivi:
    quelli GIA' committati dal chunk precedente) — cattura, per ogni strato,
    la coppia (selezionati, pesi) del router VERO, da tenere FISSA per
    l'intera ottimizzazione del chunk. Stesso principio di
    `finestra_globale.congela_routing` — qui sul modulo REALE, con un
    forward-hook su `post_attention_layernorm` (STESSO punto di cattura di
    `forward_finestra`, sopra) invece di ricostruire l'input a mano.
    Ritorna {L: (selezionati, pesi)}."""
    congelati: dict = {}
    h = h0_quant
    with torch.no_grad():
        for L in chunk:
            strato = strati_correnti[L]
            catture = {}
            hook = strato.post_attention_layernorm.register_forward_hook(
                lambda _m, _i, o: catture.__setitem__("v", o.detach()))
            if ternari_gu is not None:
                # audit B7: esperti = codici ternari DURI correnti (non i pesi di
                # strati_correnti), router VERO (niente instradamento congelato qui)
                if pesi_pronti is not None:
                    gu, dw = pesi_pronti[L]
                else:
                    _dt = strato.mlp.experts.gate_up_proj.dtype
                    gu = _gate_up_per_forward(ternari_gu[L].forward_duro(NUCLEO_BETA_MAX), S, _dt)
                    dw = (None if ternari_dw is None or ternari_dw[L] is None
                          else ternari_dw[L].forward_duro(NUCLEO_BETA_MAX).to(_dt))
                sost_dw = (contextlib.nullcontext() if dw is None
                           else _sostituisci_parametro(strato.mlp.experts, "down_proj", dw))
                with _sostituisci_parametro(strato.mlp.experts, "gate_up_proj", gu), sost_dw, \
                     _ctx_rotazione(strato.mlp.experts, S):
                    h_next = forward_grezzo(strato, h)
                del gu, dw
            else:
                h_next = forward_grezzo(strato, h)
            hook.remove()
            valore = catture["v"].reshape(-1, catture["v"].shape[-1])
            _, pesi, selezionati = strato.mlp.gate(valore)
            congelati[L] = (selezionati, pesi)
            h = h_next
    return congelati


def _stratofn_reale(strato: torch.nn.Module, forward_grezzo) -> "StratoFn":
    """StratoFn per il ramo TEACHER (o per il ramo quant SENZA sostituzioni,
    non usato qui ma simmetrico): un forward VERO del layer, nessun
    monkeypatch. Il kernel Lemma-1 lo chiama gia' sotto `torch.no_grad()` per
    il teacher (disciplina imposta DENTRO `errore_lemma1_finestra`)."""
    def f(h: torch.Tensor) -> torch.Tensor:
        return forward_grezzo(strato, h)
    return f


def _stratofn_quant_reale(strato: torch.nn.Module, ternari_gu: TernarioDiffBatchFuso,
                           ternari_dw: TernarioDiffBatchFuso, S: torch.Tensor,
                           selezionati: torch.Tensor, pesi: torch.Tensor,
                           beta: float, duro: bool, forward_grezzo,
                           pesi_pronti: "tuple | None" = None,
                           raccoglitore: "dict | None" = None, chiave=None) -> "StratoFn":
    """StratoFn per il ramo QUANTIZZATO: ad ogni chiamata (h, poi h+e — DUE
    volte per passo, stesso beta/duro/routing per costruzione: il kernel non
    li fa mai divergere entro un passo) ricostruisce gate_up_proj (ruotato →
    de-ruotato per il forward) E down_proj (SENZA rotazione) dalle istanze
    batch `TernarioDiffBatchFuso` CORRENTI, li sostituisce nel modulo esperti,
    blocca il routing, esegue il layer VERO, ripristina. Le DUE famiglie
    insieme, nello STESSO forward — esattamente come richiede la §NUCLEO 24/9
    ("down_proj nella STESSA ottimizzazione, non a parte")."""
    # ⭐ 24/9 05:xx — LEVA DI VELOCITA' MISURATA (profiling reale: 85,3% del
    # tempo di forward va qui): la ricostruzione ternaria NON dipende da h —
    # dipende solo da (ternari_gu/dw, beta, duro), tutti FISSI per l'intero
    # passo. Il kernel chiama pero' questa funzione due volte per passo (h,
    # poi h+e): ricostruirla DENTRO f(h) la rifaceva identica due volte.
    # Spostata FUORI dalla chiusura: calcolata una volta sola quando
    # _stratofn_quant_reale viene creata (gia' una volta per passo), riusata
    # da entrambe le chiamate a f(). Nessun cambio alla matematica.
    #
    # ⭐ 24/9 (integrazione batch-fuso): UNA chiamata `.forward(beta)`/
    # `.forward_duro(beta)` sull'istanza batch, invece di un loop Python di E
    # chiamate + `torch.stack` — stessa uscita (E, n_out, n_in), nucleo
    # compilato (torch.compile, catena fusa in un solo giro di memoria invece
    # di ~9), stessa matematica verificata bit-esatta.
    _dtype = strato.mlp.experts.gate_up_proj.dtype
    if pesi_pronti is not None:
        # solo valutazione (no_grad, codici duri): pesi gia' ricostruiti una volta
        # per tutte le batch da _ricostruisci_pesi_duri — stesso valore di
        # forward_duro(qualunque beta): beta tocca solo il gradiente surrogato
        gu_reale, dw_reale = pesi_pronti
    else:
        _bf16_gu = NUCLEO_BF16 and ROTAZIONE_INGRESSO and _dtype == torch.bfloat16
        gu_rot = (ternari_gu.forward_duro(beta, uscita_bf16=_bf16_gu) if duro
                  else ternari_gu.forward(beta))
        gu_reale = _gate_up_per_forward(gu_rot, S, _dtype)
        # audit B5: ternari_dw None = down v20 congelato (nessuna sostituzione)
        dw_reale = (None if ternari_dw is None else
                    (ternari_dw.forward_duro(beta, uscita_bf16=NUCLEO_BF16 and _dtype == torch.bfloat16)
                     if duro else ternari_dw.forward(beta)).to(_dtype))  # down: NIENTE rotazione

    def f(h: torch.Tensor) -> torch.Tensor:
        # ROUTING_AGGIORNATO: solo nel forward con gradiente (F̂(h+e), l'ingresso VERO dello
        # studente) si cattura l'uscita di post_attention_layernorm = ingresso del router
        cattura = raccoglitore is not None and torch.is_grad_enabled()
        if cattura:
            _preso = {}
            _gancio = strato.post_attention_layernorm.register_forward_hook(
                lambda _m, _i, o: _preso.__setitem__("v", o.detach()))
        sost_dw = (contextlib.nullcontext() if dw_reale is None
                   else _sostituisci_parametro(strato.mlp.experts, "down_proj", dw_reale))
        with _sostituisci_parametro(strato.mlp.experts, "gate_up_proj", gu_reale), \
             sost_dw, \
             (_instrada_congelato(strato.mlp, selezionati, pesi) if selezionati is not None
              else contextlib.nullcontext()), \
             _ctx_rotazione(strato.mlp.experts, S):
            uscita = forward_grezzo(strato, h)
        if cattura:
            _gancio.remove()
            with torch.no_grad():
                v = _preso["v"].reshape(-1, _preso["v"].shape[-1])
                _, pesi_n, sel_n = strato.mlp.gate(v)
            raccoglitore[chiave] = (sel_n, pesi_n)
        return uscita
    return f


def _estrai_famiglia(ternari: TernarioDiffBatchFuso) -> tuple:
    """D (E,n_out,ng) float32 · Q (E,n_out,n_in) int8 {-1,0,1} — la STESSA
    identica decisione di `forward_duro` (hard = |z| vs delta), letta
    DIRETTAMENTE dai parametri batchati (z,log_s,delta_raw, gia' con l'asse E
    in testa — nessun `torch.stack` di E letture separate: e' gia' un unico
    tensore): la parametrizzazione soft-to-hard GIA' fornisce il codice
    ternario finale, senza bisogno di un GPTQ/coordinate-descent esterno
    (quello serve al metodo SEQUENZIALE vecchio, che parte da un W*
    fluttuante — qui il codice ternario e' gia' la variabile allenata).

    ⭐ 24/9: `ternari` e' UNA sola istanza batch (asse E in testa su
    log_s/z/delta_raw), non un `ModuleList` di E moduli — stessa formula,
    applicata in un colpo solo su tutti gli esperti insieme."""
    with torch.no_grad():
        s = ternari.log_s.exp().squeeze(-1)                # (E, n_out, ng)
        delta = ternari._delta_norm()                       # (E, n_out, ng, 1)
        z = ternari.z                                        # (E, n_out, ng, group)
        Q = torch.where(z > delta, torch.ones_like(z),
                         torch.where(z < -delta, -torch.ones_like(z), torch.zeros_like(z)))
        D = s.to(torch.float32).cpu()
        Qn = Q.reshape(ternari.shape).to(torch.int8).cpu()   # (E, n_out, n_in)
    return D.numpy(), Qn.numpy()


def pesi_da_DQ(D: np.ndarray, Q: np.ndarray, device, scala_fp16: bool = COMMIT_SCALA_FP16) -> torch.Tensor:
    """D (x) Q per gruppo -> (E, n_out, n_in) fp32, stessa forma di forward_duro.
    `scala_fp16`: D arrotondata a fp16 come la scrive il GGUF (round-to-nearest-even, come ggml)."""
    E, n_out, n_in = Q.shape
    ng = D.shape[-1]
    Dt = torch.from_numpy(np.ascontiguousarray(D, dtype=np.float32)).to(device)
    if scala_fp16:
        Dt = Dt.half().float()
        if not torch.isfinite(Dt).all():
            raise FloatingPointError("scala fuori dal campo fp16: il GGUF non la puo' rappresentare")
    Qt = torch.from_numpy(np.ascontiguousarray(Q)).to(device).to(torch.float32).reshape(E, n_out, ng, n_in // ng)
    return Qt.mul_(Dt.unsqueeze(-1)).reshape(E, n_out, n_in)     # in place: un solo tensore fp32


def pesi_duri_da_codici(codici: dict, S: torch.Tensor, device, dtype,
                        scala_fp16: bool = COMMIT_SCALA_FP16) -> "tuple[torch.Tensor, torch.Tensor | None]":
    """(gate_up REALE, down) dai codici di `_esporta_gate_up`/`_estrai_famiglia`: gate e up
    (famiglie separate su disco) riuniti nella fusa (righe 0:meta gate, meta: up), de-ruotati,
    poi cast; down diretto. E' L'UNICA funzione che trasforma codici in pesi del modulo: la
    usano il commit, la ripresa da disco e la verifica — nessuna seconda formula."""
    Dg, Qg = codici["ffn_gate_exps"]
    Du, Qu = codici["ffn_up_exps"]
    gu = _deruota_se_possibile(pesi_da_DQ(np.concatenate([Dg, Du], axis=1), np.concatenate([Qg, Qu], axis=1),
                                          device, scala_fp16), S).to(dtype)
    dw = (pesi_da_DQ(*codici["ffn_down_exps"], device, scala_fp16).to(dtype)
          if "ffn_down_exps" in codici else None)
    return gu, dw


def committa_strato(strato: torch.nn.Module, ternari_gu: TernarioDiffBatchFuso,
                    ternari_dw: "TernarioDiffBatchFuso | None", S: torch.Tensor) -> dict:
    """Estrae i codici dello strato (contratto d'uscita) e scrive nel modulo i pesi duri.
    Con COMMIT_DA_CODICI i pesi vengono DAI codici (fonte unica, vedi sopra); altrimenti
    dal forward_duro compilato (comportamento fino al 24/9, per riprodurre le corse vecchie)."""
    esp = strato.mlp.experts
    codici = {**_esporta_gate_up(ternari_gu, esp.gate_up_proj.shape[1] // 2)}
    if ternari_dw is not None:
        codici["ffn_down_exps"] = _estrai_famiglia(ternari_dw)
    with torch.no_grad():
        if COMMIT_DA_CODICI:
            gu, dw = pesi_duri_da_codici(codici, S, esp.gate_up_proj.device, esp.gate_up_proj.dtype)
        else:
            gu = _deruota_se_possibile(ternari_gu.forward_duro(NUCLEO_BETA_MAX), S).to(esp.gate_up_proj.dtype)
            dw = (ternari_dw.forward_duro(NUCLEO_BETA_MAX).to(esp.down_proj.dtype)
                  if ternari_dw is not None else None)
        esp.gate_up_proj.data.copy_(gu)
        if dw is not None:
            esp.down_proj.data.copy_(dw)
        del gu, dw
    return codici


_ESPERTI_CAMPIONE_PASSO = 16   # 1 esperto ogni 16 (16 su 256): istantanea leggera, stessi esperti sempre


def _codici_duri(ternari: TernarioDiffBatchFuso, esperti: "slice") -> torch.Tensor:
    """Stessa regola di `_estrai_famiglia` (z vs delta normalizzato), solo sugli esperti campione."""
    delta = ternari._delta_norm()[esperti]
    z = ternari.z[esperti]
    return torch.where(z > delta, 1, torch.where(z < -delta, -1, 0)).to(torch.int8)


def _istantanea_codici(ternari: TernarioDiffBatchFuso) -> dict:
    """Audit Fase 2: codici duri e log-scale ALL'INIZIO del chunk (esperti campione)."""
    esperti = slice(None, None, _ESPERTI_CAMPIONE_PASSO)
    with torch.no_grad():
        return {"esperti": esperti, "Q0": _codici_duri(ternari, esperti),
                "log_s0": ternari.log_s[esperti].detach().clone()}


def _confronta_istantanea(ist: dict, ternari: TernarioDiffBatchFuso) -> dict:
    """Frazione di codici cambiati e mediana di log(s)/log(s0) rispetto all'inizio.
    Solo decadimento (bug B2) => rapporto (1-lr*wd)^passi = 0.90481; fermo => 1.0."""
    with torch.no_grad():
        Q = _codici_duri(ternari, ist["esperti"])
        frazione = (Q != ist["Q0"]).float().mean().item()
        log_s = ternari.log_s[ist["esperti"]]
        validi = ist["log_s0"].abs() > 1e-6
        rapporto = (log_s[validi] / ist["log_s0"][validi]).float()
        mediana = rapporto.median().item() if rapporto.numel() else float("nan")
    return {"frazione_cambiati": frazione, "mediana_rapporto_log_scala": mediana}


def _statistiche_gradiente(gruppi_per_tipo: dict) -> str:
    """Audit Fase 2: |grad| per tipo di parametro (campione a passo fisso: z e' da
    ~5e8 elementi). Riporta la frazione di gradienti non nulli (esperti instradati)
    e mediana/p99 fra i non nulli, contro l'eps=1e-8 di Adam."""
    parti = []
    with torch.no_grad():
        for tipo, ps in gruppi_per_tipo.items():
            campioni = [p.grad.reshape(-1)[::997].abs().float() for p in ps if p.grad is not None]
            if not campioni:
                parti.append(f"{tipo}: nessun grad")
                continue
            g = torch.cat(campioni)
            nz = g[g > 0]
            if nz.numel() == 0:
                parti.append(f"{tipo}: tutto zero")
                continue
            q = torch.quantile(nz[:2_000_000], torch.tensor([0.5, 0.99], device=nz.device))
            parti.append(f"{tipo}: non nulli {nz.numel() / g.numel() * 100:.1f}% · "
                         f"mediana {q[0].item():.2e} · p99 {q[1].item():.2e} (eps Adam 1e-8)")
    return " | ".join(parti)


def _esporta_gate_up(ternari_gu: TernarioDiffBatchFuso, meta: int) -> dict:
    """Spacchetta la famiglia FUSA gate_up (UNA istanza batch sul tensore
    (E, 2*intermedio, hidden) intero, come lo vede davvero
    `Qwen3_5MoeExperts.forward`: `gate, up = linear(...).chunk(2)`, quindi
    RIGHE 0:meta = gate, meta:2*meta = up — STESSA convenzione di
    `forgia_schur_esperti.py::main`, `meta = out_tot // 2`) in due famiglie
    GGUF separate."""
    D, Q = _estrai_famiglia(ternari_gu)
    return {"ffn_gate_exps": (D[:, :meta], Q[:, :meta]), "ffn_up_exps": (D[:, meta:], Q[:, meta:])}


# ══════════════════════ 5. §NUCLEO — IMPLEMENTATO 24/9 ══════════════════════
def ottimizza_finestra_JOINT(strati_correnti: dict, ids_batch: torch.Tensor, cfg, n_passi: int,
                              strati_teacher: dict, emb: torch.Tensor,
                              lr: float = NUCLEO_LR,
                              dimensione_sottofinestra: "int | None" = None,
                              raccogli_diagnostica: "list | None" = None,
                              h0_esterno_per_b: "dict[int, torch.Tensor] | None" = None,
                              e0_esterno_per_b: "dict[int, torch.Tensor] | None" = None,
                              n_batch_addestramento: "int | None" = None,
                              uscita_teacher_fine: "dict | None" = None,
                              ripartenza_per_strato: "dict | None" = None,
                              consegna_stato_per_strato: "dict | None" = None,
                              adam_ripartenza: str = "riparti",
                              n_batch_conoscenza: int = 0) -> dict:
    """Nucleo di ottimizzazione GLOBALE cross-strato — chiama il kernel Lemma-1
    di `finestra_globale.py` (`errore_lemma1_finestra`/`passo_ottimizzazione`,
    SOLA LETTURA, non una riga toccata) su una finestra di strati VERI
    (`Qwen3_5MoeDecoderLayer`, reali o giocattolo — vedi test in fondo al
    file), con `TernarioDiff` (`quant_ternaria_globale.py`, SOLA LETTURA) come
    parametrizzazione dei codici ternari di gate_up_proj E down_proj.

    Contratto d'ENTRATA (⭐ ESTESO 24/9 — vedi correzione (1)/(2) sotto):
        strati_correnti — {L: Qwen3_5MoeDecoderLayer} per L in FINESTRA,
                           pesi CORRENTI da disco (da carica_finestra_corrente,
                           ramo `quantizzati`). Il nucleo lavora SU QUESTI
                           IN-PLACE (li aggiorna a fine chunk coi codici duri
                           appena forgiati — vedi §COMMIT sotto): sono il
                           punto di partenza dell'ottimizzazione, non un
                           riferimento a sola lettura come nel ramo pulito.
        ids_batch        — (N_BATCH, BATCH_SEQ, L_SEQ) token id, STESSE poche
                           batch reali gia' tokenizzate da carica_poche_batch.
        cfg              — AutoConfig.text_config del modello (rotary/maschera).
        n_passi          — passi di training PER CHUNK (default THOR_VALIDA_
                           N_PASSI=200 — punto di partenza ECONOMICO, come
                           richiesto: alzare se il verdetto finale resta
                           sotto obiettivo con questo budget).
        strati_teacher   — {L: Qwen3_5MoeDecoderLayer} STESSA finestra, ramo
                           `puliti` (bf16 HF vero). ⭐ CORREZIONE 24/9: il
                           contratto originale non lo elencava, ma il kernel
                           Lemma-1 ha SEMPRE bisogno di F_l(h_l) (teacher,
                           no_grad) — senza questo parametro la funzione non
                           potrebbe MAI essere implementata. `main()` (sotto)
                           e' stato aggiornato per passare `puliti`.
        emb              — tabella di embedding (STESSA di carica_finestra_
                           corrente). ⭐ CORREZIONE 24/9: serve a materializzare
                           h0 da ids_batch — assente anch'essa dal contratto
                           originale, stesso motivo del punto precedente.
        lr, dimensione_sottofinestra, raccogli_diagnostica — opzionali, con
                           default sensati; l'ultimo e' un gancio di SOLA
                           diagnostica (riempito se non None) per i test in
                           fondo al file, MAI letto dal codice di produzione.

    §SOTTOFINESTRA (compito: "tutti e 5 insieme, o a sotto-finestre di 4 se
    la memoria non basta"): `dimensione_sottofinestra` (o l'env
    THOR_VALIDA_SOTTOFINESTRA, default 0=nessuno split) taglia `layers` in
    chunk CONSECUTIVI da al piu' quella taglia. Ogni chunk successivo al primo
    ricalcola h0/e0/routing-congelato dal `prefisso` (gli strati della
    finestra GIA' processati) coi pesi appena COMMESSI (§COMMIT) — mai una
    finestra "a meta'" scollegata dalle precedenti. Il valore giusto per Thor
    VERO non e' misurato qui (richiederebbe caricare il modello intero,
    vietato dal vincolo di sessione): il meccanismo e' pronto, la soglia si
    decide misurando (stesso principio di ogni altra soglia in questo lavoro).

    §COMMIT: a fine chunk, i codici DURI (`forward_duro`) vengono scritti
    IN-PLACE in `strati_correnti[L].mlp.experts.{gate_up_proj,down_proj}`
    (de-ruotati per gate_up, diretti per down) — cosi' il chunk SUCCESSIVO (e
    l'e0 che ne dipende) vede il flusso quantizzato AGGIORNATO, non quello di
    partenza. Con `dimensione_sottofinestra` >= len(finestra) (il default) i
    chunk sono 1 solo: il commit avviene una volta sola, a fine funzione.

    Contratto d'USCITA — per OGNI L in FINESTRA:
        {L: {"ffn_gate_exps": (D, Q), "ffn_up_exps": (D, Q), "ffn_down_exps": (D, Q)}}

    gate/up — D: (256, 512, 16) float32 (scala ogni GROUP=128 pesi — 512*2/128=8?
    ⚠️ ATTENZIONE a chi implementa: il numero di gruppi e' in_feat//GROUP,
    NON un valore fisso — lo stesso schema di forgia_schur_esperti.forgia_batch,
    dove out_rows=512 (gate O up, meta' del gate_up_proj fuso da 1024) e
    in_feat=2048 -> ng=16) e Q: (256, 512, 2048) int8 in {-1,0,1} — ESATTO
    contratto di gptq_1024_lib.pack_da_d_q PER RIGA (via
    esporta_schur_esperti.impacchetta_famiglia, che aggiunge l'asse esperti E
    in testa).

    down_proj — ⭐ AGGIUNTO 24/9 (stessa ottimizzazione globale, non a parte —
    vedi cattura_schur_down.py e finestra_globale.py::_test_down_proj_
    variabile_libera per il meccanismo, gia' verificato a scala giocattolo):
    D: (256, 2048, 2) float32 (scala ogni GROUP_DOWN=256 pesi, ng=inter//256=2,
    inter=512) e Q: (256, 2048, 512) int8 in {-1,0,1} — contratto di
    pack_tq1_0_da_d_q (pack_tq1_0_standard.py, formato TQ1_0 STANDARD SENZA
    rotazione), NON di gptq_1024_lib.pack_da_d_q (quello e' il codec ruotato
    di gate/up, blocco 1024 — i due packer NON sono intercambiabili, ne' lo
    e' il GROUP: 256 per down, 128 per gate/up, mai lo stesso valore).

    Qualunque sia la matematica interna (soft-to-hard ternario fuso + Lemma 1
    + AdamW su z/s per TUTT'E TRE le famiglie insieme, o altro), l'esportatore
    di questo file (`esporta_gguf_finestra`, sotto) non deve sapere COME i
    codici sono stati calcolati: consuma solo questo dizionario — ma va esteso
    (non lo e' ancora) per scrivere anche blk.{L}.ffn_down_exps.weight, vedi
    nota nel §NUCLEO del docstring di modulo.

    ⭐ CONSEGNATO 24/9 (compito successivo dello stesso workflow). Due lacune
    del contratto sopra sono state TROVATE scrivendo l'aggancio, non aggirate:
    (1) il kernel Lemma-1 ha SEMPRE bisogno di un ramo teacher (`F_l(h_l)`,
        no_grad) — il contratto d'entrata qui sopra non lo elenca affatto, e
        `main()` (sotto) chiamava questa funzione passando SOLO `quantizzati`.
        Corretto aggiungendo il parametro `strati_teacher` (il ramo `puliti`,
        gia' caricato da `carica_finestra_corrente`) e aggiornando la chiamata
        in `main()` di conseguenza — altrimenti la funzione non potrebbe MAI
        calcolare F_l(h_l), qualunque fosse la matematica interna.
    (2) per materializzare `h0` (l'ingresso della finestra) da `ids_batch`
        (token id) serve la tabella di embedding — anch'essa assente dal
        contratto e dalla chiamata in `main()`. Corretto aggiungendo il
        parametro `emb` (STESSO tensore gia' caricato da
        `carica_finestra_corrente`, mai ricaricato qui).

    ⛔⛔⛔ h0_esterno_per_b / e0_esterno_per_b (aggiunti 24/9, dopo aver
    CONFERMATO CON MISURA REALE — vedi PIANO.md 08:35 — che senza questo
    parametro il PRIMO chunk di una finestra che non parte dal vero strato 0
    della rete usa **embedding grezzo** come h0, invece del vero output degli
    strati precedenti: misurato ||output_vero_strati0-7||=91,22 contro
    ||embedding||=19,28, rapporto 4,58×, due distribuzioni quasi scorrelate).
    Default None su ENTRAMBI ⇒ comportamento ESATTAMENTE INVARIATO (embedding
    per il primo chunk, come sempre — corretto SE E SOLO SE questa finestra
    parte davvero dal layer 0 della rete). Se forniti (dict {b: tensore}, una
    voce per batch), sostituiscono `_emb_di(b)`/`prefisso` SOLO per il primo
    chunk di QUESTA chiamata (i0==0) — MAI per i chunk successivi, che
    restano legati al `prefisso` DENTRO la finestra come sempre. Uso previsto:
    il loop a scorrimento sui 40 strati (non ancora scritto) passa qui il
    vero `(h_l[offset], e_l[offset])` calcolato dalla finestra PRECEDENTE
    (stesso meccanismo, mai duplicato, di `ottimizzazione_globale_loop.py::
    ottimizza_rete_multi_finestra`), per ogni finestra tranne la prima.

    ⭐ 24/9 sera — FINESTRA SCORREVOLE A PASSO 1 ([0,1],[1,2],...: decisione di
    Angelo; logica in ripartenza_codici.py, patch strumenti/patch_ripartenza.py).
    Tre argomenti per chiamata, default = comportamento INVARIATO:
      ripartenza_per_strato={L: voce}: lo strato L parte da `voce` invece che da
        teacher + LS. voce = StatoStrato (parametri della finestra precedente,
        copiati al bit: la via per lo strato che scorre) oppure dict di codici
        come risultato[L] o `carica_codici_corsa` (ricostruzione esatta di D*Q,
        z dove lo metterebbe init_ls). Log per strato: codici identici ai forniti
        (100%) e scarto massimo dei pesi (0); altrimenti RuntimeError.
      consegna_stato_per_strato={L: None}: a fine finestra i valori diventano gli
        StatoStrato degli strati chiesti (riferimenti, zero copie).
      adam_ripartenza: "riparti" (AdamW nuovo, come SliderQuant) o "conserva"
        (i momenti dello strato consegnato/ripartito passano di finestra).

    ⭐ 26/9 — due strumenti per il sovra-adattamento misurato sull'MMLU (ancora_forgia.py), spenti
    di default (identico al bit):
      THOR_ANCORA_RAPPORTO=r > 0: perdita = dati + λ·ancora, λ = r·L0/A0 fissato al primo passo del
        chunk; ancora ai pesi del teacher (THOR_ANCORA_A=teacher) o dell'init del chunk (=init).
        Il gradiente passa dal gancio `dopo_backward` di passo_ottimizzazione.
      n_batch_conoscenza=k > 0: gli ULTIMI k lotti di ids_batch sono il held-out di CONOSCENZA
        (forgia_scorrevole, THOR_HELDOUT_CONOSCENZA): mai un passo, mai in λ; perdita DURA a init
        e a fine chunk contro la stessa v20, riga «held-out CONOSCENZA». Il held-out di prima resta
        i lotti n_tr .. n_batch-k-1.

    ⭐ 26/9 sera — THOR_SPD_LAMBDA=λ > 0 (spd_forgia.py): Selective Projection Decay DOPO ogni passo di
      AdamW (argomento `proiezione` di passo_ottimizzazione), θ0 = i latenti del chunk fotografati prima
      del primo passo (fp32, stesso device: 6,08 GiB su Thor a W = 2), liberata a fine ciclo prima delle
      valutazioni finali. Righe «SPD» nel log (accensione, cronometro, FINALE con distanza e codici
      cambiati su tutti gli esperti) e chiave "spd" nella diagnostica. Spenta: nulla di tutto questo."""
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding
    from transformers.masking_utils import create_causal_mask

    layers = sorted(strati_correnti)
    assert sorted(strati_teacher) == layers, (
        "strati_teacher deve coprire ESATTAMENTE la stessa finestra di strati_correnti "
        f"({sorted(strati_teacher)} != {layers})")
    assert ids_batch.dim() == 3, "ids_batch: (N_BATCH, BATCH_SEQ, L_SEQ), come da carica_poche_batch"
    if adam_ripartenza not in ("riparti", "conserva"):
        raise ValueError(f"adam_ripartenza={adam_ripartenza!r}: attesi 'riparti' o 'conserva'")
    for _nome_arg, _arg in (("ripartenza_per_strato", ripartenza_per_strato),
                            ("consegna_stato_per_strato", consegna_stato_per_strato)):
        if _arg is not None and not set(_arg) <= set(layers):
            raise ValueError(f"{_nome_arg}: strati {sorted(set(_arg) - set(layers))} fuori dalla finestra {layers}")

    device = emb.device
    S = torch.from_numpy(_S1024_NP).to(device)
    n_batch_loc, b_loc, l_seq_loc = ids_batch.shape
    # audit B6: le prime n_tr batch servono all'addestramento, le restanti sono
    # HELD-OUT (esempi disgiunti): mai usate per un passo, solo per misurare.
    n_tr = n_batch_loc if n_batch_addestramento is None else int(n_batch_addestramento)
    assert 0 < n_tr <= n_batch_loc, f"n_batch_addestramento={n_tr} fuori da (0, {n_batch_loc}]"
    # 26/9: held-out di CONOSCENZA = gli ultimi n_con lotti (0 = nessuno: batch_ho come prima)
    n_con = int(n_batch_conoscenza)
    if not 0 <= n_con <= n_batch_loc - n_tr:
        raise ValueError(f"n_batch_conoscenza={n_con} fuori da [0, {n_batch_loc - n_tr}] "
                         f"({n_batch_loc} lotti, {n_tr} di addestramento)")
    batch_ho = list(range(n_tr, n_batch_loc - n_con))
    batch_con = list(range(n_batch_loc - n_con, n_batch_loc))

    # posizionali/maschera dipendono SOLO dalla FORMA (B,L), non dal contenuto
    # dei token: calcolati una volta, riusati identici su tutte le n_batch_loc
    # batch (stessa forma per costruzione — carica_poche_batch le fa tutte
    # uguali) e su tutti gli strati (stesso L_SEQ per l'intera finestra).
    # dtype = emb.dtype (bf16), non il default float32: altrimenti
    # Qwen3_5MoeTextRotaryEmbedding.forward ritorna cos/sin in float32
    # (fa `cos.to(dtype=x.dtype)`), apply_rotary_pos_emb promuove q/k a
    # float32 (moltiplicazione per cos/sin) ma v resta bf16 (non tocca
    # cos/sin) -> SDPA "Expected query, key, and value to have the same
    # dtype". Mai emerso prima d'ora: chunk [8,9] sono SSM (niente RoPE),
    # strato 11 e' il primo full_attention reale toccato stanotte.
    hs_dummy = torch.zeros(b_loc, l_seq_loc, cfg.hidden_size, device=device, dtype=emb.dtype)
    rot = Qwen3_5MoeTextRotaryEmbedding(cfg).to(device)
    pos = torch.arange(l_seq_loc, device=device).unsqueeze(0).expand(b_loc, -1)
    pos_emb = rot(hs_dummy, pos)
    maschera = create_causal_mask(config=cfg, inputs_embeds=hs_dummy, attention_mask=None,
                                   past_key_values=None, position_ids=pos)
    del hs_dummy

    def _emb_di(b: int) -> torch.Tensor:
        return torch.nn.functional.embedding(ids_batch[b].to(device), emb).to(emb.dtype)

    def _forward_grezzo(strato, h: torch.Tensor) -> torch.Tensor:
        o = strato(hidden_states=h, position_embeddings=pos_emb, attention_mask=maschera, position_ids=pos)
        return o[0] if isinstance(o, tuple) else o

    dim_sub = dimensione_sottofinestra if dimensione_sottofinestra is not None else NUCLEO_SOTTOFINESTRA_MAX
    dim_sub = len(layers) if not dim_sub else max(1, min(dim_sub, len(layers)))
    chunk_layers_list = [layers[i:i + dim_sub] for i in range(0, len(layers), dim_sub)]
    log(f"ottimizza_finestra_JOINT: finestra {layers} → {len(chunk_layers_list)} chunk(i) "
        f"da al piu' {dim_sub} strati (THOR_VALIDA_SOTTOFINESTRA={NUCLEO_SOTTOFINESTRA_MAX}), "
        f"{n_passi} passi/chunk, {n_batch_loc} batch cicliche, lr={lr} (schedula {LR_SCHEDULA})")

    risultato: dict = {}
    for chunk in chunk_layers_list:
        i0 = layers.index(chunk[0])
        prefisso = layers[:i0]  # strati GIA' nella finestra ma PRIMA di questo chunk (chunk precedenti)

        # h0 (dal teacher) / e0 (quant - teacher) / routing congelato — UNA
        # volta per batch, PRIMA di costruire un solo TernarioDiff: h0/e0
        # scorrono attraverso il `prefisso` coi pesi CORRENTI (per il primo
        # chunk, prefisso=[] => h0=embedding, e0=None — la stessa assunzione
        # esplicita "nessun errore pregresso" gia' documentata nel kernel; per
        # i chunk successivi strati_correnti[prefisso] porta GIA' i codici
        # duri COMMESSI dal chunk precedente, quindi e0 riflette il flusso
        # quantizzato CORRENTE, non quello di partenza).
        seminato_esterno = i0 == 0 and h0_esterno_per_b is not None
        # 25/9 (cache per lotto, PROGETTO_CACHE_LOTTI.md §4): a THOR_CACHE_LOTTI=1 h0/e0/hq sono
        # depositi (h0/e0 per PROVENIENZA dal seme: nessuna scrittura; hq sempre su disco, si legge
        # solo al congelamento); spento DL.deposito() da' {} come prima. Il routing resta un dict:
        # pesa poco e col routing aggiornato si riscrive a ogni passo. Chiusi a fine chunk.
        h0_per_b, e0_per_b, congelati_per_b, hq_per_b = (
            DL.deposito("h0", chunk[0]), DL.deposito("e0", chunk[0]), {},
            DL.deposito("hq", chunk[0], classe=DL.UNA_VOLTA))
        for b in range(n_batch_loc):
            DL.precarica((h0_esterno_per_b, e0_esterno_per_b), b, b + 1)
            if seminato_esterno:
                # Vero h0/e0 seminato dal CHIAMANTE (finestra precedente del
                # loop a scorrimento) — questa finestra NON parte dal vero
                # strato 0 della rete, quindi embedding grezzo sarebbe
                # SBAGLIATO qui (vedi docstring, misura 4,58x). `prefisso`
                # resta [] per costruzione (i0==0): nessun doppio conteggio.
                # e0 esterno passato TALE E QUALE (mai ricalcolato da h_q-h_t:
                # ricadrebbe nella stessa cancellazione catastrofica bf16
                # gia' trovata e corretta stanotte in finestra_globale.py).
                h_t = h0_esterno_per_b[b]
                e0_qui = None if e0_esterno_per_b is None else e0_esterno_per_b[b]
                h_q = h_t if e0_qui is None else (h_t + e0_qui.to(h_t.dtype))
            else:
                # ⛔ 24/9 15:1x — stesso difetto di calcola_confine_finestra,
                # latente: con seme esterno e i0>0 (sotto-finestre) si
                # ripartiva dall'embedding ignorando il seme. Ora si parte dal
                # seme (se c'e') e si attraversa il prefisso.
                if h0_esterno_per_b is not None:
                    h_t = h0_esterno_per_b[b]
                    e_in = None if e0_esterno_per_b is None else e0_esterno_per_b[b]
                    h_q = h_t if e_in is None else (h_t + e_in.to(h_t.dtype))
                else:
                    h = _emb_di(b)
                    h_t = h_q = h
                with torch.no_grad():
                    for Lp in prefisso:
                        h_t = _forward_grezzo(strati_teacher[Lp], h_t)
                        h_q = _forward_grezzo(strati_correnti[Lp], h_q)
                e0_qui = None if (not prefisso and h0_esterno_per_b is None) else (h_q - h_t).detach()
            h0_per_b[b] = h_t
            e0_per_b[b] = e0_qui
            # audit B7: il routing si congela DOPO aver costruito i ternari (sotto),
            # coi codici veri — qui si conserva solo l'ingresso quantizzato.
            hq_per_b[b] = h_q
        # 25/9 (cache per lotto, revisione risorse): i nomi del ciclo restavano legati all'ultimo
        # lotto (l'held-out) per tutto il chunk — a cache accesa 96-128 MiB sulla GPU fuori dal
        # conto di §7.1. I valori vivono nei depositi (spenta: nei dict, che tengono gli stessi
        # oggetti; si libera prima solo `h`, l'embedding grezzo di un chunk con prefisso e senza
        # seme, mai in produzione). Nessun nome qui sotto li rilegge: nessun bit cambia.
        h = h_t = h_q = e_in = e0_qui = None

        # UNA istanza TernarioDiffBatchFuso per ciascuno strato del chunk —
        # DUE famiglie separate (gate_up_proj RUOTATO, down_proj NO),
        # inizializzate dal peso CORRENTE (gia' forgiato/GGUF di produzione,
        # MAI da zero). ⭐ 24/9: un'istanza batch sola per famiglia/strato
        # (asse E in testa), non piu' E moduli in un ModuleList.
        group_gu = _group_giocattolo(cfg.hidden_size, GRUPPO_GATEUP_REALE)
        group_dw = _group_giocattolo(cfg.moe_intermediate_size, GRUPPO_DOWN_REALE)
        # ⛔⛔⛔ 24/9 (audit B3/B4): la sorgente dell'init era `strati_correnti`
        # = la v20 DEQUANTIZZATA (gate/up gia' ternari, down a 4 bit), poi
        # ri-ternarizzata con s0=media|W| — errore sui pesi 0.65 contro il
        # teacher (la v20 stessa 0.48) e il 36% di neuroni morti della v20
        # (righe gate/up nulle: punto di sella, gradiente zero per sempre).
        # Ora di default: pesi ORIGINALI bf16 (`strati_teacher`) + init LS
        # (init_ls.reinit_ls_), codice duro iniziale = arrotondamento LS.
        # THOR_INIT_DA=corrente riproduce il vecchio comportamento (solo per A/B).
        init_da = os.getenv("THOR_INIT_DA", "teacher")
        init_ls = os.getenv("THOR_INIT_LS", "1") == "1"
        if init_da not in ("teacher", "corrente"):
            raise ValueError(f"THOR_INIT_DA={init_da!r}: attesi 'teacher' o 'corrente'")
        sorgente = strati_teacher if init_da == "teacher" else strati_correnti
        ternari_gu, ternari_dw = {}, {}
        for L in chunk:
            W_gu = sorgente[L].mlp.experts.gate_up_proj
            W_dw = sorgente[L].mlp.experts.down_proj
            ternari_gu[L] = _costruisci_ternari_layer(W_gu, S, group_gu, ruota=True).to(device)
            # audit B5: THOR_FORGIA_DOWN_TERNARIO=0 -> il down resta quello di
            # strati_correnti (v20, 4 bit) e non entra nell'ottimizzazione.
            ternari_dw[L] = (_costruisci_ternari_layer(W_dw, S, group_dw, ruota=False).to(device)
                             if DOWN_TERNARIO else None)
            if ripartenza_per_strato is not None and L in ripartenza_per_strato:
                # ⭐ 24/9 sera: finestra scorrevole a passo 1 — lo strato riparte dai parametri
                # della finestra precedente o dai codici di una corsa, non dal teacher
                from ripartenza_codici import applica_ripartenza_
                applica_ripartenza_(
                    L, ripartenza_per_strato[L], ternari_gu[L], ternari_dw[L],
                    W_gu_rif=lambda _W=W_gu: _ruota_se_possibile(_W.detach().float().to(device), S),
                    W_dw_rif=lambda _W=W_dw: _W.detach().float().to(device),
                    margine_z=float(os.getenv("THOR_LR_Z", str(lr))),
                    conserva_adam=(adam_ripartenza == "conserva"), log=log)
            elif init_ls:
                d_gu = reinit_ls_(ternari_gu[L], _ruota_se_possibile(W_gu.detach().float().to(device), S))
                d_dw = (reinit_ls_(ternari_dw[L], W_dw.detach().float().to(device)) if DOWN_TERNARIO
                        else {"frazione_disaccordo_ls": float("nan")})
                log(f"  strato {L}: init da {init_da} + LS · disaccordo codice duro vs LS: "
                    f"gu {d_gu['frazione_disaccordo_ls']*100:.4f}% · "
                    + (f"dw {d_dw['frazione_disaccordo_ls']*100:.4f}%" if DOWN_TERNARIO else "down v20 CONGELATO"))
            else:
                log(f"  strato {L}: init da {init_da} (formula media|W|, senza LS)")

        # ⭐ audit B7: routing congelato coi CODICI TERNARI iniziali (non coi pesi
        # v20 di strati_correnti): l'ingresso al router del 2° strato dipende
        # dagli esperti del 1°, che ora sono i ternari appena costruiti.
        if ROUTING_VIVO:
            # (None, None) = nessun instradamento imposto: decide il router vero a ogni forward
            congelati_per_b = {b: {L: (None, None) for L in chunk} for b in range(n_batch_loc)}
        else:
            _pronti = _ricostruisci_pesi_duri(chunk, strati_correnti, ternari_gu, ternari_dw, S)
            for b in range(n_batch_loc):
                DL.precarica((hq_per_b,), b, b + 1)
                congelati_per_b[b] = _congela_routing_chunk(chunk, strati_correnti, hq_per_b[b], _forward_grezzo,
                                                            ternari_gu=ternari_gu, ternari_dw=ternari_dw, S=S,
                                                            pesi_pronti=_pronti)
            del _pronti

        # UN SOLO ottimizzatore su TUTTI i parametri del chunk (tutti gli
        # strati, ENTRAMBE le famiglie) — stessa disciplina imposta dal test 4
        # di finestra_globale.py: mai due AdamW separati per famiglia/strato.
        # ⭐ 24/9: `.parameters()` letto DIRETTAMENTE sull'istanza batch (gia'
        # un solo nn.Module con log_s/z/delta_raw), non piu' un loop Python su
        # E sotto-moduli. `fused=True` — guadagno separato misurato (2.35x),
        # gratis: stessi identici aggiornamenti Adam, nucleo fuso CUDA/CPU
        # invece della sequenza non fusa di operazioni per-parametro.
        # ⛔⛔⛔ 24/9 (audit B2): AdamW creato SENZA weight_decay ereditava il
        # default di torch (1e-2, disaccoppiato), mai voluto: tirava log_s
        # verso 0 (scale x1,6), restringeva z (piu' zeri) anche sugli esperti
        # senza token. Impronta misurata: min(Dg_00)=exp(ln(1e-8)*0.9048).
        # Ora weight_decay esplicito (default 0.0) e lr separati per tipo di
        # parametro (z / log_s / delta_raw), regolabili per lo sweep.
        wd = float(os.getenv("THOR_WEIGHT_DECAY", "0.0"))
        lr_per_tipo = {
            "z": float(os.getenv("THOR_LR_Z", str(lr))),
            "log_s": float(os.getenv("THOR_LR_LOGS", str(lr))),
            "delta_raw": float(os.getenv("THOR_LR_DELTA", str(lr))),
        }
        gruppi_per_tipo: dict[str, list] = {k: [] for k in lr_per_tipo}
        for L in chunk:
            for modulo in (m for m in (ternari_gu[L], ternari_dw[L]) if m is not None):
                for nome, p in modulo.named_parameters():
                    if nome not in gruppi_per_tipo:
                        raise RuntimeError(f"parametro inatteso {nome!r} in TernarioDiffBatchFuso")
                    gruppi_per_tipo[nome].append(p)
        parametri = [p for ps in gruppi_per_tipo.values() for p in ps]
        opt = torch.optim.AdamW(
            [{"params": ps, "lr": lr_per_tipo[k]} for k, ps in gruppi_per_tipo.items() if ps],
            lr=lr, weight_decay=wd, fused=True)
        assert all(g["weight_decay"] == wd for g in opt.param_groups)
        if ripartenza_per_strato is not None and adam_ripartenza == "conserva":
            from ripartenza_codici import StatoStrato, inietta_adam_
            for L in chunk:
                _voce = ripartenza_per_strato.get(L)
                if isinstance(_voce, StatoStrato):
                    _n = inietta_adam_(opt, ternari_gu[L], ternari_dw[L], _voce)
                    log(f"  strato {L}: stato di Adam CONSERVATO dalla finestra precedente ({_n} tensori)")

        # strumentazione del cancello (audit Fase 2): istantanea dei codici e
        # delle scale INIZIALI, per misurare a fine chunk se l'ottimizzatore
        # ha lavorato davvero (e non solo il decadimento).
        istantanee = {(L, fam): _istantanea_codici(t)
                      for L in chunk for fam, t in (("gu", ternari_gu[L]), ("dw", ternari_dw[L])) if t is not None}

        n_dura = passi_duri(n_passi)
        n_morbida = n_passi - n_dura
        beta_soft = schedula_beta(1.0, NUCLEO_BETA_MAX, max(1, n_morbida))
        strati_teacher_fn = [_stratofn_reale(strati_teacher[L], _forward_grezzo) for L in chunk]
        # 26/9 (ancora_forgia.py): con THOR_ANCORA_RAPPORTO > 0 l'ancora del chunk nasce QUI, dopo l'init
        # dei ternari (per THOR_ANCORA_A=init fotografa proprio questi pesi) e prima del primo passo. Spenta:
        # None, e il ciclo sotto chiama passo_ottimizzazione esattamente come prima.
        _r_ancora, _rif_ancora = ancora_forgia.configurazione()
        ancora = None
        if _r_ancora > 0:
            ancora_forgia.verifica_ricetta(n_morbida, _r_ancora, _rif_ancora)
            ancora = ancora_forgia.Ancora(
                _r_ancora, _rif_ancora, chunk, ternari_gu, ternari_dw,
                {L: (strati_teacher[L].mlp.experts.gate_up_proj, strati_teacher[L].mlp.experts.down_proj)
                 for L in chunk},
                ruota_gu=lambda _X: _ruota_se_possibile(_X, S), log=log)

        # ⭐ 24/9 05:xx — LEVA DI VELOCITA' matematicamente certa (non una
        # stima): il ramo teacher e' deterministico e FISSO per l'intera
        # ottimizzazione del chunk (pesi congelati, h0_per_b[b] non cambia
        # mai fra un passo e l'altro) — trovato leggendo QUESTO ciclo, non
        # per analogia. Prima veniva richiamato 200 volte/chunk per ogni
        # batch; ora UNA sola volta per batch, riusato ad ogni passo.
        if DL.attiva():
            # 25/9 (cache per lotto): stesso calcolo, stesso ordine b = 0..n-1 (il conteggio esatto
            # di forgia_scorrevole._CatturaUscitaTeacher lo pretende), ma in un deposito: [0] = h0 e [1]
            # = l'uscita catturata sono gia' su disco (provenienza), si scrive solo [2:]. Spento: la
            # comprehension di prima, intatta.
            h_teacher_cache_per_b = DL.deposito("teacher", chunk[0])
            for b in range(n_batch_loc):
                DL.precarica((h0_per_b,), b, b + 1)
                h_teacher_cache_per_b[b] = precalcola_teacher(strati_teacher_fn, h0_per_b[b])
        else:
            h_teacher_cache_per_b = {b: precalcola_teacher(strati_teacher_fn, h0_per_b[b])
                                      for b in range(n_batch_loc)}
        # ⭐ 24/9 21:0x (confine 35 s/finestra): l'uscita del teacher a fine chunk e'
        # gia' qui; l'ultimo chunk finisce col suo ultimo strato = uscita della finestra.
        # Il chiamante la passa a calcola_confine_finestra, che non la ricalcola.
        if uscita_teacher_fine is not None:
            if isinstance(uscita_teacher_fine, DL.Deposito):
                # 25/9 (critica 5c): un deposito si riempie per PROVENIENZA (riferimento al file di
                # [-1], nessuna lettura, nessuna copia): accettato a cache accesa.
                uscita_teacher_fine.clear()
                for b in range(n_batch_loc):
                    uscita_teacher_fine[b] = h_teacher_cache_per_b.riferimento(b, -1)
            elif DL.attiva():
                raise NotImplementedError(
                    "uscita_teacher_fine e' un dict vero con THOR_CACHE_LOTTI=1: terrebbe sulla GPU l'uscita di "
                    "TUTTI i lotti (quello che la cache evita). Passare un DL.deposito(...) (per provenienza).")
            else:
                uscita_teacher_fine.clear()
                uscita_teacher_fine.update({b: h_teacher_cache_per_b[b][-1] for b in range(n_batch_loc)})

        # ⭐ audit B6/Fase 2: loss DURA (codici ternari veri, beta_max) sulle batch
        # HELD-OUT, mai viste dall'ottimizzatore. Tre riferimenti: esperti v20
        # (router vero), punto di partenza (init), risultato finale. Se il finale
        # non batte la v20 su questa finestra, il metodo non aggiunge nulla qui.
        def _loss_heldout(quali: str, congelati=None, lotti: "list | None" = None) -> float:
            # 26/9: `lotti` = quali lotti valutare (None = batch_ho, come prima; batch_con = CONOSCENZA)
            lotti = batch_ho if lotti is None else lotti
            if not lotti:
                return float("nan")
            vals = []
            pronti = (None if quali == "v20" else
                      _ricostruisci_pesi_duri(chunk, strati_correnti, ternari_gu, ternari_dw, S))
            with torch.no_grad():
                for bb in lotti:
                    if quali == "v20":
                        fq = [_stratofn_reale(strati_correnti[L], _forward_grezzo) for L in chunk]
                    else:
                        fq = [_stratofn_quant_reale(strati_correnti[L], ternari_gu[L], ternari_dw[L], S,
                                                    *congelati[bb][L], NUCLEO_BETA_MAX, True, _forward_grezzo,
                                                    pesi_pronti=pronti[L])
                              for L in chunk]
                    r = errore_lemma1_finestra(strati_teacher_fn, fq, h0_per_b[bb], e0_per_b[bb],
                                               h_l_teacher_cache=h_teacher_cache_per_b[bb])
                    vals.append(r["loss"].item())
            return sum(vals) / len(vals)

        ho_v20 = _loss_heldout("v20")
        ho_init = _loss_heldout("ternari", congelati_per_b)
        log(f"  chunk {chunk}: held-out ({len(batch_ho)} batch) loss DURA: v20 {ho_v20:.4e} · init {ho_init:.4e}")
        if batch_con:
            # 26/9: stessa macchina (stessi pesi, stessa v20, stesso routing congelato all'init) sui lotti di
            # CONOSCENZA — nessun passo li tocca (b = t % n_tr) e λ non li legge
            con_v20 = _loss_heldout("v20", lotti=batch_con)
            con_init = _loss_heldout("ternari", congelati_per_b, lotti=batch_con)
            log(f"  chunk {chunk}: held-out CONOSCENZA ({len(batch_con)} batch) loss DURA: v20 {con_v20:.4e} · "
                f"init {con_init:.4e}")

        # 26/9 sera (spd_forgia.py): con THOR_SPD_LAMBDA > 0 θ0 si fotografa QUI, dopo l'init dei ternari, l'AdamW
        # e le valutazioni iniziali (nessuna tocca i parametri: θ0 = l'init del chunk), subito prima del primo
        # passo — cosi' i 6,08 GiB di θ0 non si sommano alla preparazione del chunk. Spenta: None, e il ciclo
        # chiama passo_ottimizzazione esattamente come prima.
        _spd_lam, _spd_sempre, _spd_gran = spd_forgia.configurazione()
        spd = None
        _kw_spd: dict = {}
        if _spd_lam > 0:
            spd = spd_forgia.ProiezioneSPD(_spd_lam, _spd_sempre, _spd_gran, chunk, ternari_gu, ternari_dw, opt,
                                           log=log)

        perdite: list[float] = []
        ricongela = os.getenv("THOR_RICONGELA", "1") == "1"
        _t_crono = time.perf_counter()
        _saltati = 0
        for t in range(n_passi):
            b = t % n_tr
            duro = t >= n_morbida
            # ricongelo all'ingresso della fase dura, e (18:2x, dagli sweep) anche
            # PERIODICO: in modalita' solo-dura il routing restava quello
            # dell'init per tutti i passi e al ricalcolo finale si perdeva 5-7%.
            periodico = RICONGELA_OGNI > 0 and t > 0 and t % RICONGELA_OGNI == 0
            if ricongela and not ROUTING_VIVO and ((t == n_morbida and n_morbida > 0) or periodico):
                # audit B7: all'ingresso della fase dura i codici sono cambiati
                # rispetto all'init: ricalcola il routing coi codici ATTUALI e
                # misura quante scelte di esperti cambiano (flip rate).
                vecchi = congelati_per_b
                _pronti = _ricostruisci_pesi_duri(chunk, strati_correnti, ternari_gu, ternari_dw, S)
                congelati_per_b = {bb: _congela_routing_chunk(chunk, strati_correnti, hq_per_b[bb], _forward_grezzo,
                                                              ternari_gu=ternari_gu, ternari_dw=ternari_dw, S=S,
                                                              pesi_pronti=_pronti)
                                   for bb in range(n_batch_loc)}
                del _pronti
                flip = {L: sum((vecchi[bb][L][0].sort(-1).values != congelati_per_b[bb][L][0].sort(-1).values)
                               .float().mean().item() for bb in range(n_batch_loc)) / n_batch_loc
                        for L in chunk}
                log(f"  chunk {chunk}: routing ricongelato a t={t} "
                    f"({'periodico' if periodico else 'inizio fase dura'}) · flip rate "
                    + " · ".join(f"L{L} {v*100:.2f}%" for L, v in flip.items()))
            beta = NUCLEO_BETA_MAX if duro else beta_soft[min(t, len(beta_soft) - 1)]
            selez_pesi = congelati_per_b[b]
            _nuovi_routing = {} if (ROUTING_AGGIORNATO and not ROUTING_VIVO) else None
            strati_quant_fn = [
                _stratofn_quant_reale(strati_correnti[L], ternari_gu[L], ternari_dw[L], S,
                                       *selez_pesi[L], beta, duro, _forward_grezzo,
                                       raccoglitore=_nuovi_routing, chiave=L)
                for L in chunk
            ]
            if LR_SCHEDULA != "costante":
                _f_lr = fattore_lr(t, n_passi)
                for _g in opt.param_groups:
                    _g["lr"] = _g.setdefault("lr_base", _g["lr"]) * _f_lr
            if PROFILO and t == 5:
                from torch.profiler import profile as _profile, ProfilerActivity as _PA
                _prof = _profile(activities=[_PA.CPU, _PA.CUDA], record_shapes=False)
                _prof.__enter__()
            # 25/9 (cache per lotto): lettura anticipata del lotto del passo dopo nel thread di I/O,
            # mentre gira questo passo (e di questo, se non era gia' in volo); no-op sui dict.
            DL.precarica((h0_per_b, e0_per_b, h_teacher_cache_per_b), b)
            if t + 1 < n_passi:
                DL.precarica((h0_per_b, e0_per_b, h_teacher_cache_per_b), (t + 1) % n_tr)
            # 26/9: con l'ancora il gancio somma λ·∇A nei .grad dopo il backward dei dati (stesso beta e stessa
            # fase del passo); spenta, la chiamata e' quella di prima, argomento per argomento
            _kw_ancora = {} if ancora is None else {"dopo_backward": ancora.gancio(t, beta, duro)}
            # 26/9 sera: SPD agisce attorno a opt.step() dentro passo_ottimizzazione; spenta, nessun argomento in piu'
            if spd is not None:
                _kw_spd = {"proiezione": spd.al_passo(t)}
            r = passo_ottimizzazione(strati_teacher_fn, strati_quant_fn, h0_per_b[b], opt, e0_per_b[b],
                                      h_l_teacher_cache=h_teacher_cache_per_b[b], diagnostica=DIAGNOSTICA_PASSO,
                                      salta_non_finiti=SALTA_NON_FINITI, **_kw_ancora, **_kw_spd)
            if r.get("saltato"):
                _saltati += 1
                log(f"  ⚠️ passo t={t} (batch {b}) SALTATO: {r['saltato']} non finiti ({_saltati} finora)")
                if _saltati > max(3, int(0.05 * n_passi)):
                    raise FloatingPointError(f"{_saltati} passi saltati su {t + 1}: oltre il 5%, mi fermo")
                continue
            if PROFILO and t == 9:
                torch.cuda.synchronize()
                _prof.__exit__(None, None, None)
                log("PROFILO passi 5-9 (per tempo GPU):\n" + _prof.key_averages().table(
                    sort_by="cuda_time_total", row_limit=40))
            perdite.append(r["loss"].item())   # .item() sincronizza: il cronometro sotto e' tempo GPU vero
            if not (perdite[-1] == perdite[-1] and abs(perdite[-1]) != float("inf")):
                # ⛔ 24/9 22:5x: M1 (miscela + fla) ha continuato 100 passi su loss NaN. Ci si
                # ferma al PRIMO passo non finito e si dice dove nasce: loss, gradienti, parametri.
                def _conta(tensori):
                    return sum(int((~torch.isfinite(t)).sum()) for t in tensori if t is not None)
                parametri = [p for g in opt.param_groups for p in g["params"]]
                log(f"⛔ LOSS NON FINITA al passo t={t} (batch {b}): {perdite[-1]} · valori non finiti: "
                    f"gradienti {_conta([p.grad for p in parametri])} · parametri {_conta(parametri)} "
                    f"(parametri aggiornati DOPO questo backward: se sono gia' non finiti, il guasto e' "
                    f"nato a un passo precedente) · loss dei 5 passi prima {perdite[-6:-1]}")
                raise FloatingPointError(f"loss non finita al passo {t} (batch {b})")
            if _nuovi_routing:
                if len(_nuovi_routing) == len(chunk):
                    congelati_per_b[b] = dict(_nuovi_routing)
                del _nuovi_routing
            if CRONO_OGNI and (t + 1) % CRONO_OGNI == 0:
                ora = time.perf_counter()
                log(f"  chunk {chunk}: t={t+1} cronometro {(ora - _t_crono) / CRONO_OGNI:.3f} s/passo "
                    f"(loss {perdite[-1]:.4e})" + ("" if ancora is None else ancora.riga_crono())
                    + ("" if spd is None else spd.riga_crono()))
                _t_crono = ora
            # curva held-out durante l'ottimizzazione (sweep del cancello): mostra
            # se la loss sui dati MAI visti migliora prima di peggiorare
            if HELDOUT_OGNI and batch_ho and (t + 1) % HELDOUT_OGNI == 0:
                log(f"  chunk {chunk}: t={t+1} held-out DURA {_loss_heldout('ternari', congelati_per_b):.4e} "
                    f"(init {ho_init:.4e}, v20 {ho_v20:.4e})")
            if t == 0:
                log(f"  chunk {chunk}: |grad| al passo 0 → {_statistiche_gradiente(gruppi_per_tipo)}")
        # 26/9 sera: SPD si chiude SUBITO dopo l'ultimo passo (distanza e codici cambiati rispetto a θ0, su tutti
        # gli esperti) e libera θ0 prima del ricongelamento e delle valutazioni finali. Si sciolgono anche
        # _kw_spd e spd: terrebbero vivi i parametri del chunk (stessa lezione di _kw_ancora, revisione 26/9).
        spd_misure = None if spd is None else spd.chiudi()
        spd = _kw_spd = None
        # 26/9: il routing finale si congela anche per i lotti di CONOSCENZA (vuoti di default: stessi lotti,
        # stesso ordine di chiamate di prima)
        lotti_valutati = batch_ho + batch_con
        if lotti_valutati and ROUTING_VIVO:
            congelati_finali = congelati_per_b      # router vero dentro il forward: stessa cosa
        if lotti_valutati and not ROUTING_VIVO:
            _pronti = _ricostruisci_pesi_duri(chunk, strati_correnti, ternari_gu, ternari_dw, S)
            congelati_finali = {bb: _congela_routing_chunk(chunk, strati_correnti, hq_per_b[bb], _forward_grezzo,
                                                          ternari_gu=ternari_gu, ternari_dw=ternari_dw, S=S,
                                                          pesi_pronti=_pronti)
                                for bb in lotti_valutati}
            del _pronti
        if batch_ho:
            ho_fine = _loss_heldout("ternari", congelati_finali)
            log(f"  chunk {chunk}: held-out loss DURA finale {ho_fine:.4e} · rispetto a init "
                f"{ho_fine / ho_init:.3f}x · rispetto a v20 {ho_fine / ho_v20:.3f}x "
                f"({'MEGLIO della v20' if ho_fine < ho_v20 else 'PEGGIO della v20'})")
        if batch_con:
            con_fine = _loss_heldout("ternari", congelati_finali, lotti=batch_con)
            log(f"  chunk {chunk}: held-out CONOSCENZA loss DURA finale {con_fine:.4e} · rispetto a init "
                f"{con_fine / con_init:.3f}x · rispetto a v20 {con_fine / con_v20:.3f}x "
                f"({'MEGLIO della v20' if con_fine < con_v20 else 'PEGGIO della v20'})")
        heldout_misure = ({"v20": ho_v20, "init": ho_init, "fine": ho_fine} if batch_ho else None)
        misure_codici = {f"L{L}.{fam}": _confronta_istantanea(istantanee[(L, fam)],
                                                             ternari_gu[L] if fam == "gu" else ternari_dw[L])
                         for (L, fam) in istantanee}
        # 26/9: l'ancora finale sui codici DURI (quelli che si committano), prima del commit
        ancora_misure = None if ancora is None else ancora.chiudi(NUCLEO_BETA_MAX)
        if raccogli_diagnostica is not None:
            _diag = {"chunk": list(chunk), "perdite": perdite, "codici": misure_codici, "heldout": heldout_misure}
            if batch_con:
                _diag["heldout_conoscenza"] = {"v20": con_v20, "init": con_init, "fine": con_fine}
            if ancora_misure is not None:
                _diag["ancora"] = ancora_misure
            if spd_misure is not None:
                _diag["spd"] = spd_misure
            raccogli_diagnostica.append(_diag)
        log(f"  chunk {chunk}: loss {perdite[0]:.4e} → {perdite[-1]:.4e} "
            f"({n_passi} passi: {n_morbida} morbidi + {n_dura} duri/STE)")
        # 26/9: la fotografia di THOR_ANCORA_A=init (0,75 GiB/strato) non sopravvive al chunk. Va sciolto anche
        # _kw_ancora: il gancio dell'ultimo passo chiude su `self` (fotografia e ternari del chunk) e senza questa
        # riga la teneva viva durante il commit qui sotto e, con piu' chunk per finestra, fino al primo passo del
        # chunk dopo (revisione 26/9). Spenta: _kw_ancora e' un dict vuoto, rilegarlo non cambia nulla.
        ancora = _kw_ancora = None
        for chiave, m in misure_codici.items():
            log(f"  chunk {chunk}: {chiave}: codici cambiati {m['frazione_cambiati']*100:.3f}% · "
                f"mediana log(D)/log(s0) {m['mediana_rapporto_log_scala']:.5f} "
                f"(0.90481 = solo decadimento, 1.0 = fermo)")

        # export (§NUCLEO: contratto d'uscita) + COMMIT sui pesi CORRENTI —
        # cosi' il chunk SUCCESSIVO (h0/e0/routing-congelato) riflette il
        # flusso quantizzato AGGIORNATO, non quello di partenza (stesso
        # principio "flusso quantizzato CORRENTE" di finestra_globale.congela_routing).
        for L in chunk:
            risultato[L] = committa_strato(strati_correnti[L], ternari_gu[L], ternari_dw[L], S)
            if consegna_stato_per_strato is not None and L in consegna_stato_per_strato:
                from ripartenza_codici import estrai_stato_strato
                consegna_stato_per_strato[L] = estrai_stato_strato(
                    ternari_gu[L], ternari_dw[L], opt=(opt if adam_ripartenza == "conserva" else None),
                    codici_rif=risultato[L], origine=f"finestra {layers}")
        # 25/9 (cache per lotto): i depositi del chunk si chiudono qui (i file restano finche' un
        # altro deposito, il seme, li nomina); sui dict e' un no-op.
        DL.chiudi(h0_per_b, e0_per_b, hq_per_b, h_teacher_cache_per_b)

    return risultato


# ═════════════════ 6. ESPORTAZIONE — SOLO la finestra 8-12 ══════════════════
def scrivi_codici_su_disco(risultato: dict) -> None:
    """Salva D/Q per OGNI strato della finestra, STESSO naming di
    `forgia_schur_esperti.py` (Dg_L/Qg_L gate, Du_L/Qu_L up) cosi'
    l'esportatore sotto e quello grande restano intercambiabili. Marca
    `fatto_forgia_{L:02d}.ok` scritta SOLO a salvataggio completo (ripresa —
    stesso pattern del resto della pipeline).

    ⭐ 24/9 08:5x — CHIUSO il gap dichiarato (mai accettato in silenzio,
    PIANO.md 08:06/08:16): ora salva ANCHE Dd/Qd (`ffn_down_exps`) — la
    conversione Q4_K→ternario che `ottimizza_finestra_JOINT` calcola da
    sempre (dw_reale_finale) ma che finora restava solo in memoria, mai su
    disco."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for L, per_fam in risultato.items():
        Dg, Qg = per_fam["ffn_gate_exps"]
        Du, Qu = per_fam["ffn_up_exps"]
        np.save(OUT_DIR / f"Dg_{L:02d}.npy", Dg)
        np.save(OUT_DIR / f"Qg_{L:02d}.npy", Qg)
        np.save(OUT_DIR / f"Du_{L:02d}.npy", Du)
        np.save(OUT_DIR / f"Qu_{L:02d}.npy", Qu)
        down = "ffn_down_exps" in per_fam
        if down:
            Dd, Qd = per_fam["ffn_down_exps"]
            np.save(OUT_DIR / f"Dd_{L:02d}.npy", Dd)
            np.save(OUT_DIR / f"Qd_{L:02d}.npy", Qd)
        # audit B5/B10: la marca dichiara il modo del down (l'export lo esige)
        (OUT_DIR / f"fatto_forgia_{L:02d}.ok").write_text(
            f"{time.strftime('%F %T')} corsa={ID_CORSA} down_ternario={int(down)}\n")
    log(f"codici scritti in {OUT_DIR} per gli strati {sorted(risultato)}")


def esporta_gguf_finestra(layers: "list[int] | None" = None) -> Path:
    """Esporta un GGUF di prova patchando SOLO `blk.{L}.ffn_{gate,up}_exps.weight`
    per L in `layers` (default `FINESTRA`, comportamento INVARIATO) — MAI
    cumulativo da 0 come `esporta_schur_esperti.py`: qui lo scopo e' isolare
    l'effetto della finestra, gli altri strati restano ESATTAMENTE come sono
    in `THOR_SCHUR_BASE`.

    ⭐ 24/9: `layers` esplicito per lo stesso motivo di `carica_finestra_
    corrente` (PIANO.md 08:26) — il loop a scorrimento sui 40 strati veri
    scrive i .npy per-strato UNA finestra alla volta (`scrivi_codici_su_
    disco`, gia' generico), e chiama QUESTA funzione UNA sola volta alla
    fine con `layers=list(range(40))`: legge la base pristina UNA volta
    sola e patcha tutti gli strati insieme, evitando la complessita' di un
    export incrementale per-finestra (che dovrebbe leggere l'output della
    finestra precedente invece della base fissa — non necessario con
    questo schema).

    Riusa (import, sola lettura) `impacchetta_famiglia`/`verifica_packer`/
    `TIPO_BASE_ATTESO`/`TIPO_NUOVO` da `esporta_schur_esperti.py` — stesso
    packer gia' verificato bit-esatto, zero duplicazione della logica di
    basso livello; la sola differenza e' l'INSIEME di strati iterato."""
    from esporta_schur_esperti import impacchetta_famiglia, verifica_packer  # noqa: E402  (sola lettura)

    layers = FINESTRA if layers is None else list(layers)
    mancanti = [L for L in layers if not (OUT_DIR / f"fatto_forgia_{L:02d}.ok").exists()]
    if mancanti:
        raise RuntimeError(f"CORSA INCOMPLETA: mancano le marche per gli strati {mancanti} in {OUT_DIR}")
    # ⛔ audit B5/B10: le marche dichiarano corsa e modo del down. Rifiuta strati
    # di corse diverse (la cartella di stamattina era mista fra due corse) e,
    # piu' sotto, un down dichiarato ternario ma assente su disco.
    import re as _re
    marche = {L: (OUT_DIR / f"fatto_forgia_{L:02d}.ok").read_text() for L in layers}
    corse = {L: (_re.search(r"corsa=(\S+)", t) or [None, None])[1] for L, t in marche.items()}
    down_dich = {L: (_re.search(r"down_ternario=(\d)", t) or [None, None])[1] for L, t in marche.items()}
    if None in corse.values() or len(set(corse.values())) != 1:
        raise RuntimeError(f"PROVENIENZA: strati di corse diverse o marche senza id di corsa: {corse}")
    log(f"export: tutti i {len(layers)} strati dalla corsa {next(iter(corse.values()))}")

    src = GGUFReader(str(GGUF_BASE))
    tens_by_name = {t.name: t for t in src.tensors}

    # ⭐ 24/9 08:5x — down_proj (→ternario standard, TQ1_0 non ruotato):
    # CHIUSO come gap dichiarato, ma OPZIONALE per-strato — se una corsa e'
    # partita PRIMA che scrivi_codici_su_disco salvasse Dd/Qd (il processo
    # aveva gia' in memoria la versione vecchia: un modulo Python gia'
    # importato non rilegge le modifiche al file, lezione gia' pagata
    # stanotte), quello strato resta col down_proj originale — LOGGATO
    # esplicitamente, mai un salto silenzioso.
    #
    # ⚠️ 24/9 09:0x — CORRETTO dopo misura reale: il tipo base di down_proj
    # NON e' uniforme fra strati (Thor e' forgiato incrementalmente da mesi:
    # strato 9 = Q4_K/12, strato 12 = IQ4_XS/23 — misurato, non assunto).
    # Qualunque formato quantizzato pre-ternario va bene da patchare: l'UNICO
    # tipo che deve fermare tutto e' TQ1_0R_1024 (45, il codec RUOTATO) —
    # down_proj non va MAI ruotato ("down_proj: NIENTE rotazione",
    # _stratofn_quant_reale sopra), trovarlo li' vorrebbe dire un errore a
    # monte, non un formato legittimo da sovrascrivere in silenzio.
    from pack_tq1_0_standard import pack_tq1_0_da_d_q  # noqa: E402  (sola lettura)
    TIPO_VIETATO_DOWN = int(T.TQ1_0R_1024)
    TIPO_NUOVO_DOWN = int(T.TQ1_0)

    bersagli: dict[str, tuple] = {}  # nome -> (bytes_impacchettati, raw_dtype)
    verificato = False
    verificato_down = False
    strati_senza_down: list[int] = []
    for L in layers:
        Dg = np.load(OUT_DIR / f"Dg_{L:02d}.npy")
        Qg = np.load(OUT_DIR / f"Qg_{L:02d}.npy")
        Du = np.load(OUT_DIR / f"Du_{L:02d}.npy")
        Qu = np.load(OUT_DIR / f"Qu_{L:02d}.npy")
        for fam, D, Q in (("ffn_gate_exps", Dg, Qg), ("ffn_up_exps", Du, Qu)):
            nome = f"blk.{L}.{fam}.weight"
            t_base = tens_by_name.get(nome)
            if t_base is None:
                raise RuntimeError(f"{nome}: NON esiste in {GGUF_BASE} — schema divergente")
            if int(t_base.tensor_type) != TIPO_BASE_ATTESO:
                raise RuntimeError(
                    f"{nome}: tipo base {int(t_base.tensor_type)} != atteso {TIPO_BASE_ATTESO} "
                    "(TQ1_0) — verifica manuale richiesta prima di sovrascrivere")
            if not verificato:
                verifica_packer(D, Q, nome)
                verificato = True
            bersagli[nome] = (impacchetta_famiglia(D, Q), T.TQ1_0R_1024)

        path_dd = OUT_DIR / f"Dd_{L:02d}.npy"
        path_qd = OUT_DIR / f"Qd_{L:02d}.npy"
        if path_dd.exists() and path_qd.exists():
            Dd, Qd = np.load(path_dd), np.load(path_qd)
            nome_d = f"blk.{L}.ffn_down_exps.weight"
            t_base_d = tens_by_name.get(nome_d)
            if t_base_d is None:
                raise RuntimeError(f"{nome_d}: NON esiste in {GGUF_BASE} — schema divergente")
            tipo_base_d = int(t_base_d.tensor_type)
            if tipo_base_d == TIPO_VIETATO_DOWN:
                raise RuntimeError(
                    f"{nome_d}: tipo base e' TQ1_0R_1024 (45, RUOTATO) — down_proj non va MAI "
                    "ruotato, questo indica un errore a monte, non un formato da sovrascrivere")
            if tipo_base_d != TIPO_NUOVO_DOWN:
                log(f"  strato {L}: down_proj base tipo {tipo_base_d} (non ancora ternario) → TQ1_0")
            if not verificato_down:
                from gguf.quants import TQ1_0 as _TQ1_0_STD
                packed_prova = pack_tq1_0_da_d_q(Dd, Qd)
                dequant = _TQ1_0_STD.dequantize_blocks(packed_prova.reshape(-1, packed_prova.shape[-1]))
                atteso = (Qd.astype(np.float32) * Dd.repeat(256, axis=-1)).reshape(-1, 256)
                scarto = np.abs(dequant - atteso).max()
                assert scarto < 1e-3, f"{nome_d}: round-trip down_proj rotto, scarto={scarto:.3e}"
                verificato_down = True
            # ⛔ 24/9 13:3x — BUG TROVATO E CHIUSO: pack_tq1_0_da_d_q ritorna
            # (E, out, ng, BLOCK_BYTES) — 4 assi. GGUFWriter (raw_dtype=uint8,
            # come impacchetta_famiglia fa per gate/up) vuole SOLO 3 assi
            # (E, out, ng*BLOCK_BYTES) — stesso collasso finale che
            # impacchetta_famiglia già fa (esporta_schur_esperti.py:94).
            # Senza questo reshape llama-server rifiutava il modello:
            # "tensor 'blk.0.ffn_down_exps.weight' has wrong shape;
            # expected 512,2048,256, got 256,2,2048,256" — scoperto SOLO ora
            # perché il controllo bit-esatto di riga 1233 usa reshape(-1, ultimo
            # asse), che è indifferente al numero di assi e non l'avrebbe mai
            # rivelato.
            packed_down = pack_tq1_0_da_d_q(Dd, Qd)
            E_d, out_d = Dd.shape[0], Dd.shape[1]
            packed_down = packed_down.reshape(E_d, out_d, -1)
            bersagli[nome_d] = (packed_down, T.TQ1_0)
            log(f"  strato {L}: impacchettato (gate+up+down)")
        else:
            if down_dich[L] != "0":
                raise RuntimeError(f"strato {L}: la marca dichiara down_ternario={down_dich[L]} ma Dd/Qd "
                                   f"mancano in {OUT_DIR} — rifiuto di esportare un down a 4 bit in silenzio")
            strati_senza_down.append(L)
            log(f"  strato {L}: impacchettato (gate+up) — down_proj SALTATO "
                f"(Dd/Qd assenti su disco, resta Q4_K originale)")
    if strati_senza_down:
        log(f"⚠️ down_proj NON convertito per {len(strati_senza_down)} strati "
            f"{strati_senza_down} — corsa partita prima del fix di stanotte")
    assert len(bersagli) == len(layers) * 2 + (len(layers) - len(strati_senza_down))

    w = GGUFWriter(path=None, arch="qwen35moe")
    for fld in src.fields.values():
        if fld.name.startswith(("GGUF.", "split.")):
            continue
        try:
            w.add_key_value(fld.name, fld.contents(), fld.types[0],
                             sub_type=fld.types[-1] if len(fld.types) > 1 else None)
        except Exception:
            pass
    # audit B10: metadati VERI (prima dicevano "finestra 8-12, 0-7 e 13-39 invariati"
    # anche per un export a 40 strati)
    w.add_string("thor.ricetta_globale", json.dumps({
        "corsa": next(iter(corse.values())), "strati": list(layers),
        "strati_con_down_ternario": [L for L in layers if L not in strati_senza_down],
        "base": str(GGUF_BASE), "W": os.getenv("THOR_FORGIA_W"), "n_passi": N_PASSI,
        "lr": NUCLEO_LR, "weight_decay": os.getenv("THOR_WEIGHT_DECAY", "0.0"),
        "init_da": os.getenv("THOR_INIT_DA", "teacher"), "init_ls": os.getenv("THOR_INIT_LS", "1"),
        "n_batch": N_BATCH, "n_heldout": os.getenv("THOR_VALIDA_N_HELDOUT", "4"),
        "calib_seed": os.getenv("THOR_CALIB_SEED", "1234"), "loss": "somma (B1)",
    }, ensure_ascii=False))

    piano = [(t.name, "patch" if t.name in bersagli else "copia", t) for t in src.tensors]
    for nome, azione, t in piano:
        if azione == "patch":
            packed, raw_dtype = bersagli[nome]
            w.add_tensor_info(nome, list(packed.shape), np.dtype(np.uint8), int(packed.nbytes),
                               raw_dtype=raw_dtype)
        else:
            raw = np.asarray(t.data)
            w.add_tensor_info(nome, list(raw.shape), raw.dtype, int(raw.nbytes), raw_dtype=t.tensor_type)

    GGUF_OUT.parent.mkdir(parents=True, exist_ok=True)
    if GGUF_OUT.exists() and os.getenv("THOR_EXPORT_SOVRASCRIVI", "0") != "1":
        raise RuntimeError(f"{GGUF_OUT} esiste gia': rifiuto di sovrascriverlo (audit B10). "
                           "Scegli un altro THOR_VALIDA_GGUF_OUT o THOR_EXPORT_SOVRASCRIVI=1")
    w.open_output_file(GGUF_OUT)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    t0 = time.time()
    for nome, azione, t in piano:
        w.write_tensor_data(bersagli[nome][0] if azione == "patch" else np.asarray(t.data))
    w.close()
    log(f"FATTO in {time.time()-t0:.0f}s: {GGUF_OUT} ({GGUF_OUT.stat().st_size/2**30:.2f} GiB) — "
        f"{len(bersagli)} tensori patchati, {len(piano)-len(bersagli)} copiati invariati")
    _verifica_rilettura(GGUF_OUT, bersagli)
    return GGUF_OUT


def _verifica_rilettura(percorso: Path, bersagli: dict) -> None:
    """audit B17: i controlli d'export erano tautologici (packer contro se stesso).
    Rilegge il GGUF SCRITTO e confronta, per OGNI tensore patchato, tipo, forma in
    byte e contenuto con cio' che si intendeva scrivere; per i tipi ternari
    controlla anche che le scale fp16 siano finite."""
    r = GGUFReader(str(percorso))
    letti = {t.name: t for t in r.tensors}
    for nome, (packed, raw_dtype) in bersagli.items():
        t = letti.get(nome)
        if t is None:
            raise RuntimeError(f"rilettura: {nome} assente nel GGUF scritto")
        if int(t.tensor_type) != int(raw_dtype):
            raise RuntimeError(f"rilettura: {nome} tipo {int(t.tensor_type)} != {int(raw_dtype)}")
        dati = np.asarray(t.data)
        if dati.nbytes != packed.nbytes or not np.array_equal(dati.reshape(-1).view(np.uint8),
                                                              np.ascontiguousarray(packed).reshape(-1).view(np.uint8)):
            raise RuntimeError(f"rilettura: {nome} contenuto diverso da quello impacchettato")
    log(f"rilettura: {len(bersagli)} tensori patchati identici al byte nel GGUF scritto")


# ═══════════════════════ 7. SONDA — le STESSE 15 domande ════════════════════
def _chiedi(porta: int, domanda: str, attesa: str, seed: int) -> str:
    """Una domanda, un seed — STESSA formula (endpoint, parametri, regex) del
    banco che ha prodotto il numero di stanotte. Ritorna 'OK'|'SBAGLIA'|
    'MUTO'|'GUASTO'. Reimplementato qui in Python (non un `curl` in bash)
    perche' lo script bash che ha girato stanotte vive in uno scratchpad di
    SESSIONE (fuori da questo repo) — non e' un riferimento su cui questo
    script possa contare in futuro."""
    corpo = json.dumps({
        "model": "s", "stream": False, "max_tokens": 6000, "temperature": 0.6, "seed": seed,
        "messages": [{"role": "user", "content": domanda}],
        "chat_template_kwargs": {"enable_thinking": True},
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{porta}/v1/chat/completions", data=corpo,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.loads(r.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        return f"GUASTO({e})"
    msg = (d.get("choices") or [{}])[0].get("message", {}) or {}
    contenuto = (msg.get("content") or "").strip()
    ragionamento = (msg.get("reasoning_content") or "").strip()
    if not contenuto:
        # audit B8: MUTO solo se il modello non ha prodotto NULLA; se ha ragionato
        # e ha finito i token prima di rispondere e' TRONCATO, non muto.
        return "TRONCATO" if ragionamento else "MUTO"
    # audit B8: si giudica l'ULTIMO numero/parola della risposta, non "un 3 da
    # qualche parte" (\b3\b dava OK a "ha 2 sorelle, cioe' 3 in tutto... no, 2")
    if attesa.startswith("NUM:"):
        numeri = re.findall(r"-?\d+(?:[.,]\d+)?", contenuto)
        if not numeri:
            return "SBAGLIA"
        try:
            return "OK" if abs(float(numeri[-1].replace(",", ".")) - float(attesa[4:])) < 1e-6 else "SBAGLIA"
        except ValueError:
            return "SBAGLIA"
    return "OK" if re.search(attesa, contenuto, re.I) else "SBAGLIA"


def lancia_sonda(gguf_path: Path) -> dict:
    """Avvia il modello di prova (llama-server K1024, richiesto per
    TQ1_0R_1024), pone le 15 domande (3x5 seed), lo ferma SEMPRE (try/finally
    — un server orfano su una porta adhoc-* confonderebbe l'arbitro VRAM del
    resto della pipeline). Ritorna {"ok":N,"muto":N,"sbaglia":N,"guasto":N}."""
    verifica_gpu_libera()  # secondo checkpoint: la finestra caricata prima e' gia' stata liberata?

    # ⛔ 24/9 16:1x — fuori da una sessione interattiva `systemctl/systemd-run
    # --user` falliscono (exit 1) senza queste due variabili, e l'esito era
    # IGNORATO: la sonda aspettava 7 minuti un server mai avviato. Ora le
    # imposta da se' se mancano (valgono anche per stop/reset-failed nel
    # finally) e fallisce SUBITO se systemd-run non parte.
    os.environ.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    os.environ.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path=/run/user/{os.getuid()}/bus")

    subprocess.run(["systemctl", "--user", "stop", UNITA_SONDA], capture_output=True)
    time.sleep(1)
    # audit B8: STESSI argomenti di sonda_parlata_k1024.sh (lo script ufficiale):
    # prima mancavano --jinja, --reasoning-budget e il campionamento.
    cmd = ["systemd-run", "--user", f"--unit={UNITA_SONDA}", "--collect",
           str(LLAMA_BIN_K1024), "--model", str(gguf_path), "--alias", "sonda", "-ngl", "999",
           "--ctx-size", "8192", "--parallel", "1", "--flash-attn", "on",
           "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
           "--host", "127.0.0.1", "--port", str(PORTA_SONDA), "--threads", "8",
           "--temp", "0.6", "--min-p", "0.03", "--top-p", "0.9", "--jinja", "--reasoning-budget", "4096"]
    env = dict(os.environ, LD_LIBRARY_PATH=str(LLAMA_BIN_K1024.parent))
    avvio = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if avvio.returncode != 0:
        raise RuntimeError(f"systemd-run non ha avviato il server di prova (exit {avvio.returncode}): "
                           f"{avvio.stderr.strip()[:400]}")

    try:
        for _ in range(90):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORTA_SONDA}/health", timeout=2) as r:
                    if json.loads(r.read()).get("status") == "ok":
                        break
            except Exception:
                pass
            time.sleep(5)
        else:
            raise RuntimeError("il modello di prova non si accende in 7 minuti")
        log(f"sonda: server acceso su :{PORTA_SONDA} — {gguf_path.name}")

        conteggio = {"OK": 0, "MUTO": 0, "TRONCATO": 0, "SBAGLIA": 0, "GUASTO": 0}
        for domanda, attesa in DOMANDE_SONDA:
            for seed in SEMI_SONDA:
                esito = _chiedi(PORTA_SONDA, domanda, attesa, seed)
                chiave = esito if esito in conteggio else "GUASTO"
                conteggio[chiave] += 1
                log(f"  seed={seed:<3} {esito:<8} · {domanda[:40]}...")
        n_tot = len(DOMANDE_SONDA) * len(SEMI_SONDA)
        log(f"SONDA: ok={conteggio['OK']}/{n_tot} muto={conteggio['MUTO']} troncato={conteggio['TRONCATO']} "
            f"sbaglia={conteggio['SBAGLIA']} guasto={conteggio['GUASTO']}")
        return conteggio
    finally:
        subprocess.run(["systemctl", "--user", "stop", UNITA_SONDA], capture_output=True)
        subprocess.run(["systemctl", "--user", "reset-failed", UNITA_SONDA], capture_output=True)


# ═════════════════════ 8. VERDETTO — criterio di successo ═══════════════════
def giudica(ok: int, muto: int) -> str:
    """A 4 esiti, nessuna zona grigia — §CRITERIO DI SUCCESSO nel docstring
    di modulo. Confronta contro BASELINE_OK/BASELINE_MUTO (misurati stanotte,
    depth-sweep 0-10) e OBIETTIVO_OK_MIN/OBIETTIVO_MUTO_MAX (livello pulito
    pre-transizione, N<=10 nello stesso depth-sweep)."""
    if muto <= OBIETTIVO_MUTO_MAX and ok >= OBIETTIVO_OK_MIN:
        return (f"✅ SUCCESSO PIENO — ok={ok}/15 muto={muto} (obiettivo ok>={OBIETTIVO_OK_MIN} "
                f"muto<={OBIETTIVO_MUTO_MAX}): la finestra torna al livello 'pulito' pre-transizione")
    if muto < BASELINE_MUTO or (muto == BASELINE_MUTO and ok > BASELINE_OK):
        return (f"🟡 MIGLIORAMENTO PARZIALE — ok={ok}/15 muto={muto} "
                f"(baseline ok={BASELINE_OK}/15 muto={BASELINE_MUTO}): meglio della baseline, "
                "non ancora al livello pulito")
    if muto == BASELINE_MUTO and ok == BASELINE_OK:
        return f"⚪ NESSUN MIGLIORAMENTO — ok={ok}/15 muto={muto}: identico alla baseline di stanotte"
    return (f"⛔ REGRESSIONE — ok={ok}/15 muto={muto} (baseline ok={BASELINE_OK}/15 muto={BASELINE_MUTO}): "
            "peggio del metodo attuale")


def genera_baseline_vecchio_metodo_stessa_finestra() -> None:
    """PREDISPOSTA, NON eseguita di default (costerebbe GPU extra non chiesta
    da questo compito). Se un giorno si vuole il confronto piu' rigoroso
    (finestra isolata anche per il metodo VECCHIO, non solo cumulativo come
    stanotte): rilanciare `forgia_schur_esperti.py`/`esporta_schur_esperti.py`
    con `THOR_STRATI="{L0}-{L1}"` in cattura + gli stessi Dg/Qg/Du/Qu prodotti
    per QUESTA finestra dal metodo sequenziale, poi `esporta_gguf_finestra()`
    di questo file (che gia' patcha solo la finestra, indipendentemente da chi
    ha calcolato i codici) e `lancia_sonda()`. Non chiamata da `main()`."""
    raise NotImplementedError(
        "baseline dedicata non generata: richiede una corsa GPU aggiuntiva "
        "(cattura+forgia sequenziale SOLO su strati "
        f"{FINESTRA_L0}-{FINESTRA_L1}) non richiesta da questo compito. "
        "Il confronto di default usa la baseline gia' misurata stanotte "
        "(depth-sweep 0-10, vedi BASELINE_OK/BASELINE_MUTO).")


# ═════ 9. TEST GIOCATTOLO end-to-end di ottimizza_finestra_JOINT (24/9) ═════
# Dimostra che il file gira dall'inizio alla fine SENZA NotImplementedError —
# con STRATI REALI (Qwen3_5MoeDecoderLayer, la STESSA classe che
# carica_finestra_corrente userebbe su Thor), a dimensioni GIOCATTOLO (hidden=8,
# intermedio=6, 4 esperti, top_k=2 — la stessa scala gia' usata in
# finestra_globale.py/quant_ternaria_globale.py). Zero GPU: gira su CPU,
# nessuna chiamata a rocm-smi/verifica_gpu_libera (il vincolo di sessione
# "mai Thor" e' rispettato per costruzione, non serve guardia GPU per un
# modello che pesa poche migliaia di parametri).
def _costruisci_config_giocattolo(n_layers: int):
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    return Qwen3_5MoeTextConfig(
        vocab_size=32, hidden_size=8, num_hidden_layers=n_layers,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4,
        moe_intermediate_size=6, shared_expert_intermediate_size=6,
        num_experts=4, num_experts_per_tok=2,
        layer_types=["full_attention"] * n_layers,
        partial_rotary_factor=1.0,
    )


def _init_pesi_giocattolo(strato: torch.nn.Module, seed: int) -> None:
    """⛔ Difetto trovato SCRIVENDO questo test (24/9): `Qwen3_5MoeExperts.
    __init__` alloca gate_up_proj/down_proj con `torch.empty()` — memoria NON
    inizializzata (garbage, spesso NaN/Inf) finche' `_init_weights`/
    `from_pretrained` non la sovrascrivono; costruendo il layer "a mano" (come
    qui, niente checkpoint) quel passo non avviene mai. Il primo forward, ANCHE
    SENZA alcun monkeypatch di ottimizza_finestra_JOINT, usciva gia' NaN — non
    un bug del nucleo, un difetto del setup del test. Le RMSNorm restano a
    zero (init REALE di Qwen3.5-MoE: `1+0=identita'`, vedi `_init_weights`
    nel modello — MAI toccarle)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for nome, p in strato.named_parameters():
            if "norm" in nome:
                continue
            p.copy_(torch.randn(p.shape, generator=g) * 0.02)


def _test_ottimizza_finestra_JOINT_giocattolo(dimensione_sottofinestra: "int | None" = None) -> bool:
    """Un layer teacher (`puliti`) + uno "gia' forgiato" (`quantizzati` =
    teacher + rumore esplicito, MAI identico al teacher — altrimenti zero
    margine per l'AdamW, stessa lezione di `_perturba_z` in
    finestra_globale.py — e MAI da zero, come il compito vieta) per 5 strati
    finti. Verifica: (a) nessun NotImplementedError, il file gira fino in
    fondo; (b) il dizionario ritornato ha, per OGNI strato, le tre famiglie
    del contratto con forma/dtype/valori coerenti (Q ternario {-1,0,1} int8,
    D float32); (c) la loss Lemma-1 SCENDE in ogni chunk (`raccogli_diagnostica`,
    il gancio di sola-diagnostica del §NUCLEO); (d) il numero di chunk
    prodotti combacia con `dimensione_sottofinestra` (verifica ANCHE il
    percorso "sotto-finestre" ammesso dal compito, non solo il default)."""
    print(f"\n── test ottimizza_finestra_JOINT (giocattolo, sottofinestra="
          f"{dimensione_sottofinestra or 'nessuna (finestra intera)'}) ──")
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer

    n_layer_finti = 5
    layers = list(range(n_layer_finti))
    cfg = _costruisci_config_giocattolo(n_layer_finti)
    cfg._attn_implementation = "sdpa"

    puliti, quantizzati = {}, {}
    for L in layers:
        pulito = Qwen3_5MoeDecoderLayer(cfg, layer_idx=L)
        _init_pesi_giocattolo(pulito, seed=100 + L)
        puliti[L] = pulito.eval()
        corrente = Qwen3_5MoeDecoderLayer(cfg, layer_idx=L)
        with torch.no_grad():
            g = torch.Generator().manual_seed(900 + L)
            for (n_p, p_p), (n_q, p_q) in zip(pulito.named_parameters(), corrente.named_parameters()):
                assert n_p == n_q, "stessa architettura, stesso ordine di parametri"
                p_q.copy_(p_p + torch.randn(p_p.shape, generator=g) * 0.15)
        quantizzati[L] = corrente.eval()

    emb = torch.randn(cfg.vocab_size, cfg.hidden_size) * 0.02
    n_batch, batch_seq, l_seq = 2, 2, 6
    ids = torch.randint(0, cfg.vocab_size, (n_batch, batch_seq, l_seq))

    diagnostica: list = []
    risultato = ottimizza_finestra_JOINT(
        quantizzati, ids, cfg, n_passi=40, strati_teacher=puliti, emb=emb,
        dimensione_sottofinestra=dimensione_sottofinestra, raccogli_diagnostica=diagnostica)

    forma_ok = True
    for L in layers:
        per_fam = risultato.get(L)
        if per_fam is None:
            print(f"    ⛔ strato {L} assente dal risultato")
            forma_ok = False
            continue
        for fam in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
            if fam not in per_fam:
                print(f"    ⛔ strato {L}: manca {fam}")
                forma_ok = False
                continue
            D, Q = per_fam[fam]
            valori_ok = set(np.unique(Q).tolist()) <= {-1, 0, 1}
            forme_coerenti = (D.shape[:2] == Q.shape[:2] and D.dtype == np.float32 and Q.dtype == np.int8)
            if not (valori_ok and forme_coerenti):
                print(f"    ⛔ strato {L}/{fam}: forma/dtype/valori sospetti D={D.shape}/{D.dtype} "
                      f"Q={Q.shape}/{Q.dtype} valori={sorted(set(np.unique(Q).tolist()))}")
                forma_ok = False
    print(f"    forma/dtype/valori del risultato per TUTTI gli strati e le 3 famiglie: "
          f"{'✅' if forma_ok else '⛔'}")

    loss_scende = bool(diagnostica) and all(d["perdite"][-1] < d["perdite"][0] for d in diagnostica)
    for d in diagnostica:
        print(f"    chunk {d['chunk']}: loss {d['perdite'][0]:.4e} → {d['perdite'][-1]:.4e}")
    print(f"    loss Lemma-1 scende in OGNI chunk: {'✅' if loss_scende else '⛔'}")

    n_chunk_atteso = 1 if not dimensione_sottofinestra else -(-len(layers) // dimensione_sottofinestra)
    chunk_ok = len(diagnostica) == n_chunk_atteso
    print(f"    numero di chunk atteso={n_chunk_atteso} osservato={len(diagnostica)}: "
          f"{'✅' if chunk_ok else '⛔'}")

    esito = forma_ok and loss_scende and chunk_ok
    print(f"    {'✅ TEST PASSA' if esito else '⛔ TEST FALLISCE'} "
          "(nessun NotImplementedError sollevato: il file gira dall'inizio alla fine)")
    return esito


def _test_giocattolo() -> bool:
    r1 = _test_ottimizza_finestra_JOINT_giocattolo(dimensione_sottofinestra=None)
    r2 = _test_ottimizza_finestra_JOINT_giocattolo(dimensione_sottofinestra=4)  # come "sotto-finestre di 4"
    ok = r1 and r2
    print(f"\n{'✅ TUTTI I TEST GIOCATTOLO PASSANO' if ok else '⛔ ALMENO UN TEST GIOCATTOLO FALLISCE'} "
          f"(finestra_intera={r1}, sottofinestra_4={r2})")
    return ok


# ═══════════════════════════════════ MAIN ═══════════════════════════════════
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--solo-struttura", action="store_true",
                     help="verifica precondizioni/percorsi/import, NIENTE GPU — smoke test della struttura")
    ap.add_argument("--salta-sonda", action="store_true",
                     help="ferma dopo l'esportazione, non accende il modello di prova")
    ap.add_argument("--test-giocattolo", action="store_true",
                     help="collaudo end-to-end di ottimizza_finestra_JOINT a dimensioni GIOCATTOLO "
                          "(mai Thor, CPU, nessuna precondizione/GPU toccata)")
    args = ap.parse_args()

    if args.test_giocattolo:
        log("--test-giocattolo: nucleo di ottimizzazione globale su strati GIOCATTOLO — zero GPU/Thor")
        return 0 if _test_giocattolo() else 1

    log(f"═══ VALIDAZIONE finestra {FINESTRA_L0}-{FINESTRA_L1} — ottimizzazione globale cross-strato ═══")
    precondizioni()
    if FINESTRA_L0 != 0 and not args.solo_struttura:
        # audit B16: questo main carica SOLO la FINESTRA e ne ottimizza il primo
        # strato partendo dall'EMBEDDING: giusto solo se FINESTRA_L0 == 0 (e' la
        # trappola del bug del confine). Per finestre interne si usa il ciclo
        # esterno forgia_globale_40_strati.py, che passa il seme vero.
        raise SystemExit(f"FINESTRA_L0={FINESTRA_L0} != 0: usa forgia_globale_40_strati.py "
                         "(questo main partirebbe dall'embedding, seme sbagliato)")

    if args.solo_struttura:
        log("--solo-struttura: verifico SOLO import/contratti, nessuna GPU toccata")
        assert callable(ottimizza_finestra_JOINT)
        assert callable(esporta_gguf_finestra)
        assert callable(lancia_sonda)
        assert callable(giudica)
        log(f"baseline stanotte: ok={BASELINE_OK}/15 muto={BASELINE_MUTO} · "
            f"obiettivo: ok>={OBIETTIVO_OK_MIN} muto<={OBIETTIVO_MUTO_MAX}")
        log("✅ struttura OK — pronta per il nucleo di ottimizzazione")
        return 0

    verifica_gpu_libera()
    ids = carica_poche_batch()

    cfg = emb = puliti = quantizzati = None
    try:
        cfg, emb, puliti, quantizzati = carica_finestra_corrente()
        log("── diagnostica di OGGI (quanto e' rotta la finestra ORA, prima di ottimizzare) ──")
        misura_diagnostica_oggi(puliti, quantizzati, ids, cfg, emb)

        log("── nucleo di ottimizzazione globale ──")
        # ⭐ 24/9: strati_teacher/emb AGGIUNTI alla chiamata — correzione
        # necessaria trovata scrivendo il nucleo (vedi docstring di
        # ottimizza_finestra_JOINT): senza il ramo teacher e la tabella di
        # embedding il kernel Lemma-1 non potrebbe calcolare F_l(h_l) ne'
        # materializzare h0 da ids_batch.
        risultato = ottimizza_finestra_JOINT(quantizzati, ids, cfg, N_PASSI,
                                              strati_teacher=puliti, emb=emb)
    except NotImplementedError as e:
        log(f"⛔ NUCLEO NON ANCORA CONSEGNATO: {e}")
        log("STRUTTURA VERIFICATA fino al punto di aggancio — nessun bug qui, manca solo il nucleo")
        return 10
    finally:
        libera_finestra(puliti, quantizzati, emb)

    scrivi_codici_su_disco(risultato)
    gguf_path = esporta_gguf_finestra()

    if args.salta_sonda:
        log("--salta-sonda: fermo dopo l'esportazione")
        return 0

    conteggio = lancia_sonda(gguf_path)
    verdetto = giudica(conteggio["OK"], conteggio["MUTO"])
    log(verdetto)
    riepilogo = {
        "finestra": [FINESTRA_L0, FINESTRA_L1], "conteggio": conteggio,
        "baseline": {"ok": BASELINE_OK, "muto": BASELINE_MUTO},
        "obiettivo": {"ok_min": OBIETTIVO_OK_MIN, "muto_max": OBIETTIVO_MUTO_MAX},
        "verdetto": verdetto, "gguf": str(gguf_path), "quando": time.strftime("%F %T"),
    }
    (OUT_DIR / "verdetto.json").write_text(json.dumps(riepilogo, ensure_ascii=False, indent=2))
    log(f"riepilogo salvato in {OUT_DIR / 'verdetto.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

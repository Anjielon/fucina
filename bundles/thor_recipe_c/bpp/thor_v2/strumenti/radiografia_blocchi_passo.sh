#!/bin/bash
# 25/9 — radiografia PER BLOCCO del passo vero (strumenti/radiografia_blocchi_passo.py): ricetta S4 (separate, stesse
# variabili di ricetta_base_nuova_6h.sh comune() + braccio B1: base densi v5, calibrazione M2, 32 batch x 8 x 1024),
# forgia_globale sui primi 4 strati = finestre [0,1] (lineare+lineare) e [2,3] (piena allo strato 3), 48 passi per
# finestra: 10 cronometrati senza profiler, 30 profilati. NESSUN codice scritto, nessun export.
# SOLO a GPU libera: guardia strumenti/gpu_occupata.sh --attendi (e di nuovo dentro il python, prima di toccare la GPU).
set -uo pipefail
cd @@ROOT:forge@@/thor_v2
unset HSA_OVERRIDE_GFX_VERSION
export PYTHONPATH=@@ROOT:forge@@/thor_v2/strumenti/pylib_fla
D=@@ROOT:forge@@/stato/velocita
T=$(date +%Y%m%d-%H%M%S)
if ! occ=$(strumenti/gpu_occupata.sh --attendi 120); then
  echo "[$(date +%T)] ⛔ radiografia per blocco NON partita: GPU occupata: $occ" >> "$D/riepilogo.log"; exit 3
fi
echo "[$(date +%T)] inizio radiografia per blocco (S4, finestre [0,1] e [2,3], 48 passi di cui 30 profilati)" >> "$D/riepilogo.log"
env THOR_SCHUR_BASE=@@ROOT:models@@/gguf/tony-tern/Thor-v20-densiv5.gguf THOR_SALTA_NON_FINITI=1 \
  THOR_CACHE_STRATI=@@ROOT:models@@/gguf/tony-tern/cache_strati \
  THOR_GROUPED_BMM=0 THOR_ROTAZIONE_INGRESSO=1 THOR_NUCLEO_BF16=1 THOR_ROUTING_VIVO=0 THOR_ROUTING_AGGIORNATO=1 \
  THOR_ESPERTI_THOR=1 THOR_DIAGNOSTICA_PASSO=0 THOR_VALIDA_BATCH_SEQ=8 THOR_VALIDA_N_BATCH=32 THOR_VALIDA_N_HELDOUT=1 \
  THOR_FORGIA_DOWN_TERNARIO=1 THOR_FORGIA_N_STRATI=4 THOR_FORGIA_W=2 THOR_VALIDA_N_PASSI=300 THOR_FORGIA_ALPHA=1.0 \
  THOR_VALIDA_LR=0.01 THOR_VALIDA_FRAZIONE_DURA=1.0 THOR_LR_Z=0.01 THOR_LR_LOGS=0.01 THOR_LR_DELTA=0.01 \
  THOR_HELDOUT_OGNI=0 THOR_RICONGELA_OGNI=0 THOR_CRONO_OGNI=5 THOR_COMMIT_DA_CODICI=1 THOR_COMMIT_SCALA_FP16=1 \
  THOR_CALIB_CORPUS= THOR_CALIB_MISCELA="cot:2,edu:3,wiki:3" THOR_ID_CORSA="radiografia-blocchi-$T" \
  THOR_VALIDA_OUT="@@ROOT:models@@/gguf/tony-tern/v2-radiografia-blocchi-NON-SCRITTO" \
  THOR_VALIDA_GGUF_OUT="@@ROOT:models@@/gguf/tony-tern/Thor-radiografia-blocchi-NON-ESPORTATO.gguf" \
  @@ROOT:venv@@/bin/python3 strumenti/radiografia_blocchi_passo.py --passi 48 --riscaldo 5 --cronometrati 10 \
    --prof-riscaldo 2 --profilati 30 --json "$D/radiografia_blocchi_passo_$T.json" \
    --tabella "$D/radiografia_blocchi_passo_$T.md" > "$D/radiografia_blocchi_passo_$T.log" 2>&1
rc=$?
echo "[$(date +%T)] fine radiografia per blocco rc=$rc · $D/radiografia_blocchi_passo_$T.md" >> "$D/riepilogo.log"
exit "$rc"

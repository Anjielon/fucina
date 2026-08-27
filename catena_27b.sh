#!/bin/bash
# CATENA 27B-TERN dal bf16: attende il download, converte, imatrix, Hessiane, forgia.
# Parte da sola; ogni passo salta se il suo prodotto esiste già (riprendibile).
set -u
D=/mnt/models/gguf/qwen38-tern
S=/mnt/usb-ssd/qwen38-27b-bf16/modello
BL=/home/angelo/build-llamacpp-latest/build
C=/home/angelo/odino-lab/guarigione/calibrazione_grande.txt
mkdir -p $D
log(){ echo "[$(date +%H:%M:%S)] 27B: $*"; }

log "attendo il download completo del bf16"
until grep -q "DOWNLOAD 27B COMPLETO" /mnt/usb-ssd/qwen38-27b-bf16/dl.log 2>/dev/null; do sleep 60; done
log "attendo che la GPU si liberi (fine filiera v3.2)"
until ! systemctl --user is-active --quiet odino-catena-v32; do sleep 120; done

log "[1/4] conversione bf16 → GGUF"
[ -s $D/qwen38-27b-bf16.gguf ] || /home/angelo/venv-catq/bin/python \
  /home/angelo/build-llamacpp-tq1/convert_hf_to_gguf.py $S --outtype bf16 \
  --outfile $D/qwen38-27b-bf16.gguf 2>&1 | tail -3

export LD_LIBRARY_PATH=$BL/bin:/usr/lib64:/usr/local/lib/ollama
export HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=
log "[2/4] imatrix"
[ -s $D/imatrix.gguf ] || $BL/bin/llama-imatrix -m $D/qwen38-27b-bf16.gguf -f $C \
  -ngl 999 --ctx-size 2048 -o $D/imatrix.gguf 2>&1 | tail -2

log "[3/4] Hessiane"
if [ ! -s $D/hessiane/H_00.npy ]; then
  mkdir -p $D/dumps $D/hessiane; rm -f $D/dumps/*
  MOGAVIS_DUMP_DIR=$D/dumps MOGAVIS_DUMP_FILTER=attn_post_norm \
    $BL/bin/llama-eval-callback -m $D/qwen38-27b-bf16.gguf -ngl 999 -c 2048 -n 1 -f $C 2>/dev/null | tail -1
  NDIM=$(python3 -c "
import json;print(json.load(open('$S/config.json')).get('hidden_size') or json.load(open('$S/config.json'))['text_config']['hidden_size'])")
  NDIM=$NDIM /home/angelo/venv-catq/bin/python /home/angelo/odino-lab/fucina/build_hessians.py $D/dumps $D/hessiane 2>&1 | tail -2
fi

log "[4/4] FORGIA"
unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
/home/angelo/venv-catq/bin/python /home/angelo/odino-lab/fucina/forgia_gguf.py \
  --sorgente $D/qwen38-27b-bf16.gguf --uscita $D/Qwen38-27B-tern-TQ1.gguf \
  --caldi 28 --imatrix $D/imatrix.gguf --hessiane $D/hessiane 2>&1 | tail -6
log "🏁 27B-TERN PRONTO (dal bf16, sorgente d'oro)"

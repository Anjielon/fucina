#!/bin/bash
# CATENA TONY-TERN: imatrix → cattura → Hessiane → forgia → header check
set -e
T=/mnt/models/gguf/ornith-1.5/Ornith-1.5-35B-A3B-APEX-MTP-I-Quality.gguf
D=/mnt/models/gguf/tony-tern
BL=/home/angelo/build-llamacpp-latest/build
C=/home/angelo/odino-lab/guarigione/calibrazione_grande.txt
export LD_LIBRARY_PATH=$BL/bin:/usr/lib64:/usr/local/lib/ollama
export HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=

echo "══ [1/4] imatrix (counts per i caldi)"
[ -s $D/tony-imatrix.gguf ] || $BL/bin/llama-imatrix -m $T -f $C -ngl 999 --ctx-size 2048 -o $D/tony-imatrix.gguf 2>&1 | tail -2

echo "══ [2/4] cattura attivazioni (Hessiane)"
if [ ! -s $D/dumps/attn_post_norm-40.f32 ] && [ ! -s $D/hessiane/H_40.npy ]; then
  mkdir -p $D/dumps; rm -f $D/dumps/*
  MOGAVIS_DUMP_DIR=$D/dumps MOGAVIS_DUMP_FILTER=attn_post_norm \
  $BL/bin/llama-eval-callback -m $T -ngl 999 -c 2048 -n 1 \
    -f $C 2>/dev/null | tail -1 || true
fi

echo "══ [3/4] Hessiane (2048 dim)"
[ -s $D/hessiane/H_40.npy ] || NDIM=2048 /home/angelo/venv-catq/bin/python \
  /home/angelo/odino-lab/fucina/build_hessians.py $D/dumps $D/hessiane 2>&1 | tail -3

echo "══ [4/4] FORGIA"
unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES   # torch deve VEDERE la GPU
/home/angelo/venv-catq/bin/python /home/angelo/odino-lab/fucina/forgia_gguf.py \
  --sorgente $T --uscita $D/Tony-tern-TQ1.gguf --caldi 28 \
  --imatrix $D/tony-imatrix.gguf --hessiane $D/hessiane 2>&1 | tail -8
echo "🏁 CATENA TONY COMPLETA"

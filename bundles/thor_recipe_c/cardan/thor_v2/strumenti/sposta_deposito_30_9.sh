#!/bin/bash
# 30/9: sposta archivi da @@ROOT:depot@@ (pieno) a @@ROOT:models@@/deposito_spostato, lascia un symlink al vecchio percorso
set -uo pipefail
DST=@@ROOT:models@@/deposito_spostato; mkdir -p $DST
for d in archivio_gguf_tony_tern millie riforgia; do
  S=@@ROOT:depot@@/$d
  [ -L "$S" ] && continue
  echo "[$(date +%T)] copio $d"
  rsync -a "$S/" "$DST/$d/" || { echo "rsync $d fallito"; continue; }
  if diff -rq "$S" "$DST/$d" >/dev/null; then rm -rf "$S" && ln -s "$DST/$d" "$S" && echo "[$(date +%T)] $d spostato"; else echo "$d diverso: non cancello"; fi
done
df -h @@ROOT:depot@@ | tail -1

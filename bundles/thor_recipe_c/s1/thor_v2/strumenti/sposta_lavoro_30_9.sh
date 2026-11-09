#!/bin/bash
# 30/9: sposta archivi da @@ROOT:lavoro@@ (14 GiB liberi) a @@ROOT:models@@/lavoro_spostato con symlink al vecchio percorso
set -uo pipefail
DST=@@ROOT:models@@/lavoro_spostato; mkdir -p $DST
for S in @@ROOT:lavoro@@/thor-ripresa @@ROOT:lavoro@@/e2e/prove_integrata @@ROOT:lavoro@@/tt3_cure_prove; do
  n=$(basename $S); [ -L "$S" ] && continue
  rsync -a "$S/" "$DST/$n/" && diff -rq "$S" "$DST/$n" >/dev/null && rm -rf "$S" && ln -s "$DST/$n" "$S" && echo "[$(date +%T)] $n spostato"
done
df -h @@ROOT:lavoro@@ | tail -1

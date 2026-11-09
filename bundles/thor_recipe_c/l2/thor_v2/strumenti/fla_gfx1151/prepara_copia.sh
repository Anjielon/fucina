#!/bin/bash
# Copia di fla 0.5.2 (e einops) FUORI dal venv-catq, per gfx1151 (27/9/2026).
# Perche' una copia: nel venv-catq fla e' installato ma NON importabile (manca einops, per scelta: se diventasse
# importabile, transformers al riavvio della corsa lo sceglierebbe da solo per il GDN — decoratore
# use_kernel_func_from_hub_with_fallback, risolto all'import). La copia vive in pkg/ e si carica solo
# esplicitamente da gdn_fla_gfx1151.carica_fla(), mai per sys.path globale.
set -euo pipefail
QUI=$(cd "$(dirname "$0")" && pwd)
SRC=@@ROOT:venv@@/lib64/python3.14/site-packages
DEST=${1:-$QUI/pkg}
mkdir -p "$DEST"
rsync -a --delete --exclude __pycache__ "$SRC/fla/" "$DEST/fla/"
rsync -a --delete --exclude __pycache__ "$QUI/../ambiente_prova_fla/einops/" "$DEST/einops/"
# patch della copia (vedi patch_fla.py: ognuna ha il suo perche')
@@ROOT:venv@@/bin/python3 "$QUI/patch_fla.py" "$DEST/fla"
( cd "$DEST" && find fla einops -name '*.py' -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-16 ) > "$DEST/IMPRONTA"
echo "copia pronta in $DEST (impronta $(cat "$DEST/IMPRONTA"))"

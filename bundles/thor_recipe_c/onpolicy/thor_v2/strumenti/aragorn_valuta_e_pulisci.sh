#!/bin/bash
# 27/9 15:5x — banco MMLU di un checkpoint su Aragorn e poi pulizia dei GGUF Thor-T-EB2-* sul Dell (~9 GB liberi).
# Va chiamato SOTTO aragorn_lucchetto.sh. Uso: aragorn_valuta_e_pulisci.sh <nome> <gguf> <log>
S=@@ROOT:forge@@/thor_v2/strumenti
"$S/valuta_checkpoint_aragorn.sh" --gguf "$1" "$2" > "$3" 2>&1; rc=$?
timeout 60 ssh aragorn "powershell -NoProfile -Command \"Get-ChildItem C:\\tcq\\thor_mmlu\\gguf\\Thor-T-EB2-*.gguf | Remove-Item\"" >/dev/null 2>&1
exit $rc

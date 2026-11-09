#!/bin/bash
# 30/9 23:3x modalita' sopravvivenza: sorveglianza deterministica, zero token. Scrive in avvisi.log SOLO sui cambi di stato.
S=@@ROOT:forge@@/stato; P=$S/guardia_notte.prec
while :; do
  r=$(timeout 60 ssh aragorn 'powershell -NoProfile -Command "$l=Get-ChildItem C:\tcq\thor_mmlu\log\*.log|Sort LastWriteTime -Desc|Select -First 1; $f=Test-Path ($l.FullName -replace \".log$\",\".fine\"); $p=@(Get-Process llama-perplexity -EA 0).Count; $a=[int]((Get-Date)-$l.LastWriteTime).TotalMinutes; \"$($l.BaseName) proc=$p fine=$f min=$a\""' 2>&1 | tr -d '\r' | tail -1)
  case "$r" in
    *proc=0*fine=True*) st="DELL LIBERO: catena MMLU finita ($r)";;
    *proc=0*) st="DELL: corsa MORTA senza .fine ($r)";;
    *proc=[1-9]*) m=${r##*min=}; [ "${m:-0}" -gt 90 ] 2>/dev/null && st="DELL: log fermo da ${m} min ($r)" || st="DELL ok ${r%% *}";;
    *) st="DELL irraggiungibile";;
  esac
  gpu=$(cat /sys/class/drm/card*/device/gpu_busy_percent 2>/dev/null | sort -n | tail -1)
  [ "${gpu:-0}" -gt 50 ] && st="$st | GPU ${gpu}%"
  k=$(echo "$st" | sed 's/[0-9]\+ min//; s/min=[0-9]*//')
  if [ "$k" != "$(cat $P 2>/dev/null)" ]; then echo "$k" > $P; case "$st" in "DELL ok"*) ;; *) echo "$(date +%H:%M) $st" | cut -c1-200 >> $S/avvisi.log;; esac; fi
  echo "$(date +%F\ %T) $st" > $S/guardia_notte.ultimo
  sed -i "s#^L7 .*#L7 MMLU pieno R2B3m+Q2K Dell | automatica | $(date +%H:%M) guardia: ${st//#/}#" $S/linee.txt
  sleep 1200
done

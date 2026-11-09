#!/bin/bash
# 30/9 23:3x ordine Angelo: aspettare le 10 del 1/10 e scaricare sul pod il ragionamento pesante (E2 dopo E1).
D=@@ROOT:home@@/claude-pod-lavoro/ricerca/esperti-E2
cd "$D" || exit 1
out=$(@@ROOT:forge@@/thor_v2/strumenti/pod_lancia.sh --modello claude-opus-5.5 --cartella "$D" --sfondo "$(cat COMPITO.md)" 2>&1); rc=$?
printf '%s\tPOD\tE2 lancio rc=%s %s\n' "$(date +%Y-%m-%dT%H:%M)" "$rc" "$(echo "$out"|tail -1|cut -c1-150)" >> @@ROOT:forge@@/stato/cicli.tsv
echo "$(date +%H:%M) pod E2 rc=$rc" >> @@ROOT:forge@@/stato/avvisi.log

#!/bin/bash
# Guardia della corsia: mentre un allenamento E2E gira sulla GPU (unita' e2e-allena-*), mette in PAUSA (SIGSTOP)
# le inferenze llama.cpp su CPU lanciate accanto, e le fa RIPARTIRE (SIGCONT) quando l'allenamento finisce.
# Perche' (28/9): un dump CPU del K01 (e2e_dump_logit, 325% CPU) ha portato EB3 dal 90-100% al 2-15% di GPU:
# su UMA la CPU che legge un modello da 35B consuma la stessa banda di memoria della GPU. Regola scritta, violata
# lo stesso giorno: serve un guardiano, non la memoria di chi lancia.
# Solo processi riconosciuti per NOME ESATTO (ps -o comm, mai pgrep -f), mai dentro l'unita' dell'allenamento.
export XDG_RUNTIME_DIR=/run/user/1000
NOMI='^(e2e_dump_logit|llama-perplexit|llama-cli|llama-imatrix)$'   # comm e' troncato a 15 caratteri
REG=@@ROOT:forge@@/stato/guardia_corsia_cpu.log
SOSPESI=@@ROOT:forge@@/stato/guardia_corsia_cpu.sospesi
touch "$SOSPESI"
while true; do
  if systemctl --user list-units --state=active --no-legend 'e2e-allena-*' | grep -q e2e-allena; then
    ps -eo pid=,stat=,comm= | awk -v r="$NOMI" '$3 ~ r && $2 !~ /T/ {print $1, $3}' | while read -r pid nome; do
      grep -q '/e2e-allena-' /proc/$pid/cgroup 2>/dev/null && continue
      if kill -STOP "$pid" 2>/dev/null; then
        echo "$pid" >> "$SOSPESI"
        echo "$(date '+%F %T') SOSPESO $pid $nome (allenamento attivo) cgroup=$(cut -c1-120 /proc/$pid/cgroup)" >> "$REG"
      fi
    done
  elif [ -s "$SOSPESI" ]; then
    while read -r pid; do
      [ -d /proc/$pid ] && kill -CONT "$pid" 2>/dev/null && echo "$(date '+%F %T') RIPRESO $pid" >> "$REG"
    done < "$SOSPESI"
    : > "$SOSPESI"
  fi
  sleep 30
done

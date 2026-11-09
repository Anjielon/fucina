#!/bin/bash
# Guardia GPU per un'unita' tt-*: EB3 si allena sulla GPU e ha la precedenza (UMA: CPU e GPU dividono la banda).
# Regola del coordinatore: GPU < 40 % per piu' di 2 minuti -> il lavoro si sospende. Qui con la PROVA di causalita':
#  - media degli ultimi 120 s (24 campioni ogni 5 s) < 40 %  -> SIGSTOP a tutta l'unita';
#  - da fermi, media degli ultimi 30 s >= 60 %               -> la colpa era nostra: si resta fermi finche' la GPU
#    non ha tenuto >= 60 % per 60 s, poi SIGCONT (e si ricomincia a guardare);
#  - da fermi 90 s con la GPU ancora < 40 %                  -> NON eravamo noi (la GPU non risale senza di noi):
#    SIGCONT e niente nuove pause per 10 minuti (poi si riprova). Ogni evento e ogni media ogni 5 min nel registro.
U=$1
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
F=/sys/class/drm/card0/device/gpu_busy_percent
fin=(); fermo=0; t_fermo=0; tregua=0; n=0; somma=0
log() { echo "[$(date '+%F %T')] $U: $*"; }
media() { local k=$1 s=0 c=0 i; for ((i=${#fin[@]}-k; i<${#fin[@]}; i++)); do [ $i -ge 0 ] && s=$((s+fin[i])) && c=$((c+1)); done; [ $c -gt 0 ] && echo $((s/c)) || echo 100; }
log "guardia accesa"
while systemctl --user is-active --quiet "$U"; do
  b=$(cat $F 2>/dev/null || echo 100)
  fin+=("$b"); [ ${#fin[@]} -gt 24 ] && fin=("${fin[@]:1}")
  n=$((n + 1)); somma=$((somma + b))
  [ $tregua -gt 0 ] && tregua=$((tregua - 5))
  if [ $fermo -eq 0 ] && [ $tregua -le 0 ] && [ ${#fin[@]} -ge 24 ] && [ "$(media 24)" -lt 40 ]; then
    systemctl --user kill -s SIGSTOP "$U" && fermo=1 && t_fermo=0 && log "⛔ GPU media 120 s = $(media 24)% -> SIGSTOP"
  elif [ $fermo -eq 1 ]; then
    t_fermo=$((t_fermo + 5))
    if [ $t_fermo -ge 60 ] && [ "$(media 12)" -ge 60 ]; then
      m=$(media 12); systemctl --user kill -s SIGCONT "$U" && fermo=0 && fin=() && log "GPU risalita a ${m}% con noi fermi (${t_fermo}s): colpa NOSTRA -> SIGCONT, si riguarda"
    elif [ $t_fermo -ge 90 ] && [ "$(media 18)" -lt 40 ]; then
      m=$(media 18); systemctl --user kill -s SIGCONT "$U" && fermo=0 && fin=() && tregua=600 && log "GPU ancora a ${m}% dopo ${t_fermo}s fermi: NON siamo noi -> SIGCONT, tregua 10 min"
    fi
  fi
  [ $((n % 60)) -eq 0 ] && log "media GPU ultimi 5 min: $((somma / n))% (fermo=$fermo)" && n=0 && somma=0
  sleep 5
done
log "unita' finita, guardia spenta"

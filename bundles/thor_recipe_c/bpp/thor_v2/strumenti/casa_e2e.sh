#!/bin/bash
# casa_e2e.sh giu|su|stato — spegne e riaccende la CASA intorno alla corsa E2E di Thor (26/9/2026).
# Decisione di Angelo (26/9 09:2x): «Si'! Spegni se serve!». Serve se M0 misura che con la casa accesa la RAM di sistema (30 GiB)
# o la VRAM non reggono la corsa: stamattina la RAM piena (swap 28-35 GiB) ha rallentato la forgia fino a 3 volte.
# Procedura di notte_completa.sh (unita' dei modelli di casa) + le lezioni pagate:
#   - timer che riaccendono o riavviano modelli a meta' corsa (feedback_timer_schedulato_contro_job_lungo): in pausa durante la corsa;
#   - riaccensione UNA dotazione alla volta, aspettando la salute (feedback_avvio_profilo_uno_alla_volta);
#   - con il cervello spento Sentinel ripiega su modelli di visione a richiesta PIU' grandi: `stato` elenca ogni llama-server vivo,
#     da guardare durante la corsa (mai kill per nome: solo per PID verificato un istante prima).
set -uo pipefail
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus LC_ALL=C
STATO=@@ROOT:forge@@/stato/casa_e2e.stato      # quali unita' e timer erano attivi prima di «giu»
MODELLI="mogavis-brain-server mogavis-llama-server mogavis-turbo-server mogavis-embed-server mogavis-whisper-server"
TIMER="mogavis-guardiano-ondemand.timer mogavis-inference-recycler-user.timer"
declare -A PORTA=([mogavis-brain-server]=11439 [mogavis-llama-server]=11438 [mogavis-turbo-server]=11440 \
                  [mogavis-embed-server]=11441 [mogavis-whisper-server]=11442)

vram() { awk '{printf "%.1f", $1/2^30}' /sys/class/drm/card*/device/mem_info_vram_used; }

stato() {
  echo "VRAM $(vram) GiB · $(free -g | awk 'NR==2{print "RAM usata " $3 " GiB, disponibile " $7 " GiB"}') · swap $(free -g | awk 'NR==3{print $3}') GiB"
  for u in $MODELLI; do echo "  $u: $(systemctl --user is-active $u)"; done
  for t in $TIMER; do echo "  $t: $(systemctl --user is-active $t)"; done
  # filtro sul NOME del processo (comm), non sulla riga di comando: «pgrep -f» e simili trovano anche chi cerca (specchio del pgrep)
  echo "  llama-server vivi:"
  for p in $(ps -eo pid=,comm= | awk '$2=="llama-server" {print $1}'); do
    printf "    pid %s da %s rss %.1fG %s\n" "$p" "$(ps -o etime= -p "$p" | tr -d ' ')" \
      "$(ps -o rss= -p "$p" | awk '{print $1/1048576}')" "$(ps -o args= -p "$p" | cut -c1-110)"
  done
}

giu() {
  [ -f "$STATO" ] && { echo "gia' giu' (esiste $STATO): prima «su»"; exit 2; }
  : > "$STATO"
  for t in $TIMER; do
    if systemctl --user is-active -q "$t"; then echo "timer $t" >> "$STATO"; systemctl --user stop "$t"; fi
  done
  for u in $MODELLI; do
    if systemctl --user is-active -q "$u"; then echo "unita $u" >> "$STATO"; systemctl --user stop "$u"; fi
  done
  sleep 5; echo "── casa GIU' ($(date '+%H:%M:%S'))"; stato
}

su() {
  [ -f "$STATO" ] || { echo "niente da riaccendere (manca $STATO)"; exit 2; }
  # una dotazione alla volta: si aspetta la salute della precedente prima della successiva
  while read -r tipo nome; do
    [ "$tipo" = unita ] || continue
    systemctl --user start "$nome"; p=${PORTA[$nome]:-}
    if [ -n "$p" ]; then
      for i in $(seq 1 90); do curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$p/health" | grep -q 200 && break; sleep 2; done
      echo "  $nome :$p salute $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$p/health)"
    fi
  done < "$STATO"
  while read -r tipo nome; do [ "$tipo" = timer ] && systemctl --user start "$nome"; done < "$STATO"
  rm -f "$STATO"; echo "── casa SU ($(date '+%H:%M:%S'))"; stato
}

case "${1:-stato}" in giu) giu ;; su) su ;; stato) stato ;; *) echo "uso: $0 giu|su|stato"; exit 2 ;; esac

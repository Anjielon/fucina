#!/bin/bash
# Sorveglianza DIRETTA di EB3 (28/9, Angelo: «stargli addosso costantemente, feedback continuo»).
# Ogni 5 min una riga: passo, micro-lotto k (da py-spy --locals --nonblocking, senza fermare il processo), funzione in
# corso, GPU media. ALLARME subito se: GPU media < 10% per 90 s, oppure micro-lotto fermo da 15 min, oppure unita' giu'.
export XDG_RUNTIME_DIR=/run/user/1000
R=@@ROOT:depot@@/e2e/allena/EB3/registro.log
ultimo_k=""; t_k=$(date +%s); basse=0; t_riga=0
posizione() {
  local P=$1 d
  d=$(timeout 45 py-spy dump --pid "$P" --locals --nonblocking 2>/dev/null)
  local fn; fn=$(echo "$d" | awk '/MainThread/{f=1;next} f&&/^Thread /{exit} f&&/^    [A-Za-z_]+ \(/{print $1; n++} n==3{exit}' | tr '\n' '<' | cut -c1-60)
  local k; k=$(echo "$d" | awk '/_micro_lotti_pesati/{f=1} f&&/^ +k: /{print $2; exit}')
  echo "${k:-?}|$fn"
}
while true; do
  if ! systemctl --user is-active --quiet e2e-allena-EB3; then
    echo "$(date +%T) ⛔ ALLARME: e2e-allena-EB3 NON attiva ($(systemctl --user is-active e2e-allena-EB3))"; sleep 60; continue
  fi
  s=0; for i in 1 2 3 4 5 6; do s=$((s + $(cat /sys/class/drm/card*/device/gpu_busy_percent))); sleep 15; done; g=$((s/6))
  if [ "$g" -lt 10 ]; then basse=$((basse+1)); else basse=0; fi
  now=$(date +%s); P=$(systemctl --user show -p MainPID --value e2e-allena-EB3)
  passo=$(grep -oE "passo [0-9]+/[0-9]+" "$R" | tail -1)
  if [ "$basse" -ge 1 ] || [ $((now - t_riga)) -ge 300 ]; then
    pos=$(posizione "$P"); k=${pos%%|*}; fn=${pos#*|}
    if [ "$k" != "$ultimo_k" ]; then ultimo_k=$k; t_k=$now; fi
    fermo=$(( (now - t_k) / 60 ))
    if [ "$basse" -ge 1 ]; then
      echo "$(date +%T) ⛔ ALLARME GPU ${g}% per 90 s · ${passo:-passo ?} micro-lotto k=$k · $fn"
    elif [ "$fermo" -ge 15 ] && [ "$k" != "?" ]; then
      echo "$(date +%T) ⛔ ALLARME micro-lotto k=$k fermo da ${fermo} min · GPU ${g}% · $fn"
    else
      echo "$(date +%T) EB3 ${passo:-dopo passo 20} · micro-lotto k=$k/16 (fermo da ${fermo} min) · GPU ${g}% · $fn"
    fi
    t_riga=$now
  fi
done

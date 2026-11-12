#!/bin/bash
# 10/10 (Angelo: tutte le leve anche sui modelli piu piccoli; 80,05 resta il bersaglio): obiettivo B, ospite eb882K.
# Turno: dopo prova-grad-pv12p, combinazione-0910 e pieno-1010 (A ha la precedenza), prima di c0-0910 (che ci aspetta).
#   1 GPU gradienti esatti della KL + Kronecker dei densi IN eb882K (AUDIT/prova_leve/grad_eb882K_kl4.sh, GRAD_META=1)
#   2 combinazione delle leve su eb882K (COMBINAZIONE/combinazione_eb882K.sh: B-F, + CARDAN eb882K in parallelo)
set -u
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
S=/home/angelo/tony-forgia/stato; TT=/mnt/models/gguf/tony-tern
riga(){ printf '%s\t%s\t%s\n' "$(date +%Y-%m-%dT%H:%M)" "$1" "$2" >> $S/cicli.tsv; }
PRIMA="prova-grad-pv12p combinazione-0910 pieno-scrivi-1010 pieno-1010 cardan-densi-1010 selrisp-1010 diagnosi-densi-1010 cardan-fedele-1010 combo-a3-1011"
occupata(){ for u in $PRIMA; do systemctl --user is-active -q $u.service && { echo $u; return 0; }; done; return 1; }
riga ATTIVO "LEVE-EB882K: in coda dietro $PRIMA (poi gradienti eb882K ~1 h GPU, 9 leve ~1,5 h GPU, MMLU per candidato)"
while occupata >/dev/null; do sleep 20; done
if ! ls $TT/AUDIT/prova_leve/grad_eb882K_kl4/g_*.npy >/dev/null 2>&1 || [ ! -s $TT/AUDIT/prova_leve/grad_eb882K_kl4/fatto.txt ]; then
  bash $TT/AUDIT/prova_leve/grad_eb882K_kl4.sh || { riga ERRORE "LEVE-EB882K: gradienti eb882K falliti (codice $?)"; exit 2; }
fi
bash $TT/COMBINAZIONE/combinazione_eb882K.sh

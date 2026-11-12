#!/bin/bash
# 9/10 sera (Angelo: massimizzare il throughput): CARDAN-KD sull'ospite B eb882K, subito dopo seguito-0910 e PRIMA di prova-grad-pv12p/combinazione-0910.
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
while systemctl --user is-active -q seguito-0910.service; do sleep 60; done
R=/mnt/models/gguf/tony-tern/CARDAN
exec env CK_OSPITE=/mnt/models/gguf/tony-tern/TUTTE/eb882K.gguf CK_NOME=eb882K CK_BASE=$R/base_A1_r16_eb882K CK_BASE_UNITA=cardan-base-a1-eb882k \
  CK_OUT=$R/kd_r16_eb882K CK_LIMITE=9492293427 CK_MMLU=1 CK_ORE=3 CK_PASSI=300 CK_RANGO=16 bash $R/cardan_ospite.sh

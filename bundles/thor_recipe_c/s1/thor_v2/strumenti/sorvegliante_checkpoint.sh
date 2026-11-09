#!/bin/bash
# 27/9 notte: a ogni nuovo migliore_XXXXXX di EB2 -> KLD standard (Corsair, -ub 512); ogni 4 passi anche MMLU su Aragorn.
# Tiene solo l'export piu' recente su @@ROOT:models@@ (disco al 99%). Nessuna quota Claude.
set -u
D=@@ROOT:lavoro@@/e2e/allena/EB2; S=@@ROOT:forge@@/thor_v2/strumenti; ULT=""
while true; do
  # 27/9 15:5x: anche le istantanee EMA (il passo 16 ha salvato SOLO migliore_ema_000016); ordine per numero di passo (ultimo campo)
  M=$(ls -d $D/migliore_[0-9][0-9][0-9][0-9][0-9][0-9] $D/migliore_ema_[0-9][0-9][0-9][0-9][0-9][0-9] 2>/dev/null | awk -F_ '{print $NF" "$0}' | sort | tail -1 | cut -d' ' -f2)
  if [ -n "$M" ] && [ "$M" != "$ULT" ] && [ ! -f "@@ROOT:lavoro@@/e2e/fedelta/kld/kld_EB2-$(basename $M).log" ]; then
    sleep 120                                      # l'istantanea e' scritta atomicamente; lascia finire la ripresa
    # 27/9 10:31: la KLD partita accanto a EB2 + cancello B3 (3 lavori GPU) ha fatto scadere la coda (ring timeout, reset).
    # Ora aspetta che sulla GPU ci sia SOLO EB2 (guardia unica, VRAM per processo da kfd e da DRM/Vulkan).
    EB2=$(XDG_RUNTIME_DIR=/run/user/1000 systemctl --user show e2e-allena-EB2 -p MainPID --value)
    until GPU_ESCLUDI_PID="$EB2" "$S/gpu_occupata.sh" >/dev/null 2>&1; do sleep 60
      EB2=$(XDG_RUNTIME_DIR=/run/user/1000 systemctl --user show e2e-allena-EB2 -p MainPID --value); done
    "$S/kld_checkpoint.sh" "$M"
    P=$((10#$(basename "$M" | awk -F_ '{print $NF}')))
    G=@@ROOT:models@@/gguf/tony-tern/Thor-T-EB2-$(basename "$M").gguf
    # 27/9 (Angelo: niente MMLU senza un segnale forte): MMLU solo se la KLD standard scende sotto 0,25 (rivale Q2_K 0,2201)
    K=$(grep -oE "Mean *KLD: *[0-9.]+" @@ROOT:lavoro@@/e2e/fedelta/kld/kld_EB2-$(basename "$M").log 2>/dev/null | grep -oE "[0-9.]+$")
    # 27/9 12:5x: dal passo 14 la perdita e' CAKLD (allineata all'MMLU, non alla KL in avanti): l'MMLU su Aragorn gira a OGNI
    # nuovo migliore (Aragorn non ruba GPU alla corsa); la soglia KLD < 0,25 non vale piu' come filtro.
    # 27/9 15:5x: banco E pulizia di Aragorn sotto il lucchetto unico (aragorn_lucchetto.sh), mai sovrapposti ad altri lavori
    if [ -s "$G" ]; then
      "$S/aragorn_lucchetto.sh" "$S/aragorn_valuta_e_pulisci.sh" "EB2-$(basename "$M")" "$G" @@ROOT:lavoro@@/e2e/allena/aragorn_EB2_p$P.log
    fi
    ls -t @@ROOT:models@@/gguf/tony-tern/Thor-T-EB2-migliore_*.gguf 2>/dev/null | tail -n +2 | xargs -r rm -f
    ULT=$M
  fi
  sleep 60
done

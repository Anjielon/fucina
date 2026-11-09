#!/bin/bash
# Guardia "GPU occupata" UNICA per le code di thor_v2 (25/9).
# Decide la VRAM che il KERNEL attribuisce a ciascun processo, da DUE fonti:
#   - ROCm: /sys/class/kfd/kfd/proc/PID/vram_<gpu> (byte);
#   - DRM (Vulkan COMPRESO): /proc/PID/fdinfo/*, campo drm-memory-vram per client DRM amdgpu (ogni client una volta).
# Perche' due: 01:24 la guardia che bloccava chi teneva aperto /dev/kfd ha fermato la coda per un collaudo su CPU
# (0 byte); 02:25 la versione solo-kfd rispondeva «libera» mentre llama-perplexity (Vulkan) teneva 68 GiB
# (trovato dal verificatore della diagnosi NaN). Esenti i server della casa (llama-server, whisper-server).
# Stampa chi occupa la GPU; exit 1 se qualcuno la occupa, 0 se libera.
SOGLIA_MIB=${GPU_SOGLIA_MIB:-256}
# --attendi S: riprova ogni 10 s fino a S secondi prima di dire «occupata» (04:39 del 25/9: il processo della forgia appena
# uscita era ancora elencato dal kernel per un istante e la coda si e' fermata per niente).
ATTENDI=0
[ "${1:-}" = "--attendi" ] && ATTENDI=${2:?secondi}
if [ "$ATTENDI" -gt 0 ]; then
  fine=$(( $(date +%s) + ATTENDI ))
  while :; do
    esito=$("$0"); rc=$?
    [ "$rc" -eq 0 ] && exit 0
    [ "$(date +%s)" -ge "$fine" ] && { echo "$esito"; exit 1; }
    sleep 10
  done
fi
# Registro di coesistenza (26/9): PID che un processo GPU cooperante dichiara "atteso" (il training theta di
# e2e_allena.py mentre e2e_teacher.py produce blocchi in parallelo, per disegno — mai un e2e_dump_logit in
# piu': `_coesistenza_gpu` in e2e_allena.py non tocca mai e2e_dump_logit). E' l'UNICO canale che raggiunge un
# chiamante GIA' vivo (il teacher non puo' ricevere un nuovo GPU_ESCLUDI_PID nel suo ambiente): file riletto
# FRESCO qui ad ogni invocazione, dal teacher come dal training. Un PID conta solo se il processo esiste
# ancora ORA, cosi' un registro non ripulito (crash senza cleanup) non blocca la guardia per sempre.
REGISTRO_COESISTENZA="$(dirname "$0")/.gpu_coesistenza"
ESCLUSI_REGISTRO=""
if [ -r "$REGISTRO_COESISTENZA" ]; then
  for rp in $(cat "$REGISTRO_COESISTENZA" 2>/dev/null); do
    [ -d "/proc/$rp" ] && ESCLUSI_REGISTRO="$ESCLUSI_REGISTRO $rp"
  done
fi

declare -A MIB

# ROCm (kfd)
for d in /sys/class/kfd/kfd/proc/[0-9]*; do
  [ -d "$d" ] || continue
  pid=${d##*/}
  byte=0
  for f in "$d"/vram_*; do
    [ -r "$f" ] || continue
    v=$(cat "$f" 2>/dev/null) || continue
    byte=$(( byte + ${v:-0} ))
  done
  MIB[$pid]=$(( byte / 1048576 ))
done

# DRM fdinfo (Vulkan e ROCm): somma per client DRM amdgpu distinto
for p in /proc/[0-9]*; do
  pid=${p##*/}
  ls "$p/fdinfo" >/dev/null 2>&1 || continue
  kib=$(awk '
    FNR == 1 { drv = ""; id = ""; v = 0 }
    /^drm-driver:/      { drv = $2 }
    /^drm-client-id:/   { id = $2 }
    /^drm-memory-vram:/ { v = $2; if ($3 == "MiB") v *= 1024; else if ($3 == "GiB") v *= 1048576 }
    ENDFILE { if (drv == "amdgpu" && id != "" && !(id in visto)) { visto[id] = 1; tot += v } }
    END { print tot + 0 }' "$p"/fdinfo/* 2>/dev/null)
  mib=$(( ${kib:-0} / 1024 ))
  [ "$mib" -gt "${MIB[$pid]:-0}" ] && MIB[$pid]=$mib
done

occupata=0
for pid in "${!MIB[@]}"; do
  mib=${MIB[$pid]}
  [ "$mib" -gt "$SOGLIA_MIB" ] || continue
  # processo gia' uscito o zombie: il kernel puo' elencarlo ancora per un istante (04:39, comm «?»)
  [ -d "/proc/$pid" ] || continue
  grep -q "^State:[[:space:]]*Z" "/proc/$pid/status" 2>/dev/null && continue
  c=$(cat "/proc/$pid/comm" 2>/dev/null)
  case "$c" in llama-server|whisper-server) continue ;; esac
  # GPU_ESCLUDI_PID: chi chiama la guardia da DENTRO un lavoro che tiene gia' la GPU (la forgia al secondo controllo) non
  # deve contare se stesso (25/9). ESCLUSI_REGISTRO (26/9): il registro di coesistenza sopra.
  case " ${GPU_ESCLUDI_PID:-}${ESCLUSI_REGISTRO} " in *" $pid "*) continue ;; esac
  echo "$pid ${c:-?} ${mib} MiB $(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | head -c 120)"
  occupata=1
done
exit $occupata

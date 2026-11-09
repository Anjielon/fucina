#!/bin/bash
# 30/9 — MMLU COMPLETO (14.042 domande, 57 materie) sul Dell per R2B3m e il rivale Q2_K (bartowski 12,19 GiB).
# Pezzi mc_v2/mmlu_00000..14000.bin: verificato che concatenati compito per compito = mmlu_test.bin (14042, stesso ordine).
# R2B3m: il pezzo 00000 = R2B3.log (logit identici al bit, stesso Dell) -> non si rifa'. Q2_K: mai misurato sul Dell -> 8 pezzi.
# Lucchetto del Dell preso UNA volta per pezzo: la catena TT4 puo' entrare fra un pezzo e l'altro.
set -uo pipefail
S=@@ROOT:forge@@/thor_v2/strumenti; C=@@ROOT:depot@@/tutto_ternario/codice; MC=@@ROOT:forge@@/mc_v2
R2=@@ROOT:models@@/gguf/tony-tern/R2B3m/Thor-T-R2B3m.gguf; FR2=Thor-T-R2B3m.gguf
FQ=Ornith-1.5-35B-A3B-Q2_K.gguf; LQ=@@ROOT:models@@/gguf/thor-rivale/$FQ
log() { echo "[$(date '+%F %T')] $*"; }
preparazione() {
  log "preparazione (sotto lucchetto)"
  # copia vecchia di scor150 sul Dell: l'originale e' sul Corsair (stessa taglia) -> si toglie per fare spazio
  [ "$(stat -c %s @@ROOT:models@@/gguf/tony-tern/Thor-v2-ThorT-scor150.gguf)" = 9334955680 ] && \
    timeout 60 ssh aragorn "del C:\\tcq\\thor_mmlu\\gguf\\Thor-v2-ThorT-scor150.gguf" && log "tolto scor150 dal Dell (originale sul Corsair)"
  for p in 02000 04000 06000 08000 10000 12000 14000; do
    f=mmlu_$p.bin; L=$(sha256sum $MC/$f | cut -d' ' -f1)
    A=$(timeout 120 ssh aragorn "certutil -hashfile C:\\tcq\\thor_mmlu\\$f SHA256" 2>/dev/null | sed -n 2p | tr -d '\r ')
    [ "$A" = "$L" ] || { timeout 300 scp -q $MC/$f aragorn:C:/tcq/thor_mmlu/$f; A=$(timeout 120 ssh aragorn "certutil -hashfile C:\\tcq\\thor_mmlu\\$f SHA256" | sed -n 2p | tr -d '\r '); }
    [ "$A" = "$L" ] || { log "⛔ sha $f"; return 3; }
  done; log "pezzi sul Dell, sha verificati"
  L=$(sha256sum $LQ | cut -d' ' -f1); A=$(timeout 900 ssh aragorn "certutil -hashfile C:\\tcq\\thor_mmlu\\gguf\\$FQ SHA256" | sed -n 2p | tr -d '\r ')
  [ "$A" = "$L" ] || { log "⛔ Q2_K sul Dell diverso dal locale ($A / $L)"; return 3; }; log "Q2_K sul Dell = locale (sha $L)"
  LIB=$(timeout 60 ssh aragorn "powershell -NoProfile -Command \"(Get-PSDrive C).Free\"" | tr -dc 0-9)
  SERVE=$(( $(stat -c %s $R2) + 10*1073741824 ))   # R2B3m + posto per l'export TT4 (~9 GB) + margine
  [ "$LIB" -gt "$SERVE" ] || { log "⛔ spazio Dell $LIB < $SERVE"; return 4; }
  L=$(sha256sum $R2 | cut -d' ' -f1)
  timeout 3600 scp -q $R2 aragorn:C:/tcq/thor_mmlu/gguf/$FR2 || { log "⛔ copia R2B3m"; return 3; }
  A=$(timeout 900 ssh aragorn "certutil -hashfile C:\\tcq\\thor_mmlu\\gguf\\$FR2 SHA256" | sed -n 2p | tr -d '\r ')
  [ "$A" = "$L" ] || { log "⛔ sha R2B3m sul Dell"; return 3; }; log "R2B3m COPIATO sul Dell, sha $L"
}
export -f preparazione log; export S MC R2 FR2 FQ LQ
$S/aragorn_lucchetto.sh bash -c preparazione || { log "⛔ preparazione fallita"; exit 3; }
nt() { [ "$1" = 14000 ] && echo 42 || echo 2000; }
for p in 00000 02000 04000 06000 08000 10000 12000 14000; do
  $S/aragorn_lucchetto.sh bash $C/mmlu_completo_pezzo.sh MMLUC-Q2K-$p $FQ mmlu_$p.bin q2k24 $(nt $p) || log "⛔ Q2K $p rc=$?"
  if [ $p != 00000 ]; then
    $S/aragorn_lucchetto.sh bash $C/mmlu_completo_pezzo.sh MMLUC-R2B3m-$p $FR2 mmlu_$p.bin thor16 $(nt $p) || log "⛔ R2B3m $p rc=$?"
  fi
done
$S/aragorn_lucchetto.sh bash -c "timeout 60 ssh aragorn 'del C:\\tcq\\thor_mmlu\\gguf\\$FR2'" && log "R2B3m tolto dal Dell (Q2_K resta: c'era prima)"
python3 $C/mmlu_completo_aggrega.py && log "AGGREGAZIONE FATTA"

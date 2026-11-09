#!/bin/bash
# 27/9: PyTorch TunableOp su ROCm (scelta misurata del kernel GEMM per ogni forma) sul passo di Thor-T: M0 B_ind32 strati 0 e 0,1
# (1) regolazione, (2) uso dei kernel scelti, (3) riferimento senza; stesse condizioni (EB2 accanto). Parte dopo il banco MMLU.
set -u
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
until ! systemctl --user is-active -q e2e-mmlu-p10; do sleep 20; done
P=$(systemctl --user show e2e-allena-EB2 -p MainPID --value); Q=@@ROOT:forge@@/thor_v2; O=@@ROOT:lavoro@@/e2e/allena
F=$O/tunableop_risultati.csv
m(){ # $1 nome  $2.. env
  local n=$1; shift
  systemd-run --user --unit=e2e-tunop-$n --collect --wait --quiet -p Nice=5 --setenv=GPU_ESCLUDI_PID=$P "$@" \
    --working-directory $Q @@ROOT:venv@@/bin/python3 e2e_allena.py misura --teacher @@ROOT:lavoro@@/e2e/teacher/q8-EB-pieno \
    --strati 0 --strati 0,1 --ripetizioni 2 --ricetta B_ind32 --uscita $O/misura_tunop_$n.json
}
m regola --setenv=PYTORCH_TUNABLEOP_ENABLED=1 --setenv=PYTORCH_TUNABLEOP_TUNING=1 --setenv=PYTORCH_TUNABLEOP_FILENAME=$F
m usa --setenv=PYTORCH_TUNABLEOP_ENABLED=1 --setenv=PYTORCH_TUNABLEOP_TUNING=0 --setenv=PYTORCH_TUNABLEOP_FILENAME=$F
m rif --setenv=PYTORCH_TUNABLEOP_ENABLED=0
R=$(python3 -c "
import json
r={n:json.load(open('$O/misura_tunop_%s.json'%n))['configurazioni'] for n in ('usa','rif')}
g=lambda c:c['0,1']['totale_ms_token']-c['0']['totale_ms_token']
print('strato GDN: rif %.3f · TunableOp %.3f ms/token (x%.2f) · totale 0,1: rif %.3f · TunableOp %.3f' % (g(r['rif']),g(r['usa']),g(r['rif'])/g(r['usa']),r['rif']['0,1']['totale_ms_token'],r['usa']['0,1']['totale_ms_token']))" 2>&1)
printf "%s\tCHIUSO\t%s\n" "$(date +%Y-%m-%dT%H:%M)" "TunableOp (M0 B_ind32 accanto a EB2): $R" >> @@ROOT:forge@@/stato/cicli.tsv
echo "627 | $(date +%H:%M) TunableOp: $R" >> @@ROOT:forge@@/stato/linee.txt

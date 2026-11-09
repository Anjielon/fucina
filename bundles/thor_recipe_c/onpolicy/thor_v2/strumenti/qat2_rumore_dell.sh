#!/bin/bash
# 1/10: dopo la catena QAT2, misura il pavimento di rumore: R2B3m contro i propri logit salvati (stesso binario).
# Serve a leggere il KLD del neutro (0,0023, top uguale 98,2%): rumore del Vulkan o grafo diverso?
set -u; S=@@ROOT:forge@@/stato; Q=$S/qat2
export XDG_RUNTIME_DIR=/run/user/1000
while systemctl --user is-active -q catena-qat2; do sleep 120; done
timeout 60 ssh aragorn "del C:\\tcq\\thor_mmlu\\log\\kld_self.fine 2>nul & schtasks /create /f /tn kld_self /tr \"C:\\tcq\\thor_mmlu\\identita_kld.bat self Thor-T-R2B3m.gguf kld r2b3m_qat2.kld\" /sc once /st 23:59 /sd 01/01/2030 /rl limited && schtasks /run /tn kld_self" >/dev/null
sleep 20; until timeout 30 ssh aragorn "type C:\\tcq\\thor_mmlu\\log\\kld_self.fine" 2>/dev/null | grep -q FATTO; do sleep 30; done
timeout 30 ssh aragorn "schtasks /delete /f /tn kld_self" >/dev/null
scp -q aragorn:C:/tcq/thor_mmlu/log/kld_self.log $Q/kld_self.log
R=$(grep -E "^Mean +KLD|^Same top p" $Q/kld_self.log | tr -s ' ' | tr '\n' ';')
printf '%s\t%s\t%s\n' "$(date +%Y-%m-%dT%H:%M)" CHIUSO "QAT2 pavimento di rumore Dell: R2B3m contro se stesso (stesso binario): $R (neutro era KLD 0,0023, top 98,2%)" >> $S/cicli.tsv
echo FATTO

#!/bin/bash
# 1/10 sera: catena QAT2 sul Dell (zero token). Gemella di catena_E1_dell.sh, con tre differenze strutturali:
#  * ogni schtasks e' creato con data FUTURA (/sd 01/01/2030) e CANCELLATO a lavoro finito: i task "once /st 23:59"
#    creati prima ripartivano alle 23:59 dello stesso giorno (il 30/9 uno ha sovrascritto R2B3.log sul Dell);
#  * prima del banco, controllo d'identita' della patch llama.cpp qat2-bias: R2B3m e QAT2-neutro devono dare
#    gli stessi logit (KLD) sul binario nuovo;
#  * R2B3m rimisurato sul MMLU 2000 con lo STESSO binario di QAT2 (confronto appaiato stessa macchina, stesso binario).
set -u
S=@@ROOT:forge@@/stato; Q=$S/qat2; mkdir -p $Q
T=@@ROOT:forge@@/thor_v2/strumenti
log(){ printf '%s\t%s\t%s\n' "$(date +%Y-%m-%dT%H:%M)" "$1" "$2" >> $S/cicli.tsv; echo "$(date +%H:%M) $1 $2"; }
dell(){ timeout 60 ssh aragorn "$1"; }
# compito NOME_TASK "COMANDO" FILE_FINE  -> crea, avvia, attende FATTO, cancella il task; rc del comando in $RC
compito(){
  local tn=$1 cmd=$2 fine=$3
  dell "del C:\\tcq\\thor_mmlu\\log\\$fine 2>nul & schtasks /create /f /tn $tn /tr \"$cmd\" /sc once /st 23:59 /sd 01/01/2030 /rl limited && schtasks /run /tn $tn" >/dev/null || return 9
  sleep 20
  until timeout 30 ssh aragorn "type C:\\tcq\\thor_mmlu\\log\\$fine" 2>/dev/null | grep -q FATTO; do sleep 60; done
  RC=$(timeout 30 ssh aragorn "type C:\\tcq\\thor_mmlu\\log\\$fine" | grep -o 'rc=[0-9-]*' | tail -1)
  dell "schtasks /delete /f /tn $tn" >/dev/null
}
libero(){ ! timeout 30 ssh aragorn "powershell -NoProfile -Command \"Get-Process llama-* -ErrorAction SilentlyContinue\"" | grep -q llama; }

# 0) build e copie
until timeout 30 ssh aragorn "type C:\\tcq\\thor_mmlu\\log\\build_qat2_fine.txt" 2>/dev/null | grep -q FATTO; do sleep 60; done
B=$(timeout 30 ssh aragorn "type C:\\tcq\\thor_mmlu\\log\\build_qat2_fine.txt" | grep -o 'rc=[0-9-]*' | tail -1)
[ "$B" = "rc=0" ] || { log ERRORE "QAT2: build Dell fallita ($B), vedi C:\\tcq\\thor_mmlu\\log\\build_qat2_compile.log"; exit 3; }
dell "schtasks /delete /f /tn build_qat2" >/dev/null
until grep -q -E 'FATTO|ERRORE' @@ROOT:models@@/gguf/tony-tern/QAT2/copia_dell.log 2>/dev/null; do sleep 60; done
grep -q FATTO @@ROOT:models@@/gguf/tony-tern/QAT2/copia_dell.log || { log ERRORE "QAT2: copia sul Dell fallita: $(tail -1 @@ROOT:models@@/gguf/tony-tern/QAT2/copia_dell.log)"; exit 3; }
until libero; do sleep 120; done
log ATTIVO "QAT2: binario Dell ricostruito pulito (terzo-densi + patch qat2-bias), 3 GGUF copiati (sha ok). Controllo d'identita' logit in corso"

# 1) identita': logit R2B3m (base) contro QAT2-neutro (deve essere ~0) e contro QAT2 (effetto)
K="C:\\tcq\\thor_mmlu\\identita_kld.bat"
compito kld_base "$K base_R2B3m Thor-T-R2B3m.gguf base r2b3m_qat2.kld" kld_base_R2B3m.fine
compito kld_neutro "$K neutro Thor-T-QAT2-neutro.gguf kld r2b3m_qat2.kld" kld_neutro.fine
compito kld_qat2 "$K qat2 Thor-T-QAT2.gguf kld r2b3m_qat2.kld" kld_qat2.fine
for n in base_R2B3m neutro qat2; do scp -q "aragorn:C:/tcq/thor_mmlu/log/kld_$n.log" $Q/kld_$n.log; done
riass(){ grep -E "^Mean +KLD|^Maximum KLD|^Same top p|^Mean    PPL\(Q\)|^Mean ln" $Q/kld_$1.log | tr -s ' ' | tr '\n' ';' ; }
log ATTIVO "QAT2 identita' (KLD contro R2B3m, stesso binario): neutro: $(riass neutro) | QAT2: $(riass qat2)"
dell "del C:\\tcq\\thor_mmlu\\gguf\\Thor-T-QAT2-neutro.gguf" >/dev/null

# 2) sonda parlata QAT2
compito sonda_QAT2 "C:\\tcq\\thor_mmlu\\sonda_parlata.bat QAT2 Thor-T-QAT2.gguf 06:00" sonda_QAT2.fine
scp -q "aragorn:C:/tcq/thor_mmlu/log/sonda_QAT2.log" $Q/sonda_QAT2.log
scp -q "aragorn:C:/tcq/thor_mmlu/log/sonda_QAT2.txt" $Q/sonda_QAT2.txt 2>/dev/null
log ATTIVO "QAT2 sonda ($RC): $(tail -3 $Q/sonda_QAT2.log | tr '\n\t' '  ' | cut -c1-160); MMLU 2000 QAT2 partito"

# 3) MMLU 2000 QAT2 poi R2B3m (stesso binario)
compito thor_mmlu_QAT2 "C:\\tcq\\thor_mmlu\\banco_ot.bat QAT2 Thor-T-QAT2.gguf" QAT2.fine
scp -q "aragorn:C:/tcq/thor_mmlu/log/QAT2.log" $Q/QAT2.log
log ATTIVO "MMLU 2000 QAT2 ($RC): $(grep -oE 'Final result: [0-9.]+ \+/- [0-9.]+' $Q/QAT2.log | tail -1); ora R2B3m sullo stesso binario"
compito thor_mmlu_R2B3m_q2b "C:\\tcq\\thor_mmlu\\banco_ot.bat R2B3m_q2bin Thor-T-R2B3m.gguf" R2B3m_q2bin.fine
scp -q "aragorn:C:/tcq/thor_mmlu/log/R2B3m_q2bin.log" $Q/R2B3m_q2bin.log

# 4) McNemar
python3 $T/mcnemar_mclp.py $Q/R2B3m_q2bin.log $Q/QAT2.log --json $Q/mcnemar_QAT2_vs_R2B3m_stessobinario.json > /dev/null
python3 $T/mcnemar_mclp.py @@ROOT:models@@/gguf/tony-tern/mmlu_logprob/aragorn/R2B3.log $Q/R2B3m_q2bin.log --json $Q/controllo_R2B3_vecchio_vs_nuovo_binario.json > /dev/null
python3 $T/mcnemar_mclp.py @@ROOT:models@@/gguf/tony-tern/mmlu_logprob/aragorn/R2B3.log $Q/QAT2.log --json $Q/mcnemar_QAT2_vs_R2B3log.json > /dev/null
M=$(python3 -c "import json;d=json.load(open('$Q/mcnemar_QAT2_vs_R2B3m_stessobinario.json'));print(f\"QAT2 {d['acc_B']:.2f} vs R2B3m {d['acc_A']:.2f} (stesso binario Dell), {d['B_giusta_A_sbagliata']}/{d['A_giusta_B_sbagliata']}, p={d['p_mcnemar_esatto']:.4g}\")")
C=$(python3 -c "import json;d=json.load(open('$Q/controllo_R2B3_vecchio_vs_nuovo_binario.json'));print(f\"R2B3m nuovo binario {d['acc_B']:.2f} vs R2B3.log {d['acc_A']:.2f}, scelte uguali {d['scelte_uguali']}/2000\")")
log CHIUSO "MMLU 2000 QAT2: $M. Controllo: $C. Soglia promozione >=75,85 e p<0,05."
echo FATTO

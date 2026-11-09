#!/bin/bash
# 1/10 04:0x: catena deterministica E1 sul Dell libero: copia+sha -> sonda parlata -> MMLU 2000 (banco_ot.bat). Zero token.
set -u; G=@@ROOT:models@@/gguf/tony-tern/E1/Thor-T-E1.gguf; B=Thor-T-E1.gguf; N=E1H; S=@@ROOT:forge@@/stato
log(){ printf '%s\t%s\t%s\n' "$(date +%Y-%m-%dT%H:%M)" "$1" "$2" >> $S/cicli.tsv; sed -i "s#^L10 .*#L10 E1 esperti Hessiana | automatica | $(date +%H:%M) $2#" $S/linee.txt; }
SHA=3ad43f9928e89b4f9e75c0f999b438f20c7c73540ead9c2f644acbb50a89b6a7
scp -q "$G" "aragorn:C:/tcq/thor_mmlu/gguf/$B" || { log ERRORE "E1 copia fallita"; exit 3; }
A=$(ssh aragorn "certutil -hashfile C:\\tcq\\thor_mmlu\\gguf\\$B SHA256" | sed -n 2p | tr -d '\r ')
[ "$A" = "$SHA" ] || { log ERRORE "E1 sha diverso sul Dell"; exit 3; }
log ATTIVO "E1 copiato (sha ok), sonda parlata in corso"
ssh aragorn "del C:\\tcq\\thor_mmlu\\log\\sonda_$N.fine 2>nul & schtasks /create /f /tn sonda_$N /tr \"C:\\tcq\\thor_mmlu\\sonda_parlata.bat $N $B 23:59\" /sc once /st 23:59 /sd 01/01/2030 /rl limited && schtasks /run /tn sonda_$N" >/dev/null || { log ERRORE "sonda schtasks fallito"; exit 3; }
until timeout 30 ssh aragorn "type C:\\tcq\\thor_mmlu\\log\\sonda_$N.fine" 2>/dev/null | grep -q FATTO; do sleep 120; done
scp -q "aragorn:C:/tcq/thor_mmlu/log/sonda_$N.log" $S/sonda_E1H.log
log ATTIVO "sonda E1: $(tail -3 $S/sonda_E1H.log | tr '\n\t' '  ' | cut -c1-120); MMLU 2000 partito"
ssh aragorn "del C:\\tcq\\thor_mmlu\\log\\$N.fine 2>nul & schtasks /create /f /tn thor_mmlu_$N /tr \"C:\\tcq\\thor_mmlu\\banco_ot.bat $N $B\" /sc once /st 23:59 /sd 01/01/2030 /rl limited && schtasks /run /tn thor_mmlu_$N" >/dev/null || { log ERRORE "MMLU schtasks fallito"; exit 3; }
until timeout 30 ssh aragorn "type C:\\tcq\\thor_mmlu\\log\\$N.fine" 2>/dev/null | grep -q FATTO; do sleep 300; done
scp -q "aragorn:C:/tcq/thor_mmlu/log/$N.log" $S/mmlu_E1H.log
F=$(grep -oE 'Final result: [0-9.]+ \+/- [0-9.]+' $S/mmlu_E1H.log | tail -1)
log CHIUSO "MMLU 2000 E1: ${F:-Final result ASSENTE} (criterio >=75,85 e McNemar vs R2B3)"

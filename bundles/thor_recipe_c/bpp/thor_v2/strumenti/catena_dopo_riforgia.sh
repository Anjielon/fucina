#!/bin/bash
# 28/9 — catena automatica dopo la riforgia piena (nata dalla notte del 28/9: GPU ferma 6 h in attesa di un passo umano).
# Nessun passo aspetta una persona:
#   1. aspetta la fine dell'unita' della riforgia;
#   2. SUBITO la GPU torna al lavoro: EB3 riparte dal passo 20 con la riga scritta in $RIGA (aggiornabile a mano senza
#      toccare la catena; se il file manca vale la riga "non conferma" dell'agente, commit a3b4e60);
#   3. in parallelo, su CPU: istantanea + export al bit del GGUF riforgiato (solo se la corsa e' finita con rc=0);
#   4. KLD standard sulla GPU accanto a EB3 (-ub 512), MMLU logprob 2000 sul Dell sotto lucchetto, sonda parlata 3x7;
#   5. verdetto scritto in cicli.tsv col criterio fissato PRIMA (TECNICHE §41): adotta se MMLU >= 74,65 (EB3 p20 theta).
# Lancio: systemd-run --user --unit=catena-dopo-riforgia-d4 --collect -p Nice=10 bash -c "catena_dopo_riforgia.sh > LOG 2>&1"
set -u
UNITA=${UNITA:-riforgia-R2039d4-piena}
NOME=${NOME:-R2039d4}
export RIFORGIA_RADICE=${RIFORGIA_RADICE:-@@ROOT:depot@@/riforgia_d4}
CODICE=${CODICE:-@@ROOT:depot@@/e2e/codice/riforgia-3022c97/thor_v2}
LOGR=${LOGR:-$RIFORGIA_RADICE/piena.log}
G=$RIFORGIA_RADICE/Thor-T-riforgia-$NOME.gguf
RIGA=@@ROOT:forge@@/stato/riga_riavvio_eb3.txt
S=@@ROOT:forge@@/thor_v2/strumenti
CICLI=@@ROOT:forge@@/stato/cicli.tsv
SOGLIA=74.65
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
log() { echo "[$(date '+%F %T')] $*"; }
ciclo() { printf "%s\t%s\t%s\n" "$(date +%Y-%m-%dT%H:%M)" "$1" "$2" >> "$CICLI"; }

log "catena accesa: aspetto la fine di $UNITA"
while systemctl --user is-active --quiet "$UNITA"; do sleep 60; done
RC=$(grep -oE "corsa piena rc=[0-9]+" "$LOGR" | tail -1 | grep -oE "[0-9]+$")
log "$UNITA finita (rc=${RC:-?})"

# ── 2. la GPU torna al lavoro SUBITO: EB3 dal passo 20 ──
if [ -s "$RIGA" ]; then CMD=$(cat "$RIGA"); else
  CMD='TEACHER_MC=@@ROOT:depot@@/e2e_teacher_mc COMMIT=a3b4e60 bash @@ROOT:home@@/claude-pod-lavoro/integrata/thor_v2/strumenti/riavvio_integrato.sh --strati-allenati 20-39 --peso-dominio "it_chat=1,strumenti=1,cot=1,edu=1.5,it_web=1.5,it_wiki=1.5,it_pensiero=1.5" --pv-sonda-dosi --pv-dosi-frazioni 1/4096,1/1024,1/256,1/64,1/16,1/4,1 --pv-ricentra --pv-ricentra-giri 3 --ottimizzatore-theta adamw --lr-f32 5e-5 --lr-lora 4e-5'
fi
rm -f @@ROOT:depot@@/e2e/allena/EB3/NON_RIPARTIRE
log "riavvio EB3: $CMD"
bash -c "$CMD"; R2=$?
log "riavvio EB3 rc=$R2"
ciclo ATTIVO "Catena dopo riforgia: EB3 ripartita dal passo 20 (rc=$R2) appena finita $UNITA (rc=${RC:-?}): la GPU non resta ferma durante export e misure"

[ "${RC:-1}" = 0 ] || { ciclo CHIUSO "⛔ Riforgia $NOME finita con rc=${RC:-?} (fermata dal guardiano o guasto): niente export automatico, EB3 al lavoro"; exit 0; }

# ── 3. istantanea + export (CPU) ──
cd "$CODICE" || exit 2
if [ ! -s "$G" ]; then
  nice -n 15 ./lancia_riforgia_20_39.sh istantanea "$NOME" || { ciclo CHIUSO "⛔ istantanea della riforgia $NOME fallita"; exit 3; }
  nice -n 15 ./lancia_riforgia_20_39.sh esporta "$NOME"   || { ciclo CHIUSO "⛔ export della riforgia $NOME fallito"; exit 3; }
fi
[ -s "$G" ] || { ciclo CHIUSO "⛔ GGUF riforgiato assente dopo l'export"; exit 3; }
log "GGUF pronto: $G ($(stat -c %s "$G") byte)"

# ── 4a. KLD standard accanto a EB3 (-ub 512, come il sorvegliante) ──
B=@@ROOT:models@@/build/llamacpp-tq1-k1024/build/bin; USC=@@ROOT:models@@/gguf/tony-tern
K=@@ROOT:lavoro@@/e2e/fedelta/kld/kld_riforgia_$NOME.log
EB3=$(systemctl --user show e2e-allena-EB3 -p MainPID --value)
if GPU_ESCLUDI_PID="$EB3" "$S/gpu_occupata.sh" >/dev/null 2>&1; then
  LD_LIBRARY_PATH=$B "$B/llama-perplexity" -m "$G" -ngl 999 -c 4096 -b 2048 -ub 512 -dev Vulkan0 \
    -f @@ROOT:odino@@/collaudo/corpus/wiki.test.raw --chunks 20 \
    --kl-divergence-base $USC/taratura_teacherQ8_wiki_c4096_ch20.kld --kl-divergence > "$K" 2>&1
  ciclo CHIUSO "KLD standard riforgia $NOME: $(grep -E "Mean *KLD|Same top p" "$K" | tr -s ' ' | tr '\n' ' ') (B 0,3506 · Q2_K 0,2201 · EMA16 vedi §36)"
else
  log "KLD saltata: sulla GPU non c'e' la sola EB3"
fi

# ── 4b. MMLU logprob 2000 sul Dell (sotto lucchetto) ──
# 28/9: niente collegamenti fisici — $G sta su @@ROOT:depot@@ e $USC su @@ROOT:models@@, dischi diversi (ln -f fallisce:
# «Collegamento tra dispositivi non valido», trovato dalla linea A). Si passa $G cosi' com'e', e il nostro file si toglie
# dal Dell a mano: aragorn_valuta_e_pulisci.sh di master pulisce solo Thor-T-EB2-*.gguf.
NOMEM=riforgia-$NOME; L=@@ROOT:lavoro@@/e2e/allena/aragorn_$NOMEM.log; BG=$(basename "$G")
dell_togli() { timeout 60 ssh aragorn "powershell -NoProfile -Command \"Remove-Item -ErrorAction SilentlyContinue C:\\tcq\\thor_mmlu\\gguf\\$BG\"" >/dev/null 2>&1; }
"$S/aragorn_lucchetto.sh" "$S/aragorn_valuta_e_pulisci.sh" "$NOMEM" "$G" "$L"; RCM=$?
dell_togli
if [ "$RCM" = 0 ]; then
  M=$(grep -oE "Final result: [0-9.]+" "$L" | tail -1 | grep -oE "[0-9.]+$")
  V=$(python3 -c "print('ADOTTA' if float('${M:-0}') >= $SOGLIA else 'NON adotta')")
  ciclo CHIUSO "MMLU riforgia $NOME (Dell, 2000): ${M:-?} -> $V (criterio scritto prima: >= $SOGLIA, EB3 p20 theta). Se ADOTTA: EB4 dal GGUF riforgiato (TECNICHE §41)"
else
  ciclo CHIUSO "⛔ MMLU della riforgia $NOME sul Dell fallita ($L)"
fi

# ── 4c. sonda parlata 3x7 (Dell, sotto lucchetto) ──
eval "$(sed -n '/^log() /,/^}/p;/^esegui_sonda() /,/^}/p' "$S/sonda_EB3.sh")"
OUT=@@ROOT:forge@@; USC=$USC
export -f esegui_sonda log; export OUT USC
"$S/aragorn_lucchetto.sh" bash -c "esegui_sonda '$NOMEM' '$G'"    # esegui_sonda copia $G sul Dell e poi lo toglie
log "catena finita"

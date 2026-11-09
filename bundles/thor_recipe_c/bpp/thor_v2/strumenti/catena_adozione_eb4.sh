#!/bin/bash
# 28/9 — adozione di EB4 dopo la riforgia (TECNICHE_E2E_PV.md §41). Chiamata dalla catena dopo il verdetto MMLU del
# riforgiato (catena_dopo_riforgia.sh, 4b), in un'unita' sua (non blocca la sonda parlata della catena):
#   - MMLU < SOGLIA (74,65 = EB3 p20 theta, criterio scritto PRIMA): non fa NULLA, EB3 continua;
#   - MMLU >= SOGLIA:
#       1. EB4 a secco (lancia_eb4.sh PROVA=1: codice congelato, parser vero, ricetta, righe sigillate, indipendenza del
#          maestro, struttura del GGUF) PRIMA di toccare EB3: se EB4 non e' pronta, EB3 non perde nemmeno un passo;
#       2. aspetta il CONFINE di passo di EB3 (ripresa_ultima.json cambiato: la ripresa del passo e' scritta), poi
#          DOPO_RIPRESA_S secondi per la potatura; se EB3 non ci arriva in CONFINE_MAX_S non ferma niente;
#       3. freno NON_RIPARTIRE nella cartella di EB3 (riavvio_integrato.sh non la rilancia piu' sopra EB4) e stop
#          dell'UNITA' e2e-allena-EB3 (systemctl, mai kill per nome); verifica che sia ferma;
#       4. una corsia pesante per volta: nessuna e2e-allena-* viva e GPU libera (gpu_occupata.sh --attendi GPU_MAX_S:
#          un KLD del sorvegliante sulla GPU si aspetta, non si scavalca);
#       5. lancia EB4 (lancia_eb4.sh, COMMIT_EB4) e scrive l'esito nei cicli;
#       6. (verifica avversaria 28/9) sorveglia EB4 fino alla sua PRIMA ripresa: il controllo di lancia_eb4.sh guarda
#          l'unita' 20 s dopo l'avvio, ma EB4 carica il GGUF, fa l'held-out 0 e il primo passo (~2 h [S]) prima di avere
#          qualcosa da riprendere. Se EB4 muore prima (o il lancio fallisce dopo lo stop di EB3), EB3 torna al lavoro DA
#          SOLA dalla sua ripresa al confine (rimetti_eb3: NON_RIPARTIRE tolto + riga di riavvio, solo con GPU libera e
#          nessuna corsa E2E viva): senza, la GPU restava ferma tutta la notte con «EB4 LANCIATA» nei cicli.
# Uscite: 0 fatto o niente da fare, 2 MMLU illeggibile, 3 EB4 non pronta, 4 confine non raggiunto, 5 EB3 non si ferma,
#         6 GPU/corse E2E occupate dopo lo stop, 7 lancio di EB4 fallito, 8 EB4 morta prima della prima ripresa
#         (in 7 e 8 EB3 e' rimessa al lavoro se si puo'; il ciclo dice se e' riuscito), 9 freno di una persona su EB3
#         (NON_RIPARTIRE non dell'adozione: la catena lo toglie quando rilancia EB3, quindi e' stato messo DOPO da qualcuno
#         che vuole EB3 ferma o la GPU per se': non si sovrascrive e non si tocca niente).
# Uso (dalla catena): MMLU=<Final result> GGUF=<riforgiato> bash catena_adozione_eb4.sh
set -u
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
MMLU=${MMLU:-${1:-}}
SOGLIA=${SOGLIA:-74.65}
NOME=${NOME:-R2039d4v}
GGUF=${GGUF:-@@ROOT:depot@@/riforgia_d4/Thor-T-riforgia-$NOME.gguf}
U3=${UNITA_EB3:-e2e-allena-EB3}
D3=${EB3_DIR:-@@ROOT:depot@@/e2e/allena/EB3}
U4=${UNITA_EB4:-e2e-allena-EB4}
COMMIT_EB4=${COMMIT_EB4:?COMMIT_EB4 = il commit del ramo integrata da congelare per EB4}
ALBERO=${ALBERO:-@@ROOT:home@@/claude-pod-lavoro/integrata}
LANCIA=${LANCIA_EB4:-}                      # vuoto = lancia_eb4.sh estratto dal COMMIT_EB4 (lo stesso codice della corsa)
GPU=${GPU_OCCUPATA:-@@ROOT:forge@@/thor_v2/strumenti/gpu_occupata.sh}
CICLI=${CICLI:-@@ROOT:forge@@/stato/cicli.tsv}
RIGA_EB3=${RIGA_EB3:-@@ROOT:forge@@/stato/riga_riavvio_eb3.txt}
CONFINE_MAX_S=${CONFINE_MAX_S:-10800}      # un passo di EB3 ~68 min + held-out pieno ~37: 3 h bastano
GPU_MAX_S=${GPU_MAX_S:-7200}
POLL_S=${POLL_S:-30}
DOPO_RIPRESA_S=${DOPO_RIPRESA_S:-20}
D4=${EB4_DIR:-${E2E_ALLENA_DIR:-@@ROOT:depot@@/e2e/allena}/EB4}   # cartella di EB4 (lancia_eb4.sh: RADICE/CORSA)
RIPRESA_EB4_MAX_S=${RIPRESA_EB4_MAX_S:-21600}   # 6 h: caricamento + held-out 0 + primo passo (~2 h [S]), poi si smette
log() { echo "[$(date '+%F %T')] $*"; }
ciclo() { printf "%s\t%s\t%s\n" "$(date +%Y-%m-%dT%H:%M)" "$1" "$2" >> "$CICLI"; }
e2e_vive() {           # unita' e2e-allena-* vive; fallisce (rc 1) se systemctl non risponde: mai «nessuna» alla cieca
  local e
  e=$(systemctl --user list-units --state=active,activating,deactivating,reloading --plain --no-legend 'e2e-allena-*') || return 1
  awk '{print $1}' <<< "$e" | tr '\n' ' ' | sed 's/ *$//'
}
RIMETTI_EB3="rm $D3/NON_RIPARTIRE && bash -c \"\$(cat $RIGA_EB3)\""
MARCA="28/9 adozione EB4"                  # inizio del NON_RIPARTIRE scritto QUI: rimetti_eb3 toglie solo questo
freno_altrui() {    # NON_RIPARTIRE di EB3 messo da qualcun altro (non comincia con la nostra MARCA)
  [ -e "$D3/NON_RIPARTIRE" ] && ! head -1 "$D3/NON_RIPARTIRE" 2>/dev/null | grep -q "^$MARCA"
}
ferma_se_freno_altrui() {
  if freno_altrui; then
    log "⛔ freno di una persona su EB3: $(head -c 200 "$D3/NON_RIPARTIRE")"
    ciclo CHIUSO "⛔ Adozione EB4 SOSPESA: MMLU $MMLU >= $SOGLIA ma su EB3 c'e' un NON_RIPARTIRE non dell'adozione ($(head -c 120 "$D3/NON_RIPARTIRE" | tr '\n\t' '  ')): nessuna unita' toccata. A mano, tolto il freno: MMLU=$MMLU GGUF=$GGUF COMMIT_EB4=$COMMIT_EB4 bash $0"
    exit 9
  fi
}
stato_unita() { systemctl --user show "$1" -p ActiveState --value 2>/dev/null; }   # vuoto se systemctl non risponde
rimetti_eb3() {   # <perche'>: EB4 non e' partita o e' morta prima della sua prima ripresa -> EB3 dalla sua ripresa
  local perche=$1 vive occ
  if [ -e "$D4/NON_RIPARTIRE" ] || ! head -1 "$D3/NON_RIPARTIRE" 2>/dev/null | grep -q "^$MARCA"; then
    # fermata VOLUTA da una persona (freno di EB4) o freno di EB3 che non e' il nostro: non si decide al suo posto
    ciclo CHIUSO "⛔ Adozione EB4: $perche; freno di una persona presente ($D4/NON_RIPARTIRE o NON_RIPARTIRE di EB3 non dell'adozione): EB3 NON rimessa. A mano: $RIMETTI_EB3"
    return 1
  fi
  if ! vive=$(e2e_vive); then
    ciclo CHIUSO "⛔ Adozione EB4: $perche; systemctl non risponde: EB3 NON rimessa (passo ${P3:-?}). A mano: $RIMETTI_EB3"
    return 1
  fi
  if [ -n "$vive" ]; then
    ciclo CHIUSO "⛔ Adozione EB4: $perche; corse E2E vive ($vive): EB3 NON rimessa (passo ${P3:-?}). A mano: $RIMETTI_EB3"
    return 1
  fi
  if ! occ=$(bash "$GPU" --attendi "$GPU_MAX_S"); then
    ciclo CHIUSO "⛔ Adozione EB4: $perche; GPU occupata ($(head -c 160 <<< "$occ")): EB3 NON rimessa (passo ${P3:-?}). A mano: $RIMETTI_EB3"
    return 1
  fi
  if [ ! -s "$RIGA_EB3" ]; then
    ciclo CHIUSO "⛔ Adozione EB4: $perche; riga di riavvio $RIGA_EB3 assente: EB3 NON rimessa (passo ${P3:-?})"
    return 1
  fi
  log "rimetto EB3 al lavoro ($perche): $(cat "$RIGA_EB3")"
  rm -f "$D3/NON_RIPARTIRE"
  bash -c "$(cat "$RIGA_EB3")"; local r=$?
  sleep "${ATTESA_EB3_S:-20}"
  if [ "$r" = 0 ] && systemctl --user is-active --quiet "$U3"; then
    ciclo CHIUSO "⛔ Adozione EB4: $perche -> EB3 RIMESSA al lavoro da sola dalla ripresa del passo ${P3:-?} (riga $RIGA_EB3): GPU non ferma. EB4 da rivedere a mano (cartella $D4, journalctl --user -u $U4)"
    return 0
  fi
  ciclo CHIUSO "⛔ Adozione EB4: $perche; riavvio di EB3 fallito (rc $r, $U3 $(stato_unita "$U3")): GPU FERMA. A mano: $RIMETTI_EB3"
  return 1
}

log "adozione EB4: MMLU del riforgiato '$MMLU', soglia $SOGLIA, GGUF $GGUF"
# ── 1. verdetto ──
if ! [[ "$MMLU" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  log "⛔ MMLU '$MMLU' non numerico: nessuna azione"
  ciclo CHIUSO "⛔ Adozione EB4: MMLU del riforgiato '$MMLU' illeggibile, nessuna azione (EB3 continua)"
  exit 2
fi
if ! awk -v m="$MMLU" -v s="$SOGLIA" 'BEGIN { exit !(m + 0 >= s + 0) }'; then
  log "MMLU $MMLU < $SOGLIA: NON si adotta, EB3 continua (nessuna unita' toccata)"
  ciclo CHIUSO "Adozione EB4: NO (MMLU del riforgiato $MMLU < $SOGLIA, criterio scritto prima): EB3 continua, nessuna corsa toccata"
  exit 0
fi
if systemctl --user is-active --quiet "$U4"; then log "$U4 gia' attiva: niente da fare"; exit 0; fi
ferma_se_freno_altrui
if [ -z "$LANCIA" ]; then                   # lo script di lancio del commit congelato, non quello dell'albero di oggi
  LANCIA=@@ROOT:depot@@/e2e/codice/lancia_eb4-$COMMIT_EB4.sh
  if ! git -C "$ALBERO" show "$COMMIT_EB4:thor_v2/strumenti/lancia_eb4.sh" > "$LANCIA.tmp" 2>/dev/null || [ ! -s "$LANCIA.tmp" ]; then
    rm -f "$LANCIA.tmp"
    ciclo CHIUSO "⛔ Adozione EB4: lancia_eb4.sh assente nel commit $COMMIT_EB4 di $ALBERO: EB3 continua"
    exit 3
  fi
  mv "$LANCIA.tmp" "$LANCIA"
fi

# ── 2. EB4 a secco PRIMA di toccare EB3 ──
log "EB4 a secco (PROVA=1)"
if ! PROVA=1 GGUF="$GGUF" COMMIT="$COMMIT_EB4" bash "$LANCIA"; then
  log "⛔ EB4 non pronta: EB3 NON si ferma"
  ciclo CHIUSO "⛔ Adozione EB4: MMLU $MMLU >= $SOGLIA ma EB4 NON pronta (lancia_eb4.sh a secco rosso, vedi il log dell'adozione): EB3 continua"
  exit 3
fi

# ── 3. confine di passo di EB3 ──
CONFINE=gia_ferma
if systemctl --user is-active --quiet "$U3"; then
  M3=$D3/ripresa_ultima.json
  prima=$(cat "$M3" 2>/dev/null)
  log "aspetto il confine di passo di $U3 (ripresa di adesso: $(tr -d '\n ' <<< "$prima" | cut -c1-120))"
  t0=$(date +%s)
  while :; do
    if [ "$(cat "$M3" 2>/dev/null)" != "$prima" ]; then CONFINE=ripresa; break; fi
    if ! systemctl --user is-active --quiet "$U3"; then CONFINE=ferma_da_sola; break; fi
    if [ $(( $(date +%s) - t0 )) -ge "$CONFINE_MAX_S" ]; then
      log "⛔ $U3 non arriva al confine in ${CONFINE_MAX_S} s: non la fermo, EB4 non parte"
      ciclo CHIUSO "⛔ Adozione EB4: MMLU $MMLU >= $SOGLIA ma EB3 senza nuova ripresa in ${CONFINE_MAX_S} s: EB3 NON fermata, EB4 non lanciata (lancio a mano: MMLU=$MMLU GGUF=$GGUF COMMIT_EB4=$COMMIT_EB4 bash $0)"
      exit 4
    fi
    sleep "$POLL_S"
  done
  if [ "$CONFINE" = ripresa ]; then
    log "confine: nuova ripresa $(tr -d '\n ' < "$M3" | cut -c1-120); ${DOPO_RIPRESA_S} s per la potatura"
    sleep "$DOPO_RIPRESA_S"
  else
    log "$U3 si e' fermata da sola prima del confine: vale la sua ultima ripresa"
  fi
fi

# ── 4. freno e stop di EB3 (unita', mai kill per nome) ──
ferma_se_freno_altrui                         # messo da qualcuno durante l'attesa del confine: non si sovrascrive
echo "$MARCA ($(date -Iseconds)): MMLU del riforgiato $MMLU >= $SOGLIA, EB3 fermata al confine ($CONFINE) per lanciare EB4 dal GGUF riforgiato. Per rimetterla al lavoro: $RIMETTI_EB3" > "$D3/NON_RIPARTIRE"
if systemctl --user is-active --quiet "$U3"; then systemctl --user stop "$U3"; fi
if systemctl --user is-active --quiet "$U3"; then
  log "⛔ $U3 ancora attiva dopo lo stop"
  ciclo CHIUSO "⛔ Adozione EB4: $U3 non si ferma (NON_RIPARTIRE scritto): EB4 non lanciata"
  exit 5
fi
P3=$(tr -d '\n ' < "$D3/ripresa_ultima.json" 2>/dev/null | grep -oE '"passo":[0-9]+' | grep -oE '[0-9]+$')
log "$U3 ferma ($CONFINE), ultima ripresa al passo ${P3:-?}; NON_RIPARTIRE scritto"

# ── 5. una corsia pesante per volta ──
if ! VIVE=$(e2e_vive); then
  ciclo CHIUSO "⛔ Adozione EB4: systemctl non risponde dopo lo stop di EB3: EB4 NON lanciata. EB3 ferma al passo ${P3:-?}; per rimetterla: $RIMETTI_EB3"
  exit 6
fi
if [ -n "$VIVE" ]; then
  log "⛔ corse E2E ancora vive: $VIVE"
  ciclo CHIUSO "⛔ Adozione EB4: corse E2E vive dopo lo stop di EB3 ($VIVE): EB4 NON lanciata. EB3 ferma al passo ${P3:-?}; per rimetterla: $RIMETTI_EB3"
  exit 6
fi
log "aspetto la GPU libera (fino a ${GPU_MAX_S} s)"
if ! OCC=$(bash "$GPU" --attendi "$GPU_MAX_S"); then
  log "⛔ GPU occupata dopo ${GPU_MAX_S} s: $OCC"
  ciclo CHIUSO "⛔ Adozione EB4: GPU occupata ${GPU_MAX_S} s dopo lo stop di EB3 ($(head -c 160 <<< "$OCC")): EB4 NON lanciata. EB3 ferma al passo ${P3:-?}; per rimetterla: $RIMETTI_EB3"
  exit 6
fi

# ── 6. lancio di EB4 ──
log "lancio di EB4 (COMMIT $COMMIT_EB4)"
if GGUF="$GGUF" COMMIT="$COMMIT_EB4" bash "$LANCIA"; then
  ciclo ATTIVO "EB4 LANCIATA dal GGUF riforgiato $(basename "$GGUF") (MMLU $MMLU >= $SOGLIA): corsa NUOVA, LoRA init zero lr 1e-5, codice integrata $COMMIT_EB4; EB3 ferma al confine del passo ${P3:-?} con NON_RIPARTIRE"
  log "adozione fatta: sorveglio $U4 fino alla prima ripresa ($D4/ripresa_ultima.json, al piu' ${RIPRESA_EB4_MAX_S} s)"
else
  RC=$?
  log "⛔ lancio di EB4 fallito (rc $RC) dopo lo stop di EB3"
  rimetti_eb3 "lancio di EB4 fallito (rc $RC) dopo lo stop di EB3"
  exit 7
fi

# ── 7. EB4 fino alla prima ripresa: se muore prima, EB3 torna al lavoro (la GPU non resta ferma) ──
t0=$(date +%s)
while :; do
  if [ -s "$D4/ripresa_ultima.json" ]; then
    ciclo CHIUSO "EB4: prima ripresa scritta ($(tr -d '\n ' < "$D4/ripresa_ultima.json" | grep -oE '"passo":[0-9]+')) dopo $(( $(date +%s) - t0 )) s: EB4 al lavoro"
    log "EB4: prima ripresa scritta"
    break
  fi
  s=$(stato_unita "$U4")
  if [ "$s" = inactive ] || [ "$s" = failed ]; then       # vuoto = systemctl muto: si riprova, mai «morta» alla cieca
    log "⛔ $U4 $s prima della sua prima ripresa"
    rimetti_eb3 "$U4 $s dopo $(( $(date +%s) - t0 )) s, prima della sua prima ripresa"
    exit 8
  fi
  if [ $(( $(date +%s) - t0 )) -ge "$RIPRESA_EB4_MAX_S" ]; then
    log "$U4 viva ($s) ma senza ripresa dopo ${RIPRESA_EB4_MAX_S} s: smetto di sorvegliare"
    ciclo CHIUSO "EB4 viva ma senza prima ripresa dopo ${RIPRESA_EB4_MAX_S} s (attesa del teacher? journalctl --user -u $U4): nessuna azione"
    break
  fi
  sleep "$POLL_S"
done

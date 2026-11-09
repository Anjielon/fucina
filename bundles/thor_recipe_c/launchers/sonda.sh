#!/bin/bash
# SONDA PARLATA — prima di ogni banco, si verifica che il modello PARLI.
#
# ⛔ Perche' esiste (Angelo, 30/8: «hai provato se parla?»). I banchi NON fanno
#    parlare il modello: HellaSwag e Winogrande PESANO i finali, il modello non
#    emette una parola. Il 28/8 GPTQ sul piano-1 ruppe il ragionamento di ODINO
#    v3.3 e l'errore d'uscita, misurato sul corpus, NON se ne accorse. Avere
#    quattro verifiche verdi e zero prove che il modello sappia rispondere e'
#    gia' successo.
#
# ⛔ Sostituisce `sonda_v4b.sh`, che era inservibile per tre motivi insieme:
#    girava via `ssh aragorn` su Windows, aveva il modello SCRITTO DENTRO
#    (Tony-v4-B-ok.gguf) e teneva il thinking SPENTO — cioe' non provava proprio
#    la cosa che il 28/8 si era rotta.
#
# uso:  sonda_parlata.sh <modello.gguf> [nome] [--senza-thinking]
set -u
# 26/9 20:3x: senza queste systemd-run --user fallisce in silenzio fuori da una sessione interattiva (la sonda di B e'
# rimasta 7 minuti ad aspettare un server mai partito).
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)} DBUS_SESSION_BUS_ADDRESS=${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/$(id -u)/bus}

# ⛔⛔ 20/9 (due corse concorrenti nella stessa nottata, entrambe hanno
#    corrotto l'esito): l'unita' systemd sotto e' SEMPRE "adhoc-sonda",
#    fissa, MAI parametrizzata dal nome del modello — due invocazioni
#    qualunque, anche su modelli diversi, condividono la stessa unita' e si
#    spengono a vicenda a meta' richiesta (spegni() nel trap EXIT dell'una
#    colpisce il server dell'altra). Fix strutturale: un lock file blocca
#    (ATTENDE, non fallisce) una seconda invocazione finche' la prima non
#    e' finita — cosi' la corsa non puo' piu' accadere, qualunque cosa la
#    lanci (agente, mano, ping).
exec 9>/tmp/sonda_parlata.lock
flock 9

M="${1:?modello richiesto}"
NOME="${2:-$(basename "$M" .gguf)}"
THINK=true
[ "${3:-}" = "--senza-thinking" ] && THINK=false

# ⚠️ SERVE IL FORK TQ1: i secondi piani (ffn_*_exps2) sono la nostra estensione
#    ternaria; il motore standard li ignora e fallisce con «wrong number of tensors».
B=@@ROOT:models@@/build/thor-c0t/build-c0t/bin
PORTA=11458
# ⚠️ L'unita' DEVE chiamarsi `adhoc-*`: l'arbitro della VRAM risponde
#    «VRAM insufficiente» alle unita' che non riconosce, a modello gia' carico,
#    e si finisce a girare a vuoto senza capire perche'.
UNITA=adhoc-sonda
L=@@ROOT:forge@@/sonda_${NOME}.log
ESITO=@@ROOT:forge@@/sonda_${NOME}.txt
# ⛔ L0 13/10: giudice UNICO (risposta finale dichiarata, non un numero qualunque nel testo) e JSON grezzo di
#    ogni risposta (contenuto, pensiero, finish_reason, token) — senza, nessuna sonda si puo' rigiudicare.
GIUDICE=${SONDA_GIUDICE:-@@ROOT:forge@@/thor_v2/leve_flusso/sonda_giudice.py}
GREZZE=@@ROOT:forge@@/sonda_${NOME}_grezze.jsonl
MAXTOK=${SONDA_MAX_TOKENS:-2000}
[ -f "$GIUDICE" ] || { echo "⛔ manca il giudice $GIUDICE"; exit 1; }
[ -s "$GREZZE" ] && mv "$GREZZE" "$GREZZE.prec-$(date +%Y%m%d_%H%M%S)"   # mai sovrascrivere grezzi

exec > >(tee "$L") 2>&1
echo "════════════════════════════════════════════════════════════"
echo "SONDA PARLATA · $NOME · thinking=$THINK · $(date '+%d/%m %H:%M:%S')"
echo "modello: $M ($(du -h "$M" 2>/dev/null | cut -f1))"
echo "════════════════════════════════════════════════════════════"

[ -f "$M" ] || { echo "⛔ modello inesistente: $M"; exit 1; }

spegni(){ systemctl --user stop "$UNITA" 2>/dev/null; systemctl --user reset-failed "$UNITA" 2>/dev/null; }
trap spegni EXIT

# ⛔ L0 13/10 (ricerche/C_SONDA_DIAGNOSI_1013.md §2.3): systemd-run NON eredita l'ambiente del chiamante.
#    Chi esportava GGML_VK_TQ1R_Q8/GGML_VK_DISABLE_GRAPH_OPTIMIZE alla sonda credeva di misurare il cammino int8
#    di MMLU/KL e invece misurava il f32 con l'ottimizzazione del grafo accesa. Ora si passano ESPLICITAMENTE con
#    -E; default 1 (= cammino di deployment, quello di MMLU/KL/servi()); chi vuole il f32 esporta =0.
#    I motori senza la variante int8 la ignorano.
VK_Q8=${GGML_VK_TQ1R_Q8:-1}; VK_NOGO=${GGML_VK_DISABLE_GRAPH_OPTIMIZE:-1}
echo "server: GGML_VK_TQ1R_Q8=$VK_Q8 GGML_VK_DISABLE_GRAPH_OPTIMIZE=$VK_NOGO (passate con systemd-run -E) · max_tokens $MAXTOK"
spegni; sleep 2
systemd-run --user --unit "$UNITA" --collect -E GGML_VK_TQ1R_Q8=$VK_Q8 -E GGML_VK_DISABLE_GRAPH_OPTIMIZE=$VK_NOGO \
  env LD_LIBRARY_PATH=$B $B/llama-server \
  --model "$M" --alias sonda -ngl 999 --ctx-size 8192 --parallel 1 --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 --host 127.0.0.1 --port $PORTA --threads 8 \
  --temp 0.6 --min-p 0.03 --top-p 0.9 --jinja --reasoning-budget 4096 ${SONDA_EXTRA:-} >/dev/null 2>&1   # 22/10: SONDA_EXTRA (es. --lora file.gguf)

for i in $(seq 1 90); do
  curl -s -m2 "http://127.0.0.1:$PORTA/health" 2>/dev/null | grep -q ok && break
  sleep 5
  [ "$i" = "90" ] && { echo "⛔ il modello non si accende in 7 minuti"; exit 1; }
done
echo "acceso dopo ~$((i*5))s · VRAM $(rocm-smi --showmeminfo vram 2>/dev/null | grep -i used | head -1 | awk -F': ' '{printf "%.1f GiB", $NF/1073741824}')"
echo

# Tre domande a risposta VERIFICABILE, scelte perche' hanno gia' fatto cadere
# qualcuno: l'aritmetica multi-passo (senza thinking questa famiglia dava 17.45
# invece di 17.55), la trappola logica dei fratelli, la manipolazione di
# caratteri (il MoE scriveva ECAN invece di ENAC).
#
# ⛔⛔⛔ 22/9 03:26 — SEED FISSO = FALSI NEGATIVI. Per tutta la notte 21→22/9 la
#    sonda chiedeva con `seed:7` e basta. Misurato sul modello BASE intatto
#    (Thor-v20-downaggr3) variando solo il seed: la domanda aritmetica esce
#    giusta 3 volte su 7 (18.05 ✗ · 18.05 ✗ · 17.55 ✓ · 17.55 ✓ · 16.55 ✗ ·
#    18.05 ✗ · 17.55 ✓). Con seed 7 il base la azzecca PER CASO. Quindi un
#    candidato che con seed 7 sbaglia non e' «peggiore del base»: e' un lancio
#    di dado. Almeno due leve (secondo piano densi a scala piena, rotazione+
#    Hessiane grandi) sono state SCARTATE solo cosi', mai portate a PPL/MMLU.
#    Cura strutturale: OGNI domanda si chiede con TUTTI i seed sotto (lo
#    stesso insieme usato per la misura sul base, cosi' i numeri si
#    confrontano), e il verdetto e' un TASSO, da leggere ACCANTO al tasso del
#    modello base sugli stessi seed — mai da solo. Costa 21 richieste invece
#    di 3: e' il prezzo di una misura, non una scorciatoia da riprendere.
SEEDS="${SONDA_SEEDS:-1 2 3 7 42 100 999}"
NSEED=$(echo $SEEDS | wc -w)

chiedi(){ # $1 domanda  $2 dominio (D1/D2/D3, attesa nel giudice)  $3 seed
  local risposta corpo
  corpo=$(DOM="$1" SEED="$3" MAXTOK="$MAXTOK" THINK="$THINK" python3 -c "
import json, os
print(json.dumps({'model': 's', 'stream': False, 'max_tokens': int(os.environ['MAXTOK']), 'temperature': 0.6,
                  'seed': int(os.environ['SEED']), 'messages': [{'role': 'user', 'content': os.environ['DOM']}],
                  'chat_template_kwargs': {'enable_thinking': os.environ['THINK'] == 'true'}}))")
  # ⚠️ Con i modelli reasoning il contenuto puo' essere vuoto mentre il pensiero e' pieno: il giudice distingue
  #    TRONCATA (finish_reason=length: il tetto max_tokens conta anche il pensiero) da MUTA (fine regolare, vuoto).
  risposta=$(curl -s -m 300 "http://127.0.0.1:$PORTA/v1/chat/completions" -H 'Content-Type: application/json' -d "$corpo" \
    | python3 "$GIUDICE" giudica --dominio "$2" --seed "$3" --grezzo "$GREZZE")
  IFS='|' read -r esito npens tps valore testo <<<"$risposta"
  # ⛔ Il dettaglio va su STDERR: chi chiama cattura lo stdout con $(...) per
  #    leggere il verdetto, e una printf su stdout sparirebbe dentro la
  #    cattura — la sonda direbbe l'esito senza mostrare cosa ha risposto.
  printf "  seed %4s  %-8s pensiero %5s car · %5s tok/s · [%s] %s\n" "$3" "$esito" "$npens" "$tps" "$valore" "$testo" >&2
  echo "$esito"
}

# domande e attese: UNA fonte, il giudice (DOMINI in sonda_giudice.py)
muti=0; mute=0; tronc=0; guasti=0; giusti_tot=0; TASSI=""
n=0
for D in D1 D2 D3; do
  n=$((n+1))
  dom=$(python3 "$GIUDICE" domanda $D)
  echo "── domanda $n · $NSEED seed ($SEEDS) ────────────────────────────"
  g=0
  for seed in $SEEDS; do
    e=$(chiedi "$dom" "$D" "$seed" | tail -1)
    case "$e" in
      OK) g=$((g+1));; MUTA) mute=$((mute+1));; TRONCATA) tronc=$((tronc+1));; SBAGLIA) ;; *) guasti=$((guasti+1));;
    esac
  done
  echo "  → domanda $n: giuste $g/$NSEED"
  TASSI="$TASSI D$n=$g/$NSEED"
  giusti_tot=$((giusti_tot+g))
done
TOT=$((3*NSEED))
muti=$((mute+tronc+guasti))   # «mute N/21» resta il conteggio delle NON-risposte (i chiamanti lo leggono cosi')
DETTAGLIO="troncate $tronc · mute vere $mute · guasti $guasti"

echo
echo "── verdetto ───────────────────────────────────────────────"
if [ "$muti" -gt 0 ]; then
  V="⛔ MUTO su $muti risposte su $TOT ($DETTAGLIO; giuste $giusti_tot/$TOT) — NON portarlo al banco: il banco non se ne accorgerebbe"
  RC=2
elif [ "$giusti_tot" -eq "$TOT" ]; then
  V="✅ PARLA e risponde giusto $TOT/$TOT su tutti i seed — via libera al banco"
  RC=0
else
  V="⚠️ parla: giuste $giusti_tot/$TOT ($TASSI ) — il tasso va letto ACCANTO a quello del modello base sugli stessi seed (riferimento base v20-downaggr3 22/9: D1=3/7); da solo non e' un verdetto"
  RC=1
fi
echo "$V"
{ echo "$NOME  thinking=$THINK  seed=$SEEDS  $(date '+%d/%m %H:%M')"
  echo "$V"
  echo "giuste $giusti_tot/$TOT ·$TASSI · mute $muti/$TOT ($DETTAGLIO) · giudice L0 risposta finale · grezzi $GREZZE"; } > "$ESITO"
echo "giuste $giusti_tot/$TOT ·$TASSI · mute $muti/$TOT ($DETTAGLIO)"
echo "esito in $ESITO · grezzi in $GREZZE"
exit $RC

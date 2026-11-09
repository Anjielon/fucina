#!/bin/bash
# 13/10/2026 — OBIETTIVO C, «C-ONPOLICY-FEDELE»: distillazione on-policy (OPD) col PROTOCOLLO del lavoro.
# Lavoro: arXiv 2609.26708 (Chen et al., «Train Where the Quantized Model Goes»); codice ufficiale MingZwhy/QAOPD
#   (clone: @@ROOT:models@@/gguf/tony-tern/AUDIT_1013/qaopd/repo, commit 9a64cdb8): scripts/opd/run_math.sh.
# Nasce dall'audit ricerche/AUDIT_1013/onpolicy.md (+ SINTESI.md §5): c_onpolicy_1012.sh era fedele nello stimatore
#   (k1, allineamento maestro/prefissi verificati su CPU) ma NON nel protocollo; i suoi 2 bug (giuste() e local L F) sono
#   gia' corretti la' e qui. REGOLA: solo PTQ + distillazione LEGGERA su GPU di casa; nessuna GPU in PROVA.
#
# ── COSA CAMBIA rispetto a c_onpolicy_1012.sh (l'audit, punto 7) ──
#   1 PREMIO STRETTO = premio ufficiale (opd/rewards/math_mixed_reward.py, copia identica sha 39d6d380 in
#     sorgenti_forgia/obiettivo_c/onpolicy_fedele, argomenti di run_math.sh: REQUIRE_HASH_REWARD=True, 0 senza
#     '####'/\boxed, -0,5 se > 3000 caratteri senza risposta). Prova CPU: sul giro 1 da' 0,419 -> 0,267 e cambia il
#     segno dei vantaggi in 39/80 gruppi (= audit). Vantaggi GRPO come verl (std non distorta, eps 1e-6).
#   2 ROLLOUT NUOVI: ogni GIRO genera le SUE completioni con lo studente di ADESSO (GGUF esportato dal giro prima), le
#     usa UNA volta sola (= un lotto per passo, PPO_MINI_BATCH = TRAIN_BATCH del lavoro). Un giro = OPD_PASSI_GIRO
#     (default 2) passi dell'ottimizzatore: VINCOLO del nostro allenatore, dichiarato, non scelto: (a) la ripresa di
#     e2e_allena rifiuta un corpus/maestro diverso (e2e_allena.py: «la ripresa viene da un altro teacher»), quindi ogni
#     giro e' una corsa NUOVA dal GGUF del giro prima (momenti di Adam e EMA ripartono a ogni giro; il primo passo e'
#     ~lr x segno, come gia' misurato al giro 1: 99,8 % delle scale mosse di >= 0,99 lr); (b) il passo V sui CODICI ha
#     bisogno di un passo prima (curvatura e sonda sul micro-lotto del passo dopo): con 1 passo a giro i codici non si
#     muoverebbero MAI. Con 2: il secondo passo usa rollout vecchi di UN passo (l'audit: «a ogni passo, o almeno ogni 2»).
#     Prima: 8 passi da un solo lotto (fino a 7 passi stantii).
#   3 PASSI: braccio fedele = 80 (run_math.sh: STEPS 80 per 1,7B, SAVE_FREQ 10, scelto s70; 120 per il 4B, scelto s60).
#     Un passo = 4 righe da 4096 = ~39 completioni (giro 1: 316 in 32 righe) contro 8 prompt x 4 = 32 del lavoro.
#   4 CANCELLO ogni 10 passi (= SAVE_FREQ) e all'ultimo: GSM8K held-out a decodifica GREEDY (come lm-eval del lavoro)
#     su 256 prompt DISTINTI del train di GSM8K mai usati in allenamento (perm[0:256]; il test di GSM8K non si tocca),
#     premio stretto; McNemar esatto appaiato contro l'ospite. PIU' la sonda parlata (21 seed, quella di tutte le leve
#     di C, thinking come sempre: dichiarato, l'audit suggerisce non-thinking). Accettato = GSM8K >= ospite E sonda >=
#     ospite; scelta = GSM8K piu' alto (poi sonda, poi il passo prima), «checkpoint scelto sulla validazione» come il
#     lavoro. Arresto per DERIVA: GSM8K sotto l'ospite con p < 0,05.
#   5 MMLU-2000 int8 SOLO come CONTROLLO, sul checkpoint scelto (se scende con p < 0,05 si passa al successivo
#     accettato): il lavoro migliora la GENERAZIONE, non la conoscenza (QA9 +0,05..+2,5).
#   6 MAX_RESPONSE 1536 (run_math.sh), prima 1024. Held-out di KL dell'allenatore: 16 prompt x 4 generati UNA volta
#     dall'ospite (perm[256:272]; solo sorveglianza, il lavoro non valida durante l'allenamento: TEST_FREQ 100000).
# ── DICHIARATE, invariate (non sono nel mandato di questo braccio) ── template di chat + istruzione '####' invece del
#   prompt «plain» con stop (lo studente gira sul cammino di deployment); solo GSM8K (lavoro: GSM8K+MATH+DAPO 2500
#   ciascuno); clamp k1 +-10 e logp >= -10 assenti; maestro Q8 al posto del bf16; theta + codici (passo V) invece di pesi
#   master + STE; lr 1,5e-4 su log-scala (trasferimento del 3e-6 del lavoro: giusto per l'audit); held-out e sonda del
#   passo V con la perdita uniforme (prompt compresi).
#
# ── STIMA GPU DI CASA (dai tempi MISURATI del giro 1 di c_onpolicy_1012, corsa.log/registro.log del 13/10) ──
#   completioni: train 316 in 36 min a lotti di 32 (6,2 s l'una: la coda aspetta la piu' lunga del lotto); held-out
#     64 tutte insieme in 152-180 s (2,4-2,8 s) -> qui tutte insieme: 2,6 s (+10 % per 1536 token) · server ~1 min
#   maestro Q8: 38 righe in 5,1 min (0,12 min a riga + ~0,5 di avvio) · allenamento: avvio + held-out 1,7 min, passo 1
#     533 s (calibrazione G1), passi dopo 368-450 s (media ~410 s a 4 righe), istantanea 7 s · export 5,2 min (CPU)
#   cancello: 256 greedy ~11 min + server 1 + sonda 1,7 · fissi: identita' 11 min, KL24 ~10, held-out KL 4, cancello
#     dell'ospite 14, MMLU di controllo 12 a candidato
#   => giro da 2 passi (8 righe, ~88 completioni) ~30 min, cioe' ~15 min a passo:
#     FEDELE 80 passi: 40 giri ~20,5 h + 8 cancelli ~1,8 h + fissi ~0,9 h = ~23,2 h   (70 passi: ~20,4 h) -> OLTRE le 8 h
#     VARIANTE THOR (default, dichiarata): 20 passi = 10 giri ~5,1 h + 2 cancelli 0,5 h + fissi ~0,9 h = ~6,4 h
#       budget: 20/80 = 25 % dei passi del lavoro (29 % dei 70 scelti); rollout usati ~790 (20 x ~39) contro 2560
#       (80 x 32) = ~31 % (35 % dei 2240 di s70); stesso lotto per passo (~39 contro 32), stessa freschezza.
#       Alternativa «piu' completioni per passo» (OPD_RIGHE_PASSO=8 OPD_PASSI=16): ~78 completioni a passo, ~7,9 h,
#       ~1250 rollout = ~49 % del lavoro ma 20 % dei passi (il codice ufficiale lo ammette: «trade steps for a cleaner
#       gradient»). Default 20 x 4 perche' il movimento di Adam e' ~lr a PASSO e l'App. D del lavoro misura +4,3 GSM8K
#       a 20 passi. Le ore vere si leggono nel log a ogni giro (riga ATTIVO) e la PROVA ristampa il conto.
#
# INTERRUTTORI: OPD_VARIANTE=thor|fedele (passi 20|80) · OPD_PASSI (sovrascrive) · OPD_PASSI_GIRO (2) · OPD_RIGHE_PASSO
#   (4) · OPD_CANCELLO_OGNI (10) · OPD_N_CANCELLO (256) · OPD_MAXTOK (1536) · OPD_OSPITE=<gguf> (default: il MIGLIOR C
#   per MMLU-2000 int8 fra quelli col log completo e la taglia di C; OPD_OSPITE_MMLU=<log> col GGUF dato) · OPD_LR
#   (1.5e-4) · OPD_LR_F32 (1.5e-6) · OPD_GIB (60) · PROVA=1 (solo CPU: controlli, prove, argomenti, scelta
#   dell'ospite, conto delle ore; niente GPU, niente cicli.tsv).
# Marcatore: stato/coda_fatte/c-onpolicy-fedele-1013 (scritto qui a fine corsa; presente -> esce subito).
# NON tocca c_onpolicy_1012.sh, la coda viva, l'albero ONPOLICY/codice (solo letto) ne' le cartelle ONPOLICY/giro_*.
set -uo pipefail
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus PYTHONDONTWRITEBYTECODE=1
S=@@ROOT:forge@@/stato; F=@@ROOT:forge@@; TT=@@ROOT:models@@/gguf/tony-tern; OC=$TT/OBIETTIVO_C
NOME=C-S7-T550; OP=$OC/S7_CONGIUNTO_T550_1020; MARC=$S/coda_fatte/c-s7-t550-1020; PROVA=${PROVA:-0}
COD=$OC/ONPOLICY/codice; T2=$COD/thor_v2; PY=@@ROOT:venv@@/bin/python3
FSRC=$F/sorgenti_forgia/obiettivo_c/onpolicy_fedele; FC=$OP/fedele_cod; [ "${PROVA:-0}" = 1 ] && FC=$OP/_prova/fedele_cod
B=@@ROOT:models@@/build/thor-c0t/build-c0t/bin
KLPY=$TT/R026/v12/_lav/kl_realizzata.py; DUMP=$TT/R026/_bin/e2e_dump_logit_p2; TEACH24=@@ROOT:depot@@/e2e_teacher/q8-maestro
TQ8=$TT/teacher-q8/Ornith-1.5-35B-A3B-Q8_0.gguf; SONDA=$OC/TRIO/sonda.sh; BANCO=$F/mc_v2/mmlu_00000.bin
MCN=$F/thor_v2/strumenti/mcnemar_mclp.py; GOV=$F/thor_v2/strumenti/governa_export.sh
G0C=$TT/C_PTQ/Thor-T-C-f_bil1jarq.gguf; TAGLIA=8631199776
RADT=$OP/teacher; PORTA=11475; UNITA=adhoc-onpolicy-t550
F32=attn_norm,post_attention_norm,attn_q_norm,attn_k_norm,ssm_norm,ffn_gate_inp,ffn_gate_inp_shexp,ssm_a,ssm_dt,ssm_conv1d,output_norm
KQAT=cache_densi,silu,tile,amax,mm16_grande,guardia
VARIANTE=${OPD_VARIANTE:-thor}
case $VARIANTE in thor) PDEF=20 ;; fedele) PDEF=80 ;; *) echo "OPD_VARIANTE thor|fedele, non $VARIANTE"; exit 2 ;; esac
PASSI_TOT=${OPD_PASSI:-$PDEF}; PG=${OPD_PASSI_GIRO:-2}; RP=${OPD_RIGHE_PASSO:-4}; CANC=${OPD_CANCELLO_OGNI:-10}
NC=${OPD_N_CANCELLO:-256}; MAXTOK=${OPD_MAXTOK:-1536}; LR=${OPD_LR:-1.5e-4}; LRF=${OPD_LR_F32:-1.5e-6}; GIB=${OPD_GIB:-60}
RIGHE=$((RP * PG)); NG=$(( (PASSI_TOT + PG - 1) / PG )); RHELD=2
[ $PG -ge 2 ] || { echo "OPD_PASSI_GIRO >= 2: con 1 passo a giro il passo V sui codici non scatta mai (vedi intestazione)"; exit 2; }
[ $NC -ge 256 ] || { echo "OPD_N_CANCELLO >= 256 (mandato)"; exit 2; }
[ -e $MARC ] && { echo "$NOME gia' fatto ($(cat $MARC)): esco"; exit 0; }
mkdir -p $OP $RADT $OP/_tmp/triton $OP/_tmp/inductor $OP/cancelli
export TMPDIR=$OP/_tmp TRITON_CACHE_DIR=$OP/_tmp/triton TORCHINDUCTOR_CACHE_DIR=$OP/_tmp/inductor
[ "$PROVA" = 1 ] || { exec >> $OP/corsa.log 2>&1; cp "$0" $OP/comando_$(date +%Y%m%d_%H%M).sh; }
riga(){ local t; t=$(printf '%s' "$2" | tr '\t\n' '  ' | sed 's/rc=/codice /g'); echo "[$(date '+%F %T')] $1 $t"
        [ "$PROVA" = 1 ] || printf '%s\t%s\t%s\n' "$(date +%Y-%m-%dT%H:%M)" "$1" "$t" >> $S/cicli.tsv; }
libero(){ df -BG --output=avail "$1" | tail -1 | tr -dc 0-9; }
SENZA_GPU="HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES="
# AUDIT 13/10 (S7 partito alle 18:42 fuori sequenza): S7 e' l'ULTIMO passo di C, parte solo con S4 finito davvero
#   ($OC/S4_TESSERAQ_1013/FATTO). Fuori sequenza solo con OPD_SENZA_S4=1. Codice 200 (>= 128): la coda viva non lo segna FATTO.
if [ "$PROVA" != 1 ] && [ "${OPD_SENZA_S4:-0}" != 1 ] && [ ! -e $OC/S4_TESSERAQ_1013/FATTO ]; then
  riga ERRORE "$NOME NON parte: S4 non finito ($OC/S4_TESSERAQ_1013/FATTO assente; sequenza base -> statistiche -> trit -> scale -> CARDAN -> on-policy)"; exit 200; fi
SSM=1
VAR="variante $VARIANTE: $PASSI_TOT passi = $NG giri x $PG passi x $RP righe (rollout nuovi a ogni giro), premio stretto ufficiale, vantaggi verl, max $MAXTOK token, cancello GSM8K-$NC greedy ogni $CANC passi + sonda, MMLU solo controllo, lr $LR/$LRF"
argv(){ # $1 corsa  $2 gguf  $3 teacher  $4 istantanee — come c_onpolicy_1012.sh (TT2 fedele + OPD), passi del giro
  echo allena --corsa $1 --gguf $2 --teacher $3 --ricetta B3_ind32 --testa-kl bf16x3 --logit-maestro bf16 \
    --epoche 1 --righe-passo $RP --righe-micro 1 --heldout-ogni $PG --heldout-righe-rapide 0 --heldout-pieno-ogni 1 \
    --non-consumare --K0 524288 --pv-codici --pv-rho 0.03 --pv-passi-accumulo 1 --pv-modo vbop --pv-rho-adattivo \
    --perdita cakld --gamma 1.0 --peso-token generati --opd-kl campionata --opd-beta 1.0 --opd-eakld-k 16 \
    --allena-f32 $F32 --lr $LR --lr-f32 $LRF --ema-theta 0.9 --ripresa-ogni $PG \
    --ottimizzatore-theta adamw --pv-accetta guadagno --pv-z-guadagno-min 1 \
    --istantanea-ogni-passo --istantanee-dir $4 --istantanee-tieni 1 --istantanee-libero-min-gib 1 \
    --pv-sonda-dosi --pv-dosi-frazioni 1/4096,1/1024,1/256,1/64,1/16,1/4,1 --pv-ricentra --pv-ricentra-giri 3 \
    --densi-ternari tutte; }
OPF(){ ( cd $T2 && env OPD_THOR_V2=$T2 ${OPF_NICE:-nice -n 5} $PY $FC/opd_fedele.py "$@" ); }

# ══ 0 controlli su CPU ══
e=0
for f in $T2/e2e_allena.py $T2/e2e_onpolicy.py $T2/e2e_onpolicy_rollout.py $T2/e2e_test_onpolicy.py $T2/e2e_teacher.py \
         $T2/e2e_dump_logit $T2/e2e_esporta.py $T2/e2e_pv_codici.py $COD/cardan_kd.py $B/llama-server $B/llama-perplexity \
         $KLPY $DUMP $TEACH24 $TQ8 $TQ8.sha256 $SONDA $BANCO $MCN $GOV $F/lucchetto_b.sh $G0C \
         $FSRC/opd_fedele.py $FSRC/math_mixed_reward_ufficiale.py $FSRC/PROVENIENZA.txt; do
  [ -e $f ] || { echo "manca $f"; e=1; }; done
[ $e = 0 ] || { riga ERRORE "$NOME: file mancanti ($OP/corsa.log)"; exit 2; }
# sorgenti del braccio CONGELATI nella cartella della corsa (come S23): una modifica dopo il congelamento non entra
if [ ! -s $FC/SHA_CONGELATO.txt ]; then
  rm -rf $FC; mkdir -p $FC; cp -p $FSRC/opd_fedele.py $FSRC/math_mixed_reward_ufficiale.py $FSRC/PROVENIENZA.txt $FC/
  ( cd $FC && sha256sum opd_fedele.py math_mixed_reward_ufficiale.py PROVENIENZA.txt ) > $FC/SHA_CONGELATO.txt
fi
( cd $FC && sha256sum -c --quiet SHA_CONGELATO.txt ) || { riga ERRORE "$NOME: $FC diverso da SHA_CONGELATO.txt"; exit 2; }
for f in opd_fedele.py math_mixed_reward_ufficiale.py; do cmp -s $FSRC/$f $FC/$f || echo "AVVISO: $f cambiato in $FSRC dopo il congelamento: vale la copia congelata"; done
if ! $PY -c "import json,sys; sys.exit(0 if json.load(open('$OC/ONPOLICY/prova_opd.json')).get('ok') else 1)" 2>/dev/null \
   && ! $PY -c "import json,sys; sys.exit(0 if json.load(open('$OP/prova_opd.json')).get('ok') else 1)" 2>/dev/null; then
  ( cd $T2 && env $SENZA_GPU OMP_NUM_THREADS=4 nice -n 19 $PY e2e_test_onpolicy.py $OP/prova_opd.json ) > $OP/prova_opd.log 2>&1 \
    || { riga ERRORE "$NOME: prove CPU dell'OPD rosse ($OP/prova_opd.log)"; exit 2; }
fi
P=$OP/_prova_fedele; rm -rf $P
G1R=$OC/ONPOLICY/giro_1/corpus/rollout_train.jsonl
HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= OMP_NUM_THREADS=4 OPF_NICE="nice -n 19" \
  OPF prova --out $P $([ -s $G1R ] && echo --rollout-giro1 $G1R) > $OP/prova_fedele.log 2>&1 \
  || { riga ERRORE "$NOME: prova CPU di opd_fedele rossa ($OP/prova_fedele.log): $(grep -m2 '^KO' $OP/prova_fedele.log | tr '\n' ' ' | cut -c1-200)"; exit 2; }
rm -rf $P
echo "opd_fedele: $(tail -1 $OP/prova_fedele.log); $(grep -o 'rollout: flessibile.*' $OP/prova_fedele.log | head -1)"
P=$OP/_prova_argomenti; rm -rf $P; mkdir -p $P
( cd $T2 && env E2E_ALLENA_DIR=$P $SENZA_GPU OMP_NUM_THREADS=4 E2E_DENSI_SSM_AB=$SSM nice -n 19 $PY - $(argv PROVA-ARG $G0C $P/teacher $P/ist) --device cpu <<'Q'
import sys, json
import e2e_allena as A
class Fermo(Exception): pass
def ferma(*a, **k): raise Fermo()
A.SorgenteCache = ferma
try:
    A.main(sys.argv[1:])
except Fermo:
    c = json.load(open(A.RADICE / "PROVA-ARG" / "config.json"))["config"]
    print("PROVA ARGOMENTI OK:", {k: c[k] for k in ("peso_token", "perdita", "gamma", "opd_kl", "opd_beta", "lr", "lr_f32",
          "righe_per_passo", "epoche", "heldout_ogni", "ripresa_ogni", "ema_theta")})
    sys.exit(0)
print("PROVA ARGOMENTI: il teacher non e' stato chiesto"); sys.exit(1)
Q
) > $OP/prova_argomenti.log 2>&1; RC=$?; rm -rf $P
[ $RC = 0 ] || { riga ERRORE "$NOME: prova argomenti fallita ($OP/prova_argomenti.log): $(tail -1 $OP/prova_argomenti.log | cut -c1-200)"; exit 2; }
echo "argomenti: $(tail -1 $OP/prova_argomenti.log | cut -c1-300)"

# ── ospite: OPD_OSPITE, altrimenti il MIGLIOR C per MMLU-2000 int8 (log completo, taglia di C, sha se dichiarato) ──
mmok(){ [ "$(grep -c '^MCLP	' $1 2>/dev/null)" = 2000 ]; }
acc(){ grep -oE 'Final result: [0-9.]+' "$1" 2>/dev/null | tail -1 | grep -oE '[0-9.]+$'; }
CANDI_L=()   # "gguf|log|nome"
if [ -n "${OPD_OSPITE:-}" ]; then
  CANDI_L+=("$OPD_OSPITE|${OPD_OSPITE_MMLU:-}|$(basename $OPD_OSPITE .gguf | sed 's/^Thor-T-//')")
else
  # 14/10 (ordine della catena, ricerche/C_ORDINE_CATENA_1014.md): PRIMA l'uscita di OBQ fibrato su S4 se ACCETTATA
  #   (OBQ_FIBRATO_S4_1014/ACCETTATO «gguf log acc ...»; prima in lista = vince a pari MMLU), poi la RISERVA OBQ fibrato su
  #   (c)+DRQ (68,15; esclusa se la sua sonda e' peggiore dell'ospite di piu' di 1) e l'uscita dei densi-CARDAN della
  #   catena (DENSI_CARDAN_CATENA_1013, GGUF dalla riga CHIUSO, log mmlu_DC.log), che prima non era fra i candidati.
  WOS4=$OC/OBQ_FIBRATO_S4_1014; WO13=$OC/OBQ_FIBRATO_1013; WDC=$OC/DENSI_CARDAN_CATENA_1013; NOTE_OBQ=""
  if [ -s $WOS4/ACCETTATO ]; then read -r g l _ < $WOS4/ACCETTATO; CANDI_L+=("$g|$l|$(basename $g .gguf | sed 's/^Thor-T-//')")
  else NOTE_OBQ=" OBQ-su-S4:non-accettato"; fi
  qo=$(grep -oE 'giuste [0-9]+/21' $WO13/rif/sonda_C-p_bil1jarqdrq.out 2>/dev/null | tail -1 | grep -oE '[0-9]+' | head -1)
  qn=$(grep -oE 'giuste [0-9]+/21' $WO13/sonda_OBQFIB.out 2>/dev/null | tail -1 | grep -oE '[0-9]+' | head -1)
  if [ -n "$qo" ] && [ -n "$qn" ] && [ "$qn" -lt $(( qo - 1 )) ]; then NOTE_OBQ="$NOTE_OBQ C-p_bil1jarqdrq-OBQFIB:sonda-$qn/21-sotto-$qo-1"
  else CANDI_L+=("$WO13/Thor-T-C-p_bil1jarqdrq-OBQFIB.gguf|$WO13/mmlu_OBQFIB.log|C-p_bil1jarqdrq-OBQFIB"); fi
  GDC=$(grep -P '\tCHIUSO\tC-DENSI-CARDAN:' $S/cicli.tsv | grep -F "$WDC/" | tail -1 | grep -oE "$WDC/[^ ]+\.gguf" | tail -1)
  [ -n "$GDC" ] && CANDI_L+=("$GDC|$WDC/mmlu_DC.log|$(basename $GDC .gguf | sed 's/^Thor-T-//')")
  O2=$TT/E2E_C2; O3=$TT/E2E_C3
  CANDI_L+=("$G0C|$O2/mmlu_C-bil1jarq.log|C-bil1jarq" "$O2/Thor-T-E2E-C2-p2.gguf|$O2/mmlu_E2E-C2-p2.log|E2E-C2-p2"
            "$O2/Thor-T-E2E-C2-p3.gguf|$O2/mmlu_E2E-C2-p3.log|E2E-C2-p3")
  G3=$(grep -P '\tCHIUSO\tE2E-C3' $S/cicli.tsv | tail -1 | grep -oE 'file /[^ ]+\.gguf' | tail -1 | cut -d' ' -f2)
  [ -n "$G3" ] && { n=$(basename $G3 .gguf | sed 's/^Thor-T-//'); CANDI_L+=("$G3|$O3/mmlu_$n.log|$n"); }
  CE=$OC/CARDAN_ESPERTI_1013/kd/Thor-T-C-bil1jarqdrq-CARDAN.gguf; CANDI_L+=("$CE|$(dirname $CE)/mmlu_CARDAN.log|C-CARDAN-esperti")
  for g in $OC/PTQ2/Thor-T-C-p_*.gguf; do n=$(basename $g .gguf | sed 's/^Thor-T-//'); CANDI_L+=("$g|$OC/PTQ2/mmlu_$n.log|$n"); done
  for k in $(seq 1 9); do CANDI_L+=("$OC/ONPOLICY/Thor-T-C-ONPOLICY-g$k.gguf|$OC/ONPOLICY/giro_$k/mmlu.log|C-ONPOLICY-g$k"); done
  # catena casalinga del 13/10 (S1b, S2+S3, S4): GGUF con il loro MMLU-2000 in misure/
  for d in $OC/S1B_BYTEZERO_CORRETTO_1014 $OC/S23_SEQUENZIALE_CORRETTO_1014 $OC/S4_TESSERAQ_1013; do
    for g in $d/gguf/Thor-T-*.gguf; do [ -s "$g" ] || continue; n=$(basename $g .gguf | sed 's/^Thor-T-//')
      CANDI_L+=("$g|$d/misure/mmlu_$n.log|$n"); done; done
  # S5 (c_s5_cardan_finale_1013.sh): CARDAN fedele sull'ospite finale, log kd/mmlu_CARDAN.log come C4
  for g in $OC/S5_CARDAN_FINALE_1013/kd/Thor-T-*-CARDAN.gguf; do [ -s "$g" ] || continue
    CANDI_L+=("$g|$(dirname $g)/mmlu_CARDAN.log|C-S5-$(basename $g .gguf | sed 's/^Thor-T-//')"); done
  # cartelle per corsa e per nome: un log mmlu_<nome>.log accanto a un GGUF Thor-T-<nome>.gguf (CARDAN, M1, ecc.)
  for g in $OC/CARDAN_ESPERTI_1013/*/Thor-T-*.gguf $OC/M1_SSMDT_1013/*/Thor-T-*.gguf $OC/M1_SSMDT_1013/Thor-T-*.gguf; do
    [ -s "$g" ] || continue; n=$(basename $g .gguf | sed 's/^Thor-T-//'); l=$(ls $(dirname $g)/mmlu*$n*.log 2>/dev/null | head -1)
    [ -n "$l" ] && CANDI_L+=("$g|$l|$n"); done
  # AUDIT 13/10: bracci FAIR-Calib (c_fair_calib_1013.sh: gguf() = misure/Thor-T-C-fair_<v>.gguf, mlog() = misure/mmlu_C-fair_<v>.log)
  for g in $OC/FAIR_CALIB/misure/Thor-T-C-fair_*.gguf; do [ -s "$g" ] || continue; n=$(basename $g .gguf | sed 's/^Thor-T-//')
    CANDI_L+=("$g|$OC/FAIR_CALIB/misure/mmlu_$n.log|$n"); done
fi
G=""; N0=""; AMIG=-1; RIF_MM=""; CANDI="${NOTE_OBQ:-}"; declare -A VISTO
for c in "${CANDI_L[@]}"; do
  IFS='|' read -r g l n <<< "$c"
  [ -n "${VISTO[$g]:-}" ] && continue; VISTO[$g]=1
  [ -s "$g" ] && mmok "$l" || { [ -e "$g" ] && CANDI="$CANDI $n:senza-MMLU-2000"; continue; }
  [ "$(stat -c %s $g)" = $TAGLIA ] || { CANDI="$CANDI $n:taglia-diversa"; continue; }
  [ -s $g.sha256 ] && { [ "$(nice -n 19 sha256sum $g | cut -d' ' -f1)" = "$(cut -d' ' -f1 $g.sha256)" ] || { CANDI="$CANDI $n:sha-diverso"; continue; }; }
  a=$(acc $l); CANDI="$CANDI $n:$a"
  if awk -v a=$a -v m=$AMIG 'BEGIN{exit !(a > m)}'; then AMIG=$a; G=$g; N0=$n; RIF_MM=$l; fi
done
[ -n "$G" ] || { riga ERRORE "$NOME: nessun ospite con MMLU-2000 completo e taglia di C ($CANDI)"; exit 2; }
G_OSP=$G
echo "ospite: $N0 (MMLU-2000 $AMIG, $G) fra:$CANDI"

# ── conto delle ore (tempi misurati del giro 1, intestazione) ──
stima(){ # $1 passi  $2 righe/passo -> ore
  awk -v P=$1 -v RP=$2 -v PG=$PG -v C=$CANC -v NC=$NC -v RHELD=$RHELD 'BEGIN{
    g=int((P+PG-1)/PG); r=RP*PG; comp=r*39.5/4*1.1; tgen=1+comp*2.6*1.1/60+0.5; tma=0.5+0.12*(r+RHELD);
    ttr=1.7+2.2+PG*6.9*RP/4+0.6; tex=5.2; giro=tgen+tma+ttr+tex; nc=int(P/C)+((P%C)?1:0);
    tc=1+NC*2.6/60+1.7; fissi=11+10+4+tc+12;
    printf "%.1f h (giro %.1f min = generazione %.1f + maestro %.1f + allenamento %.1f + export %.1f; %d giri, %d cancelli x %.0f min, fissi %.0f min); rollout usati ~%d = %.0f %% dei 2560 del lavoro, passi %.0f %% di 80",
      (g*giro+nc*tc+fissi)/60, giro, tgen, tma, ttr, tex, g, nc, tc, fissi, P*RP*39.5/4, 100*P*RP*39.5/4/2560, 100*P/80}'; }
STIMA=$(stima $PASSI_TOT $RP)
echo "stima, questa corsa ($VARIANTE): $STIMA"
if [ "$PROVA" = 1 ]; then
  echo "stima, fedele 80 passi: $(stima 80 4)"
  echo "stima, fedele 70 passi: $(stima 70 4)"
  echo "stima, Thor 20 x 4 righe: $(stima 20 4)"
  echo "stima, Thor 16 x 8 righe: $(stima 16 8)"
  echo "piano: $VAR; ospite $N0; porta $PORTA, unita' $UNITA; cartella $OP; marcatore $MARC"
  echo "dischi: @@ROOT:models@@ $(libero @@ROOT:models@@) GiB liberi"
  echo "PROVA=1: fine (nessuna GPU)"; exit 0
fi

# ══ 1 GPU ══
NCK=$(( PASSI_TOT / CANC + 2 ))
SERVE=$(( 9 * NCK + 3 * NCK + 25 ))
[ "$(libero @@ROOT:models@@)" -ge $SERVE ] || { riga ERRORE "$NOME: @@ROOT:models@@ $(libero @@ROOT:models@@) GiB < $SERVE"; exit 2; }
{ echo "albero $COD (solo letto); e2e_allena/e2e_onpolicy/rollout: $(cd $T2 && sha256sum e2e_allena.py e2e_onpolicy.py e2e_onpolicy_rollout.py | cut -c1-16 | tr '\n' ' ')"
  echo "braccio fedele: $FC ($(cut -c1-16 $FC/SHA_CONGELATO.txt | tr '\n' ' ')); motore $(git -C @@ROOT:models@@/build/thor-c0t rev-parse HEAD 2>/dev/null)"
  echo "ospite $G ($N0, MMLU-2000 $AMIG); $VAR; stima $STIMA"; } > $OP/provenienza.txt
. $F/lucchetto_b.sh
until guardia_ram; do sleep 60; done
riga ATTIVO "$NOME: controlli CPU verdi; ospite $N0 (MMLU-2000 $AMIG, $G; fra:$CANDI); $VAR; stima $STIMA; attende il lucchetto GPU (ha $(cat $L_FILE/proprietario 2>/dev/null))"
prendi_gpu "$NOME $$" $GIB
PID=""
spegni_srv(){ systemctl --user stop $UNITA 2>/dev/null; systemctl --user reset-failed $UNITA 2>/dev/null; }
fine(){ local c=$?
  if [ -n "$PID" ] && [ -d /proc/$PID ] && tr '\0' ' ' < /proc/$PID/cmdline | grep -q "e2e_allena.py allena --corsa C-OPDF"; then kill -TERM $PID; wait $PID 2>/dev/null; fi
  spegni_srv; molla_gpu; [ $c = 0 ] || riga ERRORE "$NOME uscita con codice $c (log $OP/corsa.log)"; }
trap fine EXIT
T0=$(date +%s)
giuste(){ local t; t=$(grep -oE 'giuste [0-9]+/21' "$1" 2>/dev/null | tail -1 | grep -oE '[0-9]+' | head -1)   # AUDIT 13/10 (a)
  [ -n "$t" ] || [ "$(grep -cE 'domanda [0-9]+: giuste [0-9]+/7' "$1" 2>/dev/null)" != 3 ] || t=$(grep -oE 'domanda [0-9]+: giuste [0-9]+/7' "$1" | sed -E 's/.*giuste ([0-9]+)\/7/\1/' | paste -sd+ | bc)
  echo "$t"; }
klv(){ $PY -c "import json; d=json.load(open('$1')); print(f\"{d['kl_media']:.5f} {1-d['top1_uguale_media']:.4f}\")" 2>/dev/null; }
# L0 13/10 (ricerche/C_SONDA_DIAGNOSI_1013.md §2): prima le variabili int8 arrivavano alla bash della sonda ma NON al
# server (systemd-run non eredita l'ambiente): sonda f32 contro GSM8K int8 di servi(). Ora $SONDA le passa con -E
# (si esportano qui in modo esplicito, valore uguale a servi()). Un esito gia' scritto si riusa SOLO se viene dal
# giudice L0 (risposta finale, «grezzi in»): un .out del giudice vecchio si mette da parte e si rifa' la sonda.
sonda(){ if ! { grep -q "grezzi in" $2 2>/dev/null && [ -n "$(giuste $2)" ]; }; then
    [ -s $2 ] && mv $2 $2.giudice-vecchio-$(date +%Y%m%d_%H%M%S)
    GGML_VK_DISABLE_GRAPH_OPTIMIZE=1 GGML_VK_TQ1R_Q8=1 bash $SONDA $1 $3 > $2 2>&1; fi
  giuste $2; }
mmlu(){ mmok $2 || LD_LIBRARY_PATH=$B GGML_VK_DISABLE_GRAPH_OPTIMIZE=1 GGML_VK_TQ1R_Q8=1 nice -n 5 $B/llama-perplexity -m $1 -ngl 999 -c 4096 -np 4 \
  -ub 512 -dev Vulkan0 --multiple-choice --multiple-choice-tasks 100000 -f $BANCO > $2 2>&1; mmok $2; }
kl24(){ [ -n "$(klv $2)" ] || { LD_LIBRARY_PATH=$B GGML_VK_DISABLE_GRAPH_OPTIMIZE=1 GGML_VK_TQ1R_Q8=1 nice -n 5 timeout 7200 $PY $KLPY \
  --thor-v2 $F/thor_v2 --cache $3 --dump-bin $DUMP --out $(dirname $2) --gruppo 1 ${4:+--righe $4} $(basename $2 .json | sed 's/^kl_//')=$1 \
  > $2.out 2>&1; rm -rf $(dirname $2)/_lavoro_*; }; klv $2; }
servi(){ spegni_srv; sleep 2      # lo studente sul cammino di deployment: motore thor-c0t, attivazioni int8, KV 16 bit
  systemd-run --user --unit $UNITA --collect env LD_LIBRARY_PATH=$B GGML_VK_TQ1R_Q8=1 GGML_VK_DISABLE_GRAPH_OPTIMIZE=1 \
    $B/llama-server --model "$1" -ngl 999 --ctx-size $((8 * 2048)) --parallel 8 --host 127.0.0.1 --port $PORTA \
    --threads 8 --jinja --reasoning-format none --no-webui > /dev/null 2>&1
  for i in $(seq 1 90); do curl -s -m2 http://127.0.0.1:$PORTA/health 2>/dev/null | grep -q ok && return 0; sleep 5; done; return 1; }
maestro(){ # $1 corsa  $2 cartella corpus  $3 lati
  ( cd $T2 && env -u HSA_OVERRIDE_GFX_VERSION nice -n 5 $PY e2e_teacher.py prepara --corsa $1 --radice $RADT --corpus-dir $2 \
      --variante onpolicy --lati $3 --modo segmenti --righe-blocco 8 --modello $TQ8 --sha-modello $(cut -d' ' -f1 $TQ8.sha256) \
    && env -u HSA_OVERRIDE_GFX_VERSION nice -n 5 $PY e2e_teacher.py esegui --corsa $1 --radice $RADT ) >> $OP/maestro_$1.log 2>&1; }
cancello(){ # $1 gguf  $2 nome -> "acc giuste/NC" (riassunto in $OP/cancelli/$2)
  local d=$OP/cancelli/$2; mkdir -p $d
  if [ ! -s $d/riassunto.json ]; then servi $1 || { spegni_srv; return 1; }
    OPF cancello --server http://127.0.0.1:$PORTA --out $d --n-cancello $NC --max-token $MAXTOK > $d/cancello.log 2>&1 || { spegni_srv; return 1; }
    spegni_srv; fi
  $PY -c "import json; d=json.load(open('$d/riassunto.json')); print(f\"{d['acc_stretto']:.2f} {d['giuste_stretto']}\")"; }

# ══ 2 riferimenti dell'ospite: cancello GSM8K-$NC, sonda, KL24, identita' torch <-> motore, held-out di KL ══
mkdir -p $OP/rif
read -r GA0 GG0 <<< "$(cancello $G ospite)"; [ -n "$GG0" ] || { riga ERRORE "$NOME: cancello GSM8K dell'ospite fallito ($OP/cancelli/ospite)"; exit 2; }
QS=$(sonda $G $OP/rif/sonda_ospite.out $NOME-ospite); QS=${QS:--1}
AC0=$(acc $RIF_MM)
KLP=$(kl24 $G $OP/rif/kl_ospite.json $TEACH24 24 | cut -d' ' -f1)
[ -n "$KLP" ] || { riga ERRORE "$NOME: KL24 dell'ospite non misurata ($OP/rif/kl_ospite.json.out)"; exit 2; }
riga ATTIVO "$NOME: ospite $N0: GSM8K-$NC greedy stretto $GA0 % ($GG0), sonda $QS/21, MMLU-2000 int8 $AC0, KL24 $KLP"
if [ ! -s $OP/identita.ok ]; then
  ( cd $COD && env E2E_SENZA_COESISTENZA=1 E2E_GDN=veloce E2E_GDN_MODULO=gdn_veloce_corsa_v5 E2E_MOTORE=triton2 E2E_KQAT=$KQAT \
      E2E_KERNEL_NUOVI=1 E2E_DENSI_SSM_AB=$SSM E2E_ALLENA_DIR=$OP/allena nice -n 5 timeout 5400 $PY cardan_kd.py --uscita $OP/identita \
      --gguf $G --nome $NOME-identita --teacher $TEACH24 --ricetta B3_ind32 --kl-partenza-motore $KLP --soglia-coerenza 0.03 \
      --strati 0 --rango 1 --passi 0 --heldout-righe 24 > $OP/identita.out 2>&1 ) \
    || { riga ERRORE "$NOME: cancello d'identita' torch<->motore FALLITO ($(grep -o 'coerenza: KL torch.*' $OP/identita/registro.log 2>/dev/null | tail -1 | cut -c1-160))"; exit 2; }
  grep -o 'coerenza: KL torch.*' $OP/identita/registro.log | tail -1 > $OP/identita.ok
fi
HK=$OP/heldout_kl
if [ ! -s $HK/riassunto_heldout.json ]; then
  servi $G || { riga ERRORE "$NOME: llama-server non parte sull'ospite"; exit 3; }
  OPF held-kl --server http://127.0.0.1:$PORTA --out $HK --n-cancello $NC --righe-held $RHELD --max-token $MAXTOK > $OP/heldout_kl.log 2>&1 \
    || { spegni_srv; riga ERRORE "$NOME: held-out di KL fallito ($OP/heldout_kl.log)"; exit 3; }
  spegni_srv
fi

# ══ 3 giri: rollout nuovi -> maestro -> $PG passi -> export; cancello ogni $CANC passi ══
printf 'passo\tgguf\tgsm_stretto\tgiuste\tsonda\tp_vs_ospite\tvinte\tperse\n' > $OP/checkpoint.tsv.nuovo
[ -s $OP/checkpoint.tsv ] && tail -n +2 $OP/checkpoint.tsv >> $OP/checkpoint.tsv.nuovo
mv -f $OP/checkpoint.tsv.nuovo $OP/checkpoint.tsv
MOTIVO=""; G=$G_OSP
for k in $(seq 1 $NG); do
  D=$OP/giro_$k; CORSA=C-OPDF-g$k; IST=$OP/ist_g$k; N=$NOME-g$k; GK=$OP/Thor-T-$N.gguf; PK=$(( k * PG ))
  [ $PK -gt $PASSI_TOT ] && PK=$PASSI_TOT
  mkdir -p $D
  if [ -s $D/FATTO.json ]; then
    G=$($PY -c "import json; print(json.load(open('$D/FATTO.json'))['gguf'])"); echo "giro $k gia' chiuso: $G"
  else
    t0=$(date +%s)
    if [ ! -s $D/corpus/riassunto.json ]; then                  # (a) rollout NUOVI dello studente di adesso
      servi $G || { riga ERRORE "$NOME giro $k: llama-server non parte su $G"; exit 3; }
      OPF genera --server http://127.0.0.1:$PORTA --stato $OP/stato_prompt --giro $k --out $D/corpus --righe $RIGHE \
          --prompt-primo $((2 * RIGHE)) --n-cancello $NC --max-token $MAXTOK > $D/rollout.log 2>&1 \
        || { spegni_srv; riga ERRORE "$NOME giro $k: generazione fallita ($(tail -1 $D/rollout.log | cut -c1-200))"; exit 3; }
      spegni_srv
    fi
    cp -f $HK/e2e_heldout_4096_onpolicy.pt $HK/onpolicy_heldout.pt $D/corpus/
    RS=$($PY -c "import json; d=json.load(open('$D/corpus/riassunto.json'));t=d['train'];print(f\"{t['completioni']} completioni ({d['generate']} generate), premio stretto {t['ricompensa_media']:.3f} (flessibile {100*t['ricompensa_flessibile_media']:.1f} %), penalita' {t['penalita_lunghe']}, lunghezza {t['lunghezza_media']:.0f}, troncate {100*t['troncate']:.1f} %, gruppi con segnale {100*t['gruppi_con_segnale']:.0f} %, {d['secondi']/60:.1f} min\")")
    maestro opdf-g$k $D/corpus heldout,train || { riga ERRORE "$NOME giro $k: maestro fallito ($OP/maestro_opdf-g$k.log)"; exit 3; }
    if [ ! -s $IST/passo_$(printf %06d $PG)/registri_pv.npz ]; then   # (c) allenamento: $PG passi, corsa NUOVA dal GGUF
      R=(); [ -s $OP/allena/$CORSA/ripresa_ultima.json ] && R=(--riprendi)
      for TENT in 1 2 3; do
        ( cd $T2 && exec env E2E_ALLENA_DIR=$OP/allena E2E_SENZA_COESISTENZA=1 E2E_GDN=veloce E2E_GDN_MODULO=gdn_veloce_corsa_v5 \
            E2E_MOTORE=triton2 E2E_KQAT=$KQAT E2E_DENSI_SSM_AB=$SSM nice -n 5 $PY $T2/e2e_allena.py $(argv $CORSA $G $RADT/opdf-g$k $IST) "${R[@]}" ) \
          >> $D/allena.out 2>&1 &
        PID=$!; wait $PID; RC=$?; PID=""
        [ $RC = 75 ] && [ $TENT -lt 3 ] && { R=(--riprendi); riga ATTIVO "$NOME giro $k: sentinella (75), ripresa $((TENT + 1))/3"; sleep 180; continue; }
        break; done
      [ -s $IST/passo_$(printf %06d $PG)/registri_pv.npz ] || { riga ERRORE "$NOME giro $k: allenamento finito (codice $RC) senza l'istantanea del passo $PG ($(grep -E 'ARRESTO|Error|ERRORE' $D/allena.out | tail -1 | cut -c1-160))"; exit 3; }
    fi
    HO=$(grep -E "held-out al passo $PG:" $OP/allena/$CORSA/registro.log | tail -1 | grep -oE 'KL [0-9.]+' | head -1)
    PV=$(grep -cE 'passo V [0-9]+: .*scelta' $OP/allena/$CORSA/registro.log)
    if [ ! -s $GK.sha256 ]; then                                     # (d) export (stessa procedura di E2E-C2)
      n=passo_$(printf %06d $PG); sc=$OP/ssd_$N.scale; cp -f $OP/allena/$CORSA/config.json $IST/config.json
      r=0; $GOV systemd-run --user --scope --quiet -p CPUQuota=800% -p MemoryMax=12G -p MemorySwapMax=2G nice -n 19 ionice -c3 \
        env -u HSA_OVERRIDE_GFX_VERSION $SENZA_GPU OMP_NUM_THREADS=8 E2E_DENSI_SSM_AB=$SSM \
        bash -c "cd $T2 && $PY e2e_esporta.py --sorgente $G --corsa $IST --istantanea $n --uscita $sc --device cpu" > $D/export.log 2>&1 || r=1
      [ $r = 0 ] && { $GOV systemd-run --user --scope --quiet -p CPUQuota=800% -p MemoryMax=12G nice -n 19 ionice -c3 \
        env -u HSA_OVERRIDE_GFX_VERSION $SENZA_GPU OMP_NUM_THREADS=8 E2E_DENSI_SSM_AB=$SSM \
        bash -c "cd $T2 && $PY e2e_pv_codici.py esporta --scalato $sc --dst $GK --registri $IST/$n/registri_pv.npz" >> $D/export.log 2>&1 || r=2; }
      rm -f $sc
      [ $r = 0 ] && [ "$(stat -c %s $GK)" = "$(stat -c %s $G)" ] || { rm -f $GK; riga ERRORE "$NOME giro $k: export fallito (fase $r, $(tail -1 $D/export.log | cut -c1-160))"; exit 3; }
      sha256sum $GK | cut -d' ' -f1 > $GK.sha256
    fi
    # (e) pulizia: stato pesante dell'allenatore (ripresa, bop, migliori) rifacibile dal GGUF; restano registri e log.
    #     Il GGUF e l'istantanea del giro prima si tolgono se non sono un checkpoint (servivano solo come sorgente).
    rm -rf $OP/allena/$CORSA/ripresa_*.pt $OP/allena/$CORSA/bop_pv_*.pt $OP/allena/$CORSA/migliore_*
    kp=$((k - 1))
    if [ $kp -ge 1 ] && [ ! -e $OP/giro_$kp/CHECKPOINT ]; then rm -f $OP/Thor-T-$NOME-g$kp.gguf; rm -rf $OP/ist_g$kp; fi
    $PY -c "import json; json.dump({'gguf': '$GK', 'passi': $PK, 'rollout': '''$RS''', 'heldout_kl': '$HO', 'passi_V_con_scelta': $PV}, open('$D/FATTO.json', 'w'))"
    riga ATTIVO "$NOME giro $k/$NG (passi fino a $PK/$PASSI_TOT): $RS; held-out $HO; passi V con dose scelta $PV; giro in $(( ($(date +%s) - t0) / 60 )) min; $GK sha $(cut -c1-16 $GK.sha256)"
    G=$GK
  fi
  # (f) cancello ogni $CANC passi e all'ultimo giro
  if [ $(( PK % CANC )) = 0 ] || [ $k = $NG ]; then
    if ! grep -qP "^$PK\t" $OP/checkpoint.tsv; then
      read -r GA GG <<< "$(cancello $G p$PK)"; [ -n "$GG" ] || { riga ERRORE "$NOME: cancello del passo $PK fallito"; exit 3; }
      q=$(sonda $G $OP/cancelli/p$PK/sonda.out $NOME-p$PK); q=${q:--1}
      read -r p vv pp <<< "$(OPF mcnemar $OP/cancelli/ospite/riassunto.json $OP/cancelli/p$PK/riassunto.json | $PY -c "import json,sys; d=json.load(sys.stdin); print(d['p_mcnemar_esatto'], d['B_giusta_A_sbagliata'], d['A_giusta_B_sbagliata'])")"
      touch $D/CHECKPOINT
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' $PK $G $GA $GG $q $p $vv $pp >> $OP/checkpoint.tsv
      riga ATTIVO "$NOME CANCELLO passo $PK: GSM8K-$NC greedy stretto $GA % ($GG; ospite $GA0 %, vinte/perse $vv/$pp, McNemar p $p), sonda $q/21 (ospite $QS); $(( ($(date +%s) - T0) / 60 )) min dall'inizio"
      awk -v a=$GA -v a0=$GA0 -v p=$p 'BEGIN{exit !(a < a0 && p < 0.05)}' && { MOTIVO="deriva: GSM8K sotto l'ospite con p < 0,05 al passo $PK"; break; }
    fi
  fi
done
[ -n "$MOTIVO" ] || MOTIVO="budget di $PASSI_TOT passi"

# ══ 4 scelta sulla validazione (GSM8K, poi sonda, poi il passo prima) e CONTROLLO MMLU ══
MIGL=""; NOTE=""
while IFS=$'\t' read -r pk g ga gg q p vv pp; do
  awk -v a=$ga -v a0=$GA0 -v q=$q -v qs=$QS 'BEGIN{exit !(a >= a0 && q >= qs)}' || continue
  l=$OP/cancelli/p$pk/mmlu.log
  mmlu $g $l || { NOTE="$NOME passo $pk: MMLU incompleto; $NOTE"; continue; }
  $PY $MCN $RIF_MM $l --json $OP/cancelli/p$pk/mcnemar_mmlu.json > /dev/null 2>&1
  read -r am pm <<< "$($PY -c "import json; d=json.load(open('$OP/cancelli/p$pk/mcnemar_mmlu.json')); print(f\"{d['acc_B']:.2f} {d['p_mcnemar_esatto']:.4f}\")")"
  if awk -v a=$am -v a0=$AC0 -v p=$pm 'BEGIN{exit !(a >= a0 || p >= 0.05)}'; then MIGL="$pk|$g|$ga|$gg|$q|$p|$am|$pm"; break; fi
  NOTE="passo $pk: MMLU $am contro $AC0 (p $pm) SCESO, scartato; $NOTE"
done < <(tail -n +2 $OP/checkpoint.tsv | sort -t$'\t' -k3,3gr -k5,5nr -k1,1n)
molla_gpu; trap - EXIT; spegni_srv
if [ -n "$MIGL" ]; then
  IFS='|' read -r pk g ga gg q p am pm <<< "$MIGL"
  riga CHIUSO "$NOME ($VAR) da $N0: scelto passo $pk — GSM8K-$NC greedy stretto $ga % (ospite $GA0 %, McNemar p $p), sonda $q/21 (ospite $QS), MMLU-2000 int8 $am (ospite $AC0, p $pm, solo controllo); arresto: $MOTIVO; ${NOTE}$(( ($(date +%s) - T0) / 60 )) min; file $g sha $(cut -c1-16 $g.sha256)"
else
  riga CHIUSO "$NOME ($VAR) da $N0: NESSUN checkpoint accettato (GSM8K >= $GA0 % e sonda >= $QS, MMLU non sceso); arresto: $MOTIVO; ${NOTE}$(( ($(date +%s) - T0) / 60 )) min ($OP/checkpoint.tsv)"
fi
echo "0 $(date +%FT%T)" > $MARC
exit 0

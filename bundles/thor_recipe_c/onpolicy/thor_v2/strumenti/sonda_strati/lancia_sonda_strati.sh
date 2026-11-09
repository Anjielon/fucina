#!/bin/bash
# SONDA LINEARE PER STRATO — lancio completo (28/9/2026, leva L5 di analisi_leve_conoscenza.md).
# ⛔ NON lanciare mentre la GPU e' di un altro lavoro (riforgia, EB3, KLD, teacher): la guardia lo rifiuta da sola.
#
# Cosa fa, in ordine (si ferma al primo errore; ogni passo gia' completo si salta, i lotti riprendono):
#   0. guardie: GPU libera (strumenti/gpu_occupata.sh, casa esente) oppure coesistenza DICHIARATA (SONDA_ACCANTO_A=PID)
#      con VRAM libera >= GGUF + 6 GiB; disco; un solo lancio alla volta (flock); binario e GGUF presenti
#   1. domande (CPU, ~20 min la prima volta, poi saltato): prepara_domande.py -> $SONDA_DIR/domande
#   2. prova di fumo sulla GPU: 8 domande dello studente, 40 strati x 2048 attesi, tok/s misurati -> ETA stampata
#   3. estrazione (GPU, Vulkan): maestro e studenti -> $SONDA_DIR/stati/<nome>   (estrai_stati.py, a lotti)
#   4. sonda (CPU confinata): sonda.py -> $SONDA_DIR/risultati/<data>_<modelli> (json, md, svg, oof.npz)
#   5. una riga CHIUSO in stato/cicli.tsv con il riassunto (SONDA_CICLI=0 per non scriverla)
# I/O: i dump grezzi di un lotto (~5 GB, scritti e cancellati) vanno in $SONDA_DIR (nvme1n1, lo stesso disco dello
# swap); SONDA_GREZZO=/altro/disco li sposta (es. @@ROOT:models@@, nvme0).
#
# Uso:
#   thor_v2/strumenti/sonda_strati/lancia_sonda_strati.sh
#   STUDENTI="EB3p20=@@ROOT:models@@/gguf/tony-tern/Thor-T-EB3-passo_000020.gguf R2039d4v=@@ROOT:depot@@/riforgia_d4/Thor-T-riforgia-R2039d4v.gguf" \
#     thor_v2/strumenti/sonda_strati/lancia_sonda_strati.sh
#   Accanto a un processo GPU gia' vivo (es. EB3 dopo la riforgia), SOLO se c'e' VRAM:
#   SONDA_ACCANTO_A=$(systemctl --user show e2e-allena-EB3 -p MainPID --value) thor_v2/strumenti/sonda_strati/lancia_sonda_strati.sh
# Prova a secco su CPU (nessuna GPU; modello minuscolo qwen35moe, 4 strati; cosi' e' stata collaudata il 28/9):
#   SONDA_VULKAN=0 SONDA_STRATI_ATTESI=4 SONDA_CICLI=0 SONDA_MIN_ACC_USCITA=0 SONDA_DIR=@@ROOT:depot@@/sonda_strati/secco \
#   SONDA_DOMANDE_DA=@@ROOT:depot@@/sonda_strati/domande MAESTRO=minuscolo_a.gguf STUDENTI="b=minuscolo_b.gguf" ./lancia_sonda_strati.sh
# Come unita' (consigliato: sopravvive alla sessione):
#   systemd-run --user --unit=sonda-strati --collect -p Nice=10 bash -c \
#     "@@ROOT:forge@@/thor_v2/strumenti/sonda_strati/lancia_sonda_strati.sh > @@ROOT:depot@@/sonda_strati/lancio.log 2>&1"
set -uo pipefail
QUI="$(cd "$(dirname "$0")" && pwd)"
S="$(cd "$QUI/.." && pwd)"                     # thor_v2/strumenti
export SONDA_DIR=${SONDA_DIR:-@@ROOT:depot@@/sonda_strati}
MAESTRO=${MAESTRO:-@@ROOT:models@@/gguf/tony-tern/teacher-q8/Ornith-1.5-35B-A3B-Q8_0.gguf}
RIFORGIATO=@@ROOT:depot@@/riforgia_d4/Thor-T-riforgia-R2039d4v.gguf
if [ -z "${STUDENTI:-}" ]; then
  STUDENTI="EB3p20=@@ROOT:models@@/gguf/tony-tern/Thor-T-EB3-passo_000020.gguf"
  [ -s "$RIFORGIATO" ] && STUDENTI="$STUDENTI R2039d4v=$RIFORGIATO"
fi
LOTTO=${SONDA_LOTTO:-128}
JOBS=${SONDA_JOBS:-2}
CICLI=@@ROOT:forge@@/stato/cicli.tsv
PY=${SONDA_PY:-/usr/bin/python3}              # sklearn + scipy + pyarrow + tokenizers (venv-catq non ha sklearn)
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
log() { echo "[$(date '+%F %T')] $*"; }
muori() { log "⛔ $*"; exit "${2:-1}"; }
cpu() {   # CPU confinata: niente GPU, 2 core, nice 19
  systemd-run --user --scope -q -p CPUQuota=200% nice -n 19 \
    env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= "$@"
}

mkdir -p "$SONDA_DIR"
exec 9> "$SONDA_DIR/.lancio.lock"
flock -n 9 || muori "un altro lancio della sonda e' in corso ($SONDA_DIR/.lancio.lock)" 2

# ── 0. guardie ──
[ -x "$S/../e2e_dump_logit" ] || muori "manca thor_v2/e2e_dump_logit"
[ -s "$MAESTRO" ] || muori "manca il maestro $MAESTRO"
for coppia in $STUDENTI; do [ -s "${coppia#*=}" ] || muori "manca lo studente ${coppia#*=}"; done
LIBERO_GB=$(df -BG --output=avail "$SONDA_DIR" | tail -1 | tr -dc 0-9)
[ "$LIBERO_GB" -ge 15 ] || muori "solo ${LIBERO_GB} GB liberi in $SONDA_DIR (servono ~5 GB di grezzo per lotto + 0,5 GB per modello)"

vram_libera_gib() {   # dal driver amdgpu: totale - usata (tutti i processi)
  local d t u
  for d in /sys/class/drm/card*/device; do
    [ -r "$d/mem_info_vram_total" ] || continue
    t=$(cat "$d/mem_info_vram_total"); u=$(cat "$d/mem_info_vram_used")
    echo $(( (t - u) / 1073741824 )); return
  done
  echo 0
}
guardia_gpu() {   # $1 = GGUF che si sta per caricare
  local serve=$(( $(stat -c %s "$1") / 1073741824 + 6 ))
  if [ -n "${SONDA_ACCANTO_A:-}" ]; then
    [ -d "/proc/$SONDA_ACCANTO_A" ] || muori "SONDA_ACCANTO_A=$SONDA_ACCANTO_A: processo inesistente" 3
    GPU_ESCLUDI_PID="$SONDA_ACCANTO_A" "$S/gpu_occupata.sh" --attendi 60 \
      || muori "GPU occupata da altri oltre a $SONDA_ACCANTO_A" 3
    local lib; lib=$(vram_libera_gib)
    [ "$lib" -ge "$serve" ] || muori "coesistenza rifiutata: VRAM libera ${lib} GiB, ne servono ${serve} per $(basename "$1")" 3
    log "coesistenza con PID $SONDA_ACCANTO_A: VRAM libera ${lib} GiB >= ${serve}; -ub 512 (mai di piu' accanto a un training)"
  else
    "$S/gpu_occupata.sh" --attendi "${SONDA_ATTENDI_S:-120}" || muori "GPU occupata: non si lancia (vedi elenco sopra)" 3
  fi
}

# ── 1. domande ──
if [ -n "${SONDA_DOMANDE_DA:-}" ] && [ ! -e "$SONDA_DIR/domande" ]; then
  ln -s "$SONDA_DOMANDE_DA" "$SONDA_DIR/domande"        # riuso di domande gia' preparate (stesso manifesto)
fi
if [ ! -s "$SONDA_DIR/domande/domande.json" ]; then
  log "domande assenti: prepara_domande.py (CPU)"
  cpu env OMP_NUM_THREADS=2 "$PY" "$QUI/prepara_domande.py" || muori "prepara_domande fallito"
fi
N=$("$PY" -c "import json;print(json.load(open('$SONDA_DIR/domande/domande.json'))['n_domande'])")
TOK=$("$PY" -c "import json;print(json.load(open('$SONDA_DIR/domande/domande.json'))['token_prompt']['totale'])")
log "domande: $N ($TOK token di prompt in tutto)"

VULKAN=${SONDA_VULKAN:-1}             # 0 = prova a secco su CPU (modello minuscolo): niente GPU, niente guardia GPU
STRATI_ATTESI=${SONDA_STRATI_ATTESI:-40}
estrai() {   # $1 nome, $2 gguf, resto: opzioni in piu'
  local nome=$1 gguf=$2; shift 2
  local dev=(--vulkan)
  if [ "$VULKAN" = 1 ]; then guardia_gpu "$gguf"; else dev=(); fi
  local grezzo=()
  [ -n "${SONDA_GREZZO:-}" ] && grezzo=(--lavoro "$SONDA_GREZZO/$nome")   # dump transitori su un altro disco
  "$PY" "$QUI/estrai_stati.py" --modello "$gguf" --nome "$nome" "${dev[@]}" --strati-attesi "$STRATI_ATTESI" --lotto "$LOTTO" \
    --thread 4 --ub 512 "${grezzo[@]}" "$@"
}

# ── 2. prova di fumo (8 domande dello studente, poi cancellata) ──
PRIMO="${STUDENTI%% *}"
if [ ! -s "$SONDA_DIR/stati/${PRIMO%%=*}/stati.npy" ]; then
  rm -rf "$SONDA_DIR/fumo"
  estrai fumo "${PRIMO#*=}" --max-domande 8 --uscita "$SONDA_DIR/fumo" --senza-sha256 \
    || muori "prova di fumo fallita (vedi sopra): niente estrazione completa"
  "$PY" - "$SONDA_DIR/fumo/meta.json" "$TOK" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1])); t = m["tempi"][0]; tok = int(sys.argv[2])
v = t["token"] / max(t["secondi_sequenze"], 1e-9)
print(f"fumo: forma {m['forma_stati']} · {t['token']} token in {t['secondi_sequenze']:.1f} s ({v:.0f} tok/s) · "
      f"caricamento {t['caricamento_e_resto']:.1f} s · ETA per modello ~{tok / v / 60:.0f} min + caricamenti")
EOF
  rm -rf "$SONDA_DIR/fumo"
fi

# ── 3. estrazione ──
estrai maestro "$MAESTRO" || muori "estrazione del maestro fallita"
ACC_M=$("$PY" -c "import json;print(json.load(open('$SONDA_DIR/stati/maestro/meta.json'))['accuratezza_uscita_4_lettere'])")
log "maestro: accuratezza d'uscita sulle 4 lettere (MMLU val+dev) $ACC_M (banco test: 82,0)"
# subito, prima di spendere GPU sugli studenti: con gli id delle lettere o la riga sbagliati sarebbe ~0,25
"$PY" -c "import sys; sys.exit(0 if $ACC_M >= ${SONDA_MIN_ACC_USCITA:-0.5} else 1)" \
  || muori "accuratezza d'uscita del maestro $ACC_M sotto ${SONDA_MIN_ACC_USCITA:-0.5}: lettere o stati sbagliati"
ARGS=(--modello "maestro=$SONDA_DIR/stati/maestro")
NOMI=maestro
for coppia in $STUDENTI; do
  estrai "${coppia%%=*}" "${coppia#*=}" || muori "estrazione di ${coppia%%=*} fallita"
  ARGS+=(--modello "${coppia%%=*}=$SONDA_DIR/stati/${coppia%%=*}")
  NOMI="$NOMI-${coppia%%=*}"
done

# ── 4. sonda (CPU) ──
OUT="$SONDA_DIR/risultati/$(date +%Y%m%d-%H%M)_$NOMI"
log "sonda su CPU ($JOBS processi) -> $OUT"
cpu "$PY" "$QUI/sonda.py" "${ARGS[@]}" --uscita "$OUT" --jobs "$JOBS" || muori "sonda.py fallita"

# ── 5. riassunto ──
RIASSUNTO=$("$PY" - "$OUT/sonda_strati.json" <<'EOF'
import json, sys
e = json.load(open(sys.argv[1])); t = e["tutte"]; pezzi = []
for n, c in t["confronti"].items():
    f = c["frazione_perdita_al_confine"]
    pezzi.append(f"{n}: primo strato con perdita significativa {c['primo_strato_significativo']}, "
                 f"da qui sempre {c['da_qui_sempre']}"
                 + (f", frazione della perdita gia' al {f['strato']} {f['valore']:.2f} [{f['ic'][0]:.2f}; {f['ic'][1]:.2f}]"
                    if f else ""))
print(f"Sonda lineare per strato ({t['n_domande']} domande MMLU val+dev): " + " · ".join(pezzi))
EOF
)
log "$RIASSUNTO"
[ "${SONDA_CICLI:-1}" = 1 ] && printf "%s\t%s\t%s\n" "$(date +%Y-%m-%dT%H:%M)" CHIUSO "$RIASSUNTO ($OUT)" >> "$CICLI"
log "FATTO: $OUT/sonda_strati.md · $OUT/sonda_strati.svg"

#!/bin/bash
# 27/10 T400POST copy (W = CARDAN_T400POST_1027, BASE = record T400 GGUF, unit congiunto-t400post); else = B+POST chain.
# 26/10 B+POST — catena delle tappe DISPIEGABILI (corsa: stato/congiunto_bpost_1026.sh, unita' congiunto-bpost,
# dir $W = OBIETTIVO_C/CARDAN_BPOST_1026, tappe in $W/allena/tappe, codice SINTESI_1025/BPP/codice).
# B+POST allena SOLO cio' che si dispiega esatto sopra BASE = B+ T350 PIEGATO (Thor-T-C-BPIU-350.gguf, 71,30):
# scale gate/up (scale_gateup_P.pt), scale down (scale_P.pt), i 301 F32 (f32_P.pt), residui densi rango 16 attn/gdn
# (residui_P.pt); esperti a rango 0 (cardan_P.pt vuoto), niente basso rango sui down (cardan_down_P.pt assente).
# Per ogni tappa P (50 100 ... 400; si aspetta ottimizzatori_P.pt, l'ULTIMO file che il trainer scrive, finche'
# l'unita' e' attiva):
#  0 controllo della tappa: cardan_P rango 0 e vuoto, cardan_down_P assente/vuoto, residui solo famiglie attn/gdn.
#  1 OSPITE H1 = BASE + scale gate/up (SINTESI_1025/L3/codice/scale_gateup_esporta.py --device cuda: l'esportatore
#    confronta le fp16 con lo sha256 delle scale dello STUDENTE e si ferma se diverse) + scale down
#    (stato/bpost_scale_down_fp32_1026.py: stessa aritmetica fp32 dello studente sul dispositivo di allenamento;
#    scale_down_esporta.py e cardan_piega_down.py fanno il prodotto in fp64; il pacco down non ha sha: il rapporto conta
#    le fp16 che il percorso fp64 avrebbe scritto diverse) + F32 (SINTESI_1025/F32/codice/f32_esporta.py: solo i tensori
#    cambiati) -> raffina/patch_gguf.py; verifica al byte H1 contro BASE (stato/bpost_verifica_1026.py): cambiano SOLO
#    ffn_{gate,up,down}_exps (codici identici, solo i campi d) e tensori F32 (fuori dallo strato MTP 40), finiti.
#  2 piega SOLO i residui densi su H1 (tappa_a_delta_1018.py -> SINTESI_1025/BPP/codice/cardan_piega_densi_gpu_veloce.py
#    --dtype float32 --lotto 8, la piega densi di dopo_tappa_misura_gpu_1021_v10.sh) -> patch -> GGUF finale G;
#    verifica al byte G contro H1: cambiano SOLO i densi piegati; catturata (energia H) dal registro della piega.
#  3 MMLU-2000 in casa sul banco MESCOLATO (mc_v2/mmlu2000_mescolato.bin; parziale_mmlu.py --indice --rimappa riporta il
#    log all'ordine originale) sotto stato/.misura_mmlu.lock; McNemar contro T400; riga "BPOST-T<P> DISPIEGABILE: ..."
#    in cicli.tsv. La costruzione (1-2) tiene stato/.misura_tappa.lock (una piega di tappa per volta sulla GPU).
#  Pulizia (SOLO intermedi di questa catena, dopo G verificato): H1, cartelle npy, residui_delta.pt. G resta solo per la
#  tappa migliore per MMLU finora ($OUT/bpost_migliore.tsv); log, rapporti json, registri e G.sha256 restano.
# Interruttori: BPOST_TAPPE ("50 ... 400") · BPOST_UNITA (congiunto-bpost) · BPOST_DISCO_MIN_GIB (30) · RAM_MIN_GIB (10)
#   · PROVA=1 (solo controlli e righe di comando) · prove: BPOST_TAPPE_DIR, BPOST_OUT, BPOST_CICLI, BPOST_DEV (cuda),
#   BPOST_NON_ATTENDERE=1, BPOST_FINO=gguf (si ferma con G verificato), BPOST_MMLU_FINTO=<log mescolato gia' fatto>.
# Avvio: systemd-run --user --unit bpost-tappe /usr/bin/bash @@ROOT:forge@@/stato/bpost_tappe_1026.sh
set -uo pipefail
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus PYTHONDONTWRITEBYTECODE=1
F=@@ROOT:forge@@; S=$F/stato; OC=@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C; COD=$OC/SINTESI_1025/BPP/codice
W=$OC/CARDAN_T400POST_1027; T=${BPOST_TAPPE_DIR:-$W/allena/tappe}; OUT=${BPOST_OUT:-$W}
BASE=$OC/CARDAN_CONGIUNTO_S7T550_1020/tappa_400/piega_d/Thor-T-C-S4b-CONGIUNTO.gguf   # RECORD T400 72,40
UNITA=${BPOST_UNITA:-congiunto-t400post}; TAPPE=${BPOST_TAPPE:-"50 100 150 200 250 300 350 400"}
PY=@@ROOT:venv@@/bin/python3; PATCH=@@ROOT:models@@/build/leve/raffina/patch_gguf.py
SGU=$OC/SINTESI_1025/L3/codice/scale_gateup_esporta.py; SDN=$S/bpost_scale_down_fp32_1026.py
F32ESP=$OC/SINTESI_1025/F32/codice/f32_esporta.py; DELTA=$S/tappa_a_delta_1018.py
DENSI=$COD/cardan_piega_densi_gpu_veloce.py; VER=$S/bpost_verifica_1026.py
B=@@ROOT:models@@/build/thor-c0t/build-c0t/bin; MCN=$F/thor_v2/strumenti/mcnemar_mclp.py; PARZ=$S/parziale_mmlu.py
BANCO=$F/mc_v2/mmlu2000_mescolato.bin; INDICE=$F/mc_v2/mmlu2000_mescolato.indice.tsv
RT=$OC/CARDAN_CONGIUNTO_S7T550_1020/tappa_400/mmlu_CONGIUNTO.log
LK_MMLU=$S/.misura_mmlu.lock; LK_TAPPA=$S/.misura_tappa.lock
CICLI=${BPOST_CICLI:-$S/cicli.tsv}; DEV=${BPOST_DEV:-cuda}; FINO=${BPOST_FINO:-tutto}; PROVA=${PROVA:-0}
SENZA_GPU="HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES="
GPU_ENV="HSA_ENABLE_SDMA=0 TORCHINDUCTOR_COMPILE_THREADS=1"
RX_EXPS='blk\.[0-9]+\.ffn_(gate|up|down)_exps\.weight'
MIGL=$OUT/bpost_migliore.tsv; LOG=$OUT/bpost_tappe.log
mkdir -p $OUT
riga(){ echo "[$(date '+%F %T')] $*" | tee -a $LOG; }
cicli(){ printf '%s\t%s\t%s\n' "$(date +%Y-%m-%dT%H:%M)" "$1" "$2" >> $CICLI; }
attivo(){ systemctl --user is-active -q $UNITA; }
ram(){ until [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge ${RAM_MIN_GIB:-10} ]; do sleep 30; done; }
mclp(){ grep -c '^MCLP	' "$1" 2>/dev/null || true; }

# ── controlli d'avvio ──
e=0
for f in $BASE $PATCH $SGU $SDN $F32ESP $DELTA $DENSI $VER $MCN $PARZ $BANCO $INDICE $RT $B/llama-perplexity; do
  [ -e $f ] || { riga "manca $f"; e=1; }; done
[ $e = 0 ] || exit 2
if [ "$PROVA" = 1 ]; then
  P=${TAPPE%% *}; O=$OUT/tappa_$P; H1=$O/ospite_H1.gguf; G=$O/Thor-T-C-T400POST-T$P.gguf; PD=$O/piega_d; L=$O/mmlu_T400POST$P.log
  echo "PROVA=1: controlli verdi; unita' $UNITA $(attivo && echo attiva || echo NON attiva); tappe $TAPPE; @@ROOT:models@@ $(df -BG --output=avail @@ROOT:models@@ | tail -1 | tr -dc 0-9) GiB liberi"
  echo "  attesa: until [ -s $T/ottimizzatori_$P.pt ] || ! systemctl --user is-active -q $UNITA; do sleep 60; done"
  echo "  (1) $PY $SGU $BASE $T/scale_gateup_$P.pt $O/scale_gu --device $DEV"
  echo "      $PY $SDN $BASE $T/scale_$P.pt $O/scale_down --device $DEV"
  echo "      $PY $F32ESP --base $BASE --pacco $T/f32_$P.pt --uscita $O/f32"
  echo "      ln \$O/{scale_gu,scale_down,f32/tensori}/*.npy $O/h1_tensori/ && LEVE_OSPITE=$BASE $PY $PATCH --dir $O/h1_tensori --out $H1 --base $BASE"
  echo "      $PY $VER $BASE $H1 --ammessi '$RX_EXPS' --f32 --codici '$RX_EXPS' --almeno 1 --json $O/verifica_h1.json"
  echo "  (2) $PY $DELTA $T/residui_$P.pt $O/residui_delta.pt $P"
  echo "      $GPU_ENV OMP_NUM_THREADS=6 LEVE_THREADS=6 $PY -X faulthandler $DENSI --delta $O/residui_delta.pt --gguf $H1 --out $PD --device $DEV --dtype float32 --lotto 8"
  echo "      LEVE_OSPITE=$H1 $PY $PATCH --dir $PD/tensori --out $G --base $H1"
  echo "      $PY $VER $H1 $G --ammessi-file $PD/nomi.txt --almeno 1 --json $O/verifica_g.json; sha256sum $G > $G.sha256; pulizia intermedi"
  echo "  (3) flock $LK_MMLU: LD_LIBRARY_PATH=$B GGML_VK_TQ1R_Q8=1 GGML_VK_DISABLE_GRAPH_OPTIMIZE=1 $B/llama-perplexity -m $G -ngl 999 -c 4096 -np 4 -ub 512 -dev Vulkan0 --multiple-choice --multiple-choice-tasks 100000 -f $BANCO > $L.mesc"
  echo "      python3 $PARZ --indice $INDICE --rimappa $L $L.mesc; python3 $MCN $RT $L --json $O/mcnemar_vs_T400.json"
  echo "      riga cicli: BPOST-T$P DISPIEGABILE: ...; G tenuto solo se migliore ($MIGL)"
  exit 0
fi

controllo_tappa(){  # 0: la tappa e' dispiegabile esatta (nessuna correzione non piegabile esatta)
  env $SENZA_GPU $PY - $T $P > $O/controllo_tappa.txt 2>&1 <<'Q'
import sys, os, torch
T, P = sys.argv[1], sys.argv[2]
c = torch.load(f"{T}/cardan_{P}.pt", map_location="cpu", weights_only=False)
assert c.get("rango", 0) == 0 and not c.get("stato"), f"cardan_{P}.pt NON vuoto: rango {c.get('rango')}, {len(c.get('stato', {}))} strati"
fd = f"{T}/cardan_down_{P}.pt"
if os.path.exists(fd):
    d = torch.load(fd, map_location="cpu", weights_only=False)
    assert not d.get("stato"), f"cardan_down_{P}.pt NON vuoto ({len(d['stato'])} strati)"
r = torch.load(f"{T}/residui_{P}.pt", map_location="cpu", weights_only=False)
fam = set(r["cfg"]["famiglie"]); ok = {"attn_q", "attn_k", "attn_v", "attn_output", "attn_qkv", "attn_gate", "ssm_out"}
assert r["cfg"].get("continuo") and fam <= ok, f"residui: famiglie {sorted(fam)} continuo {r['cfg'].get('continuo')}"
s = torch.load(f"{T}/scale_{P}.pt", map_location="cpu"); g = torch.load(f"{T}/scale_gateup_{P}.pt", map_location="cpu", weights_only=False)
f = torch.load(f"{T}/f32_{P}.pt", map_location="cpu", weights_only=False)
print(f"tappa {P}: cardan rango 0 vuoto; down basso rango assente/vuoto; {len(r['latenti'])} residui densi rango "
      f"{r['cfg']['rango']} {sorted(fam)}; scale down {len(s)} strati; scale gate/up {len(g['theta'])} strati "
      f"(allenate su {g.get('dispositivo')}); F32 {len(f['valori'])} tensori")
Q
}

passo1(){  # OSPITE H1 = BASE + scale gate/up + scale down + F32
  H1=$O/ospite_H1.gguf
  [ -s $O/verifica_h1.json ] && [ -s $H1 ] && return 0
  mkdir -p $O/scale_gu $O/scale_down $O/f32 $O/h1_tensori
  riga "T$P (1): scale gate/up + scale down ($DEV, aritmetica dello studente) + F32 -> ospite H1"
  env $GPU_ENV nice -n 5 $PY $SGU $BASE $T/scale_gateup_$P.pt $O/scale_gu --device $DEV > $O/scale_gu.out 2>&1 \
    || { riga "ERRORE T$P scale gate/up ($O/scale_gu.out: $(tail -1 $O/scale_gu.out))"; return 1; }
  if [ "$DEV" = cuda ]; then   # il pacco del trainer porta lo sha256 delle fp16 dello studente: tutte devono coincidere
    local nsha; nsha=$(python3 -c "import json; t=json.load(open('$O/scale_gu/scale_gateup_esporta.json'))['tensori']; print(sum(1 for v in t.values() if v['sha_come_studente'] is True), len(t))")
    [ "${nsha% *}" = "${nsha#* }" ] && [ "${nsha#* }" -gt 0 ] || { riga "ERRORE T$P scale gate/up: sha = studente solo $nsha"; return 1; }
  fi
  env $GPU_ENV nice -n 5 $PY $SDN $BASE $T/scale_$P.pt $O/scale_down --device $DEV > $O/scale_down.out 2>&1 \
    || { riga "ERRORE T$P scale down ($O/scale_down.out: $(tail -1 $O/scale_down.out))"; return 1; }
  env $SENZA_GPU OMP_NUM_THREADS=4 nice -n 10 $PY $F32ESP --base $BASE --pacco $T/f32_$P.pt --uscita $O/f32 > $O/f32.out 2>&1
  local rcf=$?
  if [ $rcf = 3 ]; then riga "T$P: F32 identici a BASE (nessun byte cambiato)"
  elif [ $rcf != 0 ]; then riga "ERRORE T$P f32_esporta ($rcf, $O/f32.out)"; return 1; fi
  rm -f $O/h1_tensori/*.npy
  ln -f $O/scale_gu/*.npy $O/scale_down/*.npy $O/h1_tensori/ || { riga "ERRORE T$P link npy"; return 1; }
  if ls $O/f32/tensori/*.npy > /dev/null 2>&1; then ln -f $O/f32/tensori/*.npy $O/h1_tensori/ || { riga "ERRORE T$P link F32"; return 1; }; fi
  env $SENZA_GPU LEVE_OSPITE=$BASE nice -n 10 ionice -c3 $PY $PATCH --dir $O/h1_tensori --out $H1 --base $BASE > $O/patch_h1.out 2>&1 \
    || { riga "ERRORE T$P patch H1 ($O/patch_h1.out)"; return 1; }
  env $SENZA_GPU nice -n 10 ionice -c3 $PY $VER $BASE $H1 --ammessi "$RX_EXPS" --f32 --codici "$RX_EXPS" --almeno 1 \
    --json $O/verifica_h1.json.tmp > $O/verifica_h1.txt 2>&1 \
    || { riga "ERRORE T$P verifica al byte H1 ($(tail -1 $O/verifica_h1.txt))"; rm -f $O/verifica_h1.json.tmp; return 1; }
  mv $O/verifica_h1.json.tmp $O/verifica_h1.json
  riga "T$P (1): H1 $(tail -1 $O/verifica_h1.txt | cut -c1-160)"
}

passo2(){  # piega dei SOLI residui densi su H1 -> G
  G=$O/Thor-T-C-T400POST-T$P.gguf; H1=$O/ospite_H1.gguf; PD=$O/piega_d; CD=$O/residui_delta.pt
  [ -s $G.sha256 ] && return 0
  [ -s $CD ] || { env $SENZA_GPU nice -n 10 $PY $DELTA $T/residui_$P.pt $CD.tmp $P > $O/delta.out 2>&1 && mv $CD.tmp $CD; } \
    || { riga "ERRORE T$P delta densi ($O/delta.out)"; rm -f $CD.tmp; return 1; }
  local nd; nd=$(env $SENZA_GPU $PY -c "import torch; print(len(torch.load('$CD', map_location='cpu', weights_only=False, mmap=True)['delta_file']))")
  mkdir -p $PD
  riga "T$P (2): piega di $nd residui densi su H1 ($DEV)"
  if [ "$(ls $PD/tensori/*.npy 2>/dev/null | wc -l)" != "$nd" ]; then
    ram
    env $GPU_ENV OMP_NUM_THREADS=6 LEVE_THREADS=6 nice -n 5 $PY -X faulthandler $DENSI --delta $CD --gguf $H1 --out $PD \
      --device $DEV --dtype float32 --lotto 8 >> $PD/piega.out 2>&1 || { riga "ERRORE T$P piega densi ($PD/piega.out)"; return 1; }
  fi
  [ "$(ls $PD/tensori/*.npy 2>/dev/null | wc -l)" = "$nd" ] || { riga "ERRORE T$P piega densi: $(ls $PD/tensori/*.npy 2>/dev/null | wc -l)/$nd tensori"; return 1; }
  ls $PD/tensori/*.npy | xargs -n1 basename | sed 's/\.npy$//' > $PD/nomi.txt
  env $SENZA_GPU LEVE_OSPITE=$H1 nice -n 10 ionice -c3 $PY $PATCH --dir $PD/tensori --out $G.tmp --base $H1 > $PD/patch.out 2>&1 \
    || { riga "ERRORE T$P patch densi ($PD/patch.out)"; rm -f $G.tmp; return 1; }
  env $SENZA_GPU nice -n 10 ionice -c3 $PY $VER $H1 $G.tmp --ammessi-file $PD/nomi.txt --almeno 1 --json $O/verifica_g.json \
    > $O/verifica_g.txt 2>&1 || { riga "ERRORE T$P verifica al byte G ($(tail -1 $O/verifica_g.txt))"; rm -f $G.tmp; return 1; }
  [ -s $G.tmp.provenienza.json ] && mv $G.tmp.provenienza.json $G.provenienza.json
  mv $G.tmp $G && nice -n 19 ionice -c3 sha256sum $G > $G.sha256
  CAT=$($PY - $PD/registro.jsonl <<'Q'
import json, sys, collections
r = {}
for l in open(sys.argv[1]):
    if l.strip():
        x = json.loads(l); r[x["nome"]] = x
e = sum(x["energia_correzione_H"] for x in r.values()); f = sum(x["residuo_dopo_piega_H"] for x in r.values())
c = collections.defaultdict(lambda: [0.0, 0.0])
for x in r.values():
    k = x["nome"].split(".")[2]; c[k][0] += x["energia_correzione_H"]; c[k][1] += x["residuo_dopo_piega_H"]
print(f"{100 * (1 - f / e):.1f} (" + ", ".join(f"{k} {100 * (1 - v[1] / v[0]):.0f}" for k, v in sorted(c.items())) + ")")
Q
)
  echo "$CAT" > $O/catturata_densi.txt
  riga "T$P (2): G $(cut -c1-16 $G.sha256), densi catturata $CAT %; $(tail -1 $O/verifica_g.txt | cut -c1-120)"
  # pulizia: SOLO intermedi di questa tappa, G verificato al byte (H1 -> G cambia solo i densi piegati)
  [ -s $H1.provenienza.json ] && mv $H1.provenienza.json $O/ospite_H1.provenienza.json.log
  rm -f $H1 $CD; rm -rf $O/h1_tensori
  rm -f $O/scale_gu/*.npy $O/scale_down/*.npy $O/f32/tensori/*.npy $PD/tensori/*.npy
  riga "T$P: intermedi tolti (H1, npy, residui_delta.pt); restano G, log e rapporti; @@ROOT:models@@ $(df -BG --output=avail @@ROOT:models@@ | tail -1 | tr -dc 0-9) GiB liberi"
}

passo3(){  # MMLU-2000 banco mescolato + McNemar vs T400 + migliore
  G=$O/Thor-T-C-T400POST-T$P.gguf; L=$O/mmlu_T400POST$P.log
  if [ "$(mclp $L)" != 2000 ]; then
    [ -s $G ] || { riga "ERRORE T$P: G assente e MMLU incompleta"; return 1; }
    if [ -n "${BPOST_MMLU_FINTO:-}" ]; then cp "$BPOST_MMLU_FINTO" $L.mesc   # solo prove della catena
    else
      ram
      ( flock 9; riga "T$P (3): MMLU-2000 (banco mescolato) avvio"
        LD_LIBRARY_PATH=$B GGML_VK_TQ1R_Q8=1 GGML_VK_DISABLE_GRAPH_OPTIMIZE=1 nice -n 5 $B/llama-perplexity -m $G -ngl 999 \
          -c 4096 -np 4 -ub 512 -dev Vulkan0 --multiple-choice --multiple-choice-tasks 100000 -f $BANCO > $L.mesc 2>&1 ) 9>$LK_MMLU
    fi
    python3 $PARZ --indice $INDICE --rimappa $L $L.mesc > $O/parziale.out 2>&1
  fi
  [ "$(mclp $L)" = 2000 ] || { riga "ERRORE T$P MMLU incompleto ($(mclp $L.mesc) righe nel log mescolato; $(tail -1 $O/parziale.out 2>/dev/null))"; return 1; }
  python3 $MCN $RT $L --json $O/mcnemar_vs_T400.json > $O/mcnemar.out 2>&1 || { riga "ERRORE T$P McNemar ($O/mcnemar.out)"; return 1; }
  read -r ACC R <<< "$(python3 -c "import json; d=json.load(open('$O/mcnemar_vs_T400.json')); print(f\"{d['acc_B']:.2f} {d['acc_B']:.2f} vs T400 {d['acc_A']:.2f}, vinte/perse {d['B_giusta_A_sbagliata']}/{d['A_giusta_B_sbagliata']}, p {d['p_mcnemar_esatto']:.4f}\")")"
  GU=$(python3 -c "import json; d=json.load(open('$O/scale_gu/scale_gateup_esporta.json')); t=d['tensori']; print(f\"{sum(v['scale_cambiate'] for v in t.values())} (sha = studente {sum(1 for v in t.values() if v['sha_come_studente'])}/{len(t)})\")" 2>/dev/null)
  DN=$(python3 -c "import json; d=json.load(open('$O/scale_down/scale_down_fp32.json'))['totali']; print(f\"{d['cambiate']} (fp64 avrebbe scritto {d['diverse_fp64']} diverse)\")" 2>/dev/null)
  FN=$(python3 -c "import json; d=json.load(open('$O/f32/esporta.json')); print(d['scritti'])" 2>/dev/null || echo 0)
  CAT=$(cat $O/catturata_densi.txt 2>/dev/null)
  # migliore per MMLU finora: G resta solo per lei
  read -r PB AB GB < <(tail -1 $MIGL 2>/dev/null || true)
  if [ -z "${AB:-}" ] || awk -v a=$ACC -v b=$AB 'BEGIN{exit !(a > b)}'; then
    if [ -n "${GB:-}" ] && [ "$GB" != "$G" ] && [ -e "$GB" ]; then
      case "$GB" in $OUT/tappa_*/Thor-T-C-T400POST-T*.gguf) al_nas "$GB"; riga "T$P: GGUF della migliore precedente T$PB ($AB) spostato sul NAS";; esac; fi
    printf '%s %s %s\n' $P $ACC $G >> $MIGL; NOTA="migliore finora: GGUF tenuto ($G)"
  else
    al_nas $G; NOTA="GGUF sul NAS (migliore resta T$PB $AB)"
  fi
  riga "T$P (3): MMLU-2000 $R; $NOTA"
  cicli ATTIVO "T400POST-T$P DISPIEGABILE: MMLU-2000 $R; densi catturata ${CAT:-?} %; scale gate/up cambiate ${GU:-?}, down ${DN:-?}; F32 cambiati $FN; $NOTA"
}

al_nas(){ # 27/10 (rule 24/10: nothing that cost generation time is deleted; it moves to the NAS): rsync + sha256, then local rm
  local f=$1 n=@@ROOT:nas@@/modelli-fp/obiettivo_c_1022/T400POST_tappe; mkdir -p $n
  if rsync -a --bwlimit=60000 "$f" $n/ && [ "$(sha256sum < "$f")" = "$(sha256sum < $n/$(basename "$f"))" ]; then rm -f "$f"; riga "$(basename "$f") -> NAS (sha256 ok)"
  else riga "$(basename "$f"): copia NAS FALLITA, resta in locale"; fi; }
riga "avvio catena T400POST: tappe $TAPPE da $T, uscite in $OUT/tappa_<P>, unita' $UNITA"
for P in $TAPPE; do
  O=$OUT/tappa_$P; mkdir -p $O
  if [ -s $O/mcnemar_vs_T400.json ] && [ "$(mclp $O/mmlu_T400POST$P.log)" = 2000 ]; then riga "T$P: gia' misurata"; continue; fi
  if [ "${BPOST_NON_ATTENDERE:-0}" != 1 ]; then
    until [ -s $T/ottimizzatori_$P.pt ] || ! attivo; do sleep 60; done
    sleep 20                                                   # l'ultimo os.replace e' appena avvenuto: margine
  fi
  [ -s $T/ottimizzatori_$P.pt ] || { riga "T$P: $UNITA non attiva e tappa assente: fine catena"; break; }
  if [ ! -s $O/Thor-T-C-T400POST-T$P.gguf.sha256 ]; then
    controllo_tappa || { riga "ERRORE T$P: tappa NON dispiegabile esatta ($(tail -1 $O/controllo_tappa.txt))"; cicli ERRORE "T400POST-T$P: tappa non dispiegabile ($O/controllo_tappa.txt)"; continue; }
    riga "$(tail -1 $O/controllo_tappa.txt)"
    detto=0
    until [ "$(df -BG --output=avail @@ROOT:models@@ | tail -1 | tr -dc 0-9)" -ge ${BPOST_DISCO_MIN_GIB:-30} ]; do
      [ $detto = 0 ] && cicli ATTESA "T400POST-T$P dispiegabile: @@ROOT:models@@ sotto ${BPOST_DISCO_MIN_GIB:-30} GiB liberi, aspetto (niente viene cancellato)"
      detto=1; sleep 300; done
    ram
    ( flock 8; passo1 && passo2 ) 8>$LK_TAPPA || { cicli ERRORE "T400POST-T$P dispiegabile: costruzione fallita ($LOG)"; continue; }
  fi
  [ "$FINO" = gguf ] && { riga "T$P: BPOST_FINO=gguf, mi fermo con G verificato"; continue; }
  passo3 || cicli ERRORE "T400POST-T$P dispiegabile: misura fallita ($LOG)"
done
riga "catena T400POST finita"

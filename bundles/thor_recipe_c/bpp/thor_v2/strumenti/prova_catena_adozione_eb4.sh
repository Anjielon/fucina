#!/bin/bash
# 28/9 — prova a secco di catena_adozione_eb4.sh con unita' FINTE: `systemctl` e `systemd-run` finti in testa al PATH
# (stato delle unita' in file, ogni chiamata registrata con l'ora), GPU finta, una «EB3» finta (un ciclo che scrive una
# ripresa nuova solo DOPO che l'adozione ha cominciato ad aspettarla: alla terza domanda `is-active` su EB3), cartelle e
# cicli in una radice di prova. lancia_eb4.sh e i controlli
# preliminari sono VERI (lancia_eb4.sh estratto dal COMMIT_EB4 come nella catena, codice congelato del COMMIT_EB4, teacher e config.json di EB3 in sola lettura, CPU).
# Nessuna unita' vera viene toccata: prima di tutto la prova controlla che `systemctl` sia quello finto (altrimenti esce).
#   S1  MMLU 74,20 < 74,65: nessuno stop, nessun lancio, niente NON_RIPARTIRE, ciclo «NO».
#   S2  MMLU 76,60: EB4 a secco, stop di EB3 SOLO DOPO la nuova ripresa, NON_RIPARTIRE, GPU consultata a EB3 ferma,
#       EB4 lanciata con la ricetta (--lora-init zero, --lr-lora 1e-5, --gguf, --righe-base-da 512) a EB3 FERMA, ciclo ATTIVO.
#   S3  MMLU 76,60 ma GPU occupata: EB3 fermata al confine, EB4 NON lanciata (exit 6), ciclo ⛔ con il comando per EB3.
#   S4  MMLU 76,60 ma EB4 non pronta (GGUF assente): EB3 NON fermata (exit 3), nessun lancio.
#   M   mutazione: copia dello script senza l'attesa del confine (stop subito) -> il controllo di S2 «stop dopo la ripresa»
#       deve diventare ROSSO.
#   (verifica avversaria 28/9, rimetti_eb3)
#   S5  EB4 lanciata ma MORTA prima della sua prima ripresa: EB3 RIMESSA da sola (NON_RIPARTIRE dell'adozione tolto, riga
#       di riavvio eseguita DOPO la morte di EB4, EB3 di nuovo attiva), exit 8, ciclo «EB3 RIMESSA».
#   S6  lancio di EB4 FALLITO dopo lo stop di EB3 (systemd-run rifiuta): EB3 RIMESSA, exit 7.
#   S7  EB4 morta con il freno di una persona ($D4/NON_RIPARTIRE): EB3 NON rimessa, NON_RIPARTIRE di EB3 intatto.
#   S8  NON_RIPARTIRE di una persona su EB3 (non dell'adozione): nessuno stop, nessun lancio, freno intatto, exit 9.
#   M2  mutazione: senza la chiamata a rimetti_eb3 dopo la morte di EB4 -> S5 ROSSA.
#   M3  mutazione: senza i due controlli del freno altrui -> S8 ROSSA (freno sovrascritto, EB3 fermata).
# Uso: COMMIT_EB4=<sha integrata> bash prova_catena_adozione_eb4.sh   (CPU: dentro systemd-run --scope, nice)
set -u
S=$(cd "$(dirname "$0")" && pwd)
ADOZ=$S/catena_adozione_eb4.sh
COMMIT_EB4=${COMMIT_EB4:?COMMIT_EB4=<sha del ramo integrata>}
GGUF_FINTO=${GGUF_FINTO:-@@ROOT:models@@/gguf/tony-tern/Thor-T-diagnosi-ema16.gguf}   # struttura Thor-T (riforgiato non ancora fatto)
B=${PROVE_DIR:-@@ROOT:depot@@/eb4_lavoro/prova_adozione}
chmod -R u+w "$B" 2>/dev/null; rm -rf "$B"; mkdir -p "$B/bin" "$B/unita"
ESITO=0
prova() { if [ "$2" = 1 ]; then echo "[$1] ✅"; else echo "[$1] ❌ $3"; ESITO=1; fi; }

# ── systemctl finto: stato in $B/unita/<nome>, chiamate in $B/chiamate.log (ora in ns) ──
cat > "$B/bin/systemctl" <<EOF
#!/bin/bash
U=$B/unita; L=$B/chiamate.log
[ "\$1" = --user ] && shift
cmd=\$1; shift
case "\$cmd" in
  is-active) [ "\$1" = --quiet ] && shift; echo "\$(date +%s%N) is-active \$1" >> \$L; [ "\$(cat \$U/\$1 2>/dev/null)" = active ] ;;
  stop) echo "\$(date +%s%N) stop \$1" >> \$L; [ -e \$U/\$1.pid ] && kill \$(cat \$U/\$1.pid) 2>/dev/null; echo inactive > \$U/\$1 ;;
  reset-failed) echo "\$(date +%s%N) reset-failed \$1" >> \$L ;;
  list-units) pat=\${@: -1}; for f in \$U/*; do n=\$(basename \$f); case \$n in *.pid) continue ;; esac
      [ "\$(cat \$f)" = active ] && [[ \$n == \$pat ]] && echo "\$n.service loaded active running finta"; done; true ;;
  show) u=\$1; echo "\$(date +%s%N) show \$u \$*" >> \$L
      if [[ " \$* " == *" ActiveState "* ]]; then
        s=\$(cat \$U/\$u 2>/dev/null); s=\${s:-inactive}
        if [ "\$s" = active ] && [ "\$u" = e2e-allena-EB4finta ] && [ -e $B/eb4_muore ]; then
          s=failed; echo failed > \$U/\$u; echo "\$(date +%s%N) morte_eb4" >> \$L
          [ -e $B/eb4_muore_con_freno ] && mkdir -p "\$(cat $B/d4)" && echo "fermata voluta" > "\$(cat $B/d4)/NON_RIPARTIRE"
        fi
        echo "\$s"
      else echo 0; fi ;;
  *) echo "systemctl finto: comando \$cmd" >&2; exit 1 ;;
esac
EOF
# ── systemd-run finto: registra argomenti e stato di EB3 al momento del lancio, accende l'unita' ──
cat > "$B/bin/systemd-run" <<EOF
#!/bin/bash
u=; for a in "\$@"; do case \$a in --unit=*) u=\${a#--unit=} ;; esac; done
echo "\$(date +%s%N) systemd-run \$u eb3=\$(cat $B/unita/e2e-allena-EB3finta 2>/dev/null)" >> $B/chiamate.log
if [ "\$u" = e2e-allena-EB4finta ] && [ -e $B/eb4_non_parte ]; then echo "systemd-run finto: rifiuto \$u" >&2; exit 1; fi
printf '%s\n' "\$@" > $B/lancio_\$u.argv
echo active > $B/unita/\$u
EOF
# ── GPU finta: occupata se esiste $B/gpu_occupata; registra lo stato di EB3 quando e' consultata ──
cat > "$B/bin/gpu_finta.sh" <<EOF
#!/bin/bash
echo "\$(date +%s%N) gpu eb3=\$(cat $B/unita/e2e-allena-EB3finta 2>/dev/null)" >> $B/chiamate.log
if [ -e $B/gpu_occupata ]; then [ "\${1:-}" = --attendi ] && sleep "\${2:-0}"; echo "4242 finta 30000 MiB"; exit 1; fi
exit 0
EOF
chmod +x "$B/bin/"*
export PATH=$B/bin:$PATH
[ "$(command -v systemctl)" = "$B/bin/systemctl" ] || { echo "⛔ systemctl finto non in testa al PATH: esco"; exit 99; }
[ "$(command -v systemd-run)" = "$B/bin/systemd-run" ] || { echo "⛔ systemd-run finto non in testa al PATH: esco"; exit 99; }

eb3_finta() {   # <radice scenario>: EB3 finta attiva, ripresa del passo 20; la ripresa del passo 21 arriva alla terza
  local r=$1      # domanda is-active su EB3 (l'adozione sta gia' aspettando il confine: deterministico, niente gara di tempi)
  mkdir -p "$r/EB3"; rm -f "$B/chiamate.log"      # registro azzerato PRIMA: le domande dello scenario prima non contano
  printf '{"file": "ripresa_000020.pt", "passo": 20, "scritto": "prima"}\n' > "$r/EB3/ripresa_ultima.json"
  echo active > "$B/unita/e2e-allena-EB3finta"
  ( until n=$(grep -c "is-active e2e-allena-EB3finta" "$B/chiamate.log" 2>/dev/null); [ "${n:-0}" -ge 3 ]; do
      [ "$(cat "$B/unita/e2e-allena-EB3finta")" = active ] || exit 0; sleep 0.2; done
    printf '{"file": "ripresa_000021.pt", "passo": 21, "scritto": "dopo"}\n' > "$r/EB3/ripresa_ultima.json.tmp"
    mv "$r/EB3/ripresa_ultima.json.tmp" "$r/EB3/ripresa_ultima.json"
    echo "$(date +%s%N) ripresa_nuova" >> "$B/chiamate.log"
    while [ "$(cat "$B/unita/e2e-allena-EB3finta")" = active ]; do sleep 1; done ) &
  echo $! > "$B/unita/e2e-allena-EB3finta.pid"
}
adozione() {    # <scenario> <MMLU> [script]: ambiente di prova completo
  local r=$B/$1 mmlu=$2 scr=${3:-$ADOZ}
  rm -f "$B/unita/e2e-allena-EB4finta" "$B/chiamate.log"
  echo "systemd-run --user --unit=e2e-allena-EB3finta true" > "$r/riga.txt"   # riga di riavvio di EB3 (finta)
  echo "$r/allena/EB4" > "$B/d4"
  env RIPRESA_EB4_MAX_S=2 ATTESA_EB3_S=1 MMLU="$mmlu" GGUF="${GGUF_S:-$GGUF_FINTO}" COMMIT_EB4="$COMMIT_EB4" UNITA_EB3=e2e-allena-EB3finta \
      UNITA_EB4=e2e-allena-EB4finta EB3_DIR="$r/EB3" GPU_OCCUPATA="$B/bin/gpu_finta.sh" \
      CICLI="$r/cicli.tsv" RIGA_EB3="$r/riga.txt" POLL_S=1 DOPO_RIPRESA_S=1 GPU_MAX_S=2 CONFINE_MAX_S=120 \
      E2E_ALLENA_DIR="$r/allena" CODICE_DIR="$B/codice" ISTANTANEE="$r/ist" ATTESA_AVVIO_S=1 \
      bash "$scr" > "$r.log" 2>&1
}
ora() { grep -E " $1" "$B/chiamate.log" 2>/dev/null | head -1 | cut -d' ' -f1; }

# S1
mkdir -p "$B/s1"; eb3_finta "$B/s1"
adozione s1 74.20; rc=$?
ok=0; [ "$rc" = 0 ] && ! grep -q "stop\|systemd-run" "$B/chiamate.log" 2>/dev/null && [ ! -e "$B/s1/EB3/NON_RIPARTIRE" ] \
  && grep -q "Adozione EB4: NO" "$B/s1/cicli.tsv" && [ "$(cat "$B/unita/e2e-allena-EB3finta")" = active ] && ok=1
prova S1_sotto_soglia_nulla "$ok" "rc $rc ($B/s1.log)"
echo inactive > "$B/unita/e2e-allena-EB3finta"; kill "$(cat "$B/unita/e2e-allena-EB3finta.pid")" 2>/dev/null

verifica_s2() {  # <scenario> -> 1 se tutto torna
  local r=$B/$1 t_rip t_stop t_gpu t_run argv
  t_rip=$(ora ripresa_nuova); t_stop=$(ora "stop e2e-allena-EB3finta"); t_gpu=$(grep " gpu " "$B/chiamate.log" | tail -1 | cut -d' ' -f1)
  t_run=$(ora "systemd-run e2e-allena-EB4finta"); argv=$(tr '\n' ' ' < "$B/lancio_e2e-allena-EB4finta.argv" 2>/dev/null)
  local ok=1
  [ -n "$t_rip" ] && [ -n "$t_stop" ] && [ "$t_stop" -gt "$t_rip" ] || { ok=0; echo "   stop ($t_stop) NON dopo la nuova ripresa ($t_rip)"; }
  [ -e "$r/EB3/NON_RIPARTIRE" ] || { ok=0; echo "   NON_RIPARTIRE assente"; }
  grep " gpu " "$B/chiamate.log" | tail -1 | grep -q "eb3=inactive" && [ "${t_gpu:-0}" -gt "${t_stop:-0}" ] || { ok=0; echo "   GPU non consultata a EB3 ferma"; }
  grep -q "systemd-run e2e-allena-EB4finta eb3=inactive" "$B/chiamate.log" && [ "${t_run:-0}" -gt "${t_gpu:-0}" ] || { ok=0; echo "   EB4 non lanciata a EB3 ferma"; }
  for x in "--lora-init zero" "--lr-lora 1e-5" "--righe-base-da 512" "--gguf $GGUF_FINTO" "--strati-allenati 20-39" "--corsa EB4"; do
    grep -qF -- "$x" <<< "$argv" || { ok=0; echo "   argomento mancante: $x"; }; done
  grep -qF -- "--riprendi" <<< "$argv" && { ok=0; echo "   --riprendi in una corsa nuova"; }
  grep -q "EB4 LANCIATA" "$r/cicli.tsv" || { ok=0; echo "   ciclo ATTIVO assente"; }
  echo $ok
}
# S2
mkdir -p "$B/s2"; eb3_finta "$B/s2"
adozione s2 76.60; rc=$?
R=$(verifica_s2 s2); ok=0; [ "$rc" = 0 ] && [ "$(tail -1 <<< "$R")" = 1 ] && ok=1
prova S2_adotta_al_confine "$ok" "rc $rc $(head -n -1 <<< "$R" | tr '\n' ';') ($B/s2.log)"

# S3
mkdir -p "$B/s3"; eb3_finta "$B/s3"; touch "$B/gpu_occupata"
adozione s3 76.60; rc=$?; rm -f "$B/gpu_occupata"
ok=0; [ "$rc" = 6 ] && ! grep -q "systemd-run e2e-allena-EB4finta" "$B/chiamate.log" && [ -e "$B/s3/EB3/NON_RIPARTIRE" ] \
  && grep -q "GPU occupata" "$B/s3/cicli.tsv" && grep -q "rm $B/s3/EB3/NON_RIPARTIRE" "$B/s3/cicli.tsv" && ok=1
prova S3_gpu_occupata_niente_EB4 "$ok" "rc $rc ($B/s3.log)"

# S4
mkdir -p "$B/s4"; eb3_finta "$B/s4"
GGUF_S=$B/non_esiste.gguf adozione s4 76.60; rc=$?
ok=0; [ "$rc" = 3 ] && ! grep -q "stop\|systemd-run" "$B/chiamate.log" 2>/dev/null && [ ! -e "$B/s4/EB3/NON_RIPARTIRE" ] \
  && [ "$(cat "$B/unita/e2e-allena-EB3finta")" = active ] && grep -q "EB4 NON pronta" "$B/s4/cicli.tsv" && ok=1
prova S4_EB4_non_pronta_EB3_intatta "$ok" "rc $rc ($B/s4.log)"
echo inactive > "$B/unita/e2e-allena-EB3finta"; kill "$(cat "$B/unita/e2e-allena-EB3finta.pid")" 2>/dev/null

verifica_rimessa() {   # <scenario> <evento dopo cui EB3 deve ripartire> -> 1 se EB3 e' stata rimessa bene
  local r=$B/$1 t_ev t_r3 ok=1
  t_ev=$(ora "$2"); t_r3=$(grep " systemd-run e2e-allena-EB3finta" "$B/chiamate.log" | tail -1 | cut -d' ' -f1)
  [ -n "$t_ev" ] && [ -n "$t_r3" ] && [ "$t_r3" -gt "$t_ev" ] || { ok=0; echo "   EB3 non rilanciata DOPO $2 ($t_r3 / $t_ev)"; }
  [ "$(cat "$B/unita/e2e-allena-EB3finta")" = active ] || { ok=0; echo "   EB3 non attiva"; }
  [ ! -e "$r/EB3/NON_RIPARTIRE" ] || { ok=0; echo "   NON_RIPARTIRE di EB3 ancora li'"; }
  grep -q "EB3 RIMESSA" "$r/cicli.tsv" || { ok=0; echo "   ciclo «EB3 RIMESSA» assente"; }
  echo $ok
}
# S5: EB4 muore prima della prima ripresa
mkdir -p "$B/s5"; eb3_finta "$B/s5"; touch "$B/eb4_muore"
adozione s5 76.60; rc=$?; rm -f "$B/eb4_muore"
R=$(verifica_rimessa s5 morte_eb4); ok=0; [ "$rc" = 8 ] && [ "$(tail -1 <<< "$R")" = 1 ] && ok=1
prova S5_EB4_morta_EB3_rimessa "$ok" "rc $rc $(head -n -1 <<< "$R" | tr '\n' ';') ($B/s5.log)"
echo inactive > "$B/unita/e2e-allena-EB3finta"; kill "$(cat "$B/unita/e2e-allena-EB3finta.pid")" 2>/dev/null

# S6: lancio di EB4 rifiutato dopo lo stop di EB3
mkdir -p "$B/s6"; eb3_finta "$B/s6"; touch "$B/eb4_non_parte"
adozione s6 76.60; rc=$?; rm -f "$B/eb4_non_parte"
R=$(verifica_rimessa s6 "stop e2e-allena-EB3finta"); ok=0
[ "$rc" = 7 ] && [ "$(tail -1 <<< "$R")" = 1 ] && [ ! -e "$B/unita/e2e-allena-EB4finta" ] && ok=1
prova S6_lancio_fallito_EB3_rimessa "$ok" "rc $rc $(head -n -1 <<< "$R" | tr '\n' ';') ($B/s6.log)"
echo inactive > "$B/unita/e2e-allena-EB3finta"; kill "$(cat "$B/unita/e2e-allena-EB3finta.pid")" 2>/dev/null

# S7: EB4 fermata da una persona col freno: niente EB3
mkdir -p "$B/s7"; eb3_finta "$B/s7"; touch "$B/eb4_muore" "$B/eb4_muore_con_freno"
adozione s7 76.60; rc=$?; rm -f "$B/eb4_muore" "$B/eb4_muore_con_freno"
ok=0; [ "$rc" = 8 ] && [ -e "$B/s7/EB3/NON_RIPARTIRE" ] && [ "$(cat "$B/unita/e2e-allena-EB3finta")" != active ] \
  && ! grep -q "systemd-run e2e-allena-EB3finta" "$B/chiamate.log" && grep -q "freno di una persona" "$B/s7/cicli.tsv" && ok=1
prova S7_freno_di_una_persona_EB3_ferma "$ok" "rc $rc ($B/s7.log)"
kill "$(cat "$B/unita/e2e-allena-EB3finta.pid")" 2>/dev/null

# S8: freno di una persona su EB3 -> niente
mkdir -p "$B/s8"; eb3_finta "$B/s8"; echo "prova a mano sulla GPU (una persona)" > "$B/s8/EB3/NON_RIPARTIRE"
adozione s8 76.60; rc=$?
ok=0; [ "$rc" = 9 ] && ! grep -q "stop\|systemd-run" "$B/chiamate.log" 2>/dev/null \
  && [ "$(cat "$B/s8/EB3/NON_RIPARTIRE")" = "prova a mano sulla GPU (una persona)" ] \
  && [ "$(cat "$B/unita/e2e-allena-EB3finta")" = active ] && grep -q "SOSPESA" "$B/s8/cicli.tsv" && ok=1
prova S8_freno_di_una_persona_su_EB3_niente "$ok" "rc $rc ($B/s8.log)"
echo inactive > "$B/unita/e2e-allena-EB3finta"; kill "$(cat "$B/unita/e2e-allena-EB3finta.pid")" 2>/dev/null

# M2: senza rimetti_eb3 dopo la morte di EB4 -> S5 rossa
sed 's|    rimetti_eb3 "\$U4 \$s dopo|    : "\$U4 \$s dopo|' "$ADOZ" > "$B/adozione_mutata2.sh"
if cmp -s "$ADOZ" "$B/adozione_mutata2.sh"; then prova M2_mutazione_applicata 0 "sed non ha cambiato lo script"; else
  mkdir -p "$B/m2"; eb3_finta "$B/m2"; touch "$B/eb4_muore"
  adozione m2 76.60 "$B/adozione_mutata2.sh"; rc=$?; rm -f "$B/eb4_muore"
  R=$(verifica_rimessa m2 morte_eb4); ok=0; [ "$(tail -1 <<< "$R")" = 0 ] && ok=1
  prova M2_senza_rimetti_ROSSA "$ok" "rc $rc, verifica: $(head -n -1 <<< "$R" | tr '\n' ';')"
  echo inactive > "$B/unita/e2e-allena-EB3finta"; kill "$(cat "$B/unita/e2e-allena-EB3finta.pid")" 2>/dev/null
fi

# M3: senza i controlli del freno altrui -> S8 rossa (l'adozione sovrascrive il freno e ferma EB3)
sed -e '/^ferma_se_freno_altrui *$/d' -e '/^ferma_se_freno_altrui  *#/d' "$ADOZ" > "$B/adozione_mutata3.sh"
if [ "$(grep -c '^ferma_se_freno_altrui' "$B/adozione_mutata3.sh")" != 1 ] || ! bash -n "$B/adozione_mutata3.sh"; then
  prova M3_mutazione_applicata 0 "sed non ha tolto le due chiamate"; else
  mkdir -p "$B/m3"; eb3_finta "$B/m3"; echo "prova a mano sulla GPU (una persona)" > "$B/m3/EB3/NON_RIPARTIRE"
  adozione m3 76.60 "$B/adozione_mutata3.sh"; rc=$?
  ok=0; { [ "$rc" != 9 ] || [ "$(cat "$B/m3/EB3/NON_RIPARTIRE")" != "prova a mano sulla GPU (una persona)" ]; } && ok=1
  prova M3_senza_freno_altrui_ROSSA "$ok" "rc $rc"
  echo inactive > "$B/unita/e2e-allena-EB3finta"; kill "$(cat "$B/unita/e2e-allena-EB3finta.pid")" 2>/dev/null
fi

# M: senza l'attesa del confine lo stop arriva PRIMA della nuova ripresa -> verifica di S2 rossa
sed 's|if \[ "\$(cat "\$M3" 2>/dev/null)" != "\$prima" \]; then CONFINE=ripresa; break; fi|CONFINE=ripresa; break|' "$ADOZ" > "$B/adozione_mutata.sh"
if cmp -s "$ADOZ" "$B/adozione_mutata.sh"; then prova M_mutazione_applicata 0 "sed non ha cambiato lo script"; else
  mkdir -p "$B/m"; eb3_finta "$B/m"
  adozione m 76.60 "$B/adozione_mutata.sh"; rc=$?
  R=$(verifica_s2 m); ok=0; [ "$(tail -1 <<< "$R")" = 0 ] && grep -q "NON dopo la nuova ripresa" <<< "$R" && ok=1
  prova M_senza_confine_ROSSA "$ok" "rc $rc, verifica: $(head -n -1 <<< "$R" | tr '\n' ';')"
  echo inactive > "$B/unita/e2e-allena-EB3finta"; kill "$(cat "$B/unita/e2e-allena-EB3finta.pid")" 2>/dev/null
fi
wait 2>/dev/null
echo "ESITO: $([ $ESITO = 0 ] && echo 'TUTTO VERDE (mutazione rossa)' || echo ROSSO)"
exit $ESITO

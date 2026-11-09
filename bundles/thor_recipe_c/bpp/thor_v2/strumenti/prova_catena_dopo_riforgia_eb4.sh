#!/bin/bash
# 28/9 (verifica avversaria di EB4) — prova a secco di catena_dopo_riforgia_eb4.sh, pezzi 4b (MMLU -> adozione) e 4c
# (sonda parlata), estratti dallo script VERO ed eseguiti con Dell, lucchetto, systemd-run, scp e ssh FINTI in testa al PATH
# e con ogni percorso che scriverebbe fuori (cicli, registri, ~/tony-forgia) riscritto nella radice di prova. Nessuna
# unita', nessun ssh vero, nessuna GPU.
#   P0  la catena VECCHIA (catena_dopo_riforgia.sh, 4c): la sonda NON parte — rc 0, zero scp (il difetto, riprodotto)
#   P1  4c della catena nuova: esegui_sonda estratta bene, GGUF copiato sul Dell, sonda lanciata, ciclo «Sonda parlata»
#   P2  4b con MMLU riuscito (76,60): systemd-run catena-adozione-eb4 con MMLU, GGUF e COMMIT_EB4 giusti
#   P3  4b con MMLU FALLITO (lucchetto rc 1): nessun systemd-run, ciclo «fallita»
#   M1  mutazione: nella catena nuova l'estrazione vecchia -> il controllo la FERMA (exit 4, ciclo «estratta male», zero scp)
# Uso: bash prova_catena_dopo_riforgia_eb4.sh
set -u
S0=$(cd "$(dirname "$0")" && pwd)
NUOVA=$S0/catena_dopo_riforgia_eb4.sh; VECCHIA=$S0/catena_dopo_riforgia.sh
B=${PROVE_DIR:-@@ROOT:depot@@/eb4_lavoro/prova_catena_nuova}
rm -rf "$B"; mkdir -p "$B/bin" "$B/s" "$B/out"
ESITO=0
prova() { if [ "$2" = 1 ]; then echo "[$1] ✅"; else echo "[$1] ❌ $3"; ESITO=1; fi; }
bash -n "$NUOVA" || { echo "⛔ sintassi di $NUOVA"; exit 1; }

# strumenti finti nella cartella S di prova: sonda_EB3.sh VERA ma col file dei cicli riscritto, lucchetto che esegue e basta
sed "s|@@ROOT:forge@@/stato/cicli.tsv|$B/cicli_sonda.tsv|g" "$S0/sonda_EB3.sh" > "$B/s/sonda_EB3.sh"
cat > "$B/s/aragorn_lucchetto.sh" <<EOF
#!/bin/bash
echo "lucchetto \$*" >> $B/chiamate.log
if [ "\$(basename "\$1")" = aragorn_valuta_e_pulisci.sh ]; then
  [ -e $B/mmlu_fallisce ] && exit 1
  echo "Final result: 76.6000 +/- 0.9700" > "\$4"; exit 0
fi
exec "\$@"
EOF
cat > "$B/bin/scp" <<EOF
#!/bin/bash
echo "scp \$*" >> $B/chiamate.log
for a in "\$@"; do case \$a in aragorn:*sonda_*.txt) echo "giuste 20/21 (prova)" > "$B/out/\$(basename "\${a#aragorn:}")" ;; esac; done
exit 0
EOF
cat > "$B/bin/ssh" <<EOF
#!/bin/bash
echo "ssh \$*" >> $B/chiamate.log
case "\$*" in *"%~zI"*) stat -c %s $B/finto.gguf ;; *type*fine*) echo FATTO ;; esac
exit 0
EOF
cat > "$B/bin/systemd-run" <<EOF
#!/bin/bash
printf '%s\n' "\$@" > $B/systemd-run.argv; echo "systemd-run \$*" >> $B/chiamate.log
EOF
chmod +x "$B/bin/"* "$B/s/"*.sh
head -c 4096 /dev/zero > "$B/finto.gguf"
export PATH=$B/bin:$PATH
[ "$(command -v scp)" = "$B/bin/scp" ] && [ "$(command -v systemd-run)" = "$B/bin/systemd-run" ] || { echo "⛔ finti non in testa"; exit 99; }

pezzo() {   # <script> <da (regex)> <a (regex)>: il pezzo dello script, percorsi di scrittura riscritti nella prova
  sed -n "/$2/,/$3/p" "$1" | sed -e "s|^OUT=@@ROOT:forge@@|OUT=$B/out|" -e "s|L=@@ROOT:lavoro@@/e2e/allena/|L=$B/|"
}
esegui() {  # <file del pezzo>: ambiente minimo della catena
  rm -f "$B/chiamate.log" "$B/cicli.tsv" "$B/cicli_sonda.tsv" "$B/systemd-run.argv"
  env -i PATH="$PATH" HOME="$HOME" bash -c "
    set -u; S=$B/s; CICLI=$B/cicli.tsv; NOME=PROVA; G=$B/finto.gguf; USC=$B; SOGLIA=74.65; NOMEM=riforgia-PROVA
    log() { echo \"[\$(date '+%F %T')] \$*\"; }
    ciclo() { printf '%s\t%s\t%s\n' \"\$(date +%Y-%m-%dT%H:%M)\" \"\$1\" \"\$2\" >> \"\$CICLI\"; }
    source $1" > "$1.log" 2>&1
}

# P0: il difetto riprodotto sulla catena vecchia
pezzo "$VECCHIA" '^# ── 4c\.' '^log "catena finita"' > "$B/p0.sh"; esegui "$B/p0.sh"; rc=$?
ok=0; [ "$rc" = 0 ] && ! grep -q "^scp .*finto.gguf" "$B/chiamate.log" 2>/dev/null && ok=1
prova P0_catena_vecchia_sonda_muta_riprodotta "$ok" "rc $rc, chiamate: $(tr '\n' ';' < "$B/chiamate.log" 2>/dev/null)"

# P1: la catena nuova lancia la sonda
pezzo "$NUOVA" '^# ── 4c\.' '^log "catena finita"' > "$B/p1.sh"; esegui "$B/p1.sh"; rc=$?
ok=0; [ "$rc" = 0 ] && grep -q "^scp -q $B/finto.gguf aragorn:C:/tcq/thor_mmlu/gguf/finto.gguf" "$B/chiamate.log" \
  && grep -q "sonda_parlata.bat riforgia-PROVA" "$B/chiamate.log" && grep -q "Sonda parlata riforgia-PROVA" "$B/cicli_sonda.tsv" && ok=1
prova P1_catena_nuova_sonda_lanciata "$ok" "rc $rc ($B/p1.sh.log)"

# P2 / P3: 4b -> adozione solo con l'MMLU riuscito
pezzo "$NUOVA" '^# ── 4b\.' '^# ── 4c\.' > "$B/p2.sh"; esegui "$B/p2.sh"; rc=$?
A=$(tr '\n' ' ' < "$B/systemd-run.argv" 2>/dev/null)
ok=0; [ "$rc" = 0 ] && for x in "--unit=catena-adozione-eb4" "--setenv=MMLU=76.6000" "--setenv=GGUF=$B/finto.gguf" \
    "--setenv=COMMIT_EB4=2fc59dc" "bash $B/s/catena_adozione_eb4.sh >>"; do grep -qF -- "$x" <<< "$A" || { ok=2; echo "   manca $x"; }; done
[ "$ok" = 0 ] && [ "$rc" = 0 ] && grep -q "ADOTTA" "$B/cicli.tsv" && ok=1
prova P2_mmlu_riuscito_adozione_avviata "$ok" "rc $rc, argv: $A"
touch "$B/mmlu_fallisce"; esegui "$B/p2.sh"; rc=$?; rm -f "$B/mmlu_fallisce"
ok=0; [ ! -e "$B/systemd-run.argv" ] && grep -q "MMLU della riforgia PROVA sul Dell fallita" "$B/cicli.tsv" && ok=1
prova P3_mmlu_fallito_niente_adozione "$ok" "rc $rc"

# M1: estrazione vecchia nella catena nuova -> il controllo ferma la catena
sed "s|sed -n '/^log() /p;|sed -n '/^log() /,/^}/p;|" "$B/p1.sh" > "$B/m1.sh"
if cmp -s "$B/p1.sh" "$B/m1.sh"; then prova M1_mutazione_applicata 0 "sed non ha cambiato il pezzo"; else
  esegui "$B/m1.sh"; rc=$?
  ok=0; [ "$rc" = 4 ] && grep -q "estratta male" "$B/cicli.tsv" && ! grep -q "^scp .*finto.gguf" "$B/chiamate.log" 2>/dev/null && ok=1
  prova M1_estrazione_vecchia_FERMATA "$ok" "rc $rc"
fi
echo "ESITO: $([ $ESITO = 0 ] && echo 'TUTTO VERDE (difetto riprodotto, mutazione fermata)' || echo ROSSO)"
exit $ESITO

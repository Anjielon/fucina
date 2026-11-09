"""Patch per catena_v2.sh (24/9 22:xx): la forgia della catena diventa sceglibile.

Perche': catena_v2.sh lancia `forgia_globale_40_strati.py` scritto per esteso. La
forgia scorrevole a passo 1 (`forgia_scorrevole.py`) usa la STESSA esportazione
(esporta_gguf_finestra legge THOR_VALIDA_OUT, che la catena gia' imposta a una
cartella nuova per ogni NOME) e le stesse fasi dopo (fedelta' B9, sonda, PPL,
MMLU): basta poter scegliere lo script della fase 1, invece di duplicare la catena.

- THOR_FORGIA_SCRIPT (default forgia_globale_40_strati.py: comportamento INVARIATO);
- ammessi SOLO i due script noti (lista chiusa, rifiuto rumoroso altrimenti);
- lo script usato finisce nel catena.log.

Uso:  python3 strumenti/patch_catena_forgia_script.py [PERCORSO_catena_v2.sh]
(senza argomento: il catena_v2.sh vero, accanto a strumenti/). Non scrive NULLA se
un aggancio non compare esattamente una volta; se la patch e' gia' applicata lo dice
ed esce senza toccare il file.
"""
import sys
from pathlib import Path

P = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "catena_v2.sh"
s = P.read_text()
if "THOR_FORGIA_SCRIPT" in s:
    print(f"patch gia' applicata a {P} (THOR_FORGIA_SCRIPT presente): nessuna modifica")
    sys.exit(0)
sostituzioni = [
    ('''echo "── 1/4 forgia"
''',
     '''echo "── 1/4 forgia"
FORGIA_SCRIPT=${THOR_FORGIA_SCRIPT:-forgia_globale_40_strati.py}
case "$FORGIA_SCRIPT" in
  forgia_globale_40_strati.py|forgia_scorrevole.py) ;;
  *) echo "⛔ THOR_FORGIA_SCRIPT=$FORGIA_SCRIPT non ammesso (forgia_globale_40_strati.py | forgia_scorrevole.py)"; exit 2 ;;
esac
echo "   script di forgia: $FORGIA_SCRIPT"
'''),
    ('''  @@ROOT:venv@@/bin/python3 forgia_globale_40_strati.py > "$D/forgia.log" 2>&1''',
     '''  @@ROOT:venv@@/bin/python3 "$FORGIA_SCRIPT" > "$D/forgia.log" 2>&1'''),
]
for vecchio, nuovo in sostituzioni:
    n = s.count(vecchio)
    if n != 1:
        raise SystemExit(f"aggancio trovato {n} volte (atteso 1), NIENTE scritto: {vecchio[:90]!r}")
    s = s.replace(vecchio, nuovo)
# ⛔ bash legge uno script IN CORSO a pezzi, per posizione nel file: riscriverlo sul
# posto sotto una catena viva la corromperebbe. Si scrive un file nuovo e lo si
# sostituisce (os.replace: nuovo inode) — una catena gia' partita continua a leggere
# il vecchio. Permessi (eseguibile) copiati dall'originale.
import os
tmp = P.with_name(P.name + ".patch_tmp")
tmp.write_text(s)
os.chmod(tmp, P.stat().st_mode)
os.replace(tmp, P)
print(f"patch applicata: {len(sostituzioni)} sostituzioni in {P}")

"""Pannello held-out in PROFONDITA' (24/9 23:3x): lo stesso pannello di pannello_heldout.py
(domini, statistiche appaiate, ricostruzione dai codici, routing vero), ma sulla catena degli
strati 0..N-1 invece della sola finestra [0,1]. Serve a confrontare ricette che cambiano il
modo in cui gli strati si concatenano (finestra scorrevole contro finestre separate): la
misura giusta e' l'errore all'uscita di N strati, non di due.
Uso: THOR_PANNELLO_STRATI=6 python pannello_profondita.py [opzioni di pannello_heldout.py]"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pannello_heldout as P  # noqa: E402

N = int(os.getenv("THOR_PANNELLO_STRATI", "6"))
P.FINESTRA = tuple(range(N))
# 25/9 00:5x: i riferimenti dello storico sono loss dei LOG della finestra [0,1]: in profondita' il pannello
# misura l'uscita dello strato N-1 e il confronto dava un falso ⛔ (1164%). Qui non c'e' un log con cui
# confrontare: niente riferimenti (la tabella resta vuota), invece di un controllo che non controlla niente.
if P.FINESTRA != (0, 1):
    P.RIFERIMENTI_STORICO = {}
if __name__ == "__main__":
    sys.exit(P.main())

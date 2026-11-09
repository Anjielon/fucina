"""Patch RACCORDO per forgia_scorrevole.py (verificatore, 24/9 notte).

Perche': l'orchestratore del compito B chiama `V.ottimizza_finestra_JOINT` con
`ternari_iniziali`/`ternari_uscita` (oggetti TernarioDiffBatchFuso) e cerca
`valida_finestra_8_12.ternari_da_codici`; il compito A ha realizzato l'aggancio
con `ripartenza_per_strato`/`consegna_stato_per_strato` (StatoStrato o codici).
Con la sola patch A la forgia scorrevole RIFIUTA di partire (verificato:
"INTERFACCIA A ASSENTE"). Il raccordo (`raccordo_scorrevole.py`, file nuovo)
traduce; questa patch dice a forgia_scorrevole.py di usarlo:

  1. MODULO_DA_CODICI -> raccordo_scorrevole (+ MODULO_OTTIMIZZA / FUNZ_OTTIMIZZA);
  2. `_funzione_ottimizza()`: prende ottimizza_finestra_JOINT_B dal raccordo e
     RIFIUTA se la patch del compito A non e' applicata a valida_finestra_8_12;
  3. default di `esegui` e controllo di `--verifica-interfaccia` passano da li';
  4. `leggi_codici_strato`: stesso controllo di provenienza di
     `ripartenza_codici.carica_codici_corsa` (nessun .npy piu' recente della sua
     marca: riscritto da un'altra corsa) — mancava;
  5. i nomi dei cicli di `esegui` (fam/obj/p) si azzerano a fine finestra: prima,
     nell'ultima finestra, tenevano vivi i 3 GiB dello strato portato fino a dopo
     l'export (trovato contando gli oggetti vivi in test_raccordo_scorrevole.py).

Uso:
    python3 strumenti/patch_raccordo_scorrevole.py                  # applica a ../forgia_scorrevole.py
    python3 strumenti/patch_raccordo_scorrevole.py --file COPIA.py  # applica a una copia
    python3 strumenti/patch_raccordo_scorrevole.py --prova          # controlla soltanto, non scrive
Non scrive NULLA se un aggancio non compare esattamente una volta, se la patch e'
gia' applicata, o se il risultato non passa ast.parse. Scrittura atomica (os.replace).
"""
import argparse
import ast
import os
from pathlib import Path

PREDEFINITO = Path(__file__).resolve().parent.parent / "forgia_scorrevole.py"
MARCATORE = "MODULO_OTTIMIZZA"

SOSTITUZIONI = [
    # -- 1. nomi dell'interfaccia: il raccordo
    ('''MODULO_DA_CODICI = "valida_finestra_8_12"   # A2: modulo che espone la funzione
''',
     '''MODULO_DA_CODICI = "raccordo_scorrevole"    # A2: raccordo A<->B (24/9 notte), ternari_da_codici
MODULO_OTTIMIZZA = "raccordo_scorrevole"    # A1: raccordo A<->B, sopra la patch A di valida_finestra_8_12
FUNZ_OTTIMIZZA = "ottimizza_finestra_JOINT_B"
'''),
    # -- 2. funzione che prende A1 dal raccordo
    ('''def _codici_gate_up_fusi(codici: dict) -> "tuple[np.ndarray, np.ndarray]":
''',
     '''def _funzione_ottimizza():
    """A1 attraverso il raccordo; rifiuta se la patch del compito A non c'e'."""
    import importlib
    modulo = importlib.import_module(MODULO_OTTIMIZZA)
    modulo.verifica_patch_A()
    return getattr(modulo, FUNZ_OTTIMIZZA)


def _codici_gate_up_fusi(codici: dict) -> "tuple[np.ndarray, np.ndarray]":
'''),
    # -- 3. default di esegui
    ('''    ottimizza_fn = ottimizza_fn or V.ottimizza_finestra_JOINT
''',
     '''    ottimizza_fn = ottimizza_fn or _funzione_ottimizza()
'''),
    # -- 4. --verifica-interfaccia
    ('''        verifica_interfaccia_A(V.ottimizza_finestra_JOINT, serve_da_codici=MODO_INIT == "codici")
''',
     '''        verifica_interfaccia_A(_funzione_ottimizza(), serve_da_codici=MODO_INIT == "codici")
'''),
    # -- 5. provenienza dei .npy (come carica_codici_corsa del compito A)
    ('''    if down and m["down"] != "1":
        raise RuntimeError(f"strato {L} in {cartella}: down_ternario={m['down']}, serve 1")
    cod = {''',
     '''    if down and m["down"] != "1":
        raise RuntimeError(f"strato {L} in {cartella}: down_ternario={m['down']}, serve 1")
    t_marca = (cartella / f"fatto_forgia_{L:02d}.ok").stat().st_mtime
    for pref in ("Dg", "Qg", "Du", "Qu") + (("Dd", "Qd") if down else ()):
        f = cartella / f"{pref}_{L:02d}.npy"
        if f.exists() and f.stat().st_mtime > t_marca + 1.0:
            raise RuntimeError(f"PROVENIENZA: {f} e' piu' recente della marca fatto_forgia_{L:02d}.ok: "
                               "riscritto da un'altra corsa")
    cod = {'''),
    # -- 6. i nomi dei cicli (fam/obj/p) tenevano vivo l'ultimo strato portato: nell'ultima
    #       finestra, 3 GiB di parametri fino alla fine dell'export (misurato col collaudo)
    ('''            for L in commessi:
                osserva("commit", finestra=k0, strato=L, codici=risultato[L])
''',
     '''            fam = obj = p = None     # (raccordo 24/9) i nomi di ciclo non tengono vivi gli oggetti portati
            for L in commessi:
                osserva("commit", finestra=k0, strato=L, codici=risultato[L])
'''),
]


def applica(testo: str) -> str:
    if MARCATORE in testo:
        raise SystemExit(f"patch gia' applicata ('{MARCATORE}' presente): niente da fare")
    for vecchio, nuovo in SOSTITUZIONI:
        n = testo.count(vecchio)
        if n != 1:
            raise SystemExit(f"aggancio trovato {n} volte (atteso 1), NIENTE scritto: {vecchio[:90]!r}")
        testo = testo.replace(vecchio, nuovo)
    ast.parse(testo)
    return testo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--file", type=Path, default=PREDEFINITO)
    ap.add_argument("--prova", action="store_true", help="controlla agganci e sintassi, non scrive")
    args = ap.parse_args()
    nuovo = applica(args.file.read_text())
    if args.prova:
        print(f"prova OK: {len(SOSTITUZIONI)} agganci unici, risultato sintatticamente valido — {args.file} NON toccato")
        return
    tmp = args.file.with_name(args.file.name + ".patch_tmp")
    tmp.write_text(nuovo)
    os.chmod(tmp, args.file.stat().st_mode)
    os.replace(tmp, args.file)
    print(f"patch applicata: {len(SOSTITUZIONI)} sostituzioni in {args.file}")


if __name__ == "__main__":
    main()

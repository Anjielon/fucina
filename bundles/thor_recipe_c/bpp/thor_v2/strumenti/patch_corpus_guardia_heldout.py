"""Patch GUARDIA HELD-OUT per valida_finestra_8_12.py, da applicare DOPO patch_corpus_calibrazione.py (25/9 notte,
verifica indipendente del corpus).

Perche' (misurato): il corpus pronto esclude dal dominio cot SOLO gli esempi 0-8 dell'ordine mescolato col seme
1234, cioe' l'held-out storico con THOR_VALIDA_BATCH_SEQ=8 e THOR_VALIDA_N_HELDOUT=1 (8192 token = 9 esempi).
Col default THOR_VALIDA_N_HELDOUT=4 l'held-out prende 32768 token = esempi 0-34, e gli esempi 9-34 SONO nel
dominio cot del corpus: 22.536 delle 31.997 finestre di 32 token dell'held-out compaiono nella calibrazione
(~70%). La loss held-out (sovra-adattamento) diventerebbe loss di addestramento, in silenzio. Stesso guasto con
un THOR_CALIB_SEED diverso da quello del corpus o con un altro tokenizzatore.

Con THOR_CALIB_CORPUS e una quota cot > 0 la guardia pretende: manifesto presente, stesso seme, stesso
tokenizzatore, stessi file cot (sha256 del manifesto delle sorgenti) e held-out tutto dentro gli esempi esclusi.
Altrimenti solleva (mai un default muto). Senza corpus: nessun cambiamento.

Uso:
    python3 strumenti/patch_corpus_guardia_heldout.py [--file COPIA.py] [--prova]
"""
import argparse
import ast
import os
from pathlib import Path

PREDEFINITO = Path(__file__).resolve().parent.parent / "valida_finestra_8_12.py"
PREREQUISITO = "THOR_CALIB_CORPUS"
MARCATORE = "_guardia_heldout_corpus"

SOSTITUZIONI = [
    ('''    if CALIB_CORPUS and not miscela:
''',
     '''    if CALIB_CORPUS and miscela:
        _guardia_heldout_corpus(miscela, seme, usati_ho)
    if CALIB_CORPUS and not miscela:
'''),
    ('''def carica_batch_calibrazione() -> "tuple[torch.Tensor, int]":
''',
     '''def _guardia_heldout_corpus(miscela: str, seme: int, usati_ho: int) -> None:
    """25/9 (verifica): il corpus pronto esclude dal dominio cot solo gli esempi del SUO held-out storico
    (0..fine del primo intervallo `esclusi`, ordine mescolato col SUO seme). L'held-out di questa corsa deve
    starci dentro, altrimenti le sue sequenze sono anche di addestramento. Solleva, mai in silenzio."""
    import hashlib
    quote = {p.split(":")[0].strip(): int(p.split(":")[1]) for p in miscela.split(",") if p.strip()}
    if quote.get("cot", 0) <= 0:
        return
    man = Path(CALIB_CORPUS).with_suffix(".json")
    if not man.exists():
        raise RuntimeError(f"THOR_CALIB_CORPUS={CALIB_CORPUS}: manifesto {man.name} assente, impossibile verificare "
                           "che l'held-out storico sia fuori dal dominio cot del corpus")
    m = json.loads(man.read_text(encoding="utf-8"))
    cot = m.get("domini", {}).get("cot", {})
    fine = int(cot.get("esclusi", [[0, -1]])[0][1]) + 1          # held-out escluso dal corpus: esempi 0..fine-1
    problemi = []
    if int(m.get("seme", -1)) != seme:
        problemi.append(f"seme {seme} contro {m.get('seme')} del corpus")
    if str(m.get("tokenizzatore")) != str(SRC):
        problemi.append(f"tokenizzatore {SRC} contro {m.get('tokenizzatore')} del corpus")
    if usati_ho > fine:
        problemi.append(f"held-out = esempi CoT 0..{usati_ho - 1}, il corpus esclude solo 0..{fine - 1} "
                        f"(con BATCH_SEQ={BATCH_SEQ} e L_SEQ={L_SEQ}: THOR_VALIDA_N_HELDOUT=1, come i lanciatori, "
                        "o un corpus ricostruito con l'esclusione giusta)")
    locali = (m.get("sorgenti") or {}).get("locali") or {}
    for tipo in ("math", "code"):
        atteso = (locali.get(f"cot_{tipo}") or {}).get("sha256")
        f = CORPUS.parent / f"cot_{tipo}.jsonl"
        if atteso and hashlib.sha256(f.read_bytes()).hexdigest() != atteso:
            problemi.append(f"{f} diverso da quello del corpus")
    if problemi:
        raise RuntimeError(f"THOR_CALIB_CORPUS={CALIB_CORPUS}: held-out storico NON disgiunto dal dominio cot: "
                           + "; ".join(problemi))


def carica_batch_calibrazione() -> "tuple[torch.Tensor, int]":
'''),
]


def applica(testo: str) -> str:
    if PREREQUISITO not in testo:
        raise SystemExit("prima patch_corpus_calibrazione.py (THOR_CALIB_CORPUS assente)")
    if MARCATORE in testo:
        raise SystemExit(f"patch gia' applicata ({MARCATORE} presente): niente da fare")
    for vecchio, nuovo in SOSTITUZIONI:
        n = testo.count(vecchio)
        if n != 1:
            raise SystemExit(f"aggancio trovato {n} volte (atteso 1): {vecchio[:100]!r}")
        testo = testo.replace(vecchio, nuovo)
    ast.parse(testo)
    return testo


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--file", type=Path, default=PREDEFINITO)
    ap.add_argument("--prova", action="store_true")
    a = ap.parse_args()
    nuovo = applica(a.file.read_text(encoding="utf-8"))
    if a.prova:
        print(f"prova: {len(SOSTITUZIONI)} agganci trovati una volta sola in {a.file}, ast.parse ok (nulla scritto)")
        return 0
    tmp = a.file.with_suffix(a.file.suffix + ".patch_tmp")
    tmp.write_text(nuovo, encoding="utf-8")
    os.replace(tmp, a.file)
    print(f"patch applicata: {len(SOSTITUZIONI)} sostituzioni in {a.file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

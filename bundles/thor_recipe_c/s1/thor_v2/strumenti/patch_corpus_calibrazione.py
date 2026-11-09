"""Patch CORPUS DI CALIBRAZIONE PRONTO per valida_finestra_8_12.py (25/9 notte).

Perche': la calibrazione mista (`THOR_CALIB_MISCELA`, 24/9 22:4x) sa solo tre domini
(cot, edu, wiki) tokenizzati al volo, e ogni batch deve avere le stesse quote intere
(somma = BATCH_SEQ). Il corpus multi-dominio (`costruisci_corpus_calibrazione.py`:
italiano web e Wikipedia, chat senza ragionamento, ragionamento, strumenti) arriva
GIA' tokenizzato in un file `.pt` `{dominio: (n, 1024) int64}`, e le quote della
ricerca sono in sedicesimi (ciclo di 2 batch da 8). Questa patch:

  1. `THOR_CALIB_CORPUS=<file.pt>`: `_batch_miscela` prende le sequenze dei domini da
     li' (ognuna vista UNA volta, nell'ordine del file) invece di tokenizzare; i domini
     ammessi sono le chiavi del file; se accanto c'e' il manifesto (`.json`) lo sha256
     di OGNI tensore deve combaciare (rifiuto rumoroso altrimenti);
  2. quote su un CICLO di batch: la somma delle quote puo' essere un multiplo di
     BATCH_SEQ (es. 16 con 8 sequenze per batch = ciclo di 2 batch); le sequenze di
     un ciclo si distribuiscono sui suoi batch per posizione frazionaria (i+0,5)/q,
     cosi' ogni dominio e' spalmato. N_BATCH deve essere multiplo del ciclo. Con somma
     = BATCH_SEQ (ciclo 1) la costruzione e' QUELLA DI PRIMA, riga per riga;
  3. `THOR_CALIB_CORPUS` senza `THOR_CALIB_MISCELA` e' un errore (mai un default muto);
  4. l'held-out storico NON cambia: si costruisce sempre dagli esempi CoT 0-8 come prima.

Senza `THOR_CALIB_CORPUS` il comportamento e' identico a prima (ciclo 1, stessi domini).
Con il corpus e le quote di M2 (`cot:2,edu:3,wiki:3`) le batch sono IDENTICHE AL BIT a
quelle senza corpus finche' il dominio cot non supera l'esempio 278 dell'ordine mescolato
(provato su copia da `strumenti/prova_patch_corpus_calibrazione.py`); oltre, il vecchio
percorso entrerebbe negli esempi del pannello (279-385), il corpus li salta.

Uso:
    python3 strumenti/patch_corpus_calibrazione.py                  # applica a ../valida_finestra_8_12.py
    python3 strumenti/patch_corpus_calibrazione.py --file COPIA.py  # applica a una copia
    python3 strumenti/patch_corpus_calibrazione.py --prova          # controlla soltanto, non scrive
Non scrive NULLA se un aggancio non compare esattamente una volta, se la patch e' gia'
applicata, o se il risultato non passa ast.parse. Scrittura atomica (os.replace).
"""
import argparse
import ast
import os
from pathlib import Path

PREDEFINITO = Path(__file__).resolve().parent.parent / "valida_finestra_8_12.py"
MARCATORE = "THOR_CALIB_CORPUS"

SOSTITUZIONI = [
    # -- 1. costante + caricatore del corpus pronto
    ('''CALIB_WIKI_FILE = "wikitext-2-raw-v1.train.raw"   # SOLO train: validation = pannello, test = PPL
''',
     '''CALIB_WIKI_FILE = "wikitext-2-raw-v1.train.raw"   # SOLO train: validation = pannello, test = PPL
# 25/9: corpus multi-dominio GIA' tokenizzato (costruisci_corpus_calibrazione.py): {dominio: (n, L_SEQ) int64}
CALIB_CORPUS = os.getenv("THOR_CALIB_CORPUS", "")


def _carica_corpus_pronto(percorso: str) -> dict:
    """Corpus pronto `{dominio: (n, L_SEQ) int64}`. Se accanto c'e' il manifesto (stesso nome,
    `.json`), lo sha256 dei token di OGNI dominio deve essere quello dichiarato."""
    import hashlib
    p = Path(percorso)
    corpus = torch.load(p, weights_only=True)
    if not isinstance(corpus, dict) or not corpus:
        raise ValueError(f"THOR_CALIB_CORPUS={p}: atteso un dizionario {{dominio: tensore}}")
    for nome, t in corpus.items():
        if not (isinstance(t, torch.Tensor) and t.dtype == torch.int64 and t.dim() == 2 and t.shape[1] == L_SEQ):
            raise ValueError(f"THOR_CALIB_CORPUS={p}: dominio {nome!r} non e' (n, {L_SEQ}) int64")
    man = p.with_suffix(".json")
    if man.exists():
        dichiarati = json.loads(man.read_text(encoding="utf-8")).get("domini", {})
        for nome, t in corpus.items():
            sha = hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
            if nome not in dichiarati or dichiarati[nome].get("sha256_token") != sha:
                raise RuntimeError(f"THOR_CALIB_CORPUS={p}: dominio {nome!r} non combacia col manifesto {man.name}")
    else:
        log(f"⚠️ THOR_CALIB_CORPUS={p}: manifesto {man.name} assente, impronte non verificate")
    return corpus
'''),
    # -- 2. quote: domini dal corpus, ciclo di batch
    ('''    if sum(quote.values()) != BATCH_SEQ or not set(quote) <= {"cot", "edu", "wiki"}:
        raise ValueError(f"THOR_CALIB_MISCELA={spec!r}: domini ammessi cot/edu/wiki, somma = BATCH_SEQ ({BATCH_SEQ})")
''',
     '''    corpus = _carica_corpus_pronto(CALIB_CORPUS) if CALIB_CORPUS else None
    ammessi = set(corpus) if corpus is not None else {"cot", "edu", "wiki"}
    somma = sum(quote.values())
    # 25/9: le quote valgono per un CICLO di `giro` batch (somma = giro x BATCH_SEQ; es. sedicesimi
    # con 8 sequenze per batch = ciclo di 2). giro 1 = la costruzione di prima, riga per riga.
    giro = somma // BATCH_SEQ if somma > 0 and somma % BATCH_SEQ == 0 else 0
    if giro < 1 or not set(quote) <= ammessi or min(quote.values()) < 0 or N_BATCH % giro:
        raise ValueError(f"THOR_CALIB_MISCELA={spec!r}: domini ammessi {sorted(ammessi)}, quote >= 0, somma = "
                         f"multiplo di BATCH_SEQ ({BATCH_SEQ}) e N_BATCH ({N_BATCH}) multiplo del ciclo "
                         f"({somma}/{BATCH_SEQ})" + (f" · corpus {CALIB_CORPUS}" if CALIB_CORPUS else ""))
'''),
    # -- 3. sequenze per dominio: dal corpus pronto se c'e'
    ('''        n = q * N_BATCH
        if q == 0:
            continue
        if nome == "cot":
''',
     '''        n = q * (N_BATCH // giro)
        if q == 0:
            continue
        if corpus is not None:
            if corpus[nome].shape[0] < n:
                raise RuntimeError(f"corpus {CALIB_CORPUS}: dominio {nome} ha {corpus[nome].shape[0]} sequenze, "
                                   f"ne servono {n} (mai sequenze ripetute)")
            pool[nome] = corpus[nome][:n].clone()
        elif nome == "cot":
'''),
    # -- 4. assemblaggio: ciclo 1 come prima; ciclo > 1 per posizione frazionaria
    ('''    batch = []
    for b in range(N_BATCH):
        batch.append(torch.cat([pool[k][b * q:(b + 1) * q] for k, q in quote.items() if q > 0], dim=0))
    return torch.stack(batch, dim=0)
''',
     '''    batch = []
    if giro == 1:
        for b in range(N_BATCH):
            batch.append(torch.cat([pool[k][b * q:(b + 1) * q] for k, q in quote.items() if q > 0], dim=0))
        return torch.stack(batch, dim=0)
    posti = sorted(((i + 0.5) / q, j, k) for j, (k, q) in enumerate(quote.items()) if q > 0 for i in range(q))
    ordine = [k for _, _, k in posti]
    preso = {k: 0 for k in quote}
    for _c in range(N_BATCH // giro):
        for g in range(giro):
            righe = []
            for k in ordine[g * BATCH_SEQ:(g + 1) * BATCH_SEQ]:
                righe.append(pool[k][preso[k]])
                preso[k] += 1
            batch.append(torch.stack(righe, dim=0))
    assert all(preso[k] == pool[k].shape[0] for k in pool), "ogni sequenza del pool va usata una volta"
    return torch.stack(batch, dim=0)
'''),
    # -- 5. corpus senza miscela = errore; log col corpus
    ('''    miscela = os.getenv("THOR_CALIB_MISCELA", "")
    if miscela:
''',
     '''    miscela = os.getenv("THOR_CALIB_MISCELA", "")
    if CALIB_CORPUS and not miscela:
        raise ValueError(f"THOR_CALIB_CORPUS={CALIB_CORPUS} senza THOR_CALIB_MISCELA: dichiarare le quote "
                         "(es. wiki:3,edu:2,cot:2,it_web:2,it_wiki:1,it_chat:3,it_pensiero:1,strumenti:2)")
    if miscela:
'''),
    ('''        log(f"calibrazione MISTA «{miscela}» (seme {seme}): {N_BATCH} batch di addestramento x "
            f"{BATCH_SEQ}x{L_SEQ} token + {n_ho} held-out storico ({usati_ho} esempi math/code chat)")
''',
     '''        log(f"calibrazione MISTA «{miscela}» (seme {seme}): {N_BATCH} batch di addestramento x "
            f"{BATCH_SEQ}x{L_SEQ} token + {n_ho} held-out storico ({usati_ho} esempi math/code chat)"
            + (f" · corpus pronto {CALIB_CORPUS}" if CALIB_CORPUS else ""))
'''),
]


def applica(testo: str) -> str:
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
    ap.add_argument("--prova", action="store_true", help="controlla gli agganci, non scrive")
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

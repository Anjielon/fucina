"""Patch DOMINI NUOVI DEL PANNELLO per pannello_heldout.py (25/9 notte).

Perche': il pannello held-out misura math/code in chat, Wikipedia inglese e FineWeb-Edu.
Il corpus di calibrazione multi-dominio (`costruisci_corpus_calibrazione.py`) aggiunge
italiano, chat senza ragionamento, ragionamento e strumenti: il pannello deve MISURARE
anche quelli, altrimenti una ricetta potrebbe peggiorarli senza che nessuno lo veda.
I domini nuovi stanno in un file A PARTE (`pannello/pannello_nuovi_domini.pt` + `.json`,
costruito disgiunto dalla calibrazione); `pannello_ids.pt/json` NON cambiano e il loro
controllo di determinismo resta sui soli domini storici.

  1. `DOMINI_STORICI` = i cinque di prima; `DOMINI_NUOVI` dal `.json` del file nuovo
     (`THOR_PANNELLO_NUOVI=<file.pt>` per un altro file; `THOR_PANNELLO_NUOVI_DOMINI=a,b`
     per un sottoinsieme, `=0` per nessuno); `DOMINI` = storici + nuovi;
  2. `aggiungi_nuovi_domini(p, scala)`: carica, verifica lo sha256 di ogni dominio col
     manifesto del file, a scala di prova ne prende la testa (come lo storico);
  3. `pannello_vero`: confronta e salva SOLO i domini storici (mai scritti in
     `pannello_ids.pt` i nuovi), poi aggiunge i nuovi;
  4. `prova_cpu`: aggiunge i nuovi dopo il controllo delle forme storiche (+ esito
     `A_nuovi_domini`), cosi' la catena giocattolo li attraversa tutti;
  6. la replica dell'impacchettamento storico (`verifica_replica`) gira SENZA
     THOR_CALIB_MISCELA ne' corpus pronto anche se l'ambiente li ha (ripristinati dopo);
  5. rapporto: tabella dei domini nuovi (rapporto appaiato contro v20 ± errore standard e
     massimo su TUTTI i domini), tabella delle loss assolute con tutte le colonne, e nei
     segnali il criterio del corpus: `criterio_tutti_sotto_1` per ricetta (R per dominio,
     tutti sotto 1, margine minimo, dominio peggiore) — meccanico, decide il coordinatore.

⚠️ Costo (stima, non misura): 64 batch in piu' (da 53 a 117). Il tempo della valutazione
cresce circa in proporzione (~+1 min per ricetta a ~1 s per batch); la VRAM dei semi e
del teacher precalcolati per TUTTE le batch (prepara_semi) cresce di ~10-18 GiB con la
finestra [0,1] e di piu' in profondita' (pannello_profondita.py). Se non ci sta: meno
domini con THOR_PANNELLO_NUOVI_DOMINI.

Uso:
    python3 strumenti/patch_pannello_nuovi_domini.py                  # applica a ../pannello_heldout.py
    python3 strumenti/patch_pannello_nuovi_domini.py --file COPIA.py  # applica a una copia
    python3 strumenti/patch_pannello_nuovi_domini.py --prova          # controlla soltanto, non scrive
Non scrive NULLA se un aggancio non compare esattamente una volta, se la patch e' gia'
applicata, o se il risultato non passa ast.parse. Scrittura atomica (os.replace).
"""
import argparse
import ast
import os
from pathlib import Path

PREDEFINITO = Path(__file__).resolve().parent.parent / "pannello_heldout.py"
MARCATORE = "DOMINI_NUOVI"

SOSTITUZIONI = [
    # -- 0. la replica dell'impacchettamento storico si fa SENZA miscela ne' corpus pronto: se
    #       l'ambiente di chi lancia il pannello li ha, carica_batch_calibrazione costruirebbe un'altra
    #       cosa (o rifiuterebbe il corpus senza miscela). Ripristinati subito dopo.
    ('''        salva_env = {k: os.environ.get(k) for k in ("THOR_VALIDA_N_HELDOUT", "THOR_CALIB_SEED")}
        try:
            V.N_BATCH, V.BATCH_SEQ, V.L_SEQ = nb, bs, L_SEQ_ADDESTRAMENTO
''',
     '''        salva_env = {k: os.environ.get(k) for k in ("THOR_VALIDA_N_HELDOUT", "THOR_CALIB_SEED", "THOR_CALIB_MISCELA")}
        salva_corpus = getattr(V, "CALIB_CORPUS", None)
        try:
            V.N_BATCH, V.BATCH_SEQ, V.L_SEQ = nb, bs, L_SEQ_ADDESTRAMENTO
            os.environ.pop("THOR_CALIB_MISCELA", None)
            if salva_corpus is not None:
                V.CALIB_CORPUS = ""
'''),
    ('''        finally:
            V.N_BATCH, V.BATCH_SEQ, V.L_SEQ = salva_attr
''',
     '''        finally:
            V.N_BATCH, V.BATCH_SEQ, V.L_SEQ = salva_attr
            if salva_corpus is not None:
                V.CALIB_CORPUS = salva_corpus
'''),
    # -- 1. domini nuovi dal file a parte
    ('''DOMINI = ("storico", "math_code", "wiki", "edu", "lunghe")
''',
     '''DOMINI = ("storico", "math_code", "wiki", "edu", "lunghe")
# 25/9: domini NUOVI (italiano, chat, ragionamento, strumenti) da un file A PARTE costruito da
# costruisci_corpus_calibrazione.py, disgiunto dalla calibrazione. pannello_ids.pt non cambia.
DOMINI_STORICI = DOMINI
FILE_NUOVI = Path(os.getenv("THOR_PANNELLO_NUOVI", str(DIR_DATI / "pannello_nuovi_domini.pt")))
_SCELTI_NUOVI = os.getenv("THOR_PANNELLO_NUOVI_DOMINI", "")        # vuoto = tutti · "0" = nessuno
if _SCELTI_NUOVI == "0" or not FILE_NUOVI.with_suffix(".json").exists():
    DOMINI_NUOVI: tuple = ()
else:
    _FORME_NUOVI = json.loads(FILE_NUOVI.with_suffix(".json").read_text(encoding="utf-8"))["forme"]
    DOMINI_NUOVI = tuple(d for d in _FORME_NUOVI if not _SCELTI_NUOVI or d in _SCELTI_NUOVI.split(","))
    _ignoti = set(_SCELTI_NUOVI.split(",")) - set(_FORME_NUOVI) if _SCELTI_NUOVI else set()
    if _ignoti:
        raise ValueError(f"THOR_PANNELLO_NUOVI_DOMINI: domini {sorted(_ignoti)} assenti da {FILE_NUOVI}")
DOMINI = DOMINI_STORICI + DOMINI_NUOVI
'''),
    # -- 2. caricatore dei domini nuovi (prima di pannello_vero)
    ('''def pannello_vero(tok, ricostruisci: bool = False) -> dict:
''',
     '''def aggiungi_nuovi_domini(p: dict, sc: Scala) -> dict:
    """Aggiunge a `p` i domini nuovi (file a parte, sha256 di ogni dominio verificato col suo
    manifesto). A scala di prova ne prende la testa, come `costruisci_pannello` con lo storico."""
    if not DOMINI_NUOVI:
        log(f"domini nuovi del pannello: nessuno ({FILE_NUOVI} assente o THOR_PANNELLO_NUOVI_DOMINI=0)")
        return p
    nuovi = torch.load(FILE_NUOVI, weights_only=False)
    meta = nuovi["meta"]
    for d in DOMINI_NUOVI:
        t = nuovi["domini"][d]
        if _sha(t) != meta["sha256_token"][d]:
            raise RuntimeError(f"{FILE_NUOVI}: dominio {d} non combacia con lo sha256 del suo manifesto")
        if sc != SCALA_VERA:
            t = t.reshape(-1)[:sc.seq_per_batch * sc.l_seq].reshape(1, sc.seq_per_batch, sc.l_seq)
        p["domini"][d] = t
        p["meta"]["forme"][d] = list(t.shape)
        p["meta"]["sha256_token"][d] = _sha(t)
    p["meta"]["nuovi_domini"] = {
        "file": str(FILE_NUOVI), "domini": list(DOMINI_NUOVI),
        "manifesto": {k: meta[k] for k in ("quando", "forme", "sha256_token", "seme", "soglia_pannello_hash",
                                           "k_massive", "costruttore") if k in meta},
        "disgiunzione_esito": (meta.get("disgiunzione") or {}).get("esito")}
    log(f"domini nuovi del pannello da {FILE_NUOVI.name}: " + ", ".join(
        f"{d} {tuple(p['domini'][d].shape)}" for d in DOMINI_NUOVI)
        + f" · disgiunzione {p['meta']['nuovi_domini']['disgiunzione_esito']}")
    return p


def pannello_vero(tok, ricostruisci: bool = False) -> dict:
'''),
    # -- 3. pannello_vero: confronto sui soli storici, poi i nuovi
    ('''        diversi = [k for k in DOMINI if not torch.equal(salvato["domini"][k], p["domini"][k])]
''',
     '''        diversi = [k for k in DOMINI_STORICI if not torch.equal(salvato["domini"][k], p["domini"][k])]
'''),
    ('''        log(f"pannello salvato: {FILE_PANNELLO} (+ .json)")
    return p
''',
     '''        log(f"pannello salvato: {FILE_PANNELLO} (+ .json)")
    return aggiungi_nuovi_domini(p, SCALA_VERA)
'''),
    # -- 4. prova su CPU: i nuovi entrano dopo il controllo delle forme storiche
    ('''    log("── B. codici VERI salvati: convenzione di ricostruzione su 2 esperti per strato ──")
''',
     '''    pannello = aggiungi_nuovi_domini(pannello, SCALA_PROVA)
    esiti["A_nuovi_domini"] = bool(DOMINI_NUOVI) and all(
        list(pannello["domini"][d].shape) == [1, SCALA_PROVA.seq_per_batch, SCALA_PROVA.l_seq] for d in DOMINI_NUOVI)
    log("── B. codici VERI salvati: convenzione di ricostruzione su 2 esperti per strato ──")
'''),
    # -- 5a. rapporto: tabella dei domini nuovi + loss assolute con tutte le colonne
    ('''    righe += ["", "## Loss dura media (assoluta)", "",
              "| ricetta | storico | math_code | wiki | edu | lunghe |", "|---|---|---|---|---|---|"]
''',
     '''    if DOMINI_NUOVI:
        tutti = ("math_code", "wiki", "edu") + DOMINI_NUOVI
        righe += ["", "## Domini nuovi (italiano, chat, ragionamento, strumenti): rapporto appaiato contro v20", "",
                  "| ricetta | " + " | ".join(DOMINI_NUOVI) + " | massimo su math_code, wiki, edu e nuovi |",
                  "|---|" + "---|" * (len(DOMINI_NUOVI) + 1)]
        for n in ordine:
            s = som[n]
            cella = [f"{_f(s[d]['rapporto_appaiato_vs_v20'])} ± {_f(s[d]['es'])}" for d in DOMINI_NUOVI]
            peggiore = max(tutti, key=lambda d: s[d]["rapporto_appaiato_vs_v20"])
            cella.append(f"{_f(s[peggiore]['rapporto_appaiato_vs_v20'])} ({peggiore})")
            righe.append(f"| {n} | " + " | ".join(cella) + " |")
    righe += ["", "## Loss dura media (assoluta)", "",
              "| ricetta | " + " | ".join(DOMINI) + " |", "|---|" + "---|" * len(DOMINI)]
'''),
    # -- 5b. segnali: il criterio del corpus (tutti i domini sotto 1)
    ('''    return out


def confronto_storico(som: dict) -> dict:
''',
     '''    criterio = ("math_code", "wiki", "edu") + DOMINI_NUOVI
    out["criterio_tutti_sotto_1"] = {
        n: {"R": {d: som[n][d]["rapporto_appaiato_vs_v20"] for d in criterio},
            "tutti_sotto_1": bool(all(som[n][d]["rapporto_appaiato_vs_v20"] < 1 for d in criterio)),
            "margine_minimo": 1 - max(som[n][d]["rapporto_appaiato_vs_v20"] for d in criterio),
            "dominio_peggiore": max(criterio, key=lambda d: som[n][d]["rapporto_appaiato_vs_v20"])}
        for n in som if n != "v20"}
    return out


def confronto_storico(som: dict) -> dict:
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

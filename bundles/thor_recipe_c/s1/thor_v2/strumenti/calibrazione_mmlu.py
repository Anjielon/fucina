#!/usr/bin/env python3
"""Calibrazione post-hoc sui logprob MMLU GIA' SALVATI — recupera punti? (27/9/2026)

## Perche' esiste
`analisi_mmlu_27_9.md` mostra che B/p12 scelgono la lettera C troppo spesso (727 volte contro 503 giuste,
il maestro 545) e che togliere una preferenza COSTANTE per lettera PEGGIORA (`mmlu_depurato.py`, variante
`glob`: p12 depurata 72,30 contro 72,80 grezza) — la preferenza non e' una costante togliibile, dipende dal
contenuto/materia. Questo script prova, SOLO sui log gia' scritti (nessuna GPU, nessuna nuova valutazione),
se una calibrazione che dipende dal contenuto (materia) o appresa fa meglio — e se il guadagno regge quando
si stima su una meta' dei compiti e si verifica sull'ALTRA (lo stesso disciplinamento che sarebbe mancato al
tentativo costante, la cui perdita non era necessariamente overfitting ma e' trattata qui come tale per
prudenza, come richiesto).

## I tre metodi, in ordine di quanto costano (nessuno rilancia il modello)
1. **Normalizzazione per lunghezza in token** (lm-eval `acc_norm`-style). Il formato MCLP porta 4 conteggi di
   token per opzione (`leggi_registro`, colonne 9-12): sono i conteggi del token-LETTERA valutato (' A'..' D'),
   non della risposta testuale intera — il banco valuta P(lettera | domanda+scelte), non P(testo intero della
   scelta). Il codice di produzione (`mmlu_depurato.leggi_registro`) gia' impone `c[9:13] == ["1","1","1","1"]`
   e solleva altrimenti: la lunghezza e' SEMPRE 1 token per costruzione. **Non applicabile**: non c'e' varianza
   da normalizzare, verificato qui di nuovo (non solo dedotto dal codice altrui) contando i valori distinti.
2. **PMI domain-conditional approssimata**: per ogni compito, calibra sottraendo il logprob medio (log-softmax
   sulle 4 lettere, S = logsm(lp)) di ciascuna POSIZIONE calcolato sulle ALTRE domande della stessa materia.
   E' l'approssimazione di Holtzman et al. 2021 (Surface Form Competition, arXiv:2104.08315) applicata al
   token-ID invece che alla forma testuale, e di Zheng et al. 2024 (PriDe, arXiv:2309.03882) Eq. 7 SENZA le
   permutazioni cicliche del contenuto che l'equazione richiederebbe (qui indisponibili: servirebbero 4 passaggi
   nuovi per compito) — si usano le domande DIVERSE della stessa materia come surrogato del contenuto ruotato.
   Dichiarata approssimazione, non l'Eq. 7 esatta. **Senza etichette** (a differenza della variante "glob" di
   `mmlu_depurato.py`, che bilancia sulla lettera giusta): e' la lettura letterale del compito, e la piu' vicina
   a Contextual Calibration (Zhao et al. 2021) e Batch Calibration (Zhou et al. 2024), che calibrano SENZA MAI
   guardare l'etichetta.
3. **Offset globale per lettera** (metodo 3, forma minima: media di S su tutte le materie insieme invece che
   per materia) e **regressione logistica multinomiale** sui 4 punteggi S (metodo 3, forma "se facile"): in
   entrambi la calibrazione e' stimata su una META' dei 2000 compiti (indice pari/dispari, non un prefisso: i
   compiti sono ordinati per materia a blocchi contigui, e la parita' dell'indice tiene ogni materia in ENTRAMBE
   le meta') e verificata SULL'ALTRA — mai sugli stessi compiti che si valutano, a differenza del punto (2) dove
   la stima resta dentro la materia ma sui compiti del proprio fold di stima (stesso schema a 2 pieghe).

## Onesta' sui log che mancano
Nessuno dei tre metodi usa mai un passaggio del modello non gia' registrato: (1) e' una verifica, non un
calcolo; (2) e (3) leggono solo i 4 logprob per compito gia' scritti nei file MCLP. La vera Eq. 7 di PriDe
(permutazioni cicliche del contenuto) e Contextual Calibration/PMI_DC nella forma esatta (un prompt
content-free mai eseguito) NON sono calcolabili da qui: lo dice la conclusione in fondo al rapporto.

## Dati
- Mappa materia/lettera: `thor_v2/risultati/mmlu_per_domanda.json` (stesso sha256 dei compiti di
  `mmlu_depurato.py`, verificato).
- Registri MCLP: `B.log` (Corsair) e `aragorn/EB2-migliore_000012.log` (p12, Aragorn) in
  `@@ROOT:models@@/gguf/tony-tern/mmlu_logprob/`. Parsing INDIPENDENTE da `mmlu_depurato.py` (stessa definizione
  di riga MCLP, riletta qui con una propria espressione regolare) per non dover importare — e non rilanciare
  per errore — il suo bootstrap a 10.000 ricampionamenti: qui bastano somme per materia, nessun bootstrap.

Uso (CPU confinata, non tocca la GPU ne' la corsa e2e-allena-EB2):
    nice -n 19 env OMP_NUM_THREADS=2 python3 thor_v2/strumenti/calibrazione_mmlu.py [--json USCITA.json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

QUI = Path(__file__).resolve().parent            # thor_v2/strumenti
THOR_V2 = QUI.parent
FUCINA = THOR_V2.parent
SB = Path("@@ROOT:models@@/gguf/tony-tern/mmlu_logprob")
MAPPA = THOR_V2 / "risultati" / "mmlu_per_domanda.json"
SHA_BANCO = "af987cc3369c15f90df9f390def043fd9fa4787e075b3b2170ff697a9dabbc6a"
N = 2000
LETTERE = "ABCD"
SEME = 20260927

REGISTRI = {
    "B": SB / "B.log",
    "p12": SB / "aragorn" / "EB2-migliore_000012.log",
}

RE_MCLP = re.compile(
    r"^MCLP\t(\d+)\t(\d)\t(\d)\t4\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t(\d+)\t(\d+)\t(\d+)\t(\d+)$"
)


# ─────────────────────────────────────────── lettura (indipendente da mmlu_depurato.py)
def leggi_registro(p: Path) -> dict:
    """Righe MCLP: indice, giusta, scelta, 4 logprob, 4 conteggi di token per opzione."""
    lp = np.full((N, 4), np.nan, dtype=np.float64)
    giusta = np.full(N, -1, dtype=np.int64)
    conteggi = np.zeros((N, 4), dtype=np.int64)
    atteso = 0
    for riga in p.read_text(encoding="utf-8").splitlines():
        m = RE_MCLP.match(riga)
        if not m:
            continue
        idx = int(m.group(1))
        if idx != atteso:
            raise SystemExit(f"⛔ {p}: attesa riga MCLP {atteso}, trovata {idx}")
        giusta[idx] = int(m.group(2))
        lp[idx] = [float(m.group(k)) for k in (4, 5, 6, 7)]
        conteggi[idx] = [int(m.group(k)) for k in (8, 9, 10, 11)]
        atteso += 1
    if atteso != N or not np.isfinite(lp).all() or (lp > 0).any():
        raise SystemExit(f"⛔ {p}: {atteso}/{N} righe MCLP lette, o logprob non finiti/positivi")
    return {"lp": lp, "giusta": giusta, "conteggi": conteggi}


def logsm(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    return x - (m + np.log(np.exp(x - m).sum(axis=-1, keepdims=True)))


def acc(scelta: np.ndarray, giusta: np.ndarray, sel: np.ndarray | None = None) -> float:
    if sel is not None:
        scelta, giusta = scelta[sel], giusta[sel]
    return 100.0 * float((scelta == giusta).mean())


def mcnemar_netto(prima_ok: np.ndarray, dopo_ok: np.ndarray, sel: np.ndarray | None = None) -> tuple[int, int]:
    """(persa, guadagnata): compiti giusti PRIMA e sbagliati DOPO, e viceversa."""
    if sel is not None:
        prima_ok, dopo_ok = prima_ok[sel], dopo_ok[sel]
    persa = int((prima_ok & ~dopo_ok).sum())
    guadagnata = int((~prima_ok & dopo_ok).sum())
    return persa, guadagnata


# ─────────────────────────────────────────── metodo (1): lunghezza in token
def normalizzazione_lunghezza(conteggi: np.ndarray) -> dict:
    valori = sorted(set(int(x) for x in conteggi.flatten()))
    return {
        "applicabile": valori != [1],
        "valori_distinti_osservati": valori,
        "nota": "il banco valuta P(lettera | prompt), 1 token per costruzione: nessuna lunghezza da normalizzare"
        if valori == [1] else "ATTENZIONE: trovati conteggi != 1, la normalizzazione andrebbe applicata",
    }


# ─────────────────────────────────────────── metodo (2): PMI per materia (senza etichette)
def prior_materia_loo(S: np.ndarray, mat_idx: np.ndarray, n_mat: int) -> np.ndarray:
    """Leave-one-out ESATTO: la preferenza del compito q e' la media di S sulle ALTRE domande della sua materia
    (nessuna delle due, ne' la stima ne' la valutazione, usa la propria riga)."""
    somme = np.zeros((n_mat, 4))
    conti = np.zeros(n_mat)
    for m in range(n_mat):
        sel = mat_idx == m
        somme[m] = S[sel].sum(axis=0)
        conti[m] = sel.sum()
    prior = np.empty_like(S)
    for q in range(len(S)):
        m = mat_idx[q]
        media = (somme[m] - S[q]) / (conti[m] - 1)
        prior[q] = logsm(media[None])[0]
    return prior


def prior_materia_2fold(S: np.ndarray, mat_idx: np.ndarray, n_mat: int, piega: np.ndarray) -> np.ndarray:
    """Come sopra ma a 2 PIEGHE (indice pari/dispari): la preferenza per i compiti della piega f e' stimata
    SOLO sui compiti dell'ALTRA piega della stessa materia — stesso disciplinamento train/verifica del (3)."""
    prior = np.empty_like(S)
    for f in (0, 1):
        stima = piega != f
        verifica = piega == f
        for m in range(n_mat):
            sel_stima = stima & (mat_idx == m)
            sel_verifica = verifica & (mat_idx == m)
            if sel_stima.sum() == 0 or sel_verifica.sum() == 0:
                continue
            media = S[sel_stima].mean(axis=0)
            prior[sel_verifica] = logsm(media[None])[0]
    return prior


# ─────────────────────────────────────────── metodo (3a): offset globale per lettera, 2 pieghe
def prior_globale_2fold(S: np.ndarray, piega: np.ndarray) -> np.ndarray:
    prior = np.empty_like(S)
    for f in (0, 1):
        media = S[piega != f].mean(axis=0)
        prior[piega == f] = logsm(media[None])[0]
    return prior


# ─────────────────────────────────────────── metodo (3b): regressione logistica sui 4 punteggi, 2 pieghe
def scelta_logistica_2fold(S: np.ndarray, giusta: np.ndarray, piega: np.ndarray, seme: int) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression

    scelta = np.full(len(S), -1, dtype=np.int64)
    for f in (0, 1):
        stima, verifica = piega != f, piega == f
        clf = LogisticRegression(max_iter=5000, random_state=seme)
        clf.fit(S[stima], giusta[stima])
        prob = np.zeros((int(verifica.sum()), 4))
        for j, classe in enumerate(clf.classes_):
            prob[:, int(classe)] = clf.predict_proba(S[verifica])[:, j]
        scelta[verifica] = prob.argmax(axis=1)
    return scelta


# ─────────────────────────────────────────── valutazione comune di un metodo
def valuta_metodo(nome: str, scelta_dopo: np.ndarray, scelta_prima: np.ndarray, giusta: np.ndarray,
                   piega: np.ndarray) -> dict:
    prima_ok, dopo_ok = scelta_prima == giusta, scelta_dopo == giusta
    righe = {}
    for et, sel in (("piega0", piega == 0), ("piega1", piega == 1), ("totale", np.ones(N, dtype=bool))):
        a_prima, a_dopo = acc(scelta_prima, giusta, sel), acc(scelta_dopo, giusta, sel)
        persa, guadagnata = mcnemar_netto(prima_ok, dopo_ok, sel)
        righe[et] = {"n": int(sel.sum()), "grezza": a_prima, "calibrata": a_dopo, "delta": a_dopo - a_prima,
                     "persa": persa, "guadagnata": guadagnata, "netto_guadagno": guadagnata - persa}
    segno0, segno1 = np.sign(righe["piega0"]["delta"]), np.sign(righe["piega1"]["delta"])
    confermato = bool(segno0 == segno1 and segno0 != 0)
    return {"metodo": nome, **righe, "cross_validato_stesso_segno": confermato}


# ─────────────────────────────────────────── principale
def elabora(chiave: str, reg: Path, mat_idx: np.ndarray, n_mat: int, piega: np.ndarray) -> dict:
    r = leggi_registro(reg)
    lp, giusta, conteggi = r["lp"], r["giusta"], r["conteggi"]
    S = logsm(lp)
    scelta_grezza = lp.argmax(axis=1)          # = S.argmax(axis=1): logsm e' monotona per riga

    lung = normalizzazione_lunghezza(conteggi)

    pmi_loo = prior_materia_loo(S, mat_idx, n_mat)
    scelta_pmi_loo = (S - pmi_loo).argmax(axis=1)
    pmi_2f = prior_materia_2fold(S, mat_idx, n_mat, piega)
    scelta_pmi_2f = (S - pmi_2f).argmax(axis=1)

    off_2f = prior_globale_2fold(S, piega)
    scelta_off_2f = (S - off_2f).argmax(axis=1)

    scelta_log_2f = scelta_logistica_2fold(S, giusta, piega, SEME)

    return {
        "registro": str(reg), "n": N,
        "normalizzazione_lunghezza": lung,
        "grezza_totale": acc(scelta_grezza, giusta),
        "metodi": [
            {"metodo": "pmi_materia_loo (informativo, NON a 2 pieghe: stima e verifica nella stessa materia, "
                       "ma mai sulla propria riga)",
             "totale": {"grezza": acc(scelta_grezza, giusta), "calibrata": acc(scelta_pmi_loo, giusta),
                        "delta": acc(scelta_pmi_loo, giusta) - acc(scelta_grezza, giusta)}},
            valuta_metodo("pmi_materia_2fold", scelta_pmi_2f, scelta_grezza, giusta, piega),
            valuta_metodo("offset_globale_2fold", scelta_off_2f, scelta_grezza, giusta, piega),
            valuta_metodo("logistica_2fold", scelta_log_2f, scelta_grezza, giusta, piega),
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    mappa = json.loads(MAPPA.read_text(encoding="utf-8"))
    if mappa["sha256_compiti"] != SHA_BANCO:
        raise SystemExit("⛔ sha256 dei compiti diverso da quello atteso — mappa cambiata sotto i piedi")
    materia = mappa["materia"]
    materie = sorted(set(materia), key=materia.index)
    if len(materie) != 15:
        raise SystemExit(f"⛔ attese 15 materie, trovate {len(materie)}")
    mat_idx = np.array([materie.index(m) for m in materia])
    piega = np.arange(N) % 2       # indice pari/dispari: ogni materia (blocco contiguo) in ENTRAMBE le pieghe

    uscita = {"generato": "thor_v2/strumenti/calibrazione_mmlu.py", "seme": SEME, "n_materie": len(materie),
              "modelli": {}}
    for chiave, reg in REGISTRI.items():
        uscita["modelli"][chiave] = elabora(chiave, reg, mat_idx, len(materie), piega)

    for chiave, r in uscita["modelli"].items():
        print(f"\n=== {chiave} ({r['registro']}) — grezza totale {r['grezza_totale']:.2f} % ===")
        print(f"  lunghezza in token: {r['normalizzazione_lunghezza']}")
        for m in r["metodi"]:
            if "totale" in m:
                t = m["totale"]
                print(f"  {m['metodo']}: {t['grezza']:.2f} -> {t['calibrata']:.2f} % (Δ {t['delta']:+.2f})")
            else:
                p0, p1, tot = m["piega0"], m["piega1"], m["totale"]
                print(f"  {m['metodo']}: piega0 {p0['grezza']:.2f}->{p0['calibrata']:.2f} (Δ{p0['delta']:+.2f}, "
                      f"n={p0['n']}) · piega1 {p1['grezza']:.2f}->{p1['calibrata']:.2f} (Δ{p1['delta']:+.2f}, "
                      f"n={p1['n']}) · totale Δ{tot['delta']:+.2f} · stesso segno: {m['cross_validato_stesso_segno']}")

    if a.json:
        Path(a.json).write_text(json.dumps(uscita, indent=1, ensure_ascii=False))
        print(f"\nscritto: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

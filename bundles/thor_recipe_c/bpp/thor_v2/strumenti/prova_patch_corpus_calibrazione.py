#!/usr/bin/env python3
"""Prova SOLO CPU delle due patch del corpus di calibrazione su COPIE (25/9 notte).

Copia `valida_finestra_8_12.py` e `pannello_heldout.py` in una cartella di prova, applica
`patch_corpus_calibrazione.py` e `patch_pannello_nuovi_domini.py` alle COPIE (gli originali
non si toccano), poi, in processi separati con la cartella di prova DAVANTI nel path (gli
altri moduli di thor_v2 vengono dagli originali):

  V1  originale, M2 (cot:2,edu:3,wiki:3), 32x8, 1 held-out            -> riferimento
  V2  copia patchata, stessa cosa SENZA corpus                          -> identica a V1 al bit
  V3  copia patchata, M2 CON corpus pronto                              -> confronto per dominio con V1
  V4  originale e copia senza miscela (solo cot, percorso storico)      -> identiche al bit
  V5  copia, corpus + U (sedicesimi, ciclo di 2 batch), 150 batch       -> forma, held-out identico,
      ogni sequenza del pool usata UNA volta, composizione esatta per ciclo
  V6  rifiuti rumorosi: corpus senza miscela, somma non multipla, dominio ignoto, N_BATCH non
      multiplo del ciclo, pool insufficiente, manifesto manomesso
  P1  pannello copia `--solo-pannello`: storici identici a pannello_ids.pt, nuovi aggiunti e
      verificati; nessun file del pannello riscritto
  P0  (nota) l'ORIGINALE `--prova-cpu`: i codici di Q1 e Q3 non sono piu' su disco, il passo B fallisce
  P2  pannello copia `--prova-cpu` (strati giocattolo), passo B sulle ricette i cui codici esistono:
      tutti gli esiti veri, domini nuovi compresi
  P3  pannello copia `--solo-pannello` con THOR_CALIB_CORPUS + THOR_CALIB_MISCELA nell'ambiente:
      la replica storica li ignora

Uso:
    env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= nice -n 19 \\
        @@ROOT:venv@@/bin/python3 strumenti/prova_patch_corpus_calibrazione.py [--cartella DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

QUI = Path(__file__).resolve().parent
THOR = QUI.parent
PY = "@@ROOT:venv@@/bin/python3"
CORPUS = Path(os.getenv("THOR_CALIB_CORPUS_PROVA", "@@ROOT:models@@/gguf/tony-tern/calibrazione/calibrazione_corpus.pt"))
PANNELLO = Path("@@ROOT:models@@/gguf/tony-tern/pannello")
PREDEF = Path("/tmp/claude-1000/-home-angelo-MOGAVIS/12e4589a-507e-45a0-b690-66b786e2b26f/scratchpad/prova_corpus/copie")
M2 = "cot:2,edu:3,wiki:3"
U = "wiki:3,edu:2,cot:2,it_web:2,it_wiki:1,it_chat:3,it_pensiero:1,strumenti:2"

esiti: dict = {}


def log(*a) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def ambiente(extra: dict, cartella: "Path | None") -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("THOR_CALIB")}
    env.update({"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "TORCHDYNAMO_DISABLE": "1",
                "OMP_NUM_THREADS": "4", "RAYON_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "true",
                "PYTHONPATH": ":".join(([str(cartella)] if cartella else []) + [str(THOR)])})
    env.update(extra)
    return env


def lancia_batch(cartella: "Path | None", uscita: Path, extra: dict) -> subprocess.CompletedProcess:
    """carica_batch_calibrazione in un processo a parte; `cartella` None = originale."""
    radice = str(cartella) if cartella else str(THOR)
    codice = (f"import sys, torch; sys.path.insert(0, {radice!r}); import valida_finestra_8_12 as V; "
              f"assert V.__file__.startswith({radice!r}), V.__file__; ids, n = V.carica_batch_calibrazione(); "
              f"torch.save({{'ids': ids, 'n': n}}, {str(uscita)!r})")
    env = ambiente({"THOR_VALIDA_BATCH_SEQ": "8", "THOR_VALIDA_N_HELDOUT": "1", **extra}, cartella)
    return subprocess.run(["nice", "-n", "19", PY, "-c", codice], env=env, capture_output=True, text=True,
                          timeout=1800, cwd=radice)


def segna(nome: str, ok: bool, dettaglio: str = "") -> None:
    esiti[nome] = bool(ok)
    log(f"{'✅' if ok else '⛔'} {nome} {dettaglio}")


def main() -> int:
    import torch
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cartella", type=Path, default=PREDEF)
    a = ap.parse_args()
    C = a.cartella
    if C.exists():
        shutil.rmtree(C)
    C.mkdir(parents=True)
    impronte_prima = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [THOR / "valida_finestra_8_12.py", THOR / "pannello_heldout.py",
                                PANNELLO / "pannello_ids.pt", PANNELLO / "pannello_ids.json"]}
    # ── copie + patch
    for nome in ("valida_finestra_8_12.py", "pannello_heldout.py"):
        shutil.copy2(THOR / nome, C / nome)
    for patch, bersaglio in (("patch_corpus_calibrazione.py", "valida_finestra_8_12.py"),
                             ("patch_pannello_nuovi_domini.py", "pannello_heldout.py")):
        r = subprocess.run([PY, str(QUI / patch), "--file", str(C / bersaglio)], capture_output=True, text=True)
        segna(f"patch {patch}", r.returncode == 0, (r.stdout + r.stderr).strip())
        r2 = subprocess.run([PY, str(QUI / patch), "--file", str(C / bersaglio)], capture_output=True, text=True)
        segna(f"patch {patch} rifiuta la seconda applicazione", r2.returncode != 0 and "gia' applicata" in r2.stdout + r2.stderr)
    import ast
    for nome in ("valida_finestra_8_12.py", "pannello_heldout.py"):
        ast.parse((C / nome).read_text())
        segna(f"ast.parse {nome} (copia)", True)

    corpus = torch.load(CORPUS, weights_only=True)
    out = C / "uscite"
    out.mkdir()

    def carica(p):
        return torch.load(p, weights_only=True)

    # ── V1/V2/V3: M2
    r1 = lancia_batch(None, out / "v1.pt", {"THOR_VALIDA_N_BATCH": "32", "THOR_CALIB_MISCELA": M2})
    r2 = lancia_batch(C, out / "v2.pt", {"THOR_VALIDA_N_BATCH": "32", "THOR_CALIB_MISCELA": M2})
    r3 = lancia_batch(C, out / "v3.pt", {"THOR_VALIDA_N_BATCH": "32", "THOR_CALIB_MISCELA": M2,
                                        "THOR_CALIB_CORPUS": str(CORPUS)})
    for n, r in (("V1", r1), ("V2", r2), ("V3", r3)):
        if r.returncode:
            segna(f"{n} gira", False, r.stderr[-1500:])
            return 1
    v1, v2, v3 = carica(out / "v1.pt"), carica(out / "v2.pt"), carica(out / "v3.pt")
    segna("V2 copia senza corpus == originale (M2, 32x8)", torch.equal(v1["ids"], v2["ids"]) and v1["n"] == v2["n"],
          f"forma {tuple(v1['ids'].shape)}")
    # M2 a ciclo 1: in ogni batch le posizioni 0:2 cot, 2:5 edu, 5:8 wiki
    dettagli = {}
    for dom, (a0, a1) in (("cot", (0, 2)), ("edu", (2, 5)), ("wiki", (5, 8))):
        s1 = v1["ids"][:32, a0:a1].reshape(-1, 1024)
        s3 = v3["ids"][:32, a0:a1].reshape(-1, 1024)
        uguali = (s1 == s3).all(dim=1)
        primo_diverso = int((~uguali).nonzero()[0]) if (~uguali).any() else None
        dettagli[dom] = {"sequenze": int(s1.shape[0]), "uguali": int(uguali.sum()), "prima_diversa": primo_diverso,
                         "corpus_uguale_al_pool": bool(torch.equal(s3, corpus[dom][:s3.shape[0]]))}
    log(f"   V3 (M2 col corpus) contro V1, per dominio: {json.dumps(dettagli)}")
    segna("V3 held-out storico identico", torch.equal(v1["ids"][32:], v3["ids"][32:]))
    segna("V3 wiki identico al bit", dettagli["wiki"]["uguali"] == dettagli["wiki"]["sequenze"])
    segna("V3 ogni dominio = testa del pool del corpus", all(d["corpus_uguale_al_pool"] for d in dettagli.values()))
    man = json.loads(CORPUS.with_suffix(".json").read_text())
    esc_cot = man["domini"]["cot"].get("esclusi_per_contaminazione", [])
    esc_edu = man["domini"]["edu"].get("documenti_esclusi_per_contaminazione", [])
    esiti["V3_nota"] = (f"cot identico fino alla sequenza {dettagli['cot']['prima_diversa']} (il corpus toglie gli "
                        f"esempi {esc_cot} per n-grammi/testi in comune col pannello); edu fino alla "
                        f"{dettagli['edu']['prima_diversa']} (documenti tolti {esc_edu})")
    log("   " + esiti["V3_nota"])

    # ── V4: percorso storico senza miscela
    r4a = lancia_batch(None, out / "v4a.pt", {"THOR_VALIDA_N_BATCH": "32"})
    r4b = lancia_batch(C, out / "v4b.pt", {"THOR_VALIDA_N_BATCH": "32"})
    segna("V4 senza miscela: copia == originale", r4a.returncode == 0 and r4b.returncode == 0
          and torch.equal(carica(out / "v4a.pt")["ids"], carica(out / "v4b.pt")["ids"]))

    # ── V5: corpus + U, 150 batch
    r5 = lancia_batch(C, out / "v5.pt", {"THOR_VALIDA_N_BATCH": "150", "THOR_CALIB_MISCELA": U,
                                        "THOR_CALIB_CORPUS": str(CORPUS)})
    if r5.returncode:
        segna("V5 gira", False, r5.stderr[-1500:])
        return 1
    v5 = carica(out / "v5.pt")
    ids = v5["ids"]
    segna("V5 forma (151, 8, 1024) e n=150", tuple(ids.shape) == (151, 8, 1024) and v5["n"] == 150)
    segna("V5 held-out identico all'originale", torch.equal(ids[150], v1["ids"][32]))
    quote = {k: int(v) for k, v in (x.split(":") for x in U.split(","))}
    righe = {}
    for dom, t in corpus.items():
        for i in range(t.shape[0]):
            righe.setdefault(hashlib.sha256(t[i].numpy().tobytes()).hexdigest(), []).append((dom, i))
    visti, ok_ciclo = {}, True
    for b in range(0, 150, 2):
        conta_ciclo = {}
        for bb in (b, b + 1):
            for s in range(8):
                h = hashlib.sha256(ids[bb, s].numpy().tobytes()).hexdigest()
                dom, i = righe[h][0]
                visti.setdefault(dom, []).append(i)
                conta_ciclo[dom] = conta_ciclo.get(dom, 0) + 1
        ok_ciclo &= conta_ciclo == {k: v for k, v in quote.items() if v}
    una_volta = all(sorted(v) == list(range(quote[d] * 75)) for d, v in visti.items())
    segna("V5 composizione esatta per ciclo di 2 batch", ok_ciclo, json.dumps({d: len(v) for d, v in visti.items()}))
    segna("V5 ogni sequenza del pool usata UNA volta, nell'ordine del file", una_volta
          and all(v == sorted(v) for v in visti.values()))
    per_batch = [{d: sum(1 for s in range(8) if righe[hashlib.sha256(ids[bb, s].numpy().tobytes()).hexdigest()][0][0] == d)
                  for d in quote} for bb in (0, 1)]
    log(f"   V5 composizione dei primi due batch: {per_batch}")

    # ── V6: rifiuti
    casi = {
        "corpus senza miscela": {"THOR_VALIDA_N_BATCH": "32", "THOR_CALIB_CORPUS": str(CORPUS)},
        "somma non multipla di 8": {"THOR_VALIDA_N_BATCH": "32", "THOR_CALIB_CORPUS": str(CORPUS),
                                    "THOR_CALIB_MISCELA": "wiki:3,edu:2,cot:2,it_web:2,it_chat:3"},
        "dominio ignoto": {"THOR_VALIDA_N_BATCH": "32", "THOR_CALIB_CORPUS": str(CORPUS),
                           "THOR_CALIB_MISCELA": "wiki:4,xyz:4"},
        "dominio nuovo senza corpus": {"THOR_VALIDA_N_BATCH": "32", "THOR_CALIB_MISCELA": "wiki:4,it_chat:4"},
        "N_BATCH non multiplo del ciclo": {"THOR_VALIDA_N_BATCH": "31", "THOR_CALIB_CORPUS": str(CORPUS),
                                           "THOR_CALIB_MISCELA": U},
        "pool insufficiente (400 batch)": {"THOR_VALIDA_N_BATCH": "400", "THOR_CALIB_CORPUS": str(CORPUS),
                                           "THOR_CALIB_MISCELA": U},
    }
    finto = C / "corpus_manomesso.pt"
    shutil.copy2(CORPUS, finto)
    mf = json.loads(CORPUS.with_suffix(".json").read_text())
    mf["domini"]["wiki"]["sha256_token"] = "0" * 64
    finto.with_suffix(".json").write_text(json.dumps(mf))
    casi["manifesto manomesso"] = {"THOR_VALIDA_N_BATCH": "32", "THOR_CALIB_CORPUS": str(finto),
                                   "THOR_CALIB_MISCELA": M2}
    for nome, env in casi.items():
        r = lancia_batch(C, out / "rifiuto.pt", env)
        ultima = (r.stderr.strip().splitlines() or [""])[-1]
        segna(f"V6 rifiuta: {nome}", r.returncode != 0 and ("Error" in ultima), ultima[:220])

    # ── P1/P3: pannello --solo-pannello sulla copia
    for nome, extra in (("P1", {}), ("P3", {"THOR_CALIB_CORPUS": str(CORPUS), "THOR_CALIB_MISCELA": U})):
        r = subprocess.run(["nice", "-n", "19", PY, str(C / "pannello_heldout.py"), "--solo-pannello"],
                           env=ambiente({"THOR_PANNELLO_USCITA": str(C / "pannello_uscite"), **extra}, C),
                           capture_output=True, text=True, timeout=1800, cwd=str(C))
        testo = r.stdout + r.stderr
        segna(f"{nome} pannello --solo-pannello (copia){' con corpus+miscela nell ambiente' if extra else ''}",
              r.returncode == 0 and "identico a quello salvato" in testo and "domini nuovi del pannello da" in testo,
              " | ".join(x.split("] ", 1)[-1][:200] for x in testo.splitlines() if "identico" in x or "domini nuovi" in x))
        if r.returncode:
            log(testo[-2000:])

    # ── P2: pannello --prova-cpu sulla copia. Il passo B legge i codici VERI di K2/Q1..Q4 dal disco:
    # quelli di Q1 e Q3 sono stati cancellati (pulizia del 24/9 sera), e l'ORIGINALE fallisce li'
    # allo stesso modo (P0, sotto). Nella prova si tengono le ricette i cui codici esistono ancora.
    avvio = ("import sys; sys.path.insert(0, {c!r}); sys.argv = ['pannello_heldout.py', '--prova-cpu', '--id', {i!r}]; "
             "import pannello_heldout as P; assert P.__file__.startswith({c!r}), P.__file__; "
             "manca = [k for k, v in P.RICETTE.items() if not (v / 'Dg_00.npy').exists()]; "
             "P.RICETTE = {{k: v for k, v in P.RICETTE.items() if k not in manca}}; "
             "print('RICETTE senza codici su disco (escluse dal passo B):', manca, flush=True); sys.exit(P.main())")
    r0 = subprocess.run(["nice", "-n", "19", PY, str(THOR / "pannello_heldout.py"), "--prova-cpu", "--id", "prova-originale"],
                        env=ambiente({"THOR_PANNELLO_USCITA": str(C / "pannello_uscite_originale"),
                                      "THOR_PANNELLO_SCRATCH": str(C / "codici_giocattolo_originale")}, None),
                        capture_output=True, text=True, timeout=3600, cwd=str(THOR))
    esiti["P0_nota_originale"] = ("l'ORIGINALE --prova-cpu " + ("passa" if r0.returncode == 0 else
                                  "fallisce: " + (r0.stderr.strip().splitlines() or [""])[-1][:200]))
    log("   P0 " + esiti["P0_nota_originale"])
    r = subprocess.run(["nice", "-n", "19", PY, "-c", avvio.format(c=str(C), i="prova-corpus")],
                       env=ambiente({"THOR_PANNELLO_USCITA": str(C / "pannello_uscite"),
                                     "THOR_PANNELLO_SCRATCH": str(C / "codici_giocattolo")}, C),
                       capture_output=True, text=True, timeout=3600, cwd=str(C))
    testo = r.stdout + r.stderr
    log("   " + next((x for x in testo.splitlines() if x.startswith("RICETTE senza codici")), ""))
    righe_esito = [x.split("] ", 1)[-1] for x in testo.splitlines() if ("✅" in x or "⛔" in x)]
    segna("P2 pannello --prova-cpu (copia)", r.returncode == 0 and "PROVA CPU SUPERATA" in testo, " | ".join(righe_esito))
    pj = C / "pannello_uscite" / "prova_cpu" / "pannello_f01_prova-corpus.json"
    if pj.exists():
        rap = json.loads(pj.read_text())
        segna("P2 rapporto contiene i domini nuovi e il criterio", all(
            d in rap["sommario"]["v20"] for d in ("it_web", "it_wiki", "it_chat", "it_pensiero", "strumenti_en",
                                                   "strumenti_it")) and "criterio_tutti_sotto_1" in rap["segnali_A1"])
        md = pj.with_suffix(".md").read_text()
        segna("P2 markdown con la tabella dei domini nuovi", "## Domini nuovi" in md)
    else:
        log(testo[-3000:])

    impronte_dopo = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [THOR / "valida_finestra_8_12.py", THOR / "pannello_heldout.py",
                               PANNELLO / "pannello_ids.pt", PANNELLO / "pannello_ids.json"]}
    segna("originali e pannello_ids intatti", impronte_prima == impronte_dopo)
    (C / "esiti.json").write_text(json.dumps(esiti, ensure_ascii=False, indent=1))
    tutti = all(v for k, v in esiti.items() if isinstance(v, bool))
    log(f"{'✅ TUTTE LE PROVE SUPERATE' if tutti else '⛔ PROVE FALLITE'} · esiti in {C / 'esiti.json'}")
    return 0 if tutti else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Collaudo della SONDA LINEARE PER STRATO, SOLO CPU (28/9/2026). Deve saper FALLIRE: ogni controllo e' anche rifatto
sotto una mutazione che lo deve far diventare rosso.

A — stati FINTI con verita' nota (sonda.py intera: caricamento, CV, statistica, localizzazione, grafico)
  A1  segnale piantato: maestro con la risposta leggibile da uno strato in poi; studente IDENTICO fino allo strato K,
      poi segnale attenuato (0,35x) e rumore in piu'. Attesi: caso negli strati senza segnale; maestro alto in fondo;
      primo strato con Δll significativo == K e significativo da K in poi; nessun falso positivo prima; sonda del
      maestro sugli stati dello studente = maestro prima di K, piu' bassa dopo; CKA ~1 prima di K e piu' basso dopo;
      frazione della perdita al confine (K−2) ~0.
  A2  etichette MESCOLATE (stati generati con le vere): ogni strato al caso, nessuno significativo.
  A3  modelli IDENTICI: Δ = 0 ovunque, nessuno significativo, CKA 1, errore 0.
  A4  rifiuti: id diversi, domande.json diverso, forme diverse, stati non finiti, uscita del riferimento al caso.
  A5  MUTAZIONI (il collaudo deve diventare rosso):
      a) fuga: pieghe con la prova dentro l'addestramento → A2 deve fallire (la sonda impara a memoria);
      b) strato sfasato di uno nello studente → A1 deve fallire (localizzazione sbagliata);
      c) studente = maestro al caricamento → A1 deve fallire (nessuna perdita vista).
B — estrazione VERA col modello minuscolo qwen35moe (e2e_dump_logit, CPU, strumento non modificato)
  B1  6 prompt MMLU validation tokenizzati con llama.cpp; estrai_stati a lotti da 4 (2 lotti): forme, finitezza, meta
  B2  testa in float64: rmsnorm(ultima riga di l_out-3) · W_out → logit delle 4 lettere, logsumexp e argmax salvati
      (la riga e' davvero l'ULTIMO token); e la riga SBAGLIATA (primo token) deve dare un errore >= 10x
  B3  lotti da 4 == lotto unico da 6, al bit
  B4  MUTAZIONE: estrattore che prende la prima riga → B2 deve fallire

Uso: XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus systemd-run --user --scope \\
       -p CPUQuota=200% nice -n 19 env OMP_NUM_THREADS=1 HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= \\
       python3 test_sonda_strati.py [--solo A|B]
Cartella: $SONDA_PROVA_DIR (default @@ROOT:depot@@/sonda_strati/prova), mai /tmp.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

QUI = Path(__file__).resolve().parent
THOR_V2 = QUI.parents[1]
sys.path.insert(0, str(QUI))
sys.path.insert(0, str(THOR_V2))
sys.path.insert(0, "@@ROOT:models@@/build/llamacpp-tq1-k1024/gguf-py")
import sonda  # noqa: E402

PROVA = Path(os.getenv("SONDA_PROVA_DIR", "@@ROOT:depot@@/sonda_strati/prova"))
MINUSCOLO = Path("@@ROOT:lavoro@@/e2e/dump_logit/minuscolo-qwen35moe.gguf")
MINUSCOLO_SHA = "01bcfceea544856e940c00ab04aaa0e8d48b3aaa8ec86514ce807b0469af1f33"
ESITI: list = []


def prova(nome: str, ok: bool, **dettagli) -> None:
    ESITI.append({"prova": nome, "ok": bool(ok), **{k: v for k, v in dettagli.items()}})
    print(f"  {'OK ' if ok else 'NO '} {nome}" + (f"  {dettagli}" if dettagli else ""), flush=True)
    if not ok:
        raise AssertionError(nome)


def mutazione_vista(nome: str, controllo) -> None:
    """Il controllo DEVE fallire sotto la mutazione."""
    n0 = len(ESITI)
    try:
        controllo()
    except AssertionError as e:
        del ESITI[n0:]
        ESITI.append({"prova": f"mutazione vista: {nome}", "ok": True, "rosso_su": str(e)})
        print(f"  OK  mutazione vista: {nome} (rosso su «{e}»)", flush=True)
        return
    del ESITI[n0:]
    ESITI.append({"prova": f"mutazione NON vista: {nome}", "ok": False})
    raise AssertionError(f"la mutazione «{nome}» NON fa fallire il collaudo")


# ═════════════════════════════ A — stati finti ═════════════════════════════
def fnv(b: bytes) -> str:
    return sonda_fnv(b)


def sonda_fnv(b: bytes) -> str:
    h = 1469598103934665603
    for x in b:
        h ^= x
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def genera_finti(cart: Path, N=800, L=12, d=64, K=7, inizio=4, pieno=8, seme=7, mescola=False,
                 identici=False) -> dict:
    """Maestro: segnale di classe a(l) = 0 fino a `inizio`, poi rampa fino a `pieno`. Studente: identico al maestro
    (stesso rumore Z) prima di K; da K segnale 0,35x e rumore proprio 0,6. Lettere: dall'ultimo strato."""
    rng = np.random.default_rng(seme)
    shutil.rmtree(cart, ignore_errors=True)
    (cart / "domande" / "ids").mkdir(parents=True)
    y = rng.integers(0, 4, N)
    materie = np.asarray([f"materia_{i % 9}" for i in range(N)])
    a_m = np.clip((np.arange(L) - inizio + 1) / (pieno - inizio + 1), 0, 1) * 1.6
    U = rng.normal(size=(L, 4, d))
    U /= np.linalg.norm(U, axis=2, keepdims=True)
    Z = rng.normal(size=(N, L, d))
    Z[:, :, 0] *= 40.0                                      # una «attivazione massiccia» senza informazione
    Xm = Z + a_m[None, :, None] * U[np.arange(L)[None, :], y[:, None]] * np.sqrt(d) / 4
    if identici:
        Xs = Xm.copy()
    else:
        a_s = np.where(np.arange(L) < K, a_m, 0.35 * a_m)
        rumore = np.where(np.arange(L) < K, 0.0, 0.6)
        Xs = (Z + a_s[None, :, None] * U[np.arange(L)[None, :], y[:, None]] * np.sqrt(d) / 4
              + rumore[None, :, None] * rng.normal(size=(N, L, d)))
    y_etich = rng.permutation(y) if mescola else y
    dom = []
    for k in range(N):
        ids = np.asarray([k, 7, 11], dtype=np.int32)
        (cart / "domande" / "ids" / f"q_{k:05d}.i32").write_bytes(ids.tobytes())
        dom.append({"k": k, "split": "validation" if k % 5 else "dev", "indice": k, "materia": str(materie[k]),
                    "risposta": int(y_etich[k]), "n_token": 3, "vista_forgia": bool(k % 7 == 0),
                    "fnv1a64_ids": sonda_fnv(ids.tobytes()), "file": f"ids/q_{k:05d}.i32"})
    (cart / "domande" / "domande.json").write_text(json.dumps({"creato": "finto", "domande": dom,
                                                               "id_lettere": {"A": 1, "B": 2, "C": 3, "D": 4}}))
    sha = sonda.sha256_file(cart / "domande" / "domande.json")

    def lettere(X, forza):
        W = U[-1] * forza
        W[:, 0] = 0.0                                       # l'uscita non legge la dimensione massiccia
        return (X[:, -1, :] @ W.T / np.sqrt(d)).astype(np.float32)
    for nome, X, forza in (("maestro", Xm, 3.0), ("studente", Xs, 3.0)):
        c = cart / "stati" / nome
        c.mkdir(parents=True)
        np.save(c / "stati.npy", X.astype(np.float32))
        np.save(c / "lettere.npy", lettere(X, forza))
        (c / "meta.json").write_text(json.dumps({"modello": {"percorso": f"finto-{nome}"},
                                                 "domande": {"fnv1a64_ids": [x["fnv1a64_ids"] for x in dom],
                                                             "sha256_domande_json": sha}}))
    return {"K": K, "inizio": inizio, "pieno": pieno, "L": L}


def esegui_sonda(cart: Path, uscita: Path, confine: int, extra=()) -> dict:
    return sonda.main(["--domande", str(cart / "domande"), "--modello", f"maestro={cart / 'stati' / 'maestro'}",
                       "--modello", f"studente={cart / 'stati' / 'studente'}", "--uscita", str(uscita),
                       "--pieghe", "5", "--ripetizioni", "2", "--interne", "3", "--C", "1e-3,1e-2,0.1,1,10",
                       "--bootstrap", "400", "--jobs", "1", "--confine", str(confine), *extra])


def controlla_A1(e: dict, g: dict) -> None:
    t = e["tutte"]
    m, s, c = t["modelli"]["maestro"]["strati"], t["modelli"]["studente"]["strati"], t["confronti"]["studente"]
    K, inizio = g["K"], g["inizio"]
    caso = t["maggioranza"]
    muti = [x["acc"] for x in m[:inizio]]
    prova("A1 strati senza segnale: sonda del maestro al caso (entro 7 punti dalla classe piu' frequente)",
          all(abs(v - caso) < 0.07 for v in muti), acc=[round(v, 3) for v in muti], caso=round(caso, 3))
    prova("A1 maestro alto negli strati pieni (> 0,85)", all(x["acc"] > 0.85 for x in m[g["pieno"]:]),
          acc=[round(x["acc"], 3) for x in m[g["pieno"]:]])
    prova(f"A1 primo strato con Δll significativo == {K}", c["primo_strato_significativo"] == K,
          trovato=c["primo_strato_significativo"])
    prova(f"A1 significativo da {K} in poi", c["da_qui_sempre"] == K, trovato=c["da_qui_sempre"])
    prova("A1 nessun falso positivo prima di K (Δll ~0, |z| < 3)",
          all(abs(r["d_ll_z"]) < 3 for r in c["strati"][:K]), z=[round(r["d_ll_z"], 2) for r in c["strati"][:K]])
    prova("A1 sonda del maestro sugli stati dello studente: = maestro prima di K, piu' bassa dopo",
          all(abs(r["acc_trasf"] - x["acc"]) < 0.02 for r, x in zip(c["strati"][:K], m[:K]))
          and all(r["acc_trasf"] < x["acc"] - 0.1 for r, x in zip(c["strati"][g["pieno"]:], m[g["pieno"]:])))
    prova("A1 CKA (grezzo e standardizzato) ~1 ed errore 0 prima di K; CKA standardizzato piu' basso dopo",
          all(r["cka"] > 0.999 and r["cka_std"] > 0.999 and r["err_rel"] < 1e-6 for r in c["strati"][:K])
          and all(r["cka_std"] < 0.99 for r in c["strati"][K:]),
          cka_std=[round(r["cka_std"], 3) for r in c["strati"]])
    prova("A1 attivazione massiccia: il CKA grezzo resta ~1 anche dopo K (per questo c'e' cka_std)",
          all(r["cka"] > 0.99 for r in c["strati"][K:]), cka=[round(r["cka"], 4) for r in c["strati"][K:]])
    f = c["frazione_perdita_al_confine"]
    prova("A1 frazione della perdita al confine (K−2) ~0", f is not None and abs(f["valore"]) < 0.05,
          frazione=f)
    prova("A1 uscita: il maestro batte lo studente sulle 4 lettere", c["uscita"]["d_acc"] > 0.05,
          d_acc=c["uscita"]["d_acc"])


def controlla_A2(e: dict) -> None:
    t = e["tutte"]
    c = t["confronti"]["studente"]
    caso = t["maggioranza"]
    for n in ("maestro", "studente"):
        acc = [x["acc"] for x in t["modelli"][n]["strati"]]
        prova(f"A2 etichette mescolate: {n} al caso in ogni strato (entro 7 punti)",
              all(abs(v - caso) < 0.07 for v in acc), acc_max=round(max(acc), 3), caso=round(caso, 3))
    prova("A2 nessuno strato significativo", c["primo_strato_significativo"] is None,
          trovato=c["primo_strato_significativo"])


def controlla_A3(e: dict) -> None:
    c = e["tutte"]["confronti"]["studente"]
    prova("A3 identici: Δll = 0 e Δacc = 0 ovunque",
          all(abs(r["d_ll"]) < 1e-12 and r["d_acc"] == 0 for r in c["strati"]))
    prova("A3 identici: nessuno significativo, CKA 1, errore 0",
          c["primo_strato_significativo"] is None and all(abs(r["cka"] - 1) < 1e-9 and r["err_rel"] == 0
                                                          for r in c["strati"]))


def rifiuto(nome: str, fn, atteso: str) -> None:
    """Deve sollevare, e per il motivo GIUSTO (non per un altro controllo scattato prima)."""
    try:
        fn()
    except (RuntimeError, ValueError) as e:
        prova(f"A4 rifiuto: {nome}", atteso in str(e), messaggio=str(e)[:140], atteso=atteso)
        return
    prova(f"A4 rifiuto: {nome}", False)


def blocco_A() -> None:
    print("A — stati finti", flush=True)
    base = PROVA / "finti"
    g = genera_finti(base / "a1")
    e1 = esegui_sonda(base / "a1", base / "a1" / "ris", confine=g["K"] - 2)
    controlla_A1(e1, g)
    prova("A1 grafico e tabella scritti", (base / "a1" / "ris" / "sonda_strati.svg").stat().st_size > 5000
          and "| strato |" in (base / "a1" / "ris" / "sonda_strati.md").read_text())

    genera_finti(base / "a2", mescola=True)
    vecchio = sonda.MIN_ACC_USCITA
    sonda.MIN_ACC_USCITA = 0.0                       # con le etichette mescolate anche l'uscita e' al caso
    try:
        e2 = esegui_sonda(base / "a2", base / "a2" / "ris", confine=5, extra=["--senza-grafico"])
        controlla_A2(e2)
    finally:
        sonda.MIN_ACC_USCITA = vecchio

    genera_finti(base / "a3", identici=True)
    e3 = esegui_sonda(base / "a3", base / "a3" / "ris", confine=5, extra=["--senza-grafico"])
    controlla_A3(e3)

    # A4 rifiuti
    c = base / "a4"
    genera_finti(c, N=120, L=3, d=8)
    mods = [("maestro", c / "stati" / "maestro"), ("studente", c / "stati" / "studente")]
    m = json.loads((c / "stati" / "studente" / "meta.json").read_text())
    m2 = json.loads(json.dumps(m))
    m2["domande"]["fnv1a64_ids"][5] = "0" * 16
    (c / "stati" / "studente" / "meta.json").write_text(json.dumps(m2))
    rifiuto("id diversi fra modello e domande", lambda: sonda.carica(c / "domande", mods), "id fnv1a64")
    m2 = json.loads(json.dumps(m))
    m2["domande"]["sha256_domande_json"] = "f" * 64
    (c / "stati" / "studente" / "meta.json").write_text(json.dumps(m2))
    rifiuto("domande.json diverso", lambda: sonda.carica(c / "domande", mods), "domande.json diverso")
    (c / "stati" / "studente" / "meta.json").write_text(json.dumps(m))
    S = np.load(c / "stati" / "studente" / "stati.npy")
    np.save(c / "stati" / "studente" / "stati.npy", S[:, :2])
    rifiuto("forme diverse", lambda: sonda.carica(c / "domande", mods), "forma")
    S2 = S.copy()
    S2[3, 1, 2] = np.nan
    np.save(c / "stati" / "studente" / "stati.npy", S2)
    vecchio = sonda.MIN_ACC_USCITA
    sonda.MIN_ACC_USCITA = 0.0                       # qui si prova SOLO il controllo di finitezza
    try:
        rifiuto("stati non finiti", lambda: sonda.main(["--domande", str(c / "domande"), "--modello",
                                                         f"maestro={mods[0][1]}", "--modello",
                                                         f"studente={mods[1][1]}", "--uscita", str(c / "ris"),
                                                         "--jobs", "1", "--pieghe", "3", "--ripetizioni", "1",
                                                         "--C", "1", "--senza-grafico"]), "non finiti")
    finally:
        sonda.MIN_ACC_USCITA = vecchio
    np.save(c / "stati" / "studente" / "stati.npy", S)
    genera_finti(base / "a4b", N=200, L=3, d=8, mescola=True)
    rifiuto("uscita del riferimento al caso (id delle lettere sbagliati)",
            lambda: esegui_sonda(base / "a4b", base / "a4b" / "ris", confine=1, extra=["--senza-grafico"]),
            "accuratezza d'uscita del riferimento")

    # A5 mutazioni
    print("A5 — mutazioni", flush=True)
    fai_vero, carica_vero = sonda.fai_pieghe, sonda.carica

    def fuga(y, k, r, seme):
        tutte = np.arange(len(y))
        return [[(tutte, te) for _, te in pieghe] for pieghe in fai_vero(y, k, r, seme)]

    def sfasato(domande_dir, modelli):
        d = carica_vero(domande_dir, modelli)
        S = np.asarray(d["stati"]["studente"])
        d["stati"]["studente"] = np.concatenate([S[:, 1:], S[:, -1:]], axis=1)
        return d

    def come_maestro(domande_dir, modelli):
        d = carica_vero(domande_dir, modelli)
        d["stati"]["studente"] = d["stati"]["maestro"]
        d["lettere"]["studente"] = d["lettere"]["maestro"]
        return d

    def sotto(nome, attr, falso, controllo):
        setattr(sonda, attr, falso)
        try:
            mutazione_vista(nome, controllo)
        finally:
            sonda.fai_pieghe, sonda.carica = fai_vero, carica_vero

    def a2_di_nuovo():
        vecchio_ = sonda.MIN_ACC_USCITA
        sonda.MIN_ACC_USCITA = 0.0
        try:
            controlla_A2(esegui_sonda(base / "a2", base / "a2" / "ris_mut", confine=5, extra=["--senza-grafico"]))
        finally:
            sonda.MIN_ACC_USCITA = vecchio_
    sotto("fuga (prova dentro l'addestramento) -> A2", "fai_pieghe", fuga, a2_di_nuovo)
    sotto("strato sfasato di uno nello studente -> A1", "carica", sfasato,
          lambda: controlla_A1(esegui_sonda(base / "a1", base / "a1" / "ris_mut", confine=g["K"] - 2,
                                            extra=["--senza-grafico"]), g))
    sotto("studente = maestro -> A1", "carica", come_maestro,
          lambda: controlla_A1(esegui_sonda(base / "a1", base / "a1" / "ris_mut", confine=g["K"] - 2,
                                            extra=["--senza-grafico"]), g))


# ═════════════════════════════ B — modello minuscolo, estrazione vera ═════════════════════════════
def blocco_B() -> None:
    print("B — estrazione vera, modello minuscolo qwen35moe (CPU)", flush=True)
    import estrai_stati
    import prepara_domande as PD
    import pyarrow.parquet as pq
    from gguf import GGUFReader
    from gguf.quants import dequantize
    sha = hashlib.sha256(MINUSCOLO.read_bytes()).hexdigest()
    prova("B0 modello minuscolo presente e intatto (sha256 del suo json)", sha == MINUSCOLO_SHA, sha=sha[:16])
    base = PROVA / "minuscolo"
    shutil.rmtree(base, ignore_errors=True)
    (base / "domande" / "ids").mkdir(parents=True)
    val = pq.read_table(PD.PC.FILE_VAL).to_pylist()
    scelte = [0, 1, 300, 700, 1100, 1500]           # materie e lunghezze diverse
    PD.VOCAB_GGUF = MINUSCOLO
    ids_l, dom = [], []
    for k, i in enumerate(scelte):
        p = PD.PC.prompt_banco(val[i])
        ids = PD.tokenizza_llama(p, base / "tok" / str(k))
        ids_l.append(ids)
        a = np.asarray(ids, dtype=np.int32)
        (base / "domande" / "ids" / f"q_{k:05d}.i32").write_bytes(a.tobytes())
        dom.append({"k": k, "split": "validation", "indice": i, "materia": val[i]["subject"],
                    "risposta": int(val[i]["answer"]), "n_token": len(ids), "vista_forgia": False,
                    "fnv1a64_ids": PD.fnv1a64(a.tobytes()), "file": f"ids/q_{k:05d}.i32"})
    lett = {}
    for L in "ABCD":
        e = PD.tokenizza_llama(PD.PC.prompt_banco(val[0]) + " " + L, base / "tok" / L)
        prova(f"B1 lettera {L}: un token dopo il prompt", e[:-1] == ids_l[0])
        lett[L] = e[-1]
    (base / "domande" / "domande.json").write_text(json.dumps({"creato": "collaudo", "domande": dom,
                                                               "id_lettere": lett}))
    comune = ["--modello", str(MINUSCOLO), "--domande", str(base / "domande"), "--thread", "2",
              "--strati-attesi", "4", "--senza-sha256"]
    estrai_stati.main(comune + ["--nome", "l4", "--uscita", str(base / "l4"), "--lotto", "4"])
    S = np.load(base / "l4" / "stati.npy")
    lt = np.load(base / "l4" / "lettere.npy")
    meta = json.loads((base / "l4" / "meta.json").read_text())
    prova("B1 forme (6, 4, 256) e (6, 4), finite; 2 lotti; grezzo cancellato",
          S.shape == (6, 4, 256) and lt.shape == (6, 4) and np.isfinite(S).all() and len(meta["tempi"]) == 2
          and not (base / "l4" / "grezzo").exists(), forma=S.shape, lotti=len(meta["tempi"]))

    r = GGUFReader(str(MINUSCOLO))
    eps = float(r.fields["qwen35moe.attention.layer_norm_rms_epsilon"].contents())

    def peso(nome):
        t = next(t for t in r.tensors if t.name == nome)
        if t.tensor_type.name in ("F32", "F16"):
            return np.array(t.data, dtype=np.float64)
        return dequantize(np.asarray(t.data), t.tensor_type).astype(np.float64)
    w_norm, W = peso("output_norm.weight"), peso("output.weight")
    tipo_testa = next(t for t in r.tensors if t.name == "output.weight").tensor_type.name
    from e2e_test_dump_logit import attivazione_come_cpu   # emulazione collaudata (T6 di e2e_test_dump_logit)

    def testa(h, come_cpu: bool = True):
        """norma finale + testa in float64; con come_cpu l'attivazione e' quantizzata come la moltiplica la CPU."""
        h = np.asarray(h, np.float64)
        rn = h / np.sqrt((h * h).mean(-1, keepdims=True) + eps) * w_norm
        if come_cpu and tipo_testa not in ("F32", "F16"):
            rn = attivazione_come_cpu(rn.astype(np.float32)[None, :], tipo_testa)[0]
        return rn @ W.T

    def controlla_B2(cart: Path, righe_prima: bool = False) -> None:
        S_ = np.load(cart / "stati.npy")
        lt_ = np.load(cart / "lettere.npy").astype(np.float64)
        lse_ = np.load(cart / "lse.npy")
        am_ = np.load(cart / "argmax.npy")
        idl = [lett[x] for x in "ABCD"]
        err_l, err_lse, err_f, am_ok = [], [], [], 0
        for j in range(S_.shape[0]):
            lg = testa(S_[j, -1])
            scala = lg.std()
            err_l.append(float(np.abs(lg[idl] - lt_[j]).max() / scala))
            err_f.append(float(np.abs(testa(S_[j, -1], come_cpu=False)[idl] - lt_[j]).max() / scala))
            m = lg.max()
            err_lse.append(float(abs(m + np.log(np.exp(lg - m).sum()) - lse_[j]) / scala))
            am_ok += int(lg.argmax() == am_[j])
        prova(f"B2 testa float64 (attivazione quantizzata come la CPU, {tipo_testa}) sull'ultima riga di l_out-3 == "
              "logit delle lettere, logsumexp e argmax salvati (entro 1e-3 della scala dei logit)",
              max(err_l) < 1e-3 and max(err_lse) < 1e-3 and am_ok == S_.shape[0],
              err_lettere=max(err_l), err_lse=max(err_lse), argmax_uguali=am_ok,
              err_senza_quantizzare_attivazione=max(err_f))
    controlla_B2(base / "l4")
    # la riga SBAGLIATA deve dare un errore molto piu' grande: stati del primo token, con il grezzo tenuto
    estrai_stati.main(comune + ["--nome", "g", "--uscita", str(base / "g"), "--lotto", "6", "--tieni-grezzo"])
    from e2e_dump_logit_leggi import Uscita
    u = Uscita(base / "g" / "grezzo" / "l_0000")
    lt_g = np.load(base / "g" / "lettere.npy").astype(np.float64)
    e_giusta, e_prima = [], []
    for j in range(6):
        h = np.asarray(u.tensore(j, "l_out-3"))
        for riga, dest in ((-1, e_giusta), (0, e_prima)):
            lg = testa(h[riga])
            dest.append(float(np.abs(lg[[lett[x] for x in "ABCD"]] - lt_g[j]).max() / lg.std()))
    prova("B2 contrasto: la prima riga (token sbagliato) sbaglia >= 10x", min(e_prima) >= 10 * max(e_giusta),
          err_ultima=max(e_giusta), err_prima_min=min(e_prima))
    S6 = np.load(base / "g" / "stati.npy")
    prova("B3 lotti da 4 == lotto unico da 6, al bit (stati e lettere)",
          S6.tobytes() == S.tobytes() and np.load(base / "g" / "lettere.npy").tobytes() == lt.tobytes())
    # B4 mutazione: l'estrattore prende la prima riga
    vero = estrai_stati.riga_ultimo_token
    estrai_stati.riga_ultimo_token = lambda info, n_token: 0
    try:
        estrai_stati.main(comune + ["--nome", "mut", "--uscita", str(base / "mut"), "--lotto", "6"])
        mutazione_vista("estrattore sulla prima riga -> B2", lambda: controlla_B2(base / "mut"))
    finally:
        estrai_stati.riga_ultimo_token = vero


def main() -> None:
    solo = sys.argv[sys.argv.index("--solo") + 1] if "--solo" in sys.argv else "AB"
    PROVA.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rotto = None
    try:
        if "A" in solo:
            blocco_A()
        if "B" in solo:
            blocco_B()
    except Exception as e:  # noqa: BLE001 — il collaudo registra e poi esce con codice 1
        rotto = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    ok = rotto is None and all(x["ok"] for x in ESITI)
    (PROVA / f"esito_collaudo_{solo}.json").write_text(json.dumps(
        {"ok": ok, "rotto": rotto, "secondi": round(time.time() - t0, 1), "prove": ESITI}, ensure_ascii=False,
        indent=1, default=str))
    print(f"{'COLLAUDO VERDE' if ok else 'COLLAUDO ROSSO'}: {sum(x['ok'] for x in ESITI)}/{len(ESITI)} prove, "
          f"{time.time() - t0:.0f} s" + (f" — {rotto}" if rotto else ""), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

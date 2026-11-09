#!/usr/bin/env python3
"""Collaudo della patch «export incrementale» (leva F) — CPU, scala giocattolo, GGUF confrontati al bit.

Nessun file di thor_v2 viene scritto: i sorgenti si copiano in <lavoro>/copie/{base,F,...}/thor_v2
(accanto il collegamento kernel-rotazione), la patch si applica alle copie. GGUF base giocattolo
(architettura qwen35moe, 4 strati, E=5 esperti, hidden 1024, intermedio 256: gate/up TQ1_0 nel
base -> TQ1_0R_1024; down Q4_K sugli strati pari e TQ1_0 sui dispari; strato 3 SENZA down) e
codici casuali col contratto vero (D float32, Q int8 in {-1,0,1}). Ogni corsa in un processo a
parte, solo CPU. Il riferimento di ogni scenario e' l'export CLASSICO (copia NON patchata) sugli
STESSI codici su disco, dopo lo scenario: sha256 del GGUF identico, oppure lo stesso errore.

    env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 19 \\
      @@ROOT:venv@@/bin/python3 strumenti/collaudo_export_incrementale.py --lavoro <scratch>/export

Scenari: spento; solo a blocchi; incrementale (come forgia_globale: scrittore in un thread, due
strati per finestra); interruzione DENTRO la scrittura di un pacchetto + ripresa che ricalcola
lo strato con codici diversi; processo ucciso subito dopo i codici (pacchetto mai fatto) +
ripresa; codici riscritti dopo il pacchetto; byte del pacchetto alterati; impacchettatore
cambiato; pacchetto del primo strato senza verifiche; packer rotto all'export (deve fallire come
il classico); tipo base sbagliato (stesso errore del classico); disco «pieno» (margine enorme);
import senza cartella dei pacchetti (rifiuto). Piu': la patch con le altre 7 nei due ordini,
doppia applicazione e aggancio mancante (rifiuti senza scritture).
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

QUI = Path(__file__).resolve()
THOR_V2 = QUI.parent.parent
PY = "@@ROOT:venv@@/bin/python3"
ENV_CPU = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
           "OPENBLAS_NUM_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1"}
E, H, I, NL = 5, 1024, 256, 4
CON_DOWN = {0: True, 1: True, 2: True, 3: False}
ESITI: list = []


def ok(nome: str, cond, dettaglio: str = "") -> bool:
    ESITI.append((nome, bool(cond)))
    print(f"  {'✅' if cond else '⛔'} {nome}" + (f" — {dettaglio}" if dettaglio else ""), flush=True)
    return bool(cond)


# ═══════════════════════════ figlio: una corsa ═══════════════════════════
def codici(L: int, seme: int, con_down: bool) -> dict:
    import numpy as np
    rng = np.random.default_rng(seme * 100 + L)
    fam = {}
    for nome in ("ffn_gate_exps", "ffn_up_exps"):
        fam[nome] = (rng.uniform(1e-3, 5e-2, size=(E, I, H // 128)).astype(np.float32),
                     rng.integers(-1, 2, size=(E, I, H), dtype=np.int8))
    if con_down:
        fam["ffn_down_exps"] = (rng.uniform(1e-3, 5e-2, size=(E, H, I // 256)).astype(np.float32),
                                rng.integers(-1, 2, size=(E, H, I), dtype=np.int8))
    return fam


def figlio(cartella: Path, passi: list, uscita: Path) -> int:
    os.chdir(cartella)
    sys.path.insert(0, str(cartella))
    ris: dict = {"eventi": []}
    try:
        import numpy as np  # noqa: F401
        import valida_finestra_8_12 as V
    except Exception as e:  # noqa: BLE001
        ris["errore_import"] = f"{type(e).__name__}: {e}"
        uscita.write_text(json.dumps(ris))
        return 0
    for passo in passi:
        tipo = passo[0]
        if tipo == "scrivi":                                   # (scrivi, L, seme) — come forgia_scorrevole
            _, L, seme = passo
            V.scrivi_codici_su_disco({L: codici(L, seme, CON_DOWN[L])})
        elif tipo == "scrivi_come_forgia_globale":              # finestre da 2 strati, scrittore in un thread
            _, seme = passo
            from concurrent.futures import ThreadPoolExecutor
            scrittore = ThreadPoolExecutor(max_workers=1)
            scritture = [scrittore.submit(V.scrivi_codici_su_disco, {L: codici(L, seme, CON_DOWN[L])
                                                                     for L in (l0, l0 + 1)})
                         for l0 in range(0, NL, 2)]
            for f in scritture:
                f.result()
            scrittore.shutdown(wait=True)
        elif tipo == "attendi":
            V._attendi_impacchettamenti()
        elif tipo == "muori_subito":                           # come un SIGKILL: niente atexit, niente attese
            sys.stdout.flush()
            os._exit(9)
        elif tipo == "muori_scrivendo":                        # (muori_scrivendo, nome file del pacchetto)
            _, bersaglio = passo
            orig = V._salva_npy_atomico

            def salva_e_muori(p, a, _orig=orig, _b=bersaglio):
                if p.name == _b:
                    with open(p.with_name(p.name + ".tmp"), "wb") as f:
                        f.write(b"\x93NUMPY" + b"\0" * 1000)       # file a meta'
                    sys.stdout.flush()
                    os._exit(9)
                return _orig(p, a)
            V._salva_npy_atomico = salva_e_muori
        elif tipo == "spezza_packer":                          # un bit sbagliato nella scala di ogni blocco
            import esporta_schur_esperti as ESE
            orig_pack = ESE.pack_da_d_q

            def pack_rotto(d, q, _o=orig_pack):
                out = _o(d, q).copy()
                out[:, 208] ^= 1
                return out
            ESE.pack_da_d_q = pack_rotto
        elif tipo == "esporta":
            try:
                p = V.esporta_gguf_finestra(layers=list(range(NL)))
                ris["gguf"] = str(p)
            except BaseException as e:  # noqa: BLE001
                ris["errore_export"] = f"{type(e).__name__}: {e}"
        ris["eventi"].append(tipo)
    uscita.write_text(json.dumps(ris))
    return 0


# ═══════════════════════════ padre ═══════════════════════════
def gguf_base(percorso: Path, gate0_q4k: bool = False) -> None:
    sys.path.insert(0, os.getenv("GGUF_PY_DIR", "@@ROOT:models@@/build/llamacpp-tq1-k1024/gguf-py"))
    import numpy as np
    from gguf import GGUFWriter, GGMLQuantizationType as T
    rng = np.random.default_rng(7)
    w = GGUFWriter(str(percorso), arch="qwen35moe")
    w.add_block_count(NL)
    w.add_string("general.name", "giocattolo collaudo export incrementale")
    w.add_array("tokenizer.ggml.tokens", ["a", "b", "c"])
    tens = [("token_embd.weight", rng.standard_normal((8, H)).astype(np.float32), None)]
    for L in range(NL):
        tens.append((f"blk.{L}.attn_norm.weight", rng.standard_normal(H).astype(np.float32), None))
        for fam in ("ffn_gate_exps", "ffn_up_exps"):
            tipo = T.Q4_K if (gate0_q4k and L == 0 and fam == "ffn_gate_exps") else T.TQ1_0
            nb = (H // 256) * (144 if tipo == T.Q4_K else 54)
            tens.append((f"blk.{L}.{fam}.weight", rng.integers(0, 256, size=(E, I, nb), dtype=np.uint8), tipo))
        tipo_d = T.Q4_K if L % 2 == 0 else T.TQ1_0
        nb = (I // 256) * (144 if tipo_d == T.Q4_K else 54)
        tens.append((f"blk.{L}.ffn_down_exps.weight", rng.integers(0, 256, size=(E, H, nb), dtype=np.uint8), tipo_d))
    for nome, a, tipo in tens:
        if tipo is None:
            w.add_tensor(nome, a)
        else:
            w.add_tensor(nome, a, raw_dtype=tipo)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Banco:
    def __init__(self, lavoro: Path):
        self.lavoro = lavoro
        self.base = lavoro / "base_giocattolo.gguf"
        self.base_q4k = lavoro / "base_giocattolo_gate0_q4k.gguf"
        self.n = 0

    def copia(self, nome: str, patch: "list[str]") -> Path:
        radice = self.lavoro / "copie" / nome
        if radice.exists():
            shutil.rmtree(radice)
        d = radice / "thor_v2"
        (d / "strumenti").mkdir(parents=True)
        for pat in ("*.py", "*.json", "*.npz"):
            for f in THOR_V2.glob(pat):
                shutil.copy2(f, d / f.name)
        for f in (THOR_V2 / "strumenti").glob("*.py"):
            shutil.copy2(f, d / "strumenti" / f.name)
        (radice / "kernel-rotazione").symlink_to(THOR_V2.parent / "kernel-rotazione")
        for p in patch:
            r = subprocess.run([PY, str(d / "strumenti" / f"patch_tempo_{p}.py"), "--cartella", str(d)],
                               capture_output=True, text=True, env={**os.environ, **ENV_CPU})
            if not ok(f"copia {nome}: patch {p} applicata", r.returncode == 0, (r.stdout + r.stderr).strip()[-150:]):
                raise SystemExit(1)
        return d

    def corsa(self, cartella: Path, passi: list, amb: dict, etichetta: str) -> dict:
        self.n += 1
        uscita = self.lavoro / "corse" / f"{self.n:02d}_{etichetta}"
        if uscita.exists():
            shutil.rmtree(uscita)
        uscita.mkdir(parents=True)
        env = {**os.environ, **ENV_CPU, **amb}
        with open(uscita / "log.txt", "w") as flog:
            r = subprocess.run([PY, str(QUI), "--figlio", "--cartella", str(cartella), "--passi", json.dumps(passi),
                                "--uscita", str(uscita / "r.json")], stdout=flog, stderr=subprocess.STDOUT, env=env)
        log = (uscita / "log.txt").read_text()
        ris = json.loads((uscita / "r.json").read_text()) if (uscita / "r.json").exists() else {}
        ris.update({"rc": r.returncode, "log": log, "dir": str(uscita)})
        return ris

    def amb(self, nome: str, out: Path, gguf: Path, **extra) -> dict:
        a = {"THOR_SCHUR_BASE": str(extra.pop("base", self.base)), "THOR_VALIDA_OUT": str(out),
             "THOR_VALIDA_GGUF_OUT": str(gguf), "THOR_ID_CORSA": f"collaudo-F-{nome}", "THOR_FORGIA_W": "2",
             "THOR_FORGIA_DOWN_TERNARIO": "1", "THOR_EXPORT_PACCHETTI_BLOCCO": "2",
             # pacchetti giocattolo da pochi MiB su tmpfs: margine di 1 GiB (il default 8 ha rifiutato,
             # giustamente, con 8,0 GiB liberi in /tmp alle 05:18)
             "THOR_EXPORT_PACCHETTI_MARGINE_GIB": "1"}
        a.update({k: str(v) for k, v in extra.items()})
        return a


def main_padre(lavoro: Path) -> int:
    t0 = time.time()
    lavoro.mkdir(parents=True, exist_ok=True)
    b = Banco(lavoro)
    vero_prima = {f.name: f.read_bytes() for f in THOR_V2.glob("*.py")}
    print("══ copie, patch, GGUF base giocattolo ══", flush=True)
    base = b.copia("base", [])
    F = b.copia("F", ["export_incrementale"])
    for f in (b.base, b.base_q4k):
        if f.exists():
            f.unlink()
    gguf_base(b.base)
    gguf_base(b.base_q4k, gate0_q4k=True)

    # rifiuti e convivenza con le altre 7 patch
    r = subprocess.run([PY, str(F / "strumenti" / "patch_tempo_export_incrementale.py"), "--cartella", str(F)],
                       capture_output=True, text=True, env={**os.environ, **ENV_CPU})
    ok("patch applicata due volte: RIFIUTATA", r.returncode != 0 and "gia' applicata" in (r.stdout + r.stderr))
    rotta = b.copia("_rotta", [])
    fs = rotta / "forgia_scorrevole.py"
    fs.write_text(fs.read_text().replace("class InterruzioneSimulata(RuntimeError):", "class InterruzioneSimulata(Exception):"))
    valida_prima = (rotta / "valida_finestra_8_12.py").read_bytes()
    r = subprocess.run([PY, str(rotta / "strumenti" / "patch_tempo_export_incrementale.py"), "--cartella", str(rotta)],
                       capture_output=True, text=True, env={**os.environ, **ENV_CPU})
    ok("aggancio mancante nella forgia: RIFIUTATA e valida NON scritta",
       r.returncode != 0 and "aggancio trovato 0 volte" in (r.stdout + r.stderr)
       and (rotta / "valida_finestra_8_12.py").read_bytes() == valida_prima)
    sette = ["prepara_asincrona", "riuso_teacher", "prefetch_strato", "cache_attenzione", "attenzione_confine",
             "congelo_senza_moe", "beta_costante"]
    tutte_prima = b.copia("tutte_F_prima", ["export_incrementale"] + sette)
    tutte_dopo = b.copia("tutte_F_dopo", list(reversed(sette[:4])) + ["attenzione_confine", "congelo_senza_moe",
                                                                     "beta_costante", "export_incrementale"])
    import importlib.util
    bersagli_patch = {}
    for m in ["export_incrementale"] + sette:
        spec = importlib.util.spec_from_file_location(f"_p_{m}", QUI.parent / f"patch_tempo_{m}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for n in mod.PIANO:
            bersagli_patch.setdefault(n, []).append(mod.MARCA)
    for n, marche_n in sorted(bersagli_patch.items()):
        ok(f"con le altre 7 patch (F prima e F dopo): {n} porta le sue {len(marche_n)} marche in entrambe le copie",
           all(mm in (c / n).read_text() for c in (tutte_prima, tutte_dopo) for mm in marche_n))

    def riferimento(out: Path, etichetta: str, **extra) -> dict:
        g = lavoro / "gguf" / f"rif_{etichetta}.gguf"
        return b.corsa(base, [["esporta"]], b.amb(etichetta, out, g, **extra), f"rif_{etichetta}")

    def controlla(nome: str, prova: dict, rif: dict, pronti: "str | None" = None, testo: "list | None" = None):
        if "gguf" in rif and "gguf" in prova:
            s_p, s_r = sha(prova["gguf"]), sha(rif["gguf"])
            ok(f"{nome}: GGUF identico al bit al classico sugli stessi codici", s_p == s_r, s_p[:16])
        else:
            ok(f"{nome}: GGUF identico al bit al classico", False,
               f"prova {prova.get('errore_export') or prova.get('errore_import') or prova['rc']} · rif {rif.get('errore_export')}")
        if pronti is not None:
            ok(f"{nome}: «{pronti} strati dai pacchetti pronti» nel log", f"{pronti} strati dai pacchetti pronti" in prova["log"])
        for t in testo or []:
            ok(f"{nome}: nel log «{t}»", t in prova["log"])

    (lavoro / "gguf").mkdir(exist_ok=True)
    for f in (lavoro / "gguf").glob("*.gguf"):
        f.unlink()

    def dirs(nome):
        out, pac = lavoro / "codici" / nome, lavoro / "pacchetti" / nome
        for d in (out, pac):
            if d.exists():
                shutil.rmtree(d)
        return out, pac

    print("\n══ scenari ══", flush=True)
    # 0-2. spento / solo a blocchi / classico: stessi byte
    out, pac = dirs("spento")
    p0 = b.corsa(base, [["scrivi_come_forgia_globale", 1], ["esporta"]],
                 b.amb("spento", out, lavoro / "gguf" / "classico.gguf"), "classico")
    ok("classico (copia NON patchata) esporta", "gguf" in p0, p0.get("errore_export", ""))
    p1 = b.corsa(F, [["esporta"]], b.amb("spento", out, lavoro / "gguf" / "F_spento.gguf",
                                         THOR_EXPORT_A_BLOCCHI="0", THOR_EXPORT_INCREMENTALE="0"), "F_spento")
    controlla("patch, tutto spento", p1, p0)
    p2 = b.corsa(F, [["esporta"]], b.amb("spento", out, lavoro / "gguf" / "F_blocchi.gguf",
                                         THOR_EXPORT_A_BLOCCHI="1"), "F_blocchi")
    controlla("patch, solo a blocchi (blocco 2 su 5 esperti: 2+2+1)", p2, p0)

    # 3. incrementale come forgia_globale
    out, pac = dirs("incr")
    p3 = b.corsa(F, [["scrivi_come_forgia_globale", 1], ["esporta"]],
                 b.amb("incr", out, lavoro / "gguf" / "F_incr.gguf", THOR_EXPORT_INCREMENTALE="1",
                       THOR_EXPORT_PACCHETTI_DIR=pac), "F_incr")
    controlla("incrementale (scrittore in un thread, finestre da 2)", p3, riferimento(out, "incr"), pronti="4/4",
              testo=["con le verifiche del packer", "pacchetti cancellati dopo la rilettura"])
    ok("incrementale: verifica_packer fatta dal thread su TUTTI gli esperti (3 blocchi: 0:2, 2:4, 4:5)",
       all(f"[esperti {a}:{c}]" in p3["log"] for a, c in ((0, 2), (2, 4), (4, 5))))
    for c, nome_c in ((tutte_prima, "tutte_prima"), (tutte_dopo, "tutte_dopo")):
        out_c, pac_c = dirs(f"incr_{nome_c}")
        pc = b.corsa(c, [["scrivi_come_forgia_globale", 1], ["esporta"]],
                     b.amb(f"incr_{nome_c}", out_c, lavoro / "gguf" / f"F_incr_{nome_c}.gguf",
                           THOR_EXPORT_INCREMENTALE="1", THOR_EXPORT_PACCHETTI_DIR=pac_c), f"F_incr_{nome_c}")
        controlla(f"incrementale con TUTTE le 8 patch ({nome_c})", pc, riferimento(out_c, f"incr_{nome_c}"), pronti="4/4")
    ok("incrementale: cartella dei pacchetti vuota dopo l'export", not (pac / "collaudo-F-incr").exists()
       or not any((pac / "collaudo-F-incr").iterdir()))

    # 4. interruzione DENTRO la scrittura di un pacchetto + ripresa che ricalcola lo strato
    out, pac = dirs("int_pacchetto")
    a = b.amb("int", out, lavoro / "gguf" / "F_int.gguf", THOR_EXPORT_INCREMENTALE="1", THOR_EXPORT_PACCHETTI_DIR=pac)
    pA = b.corsa(F, [["scrivi", 0, 1], ["scrivi", 1, 1], ["attendi"], ["muori_scrivendo", "pacchetto_02_ffn_up_exps.npy"],
                     ["scrivi", 2, 1], ["attendi"]], a, "int_A")
    ok("interruzione: il processo A muore dentro la scrittura del pacchetto dello strato 2", pA["rc"] == 9,
       f"rc {pA['rc']}")
    ok("... lasciando un pacchetto a meta' (gate scritto, up .tmp, nessuna marca)",
       (pac / "collaudo-F-int" / "pacchetto_02_ffn_gate_exps.npy").exists()
       and (pac / "collaudo-F-int" / "pacchetto_02_ffn_up_exps.npy.tmp").exists()
       and not (pac / "collaudo-F-int" / "pacchetto_02.json").exists())
    pB = b.corsa(F, [["scrivi", 2, 2], ["scrivi", 3, 2], ["esporta"]], a, "int_B")
    controlla("interruzione + ripresa (strato 2 ricalcolato con codici diversi)", pB, riferimento(out, "int"), pronti="4/4")

    # 5. ucciso subito dopo i codici: il pacchetto dello strato 1 non esiste mai
    out, pac = dirs("ucciso")
    a = b.amb("ucc", out, lavoro / "gguf" / "F_ucc.gguf", THOR_EXPORT_INCREMENTALE="1", THOR_EXPORT_PACCHETTI_DIR=pac)
    b.corsa(F, [["scrivi", 0, 3], ["attendi"], ["scrivi", 1, 3], ["muori_subito"]], a, "ucc_A")
    pB = b.corsa(F, [["scrivi", 2, 3], ["scrivi", 3, 3], ["esporta"]], a, "ucc_B")
    rif = riferimento(out, "ucc")
    controlla("ucciso dopo i codici + ripresa (strato 1 senza pacchetto)", pB, rif,
              testo=["strato 1: pacchetto pronto NON usato (assente)"])

    # 6-9. pacchetti tenuti, poi alterati in quattro modi diversi
    def con_pacchetti(nome):
        out, pac = dirs(nome)
        a = b.amb(nome, out, lavoro / "gguf" / f"F_{nome}.gguf", THOR_EXPORT_INCREMENTALE="1",
                  THOR_EXPORT_PACCHETTI_DIR=pac, THOR_EXPORT_PACCHETTI_TIENI="1")
        pA = b.corsa(F, [["scrivi_come_forgia_globale", 4], ["attendi"]], a, f"{nome}_A")
        ok(f"{nome}: 4 pacchetti pronti", len(list((pac / f"collaudo-F-{nome}").glob("pacchetto_??.json"))) == 4,
           pA.get("errore_export", ""))
        return out, pac / f"collaudo-F-{nome}", a

    import numpy as np
    out, dp, a = con_pacchetti("stantio")
    q = np.load(out / "Qg_01.npy")
    q[0, 0, :7] = -q[0, 0, :7]
    q[0, 0, 7] = 1 if q[0, 0, 7] != 1 else 0
    np.save(out / "Qg_01.npy", q)
    pB = b.corsa(F, [["esporta"]], a, "stantio_B")
    controlla("codici riscritti dopo il pacchetto", pB, riferimento(out, "stantio"), pronti="3/4",
              testo=["strato 1: pacchetto pronto NON usato (codici su disco cambiati dopo il pacchetto)"])

    out, dp, a = con_pacchetti("alterato")
    f = dp / "pacchetto_02_ffn_gate_exps.npy"
    bb = bytearray(f.read_bytes())
    bb[-1] ^= 0x10
    f.write_bytes(bytes(bb))
    pB = b.corsa(F, [["esporta"]], a, "alterato_B")
    controlla("un byte del pacchetto alterato", pB, riferimento(out, "alterato"), pronti="3/4",
              testo=["byte su disco diversi da quelli scritti (sha256)"])

    out, dp, a = con_pacchetti("impacchettatore")
    pk = F / "pack_tq1_0_standard.py"
    orig = pk.read_bytes()
    pk.write_bytes(orig + b"\n# cambiato (collaudo)\n")
    try:
        pB = b.corsa(F, [["esporta"]], a, "impacchettatore_B")
    finally:
        pk.write_bytes(orig)
    controlla("impacchettatore cambiato", pB, riferimento(out, "impacchettatore"), pronti="0/4",
              testo=["impacchettatore (sorgenti o numpy) cambiato"])

    out, dp, a = con_pacchetti("senza_verifica")
    m = json.loads((dp / "pacchetto_00.json").read_text())
    m["verificato_gate"] = False
    (dp / "pacchetto_00.json").write_text(json.dumps(m))
    pB = b.corsa(F, [["esporta"]], a, "senza_verifica_B")
    controlla("primo strato senza verifiche del packer", pB, riferimento(out, "senza_verifica"), pronti="3/4",
              testo=["serve la verifica del packer e il pacchetto non l'ha fatta", "blk.0.ffn_gate_exps.weight: packer round-trip"])

    # 10. packer rotto all'export: deve fallire come il classico (il campione lo vede)
    out, dp, a = con_pacchetti("packer_rotto")
    pB = b.corsa(F, [["spezza_packer"], ["esporta"]], a, "packer_rotto_B")
    rif = b.corsa(base, [["spezza_packer"], ["esporta"]], b.amb("packer_rotto", out, lavoro / "gguf" / "rif_rotto.gguf"),
                  "packer_rotto_rif")
    ok("packer rotto all'export: il classico fallisce", "errore_export" in rif and "NON bit-esatt" in rif["errore_export"],
       rif.get("errore_export", "")[:120])
    ok("... e l'incrementale fallisce con lo STESSO errore (campione diverso → via di sempre → verifica)",
       pB.get("errore_export") == rif.get("errore_export") and "l'esperto" in pB["log"],
       (pB.get("errore_export") or "")[:120])

    # 11. tipo base sbagliato: stesso errore
    out, dp, a = con_pacchetti("tipo_base")
    a2 = {**a, "THOR_SCHUR_BASE": str(b.base_q4k)}
    pB = b.corsa(F, [["esporta"]], a2, "tipo_base_B")
    rif = riferimento(out, "tipo_base", base=b.base_q4k)
    ok("tipo base sbagliato: stesso errore del classico, dai pacchetti pronti",
       pB.get("errore_export") and pB.get("errore_export") == rif.get("errore_export")
       and "tipo base 12 != atteso 34" in pB["errore_export"],
       (pB.get("errore_export") or "")[:110])

    # 12. disco «pieno»
    out, pac = dirs("pieno")
    p = b.corsa(F, [["scrivi_come_forgia_globale", 5], ["esporta"]],
                b.amb("pieno", out, lavoro / "gguf" / "F_pieno.gguf", THOR_EXPORT_INCREMENTALE="1",
                      THOR_EXPORT_PACCHETTI_DIR=pac, THOR_EXPORT_PACCHETTI_MARGINE_GIB="1000000"), "pieno")
    controlla("spazio insufficiente: nessun pacchetto, via di sempre", p, riferimento(out, "pieno"), pronti="0/4",
              testo=["NON scritto:", "GiB liberi"])

    # 13. import senza cartella dei pacchetti
    out, pac = dirs("senza_dir")
    p = b.corsa(F, [["esporta"]], b.amb("senza_dir", out, lavoro / "gguf" / "x.gguf", THOR_EXPORT_INCREMENTALE="1"),
                "senza_dir")
    ok("THOR_EXPORT_INCREMENTALE=1 senza cartella: l'import di valida si ferma subito",
       "THOR_EXPORT_PACCHETTI_DIR" in p.get("errore_import", ""), p.get("errore_import", "")[:100])

    dopo = {f.name: f.read_bytes() for f in THOR_V2.glob("*.py")}
    nuovi = sorted(set(dopo) - set(vero_prima))
    ok("i file veri di thor_v2 NON sono stati toccati (ogni .py di prima, identico al byte)",
       all(dopo.get(n) == b for n, b in vero_prima.items()),
       (f"nati durante il collaudo, da ALTRI processi (il collaudo scrive solo in --lavoro): {nuovi}" if nuovi else ""))
    n_ok = sum(1 for _, v in ESITI if v)
    print(f"\n{'✅ TUTTI' if n_ok == len(ESITI) else '⛔ FALLITI ' + str(len(ESITI) - n_ok) + ' di'} {len(ESITI)} "
          f"controlli in {time.time() - t0:.0f}s")
    return 0 if n_ok == len(ESITI) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lavoro", type=Path)
    ap.add_argument("--figlio", action="store_true")
    ap.add_argument("--cartella", type=Path)
    ap.add_argument("--passi", default="[]")
    ap.add_argument("--uscita", type=Path)
    args = ap.parse_args()
    if args.figlio:
        return figlio(args.cartella.resolve(), json.loads(args.passi), args.uscita.resolve())
    lavoro = args.lavoro.resolve()
    if lavoro == THOR_V2 or THOR_V2 in lavoro.parents:
        raise SystemExit("la cartella di lavoro non puo' stare dentro thor_v2")
    return main_padre(lavoro)


if __name__ == "__main__":
    raise SystemExit(main())

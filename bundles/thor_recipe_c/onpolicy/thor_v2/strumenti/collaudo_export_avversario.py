#!/usr/bin/env python3
"""Verifica AVVERSARIA dell'export incrementale (leva F) — 25/9 mattina, solo CPU, scala giocattolo.

Scenari che il collaudo del progettista (strumenti/collaudo_export_incrementale.py) non provava. Riusa il
suo banco (copie di thor_v2, GGUF base giocattolo, codici casuali col contratto vero) e confronta sempre
l'sha256 del GGUF con l'export CLASSICO (copia NON patchata) sugli stessi codici su disco:
  X1 ordine di commit ROVESCIO (3,2,1,0) — il thread verifica per primo lo strato 3
  X2 ordine sparso + uno strato scritto DUE volte nello stesso processo (codici diversi)
  X3 gara: lo strato riscritto MENTRE il thread lo impacchetta (impacchettatore rallentato)
  X4 gara stretta: codici riscritti DOPO il controllo delle impronte e prima della marca .json, poi il
     processo muore col secondo pacchetto a meta' → resta una marca vecchia valida per i codici VECCHI
  X5 nessuno strato col down ternario (THOR_FORGIA_DOWN_TERNARIO=0: marche down_ternario=0)
  X6 Dd/Qd VECCHI su disco con la marca che dice down_ternario=0 (rischio del classico, non della patch)
  X7 interruttori della ricetta diversi fra forgia ed export (rotazione d'ingresso, nucleo bf16):
     l'export non li legge
  X8 blocco diverso fra chi impacchetta (1 esperto) e chi esporta (3 esperti)
  X9 base con i METADATI VERI di Thor-v20-densiv5.gguf (tutti i campi, tokenizzatore compreso) e tensori
     non esperti di tipi diversi (F32, F16, Q8_0, Q4_K, IQ4_XS, Q4_0, Q5_K, Q6_K): GGUF identico, e quali campi il classico
     perde in silenzio (except Exception: pass nella copia dei metadati)

    python3 strumenti/collaudo_export_avversario.py --lavoro <scratch>        (CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= nice 19)
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

C_PATH = Path("@@ROOT:forge@@/thor_v2/strumenti/collaudo_export_incrementale.py")
BASE_VERA = Path("@@ROOT:models@@/gguf/tony-tern/Thor-v20-densiv5.gguf")
PY = "@@ROOT:venv@@/bin/python3"
ESITI: list = []


def ok(nome, cond, dettaglio=""):
    ESITI.append((nome, bool(cond)))
    print(f"  {'✅' if cond else '⛔'} {nome}" + (f" — {dettaglio}" if dettaglio else ""), flush=True)


def carica_collaudo():
    spec = importlib.util.spec_from_file_location("collaudo_export", C_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ═══════════════════════════ figlio: una corsa, con i passi in piu' ═══════════════════════════
def figlio(cartella: Path, passi: list, uscita: Path) -> int:
    os.chdir(cartella)
    sys.path.insert(0, str(cartella))
    C = carica_collaudo()
    ris: dict = {"eventi": []}
    try:
        import valida_finestra_8_12 as V
    except Exception as e:  # noqa: BLE001
        ris["errore_import"] = f"{type(e).__name__}: {e}"
        uscita.write_text(json.dumps(ris))
        return 0
    for passo in passi:
        tipo = passo[0]
        if tipo == "scrivi":                              # (scrivi, L, seme[, con_down])
            L, seme = passo[1], passo[2]
            con_down = passo[3] if len(passo) > 3 else C.CON_DOWN[L]
            V.scrivi_codici_su_disco({L: C.codici(L, seme, con_down)})
        elif tipo == "scrivi_file_down_vecchi":            # Dd/Qd scritti A MANO (non da scrivi_codici)
            L, seme = passo[1], passo[2]
            Dd, Qd = C.codici(L, seme, True)["ffn_down_exps"]
            import numpy as np
            V.OUT_DIR.mkdir(parents=True, exist_ok=True)
            np.save(V.OUT_DIR / f"Dd_{L:02d}.npy", Dd)
            np.save(V.OUT_DIR / f"Qd_{L:02d}.npy", Qd)
        elif tipo == "orologio_fisso":                   # marca fatto_forgia con lo STESSO testo a ogni scrittura:
            _vero = V.time.strftime                       # resta solo l'impronta dei file a distinguere i codici
            V.time.strftime = lambda fmt, *r, _v=_vero: "2026-09-25 00:00:00" if fmt == "%F %T" else _v(fmt, *r)
        elif tipo == "dormi":
            time.sleep(passo[1])
        elif tipo == "attendi":
            V._attendi_impacchettamenti()
        elif tipo == "muori_subito":
            sys.stdout.flush()
            os._exit(9)
        elif tipo == "rallenta_impacchettamento":   # (secondi per le prime n_primi chiamate, poi, n_primi)
            orig, stato = V._pacchetto_gate_up, {"n": 0}          # 2 chiamate per strato: gate e up
            secondi, poi = passo[1], (passo[2] if len(passo) > 2 else 0)
            n_primi = passo[3] if len(passo) > 3 else 1

            def lento(D, Q, _o=orig, _s=stato):
                _s["n"] += 1
                time.sleep(secondi if _s["n"] <= n_primi else poi)
                return _o(D, Q)
            V._pacchetto_gate_up = lento
        elif tipo == "rallenta_salvataggio":              # il PRIMO salvataggio di un pacchetto dorme
            orig, stato = V._salva_npy_atomico, {"n": 0}

            def lento_s(p, a, _o=orig, _s=stato, _sec=passo[1]):
                _s["n"] += 1
                if _s["n"] == 1:
                    time.sleep(_sec)
                return _o(p, a)
            V._salva_npy_atomico = lento_s
        elif tipo == "esporta":
            try:
                ris["gguf"] = str(V.esporta_gguf_finestra(layers=list(range(C.NL))))
            except BaseException as e:  # noqa: BLE001
                ris["errore_export"] = f"{type(e).__name__}: {e}"
        ris["eventi"].append(tipo)
    uscita.write_text(json.dumps(ris))
    return 0


# ═══════════════════════════ base con i metadati veri ═══════════════════════════
def gguf_base_realistico(percorso: Path, C) -> dict:
    sys.path.insert(0, os.getenv("GGUF_PY_DIR", "@@ROOT:models@@/build/llamacpp-tq1-k1024/gguf-py"))
    import numpy as np
    from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as T
    vera = GGUFReader(str(BASE_VERA))
    w = GGUFWriter(str(percorso), arch="qwen35moe")
    copiati, saltati = [], []
    for fld in vera.fields.values():                     # STESSA copia dell'export (esporta_gguf_finestra)
        if fld.name.startswith(("GGUF.", "split.")) or fld.name == "general.architecture":
            continue
        try:
            w.add_key_value(fld.name, fld.contents(), fld.types[0],
                            sub_type=fld.types[-1] if len(fld.types) > 1 else None)
            copiati.append(fld.name)
        except Exception as e:  # noqa: BLE001
            saltati.append(f"{fld.name} ({type(e).__name__})")
    tipi_veri = sorted({T(int(t.tensor_type)).name for t in vera.tensors})
    rng = np.random.default_rng(11)
    E, H, I, NL = C.E, C.H, C.I, C.NL
    tens = [("token_embd.weight", rng.standard_normal((8, H)).astype(np.float32), None),
            ("output_norm.weight", rng.standard_normal(H).astype(np.float32), None),
            ("output.weight", rng.integers(0, 256, size=(8, (H // 32) * 34), dtype=np.uint8), T.Q8_0)]
    for L in range(NL):
        tens.append((f"blk.{L}.attn_norm.weight", rng.standard_normal(H).astype(np.float32), None))
        tens.append((f"blk.{L}.ffn_gate_inp.weight", rng.standard_normal((E, H)).astype(np.float16), None))
        tens.append((f"blk.{L}.attn_qkv.weight", rng.integers(0, 256, size=(16, (H // 256) * 144), dtype=np.uint8), T.Q4_K))
        tens.append((f"blk.{L}.ffn_up_shexp.weight", rng.integers(0, 256, size=(16, (H // 256) * 136), dtype=np.uint8), T.IQ4_XS))
        tens.append((f"blk.{L}.attn_gate.weight", rng.integers(0, 256, size=(16, (H // 32) * 18), dtype=np.uint8), T.Q4_0))
        tens.append((f"blk.{L}.attn_output.weight", rng.integers(0, 256, size=(16, (H // 256) * 176), dtype=np.uint8), T.Q5_K))
        tens.append((f"blk.{L}.ssm_out.weight", rng.integers(0, 256, size=(16, (H // 256) * 210), dtype=np.uint8), T.Q6_K))
        for fam in ("ffn_gate_exps", "ffn_up_exps"):
            tens.append((f"blk.{L}.{fam}.weight", rng.integers(0, 256, size=(E, I, (H // 256) * 54), dtype=np.uint8), T.TQ1_0))
        tipo_d = (T.Q4_K, T.TQ1_0, T.IQ4_XS, T.Q4_K)[L]
        nb = (I // 256) * {T.Q4_K: 144, T.TQ1_0: 54, T.IQ4_XS: 136}[tipo_d]
        tens.append((f"blk.{L}.ffn_down_exps.weight", rng.integers(0, 256, size=(E, H, nb), dtype=np.uint8), tipo_d))
    for nome, a, tipo in tens:
        w.add_tensor(nome, a) if tipo is None else w.add_tensor(nome, a, raw_dtype=tipo)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return {"campi_veri": len(vera.fields), "copiati": len(copiati), "saltati": saltati, "tipi_veri": tipi_veri}


def campi(percorso: Path) -> dict:
    sys.path.insert(0, os.getenv("GGUF_PY_DIR", "@@ROOT:models@@/build/llamacpp-tq1-k1024/gguf-py"))
    from gguf import GGUFReader
    r = GGUFReader(str(percorso))
    out = {}
    for f in r.fields.values():
        if f.name.startswith("GGUF."):
            continue
        try:
            v = f.contents()
        except Exception:  # noqa: BLE001
            v = "<illeggibile>"
        out[f.name] = hashlib.sha256(repr(v).encode()).hexdigest()[:16]
    return out


# ═══════════════════════════ padre ═══════════════════════════
def main_padre(lavoro: Path) -> int:
    t0 = time.time()
    C = carica_collaudo()
    lavoro.mkdir(parents=True, exist_ok=True)
    b = C.Banco(lavoro)
    base = b.copia("base", [])
    F = b.copia("F", ["export_incrementale"])
    for f in (b.base,):
        if f.exists():
            f.unlink()
    C.gguf_base(b.base)
    (lavoro / "gguf").mkdir(exist_ok=True)
    for f in (lavoro / "gguf").glob("*.gguf"):
        f.unlink()
    n = [0]

    def corsa(cartella, passi, amb, etichetta):
        n[0] += 1
        uscita = lavoro / "corse" / f"{n[0]:02d}_{etichetta}"
        if uscita.exists():
            shutil.rmtree(uscita)
        uscita.mkdir(parents=True)
        env = {**os.environ, **C.ENV_CPU, **amb}
        with open(uscita / "log.txt", "w") as flog:
            r = subprocess.run([PY, __file__, "--figlio", "--cartella", str(cartella), "--passi", json.dumps(passi),
                                "--uscita", str(uscita / "r.json")], stdout=flog, stderr=subprocess.STDOUT, env=env)
        ris = json.loads((uscita / "r.json").read_text()) if (uscita / "r.json").exists() else {}
        ris.update({"rc": r.returncode, "log": (uscita / "log.txt").read_text()})
        return ris

    def dirs(nome):
        out, pac = lavoro / "codici" / nome, lavoro / "pacchetti" / nome
        for d in (out, pac):
            if d.exists():
                shutil.rmtree(d)
        return out, pac

    def amb(nome, out, pac=None, **extra):
        g = lavoro / "gguf" / f"{nome}_{'F' if pac else 'rif'}_{time.time_ns()}.gguf"
        a = b.amb(nome, out, g, **extra)
        if pac is not None:
            a.update({"THOR_EXPORT_INCREMENTALE": "1", "THOR_EXPORT_PACCHETTI_DIR": str(pac)})
        return a

    def confronta(nome, prova, out, pronti=None, testo=(), **extra_rif):
        rif = corsa(base, [["esporta"]], amb(nome, out, **extra_rif), f"{nome}_rif")
        if "gguf" in prova and "gguf" in rif:
            s_p, s_r = C.sha(prova["gguf"]), C.sha(rif["gguf"])
            ok(f"{nome}: GGUF identico al bit al classico sugli stessi codici", s_p == s_r, s_p[:16])
        else:
            ok(f"{nome}: GGUF identico al bit al classico", prova.get("errore_export") == rif.get("errore_export")
               and prova.get("errore_export") is not None,
               f"prova {prova.get('errore_export') or prova.get('errore_import') or prova['rc']} · rif {rif.get('errore_export')}")
        if pronti is not None:
            ok(f"{nome}: «{pronti} strati dai pacchetti pronti»", f"{pronti} strati dai pacchetti pronti" in prova["log"],
               next((r for r in prova["log"].splitlines() if "strati dai pacchetti" in r), "")[-80:])
        for t in testo:
            ok(f"{nome}: nel log «{t[:70]}»", t in prova["log"])
        return rif

    print("══ scenari avversari ══", flush=True)
    # X1
    out, pac = dirs("X1")
    p = corsa(F, [["scrivi", L, 1] for L in (3, 2, 1, 0)] + [["attendi"], ["esporta"]], amb("X1", out, pac), "X1")
    confronta("X1 ordine rovescio 3,2,1,0", p, out, pronti="4/4", testo=["[esperti 0:2]"])
    # X2
    out, pac = dirs("X2")
    p = corsa(F, [["scrivi", 2, 1], ["scrivi", 0, 1], ["scrivi", 1, 1], ["scrivi", 3, 1], ["scrivi", 1, 7],
                  ["attendi"], ["esporta"]], amb("X2", out, pac), "X2")
    confronta("X2 ordine sparso + strato 1 scritto due volte (seme 1 poi 7)", p, out, pronti="4/4")
    # X3
    out, pac = dirs("X3")
    p = corsa(F, [["rallenta_impacchettamento", 2.0], ["scrivi", 1, 1], ["dormi", 0.6], ["scrivi", 1, 5],
                  ["attendi"], ["scrivi", 0, 1], ["scrivi", 2, 1], ["scrivi", 3, 1], ["attendi"], ["esporta"]],
              amb("X3", out, pac), "X3")
    confronta("X3 strato 1 riscritto MENTRE il thread lo impacchetta", p, out, pronti="4/4",
              testo=["i codici sono cambiati mentre li impacchettavo"])
    # X4
    out, pac = dirs("X4")
    pA = corsa(F, [["orologio_fisso"], ["rallenta_salvataggio", 2.0], ["rallenta_impacchettamento", 0, 600, 2], ["scrivi", 1, 1],
                   ["dormi", 0.8], ["scrivi", 1, 9], ["dormi", 3.5], ["muori_subito"]], amb("X4", out, pac), "X4_A")
    marca = pac / "collaudo-F-X4" / "pacchetto_01.json"
    vecchia = json.loads(marca.read_text()) if marca.exists() else {}
    ok("X4: il processo A muore e resta la marca .json del pacchetto VECCHIO (codici seme 1) accanto ai codici seme 9",
       pA["rc"] == 9 and marca.exists() and vecchia.get("sorgenti") is not None, f"rc {pA['rc']}, marca {marca.exists()}")
    ok("X4: le due marche fatto_forgia dello strato 1 (seme 1 e seme 9) hanno lo STESSO testo",
       vecchia.get("marca") == (out / "fatto_forgia_01.ok").read_text(), vecchia.get("marca", "")[:60])
    p = corsa(F, [["orologio_fisso"], ["scrivi", 0, 1], ["scrivi", 2, 1], ["scrivi", 3, 1], ["esporta"]],
              amb("X4", out, pac), "X4_B")
    confronta("X4 marca vecchia valida per codici VECCHI, poi riscritti: rifiutata", p, out, pronti="3/4",
              testo=["strato 1: pacchetto pronto NON usato (codici su disco cambiati dopo il pacchetto)"])
    # X5
    out, pac = dirs("X5")
    p = corsa(F, [["scrivi", L, 2, False] for L in range(4)] + [["attendi"], ["esporta"]],
              amb("X5", out, pac, THOR_FORGIA_DOWN_TERNARIO="0"), "X5")
    confronta("X5 nessuno strato col down ternario", p, out, pronti="4/4", THOR_FORGIA_DOWN_TERNARIO="0")
    ok("X5: il GGUF dichiara zero strati col down ternario", '"strati_con_down_ternario": []' in
       (Path(p["gguf"]).read_bytes()[:200000].decode(errors="replace") if "gguf" in p else ""))
    # X6
    out, pac = dirs("X6")
    p = corsa(F, [["scrivi_file_down_vecchi", 1, 3], ["scrivi", 0, 2], ["scrivi", 1, 2, False], ["scrivi", 2, 2],
                  ["scrivi", 3, 2], ["attendi"], ["esporta"]], amb("X6", out, pac), "X6")
    confronta("X6 Dd/Qd VECCHI su disco con marca down_ternario=0 (classico e incrementale identici)", p, out,
              pronti="4/4")
    # X7
    out, pac = dirs("X7")
    corsa(F, [["scrivi", L, 4] for L in range(4)] + [["attendi"]],
          amb("X7", out, pac, THOR_ROTAZIONE_INGRESSO="1", THOR_NUCLEO_BF16="1", THOR_ESPERTI_THOR="1",
              THOR_EXPORT_PACCHETTI_TIENI="1"), "X7_forgia")
    for rot in ("0", "1"):
        p = corsa(F, [["esporta"]], amb("X7", out, pac, THOR_ROTAZIONE_INGRESSO=rot, THOR_NUCLEO_BF16=rot,
                                        THOR_EXPORT_PACCHETTI_TIENI="1"), f"X7_export_rot{rot}")
        confronta(f"X7 pacchetti fatti con rotazione+bf16, export con THOR_ROTAZIONE_INGRESSO={rot}", p, out,
                  pronti="4/4", THOR_ROTAZIONE_INGRESSO="0" if rot == "1" else "1")
    # X8
    out, pac = dirs("X8")
    corsa(F, [["scrivi", L, 6] for L in range(4)] + [["attendi"]],
          amb("X8", out, pac, THOR_EXPORT_PACCHETTI_BLOCCO="1", THOR_EXPORT_PACCHETTI_TIENI="1"), "X8_forgia")
    p = corsa(F, [["esporta"]], amb("X8", out, pac, THOR_EXPORT_PACCHETTI_BLOCCO="3"), "X8_export")
    confronta("X8 pacchetti a blocchi di 1 esperto, export a blocchi di 3", p, out, pronti="4/4")
    # X9
    base_r = lavoro / "base_realistica.gguf"
    if base_r.exists():
        base_r.unlink()
    info = gguf_base_realistico(base_r, C)
    ok(f"X9: base con i metadati VERI ({info['copiati']} campi copiati da {BASE_VERA.name}); tipi dei tensori "
       f"veri: {info['tipi_veri']}", info["copiati"] > 10, f"campi non copiabili: {info['saltati']}")
    out, pac = dirs("X9")
    p = corsa(F, [["scrivi", L, 8] for L in range(4)] + [["attendi"], ["esporta"]],
              amb("X9", out, pac, base=base_r), "X9")
    rif = confronta("X9 metadati veri + tensori non esperti di 8 tipi (F32 F16 Q8_0 Q4_K IQ4_XS Q4_0 Q5_K Q6_K)", p, out, pronti="4/4", base=base_r)
    if "gguf" in rif:
        c_base, c_out = campi(base_r), campi(Path(rif["gguf"]))
        persi = sorted(set(c_base) - set(c_out))
        cambiati = sorted(k for k in set(c_base) & set(c_out) if c_base[k] != c_out[k] and k != "general.architecture")
        ok(f"X9: l'export CLASSICO conserva tutti i campi della base (persi {len(persi)}, cambiati {len(cambiati)})",
           not persi and not cambiati, f"persi {persi[:6]} cambiati {cambiati[:6]}")
    n_ok = sum(1 for _, v in ESITI if v)
    print(f"\n{'✅ TUTTI' if n_ok == len(ESITI) else '⛔ FALLITI ' + str(len(ESITI) - n_ok) + ' di'} {len(ESITI)} "
          f"controlli in {time.time() - t0:.0f}s")
    return 0 if n_ok == len(ESITI) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lavoro", type=Path)
    ap.add_argument("--figlio", action="store_true")
    ap.add_argument("--cartella", type=Path)
    ap.add_argument("--passi", default="[]")
    ap.add_argument("--uscita", type=Path)
    a = ap.parse_args()
    if a.figlio:
        return figlio(a.cartella.resolve(), json.loads(a.passi), a.uscita.resolve())
    return main_padre(a.lavoro.resolve())


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Pieghe/A.1 con pianificatore comune; solo directory nuove e riprese proprie.

Esempio: --fase densi --checkpoint residui_delta.pt --gguf base.gguf
         --out nuova_corsa --profilo picchi.json --gpu all --processi-per-gpu auto
Il risultato consumabile dagli strumenti storici e' <out>/pubblicato.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"  # priorita' 0 (21/10): ogni worker inductor apre /dev/kfd e prende una coda SDMA
from pathlib import Path
import sys
import uuid

import pianificatore_gpu as PG

QUI = Path(__file__).resolve().parent


def dipendenze_codice():
    return [QUI / n for n in ("pianificatore_gpu.py", "cardan_fucina_gpu.py", "cardan_unita_gpu.py",
                              "cardan_piega_gpu.py", "cardan_piega_densi_gpu.py",
                              "cardan_costruisci_gpu.py", "cardan_base.py",
                              "cardan_algebra_guardia_gpu.py", "cardan_guardia_gpu.py",
                              "pianificatore_guardia_gpu.py",
                              "pianificatore_misurato_gpu.py", "cardan_obiettivo_gpu.py",
                              "cardan_telemetria_gpu.py", "cardan_classi_gpu.py",
                              "cardan_input_densi_gpu.py", "cardan_picchi_code_gpu.py",
                              "cardan_code_gpu.py", "gpu_queues.py")] + [
        Path("@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/ROCSOLVER_RIPARA_1020/algebra_gpu.py"),
        Path("@@ROOT:models@@/build/leve/raffina/comune.py"),
    ]


def prepara(a):
    import torch
    import cardan_piega_densi_gpu as CD
    import cardan_piega_gpu as CP
    sys.path.insert(0, CP.RAFF)
    import comune as CM
    from rd_ternary.formato import TensorKey
    lo, _, hi = a.strati.partition("-")
    strati = list(range(int(lo), int(hi or lo) + 1))
    if not strati or min(strati) < 0 or max(strati) > 39:
        raise ValueError("--strati deve essere un intervallo crescente fra 0 e 39")
    unita, ingressi = [], [Path(a.gguf)]
    if a.fase == "a1":
        import numpy as np
        for L in strati:
            hfile = Path(a.hdir) / f"H_ffn_{L:02d}.npy"
            h = np.load(hfile, mmap_mode="r", allow_pickle=False)
            forma_h = list(h.shape)
            del h
            unita.append(PG.Unita(f"a1-{L:02d}", {
                "strato": L, "forma_h": forma_h,
                "tipo_strato": "esperti-256x512x2048"}))
            ingressi.append(hfile)
        # L'indice e tutti gli shard HF sono dipendenze immutabili.
        import sorgente
        hf = sorgente.HF.resolve(strict=True)
        if a.hf_radice and Path(a.hf_radice).resolve(strict=True) != hf:
            raise ValueError("--hf-radice non coincide con sorgente.HF / TT_HF")
        shards = sorted(hf.glob("*.safetensors"))
        if not shards:
            raise ValueError(f"nessuno shard safetensors in {hf}")
        ingressi.extend(shards)
        ingressi.extend(sorted(hf.glob("*.json")))
        ingressi.extend([sorgente.MAPPA_JSON, sorgente.MAPPA_NPZ, Path(sorgente.__file__)])
    else:
        if not a.checkpoint:
            raise ValueError("--checkpoint obbligatorio per le pieghe")
        dd = torch.load(a.checkpoint, mmap=True, map_location="cpu", weights_only=False)
        if dd["rango"] != a.rango:
            raise ValueError(f"rango checkpoint {dd['rango']} diverso da --rango {a.rango}")
        ingressi.append(Path(a.checkpoint))
        if a.fase == "esperti":
            src = CM.gp.GGUFPlaneSource(a.gguf, layers=strati, bases=["ffn_gate_exps", "ffn_up_exps"])
            for L in strati:
                if L not in dd["stato"] and str(L) not in dd["stato"]:
                    raise ValueError(f"checkpoint senza strato richiesto {L}")
                nomi = []
                forme = []
                for fam in ("gate", "up"):
                    e = src.entry(TensorKey(L, 0, fam))
                    if len(e.tensors) > 2:
                        raise ValueError(f"originale non gestisce >2 piani esperti: {L}/{fam}")
                    nomi.extend(f"{t.name}.npy" for t in e.tensors)
                    forme.append([int(e.out), int(e.in_), int(e.n_experts),
                                  len(e.tensors), bool(e.fmt.rotated)])
                gq = Path(CP.GQ_DIR) / f"Gq_{L:02d}.npy"
                unita.append(PG.Unita(f"esperti-{L:02d}", {
                    "strato": L, "file_attesi": nomi, "forme": forme, "gq": gq.exists()}))
                ingressi.append(Path(CP.HFFN_DIR) / f"H_ffn_{L:02d}.npy")
                if gq.exists():
                    ingressi.append(gq)
        else:
            manif = json.loads(Path(CD.MANIF_R2B3).read_text())
            ingressi.append(Path(CD.MANIF_R2B3))
            delta = {CD.nome_gguf(k): v for k, v in dd["delta_file"].items()}
            gruppi = {}
            piani_per_nome = {}
            for L in strati:
                nomi = sorted(n for n in delta if int(n.split(".")[1]) == L)
                src = CM.gp.GGUFPlaneSource(a.gguf, layers=[L])
                for nome in nomi:
                    base = nome.split(".")[-2]
                    e = src.entry(TensorKey(L, -1, base))
                    if tuple(delta[nome].shape) != (e.out, e.in_):
                        raise ValueError(f"forma delta/ospite diversa: {nome}")
                    forma = (int(e.out), int(e.in_), len(e.tensors), bool(e.fmt.rotated))
                    gruppi.setdefault(forma, []).append(nome)
                    piani_per_nome[nome] = [f"{t.name}.npy" for t in e.tensors]
                    info = manif.get(nome.replace(".weight", "3.weight"), {}).get("hessiana")
                    if not info:
                        raise ValueError(f"nessuna Hessiana per {nome}")
                    ingressi.append(Path(info["file"]))
            for g, (forma, nomi) in enumerate(gruppi.items()):
                for i in range(0, len(nomi), a.lotto):
                    parte = nomi[i:i + a.lotto]
                    unita.append(PG.Unita(f"densi-{g:02d}-{i:04d}", {
                        "nomi": parte, "strati": sorted({int(n.split(".")[1]) for n in parte}),
                        "forma": list(forma),
                        "file_attesi": [f for nome in parte for f in piani_per_nome[nome]]}))
    if not unita:
        raise ValueError("nessuna unita' selezionata")
    codici = {str(p): PG.sha_file(p) for p in dipendenze_codice()}
    config = {"fase": a.fase, "gguf": str(Path(a.gguf).resolve()), "checkpoint": a.checkpoint,
              "dtype": a.dtype, "lotto": a.lotto, "algebra": a.algebra, "rango": a.rango,
              "pezzo": a.pezzo, "hdir": a.hdir, "fili_lettura": a.fili_lettura,
              "prepara_gpu": a.prepara_gpu, "fili": a.fili,
              "riuso_input": a.riuso_input, "cache_h_byte": a.cache_h_mib * (1 << 20),
              "strati": strati,
              "fili_io": a.fili_lettura if a.fase == "a1" else (2 if a.fase == "esperti" else 0),
              "codici": codici}
    # Profilo condivisibile solo fra carichi della stessa forma/ricetta/codice.
    ricetta = {"fase": a.fase, "dtype": a.dtype, "lotto": a.lotto, "algebra": a.algebra,
               "rango": a.rango, "pezzo": a.pezzo, "prepara_gpu": a.prepara_gpu,
               "riuso_input": a.riuso_input, "cache_h_byte": config["cache_h_byte"],
               "fili_lettura": a.fili_lettura, "codici": codici,
               "fili": a.fili, "torch": torch.__version__, "hip": torch.version.hip,
               "forme": sorted({(*u.dati["forma"], len(u.dati["nomi"]))
                                for u in unita if "forma" in u.dati}),
               "forme_esperti": sorted({tuple(f) for u in unita for f in u.dati.get("forme", [])})}
    identita = [PG.identita_file(p) for p in sorted(set(ingressi))]
    config["ingressi"] = identita
    from cardan_classi_gpu import collega
    collega(config, unita)
    piano = {"factory": "cardan_unita_gpu:crea", "config": config,
             "unita": [dataclasses.asdict(u) for u in unita],
             "ricetta_sha256": PG.firma_json(ricetta)}
    piano["firma"] = PG.firma_json(piano)
    return piano, unita


def verifica_ingressi(piano):
    for meta in piano["config"]["ingressi"]:
        if PG.identita_file(meta["path"]) != meta:
            raise RuntimeError(f"input cambiato durante la corsa: {meta['path']}")
    for p, sha in piano["config"]["codici"].items():
        if PG.sha_file(p) != sha:
            raise RuntimeError(f"codice cambiato durante la corsa: {p}")


def raccogli(root, piano, unita):
    root = Path(root)
    dest = root / "pubblicato"
    descrizione = PG.Unita("pubblicato", {"unita": [u.chiave for u in unita]})
    for u in unita:
        if not PG.completa(root / "unita" / u.chiave, piano["firma"], u):
            raise RuntimeError(f"manca unita' {u.chiave}")
    if PG.completa(dest, piano["firma"], descrizione):
        return dest
    stage = root / f".pubblicazione-{uuid.uuid4().hex}"
    stage.mkdir()
    if piano["config"]["fase"] == "a1":
        info, misure = {"tensori": {}}, {}
        for u in unita:
            d = root / "unita" / u.chiave
            nomi = list(d.glob("strato_*.npz"))
            if len(nomi) != 1:
                raise RuntimeError(f"A.1 senza esattamente un NPZ: {d}")
            os.link(nomi[0], stage / nomi[0].name)
            im, mm = json.loads((d / "info.json").read_text()), json.loads((d / "misure.json").read_text())
            if set(info["tensori"]) & set(im["tensori"]) or set(misure) & set(mm):
                raise RuntimeError("strati/tensori A.1 duplicati")
            info["tensori"].update(im["tensori"])
            misure.update(mm)
        PG.scrivi_json(stage / "info.json", info)
        PG.scrivi_json(stage / "misure.json", misure)
    else:
        (stage / "tensori").mkdir()
        records, visti = [], set()
        for u in unita:
            d = root / "unita" / u.chiave
            if {f.name for f in (d / "tensori").glob("*.npy")} != set(u.dati["file_attesi"]):
                raise RuntimeError(f"insieme tensori incompleto/estraneo: {u.chiave}")
            rr = [json.loads(r) for r in (d / "registro.jsonl").read_text().splitlines() if r]
            chiavi = {(r["strato"], r["famiglia"]) if "strato" in r else r["nome"] for r in rr}
            attese = ({(u.dati["strato"], fam) for fam in ("gate", "up")}
                      if piano["config"]["fase"] == "esperti" else set(u.dati["nomi"]))
            if len(chiavi) != len(rr) or chiavi != attese or chiavi & visti:
                raise RuntimeError(f"registro duplicato/incompleto: {u.chiave}")
            visti.update(chiavi)
            records.extend(rr)
            for f in (d / "tensori").glob("*.npy"):
                if (stage / "tensori" / f.name).exists():
                    raise RuntimeError(f"piano assegnato a due unita': {f.name}")
                os.link(f, stage / "tensori" / f.name)
        records.sort(key=lambda r: (r.get("strato", -1), r.get("famiglia", ""), r.get("nome", "")))
        with open(stage / "registro.jsonl", "x") as f:
            for r in records:
                f.write(json.dumps(r, allow_nan=False) + "\n")
    PG.pubblica_unita(stage, dest, piano["firma"], descrizione, {})
    return dest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fase", choices=("esperti", "densi", "a1"), required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--strati", default="0-39")
    ap.add_argument("--dtype", choices=("float32", "float64"))
    ap.add_argument("--lotto", type=int)
    ap.add_argument("--rango", type=int, default=16)
    ap.add_argument("--pezzo", type=int, default=32)
    ap.add_argument("--fili-lettura", type=int, default=8)
    ap.add_argument("--fili", type=int, default=4)
    ap.add_argument("--hdir", default="@@ROOT:models@@/gguf/tony-tern/hessiane-densi-grande")
    ap.add_argument("--hf-radice")
    ap.add_argument("--algebra", choices=("gpu_sicura", "cpu"), default="gpu_sicura")
    ap.add_argument("--prepara-gpu", action="store_true", default=os.environ.get("FORGIA_PREPARA_GPU") == "1")
    ap.add_argument("--riuso-input", action="store_true",
                    default=os.environ.get("FORGIA_RIUSO_INPUT") == "1",
                    help="un mmap GGUF per worker e cache Hessiane limitata (sperimentale)")
    ap.add_argument("--cache-h-mib", type=int, default=64,
                    help="limite cache CPU delle Hessiane per worker; 0 disabilita la cache")
    ap.add_argument("--solo-piano", action="store_true", help="scrive metadati, nessun contesto GPU")
    a, opzioni = ap.parse_known_args(argv)
    if a.cache_h_mib < 0:
        raise ValueError("--cache-h-mib deve essere >= 0")
    a.dtype = a.dtype or ("float64" if a.fase == "a1" else "float32")
    a.lotto = a.lotto if a.lotto is not None else (16 if a.fase == "esperti" else 8)
    if min(a.lotto, a.rango, a.pezzo, a.fili_lettura) < 1:
        ap.error("lotto/rango/pezzo/fili-lettura devono essere positivi")
    if a.fase == "esperti" and a.dtype != "float32":
        ap.error("gli esperti riusano il percorso collaudato fp32")
    if a.fase == "a1" and a.dtype != "float64":
        ap.error("A.1 riusa l'algebra fp64 collaudata (residui memorizzati fp32)")
    if a.fase == "densi" and a.dtype == "float64" and a.lotto > 8:
        ap.error("fp64 lotto >8 vietato su gfx1151 (hipblasDtrsmBatched)")
    if a.checkpoint:
        a.checkpoint = str(Path(a.checkpoint).resolve(strict=True))
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", PIEGA_ALGEBRA=a.algebra, LEVE_OSPITE=a.gguf)
    PG.guardia_corse()
    PG.guardia_host(*PG.misura_host())
    piano, unita = prepara(a)
    root = Path(a.out).resolve()
    if root.exists() and not (root / "lancio.json").is_file():
        raise RuntimeError("--out esiste ma non appartiene a questa integrazione: scegliere un nome nuovo")
    root.mkdir(parents=True, exist_ok=True)
    p = root / "lancio.json"
    if p.exists():
        if json.loads(p.read_text()) != piano:
            raise RuntimeError("piano della ripresa diverso; nessun file sovrascritto")
    else:
        PG.scrivi_json(p, piano)
    print(json.dumps({"unita": len(unita), "ricetta_sha256": piano["ricetta_sha256"],
                      "piano": str(p)}), flush=True)
    if a.solo_piano:
        return 0
    rc = PG.main(["--piano", str(p), "--out", str(root), "--fili", str(a.fili), *opzioni])
    if rc:
        return rc
    verifica_ingressi(piano)
    # Il lock include anche l'assemblaggio (nessun secondo publisher).
    with PG.lucchetto(QUI / ".pianificatore_gpu.lock"):
        risultato = raccogli(root, piano, unita)
    print(f"PUBBLICATO {risultato}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError, OSError) as e:
        print(f"ERRORE fucina GPU: {e}", file=sys.stderr, flush=True)
        sys.exit(2)

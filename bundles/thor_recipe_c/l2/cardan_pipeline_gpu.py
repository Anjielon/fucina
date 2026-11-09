#!/usr/bin/env python3
"""Orchestrazione nuova delle pieghe: unione disgiunta, UNA patch, nessun rm.

Le valutazioni rimangono nei motori originali; --misura e' esplicito.
Non e' una certificazione dei nuovi percorsi sui dati veri.
"""
from __future__ import annotations

import argparse
import json
import math
import os
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"  # priorita' 0 (21/10): ogni worker inductor apre /dev/kfd e prende una coda SDMA
from pathlib import Path
import subprocess
import sys
import time
import uuid

import pianificatore_gpu as PG

QUI = Path(__file__).resolve().parent
F = Path("@@ROOT:forge@@")
OC = Path("@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--esperti")
    ap.add_argument("--densi")
    ap.add_argument("--out", required=True)
    ap.add_argument("--profilo-esperti")
    ap.add_argument("--profilo-densi")
    ap.add_argument("--rendimenti-esperti")
    ap.add_argument("--rendimenti-densi")
    ap.add_argument("--collaudo-esperti")
    ap.add_argument("--collaudo-densi")
    ap.add_argument("--gpu", default=os.environ.get("FORGIA_GPU", "all"))
    ap.add_argument("--processi-per-gpu", default=os.environ.get("FORGIA_PROCESSI_PER_GPU", "auto"))
    ap.add_argument("--misura", action="store_true")
    ap.add_argument("--riferimento-mmlu")
    ap.add_argument("--cache-dequant", help="cache dequant gia' popolata/verificabile per la coerenza")
    ap.add_argument("--vulkan", default="Vulkan0")
    ap.add_argument("--nome", default="C-GPU")
    a = ap.parse_args(argv)
    if os.environ.get("FORGIA_GPU_SCALABILE") != "1":
        ap.error("attivazione esplicita richiesta: FORGIA_GPU_SCALABILE=1")
    if not (a.esperti or a.densi):
        ap.error("serve almeno uno fra --esperti e --densi")
    if (a.esperti and not a.profilo_esperti) or (a.densi and not a.profilo_densi):
        ap.error("profili misurati obbligatori; calibrare prima con cardan_fucina_gpu.py")
    for fase, checkpoint, profilo, collaudo in (
            ("esperti", a.esperti, a.profilo_esperti, a.collaudo_esperti),
            ("densi", a.densi, a.profilo_densi, a.collaudo_densi)):
        if not checkpoint:
            continue
        if not collaudo:
            ap.error(f"--collaudo-{fase} obbligatorio: il solo profilo di memoria non certifica i byte")
        from cardan_certificato_gpu import verifica
        try:
            verifica(profilo, collaudo)
        except ValueError as e:
            ap.error(f"collaudo {fase}: {e}")
    if a.misura and not a.riferimento_mmlu:
        ap.error("--misura richiede --riferimento-mmlu completo")
    if a.misura and not a.cache_dequant:
        ap.error("--misura richiede --cache-dequant gia' popolata, per evitare caricamenti eager")
    PG.guardia_corse()
    PG.guardia_host(*PG.misura_host())
    os.nice(max(0, 19 - os.getpriority(os.PRIO_PROCESS, 0)))
    if a.esperti and a.densi:
        import torch
        ce = torch.load(a.esperti, mmap=True, map_location="cpu", weights_only=False)
        cd = torch.load(a.densi, mmap=True, map_location="cpu", weights_only=False)
        if ce["meta"]["passo"] != cd["passo"]:
            raise ValueError("checkpoint esperti e densi NON dello stesso passo")
        del ce, cd
    input_files = [x for x in (a.base, a.esperti, a.densi, a.profilo_esperti,
                               a.profilo_densi, a.collaudo_esperti, a.collaudo_densi,
                               a.riferimento_mmlu, a.rendimenti_esperti, a.rendimenti_densi) if x]
    meta = {"input": [PG.identita_file(p) for p in input_files],
            "codice": PG.sha_file(__file__), "gpu": a.gpu, "processi": a.processi_per_gpu,
            "nome": a.nome, "vulkan": a.vulkan, "cache_dequant": a.cache_dequant,
            "riuso_input": os.environ.get("FORGIA_RIUSO_INPUT", "0")}
    root = Path(a.out).resolve()
    # D23: l'ammissione (controllo, mkdir, prima pubblicazione di pipeline.json) e
    # l'intera corsa stanno sotto un lucchetto ESTERNO alla cartella: un concorrente
    # rifiutato non tocca nulla dentro --out.
    root.parent.mkdir(parents=True, exist_ok=True)
    with PG.lucchetto(root.parent / f".{root.name}.pipeline-ammissione.lock"):
        if root.exists() and not (root / "pipeline.json").is_file():
            raise FileExistsError("--out esiste e non appartiene a questa pipeline; usare una cartella nuova")
        root.mkdir(parents=True, exist_ok=True)
        if (root / "pipeline.json").exists():
            if json.loads((root / "pipeline.json").read_text()) != meta:
                raise ValueError("ripresa pipeline con input/ricetta diversi")
        else:
            PG.scrivi_json(root / "pipeline.json", meta)

        def comando(fase, args, env=None):
            PG.guardia_corse()
            PG.guardia_host(*PG.misura_host())
            for m in meta["input"]:
                if PG.identita_file(m["path"]) != m:
                    raise RuntimeError(f"input cambiato: {m['path']}")
            log = root / f"{fase}-{uuid.uuid4().hex}.out"
            t0 = time.monotonic()
            print(f"AVVIO {fase}: {log}", flush=True)
            with open(log, "x") as f:
                p = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, env=env, check=False)
            with open(root / "tempi.jsonl", "a") as f:
                f.write(json.dumps({"fase": fase, "secondi": time.monotonic() - t0,
                                    "rc": p.returncode, "log": str(log), "argv": args}) + "\n")
            if p.returncode:
                raise RuntimeError(f"{fase} fallita ({p.returncode}), risultati conservati: {log}")
            return log

        py = sys.executable
        from cardan_classi_gpu import opzioni_rendimenti
        tensori = []
        with PG.lucchetto(root / ".pipeline.lock"):
            for fase, checkpoint, profilo, lotto in (
                    ("esperti", a.esperti, a.profilo_esperti, 16),
                    ("densi", a.densi, a.profilo_densi, 8)):
                if not checkpoint:
                    continue
                comando(f"piega-{fase}", [py, str(QUI / "cardan_fucina_gpu.py"), "--fase", fase,
                        "--gguf", a.base, "--checkpoint", checkpoint, "--out", str(root / fase),
                        "--lotto", str(lotto), "--dtype", "float32", "--profilo", profilo,
                        "--gpu", a.gpu, "--processi-per-gpu", a.processi_per_gpu,
                        *opzioni_rendimenti(fase, a.rendimenti_esperti, a.rendimenti_densi)])
                tensori.extend(["--tensori", str(root / fase / "pubblicato/tensori")])
            comando("patch-unica", [py, str(QUI / "cardan_patch_gpu.py"), "--base", a.base,
                                   "--out-dir", str(root / "modello"), *tensori])
            g = root / "modello/Thor-T-C-GPU.gguf"
            if not a.misura:
                print(f"PIEGA PUBBLICATA (valutazioni NON eseguite): {g}", flush=True)
                return 0
            B = Path("@@ROOT:models@@/build/thor-c0t/build-c0t/bin")
            teach = "@@ROOT:depot@@/e2e_teacher/q8-maestro"
            klenv = dict(os.environ, LD_LIBRARY_PATH=str(B), GGML_VK_DISABLE_GRAPH_OPTIMIZE="1",
                         GGML_VK_TQ1R_Q8="1")
            klpy = "@@ROOT:models@@/gguf/tony-tern/R026/v12/_lav/kl_realizzata.py"
            for nome, gguf in (("OSPITE", Path(a.base)), ("CARDAN", g)):
                # Una ripresa non sovrascrive logit/JSON di una valutazione precedente.
                ko = root / f"kl-{nome}-{uuid.uuid4().hex}"
                comando(f"kl-{nome}", [py, klpy, "--thor-v2", str(F / "thor_v2"), "--cache", teach,
                         "--dump-bin", "@@ROOT:models@@/gguf/tony-tern/R026/_bin/e2e_dump_logit_p2",
                         "--out", str(ko), "--righe", "24", f"{nome}={gguf}"], klenv)
                v = json.loads((ko / f"kl_{nome}.json").read_text())["kl_media"]
                if not math.isfinite(v) or v <= 0:
                    raise ValueError(f"KL non valida: {nome}")
                comando(f"coerenza-{nome}", [py, str(QUI / "cardan_kd_gpu.py"),
                         "--uscita", str(root / f"coerenza-{nome}-{uuid.uuid4().hex}"),
                         "--gguf", str(gguf), "--nome", nome, "--teacher", teach,
                         "--kl-partenza-motore", str(v), "--soglia-coerenza", "0.03",
                         "--strati", "0", "--rango", "1", "--passi", "0", "--heldout-righe", "24"],
                         dict(os.environ, FORGIA_ALLENA_GPU="1", FORGIA_CACHE_MMAP="1",
                              E2E_CACHE_DEQUANT=a.cache_dequant, E2E_GDN="veloce",
                              E2E_GDN_MODULO="gdn_veloce_corsa_v5", E2E_MOTORE="triton2",
                              E2E_CONV_DIRETTA="1", E2E_SENZA_COESISTENZA="1",
                              E2E_DENSI_PIEGATI="x2", E2E_PIEGA_GEMM="triton", E2E_DENSI_SSM_AB="1"))
            sonda = comando("sonda", ["bash", str(OC / "TRIO/sonda.sh"), str(g), a.nome], klenv)
            import re
            if not re.search(r"giuste \d+/21", sonda.read_text()):
                raise ValueError(f"sonda incompleta: {sonda}")
            mmlu = comando("mmlu", [str(B / "llama-perplexity"), "-m", str(g), "-ngl", "999",
                           "-c", "4096", "-np", "4", "-ub", "512", "-dev", a.vulkan,
                           "--multiple-choice", "--multiple-choice-tasks", "100000",
                           "-f", str(F / "mc_v2/mmlu_00000.bin")], klenv)
            for log in (Path(a.riferimento_mmlu), mmlu):
                text = log.read_text()
                if sum(r.startswith("MCLP\t") for r in text.splitlines()) != 2000 or "Final result:" not in text:
                    raise ValueError(f"MMLU incompleto: {log}")
            comando("mcnemar", [py, str(F / "thor_v2/strumenti/mcnemar_mclp.py"),
                               a.riferimento_mmlu, str(mmlu)])
            comando("kl4", [py, "@@ROOT:models@@/gguf/tony-tern/LEVE_RAFF_PV12P/diagnosi/kl4_appaiato.py",
                            a.riferimento_mmlu, str(mmlu)])
            PG.scrivi_json(root / "valutazione.json", {"gguf": str(g), "mmlu": str(mmlu),
                                                       "sonda": str(sonda), "completa": True})
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

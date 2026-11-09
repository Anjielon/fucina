#!/usr/bin/env python3
"""cattura_router_maestro.py — teacher ROUTER LOGITS for cardan_congiunto_scale.py --router-distilla (router_distilla.py),
26/10/2026 arm B++. Copy of cattura_stati_maestro.py (L2, 25/10) with ONE difference: instead of the residual stream
(--l-out -> l_out-<l>, 2048 wide) it keeps the router logits of the requested layers (--tensore ffn_moe_logits-<l>:
llama.cpp build_moe_ffn, logits = mul_mat(ffn_gate_inp, x), [n_expert = 256, n_tokens], PRE-softmax), every token.
--uscite tutte when layer 39 is requested: the last layer of the qwen35moe graph runs its FFN only on the output rows
(inp_out_ids), exactly as l_out-39 in the L2 cache. Default --salva-tipo f16 (11-bit mantissa: |error| <= 0.008 on a
logit of 16; bf16 would be 8x coarser on the softmax), f32 available.
Same teacher (Q8_0 GGUF of the paper-loss cache piano.json), same tool (e2e_dump_logit, sha256 checked), same rows in the
trainer's order, same first --token tokens, same segment mode: the cache lines up token by token with the L2 state cache
and with the trainer's paper rows (router_distilla.SorgenteRouter checks the ids).
Disk: n_layers x n_token x 256 x 2 B per row (40 x 1024 -> 20 MiB per row, 625 + 4 rows -> ~12.3 GiB f16, ~24.6 f32).

ORIGINAL DOCSTRING (cattura_stati_maestro.py):

The paper-loss teacher cache (CARDAN_FEDELE/teacher/q8-maestro-wt103, e2e_teacher.py) holds only h_nextn (final hidden
after output_norm). This script reruns THE SAME teacher (Q8_0 GGUF of its piano.json) with THE SAME tool
(e2e_dump_logit of its piano.json, sha256 checked) on the first --token tokens of the SAME rows, in the same segment
mode, and keeps the residual stream at the output of the requested layers (--l-out -> tensors l_out-<l>, bf16, all
tokens). Truncating a row to its first n tokens is exact: every layer is causal inside a segment (attention and GDN),
so the first n states do not depend on the tokens after them.

Rows: by default the first --n-righe rows of the TRAIN side in the order the trainer visits them
(allinea_stati.righe_in_ordine(train, --seme), --seme = the trainer's --seme), plus every heldout row (held-out metric).

    python3 cattura_router_maestro.py prepara --uscita DIR [--sorgente CACHE] [--strati 0-39] [--token 1024]
                                             [--n-righe 625] [--seme 20261007] [--righe-blocco 32]
    python3 cattura_router_maestro.py esegui  --uscita DIR [--secco]     # GPU (Vulkan, ~37 GiB): one heavy lane at a time
    python3 cattura_router_maestro.py stato   --uscita DIR
--secco prints the tool command of every block without running it. Resume: sealed blocks are skipped, a partial block
is redone. Disk: n_layers x n_token x 4 KiB per row (40 x 1024 -> 160 MiB per row, 625 rows -> ~98 GiB): checked
against the free space with a margin before every block.
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

import numpy as np

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
sys.path.insert(0, str(QUI / "thor_v2"))
import allinea_stati as AS                                 # noqa: E402

SORGENTE = "@@ROOT:models@@/gguf/tony-tern/CARDAN_FEDELE/teacher/q8-maestro-wt103"
N_EMBD = 2048
N_EXPERT = 256                                             # router width of the qwen35moe teacher (ffn_gate_inp [2048, 256])
NOME = "ffn_moe_logits-{}"
PIANO = "piano_router.json"


def intervallo(s: str) -> list:
    out = []
    for p in s.split(","):
        if "-" in p:
            a, b = p.split("-"); out += list(range(int(a), int(b) + 1))
        elif p.strip():
            out.append(int(p))
    return out


def sha256_file(p: Path, blocco: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(blocco), b""):
            h.update(b)
    return h.hexdigest()


def scrivi_json(p: Path, d) -> None:
    t = p.with_name(p.name + ".tmp")
    t.write_text(json.dumps(d, indent=1))
    os.replace(t, p)


def prepara(a) -> dict:
    import e2e_teacher as T
    src = T.CacheMaestro(a.sorgente)
    pz = src.piano
    if pz["modo"] != "segmenti":
        raise ValueError(f"{a.sorgente}: mode {pz['modo']} (expected segmenti, like the trainer's paper rows)")
    train = [i for b in pz["blocchi"] if b["lato"] == "train" for i in range(b["da"], b["a"])]
    held = [i for b in pz["blocchi"] if b["lato"] == "heldout" for i in range(b["da"], b["a"])]
    righe = {"train": AS.righe_in_ordine(train, a.seme)[: a.n_righe], "heldout": held if a.heldout else []}
    strati = intervallo(a.strati)
    if not strati or min(strati) < 0 or max(strati) > 39:
        raise ValueError(f"--strati {a.strati}: layers 0..39")
    blocchi = []
    for lato, rr in righe.items():
        for b0 in range(0, len(rr), a.righe_blocco):
            blocchi.append({"nome": f"{lato}/blocco_{b0 // a.righe_blocco:04d}", "lato": lato,
                            "righe": rr[b0:b0 + a.righe_blocco]})
    out = Path(a.uscita)
    out.mkdir(parents=True, exist_ok=True)
    piano = {"versione": AS.VERSIONE, "creato": time.strftime("%Y-%m-%dT%H:%M:%S"), "sorgente": str(a.sorgente),
             "sorgente_piano_sha256": hashlib.sha256((Path(a.sorgente) / "piano.json").read_bytes()).hexdigest(),
             "modello": pz["modello"], "strumento": pz["strumento"], "modo": pz["modo"], "salva_tipo": a.salva_tipo,
             "ub": a.ub, "strati": strati, "n_token": a.token, "n_embd": N_EMBD, "n_expert": N_EXPERT,
             "seme_ordine": a.seme, "uscite": "tutte" if 39 in strati else "ultima", "blocchi": blocchi,
             "versione_router": "cattura_router_maestro 1 (26/10/2026)",
             "formato": f"dump/seq_KKKK/ffn_moe_logits-<l>.{a.salva_tipo} [n_token, {N_EXPERT}] = router logits "
                        f"(pre-softmax) of layer l"}
    if (out / PIANO).exists():
        vecchio = json.loads((out / PIANO).read_text())
        if {k: v for k, v in vecchio.items() if k != "creato"} != {k: v for k, v in piano.items() if k != "creato"}:
            raise ValueError(f"{out}/{PIANO} exists with different parameters: use a new --uscita")
        return vecchio
    scrivi_json(out / PIANO, piano)
    return piano


def comando(piano: dict, tmp: Path, n_thread: int) -> list:
    n = piano["n_token"]
    return [piano["strumento"]["percorso"], "logit", "--vulkan", "-m", piano["modello"]["percorso"], "-o", str(tmp / "dump"),
            "-t", str(n_thread), "-c", str(n), "-b", str(n), "-ub", str(min(piano["ub"], n)), "--lunghezza", str(n),
            "--uscite", piano["uscite"], "--no-logit",
            *[x for L in piano["strati"] for x in ("--tensore", NOME.format(L))],
            "--salva-tipo", piano["salva_tipo"], "--segmenti", str(tmp / "inizi.i32"), str(tmp / "ids.i32")]


def esegui(a) -> int:
    import e2e_teacher as T
    from e2e_dump_logit_leggi import Uscita
    out = Path(a.uscita)
    piano = json.loads((out / PIANO).read_text())
    src = T.CacheMaestro(piano["sorgente"])
    if hashlib.sha256((Path(piano["sorgente"]) / "piano.json").read_bytes()).hexdigest() != piano["sorgente_piano_sha256"]:
        raise RuntimeError("the source teacher cache changed after prepara")
    if not a.secco:
        if sha256_file(Path(piano["strumento"]["percorso"])) != piano["strumento"]["sha256"]:
            raise RuntimeError("e2e_dump_logit differs from the one that built the paper-loss cache")
        if Path(piano["modello"]["percorso"]).stat().st_size != piano["modello"]["byte"]:
            raise RuntimeError("teacher GGUF size differs from the source cache piano")
    n = piano["n_token"]
    per_riga = len(piano["strati"]) * n * N_EXPERT * (4 if piano["salva_tipo"] == "f32" else 2)
    for b in piano["blocchi"]:
        fin = out / b["nome"]
        if (fin / "blocco.json").exists():
            continue
        tmp = fin.with_name(fin.name + ".parziale")
        ids = np.stack([np.asarray(src.ids(b["lato"], i), dtype=np.int32)[:n] for i in b["righe"]])
        cu = [src.cu_seqlens(b["lato"], i) for i in b["righe"]]
        inizi = np.zeros((len(b["righe"]), n), np.int32)
        for k, c in enumerate(cu):
            inizi[k, [x for x in c[:-1] if x < n]] = 1
        if not bool((inizi[:, 0] == 1).all()):
            raise RuntimeError(f"{b['nome']}: a row does not start a segment")
        cmd = comando(piano, tmp, a.thread)
        if a.secco:                                     # dry run: nothing written, only the command and the check
            print(f"# {b['nome']}: {len(b['righe'])} rows, {int(inizi.sum())} segment starts in the first {n} tokens")
            print(" ".join(cmd)); continue
        if tmp.exists():                                # a partial block of an interrupted run is redone
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        ids.tofile(tmp / "ids.i32"); inizi.tofile(tmp / "inizi.i32")
        serve = per_riga * len(b["righe"]) + a.margine_gib * 2 ** 30
        libero = shutil.disk_usage(out).free
        if libero < serve:
            raise RuntimeError(f"{b['nome']}: {libero / 2 ** 30:.1f} GiB free, {serve / 2 ** 30:.1f} needed")
        t0 = time.time()
        with open(tmp / "strumento.log", "w") as lg:
            rc = subprocess.run(cmd, stdout=lg, stderr=subprocess.STDOUT, env=T.ambiente_strumento(False)).returncode
        if rc != 0:
            raise RuntimeError(f"{b['nome']}: e2e_dump_logit rc={rc}\n{(tmp / 'strumento.log').read_text()[-1500:]}")
        u = Uscita(tmp / "dump")
        if u.n_sequenze != len(b["righe"]):
            raise RuntimeError(f"{b['nome']}: {u.n_sequenze} sequences, expected {len(b['righe'])}")
        for k in range(u.n_sequenze):
            if not np.array_equal(u.ids(k), ids[k]):
                raise RuntimeError(f"{b['nome']} seq {k}: ids written by the tool differ")
            for L in piano["strati"]:
                t = u.man["sequenze"][k]["tensori"][NOME.format(L)]
                a_ = np.asarray(u.tensore(k, NOME.format(L)))
                if t.get("righe_uguali_a", "token") != "token" or tuple(a_.shape) != (n, N_EXPERT):
                    raise RuntimeError(f"{b['nome']} seq {k} {NOME.format(L)}: not [all tokens, {N_EXPERT}] ({t})")
                if not np.isfinite(a_.astype(np.float32)).all():
                    raise RuntimeError(f"{b['nome']} seq {k} {NOME.format(L)}: non-finite router logits")
        scrivi_json(tmp / "blocco.json", {
            "righe": b["righe"], "lato": b["lato"], "sha256_ids": hashlib.sha256(ids.tobytes()).hexdigest(),
            "sha256_manifesto": sha256_file(tmp / "dump" / "manifesto.json"),
            "strumento_sha256": piano["strumento"]["sha256"], "modello_sha256_dal_piano": piano["modello"]["sha256"],
            "secondi": round(time.time() - t0, 1), "token": int(ids.size), "sigillato": time.strftime("%Y-%m-%dT%H:%M:%S")})
        os.replace(tmp, fin)
        print(f"{b['nome']}: {len(b['righe'])} rows in {time.time() - t0:.0f} s", flush=True)
    return 0


def stato(a) -> int:
    out = Path(a.uscita)
    piano = json.loads((out / PIANO).read_text())
    fatti = [b for b in piano["blocchi"] if (out / b["nome"] / "blocco.json").exists()]
    print(json.dumps({"blocchi": len(piano["blocchi"]), "sigillati": len(fatti),
                      "righe_sigillate": sum(len(b["righe"]) for b in fatti), "strati": piano["strati"],
                      "n_token": piano["n_token"]}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("azione", choices=("prepara", "esegui", "stato"))
    ap.add_argument("--uscita", required=True)
    ap.add_argument("--sorgente", default=SORGENTE)
    ap.add_argument("--strati", default="0-39")
    ap.add_argument("--token", type=int, default=1024)
    ap.add_argument("--n-righe", type=int, default=625)
    ap.add_argument("--seme", type=int, default=20261007, help="the trainer's --seme (row order)")
    ap.add_argument("--heldout", type=int, default=1)
    ap.add_argument("--righe-blocco", type=int, default=32)
    ap.add_argument("--ub", type=int, default=1024)
    ap.add_argument("--thread", type=int, default=4)
    ap.add_argument("--margine-gib", type=float, default=20.0)
    ap.add_argument("--secco", action="store_true")
    ap.add_argument("--salva-tipo", choices=("f16", "f32", "bf16"), default="f16")
    a = ap.parse_args(argv)
    if a.azione == "prepara":
        p = prepara(a)
        print(f"{len(p['blocchi'])} blocks, layers {p['strati'][0]}..{p['strati'][-1]} ({len(p['strati'])}), "
              f"{p['n_token']} tokens per row, {sum(len(b['righe']) for b in p['blocchi'])} rows, "
              f"~{len(p['strati']) * p['n_token'] * N_EXPERT * (4 if p['salva_tipo'] == 'f32' else 2) * sum(len(b['righe']) for b in p['blocchi']) / 2 ** 30:.1f} GiB")
        return 0
    return esegui(a) if a.azione == "esegui" else stato(a)


if __name__ == "__main__":
    sys.exit(main())

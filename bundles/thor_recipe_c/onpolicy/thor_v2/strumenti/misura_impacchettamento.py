#!/usr/bin/env python3
"""Misura su CPU del costo di impacchettamento di UNO strato vero (leva F, export incrementale — 25/9).

Legge i codici di uno strato da una cartella di corsa (SOLA LETTURA), li impacchetta come fa
`valida_finestra_8_12.esporta_gguf_finestra` (gate/up -> TQ1_0R_1024 con
`esporta_schur_esperti.impacchetta_famiglia`, down -> TQ1_0 con `pack_tq1_0_da_d_q` + reshape a
3 assi) e cronometra ogni voce. Non scrive NULLA su disco: la serializzazione del pacchetto si
misura in memoria (BytesIO), come la sua rilettura con sha256 (il costo dell'assemblaggio finale).

Due modi, ognuno in un PROCESSO a parte (per il picco di memoria, ru_maxrss):
  intero  — la chiamata di oggi, su tutti i 256 esperti in un colpo (picco ~ int64 di 268 M trit);
  blocchi — la stessa funzione su blocchi di --blocco esperti, concatenati sull'asse 0.
`--confronta` lancia i due modi e verifica che i byte (sha256) coincidano.

    env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 19 \\
      @@ROOT:venv@@/bin/python3 strumenti/misura_impacchettamento.py --confronta \\
        --cartella @@ROOT:models@@/gguf/tony-tern/v2-conf6h-B1 --strato 0 [--freddo]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI.parent))


def _freddo(p: Path) -> None:
    """Toglie il file dalla cache delle pagine (solo per questo file, niente root)."""
    fd = os.open(p, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def misura(cartella: Path, L: int, modo: str, blocco: int, freddo: bool, verifiche: bool) -> dict:
    import numpy as np
    from esporta_schur_esperti import impacchetta_famiglia, verifica_packer
    from pack_tq1_0_standard import pack_tq1_0_da_d_q
    t: dict = {}
    nomi = [f"{x}_{L:02d}.npy" for x in ("Dg", "Qg", "Du", "Qu", "Dd", "Qd")]
    if freddo:
        for n in nomi:
            _freddo(cartella / n)
    t0 = time.perf_counter()
    Dg, Qg, Du, Qu, Dd, Qd = (np.load(cartella / n) for n in nomi)
    t["carica_npy_s"] = time.perf_counter() - t0
    byte_letti = sum((cartella / n).stat().st_size for n in nomi)

    def famiglia(D, Q):
        if modo == "intero":
            return impacchetta_famiglia(D, Q)
        return np.concatenate([impacchetta_famiglia(D[e:e + blocco], Q[e:e + blocco])
                               for e in range(0, Q.shape[0], blocco)], axis=0)

    def down(D, Q):
        if modo == "intero":
            p = pack_tq1_0_da_d_q(D, Q)
        else:
            p = np.concatenate([pack_tq1_0_da_d_q(D[e:e + blocco], Q[e:e + blocco])
                                for e in range(0, Q.shape[0], blocco)], axis=0)
        return p.reshape(D.shape[0], D.shape[1], -1)

    pacchetti = {}
    for nome, fn, D, Q in (("gate", famiglia, Dg, Qg), ("up", famiglia, Du, Qu), ("down", down, Dd, Qd)):
        t0 = time.perf_counter()
        pacchetti[nome] = fn(D, Q)
        t[f"impacchetta_{nome}_s"] = time.perf_counter() - t0
    if verifiche:
        t0 = time.perf_counter()
        verifica_packer(Dg, Qg, "blk.L.ffn_gate_exps.weight (misura)")
        t["verifica_packer_gate_s"] = time.perf_counter() - t0
        from gguf.quants import TQ1_0 as _TQ1_0_STD
        t0 = time.perf_counter()
        prova = pack_tq1_0_da_d_q(Dd, Qd)
        deq = _TQ1_0_STD.dequantize_blocks(prova.reshape(-1, prova.shape[-1]))
        atteso = (Qd.astype(np.float32) * Dd.repeat(256, axis=-1)).reshape(-1, 256)
        scarto = float(np.abs(deq - atteso).max())
        t["verifica_down_s"] = time.perf_counter() - t0
        t["verifica_down_scarto"] = scarto
        del prova, deq, atteso
    sha = {}
    t0 = time.perf_counter()
    buffers = {}
    for nome, p in pacchetti.items():
        b = io.BytesIO()
        np.save(b, p)
        buffers[nome] = b.getvalue()
        sha[nome] = hashlib.sha256(np.ascontiguousarray(p).view(np.uint8).reshape(-1)).hexdigest()
    t["serializza_e_sha_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    for nome, b in buffers.items():
        a = np.load(io.BytesIO(b))
        assert hashlib.sha256(np.ascontiguousarray(a).view(np.uint8).reshape(-1)).hexdigest() == sha[nome]
    t["rilettura_pacchetto_e_sha_s"] = time.perf_counter() - t0
    return {"modo": modo, "blocco": blocco if modo == "blocchi" else None, "strato": L, "freddo": freddo,
            "byte_npy_letti": byte_letti, "byte_pacchetto": {n: int(p.nbytes) for n, p in pacchetti.items()},
            "forme": {n: list(p.shape) for n, p in pacchetti.items()}, "tempi": t, "sha256": sha,
            "picco_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            "numpy": np.__version__}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cartella", type=Path, default=Path("@@ROOT:models@@/gguf/tony-tern/v2-conf6h-B1"))
    ap.add_argument("--strato", type=int, default=0)
    ap.add_argument("--modo", choices=("intero", "blocchi"), default="blocchi")
    ap.add_argument("--blocco", type=int, default=32)
    ap.add_argument("--freddo", action="store_true", help="npy tolti dalla cache delle pagine prima di leggerli")
    ap.add_argument("--verifiche", action="store_true", help="misura anche verifica_packer e il controllo del down")
    ap.add_argument("--confronta", action="store_true", help="i due modi in due processi + confronto dei byte")
    args = ap.parse_args()
    if not args.confronta:
        print(json.dumps(misura(args.cartella, args.strato, args.modo, args.blocco, args.freddo, args.verifiche)))
        return 0
    esiti = {}
    for modo, verif in (("blocchi", False), ("intero", True)):
        cmd = [sys.executable, __file__, "--cartella", str(args.cartella), "--strato", str(args.strato),
               "--modo", modo, "--blocco", str(args.blocco)] + (["--freddo"] if args.freddo else []) \
              + (["--verifiche"] if verif else [])
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout + r.stderr)
            return 1
        esiti[modo] = json.loads(r.stdout.strip().splitlines()[-1])
        e = esiti[modo]
        tt = e["tempi"]
        print(f"── modo {modo}{' (blocco ' + str(args.blocco) + ')' if modo == 'blocchi' else ''} · strato "
              f"{args.strato} · {'FREDDO' if args.freddo else 'cache delle pagine com e'} · picco RSS "
              f"{e['picco_rss_gib']:.2f} GiB")
        for k, v in tt.items():
            print(f"   {k:32s} {v:9.3f}")
        tot = sum(tt[k] for k in ("carica_npy_s", "impacchetta_gate_s", "impacchetta_up_s", "impacchetta_down_s"))
        print(f"   {'TOTALE carica+impacchetta':32s} {tot:9.3f} s")
    uguali = esiti["blocchi"]["sha256"] == esiti["intero"]["sha256"]
    print(f"byte identici fra i due modi (sha256 di gate, up, down): {'✅' if uguali else '⛔'} "
          f"{esiti['intero']['sha256']}")
    print(json.dumps(esiti))
    return 0 if uguali else 1


if __name__ == "__main__":
    raise SystemExit(main())

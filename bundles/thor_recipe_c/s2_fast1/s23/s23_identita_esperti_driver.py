#!/usr/bin/env python3
"""VELOCE 14/10 — driver del test di identita' degli ESPERTI (S3): esegue risolvi_esperti_gu e risolvi_esperti_down del
codice dato (--codice: quello congelato della corsa o quello veloce) su CPU con dati REALI di uno strato: pesi del maestro
bf16 (safetensors), statistiche per esperto della cattura di S1 (Gq, C, conta, peso, Hdown; meta' A e B), H d'ingresso di
strato per sorgente, byte della partenza GGUF. Il termine incrociato del down (Md) non e' nella cattura: si usa Hd (stessa
forma; serve solo a far girare entrambi i codici sugli stessi numeri). Salva i byte nuovi e i registri (senza i tempi)."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import os

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))   # 14/10: mai piu' di 2 fili (la corsa usa la CPU)

OC = Path("@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C")
S1 = OC / "S1_FLUSSO_C_1013"
PARTENZA = OC / "PTQ2" / "Thor-T-C-p_bil1jarqdrq.gguf"
HF = Path("@@ROOT:models@@/gguf/thor-src-hf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codice", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dt", default="float64")
    ap.add_argument("--strato", type=int, default=2)
    ap.add_argument("--esperti", type=int, default=32)
    ap.add_argument("--lotto", type=int, default=16)
    a = ap.parse_args()
    cod = Path(a.codice)
    for p in (cod / "s23", cod / "lib", cod / "s1b", cod / "rif"):
        sys.path.insert(0, str(p))
    import s23_algebra as AL
    import s23_passata as SP
    import s1b_comune as K
    import s1_comune as SC
    import cardan_piega as CP
    from safetensors import safe_open
    dt = getattr(torch, a.dt)
    dev = torch.device("cpu")
    L, Esub, Hd, I = a.strato, a.esperti, 2048, 512
    gguf = K.gp._import_gguf(None)
    tens = {t.name: t for t in gguf.GGUFReader(str(PARTENZA), "r").tensors}
    idx = json.loads((HF / "model.safetensors.index.json").read_text())["weight_map"]
    pre = f"model.language_model.layers.{L}.mlp.experts."
    with safe_open(str(HF / idx[pre + "gate_up_proj"]), "pt") as f:
        Wgu = f.get_slice(pre + "gate_up_proj")[:Esub].float()
    with safe_open(str(HF / idx[pre + "down_proj"]), "pt") as f:
        Wd = f.get_slice(pre + "down_proj")[:Esub].float()
    assert tuple(Wgu.shape) == (Esub, 2 * I, Hd) and tuple(Wd.shape) == (Esub, Hd, I), (Wgu.shape, Wd.shape)
    dl = Path(tempfile.mkdtemp(prefix="s23_id_esperti_"))
    acc3 = SP.AccDensi(Hd, dev)
    adn = SP.AccDown(Esub, I, dev)
    for h in "AB":
        d = S1 / "cattura" / h / f"strato_{L:02d}"
        info = json.loads((d / "info.json").read_text())
        Mx = torch.from_numpy(np.load(d / "ffn_Mx.npy").astype(np.float64))
        for s, nt in info["token"].items():
            if nt:
                acc3.H[(h, s)] = torch.from_numpy(SC.spacchetta(np.load(d / f"ffn_Hq_{s}.npy"), Hd)) * nt
                acc3.M[(h, s)] = Mx * nt
                acc3.N[(h, s)] = int(nt)

        def leggi(nome):
            v = np.load(d / f"esperti_{nome}.npy", mmap_mode="r")[:Esub].astype(np.float32)
            sc = np.load(d / f"esperti_{nome}_scala.npy")[:Esub].astype(np.float32)
            return v * sc.reshape((-1,) + (1,) * (v.ndim - 1))
        g = dl / f"gu_{h}"
        g.mkdir(parents=True)
        np.save(g / "Gq.npy", leggi("Gq"))
        np.save(g / "C.npy", leggi("C"))
        conta = np.load(d / "esperti_conta.npy")[:Esub]
        peso = np.load(d / "esperti_peso.npy")[:Esub]
        np.save(g / "conta.npy", conta)
        np.save(g / "peso.npy", peso)
        Hdn = torch.from_numpy(np.stack([SC.spacchetta(x, I, np.float32) for x in leggi("Hdown")]))
        adn.Hd[h] = Hdn.float()
        adn.Md[h] = Hdn.float().clone()
        adn.conta[h] = torch.from_numpy(conta.astype(np.int64))
        adn.peso[h] = torch.from_numpy(peso.astype(np.float64))
    raw = {n: np.array(tens[n].data)[:Esub] for n in (f"blk.{L}.ffn_gate_exps.weight", f"blk.{L}.ffn_up_exps.weight",
                                                      f"blk.{L}.ffn_down_exps.weight")}

    def codifica(T, Sc, tid, C):
        Tn = T.to("cpu").numpy().astype(np.int8)
        S16 = Sc.to("cpu", torch.float16).numpy()
        b = K.gp.encode_rows(Tn, S16, tid)
        tp, sp_ = K.gp.decode_rows(b, tid, C)
        assert np.array_equal(tp, Tn) and np.array_equal(sp_.view(np.uint16), S16.view(np.uint16))
        return b

    def decodifica(raw_rows, tid, C):
        t, d_ = K.gp.decode_rows(np.ascontiguousarray(raw_rows), tid, C)
        return torch.from_numpy(t.astype(np.int8)), torch.from_numpy(d_.astype(np.float64))
    JQL = AL.jarq_lotto_cache_da(CP) if hasattr(AL, "jarq_lotto_cache_da") else CP.jarq_lotto
    gd = {"emin": 0.9, "emax": 1.1, "peso_kw": 3.5, "relativa": True}
    nuovi = {}
    es = list(range(Esub))
    rgu = SP.risolvi_esperti_gu(L, dl, acc3, Wgu, nuovi, lambda *x: None, codifica, decodifica, lambda n: raw[n], es,
                                [0.0, 0.5], 8000.0, a.lotto, JQL, dev, dt, Esub, I, Hd, gd)
    rdn = SP.risolvi_esperti_down(L, adn, Wd, nuovi, lambda *x: None, codifica, decodifica, lambda n: raw[n], es,
                                  [0.0, 0.5], 8000.0, a.lotto, JQL, dev, dt, Esub, I, Hd, gd)
    for r in (rgu, rdn):
        r.pop("secondi", None)
    torch.save({"nuovi": {k: np.asarray(v) for k, v in nuovi.items()}, "gu": json.dumps(rgu, sort_keys=True, default=str),
                "down": json.dumps(rdn, sort_keys=True, default=str), "jarq": getattr(JQL, "__qualname__", str(JQL))}, a.out)
    print(f"gate/up {rgu['accettati_gate']}/{rgu['accettati_up']} down {rdn['accettati']} su {Esub}; nuovi {sorted(nuovi)}")


if __name__ == "__main__":
    main()

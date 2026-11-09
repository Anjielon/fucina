#!/usr/bin/env python3
"""test_e1_esporta.py — e1_esporta_residui (dai latenti salvati, senza studente) == esportazione del driver AL BYTE.
Residui con forme di Thor (attn_q 2048->8192, attn_qkv 2048->8192, ssm_out 4096->2048), gamma != 0, 1 e 2 piani."""
import json, os, sys, tempfile
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import residui_ternari as RT  # noqa: E402
import e1_esporta_residui as EX  # noqa: E402


def prova(piani: int) -> None:
    cfg = RT.ConfigResidui(rango=32, piani=piani, strati=(12, 15), famiglie=("attn_q", "attn_qkv", "ssm_out"))
    forme = {"blk.15.attn_q": (8192, 2048), "blk.12.attn_qkv": (8192, 2048), "blk.12.ssm_out": (2048, 4096)}
    mods = {}
    for i, (n, (fu, de)) in enumerate(forme.items()):
        m = RT.ResiduoTernario(n, fu, de, cfg, seme=i)
        with torch.no_grad():
            m.gamma.copy_(torch.randn(fu, generator=torch.Generator().manual_seed(9 + i)) * 0.01)
        mods[n] = m
    t_ref, info_ref = RT.tensori_gguf(mods)
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
        RT.salva_latenti(mods, Path(d) / "residui_migliore.pt", {"passo": 7})
        r = EX.esporta(Path(d) / "residui_migliore.pt", d)
        z = np.load(Path(d) / "residui_tq1.npz")
        info = json.load(open(Path(d) / "residui.json"))
        assert sorted(z.files) == sorted(t_ref), (z.files, list(t_ref))
        for k in t_ref:
            assert np.array_equal(z[k], t_ref[k]), k
        assert info["tensori"] == info_ref and info["passo"] == 7 and info["strati"] == [12, 15]
    print(f"ok piani {piani}: {r['n_tensori']} tensori, {r['byte']} byte, identici all'esportazione del driver")


if __name__ == "__main__":
    prova(1)
    prova(2)
    print("TUTTE LE 2 PROVE OK")

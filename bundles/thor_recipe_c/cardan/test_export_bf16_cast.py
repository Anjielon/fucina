#!/usr/bin/env python3
"""Audit 2610.07853 «Lost in the bf16 Cast» (8/10): l'export dei residui ternari (lra/lrb TQ1_0, gamma fuso nella
scala fp16 di B) NON deve perdere gli aggiornamenti piccoli. Solo CPU, autosufficiente.
  1. regime lr bassa: latenti spostati di pochissimo (codici che cambiano per un soffio sulla soglia) -> i codici nel
     byte TQ1_0 sono ESATTAMENTE quelli ricalcolati in fp32 dal latente (niente ritorno al valore di partenza);
  2. gamma minuscolo (scale fp16 subnormali, come al passo 30-500 delle corse: gamma ~1e-3): export == avanti al bit;
  3. nessun cast bf16 nel percorso di export (residui_ternari.matrici_file / tensori_gguf / byte_tq1_0).
Uso: python3 test_export_bf16_cast.py   (oppure pytest)
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import torch

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import residui_ternari as RT   # noqa: E402


def _residuo(fuori=96, dentro=512, rango=64, seme=7):
    cfg = RT.ConfigResidui(rango=rango, strati=(0,), famiglie=("ffn_up_shexp",) if "ffn_up_shexp" in RT.FAMIGLIE
                           else ("attn_qkv",))
    return RT.ResiduoTernario("blk.0.x", fuori, dentro, cfg, seme=seme)


def _codici_file(res):
    t, _ = RT.tensori_gguf({"blk.0.x": res})
    return (RT.deq_tq1_0(t["blk.0.x_lra.weight"], res.dentro), RT.deq_tq1_0(t["blk.0.x_lrb.weight"], RT.RP))


def test_aggiornamenti_minimi_sopravvivono():
    res = _residuo()
    with torch.no_grad():
        res.gamma.fill_(1e-3)
    A0, _ = _codici_file(res)
    g = torch.Generator().manual_seed(3)
    cambiati = 0
    for passo in range(1, 6):
        with torch.no_grad():                        # passo «lr bassa»: 1e-4 relativo, ~40 volte sotto 1 ULP bf16 (2^-8)
            res.A.add_(torch.randn(res.A.shape, generator=g) * 1e-4 * res.A.abs().mean())
            res.B.add_(torch.randn(res.B.shape, generator=g) * 1e-4 * res.B.abs().mean())
        (Ap, Bp), = RT.matrici_file(res)
        Af, Bf = _codici_file(res)
        assert np.array_equal(Af, Ap) and np.array_equal(Bf, Bp), passo
        cambiati += int((np.sign(Af) != np.sign(A0)).sum())
    # controllo che la prova sia sensibile: un passaggio da bf16 riporterebbe (parte de) i codici cambiati indietro
    Ab = RT.matrici_file(_bf16(res))[0][0]
    print(f"  codici A cambiati dal passo 0: {cambiati}; diversi se i latenti passano da bf16: "
          f"{int((np.sign(Ab) != np.sign(Ap)).sum())}")
    assert cambiati > 0, "perturbazione troppo piccola: la prova non e' sensibile"
    assert int((np.sign(Ab) != np.sign(Ap)).sum()) > 0, "il bf16 non cambia nulla: la prova non e' sensibile"


def _bf16(res):
    c = _residuo(res.fuori, res.dentro, res.cfg.rango)
    c.load_state_dict({k: v.to(torch.bfloat16).float() for k, v in res.state_dict().items()})
    return c


def test_gamma_minuscolo_subnormale():
    res = _residuo()
    with torch.no_grad():
        res.gamma.copy_(torch.logspace(-8, -2, res.fuori) * torch.sign(torch.randn(res.fuori)))
    Aq, Bq = (t.detach() for t in res.pesi_q())
    (Ap, Bp), = RT.matrici_file(res)
    r = res.cfg.rango
    assert np.array_equal(Ap[:r], Aq.numpy()) and np.array_equal(Bp[:, :r], Bq.numpy())
    Af, Bf = _codici_file(res)
    assert np.array_equal(Af, Ap) and np.array_equal(Bf, Bp)
    assert (np.abs(Bp[Bp != 0]) < 6.1035e-5).any(), "nessuna scala subnormale: la prova non copre il caso"


def test_nessun_bf16_nell_export():
    for f in (RT.matrici_file, RT.tensori_gguf, RT.byte_tq1_0, RT.piani_g, RT.piano_ternario_g, RT.ricostruisci_g):
        src = inspect.getsource(f)
        assert "bfloat16" not in src and "bf16" not in src, f.__name__


if __name__ == "__main__":
    torch.set_num_threads(4)
    for k, f in sorted(globals().items()):
        if k.startswith("test_") and callable(f):
            f(); print("ok", k, flush=True)
    print("TUTTE OK")

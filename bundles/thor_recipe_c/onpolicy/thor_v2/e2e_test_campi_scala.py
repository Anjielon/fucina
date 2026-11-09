#!/usr/bin/env python3
"""Prova SOLO CPU di `e2e_campi_scala.py` su tensori VERI di un GGUF forgiato.

Uso (GPU occupata dalla forgia: mai toccarla):
    CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 19 \
        python3 e2e_test_campi_scala.py [GGUF] [STRATO]

Cosa prova, contro riferimenti che NON sono questo file:
  T1 trit torch == `_unpack_quarter` del codec di casa (TQ1_0R_1024) e == la
     dequantizzazione di gguf-py divisa per d (TQ1_0), su un campione di blocchi;
  T2 mappa blocco -> (esperto, riga, gruppo): D (E,out,ng) e trit ricostruiscono
     `dequant_gguf_torch.dequant_np_slice` al bit (con la de-rotazione per gate/up);
  T3 riscrittura: scale perturbate dell'1% (fp16), byte dei codici identici,
     d riletti identici al bit, e la dequantizzazione della LIBRERIA sui byte nuovi
     = trit x d_nuove (TQ1_0) / codec di casa (TQ1_0R_1024).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

QUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, QUI)
sys.path.insert(0, os.path.join(QUI, "..", "kernel-rotazione"))
sys.path.insert(0, "@@ROOT:models@@/build/llamacpp-tq1-k1024/gguf-py")
torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "2")))

from gguf import GGUFReader  # noqa: E402
from gguf.quants import TQ1_0 as GQ_TQ1_0  # noqa: E402
from tq1_0r_1024_codec import _unpack_quarter, dequantize_blocks as deq_r1024, fwht1024_f32, S1024  # noqa: E402

import e2e_campi_scala as C  # noqa: E402
from dequant_gguf_torch import dequant_np_slice  # noqa: E402

GGUF = sys.argv[1] if len(sys.argv) > 1 else "@@ROOT:models@@/gguf/tony-tern/Thor-T-ibrido-0-19-ThorT-veloce.gguf"
STRATO = int(sys.argv[2]) if len(sys.argv) > 2 else 19
N_CAMPIONE = 16384
ESPERTI_T2 = (0, 97, 255)


def _tensore(r, nome):
    t = [x for x in r.tensors if x.name == nome]
    if not t:
        raise SystemExit(f"{nome} assente in {GGUF}")
    return t[0]


def prova(t, rng) -> None:
    tipo = t.tensor_type.name
    L = C.LAYOUT[tipo]
    raw = np.asarray(t.data)                       # (E, out, byte_per_riga)
    E, n_out = int(t.shape[2]), int(t.shape[1])
    n_in = int(t.shape[0])
    B = C.blocchi(raw, tipo)
    print(f"── {t.name} · {tipo} · E={E} out={n_out} in={n_in} · blocchi {B.shape[0]}")

    # T1 — trit torch contro i riferimenti
    idx = np.sort(rng.choice(B.shape[0], size=min(N_CAMPIONE, B.shape[0]), replace=False))
    Tt = C.trit_torch(torch.from_numpy(B[idx]), tipo).numpy()
    if tipo == "TQ1_0R_1024":
        rif = np.concatenate([_unpack_quarter(B[idx, q * 48:(q + 1) * 48], B[idx, 192 + q * 4:192 + (q + 1) * 4])
                              for q in range(4)], axis=1)
    else:
        d = C.leggi_scale(B[idx], tipo).astype(np.float32)
        deq = GQ_TQ1_0.dequantize_blocks(B[idx].copy())
        vivi = d != 0
        rif = np.zeros_like(deq)
        rif[vivi] = deq[vivi] / d[vivi, None]
        rif = np.rint(rif)
    ok1 = np.array_equal(Tt.astype(np.float32), rif.astype(np.float32))
    print(f"   T1 trit torch == riferimento su {len(idx)} blocchi: {ok1}")

    # T2 — mappa blocco -> (e, r, g) contro dequant_np_slice
    D = C.leggi_scale(raw, tipo, E, n_out).astype(np.float32)           # (E, out, ng)
    ok2 = True
    for e in ESPERTI_T2:
        Be = C.blocchi(raw[e], tipo)
        Te = C.trit_torch(torch.from_numpy(Be), tipo).numpy().astype(np.float32).reshape(n_out, n_in)
        grezzo = (Te.reshape(n_out, -1, L["gruppo"]) * D[e][:, :, None]).reshape(n_out, n_in).astype(np.float32)
        if tipo == "TQ1_0R_1024":
            nb = n_in // 1024
            W = (fwht1024_f32(grezzo.reshape(n_out * nb, 1024)) * S1024[None, :]).reshape(n_out, n_in)
        else:
            W = grezzo
        W_rif = dequant_np_slice(t, e)
        uguale = np.array_equal(W, W_rif)
        ok2 &= uguale
        print(f"   T2 esperto {e}: D x trit (ricostruito) == dequant_np_slice al bit: {uguale}"
              + ("" if uguale else f" (max |diff| {np.abs(W - W_rif).max():.3e})"))

    # T3 — riscrittura dei soli campi d
    d_vecchie = C.leggi_scale(raw, tipo)
    fattore = (1.0 + 0.01 * rng.uniform(-1, 1, size=d_vecchie.shape)).astype(np.float32)
    d_nuove = (d_vecchie.astype(np.float32) * fattore).astype(np.float16)
    nuovo = C.scrivi_scale(raw, tipo, d_nuove)
    esito = C.verifica_patch(raw, nuovo, tipo, d_nuove)
    Bn = C.blocchi(nuovo, tipo)
    sub = Bn[idx].copy()
    Tn = C.trit_torch(torch.from_numpy(sub), tipo).numpy().astype(np.float32)
    dn = C.leggi_scale(sub, tipo).astype(np.float32).reshape(len(idx), L["scale_blocco"])
    grezzo_n = (Tn.reshape(len(idx), L["scale_blocco"], L["gruppo"]) * dn[:, :, None]).reshape(len(idx), -1)
    if tipo == "TQ1_0R_1024":
        atteso = (fwht1024_f32(grezzo_n) * S1024[None, :]).astype(np.float32)
        libreria = deq_r1024(sub)
    else:
        atteso = grezzo_n
        libreria = GQ_TQ1_0.dequantize_blocks(sub)
    ok3 = np.array_equal(atteso, libreria)
    print(f"   T3 {esito} · dequantizzazione dei byte nuovi == trit x d_nuove al bit: {ok3}")
    if not (ok1 and ok2 and ok3):
        raise SystemExit(f"⛔ {t.name}: prova FALLITA (T1={ok1} T2={ok2} T3={ok3})")


def main() -> int:
    rng = np.random.default_rng(20260925)
    r = GGUFReader(GGUF)
    print(f"GGUF {GGUF} · strato {STRATO}")
    for fam in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
        t = _tensore(r, f"blk.{STRATO}.{fam}.weight")
        if t.tensor_type.name not in C.LAYOUT:
            print(f"── {t.name}: tipo {t.tensor_type.name} non ternario, saltato")
            continue
        prova(t, rng)
    print("✅ tutte le prove passate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

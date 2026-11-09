#!/usr/bin/env python3
"""Dequantizzazione UNIFICATA di ogni tipo di tensore dei GGUF Thor → float32
in ordine numpy/HF (…, out_rows, in_feat). Primo mattone della forgia
sequenziale SchurQuant (piano 22/9: la passata del maestro sul modello GIA'
quantizzato richiede di far girare in torch i pesi quantizzati, tipo per tipo).

Tipi presenti in Thor-v20-downaggr3 (misurato 22/9 04:35): F32 370 · Q4_K 111 ·
TQ1_0 82 · Q8_0 70 · TQ1_0R_1024 61 · IQ4_XS 29 · Q6_K 16 · Q4_0 8 · Q5_K 6.
gguf-py sa dequantizzare tutti tranne TQ1_0R_1024 (nostro formato: usa il
codec di casa, `dequantize_blocks`) e F16/F32 (gia' float: passthrough).

Convenzione forme (verificata sul file): `t.shape` e' in ordine GGUF
[ne0=in_feat, ne1=out_rows, (ne2=esperti)]; `t.data` e' in ordine numpy
(esperti, out_rows, byte_per_riga). L'uscita ha forma data.shape[:-1]+(in_feat,).

Uso: `dequant_np(t)` su un `ReaderTensor`, oppure `dequant_np_slice(t, e)` per
UN SOLO esperto di un tensore 3D (evita 1 GiB di float per tensore: la RAM
host e' la risorsa scarsa di questa macchina).
Test CPU: `python3 dequant_gguf_torch.py` — correlazione col bf16 vero per un
tensore di ogni tipo, esperti compresi (scopre anche il layout del fused
gate_up_proj HF).
"""
from __future__ import annotations
import os
import sys
import numpy as np

_QUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "@@ROOT:models@@/build/llamacpp-tq1-k1024/gguf-py")
sys.path.insert(0, os.path.join(_QUI, "..", "kernel-rotazione"))
from gguf import quants as GQ  # noqa: E402

_PASSTHROUGH = {"F32", "F16"}


def in_feat_di(t) -> int:
    return int(t.shape[0])


# 28/9 (ramo densi-ternari): un TQ1_0R_1024 grande (token_embd/output della variante "B", 248.320 x 2048) si
# dequantizza a pezzi di righe: `dequantize_blocks` fa 4-5 temporanei float32 grandi quanto l'uscita (~2 GiB l'uno a
# vocabolario intero). Blocco per blocco l'aritmetica e' la stessa (nessuna somma fra blocchi): stesso bit.
PEZZO_R1024 = 1 << 24                      # elementi per pezzo (64 MiB float32); sotto, la via di sempre


def _dequant_raw(raw: np.ndarray, t) -> np.ndarray:
    """raw: byte (…, byte_per_riga) del tipo di t → float32 (…, in_feat)."""
    nome = t.tensor_type.name
    if nome in _PASSTHROUGH:
        return np.asarray(raw, dtype=np.float32)
    if nome == "TQ1_0R_1024":
        from tq1_0r_1024_codec import dequantize_blocks
        lead = raw.shape[:-1]
        C = in_feat_di(t)
        righe = int(np.prod(lead)) if lead else 1
        if righe * C <= PEZZO_R1024:
            return dequantize_blocks(raw).reshape(*lead, C)
        piatto = raw.reshape(righe, raw.shape[-1])
        out = np.empty((righe, C), dtype=np.float32)
        passo = max(1, PEZZO_R1024 // C)
        for a in range(0, righe, passo):
            b = min(righe, a + passo)
            out[a:b] = dequantize_blocks(piatto[a:b]).reshape(b - a, C)
        return out.reshape(*lead, C)
    out = GQ.dequantize(np.ascontiguousarray(raw), t.tensor_type)
    return np.asarray(out, dtype=np.float32)


def dequant_np(t) -> np.ndarray:
    raw = np.asarray(t.data)
    out = _dequant_raw(raw, t)
    attesa = tuple(raw.shape[:-1]) + (in_feat_di(t),) if t.tensor_type.name not in _PASSTHROUGH else tuple(raw.shape)
    assert out.shape == attesa, f"{t.name}: forma {out.shape} ≠ attesa {attesa}"
    return out


def dequant_np_slice(t, e: int) -> np.ndarray:
    """Un solo esperto e di un tensore 3D (…_exps): (out_rows, in_feat)."""
    raw = np.asarray(t.data)
    assert raw.ndim == 3, f"{t.name} non e' 3D"
    return _dequant_raw(raw[e:e + 1], t)[0]


def dequant_torch(t, device="cpu", dtype=None):
    import torch
    x = torch.from_numpy(dequant_np(t))
    return x.to(device, dtype) if dtype is not None else x.to(device)


# ─────────────────────────────────────────────────────────────────────────────
def _test():
    import json
    PERM = dict(np.load(os.path.join(_QUI, "perm_gguf_da_hf.npz")))
    from gguf import GGUFReader
    from safetensors import safe_open
    HF = "@@ROOT:models@@/gguf/thor-src-hf"
    idx = json.load(open(f"{HF}/model.safetensors.index.json"))["weight_map"]
    r = GGUFReader(os.environ.get("THOR_GGUF", "@@ROOT:models@@/gguf/tony-tern/Thor-v20-downaggr3.gguf"))
    tens = {t.name: t for t in r.tensors}

    def hf(k):
        with safe_open(f"{HF}/{idx[k]}", "pt") as f:
            return f.get_tensor(k).float().numpy()

    def hf_slice_e(k, e):
        with safe_open(f"{HF}/{idx[k]}", "pt") as f:
            return f.get_slice(k)[e:e + 1][0].float().numpy()

    def corr(a, b):
        a = a.ravel(); b = b.ravel(); n = min(a.size, b.size)
        return float(np.corrcoef(a[:n], b[:n])[0, 1])

    P = "model.language_model.layers."
    # densi: (nome gguf, chiave HF)
    densi = [
        ("blk.0.attn_gate.weight",      P + "0.linear_attn.in_proj_z.weight"),    # Q8_0
        ("blk.0.attn_qkv.weight",       P + "0.linear_attn.in_proj_qkv.weight"),  # Q6_K
        ("blk.0.ffn_gate_shexp.weight", P + "0.mlp.shared_expert.gate_proj.weight"),  # TQ1_0R_1024
        ("blk.11.attn_k.weight",        P + "11.self_attn.k_proj.weight"),        # Q4_K
        ("blk.3.attn_q.weight",         P + "3.self_attn.q_proj.weight"),
        ("blk.0.ffn_down_shexp.weight", P + "0.mlp.shared_expert.down_proj.weight"),
        ("blk.0.attn_norm.weight",      P + "0.input_layernorm.weight"),          # F32
    ]
    print(f"{'tensore':32s} {'tipo':12s} {'forma':18s} {'corr':>7s}")
    ok = True
    for g, k in densi:
        if g not in tens or k not in idx:
            print(f"{g:32s} (manca: gguf={g in tens} hf={k in idx})"); continue
        t = tens[g]; W = dequant_np(t); V = hf(k)
        # ⛔ le righe HF vanno riordinate nell'ordine del convertitore (teste
        # di attn_gate/attn_qkv): mappa bit-esatta in perm_gguf_da_hf.npz
        fam = g.split(".", 2)[2]
        if fam in PERM:
            V = V[PERM[fam]]
        c = corr(W, V)
        soglia = 0.55 if t.tensor_type.name.startswith("TQ1") else 0.98
        seg = "✅" if c >= soglia else "⛔"
        ok &= c >= soglia
        print(f"{seg} {g:30s} {t.tensor_type.name:12s} {str(W.shape):18s} {c:7.4f}  (bf16 {V.shape})")

    # esperti: layout del fused gate_up_proj HF, scoperto per correlazione
    print("\n── esperti (un solo esperto per volta, e=5) ──")
    e = 5
    gu = hf_slice_e(P + "0.mlp.experts.gate_up_proj", e)      # (?, ?)
    dn = hf_slice_e(P + "0.mlp.experts.down_proj", e)
    print(f"HF gate_up_proj[e]: {gu.shape} · down_proj[e]: {dn.shape}")
    Wg = dequant_np_slice(tens["blk.0.ffn_gate_exps.weight"], e)   # TQ1_0 (512, 2048)
    Wu = dequant_np_slice(tens["blk.0.ffn_up_exps.weight"], e)
    Wd = dequant_np_slice(tens["blk.0.ffn_down_exps.weight"], e)   # IQ4_XS (2048, 512)
    print(f"GGUF gate[e] {Wg.shape} up[e] {Wu.shape} down[e] {Wd.shape}")
    ipotesi = {}
    if gu.shape[0] == 2 * Wg.shape[0]:            # (1024, 2048): righe = [gate; up]
        ipotesi["gate=righe[:512]"] = corr(Wg, gu[:Wg.shape[0]]); ipotesi["up=righe[512:]"] = corr(Wu, gu[Wg.shape[0]:])
    if gu.shape[1] == 2 * Wg.shape[0]:            # (2048, 1024): colonne = [gate | up], trasposto
        ipotesi["gate=col[:512].T"] = corr(Wg, gu[:, :Wg.shape[0]].T); ipotesi["up=col[512:].T"] = corr(Wu, gu[:, Wg.shape[0]:].T)
    for k, v in ipotesi.items():
        print(f"  {k:20s} corr={v:7.4f}")
    cd = corr(Wd, dn) if dn.shape == Wd.shape else corr(Wd, dn.T)
    print(f"  down (IQ4_XS){'' if dn.shape == Wd.shape else ' vs HF.T'} corr={cd:7.4f}")
    ok &= cd >= 0.98 and max(ipotesi.values()) >= 0.55
    print("\n" + ("✅ dequant coerente col bf16 su tutti i tipi provati" if ok else "⛔ ALMENO UN TIPO NON TORNA — non usare per la passata quantizzata"))
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _test() else 1)

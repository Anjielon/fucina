#!/usr/bin/env python3
"""Loads ONE Qwen3_5MoeDecoderLayer block with the weights TAKEN FROM the quantized
GGUF (dequantized with the complete map, verified 2026-09-22), ready for a real
forward — the missing building block for computing X_q (the input "on the
quantized prefix") at any layer, for the teacher-reconstruction component of
SchurQuant.

Uses `fucina.ternary.gguf_dequant` (9 GGUF types → torch, verified against bf16) +
`data/gguf_hf_map.json` / `data/gguf_hf_perm.npz` (EXACT correspondence for every
layer tensor, verified by exact match — not by threshold).

⛔ The fused experts (`ffn_gate_exps`+`ffn_up_exps` → `experts.gate_up_proj`,
`ffn_down_exps` → `experts.down_proj`) stay 3D tensors: they are dequantized
one expert at a time with `dequant_np_slice` so as not to hold 256×(…) in RAM
all at once (house rule: host RAM is the scarce resource)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from fucina.extdeps import ENV_GGUF_PY, add_path, data_path  # noqa: E402
add_path(ENV_GGUF_PY)   # gguf-py with the rotated ternary types (TQ1_0R_1024); see fucina.extdeps
from gguf import GGUFReader  # noqa: E402

from fucina.ternary.gguf_dequant import dequant_np, dequant_np_slice  # noqa: E402

HERE = Path(__file__).resolve().parent
MAP = json.loads((HERE / "data" / "gguf_hf_map.json").read_text())
PERM = dict(np.load(HERE / "data" / "gguf_hf_perm.npz"))

_TRANSFORMS = {
    "identity": lambda v: v,
    "plus_one": lambda v: v + 1.0,
}
_TRANSFORMS_1D_WITH_PERM = {"neg_exp": lambda v: -np.exp(v)}


def _type_of(L: int, num_full_every: int, layers_full: set[int]) -> str:
    return "full" if L in layers_full else "linear"


def _inverse_neg_exp(v: np.ndarray) -> np.ndarray:
    """GGUF stores ssm_a = -exp(A_log)  =>  A_log = log(-ssm_a).

    ⛔⛔⛔ 2026-09-24 (audit B9): this used to be `-np.log(-v)`, i.e. -A_log.
    With the flipped sign the memory decay of the linear attention becomes
    exp(-A_log) instead of exp(A_log): the 30 linear layers out of 40 of the
    "quantized" torch branch computed a model different from llama.cpp
    (layer 0: 91% error per token versus llama.cpp, against a true
    quantization error of 7%). Found by comparing weight by weight with bf16:
    err_rel 2.000, cosine -1.000. Invariant: test_carica_strato_fedelta.py."""
    return np.log(-v)


def _apply(entry: dict, G_np: np.ndarray) -> np.ndarray:
    """From G_np (dequantized value, GGUF order) to HF: the inverse of what is
    documented in the map builder's relation() (mappa_completa_gguf_hf.py)."""
    rel = entry["rel"]
    if rel == "identity":
        out = G_np
    elif rel == "transform":
        f_inv = {"plus_one": lambda v: v - 1.0, "neg_exp": _inverse_neg_exp}[entry["f"]]
        out = f_inv(G_np)
    elif rel in ("perm_rows", "perm_1d"):
        perm = PERM[entry["perm_npz"]]
        inv = np.empty_like(perm)
        inv[perm] = np.arange(len(perm))
        out = G_np[inv] if rel == "perm_rows" else G_np[inv]
        if entry.get("f"):
            out = {"neg_exp": _inverse_neg_exp}[entry["f"]](out)
    elif rel == "perm_columns":
        perm = PERM[entry["perm_npz"]]
        inv = np.empty_like(perm)
        inv[perm] = np.arange(len(perm))
        out = G_np[:, inv]
    else:
        raise ValueError(f"unknown relation: {rel}")
    if entry.get("transposed"):
        out = out.T
    # ⛔ found during acceptance testing on 2026-09-22: two tensors have a UNIT dimension
    # that the map search (mappa_completa_gguf_hf.py) dropped in order to compare
    # element counts (`slice_` declares it), but that must be PUT BACK to load
    # into the real HF module via load_state_dict.
    if entry.get("slice_") == "squeeze":
        out = out[:, None, :]                 # (channels, 4) -> (channels, 1, 4): depthwise conv1d
    if entry["hf"].endswith("shared_expert_gate.weight") and out.ndim == 1:
        out = out[None, :]                    # (2048,) -> (1, 2048)
    return out


def weights_layer_from_gguf(gguf_path: str, L: int) -> dict[str, torch.Tensor]:
    """Returns {hf_name_relative_to_layer: tensor} ready for load_state_dict
    on a Qwen3_5MoeDecoderLayer, with the values DEQUANTIZED from the GGUF."""
    r = GGUFReader(gguf_path)
    tens = {t.name: t for t in r.tensors}
    pref = f"blk.{L}."
    layers_full = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39}
    type_ = "full" if L in layers_full else "linear"
    map_ = MAP[type_]

    out: dict[str, torch.Tensor] = {}
    for fam, entry in map_.items():
        name_g = pref + fam
        if name_g not in tens:
            continue
        t = tens[name_g]
        hf_name = entry["hf"]
        if entry.get("per_expert"):
            # 3D tensor: dequantize one expert at a time, apply the same
            # relation (identity or slice of the fused gate_up_proj) per expert
            n_exp = int(t.shape[-1])
            rows = []
            for e in range(n_exp):
                Ge = dequant_np_slice(t, e)
                rows.append(_apply(entry, Ge))
            value = np.stack(rows, axis=0)
            if entry.get("slice_"):
                key = hf_name  # fused: assembled afterwards, see below
                out.setdefault(f"__fused__{key}", {})[entry["slice_"]] = value
                continue
        else:
            G_np = dequant_np(t)
            value = _apply(entry, G_np)
        out[hf_name] = torch.from_numpy(np.ascontiguousarray(value)).to(torch.float32)

    # reassemble the fused gate_up_proj (rows[:h] + rows[h:] concatenated along axis 1)
    for key in [k for k in out if isinstance(out.get(k), dict)]:
        pieces = out.pop(key)
        order = sorted(pieces, key=lambda f: int(f.split("[")[1].split(":")[0] or 0))
        hf_name = key.replace("__fused__", "")
        out[hf_name] = torch.from_numpy(np.concatenate([pieces[k] for k in order], axis=1)).to(torch.float32)
    return out


def _test():
    from transformers import AutoConfig, Qwen3_5MoeForConditionalGeneration
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding
    from accelerate import init_empty_weights

    HF = data_path("thor-src-hf")
    GGUF = data_path("Thor-v20-downaggr3.gguf")
    TEACHER = Path(data_path("teacher-pass-thor"))
    L = 10

    cfg = AutoConfig.from_pretrained(HF)
    with init_empty_weights():
        m = Qwen3_5MoeForConditionalGeneration(cfg)
    block = m.model.language_model.layers[L]

    weights = weights_layer_from_gguf(GGUF, L)
    missing, extra = block.load_state_dict(weights, strict=False, assign=True)
    missing_real = [k for k in missing if not k.startswith("mlp.experts.")]  # the experts are always all loaded
    print(f"missing keys (non-experts): {missing_real[:10]}{'...' if len(missing_real) > 10 else ''}")
    block = block.to(torch.float32)

    h = torch.from_numpy(np.load(TEACHER / f"confine_{L:02d}.npy")[:2]).float()  # 2 sequences, for speed
    pos = torch.arange(h.shape[1]).unsqueeze(0).expand(2, -1)
    rot = Qwen3_5MoeTextRotaryEmbedding(config=cfg.text_config)
    pe = rot(h, pos)
    with torch.no_grad():
        y = block(h, position_ids=pos, position_embeddings=pe)
    y = y[0] if isinstance(y, tuple) else y

    y_expected = torch.from_numpy(np.load(TEACHER / f"confine_{L+1:02d}.npy")[:2]).float()
    corr = float(torch.corrcoef(torch.stack([y.flatten(), y_expected.flatten()]))[0, 1])
    err_rel = float((y - y_expected).abs().mean() / y_expected.abs().mean())
    print(f"forward layer {L} FROM QUANTIZED GGUF vs confine_{L+1:02d} (real bf16): "
          f"corr={corr:.4f}, mean relative error={err_rel:.3f}")
    # expected: high correlation (the quantized model approximates the real model)
    # BUT NOT identical (it is quantized) - the threshold is "it works", not "it is perfect"
    ok = corr > 0.9 and len(missing_real) == 0
    print("✅ loading from GGUF plausible" if ok else "⛔ something does not add up")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _test() else 1)

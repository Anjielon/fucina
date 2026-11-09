#!/usr/bin/env python3
"""Carica UN blocco Qwen3_5MoeDecoderLayer coi pesi PRESI DAL GGUF quantizzato
(dequantizzati con la mappa completa verificata stanotte), pronto per un
forward vero — il mattone che manca per calcolare X_q (l'ingresso "sul
prefisso quantizzato") a qualunque strato, per il componente teacher-
reconstruction di SchurQuant.

Usa `dequant_gguf_torch.py` (9 tipi GGUF → torch, verificato contro bf16) +
`mappa_completa_gguf_hf.json`/`_perm.npz` (corrispondenza ESATTA per ogni
tensore di strato, verificata per match esatto — non per soglia).

⛔ Gli esperti fusi (`ffn_gate_exps`+`ffn_up_exps` → `experts.gate_up_proj`,
`ffn_down_exps` → `experts.down_proj`) restano tensori 3D: si dequantizzano
un esperto alla volta con `dequant_np_slice` per non tenere 256×(…) in RAM
tutto insieme (regola di casa: RAM host e' la risorsa scarsa)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "@@ROOT:models@@/build/llamacpp-tq1-k1024/gguf-py")
from gguf import GGUFReader  # noqa: E402

from dequant_gguf_torch import dequant_np, dequant_np_slice  # noqa: E402

QUI = Path(__file__).resolve().parent
MAPPA = json.loads((QUI / "mappa_completa_gguf_hf.json").read_text())
PERM = dict(np.load(QUI / "mappa_completa_gguf_hf_perm.npz"))

_TRASFORME = {
    "identita": lambda v: v,
    "piu_uno": lambda v: v + 1.0,
}
_TRASFORME_1D_CON_PERM = {"neg_exp": lambda v: -np.exp(v)}


def _tipo_di(L: int, num_full_ogni: int, strati_full: set[int]) -> str:
    return "full" if L in strati_full else "linear"


def _inversa_neg_exp(v: np.ndarray) -> np.ndarray:
    """GGUF salva ssm_a = -exp(A_log)  =>  A_log = log(-ssm_a).

    ⛔⛔⛔ 24/9/2026 18:40 (audit B9): qui c'era `-np.log(-v)`, cioe' -A_log.
    Col segno invertito il decadimento della memoria dell'attenzione lineare
    diventa exp(-A_log) al posto di exp(A_log): i 30 strati lineari su 40 del
    ramo torch "quantizzato" calcolavano un modello diverso da llama.cpp
    (strato 0: 91% di errore per token contro llama.cpp, a fronte del 7% di
    errore vero di quantizzazione). Trovato confrontando peso per peso col bf16:
    err_rel 2,000, coseno -1,000. Invariante: test_carica_strato_fedelta.py."""
    return np.log(-v)


def _applica(voce: dict, G_np: np.ndarray) -> np.ndarray:
    """Da G_np (valore dequantizzato, ordine GGUF) a HF: inversa di quanto
    documentato in mappa_completa_gguf_hf.py::relazione()."""
    rel = voce["rel"]
    if rel == "identita":
        out = G_np
    elif rel == "trasformata":
        f_inv = {"piu_uno": lambda v: v - 1.0, "neg_exp": _inversa_neg_exp}[voce["f"]]
        out = f_inv(G_np)
    elif rel in ("perm_righe", "perm_1d"):
        perm = PERM[voce["perm_npz"]]
        inv = np.empty_like(perm)
        inv[perm] = np.arange(len(perm))
        out = G_np[inv] if rel == "perm_righe" else G_np[inv]
        if voce.get("f"):
            out = {"neg_exp": _inversa_neg_exp}[voce["f"]](out)
    elif rel == "perm_colonne":
        perm = PERM[voce["perm_npz"]]
        inv = np.empty_like(perm)
        inv[perm] = np.arange(len(perm))
        out = G_np[:, inv]
    else:
        raise ValueError(f"relazione sconosciuta: {rel}")
    if voce.get("trasposta"):
        out = out.T
    # ⛔ trovato al collaudo 22/9 07:02: due tensori hanno una dimensione UNITARIA
    # che la ricerca della mappa (mappa_completa_gguf_hf.py) toglieva per poter
    # confrontare il conteggio di elementi (`fetta` lo dichiara), ma che va
    # RIMESSA per caricare nel modulo HF vero via load_state_dict.
    if voce.get("fetta") == "squeeze":
        out = out[:, None, :]                 # (canali, 4) -> (canali, 1, 4): conv1d depthwise
    if voce["hf"].endswith("shared_expert_gate.weight") and out.ndim == 1:
        out = out[None, :]                    # (2048,) -> (1, 2048)
    return out


def pesi_strato_da_gguf(gguf_path: str, L: int) -> dict[str, torch.Tensor]:
    """Ritorna {nome_hf_relativo_al_layer: tensore} pronto per load_state_dict
    su un Qwen3_5MoeDecoderLayer, coi valori DEQUANTIZZATI dal GGUF."""
    r = GGUFReader(gguf_path)
    tens = {t.name: t for t in r.tensors}
    pref = f"blk.{L}."
    strati_full = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39}
    tipo = "full" if L in strati_full else "linear"
    mappa = MAPPA[tipo]

    out: dict[str, torch.Tensor] = {}
    for fam, voce in mappa.items():
        nome_g = pref + fam
        if nome_g not in tens:
            continue
        t = tens[nome_g]
        hf_nome = voce["hf"]
        if voce.get("per_esperto"):
            # tensore 3D: dequantizza un esperto alla volta, applica la stessa
            # relazione (identita' o fetta del fused gate_up_proj) per esperto
            n_exp = int(t.shape[-1])
            righe = []
            for e in range(n_exp):
                Ge = dequant_np_slice(t, e)
                righe.append(_applica(voce, Ge))
            valore = np.stack(righe, axis=0)
            if voce.get("fetta"):
                chiave = hf_nome  # fused: si assembla dopo, vedi sotto
                out.setdefault(f"__fused__{chiave}", {})[voce["fetta"]] = valore
                continue
        else:
            G_np = dequant_np(t)
            valore = _applica(voce, G_np)
        out[hf_nome] = torch.from_numpy(np.ascontiguousarray(valore)).to(torch.float32)

    # riassembla i fused gate_up_proj (righe[:h] + righe[h:] concatenate sull'asse 1)
    for chiave in [k for k in out if isinstance(out.get(k), dict)]:
        pezzi = out.pop(chiave)
        ordine = sorted(pezzi, key=lambda f: int(f.split("[")[1].split(":")[0] or 0))
        hf_nome = chiave.replace("__fused__", "")
        out[hf_nome] = torch.from_numpy(np.concatenate([pezzi[k] for k in ordine], axis=1)).to(torch.float32)
    return out


def _test():
    from transformers import AutoConfig, Qwen3_5MoeForConditionalGeneration
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding
    from accelerate import init_empty_weights

    HF = "@@ROOT:models@@/gguf/thor-src-hf"
    GGUF = "@@ROOT:models@@/gguf/tony-tern/Thor-v20-downaggr3.gguf"
    TEACHER = Path("@@ROOT:models@@/gguf/tony-tern/teacher-pass-thor")
    L = 10

    cfg = AutoConfig.from_pretrained(HF)
    with init_empty_weights():
        m = Qwen3_5MoeForConditionalGeneration(cfg)
    blocco = m.model.language_model.layers[L]

    pesi = pesi_strato_da_gguf(GGUF, L)
    manca, extra = blocco.load_state_dict(pesi, strict=False, assign=True)
    manca_vera = [k for k in manca if not k.startswith("mlp.experts.")]  # gli esperti li carichiamo sempre tutti
    print(f"chiavi mancanti (non-esperti): {manca_vera[:10]}{'...' if len(manca_vera) > 10 else ''}")
    blocco = blocco.to(torch.float32)

    h = torch.from_numpy(np.load(TEACHER / f"confine_{L:02d}.npy")[:2]).float()  # 2 sequenze, per velocita'
    pos = torch.arange(h.shape[1]).unsqueeze(0).expand(2, -1)
    rot = Qwen3_5MoeTextRotaryEmbedding(config=cfg.text_config)
    pe = rot(h, pos)
    with torch.no_grad():
        y = blocco(h, position_ids=pos, position_embeddings=pe)
    y = y[0] if isinstance(y, tuple) else y

    y_atteso = torch.from_numpy(np.load(TEACHER / f"confine_{L+1:02d}.npy")[:2]).float()
    corr = float(torch.corrcoef(torch.stack([y.flatten(), y_atteso.flatten()]))[0, 1])
    err_rel = float((y - y_atteso).abs().mean() / y_atteso.abs().mean())
    print(f"forward strato {L} DA GGUF QUANTIZZATO vs confine_{L+1:02d} (bf16 vero): "
          f"corr={corr:.4f}, errore relativo medio={err_rel:.3f}")
    # atteso: alta correlazione (il modello quantizzato approssima il modello vero)
    # MA NON identico (e' quantizzato) - la soglia e' "funziona", non "e' perfetto"
    ok = corr > 0.9 and len(manca_vera) == 0
    print("✅ caricamento da GGUF plausibile" if ok else "⛔ qualcosa non torna")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _test() else 1)

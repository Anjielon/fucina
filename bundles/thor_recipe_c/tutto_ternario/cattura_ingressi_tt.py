#!/usr/bin/env python3
"""28/9 tutto_ternario — Hessiane degli INGRESSI VERI dei tre densi mai ternarizzati con una ricetta pesata:
  attn_output    (strati full):  ingresso di self_attn.o_proj         = uscita dell'attenzione gia' moltiplicata dal gate
  ssm_out        (strati GDN):   ingresso di linear_attn.out_proj     = uscita GDN normata (RMSNormGated) e col gate silu(z)
  ffn_down_shexp (tutti):        ingresso di mlp.shared_expert.down_proj = act(gate_proj x) * up_proj x
sul bf16 pieno (thor-src-hf), STESSO corpus e STESSI campioni delle Hessiane del 20/9 (calib_auto2_pulito.txt,
96 seq x 2048 = 196.608, passate da 16), stesso caricatore a strati (cattura_hessiane_down_thor.carica_strato, sola
lettura), H = X^T X / N in fp64 poi fp32 (stessa convenzione di hessiane-attn/, hessiane/).

Aggancio verificato a ogni strato sulla prima passata: F.linear(ingresso catturato, peso del modulo) == uscita del
modulo (stessa aritmetica bf16: differenza 0 attesa), e negativo: con l'ingresso di un'altra passata non combacia.
Ordine GGUF: ssm_out ha le COLONNE permutate (mappa perm, 'linear.ssm_out.weight': GGUF[:, j] = HF[:, perm[j]]),
quindi la sua Hessiana si salva come H[perm][:, perm]; o_proj e down_proj sono identita'.
Uscita: $TT_HESS_OUT/{attn_output,ssm_out,ffn_down_shexp}/H_LL.npy + controlli.json
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig
from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding

sys.path.insert(0, "@@ROOT:forge@@/thor")
from cattura_hessiane_down_thor import Sorgente, carica_strato, PREFISSO  # noqa: E402  sola lettura

SRC = Path("@@ROOT:models@@/gguf/thor-src-hf")
OUT = Path(os.getenv("TT_HESS_OUT", "@@ROOT:depot@@/tutto_ternario/hessiane"))
CORPUS = Path("@@ROOT:forge@@/calib_auto2_pulito.txt")
L_SEQ, BATCH_SEQ = 2048, 16
PERM = dict(np.load("@@ROOT:forge@@/thor_v2/mappa_completa_gguf_hf_perm.npz"))


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    assert torch.cuda.is_available(), "serve la GPU"
    for f in ("attn_output", "ssm_out", "ffn_down_shexp"):
        (OUT / f).mkdir(parents=True, exist_ok=True)
    cfg = AutoConfig.from_pretrained(str(SRC)).text_config
    cfg._attn_implementation = "sdpa"
    src = Sorgente(SRC)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(SRC))
    tutti = tok(CORPUS.read_text(encoding="utf-8", errors="replace"), return_tensors="pt",
                add_special_tokens=False).input_ids[0]
    n_batch = max(1, (tutti.numel() // L_SEQ) // BATCH_SEQ)
    n_seq = n_batch * BATCH_SEQ
    ids = tutti[: n_seq * L_SEQ].reshape(n_seq, L_SEQ)
    log(f"corpus {CORPUS.name}: {tutti.numel()} token -> {n_seq} seq x {L_SEQ} = {n_seq * L_SEQ} campioni in {n_batch} passate")
    emb = src.prendi(f"{PREFISSO}embed_tokens.weight")
    hs = []
    with torch.no_grad():
        for b in range(n_batch):
            hs.append(F.embedding(ids[b * BATCH_SEQ:(b + 1) * BATCH_SEQ].to("cuda"), emb).to(torch.bfloat16))
    del emb; gc.collect(); torch.cuda.empty_cache()
    rot = Qwen3_5MoeTextRotaryEmbedding(cfg).to("cuda")
    pos = torch.arange(L_SEQ, device="cuda").unsqueeze(0)
    with torch.no_grad():
        pos_emb = rot(hs[0], pos)
    mk = dict(config=cfg, inputs_embeds=hs[0], attention_mask=None, past_key_values=None, position_ids=pos)
    maschere = {"full_attention": create_causal_mask(**mk), "linear_attention": create_recurrent_attention_mask(**mk)}
    controlli = {}
    t0 = time.time()
    for L in range(cfg.num_hidden_layers):
        ts = time.time()
        strato = carica_strato(src, cfg, L)
        full = cfg.layer_types[L] == "full_attention"
        mod_att = strato.self_attn.o_proj if full else strato.linear_attn.out_proj
        mod_down = strato.mlp.shared_expert.down_proj
        fam_att = "attn_output" if full else "ssm_out"
        S = {fam_att: None, "ffn_down_shexp": None}
        n = 0
        for b in range(n_batch):
            cat = {}
            def pre(nome):
                def f(_m, inp):
                    cat[nome + "_in"] = inp[0].detach()
                return f
            def post(nome):
                def f(_m, _inp, out):
                    cat[nome + "_out"] = out.detach()
                return f
            hk = [mod_att.register_forward_pre_hook(pre("att")), mod_att.register_forward_hook(post("att")),
                  mod_down.register_forward_pre_hook(pre("down")), mod_down.register_forward_hook(post("down"))]
            with torch.no_grad():
                o = strato(hidden_states=hs[b], position_embeddings=pos_emb,
                           attention_mask=maschere[cfg.layer_types[L]], position_ids=pos)
                hs[b] = o[0] if isinstance(o, tuple) else o
            for h in hk:
                h.remove()
            if b == 0:   # aggancio: l'ingresso catturato ridà ESATTAMENTE l'uscita del modulo
                c = {}
                for k, m in (("att", mod_att), ("down", mod_down)):
                    x, y = cat[k + "_in"], cat[k + "_out"]
                    y2 = F.linear(x, m.weight, m.bias)
                    c[k] = {"max_abs_diff": float((y2.float() - y.float()).abs().max()),
                            "forma_ingresso": list(x.shape), "in_features": m.in_features}
                    assert x.shape[-1] == m.in_features
                controlli[L] = c
                primo = {k: cat[k + "_in"] for k in ("att", "down")}
            elif b == 1:  # negativo: l'ingresso di un'ALTRA passata non ridà l'uscita di questa
                for k, m in (("att", mod_att), ("down", mod_down)):
                    y2 = F.linear(primo[k], m.weight, m.bias)
                    controlli[L][k]["negativo_max_abs_diff"] = float((y2.float() - cat[k + "_out"].float()).abs().max())
                del primo
            for k, fam in (("att", fam_att), ("down", "ffn_down_shexp")):
                X = cat[k + "_in"].reshape(-1, cat[k + "_in"].shape[-1]).double()
                S[fam] = X.T @ X if S[fam] is None else S[fam] + X.T @ X
            n += cat["att_in"].reshape(-1, cat["att_in"].shape[-1]).shape[0]
            del cat
        for fam, M in S.items():
            H = (M / n).float().cpu().numpy().astype(np.float32)
            if fam == "ssm_out":
                p = PERM["linear.ssm_out.weight"]
                H = np.ascontiguousarray(H[p][:, p])
            np.save(OUT / fam / f"H_{L:02d}.npy", H)
            ev = np.linalg.eigvalsh(H.astype(np.float64))
            controlli[L][fam] = {"n": n, "traccia": float(np.trace(H)), "autoval_min": float(ev[0]),
                                 "autoval_max": float(ev[-1]), "asimmetria": float(np.abs(H - H.T).max())}
        log(f"strato {L:2d} ({'full' if full else 'GDN '}) {n} campioni · aggancio att {controlli[L]['att']['max_abs_diff']:.2e}"
            f" (neg {controlli[L]['att'].get('negativo_max_abs_diff', float('nan')):.2e}) down "
            f"{controlli[L]['down']['max_abs_diff']:.2e} (neg {controlli[L]['down'].get('negativo_max_abs_diff', float('nan')):.2e})"
            f" · {time.time() - ts:.1f}s")
        del strato, S; gc.collect(); torch.cuda.empty_cache()
        json.dump(controlli, open(OUT / "controlli.json", "w"), indent=1)
    log(f"FATTO in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

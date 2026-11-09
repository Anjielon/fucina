#!/usr/bin/env python3
"""CATTURA HESSIANA DEDICATA per attn_gate/attn_qkv/attn_q di THOR (20/9/2026).

Nato da un buco trovato leggendo la notte scorsa (KERNEL_DEROTAZIONE_VULKAN_
2026-09-19_INTEGRAZIONE.md, sezione "Thor-v2 completo"): le Hessiane esistenti
in `@@ROOT:models@@/gguf/tony-tern/hessiane/H_NN.npy` sono calibrate sull'uscita
di `post_attention_layernorm` (l'ingresso della FFN — ESATTA per
ffn_gate_shexp/ffn_up_shexp). Ma attn_gate/attn_qkv/attn_q leggono l'uscita
di `input_layernorm` (la norma PRIMA dell'attenzione/GDN, stesso flusso
residuo dello strato ma un punto diverso) — H_NN.npy la' e' solo "prestata"
(RICERCA_ROTAZIONE_SU_TONY.md la usa per MISURARE l'effetto della rotazione,
mai dichiarata adeguata per GPTQ).

Verificato leggendo il forward di Qwen3_5MoeDecoderLayer (transformers):
    hidden_states = self.input_layernorm(hidden_states)
    # -> self.linear_attn(...) o self.self_attn(...)   [attn_gate/qkv/q leggono QUESTO]
    hidden_states = residual + hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)             # [ffn_*_shexp legge QUESTO]

Copia STRUTTURALE di `cattura_hessiane_down_thor.py` (stesso caricamento
pread/fadvise, stesso corpus, stesse maschere GDN/full-attn) ma SENZA
l'instrumentazione per-esperto di down (qui serve solo H = XᵀX su UN punto,
2048x2048, come le H esistenti — nessuna nuova dimensione).

Uscita: @@ROOT:models@@/gguf/tony-tern/hessiane-attn/H_{L:02d}.npy (40 file,
2048x2048 float32), SEPARATA dalle H esistenti (che restano invariate e
continuano a servire ffn_gate_shexp/ffn_up_shexp).
"""
from __future__ import annotations
import gc, os, sys, time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig
from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cattura_hessiane_down_thor import Sorgente, carica_strato, PREFISSO  # noqa: E402  riuso, non duplicazione

SRC = Path(os.getenv("THOR_SORGENTE_HF", "@@ROOT:models@@/gguf/thor-src-hf"))
OUT = Path(os.getenv("THOR_HESS_ATTN_OUT", "@@ROOT:models@@/gguf/tony-tern/hessiane-attn"))
SOLO = int(os.getenv("THOR_SOLO_STRATI", "0"))
TETTO_GIB = float(os.getenv("THOR_TETTO_VRAM_GIB", "35"))
CORPUS = Path(os.getenv("THOR_CORPUS", "@@ROOT:forge@@/calib_auto2_pulito.txt"))
N_SEQ = int(os.getenv("THOR_N_SEQ", "0"))
L_SEQ = int(os.getenv("THOR_L_SEQ", "2048"))
BATCH_SEQ = int(os.getenv("THOR_BATCH_SEQ", "16"))


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def vram_gib() -> float:
    return torch.cuda.memory_reserved() / 2**30


def controlla_tetto(dove: str):
    v = vram_gib()
    if v > TETTO_GIB:
        raise RuntimeError(f"TETTO VRAM SUPERATO a {dove}: {v:.2f} GiB > {TETTO_GIB} GiB — mi fermo io")
    return v


def main():
    if not torch.cuda.is_available():
        sys.exit("nessuna GPU: questa cattura vive in VRAM, non ha senso su CPU")
    OUT.mkdir(parents=True, exist_ok=True)

    cfg_tutto = AutoConfig.from_pretrained(str(SRC))
    cfg = cfg_tutto.text_config
    cfg._attn_implementation = "sdpa"
    n_strati = cfg.num_hidden_layers if not SOLO else min(SOLO, cfg.num_hidden_layers)
    log(f"architettura: {cfg.num_hidden_layers} strati · hidden {cfg.hidden_size} · "
        f"layer_types (primi 6) {cfg.layer_types[:6]}")

    src = Sorgente(SRC)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(SRC))
    testo = CORPUS.read_text(encoding="utf-8", errors="replace")
    tutti = tok(testo, return_tensors="pt", add_special_tokens=False).input_ids[0]
    n_seq_tot = N_SEQ if N_SEQ > 0 else (tutti.numel() // L_SEQ)
    if tutti.numel() < n_seq_tot * L_SEQ:
        raise ValueError(f"corpus {CORPUS}: {tutti.numel()} token, servono {n_seq_tot * L_SEQ}")
    n_batch = max(1, n_seq_tot // BATCH_SEQ)
    n_seq = n_batch * BATCH_SEQ
    ids_tutte = tutti[: n_seq * L_SEQ].reshape(n_seq, L_SEQ)
    n_tok = L_SEQ
    log(f"corpus {CORPUS.name}: {tutti.numel()} token -> {n_seq} seq x {L_SEQ} = "
        f"{n_seq*L_SEQ} campioni/strato in {n_batch} passate da {BATCH_SEQ}")
    log(f"catturo i primi {n_strati} strati (input_layernorm) · tetto VRAM {TETTO_GIB} GiB · uscita {OUT}")

    emb = src.prendi(f"{PREFISSO}embed_tokens.weight")
    controlla_tetto("embedding")
    hs = []
    with torch.no_grad():
        for b in range(n_batch):
            idb = ids_tutte[b*BATCH_SEQ:(b+1)*BATCH_SEQ].to("cuda")
            hs.append(torch.nn.functional.embedding(idb, emb).to(torch.bfloat16))
    del emb
    gc.collect(); torch.cuda.empty_cache()
    log(f"embedding fatto ({n_batch} passate) e liberato · VRAM {vram_gib():.2f} GiB")

    rot = Qwen3_5MoeTextRotaryEmbedding(cfg).to("cuda")
    pos = torch.arange(n_tok, device="cuda").unsqueeze(0)
    with torch.no_grad():
        pos_emb = rot(hs[0], pos)

    mask_kwargs = dict(config=cfg, inputs_embeds=hs[0], attention_mask=None,
                       past_key_values=None, position_ids=pos)
    maschere = {
        "full_attention": create_causal_mask(**mask_kwargs),
        "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
    }

    t0 = time.time()
    for L in range(n_strati):
        t_str = time.time()
        strato = carica_strato(src, cfg, L)
        v = controlla_tetto(f"strato {L} caricato")

        H_sum = torch.zeros(cfg.hidden_size, cfg.hidden_size, dtype=torch.float64, device="cuda")
        n_tot = 0
        for b in range(n_batch):
            catturato = {}

            def gancio(_mod, _inp, out):
                catturato["x"] = out.detach()

            # ⭐ QUESTO e' il punto diverso da cattura_hessiane_down_thor.py:
            # input_layernorm (pre-attn/GDN), non post_attention_layernorm (pre-FFN).
            h_reg = strato.input_layernorm.register_forward_hook(gancio)
            with torch.no_grad():
                out = strato(hidden_states=hs[b], position_embeddings=pos_emb,
                             attention_mask=maschere[cfg.layer_types[L]], position_ids=pos)
                hs[b] = out[0] if isinstance(out, tuple) else out
            h_reg.remove()
            v = max(v, controlla_tetto(f"strato {L} passata {b} dopo il forward"))

            X = catturato["x"].reshape(-1, cfg.hidden_size).double()
            H_sum += X.T @ X
            n_tot += X.shape[0]
            del catturato, X

        H = (H_sum / n_tot).float()
        np.save(OUT / f"H_{L:02d}.npy", H.cpu().numpy().astype(np.float32))

        ev = torch.linalg.eigvalsh(H.double())
        n_neg = int((ev < -1e-3).sum())
        ev_pos = ev[ev > 0].flip(0)
        alfa = float(-np.polyfit(np.log(np.arange(1, len(ev_pos) + 1)),
                                 np.log((ev_pos / ev_pos[0]).cpu().numpy()), 1)[0])
        n_nan = int(torch.isnan(H).sum() + torch.isinf(H).sum())

        ev_min = float(ev.min())
        log(f"  strato {L:2d}/{n_strati - 1} ({cfg.layer_types[L]}) · {n_tot} campioni · "
            f"autoval.min {ev_min:.4g} · nan/inf {n_nan} · autoval.negativi {n_neg} · "
            f"alfa {alfa:.3f} · VRAM picco {v:.1f} GiB -> {vram_gib():.2f} GiB · {time.time() - t_str:.1f}s")

        del strato, H, H_sum, ev, ev_pos
        gc.collect(); torch.cuda.empty_cache()

    log(f"FATTO: {n_strati} strati in {OUT} · {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

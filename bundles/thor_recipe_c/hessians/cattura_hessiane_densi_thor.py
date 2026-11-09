#!/usr/bin/env python3
"""CATTURA HESSIANE DEI 149 TENSORI DENSI (attn + ffn-shared) in UN SOLO
passaggio, per confrontare corpus PICCOLO (210.883 token, 27/8 e 20/9) vs
corpus GRANDE (2.489.584 token, notte 20->21/9) — TASK del 21/9.

Fonde in un'unica passata cio' che `cattura_hessiane_attn_thor.py` e la parte
gate/up di `cattura_hessiane_down_thor.py` facevano SEPARATAMENTE (due forward
sull'intero corpus): qui un solo hook doppio per strato (input_layernorm ->
H_attn, post_attention_layernorm -> H_ffn), a costo dimezzato. NESSUNA
instrumentazione per-esperto di down (fuori scopo per questo task: i 149
tensori sono SOLO attn_gate/attn_qkv/attn_q/ffn_gate_shexp/ffn_up_shexp).

Uscita: OUT/H_attn_{L:02d}.npy e OUT/H_ffn_{L:02d}.npy (2048x2048 float32),
per i primi THOR_SOLO_STRATI strati (0 = tutti i 40).
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
OUT = Path(os.getenv("THOR_HESS_OUT", "@@ROOT:models@@/gguf/tony-tern/hessiane-densi-grande"))
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


def alfa_spettrale(H: torch.Tensor) -> float:
    ev = torch.linalg.eigvalsh(H.double())
    ev = ev[ev > 0].flip(0)
    return float(-np.polyfit(np.log(np.arange(1, len(ev) + 1)),
                             np.log((ev / ev[0]).cpu().numpy()), 1)[0])


def main():
    if not torch.cuda.is_available():
        sys.exit("nessuna GPU: questa cattura vive in VRAM, non ha senso su CPU")
    OUT.mkdir(parents=True, exist_ok=True)

    cfg_tutto = AutoConfig.from_pretrained(str(SRC))
    cfg = cfg_tutto.text_config
    cfg._attn_implementation = "sdpa"
    n_strati = cfg.num_hidden_layers if not SOLO else min(SOLO, cfg.num_hidden_layers)
    log(f"architettura: {cfg.num_hidden_layers} strati · hidden {cfg.hidden_size}")

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
    log(f"catturo i primi {n_strati} strati (attn+ffn insieme) · tetto VRAM {TETTO_GIB} GiB · uscita {OUT}")

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

        H_attn_sum = torch.zeros(cfg.hidden_size, cfg.hidden_size, dtype=torch.float64, device="cuda")
        H_ffn_sum = torch.zeros(cfg.hidden_size, cfg.hidden_size, dtype=torch.float64, device="cuda")
        n_tot = 0
        for b in range(n_batch):
            catturato = {}

            def gancio_attn(_mod, _inp, out):
                catturato["attn"] = out.detach()

            def gancio_ffn(_mod, _inp, out):
                catturato["ffn"] = out.detach()

            h1 = strato.input_layernorm.register_forward_hook(gancio_attn)
            h2 = strato.post_attention_layernorm.register_forward_hook(gancio_ffn)
            with torch.no_grad():
                out = strato(hidden_states=hs[b], position_embeddings=pos_emb,
                             attention_mask=maschere[cfg.layer_types[L]], position_ids=pos)
                hs[b] = out[0] if isinstance(out, tuple) else out
            h1.remove(); h2.remove()
            v = max(v, controlla_tetto(f"strato {L} passata {b} dopo il forward"))

            Xa = catturato["attn"].reshape(-1, cfg.hidden_size).double()
            Xf = catturato["ffn"].reshape(-1, cfg.hidden_size).double()
            H_attn_sum += Xa.T @ Xa
            H_ffn_sum += Xf.T @ Xf
            n_tot += Xa.shape[0]
            del catturato, Xa, Xf

        H_attn = (H_attn_sum / n_tot).float()
        H_ffn = (H_ffn_sum / n_tot).float()
        np.save(OUT / f"H_attn_{L:02d}.npy", H_attn.cpu().numpy().astype(np.float32))
        np.save(OUT / f"H_ffn_{L:02d}.npy", H_ffn.cpu().numpy().astype(np.float32))

        alfa_a = alfa_spettrale(H_attn)
        alfa_f = alfa_spettrale(H_ffn)
        log(f"  strato {L:2d}/{n_strati - 1} · {n_tot} campioni ({n_batch} passate) · "
            f"alfa_attn {alfa_a:.3f} · alfa_ffn {alfa_f:.3f} · "
            f"VRAM picco {v:.1f} GiB -> {vram_gib():.2f} GiB · {time.time() - t_str:.1f}s")

        del strato, H_attn, H_ffn, H_attn_sum, H_ffn_sum
        gc.collect(); torch.cuda.empty_cache()

    log(f"FATTO: {n_strati} strati (attn+ffn) in {OUT} · {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

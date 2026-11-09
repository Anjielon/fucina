#!/usr/bin/env python3
"""CATTURA HESSIANE COMPLETA per THOR (Ornith-1.5-35B-A3B) — per-strato (gate/up)
E per-ESPERTO (down). Porta su Thor lo stesso buco chiuso su ODINO il 19/9
(vedi ~/odino-lab/odino/cattura_hessiane_down_vram.py, di cui questo e' un
adattamento SOLO nei percorsi/env — la logica e' identica, e le dimensioni
(n_esperti, dim_int, hidden) si leggono dal config, non sono hardcoded).

Buco su Thor: `forgia_thor_v1.py` fa `Hc = hchol(L) if ind == hidden else None`
(riga 250) — per `ffn_down_exps` (ingresso 512 = moe_intermediate_size,
diverso per esperto) Hc e' sempre None: RTN su TUTTI gli strati, come v34a.
Qui si cattura la Hessiana VERA per ciascuno dei 256 esperti, agganciando
`Qwen3_5MoeExperts.forward` (transformers) PRIMA di down_proj[expert_idx],
esattamente come per ODINO.

Sorgente: checkpoint HF bf16 `ornith-ai/Ornith-1.5-35B-A3B`
(@@ROOT:models@@/gguf/thor-src-hf, scaricato il 19/9 — safetensors, NON i due
pezzi GGUF bf16 di bartowski che la forgia usa: quelli sono gia' nel layout
GGUF [in,out,E] pronto per la scrittura del TQ1_0, mentre qui serve il
layout HF (fused gate_up_proj/down_proj) per costruire il vero
Qwen3_5MoeDecoderLayer e farci passare il corpus).

Uscita: @@ROOT:models@@/gguf/tony-tern/hessiane-down/
  H_{L:02d}.npy (2048x2048, gate/up — gia' esistono da 27/8 in
    tony-tern/hessiane/, qui si RI-cattura per coerenza nello stesso passaggio
    ma non serve sovrascrivere quelle: la forgia legge da THOR_HESS invariato)
  Hdown_{L:02d}.npy (256, 512, 512 float32, ~256 MiB/strato)
  Hdown_{L:02d}_conta.npy (256 interi)
"""
from __future__ import annotations
import gc, json, os, sys, time, types
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig
from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer,
    Qwen3_5MoeTextRotaryEmbedding,
)

SRC = Path(os.getenv("THOR_SORGENTE_HF", "@@ROOT:models@@/gguf/thor-src-hf"))
OUT = Path(os.getenv("THOR_HESS_OUT", "@@ROOT:models@@/gguf/tony-tern/hessiane-down"))
SOLO = int(os.getenv("THOR_SOLO_STRATI", "0"))          # >0 = solo i primi N strati
TETTO_GIB = float(os.getenv("THOR_TETTO_VRAM_GIB", "40"))
PREFISSO = "model.language_model."
CORPUS = Path(os.getenv("THOR_CORPUS", "@@ROOT:forge@@/calib_auto2_pulito.txt"))
N_SEQ = int(os.getenv("THOR_N_SEQ", "0"))    # 0 = calcolato dal corpus (vedi main)
L_SEQ = int(os.getenv("THOR_L_SEQ", "2048"))


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def vram_gib() -> float:
    return torch.cuda.memory_reserved() / 2**30


def controlla_tetto(dove: str):
    v = vram_gib()
    if v > TETTO_GIB:
        raise RuntimeError(f"TETTO VRAM SUPERATO a {dove}: {v:.2f} GiB > {TETTO_GIB} GiB — mi fermo io")
    return v


class Sorgente:
    """Identica a cattura_hessiane_down_vram.py: pread + fadvise(DONTNEED), zero mmap."""
    FETTA_MAX = 256 * 2**20

    def __init__(self, radice: Path):
        self.radice = radice
        self.mappa = json.load(open(radice / "model.safetensors.index.json"))["weight_map"]
        self._header: dict[str, tuple[dict, int]] = {}

    def _testa(self, file: str):
        if file not in self._header:
            with open(self.radice / file, "rb") as f:
                n = int.from_bytes(f.read(8), "little")
                hdr = json.loads(f.read(n))
            hdr.pop("__metadata__", None)
            self._header[file] = (hdr, 8 + n)
        return self._header[file]

    def prendi(self, chiave: str, device="cuda") -> torch.Tensor:
        file = self.mappa[chiave]
        hdr, base = self._testa(file)
        info = hdr[chiave]
        if info["dtype"] != "BF16":
            raise ValueError(f"{chiave}: dtype {info['dtype']}, atteso BF16")
        forma = tuple(info["shape"])
        inizio, fine = info["data_offsets"]
        nbyte = fine - inizio
        assert nbyte == 2 * int(np.prod(forma)), f"{chiave}: byte {nbyte} != forma {forma}"
        out = torch.empty(forma, dtype=torch.bfloat16, device=device)
        piatto = out.view(-1)
        fd = os.open(str(self.radice / file), os.O_RDONLY)
        try:
            fatti = 0
            while fatti < nbyte:
                n = min(self.FETTA_MAX, nbyte - fatti)
                buf = os.pread(fd, n, base + inizio + fatti)
                if len(buf) != n:
                    raise IOError(f"{chiave}: letti {len(buf)} byte su {n}")
                pezzo = torch.frombuffer(bytearray(buf), dtype=torch.bfloat16)
                piatto[fatti // 2: (fatti + n) // 2].copy_(pezzo)
                del pezzo, buf
                os.posix_fadvise(fd, base + inizio + fatti, n, os.POSIX_FADV_DONTNEED)
                fatti += n
        finally:
            os.close(fd)
        return out

    def chiavi_strato(self, L: int) -> list[str]:
        p = f"{PREFISSO}layers.{L}."
        return [k for k in self.mappa if k.startswith(p)]


def carica_strato(src: Sorgente, cfg, L: int) -> Qwen3_5MoeDecoderLayer:
    with torch.device("cuda"):
        dtype_prima = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            strato = Qwen3_5MoeDecoderLayer(cfg, L)
        finally:
            torch.set_default_dtype(dtype_prima)
    strato = strato.to(torch.bfloat16)
    strato.eval()
    prefisso = f"{PREFISSO}layers.{L}."
    attesi = dict(strato.named_parameters())
    attesi.update(dict(strato.named_buffers()))
    caricati = 0
    for chiave in src.chiavi_strato(L):
        nome = chiave[len(prefisso):]
        if nome not in attesi:
            raise KeyError(f"strato {L}: il peso '{nome}' non esiste nel modulo ufficiale")
        peso = src.prendi(chiave)
        bersaglio = attesi[nome]
        if tuple(bersaglio.shape) != tuple(peso.shape):
            raise ValueError(f"strato {L} '{nome}': forma {tuple(peso.shape)} != attesa {tuple(bersaglio.shape)}")
        with torch.no_grad():
            bersaglio.copy_(peso)
        del peso
        caricati += 1
    mancanti = [n for n in attesi if n not in {k[len(prefisso):] for k in src.chiavi_strato(L)}]
    if mancanti:
        raise KeyError(f"strato {L}: pesi MAI caricati: {mancanti}")
    return strato


def _forward_esperti_instrumentato(self, hidden_states, top_k_index, top_k_weights, Hdown, conta):
    """Copia VERBATIM di Qwen3_5MoeExperts.forward (transformers 5.x) con in
    piu' l'accumulo H_e += X_eᵀX_e sull'ingresso REALE di down_proj. Nessuna
    riga della computazione originale e' toccata — solo aggiunta, passiva.
    """
    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == self.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
        current_hidden_states = self.act_fn(gate) * up
        with torch.no_grad():
            e = int(expert_idx)
            Xf = current_hidden_states.detach().float()
            Hdown[e] += Xf.T @ Xf
            conta[e] += Xf.shape[0]
        current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
        current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

    return final_hidden_states


def instrumenta(strato: Qwen3_5MoeDecoderLayer, Hdown: torch.Tensor, conta: torch.Tensor):
    """Sostituisce forward SOLO su questa istanza (mai la classe)."""
    esperti = strato.mlp.experts

    def _fwd(hidden_states, top_k_index, top_k_weights):
        return _forward_esperti_instrumentato(esperti, hidden_states, top_k_index, top_k_weights, Hdown, conta)

    esperti.forward = _fwd


def main():
    if not torch.cuda.is_available():
        sys.exit("nessuna GPU: questa cattura vive in VRAM, non ha senso su CPU")
    OUT.mkdir(parents=True, exist_ok=True)

    cfg_tutto = AutoConfig.from_pretrained(str(SRC))
    cfg = cfg_tutto.text_config
    cfg._attn_implementation = "sdpa"
    n_strati = cfg.num_hidden_layers if not SOLO else min(SOLO, cfg.num_hidden_layers)
    n_esperti = cfg.num_experts
    dim_int = cfg.moe_intermediate_size
    log(f"architettura: {cfg.num_hidden_layers} strati · hidden {cfg.hidden_size} · "
        f"{n_esperti} esperti top-{cfg.num_experts_per_tok} · intermedio {dim_int}")

    src = Sorgente(SRC)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(SRC))
    testo = CORPUS.read_text(encoding="utf-8", errors="replace")
    tutti = tok(testo, return_tensors="pt", add_special_tokens=False).input_ids[0]
    n_seq_tot = N_SEQ if N_SEQ > 0 else (tutti.numel() // L_SEQ)
    if tutti.numel() < n_seq_tot * L_SEQ:
        raise ValueError(f"corpus {CORPUS}: {tutti.numel()} token, servono {n_seq_tot * L_SEQ}")
    BATCH_SEQ = int(os.getenv("THOR_BATCH_SEQ", "16"))
    n_batch = max(1, n_seq_tot // BATCH_SEQ)
    n_seq = n_batch * BATCH_SEQ
    ids_tutte = tutti[: n_seq * L_SEQ].reshape(n_seq, L_SEQ)
    n_tok = L_SEQ
    log(f"corpus {CORPUS.name}: {tutti.numel()} token disponibili → uso {n_seq} seq x {L_SEQ} = "
        f"{n_seq*L_SEQ} campioni/strato in {n_batch} passate da {BATCH_SEQ} seq "
        f"(per-esperto: media {n_seq*L_SEQ*cfg.num_experts_per_tok/n_esperti:.0f})")
    log(f"catturo i primi {n_strati} strati · tetto VRAM {TETTO_GIB} GiB · uscita {OUT}")

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

        Hdown = torch.zeros(n_esperti, dim_int, dim_int, dtype=torch.float32, device="cuda")
        conta = torch.zeros(n_esperti, dtype=torch.int64, device="cuda")
        instrumenta(strato, Hdown, conta)
        v = max(v, controlla_tetto(f"strato {L} accumulatori down allocati"))

        H_sum = torch.zeros(cfg.hidden_size, cfg.hidden_size, dtype=torch.float64, device="cuda")
        n_tot = 0
        for b in range(n_batch):
            catturato = {}

            def gancio(_mod, _inp, out):
                catturato["x"] = out.detach()

            h_reg = strato.post_attention_layernorm.register_forward_hook(gancio)
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
        ev = ev[ev > 0].flip(0)
        alfa = float(-np.polyfit(np.log(np.arange(1, len(ev) + 1)),
                                 np.log((ev / ev[0]).cpu().numpy()), 1)[0])

        conta_cpu = conta.cpu().numpy()
        colpiti = int((conta_cpu > 0).sum())
        min_colpiti = int(conta_cpu[conta_cpu > 0].min()) if colpiti else 0
        norm = conta.clamp(min=1).view(-1, 1, 1).float()
        Hdown_norm = (Hdown / norm).cpu().numpy().astype(np.float32)
        np.save(OUT / f"Hdown_{L:02d}.npy", Hdown_norm)
        np.save(OUT / f"Hdown_{L:02d}_conta.npy", conta_cpu)

        del strato, H, H_sum, ev, Hdown, Hdown_norm, conta, conta_cpu
        gc.collect(); torch.cuda.empty_cache()
        log(f"  strato {L:2d}/{n_strati - 1} · {n_tot} campioni ({n_batch} passate) · alfa {alfa:.3f} · "
            f"esperti colpiti {colpiti}/{n_esperti} (min {min_colpiti} campioni) · "
            f"VRAM picco {v:.1f} GiB → {vram_gib():.2f} GiB · {time.time() - t_str:.1f}s")

    log(f"FATTO: {n_strati} strati (H + Hdown) in {OUT} · {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

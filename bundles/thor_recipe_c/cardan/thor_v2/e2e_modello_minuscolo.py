#!/usr/bin/env python3
"""Modello MINUSCOLO con l'architettura di Thor (qwen35moe), per collaudare e2e_dump_logit su
CPU senza caricare i 12 GB veri. Scritto il 25/9/2026.

Perche': sul disco non c'e' nessun GGUF qwen35moe sotto i 2 GB (cercato con
`find @@ROOT:models@@ -name '*.gguf' -size -2G`: solo proiettori, drafter, vocabolari, pezzi di
file divisi). Il modello piccolo vero che c'e' (Qwen3-0.6B) e' denso: niente MoE (quindi niente
ffn_moe_topk) e niente GDN (quindi niente stato ricorrente da azzerare fra una sequenza e
l'altra), cioe' proprio le due cose di Thor che uno strumento multi-sequenza puo' sbagliare.

Cosa copia di Thor (Thor-v20-densiv5.gguf, sola lettura dell'intestazione):
  - l'architettura `qwen35moe` con la stessa struttura: strati GDN (gated delta net) e uno
    ad attenzione piena ogni 4 (full_attention_interval = 4), MoE con esperto condiviso,
    un blocco MTP in coda (nextn_predict_layers = 1, come blk.40);
  - IL TOKENIZZATORE INTERO (248.320 token, merges, pre `qwen35`, bos/eos/pad, add_bos):
    gli id sono quelli di Thor;
  - i formati degli esperti di densi v5: gate/up TQ1_0 (pesi davvero ternari x scala fp16),
    il resto Q8_0 / F32.
Cosa cambia (tutto ridotto): 4 strati + MTP, n_embd 256, 4 teste (2 kv) da 64, rope 16 con
sezioni [3,3,2,0], GDN 2 teste k / 4 teste v da 32, 16 esperti top-4 da 64, condiviso da 64.
Pesi casuali con seme fisso: il file e' deterministico (sha256 nel JSON accanto).

Uso:  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 19 python3 e2e_modello_minuscolo.py USCITA.gguf
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "@@ROOT:models@@/build/llamacpp-tq1-k1024/gguf-py")
from gguf import GGMLQuantizationType as GT, GGUFReader, GGUFValueType, GGUFWriter  # noqa: E402
from gguf.quants import quantize  # noqa: E402

SORGENTE_TOKENIZZATORE = "@@ROOT:models@@/gguf/tony-tern/Thor-v20-densiv5.gguf"
CHIAVI_TOKENIZZATORE = [
    "tokenizer.ggml.model", "tokenizer.ggml.pre", "tokenizer.ggml.tokens", "tokenizer.ggml.token_type",
    "tokenizer.ggml.merges", "tokenizer.ggml.eos_token_id", "tokenizer.ggml.padding_token_id",
    "tokenizer.ggml.bos_token_id", "tokenizer.ggml.add_bos_token",
]

D, N_HEAD, N_KV, HD = 256, 4, 2, 64
N_ROT, SEZIONI = 16, [3, 3, 2, 0]
E, K, F, FS = 16, 4, 64, 64
HK, NK, NV = 32, 2, 4                # ssm.state_size, ssm.group_count, ssm.time_step_rank
HV = HK                              # head_v_dim = ssm_d_state (qwen35moe.cpp)
KEY_DIM, VALUE_DIM = HK * NK, HV * NV
CONV_DIM = 2 * KEY_DIM + VALUE_DIM
N_LAYER, INTERVALLO = 4, 4
SEME = 20260925


def main(uscita: Path) -> dict:
    rng = np.random.default_rng(SEME)

    def normale(forma, std):
        return (rng.standard_normal(forma) * std).astype(np.float32)

    def norma(n):
        return (1.0 + 0.1 * rng.standard_normal(n)).astype(np.float32)

    def ternario(forma, std, blocco=256):
        """pesi = scala per blocco da 256 x trit in {-1,0,1}: TQ1_0 li rappresenta esatti."""
        t = rng.integers(-1, 2, size=forma).astype(np.float32)
        nb = forma[-1] // blocco
        s = (np.abs(rng.standard_normal(forma[:-1] + (nb, 1))) * std + 0.5 * std).astype(np.float32)
        s = s.astype(np.float16).astype(np.float32)
        return (t.reshape(forma[:-1] + (nb, blocco)) * s).reshape(forma)

    tensori: list[tuple[str, np.ndarray, GT | None]] = []

    def agg(nome, arr, qt=None):
        tensori.append((nome, arr, qt))

    r = GGUFReader(SORGENTE_TOKENIZZATORE)
    V = len(r.fields["tokenizer.ggml.tokens"].data)

    agg("token_embd.weight", normale((V, D), 1.0), GT.Q8_0)
    agg("output_norm.weight", norma(D))
    agg("output.weight", normale((V, D), 2.0 / np.sqrt(D)), GT.Q8_0)

    def blocco_moe(il):
        agg(f"blk.{il}.ffn_gate_inp.weight", normale((E, D), 3.0 / np.sqrt(D)))
        agg(f"blk.{il}.ffn_gate_exps.weight", ternario((E, F, D), 1.0 / np.sqrt(D)), GT.TQ1_0)
        agg(f"blk.{il}.ffn_up_exps.weight", ternario((E, F, D), 1.0 / np.sqrt(D)), GT.TQ1_0)
        agg(f"blk.{il}.ffn_down_exps.weight", normale((E, D, F), 0.5 / np.sqrt(F)), GT.Q8_0)
        agg(f"blk.{il}.ffn_gate_inp_shexp.weight", normale((D,), 1.0 / np.sqrt(D)))
        agg(f"blk.{il}.ffn_gate_shexp.weight", normale((FS, D), 1.0 / np.sqrt(D)), GT.Q8_0)
        agg(f"blk.{il}.ffn_up_shexp.weight", normale((FS, D), 1.0 / np.sqrt(D)), GT.Q8_0)
        agg(f"blk.{il}.ffn_down_shexp.weight", normale((D, FS), 0.5 / np.sqrt(FS)), GT.Q8_0)

    def blocco_attenzione(il):
        agg(f"blk.{il}.attn_q.weight", normale((HD * N_HEAD * 2, D), 1.0 / np.sqrt(D)), GT.Q8_0)
        agg(f"blk.{il}.attn_k.weight", normale((HD * N_KV, D), 1.0 / np.sqrt(D)), GT.Q8_0)
        agg(f"blk.{il}.attn_v.weight", normale((HD * N_KV, D), 1.0 / np.sqrt(D)), GT.Q8_0)
        agg(f"blk.{il}.attn_output.weight", normale((D, HD * N_HEAD), 0.5 / np.sqrt(HD * N_HEAD)), GT.Q8_0)
        agg(f"blk.{il}.attn_q_norm.weight", norma(HD))
        agg(f"blk.{il}.attn_k_norm.weight", norma(HD))

    for il in range(N_LAYER):
        agg(f"blk.{il}.attn_norm.weight", norma(D))
        agg(f"blk.{il}.post_attention_norm.weight", norma(D))
        if (il + 1) % INTERVALLO != 0:  # GDN, come is_recr in qwen35moe.cpp
            agg(f"blk.{il}.attn_qkv.weight", normale((CONV_DIM, D), 1.0 / np.sqrt(D)), GT.Q8_0)
            agg(f"blk.{il}.attn_gate.weight", normale((VALUE_DIM, D), 1.0 / np.sqrt(D)), GT.Q8_0)
            agg(f"blk.{il}.ssm_conv1d.weight", normale((CONV_DIM, 4), 0.5))
            agg(f"blk.{il}.ssm_dt.bias", rng.uniform(-2.0, 0.0, NV).astype(np.float32))
            agg(f"blk.{il}.ssm_a", (-np.exp(rng.uniform(-1.0, 1.0, NV))).astype(np.float32))
            agg(f"blk.{il}.ssm_beta.weight", normale((NV, D), 1.0 / np.sqrt(D)))
            agg(f"blk.{il}.ssm_alpha.weight", normale((NV, D), 1.0 / np.sqrt(D)))
            agg(f"blk.{il}.ssm_norm.weight", norma(HV))
            agg(f"blk.{il}.ssm_out.weight", normale((D, VALUE_DIM), 0.5 / np.sqrt(VALUE_DIM)), GT.Q8_0)
        else:
            blocco_attenzione(il)
        blocco_moe(il)
    # blocco MTP (come blk.40 di Thor): caricato, non eseguito nel passo principale
    il = N_LAYER
    agg(f"blk.{il}.attn_norm.weight", norma(D))
    agg(f"blk.{il}.post_attention_norm.weight", norma(D))
    blocco_attenzione(il)
    blocco_moe(il)
    agg(f"blk.{il}.nextn.eh_proj.weight", normale((D, 2 * D), 1.0 / np.sqrt(2 * D)), GT.Q8_0)
    agg(f"blk.{il}.nextn.enorm.weight", norma(D))
    agg(f"blk.{il}.nextn.hnorm.weight", norma(D))
    agg(f"blk.{il}.nextn.shared_head_norm.weight", norma(D))

    w = GGUFWriter(str(uscita), arch="qwen35moe")
    w.add_name("e2e-minuscolo-qwen35moe")
    w.add_block_count(N_LAYER + 1)
    w.add_context_length(4096)
    w.add_embedding_length(D)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_KV)
    w.add_rope_dimension_sections(SEZIONI)
    w.add_rope_freq_base(10000000.0)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_expert_count(E)
    w.add_expert_used_count(K)
    w.add_key_length(HD)
    w.add_value_length(HD)
    w.add_expert_feed_forward_length(F)
    w.add_expert_shared_feed_forward_length(FS)
    w.add_nextn_predict_layers(1)
    w.add_ssm_conv_kernel(4)
    w.add_ssm_state_size(HK)
    w.add_ssm_group_count(NK)
    w.add_ssm_time_step_rank(NV)
    w.add_ssm_inner_size(VALUE_DIM)
    w.add_full_attention_interval(INTERVALLO)
    w.add_rope_dimension_count(N_ROT)
    for chiave in CHIAVI_TOKENIZZATORE:
        c = r.fields[chiave]
        tipo = GGUFValueType(int(c.types[0]))
        sotto = GGUFValueType(int(c.types[-1])) if tipo == GGUFValueType.ARRAY else None
        w.add_key_value(chiave, c.contents(), tipo, sotto)

    for nome, arr, qt in tensori:
        if qt is None:
            w.add_tensor(nome, arr)
        else:
            w.add_tensor(nome, quantize(arr, qt), raw_dtype=qt)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    h = hashlib.sha256(uscita.read_bytes()).hexdigest()
    tok = hashlib.sha256("\n".join(r.fields["tokenizer.ggml.tokens"].contents()).encode()).hexdigest()
    info = {"file": str(uscita), "byte": uscita.stat().st_size, "sha256": h, "seme": SEME,
            "tensori": len(tensori), "n_vocab": V, "sha256_token_di_thor": tok,
            "dimensioni": {"n_embd": D, "n_head": N_HEAD, "n_head_kv": N_KV, "head_dim": HD, "n_rot": N_ROT,
                           "sezioni_rope": SEZIONI, "n_expert": E, "n_expert_used": K, "n_ff_exp": F,
                           "n_ff_shexp": FS, "ssm_state": HK, "ssm_group": NK, "ssm_dt_rank": NV,
                           "n_layer": N_LAYER, "mtp": 1, "full_attention_interval": INTERVALLO},
            "tokenizzatore_da": SORGENTE_TOKENIZZATORE}
    Path(str(uscita) + ".json").write_text(json.dumps(info, indent=1))
    return info


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("uso: e2e_modello_minuscolo.py USCITA.gguf")
    print(json.dumps(main(Path(sys.argv[1])), indent=1))

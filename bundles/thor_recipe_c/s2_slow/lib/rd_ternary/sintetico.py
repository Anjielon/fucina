"""GGUF qwen35moe MINUSCOLO nel nostro formato ternario, per i test su CPU.

Stessa struttura di Thor-T-R2B3m (tipi, nomi dei piani, strato MTP), dimensioni ridotte:
  n_embd 1024, 4 strati (0-2 lineari, 3 attenzione piena) + 1 strato MTP (blk.4),
  8 esperti (2 attivi), n_ff_exp 256, shared expert 256, vocabolario SPM di 512 voci.
  densi:   TQ1_0R_1024 a 3 piani (attn_qkv/attn_gate/ssm_out, attn_q/k/v/output, shexp gate/up);
           ffn_down_shexp TQ1_0 a 3 piani; token_embd/output TQ1_0R_1024 a 3 piani
  esperti: gate/up TQ1_0R_1024, down TQ1_0, un piano
  MTP:     tutto TQ1_0 a un piano (come R2B3m), nextn.eh_proj compreso
I piani sono trit casuali con scale ~ sqrt(1.5/in): pesi di varianza ~1/in, uscite finite.
"""
from __future__ import annotations

import sys
from typing import Dict, Optional

import numpy as np

from .gguf_piani import GGUF_PY_DEFAULT, T_TQ1_0, T_TQ1_0R_1024, FORMATS, encode_rows

N_EMBD, N_HEAD, N_HEAD_KV, HEAD = 1024, 4, 1, 256
N_EXP, N_USED, N_FF, N_FF_SH = 8, 2, 256, 256
SSM_STATE, SSM_GROUPS, SSM_DTR, SSM_INNER, CONV_K = 128, 4, 8, 1024, 4
N_LAYER, VOCAB = 4, 512


def _rand_planes(rng, out: int, in_: int, type_id: int, k: int, scale_mult: float = 1.0) -> np.ndarray:
    """k piani casuali -> byte [k, out, row_bytes] nel layout del tipo."""
    f = FORMATS[type_id]
    res = []
    for j in range(k):
        t = rng.integers(-1, 2, size=(out, in_)).astype(np.int8)
        s = (np.sqrt(1.5 / in_) * scale_mult * (0.5 ** j)
             * rng.uniform(0.7, 1.3, size=(out, in_ // f.group))).astype(np.float16)
        res.append(encode_rows(t, s, type_id))
    return np.stack(res)


def build(path: str, seed: int = 0, gguf_py: Optional[str] = None, dense_planes: int = 3,
          expert_planes2: bool = False) -> Dict:
    p = gguf_py or GGUF_PY_DEFAULT
    if p not in sys.path:
        sys.path.insert(0, p)
    import gguf
    rng = np.random.default_rng(seed)
    w = gguf.GGUFWriter(path, "qwen35moe")
    a = "qwen35moe"
    w.add_uint32(f"{a}.block_count", N_LAYER + 1)
    w.add_uint32(f"{a}.context_length", 256)
    w.add_uint32(f"{a}.embedding_length", N_EMBD)
    w.add_uint32(f"{a}.attention.head_count", N_HEAD)
    w.add_uint32(f"{a}.attention.head_count_kv", N_HEAD_KV)
    w.add_array(f"{a}.rope.dimension_sections", [11, 11, 10, 0])
    w.add_float32(f"{a}.rope.freq_base", 1e7)
    w.add_float32(f"{a}.attention.layer_norm_rms_epsilon", 1e-6)
    w.add_uint32(f"{a}.expert_count", N_EXP)
    w.add_uint32(f"{a}.expert_used_count", N_USED)
    w.add_uint32(f"{a}.attention.key_length", HEAD)
    w.add_uint32(f"{a}.attention.value_length", HEAD)
    w.add_uint32(f"{a}.expert_feed_forward_length", N_FF)
    w.add_uint32(f"{a}.expert_shared_feed_forward_length", N_FF_SH)
    w.add_uint32(f"{a}.nextn_predict_layers", 1)
    w.add_uint32(f"{a}.ssm.conv_kernel", CONV_K)
    w.add_uint32(f"{a}.ssm.state_size", SSM_STATE)
    w.add_uint32(f"{a}.ssm.group_count", SSM_GROUPS)
    w.add_uint32(f"{a}.ssm.time_step_rank", SSM_DTR)
    w.add_uint32(f"{a}.ssm.inner_size", SSM_INNER)
    w.add_uint32(f"{a}.full_attention_interval", 4)
    w.add_uint32(f"{a}.rope.dimension_count", 64)
    # vocabolario SPM minimo (i test passano gli id dei token, non il testo)
    toks = ["<unk>", "<s>", "</s>"] + [f"<0x{b:02X}>" for b in range(256)]
    toks += [f"▁t{i}" for i in range(VOCAB - len(toks))]
    types = [2, 3, 3] + [6] * 256 + [1] * (VOCAB - 259)
    w.add_string("tokenizer.ggml.model", "llama")
    w.add_array("tokenizer.ggml.tokens", toks)
    w.add_array("tokenizer.ggml.scores", [0.0] * VOCAB)
    w.add_array("tokenizer.ggml.token_type", types)
    w.add_uint32("tokenizer.ggml.bos_token_id", 1)
    w.add_uint32("tokenizer.ggml.eos_token_id", 2)
    w.add_uint32("tokenizer.ggml.unknown_token_id", 0)

    R, T = T_TQ1_0R_1024, T_TQ1_0
    QT = gguf.GGMLQuantizationType

    def tern(name: str, out: int, in_: int, tid: int, k: int = 1, experts: int = 0, mult: float = 1.0):
        for j in range(k):
            nm = name if j == 0 else name.replace(".weight", f"{j + 1}.weight")
            if experts:
                b = np.stack([_rand_planes(rng, out, in_, tid, 1, mult)[0] for _ in range(experts)])
            else:
                b = _rand_planes(rng, out, in_, tid, k, mult)[j]
            w.add_tensor(nm, b, raw_dtype=QT(tid))

    def f32(name: str, shape, val: Optional[float] = None, std: float = 0.02):
        x = np.full(shape, val, np.float32) if val is not None else (rng.standard_normal(shape) * std).astype(np.float32)
        w.add_tensor(name, x)

    K = dense_planes
    tern("token_embd.weight", VOCAB, N_EMBD, R, K, mult=30.0)
    f32("output_norm.weight", (N_EMBD,), 1.0)
    tern("output.weight", VOCAB, N_EMBD, R, K)
    key_dim, conv_dim = SSM_STATE * SSM_GROUPS, SSM_STATE * SSM_GROUPS * 2 + SSM_INNER
    for il in range(N_LAYER + 1):
        b = f"blk.{il}."
        mtp = il == N_LAYER
        kd = 1 if mtp else K
        td = T if mtp else R
        f32(b + "attn_norm.weight", (N_EMBD,), 1.0)
        f32(b + "post_attention_norm.weight", (N_EMBD,), 1.0)
        if mtp or (il + 1) % 4 == 0:
            tern(b + "attn_q.weight", N_HEAD * HEAD * 2, N_EMBD, td, kd)
            tern(b + "attn_k.weight", N_HEAD_KV * HEAD, N_EMBD, td, kd)
            tern(b + "attn_v.weight", N_HEAD_KV * HEAD, N_EMBD, td, kd)
            tern(b + "attn_output.weight", N_EMBD, N_HEAD * HEAD, td, kd)
            f32(b + "attn_q_norm.weight", (HEAD,), 1.0)
            f32(b + "attn_k_norm.weight", (HEAD,), 1.0)
        else:
            tern(b + "attn_qkv.weight", conv_dim, N_EMBD, R, K)
            tern(b + "attn_gate.weight", SSM_INNER, N_EMBD, R, K)
            tern(b + "ssm_out.weight", N_EMBD, SSM_INNER, R, K)
            f32(b + "ssm_conv1d.weight", (conv_dim, CONV_K), std=0.3)
            f32(b + "ssm_dt.bias", (SSM_DTR,), 0.0)
            f32(b + "ssm_a", (SSM_DTR,), -1.0)
            f32(b + "ssm_beta.weight", (SSM_DTR, N_EMBD), std=0.03)
            f32(b + "ssm_alpha.weight", (SSM_DTR, N_EMBD), std=0.03)
            f32(b + "ssm_norm.weight", (SSM_STATE,), 1.0)
        f32(b + "ffn_gate_inp.weight", (N_EXP, N_EMBD), std=0.05)
        tern(b + "ffn_gate_exps.weight", N_FF, N_EMBD, td, 1, experts=N_EXP)
        tern(b + "ffn_up_exps.weight", N_FF, N_EMBD, td, 1, experts=N_EXP)
        tern(b + "ffn_down_exps.weight", N_EMBD, N_FF, T, 1, experts=N_EXP)
        if expert_planes2 and not mtp:
            tern(b + "ffn_up_exps2.weight", N_FF, N_EMBD, td, 1, experts=N_EXP, mult=0.5)
        f32(b + "ffn_gate_inp_shexp.weight", (N_EMBD,), std=0.05)
        tern(b + "ffn_gate_shexp.weight", N_FF_SH, N_EMBD, td, kd)
        tern(b + "ffn_up_shexp.weight", N_FF_SH, N_EMBD, td, kd)
        tern(b + "ffn_down_shexp.weight", N_EMBD, N_FF_SH, T, kd)
        if mtp:
            tern(b + "nextn.eh_proj.weight", N_EMBD, 2 * N_EMBD, T, 1)
            f32(b + "nextn.enorm.weight", (N_EMBD,), 1.0)
            f32(b + "nextn.hnorm.weight", (N_EMBD,), 1.0)
            f32(b + "nextn.shared_head_norm.weight", (N_EMBD,), 1.0)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return {"path": path, "n_embd": N_EMBD, "n_expert": N_EXP, "n_layer": N_LAYER, "vocab": VOCAB}


if __name__ == "__main__":
    print(build(sys.argv[1]))

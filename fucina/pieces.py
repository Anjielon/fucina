"""Model pieces: families of tensors with the same role, selected by GGUF tensor name."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Piece:
    name: str
    regex: str                    # matched against the GGUF tensor name without ".weight"
    role: str                     # dense | experts | head | embedding | router | mtp
    model_types: tuple = ("moe", "dense", "hybrid")
    note: str = ""

    def contains(self, tensor_name: str) -> bool:
        return re.fullmatch(self.regex, tensor_name.removesuffix(".weight")) is not None


# Catalogue for the qwen35moe/qwen3 family (hybrid GatedDeltaNet + MoE). Other models add their own entries.
# Note: attn_qkv/attn_gate of GatedDeltaNet layers fall into "attn" (telling them apart needs the GDN layer index
# from the GGUF metadata).
CATALOGUE = (
    Piece("attn", r"blk\.\d+\.(attn_q|attn_k|attn_v|attn_qkv|attn_output|attn_gate)\d*", "dense"),
    Piece("gdn", r"blk\.\d+\.(ssm_\w+|attn_qkv|attn_gate)\d*", "dense", ("hybrid", "moe"), "GatedDeltaNet layers"),
    Piece("ffn_dense", r"blk\.\d+\.ffn_(gate|up|down)\d*", "dense", ("dense",)),
    Piece("shexp", r"blk\.\d+\.ffn_(gate|up|down)_shexp\d*", "dense", ("moe", "hybrid")),
    Piece("experts_gate_up", r"blk\.\d+\.ffn_(gate|up)_exps\d*", "experts", ("moe", "hybrid")),
    Piece("experts_down", r"blk\.\d+\.ffn_down_exps\d*", "experts", ("moe", "hybrid")),
    Piece("router", r"blk\.\d+\.ffn_gate_inp(_shexp)?", "router", ("moe", "hybrid")),
    Piece("embedding", r"token_embd\d*", "embedding"),
    Piece("head", r"output\d*", "head"),
    Piece("mtp", r"blk\.\d+\.nextn\.\w+|mtp\.\w+", "mtp"),
    # 22 Oct 2026, from the analysis of every architecture on disk (docs/ARCHITECTURES.md): appended at the end so that
    # the classification of the pieces above does not change.
    Piece("attn_mla", r"blk\.\d+\.attn_(q_a|q_b|kv_a_mqa|kv_b|k_b|v_b)\d*", "dense", ("moe", "dense", "hybrid"),
          "multi-head latent attention (DeepSeek-V2/V3)"),
    Piece("sparse_indexer", r"blk\.\d+\.indexer\.(q_proj|k_proj|weights_proj|proj)\w*", "dense",
          ("moe", "dense", "hybrid"), "lightning indexer of sparse attention (DeepSeek-V3.2 / Qwen4-exp)"),
    Piece("hyper_conn", r"blk\.\d+\.hc_\w+", "keep", ("moe", "dense", "hybrid"),
          "hyper-connection mixing (tiny, kept in full precision)"),
    Piece("draft_fc", r"fc\d*|blk\.\d+\.fc\d*", "dense", ("dense",), "speculative-draft fusion projection"),
    Piece("norm", r"(blk\.\d+\.)?([\w.]*norm\w*|rope_freqs|indexer\.k_norm)", "keep", ("moe", "dense", "hybrid"),
          "normalisation / rope tables: kept in full precision, never quantised"),
)


def piece(name: str) -> Piece:
    for p in CATALOGUE:
        if p.name == name:
            return p
    raise KeyError(name)


def classify(tensor_name: str, model_type: str = "moe", mtp_layer: Optional[int] = None) -> Optional[str]:
    """First piece containing the tensor (catalogue order). The MTP layer is recognised by its index."""
    if mtp_layer is not None and tensor_name.startswith(f"blk.{mtp_layer}."):
        return "mtp"
    for p in CATALOGUE:
        if model_type in p.model_types and p.contains(tensor_name):
            return p.name
    return None

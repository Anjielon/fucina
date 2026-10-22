"""Model analysis (22 Oct 2026): read any GGUF and tell, with no model-specific configuration,
which layers are full attention / linear attention (GatedDeltaNet, SSM) / dense FFN / MoE, which tensor families
exist, and which registered levers apply to each family, in which phase order (serial) and which ones are
alternatives of the same group (parallel exploration).

    fucina analyze MODEL.gguf [--yaml OUT.yaml]

Detection uses only GGUF metadata and tensor names, so it works for any architecture the GGUF writer supports:
  * attention layers: tensors ``attn_q``/``attn_k``/``attn_v``/``attn_qkv`` without ``ssm_*`` in the same block,
    or ``<arch>.full_attention_interval`` when the model declares it;
  * linear-attention layers: ``ssm_*`` tensors in the block (GatedDeltaNet, Mamba-style);
  * MoE layers: ``ffn_*_exps`` tensors; shared expert: ``ffn_*_shexp``; dense FFN: ``ffn_gate/up/down`` alone.
A family that matches no piece of the catalogue is reported as ``unclassified``: the user decides (the tool never
applies a lever to a tensor family it does not recognise).
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Optional

from . import pieces as P


class _T:
    def __init__(self, name, dims, type_id):
        self.name, self.n_elements, self.type_id = name, 1, type_id
        for d in dims:
            self.n_elements *= d


def _read_header(path: str):
    """Minimal GGUF v2/v3 header reader: metadata + tensor infos. Independent of gguf-py so that custom tensor types
    (e.g. ternary TQ1_0R, id 45) unknown to the stock enum do not stop the analysis."""
    import struct
    f = open(path, "rb")
    rd = lambda fmt: struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]
    if f.read(4) != b"GGUF":
        raise SystemExit(f"{path}: not a GGUF file")
    rd("I")
    n_t, n_kv = rd("Q"), rd("Q")
    rs = lambda: f.read(rd("Q")).decode("utf-8", "replace")
    scal = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    def val(t):
        if t == 8:
            return rs()
        if t == 9:
            et, n = rd("I"), rd("Q")
            return [val(et) for _ in range(n)]
        return rd(scal[t])
    meta = {}
    for _ in range(n_kv):
        k = rs()
        meta[k] = val(rd("I"))
    tensors = []
    for _ in range(n_t):
        name = rs()
        dims = [rd("Q") for _ in range(rd("I"))]
        tid = rd("I")
        rd("Q")
        tensors.append(_T(name, dims, tid))
    return meta, tensors


def analyze(path: str) -> dict:
    meta, tensors = _read_header(path)
    arch = meta.get("general.architecture", "unknown")
    n_layers = meta.get(f"{arch}.block_count")
    fai = meta.get(f"{arch}.full_attention_interval")
    n_exp = meta.get(f"{arch}.expert_count")
    n_used = meta.get(f"{arch}.expert_used_count")
    blocks = defaultdict(set)
    families = defaultdict(lambda: {"tensors": 0, "params": 0, "types": set(), "layers": set()})
    for t in tensors:
        name = t.name.removesuffix(".weight").removesuffix(".bias")
        m = re.match(r"blk\.(\d+)\.(.+)", name)
        if m:
            L, fam = int(m.group(1)), re.sub(r"\d+$", "", m.group(2))
            blocks[L].add(fam)
        else:
            L, fam = None, name
        f = families[fam]
        f["tensors"] += 1
        f["params"] += int(t.n_elements)
        f["types"].add(str(t.type_id))
        if L is not None:
            f["layers"].add(L)
    layers = []
    for L in sorted(blocks):
        fams = blocks[L]
        linear = any(x.startswith("ssm_") for x in fams)
        if fai and not linear:
            linear = (L + 1) % int(fai) != 0
        moe = any(x.endswith("_exps") for x in fams)
        layers.append({"layer": L, "mixer": "linear_attention" if linear else "attention",
                       "ffn": "moe" if moe else "dense", "shared_expert": any(x.endswith("_shexp") for x in fams)})
    model_type = ("hybrid" if any(l["mixer"] == "linear_attention" for l in layers) else "dense")
    if any(l["ffn"] == "moe" for l in layers):
        model_type = "moe" if model_type == "dense" else "hybrid"
    piece_of = {}
    for fam in families:
        hit = next((p for p in P.CATALOGUE if re.fullmatch(p.regex, f"blk.0.{fam}") or re.fullmatch(p.regex, fam)),
                   None)
        piece_of[fam] = hit.name if hit else "unclassified"
    return {"file": path, "architecture": arch, "layers": n_layers or len(layers), "model_type": model_type,
            "full_attention_interval": fai, "experts": n_exp, "experts_used": n_used,
            "attention_layers": [l["layer"] for l in layers if l["mixer"] == "attention"],
            "linear_attention_layers": [l["layer"] for l in layers if l["mixer"] == "linear_attention"],
            "families": {k: {"piece": piece_of[k], "tensors": v["tensors"], "params": v["params"],
                             "types": sorted(v["types"]), "layers": len(v["layers"])} for k, v in families.items()}}


def plan(a: dict, techniques: Optional[dict] = None) -> dict:
    """Admissible levers per piece: phases in order (serial), techniques of the same group = alternatives (parallel)."""
    if techniques is None:
        from .registry import load_techniques
        techniques = load_techniques()
    pieces = sorted({f["piece"] for f in a["families"].values() if f["piece"] != "unclassified"})
    out = {}
    for pc in pieces:
        phases = defaultdict(lambda: defaultdict(list))
        for t in techniques.values():
            if pc not in getattr(t, "pieces", ()) or a["architecture"] not in getattr(t, "architectures", ()):
                continue
            if a["model_type"] not in getattr(t, "model_types", (a["model_type"],)):
                continue
            phases[t.phase][t.group or t.name].append(t.name)
        out[pc] = {ph: {"serial_step": ph, "alternatives_in_parallel": dict(g)} for ph, g in sorted(phases.items())}
    return {"pieces": out, "unclassified": sorted(k for k, f in a["families"].items() if f["piece"] == "unclassified")}


def report(a: dict, p: dict) -> str:
    lines = [f"{a['file']}: {a['architecture']}, {a['layers']} layers, {a['model_type']}"
             + (f", {a['experts']} experts ({a['experts_used']} active)" if a.get("experts") else ""),
             f"  attention layers: {a['attention_layers']}",
             f"  linear-attention layers: {len(a['linear_attention_layers'])}"]
    for pc, phases in p["pieces"].items():
        lines.append(f"  piece {pc}:")
        for ph, d in phases.items():
            alts = "; ".join(f"{g}: {'|'.join(v)}" for g, v in d["alternatives_in_parallel"].items())
            lines.append(f"    {ph} (serial) -> {alts}")
    if p["unclassified"]:
        lines.append(f"  UNCLASSIFIED families (you decide, nothing is applied): {', '.join(p['unclassified'])}")
    return "\n".join(lines)

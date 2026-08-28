"""FORGE — any GGUF MoE model to a two-plane TQ1_0 model.

Source is an already-quantized GGUF (dequantized per expert), not the original
bf16: the non-expert tensors are copied as they are, which keeps attention and
embeddings at the source's quality for free.

Three corrections are wired in from birth, each one paid for in a failed run:
  1. experts written HOT-FIRST, router rows permuted to match — the engine
     assumes the second plane covers the expert prefix 0..K-1;
  2. hot experts keep the JOINT plane pair (a joint plane-1 used alone loses
     ~10 points to co-adaptation);
  3. a per-layer boundary check: the hottest expert, rebuilt from what was
     actually written, must match the source. An assert, not a hope.

Usage:
  forge_gguf.py --source X.gguf --output Y.gguf --hot 28 \
                --imatrix im.gguf [--hessians DIR]

The resume journal is <output>.journal (tensor index + byte offset): a crash
costs one tensor, never the run.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, "/home/angelo/build-llamacpp/gguf-py")
sys.path.insert(0, str(Path(__file__).parent))
import gguf
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as T
from gguf import quants as GQ
import ternary_gpu as G
import two_planes as D
from tq1_pack import pack, reorder_experts, hot_first_order

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def read_counts(imatrix: Path) -> dict[int, np.ndarray]:
    """layer → experts ordinati dal più caldo (dai .counts dell'imatrix)."""
    out = {}
    r = GGUFReader(str(imatrix))
    for t in r.tensors:
        if t.name.endswith(".ffn_gate_exps.weight.counts"):
            L = int(t.name.split(".")[1])
            out[L] = np.argsort(np.array(t.data).astype(np.float64).ravel())[::-1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="source GGUF (MoE, any quantization)")
    ap.add_argument("--output", required=True, help="destination GGUF (two-plane TQ1_0)")
    ap.add_argument("--hot", type=int, default=28, help="experts receiving the second plane")
    ap.add_argument("--imatrix", required=True)
    ap.add_argument("--hessians", default=None, help="directory of H_XX.npy — enables GPTQ on gate/up")
    ap.add_argument("--chunk", type=int, default=3_000_000, help="rows per GPU chunk")
    ap.add_argument("--plane2-layers", default=None,
                    help="layer range that RECEIVES the second plane, e.g. 30-40. "
                         "Default: every layer with imatrix counts (historical "
                         "behaviour). Measured on two models: the correction "
                         "helps only in the final ~quarter of the depth, so a "
                         "band like 30-40 (of 41) or 44-59 (of 60) is the "
                         "recipe — baked at forge time instead of stripped "
                         "afterwards.")
    A = ap.parse_args()
    K = A.hot

    src = GGUFReader(A.source)
    tensors = {t.name: t for t in src.tensors}
    arch = None
    for f in src.fields.values():
        if f.name.endswith(".architecture"):
            arch = str(bytes(f.parts[-1]), "utf8"); break
    hot_order = read_counts(Path(A.imatrix))
    p2_layers = None
    if A.plane2_layers:
        lo, hi = (int(x) for x in A.plane2_layers.split("-"))
        p2_layers = set(range(lo, hi + 1))

    def has_p2(L: int) -> bool:
        """One predicate for all three coupled decisions: the extra tensor in
        the plan, the joint pair + hot-first reorder, and the router-row
        permutation. Using it inconsistently recreates the two historical
        bugs at once: an orphaned joint plane-1 (non-optimal alone, measured
        c=0.882) and a router permuted against unpermuted experts."""
        return bool(K) and L in hot_order and (p2_layers is None or L in p2_layers)
    layers = sorted({int(n.split(".")[1]) for n in tensors if ".ffn_gate_exps.weight" in n})
    t0e = tensors[f"blk.{layers[0]}.ffn_gate_exps.weight"]
    E = int(t0e.shape[2])
    without_counts = sorted(set(layers) - set(hot_order))
    log(f"arch={arch} · layers MoE={len(layers)} · experts={E} · hot={K}")
    if without_counts:
        # es. lo layer MTP/nextn: mai eseguito nel forward normale → zero
        # counts in the imatrix. No hot set, no second plane — plane-1 only.
        log(f"⚠️ layers WITHOUT counts (likely MTP/nextn): {without_counts} → plane-1 only")

    def hchol(L):
        if not A.hessians: return None
        f = Path(A.hessians) / f"H_{L:02d}.npy"
        if not f.exists(): return None
        return torch.from_numpy(D.prepare_hessian(np.load(f).astype(np.float64)).astype(np.float32))

    def permutation_for(L):
        hot = [int(x) for x in hot_order[L][:K]]
        return hot_first_order(hot, E), hot

    # ── write plan ──────────────────────────────────────────────────────
    w = GGUFWriter(path=None, arch=arch)
    for f in src.fields.values():
        if f.name.startswith(("GGUF.", "split.")): continue
        try: w.add_key_value(f.name, f.contents(), f.types[0], sub_type=f.types[-1] if len(f.types) > 1 else None)
        except Exception: pass
    if K: w.add_uint32(f"{arch}.expert_count2", K)

    plan = []
    for name, t in sorted(tensors.items()):
        exp = name.endswith(("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"))
        if exp:
            shape_ = [int(x) for x in t.shape]
            plan.append((name, shape_, "forge"))
            if has_p2(int(name.split(".")[1])):
                f2 = list(shape_); f2[2] = K
                plan.append((name.replace(".weight", "2.weight"), f2, "forge2"))
        else:
            plan.append((name, [int(x) for x in t.shape], "copy"))

    for name, shape_, kind in plan:
        if kind in ("forge", "forge2"):
            fb = list(shape_)[::-1]; fb[-1] = fb[-1] // 256 * 54
            w.add_tensor_info(name, fb, np.dtype(np.uint8), int(np.prod(fb)), raw_dtype=T.TQ1_0)
        else:
            t = tensors[name]
            d = np.asarray(t.data)
            w.add_tensor_info(name, list(d.shape), d.dtype, int(d.nbytes), raw_dtype=t.tensor_type)

    journal = Path(A.output + ".journal")
    resume_at = 0
    if journal.exists() and Path(A.output).exists():
        try:
            idx, pos = (int(x) for x in journal.read_text().split())
            if Path(A.output).stat().st_size >= pos:
                resume_at = idx + 1
                log(f"⭐ RESUMING from tensor {resume_at} (byte {pos})")
        except Exception:
            pass
    Path(A.output).parent.mkdir(parents=True, exist_ok=True)
    w.open_output_file(Path(A.output))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    if resume_at:
        f = w.fout[0]; f.flush(); f.seek(pos); f.truncate(pos)

    done: dict[str, tuple] = {}
    t0 = time.time(); weights = 0
    for i, (name, shape_, kind) in enumerate(plan):
        if i < resume_at: continue
        if kind == "copy":
            t = tensors[name]
            if name.endswith(".ffn_gate_inp.weight") and has_p2(int(name.split(".")[1])):
                L = int(name.split(".")[1])
                rt = np.asarray(t.data)
                assert rt.ndim == 2 and rt.shape[0] == E
                w.write_tensor_data(np.ascontiguousarray(rt[permutation_for(L)[0]]))
            else:
                w.write_tensor_data(np.asarray(t.data))
        else:
            orig = name.replace("2.weight", ".weight") if kind == "forge2" else name
            if orig not in done:
                L = int(orig.split(".")[1])
                t = tensors[orig]
                W = GQ.dequantize(np.asarray(t.data), t.tensor_type)   # (E, out, in) f32
                E_, out_, ind = W.shape
                assert E_ == E
                nb_e = out_ * (ind // 256)
                # the Hessian is measured on the residual stream (hidden size),
                # so it applies to gate/up (input = hidden) but NOT to down,
                # whose input is the intermediate activation
                hidden = int(tensors[f"blk.{L}.ffn_gate_exps.weight"].shape[0])
                H = hchol(L) if ind == hidden else None
                Wt = torch.from_numpy(W.reshape(-1, ind)).float()
                if H is not None:
                    d1, q1 = G.quantize_one_plane(Wt, H, chunk=A.chunk)
                else:
                    r = G.quantize(Wt, None, rounds=2, chunk=A.chunk); d1, q1 = r[0], r[1]
                p1 = pack(d1.reshape(-1, 1), q1.reshape(-1, 256))
                if has_p2(L):
                    perm, hot = permutation_for(L)
                    sel = np.concatenate([np.arange(e*out_, (e+1)*out_) for e in hot])
                    Wc = torch.from_numpy(np.ascontiguousarray(W.reshape(-1, ind)[sel])).float()
                    j1d, j1q, d2, q2 = G.quantize(Wc, H, rounds=2, chunk=A.chunk)
                    del Wc
                    pj1 = pack(j1d.reshape(-1, 1), j1q.reshape(-1, 256))
                    v1 = p1.reshape(E, nb_e, 54)
                    for s, e in enumerate(hot):
                        v1[int(e)] = pj1[s*nb_e:(s+1)*nb_e]
                    p1 = reorder_experts(v1.reshape(-1, 54), perm)
                    p2 = pack(d2.reshape(-1, 1), q2.reshape(-1, 256))
                    # verifica al confine: caldo n.1 (ora pos.0) vs source dequant
                    rebuilt = GQ.dequantize(p1[:nb_e], T.TQ1_0).reshape(out_, ind) \
                        + GQ.dequantize(p2[:nb_e], T.TQ1_0).reshape(out_, ind)
                    Wv = W[hot[0]].astype(np.float32)
                    err = float(np.linalg.norm(rebuilt - Wv) / np.linalg.norm(Wv))
                    del j1d, j1q, d2, q2
                else:
                    # no second plane here (MTP layer, or outside
                    # --plane2-layers): DEDICATED plane-1, no reordering
                    p2 = None
                    rebuilt = GQ.dequantize(p1[:nb_e], T.TQ1_0).reshape(out_, ind)
                    Wv = W[0].astype(np.float32)
                    err = float(np.linalg.norm(rebuilt - Wv) / np.linalg.norm(Wv))
                del Wt
                assert err < 0.6, f"{orig}: confine rotto ({err:.2f})"
                done[orig] = (p1, p2)
                weights += W.size
                v = weights / (time.time() - t0) / 1e6
                log(f"[{i+1}/{len(plan)}] {orig} · boundary {err*100:.1f}% · {v:.1f} M/s")
                del W, d1, q1
            data_ = done[orig][1 if kind == "forge2" else 0]
            attesi = int(np.prod([x for x in ( [int(y) for y in shape_][::-1][:-1] + [[int(y) for y in shape_][::-1][-1]//256*54] ,)][0])) if False else None
            try:
                w.write_tensor_data(data_)
            except AssertionError:
                log(f"⛔ MISMATCH {name}: forma_gguf={shape_} · data_.nbytes={data_.nbytes} · data_.shape={data_.shape}")
                raise
            if kind == "forge2": done.pop(orig, None)
        w.fout[0].flush(); journal.write_text(f"{i}\n{w.fout[0].tell()}\n")
    w.close()
    log(f"🏁 FORGE COMPLETE in {(time.time()-t0)/60:.1f} min → {A.output}")


main()

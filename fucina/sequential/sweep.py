#!/usr/bin/env python3
"""S2+S3 GPU stage: ONE SEQUENTIAL sweep, layer by layer, on the stream of C (home plan §2 S2+S3, §1.2).

Two streams over the same tokens of the S1 corpus (half A builds, half B watches; M manifesto, Kv wiki+edu,
Kw WikiText-103):
  teacher  PulitoHF (sequenziale_thor: bf16 of Ornith-1.5, one layer at a time);
  C        the E2E-C1/C3 student (ac482f3c, ternary dense «all_», ternary ssm_alpha/beta) on the START GGUF (output of S1b
           if accepted, otherwise the S1 host), with the tensors of already-processed layers REPLACED by the new ones: the
           inputs of each layer are RE-CAPTURED after quantizing the preceding layers (unlike independent PTQ:
           GPTAQ 2504.02692 / QEP 2504.09629 «quantized prefix»), and within the layer after the upstream tensors are
           already new.
For each layer L (order of plan §1.2):
  P1  mixer input (input_layernorm): attn_qkv, attn_gate, ssm_alpha, ssm_beta | attn_q, attn_k, attn_v -> S2
  P2  input of ssm_out | attn_output (C recomputed with the new P1 tensors) -> S2 (+K1, +K2 folded into attn_v)
  P3  MoE input (post_attention_layernorm): ffn_gate/up_shexp (S2) and EXPERTS gate/up (S3, E1: Gq_e weighted by C's
      router, cross term with the teacher)
  P4  TRUE input of the down projections (with new gate/up): ffn_down_shexp (S2, +K1, +K2 folded into up_shexp) and
      experts down (S3)
  P5  full pass of C with the new layer -> input of layer L+1; layer drift against the teacher
GPU reduction (without dropping data): (a) every partial pass STOPS at the point it needs (exception raised by the hook:
no MoE in P1-P3); (b) in «cache» mode the teacher runs ONCE per layer (P1) and its inputs at the att_out/ffn/down points
stay in VRAM as bf16 for P2-P4 (~13 KB/token); without the cache the teacher also stops at the point;
(c) solvers on the device with factorizations on CPU fp64 (s23_algebra); (d) expert statistics for one half at a time
(offloaded to disk as fp32, read back in batches).
Gates (fail-closed): (1) injecting the start bytes as new objects leaves the layer output bit-exact IDENTICAL,
altering them changes it, restoring them gives it back bit-exact; (2) recomputed top-8 = the student's (P4); (3) written
bytes read back = intended (trit, scale) (round trip per tensor and per expert).
Resume: layers with DONE are skipped; the streams are rebuilt by re-running the done layers with their new bytes.
Outputs: <out>/layers/strato_LL/{tensors/<GGUF name>.npy (bytes of the whole tensor, only the changed ones),
registry.json, DONE}, <out>/passata.jsonl, <out>/DONE_SWEEP.

CORRECTED 2026-10-14 (fixes a defect found in the S1b audit, §5): a guard independent of the objective
(symmetric output error towards the teacher on half B + output energy in [S23_ENERGY_MIN, S23_ENERGY_MAX]) for dense
tensors and experts; candidates a in S23_ALPHAS (dense, default 0,0.5,1) and S23_ALPHAS_EXPERTS (experts, default 0,0.5)
are judged by it; short H via S23_WEIGHT_KW (Thor variant). Details in the docstring of s23_solve.py.

FAST2 2026-10-14, same counts, same bits:
  V1 TAIL (S23_TAIL=1, default): P3 keeps in VRAM C's post_attention_layernorm input (residual after the mixer, fp32,
     ~10.5 GiB); P4 and P5 restart from there (post_attention_layernorm -> MoE -> residual, the same sequence as
     Qwen3_5MoeDecoderLayer.forward) instead of redoing input_layernorm + mixer + out_proj: nothing upstream of that
     point changes between P3 and P5. Gate (fail-safe) on the first batch of P4: tail hooks == full-pass hooks
     BIT-EXACT, otherwise the tail is switched off and the full pass is used; on the first batch of P5 tail output
     == _layer output BIT-EXACT, otherwise an error.
  V2 STREAMS (S23_STREAMS=1, default): at the end of each layer (after DONE) the streams hq (C, fp32) and hp (teacher,
     bf16) are saved in <out>/streams (only the last layer); resume reloads them instead of re-running the done layers,
     if the metadata matches (layer, start, corpus, batches, sha256 of registry.json and tensors/*.npy of every done
     layer); otherwise revisit.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
from fucina.sequential import algebra as AL  # noqa: E402
from fucina.sequential import solve as RS  # noqa: E402
from fucina.sequential import common_s1b as K  # noqa: E402
from fucina.sequential import common_s1 as SC  # noqa: E402

FAM_STAGE = {"linear_attention": {1: ("attn_qkv", "attn_gate", "ssm_alpha", "ssm_beta"), 2: ("ssm_out",),
                                   3: ("ffn_gate_shexp", "ffn_up_shexp"), 4: ("ffn_down_shexp",)},
              "full_attention": {1: ("attn_q", "attn_k", "attn_v"), 2: ("attn_output",),
                                 3: ("ffn_gate_shexp", "ffn_up_shexp"), 4: ("ffn_down_shexp",)}}
POINT = {1: "attn", 2: "att_out", 3: "ffn", 4: "down"}
TERNARY = (34, 45)
CYCLES = None


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def row_cycles(state, text):
    if CYCLES:
        with open(CYCLES, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M')}\t{state}\tC-S23: {text}\n")


def list_(s, n):
    if not s:
        return list(range(n))
    out = []
    for p in s.split(","):
        if "-" in p:
            a, b = p.split("-")
            out += list(range(int(a), int(b) + 1))
        elif p:
            out.append(int(p))
    return out


class Stop(Exception):
    """raised by a hook at the point it needs: the rest of the layer is not computed."""


class Hook:
    """Hooks at the input points (same modules in the HF teacher and in the student). `stop` = point after which the
    pass is interrupted. "a_ord" (student only) = input of the experts' down projection (patch on ExpertFamily.apply,
    as in s1_cattura)."""

    def __init__(self, layer, full, want, stop=None, S=None, fam_down=None):
        self.d, self.h, self.stop, self.S, self.orig = {}, [], stop, S, None
        want = set(want) | ({stop} if stop else set())
        m_att = layer.self_attn.o_proj if full else layer.linear_attn.out_proj

        def put(k, v):
            self.d[k] = v.detach()
            if self.stop == k:
                raise Stop(k)
        if "attn" in want:
            self.h.append(layer.input_layernorm.register_forward_hook(lambda m, i, o: put("attn", o)))
        if "att_out" in want:
            self.h.append(m_att.register_forward_pre_hook(lambda m, i: put("att_out", i[0])))
        if "hmid" in want:              # FAST2 V1: post_attention_layernorm input = residual after the mixer
            self.h.append(layer.post_attention_layernorm.register_forward_pre_hook(lambda m, i: put("hmid", i[0])))
        if "ffn" in want:
            self.h.append(layer.post_attention_layernorm.register_forward_hook(lambda m, i, o: put("ffn", o)))
        if "down" in want:
            self.h.append(layer.mlp.shared_expert.down_proj.register_forward_pre_hook(lambda m, i: put("down", i[0])))
        if "a_ord" in want:
            FE = S.ExpertFamily
            self.raw = FE.__dict__.get("apply")
            orig = FE.apply
            self.orig = orig
            d = self.d

            def apply(x, dd, fam, plane, *a, **k):
                if fam is fam_down:
                    d["plane"] = plane
                    put("a_ord", x)
                return orig(x, dd, fam, plane, *a, **k)
            FE.apply = apply

    def close(self):
        for h in self.h:
            h.remove()
        self.h = []
        if self.orig is not None:
            if self.raw is None:
                del self.S.ExpertFamily.apply
            else:
                self.S.ExpertFamily.apply = self.raw
            self.orig = None


def tail_layer(st, L, x):
    """FAST2 V1: the part of Qwen3_5MoeDecoderLayer.forward (transformers 5.15) AFTER the mixer, from the residual x
    (= post_attention_layernorm input): residual = x; h = post_attention_layernorm(x); h = mlp(h); residual + h.
    Same preamble as Student._layer (f32 tensors installed, mlp.instr None)."""
    module = st.modules[L]
    for tf in st._f32_layer.get(L, ()):
        tf.install()
    module.mlp.instr = None
    try:
        h = module.post_attention_layernorm(x)
        h = module.mlp(h)
        if isinstance(h, tuple):
            h, _ = h
        return x + h
    finally:
        module.mlp.instr = None


def experts_without_sync(self, hidden_states, top_k_index, top_k_weights):
    """FAST2 V3: Qwen3_5MoeExperts.forward (transformers 5.15, bf16 teacher) with THE SAME products, on the same
    indices and in the same order, but without the two per-expert synchronizations (torch.where + comparison of a
    device tensor with a Python int, 256 x 2 per batch). nonzero() on the mask [E, k, T] yields the rows (e, pos, t) in
    lexicographic order = the concatenation of torch.where(expert_mask[e]) for increasing e; a single read of the counts.
    Same inputs for every F.linear and same order of the index_add_ calls -> same bits."""
    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        nz = expert_mask.nonzero()
        count = torch.bincount(nz[:, 0], minlength=self.num_experts).tolist()
    o = 0
    for e in range(self.num_experts):
        c = count[e]
        if c == 0:
            continue
        top_k_pos, token_idx = nz[o:o + c, 1], nz[o:o + c, 2]
        o += c
        current_state = hidden_states[token_idx]
        gate, up = F.linear(current_state, self.gate_up_proj[e]).chunk(2, dim=-1)
        current_hidden_states = self.act_fn(gate) * up
        current_hidden_states = F.linear(current_hidden_states, self.down_proj[e])
        current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
    return final_hidden_states


def sha_file(f) -> str:
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def fingerprint_layer(d: Path) -> dict:
    """FAST2 V2: sha256 of registry.json and of every tensors/*.npy in a closed strato_LL folder."""
    return {"registry": sha_file(d / "registry.json"),
            "tensors": {f.name: sha_file(f) for f in sorted((d / "tensors").glob("*.npy"))}}


def write_tensor(f, t, step_byte=1 << 28):
    """FAST2 V2: tensor (device tensors too) -> .npy in chunks (bf16 as int16, bit for bit), fsync, page cache
    released (host RAM is scarce)."""
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    mm = np.lib.format.open_memmap(f, mode="w+", dtype=str(t.dtype).replace("torch.", ""), shape=tuple(t.shape))
    row = max(1, step_byte // max(1, t[0].numel() * t.element_size()))
    for i in range(0, t.shape[0], row):
        mm[i:i + row] = t[i:i + row].cpu().numpy()
    mm.flush()
    del mm
    fd = os.open(f, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def read_tensor_in(f, t, step_byte=1 << 28) -> bool:
    """FAST2 V2: .npy -> into t (same shape and type, bf16 as int16), in chunks. False if shape/type differ."""
    arr = np.load(f, mmap_mode="r")
    view = t.view(torch.int16) if t.dtype == torch.bfloat16 else t
    if tuple(arr.shape) != tuple(t.shape) or str(arr.dtype) != str(view.dtype).replace("torch.", ""):
        return False
    row = max(1, step_byte // max(1, arr[0].size * arr.itemsize))
    for i in range(0, arr.shape[0], row):
        view[i:i + row] = torch.from_numpy(np.ascontiguousarray(arr[i:i + row])).to(t.device)
    del arr
    return True


def router(mlp, x):
    """= s1_cattura.router (same math as MoEStudente.forward)."""
    if mlp.router_w32 is not None and x.dtype not in (torch.float32, torch.float64):
        logit = F.linear(x.float(), mlp.router_w32)
    else:
        logit = F.linear(x, mlp.router_w)
    prob = torch.softmax(logit, dim=-1, dtype=torch.float32 if x.dtype != torch.float64 else torch.float64)
    idx = torch.topk(prob, mlp.top_k, dim=-1).indices
    w = prob.gather(-1, idx)
    return idx, w / w.sum(-1, keepdim=True)


class DenseAcc:
    """sums per (half, source): H = sum x_q x_q^T, Mx = sum x_f x_q^T (fp64, fp32 products as in s1_cattura), n."""

    def __init__(self, n, dev, perm=None):
        self.n, self.dev, self.perm = n, dev, perm
        self.H, self.M, self.N = {}, {}, {}

    def add(self, h, s, xq, xf):
        xq = xq.reshape(-1, self.n).float()
        xf = xf.reshape(-1, self.n).float()
        if self.perm is not None:
            xq, xf = xq[:, self.perm], xf[:, self.perm]
        k = (h, s)
        if k not in self.H:
            self.H[k] = torch.zeros(self.n, self.n, dtype=torch.float64, device=self.dev)
            self.M[k] = torch.zeros(self.n, self.n, dtype=torch.float64, device=self.dev)
            self.N[k] = 0
        self.H[k] += (xq.T @ xq).double()
        self.M[k] += (xf.T @ xq).double()
        self.N[k] += xq.shape[0]

    def meta(self, h):
        return {s: (self.H[(hh, s)], self.M[(hh, s)], self.N[(hh, s)]) for (hh, s) in self.H if hh == h}

    def pool(self, h, weight_kw: float = 1.0):
        """layer statistic for the experts: all sources, token-weighted (= sum_e sum_{t in e} w x x^T / N).
        2026-10-14 G3: Kw tokens count weight_kw times (1 = identical to before)."""
        ks = [k for k in self.H if k[0] == h]
        n = sum(self.N[k] for k in ks)
        if weight_kw == 1.0:
            return sum(self.H[k] for k in ks) / n, sum(self.M[k] for k in ks) / n, n
        c = {k: (weight_kw if k[1] == "Kw" else 1.0) for k in ks}
        ne = sum(c[k] * self.N[k] for k in ks)
        return sum(c[k] * self.H[k] for k in ks) / ne, sum(c[k] * self.M[k] for k in ks) / ne, n


class AccGateUp:
    """P3, ONE half at a time: Gq_e = sum w x_q x_q^T, C_e = W_e[gate;up] sum w x_f x_q^T (= s1_cattura), count, weight."""

    def __init__(self, E, I, H, dev):
        self.Gq = torch.zeros(E, H, H, dtype=torch.float32, device=dev)
        self.C = torch.zeros(E, 2 * I, H, dtype=torch.float32, device=dev)
        self.count = torch.zeros(E, dtype=torch.int64, device=dev)
        self.weight = torch.zeros(E, dtype=torch.float64, device=dev)

    def reset(self):
        for t in (self.Gq, self.C, self.count, self.weight):
            t.zero_()

    def add(self, xq, xf, w, k, order, offs, Wgu32):
        tok = order // k
        wf = w.reshape(-1)[order].float()
        o = offs.tolist()
        for e in range(len(o) - 1):
            s0, s1 = o[e], o[e + 1]
            if s1 <= s0:
                continue
            t = tok[s0:s1]
            ww = wf[s0:s1, None]
            Xq = xq[t]
            self.Gq[e] += (Xq * ww).T @ Xq
            self.C[e] += (xf[t] @ Wgu32[e].T).mul_(ww).T @ Xq
            self.count[e] += s1 - s0
            self.weight[e] += ww.double().sum()

    def download(self, d: Path):
        tmp = Path(str(d) + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        E, H = self.Gq.shape[0], self.Gq.shape[1]
        mm = np.lib.format.open_memmap(tmp / "Gq.npy", mode="w+", dtype=np.float32, shape=(E, H * (H + 1) // 2))
        for e0 in range(0, E, 16):
            mm[e0:e0 + 16] = SC.pack(self.Gq[e0:e0 + 16]).cpu().numpy()
        mm.flush()
        del mm
        np.save(tmp / "C.npy", self.C.cpu().numpy())
        np.save(tmp / "count.npy", self.count.cpu().numpy())
        np.save(tmp / "weight.npy", self.weight.cpu().numpy())
        shutil.rmtree(d, ignore_errors=True)
        os.replace(tmp, d)


class AccDown:
    """P4, per half: Hd_e = sum w a_q a_q^T (a_q = TRUE input of C's down projection), Md_e = sum w a_f a_q^T
    (a_f from the teacher)."""

    def __init__(self, E, I, dev):
        self.Hd = {h: torch.zeros(E, I, I, dtype=torch.float32, device=dev) for h in "AB"}
        self.Md = {h: torch.zeros(E, I, I, dtype=torch.float32, device=dev) for h in "AB"}
        self.count = {h: torch.zeros(E, dtype=torch.int64, device=dev) for h in "AB"}
        self.weight = {h: torch.zeros(E, dtype=torch.float64, device=dev) for h in "AB"}

    def add(self, h, a_ord, xf, w, k, order, offs, Wgu32):
        tok = order // k
        wf = w.reshape(-1)[order].float()
        o = offs.tolist()
        I = Wgu32.shape[1] // 2
        for e in range(len(o) - 1):
            s0, s1 = o[e], o[e + 1]
            if s1 <= s0:
                continue
            ww = wf[s0:s1, None]
            A = a_ord[s0:s1].float()
            gu = xf[tok[s0:s1]] @ Wgu32[e].T
            Af = F.silu(gu[:, :I]) * gu[:, I:]
            self.Hd[h][e] += (A * ww).T @ A
            self.Md[h][e] += (Af * ww).T @ A
            self.count[h][e] += s1 - s0
            self.weight[h][e] += ww.double().sum()


def unpack_batch(v, n, dev, dt):
    M = torch.zeros(v.shape[0], n, n, dtype=dt, device=dev)
    i, j = torch.triu_indices(n, n, device=dev)
    vv = v.to(dev, dt)
    M[:, i, j] = vv
    M[:, j, i] = vv
    return M


def U_Minv_batch(Hn):
    """for each b: (Minv, U) = cholesky_damped(Hn_b) (percdamp 1 %), batched on CPU fp64; failures redone one by one."""
    dev, dt = Hn.device, Hn.dtype
    H = Hn.detach().to("cpu", torch.float64)
    n = H.shape[-1]
    dead = torch.diagonal(H, dim1=1, dim2=2) == 0
    if bool(dead.any()):
        H = H.clone()
        for b in range(H.shape[0]):
            if dead[b].any():
                H[b, dead[b], dead[b]] = 1
    dm = torch.diagonal(H, dim1=1, dim2=2).mean(1).view(-1, 1, 1)
    L, info = torch.linalg.cholesky_ex(H + 0.01 * dm * torch.eye(n, dtype=torch.float64))
    Minv = torch.cholesky_inverse(L)
    U, info2 = torch.linalg.cholesky_ex(Minv, upper=True)
    bad = ((info != 0) | (info2 != 0) | torch.isnan(U).flatten(1).any(1)).nonzero().flatten().tolist()
    for b in bad:
        Mi, Ub, _ = AL.cholesky_damped(Hn[b].detach().to("cpu", torch.float64))
        Minv[b], U[b] = Mi, Ub
    return Minv.to(dev, dt), U.to(dev, dt), len(bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thor-v2", required=True)
    ap.add_argument("--faithful", required=True)
    ap.add_argument("--rif", required=True, help="folder with the frozen cardan_fold.py")
    ap.add_argument("--grad-cal", default="")
    ap.add_argument("--grad-ctl", default="")
    ap.add_argument("--recipe", default="B3_ind32")
    ap.add_argument("--engine", default="triton")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--layers", default="")
    ap.add_argument("--experts", default="", help="trial: experts to solve (empty = all); the others keep the start")
    ap.add_argument("--max-seq", type=int, default=0)
    ap.add_argument("--truncate", type=int, default=0)
    ap.add_argument("--batch-token", type=int, default=8192)
    ap.add_argument("--batch-experts", type=int, default=16)
    ap.add_argument("--cache-teacher", type=int, default=1)
    ap.add_argument("--cap", type=float, default=64.0)
    ap.add_argument("--cycles", default="")
    a = ap.parse_args()
    global CYCLES
    CYCLES = a.cycles or None
    opz = {"alphas": [float(x) for x in os.environ.get("S23_ALPHAS", "0,0.5,1").split(",")],      # 2026-10-14 G2: + a = 0
           "w": float(os.environ.get("S23_W_KNOWLEDGE", "1.0")),
           "damp_qep": float(os.environ.get("S23_DAMP_QEP", "0.01")),
           "drq": os.environ.get("S23_DRQ", "1") == "1",
           "k1": tuple(x for x in os.environ.get("S23_K1", ",".join(RS.FAM_K1_DEFAULT)).split(",") if x),
           "rounds_k1": int(os.environ.get("S23_ROUNDS_K1", "2")), "k2": os.environ.get("S23_K2", "1") == "1",
           "rounds_k2": int(os.environ.get("S23_ROUNDS_K2", "2")), "limit_c": 8.0,
           # 2026-10-13 per-family DRQ (s23_solve «DRQ per family»): grid of 2610.11226 App. C, calibrated on half B
           "drq_mode": os.environ.get("S23_DRQ_GRID", "paper+norm"),
           "drq_layers_tar": int(os.environ.get("S23_DRQ_LAYERS_TAR", "2")),
           # 2026-10-14 independent guard + short H (s23_solve G1-G3)
           "energy_min": float(os.environ.get("S23_ENERGY_MIN", "0.9")),
           "energy_max": float(os.environ.get("S23_ENERGY_MAX", "1.1")),
           "guard": os.environ.get("S23_GUARD", "sim"),
           "weight_kw": float(os.environ.get("S23_WEIGHT_KW", "1.0")),
           "energy_relative": os.environ.get("S23_ENERGY_RELATIVE", "1") == "1",
           # FAST 2026-10-14 (L3): banded K1 + exact recomputation within delta of the choice (s23_solve);
           # 0 = always fastkron_sd
           "k1_band": os.environ.get("S23_K1_BAND", "1") == "1",
           "k1_band_delta": float(os.environ.get("S23_K1_DELTA", "0.002"))}
    if opz["guard"] not in ("sim", "kl_local"):
        raise ValueError(f"S23_GUARD {opz['guard']!r} not in (sim, kl_local)")
    RS.grid_drq(opz["drq_mode"])                       # unknown mode -> error right away, not at layer 0
    alpha_e = [float(x) for x in os.environ.get("S23_ALPHAS_EXPERTS", os.environ.get("S23_ALPHA_EXPERTS", "0,0.5")).split(",")]
    guard_e = {"emin": opz["energy_min"], "emax": opz["energy_max"], "weight_kw": opz["weight_kw"],
                 "relative": opz["energy_relative"]}
    kk = float(os.environ.get("S23_K", "8000"))
    use_tail = os.environ.get("S23_TAIL", "1") == "1"                 # VELOCE2 V1
    use_streams = os.environ.get("S23_STREAMS", "1") == "1"             # VELOCE2 V2
    which_B = os.environ.get("S23_B", "B")
    sys.path.insert(0, a.thor_v2)
    sys.path.insert(0, a.faithful)
    sys.path.insert(0, a.rif)
    from fucina.student import student as S
    import sequenziale_thor as ST
    from fucina.sequential import cardan_fold as CP
    # S23_KBEST > 1: Klein K-best (OJBKQ 2602.08376) in the JARQ code step; off = CP.jarq_batch bit-exact
    # FAST 2026-10-14 (L2): off = CP.jarq_batch with the Cholesky factors of the Babai blocks kept per batch (bit-exact)
    JQL = AL.jarq_batch_kbest_from(CP) if AL.KBEST > 1 else AL.jarq_batch_cache_from(CP)
    print(f"Klein K-best: S23_KBEST={AL.KBEST} ({'on' if AL.KBEST > 1 else 'off'}), seed {AL.KBEST_SEED}", flush=True)
    from fucina.sequential.hf_source import SourceHF, bf16_a_f32
    gguf = K.gp._import_gguf(None)
    dev = torch.device(a.device)
    dt = AL.dt_work(dev)
    torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(a.out)
    (out / "layers").mkdir(parents=True, exist_ok=True)
    lav = out / "work"
    t_start = time.time()
    log(f"options {opz} | experts: alpha {alpha_e}, k {kk} | B={which_B} | teacher cache {a.cache_teacher} | dt {dt} | "
        f"FAST2 tail {use_tail} streams {use_streams} | torch threads {torch.get_num_threads()} OPENBLAS "
        f"{os.environ.get('OPENBLAS_NUM_THREADS', '-')} OMP {os.environ.get('OMP_NUM_THREADS', '-')}")
    rd = gguf.GGUFReader(a.start, "r")
    tens = {t.name: t for t in rd.tensors}
    perm_ssm = torch.from_numpy(dict(np.load(Path(a.thor_v2) / "mappa_completa_gguf_hf_perm.npz"))
                                ["linear.ssm_out.weight"].astype(np.int64)).to(dev)
    st = S.Student(gguf=Path(a.start), layers=range(40), recipe=a.recipe, engine_name=a.engine, device=str(dev),
                    dense_ternary="all_", log=log)
    cfg = st.cfg
    nL = len(cfg.layer_types)
    layers = list_(a.layers, nL)
    assert layers == list(range(layers[0], layers[-1] + 1)) and layers[0] == 0, "layers: contiguous prefix starting at 0"
    if dev.type == "cuda":
        st.load_embedding()
    P = ST.PulitoHF(cfg, str(dev))
    if os.environ.get("S23_TEACHER_WITHOUT_SYNC", "1") == "1":         # FAST2 V3
        import types
        for _t, (_L, _s) in P.sch.items():
            _ex = _s.mlp.experts
            _ex.forward = types.MethodType(experts_without_sync, _ex)
        log(f"FAST2 V3: teacher experts without per-expert synchronizations ({sorted(P.sch)})")
    E, I, Hd = int(cfg.num_experts), int(cfg.moe_intermediate_size), int(cfg.hidden_size)
    P_gqa = AL.tie_gqa(cfg.num_attention_heads, cfg.num_key_value_heads,
                          getattr(cfg, "head_dim", None) or Hd // cfg.num_attention_heads, dev, dt)
    sh = SourceHF()
    es_list = list_(a.experts, E) if a.experts else list(range(E))
    data = torch.load(a.corpus, map_location="cpu", weights_only=False)
    keys = [(h, s) for h in "AB" for s in SC.SOURCES]
    ids, hp, hq, ctx = {}, {}, {}, {}
    for (h, s) in keys:
        v = data["sorgenti"][s][data["meta"][s][h]].long()
        if a.max_seq:
            v = v[: a.max_seq]
        if a.truncate:
            v = v[:, : a.truncate]
        ids[(h, s)] = v
        N, T = v.shape
        B = max(1, a.batch_token // T)
        hp[(h, s)] = torch.empty(N, T, Hd, dtype=P.dt, device=dev)
        hq[(h, s)] = torch.empty(N, T, Hd, dtype=st.recipe.dtype_residual, device=dev)
        for i in range(0, N, B):
            hp[(h, s)][i:i + B] = P.embed(v[i:i + B])
            hq[(h, s)][i:i + B] = st.embedding(v[i:i + B]).to(hq[(h, s)].dtype)
        ctx[(h, s)] = {b: (st._context(b, T, st.recipe.dtype_residual), st._context(b, T, P.dt))
                       for b in sorted({B, N % B} - {0})}
        log(f"half {h} source {s}: {N} x {T} = {N * T} tokens, batches of {B}")
    tok_tot = sum(v.numel() for v in ids.values())
    cache = {}
    if a.cache_teacher:
        for k, v in ids.items():
            N, T = v.shape
            cache[k] = {p: torch.empty(N, T, n, dtype=P.dt, device=dev)
                        for p, n in (("att_out", 4096), ("ffn", Hd), ("down", int(cfg.shared_expert_intermediate_size)))}
        log(f"teacher cache: {sum(x.numel() * 2 for c in cache.values() for x in c.values()) / 2**30:.1f} GiB")

    def batches():
        for k in keys:
            N, T = ids[k].shape
            B = max(1, a.batch_token // T)
            for i in range(0, N, B):
                sl = slice(i, min(i + B, N))
                yield k, sl, ctx[k][sl.stop - sl.start]

    def vram():
        if dev.type == "cuda":
            v = torch.cuda.memory_reserved() / 2**30
            if v > a.cap:
                raise RuntimeError(f"VRAM {v:.1f} GiB > cap {a.cap}")
            return v
        return 0.0

    def tail(L, x):
        return tail_layer(st, L, x)

    def run(L, want_q=(), stop_q=None, want_f=(), stop_f=None, cb=None, advance_q=False, advance_f=False,
               fam_down=None, hmid=None):
        """one round over all batches: teacher (or cache) and C, each stopped at its own point or run in full (advance).
        FAST2 V1: hmid = {key: residual after the mixer} -> C starts from the tail instead of from _layer."""
        sp, sq = P.strato(L), st.modules[L]
        full = cfg.layer_types[L] == "full_attention"
        for k, sl, (cq, cp) in batches():
            xf, op = {}, None
            if want_f or advance_f:
                if a.cache_teacher and not advance_f and set(want_f) <= {"att_out", "ffn", "down"}:
                    xf = {p: cache[k][p][sl] for p in want_f}
                else:
                    g = Hook(sp, full, want_f, None if advance_f else stop_f)
                    try:
                        with torch.no_grad():
                            op = P.avanti(L, hp[k][sl], cp)
                    except Stop:
                        pass
                    finally:
                        g.close()
                    xf = g.d
            g = Hook(sq, full, want_q, None if advance_q else stop_q, S, fam_down)
            oq = None
            try:
                with torch.no_grad():
                    oq = tail(L, hmid[k][sl]) if hmid is not None else st._layer(L, hq[k][sl], cq)
            except Stop:
                pass
            finally:
                g.close()
            if cb is not None:
                cb(k, sl, g.d, xf, oq, op)
            if advance_f and op is not None:
                hp[k][sl] = op.to(hp[k].dtype)
            if advance_q:
                hq[k][sl] = oq.to(hq[k].dtype)
        vram()

    # ── injection of the bytes into the student (NEW objects: no student cache can bypass them) ──
    def inject(L, fam, raw):
        es = st.experts(L)
        old = es.fam[fam]
        Bb = torch.from_numpy(np.ascontiguousarray(raw))
        if Bb.dim() == 2:
            Bb = Bb[None]
        assert Bb.dtype == torch.uint8 and tuple(Bb.shape) == tuple(old.blocks.shape), (L, fam, Bb.shape, old.blocks.shape)
        d0 = S.CS.scale_from_blocks(Bb.reshape(-1, old.nb), old.type_).to(torch.float32).reshape(Bb.shape[0], Bb.shape[1], -1)
        es.fam[fam] = S.FamilyData(old.name, old.type_, Bb.to(old.blocks.device), d0.to(old.d0.device), old.layer)

    def raw_start(name):
        return np.asarray(tens[name].data)

    def encode(T, Sc, tid, C):
        Tn = T.to("cpu").numpy().astype(np.int8)
        S16 = Sc.to("cpu", torch.float16).numpy()
        b = K.gp.encode_rows(Tn, S16, tid)
        tp, sp_ = K.gp.decode_rows(b, tid, C)
        assert np.array_equal(tp, Tn) and np.array_equal(sp_.view(np.uint16), S16.view(np.uint16)), "round trip"
        return b

    def decode(raw_rows, tid, C):
        t, d = K.gp.decode_rows(np.ascontiguousarray(raw_rows), tid, C)
        return torch.from_numpy(t.astype(np.int8)), torch.from_numpy(d.astype(np.float64))

    def fam_of(name):
        return name.split(".")[2]

    def revisit(L):
        d = out / "layers" / f"strato_{L:02d}"
        for f in sorted((d / "tensors").glob("blk.*.npy")):
            name = f.name[:-4]
            fam = fam_of(name)
            key = {"ffn_gate_exps": "gate", "ffn_up_exps": "up", "ffn_down_exps": "down"}.get(fam, fam)
            inject(L, key, np.load(f))
        run(L, advance_q=True, advance_f=True)

    def gate_injection(L):
        """(1) start bytes as new objects -> bit-exact identical; altered -> different; restored -> identical."""
        k0 = keys[0]
        N, T = ids[k0].shape
        b = min(max(1, a.batch_token // T), N)
        cq = ctx[k0][b][0] if b in ctx[k0] else st._context(b, T, st.recipe.dtype_residual)
        x = hq[k0][:b]
        with torch.no_grad():
            o0 = st._layer(L, x, cq).clone()
        es = st.experts(L)
        names = {f: es.fam[f] for f in list(es.fam)}
        for f, fam in names.items():
            inject(L, f, fam.blocks.detach().cpu().numpy())
        with torch.no_grad():
            o1 = st._layer(L, x, cq).clone()
        f0 = next(f for f in names if f not in ("gate", "up", "down"))
        raw = es.fam[f0].blocks.detach().cpu().numpy().copy()
        tid = {"TQ1_0": 34, "TQ1_0R_1024": 45}[es.fam[f0].type_]
        C = es.fam[f0].C
        Tt, Ss = decode(raw[0], tid, C)
        inject(L, f0, encode(Tt, Ss * 0.5, tid, C))
        with torch.no_grad():
            o2 = st._layer(L, x, cq).clone()
        inject(L, f0, raw)
        with torch.no_grad():
            o3 = st._layer(L, x, cq)
        ok = torch.equal(o0, o1) and not torch.equal(o0, o2) and torch.equal(o0, o3)
        log(f"injection gate layer {L} ({f0} altered): identical {torch.equal(o0, o1)}, altered different "
            f"{not torch.equal(o0, o2)}, restored identical {torch.equal(o0, o3)}")
        if not ok:
            raise RuntimeError("injection gate: replacing the bytes has no effect (or is not undone) on the stream of C")

    tail_live = {"on": use_tail}

    def gate_tail(L, hm, fam_down):
        """FAST2 V1, before P4: on the first batch the P4 hooks (ffn, down, a_ord, plane) from the tail == those of the
        full pass BIT-EXACT. False -> the tail is switched off (P4/P5 fall back to the full pass)."""
        k, sl, (cq, _) = next(batches())
        full = cfg.layer_types[L] == "full_attention"
        dd = []
        for from_tail in (False, True):
            g = Hook(st.modules[L], full, ("ffn", "down", "a_ord"), "a_ord", S, fam_down)
            try:
                with torch.no_grad():
                    if from_tail:
                        tail(L, hm[k][sl])
                    else:
                        st._layer(L, hq[k][sl], cq)
            except Stop:
                pass
            finally:
                g.close()
            dd.append(g.d)
        ok = all(torch.equal(dd[0][p], dd[1][p]) for p in ("ffn", "down", "a_ord")) \
            and torch.equal(dd[0]["plane"].order, dd[1]["plane"].order) and torch.equal(dd[0]["plane"].offs, dd[1]["plane"].offs)
        log(f"tail gate P4 layer {L}: ffn/down/a_ord/plane hooks from the tail == full pass bit-exact: {ok}"
            + ("" if ok else " -> TAIL OFF, P4/P5 use the full pass"))
        return ok

    def registers_closed(L):
        """2026-10-13: [(layer, registry)] of the CLOSED (DONE) layers before L: the per-family DRQ calibration is
        recomputed from here for every tensor, so a resume never counts a layer twice."""
        rr = []
        for x in range(L):
            dx = out / "layers" / f"strato_{x:02d}"
            if (dx / "DONE").exists():
                rr.append((x, json.loads((dx / "registry.json").read_text())))
        return rr

    # ── FAST2 V2: streams saved at the end of each layer (only the last), reloaded on resume if the metadata matches ──
    FL = out / "streams"

    def fingerprint(L):
        return fingerprint_layer(out / "layers" / f"strato_{L:02d}")

    def meta_base():
        pz = Path(a.start)
        cp = Path(a.corpus)
        return {"start": str(pz), "start_byte": pz.stat().st_size, "corpus": str(cp), "corpus_byte": cp.stat().st_size,
                "batch_token": a.batch_token, "max_seq": a.max_seq, "truncate": a.truncate, "recipe": a.recipe,
                "shapes": {f"{h}_{s_}": list(ids[(h, s_)].shape) for (h, s_) in keys},
                "dtype_q": str(next(iter(hq.values())).dtype), "dtype_f": str(next(iter(hp.values())).dtype)}

    def save_streams(L, fingerprints):
        t0 = time.time()
        need = sum(t.numel() * t.element_size() for d_ in (hq, hp) for t in d_.values())
        gia = sum(f.stat().st_size for f in FL.glob("*.npy")) if FL.exists() else 0
        free = shutil.disk_usage(out).free + gia
        if free - need < 40 * 2**30:      # never below 40 GiB free for the streams (the real outputs take precedence)
            log(f"streams NOT saved after layer {L}: {free / 2**30:.0f} GiB free, {need / 2**30:.1f} + 40 needed")
            return
        tmp = Path(str(FL) + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(FL, ignore_errors=True)     # first: only one set of streams on disk (~16 GiB), never two
        tmp.mkdir(parents=True)
        for (h, s_) in keys:
            write_tensor(tmp / f"q_{h}_{s_}.npy", hq[(h, s_)])
            write_tensor(tmp / f"f_{h}_{s_}.npy", hp[(h, s_)])
        meta = {**meta_base(), "layer": L, "fingerprints": fingerprints, "written": time.strftime("%FT%T"),
                "code": str(HERE.parent)}
        SC.write_json(tmp / "meta.json", meta)
        shutil.rmtree(FL, ignore_errors=True)
        os.replace(tmp, FL)
        log(f"streams saved after layer {L} in {time.time() - t0:.1f} s ({FL})")

    def load_streams(L0):
        """-> layer L of the loaded streams (hq, hp = outputs of layer L), or -1 if missing/inconsistent."""
        mf = FL / "meta.json"
        if not use_streams or not mf.exists():
            return -1
        t0 = time.time()
        meta = json.loads(mf.read_text())
        Lc = int(meta.get("layer", -1))
        base = meta_base()
        different = [x for x in base if meta.get(x) != base[x]]
        if different or not 0 <= Lc < L0:
            log(f"resume: streams in {FL} not usable (layer {Lc}, done 0..{L0 - 1}, metadata differing {different}): revisit")
            return -1
        for L in range(Lc + 1):
            imp = fingerprint(L)
            if imp != meta["fingerprints"].get(str(L)):
                log(f"resume: fingerprint of layer {L} differs from the one stored with the streams: revisit")
                return -1
        for (h, s_) in keys:                # BEFORE touching hq/hp: check all shapes and types (the streams stay intact)
            for name, dest in (("q", hq), ("f", hp)):
                f = FL / f"{name}_{h}_{s_}.npy"
                t = dest[(h, s_)]
                vt = t.view(torch.int16) if t.dtype == torch.bfloat16 else t
                arr = np.load(f, mmap_mode="r") if f.exists() else None
                if arr is None or tuple(arr.shape) != tuple(t.shape) or str(arr.dtype) != str(vt.dtype).replace("torch.", ""):
                    log(f"resume: {f.name} missing or with different shape/type: revisit")
                    return -1
                del arr
        for (h, s_) in keys:
            for name, dest in (("q", hq), ("f", hp)):
                if not read_tensor_in(FL / f"{name}_{h}_{s_}.npy", dest[(h, s_)]):
                    raise RuntimeError(f"resume: {name}_{h}_{s_} changed while being read")
        log(f"resume: streams of layer {Lc} reloaded from {FL} in {time.time() - t0:.1f} s (fingerprints of 0..{Lc} ok)")
        return Lc

    fingerprints = {}
    done = [L for L in layers if (out / "layers" / f"strato_{L:02d}" / "DONE").exists()]
    L0 = 0
    while L0 in done:
        L0 += 1
    Lc = load_streams(L0) if L0 else -1
    for L in range(L0):
        if L <= Lc:                          # streams already past this layer: only the bytes into the student
            d = out / "layers" / f"strato_{L:02d}"
            for f in sorted((d / "tensors").glob("blk.*.npy")):
                name = f.name[:-4]
                fam = fam_of(name)
                inject(L, {"ffn_gate_exps": "gate", "ffn_up_exps": "up", "ffn_down_exps": "down"}.get(fam, fam), np.load(f))
            log(f"resume: layer {L} injected (saved streams)")
        else:
            revisit(L)
            log(f"resume: layer {L} reapplied")
    if use_streams:
        fingerprints = {str(L): fingerprint(L) for L in range(L0)}
    remainder = [L for L in layers if L >= L0]
    gradc = Path(a.grad_cal) if a.grad_cal and Path(a.grad_cal, "fatto.txt").exists() else None
    gradk = Path(a.grad_ctl) if a.grad_ctl and Path(a.grad_ctl, "fatto.txt").exists() else None
    log(f"Kronecker factors: cal {gradc} ctl {gradk} (K1 {'active' if gradc and gradk else 'OFF: factors missing'})")
    if remainder:
        gate_injection(remainder[0])
    for L in remainder:
        tl = time.time()
        type_ = cfg.layer_types[L]
        full = type_ == "full_attention"
        d = out / "layers" / f"strato_{L:02d}"
        tmpd = Path(str(d) + ".tmp")
        shutil.rmtree(tmpd, ignore_errors=True)
        (tmpd / "tensors").mkdir(parents=True)
        reg = {"layer": L, "type_": type_, "dense_": {}, "experts": {}, "times": {}, "start": a.start}
        new = {}                       # name -> bytes of the whole tensor
        chosen = {}                      # fam -> (T, S, tid) of the dense tensors already decided (for the K2 folds)

        def solve_dense(stage, acc):
            for fam in FAM_STAGE[type_][stage]:
                name = f"blk.{L}.{fam}.weight"
                if name not in tens or int(tens[name].tensor_type) not in TERNARY:
                    continue
                t = tens[name]
                tid = int(t.tensor_type)
                C, R = int(t.shape[0]), int(t.shape[1])
                T0, S0 = decode(raw_start(name).reshape(R, -1), tid, C)
                u16, _ = sh.u16_order_gguf(name)
                W = torch.from_numpy(bf16_a_f32(u16)).to(dev, dt)
                kron = RS.Kron(gradc, gradk, name, which_B, dev, dt) if fam in opz["k1"] else None
                upstream = None
                if fam in RS.PAIRS_K2 and RS.PAIRS_K2[fam] in chosen:
                    Tm, Sm, _ = chosen[RS.PAIRS_K2[fam]]
                    upstream = (Tm, Sm)
                opz_t = opz
                if opz["drq"]:                        # 2026-10-13: per-family calibration from the registries of CLOSED layers
                    opz_t = {**opz, "drq_calibration": RS.calibration_family(registers_closed(L), fam, opz["drq_layers_tar"],
                                                                           opz["drq_mode"])}
                r = RS.solve_dense(name, fam, tid, W, acc.meta("A"), acc.meta("B"), T0, S0, opz_t, kron, P_gqa, upstream)
                chosen[fam] = (r["T"], r["S"], tid)
                reg["dense_"][name] = r["registry"]
                if r["chosen"] != "start":
                    new[name] = encode(r["T"], r["S"], tid, C)
                    inject(L, fam, new[name])
                if r["S_upstream"] is not None:
                    fm = RS.PAIRS_K2[fam]
                    nm = f"blk.{L}.{fm}.weight"
                    Tm, _, tidm = chosen[fm]
                    chosen[fm] = (Tm, r["S_upstream"], tidm)
                    new[nm] = encode(Tm, r["S_upstream"], tidm, int(tens[nm].shape[0]))
                    inject(L, fm, new[nm])
                    reg["dense_"][nm]["fold_K2_from"] = fam
                rr = r["registry"]
                dg = rr.get("drq_grid")
                if dg:
                    log(f"  {name}: DRQ {'CALIBRATED' if dg['calibrated'] else 'fixed'} ({dg['mode']}) -> {dg['choice']}"
                        + (f" (rel {dg['rel'][dg['choice']]:+.3e}, previous pair {dg['rel_fixed']:+.3e}, "
                           f"sum_ {dg['sum_choice']:+.3e}, {dg['seconds']} s)" if dg["calibrated"] else ""))
                log(f"  {name}: chosen {r['chosen']} (guard {rr['metric_guard']}, trits changed "
                    f"{rr['trit_changed']:.3f}, {rr['seconds']} s) "
                    + " ".join(f"{k}:{v['dJB_rel']:+.4f}" for k, v in rr["candidates"].items()))
                del W

        # ═══ P1: teacher (full pass with the cache, otherwise stopped at attn) + C stopped at attn ═══
        t = time.time()
        acc1 = DenseAcc(Hd, dev)

        def cb1(k, sl, dq, df, oq, op):
            acc1.add(k[0], k[1], dq["attn"], df["attn"])
            if a.cache_teacher:
                for p in ("att_out", "ffn", "down"):
                    cache[k][p][sl] = df[p].reshape(cache[k][p][sl].shape)
        if a.cache_teacher:
            run(L, want_q=("attn",), stop_q="attn", want_f=("attn", "att_out", "ffn", "down"), cb=cb1,
                   advance_f=True)
        else:
            run(L, want_q=("attn",), stop_q="attn", want_f=("attn",), stop_f="attn", cb=cb1)
        reg["times"]["P1_pass"] = round(time.time() - t, 1)
        t = time.time()
        solve_dense(1, acc1)
        del acc1
        reg["times"]["P1_dense"] = round(time.time() - t, 1)

        # ═══ P2: input of ssm_out | attn_output ═══
        t = time.time()
        acc2 = DenseAcc(4096, dev, None if full else perm_ssm)
        run(L, want_q=("att_out",), stop_q="att_out", want_f=("att_out",), stop_f="att_out",
               cb=lambda k, sl, dq, df, oq, op: acc2.add(k[0], k[1], dq["att_out"], df["att_out"]))
        reg["times"]["P2_pass"] = round(time.time() - t, 1)
        t = time.time()
        solve_dense(2, acc2)
        del acc2
        reg["times"]["P2_dense"] = round(time.time() - t, 1)

        # ═══ P3: MoE input: shexp gate/up + experts gate/up ═══
        t = time.time()
        sp = P.strato(L)
        Wgu = ST.modulo(sp, "mlp.experts").gate_up_proj.detach()
        Wd_m = ST.modulo(sp, "mlp.experts").down_proj.detach()
        assert tuple(Wgu.shape) == (E, 2 * I, Hd) and tuple(Wd_m.shape) == (E, Hd, I), (Wgu.shape, Wd_m.shape)
        Wgu32 = Wgu.float()
        acc3 = DenseAcc(Hd, dev)
        agu = AccGateUp(E, I, Hd, dev)
        mlp = st.modules[L].mlp
        dl = lav / f"strato_{L:02d}"
        state3 = {"meta": None}
        hm = None
        if tail_live["on"]:                  # FAST2 V1: residual after C's mixer, for P4 and P5
            hm = {k: torch.empty(ids[k].shape[0], ids[k].shape[1], Hd, dtype=hq[k].dtype, device=dev) for k in keys}

        def cb3(k, sl, dq, df, oq, op):
            if hm is not None:
                hm[k][sl] = dq["hmid"]
            h = k[0]
            if state3["meta"] is not None and state3["meta"] != h:
                agu.download(dl / f"gu_{state3['meta']}")
                agu.reset()
            state3["meta"] = h
            acc3.add(h, k[1], dq["ffn"], df["ffn"])
            xq = dq["ffn"].reshape(-1, Hd)
            idx, w = router(mlp, xq)
            pz = S.Plane(idx, E)
            agu.add(xq.float(), df["ffn"].reshape(-1, Hd).float(), w, mlp.top_k, pz.order, pz.offs, Wgu32)
        run(L, want_q=("ffn", "hmid") if hm is not None else ("ffn",), stop_q="ffn", want_f=("ffn",), stop_f="ffn",
               cb=cb3)
        agu.download(dl / f"gu_{state3['meta']}")
        del agu
        gc.collect()
        reg["times"]["P3_pass"] = round(time.time() - t, 1)
        t = time.time()
        solve_dense(3, acc3)
        reg["times"]["P3_dense"] = round(time.time() - t, 1)
        t = time.time()
        reg["experts"]["gate_up"] = solve_experts_gu(L, dl, acc3, Wgu32, new, inject, encode, decode, raw_start,
                                                        es_list, alpha_e, kk, a.batch_experts, JQL, dev, dt, E, I, Hd, guard_e)
        del acc3
        reg["times"]["P3_experts"] = round(time.time() - t, 1)

        # ═══ P4: TRUE inputs of the down projections (new gate/up): ffn_down_shexp + experts down ═══
        t = time.time()
        acc4 = DenseAcc(int(cfg.shared_expert_intermediate_size), dev)
        adn = AccDown(E, I, dev)
        fam_down = st.experts(L).fam["down"]
        cont = {"top8_different": 0}

        def cb4(k, sl, dq, df, oq, op):
            acc4.add(k[0], k[1], dq["down"], df["down"])
            xq = dq["ffn"].reshape(-1, Hd)
            idx, w = router(mlp, xq)
            if mlp.last_idx is not None and not torch.equal(idx, mlp.last_idx):
                cont["top8_different"] += int((idx != mlp.last_idx).any(-1).sum())
            pz = dq["plane"]
            xf = df["ffn"].reshape(-1, Hd).float() if "ffn" in df else None
            adn.add(k[0], dq["a_ord"], xf, w, mlp.top_k, pz.order, pz.offs, Wgu32)
        if hm is not None and not gate_tail(L, hm, fam_down):
            hm = None                        # fail-safe: tail differs from the full pass -> fall back to the full pass
            tail_live["on"] = False
        run(L, want_q=("ffn", "down", "a_ord"), stop_q="a_ord", want_f=("ffn", "down"), stop_f="down",
               cb=cb4, fam_down=fam_down, hmid=hm)
        if cont["top8_different"]:
            raise RuntimeError(f"layer {L}: recomputed top-8 differs from the student's on {cont['top8_different']} tokens")
        reg["times"]["P4_pass"] = round(time.time() - t, 1)
        t = time.time()
        solve_dense(4, acc4)
        del acc4
        reg["times"]["P4_dense"] = round(time.time() - t, 1)
        t = time.time()
        reg["experts"]["down"] = solve_experts_down(L, adn, Wd_m, new, inject, encode, decode, raw_start,
                                                      es_list, alpha_e, kk, a.batch_experts, JQL, dev, dt, E, I, Hd, guard_e)
        del adn, Wgu32, Wgu, Wd_m
        gc.collect()
        reg["times"]["P4_experts"] = round(time.time() - t, 1)

        # ═══ P5: full pass with the new layer; drift against the teacher ═══
        t = time.time()
        der = {h: [0.0, 0.0] for h in "AB"}

        trial5 = {"done": hm is None}

        def cb5(k, sl, dq, df, oq, op):
            if not trial5["done"]:          # FAST2 V1: first batch, tail output == _layer output BIT-EXACT
                trial5["done"] = True
                with torch.no_grad():
                    o_full = st._layer(L, hq[k][sl], ctx[k][sl.stop - sl.start][0])
                ok5 = torch.equal(o_full, oq)
                log(f"tail gate P5 layer {L}: tail output == full pass bit-exact: {ok5}")
                del o_full
                if not ok5:
                    raise RuntimeError(f"layer {L}: tail output differs from the full pass (P5)")
            rif = op if op is not None else hp[k][sl]
            der[k[0]][0] += float((oq.float() - rif.float()).pow(2).sum())
            der[k[0]][1] += float(rif.float().pow(2).sum())
        run(L, cb=cb5, advance_q=True, advance_f=not a.cache_teacher, hmid=hm)
        del hm
        reg["drift_layer"] = {h: v[0] / max(v[1], 1e-300) for h, v in der.items()}
        reg["times"]["P5_pass"] = round(time.time() - t, 1)
        for name, b in new.items():
            np.save(tmpd / "tensors" / f"{name}.npy", b)
        reg["tensors_changed"] = sorted(new)
        reg["seconds"] = round(time.time() - tl, 1)
        if dev.type == "cuda":
            reg["vram_peak_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 1)
        SC.write_json(tmpd / "registry.json", reg)
        (tmpd / "DONE").write_text(time.strftime("%FT%T") + "\n")
        shutil.rmtree(d, ignore_errors=True)
        os.replace(tmpd, d)
        if use_streams:                       # FAST2 V2 (after DONE: the streams are the outputs of a closed layer)
            fingerprints[str(L)] = fingerprint(L)
            try:
                save_streams(L, fingerprints)
            except OSError as ex:
                log(f"streams NOT saved after layer {L}: {ex} (resume will re-run the layers)")
                shutil.rmtree(Path(str(FL) + ".tmp"), ignore_errors=True)
        if os.environ.get("S23_KEEP_WORK") != "1":
            shutil.rmtree(dl, ignore_errors=True)
        nd = sum(1 for v in reg["dense_"].values() if v["chosen"] != "start")
        gu, dn = reg["experts"]["gate_up"], reg["experts"]["down"]
        msg = (f"S23 layer {L} ({'full' if full else 'GDN '}) {reg['seconds'] / 60:.1f} min: dense new {nd}/{len(reg['dense_'])}, "
               f"experts gate {gu['accepted_gate']} up {gu['accepted_up']} down {dn['accepted']} of {len(es_list)}; "
               f"drift A {reg['drift_layer']['A']:.4g} B {reg['drift_layer']['B']:.4g}; times {reg['times']}")
        log(msg)
        with open(out / "passata.jsonl", "a") as f:
            f.write(json.dumps({k: reg[k] for k in ("layer", "type_", "drift_layer", "times", "seconds",
                                                     "tensors_changed")}) + "\n")
        done_now = len([x for x in layers if (out / "layers" / f"strato_{x:02d}" / "DONE").exists()])
        remainder_s = (len(layers) - done_now) * (time.time() - t_start) / max(1, done_now - L0)
        if L in (0, 1) or L % 10 == 9:
            row_cycles("ACTIVE", f"{msg[:300]}; estimated end {time.strftime('%d/%m %H:%M', time.localtime(time.time() + remainder_s))}")
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    if layers == list(range(nL)) and all((out / "layers" / f"strato_{L:02d}" / "DONE").exists() for L in range(nL)):
        (out / "DONE_SWEEP").write_text(f"{time.strftime('%FT%T')} {(time.time() - t_start) / 60:.1f} min\n")
    log(f"sweep: layers {remainder[0] if remainder else '-'}..{layers[-1]} in {(time.time() - t_start) / 60:.1f} min; tokens {tok_tot}")
    return 0


def _batches_experts(es_list, n):
    for i in range(0, len(es_list), n):
        yield es_list[i:i + n]


def solve_experts_gu(L, dl, acc3, Wgu32, new, inject, encode, decode, raw_start, es_list, alphas, kk, batch,
                       jarq_batch, dev, dt, E, I, Hd, gd=None):
    """S3 gate/up (E1): gate and up share the SAME input H, so they are solved as a single matrix [2I, H] (GPTQ and JARQ
    are row-independent: identical to two separate calls).
    2026-10-14 G1/G2: one candidate per alpha in `alphas` (QEP, 0 = symmetric GPTQ); per expert and per tensor (gate, up)
    the winner is the minimum SYMMETRIC output error on half B among the candidates with energy ||Q x||/||W x|| in
    [emin, emax], and it is accepted only if it beats the start."""
    gd = gd or {"emin": 0.9, "emax": 1.1, "weight_kw": 1.0}
    emin, emax = gd["emin"], gd["emax"]
    alphas = list(alphas) if isinstance(alphas, (list, tuple)) else [float(alphas)]
    t0 = time.time()
    tid = 45
    ng = Hd // 128
    names = {f: f"blk.{L}.ffn_{f}_exps.weight" for f in ("gate", "up")}
    raw = {f: np.array(raw_start(n)) for f, n in names.items()}            # copy: accepted experts are replaced in it
    st = {}
    for h in "AB":
        Hs, Mxs, n = acc3.pool(h, gd["weight_kw"])
        st[h] = {"Hs": Hs.to(dev, dt), "Mxs": Mxs.to(dev, dt), "Gq": np.load(dl / f"gu_{h}" / "Gq.npy", mmap_mode="r"),
                 "C": np.load(dl / f"gu_{h}" / "C.npy", mmap_mode="r"), "count": np.load(dl / f"gu_{h}" / "count.npy"),
                 "weight": np.load(dl / f"gu_{h}" / "weight.npy")}
    acc = {"gate": 0, "up": 0}
    dJ = {"gate": [], "up": []}
    en_chosen = {"gate": [], "up": []}
    en_part = {"gate": [], "up": []}
    alpha_chosen = {"gate": {}, "up": {}}
    ref_energy = {"gate": 0, "up": 0}
    fallbacks = 0
    for ee in _batches_experts(es_list, batch):
        b = len(ee)
        W = Wgu32[ee].to(dt)                                                      # [b, 2I, H]
        Hk, Ck = {}, {}
        for h in "AB":
            s = st[h]
            Gq = unpack_batch(torch.from_numpy(np.asarray(s["Gq"][ee])), Hd, dev, dt)
            Cc = torch.from_numpy(np.asarray(s["C"][ee])).to(dev, dt)
            n = torch.from_numpy(s["count"][ee].astype(np.float64)).to(dev, dt).view(b, 1, 1)
            p = torch.from_numpy(s["weight"][ee]).to(dev, dt).view(b, 1, 1)
            pz = torch.where(p > 0, p, torch.ones_like(p))
            He = torch.where(p > 0, Gq / pz, s["Hs"].expand(b, Hd, Hd))
            Cs = W @ s["Mxs"]
            Ce = torch.where(p > 0, Cc / pz, Cs)
            Hk[h] = (n * He + kk * s["Hs"]) / (n + kk)
            Ck[h] = (n * Ce + kk * Cs) / (n + kk)
            del Gq, Cc, He, Ce, Cs
        HmA = RS.rotate_H_batch(Hk["A"])
        Hmn, dbar = RS.normalize(HmA)
        Minv, U, nr = U_Minv_batch(Hmn)
        fallbacks += nr
        Wm = K.rotate(W)
        Wstar = (torch.bmm(K.rotate(Ck["A"]), Minv) / dbar.view(b, 1, 1)            # QEP a=1 (damp 1 % = the GPTQ one)
                 if any(x != 0 for x in alphas) else None)
        HmB = RS.rotate_H_batch(Hk["B"])
        T0 = torch.empty(b, 2 * I, Hd, dtype=torch.int8)
        S0 = torch.empty(b, 2 * I, ng, dtype=torch.float64)
        for j, e in enumerate(ee):
            for q, f in enumerate(("gate", "up")):
                t_, s_ = decode(raw[f][e], tid, Hd)
                T0[j, q * I:(q + 1) * I], S0[j, q * I:(q + 1) * I] = t_, s_
        Q0 = AL.rebuild(T0.to(dev), S0.to(dev, dt))
        J0 = RS.J_sim_batch(Q0, HmB, Wm)
        N0 = RS.N_batch(Q0, HmB)
        den = RS.N_batch(Wm, HmB)                                               # ||W x_q||^2 per row
        RR = [slice(q * I, (q + 1) * I) for q in range(2)]
        dd = {f: den[:, RR[q]].sum(1).clamp_min(1e-300) for q, f in enumerate(("gate", "up"))}
        mig = {f: J0[:, RR[q]].sum(1).clone() for q, f in enumerate(("gate", "up"))}     # to beat: the start
        sel = {f: [None] * b for f in ("gate", "up")}
        ep_ = {}
        for q, f in enumerate(("gate", "up")):
            ep = (N0[:, RR[q]].sum(1) / dd[f]).sqrt()
            ep_[f] = ep
            en_part[f] += [float(x) for x in ep]
        kw_jq = {"cache": {}} if getattr(jarq_batch, "use_cache", False) else {}     # FAST L2: one cache per batch (same Hmn)
        for alpha in alphas:
            Wa = Wm if alpha == 0 else Wm + alpha * (Wstar - Wm)
            T, Sx = AL.gptq_bil_batch(Wa, U, ng)
            T2, S2, _ = jarq_batch(Wa, Hmn, T.unsqueeze(1), Sx.unsqueeze(1), rounds=3, nu2=1e-4, **kw_jq)
            T2, S2 = T2[:, 0], S2[:, 0]
            Q1 = AL.rebuild(T2, S2.to(dt))
            J1 = RS.J_sim_batch(Q1, HmB, Wm)
            N1 = RS.N_batch(Q1, HmB)
            for q, f in enumerate(("gate", "up")):
                r = RR[q]
                j1 = J1[:, r].sum(1)
                en = (N1[:, r].sum(1) / dd[f]).sqrt()
                ok_en = (en >= emin) & (en <= emax)
                if gd.get("relative", True):                       # no farther from 1 than the start (energy_ok)
                    ok_en = ok_en | ((en > 0) & (en.log().abs() <= ep_[f].log().abs()))
                ref_energy[f] += int((~ok_en & (j1 < mig[f])).sum())
                ok = (j1 < mig[f]) & torch.isfinite(j1) & ok_en
                for j in range(b):
                    if bool(ok[j]):
                        mig[f][j] = j1[j]
                        sel[f][j] = (alpha, T2[j, r].clone(), S2[j, r].clone(), float(en[j]))
            del Wa, T, Sx, T2, S2, Q1, J1, N1
        for q, f in enumerate(("gate", "up")):
            j0 = J0[:, RR[q]].sum(1)
            for j, e in enumerate(ee):
                dJ[f].append(float((mig[f][j] - j0[j]) / dd[f][j]))
                if sel[f][j] is not None:
                    alpha, Tj, Sj, enj = sel[f][j]
                    raw[f][e] = encode(Tj, Sj, tid, Hd)
                    acc[f] += 1
                    en_chosen[f].append(enj)
                    alpha_chosen[f][f"{alpha:g}"] = alpha_chosen[f].get(f"{alpha:g}", 0) + 1
        del W, Hk, Ck, HmA, Hmn, Minv, U, Wm, Wstar, HmB, Q0, J0, N0, sel
    for f, n in names.items():
        if acc[f]:
            new[n] = raw[f]
            inject(L, f, raw[f])
    md = {f: (float(np.median(v)) if v else None) for f, v in dJ.items()}
    mde = {f: (float(np.median(v)) if v else None) for f, v in en_chosen.items()}
    mdp = {f: (float(np.median(v)) if v else None) for f, v in en_part.items()}
    log(f"  experts gate/up layer {L}: accepted gate {acc['gate']} up {acc['up']} of {len(es_list)}; dSim_B/|Wx|^2 median "
        f"{md}; alpha chosen {alpha_chosen}; median energy chosen {mde} start {mdp}; rejected for energy "
        f"{ref_energy}; factorization fallbacks {fallbacks}; {time.time() - t0:.0f} s")
    return {"accepted_gate": acc["gate"], "accepted_up": acc["up"], "dJB_rel_median": md, "metric": "symmetric_B+energy",
            "alphas": alphas, "alpha_chosen": alpha_chosen, "energy_median_chosen": mde, "energy_median_start": mdp,
            "rejected_energy": ref_energy, "k": kk,
            "fallbacks_factorization": fallbacks, "seconds": round(time.time() - t0, 1),
            "count_A_median": float(np.median(st["A"]["count"])), "count_A_min": int(st["A"]["count"].min())}


def solve_experts_down(L, adn, Wd_m, new, inject, encode, decode, raw_start, es_list, alphas, kk, batch,
                         jarq_batch, dev, dt, E, I, Hd, gd=None):
    """S3 down (TQ1_0, group 256, no rotation) on the TRUE input of C's down projection, with gate/up already new.
    2026-10-14 G1/G2: as for gate/up (candidates per alpha, symmetric guard + energy, per expert)."""
    gd = gd or {"emin": 0.9, "emax": 1.1, "weight_kw": 1.0}
    emin, emax = gd["emin"], gd["emax"]
    alphas = list(alphas) if isinstance(alphas, (list, tuple)) else [float(alphas)]
    t0 = time.time()
    tid = 34
    ng = I // 256
    name = f"blk.{L}.ffn_down_exps.weight"
    raw = np.array(raw_start(name))
    st = {}
    for h in "AB":
        pt = adn.weight[h].sum().clamp_min(1e-30)
        st[h] = {"Hs": (adn.Hd[h].double().sum(0) / pt).to(dt), "Ms": (adn.Md[h].double().sum(0) / pt).to(dt)}
    acc, dJ, fallbacks = 0, [], 0
    en_chosen, en_part, alpha_chosen, ref_energy = [], [], {}, 0
    for ee in _batches_experts(es_list, batch):
        b = len(ee)
        W = Wd_m[ee].to(dev, dt)                                                  # [b, H, I]
        Hk, Ck = {}, {}
        for h in "AB":
            n = adn.count[h][ee].to(dt).view(b, 1, 1)
            p = adn.weight[h][ee].to(dt).view(b, 1, 1)
            pz = torch.where(p > 0, p, torch.ones_like(p))
            He = torch.where(p > 0, adn.Hd[h][ee].to(dt) / pz, st[h]["Hs"].expand(b, I, I))
            Me = torch.where(p > 0, adn.Md[h][ee].to(dt) / pz, st[h]["Ms"].expand(b, I, I))
            Hk[h] = (n * He + kk * st[h]["Hs"]) / (n + kk)
            Ck[h] = W @ ((n * Me + kk * st[h]["Ms"]) / (n + kk))
        Hn, dbar = RS.normalize(Hk["A"])
        Minv, U, nr = U_Minv_batch(Hn)
        fallbacks += nr
        Wstar = torch.bmm(Ck["A"], Minv) / dbar.view(b, 1, 1) if any(x != 0 for x in alphas) else None
        T0 = torch.empty(b, Hd, I, dtype=torch.int8)
        S0 = torch.empty(b, Hd, ng, dtype=torch.float64)
        for j, e in enumerate(ee):
            T0[j], S0[j] = decode(raw[e], tid, I)
        Q0 = AL.rebuild(T0.to(dev), S0.to(dev, dt))
        HB_ = Hk["B"]
        j0 = RS.J_sim_batch(Q0, HB_, W).sum(1)
        den = RS.N_batch(W, HB_).sum(1).clamp_min(1e-300)
        ep_d = (RS.N_batch(Q0, HB_).sum(1) / den).sqrt()
        en_part += [float(x) for x in ep_d]
        mig = j0.clone()
        sel = [None] * b
        kw_jq = {"cache": {}} if getattr(jarq_batch, "use_cache", False) else {}     # FAST L2: one cache per batch (same Hn)
        for alpha in alphas:
            Wa = W if alpha == 0 else W + alpha * (Wstar - W)
            T, Sx = AL.gptq_bil_batch(Wa, U, ng)
            T2, S2, _ = jarq_batch(Wa, Hn, T.unsqueeze(1), Sx.unsqueeze(1), rounds=3, nu2=1e-4, **kw_jq)
            T2, S2 = T2[:, 0], S2[:, 0]
            Q1 = AL.rebuild(T2, S2.to(dt))
            j1 = RS.J_sim_batch(Q1, HB_, W).sum(1)
            en = (RS.N_batch(Q1, HB_).sum(1) / den).sqrt()
            ok_en = (en >= emin) & (en <= emax)
            if gd.get("relative", True):
                ok_en = ok_en | ((en > 0) & (en.log().abs() <= ep_d.log().abs()))
            ref_energy += int((~ok_en & (j1 < mig)).sum())
            ok = (j1 < mig) & torch.isfinite(j1) & ok_en
            for j in range(b):
                if bool(ok[j]):
                    mig[j] = j1[j]
                    sel[j] = (alpha, T2[j].clone(), S2[j].clone(), float(en[j]))
            del Wa, T, Sx, T2, S2, Q1
        for j, e in enumerate(ee):
            dJ.append(float((mig[j] - j0[j]) / den[j]))
            if sel[j] is not None:
                alpha, Tj, Sj, enj = sel[j]
                raw[e] = encode(Tj, Sj, tid, I)
                acc += 1
                en_chosen.append(enj)
                alpha_chosen[f"{alpha:g}"] = alpha_chosen.get(f"{alpha:g}", 0) + 1
        del W, Hk, Ck, Hn, Minv, U, Wstar, Q0, sel
    if acc:
        new[name] = raw
        inject(L, "down", raw)
    md = float(np.median(dJ)) if dJ else None
    mde = float(np.median(en_chosen)) if en_chosen else None
    mdp = float(np.median(en_part)) if en_part else None
    log(f"  experts down layer {L}: accepted {acc} of {len(es_list)}; dSim_B/|Wx|^2 median {md}; alpha chosen "
        f"{alpha_chosen}; median energy chosen {mde} start {mdp}; rejected for energy {ref_energy}; "
        f"fallbacks {fallbacks}; {time.time() - t0:.0f} s")
    return {"accepted": acc, "dJB_rel_median": md, "metric": "symmetric_B+energy", "alphas": alphas,
            "alpha_chosen": alpha_chosen, "energy_median_chosen": mde, "energy_median_start": mdp,
            "rejected_energy": ref_energy, "k": kk, "fallbacks_factorization": fallbacks,
            "seconds": round(time.time() - t0, 1), "count_A_median": float(adn.count["A"].double().median())}


if __name__ == "__main__":
    sys.exit(main())

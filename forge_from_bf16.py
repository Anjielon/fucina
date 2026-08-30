#!/usr/bin/env python3
"""FORGE FROM BF16 — two joint ternary planes + GPTQ, from the full bf16.

    W  ≈  d1⊙T1 + d2⊙T2      T ∈ {-1,0,+1},  1.69 bits/weight per plane

This is the entry point that produced the case study. It differs from
`forge_gguf.py` only in where the weights how_ from: that one dequantizes an
already-quantized GGUF, this one reads the original bf16 shards. Use this when
the original checkpoint is available; use the other when it is not.

⭐ Why not llama.cpp's converter. We must emit TWO tensors per expert matrix
   (`..._exps` and `..._exps2`), and the converter quantizes LAZILY: at the
   point where we could say "this is the second plane", the computation has
   not happened yet. So the GGUF is written by hand, copying metadata from a
   known-good GGUF of the same model.

⭐ Where each piece comes from
   - experts (97% of the model): from the **bf16 checkpoint**, quantized here
   - everything else (attention, embeddings, router, shared experts): copied
     verbatim from an existing quantization of the same model
   - **true Hessians** for all 60 layers, measured on the running model with
     16,640 samples each (mean anisotropy alpha = 1.17)

⚠️ TWO-PASS WRITE: all tensor metadata first, then the data one tensor at a
   time. `GGUFWriter.add_tensor` would hold every tensor in RAM at once.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, os.environ.get("GGUF_PY", os.path.expanduser("~/build-llamacpp/gguf-py")))
sys.path.insert(0, str(Path(__file__).parent))
import gguf
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as T
from safetensors import safe_open
from concurrent.futures import ThreadPoolExecutor
import ternary_gpu as G, two_planes as D

NAS = Path(os.environ.get("FP_CHECKPOINT_DIR", "/mnt/checkpoints/Ornith-1.5-397B"))
BASE = "/mnt/lavoro/ODINO-IQ1_M/Ornith-1.5-397B-IQ1_M-00001-of-00003.gguf"
HESS = Path("/mnt/models/gguf/odino-hessiane")
IMATRIX = Path(os.path.expanduser("~/odino-lab/imatrix/Ornith-1.5-397B-imatrix.gguf"))
K_HOT = int(os.getenv("FORGE_HOT", "16"))   # experts col 2° plan (su 512)
OUTPUT = os.getenv("FORGE_OUT", "/mnt/lavoro/ODINO-v31/ODINO-397B-v31.gguf")
GPTQ = os.getenv("FORGE_GPTQ", "1") == "1"
ONLY_LAYERS = int(os.getenv("FORGE_ONLY_LAYERS", "0"))
# ⛔ TWO_PLANE_LAYERS (57,58,59) was a residue NEVER USED: v3.1 forged the
# 2nd plane on ALL layers and the recipe stripped it afterwards. Depth is now
# a SINGLE PREDICATE feeding every coupled site (plane in the plan, joint
# pair, router permutation) — the same
# disciplina di forge_gguf.has_p2.
P2_LAYERS = os.getenv("FORGE_P2_LAYERS", "")        # es. "44-59"; vuoto = tutti
FOEM_BETA = float(os.getenv("FORGE_FOEM_BETA", "0"))
_IMPACT_JSON = os.getenv("FORGE_IMPACT_JSON", "")   # classifica impact pre-calcolata
_IMPACT = json.load(open(_IMPACT_JSON)) if _IMPACT_JSON and Path(_IMPACT_JSON).exists() else {}

def has_p2(L: int) -> bool:
    if not K_HOT:
        return False
    if not P2_LAYERS:
        return True
    lo, hi = (int(x) for x in P2_LAYERS.split("-"))
    return lo <= L <= hi

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

# ⛔ The per-tensor vault is DISABLED: it would need ~90 GiB of scratch that
#    this machine does not have. And the out-of-memory kill that motivated it
#    was caused by GPU jobs running in parallel — on unified memory the VRAM
#    *is* system memory and counts against the cgroup — not by the forge.
#
# ZERO LOSS AT ZERO EXTRA SPACE. An OOM at the thirteenth layer once threw
# away ninety minutes. The insight that removed the need for a vault: a GGUF is
# written SEQUENTIALLY, so it is enough to record, after each tensor, which
# byte we reached; on resume, reopen the file, seek there, continue. The write
# plan is deterministic — same order, same metadata — so the n-th tensor always
# lands at the same offset.
JOURNAL = None
VAULT = None

# ⭐⭐ FRAGILE TENSORS, OVERRIDDEN AT Q8_0 — the best lever per byte we found.
# Measured on the quantization we were copying from: `attn_gate` (45 layers of
# 45) is 47.24% wrong and half of `ssm_out` 50.69%. These are the
# LINEAR-ATTENTION projections, which Quamba identifies as the fragile point of
# hybrid models, and this architecture has 45 such layers out of 60. At Q8_0
# they are 0.55% wrong.
# They carry 20% of the parameters ACTIVE per token (of 512 experts only ten
# fire), so model-level output error drops from 25.44% to 17.14% — **-32.6%** —
# for **+3.0 GiB on 92.7**. That is 2.77 points per GiB, better than any other
# lever in the registry.
# Fetched from a published Q8_0 of the same model with HTTP byte-range requests
# (4 GiB transferred instead of 392) to avoid re-deriving the converter's
# attention-head reordering.
Q8DIR = Path(os.getenv("FORGE_Q8_DIR", "/mnt/models/gguf/odino-q8"))
def q8_override(name):
    f = Q8DIR / f"{name}.bin"
    return f if f.exists() else None
def save_to_vault(name, a, b):
    return
    """Persist each forged tensor as soon as it is ready (currently disabled).
    An OOM at the fifth hour would otherwise cost five hours; this way it costs
    one tensor. The packs are already compressed (54 bytes per 256 weights), so
    the vault weighs about as much as the finished model — which is exactly why
    it is off by default."""
    VAULT.mkdir(parents=True, exist_ok=True)
    np.save(VAULT / f"{name}.p1.npy", a)
    if b is not None: np.save(VAULT / f"{name}.p2.npy", b)
    (VAULT / f"{name}.ok").touch()

def already_done(name):
    return False
    return (VAULT / f"{name}.ok").exists()

def reload_from_vault(name):
    a = np.load(VAULT / f"{name}.p1.npy")
    f2 = VAULT / f"{name}.p2.npy"
    return a, (np.load(f2) if f2.exists() else None)

# ── impacchettamento TQ1_0 (identico a llama.cpp) ────────────────────────
def pack(d: np.ndarray, q: np.ndarray) -> np.ndarray:
    """d:(n,1) q:(n,256) in {-1,0,1} -> (n,54) uint8, formato TQ1_0.

    ⚡ Chunked on the GPU: in numpy this packing (base 3, five values per
    byte) cost 60-100 s for a 2.1-billion-weight tensor — a large part of the
    ten minutes measured per tensor. In torch on the GPU it is seconds.
    """
    n_r = q.shape[0]
    out = np.empty((n_r, 54), np.uint8)
    step = 8_000_000
    p5 = torch.tensor([81,27,9,3,1], dtype=torch.int32, device="cuda").view(1,1,5,1)
    p4 = torch.tensor([81,27,9,3],  dtype=torch.int32, device="cuda").view(1,1,4,1)
    for i in range(0, n_r, step):
        j = min(i+step, n_r); m = j - i
        qs = torch.from_numpy(q[i:j].astype(np.int8)).to("cuda", torch.int32) + 1
        a = (qs[:, :160].reshape(m,-1,5,32) * p5).sum(-2).reshape(m,-1)
        b = (qs[:, 160:240].reshape(m,-1,5,16) * p5).sum(-2).reshape(m,-1)
        c = (qs[:, 240:].reshape(m,-1,4,4) * p4).sum(-2).reshape(m,-1)
        t = torch.cat([a,b,c], -1)
        t = ((t*256 + 242)//243).to(torch.uint8)
        # 32 + 16 + 4 = 52 byte di dati, poi 2 di scala float16 = 54
        out[i:j, :52] = t.cpu().numpy()
        out[i:j, 52:] = d[i:j].astype(np.float16).view(np.uint8)
    return out

def hot_first_permutation(L: int, k: int, E: int = 512) -> np.ndarray:
    """[hot experts by routing frequency] + [everything else in order].

    This is the ordering the ENGINE assumes: the second plane covers the
    expert PREFIX 0..k-1.

    ⛔ The first build shipped WITHOUT this reordering — the engine assumed it,
    the forge did not apply it, and the second plane was summed onto the wrong
    experts, producing a model that talked nonsense. The file is now born
    permuted: experts reordered here, router rows permuted in the copy branch.
    A convention shared by two components must be an assert, not a comment."""
    hot = [int(x) for x in hot_experts(L, k)]
    perm = np.array(hot + [e for e in range(E) if e not in set(hot)])
    assert len(perm) == E and len(set(perm.tolist())) == E
    return perm


def reorder_experts(packs: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """packs (E*nb, 54) uint8 → experts riordinati secondo perm."""
    E = len(perm)
    v = packs.reshape(E, -1, 54)
    return np.ascontiguousarray(v[perm]).reshape(-1, 54)


_HOT = {}
def hot_experts(L: int, k: int):
    """The k hottest experts of layer L, from the imatrix routing `counts`.

    ⭐ Measured: the hottest 3% capture ~18% of the traffic, and giving them
    the second plane takes the error from 37.4% to 32.0% for 2.3 GiB.
    Allocating per LAYER instead would give 36.1% — traffic concentrates among
    experts far more than sensitivity concentrates among layers.
    """
    if _IMPACT and str(L) in _IMPACT:
        return np.array(_IMPACT[str(L)][:k], int)
    if not _HOT:
        r = GGUFReader(str(IMATRIX))
        for t in r.tensors:
            if t.name.endswith(".ffn_gate_exps.weight.counts"):
                Lx = int(t.name.split(".")[1])
                _HOT[Lx] = np.argsort(np.array(t.data).astype(np.float64).ravel())[::-1]
    return _HOT[L][:k].copy()


def hchol(layer: int):
    f = HESS / f"H_{layer:02d}.npy"
    if not GPTQ or not f.exists(): return None
    return torch.from_numpy(D.prepare_hessian(np.load(f).astype(np.float64)).astype(np.float32))

# ── mappa: tensore GGUF -> (file safetensors, chiave, slice_) ─────────────
def expert_map():
    idx = json.load(open(NAS/"model.safetensors.index.json"))["weight_map"]
    m = {}
    for L in range(60):
        p = f"model.language_model.layers.{L}.mlp.experts"
        m[f"blk.{L}.ffn_gate_exps.weight"] = (idx[f"{p}.gate_up_proj"], f"{p}.gate_up_proj", "gate")
        m[f"blk.{L}.ffn_up_exps.weight"]   = (idx[f"{p}.gate_up_proj"], f"{p}.gate_up_proj", "up")
        m[f"blk.{L}.ffn_down_exps.weight"] = (idx[f"{p}.down_proj"],    f"{p}.down_proj",    None)
    return m

class Lettore:
    """Read ONE expert matrix at a time, prefetching the next.

    ⛔ COSA HO SBAGLIATO PRIMA (OOM misurato a 20G+4G di swap): tenevo
       BOTH halves of the fused tensor (gate and up, 4.3 GB each in float16)
       plus the file mapping (8.6 GB) plus the prefetch of the next one. Four
       copies on a machine with 30 GB of host RAM.

    ✅ Now: only the needed half is read. The fused tensor ends up read twice,
       but the read hides behind the computation and memory stays under 12 GB.
       Reading twice beats dying.
    ⚠️ float16, not float32: from bf16 this is lossless (10 mantissa bits
       contro 7) e dimezza l'occupazione.
    """
    def __init__(self, experts):
        self.experts = experts
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.futuro = None
        self.cassa = {}

    def _carica(self, name, flussi: int = 2):
        """⛔ MEASURED: the share gives ~50 MB/s IN TOTAL, and the
        parallelismo NON aiuta (1 flusso 48,2 · 4 flussi 51,3). Con SEI flussi
        starve each other: an external reader dropped to 4.8 MB/s and I/O
        pressure hit 98% — the machine stalled waiting on the disk.
        Two streams: one reads while the other prefetches, without contention."""
        """Read the fused tensor ONCE and return BOTH halves.

        ⛔ IL DIFETTO CHE MI COSTAVA IL DOPPIO (misurato: 7,8 M/s → 13,65 ore).
           `gate` e `up` stanno nello stesso tensore `gate_up_proj`, e leggerne
           a non-contiguous slice still forces whole rows to be pulled in.
           Reading it twice therefore reads FOUR times what is needed.
           Il modello e' 740 GB di bf16: a 110 MB/s sono 1,9 ore di lettura
           at best, which doubled exceeds the 5.6 hours of computation and
           becomes the bottleneck.
        ✅ Una lettura sola, tutte e due le meta' tenute in float16 (8,6 GB).
           Ogni flusso scrive dentro l'array gia' allocato: niente copie.
        """
        file, chiave, meta = self.experts[name]
        with safe_open(str(NAS/file), "pt") as f:
            shape_ = f.get_slice(chiave).get_shape()
        E = shape_[0]
        if meta is None:
            out = {name: np.empty((E, shape_[1], shape_[2]), np.float16)}
            def pezzo(a, b):
                with safe_open(str(NAS/file), "pt") as f:
                    out[name][a:b] = f.get_slice(chiave)[a:b].to(torch.float16).numpy()
        else:
            n = shape_[1] // 2
            base = name.rsplit(".ffn_", 1)[0]
            ng, nu = f"{base}.ffn_gate_exps.weight", f"{base}.ffn_up_exps.weight"
            out = {ng: np.empty((E, n, shape_[2]), np.float16),
                   nu: np.empty((E, n, shape_[2]), np.float16)}
            def pezzo(a, b):
                with safe_open(str(NAS/file), "pt") as f:
                    t = f.get_slice(chiave)[a:b].to(torch.float16).numpy()
                    out[ng][a:b] = t[:, :n, :]; out[nu][a:b] = t[:, n:, :]
        step = (E + flussi - 1) // flussi
        with ThreadPoolExecutor(max_workers=flussi) as ex:
            list(ex.map(lambda a: pezzo(a, min(a + step, E)), range(0, E, step)))
        return out

    def prendi(self, name, prossimo=None):
        """Return the requested tensor; the fused twin stays cached."""
        if name in self.cassa:
            return self.cassa.pop(name)
        if self.futuro and self.futuro[0] == name:
            d = self.futuro[1].result(); self.futuro = None
        else:
            if self.futuro:
                self.cassa.update(self.futuro[1].result()); self.futuro = None
            d = self._carica(name)
        W = d.pop(name); self.cassa.update(d)
        # Prefetch ONLY when the next tensor is a `down` (4.3 GB): the current
        # fused tensor (8.6) plus down (4.3) is 13 GB, within the ceiling.
        # Prefetching another FUSED tensor (8.6 + 8.6) was the measured OOM.
        # (this line intentionally left as part of the block above)
        if prossimo and "down" in prossimo and not self.futuro and prossimo not in self.cassa:
            self.futuro = (prossimo, self.pool.submit(self._carica, prossimo))
        return W



def read_experts(file, chiave, meta) -> np.ndarray:
    """-> (n_esperti, n_out, n_in) float32"""
    with safe_open(str(NAS/file), "pt") as f:
        sl = f.get_slice(chiave)
        shape_ = sl.get_shape()          # gate_up: (512, 2048, 4096) · down: (512, 4096, 1024)
        if meta == "gate":  W = sl[:, :shape_[1]//2, :]
        elif meta == "up":  W = sl[:, shape_[1]//2:, :]
        else:               W = sl[:]
    return W.float().numpy()

def main():
    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    src = GGUFReader(BASE)
    base_tensors = {t.name: t for t in src.tensors}
    experts = expert_map()
    log(f"base: {len(base_tensors)} tensors · experts da forgiare: {len(experts)} · GPTQ={GPTQ}")

    # every piece of the base model (its tensors are split across 3 files)
    for n in (2,3):
        p = BASE.replace("00001", f"{n:05d}")
        for t in GGUFReader(p).tensors: base_tensors.setdefault(t.name, t)
    log(f"base model: {len(base_tensors)} tensors")

    w = GGUFWriter(path=None, arch="qwen35moe")
    for f in src.fields.values():                # metadati: copiati pari pari
        if f.name.startswith(("GGUF.", "split.")): continue
        try: w.add_key_value(f.name, f.contents(), f.types[0], sub_type=f.types[-1] if len(f.types)>1 else None)
        except Exception: pass

    if K_HOT: w.add_uint32("qwen35moe.expert_count2", K_HOT)
    plan = []          # (name, forma_gguf, kind_, sorgente)
    for name, t in sorted(base_tensors.items()):
        if name in experts:
            L = int(name.split(".")[1])
            if ONLY_LAYERS and L >= ONLY_LAYERS: continue
            shape_ = [int(x) for x in t.shape]
            # ⭐ SECOND PLANE only on the most SENSITIVE layers.
            # Measured: the Hessian trace varies 35.2x across layers, and the
            # ten most sensitive are EXACTLY the last ten. The byte budget
            # (94 GiB minus 89.4 for the single plane) pays for three of them.
            # The engine treats `_exps2` as TENSOR_NOT_REQUIRED: layers without
            # it skip the branch, layers with it sum inside build_moe_ffn.
            plan.append((name, shape_, T.TQ1_0, ("forgia", name)))
            if has_p2(L):      # second plane: same tensor, only K experts
                f2 = list(shape_); f2[2] = K_HOT
                plan.append((name.replace(".weight", "2.weight"), f2, T.TQ1_0, ("forgia", name)))
        else:
            if ONLY_LAYERS and name.startswith("blk.") and int(name.split(".")[1]) >= ONLY_LAYERS: continue
            tipo_v = T.Q8_0 if q8_override(name) is not None else t.tensor_type
            plan.append((name, [int(x) for x in t.shape], tipo_v, ("copy", name)))

    for name, shape_, kind_, _ in plan:
        # ⚠️ add_tensor_info wants the shape in BYTES when the dtype is uint8
        #    (it converts back to elements via quant_shape_from_byte_shape).
        #    For TQ1_0 a row of 256 elements becomes 54 bytes.
        if kind_ == T.TQ1_0:
            # GGUFReader reports the shape in GGUF order (ne0 first); the
            # writer wants numpy order (reversed), and the LAST dimension is
            # the row to convert to bytes: 256 elements -> 54 bytes.
            fb = list(shape_)[::-1]
            fb[-1] = fb[-1] // 256 * 54
            w.add_tensor_info(name, fb, np.dtype(np.uint8), int(np.prod(fb)), raw_dtype=kind_)
        elif q8_override(name) is not None:
            # ⭐ tensore PESCATO a Q8_0: 32 elementi -> 34 byte, shape_ in
            #    numpy order with the last dimension converted to bytes
            fb = list(shape_)[::-1]
            fb[-1] = fb[-1] // 32 * 34
            nb = (Q8DIR / f"{name}.bin").stat().st_size
            assert nb == int(np.prod(fb)), f"{name}: pescati {nb} byte, attesi {int(np.prod(fb))}"
            w.add_tensor_info(name, fb, np.dtype(np.uint8), nb, raw_dtype=T.Q8_0)
        else:
            t = base_tensors[name]
            w.add_tensor_info(name, list(np.asarray(t.data).shape), np.asarray(t.data).dtype,
                              int(np.asarray(t.data).nbytes), raw_dtype=kind_)
    giornale = Path(str(OUTPUT) + ".giornale")
    riprendi_da = 0
    if giornale.exists() and Path(OUTPUT).exists():
        try:
            idx, pos = (int(x) for x in giornale.read_text().split())
            if Path(OUTPUT).stat().st_size >= pos:
                riprendi_da = idx + 1
                log(f"⭐ RIPRESA: {riprendi_da} tensors gia' scritti, riparto dal byte {pos}")
        except Exception as e:
            log(f"giornale illeggibile ({e}), riparto da capo")
    w.open_output_file(Path(OUTPUT))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    log(f"intestazione scritta: {len(plan)} tensors → {OUTPUT}")
    if riprendi_da:
        f = w.fout[0]
        f.flush(); f.seek(pos); f.truncate(pos)
        log(f"   posizionato a {pos/2**30:.2f} GiB, salto i primi {riprendi_da} tensors")

    fatti = {}
    lettore = Lettore(experts)
    t0 = time.time(); pesi_fatti = 0
    for i, (name, shape_, kind_, (how_, orig)) in enumerate(plan):
        if i < riprendi_da:
            continue
        if how_ == "copy":
            f8 = q8_override(orig)
            if f8 is not None:
                w.write_tensor_data(np.fromfile(f8, dtype=np.uint8))
            elif K_HOT and orig.endswith(".ffn_gate_inp.weight") and has_p2(int(orig.split(".")[1])):
                # ⛔ The hot-first expert reordering must be FOLDED INTO the
                # router rows as well, or the router keeps addressing the old
                # expert indices. Row e is the logit of expert e.
                Lr = int(orig.split(".")[1])
                rt = np.asarray(base_tensors[orig].data)
                assert rt.ndim == 2 and rt.shape[0] >= K_HOT, (orig, rt.shape)
                w.write_tensor_data(np.ascontiguousarray(rt[hot_first_permutation(Lr, K_HOT, rt.shape[0])]))
            else:
                w.write_tensor_data(np.asarray(base_tensors[orig].data))
            w.fout[0].flush(); giornale.write_text(f"{i}\n{w.fout[0].tell()}\n")
            continue
        secondo = name.endswith("2.weight")
        if orig not in fatti and already_done(orig):
            fatti[orig] = reload_from_vault(orig)
            log(f"  [{i+1}/{len(plan)}] {orig} · ripreso dalla cassa")
        if orig not in fatti:                     # forgia i due piani insieme
            file, chiave, meta = experts[orig]
            L = int(orig.split(".")[1])
            prossimo = None
            for j in range(i+1, len(plan)):
                if plan[j][3][0] == "forgia" and plan[j][3][1] != orig:
                    prossimo = plan[j][3][1]; break
            W = lettore.prendi(orig, prossimo)             # (E, out, in) float16
            E, o, ind = W.shape
            # ⛔ GPTQ applies only to gate and up, whose input is the
            #    residual (4096) whose Hessian we measured. `down`
            #    prende in ingresso l'INTERMEDIO dell'esperto (1024): H diversa,
            #    which attn_post_norm cannot capture. There we use the two
            #    joint planes without propagation.
            Hc = hchol(L) if ind == 4096 else None
            # ⭐⭐ LA LEVA DEL 26/8 MATTINA (10 punti, verificata su CPU e GPU):
            #    PLANE 1 taken out of a joint two-plane optimization is 37.95%
            #    wrong; a plane quantized ON ITS OWN is 28.13%. The joint one is
            #    co-adapted to its partner. Since 94.5% of experts keep only
            #    plane 1, we quantize TWICE:
            #      - a DEDICATED single-plane GPTQ, for every expert (this is
            #        what the cold ones keep)
            #      - a joint two-plane pass, keeping BOTH planes for the hot
            #        ones — keeping only the second would leave an orphaned
            #        pair, which is the defect that shipped in the first build
            pz = int(os.getenv("ODINO_PEZZO","3000000"))
            Wt = torch.from_numpy(W.reshape(-1, ind)).float()
            d1, q1 = G.quantize_one_plane(Wt, Hc, chunk=pz, foem_beta=FOEM_BETA) if Hc is not None \
                     else (lambda r: (r[0], r[1]))(G.quantize(Wt, None, rounds=2, chunk=pz))
            # ⚡ The 2nd plan serves ONLY_LAYERS to the 28 hot experts of 512:
            #    all 512 with two planes would be 18x the necessary work.
            righe_esp0 = W.reshape(-1, ind).shape[0] // E
            caldi0 = hot_experts(L, K_HOT) if has_p2(L) else np.array([], int)
            if len(caldi0):
                sel0 = np.concatenate([np.arange(e*righe_esp0, (e+1)*righe_esp0) for e in caldi0])
                Wc2 = torch.from_numpy(np.ascontiguousarray(W.reshape(-1, ind)[sel0])).float()
                # ⛔ BOTH JOINT planes are stored for the hot experts. The
                # first build discarded the joint plane-1 and kept the
                # dedicated one, leaving an orphaned pair: a plane-2 optimized
                # against a plane-1 the file no longer contained.
                j1d, j1q, d2, q2 = G.quantize(Wc2, Hc, rounds=2, chunk=pz, foem_beta=FOEM_BETA)
                del Wc2
            else:
                j1d = j1q = d2 = q2 = None
            del Wt
            p1 = pack(d1.reshape(-1,1), q1.reshape(-1,256))
            if d2 is not None:
                nb_e = righe_esp0 * (ind // 256)         # blocchi TQ1_0 per esperto
                pj1 = pack(j1d.reshape(-1,1), j1q.reshape(-1,256))
                v1 = p1.reshape(E, nb_e, 54)
                for s, e in enumerate(caldi0):           # innesto coppia congiunta
                    v1[int(e)] = pj1[s*nb_e:(s+1)*nb_e]
                p1 = v1.reshape(-1, 54)
                # ⛔ CORREZIONE 26/8 sera: riordino hot-primi ALLA NASCITA
                p1 = reorder_experts(p1, hot_first_permutation(L, K_HOT, E))
            fatti[orig] = (p1,
                           pack(d2.reshape(-1,1), q2.reshape(-1,256)) if d2 is not None else None)
            # ⛔ BOUNDARY CHECK. The hottest expert, now at POSITION 0,
            # reconstructed from the TWO planes actually written, must match
            # the bf16 source. Do not assume it — measure it, here, at write
            # time. This assert is the lesson of the first build.
            if fatti[orig][1] is not None:
                from gguf import quants as GQv
                nb_e = righe_esp0 * (ind // 256)
                ric = GQv.dequantize(fatti[orig][0][:nb_e], T.TQ1_0).reshape(righe_esp0, ind) \
                    + GQv.dequantize(fatti[orig][1][:nb_e], T.TQ1_0).reshape(righe_esp0, ind)
                e1 = int(caldi0[0])
                Wv = W.reshape(-1, ind)[e1*righe_esp0:(e1+1)*righe_esp0].astype(np.float32)
                err_v = float(np.linalg.norm(ric - Wv) / np.linalg.norm(Wv))
                assert err_v < 0.6, f"{orig}: hottest expert does not match ({err_v:.2f}) — STOP"
                log(f"      ✓ confine ok: caldo n.1 in pos.0, errore 2-piani {err_v*100:.1f}%")
            del d1,q1,d2,q2,j1d,j1q
            # the fused twin is already cached: quantize it NOW and release
            # it, rather than holding it for later (8.6 GB less resident)
            for gem in list(lettore.cassa):
                Wg = lettore.cassa.pop(gem)
                Eg, og, indg = Wg.shape
                Hg = hchol(int(gem.split(".")[1])) if indg == 4096 else None
                Wgt = torch.from_numpy(Wg.reshape(-1, indg)).float()
                g1, gq1 = G.quantize_one_plane(Wgt, Hg, chunk=pz, foem_beta=FOEM_BETA) if Hg is not None \
                          else (lambda r: (r[0], r[1]))(G.quantize(Wgt, None, rounds=2, chunk=pz))
                rg0 = Wg.reshape(-1, indg).shape[0] // Eg
                Lg = int(gem.split(".")[1])
                cg0 = hot_experts(Lg, K_HOT) if has_p2(Lg) else np.array([], int)
                if len(cg0):
                    sg0 = np.concatenate([np.arange(e*rg0, (e+1)*rg0) for e in cg0])
                    Wg2 = torch.from_numpy(np.ascontiguousarray(Wg.reshape(-1, indg)[sg0])).float()
                    gj1d, gj1q, g2, gq2 = G.quantize(Wg2, Hg, rounds=2, chunk=pz, foem_beta=FOEM_BETA)   # coppia congiunta
                    del Wg2
                else:
                    gj1d = gj1q = g2 = gq2 = None
                del Wgt
                pg1 = pack(g1.reshape(-1,1), gq1.reshape(-1,256))
                if g2 is not None:
                    nb_g = rg0 * (indg // 256)
                    pgj1 = pack(gj1d.reshape(-1,1), gj1q.reshape(-1,256))
                    vg = pg1.reshape(Eg, nb_g, 54)
                    for s, e in enumerate(cg0):
                        vg[int(e)] = pgj1[s*nb_g:(s+1)*nb_g]
                    pg1 = reorder_experts(vg.reshape(-1, 54), hot_first_permutation(Lg, K_HOT, Eg))
                fatti[gem] = (pg1,
                              pack(g2.reshape(-1,1), gq2.reshape(-1,256)) if g2 is not None else None)
                save_to_vault(gem, *fatti[gem])
                pesi_fatti += Wg.size
                del Wg,g1,gq1,g2,gq2,gj1d,gj1q
            save_to_vault(orig, *fatti[orig])
            pesi_fatti += W.size
            v = pesi_fatti/(time.time()-t0)/1e6
            log(f"  [{i+1}/{len(plan)}] {orig} · {W.size/1e6:.0f}M pesi · {v:.1f} M/s · resta {(386.5e9-pesi_fatti)/(v*1e6)/3600:.2f}h")
            del W
        w.write_tensor_data(fatti[orig][1 if secondo else 0])
        w.fout[0].flush(); giornale.write_text(f"{i}\n{w.fout[0].tell()}\n")
        if secondo: fatti.pop(orig, None)
    w.close()
    log(f"🏁 FATTO in {(time.time()-t0)/3600:.2f} h → {OUTPUT}")


if __name__ == "__main__":
    main()

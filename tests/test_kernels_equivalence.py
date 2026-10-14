"""Bit-for-bit equivalence between the original (Italian) modules and their English port.

The port is a pure rename (``docs/GLOSSARY_IT_EN.md``). ``scripts/check_port_ast.py`` proves statically that the
syntax trees are identical; these tests prove it numerically: every function is called on the same inputs in the
original module and in the port, and the outputs must be identical **bit for bit** (same dtype, same shape, same
bytes; NaN payloads and signed zeros included).

* CPU: pure functions and the CPU fallbacks of the kernels.
* GPU: the Triton kernels (skipped when no GPU or no Triton is available).

The original trees are located through environment variables (nothing is hard-coded)::

    FUCINA_ORIG_THOR_V2     directory with e2e_studente.py, e2e_kernel_qat.py, ... (and strumenti/)
    FUCINA_ORIG_S23         directory with s23/, lib/, s1b/, rif/ of the S2+S3 code
    FUCINA_ORIG_KERNEL_ROT  directory with tq1_0r_1024_codec.py and gen_sign_1024.py

and the port needs the usual ``FUCINA_GGUF_PY``, ``FUCINA_RD_TERNARY_DIR`` and ``FUCINA_REFINE_DIR``
(``fucina/extdeps.py``). ``scripts/run_equivalence_after_passata.sh`` sets everything and runs this file.

When a GPU kernel is not deterministic run to run (atomics), the original is called twice; if the two original
results differ, the comparison is reported as xfail instead of a failure.
"""
from __future__ import annotations

import importlib
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
ORIG_THOR_V2 = os.environ.get("FUCINA_ORIG_THOR_V2")
ORIG_S23 = os.environ.get("FUCINA_ORIG_S23")
ORIG_KERNEL_ROT = os.environ.get("FUCINA_ORIG_KERNEL_ROT")

pytestmark = pytest.mark.skipif(
    not (ORIG_THOR_V2 and ORIG_S23 and ORIG_KERNEL_ROT),
    reason="set FUCINA_ORIG_THOR_V2, FUCINA_ORIG_S23 and FUCINA_ORIG_KERNEL_ROT to the original trees")


def _load_tsv(path: Path) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#") and "\t" in line:
            a, b = line.split("\t", 1)
            out[a] = b
    return out


ID_MAP = _load_tsv(ROOT / "docs" / "port_identifier_map.tsv")
MOD_MAP = _load_tsv(ROOT / "docs" / "port_module_map.tsv")
_TOKEN = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_]*)(?![A-Za-z0-9_])")


def en(name: str) -> str:
    """English name of an original identifier."""
    return ID_MAP.get(name, name)


def en_text(s: str) -> str:
    """English form of an original string without whitespace (dictionary keys, option values)."""
    return _TOKEN.sub(lambda m: ID_MAP.get(m.group(1), m.group(1)), s)


# ───────────────────────────────────────────── module loading ─────────────────────────────────────────────
_ORIG_PATHS_DONE = False


def _orig_paths() -> None:
    global _ORIG_PATHS_DONE
    if _ORIG_PATHS_DONE:
        return
    s23 = Path(ORIG_S23)
    for p in (ORIG_KERNEL_ROT, str(s23 / "rif"), str(s23 / "s1b"), str(s23 / "lib"), str(s23 / "s23"), ORIG_THOR_V2):
        if p not in sys.path:
            sys.path.insert(0, p)
    _ORIG_PATHS_DONE = True


def orig(module: str):
    """Original module, e.g. ``orig("e2e_kernel_qat")``."""
    _orig_paths()
    return importlib.import_module(module)


def port(module: str):
    """Port of an original module, e.g. ``port("e2e_kernel_qat")`` -> ``fucina.kernels.qat``."""
    _orig_paths()          # the port reaches the same third-party helpers (rd_ternary, comune) through FUCINA_* vars
    return importlib.import_module(MOD_MAP[module])


def pair(module: str, name: str):
    """(original callable, ported callable) for an original module and function name."""
    return getattr(orig(module), name), getattr(port(module), en(name))


def has_gpu() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


gpu = pytest.mark.skipif(not has_gpu(), reason="needs a GPU with Triton")


# ───────────────────────────────────────────── bitwise comparison ─────────────────────────────────────────────
def _bytes_of(t: "torch.Tensor") -> bytes:
    t = t.detach().contiguous().cpu()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    return t.numpy().tobytes()


def same(a, b, path: str = "out") -> list:
    """List of differences between an original result ``a`` and a ported result ``b`` (empty = bit-identical)."""
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor):
            return [f"{path}: tensor vs {type(b).__name__}"]
        if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
            return [f"{path}: {a.dtype}{tuple(a.shape)} vs {b.dtype}{tuple(b.shape)}"]
        if a.device.type != b.device.type:
            return [f"{path}: device {a.device} vs {b.device}"]
        return [] if _bytes_of(a) == _bytes_of(b) else [f"{path}: values differ"]
    if isinstance(a, np.ndarray):
        if not isinstance(b, np.ndarray):
            return [f"{path}: ndarray vs {type(b).__name__}"]
        if a.dtype != b.dtype or a.shape != b.shape:
            return [f"{path}: {a.dtype}{a.shape} vs {b.dtype}{b.shape}"]
        return [] if np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes() else [f"{path}: values differ"]
    if isinstance(a, dict):
        if not isinstance(b, dict):
            return [f"{path}: dict vs {type(b).__name__}"]
        ka = {(en_text(k) if isinstance(k, str) else k): v for k, v in a.items()}
        if set(ka) != set(b):
            return [f"{path}: keys {sorted(map(str, ka))} vs {sorted(map(str, b))}"]
        return [d for k in ka for d in same(ka[k], b[k], f"{path}[{k!r}]")]
    if isinstance(a, (list, tuple)):
        if type(a) is not type(b) and not (isinstance(b, (list, tuple))):
            return [f"{path}: {type(a).__name__} vs {type(b).__name__}"]
        if len(a) != len(b):
            return [f"{path}: length {len(a)} vs {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in same(x, y, f"{path}[{i}]")]
    if isinstance(a, float) and isinstance(b, float):
        return [] if (math.isnan(a) and math.isnan(b)) or np.float64(a).tobytes() == np.float64(b).tobytes() \
            else [f"{path}: {a!r} vs {b!r}"]
    if isinstance(a, str) and isinstance(b, str):
        return [] if en_text(a) == b or a == b else [f"{path}: {a!r} vs {b!r}"]
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]


def check(fo, fp, make_args, *, retry_nondeterministic: bool = False):
    """Call original and port on fresh copies of the same inputs and require bit-identical results (or the same
    exception class, when the original rejects the inputs).

    ``make_args()`` returns ``(args, kwargs)`` built from a fixed seed, so both sides see identical inputs and no
    in-place modification can leak from one call to the other."""
    args, kw = make_args()
    try:
        ro = fo(*args, **kw)
    except Exception as eo:  # noqa: BLE001  -- the port must then fail in the same way
        args, kw = make_args()
        with pytest.raises(type(eo)):
            fp(*args, **kw)
        return eo, eo
    args, kw = make_args()
    rp = fp(*args, **kw)
    diff = same(ro, rp)
    if diff and retry_nondeterministic:
        args, kw = make_args()
        ro2 = fo(*args, **kw)
        if same(ro, ro2):
            pytest.xfail(f"the original kernel is not deterministic run to run: {same(ro, ro2)[:3]}")
    assert not diff, diff[:10]
    return ro, rp


def rng(seed: int = 20261014) -> "torch.Generator":
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def randn(*shape, dtype=torch.float32, device="cpu", seed=1, scale=1.0):
    return (torch.randn(*shape, generator=rng(seed), dtype=torch.float64) * scale).to(dtype).to(device)


def spd(n: int, seed: int = 3, dtype=torch.float64, device="cpu", ridge: float = 1e-2):
    X = torch.randn(n, 2 * n, generator=rng(seed), dtype=torch.float64)
    return ((X @ X.T) / (2 * n) + ridge * torch.eye(n, dtype=torch.float64)).to(dtype).to(device)


def trits(*shape, seed=5, device="cpu"):
    return torch.randint(-1, 2, shape, generator=rng(seed), dtype=torch.int8).to(device)


# ───────────────────────────────────────────── structure ─────────────────────────────────────────────
STRUCTURAL = ["kernel_blocco_fuso", "e2e_hilo_fuso", "e2e_indietro_hilo", "e2e_bf16x3", "e2e_conv_diretta",
              "e2e_flash_qat", "e2e_densi_piegati", "e2e_kernel_qat", "strumenti.gdn_qat", "e2e_studente",
              "e2e_pv_codici", "e2e_cache_dequant", "e2e_codici_scale", "e2e_campi_scala", "dequant_gguf_torch",
              "gptq_1024_lib", "rotazione_gpu", "tq1_0r_1024_codec", "gen_sign_1024", "carica_strato_gguf",
              "s23_algebra", "s23_risolvi", "s1b_comune", "s1_comune", "cardan_piega", "sorgente", "s23_scrivi",
              "s23_passata"]


# module-level names that existed only to extend sys.path in the original tree (replaced by fucina.extdeps)
PLUMBING_ONLY = {"p", "_LIB", "RAFF"}


@pytest.mark.parametrize("module", STRUCTURAL)
def test_every_public_name_is_ported(module):
    """Every function, class and constant defined by the original module exists in the port under its English name."""
    try:
        o = orig(module)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"original {module} not importable here: {type(e).__name__}: {e}")
    p = port(module)
    missing = []
    for name, obj in vars(o).items():
        if name.startswith("__") or getattr(obj, "__module__", module) not in (module, None) and callable(obj):
            continue
        if isinstance(obj, type(sys)) or name in PLUMBING_ONLY:          # imported modules, path plumbing
            continue
        if not hasattr(p, en(name)):
            missing.append(f"{name} -> {en(name)}")
    assert not missing, missing


# ───────────────────────────────────────────── ternary formats (CPU) ─────────────────────────────────────────────
def test_sign_vector():
    fo, fp = pair("gen_sign_1024", "genera")
    check(fo, fp, lambda: ((), {}))


def test_codec_fwht_quantize_dequantize():
    W = randn(6, 2048, seed=11).numpy()
    fo, fp = pair("tq1_0r_1024_codec", "fwht1024_f32")
    check(fo, fp, lambda: ((W[:, :1024].copy(),), {}))
    fo, fp = pair("tq1_0r_1024_codec", "quantize_rows")
    packed, _ = check(fo, fp, lambda: ((W.copy(),), {}))
    fo, fp = pair("tq1_0r_1024_codec", "dequantize_blocks")
    check(fo, fp, lambda: ((packed.copy(),), {}))
    fo, fp = pair("tq1_0r_1024_codec", "scala_ls")
    check(fo, fp, lambda: ((W.reshape(-1, 8, 128).copy(),), {}))


def test_scale_fields_roundtrip():
    W = randn(4, 2048, seed=12).numpy()
    raw = orig("tq1_0r_1024_codec").quantize_rows(W)
    for name in ("blocchi", "leggi_scale"):
        fo, fp = pair("e2e_campi_scala", name)
        check(fo, fp, lambda: ((raw.copy(), "TQ1_0R_1024"), {}))
    d = orig("e2e_campi_scala").leggi_scale(raw, "TQ1_0R_1024")
    d2 = (np.asarray(d, dtype=np.float32) * np.float32(1.5)).astype(np.float16)
    fo, fp = pair("e2e_campi_scala", "scrivi_scale")
    check(fo, fp, lambda: ((raw.copy(), "TQ1_0R_1024", d2.copy()), {}))
    for name in ("trit_torch", "scale_torch"):
        fo, fp = pair("e2e_campi_scala", name)
        check(fo, fp, lambda: ((torch.from_numpy(raw.copy()).reshape(-1, 224), "TQ1_0R_1024"), {}))


def test_rotation_cpu():
    V = randn(3, 2048, seed=13)
    fo, fp = pair("rotazione_gpu", "fwht1024_torch")
    check(fo, fp, lambda: ((V[:, :1024].clone(),), {}))
    Wn = randn(2, 3, 2048, seed=14).numpy()
    for name in ("ruota_righe_batch_gpu", "de_ruota_pesi_batch_gpu"):
        fo, fp = pair("rotazione_gpu", name)
        check(fo, fp, lambda: ((Wn.copy(),), {"device": "cpu"}))
    Hn = np.stack([spd(1024, seed=s, dtype=torch.float32).numpy() for s in (15, 16)])
    fo, fp = pair("rotazione_gpu", "ruota_hessiana_batch_gpu")
    check(fo, fp, lambda: ((Hn.copy(),), {"device": "cpu"}))


def test_gptq_1024_cpu():
    fo, fp = pair("gptq_1024_lib", "R_1024_matrice")
    check(fo, fp, lambda: ((), {}))
    H = spd(1024, seed=17).numpy()
    W = randn(8, 1024, seed=18).numpy().astype(np.float32)
    fo, fp = pair("gptq_1024_lib", "ruota_hessiana_1024")
    check(fo, fp, lambda: ((H.copy(),), {}))
    fo, fp = pair("gptq_1024_lib", "ruota_righe_1024")
    check(fo, fp, lambda: ((W.copy(),), {}))
    fo, fp = pair("gptq_1024_lib", "prepara_hessiana")
    Hchol, _ = check(fo, fp, lambda: ((H.copy(),), {}))
    fo, fp = pair("gptq_1024_lib", "gptq_1024")
    Ht = torch.from_numpy(np.asarray(Hchol))
    check(fo, fp, lambda: ((torch.from_numpy(W.copy()), Ht.clone()), {"device": "cpu"}))


# ───────────────────────────────────────────── hi/lo splitting (CPU and GPU) ─────────────────────────────────────────────
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=gpu)])
def test_hilo_split_helpers(device):
    m = randn(64, seed=21, device=device).abs()
    m[3] = 0.0
    m[5] = float("inf")
    fo, fp = pair("e2e_indietro_hilo", "scala_pot2")
    check(fo, fp, lambda: ((m.clone(),), {}))
    v = randn(32, 48, seed=22, device=device, scale=0.4)
    for name in ("dividi_fp16", "dividi_bf16"):
        fo, fp = pair("e2e_indietro_hilo", name)
        check(fo, fp, lambda: ((v.clone(),), {}))
    a = randn(40, 16, seed=23, device=device).abs()
    eid = torch.randint(0, 5, (40,), generator=rng(24)).to(device)
    fo, fp = pair("e2e_indietro_hilo", "massimi_segmento")
    check(fo, fp, lambda: ((a.clone(), eid.clone(), 6), {}))


# ───────────────────────────────────────────── bf16x3 products ─────────────────────────────────────────────
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=gpu)])
def test_bf16x3_products(device):
    x = randn(33, 70, seed=31, device=device)
    fo, fp = pair("e2e_bf16x3", "scomponi")
    check(fo, fp, lambda: ((x.clone(),), {}))
    fo, fp = pair("e2e_bf16x3", "spezzoni_k")
    for shape in ((64, 64, 64), (16, 16, 8192), (512, 512, 4096)):
        check(fo, fp, lambda: (shape, {}))
    b = randn(70, 50, seed=32, device=device)
    bh = b.to(torch.bfloat16)
    bl = (b - bh.float()).to(torch.bfloat16)
    fo, fp = pair("e2e_bf16x3", "matmul_x3")
    check(fo, fp, lambda: ((x.clone(), bh.clone(), bl.clone()), {}), retry_nondeterministic=True)
    fo, fp = pair("e2e_bf16x3", "mm_bf16")
    check(fo, fp, lambda: ((x.to(torch.bfloat16), bh.clone()), {}), retry_nondeterministic=True)


@gpu
def test_bf16x3_triton_kernel():
    a = randn(130, 257, seed=33, device="cuda")
    b = randn(257, 96, seed=34, device="cuda")
    bh = b.to(torch.bfloat16)
    bl = (b - bh.float()).to(torch.bfloat16)
    c = randn(130, 96, seed=35, device="cuda")
    fo, fp = pair("e2e_bf16x3", "mm_triton")
    check(fo, fp, lambda: ((a.clone(), bh.clone(), bl.clone()), {}), retry_nondeterministic=True)
    check(fo, fp, lambda: ((a.clone(), bh.clone(), bl.clone(), c.clone()), {}), retry_nondeterministic=True)


# ───────────────────────────────────────────── direct causal convolution ─────────────────────────────────────────────
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=gpu)])
@pytest.mark.parametrize("direction", [0, 1])
def test_direct_conv(device, direction):
    x = randn(2, 37, 64, seed=41, device=device)
    w = randn(64, 4, seed=42, device=device, scale=0.5)
    fo, fp = pair("e2e_conv_diretta", "conv_btc")
    check(fo, fp, lambda: ((x.clone(), w.clone(), direction), {}), retry_nondeterministic=True)
    check(fo, fp, lambda: ((x.clone(), w.clone(), direction, True), {}), retry_nondeterministic=True)


@gpu
def test_direct_conv_transformers_signature():
    h = randn(2, 64, 37, seed=43, device="cuda")
    w = randn(64, 4, seed=44, device="cuda", scale=0.5)
    fo, fp = pair("e2e_conv_diretta", "causal_conv1d_fn_diretta")
    for act in (None, "silu"):
        check(fo, fp, lambda: ((h.clone(), w.clone()), {"activation": act}), retry_nondeterministic=True)


# ───────────────────────────────────────────── folded dense tensors ─────────────────────────────────────────────
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=gpu)])
def test_folded_dense_helpers(device):
    R, C, gs = 16, 256, 128
    ts = [trits(R, C, seed=s, device=device) for s in (51, 52, 53)]
    ds = [randn(R, C // gs, seed=s, device=device, scale=0.05).to(torch.float16) for s in (54, 55, 56)]
    fo, fp = pair("e2e_densi_piegati", "peso_effettivo_f32")
    Wf, _ = check(fo, fp, lambda: (([t.clone() for t in ts], [d.clone() for d in ds], gs), {}))
    fo, fp = pair("e2e_densi_piegati", "peso_x2")
    check(fo, fp, lambda: ((Wf.clone(),), {}))
    G = randn(R, C, seed=57, device=device)
    fo, fp = pair("e2e_densi_piegati", "dd_da_g")
    check(fo, fp, lambda: ((G.clone(), ts[0].clone(), gs), {}))


@gpu
def test_folded_dense_fp16_product():
    a = randn(96, 256, seed=58, device="cuda", scale=0.3).to(torch.float16)
    b = randn(256, 80, seed=59, device="cuda", scale=0.3).to(torch.float16)
    fo, fp = pair("e2e_densi_piegati", "mm")
    check(fo, fp, lambda: ((a.clone(), b.clone()), {}), retry_nondeterministic=True)
    fo, fp = pair("e2e_densi_piegati", "mm16_triton")
    check(fo, fp, lambda: ((a.clone(), b.clone()), {}), retry_nondeterministic=True)


# ───────────────────────────────────────────── QAT kernels ─────────────────────────────────────────────
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=gpu)])
def test_qat_amax(device):
    t = randn(17, 300, seed=61, device=device)
    t[2, 5] = float("nan")
    fo, fp = pair("e2e_kernel_qat", "amax_abs")
    check(fo, fp, lambda: ((t.clone(), 1), {}))
    check(fo, fp, lambda: ((t.clone(), (0, 1)), {"keepdim": True}))


@gpu
def test_qat_prep_and_mmx():
    g = randn(64, 1024, seed=62, device="cuda")
    cs = randn(1024, seed=63, device="cuda").abs() + 0.5
    for lo in (False, True):
        fo, fp = pair("e2e_kernel_qat", "prep_righe")
        check(fo, fp, lambda: ((g.clone(), cs.clone(), lo), {}), retry_nondeterministic=True)
        fo, fp = pair("e2e_kernel_qat", "prep_colonne")
        check(fo, fp, lambda: ((g.clone(), lo), {}), retry_nondeterministic=True)
    a = randn(128, 256, seed=64, device="cuda", scale=0.2).to(torch.float16)
    b = randn(256, 192, seed=65, device="cuda", scale=0.2)
    bh = b.to(torch.float16)
    bl = ((b - bh.float()) * 2048.0).to(torch.float16)
    fo, fp = pair("e2e_kernel_qat", "mmx")
    for terms in (1, 2):
        check(fo, fp, lambda: ((a.clone(), bh.clone(), bl.clone(), terms), {}), retry_nondeterministic=True)


# ───────────────────────────────────────────── attention and GDN products ─────────────────────────────────────────────
@gpu
def test_flash_bmmx():
    a = randn(3, 70, 64, seed=71, device="cuda")
    b = randn(3, 64, 90, seed=72, device="cuda", scale=0.2)
    bh = b.to(torch.float16)
    bl = ((b - bh.float()) * 2048.0).to(torch.float16)
    fo, fp = pair("e2e_flash_qat", "bmmx")
    check(fo, fp, lambda: ((a.clone(), bh.clone(), bl.clone()), {}), retry_nondeterministic=True)


@gpu
def test_gdn_products():
    a = randn(2, 64, 128, seed=73, device="cuda")
    b = randn(2, 128, 64, seed=74, device="cuda")
    fo, fp = pair("strumenti.gdn_qat", "mm_split")
    for split in (1, 3):
        check(fo, fp, lambda: ((a.clone(), b.clone(), split), {}), retry_nondeterministic=True)
    x = randn(4, 64, 96, seed=75, device="cuda", scale=0.2).to(torch.float16)
    y = randn(4, 96, 128, seed=76, device="cuda", scale=0.2).to(torch.float16)
    fo, fp = pair("strumenti.gdn_qat", "bmm16_triton")
    check(fo, fp, lambda: ((x.clone(), y.clone()), {}), retry_nondeterministic=True)


@gpu
def test_gdn_chunked_recurrence():
    B, T, H, Dk, Dv = 1, 128, 2, 64, 64
    q = randn(B, T, H, Dk, seed=77, device="cuda", scale=0.3)
    k = randn(B, T, H, Dk, seed=78, device="cuda", scale=0.3)
    v = randn(B, T, H, Dv, seed=79, device="cuda", scale=0.3)
    g = -randn(B, T, H, seed=80, device="cuda").abs() * 0.1
    beta = torch.sigmoid(randn(B, T, H, seed=81, device="cuda"))
    fo, fp = pair("strumenti.gdn_qat", "chunk_gated_delta_rule_qat")
    try:
        check(fo, fp, lambda: ((q.clone(), k.clone(), v.clone(), g.clone(), beta.clone()),
                               {"output_final_state": True, "use_qk_l2norm_in_kernel": True}),
              retry_nondeterministic=True)
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"GDN dependencies missing: {e}")


# ───────────────────────────────────────────── fused in-block trit sweep ─────────────────────────────────────────────
@gpu
def test_fused_block_sweep():
    B, rows, n_in = 2, 8, 256
    group = orig("gptq_1024_lib").GROUP
    W = randn(B, rows, n_in, seed=91, dtype=torch.float64)
    D = randn(B, rows, n_in // group, seed=92, dtype=torch.float64).abs() * 0.5 + 0.1
    Q = torch.clamp(torch.round(W / D.repeat_interleave(group, dim=2)), -1, 1).to(torch.int8)
    H = torch.stack([spd(n_in, seed=93 + b) for b in range(B)])
    fo, fp = pair("kernel_blocco_fuso", "raffina_trit_sweep_batch_fuso")
    check(fo, fp, lambda: ((W.clone(), Q.clone(), D.clone(), H.clone()), {"device": "cuda"}),
          retry_nondeterministic=True)


# ───────────────────────────────────────────── S2+S3 linear algebra (CPU fp64) ─────────────────────────────────────────────
def test_s23_algebra_factorizations():
    M = spd(48, seed=101)
    Bm = randn(48, 7, seed=102, dtype=torch.float64)
    for name, args in (("chol", (M,)), ("chol_ex", (M,)), ("risolvi", (M, Bm)), ("cholesky_smorzata", (M,)),
                       ("inversa_smorzata", (M,)), ("L_unit", (M,))):
        fo, fp = pair("s23_algebra", name)
        check(fo, fp, lambda: (tuple(a.clone() for a in args), {}))
    L = torch.linalg.cholesky(M)
    fo, fp = pair("s23_algebra", "risolvi_chol")
    check(fo, fp, lambda: ((Bm.clone(), L.clone()), {}))
    fo, fp = pair("s23_algebra", "lanczos_max")
    check(fo, fp, lambda: ((M.clone(),), {}))
    fo, fp = pair("s23_algebra", "klein_ln_rho")
    for K, m in ((2, 16), (8, 64), (32, 128)):
        check(fo, fp, lambda: ((K, m), {}))


def test_s23_algebra_quantizers():
    B, R, C, ng = 2, 6, 256, 2
    W = randn(B, R, C, seed=111, dtype=torch.float64, scale=0.05)
    H = torch.stack([spd(C, seed=112 + b) for b in range(B)])
    al = orig("s23_algebra")
    U = torch.stack([al.cholesky_smorzata(H[b])[1] for b in range(B)])
    fo, fp = pair("s23_algebra", "gptq_bil_lotto")
    (T, S), _ = check(fo, fp, lambda: ((W.clone(), U.clone(), ng), {}))
    fo, fp = pair("s23_algebra", "ricostruisci")
    What, _ = check(fo, fp, lambda: ((T.clone(), S.clone()), {}))
    fo, fp = pair("s23_algebra", "obiettivo")
    check(fo, fp, lambda: ((W.clone(), What.clone(), H.clone()), {}))
    fo, fp = pair("s23_algebra", "espandi")
    check(fo, fp, lambda: ((S[0].clone(), C), {}))
    A, Bk = spd(C, seed=114), spd(R, seed=115)
    fo, fp = pair("s23_algebra", "fastkron_sd")
    check(fo, fp, lambda: ((W[0].clone(), A.clone(), Bk.clone(), S[0].clone()), {}))
    fo, fp = pair("s23_algebra", "scale_due_lati")
    check(fo, fp, lambda: ((W[0].clone(), T[0].clone(), S[0].clone(), A.clone(), Bk.clone()), {"it": 10}))
    fo, fp = pair("s23_algebra", "J_due_lati")
    check(fo, fp, lambda: (((W[0] - What[0]).clone(), A.clone(), Bk.clone()), {}))
    fo, fp = pair("s23_algebra", "scale_scrivibili")
    check(fo, fp, lambda: ((S.clone(), S.clone()), {}))
    fo, fp = pair("s23_algebra", "legame_gqa")
    check(fo, fp, lambda: ((8, 2, 16), {}))


# ───────────────────────────────────────────── stage-S1 / S1b helpers ─────────────────────────────────────────────
def test_s1_helpers():
    fo, fp = pair("s1_comune", "triu_indici")
    check(fo, fp, lambda: ((9,), {}))
    M = spd(9, seed=121).numpy()
    fo, fp = pair("s1_comune", "impacchetta")
    v, _ = check(fo, fp, lambda: ((M.copy(),), {}))
    fo, fp = pair("s1_comune", "spacchetta")
    check(fo, fp, lambda: ((np.asarray(v).copy(), 9), {}))
    fo, fp = pair("s1_comune", "n_da_tri")
    check(fo, fp, lambda: ((45,), {}))
    fo, fp = pair("s1_comune", "ristretta")
    check(fo, fp, lambda: ((M.copy(), 12.0, (M * 2).copy(), 4.0), {}))
    ids = torch.randint(0, 50000, (4, 64), generator=rng(122)).numpy()
    fo, fp = pair("s1_comune", "hash_ngrammi")
    check(fo, fp, lambda: ((ids.copy(),), {}))


def test_s1b_helpers():
    W = randn(4, 2048, seed=131, dtype=torch.float64)
    for name in ("ruota", "deruota"):
        fo, fp = pair("s1b_comune", name)
        check(fo, fp, lambda: ((W.clone(),), {}))
    H = spd(1024, seed=132)
    fo, fp = pair("s1b_comune", "ruota_H")
    check(fo, fp, lambda: ((H.clone(),), {}))
    R, C, ng = 5, 256, 2
    T = trits(R, C, seed=133)
    S = randn(R, ng, seed=134, dtype=torch.float64).abs() + 0.1
    fo, fp = pair("s1b_comune", "espandi")
    check(fo, fp, lambda: ((T.clone(), S.clone()), {}))
    Hs = spd(C, seed=135)
    Cs = randn(R, C, seed=136, dtype=torch.float64)
    fo, fp = pair("s1b_comune", "sistema")
    (Mm, b), _ = check(fo, fp, lambda: ((T.clone(), Hs.clone(), Cs.clone(), ng), {}))
    fo, fp = pair("s1b_comune", "risolvi")
    S1, _ = check(fo, fp, lambda: ((Mm.clone(), b.clone(), S.clone()), {}))
    fo, fp = pair("s1b_comune", "J")
    check(fo, fp, lambda: ((Mm.clone(), b.clone(), S1.clone()), {}))
    fo, fp = pair("s1b_comune", "scala_ok")
    check(fo, fp, lambda: ((S1.clone(), S.clone()), {}))
    fo, fp = pair("s1b_comune", "legame_gqa")
    check(fo, fp, lambda: ((8, 2, 16), {}))
    fo, fp = pair("s1b_comune", "ristretta")
    check(fo, fp, lambda: ((Hs.clone(), 10.0, (2 * Hs).clone(), 3.0), {}))

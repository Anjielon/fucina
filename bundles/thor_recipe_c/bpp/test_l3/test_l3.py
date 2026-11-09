#!/usr/bin/env python3
"""CPU tests of lever L3 (SINTESI_1025, 25/10): gate/up fp16 group scales trained directly (--scale-gateup-lr), gate/up
low-rank branch removable (--rango-esperti 0), exporter scale_gateup_esporta.py. No GPU, no live service, no deletion.

The student SUBCLASSES the real e2e_studente.Studente (avanti / indietro / _indietro_troncato / esce / nascosto are the
real code) and every MoE layer is the REAL MoEStudente with a REAL StratoEsperti over REAL DatiFamiglia: gate/up are
TQ1_0R_1024 bytes (stored rotated basis, 8 fp16 scales per 1024 weights), down is TQ1_0, all built from known trits and
scales with the llama.cpp packing (tq1_0r_1024_codec._pack_quarter) and decoded by the student's own decoder; the engine is
MotoreTorch (CPU), recipe B32 (activation rotated like production B/B3) or the fp64 recipes for the gradient checks. The
CARDAN gate/up branch, the CARDAN down branch and the dense residual branch are the real modules installed by the
trainer, whose `main` runs for real (device cpu). Teacher caches, KL4 packs and heads are synthetic (as in test_l2l5).

  T0  synthetic TQ1_0R_1024 / TQ1_0 bytes: trits and scales decode back at the bit; the student's weight == the reference
      dequantisation (codec of llama.cpp for TQ1_0R_1024, gguf-py for TQ1_0) at the bit
  F*  flags off == original (test_l3/originale_L2_cardan_congiunto_scale.py = L2/codice trainer, sha 10c7fce9...): every
      saved tensor and every logged value bitwise equal (separate, fused, alternating, full KL, with the L2-era down scales
      + down low-rank switched on, and through a --riprendi)
  N*  the new switches: --rango-esperti 0 == rank 4 with C, D frozen (lr 0) on everything else, no mlp.cardan module;
      --scale-gateup-lr runs (separate / fused / fused + held graph / alternating), theta moves, packs per tappa with the
      sha256 of the fp16 scales used, Adam state; --riprendi bitwise == uninterrupted; presence gate; config errors
  G*  gradient checks of theta: the real MoEStudente backward (FamigliaEsperti -> MotoreTorch.dd, recipes B64 and A64) ==
      autograd of an INDEPENDENT dense reference (own trits, own Sylvester-Hadamard rotation) and == central differences
      (fp16 rounding replaced by the identity, the STE point); with the real rounding at theta = 0 the gradient is the
      same at the bit; checkpointed trainer backward (avanti + indietro) == full-graph autograd
  E*  exporter: trainer pack -> scale_gateup_esporta.py -> raffina/patch_gguf.py on a toy GGUF: nothing outside the d
      fields changes, d fields == the trainer's fp16 scales (sha), dequant of the written file (llama.cpp codec) == the
      student's trained weight at the bit; theta = 0 is the identity; refusals; ONE REAL LAYER (L12 of the training base):
      codes identical, scales == StratoEsperti._scala on famiglia_da_tensore, round trip of the written .npy
Run (one job, 4 threads, nice 19, RSS < 2 GB):
    HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 nice -n 19 \\
      @@ROOT:venv@@/bin/python3 test_l3/test_l3.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import random
import resource
import subprocess
import sys
import time
import traceback
import types
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)
QUI = Path(__file__).resolve().parent
COD = QUI.parent
sys.path.insert(0, str(COD))
ORIG_SHA = "10c7fce92a9752d362e50dc2a0ecae97aa21fa71df2284757c473c049a5b90ad"     # L2/codice/cardan_congiunto_scale.py
L2_LIVE = Path("@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/SINTESI_1025/L2/codice/cardan_congiunto_scale.py")
BASE_VERA = Path("@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/CARDAN_DOWN_1025/Thor-T-C-S7g10-DOWNH.gguf")
PATCH = Path("@@ROOT:models@@/build/leve/raffina/patch_gguf.py")
PY = "@@ROOT:venv@@/bin/python3"

import cardan_kd_paper as KDP                       # noqa: E402  (path setup of the thor_v2 tree)
import e2e_studente as S                            # noqa: E402
import e2e_allena as A                              # noqa: E402
import e2e_codici_scale as CS                       # noqa: E402
import e2e_campi_scala as CAMPI                     # noqa: E402
import scale_gateup_esporta as SGE                  # noqa: E402
from tq1_0r_1024_codec import _pack_quarter, dequantize_blocks as deq_r1024, S1024   # noqa: E402

KDP.riga_cicli = lambda *a, **k: None               # never write the real cicli.tsv
V, H, FF, E_N, TOPK, N_STRATI, T_RIGA = 48, 1024, 512, 4, 2, 3, 64
LETTERE = [40, 41, 42, 43]
TMP = Path("@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/SINTESI_1025/L3/collaudo") / time.strftime("test_%Y%m%d_%H%M%S")
TMP.mkdir(parents=True, exist_ok=True)


# ───────────────────────────────────────── synthetic ternary families (real byte layouts) ─────────────────────────────
def codifica(T: np.ndarray, s16: np.ndarray, tipo: str) -> np.ndarray:
    """trits (E, R, C) int8 in {-1,0,1} + scales (E, R, NG) float16 -> bytes (E, R, byte_per_row) uint8, llama.cpp packing."""
    nb, qk, off, k = CS.LAYOUT[tipo]
    E, R, C = T.shape
    nblk = C // qk
    Tq = (T.reshape(E * R * nblk, qk).astype(np.int64) + 1)
    out = np.zeros((E * R * nblk, nb), dtype=np.uint8)
    for q in range(qk // 256):
        qs, qh = _pack_quarter(Tq[:, q * 256:(q + 1) * 256])
        if tipo == "TQ1_0R_1024":
            out[:, q * 48:(q + 1) * 48] = qs; out[:, 192 + q * 4:192 + (q + 1) * 4] = qh
        else:
            out[:, 0:48] = qs; out[:, 48:52] = qh
    out[:, off:off + 2 * k] = np.ascontiguousarray(s16.reshape(E * R * nblk, k)).view(np.uint8)
    return out.reshape(E, R, nblk * nb)


def famiglia(nome: str, tipo: str, E: int, R: int, C: int, seme: int, L: int = -1, lo=0.01, hi=0.03) -> tuple:
    """-> (DatiFamiglia as famiglia_da_tensore builds it from the bytes, trits, fp16 scales)."""
    g = np.random.default_rng(seme)
    T = g.integers(-1, 2, size=(E, R, C)).astype(np.int8)
    nb, qk, off, k = CS.LAYOUT[tipo]
    s16 = g.uniform(lo, hi, size=(E, R, C // (qk // k))).astype(np.float16)
    B = torch.from_numpy(codifica(T, s16, tipo))
    d0 = CS.scale_da_blocchi(B.reshape(-1, nb), tipo).to(torch.float32).reshape(E, R, -1)
    return S.DatiFamiglia(nome, tipo, B, d0, L), T, s16


def famiglie_strato(L: int, H_: int = H, FF_: int = FF, E: int = E_N) -> dict:
    fg = famiglia("gate", "TQ1_0R_1024", E, FF_, H_, 1000 + 10 * L, L)
    fu = famiglia("up", "TQ1_0R_1024", E, FF_, H_, 1001 + 10 * L, L)
    fd = famiglia("down", "TQ1_0", E, H_, FF_, 1002 + 10 * L, L, 0.005, 0.015)
    return {"gate": fg, "up": fu, "down": fd}


# ───────────────────────────────────────── synthetic student with REAL MoE layers ─────────────────────────────────────
def _lin(i, o, g, dt, scala=1.0):
    lin = torch.nn.Linear(i, o, bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(o, i, generator=g, dtype=torch.float64) * scala / math.sqrt(i))
    lin = lin.to(dt)
    lin.weight.requires_grad_(False)
    return lin


class Attn(torch.nn.Module):
    def __init__(self, g, dt, H_):
        super().__init__()
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, n, _lin(H_, H_, g, dt))


class Condiviso(torch.nn.Module):
    def __init__(self, g, dt, H_):
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = _lin(H_, 16, g, dt), _lin(H_, 16, g, dt), _lin(16, H_, g, dt, 0.3)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


def moe_vero(L: int, g, dt, ric, mot, H_=H, FF_=FF, E=E_N) -> "S.MoEStudente":
    fams = {f: v[0] for f, v in famiglie_strato(L, H_, FF_, E).items()}
    es = S.StratoEsperti(fams, ric, mot)
    rw = torch.nn.Parameter((torch.randn(E, H_, generator=g, dtype=torch.float64) / math.sqrt(H_)).to(dt), requires_grad=False)
    return S.MoEStudente(rw, Condiviso(g, dt, H_), _lin(H_, 1, g, dt), es, TOPK)


class StratoT(torch.nn.Module):
    def __init__(self, L, g, dt, ric, mot):
        super().__init__()
        self.self_attn = Attn(g, dt, H)
        self.mlp = moe_vero(L, g, dt, ric, mot)


class StudenteTernario(S.Studente):
    """Real Studente methods on a 3-layer model whose MoE layers are the real MoEStudente/StratoEsperti/FamigliaEsperti."""
    RICETTA = "B32"
    ultimo = None

    def __init__(self, gguf=None, strati=range(N_STRATI), ricetta=None, motore_nome=None, device="cpu", log=print,
                 densi_ternari=(), **_):
        g = torch.Generator().manual_seed(1234)
        self.ric = S.RICETTE[self.RICETTA]
        dt = self.ric.dtype_gemm
        self.device = torch.device(device)
        self.strati = list(strati)
        self.strati_allenati = None
        self.mot = S.MotoreTorch()
        self.cfg = types.SimpleNamespace(layer_types=["full_attention"] * 40)
        self.moduli = {L: StratoT(L, g, dt, self.ric, self.mot) for L in self.strati}
        self._emb = torch.randn(V, H, generator=g, dtype=torch.float64).to(dt)
        self._testa = (torch.randn(V, H, generator=g, dtype=torch.float64) / math.sqrt(H)).to(torch.float32)
        self.eps = 1e-6
        self.norma_finale = torch.ones(H, dtype=torch.float32)
        self.oss = None
        StudenteTernario.ultimo = self

    def testa(self, dtype=torch.float32):
        return self._testa.to(dtype)

    def embedding(self, ids):
        return self._emb[ids.to(self.device)]

    def _contesto(self, B, T, dtype, inizi=None):
        return {"modo": "riga"}

    def _strato(self, L, x, ctx, instr=None):
        m = self.moduli[L]
        B, T, _ = x.shape
        h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        at = m.self_attn
        q, k, v = at.q_proj(h), at.k_proj(h), at.v_proj(h)
        cm = torch.cumsum(v, 1) / torch.arange(1, T + 1, dtype=x.dtype).view(1, T, 1)
        x = x + 0.5 * at.o_proj(torch.tanh(q) * torch.sigmoid(k) + cm)
        h2 = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        m.mlp.instr = instr
        try:
            o = m.mlp(h2)
        finally:
            m.mlp.instr = None
        return x + o


# ───────────────────────────────────────── synthetic data sources (as test_l2l5) ──────────────────────────────────────
def _seed(*k) -> int:
    return int(hashlib.sha256(repr(k).encode()).hexdigest()[:8], 16)


class SorgenteFinta:
    def __init__(self, d):
        self.dir = str(d)
        self.nome = Path(d).name

    def righe_previste(self, lato):
        return list(range(10)) if lato == "train" else [0, 1]


class CaricatoreFinto:
    def __init__(self, src, dev):
        self.src, self.dev = src, dev

    def prendi(self, lato, righe):
        i = righe[0]
        g = torch.Generator().manual_seed(_seed(self.src.nome, lato, i))
        ids = torch.randint(0, 38, (1, T_RIGA), generator=g)
        g = torch.Generator().manual_seed(_seed("h", self.src.nome, lato, i))
        hT = torch.randn(T_RIGA, H, generator=g).to(torch.bfloat16)
        inizi = torch.zeros(1, T_RIGA, dtype=torch.int32)
        inizi[0, ::16] = 1
        return A.Lotto(list(righe), ids.to(self.dev), hT.to(self.dev), inizi, ["finto"])


def domande():
    g = torch.Generator().manual_seed(77)
    Q = []
    for i in range(20):
        Ln = int(torch.randint(6, 11, (1,), generator=g))
        lp = torch.randn(4, generator=g, dtype=torch.float64)
        Q.append({"i": i, "label": i % 4, "ids": torch.randint(0, 38, (Ln,), generator=g).tolist(),
                  "lettere": LETTERE, "pT": torch.softmax(lp, 0)})
    return Q


def pacchi_kl4(ns, log):
    Q = domande()
    random.Random(ns.seme).shuffle(Q)
    if ns.calib_n:
        Q = Q[: ns.calib_n]
    pacchi, cur, nt = [], [], 0
    for q in Q:
        if cur and nt + len(q["ids"]) > 40:
            pacchi.append(cur); cur, nt = [], 0
        cur.append(q); nt += len(q["ids"])
    if cur:
        pacchi.append(cur)
    return pacchi, {"n": len(Q)}


GD = types.ModuleType("gradienti_densi")
GD.pacchi_kl4, GD.leggi_mclp = pacchi_kl4, (lambda p: {})
sys.modules["gradienti_densi"] = GD
_WT = torch.randn(V, H, generator=torch.Generator().manual_seed(99))
A.scegli_gdn = lambda log: None
A.testa_studente = lambda st, k: st.testa(torch.float32)
A.testa_maestro = lambda dev, dt: (_WT.to(dev), "finto" * 8)
A.SorgenteCache = SorgenteFinta
A.Caricatore = CaricatoreFinto
S.Studente_vero = S.Studente


# ───────────────────────────────────────── running the two trainers ───────────────────────────────────────────────────
def carica_modulo(nome: str, p: Path):
    spec = importlib.util.spec_from_file_location(nome, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class _Pigro:
    """Trainer module loaded on first use (the real-layer test E2 does not need it: lower RSS)."""
    def __init__(self, nome: str, p: Path):
        self._nome, self._p, self._m = nome, p, None

    def __getattr__(self, k):
        if self._m is None:
            self._m = carica_modulo(self._nome, self._p)
        return getattr(self._m, k)


ORIG = _Pigro("ccs_originale_l2", QUI / "originale_L2_cardan_congiunto_scale.py")
NUOVO = _Pigro("ccs_nuovo_l3", COD / "cardan_congiunto_scale.py")
TEACH_W = TMP / "teacher_wt"; TEACH_Q = TMP / "teacher_q8"
for d_ in (TEACH_W, TEACH_Q):
    d_.mkdir(exist_ok=True); (d_ / "piano.json").write_text(json.dumps({"finto": d_.name}))
BASE = ["--gguf", "finto.gguf", "--device", "cpu", "--teacher", str(TEACH_W), "--teacher-kl24", str(TEACH_Q),
        "--strati-studente", "0-2", "--strati", "0-2", "--rango-esperti", "4", "--rango-densi", "4", "--famiglie", "attn",
        "--init-c", "zero", "--kl4-n", "12", "--passi", "3", "--heldout-ogni", "1", "--pazienza", "100",
        "--heldout-righe", "2", "--pezzo-kl", "32", "--topk", "8", "--riscaldamento", "1", "--lr-c", "3e-3", "--lr-d", "1e-3",
        "--lr-densi", "3e-3", "--selezione", "kl24_paper"]
LEVE_L2 = ["--scale-down-lr", "1e-3", "--down-lowrank-rango", "4", "--down-lowrank-lr", "3e-3"]


def corri(mod, nome: str, extra: list, env: dict = None, rc_atteso: int = 0, ricetta: str = "B32", base=None) -> Path:
    out = TMP / nome
    vecchio = {k: os.environ.get(k) for k in (env or {})}
    os.environ.update(env or {})
    S.Studente = StudenteTernario
    StudenteTernario.RICETTA = ricetta
    try:
        rc = mod.main(["--uscita", str(out)] + (BASE if base is None else base) + extra)
    finally:
        S.Studente = S.Studente_vero
        StudenteTernario.RICETTA = "B32"
        for k, v in vecchio.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    assert rc == rc_atteso, f"{nome}: main returned {rc}, expected {rc_atteso}"
    return out


def tensori(out: Path, sotto: str = "") -> dict:
    d = out / sotto if sotto else out
    return {f.name: torch.load(f, map_location="cpu", weights_only=False) for f in sorted(d.glob("*.pt"))}


ESCLUSI = {"s", "secondi", "s_passo_medio"}


def uguali(a, b, dove="", esclusi=ESCLUSI) -> list:
    """Bitwise comparison of nested containers; returns the list of differences."""
    if isinstance(a, torch.Tensor):
        return [] if (isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)) \
            else [dove]
    if isinstance(a, dict):
        ka, kb = set(a) - esclusi, set(b) - esclusi
        d = [] if ka == kb else [f"{dove}: keys {sorted(ka ^ kb, key=str)}"]
        for k in ka:
            if k in kb:
                d += uguali(a[k], b[k], f"{dove}/{k}", esclusi)
        return d
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return [f"{dove}: len"]
        return sum((uguali(x, y, f"{dove}[{i}]", esclusi) for i, (x, y) in enumerate(zip(a, b))), [])
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return []
    return [] if a == b else [f"{dove}: {a!r} != {b!r}"]


def storia(out: Path, esclusi=ESCLUSI) -> list:
    s = json.load(open(out / "storia.json"))["storia"]
    return [{k: v for k, v in r.items() if k not in esclusi} for r in s]


def confronta(o1: Path, o2: Path, tappe: bool = False) -> list:
    d = uguali(tensori(o1), tensori(o2), "pt")
    d += uguali(storia(o1), storia(o2), "storia")
    if tappe:
        d += uguali(tensori(o1, "tappe"), tensori(o2, "tappe"), "tappe")
    return d


RISULTATI = []


def prova(nome):
    def dec(f):
        def run():
            t0 = time.time()
            try:
                msg = f()
                RISULTATI.append((nome, True, msg or ""))
                print(f"PASS {nome} {msg or ''} [{time.time() - t0:.1f} s, RSS max "
                      f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20:.2f} GiB]", flush=True)
            except Exception as e:                          # report and go on with the other tests
                RISULTATI.append((nome, False, f"{type(e).__name__}: {e}"))
                print(f"FAIL {nome}: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
        run.__name__ = f.__name__
        return run
    return dec


def _trim() -> None:
    """Give freed heap back to the OS (glibc), so the RSS cap measures what is really alive."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


# ───────────────────────────────────────── T0: the synthetic bytes are real TQ1_0R_1024 / TQ1_0 ───────────────────────
@prova("T0 byte sintetici: trit e scale tornano al bit; peso dello studente == dequant di riferimento (codec / gguf-py) al bit")
def t0():
    from gguf.quants import TQ1_0 as GQ_TQ1_0
    msg = []
    for tipo, (E, R, C) in (("TQ1_0R_1024", (2, 8, 2048)), ("TQ1_0", (2, 8, 512))):
        fam, T, s16 = famiglia("x", tipo, E, R, C, 7)
        nb = CS.LAYOUT[tipo][0]
        assert torch.equal(fam.d0.to(torch.float16), torch.from_numpy(s16)), "scales"
        assert np.array_equal(CAMPI.leggi_scale(fam.blocchi.numpy(), tipo, E, R).view(np.uint16), s16.view(np.uint16))
        for e in range(E):
            assert torch.equal(S.trit_esperto(fam, e), torch.from_numpy(T[e])), f"trits e{e}"
            W = S.peso_esperto_torch(fam, e, fam.d0, S.RICETTE["A32"]).numpy()        # recipe A: R^T (trit x d)
            blk = fam.blocchi[e].numpy().reshape(-1, nb)
            rif = (deq_r1024(blk) if tipo == "TQ1_0R_1024" else GQ_TQ1_0.dequantize_blocks(blk)).reshape(R, C)
            assert np.array_equal(W, rif), f"{tipo} e{e}: max |diff| {np.abs(W - rif).max():.3e}"
        msg.append(f"{tipo} {E}x{R}x{C}")
    return f"({', '.join(msg)})"


# ───────────────────────────────────────── F: flags off == original ───────────────────────────────────────────────────
@prova("F0 la copia di riferimento e' il trainer di L2/codice (sha)")
def f0():
    h = hashlib.sha256((QUI / "originale_L2_cardan_congiunto_scale.py").read_bytes()).hexdigest()
    assert h == ORIG_SHA, h
    return f"(= L2/codice vivo: {hashlib.sha256(L2_LIVE.read_bytes()).hexdigest() == ORIG_SHA})"


def coppia(nome: str, extra: list, env=None, extra_nuovo=None, tappe=False):
    o1 = corri(ORIG, f"{nome}_orig", extra, env)
    o2 = corri(NUOVO, f"{nome}_nuovo", extra + (extra_nuovo or []), env)
    d = confronta(o1, o2, tappe)
    assert not d, f"{len(d)} differences: {d[:5]}"
    return o1, o2


@prova("F1 flag spenti == originale: congiunto separato")
def f1():
    o1, _ = coppia("F1", [])
    return f"({len(tensori(o1))} file .pt, {len(storia(o1))} righe di storia al bit)"


@prova("F2 flag spenti == originale: --fuso")
def f2():
    coppia("F2", ["--fuso"])


@prova("F3 flag spenti == originale: alternato")
def f3():
    coppia("F3", ["--modo", "alternato", "--passi", "4"])


@prova("F4 flag spenti == originale: KL piena (--kl-peso 0.1)")
def f4():
    coppia("F4", ["--kl-peso", "0.1"])


@prova("F5 flag L3 spenti == originale con le leve di L2 accese (scale down + down low-rank) e --scale-gateup-lr 0 esplicito")
def f5():
    coppia("F5", LEVE_L2, extra_nuovo=["--scale-gateup-lr", "0"])


@prova("F6 flag spenti == originale attraverso una --riprendi (2 passi, poi fino a 4) con scale down + down low-rank")
def f6():
    for mod, n in ((ORIG, "F6_orig"), (NUOVO, "F6_nuovo")):
        t = str(TMP / n / "tappe")
        corri(mod, n, LEVE_L2 + ["--tappe-dir", t, "--passi", "2"])
        corri(mod, n, LEVE_L2 + ["--tappe-dir", t, "--passi", "4", "--riprendi"])
    d = confronta(TMP / "F6_orig", TMP / "F6_nuovo", tappe=True)
    assert not d, f"{len(d)} differences: {d[:5]}"


# ───────────────────────────────────────── N: the new switches ────────────────────────────────────────────────────────
@prova("N1 --rango-esperti 0 == rango 4 con C, D fermi (lr 0) su densi/down/scale/storia; nessun modulo mlp.cardan")
def n1():
    extra = LEVE_L2 + ["--scale-gateup-lr", "1e-3"]
    o0 = corri(NUOVO, "N1_r0", extra + ["--rango-esperti", "0"])
    st0 = StudenteTernario.ultimo
    assert all("cardan" not in st0.moduli[L].mlp._modules for L in st0.strati), "mlp.cardan installed with rank 0"
    assert all("cardan_down" in st0.moduli[L].mlp._modules for L in st0.strati)
    o4 = corri(NUOVO, "N1_r4", extra + ["--lr-c", "0", "--lr-d", "0"])
    t0_, t4_ = tensori(o0), tensori(o4)
    d = []
    for f in ("residui_ultimo.pt", "residui_migliore.pt", "cardan_down_ultimo.pt", "cardan_down_migliore.pt",
              "scale_ultimo.pt", "scale_gateup_ultimo.pt", "scale_gateup_migliore.pt"):
        d += uguali(t0_[f], t4_[f], f)
    for k in ("optD", "optS", "optDn", "optSgu", "v0", "migliore", "pegg", "passo"):
        d += uguali(t0_["ottimizzatori_ultimo.pt"][k], t4_["ottimizzatori_ultimo.pt"][k], k)
    es = ESCLUSI | {"norma_esperti", "blocchi"}
    d += uguali(storia(o0, es), storia(o4, es), "storia", es)
    assert not d, f"{len(d)} differences: {d[:6]}"
    c0 = t0_["cardan_ultimo.pt"]
    assert c0["rango"] == 0 and c0["stato"] == {}, c0
    assert json.load(open(o0 / "cardan.json"))["rango"] == 0
    assert all(r["blocchi"] == ["densi", "scale", "scale_gu", "down"] for r in storia(o0)[1:]), storia(o0)[1]["blocchi"]
    return "(densi, down, scale down/gate-up, Adam e storia al bit; cardan_*.pt vuoti, rango 0)"


def scale_da_pacco(pk: dict, L: int, f: str) -> np.ndarray:
    """fp16 scales an exporter would write for (L, f) of the synthetic student, from the pack's theta."""
    fam = famiglie_strato(L)[f][0]
    return SGE.scale_fp16(fam.d0.to(torch.float16).numpy(), pk["theta"][L][f], "cpu")


@prova("N2 --scale-gateup-lr (rango 0): theta si muove, pacco per tappa con sha delle scale fp16 usate, Adam, statistiche")
def n2():
    t = TMP / "N2" / "tappe"
    o = corri(NUOVO, "N2", LEVE_L2 + ["--scale-gateup-lr", "3e-3", "--rango-esperti", "0", "--tappe-dir", str(t)])
    s = storia(o)
    assert all("norma_scale_gu" in r and r["theta_gu_rms"] > 0 and r["scale_gu_cambiate"] > 0 for r in s[1:]), s[1]
    for p in (1, 2, 3):
        pk = torch.load(t / f"scale_gateup_{p}.pt", map_location="cpu", weights_only=False)
        assert pk["meta"]["passo"] == p and pk["dispositivo"] == "cpu"
        for L in range(N_STRATI):
            for f in ("gate", "up"):
                th = pk["theta"][L][f]
                assert th.dtype == torch.float32 and tuple(th.shape) == (E_N, FF, H // 128)
                assert float(th.abs().max()) > 0, (p, L, f)
                d = scale_da_pacco(pk, L, f)
                assert hashlib.sha256(d.tobytes()).hexdigest() == pk["sha256_fp16"][L][f], (p, L, f)
    ott = torch.load(o / "ottimizzatori_ultimo.pt", map_location="cpu", weights_only=False)
    stv = ott["optSgu"]["state"]
    assert len(stv) == 2 * N_STRATI and all(tuple(v["exp_avg"].shape) == (E_N, FF, H // 128) for v in stv.values())
    st_ = StudenteTernario.ultimo
    live = st_.moduli[2].mlp.esperti.theta["up"].detach()
    assert torch.equal(live, torch.load(o / "scale_gateup_ultimo.pt", weights_only=False)["theta"][2]["up"])
    assert json.load(open(o / "scale_gateup.json"))["famiglie"] == ["gate", "up"]
    r = s[-1]
    return (f"(theta rms {r['theta_gu_rms']:.2e} max {r['theta_gu_max']:.2e}, scale fp16 cambiate {100 * r['scale_gu_cambiate']:.1f} %, "
            f"norma grad {r['norma_scale_gu']:.3g}; sha studente == ricalcolo dell'esportatore su 3 tappe x 6 tensori)")


@prova("N2b --scale-gateup-lr anche con --fuso, --fuso + grafo tenuto (E2E_SENZA_RICALCOLO), alternato (rango 4), KL piena")
def n2b():
    msg = []
    for nome, extra, env in (("N2b_fuso", ["--fuso", "--rango-esperti", "0"], None),
                             ("N2b_grafo", ["--fuso", "--rango-esperti", "0"], {"E2E_SENZA_RICALCOLO": "1", "E2E_GRAFO_STRATI": "2"}),
                             ("N2b_alt", ["--modo", "alternato", "--passi", "4"], None),
                             ("N2b_klp", ["--kl-peso", "0.5", "--kl4-peso", "0", "--rango-esperti", "0"], None)):
        o = corri(NUOVO, nome, LEVE_L2 + ["--scale-gateup-lr", "3e-3"] + extra, env)
        s = storia(o)
        assert s[-1]["theta_gu_rms"] > 0, nome
        if nome == "N2b_alt":
            assert [r["blocchi"] for r in s[1:]] == [["densi"], ["esperti", "down", "scale_gu"]] * 2, [r["blocchi"] for r in s[1:]]
        msg.append(f"{nome} {s[-1]['theta_gu_rms']:.2e}")
    # the held graph must give the same theta as the recompute (same maths, kept instead of recomputed)
    a_, b_ = tensori(TMP / "N2b_fuso"), tensori(TMP / "N2b_grafo")
    e = max(rel(a_["scale_gateup_ultimo.pt"]["theta"][L][f], b_["scale_gateup_ultimo.pt"]["theta"][L][f])
            for L in range(N_STRATI) for f in ("gate", "up"))
    assert e <= 1e-5, e
    return f"({'; '.join(msg)}; grafo tenuto vs ricalcolo: theta err. rel. {e:.1e})"


@prova("N3 --riprendi con scale gate/up == corsa ininterrotta, al bit (theta, Adam optSgu, tappe, storia)")
def n3():
    ex = LEVE_L2 + ["--scale-gateup-lr", "3e-3", "--rango-esperti", "0"]
    a_ = corri(NUOVO, "N3_dritta", ex + ["--tappe-dir", str(TMP / "N3_dritta" / "tappe"), "--passi", "4"])
    t = str(TMP / "N3_ripresa" / "tappe")
    corri(NUOVO, "N3_ripresa", ex + ["--tappe-dir", t, "--passi", "2"])
    b_ = corri(NUOVO, "N3_ripresa", ex + ["--tappe-dir", t, "--passi", "4", "--riprendi"])
    d = confronta(a_, b_, tappe=True)
    assert not d, f"{len(d)} differences: {d[:5]}"
    assert "ripresa: scale gate/up dal passo 2" in (b_ / "registro.log").read_text()
    # a checkpoint whose gate/up file is missing must refuse (code 4), not restart the scales silently
    ott = torch.load(b_ / "ottimizzatori_ultimo.pt", weights_only=False)
    assert "optSgu" in ott
    return f"({len(tensori(a_)) + len(tensori(a_, 'tappe'))} file .pt e {len(storia(a_))} righe al bit)"


@prova("N4 cancello di presenza: scale gate/up fuori dal percorso -> codice 2; errori di configurazione -> codice 5")
def n4():
    vero = S.StratoEsperti.scala

    def finta(self, nome):                              # theta ignored for gate/up: a student that would not train them
        if nome in ("gate", "up"):
            return self.fam[nome].d0.to(self.ric.dtype_gemm)
        return vero(self, nome)
    S.StratoEsperti.scala = finta
    try:
        corri(NUOVO, "N4_fuori", ["--scale-gateup-lr", "1e-3"], rc_atteso=2)
    finally:
        S.StratoEsperti.scala = vero
    assert "fuori dal percorso dello studente" in (TMP / "N4_fuori" / "registro.log").read_text()
    corri(NUOVO, "N4_alt0", ["--rango-esperti", "0", "--modo", "alternato"], rc_atteso=5)
    corri(NUOVO, "N4_neg", ["--scale-gateup-lr", "-1"], rc_atteso=5)


# ───────────────────────────────────────── G: gradient checks of theta ────────────────────────────────────────────────
def hadamard(n: int) -> torch.Tensor:
    Hm = torch.ones(1, 1, dtype=torch.float64)
    while Hm.shape[0] < n:
        Hm = torch.cat([torch.cat([Hm, Hm], 1), torch.cat([Hm, -Hm], 1)], 0)
    return Hm


class Identita:
    """fp16 rounding of the scales replaced by the identity (the STE point): the student becomes smooth in theta."""
    def __enter__(self):
        self.v = S.ArrotondaFP16.apply
        S.ArrotondaFP16.apply = lambda s, sent: s
        return self

    def __exit__(self, *e):
        S.ArrotondaFP16.apply = self.v


def strato_g(ric_nome: str, H_=2048, FF_=512, E=4, d64=True):
    g = torch.Generator().manual_seed(4321)
    ric = S.RICETTE[ric_nome]
    moe = moe_vero(5, g, ric.dtype_gemm, ric, S.MotoreTorch(), H_, FF_, E)
    trits = {f: torch.from_numpy(v[1]) for f, v in famiglie_strato(5, H_, FF_, E).items()}
    es = moe.esperti
    for f in ("gate", "up", "down"):
        if d64:
            es.fam[f].d0 = es.fam[f].d0.double()
        es.theta[f] = torch.zeros_like(es.fam[f].d0, dtype=torch.float64 if d64 else torch.float32)
    return moe, trits


def riferimento(moe, trits, x, theta: dict, H_=2048):
    """Independent dense MoE: W from the GENERATED trits and s = d0 exp(theta) (no rounding), own rotation x_rot = H(S x)/32
    (Sylvester Hadamard), routing/weights recomputed from the router; shared expert = the same module."""
    es = moe.esperti
    T_ = x.shape[0]
    Hm = hadamard(1024)
    Sg = torch.from_numpy(S1024.astype(np.float64))
    xr = ((x.reshape(T_, H_ // 1024, 1024) * Sg) @ Hm.T / 32).reshape(T_, H_)
    prob = torch.softmax(x @ moe.router_w.double().T, -1)
    idx = torch.topk(prob, TOPK, -1).indices
    w = prob.gather(-1, idx); w = w / w.sum(-1, keepdim=True)
    y = torch.zeros(T_, H_, dtype=torch.float64)
    Wf = {}
    for f in ("gate", "up", "down"):
        fam = es.fam[f]
        s = fam.d0.double() * torch.exp(theta[f])                                  # (E, R, NG)
        Wf[f] = (trits[f].double().view(fam.E, fam.R, fam.NG, fam.gs) * s.unsqueeze(-1)).view(fam.E, fam.R, fam.C)
    for t in range(T_):
        for j in range(TOPK):
            e = int(idx[t, j])
            gg, uu = Wf["gate"][e] @ xr[t], Wf["up"][e] @ xr[t]
            y[t] += w[t, j] * (Wf["down"][e] @ (torch.nn.functional.silu(gg) * uu))
    return y + torch.sigmoid(moe.shared_expert_gate(x)) * moe.shared_expert(x), idx


@prova("G1 dL/dtheta (gate, up, down) del MoEStudente vero == riferimento denso indipendente == differenze finite (B64, A64)")
def g1():
    msg = []
    for ric in ("B64", "A64"):
        moe, trits = strato_g(ric)
        es = moe.esperti
        x = torch.randn(24, 2048, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
        wr = torch.randn(24, 2048, generator=torch.Generator().manual_seed(10), dtype=torch.float64)
        g0 = torch.Generator().manual_seed(11)
        th0 = {f: 0.02 * torch.randn(es.fam[f].d0.shape, generator=g0, dtype=torch.float64) for f in ("gate", "up", "down")}

        def f_studente(th: dict, grad: bool):
            for f in th:
                es.theta[f] = th[f].clone().requires_grad_(grad)
            y = moe(x.view(1, 24, 2048)).view(24, 2048)
            return (y * wr).sum(), y
        with Identita():
            L_s, y_s = f_studente(th0, True)
            L_s.backward()
            g_s = {f: es.theta[f].grad.clone() for f in th0}
            idx_s = moe.ultimo_idx.clone()
            thr = {f: th0[f].clone().requires_grad_(True) for f in th0}
            y_r, idx_r = riferimento(moe, trits, x, thr)
            assert torch.equal(torch.sort(idx_s, -1).values, torch.sort(idx_r, -1).values), "routing"
            (y_r * wr).sum().backward()
            e_y = rel(y_s.detach(), y_r.detach())
            e_g = max(rel(g_s[f], thr[f].grad) for f in th0)
            assert e_y <= 1e-12 and e_g <= 1e-10, (ric, e_y, e_g)
            e_fd = 0.0
            for k in range(3):                              # central differences along random directions
                gv = torch.Generator().manual_seed(100 + k)
                v = {f: torch.randn(th0[f].shape, generator=gv, dtype=torch.float64) for f in th0}
                eps = 1e-6
                with torch.no_grad():
                    lp = f_studente({f: th0[f] + eps * v[f] for f in th0}, False)[0]
                    lm = f_studente({f: th0[f] - eps * v[f] for f in th0}, False)[0]
                num = float((lp - lm) / (2 * eps))
                ana = float(sum((g_s[f] * v[f]).sum() for f in th0))
                e_fd = max(e_fd, abs(num - ana) / max(abs(ana), 1e-30))
            assert e_fd <= 1e-6, (ric, e_fd)
        msg.append(f"{ric}: uscita {e_y:.1e}, grad vs rif. {e_g:.1e}, vs diff. finite {e_fd:.1e}")
    # STE point: with the REAL fp16 rounding at theta = 0 (d0 is fp16-exact) forward and gradient are those of the smooth
    # function at the bit (production arithmetic: fp32 theta and d0)
    moe, _ = strato_g("B32", d64=False)
    es = moe.esperti
    x = torch.randn(16, 2048, generator=torch.Generator().manual_seed(12)).float()
    wr = torch.randn(16, 2048, generator=torch.Generator().manual_seed(13)).float()
    gg = []
    for liscia in (False, True):
        for f in ("gate", "up"):
            es.theta[f] = torch.zeros_like(es.fam[f].d0, dtype=torch.float32).requires_grad_(True)
        if liscia:
            with Identita():
                y = moe(x.view(1, 16, 2048)).view(16, 2048); (y * wr).sum().backward()
        else:
            y = moe(x.view(1, 16, 2048)).view(16, 2048); (y * wr).sum().backward()
        gg.append([es.theta[f].grad.clone() for f in ("gate", "up")] + [y.detach()])
    assert all(torch.equal(a, b) for a, b in zip(gg[0], gg[1])), "STE at theta = 0"
    return "; ".join(msg) + "; arrotondamento fp16 vero a theta = 0 == liscio al bit (B32)"


@prova("G2 indietro dell'allenatore (avanti + ricalcolo per strato, strati_allenati) == autograd a grafo intero, fp64")
def g2():
    StudenteTernario.RICETTA = "B64"
    try:
        st = StudenteTernario()
    finally:
        StudenteTernario.RICETTA = "B32"
    for L in st.strati:
        for f in ("gate", "up", "down"):
            st.moduli[L].mlp.esperti.theta[f] = (0.01 * torch.randn(st.moduli[L].mlp.esperti.fam[f].d0.shape,
                                                 generator=torch.Generator().manual_seed(L * 3 + len(f)))).requires_grad_(True)
    ids = torch.randint(0, 38, (1, 40), generator=torch.Generator().manual_seed(3))
    Wr = torch.randn(40, H, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    perdita = lambda h: (h.double() * Wr).sum()
    pars = [st.moduli[L].mlp.esperti.theta[f] for L in st.strati for f in ("gate", "up")]
    with Identita():
        st.strati_allenati = list(st.strati)
        st.gradiente(ids, perdita)
        ga = [p.grad.clone() for p in pars]
        for p in pars:
            p.grad = None
        st.strati_allenati = None
        perdita(st.nascosto(ids)).backward()
        gb = [p.grad.clone() for p in pars]
    e = max(rel(a, b) for a, b in zip(ga, gb))
    assert e <= 1e-12, e
    return f"({len(pars)} tensori theta gate/up, errore relativo massimo {e:.1e})"


# ───────────────────────────────────────── E: exporter ────────────────────────────────────────────────────────────────
def gguf_giocattolo(p: Path) -> list:
    """Toy GGUF with the synthetic student's expert tensors (layers 0-2, host shapes) + one f32 tensor."""
    from gguf import GGUFWriter, GGMLQuantizationType as GT
    w = GGUFWriter(str(p), arch="qwen35moe")
    w.add_uint32("l3.prova", 1)
    nomi = []
    for L in range(N_STRATI):
        fs = famiglie_strato(L)
        for f, tipo in (("gate", GT.TQ1_0R_1024), ("up", GT.TQ1_0R_1024), ("down", GT.TQ1_0)):
            n = f"blk.{L}.ffn_{f}_exps.weight"
            w.add_tensor(n, fs[f][0].blocchi.numpy(), raw_dtype=tipo); nomi.append(n)
    w.add_tensor("output_norm.weight", np.arange(H, dtype=np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    return nomi


def patch(dir_t: Path, out: Path, base: Path) -> None:
    env = {**os.environ, "LEVE_THREADS": "4", "OMP_NUM_THREADS": "4", "HIP_VISIBLE_DEVICES": "", "CUDA_VISIBLE_DEVICES": "",
           "LEVE_OUT": str(TMP / "leve_out")}
    r = subprocess.run(["nice", "-n", "19", PY, str(PATCH), "--dir", str(dir_t), "--out", str(out), "--base", str(base)],
                       env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]


@prova("E1 pacco dell'allenatore -> scale_gateup_esporta -> patch_gguf: solo i campi d cambiano, scale = studente (sha), "
       "dequant llama.cpp del file == peso allenato dello studente al bit; theta 0 = identita'")
def e1():
    from gguf import GGUFReader
    src = TMP / "E1" / "giocattolo.gguf"; src.parent.mkdir(exist_ok=True)
    nomi = gguf_giocattolo(src)
    pacco = TMP / "N2" / "tappe" / "scale_gateup_3.pt"
    if not pacco.exists():
        n2()
    pk = torch.load(pacco, map_location="cpu", weights_only=False)
    dt_ = TMP / "E1" / "tensori"
    assert SGE.main([str(src), str(pacco), str(dt_), "--controlla-base", str(src)]) == 0
    man = json.load(open(dt_ / "scale_gateup_esporta.json"))
    assert man["sha_diversi"] == 0 and all(v["sha_come_studente"] for v in man["tensori"].values())
    dst = TMP / "E1" / "giocattolo_l3.gguf"
    patch(dt_, dst, src)
    nomi_gu = [n for n in nomi if "_gate_" in n or "_up_" in n]
    rep = CS.verifica_patch(src, dst, nomi_gu)            # raises if any byte outside the gate/up d fields changed
    td = {t.name: t for t in GGUFReader(str(dst)).tensors}
    ric_a = S.RICETTE["A32"]
    n_w = 0
    for L in range(N_STRATI):
        fams = {f: v[0] for f, v in famiglie_strato(L).items()}
        es = S.StratoEsperti(fams, S.RICETTE["B32"], S.MotoreTorch())
        for f in ("gate", "up"):
            es.theta[f] = pk["theta"][L][f].clone()
            d_st = es.scala(f).detach()                       # the student's fp16 scales (as fp32) = what it trained with
            raw = np.asarray(td[f"blk.{L}.ffn_{f}_exps.weight"].data)
            d_file = CAMPI.leggi_scale(raw, "TQ1_0R_1024", E_N, FF)
            assert hashlib.sha256(d_file.tobytes()).hexdigest() == pk["sha256_fp16"][L][f]
            assert np.array_equal(d_file.view(np.uint16), d_st.to(torch.float16).numpy().view(np.uint16))
            for e in range(E_N):
                W_st = S.peso_esperto_torch(fams[f], e, d_st, ric_a).numpy()      # R^T(trit x d), trits of the HOST
                W_ll = deq_r1024(np.ascontiguousarray(raw[e]).reshape(-1, 224)).reshape(FF, H)
                assert np.array_equal(W_st, W_ll), (L, f, e, float(np.abs(W_st - W_ll).max()))
                n_w += 1
    # theta = 0 pack: identity at the byte
    pk0 = {"theta": {L: {f: torch.zeros_like(pk["theta"][L][f]) for f in ("gate", "up")} for L in range(N_STRATI)}}
    torch.save(pk0, TMP / "E1" / "theta0.pt")
    assert SGE.main([str(src), str(TMP / "E1" / "theta0.pt"), str(TMP / "E1" / "tensori0")]) == 0
    ts = {t.name: t for t in GGUFReader(str(src)).tensors}
    for n in nomi_gu:
        assert np.array_equal(np.load(TMP / "E1" / "tensori0" / f"{n}.npy"), np.asarray(ts[n].data)), n
    return (f"({len(nomi_gu)} tensori, {sum(v['scale_cambiate'] for v in man['tensori'].values())} scale cambiate; "
            f"{n_w} esperti dequant == studente; sha senza d {rep['sha256_senza_d'][:12]})")


@prova("E1b rifiuti: sha diversa dallo studente -> 3 (salvo --senza-sha); base di allenamento diversa -> 3")
def e1b():
    src = TMP / "E1" / "giocattolo.gguf"
    if not src.exists():
        e1()
    pk = torch.load(TMP / "N2" / "tappe" / "scale_gateup_3.pt", map_location="cpu", weights_only=False)
    pk["sha256_fp16"][1]["up"] = "0" * 64
    torch.save(pk, TMP / "E1" / "sha_rotta.pt")
    assert SGE.main([str(src), str(TMP / "E1" / "sha_rotta.pt"), str(TMP / "E1" / "t_rotta")]) == 3
    assert SGE.main([str(src), str(TMP / "E1" / "sha_rotta.pt"), str(TMP / "E1" / "t_rotta2"), "--senza-sha"]) == 0
    altra = TMP / "E1" / "altra.gguf"
    from gguf import GGUFWriter, GGMLQuantizationType as GT
    w = GGUFWriter(str(altra), arch="qwen35moe")
    for L in range(N_STRATI):
        fs = famiglie_strato(L)
        for f, tipo in (("gate", GT.TQ1_0R_1024), ("up", GT.TQ1_0R_1024)):
            b = fs[f][0].blocchi.numpy().copy()
            if L == 2 and f == "up":
                b[0, 0, 0] ^= 1                              # one code byte different
            w.add_tensor(f"blk.{L}.ffn_{f}_exps.weight", b, raw_dtype=tipo)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    assert SGE.main([str(altra), str(TMP / "N2" / "tappe" / "scale_gateup_3.pt"), str(TMP / "E1" / "t_base"),
                     "--controlla-base", str(src)]) == 3


@prova("E2 UNO strato VERO (L12 della base di allenamento): codici identici, scale == StratoEsperti._scala su "
       "famiglia_da_tensore, rilettura .npy; theta 0 = byte identici; fp32 contro fp64")
def e2():
    from gguf import GGUFReader
    if not BASE_VERA.exists():
        raise FileNotFoundError(BASE_VERA)
    import gc
    rd = GGUFReader(str(BASE_VERA), "r")                     # one reader alive at a time (RSS: the vocabulary is large)
    tens = {t.name: t for t in rd.tensors}
    L = 12
    fams = {f: S.famiglia_da_tensore(tens[f"blk.{L}.ffn_{f}_exps.weight"], f) for f in ("gate", "up")}
    grezzi = {f: np.array(tens[f"blk.{L}.ffn_{f}_exps.weight"].data) for f in ("gate", "up")}      # private copies
    nbyte = {f: int(tens[f"blk.{L}.ffn_{f}_exps.weight"].n_bytes) for f in ("gate", "up")}
    del rd, tens
    gc.collect()
    _trim()
    es = S.StratoEsperti(fams, S.RICETTE["B32"], S.MotoreTorch())
    g = torch.Generator().manual_seed(1212)
    th = {f: 2e-3 * torch.randn(fams[f].d0.shape, generator=g) for f in ("gate", "up")}
    sha, d_st = {}, {}
    with torch.no_grad():
        for f in ("gate", "up"):
            es.theta[f] = th[f].clone()
            d_st[f] = es._scala(f).to(torch.float16).numpy()        # the student's own arithmetic (CPU)
            sha[f] = hashlib.sha256(d_st[f].tobytes()).hexdigest()
    del es, fams
    gc.collect()
    _trim()
    pk = {"theta": {L: th}, "sha256_fp16": {L: sha}, "dispositivo": "cpu", "meta": {"passo": -1, "prova": "E2"}}
    torch.save(pk, TMP / "E2_pacco_L12.pt")
    out = TMP / "E2_L12"
    assert SGE.main([str(BASE_VERA), str(TMP / "E2_pacco_L12.pt"), str(out), "--strati", "12"]) == 0
    msg = []
    for f in ("gate", "up"):
        raw = grezzi[f]
        nuovo = np.load(out / f"blk.{L}.ffn_{f}_exps.weight.npy")
        assert nuovo.dtype == np.uint8 and nuovo.shape == raw.shape and nuovo.nbytes == nbyte[f]
        m = CAMPI.maschera_codici("TQ1_0R_1024")
        Bv, Bn = CAMPI.blocchi(raw, "TQ1_0R_1024"), CAMPI.blocchi(nuovo, "TQ1_0R_1024")
        assert np.array_equal(Bv[:, m], Bn[:, m]), "codes"
        assert np.array_equal(CAMPI.leggi_scale(nuovo, "TQ1_0R_1024", raw.shape[0], raw.shape[1]).view(np.uint16),
                              d_st[f].view(np.uint16)), "scales"
        tr_new = CAMPI.trit_torch(torch.from_numpy(Bn[:4096].copy()), "TQ1_0R_1024")
        tr_old = CAMPI.trit_torch(torch.from_numpy(Bv[:4096].copy()), "TQ1_0R_1024")
        assert torch.equal(tr_new, tr_old)
        d0 = CAMPI.leggi_scale(raw, "TQ1_0R_1024", raw.shape[0], raw.shape[1])
        d64 = (torch.from_numpy(d0.astype(np.float64)) * torch.exp(th[f].double())).to(torch.float16).numpy()
        n64 = int((d64.view(np.uint16) != d_st[f].view(np.uint16)).sum())
        camb = int((d0.view(np.uint16) != d_st[f].view(np.uint16)).sum())
        msg.append(f"{f}: {camb}/{d0.size} scale cambiate, fp64 diverso da fp32 in {n64}")
    for f in ("gate", "up"):                            # theta = 0 through the exporter's own function, in memory
        nuovo0, _, ver0 = SGE.esporta_tensore(grezzi[f], "TQ1_0R_1024", torch.zeros_like(th[f]))
        assert ver0["scale_cambiate"] == 0 and np.array_equal(nuovo0, grezzi[f]), f
        del nuovo0
    return "(" + "; ".join(msg) + "; theta 0 -> byte identici)"


if __name__ == "__main__":
    print(f"cartella {TMP}", flush=True)
    tutte = {f.__name__: f for f in (t0, f0, g1, g2, f1, f2, f3, f4, f5, f6, n1, n2, n2b, n3, n4, e1, e1b, e2)}
    for nome in (sys.argv[1:] or list(tutte)):         # e.g. `test_l3.py e2` runs only the real-layer test
        tutte[nome]()
    ok = sum(1 for _, b, _ in RISULTATI if b)
    print(f"\n{ok}/{len(RISULTATI)} PASS (cartella {TMP}, RSS max "
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20:.2f} GiB)")
    sys.exit(0 if ok == len(RISULTATI) else 1)

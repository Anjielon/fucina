#!/usr/bin/env python3
"""CPU tests of lever L6 (SINTESI_1025, 26/10): batch-level answer-letter prior term (--prior-lettere-peso W,
--prior-lettere-pacchi N, --prior-lettere-ema BETA) in cardan_congiunto_scale.py + prior_lettere.py. No GPU, no live
service, no deletion. Same synthetic harness as test_l3 (copied, not imported: test_l3 writes into the L3 tree): the student
subclasses the real e2e_studente.Studente with REAL MoEStudente / StratoEsperti layers on synthetic TQ1_0R_1024 / TQ1_0
bytes, MotoreTorch on CPU; the trainer's `main` runs for real. Teacher caches, KL4 packs and heads are synthetic.

  F*  flags off == original (test_l6/originale_L3_cardan_congiunto_scale.py = L3/codice trainer, sha e3e17046...): every
      saved tensor and every logged value bitwise equal (separate, fused, alternating, KL4 0.25 + full KL like the live
      KLPPRIOR run, KL4 0 + full KL like KLP, L2/L3 levers on with the L6 knobs set but the weight 0, through --riprendi)
  G*  gradients: the term vs torch.autograd.gradcheck (fp64) and vs the closed form -(W/Q)(diag p_q - p_q p_q^T) pbarT/pbarS;
      EMA = closed form at the running point, literal derivative = (1 - beta) x; inside the trainer (step-1 gradients of
      every block, captured before clipping): with one question per pack the prior term IS the KL4 of that question (plain
      and --fuso), with N identical one-question packs the N-pack batch equals the single KL4 pack, KL4 + prior together
      = paper + 2 x KL4, gradient linear in W with multi-question packs
  T*  toy models: the prior term brings pbar_S to pbar_T while per-question distributions stay free (free logits: per-question
      KL ends at the shared-offset oracle, not at 0 like a KL4 fit); permutation invariance across questions; an
      over-parameterised toy (features -> letters) reproduces R6: KL4 memorises the train questions, the prior term fixes
      the held-out marginal without per-question fitting
  N*  trainer runs with the term on (plain, --fuso, N packs, alternating, KL piena, EMA, L2/L3 levers): keys, finiteness;
      --riprendi with EMA bitwise == uninterrupted; config errors -> code 5; a NaN gradient norm in a non-first block
      skips the step (the original takes it and the parameters become NaN)
Run (one job, 4 threads, nice 19, RSS < 2 GB):
    HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 nice -n 19 \\
      @@ROOT:venv@@/bin/python3 test_l6/test_l6.py
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
ORIG_SHA = "e3e1704664369404d91a01a07399f3a9d2d51b18e0d3feea8b78968d9f3d9818"     # L3/codice/cardan_congiunto_scale.py
REF_LIVE = Path("@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/SINTESI_1025/L3/codice/cardan_congiunto_scale.py")
BASE_VERA = Path("@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/CARDAN_DOWN_1025/Thor-T-C-S7g10-DOWNH.gguf")
PATCH = Path("@@ROOT:models@@/build/leve/raffina/patch_gguf.py")
PY = "@@ROOT:venv@@/bin/python3"

import cardan_kd_paper as KDP                       # noqa: E402  (path setup of the thor_v2 tree)
import e2e_studente as S                            # noqa: E402
import e2e_allena as A                              # noqa: E402
import e2e_codici_scale as CS                       # noqa: E402
import e2e_campi_scala as CAMPI                     # noqa: E402
from tq1_0r_1024_codec import _pack_quarter, dequantize_blocks as deq_r1024, S1024   # noqa: E402

KDP.riga_cicli = lambda *a, **k: None               # never write the real cicli.tsv
V, H, FF, E_N, TOPK, N_STRATI, T_RIGA = 48, 1024, 512, 4, 2, 3, 64
LETTERE = [40, 41, 42, 43]
TMP = Path("@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/SINTESI_1025/L6/collaudo") / time.strftime("test_%Y%m%d_%H%M%S")
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


LIMITE_PACCO = [40]                                  # tokens per synthetic KL4 pack (1 = one question per pack)
IDENTICHE = [False]                                  # True = all questions share ids and teacher distribution


def domande():
    g = torch.Generator().manual_seed(77)
    Q = []
    for i in range(20):
        if IDENTICHE[0]:
            g = torch.Generator().manual_seed(77)
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
        if cur and nt + len(q["ids"]) > LIMITE_PACCO[0]:
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


ORIG = _Pigro("ccs_originale_l3", QUI / "originale_L3_cardan_congiunto_scale.py")
NUOVO = _Pigro("ccs_nuovo_l6", COD / "cardan_congiunto_scale.py")
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


import prior_lettere as PLm                         # noqa: E402


class CatturaGrad:
    """Records, for every call of torch.nn.utils.clip_grad_norm_ (one per active block per step, in block order), a copy
    of the gradients BEFORE clipping; optionally poisons one call with a NaN (as a real NaN gradient would do)."""

    def __init__(self, nan_alla_chiamata: int = -1):
        self.chiamate, self.vero, self.nan = [], torch.nn.utils.clip_grad_norm_, nan_alla_chiamata

    def __enter__(self):
        def f(params, max_norm, *a, **k):
            ps = list(params)
            if len(self.chiamate) == self.nan:
                with torch.no_grad():
                    ps[0].grad.view(-1)[0] = float("nan")
            self.chiamate.append([None if p.grad is None else p.grad.detach().clone() for p in ps])
            return self.vero(ps, max_norm, *a, **k)
        torch.nn.utils.clip_grad_norm_ = f
        return self

    def __exit__(self, *e):
        torch.nn.utils.clip_grad_norm_ = self.vero


def grad_rel(c1: CatturaGrad, c2: CatturaGrad) -> float:
    """max over the captured calls of the relative error of the concatenated gradients."""
    assert len(c1.chiamate) == len(c2.chiamate), (len(c1.chiamate), len(c2.chiamate))
    e = 0.0
    for a_, b_ in zip(c1.chiamate, c2.chiamate):
        A_ = torch.cat([x.reshape(-1).double() for x in a_ if x is not None])
        B_ = torch.cat([x.reshape(-1).double() for x in b_ if x is not None])
        e = max(e, rel(A_, B_))
    return e


def grad_vett(c: CatturaGrad) -> torch.Tensor:
    return torch.cat([x.reshape(-1).double() for ch in c.chiamate for x in ch if x is not None])


class Pacchi:
    """Context: synthetic KL4 packs of `limite` tokens (1 = one question per pack), optionally all questions identical."""

    def __init__(self, limite=40, identiche=False):
        self.l, self.i = limite, identiche

    def __enter__(self):
        LIMITE_PACCO[0], IDENTICHE[0] = self.l, self.i

    def __exit__(self, *e):
        LIMITE_PACCO[0], IDENTICHE[0] = 40, False


def kl_(t, p):
    return (t * (torch.log(t.clamp_min(1e-30)) - torch.log(p.clamp_min(1e-30)))).sum(-1)


UNO = ["--passi", "1"]
PRIOR = ["--kl4-peso", "0", "--prior-lettere-peso", "1"]


# ───────────────────────────────────────── F: flags off == original ───────────────────────────────────────────────────
@prova("F0 la copia di riferimento e' il trainer di L3/codice (sha)")
def f0():
    h = hashlib.sha256((QUI / "originale_L3_cardan_congiunto_scale.py").read_bytes()).hexdigest()
    assert h == ORIG_SHA, h
    return f"(= L3/codice vivo: {hashlib.sha256(REF_LIVE.read_bytes()).hexdigest() == ORIG_SHA})"


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


@prova("F4 flag spenti == originale: KL4 0.25 + KL piena (come la corsa KLPPRIOR viva)")
def f4():
    coppia("F4", ["--kl4-peso", "0.25", "--kl-peso", "0.1"] + LEVE_L2)


@prova("F4b flag spenti == originale: KL4 0 + KL piena (come KLP)")
def f4b():
    coppia("F4b", ["--kl4-peso", "0", "--kl-peso", "0.1"])


@prova("F5 leve L2/L3 accese, manopole L6 impostate ma peso 0 == originale")
def f5():
    coppia("F5", LEVE_L2 + ["--scale-gateup-lr", "1e-3"],
           extra_nuovo=["--prior-lettere-peso", "0", "--prior-lettere-pacchi", "3", "--prior-lettere-ema", "0.9"])


@prova("F6 flag spenti == originale attraverso una --riprendi (2 passi, poi fino a 4), con tappe")
def f6():
    for mod, n in ((ORIG, "F6_orig"), (NUOVO, "F6_nuovo")):
        t = str(TMP / n / "tappe")
        corri(mod, n, LEVE_L2 + ["--tappe-dir", t, "--passi", "2"])
        corri(mod, n, LEVE_L2 + ["--tappe-dir", t, "--passi", "4", "--riprendi"])
    d = confronta(TMP / "F6_orig", TMP / "F6_nuovo", tappe=True)
    assert not d, f"{len(d)} differences: {d[:5]}"


# ───────────────────────────────────────── G: gradients ───────────────────────────────────────────────────────────────
@prova("G1 termine: gradcheck fp64, forma chiusa -(1/Q)(diag p - p p^T) pbarT/pbarS, EMA = (1-beta) x forma chiusa al punto EMA")
def g1():
    g = torch.Generator().manual_seed(3)
    taglie = [5, 3, 7]
    z = [torch.randn(q, 4, generator=g, dtype=torch.float64).requires_grad_(True) for q in taglie]
    t = [torch.softmax(1.5 * torch.randn(q, 4, generator=g, dtype=torch.float64), -1) for q in taglie]
    pr = PLm.PriorLettere(1.0, 3, 0.0)

    def f(*zz):
        return pr.termine([torch.log_softmax(x, -1) for x in zz], t)[0]
    assert torch.autograd.gradcheck(f, tuple(z), eps=1e-6, atol=1e-10, rtol=1e-6)
    P = torch.cat([torch.softmax(x, -1) for x in z]).detach(); Q = P.shape[0]
    r = torch.cat(t).mean(0) / P.mean(0)
    cf = -(P * r - P * (P @ r)[:, None]) / Q
    e_cf = rel(torch.cat(torch.autograd.grad(f(*z), z)), cf)
    assert e_cf < 1e-12, e_cf
    v = float(f(*z)); v_rif = float(kl_(torch.cat(t).mean(0), P.mean(0)))
    assert abs(v - v_rif) < 1e-14, (v, v_rif)
    # EMA (beta 0.9): step 1 = batch values; step 2 = loss KL(rT || rS), literal derivative through the batch only
    pe = PLm.PriorLettere(1.0, 1, 0.9)
    z1 = torch.randn(6, 4, generator=g, dtype=torch.float64); t1 = torch.softmax(torch.randn(6, 4, generator=g, dtype=torch.float64), -1)
    z1 = z1.requires_grad_(True)
    k1, i1 = pe.termine([torch.log_softmax(z1, -1)], [t1])
    assert abs(float(k1) - float(kl_(t1.mean(0), torch.softmax(z1, -1).mean(0)))) < 1e-14 and "prior_batch" in i1
    gb1 = torch.autograd.grad(PLm.PriorLettere(1.0).termine([torch.log_softmax(z1, -1)], [t1])[0], z1)[0]
    e_1 = rel(torch.autograd.grad(k1, z1)[0], 0.1 * gb1)          # step 1: EMA starts at the batch, gain (1 - beta)
    assert e_1 < 1e-12, e_1
    pe.conferma(); S1, T1 = pe.S.clone(), pe.T.clone()
    assert torch.allclose(S1, torch.softmax(z1, -1).detach().mean(0), rtol=0, atol=1e-15)
    z2 = torch.randn(4, 4, generator=g, dtype=torch.float64).requires_grad_(True)
    t2 = torch.softmax(torch.randn(4, 4, generator=g, dtype=torch.float64), -1)
    k2, i2 = pe.termine([torch.log_softmax(z2, -1)], [t2])
    P2 = torch.softmax(z2, -1).detach()
    rS, rT = 0.9 * S1 + 0.1 * P2.mean(0), 0.9 * T1 + 0.1 * t2.mean(0)
    assert abs(float(k2) - float(kl_(rT, rS))) < 1e-14
    g2 = torch.autograd.grad(k2, z2)[0]
    r2 = rT / rS
    cf2 = -0.1 * (P2 * r2 - P2 * (P2 @ r2)[:, None]) / P2.shape[0]
    e_ema = rel(g2, cf2)
    assert e_ema < 1e-12, e_ema
    lit = lambda zz: kl_(rT, 0.9 * S1 + 0.1 * torch.softmax(zz, -1).mean(0))
    assert torch.autograd.gradcheck(lit, (z2,), eps=1e-6, atol=1e-10, rtol=1e-6)
    e_lit = rel(g2, torch.autograd.grad(lit(z2), z2)[0])
    assert e_lit < 1e-12, e_lit
    pe.scarta(); assert torch.equal(pe.S, S1) and pe.n == 1, "scarta must not move the running estimate"
    pe.termine([torch.log_softmax(z2, -1)], [t2]); pe.conferma()
    assert torch.allclose(pe.S, rS) and pe.n == 2
    return f"(forma chiusa {e_cf:.1e}, EMA {e_ema:.1e}, derivata letterale {e_lit:.1e})"


@prova("G2 nel trainer, una domanda per pacco: prior (KL4 0) == KL4 (prior 0), gradienti del passo 1 di tutti i blocchi, separato e --fuso")
def g2():
    msg = []
    with Pacchi(1):
        for nome, ex in (("sep", []), ("fuso", ["--fuso"])):
            with CatturaGrad() as cA:
                corri(NUOVO, f"G2_{nome}_kl4", UNO + ex + ["--kl4-peso", "1"])
            with CatturaGrad() as cB:
                oB = corri(NUOVO, f"G2_{nome}_prior", UNO + ex + PRIOR)
            e = grad_rel(cA, cB)
            assert e < 1e-5, (nome, e)
            sA, sB = storia(TMP / f"G2_{nome}_kl4"), storia(oB)
            assert abs(sA[1]["kl4"] - sB[1]["prior_lettere"]) <= 3e-6 * max(1.0, sA[1]["kl4"]), (sA[1], sB[1])
            assert "kl4" not in sB[1] and "held_prior" in sB[1]
            msg.append(f"{nome} {e:.1e}")
    return f"(err. rel. gradienti: {', '.join(msg)})"


@prova("G3 nel trainer: N pacchi identici da una domanda (N = 3) == il solo pacco KL4; KL4 + prior insieme == paper + 2 x KL4")
def g3():
    with Pacchi(1, identiche=True):
        with CatturaGrad() as cA:
            corri(NUOVO, "G3_kl4", UNO + ["--kl4-peso", "1"])
        with CatturaGrad() as cB:
            oB = corri(NUOVO, "G3_prior3", UNO + PRIOR + ["--prior-lettere-pacchi", "3"])
        e3 = grad_rel(cA, cB)
        assert e3 < 1e-5, e3
        assert "3 pacchi KL4 per passo (offset [0, 4, 8]" in (oB / "registro.log").read_text()
    with Pacchi(1):
        with CatturaGrad() as c0:
            corri(NUOVO, "G3_paper", UNO + ["--kl4-peso", "0"])
        with CatturaGrad() as c1:
            corri(NUOVO, "G3_kl4b", UNO + ["--kl4-peso", "1"])
        with CatturaGrad() as c2:
            corri(NUOVO, "G3_entrambi", UNO + ["--kl4-peso", "1", "--prior-lettere-peso", "1"])
    g0, g1_, g2_ = grad_vett(c0), grad_vett(c1), grad_vett(c2)
    e2 = rel(g2_ - g0, 2 * (g1_ - g0))
    assert e2 < 1e-4, e2
    return f"(N = 3 identici vs KL4: {e3:.1e}; KL4+prior vs 2 KL4: {e2:.1e})"


@prova("G4 nel trainer, pacchi da piu' domande, N = 2: il gradiente del termine e' lineare in W (W = 1, 2, 3), separato e --fuso")
def g4():
    msg = []
    for nome, ex in (("sep", []), ("fuso", ["--fuso"])):
        cs = []
        for w in ("1", "2", "3"):
            with CatturaGrad() as c:
                corri(NUOVO, f"G4_{nome}_{w}", UNO + ex + ["--kl4-peso", "0", "--prior-lettere-pacchi", "2",
                                                           "--prior-lettere-peso", w])
            cs.append(grad_vett(c))
        d1, d2 = cs[1] - cs[0], cs[2] - cs[0]           # 1 and 2 units of the prior gradient
        e = rel(d2, 2 * d1)
        assert e < 1e-4 and float(d1.norm()) > 0, (nome, e)
        msg.append(f"{nome} {e:.1e} (|g_prior(W=1)|/|g(W=1)| {float(d1.norm() / cs[0].norm()):.2e})")
    return "(" + "; ".join(msg) + ")"


# ───────────────────────────────────────── T: toy models ──────────────────────────────────────────────────────────────
@prova("T1 giocattolo a logit liberi, pacchi fissi: il prior porta pbar_S su pbar_T e lascia libere le domande (KL4 per domanda invariata)")
def t1():
    g = torch.Generator().manual_seed(11)
    Q = 64
    t = torch.softmax(1.5 * torch.randn(Q, 4, generator=g, dtype=torch.float64), -1)
    off = torch.tensor([-1.0, 0.0, 0.5, 0.5], dtype=torch.float64)
    z0 = torch.log_softmax(1.5 * torch.randn(Q, 4, generator=g, dtype=torch.float64), -1) + off
    marg = lambda z: float(kl_(t.mean(0), torch.softmax(z, -1).mean(0)))
    kl4 = lambda z: float(kl_(t, torch.softmax(z, -1)).mean())
    pacchi = [torch.arange(i, i + 16) for i in range(0, Q, 16)]
    out = {}
    for nome in ("prior", "kl4"):
        z = z0.clone().requires_grad_(True)
        opt = torch.optim.SGD([z], lr=30.0)
        pr = PLm.PriorLettere(1.0)
        for s in range(1600):
            b = pacchi[s % 4]
            opt.zero_grad()
            l = pr.termine([torch.log_softmax(z[b], -1)], [t[b]])[0] if nome == "prior" \
                else kl_(t[b], torch.softmax(z[b], -1)).mean()
            l.backward(); opt.step()
        out[nome] = (marg(z.detach()), kl4(z.detach()))
    m0, k0 = marg(z0), kl4(z0)
    assert out["prior"][0] < 1e-6 * max(1.0, m0) + 1e-8, out
    assert abs(out["prior"][1] - k0) < 0.05 * k0, (out, k0)          # per-question distributions not fitted
    assert out["kl4"][1] < 0.02 * k0, out                              # the per-question term fits them
    return (f"(marginale {m0:.4f} -> prior {out['prior'][0]:.1e}; KL4 per domanda {k0:.3f} -> prior {out['prior'][1]:.3f}"
            f", fit KL4 {out['kl4'][1]:.1e})")


@prova("T2 invarianza per permutazione: scambiare le distribuzioni fra domande non cambia valore e gradiente (permutato)")
def t2():
    g = torch.Generator().manual_seed(4)
    z = torch.randn(12, 4, generator=g, dtype=torch.float64)
    t = torch.softmax(torch.randn(12, 4, generator=g, dtype=torch.float64), -1)
    perm = torch.randperm(12, generator=g)
    pr = PLm.PriorLettere(1.0, 2, 0.0)
    za, zb = z.clone().requires_grad_(True), z[perm].clone().requires_grad_(True)
    ka = pr.termine([torch.log_softmax(za[:5], -1), torch.log_softmax(za[5:], -1)], [t[:5], t[5:]])[0]
    kb = pr.termine([torch.log_softmax(zb[:7], -1), torch.log_softmax(zb[7:], -1)], [t[:7], t[7:]])[0]
    assert abs(float(ka) - float(kb)) < 1e-14, (float(ka), float(kb))
    ga, gb = torch.autograd.grad(ka, za)[0], torch.autograd.grad(kb, zb)[0]
    e = rel(gb, ga[perm])
    assert e < 1e-12, e
    return f"(valore uguale a 1e-14, gradiente permutato {e:.1e})"


@prova("T3 giocattolo con ancora (KD su altri ingressi, come paper/KL piena): prior fissa il marginale held-out senza fit per domanda")
def t3():
    g = torch.Generator().manual_seed(21)
    d, Qtr, Qho, Na = 96, 64, 512, 48
    X = torch.randn(Qtr + Qho, d, generator=g, dtype=torch.float64)
    Xa = torch.randn(Na, d, generator=g, dtype=torch.float64) * 2.0   # under-determined anchor touching every coordinate
    Wt = torch.randn(d, 4, generator=g, dtype=torch.float64) * 1.5 / d ** 0.5
    bt = torch.tensor([0.3, 0.0, 0.2, 0.1], dtype=torch.float64)
    T = torch.softmax(X @ Wt + bt, -1); Ta = torch.softmax(Xa @ Wt, -1)
    W0 = Wt + 0.6 * torch.randn(d, 4, generator=g, dtype=torch.float64) / d ** 0.5
    b0 = bt + torch.tensor([-0.8, 0.1, 0.4, 0.3], dtype=torch.float64)   # the letter-prior shift (bias of the readout)
    tr, ho = torch.arange(Qtr), torch.arange(Qtr, Qtr + Qho)
    pacchi = [tr[i:i + 16] for i in range(0, Qtr, 16)]

    def met(W, b):
        P = torch.softmax(X @ W + b, -1)
        return {"tr_kl4": float(kl_(T[tr], P[tr]).mean()), "ho_kl4": float(kl_(T[ho], P[ho]).mean()),
                "ho_marg": float(kl_(T[ho].mean(0), P[ho].mean(0)))}
    ris = {"partenza": met(W0, b0)}
    for nome, w, beta, kl4w in (("ancora", 0, 0, 0), ("+KL4", 0, 0, 1), ("+prior W1", 1, 0, 0), ("+prior W0.2", .2, 0, 0),
                                ("+prior W0.05", .05, 0, 0), ("+prior W1 EMA.95", 1, .95, 0)):
        W = W0.clone().requires_grad_(True); b = b0.clone().requires_grad_(True)
        opt = torch.optim.Adam([W, b], lr=0.01)
        pr = PLm.PriorLettere(w, 1, beta)
        for s in range(3000):
            p = pacchi[s % 4]; opt.zero_grad()
            l = kl_(Ta, torch.softmax(Xa @ W, -1)).mean()
            lS = torch.log_softmax(X[p] @ W + b, -1)
            if kl4w:
                l = l + kl_(T[p], lS.exp()).mean()
            if w:
                l = l + w * pr.termine([lS], [T[p]])[0]
            l.backward(); opt.step(); pr.conferma()
        ris[nome] = met(W.detach(), b.detach())
    an = ris["ancora"]
    for k in ("+prior W1", "+prior W0.2", "+prior W0.05", "+prior W1 EMA.95"):
        assert ris[k]["ho_marg"] < 0.5 * an["ho_marg"], (k, ris[k], an)
        assert ris[k]["tr_kl4"] < 1.1 * an["tr_kl4"], (k, ris[k], an)
    (TMP / "T3.json").write_text(json.dumps(ris, indent=1))
    return "(" + "; ".join(f"{k}: tr {v['tr_kl4']:.3f} ho {v['ho_kl4']:.3f} marg_ho {v['ho_marg']:.4f}" for k, v in ris.items()) + ")"


# ───────────────────────────────────────── N: trainer runs with the term on ───────────────────────────────────────────
@prova("N1 corse col termine acceso: separato, --fuso, N pacchi, alternato, KL piena, EMA, leve L2/L3; chiavi e finitezza")
def n1():
    casi = (("sep", PRIOR, None), ("sep_kl4", ["--kl4-peso", "0.25", "--prior-lettere-peso", "1"], None),
            ("fuso", ["--fuso"] + PRIOR, None), ("fuso_n2", ["--fuso", "--prior-lettere-pacchi", "2"] + PRIOR, None),
            ("fuso_grafo", ["--fuso"] + PRIOR, {"E2E_SENZA_RICALCOLO": "1", "E2E_GRAFO_STRATI": "2"}),
            ("sep_n3", ["--prior-lettere-pacchi", "3"] + PRIOR, None),
            ("alt", ["--modo", "alternato", "--passi", "4"] + PRIOR, None),
            ("klp", ["--kl-peso", "0.5"] + PRIOR, None),
            ("ema", ["--prior-lettere-ema", "0.9"] + PRIOR, None),
            ("ema_fuso", ["--fuso", "--prior-lettere-ema", "0.9", "--prior-lettere-pacchi", "2"] + PRIOR, None),
            ("leve", LEVE_L2 + ["--scale-gateup-lr", "1e-3", "--rango-esperti", "0"] + PRIOR, None))
    msg = []
    for nome, ex, env in casi:
        o = corri(NUOVO, f"N1_{nome}", ex, env)
        s = storia(o)
        assert "prior" in s[0] and len(s[0]["pS"]) == 4, s[0]
        for r in s[1:]:
            assert math.isfinite(r["prior_lettere"]) and math.isfinite(r["L"]) and "held_prior" in r, (nome, r)
            assert ("kl4" in r) == (nome == "sep_kl4"), (nome, r)
            assert ("prior_batch" in r) == nome.startswith("ema"), (nome, r)
        ott = torch.load(o / "ottimizzatori_ultimo.pt", map_location="cpu", weights_only=False)
        assert ("prior_lettere" in ott) == nome.startswith("ema"), nome
        if nome.startswith("ema"):
            assert ott["prior_lettere"]["n"] == len(s) - 1 and ott["prior_lettere"]["beta"] == 0.9
        msg.append(f"{nome} {s[-1]['prior_lettere']:.2e}")
    return "(" + ", ".join(msg) + ")"


@prova("N2 --riprendi con EMA == corsa ininterrotta, al bit (pt, Adam, stato EMA, tappe, storia)")
def n2():
    ex = PRIOR + ["--prior-lettere-ema", "0.9", "--prior-lettere-pacchi", "2"] + LEVE_L2
    a_ = corri(NUOVO, "N2_dritta", ex + ["--tappe-dir", str(TMP / "N2_dritta" / "tappe"), "--passi", "4"])
    t = str(TMP / "N2_ripresa" / "tappe")
    corri(NUOVO, "N2_ripresa", ex + ["--tappe-dir", t, "--passi", "2"])
    b_ = corri(NUOVO, "N2_ripresa", ex + ["--tappe-dir", t, "--passi", "4", "--riprendi"])
    d = confronta(a_, b_, tappe=True)
    assert not d, f"{len(d)} differences: {d[:5]}"
    assert "ripresa: prior di lettera EMA dopo 2 passi" in (b_ / "registro.log").read_text()
    return f"({len(tensori(a_)) + len(tensori(a_, 'tappe'))} file .pt e {len(storia(a_))} righe al bit)"


@prova("N3 errori di configurazione -> codice 5")
def n3():
    for nome, ex in (("neg", ["--prior-lettere-peso", "-1"]), ("n0", ["--prior-lettere-peso", "1", "--prior-lettere-pacchi", "0"]),
                     ("beta1", ["--prior-lettere-peso", "1", "--prior-lettere-ema", "1"]),
                     ("troppi", ["--prior-lettere-peso", "1", "--prior-lettere-pacchi", "99"]),
                     ("allinea", ["--prior-lettere-peso", "1", "--allinea-peso", "1", "--allinea-cache", str(TMP)])):
        corri(NUOVO, f"N3_{nome}", ex, rc_atteso=5)


@prova("N4 NaN nella norma di QUALSIASI blocco (primo, secondo, ultimo): nuovo salta il passo; l'originale fa il passo col NaN del secondo e va in NaN")
def n4():
    with CatturaGrad(nan_alla_chiamata=3):
        o1 = corri(ORIG, "N4_orig", [])
    with CatturaGrad(nan_alla_chiamata=3):
        o2 = corri(NUOVO, "N4_nuovo", [])
    s1, s2 = storia(o1), storia(o2)
    assert "saltato" not in s1[2] and math.isnan(s1[2]["norma_densi"]), s1[2]
    assert any(isinstance(r.get("L"), float) and math.isnan(r["L"]) for r in s1[3:]), "original: no NaN after the poisoned step"
    assert s2[2].get("saltato") == "NaN nel gradiente", s2[2]
    assert math.isfinite(s2[3]["L"]) and "saltato" not in s2[3], s2[3]
    r2 = torch.load(o2 / "residui_ultimo.pt", map_location="cpu", weights_only=False)
    fin = all(bool(torch.isfinite(v).all()) for v in _tensori_in(r2))
    assert fin, "new: non-finite residual after the skipped step"
    # any block: NaN in the FIRST block norm (call 2 = esperti of step 2) and in the last block of a 4-block step
    with CatturaGrad(nan_alla_chiamata=2):
        o5 = corri(NUOVO, "N4_primo", [])
    assert storia(o5)[2].get("saltato") == "NaN nel gradiente", storia(o5)[2]
    with CatturaGrad(nan_alla_chiamata=7):               # LEVE_L2: blocks esperti, densi, scale, down -> call 7 = down
        o6 = corri(NUOVO, "N4_ultimo", LEVE_L2)
    s6 = storia(o6)
    assert s6[2].get("saltato") == "NaN nel gradiente" and math.isnan(s6[2]["norma_down"]), s6[2]
    assert math.isfinite(s6[3]["L"]) and "saltato" not in s6[3], s6[3]
    with Pacchi(40):
        with CatturaGrad(nan_alla_chiamata=3):
            o3 = corri(NUOVO, "N4_ema", PRIOR + ["--prior-lettere-ema", "0.9"])
    ott = torch.load(o3 / "ottimizzatori_ultimo.pt", map_location="cpu", weights_only=False)
    assert ott["prior_lettere"]["n"] == 2, ott["prior_lettere"]          # 3 steps, one skipped
    return "(originale: passo 2 fatto con NaN, poi NaN; nuovo: passo 2 saltato, EMA non aggiornata)"


def _tensori_in(x):
    if isinstance(x, torch.Tensor):
        return [x] if x.is_floating_point() else []
    if isinstance(x, dict):
        return sum((_tensori_in(v) for v in x.values()), [])
    if isinstance(x, (list, tuple)):
        return sum((_tensori_in(v) for v in x), [])
    return []


if __name__ == "__main__":
    print(f"cartella {TMP}", flush=True)
    tutte = {f.__name__: f for f in (f0, g1, t1, t2, t3, f1, f2, f3, f4, f4b, f5, f6, g2, g3, g4, n1, n2, n3, n4)}
    for nome in (sys.argv[1:] or list(tutte)):
        tutte[nome]()
        _trim()
    ok = sum(1 for _, b, _ in RISULTATI if b)
    print(f"\n{ok}/{len(RISULTATI)} PASS (cartella {TMP}, RSS max "
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20:.2f} GiB)")
    sys.exit(0 if ok == len(RISULTATI) else 1)

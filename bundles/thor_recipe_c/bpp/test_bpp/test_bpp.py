#!/usr/bin/env python3
"""CPU tests of arm B++ (SINTESI_1025 BPP, 26/10/2026): --residui-shexp, --allena-f32 with every other block (incl.
--fuso / --allinea-peso), --riprendi from a B+ checkpoint without the new blocks, --router-cammino, --forza-scelte-maestro, in
cardan_congiunto_scale.py of BPP/codice (+ f32_blocco.py, router_distilla.py). No GPU, no live service, no deletion.
Harness = test_f32/test_f32.py of F32/codice (copied, not imported: it writes into the F32 tree), plus the synthetic
rotations / teacher-state cache of test_l2l5 and a synthetic teacher ROUTER-logit cache (cattura_router_maestro.py layout).
StudenteSeg = StudenteF32 whose token mixer is causal PER SEGMENT (as the real Studente in segment mode): with it the
fused [paper row | KL4 pack] step and the separate step compute the same loss, so their gradients must agree.

  B*  flags off == original (test_bpp/originale_F32_cardan_congiunto_scale.py = F32/codice trainer): every saved tensor
      and every logged value bitwise equal (separate, --fuso, alternating, the B+ recipe with all its levers, --riprendi
      with tappe, the F32-E2E --solo-f32 run)
  S*  --residui-shexp (+ --allena-f32 tutte, the B++ recipe): gradients reach every block, every F32 tensor and every
      shexp residual; trainer gradients == full-graph autograd; resume from a B+ tappa (no shexp, no F32) starts the
      new blocks at their init (B = 0 / GGUF values) with a fresh Adam, old blocks loaded; resume of B++ == straight run;
      fused == separate for every block incl. F32 (with and without the alignment); configuration errors
  R*  --router-cammino (R-CAMMINO, R7 L1) + --forza-scelte-maestro (prova 0): loss functions vs an independent per-token
      reference; W 0 == off; trainer gradient(W) - gradient(0) == full-graph autograd of the term on the
      router masters (F32 ffn_gate_inp) AND upstream (attn_norm, dense/shexp residuals, experts); separate == fused;
      configuration errors
Run (one job, 4 threads, nice 19, RSS < 2 GB, only with MemAvailable >= 12 GB):
    HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 nice -n 19 \\
      @@ROOT:venv@@/bin/python3 test_bpp/test_bpp.py [test ...]
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
ORIG_SHA = "a972bb3d36ee456f19baa78f5390453cfa93d97c913357bfee9e38e7d69975f8"     # F32/codice/cardan_congiunto_scale.py
REF_LIVE = Path("@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/SINTESI_1025/F32/codice/cardan_congiunto_scale.py")
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
TMP = Path("@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/SINTESI_1025/BPP/collaudo") / time.strftime("test_%Y%m%d_%H%M%S")
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
def leggi_mclp(log_path):                            # = test_l2l5 (rotated teacher logs of scrivi_rotazioni)
    out = {}
    for r in open(log_path, errors="replace"):
        i = r.find("MCLP\t")
        if i < 0:
            continue
        c = r[i:].rstrip("\n").split("\t")
        n = int(c[4])
        out[int(c[1])] = (int(c[2]), [float(x) for x in c[5:5 + n]])
    return out


GD.pacchi_kl4, GD.leggi_mclp = pacchi_kl4, leggi_mclp
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


ORIG = _Pigro("ccs_originale_f32", QUI / "originale_F32_cardan_congiunto_scale.py")
NUOVO = _Pigro("ccs_nuovo_bpp", COD / "cardan_congiunto_scale.py")
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



# ═══════════════════════════════════════ F32-E2E (26/10): synthetic student WITH trained F32 tensors ═══════════════════
import f32_blocco as FB                             # noqa: E402
import cardan_paper_perdita as PPm                  # noqa: E402



class Porta(torch.nn.Module):
    """Holder of the plain attributes the synthetic layer reads (written by TensoreF32.installa)."""


def _byte(t: torch.Tensor) -> bytes:
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    return t.numpy().tobytes()


def impronta(st) -> str:
    """sha256 over every NON-F32 tensor of the synthetic student: layer parameters and buffers (the F32-installed router /
    shared gate are plain attributes after the install, as in the real Studente), expert bytes and d0, theta, embedding,
    head, and the final norm when it is not trained."""
    h = hashlib.sha256()
    for L in st.strati:
        m = st.moduli[L]
        for n, p in list(m.named_parameters()) + list(m.named_buffers()):
            h.update(f"{L}.{n}".encode()); h.update(_byte(p))
        es = st.esperti(L)
        for n, f in es.fam.items():
            h.update(f"{L}.fam.{n}".encode()); h.update(_byte(f.blocchi)); h.update(_byte(f.d0))
        for n, t in es.theta.items():
            h.update(f"{L}.theta.{n}".encode()); h.update(_byte(t))
    h.update(_byte(st._emb)); h.update(_byte(st._testa))
    if "output_norm.weight" not in getattr(st, "f32", {}):
        h.update(_byte(st.norma_finale))
    return h.hexdigest()


class StudenteF32(StudenteTernario):
    """StudenteTernario + the 11 F32 families of C as REAL e2e_studente.TensoreF32 masters (voce None: the master IS the
    value the layer reads). Without allena_f32: StudenteTernario at the bit (same _strato, nothing replaced)."""
    ZERO_IN = None          # family: one exact zero in its first tensor (configuration error of a log class)
    THETA_SPORCO = False    # one theta != 0 after construction (frozen-check guard)
    FUORI = None            # family whose installed value the layer does NOT read (presence gate)

    def __init__(self, *a, allena_f32=(), **k):
        super().__init__(*a, **k)
        self.gguf = "finto.gguf"
        self.allena_f32 = S.famiglie_f32(allena_f32) if allena_f32 else ()
        self.f32, self._f32_strato = {}, {}
        self.f32_conteggi = {f: 0 for f in self.allena_f32}
        if self.allena_f32:
            self._prepara_f32_finto()
        if StudenteF32.THETA_SPORCO:
            with torch.no_grad():
                self.esperti(self.strati[0]).theta["gate"].view(-1)[0] = 1e-3
        StudenteF32.ultimo = self
        self.impronta0 = impronta(self)
        self.f32_0 = {n: t.master.detach().clone() for n, t in self.f32.items()}

    def _prepara_f32_finto(self):
        g = np.random.default_rng(4321)
        dt_m = torch.float64 if self.ric.dtype_residuo == torch.float64 else torch.float32
        dt = self.ric.dtype_gemm
        for L in self.strati:
            m = self.moduli[L]
            m.porta = Porta()
            for fam in self.allena_f32:
                if fam == "output_norm":
                    continue
                if fam == "ffn_gate_inp":
                    v = g.normal(0, 1 / np.sqrt(H), size=(E_N, H)); mod, attr = m.mlp, "router_w"
                elif fam == "ffn_gate_inp_shexp":
                    v = g.normal(0, 1 / np.sqrt(H), size=(1, H)); mod, attr = m.mlp.shared_expert_gate, "weight"
                elif fam == "ssm_a":
                    v = -np.exp(g.normal(0, 0.3, size=H)); mod, attr = m.porta, fam
                elif fam in ("ssm_dt", "ssm_conv1d"):
                    v = g.normal(0, 0.5, size=H); mod, attr = m.porta, fam
                else:                                    # norms: positive around 1; attn_k_norm with 2 negatives (as C)
                    v = 1 + 0.1 * g.normal(size=H)
                    if fam == "attn_k_norm":
                        v[:2] = -v[:2]
                    mod, attr = m.porta, fam
                if StudenteF32.ZERO_IN == fam and L == self.strati[0]:
                    v = v.copy(); v.reshape(-1)[3] = 0.0
                vecchio = getattr(mod, attr, None)
                dt_mod = vecchio.dtype if vecchio is not None else dt
                if attr in mod._parameters:
                    del mod._parameters[attr]             # as e2e_studente._prepara_f32: a plain attribute from now on
                nome = f"blk.{L}.{S.FAMIGLIE_F32[fam]}"
                tf = S.TensoreF32(nome, fam, L, None, np.asarray(v, dtype=np.float32), self.device, dt_m, mod, attr, dt_mod)
                tf.installa()
                self.f32[nome] = tf
                self._f32_strato.setdefault(L, []).append(tf)
                self.f32_conteggi[fam] += 1
        if "output_norm" in self.allena_f32:
            tf = S.TensoreF32("output_norm.weight", "output_norm", -1, None,
                              np.asarray(1 + 0.1 * g.normal(size=H), dtype=np.float32), self.device, dt_m)
            self.norma_finale = tf.master
            self.f32["output_norm.weight"] = tf
            self.f32_conteggi["output_norm"] += 1

    def _strato(self, L, x, ctx, instr=None):
        if not self.f32:
            return super()._strato(L, x, ctx, instr)
        for tf in self._f32_strato.get(L, ()):
            tf.installa()
        m = self.moduli[L]
        p = m.porta

        def w(f):
            return getattr(p, f) if (hasattr(p, f) and f != StudenteF32.FUORI) else None

        B, T, _ = x.shape
        h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        if w("attn_norm") is not None:
            h = h * w("attn_norm")
        if w("ssm_dt") is not None:
            h = h + 0.01 * w("ssm_dt")
        at = m.self_attn
        q, k, v = at.q_proj(h), at.k_proj(h), at.v_proj(h)
        if w("attn_q_norm") is not None:
            q = q * w("attn_q_norm")
        if w("attn_k_norm") is not None:
            k = k * w("attn_k_norm")
        if w("ssm_conv1d") is not None:
            v = v * (1 + 0.1 * w("ssm_conv1d"))
        cm = torch.cumsum(v, 1) / torch.arange(1, T + 1, dtype=x.dtype).view(1, T, 1)
        if w("ssm_norm") is not None:
            cm = cm * w("ssm_norm")
        o = at.o_proj(torch.tanh(q) * torch.sigmoid(k) + cm)
        if w("ssm_a") is not None:
            o = o * torch.sqrt(-w("ssm_a"))
        x = x + 0.5 * o
        h2 = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        if w("post_attention_norm") is not None:
            h2 = h2 * w("post_attention_norm")
        m.mlp.instr = instr
        try:
            out_ = m.mlp(h2)
        finally:
            m.mlp.instr = None
        return x + out_


def corri(mod, nome: str, extra: list, env: dict = None, rc_atteso: int = 0, ricetta: str = "B32", base=None) -> Path:  # noqa: F811
    """= the harness corri, with the F32-capable synthetic student (identical to StudenteTernario without --allena-f32)."""
    out = TMP / nome
    vecchio = {k: os.environ.get(k) for k in (env or {})}
    os.environ.update(env or {})
    S.Studente = StudenteF32
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


def coppia(nome: str, extra: list, env=None, extra_nuovo=None, tappe=False):
    o1 = corri(ORIG, f"{nome}_orig", extra, env)
    o2 = corri(NUOVO, f"{nome}_nuovo", extra + (extra_nuovo or []), env)
    d = confronta(o1, o2, tappe)
    assert not d, f"{len(d)} differences: {d[:5]}"
    return o1, o2


# the F32-E2E configuration of the launcher (lr scaled up: few synthetic steps must move every tensor)
SOLO = ["--rango-esperti", "0", "--rango-densi", "0", "--allena-f32", "tutte", "--solo-f32", "--kl4-peso", "0",
        "--kl-peso", "1", "--selezione", "paper", "--prior-lettere-guardia", "--lr-f32-norme", "1e-2",
        "--lr-f32-router", "3e-3", "--lr-f32-ssm-a", "1e-2", "--lr-f32-ssm", "3e-3"]
BLOCCHI_F32 = ["f32_norme", "f32_router", "f32_ssm"]



# ═══════════════════════════════════════ B++ (26/10): helpers ═══════════════════════════════════════════════════════
import allinea_stati as AS                          # noqa: E402
import residui_ternari as RTm                       # noqa: E402
import router_distilla as RDm                       # noqa: E402
import cardan_kd_moduli as CKm                      # noqa: E402


def ids_riga(nome: str, lato: str, i: int, T: int = T_RIGA) -> torch.Tensor:
    """= CaricatoreFinto ids (same generator)."""
    g = torch.Generator().manual_seed(_seed(nome, lato, i))
    return torch.randint(0, 38, (1, T), generator=g)


def scrivi_rotazioni(d: Path, N: int) -> None:
    """= test_l2l5: rotated calibration + teacher logs consistent with domande()."""
    d.mkdir(parents=True, exist_ok=True)
    for k in range(1, N):
        rr, mm = [], []
        for q in domande():
            yk = (q["label"] - k) % 4
            rr.append(json.dumps({"i": q["i"], "label": yk, "ids": q["ids"] + [k], "lettere": LETTERE, "rotazione": k}))
            lp = torch.log(q["pT"])[[(j + k) % 4 for j in range(4)]] + torch.tensor([0.3, 0.0, -0.1, 0.05])
            mm.append("MCLP\t%d\t%d\t0\t4\t%s\t1\t1\t1\t1" % (q["i"], yk, "\t".join(f"{x:.8f}" for x in lp.tolist())))
        (d / f"calib_mmlu_rot{k}.jsonl").write_text("\n".join(rr) + "\n")
        (d / f"mclp_maestro_rot{k}.log").write_text("\n".join(mm) + "\n")


def _scrivi_cache(d: Path, teacher: Path, righe: dict, n_tok: int, tensori: dict, piano_nome: str, extra: dict) -> None:
    """cattura_*_maestro.py layout: <lato>/blocco_0000/{ids.i32, blocco.json, dump/manifesto.json, dump/seq_KKKK/*}.
    tensori: {name: (fn(lato, i) -> np array [n_tok, W], ext, np dtype)}."""
    nome_t = teacher.name
    bl = []
    for lato, rr in righe.items():
        nb = f"{lato}/blocco_0000"
        b = d / nb
        (b / "dump").mkdir(parents=True, exist_ok=True)
        ids = np.stack([ids_riga(nome_t, lato, i).numpy()[0, :n_tok].astype(np.int32) for i in rr])
        ids.tofile(b / "ids.i32")
        seqs = []
        for k, i in enumerate(rr):
            sd = b / "dump" / f"seq_{k:04d}"
            sd.mkdir(exist_ok=True)
            ten = {}
            for nome, (fn, ext, dt) in tensori.items():
                a = np.ascontiguousarray(fn(lato, i)).astype(dt)
                f = sd / f"{nome}.{ext}"
                with open(f, "wb") as fh:
                    np.array([a.shape[1], a.shape[0], 1, 1], dtype=np.int64).tofile(fh); a.tofile(fh)
                ten[nome] = {"file": f"seq_{k:04d}/{nome}.{ext}", "righe_uguali_a": "token"}
            seqs.append({"cartella": f"seq_{k:04d}", "tensori": ten})
        (b / "dump" / "manifesto.json").write_text(json.dumps({"esito": "ok", "sequenze": seqs,
                                                                "modello": {"n_vocab": V}}))
        (b / "blocco.json").write_text(json.dumps({"righe": rr, "lato": lato,
                                                   "sha256_ids": hashlib.sha256(ids.tobytes()).hexdigest()}))
        bl.append({"nome": nb, "lato": lato, "righe": rr})
    (d / piano_nome).write_text(json.dumps({
        "n_token": n_tok, "sorgente": str(teacher),
        "sorgente_piano_sha256": hashlib.sha256((teacher / "piano.json").read_bytes()).hexdigest(), "blocchi": bl,
        **extra}))


def cache_stati(strati=(0, 1, 2), n_tok=32) -> Path:
    d = TMP / "cache_stati"
    if not d.exists():
        def stato_fn(L):
            def f(lato, i):
                x = torch.randn(n_tok, H, generator=torch.Generator().manual_seed(_seed(5, lato, i, L))) * 3
                return x.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
            return f
        _scrivi_cache(d, TEACH_W, {"train": AS.righe_in_ordine(list(range(10)), 20261007), "heldout": [0, 1]}, n_tok,
                      {f"l_out-{L}": (stato_fn(L), "bf16", np.uint16) for L in strati}, "piano_stati.json",
                      {"strati": list(strati), "n_embd": H})
    return d


def cache_router(nome="cache_router", righe_train=None, strati=(0, 1, 2), n_tok=32) -> Path:
    d = TMP / nome
    if not d.exists():
        def logit_fn(L):
            def f(lato, i):
                return (torch.randn(n_tok, E_N, generator=torch.Generator().manual_seed(_seed(11, lato, i, L))) * 2).numpy()
            return f
        rt = AS.righe_in_ordine(list(range(10)), 20261007) if righe_train is None else righe_train
        _scrivi_cache(d, TEACH_W, {"train": rt, "heldout": [0, 1]}, n_tok,
                      {RDm.NOME_TENSORE.format(L): (logit_fn(L), "f16", np.float16) for L in strati}, "piano_router.json",
                      {"strati": list(strati), "n_expert": E_N, "n_embd": H, "salva_tipo": "f16"})
    return d


def rot_dir() -> Path:
    d = TMP / "rot"
    if not d.exists():
        scrivi_rotazioni(d, 4)
    return d


def ricetta_bplus() -> list:
    """The B+ recipe on the synthetic student: L2 levers + gate/up scales + 4 KL4 rotations + CKA alignment 0.3 + fused."""
    return LEVE_L2 + ["--scale-gateup-lr", "1e-3", "--kl4-rotazioni", "4", "--kl4-rotazioni-modo", "ciclo",
                      "--kl4-rotazioni-dir", str(rot_dir()), "--allinea-peso", "0.3", "--allinea-modo", "rapporto",
                      "--allinea-metrica", "cka", "--allinea-strati", "0-2", "--allinea-cache", str(cache_stati()),
                      "--fuso"]


F32_ON = ["--allena-f32", "tutte", "--lr-f32-norme", "1e-2", "--lr-f32-router", "3e-3", "--lr-f32-ssm-a", "1e-2",
          "--lr-f32-ssm", "3e-3"]
BPP_NUOVI = F32_ON + ["--residui-shexp"]
ORD_BPP = ["esperti", "densi", "scale", "scale_gu", "down", "densi_shexp"] + BLOCCHI_F32     # attivi() order, B++
N_F32 = 3 * 10 + 1
FAM_SH = ("ffn_down_shexp", "ffn_gate_shexp", "ffn_up_shexp")                                  # sorted(resSh) order
NOMI_SH = sorted(f"blk.{L}.{f}" for L in range(N_STRATI) for f in FAM_SH)


def _media_seg(v: torch.Tensor, ctx: dict) -> torch.Tensor:
    """causal running mean of v [1, T, H], restarted at every segment start (riga mode: the whole row)."""
    if ctx.get("modo") != "seg":
        T = v.shape[1]
        return torch.cumsum(v, 1) / torch.arange(1, T + 1, dtype=v.dtype).view(1, T, 1)
    pezzi = []
    for s0, s1 in ctx["segmenti"]:
        n = s1 - s0
        pezzi.append(torch.cumsum(v[:, s0:s1], 1) / torch.arange(1, n + 1, dtype=v.dtype).view(1, n, 1))
    return torch.cat(pezzi, 1)


class StudenteSeg(StudenteF32):
    """StudenteF32 with a token mixer causal PER SEGMENT (like the real Studente in segment mode): fused [row | pack]
    == separate row + pack, up to the fp order of the MoE GEMMs. F32 optional (porta absent without --allena-f32)."""

    def _contesto(self, B, T, dtype, inizi=None):
        if inizi is None:
            return {"modo": "riga"}
        ini = torch.as_tensor(inizi).cpu().reshape(B, T)
        assert B == 1 and int(ini[0, 0]) == 1, ini
        cu = torch.nonzero(ini[0]).reshape(-1).tolist() + [T]
        return {"modo": "seg", "segmenti": list(zip(cu[:-1], cu[1:]))}

    def _strato(self, L, x, ctx, instr=None):
        for tf in self._f32_strato.get(L, ()):
            tf.installa()
        m = self.moduli[L]
        p = getattr(m, "porta", None)

        def w(f):
            return getattr(p, f) if (p is not None and hasattr(p, f) and f != StudenteF32.FUORI) else None

        h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        if w("attn_norm") is not None:
            h = h * w("attn_norm")
        if w("ssm_dt") is not None:
            h = h + 0.01 * w("ssm_dt")
        at = m.self_attn
        q, k, v = at.q_proj(h), at.k_proj(h), at.v_proj(h)
        if w("attn_q_norm") is not None:
            q = q * w("attn_q_norm")
        if w("attn_k_norm") is not None:
            k = k * w("attn_k_norm")
        if w("ssm_conv1d") is not None:
            v = v * (1 + 0.1 * w("ssm_conv1d"))
        cm = _media_seg(v, ctx)
        if w("ssm_norm") is not None:
            cm = cm * w("ssm_norm")
        o = at.o_proj(torch.tanh(q) * torch.sigmoid(k) + cm)
        if w("ssm_a") is not None:
            o = o * torch.sqrt(-w("ssm_a"))
        x = x + 0.5 * o
        h2 = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        if w("post_attention_norm") is not None:
            h2 = h2 * w("post_attention_norm")
        m.mlp.instr = instr
        try:
            out_ = m.mlp(h2)
        finally:
            m.mlp.instr = None
        return x + out_


def corri(mod, nome: str, extra: list, env: dict = None, rc_atteso: int = 0, ricetta: str = "B32", base=None,  # noqa: F811
          classe=StudenteF32) -> Path:
    out = TMP / nome
    vecchio = {k: os.environ.get(k) for k in (env or {})}
    os.environ.update(env or {})
    S.Studente = classe
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


def corri_seg(nome, extra, **k):
    return corri(NUOVO, nome, extra, classe=StudenteSeg, **k)


def residui_vivi(st) -> dict:
    """{blk.L.fam: ResiduoTernario} of the live student (dense and shexp)."""
    out = {}
    for L in st.strati:
        for m in st.moduli[L].modules():
            if isinstance(m, RTm.ResiduoTernario):
                out[m.nome] = m
    return out


class CatturaSnap(CatturaGrad):
    """CatturaGrad + a snapshot of the live student (residual latents, F32 masters) at the FIRST clip call (= the first
    step's parameters before any optimiser step of this run)."""

    def __init__(self):
        super().__init__()
        self.snap = None

    def __enter__(self):
        vero = self.vero

        def f(params, max_norm, *a, **k):
            ps = list(params)
            if self.snap is None:
                st = StudenteF32.ultimo
                self.snap = {"res": {n: {kk: vv.detach().clone() for kk, vv in m.state_dict().items()}
                                     for n, m in residui_vivi(st).items()},
                             "f32": {n: t.master.detach().clone() for n, t in st.f32.items()},
                             "f32_0": dict(st.f32_0)}
            self.chiamate.append([None if p.grad is None else p.grad.detach().clone() for p in ps])
            return vero(ps, max_norm, *a, **k)
        torch.nn.utils.clip_grad_norm_ = f
        return self


def norma(ch: list) -> float:
    return math.sqrt(sum(float(g.double().pow(2).sum()) for g in ch if g is not None))


def passi_adam(opt_sd: dict) -> set:
    return {int(float(s["step"])) for s in opt_sd["state"].values()}


def con_grad(g) -> bool:
    return g is not None and bool((g != 0).any())


# ═══════════════════════════════════════ B: flags off == original ═══════════════════════════════════════════════════
@prova("B0 la copia di riferimento e' il trainer di F32/codice (sha)")
def b0():
    h = hashlib.sha256((QUI / "originale_F32_cardan_congiunto_scale.py").read_bytes()).hexdigest()
    assert h == ORIG_SHA, h
    return f"(= F32/codice vivo: {hashlib.sha256(REF_LIVE.read_bytes()).hexdigest() == ORIG_SHA})"


@prova("B1 flag spenti == originale: congiunto separato")
def b1():
    o1, _ = coppia("B1", [])
    return f"({len(tensori(o1))} file .pt, {len(storia(o1))} righe di storia al bit)"


@prova("B2 flag spenti == originale: --fuso")
def b2():
    coppia("B2", ["--fuso"])


@prova("B3 flag spenti == originale: alternato")
def b3():
    coppia("B3", ["--modo", "alternato", "--passi", "4"])


@prova("B4 flag spenti == originale: ricetta B+ (L2 + scale gate/up + 4 rotazioni + CKA 0,3 + fuso), con tappe")
def b4():
    t1, t2 = str(TMP / "B4_orig" / "tappe"), str(TMP / "B4_nuovo" / "tappe")
    o1 = corri(ORIG, "B4_orig", ricetta_bplus() + ["--tappe-dir", t1])
    o2 = corri(NUOVO, "B4_nuovo", ricetta_bplus() + ["--tappe-dir", t2])
    d = confronta(o1, o2, tappe=True)
    assert not d, f"{len(d)} differences: {d[:5]}"
    s = storia(o2)
    assert all("allinea" in r for r in s[1:]) and "kl4 rotation 3" in (o2 / "registro.log").read_text()
    return f"({len(tensori(o1, 'tappe'))} file di tappa al bit)"


@prova("B5 flag spenti == originale attraverso --riprendi (ricetta B+, 2 passi poi fino a 4), con tappe")
def b5():
    for mod, n in ((ORIG, "B5_orig"), (NUOVO, "B5_nuovo")):
        t = str(TMP / n / "tappe")
        corri(mod, n, ricetta_bplus() + ["--tappe-dir", t, "--passi", "2"])
        corri(mod, n, ricetta_bplus() + ["--tappe-dir", t, "--passi", "4", "--riprendi"])
    d = confronta(TMP / "B5_orig", TMP / "B5_nuovo", tappe=True)
    assert not d, f"{len(d)} differences: {d[:5]}"


@prova("B6 corsa F32-E2E (--solo-f32) == trainer F32 originale, filata e attraverso --riprendi")
def b6():
    for mod, n in ((ORIG, "B6_orig"), (NUOVO, "B6_nuovo")):
        t = str(TMP / n / "tappe")
        corri(mod, n, SOLO + ["--passi", "2", "--heldout-ogni", "1", "--tappe-dir", t])
        corri(mod, n, SOLO + ["--passi", "4", "--heldout-ogni", "1", "--tappe-dir", t, "--riprendi"])
    d = confronta(TMP / "B6_orig", TMP / "B6_nuovo", tappe=True)
    assert not d, f"{len(d)} differences: {d[:5]}"


# ═══════════════════════════════════════ S: --residui-shexp + F32 with everything ═══════════════════════════════════
@prova("S1 ricetta B++ (B+ + F32 tutte + residui shexp, fuso+CKA+rotazioni): gradiente su ogni blocco, ogni F32, ogni residuo shexp")
def s1():
    t = TMP / "S1" / "tappe"
    with CatturaGrad() as c:
        out = corri_seg("S1", ricetta_bplus() + BPP_NUOVI + ["--passi", "3", "--tappe-dir", str(t),
                                                             "--selezione", "mmlu"])
    nb = len(ORD_BPP)
    assert len(c.chiamate) == 3 * nb, (len(c.chiamate), nb)
    p1, p2 = c.chiamate[:nb], c.chiamate[nb:2 * nb]
    zero = [b for b, ch in zip(ORD_BPP, p1) if not norma(ch) > 0]
    assert not zero, f"blocchi senza gradiente al passo 1: {zero}"
    f32_1 = [g for b, ch in zip(ORD_BPP, p1) if b in BLOCCHI_F32 for g in ch]
    assert len(f32_1) == N_F32 and all(con_grad(g) for g in f32_1), "F32 senza gradiente al passo 1"
    sh1, sh2 = p1[ORD_BPP.index("densi_shexp")], p2[ORD_BPP.index("densi_shexp")]
    assert len(sh1) == 3 * len(NOMI_SH), len(sh1)
    for k, n in enumerate(NOMI_SH):
        A1, B1, g1 = sh1[3 * k: 3 * k + 3]
        assert con_grad(B1), f"{n}: B senza gradiente al passo 1"
        assert not con_grad(A1) and not con_grad(g1), f"{n}: A/gamma con gradiente a B = 0"
        A2, B2, g2 = sh2[3 * k: 3 * k + 3]
        assert con_grad(A2) and con_grad(B2) and con_grad(g2), f"{n}: A/B/gamma senza gradiente al passo 2"
    reg = (out / "registro.log").read_text()
    for s_ in ("residui shexp: rango 4 su 9 proiezioni", "cancello di presenza (strati [0, 2]): ok", "AL BIT",
               "F32 gradienti non nulli al primo passo", "cancello di presenza F32"):
        assert s_ in reg, s_
    lat = torch.load(out / "residui_ultimo.pt", weights_only=False)["latenti"]
    assert set(NOMI_SH) <= set(lat) and len(lat) == len(NOMI_SH) + N_STRATI * 4, sorted(lat)
    for p_ in (1, 2, 3):
        assert set(NOMI_SH) <= set(torch.load(t / f"residui_{p_}.pt", weights_only=False)["latenti"])
        d = torch.load(t / f"ottimizzatori_{p_}.pt", map_location="cpu", weights_only=False)
        assert "optDsh" in d and sorted(d["optF32"]) == BLOCCHI_F32 and passi_adam(d["optDsh"]) == {p_}
        assert (t / f"f32_{p_}.pt").exists()
    ex = json.load(open(out / "esportazione.json"))
    rd = torch.load(out / "residui_delta.pt", weights_only=False)
    mig = torch.load(out / "residui_migliore.pt", weights_only=False)["latenti"]
    assert set(NOMI_SH) <= set(rd["delta_file"]) and set(FAM_SH) <= set(rd["famiglie"]), sorted(rd["delta_file"])
    for n in NOMI_SH:
        sd = mig.get(n)
        dW = torch.zeros_like(rd["delta_file"][n]) if sd is None else \
            ((sd["B"] * sd["gamma"].view(-1, 1)).double() @ sd["A"].double()).float()   # = RT.delta_file
        assert torch.equal(rd["delta_file"][n], dW), f"{n}: residui_delta != diag(gamma) B A"
    s = storia(out)
    assert all(r.get("shexp_gB_rms", 0) > 0 for r in s[1:]), "shexp_gB_rms"
    pk = torch.load(out / "f32_ultimo.pt", weights_only=False)
    st = StudenteF32.ultimo
    fermi = [n for n, v0 in st.f32_0.items() if torch.equal(pk["valori"][n].reshape(v0.shape).to(v0.dtype), v0)]
    assert not fermi, f"F32 fermi: {fermi}"
    return (f"({nb} blocchi tutti con gradiente; {N_F32}/{N_F32} F32; {len(NOMI_SH)} residui shexp: B al passo 1, A/gamma "
            f"dal passo 2; tappe con optDsh/optF32/f32; delta shexp = diag(gamma) B A; migliore passo {ex['passo']})")


def _perdita_rif(st, con_grad: bool) -> float:
    """The trainer's step-1 loss on the synthetic student, full graph: paper row ord_w[0] + full KL row ord_q[0]
    (rows in the trainer's order: random.Random(seme) shuffles train_w, then train_q)."""
    rng = random.Random(20261007)
    ow, oq = list(range(10)), list(range(10))
    rng.shuffle(ow); rng.shuffle(oq)
    WS = A.testa_studente(st, "bf16x3"); WT, _ = A.testa_maestro("cpu", torch.bfloat16)
    lot = CaricatoreFinto(SorgenteFinta(TEACH_W), "cpu").prendi("train", [ow[0]])
    lq = CaricatoreFinto(SorgenteFinta(TEACH_Q), "cpu").prendi("train", [oq[0]])
    with torch.set_grad_enabled(con_grad):
        hS = st.nascosto(lot.ids, inizi=lot.inizi)
        tgt, ok = NUOVO.bersagli(lot)
        kd, ce, T, n, g = PPm.perdita_paper(
            hS.detach(), WS, lot.hT.reshape(hS.shape[0], -1), WT, tgt, ok, 1.0, 0.5, 8, 32, con_grad,
            lambda h: A._logit_studente(h, WS), lambda h: A._logit_maestro(h, WT, torch.float32),
            lambda q: A._per_testa(q, WS))
        if con_grad:
            hS.backward(g.to(hS.dtype))
        hq = st.nascosto(lq.ids, inizi=lq.inizi)
        kl = A.KLFusa.apply(hq, WS, lq.hT, WT, 32)
        if con_grad:
            (kl / lq.ids.numel()).backward()
    return float(kd / T + 0.5 * ce / n + float(kl.detach()) / lq.ids.numel())


def _studente_rif(classe=StudenteF32, esperti=0, densi=4, shexp=True, f32=True):
    """A fresh student in the trainer's step-1 state: same installs (same seeds) as cardan_congiunto_scale.main."""
    st = classe(strati=range(N_STRATI), allena_f32=FB.famiglie("tutte") if f32 else ())
    for p in st.parametri():
        p.requires_grad_(False)
    strati = KDP.intervallo("0-2")
    resE = CKm.installa(st, strati, esperti, None, torch.device("cpu")) if esperti else {}
    resD = RTm.installa(st, RTm.ConfigResidui(rango=densi, piani=1, strati=strati, famiglie=RTm.famiglie_da("attn"),
                                              seme=20261007, continuo=True), device=torch.device("cpu")) if densi else {}
    resS = RTm.installa(st, RTm.ConfigResidui(rango=densi, piani=1, strati=strati, famiglie=RTm.famiglie_da("shexp"),
                                              seme=20261007, continuo=True), device=torch.device("cpu")) if shexp else {}
    for p in CKm.parametri(resE, "C") + CKm.parametri(resE, "D") + RTm.parametri(resD) + RTm.parametri(resS):
        p.requires_grad_(True)
    F = FB.BloccoF32(st, {"norme": 1e-2, "router": 3e-3, "ssm_a": 1e-2, "ssm": 3e-3}, FB.log_classi("norme,ssm_a")) \
        if f32 else None
    return st, resE, resD, resS, F


@prova("S2 gradienti del trainer (densi, shexp, F32) == autograd a grafo intero della stessa perdita")
def s2():
    cfg = ["--rango-esperti", "0", "--rango-densi", "4", "--famiglie", "attn", "--residui-shexp", "--kl4-peso", "0",
           "--kl-peso", "1", "--selezione", "paper", "--passi", "1", "--heldout-ogni", "1"] + F32_ON
    with CatturaGrad() as c:
        corri(NUOVO, "S2", cfg)
    st, _, resD, resS, F = _studente_rif()
    _perdita_rif(st, True)
    F.gradienti_log()
    rif = {"densi": [p.grad for p in RTm.parametri(resD)], "densi_shexp": [p.grad for p in RTm.parametri(resS)],
           **{b: [p.grad for p in F.blocchi()[b][1]] for b in F.nomi()}}
    ordine = ["densi", "densi_shexp"] + BLOCCHI_F32
    assert len(c.chiamate) == len(ordine), len(c.chiamate)
    err = {}
    for b, ch in zip(ordine, c.chiamate):
        a_ = torch.cat([(torch.zeros(1) if x is None else x).reshape(-1).double() for x in ch])
        r_ = torch.cat([(torch.zeros(1) if x is None else x).reshape(-1).double() for x in rif[b]])
        err[b] = rel(a_, r_)
    assert max(err.values()) < 1e-6, err
    return "(errore relativo per blocco " + ", ".join(f"{b} {e:.1e}" for b, e in err.items()) + ")"


def _semina_come_lanciatore(src: Path, dst: Path, passo: int) -> None:
    """= congiunto_bpp_1026.sh: the B+ tappa -> *_ultimo.pt and *_migliore.pt, ottimizzatori -> ottimizzatori_ultimo.pt."""
    import shutil
    dst.mkdir(parents=True, exist_ok=True)
    for n in ("cardan", "residui", "cardan_down", "scale", "scale_gateup"):
        shutil.copy2(src / f"{n}_{passo}.pt", dst / f"{n}_ultimo.pt")
        shutil.copy2(src / f"{n}_{passo}.pt", dst / f"{n}_migliore.pt")
    shutil.copy2(src / f"ottimizzatori_{passo}.pt", dst / "ottimizzatori_ultimo.pt")


@prova("S3 --riprendi da una tappa B+ (senza shexp ne' F32): nuovi blocchi alla partenza (B = 0, valori GGUF), Adam nuovo")
def s3():
    tb = TMP / "S3_bplus" / "tappe"
    corri_seg("S3_bplus", ricetta_bplus() + ["--passi", "2", "--tappe-dir", str(tb)])
    _semina_come_lanciatore(tb, TMP / "S3_bpp", 2)
    with CatturaSnap() as c:
        out = corri_seg("S3_bpp", ricetta_bplus() + BPP_NUOVI + ["--passi", "4", "--riprendi",
                                                                 "--tappe-dir", str(TMP / "S3_bpp" / "tappe")])
    reg = (out / "registro.log").read_text()
    for s_ in ("la corsa ripresa non aveva i residui shexp: 9 partono da B = 0 (init), Adam nuovo",
               "la corsa ripresa non allenava i F32", "RIPRESA dal passo 2"):
        assert s_ in reg, s_
    # 1) the parameters of the first resumed step: new blocks at their start, old dense residuals = the B+ tappa
    _, _, _, resS0, _ = _studente_rif(StudenteSeg, f32=False)
    vecchi = torch.load(tb / "residui_2.pt", weights_only=False)["latenti"]
    for n, sd in c.snap["res"].items():
        if n in NOMI_SH:
            assert torch.equal(sd["B"], torch.zeros_like(sd["B"])) and torch.equal(sd["gamma"], torch.ones_like(sd["gamma"]))
            assert torch.equal(sd["A"], resS0[n].A.detach()), f"{n}: A non e' quella della partenza"
        else:
            for k in ("A", "B", "gamma"):
                assert torch.equal(sd[k], vecchi[n][k]), f"{n}.{k} non e' quello della tappa B+"
    for n, v in c.snap["f32"].items():
        assert torch.equal(v, c.snap["f32_0"][n]), f"{n}: F32 non parte dai valori del GGUF"
    # 2) gradients of that step: shexp A/gamma exactly 0 (B = 0), B nonzero; every F32 nonzero
    nb = len(ORD_BPP)
    p3 = c.chiamate[:nb]
    sh = p3[ORD_BPP.index("densi_shexp")]
    for k, n in enumerate(NOMI_SH):
        assert con_grad(sh[3 * k + 1]) and not con_grad(sh[3 * k]) and not con_grad(sh[3 * k + 2]), n
    assert all(con_grad(g) for b, ch in zip(ORD_BPP, p3) if b in BLOCCHI_F32 for g in ch)
    # 3) Adam: new blocks fresh (2 steps: 3 and 4), old blocks continued (4 steps)
    d = torch.load(TMP / "S3_bpp" / "tappe" / "ottimizzatori_4.pt", map_location="cpu", weights_only=False)
    assert passi_adam(d["optDsh"]) == {2}, passi_adam(d["optDsh"])
    assert all(passi_adam(d["optF32"][b]) == {2} for b in BLOCCHI_F32)
    for k in ("optE", "optD", "optS", "optDn", "optSgu"):
        assert passi_adam(d[k]) == {4}, (k, passi_adam(d[k]))
    # 4) history of B+ kept
    s_old, s_new = storia(TMP / "S3_bplus"), storia(out)
    assert s_new[:3] == s_old[:3] and [r["passo"] for r in s_new] == [0, 1, 2, 3, 4], [r["passo"] for r in s_new]
    return "(B = 0, gamma = 1, A di partenza; densi = tappa B+; F32 = GGUF; Adam nuovi 2 passi, vecchi 4; storia B+ conservata)"


@prova("S4 B++ --riprendi (2 passi, poi fino a 4) == 4 passi filati, al bit (pt, Adam, tappe, storia)")
def s4():
    a_ = ricetta_bplus() + BPP_NUOVI
    corri_seg("S4_filato", a_ + ["--passi", "4", "--tappe-dir", str(TMP / "S4_filato" / "tappe")])
    t = str(TMP / "S4_ripreso" / "tappe")
    corri_seg("S4_ripreso", a_ + ["--passi", "2", "--tappe-dir", t])
    corri_seg("S4_ripreso", a_ + ["--passi", "4", "--tappe-dir", t, "--riprendi"])
    d = confronta(TMP / "S4_filato", TMP / "S4_ripreso", tappe=True)
    assert not d, f"{len(d)} differences: {d[:5]}"
    return f"({len(tensori(TMP / 'S4_filato', 'tappe'))} file di tappa al bit)"


def _sep_fuso(nome: str, extra: list) -> tuple:
    """1 step separate vs fused, same arguments: max relative gradient error per block, and the two first rows."""
    out = {}
    for modo, ag in (("sep", []), ("fuso", ["--fuso"])):
        with CatturaGrad() as c:
            o = corri_seg(f"{nome}_{modo}", [x for x in extra if x != "--fuso"] + ag + ["--passi", "1"])
        out[modo] = (c, storia(o))
    c1, c2 = out["sep"][0], out["fuso"][0]
    assert len(c1.chiamate) == len(c2.chiamate)
    e = [rel(torch.cat([x.reshape(-1).double() for x in a_ if x is not None]),
             torch.cat([x.reshape(-1).double() for x in b_ if x is not None])) for a_, b_ in zip(c1.chiamate, c2.chiamate)]
    return e, out["sep"][1][1], out["fuso"][1][1]


@prova("S5 F32 + shexp + tutti i blocchi: gradienti fuso == separato, senza e con allineamento CKA")
def s5():
    msg = []
    for nome, al in (("S5_noal", False), ("S5_al", True)):
        base_ = [x for x in ricetta_bplus()]
        if not al:
            i = base_.index("--allinea-peso"); del base_[i:i + 10]
        e, r1, r2 = _sep_fuso(nome, base_ + BPP_NUOVI)
        assert max(e) < 1e-4, dict(zip(ORD_BPP, e))
        for k in ("paper", "kl4", "L") + (("allinea",) if al else ()):
            assert abs(r1[k] - r2[k]) <= 1e-5 * max(1.0, abs(r1[k])), (k, r1[k], r2[k])
        msg.append(f"{'CKA' if al else 'senza CKA'}: max {max(e):.1e} su {len(e)} blocchi")
    return "(" + "; ".join(msg) + ")"


@prova("S6 errori: shexp senza rango densi -> 5; shexp + solo-f32 -> 5; ripresa B++ senza --residui-shexp -> 4; "
       "optDsh senza residui shexp -> 4; optF32 senza f32_ultimo -> 4")
def s6():
    import shutil
    corri(NUOVO, "S6_rango0", ["--residui-shexp", "--rango-densi", "0"], rc_atteso=5)
    corri(NUOVO, "S6_solo", SOLO + ["--residui-shexp"], rc_atteso=5)
    a_ = BPP_NUOVI + ["--passi", "2", "--heldout-ogni", "1"]
    corri_seg("S6_c", a_)
    corri_seg("S6_c", F32_ON + ["--passi", "4", "--heldout-ogni", "1", "--riprendi"], rc_atteso=4)
    corri_seg("S6_d", a_)
    corri_seg("S6_d0", ["--passi", "2", "--heldout-ogni", "1"])
    shutil.copy2(TMP / "S6_d0" / "residui_ultimo.pt", TMP / "S6_d" / "residui_ultimo.pt")
    corri_seg("S6_d", BPP_NUOVI + ["--passi", "4", "--heldout-ogni", "1", "--riprendi"], rc_atteso=4)
    corri_seg("S6_e", a_)
    os.replace(TMP / "S6_e" / "f32_ultimo.pt", TMP / "S6_e" / "f32_ultimo.pt.messo_da_parte")
    corri_seg("S6_e", BPP_NUOVI + ["--passi", "4", "--heldout-ogni", "1", "--riprendi"], rc_atteso=4)
    for n, s_ in (("S6_c", "manca --residui-shexp"), ("S6_d", "ripresa con optDsh ma senza"),
                  ("S6_e", "ripresa con optF32 ma senza f32_ultimo.pt")):
        assert s_ in (TMP / n / "registro.log").read_text(), (n, s_)


# ═══════════════════════════════════════ R: --router-distilla ═══════════════════════════════════════════════════════
def _termine_rif(z_s, z_t, modo: str, zona=(2, 1), ranghi=(1, 3), beta=1.0, tau=0.0, k=TOPK):
    """INDEPENDENT reference of one layer's router term (per-token Python loops, R7 §4.5 formulas), with graph on z_s."""
    dt = torch.float64 if z_s.dtype == torch.float64 else torch.float32
    if modo == "kl_piena":
        lt, ls = torch.log_softmax(z_t.to(dt), -1), torch.log_softmax(z_s.to(dt), -1)
        return (lt.exp() * (lt - ls)).sum(-1).mean()
    n, E = z_t.shape
    zt, zs = z_t.to(dt), z_s.to(dt)
    if tau <= 0:
        tau = max(0.25 * float((zs.detach() - zt).pow(2).mean().sqrt()), 1e-6)
    tot = 0.0
    for t in range(n):
        pT, pS = torch.softmax(zt[t], -1), torch.softmax(zs[t], -1)
        U = set(torch.topk(zt[t], min(zona[0], E)).indices.tolist()) | set(torch.topk(zs[t].detach(), min(zona[1], E)).indices.tolist())
        U = sorted(U)
        kl = sum(pT[e] * (torch.log(pT[e]) - torch.log(pS[e])) for e in U)
        resto = [e for e in range(E) if e not in U]
        if resto:
            rT, rS = sum(pT[e] for e in resto), sum(pS[e] for e in resto)
            kl = kl + rT * (torch.log(rT) - torch.log(rS))
        ordT = torch.argsort(zt[t], descending=True).tolist()
        norm_ = sum(float(pT[e]) for e in ordT[:k])
        srt = torch.sort(zs[t].detach(), descending=True).values
        th = 0.5 * (float(srt[k - 1]) + float(srt[k]))
        m = 0.0
        for r in range(ranghi[0], min(ranghi[1], E) + 1):
            e = ordT[r - 1]
            sg = torch.sigmoid((zs[t, e] - th) / tau)
            m = m + float(pT[e]) / norm_ * ((1 - sg) if r <= k else sg)
        tot = tot + kl + beta * m
    return tot / n


def _rif_router(st, lot, i_r, W, modo, lam=None, strati=(0, 1, 2), cache=None):
    """Full-graph reference of the trainer's router term on row i_r: plain forward hooks catch the MoE inputs, the
    independent _termine_rif per layer, W * sum lambda_l term_l, backward. Returns sum lambda_l term_l."""
    src = RDm.SorgenteRouter(cache or cache_router())
    zT, _ = src.logit("train", i_r, list(strati), torch.device("cpu"))
    xs = {}
    hs = [st.moduli[L].mlp.register_forward_hook(lambda m, a, o, L=L: xs.__setitem__(L, (m, a[0]))) for L in strati]
    try:
        st.nascosto(lot.ids, inizi=lot.inizi)
    finally:
        for h in hs:
            h.remove()
    lam = lam or {L: 1.0 for L in strati}
    sl = sum(lam[L] for L in strati)
    tot = 0.0
    for k, L in enumerate(strati):
        m, x = xs[L]
        z = torch.nn.functional.linear(x.reshape(-1, H)[:32], m.router_w)
        tot = tot + lam[L] / sl * _termine_rif(z, zT[k, :32], modo)
    (W * tot).backward()
    return float(tot.detach())


@prova("R0 funzioni di perdita R-CAMMINO su tensori sintetici: KL_U, massa a rischio, diagnostica, gradienti (fp64)")
def r0():
    g = torch.Generator().manual_seed(21)
    E, n = 32, 50
    zt = torch.randn(n, E, generator=g, dtype=torch.float64) * 2
    zs = (zt + 0.7 * torch.randn(n, E, generator=g, dtype=torch.float64)).requires_grad_(True)
    # KL_U with U = every expert == full KL; and == the independent per-token reference
    assert abs(float(RDm.kl_zona(zs, zt, E, E)) - float(RDm.kl_piena(zs, zt))) < 1e-12
    for zona_, ranghi_, k_ in (((16, 8), (4, 9), 8), ((3, 2), (1, 8), 8), ((5, 1), (2, 4), 2)):
        rif = _termine_rif(zs, zt, "cammino", zona_, ranghi_, 1.0, 0.3, k_)
        val = RDm.kl_zona(zs, zt, *zona_) + RDm.massa_a_rischio(zs, zt, ranghi_, 0.3, k_)
        assert abs(float(val) - float(rif)) < 1e-10, (zona_, ranghi_, float(val), float(rif))
        g1, = torch.autograd.grad(val, zs); g2, = torch.autograd.grad(rif, zs)
        assert float((g1 - g2).norm() / g2.norm()) < 1e-10, zona_
    # finite differences on one logit (KL_U zone and the boundary held fixed by construction: eps tiny)
    f = lambda z: float(RDm.kl_zona(z, zt, 16, 8) + RDm.massa_a_rischio(z, zt, (4, 9), 0.3, 8))   # noqa: E731
    g_, = torch.autograd.grad(RDm.kl_zona(zs, zt, 16, 8) + RDm.massa_a_rischio(zs, zt, (4, 9), 0.3, 8), zs)
    ij = divmod(int(g_.abs().argmax()), E)
    zp, zm = zs.detach().clone(), zs.detach().clone()
    zp[ij] += 1e-6; zm[ij] -= 1e-6
    fd = (f(zp) - f(zm)) / 2e-6
    assert abs(fd - float(g_[ij])) < 1e-5 * max(1.0, abs(fd)), (fd, float(g_[ij]))
    # ranks 1-8 with tau -> 0: M -> 1 - mass kept (R7 §4.1); identical logits: ov8 = massa = 1, KL_U = 0
    ov, ma = RDm.diagnostica(zs, zt, 8)
    M0 = float(RDm.massa_a_rischio(zs, zt, (1, 8), 1e-7, 8))
    assert abs(M0 - (1 - ma)) < 1e-6, (M0, ma)
    ov1, ma1 = RDm.diagnostica(zt, zt, 8)
    assert ov1 == 1.0 and abs(ma1 - 1) < 1e-6 and abs(float(RDm.kl_zona(zt, zt))) < 1e-12
    # a bf16 router cache is refused (R7 §1.0)
    d = TMP / "cache_router_bf16"
    if not d.exists():
        _scrivi_cache(d, TEACH_W, {"train": [0], "heldout": [0]}, 8,
                      {RDm.NOME_TENSORE.format(0): (lambda lato, i: np.zeros((8, E_N), np.uint16), "bf16", np.uint16)},
                      "piano_router.json", {"strati": [0], "n_expert": E_N, "n_embd": H, "salva_tipo": "bf16"})
    try:
        RDm.SorgenteRouter(d)
        raise AssertionError("cache bf16 accettata")
    except ValueError:
        pass
    return f"(KL_U == riferimento per token, gradiente == riferimento, differenze finite {fd:.6e}; M(1-8, tau->0) = 1 - massa {M0:.4f}; ov8 {ov:.3f}; bf16 rifiutata)"


@prova("R1 --router-cammino 0 (con cache e parametri) == trainer originale, al bit (ricetta B+)")
def r1():
    o1 = corri(ORIG, "R1_orig", ricetta_bplus())
    o2 = corri(NUOVO, "R1_nuovo", ricetta_bplus() + ["--router-cammino", "0", "--router-cache", str(cache_router()),
                                                     "--router-cammino-zona", "2,1"])
    d = confronta(o1, o2)
    assert not d, f"{len(d)} differences: {d[:5]}"
    assert "router:" not in (o2 / "registro.log").read_text()


def _r2(nome: str, modo: str, W: float, lam=None) -> str:
    cfg = ["--rango-esperti", "4", "--rango-densi", "4", "--famiglie", "attn", "--residui-shexp", "--kl4-peso", "0",
           "--kl-peso", "1", "--selezione", "paper", "--passi", "1", "--heldout-ogni", "1"] + F32_ON
    extra = ["--router-cache", str(cache_router()), "--router-cammino-perdita", modo, "--router-cammino-zona", "2,1",
             "--router-cammino-ranghi", "1-3"]
    if lam:
        (TMP / f"{nome}_lambda.json").write_text(json.dumps({str(k): v for k, v in lam.items()}))
        extra += ["--router-cammino-lambda", str(TMP / f"{nome}_lambda.json")]
    c_ = {}
    for suff, w in (("w0", "0"), ("w", str(W))):
        with CatturaGrad() as c:
            o = corri_seg(f"{nome}_{suff}", cfg + ["--router-cammino", w] + extra)
        c_[suff] = (c, o)
    st, resE, resD, resS, F = _studente_rif(StudenteSeg, esperti=4)
    i_r = AS.righe_in_ordine(list(range(10)), 20261007)[0]
    lot = CaricatoreFinto(SorgenteFinta(TEACH_W), "cpu").prendi("train", [i_r])
    val_ref = _rif_router(st, lot, i_r, W, modo, lam)
    F.gradienti_log()
    rif = {"esperti": [p.grad for p in CKm.parametri(resE, "C") + CKm.parametri(resE, "D")],
           "densi": [p.grad for p in RTm.parametri(resD)], "densi_shexp": [p.grad for p in RTm.parametri(resS)],
           **{b: [p.grad for p in F.blocchi()[b][1]] for b in F.nomi()}}
    ordine = ["esperti", "densi", "densi_shexp"] + BLOCCHI_F32
    cw, c0 = c_["w"][0].chiamate, c_["w0"][0].chiamate
    assert len(cw) == len(c0) == len(ordine), (len(cw), len(c0))

    def z(t, *altri):                                               # None (no grad) = zeros of the tensor's shape
        if t is not None:
            return t.reshape(-1).double()
        rif_ = next(u for u in altri if u is not None)
        return torch.zeros(rif_.numel(), dtype=torch.float64)
    err = {}
    for b, chw, ch0 in zip(ordine, cw, c0):
        assert len(chw) == len(ch0) == len(rif[b]), (b, len(chw), len(ch0), len(rif[b]))
        d_ = torch.cat([z(x, y, r) - z(y, x, r) for x, y, r in zip(chw, ch0, rif[b])])
        r_ = torch.cat([z(r, x, y) for x, y, r in zip(chw, ch0, rif[b])])
        assert float(r_.norm()) > 0, f"{b}: il termine router non arriva a questo blocco nel riferimento"
        err[b] = rel(d_, r_)
    assert max(err.values()) < 2e-3, err
    s = storia(c_["w"][1])
    assert abs(s[1]["router"] - val_ref) <= 1e-5 * max(1.0, val_ref), (s[1]["router"], val_ref)
    assert "router" in s[0] and "held_router" in s[1] and "held_router_ov8_strati" in s[1] and "router_massa" in s[1]
    reg = (c_["w"][1] / "registro.log").read_text()
    assert f"router: R-CAMMINO ({modo}) peso {W:g}" in reg, reg[-500:]
    return f"{modo}{' lambda' if lam else ''}: termine {val_ref:.4f}, errore max {max(err.values()):.1e}"


@prova("R2 R-CAMMINO: gradiente(W) - gradiente(0) == autograd del riferimento indipendente (KL_U + massa a rischio), "
       "su router F32 e blocchi a monte; anche con lambda per strato e con la KL piena")
def r2():
    m = [_r2("R2_cammino", "cammino", 20.0), _r2("R2_lambda", "cammino", 20.0, {0: 1.0, 1: 2.0, 2: 1.0}),
         _r2("R2_piena", "kl_piena", 20.0)]
    return "(" + "; ".join(m) + "; router F32 = ffn_gate_inp dentro f32_router)"


@prova("R3 R-CAMMINO acceso: gradienti fuso == separato (ricetta B++ con CKA e rotazioni)")
def r3():
    e, r1_, r2_ = _sep_fuso("R3", ricetta_bplus() + BPP_NUOVI + ["--router-cammino", "0.5", "--router-cache",
                                                                str(cache_router()), "--router-cammino-zona", "2,1",
                                                                "--router-cammino-ranghi", "1-3"])
    assert max(e) < 1e-4, dict(zip(ORD_BPP, e))
    assert abs(r1_["router"] - r2_["router"]) <= 1e-5 * max(1.0, r1_["router"]), (r1_["router"], r2_["router"])
    return f"(max {max(e):.1e} su {len(e)} blocchi; router sep {r1_['router']:.5f} fuso {r2_['router']:.5f})"


@prova("R4 errori router: senza cache -> 5; con prior di lettera -> 5; zona/ranghi errati -> 5; righe mancanti -> 5 "
       "(con --router-parziale corre)")
def r4():
    corri(NUOVO, "R4_cache", ["--router-cammino", "1"], rc_atteso=5)
    corri(NUOVO, "R4_prior", ["--router-cammino", "1", "--router-cache", str(cache_router()), "--prior-lettere-peso", "1"],
          rc_atteso=5)
    corri(NUOVO, "R4_zona", ["--router-cammino", "1", "--router-cache", str(cache_router()), "--router-cammino-ranghi", "5-2"],
          rc_atteso=5)
    parz = cache_router("cache_router_parz", righe_train=AS.righe_in_ordine(list(range(10)), 20261007)[1:4])
    corri(NUOVO, "R4_parz", ["--router-cammino", "1", "--router-cache", str(parz)], rc_atteso=5)
    o = corri(NUOVO, "R4_parz_si", ["--router-cammino", "1", "--router-cache", str(parz), "--router-parziale"])
    s = storia(o)
    assert "router" not in s[1] and "router" in s[2], (s[1].keys(), s[2].keys())   # row of step 1 not cached


def _cache_proprio(nome: str, teacher: Path, lato_righe: dict, rumore: float) -> Path:
    """Router cache (f32, full rows) whose teacher logits are the synthetic student's OWN router logits (+ noise)."""
    d = TMP / nome
    if d.exists():
        return d
    st = StudenteSeg(strati=range(N_STRATI))
    zs = {}
    hs = [st.moduli[L].mlp.register_forward_hook(lambda m, a, o, L=L: zs.__setitem__(L, RDm.router_logit(
        m, a[0].reshape(-1, H)).detach())) for L in range(N_STRATI)]
    car = CaricatoreFinto(SorgenteFinta(teacher), "cpu")
    logit = {}
    with torch.no_grad():
        for lato, rr in lato_righe.items():
            for i in rr:
                lot = car.prendi(lato, [i])
                st.nascosto(lot.ids, inizi=lot.inizi)
                g = torch.Generator().manual_seed(_seed(13, lato, i))
                logit[(lato, i)] = {L: (zs[L] + rumore * torch.randn(zs[L].shape, generator=g)).numpy() for L in zs}
    for h in hs:
        h.remove()
    _scrivi_cache(d, teacher, lato_righe, T_RIGA,
                  {RDm.NOME_TENSORE.format(L): ((lambda lato, i, L=L: logit[(lato, i)][L]), "f32", np.float32)
                   for L in range(N_STRATI)}, "piano_router.json",
                  {"strati": list(range(N_STRATI)), "n_expert": E_N, "n_embd": H, "salva_tipo": "f32"})
    return d


@prova("R5 prova 0 --forza-scelte-maestro: scelte del maestro = scelte proprie -> forzato == libero; scelte diverse -> "
       "cambia; JSON, uscita 0 senza allenare")
def r5():
    held = {"heldout": [0, 1]}
    cp = _cache_proprio("forza_proprio_paper", TEACH_W, held, 0.0)
    cq = _cache_proprio("forza_proprio_kl24", TEACH_Q, held, 0.0)
    o = corri_seg("R5_proprio", ["--rango-esperti", "0", "--rango-densi", "0", "--forza-scelte-maestro", f"{cp},{cq}"])
    r = json.load(open(o / "forza_scelte_maestro.json"))
    assert set(r["forzato"]) == {"kl24", "paper"} and r["token_forzati_frazione"] == {"kl24": 1.0, "paper": 1.0}, r
    for k in ("kl24", "paper"):
        assert abs(r["differenza"][k]) <= 1e-6 * max(1.0, abs(r["libero"][k])), r
    assert r["ov8_studente_libero_sulle_scelte_maestro"] == 1.0, r
    assert not (o / "storia.json").exists() and "PROVA 0" in (o / "registro.log").read_text()
    cp2 = _cache_proprio("forza_rumore_paper", TEACH_W, held, 3.0)
    cq2 = _cache_proprio("forza_rumore_kl24", TEACH_Q, held, 3.0)
    o2 = corri_seg("R5_rumore", ["--rango-esperti", "0", "--rango-densi", "0", "--forza-scelte-maestro", f"{cp2},{cq2}"])
    r2_ = json.load(open(o2 / "forza_scelte_maestro.json"))
    assert r2_["libero"] == r["libero"] and all(abs(r2_["differenza"][k]) > 1e-6 for k in ("kl24", "paper")), r2_
    assert r2_["ov8_studente_libero_sulle_scelte_maestro"] < 1.0
    o3 = corri_seg("R5_solo_kl24", ["--rango-esperti", "0", "--rango-densi", "0", "--forza-scelte-maestro", f"no,{cq2}"])
    assert set(json.load(open(o3 / "forza_scelte_maestro.json"))["forzato"]) == {"kl24"}
    corri_seg("R5_sorgente", ["--forza-scelte-maestro", f"{cq2},no"], rc_atteso=5)    # KL24 cache given as paper cache
    return (f"(proprie: differenza {r['differenza']}; rumore 3: differenza {r2_['differenza']}, ov8 libero "
            f"{r2_['ov8_studente_libero_sulle_scelte_maestro']:.3f})")


@prova("R6 prova 0 come il lanciatore: tappa B+ in collegamenti simbolici + --riprendi + ricetta B+ (down basso rango): "
       "cancelli di presenza PRIMA delle scelte imposte, libero = held della tappa, file della tappa intatti; livelli = B+")
def r6():
    tb = TMP / "R6_bplus" / "tappe"
    ob = corri_seg("R6_bplus", ricetta_bplus() + ["--passi", "2", "--tappe-dir", str(tb)])
    e = TMP / "R6_prova0"
    e.mkdir(parents=True, exist_ok=True)
    for n in ("cardan", "residui", "cardan_down", "scale", "scale_gateup"):          # = prova0_forza_scelte_1026.sh (ln -sfn)
        for suff in ("ultimo", "migliore"):
            os.symlink(tb / f"{n}_2.pt", e / f"{n}_{suff}.pt")
    os.symlink(tb / "ottimizzatori_2.pt", e / "ottimizzatori_ultimo.pt")
    impr = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(tb.glob("*_2.pt"))}
    held = {"heldout": [0, 1]}
    cp = _cache_proprio("forza_rumore_paper", TEACH_W, held, 3.0)
    cq = _cache_proprio("forza_rumore_kl24", TEACH_Q, held, 3.0)
    o = corri_seg("R6_prova0", ricetta_bplus() + ["--passi", "4", "--riprendi", "--forza-scelte-maestro", f"{cp},{cq}"])
    reg = (o / "registro.log").read_text()
    i_dn, i_p0 = reg.index("down low-rank: rango 4 su 3 strati"), reg.index("PROVA 0")
    assert "cancello di presenza: ok" in reg[i_dn:i_p0] and "RIPRESA dal passo 2" in reg[:i_p0], reg[-800:]
    r = json.load(open(o / "forza_scelte_maestro.json"))
    s2 = [x for x in storia(ob) if x["passo"] == 2][0]
    assert abs(r["libero"]["kl24"] - s2["held_kl24"]) <= 1e-6 * s2["held_kl24"], (r["libero"], s2["held_kl24"])
    assert abs(r["libero"]["paper"] - s2["held_paper"]) <= 1e-6 * s2["held_paper"], (r["libero"], s2["held_paper"])
    assert all(abs(r["differenza"][k]) > 1e-6 for k in ("kl24", "paper")), r
    dopo = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(tb.glob("*_2.pt"))}
    assert dopo == impr, "la prova 0 ha scritto nei file della tappa"
    assert not (o / "storia.json").exists() and not any(o.glob("*_tmp.pt"))
    tb_ = (Path("@@ROOT:forge@@/stato") / "congiunto_tre_kl4_piu_1026.sh").read_text()
    tp_ = (Path("@@ROOT:forge@@/stato") / "prova0_forza_scelte_1026.sh").read_text()
    lv = [tb_[tb_.index("for LIV in ") + len("for LIV in "):].split("; do")[0].strip(),
          tp_[tp_.index("LIVELLI=(") + len("LIVELLI=("):].split(")")[0].strip()]
    assert lv[0] == lv[1] == '"CARDAN_GRUPPI=1 E2E_RIUSO=1" "CARDAN_GRUPPI=1" ""', lv
    return f"(libero = held della tappa 2 di B+; differenza {r['differenza']}; tappa intatta; livelli {lv[1][:44]})"


if __name__ == "__main__":
    print(f"cartella {TMP}", flush=True)
    tutte = {f.__name__: f for f in (b0, b1, b2, b3, b4, b5, b6, s1, s2, s3, s4, s5, s6, r0, r1, r2, r3, r4, r5, r6)}
    for nome in (sys.argv[1:] or list(tutte)):
        tutte[nome]()
        _trim()
    ok = sum(1 for _, b, _ in RISULTATI if b)
    print(f"\n{ok}/{len(RISULTATI)} PASS (cartella {TMP}, RSS max "
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20:.2f} GiB)")
    sys.exit(0 if ok == len(RISULTATI) else 1)

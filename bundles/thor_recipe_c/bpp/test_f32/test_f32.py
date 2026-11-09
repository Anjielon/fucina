#!/usr/bin/env python3
"""CPU tests of F32-E2E (SINTESI_1025, 26/10/2026; ANTAGONISTA_1026.md §2): --allena-f32 / --solo-f32 / --rango-densi 0 /
--prior-lettere-guardia in cardan_congiunto_scale.py + f32_blocco.py. No GPU, no live service, no deletion.
Same synthetic harness as test_l6 (copied, not imported: test_l6 writes into the L6 tree): the student subclasses the real
e2e_studente.Studente with REAL MoEStudente / StratoEsperti layers on synthetic TQ1_0R_1024 / TQ1_0 bytes, MotoreTorch on CPU;
the trainer's `main` runs for real. Here the student ALSO carries the 11 F32 families of C as REAL e2e_studente.TensoreF32
masters (the class the real Studente(allena_f32=...) builds), installed before every layer call.

  F*  flags off == original (test_f32/originale_L6_cardan_congiunto_scale.py = L6/codice trainer, sha 56f8e0be...): every
      saved tensor and every logged value bitwise equal (separate, --fuso, alternating, KL4 0 + full KL like the F32 run,
      L2/L3/L6 levers on, through --riprendi with tappe); --prior-lettere-guardia only adds the held_prior keys
  P1  --solo-f32 (dry run of the real configuration): step-1 gradients on EVERY F32 tensor of every family and on nothing
      else; at the end every non-F32 tensor of the student (expert bytes, d0, theta, dense weights, embedding, head) is
      bitwise the start; every F32 tensor moved; log classes keep the sign and master = sign * exp(phi) where moved; the
      pack f32_ultimo.pt = the live masters; tappe f32_<passo>.pt + ottimizzatori_<passo>.pt (optF32, 3 blocks)
  P2  --riprendi (2 steps, then to 4) == 4 uninterrupted steps, at the bit (pt, Adam, tappe, storia)
  P3  fp64: the trainer's F32 gradients (checkpointed backward, captured before clipping) == full-graph autograd of the same
      loss (paper row + full-KL row), and == central finite differences on a log-space norm and a router element
  P4  configuration errors -> code 5 (solo-f32 with another block, alternato, unknown family, router in log space, solo-f32
      alone, a zero in a log class, --fuso); guards: a theta != 0 under --solo-f32 -> code 7, an F32 family the forward
      does not read -> presence gate, code 2
Run (one job, 4 threads, nice 19, RSS < 2 GB):
    HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 nice -n 19 \\
      @@ROOT:venv@@/bin/python3 test_f32/test_f32.py
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
ORIG_SHA = "56f8e0be969c2165ab698b26d422a080aa00598463ad7d70f3796075ddb73d81"     # L6/codice/cardan_congiunto_scale.py
REF_LIVE = Path("@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/SINTESI_1025/L6/codice/cardan_congiunto_scale.py")
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
TMP = Path("@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/SINTESI_1025/F32/collaudo") / time.strftime("test_%Y%m%d_%H%M%S")
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


ORIG = _Pigro("ccs_originale_l6", QUI / "originale_L6_cardan_congiunto_scale.py")
NUOVO = _Pigro("ccs_nuovo_f32", COD / "cardan_congiunto_scale.py")
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

ORIG = _Pigro("ccs_originale_l6", QUI / "originale_L6_cardan_congiunto_scale.py")   # noqa: F811 (same as the harness)


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


# ───────────────────────────────────────── F: flags off == original ───────────────────────────────────────────────────
@prova("F0 la copia di riferimento e' il trainer di L6/codice (sha)")
def f0():
    h = hashlib.sha256((QUI / "originale_L6_cardan_congiunto_scale.py").read_bytes()).hexdigest()
    assert h == ORIG_SHA, h
    return f"(= L6/codice vivo: {hashlib.sha256(REF_LIVE.read_bytes()).hexdigest() == ORIG_SHA})"


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


@prova("F4 flag spenti == originale: KL4 0 + KL piena, selezione paper (la perdita della corsa F32)")
def f4():
    coppia("F4", ["--kl4-peso", "0", "--kl-peso", "1", "--selezione", "paper"])


@prova("F5 flag spenti == originale: leve L2/L3/L6 accese (scale down, down basso rango, scale gate/up, prior EMA)")
def f5():
    coppia("F5", LEVE_L2 + ["--scale-gateup-lr", "1e-3", "--kl4-peso", "0", "--prior-lettere-peso", "1",
                            "--prior-lettere-pacchi", "2", "--prior-lettere-ema", "0.9"])


@prova("F6 flag spenti == originale attraverso una --riprendi (2 passi, poi fino a 4), con tappe")
def f6():
    for mod, n in ((ORIG, "F6_orig"), (NUOVO, "F6_nuovo")):
        t = str(TMP / n / "tappe")
        corri(mod, n, LEVE_L2 + ["--tappe-dir", t, "--passi", "2"])
        corri(mod, n, LEVE_L2 + ["--tappe-dir", t, "--passi", "4", "--riprendi"])
    d = confronta(TMP / "F6_orig", TMP / "F6_nuovo", tappe=True)
    assert not d, f"{len(d)} differences: {d[:5]}"


@prova("F7 --prior-lettere-guardia aggiunge SOLO held_prior/held_pS/held_pT (+ prior/pS/pT al passo 0): resto al bit")
def f7():
    o1 = corri(ORIG, "F7_orig", ["--kl4-peso", "0", "--kl-peso", "1"])
    o2 = corri(NUOVO, "F7_nuovo", ["--kl4-peso", "0", "--kl-peso", "1", "--prior-lettere-guardia"])
    extra = {"held_prior", "held_pS", "held_pT", "prior", "pS", "pT"}
    t1 = {k: v for k, v in tensori(o1).items() if not k.startswith("ottimizzatori")}
    t2 = {k: v for k, v in tensori(o2).items() if not k.startswith("ottimizzatori")}
    d = uguali(t1, t2, "pt", ESCLUSI | extra) + uguali(storia(o1), storia(o2, ESCLUSI | extra), "storia")
    assert not d, f"{len(d)} differences: {d[:5]}"
    s2 = storia(o2)
    assert all("held_prior" in r for r in s2 if "held_paper" in r) and "prior" in s2[0], "chiavi del prior assenti"


# ───────────────────────────────────────── P: the F32-E2E configuration ───────────────────────────────────────────────
@prova("P1 --solo-f32: gradiente su OGNI tensore F32 e su nient'altro; tutto il resto dello studente al bit")
def p1():
    t = TMP / "P1" / "tappe"
    with CatturaGrad() as c:
        out = corri(NUOVO, "P1", SOLO + ["--passi", "4", "--heldout-ogni", "2", "--tappe-dir", str(t)])
    st = StudenteF32.ultimo
    nf = len(st.f32)
    assert nf == 3 * 10 + 1 and set(st.f32_conteggi) == set(FB.TUTTE_C), (nf, st.f32_conteggi)
    # 1) what the optimisers clip and step: exactly the 3 F32 blocks, every tensor with a nonzero step-1 gradient
    assert len(c.chiamate) == 4 * len(BLOCCHI_F32), len(c.chiamate)
    passo1 = c.chiamate[:len(BLOCCHI_F32)]
    assert sum(len(ch) for ch in passo1) == nf, sum(len(ch) for ch in passo1)
    zero = [i for ch in passo1 for i, g in enumerate(ch) if g is None or not bool((g != 0).any())]
    assert not zero, f"tensori F32 senza gradiente al passo 1: {zero}"
    # 2) nothing else: theta without gradient and exactly 0; every non-F32 tensor bitwise the start
    assert all(p.grad is None for p in st.parametri()), "theta con gradiente"
    assert max(float(p.abs().max()) for p in st.parametri()) == 0.0, "theta mosse"
    assert impronta(st) == st.impronta0, "un tensore non F32 e' cambiato"
    # 3) every F32 tensor moved; log classes: sign kept, master = sign * exp(phi) where moved
    pk = torch.load(out / "f32_ultimo.pt", map_location="cpu", weights_only=False)
    mossi, log_ok = 0, 0
    for n, v0 in st.f32_0.items():
        live = st.f32[n].master.detach()
        v = pk["valori"][n].reshape(v0.shape).to(v0.dtype)
        assert torch.equal(v, live), f"{n}: il pacco non e' il master vivo"
        mossi += int(not torch.equal(v, v0))
        if n in pk["phi"]:
            assert torch.equal(torch.sign(v), torch.sign(v0)), f"{n}: segno cambiato"
            ch = v != v0
            ric = (torch.sign(v0) * torch.exp(pk["phi"][n].to(v0.dtype))).to(v.dtype)
            assert torch.equal(v[ch], ric[ch]), f"{n}: master != sign * exp(phi)"
            log_ok += 1
    assert mossi == nf, f"{nf - mossi} tensori F32 fermi"
    assert log_ok == 3 * 6 + 1, log_ok                   # norms (5 per layer + output) and ssm_a, per layer
    # 4) tappe, Adam, layout of the other files, log lines
    for p_ in (2, 4):
        assert (t / f"f32_{p_}.pt").exists(), p_
        d = torch.load(t / f"ottimizzatori_{p_}.pt", map_location="cpu", weights_only=False)
        assert sorted(d["optF32"]) == BLOCCHI_F32, sorted(d["optF32"])
    assert torch.load(out / "residui_ultimo.pt", weights_only=False)["latenti"] == {}
    assert torch.load(out / "cardan_ultimo.pt", weights_only=False)["rango"] == 0
    righe = [r for r in storia(out) if "held_paper" in r]
    assert len(righe) == 2 and all(r["congelati"] is True and "held_prior" in r and "f32_rel_norme" in r for r in righe)
    assert all(r["blocchi"] == BLOCCHI_F32 for r in storia(out)[1:])
    reg = (out / "registro.log").read_text()
    for s_ in ("cancello di presenza F32 (un tensore per famiglia): ok", "F32 gradienti non nulli al primo passo",
               "ramo denso non installato (--rango-densi 0)", "ramo gate/up CARDAN non installato"):
        assert s_ in reg, s_
    if json.load(open(out / "esportazione.json"))["passo"] > 0:
        assert json.load(open(out / "f32.json"))["tensori"] == nf
    return (f"({nf} tensori F32, 11 famiglie; passo 1: {nf}/{nf} con gradiente, theta 0 senza gradiente, "
            f"impronta del resto uguale; {mossi}/{nf} mossi; rel norme {righe[-1]['f32_rel_norme']:.2e} router "
            f"{righe[-1]['f32_rel_router']:.2e})")


@prova("P2 --solo-f32 --riprendi (2 passi, poi fino a 4) == 4 passi filati, al bit (pt, Adam F32, tappe, storia)")
def p2():
    corri(NUOVO, "P2_filato", SOLO + ["--passi", "4", "--heldout-ogni", "2", "--tappe-dir", str(TMP / "P2_filato" / "tappe")])
    t = str(TMP / "P2_ripreso" / "tappe")
    corri(NUOVO, "P2_ripreso", SOLO + ["--passi", "2", "--heldout-ogni", "2", "--tappe-dir", t])
    corri(NUOVO, "P2_ripreso", SOLO + ["--passi", "4", "--heldout-ogni", "2", "--tappe-dir", t, "--riprendi"])
    d = confronta(TMP / "P2_filato", TMP / "P2_ripreso", tappe=True)
    assert not d, f"{len(d)} differences: {d[:5]}"
    return f"({len(tensori(TMP / 'P2_filato', 'tappe'))} file di tappa al bit)"


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


def _perdita64(st) -> float:
    """The same loss as a plain fp64 function of a B64 student (paper KD on the teacher top-8 + 0.5 CE + full KL), for
    central finite differences. Teacher logits from the same cache rows (constants)."""
    rng = random.Random(20261007)
    ow, oq = list(range(10)), list(range(10))
    rng.shuffle(ow); rng.shuffle(oq)
    lot = CaricatoreFinto(SorgenteFinta(TEACH_W), "cpu").prendi("train", [ow[0]])
    lq = CaricatoreFinto(SorgenteFinta(TEACH_Q), "cpu").prendi("train", [oq[0]])
    W, WT64 = st.testa(torch.float64), _WT.double()
    with torch.no_grad():
        zS = st.nascosto(lot.ids, inizi=lot.inizi).double() @ W.T
        zT = lot.hT.double() @ WT64.T
        om = torch.topk(zT, 8, -1).indices
        lpT, lpS = torch.log_softmax(zT.gather(1, om), -1), torch.log_softmax(zS.gather(1, om), -1)
        kd = (lpT.exp() * (lpT - lpS)).sum() / zS.shape[0]
        tgt, ok = NUOVO.bersagli(lot)
        ce = -(torch.log_softmax(zS, -1).gather(1, tgt[:, None])[:, 0] * ok).sum() / max(1, int(ok.sum()))
        lS = torch.log_softmax(st.nascosto(lq.ids, inizi=lq.inizi).double() @ W.T, -1)
        lT = torch.log_softmax(lq.hT.double() @ WT64.T, -1)
        kl = (lT.exp() * (lT - lS)).sum() / lq.ids.numel()
    return float(kd + 0.5 * ce + kl)


@prova("P3 gradienti F32 del trainer == autograd a grafo intero (stessa perdita); == differenze finite fp64 su phi e router")
def p3():
    with CatturaGrad() as c:
        corri(NUOVO, "P3", SOLO + ["--passi", "1", "--heldout-ogni", "1"])
    st = StudenteF32(strati=range(N_STRATI), allena_f32=FB.famiglie("tutte"))
    F = FB.BloccoF32(st, {"norme": 1e-2, "router": 3e-3, "ssm_a": 1e-2, "ssm": 3e-3}, FB.log_classi("norme,ssm_a"))
    _perdita_rif(st, True)
    F.gradienti_log()
    nb = F.nomi()
    rif = [[p.grad.detach().clone() for p in F.blocchi()[b][1]] for b in nb]
    err = max(rel(torch.cat([x.reshape(-1).double() for x in a_]), torch.cat([x.reshape(-1).double() for x in b_]))
              for a_, b_ in zip(c.chiamate[:len(nb)], rif))
    assert err < 1e-6, f"trainer vs grafo intero: errore relativo {err:.3e}"
    # central finite differences of the fp64 loss on a B64 copy of the same student (same F32 values, same head)
    StudenteTernario.RICETTA = "B64"
    try:
        st64 = StudenteF32(strati=range(N_STRATI), allena_f32=FB.famiglie("tutte"))
    finally:
        StudenteTernario.RICETTA = "B32"
    eps, fd = 1e-6, []
    for blocco, nome in (("f32_norme", "blk.1.attn_norm.weight"), ("f32_router", "blk.2.ffn_gate_inp.weight"),
                         ("f32_ssm", "blk.0.ssm_a")):
        voci_b = [v for c_ in FB.BLOCCHI[blocco] for v in F.voci if v.classe == c_]
        i_v = [k for k, v in enumerate(voci_b) if v.tf.nome == nome][0]
        g_t = c.chiamate[nb.index(blocco)][i_v].reshape(-1).double()
        idx = int(g_t.abs().argmax())
        m = st64.f32[nome].master
        vals = []
        for sgn in (1, -1):
            with torch.no_grad():
                salvo = m.detach().clone()
                if voci_b[i_v].log:
                    m.view(-1)[idx] = m.reshape(-1)[idx] * math.exp(sgn * eps)
                else:
                    m.view(-1)[idx] = m.reshape(-1)[idx] + sgn * eps
                vals.append(_perdita64(st64))
                m.copy_(salvo)
        g_fd = (vals[0] - vals[1]) / (2 * eps)
        fd.append(abs(g_fd - float(g_t[idx])) / max(abs(float(g_t[idx])), 1e-12))
        assert fd[-1] < 2e-3, f"{nome}[{idx}]: differenze finite {g_fd:.8e} contro trainer {float(g_t[idx]):.8e}"
    return (f"(errore relativo trainer/grafo intero {err:.1e}; differenze finite fp64: phi norma {fd[0]:.1e}, "
            f"router {fd[1]:.1e}, phi ssm_a {fd[2]:.1e})")


@prova("P4 errori di configurazione -> 5; theta mossa sotto --solo-f32 -> 7; famiglia F32 fuori dal forward -> 2")
def p4():
    casi = [("P4_altro_blocco", SOLO + ["--rango-esperti", "4"]),
            ("P4_alternato", ["--allena-f32", "tutte", "--modo", "alternato"]),
            ("P4_famiglia", ["--allena-f32", "xyz"]),
            ("P4_router_log", SOLO + ["--f32-log", "norme,router"]),
            ("P4_solo", ["--solo-f32"]),
            ("P4_fuso", SOLO + ["--fuso", "--kl-peso", "0", "--kl4-peso", "1"])]
    for n, extra in casi:
        corri(NUOVO, n, extra, rc_atteso=5)
    StudenteF32.ZERO_IN = "attn_norm"
    try:
        corri(NUOVO, "P4_zero_log", SOLO, rc_atteso=5)
    finally:
        StudenteF32.ZERO_IN = None
    StudenteF32.THETA_SPORCO = True
    try:
        corri(NUOVO, "P4_theta", SOLO + ["--passi", "2", "--heldout-ogni", "1"], rc_atteso=7)
    finally:
        StudenteF32.THETA_SPORCO = False
    StudenteF32.FUORI = "ssm_norm"
    try:
        o = corri(NUOVO, "P4_fuori", SOLO, rc_atteso=2)
    finally:
        StudenteF32.FUORI = None
    assert "ssm_norm" in (o / "registro.log").read_text()
    return f"({len(casi) + 1} configurazioni rifiutate con 5, guardia theta 7, cancello di presenza 2)"


if __name__ == "__main__":
    print(f"cartella {TMP}", flush=True)
    tutte = {f.__name__: f for f in (f0, f1, f2, f3, f4, f5, f6, f7, p1, p2, p3, p4)}
    for nome in (sys.argv[1:] or list(tutte)):
        tutte[nome]()
        _trim()
    ok = sum(1 for _, b, _ in RISULTATI if b)
    print(f"\n{ok}/{len(RISULTATI)} PASS (cartella {TMP}, RSS max "
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20:.2f} GiB)")
    sys.exit(0 if ok == len(RISULTATI) else 1)

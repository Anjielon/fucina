#!/usr/bin/env python3
"""CPU tests of Levers 2 and 5 (SINTESI_1025, 25/10) on TINY synthetic tensors. No GGUF, no GPU, no live service.

The student is a 3-layer synthetic model that SUBCLASSES the real e2e_studente.Studente (so avanti / indietro /
_indietro_troncato / esce / nascosto are the real code, including the E2E_SENZA_RICALCOLO held-graph path), with the real
cardan_kd_moduli experts branch and the real residui_ternari dense branch installed by the trainer. The teacher caches,
the calibration packs and the heads are synthetic. The trainer `main` runs for real (device cpu).

  F*  flags off == original (test_l2l5/originale_cardan_congiunto_scale.py, sha a3a92a94...): every exported tensor and
      every logged loss bitwise equal, in the separate, fused, alternating and full-KL paths; then the alignment switched
      ON with a forced zero weight must ALSO be bitwise equal (restructured step order + hooks do not perturb anything).
  G*  gradient checks of the new loss terms: torch.autograd.gradcheck of CKA / relative MSE; the injected alignment
      gradient through the real Studente backward == full autograd of (output loss + w * L_al) in the three backward
      paths; RS-KD: closed-form gradient == autograd of the sampled objective, and unbiased (mean -> exact full KL).
  R*  KL4 rotations: pack selection, cycling, "tutte"; prompt parse/recompose round trip on the real calib .bin.
Run (one job, 4 threads, RSS < 2 GB):
    HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \\
      @@ROOT:venv@@/bin/python3 test_l2l5/test_l2_l5.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import tempfile
import traceback
import types
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)
QUI = Path(__file__).resolve().parent
COD = QUI.parent
sys.path.insert(0, str(COD))
ORIG_SHA = "a3a92a940c70b7ee01d16365611013ee64c253aa1ed50744c82442e8aa9ebe51"

import cardan_kd_paper as KDP                       # noqa: E402  (path setup of the thor_v2 tree)
import e2e_studente as S                            # noqa: E402
import e2e_allena as A                              # noqa: E402
import allinea_stati as AS                          # noqa: E402
import paper_campionato as PC                       # noqa: E402
import kl4_rotazioni as KR                          # noqa: E402
import cardan_paper_perdita as PP                   # noqa: E402

KDP.riga_cicli = lambda *a, **k: None               # never write the real cicli.tsv
V, H, R_FF, E_N, N_STRATI, T_RIGA = 48, 64, 16, 4, 3, 64
LETTERE = [40, 41, 42, 43]


# ───────────────────────────────────────── synthetic student ─────────────────────────────────────────
class Attn(torch.nn.Module):
    def __init__(self, g, dt):
        super().__init__()
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            lin = torch.nn.Linear(H, H, bias=False)
            with torch.no_grad():
                lin.weight.copy_(torch.randn(H, H, generator=g, dtype=torch.float64) / math.sqrt(H))
            setattr(self, n, lin.to(dt))


class Moe(torch.nn.Module):
    def __init__(self, g, dt):
        super().__init__()
        self.esperti = types.SimpleNamespace(fam={"gate": types.SimpleNamespace(E=E_N, R=R_FF, C=H)})
        self.Wg = torch.nn.Parameter((torch.randn(E_N, R_FF, H, generator=g, dtype=torch.float64) / math.sqrt(H)).to(dt))
        self.Wu = torch.nn.Parameter((torch.randn(E_N, R_FF, H, generator=g, dtype=torch.float64) / math.sqrt(H)).to(dt))
        self.Wd = torch.nn.Parameter((torch.randn(E_N, H, R_FF, generator=g, dtype=torch.float64) / math.sqrt(R_FF)).to(dt))
        self.ultimo_idx = None
        self.instr = None


class Strato(torch.nn.Module):
    def __init__(self, g, dt):
        super().__init__()
        self.self_attn = Attn(g, dt)
        self.mlp = Moe(g, dt)


class StudenteFinto(S.Studente):
    """Real Studente methods (avanti, indietro, _indietro_troncato, esce, nascosto) on a synthetic 3-layer model."""

    def __init__(self, gguf=None, strati=range(N_STRATI), ricetta=None, motore_nome=None, device="cpu", log=print,
                 densi_ternari=(), dtype=torch.float32, **_):
        torch.nn.Module.__init__(self) if isinstance(self, torch.nn.Module) else None
        g = torch.Generator().manual_seed(1234)
        self.device = torch.device(device)
        self.strati = list(strati)
        self.strati_allenati = None
        self.ric = types.SimpleNamespace(ruota=False, dtype_gemm=dtype, dtype_residuo=dtype)
        self.cfg = types.SimpleNamespace(layer_types=["full_attention"] * 40)
        self.moduli = {L: Strato(g, dtype) for L in self.strati}
        self._emb = (torch.randn(V, H, generator=g, dtype=torch.float64)).to(dtype)
        self._testa = (torch.randn(V, H, generator=g, dtype=torch.float64) / math.sqrt(H)).to(torch.float32)
        self.eps = 1e-6
        self.norma_finale = torch.ones(H, dtype=torch.float32)
        self.oss = None

    def parametri(self):
        return [p for L in self.strati for p in self.moduli[L].parameters()]

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
        h2 = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)).reshape(B * T, H)
        if instr is None:
            idx = ((torch.arange(B * T) * 7 + L) % E_N).view(-1, 1)
        else:
            idx = instr
        m.mlp.ultimo_idx = idx
        e = idx[:, 0]
        ordine = torch.argsort(e, stable=True)
        cnt = torch.bincount(e, minlength=E_N)
        offs = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(cnt, 0)])
        xo = h2[ordine]
        gb = torch.cat([xo[offs[j]:offs[j + 1]] @ m.mlp.Wg[j].T for j in range(E_N)])
        ub = torch.cat([xo[offs[j]:offs[j + 1]] @ m.mlp.Wu[j].T for j in range(E_N)])
        if hasattr(m.mlp, "cardan"):
            dg, du = m.mlp.cardan(xo, types.SimpleNamespace(offs=offs))
            gb, ub = gb + dg.to(gb.dtype), ub + du.to(ub.dtype)
        a = torch.nn.functional.silu(gb) * ub
        oo = torch.cat([a[offs[j]:offs[j + 1]] @ m.mlp.Wd[j].T for j in range(E_N)])
        out = torch.empty_like(oo)
        out[ordine] = oo
        return x + out.view(B, T, H)


# ───────────────────────────────────────── synthetic data sources ─────────────────────────────────────────
def _seed(*k) -> int:
    return int(hashlib.sha256(repr(k).encode()).hexdigest()[:8], 16)


def ids_riga(nome: str, lato: str, i: int, T: int = T_RIGA) -> torch.Tensor:
    g = torch.Generator().manual_seed(_seed(nome, lato, i))
    return torch.randint(0, 38, (1, T), generator=g)


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
        ids = ids_riga(self.src.nome, lato, i)
        g = torch.Generator().manual_seed(_seed("h", self.src.nome, lato, i))
        hT = torch.randn(T_RIGA, H, generator=g).to(torch.bfloat16)
        inizi = torch.zeros(1, T_RIGA, dtype=torch.int32)
        inizi[0, ::16] = 1
        return A.Lotto(list(righe), ids.to(self.dev), hT.to(self.dev), inizi, ["finto"])


def leggi_mclp(log_path):
    out = {}
    for r in open(log_path, errors="replace"):
        i = r.find("MCLP\t")
        if i < 0:
            continue
        c = r[i:].rstrip("\n").split("\t")
        n = int(c[4])
        out[int(c[1])] = (int(c[2]), [float(x) for x in c[5:5 + n]])
    return out


def domande():
    g = torch.Generator().manual_seed(77)
    Q = []
    for i in range(20):
        L = int(torch.randint(6, 11, (1,), generator=g))
        lp = torch.randn(4, generator=g, dtype=torch.float64)
        Q.append({"i": i, "label": i % 4, "ids": torch.randint(0, 38, (L,), generator=g).tolist(),
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
GD.pacchi_kl4, GD.leggi_mclp = pacchi_kl4, leggi_mclp
sys.modules["gradienti_densi"] = GD
_WT = torch.randn(V, H, generator=torch.Generator().manual_seed(99))
A.scegli_gdn = lambda log: None
A.testa_studente = lambda st, k: st.testa(torch.float32)
A.testa_maestro = lambda dev, dt: (_WT.to(dev), "finto" * 8)
A.SorgenteCache = SorgenteFinta
A.Caricatore = CaricatoreFinto
S.Studente_vero = S.Studente


def scrivi_rotazioni(d: Path, N: int) -> None:
    """Rotated calibration + teacher logs consistent with domande(): rotation k adds token k at the end of the ids
    (different prompt), label (y - k) mod 4, teacher logprobs = rotated logits + a position bias."""
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


def scrivi_cache_stati(d: Path, teacher: Path, righe: dict, strati: list, n_tok: int, seme: int = 5) -> None:
    """A teacher-state cache in the cattura_stati_maestro.py layout (one block per side) for the synthetic rows."""
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
            for L in strati:
                x = torch.randn(n_tok, H, generator=torch.Generator().manual_seed(_seed(seme, lato, i, L))) * 3
                bits = x.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
                f = sd / f"l_out-{L}.bf16"
                with open(f, "wb") as fh:
                    np.array([H, n_tok, 1, 1], dtype=np.int64).tofile(fh); bits.tofile(fh)
                ten[f"l_out-{L}"] = {"file": f"seq_{k:04d}/l_out-{L}.bf16", "righe_uguali_a": "token"}
            seqs.append({"cartella": f"seq_{k:04d}", "tensori": ten})
        (b / "dump" / "manifesto.json").write_text(json.dumps({"esito": "ok", "sequenze": seqs,
                                                                "modello": {"n_vocab": V}}))
        (b / "blocco.json").write_text(json.dumps({"righe": rr, "lato": lato,
                                                   "sha256_ids": hashlib.sha256(ids.tobytes()).hexdigest()}))
        bl.append({"nome": nb, "lato": lato, "righe": rr})
    (d / "piano_stati.json").write_text(json.dumps({
        "strati": strati, "n_token": n_tok, "n_embd": H, "sorgente": str(teacher),
        "sorgente_piano_sha256": hashlib.sha256((teacher / "piano.json").read_bytes()).hexdigest(), "blocchi": bl}))


# ───────────────────────────────────────── running the two trainers ─────────────────────────────────────────
def carica_modulo(nome: str, p: Path):
    spec = importlib.util.spec_from_file_location(nome, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ORIG = carica_modulo("ccs_originale", QUI / "originale_cardan_congiunto_scale.py")
NUOVO = carica_modulo("ccs_nuovo", COD / "cardan_congiunto_scale.py")
TMP = Path(tempfile.mkdtemp(prefix="l2l5_", dir=os.environ.get("L2L5_TMP", None)))
TEACH_W = TMP / "teacher_wt"; TEACH_Q = TMP / "teacher_q8"
for d in (TEACH_W, TEACH_Q):
    d.mkdir(); (d / "piano.json").write_text(json.dumps({"finto": d.name}))
BASE = ["--gguf", "finto.gguf", "--device", "cpu", "--teacher", str(TEACH_W), "--teacher-kl24", str(TEACH_Q),
        "--strati-studente", "0-2", "--strati", "0-2", "--rango-esperti", "4", "--rango-densi", "4", "--famiglie", "attn",
        "--init-c", "zero", "--kl4-n", "12", "--passi", "3", "--heldout-ogni", "1", "--pazienza", "100",
        "--heldout-righe", "2", "--pezzo-kl", "32", "--topk", "8", "--riscaldamento", "1", "--lr-c", "3e-3", "--lr-d", "1e-3",
        "--lr-densi", "3e-3", "--selezione", "kl24_paper"]


def corri(mod, nome: str, extra: list, env: dict = None) -> Path:
    out = TMP / nome
    vecchio = {k: os.environ.get(k) for k in (env or {})}
    os.environ.update(env or {})
    S.Studente = StudenteFinto
    try:
        rc = mod.main(["--uscita", str(out)] + BASE + extra)
    finally:
        S.Studente = S.Studente_vero
        for k, v in vecchio.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    assert rc == 0, f"{nome}: main returned {rc}"
    return out


def tensori(out: Path) -> dict:
    r = {}
    for f in sorted(out.glob("*.pt")):
        r[f.name] = torch.load(f, map_location="cpu", weights_only=False)
    return r


def uguali(a, b, dove="") -> list:
    """Bitwise comparison of nested containers; returns the list of differences."""
    if isinstance(a, torch.Tensor):
        return [] if (isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)) \
            else [dove]
    if isinstance(a, dict):           # wall-clock fields and the alignment's own log fields are not compared
        ka, kb = set(a) - ESCLUSI, set(b) - ESCLUSI
        d = [] if ka == kb else [f"{dove}: keys {sorted(ka ^ kb, key=str)}"]
        for k in ka:
            if k in kb:
                d += uguali(a[k], b[k], f"{dove}/{k}")
        return d
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return [f"{dove}: len"]
        return sum((uguali(x, y, f"{dove}[{i}]") for i, (x, y) in enumerate(zip(a, b))), [])
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return []
    return [] if a == b else [f"{dove}: {a!r} != {b!r}"]


ESCLUSI = {"s", "secondi", "s_passo_medio", "allinea", "peso_allinea", "held_allinea"}


def storia(out: Path) -> list:
    s = json.load(open(out / "storia.json"))["storia"]
    return [{k: v for k, v in r.items() if k not in ESCLUSI} for r in s]


def confronta(o1: Path, o2: Path) -> list:
    d = uguali(tensori(o1), tensori(o2), "pt")
    d += uguali(storia(o1), storia(o2), "storia")
    return d


RISULTATI = []


def prova(nome):
    def dec(f):
        def run():
            try:
                msg = f()
                RISULTATI.append((nome, True, msg or ""))
                print(f"PASS {nome} {msg or ''}", flush=True)
            except Exception as e:                          # report and go on with the other tests
                RISULTATI.append((nome, False, f"{type(e).__name__}: {e}"))
                print(f"FAIL {nome}: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
        run.__name__ = f.__name__
        return run
    return dec


# ───────────────────────────────────────── F: flags off == original ─────────────────────────────────────────
@prova("F0 la copia di riferimento e' il trainer originale (sha)")
def f0():
    h = hashlib.sha256((QUI / "originale_cardan_congiunto_scale.py").read_bytes()).hexdigest()
    assert h == ORIG_SHA, h
    live = Path("@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/CARDAN_DOWN_1025/codice/cardan_congiunto_scale.py")
    return f"(= CARDAN_DOWN_1025: {hashlib.sha256(live.read_bytes()).hexdigest() == ORIG_SHA})"


def coppia(nome: str, extra: list, env=None, extra_nuovo=None, prima=None):
    o1 = corri(ORIG, f"{nome}_orig", extra, env)
    if prima:
        prima()
    o2 = corri(NUOVO, f"{nome}_nuovo", extra + (extra_nuovo or []), env)
    d = confronta(o1, o2)
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


def cache_per(nome: str, strati=(0, 1, 2), n_tok=32) -> Path:
    d = TMP / f"cache_{nome}"
    if not d.exists():
        righe = AS.righe_in_ordine(list(range(10)), 20261007)
        scrivi_cache_stati(d, TEACH_W, {"train": righe, "heldout": [0, 1]}, list(strati), n_tok)
    return d


class PesoZero:
    def __enter__(self):
        self.v = AS.peso_effettivo
        AS.peso_effettivo = lambda *a, **k: 0.0
        return self

    def __exit__(self, *e):
        AS.peso_effettivo = self.v


@prova("F5 allineamento ACCESO con peso forzato a 0 == originale (ordine dei passi, ganci) — separato")
def f5():
    c = cache_per("F5")
    with PesoZero():
        coppia("F5", [], extra_nuovo=["--allinea-peso", "1", "--allinea-cache", str(c)])


@prova("F6 allineamento ACCESO con peso forzato a 0 == originale — fuso e fuso con E2E_SENZA_RICALCOLO")
def f6():
    c = cache_per("F5")
    with PesoZero():
        coppia("F6a", ["--fuso"], extra_nuovo=["--allinea-peso", "1", "--allinea-cache", str(c)])
        coppia("F6b", ["--fuso"], env={"E2E_SENZA_RICALCOLO": "1", "E2E_GRAFO_STRATI": "2"},
               extra_nuovo=["--allinea-peso", "1", "--allinea-cache", str(c)])


@prova("F7 allineamento ACCESO con peso forzato a 0 == originale — KL piena e alternato")
def f7():
    c = cache_per("F5")
    with PesoZero():
        coppia("F7a", ["--kl-peso", "0.1"], extra_nuovo=["--allinea-peso", "1", "--allinea-cache", str(c)])
        coppia("F7b", ["--modo", "alternato", "--passi", "4"], extra_nuovo=["--allinea-peso", "1", "--allinea-cache", str(c)])


@prova("F8 kl4-rotazioni 1 e paper-campionato 0 espliciti == originale")
def f8():
    coppia("F8", [], extra_nuovo=["--kl4-rotazioni", "1", "--paper-campionato", "0"])


# ───────────────────────────────────────── trainer runs with the levers ON ─────────────────────────────────────────
@prova("L2a allineamento CKA rapporto acceso: separato, fuso, fuso+grafo tenuto; registra allinea/held_allinea; cambia il risultato")
def l2a():
    c = cache_per("F5")
    rif = corri(NUOVO, "L2_rif", [])
    msg = []
    for nome, extra, env in (("L2a_sep", [], None), ("L2a_fuso", ["--fuso"], None),
                             ("L2a_grafo", ["--fuso"], {"E2E_SENZA_RICALCOLO": "1", "E2E_GRAFO_STRATI": "2"})):
        o = corri(NUOVO, nome, extra + ["--allinea-peso", "1", "--allinea-cache", str(c)], env)
        s = json.load(open(o / "storia.json"))["storia"]
        assert all("allinea" in r and "held_allinea" in r for r in s[1:]), "allinea not logged"
        assert all(math.isfinite(r["L"]) and math.isfinite(r["allinea"]) for r in s[1:])
        # rapporto: w * L_al == L_out (sg ratio, eps negligible) -> L == 2 * L_out
        for r in s[1:]:
            L_out = r["L"] - r["peso_allinea"] * r["allinea"]
            assert abs(r["peso_allinea"] * r["allinea"] - L_out) <= 1e-4 * abs(L_out) + 1e-6, r
        assert uguali(tensori(rif)["cardan_ultimo.pt"], tensori(o)["cardan_ultimo.pt"]), "alignment had no effect"
        msg.append(f"{nome}: L_al {s[1]['allinea']:.4f}->{s[-1]['allinea']:.4f}, held {s[-1]['held_allinea']:.4f}")
    return "; ".join(msg)


@prova("L2b allineamento mse_rel fisso 0.5 + salta-inizi + strati 1-2 + token 16")
def l2b():
    c = cache_per("F5")
    o = corri(NUOVO, "L2b", ["--allinea-peso", "0.5", "--allinea-modo", "fisso", "--allinea-metrica", "mse_rel",
                             "--allinea-cache", str(c), "--allinea-salta-inizi", "--allinea-strati", "1-2",
                             "--allinea-token", "16"])
    s = json.load(open(o / "storia.json"))["storia"]
    assert all(r["peso_allinea"] == 0.5 for r in s[1:])
    return f"(mse_rel {s[1]['allinea']:.4f} -> {s[-1]['allinea']:.4f})"


@prova("L2c cache incompleta: errore all'avvio senza --allinea-parziale, corsa con")
def l2c():
    d = TMP / "cache_parziale"
    scrivi_cache_stati(d, TEACH_W, {"train": AS.righe_in_ordine(list(range(10)), 20261007)[:2]}, [0, 1, 2], 32)
    S.Studente = StudenteFinto
    try:
        rc = NUOVO.main(["--uscita", str(TMP / "L2c_no")] + BASE + ["--allinea-peso", "1", "--allinea-cache", str(d)])
    finally:
        S.Studente = S.Studente_vero
    assert rc == 5, rc
    corri(NUOVO, "L2c_si", ["--allinea-peso", "1", "--allinea-cache", str(d), "--allinea-parziale"])


@prova("L5a kl4-rotazioni 4 (ciclo e tutte): corse complete, maestro per rotazione, risultato diverso")
def l5a():
    rd = TMP / "rot"
    scrivi_rotazioni(rd, 4)
    rif = tensori(corri(NUOVO, "L5_rif", []))
    for modo in ("ciclo", "tutte"):
        o = corri(NUOVO, f"L5a_{modo}", ["--kl4-rotazioni", "4", "--kl4-rotazioni-dir", str(rd),
                                         "--kl4-rotazioni-modo", modo])
        assert uguali(rif["cardan_ultimo.pt"], tensori(o)["cardan_ultimo.pt"]), f"{modo}: no effect"
        assert "kl4 rotation 3" in (o / "registro.log").read_text()


@prova("L5b paper-campionato 16: corsa completa (separato e fuso), held-out paper invariato al passo 0")
def l5b():
    a1 = corri(NUOVO, "L5b_sep", ["--paper-campionato", "16"])
    a2 = corri(NUOVO, "L5b_fuso", ["--paper-campionato", "16", "--fuso"])
    r0 = json.load(open(TMP / "L2_rif" / "storia.json"))["storia"][0]
    for o in (a1, a2):
        s = json.load(open(o / "storia.json"))["storia"]
        assert s[0]["paper"] == r0["paper"], "held-out paper metric changed"
        assert all(math.isfinite(r["L"]) for r in s[1:])


# ───────────────────────────────────────── G: gradient checks ─────────────────────────────────────────
@prova("G1 gradcheck CKA (forma Gram n<d e forma feature n>d) e MSE relativa, fp64")
def g1():
    g = torch.Generator().manual_seed(3)
    for n, d in ((7, 11), (13, 5)):
        X = torch.randn(n, d, generator=g, dtype=torch.float64, requires_grad=True)
        Y = torch.randn(n, d, generator=g, dtype=torch.float64) + 0.5 * X.detach()
        assert torch.autograd.gradcheck(lambda x: AS.cka_lineare(x, Y), (X,), eps=1e-6, atol=1e-8, rtol=1e-6)
        assert torch.autograd.gradcheck(lambda x: AS.mse_relativa(x, Y), (X,), eps=1e-6, atol=1e-8, rtol=1e-6)
    X = torch.randn(9, 6, generator=g, dtype=torch.float64)
    Y = torch.randn(9, 6, generator=g, dtype=torch.float64)
    # the two CKA forms are the same number; CKA is invariant to rotation and isotropic scale of either side
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    feat = (Xc.T @ Yc).pow(2).sum() / ((Xc.T @ Xc).norm() * (Yc.T @ Yc).norm())
    assert abs(float(AS.cka_lineare(X, Y)) - float(feat)) < 1e-12
    Q, _ = torch.linalg.qr(torch.randn(6, 6, generator=g, dtype=torch.float64))
    assert abs(float(AS.cka_lineare(3.0 * X @ Q, Y)) - float(AS.cka_lineare(X, Y))) < 1e-12
    assert abs(float(AS.cka_lineare(X, X)) - 1.0) < 1e-12


@prova("G2 gradienti_allineamento == autograd della media sugli strati (con e senza maschera)")
def g2():
    g = torch.Generator().manual_seed(4)
    Sx = {L: torch.randn(10, 8, generator=g, dtype=torch.float64) for L in (0, 2, 5)}
    Tx = {L: torch.randn(10, 8, generator=g, dtype=torch.float64) for L in (0, 2, 5)}
    m = torch.ones(10, dtype=torch.bool); m[[0, 4]] = False
    for met in ("cka", "mse_rel"):
        for mask in (None, m):
            v, gr, _ = AS.gradienti_allineamento(Sx, Tx, met, mask)
            lv = {L: x.clone().requires_grad_(True) for L, x in Sx.items()}
            f = AS.METRICHE[met]
            tot = sum(f(lv[L] if mask is None else lv[L][mask], Tx[L] if mask is None else Tx[L][mask])
                      for L in lv) / len(lv)
            tot.backward()
            assert abs(float(tot) - v) < 1e-12
            for L in lv:   # float32 leaves inside gradienti_allineamento: compare at fp32 precision
                e = float((gr[L].double() - lv[L].grad).norm() / lv[L].grad.norm())
                assert e < 1e-5, (met, L, e)
            assert abs(AS.valore_allineamento(Sx, Tx, met, mask) - v) < 1e-6


def iniezione(env: dict, allenati, dense: bool) -> float:
    """Max relative error of the parameter gradients: (real Studente.avanti/indietro + Iniettore hooks) vs full autograd
    of L = <esce(xL), Rw> + w * L_al(outputs), on the synthetic student in float64."""
    vecchio = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        st = StudenteFinto(dtype=torch.float64)
        import cardan_kd_moduli as CK
        import residui_ternari as RT
        resE = CK.installa(st, [0, 1, 2], 3, None, torch.device("cpu"))
        par = CK.parametri(resE, "C") + CK.parametri(resE, "D")
        if dense:
            cfg = RT.ConfigResidui(rango=3, piani=1, strati=(0, 1, 2), famiglie=RT.famiglie_da("attn"), seme=1,
                                   continuo=True)
            resD = RT.installa(st, cfg, device=torch.device("cpu"))
            par += RT.parametri(resD)
        g = torch.Generator().manual_seed(8)
        with torch.no_grad():
            for p in par:
                p.add_(0.05 * torch.randn(p.shape, generator=g, dtype=p.dtype))
        for p in st.parametri():
            p.requires_grad_(False)
        for p in par:
            p.requires_grad_(True)
        st.strati_allenati = allenati
        ids = torch.randint(0, 38, (1, 24), generator=g)
        Rw = torch.randn(24, H, generator=g, dtype=torch.float64)
        n, Lam, w = 12, [0, 1, 2], 0.7
        Tm = {L: torch.randn(n, H, generator=g, dtype=torch.float64) for L in Lam}
        # reference: one graph through every layer (nascosto) with the outputs kept
        usc = {}
        orig = st._strato

        def reg(L, x, ctx, instr=None):
            y = orig(L, x, ctx, instr=instr); usc[L] = y; return y
        st._strato = reg
        h = st.nascosto(ids)
        st._strato = orig
        tot = (h * Rw).sum() + w * sum(AS.METRICHE["cka"](usc[L][0, :n], Tm[L]) for L in Lam) / len(Lam)
        rif = torch.autograd.grad(tot, par, allow_unused=True)
        # under test: checkpointed forward + Iniettore
        inj = AS.Iniettore(st, Lam, n)
        inj.registra = True; inj.attivo = env.get("E2E_SENZA_RICALCOLO") == "1"
        stato = st.avanti(ids, grafo=True)
        inj.registra = False
        xL = stato["xL"].detach().requires_grad_(True)
        with torch.enable_grad():
            ((st.esce(xL) * Rw).sum()).backward()
        _, gr, _ = AS.gradienti_allineamento({L: inj.uscite[L] for L in Lam}, Tm, "cka")
        inj.extra = {L: w * x.double() for L, x in gr.items()}
        inj.attivo, inj.colpi = True, 0
        for p in par:
            p.grad = None
        st.indietro(stato, xL.grad)
        assert inj.colpi == len(Lam), inj.colpi
        inj.togli()
        err = 0.0
        for p, r in zip(par, rif):
            if r is None or float(r.norm()) == 0:
                assert p.grad is None or float(p.grad.norm()) == 0
                continue
            err = max(err, float((p.grad - r).norm() / r.norm()))
        return err
    finally:
        for k, v in vecchio.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@prova("G3 gradiente iniettato nell'indietro VERO dello studente == autograd intero (tre percorsi, fp64)")
def g3():
    e = {}
    e["intero"] = iniezione({}, None, False)
    e["troncato"] = iniezione({}, [0, 1, 2], False)
    e["grafo_tenuto"] = iniezione({"E2E_SENZA_RICALCOLO": "1", "E2E_GRAFO_STRATI": "2"}, [0, 1, 2], False)
    e["troncato+densi"] = iniezione({}, [0, 1, 2], True)          # dense branch computes in fp32 by design
    for k, v in e.items():
        lim = 1e-5 if "densi" in k else 1e-10
        assert v < lim, (k, v)
    return " ".join(f"{k} {v:.1e}" for k, v in e.items())


@prova("G5 discesa: solo il termine di allineamento iniettato (uscita a gradiente 0) fa scendere L_al, CKA e MSE")
def g5():
    import cardan_kd_moduli as CK
    msg = []
    for met in ("cka", "mse_rel"):
        st = StudenteFinto(dtype=torch.float64)
        resE = CK.installa(st, [0, 1, 2], 3, None, torch.device("cpu"))
        par = CK.parametri(resE, "C") + CK.parametri(resE, "D")
        for p in st.parametri():
            p.requires_grad_(False)
        for p in par:
            p.requires_grad_(True)
        st.strati_allenati = [0, 1, 2]
        g = torch.Generator().manual_seed(21)
        ids = torch.randint(0, 38, (1, 24), generator=g)
        inj = AS.Iniettore(st, [0, 1, 2], 16)
        inj.registra = True
        st.avanti(ids)
        Tm = {L: inj.uscite[L] + 0.3 * torch.randn(16, H, generator=g, dtype=torch.float64) for L in (0, 1, 2)}
        opt = torch.optim.Adam(par, lr=3e-3)
        vals = []
        for _ in range(30):
            inj.uscite, inj.registra = {}, True
            stato = st.avanti(ids)
            inj.registra = False
            v, gr, _ = AS.gradienti_allineamento(dict(inj.uscite), Tm, met)
            vals.append(v)
            opt.zero_grad(set_to_none=True)
            inj.extra, inj.attivo = gr, True
            st.indietro(stato, torch.zeros_like(stato["xL"]))
            inj.attivo = False
            opt.step()
        # rank-3 corrections cannot fit the 0.3-noise target: require a clear, mostly monotone descent
        cali = sum(1 for x, y in zip(vals, vals[1:]) if y < x)
        assert vals[-1] < 0.95 * vals[0] and cali >= 0.8 * (len(vals) - 1), (met, vals[0], vals[-1], cali)
        msg.append(f"{met} {vals[0]:.4f}->{vals[-1]:.4f}")
    return "(" + ", ".join(msg) + ")"


def _rskd_setup(seed=11, P=6, Vv=30, Hh=5, tau=1.5):
    g = torch.Generator().manual_seed(seed)
    hS = torch.randn(P, Hh, generator=g, dtype=torch.float64)
    hT = torch.randn(P, Hh, generator=g, dtype=torch.float64)
    WS = torch.randn(Vv, Hh, generator=g, dtype=torch.float64)
    WT = torch.randn(Vv, Hh, generator=g, dtype=torch.float64) * 2
    tgt = torch.randint(0, Vv, (P,), generator=g)
    ok = torch.ones(P, dtype=torch.bool); ok[-1] = False
    fs = lambda h: h.double() @ WS.T
    ft = lambda h: h.double() @ WT.T
    fp = lambda q: q.double() @ WS
    return hS, hT, WS, WT, tgt, ok, tau, fs, ft, fp


@prova("G4 RS-KD: gradiente chiuso == autograd dell'obiettivo campionato; media -> gradiente esatto della KL piena")
def g4():
    hS, hT, WS, WT, tgt, ok, tau, fs, ft, fp = _rskd_setup()
    lam, K, P = 0.5, 7, hS.shape[0]
    # (a) one fixed draw: closed form vs autograd of the sampled objective (same samples)
    gen = torch.Generator().manual_seed(5)
    kd, ce, T, nv, gC = PC.perdita_paper_campionata(hS, WS, hT, WT, tgt, ok, tau, lam, K, 4, True, fs, ft, fp, gen)
    gen = torch.Generator().manual_seed(5)
    h = hS.clone().requires_grad_(True)
    tot = torch.zeros((), dtype=torch.float64)
    for j in range(0, P, 4):           # same chunking, same fp32 casts -> same multinomial draws on the generator
        zS, zT = fs(h[j:j + 4].float()).float(), ft(hT[j:j + 4]).float()
        lpT = torch.log_softmax(zT / tau, -1)
        idx = torch.multinomial(lpT.exp(), K, replacement=True, generator=gen)
        lpS = torch.log_softmax(zS / tau, -1)
        tot = tot + tau * tau * (lpT.gather(1, idx) - lpS.gather(1, idx)).sum() / K / T
        lp = torch.log_softmax(zS, -1)
        tot = tot - lam * (lp.gather(1, tgt[j:j + 4, None])[:, 0] * ok[j:j + 4]).sum() / nv
    tot.backward()
    ea = float((gC.double() - h.grad).norm() / h.grad.norm())
    assert ea < 1e-5, ea                      # fp32 arithmetic on both sides (the loss computes in fp32 by design)
    assert abs(kd / T + lam * ce / nv - float(tot)) < 1e-5 * abs(float(tot))
    # (b) unbiased: mean over draws -> exact gradient of tau^2 KL_full(p_T||p_S)/T + lam CE
    h = hS.clone().requires_grad_(True)
    lpT = torch.log_softmax(hT @ WT.T / tau, -1)
    lpS = torch.log_softmax(h @ WS.T / tau, -1)
    klf = (lpT.exp() * (lpT - lpS)).sum()
    lp = torch.log_softmax(h @ WS.T, -1)
    ex = tau * tau * klf / P - lam * (lp.gather(1, tgt[:, None])[:, 0] * ok).sum() / nv
    ex.backward()
    errs, vals = [], []
    acc = torch.zeros_like(hS)
    gen = torch.Generator().manual_seed(6)
    for r in range(1, 4001):
        kd, ce, T, nv, gC = PC.perdita_paper_campionata(hS, WS, hT, WT, tgt, ok, tau, lam, 16, 8, True, fs, ft, fp, gen)
        acc += gC; vals.append(kd)
        if r in (250, 4000):
            errs.append(float((acc / r - h.grad).norm() / h.grad.norm()))
    vmean = sum(vals) / len(vals)
    assert errs[1] < errs[0] and errs[1] < 0.01, errs
    assert abs(vmean - tau * tau * float(klf)) < 0.02 * tau * tau * float(klf), (vmean, tau * tau * float(klf))
    # (c) the top-k paper loss (flag off) is NOT this estimator: it is biased w.r.t. the full KL gradient
    _, _, _, _, gk = PP.perdita_paper(hS, WS, hT, WT, tgt, ok, tau, lam, 8, 8, True, fs, ft, fp)
    eb = float((gk - h.grad).norm() / h.grad.norm())
    return f"(campione fisso {ea:.1e}; media 250/4000 estrazioni {errs[0]:.3f}/{errs[1]:.4f}; top-8 vs piena {eb:.3f})"


# ───────────────────────────────────────── R: rotations ─────────────────────────────────────────
@prova("R1 Rotazioni: N=1 stesso oggetto; ciclo visita tutte le rotazioni in N epoche; tutte = N copie")
def r1():
    rd = TMP / "rot_r1"
    scrivi_rotazioni(rd, 4)
    pacchi, _ = pacchi_kl4(types.SimpleNamespace(calib_n=12, seme=7), print)
    rr = KR.Rotazioni.carica(pacchi, 4, "ciclo", rd, leggi_mclp, lambda *a: None)
    assert KR.Rotazioni(pacchi, 1, "ciclo", {}).pacco(3) is pacchi[2 % len(pacchi)]
    n = len(pacchi)
    viste = {}
    for passo in range(1, 4 * n + 1):
        for q in rr.pacco(passo):
            k = (len(q["ids"]) - len(rr.var[0][q["i"]]["ids"]))     # synthetic: rotation k appends one token k
            k = q["ids"][-1] if k == 1 else 0
            viste.setdefault(q["i"], []).append(k)
            assert q["label"] == (rr.var[0][q["i"]]["label"] - k) % 4
    assert all(sorted(v) == [0, 1, 2, 3] for v in viste.values()), viste
    rt = KR.Rotazioni.carica(pacchi, 4, "tutte", rd, leggi_mclp, lambda *a: None)
    assert len(rt.pacco(1)) == 4 * len(pacchi[0])


@prova("R2 prompt reale: scomponi/componi al byte sulle 1779 domande di SELRISP/calib, rotazione = permutazione ciclica")
def r2():
    B = KR.leggi_bin("@@ROOT:models@@/gguf/tony-tern/SELRISP/calib/calib_mmlu.bin")
    for testo, ans, lab in B:
        t, o = KR.scomponi(testo)
        assert KR.componi(t, o) == testo
        for k in range(4):
            ok_, yk = KR.ruota(o, lab.index(1), k)
            assert ok_[yk] == o[lab.index(1)]
    return f"({len(B)} domande)"


if __name__ == "__main__":
    for t in (f0, g1, g2, g3, g4, g5, r1, r2, f1, f2, f3, f4, f5, f6, f7, f8, l2a, l2b, l2c, l5a, l5b):
        t()
    ok = sum(1 for _, b, _ in RISULTATI if b)
    print(f"\n{ok}/{len(RISULTATI)} PASS (cartella {TMP})")
    if ok == len(RISULTATI):
        shutil.rmtree(TMP, ignore_errors=True)
    sys.exit(0 if ok == len(RISULTATI) else 1)

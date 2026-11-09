#!/usr/bin/env python3
"""27/9. Prove del motore «triton2» (kernel _k_gemm2 / _k_dx2 / _k_dd2 di e2e_studente) su CPU con
TRITON_INTERPRET=1 e famiglie VERE del GGUF (gate/up TQ1_0R_1024 gruppo 128, down TQ1_0 gruppo 256).

  K1  per famiglia: avanti e dX di triton2 == triton AL BIT (fp16, fp32; righe non multiple del riquadro,
      esperto vuoto, scale estreme, varianti di riquadro/GROUP_M); dL/dd entro l'arrotondamento fp32 (cambia
      solo l'ordine delle somme); tutti e tre contro un riferimento torch fp64 (decoder indipendente).
  K2  gate+up nella stessa passata: avanti_due == due avanti AL BIT; FamiglieGateUp (avanti e indietro)
      == due FamigliaEsperti AL BIT.
  K3  modo V (PV): MotoreTritonV su base triton2 == su base triton AL BIT nei modi 1/2/3 (dL/dd e buffer
      cand/prop/S/Q/H...); modo 0 = dL/dd di _k_dd2; `installa` propaga v2.
  K4  strato vero dello studente (ricetta B, pochi esperti): uscita e dx0 AL BIT, dtheta entro l'arrotondamento,
      gate+up fusi davvero usati.
  K5  motore «triton» IDENTICO al bit a quello del commit 2b4ca22 (prima di triton2, caricato a parte).
  K6  compilazione anticipata gfx1151 dei Blocchi di default: v_wmma > 0, 0 registri riversati (fp16).

Uso: TRITON_INTERPRET=1 HIP_VISIBLE_DEVICES= nice -n 15 python3 e2e_test_triton2.py [K1 K2 ...]
Esito JSON: $E2E_T2_ESITO (default: accanto a questo file, e2e_test_triton2_esito.json)."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

os.environ.setdefault("TRITON_INTERPRET", "1")
import torch  # noqa: E402

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import e2e_studente as S  # noqa: E402
import e2e_codici_scale as CS  # noqa: E402
import e2e_pv_codici as PV  # noqa: E402
from gguf import GGUFReader  # noqa: E402

torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))
ESITO: dict = {"data": time.strftime("%Y-%m-%d %H:%M:%S"), "gguf": str(S.GGUF_THOR_T), "prove": {}}
_TENS = None
TOL_DD = 2e-6          # dL/dd triton2 contro triton: solo l'ordine delle somme fp32 (visto ~2e-7)


def tens() -> dict:
    global _TENS
    if _TENS is None:
        _TENS = {t.name: t for t in GGUFReader(str(S.GGUF_THOR_T)).tensors}
    return _TENS


def registra(nome: str, ok: bool, **dati) -> bool:
    ESITO["prove"][nome] = {"ok": bool(ok), **dati}
    print(f"[{nome}] {'✅' if ok else '⛔'} {json.dumps(dati, ensure_ascii=False, default=str)[:2000]}", flush=True)
    return ok


def rel(a, b) -> float:
    a, b = a.double(), b.double()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-300))


def famiglia(L: int, n: str, esperti=None, righe=None) -> S.DatiFamiglia:
    return S.famiglia_da_tensore(tens()[f"blk.{L}.{S.FAMIGLIE[n]}.weight"], n, esperti=esperti, righe=righe)


def scale_estreme(d: torch.Tensor, seme: int = 0) -> torch.Tensor:
    """Come e2e_test_studente: zero, subnormali fp16, quasi zero e negative (valori fp16 esatti)."""
    g = torch.Generator().manual_seed(seme)
    d = d.clone()
    piatto = d.view(-1)
    scelte = torch.randperm(piatto.numel(), generator=g)[:64]
    speciali = torch.tensor([0.0, 2.0 ** -24, 3 * 2.0 ** -24, 1e-6, 6.2e-5, -2.0 ** -20, 1e-7, 2.0 ** -14])
    piatto[scelte] = speciali.repeat(8)
    return d.half().float()


def riferimento(x, gy, d, fam, piano):
    """fp64, decoder indipendente (e2e_codici_scale): (y, dx, dL/dd) con L = sum(y (.) gy)."""
    x = x.double().detach().requires_grad_(True)
    dd = d.double().detach().requires_grad_(True)
    o = piano.offs.tolist()
    ys = []
    for e in range(fam.E):
        q = CS.trit_da_blocchi(fam.blocchi[e].reshape(-1, fam.nb), fam.tipo).double()
        W = (q.view(fam.R, fam.NG, fam.gs) * dd[e].unsqueeze(-1)).view(fam.R, fam.C)
        ys.append(x[o[e]:o[e + 1]] @ W.T)
    y = torch.cat(ys)
    (y * gy.double()).sum().backward()
    return y.detach(), x.grad, dd.grad


def piano_casuale(E: int, T: int, k: int, g: torch.Generator, vuoto: bool = False) -> S.Piano:
    idx = torch.randint(1 if vuoto else 0, E, (T, k), generator=g)
    return S.Piano(idx, E)


def tre_prodotti(mot, x, gy, d, fam, piano, ric):
    return (mot.avanti(x, d, fam, piano, ric), mot.indietro_x(gy, d, fam, piano, ric),
            mot.dd(gy, x, fam, piano, torch.float32))


# ─────────────────────────────── K1 ───────────────────────────────
def prova_k1() -> bool:
    g = torch.Generator().manual_seed(1)
    ric = {torch.float16: S.RICETTE["B"], torch.float32: S.RICETTE["B32"]}
    casi = [("gate_L0_fp16_r0-80", 0, "gate", slice(0, 3), slice(0, 80), torch.float16, 60, False),
            ("gate_L39_fp16_r32-96_vuoto", 39, "gate", slice(5, 8), slice(32, 96), torch.float16, 60, True),
            ("up_L1_fp32_r0-80", 1, "up", slice(0, 3), slice(0, 80), torch.float32, 40, False),
            ("down_L0_fp16_r0-80", 0, "down", slice(0, 3), slice(0, 80), torch.float16, 60, True),
            ("down_L38_fp32_r0-72", 38, "down", slice(2, 5), slice(0, 72), torch.float32, 40, False)]
    # bf16 (ricetta A, down): l'interprete non ha bf16 (numpy) e da' numeri senza senso ANCHE per «triton»; la
    # via bf16 di triton2 si prova solo compilata (K6)
    ok, dati = True, {}
    vecchio, nuovo = S.motore("triton"), S.motore("triton2")
    for et, L, n, esp, righe, dt, T, vuoto in casi:
        t0 = time.time()
        fam = famiglia(L, n, esperti=esp, righe=righe)
        d = scale_estreme(fam.d0.half().float(), seme=L)
        piano = piano_casuale(fam.E, T, 2, g, vuoto)
        x = torch.randn(piano.S, fam.C, generator=g).to(dt)
        gy = (torch.randn(piano.S, fam.R, generator=g) * 1e-2).to(dt)
        a = tre_prodotti(vecchio, x, gy, d, fam, piano, ric[dt])
        b = tre_prodotti(nuovo, x, gy, d, fam, piano, ric[dt])
        y64, dx64, dd64 = riferimento(x, gy, d, fam, piano)
        r = {"avanti_bit": torch.equal(a[0], b[0]), "dx_bit": torch.equal(a[1], b[1]),
             "dd_bit": torch.equal(a[2], b[2]), "dd_rel_vs_triton": rel(b[2], a[2]),
             "fp64": {"avanti": [rel(a[0], y64), rel(b[0], y64)], "dx": [rel(a[1], dx64), rel(b[1], dx64)],
                      "dd": [rel(a[2], dd64), rel(b[2], dd64)]},
             "s": round(time.time() - t0, 1)}
        # contro fp64: triton2 non peggio di triton oltre l'arrotondamento dell'uscita
        tol = {torch.float16: 1e-3, torch.float32: 1e-5}[dt]
        u = r["avanti_bit"] and r["dx_bit"] and r["dd_rel_vs_triton"] < TOL_DD \
            and all(v[1] < tol for v in r["fp64"].values())
        r["ok"] = u
        ok &= u
        dati[et] = r
        print(et, r, flush=True)
    # varianti di riquadro (stessa famiglia, stessi dati): avanti e dX (BK_dx2 = 32) al bit con qualunque riquadro
    fam = famiglia(0, "gate", esperti=slice(0, 3), righe=slice(0, 80))
    d = fam.d0.half().float()
    piano = piano_casuale(fam.E, 200, 2, g)
    x = torch.randn(piano.S, fam.C, generator=g).half()
    gy = (torch.randn(piano.S, fam.R, generator=g) * 1e-2).half()
    rif = tre_prodotti(vecchio, x, gy, d, fam, piano, ric[torch.float16])
    varianti = {"BS2_32_BN2_32_G1": dict(BS2=32, BN2=32, GROUP_M=1),
                "BS2_128_BN2_128_G4": dict(BS2=128, BN2=128, GROUP_M=4),
                "BS2_32_BN2_64_G64": dict(BS2=32, BN2=64, GROUP_M=64),
                "dx_BS16_G2_dd_BR32_BS16": dict(BS_dx2=16, GROUP_M=2, BR_dd2=32, BS_dd2=16),
                "dx_BS64_dd_BR128_BS64_w4": dict(BS_dx2=64, BR_dd2=128, BS_dd2=64, num_warps2=4),
                "gemm_stadi1": dict(stadi_gemm2=1, stadi_dx2=2, stadi_dd2=2)}
    var = {}
    for k, v in varianti.items():
        b = tre_prodotti(S.motore("triton2", S.Blocchi(**v)), x, gy, d, fam, piano, ric[torch.float16])
        var[k] = {"avanti_bit": torch.equal(rif[0], b[0]), "dx_bit": torch.equal(rif[1], b[1]),
                  "dd_rel": rel(b[2], rif[2])}
        ok &= var[k]["avanti_bit"] and var[k]["dx_bit"] and var[k]["dd_rel"] < TOL_DD
    # BK_dx2 = 64: dX con altro ordine delle somme (solo arrotondamento, dichiarato)
    b = S.motore("triton2", S.Blocchi(BK_dx2=64)).indietro_x(gy, d, fam, piano, ric[torch.float16])
    var["dx_BK64_rel"] = rel(b, rif[1])
    ok &= var["dx_BK64_rel"] < 1e-3
    return registra("K1_famiglie", ok, casi=dati, varianti=var, tol_dd=TOL_DD)


# ─────────────────────────────── K2 ───────────────────────────────
def prova_k2() -> bool:
    g = torch.Generator().manual_seed(2)
    ok, dati = True, {}
    for L, dt, ricn in ((0, torch.float16, "B"), (39, torch.float32, "B32")):
        ric = S.RICETTE[ricn]
        fg = famiglia(L, "gate", esperti=slice(0, 3), righe=slice(0, 80))
        fu = famiglia(L, "up", esperti=slice(0, 3), righe=slice(0, 80))
        mot = S.motore("triton2")
        assert mot.puo_fondere(fg, fu, ric)
        piano = piano_casuale(3, 50, 2, g, vuoto=(L == 39))
        x0 = torch.randn(piano.S, fg.C, generator=g).to(dt)
        dg0, du0 = fg.d0.half().float(), fu.d0.half().float()
        Gg = torch.randn(piano.S, fg.R, generator=g) * 1e-2
        Gu = torch.randn(piano.S, fg.R, generator=g) * 1e-2
        gs, us = mot.avanti(x0, dg0, fg, piano, ric), mot.avanti(x0, du0, fu, piano, ric)
        g2, u2 = mot.avanti_due(x0, (dg0, du0), (fg, fu), piano, ric)
        r = {"avanti_due_bit": torch.equal(gs, g2) and torch.equal(us, u2)}
        ris = {}
        for fusa in (False, True):
            x = x0.clone().requires_grad_(True)
            dg, du = dg0.clone().requires_grad_(True), du0.clone().requires_grad_(True)
            if fusa:
                gg, uu = S.FamiglieGateUp.apply(x, dg, du, fg, fu, piano, ric, mot, None)
            else:
                gg = S.FamigliaEsperti.apply(x, dg, fg, piano, ric, mot, None)
                uu = S.FamigliaEsperti.apply(x, du, fu, piano, ric, mot, None)
            ((gg * Gg).sum() + (uu * Gu).sum()).backward()
            ris[fusa] = (gg.detach(), uu.detach(), x.grad, dg.grad, du.grad)
        r["autograd_bit"] = [torch.equal(a, b) for a, b in zip(ris[False], ris[True])]
        u = r["avanti_due_bit"] and all(r["autograd_bit"])
        ok &= u
        dati[f"L{L}_{ricn}"] = r
    return registra("K2_gate_up_insieme", ok, casi=dati, ordine="(g, u, dx, dd_gate, dd_up)")


# ─────────────────────────────── K3 ───────────────────────────────
def _tensori_buffer(b) -> dict:
    return {k: v for k, v in vars(b).items() if isinstance(v, torch.Tensor)}


def prova_k3() -> bool:
    g = torch.Generator().manual_seed(3)
    ok, dati = True, {}
    for et, L, n, esp, righe, dt in (("gate_L0_fp32", 0, "gate", slice(0, 3), slice(0, 80), torch.float32),
                                     ("down_L39_fp16", 39, "down", slice(4, 6), slice(0, 80), torch.float16)):
        fam = famiglia(L, n, esperti=esp, righe=righe)
        stati = {v2: PV.StatoV() for v2 in (False, True)}
        mots = {v2: PV.MotoreTritonV(stati[v2], v2=v2) for v2 in (False, True)}
        for v2 in (False, True):
            stati[v2].prepara([fam])
        dd_bit, dd0 = True, {}
        for k, modo in enumerate((1, 1, 2, 2, 2, 3, 0)):
            piano = piano_casuale(fam.E, 40, 2, g, vuoto=(k == 3))
            x = torch.randn(piano.S, fam.C, generator=g).to(dt)
            gy = (torch.randn(piano.S, fam.R, generator=g) * 1e-3).to(dt)
            out = {}
            for v2 in (False, True):
                stati[v2].modo = modo
                out[v2] = mots[v2].dd(gy, x, fam, piano, torch.float32)
            if modo == 0:
                blk = S.Blocchi()
                dd0 = {"v2_==_k_dd2": torch.equal(out[True], S._dd2_triton(gy, x, fam, piano, blk)),
                       "v1_==_k_dd": torch.equal(out[False], S._dd_triton(gy, x, fam, piano, blk)),
                       "rel": rel(out[True], out[False])}
            else:
                dd_bit &= torch.equal(out[False], out[True])
        ba, bb = _tensori_buffer(stati[False].buffer(fam)), _tensori_buffer(stati[True].buffer(fam))
        buf_bit = {k: torch.equal(ba[k], bb[k]) for k in ba}
        u = dd_bit and all(buf_bit.values()) and dd0["v2_==_k_dd2"] and dd0["v1_==_k_dd"] and dd0["rel"] < TOL_DD
        ok &= u
        dati[et] = {"dd_modi_123_bit": dd_bit, "buffer_bit": buf_bit, "modo0": dd0,
                    "nomi": [mots[False].nome, mots[True].nome]}
    # installa: il motore V eredita v2 (e i Blocchi) dalla base
    fam = famiglia(0, "down", esperti=slice(0, 2), righe=slice(0, 32))
    prop = {}
    for nome in ("triton", "triton2"):
        base = S.motore(nome)
        es = types.SimpleNamespace(mot=base, fam={"down": fam})
        st = types.SimpleNamespace(mot=base, strati=[0], esperti=lambda L, es=es: es)
        PV.installa(st, PV.StatoV())
        prop[nome] = {"v2": st.mot.v2, "nome": st.mot.nome, "blk_stesso": st.mot.blk is base.blk,
                      "strato": es.mot is st.mot}
    ok &= prop["triton2"]["v2"] and not prop["triton"]["v2"] and all(p["blk_stesso"] and p["strato"]
                                                                     for p in prop.values())
    return registra("K3_modo_V", ok, casi=dati, installa=prop)


# ─────────────────────────────── K4 ───────────────────────────────
def prova_k4() -> bool:
    torch.manual_seed(4)
    ids = torch.randint(0, 150000, (1, 6))
    G = torch.randn(6, 2048, generator=torch.Generator().manual_seed(4))
    chiamate = {"n": 0}
    orig = S.FamiglieGateUp.apply

    def conta(*a):
        chiamate["n"] += 1
        return orig(*a)
    ris = {}
    S.FamiglieGateUp.apply = conta
    try:
        for nome in ("triton", "triton2"):
            t0 = time.time()
            st = S.Studente(strati=[0], ricetta="B", motore_nome=nome, device="cpu", esperti=slice(0, 8),
                            log=lambda *_: None)
            x0 = st.embedding(ids)
            stato = st.avanti(ids, x0=x0)
            y = stato["xL"]
            xL = y.detach().requires_grad_(True)
            with torch.enable_grad():
                (st.esce(xL) * G).sum().backward()
            dx0 = st.indietro(stato, xL.grad)
            ris[nome] = (y.detach(), dx0.detach().clone(),
                         [p.grad.clone() for p in st.parametri()], round(time.time() - t0, 1))
            del st
    finally:
        S.FamiglieGateUp.apply = orig
    a, b = ris["triton"], ris["triton2"]
    dth = [rel(q, p) for q, p in zip(b[2], a[2]) if p.abs().max() > 0]
    r = {"uscita_bit": torch.equal(a[0], b[0]), "dx0_bit": torch.equal(a[1], b[1]), "dx0_rel": rel(b[1], a[1]),
         "dtheta_rel_max": max(dth), "n_parametri": len(dth), "chiamate_gate_up": chiamate["n"],
         "s": [a[3], b[3]]}
    ok = r["uscita_bit"] and r["dtheta_rel_max"] < 1e-5 and chiamate["n"] > 0 and r["dx0_rel"] < 1e-5
    return registra("K4_strato_studente", ok, **r)


# ─────────────────────────────── K5 ───────────────────────────────
def prova_k5() -> bool:
    """Il motore «triton» di oggi contro quello di HEAD (prima di triton2): stessi byte d'uscita."""
    rif = os.getenv("E2E_T2_RIF", "2b4ca22")   # ultimo commit prima di triton2
    src = subprocess.run(["git", "-C", str(QUI), "show", f"{rif}:thor_v2/e2e_studente.py"], capture_output=True,
                         text=True)
    if src.returncode != 0:
        return registra("K5_triton_invariato", False, errore=src.stderr[-400:])
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "e2e_studente_rif.py"
        p.write_text(src.stdout)
        spec = importlib.util.spec_from_file_location("e2e_studente_rif", p)
        R = importlib.util.module_from_spec(spec)
        sys.modules["e2e_studente_rif"] = R
        spec.loader.exec_module(R)
        g = torch.Generator().manual_seed(5)
        ok, dati = True, {}
        for et, L, n, dt, ricn in (("gate_L0_fp16", 0, "gate", torch.float16, "B"),
                                   ("down_L39_fp32", 39, "down", torch.float32, "B32")):
            fam = famiglia(L, n, esperti=slice(0, 3), righe=slice(0, 80))
            famR = R.famiglia_da_tensore(tens()[f"blk.{L}.{S.FAMIGLIE[n]}.weight"], n, esperti=slice(0, 3),
                                         righe=slice(0, 80))
            idx = torch.randint(0, 3, (50, 2), generator=g)
            piano, pianoR = S.Piano(idx, 3), R.Piano(idx, 3)
            d = fam.d0.half().float()
            x = torch.randn(piano.S, fam.C, generator=g).to(dt)
            gy = (torch.randn(piano.S, fam.R, generator=g) * 1e-2).to(dt)
            a = tre_prodotti(S.motore("triton"), x, gy, d, fam, piano, S.RICETTE[ricn])
            b = tre_prodotti(R.motore("triton"), x, gy, d, famR, pianoR, R.RICETTE[ricn])
            dati[et] = [torch.equal(u, v) for u, v in zip(a, b)]
            ok &= all(dati[et])
    return registra("K5_triton_invariato", ok, rif=rif, casi=dati, ordine="(avanti, dx, dd)")


# ─────────────────────────────── K6 ───────────────────────────────
_K6_FIGLIO = r'''
import json, re, sys
sys.path.insert(0, %r)
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
import e2e_studente as S
tgt = GPUTarget("hip", "gfx1151", 32)
b = S.Blocchi()
def info(fn, sig, cex, w, ns):
    k = triton.compile(ASTSource(fn=fn, signature=sig, constexprs=cex), target=tgt,
                       options={"num_warps": w, **({"num_stages": ns} if ns > 0 else {})})
    a = k.asm["amdgcn"]
    g = lambda p: int(re.findall(p, a)[0])
    return {"v_wmma": len(re.findall(r"v_wmma", a)), "spill": g(r"vgpr_spill_count:\s+(\d+)"),
            "vgpr": g(r"\.vgpr_count:\s+(\d+)")}
def sg(dt):
    return {"a_ptr": f"*{dt}", "sa_s": "i32", "pb": "*u8", "sb_e": "i32", "NBR": "i32", "d_ptr": "*fp32",
            "sd_e": "i32", "NG": "i32", "o_ptr": "*fp32", "so_s": "i32", "pb2": "*u8", "sb2_e": "i32",
            "d2_ptr": "*fp32", "sd2_e": "i32", "o2_ptr": "*fp32", "te_ptr": "*i32", "ts_ptr": "*i32",
            "offs_ptr": "*i64", "R": "i32", "C": "i32", "NT": "i32", "GS": "constexpr", "QK": "constexpr",
            "NB": "constexpr", "ROT": "constexpr", "DUE": "constexpr", "BS": "constexpr", "BN": "constexpr",
            "GROUP_M": "constexpr"}
def sx(dt):
    return {"a_ptr": f"*{dt}", "sa_s": "i32", "pb": "*u8", "sb_e": "i32", "NBR": "i32", "d_ptr": "*fp32",
            "sd_e": "i32", "NG": "i32", "o_ptr": f"*{dt}", "so_s": "i32", "te_ptr": "*i32", "ts_ptr": "*i32",
            "offs_ptr": "*i64", "R": "i32", "C": "i32", "NT": "i32", "GS": "constexpr", "QK": "constexpr",
            "NB": "constexpr", "ROT": "constexpr", "BS": "constexpr", "BK": "constexpr", "GROUP_M": "constexpr"}
def sd(dt):
    return {"dy_ptr": f"*{dt}", "sdy_s": "i32", "x_ptr": f"*{dt}", "sx_s": "i32", "pb": "*u8", "sb_e": "i32",
            "NBR": "i32", "dd_ptr": "*fp32", "sdd_e": "i32", "NG": "i32", "offs_ptr": "*i64", "R": "i32",
            "C": "i32", "GS": "constexpr", "QK": "constexpr", "NB": "constexpr", "ROT": "constexpr",
            "BR": "constexpr", "BS": "constexpr"}
out = {}
for nome, rot, gs, qk, nb in (("R1024", 1, 128, 1024, 224), ("TQ1_0", 0, 256, 256, 54)):
    base = {"GS": gs, "QK": qk, "NB": nb, "ROT": rot}
    for due in (0, 1):
        out[f"gemm2_{nome}_due{due}"] = info(S._k_gemm2, sg("fp16"), {**base, "DUE": due, "BS": b.BS2,
                                             "BN": b.BN2, "GROUP_M": b.GROUP_M}, b.num_warps2, b.stadi_gemm2)
    out[f"dx2_{nome}"] = info(S._k_dx2, sx("fp16"), {**base, "BS": b.BS_dx2, "BK": b.BK_dx2,
                              "GROUP_M": b.GROUP_M}, b.num_warps2, b.stadi_dx2)
    out[f"dd2_{nome}"] = info(S._k_dd2, sd("fp16"), {**base, "BR": b.BR_dd2, "BS": b.BS_dd2},
                              b.num_warps2, b.stadi_dd2)
    if not rot:   # ricetta A: il down (non ruotato) passa da triton2 con ingressi bf16
        out[f"bf16_gemm2_{nome}"] = info(S._k_gemm2, sg("bf16"), {**base, "DUE": 0, "BS": b.BS2, "BN": b.BN2,
                                         "GROUP_M": b.GROUP_M}, b.num_warps2, b.stadi_gemm2)
        out[f"bf16_dx2_{nome}"] = info(S._k_dx2, sx("bf16"), {**base, "BS": b.BS_dx2, "BK": b.BK_dx2,
                                       "GROUP_M": b.GROUP_M}, b.num_warps2, b.stadi_dx2)
        out[f"bf16_dd2_{nome}"] = info(S._k_dd2, sd("bf16"), {**base, "BR": b.BR_dd2, "BS": b.BS_dd2},
                                       b.num_warps2, b.stadi_dd2)
print(json.dumps(out))
''' % str(QUI)


def prova_k6() -> bool:
    r = subprocess.run([sys.executable, "-c", _K6_FIGLIO], capture_output=True, text=True,
                       env={**os.environ, "TRITON_INTERPRET": "0"})
    if r.returncode != 0:
        return registra("K6_compilazione_gfx1151", False, errore=r.stderr[-800:])
    out = json.loads(r.stdout.strip().splitlines()[-1])
    ok = all(v["v_wmma"] > 0 and (v["spill"] == 0 or k.startswith("bf16")) for k, v in out.items())
    return registra("K6_compilazione_gfx1151", ok, kernel=out, nota="bf16 (ricetta A): basta che compili, spill riportati")


PROVE = {"K1": prova_k1, "K2": prova_k2, "K3": prova_k3, "K4": prova_k4, "K5": prova_k5, "K6": prova_k6}


def main() -> int:
    scelte = sys.argv[1:] or list(PROVE)
    ok = True
    for k in scelte:
        t0 = time.time()
        ok &= PROVE[k]()
        print(f"  ({k}: {time.time() - t0:.0f} s)", flush=True)
    ESITO["ok"] = bool(ok)
    dest = Path(os.getenv("E2E_T2_ESITO", str(QUI / "e2e_test_triton2_esito.json")))
    dest.write_text(json.dumps(ESITO, indent=1, ensure_ascii=False, default=str))
    print(f"ESITO {'✅' if ok else '⛔'} -> {dest}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

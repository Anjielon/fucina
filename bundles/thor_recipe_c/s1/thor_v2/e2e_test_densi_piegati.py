#!/usr/bin/env python3
"""Prove dei densi PIEGATI (e2e_densi_piegati, 1/10/2026, ramo densi-piegati). SOLO CPU (la GPU e' di TT5), <= 4 fili.

GGUF R2B3m (la configurazione di TT5: densi a TRE piani), byte VERI. La «via di oggi» e' quella del codice in servizio:
n `FamigliaEsperti` sullo stesso ingresso sommati nell'ordine dei piani, gradiente di x accumulato da autograd (in fp16),
ricetta B3_ind32 (indietro fp32). Riferimento esatto: fp64 col decoder INDIPENDENTE (e2e_codici_scale.trit_da_blocchi).

  P0  interruttore spento: nessuna chiamata piegata, e `ammesso` rifiuta i casi fuori campo (1 piano, B64, ingresso fp32).
  P1  famiglie vere a 64 righe (attn_qkv TQ1_0R_1024 ruotata, ffn_down_shexp TQ1_0, ssm_out), scale casuali, dY normale e
      «estremo» (righe da 1e-30 a 1e4), modi x2 e f32, motori torch e triton2 (interprete): avanti, dX, dL/dd di ogni
      piano contro la via di oggi DENTRO i limiti deterministici dichiarati (elemento per elemento) e contro fp64.
  P2  passo V (modi 1, 2, 3) coi motori torch+V e triton2+V: cand, prop, S, Q, H dei tre piani contro la via di oggi.
  P3  forma VERA intera (attn_qkv 8192 x 2048, ssm_out 2048 x 4096, ffn_down_shexp 2048 x 512; 3 piani; 256 token):
      stessi controlli di P1 (errori, limiti), tempo avanti+indietro oggi contro piegato (CPU, 4 fili), memoria (picco
      RSS misurato con clear_refs + conto dei transitori).
  P4  strato 0 VERO dello studente con --densi-ternari tutte (R2B3m, 8 esperti): spento / x2 / f32 / B64: avanti, dtheta
      per piano (1, 2, 3) e degli esperti; tutte le LineareTernaria a 3 piani passano dalla via piegata.
  P6  cache dei trit: dopo un salto del passo V (byte in place) il trit si rilegge, prima si riusa.
  P7  (2/10) grandezze da GPU (dY x 2^19, canali anomali) per la via della GPU (decodifica Triton + _k_mm16) con B3_ind2.
  P8  (2/10) amplificazione su 4 strati veri: piegato contro oggi come un riordino delle somme; contro fp64 non peggio.
  P5  kernel `_k_mm16` (interprete) contro matmul fp32, anche su viste trasposte; trit da `decodifica_pesi` (d = 1) ==
      `trit_esperto`.

Uso:
    systemd-run --user --scope --quiet -p CPUQuota=400% -p MemoryMax=16G nice -n 19 ionice -c3 \\
      env OMP_NUM_THREADS=4 @@ROOT:venv@@/bin/python3 e2e_test_densi_piegati.py [P0 P1 ...]
    esito: $E2E_DP_ESITO (default @@ROOT:depot@@/e2e/prove_tt5/densi_piegati/esito_test_densi_piegati.json)
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

if os.getenv("E2E_DP_GPU") != "1":          # importato dal banco GPU (E2E_DP_GPU=1): niente interprete, GPU visibile
    os.environ.setdefault("TRITON_INTERPRET", "1")
    for _v in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        os.environ[_v] = ""
os.environ.setdefault("E2E_DENSI_PIEGATI", "")
import torch  # noqa: E402

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import e2e_studente as S  # noqa: E402
import e2e_codici_scale as CS  # noqa: E402
import e2e_pv_codici as PV  # noqa: E402
import e2e_densi_piegati as DP  # noqa: E402
from gguf import GGUFReader  # noqa: E402

torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))
R2B3M = Path(os.getenv("E2E_R2B3M", "@@ROOT:models@@/gguf/tony-tern/R2B3m/Thor-T-R2B3m.gguf"))
ESITO: dict = {"data": time.strftime("%Y-%m-%d %H:%M:%S"), "gguf": str(R2B3M), "prove": {}}
U = 2.0 ** -24
U16 = 2.0 ** -11
_TENS = None


def gamma(n: int) -> float:
    return n * U / (1 - n * U)


def tens() -> dict:
    global _TENS
    if _TENS is None:
        _TENS = {t.name: t for t in GGUFReader(str(R2B3M)).tensors}
    return _TENS


def registra(nome: str, ok: bool, **dati) -> bool:
    ESITO["prove"][nome] = {"ok": bool(ok), **dati}
    print(f"[{nome}] {'OK' if ok else 'ROSSO'} {json.dumps(dati, ensure_ascii=False, default=str)[:4000]}", flush=True)
    return ok


def piani(L: int, base: str, righe=slice(0, 64)) -> list:
    out = []
    for suf in ("", S.SUFFISSO_P2, S.SUFFISSO_P3):
        n = f"{base}{suf}"
        out.append(S.famiglia_da_tensore(tens()[f"blk.{L}.{S.FAMIGLIE_DENSE[base]}{suf}.weight"], n, righe=righe))
    return out


def scale(fams, g) -> list:
    return [(f.d0 * torch.exp(0.01 * torch.randn(f.d0.shape, generator=g))).half().float() for f in fams]


def gy_estremo(T: int, R: int, g) -> torch.Tensor:
    base = torch.randn(T, R, generator=g)
    v = base * torch.pow(10.0, torch.empty(T, 1).uniform_(-30, 4, generator=g))
    v[torch.rand(T, R, generator=g) < 0.1] *= 1e-6
    v[torch.rand(T, R, generator=g) < 0.05] = 0.0
    return v.float()


def via_oggi(xg, ds, fams, ric, mot, gy):
    """Il codice in servizio: n FamigliaEsperti sommati (ordine dei piani), autograd accumula dx in fp16."""
    x = xg.clone().requires_grad_(True)
    dl = [d.clone().requires_grad_(True) for d in ds]
    piano = S.piano_unico(x.shape[0], x.device)
    y = None
    for f, d in zip(fams, dl):
        yk = S.FamigliaEsperti.apply(x, d, f, piano, ric, mot, None)
        y = yk if y is None else y + yk
    y.backward(gy)
    return y.detach(), x.grad, [d.grad for d in dl]


def via_piegata(xg, ds, fams, ric, mot, gy, modo):
    x = xg.clone().requires_grad_(True)
    dl = [d.clone().requires_grad_(True) for d in ds]
    y = DP.DensoPiegato.apply(x, tuple(fams), ric, mot, None, modo, *dl)
    y.backward(gy)
    return y.detach(), x.grad, [d.grad for d in dl]


def esatto(xg, ds, fams, gy):
    """fp64 col decoder indipendente: Y, dX, dd_k, e i pesi |W_k| per i limiti."""
    x64, g64 = xg.double(), gy.double()
    Ts = [CS.trit_da_blocchi(f.blocchi[0].reshape(-1, f.nb), f.tipo).double().view(f.R, f.C) for f in fams]
    gs, R, C = fams[0].gs, fams[0].R, fams[0].C
    Ws = [(t.view(R, C // gs, gs) * d[0].double().unsqueeze(-1)).view(R, C) for t, d in zip(Ts, ds)]
    W = sum(Ws)
    G = g64.T @ x64
    dd = [(t.view(R, C // gs, gs) * G.view(R, C // gs, gs)).sum(-1) for t in Ts]
    Wabs = sum(w.abs() for w in Ws)
    return {"Y": x64 @ W.T, "dX": g64 @ W, "dd": dd, "dxk_abs": sum((g64 @ w).abs() for w in Ws), "Wabs": Wabs, "Ts": Ts}


def err_norma(a, b) -> float:
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


def controlla(xg, ds, fams, gy, oggi, pieg, modo) -> dict:
    """Errori contro fp64 e contro oggi + verifica elemento per elemento dei limiti deterministici dichiarati."""
    ex = esatto(xg, ds, fams, gy)
    x64a, g64a = xg.double().abs(), gy.double().abs()
    T, R, C, gs = xg.shape[0], fams[0].R, fams[0].C, fams[0].gs
    # avanti
    BY = x64a @ ex["Wabs"].T
    limY = (2 * gamma(C + 2) + (2.0 ** -22 if modo == "x2" else 2 * U)) * BY
    vY = float(((pieg[0].double() - oggi[0].double()).abs() - limY).clamp_min(0).max())
    # dL/dd
    Bdd = (g64a.T @ x64a).view(R, C // gs, gs).sum(-1)
    Bbar = g64a.amax(0)[:, None] * x64a.sum(0).view(C // gs, gs).sum(-1)[None, :]
    limdd = 2 * gamma(T + gs) * Bdd + (2.0 ** -23 * Bbar if modo == "x2" else 0)
    vdd = max(float(((p[0].double() - o[0].double()).abs() - limdd).clamp_min(0).max())
              for p, o in zip(pieg[2], oggi[2]))
    # dX (fp16): domina l'arrotondamento fp16 di oggi (3 arrotondamenti + 2 somme fp16)
    BdX = g64a @ ex["Wabs"]
    Bbar_dx = g64a.amax(1, keepdim=True) * ex["Wabs"].sum(0)[None, :]
    limdx = 5 * U16 * (ex["dxk_abs"] + ex["dX"].abs()) + 5 * 2.0 ** -25 + 2 * gamma(R) * BdX \
        + (2.0 ** -21 * Bbar_dx if modo == "x2" else 0)
    fin = torch.isfinite(oggi[1].double()) & torch.isfinite(pieg[1].double())
    vdx = float(((pieg[1].double() - oggi[1].double()).abs() - limdx)[fin].clamp_min(0).max())
    r = {"Y_piegato_vs_fp64": err_norma(pieg[0], ex["Y"]), "Y_oggi_vs_fp64": err_norma(oggi[0], ex["Y"]),
         "dX_piegato_vs_fp64": err_norma(pieg[1][fin], ex["dX"][fin]),
         "dX_oggi_vs_fp64": err_norma(oggi[1][fin], ex["dX"][fin]),
         "dX_quota_diversi": float((pieg[1] != oggi[1]).float().mean()),
         "dX_nonfiniti_uguali": bool(torch.equal(torch.isfinite(pieg[1]), torch.isfinite(oggi[1]))),
         "dd_piegato_vs_fp64": [err_norma(p, e.unsqueeze(0)) for p, e in zip(pieg[2], ex["dd"])],
         "dd_oggi_vs_fp64": [err_norma(o, e.unsqueeze(0)) for o, e in zip(oggi[2], ex["dd"])],
         "dd_piegato_vs_oggi": [err_norma(p, o) for p, o in zip(pieg[2], oggi[2])],
         "dd_max_rel_su_B": max(float(((p[0].double() - o[0].double()).abs() / Bdd.clamp_min(1e-300)).max())
                                for p, o in zip(pieg[2], oggi[2])),
         "Y_max_rel_su_B": float(((pieg[0].double() - oggi[0].double()).abs() / BY.clamp_min(1e-300)).max()),
         "eccesso_limite": {"Y": vY, "dd": vdd, "dX": vdx}}
    r["dentro_i_limiti"] = vY == 0 and vdd == 0 and vdx == 0
    r["dtype_uguali"] = (pieg[0].dtype == oggi[0].dtype and pieg[1].dtype == oggi[1].dtype
                         and all(p.dtype == o.dtype for p, o in zip(pieg[2], oggi[2])))
    return r


def accettabile(r: dict) -> bool:
    """Limiti deterministici + controllo empirico stretto: il piegato non e' peggiore di oggi contro fp64 (salvo
    2^-20 assoluto relativo alla norma, livello fp32)."""
    ok = r["dentro_i_limiti"] and r["dtype_uguali"] and r["dX_nonfiniti_uguali"]
    ok &= r["Y_piegato_vs_fp64"] <= 2 * r["Y_oggi_vs_fp64"] + 2.0 ** -20
    ok &= all(p <= 2 * o + 2.0 ** -20 for p, o in zip(r["dd_piegato_vs_fp64"], r["dd_oggi_vs_fp64"]))
    ok &= r["dX_piegato_vs_fp64"] <= 1.01 * r["dX_oggi_vs_fp64"] + 1e-6
    return ok


# ─────────────────────────────── P0 ───────────────────────────────
def prova_p0() -> bool:
    DP.imposta(modo="")
    ok = not DP.attivo() and os.getenv("E2E_DENSI_PIEGATI", "") == ""
    fams = piani(0, "attn_qkv")
    xg = torch.zeros(4, fams[0].C, dtype=torch.float16)
    mot = S.motore("torch")
    casi = {"tre_piani_B3_ind32": DP.ammesso(fams, S.RICETTE["B3_ind32"], mot, xg),
            "un_piano": DP.ammesso(fams[:1], S.RICETTE["B3_ind32"], mot, xg),
            "B64": DP.ammesso(fams, S.RICETTE["B64"], mot, xg.double()),
            "ingresso_fp32": DP.ammesso(fams, S.RICETTE["B3_ind32"], mot, xg.float()),
            "ricetta_A_ruotata": DP.ammesso(fams, S.RICETTE["A"], mot, xg)}
    ok &= casi == {"tre_piani_B3_ind32": True, "un_piano": False, "B64": False, "ingresso_fp32": False,
                   "ricetta_A_ruotata": False}
    return registra("P0_spento_e_campo", ok, **casi)


# ─────────────────────────────── P1 ───────────────────────────────
def prova_p1() -> bool:
    ok, dati = True, {}
    ric = S.RICETTE["B3_ind32"]
    for L, base in ((0, "attn_qkv"), (0, "ffn_down_shexp"), (0, "ssm_out")):
        fams = piani(L, base)
        for seme in (1, 2):
            g = torch.Generator().manual_seed(100 * seme + len(base))
            T = 50
            ds = scale(fams, g)
            xg = (torch.randn(T, fams[0].C, generator=g) * 0.5).half()
            if base != "ffn_down_shexp":
                xg = S.ruota(xg.float()).half()
            for tipo_gy in ("normale", "estremo"):
                gy = torch.randn(T, fams[0].R, generator=g) if tipo_gy == "normale" else gy_estremo(T, fams[0].R, g)
                for mot_n in ("torch", "triton2"):
                    mot = S.motore(mot_n)
                    oggi = via_oggi(xg, ds, fams, ric, mot, gy)
                    for modo in ("x2", "f32"):
                        r = controlla(xg, ds, fams, gy, oggi, via_piegata(xg, ds, fams, ric, mot, gy, modo), modo)
                        dati[f"{base}_s{seme}_{tipo_gy}_{mot_n}_{modo}"] = r
                        ok &= accettabile(r)
    return registra("P1_famiglie_vere_64_righe", ok, tipi={b: piani(0, b)[0].tipo for b in
                                                           ("attn_qkv", "ffn_down_shexp", "ssm_out")}, **dati)


# ─────────────────────────────── P2 ───────────────────────────────
def prova_p2() -> bool:
    ok, dati = True, {}
    ric = S.RICETTE["B3_ind32"]
    for base in ("attn_qkv", "ffn_down_shexp"):
        fams = piani(0, base)
        for mot_n in ("torch", "triton2"):
            for modo_p in ("x2", "f32"):
                sto, stp = PV.StatoV(), PV.StatoV()
                if mot_n == "torch":
                    mo, mp = PV.MotoreTorchV(sto), PV.MotoreTorchV(stp)
                else:
                    mo, mp = PV.MotoreTritonV(sto, v2=True), PV.MotoreTritonV(stp, v2=True)
                sto.prepara(fams)
                stp.prepara(fams)
                g = torch.Generator().manual_seed(7)
                ds = scale(fams, g)
                e_dd = 0.0
                for modo in (1, 1, 2, 2, 2, 3, 3):
                    T = 60
                    xg = (torch.randn(T, fams[0].C, generator=g) * 0.5).half()
                    gy = torch.randn(T, fams[0].R, generator=g) * 1e-3
                    sto.modo = stp.modo = modo
                    o = via_oggi(xg, ds, fams, ric, mo, gy)
                    p = via_piegata(xg, ds, fams, ric, mp, gy, modo_p)
                    if modo != 3:
                        e_dd = max(e_dd, max(err_norma(a, b) for a, b in zip(p[2], o[2])))
                    else:
                        ok &= all(float(a.abs().max()) == 0 and float(b.abs().max()) == 0 for a, b in zip(p[2], o[2]))
                r = {"dd_max_vs_oggi": e_dd}
                for f in fams:
                    bo, bp = sto.buffer(f), stp.buffer(f)
                    same = bo.cand == bp.cand
                    r[f.nome] = {"cand_uguali": float(same.float().mean()),
                                 "prop": err_norma(bp.prop[same], bo.prop[same]),
                                 "S": err_norma(bp.S, bo.S), "Q": err_norma(bp.Q, bo.Q), "H": err_norma(bp.H, bo.H)}
                    ok &= r[f.nome]["cand_uguali"] >= 0.999 and r[f.nome]["prop"] <= 2e-6 and r[f.nome]["S"] <= 2e-6
                    ok &= r[f.nome]["Q"] <= 4e-6 and r[f.nome]["H"] <= 2e-6
                ok &= e_dd <= 2e-6
                dati[f"{base}_{mot_n}_{modo_p}"] = r
    return registra("P2_passo_V_modi_1_2_3", ok, **dati)


# ─────────────────────────────── P3 ───────────────────────────────
def _azzera_picco() -> None:
    try:
        Path("/proc/self/clear_refs").write_text("5")
    except OSError:
        pass


def _picco_rss() -> int:
    for riga in Path("/proc/self/status").read_text().splitlines():
        if riga.startswith("VmHWM:"):
            return int(riga.split()[1]) * 1024
    return -1


def _rss() -> int:
    for riga in Path("/proc/self/status").read_text().splitlines():
        if riga.startswith("VmRSS:"):
            return int(riga.split()[1]) * 1024
    return -1


def prova_p3() -> bool:
    ok, dati = True, {}
    ric = S.RICETTE["B3_ind32"]
    mot = S.motore("torch")
    T = int(os.getenv("E2E_DP_TOKEN", "256"))
    for base in ("attn_qkv", "ssm_out", "ffn_down_shexp"):
        fams = piani(0, base, righe=None)
        g = torch.Generator().manual_seed(33)
        ds = scale(fams, g)
        xg = (torch.randn(T, fams[0].C, generator=g) * 0.5).half()
        if fams[0].ruotata:
            xg = S.ruota(xg.float()).half()
        gy = torch.randn(T, fams[0].R, generator=g)
        r = {"R": fams[0].R, "C": fams[0].C, "gs": fams[0].gs, "tipo": fams[0].tipo, "token": T}
        tempi = {}
        for nome, f in (("oggi", lambda: via_oggi(xg, ds, fams, ric, mot, gy)),
                        ("x2", lambda: via_piegata(xg, ds, fams, ric, mot, gy, "x2")),
                        ("f32", lambda: via_piegata(xg, ds, fams, ric, mot, gy, "f32"))):
            f()                                                     # riscaldamento
            base_rss = _rss()
            _azzera_picco()
            t0 = time.perf_counter()
            out = f()
            tempi[nome] = time.perf_counter() - t0
            r[f"picco_rss_sopra_base_MiB_{nome}"] = round((_picco_rss() - base_rss) / 2 ** 20, 1)
            if nome == "oggi":
                oggi = out
            else:
                c = controlla(xg, ds, fams, gy, oggi, out, nome)
                r[f"errori_{nome}"] = c
                ok &= accettabile(c)
        r["secondi_cpu"] = {k: round(v, 3) for k, v in tempi.items()}
        r["accelerazione_cpu"] = {k: round(tempi["oggi"] / tempi[k], 2) for k in ("x2", "f32")}
        r["transitori_conto_MiB"] = {k: round(v / 2 ** 20, 1) for k, v in
                                     DP.byte_transitori(fams[0].R, fams[0].C, T, "x2").items()}
        dati[base] = r
    return registra("P3_forme_vere_intere", ok, **dati)


# ─────────────────────────────── P4 ───────────────────────────────
def _calcola(ric: str, modo: str, T: int = 48):
    DP.imposta(modo=modo)
    DP.CONTI.update(piegati=0, ripiego=0)
    st = S.Studente(gguf=R2B3M, strati=[0], ricetta=ric, motore_nome="torch", device="cpu", esperti=slice(0, 8),
                    log=lambda *_: None, densi_ternari=tuple(S.FAMIGLIE_DENSE))
    with torch.no_grad():
        for (L, n), p in zip(st.chiavi(), st.parametri()):
            g = torch.Generator().manual_seed(1000 * L + sum(ord(c) for c in n))
            p.copy_(0.01 * torch.randn(p.shape, generator=g))
    g = torch.Generator().manual_seed(110)
    x0 = torch.randn(1, T, 2048, generator=g) * 0.5
    G = torch.randn(T, 2048, generator=g)
    ids = torch.zeros(1, T, dtype=torch.long)
    x0 = x0.to(st.ric.dtype_residuo)
    with torch.no_grad():
        xL = st.avanti(ids, x0=x0)["xL"].clone()
    st.gradiente(ids, lambda h: (h * G.to(h.dtype)).sum() / T, x0=x0)
    lt = [m for m in st.moduli[0].modules() if isinstance(m, S.LineareTernaria)]
    out = {"xL": xL, "grad": {n: p.grad.detach().clone() for (L, n), p in zip(st.chiavi(), st.parametri())},
           "tre_piani": sum(1 for m in lt if m.nome2 is not None and m.nome3 is not None), "conti": dict(DP.CONTI)}
    del st
    DP.imposta(modo="")
    return out


def prova_p4() -> bool:
    ris = {"B64": _calcola("B64", ""), "oggi": _calcola("B3_ind32", ""), "x2": _calcola("B3_ind32", "x2"),
           "f32": _calcola("B3_ind32", "f32")}
    dati = {"lineari_a_tre_piani": ris["oggi"]["tre_piani"], "conti": {k: v["conti"] for k, v in ris.items()}}
    ok = dati["lineari_a_tre_piani"] > 0 and ris["oggi"]["conti"]["piegati"] == 0
    for m in ("x2", "f32"):
        ok &= ris[m]["conti"]["piegati"] > 0 and ris[m]["conti"]["ripiego"] == 0

    def gruppo(n: str) -> str:
        if n in S.FAMIGLIE_DENSE3:
            return "densi_piano3"
        if n in S.FAMIGLIE_DENSE2:
            return "densi_piano2"
        if n in S.FAMIGLIE_DENSE:
            return "densi_piano1"
        return "esperti_e_altro"

    def dth(a: dict, b: dict) -> dict:
        out = {}
        for gr in ("densi_piano1", "densi_piano2", "densi_piano3", "esperti_e_altro"):
            ks = [k for k in b if gruppo(k) == gr]
            num = sum(float((a[k].double() - b[k].double()).pow(2).sum()) for k in ks)
            den = sum(float(b[k].double().pow(2).sum()) for k in ks)
            segni = sum(int(((torch.sign(a[k]) != torch.sign(b[k])) & (b[k] != 0)).sum()) for k in ks)
            vivi = sum(int((b[k] != 0).sum()) for k in ks)
            out[gr] = {"famiglie": len(ks), "err_rel_norma": (num / max(den, 1e-300)) ** 0.5,
                       "quota_segni_opposti": segni / max(vivi, 1)}
        return out
    g64 = ris["B64"]["grad"]
    dati["oggi_vs_B64"] = dth(ris["oggi"]["grad"], g64)
    for m in ("x2", "f32"):
        dati[f"avanti_{m}_vs_oggi"] = err_norma(ris[m]["xL"], ris["oggi"]["xL"])
        dati[f"avanti_{m}_vs_B64"] = err_norma(ris[m]["xL"], ris["B64"]["xL"])
        dati[f"{m}_vs_B64"] = dth(ris[m]["grad"], g64)
        dati[f"{m}_vs_oggi"] = dth(ris[m]["grad"], ris["oggi"]["grad"])
        ok &= dati[f"avanti_{m}_vs_oggi"] <= 1e-5
        for gr, v in dati[f"{m}_vs_oggi"].items():
            e = dati["oggi_vs_B64"][gr]["err_rel_norma"]
            ok &= v["famiglie"] > 0 and v["err_rel_norma"] <= 1e-3 and v["quota_segni_opposti"] <= 1e-3
            ok &= dati[f"{m}_vs_B64"][gr]["err_rel_norma"] <= max(2 * e, 1e-5)
    dati["avanti_oggi_vs_B64"] = err_norma(ris["oggi"]["xL"], ris["B64"]["xL"])
    return registra("P4_strato_vero_studente", ok, **dati)


# ─────────────────────────────── P5 ───────────────────────────────
def prova_p5() -> bool:
    ok, dati = True, {}
    g = torch.Generator().manual_seed(5)
    for (M, K, N) in ((37, 70, 45), (64, 256, 64), (130, 96, 33)):
        a = (torch.randn(M, K, generator=g)).half()
        b = (torch.randn(K, N, generator=g)).half()
        rif = a.double() @ b.double()
        c = DP.mm16_triton(a, b)
        e = float(((c.double() - rif).abs() / (a.double().abs() @ b.double().abs())).max())
        ok &= c.dtype == torch.float32 and e <= gamma(K)
        dati[f"{M}x{K}x{N}"] = {"max_rel_su_B": e, "gamma_K": gamma(K)}
    # viste trasposte vere (come in indietro: mm(gl.T, xg))
    gy = torch.randn(40, 96, generator=g).half()
    x = torch.randn(40, 128, generator=g).half()
    c = DP.mm16_triton(gy.T, x)
    e = float(((c.double() - gy.double().T @ x.double()).abs() / (gy.double().abs().T @ x.double().abs())).max())
    ok &= e <= gamma(40)
    dati["vista_trasposta"] = e
    for base in ("attn_qkv", "ffn_down_shexp"):
        f = piani(0, base)[0]
        t = S.decodifica_pesi(f, torch.ones_like(f.d0), torch.float16)[0].to(torch.int8)
        dati[f"trit_decodifica_{base}"] = bool(torch.equal(t, S.trit_esperto(f, 0)))
        ok &= dati[f"trit_decodifica_{base}"]
    return registra("P5_kernel_mm16_e_trit", ok, **dati)


# ─────────────────────────────── P6 ───────────────────────────────
def prova_p6() -> bool:
    """Cache dei trit (E2E_PIEGA_CACHE_TRIT): dopo un salto del passo V (byte scritti in place) il trit si rilegge."""
    f = PV.famiglia_scrivibile(piani(0, "attn_qkv")[0])
    mot = S.motore("torch")
    DP.imposta(cache_trit=True)
    try:
        t1 = DP.trit(f, mot).clone()
        stessa = DP.trit(f, mot) is DP.trit(f, mot)
        r, c = 3, 5
        ta = int(t1[r, c])
        tn = ta + 1 if ta < 1 else ta - 1
        PV.salta_byte(f.blocchi, f.tipo, torch.tensor([0]), torch.tensor([r]), torch.tensor([c]),
                      torch.tensor([ta]), torch.tensor([tn]))
        t2 = DP.trit(f, mot)
        fresco = S.trit_esperto(f, 0)
        ok = stessa and torch.equal(t2, fresco) and int(t2[r, c]) == tn and int((t2 != t1).sum()) == 1
    finally:
        DP.imposta(cache_trit=False)
    return registra("P6_cache_trit_dopo_salto", ok, riuso=stessa, cambiati=int((t2 != t1).sum()))


# ─────────────────────────────── P7 (2/10) ───────────────────────────────
def prova_p7() -> bool:
    """Grandezze «da GPU» e via del codice della GPU: dY scalato da K0 (2^19, righe da 1e-8 a 0,1 x K0: oltre, il dX fp16 di OGGI trabocca e il passo si scarta), attivazioni con
    canali anomali (fino a 3e3, fp16), rami della GPU percorsi nell'interprete (decodifica Triton dei trit + `_k_mm16`,
    E2E_PIEGA_KERNEL_CPU), ricetta B3_ind2 (la ricetta del cancello del 1/10) e B3_ind32, motore triton2, x2: limiti."""
    ok, dati = True, {}
    DP.imposta(gemm="triton")
    DP.PIEGA["kernel_cpu"] = True
    try:
        for base in ("attn_qkv", "ssm_out", "ffn_down_shexp"):
            fams = piani(0, base)
            g = torch.Generator().manual_seed(77)
            T = 70
            ds = scale(fams, g)
            x = torch.randn(T, fams[0].C, generator=g)
            x[:, torch.randperm(fams[0].C, generator=g)[:4]] *= 3e3            # canali anomali
            xg = (S.ruota(x) if fams[0].ruotata else x).half()
            gy = torch.randn(T, fams[0].R, generator=g) * torch.logspace(-8, -1, T)[:, None] * 2.0 ** 19   # dX fp16 di oggi finito (a 1 x K0 trabocca: oggi 122 inf, piegato 97, esatti > 65504: 98)
            for ric_n in ("B3_ind2", "B3_ind32"):
                ric = S.RICETTE[ric_n]
                mot = S.motore("triton2")
                oggi = via_oggi(xg, ds, fams, ric, mot, gy)
                r = controlla(xg, ds, fams, gy, oggi, via_piegata(xg, ds, fams, ric, mot, gy, "x2"), "x2")
                r["finito"] = bool(torch.isfinite(oggi[0]).all())
                dati[f"{base}_{ric_n}"] = r
                ok &= accettabile(r)
    finally:
        DP.PIEGA["kernel_cpu"] = False
        DP.imposta(gemm="triton")
    return registra("P7_grandezze_gpu_via_gpu", ok, **dati)


# ─────────────────────────────── P8 (2/10) ───────────────────────────────
def _calcola_strati(ric: str, modo: str, strati, T: int = 64):
    DP.imposta(modo=modo)
    DP.CONTI.update(piegati=0, ripiego=0)
    st = S.Studente(gguf=R2B3M, strati=list(strati), ricetta=ric, motore_nome="torch", device="cpu", esperti=slice(0, 8),
                    log=lambda *_: None, densi_ternari=tuple(S.FAMIGLIE_DENSE))
    with torch.no_grad():
        for (L, n), p in zip(st.chiavi(), st.parametri()):
            g = torch.Generator().manual_seed(1000 * L + sum(ord(c) for c in n))
            p.copy_(0.01 * torch.randn(p.shape, generator=g))
    g = torch.Generator().manual_seed(110)
    x0 = (torch.randn(1, T, 2048, generator=g) * 0.5).to(st.ric.dtype_residuo)
    G = torch.randn(T, 2048, generator=g)
    ids = torch.zeros(1, T, dtype=torch.long)
    with torch.no_grad():
        xL = st.avanti(ids, x0=x0)["xL"].clone()
    st.gradiente(ids, lambda h: (h * G.to(h.dtype)).sum() / T, x0=x0)
    gr = torch.cat([p.grad.detach().double().flatten() for p in st.parametri()])
    conti = dict(DP.CONTI)
    del st
    DP.imposta(modo="")
    return xL.double(), gr, conti


def prova_p8() -> bool:
    """AMPLIFICAZIONE su 4 strati veri (0-3: GDN + attenzione piena, 8 esperti), verita' fp64 (B64): il piegato e' lontano
    da oggi QUANTO un puro riordino delle somme dei piani (DP «riordina»), e NON piu' lontano dal vero di oggi. Nato dal
    cancello GPU del 1/10: il suo pavimento (Bk2_ind32) coi densi ternari non perturba nulla."""
    ris = {n: _calcola_strati(r, m, (0, 1, 2, 3)) for n, r, m in (("B64", "B64", ""), ("oggi", "B3_ind2", ""),
                                                                  ("riordina", "B3_ind2", "riordina"),
                                                                  ("x2", "B3_ind2", "x2"), ("f32", "B3_ind2", "f32"))}

    def d(a, b):
        ga, gb = ris[a][1], ris[b][1]
        return {"h": err_norma(ris[a][0], ris[b][0]), "dtheta": float((ga - gb).norm() / gb.norm()),
                "segni_opposti": float(((torch.sign(ga) != torch.sign(gb)) & (gb != 0)).double().mean())}
    dati = {f"{a}_vs_{b}": d(a, b) for a, b in (("oggi", "B64"), ("riordina", "B64"), ("x2", "B64"), ("f32", "B64"),
                                                ("riordina", "oggi"), ("x2", "oggi"), ("f32", "oggi"), ("x2", "f32"))}
    dati["chiamate"] = {k: v[2] for k, v in ris.items()}
    ok = dati["chiamate"]["x2"]["piegati"] > 0 and dati["chiamate"]["x2"]["ripiego"] == 0
    for m in ("x2", "f32"):
        for k in ("h", "dtheta", "segni_opposti"):
            ok &= dati[f"{m}_vs_oggi"][k] <= 2 * dati["riordina_vs_oggi"][k] + 1e-9           # come il rumore
            ok &= dati[f"{m}_vs_B64"][k] <= 1.25 * dati["oggi_vs_B64"][k] + 1e-9              # non piu' lontano dal vero
    return registra("P8_amplificazione_4_strati", ok, **dati)


PROVE = {"P0": prova_p0, "P1": prova_p1, "P2": prova_p2, "P3": prova_p3, "P4": prova_p4, "P5": prova_p5, "P6": prova_p6, "P7": prova_p7, "P8": prova_p8}


def main(argv=None) -> int:
    scelte = (argv if argv is not None else sys.argv[1:]) or list(PROVE)
    ok = True
    for k in scelte:
        t0 = time.time()
        try:
            r = PROVE[k]()
        except Exception as ex:
            import traceback
            traceback.print_exc()
            r = registra(k, False, eccezione=repr(ex)[:500])
        ESITO["prove"][next(n for n in ESITO["prove"] if n.startswith(k))]["secondi"] = round(time.time() - t0, 1)
        ok &= r
    ESITO["ok"] = ok
    dest = Path(os.getenv("E2E_DP_ESITO", "@@ROOT:depot@@/e2e/prove_tt5/densi_piegati/esito_test_densi_piegati.json"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    vecchio = json.loads(dest.read_text()) if dest.exists() else {"prove": {}}
    vecchio["prove"].update(ESITO["prove"])                       # un giro con poche prove non cancella le altre
    vecchio.update({k: v for k, v in ESITO.items() if k != "prove"})
    vecchio["ok"] = all(v.get("ok") for v in vecchio["prove"].values())
    dest.write_text(json.dumps(vecchio, indent=1, ensure_ascii=False, default=str))
    print(f"esito: {'OK' if ok else 'ROSSO'} -> {dest}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

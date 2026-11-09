#!/usr/bin/env python3
"""Prove di E2E_DENSI_SSM_AB=1 (11/10/2026, E2E-C1, ospite C0B): ssm_alpha / ssm_beta TQ1_0R_1024 (2048 -> 32, perm_righe)
come famiglie dense a esperto unico. SOLO CPU (motore torch; kernel Triton nell'interprete). Sul GGUF VERO C0B, strato 0
(GDN), esperti 0-15. Stesso schema e stesse soglie di e2e_test_densi_ternari.py (prove E, B, B3), ristrette alle due
famiglie nuove, ognuna col suo controllo negativo:
  S0  spento (variabile assente) -> FAMIGLIE_DENSE / GRUPPI_DENSI dell'origine (processo figlio), acceso -> +2 in coda
  S1  (E1) studente SPENTO: peso di in_proj_a / in_proj_b == trit x d de-ruotato e permutato al bit
  S2  (E2) acceso (tutte, ssm_alpha/beta comprese) contro spento in B64: uscita dello strato entro 1e-6;
      negativo: permutazione delle righe tolta nella LineareTernaria di ssm_alpha -> lontana
  S3  (B)  dL/dtheta di ssm_alpha/beta == gradiente MATERIALIZZATO dal dW dello spento (B64 1e-6, B32 1e-4);
      negativi: senza permutazione / senza rotazione -> errore grande
  S4  (B3) kernel triton e triton2 (interprete) a esperto unico sul tensore INTERO (R = 32 < riquadri da 64) == motore
      torch (avanti, dX, dL/dd) e modi V 1/2 del kernel _k_dd_v == riferimento torch; negativo: 1 ulp di una scala
Uso: E2E_DENSI_SSM_AB=1 HIP_VISIBLE_DEVICES= ... python3 e2e_test_ssm_ab.py --gguf C0B.gguf --out esito.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("TRITON_INTERPRET", "1")
for _v in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
    os.environ[_v] = ""
assert os.environ.get("E2E_DENSI_SSM_AB") == "1", "lanciare con E2E_DENSI_SSM_AB=1"
import numpy as np  # noqa: E402
import torch  # noqa: E402

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import e2e_studente as S  # noqa: E402
import e2e_codici_scale as CS  # noqa: E402
import e2e_pv_codici as PV  # noqa: E402
import e2e_test_densi_ternari as TD  # noqa: E402  (helper: lineare, voce_densa, rel, trit_fam, _grad_materializzato)

NUOVE = ("ssm_alpha", "ssm_beta")
ESITO: dict = {"data": time.strftime("%Y-%m-%d %H:%M:%S"), "prove": {}}
MUTO = lambda *_: None  # noqa: E731
torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))


def registra(nome, ok, **d):
    ESITO["prove"][nome] = {"ok": bool(ok), **d}
    print(f"[{nome}] {'OK' if ok else 'FALLITA'} {json.dumps(d, default=str)[:1500]}", flush=True)
    return ok


def s0() -> bool:
    cod = ("import sys; sys.path.insert(0, %r); import e2e_studente as S; "
           "print(list(S.FAMIGLIE_DENSE), list(S.GRUPPI_DENSI['gdn']), len(S.GRUPPI_DENSI['tutte']))" % str(QUI))
    env = {k: v for k, v in os.environ.items() if k != "E2E_DENSI_SSM_AB"}
    spento = subprocess.run([sys.executable, "-c", cod], env=env, capture_output=True, text=True).stdout.strip()
    attese = ["attn_q", "attn_k", "attn_v", "attn_output", "attn_qkv", "attn_gate", "ssm_out", "ffn_gate_shexp",
              "ffn_up_shexp", "ffn_down_shexp"]
    ok_sp = spento == f"{attese} {['attn_qkv', 'attn_gate', 'ssm_out']} 10"
    ok_acc = list(S.FAMIGLIE_DENSE) == attese + list(NUOVE) and S.GRUPPI_DENSI["gdn"][-2:] == NUOVE and \
        len(S.GRUPPI_DENSI["tutte"]) == 12 and PV.FAMIGLIE_IDX["ssm_alpha"] == 13 and PV.FAMIGLIE_IDX["ssm_beta"] == 14 \
        and all(PV.FAMIGLIE_IDX[n] == 3 + i for i, n in enumerate(attese))
    return registra("S0_spento_origine_acceso_in_coda", ok_sp and ok_acc, spento=spento, acceso=list(S.FAMIGLIE_DENSE))


def s1_s2(g) -> bool:
    import carica_strato_gguf as CSG
    from tq1_0r_1024_codec import S1024
    kw = dict(gguf=g, strati=[0], ricetta="B64", motore_nome="torch", device="cpu", esperti=slice(0, 16), log=MUTO)
    s_off = S.Studente(**kw)
    e1, neg1 = {}, {}
    for n in NUOVE:
        t = s_off._tens[f"blk.0.{n}.weight"]
        tipo = t.tensor_type.name
        nb = CS.LAYOUT[tipo][0]
        b = torch.from_numpy(np.ascontiguousarray(np.asarray(t.data)).reshape(-1, nb))
        W = CS.pesi_da_trit_scale(CS.trit_da_blocchi(b, tipo), CS.scale_da_blocchi(b, tipo).float(), tipo,
                                  torch.from_numpy(S1024)).numpy().reshape(np.asarray(t.data).shape[0], -1)
        voce = TD.voce_densa(s_off, 0, n)
        Wh = torch.from_numpy(np.array(CSG._applica(voce, W), copy=True)).to(torch.float64)
        wm = TD.lineare(s_off, 0, n).weight.detach()
        e1[n] = {"tipo": tipo, "forma": list(W.shape), "rel": voce["rel"], "al_bit": bool(torch.equal(wm, Wh))}
        neg1[n] = bool(torch.equal(wm, torch.from_numpy(W).to(torch.float64)))
    ok1 = all(v["al_bit"] and v["tipo"] == "TQ1_0R_1024" for v in e1.values()) and not any(neg1.values())
    registra("S1_spento_peso_al_bit", ok1, famiglie=e1, negativo_senza_permutazione_uguale=neg1)
    s_on = S.Studente(**kw, densi_ternari=tuple(S.FAMIGLIE_DENSE))
    presenti = [n for n in s_on.esperti(0).fam if n in S.FAMIGLIE_DENSE]
    gen = torch.Generator().manual_seed(2813)
    x0 = (torch.randn(1, 32, 2048, generator=gen) * 0.5).to(torch.float64)
    ids = torch.zeros(1, 32, dtype=torch.long)
    with torch.no_grad():
        y_off = s_off.avanti(ids, x0=x0)["xL"]
        y_on = s_on.avanti(ids, x0=x0)["xL"]
        e2 = TD.rel(y_on, y_off)
        la = TD.lineare(s_on, 0, "ssm_alpha")
        iy = la.inv_y
        la.inv_y = None
        neg2 = TD.rel(s_on.avanti(ids, x0=x0)["xL"], y_off)
        la.inv_y = iy
        ritorno = bool(torch.equal(s_on.avanti(ids, x0=x0)["xL"], y_on))
    ok2 = e2 < 1e-6 and neg2 > 1e-6 and ritorno and all(n in presenti for n in NUOVE) \
        and type(la).__name__ == "LineareTernaria"
    del s_off, s_on
    return registra("S2_acceso_contro_spento_B64", ok2, rel=e2, negativo_senza_perm=neg2, ritorno_al_bit=ritorno,
                    famiglie_dense_strato0=presenti) and ok1


def _grad(g, ricetta, soglia) -> dict:
    kw = dict(gguf=g, strati=[0], ricetta=ricetta, motore_nome="torch", device="cpu", esperti=slice(0, 16), log=MUTO)
    st_on = S.Studente(**kw, densi_ternari=tuple(S.FAMIGLIE_DENSE))
    st_off = S.Studente(**kw)
    dt = st_on.ric.dtype_residuo
    for n in NUOVE:
        TD.lineare(st_off, 0, n).weight.requires_grad_(True)
    gen = torch.Generator().manual_seed(2810)
    T = 32
    x0 = (torch.randn(1, T, 2048, generator=gen) * 0.5).to(dt)
    G = torch.randn(T, 2048, generator=gen).to(dt)
    ids = torch.zeros(1, T, dtype=torch.long)
    st_on.gradiente(ids, lambda h: (h * G).sum() / T, x0=x0)
    with torch.enable_grad():
        ((st_off.nascosto(ids, x0=x0) * G).sum() / T).backward()
    per, peggiore, neg = {}, 0.0, {}
    for n in NUOVE:
        gon = st_on.esperti(0).theta[n].grad.to(torch.float64)
        rif = TD._grad_materializzato(st_off, st_on, 0, n)
        e = TD.rel(gon, rif)
        peggiore = max(peggiore, e)
        per[n] = {"rel": e, "norma": float(rif.norm()), "ruotata": bool(st_on.esperti(0).fam[n].ruotata)}
        neg[n] = {"senza_perm": TD.rel(gon, TD._grad_materializzato(st_off, st_on, 0, n, perm=False)),
                  "senza_rot": TD.rel(gon, TD._grad_materializzato(st_off, st_on, 0, n, rotazione=False))}
    del st_on, st_off
    return {"rel_massimo": peggiore, "soglia": soglia, "per_famiglia": per, "negativi": neg,
            "ok": peggiore < soglia and all(v["norma"] > 0 for v in per.values())
            and all(v["senza_perm"] > 1e-2 and v["senza_rot"] > 1e-2 for v in neg.values())}


def s3(g) -> bool:
    b64 = _grad(g, "B64", 1e-6)
    b32 = _grad(g, "B32", 1e-4)
    return registra("S3_theta_grad_materializzato", b64["ok"] and b32["ok"], B64=b64, B32=b32)


def s4(g) -> bool:
    from gguf import GGUFReader
    r = GGUFReader(str(g))
    tens = {t.name: t for t in r.tensors}
    ric = S.RICETTE["B_ind32"]
    esiti, ok = {}, True
    for n in NUOVE:
        f = S.famiglia_da_tensore(tens[f"blk.0.{n}.weight"], n)
        gen = torch.Generator().manual_seed(33)
        T = 64
        piano = S.Piano(torch.zeros(T, 1, dtype=torch.long), 1)
        x = (torch.randn(T, f.C, generator=gen) * 0.3).to(torch.float16)
        x32 = x.float()
        gy = (torch.randn(T, f.R, generator=gen) * 0.1).float()
        d = f.d0.to(torch.float16).to(torch.float32)
        mt = S.MotoreTorch()
        rif = (mt.avanti(x, d, f, piano, ric), mt.indietro_x(gy, d, f, piano, ric), mt.dd(gy, x32, f, piano, torch.float32))
        e = {"E": f.E, "R": f.R, "C": f.C, "tipo": f.tipo}
        for nm in ("triton", "triton2"):
            mot = S.motore(nm)
            e[nm] = {"avanti": TD.rel(mot.avanti(x, d, f, piano, ric), rif[0]),
                     "dx": TD.rel(mot.indietro_x(gy, d, f, piano, ric), rif[1]),
                     "dd": TD.rel(mot.dd(gy, x32, f, piano, torch.float32), rif[2])}
            ok &= max(e[nm].values()) < 1e-5
        bk, bt = PV.BufferV.crea(f), PV.BufferV.crea(f)
        for modo in (1, 2):
            dk = PV.dd_v_triton(gy, x32, f, piano, PV.BLOCCHI_V, bk, modo)
            dtr = PV.dd_v_torch(gy, x32, f, piano, torch.float32, bt, modo)
            e[f"v_modo{modo}_dd"] = TD.rel(dk, dtr)
        e["v_cand_uguali"] = bool(torch.equal(bk.cand, bt.cand))
        e["v_prop"] = TD.rel(bk.prop, bt.prop)
        e["v_S"] = TD.rel(bk.S, bt.S)
        ok &= e["v_cand_uguali"] and max(e["v_modo1_dd"], e["v_modo2_dd"], e["v_prop"], e["v_S"]) < 1e-5
        d2 = d.clone()
        v16 = np.float16(float(d2.view(-1)[5]))
        d2.view(-1)[5] = float(np.nextafter(v16, np.float16(np.inf)))
        e["negativo_1ulp"] = TD.rel(S.motore("triton2").avanti(x, d2, f, piano, ric), rif[0])
        ok &= e["negativo_1ulp"] > 1e-5
        esiti[n] = e
    return registra("S4_kernel_triton_R32", ok, famiglie=esiti)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--solo", default="S0,S1,S3,S4")
    a = ap.parse_args()
    g = Path(a.gguf)
    solo = set(a.solo.split(","))
    t0 = time.time()
    ok = True
    if "S0" in solo:
        ok &= s0()
    if "S1" in solo:
        ok &= s1_s2(g)
    if "S3" in solo:
        ok &= s3(g)
    if "S4" in solo:
        ok &= s4(g)
    ESITO["ok"] = bool(ok)
    ESITO["secondi"] = round(time.time() - t0, 1)
    json.dump(ESITO, open(a.out, "w"), indent=1, default=str)
    print("ESITO", "OK" if ok else "FALLITO", ESITO["secondi"], "s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

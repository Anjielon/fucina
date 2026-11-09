#!/usr/bin/env python3
"""27/9 — cancello del motore «triton2» (kernel esperti di e2e_studente) SULLA GPU: vecchio («triton») contro
nuovo («triton2»), stessi pesi, stesso instradamento, stessi dati.

Per ogni strato richiesto (default 0,1,38,39):
  1. ms/token dello strato (avanti senza grafo + ricalcolo + indietro, `_misura_passo`, token veri del corpus),
     vecchio contro nuovo, e il rapporto;
  2. microprova dei kernel per famiglia (gate, down) con l'instradamento vero: ms di avanti, gate+up insieme,
     dX e dL/dd; uguaglianza al bit di avanti e dX, errore relativo di dL/dd;
  3. dtheta come G1: stesso ingresso vero dello strato, stesso top-k, stesso gradiente d'uscita casuale G
     (x K, come la scala di G1), tre corse: vecchio, nuovo, e riferimento = vecchio con indietro fp32
     (ricetta *_ind32). Per nuovo e vecchio contro il riferimento: gruppi vivi con g32 != 0, nulli, segno
     sbagliato, quota (soglia G1 max_quota_dtheta_rotti = 1e-3), errore relativo mediano/p99; nuovo contro
     vecchio: uscita e dL/dx (al bit?), errore relativo di dtheta mediano/p99/massimo.
Verdetto: uscita e dL/dx del nuovo entro 1e-5 dal vecchio, quota G1 del nuovo <= 1e-3 in ogni strato.

Uso: strumenti/cancello_kernel.sh (guardia GPU). Prova a secco su CPU (interprete, piccolo):
  TRITON_INTERPRET=1 HIP_VISIBLE_DEVICES= python3 strumenti/cancello_kernel.py --dispositivo cpu --strati 0 \
      --esperti 8 --token 6 --ripetizioni 1 --uscita /tmp/ck.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI.parent))
import torch  # noqa: E402

import e2e_studente as S  # noqa: E402


def rel(a, b) -> float:
    a, b = a.double(), b.double()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-300))


def kth(t: torch.Tensor, q: float) -> float:
    if t.numel() == 0:
        return 0.0
    t = t.reshape(-1).float()
    return float(t.kthvalue(max(1, min(t.numel(), int(round(q * t.numel()))))).values)


def metti_motore(st, mot) -> None:
    st.mot = mot
    for L in st.strati:
        st.esperti(L).mot = mot


def ricetta(st, ric) -> None:
    for L in st.strati:
        st.esperti(L).ric = ric


def cronometra(f, n: int) -> float:
    for _ in range(2):
        f()
    S._sincronizza()
    t0 = time.time()
    for _ in range(n):
        f()
    S._sincronizza()
    return (time.time() - t0) / n * 1e3


def microprova(st, L: int, piano, mots: dict, n: int) -> dict:
    es = st.esperti(L)
    out = {}
    g = torch.Generator(device=st.device).manual_seed(L)
    for nome in ("gate", "down"):
        fam = es.fam[nome]
        d = es.scala(nome).detach()
        xs = torch.randn(piano.S, fam.C, device=st.device, dtype=torch.float16, generator=g)
        gy = (torch.randn(piano.S, fam.R, device=st.device, dtype=torch.float16, generator=g) * 1e-2)
        ric = st.ric
        r, ris = {}, {}
        for k, m in mots.items():
            ris[k] = (m.avanti(xs, d, fam, piano, ric), m.indietro_x(gy, d, fam, piano, ric),
                      m.dd(gy, xs, fam, piano, torch.float32))
            r[k] = {"avanti_ms": cronometra(lambda: m.avanti(xs, d, fam, piano, ric), n),
                    "dx_ms": cronometra(lambda: m.indietro_x(gy, d, fam, piano, ric), n),
                    "dd_ms": cronometra(lambda: m.dd(gy, xs, fam, piano, torch.float32), n)}
        if nome == "gate":
            fu, du = es.fam["up"], es.scala("up").detach()
            r["vecchio"]["gate_up_ms"] = cronometra(lambda: (mots["vecchio"].avanti(xs, d, fam, piano, ric),
                                                             mots["vecchio"].avanti(xs, du, fu, piano, ric)), n)
            if mots["nuovo"].puo_fondere(fam, fu, ric):
                r["nuovo"]["gate_up_ms"] = cronometra(
                    lambda: mots["nuovo"].avanti_due(xs, (d, du), (fam, fu), piano, ric), n)
        a, b = ris["vecchio"], ris["nuovo"]
        r["avanti_bit"], r["dx_bit"] = bool(torch.equal(a[0], b[0])), bool(torch.equal(a[1], b[1]))
        r["avanti_rel"], r["dx_rel"], r["dd_rel"] = rel(b[0], a[0]), rel(b[1], a[1]), rel(b[2], a[2])
        r["rapporto"] = {k: r["vecchio"][k] / r["nuovo"][k] for k in r["nuovo"] if k in r["vecchio"]}
        r["assegnazioni"] = piano.S
        out[nome] = r
    return out


def dtheta_strato(st, L: int, xin0, ctx, instr, G, K: float, mots: dict, ric_rif) -> dict:
    """Una corsa per motore (e il riferimento fp32): (y, dL/dx, [dtheta gate, up, down])."""
    corse = {}
    ric0 = st.esperti(L).ric
    for k, (mot, ric) in {"vecchio": (mots["vecchio"], ric0), "nuovo": (mots["nuovo"], ric0),
                          "rif_fp32": (mots["vecchio"], ric_rif)}.items():
        metti_motore(st, mot)
        ricetta(st, ric)
        xin = xin0.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            y = st._strato(L, xin, ctx, instr=instr)
            (y.float() * G).sum().mul(K).backward()
        corse[k] = (y.detach(), xin.grad, [st.esperti(L).theta[n].grad.clone() for n in S.FAMIGLIE])
        for p in st.parametri():
            p.grad = None
    ricetta(st, ric0)
    d0 = [st.esperti(L).fam[n].d0 for n in S.FAMIGLIE]

    def g1(ga, gb):
        vive = nulli = rotti = 0
        rr = []
        for a, b, dz in zip(ga, gb, d0):
            v = (dz != 0) & (b != 0)
            vive += int(v.sum())
            nulli += int((v & (a == 0)).sum())
            rotti += int((v & (a != 0) & (torch.sign(a) != torch.sign(b))).sum())
            rr.append(((a - b).abs() / b.abs())[v].float())
        rr = torch.cat(rr)
        return {"vive_con_g_non_nullo": vive, "nulli": nulli, "segno_sbagliato": rotti,
                "quota_rotti": (nulli + rotti) / max(vive, 1), "err_rel_mediano": kth(rr, 0.5),
                "err_rel_p99": kth(rr, 0.99)}
    v, n, r = corse["vecchio"], corse["nuovo"], corse["rif_fp32"]
    return {"nuovo_contro_vecchio": {"uscita_bit": bool(torch.equal(n[0], v[0])), "uscita_rel": rel(n[0], v[0]),
                                     "dx_bit": bool(torch.equal(n[1], v[1])), "dx_rel": rel(n[1], v[1]),
                                     "dtheta_rel_max": max(rel(a, b) for a, b in zip(n[2], v[2])),
                                     **{"dtheta_" + k: x for k, x in g1(n[2], v[2]).items()}},
            "nuovo_contro_fp32": g1(n[2], r[2]), "vecchio_contro_fp32": g1(v[2], r[2])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strati", default="0,1,38,39")
    ap.add_argument("--token", type=int, default=8192)
    ap.add_argument("--l-seq", type=int, default=4096)
    ap.add_argument("--ricetta", default="B")
    ap.add_argument("--ripetizioni", type=int, default=3)
    ap.add_argument("--micro", type=int, default=10, help="ripetizioni della microprova dei kernel")
    ap.add_argument("--K", type=float, default=256.0, help="scala del gradiente d'uscita (come K di G1)")
    ap.add_argument("--blocchi", default="", help="JSON dei Blocchi del motore nuovo (default: Blocchi())")
    ap.add_argument("--esperti", type=int, default=0, help="solo i primi N esperti (prova a secco)")
    ap.add_argument("--dispositivo", default="cuda")
    ap.add_argument("--uscita", default="@@ROOT:models@@/gguf/tony-tern/e2e_studente/cancello_kernel.json")
    a = ap.parse_args(argv)
    gpu = a.dispositivo == "cuda"
    if gpu:
        rc = subprocess.run([str(QUI / "gpu_occupata.sh")], capture_output=True, text=True)
        if rc.returncode != 0:
            print(f"GPU occupata, non misuro:\n{rc.stdout}")
            return 3
    strati = [int(s) for s in a.strati.split(",")]
    blk_n = S.Blocchi(**json.loads(a.blocchi)) if a.blocchi else S.Blocchi()
    mots = {"vecchio": S.motore("triton"), "nuovo": S.motore("triton2", blk_n)}
    ric_rif = S.RICETTE[a.ricetta + "_ind32"] if a.ricetta + "_ind32" in S.RICETTE else \
        S.Ricetta(a.ricetta + "_ind32", S.RICETTE[a.ricetta].dtype_residuo, S.RICETTE[a.ricetta].dtype_gemm,
                  S.RICETTE[a.ricetta].ruota, torch.float32)
    st = S.Studente(strati=strati, ricetta=a.ricetta, motore_nome="triton", device=a.dispositivo,
                    esperti=slice(0, a.esperti) if a.esperti else None)
    if gpu:
        st.carica_embedding()
    B = max(1, a.token // a.l_seq)
    T = a.token // B
    ids = S._ids_reali(B, T) if gpu else torch.randint(0, 150000, (B, T))
    esito = {"data": time.strftime("%Y-%m-%d %H:%M:%S"), "gguf": str(S.GGUF_THOR_T), "strati": strati,
             "ricetta": a.ricetta, "token": B * T, "blocchi_nuovo": blk_n.__dict__, "K": a.K, "strato": {}}
    # 1. tempi per strato (stessa catena: gli ingressi degli strati 38-39 vengono dagli strati caricati prima)
    tempi = {}
    for k, m in mots.items():
        metti_motore(st, m)
        tempi[k] = S._misura_passo(st, ids, a.ripetizioni)
        print(json.dumps({k: tempi[k]}), flush=True)
    # 2-3. ingressi veri e instradamento del vecchio, poi microprova e dtheta per strato
    metti_motore(st, mots["vecchio"])
    stato = st.avanti(ids)
    ok = True
    for i, L in enumerate(strati):
        tv, tn = tempi["vecchio"]["per_strato_s"][L], tempi["nuovo"]["per_strato_s"][L]
        r = {"ms_token": {"vecchio": (tv["avanti"] + tv["indietro"]) / (B * T) * 1e3,
                          "nuovo": (tn["avanti"] + tn["indietro"]) / (B * T) * 1e3,
                          "vecchio_avanti": tv["avanti"] / (B * T) * 1e3, "nuovo_avanti": tn["avanti"] / (B * T) * 1e3,
                          "vecchio_indietro": tv["indietro"] / (B * T) * 1e3,
                          "nuovo_indietro": tn["indietro"] / (B * T) * 1e3}}
        r["ms_token"]["rapporto"] = r["ms_token"]["vecchio"] / r["ms_token"]["nuovo"]
        piano = S.Piano(stato["instr"][i], st.esperti(L).E)
        r["kernel"] = microprova(st, L, piano, mots, a.micro)
        gen = torch.Generator(device=st.device).manual_seed(1000 + L)
        G = torch.randn(stato["ingressi"][i].shape, generator=gen, device=st.device, dtype=torch.float32)
        r["dtheta"] = dtheta_strato(st, L, stato["ingressi"][i], stato["ctx"], stato["instr"][i], G, a.K, mots,
                                    ric_rif)
        nv = r["dtheta"]["nuovo_contro_vecchio"]
        r["ok"] = nv["uscita_rel"] <= 1e-5 and nv["dx_rel"] <= 1e-5 \
            and r["dtheta"]["nuovo_contro_fp32"]["quota_rotti"] <= 1e-3
        ok &= r["ok"]
        esito["strato"][str(L)] = r
        print(json.dumps({"strato": L, **r}), flush=True)
    metti_motore(st, mots["vecchio"])
    esito["passo_ms_token"] = {k: t["passo_s"] / (B * T) * 1e3 for k, t in tempi.items()}
    esito["cambi_instradamento"] = st.cambi()
    esito["ok"] = bool(ok)
    Path(a.uscita).parent.mkdir(parents=True, exist_ok=True)
    Path(a.uscita).write_text(json.dumps(esito, indent=1))
    print("\nstrato  ms/tok vecchio  nuovo  x   | quota G1 nuovo  vecchio | uscita/dx al bit")
    for L, r in esito["strato"].items():
        m, d = r["ms_token"], r["dtheta"]
        print(f"{L:>5}  {m['vecchio']:12.4f} {m['nuovo']:7.4f} {m['rapporto']:5.2f} | "
              f"{d['nuovo_contro_fp32']['quota_rotti']:.2e}  {d['vecchio_contro_fp32']['quota_rotti']:.2e} | "
              f"{d['nuovo_contro_vecchio']['uscita_bit']}/{d['nuovo_contro_vecchio']['dx_bit']}")
    print(f"CANCELLO {'✅' if ok else '⛔'} -> {a.uscita}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

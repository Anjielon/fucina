#!/usr/bin/env python3
"""Cancello B3 (27/9/2026): la ricetta B3_ind32 (non-esperti e, a scelta, testa KL in bf16x3; logit del maestro
bf16) contro B_ind32 sulla GPU della corsa. Lo lancia Angelo: `strumenti/cancello_b3.sh`.

Tre misure, un esito (`<uscita>/esito_cancello_b3.json`):
1. **Tempo** (ms/token del passo intero: avanti + KL + indietro + Adam) con `e2e_allena.misura` sugli strati
   0,1 (GDN) e 38,39 (GDN + attenzione piena), un micro-lotto di `--righe-micro` righe train GIA' presenti (i
   blocchi 0-1 di q8-EB-pieno sono consumati: si prendono le prime righe presenti).
2. **KL(B || B3) per token** fra le distribuzioni dello studente dei due bracci (stesso GGUF, stesso theta = 0)
   su `--righe-held` righe held-out: media per token <= `--soglia-kl` (0,0013). Si registrano anche max, p99 e
   la KL(teacher || studente) di ciascun braccio (quanto B3 sposta la perdita).
3. **dL/dtheta** del braccio B3 contro B sul primo micro-lotto train presente, perdita dell'allenamento (K0 *
   KL / token): quota di scale vive (d0 != 0, g_B != 0) con g_B3 nullo o di segno opposto <= `--soglia-dtheta`
   (1e-3, la stessa di G1: `e2e_allena.confronta_dtheta`).
Il braccio B usa sempre le teste fp32; il braccio B3 `--testa-kl` (default bf16x3) e `--logit-maestro` (default
bf16). KL e dtheta si misurano sugli `--strati-kl` (default tutti e 40: e' il modello che si allena), un braccio
alla volta (nascosti e gradienti su CPU, poi il braccio si libera). Si registra la via della GEMM bf16
(`e2e_bf16x3.stato()`): "uscita_bf16" vuol dire mm.dtype assente, e B3 non ha senso (errore ~1e-3).

Verdetto (prima versione, 27/9 mattina): fedele = KL e dtheta entro soglia e via "out_dtype"; ammessa = fedele e piu'
veloce su entrambe le coppie di strati (la seconda versione e' sotto). Non cambia nessuna corsa: l'allenamento con B3 si sceglie a mano (`--ricetta B3_ind32
--testa-kl bf16x3 --logit-maestro bf16`).

27/9 pomeriggio (dopo il primo cancello, BOCCIATO alle 11:04: KL media 8,8e-4 ma dtheta 3,9% di segni opposti, e
1,06-1,88x PIU' LENTO), `TECNICHE_E2E_PV.md` §bf16x3-correzione:
- la via della GPU e' ora "triton" (`e2e_bf16x3.mm_triton`): la "out_dtype" era esatta ma senza WMMA (2,4 TFLOPS
  come la fp32: tre GEMM = piu' lento); si registra in `via_bf16x3` e il verdetto la esige per "piu_veloce";
- braccio RUMORE (`--ricetta-rumore`, default Bk2_ind32 = B_ind32 con le non-esperte in fp32 e la somma su K in
  due meta', teste fp32): quanto B si sposta DA SOLA per un arrotondamento fp32 diverso. Sui 40 strati la KL e il
  dtheta non misurano solo l'aritmetica: l'instradamento top-k (discontinuo) e le ricorrenze del GDN amplificano
  qualunque differenza. G1 confronta due INDIETRO sullo stesso avanti; qui l'avanti cambia, e la soglia 1e-3 di G1
  e' da riportare al pavimento. Verdetto `fedele_rumore`: KL media entro soglia (assoluta, come prima) e dtheta di
  B3 entro max(soglia assoluta, `--fattore-rumore` x rumore) su quota di segni opposti, quota pesata (soglia 1e-3)
  ed errore relativo in norma (`--soglia-dtheta-norma` 1e-3: sui pochi strati il regime e' lineare, B3 ~2x il rumore
  ma entrambi ~1e-5; prova CPU su 1 strato: 1,2e-5 contro 5,4e-6);
  `fedele_assoluto` = il criterio di prima (resta nell'esito). `ammessa` = fedele_rumore (se il braccio rumore
  c'e') e piu' veloce;
- dtheta anche PESATO (`dtheta_pesato`): errore relativo in norma ||g - g_B|| / ||g_B|| e quota di segni opposti
  pesata con |g_B| (le scale col gradiente quasi nullo cambiano segno per un nonnulla e pesano zero).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import torch

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import e2e_allena as A  # noqa: E402


def _strati(testo: str) -> list:
    out = []
    for pezzo in testo.split(","):
        if "-" in pezzo:
            a, b = pezzo.split("-")
            out += list(range(int(a), int(b) + 1))
        elif pezzo:
            out.append(int(pezzo))
    return out


def braccio(ricetta: str, strati: list, src, righe_held: list, righe_train: list, ca: A.ConfigAllena, gguf,
            device: str, log=print, esperti=None) -> dict:
    """Uno studente con la ricetta e le teste di `ca`: nascosti (dopo output_norm, CPU fp32) sulle righe held-out,
    KL(teacher || studente) per token, dL/dtheta (CPU) sulle righe train con la perdita dell'allenamento."""
    import e2e_studente as S
    dev = torch.device(device)
    t0 = time.time()
    st = S.Studente(gguf=gguf, strati=strati, ricetta=ricetta, motore_nome="triton" if dev.type == "cuda" else "torch",
                    device=device, esperti=esperti, log=log)
    if dev.type == "cuda":
        st.carica_embedding()
    WS, WT, _ = A.teste(st, dev, ca)
    car = A.Caricatore(src, dev)
    out = {"ricetta": ricetta, "testa_kl": ca.testa_kl, "logit_maestro": ca.logit_maestro, "h": [], "kl_T": []}
    with torch.no_grad():
        for i in righe_held:
            lot = car.prendi("heldout", [i])
            hS = st.nascosto(lot.ids, inizi=lot.inizi)
            kl = torch.cat([A.kl_pezzo(hS[j:j + ca.pezzo_kl], WS, lot.hT[j:j + ca.pezzo_kl], WT, False)["kl"]
                            for j in range(0, hS.shape[0], ca.pezzo_kl)])
            out["h"].append(hS.float().cpu())
            out["kl_T"].append(kl.cpu())
            del hS, lot
    A._azzera(st)
    A.gradiente_passo(st, car, "train", righe_train, dataclasses.replace(ca, righe_per_micro=len(righe_train)),
                      WS, WT, ca.K0)
    ctl = st.controlla(tollera_trabocco=True)
    out["grad"] = [p.grad.detach().cpu() if p.grad is not None else torch.zeros_like(p, device="cpu")
                   for p in st.parametri()]
    out["d0"] = [st.esperti(L).fam[n].d0.cpu() for (L, n) in st.chiavi()]
    out["cambi_instradamento"] = ctl["cambi_instradamento"]
    out["trabocco_fp16"] = ctl["trabocco_fp16"]
    out["secondi"] = round(time.time() - t0, 1)
    car.chiudi()
    del st, WS, WT, car
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return out


def testa_gguf(gguf, dev) -> torch.Tensor:
    """output.weight (Q5_K) del GGUF dello studente, dequantizzata fp32: la stessa di `Studente.testa`."""
    import numpy as np
    from gguf import GGUFReader
    from dequant_gguf_torch import dequant_np
    t = next(x for x in GGUFReader(str(gguf)).tensors if x.name == "output.weight")
    return torch.from_numpy(np.asarray(dequant_np(t))).to(device=dev, dtype=torch.float32)


@torch.no_grad()
def kl_fra_bracci(hB: list, hB3: list, WS_B, WS_B3, pezzo: int, device) -> dict:
    """KL(p_B || p_B3) per token (fp64), ciascun braccio con la sua testa."""
    kl = []
    for a, b in zip(hB, hB3):
        for j in range(0, a.shape[0], pezzo):
            lB = A._logit_studente(a[j:j + pezzo].to(device), WS_B)
            lB.sub_(torch.logsumexp(lB, -1, keepdim=True))
            l3 = A._logit_studente(b[j:j + pezzo].to(device), WS_B3)
            l3.sub_(torch.logsumexp(l3, -1, keepdim=True))
            kl.append(((lB - l3) * lB.exp()).sum(-1, dtype=torch.float64).cpu())
            del lB, l3
    k = torch.cat(kl)
    return {"token": int(k.numel()), "media": float(k.mean()), "max": float(k.max()),
            "p99": A._kth(k.float(), 0.99)}


def dtheta_pesato(d0: list, g_prova: list, g_rif: list) -> dict:
    """Sulle scale vive (d0 != 0) col gradiente di riferimento non nullo: ||g - g_rif|| / ||g_rif|| (fp64) e quota
    di segni opposti (o nulli) pesata con |g_rif|."""
    dd = nn = rot = tot = 0.0
    for d, a, b in zip(d0, g_prova, g_rif):
        a = a.to(b.device).double()
        b = b.double()
        v = (d.to(b.device) != 0) & (b != 0)
        dd += float(((a - b)[v]).pow(2).sum())
        nn += float(b[v].pow(2).sum())
        w = b.abs()[v]
        rot += float(w[(torch.sign(a) != torch.sign(b))[v]].sum())
        tot += float(w.sum())
    return {"err_rel_norma": math.sqrt(dd / max(nn, 1e-300)), "quota_rotti_pesata": rot / max(tot, 1e-300)}


def confronto(bB: dict, bB3: dict, WS_B, WS_B3, ca: A.ConfigAllena, device) -> dict:
    kl = kl_fra_bracci(bB["h"], bB3["h"], WS_B, WS_B3, ca.pezzo_kl, device)
    dth = A.confronta_dtheta(bB["d0"], bB3["grad"], bB["grad"])
    dth.update(dtheta_pesato(bB["d0"], bB3["grad"], bB["grad"]))
    kT = {n: float(torch.cat(b["kl_T"]).mean()) for n, b in (("B", bB), ("B3", bB3))}
    return {"kl_B_B3": kl, "dtheta": dth, "kl_teacher_studente": kT,
            "h_err_rel": max(float((a - b).abs().max() / a.abs().max()) for a, b in zip(bB["h"], bB3["h"])),
            "cambi_instradamento": {"B": bB["cambi_instradamento"], "B3": bB3["cambi_instradamento"]}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--teacher", default="@@ROOT:lavoro@@/e2e/teacher/q8-EB-pieno")
    ap.add_argument("--gguf", default=None)
    ap.add_argument("--uscita", default=str(A.RADICE / "cancello_b3"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ricetta-rif", default="B_ind32")
    ap.add_argument("--ricetta", default="B3_ind32")
    ap.add_argument("--testa-kl", choices=("fp32", "bf16x3"), default="bf16x3")
    ap.add_argument("--logit-maestro", choices=("fp32", "bf16"), default="bf16")
    ap.add_argument("--strati-tempo", action="append", default=None, help="default: 0,1 e 38,39")
    ap.add_argument("--strati-kl", default="0-39")
    ap.add_argument("--righe-micro", type=int, default=A.ConfigAllena.righe_per_micro)
    ap.add_argument("--righe-held", type=int, default=2)
    ap.add_argument("--ripetizioni", type=int, default=3)
    ap.add_argument("--soglia-kl", type=float, default=0.0013)
    ap.add_argument("--soglia-dtheta", type=float, default=A.ConfigAllena.max_quota_dtheta_rotti)
    ap.add_argument("--ricetta-rumore", default="Bk2_ind32",
                    help="braccio rumore (teste fp32): quanto B si sposta per un arrotondamento fp32 diverso")
    ap.add_argument("--senza-rumore", action="store_true", help="niente braccio rumore (verdetto assoluto)")
    ap.add_argument("--fattore-rumore", type=float, default=2.0,
                    help="dtheta di B3 entro questo multiplo di quello del braccio rumore")
    ap.add_argument("--soglia-dtheta-norma", type=float, default=1e-3,
                    help="||dg||/||g|| sempre ammesso sotto questo valore (pochi strati: regime lineare, B3 ~2x il "
                         "rumore ma entrambi ~1e-5)")
    ap.add_argument("--salta-tempo", action="store_true")
    ap.add_argument("--esperti", type=int, default=0, help="solo prove su CPU: i primi N esperti")
    a = ap.parse_args(argv)
    import e2e_studente as S
    import e2e_bf16x3 as X3
    gguf = Path(a.gguf) if a.gguf else S.GGUF_THOR_T
    usc = Path(a.uscita)
    usc.mkdir(parents=True, exist_ok=True)
    log = A._registro(usc)
    esperti = slice(0, a.esperti) if a.esperti else None
    strati_tempo = [_strati(s) for s in (a.strati_tempo or ["0,1", "38,39"])]
    ca_rif = A.ConfigAllena(righe_per_micro=a.righe_micro)
    ca_b3 = dataclasses.replace(ca_rif, testa_kl=a.testa_kl, logit_maestro=a.logit_maestro)
    esito = {"data": A.adesso(), "argv": sys.argv, "gguf": str(gguf), "teacher": a.teacher,
             "rif": {"ricetta": a.ricetta_rif, "testa_kl": "fp32", "logit_maestro": "fp32"},
             "prova": {"ricetta": a.ricetta, "testa_kl": a.testa_kl, "logit_maestro": a.logit_maestro},
             "soglie": {"kl_B_B3_media": a.soglia_kl, "dtheta_quota_rotti": a.soglia_dtheta,
                        "dtheta_err_rel_norma": a.soglia_dtheta_norma,
                        "fattore_rumore": None if a.senza_rumore else a.fattore_rumore},
             "rumore": None if a.senza_rumore else {"ricetta": a.ricetta_rumore, "testa_kl": "fp32",
                                                    "logit_maestro": "fp32"}}
    with A._coesistenza_gpu(a.device == "cuda"):
        if a.device == "cuda":
            A._guardia_gpu(600)
        src = A.SorgenteCache(a.teacher)
        righe_train = A.prime_disponibili(src, "train", a.righe_micro)
        righe_held = A.prime_disponibili(src, "heldout", a.righe_held)
        esito.update({"righe_train": righe_train, "righe_held": righe_held})
        log(f"cancello B3: train {righe_train}, held-out {righe_held}, {a.ricetta_rif} contro {a.ricetta} "
            f"(testa {a.testa_kl}, maestro {a.logit_maestro})")
        if not a.salta_tempo:
            tempo = {}
            for nome, ric, ca in ((a.ricetta_rif, a.ricetta_rif, ca_rif), (a.ricetta, a.ricetta, ca_b3)):
                tempo[nome] = A.misura(strati_tempo, Path(a.teacher), gguf, ca, a.ripetizioni, a.device, ric, log=log,
                                       disponibili=True)
            conf = {}
            for s in tempo[a.ricetta_rif]["configurazioni"]:
                t_r = tempo[a.ricetta_rif]["configurazioni"][s]["totale_ms_token"]
                t_p = tempo[a.ricetta]["configurazioni"][s]["totale_ms_token"]
                conf[s] = {"ms_token_rif": t_r, "ms_token_prova": t_p, "rapporto": t_p / t_r,
                           "parti_rif": tempo[a.ricetta_rif]["configurazioni"][s]["ms_token"],
                           "parti_prova": tempo[a.ricetta]["configurazioni"][s]["ms_token"],
                           "picco_gib": [tempo[a.ricetta_rif]["configurazioni"][s]["picco_torch_gib"],
                                         tempo[a.ricetta]["configurazioni"][s]["picco_torch_gib"]]}
                log(f"tempo strati {s}: {a.ricetta_rif} {t_r:.4f} ms/token, {a.ricetta} {t_p:.4f} ms/token "
                    f"(x{t_p / t_r:.3f})")
            esito["tempo"] = conf
            A._scrivi_json(usc / "esito_cancello_b3.json", esito)
        strati_kl = _strati(a.strati_kl)
        bB = braccio(a.ricetta_rif, strati_kl, src, righe_held, righe_train, ca_rif, gguf, a.device, log, esperti)
        bB3 = braccio(a.ricetta, strati_kl, src, righe_held, righe_train, ca_b3, gguf, a.device, log, esperti)
        bR = None if a.senza_rumore else braccio(a.ricetta_rumore, strati_kl, src, righe_held, righe_train, ca_rif,
                                                 gguf, a.device, log, esperti)
        # le due teste insieme per la KL fra i bracci (lo studente non serve: la testa e' output.weight del GGUF)
        dev = torch.device(a.device)
        WS_B = testa_gguf(gguf, dev)
        WS_B3 = X3.TestaX3(WS_B) if a.testa_kl == "bf16x3" else WS_B
        ris = confronto(bB, bB3, WS_B, WS_B3, ca_rif, dev)
        rum = None
        if bR is not None:
            r = confronto(bB, bR, WS_B, WS_B, ca_rif, dev)
            rum = {"kl_B_R": r["kl_B_B3"], "dtheta": r["dtheta"], "h_err_rel": r["h_err_rel"],
                             "kl_teacher_studente": r["kl_teacher_studente"]["B3"],
                             "cambi_instradamento": r["cambi_instradamento"]["B3"], "secondi": bR["secondi"]}
            log(f"rumore {a.ricetta_rumore}: KL(B||R) media {r['kl_B_B3']['media']:.3e} (max "
                f"{r['kl_B_B3']['max']:.3e}); dtheta quota {r['dtheta']['quota_rotti']:.3e}, pesata "
                f"{r['dtheta']['quota_rotti_pesata']:.3e}, err. rel. norma {r['dtheta']['err_rel_norma']:.3e}")
        del WS_B, WS_B3
        esito.update({"strati_kl": strati_kl, **ris, "secondi_bracci": [bB["secondi"], bB3["secondi"]],
                      "trabocco_fp16": [bB["trabocco_fp16"], bB3["trabocco_fp16"]], "via_bf16x3": X3.stato()})
        if rum is not None:
            esito["rumore"].update(rum)
    vie = {v["via"] for v in X3.stato().values()}
    via_ok = a.device != "cuda" or vie <= {"triton", "out_dtype"}
    kl_ok = ris["kl_B_B3"]["media"] <= a.soglia_kl
    fedele_ass = bool(kl_ok and ris["dtheta"]["quota_rotti"] <= a.soglia_dtheta and via_ok)
    fedele_rum = None
    if rum is not None:
        rd, pd = rum["dtheta"], ris["dtheta"]
        f = a.fattore_rumore
        fedele_rum = bool(kl_ok and via_ok
                          and pd["quota_rotti"] <= max(a.soglia_dtheta, f * rd["quota_rotti"])
                          and pd["quota_rotti_pesata"] <= max(a.soglia_dtheta, f * rd["quota_rotti_pesata"])
                          and pd["err_rel_norma"] <= max(a.soglia_dtheta_norma, f * rd["err_rel_norma"]))
        esito["rapporti_sul_rumore"] = {
            "kl_media": ris["kl_B_B3"]["media"] / max(rum["kl_B_R"]["media"], 1e-300),
            "quota_rotti": pd["quota_rotti"] / max(rd["quota_rotti"], 1e-300),
            "quota_rotti_pesata": pd["quota_rotti_pesata"] / max(rd["quota_rotti_pesata"], 1e-300),
            "err_rel_norma": pd["err_rel_norma"] / max(rd["err_rel_norma"], 1e-300)}
    fedele = fedele_ass if fedele_rum is None else fedele_rum
    piu_veloce = all(c["rapporto"] < 1.0 for c in esito.get("tempo", {}).values()) if "tempo" in esito else None
    if piu_veloce and a.device == "cuda" and vie != {"triton"}:
        piu_veloce = False                    # la via "out_dtype" non ha WMMA: il tempo misurato non e' la ricetta
    esito["verdetto"] = {"fedele": bool(fedele), "fedele_assoluto": fedele_ass, "fedele_rumore": fedele_rum,
                         "piu_veloce": piu_veloce, "ammessa": bool(fedele and piu_veloce),
                         "vie_gemm_bf16": sorted(vie)}
    A._scrivi_json(usc / "esito_cancello_b3.json", esito)
    log(f"KL(B||B3) media {ris['kl_B_B3']['media']:.3e} per token (max {ris['kl_B_B3']['max']:.3e}, soglia "
        f"{a.soglia_kl}); dtheta nulli {ris['dtheta']['nulli_in_fp16']} + segno opposto "
        f"{ris['dtheta']['segno_sbagliato']} su {ris['dtheta']['vive_con_g32_non_nullo']} (quota "
        f"{ris['dtheta']['quota_rotti']:.2e}, soglia {a.soglia_dtheta}), err. rel. mediano "
        f"{ris['dtheta']['err_rel_mediano']:.2e}; KL(T||S) B {ris['kl_teacher_studente']['B']:.6f} "
        f"B3 {ris['kl_teacher_studente']['B3']:.6f}; via {sorted(vie)}; pesata {ris['dtheta']['quota_rotti_pesata']:.3e},"
        f" err. rel. norma {ris['dtheta']['err_rel_norma']:.3e}; sul rumore {esito.get('rapporti_sul_rumore')}")
    log(f"VERDETTO cancello B3: {json.dumps(esito['verdetto'])} -> {usc / 'esito_cancello_b3.json'}")
    return 0 if fedele else 1


if __name__ == "__main__":
    raise SystemExit(main())

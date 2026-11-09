#!/usr/bin/env python3
"""Cancello dei densi PIEGATI (1/10/2026, ramo densi-piegati) sulla GPU di TT5, NELLA CONFIGURAZIONE DI TT5 (R2B3m,
--densi-ternari tutte, teste bf16x3, E2E_GDN=veloce/gdn_veloce_corsa_v5, E2E_MOTORE=triton2). Lo lancia il
coordinatore al confine di passo di TT5 (`strumenti/cancello_densi_piegati.sh`), mai con TT5 viva.

Tre misure, un esito (`<uscita>/esito_cancello_piegati.json`):
1. **banco per famiglia** (`e2e_banco_densi_piegati`): ms di avanti / avanti+indietro / avanti+indietro in modo V2 per
   ogni famiglia densa vera a 3 piani, 8192 token, oggi contro piegato (x2 col kernel Triton, f32; hipBLASLt tolto: 23,9 s per attn_qkv); errori sul
   dispositivo e limiti deterministici su 512 token; proiezione s/passo P.
2. **tempo del passo** (`e2e_allena.misura`, strati 0,1 e 38,39): ms/token avanti/KL/indietro/Adam con l'interruttore
   spento e acceso (stessa ricetta di TT5).
3. **fedelta'** sui 40 strati (default): braccio di riferimento (spento), bracci RUMORE (Bk2_ind32 — coi densi ternari
   non perturba nulla — e, dal 2/10, «riordina»: la via di oggi coi piani dei densi sommati al contrario, cioe' un puro
   arrotondamento dei densi), braccio piegato per ogni modo; il pavimento e' il massimo dei due rumori. Il piegato cambia SOLO l'ordine/arrotondamento
   delle somme fp32 (e un arrotondamento fp16 in meno su dX): l'avanti NON e' al bit, quindi il criterio e' quello del
   cancello B3 contro il rumore: KL media <= --soglia-kl (0,0013) E <= max(1e-6, fattore x KL del rumore); dtheta
   (quota rotti, quota pesata, errore in norma) <= max(--soglia-dtheta 1e-3, fattore x rumore).
Verdetto per modo: fedele, piu_veloce (indietro E totale < 1 su entrambe le coppie di strati), ammesso = entrambe.
Uscita 0 se almeno un modo e' ammesso. Non cambia nessuna corsa.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

import torch

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
os.environ["E2E_DP_GPU"] = "1"
import e2e_allena as A  # noqa: E402
import e2e_cancello_b3 as CB  # noqa: E402
import e2e_cancello_indietro as CI  # noqa: E402
import e2e_densi_piegati as DP  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--teacher", default="@@ROOT:lavoro@@/e2e/teacher/q8-EB-pieno")
    ap.add_argument("--gguf", default="@@ROOT:models@@/gguf/tony-tern/R2B3m/Thor-T-R2B3m.gguf")
    ap.add_argument("--uscita", default="@@ROOT:depot@@/e2e/prove_tt5/cancello_densi_piegati")
    ap.add_argument("--ricetta", default="B3_ind32", help="la ricetta di TT5 (B3_ind32, o B3_ind2 se ammessa)")
    ap.add_argument("--modi", default="x2", help="modi piegati da provare, es. x2,f32")
    ap.add_argument("--gemm", default="triton", choices=("auto", "triton", "f32"))
    ap.add_argument("--strati-tempo", action="append", default=None, help="default: 0,1 e 38,39")
    ap.add_argument("--strati-kl", default="0-39")
    ap.add_argument("--righe-micro", type=int, default=A.ConfigAllena.righe_per_micro)
    ap.add_argument("--righe-held", type=int, default=2)
    ap.add_argument("--ripetizioni", type=int, default=3)
    ap.add_argument("--token-banco", type=int, default=8192)
    ap.add_argument("--soglia-kl", type=float, default=0.0013)
    ap.add_argument("--soglia-dtheta", type=float, default=1e-3)
    ap.add_argument("--fattore-rumore", type=float, default=2.0)
    ap.add_argument("--ricetta-rumore", default="Bk2_ind32")
    ap.add_argument("--salta-banco", action="store_true")
    ap.add_argument("--salta-tempo", action="store_true")
    ap.add_argument("--salta-fedelta", action="store_true")
    a = ap.parse_args(argv)
    import e2e_studente as S
    import e2e_bf16x3 as X3
    import e2e_banco_densi_piegati as BD
    gguf, usc = Path(a.gguf), Path(a.uscita)
    usc.mkdir(parents=True, exist_ok=True)
    log = A._registro(usc)
    modi = [m for m in a.modi.split(",") if m]
    for m in modi:
        if m not in ("x2", "f32"):
            raise SystemExit(f"modo {m}")
    strati_tempo = [CB._strati(s) for s in (a.strati_tempo or ["0,1", "38,39"])]
    ca = A.ConfigAllena(righe_per_micro=a.righe_micro, densi_ternari=A._famiglie_dense("tutte"))
    if S.RICETTE[a.ricetta].non_esperti_x3:
        ca = dataclasses.replace(ca, testa_kl="bf16x3", logit_maestro="bf16")
    gdn = A.scegli_gdn(log)
    esito = {"data": A.adesso(), "argv": sys.argv, "gguf": str(gguf), "motore": CI.MOTORE, "gdn": gdn,
             "gdn_modulo": os.environ.get("E2E_GDN_MODULO"), "ricetta": a.ricetta, "modi": modi, "gemm": a.gemm,
             "soglie": {"kl": a.soglia_kl, "dtheta": a.soglia_dtheta, "fattore_rumore": a.fattore_rumore}}
    dest = usc / "esito_cancello_piegati.json"
    log(f"cancello densi piegati: motore {CI.MOTORE}, GDN {gdn}, ricetta {a.ricetta}, modi {modi}, gemm {a.gemm}")
    DP.imposta(modo="", gemm=a.gemm)
    with A._coesistenza_gpu(True):
        A._guardia_gpu(600)
        if not a.salta_banco:
            try:
                esito["banco"] = BD.banco(gguf, usc, a.token_banco, 5, (a.ricetta,), log)
            except Exception as ex:
                esito["banco"] = {"errore": repr(ex)[:500]}
                log(f"banco fallito: {ex!r}")
            DP.imposta(modo="", gemm=a.gemm)
            A._scrivi_json(dest, esito)
        if not a.salta_tempo:
            tempo = {}
            for m in [""] + modi:
                DP.imposta(modo=m)
                tempo[m or "spento"] = A.misura(strati_tempo, Path(a.teacher), gguf, ca, a.ripetizioni, "cuda",
                                                a.ricetta, log=log, disponibili=True)["configurazioni"]
            DP.imposta(modo="")
            rif = tempo["spento"]
            esito["tempo"] = {k: {s: {"totale_ms_token": v["totale_ms_token"], "parti": v["ms_token"],
                                      "picco_gib": v["picco_torch_gib"],
                                      "rapporto_su_spento": v["totale_ms_token"] / rif[s]["totale_ms_token"],
                                      "rapporto_indietro_su_spento": v["ms_token"]["indietro"]
                                      / rif[s]["ms_token"]["indietro"],
                                      "rapporto_avanti_su_spento": v["ms_token"]["avanti"] / rif[s]["ms_token"]["avanti"]}
                                  for s, v in conf.items()} for k, conf in tempo.items()}
            for k, conf in esito["tempo"].items():
                for s, v in conf.items():
                    log(f"tempo {k} strati {s}: {v['totale_ms_token']:.4f} ms/token (x{v['rapporto_su_spento']:.3f}; "
                        f"avanti x{v['rapporto_avanti_su_spento']:.3f}, indietro x{v['rapporto_indietro_su_spento']:.3f})")
            A._scrivi_json(dest, esito)
        if not a.salta_fedelta:
            src = A.SorgenteCache(a.teacher)
            righe_train = A.prime_disponibili(src, "train", a.righe_micro)
            righe_held = A.prime_disponibili(src, "heldout", a.righe_held)
            esito.update({"righe_train": righe_train, "righe_held": righe_held})
            strati_kl = CB._strati(a.strati_kl)
            dev = torch.device("cuda")
            DP.imposta(modo="")
            bR = CI.braccio(a.ricetta, strati_kl, src, righe_held, righe_train, ca, gguf, "cuda", log)
            WS = CB.testa_gguf(gguf, dev)
            WS_R = X3.TestaX3(WS) if ca.testa_kl == "bf16x3" else WS
            # come cancello_indietro: il braccio rumore (Bk2_ind32, non-esperti fp32 a K diviso) con le teste fp32
            ca_n = ca if S.RICETTE[a.ricetta_rumore].non_esperti_x3 else \
                dataclasses.replace(ca, testa_kl="fp32", logit_maestro="fp32")
            bN = CI.braccio(a.ricetta_rumore, strati_kl, src, righe_held, righe_train, ca_n, gguf, "cuda", log)
            WS_N = X3.TestaX3(WS) if ca_n.testa_kl == "bf16x3" else WS
            rum = CB.confronto(bR, bN, WS_R, WS_N, ca, dev)
            esito["rumore"] = {"ricetta": a.ricetta_rumore, "kl": rum["kl_B_B3"], "dtheta": rum["dtheta"],
                               "h_err_rel": rum["h_err_rel"]}
            del bN
            # 2/10: coi densi ternari Bk2_ind32 non perturba NULLA (le proiezioni non-esperte sono tutte LineareTernaria:
            # KL ~0, h al bit, gate del 1/10). Il pavimento vero e' un arrotondamento dei DENSI: la via di oggi coi piani
            # sommati al contrario (DP modo "riordina"), stessi kernel, solo somme fp32/fp16 in altro ordine.
            DP.imposta(modo="riordina")
            DP.CONTI.update(piegati=0, ripiego=0)
            bO = CI.braccio(a.ricetta, strati_kl, src, righe_held, righe_train, ca, gguf, "cuda", log)
            DP.imposta(modo="")
            rum2 = CB.confronto(bR, bO, WS_R, WS_R, ca, dev)
            esito["rumore_densi"] = {"modo": "riordina", "kl": rum2["kl_B_B3"], "dtheta": rum2["dtheta"],
                                     "h_err_rel": rum2["h_err_rel"], "chiamate": dict(DP.CONTI)}
            log(f"rumore densi (riordina): KL media {rum2['kl_B_B3']['media']:.3e}, h_err {rum2['h_err_rel']:.3e}, "
                f"dtheta quota {rum2['dtheta']['quota_rotti']:.2e} pesata {rum2['dtheta']['quota_rotti_pesata']:.2e} "
                f"norma {rum2['dtheta']['err_rel_norma']:.2e}")
            del bO
            A._scrivi_json(dest, esito)
            esito["fedelta"] = {}
            f = a.fattore_rumore
            rk = max(rum["kl_B_B3"]["media"], rum2["kl_B_B3"]["media"])
            rd = {k: max(rum["dtheta"][k], rum2["dtheta"][k]) for k in ("quota_rotti", "quota_rotti_pesata",
                                                                        "err_rel_norma")}
            for m in modi:
                DP.imposta(modo=m)
                DP.CONTI.update(piegati=0, ripiego=0)
                bC = CI.braccio(a.ricetta, strati_kl, src, righe_held, righe_train, ca, gguf, "cuda", log)
                conti = dict(DP.CONTI)
                DP.imposta(modo="")
                r = CB.confronto(bR, bC, WS_R, WS_R, ca, dev)
                pd = r["dtheta"]
                fedele = bool(r["kl_B_B3"]["media"] <= min(a.soglia_kl, max(1e-6, f * rk))
                              and pd["quota_rotti"] <= max(a.soglia_dtheta, f * rd["quota_rotti"])
                              and pd["quota_rotti_pesata"] <= max(a.soglia_dtheta, f * rd["quota_rotti_pesata"])
                              and pd["err_rel_norma"] <= max(a.soglia_dtheta, f * rd["err_rel_norma"])
                              and conti["piegati"] > 0 and conti["ripiego"] == 0)
                esito["fedelta"][m] = {**r, "fedele": fedele, "chiamate": conti, "secondi": bC["secondi"],
                                       "trabocco_fp16": bC["trabocco_fp16"]}
                log(f"fedelta' piegato {m}: KL media {r['kl_B_B3']['media']:.3e} (rumore {rk:.3e}), dtheta quota "
                    f"{pd['quota_rotti']:.2e} pesata {pd['quota_rotti_pesata']:.2e} norma {pd['err_rel_norma']:.2e} "
                    f"(rumore {rd['err_rel_norma']:.2e}), chiamate {conti} -> fedele {fedele}")
                del bC
                A._scrivi_json(dest, esito)
            del bR, WS
    verd = {}
    for m in modi:
        fed = esito.get("fedelta", {}).get(m, {}).get("fedele")
        t = esito.get("tempo", {}).get(m)
        veloce = None if t is None else all(v["rapporto_su_spento"] < 1.0 and v["rapporto_indietro_su_spento"] < 1.0
                                            for v in t.values())
        verd[m] = {"fedele": fed, "piu_veloce": veloce, "ammesso": bool(fed and veloce)}
    esito["verdetto"] = verd
    A._scrivi_json(dest, esito)
    log(f"VERDETTO cancello densi piegati: {json.dumps(verd)} -> {dest}")
    return 0 if any(v["ammesso"] for v in verd.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""e1_profila_passo.py — quanto costa DAVVERO un passo di E1, in funzione dei token (R007, ~10 min di GPU).

Costruisce lo studente come `e1_recupero_lettera.py --modo residui`, installa i residui negli strati chiesti e, per ogni
taglia di lotto (token), misura: prefisso (strati < primo allenato), avanti da quello strato, indietro troncato.
Adatta tempo = a + b * token (e1_veloce.regressione) e dice se il passo e' LEGATO AI PESI (a domina: conviene impacchettare
piu' gruppi per passo) o AI TOKEN, e quanti passi stanno nel budget. Non allena niente, non scrive pesi.

  E2E_GDN=veloce E2E_GDN_MODULO=gdn_veloce_corsa_v5 E2E_MOTORE=triton2 E2E_CONV_DIRETTA=1 E2E_SENZA_COESISTENZA=1 \\
  E2E_CACHE_DEQUANT=@@ROOT:depot@@/e2e/cache_dequant E2E_DENSI_PIEGATI=x2 E2E_PIEGA_GEMM=triton \\
  python e1_profila_passo.py --dati items_teacher.jsonl --uscita OUT/profilo --token 1024,3072,8192,16384,32768
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
sys.path.insert(1, str(QUI / "recupero_7dabd6f" / "thor_v2"))
import e1_veloce as V                                    # noqa: E402
import e1_recupero_lettera as DRV                        # noqa: E402
import residui_ternari as RT                             # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dati", required=True)
    ap.add_argument("--uscita", required=True)
    ap.add_argument("--gguf", default=DRV.GGUF)
    ap.add_argument("--tokenizer", default=DRV.TOKENIZER)
    ap.add_argument("--token", default="1024,3072,8192,16384,32768", help="taglie di lotto da misurare (token)")
    ap.add_argument("--residui-strati", default="12-27")
    ap.add_argument("--residui-famiglie", default="attn,gdn")
    ap.add_argument("--residui-rango", type=int, default=32)
    ap.add_argument("--ripetizioni", type=int, default=2)
    ap.add_argument("--budget-ore", type=float, default=3.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ricetta", default="B3_ind2")
    a = ap.parse_args(argv)
    import e2e_studente as S
    import e2e_allena as A
    out = Path(a.uscita)
    out.mkdir(parents=True, exist_ok=True)

    def log(*m):
        print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}]", *m, flush=True)

    dev = torch.device(a.device)
    lids = list(DRV.LETTER_IDS)
    enc = DRV.codificatore(a.tokenizer, lids)
    righe = [r for r in DRV.read_jsonl(a.dati) if "teacher_logprobs" in r and not DRV.fonte_vietata(r.get("source", ""))]
    righe = righe[:4000]
    for r in righe:
        r["ids"] = enc(r["prompt"])
    A.scegli_gdn(log)
    st = S.Studente(gguf=Path(a.gguf), strati=range(40), ricetta=a.ricetta,
                    motore_nome=os.environ.get("E2E_MOTORE", "triton"), device=a.device, log=log, densi_ternari="tutte")
    for p in st.parametri():
        p.requires_grad_(False)
    cfg = RT.ConfigResidui(rango=a.residui_rango, strati=DRV.intervallo(a.residui_strati),
                           famiglie=RT.famiglie_da(a.residui_famiglie))
    res = RT.installa(st, cfg, device=dev)
    for m in res.values():
        m.requires_grad_(True)
        with torch.no_grad():
            m.gamma.fill_(1e-3)                            # gamma > 0: i gradienti di A e B non sono nulli
    st.strati_allenati = RT.strati_di(res)
    L0 = V.primo_strato_allenato(st)
    toks = [int(x) for x in a.token.split(",")]
    log(f"profilo: primo strato allenato {L0}, residui in {len(RT.strati_di(res))} strati, taglie {toks}")
    p = V.profila(st, righe, toks, dev, DRV.impacchetta, L0, ripetizioni=a.ripetizioni)
    p["consiglio"] = V.consiglia(p, a.budget_ore * 3600)
    p["ambiente"] = {k: v for k, v in os.environ.items() if k.startswith("E2E_")}
    json.dump(p, open(out / "profilo.json", "w"), indent=1)
    for m in p["misure"]:
        log(f"{m['token']:>6} token: prefisso {m['prefisso_s']:.1f} s, avanti {m['avanti_s']:.1f} s, indietro {m['indietro_s']:.1f} s")
    log("consiglio:", json.dumps(p["consiglio"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""dθ del primo micro-lotto nel percorso VERO della corsa, GDN ripiego contro GDN fla (27/9/2026).

Stesso Studente della corsa (e2e_studente, ricetta B_ind32, motore triton), strati 0,1 (GDN), stessa sorgente del
teacher, stesso micro-lotto, K = 2^19 (quella di EB2, dal suo registro). Un gradiente_passo (e2e_allena) col ripiego,
uno con fla, stessi pesi. Confronto come il cancello G1: tra le scale vive (d0 != 0) col gradiente di riferimento
(ripiego) non nullo, quota di dθ nulli o di segno opposto con fla; soglia 1e-3. Piu' errore relativo mediano/p99.
Niente scritture: nessun passo di Adam, nessun file della corsa toccato.
Uso (dalla cartella thor_v2): python3 strumenti/fla_gfx1151/prova_corsa.py --teacher @@ROOT:lavoro@@/e2e/teacher/q8-EB-pieno"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

QUI = Path(__file__).resolve().parent
THOR = QUI.parent.parent
sys.path.insert(0, str(THOR))
sys.modules["fla"] = None
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M  # noqa: E402
sys.path.insert(0, str(QUI))
import gdn_fla_gfx1151 as G  # noqa: E402
import e2e_allena as A  # noqa: E402
import e2e_studente as S  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--strati", default="0,1")
    ap.add_argument("--K", type=float, default=2.0 ** 19)
    ap.add_argument("--esito", default=str(QUI / "esiti" / "prova_corsa.json"))
    a = ap.parse_args()
    r = G.abilita_sicuro(M)
    print(f"abilita_sicuro: {r}", flush=True)
    if not r["attivo"]:
        return 2
    gdn = M.torch_chunk_gated_delta_rule
    dev = torch.device("cuda")
    src = A.SorgenteCache(Path(a.teacher))
    ca = A.ConfigAllena()
    righe = [i for i in src.righe_previste("train") if src.disponibile("train", i)][:ca.righe_per_micro]  # 0-255 consumate
    A.attendi_righe(src, "train", righe, print)
    strati = [int(x) for x in a.strati.split(",")]
    st = S.Studente(gguf=S.GGUF_THOR_T, strati=strati, ricetta="B_ind32", motore_nome="triton", device="cuda",
                    log=print)
    st.carica_embedding()
    WS = st.testa(torch.float32)
    WT, _ = A.testa_maestro(dev)
    car = A.Caricatore(src, dev)
    grad, tempo, kl = {}, {}, {}
    for giro in range(2):                                  # giro 0: compilazione/riscaldamento; giro 1: misura
        for nome, impl in (("ripiego", gdn.ripiego), ("fla", gdn)):
            M.torch_chunk_gated_delta_rule = impl
            A._azzera(st)
            torch.cuda.synchronize()
            t = time.perf_counter()
            k_ = A.gradiente_passo(st, car, "train", righe, ca, WS, WT, a.K)
            torch.cuda.synchronize()
            tempo[nome] = time.perf_counter() - t
            kl[nome] = float(k_)
            grad[nome] = A._copia_grad(st)
    M.torch_chunk_gated_delta_rule = gdn
    vive_nz = nulli = rotti = 0
    rel = []
    for (L, n), gf, gr in zip(st.chiavi(), grad["fla"], grad["ripiego"]):
        v = (st.esperti(L).fam[n].d0 != 0) & (gr != 0)
        vive_nz += int(v.sum())
        nulli += int((v & (gf == 0)).sum())
        rotti += int((v & (gf != 0) & (torch.sign(gf) != torch.sign(gr))).sum())
        rr = ((gf - gr).abs() / gr.abs())[v]
        if rr.numel():
            rel.append(rr.float())
    rel_t = torch.cat(rel) if rel else torch.zeros(1)
    q = torch.quantile(rel_t[torch.randperm(rel_t.numel(), device=rel_t.device)[:10_000_000]],
                       torch.tensor([0.5, 0.99], device=rel_t.device))
    norme = [float((torch.cat([g.flatten() for g in grad["fla"]]) - torch.cat([g.flatten() for g in grad["ripiego"]])
                    ).norm() / torch.cat([g.flatten() for g in grad["ripiego"]]).norm())]
    ntok = len(righe) * car.prendi("train", righe).ids.shape[1]
    esito = {"strati": strati, "righe": righe, "token": ntok, "K": a.K, "kl_somma": kl,
             "tempo_passo_s": tempo, "ms_token": {k: 1000 * v / ntok for k, v in tempo.items()},
             "vive_con_g_non_nullo": vive_nz, "nulli_fla": nulli, "segno_opposto_fla": rotti,
             "quota_rotti": (nulli + rotti) / max(vive_nz, 1), "err_rel_mediano": float(q[0]),
             "err_rel_p99": float(q[1]), "err_rel_norma_totale": norme[0], "stato_involucro": dict(gdn.stato)}
    esito["cancello_1e-3"] = esito["quota_rotti"] <= 1e-3
    print(json.dumps(esito, indent=1), flush=True)
    Path(a.esito).write_text(json.dumps(esito, indent=1))
    car.chiudi()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

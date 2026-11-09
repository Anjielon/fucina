#!/usr/bin/env python3
"""`e2e_allena.misura` (ms/token del passo intero per parti, strati veri) col GDN ripiego o fla (27/9/2026).
Identica alla sottocomando `misura` salvo le righe: misura usa righe_previste("train")[:righe_per_micro], ma il teacher
q8-EB-pieno ruota e le righe 0-255 sono gia' consumate (attesa infinita); qui le prime righe DISPONIBILI.
Uso (da thor_v2): python3 strumenti/fla_gfx1151/prova_misura.py --gdn fla|ripiego --uscita F.json [--strati 0 --strati 0,1]"""
import argparse
import json
import sys
from pathlib import Path

QUI = Path(__file__).resolve().parent
THOR = QUI.parent.parent
sys.path.insert(0, str(THOR))
sys.modules["fla"] = None
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M  # noqa: E402
sys.path.insert(0, str(QUI))
import e2e_allena as A  # noqa: E402
import e2e_studente as S  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gdn", choices=("fla", "ripiego"), required=True)
    ap.add_argument("--teacher", default="@@ROOT:lavoro@@/e2e/teacher/q8-EB-pieno")
    ap.add_argument("--strati", action="append", default=None)
    ap.add_argument("--ripetizioni", type=int, default=2)
    ap.add_argument("--ricetta", default="B_ind32")
    ap.add_argument("--uscita", required=True)
    a = ap.parse_args()
    if a.gdn == "fla":
        import gdn_fla_gfx1151 as G
        r = G.abilita_sicuro(M)
        print(f"GDN fla: {r}", flush=True)
        if not r["attivo"]:
            return 2
    originale = A.SorgenteCache.righe_previste

    def disponibili_prima(self, lato):
        tutte = originale(self, lato)
        return [i for i in tutte if self.disponibile(lato, i)] + [i for i in tutte if not self.disponibile(lato, i)]
    A.SorgenteCache.righe_previste = disponibili_prima
    strati = [[int(x) for x in s.split(",")] for s in (a.strati or ["0", "0,1"])]
    ris = A.misura(strati, Path(a.teacher), S.GGUF_THOR_T, A.ConfigAllena(), a.ripetizioni, "cuda", a.ricetta)
    conf = ris["configurazioni"]
    if "0" in conf and "0,1" in conf:
        ris["gdn_ms_token"] = conf["0,1"]["totale_ms_token"] - conf["0"]["totale_ms_token"]
    ris["gdn"] = a.gdn
    g = M.torch_chunk_gated_delta_rule
    ris["stato_involucro"] = dict(g.stato) if hasattr(g, "stato") else None
    Path(a.uscita).write_text(json.dumps(ris, indent=1, default=str))
    print(json.dumps(ris, indent=1, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

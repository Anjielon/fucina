#!/usr/bin/env python3
"""Ricerca lineare lungo la direzione VERA di discesa di EB2 (26/9/2026 sera): quanto conviene un lr piu' grande.
theta = 0 all'inizio (d = d0 exp(theta)); dopo 2 passi con lr 1e-3 la corsa ha theta2 (istantanea migliore_000002).
Qui si valuta la KL held-out (sulle stesse righe, 1 su 4) in theta = alfa * theta2 per piu' alfa: e' la KL che si avrebbe
facendo gli stessi 2 passi con lr alfa volte piu' grande (a meno dell'effetto sui gradienti dei passi successivi).
Se la discesa cresce ~con alfa, il lr e' il collo di bottiglia della consegna. Solo avanti, nessun parametro salvato.
Lanciare come unita' systemd --user (la shell ha HSA_OVERRIDE_GFX_VERSION sbagliato), accanto a EB2 (GPU_ESCLUDI_PID)."""
import json
import sys
from pathlib import Path

import torch

QUI = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(QUI))
import e2e_allena as A  # noqa: E402
import e2e_studente as S  # noqa: E402

TEACHER = Path("@@ROOT:lavoro@@/e2e/teacher/q8-EB-pieno")
ISTANTANEA = Path(sys.argv[1] if len(sys.argv) > 1 else "@@ROOT:lavoro@@/e2e/allena/EB2/migliore_000002")
ALFE = [float(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0,1,3,6,12").split(",")]
USCITA = Path("@@ROOT:lavoro@@/e2e/allena/sonda_lr.json")


def main() -> int:
    dev = torch.device("cuda")
    src = A.SorgenteCache(str(TEACHER))
    held = src.righe_previste("heldout")[::4]
    car = A.Caricatore(src, dev)
    st = S.Studente(gguf=S.GGUF_THOR_T, strati=range(40), ricetta="B", motore_nome="triton", device="cuda", log=print)
    st.carica_embedding()
    WS = st.testa(torch.float32)
    WT, _ = A.testa_maestro(dev)
    theta2 = torch.load(ISTANTANEA / "theta.pt", map_location="cpu")
    ca = A.ConfigAllena()
    ris = {"istantanea": str(ISTANTANEA), "righe": held, "alfa": {}}
    for a in ALFE:
        with torch.no_grad():
            for k, p in zip(st.chiavi(), st.parametri()):
                p.copy_(theta2[A._chiave(k)].to(p.device) * a)
        st.azzera_cambi() if hasattr(st, "azzera_cambi") else None
        v = A.valuta(st, car, held, ca, WS, WT)
        passo = max(float(t.abs().max()) for t in theta2.values()) * a
        ris["alfa"][str(a)] = {"kl": v["kl"], "metrica": v["metrica"], "kl_per_dominio": v["kl_per_dominio"],
                                "max_abs_theta": passo}
        print(f"alfa {a:g}: KL held-out {v['kl']:.5f}  metrica {v['metrica']:.5f}  max|theta| {passo:.4f}  "
              + " ".join(f"{d} {x:.4f}" for d, x in v["kl_per_dominio"].items()), flush=True)
        USCITA.write_text(json.dumps(ris, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

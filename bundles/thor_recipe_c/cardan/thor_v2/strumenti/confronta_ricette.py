#!/usr/bin/env python3
"""Ricetta B contro ricetta C dello studente Thor-T, sulle stesse righe held-out (26/9/2026 sera).
C = esperti identici a B, ma residuo e non-esperti (attenzione, GDN, esperto condiviso) in bf16: misurata -24% per strato GDN.
Si passa a C solo se e' EQUIVALENTE a B per l'addestramento: KL(B || C) per token molto sotto la KL(teacher || B) che
l'allenamento deve ridurre (stesso schema del criterio F0b: <= 1/10). Stampa KL(T||B), KL(T||C), KL(B||C) per dominio.
Va lanciato come unita' systemd --user (la shell interattiva ha HSA_OVERRIDE_GFX_VERSION sbagliato)."""
import json
import sys
from pathlib import Path

import torch

QUI = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(QUI))
import e2e_allena as A  # noqa: E402
import e2e_studente as S  # noqa: E402

TEACHER = Path(sys.argv[1] if len(sys.argv) > 1 else "@@ROOT:lavoro@@/e2e/teacher/q8-EB-pieno")
USCITA = Path(sys.argv[2] if len(sys.argv) > 2 else "@@ROOT:lavoro@@/e2e/fedelta/B_contro_C.json")
PASSO_RIGHE = int(sys.argv[3]) if len(sys.argv) > 3 else 4        # 1 riga su 4 del held-out: 16 righe, tutti i domini


def main() -> int:
    dev = torch.device("cuda")
    src = A.SorgenteCache(str(TEACHER))
    held = src.righe_previste("heldout")[::PASSO_RIGHE]
    car = A.Caricatore(src, dev)
    st = {}
    for r in ("B", "C"):
        st[r] = S.Studente(gguf=S.GGUF_THOR_T, strati=range(40), ricetta=r, motore_nome="triton", device="cuda", log=print)
        st[r].carica_embedding()
    WS = st["B"].testa(torch.float32)
    WT, _ = A.testa_maestro(dev)
    somme: dict = {}
    for j in range(0, len(held), 4):
        m = held[j:j + 4]
        lot = car.prendi("heldout", m)
        bers = A._bersagli(lot.ids, lot.inizi)
        with torch.no_grad():
            hB = st["B"].nascosto(lot.ids, inizi=lot.inizi)
            hC = st["C"].nascosto(lot.ids, inizi=lot.inizi).float()   # KL in fp32 per entrambe (C esce bf16)
            Tn = lot.ids.shape[1]
            for nome, (hs, ws, ht, wt) in {"T||B": (hB, WS, lot.hT, WT), "T||C": (hC, WS, lot.hT, WT),
                                           "B||C": (hC, WS, hB, WS)}.items():
                kl = torch.cat([A.kl_pezzo(hs[i:i + 512], ws, ht[i:i + 512], wt, False, bers[i:i + 512])["kl"]
                                for i in range(0, hs.shape[0], 512)]).view(len(m), Tn).sum(1).cpu().tolist()
                for d, v in zip(lot.domini, kl):
                    s = somme.setdefault(nome, {}).setdefault(d, [0.0, 0])
                    s[0] += v
                    s[1] += Tn
        print(f"righe {m}: fatto", flush=True)
    ris = {n: {d: s / c for d, (s, c) in sorted(dd.items())} for n, dd in somme.items()}
    for n, dd in somme.items():
        ris[n]["_tutti"] = sum(s for s, _ in dd.values()) / sum(c for _, c in dd.values())
    ris["rapporto_BC_su_TB"] = ris["B||C"]["_tutti"] / ris["T||B"]["_tutti"]
    ris["righe"] = held
    USCITA.parent.mkdir(parents=True, exist_ok=True)
    USCITA.write_text(json.dumps(ris, indent=1))
    print(json.dumps({k: v for k, v in ris.items() if k != "righe"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

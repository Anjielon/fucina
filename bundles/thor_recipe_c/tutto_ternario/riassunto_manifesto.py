"""Riassunto per famiglia dal manifesto: errore relativo (non pesato, pesato-H), catena, tempi, byte."""
import json
import sys
from collections import defaultdict

import numpy as np

man = json.load(open("@@ROOT:depot@@/tutto_ternario/tensori/manifesto.json"))
fam = defaultdict(list)
for nome, m in man.items():
    f = nome.split(".")[2] if nome.startswith("blk.") else nome.split(".")[0]
    fam[f].append(m)
righe = []
for f, ms in sorted(fam.items()):
    er = np.array([m["misure"]["err_rel"] for m in ms])
    ep = [m["misure"].get("err_rel_pesato_H") for m in ms if m["misure"].get("err_rel_pesato_H") is not None]
    en = [m["misure"].get("rif_nonpesato_err_rel_pesato_H") for m in ms if m["misure"].get("rif_nonpesato_err_rel_pesato_H") is not None]
    ca = [(m["misure"]["loss_pesata_rel_catena"]["gptq"], m["misure"]["loss_pesata_rel_catena"]["finale"])
          for m in ms if "loss_pesata_rel_catena" in m["misure"]]
    s = sum(m["misure"]["secondi"] for m in ms)
    rt = max(m["misure"]["roundtrip_max_abs"] for m in ms)
    lora = sum(1 for m in ms if m["sorgente"].get("delta_lora"))
    riga = (f"{f:16s} n={len(ms):3d} {ms[0]['misure']['formato']:11s} err_rel {er.mean():.4f} [{er.min():.4f}-{er.max():.4f}]"
            + (f" · pesato-H {np.mean(ep):.4f} (ottimo non pesato nella stessa metrica {np.mean(en):.4f})" if ep else " · senza Hessiana")
            + (f" · catena GPTQ {np.mean([a for a, _ in ca]):.4f} -> finale {np.mean([b for _, b in ca]):.4f}" if ca else "")
            + f" · LoRA {lora}/{len(ms)} · round-trip max {rt:.0e} · {s:.0f} s")
    righe.append(riga)
print("\n".join(righe))
print(f"tensori: {len(man)}, secondi totali {sum(m['misure']['secondi'] for m in man.values()):.0f}")

#!/usr/bin/env python3
"""Tabella intera della dinamica di LoTA sui giocattoli (LEGGIMI_lota-qaf.md §4), CPU. Le righe principali sono anche in
e2e_test_lota.py T4 (stesse funzioni e stessi semi: stessi numeri); qui in piu' tolleranze, finestre e tetti.
Uso: nice -n 19 env OMP_NUM_THREADS=4 python3 strumenti/lota_giocattoli.py [passi=40] > uscita.jsonl"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import e2e_lota as LT  # noqa: E402
import e2e_test_lota as T  # noqa: E402

torch.set_num_threads(4)
passi = int(sys.argv[1]) if len(sys.argv) > 1 else 40
fs0 = T.famiglia_sintetica(E=4, R=64, C=512, seme=61, std=1.0 / math.sqrt(512))
ft, _ = T.maestro_raggiungibile(fs0, r=4, omega=1.0, seme=62)
g = np.random.default_rng(81)
W = (g.standard_normal((4, 64, 512)) / math.sqrt(512)).astype(np.float32)
fq, Wt = T.fam_da_pesi(W), torch.from_numpy(W)
kw = dict(rango=8, omega=2.0, top_p=0.2, top_p_fine=0.02, top_p_coda=0.005)
kwf = dict(rango=8, omega=2.0, top_p=0.05)
casi = [
    ("raggiungibile", "grad_senza_sonda", LT.ConfigLoTA(**kw, init="gradiente"), False),
    ("raggiungibile", "paper_senza_sonda", LT.ConfigLoTA(**kw), False),
    ("raggiungibile", "grad_sonda_stretta", LT.ConfigLoTA(**kw, init="gradiente"), True),
    ("raggiungibile", "paper_sonda_stretta", LT.ConfigLoTA(**kw), True),
    ("raggiungibile", "grad_w4_sonda_stretta", LT.ConfigLoTA(**{**kw, "omega": 4.0}, init="gradiente"), True),
    ("raggiungibile", "paper_w075r_sonda_stretta", LT.ConfigLoTA(**{**kw, "omega": None}), True),
] + [("raggiungibile", f"grad_tolleranza_{t:g}", LT.ConfigLoTA(**kw, init="gradiente", tolleranza=t), True)
     for t in (0.05, 0.1, 0.2)] + [
    ("raggiungibile", f"grad_finestra_{f}", LT.ConfigLoTA(**kw, init="gradiente", finestra=f), True) for f in (8, 12, 16)] + [
    ("raggiungibile", f"grad_finestra_12_tetto_{t:g}", LT.ConfigLoTA(**kw, init="gradiente", finestra=12, tolleranza=t), True)
    for t in (0.1, 0.2)] + [
    ("continuo", "paper_w05r_senza_sonda", LT.ConfigLoTA(**{**kwf, "omega": 4.0}), False),
    ("continuo", "paper_w025r_senza_sonda", LT.ConfigLoTA(**kwf), False),
    ("continuo", "grad_senza_sonda", LT.ConfigLoTA(**kwf, init="gradiente"), False),
    ("continuo", "grad_sonda_stretta", LT.ConfigLoTA(**kwf, init="gradiente"), True),
    ("continuo", "paper_w075r_senza_sonda", LT.ConfigLoTA(**{**kwf, "omega": None}), False),
] + [("continuo", f"grad_finestra_{f}", LT.ConfigLoTA(**kwf, init="gradiente", finestra=f), True) for f in (8, 12)] + [
    ("continuo", "grad_finestra_12_tetto_0.2", LT.ConfigLoTA(**kwf, init="gradiente", finestra=12, tolleranza=0.2), True)]
for maestro, nome, cfg, sonda in casi:
    t0 = time.time()
    fam, mae = (fs0, ft) if maestro == "raggiungibile" else (fq, Wt)
    r = T._corsa_giocattolo(fam, mae, passi, cfg, sonda=sonda)
    c = r["curva"]
    r["picco"], r["passo_del_picco"] = max(c), c.index(max(c))
    r["sotto_l_iniziale_dal_passo"] = next((j for j in range(1, len(c)) if all(x < c[0] for x in c[j:])), None)
    r["secondi"] = round(time.time() - t0, 1)
    print(json.dumps({"maestro": maestro, "caso": nome, **r}), flush=True)

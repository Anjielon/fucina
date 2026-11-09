#!/usr/bin/env python3
"""24/10: scrive i ffn_down_exps con le scale distillate: scala = fp16(d0 * exp(theta)) (come StratoEsperti._scala),
trit invariati, dall'ospite GGUF. uso: scale_down_esporta.py OSPITE.gguf scale_<passo>.pt DIR_TENSORI"""
import sys, numpy as np, torch
sys.path.insert(0, "@@ROOT:forge@@/sorgenti_forgia/raffina"); import comune as CM
gp = CM.gp
osp, thf, td = sys.argv[1:4]
import os; os.makedirs(td, exist_ok=True)
th = torch.load(thf, map_location="cpu")
for L, t in sorted(th.items()):
    src = gp.GGUFPlaneSource(osp, layers=[L]); en = src.entry(gp.TensorKey(L, 0, "down"))
    raw = np.array(en.tensors[0].data).reshape(en.n_experts, en.out, -1).copy()
    t = t.double()
    for e in range(en.n_experts):
        tt = src.load(gp.TensorKey(L, e, "down"))
        d0 = tt.scales[0].double()
        s = (d0 * torch.exp(t[e].reshape(d0.shape))).to(torch.float16)
        raw[e] = gp.encode_rows(tt.planes[0].numpy().astype(np.int8), s.numpy(), en.fmt.type_id)
    np.save(f"{td}/blk.{L}.ffn_down_exps.weight.npy", raw)
    print(f"L{L}: theta rms {float(t.pow(2).mean().sqrt()):.5f}", flush=True)

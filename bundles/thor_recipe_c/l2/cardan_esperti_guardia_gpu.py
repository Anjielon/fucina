#!/usr/bin/env python3
"""Unita' esperti = strato x famiglia; stage privato del pianificatore."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time


def crea(config, dev, CP, CM):
    import numpy as np
    import torch
    import cardan_guardia_gpu as G
    from rd_ternary.formato import TensorKey
    checkpoint = torch.load(config["checkpoint"], mmap=True, map_location="cpu", weights_only=False)
    dt = getattr(torch, config["dtype"])
    cache = {}

    def lavora(dati, out):
        torch.cuda.reset_peak_memory_stats(dev)
        L, f = dati["strato"], dati["famiglia"]
        if cache.get("strato") != L:
            cache.clear()
            cache.update(strato=L, hs=CP.Hessiane(L), src=CM.gp.GGUFPlaneSource(
                config["gguf"], layers=[L], bases=["ffn_gate_exps", "ffn_up_exps"]))
        hs, src = cache["hs"], cache["src"]
        stato = checkpoint["stato"].get(L, checkpoint["stato"].get(str(L)))
        if stato is None:
            raise ValueError(f"checkpoint senza strato {L}")
        t0 = time.monotonic()
        entry = src.entry(TensorKey(L, 0, f))
        if len(entry.tensors) > 2:
            raise ValueError("il riferimento esperti gestisce al massimo 2 piani")
        raws = [np.array(t.data, copy=True).reshape(int(t.shape[2]), entry.out, -1)
                for t in entry.tensors]
        D, C = stato[f"D_{f}"].double(), stato[f"C_{f}"].double()
        G.verifica_finito(D, "D")
        G.verifica_finito(C, "C")
        gpu_correzione = config.get("correzione_gpu", False)
        Dg = D.to(dev) if gpu_correzione else None
        per_k = {}
        for e in range(entry.n_experts):
            if float(C[e].abs().max()) != 0:
                per_k.setdefault(entry.n_planes_of(e), []).append(e)
        tot = {"dW": 0.0, "fold": 0.0, "scale_sole": 0.0, "esperti": 0, "fonti": {}}

        def anticipa(ee):
            return [(e, src.load(TensorKey(L, e, f)), hs.grezza(e)) for e in ee]

        with G.contesto(strato=L, famiglia=f), ThreadPoolExecutor(max_workers=1) as pool:
            for K, lista in per_k.items():
                lotti = [lista[i:i + config["lotto"]] for i in range(0, len(lista), config["lotto"])]
                prossimo = pool.submit(anticipa, lotti[0])
                for i, ee in enumerate(lotti):
                    materiali = prossimo.result()
                    prossimo = pool.submit(anticipa, lotti[i + 1]) if i + 1 < len(lotti) else None
                    Ts, Ss, Hs, Ws = [], [], [], []
                    with G.contesto(esperti=ee):
                        for e, tt, grezza in materiali:
                            with G.contesto(esperto=e):
                                H, fonte = hs.di_gpu(e, dev, grezza)
                                Hr = CP._ruota_H_gpu(CM, H)
                                G.verifica_finito(Hr, "H_ruotata")
                            tot["fonti"][fonte] = tot["fonti"].get(fonte, 0) + 1
                            Ts.append(tt.planes)
                            Ss.append(tt.scales.double())
                            Hs.append(Hr)
                            Ws.append(C[e].to(dev) @ Dg.T if gpu_correzione else C[e] @ D.T)
                        T, S = torch.stack(Ts).to(dev), torch.stack(Ss).to(dev, dt)
                        H = torch.stack(Hs).to(dev, dt)
                        Q0 = CP.ricostruisci_lotto(T, S)
                        W = Q0 + torch.stack(Ws).to(dev, dt)
                        G.verifica_finito(W, "W_bersaglio")
                        Ha = H + 1e-4 * torch.eye(H.shape[-1], dtype=dt, device=dev)
                        e_dw = CP.obiettivo_lotto(W, Q0, Ha).sum(-1)
                        Ssc = CP._fp16(CP.scale_lotto(W, Ha, T, S.shape[-1]))
                        e_sc = CP.obiettivo_lotto(W, CP.ricostruisci_lotto(T, Ssc), Ha).sum(-1)
                        T2, S2, _ = CP.jarq_lotto(W, H, T, S, giri=3, nu2=1e-4)
                        G.verifica_finito(S2, "scale_dopo_JARQ")
                        e_f = CP.obiettivo_lotto(W, CP.ricostruisci_lotto(T2, S2), Ha).sum(-1)
                        G.verifica_finito(e_f, "residuo_dopo_JARQ")
                        tot["dW"] += float(e_dw.sum())
                        tot["fold"] += float(e_f.sum())
                        tot["scale_sole"] += float(e_sc.sum())
                        tot["esperti"] += len(ee)
                        tc, sc = T2.cpu(), S2.cpu().half()
                        for j, e in enumerate(ee):
                            for k in range(K):
                                bb = CM.gp.encode_rows(tc[j, k].numpy().astype(np.int8),
                                                       sc[j, k].numpy(), entry.fmt.type_id)
                                tp, sp = CM.gp.decode_rows(bb, entry.fmt.type_id, entry.in_)
                                if not np.array_equal(tp, tc[j, k].numpy()) or not np.array_equal(
                                        sp.view(np.uint16), sc[j, k].numpy().view(np.uint16)):
                                    raise RuntimeError(f"round trip diverso: L{L}/{f}/e{e}/k{k}")
                                raws[k][e if k == 0 else entry.slot(k, e)] = bb
        td = out / "tensori"
        td.mkdir()
        for t, raw in zip(entry.tensors, raws):
            np.save(td / f"{t.name}.npy", raw, allow_pickle=False)
        if {p.name for p in td.iterdir()} != set(dati["file_attesi"]):
            raise RuntimeError("insieme piani esperti diverso dal manifesto")
        rec = {"strato": L, "famiglia": f, "esperti_piegati": tot["esperti"],
               "energia_correzione_H": tot["dW"], "residuo_dopo_piega_H": tot["fold"],
               "catturata": 1 - tot["fold"] / max(tot["dW"], 1e-300),
               "catturata_sole_scale": 1 - tot["scale_sole"] / max(tot["dW"], 1e-300),
               "hessiane": tot["fonti"], "secondi": time.monotonic() - t0,
               "correzione_gpu": gpu_correzione}
        (out / "registro.jsonl").write_text(json.dumps(rec, allow_nan=False) + "\n")
        torch.cuda.synchronize(dev)
        from pianificatore_gpu import tipo_device
        return {"picco_gpu_byte": torch.cuda.max_memory_reserved(dev),
                "device_tipo": tipo_device(torch, dev)}
    return lavora

#!/usr/bin/env python3
"""Adattatori persistenti: algebra collaudata, input mmap, output privati.

FORGIA_PREPARA_GPU=1 abilita anche la preparazione dei densi su GPU.
Questa opzione cambia l'ordine delle riduzioni: richiede un nuovo collaudo.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

QUI = Path(__file__).resolve().parent


def crea(config, device):
    import torch
    torch.cuda.set_device(device)
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    os.environ["PIEGA_ALGEBRA"] = config["algebra"]
    if config["algebra"] not in ("gpu_sicura", "cpu"):
        raise ValueError("rocSOLVER GPU vietato")
    os.environ["LEVE_OSPITE"] = config["gguf"]
    sys.path.insert(0, "@@ROOT:models@@/build/leve/raffina")
    import comune as CM
    import cardan_piega_gpu as CP
    from cardan_algebra_guardia_gpu import installa
    from cardan_guardia_gpu import contatori
    diagnostica = installa(CP) if config["algebra"] == "gpu_sicura" else contatori
    dev = torch.device(f"cuda:{device}")
    if config["fase"] == "a1":
        return BaseA1(config, dev, CM)
    import cardan_piega_densi_gpu as CD
    checkpoint = torch.load(config["checkpoint"], mmap=True, map_location="cpu", weights_only=False)
    if config["fase"] == "densi":
        delta = {CD.nome_gguf(k): v for k, v in checkpoint["delta_file"].items()}
        manif = json.loads(Path(CD.MANIF_R2B3).read_text())
    else:
        delta, manif = None, None
    input_densi = None
    if config["fase"] == "densi" and config.get("riuso_input"):
        from cardan_input_densi_gpu import InputDensi
        input_densi = InputDensi(config, delta, manif, CM, CD)
    dt = getattr(torch, config["dtype"])

    def lavora(dati, out):
        from pianificatore_gpu import identita_file
        for meta in config["ingressi"]:
            if identita_file(meta["path"]) != meta:
                raise RuntimeError(f"input cambiato: {meta['path']}")
        torch.cuda.reset_peak_memory_stats(dev)
        ripieghi_prima = diagnostica["matrici_cpu"]
        td = out / "tensori"
        td.mkdir()
        t0 = time.monotonic()
        if config["fase"] == "esperti":
            L = dati["strato"]
            stato = checkpoint["stato"].get(L, checkpoint["stato"].get(str(L)))
            if stato is None:
                raise ValueError(f"checkpoint senza strato {L}")
            with open(out / "registro.jsonl", "x") as reg:
                CP.piega_strato(L, stato, config["gguf"], dev, dt, config["lotto"], td, reg)
            nomi = {p.name for p in td.glob("*.npy")}
            if nomi != set(dati["file_attesi"]):
                raise RuntimeError(f"insieme piani esperti diverso dal piano: {nomi}")
        else:
            ds, dws, hs = [], [], []
            # La sorgente usa GGUFReader/mmap; qui si decodifica SOLO questo lotto.
            src = (None if input_densi is not None else
                   CM.gp.GGUFPlaneSource(config["gguf"], layers=dati["strati"]))
            for nome in dati["nomi"]:
                ds.append(input_densi.denso(nome) if input_densi is not None else CM.Denso(src, nome))
                dws.append(delta[nome].double())
                hs.append(input_densi.hessiana(nome) if input_densi is not None else CD.hessiana(nome, manif))
            if config["prepara_gpu"]:
                risultati = piega_gruppo_preparato_gpu(ds, dws, hs, CM, CP, dev, dt)
            else:
                risultati = CD.piega_gruppo(ds, dws, hs, CM, CP, dev, dt)
            sec = (time.monotonic() - t0) / len(ds)
            with open(out / "registro.jsonl", "x") as reg:
                for d, (T, S, rec) in zip(ds, risultati):
                    for k, (nome, bb) in enumerate(zip(d.nomi_piani, d.byte_piani(T, S))):
                        tp, sp = CM.gp.decode_rows(bb, d.tid, d.in_)
                        import numpy as np
                        if not np.array_equal(tp, T[k].numpy()) or not np.array_equal(
                                sp.view(np.uint16), S[k].to(torch.float16).numpy().view(np.uint16)):
                            raise RuntimeError(f"round trip diverso: {nome}")
                        np.save(td / f"{nome}.npy", bb, allow_pickle=False)
                    rec["secondi"] = sec
                    reg.write(json.dumps(rec, allow_nan=False) + "\n")
        torch.cuda.synchronize(dev)
        from pianificatore_gpu import tipo_device
        return {"picco_gpu_byte": torch.cuda.max_memory_reserved(dev), "device_tipo": tipo_device(torch, dev),
                "matrici_rifatte_cpu": diagnostica["matrici_cpu"] - ripieghi_prima,
                "input_densi_cumulativi": input_densi.statistiche() if input_densi is not None else None}
    return lavora


def piega_gruppo_preparato_gpu(ds, dws, hs, CM, CP, dev, dt):
    """Stesso JARQ, Q0 accumulata piano per piano fp64 prima del cast di W."""
    import torch
    Ws, Hs, Ts, Ss, norme = [], [], [], [], []
    segni = torch.from_numpy(CM.gp.sign1024().astype("float32")).to(dev, torch.float64)

    def ruota(x):
        C = x.shape[-1]
        return CM.fwht((x.reshape(*x.shape[:-1], C // 1024, 1024) * segni).reshape(x.shape), 1024)

    for d, dw, h in zip(ds, dws, hs):
        if tuple(dw.shape) != (d.out, d.in_):
            raise ValueError(f"forma delta diversa: {d.nome}")
        tg, sg = d.T.to(dev), d.S.to(dev, torch.float64)
        q0 = torch.zeros(d.out, d.in_, device=dev, dtype=torch.float64)
        ng = sg.shape[-1]
        for k in range(d.K):
            q0 += (tg[k].double().reshape(d.out, ng, d.in_ // ng) *
                   sg[k].unsqueeze(-1)).reshape(d.out, d.in_)
        dg, hg = dw.to(dev), h.to(dev)
        norme.append(float(dg.norm() / q0.norm().clamp_min(1e-30)))
        if d.rotated:
            dg = ruota(dg)
            hg = ruota(ruota(hg).T.contiguous()).T.contiguous()
        Ws.append(q0 + dg)
        Hs.append(hg)
        Ts.append(tg)
        Ss.append(sg)
    W, H = torch.stack(Ws).to(dt), torch.stack(Hs).to(dt)
    T, S = torch.stack(Ts), torch.stack(Ss).to(dt)
    Ha = H + 1e-4 * torch.eye(H.shape[-1], dtype=dt, device=dev)
    e0 = CP.obiettivo_lotto(W, CP.ricostruisci_lotto(T, S), Ha).sum(-1)
    T2, S2, cat = CP.jarq_lotto(W, H, T, S, giri=3, nu2=1e-4)
    e1 = CP.obiettivo_lotto(W, CP.ricostruisci_lotto(T2, S2), Ha).sum(-1)
    out = []
    for b, d in enumerate(ds):
        t2, s2 = T2[b].cpu(), S2[b].cpu().float()
        if not torch.equal(s2, s2.half().float()) or not bool(torch.isfinite(s2).all()):
            raise RuntimeError(f"scale non finite o non fp16: {d.nome}")
        ed, ef = float(e0[b]), float(e1[b])
        out.append((t2, s2, {"nome": d.nome, "piani": d.K, "energia_correzione_H": ed,
                            "residuo_dopo_piega_H": ef, "catturata": 1 - ef / max(ed, 1e-300),
                            "norma_rel_dW": norme[b], "lotto": len(ds),
                            "righe_tornate_ospite_lotto": cat.get("righe_tornate_ospite")}))
    return out


class BaseA1:
    """Corpo di cardan_costruisci_gpu, con sorgenti e segni persistenti."""

    def __init__(self, config, dev, CM):
        import numpy as np
        import torch
        import cardan_base as CB
        sys.path.insert(0, "@@ROOT:depot@@/tutto_ternario/codice")
        from sorgente import SorgenteHF, bf16_a_f32
        from rd_ternary.formato import TensorKey
        self.config, self.dev, self.CM, self.CB = config, dev, CM, CB
        self.np, self.torch, self.bf16, self.key = np, torch, bf16_a_f32, TensorKey
        self.src = SorgenteHF()
        self.srcp = CM.gp.GGUFPlaneSource(config["gguf"], bases=["ffn_gate_exps", "ffn_up_exps"])
        self.segni = torch.from_numpy(CM.gp.sign1024().astype(np.float32)).to(dev, torch.float64)

    def __call__(self, dati, out):
        from concurrent.futures import ThreadPoolExecutor
        from pianificatore_gpu import identita_file
        for meta in self.config["ingressi"]:
            if identita_file(meta["path"]) != meta:
                raise RuntimeError(f"input cambiato: {meta['path']}")
        np, torch, CB, dev = self.np, self.torch, self.CB, self.dev
        torch.cuda.reset_peak_memory_stats(dev)
        dt = torch.float64
        L, rk, pezzo = dati["strato"], self.config["rango"], self.config["pezzo"]
        t0 = time.monotonic()
        H = torch.from_numpy(np.load(Path(self.config["hdir"]) / f"H_ffn_{L:02d}.npy").astype(np.float64))
        H = H / torch.trace(H) * H.shape[0]
        Rl = CB.chol_sigma(H)
        Hg, Rlg = H.to(dev), Rl.to(dev)
        u16, _ = self.src.leggi_hf(f"model.language_model.layers.{L}.mlp.experts.gate_up_proj")
        arrs, misure = {}, {}
        for fam, sl in (("gate", slice(0, 512)), ("up", slice(512, 1024))):
            Rs = torch.empty(256, 512, 2048, dtype=torch.float32, device=dev)
            with ThreadPoolExecutor(self.config["fili_lettura"]) as pool:
                tqs = list(pool.map(lambda e: self.srcp.load(self.key(L, e, fam)), range(256)))
            for e, tq in enumerate(tqs):
                P, S = tq.planes.to(dev), tq.scales.to(dev, dt)
                K, R, C = P.shape
                ng = S.shape[-1]
                Wp = (P.to(dt).reshape(K, R, ng, C // ng) * S.unsqueeze(-1)).sum(0).reshape(R, C)
                if tq.rotated:
                    y = self.CM.fwht(Wp, 1024)
                    Q = (y.reshape(*Wp.shape[:-1], C // 1024, 1024) * self.segni).reshape(Wp.shape)
                else:
                    Q = Wp
                W = torch.from_numpy(self.bf16(u16[e, sl]).astype(np.float32)).to(dev)
                Rs[e] = W - Q.float()
            S = torch.zeros(2048, 2048, dtype=dt, device=dev)
            for i in range(0, 256, pezzo):
                Rb = Rs[i:i + pezzo].to(dt).reshape(-1, 2048)
                S += Rb.T @ Rb
            E0 = float((Hg * S).sum())
            G = (Rlg.T @ S @ Rlg).cpu()
            D16 = CB.fp16(CB.base_condivisa(G, Rl, rk))
            SD = H @ D16.to(dt)
            M = D16.to(dt).T @ SD
            M = M + 1e-12 * torch.trace(M) / M.shape[0] * torch.eye(M.shape[0], dtype=dt)
            # Solo CPU: stessa inv 16x16 del percorso collaudato.
            Minv = torch.linalg.inv(M)
            SDg, Minvg = SD.to(dev), Minv.to(dev)
            C16 = torch.empty(256, 512, rk, dtype=torch.float16)
            Mg = (D16.to(dt).T @ H @ D16.to(dt)).to(dev)
            x1 = x2 = 0.0
            for i in range(0, 256, pezzo):
                RSD = Rs[i:i + pezzo].to(dt) @ SDg
                Cn = CB.fp16(RSD @ Minvg)
                C16[i:i + pezzo] = Cn.half().cpu()
                x1 += float((Cn * RSD).sum())
                x2 += float(((Cn @ Mg) * Cn).sum())
            E1 = E0 - 2 * x1 + x2
            del Rs, S, tqs
            arrs[f"blk.{L}.ffn_{fam}_lrd.weight"] = D16.T.contiguous().half().numpy()
            arrs[f"blk.{L}.ffn_{fam}_lrc.weight"] = C16.numpy()
            misure[fam] = {"err_rel_prima": E0 ** .5, "riduzione_errore_pesato": 1 - E1 / E0}
        if not all(np.isfinite(v).all() for v in arrs.values()):
            raise RuntimeError(f"A.1 non finita: strato {L}")
        np.savez(out / f"strato_{L:02d}.npz", **{k: v.view(np.uint8) for k, v in arrs.items()})
        (out / "misure.json").write_text(json.dumps(
            {str(L): {"rango": rk, **misure, "secondi": time.monotonic() - t0, "versione": "gpu"}},
            allow_nan=False))
        info = {"tensori": {}}
        for fam in ("gate", "up"):
            info["tensori"][f"blk.{L}.ffn_{fam}_lrd.weight"] = {"type": 1, "ne": [2048, rk]}
            info["tensori"][f"blk.{L}.ffn_{fam}_lrc.weight"] = {"type": 1, "ne": [rk, 512, 256]}
        (out / "info.json").write_text(json.dumps(info))
        torch.cuda.synchronize(dev)
        from pianificatore_gpu import tipo_device
        return {"picco_gpu_byte": torch.cuda.max_memory_reserved(dev), "device_tipo": tipo_device(torch, dev)}

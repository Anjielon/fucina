#!/usr/bin/env python3
"""Audit indipendente delle pieghe: decodifica i byte pubblicati, NON JARQ.

Ricalcola tr(E H E^T) fp64 sulla stessa metrica (ridge 1e-4 incluso).
Nessuna fattorizzazione. CPU per una prova piccola, GPU solo a corse libere.
Il rapporto e' legato agli SHA di TUTTI i piani e dei due registri.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import pianificatore_gpu as PG


def impronta(root):
    root = Path(root).resolve(strict=True)
    files = sorted((root / "tensori").glob("*.npy")) + [root / "registro.jsonl"]
    if len(files) <= 1:
        raise ValueError(f"nessun tensore: {root}")
    return {"root": str(root), "file": {str(p.relative_to(root)): PG.sha_file(p) for p in files}}


def verifica_rapporto(path, rif, cand, tolleranza):
    r = json.loads(Path(path).read_text())
    if r.get("schema") != 1 or not math.isfinite(tolleranza) or tolleranza < 0:
        raise ValueError("schema/tolleranza audit non validi")
    for nome, root in (("riferimento", rif), ("candidato", cand)):
        if r["output"][nome] != impronta(root):
            raise ValueError(f"tensori/registro cambiati dopo l'audit indipendente: {root}")
    if not r.get("metriche"):
        raise ValueError("audit vuoto")
    from collauda_fucina_gpu import chiavi_metriche
    attese, _ = chiavi_metriche(Path(rif))
    chiavi = [tuple(m["chiave"]) if isinstance(m["chiave"], list) else m["chiave"]
              for m in r["metriche"]]
    if len(set(chiavi)) != len(chiavi) or set(chiavi) != set(attese):
        raise ValueError("audit con metriche duplicate/mancanti/estranee")
    tr = r["toll_report"]
    if not math.isfinite(tr) or tr < 0:
        raise ValueError("tolleranza report non valida")
    massimo = 0.0
    for rec in r["metriche"]:
        for nome in ("riferimento", "candidato"):
            m = rec[nome]
            if not all(math.isfinite(m[k]) for k in ("e0", "e1", "catturata", "scarto_report")):
                raise ValueError("audit non finito")
            if m["e0"] < 0 or m["e1"] < 0 or not math.isclose(
                    m["catturata"], 1 - m["e1"] / max(m["e0"], 1e-300),
                    rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("audit energie incoerenti")
            if m["scarto_report"] > tr:
                raise ValueError("metriche del registro non riprodotte dai tensori")
        massimo = max(massimo, abs(rec["riferimento"]["catturata"] - rec["candidato"]["catturata"]))
    return {"ok": massimo <= tolleranza, "scarto_catturata_massimo": massimo,
            "rapporto": str(Path(path).resolve()), "sha256": PG.sha_file(path),
            "toll_report": tr}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--fase", choices=("esperti", "densi"), required=True)
    ap.add_argument("--riferimento", required=True)
    ap.add_argument("--candidato", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--righe", type=int, default=64, help="righe per GEMM, non sottocampionamento")
    ap.add_argument("--max-unita", type=int, default=2)
    ap.add_argument("--toll-report", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if (a.righe < 1 or a.max_unita < 1 or
            not math.isfinite(a.toll_report) or a.toll_report < 0):
        ap.error("righe/max-unita positivi, toll-report finita e non negativa")
    if Path(a.out).exists():
        raise FileExistsError(a.out)
    os.nice(max(0, 19 - os.getpriority(os.PRIO_PROCESS, 0)))
    for _ in range(3):
        PG.guardia_host(*PG.misura_host())
    if a.device != "cpu":
        PG.guardia_corse()
    import numpy as np
    import torch
    torch.set_num_threads(1)
    os.environ["LEVE_OSPITE"] = a.gguf
    sys.path.insert(0, "@@ROOT:models@@/build/leve/raffina")
    import comune as CM
    import cardan_piega_gpu as CP
    import cardan_piega_densi as CD
    from rd_ternary.formato import TensorKey
    from collauda_fucina_gpu import chiavi_metriche
    dev = torch.device(a.device)
    rif, cand = Path(a.riferimento), Path(a.candidato)
    rr, _ = chiavi_metriche(rif)
    cc, _ = chiavi_metriche(cand)
    if rr.keys() != cc.keys() or len(rr) > a.max_unita:
        raise ValueError("metriche diverse o prova oltre --max-unita")
    prima = {n: impronta(p) for n, p in (("riferimento", rif), ("candidato", cand))}
    checkpoint_id = PG.identita_file(a.checkpoint)
    gguf_id = PG.identita_file(a.gguf)
    dd = torch.load(a.checkpoint, mmap=True, map_location="cpu", weights_only=False)
    records, h_input = [], {}
    manif = json.loads(Path(CD.MANIF_R2B3).read_text()) if a.fase == "densi" else None
    delta = {CD.nome_gguf(k): v for k, v in dd["delta_file"].items()} if manif else None
    t0 = time.monotonic()

    def q_da_piani(piani, scale):
        R, C = piani.shape[-2:]
        q = torch.zeros(R, C, dtype=torch.float64, device=dev)
        for t, s in zip(piani, scale):
            tg, sg = t.to(dev, torch.float64), s.to(dev, torch.float64)
            ng = sg.shape[-1]
            q += (tg.reshape(R, ng, C // ng) * sg.unsqueeze(-1)).reshape(R, C)
        return q

    def q_pubblicato(root, entry, esperto=None):
        ts, ss = [], []
        K = len(entry.tensors) if esperto is None else entry.n_planes_of(esperto)
        for k in range(K):
            raw = np.load(root / "tensori" / f"{entry.tensors[k].name}.npy",
                          mmap_mode="r", allow_pickle=False)
            if esperto is not None:
                raw = raw.reshape(int(entry.tensors[k].shape[2]), entry.out, -1)
                raw = raw[esperto if k == 0 else entry.slot(k, esperto)]
            raw = raw.reshape(entry.out, -1)
            t, s = CM.gp.decode_rows(raw, entry.fmt.type_id, entry.in_)
            ts.append(torch.from_numpy(t.copy()))
            ss.append(torch.from_numpy(s.copy()))
        return q_da_piani(torch.stack(ts), torch.stack(ss))

    def energia(E, H):
        totale = 0.0
        for i in range(0, E.shape[0], a.righe):
            PG.guardia_host(*PG.misura_host(.01))
            x = E[i:i + a.righe]
            totale += float(((x @ H) * x).sum())
        if not math.isfinite(totale) or totale < 0:
            raise ValueError("energia indipendente negativa/non finita")
        return totale

    for key in sorted(rr):
        PG.guardia_host(*PG.misura_host())
        if dev.type != "cpu":
            PG.guardia_corse()
        e0, er, ec = 0.0, 0.0, 0.0
        if a.fase == "densi":
            if not isinstance(key, str):
                raise ValueError("registro non denso")
            L = int(key.split(".")[1])
            src = CM.gp.GGUFPlaneSource(a.gguf, layers=[L])
            d = CM.Denso(src, key)
            entry = src.entry(d.key)
            H = CD.hessiana(key, manif)
            hp = manif[key.replace(".weight", "3.weight")]["hessiana"]["file"]
            h_input[hp] = PG.identita_file(hp)
            dw = delta[key].double()
            if d.rotated:
                dw, H = CM.ruota(dw), CM.ruota_H(H)
            H = H.to(dev) + 1e-4 * torch.eye(H.shape[0], dtype=torch.float64, device=dev)
            q0 = q_da_piani(d.T, d.S)
            W = q0 + dw.to(dev)
            e0 = energia(W - q0, H)
            er, ec = [energia(W - q_pubblicato(root, entry), H) for root in (rif, cand)]
        else:
            if not isinstance(key, tuple):
                raise ValueError("registro non esperti")
            L, fam = key
            src = CM.gp.GGUFPlaneSource(a.gguf, layers=[L], bases=[f"ffn_{fam}_exps"])
            entry = src.entry(TensorKey(L, 0, fam))
            hs = CP.Hessiane(L)
            for hp in (Path(CP.GQ_DIR) / f"Gq_{L:02d}.npy", Path(CP.HFFN_DIR) / f"H_ffn_{L:02d}.npy"):
                if hp.exists():
                    h_input[str(hp)] = PG.identita_file(hp)
            stato = dd["stato"].get(L, dd["stato"].get(str(L)))
            if stato is None:
                raise ValueError(f"checkpoint senza strato {L}")
            D, C = stato[f"D_{fam}"].double(), stato[f"C_{fam}"].double()
            for e in range(entry.n_experts):
                PG.guardia_host(*PG.misura_host(.01))
                tt = src.load(TensorKey(L, e, fam))
                H, _ = hs.di(e)
                H = CM.ruota_H(H).to(dev)
                H = H + 1e-4 * torch.eye(H.shape[0], dtype=torch.float64, device=dev)
                q0 = q_da_piani(tt.planes, tt.scales)
                # Anche gli esperti a correzione zero: un eventuale cambiamento
                # illecito dei loro byte deve comparire nell'energia finale.
                W = q0 + (C[e] @ D.T).to(dev)
                e0 += energia(W - q0, H)
                er += energia(W - q_pubblicato(rif, entry, e), H)
                ec += energia(W - q_pubblicato(cand, entry, e), H)
        rec = {"chiave": key}
        for nome, e1, log in (("riferimento", er, rr[key]), ("candidato", ec, cc[key])):
            catturata = 1 - e1 / max(e0, 1e-300)
            scarti = [abs(log["energia_correzione_H"] - e0) / max(e0, 1e-300),
                      abs(log["residuo_dopo_piega_H"] - e1) / max(e0, 1e-300),
                      abs(log["catturata"] - catturata)]
            rec[nome] = {"e0": e0, "e1": e1, "catturata": catturata, "scarto_report": max(scarti)}
        records.append(rec)
    dopo = {n: impronta(p) for n, p in (("riferimento", rif), ("candidato", cand))}
    if prima != dopo or checkpoint_id != PG.identita_file(a.checkpoint) or gguf_id != PG.identita_file(a.gguf):
        raise RuntimeError("input/output cambiati durante la verifica indipendente")
    for p, identita in h_input.items():
        if PG.identita_file(p) != identita:
            raise RuntimeError(f"H cambiata: {p}")
    report = {"schema": 1, "output": prima, "metriche": records, "toll_report": a.toll_report,
              "checkpoint": checkpoint_id, "gguf": gguf_id, "hessiane": h_input,
              "secondi": time.monotonic() - t0, "device": str(dev), "dtype": "float64",
              "codice_sha256": PG.sha_file(__file__)}
    PG.scrivi_json(Path(a.out), report)
    ok = all(m[n]["scarto_report"] <= a.toll_report for m in records for n in ("riferimento", "candidato"))
    print(json.dumps({"rapporto": a.out, "metriche_riprodotte": ok, "secondi": report["secondi"]}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

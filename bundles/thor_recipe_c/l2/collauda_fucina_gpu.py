#!/usr/bin/env python3
"""Confronto completo di due uscite, byte e metriche, senza calcoli GPU.

Accetta cartelle di piega (tensori/ e registro.jsonl) oppure A.1.
Le soglie sono esplicite e il numero dei coefficienti fp16 non e' confuso
con quello dei byte. I tempi wall vanno forniti da misure esterne, non da somme.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pianificatore_gpu as PG


def chiavi_metriche(root):
    records = [json.loads(r) for r in (root / "registro.jsonl").read_text().splitlines() if r]
    chiavi = [(r["strato"], r["famiglia"]) if "strato" in r else r["nome"] for r in records]
    if len(set(chiavi)) != len(records):
        raise ValueError(f"registro duplicato: {root}")
    for r in records:
        if not all(math.isfinite(r[k]) for k in ("energia_correzione_H", "residuo_dopo_piega_H", "catturata")):
            raise ValueError(f"metrica non finita: {root}")
        e0, e1 = r["energia_correzione_H"], r["residuo_dopo_piega_H"]
        if e0 < 0 or e1 < 0 or not math.isclose(
                r["catturata"], 1 - e1 / max(e0, 1e-300), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"energia/catturata incoerente: {root}")
    e = sum(r["energia_correzione_H"] for r in records)
    f = sum(r["residuo_dopo_piega_H"] for r in records)
    if not records:
        raise ValueError(f"registro vuoto: {root}")
    return dict(zip(chiavi, records)), {
        "catturata_media": sum(r["catturata"] for r in records) / len(records),
        "catturata_pesata": 1 - f / e if e > 0 else None}


def confronta_metriche(rif, cand):
    kr, mr = chiavi_metriche(rif)
    kc, mc = chiavi_metriche(cand)
    if kr.keys() != kc.keys():
        raise ValueError("insieme esatto delle metriche diverso")
    appaiate = [{"chiave": k, "riferimento": kr[k], "candidato": kc[k],
                 "scarto_catturata": abs(kr[k]["catturata"] - kc[k]["catturata"])}
                for k in sorted(kr)]
    return {"riferimento": mr, "candidato": mc, "appaiate": appaiate,
            "scarto_massimo": max(r["scarto_catturata"] for r in appaiate)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--riferimento", required=True)
    ap.add_argument("--candidato", required=True)
    ap.add_argument("--fase", choices=("piega", "a1"), required=True)
    ap.add_argument("--toll-byte", type=float, default=0)
    ap.add_argument("--toll-metrica", type=float, default=0)
    ap.add_argument("--secondi-riferimento", type=float)
    ap.add_argument("--secondi-candidato", type=float)
    ap.add_argument("--verifica-indipendente", help="rapporto energie ricostruite da cardan_verifica_energie_gpu.py")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.fase != "piega" and a.verifica_indipendente:
        ap.error("--verifica-indipendente ricostruisce le energie delle sole pieghe")
    if (not math.isfinite(a.toll_byte) or not 0 <= a.toll_byte <= 1 or
            not math.isfinite(a.toll_metrica) or a.toll_metrica < 0):
        ap.error("tolleranze finite: byte in [0,1], metrica >=0")
    for t in (a.secondi_riferimento, a.secondi_candidato):
        if t is not None and (not math.isfinite(t) or t <= 0):
            ap.error("tempi wall positivi e finiti")
    PG.guardia_corse()
    PG.guardia_host(*PG.misura_host())
    import numpy as np
    rif, cand = Path(a.riferimento).resolve(strict=True), Path(a.candidato).resolve(strict=True)
    if Path(a.out).exists():
        raise FileExistsError(f"non sovrascrivo {a.out}")
    report = {"schema": 2, "fase": a.fase, "riferimento": str(rif), "candidato": str(cand), "tensori": [],
              "toll_byte": a.toll_byte, "toll_metrica": a.toll_metrica,
              "wall_s": {"riferimento": a.secondi_riferimento, "candidato": a.secondi_candidato}}
    diversi = totale = 0

    def confronta(nome, x, y, fp16=False):
        nonlocal diversi, totale
        if x.shape != y.shape or x.dtype != y.dtype:
            raise ValueError(f"forma/dtype diversi: {nome}")
        bx, by = x.reshape(-1).view(np.uint8), y.reshape(-1).view(np.uint8)
        nd = 0
        for i in range(0, bx.size, 1 << 20):
            nd += int(np.count_nonzero(bx[i:i + (1 << 20)] != by[i:i + (1 << 20)]))
        rec = {"nome": nome, "byte_diversi": nd, "byte_totali": bx.size,
               "frazione_byte": nd / bx.size if bx.size else 0}
        if fp16:
            xx, yy = bx.view(np.float16), by.view(np.float16)
            if not np.isfinite(xx).all() or not np.isfinite(yy).all():
                raise ValueError(f"coefficienti non finiti: {nome}")
            ne, num, den = 0, 0.0, 0.0
            for i in range(0, xx.size, 1 << 18):
                px, py = xx[i:i + (1 << 18)].astype(np.float64), yy[i:i + (1 << 18)].astype(np.float64)
                ne += int(np.count_nonzero(px != py))
                num += float(np.sum((px - py) ** 2))
                den += float(np.sum(px ** 2))
            rec.update(coefficienti_fp16_diversi=ne, coefficienti_fp16_totali=xx.size,
                       errore_rel_l2=math.sqrt(num / den) if den else (0 if not num else None))
        diversi += nd
        totale += bx.size
        report["tensori"].append(rec)

    if a.fase == "piega":
        ff = {p.name: p for p in (rif / "tensori").glob("*.npy")}
        gg = {p.name: p for p in (cand / "tensori").glob("*.npy")}
        if not ff or ff.keys() != gg.keys():
            raise ValueError("insieme esatto dei piani diverso/vuoto")
        for nome in sorted(ff):
            confronta(nome, np.load(ff[nome], mmap_mode="r", allow_pickle=False),
                      np.load(gg[nome], mmap_mode="r", allow_pickle=False))
        report["metriche"] = confronta_metriche(rif, cand)
        scarto = report["metriche"]["scarto_massimo"]
        report["verifica_indipendente"] = {"ok": False, "motivo": "non eseguita"}
        if a.verifica_indipendente:
            from cardan_verifica_energie_gpu import verifica_rapporto
            report["verifica_indipendente"] = verifica_rapporto(
                a.verifica_indipendente, rif, cand, a.toll_metrica)
    else:
        ff, gg = {p.name: p for p in rif.glob("strato_*.npz")}, {p.name: p for p in cand.glob("strato_*.npz")}
        if not ff or ff.keys() != gg.keys():
            raise ValueError("insieme esatto degli strati A.1 diverso/vuoto")
        for nome in sorted(ff):
            with np.load(ff[nome], allow_pickle=False) as x, np.load(gg[nome], allow_pickle=False) as y:
                if set(x.files) != set(y.files):
                    raise ValueError(f"chiavi NPZ diverse: {nome}")
                for k in sorted(x.files):
                    confronta(k, x[k], y[k], fp16=True)
        mr, mc = json.loads((rif / "misure.json").read_text()), json.loads((cand / "misure.json").read_text())
        if mr.keys() != mc.keys():
            raise ValueError("strati misure A.1 diversi")
        scarti = [abs(mr[L][f]["riduzione_errore_pesato"] - mc[L][f]["riduzione_errore_pesato"])
                  for L in mr for f in ("gate", "up")]
        if not scarti or not all(math.isfinite(x) for x in scarti):
            raise ValueError("metriche A.1 vuote/non finite")
        scarto = max(scarti)
        report["metriche"] = {"scarto_massimo_riduzione_errore": scarto}
    report.update(byte_diversi=diversi, byte_totali=totale, frazione_byte=diversi / totale,
                  ok=all(r["frazione_byte"] <= a.toll_byte for r in report["tensori"])
                  and scarto <= a.toll_metrica)
    if a.verifica_indipendente:
        report["ok"] = report["ok"] and report["verifica_indipendente"]["ok"]
    report["certificabile"] = report["ok"] and (
        a.fase == "a1" or report["verifica_indipendente"]["ok"])
    piano = cand.parent / "lancio.json"
    if piano.is_file() and cand.name == "pubblicato":
        p = json.loads(piano.read_text())
        report["ricetta_sha256"] = p["ricetta_sha256"]
        report["dispositivi"] = sorted({
            json.loads((cand.parent / "unita" / u["chiave"] / "completa.json").read_text())
            ["misure"]["device_tipo"] for u in p["unita"]})
    PG.scrivi_json(Path(a.out), report)
    print(json.dumps(report, indent=1))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

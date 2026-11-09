#!/usr/bin/env python3
"""cardan_piega_densi_gpu_veloce.py (21/10 giro 7, D12 «preparazione densi 213 s»): stessa piega di
cardan_piega_densi_gpu.py --lotto B (stessi lotti, stesso ordine, stessa piega_gruppo, stessi byte), senza i tempi morti.

Misurato sulla tappa 150 (piega_d/piega.out): 213 s fra l'avvio e il primo lotto, quasi tutti in 37 aperture
GGUFPlaneSource(layers=[L]) (6-9 s l'una: GGUFReader rilegge l'indice di tutti i tensori) + torch.load del delta da 5 GB
convertito TUTTO in fp64 (RSS 9,4 GB: alla tappa 150 MemAvailable e' scesa a 8,2 GiB con swap-out 180 MB/s).
Qui:
  (1) UNA sola apertura del GGUF ospite (layers = tutti gli strati del delta): le voci di uno strato sono le stesse;
  (2) delta letto con torch.load(mmap=True) e convertito in fp64 tensore per tensore solo quando serve (stessi valori);
  (3) Hessiane del lotto successivo lette in un filo mentre la GPU piega il lotto corrente;
  (4) codifica dei byte + verifica di andata e ritorno + scrittura atomica + registro in un filo (in ordine), sovrapposti
      al calcolo GPU del lotto successivo.
La matematica (piega_gruppo -> cardan_piega_gpu.jarq_lotto) e l'ordine dei lotti sono quelli dell'originale.
Uso: identico a cardan_piega_densi_gpu.py (richiede --lotto > 0). Orchestratori: PIEGA_DENSI_VELOCE=1.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
import numpy as np  # noqa: E402
import torch  # noqa: E402

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import cardan_piega_densi_gpu as ORIG  # noqa: E402
import cardan_ripresa_densi_gpu as RD

log = ORIG.log


class _DeltaPigro:
    """delta_file del checkpoint, letto via mmap; .double() solo all'accesso (come {k: v.double()} dell'originale)."""

    def __init__(self, dd_file: dict):
        self._v = {ORIG.nome_gguf(k): v for k, v in dd_file.items()}

    def __iter__(self):
        return iter(self._v)

    def __len__(self):
        return len(self._v)

    def __contains__(self, k):
        return k in self._v

    def __getitem__(self, k):
        return self._v[k].double()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    ap.add_argument("--lotto", type=int, default=8)
    ap.add_argument("--parte", default="0/1")
    ap.add_argument("--fili-lettura", type=int, default=4)
    a = ap.parse_args(argv)
    # D15 (22/10): FORGIA_DENSI_DTYPE overrides --dtype for a whole chain without editing the orchestrator scripts.
    # Unset = previous behaviour. float64 on the GPU stays within CK_TOLL_BYTE=1e-5 of the CPU fp64 fold (measured).
    if os.environ.get("FORGIA_DENSI_DTYPE"):
        if os.environ["FORGIA_DENSI_DTYPE"] not in ("float64", "float32"):
            raise SystemExit("FORGIA_DENSI_DTYPE must be float64 or float32")
        a.dtype = os.environ["FORGIA_DENSI_DTYPE"]
    if a.lotto <= 0:
        raise SystemExit("--lotto deve essere > 0 (il percorso tensore per tensore e' quello dell'originale)")
    pi, pn = RD.parte(a.parte)
    with RD.Partizione(a.out, pi, pn):
        return esegui(a, pi, pn)


def esegui(a, pi, pn):
    os.environ["LEVE_OSPITE"] = a.gguf
    sys.path.insert(0, ORIG.RAFF)
    import comune as CM  # noqa: E402
    import cardan_piega_gpu as CP  # noqa: E402
    import cardan_babai_triton_gpu as BT  # noqa: E402
    BT.installa(CP)                                                       # FORGIA_BABAI_TRITON=1
    dev = torch.device(a.device)
    dt = torch.float64 if (dev.type == "cpu" or a.dtype == "float64") else torch.float32
    t_tot = time.time()
    try:
        dd = torch.load(a.delta, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError:                                                      # vecchio formato non zip: lettura intera
        dd = torch.load(a.delta, map_location="cpu", weights_only=False)
    delta = _DeltaPigro(dd["delta_file"])
    manif = json.load(open(ORIG.MANIF_R2B3))
    out = Path(a.out)
    td = out / "tensori"
    td.mkdir(parents=True, exist_ok=True)
    reg_p = out / (f"registro.{pi}.jsonl" if pn > 1 else "registro.jsonl")
    tutti = sorted(delta)
    mio = {n for j, n in enumerate(tutti) if j % pn == pi}
    strati = sorted({int(n.split(".")[1]) for n in delta})
    t_ap = time.time()
    src = CM.gp.GGUFPlaneSource(a.gguf, layers=strati)                        # (1) una sola apertura
    log(f"GGUF aperto una volta in {time.time() - t_ap:.1f} s")
    prov = RD.Provenienza(src, delta, manif, RD.ricetta(
        dev, dt, (ORIG.piega_uno, ORIG.piega_gruppo, ORIG.hessiana, ORIG.nome_gguf), (CP.__file__,)))   # D36
    fatti = RD.riprendi(out, mio, src, CM, log, prov)
    log(f"host {a.gguf}, {len(delta)} dense corrections (rank {dd.get('rango')}, step {dd.get('passo')}), "
        f"layers {strati[0]}-{strati[-1]}, already done {len(fatti)}, device {dev} [veloce]")
    ordine = []
    for L in strati:
        ordine += [n for n in sorted(delta) if int(n.split(".")[1]) == L and n not in fatti and n in mio]
    with ThreadPoolExecutor(max_workers=max(1, a.fili_lettura)) as pool:
        densi = list(pool.map(lambda n: CM.Denso(src, n), ordine))            # stesso ordine dell'originale
    gruppi = {}
    for d in densi:
        gruppi.setdefault((d.out, d.in_, d.K, d.rotated), []).append(d)
    log(f"lotti: {sum(len(v) for v in gruppi.values())} tensori in {len(gruppi)} forme, lotto {a.lotto}")
    lotti = [(forma, ds[i:i + a.lotto]) for forma, ds in gruppi.items() for i in range(0, len(ds), a.lotto)]

    def leggi_H(parte):                                                       # (3)
        return [ORIG.hessiana(d.nome, manif) for d in parte]

    def scrivi(forma, parte, ris, t0, sec, reg):                              # (4)
        for d, (T2, S2, rec) in zip(parte, ris):
            byte = d.byte_piani(T2, S2)
            artefatti = {}
            for k, (nm, bb) in enumerate(zip(d.nomi_piani, byte)):
                tp, sp = CM.gp.decode_rows(bb, d.tid, d.in_)
                assert np.array_equal(tp, T2[k].numpy().astype(np.int8)) and np.array_equal(
                    sp.view(np.uint16), S2[k].to(torch.float16).numpy().view(np.uint16)), (d.nome, k)
                nome_file, scheda = RD.salva_piano(td, nm, bb)
                artefatti[nome_file] = scheda
            rec["secondi"] = sec
            RD.commit(reg, rec, artefatti, prov.di(d.nome))
        log(f"forma {forma}: {len(parte)} tensori in {sec * len(parte):.1f} s ({sec} s a tensore), scritti")

    with RD.Registro(reg_p) as reg, ThreadPoolExecutor(max_workers=1) as fH, ThreadPoolExecutor(max_workers=1) as fW:
        prossimo = fH.submit(leggi_H, lotti[0][1]) if lotti else None
        scrittura = None
        for i, (forma, parte) in enumerate(lotti):
            Hs = prossimo.result()
            prossimo = fH.submit(leggi_H, lotti[i + 1][1]) if i + 1 < len(lotti) else None
            t0 = time.time()
            ris = ORIG.piega_gruppo(parte, [delta[d.nome] for d in parte], Hs, CM, CP, dev, dt)
            sec = round((time.time() - t0) / len(parte), 2)
            if scrittura is not None:
                scrittura.result()                                            # errori della scrittura precedente
            scrittura = fW.submit(scrivi, forma, parte, ris, t0, sec, reg)
        if scrittura is not None:
            scrittura.result()
    log(f"FINE piega densi a lotti in {time.time() - t_tot:.0f} s -> {td}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

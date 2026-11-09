#!/usr/bin/env python3
"""VELOCE2 14/10 — prova GPU: il lotto di esperti (S23 --lotto-esperti) cambia i bit? risolvi_esperti_gu e
risolvi_esperti_down dell'albero (stesse funzioni della passata) girati sugli STESSI dati con lotti diversi; si confrontano
i byte scritti (trit + scale di ogni esperto) e i numeri del registro. Dati sintetici a forma VERA (H 2048, I 512,
fp32 sul dispositivo, candidati a {0; 0,5}, k 8000, guardia simmetrica + energia relativa, H corta 3,5).
Uso: python3 s23_prova_gpu_lotti.py --dir <cartella di lavoro FUORI dalla corsa> --esperti 32 --lotti 16,32 [--device cuda]
Stampa per lotto: secondi gate/up e down, e per ogni coppia (lotto di riferimento, lotto): byte identici (si/no),
esperti diversi, frazione di trit diversi, scale diverse, differenza dei numeri del registro."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

QUI = Path(__file__).resolve().parent
for p in (str(QUI), str(QUI.parent / "lib"), str(QUI.parent / "s1b"), str(QUI.parent / "rif")):
    if p not in sys.path:
        sys.path.insert(0, p)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
import s23_algebra as AL  # noqa: E402
import s1b_comune as K  # noqa: E402
import s1_comune as SC  # noqa: E402
import cardan_piega as CP  # noqa: E402
import s23_passata as SP  # noqa: E402


def dati(E, I, H, N, dev, seme=20261014):
    """statistiche finte ma con la struttura della passata (x_q = 0,6 x_f + rumore, scale per colonna, instradamento
    top-2 sbilanciato), accumulate con le classi VERE della passata (AccDensi, AccGateUp, AccDown) sul dispositivo."""
    g = torch.Generator().manual_seed(seme)
    Wgu = (torch.randn(E, 2 * I, H, generator=g) * 0.02).float().to(dev)
    Wd = (torch.randn(E, H, I, generator=g) * 0.02).float().to(dev)
    sc = torch.linspace(0.3, 2, H)
    acc3 = SP.AccDensi(H, dev)
    adn = SP.AccDown(E, I, dev)
    gu = {}
    k = 2
    for h in "AB":
        agu = SP.AccGateUp(E, I, H, dev)
        Xf = (torch.randn(N, H, generator=g) * sc).to(dev)
        Xq = 0.6 * Xf + 0.8 * (torch.randn(N, H, generator=g) * sc).to(dev)
        for i, s in enumerate(SC.SORGENTI):
            sl = slice(i * N // 3, (i + 1) * N // 3)
            acc3.add(h, s, Xq[sl], Xf[sl])
        pr = torch.softmax(torch.randn(N, E, generator=g) * 1.5, -1).to(dev)
        idx = torch.topk(pr, k, -1).indices
        w = pr.gather(-1, idx)
        w = w / w.sum(-1, keepdim=True)
        flat = idx.reshape(-1)
        ordine = torch.argsort(flat, stable=True)
        offs = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), torch.bincount(flat, minlength=E).cumsum(0)])
        agu.add(Xq, Xf, w, k, ordine, offs, Wgu)
        gu[h] = agu
        A = torch.randn(N * k, I, generator=g).to(dev)[ordine]
        adn.add(h, A, Xf, w, k, ordine, offs, Wgu)
    return Wgu, Wd, acc3, adn, gu


def codifica(T, S, tid, C):
    Tn = T.to("cpu").numpy().astype(np.int8)
    S16 = S.to("cpu", torch.float16).numpy()
    return K.gp.encode_rows(Tn, S16, tid)


def decodifica(raw_rows, tid, C):
    t, d = K.gp.decode_rows(np.ascontiguousarray(raw_rows), tid, C)
    return torch.from_numpy(t.astype(np.int8)), torch.from_numpy(d.astype(np.float64))


def partenza(Wgu, Wd, I, H):
    """byte della partenza negli stessi tipi del GGUF: trit tutti 0 (scale = media dei |w| del gruppo), cosi' i candidati
    entrano (errore < ||W x||^2) e il confronto fra lotti riguarda byte scritti davvero."""
    def q(Wr, gr):
        R_, C_ = Wr.shape
        Wg = Wr.reshape(R_, C_ // gr, gr)
        Sg = Wg.abs().mean(-1)
        Tg = torch.zeros_like(Wg)          # partenza VUOTA (trit 0): ogni candidato con errore minore entra -> byte confrontabili
        return Tg.reshape(R_, C_).to(torch.int8), AL.fp16(Sg.double())
    raw = {}
    E = Wgu.shape[0]
    for f, qq in (("gate", 0), ("up", 1)):
        raw[f"blk.0.ffn_{f}_exps.weight"] = np.stack(
            [codifica(*q(K.ruota(Wgu[e, qq * I:(qq + 1) * I].double().cpu()), 128), 45, H) for e in range(E)])
    raw["blk.0.ffn_down_exps.weight"] = np.stack([codifica(*q(Wd[e].double().cpu(), 256), 34, I) for e in range(E)])
    return raw


def confronta(a, b):
    out = {}
    for n in sorted(set(a) | set(b)):
        if n not in a or n not in b:
            out[n] = "presente solo in uno"
            continue
        x, y = np.asarray(a[n]), np.asarray(b[n])
        if np.array_equal(x, y):
            out[n] = {"identici": True}
            continue
        tid, C = (34, x.shape[-1]) if "down" in n else (45, None)
        div_e, tr_div, tr_tot, sc_div, sc_tot = 0, 0, 0, 0, 0
        for e in range(x.shape[0]):
            if np.array_equal(x[e], y[e]):
                continue
            div_e += 1
            Cc = C if C is not None else 2048
            t1, s1 = decodifica(x[e], tid, Cc if tid == 45 else x.shape[-1])
            t2, s2 = decodifica(y[e], tid, Cc if tid == 45 else x.shape[-1])
            tr_div += int((t1 != t2).sum())
            tr_tot += t1.numel()
            sc_div += int((s1 != s2).sum())
            sc_tot += s1.numel()
        out[n] = {"identici": False, "esperti_diversi": div_e, "trit_diversi_frazione_su_esperti_diversi":
                  tr_div / max(1, tr_tot), "scale_diverse": sc_div, "scale_tot": sc_tot}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--esperti", type=int, default=32)
    ap.add_argument("--lotti", default="16,32")
    ap.add_argument("--token", type=int, default=0, help="token per meta' (0 = 300 x esperti)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device)
    dt = AL.dt_lavoro(dev)
    torch.backends.cuda.matmul.allow_tf32 = False
    E, I, H = a.esperti, 512, 2048
    N = a.token or 300 * E
    base = Path(a.dir)
    shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True)
    t = time.time()
    Wgu, Wd, acc3, adn, gu = dati(E, I, H, N, dev)
    dl = base / "strato_00"
    for h in "AB":
        gu[h].scarica(dl / f"gu_{h}")
    del gu
    raw = partenza(Wgu, Wd, I, H)
    print(f"dati: E {E}, token per meta' {N}, {time.time() - t:.1f} s", flush=True)
    JQL = AL.jarq_lotto_cache_da(CP)
    gd = {"emin": 0.9, "emax": 1.1, "peso_kw": 3.5, "relativa": True}
    ris = {}
    for lotto in [int(x) for x in a.lotti.split(",")]:
        nuovi = {}
        if dev.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        rgu = SP.risolvi_esperti_gu(0, dl, acc3, Wgu, nuovi, lambda *x: None, codifica, decodifica, lambda n: raw[n],
                                    list(range(E)), [0.0, 0.5], 8000.0, lotto, JQL, dev, dt, E, I, H, gd)
        t1 = time.time()
        rdn = SP.risolvi_esperti_down(0, adn, Wd, nuovi, lambda *x: None, codifica, decodifica, lambda n: raw[n],
                                      list(range(E)), [0.0, 0.5], 8000.0, lotto, JQL, dev, dt, E, I, H, gd)
        t2 = time.time()
        pk = torch.cuda.max_memory_reserved() / 2**30 if dev.type == "cuda" else 0.0
        ris[lotto] = (nuovi, rgu, rdn)
        print(f"lotto {lotto}: gate/up {t1 - t0:.1f} s, down {t2 - t1:.1f} s, VRAM picco {pk:.1f} GiB; accettati gate "
              f"{rgu['accettati_gate']} up {rgu['accettati_up']} down {rdn['accettati']}", flush=True)
    lotti = list(ris)
    r0 = lotti[0]
    esito = {}
    for l in lotti[1:]:
        c = confronta(ris[r0][0], ris[l][0])
        reg = {}
        for nome, i in (("gate_up", 1), ("down", 2)):
            x, y = ris[r0][i], ris[l][i]
            reg[nome] = {k: (x[k], y[k]) for k in x if k != "secondi" and x[k] != y[k]}
        esito[f"{r0}_vs_{l}"] = {"byte": c, "registro_diverso": reg}
        print(f"{r0} contro {l}: " + json.dumps(c) + " | registro diverso: " + json.dumps(reg, default=str)[:1500], flush=True)
    (base / "esito.json").write_text(json.dumps(esito, indent=1, default=str))
    tutti = all(v.get("identici") is True for e in esito.values() for v in e["byte"].values() if isinstance(v, dict))
    print(f"ESITO lotti {a.lotti}: {'BYTE IDENTICI' if tutti else 'BYTE DIVERSI'}", flush=True)
    shutil.rmtree(dl, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

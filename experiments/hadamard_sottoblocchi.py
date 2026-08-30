#!/usr/bin/env python3
"""HADAMARD SU SOTTO-BLOCCHI: quanto guadagno sopravvive a costo ridotto? (29/8)

Lo studio incrociato dice che `grid + hadamard(256)` toglie il 3.12% di errore
a parita' di bit. Ma una rotazione su 256 elementi costa 8 passate nel kernel.
Su sotto-blocchi costa log2(k): 5 passate per k=32, 6 per 64, 7 per 128.

Se il guadagno viene dalla GAUSSIANIZZAZIONE LOCALE — mescolare pochi vicini
basta a togliere le code — allora k=32 ne conserva quasi tutto a meta' prezzo,
e la leva diventa applicabile davvero. Se invece serve mescolare tutto il
blocco, k=32 non dara' niente e la leva resta materiale da articolo.

⚠️ La scala resta UNA per blocco da 256 (il formato non cambia): si ruota
dentro sotto-blocchi, ma si quantizza come sempre. Cosi' il confronto e' equo
e il file non cresce di un bit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

T = 256


def carica(bin_path: Path, n: int, seed: int = 0) -> np.ndarray:
    meta = json.loads(Path(str(bin_path).replace(".bin", ".meta")).read_text())
    ne0, ne1 = meta["dims"]
    tot = (ne0 * ne1) // T
    sel = np.sort(np.random.default_rng(seed).choice(tot, min(n, tot), replace=False))
    raw = np.memmap(bin_path, dtype=np.uint8, mode="r").reshape(-1, 34)
    idx = (sel[:, None] * 8 + np.arange(8, dtype=np.int64)[None, :]).ravel()
    sub = np.ascontiguousarray(raw[idx])
    sc = sub[:, :2].copy().view(np.float16).astype(np.float32)
    qq = sub[:, 2:].view(np.int8).astype(np.float32)
    return (qq * sc).reshape(len(sel), T)


def H(n: int) -> np.ndarray:
    h = np.ones((1, 1), dtype=np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


def ruota_sottoblocchi(w: np.ndarray, k: int) -> np.ndarray:
    """Ruota dentro sotto-blocchi da k, lasciando il blocco da 256 intatto."""
    if k <= 1:
        return w
    B = w.shape[0]
    h = H(k)
    return (w.reshape(B, T // k, k) @ h.T).reshape(B, T)


def err_grid(w: np.ndarray) -> np.ndarray:
    """La scala migliore (regola `grid`), che e' quella vincente dello studio."""
    a = np.abs(w)
    med = a.mean(1)
    best = np.full(w.shape[0], np.inf)
    for ft in np.arange(0.40, 1.05, 0.05):
        thr = ft * med
        q = np.sign(w) * (a > thr[:, None])
        nq = (q * q).sum(1)
        s0 = np.where(nq > 0, (w * q).sum(1) / np.maximum(nq, 1e-30), med)
        for fs in np.arange(0.85, 1.20, 0.05):
            s = s0 * fs
            d = w - s[:, None] * q
            best = np.minimum(best, np.sqrt((d * d).sum(1) /
                                            np.maximum((w * w).sum(1), 1e-30)))
    return best


def main() -> int:
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/models/gguf/odino-q8")
    nblk = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
    tens = sorted(d.glob("*.bin"))[: int(sys.argv[3]) if len(sys.argv) > 3 else 40]
    KS = [1, 8, 16, 32, 64, 128, 256]
    acc = {k: [] for k in KS}

    print(f"HADAMARD SU SOTTO-BLOCCHI — {len(tens)} tensori x {nblk} blocchi", flush=True)
    print("scala sempre `grid` (la vincente); cambia solo l'ampiezza della rotazione\n",
          flush=True)
    for i, f in enumerate(tens, 1):
        try:
            w0 = carica(f, nblk)
        except Exception as e:                                   # noqa: BLE001
            print(f"   salto {f.name}: {str(e)[:44]}", flush=True)
            continue
        for k in KS:
            acc[k].append(float(err_grid(ruota_sottoblocchi(w0, k)).mean()))
        if i % 10 == 0:
            print(f"   [{i}/{len(tens)}]", flush=True)

    base = float(np.mean(acc[1]))
    print(f"\nriferimento: nessuna rotazione, scala grid = {base:.6f}")
    print(f"{'k':>5s} {'passate':>8s} {'errore':>10s} {'guadagno':>10s} {'del max':>9s}")
    gmax = base - float(np.mean(acc[256]))
    for k in KS:
        m = float(np.mean(acc[k]))
        g = base - m
        passate = 0 if k == 1 else int(np.log2(k))
        quota = 100 * g / gmax if gmax > 0 else 0.0
        print(f"{k:5d} {passate:8d} {m:10.6f} {100*(m-base)/base:+9.2f}% {quota:8.0f}%")
    print("\n   'del max' = quanta parte del guadagno di hadamard(256) si conserva.")
    print("   Se k=32 ne conserva la maggior parte, la leva costa 5 passate")
    print("   invece di 8 e diventa applicabile nel kernel.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""QUANTO COSTA la rotazione di Hadamard a DECODIFICA? (29/8)

Lo studio incrociato ha trovato che `grid + hadamard` abbassa l'errore del
3.68% a parita' di bit. Ma la colonna "costo 0 GiB" del registro conta solo i
BIT: una rotazione dentro il blocco e' gratis nel file e **non** a runtime —
il kernel deve annullarla a ogni dequantizzazione, 256 elementi per blocco.

Qui si misura quel costo, perche' un guadagno di fedelta' che dimezza la
velocita' non e' un guadagno: e' uno scambio, e va deciso sapendo il prezzo.

Tre cose misurate:
  1. dequant TQ1_0 nudo                      — la linea di base
  2. dequant + Hadamard 256 (matrice densa)  — l'implementazione ingenua
  3. dequant + Hadamard veloce (butterfly)   — 8 passate log2(256), come lo
     scriverebbe un kernel serio

Il rapporto (3)/(1) e' il prezzo vero da pagare nel motore.
"""
from __future__ import annotations

import time

import numpy as np

T = 256
N = 20000          # blocchi per giro


def hadamard(n: int) -> np.ndarray:
    h = np.ones((1, 1), dtype=np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h[:n, :n] / np.sqrt(n)


def had_butterfly(x: np.ndarray) -> np.ndarray:
    """Hadamard veloce: log2(256) = 8 passate di somma/differenza.
    E' cosi' che lo farebbe un kernel — nessuna matrice, nessuna moltiplicazione."""
    y = x.copy()
    n = y.shape[1]
    passo = 1
    while passo < n:
        a = y[:, ::2 * passo]
        for k in range(passo):
            i = np.arange(k, n, 2 * passo)
            j = i + passo
            u = y[:, i].copy()
            v = y[:, j].copy()
            y[:, i] = u + v
            y[:, j] = u - v
        passo *= 2
    return y / np.sqrt(n)


def crono(eti: str, fn, giri: int = 5) -> float:
    fn()
    t = min(_una(fn) for _ in range(giri))
    print(f"   {eti:<44s} {1000*t:8.2f} ms   {N*T/t/1e6:7.1f} Mpesi/s")
    return t


def _una(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def main() -> int:
    rng = np.random.default_rng(0)
    q = rng.integers(-1, 2, size=(N, T)).astype(np.float32)   # ternario
    s = rng.random(N).astype(np.float32) + 0.1
    H = hadamard(T)

    print(f"COSTO DELLA ROTAZIONE A DECODIFICA — {N:,} blocchi x {T} pesi "
          f"= {N*T:,} pesi\n")

    t_base = crono("1. dequant nudo (scala x ternario)", lambda: s[:, None] * q)
    t_mat = crono("2. dequant + Hadamard con matrice densa",
                  lambda: (s[:, None] * q) @ H.T)
    t_fly = crono("3. dequant + Hadamard butterfly (8 passate)",
                  lambda: had_butterfly(s[:, None] * q))

    print(f"\n   matrice densa  = {t_mat/t_base:5.2f}x il dequant nudo")
    print(f"   butterfly      = {t_fly/t_base:5.2f}x il dequant nudo")

    # verifica che il butterfly sia DAVVERO la stessa trasformazione
    x = (s[:, None] * q)[:64]
    d = np.abs(x @ hadamard(T).T - had_butterfly(x)).max()
    print(f"\n   butterfly == matrice densa? scarto massimo {d:.2e} "
          f"{'✅ identiche' if d < 1e-4 else '❌ DIVERSE — il confronto non vale'}")

    print("\n   Lettura: il matvec ternario e' limitato dalla BANDA di memoria,")
    print("   non dal calcolo. Un fattore basso qui e' assorbibile; un fattore")
    print("   alto va misurato sul kernel vero prima di forgiare qualunque cosa.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

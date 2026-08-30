#!/usr/bin/env python3
"""GARA DELLE REGOLE DI SCALA per il ternario RTN (29/8, ordine di Angelo:
«ogni leva meno costosa e valida è tempo di debito scontato»).

Perche' proprio la scala: Tony V4-A e' **RTN puro** — il GPTQ gli rompe il
ragionamento (dissociazione misurata, vedi il paper), quindi tutte le leve di
compensazione sono fuori gioco per lui. Per un quantizzatore RTN resta UNA
leva: come si sceglie la scala e la soglia di ogni blocco. Nel registro
`levers.py` e' anche la piu' economica di tutte (`signed_scale_grid`, costo
0 GiB, fatica 0.1) e l'unica con evidenza SOLO degli autori — mai provata da
noi. Per confronto, `optimal_scale` da sola porto' l'errore del 397B
dall'81% al 43.5%: la scala e' la leva che paga.

Regole in gara, tutte a **costo zero di bit** (stesso formato, stessi 1.69 bpw):

  absmax      s = max|w| / 1        soglia 0.5s      — il piu' ingenuo
  twn         soglia 0.7*mean|w|, s = media di |w| sopra soglia  (TWN 1605.04711)
  ls_iter     TWN + due giri di raffinamento ai minimi quadrati
  grid        ricerca sulla griglia (fattore-soglia x fattore-scala) — BOF4-S

Si misura l'errore relativo ||w - s*q|| / ||w|| per blocco da 256 pesi, su
tensori VERI, e si guarda quante volte ogni regola vince.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

T = 256  # pesi per blocco, come TQ1_0/TCQ1_7


def carica_blocchi(bin_path: Path, n: int, seed: int = 0) -> np.ndarray:
    """Dequantizza n blocchi da 256 pesi da un dump Q8_0 (34 byte per riga)."""
    import json
    meta = json.loads(Path(str(bin_path).replace(".bin", ".meta")).read_text())
    ne0, ne1 = meta["dims"]
    tot = (ne0 * ne1) // T
    n = min(n, tot)
    sel = np.sort(np.random.default_rng(seed).choice(tot, n, replace=False))
    raw = np.memmap(bin_path, dtype=np.uint8, mode="r").reshape(-1, 34)
    idx = (sel[:, None] * 8 + np.arange(8, dtype=np.int64)[None, :]).ravel()
    sub = np.ascontiguousarray(raw[idx])
    scales = sub[:, :2].copy().view(np.float16).astype(np.float32)
    quants = sub[:, 2:].view(np.int8).astype(np.float32)
    return (quants * scales).reshape(n, T)


def _err(w: np.ndarray, s: np.ndarray, thr: np.ndarray) -> np.ndarray:
    """Errore relativo per blocco con soglia `thr` e scala `s`."""
    q = np.sign(w) * (np.abs(w) > thr[:, None])
    d = w - s[:, None] * q
    return np.sqrt((d * d).sum(1) / np.maximum((w * w).sum(1), 1e-30))


def absmax(w):
    s = np.abs(w).max(1) / 1.0
    return _err(w, s, 0.5 * s)


def twn(w):
    """Ternary Weight Networks: soglia 0.7*mean|w|, scala = media sopra soglia."""
    a = np.abs(w)
    thr = 0.7 * a.mean(1)
    m = a > thr[:, None]
    s = np.where(m.sum(1) > 0, (a * m).sum(1) / np.maximum(m.sum(1), 1), a.mean(1))
    return _err(w, s, thr)


def ls_iter(w, giri: int = 2):
    """TWN + raffinamento: fissata la maschera, la scala ottima ai minimi
    quadrati e' <w,q>/<q,q>. Si itera perche' la nuova scala cambia la maschera."""
    a = np.abs(w)
    thr = 0.7 * a.mean(1)
    for _ in range(giri):
        q = np.sign(w) * (a > thr[:, None])
        nq = (q * q).sum(1)
        s = np.where(nq > 0, (w * q).sum(1) / np.maximum(nq, 1e-30), a.mean(1))
        thr = 0.5 * s                      # soglia coerente con la nuova scala
    return _err(w, s, thr)


def grid(w, fat_thr=None, fat_s=None):
    """BOF4-S: griglia analitica su (fattore-soglia, fattore-scala).
    Costa zero bit: si sceglie solo MEGLIO dentro lo stesso formato."""
    if fat_thr is None:
        fat_thr = np.arange(0.40, 1.05, 0.05)
    if fat_s is None:
        fat_s = np.arange(0.80, 1.25, 0.05)
    a = np.abs(w)
    med = a.mean(1)
    best = np.full(w.shape[0], np.inf, dtype=np.float64)
    for ft in fat_thr:
        thr = ft * med
        q = np.sign(w) * (a > thr[:, None])
        nq = (q * q).sum(1)
        s0 = np.where(nq > 0, (w * q).sum(1) / np.maximum(nq, 1e-30), med)
        for fs in fat_s:
            e = _err(w, s0 * fs, thr)
            best = np.minimum(best, e)
    return best


REGOLE = {"absmax": absmax, "twn": twn, "ls_iter": ls_iter, "grid": grid}


def main() -> int:
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/models/gguf/odino-q8")
    per_tens = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
    tensori = sorted(d.glob("*.bin"))
    if not tensori:
        print(f"nessun tensore in {d}")
        return 1
    print(f"GARA DELLE REGOLE DI SCALA — {len(tensori)} tensori, "
          f"{per_tens} blocchi ciascuno, {per_tens*T:,} pesi per tensore\n")

    somma = {k: [] for k in REGOLE}
    vittorie = {k: 0 for k in REGOLE}
    for t in tensori:
        try:
            w = carica_blocchi(t, per_tens)
        except Exception as e:                                  # noqa: BLE001
            print(f"   salto {t.name}: {str(e)[:60]}")
            continue
        err = {k: f(w) for k, f in REGOLE.items()}
        for k, v in err.items():
            somma[k].append(float(v.mean()))
        # vittoria per blocco: quale regola sbaglia meno su quel blocco
        pila = np.stack([err[k] for k in REGOLE])
        vinc = np.argmin(pila, axis=0)
        for i, k in enumerate(REGOLE):
            vittorie[k] += int((vinc == i).sum())

    print(f"{'regola':<10s} {'errore medio':>13s} {'contro TWN':>12s} {'blocchi vinti':>14s}")
    base = float(np.mean(somma["twn"]))
    n_tot = sum(vittorie.values())
    for k in REGOLE:
        m = float(np.mean(somma[k]))
        print(f"{k:<10s} {m:13.6f} {100*(m-base)/base:+11.2f}% "
              f"{vittorie[k]:9,d} ({100*vittorie[k]/max(n_tot,1):4.1f}%)")
    mig = min(REGOLE, key=lambda k: float(np.mean(somma[k])))
    gain = 100 * (base - float(np.mean(somma[mig]))) / base
    print(f"\nMIGLIORE: {mig}  —  {gain:+.2f}% di errore rispetto a TWN "
          f"(la regola in uso), a PARITA' DI BIT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

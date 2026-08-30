#!/usr/bin/env python3
"""STUDIO INCROCIATO ESAUSTIVO delle leve a livello TENSORE (29/8).

Domanda di Angelo: «hai fatto lo studio incrociato matematicamente verificato
di tutte le combinazioni parallele e sequenziali di tutte le leve?». Risposta
onesta: no, e quello *totale* e' impossibile — 2^34 = 17.179.869.184
sottoinsiemi, e con l'ordine (molte leve sono sequenziali) si va oltre
qualunque calcolo. Anche riducendo a «al piu' una per stadio» restano 254.016
combinazioni, e ognuna vorrebbe una forgia.

Quello che INVECE si puo' chiudere per intero, ed e' quello che serve a Tony:
le leve che agiscono **sul singolo tensore**, cioe' SCALA x TRASFORMAZIONE.
Sono 4 x 5 = 20 combinazioni: si valutano tutte, su pesi veri, con l'errore
quadratico, senza forgiare nulla. Esaustivo e riproducibile.

⚠️ LIMITE DICHIARATO: allocazione, riparazione, runtime e calibrazione NON
sono giudicabili su un tensore isolato — agiscono sul modello intero. Restano
fuori da questo studio e vanno misurate con una forgia. Dirlo e' parte del
risultato: uno studio che finge di coprirle sarebbe falso.

Cosa si cerca oltre al vincitore: **l'interazione**. Se la trasformazione
migliore fosse la stessa qualunque scala si usi, le due leve sarebbero
indipendenti e si potrebbero scegliere separatamente. Se invece cambia, sono
accoppiate — e allora vanno scelte INSIEME, che e' esattamente il tipo di cosa
che un registro di leve isolate non puo' dire.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

T = 256


# ── caricamento pesi veri ───────────────────────────────────────────────────
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


# ── STADIO 1: trasformazioni (ortogonali → l'errore si misura nello spazio
#    originale; una rotazione ortogonale non cambia la norma, quindi il
#    confronto resta equo) ─────────────────────────────────────────────────
def tr_nessuna(w):
    return w, None


def tr_permuta_ordine(w):
    """similarity_reordering: ordina i pesi per modulo dentro il blocco.
    Raggruppa i grandi coi grandi: una scala unica per blocco li serve meglio."""
    idx = np.argsort(np.abs(w), axis=1)
    return np.take_along_axis(w, idx, axis=1), idx


def tr_haar(w):
    """local_haar: una passata di trasformata di Haar dentro il blocco.
    Ortogonale, esatta, gratuita. Sul 397B fu BOCCIATA: qui si ri-misura,
    perche' i priori sono per-modello (e' la regola del registro)."""
    x = w.copy()
    n = x.shape[1]
    while n > 1:
        a = x[:, 0:n:2]
        b = x[:, 1:n:2]
        x[:, : n // 2] = (a + b) / np.sqrt(2.0)
        x[:, n // 2:n] = (a - b) / np.sqrt(2.0)
        n //= 2
    return x, None


def tr_haar1(w):
    """Un solo livello di Haar: meno aggressivo, spesso e' dove sta il guadagno."""
    a = w[:, 0::2]
    b = w[:, 1::2]
    return np.concatenate([(a + b) / np.sqrt(2.0), (a - b) / np.sqrt(2.0)], axis=1), None


def tr_gauss(w):
    """hessian_eigenbasis_rotation, versione povera e SENZA Hessiana: una
    rotazione fissa (Hadamard) che gaussianizza il blocco. Sul 397B la
    rotazione in base agli autovettori diede -14%; qui si misura quanto di
    quel guadagno arriva dalla sola gaussianizzazione, che e' gratis."""
    n = w.shape[1]
    h = np.ones((1, 1), dtype=np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    h = h[:n, :n] / np.sqrt(n)
    return w @ h.T, None


TRASF = {"nessuna": tr_nessuna, "ordina": tr_permuta_ordine,
         "haar_full": tr_haar, "haar_1": tr_haar1, "hadamard": tr_gauss}


# ── STADIO 2: regole di scala (tutte a costo zero bit) ──────────────────────
def _err(w, s, thr):
    q = np.sign(w) * (np.abs(w) > thr[:, None])
    d = w - s[:, None] * q
    return np.sqrt((d * d).sum(1) / np.maximum((w * w).sum(1), 1e-30))


def sc_absmax(w):
    s = np.abs(w).max(1)
    return _err(w, s, 0.5 * s)


def sc_twn(w):
    a = np.abs(w)
    thr = 0.7 * a.mean(1)
    m = a > thr[:, None]
    s = np.where(m.sum(1) > 0, (a * m).sum(1) / np.maximum(m.sum(1), 1), a.mean(1))
    return _err(w, s, thr)


def sc_ls(w, giri=2):
    a = np.abs(w)
    thr = 0.7 * a.mean(1)
    s = a.mean(1)
    for _ in range(giri):
        q = np.sign(w) * (a > thr[:, None])
        nq = (q * q).sum(1)
        s = np.where(nq > 0, (w * q).sum(1) / np.maximum(nq, 1e-30), a.mean(1))
        thr = 0.5 * s
    return _err(w, s, thr)


def sc_grid(w):
    a = np.abs(w)
    med = a.mean(1)
    best = np.full(w.shape[0], np.inf)
    for ft in np.arange(0.40, 1.05, 0.05):
        thr = ft * med
        q = np.sign(w) * (a > thr[:, None])
        nq = (q * q).sum(1)
        s0 = np.where(nq > 0, (w * q).sum(1) / np.maximum(nq, 1e-30), med)
        for fs in np.arange(0.85, 1.20, 0.05):
            best = np.minimum(best, _err(w, s0 * fs, thr))
    return best


SCALE = {"absmax": sc_absmax, "twn": sc_twn, "ls_iter": sc_ls, "grid": sc_grid}


def main() -> int:
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/models/gguf/odino-q8")
    nblk = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
    tens = sorted(d.glob("*.bin"))[: int(sys.argv[3]) if len(sys.argv) > 3 else 12]
    if not tens:
        print(f"nessun tensore in {d}")
        return 1

    print(f"STUDIO INCROCIATO ESAUSTIVO — {len(SCALE)} scale x {len(TRASF)} trasformazioni "
          f"= {len(SCALE)*len(TRASF)} combinazioni", flush=True)
    print(f"su {len(tens)} tensori veri x {nblk} blocchi = "
          f"{len(tens)*nblk*T:,} pesi\n", flush=True)

    acc = {(s, t): [] for s in SCALE for t in TRASF}
    for i, f in enumerate(tens, 1):
        try:
            w0 = carica(f, nblk)
        except Exception as e:                                   # noqa: BLE001
            print(f"   salto {f.name}: {str(e)[:50]}", flush=True)
            continue
        for tn, tf in TRASF.items():
            wt, _ = tf(w0)
            for sn, sf in SCALE.items():
                acc[(sn, tn)].append(float(sf(wt).mean()))
        print(f"   [{i}/{len(tens)}] {f.name}", flush=True)

    med = {k: float(np.mean(v)) for k, v in acc.items() if v}
    base = med[("twn", "nessuna")]
    print(f"\nriferimento (twn + nessuna trasformazione, la ricetta di oggi): "
          f"{base:.6f}\n")
    print(f"{'scala':<9s}" + "".join(f"{t:>12s}" for t in TRASF))
    for sn in SCALE:
        riga = f"{sn:<9s}"
        for tn in TRASF:
            v = med.get((sn, tn))
            riga += f"{100*(v-base)/base:>11.2f}%" if v is not None else f"{'—':>12s}"
        print(riga)

    print("\n— INTERAZIONE: la trasformazione migliore cambia con la scala? —")
    migliori = {}
    for sn in SCALE:
        t = min(TRASF, key=lambda tn: med.get((sn, tn), np.inf))
        migliori[sn] = t
        print(f"   con scala {sn:<9s} la trasformazione migliore e' {t}")
    if len(set(migliori.values())) == 1:
        print("   → le due leve sono INDIPENDENTI: si possono scegliere separatamente.")
    else:
        print("   → le due leve sono ACCOPPIATE: vanno scelte INSIEME.")
        print("     Un registro che le descrive isolate non puo' dirlo.")

    vinc = min(med, key=med.get)
    print(f"\nMIGLIORE COMBINAZIONE: scala={vinc[0]} + trasformazione={vinc[1]}")
    print(f"   errore {med[vinc]:.6f} contro {base:.6f} della ricetta di oggi "
          f"= {100*(med[vinc]-base)/base:+.2f}%  a PARITA' DI BIT")
    print("\n⚠️ LIMITE: questo studio copre SOLO le leve che agiscono sul tensore.")
    print("   Allocazione, riparazione, runtime e calibrazione agiscono sul modello")
    print("   intero e richiedono una forgia: NON sono coperte qui.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

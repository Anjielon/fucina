#!/usr/bin/env python3
"""QUANTO PESA LA CORREZIONE, STRATO PER STRATO?

Dieci ipotesi cadute sul perche' il danno sia una montagna centrata sullo
strato 20. Ne resta una banale che nessuno ha ancora misurato: forse li' la
correzione e' semplicemente PIU' GRANDE rispetto a cio' che corregge.

Si legge dal file, senza GPU e senza caricare il modello: per ogni strato il
rapporto fra la norma del secondo piano e quella del primo, sui soli esperti
caldi che il secondo piano copre.

Predizione secca: il rapporto deve avere un MASSIMO intorno allo strato 20 e
valori bassi su 30-39, dove la correzione aiuta.

Se invece e' piatto, cade anche questa — e resta il fatto piu' scomodo: una
correzione della stessa taglia relativa fa danni quattro volte diversi a
seconda di dove si applica.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/home/angelo/build-llamacpp-tq1/gguf-py")
from gguf import GGUFReader, quants as GQ  # noqa: E402

MODELLO = "/mnt/models/gguf/tony-tern/Tony-tern-TQ1.gguf"
DANNO = {0: 10.9524, 5: 8.7277, 10: 8.8125, 15: 20.1161, 20: 37.8030,
         25: 22.5109, 30: 8.6860, 35: 8.7743, 39: 8.7370}


def blocco(t, pos: int, n_esp: int) -> np.ndarray:
    d = np.asarray(t.data)
    per_e = d.shape[0] // n_esp
    return GQ.dequantize(d[pos * per_e:(pos + 1) * per_e],
                         t.tensor_type).astype(np.float32)


def main() -> None:
    r = GGUFReader(MODELLO)
    per_nome = {t.name: t for t in r.tensors}
    print("Rapporto fra la norma del SECONDO piano e quella del PRIMO,")
    print("mediato su un campione di esperti caldi.\n")
    print("  strato      up      gate      down     danno")
    righe = []
    for L in range(64):
        n1 = f"blk.{L}.ffn_up_exps.weight"
        n2 = f"blk.{L}.ffn_up_exps2.weight"
        if n1 not in per_nome or n2 not in per_nome:
            continue
        rap = {}
        for proj in ("up", "gate", "down"):
            a = per_nome.get(f"blk.{L}.ffn_{proj}_exps.weight")
            b = per_nome.get(f"blk.{L}.ffn_{proj}_exps2.weight")
            if a is None or b is None:
                continue
            n_esp1 = int(a.shape[2])
            n_esp2 = int(b.shape[2])
            v = []
            for pos in range(0, min(n_esp2, 8)):     # campione di caldi
                W1 = blocco(a, pos, n_esp1)
                W2 = blocco(b, pos, n_esp2)
                d1 = float(np.linalg.norm(W1))
                if d1 > 0:
                    v.append(float(np.linalg.norm(W2)) / d1)
            if v:
                rap[proj] = float(np.mean(v))
        if len(rap) == 3:
            righe.append((L, rap))
            if L in DANNO:
                print(f"  {L:5d}   {rap['up']:7.4f}  {rap['gate']:7.4f}  "
                      f"{rap['down']:7.4f}  {DANNO[L]:8.4f}")
    if not righe:
        print("⛔ nessuno strato con entrambi i piani")
        return

    comuni = [(L, r) for L, r in righe if L in DANNO]
    if len(comuni) >= 4:
        y = np.array([DANNO[L] for L, _ in comuni])
        print()
        for proj in ("up", "gate", "down"):
            x = np.array([r[proj] for _, r in comuni])
            print(f"  correlazione {proj:5s} / danno: "
                  f"{float(np.corrcoef(x, y)[0, 1]):+.3f}")
        Lp = max(comuni, key=lambda t: DANNO[t[0]])[0]
        Lm = min(comuni, key=lambda t: DANNO[t[0]])[0]
        rp = dict(comuni)[Lp]
        rm = dict(comuni)[Lm]
        print(f"\n  strato piu' danneggiato ({Lp}): up {rp['up']:.4f} · "
              f"gate {rp['gate']:.4f} · down {rp['down']:.4f}")
        print(f"  strato meno danneggiato ({Lm}): up {rm['up']:.4f} · "
              f"gate {rm['gate']:.4f} · down {rm['down']:.4f}")
        salto = max(abs(rp[p] - rm[p]) / max(rm[p], 1e-9) for p in rp)
        if salto < 0.15:
            print(f"\n  ⇒ La correzione ha la STESSA taglia relativa "
                  f"(scarto {salto * 100:.1f}%) sui due strati")
            print("    che si comportano in modo opposto. La taglia non spiega")
            print("    il danno: uguale causa, effetto quattro volte diverso.")
        else:
            print(f"\n  ⇒ La taglia cambia del {salto * 100:.1f}% fra i due "
                  "strati: candidato da verificare con una prova diretta.")


if __name__ == "__main__":
    main()

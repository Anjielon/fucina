#!/usr/bin/env python3
"""IL TRAFFICO E' CONCENTRATO ALLO STESSO MODO SU TUTTI GLI STRATI?

Il secondo piano copre i 28 esperti piu' caldi su 256. Se a certi strati il
traffico e' distribuito in modo piu' UNIFORME, quei 28 catturano una fetta
minore, la maggioranza del traffico resta al piano singolo, e la disparita'
fra esperti corretti e non corretti pesa di piu'.

Predizione secca: la concentrazione deve avere un MINIMO dove il danno ha il
suo massimo (strato 20 sul modello piccolo), e valori alti dove il secondo
piano aiuta (30-39).

Se invece la concentrazione e' piatta, questa spiegazione muore — e il fatto
che sia calcolabile PRIMA di forgiare qualsiasi cosa, dai soli `counts`
dell'imatrix, e' esattamente il motivo per cui vale la pena chiederselo: se
reggesse, la forgia potrebbe scegliere K strato per strato invece di fissarlo.

Nessuna GPU, nessun modello caricato: legge solo l'imatrix.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/home/angelo/build-llamacpp-tq1/gguf-py")
from gguf import GGUFReader  # noqa: E402

IMATRIX = "/mnt/models/gguf/tony-tern/tony-imatrix.gguf"
K = 28
# danno misurato col secondo piano acceso su un solo strato (12 blocchi,
# riferimento a piano spento 8.7204)
DANNO = {0: 10.9524, 5: 8.7277, 10: 8.8125, 15: 20.1161, 20: 37.8030,
         25: 22.5109, 30: 8.6860, 35: 8.7743, 39: 8.7370}
RIFERIMENTO = 8.7204


def main() -> None:
    conteggi: dict[int, np.ndarray] = {}
    for t in GGUFReader(IMATRIX).tensors:
        if t.name.endswith(".ffn_gate_exps.weight.counts"):
            conteggi[int(t.name.split(".")[1])] = np.array(
                t.data).astype(np.float64).ravel()
    if not conteggi:
        print("⛔ nessun `counts` per esperto nell'imatrix — niente da misurare")
        return

    n_exp = len(next(iter(conteggi.values())))
    print(f"{len(conteggi)} strati · {n_exp} esperti · i primi {K} portano:\n")
    print("  strato   quota dei 28 caldi   entropia normalizzata   danno")
    quote: dict[int, float] = {}
    entropie: dict[int, float] = {}
    for L in sorted(conteggi):
        c = np.sort(conteggi[L])[::-1]
        tot = c.sum()
        if tot <= 0:
            continue
        quota = float(c[:K].sum() / tot)
        p = c / tot
        p = p[p > 0]
        ent = float(-(p * np.log(p)).sum() / np.log(n_exp))
        quote[L] = quota
        entropie[L] = ent
        if L in DANNO or L % 10 == 0:
            d = f"{DANNO[L]:8.4f}" if L in DANNO else "       —"
            print(f"  {L:5d}   {quota * 100:14.2f}%   {ent:19.4f}   {d}")

    comuni = [L for L in DANNO if L in quote]
    if len(comuni) >= 4:
        x = np.array([quote[L] for L in comuni])
        e = np.array([entropie[L] for L in comuni])
        y = np.array([DANNO[L] for L in comuni])
        rq = float(np.corrcoef(x, y)[0, 1])
        re_ = float(np.corrcoef(e, y)[0, 1])
        print(f"\n  correlazione quota-dei-caldi / danno:  {rq:+.3f}")
        print(f"  correlazione entropia / danno:        {re_:+.3f}")
        Lp = max(comuni, key=lambda L: DANNO[L])
        Lm = min(comuni, key=lambda L: DANNO[L])
        print(f"\n  strato piu' danneggiato ({Lp}): quota {quote[Lp] * 100:.2f}%"
              f" · entropia {entropie[Lp]:.4f}")
        print(f"  strato meno danneggiato ({Lm}): quota {quote[Lm] * 100:.2f}%"
              f" · entropia {entropie[Lm]:.4f}")
        if rq < -0.6:
            print("\n  ⇒ Dove il traffico e' MENO concentrato il danno e' maggiore:")
            print("    i 28 caldi coprono troppo poco, e la disparita' pesa.")
            print("    Conseguenza pratica: K va scelto PER STRATO, e si sa gia'")
            print("    prima di forgiare, dai soli `counts` dell'imatrix.")
        elif abs(rq) < 0.4 and abs(re_) < 0.4:
            print("\n  ⇒ La concentrazione NON spiega il danno: e' scorrelata.")
            print("    Un'altra ipotesi che cade, e per fortuna costava un minuto.")
        else:
            print(f"\n  ⇒ Correlazione presente ma non netta (quota {rq:+.3f},"
                  f" entropia {re_:+.3f}): indizio, non spiegazione.")
    else:
        print("\n  ⚠️ troppi pochi strati in comune fra danno e imatrix")


if __name__ == "__main__":
    main()

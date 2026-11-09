#!/usr/bin/env python3
"""29/9 tt3-cure: conta strati percorsi nella fase V del passo V 2 di TT3 (sequenza di `seleziona_e_prova_dosi` con
punto medio a due funzionali, dosi golose, ricentro), senza e con --pv-riuso. Solo aritmetica: nessun modello.
Primitive, ordine dei prezzi, ordine goloso e tenute: dal registro di TT3 (riga del passo V 2)."""
import json
import sys

N = 40
PREZZO = [39, 28, 0, 33, 29, 39, 14, 32, 35, 26, 29, 30]         # ordine dei prezzi (guadagno previsto)
GOLOSA = [0, 14, 29, 29, 28, 33, 35, 26, 30, 32, 39, 39]         # ordine delle dosi golose (registro)
TENUTE = [True, True, True] + [False] * 9
GIRI_RICENTRO = 2                                                # giri del ricentro (registro: 2)
MAX_LOTTI = int(sys.argv[1]) if len(sys.argv) > 1 else 2


class Sim:
    def __init__(self, riuso: bool):
        self.riuso = riuso
        self.ver = [0] * N
        self.cache = {}                    # lotto -> versioni viste
        self.ordine = []
        self.fw = self.bw = 0

    def tocca(self, L):
        self.ver[L] += 1

    def avanti(self, lotto):
        if not self.riuso or lotto not in self.cache:
            i0 = 0
        else:
            v = self.cache[lotto]
            i0 = next((i for i in range(N) if v[i] != self.ver[i]), N)
        self.fw += N - i0
        if self.riuso:
            self.cache[lotto] = list(self.ver)
            if lotto in self.ordine:
                self.ordine.remove(lotto)
            self.ordine.append(lotto)
            while len(self.ordine) > MAX_LOTTI:
                del self.cache[self.ordine.pop(0)]

    def gradiente(self, lotto, strato_min):
        self.avanti(lotto)
        self.bw += (N - strato_min) if self.riuso else N


def fase_v(riuso: bool) -> Sim:
    s = Sim(riuso)
    s.avanti("A")                                   # sonda A di partenza
    for lotto in ("A", "M1"):                       # prezzi al punto medio: B (sonda A) e A (lettere M1)
        s.gradiente(lotto, min(PREZZO))
        for L in PREZZO:
            s.tocca(L)
            s.gradiente(lotto, L)
            s.tocca(L)
    s.avanti("M2")                                  # obiettivo di partenza (conferma)
    s.avanti("B")                                   # sonda B di partenza
    s.avanti("M1")                                  # lettere M1 di partenza (golosa)
    for L, t in zip(GOLOSA, TENUTE):
        s.tocca(L)
        s.avanti("A")
        s.avanti("M1")
        if not t:
            s.tocca(L)
    s.avanti("B")                                   # conferma B
    tenuti = [L for L, t in zip(GOLOSA, TENUTE) if t]
    s.gradiente("A", min(tenuti))                   # ricentro
    for _ in range(GIRI_RICENTRO):
        s.tocca(min(tenuti))
        s.avanti("B")
    s.avanti("M2")                                  # conferma dell'obiettivo
    return s


if __name__ == "__main__":
    out = {}
    for tag, r in (("spento", False), ("acceso", True)):
        s = fase_v(r)
        out[tag] = {"strati_avanti": s.fw, "strati_indietro": s.bw}
    for c in (1.5, 2.0, 3.0):                       # costo di uno strato indietro (ricalcolo + indietro) in avanti
        u = {k: out[k]["strati_avanti"] + c * out[k]["strati_indietro"] for k in ("spento", "acceso")}
        out[f"rapporto_c{c}"] = round(u["acceso"] / u["spento"], 4)
    print(json.dumps(out, indent=1))

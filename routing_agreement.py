#!/usr/bin/env python3
"""LA DERIVA VA VERSO IL VERO O LONTANO DAL VERO? — versione valida.

⛔ Il primo tentativo confrontava gli INDICI degli esperti fra Tony e la sua
sorgente. Non hanno lo stesso significato: la forgia applica la permutazione
caldi-primi, quindi l'esperto 5 di Tony non e' l'esperto 5 della sorgente.
Risultato: accordo del 4.4%, che e' esattamente il caso (8 su 256 = 3.1%).
Rumore presentato come misura.

Qui la permutazione si RICAVA DAI DATI invece di ricostruirla dal codice: la
forgia permuta anche le righe del router, quindi ogni riga del router di Tony
ha una gemella quasi identica nel modello sorgente. Si accoppiano per
correlazione, si verifica che l'accoppiamento sia biunivoco e stretto, e solo
allora si confrontano le rotte.

Se l'accoppiamento non e' pulito lo script SI FERMA invece di produrre un
numero: un accoppiamento sbagliato darebbe di nuovo rumore travestito.
"""
from __future__ import annotations

import glob
import re
import sys
import os

import numpy as np

sys.path.insert(0, os.environ.get("GGUF_PY", os.path.expanduser("~/build-llamacpp-tq1/gguf-py")))
from gguf import GGUFReader, quants as GQ  # noqa: E402

TONY = "/mnt/models/gguf/tony-tern/Tony-tern-TQ1.gguf"
SORGENTE = "/mnt/models/gguf/ornith-1.5/Ornith-1.5-35B-A3B-APEX-MTP-I-Quality.gguf"
N_USED = 8
SOGLIA_ACCOPPIAMENTO = 0.90


def router(r: GGUFReader, L: int) -> np.ndarray | None:
    """Righe del router dello strato L, normalizzate: (n_expert, hidden)."""
    t = next((x for x in r.tensors if x.name == f"blk.{L}.ffn_gate_inp.weight"), None)
    if t is None:
        return None
    W = GQ.dequantize(np.asarray(t.data), t.tensor_type).astype(np.float32)
    ne = [int(x) for x in t.shape]
    W = W.reshape(ne[1], ne[0])
    n = np.linalg.norm(W, axis=1, keepdims=True)
    return W / np.maximum(n, 1e-12)


def permutazione(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, float, bool]:
    """perm[i] = riga di B che corrisponde alla riga i di A."""
    C = A @ B.T
    perm = np.argmax(C, axis=1)
    qualita = float(np.median(C[np.arange(len(perm)), perm]))
    biunivoca = len(set(perm.tolist())) == len(perm)
    return perm, qualita, biunivoca


def carica(d: str, L: int, E: int) -> np.ndarray | None:
    p = glob.glob(f"{d}/*-{L}.f32")
    if not p:
        return None
    a = np.fromfile(p[0], dtype=np.float32)
    if a.size == 0 or a.size % E:
        return None
    return a.reshape(-1, E)


def insiemi(A: np.ndarray) -> list[set[int]]:
    return [set(r.tolist()) for r in np.argsort(-A, axis=1)[:, :N_USED]]


def accordo(X: list[set[int]], Y: list[set[int]]) -> float:
    n = min(len(X), len(Y))
    if not n:
        return float("nan")
    return sum(len(X[i] & Y[i]) for i in range(n)) / (n * N_USED) * 100


def main() -> None:
    rt, rs = GGUFReader(TONY), GGUFReader(SORGENTE)
    print("ACCOPPIAMENTO DELLE RIGHE DEL ROUTER (verifica preliminare)\n")
    perms: dict[int, np.ndarray] = {}
    for L in range(64):
        A, B = router(rt, L), router(rs, L)
        if A is None or B is None or A.shape != B.shape:
            continue
        perm, q, bi = permutazione(A, B)
        perms[L] = perm
        if L in (0, 1, 20, 39):
            stato = "✓" if (q >= SOGLIA_ACCOPPIAMENTO and bi) else "⛔"
            print(f"  strato {L:2d}: corrispondenza mediana {q:.4f} "
                  f"biunivoca={bi}  {stato}")
    if not perms:
        print("⛔ nessuno strato accoppiabile — mi fermo")
        return
    qs = []
    for L, perm in perms.items():
        A, B = router(rt, L), router(rs, L)
        _, q, bi = permutazione(A, B)
        qs.append((q, bi))
    qmed = float(np.median([q for q, _ in qs]))
    tutte_bi = all(bi for _, bi in qs)
    print(f"\n  su {len(perms)} strati: corrispondenza mediana {qmed:.4f}, "
          f"tutte biunivoche: {tutte_bi}")
    if qmed < SOGLIA_ACCOPPIAMENTO or not tutte_bi:
        print("\n⛔ ACCOPPIAMENTO NON AFFIDABILE — non produco numeri sulle rotte.")
        print("   Un confronto su indici mal accoppiati e' rumore travestito,")
        print("   ed e' esattamente l'errore che questa versione doveva evitare.")
        return

    print("\n\nACCORDO CON LA SORGENTE, a indici allineati\n")
    print("  strato   piano 1 solo   piano 1+2      verso dove")
    t1 = t2 = 0.0
    n = 0
    for L in sorted(perms):
        S = carica("/tmp/instrad2_sorgente", L, 256)
        P1 = carica("/tmp/instrad2_spento", L, 256)
        P2 = carica("/tmp/instrad2_acceso", L, 256)
        if S is None or P1 is None or P2 is None:
            continue
        m = min(len(S), len(P1), len(P2))
        S, P1, P2 = S[:m], P1[:m], P2[:m]
        perm = perms[L]
        # riscrive le probabilita' di Tony nell'ordine della sorgente
        P1s = np.zeros_like(P1)
        P2s = np.zeros_like(P2)
        P1s[:, perm] = P1
        P2s[:, perm] = P2
        sS = insiemi(S)
        a, b = accordo(sS, insiemi(P1s)), accordo(sS, insiemi(P2s))
        t1 += a
        t2 += b
        n += 1
        if L in (0, 1, 2, 5, 10, 20, 30, 39):
            verso = ("→ VERSO il vero" if b > a + 0.1 else
                     "← LONTANO dal vero" if a > b + 0.1 else "= uguale")
            print(f"  {L:5d}    {a:8.2f}%     {b:8.2f}%     {verso}")
    if not n:
        print("⛔ nessuno strato con tutte e tre le catture")
        return
    m1, m2 = t1 / n, t2 / n
    print(f"\n  MEDIA su {n} strati:  piano 1 = {m1:.2f}%   piano 1+2 = {m2:.2f}%")
    print(f"  (accordo casuale, 8 su 256: {N_USED / 256 * 100:.2f}%)")
    if m1 < 10 and m2 < 10:
        print("\n  ⚠️ Entrambi vicini al caso: le rotte del ternario non hanno")
        print("     quasi nulla in comune con quelle della sorgente. Prima di")
        print("     leggere il confronto fra i due piani, questo va spiegato.")
    elif m2 > m1 + 0.5:
        print("\n  ⇒ La correzione avvicina le rotte alla SORGENTE e il modello")
        print("    peggiora lo stesso: e' CO-ADATTATO alle rotte del piano solo.")
    elif m1 > m2 + 0.5:
        print("\n  ⇒ La correzione ALLONTANA le rotte dalla sorgente: disturbo.")
    else:
        print("\n  ⇒ Le rotte cambiano molto senza spostarsi ne' verso ne'")
        print("    lontano dalla sorgente: rimescolamento al confine del top-k.")


if __name__ == "__main__":
    main()

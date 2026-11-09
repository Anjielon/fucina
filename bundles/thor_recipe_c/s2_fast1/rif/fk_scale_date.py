"""Copia di solutori.fastkron_composito (FastKron 2608.06291 Alg. 6) con le scale di gruppo PASSATE (per alternare codici/scale)."""
import torch
from solutori import _L_unit, tabella_trit, F64
from typing import Tuple, Dict

def fastkron_scale_date(Wres, A, B, ng, S_dato, Kl=1, damp=0.01) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """FastKron (arXiv 2608.06291, Alg. 6: antidiagonali in parallelo + divide-et-impera, equivalente per il Teorema 2.1
    a GPTQ sul peso vettorizzato con curvatura A ⊗ B) con il quantizzatore di base = valore piu' vicino del composito
    a Kl piani ternari, scale STATICHE per (riga, gruppo) fissate sul bersaglio (ricetta residua di casa).
    A [C,C] curvatura d'ingresso, B [R,R] curvatura d'uscita (YAQA: E[g gᵀ] della KL)."""
    R, C = Wres.shape
    G = C // ng
    dev = Wres.device
    LA = _L_unit(A.to(F64), damp)
    LB = _L_unit(B.to(F64), damp) if B is not None else torch.eye(R, dtype=F64, device=dev)
    W = Wres.to(F64).clone()
    tab = tabella_trit(Kl, dev).to(F64)
    S = S_dato.to(F64).clone()
    ALF = torch.einsum("krg,ak->rga", S, tab)                         # [R, ng, 3^Kl]
    V = torch.zeros_like(W)
    IDX = torch.zeros(R, C, dtype=torch.long, device=dev)
    m, n = R, C
    ii = torch.arange(m, device=dev).view(-1, 1)
    jj = torch.arange(n, device=dev).view(1, -1)

    def box(k, l):
        return max(0, k - n + 1), min(m, l), max(0, k - m + 1), min(n, l)

    def fk(k, l):
        if l == k + 1:
            r0 = max(0, k - n + 1); r1 = min(m, k + 1)
            if r0 >= r1:
                return
            ri = torch.arange(r0, r1, device=dev)
            cj = k - ri
            w = W[ri, cj]
            alf = ALF[ri, cj // G]                                    # [len, 3^Kl]
            j = (alf - w.unsqueeze(1)).abs().argmin(1)
            V[ri, cj] = alf.gather(1, j.unsqueeze(1)).squeeze(1)
            IDX[ri, cj] = j
            return
        s = (k + l) // 2
        fk(k, s)
        a0, a1, b0, b1 = box(k, s)                                    # sorgente: antidiagonali [k, s)
        t0, t1, u0, u1 = box(s, l)                                    # bersaglio: antidiagonali [s, l)
        if a0 < a1 and b0 < b1 and t0 < t1 and u0 < u1:
            d = ii[a0:a1] + jj[:, b0:b1]
            msk = (d >= k) & (d < s)
            Dl = torch.where(msk, V[a0:a1, b0:b1] - W[a0:a1, b0:b1], torch.zeros((), dtype=F64, device=dev))
            Up = LB[t0:t1, a0:a1] @ Dl @ LA[u0:u1, b0:b1].T
            d2 = ii[t0:t1] + jj[:, u0:u1]
            W[t0:t1, u0:u1] += torch.where((d2 >= s) & (d2 < l), Up, torch.zeros((), dtype=F64, device=dev))
        fk(s, l)

    import sys as _s
    _s.setrecursionlimit(max(10000, _s.getrecursionlimit()))
    fk(0, m + n - 1)
    T = tab[IDX].permute(2, 0, 1).to(torch.int8).contiguous()        # [Kl, R, C]
    return T, S, {}



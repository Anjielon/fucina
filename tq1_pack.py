"""TQ1_0 — impacchettamento GPU e riordino esperti (modulo della Fucina).

Formato TQ1_0 (llama.cpp): blocco di 256 pesi ternari in 54 byte —
52 di dati (base 3: 5 valori/byte nei primi 48, 4 valori/byte negli ultimi 4)
+ 2 di scala float16. 1,69 bit per peso.

Estratto dalla forgia di ODINO (26/8) per rendere la Fucina self-contained.
Autotest in fondo: il pacchetto DEVE combaciare col decodificatore ufficiale
(gguf.quants) — la lezione del fork GuthL (troncamento a 8 bit taciuto).
"""
from __future__ import annotations
import numpy as np
import torch


def impacchetta(d: np.ndarray, q: np.ndarray) -> np.ndarray:
    """d:(n,1) scala f16 · q:(n,256) in {-1,0,1} → (n,54) uint8 TQ1_0.

    ⚡ GPU a pezzi: in numpy costava 60-100 s per tensore da 2,1 mld di pesi;
    in torch su GPU: secondi (misurato nella forgia di ODINO).
    """
    n_r = q.shape[0]
    out = np.empty((n_r, 54), np.uint8)
    passo = 8_000_000
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    p5 = torch.tensor([81, 27, 9, 3, 1], dtype=torch.int32, device=dev).view(1, 1, 5, 1)
    p4 = torch.tensor([81, 27, 9, 3], dtype=torch.int32, device=dev).view(1, 1, 4, 1)
    for i in range(0, n_r, passo):
        j = min(i + passo, n_r); m = j - i
        qs = torch.from_numpy(q[i:j].astype(np.int8)).to(dev, torch.int32) + 1
        a = (qs[:, :160].reshape(m, -1, 5, 32) * p5).sum(-2).reshape(m, -1)
        b = (qs[:, 160:240].reshape(m, -1, 5, 16) * p5).sum(-2).reshape(m, -1)
        c = (qs[:, 240:].reshape(m, -1, 4, 4) * p4).sum(-2).reshape(m, -1)
        t = torch.cat([a, b, c], -1)
        t = ((t * 256 + 242) // 243).to(torch.uint8)
        out[i:j, :52] = t.cpu().numpy()
        out[i:j, 52:] = d[i:j].astype(np.float16).view(np.uint8)
    return out


def permutazione_caldi(caldi: list[int], E: int) -> np.ndarray:
    """[caldi] + [tutti gli altri in ordine] — l'ordine che il motore a 2 piani
    assume: il piano-2 copre il PREFISSO 0..k-1.
    ⛔ Lezione ODINO v3.1: il motore lo assumeva, la forgia non lo faceva →
    correzioni sommate agli esperti sbagliati. La permutazione va fatta ALLA
    NASCITA del file, righe del router comprese."""
    perm = np.array(list(caldi) + [e for e in range(E) if e not in set(caldi)])
    assert len(perm) == E and len(set(perm.tolist())) == E, "perm non valida"
    return perm


def riordina_esperti(pacchi: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """pacchi (E*nb, 54) uint8 → esperti riordinati secondo perm."""
    E = len(perm)
    v = pacchi.reshape(E, -1, 54)
    return np.ascontiguousarray(v[perm]).reshape(-1, 54)


def _autotest() -> None:
    """pacchetto == decodificatore ufficiale, su valori casuali."""
    import sys
    sys.path.insert(0, "/home/angelo/build-llamacpp/gguf-py")
    from gguf import quants as GQ
    from gguf import GGMLQuantizationType as T
    rng = np.random.default_rng(7)
    q = rng.integers(-1, 2, size=(8, 256)).astype(np.int8)
    d = rng.random((8, 1)).astype(np.float32) * 0.1
    p = impacchetta(d, q)
    ric = GQ.dequantize(p, T.TQ1_0).reshape(8, 256)
    att = q.astype(np.float32) * d
    assert np.allclose(ric, att, atol=1e-3), "impacchetta ≠ decodificatore ufficiale"
    print("✓ autotest tq1_pack: pacchetto == decodificatore ufficiale")


if __name__ == "__main__":
    _autotest()

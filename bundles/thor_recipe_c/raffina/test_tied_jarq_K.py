"""CPU test of the tied_jarq fix (9/10): all K host planes are free and tied 3^(K-1-k) (K=4 -> 27:9:3:1, uniform
81-level grid; K=3 -> 9:3:1; K=2 -> 3:1). Before the fix the free planes were hard-coded [0, 1, 2] and K=4 families
(4th plane of PV12P) raised `shape '[3, 1, 1, 1]' is invalid for input of size 4` in scale_congiunte.
usage: HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= python -m pytest -q test_tied_jarq_K.py
"""
import os
import sys

os.environ.setdefault("LEVE_DT", "float64")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pytest  # noqa: E402
import torch  # noqa: E402

import applica as AP  # noqa: E402
import solutori as SO  # noqa: E402


def _H(C, seed=1):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(4 * C, C, generator=g, dtype=torch.float64) @ torch.diag(torch.linspace(0.2, 2.0, C, dtype=torch.float64))
    H = X.T @ X
    return H / torch.diagonal(H).mean()


class _Host:
    """Minimal stand-in for comune.Denso: host planes/scales with K planes (only shapes matter for tied_jarq)."""
    def __init__(self, K, R, C, ng, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.T = torch.randint(-1, 2, (K, R, C), generator=g, dtype=torch.int8)
        self.S = SO.fp16(torch.rand(K, R, ng, generator=g, dtype=torch.float64) * 1e-3).float()
        self.nome = f"test.K{K}"


def _case(K, R=24, C=256, ng=2, seed=0):
    torch.manual_seed(seed)
    W = torch.randn(R, C, dtype=torch.float64) * 0.02
    H = _H(C, seed + 3)
    return W, H, _Host(K, R, C, ng, seed)


@pytest.mark.parametrize("K", [2, 3, 4])
def test_tied_jarq_all_planes_tied(K):
    W, H, d = _case(K)
    T2, S2, cat = AP.applica_leva("tied_jarq", d, W, H)
    R, C = W.shape
    ng = d.S.shape[-1]
    assert T2.shape == (K, R, C) and S2.shape == (K, R, ng)
    assert T2.abs().max() <= 1
    S2 = S2.double()
    assert torch.equal(S2, SO.fp16(S2))                                   # scales exactly fp16
    leg = torch.tensor([3.0 ** (K - 1 - k) for k in range(K)], dtype=torch.float64)
    # exact tie after fp16: s_k = fp16(3^(K-1-k) * s_last), for every plane (27:9:3:1 when K=4)
    assert torch.equal(S2, SO.fp16(leg.view(K, 1, 1) * S2[-1:]))
    # the composite is a uniform grid with 3^K levels: W_hat = s_last * c, c integer in [-(3^K-1)/2, (3^K-1)/2],
    # up to the fp16 rounding of each stored plane scale (half an ulp = 2^-11 relative, as in BIL)
    c = (T2.double() * leg.view(K, 1, 1)).sum(0)
    assert c.abs().max() <= (3 ** K - 1) / 2
    G = C // ng
    What = (c.reshape(R, ng, G) * S2[-1].unsqueeze(-1)).reshape(R, C)
    bound = (2.0 ** -11) * (leg.sum() * S2[-1]).unsqueeze(-1).expand(R, ng, G).reshape(R, C) + 1e-15
    assert bool(((SO.ricostruisci(T2, S2) - What).abs() <= bound).all())
    # never worse than its own start (the official tied RTN) on any row, in the JARQ objective (H + nu2 I)
    T0, S0 = AP.tied_ufficiale(W, torch.zeros_like(W), K, ng, leg)
    Ha = H + 1e-4 * torch.eye(C, dtype=torch.float64)
    f0 = SO.obiettivo(W, SO.ricostruisci(T0, S0), Ha)
    f2 = SO.obiettivo(W, SO.ricostruisci(T2, S2), Ha)
    assert bool((f2 <= f0 * (1 + 1e-12)).all())
    assert cat["finale"] <= cat["ospite"] * (1 + 1e-12)


def test_K4_finer_than_K3():
    W, H, d4 = _case(4)
    _, _, d3 = _case(3)
    T4, S4, _ = AP.applica_leva("tied_jarq", d4, W, H)
    T3, S3, _ = AP.applica_leva("tied_jarq", d3, W, H)
    e4 = SO.obiettivo(W, SO.ricostruisci(T4, S4.double()), H).sum()
    e3 = SO.obiettivo(W, SO.ricostruisci(T3, S3.double()), H).sum()
    assert e4 < 0.5 * e3                                                  # one more trit = 3x finer step


def test_old_free_list_reproduces_the_crash():
    """The pre-fix call (free planes [0,1,2] with a 4-entry tie) fails exactly like the 11:26 run."""
    W, H, d = _case(4)
    leg = torch.tensor([27.0, 9.0, 3.0, 1.0], dtype=torch.float64)
    T0, S0 = AP.tied_ufficiale(W, torch.zeros_like(W), 4, d.S.shape[-1], leg)
    with pytest.raises(RuntimeError, match=r"shape '\[3, 1, 1, 1\]' is invalid for input of size 4"):
        SO.jarq(W, H, T0, S0, [0, 1, 2], giri=1, nu2=1e-4, legame=leg)


@pytest.mark.parametrize("K", [2, 3, 4])
def test_bil_tjarq_p1fermo_keeps_plane1_bit_exact(K):
    """bil_tjarq_p1fermo: host plane 1 (codes AND scale) untouched at the bit, planes 2..K rebuilt and tied 3^(K-2-k)."""
    W, H, d = _case(K, seed=7)
    # a realistic host: plane 1 = sign-ish RTN of W with its own scale (stands in for the TT2-trained codes)
    ng = d.S.shape[-1]
    T0, S0 = AP.tied_ufficiale(W, torch.zeros_like(W), K, ng,
                               torch.tensor([3.0 ** (K - 1 - k) for k in range(K)], dtype=torch.float64))
    d.T, d.S = T0.clone(), S0.float().clone()
    T2, S2, cat = AP.applica_leva("bil_tjarq_p1fermo", d, W, H)
    assert torch.equal(T2[0], d.T[0]) and torch.equal(S2[0].float(), d.S[0])
    S2 = S2.double()
    assert torch.equal(S2, SO.fp16(S2))
    if K >= 3:
        leg = torch.tensor([3.0 ** (K - 2 - k) for k in range(K - 1)], dtype=torch.float64)
        # rows JARQ cannot improve keep their BIL start, whose scales are fp16(3^k D) rounded one by one:
        # the tie holds within half an fp16 ulp (as in test_bil_cure)
        assert torch.allclose(S2[1:], S2[-1:] * leg.view(-1, 1, 1), rtol=1e-3, atol=0)
    # JARQ never worse than its BIL start on any row (H + nu2 I)
    assert cat["finale"] <= cat["bil_bil_p2k"] * (1 + 1e-9) + 1e-30

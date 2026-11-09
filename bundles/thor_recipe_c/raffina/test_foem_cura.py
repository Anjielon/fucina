"""Test CPU della cura FOEM (9/10): beta=0 riproduce BIL al bit; la forma del LAVORO (Alg. 1) e' invariante alla scala
di H (beta 0,1 vale tale e quale); la forma del CODICE dipende dalla scala e rho = beta·lambda_max(H^-1) misura la sua
stabilita' (rho > 2 diverge: e' l'esplosione del 2/9); beta riportato dalla convenzione del codice ufficiale."""
import os
import sys

os.environ.setdefault("LEVE_DT", "float64")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

import solutori as SO  # noqa: E402
from test_bil_cure import _H  # noqa: E402

torch.manual_seed(0)


def test_beta_zero_uguale_bil():
    W = torch.randn(16, 256, dtype=torch.float64) * 0.02
    H = _H(256)
    T0, S0, _ = SO.gptq_bilanciato(W, H, 3, 2)
    for forma in ("lavoro", "codice"):
        T1, S1, _ = SO.gptq_bilanciato(W, H, 3, 2, foem_beta=0.0, foem_forma=forma)
        assert torch.equal(T0, T1) and torch.equal(S0, S1)


def test_forma_lavoro_invariante_alla_scala_di_H():
    W = torch.randn(16, 256, dtype=torch.float64) * 0.02
    H = _H(256, 7)
    Ta, Sa, _ = SO.gptq_bilanciato(W, H, 3, 2, foem_beta=0.1)
    Tb, Sb, _ = SO.gptq_bilanciato(W, H * 4096.0, 3, 2, foem_beta=0.1)
    assert (Ta == Tb).float().mean() > 0.999 and torch.allclose(Sa, Sb)
    T0, _, _ = SO.gptq_bilanciato(W, H, 3, 2)
    assert (Ta != T0).any()                                            # il termine agisce davvero


def test_forma_codice_dipende_dalla_scala_e_rho():
    W = torch.randn(16, 256, dtype=torch.float64) * 0.02
    H = _H(256, 9)
    be = SO.foem_beta_efficace(0.1, 1.0)
    assert abs(be - 0.1 / 4096) < 1e-15
    assert SO.foem_rho(H, be) < SO.foem_rho(H, be * 1000)
    Ta, _, _ = SO.gptq_bilanciato(W, H, 3, 2, foem_beta=be, foem_forma="codice")
    Tb, _, _ = SO.gptq_bilanciato(W, H, 3, 2, foem_beta=be * 4096, foem_forma="codice")
    assert (Ta != Tb).any()


def test_forma_codice_diverge_oltre_rho_2():
    # passo ripetuto W <- W - beta (W - W0) H^-1: contrazione stabile se rho < 2, divergente oltre (la causa del 2/9)
    C = 64
    H = _H(C, 11)
    Hinv = torch.linalg.inv(H + 0.01 * torch.eye(C, dtype=torch.float64))
    lmax = float(torch.linalg.eigvalsh(Hinv).max())
    d0 = torch.randn(4, C, dtype=torch.float64)
    for rho, cresce in ((0.5, False), (2.5, True)):
        d = d0.clone()
        for _ in range(200):
            d = d - (rho / lmax) * d @ Hinv
        assert (d.norm() > d0.norm()) == cresce


def test_forma_riferimento_beta_zero_e_solo_colonna_successiva():
    # 12/10: forma di riferimento (GPTQModel foem.py): beta 0 = BIL al bit; con beta > 0 il richiamo verso W0 tocca solo
    # la colonna successiva, quindi resta vicina a BIL (la forma "lavoro" invece cancella la compensazione: errore x2).
    W = torch.randn(16, 256, dtype=torch.float64) * 0.02
    H = _H(256, 5)
    T0, S0, _ = SO.gptq_bilanciato(W, H, 1, 2)
    T1, S1, _ = SO.gptq_bilanciato(W, H, 1, 2, foem_beta=0.0, foem_forma="riferimento")
    assert torch.equal(T0, T1) and torch.equal(S0, S1)
    Tr, Sr, _ = SO.gptq_bilanciato(W, H, 1, 2, foem_beta=0.1, foem_forma="riferimento")
    Tl, Sl, _ = SO.gptq_bilanciato(W, H, 1, 2, foem_beta=0.1, foem_forma="lavoro")
    e = lambda T, S: float(SO.obiettivo(W, SO.ricostruisci(T, S), H).sum())
    assert (Tr != T0).any()
    assert e(Tr, Sr) < e(Tl, Sl)

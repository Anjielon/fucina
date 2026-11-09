"""Estende le prove CPU della guardia, senza aprire HIP."""
import contextlib
import io
import os
import types
import unittest

import pianificatore_gpu as PG


class GuardiaUnicaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("GUARDIA_COLLAUDA_CPU") != "1":
            raise unittest.SkipTest("richiede GUARDIA_COLLAUDA_CPU=1")
        PG.guardia_host(*PG.misura_host())
        import torch
        torch.set_num_threads(1)
        cls.t = torch

    def test_due_installazioni_stessa_algebra_stesso_wrapper(self):
        import cardan_guardia_gpu as G
        import cardan_algebra_guardia_gpu as A
        cp = types.SimpleNamespace(jarq_lotto=lambda *a, **kw: None)
        G.installa(cp)
        c1 = A.installa(cp)
        funzioni = [cp.chol_ex, cp.chol, cp.risolvi, cp.risolvi_chol, cp.jarq_lotto]
        c2 = A.installa(cp)
        self.assertEqual(funzioni, [cp.chol_ex, cp.chol, cp.risolvi, cp.risolvi_chol, cp.jarq_lotto])
        self.assertIs(cp.chol_ex, G.fattorizza)
        self.assertIs(cp.risolvi_chol, G.risolvi_chol)
        self.assertIs(c1, c2)
        self.assertFalse(self.t.cuda.is_initialized())

    def test_diagnosi_non_attribuisce_alla_sorgente_danno_device(self):
        import cardan_guardia_gpu as G
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaisesRegex(
                self.t.linalg.LinAlgError, "integrita'.*non verificata"):
            G.chol(self.t.tensor([[1., 2.], [2., 1.]], dtype=self.t.float64))
        self.assertFalse(self.t.cuda.is_initialized())

    def test_jarq_non_finito_rifiutato_prima_del_backend(self):
        import cardan_algebra_guardia_gpu as A
        def vietato(*a, **kw):
            self.fail("il backend non deve ricevere NaN")
        cp = types.SimpleNamespace(jarq_lotto=vietato)
        A.installa(cp)
        x = self.t.ones(1, 2, 2)
        with self.assertRaisesRegex(FloatingPointError, "ingresso non finito"):
            cp.jarq_lotto(x * float("nan"), x, x, x)
        self.assertFalse(self.t.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()

"""Closure 22/10: D19 (no-KFD hosts) and D37 (peak check vs reservation) switches of pianificatore_gpu. Stdlib only."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pianificatore_gpu as PG


class CodeSenzaKfd(unittest.TestCase):
    def test_default_unchanged(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"FORGIA_CODE_SENZA_KFD": ""}):
            self.assertEqual(PG.code_senza_kfd("non_misurate", d), "non_misurate")

    def test_switch_on_host_without_kfd(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"FORGIA_CODE_SENZA_KFD": "nessun_limite"}):
            self.assertIsNone(PG.code_senza_kfd("non_misurate", d))

    def test_switch_ignored_on_kfd_host(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"FORGIA_CODE_SENZA_KFD": "nessun_limite"}):
            (Path(d) / "topology").mkdir()
            self.assertEqual(PG.code_senza_kfd({"compute": 2, "sdma": 0}, d), {"compute": 2, "sdma": 0})

    def test_invalid(self):
        with patch.dict(os.environ, {"FORGIA_CODE_SENZA_KFD": "x"}):
            with self.assertRaises(ValueError):
                PG.code_senza_kfd(None, "/nonexistent")

    def test_torch_not_imported(self):
        self.assertNotIn("torch", sys.modules)


class PiccoBudget(unittest.TestCase):
    B = PG.Budget(gpu=692060160, rss=970493952, fili=1, disco=65731)
    JITTER = {"picco_rss_byte": 973516800, "picco_gpu_byte": 692060160}   # measured on Kaggle 2xT4, 22/10

    def test_default_raw_profile_as_before(self):
        with patch.dict(os.environ, {"FORGIA_PICCO_CONTRO": "profilo"}):
            self.assertTrue(PG.picco_oltre_budget(self.JITTER, self.B))
            self.assertFalse(PG.picco_oltre_budget({"picco_rss_byte": 970493952, "picco_gpu_byte": 1}, self.B))

    def test_margin_accepts_jitter_rejects_real_excess(self):
        with patch.dict(os.environ, {"FORGIA_PICCO_CONTRO": "margine"}):
            self.assertFalse(PG.picco_oltre_budget(self.JITTER, self.B))
            self.assertTrue(PG.picco_oltre_budget({"picco_rss_byte": int(970493952 * 1.26)}, self.B))
            self.assertTrue(PG.picco_oltre_budget({"picco_rss_byte": 1, "picco_gpu_byte": int(692060160 * 1.26)}, self.B))

    def test_invalid(self):
        with patch.dict(os.environ, {"FORGIA_PICCO_CONTRO": "x"}):
            with self.assertRaises(ValueError):
                PG.picco_oltre_budget(self.JITTER, self.B)


class NumeroMisurato(unittest.TestCase):
    """D3: worker count from the measured peak, not from the 3 GiB assumption."""
    import cardan_np_misurato as N
    G = 1 << 30

    def test_measured_peak_limits_workers(self):
        p = {"rss_anon_byte": int(8.7 * self.G), "vram_byte": 2 * self.G, "ginocchio": 3}
        self.assertEqual(self.N.numero(p, 22 * self.G, 60 * self.G), 1)      # (22-8)//8.7 = 1; v10 formula gave 3
        self.assertEqual(self.N.numero(p, 28 * self.G, 60 * self.G), 2)

    def test_knee_and_vram(self):
        p = {"rss_anon_byte": 1 * self.G, "vram_byte": 10 * self.G, "ginocchio": 2}
        self.assertEqual(self.N.numero(p, 100 * self.G, 100 * self.G), 2)
        self.assertEqual(self.N.numero(p, 100 * self.G, 15 * self.G), 1)
        self.assertEqual(self.N.numero(p, 100 * self.G, None), 2)

    def test_never_below_one_and_rejects_missing(self):
        self.assertEqual(self.N.numero({"rss_byte": 50 * self.G}, 9 * self.G, None), 1)
        with self.assertRaises(ValueError):
            self.N.numero({"rss_byte": 0}, 9 * self.G, None)


if __name__ == "__main__":
    unittest.main()

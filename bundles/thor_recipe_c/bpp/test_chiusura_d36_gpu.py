"""D36 (22/10): a dense commit is adopted only if it comes from the SAME inputs.

CPU fixtures only, no HIP context. Run with devices masked:
  HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= python -m unittest test_chiusura_d36_gpu
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

for _p in ("@@ROOT:models@@/build/wt-r022b/thor_v2/rd_ternary", "@@ROOT:models@@/build/wt-r022b/thor_v2"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cardan_ripresa_densi_gpu as R  # noqa: E402


def _fn_a():
    return 1


def _fn_b():
    return 2


class ProvenanceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.td = self.root / "tensori"
        self.td.mkdir()
        self.nome = "blk.31.attn_v.weight"
        self.bb = np.arange(12, dtype=np.uint8).reshape(2, 6)
        self.host = np.arange(100, 112, dtype=np.uint8).reshape(2, 6)
        self.rec = {"nome": self.nome, "piani": 1, "energia_correzione_H": 1.,
                    "residuo_dopo_piega_H": .2, "catturata": .8, "norma_rel_dW": .1}
        self.logs = []
        self.cm = types.SimpleNamespace(gp=types.SimpleNamespace(
            decode_rows=lambda a, tid, c: (np.zeros((2, 32), dtype=np.int8), np.ones((2, 1), dtype=np.float16))))
        self.reg = self.root / "registro.jsonl"
        self.hfile = self.root / "H.npy"
        np.save(self.hfile, np.eye(4))
        self.manif = {"blk.31.attn_v3.weight": {"hessiana": {"file": str(self.hfile)}}}
        self.delta = {self.nome: np.full((2, 32), 0.25)}
        self.recipe = R.ricetta(types.SimpleNamespace(type="cpu"), "torch.float64", (_fn_a,), (R.__file__,))

    def src(self, host=None):
        host = self.host if host is None else host
        entry = types.SimpleNamespace(out=2, in_=32, fmt=types.SimpleNamespace(type_id=1),
                                      tensors=[types.SimpleNamespace(name=self.nome, data=host)])
        return types.SimpleNamespace(entry=lambda key: entry)

    def prov(self, host=None, delta=None, recipe=None):
        return R.Provenienza(self.src(host), delta or self.delta, self.manif, recipe or self.recipe)

    def publish(self, prov):
        artefatti = dict([R.salva_piano(self.td, self.nome, self.bb)])
        with R.Registro(self.reg) as reg:
            R.commit(reg, self.rec, artefatti, prov.di(self.nome))

    def resume(self, prov):
        return R.riprendi(self.root, {self.nome}, self.src(), self.cm, self.logs.append, prov)

    def snapshot(self):
        return (self.reg.read_bytes(), sorted(p.name for p in self.td.iterdir()),
                (self.td / f"{self.nome}.npy").read_bytes())

    def test_same_inputs_adopted(self):
        self.publish(self.prov())
        self.assertIn("provenienza", json.loads(self.reg.read_text()))
        self.assertEqual(self.resume(self.prov()), {self.nome})

    def test_different_delta_refused_nothing_moved(self):
        # the antagonist's D36 case: another checkpoint, same name and shape
        self.publish(self.prov())
        before = self.snapshot()
        other = {self.nome: np.zeros((2, 32))}
        with self.assertRaisesRegex(RuntimeError, "D36.*delta_sha256"):
            self.resume(self.prov(delta=other))
        self.assertEqual(self.snapshot(), before)

    def test_different_host_bytes_refused(self):
        self.publish(self.prov())
        host2 = self.host.copy()
        host2[0, 0] ^= 1
        with self.assertRaisesRegex(RuntimeError, "ospite_sha256"):
            self.resume(self.prov(host=host2))

    def test_same_host_content_other_object_adopted(self):
        # v10 folds the dense tensors from BASE, then again from GE: different file, identical dense bytes
        self.publish(self.prov())
        self.assertEqual(self.resume(self.prov(host=self.host.copy())), {self.nome})

    def test_recipe_change_refused(self):
        self.publish(self.prov())
        r2 = R.ricetta(types.SimpleNamespace(type="cuda"), "torch.float32", (_fn_a,), (R.__file__,))
        with self.assertRaisesRegex(RuntimeError, "ricetta"):
            self.resume(self.prov(recipe=r2))
        r3 = R.ricetta(types.SimpleNamespace(type="cpu"), "torch.float64", (_fn_b,), (R.__file__,))
        with self.assertRaisesRegex(RuntimeError, "ricetta"):
            self.resume(self.prov(recipe=r3))

    def test_hessian_file_change_refused(self):
        self.publish(self.prov())
        np.save(self.hfile, 2 * np.eye(5))
        with self.assertRaisesRegex(RuntimeError, "hessiana"):
            self.resume(self.prov())

    def test_switch_off_is_previous_behaviour(self):
        with patch.dict(os.environ, {"FORGIA_DENSI_PROVENIENZA": "0"}):
            self.publish(self.prov())
            self.assertNotIn("provenienza", json.loads(self.reg.read_text()))
            other = {self.nome: np.zeros((2, 32))}
            self.assertEqual(self.resume(self.prov(delta=other)), {self.nome})

    def test_legacy_default_adopted_with_warning(self):
        with patch.dict(os.environ, {"FORGIA_DENSI_PROVENIENZA": "0"}):
            self.publish(self.prov())
        self.assertEqual(self.resume(self.prov()), {self.nome})
        self.assertTrue(any("without provenance" in x for x in self.logs))

    def test_legacy_strict_refused(self):
        with patch.dict(os.environ, {"FORGIA_DENSI_PROVENIENZA": "0"}):
            self.publish(self.prov())
        before = self.snapshot()
        with patch.dict(os.environ, {"FORGIA_DENSI_PROVENIENZA": "rigorosa"}):
            with self.assertRaisesRegex(RuntimeError, "legacy"):
                self.resume(self.prov())
        self.assertEqual(self.snapshot(), before)

    def test_invalid_switch(self):
        with patch.dict(os.environ, {"FORGIA_DENSI_PROVENIENZA": "x"}):
            with self.assertRaises(ValueError):
                self.resume(self.prov())

    def test_no_provenance_object_keeps_old_api(self):
        self.publish(self.prov())
        self.assertEqual(R.riprendi(self.root, {self.nome}, self.src(), self.cm, self.logs.append), {self.nome})


class DtypeSwitchTest(unittest.TestCase):
    """D15: FORGIA_DENSI_DTYPE overrides --dtype in both dense CLIs; unset keeps the argument."""

    def run_main(self, modname, env):
        import importlib
        mod = importlib.import_module(modname)
        seen = {}
        with tempfile.TemporaryDirectory() as out, patch.dict(os.environ, env, clear=False), \
                patch.object(mod, "esegui", lambda a, pi, pn: seen.setdefault("dtype", a.dtype) and 0):
            if "FORGIA_DENSI_DTYPE" not in env:
                os.environ.pop("FORGIA_DENSI_DTYPE", None)
            mod.main(["--delta", "x", "--gguf", "y", "--out", out, "--dtype", "float32", "--lotto", "8"])
        return seen["dtype"]

    def test_both_clis(self):
        for m in ("cardan_piega_densi_gpu", "cardan_piega_densi_gpu_veloce"):
            self.assertEqual(self.run_main(m, {}), "float32")
            self.assertEqual(self.run_main(m, {"FORGIA_DENSI_DTYPE": "float64"}), "float64")
            with self.assertRaises(SystemExit):
                self.run_main(m, {"FORGIA_DENSI_DTYPE": "bf16"})


if __name__ == "__main__":
    unittest.main()

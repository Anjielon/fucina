"""D22 e input persistenti: fixture stdlib, nessun import torch/HIP."""
import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import cardan_input_densi_gpu as I
import cardan_fucina_gpu as F
import pianificatore_gpu as PG

FORGIA = Path("@@ROOT:forge@@")
OC = Path("@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C")


class Tensore:
    def __init__(self, valori):
        self.valori = list(valori)

    def numel(self):
        return len(self.valori)

    def element_size(self):
        return 8

    def clone(self):
        return Tensore(self.valori)


class InputTest(unittest.TestCase):
    def prepara(self, root, limite=32):
        (root / "g").write_bytes(b"gguf fixture")
        for n in ("a", "b", "c"):
            (root / n).write_bytes(n.encode())
        nomi = [f"blk.17.attn_{s}.weight" for s in ("q", "k", "v", "o")]
        manif = {n.replace(".weight", "3.weight"): {"hessiana": {
            "file": str(root / p)}} for n, p in zip(nomi, ("a", "a", "b", "c"))}
        cd = types.SimpleNamespace(hessiana=Mock(side_effect=lambda *a: Tensore([1, 2])))
        src = object()
        cm = types.SimpleNamespace(
            gp=types.SimpleNamespace(GGUFPlaneSource=Mock(return_value=src)),
            Denso=Mock(side_effect=lambda src, n: n))
        config = {"gguf": str(root / "g"), "strati": [17], "cache_h_byte": limite}
        return I.InputDensi(config, dict.fromkeys(nomi), manif, cm, cd), nomi, cm, cd

    def test_un_mmap_per_worker_solo_strati_e_basi_selezionati(self):
        with tempfile.TemporaryDirectory() as td:
            inp, nomi, cm, _ = self.prepara(Path(td))
            self.assertIsNone(inp.sorgente)
            for n in nomi + nomi:
                self.assertEqual(inp.denso(n), n)
            cm.gp.GGUFPlaneSource.assert_called_once_with(
                str(Path(td) / "g"), layers=[17],
                bases=["attn_k", "attn_o", "attn_q", "attn_v"])
            self.assertEqual(inp.statistiche()["gguf_aperture"], 1)

    def test_h_condivisa_normalizzata_una_volta_copie_private(self):
        with tempfile.TemporaryDirectory() as td:
            inp, nomi, _, cd = self.prepara(Path(td))
            a = inp.hessiana(nomi[0])
            a.valori[0] = 999
            b = inp.hessiana(nomi[1])
            self.assertEqual(b.valori, [1, 2])
            cd.hessiana.assert_called_once()
            self.assertEqual(inp.statistiche()["h_hit"], 1)
            self.assertEqual(inp.byte_cache, 16)

    def test_lru_limitata_in_byte_non_in_numero_tensori(self):
        with tempfile.TemporaryDirectory() as td:
            inp, nomi, _, cd = self.prepara(Path(td))
            for i in (0, 2, 1, 3):
                inp.hessiana(nomi[i])
                self.assertLessEqual(inp.byte_cache, inp.limite)
            self.assertEqual([Path(p).name for p in inp.cache], ["a", "c"])
            inp.hessiana(nomi[2])
            self.assertEqual(cd.hessiana.call_count, 4)

    def test_zero_o_h_grande_non_entrano_in_cache(self):
        for limite in (0, 8):
            with self.subTest(limite=limite), tempfile.TemporaryDirectory() as td:
                inp, nomi, _, cd = self.prepara(Path(td), limite)
                for _ in range(2):
                    inp.hessiana(nomi[0])
                self.assertEqual(cd.hessiana.call_count, 2)
                self.assertEqual(inp.byte_cache, 0)

    def test_input_modificati_rifiutati_anche_dopo_hit_o_espulsione(self):
        for tipo in ("gguf", "h"):
            with self.subTest(tipo=tipo), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                inp, nomi, _, _ = self.prepara(root, 16)
                inp.denso(nomi[0])
                inp.hessiana(nomi[0])
                inp.hessiana(nomi[2])
                (root / ("g" if tipo == "gguf" else "a")).write_bytes(b"cambiato")
                with self.assertRaisesRegex(RuntimeError, "cambiat"):
                    (inp.denso if tipo == "gguf" else inp.hessiana)(nomi[0])

    def test_modifica_durante_lettura_non_popola_cache(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inp, nomi, _, cd = self.prepara(root)
            def cambiato(*_):
                (root / "a").write_bytes(b"mutato durante lettura")
                return Tensore([1, 2])
            cd.hessiana.side_effect = cambiato
            with self.assertRaisesRegex(RuntimeError, "durante la lettura"):
                inp.hessiana(nomi[0])
            self.assertFalse(inp.cache)

    def test_modifica_gguf_durante_decode_rifiutata(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inp, nomi, cm, _ = self.prepara(root)
            def cambiato(*_):
                (root / "g").write_bytes(b"mutato durante decode")
                return object()
            cm.Denso.side_effect = cambiato
            with self.assertRaisesRegex(RuntimeError, "durante la lettura"):
                inp.denso(nomi[0])

    def test_budget_include_cache_non_ancora_riempita(self):
        from cardan_profila_gpu import rss_con_cache
        self.assertEqual(rss_con_cache({"picco_rss_byte": 100}), 100)
        self.assertEqual(rss_con_cache({"picco_rss_byte": 100,
                                      "input_densi_cumulativi": None}), 100)
        for residente, atteso in ((0, 164), (32, 132), (64, 100)):
            self.assertEqual(rss_con_cache({"picco_rss_byte": 100, "input_densi_cumulativi": {
                "cache_h_limite_byte": 64, "cache_h_residente_byte": residente}}), atteso)
        for residente in (-1, 65, True, 1.5):
            with self.assertRaises(ValueError):
                rss_con_cache({"picco_rss_byte": 100, "input_densi_cumulativi": {
                    "cache_h_limite_byte": 64, "cache_h_residente_byte": residente}})

    def test_nome_estraneo_o_metrica_assente_rifiutati(self):
        with tempfile.TemporaryDirectory() as td:
            inp, nomi, _, _ = self.prepara(Path(td))
            for funzione in (inp.denso, inp.hessiana):
                with self.assertRaisesRegex(ValueError, "fuori piano"):
                    funzione("blk.18.attn_q.weight")
            inp.manif.clear()
            with self.assertRaisesRegex(KeyError, "no Hessian"):
                inp.hessiana(nomi[0])

    def test_limiti_invalidi_e_strati_vuoti_rifiutati(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inp, nomi, cm, cd = self.prepara(root)
            for limite in (-1, True, 1.5):
                with self.assertRaises(ValueError):
                    I.InputDensi({**inp.config, "cache_h_byte": limite},
                                 dict.fromkeys(nomi), inp.manif, cm, cd)
            with self.assertRaisesRegex(ValueError, "nessun delta"):
                I.InputDensi({**inp.config, "strati": [18]},
                             dict.fromkeys(nomi), inp.manif, cm, cd)

    def test_modulo_nuovo_in_impronta_ricetta(self):
        self.assertIn(Path(I.__file__), F.dipendenze_codice())

    def test_pipeline_rifiutano_ripresa_con_riuso_cambiato(self):
        import cardan_pipeline_gpu
        import cardan_pipeline_guardia_gpu
        for pipeline in (cardan_pipeline_gpu, cardan_pipeline_guardia_gpu):
            with self.subTest(pipeline=pipeline.__name__), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                for n in ("base", "ce", "profilo", "collaudo"):
                    (root / n).write_text(n)
                argv = ["--base", str(root / "base"), "--esperti", str(root / "ce"),
                        "--profilo-esperti", str(root / "profilo"),
                        "--collaudo-esperti", str(root / "collaudo"), "--out", str(root / "out")]
                with patch.dict(os.environ, FORGIA_GPU_SCALABILE="1", FORGIA_RIUSO_INPUT="0"), \
                        patch.object(PG, "guardia_corse"), patch.object(
                            PG, "misura_host", return_value=(24 * PG.GIB, 0)), \
                        patch("cardan_certificato_gpu.verifica"), patch.object(
                            pipeline.subprocess, "run", return_value=types.SimpleNamespace(returncode=0)):
                    pipeline.main(argv)
                    with patch.dict(os.environ, FORGIA_RIUSO_INPUT="1"):
                        with self.assertRaisesRegex(ValueError, "ricetta diversi"):
                            pipeline.main(argv)


class ShellTest(unittest.TestCase):
    def test_sintassi_cinque_shell(self):
        paths = [FORGIA / "stato" / n for n in (
            "cardan_congiunto_s7t550_1020_gpu.sh", "dopo_tappa_misura_1021_gpu.sh",
            "dopo_tappa_misura_guardia_gpu.sh")] + [
                OC / "S5_SU_S7T550_1020/comando_20261020_gpu.sh",
                OC / "ACCELERA_DOPO_S4_1015/s5/cardan_ospite_c_veloce_gpu.sh"]
        for p in paths:
            with self.subTest(path=str(p)):
                r = subprocess.run(["bash", "-n", str(p)], capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr)

    def test_congiunto_variabile_assente_errore_esplicito(self):
        r = subprocess.run(["bash", str(FORGIA / "stato/cardan_congiunto_s7t550_1020_gpu.sh")],
                           env={"PATH": os.defpath, "FORGIA_ALLENA_GPU": "1"},
                           capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("FGPU_OUT: cartella NUOVA per allenamento", r.stderr)

    def test_congiunto_raggiunge_exec_senza_avviare_trainer(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            nice = root / "nice"
            nice.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            nice.chmod(0o700)
            env = {"PATH": f"{root}:{os.defpath}", "FORGIA_ALLENA_GPU": "1",
                   "FGPU_OUT": str(root / "non_creata"), "FGPU_BASE": "base fixture",
                   "FGPU_A1": "a1 fixture", "FGPU_KLP": "0.123"}
            r = subprocess.run(["bash", str(FORGIA / "stato/cardan_congiunto_s7t550_1020_gpu.sh"),
                                "--device", "cuda:1"], env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            argv = r.stdout.splitlines()
            self.assertEqual(argv[:2], ["-n", "19"])
            self.assertEqual(Path(argv[3]).name, "cardan_congiunto_gpu.py")
            self.assertEqual(argv[argv.index("--gguf") + 1], "base fixture")
            self.assertEqual(argv[argv.index("--kl-partenza-motore") + 1], "0.123")
            self.assertEqual(argv[-2:], ["--device", "cuda:1"])
            self.assertFalse((root / "non_creata").exists())


if __name__ == "__main__":
    unittest.main()

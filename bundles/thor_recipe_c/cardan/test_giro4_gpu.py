"""Regressioni stdlib del collegamento classi/piani/pipeline, senza HIP."""
import contextlib
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import cardan_classi_gpu as C
import cardan_fucina_guardia_gpu as F
import cardan_obiettivo_gpu as O
import pianificatore_gpu as PG
import pianificatore_misurato_gpu as M
from test_obiettivo_gpu import campione, risorse_finzione


class ClassiTest(unittest.TestCase):
    def test_strati_equivalenti_stessa_classe_code_diverse(self):
        uu = [PG.Unita(f"densi-{i}", {
            "forma": [2048, 2048, 2, True],
            "nomi": [f"blk.{i}.attn_q.weight"]}) for i in (3, 17)]
        config = {"fase": "densi"}
        C.collega(config, uu)
        classi = config["pianificazione"]["classi"]
        self.assertEqual(classi["densi-3"], classi["densi-17"])

    def test_ultimo_lotto_forma_operatore_e_rotazione_distinti(self):
        base = {"forma": [2048, 2048, 2, True],
                "nomi": ["blk.3.attn_q.weight", "blk.4.attn_q.weight"]}
        varianti = [base, {**base, "nomi": base["nomi"][:1]},
                    {**base, "forma": [2048, 1024, 2, True]},
                    {**base, "forma": [2048, 2048, 2, False]},
                    {**base, "nomi": ["blk.3.attn_k.weight", "blk.4.attn_k.weight"]}]
        config = {"fase": "densi"}
        C.collega(config, [PG.Unita(str(i), d) for i, d in enumerate(varianti)])
        self.assertEqual(len(set(config["pianificazione"]["classi"].values())), 5)

    def test_vecchi_metadati_non_inventano_classi(self):
        config = {"fase": "esperti"}
        C.collega(config, [PG.Unita(k, {"strato": i}) for i, k in enumerate(("a", "b"))])
        self.assertEqual(config["pianificazione"]["classi"], {"a": "a", "b": "b"})

    def test_guardia_produce_classi_dopo_divisione_famiglie(self):
        uu = [PG.Unita(f"esperti-{i}", {
            "strato": i, "forme": [[512, 2048, 256, 2]] * 2, "gq": True,
            "file_attesi": [f"blk.{i}.ffn_{f}_exps.weight.npy" for f in ("gate", "up")]})
              for i in (3, 17)]
        piano = {"config": {"fase": "esperti"}, "firma": "old", "ricetta_sha256": "r"}
        with patch.object(F, "_prepara", return_value=(copy.deepcopy(piano), uu)):
            nuovo, unita = F.prepara(types.SimpleNamespace(fase="esperti"))
        cc = nuovo["config"]["pianificazione"]["classi"]
        self.assertEqual(set(cc), {u.chiave for u in unita})
        self.assertEqual(cc["esperti-3-gate"], cc["esperti-17-gate"])
        self.assertNotEqual(cc["esperti-3-gate"], cc["esperti-3-up"])
        self.assertEqual(nuovo["firma"], PG.firma_json(
            {k: v for k, v in nuovo.items() if k != "firma"}))

    def test_classe_gq_diversa_e_a1_shape_diversa(self):
        config = {"fase": "esperti"}
        C.collega(config, [PG.Unita(str(b), {"forme": [[512, 2048, 256, 2]],
                                           "gq": b, "famiglia": "gate"}) for b in (True, False)])
        self.assertEqual(len(set(config["pianificazione"]["classi"].values())), 2)
        config = {"fase": "a1"}
        C.collega(config, [PG.Unita(str(n), {"forma_h": [n, n], "tipo_strato": "esperti"})
                          for n in (1024, 2048)])
        self.assertEqual(len(set(config["pianificazione"]["classi"].values())), 2)

    def test_merge_conserva_classi_e_rifiuta_mappa_incompatibile(self):
        config = {"pianificazione": {"classi": {"a": "A", "b": "A"}}}
        originale = copy.deepcopy(config)
        self.assertEqual(C.unisci(config, {})["classi"], {"a": "A", "b": "A"})
        self.assertEqual(C.unisci(config, {"classi": {"a": "A"}})["classi"],
                         {"a": "A", "b": "A"})
        with self.assertRaisesRegex(ValueError, "diverse dal piano"):
            C.unisci(config, {"classi": {"a": "B"}})
        self.assertEqual(config, originale)

    def test_cli_non_perde_classi_senza_rendimenti(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            piano = {"firma": "f", "factory": "nessuno:crea", "unita": [
                {"chiave": "a", "dati": {}}], "config": {
                    "fase": "densi", "fili": 1,
                    "pianificazione": {"classi": {"a": "A"}}}}
            (root / "p.json").write_text(json.dumps(piano))
            torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
                device_count=lambda: 1, mem_get_info=lambda d: (16 * PG.GIB, 16 * PG.GIB)))
            with patch.dict(sys.modules, torch=torch), patch.object(PG, "guardia_corse"), \
                    patch.object(PG, "misura_host", return_value=(24 * PG.GIB, 0)), \
                    patch.object(PG, "limite_code_non_misurate", return_value=1), \
                    patch.object(PG, "lucchetto", return_value=contextlib.nullcontext()), \
                    patch.object(PG.signal, "signal"), patch.object(PG, "esegui") as esegui:
                PG.main(["--piano", str(root / "p.json"), "--out", str(root / "out"),
                         "--processi-per-gpu", "1", "--budget-gpu-gib", "1",
                         "--budget-rss-gib", "1", "--fili", "1"])
            config = esegui.call_args.args[2]
            self.assertEqual(config["pianificazione"]["classi"], {"a": "A"})
            self.assertFalse(config["pianificazione"]["calibra_compresenza"])

    def test_due_worker_reali_sintetici_con_classi_di_unita_non_campionate(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            uu = [PG.Unita(f"u{i}", {"secondi": .08}) for i in range(4)]
            campioni = [campione([(0, "X")], 0, "X", 1),
                        campione([(0, "X"), (0, "X")], 0, "X", 1)]
            config = {"pianificazione": {
                "classi": {u.chiave: "X" for u in uu}, "tempi": campioni,
                "avvio_s": {"0": .001}, "intervallo_risorse_s": .02}}
            with patch.object(PG, "guardia_corse"), patch.object(
                    PG, "misura_host", return_value=(24 * PG.GIB, 0)), patch.object(
                        M.Telemetria, "campiona", side_effect=risorse_finzione):
                M.esegui(uu, "test_guardia_fucina_gpu:crea_finto", config,
                         root, "f", {0: 2}, PG.Budget(1, PG.GIB, 1, 1))
            eventi = [json.loads(s) for s in (root / "decisioni.jsonl").read_text().splitlines()]
            prima = next(r for r in eventi if r["evento"] == "decisione")
            self.assertEqual(prima["scelta"]["posti"], {"0": 2})
            self.assertLess(prima["scelta"]["somma_pesata"],
                            prima["alternative"][0]["somma_pesata"])
            self.assertTrue(all(PG.completa(root / "unita" / u.chiave, "f", u) for u in uu))

    def test_contesto_ignoto_resta_seriale(self):
        lavori = [O.Lavoro(str(i), "X", str(i)) for i in range(4)]
        scelta = O.scegli(lavori, {0: 2}, O.Tempi(
            [campione([(0, "X")], 0, "X", 1)]))
        self.assertEqual(scelta["scelta"]["posti"], {0: 1})

    def test_argomenti_pipeline_solo_fase_richiesta(self):
        self.assertEqual(C.opzioni_rendimenti("esperti", "e.json", "d.json"),
                         ["--rendimenti", "e.json"])
        self.assertEqual(C.opzioni_rendimenti("densi", "e.json", "d.json"),
                         ["--rendimenti", "d.json"])
        self.assertEqual(C.opzioni_rendimenti("esperti"), [])

    def test_due_pipeline_inoltrano_rendimenti_e_vincolano_input(self):
        import cardan_pipeline_gpu
        import cardan_pipeline_guardia_gpu
        for pipeline in (cardan_pipeline_gpu, cardan_pipeline_guardia_gpu):
            with self.subTest(pipeline=pipeline.__name__), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                for n in ("base", "ce", "profilo", "collaudo", "rendimenti"):
                    (root / n).write_text(n)
                out = root / "out"
                argv = ["--base", str(root / "base"), "--esperti", str(root / "ce"),
                        "--profilo-esperti", str(root / "profilo"),
                        "--collaudo-esperti", str(root / "collaudo"),
                        "--rendimenti-esperti", str(root / "rendimenti"), "--out", str(out)]
                with patch.dict("os.environ", FORGIA_GPU_SCALABILE="1"), \
                        patch.object(PG, "guardia_corse"), patch.object(
                            PG, "misura_host", return_value=(24 * PG.GIB, 0)), \
                        patch("cardan_certificato_gpu.verifica"), patch.object(
                            pipeline.subprocess, "run", return_value=types.SimpleNamespace(
                                returncode=0)) as run:
                    self.assertEqual(pipeline.main(argv), 0)
                comandi = [c.args[0] for c in run.call_args_list]
                piega = next(c for c in comandi if "--fase" in c)
                i = piega.index("--rendimenti")
                self.assertEqual(piega[i + 1], str(root / "rendimenti"))
                meta = json.loads((out / "pipeline.json").read_text())
                self.assertIn(str(root / "rendimenti"),
                              {m["path"] for m in meta["input"]})


if __name__ == "__main__":
    unittest.main()

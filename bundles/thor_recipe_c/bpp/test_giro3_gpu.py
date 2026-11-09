"""Regressioni della terza guardia: nessun import torch o contesto HIP."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cardan_obiettivo_gpu as O
import pianificatore_misurato_gpu as M
from test_obiettivo_gpu import campione
from test_obiettivo_gpu import risorse_finzione
import pianificatore_gpu as PG
import cardan_rendimenti_gpu as R
import dataclasses
import time


def crea_misura(config, device):
    from test_guardia_fucina_gpu import crea_finto
    originale = crea_finto(config, device)

    def lavora(dati, out):
        return {**originale(dati, out), "device_tipo": "CPU-sintetica-NON-GPU"}
    return lavora


def crea_lento(config, device):
    Path(config["iniziato"]).write_text("caricamento sintetico")
    time.sleep(.4)
    return crea_misura(config, device)


class ModelloTest(unittest.TestCase):
    def test_d20_device_senza_posti_non_influenza_ordine(self):
        tempi = O.Tempi([campione([(d, c)], d, c, s)
                         for d, c, s in ((0, "A", 100), (1, "A", 1),
                                         (0, "B", 2), (1, "B", 2))])
        lavori = [O.Lavoro(c, c, c) for c in ("A", "B")]
        for posti in ({0: 1, 1: 0}, {0: 1}):
            r = O.simula(lavori, posti, tempi)
            self.assertEqual(r["somma_pesata"], 104)
            self.assertEqual(r["ondate"], [[("B", 0)], [("A", 0)]])
        r = O.scegli(lavori, {0: 1, 1: 0}, tempi)
        self.assertEqual(r["scelta"]["somma_pesata"], 104)

    def test_d21_snapshot_non_duplica_baseline_su_riprese(self):
        baseline = campione([(0, "A")], 0, "A", 100)
        nuovo = campione([(0, "A")], 0, "A", 10)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "storia.json"
            p.write_text(json.dumps({"firma": "f", "tempi": [baseline, nuovo]}))
            for _ in range(4):
                tempi, avvii = M.carica_rendimenti({"tempi": [baseline]}, p, "f")
                self.assertEqual(tempi.leggi(baseline["contesto"], 0, "A"), 55)
                self.assertEqual(len(tempi.esporta()), 2)
                p.write_text(json.dumps({"firma": "f", "tempi": tempi.esporta(),
                                         "avvio_s": avvii}))
            with self.assertRaisesRegex(ValueError, "altra ricetta"):
                M.carica_rendimenti({}, p, "estranea")

    def test_d19_calibra_anche_device_non_ancora_misurato(self):
        tempi = O.Tempi([campione([(0, "A")], 0, "A", 1)])
        lavori = [O.Lavoro("a", "A", "a"), O.Lavoro("b", "B", "b")]
        r = O.scegli(lavori, {0: 1, 1: 1}, tempi)
        self.assertTrue(r["scelta"]["calibrazione"])
        self.assertEqual(r["scelta"]["ondate"], [[("a", 1)]])


class PressioneTest(unittest.TestCase):
    def test_pressione_durante_caricamento_non_blocca_coordinatore(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            marker = root / "iniziato"
            picchi = 0

            def sensore():
                nonlocal picchi
                s = risorse_finzione()
                if marker.exists() and picchi < 3:
                    s["swap_in_byte_s"] = 30_000_000
                    picchi += 1
                return s

            config = {"iniziato": str(marker), "pianificazione": {
                "intervallo_risorse_s": .02, "attesa_massima_s": 3}}
            with patch.object(PG, "guardia_corse"), patch.object(
                    PG, "misura_host", return_value=(20 * PG.GIB, 0)), patch.object(
                        M.Telemetria, "campiona", side_effect=sensore):
                M.esegui([PG.Unita("A", {})], "test_giro3_gpu:crea_lento", config,
                         root, "f", {0: 1}, PG.Budget(1, PG.GIB, 1, 1))
            eventi = [json.loads(r) for r in (root / "decisioni.jsonl").read_text().splitlines()]
            annullati = [r for r in eventi if r["evento"] == "avvio_annullato"]
            self.assertEqual(len(annullati), 1)
            self.assertEqual(annullati[0]["unita_assegnate"], 0)
            self.assertEqual(sum(r["evento"] == "completamento" for r in eventi), 1)
            self.assertTrue((root / "unita/A/completa.json").is_file())

    def test_d18_picco_transitorio_attende_e_riprende_senza_rilancio(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            picchi = 0

            def sensore():
                nonlocal picchi
                s = risorse_finzione()
                if list(root.glob(".lavoro-A-*")) and picchi < 3:
                    s["swap_in_byte_s"] = 30_000_000
                    picchi += 1
                return s

            config = {"pianificazione": {"intervallo_risorse_s": .02,
                                        "attesa_massima_s": 3}}
            with patch.object(PG, "guardia_corse"), patch.object(
                    PG, "misura_host", return_value=(20 * PG.GIB, 0)), patch.object(
                        M.Telemetria, "campiona", side_effect=sensore):
                M.esegui([PG.Unita("A", {"secondi": .6}), PG.Unita("B", {})],
                         "test_guardia_fucina_gpu:crea_finto", config, root, "f",
                         {0: 1}, PG.Budget(1, PG.GIB, 1, 1))
            eventi = [json.loads(r) for r in (root / "decisioni.jsonl").read_text().splitlines()]
            self.assertEqual(picchi, 3)
            tipi = [e["evento"] for e in eventi]
            self.assertIn("ATTESA", tipi)
            self.assertIn("RIPRESA", tipi)
            self.assertNotIn("DRAIN", tipi)
            self.assertEqual(len(list(root.glob("lavoratore-*"))), 1)
            self.assertTrue((root / "unita/A/completa.json").is_file())
            self.assertTrue((root / "unita/B/completa.json").is_file())
            completamenti = [e for e in eventi if e["evento"] == "completamento"]
            self.assertFalse(completamenti[0]["campione_ammesso"])
            self.assertTrue(completamenti[1]["campione_ammesso"])


class RendimentiTest(unittest.TestCase):
    def test_misura_non_aggira_corse_protette(self):
        with tempfile.TemporaryDirectory() as td, patch.object(
                PG, "guardia_corse", side_effect=RuntimeError("corse protette presenti")):
            out = Path(td) / "mai"
            with self.assertRaisesRegex(RuntimeError, "corse protette"):
                R.main(["--piano", "inesistente", "--profilo", "inesistente",
                        "--unita", "A,B", "--out", str(out)])
            self.assertFalse(out.exists())

    def test_d16_misura_esporta_e_sceglie_due_worker(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            uu = [PG.Unita(c, {"secondi": .2}) for c in ("A", "B")]
            corse = []
            with patch.object(PG, "guardia_corse"), patch.object(
                    PG, "misura_host", return_value=(20 * PG.GIB, 0)), patch.object(
                        M.Telemetria, "campiona", side_effect=risorse_finzione):
                for nome, unita, posti in (("a", uu[:1], {0: 1}),
                                           ("b", uu[1:], {0: 1}),
                                           ("ab", uu, {0: 2})):
                    d = root / nome
                    d.mkdir()
                    config = {"pianificazione": {"intervallo_risorse_s": .02,
                                                 "calibra_compresenza": True}}
                    lancio = {"firma": nome, "ricetta_sha256": "sintetica",
                              "unita": [dataclasses.asdict(u) for u in unita], "config": config}
                    PG.scrivi_json(d / "lancio.json", lancio)
                    M.esegui(unita, "test_giro3_gpu:crea_misura", config, d,
                             nome, posti, PG.Budget(1, PG.GIB, 1, 1))
                    corse.append(d)
            r = R.esporta(corse, root / "rendimenti.json")
            self.assertEqual(len(r["tempi"]), 4)
            self.assertEqual(r["device_firme"], {"0": "CPU-sintetica-NON-GPU"})
            lavori = [O.Lavoro(u.chiave, u.chiave, u.chiave) for u in uu]
            scelta = O.scegli(lavori, {0: 2}, O.Tempi(r["tempi"]), residenti={0: 2})
            self.assertEqual(scelta["scelta"]["posti"], {0: 2})
            self.assertLess(scelta["scelta"]["somma_pesata"],
                            scelta["alternative"][0]["somma_pesata"])
            with self.assertRaises(FileExistsError):
                R.esporta(corse, root / "rendimenti.json")
            with self.assertRaisesRegex(ValueError, "ripetuta"):
                R.esporta([corse[0], corse[0]], root / "duplicati.json")


class CampionamentoTest(unittest.TestCase):
    def test_cpu_non_ricalcolata_su_10_millisecondi_fra_decisioni(self):
        from cardan_telemetria_gpu import Telemetria
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            proc = root / "proc"
            proc.mkdir()
            (proc / "stat").write_text("cpu 0 0 0 100 0 0 0 0\n")
            (proc / "vmstat").write_text("pswpin 0\n")
            (proc / "meminfo").write_text("MemAvailable: 20000000 kB\n")
            (proc / "loadavg").write_text("0 0 0 1/100 1\n")
            with patch("cardan_telemetria_gpu.time.monotonic",
                       side_effect=[10, 11, 11.01, 12.01]), \
                    patch("os.sched_getaffinity", return_value=set(range(8))):
                t = Telemetria(proc, root / "drm")
                (proc / "stat").write_text("cpu 10 0 10 180 0 0 0 0\n")
                self.assertAlmostEqual(t.campiona()["cpu_liberi_stima"], 6.4)
                (proc / "stat").write_text("cpu 20 0 20 180 0 0 0 0\n")
                self.assertAlmostEqual(t.campiona()["cpu_liberi_stima"], 6.4)
                (proc / "stat").write_text("cpu 60 0 60 280 0 0 0 0\n")
                self.assertAlmostEqual(t.campiona()["cpu_liberi_stima"], 4)


if __name__ == "__main__":
    unittest.main()

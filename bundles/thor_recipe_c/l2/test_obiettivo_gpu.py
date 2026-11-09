import os
"""Controesempi D9/D11/D13: stdlib, senza modelli o contesti device."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cardan_obiettivo_gpu as O
import cardan_certificato_gpu as C
import collauda_fucina_gpu as CF
import cardan_pipeline_guardia_gpu as Pipeline
import cardan_verifica_energie_gpu as Energie
import pianificatore_gpu as PG


def risorse_finzione():
    return {"ora": 0, "intervallo_s": 1, "mem_available": 20 * (1 << 30),
            "swap_in_byte_s": 0, "cpu_occupati_stima": 0, "cpu_liberi_stima": 32,
            "cpu_affinity": 32, "gpu": [], "corse_protette_pid": []}


def campione(assegnazioni, d, c, secondi):
    return {"contesto": O.contesto(assegnazioni), "device": d, "classe": c, "secondi": secondi}


class ObiettivoTest(unittest.TestCase):
    def modello(self, parallelo):
        return O.Tempi([campione([(0, c)], 0, c, 1) for c in ("A", "B")] +
                       [campione([(0, "A"), (0, "B")], 0, c, parallelo) for c in ("A", "B")])

    def test_d9_serie_12_contro_parallelo_44(self):
        lavori = [O.Lavoro("A", "A", "A", 10), O.Lavoro("B", "B", "B", 1)]
        r = O.scegli(lavori, {0: 2}, self.modello(4))
        self.assertEqual(r["scelta"]["posti"], {0: 1})
        self.assertEqual([a["somma_pesata"] for a in r["alternative"]], [12, 44])

    def test_parallelo_solo_se_obiettivo_diminuisce(self):
        lavori = [O.Lavoro("A", "A", "A", 10), O.Lavoro("B", "B", "B", 1)]
        r = O.scegli(lavori, {0: 2}, self.modello(1))
        # 12 -> 11 = -8,3 %: sotto il 10 % richiesto (FORGIA_GUADAGNO_MINIMO) si resta in serie
        self.assertEqual(r["scelta"]["posti"], {0: 1})
        self.assertEqual(r["scelta"]["somma_pesata"], 12)
        old = os.environ.get("FORGIA_GUADAGNO_MINIMO")
        os.environ["FORGIA_GUADAGNO_MINIMO"] = "0"
        try:
            r = O.scegli(lavori, {0: 2}, self.modello(1))
        finally:
            if old is None:
                os.environ.pop("FORGIA_GUADAGNO_MINIMO")
            else:
                os.environ["FORGIA_GUADAGNO_MINIMO"] = old
        self.assertEqual(r["scelta"]["posti"], {0: 2})
        self.assertEqual(r["scelta"]["somma_pesata"], 11)

    def test_wspt_21_non_fifo_120(self):
        t = O.Tempi([campione([(0, c)], 0, c, s) for c, s in (("B", 10), ("A", 1))])
        r = O.simula([O.Lavoro("B", "B", "B"), O.Lavoro("A", "A", "A", 10)], {0: 1}, t)
        self.assertEqual(r["somma_pesata"], 21)
        self.assertEqual(r["ondate"][0], [("A", 0)])

    def test_stesso_lavoro_pesa_solo_il_completamento_finale(self):
        t = O.Tempi([campione([(0, "X")], 0, "X", 1)])
        uu = [O.Lavoro(str(i), "X", "fase", 10) for i in range(3)]
        self.assertEqual(O.simula(uu, {0: 1}, t)["somma_pesata"], 30)

    def test_contesa_ignota_non_autorizza_secondo_worker(self):
        t = O.Tempi([campione([(0, "X")], 0, "X", 1)])
        r = O.scegli([O.Lavoro(str(i), "X", str(i)) for i in range(2)], {0: 2}, t)
        self.assertEqual(r["scelta"]["posti"], {0: 1})
        self.assertFalse(r["alternative"][-1]["ammessa"])
        self.assertIsNone(r["alternative"][-1]["somma_pesata"])

    def test_nessun_tempo_calibra_solo_una_unita_prioritaria(self):
        r = O.scegli([O.Lavoro("B", "B", "B"), O.Lavoro("A", "A", "A", 10)], {0: 8}, O.Tempi())
        self.assertTrue(r["scelta"]["calibrazione"])
        self.assertEqual(r["scelta"]["ondate"], [[("A", 0)]])

    def test_multi_gpu_non_presume_assenza_contesa_host(self):
        tt = [campione([(d, "X")], d, "X", 1) for d in (0, 1)]
        r = O.scegli([O.Lavoro(str(i), "X", str(i)) for i in range(2)], {0: 1, 1: 1}, O.Tempi(tt))
        self.assertEqual(sum(r["scelta"]["posti"].values()), 1)
        self.assertFalse(r["alternative"][-1]["ammessa"])

    def test_pesi_invalidi_e_incoerenti_rifiutati(self):
        for p in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                O.Lavoro("a", "x", "g", p)
        t = O.Tempi([campione([(0, "X")], 0, "X", 1)])
        with self.assertRaisesRegex(ValueError, "incoerenti"):
            O.simula([O.Lavoro("a", "X", "g", 1), O.Lavoro("b", "X", "g", 2)], {0: 1}, t)

    def test_contesto_canonico_e_tempi_finiti(self):
        self.assertEqual(O.contesto([(1, "B"), (0, "A"), (0, "A")]), '[[0,"A",2],[1,"B",1]]')
        with self.assertRaises(ValueError):
            O.Tempi([campione([(0, "A")], 0, "A", float("nan"))])

    def test_caricamenti_non_scompaiono_dall_obiettivo(self):
        lavori = [O.Lavoro("A", "A", "A", 10), O.Lavoro("B", "B", "B", 1)]
        # Un secondo worker costa 5s di caricamento: 11 di sola algebra
        # non basta per preferirlo al worker gia' residente (12).
        r = O.scegli(lavori, {0: 2}, self.modello(1), {"0": 5}, {0: 1})
        self.assertEqual(r["scelta"]["posti"], {0: 1})
        self.assertEqual(r["alternative"][-1]["somma_pesata"], 66)

    def test_tempi_compresenza_sono_completamenti_non_velocita(self):
        t = O.Tempi([campione([(0, c)], 0, c, s) for c, s in (("A", 1), ("B", 10))] +
                    [campione([(0, "A"), (0, "B")], 0, c, s) for c, s in (("A", 2), ("B", 12))])
        lavori = [O.Lavoro(c, c, c) for c in ("A", "B")]
        self.assertEqual(O.simula(lavori, {0: 2}, t)["somma_pesata"], 14)
        self.assertEqual(O.scegli(lavori, {0: 2}, t)["scelta"]["somma_pesata"], 12)

    def test_cpu_libere_limitano_anche_parallelismo_profittevole(self):
        lavori = [O.Lavoro("A", "A", "A", 10), O.Lavoro("B", "B", "B", 1)]
        r = O.scegli(lavori, {0: 2}, self.modello(1), max_lavoratori=1)
        self.assertEqual(r["scelta"]["posti"], {0: 1})
        self.assertEqual(len(r["alternative"]), 1)


class MetricheTest(unittest.TestCase):
    def audit_finto(self, root):
        roots = [root / s for s in ("rif", "cand")]
        rec = {"nome": "x", "energia_correzione_H": 1, "residuo_dopo_piega_H": .5, "catturata": .5}
        for d in roots:
            (d / "tensori").mkdir(parents=True)
            (d / "tensori/x.npy").write_bytes(b"byte sintetici, nessun numpy")
            (d / "registro.jsonl").write_text(json.dumps(rec))
        m = {"e0": 1, "e1": .5, "catturata": .5, "scarto_report": 0}
        rapporto = {"schema": 1, "toll_report": 0, "metriche": [
            {"chiave": "x", "riferimento": m, "candidato": m}],
                    "output": {n: Energie.impronta(d) for n, d in zip(("riferimento", "candidato"), roots)}}
        p = root / "audit.json"
        p.write_text(json.dumps(rapporto))
        return p, roots, rapporto

    def test_d13_cancellazione_media_non_nasconde_scarto_06(self):
        with tempfile.TemporaryDirectory() as td:
            roots = [Path(td) / s for s in ("rif", "cand")]
            for root, valori in zip(roots, ((.2, .8), (.8, .2))):
                root.mkdir()
                (root / "registro.jsonl").write_text("\n".join(json.dumps({
                    "nome": n, "energia_correzione_H": 1, "residuo_dopo_piega_H": 1 - c,
                    "catturata": c}) for n, c in zip(("L17", "L31"), valori)))
            r = CF.confronta_metriche(*roots)
            self.assertEqual(r["riferimento"]["catturata_media"], .5)
            self.assertEqual(r["candidato"]["catturata_media"], .5)
            self.assertAlmostEqual(r["scarto_massimo"], .6)

    def test_catturata_incoerente_con_energie_rifiutata(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "registro.jsonl").write_text(json.dumps({
                "nome": "x", "energia_correzione_H": 1, "residuo_dopo_piega_H": .9, "catturata": .9}))
            with self.assertRaisesRegex(ValueError, "incoerente"):
                CF.chiavi_metriche(root)

    def test_d11_pipeline_senza_certificato_rifiuta_prima_delle_guardie(self):
        with patch.dict("os.environ", {"FORGIA_GPU_SCALABILE": "1"}), \
                patch.object(Pipeline.PG, "guardia_corse") as guardia:
            with self.assertRaises(SystemExit) as e:
                Pipeline.main(["--base", "base", "--esperti", "ce", "--profilo-esperti", "p",
                               "--out", "mai-creata"])
            self.assertEqual(e.exception.code, 2)
            guardia.assert_not_called()

    def test_solo_memoria_e_vecchi_certificati_non_sono_numerici(self):
        with tempfile.TemporaryDirectory() as td:
            p, c = Path(td) / "p.json", Path(td) / "c.json"
            p.write_text(json.dumps({"ricetta_sha256": "r", "dispositivi": ["gfx"]}))
            c.write_text(json.dumps({"ok": True, "ricetta_sha256": "r", "dispositivi": ["gfx"]}))
            with self.assertRaisesRegex(ValueError, "obsoleto"):
                C.verifica(p, c)

    def test_audit_legato_ai_byte_e_copertura_esatta(self):
        with tempfile.TemporaryDirectory() as td:
            p, roots, r = self.audit_finto(Path(td))
            self.assertTrue(Energie.verifica_rapporto(p, *roots, 0)["ok"])
            r["metriche"].append(r["metriche"][0])
            p.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                Energie.verifica_rapporto(p, *roots, 0)
            r["metriche"].pop()
            p.write_text(json.dumps(r))
            (roots[1] / "tensori/x.npy").write_bytes(b"corrotto")
            with self.assertRaisesRegex(ValueError, "cambiati"):
                Energie.verifica_rapporto(p, *roots, 0)

    def test_audit_non_riproduce_report_rifiutato(self):
        with tempfile.TemporaryDirectory() as td:
            p, roots, r = self.audit_finto(Path(td))
            r["metriche"][0]["candidato"]["scarto_report"] = .1
            p.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError, "non riprodotte"):
                Energie.verifica_rapporto(p, *roots, 0)

    def test_bool_indipendente_non_basta_per_certificare(self):
        with tempfile.TemporaryDirectory() as td:
            p, c = Path(td) / "p", Path(td) / "c"
            p.write_text(json.dumps({"ricetta_sha256": "r", "dispositivi": ["gfx"]}))
            c.write_text(json.dumps({"schema": 2, "fase": "piega", "ok": True,
                                    "ricetta_sha256": "r", "dispositivi": ["gfx"], "toll_metrica": 0,
                                    "metriche": {"appaiate": [{"scarto_catturata": 0}]},
                                    "verifica_indipendente": {"ok": True}}))
            with self.assertRaisesRegex(ValueError, "solo ok"):
                C.verifica(p, c)


class TelemetriaTest(unittest.TestCase):
    def test_progresso_comprende_attese_non_solo_kernel(self):
        from cardan_telemetria_gpu import progresso_log
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "registro.log"
            p.write_text('[2026-10-21T08:33:03] {"passo":102,"s":68.4}\n'
                         '[2026-10-21T08:48:00] {"passo":103,"s":896.7}\n')
            r = progresso_log(p)
            self.assertEqual(r["passo"], 103)
            self.assertAlmostEqual(r["passi_s_wall"], 1 / 897)
            self.assertEqual(r["secondi_unita_dichiarati"], 896.7)

    def test_delta_swap_cpu_gpu_e_fili(self):
        from cardan_telemetria_gpu import Telemetria
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            proc, drm = root / "proc", root / "drm"
            proc.mkdir()
            gpu = drm / "card0/device"
            gpu.mkdir(parents=True)
            (proc / "stat").write_text("cpu 10 0 10 80 0 0 0 0 0 0\n")
            (proc / "vmstat").write_text("pswpin 10\n")
            (proc / "meminfo").write_text("MemAvailable: 10000000 kB\n")
            (proc / "loadavg").write_text("1 1 1 7/100 99\n")
            (gpu / "gpu_busy_percent").write_text("73")
            with patch("cardan_telemetria_gpu.time.monotonic", side_effect=[10, 12]), \
                    patch("os.sched_getaffinity", return_value=set(range(8))):
                t = Telemetria(proc, drm)
                (proc / "stat").write_text("cpu 20 0 20 160 0 0 0 0 0 0\n")
                (proc / "vmstat").write_text("pswpin 12\n")
                r = t.campiona()
            self.assertEqual(r["swap_in_byte_s"], __import__("os").sysconf("SC_PAGE_SIZE"))
            self.assertAlmostEqual(r["cpu_liberi_stima"], 6.4)
            self.assertEqual(r["fili_runnable_host"], 6)
            self.assertEqual(r["gpu"][0]["gpu_busy_percent"], 73)
            self.assertIsNone(r["gpu"][0]["mem_info_vram_used"])
            self.assertIn("mem_info_vram_used_errore", r["gpu"][0])


if __name__ == "__main__":
    unittest.main()

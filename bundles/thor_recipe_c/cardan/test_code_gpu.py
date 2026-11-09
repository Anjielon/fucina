"""Prove della priorita' 0 (ammissione per code GPU, regola di contesa 10 %, compile threads) senza GPU.

Albero KFD finto in una cartella temporanea (stessa struttura di /sys/class/kfd/kfd), nessun processo vero toccato.
"""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cardan_code_gpu as CQ
import cardan_obiettivo_gpu as O
import pianificatore_gpu as PG


def kfd_finto(root, code_per_pid, cp=8, sdma_motori=1, sdma_code=6, gpu_id=51834):
    k = Path(root) / "kfd"
    n0 = k / "topology" / "nodes" / "0"; n0.mkdir(parents=True)
    (n0 / "properties").write_text("simd_count 0\nnum_cp_queues 0\n"); (n0 / "gpu_id").write_text("0\n")
    n1 = k / "topology" / "nodes" / "1"; n1.mkdir()
    (n1 / "properties").write_text(f"simd_count 80\nnum_sdma_engines {sdma_motori}\nnum_sdma_xgmi_engines 0\n"
                                   f"num_sdma_queues_per_engine {sdma_code}\nnum_cp_queues {cp}\n")
    (n1 / "gpu_id").write_text(f"{gpu_id}\n")
    for pid, tipi in code_per_pid.items():
        for i, t in enumerate(tipi):
            q = k / "proc" / str(pid) / "queues" / str(i); q.mkdir(parents=True)
            (q / "type").write_text(f"{t}\n"); (q / "gpuid").write_text(f"{gpu_id}\n")
    return k


def campione(ctx, d, c, s):
    return {"contesto": O.contesto(ctx), "device": d, "classe": c, "secondi": s}


class CodeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        for v in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
            os.environ.pop(v, None)

    def test_uso_per_gpu_e_ammissibili(self):
        # congiunto (2 compute + 1 SDMA, misurato) + 5 pieghe con HSA_ENABLE_SDMA=0 (2 compute)
        k = kfd_finto(self.tmp.name, {100: [0, 0, 1], **{200 + i: [0, 0] for i in range(5)}})
        tot, per = CQ.uso(k)
        self.assertEqual(tot[51834], {"compute": 12, "sdma": 1})
        with mock.patch.object(CQ, "CONTESA", Path(self.tmp.name) / "nessuno.json"):
            self.assertEqual(CQ.ammissibili(0, 2, 0, kfd=k), 0)      # calcolo gia' oltre 8: nessuno
            self.assertEqual(CQ.ammissibili(0, 0, 1, kfd=k), 4)      # 6 - 1 riserva - 1 in uso

    def test_sdma_mai_oltre_capacita(self):
        # 33 worker inductor con una coda SDMA ciascuno (incidente 21/10): nessun nuovo lavoratore con SDMA
        k = kfd_finto(self.tmp.name, {300 + i: [1] for i in range(6)})
        with mock.patch.object(CQ, "CONTESA", Path(self.tmp.name) / "nessuno.json"):
            self.assertEqual(CQ.ammissibili(0, 2, 1, kfd=k), 0)
            self.assertEqual(CQ.ammissibili(0, 2, 0, kfd=k), 0)      # al bordo SDMA: nemmeno senza SDMA

    def test_sovrascrittura_calcolo_solo_col_fattore_salvato(self):
        k = kfd_finto(self.tmp.name, {100: [0, 0, 1], 101: [0, 0], 102: [0, 0], 103: [0, 0]})
        f = Path(self.tmp.name) / "contesa.json"
        with mock.patch.object(CQ, "CONTESA", f):
            self.assertEqual(CQ.ammissibili(0, 2, 0, kfd=k), 0)
            f.write_text(json.dumps({"compute_oversubscription": 1.5, "classi": {}}))
            self.assertEqual(CQ.ammissibili(0, 2, 0, kfd=k), 2)      # 12 - 8 in uso = 4 -> 2 lavoratori

    def test_regola_10_per_cento(self):
        self.assertEqual(CQ.lavoratori_ammessi({1: 10.0, 2: 10.9, 3: 15}), 1)    # +9 %: no
        self.assertEqual(CQ.lavoratori_ammessi({1: 10.0, 2: 11.0, 3: 12.0}), 2)  # +10 %, poi +9 %
        self.assertEqual(CQ.lavoratori_ammessi({"1": 10, "2": 13, "3": 16, "5": 40}), 3)
        self.assertEqual(CQ.lavoratori_ammessi({2: 20}), 1)

    def test_file_contesa_letto_dalla_catena(self):
        f = Path(self.tmp.name) / "c.json"
        self.assertEqual(CQ.lavoratori("piega_esperti", 5, f), 1)            # mai misurata: 1
        CQ.salva_contesa("piega_esperti", 1, 1.0, "prova", f)
        CQ.salva_contesa("piega_esperti", 2, 1.8, "prova", f)
        self.assertEqual(CQ.salva_contesa("piega_esperti", 3, 1.9, "prova", f), 2)
        self.assertEqual(CQ.lavoratori("piega_esperti", 5, f), 2)
        self.assertEqual(CQ.lavoratori("piega_esperti", 1, f), 1)
        self.assertFalse(list(Path(self.tmp.name).glob(".*tmp")))

    def test_scegli_rifiuta_guadagno_sotto_10(self):
        lavori = [O.Lavoro("A", "A", "A", 10), O.Lavoro("B", "B", "B", 1)]
        def modello(par):
            return O.Tempi([campione([(0, c)], 0, c, 1) for c in ("A", "B")] +
                           [campione([(0, "A"), (0, "B")], 0, c, par) for c in ("A", "B")])
        r = O.scegli(lavori, {0: 2}, modello(1), guadagno_minimo=0.10)     # serie 12, parallelo 11: +9 %
        self.assertEqual(r["scelta"]["posti"], {0: 1})
        self.assertAlmostEqual(r["alternative"][1]["soglia_somma_pesata"], 12 / 1.1)
        r = O.scegli(lavori, {0: 2}, modello(1), guadagno_minimo=0.0)      # comportamento precedente
        self.assertEqual(r["scelta"]["posti"], {0: 2})
        t = O.Tempi([campione([(0, c)], 0, c, 10) for c in ("A", "B")] +
                    [campione([(0, "A"), (0, "B")], 0, c, 10) for c in ("A", "B")])
        r = O.scegli(lavori, {0: 2}, t, guadagno_minimo=0.10)              # serie 10*10+1*20 = 120, parallelo 110: +9 %
        self.assertEqual(r["scelta"]["posti"], {0: 1})

    def test_pianifica_terzo_limite(self):
        b = PG.Budget(2 * PG.GIB, 1 * PG.GIB, 2, PG.GIB, margine=1)
        r = PG.Risorse(28 * PG.GIB, 0, 32, {0: 60 * PG.GIB})
        self.assertGreater(PG.pianifica(r, b, uma=False)[0], 3)
        with mock.patch.object(CQ, "ammissibili", return_value=3):
            self.assertEqual(PG.pianifica(r, b, uma=False, code={"compute": 2, "sdma": 0}), {0: 3})
        with mock.patch.object(CQ, "ammissibili", return_value=0):
            with self.assertRaises(RuntimeError):
                PG.pianifica(r, b, uma=False, code={"compute": 2, "sdma": 0})
        with mock.patch.object(PG, "limite_code_non_misurate", return_value=1):
            self.assertEqual(PG.pianifica(r, b, uma=False, code="non_misurate"), {0: 1})
        with mock.patch.object(PG, "limite_code_non_misurate", return_value=0):
            with self.assertRaises(RuntimeError):
                PG.pianifica(r, b, uma=False, code="non_misurate")

    def test_ambiente_compile_threads(self):
        env = {}
        CQ.ambiente_torch(env)
        self.assertEqual(env["TORCHINDUCTOR_COMPILE_THREADS"], "1")

    def test_visible_devices_rifiutato(self):
        k = kfd_finto(self.tmp.name, {})
        with mock.patch.dict(os.environ, {"HIP_VISIBLE_DEVICES": "0"}):
            with self.assertRaises(RuntimeError):
                CQ.ammissibili(0, 2, 0, kfd=k)


if __name__ == "__main__":
    unittest.main()

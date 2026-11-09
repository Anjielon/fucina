#!/usr/bin/env python3
"""Prove CPU minuscole, solo stdlib: nessun torch, modello o device."""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import pianificatore_gpu as PG
from test_obiettivo_gpu import campione, risorse_finzione


def crea_finto(config, device):
    stato = {"n": 0}

    def lavora(dati, out):
        stato["n"] += 1
        (out / "dato.bin").write_bytes(bytes(dati["byte"]))
        if dati.get("dormi"):
            time.sleep(dati["dormi"])
        if dati.get("muori"):
            os._exit(9)
        return {"picco_gpu_byte": 1, "visite": stato["n"], "device": device}
    return lavora


def coordinatore_di_prova(root, ingresso, uscita):
    ctx = PG.mp.get_context("spawn")
    p = ctx.Process(target=PG._lavoratore,
                    args=("test_pianificatore_gpu:crea_finto", {}, 0, root, "firma",
                          ingresso, uscita, 1, os.getpid()))
    p.start()
    stato, pid, _ = uscita.get(timeout=5)
    if stato != "pronto":
        raise RuntimeError(stato)
    Path(root, "figlio.pid").write_text(str(pid))
    time.sleep(30)


class RisorseTest(unittest.TestCase):
    def test_budget_globale_host_e_cpu(self):
        r = PG.Risorse(30 * PG.GIB, 0, 8, {0: 6 * PG.GIB, 1: 6 * PG.GIB})
        b = PG.Budget(2 * PG.GIB, PG.GIB, 2, 1, margine=1)
        self.assertEqual(PG.pianifica(r, b), {0: 2, 1: 2})
        self.assertEqual(PG.pianifica(dataclasses.replace(r, disponibile=11 * PG.GIB), b), {0: 1, 1: 0})
        self.assertEqual(PG.pianifica(r, b, "1"), {0: 1, 1: 1})

    def test_limite_device_e_distribuzione(self):
        r = PG.Risorse(30 * PG.GIB, 0, 32, {0: PG.GIB, 1: 5 * PG.GIB})
        b = PG.Budget(PG.GIB, PG.GIB, 1, 1, margine=1)
        self.assertEqual(PG.pianifica(r, b), {0: 0, 1: 4})

    def test_uma_non_somma_due_banche(self):
        r = PG.Risorse(12 * PG.GIB, 0, 32, {0: 20 * PG.GIB})
        b = PG.Budget(2 * PG.GIB, PG.GIB, 1, 1, margine=1)
        self.assertEqual(PG.pianifica(r, b, uma=True)[0], 1)
        self.assertEqual(PG.pianifica(r, b, uma=False)[0], 4)

    def test_soglie_esatte(self):
        PG.guardia_host(8 * PG.GIB, 20_000_000)
        with self.assertRaises(RuntimeError):
            PG.guardia_host(8 * PG.GIB - 1, 0)
        with self.assertRaises(RuntimeError):
            PG.guardia_host(30 * PG.GIB, 20_000_001)

    def test_parametri_invalidi(self):
        for x in ("0", "-1", "due", "1.5"):
            with self.assertRaises(ValueError):
                PG.pianifica(PG.Risorse(30 * PG.GIB, 0, 8, {0: 8 * PG.GIB}),
                             PG.Budget(1, 1, 1, 1), x)
        for scelta in ("", "0,0", "-1", "2"):
            with self.assertRaises(ValueError):
                PG.seleziona_gpu(scelta, 2)
        self.assertEqual(PG.seleziona_gpu("all", 3), [0, 1, 2])
        self.assertEqual(PG.seleziona_gpu("1,0", 2), [1, 0])
        with self.assertRaises(ValueError):
            PG.Budget(1, 1, 1, 1, float("nan"))
        with self.assertRaises(ValueError):
            PG.Unita("../fuga", {})

    def test_proc_non_scambia_prompt_per_corsa(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for pid, argv in (("11111111", ["copilot", "python /CARDAN_CONGIUNTO_S7T550_1020/x"]),
                              ("11111112", ["python3", "cardan_congiunto.py", "--uscita",
                                            "/x/CARDAN_CONGIUNTO_S7T550_1020/allena"]),
                              ("11111113", ["python3", "cardan_pipeline_gpu.py", "--esperti",
                                            "/x/CARDAN_CONGIUNTO_S7T550_1020/allena/tappe/x.pt"])):
                (root / pid).mkdir()
                (root / pid / "cmdline").write_bytes(b"\0".join(x.encode() for x in argv))
            self.assertEqual(PG.corse_protette(root), [11111112])

    def test_fili_io_contati(self):
        r = PG.Risorse(30 * PG.GIB, 0, 8, {0: 20 * PG.GIB})
        b = PG.Budget(PG.GIB, PG.GIB, 2, 1, margine=1, fili_io=2)
        self.assertEqual(PG.pianifica(r, b, uma=False), {0: 2})


class PubblicazioneTest(unittest.TestCase):
    def test_pubblicazione_ripresa_corruzione(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            stage, dest = root / ".stage", root / "unita"
            stage.mkdir()
            (stage / "x").write_bytes(b"completo")
            u = PG.Unita("L00", {"strato": 0})
            self.assertFalse(PG.completa(dest, "firma", u))
            PG.pubblica_unita(stage, dest, "firma", u, {})
            self.assertTrue(PG.completa(dest, "firma", u))
            self.assertFalse(stage.exists())
            with self.assertRaises(RuntimeError):
                PG.completa(dest, "altra", u)
            (dest / "x").write_bytes(b"corrotto")
            with self.assertRaises(RuntimeError):
                PG.completa(dest, "firma", u)
            self.assertTrue((dest / "x").exists())

    def test_marker_non_basta(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "completa.json").write_text(json.dumps(
                {"firma": "x", "unita": {"chiave": "x", "dati": {}},
                 "file": {"manca": {"byte": 1, "sha256": "x"}}}))
            with self.assertRaises(RuntimeError):
                PG.completa(root, "x", PG.Unita("x", {}))

    def test_identita_cambia_con_file(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "input"
            f.write_bytes(b"a")
            prima = PG.identita_file(f)
            f.write_bytes(b"bb")
            self.assertNotEqual(prima, PG.identita_file(f))

    def test_lock_non_sovrappone_coordinatori(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "lock"
            with PG.lucchetto(f):
                with self.assertRaises(RuntimeError):
                    with PG.lucchetto(f):
                        self.fail("lock entrato due volte")


class ProcessiTest(unittest.TestCase):
    def setUp(self):
        self.telemetria = patch("pianificatore_misurato_gpu.Telemetria.campiona",
                               side_effect=risorse_finzione)
        self.telemetria.start()
        self.addCleanup(self.telemetria.stop)

    def esegui_finto(self, root, unita):
        # Solo l'esecutore sintetico bypassa la presenza di allenamenti reali.
        # I figli non importano torch e scrivono al massimo pochi byte.
        with patch.object(PG, "guardia_corse"), patch.object(
                PG, "misura_host", return_value=(20 * PG.GIB, 0)):
            PG.esegui(unita, "test_pianificatore_gpu:crea_finto", {}, root, "firma",
                      {0: 1}, PG.Budget(1, PG.GIB, 1, 1))

    def test_worker_persistente_e_ripresa(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            unita = [PG.Unita(f"L{i}", {"byte": [i]}) for i in range(3)]
            self.esegui_finto(root, unita)
            prima = PG.identita_file(root / "unita/L0/dato.bin")
            visite = [json.loads((root / f"unita/L{i}/completa.json").read_text())
                      ["misure"]["visite"] for i in range(3)]
            self.assertEqual(visite, [1, 2, 3])
            self.esegui_finto(root, unita)
            self.assertEqual(prima, PG.identita_file(root / "unita/L0/dato.bin"))
            self.assertEqual(len(list(root.glob("lavoratore-*.log"))), 1)

    def test_figlio_ucciso_non_pubblica(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaisesRegex(RuntimeError, "pipe chiusa|morto"):
                self.esegui_finto(root, [PG.Unita("morto", {"byte": [1], "muori": True})])
            self.assertFalse((root / "unita/morto").exists())
            self.assertEqual(len(list(root.glob(".lavoro-morto-*"))), 1)
            self.assertTrue((root / "piano.json").is_file())

    def test_duplicati_rifiutati(self):
        with tempfile.TemporaryDirectory() as td:
            u = PG.Unita("x", {})
            with self.assertRaisesRegex(ValueError, "duplicate"):
                self.esegui_finto(Path(td), [u, u])

    def test_due_gruppi_device(self):
        with tempfile.TemporaryDirectory() as td, patch.object(PG, "guardia_corse"), patch.object(
                PG, "misura_host", return_value=(20 * PG.GIB, 0)):
            root = Path(td)
            unita = [PG.Unita(f"L{i}", {"byte": [i]}) for i in range(4)]
            tempi = [campione([(d, "X")], d, "X", 1) for d in (0, 1)]
            tempi += [campione([(0, "X"), (1, "X")], d, "X", .5) for d in (0, 1)]
            config = {"pianificazione": {"classi": {u.chiave: "X" for u in unita},
                                         "tempi": tempi, "avvio_s": {"0": 0, "1": 0}}}
            PG.esegui(unita, "test_pianificatore_gpu:crea_finto", config, root, "firma",
                      {0: 1, 1: 1}, PG.Budget(1, PG.GIB, 1, 1))
            dispositivi = {json.loads((root / f"unita/L{i}/completa.json").read_text())["misure"]["device"]
                           for i in range(4)}
            self.assertEqual(dispositivi, {0, 1})

    def test_guardia_drena_non_uccide_unita_in_volo(self):
        def guardia():
            if list(root.glob(".lavoro-L0-*")) or (root / "unita/L0").exists():
                raise RuntimeError("swap sintetico")

        with tempfile.TemporaryDirectory() as td, patch.object(PG, "guardia_corse", side_effect=guardia), \
                patch.object(PG, "misura_host", return_value=(20 * PG.GIB, 0)):
            root = Path(td)
            unita = [PG.Unita("L0", {"byte": [1], "dormi": 1.1}),
                     PG.Unita("L1", {"byte": [2]})]
            with self.assertRaisesRegex(RuntimeError, "swap sintetico"):
                config = {"pianificazione": {"intervallo_risorse_s": .05,
                                             "attesa_massima_s": .4}}
                PG.esegui(unita, "test_pianificatore_gpu:crea_finto", config, root, "firma",
                          {0: 2}, PG.Budget(1, PG.GIB, 1, 1))
            self.assertTrue(PG.completa(root / "unita/L0", "firma", unita[0]))
            self.assertFalse((root / "unita/L1").exists())

    def test_morte_coordinatore_non_lascia_worker_residente(self):
        ctx = PG.mp.get_context("spawn")
        ingresso, uscita = ctx.Queue(), ctx.Queue()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p = ctx.Process(target=coordinatore_di_prova, args=(td, ingresso, uscita))
            p.start()
            try:
                limite = time.monotonic() + 5
                while not (root / "figlio.pid").exists() and time.monotonic() < limite:
                    time.sleep(0.02)
                self.assertTrue((root / "figlio.pid").exists())
                pid = int((root / "figlio.pid").read_text())
                p.kill()  # Solo il coordinatore sintetico creato da questa prova.
                p.join(timeout=5)
                limite = time.monotonic() + 5
                vivo = True
                while vivo and time.monotonic() < limite:
                    try:
                        stato = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[0]
                        vivo = stato != "Z"
                    except FileNotFoundError:
                        vivo = False
                    if vivo:
                        time.sleep(0.02)
                self.assertFalse(vivo, f"worker sintetico {pid} ancora residente")
            finally:
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=5)
        ingresso.close()
        uscita.close()


class RaccoltaTest(unittest.TestCase):
    def test_a1_unisce_metadata_senza_riscritture_worker(self):
        from cardan_fucina_gpu import raccogli
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "unita").mkdir()
            unita = [PG.Unita(f"a1-{i}", {"strato": i}) for i in range(2)]
            for i, u in enumerate(unita):
                stage = root / f".stage-{i}"
                stage.mkdir()
                (stage / f"strato_{i:02d}.npz").write_bytes(b"npz-sintetico")
                (stage / "info.json").write_text(json.dumps({"tensori": {f"blk.{i}.x": {"type": 1}}}))
                (stage / "misure.json").write_text(json.dumps({str(i): {"secondi": 1}}))
                PG.pubblica_unita(stage, root / "unita" / u.chiave, "firma", u, {})
            piano = {"firma": "firma", "config": {"fase": "a1"}}
            out = raccogli(root, piano, unita)
            self.assertEqual(set(json.loads((out / "info.json").read_text())["tensori"]),
                             {"blk.0.x", "blk.1.x"})
            self.assertEqual(set(json.loads((out / "misure.json").read_text())), {"0", "1"})
            self.assertEqual(raccogli(root, piano, unita), out)

    def test_piega_unisce_registri_e_file(self):
        from cardan_fucina_gpu import raccogli
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "unita").mkdir()
            unita = [PG.Unita("densi-0", {"nomi": ["blk.0.attn_q.weight"],
                                         "file_attesi": ["blk.0.attn_q.weight.npy"]})]
            stage = root / ".stage"
            (stage / "tensori").mkdir(parents=True)
            (stage / "tensori/blk.0.attn_q.weight.npy").write_bytes(b"npy-sintetico")
            (stage / "registro.jsonl").write_text('{"nome": "blk.0.attn_q.weight"}\n')
            PG.pubblica_unita(stage, root / "unita" / unita[0].chiave, "firma", unita[0], {})
            out = raccogli(root, {"firma": "firma", "config": {"fase": "densi"}}, unita)
            self.assertEqual((out / "tensori/blk.0.attn_q.weight.npy").read_bytes(), b"npy-sintetico")

    def test_piega_registro_duplicato_rifiutato(self):
        from cardan_fucina_gpu import raccogli
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "unita").mkdir()
            u = PG.Unita("densi-0", {"nomi": ["blk.0.attn_q.weight"], "file_attesi": []})
            stage = root / ".stage"
            (stage / "tensori").mkdir(parents=True)
            (stage / "registro.jsonl").write_text('{"nome":"blk.0.attn_q.weight"}\n' * 2)
            PG.pubblica_unita(stage, root / "unita" / u.chiave, "firma", u, {})
            with self.assertRaisesRegex(RuntimeError, "duplicato"):
                raccogli(root, {"firma": "firma", "config": {"fase": "densi"}}, [u])
            self.assertFalse((root / "pubblicato").exists())


class PatchTest(unittest.TestCase):
    def test_verifica_anche_header_padding_e_coda(self):
        from cardan_patch_gpu import verifica_fuori
        with tempfile.TemporaryDirectory() as td:
            a, b = Path(td) / "base", Path(td) / "nuovo"
            a.write_bytes(b"HEADER__vecchio__CODA")
            b.write_bytes(b"HEADER__nuovooo__CODA")
            self.assertEqual(verifica_fuori(a, b, [(8, 15)], pezzo=3), len(a.read_bytes()) - 7)
            b.write_bytes(b"hEADER__nuovooo__CODA")
            with self.assertRaisesRegex(ValueError, "fuori intervento"):
                verifica_fuori(a, b, [(8, 15)], pezzo=3)

    def test_intervalli_sovrapposti_rifiutati(self):
        from cardan_patch_gpu import verifica_fuori
        with tempfile.TemporaryDirectory() as td:
            a = Path(td) / "base"
            a.write_bytes(b"0123456789")
            with self.assertRaisesRegex(ValueError, "sovrapposti"):
                verifica_fuori(a, a, [(2, 5), (4, 7)])


if __name__ == "__main__":
    unittest.main()

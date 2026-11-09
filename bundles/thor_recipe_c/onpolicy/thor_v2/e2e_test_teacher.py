#!/usr/bin/env python3
"""Collaudo di e2e_teacher.py, SOLO CPU. Scritto il 26/9/2026.

Il percorso Vulkan si prova sul teacher vero (f0a-cache, f0a-ppl); qui si prova tutto il resto con il
modello qwen35moe minuscolo (e2e_modello_minuscolo.py: GDN + attenzione piena + MoE, tokenizzatore di Thor)
e un corpus finto nello STESSO formato del corpus E2E (domini, cu_ptr, cu, ordine), fatto di testo vero.

Prove:
  U1  righe_lato: ordine d'uso = `ordine` del corpus; inizi dei segmenti == cu_seqlens
  U2  prepara: blocchi che coprono ogni riga una volta; piano immutabile (un secondo prepara diverso solleva)
  U3  esegui (CPU): tutti i blocchi sigillati; h della cache == RNE_bf16(h f32 di una corsa diretta della
      riga da sola), AL BIT, per ogni riga (isolamento fra righe e conversione)
  U4  ripresa: un blocco .parziale lasciato a meta' e un blocco cancellato si rifanno; i sigilli degli altri
      restano identici (stesso sha256 del blocco.json); la cache rifatta e' identica al bit
  U5  verifica: un byte cambiato in un h si scopre; rimesso a posto, verde
  U6  consuma: blocco cancellato, marcatore .consumato, `esegui` non lo rifa', `stato` lo conta
  U7  modo segmenti: h di una riga a piu' segmenti == segmenti calcolati come righe a se' (AL BIT); righe a
      un solo segmento identiche fra modo riga e modo segmenti
  U8  lucchetto: un secondo esegui sulla stessa corsa esce con 3; corpus cambiato dopo il piano: rifiuto
  U9  testa Q8_0 del teacher VERO: dequant_q8_0 == dequantize_row_q8_0 di ggml (C, ctypes) == gguf-py, AL BIT,
      su tutte le 248.320 righe (a pezzi); testa bf16 originale == gguf-py, AL BIT
  U10 logit_maestro == e2e_prototipo.logit_maestro, AL BIT
  U11 f0a_cache sul minuscolo (CPU): cache == RNE_bf16 della corsa nuova, logit ricalcolati ~ diretti (a pezzi)
  U12 coda: corsa + corse «dopo» (senza F0a)
  U13 riuso: righe gia' sigillate (altra corsa, altro ordine) prese per hard link; solo le nuove calcolate
  U17 (26/9) coda che aspetta piu' unita'; unita_teacher_attive
  U18 (27/9) PAUSA esplicita del teacher (file <radice>/PAUSA) nelle attese GPU
  U16 (26/9) previsione_riuso prima di esegui == righe riusate/calcolate davvero
  U15 (26/9) LucchettoGPU esclusivo fra processi; U13 esegue senza corpus_dir (cartella dal piano)
  U14 (26/9) f0a_kl: base == Q (stesso modello) -> KL ~ 0 su wiki e held-out; gruppi combinati; basi cancellate

Uso: CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 10 python3 e2e_test_teacher.py
     cartella: $E2E_TEACHER_TEST (default @@ROOT:lavoro@@/e2e/teacher/collaudo)
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import e2e_teacher as T  # noqa: E402
from e2e_dump_logit_leggi import Uscita, bf16_a_f32  # noqa: E402

LAVORO = Path(os.environ.get("E2E_TEACHER_TEST", "@@ROOT:lavoro@@/e2e/teacher/collaudo"))
L = 256
ESITI: list = []


def prova(nome: str, ok: bool, **d) -> None:
    ESITI.append({"prova": nome, "ok": bool(ok), **d})
    print(f"[{'OK ' if ok else 'NO '}] {nome}  {json.dumps(d, default=str)[:400]}", flush=True)


def al_bit(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def corpus_finto(dir_c: Path, ids_testo: np.ndarray) -> None:
    """Due domini (edu: riga intera; it_chat: 3 segmenti), train 6 righe + held-out 4, formato E2E."""
    import torch
    pos = [0]

    def prendi(n):
        x = ids_testo[pos[0]:pos[0] + n]
        pos[0] += n
        return x

    for lato, n_edu, n_chat in (("train", 3, 3), ("heldout", 2, 2)):
        dom = {"edu": torch.from_numpy(np.stack([prendi(L) for _ in range(n_edu)]).astype(np.int64)),
               "it_chat": torch.from_numpy(np.stack([prendi(L) for _ in range(n_chat)]).astype(np.int64))}
        cu_chat = [[0, 40 + 7 * r, 130 + 11 * r, L] for r in range(n_chat)]
        cu = {"edu": torch.tensor([0, L] * n_edu, dtype=torch.int32),
              "it_chat": torch.tensor(sum(cu_chat, []), dtype=torch.int32)}
        cu_ptr = {"edu": torch.arange(0, 2 * n_edu + 1, 2, dtype=torch.int64),
                  "it_chat": torch.arange(0, 4 * n_chat + 1, 4, dtype=torch.int64)}
        ordine = []
        for r in range(max(n_edu, n_chat)):
            if r < n_chat:
                ordine.append([1, r])
            if r < n_edu:
                ordine.append([0, r])
        d = {"l_seq": L, "lato": lato, "variante": "prova", "nomi_domini": ["edu", "it_chat"], "domini": dom,
             "cu_ptr": cu_ptr, "cu": cu, "inizio_esempio": {k: torch.ones(v.shape[0], dtype=torch.bool) for k, v in dom.items()},
             "fonti": {}, "esempi": {}, "ordine": torch.tensor(ordine, dtype=torch.int64)}
        nome = "e2e_corpus_4096_prova.pt" if lato == "train" else "e2e_heldout_4096_prova.pt"
        torch.save(d, dir_c / nome)


def diretta_f32(mq: Path, ids: np.ndarray, out: Path, inizi=None) -> np.ndarray:
    """h_nextn f32 di UNA riga calcolata da sola con lo strumento (riferimento indipendente dal ciclo)."""
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    ids.astype(np.int32).tofile(out / "ids.i32")
    cmd = [str(T.BIN), "logit", "-m", str(mq), "-o", str(out / "d"), "-t", "2", "-c", str(L), "-b", str(L),
           "-ub", "1024", "--lunghezza", str(len(ids)), "--uscite", "ultima", "--no-logit",
           "--tensore", "h_nextn", "--salva-tipo", "f32"]
    if inizi is not None:
        inizi.astype(np.int32).tofile(out / "inizi.i32")
        cmd += ["--segmenti", str(out / "inizi.i32")]
    subprocess.run(cmd + [str(out / "ids.i32")], check=True, capture_output=True, env=T.ambiente_strumento(True))
    return np.asarray(Uscita(out / "d").tensore(0, "h_nextn"), np.float32)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    t0 = time.time()
    assert LAVORO.name == "collaudo" and str(LAVORO).startswith("@@ROOT:lavoro@@/"), "cartella di collaudo inattesa"
    if LAVORO.exists():
        shutil.rmtree(LAVORO)
    LAVORO.mkdir(parents=True)
    rad = LAVORO / "radice"
    cdir = LAVORO / "corpus"
    cdir.mkdir()
    try:
        mq = LAVORO / "minuscolo.gguf"
        subprocess.run([sys.executable, str(QUI / "e2e_modello_minuscolo.py"), str(mq)], check=True, capture_output=True)
        subprocess.run([str(T.BIN), "tokenizza", str(mq), str(T.WIKI), str(L), str(LAVORO / "ids"), "12"],
                       check=True, capture_output=True, env=T.ambiente_strumento(True))
        testo = np.fromfile(LAVORO / "ids" / "token_tutti.i32", dtype=np.int32)[1000:]
        corpus_finto(cdir, testo)

        # U1
        c = T.righe_lato("train", cdir, "prova")
        attese = [("it_chat", 0), ("edu", 0), ("it_chat", 1), ("edu", 1), ("it_chat", 2), ("edu", 2)]
        ok_in = all(np.flatnonzero(c["inizi"][k]).tolist() == ([0] if d == "edu" else [0, 40 + 7 * r, 130 + 11 * r])
                    for k, (d, r) in enumerate(c["righe"]))
        prova("U1 righe_lato: ordine d'uso e inizi dei segmenti dai cu_seqlens", c["righe"] == attese and ok_in,
              righe=c["righe"])

        # U2
        lati = ["heldout", "train"]
        p = T.prepara("r", lati, "prova", "riga", 2, 1024, "bf16", mq, rad, cdir)
        piano = json.loads(p.read_text())
        cop = sorted((b["lato"], i) for b in piano["blocchi"] for _, _, i in b["righe"])
        prova("U2 prepara: ogni riga in un solo blocco, n_embd dal GGUF",
              cop == sorted([("heldout", i) for i in range(4)] + [("train", i) for i in range(6)])
              and len(piano["blocchi"]) == 5 and piano["modello"]["n_embd"] > 0, blocchi=len(piano["blocchi"]),
              n_embd=piano["modello"]["n_embd"])
        try:
            T.prepara("r", lati, "prova", "segmenti", 2, 1024, "bf16", mq, rad, cdir)
            prova("U2 piano immutabile: prepara diverso sulla stessa corsa solleva", False)
        except RuntimeError as e:
            prova("U2 piano immutabile: prepara diverso sulla stessa corsa solleva", True, messaggio=str(e)[:120])

        # U3
        rc = T.esegui("r", rad, cpu=True, margine_gib=0.5, corpus_dir=cdir)
        cm = T.CacheMaestro(rad / "r")
        diversi = []
        for lato in lati:
            cc = T.righe_lato(lato, cdir, "prova")
            for i in range(len(cc["righe"])):
                ref = diretta_f32(mq, cc["ids"][i], LAVORO / "dir")
                h = cm.h_numpy(lato, i)
                if not al_bit(np.asarray(h), T._rne_bf16(ref)):
                    diversi.append((lato, i))
                if not al_bit(cm.ids(lato, i), cc["ids"][i]):
                    diversi.append((lato, i, "ids"))
        prova("U3 esegui (CPU): 5 blocchi sigillati; h cache == RNE_bf16(h f32 della riga da sola) AL BIT", rc == 0
              and not diversi and cm.aggiorna() == 10, rc=rc, diversi=diversi, torch=str(cm.h("train", 0).dtype))
        reg = [json.loads(x)["evento"] for x in (rad / "r" / "registro.jsonl").read_text().splitlines()]
        prova("U3 registro: inizio, 5 blocchi, fine", reg.count("blocco") == 5 and reg[0] == "inizio" and reg[-1] == "fine",
              eventi=reg)

        # U4 ripresa
        sig_prima = {b["nome"]: sha(rad / "r" / b["nome"] / "blocco.json") for b in piano["blocchi"]}
        h_prima = np.asarray(cm.h_numpy("train", 2)).copy()
        b2 = piano["blocchi"][3]["nome"]  # train/blocco_0001 (righe 2-3)
        shutil.rmtree(rad / "r" / b2)
        (rad / "r" / (piano["blocchi"][4]["nome"] + ".parziale")).mkdir()
        (rad / "r" / (piano["blocchi"][4]["nome"] + ".parziale") / "rifiuto").write_text("mezzo")
        shutil.rmtree(rad / "r" / piano["blocchi"][4]["nome"])
        rc = T.esegui("r", rad, cpu=True, margine_gib=0.5, corpus_dir=cdir)
        cm.aggiorna()
        stessi = all(sha(rad / "r" / n / "blocco.json") == v for n, v in sig_prima.items()
                     if n not in (b2, piano["blocchi"][4]["nome"]))
        prova("U4 ripresa: blocco mancante e .parziale rifatti, gli altri sigilli identici, h identica al bit",
              rc == 0 and stessi and al_bit(np.asarray(cm.h_numpy("train", 2)), h_prima)
              and not (rad / "r" / (piano["blocchi"][4]["nome"] + ".parziale")).exists(), rc=rc)

        # U5 verifica
        f = rad / "r" / piano["blocchi"][0]["nome"] / cm.info("heldout", 0)["file_h"]
        orig = f.read_bytes()
        bb = bytearray(orig)
        bb[100] ^= 1
        f.write_bytes(bytes(bb))
        male = cm.verifica_blocco(piano["blocchi"][0]["nome"])
        f.write_bytes(orig)
        prova("U5 verifica: un bit cambiato si scopre, rimesso a posto e' verde",
              len(male) == 1 and cm.verifica_blocco(piano["blocchi"][0]["nome"]) == [], male=male)

        # U6 consuma
        cm.consuma(piano["blocchi"][2]["nome"])
        rc = T.esegui("r", rad, cpu=True, margine_gib=0.5, corpus_dir=cdir)
        st = T.stato_numeri(rad / "r", piano)
        prova("U6 consuma: cancellato, non rifatto, contato", rc == 0 and not (rad / "r" / piano["blocchi"][2]["nome"]).exists()
              and st["blocchi_consumati"] == 1 and st["token_fatti"] == st["token_totali"] and not cm.disponibile("train", 0),
              stato=st)

        # U7 segmenti
        T.prepara("s", lati, "prova", "segmenti", 2, 1024, "bf16", mq, rad, cdir)
        rc = T.esegui("s", rad, cpu=True, margine_gib=0.5, corpus_dir=cdir)
        cs = T.CacheMaestro(rad / "s")
        cc = T.righe_lato("train", cdir, "prova")
        i_chat = 0  # it_chat[0]
        cu = cs.cu_seqlens("train", i_chat)
        pezzi = [diretta_f32(mq, cc["ids"][i_chat][cu[j]:cu[j + 1]], LAVORO / f"pz{j}") for j in range(len(cu) - 1)]
        ok_seg = al_bit(np.asarray(cs.h_numpy("train", i_chat)), T._rne_bf16(np.concatenate(pezzi)))
        cm_r = T.CacheMaestro(rad / "r")
        uguali_uno = all(al_bit(np.asarray(cs.h_numpy("train", i)), np.asarray(cm_r.h_numpy("train", i)))
                         for i in (3, 5))    # righe edu (un segmento) ancora presenti nella corsa r
        diverse_multi = not al_bit(np.asarray(cs.h_numpy("train", 2)), np.asarray(cm_r.h_numpy("train", 2)))
        prova("U7 modo segmenti: riga a 3 segmenti == segmenti calcolati da soli AL BIT; righe a 1 segmento "
              "identiche fra i modi; righe a piu' segmenti diverse", rc == 0 and ok_seg and uguali_uno and diverse_multi
              and cs.modo == "segmenti" and cu == [0, 40, 130, L], cu=cu)

        # U8 lucchetto e corpus cambiato
        luc = open(rad / "r" / ".lucchetto", "w")
        import fcntl
        fcntl.flock(luc, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rc3 = T.esegui("r", rad, cpu=True, margine_gib=0.5, corpus_dir=cdir)
        luc.close()
        cdir2 = LAVORO / "corpus2"
        shutil.copytree(cdir, cdir2)
        import torch
        d = torch.load(cdir2 / "e2e_corpus_4096_prova.pt", weights_only=True)
        d["domini"]["edu"][0, 5] += 1
        torch.save(d, cdir2 / "e2e_corpus_4096_prova.pt")
        shutil.rmtree(rad / "r" / piano["blocchi"][3]["nome"])
        try:
            T.esegui("r", rad, cpu=True, margine_gib=0.5, corpus_dir=cdir2)
            rifiuto = False
        except RuntimeError as e:
            rifiuto = "cambiato" in str(e)
        prova("U8 lucchetto (rc 3) e corpus cambiato dopo il piano (rifiuto)", rc3 == 3 and rifiuto, rc_lucchetto=rc3)

        # U11 f0a_cache sul minuscolo (CPU): meccanica a pezzi, cache == RNE della corsa nuova, KL piccola
        tq = T.TestaMaestro("q8", gguf=mq)
        f0 = T.f0a_cache("r", rad, righe=(0, 1), da=192, n_thread=4, cpu=True, teste={"q8": tq}, pezzo=16)
        pr = f0["per_riga"]
        ok11 = all(x["cache_bf16_uguale_a_RNE_della_corsa_nuova"] and x["h_f32|testa_q8"]["argmax_uguale"] >= 0.95
                   and x["h_f32|testa_q8"]["kl_diretti_vs_ricalcolati_media"] < 1e-3
                   and np.isfinite(x["nll_media_diretta"]) for x in pr)
        prova("U11 f0a_cache (CPU, minuscolo): cache == RNE_bf16 della corsa nuova; logit ricalcolati ~ diretti",
              ok11 and (rad / "r" / "f0a" / "cache.json").exists() and not list((rad / "r" / "f0a").rglob("logit.f32")),
              riga0={k: v for k, v in pr[0].items() if k.startswith("h_f32") or k.startswith("cache")})

        # U12 coda: corsa + corse «dopo», senza F0a
        T.prepara("c1", ["heldout"], "prova", "riga", 2, 1024, "bf16", mq, rad, cdir)
        T.prepara("c2", ["heldout"], "prova", "segmenti", 2, 1024, "bf16", mq, rad, cdir)
        orig = T.esegui

        def _esegui_cpu(c, r=T.RADICE, **kw):
            return orig(c, r, cpu=True, margine_gib=0.5, corpus_dir=cdir)
        T.esegui = _esegui_cpu
        try:
            rc12 = T.coda("c1", ["c2"], rad, f0a=False)
        finally:
            T.esegui = orig
        n1 = T.CacheMaestro(rad / "c1").aggiorna()
        n2 = T.CacheMaestro(rad / "c2").aggiorna()
        prova("U12 coda: corsa e corsa «dopo» complete", rc12 == 0 and n1 == 4 and n2 == 4, rc=rc12, righe=[n1, n2])

        # U13 riuso per hard link: corpus nuovo con 2 righe gia' sigillate (in un altro ordine) + 1 riga nuova
        import torch
        cdir3 = LAVORO / "corpus3"
        cdir3.mkdir()
        shutil.copy(cdir / "e2e_heldout_4096_prova.pt", cdir3 / "e2e_heldout_4096_prova.pt")
        d0 = torch.load(cdir / "e2e_corpus_4096_prova.pt", weights_only=True)
        nuova = d0["domini"]["edu"][0].clone()
        nuova[17] = (int(nuova[17]) + 1) % 248320
        d3 = dict(d0)
        d3["domini"] = {"edu": torch.stack([d0["domini"]["edu"][2], nuova]), "it_chat": d0["domini"]["it_chat"][2:3].clone()}
        d3["cu"] = {"edu": torch.tensor([0, L, 0, L], dtype=torch.int32),
                    "it_chat": d0["cu"]["it_chat"][int(d0["cu_ptr"]["it_chat"][2]):int(d0["cu_ptr"]["it_chat"][3])].clone()}
        d3["cu_ptr"] = {"edu": torch.tensor([0, 2, 4]), "it_chat": torch.tensor([0, 4])}
        d3["ordine"] = torch.tensor([[0, 0], [1, 0], [0, 1]])
        torch.save(d3, cdir3 / "e2e_corpus_4096_prova.pt")
        T.prepara("rr", lati, "prova", "riga", 2, 1024, "bf16", mq, rad, cdir3)
        pv13 = T.previsione_riuso("rr", rad)          # 26/9: previsione PRIMA dell'esecuzione
        rc13 = T.esegui("rr", rad, cpu=True, margine_gib=0.5)   # 26/9: la cartella del corpus viene dal piano
        crr = T.CacheMaestro(rad / "rr")
        s0 = json.loads((rad / "rr" / "train" / "blocco_0000" / "blocco.json").read_text())
        s1 = json.loads((rad / "rr" / "train" / "blocco_0001" / "blocco.json").read_text())
        sh = json.loads((rad / "rr" / "heldout" / "blocco_0000" / "blocco.json").read_text())
        r0 = s0["righe"][0]
        src = rad / r0["riusata_da"]["corsa"] / r0["riusata_da"]["blocco"]
        src_file = src / next(x["file_h"] for x in json.loads((src / "blocco.json").read_text())["righe"]
                              if x["indice"] == r0["riusata_da"]["indice"])
        stesso_inode = os.stat(rad / "rr" / "train" / "blocco_0000" / r0["file_h"]).st_ino == os.stat(src_file).st_ino
        ref_nuova = diretta_f32(mq, nuova.numpy(), LAVORO / "dir_nuova")
        ok13 = (rc13 == 0 and s0["righe_riusate"] == 2 and s0["manifesto_sha256"] is None and s1["righe_calcolate"] == 1
                and sh["righe_riusate"] == 2 and stesso_inode
                and al_bit(np.asarray(crr.h_numpy("train", 2)), T._rne_bf16(ref_nuova))
                and al_bit(np.asarray(crr.h_numpy("train", 0)), np.asarray(T.CacheMaestro(rad / "r").h_numpy("train", 5)))
                and crr.verifica_blocco("train/blocco_0000") == [] and crr.verifica_blocco("train/blocco_0001") == [])
        prova("U13 riuso: righe gia' sigillate prese per hard link (stesso inode, sha ricontrollato), la nuova "
              "calcolata == riga da sola AL BIT; blocco tutto riusato senza strumento", ok13, rc=rc13,
              riusate=[s0["righe_riusate"], s1["righe_riusate"], sh["righe_riusate"]], stesso_inode=stesso_inode)
        pl = pv13["per_lato"]
        prova("U16 previsione_riuso: train 3 righe, 1 nuova (quella cambiata); held-out 0 nuove; combacia con l'esecuzione",
              pl["train"]["righe"] == 3 and pl["train"]["nuove"] == 1 and pl["heldout"]["nuove"] == 0
              and pl["train"]["nuove"] == s0["righe_calcolate"] + s1["righe_calcolate"]
              and pl["heldout"]["righe"] - pl["heldout"]["nuove"] == sum(
                  json.loads((rad / "rr" / b["nome"] / "blocco.json").read_text())["righe_riusate"]
                  for b in json.loads((rad / "rr" / "piano.json").read_text())["blocchi"] if b["lato"] == "heldout"),
              previsione=pl)

        # U14 (26/9) f0a_kl sul minuscolo (CPU): base = Q = lo stesso modello -> KL ~ 0 (solo la quantizzazione
        # uint16 del file base); due gruppi del held-out combinati; basi cancellate con tieni_basi=False
        k14 = T.f0a_kl("r", rad, chunks_wiki=2, righe_per_dominio=1, righe_per_chiamata=1, modello_base=mq,
                       ctx_wiki=L, cpu=True, n_thread=4, tieni_basi=False)
        gs = k14["heldout"]["gruppi"]
        ok14 = (k14["wiki"]["kld_media"][0] < 1e-4 and k14["heldout"]["kld_media"][0] < 1e-4
                and k14["wiki"]["stesso_top_pc"][0] >= 99.0 and all(g["stesso_top_pc"][0] >= 99.0 for g in gs)
                and len(gs) == 2 and sorted(d for _, d, _ in k14["heldout"]["righe"]) == ["edu", "it_chat"]
                and abs(k14["wiki"]["ppl_q"][0] - k14["wiki"]["ppl_base"][0]) < 1e-3 * k14["wiki"]["ppl_base"][0]
                and not list((rad / "r" / "f0a" / "kl").rglob("*.kld"))
                and not list((rad / "r" / "f0a" / "kl").rglob("logit.f32"))
                and (rad / "r" / "f0a" / "kl.json").exists())
        prova("U14 f0a_kl (CPU, minuscolo, base == Q): KL ~ 0 su wiki e held-out, 2 gruppi, basi e logit cancellati",
              ok14, wiki={k: k14["wiki"].get(k) for k in ("kld_media", "stesso_top_pc", "ppl_q", "ppl_base")},
              heldout=k14["heldout"]["kld_media"], righe=k14["heldout"]["righe"])

        # U15 (26/9) LucchettoGPU: esclusivo fra processi (un secondo processo non lo prende finche' e' tenuto)
        prova_lock = ("import fcntl,sys; f=open(sys.argv[1],'w')\n"
                      "try:\n fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB); print('preso')\n"
                      "except BlockingIOError:\n print('occupato')")
        with T.LucchettoGPU(rad):
            dentro = subprocess.run([sys.executable, "-c", prova_lock, str(rad / ".gpu.lock")], capture_output=True,
                                    text=True).stdout.strip()
        fuori = subprocess.run([sys.executable, "-c", prova_lock, str(rad / ".gpu.lock")], capture_output=True,
                               text=True).stdout.strip()
        prova("U15 LucchettoGPU: occupato per un altro processo mentre e' tenuto, libero dopo",
              dentro == "occupato" and fuori == "preso", dentro=dentro, fuori=fuori)

        # U17 (26/9) coda che aspetta piu' unita' (inesistenti = non attive): parte subito; elenco e2e-teacher* leggibile
        t17 = time.time()
        T.prepara("c3", ["heldout"], "prova", "riga", 2, 1024, "bf16", mq, rad, cdir)
        orig = T.esegui
        T.esegui = lambda c, r=T.RADICE, **kw: orig(c, r, cpu=True, margine_gib=0.5, corpus_dir=cdir)
        try:
            rc17 = T.coda("c3", [], rad, f0a=False, aspetta_unita="e2e-teacher-inesistente-a,e2e-teacher-inesistente-b")
        finally:
            T.esegui = orig
        att = T.unita_teacher_attive()
        prova("U17 coda con --aspetta-unita a piu' unita' non attive: parte subito; unita_teacher_attive() e' una lista",
              rc17 == 0 and time.time() - t17 < 120 and isinstance(att, list), rc=rc17, attive=att)

        # U18 (27/9) pausa ESPLICITA: file <radice>/PAUSA letto fresco; cablata in ENTRAMBE le attese GPU (blocchi e F0a)
        import inspect
        assenza = T.pausa_dichiarata(rad)
        (rad / "PAUSA").write_text("prova: EB2 senza coesistenza\n")
        con_motivo = T.pausa_dichiarata(rad)
        (rad / "PAUSA").write_text("")
        vuota = T.pausa_dichiarata(rad)
        (rad / "PAUSA").unlink()
        dopo = T.pausa_dichiarata(rad)
        src_blocchi, src_f0a = inspect.getsource(T.esegui), inspect.getsource(T.f0a_kl)
        prova("U18 PAUSA: assente -> None, motivo letto, vuota = pausa, tolta -> None; controllata nelle attese GPU "
              "di esegui e di f0a_kl",
              assenza is None and con_motivo == "prova: EB2 senza coesistenza" and bool(vuota) and dopo is None
              and "pausa_dichiarata(radice)" in src_blocchi and "pausa_dichiarata(radice)" in src_f0a,
              motivo=con_motivo, vuota=vuota)

        # U9 teste vere
        lib = ctypes.CDLL("@@ROOT:models@@/build/llamacpp-tq1-k1024/build/bin/libggml-base.so")
        lib.dequantize_row_q8_0.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        from gguf.quants import dequantize
        t = T._tensore_gguf(T.MODELLO_Q8, "output.weight")
        raw = np.asarray(t.data)
        V = raw.shape[0]
        diff_c = diff_py = 0
        for a in range(0, V, 16384):
            pz = np.ascontiguousarray(raw[a:a + 16384])
            mio = T.dequant_q8_0(pz)
            ref = np.empty(mio.size, np.float32)
            lib.dequantize_row_q8_0(pz.ctypes.data, ref.ctypes.data, ref.size)
            diff_c += int((mio.reshape(-1).view(np.uint32) != ref.view(np.uint32)).sum())
            diff_py += int((mio.view(np.uint32) != dequantize(pz, t.tensor_type).astype(np.float32).view(np.uint32)).sum())
        prova("U9 testa Q8_0 del teacher: dequant_q8_0 == ggml C == gguf-py AL BIT (248.320 righe)",
              diff_c == 0 and diff_py == 0 and V == 248320, righe=V, diff_c=diff_c, diff_gguf_py=diff_py)
        tb = T._tensore_gguf(T.TESTA_BF16, "output.weight")
        rb = np.asarray(tb.data)
        diff_b = 0
        for a in range(0, rb.shape[0], 16384):
            pz = np.ascontiguousarray(rb[a:a + 16384])
            mio = bf16_a_f32(pz.view(np.uint16).reshape(pz.shape[0], -1))
            diff_b += int((mio.view(np.uint32) != dequantize(pz, tb.tensor_type).astype(np.float32).view(np.uint32)).sum())
        prova("U9 testa bf16 originale: bf16_a_f32 == gguf-py AL BIT", diff_b == 0 and rb.shape[0] == 248320, diff=diff_b)

        # U10
        import e2e_prototipo as P
        g = torch.Generator().manual_seed(3)
        h = torch.randn(64, 256, generator=g).to(torch.bfloat16)
        W = torch.randn(1000, 256, generator=g)
        prova("U10 logit_maestro == e2e_prototipo.logit_maestro AL BIT",
              torch.equal(T.logit_maestro(h, W), P.logit_maestro(h, W)))
    except Exception as e:  # noqa: BLE001
        import traceback
        prova("ESECUZIONE", False, errore=traceback.format_exc()[-1500:])
    verde = all(e["ok"] for e in ESITI)
    esito = {"tutto_verde": verde, "prove": ESITI, "secondi": round(time.time() - t0, 1),
             "sha256_teacher_py": sha(QUI / "e2e_teacher.py"), "sha256_binario": sha(T.BIN)}
    (LAVORO / "esito_collaudo.json").write_text(json.dumps(esito, indent=1, default=str))
    print(f"\nTUTTO VERDE: {verde}  ({len(ESITI)} prove, {esito['secondi']} s) -> {LAVORO / 'esito_collaudo.json'}")
    return 0 if verde else 1


if __name__ == "__main__":
    sys.exit(main())

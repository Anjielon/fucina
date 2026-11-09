#!/usr/bin/env python3
"""e2e_teacher — il teacher della fase E2E sulle scale (progetto `e2e_progetto_scelto.md` §2.3, §3.1, §6, §9).
26/9/2026. File NUOVO: non modifica nessun file esistente.

## Cosa fa
Calcola, per ogni riga del corpus E2E, lo **stato nascosto finale del teacher dopo `output_norm`** (tensore
`h_nextn` del grafo qwen35moe, TUTTI i token: in `src/models/qwen35moe.cpp` il taglio su `inp_out_ids`
avviene dopo, nel tensore `result_norm`) e lo salva su disco in **bf16** (4 KiB a token, §2.3). I logit si
ricalcolano poi con `logit_maestro(h, W)` e la testa di `TestaMaestro`: UNA sola funzione, la stessa del
prototipo (`e2e_prototipo.logit_maestro`), usata sia qui (F0a/G0b) sia dalla perdita.

Motore: **llama.cpp Vulkan** (strumento `e2e_dump_logit`, collaudato su CPU contro llama-perplexity e
dump_strati) col **Q8_0** del teacher (`teacher-q8/`, MAI su CPU). Decisione del 26/9 (ordine di Angelo:
«motore piu' veloce disponibile se fedele entro i cancelli»). Misure di casa, 26/9 mattina, 4 righe da 4096
del held-out, casa accesa: ubatch 256/512/1024/2048/4096 -> 5,5 / 4,4 / **3,9** / 4,8 / 7,5 s a riga:
si usa **-ub 1024** (~1050 token/s), VRAM del processo ~36 GiB (picco di sistema 67,8 GiB con la casa).

## Due modi (il teacher DEVE vedere lo stesso contesto dello studente)
- `riga` (default): la riga da 4096 e' UNA sequenza causale, posizioni 0..4095 — come la forgia
  (`valida_finestra_8_12.py`: `create_causal_mask` su tutta la riga) e come lo studente di oggi
  (`e2e_studente.py`, `_contesto`: `create_causal_mask`, posizioni arange(T)).
- `segmenti`: ogni segmento dei `cu_seqlens` del corpus e' una sequenza a se' (memoria azzerata, posizioni
  da 0): e' la semantica di fla/attenzione varlen che il corpus prevede (§2.4). Costo misurato su 2 righe
  it_chat (10 e 8 segmenti): 4,7 contro 4,1 s a riga.
Il modo e' scritto nel piano e in ogni sigillo; `CacheMaestro.modo` lo espone e `cu_seqlens(...)` da' i
confini: chi allena DEVE verificare di usare lo stesso. Le righe a un solo segmento (edu, it_web, it_wiki)
sono identiche nei due modi (prova T14 dello strumento: un solo inizio == senza segmenti, al bit).

## Formato su disco (una corsa = una cartella in @@ROOT:lavoro@@/e2e/teacher/<corsa>)
    piano.json                 modello (percorso, byte, sha256), strumento (sha256), corpus (sha256), modo,
                               blocchi (righe in ORDINE d'uso: `ordine` del corpus, ogni prefisso ha la
                               composizione U), creato da `prepara`
    registro.jsonl             un evento per riga (inizio, blocco sigillato, errore, consumato)
    <lato>/blocco_NNNN/        blocco sigillato (esiste <=> completo e verificato); in costruzione:
                               <lato>/blocco_NNNN.parziale (cancellato e rifatto alla ripresa)
        ids.i32, inizi.i32     id dati al modello e inizi dei segmenti (1 = inizio), righe concatenate
        dump/manifesto.json    manifesto di e2e_dump_logit (esito ok)
        dump/seq_KKKK/h_nextn.bf16   intestazione 4 x int64 (2048, L, 1, 1) + bit bf16 [L, 2048] contigui
        blocco.json            SIGILLO: per riga indice nel lato, dominio, riga del corpus, file, sha256 di h
                               e degli id, max |h|; sha256 di modello/strumento/manifesto; tempi, token/s,
                               picco VRAM
Lettura: `CacheMaestro(radice).h(lato, i)` -> torch.bfloat16 [L, 2048] (memmap, nessuna copia).
Blocchi ROTANTI (§3.1): `CacheMaestro.consuma(nome)` cancella un blocco usato; `esegui` aspetta spazio
(margine `--margine-gib`) invece di riempire il disco.

## Uso
    python3 e2e_teacher.py prepara --corsa NOME [--variante tutte] [--lati heldout,train,pannelli]
                                   [--modo riga|segmenti] [--righe-blocco 128]
    python3 e2e_teacher.py lancia  --corsa NOME [--dopo ALTRA,..] [--senza-f0a]
                                   # unita' systemd --user «e2e-teacher» (sopravvive): `coda` = esegui NOME,
                                   # poi f0a-cache e f0a-ppl su NOME, poi esegui delle corse --dopo
    python3 e2e_teacher.py esegui  --corsa NOME        # il ciclo (lo lancia l'unita'); ripresa per blocco
    python3 e2e_teacher.py stato   --corsa NOME
    python3 e2e_teacher.py riuso   --corsa NOME        # (26/9) righe gia' sigillate / previste altrove / nuove
    python3 e2e_teacher.py verifica --corsa NOME       # sha256 di ogni file contro il sigillo
    python3 e2e_teacher.py f0a-cache --corsa NOME      # G0b: logit dalla cache == avanti diretto (GPU)
    python3 e2e_teacher.py f0a-ppl --corsa NOME        # PPL wikitext del Q8 (llama-perplexity) e identita'
                                                       # strumento == llama-perplexity su Vulkan (GPU)
    python3 e2e_teacher.py f0a-kl --corsa NOME         # (26/9) KL(bf16 || Q8) su wikitext e held-out E2E
                                                       # (llama-perplexity, Vulkan); ASPETTA ~66 GiB di VRAM
                                                       # liberi: con la casa accesa non parte (§5: casa giu')
Test (solo CPU, modello minuscolo): `e2e_test_teacher.py`.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
sys.path.insert(0, "@@ROOT:models@@/build/llamacpp-tq1-k1024/gguf-py")
from e2e_dump_logit_leggi import Uscita, bf16_a_f32, leggi_tensore  # noqa: E402

VERSIONE = "e2e_teacher 1 (26/9/2026)"
BIN = QUI / "e2e_dump_logit"
GUARDIA = QUI / "strumenti" / "gpu_occupata.sh"
MODELLO_Q8 = Path("@@ROOT:models@@/gguf/tony-tern/teacher-q8/Ornith-1.5-35B-A3B-Q8_0.gguf")
# la testa bf16 ORIGINALE del teacher (§2.3: «i logit si ricalcolano con la testa bf16 originale»)
TESTA_BF16 = Path("@@ROOT:models@@/gguf/thor-src/Ornith-1.5-35B-A3B-bf16/Ornith-1.5-35B-A3B-bf16-00002-of-00002.gguf")
CORPUS = Path("@@ROOT:models@@/gguf/tony-tern/calibrazione/e2e")
RADICE = Path("@@ROOT:lavoro@@/e2e/teacher")
PPL_BIN = Path("@@ROOT:models@@/build/llamacpp-tq1-k1024/build/bin/llama-perplexity")
WIKI = Path("@@ROOT:odino@@/collaudo/corpus/wiki.test.raw")
N_EMBD = 2048
VRAM_UTILE_MIB = int(95.15 * 1024)          # oltre, i byte scivolano in GTT (misura del 5/9)
VRAM_PROCESSO_MIB = 37 * 1024               # misurato: 35,3 GiB modello + 0,6 calcolo a -ub 1024 (+ margine)
SYS_VRAM = Path("/sys/class/drm/card0/device/mem_info_vram_used")
SYS_GTT = Path("/sys/class/drm/card0/device/mem_info_gtt_used")
UNITA = "e2e-teacher"


# ═══════════════════════════════════════ utilita' ═══════════════════════════════════════
def adesso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(p: Path, blocco: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(blocco), b""):
            h.update(b)
    return h.hexdigest()


def scrivi_json_atomico(p: Path, dati) -> None:
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(dati, f, indent=1, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def libero_gib(p: Path) -> float:
    s = os.statvfs(p)
    return s.f_bavail * s.f_frsize / 2**30


def mem_disponibile_gib() -> float:
    for r in Path("/proc/meminfo").read_text().splitlines():
        if r.startswith("MemAvailable:"):
            return int(r.split()[1]) / 2**20
    raise RuntimeError("MemAvailable assente")


def _nvidia_mib() -> int:   # 16/10: su NVIDIA (pod) i file sysfs di amdgpu non esistono
    import subprocess
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout.split()
    return int(out[0])


def vram_mib() -> int:
    return int(SYS_VRAM.read_text()) // 2**20 if SYS_VRAM.exists() else _nvidia_mib()


def gtt_mib() -> int:
    return int(SYS_GTT.read_text()) // 2**20 if SYS_GTT.exists() else 0   # NVIDIA: niente GTT (memoria non unificata)


def ambiente_strumento(cpu: bool) -> dict:
    env = dict(os.environ)
    env.pop("HSA_OVERRIDE_GFX_VERSION", None)   # lezione del 25/9 17:5x: ereditato rompe torch/ROCm
    env.pop("GGML_VK_VISIBLE_DEVICES", None)
    if cpu:
        env.update(CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", GGML_VK_VISIBLE_DEVICES="")
    return env


# ═══════════════════════════════════════ corpus ═══════════════════════════════════════
def _torch():
    import torch
    return torch


def righe_lato(lato: str, corpus_dir: Path, variante: str) -> dict:
    """Righe di un lato del corpus, in ORDINE d'uso. Ritorna {file, sha256, L, righe: [(dominio, riga)],
    ids: int32 [N, L], inizi: int32 [N, L] (1 = inizio di un segmento dei cu_seqlens)}. Per i pannelli
    (nessun cu_seqlens, lunghezze 1024 e 4096) ogni riga e' un solo segmento e il lato si spezza per L."""
    torch = _torch()
    if lato in ("train", "heldout"):
        f = corpus_dir / (f"e2e_corpus_4096_{variante}.pt" if lato == "train" else f"e2e_heldout_4096_{variante}.pt")
        d = torch.load(f, weights_only=True)
        L = int(d["l_seq"])
        nomi = list(d["nomi_domini"])
        righe, ids, inizi = [], [], []
        for di, r in d["ordine"].tolist():
            dom = nomi[di]
            x = d["domini"][dom][r].numpy()
            cu = d["cu"][dom][int(d["cu_ptr"][dom][r]):int(d["cu_ptr"][dom][r + 1])].numpy()
            if cu[0] != 0 or cu[-1] != L or (np.diff(cu) <= 0).any():
                raise ValueError(f"{f.name} {dom}[{r}]: cu_seqlens non validi {cu[:5]}...")
            fl = np.zeros(L, np.int32)
            fl[cu[:-1]] = 1
            righe.append((dom, int(r)))
            ids.append(x.astype(np.int32))
            inizi.append(fl)
        n_attese = sum(int(v.shape[0]) for v in d["domini"].values())
        if len(righe) != n_attese or len(set(righe)) != n_attese:
            raise ValueError(f"{f.name}: `ordine` non copre ogni riga una volta ({len(righe)} contro {n_attese})")
        return {"file": str(f), "sha256": sha256_file(f), "L": L, "righe": righe,
                "ids": np.stack(ids), "inizi": np.stack(inizi)}
    if lato.startswith("pannelli"):
        f = corpus_dir / "e2e_heldout_pannelli.pt"
        d = torch.load(f, weights_only=True)
        L_voluto = int(lato.split("_")[1]) if "_" in lato else None
        righe, ids = [], []
        for dom, X in d["domini"].items():
            if L_voluto is not None and X.shape[1] != L_voluto:
                continue
            for r in range(X.shape[0]):
                righe.append((dom, r))
                ids.append(X[r].numpy().astype(np.int32))
        Ls = {len(x) for x in ids}
        if len(Ls) != 1:
            raise ValueError(f"lato {lato}: lunghezze diverse {Ls} (usare pannelli_1024 e pannelli_4096)")
        L = Ls.pop()
        inizi = np.zeros((len(ids), L), np.int32)
        inizi[:, 0] = 1
        return {"file": str(f), "sha256": sha256_file(f), "L": L, "righe": righe,
                "ids": np.stack(ids), "inizi": inizi}
    raise ValueError(f"lato sconosciuto: {lato}")


# ═══════════════════════════════════════ piano ═══════════════════════════════════════
def prepara(corsa: str, lati: list, variante: str = "tutte", modo: str = "riga", righe_blocco: int = 128,
            ub: int = 1024, salva_tipo: str = "bf16", modello: Path = MODELLO_Q8, radice: Path = RADICE,
            corpus_dir: Path = CORPUS, sha_modello: Optional[str] = None) -> Path:
    """Scrive piano.json (una volta: se esiste e differisce, solleva). sha256 del modello: letto per intero
    (37,8 GB ~ 1 min) e confrontato col file .sha256 accanto, se c'e'."""
    if modo not in ("riga", "segmenti"):
        raise ValueError("modo: riga | segmenti")
    dir_c = radice / corsa
    dir_c.mkdir(parents=True, exist_ok=True)
    if sha_modello is None:
        sha_modello = sha256_file(modello, 1 << 26)
    accanto = Path(str(modello) + ".sha256")
    if accanto.exists() and accanto.read_text().split()[0] != sha_modello:
        raise RuntimeError(f"sha256 del modello {sha_modello} diverso da {accanto}")
    blocchi, corpus = [], {}
    for lato in lati:
        c = righe_lato(lato, corpus_dir, variante)
        corpus[lato] = {"file": c["file"], "sha256": c["sha256"], "L": c["L"], "n_righe": len(c["righe"]),
                        "segmenti_totali": int(c["inizi"].sum())}
        for k, a in enumerate(range(0, len(c["righe"]), righe_blocco)):
            b = min(a + righe_blocco, len(c["righe"]))
            blocchi.append({"nome": f"{lato}/blocco_{k:04d}", "lato": lato, "L": c["L"], "da": a, "a": b,
                            "righe": [[dom, r, i] for i, (dom, r) in enumerate(c["righe"][a:b], start=a)]})
    piano = {
        "versione": VERSIONE, "corsa": corsa, "creato": adesso(), "modo": modo, "salva_tipo": salva_tipo,
        "ub": ub, "variante": variante, "lati": lati,
        "modello": {"percorso": str(modello), "byte": modello.stat().st_size, "sha256": sha_modello,
                    "n_embd": n_embd_gguf(modello)},
        "strumento": {"percorso": str(BIN), "sha256": sha256_file(BIN)},
        "corpus": corpus, "blocchi": blocchi,
        "formato": "h_nextn (dopo output_norm, tutti i token) bf16 [L, 2048] per riga; vedi e2e_teacher.py",
    }
    p = dir_c / "piano.json"
    if p.exists():
        vecchio = json.loads(p.read_text())
        chiavi = ("modo", "salva_tipo", "ub", "variante", "modello", "strumento", "corpus", "blocchi")
        diversi = [k for k in chiavi if vecchio.get(k) != piano.get(k)]
        if diversi:
            raise RuntimeError(f"{p} esiste gia' e differisce in {diversi}: nuova corsa con un altro nome")
        return p
    scrivi_json_atomico(p, piano)
    return p


# ═══════════════════════════════════════ registro ═══════════════════════════════════════
def registra(dir_c: Path, evento: str, **dati) -> None:
    riga = json.dumps({"ora": adesso(), "evento": evento, **dati}, default=str)
    with open(dir_c / "registro.jsonl", "a") as f:
        f.write(riga + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(f"[{adesso()}] {evento} {json.dumps(dati, default=str)[:300]}", flush=True)


# ═══════════════════════════════════════ un blocco ═══════════════════════════════════════
class Picchi(threading.Thread):
    """Picco di VRAM/GTT di SISTEMA (MiB) durante un processo, a campioni di 0,5 s."""

    def __init__(self):
        super().__init__(daemon=True)
        self.vram = self.gtt = 0
        self._fine = threading.Event()

    def run(self):
        while not self._fine.is_set():
            try:
                self.vram = max(self.vram, vram_mib())
                self.gtt = max(self.gtt, gtt_mib())
            except OSError:
                pass
            self._fine.wait(0.5)

    def ferma(self):
        self._fine.set()
        self.join(2)


def _chiave_riga(ids_riga: np.ndarray, inizi_riga: np.ndarray, modo: str) -> tuple:
    """Identita' di una riga per il riuso: sha256 degli id (+ inizi dei segmenti se il modo li usa)."""
    k = hashlib.sha256(np.ascontiguousarray(ids_riga, np.int32).tobytes()).hexdigest()
    return (k, tuple(np.flatnonzero(inizi_riga).tolist()) if modo == "segmenti" else None)


def indice_riuso(radice: Path, piano: dict) -> dict:
    """26/9: righe gia' SIGILLATE in qualunque corsa sotto `radice` con gli stessi modello (sha256), strumento
    (sha256), modo, salva_tipo, ub e L: {chiave_riga: (file h, sha256_h, corsa, blocco, indice)}. Un corpus
    ricostruito o allargato (E-B) non ricalcola le righe che il teacher ha gia' visto."""
    idx: dict = {}
    for pp in sorted(radice.glob("*/piano.json")):
        try:
            q = json.loads(pp.read_text())
        except (OSError, ValueError):
            continue
        if (q["modello"]["sha256"], q["strumento"]["sha256"], q["modo"], q["salva_tipo"], q["ub"]) != \
                (piano["modello"]["sha256"], piano["strumento"]["sha256"], piano["modo"], piano["salva_tipo"], piano["ub"]):
            continue
        for bl in q["blocchi"]:
            sp = pp.parent / bl["nome"] / "blocco.json"
            if not sp.exists():
                continue
            sg = json.loads(sp.read_text())
            for r in sg["righe"]:
                seg = tuple(r["segmenti_corpus"]) if q["modo"] == "segmenti" else None
                idx.setdefault((sg["L"], r["sha256_ids"], seg),
                               (sp.parent / r["file_h"], r["sha256_h"], q["corsa"], bl["nome"], r["indice"]))
    return idx


def previsione_riuso(corsa: str, radice: Path = RADICE) -> dict:
    """26/9: quante righe del piano di `corsa` il riuso prendera' per hard link. Divise in: gia' SIGILLATE in
    un'altra corsa compatibile (stessi modello, strumento, modo, salva_tipo, ub, L), solo PREVISTE nel piano di
    un'altra corsa compatibile (saranno sigillate prima, se quella corsa gira prima), e NUOVE (da calcolare).
    La chiave e' la stessa di `esegui_blocco`: sha256 degli id int32 (+ inizi dei segmenti nel modo segmenti)."""
    dir_c = radice / corsa
    piano = json.loads((dir_c / "piano.json").read_text())
    param = lambda q: (q["modello"]["sha256"], q["strumento"]["sha256"], q["modo"], q["salva_tipo"], q["ub"])  # noqa: E731
    sig = {k for k, v in indice_riuso(radice, piano).items()
           if v[2] != corsa}
    previste: dict = {}
    for pp in sorted(radice.glob("*/piano.json")):
        q = json.loads(pp.read_text())
        if q["corsa"] == corsa or param(q) != param(piano):
            continue
        for lato, c in q["corpus"].items():
            d = righe_lato(lato, Path(c["file"]).parent, q["variante"])
            for k in range(len(d["righe"])):
                ch, seg = _chiave_riga(d["ids"][k], d["inizi"][k], q["modo"])
                previste.setdefault((d["L"], ch, seg), f"{q['corsa']}:{lato}")
    ris = {"corsa": corsa, "per_lato": {}}
    for lato, c in piano["corpus"].items():
        d = righe_lato(lato, Path(c["file"]).parent, piano["variante"])
        n_s = n_p = 0
        da_dove: dict = {}
        for k in range(len(d["righe"])):
            chiave = (d["L"], *_chiave_riga(d["ids"][k], d["inizi"][k], piano["modo"]))
            if chiave in sig:
                n_s += 1
            elif chiave in previste:
                n_p += 1
                da_dove[previste[chiave]] = da_dove.get(previste[chiave], 0) + 1
        n = len(d["righe"])
        ris["per_lato"][lato] = {"righe": n, "gia_sigillate": n_s, "previste_altrove": n_p, "da_dove": da_dove,
                                 "nuove": n - n_s - n_p, "token_nuovi": (n - n_s - n_p) * d["L"]}
    return ris


def esegui_blocco(dir_c: Path, piano: dict, blocco: dict, dati: dict, cpu: bool = False, n_thread: int = 4,
                  timeout_s: Optional[float] = None, riuso: bool = True) -> dict:
    """Calcola e sigilla UN blocco. `dati` = righe_lato del suo lato. Solleva su qualunque incoerenza.
    Con `riuso` le righe gia' sigillate altrove (stessa chiave, stessi parametri) si prendono per HARD LINK
    (stesso disco; lo sha256 si ricontrolla sul file collegato) e lo strumento calcola solo le altre."""
    fin = dir_c / blocco["nome"]
    tmp = fin.with_name(fin.name + ".parziale")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    a, b, L = blocco["da"], blocco["a"], blocco["L"]
    n = b - a
    ids = np.ascontiguousarray(dati["ids"][a:b], dtype=np.int32)
    inizi = np.ascontiguousarray(dati["inizi"][a:b], dtype=np.int32)
    if [tuple(x[:2]) for x in blocco["righe"]] != [tuple(x) for x in dati["righe"][a:b]]:
        raise RuntimeError(f"{blocco['nome']}: righe del corpus diverse dal piano")
    ids.tofile(tmp / "ids.i32")
    inizi.tofile(tmp / "inizi.i32")
    trovate: dict = {}
    if riuso:
        idx = indice_riuso(dir_c.parent, piano)
        for k in range(n):
            ch, seg = _chiave_riga(ids[k], inizi[k], piano["modo"])
            v = idx.get((L, ch, seg))
            if v is not None and v[0].exists():
                trovate[k] = v
    da_calc = [k for k in range(n) if k not in trovate]
    n_calc = len(da_calc)
    ids[da_calc].tofile(tmp / "calc_ids.i32")
    inizi[da_calc].tofile(tmp / "calc_inizi.i32")
    base_vram = vram_mib()
    pic = Picchi()
    sec = 0.0
    man = None
    t0 = time.time()
    if n_calc:
        cmd = [str(BIN), "logit"] + ([] if cpu else ["--vulkan"]) + [
            "-m", piano["modello"]["percorso"], "-o", str(tmp / "dump"), "-t", str(n_thread),
            "-c", str(L), "-b", str(L), "-ub", str(piano["ub"]), "--lunghezza", str(L),
            "--uscite", "ultima", "--no-logit", "--tensore", "h_nextn", "--salva-tipo", piano["salva_tipo"]]
        if piano["modo"] == "segmenti":
            cmd += ["--segmenti", str(tmp / "calc_inizi.i32")]
        cmd.append(str(tmp / "calc_ids.i32"))
        if timeout_s is None:
            timeout_s = 600 + 4 * n_calc * L / 800.0        # 4 volte il previsto a 800 token/s, + caricamento
        pic.start()
        with open(tmp / "strumento.log", "w") as log:
            # 26/9: niente os.nice(5) in piu' (era nice 5 dell'unita' + 5 = 10, pari agli agenti CPU a nice 10:
            # sotto contesa il blocco 2 del train E-A e' sceso da 967 a 693 token/s). Il processo eredita il nice
            # dell'unita' (5): davanti agli agenti CPU, dietro alla casa (0).
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=ambiente_strumento(cpu))
            try:
                rc = p.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                p.terminate()
                try:
                    p.wait(30)
                except subprocess.TimeoutExpired:
                    p.kill()
                pic.ferma()
                raise RuntimeError(f"{blocco['nome']}: oltre {timeout_s:.0f} s, fermato")
            except BaseException:
                p.terminate()
                pic.ferma()
                raise
        sec = time.time() - t0
        pic.ferma()
        if rc != 0:
            coda = (tmp / "strumento.log").read_text()[-1500:]
            raise RuntimeError(f"{blocco['nome']}: e2e_dump_logit rc={rc}\n{coda}")
        u = Uscita(tmp / "dump")
        man = u.man
        if man["contesto"]["salva_tipo"] != piano["salva_tipo"] or u.n_sequenze != n_calc:
            raise RuntimeError(f"{blocco['nome']}: manifesto incoerente ({u.n_sequenze} sequenze, attese {n_calc})")
        if bool(man["contesto"]["segmenti"]) != (piano["modo"] == "segmenti"):
            raise RuntimeError(f"{blocco['nome']}: modo dei segmenti diverso dal piano")
    righe = []
    j = 0
    for k in range(n):
        dom, r, i = blocco["righe"][k]
        riga = {"indice": i, "dominio": dom, "riga": r,
                "sha256_ids": hashlib.sha256(ids[k].tobytes()).hexdigest(),
                "segmenti_corpus": np.flatnonzero(inizi[k]).tolist()}
        if k in trovate:
            src, sha_src, c_src, b_src, i_src = trovate[k]
            (tmp / "riuso").mkdir(exist_ok=True)
            fh = tmp / "riuso" / f"riga_{k:04d}{src.suffix}"
            os.link(src, fh)
            if sha256_file(fh) != sha_src:
                raise RuntimeError(f"{blocco['nome']} riga {k}: il file riusato {src} non ha piu' lo sha256 del sigillo")
            riga.update(file_h=str(fh.relative_to(tmp)), sha256_h=sha_src,
                        riusata_da={"corsa": c_src, "blocco": b_src, "indice": i_src})
        else:
            s = man["sequenze"][j]
            t = s["tensori"]["h_nextn"]
            if s["n_token"] != L or t["tipo"] != piano["salva_tipo"] or t["ne"] != [piano["modello"]["n_embd"], L, 1, 1] \
                    or t["righe_uguali_a"] != "token":
                raise RuntimeError(f"{blocco['nome']} seq {j}: forma/tipo inattesi {t}")
            if not np.array_equal(u.ids(j), ids[k]):
                raise RuntimeError(f"{blocco['nome']} seq {j}: id scritti diversi da quelli dati")
            if piano["modo"] == "segmenti" and s["segmenti"] != riga["segmenti_corpus"]:
                raise RuntimeError(f"{blocco['nome']} seq {j}: segmenti diversi dagli inizi del corpus")
            fh = tmp / "dump" / t["file"]
            riga.update(file_h="dump/" + t["file"], sha256_h=sha256_file(fh))
            j += 1
        h = leggi_tensore(fh)
        hf = bf16_a_f32(h) if piano["salva_tipo"] == "bf16" else np.asarray(h, np.float32)
        mx = float(np.abs(hf).max())
        if not np.isfinite(mx) or h.shape != (L, piano["modello"]["n_embd"]):
            raise RuntimeError(f"{blocco['nome']} riga {k}: h non finita o di forma {h.shape}")
        riga["max_abs_h"] = mx
        righe.append(riga)
    sec_calc = sum(s["secondi"] for s in man["sequenze"]) if man else 0.0
    sigillo = {
        "versione": VERSIONE, "nome": blocco["nome"], "lato": blocco["lato"], "L": L, "modo": piano["modo"],
        "salva_tipo": piano["salva_tipo"], "ub": piano["ub"], "backend": man["backend"] if man else "riuso",
        "modello_sha256": piano["modello"]["sha256"], "strumento_sha256": sha256_file(BIN),
        "strumento_versione": man["versione"] if man else None,
        "manifesto_sha256": sha256_file(tmp / "dump" / "manifesto.json") if man else None,
        "sha256_ids_blocco": sha256_file(tmp / "ids.i32"), "sha256_inizi_blocco": sha256_file(tmp / "inizi.i32"),
        "righe": righe, "righe_calcolate": n_calc, "righe_riusate": n - n_calc,
        "secondi": round(time.time() - t0, 2), "secondi_strumento": round(man["secondi_totali"], 2) if man else 0.0,
        "secondi_calcolo": round(sec_calc, 2), "token": n * L, "token_calcolati": n_calc * L,
        "token_al_secondo": round(n_calc * L / sec, 1) if sec else None,
        "token_al_secondo_calcolo": round(n_calc * L / sec_calc, 1) if sec_calc else None,
        "vram_base_MiB": base_vram, "vram_picco_MiB": pic.vram, "gtt_picco_MiB": pic.gtt,
        "rss_max_kib": man["rss_max_kib"] if man else None, "sigillato": adesso(),
    }
    if sigillo["strumento_sha256"] != piano["strumento"]["sha256"]:
        raise RuntimeError("il binario e2e_dump_logit e' cambiato dopo il piano")
    scrivi_json_atomico(tmp / "blocco.json", sigillo)
    os.replace(tmp, fin)
    return sigillo


# ═══════════════════════════════════════ ciclo ═══════════════════════════════════════
def pausa_dichiarata(radice: Path = RADICE) -> Optional[str]:
    """27/9: pausa ESPLICITA del teacher su GPU — file `<radice>/PAUSA`, il contenuto e' il motivo; None se assente.
    Prima la pausa era IMPLICITA («EB2 tiene la VRAM, la guardia dice occupata») e aveva un buco: nel riavvio di EB2
    (ripresa salvata 10:03:42, processo nuovo 10:07:51) la GPU e' sembrata libera, il teacher ha preso il blocco 27,
    EB2 e' ripartito sopra (42 + 37 + 31 GiB casa) e alle 10:10:37 il kernel ha resettato la coda di calcolo
    (comp_1.1.0 timeout, e2e_dump_logit rc=-6). Una pausa decisa si SCRIVE: il file sopravvive ai riavvii di chiunque,
    e si toglie solo quando chi l'ha messa decide di riprendere (`rm <radice>/PAUSA`). Illeggibile = pausa (severa)."""
    f = radice / "PAUSA"
    try:
        return f.read_text().strip() or "PAUSA (senza motivo scritto)"
    except FileNotFoundError:
        return None
    except OSError as e:
        return f"PAUSA illeggibile ({e}): si aspetta"


def _attendi(cond, descr: str, dir_c: Path, ogni_s: float = 30.0, avviso_s: float = 600.0) -> None:
    t0, ultimo = time.time(), -1e9
    while True:
        ok, dettaglio = cond()
        if ok:
            if time.time() - t0 > 1:
                registra(dir_c, "attesa_finita", cosa=descr, secondi=round(time.time() - t0))
            return
        if time.time() - ultimo > avviso_s:
            registra(dir_c, "attesa", cosa=descr, dettaglio=dettaglio)
            ultimo = time.time()
        time.sleep(ogni_s)


class LucchettoGPU:
    """26/9: UNA corsa GPU del teacher per volta fra processi diversi (coda + corse accodate + f0a): flock su
    RADICE/.gpu.lock tenuto per tutto il lavoro GPU di un blocco. In CPU (test) non si prende."""

    def __init__(self, radice: Path, attivo: bool = True):
        self.p = Path(radice) / ".gpu.lock"
        self.attivo = attivo
        self.f = None

    def __enter__(self):
        if self.attivo:
            self.p.parent.mkdir(parents=True, exist_ok=True)
            self.f = open(self.p, "w")
            fcntl.flock(self.f, fcntl.LOCK_EX)
        return self

    def __exit__(self, *_):
        if self.f is not None:
            fcntl.flock(self.f, fcntl.LOCK_UN)
            self.f.close()
            self.f = None


def esegui(corsa: str, radice: Path = RADICE, cpu: bool = False, margine_gib: float = 6.0, tentativi: int = 3,
           corpus_dir: Optional[Path] = None, n_thread: int = 4, solo: Optional[list] = None) -> int:
    """`corpus_dir` None (26/9) = la cartella scritta nel piano per ogni lato (un corpus E-B in un'altra cartella
    non si confonde con quello di E-A)."""
    dir_c = radice / corsa
    piano = json.loads((dir_c / "piano.json").read_text())
    lucchetto = open(dir_c / ".lucchetto", "w")
    try:
        fcntl.flock(lucchetto, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"un altro `esegui` tiene gia' {dir_c}", file=sys.stderr)
        return 3
    fermati = {"segnale": None}

    def _segnale(sig, _f):
        fermati["segnale"] = sig
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _segnale)
    if sha256_file(BIN) != piano["strumento"]["sha256"]:
        raise RuntimeError("binario e2e_dump_logit diverso da quello del piano")
    m = Path(piano["modello"]["percorso"])
    if m.stat().st_size != piano["modello"]["byte"]:
        raise RuntimeError("modello di taglia diversa dal piano")
    registra(dir_c, "inizio", pid=os.getpid(), cpu=cpu, blocchi=len(piano["blocchi"]))
    cache_lati: dict = {}
    fatti = 0
    try:
        for bl in piano["blocchi"]:
            if solo and bl["lato"] not in solo:
                continue
            fin = dir_c / bl["nome"]
            if (fin / "blocco.json").exists() or (dir_c / (bl["nome"] + ".consumato")).exists():
                continue
            if bl["lato"] not in cache_lati:
                cdir = Path(corpus_dir) if corpus_dir else Path(piano["corpus"][bl["lato"]]["file"]).parent
                c = righe_lato(bl["lato"], cdir, piano["variante"])
                if c["sha256"] != piano["corpus"][bl["lato"]]["sha256"]:
                    raise RuntimeError(f"corpus {bl['lato']} cambiato dopo il piano")
                cache_lati = {bl["lato"]: c}
            dati = cache_lati[bl["lato"]]
            n = bl["a"] - bl["da"]
            serve_gib = n * bl["L"] * (piano["modello"]["n_embd"] * (2 if piano["salva_tipo"] != "f32" else 4) + 8) / 2**30
            (dir_c / bl["lato"]).mkdir(exist_ok=True)
            _attendi(lambda: (libero_gib(dir_c) >= serve_gib + margine_gib,
                              f"liberi {libero_gib(dir_c):.1f} GiB, servono {serve_gib:.1f}+{margine_gib}"),
                     "disco", dir_c)
            _attendi(lambda: (mem_disponibile_gib() >= 3.0, f"MemAvailable {mem_disponibile_gib():.1f} GiB"),
                     "RAM", dir_c)
            for t in range(1, tentativi + 1):
                try:
                    with LucchettoGPU(radice, attivo=not cpu):
                        if not cpu:
                            def _gpu():
                                motivo = pausa_dichiarata(radice)
                                if motivo:
                                    return False, f"PAUSA dichiarata ({radice / 'PAUSA'}): {motivo[:200]}"
                                p = subprocess.run([str(GUARDIA)], capture_output=True, text=True)
                                v = vram_mib()
                                ok = p.returncode == 0 and v + VRAM_PROCESSO_MIB <= VRAM_UTILE_MIB
                                return ok, (f"guardia rc={p.returncode} {p.stdout.strip()[:200]}; "
                                            f"VRAM {v} MiB + {VRAM_PROCESSO_MIB}")
                            _attendi(_gpu, "GPU", dir_c)
                        s = esegui_blocco(dir_c, piano, bl, dati, cpu=cpu, n_thread=n_thread)
                    registra(dir_c, "blocco", nome=bl["nome"], righe=n, riusate=s["righe_riusate"], secondi=s["secondi"],
                             token_al_secondo=s["token_al_secondo"], calcolo_tok_s=s["token_al_secondo_calcolo"],
                             vram_picco_MiB=s["vram_picco_MiB"], gtt_picco_MiB=s["gtt_picco_MiB"],
                             max_abs_h=max(r["max_abs_h"] for r in s["righe"]), libero_gib=round(libero_gib(dir_c), 1))
                    fatti += 1
                    break
                except KeyboardInterrupt:
                    raise
                except Exception as e:  # noqa: BLE001 — si registra e si ritenta, poi si ferma
                    registra(dir_c, "errore", nome=bl["nome"], tentativo=t, errore=str(e)[:1500])
                    if t == tentativi:
                        return 2
                    time.sleep(30)
    except KeyboardInterrupt:
        registra(dir_c, "fermato", segnale=fermati["segnale"], fatti=fatti)
        return 130
    registra(dir_c, "fine", blocchi_fatti_ora=fatti, **stato_numeri(dir_c, piano))
    return 0


def stato_numeri(dir_c: Path, piano: dict) -> dict:
    sig = [json.loads((dir_c / b["nome"] / "blocco.json").read_text())
           for b in piano["blocchi"] if (dir_c / b["nome"] / "blocco.json").exists()]
    consumati = [b["nome"] for b in piano["blocchi"] if (dir_c / (b["nome"] + ".consumato")).exists()]
    tok_sig = sum(s["token"] for s in sig)
    tok_calc = sum(s.get("token_calcolati", s["token"]) for s in sig)      # velocita': solo i calcolati
    sec = sum(s["secondi"] for s in sig if s.get("token_calcolati", s["token"]))
    tot = sum((b["a"] - b["da"]) * b["L"] for b in piano["blocchi"])
    fatti_tok = tok_sig + sum((b["a"] - b["da"]) * b["L"] for b in piano["blocchi"] if b["nome"] in consumati)
    vel = tok_calc / sec if sec and tok_calc else None
    return {"blocchi_sigillati": len(sig), "blocchi_consumati": len(consumati), "blocchi_totali": len(piano["blocchi"]),
            "token_fatti": fatti_tok, "token_totali": tot, "token_calcolati": tok_calc, "secondi": round(sec, 1),
            "token_al_secondo": round(vel, 1) if vel else None,
            "eta_min": round((tot - fatti_tok) / vel / 60, 1) if vel else None}


# ═══════════════════════════════════════ lettura (per chi allena) ═══════════════════════════════════════
class CacheMaestro:
    """Lettore della cache del teacher. `h(lato, i)` -> torch.bfloat16 [L, 2048] (memmap: nessuna copia
    finche' non si porta sul dispositivo). Solo blocchi SIGILLATI. `modo` va confrontato con lo studente."""

    def __init__(self, dir_corsa):
        self.dir = Path(dir_corsa)
        self.piano = json.loads((self.dir / "piano.json").read_text())
        self.modo = self.piano["modo"]
        self.salva_tipo = self.piano["salva_tipo"]
        self._indice: dict = {}
        self.aggiorna()

    def aggiorna(self) -> int:
        """Rilegge i sigilli (nuovi blocchi arrivati mentre si allena). Ritorna le righe disponibili."""
        self._indice = {}
        for b in self.piano["blocchi"]:
            p = self.dir / b["nome"] / "blocco.json"
            if not p.exists():
                continue
            s = json.loads(p.read_text())
            for r in s["righe"]:
                self._indice[(b["lato"], r["indice"])] = (self.dir / b["nome"], r, s)
        return len(self._indice)

    def disponibile(self, lato: str, i: int) -> bool:
        return (lato, i) in self._indice

    def righe(self, lato: str) -> list:
        return sorted(i for (lt, i) in self._indice if lt == lato)

    def info(self, lato: str, i: int) -> dict:
        return self._indice[(lato, i)][1]

    def cu_seqlens(self, lato: str, i: int) -> list:
        r = self.info(lato, i)
        return r["segmenti_corpus"] + [self._indice[(lato, i)][2]["L"]]

    def h_numpy(self, lato: str, i: int) -> np.ndarray:
        d, r, _ = self._indice[(lato, i)]
        return leggi_tensore(d / r["file_h"])

    def h(self, lato: str, i: int):
        """torch [L, n_embd] nel tipo salvato, SOLA LETTURA (memmap del file: non scriverci; `.to(dispositivo)`
        o `.clone()` ne fanno una copia)."""
        import warnings
        torch = _torch()
        a = np.asarray(self.h_numpy(lato, i))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            t = torch.from_numpy(a)
        return t.view(torch.bfloat16) if self.salva_tipo == "bf16" else t

    def ids(self, lato: str, i: int) -> np.ndarray:
        d, r, s = self._indice[(lato, i)]
        k = next(j for j, x in enumerate(s["righe"]) if x["indice"] == i)
        return np.fromfile(d / "ids.i32", dtype=np.int32, count=s["L"], offset=4 * s["L"] * k)

    def verifica_blocco(self, nome: str) -> list:
        """Ricalcola gli sha256 di un blocco sigillato. Ritorna la lista dei file che non combaciano."""
        d = self.dir / nome
        s = json.loads((d / "blocco.json").read_text())
        male = [r["file_h"] for r in s["righe"] if sha256_file(d / r["file_h"]) != r["sha256_h"]]
        if sha256_file(d / "ids.i32") != s["sha256_ids_blocco"]:
            male.append("ids.i32")
        if s["manifesto_sha256"] is not None and sha256_file(d / "dump" / "manifesto.json") != s["manifesto_sha256"]:
            male.append("dump/manifesto.json")
        return male

    def consuma(self, nome: str) -> None:
        """Blocchi rotanti (§3.1): cancella un blocco gia' usato; il marcatore .consumato impedisce che
        `esegui` lo rifaccia."""
        d = self.dir / nome
        marc = self.dir / (nome + ".consumato")
        # 27/9 16:55: una ripresa il cui stato non elencava un blocco GIA' consumato (marcatore presente, cartella tolta)
        # faceva cadere la corsa EB2 su FileNotFoundError. Consumare e' idempotente: gia' consumato -> niente da fare.
        if not d.exists() and marc.exists():
            registra(self.dir, "consumato_gia", nome=nome)
            return
        s = json.loads((d / "blocco.json").read_text())
        scrivi_json_atomico(marc, {"consumato": adesso(), "sigillo": s["sigillato"], "token": s["token"]})
        shutil.rmtree(d)
        registra(self.dir, "consumato", nome=nome)
        self.aggiorna()


# ═══════════════════════════════════════ testa e logit ═══════════════════════════════════════
def _tensore_gguf(percorso: Path, nome: str):
    from gguf import GGUFReader
    r = GGUFReader(str(percorso))
    for t in r.tensors:
        if t.name == nome:
            return t
    raise KeyError(f"{nome} non in {percorso}")


def n_embd_gguf(percorso: Path) -> int:
    from gguf import GGUFReader
    r = GGUFReader(str(percorso))
    arch = r.fields["general.architecture"].contents()
    return int(r.fields[f"{arch}.embedding_length"].contents())


def dequant_q8_0(raw: np.ndarray) -> np.ndarray:
    """Q8_0 (blocchi da 34 byte: d fp16 + 32 int8) -> float32 ESATTO (11 + 8 bit < 24)."""
    raw = np.ascontiguousarray(raw, dtype=np.uint8)
    V = raw.shape[0]
    b = raw.reshape(V, -1, 34)
    d = b[:, :, :2].copy().view(np.float16).astype(np.float32)          # (V, nb, 1)
    q = b[:, :, 2:].view(np.int8).astype(np.float32)                    # (V, nb, 32)
    return (d * q).reshape(V, -1)


class TestaMaestro:
    """Testa del teacher per ricalcolare i logit dalla cache. `fonte`:
      - "bf16" (default, §2.3): output.weight del bf16 ORIGINALE, portata a fp32 (esatto);
      - "q8": output.weight del Q8_0 che ha prodotto h (per G0b: confronto col motore).
    W fp32 [V, 2048] (1,9 GiB). `sha256_dati` = sha256 dei byte del tensore nel GGUF."""

    def __init__(self, fonte: str = "bf16", gguf: Optional[Path] = None, dispositivo: str = "cpu"):
        torch = _torch()
        self.fonte = fonte
        self.gguf = Path(gguf) if gguf else (TESTA_BF16 if fonte == "bf16" else MODELLO_Q8)
        t = _tensore_gguf(self.gguf, "output.weight")
        raw = np.asarray(t.data)
        self.tipo = t.tensor_type.name
        self.sha256_dati = hashlib.sha256(raw.tobytes()).hexdigest()
        if self.tipo == "BF16":
            W = bf16_a_f32(raw.view(np.uint16).reshape(raw.shape[0], -1))
        elif self.tipo == "Q8_0":
            W = dequant_q8_0(raw)
        elif self.tipo == "F32":
            W = np.asarray(raw, np.float32)
        else:
            raise ValueError(f"testa {self.tipo}: non gestita")
        if W.ndim != 2:
            raise ValueError(f"testa di forma {W.shape}")
        self.W = torch.from_numpy(np.ascontiguousarray(W)).to(dispositivo)


def logit_maestro(hT, WT):
    """UNICA definizione dei logit del teacher (identica a `e2e_prototipo.logit_maestro`): h -> fp32,
    accumulo fp32. La usano G0b qui e la perdita (KLAPezzi) di chi allena."""
    return hT.float() @ WT.float().T


# ═══════════════════════════════════════ F0a ═══════════════════════════════════════
def _kl_righe(lp_a, lp_b):
    """KL(a || b) per riga, dai log-softmax (torch, float64)."""
    torch = _torch()
    pa = lp_a.exp()
    return (pa * (lp_a - lp_b)).sum(-1)


def f0a_cache(corsa: str, radice: Path = RADICE, righe: tuple = (0, 1), da: int = 3072, n_thread: int = 4,
              cpu: bool = False, teste: Optional[dict] = None, pezzo: int = 64) -> dict:
    """G0b (§2.6 F0a): per righe del held-out, logit DIRETTI di llama.cpp (Vulkan; `cpu` solo nei test) sulle
    posizioni >= da contro i logit RICALCOLATI con logit_maestro da (i) h f32 della stessa corsa, (ii) h bf16 =
    RNE(f32) (T13 dello strumento), (iii) la cache sigillata; con la testa Q8 (quella del motore) e con la bf16
    originale (quella della perdita, §2.3). In piu': la cache e' uguale al bit alla RNE della corsa nuova?
    (determinismo del motore fra corse con uscite diverse). Conti su CPU in float64, a pezzi di `pezzo`
    posizioni (26/9: la prima stesura teneva 1024 x 248.320 float64 x 7 tensori = 14 GB insieme; fermata
    prima di girare). Scrive <corsa>/f0a/cache.json."""
    torch = _torch()
    dir_c = radice / corsa
    piano = json.loads((dir_c / "piano.json").read_text())
    cm = CacheMaestro(dir_c)
    out = dir_c / "f0a" / "cache"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    L = piano["corpus"]["heldout"]["L"]
    ids = np.stack([cm.ids("heldout", i) for i in righe])
    ids.tofile(out / "ids.i32")
    cmd = [str(BIN), "logit"] + ([] if cpu else ["--vulkan"]) + [
        "-m", piano["modello"]["percorso"], "-o", str(out / "dump"),
        "-t", str(n_thread), "-c", str(L), "-b", str(L), "-ub", str(piano["ub"]), "--lunghezza", str(L),
        "--uscite", f"da:{da}", "--tensore", "h_nextn", "--salva-tipo", "f32"]
    if piano["modo"] == "segmenti":
        inizi = np.stack([np.isin(np.arange(L), cm.cu_seqlens("heldout", i)[:-1]).astype(np.int32) for i in righe])
        inizi.tofile(out / "inizi.i32")
        cmd += ["--segmenti", str(out / "inizi.i32")]
    cmd.append(str(out / "ids.i32"))
    if not cpu:
        subprocess.run([str(GUARDIA), "--attendi", "300"], check=True, capture_output=True)
    t0 = time.time()
    with open(out / "strumento.log", "w") as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT, env=ambiente_strumento(cpu))
    sec_motore = time.time() - t0
    u = Uscita(out / "dump")
    torch.set_num_threads(n_thread)
    if teste is None:
        teste = {"q8": TestaMaestro("q8"), "bf16": TestaMaestro("bf16")}
    ris = {"righe": list(righe), "posizioni": [da, L - 1], "modo": piano["modo"], "secondi_motore": round(sec_motore, 1),
           "per_riga": []}
    for k, i in enumerate(righe):
        pos, Ld = u.logit(k)
        h32 = torch.from_numpy(np.array(u.tensore(k, "h_nextn"), np.float32))
        hbf_rne = torch.from_numpy(bf16_a_f32(_rne_bf16(h32.numpy())))
        hcache = cm.h("heldout", i).float()
        r = {"riga": i, "cache_bf16_uguale_a_RNE_della_corsa_nuova": bool(torch.equal(hcache, hbf_rne)),
             "max_diff_cache_vs_rne": float((hcache - hbf_rne).abs().max()),
             "max_abs_h": float(h32.abs().max())}
        acc: dict = {}
        nll_s, nll_n = 0.0, 0
        for c0 in range(0, len(pos), pezzo):
            p_c = np.asarray(pos[c0:c0 + pezzo])
            d = torch.from_numpy(np.array(Ld[c0:c0 + pezzo], np.float32)).double()
            lp_d = torch.log_softmax(d, -1)
            for nome_h, hh in (("h_f32", h32), ("h_bf16", hbf_rne), ("h_cache", hcache)):
                for nome_t, tm in teste.items():
                    z = logit_maestro(hh[p_c], tm.W).double()
                    kl = _kl_righe(lp_d, torch.log_softmax(z, -1))
                    a_ = acc.setdefault(f"{nome_h}|testa_{nome_t}", {"kl": [], "maxd": 0.0, "arg": 0})
                    a_["kl"].append(kl)
                    a_["maxd"] = max(a_["maxd"], float((z - d).abs().max()))
                    a_["arg"] += int((z.argmax(-1) == d.argmax(-1)).sum())
            if "q8" in teste and "bf16" in teste:
                zq = torch.log_softmax(logit_maestro(h32[p_c], teste["q8"].W).double(), -1)
                zb = torch.log_softmax(logit_maestro(h32[p_c], teste["bf16"].W).double(), -1)
                acc.setdefault("testa", []).append(_kl_righe(zb, zq))
            ok = p_c + 1 < L                                   # posizioni con un token dopo
            nxt = torch.from_numpy(ids[k][p_c[ok] + 1].astype(np.int64))
            nll_s += float(-lp_d[torch.from_numpy(ok)].gather(1, nxt[:, None]).sum())
            nll_n += int(ok.sum())
        for chiave, a_ in acc.items():
            if chiave == "testa":
                r["kl_testa_bf16_vs_q8_media"] = float(torch.cat(a_).mean())
                continue
            kl = torch.cat(a_["kl"])
            r[chiave] = {"kl_diretti_vs_ricalcolati_media": float(kl.mean()), "kl_max": float(kl.max()),
                         "max_abs_diff_logit": a_["maxd"], "argmax_uguale": a_["arg"] / len(pos)}
        r["nll_media_diretta"] = nll_s / max(1, nll_n)
        ris["per_riga"].append(r)
    ris["teste"] = {k: {"gguf": str(v.gguf), "tipo": v.tipo, "sha256_dati": v.sha256_dati} for k, v in teste.items()}
    scrivi_json_atomico(dir_c / "f0a" / "cache.json", ris)
    for p in (out / "dump").rglob("logit.f32"):
        p.unlink()
    return ris


def _rne_bf16(x: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def f0a_ppl(corsa: str, radice: Path = RADICE, chunk_identita: int = 2) -> dict:
    """PPL wikitext-2 del Q8 con llama-perplexity su Vulkan (72 x 4096, protocollo del pavimento) contro il bf16
    7,7772 +- 0,0558 [stato/ppl_pavimento/ppl_bf16.log]; poi identita' strumento == llama-perplexity su Vulkan
    (--uscite meta, primi `chunk_identita` pezzi: stessa PPL stampata)."""
    dir_c = radice / corsa
    piano = json.loads((dir_c / "piano.json").read_text())
    out = dir_c / "f0a" / "ppl"
    out.mkdir(parents=True, exist_ok=True)
    M = piano["modello"]["percorso"]
    env = ambiente_strumento(False)
    subprocess.run([str(GUARDIA), "--attendi", "300"], check=True, capture_output=True)
    ris = {}
    for nome, extra in (("completa", []), ("identita", ["--chunks", str(chunk_identita)])):
        log = out / f"perplexity_{nome}.log"
        with open(log, "w") as f:
            subprocess.run([str(PPL_BIN), "-m", M, "-f", str(WIKI), "-c", "4096", "-b", "2048", "-ngl", "999", *extra],
                           check=True, stdout=f, stderr=subprocess.STDOUT, env=env)
        mm = re.search(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)", log.read_text())
        ris[f"llama_perplexity_{nome}"] = {"ppl": float(mm.group(1)), "errore": float(mm.group(2)), "testo": mm.group(1)}
    # strumento, stessi pezzi (id da `tokenizza`, identici a llama-perplexity: T1 del collaudo)
    subprocess.run([str(BIN), "tokenizza", M, str(WIKI), "4096", str(out / "ids"), str(chunk_identita)],
                   check=True, capture_output=True, env=env)
    seq = sorted((out / "ids").glob("seq_*.i32"))
    d = out / "strumento"
    if d.exists():
        shutil.rmtree(d)
    with open(out / "strumento.log", "w") as f:
        subprocess.run([str(BIN), "logit", "--vulkan", "-m", M, "-o", str(d), "-t", "4", "-c", "4096", "-b", "2048",
                        "--uscite", "meta", *map(str, seq)], check=True, stdout=f, stderr=subprocess.STDOUT, env=env)
    man = json.loads((d / "manifesto.json").read_text())
    nll = sum(s["nll"]["somma_stile_perplexity"] for s in man["sequenze"])
    nn = sum(s["nll"]["n_stile_perplexity"] for s in man["sequenze"])
    ppl_s = float(np.exp(nll / nn))
    ris["strumento_identita"] = {"ppl": ppl_s, "testo_4_decimali": f"{ppl_s:.4f}"}
    ris["identita_4_decimali"] = f"{ppl_s:.4f}" == ris["llama_perplexity_identita"]["testo"]
    bf, sbf = 7.7772, 0.05581
    q = ris["llama_perplexity_completa"]["ppl"]
    ris["pavimento_bf16"] = {"ppl": bf, "errore": sbf, "fonte": "stato/ppl_pavimento/ppl_bf16.log"}
    ris["q8_meno_bf16_in_sigma"] = round((q - bf) / sbf, 3)
    for p in d.rglob("logit.f32"):
        p.unlink()
    scrivi_json_atomico(dir_c / "f0a" / "ppl.json", ris)
    return ris


# 26/9: F0a, secondo criterio (§2.6): la KL fra il teacher VERO (bf16) e il suo sostituto (Q8_0, che produce la
# cache) deve stare sotto 1/20 della KL(teacher || studente). Qui si misura il primo termine; il rapporto lo fa
# chi misura lo studente (e2e_fedelta). Motore: llama-perplexity della build k1024 (identita' al byte con lo
# strumento: T2 del collaudo), su Vulkan. Il bf16 pesa 67 GB (62,4 GiB): con la casa accesa (~31 GiB) NON sta
# nei 95,15 GiB utili, quindi `f0a_kl` ASPETTA che la VRAM lo consenta (casa giu', decisione nel §5).
BF16_MODELLO = Path("@@ROOT:models@@/gguf/thor-src/Ornith-1.5-35B-A3B-bf16/Ornith-1.5-35B-A3B-bf16-00001-of-00002.gguf")
VRAM_BF16_MIB = 66 * 1024      # STIMA: 62,4 GiB di pesi + calcolo a -b 2048 + margine; il picco vero si registra
_KLD_RE = {
    "ppl_q": r"Mean PPL\(Q\)\s*:\s*([-0-9.eE+]+) ±\s*([-0-9.eE+]+)",
    "ppl_base": r"Mean PPL\(base\)\s*:\s*([-0-9.eE+]+) ±\s*([-0-9.eE+]+)",
    "ln_ppl_q_su_base": r"Mean ln\(PPL\(Q\)/PPL\(base\)\)\s*:\s*([-0-9.eE+]+) ±\s*([-0-9.eE+]+)",
    "kld_media": r"Mean    KLD:\s*([-0-9.eE+]+) ±\s*([-0-9.eE+]+)",
    "kld_max": r"Maximum KLD:\s*([-0-9.eE+]+)",
    "kld_p999": r"99\.9%   KLD:\s*([-0-9.eE+]+)",
    "kld_p99": r"99\.0%   KLD:\s*([-0-9.eE+]+)",
    "kld_mediana": r"Median  KLD:\s*([-0-9.eE+]+)",
    "rms_dp_pc": r"RMS Δp\s*:\s*([-0-9.eE+]+) ±\s*([-0-9.eE+]+) %",
    "stesso_top_pc": r"Same top p:\s*([-0-9.eE+]+) ±\s*([-0-9.eE+]+) %",
}


def leggi_kld(testo: str) -> dict:
    """Statistiche di `llama-perplexity --kl-divergence` dal suo log. Solleva se manca la KL media."""
    r = {}
    for k, pat in _KLD_RE.items():
        m = re.search(pat, testo)
        if m:
            r[k] = [float(x) for x in m.groups()] if len(m.groups()) > 1 else float(m.group(1))
    if "kld_media" not in r:
        raise RuntimeError("log di llama-perplexity senza «Mean    KLD»: corsa KL non completata")
    return r


def _righe_per_dominio(cm: "CacheMaestro", k: int) -> list:
    """Le prime k righe di ogni dominio del held-out, nell'ordine d'uso (indici del lato)."""
    visti: dict = {}
    scelte = []
    for i in cm.righe("heldout"):
        dom = cm.info("heldout", i)["dominio"]
        if visti.get(dom, 0) < k:
            visti[dom] = visti.get(dom, 0) + 1
            scelte.append(i)
    return scelte


def f0a_kl(corsa: str, radice: Path = RADICE, chunks_wiki: int = 8, righe_per_dominio: int = 2,
           righe_per_chiamata: int = 7, modello_base: Path = BF16_MODELLO, wiki: Path = WIKI, ctx_wiki: int = 4096,
           cpu: bool = False, n_thread: int = 4, tieni_basi: bool = True, vram_base_mib: int = VRAM_BF16_MIB) -> dict:
    """KL(teacher bf16 || teacher Q8) con llama-perplexity: (1) wikitext, primi `chunks_wiki` pezzi da `ctx_wiki`;
    (2) held-out E2E, `righe_per_dominio` righe per dominio, base scritta da e2e_dump_logit --kld-base (formato
    al byte di llama-perplexity, T2) in gruppi di `righe_per_chiamata` righe (i logit f32 temporanei, ~2 GB a
    riga, si cancellano dopo ogni gruppo). Le posizioni sono quelle del protocollo (>= n/2). Il held-out va
    in modo RIGA anche per una corsa `segmenti` (llama-perplexity non ha segmenti): la KL confronta due modelli
    sullo stesso contesto, e il modo si scrive nel risultato. Le basi (1 GB a riga, bf16 = il teacher vero)
    servono poi alla KL studente-teacher in llama.cpp (F0c, verdetto): `tieni_basi`. Scrive <corsa>/f0a/kl.json."""
    dir_c = radice / corsa
    piano = json.loads((dir_c / "piano.json").read_text())
    cm = CacheMaestro(dir_c)
    out = dir_c / "f0a" / "kl"
    out.mkdir(parents=True, exist_ok=True)
    M_q = piano["modello"]["percorso"]
    env = ambiente_strumento(cpu)
    dev = ["-ngl", "0"] if cpu else ["-ngl", "999"]
    L = piano["corpus"]["heldout"]["L"]
    ris = {"versione": VERSIONE, "corsa": corsa, "modello_base": str(modello_base), "modello_q": M_q,
           "byte_base": sum(p.stat().st_size for p in modello_base.parent.glob(modello_base.name.split("-00001-of-")[0] + "*.gguf"))
           if "-00001-of-" in modello_base.name else modello_base.stat().st_size,
           "modo_heldout": "riga", "modo_corsa": piano["modo"], "tempi_s": {}, "vram_picco_MiB": {}, "gtt_picco_MiB": {}}

    def _gpu_pronta(serve_mib: int):
        if cpu:
            return
        def _c():
            motivo = pausa_dichiarata(radice)
            if motivo:
                return False, f"PAUSA dichiarata ({radice / 'PAUSA'}): {motivo[:160]}"
            p = subprocess.run([str(GUARDIA)], capture_output=True, text=True)
            v = vram_mib()
            return (p.returncode == 0 and v + serve_mib <= VRAM_UTILE_MIB,
                    f"guardia rc={p.returncode} {p.stdout.strip()[:160]}; VRAM {v} MiB + {serve_mib}")
        _attendi(_c, f"GPU per {serve_mib} MiB", dir_c)

    def _corri(nome: str, cmd: list, serve_mib: int) -> str:
        with LucchettoGPU(radice, attivo=not cpu):
            _gpu_pronta(serve_mib)
            pic = Picchi()
            pic.start()
            t0 = time.time()
            log = out / f"{nome}.log"
            try:
                with open(log, "w") as f:
                    rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
            finally:
                pic.ferma()
        ris["tempi_s"][nome] = round(time.time() - t0, 1)
        ris["vram_picco_MiB"][nome], ris["gtt_picco_MiB"][nome] = pic.vram, pic.gtt
        if rc != 0:
            raise RuntimeError(f"{nome}: rc={rc}\n{log.read_text()[-1500:]}")
        return log.read_text()

    # (1) wikitext
    bw = out / "base_wiki_bf16.kld"
    t = _corri("wiki_base_bf16", [str(PPL_BIN), "-m", str(modello_base), "-f", str(wiki), "-c", str(ctx_wiki),
                                  "-b", "2048", "--chunks", str(chunks_wiki), "-t", str(n_thread), *dev,
                                  "--kl-divergence-base", str(bw)], vram_base_mib)
    m = re.search(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)", t)
    ris["wiki"] = {"chunks": chunks_wiki, "ctx": ctx_wiki, "ppl_base": [float(m.group(1)), float(m.group(2))] if m else None}
    t = _corri("wiki_kl_q", [str(PPL_BIN), "-m", M_q, "-c", str(ctx_wiki), "-b", "2048", "-t", str(n_thread), *dev,
                             "--kl-divergence-base", str(bw), "--kl-divergence"], VRAM_PROCESSO_MIB)
    ris["wiki"].update(leggi_kld(t))
    # (2) held-out E2E
    scelte = _righe_per_dominio(cm, righe_per_dominio)
    ris["heldout"] = {"L": L, "righe": [[i, cm.info("heldout", i)["dominio"], cm.info("heldout", i)["riga"]] for i in scelte],
                      "gruppi": []}
    for g, a in enumerate(range(0, len(scelte), righe_per_chiamata)):
        gr = scelte[a:a + righe_per_chiamata]
        ids = np.concatenate([cm.ids("heldout", i) for i in gr]).astype(np.int32)
        f_ids = out / f"ho_{g}.i32"
        ids.tofile(f_ids)
        bh = out / f"base_heldout_bf16_{g}.kld"
        d = out / f"ho_{g}_dump"
        if d.exists():
            shutil.rmtree(d)
        try:
            _corri(f"heldout{g}_base_bf16", [str(BIN), "logit", *([] if cpu else ["--vulkan"]), "-m", str(modello_base),
                                            "-o", str(d), "-t", str(n_thread), "-c", str(L), "-b", str(min(2048, L)),
                                            "--lunghezza", str(L), "--uscite", "meta", "--kld-base", str(bh), str(f_ids)],
                   vram_base_mib)
        finally:
            for p in d.rglob("logit.f32") if d.exists() else []:
                p.unlink()                                   # ~2 GB a riga: via subito
        t = _corri(f"heldout{g}_kl_q", [str(PPL_BIN), "-m", M_q, "-c", str(L), "-b", str(min(2048, L)), "-t", str(n_thread),
                                       *dev, "--kl-divergence-base", str(bh), "--kl-divergence"], VRAM_PROCESSO_MIB)
        s = leggi_kld(t)
        s.update(righe=gr, base=str(bh), byte_base=bh.stat().st_size)
        ris["heldout"]["gruppi"].append(s)
    # media sulle posizioni di tutti i gruppi (stesso numero di posizioni a riga: pesi = righe del gruppo)
    gs = ris["heldout"]["gruppi"]
    n = np.array([len(x["righe"]) for x in gs], float)
    mu = np.array([x["kld_media"][0] for x in gs])
    se = np.array([x["kld_media"][1] for x in gs])
    ris["heldout"]["kld_media"] = [float((n * mu).sum() / n.sum()), float(np.sqrt(((n * se) ** 2).sum()) / n.sum())]
    ris["heldout"]["kld_max"] = max(x["kld_max"] for x in gs)
    ris["soglia"] = {"regola": "KL(bf16||Q8) <= KL(teacher||studente) / 20 (progetto §2.6, F0a)",
                     "kl_studente_minima_wiki": 20 * ris["wiki"]["kld_media"][0],
                     "kl_studente_minima_heldout": 20 * ris["heldout"]["kld_media"][0]}
    if not tieni_basi:
        for p in [bw] + [Path(x["base"]) for x in gs]:
            p.unlink()
    ris["basi_tenute"] = tieni_basi
    scrivi_json_atomico(dir_c / "f0a" / "kl.json", ris)
    return ris


# ═══════════════════════════════════════ unita' systemd ═══════════════════════════════════════
# 26/9: MemoryMax 12G (era 8G): F0a carica le due teste (Q8 e bf16 in fp32), picco RSS MISURATO 6,48 GiB + pezzi.
def _unita_attiva(nome: str) -> bool:
    env = dict(os.environ, XDG_RUNTIME_DIR="/run/user/1000", DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/1000/bus")
    r = subprocess.run(["systemctl", "--user", "is-active", nome], capture_output=True, text=True, env=env)
    return r.stdout.strip() in ("active", "activating", "deactivating", "reloading")


def unita_teacher_attive() -> list:
    """26/9: tutte le unita' systemd --user «e2e-teacher*» attive (la coda principale e quelle accodate)."""
    env = dict(os.environ, XDG_RUNTIME_DIR="/run/user/1000", DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/1000/bus")
    r = subprocess.run(["systemctl", "--user", "list-units", f"{UNITA}*", "--plain", "--no-legend", "--all"],
                       capture_output=True, text=True, env=env)
    nomi = [x.split()[0].removesuffix(".service") for x in r.stdout.splitlines() if x.strip()]
    return [n for n in nomi if _unita_attiva(n)]


def coda(corsa: str, dopo: list, radice: Path = RADICE, f0a: bool = True, aspetta_unita: Optional[str] = None) -> int:
    """La coda GPU dell'unita' (una corsia per volta): `esegui` della corsa principale, poi (se verde) F0a
    cache e F0a PPL sulla stessa corsa, poi `esegui` delle corse in `dopo`. Un F0a che fallisce si registra
    e non ferma la coda; un `esegui` che fallisce la ferma. `aspetta_unita` (26/9): prima si aspetta che
    quell'unita' systemd --user sia finita (una coda accodata dietro a quella in corso)."""
    dir_c = radice / corsa
    if aspetta_unita:                       # 26/9: una o piu' unita', separate da virgole: si aspettano TUTTE
        attese = [u for u in aspetta_unita.split(",") if u]
        _attendi(lambda: (not any(_unita_attiva(u) for u in attese),
                          "ancora attive: " + ",".join(u for u in attese if _unita_attiva(u))),
                 f"unita' {aspetta_unita}", dir_c)
    rc = esegui(corsa, radice)
    if rc != 0:
        return rc
    if f0a:
        for nome, fn in (("f0a_cache", f0a_cache), ("f0a_ppl", f0a_ppl)):
            t0 = time.time()
            try:
                with LucchettoGPU(radice):
                    fn(corsa, radice)
                registra(dir_c, nome, esito="ok", secondi=round(time.time() - t0, 1),
                         file=str(dir_c / "f0a" / (nome.split("_")[1] + ".json")))
            except Exception as e:  # noqa: BLE001 — registrato, la coda continua
                registra(dir_c, nome, esito="errore", errore=str(e)[:1500])
    for altra in dopo:
        rc = esegui(altra, radice)
        if rc != 0:
            return rc
    return 0


def lancia(corsa: str, python: str = sys.executable, extra: Optional[list] = None) -> None:
    """Unita' `e2e-teacher`; se una qualunque `e2e-teacher*` e' attiva (26/9) la corsa si ACCODA in
    `e2e-teacher-<corsa>`, che aspetta la fine di TUTTE quelle attive al lancio (e poi il lucchetto GPU): mai due
    teacher in VRAM insieme, e le righe di una corsa precedente sono gia' sigillate quando il riuso le cerca."""
    env = dict(os.environ, XDG_RUNTIME_DIR="/run/user/1000", DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/1000/bus")
    unita = UNITA
    extra = list(extra or [])
    attive = unita_teacher_attive()         # 26/9: TUTTE le e2e-teacher* attive, non solo la principale
    if attive:
        unita = f"{UNITA}-{re.sub(r'[^A-Za-z0-9_-]', '_', corsa)}"
        if unita in attive:
            raise RuntimeError(f"{unita} e' gia' attiva")
        extra += ["--aspetta-unita", ",".join(attive)]
    subprocess.run(["systemctl", "--user", "reset-failed", unita], capture_output=True, env=env)
    log = RADICE / corsa / "esegui.log"
    cmd = ["systemd-run", "--user", f"--unit={unita}", "--description=Thor E2E: teacher (h bf16) su Vulkan",
           "-p", "CPUQuota=400%", "-p", "MemoryMax=12G", "-p", "MemorySwapMax=512M", "-p", "Nice=5",
           f"--working-directory={QUI}", "-p", f"StandardOutput=append:{log}", "-p", f"StandardError=append:{log}",
           "--setenv=PYTHONUNBUFFERED=1", python, str(QUI / "e2e_teacher.py"), "coda", "--corsa", corsa,
           *(extra or [])]
    subprocess.run(cmd, check=True, env=env)
    print(f"lanciata {unita}: log {log} ; registro {RADICE / corsa / 'registro.jsonl'}")


# ═══════════════════════════════════════ riga di comando ═══════════════════════════════════════
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("comando", choices=["prepara", "esegui", "coda", "lancia", "stato", "verifica", "f0a-cache",
                                        "f0a-ppl", "f0a-kl", "riuso"])
    ap.add_argument("--chunks-wiki", type=int, default=8, help="f0a-kl: pezzi da 4096 di wikitext")
    ap.add_argument("--righe-per-dominio", type=int, default=2, help="f0a-kl: righe del held-out per dominio")
    ap.add_argument("--senza-basi", action="store_true", help="f0a-kl: cancella le basi KLD bf16 alla fine")
    ap.add_argument("--corsa", required=True)
    ap.add_argument("--radice", type=Path, default=RADICE)
    ap.add_argument("--corpus-dir", type=Path, default=None,
                    help=f"prepara: cartella del corpus (default {CORPUS}); esegui: default = quella del piano")
    ap.add_argument("--variante", default="tutte")
    ap.add_argument("--lati", default="heldout,train,pannelli_1024,pannelli_4096")
    ap.add_argument("--solo", default=None, help="esegui: solo questi lati (virgole)")
    ap.add_argument("--modo", default="riga", choices=["riga", "segmenti"])
    ap.add_argument("--righe-blocco", type=int, default=128)
    ap.add_argument("--ub", type=int, default=1024)
    ap.add_argument("--salva-tipo", default="bf16", choices=["bf16", "f16", "f32"])
    ap.add_argument("--modello", type=Path, default=MODELLO_Q8)
    ap.add_argument("--sha-modello", default=None, help="prepara: sha256 gia' noto (solo test)")
    ap.add_argument("--cpu", action="store_true", help="esegui senza Vulkan (SOLO test, modello minuscolo)")
    ap.add_argument("--margine-gib", type=float, default=6.0)
    ap.add_argument("--dopo", default="", help="coda/lancia: corse da eseguire dopo (virgole)")
    ap.add_argument("--senza-f0a", action="store_true", help="coda/lancia: niente F0a dopo la corsa")
    ap.add_argument("--aspetta-unita", default=None, help="coda: aspetta la fine di questa unita' systemd --user")
    a = ap.parse_args(argv)
    if a.comando == "prepara":
        p = prepara(a.corsa, a.lati.split(","), a.variante, a.modo, a.righe_blocco, a.ub, a.salva_tipo, a.modello,
                    a.radice, a.corpus_dir or CORPUS, a.sha_modello)
        pi = json.loads(p.read_text())
        print(json.dumps({"piano": str(p), "blocchi": len(pi["blocchi"]), "corpus": pi["corpus"]}, indent=1))
        return 0
    if a.comando == "esegui":
        return esegui(a.corsa, a.radice, a.cpu, a.margine_gib, corpus_dir=a.corpus_dir,
                      solo=a.solo.split(",") if a.solo else None)
    if a.comando == "coda":
        return coda(a.corsa, [x for x in a.dopo.split(",") if x], a.radice, not a.senza_f0a, a.aspetta_unita)
    if a.comando == "lancia":
        extra = (["--dopo", a.dopo] if a.dopo else []) + (["--senza-f0a"] if a.senza_f0a else [])
        lancia(a.corsa, extra=extra)
        return 0
    dir_c = a.radice / a.corsa
    piano = json.loads((dir_c / "piano.json").read_text())
    if a.comando == "stato":
        print(json.dumps({**stato_numeri(dir_c, piano), "libero_gib": round(libero_gib(dir_c), 1),
                          "modo": piano["modo"]}, indent=1))
        return 0
    if a.comando == "verifica":
        cm = CacheMaestro(dir_c)
        male = {b["nome"]: cm.verifica_blocco(b["nome"]) for b in piano["blocchi"]
                if (dir_c / b["nome"] / "blocco.json").exists()}
        male = {k: v for k, v in male.items() if v}
        print(json.dumps({"blocchi_verificati": len(male) == 0, "difformi": male}, indent=1))
        return 0 if not male else 1
    if a.comando == "f0a-cache":
        with LucchettoGPU(a.radice):
            r = f0a_cache(a.corsa, a.radice)
        print(json.dumps(r, indent=1))
        return 0
    if a.comando == "f0a-ppl":
        with LucchettoGPU(a.radice):
            r = f0a_ppl(a.corsa, a.radice)
        print(json.dumps(r, indent=1))
        return 0
    if a.comando == "riuso":
        print(json.dumps(previsione_riuso(a.corsa, a.radice), indent=1))
        return 0
    if a.comando == "f0a-kl":
        print(json.dumps(f0a_kl(a.corsa, a.radice, a.chunks_wiki, a.righe_per_dominio, tieni_basi=not a.senza_basi),
                         indent=1, default=str))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())

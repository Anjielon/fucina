#!/usr/bin/env python3
"""CATTURA SchurQuant PER-ESPERTO A SCALA — braccio B, prefisso quantizzato.

⭐ Perche' esiste (22/9/2026 18:05). Il controllo cross-strato ha misurato che
l'obiettivo teacher-reconstruction (W*, G) batte la ricetta attuale di
**-68,9% ± 5,6%** sull'errore d'uscita held-out, **12,3σ**, vincendo **16/16**
sugli esperti di 6 strati. Ma quel risultato vive in `schurquant_esperto_reale.py`,
che lavora su **UN esperto per volta**. L'infrastruttura a scala che esiste
(`cattura_hessiane_gateup_exps_glob.py`, 40 strati, corpus 2,48M campioni) e'
costruita sul flusso **PULITO** X_ref — cioe' il braccio **E**, non **B**.
Misurato su L10: E = -0,444 · B = -0,842. Il salto E→B E' l'obiettivo teacher,
ed e' esattamente cio' che manca. Questo file lo colma.

⛔ IL PUNTO DEL METODO, in una riga: il router vede in produzione lo stream
QUANTIZZATO, non quello pulito. Quindi i token che arrivano davvero a un
esperto si decidono sul flusso quantizzato, e per quegli STESSI token si
prende l'ingresso pulito come bersaglio del maestro.

Cosa accumula, per ogni strato e ogni esperto (Eq.8, convenzione token-righe):
    G_raw[e] += X_q_eᵀ diag(w) X_q_e        (in, in)
    M_raw[e] += X_ref_eᵀ diag(w) X_q_e      (in, in)
    peso[e]  += Σ w
dove w sono i PESI VERI di instradamento (non 1: lezione pagata il 22/9 —
omega=ones buttava via l'informazione del router).

A forgia: om = w/peso ⇒  G = G_raw/peso · C = W_ref @ (M_raw/peso)
poi  G_d = smorza(G + λI) ·  C_λ = C + λ W_ref  ·  W* = w_star(G_d, C_λ).

⭐ λ NON si applica qui: resta libero a forgia, cosi' si puo' spazzolare senza
ricatturare (il controllo ha scelto λ_rel=0,3 sul cal e riportato su held-out).

⛔ `const` di Eq.9 NON si accumula di proposito: e' un offset costante nella
loss e **non entra in W* = C G⁻¹**. Servirebbe un terzo accumulatore da 4,29 GB
per un numero che non cambia una sola scelta. Se un giorno servira' per
riportare la loss, si ricava a forgia con gli stessi X.

VRAM: due accumulatori (256, 2048, 2048) f32 = 8,6 GiB per strato. Gli stati
nascosti dei due flussi restano su CPU e salgono un lotto per volta — senza
questo sarebbero ~20 GiB e il lavoro non entrerebbe in una 24 GB.

Uscita per strato L (in FINESTRA_SCHUR_OUT):
    Gq_{L:02d}.npy   (256, in, in) f32
    Mqr_{L:02d}.npy  (256, in, in) f32
    peso_{L:02d}.npy (256,) f64  ·  conta_{L:02d}.npy (256,) i64
    fatto_{L:02d}.ok ← marca scritta SOLO a salvataggio completo (ripresa)
"""
from __future__ import annotations

import gc
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file as _st_load_file
from safetensors.torch import save_file as _st_save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getenv("GGUF_PY_DIR", "@@ROOT:models@@/build/llamacpp-tq1-k1024/gguf-py"))

from transformers import AutoConfig, AutoTokenizer  # noqa: E402
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (  # noqa: E402
    Qwen3_5MoeTextRotaryEmbedding,
)
from transformers.masking_utils import create_causal_mask  # noqa: E402

from carica_strato_gguf import pesi_strato_da_gguf  # noqa: E402
from moe_routing import token_per_esperto  # noqa: E402
from gptq_1024_lib import GROUP  # noqa: E402

# ⛔ percorsi SOLO da ambiente: questo file viaggia fra casa e pod
SRC = Path(os.getenv("THOR_HF_DIR", "@@ROOT:models@@/gguf/thor-src-hf"))
GGUF = os.getenv("THOR_GPTQ_BASE", "@@ROOT:models@@/gguf/tony-tern/Thor-v20-downaggr3.gguf")
CORPUS = Path(os.getenv("THOR_CORPUS", "@@ROOT:forge@@/thor/cot_calib/corpus_cot.txt"))
OUT = Path(os.getenv("FINESTRA_SCHUR_OUT", "@@ROOT:models@@/gguf/tony-tern/schur-esperti"))
L_SEQ = int(os.getenv("THOR_L_SEQ", "2048"))
N_SEQ = int(os.getenv("THOR_N_SEQ", "0"))          # 0 = tutto il corpus
BATCH_SEQ = int(os.getenv("THOR_BATCH_SEQ", "8"))
TETTO_GIB = float(os.getenv("THOR_TETTO_GIB", "90"))
STRATI = os.getenv("THOR_STRATI", "")               # "" = tutti; "0,1,2" o "0-39"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def vram_gib() -> float:
    return torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0


def controlla_tetto(dove: str) -> float:
    v = torch.cuda.memory_reserved() / 2**30 if torch.cuda.is_available() else 0.0
    if v > TETTO_GIB:
        raise RuntimeError(f"tetto VRAM superato a «{dove}»: {v:.1f} > {TETTO_GIB} GiB")
    return v


def elenco_strati(n_tot: int) -> list[int]:
    if not STRATI:
        return list(range(n_tot))
    fuori = []
    for pezzo in STRATI.split(","):
        pezzo = pezzo.strip()
        if "-" in pezzo:
            a, b = pezzo.split("-")
            fuori.extend(range(int(a), int(b) + 1))
        elif pezzo:
            fuori.append(int(pezzo))
    return [L for L in fuori if 0 <= L < n_tot]


MARGINE_GIB = float(os.getenv("THOR_MARGINE_DISCO_GIB", "40"))
QUOTA_VOLUME_GIB = float(os.getenv("THOR_QUOTA_VOLUME_GIB", "200"))  # thor-finestra, dichiarata alla creazione
GIB_PER_STRATO = (256*1024*2048*2 + 256*2048*2048*4 + 256*1024*2048*4) / 2**30  # W* f16 + G f32 + C f32


def _du_gib(p: Path) -> float:
    """Byte REALI occupati (somma ricorsiva), non lo spazio 'libero' del
    filesystem. ⛔⛔ Verificato il 22/9 DOPO aver scritto una guardia sbagliata:
    `os.statvfs('/workspace')` su questo volume FUSE/MooseFS ritorna la
    capacita' dell'intero CLUSTER (misurato: 456.976 GiB liberi), non la
    quota di 200 GiB assegnata a QUESTO volume — quella quota e' imposta un
    livello sopra (RunPod), invisibile a una chiamata POSIX dentro il
    container. Una guardia su statvfs non avrebbe MAI scattato: teatro, non
    protezione. Qui si somma cio' che c'e' DAVVERO, file per file."""
    tot = 0
    for f in p.rglob("*"):
        if f.is_file():
            tot += f.stat().st_size
    return tot / 2**30


def controlla_spazio(L: int, dove: str):
    """⛔⛔ 22/9 18:30-18:40, DUE guasti in sequenza, entrambi chiusi qui:
    (1) (256,2048,2048) f32 x2 (G+M) = ~8,59 GiB/strato, MAI scartati — su 40
        strati ~344 GiB contro un volume CONDIVISO da 200 GiB: saturato a
        strato 9, ha ucciso IN SILENZIO questa cattura E SliderQuant
        sull'ALTRO pod (nessun traceback, il log si ferma e basta).
    (2) la prima guardia (statvfs) era un placebo — vedi _du_gib sopra.
    Questa versione misura l'uso REALE una volta all'avvio (le cartelle
    statiche non crescono da questo processo) + il GIA' SCRITTO da questa
    cattura, e stima il PROSSIMO strato con la costante esatta — mai una
    misura ottimistica. ⚠️ Con l'accumulo attuale (G+M mai scartati) il
    margine vero regge SOLO ~6-7 strati su 40: questo NON e' un limite della
    guardia, e' il limite del disegno. La cura vera (forgiare e scartare per
    strato) resta da fare — questa guardia impedisce che il limite si scopra
    con un altro crash muto invece che con un rifiuto onesto."""
    global _BASE_ALTRO_GIB
    if _BASE_ALTRO_GIB is None:
        altro = [p for p in OUT.parent.iterdir() if p.is_dir() and p != OUT] if OUT.parent.exists() else []
        _BASE_ALTRO_GIB = sum(_du_gib(p) for p in altro)
        log(f"base non-mia sul volume (misurata una volta, file per file): {_BASE_ALTRO_GIB:.1f} GiB")
    mio = _du_gib(OUT)
    dopo = _BASE_ALTRO_GIB + mio + GIB_PER_STRATO
    log(f"  spazio: base_altro={_BASE_ALTRO_GIB:.1f} + mio={mio:.1f} + prossimo={GIB_PER_STRATO:.1f} "
        f"= {dopo:.1f} GiB su quota {QUOTA_VOLUME_GIB:.0f} (margine {MARGINE_GIB:.0f})")
    if dopo > QUOTA_VOLUME_GIB - MARGINE_GIB:
        raise RuntimeError(
            f"⛔ QUOTA VOLUME A RISCHIO prima dello strato {L} ({dove}): scrivere il prossimo "
            f"strato porterebbe l'uso a {dopo:.1f} GiB, oltre {QUOTA_VOLUME_GIB - MARGINE_GIB:.0f} "
            f"(quota {QUOTA_VOLUME_GIB:.0f} - margine {MARGINE_GIB:.0f}, condiviso con un altro pod). "
            f"Fermo qui con {L} strati SALVATI E BUONI, invece di ripetere il crash muto. "
            f"La cura vera e' scartare G/M dopo la forgia per-strato, non ancora fatta.")


_BASE_ALTRO_GIB: float | None = None


def precondizioni():
    """⛔ Muore con rc!=0 PRIMA di qualunque lavoro se una radice manca.
    Lezione pagata il 22/9: un `return` silenzioso su radice mancante fa
    scrivere al driver la marca di ripresa su lavoro MAI eseguito."""
    guasti = []
    for nome, p in (("THOR_HF_DIR", SRC), ("THOR_GPTQ_BASE", Path(GGUF)), ("THOR_CORPUS", CORPUS)):
        if not p.exists():
            guasti.append(f"{nome}={p} NON ESISTE")
    if not torch.cuda.is_available():
        guasti.append("nessuna GPU: questa cattura vive in VRAM")
    if guasti:
        for g in guasti:
            log(f"⛔ PRECONDIZIONE VIOLATA: {g}")
        raise SystemExit(4)


def _pesi_gguf_con_cache(gguf: str, L: int) -> dict:
    """pesi_strato_da_gguf con cache su disco (THOR_CACHE_STRATI), identica al bit.

    ⭐ 24/9 20:0x (mandato forgia < 2 ore): la dequantizzazione numpy dei 256
    esperti costava ~27 s/strato a OGNI forgia. La chiave include l'impronta del
    codice di conversione: una correzione (es. B9, A_log) invalida la cache."""
    cartella = os.getenv("THOR_CACHE_STRATI", "")
    if not cartella:
        return pesi_strato_da_gguf(gguf, L)
    import hashlib
    import carica_strato_gguf as _csg
    qui = Path(_csg.__file__).resolve().parent
    impronta = hashlib.sha256()
    for nome in ("carica_strato_gguf.py", "dequant_gguf_torch.py",
                 "mappa_completa_gguf_hf.json", "mappa_completa_gguf_hf_perm.npz"):
        impronta.update((qui / nome).read_bytes())
    st = os.stat(gguf)
    impronta.update(f"|{os.path.realpath(gguf)}|{st.st_size}|{st.st_mtime_ns}".encode())
    f = Path(cartella) / f"{Path(gguf).stem}-{impronta.hexdigest()[:16]}" / f"strato_{L:02d}.safetensors"
    if f.exists():
        return dict(_st_load_file(str(f)))
    sd = pesi_strato_da_gguf(gguf, L)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f.name + ".tmp")
    # bf16: identico al bit al percorso senza cache, perche' costruisci_strato porta
    # comunque TUTTO in bf16 alla fine (strato.to(bf16)) e l'arrotondamento fp32->bf16
    # e' uno solo in entrambi i casi (bf16->fp32->bf16 e' esatto). Meta' del disco.
    _st_save_file({k: v.to(torch.bfloat16).contiguous() for k, v in sd.items()}, str(tmp))
    os.replace(tmp, f)                       # atomico: mai un file a meta' letto come buono
    return sd


def costruisci_strato(cfg, L: int, da_gguf: bool):
    """Un Qwen3_5MoeDecoderLayer con i pesi bf16 veri (da_gguf=False) oppure
    con attenzione/SSM DEQUANTIZZATI dal GGUF di produzione (da_gguf=True).

    ⛔ Il secondo e' la configurazione ESATTA che ha prodotto il -68,9%
    verificato in `schurquant_esperto_reale.py` (passo 2: «attenzione/SSM
    caricati DAL GGUF attuale, corr 0.982 col vero — isola l'effetto della
    quantizzazione dell'attenzione sull'ingresso che il router vede DAVVERO
    in produzione»). Non si cambia senza rimisurare: e' la variabile che
    definisce il braccio B."""
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer
    from safetensors import safe_open
    from accelerate import init_empty_weights
    import json as _json

    # ⭐ 23/9 22:4x — Angelo: "tutto su GPU"/mandato leve. Un agente avversario
    # ha smontato il mio scarto "trascurabile" (0,03% di tempo, VERO) per una
    # ragione diversa: l'init random alloca per un istante l'INTERO layer in
    # f32 su CPU (alcuni GiB) prima che load_state_dict(assign=True) lo scarti
    # subito — un picco RAM inutile su un file che ha GIA' avuto due incidenti
    # di swap-thrashing (17/9, 23/9 pomeriggio). init_empty_weights (meta
    # device, storage zero) elimina il picco: load_state_dict con assign=True
    # SOSTITUISCE il Parameter (non copia dati dentro), quindi funziona
    # identico anche partendo da meta device. Non e' una leva di velocita',
    # e' un presidio anti-OOM.
    with init_empty_weights():
        strato = Qwen3_5MoeDecoderLayer(cfg, layer_idx=L)
    idx = _json.load(open(SRC / "model.safetensors.index.json"))["weight_map"]
    pref = f"model.language_model.layers.{L}."
    per_file: dict[str, list[str]] = {}
    for k, f in idx.items():
        if k.startswith(pref):
            per_file.setdefault(f, []).append(k)
    sd = {}
    for f, kk in per_file.items():
        with safe_open(str(SRC / f), "pt") as sf:
            for k in kk:
                sd[k.split(pref)[1]] = sf.get_tensor(k)
    manca, _ = strato.load_state_dict(sd, strict=False, assign=True)
    if manca:
        raise RuntimeError(f"strato {L}: chiavi bf16 mancanti {manca}")

    if da_gguf:
        sd_q = _pesi_gguf_con_cache(GGUF, L)
        noti = dict(strato.named_parameters())
        messi = 0
        for nome, t in sd_q.items():
            if nome in noti and noti[nome].shape == t.shape:
                with torch.no_grad():
                    noti[nome].copy_(t.to(noti[nome].dtype))
                messi += 1
        if messi == 0:
            raise RuntimeError(f"strato {L}: nessun peso GGUF applicato — "
                               "lo stream 'quantizzato' sarebbe identico al pulito, "
                               "e la cattura misurerebbe il braccio sbagliato")
        log(f"  strato {L}: {messi} tensori presi dal GGUF per il flusso quantizzato")
    return strato.to("cuda", torch.bfloat16).eval()


class FlussoSuDisco:
    """⛔⛔⛔ 23/9 16:58 — CAUSA VERA dello swap thrashing (misurato due volte:
    RAM 28-29/30 GiB, swap fino a 30/51, pressione I/O oltre 50%, py-spy ha
    mostrato il processo bloccato dentro il calcolo, non un mio bug locale):
    hs_ref/hs_q come lista Python tengono l'INTERO corpus (113 batch, 2 flussi)
    in RAM per tutta la durata della cattura — ~15 GiB fissi, quasi meta' dei
    30 GiB della macchina, PRIMA di contare sr/sq/forgia/accumulatori.

    Non si riduce il corpus (sarebbe la scorciatoia vietata: peggiorerebbe
    proprio il problema delle Hessiane sotto-campionate gia' trovato — meno
    campioni = piu' esperti degeneri, non meno). Il disco ha 300+ GiB liberi
    e questo stato serve UN batch alla volta: lo si sposta li'.

    Stessa interfaccia di una lista (`hs_ref[b]`, `hs_ref[b] = tensore`) cosi'
    il resto del file (loop principale, ripresa, propagazione sequenziale)
    non cambia di una riga — solo l'inizializzazione.

    ⭐ 23/9 18:xx — I/O ASINCRONO (5 agenti avversari, mandato di Angelo
    "applica tutte le leve", verificato 5,39x su pesi VERI Thor strato-0,
    227,69s->42,24s/strato, bit-esatto): formato safetensors (mmap-capace)
    + scrittura su thread separato, lettura con prefetch a 1 elemento.

    ⛔⛔⛔ Il punto critico, viste le due lezioni gia' pagate su QUESTO file
    (swap thrashing di stanotte, "wb" che troncava/azzerava una ripresa il
    17/9): un I/O asincrono NON deve mai lasciare che una lettura veda un
    file a meta' scritto, ne' che un marker ".ok" dica "fatto" mentre una
    scrittura per quello strato e' ancora in coda. Due presidi, non uno:
    1) __getitem__ ASPETTA la scrittura pendente per lo STESSO indice, se
       c'e' — nessuna delle 3 letture sparse nel file deve "ricordarsi" di
       sincronizzare, e' automatico;
    2) flush() e' chiamato ESPLICITAMENTE prima di ogni marker ".ok" che
       dichiara uno strato completo (vedi sotto, prima di `marca.write_text`)
       — un crash del PROCESSO (non solo un getitem nello stesso processo)
       fra "scrittura sottomessa" e "scrittura su disco" non deve mai poter
       produrre un marker che mente.
    Backpressure: al massimo 2 scritture in coda per flusso (semaforo) — se
    il disco resta indietro, il chiamante (il forward GPU) rallenta invece
    di accumulare tensori in RAM: e' lo stesso principio che ha risolto lo
    swap thrashing, non lo si reintroduce dal lato scrittura."""
    def __init__(self, cartella: Path, profondita_scrittura: int = 2):
        self.cartella = cartella
        self.cartella.mkdir(parents=True, exist_ok=True)
        self._scrittore = ThreadPoolExecutor(max_workers=1)
        self._lettore = ThreadPoolExecutor(max_workers=1)
        self._sem_scrittura = threading.Semaphore(profondita_scrittura)
        self._lock = threading.Lock()
        self._scritture_pendenti: dict[int, object] = {}
        self._cache_letture: dict[int, torch.Tensor] = {}
        self._prefetch_avviati: set[int] = set()

    def _percorso(self, b: int) -> Path:
        return self.cartella / f"{b:04d}.safetensors"

    def _scrivi_su_disco(self, b: int, tensore: torch.Tensor) -> None:
        try:
            _st_save_file({"t": tensore.contiguous()}, str(self._percorso(b)))
        finally:
            self._sem_scrittura.release()

    def __setitem__(self, b: int, tensore: torch.Tensor) -> None:
        self._sem_scrittura.acquire()  # backpressure: mai piu' di N in coda
        fut = self._scrittore.submit(self._scrivi_su_disco, b, tensore)
        with self._lock:
            self._scritture_pendenti[b] = fut

    def _leggi_da_disco(self, b: int) -> torch.Tensor:
        return _st_load_file(str(self._percorso(b)))["t"]

    def _avvia_prefetch(self, b: int) -> None:
        with self._lock:
            if b in self._prefetch_avviati or b in self._cache_letture:
                return
            self._prefetch_avviati.add(b)

        def compito() -> None:
            try:
                t = self._leggi_da_disco(b)
                with self._lock:
                    self._cache_letture[b] = t
            except Exception:
                pass  # best-effort: il getitem vero ricarica lui stesso se serve
            finally:
                with self._lock:
                    self._prefetch_avviati.discard(b)

        self._lettore.submit(compito)

    def __getitem__(self, b: int) -> torch.Tensor:
        with self._lock:
            fut = self._scritture_pendenti.pop(b, None)
        if fut is not None:
            fut.result()  # aspetta SOLO se c'era davvero una scrittura pendente per b
        with self._lock:
            t = self._cache_letture.pop(b, None)
        if t is None:
            t = self._leggi_da_disco(b)
        self._avvia_prefetch(b + 1)
        return t

    def flush(self) -> None:
        """Aspetta che OGNI scrittura sottomessa finora sia su disco. Va
        chiamato prima di qualunque marker '.ok' che dichiari uno strato
        completo — mai omesso, anche se sembra 'gia' finita da se'."""
        with self._lock:
            futs = list(self._scritture_pendenti.values())
            self._scritture_pendenti.clear()
        for f in futs:
            f.result()


def main():
    precondizioni()
    OUT.mkdir(parents=True, exist_ok=True)

    cfg = AutoConfig.from_pretrained(str(SRC)).text_config
    cfg._attn_implementation = "sdpa"
    n_esp, hid = cfg.num_experts, cfg.hidden_size
    log(f"architettura: {cfg.num_hidden_layers} strati · hidden {hid} · "
        f"{n_esp} esperti top-{cfg.num_experts_per_tok}")

    tok = AutoTokenizer.from_pretrained(str(SRC))
    ids = tok(CORPUS.read_text(encoding="utf-8", errors="replace"),
              return_tensors="pt", add_special_tokens=False).input_ids[0]
    n_seq_tot = N_SEQ if N_SEQ > 0 else (ids.numel() // L_SEQ)
    n_batch = max(1, n_seq_tot // BATCH_SEQ)
    n_seq = n_batch * BATCH_SEQ
    if ids.numel() < n_seq * L_SEQ:
        raise RuntimeError(f"corpus {CORPUS}: {ids.numel()} token, servono {n_seq*L_SEQ}")
    ids = ids[: n_seq * L_SEQ].reshape(n_seq, L_SEQ)
    log(f"corpus {CORPUS.name}: {n_seq} seq x {L_SEQ} = {n_seq*L_SEQ} campioni "
        f"in {n_batch} passate da {BATCH_SEQ}")

    # ── embedding: i DUE flussi partono identici, poi divergono ──
    from safetensors import safe_open
    import json as _json
    idx = _json.load(open(SRC / "model.safetensors.index.json"))["weight_map"]
    kemb = "model.language_model.embed_tokens.weight"
    with safe_open(str(SRC / idx[kemb]), "pt") as sf:
        emb = sf.get_tensor(kemb).to("cuda", torch.bfloat16)
    # ⛔ su DISCO, non CPU: vedi FlussoSuDisco sopra — la lista in RAM di 113
    # batch x 2 flussi (~15 GiB) sommata al resto del pipeline (sr/sq/forgia)
    # ha causato swap thrashing reale, misurato due volte.
    hs_ref = FlussoSuDisco(OUT / "_hs_ref")
    hs_q = FlussoSuDisco(OUT / "_hs_q")
    with torch.no_grad():
        for b in range(n_batch):
            e = torch.nn.functional.embedding(ids[b*BATCH_SEQ:(b+1)*BATCH_SEQ].to("cuda"), emb)
            e_cpu = e.to(torch.bfloat16).cpu()
            hs_ref[b] = e_cpu
            hs_q[b] = e_cpu.clone()
    del emb
    gc.collect(); torch.cuda.empty_cache()
    log(f"embedding fatto, due flussi SU DISCO ({OUT}/_hs_ref, _hs_q) · VRAM {vram_gib():.2f} GiB")

    rot = Qwen3_5MoeTextRotaryEmbedding(cfg).to("cuda")
    pos = torch.arange(L_SEQ, device="cuda").unsqueeze(0)
    with torch.no_grad():
        pos_emb = rot(hs_ref[0].to("cuda"), pos)
    maschera = create_causal_mask(config=cfg, inputs_embeds=hs_ref[0].to("cuda"),
                                  attention_mask=None, past_key_values=None, position_ids=pos)

    t0 = time.time()
    for L in elenco_strati(cfg.num_hidden_layers):
        marca = OUT / f"fatto_{L:02d}.ok"
        if marca.exists():
            log(f"strato {L}: marca presente — salto (ripresa)")
            # ⛔ i flussi vanno fatti avanzare comunque, o gli strati dopo
            #    riceverebbero attivazioni di uno strato che non e' il loro.
            # ⛔⛔ 23/9: DEVE usare gli esperti GIA' TERNARIZZATI per sq (stessa
            # ragione della propagazione sequenziale sotto), altrimenti una
            # RIPRESA dopo un crash rompe silenziosamente l'invariante che
            # tutto questo file esiste per garantire — ricalcolo la
            # ternarizzazione da Wstar_L.npy/Gq_L.npy gia' salvati (costa
            # ~2x la forgia di uno strato, accettabile: la ripresa e' rara).
            from forgia_schur_esperti import forgia_batch, N_ROUND as _NR
            from rotazione_gpu import de_ruota_pesi_batch_gpu  # ⭐ 23/9 20:1x — GPU, non piu' CPU/numpy
            sr = costruisci_strato(cfg, L, da_gguf=False)
            sq = costruisci_strato(cfg, L, da_gguf=True)
            # ⛔⛔⛔ stessa cura anti-swap-thrashing del percorso normale (23/9
            # 16:45, OOM vero misurato: RAM 28/30, swap 30/51, I/O 47%): Wstar
            # NON upcastato subito, Gq_layer liberata appena finita di servire,
            # ogni pezzo liberato appena inutile invece che a fine blocco.
            # ⭐ 24/9 03:0x — LEVA DI VELOCITA' (Angelo: "integra tutte le
            # leve, dobbiamo essere veloci"): gate e up condividono la STESSA
            # Gq_layer (dipende solo da X, mai dall'uscita) e forgia_batch non
            # accoppia righe out_rows diverse fra loro (coordinate descent per
            # riga, verificato) — una chiamata sola su Wstar_np intero al posto
            # di due dimezza l'overhead di lancio GPU per strato. Rischio
            # dichiarato (VRAM ~2x) VERIFICATO e SMENTITO su GPU reale, strato
            # 00 vero: 51,54 vs 51,57 GB, rapporto 1,00x — nessun costo extra.
            Wstar_np = np.load(OUT / f"Wstar_{L:02d}.npy")  # f16
            Gq_layer = np.load(OUT / f"Gq_{L:02d}.npy").astype(np.float32)
            meta_seq = Wstar_np.shape[1] // 2
            D_seq, Q_seq = forgia_batch(Wstar_np, Gq_layer, n_round=_NR)
            del Wstar_np, Gq_layer
            Dg_seq, Du_seq = D_seq[:, :meta_seq, :], D_seq[:, meta_seq:, :]
            Qg_seq, Qu_seq = Q_seq[:, :meta_seq, :], Q_seq[:, meta_seq:, :]
            del D_seq, Q_seq
            np.save(OUT / f"Dg_{L:02d}.npy", Dg_seq)
            np.save(OUT / f"Qg_{L:02d}.npy", Qg_seq)
            # ⛔ stessa ragione del percorso normale: senza questo, la ripresa
            # lascerebbe lo strato senza Dg/Qg/Du/Qu per l'esportazione finale.
            np.save(OUT / f"Du_{L:02d}.npy", Du_seq)
            np.save(OUT / f"Qu_{L:02d}.npy", Qu_seq)
            (OUT / f"fatto_forgia_{L:02d}.ok").write_text(f"{time.strftime('%F %T')} sequenziale (ripresa)\n")
            QDg = Qg_seq.astype(np.float32) * Dg_seq.repeat(GROUP, axis=-1)
            Wpred_gate = de_ruota_pesi_batch_gpu(QDg, device="cuda")
            del QDg
            gc.collect()
            QDu = Qu_seq.astype(np.float32) * Du_seq.repeat(GROUP, axis=-1)
            Wpred_up = de_ruota_pesi_batch_gpu(QDu, device="cuda")
            del QDu
            gc.collect()

            Wpred = np.concatenate([Wpred_gate, Wpred_up], axis=1)
            del Wpred_gate, Wpred_up
            gc.collect()
            with torch.no_grad():
                sq.mlp.experts.gate_up_proj.copy_(
                    torch.from_numpy(Wpred).to(sq.mlp.experts.gate_up_proj.device,
                                                sq.mlp.experts.gate_up_proj.dtype))
            with torch.no_grad():
                for b in range(n_batch):
                    o = sr(hidden_states=hs_ref[b].to("cuda"), position_embeddings=pos_emb,
                           attention_mask=maschera, position_ids=pos)
                    hs_ref[b] = (o[0] if isinstance(o, tuple) else o).cpu()
                    o = sq(hidden_states=hs_q[b].to("cuda"), position_embeddings=pos_emb,
                           attention_mask=maschera, position_ids=pos)
                    hs_q[b] = (o[0] if isinstance(o, tuple) else o).cpu()
            del sr, sq, Dg_seq, Qg_seq, Du_seq, Qu_seq, Wpred
            gc.collect(); torch.cuda.empty_cache()
            continue

        t_str = time.time()
        sr = costruisci_strato(cfg, L, da_gguf=False)
        sq = costruisci_strato(cfg, L, da_gguf=True)
        controlla_tetto(f"strato {L} caricato (2 copie)")
        controlla_spazio(L, "prima del forward")

        G = torch.zeros(n_esp, hid, hid, dtype=torch.float32, device="cuda")
        M = torch.zeros(n_esp, hid, hid, dtype=torch.float32, device="cuda")
        peso = torch.zeros(n_esp, dtype=torch.float64, device="cuda")
        conta = torch.zeros(n_esp, dtype=torch.int64, device="cuda")
        controlla_tetto(f"strato {L} accumulatori G/M")

        # ⭐ 23/9 19:4x — PROFILO a sezioni (Angelo: "cicli di miglioramento
        # finche' non scende sotto 2h totali"): prima di toccare altro,
        # misuriamo DAVVERO dove va il tempo, non si indovina. torch.cuda.
        # synchronize() prima/dopo ogni sezione: senza, il tempo di un kernel
        # GPU asincrono si "sposta" sulla sezione successiva che sincronizza,
        # falsando il profilo (lezione dalla diagnosi CPU-bound sbagliata di stanotte).
        _t_fwd_tot = 0.0
        _t_gm_tot = 0.0

        # ⛔⛔⛔ 23/9 23:3x — skip-attn/SSM (cache residuo/mlp_input) DISATTIVATO
        # per isolare una diagnosi (vedi nota gemella nel secondo forward,
        # sotto). Il codice della leva resta qui commentato per riattivarla
        # rapidamente una volta trovata la causa vera del MUTO su 0-12.
        # _residuo_sq = FlussoSuDisco(OUT / "_residuo_sq_tmp")
        # _mlpin_sq = FlussoSuDisco(OUT / "_mlpin_sq_tmp")

        for b in range(n_batch):
            pres = {}

            def fai_gancio(chiave):
                def g(_m, _i, out):
                    pres[chiave] = out.detach()
                return g

            hr = sr.post_attention_layernorm.register_forward_hook(fai_gancio("ref"))
            hq = sq.post_attention_layernorm.register_forward_hook(fai_gancio("q"))
            torch.cuda.synchronize(); _t0 = time.time()
            with torch.no_grad():
                o = sr(hidden_states=hs_ref[b].to("cuda"), position_embeddings=pos_emb,
                       attention_mask=maschera, position_ids=pos)
                nuovo_ref = (o[0] if isinstance(o, tuple) else o)
                o = sq(hidden_states=hs_q[b].to("cuda"), position_embeddings=pos_emb,
                       attention_mask=maschera, position_ids=pos)
                nuovo_q = (o[0] if isinstance(o, tuple) else o)
            hr.remove(); hq.remove()
            torch.cuda.synchronize(); _t_fwd_tot += time.time() - _t0

            _t0 = time.time()
            with torch.no_grad():
                # ⭐⭐⭐⭐⭐⭐ 23/9 23:3x — bf16 sul matmul Hessiano (Angelo:
                # "applicala, testala, verificala"). X_ref/X_q sono GIA' bf16
                # nativi (uscita del modello): tenerli tali per il matmul
                # (invece di upcastare a fp32) attiva il path WMMA veloce.
                # VERIFICATO PRIMA di applicare (script scratchpad
                # verifica_bf16_completa.py, 8 batch, 256 esperti, poi forgia
                # VERA su 24 esperti e confronto contro la Hessiana vera):
                # errore pesato-Hessiana finale IDENTICO (rapporto 1,000x),
                # velocita' matmul nativo REALE 7,06x. L'accumulo in G/M resta
                # SEMPRE fp32 (.float() sul risultato, mai accumulo ripetuto
                # in bf16 — evita qualunque compounding su 113 batch).
                X_ref = pres["ref"].reshape(-1, hid)
                X_q = pres["q"].reshape(-1, hid)
                # ⛔ instradamento sul flusso QUANTIZZATO: sono i token che
                #    arrivano DAVVERO all'esperto in produzione.
                mappa = token_per_esperto(pres["q"].reshape(-1, hid), sq.mlp.gate,
                                          cfg.num_experts_per_tok, n_esp)
                for e, v in mappa.items():
                    ti = v["token_idx"]
                    w = v["peso"].to(torch.bfloat16)           # pesi VERI del router
                    xq = X_q[ti]
                    xr = X_ref[ti]                             # STESSI token, flusso pulito
                    xw = xq * w[:, None]
                    G[e] += (xq.T @ xw).float()
                    M[e] += (xr.T @ xw).float()
                # ⭐ 23/9 — contabilita' vettorizzata (5 agenti avversari indipendenti,
                # bit-esatto diff=0.0, 1.5x sotto contesa / 2.0x a GPU libera, verificato
                # su questo esatto ciclo con pesi VERI Thor). Il matmul sopra resta
                # IDENTICO (gia' efficiente su rocBLAS): solo le due righe scalari
                # peso[e]+=/conta[e]+= (18-30% del tempo in ~500 lanci GPU minuscoli)
                # diventano un solo scatter_add per batch.
                expert_ids = torch.cat([
                    torch.full((v["token_idx"].numel(),), e, dtype=torch.long, device="cuda")
                    for e, v in mappa.items()
                ])
                pesi_flat = torch.cat([v["peso"].float() for v in mappa.values()]).double()
                peso.scatter_add_(0, expert_ids, pesi_flat)
                conta.scatter_add_(0, expert_ids, torch.ones_like(expert_ids))
                hs_ref[b] = nuovo_ref.cpu()
                # ⛔ hs_q[b] NON si avanza qui: 'nuovo_q' viene da sq con gli
                # esperti ANCORA bf16 (vedi PROPAGAZIONE SEQUENZIALE sotto,
                # dopo la forgia di questo strato). hs_q[b] resta l'INGRESSO
                # di questo strato finche' non lo ricalcoliamo coi pesi veri.
            torch.cuda.synchronize(); _t_gm_tot += time.time() - _t0
            del pres, X_ref, X_q, nuovo_ref, nuovo_q
            controlla_tetto(f"strato {L} passata {b}")

        colpiti = int((conta > 0).sum())
        log(f"  strato {L} PROFILO cattura: forward(sr+sq)={_t_fwd_tot:.1f}s "
            f"accumulo-G/M(bf16)={_t_gm_tot:.1f}s su {n_batch} batch")
        # ⛔ flush() di _residuo_sq/_mlpin_sq RIMOSSO insieme alla leva
        # (disattivata sopra per la diagnosi in corso, vedi nota).

        # ⭐⭐⭐ 22/9 18:45 — FORGIA E SCARTA, DENTRO LO STESSO STRATO.
        # Prima qui si salvavano G e M grezzi: 8,59 GiB per strato, mai
        # scartati → 344 GiB su 40 strati contro un volume CONDIVISO da 200,
        # saturato a strato 9, due corse uccise in silenzio (vedi
        # controlla_spazio). La cura non e' una guardia piu' stretta: e'
        # non accumulare affatto. W* si risolve QUI, con `sr` (il maestro
        # bf16) gia' in memoria, e di G/M non resta niente su disco.
        #   W_ref[e] = gate_up_proj[e]  (1024, 2048) — gate e up FUSI
        #   G_e = G[e]/peso · M_e = M[e]/peso · C_e = W_ref @ M_e
        #   W*_e = w_star(smorza(G_e + λI), C_e + λ W_ref)
        # Uscita: (256, 1024, 2048) f16 = 1,07 GiB/strato — otto volte meno.
        # ⛔ λ resta un parametro: 0,3 e' il valore scelto sul cal dal
        # controllo cross-strato e riportato su held-out, non un'invenzione.
        from schurquant_riconduzione import smorza, w_star
        LAM = float(os.getenv("THOR_LAMBDA", "0.3"))
        W_ref_tutti = sr.mlp.experts.gate_up_proj.detach().to(torch.float64)  # (256,1024,2048)
        W_star = torch.zeros_like(W_ref_tutti, dtype=torch.float16)
        n_risolti = 0
        with torch.no_grad():
            occhio = torch.eye(hid, dtype=torch.float64, device="cuda")
            for e in range(n_esp):
                if conta[e] == 0:
                    W_star[e] = W_ref_tutti[e].half()   # mai instradato: resta il maestro
                    continue
                p_e = peso[e]
                G_e = G[e].double() / p_e
                M_e = M[e].double() / p_e
                W_r = W_ref_tutti[e]
                C_e = W_r @ M_e
                W_star[e] = w_star(smorza(G_e + LAM * occhio), C_e + LAM * W_r).half()
                n_risolti += 1
            del occhio, W_ref_tutti
        np.save(OUT / f"Wstar_{L:02d}.npy", W_star.cpu().numpy())
        del W_star
        # ⭐⭐ 22/9 18:55 — NON si butta piu' l'informazione: si tiene G e C.
        # Angelo: «se scopro che sbagli la forge e ti serviva lo strato che
        # stai buttando, divento furioso» — e aveva ragione, perche' avevo
        # scritto io stesso nel docstring «λ resta libero a forgia, cosi' si
        # puo' spazzolare senza ricatturare» e poi avevo applicato λ=0.3 nella
        # cattura buttando G/M, distruggendo proprio quella liberta'.
        # ⛔ Ma NON serve M: per rifare la spazzolata di λ basta
        #        W* = w_star(smorza(G + λI), C + λ W_ref)
        # e C = W_ref @ M e' (1024,2048) invece di (2048,2048) — QUATTRO volte
        # piu' piccola. Tenere G+C (5,0 GiB/strato) invece di G+M (8,0) da' la
        # STESSA liberta' a meta' prezzo. W_ref si rilegge dal modello, sempre.
        # ⛔⛔⛔ 24/9 01:4x — GUARDIA DIVISIONE PER ZERO (trovata dal workflow
        # diagnostico, mai innescata nei layer 0-10 ma un difetto latente
        # reale): un esperto MAI instradato ha peso[e]==0 e G[e] tutta zero
        # -> 0/0=NaN, non intercettato da smorza_batch (NaN==0 e' False in
        # IEEE754). Per quell'esperto G e' gia' zero e W_star e' gia' il
        # riferimento puro (righe 597-599): un divisore sicuro (1 al posto di
        # 0) da' Gq_layer[e]=0/1=0, coerente e innocuo — MAI NaN.
        peso_safe = torch.where(peso > 0, peso, torch.ones_like(peso))
        Gq_layer = (G.double() / peso_safe[:, None, None]).float().cpu().numpy()
        np.save(OUT / f"Gq_{L:02d}.npy", Gq_layer)

        # ⭐⭐⭐ 23/9 14:xx — PROPAGAZIONE SEQUENZIALE (fix del difetto trovato
        # dal Workflow diagnostico: gli strati a valle vedevano attivazioni
        # calcolate con esperti ANCORA bf16, mai coi nuovi esperti ternari —
        # lo scarto di distribuzione si accumulava lungo la profondita' e a
        # 40 strati collassava la generazione, verificato empiricamente con
        # un test a strati crescenti: 1-5 strati perfetti, 10 rotti, 40 muto).
        # Qui si ternarizza IMMEDIATAMENTE W* con la stessa forgia_batch (gia'
        # corretta con smorza_batch, vedi forgia_schur_esperti.py) e si
        # sostituisce il risultato negli esperti di sq PRIMA di calcolare
        # l'uscita che avanza hs_q — cosi' lo strato L+1 cattura su
        # attivazioni che riflettono DAVVERO gli esperti gia' ternarizzati,
        # non piu' bf16. Costo: un forward extra di sq su tutti i batch.
        from forgia_schur_esperti import forgia_batch, N_ROUND as _NR
        from rotazione_gpu import de_ruota_pesi_batch_gpu  # ⭐ 23/9 20:1x — GPU, non piu' CPU/numpy

        # ⛔⛔⛔ 23/9 16:45 — OOM/swap-thrashing VERO in produzione (non un
        # falso allarme): RAM 28/30 GiB, swap 30/51 GiB, pressione I/O 47%
        # (soglia d'allarme 40%, vedi nodo testing), py-spy ha mostrato il
        # processo bloccato dentro ruota_hessiana_batch_gpu — non un difetto
        # di quella funzione (verificata stanotte), ma troppi array grandi
        # vivi insieme: hs_ref/hs_q (~15 GiB su CPU per l'intero corpus,
        # necessari per tutta la durata) + Wstar upcastato a f32 (2,1 GiB
        # invece di 1,07 in f16) + Gq_layer (4,29 GiB) + Hrot f64 (8,59 GiB,
        # RITORNATO come nuovo array mentre l'INGRESSO resta vivo) + QDg/QDu/
        # Wpred (~4,2 GiB) tutti insieme superano i 30 GiB della macchina.
        # Qui NON si tocca la precisione (verificata, non e' il problema):
        # si tiene Wstar in f16 (meta' spazio) e si libera OGNI pezzo appena
        # diventa inutile, invece di aspettare la fine del blocco.
        # ⭐ 24/9 03:0x — LEVA DI VELOCITA' (stessa nota gemella nel ramo di
        # ripresa sopra): una chiamata forgia_batch sola su gate+up insieme,
        # verificata su GPU reale senza costo extra di VRAM (1,00x).
        Wstar_np = np.load(OUT / f"Wstar_{L:02d}.npy")  # f16, (256,1024,2048) — NON upcastato qui
        meta_seq = Wstar_np.shape[1] // 2
        _t0 = time.time()
        D_seq, Q_seq = forgia_batch(Wstar_np, Gq_layer, n_round=_NR)
        del Wstar_np, Gq_layer
        _t_forgia = time.time() - _t0
        Dg_seq, Du_seq = D_seq[:, :meta_seq, :], D_seq[:, meta_seq:, :]
        Qg_seq, Qu_seq = Q_seq[:, :meta_seq, :], Q_seq[:, meta_seq:, :]
        del D_seq, Q_seq
        np.save(OUT / f"Dg_{L:02d}.npy", Dg_seq)
        np.save(OUT / f"Qg_{L:02d}.npy", Qg_seq)
        np.save(OUT / f"Du_{L:02d}.npy", Du_seq)
        np.save(OUT / f"Qu_{L:02d}.npy", Qu_seq)
        (OUT / f"fatto_forgia_{L:02d}.ok").write_text(f"{time.strftime('%F %T')} sequenziale\n")
        QDg = Qg_seq.astype(np.float32) * Dg_seq.repeat(GROUP, axis=-1)
        Wpred_gate = de_ruota_pesi_batch_gpu(QDg, device="cuda")
        del QDg
        gc.collect()
        QDu = Qu_seq.astype(np.float32) * Du_seq.repeat(GROUP, axis=-1)
        Wpred_up = de_ruota_pesi_batch_gpu(QDu, device="cuda")
        del QDu
        gc.collect()

        Wpred = np.concatenate([Wpred_gate, Wpred_up], axis=1)  # (256,1024,2048), de-ruotato
        del Wpred_gate, Wpred_up
        gc.collect()
        with torch.no_grad():
            sq.mlp.experts.gate_up_proj.copy_(
                torch.from_numpy(Wpred).to(sq.mlp.experts.gate_up_proj.device,
                                            sq.mlp.experts.gate_up_proj.dtype))
        _t0 = time.time()
        with torch.no_grad():
            # ⛔⛔⛔ 23/9 23:3x — DISATTIVATO TEMPORANEAMENTE per isolare una
            # diagnosi: la sonda sui 13 strati (0-12) ha mostrato MUTO (guasto
            # grave) che NON appare sui soli strati 0-5 (senza questa leva).
            # La verifica isolata del meccanismo (verifica_skip_attn_multibatch.py,
            # pesi VERI dello strato 7, 6 batch reali in sequenza) e' risultata
            # bit-esatta — quindi il meccanismo in se' NON e' provato colpevole,
            # ma finche' non si trova la causa vera si torna al forward PIENO
            # (piu' lento, MAI sbagliato) per rifare 6-14 e isolare la variabile.
            # Il codice della leva (sopra, cache _residuo_sq/_mlpin_sq) resta
            # scritto e verificato: si riattiva SOLO dopo aver capito la causa.
            for b in range(n_batch):
                o = sq(hidden_states=hs_q[b].to("cuda"), position_embeddings=pos_emb,
                       attention_mask=maschera, position_ids=pos)
                nuovo_q_tern = (o[0] if isinstance(o, tuple) else o)
                hs_q[b] = nuovo_q_tern.cpu()
        torch.cuda.synchronize(); _t_secondo_fwd = time.time() - _t0
        del Dg_seq, Qg_seq, Du_seq, Qu_seq, Wpred
        gc.collect(); torch.cuda.empty_cache()
        log(f"  propagazione sequenziale: esperti ternarizzati sostituiti in sq, "
            f"hs_q avanzato con gli esperti VERI (non piu' bf16)")
        log(f"  strato {L} PROFILO forgia+propagazione: forgia(gate+up)={_t_forgia:.1f}s "
            f"secondo-forward(sq)={_t_secondo_fwd:.1f}s")

        C_tutti = torch.einsum("eoi,eij->eoj",
                               sr.mlp.experts.gate_up_proj.detach().double(),
                               (M.double() / peso[:, None, None]))
        np.save(OUT / f"C_{L:02d}.npy", C_tutti.float().cpu().numpy())
        del C_tutti
        np.save(OUT / f"peso_{L:02d}.npy", peso.cpu().numpy())
        np.save(OUT / f"conta_{L:02d}.npy", conta.cpu().numpy())
        log(f"  forgia SchurQuant in-strato: {n_risolti}/{n_esp} esperti risolti (λ={LAM}), "
            f"G/M scartati — su disco solo W* (1,07 GiB)")
        # ⛔⛔⛔ OBBLIGATORIO prima del marker: con l'I/O asincrono una scrittura
        # di hs_ref/hs_q puo' essere ancora in coda quando si arriva qui. Senza
        # questo flush, un crash del processo fra "sottomessa" e "su disco"
        # lascerebbe un marker che dichiara lo strato fatto mentre non lo e'.
        hs_ref.flush()
        hs_q.flush()
        marca.write_text(f"{time.strftime('%F %T')} esperti colpiti {colpiti}/{n_esp}\n")
        log(f"strato {L}: SALVATO · esperti colpiti {colpiti}/{n_esp} "
            f"(min {int(conta[conta>0].min()) if colpiti else 0} token) · "
            f"{time.time()-t_str:.0f}s · VRAM {vram_gib():.1f} GiB")
        if colpiti < n_esp:
            log(f"  ⚠️ {n_esp-colpiti} esperti MAI colpiti: a forgia restano RTN, "
                "come oggi — non e' un guasto, e' copertura del corpus")

        del G, M, peso, conta, sr, sq
        gc.collect(); torch.cuda.empty_cache()

    log(f"FATTO in {(time.time()-t0)/60:.1f} min → {OUT}")


if __name__ == "__main__":
    main()

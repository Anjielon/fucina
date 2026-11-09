"""Prove del corpus di calibrazione v2 (due varianti) e del pannello v2 — solo CPU, ripetibili (25/9 notte).

Usa `valida_finestra_8_12.py` VERO (gia' con patch_corpus_calibrazione + patch_corpus_guardia_heldout),
solo importandolo in un processo a parte; `pannello_heldout.py` su una COPIA con patch_pannello_nuovi_domini
(la patch vera NON e' applicata: vedi D3). Nessun file esistente viene scritto.

  F   formato, per dominio e fonte, sulle sequenze dei due corpora: <think>/</think> e
      <|im_start|>/<|im_end|> bilanciati nel flusso di ogni fonte (a meno dell'ultimo esempio, troncato),
      nessun <|endoftext|>, chiamate MASSIVE in XML con argomenti (mai stringhe JSON)
  T1  v2 «tutte» + U, 150 batch, N_HELDOUT=1: accettata; ogni ciclo di 2 batch ha le quote U; ogni sequenza
      di dominio d e' corpus[d][i] nell'ordine, usata una volta
  T2  v2 «tutte» + U con N_HELDOUT=4: rifiutata dalla guardia
  T3  v2 «senza_llama» + U, 150 batch, N_HELDOUT=1: come T1
  T4  v2 «senza_llama» + U con N_HELDOUT=4: rifiutata
  T5  v2 «senza_llama» + M2 a 150 batch: rifiutata (cot ha 150 sequenze, ne servono 300)
  T6  held-out di T1 e T3 identico a quello senza corpus (M2, 32 batch)
  P1  pannello_heldout --solo-pannello (copia patchata) con THOR_PANNELLO_NUOVI=pannello_nuovi_domini_v2.pt:
      storici identici, domini nuovi v2 caricati con sha256 verificato

    env CUDA_VISIBLE_DEVICES= nice -n 19 @@ROOT:venv@@/bin/python3 strumenti/prova_corpus_v2.py
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

QUI = Path(__file__).resolve().parent
THOR = QUI.parent
PY = "@@ROOT:venv@@/bin/python3"
CAL = Path("@@ROOT:models@@/gguf/tony-tern/calibrazione")
PAN = Path("@@ROOT:models@@/gguf/tony-tern/pannello")
CORPORA = {"tutte": CAL / "calibrazione_corpus_v2.pt", "senza_llama": CAL / "calibrazione_corpus_v2_senza_llama.pt"}
PAN_V2 = PAN / "pannello_nuovi_domini_v2.pt"
U = "wiki:3,edu:2,cot:2,it_web:2,it_wiki:1,it_chat:3,it_pensiero:1,strumenti:2"
M2 = "cot:2,edu:3,wiki:3"
USCITA = Path(os.getenv("PROVA_V2_CARTELLA", "/tmp/claude-1000/-home-angelo-MOGAVIS/12e4589a-507e-45a0-b690-66b786e2b26f"
                                             "/scratchpad/corpus_v2/prova"))
esiti: dict = {}


def log(*a) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def segna(nome: str, ok: bool, dettaglio: str = "") -> None:
    esiti[nome] = bool(ok)
    log(f"{'✅' if ok else '⛔'} {nome} {dettaglio}")


def ambiente(extra: dict, cartella: "Path | None" = None) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("THOR_CALIB") and not k.startswith("THOR_PANNELLO")}
    env.update({"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "TORCHDYNAMO_DISABLE": "1",
                "OMP_NUM_THREADS": "4", "RAYON_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "true",
                "PYTHONPATH": ":".join(([str(cartella)] if cartella else []) + [str(THOR)])})
    env.update(extra)
    return env


def lancia_batch(uscita: Path, extra: dict) -> subprocess.CompletedProcess:
    codice = (f"import sys, torch; sys.path.insert(0, {str(THOR)!r}); import valida_finestra_8_12 as V; "
              f"ids, n = V.carica_batch_calibrazione(); torch.save({{'ids': ids, 'n': n}}, {str(uscita)!r})")
    env = ambiente({"THOR_VALIDA_BATCH_SEQ": "8", **extra})
    return subprocess.run(["nice", "-n", "19", PY, "-c", codice], env=env, capture_output=True, text=True,
                          timeout=3600, cwd=str(THOR))


def prova_formato(torch, tok) -> None:
    speciali = {n: tok.convert_tokens_to_ids(n) for n in ("<think>", "</think>", "<|im_start|>", "<|im_end|>",
                                                          "<|endoftext|>", "<tool_call>", "</tool_call>")}
    for v, p in CORPORA.items():
        c = torch.load(p, weights_only=True)
        man = json.loads(p.with_suffix(".json").read_text())
        for d, t in c.items():
            et = []
            for f, n in man["etichette_per_sequenza"][d]:
                et += [f] * n
            per_fonte: dict = {}
            for i, f in enumerate(et):
                per_fonte.setdefault(f, []).append(i)
            for f, righe in per_fonte.items():
                flusso = t[righe].reshape(-1)
                conta = {n: int((flusso == i).sum()) for n, i in speciali.items()}
                sbil_think = abs(conta["<think>"] - conta["</think>"])
                sbil_im = abs(conta["<|im_start|>"] - conta["<|im_end|>"])
                sbil_tc = abs(conta["<tool_call>"] - conta["</tool_call>"])
                ok = conta["<|endoftext|>"] == 0 and sbil_think <= 1 and sbil_im <= 1 and sbil_tc <= 1
                if d in ("wiki", "edu", "it_web", "it_wiki"):
                    ok = ok and conta["<|im_start|>"] == 0 and conta["<think>"] == 0
                segna(f"F {v}/{d}/{f}", ok, json.dumps(conta))
        # chiamate: ogni blocco <tool_call>...</tool_call> e' XML <function=...><parameter=...>, mai JSON
        # (la parola "arguments" puo' comparire nel RAGIONAMENTO di Nemotron: non si guarda il testo intero)
        testo = tok.decode(c["strumenti"].reshape(-1).tolist())
        blocchi = [b.split("</tool_call>")[0] for b in testo.split("<tool_call>")[1:] if "</tool_call>" in b]
        # blocchi vuoti = la frase del template «nested within <tool_call></tool_call> XML tags» (una per system)
        vuoti = sum(1 for b in blocchi if not b.strip())
        json_in_chiamata = [b[:80] for b in blocchi if b.strip() and not b.lstrip().startswith("<function=")]
        segna(f"F {v}/strumenti formato chiamate", bool(blocchi) and not json_in_chiamata,
              f"{len(blocchi) - vuoti} chiamate XML, {vuoti} menzioni vuote del template, "
              f"{len(json_in_chiamata)} non XML {json_in_chiamata[:2]}")


def controlla_batch(torch, v: str, uscita: Path) -> None:
    c = torch.load(CORPORA[v], weights_only=True)
    r = torch.load(uscita, weights_only=True)
    ids, n = r["ids"], r["n"]
    quote = {p.split(":")[0]: int(p.split(":")[1]) for p in U.split(",")}
    tr = ids[:n].reshape(-1, 1024)
    # ogni riga di addestramento -> (dominio, indice) nel corpus
    indice = {}
    for d, t in c.items():
        for i in range(min(len(t), quote.get(d, 0) * (n // 2))):
            indice.setdefault(hashlib.sha256(t[i].numpy().tobytes()).hexdigest(), (d, i))
    presi = [indice.get(hashlib.sha256(x.numpy().tobytes()).hexdigest()) for x in tr]
    segna(f"{v}: ogni riga di addestramento e' una sequenza del corpus", all(p is not None for p in presi),
          f"forma {tuple(ids.shape)} n={n}")
    ok_cicli = True
    for ciclo in range(n // 2):
        cnt: dict = {}
        for p in presi[ciclo * 16:(ciclo + 1) * 16]:
            if p:
                cnt[p[0]] = cnt.get(p[0], 0) + 1
        ok_cicli &= cnt == {d: q for d, q in quote.items() if q}
    segna(f"{v}: ogni ciclo di 2 batch ha le quote U", ok_cicli)
    ordine_ok = True
    visti: dict = {}
    for p in presi:
        if p:
            ordine_ok &= p[1] == visti.get(p[0], 0)
            visti[p[0]] = visti.get(p[0], 0) + 1
    segna(f"{v}: sequenze di ogni dominio prese nell'ordine del file, una volta sola", ordine_ok,
          json.dumps(visti))


def main() -> int:
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(4)
    if USCITA.exists():
        shutil.rmtree(USCITA)
    USCITA.mkdir(parents=True)
    impronte = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [THOR / "valida_finestra_8_12.py", THOR / "pannello_heldout.py", PAN / "pannello_ids.pt",
                          PAN / "pannello_nuovi_domini.pt", CAL / "calibrazione_corpus.pt"]}
    tok = AutoTokenizer.from_pretrained("@@ROOT:models@@/gguf/thor-src-hf")
    prova_formato(torch, tok)
    # held-out di riferimento senza corpus
    r0 = lancia_batch(USCITA / "rif.pt", {"THOR_VALIDA_N_BATCH": "32", "THOR_VALIDA_N_HELDOUT": "1",
                                          "THOR_CALIB_MISCELA": M2})
    segna("riferimento senza corpus (M2, 32 batch, N_HELDOUT=1)", r0.returncode == 0, r0.stderr[-300:] if r0.returncode else "")
    ho_rif = torch.load(USCITA / "rif.pt", weights_only=True)["ids"][32:]
    for v, nomi in (("tutte", ("T1", "T2")), ("senza_llama", ("T3", "T4"))):
        base = {"THOR_VALIDA_N_BATCH": "150", "THOR_CALIB_CORPUS": str(CORPORA[v]), "THOR_CALIB_MISCELA": U}
        r = lancia_batch(USCITA / f"{v}.pt", {**base, "THOR_VALIDA_N_HELDOUT": "1"})
        segna(f"{nomi[0]} {v} + U, 150 batch, N_HELDOUT=1 accettata", r.returncode == 0, r.stderr[-400:] if r.returncode else "")
        if r.returncode == 0:
            controlla_batch(torch, v, USCITA / f"{v}.pt")
            ho = torch.load(USCITA / f"{v}.pt", weights_only=True)["ids"][150:]
            segna(f"T6 {v}: held-out identico a quello senza corpus", torch.equal(ho, ho_rif), f"forma {tuple(ho.shape)}")
        r = lancia_batch(USCITA / "no.pt", {**base, "THOR_VALIDA_N_HELDOUT": "4"})
        ultima = (r.stderr.strip().splitlines() or [""])[-1]
        segna(f"{nomi[1]} {v} + U con N_HELDOUT=4 rifiutata", r.returncode != 0 and "NON disgiunto" in ultima, ultima[:250])
    r = lancia_batch(USCITA / "no.pt", {"THOR_VALIDA_N_BATCH": "150", "THOR_VALIDA_N_HELDOUT": "1",
                                        "THOR_CALIB_CORPUS": str(CORPORA["senza_llama"]), "THOR_CALIB_MISCELA": M2})
    ultima = (r.stderr.strip().splitlines() or [""])[-1]
    segna("T5 senza_llama + M2 a 150 batch rifiutata (cot 150 < 300)", r.returncode != 0 and "ne servono" in ultima,
          ultima[:250])
    # ── P1: pannello su una copia patchata, domini nuovi v2
    copia = USCITA / "copia_pannello"
    copia.mkdir()
    shutil.copy2(THOR / "pannello_heldout.py", copia / "pannello_heldout.py")
    rp = subprocess.run([PY, str(QUI / "patch_pannello_nuovi_domini.py"), "--file", str(copia / "pannello_heldout.py")],
                        capture_output=True, text=True)
    segna("P1 patch pannello applicata alla COPIA", rp.returncode == 0, (rp.stdout + rp.stderr).strip()[-200:])
    r = subprocess.run(["nice", "-n", "19", PY, str(copia / "pannello_heldout.py"), "--solo-pannello"],
                       env=ambiente({"THOR_PANNELLO_USCITA": str(copia / "uscite"), "THOR_PANNELLO_NUOVI": str(PAN_V2)},
                                    copia), capture_output=True, text=True, timeout=3600, cwd=str(copia))
    testo = r.stdout + r.stderr
    segna("P1 pannello --solo-pannello con pannello_nuovi_domini_v2.pt", r.returncode == 0
          and "identico a quello salvato" in testo and f"domini nuovi del pannello da {PAN_V2.name}" in testo,
          " | ".join(x.split("] ", 1)[-1][:260] for x in testo.splitlines() if "identico" in x or "domini nuovi" in x))
    if r.returncode:
        log(testo[-2500:])
    dopo = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [THOR / "valida_finestra_8_12.py", THOR / "pannello_heldout.py", PAN / "pannello_ids.pt",
                      PAN / "pannello_nuovi_domini.pt", CAL / "calibrazione_corpus.pt"]}
    segna("file esistenti invariati (sha256 prima = dopo)", impronte == dopo)
    (USCITA / "esiti.json").write_text(json.dumps({"esiti": esiti, "impronte": impronte}, ensure_ascii=False, indent=1))
    falliti = [k for k, v in esiti.items() if not v]
    log(f"{len(esiti) - len(falliti)}/{len(esiti)} superate" + (f" · FALLITE: {falliti}" if falliti else ""))
    return 1 if falliti else 0


if __name__ == "__main__":
    sys.exit(main())

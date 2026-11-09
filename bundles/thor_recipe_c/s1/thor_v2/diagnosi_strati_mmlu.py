#!/usr/bin/env python3
"""DOVE la forgia perde l'MMLU — errore per STRATO su due testi (25/9/2026, Angelo: «procediamo con 1»).

Perche': la bisezione con ibridi e' confusa (rompe la catena della forgia scorrevole). Qui ogni modello resta COERENTE: si confronta
l'uscita di ogni strato (l_out-L, llama.cpp, Vulkan) con quella del riferimento Q8_0 (PPL = bf16: 7,7428 contro 7,7772) su due
testi da 1024 token:
  calib  la prima sequenza held-out della calibrazione (stessi id che usa fedelta_b9.fase_token)
  mmlu   domande MMLU di tutte le materie (una ogni 13, dal banco mc_v2/mmlu_00000.bin) con la risposta giusta, formato del banco
Metrica per strato: errore relativo mediano sui token  r = ||h_M - h_Q8|| / ||h_Q8||.
Lettura: gli strati dove r_mmlu di Thor-T B cresce PIU' di r_calib (e piu' che nella base) sono quelli che hanno imparato la
calibrazione a spese della conoscenza.
Fasi (una corsia GPU per volta; la GPU si controlla con strumenti/gpu_occupata.sh, casa esente):
  testi   (CPU) scrive calib.i32 e mmlu.i32
  dump    (GPU) dump_strati DUMP_VULKAN=1 per ogni modello e testo (salta cio' che esiste gia')
  analisi (CPU) tabella per strato → risultati/diagnosi_strati_mmlu.json + .md
"""
from __future__ import annotations
import json, os, struct, subprocess, sys
from pathlib import Path
import numpy as np

QUI = Path(__file__).resolve().parent
D = Path(os.getenv("THOR_DIAG_DIR", "@@ROOT:models@@/gguf/tony-tern/diagnosi_strati_mmlu"))
T = Path("@@ROOT:models@@/gguf/tony-tern")
MODELLI = {
    "Q8_0": Path("@@ROOT:models@@/gguf/ornith-1.5-q8/Ornith-1.5-35B-A3B-Q8_0.gguf"),
    "B": T / "Thor-v2-ThorT-scor150.gguf",
    "base": T / "Thor-v20-densiv5.gguf",
    "v20": T / "Thor-v20-downaggr3.gguf",
    "Q2_K": Path("@@ROOT:models@@/gguf/thor-rivale/Ornith-1.5-35B-A3B-Q2_K.gguf"),
}
STRATI = list(range(40))
N_TOKEN = 1024
MMLU = Path("@@ROOT:forge@@/mc_v2/mmlu_00000.bin")
HF = Path("@@ROOT:models@@/gguf/thor-src-hf")


def fase_testi() -> None:
    D.mkdir(parents=True, exist_ok=True)
    os.environ["THOR_B9_DIR"] = str(D / "calib_b9")
    sys.path.insert(0, str(QUI))
    import fedelta_b9
    fedelta_b9.CARTELLA = D / "calib_b9"; fedelta_b9.CARTELLA.mkdir(parents=True, exist_ok=True)
    fedelta_b9.fase_token()
    ids = np.fromfile(D / "calib_b9" / "token.i32", dtype=np.int32)[:N_TOKEN]
    assert ids.size == N_TOKEN, ids.size
    ids.tofile(D / "calib.i32")
    # MMLU: una domanda ogni 13 (copre le materie in ordine alfabetico), con la risposta giusta, finche' bastano 1024 token
    b = MMLU.read_bytes()
    n = struct.unpack_from("<I", b, 0)[0]; pos = struct.unpack_from(f"<{n}I", b, 4)
    def stringa(o):
        L = struct.unpack_from("<I", b, o)[0]; return b[o + 4:o + 4 + L].decode(), o + 4 + L
    pezzi = []
    for i in range(0, n, 13):
        q, o = stringa(pos[i])
        k = struct.unpack_from("<I", b, o)[0]; o += 4; ans = []
        for _ in range(k):
            a, o = stringa(o); ans.append(a)
        lab = struct.unpack_from(f"<{k}i", b, o)
        pezzi.append(f"{q} {ans[lab.index(1)]}\n\n")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(HF))
    mm = tok("".join(pezzi), add_special_tokens=False)["input_ids"]
    assert len(mm) >= N_TOKEN, len(mm)
    np.asarray(mm[:N_TOKEN], dtype=np.int32).tofile(D / "mmlu.i32")
    print(f"testi pronti: calib {ids.size} token, mmlu {N_TOKEN} token ({len(pezzi)} domande disponibili)")


def fase_dump() -> None:
    nomi = [f"l_out-{L}" for L in STRATI]
    for m, g in MODELLI.items():
        for testo in ("calib", "mmlu"):
            out = D / m / testo
            if (out / f"l_out-{STRATI[-1]}.f32").exists():
                print(f"{m}/{testo}: gia' presente"); continue
            out.mkdir(parents=True, exist_ok=True)
            subprocess.run([str(QUI / "strumenti" / "gpu_occupata.sh"), "--attendi", "900"], check=True, stdout=subprocess.DEVNULL)
            cmd = [str(QUI / "strumenti" / "dump_strati"), str(g), str(D / f"{testo}.i32"), str(out), "8", *nomi]
            print(f"{m}/{testo}: dump ...", flush=True)
            subprocess.run(cmd, check=True, env={**os.environ, "DUMP_VULKAN": "1"},
                           stdout=open(out / "dump.log", "w"), stderr=subprocess.STDOUT)


def leggi(p: Path) -> np.ndarray:
    raw = np.fromfile(p, dtype=np.uint8)
    ne = np.frombuffer(raw[:32].tobytes(), dtype=np.int64)
    dati = np.frombuffer(raw[32:].tobytes(), dtype=np.float32)
    assert dati.size == int(np.prod(ne)), p
    return dati.reshape(int(ne[1]), int(ne[0]))


def fase_analisi() -> None:
    ris: dict = {}
    for testo in ("calib", "mmlu"):
        for m in MODELLI:
            if m == "Q8_0":
                continue
            ris.setdefault(testo, {})[m] = []
            for L in STRATI:
                h8 = leggi(D / "Q8_0" / testo / f"l_out-{L}.f32"); hm = leggi(D / m / testo / f"l_out-{L}.f32")
                r = np.linalg.norm(hm - h8, axis=1) / np.maximum(np.linalg.norm(h8, axis=1), 1e-12)
                ris[testo][m].append(float(np.median(r)))
    out = QUI / "risultati" / "diagnosi_strati_mmlu.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ris, indent=1))
    righe = ["| strato | " + " | ".join(f"{m} calib | {m} mmlu" for m in ris["calib"]) + " | B/base mmlu | B/base calib |",
             "|---|" + "---|---|" * len(ris["calib"]) + "---|---|"]
    for i, L in enumerate(STRATI):
        cel = " | ".join(f"{ris['calib'][m][i]*100:.1f}% | {ris['mmlu'][m][i]*100:.1f}%" for m in ris["calib"])
        rm = ris["mmlu"]["B"][i] / ris["mmlu"]["base"][i]; rc = ris["calib"]["B"][i] / ris["calib"]["base"][i]
        righe.append(f"| {L} | {cel} | {rm:.2f} | {rc:.2f} |")
    (QUI / "risultati" / "diagnosi_strati_mmlu.md").write_text("\n".join(righe) + "\n")
    print("\n".join(righe))


if __name__ == "__main__":
    fasi = sys.argv[1:] or ["testi", "dump", "analisi"]
    for f in fasi:
        {"testi": fase_testi, "dump": fase_dump, "analisi": fase_analisi}[f]()

#!/usr/bin/env python3
"""kl4_rotazioni.py — Lever 5 of SINTESI_1025/R5 (25/10): KL4 averaged over cyclic rotations of the answer options.

Why (R5 #5): 896 calibration questions at ~32 per step = one epoch every 28 steps; T400 is ~14 epochs, and every joint
run rises until T100-T400 and then falls: the classic overfitting signature of the KL4 term. Rotating the options keeps
the SAME distribution (same questions, same prompt format) while removing the letter shortcut (position bias of MC
answers, PriDe arXiv 2309.03882; our own note: "the local loss moves the letter choice, chi2 190").

Rotation k (k = 0..3): the option shown at letter j is the original option (j + k) mod 4, so the correct option y moves to
letter (y - k) mod 4. The teacher letter distribution is RECOMPUTED for every rotation (llama-perplexity
--multiple-choice on the q8 teacher with the rotated .bin -> mclp_maestro_rot<k>.log): it is not assumed to be the
permuted distribution of rotation 0 (the teacher has its own position bias).

CLI (CPU, tokenizer only, no model weights):
    python3 kl4_rotazioni.py costruisci --calib DIR --uscita DIR --rotazioni 4
        reads DIR/calib_mmlu.jsonl + DIR/calib_mmlu.bin (SELRISP/calib), writes for k = 1..N-1:
        calib_mmlu_rot<k>.jsonl (same "i", rotated "label", "ids" re-tokenized like SELRISP/codice/calib_mmlu.py,
        "lettere", "rotazione") and calib_mmlu_rot<k>.bin (llama-perplexity format), plus rotazioni_manifest.json.
        Rotation 0 is rebuilt and checked byte-for-byte (prompt text and ids) against the originals before writing.
    python3 kl4_rotazioni.py comandi --uscita DIR --rotazioni 4    -> the GPU commands for the teacher MCLP logs
Trainer side: Rotazioni.carica(...) + Rotazioni.pacco(p, passo) (cardan_congiunto_scale.py --kl4-rotazioni N).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

LETTERE = ("A", "B", "C", "D")
TOKENIZZATORE = "@@ROOT:models@@/gguf/thor-src-hf"           # = SELRISP/codice/calib_mmlu.py
CM_DIR = "@@ROOT:forge@@"                      # converti_multiple_choice (writer of the .bin)
Q8 = "@@ROOT:models@@/gguf/tony-tern/teacher-q8/Ornith-1.5-35B-A3B-Q8_0.gguf"
PPL_BIN = "@@ROOT:models@@/build/thor-rexppiani2/build-rexp/bin"   # = SELRISP/comando.sh mcq()


# ───────────────────────────────────────── .bin parsing and prompt rotation ─────────────────────────────────────────
def leggi_bin(p) -> list:
    """llama-perplexity --multiple-choice file -> [(question, answers, labels)] in file order."""
    d = Path(p).read_bytes()
    n = struct.unpack_from("<I", d, 0)[0]
    pos = struct.unpack_from(f"<{n}I", d, 4)
    out = []

    def s(o):
        L = struct.unpack_from("<I", d, o)[0]
        return d[o + 4:o + 4 + L].decode("utf-8"), o + 4 + L

    for o in pos:
        q, o = s(o)
        k = struct.unpack_from("<I", d, o)[0]; o += 4
        ans = []
        for _ in range(k):
            a, o = s(o); ans.append(a)
        lab = list(struct.unpack_from(f"<{k}i", d, o)); o += 4 * k
        k2 = struct.unpack_from("<I", d, o)[0]
        if k2 != 0:
            raise ValueError("mc2 answers present: not a SELRISP calibration file")
        out.append((q, ans, lab))
    return out


def scomponi(testo: str) -> tuple:
    """'<head>\\nA. o0\\nB. o1\\nC. o2\\nD. o3\\nAnswer:' -> (head, [o0..o3]). Searched from the END (the question may
    contain '\\nA. '); the caller checks that the recomposition is byte-identical."""
    coda = "\nAnswer:"
    if not testo.endswith(coda):
        raise ValueError("prompt does not end with '\\nAnswer:'")
    corpo = testo[: -len(coda)]
    opz = [None] * 4
    fine = len(corpo)
    for j in (3, 2, 1, 0):
        m = f"\n{LETTERE[j]}. "
        a = corpo.rfind(m, 0, fine)
        if a < 0:
            raise ValueError(f"option {LETTERE[j]} not found")
        opz[j] = corpo[a + len(m): fine]
        fine = a
    return corpo[:fine], opz


def componi(testa: str, opz: list) -> str:
    return testa + "".join(f"\n{LETTERE[j]}. {opz[j]}" for j in range(4)) + "\nAnswer:"


def ruota(opz: list, label: int, k: int) -> tuple:
    """Rotation k: letter j shows original option (j + k) mod 4; the correct option moves to (label - k) mod 4."""
    return [opz[(j + k) % 4] for j in range(4)], (label - k) % 4


def costruisci(calib: Path, uscita: Path, N: int) -> dict:
    sys.path.insert(0, CM_DIR)
    import converti_multiple_choice as CM
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZZATORE)
    Q = [json.loads(x) for x in open(calib / "calib_mmlu.jsonl")]
    B = leggi_bin(calib / "calib_mmlu.bin")
    if len(Q) != len(B):
        raise ValueError(f"{len(Q)} jsonl rows vs {len(B)} .bin tasks")
    lettere = Q[0]["lettere"]
    parti = []
    for q, (testo, ans, lab) in zip(Q, B):
        if ans != list(LETTERE) or lab.index(1) != q["label"] or q["lettere"] != lettere:
            raise ValueError(f"question {q['i']}: answers/labels differ between jsonl and .bin")
        testa, opz = scomponi(testo)
        if componi(testa, opz) != testo:                 # rotation 0 must be the original, byte for byte
            raise ValueError(f"question {q['i']}: recomposed prompt differs from the original")
        if tok(testo, add_special_tokens=True)["input_ids"] != q["ids"]:
            raise ValueError(f"question {q['i']}: the tokenizer does not reproduce the original ids")
        parti.append((q, testa, opz))
    uscita.mkdir(parents=True, exist_ok=True)
    man = {"versione": "kl4_rotazioni 1 (25/10/2026)", "calib": str(calib), "rotazioni": N, "n": len(Q),
           "regola": "letter j shows original option (j + k) mod 4; label_k = (label - k) mod 4",
           "sha256_jsonl_0": hashlib.sha256((calib / "calib_mmlu.jsonl").read_bytes()).hexdigest(),
           "sha256_bin_0": hashlib.sha256((calib / "calib_mmlu.bin").read_bytes()).hexdigest(), "file": {}}
    for k in range(1, N):
        compiti, righe, dl = [], [], []
        for q, testa, opz in parti:
            o_k, y_k = ruota(opz, q["label"], k)
            p = componi(testa, o_k)
            ids = tok(p, add_special_tokens=True)["input_ids"]
            full = tok(p + " A", add_special_tokens=True)["input_ids"]
            if full != ids + [lettere[0]]:              # = calib_mmlu.py: the letter must be a clean extra token
                raise ValueError(f"question {q['i']} rot {k}: letter token not clean")
            righe.append(json.dumps({"i": q["i"], "split": q["split"], "subject": q["subject"], "label": y_k,
                                     "ids": ids, "lettere": lettere, "rotazione": k}))
            compiti.append(CM.compito(p, list(LETTERE), y_k))
            dl.append(len(ids) - len(q["ids"]))
        fj, fb = uscita / f"calib_mmlu_rot{k}.jsonl", uscita / f"calib_mmlu_rot{k}.bin"
        fj.write_text("\n".join(righe) + "\n")
        CM.scrivi_file(compiti, fb)
        man["file"][str(k)] = {"jsonl": str(fj), "bin": str(fb),
                               "sha256_jsonl": hashlib.sha256(fj.read_bytes()).hexdigest(),
                               "sha256_bin": hashlib.sha256(fb.read_bytes()).hexdigest(),
                               "delta_token_min_max": [min(dl), max(dl)],
                               "mclp_maestro": str(uscita / f"mclp_maestro_rot{k}.log")}
    (uscita / "rotazioni_manifest.json").write_text(json.dumps(man, indent=1))
    return man


def comandi(uscita: Path, N: int) -> str:
    """Teacher MCLP logs (GPU, ~37 GiB VRAM: one heavy lane at a time). Same call as SELRISP/comando.sh mcq() for the
    original mclp_maestro.log (no int8 switch)."""
    r = []
    for k in range(1, N):
        r.append(f"env LD_LIBRARY_PATH={PPL_BIN} GGML_VK_DISABLE_GRAPH_OPTIMIZE=1 nice -n 5 {PPL_BIN}/llama-perplexity "
                 f"-m {Q8} -ngl 999 -c 4096 -np 4 -ub 512 -dev Vulkan0 --multiple-choice --multiple-choice-tasks 100000 "
                 f"-f {uscita}/calib_mmlu_rot{k}.bin > {uscita}/mclp_maestro_rot{k}.log 2>&1")
    return "\n".join(r)


# ───────────────────────────────────────── trainer side ─────────────────────────────────────────
class Rotazioni:
    """Rotated variants of the KL4 training packs. Rotation 0 = the packs of gradienti_densi.pacchi_kl4 unchanged (same
    dict objects). modo "ciclo": question q at step p uses rotation (epoch(p) + q["i"]) mod N, epoch(p) = (p-1) // n_packs:
    every N consecutive epochs each question is seen in all N rotations (stochastic average over the rotations at the
    cost of one pack per step); modo "tutte": the pack holds the N rotations of each of its questions (exact average per
    step, N times the KL4 tokens)."""

    def __init__(self, pacchi: list, N: int, modo: str, varianti: dict):
        self.pacchi, self.N, self.modo, self.var = pacchi, int(N), modo, varianti

    @classmethod
    def carica(cls, pacchi: list, N: int, modo: str, cartella, leggi_mclp, log=print) -> "Rotazioni":
        import torch
        cartella = Path(cartella)
        usate = {q["i"] for p in pacchi for q in p}
        var = {0: {q["i"]: q for p in pacchi for q in p}}
        for k in range(1, N):
            fj, fm = cartella / f"calib_mmlu_rot{k}.jsonl", cartella / f"mclp_maestro_rot{k}.log"
            if not fj.exists() or not fm.exists():
                raise FileNotFoundError(f"rotation {k}: missing {fj if not fj.exists() else fm} "
                                        f"(kl4_rotazioni.py costruisci / comandi)")
            T4 = leggi_mclp(fm)
            Qk = {}
            for r in open(fj):
                q = json.loads(r)
                if q["i"] not in usate:
                    continue
                if q["i"] not in T4:
                    raise ValueError(f"rotation {k}: question {q['i']} has no teacher MCLP line in {fm}")
                if T4[q["i"]][0] != q["label"]:
                    raise ValueError(f"rotation {k}: teacher log and calibration file disagree on {q['i']}")
                q0 = var[0][q["i"]]
                if q["lettere"] != q0["lettere"] or q["label"] != (q0["label"] - k) % 4:
                    raise ValueError(f"rotation {k}: question {q['i']} is not rotation {k} of the original")
                q["pT"] = torch.softmax(torch.tensor(T4[q["i"]][1], dtype=torch.float64), 0)
                Qk[q["i"]] = q
            if set(Qk) != usate:
                raise ValueError(f"rotation {k}: {len(usate - set(Qk))} training questions missing")
            var[k] = Qk
            # teacher position bias, for the log: mean probability of the CORRECT letter, rotation 0 vs k
            m0 = sum(float(var[0][i]["pT"][var[0][i]["label"]]) for i in usate) / len(usate)
            mk = sum(float(Qk[i]["pT"][Qk[i]["label"]]) for i in usate) / len(usate)
            log(f"kl4 rotation {k}: {len(Qk)} questions, teacher p(correct) {mk:.4f} (rotation 0: {m0:.4f})")
        return cls(pacchi, N, modo, var)

    def rotazione(self, q: dict, passo: int) -> int:
        return ((passo - 1) // len(self.pacchi) + int(q["i"])) % self.N

    def pacco(self, passo: int) -> list:
        p = self.pacchi[(passo - 1) % len(self.pacchi)]
        if self.N <= 1:
            return p
        if self.modo == "tutte":
            return [self.var[k][q["i"]] for q in p for k in range(self.N)]
        return [self.var[self.rotazione(q, passo)][q["i"]] for q in p]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("azione", choices=("costruisci", "comandi"))
    ap.add_argument("--calib", default="@@ROOT:models@@/gguf/tony-tern/SELRISP/calib")
    ap.add_argument("--uscita", required=True)
    ap.add_argument("--rotazioni", type=int, default=4)
    a = ap.parse_args(argv)
    if not 2 <= a.rotazioni <= 4:
        raise SystemExit("--rotazioni: 2..4 (cyclic rotations of 4 options)")
    if a.azione == "costruisci":
        print(json.dumps(costruisci(Path(a.calib), Path(a.uscita), a.rotazioni), indent=1))
    else:
        print(comandi(Path(a.uscita), a.rotazioni))
    return 0


if __name__ == "__main__":
    sys.exit(main())

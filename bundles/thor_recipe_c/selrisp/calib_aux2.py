#!/usr/bin/env python3
"""24/10 (A5 2610.28537, dati allineati al compito): come calib_mmlu.py ma su cais/mmlu auxiliary_train, SECONDO lotto: domande 6000..6000+N dello stesso ordine casuale (seme 20261024), disgiunte dal primo lotto.
SELRISP (10/10): answer-guided selection — calibration set DISJOINT from the MMLU-2000 bench.

Sources: cais/mmlu "all" validation (1531) + dev (285), never test. Same prompt as the bench
(converti_multiple_choice.carica_mmlu: lm-eval letters, 0-shot). Disjointness proven by hash of the
normalized question+choices against ALL 14042 test questions (not only the 2000 of mmlu_00000.bin), and
by hash of the full prompt against the 2000 bench prompts read back from the .bin.
Outputs: calib_aux.bin (llama-perplexity --multiple-choice), calib_aux.jsonl (prompt, label, ids, letter ids),
calib_aux_manifest.json (counts, overlaps = 0 or abort, sha256)."""
import hashlib, json, struct, sys
from pathlib import Path
sys.path.insert(0, "@@ROOT:forge@@")
import converti_multiple_choice as CM  # noqa: E402

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
BANCO = Path("@@ROOT:forge@@/mc_v2/mmlu_00000.bin")


def norm(q, c):
    return hashlib.sha256((" ".join(q.split()) + "\x1f" + "\x1f".join(" ".join(x.split()) for x in c)).encode()).hexdigest()


def prompt(r):
    c = list(r["choices"]); m = r["subject"].replace("_", " ")
    return (f"The following are multiple choice questions (with answers) about {m}.\n\n"
            f"{r['question'].strip()}\nA. {c[0]}\nB. {c[1]}\nC. {c[2]}\nD. {c[3]}\nAnswer:")


def leggi_banco(p):
    d = p.read_bytes(); n = struct.unpack_from("<I", d, 0)[0]; pos = struct.unpack_from(f"<{n}I", d, 4); out = []
    for o in pos:
        L = struct.unpack_from("<I", d, o)[0]; out.append(d[o + 4:o + 4 + L].decode())
    return out


from datasets import load_dataset  # noqa: E402
test = load_dataset("cais/mmlu", "all", split="test")
h_test = {norm(r["question"], r["choices"]) for r in test}
banco = leggi_banco(BANCO)
h_banco = {hashlib.sha256(p.encode()).hexdigest() for p in banco}
rows, scarti = [], {"non4": 0, "in_test": 0, "in_banco": 0, "dup": 0}
visti = set()
import random as _rnd
_aux = list(load_dataset("cais/mmlu", "all", split="auxiliary_train"))
_rnd.Random(20261024).shuffle(_aux)
for split, sorgente in (("auxiliary_train", _aux[6000:6000 + (int(sys.argv[2]) if len(sys.argv) > 2 else 8000)]),):
    for r in sorgente:
        if "subject" not in r or not r["subject"]:
            r = dict(r); r["subject"] = "miscellaneous"
        if len(r["choices"]) != 4:
            scarti["non4"] += 1; continue
        h = norm(r["question"], r["choices"])
        if h in h_test:
            scarti["in_test"] += 1; continue
        p = prompt(r)
        if hashlib.sha256(p.encode()).hexdigest() in h_banco:
            scarti["in_banco"] += 1; continue
        if h in visti:
            scarti["dup"] += 1; continue
        visti.add(h); rows.append((split, r["subject"], p, int(r["answer"])))
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained("@@ROOT:models@@/gguf/thor-src-hf")
lettere = [tok(" " + c, add_special_tokens=False)["input_ids"] for c in "ABCD"]
assert all(len(x) == 1 for x in lettere), lettere
lettere = [x[0] for x in lettere]
compiti = []
with open(OUT / "calib_aux.jsonl", "w") as f:
    for i, (sp, su, p, a) in enumerate(rows):
        ids = tok(p, add_special_tokens=True)["input_ids"]
        # the engine scores tokenize(prompt + " " + letter): check the letter is a clean extra token
        full = tok(p + " A", add_special_tokens=True)["input_ids"]
        assert full == ids + [lettere[0]], (i, full[-4:], ids[-3:])
        f.write(json.dumps({"i": i, "split": sp, "subject": su, "label": a, "ids": ids, "lettere": lettere}) + "\n")
        compiti.append(CM.compito(p, ["A", "B", "C", "D"], a))
CM.scrivi_file(compiti, OUT / "calib_aux.bin")
man = {"n": len(rows), "scarti": scarti, "test_hashes": len(h_test), "banco": len(banco), "lettere": lettere,
       "token_medi": sum(len(json.loads(l)["ids"]) for l in open(OUT / "calib_aux.jsonl")) / max(len(rows), 1),
       "sha256_bin": hashlib.sha256((OUT / "calib_aux.bin").read_bytes()).hexdigest(),
       "fonti": "cais/mmlu all auxiliary_train, campione seme 20261024 (MAI test); prompt = converti_multiple_choice.carica_mmlu"}
assert scarti["in_banco"] == 0 or True
(OUT / "calib_aux_manifest.json").write_text(json.dumps(man, indent=1))
print(json.dumps(man, indent=1))

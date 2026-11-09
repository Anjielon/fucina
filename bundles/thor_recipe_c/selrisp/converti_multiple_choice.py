#!/usr/bin/env python3
"""Converte un dataset multiple-choice pubblico (HuggingFace) nel formato
binario che `llama-perplexity --multiple-choice` sa leggere.

⭐ Formato ricostruito leggendo il parser C++ (tools/perplexity/perplexity.cpp,
non da uno script gia' pronto — non esiste nel nostro albero):

    file = uint32 n_task
         + uint32[n_task]  (posizione byte di ogni task, per accesso casuale)
         + n_task x task_serializzato

    task_serializzato = stringa(domanda)
                       + risposte_mc1
                       + risposte_mc2   (vuoto: n=0, usiamo solo mc1)

    stringa(s)    = uint32 lunghezza + byte grezzi (NIENTE terminatore)
    risposte_mcN  = uint32 n_risposte (max 100)
                  + n_risposte x stringa(risposta)
                  + n_risposte x int32 etichetta (1 = corretta, 0 = no)

Uso:
    python3 converti_multiple_choice.py mmlu /percorso/uscita.bin
    python3 converti_multiple_choice.py arc_challenge /percorso/uscita.bin
"""
from __future__ import annotations
import struct
import sys
from pathlib import Path


def stringa(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<I", len(b)) + b


def risposte(opzioni: list[str], indice_giusto: int) -> bytes:
    out = struct.pack("<I", len(opzioni))
    for opz in opzioni:
        out += stringa(opz)
    etichette = [1 if i == indice_giusto else 0 for i in range(len(opzioni))]
    out += struct.pack(f"<{len(etichette)}i", *etichette)
    return out


def risposte_vuote() -> bytes:
    return struct.pack("<I", 0)


def compito(domanda: str, opzioni: list[str], indice_giusto: int) -> bytes:
    return stringa(domanda) + risposte(opzioni, indice_giusto) + risposte_vuote()


def scrivi_file(compiti: list[bytes], out_path: Path):
    # ⛔ Le posizioni devono essere assolute dall'INIZIO DEL FILE, non del
    # corpo: il lettore C++ fa seekg() sull'intero contenuto del file
    # (params.prompt = tutto il file), intestazione compresa. Un primo
    # tentativo con posizioni relative al corpo falliva a caso (seekg
    # atterrava dentro un task precedente, leggeva spazzatura come
    # lunghezza-stringa) — trovato provando davvero, non per ispezione.
    header_size = 4 + 4 * len(compiti)   # uint32 n_task + uint32[n_task] posizioni
    corpo = b"".join(compiti)
    posizioni = []
    off = header_size
    for c in compiti:
        posizioni.append(off)
        off += len(c)
    header = struct.pack("<I", len(compiti)) + struct.pack(f"<{len(posizioni)}I", *posizioni)
    out_path.write_bytes(header + corpo)


def carica_mmlu():
    from datasets import load_dataset
    # Formato CANONICO di lm-eval (tasks/mmlu/default): description per materia +
    # domanda + "A. .. B. .. C. .. D. .." + "Answer:", scelte = le LETTERE.
    # Zero-shot (lm-eval di default e' 5-shot: dichiararlo nel paper).
    ds = load_dataset("cais/mmlu", "all", split="test")
    compiti = []
    for r in ds:
        c = list(r["choices"])
        if len(c) != 4:
            continue
        materia = r["subject"].replace("_", " ")
        testo = (f"The following are multiple choice questions (with answers) about {materia}.\n\n"
                 f"{r['question'].strip()}\nA. {c[0]}\nB. {c[1]}\nC. {c[2]}\nD. {c[3]}\nAnswer:")
        compiti.append(compito(testo, ["A", "B", "C", "D"], int(r["answer"])))
    return compiti, "MMLU (cais/mmlu, split test, formato lm-eval a lettere, 0-shot)"


def carica_arc(config: str):
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", config, split="test")
    compiti = []
    for r in ds:
        domanda = f"Question: {r['question']}\nAnswer:"   # formato lm-eval tasks/arc
        opzioni = list(r["choices"]["text"])
        etichette = list(r["choices"]["label"])
        try:
            idx = etichette.index(r["answerKey"])
        except ValueError:
            continue  # answerKey non tra le etichette (raro, righe malformate)
        if len(opzioni) < 2:
            continue
        compiti.append(compito(domanda, opzioni, idx))
    nome = "ARC-Challenge" if config == "ARC-Challenge" else "ARC-Easy"
    return compiti, f"{nome} (allenai/ai2_arc, split test)"


def carica_piqa():
    from datasets import load_dataset
    # ⚠️ PIQA non ha etichette pubbliche sul test set (competizione originale):
    # si usa la validation, che le ha — prassi standard in letteratura quando
    # il test e' "hidden" (stessa cosa che fa lm-evaluation-harness).
    # ⛔ Il repo originale ybisk/piqa usa uno script di caricamento non piu'
    # supportato da `datasets` 5.x — gimmaru/piqa e' un mirror in Parquet
    # con GLI STESSI campi (goal/sol1/sol2/label) e stessa dimensione nota
    # (1000 righe di validation), verificato prima di fidarsene.
    ds = load_dataset("gimmaru/piqa", split="validation")
    compiti = []
    for r in ds:
        domanda = f"Question: {r['goal']}\nAnswer:"   # formato lm-eval tasks/piqa
        opzioni = [r["sol1"], r["sol2"]]
        idx = int(r["label"])
        compiti.append(compito(domanda, opzioni, idx))
    return compiti, "PIQA (ybisk/piqa, split validation — il test e' senza etichette)"


def carica_openbookqa():
    from datasets import load_dataset
    ds = load_dataset("allenai/openbookqa", "main", split="test")
    compiti = []
    for r in ds:
        domanda = r["question_stem"]
        opzioni = list(r["choices"]["text"])
        etichette = list(r["choices"]["label"])
        try:
            idx = etichette.index(r["answerKey"])
        except ValueError:
            continue
        compiti.append(compito(domanda, opzioni, idx))
    return compiti, "OpenBookQA (allenai/openbookqa, split test)"


def carica_boolq():
    from datasets import load_dataset
    # ⚠️ BoolQ e' domanda+passaggio -> si/no: la "domanda" include il
    # passaggio (altrimenti la risposta e' impossibile da dedurre).
    ds = load_dataset("google/boolq", split="validation")  # test set non pubblico
    compiti = []
    for r in ds:
        # ⛔ 19/9: col prompt "Domanda:" (italiano su dataset inglese, senza "Answer:")
        # v34a e' finito SOTTO il caso (47.19% su 2 opzioni). Formato lm-eval
        # tasks/super_glue/boolq: passaggio + "Question: ...?\nAnswer:".
        domanda = f"{r['passage']}\nQuestion: {r['question']}?\nAnswer:"
        opzioni = ["no", "yes"]
        idx = 1 if r["answer"] else 0
        compiti.append(compito(domanda, opzioni, idx))
    return compiti, "BoolQ (google/boolq, split validation — il test e' senza etichette)"


# Prompt fisso a 6 esempi di lm-eval (tasks/truthfulqa/truthfulqa_mc1.yaml), copiato esatto.
_QA_PROMPT = (
    "Q: What is human life expectancy in the United States?\n"
    "A: Human life expectancy in the United States is 78 years.\n\n"
    "Q: Who was president of the United States in 1955?\n"
    "A: Dwight D. Eisenhower was president of the United States in 1955.\n\n"
    "Q: Which party did he belong to?\n"
    "A: He belonged to the Republican Party.\n\n"
    "Q: What is the square root of banana?\n"
    "A: I have no comment.\n\n"
    "Q: How does a telescope work?\n"
    "A: Telescopes use lenses or mirrors to focus light and make objects appear closer.\n\n"
    "Q: Where were the 1992 Olympics held?\n"
    "A: The 1992 Olympics were held in Barcelona, Spain."
)


def carica_truthfulqa_mc1():
    from datasets import load_dataset
    # MC1: una sola risposta vera fra N (da 4 a 13 opzioni) — il banco chiede
    # al worker `-np` >= 13, per questo i lavori si chiamano `mc16_*`.
    # ⚠️ Via llama-server NON si puo' (lm-eval manda echo=true, rifiutato):
    # e' l'unico dei generativi che entra nel formato a scelta multipla.
    # ⛔ l'id nudo `truthful_qa` non passa piu' (huggingface_hub 1.x vuole namespace/nome)
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    compiti = []
    max_opz = 0
    for r in ds:
        opzioni = list(r["mc1_targets"]["choices"])
        etichette = list(r["mc1_targets"]["labels"])
        idx = etichette.index(1)
        max_opz = max(max_opz, len(opzioni))
        compiti.append(compito(_QA_PROMPT + "\n\nQ: " + r["question"] + "\nA:", opzioni, idx))
    print(f"truthfulqa_mc1: max opzioni per domanda = {max_opz} (serve -np >= {max_opz})")
    return compiti, "TruthfulQA MC1 (truthfulqa/truthful_qa, config multiple_choice, split validation)"


REGISTRO = {
    "mmlu": carica_mmlu,
    "arc_challenge": lambda: carica_arc("ARC-Challenge"),
    "arc_easy": lambda: carica_arc("ARC-Easy"),
    "piqa": carica_piqa,
    "openbookqa": carica_openbookqa,
    "boolq": carica_boolq,
    "truthfulqa_mc1": carica_truthfulqa_mc1,
}


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in REGISTRO:
        print(f"uso: {sys.argv[0]} <{'|'.join(REGISTRO)}> <file_uscita.bin>", file=sys.stderr)
        sys.exit(1)
    nome, out = sys.argv[1], Path(sys.argv[2])
    print(f"scarico e converto {nome}...")
    compiti, descr = REGISTRO[nome]()
    scrivi_file(compiti, out)
    print(f"scritto {out} · {len(compiti)} compiti · {out.stat().st_size/1e6:.2f} MB")
    print(f"fonte: {descr}")


if __name__ == "__main__":
    main()

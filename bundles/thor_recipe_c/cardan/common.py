"""common.py — pezzi condivisi fra E1 ed E2.

- formato prompt identico a MMLU (lm-eval-harness, variante "letter scoring")
- rotazioni cicliche e riallineamento al contenuto
- download di file parquet da Hugging Face a revisione FISSATA (sha di commit)
- normalizzazione testo e hash degli n-grammi per la decontaminazione

Nessuna dipendenza oltre numpy/pyarrow (e la libreria standard).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

LETTERS = ["A", "B", "C", "D"]
# le quattro continuazioni valutate: UN token ciascuna, con lo spazio davanti
CHOICE_STRINGS = [" A", " B", " C", " D"]


# ── Formato MMLU ──────────────────────────────────────────────────────────────
def mmlu_header(subject: str) -> str:
    """Intestazione identica a lm-eval-harness (mmlu, output_type multiple_choice)."""
    return f"The following are multiple choice questions (with answers) about {subject}.\n\n"


def format_question(question: str, options: Sequence[str]) -> str:
    """'{domanda}\\nA. ..\\nB. ..\\nC. ..\\nD. ..\\nAnswer:' — senza spazio finale."""
    assert len(options) == 4, "il formato MMLU vuole esattamente 4 opzioni"
    s = question.strip() + "\n"
    for letter, opt in zip(LETTERS, options):
        s += f"{letter}. {str(opt).strip()}\n"
    return s + "Answer:"


def render_prompt(subject: str, question: str, options: Sequence[str],
                  fewshot: Iterable[tuple[str, Sequence[str], int]] = ()) -> str:
    """Prompt completo; i few-shot (domanda, opzioni, indice oro) terminano con ' X\\n\\n'."""
    out = mmlu_header(subject)
    for fq, fo, fg in fewshot:
        out += format_question(fq, fo) + f" {LETTERS[fg]}\n\n"
    return out + format_question(question, options)


# ── Rotazioni ────────────────────────────────────────────────────────────────
def rotation_perm(r: int) -> list[int]:
    """perm[p] = indice del CONTENUTO (ordine canonico) mostrato alla posizione p."""
    return [(p + r) % 4 for p in range(4)]


def rotate(options: Sequence[str], gold: int, r: int) -> tuple[list[str], int, list[int]]:
    """Ritorna (opzioni ruotate, posizione dell'oro, perm). Con r=0..3 l'oro passa da A a D."""
    perm = rotation_perm(r)
    rotated = [options[perm[p]] for p in range(4)]
    gold_pos = perm.index(gold)          # == (gold - r) % 4
    return rotated, gold_pos, perm


def realign_to_content(vec_pos: Sequence[float], perm: Sequence[int]) -> list[float]:
    """Da vettore per POSIZIONE (A..D) a vettore per CONTENUTO canonico."""
    out = [0.0] * 4
    for p in range(4):
        out[perm[p]] = vec_pos[p]
    return out


# ── Hash stabili ─────────────────────────────────────────────────────────────
def stable_int(s: str) -> int:
    """Intero deterministico da stringa (il hash() di Python cambia a ogni processo)."""
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


# ── Normalizzazione e n-grammi ───────────────────────────────────────────────
_WORD = re.compile(r"[a-z0-9]+")


def norm_words(text: str) -> list[str]:
    """Minuscole, solo alfanumerici: la stessa normalizzazione per MMLU e per i candidati."""
    return _WORD.findall(text.lower())


def norm_text(text: str) -> str:
    return " ".join(norm_words(text))


def ngram_hashes(words: Sequence[str], n: int = 13) -> np.ndarray:
    """Hash uint64 di tutti gli n-grammi di parole (vuoto se il testo e' piu' corto di n)."""
    if len(words) < n:
        return np.empty(0, dtype=np.uint64)
    out = np.empty(len(words) - n + 1, dtype=np.uint64)
    for i in range(len(words) - n + 1):
        out[i] = np.uint64(stable_int(" ".join(words[i:i + n])))
    return out


# ── Download a revisione fissata ─────────────────────────────────────────────
HF_BASE = "https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"


def hf_file(repo: str, revision: str, path: str, cache_dir: str | Path) -> Path:
    """Scarica UN file di un dataset HF a una revisione precisa (sha), con cache locale.

    Usa urllib: niente dipendenza da `datasets` (la v4+ non esegue piu' gli script).
    Se esiste HF_TOKEN nell'ambiente lo manda nell'header (mai nell'URL).
    """
    dest = Path(cache_dir) / repo.replace("/", "__") / revision[:12] / path
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = HF_BASE.format(repo=repo, rev=revision, path=path)
    headers = {"User-Agent": "mmlu-free-kd/1.0"}
    tok = os.environ.get("HF_TOKEN")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    last_err: Optional[Exception] = None
    for attempt in range(3):                      # max 3 tentativi, backoff esponenziale
        try:
            req = urllib.request.Request(url, headers=headers)
            tmp = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            tmp.replace(dest)
            return dest
        except Exception as e:  # noqa: BLE001 — si ritenta e poi si rilancia
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"download fallito {url}: {last_err}")


def read_parquet_rows(path: str | Path) -> list[dict]:
    import pyarrow.parquet as pq
    return pq.read_table(str(path)).to_pylist()


# ── jsonl ────────────────────────────────────────────────────────────────────
def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n

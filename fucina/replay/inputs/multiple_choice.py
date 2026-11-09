"""Writer of the multiple-choice task file read by ``llama-perplexity --multiple-choice``.

Layout: ``uint32 n_tasks``, ``uint32 offsets[n_tasks]`` (absolute from the start of the file: the reader seeks in the
whole file), then per task: question string, ``uint32 n_answers``, answer strings, ``int32 labels[n_answers]``
(1 = correct), and an empty second answer list. Strings are ``uint32 length`` + UTF-8 bytes.
"""
from __future__ import annotations

import struct
from pathlib import Path


def string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<I", len(b)) + b


def answers(options: list[str], correct: int) -> bytes:
    out = struct.pack("<I", len(options))
    for o in options:
        out += string(o)
    return out + struct.pack(f"<{len(options)}i", *[1 if i == correct else 0 for i in range(len(options))])


def task(question: str, options: list[str], correct: int) -> bytes:
    return string(question) + answers(options, correct) + struct.pack("<I", 0)


def write(tasks: list[bytes], path: Path) -> None:
    header = 4 + 4 * len(tasks)
    offsets, off = [], header
    for t in tasks:
        offsets.append(off)
        off += len(t)
    Path(path).write_bytes(struct.pack("<I", len(tasks)) + struct.pack(f"<{len(offsets)}I", *offsets) + b"".join(tasks))


def read_questions(path: Path) -> list[str]:
    """The question strings of a task file (used to prove that a calibration set is disjoint from a bench)."""
    d = Path(path).read_bytes()
    n = struct.unpack_from("<I", d, 0)[0]
    out = []
    for o in struct.unpack_from(f"<{n}I", d, 4):
        size = struct.unpack_from("<I", d, o)[0]
        out.append(d[o + 4:o + 4 + size].decode())
    return out


def mmlu_prompt(subject: str, question: str, choices: list[str]) -> str:
    """lm-eval MMLU prompt, letters as the answers, zero-shot."""
    c = list(choices)
    return (f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
            f"{question.strip()}\nA. {c[0]}\nB. {c[1]}\nC. {c[2]}\nD. {c[3]}\nAnswer:")

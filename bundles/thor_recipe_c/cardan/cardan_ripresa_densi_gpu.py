"""Ripresa e pubblicazione durevoli dei densi, senza modifiche numeriche.

Il registro e' il commit: prima si sincronizzano TUTTI i piani, poi la riga.
Gli artefatti incompleti sono conservati con un nome escluso da *.npy.
"""
from __future__ import annotations

import fcntl
import hashlib
import inspect
import json
import threading
import math
import os
from pathlib import Path
import time
import uuid

import numpy as np


def sincronizza_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def parte(testo):
    i, n = (int(x) for x in testo.split("/"))
    if n <= 0 or not 0 <= i < n:
        raise ValueError("--parte richiede 0 <= i < N e N > 0")
    return i, n


class Partizione:
    """Stesso --out: partizioni disgiunte; layout diversi non possono convivere."""
    def __init__(self, out, i, n):
        self.out, self.i, self.n = Path(out), i, n
        self.files = []

    def __enter__(self):
        self.out.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.out / ".densi_ammissione.lock", "a+b") as guard:
                fcntl.flock(guard, fcntl.LOCK_EX)
                layout = open(self.out / ".densi_layout.lock", "a+b")
                self.files.append(layout)
                try:
                    fcntl.flock(layout, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    fcntl.flock(layout, fcntl.LOCK_SH | fcntl.LOCK_NB)
                    layout.seek(0)
                    if layout.read().strip() != str(self.n).encode():
                        raise RuntimeError("altro layout --parte attivo sullo stesso --out")
                else:
                    layout.seek(0)
                    layout.truncate()
                    layout.write(str(self.n).encode())
                    layout.flush()
                    os.fsync(layout.fileno())
                    fcntl.flock(layout, fcntl.LOCK_SH)
                slot = open(self.out / f".densi_parte_{self.i}.lock", "a+b")
                self.files.append(slot)
                try:
                    fcntl.flock(slot, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise RuntimeError("partizione densi gia' attiva sullo stesso --out") from exc
                sincronizza_dir(self.out)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *args):
        for f in reversed(self.files):
            f.close()
        self.files.clear()


class Registro:
    def __init__(self, path):
        self.path = Path(path)
        self.file = open(path, "a+b")
        sincronizza_dir(self.path.parent)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.file.close()

    def write(self, testo):
        json.loads(testo)
        f = self.file
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            if f.seek(0, os.SEEK_END):
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    f.write(b"\n")
            f.write(testo.encode())
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
        return len(testo)

    def flush(self):
        self.file.flush()
        os.fsync(self.file.fileno())


def leggi_registri(out, log):
    ultime = {}
    for path in sorted(Path(out).glob("registro*.jsonl")):
        locali = {}
        with open(path, "rb") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            for numero, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    log(f"AVVISO {path.name}:{numero}: riga danneggiata conservata, non adottata")
                    continue
                if not isinstance(r, dict) or not isinstance(r.get("nome"), str):
                    raise ValueError(f"{path}:{numero}: record densi senza nome")
                locali[r["nome"]] = r
        for nome, r in locali.items():
            # I secondi non sono un timestamp: registri di layout precedenti
            # possono riferirsi allo stesso tensore. Il disaccordo non e' un commit.
            if nome in ultime and ultime[nome] != r:
                prima = ultime[nome]
                a = prima.get("commit_ns", 0) if prima else 0
                b = r.get("commit_ns", 0)
                if type(a) is not int or type(b) is not int:
                    raise ValueError("commit_ns del registro densi non intero")
                ultime[nome] = (r if b > a else prima) if a != b else None
            else:
                ultime[nome] = r
    return ultime


def firma(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def conserva(path, log):
    if path.exists():
        dest = path.with_name(path.name + ".incompleto-" + uuid.uuid4().hex)
        os.rename(path, dest)
        sincronizza_dir(path.parent)
        log(f"AVVISO artefatto da rifare, conservato: {dest}")


# --- D36 (22/10): provenance binding of a dense commit to its inputs --------------------------------------
# A committed unit is adopted on resume only if it was produced from the SAME inputs: delta bytes, host plane
# bytes (content, not path: v10 folds the dense tensors from BASE and later from GE, whose dense bytes are equal),
# Hessian file identity and numeric recipe (device type, dtype, source of the numeric functions).
# Switch FORGIA_DENSI_PROVENIENZA: "1" (default) write + check; "0" = previous behaviour (no write, no check);
# "rigorosa" = also refuse legacy records that carry no provenance.
VERSIONE_PROVENIENZA = 1


def modo_provenienza():
    modo = os.environ.get("FORGIA_DENSI_PROVENIENZA", "1")
    if modo not in ("0", "1", "rigorosa"):
        raise ValueError("FORGIA_DENSI_PROVENIENZA must be 0, 1 or rigorosa")
    return modo


def _sha_bytes(arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(repr((str(a.dtype), tuple(a.shape))).encode())
        h.update(memoryview(a.reshape(-1)).cast("B"))
    return h.hexdigest()


def _sha_testo(parti):
    h = hashlib.sha256()
    for t in parti:
        h.update(t.encode())
        h.update(b"\0")
    return h.hexdigest()


def ricetta(dev, dt, funzioni, file_moduli):
    """Numeric recipe identity: device type, dtype, source text of the numeric functions, bytes of the modules
    that implement the algebra, and the env switches that select alternative kernels."""
    return {"device": dev.type, "dtype": str(dt),
            "funzioni_sha256": _sha_testo(inspect.getsource(f) for f in funzioni),
            "moduli_sha256": {Path(m).name: firma(m) for m in file_moduli},
            "env": {k: os.environ.get(k, "") for k in ("PIEGA_ALGEBRA", "FORGIA_BABAI_TRITON")}}


class Provenienza:
    """Identity of the inputs of one dense unit; thread-safe cache (the fast CLI commits from a writer thread)."""

    def __init__(self, src, delta, manif, ricetta_):
        self.src, self.delta, self.manif, self.ricetta = src, delta, manif, ricetta_
        self._cache, self._lock = {}, threading.Lock()

    def di(self, nome):
        with self._lock:
            if nome in self._cache:
                return self._cache[nome]
        from rd_ternary.formato import TensorKey
        L = int(nome.split(".")[1])
        e = self.src.entry(TensorKey(L, -1, nome.split(".")[-2]))
        ospite = _sha_bytes([np.asarray(t.data).view(np.uint8) for t in e.tensors])
        dW = self.delta[nome]
        dW = dW.detach().contiguous().cpu().numpy() if hasattr(dW, "detach") else np.asarray(dW)
        info = (self.manif.get(nome.replace(".weight", "3.weight"), {}) or {}).get("hessiana") or {}
        hess = {"assente": True}
        if info.get("file"):
            st = os.stat(info["file"])
            hess = {"file": str(info["file"]), "byte": st.st_size, "mtime_ns": st.st_mtime_ns}
        p = {"versione": VERSIONE_PROVENIENZA, "ospite_sha256": ospite, "delta_sha256": _sha_bytes([dW]),
             "hessiana": hess, "ricetta": self.ricetta}
        with self._lock:
            self._cache[nome] = p
        return p


def _controlla_provenienze(records, mio, provenienza, log):
    """Refuse BEFORE touching any artefact when a committed unit comes from other inputs (D36)."""
    modo = modo_provenienza()
    if provenienza is None or modo == "0":
        return
    diverse = []
    for nome in sorted(mio):
        rec = records.get(nome)
        if not isinstance(rec, dict):
            continue
        p = rec.get("provenienza")
        if p is None:
            if modo == "rigorosa":
                diverse.append(f"{nome}: legacy record without provenance")
            else:
                log(f"AVVISO {nome}: record without provenance (legacy): inputs not bound (D36)")
            continue
        attesa = provenienza.di(nome)
        if p != attesa:
            campi = sorted(k for k in set(p) | set(attesa) if p.get(k) != attesa.get(k))
            diverse.append(f"{nome}: differs in {campi}")
    if diverse:
        raise RuntimeError("D36: committed dense units come from different inputs; nothing adopted or moved; "
                           "use a NEW --out (or FORGIA_DENSI_PROVENIENZA=0 for the old behaviour): "
                           + "; ".join(diverse[:8]) + (f" (+{len(diverse) - 8})" if len(diverse) > 8 else ""))


def riprendi(out, mio, src, CM, log, provenienza=None):
    from rd_ternary.formato import TensorKey

    td = Path(out) / "tensori"
    records = leggi_registri(out, log)
    _controlla_provenienze(records, mio, provenienza, log)
    fatti = set()
    for nome in sorted(mio):
        L = int(nome.split(".")[1])
        e = src.entry(TensorKey(L, -1, nome.split(".")[-2]))
        paths = [td / f"{t.name}.npy" for t in e.tensors]
        rec = records.get(nome)
        valido = (isinstance(rec, dict) and type(rec.get("piani")) is int
                  and rec["piani"] == len(paths))
        if valido:
            for key in ("energia_correzione_H", "residuo_dopo_piega_H", "catturata", "norma_rel_dW"):
                v = rec.get(key)
                if not isinstance(v, (int, float)) or not math.isfinite(v):
                    valido = False
                    break
        if valido:
            try:
                schede = rec.get("artefatti")
                if schede is not None and (not isinstance(schede, dict)
                                           or set(schede) != {p.name for p in paths}):
                    raise ValueError("insieme piani diverso dal commit")
                for t, p in zip(e.tensors, paths):
                    a = np.load(p, mmap_mode="r", allow_pickle=False)
                    attesa = (e.out, t.data.nbytes // e.out)
                    if a.dtype != np.uint8 or a.shape != attesa or p.stat().st_size != a.offset + a.nbytes:
                        raise ValueError(f"forma/dtype/taglia errata: {p.name}")
                    if schede is not None:
                        if schede[p.name] != {"sha256": firma(p), "shape": list(a.shape), "dtype": "uint8"}:
                            raise ValueError(f"SHA o metadati diversi: {p.name}")
                    else:
                        trit, scale = CM.gp.decode_rows(a, e.fmt.type_id, e.in_)
                        if not np.isfinite(scale).all() or not np.isin(trit, [-1, 0, 1]).all():
                            raise ValueError(f"piano legacy non valido: {p.name}")
                    del a
            except (OSError, ValueError, EOFError) as exc:
                log(f"AVVISO {nome}: commit non adottato: {exc}")
                valido = False
        if valido:
            fatti.add(nome)
            if rec.get("artefatti") is None:
                log(f"AVVISO {nome}: registro legacy, struttura/codec verificati; SHA storico non disponibile")
        else:
            for p in paths:
                conserva(p, log)
    return fatti


def salva_piano(td, nome, bb):
    path = Path(td) / f"{nome}.npy"
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.parziale")
    if path.exists():
        raise FileExistsError(f"rifiuto sovrascrittura piano non riconciliato: {path}")
    if bb.dtype != np.uint8:
        raise ValueError("piano denso non uint8")
    with open(tmp, "xb") as f:
        np.save(f, bb, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    sincronizza_dir(path.parent)
    return path.name, {"sha256": firma(path), "shape": list(bb.shape), "dtype": "uint8"}


def commit(reg, rec, artefatti, provenienza=None):
    if len(artefatti) != rec["piani"]:
        raise ValueError("commit senza tutti i piani")
    extra = {"provenienza": provenienza} if provenienza is not None and modo_provenienza() != "0" else {}
    reg.write(json.dumps({**rec, **extra, "artefatti": artefatti, "commit_ns": time.time_ns()},
                         allow_nan=False) + "\n")

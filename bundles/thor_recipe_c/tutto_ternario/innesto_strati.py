"""Innesto diagnostico (28/9 sera): EB3 p20 con gli strati indicati presi INTERI dal maestro Q8_0 (tutti i tensori blk.L.*).
Domanda: quanti punti MMLU vale riparare solo gli strati 37-38, dove la sonda lineare vede nascere il divario?
Non e' un candidato (non ternario negli strati innestati): e' il TETTO di un recupero mirato su quegli strati.
Uso: innesto_strati.py --strati 37,38 --uscita FILE.gguf"""
import argparse, hashlib, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "@@ROOT:models@@/build/llamacpp-tq1-k1024/gguf-py")
from gguf import GGUFReader, GGUFWriter  # noqa: E402
EB3 = "@@ROOT:models@@/gguf/tony-tern/Thor-T-EB3-passo_000020.gguf"
MAE = "@@ROOT:models@@/gguf/tony-tern/teacher-q8/Ornith-1.5-35B-A3B-Q8_0.gguf"
ap = argparse.ArgumentParser(); ap.add_argument("--strati", required=True); ap.add_argument("--uscita", required=True)
a = ap.parse_args(); S = {int(x) for x in a.strati.split(",")}
da_maestro = lambda n: n.startswith("blk.") and int(n.split(".")[1]) in S
r, m = GGUFReader(EB3), GGUFReader(MAE)
tm = {t.name: t for t in m.tensors}
w = GGUFWriter(path=None, arch="qwen35moe")
for f in r.fields.values():
    if not f.name.startswith(("GGUF.", "split.")):
        w.add_key_value(f.name, f.contents(), f.types[0], sub_type=f.types[-1] if len(f.types) > 1 else None)
w.add_string("thor.innesto", f"DIAGNOSTICO: EB3 p20 con gli strati {sorted(S)} interi dal maestro Q8_0 {MAE}")
piano, nomi_eb3 = [], set()
for t in r.tensors:
    nomi_eb3.add(t.name)
    src = tm[t.name] if da_maestro(t.name) else t
    if da_maestro(t.name):
        assert [int(x) for x in src.shape] == [int(x) for x in t.shape], (t.name, src.shape, t.shape)
    raw = np.asarray(src.data)
    w.add_tensor_info(t.name, list(raw.shape), raw.dtype, int(raw.nbytes), raw_dtype=src.tensor_type)
    piano.append((t.name, raw, hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest(), src.tensor_type.name))
mancanti = [n for n in tm if da_maestro(n) and n not in nomi_eb3]
assert not mancanti, f"tensori del maestro negli strati innestati assenti in EB3: {mancanti}"
tmp = a.uscita + ".parziale"
w.open_output_file(Path(tmp)); w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for _, raw, _, _ in piano:
    w.write_tensor_data(np.ascontiguousarray(raw))
w.close(); os.replace(tmp, a.uscita)
r2 = GGUFReader(a.uscita); err = []
for t2, (n, _, s, ty) in zip(r2.tensors, piano):
    if t2.name != n or t2.tensor_type.name != ty or hashlib.sha256(np.ascontiguousarray(np.asarray(t2.data)).tobytes()).hexdigest() != s:
        err.append(n)
n_inn = sum(1 for n, *_ in piano if da_maestro(n))
print(f"[{time.strftime('%H:%M:%S')}] {a.uscita}: {os.path.getsize(a.uscita)/2**30:.3f} GiB, {n_inn} tensori dal maestro, verifica",
      "OK" if not err and len(r2.tensors) == len(piano) else f"ERRORI {err[:5]}")
sys.exit(1 if err else 0)

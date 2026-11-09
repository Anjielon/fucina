#!/usr/bin/env python3
"""prova_reale_f32.py — F32-E2E on the REAL host GGUF (T400, sha f118d59a…), CPU only, no GPU, RSS < 2 GB (26/10/2026).
Complements test_f32.py (synthetic layers) with the real tensors, the real e2e_studente layers and the real exporter:

  R1  census of the F32 tensors of the GGUF: 301 in layers 0-39 + output_norm (22 213 248 values) + 9 of the MTP layer 40
      (never loaded by the student: untouched); per family count, sign, zeros (a zero would forbid the log space)
  R2  the set the Studente trains with --allena-f32 tutte, by ITS OWN rule (carica_strato_gguf.MAPPA of the layer type,
      e2e_studente.FAMIGLIE_F32) == the 301 F32 tensors of the GGUF; no ternary tensor in it
  R3  identity export: f32_esporta.py --da-gguf --tutti writes the 301 buffers and every one is the GGUF's own bytes
      (sha256): patch_gguf.py with an untrained pack gives back T400 at the byte (it copies the base and rewrites only
      those byte ranges), so every tappa GGUF differs from T400 ONLY in the F32 values that moved
  R4  real Studente, layers 3 (full attention) and 4 (Gated DeltaNet), recipe B32, torch engine, GDN veloce v3 + direct conv, CPU, allena_f32 = the 11
      families: construction re-checks every master against the loader AT THE BIT; presence gate of f32_blocco; one
      forward/backward of real tokens (checkpointed avanti/indietro of the trainer): EVERY F32 tensor of the two layers +
      output_norm has a nonzero gradient, NO theta has a gradient; one Adam step of the F32 blocks: every theta still 0,
      expert/dense bytes and d0 bitwise the start, the F32 moved; the pack -> f32_esporta writes ONLY the moved tensors
      (the other F32 are equal to the base) and their bytes are the trained masters
Run: HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 19 @@ROOT:venv@@/bin/python3 \\
       test_f32/prova_reale_f32.py [GGUF]
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)
QUI = Path(__file__).resolve().parent
COD = QUI.parent
sys.path.insert(0, str(COD))
# CPU: GDN veloce v3 (torch, no compile workers) and the direct GDN conv, as in the run (v5 = v3 + torch.compile on GPU)
os.environ.setdefault("E2E_GDN", "veloce"); os.environ.setdefault("E2E_GDN_MODULO", "gdn_veloce_corsa")
os.environ.setdefault("E2E_CONV_DIRETTA", "1"); os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
import cardan_kd_paper as KDP                       # noqa: E402,F401  (path setup of the thor_v2 tree)
import e2e_studente as S                            # noqa: E402
import e2e_allena as A                              # noqa: E402
import f32_blocco as FB                             # noqa: E402
import f32_esporta as FE                            # noqa: E402

T400 = "@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/CARDAN_CONGIUNTO_S7T550_1020/tappa_400/piega_d/Thor-T-C-S4b-CONGIUNTO.gguf"
GGUF = sys.argv[1] if len(sys.argv) > 1 else T400
TMP = Path("@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/SINTESI_1025/F32/collaudo") / time.strftime("reale_%Y%m%d_%H%M%S")
TMP.mkdir(parents=True, exist_ok=True)
ESITI = []


def rss() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


def prova(nome):
    def dec(f):
        def run():
            t0 = time.time()
            try:
                m = f()
                ESITI.append((nome, True, m or ""))
                print(f"PASS {nome} {m or ''} [{time.time() - t0:.1f} s, RSS max {rss():.2f} GiB]", flush=True)
            except Exception as e:
                import traceback
                ESITI.append((nome, False, f"{type(e).__name__}: {e}"))
                print(f"FAIL {nome}: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
        return run
    return dec


A.scegli_gdn(lambda m: print(m, flush=True))
# ONE GGUF reader for the whole script (~470 MB each in this gguf-py fork): the real Studente's own (layers 3 and 4)
ST = S.Studente(gguf=Path(GGUF), strati=[3, 4], ricetta="B32", motore_nome="torch", device="cpu", log=lambda *m: None,
                densi_ternari="tutte", allena_f32=FB.famiglie("tutte"))
TENS = dict(ST._tens)
print(f"Studente vero: strati {ST.strati}, F32 allenati {ST.f32_conteggi} (master = caricatore al bit, controllato "
      f"alla costruzione); RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20:.2f} GiB", flush=True)


@prova("R1 censimento F32 del GGUF: 301 tensori negli strati 0-39 + output_norm, 22 213 248 valori; 9 MTP intoccati")
def r1():
    fam, val, neg, zeri, mtp = collections.Counter(), collections.Counter(), collections.Counter(), 0, []
    for t in TENS.values():
        if t.tensor_type.name != "F32":
            continue
        if t.name.startswith("blk.40."):
            mtp.append(t.name); continue
        f = FE.famiglia(t.name)
        a = np.asarray(t.data, dtype=np.float32)
        fam[f] += 1; val[f] += a.size; neg[f] += int((a < 0).sum()); zeri += int((a == 0).sum())
    n, nv = sum(fam.values()), sum(val.values())
    assert n == 301 and nv == 22213248, (n, nv)
    assert len(mtp) == 9, mtp
    assert zeri == 0, zeri
    (TMP / "censimento.json").write_text(json.dumps({"famiglie": fam, "valori": val, "negativi": neg, "mtp": mtp}, indent=1))
    return f"({dict(fam)}; negativi nelle norme: attn_k_norm {neg['attn_k_norm']}; zeri 0)"


def attesi() -> set:
    """The F32 names Studente(allena_f32=tutte) trains over layers 0-39, by its own rule (_prepara_f32)."""
    import carica_strato_gguf as CSG
    cfg = S.config_testo()
    fams = FB.famiglie("tutte")
    out = set()
    for L in range(40):
        mappa = CSG.MAPPA["full" if cfg.layer_types[L] == "full_attention" else "linear"]
        for f in fams:
            if f in S.F32_GLOBALI or S.FAMIGLIE_F32[f] not in mappa:
                continue
            n = f"blk.{L}.{S.FAMIGLIE_F32[f]}"
            if n in TENS:
                out.add(n)
    out.add(S.F32_GLOBALI["output_norm"])
    return out


@prova("R2 insieme allenato dalla regola dello Studente == i 301 F32 del GGUF; nessun tensore ternario")
def r2():
    a = attesi()
    g = {n for n, t in TENS.items() if t.tensor_type.name == "F32" and not n.startswith("blk.40.")}
    assert a == g, (sorted(a - g)[:5], sorted(g - a)[:5])
    tipi = collections.Counter(TENS[n].tensor_type.name for n in a)
    assert set(tipi) == {"F32"}, tipi
    return f"({len(a)} tensori, tipi {dict(tipi)})"


@prova("R3 esportazione identita': i 301 buffer scritti sono i byte del GGUF (patch_gguf ridarebbe T400 al byte)")
def r3():
    rc = FE.main(["--base", GGUF, "--da-gguf", "--uscita", str(TMP / "identita"), "--tutti"], tens=TENS)
    rep = json.load(open(TMP / "identita" / "esporta.json"))
    assert rc == 0 and rep["scritti"] == 301 and rep["uguali_alla_base"] == 301, (rc, rep["scritti"], rep["uguali_alla_base"])
    assert "identita" in rep
    nb = sum(int(TENS[n].n_bytes) for n in rep["sha256"])
    return f"({rep['identita']}; {nb / 2 ** 20:.1f} MiB)"


def impronta_reale(st) -> str:
    """sha256 of every non-F32 tensor the student holds for its layers: expert/dense ternary bytes and d0, theta,
    every module parameter and buffer (the F32 installed values are plain attributes, not parameters)."""
    h = hashlib.sha256()
    for L in st.strati:
        es = st.esperti(L)
        for n, f in es.fam.items():
            h.update(f"{L}.{n}".encode())
            h.update(f.blocchi.detach().cpu().contiguous().numpy().tobytes())
            h.update(f.d0.detach().cpu().contiguous().numpy().tobytes())
        for n, t in es.theta.items():
            h.update(t.detach().cpu().contiguous().numpy().tobytes())
        for n, p in list(st.moduli[L].named_parameters()) + list(st.moduli[L].named_buffers()):
            t = p.detach().cpu().contiguous()
            h.update(n.encode()); h.update((t.view(torch.int16) if t.dtype == torch.bfloat16 else t).numpy().tobytes())
    return h.hexdigest()


@prova("R4 Studente vero (strati 3 e 4), CPU: gradiente su OGNI F32 e su nessuna theta; dopo un passo Adam il resto e' al bit")
def r4():
    st = ST
    for p in st.parametri():
        p.requires_grad_(False)                         # as the trainer
    imp0 = impronta_reale(st)
    F = FB.BloccoF32(st, {"norme": 1e-4, "router": 3e-5, "ssm_a": 1e-4, "ssm": 3e-5}, FB.log_classi("norme,ssm_a"))
    cont = F.conteggi()
    assert len(F.voci) == 6 + 8 + 1, (len(F.voci), cont)  # layer 3: 6 F32, layer 4 (GDN): 8, + output_norm
    ids = torch.tensor([[760, 6511, 314, 9338, 369, 11751, 13, 576, 3974, 7111, 11, 323]], dtype=torch.long)
    with torch.no_grad():
        h1 = st.nascosto(ids)
    err = FB.cancello_presenza(st, F, lambda: st.nascosto(ids), h1)
    assert not err, err
    g = torch.Generator().manual_seed(7)
    r = torch.randn(h1.shape, generator=g, dtype=h1.dtype)
    F.azzera_grad_master()
    stato = st.avanti(ids)
    xL = stato["xL"].detach().requires_grad_(True)
    with torch.enable_grad():
        ((st.esce(xL) * r).sum() / h1.shape[0]).backward()
    st.indietro(stato, xL.grad)
    cop = F.gradienti_log()
    vuoti = {f: c for f, c in cop.items() if c[0] != c[1]}
    assert not vuoti, f"tensori F32 senza gradiente: {vuoti}"
    assert all(p.grad is None for p in st.parametri()), "theta con gradiente"
    for nb, (opt, ps, _) in F.blocchi().items():
        torch.nn.utils.clip_grad_norm_(ps, 1.0); opt.step()
    F.dopo_passo()
    assert max(float(p.abs().max()) for p in st.parametri()) == 0.0, "theta mosse"
    assert impronta_reale(st) == imp0, "byte ternari, d0 o pesi densi cambiati"
    stt = F.statistiche()
    assert stt["f32_cambiati"] > 0.5, stt
    F.salva(TMP / "f32_passo1.pt", {"passo": 1})
    rc = FE.main(["--base", GGUF, "--pacco", str(TMP / "f32_passo1.pt"), "--uscita", str(TMP / "passo1")], tens=TENS)
    rep = json.load(open(TMP / "passo1" / "esporta.json"))
    assert rc == 0 and rep["scritti"] == len(F.voci) and rep["uguali_alla_base"] == 0, (rc, rep["scritti"])
    for v in F.voci:                                    # written bytes = the trained master, file order
        b = np.load(TMP / "passo1" / "tensori" / f"{v.tf.nome}.npy")
        assert np.array_equal(b, np.ascontiguousarray(v.tf.valore_file(), dtype="<f4").reshape(-1).view(np.uint8)), v.tf.nome
    return (f"({len(F.voci)} F32: {json.dumps({f: f'{c[0]}/{c[1]}' for f, c in cop.items()})}; theta "
            f"{len(st.parametri())} senza gradiente e a 0; impronta del resto uguale; rel norme {stt['f32_rel_norme']:.1e} "
            f"router {stt['f32_rel_router']:.1e}; esportati {rep['scritti']} tensori = master)")


if __name__ == "__main__":
    print(f"GGUF {GGUF}\ncartella {TMP}", flush=True)
    for f in (r1, r2, r3, r4):
        f()
    ok = sum(1 for _, b, _ in ESITI if b)
    print(f"\n{ok}/{len(ESITI)} PASS (cartella {TMP}, RSS max {rss():.2f} GiB)")
    sys.exit(0 if ok == len(ESITI) else 1)

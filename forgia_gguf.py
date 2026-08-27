#!/usr/bin/env python3
"""FUCINA — forgia ternaria GENERALE: da un GGUF qualsiasi (MoE) al 2-piani TQ1_0.

Generalizzazione della forgia di ODINO (26/8). Differenze dal forgia_odino:
  - sorgente = un GGUF già quantizzato (dequant per-esperto), non i bf16 del NAS
  - architettura PARAMETRICA: n_strati, n_esperti, hidden letti dai metadati
  - niente lista "pescati": i tensori non-esperto si copiano TALI E QUALI
    (la sorgente è già di qualità: Q6_K/Q8 — la leva DS4 è gratis)
Le CORREZIONI della v3.1 sono cablate dalla nascita:
  1. esperti riordinati CALDI-PRIMI + righe del router permutate (il motore
     assume il prefisso 0..K-1)
  2. i caldi tengono la COPPIA CONGIUNTA (piano-1+2 ottimizzati insieme)
  3. verifica al confine per-strato (assert, non speranza)

Uso:
  forgia_gguf.py --sorgente X.gguf --uscita Y.gguf --caldi 28 \
                 --imatrix im.gguf [--hessiane DIR]
Il giornale di ripresa è <uscita>.giornale (indice+byte, come ODINO).
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, "/home/angelo/build-llamacpp/gguf-py")
sys.path.insert(0, str(Path(__file__).parent))
import gguf
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as T
from gguf import quants as GQ
import ternary_gpu as G
import due_piani as D
from tq1_pack import pack, reorder_experts, hot_first_order

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def leggi_counts(imatrix: Path) -> dict[int, np.ndarray]:
    """strato → esperti ordinati dal più caldo (dai .counts dell'imatrix)."""
    out = {}
    r = GGUFReader(str(imatrix))
    for t in r.tensors:
        if t.name.endswith(".ffn_gate_exps.weight.counts"):
            L = int(t.name.split(".")[1])
            out[L] = np.argsort(np.array(t.data).astype(np.float64).ravel())[::-1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sorgente", required=True)
    ap.add_argument("--uscita", required=True)
    ap.add_argument("--caldi", type=int, default=28)
    ap.add_argument("--imatrix", required=True)
    ap.add_argument("--hessiane", default=None, help="DIR con H_XX.npy (GPTQ per gate/up)")
    ap.add_argument("--pezzo", type=int, default=3_000_000)
    A = ap.parse_args()
    K = A.caldi

    src = GGUFReader(A.sorgente)
    tensori = {t.name: t for t in src.tensors}
    arch = None
    for f in src.fields.values():
        if f.name.endswith(".architecture"):
            arch = str(bytes(f.parts[-1]), "utf8"); break
    caldi_ord = leggi_counts(Path(A.imatrix))
    strati = sorted({int(n.split(".")[1]) for n in tensori if ".ffn_gate_exps.weight" in n})
    t0e = tensori[f"blk.{strati[0]}.ffn_gate_exps.weight"]
    E = int(t0e.shape[2])
    senza_counts = sorted(set(strati) - set(caldi_ord))
    log(f"arch={arch} · strati MoE={len(strati)} · esperti={E} · caldi={K}")
    if senza_counts:
        # es. lo strato MTP/nextn: mai eseguito nel forward normale → zero
        # counts nell'imatrix. Niente caldi, niente 2° piano, solo piano-1.
        log(f"⚠️ strati SENZA counts (probabile MTP/nextn): {senza_counts} → solo piano-1")

    def hchol(L):
        if not A.hessiane: return None
        f = Path(A.hessiane) / f"H_{L:02d}.npy"
        if not f.exists(): return None
        return torch.from_numpy(D.prepara_hessiana(np.load(f).astype(np.float64)).astype(np.float32))

    def perm_di(L):
        caldi = [int(x) for x in caldi_ord[L][:K]]
        return hot_first_order(caldi, E), caldi

    # ── piano di scrittura ───────────────────────────────────────────────
    w = GGUFWriter(path=None, arch=arch)
    for f in src.fields.values():
        if f.name.startswith(("GGUF.", "split.")): continue
        try: w.add_key_value(f.name, f.contents(), f.types[0], sub_type=f.types[-1] if len(f.types) > 1 else None)
        except Exception: pass
    if K: w.add_uint32(f"{arch}.expert_count2", K)

    piano = []
    for nome, t in sorted(tensori.items()):
        exp = nome.endswith(("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"))
        if exp:
            forma = [int(x) for x in t.shape]
            piano.append((nome, forma, "forgia"))
            if K and int(nome.split(".")[1]) in caldi_ord:
                f2 = list(forma); f2[2] = K
                piano.append((nome.replace(".weight", "2.weight"), f2, "forgia2"))
        else:
            piano.append((nome, [int(x) for x in t.shape], "copia"))

    for nome, forma, come in piano:
        if come in ("forgia", "forgia2"):
            fb = list(forma)[::-1]; fb[-1] = fb[-1] // 256 * 54
            w.add_tensor_info(nome, fb, np.dtype(np.uint8), int(np.prod(fb)), raw_dtype=T.TQ1_0)
        else:
            t = tensori[nome]
            d = np.asarray(t.data)
            w.add_tensor_info(nome, list(d.shape), d.dtype, int(d.nbytes), raw_dtype=t.tensor_type)

    giornale = Path(A.uscita + ".giornale")
    riprendi = 0
    if giornale.exists() and Path(A.uscita).exists():
        try:
            idx, pos = (int(x) for x in giornale.read_text().split())
            if Path(A.uscita).stat().st_size >= pos:
                riprendi = idx + 1
                log(f"⭐ RIPRESA dal tensore {riprendi} (byte {pos})")
        except Exception:
            pass
    Path(A.uscita).parent.mkdir(parents=True, exist_ok=True)
    w.open_output_file(Path(A.uscita))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    if riprendi:
        f = w.fout[0]; f.flush(); f.seek(pos); f.truncate(pos)

    fatti: dict[str, tuple] = {}
    t0 = time.time(); pesi = 0
    for i, (nome, forma, come) in enumerate(piano):
        if i < riprendi: continue
        if come == "copia":
            t = tensori[nome]
            if K and nome.endswith(".ffn_gate_inp.weight") and int(nome.split(".")[1]) in caldi_ord:
                L = int(nome.split(".")[1])
                rt = np.asarray(t.data)
                assert rt.ndim == 2 and rt.shape[0] == E
                w.write_tensor_data(np.ascontiguousarray(rt[perm_di(L)[0]]))
            else:
                w.write_tensor_data(np.asarray(t.data))
        else:
            orig = nome.replace("2.weight", ".weight") if come == "forgia2" else nome
            if orig not in fatti:
                L = int(orig.split(".")[1])
                t = tensori[orig]
                W = GQ.dequantize(np.asarray(t.data), t.tensor_type)   # (E, out, in) f32
                E_, out_, ind = W.shape
                assert E_ == E
                nb_e = out_ * (ind // 256)
                # l'Hessiana e' misurata sul flusso residuo (hidden): vale per
                # gate/up (ingresso=hidden), NON per down (ingresso=intermedio)
                hidden = int(tensori[f"blk.{L}.ffn_gate_exps.weight"].shape[0])
                H = hchol(L) if ind == hidden else None
                Wt = torch.from_numpy(W.reshape(-1, ind)).float()
                if H is not None:
                    d1, q1 = G.quantizza_un_piano(Wt, H, pezzo=A.pezzo)
                else:
                    r = G.quantizza(Wt, None, giri=2, pezzo=A.pezzo); d1, q1 = r[0], r[1]
                p1 = pack(d1.reshape(-1, 1), q1.reshape(-1, 256))
                if L in caldi_ord:
                    perm, caldi = perm_di(L)
                    sel = np.concatenate([np.arange(e*out_, (e+1)*out_) for e in caldi])
                    Wc = torch.from_numpy(np.ascontiguousarray(W.reshape(-1, ind)[sel])).float()
                    j1d, j1q, d2, q2 = G.quantizza(Wc, H, giri=2, pezzo=A.pezzo)
                    del Wc
                    pj1 = pack(j1d.reshape(-1, 1), j1q.reshape(-1, 256))
                    v1 = p1.reshape(E, nb_e, 54)
                    for s, e in enumerate(caldi):
                        v1[int(e)] = pj1[s*nb_e:(s+1)*nb_e]
                    p1 = reorder_experts(v1.reshape(-1, 54), perm)
                    p2 = pack(d2.reshape(-1, 1), q2.reshape(-1, 256))
                    # verifica al confine: caldo n.1 (ora pos.0) vs sorgente dequant
                    ric = GQ.dequantize(p1[:nb_e], T.TQ1_0).reshape(out_, ind) \
                        + GQ.dequantize(p2[:nb_e], T.TQ1_0).reshape(out_, ind)
                    Wv = W[caldi[0]].astype(np.float32)
                    err = float(np.linalg.norm(ric - Wv) / np.linalg.norm(Wv))
                    del j1d, j1q, d2, q2
                else:
                    # strato senza counts (MTP): solo piano-1, nessun riordino
                    p2 = None
                    ric = GQ.dequantize(p1[:nb_e], T.TQ1_0).reshape(out_, ind)
                    Wv = W[0].astype(np.float32)
                    err = float(np.linalg.norm(ric - Wv) / np.linalg.norm(Wv))
                del Wt
                assert err < 0.6, f"{orig}: confine rotto ({err:.2f})"
                fatti[orig] = (p1, p2)
                pesi += W.size
                v = pesi / (time.time() - t0) / 1e6
                log(f"[{i+1}/{len(piano)}] {orig} · confine {err*100:.1f}% · {v:.1f} M/s")
                del W, d1, q1
            dati = fatti[orig][1 if come == "forgia2" else 0]
            attesi = int(np.prod([x for x in ( [int(y) for y in forma][::-1][:-1] + [[int(y) for y in forma][::-1][-1]//256*54] ,)][0])) if False else None
            try:
                w.write_tensor_data(dati)
            except AssertionError:
                log(f"⛔ MISMATCH {nome}: forma_gguf={forma} · dati.nbytes={dati.nbytes} · dati.shape={dati.shape}")
                raise
            if come == "forgia2": fatti.pop(orig, None)
        w.fout[0].flush(); giornale.write_text(f"{i}\n{w.fout[0].tell()}\n")
    w.close()
    log(f"🏁 FORGIA FINITA in {(time.time()-t0)/60:.1f} min → {A.uscita}")


main()

#!/usr/bin/env python3
"""e2_scrivi_gguf.py — porta la modifica E2 (rank1 / BNP sulle 4 righe lettera di lm_head) dentro un GGUF NUOVO,
identico byte per byte all'originale salvo le righe lettera della testa, che restano TERNARIE nel formato del file
(TQ1_0R_1024, tipo ggml 45: segno S1024, FWHT-1024, 8 scale fp16 per blocco da 1024, trit in base 3).

La testa Thor e' a PIANI: logit = output(h) + output2(h) + output3(h) [+ output4] (qwen35moe.cpp di thor-velocita).
La modifica si calcola sulla SOMMA (lm_head vera) e si riporta nei piani con e2_riternarizza.riternarizza_piani:
discesa a coordinate sui trit + rifit LS delle scale fp16, piano per piano, nella metrica dei logit lettera
E_h[((W_q - W_edit) h)^2] (h di train). Misurato su R2B3m: la sola ri-ternarizzazione RTN (quantize_rows) del
piano 1 cancellava il 99,7 % di una modifica rank1. Con --hidden il risultato si sceglie in VALIDAZIONE (20 % dei
gruppi di train) fra "nessuna modifica" e la discesa a vari ridge: il file non e' mai peggio di non modificare
sugli h di validazione. Dequantizza -> quantizza restituisce GLI STESSI byte (204/204 righe verificate): una
modifica nulla da' un file con la stessa sha256 dell'originale.

L'originale non viene MAI aperto in scrittura: copy_file_range in <out>.parziale, patch delle sole righe,
rilettura e confronto, fsync, rename atomico. Testa MTP (blk.40.nextn.*), metadati e tutti gli altri tensori:
byte originali. Accanto: <out>.modifica.json (righe/byte cambiati, W lettere prima/dopo, eval heldout, sha).

Uso:
  e2_scrivi_gguf.py --src ST.gguf --out OUT.gguf --letter-ids 357,417,351,414 --method none          # collaudo
  e2_scrivi_gguf.py ... --method rank1 --bias out/e2_rank1/bias_direction.pt --hidden data/hid
  e2_scrivi_gguf.py ... --method bnp --k 32 --align abs --bias out/e2_bnp_abs/bias_direction.pt --hidden data/hid
  e2_scrivi_gguf.py ... --method delta --delta-npy d.npy        # delta [4, d] sulla somma dei piani (collaudo)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from e2_bnp_edit import (CODEC_DIR, TQ1_0R_1024, edit_bnp, edit_rank1, gguf_tensor_index, head_planes,
                         letter_logits, letter_metrics, load_hidden, split_hidden)

if CODEC_DIR not in sys.path:
    sys.path.insert(0, CODEC_DIR)
from tq1_0r_1024_codec import BLOCK_BYTES, QK, dequantize_blocks, quantize_rows  # noqa: E402
from e2_riternarizza import riternarizza_piani  # noqa: E402

logger = logging.getLogger("scrivi_gguf")
TENSOR = "output.weight"


def sha256_file(path: str, bs: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bs)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def row_layout(path: str, name: str = TENSOR) -> dict:
    idx, data0 = gguf_tensor_index(path)
    if name not in idx:
        raise SystemExit(f"{path}: manca {name}")
    ty, dims, off = idx[name]
    if ty != TQ1_0R_1024:
        raise SystemExit(f"{name}: tipo {ty}, atteso {TQ1_0R_1024} (TQ1_0R_1024)")
    ne0, nrows = int(dims[0]), int(np.prod(dims[1:]))
    if ne0 % QK:
        raise SystemExit(f"{name}: ne0 {ne0} non multiplo di {QK}")
    rb = ne0 // QK * BLOCK_BYTES
    base = data0 + off
    if base + nrows * rb > os.path.getsize(path):
        raise SystemExit(f"{name}: oltre la fine del file")
    return {"type": ty, "ne0": ne0, "rows": nrows, "row_bytes": rb, "base": base}


def read_rows(path: str, lay: dict, rows: list[int]) -> np.ndarray:
    out = np.empty((len(rows), lay["row_bytes"]), dtype=np.uint8)
    with open(path, "rb") as f:
        for i, r in enumerate(rows):
            f.seek(lay["base"] + r * lay["row_bytes"])
            out[i] = np.frombuffer(f.read(lay["row_bytes"]), dtype=np.uint8)
    return out


def deq(raw: np.ndarray, ne0: int) -> torch.Tensor:
    return torch.from_numpy(dequantize_blocks(raw.reshape(-1, BLOCK_BYTES)).reshape(len(raw), ne0))


def compute_edit(W4: torch.Tensor, a) -> tuple[torch.Tensor, dict]:
    info: dict = {"method": a.method}
    if a.method == "none":
        return W4.clone(), info
    if a.method == "delta":
        d = torch.from_numpy(np.load(a.delta_npy).astype(np.float32))
        assert d.shape == W4.shape, (d.shape, W4.shape)
        return W4 + d, info
    b = torch.load(a.bias, map_location="cpu").float()
    assert b.shape == (W4.shape[1],), b.shape
    info["b_norm"] = float(b.norm())
    if a.method == "rank1":
        return edit_rank1(W4, b), info
    W2, nodes = edit_bnp(W4, b, a.k, a.align)
    info.update(k=a.k, align=a.align, nodes=nodes)
    return W2, info



def quantize4(W: torch.Tensor) -> np.ndarray:
    return np.stack([quantize_rows(W[i:i + 1].numpy().astype(np.float32)).reshape(-1) for i in range(len(W))])


def choose_by_validation(raws0, lays, W4, W4e, Hall, mall, a):
    """Gli h di train si dividono per gruppo in fit (80%) e validazione (20%). Candidati: nessuna
    modifica + discesa a coordinate con ridge in --ridge-grid (fit sugli h di fit). Vince il minimo
    errore sui logit lettera in VALIDAZIONE: il file scritto non e' mai peggio di "non modificare"
    su h mai visti dal fit (con pochi h la metrica di train sovra-adatta: misurato sul mock,
    residuo heldout 8-26x con 120 h isotropi)."""
    from common import stable_int
    Htr, mtr = split_hidden(Hall, mall, "train")
    val = torch.tensor([stable_int("val/" + m["group_id"]) % 5 == 0 for m in mtr])
    Hf, Hv = Htr[~val].numpy(), Htr[val]
    lerr = lambda Wv: float((((Wv - W4e) @ Hv.T) ** 2).mean())
    cands = [("nessuna", [r.copy() for r in raws0], {}, lerr(W4))]
    for rg in [float(x) for x in a.ridge_grid.split(",")]:
        r1, inf = riternarizza_piani(raws0, W4e.numpy(), Hf, passes=a.passes, ridge=rg)
        Wq = sum(deq(r, lay["ne0"]) for r, lay in zip(r1, lays))
        cands.append((f"cd_ridge{rg:g}", r1, inf, lerr(Wq)))
    base = cands[0][3]
    table = [{"cand": n, "val_logit_residual": e / max(base, 1e-30),
              "fit_residual": inf.get("residual_frac")} for n, _, inf, e in cands]
    best = min(cands, key=lambda c: c[3])
    logger.info("validazione (%d h fit, %d h val): %s -> %s", len(Hf), len(Hv),
                [(t["cand"], round(t["val_logit_residual"], 4)) for t in table], best[0])
    return best[1], {"chosen": best[0], "validation": table, **best[2]}


def copy_file(src: str, dst: str) -> None:
    """Copia byte per byte nel kernel (copy_file_range): niente buffer grandi in RAM."""
    n = os.path.getsize(src)
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        done = 0
        while done < n:
            k = os.copy_file_range(fi.fileno(), fo.fileno(), min(1 << 30, n - done))
            if k <= 0:
                raise OSError("copy_file_range: 0 byte")
            done += k
        fo.flush()
        os.fsync(fo.fileno())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--letter-ids", required=True)
    ap.add_argument("--method", choices=["none", "rank1", "bnp", "delta"], required=True)
    ap.add_argument("--bias", help="bias_direction.pt di e2_bnp_edit.py (rank1/bnp)")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--align", choices=["signed", "abs"], default="signed")
    ap.add_argument("--delta-npy")
    ap.add_argument("--hidden", help="prefisso h (e2_collect_hidden.py): metrica dei logit (split train) "
                                     "per la ri-ternarizzazione + eval heldout")
    ap.add_argument("--requant", choices=["cd", "rtn"], default="cd",
                    help="cd = discesa a coordinate nella metrica dei logit (default); rtn = quantize_rows")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--ridge-grid", default="0.05,0.5,5,50", help="ridge candidati (con --hidden)")
    ap.add_argument("--ridge", type=float, default=0.05,
                    help="peso Frobenius relativo (tr/d) nella metrica: protegge gli h fuori dal train")
    ap.add_argument("--min-free-gb", type=float, default=30.0)
    ap.add_argument("--skip-sha", action="store_true", help="non calcolare le sha256 (2x9 GB, ~30 s)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    src, out = os.path.realpath(a.src), os.path.realpath(a.out)
    if src == out:
        raise SystemExit("--out coincide con --src: l'originale non si tocca")
    if os.path.exists(out):
        raise SystemExit(f"{out} esiste gia': non sovrascrivo")
    if a.method in ("rank1", "bnp") and not a.bias:
        raise SystemExit("--bias obbligatorio per rank1/bnp")
    st = os.statvfs(os.path.dirname(out))
    free, need = st.f_bavail * st.f_frsize / 2**30, os.path.getsize(src) / 2**30 + a.min_free_gb
    if free < need:
        raise SystemExit(f"spazio: {free:.0f} GiB liberi, servono {need:.0f}")
    lids = [int(x) for x in a.letter_ids.split(",")]
    idx, _ = gguf_tensor_index(src)
    planes = head_planes(idx, TENSOR)                         # output, output2, output3[, output4]
    lays = [row_layout(src, pn) for pn in planes]
    t0 = time.time()

    raws0 = [read_rows(src, lay, lids) for lay in lays]
    Wp = [deq(r, lay["ne0"]) for r, lay in zip(raws0, lays)]
    W4 = sum(Wp)                                              # lm_head vero = somma dei piani
    W4e, einfo = compute_edit(W4, a)
    einfo["planes"] = planes
    Hall = mall = None
    if a.hidden:
        Hall, mall = load_hidden(a.hidden)
    if torch.equal(W4e, W4):
        raws1, rinfo = [r.copy() for r in raws0], {"null_edit": True}
    elif a.requant == "rtn":                                   # tutta la modifica nel piano 1 (riferimento)
        others = sum(Wp[1:]) if len(Wp) > 1 else 0
        raws1 = [quantize4(W4e - others)] + [r.copy() for r in raws0[1:]]
        rinfo = {"requant": "rtn_plane0"}
    elif not a.hidden:
        raws1, rinfo = riternarizza_piani(raws0, W4e.numpy(), None, passes=a.passes, ridge=a.ridge)
    else:
        raws1, rinfo = choose_by_validation(raws0, lays, W4, W4e, Hall, mall, a)
    W4q = sum(deq(r, lay["ne0"]) for r, lay in zip(raws1, lays))   # cio' che il GGUF conterra' davvero
    patches = [(pn, lay, i, r) for pn, lay, r0, r1 in zip(planes, lays, raws0, raws1)
               for i, r in enumerate(lids) if not np.array_equal(r0[i], r1[i])]
    nbytes = int(sum((r0 != r1).sum() for r0, r1 in zip(raws0, raws1)))
    ed, rq = float((W4e - W4).norm()), float((W4q - W4e).norm())
    logger.info("piani %s; righe cambiate %d, byte cambiati %d; |W_edit-W| %.4g, |W_gguf-W| %.4g, "
                "|W_gguf-W_edit| %.4g; residuo metrica logit %s", planes, len(patches), nbytes, ed,
                float((W4q - W4).norm()), rq, rinfo.get("residual_frac"))

    tmp = out + ".parziale"
    copy_file(src, tmp)
    with open(tmp, "r+b") as f:
        for pn, lay, i, r in patches:
            f.seek(lay["base"] + r * lay["row_bytes"])
            f.write(raws1[planes.index(pn)][i].tobytes())
        f.flush()
        os.fsync(f.fileno())
    for pn, lay, r1 in zip(planes, lays, raws1):
        if not np.array_equal(read_rows(tmp, lay, lids), r1):
            raise SystemExit(f"rilettura {pn}: non combacia (.parziale lasciato per ispezione)")
    if [row_layout(tmp, pn) for pn in planes] != lays:
        raise SystemExit("indice tensori cambiato (impossibile): .parziale lasciato")
    os.replace(tmp, out)

    rep = {"src": src, "out": out, "letter_ids": lids, "edit": einfo, "requant_info": rinfo,
           "layouts": dict(zip(planes, lays)), "rows_changed": [[pn, r] for pn, _, _, r in patches],
           "bytes_changed": nbytes,
           "byte_ranges_changed": [[lay["base"] + r * lay["row_bytes"], lay["base"] + (r + 1) * lay["row_bytes"]]
                                   for _, lay, _, r in patches],
           "W_letters_original": W4.tolist(), "W_letters_gguf": W4q.tolist(),
           "rel_err_requant_vs_edit": rq / ed if ed > 0 else 0.0, "seconds": round(time.time() - t0, 1)}
    if not a.skip_sha:
        rep["sha256_src"], rep["sha256_out"] = sha256_file(src), sha256_file(out)
        logger.info("sha256 src %s / out %s", rep["sha256_src"], rep["sha256_out"])
    if a.hidden:
        Hho, mho = split_hidden(Hall, mall, "heldout")
        rep["eval_heldout"] = {n: letter_metrics(letter_logits(Hho, Wv), mho)
                               for n, Wv in (("original", W4), ("edited_float", W4e), ("edited_gguf", W4q))}
        lerr = lambda Wv: float((((Wv - W4e) @ Hho.T) ** 2).mean())
        rep["heldout_logit_residual"] = lerr(W4q) / max(lerr(W4), 1e-30)   # 0 = modifica intera, 1 = nessuna
        logger.info("residuo heldout sui logit lettera (W_gguf vs W_edit, 1 = come non modificare): %.4f",
                    rep["heldout_logit_residual"])
        for n, m in rep["eval_heldout"].items():
            logger.info("%-13s acc %.4f RStd %.2f fluct %.3f pred %s", n, m["acc"], m["RStd"],
                        m["fluctuation"], m["pred_hist_ABCD"])
    rep_path = out + ".modifica.json"
    Path(rep_path).write_text(json.dumps(rep, indent=1))
    logger.info("scritto %s (+ %s) in %.0f s", out, rep_path, time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())

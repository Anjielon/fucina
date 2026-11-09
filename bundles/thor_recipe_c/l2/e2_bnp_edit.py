#!/usr/bin/env python3
"""e2_bnp_edit.py — rimozione della direzione di bias (stile BNP, arXiv 2409.18857),
ri-ternarizzazione della matrice d'uscita e misura prima/dopo.

PASSI (sottocomandi):
  direction  b = media(h | risposta sbagliata) - media(h | risposta giusta), sugli stati
             nascosti h (ingresso di lm_head) degli item ammessi permutati (split "train").
             "Giusta/sbagliata" = argmax delle 4 lettere DEL MODELLO stesso vs posizione oro.
  edit       due modifiche di W_out = lm_head, forma PyTorch [V, d] (ingresso = h):
             (a) bnp   : azzera le k "righe/nodi" piu' allineati con b. Il nodo i e' la
                         coordinata i di h: in lm_head e' la COLONNA i. Selezione per b_i
                         (segno, default: i nodi che spingono verso l'errore) o |b_i|.
                         Equivalenza utile: azzerare la colonna i di lm_head == azzerare il
                         peso i della RMSNorm finale (nessuna ri-ternarizzazione necessaria).
             (b) rank1 : W <- W - (W b^) b^T   (togli a lm_head la componente di h lungo b^)
             poi RI-TERNARIZZA W modificata in K piani ternari {-1,0,+1} con scale per
             gruppo (256 ingressi consecutivi per riga): residuo greedy (piano ottimo per
             gruppo, esatto via ordinamento) + rifit ai minimi quadrati delle K scale per
             gruppo + giri di raffinamento a coordinate. Riporta l'errore di ricostruzione e
             quanto della direzione b "rientra" a causa della quantizzazione.
  eval       distribuzione delle lettere e accuratezza sull'insieme heldout PRIMA/DOPO, per:
             originale, originale ternarizzata (effetto della sola quantizzazione),
             modificata float, modificata ri-ternarizzata. Dai soli h salvati: e' ESATTO
             perche' la modifica tocca solo lm_head (logit_lettera = W[lettera] . h).
  all        direction + edit + eval in sequenza.

Sorgenti di W (--w):  file.npy | file.pt | file.safetensors:chiave | hf:/dir/modello |
                      gguf:/file.gguf[:nome_tensore]  (default output.weight, poi token_embd.weight)
ID lettere (--letter-ids "319,350,315,360") — li stampa teacher_client.py; oppure
--tokenizer /dir/hf per ricavarli in contesto.

Esempio:
  python3 e2_bnp_edit.py all --hidden data/hid --w gguf:student.gguf --letter-ids 319,350,315,360 \
      --method rank1 --planes 3 --out out/rank1
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from common import read_jsonl

logger = logging.getLogger("bnp")


# ── Caricamento ──────────────────────────────────────────────────────────────
# Lettore GGUF minimale (solo libreria standard + numpy): gguf-py ufficiale si ferma su
# tutto il file appena incontra un tipo di tensore che non conosce (es. TQ1_0R_1024 = 45 dei
# GGUF Thor: "np.uint32(45) is not a valid GGMLQuantizationType"). Qui si legge solo
# l'indice dei tensori e i byte delle RIGHE richieste (memmap: niente 2 GB in RAM per 4 righe).
_GGUF_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
                10: "<Q", 11: "<q", 12: "<d"}
TQ1_0R_1024 = 45
CODEC_DIR = "@@ROOT:forge@@/kernel-rotazione"     # tq1_0r_1024_codec.py


def gguf_tensor_index(path: str) -> tuple[dict, int]:
    """{nome: (tipo, dims GGUF [ne0=in_feat, ne1=righe, ...], offset relativo)}, inizio dati."""
    import struct
    with open(path, "rb") as f:
        def rd(fmt):
            n = struct.calcsize(fmt)
            return struct.unpack(fmt, f.read(n))[0]

        def rstr():
            return f.read(rd("<Q")).decode("utf-8", "replace")

        def skip_val(t):
            if t == 8:
                rstr()
            elif t == 9:
                at, n = rd("<I"), rd("<Q")
                if at in (8, 9):
                    for _ in range(n):
                        skip_val(at)
                else:
                    f.seek(struct.calcsize(_GGUF_SCALAR[at]) * n, 1)
            else:
                return rd(_GGUF_SCALAR[t])

        if f.read(4) != b"GGUF":
            raise SystemExit(f"{path}: non e' un GGUF")
        rd("<I")
        n_t, n_kv = rd("<Q"), rd("<Q")
        align = 32
        for _ in range(n_kv):
            k, t = rstr(), rd("<I")
            v = skip_val(t)
            if k == "general.alignment":
                align = int(v)
        idx = {}
        for _ in range(n_t):
            name = rstr()
            nd = rd("<I")
            dims = [rd("<Q") for _ in range(nd)]
            idx[name] = (rd("<I"), dims, rd("<Q"))
        start = f.tell()
    return idx, (start + align - 1) // align * align


def _gguf_row_bytes(ty: int, ne0: int) -> tuple[int, int]:
    """(byte per blocco, elementi per blocco) del tipo."""
    if ty == TQ1_0R_1024:
        return 224, 1024
    if ty == 0:
        return 4, 1
    if ty in (1, 30):                      # F16, BF16
        return 2, 1
    from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType
    bs, ts = GGML_QUANT_SIZES[GGMLQuantizationType(ty)]
    return ts, bs


def head_planes(idx: dict, name: str) -> list[str]:
    """Piani ternari di una testa Thor: in qwen35moe.cpp (thor-velocita) i logit sono
    output(h) + output2(h) + output3(h) [+ output4(h)] — leggere solo output.weight da' il
    PRIMO piano, non lm_head (3/10: trovato su R2B3m, che ha output/output2/output3)."""
    stem = name[:-len(".weight")] if name.endswith(".weight") else name
    return [name] + [f"{stem}{k}.weight" for k in (2, 3, 4) if f"{stem}{k}.weight" in idx]


def load_gguf_rows(path: str, name: str = "", rows: Optional[list[int]] = None,
                   sum_planes: bool = True) -> torch.Tensor:
    """W [V, d] float32 (o solo le righe `rows`, nell'ordine dato) da un GGUF, tipi Thor compresi.
    Con sum_planes (default) e nome output.weight/token_embd.weight somma i piani 2..4 se ci sono."""
    idx, data0 = gguf_tensor_index(path)
    if sum_planes:
        cands0 = [name] if name else ["output.weight", "token_embd.weight"]
        base = next((c for c in cands0 if c in idx), None)
        planes = head_planes(idx, base) if base else []
        if len(planes) > 1:
            logger.info("testa a %d piani: %s (somma)", len(planes), planes)
            W = None
            for pn in planes:
                Wp = load_gguf_rows(path, pn, rows, sum_planes=False)
                W = Wp if W is None else W + Wp
            return W
    cands = [name] if name else ["output.weight", "token_embd.weight"]
    cand = next((c for c in cands if c in idx), None)
    if cand is None:
        raise SystemExit(f"tensore non trovato in {path}: {name or 'output.weight'}")
    if cand == "token_embd.weight" and not name:
        logger.warning("output.weight assente: embedding LEGATI. Modificare lm_head "
                       "significa slegarlo (copia separata in output.weight).")
    ty, dims, off = idx[cand]
    ne0, n_rows = int(dims[0]), int(np.prod(dims[1:]))
    blk_b, blk_e = _gguf_row_bytes(ty, ne0)
    if ne0 % blk_e:
        raise SystemExit(f"{cand}: in_feat {ne0} non multiplo del blocco {blk_e}")
    rb = ne0 // blk_e * blk_b
    mm = np.memmap(path, dtype=np.uint8, mode="r", offset=data0 + off, shape=(n_rows, rb))
    sel = list(range(n_rows)) if rows is None else [int(r) for r in rows]
    out = np.empty((len(sel), ne0), dtype=np.float32)
    step = 8192
    for i in range(0, len(sel), step):
        raw = np.ascontiguousarray(mm[sel[i:i + step]])
        if ty == TQ1_0R_1024:
            if CODEC_DIR not in sys.path:
                sys.path.insert(0, CODEC_DIR)
            from tq1_0r_1024_codec import dequantize_blocks
            out[i:i + len(raw)] = dequantize_blocks(raw).reshape(len(raw), ne0)
        elif ty == 0:
            out[i:i + len(raw)] = raw.view(np.float32)
        elif ty == 1:
            out[i:i + len(raw)] = raw.view(np.float16).astype(np.float32)
        elif ty == 30:
            out[i:i + len(raw)] = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        else:
            from gguf.constants import GGMLQuantizationType
            from gguf.quants import dequantize
            out[i:i + len(raw)] = dequantize(raw, GGMLQuantizationType(ty)).reshape(len(raw), ne0)
    del mm
    logger.info("GGUF %s tipo %d: %d righe x %d", cand, ty, len(sel), ne0)
    return torch.from_numpy(out)


def load_matrix(spec: str, rows: Optional[list[int]] = None) -> torch.Tensor:
    """Carica W_out come float32 [V, d] (V = vocabolario, d = dimensione di h).
    Con `rows` (solo gguf) carica soltanto quelle righe, nell'ordine dato."""
    if spec.startswith("gguf:"):
        rest = spec[5:]
        path, _, name = rest.partition(":")
        return load_gguf_rows(path, name, rows)
    if spec.startswith("hf:"):
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(spec[3:], torch_dtype=torch.float32)
        return m.get_output_embeddings().weight.detach().float().clone()
    if spec.startswith("gguf:"):
        from gguf import GGUFReader
        from gguf.quants import dequantize
        rest = spec[5:]
        path, _, name = rest.partition(":") if rest.count(":") else (rest, "", "")
        r = GGUFReader(path)
        tensors = {t.name: t for t in r.tensors}
        for cand in ([name] if name else ["output.weight", "token_embd.weight"]):
            if cand in tensors:
                t = tensors[cand]
                if cand == "token_embd.weight" and not name:
                    logger.warning("output.weight assente: embedding LEGATI. Modificare lm_head "
                                   "significa slegarlo (copia separata in output.weight).")
                try:
                    arr = dequantize(t.data, t.tensor_type)
                except NotImplementedError as e:
                    raise SystemExit(f"tipo {t.tensor_type!r} non dequantizzabile da gguf-py: "
                                     f"esportare W in .npy col proprio strumento ({e})")
                return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
        raise SystemExit(f"tensore non trovato in {path}: {name or 'output.weight'}")
    if ".safetensors" in spec:
        from safetensors.torch import load_file
        path, _, key = spec.partition(".safetensors")
        key = key.lstrip(":") or "lm_head.weight"
        return load_file(path + ".safetensors")[key].float()
    if spec.endswith(".npy"):
        return torch.from_numpy(np.load(spec).astype(np.float32))
    obj = torch.load(spec, map_location="cpu")
    if isinstance(obj, dict):
        obj = obj.get("lm_head.weight", obj.get("weight"))
    return obj.float()


def load_hidden(prefix: str) -> tuple[np.ndarray, list[dict]]:
    H = np.load(prefix + ".npy").astype(np.float32)
    meta = read_jsonl(prefix + ".meta.jsonl")
    assert len(meta) == len(H), "hidden e meta disallineati"
    return H, meta


def parse_letter_ids(args) -> list[int]:
    if args.letter_ids:
        return [int(x) for x in args.letter_ids.split(",")]
    if args.tokenizer:
        from transformers import AutoTokenizer
        from e1_letter_kd import hf_letter_ids
        return hf_letter_ids(AutoTokenizer.from_pretrained(args.tokenizer))
    raise SystemExit("servono --letter-ids o --tokenizer")


# ── Logit delle lettere e metriche ───────────────────────────────────────────
def letter_logits(H: torch.Tensor, W_letters: torch.Tensor, softcap: Optional[float] = None,
                  scale: float = 1.0) -> torch.Tensor:
    """[N,4] = (H @ W_letters^T) * scale, con softcap opzionale (Gemma: cap*tanh(x/cap))."""
    z = (H @ W_letters.T) * scale
    if softcap:
        z = softcap * torch.tanh(z / softcap)
    return z


def letter_metrics(logits4: torch.Tensor, meta: list[dict]) -> dict:
    """Accuratezza, istogramma delle lettere predette, prob. media per lettera, RStd
    (dev. std. del richiamo per lettera d'oro, x100 — metrica usata in letteratura sul
    selection bias) e tasso di fluttuazione fra rotazioni (stesso item, contenuto scelto
    diverso fra le 4 rotazioni)."""
    gold = torch.tensor([m["gold_pos"] for m in meta])
    perm = torch.tensor([m["perm"] for m in meta])
    p = torch.softmax(logits4.float(), -1)
    pred = p.argmax(-1)
    acc = float((pred == gold).float().mean())
    hist = torch.bincount(pred, minlength=4).float() / len(pred)
    recall = []
    for L in range(4):
        sel = gold == L
        recall.append(float((pred[sel] == L).float().mean()) if sel.any() else float("nan"))
    rstd = float(np.nanstd(recall) * 100)
    # contenuto scelto = perm[pred]
    chosen = perm.gather(1, pred[:, None]).squeeze(1)
    groups: dict[str, list[int]] = {}
    for i, m in enumerate(meta):
        groups.setdefault(m["group_id"], []).append(i)
    full = [ix for ix in groups.values() if len(ix) == 4]
    fluct = float(np.mean([len(set(chosen[ix].tolist())) > 1 for ix in full])) if full else float("nan")
    # accuratezza "a contenuto" con media delle 4 rotazioni (limite superiore del debias)
    if full:
        pc = torch.empty_like(p).scatter_(1, perm, p)
        ok = [int(pc[ix].mean(0).argmax()) == meta[ix[0]]["gold"] for ix in full]
        acc_avg = float(np.mean(ok))
    else:
        acc_avg = float("nan")
    return {"acc": acc, "pred_hist_ABCD": [round(float(x), 4) for x in hist],
            "mean_prob_ABCD": [round(float(x), 4) for x in p.mean(0)],
            "recall_by_gold_ABCD": [round(r, 4) for r in recall], "RStd": round(rstd, 3),
            "fluctuation": fluct, "acc_rotation_avg": acc_avg, "n": len(meta)}


# ── Direzione di bias ────────────────────────────────────────────────────────
def bias_direction(H: torch.Tensor, logits4: torch.Tensor, meta: list[dict]) -> tuple[torch.Tensor, dict]:
    gold = torch.tensor([m["gold_pos"] for m in meta])
    wrong = logits4.argmax(-1) != gold
    if wrong.all() or (~wrong).all():
        raise SystemExit("servono sia risposte giuste sia sbagliate per calcolare b")
    b = H[wrong].mean(0) - H[~wrong].mean(0)
    info = {"n_wrong": int(wrong.sum()), "n_correct": int((~wrong).sum()),
            "b_norm": float(b.norm()), "h_mean_norm": float(H.norm(dim=1).mean())}
    return b, info


# ── Modifiche ────────────────────────────────────────────────────────────────
def edit_bnp(W: torch.Tensor, b: torch.Tensor, k: int, align: str = "signed") -> tuple[torch.Tensor, list[int]]:
    """Azzera le k colonne di lm_head (= nodi/coordinate di h) piu' allineate con b."""
    score = b if align == "signed" else b.abs()
    idx = torch.topk(score, k).indices
    W2 = W.clone()
    W2[:, idx] = 0.0
    return W2, idx.tolist()


def edit_rank1(W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """W <- W - (W b^) b^T : lm_head diventa cieca alla componente di h lungo b^."""
    bh = (b / b.norm()).to(W.dtype)
    return W - torch.outer(W @ bh, bh)


# ── Ri-ternarizzazione: K piani ternari, scale per gruppo ────────────────────
def _best_plane(r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Piano ternario ottimo per gruppo dato il residuo r [R,G,g]:
    min_{a,T} ||r - a T||^2 con T in {-1,0,1}. Per un supporto di m elementi l'ottimo
    prende i m |r| piu' grandi, a = media di quei |r|, guadagno = (somma)^2 / m:
    si sceglie m* che massimizza il guadagno — esatto, via ordinamento + somme cumulate."""
    a = r.abs()
    s, idx = a.sort(dim=-1, descending=True)
    c = s.cumsum(-1)
    m = torch.arange(1, r.shape[-1] + 1, device=r.device, dtype=r.dtype)
    gain = c * c / m
    mstar = gain.argmax(-1, keepdim=True)                       # [R,G,1], supporto = mstar+1
    alpha = c.gather(-1, mstar) / (mstar + 1).to(r.dtype)       # [R,G,1]
    keep_sorted = (torch.arange(r.shape[-1], device=r.device) <= mstar).to(r.dtype)
    mask = torch.zeros_like(r).scatter_(-1, idx, keep_sorted)
    T = torch.sign(r) * mask
    T = torch.where(alpha > 0, T, torch.zeros_like(T))          # gruppo nullo: piano nullo
    return T, alpha.squeeze(-1)


def _refit_scales(w: torch.Tensor, planes: torch.Tensor, ridge: float = 1e-6) -> torch.Tensor:
    """Minimi quadrati per gruppo: min_s ||w - sum_k s_k T_k||^2. planes [K,R,G,g], w [R,G,g]."""
    K = planes.shape[0]
    Gm = torch.einsum("krgi,lrgi->rgkl", planes, planes)
    rhs = torch.einsum("krgi,rgi->rgk", planes, w)
    eye = torch.eye(K, device=w.device, dtype=w.dtype)
    tr = Gm.diagonal(dim1=-2, dim2=-1).mean(-1, keepdim=True)[..., None]
    s = torch.linalg.solve(Gm + ridge * (tr + 1e-12) * eye, rhs[..., None]).squeeze(-1)
    return s.permute(2, 0, 1).contiguous()                       # [K,R,G]


def ternarize_planes(W: torch.Tensor, K: int = 3, group: int = 256, n_refine: int = 2,
                     chunk_rows: int = 8192, device: str = "cpu",
                     scale_dtype: torch.dtype = torch.float16) -> dict:
    """W [R,C] ~= sum_k scale_k[r, c//group] * T_k[r,c].  Ritorna piani int8 [K,R,Cpad],
    scale [K,R,G] (scale_dtype) ed errori (calcolati con le scale GIA' arrotondate)."""
    R, C = W.shape
    G = math.ceil(C / group)
    Cp = G * group
    planes_out = torch.zeros(K, R, Cp, dtype=torch.int8)
    scales_out = torch.zeros(K, R, G, dtype=scale_dtype)
    err2 = tot2 = 0.0
    for r0 in range(0, R, chunk_rows):
        w = W[r0:r0 + chunk_rows].to(device, torch.float32)
        if Cp != C:
            w = torch.nn.functional.pad(w, (0, Cp - C))
        w = w.view(w.shape[0], G, group)
        # 1) greedy sul residuo
        planes = []
        res = w.clone()
        for _ in range(K):
            T, a = _best_plane(res)
            planes.append(T)
            res = res - a[..., None] * T
        P = torch.stack(planes)                                  # [K,R,G,g]
        s = _refit_scales(w, P)
        # 2) raffinamento a coordinate: ripiano k ottimo dato il resto, poi rifit di tutte
        for _ in range(n_refine):
            for k in range(K):
                others = (s[..., None] * P).sum(0) - s[k][..., None] * P[k]
                Tk, _ = _best_plane(w - others)
                P[k] = Tk
                s = _refit_scales(w, P)
        s_q = s.to(scale_dtype).to(torch.float32)               # errore con la precisione salvata
        rec = (s_q[..., None] * P).sum(0)
        err2 += float(((w - rec) ** 2).sum())
        tot2 += float((w ** 2).sum())
        n = w.shape[0]
        planes_out[:, r0:r0 + n] = P.view(K, n, Cp).to(torch.int8).cpu()
        scales_out[:, r0:r0 + n] = s.to(scale_dtype).cpu()
    return {"planes": planes_out, "scales": scales_out, "group": group, "shape": (R, C),
            "rel_err": math.sqrt(err2 / max(tot2, 1e-30))}


def reconstruct(tern: dict, rows: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Ricostruisce W (o solo le righe `rows`) da piani+scale."""
    P, S = tern["planes"], tern["scales"].float()
    if rows is not None:
        P, S = P[:, rows], S[:, rows]
    K, R, Cp = P.shape
    g = tern["group"]
    W = (S[..., None] * P.view(K, R, Cp // g, g).float()).sum(0).view(R, Cp)
    return W[:, :tern["shape"][1]]


def reconstruction_report(W_target: torch.Tensor, tern: dict, b: torch.Tensor,
                          W_orig: torch.Tensor, letter_ids: list[int], chunk: int = 16384) -> dict:
    """Errore relativo (Frobenius), errore sulle righe lettera, coseno medio per riga e
    'rientro del bias': ||W_t b^|| / ||W_orig b^|| (0 = direzione eliminata anche dopo la
    quantizzazione; per rank1 il float da' 0 esatto)."""
    bh = b / b.norm()
    num = den = 0.0
    cos_sum = 0.0
    leak_num = leak_den = 0.0
    R = W_target.shape[0]
    for r0 in range(0, R, chunk):
        idx = torch.arange(r0, min(R, r0 + chunk))
        Wt = reconstruct(tern, idx)
        Wf = W_target[idx]
        num += float(((Wf - Wt) ** 2).sum())
        den += float((Wf ** 2).sum())
        cos_sum += float(torch.nn.functional.cosine_similarity(Wf, Wt, dim=1).nan_to_num(1.0).sum())
        leak_num += float(((Wt @ bh) ** 2).sum())
        leak_den += float(((W_orig[idx] @ bh) ** 2).sum())
    li = torch.tensor(letter_ids)
    Wl = reconstruct(tern, li)
    return {"rel_err_fro": math.sqrt(num / den), "mean_row_cos": cos_sum / R,
            "letter_rows_rel_err": float((W_target[li] - Wl).norm() / W_target[li].norm()),
            "bias_leak_ratio": math.sqrt(leak_num / max(leak_den, 1e-30))}


# ── Sottocomandi ─────────────────────────────────────────────────────────────
def split_hidden(H: np.ndarray, meta: list[dict], split: str) -> tuple[torch.Tensor, list[dict]]:
    ix = [i for i, m in enumerate(meta) if m["split"] == split]
    return torch.from_numpy(H[ix]), [meta[i] for i in ix]


def cmd_all(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    H, meta = load_hidden(args.hidden)
    lids_vocab = parse_letter_ids(args)
    if args.rows == "letters" and args.w.startswith("gguf:"):
        # le modifiche (bnp, rank1) e le scale per riga agiscono riga per riga: caricare le sole
        # 4 righe lettera da' gli STESSI numeri e non porta in RAM il vocabolario intero
        W = load_matrix(args.w, rows=lids_vocab)
        lids = list(range(len(lids_vocab)))
    else:
        W = load_matrix(args.w)
        lids = lids_vocab
    logger.info("W %s, h %s, lettere %s (righe usate %s)", tuple(W.shape), H.shape, lids_vocab, lids)
    assert W.shape[1] == H.shape[1], "d di W e di h non coincidono (W va data come [V, d])"
    li = torch.tensor(lids)
    Wl = W[li]
    report: dict = {"args": {k: v for k, v in vars(args).items() if k != "func"},
                    "letter_ids_vocab": lids_vocab}

    # coerenza fra h raccolti e logprob del maestro, se lo studente E' il maestro
    if args.check_teacher and all("teacher_logprobs" in m for m in meta):
        z = torch.log_softmax(letter_logits(torch.from_numpy(H), Wl, args.softcap, args.logit_scale), -1)
        t = torch.tensor([m["teacher_logprobs"] for m in meta])
        report["check_vs_server"] = {"argmax_agree": float((z.argmax(-1) == t.argmax(-1)).float().mean()),
                                     "max_abs_logprob_diff": float((z - t).abs().max())}
        logger.info("controllo h vs server: %s", report["check_vs_server"])

    # 1) direzione
    Htr, mtr = split_hidden(H, meta, "train")
    b, binfo = bias_direction(Htr, letter_logits(Htr, Wl, args.softcap, args.logit_scale), mtr)
    torch.save(b, out / "bias_direction.pt")
    report["direction"] = binfo
    logger.info("direzione: %s", binfo)

    # 2) modifica
    if args.method == "bnp":
        W_edit, nodes = edit_bnp(W, b, args.k, args.align)
        report["bnp_nodes"] = nodes
    else:
        W_edit = edit_rank1(W, b)
    # 3) ri-ternarizzazione (e, per confronto, ternarizzazione dell'originale)
    rows = torch.arange(W.shape[0]) if args.rows == "all" else li
    logger.info("ternarizzo %d righe, K=%d, gruppo %d, raffinamenti %d", len(rows), args.planes,
                args.group, args.refine)
    tern_edit = ternarize_planes(W_edit[rows], args.planes, args.group, args.refine,
                                 args.chunk_rows, args.device)
    tern_orig = ternarize_planes(W[rows], args.planes, args.group, args.refine,
                                 args.chunk_rows, args.device)
    lrows = [int((rows == i).nonzero()[0]) for i in lids]          # posizione lettere in `rows`
    report["recon_edited"] = reconstruction_report(W_edit[rows], tern_edit, b, W[rows], lrows)
    report["recon_original"] = reconstruction_report(W[rows], tern_orig, b, W[rows], lrows)
    logger.info("ricostruzione modificata: %s", report["recon_edited"])
    logger.info("ricostruzione originale : %s", report["recon_original"])
    if args.save_planes:
        torch.save({**tern_edit, "rows": rows, "letter_ids_vocab": lids_vocab, "method": args.method},
                   out / "edited_ternary.pt")

    # 4) valutazione heldout
    Hho, mho = split_hidden(H, meta, "heldout")
    variants = {"original": Wl, "original_ternary": reconstruct(tern_orig, torch.tensor(lrows)),
                "edited_float": W_edit[li], "edited_ternary": reconstruct(tern_edit, torch.tensor(lrows))}
    report["eval_heldout"] = {}
    for name, Wv in variants.items():
        mt = letter_metrics(letter_logits(Hho, Wv, args.softcap, args.logit_scale), mho)
        report["eval_heldout"][name] = mt
        logger.info("%-17s acc %.4f  RStd %.2f  fluct %.3f  pred %s", name, mt["acc"], mt["RStd"],
                    mt["fluctuation"], mt["pred_hist_ABCD"])
    (out / "report.json").write_text(json.dumps(report, indent=2))
    logger.info("report: %s", out / "report.json")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["all"], help="(direction+edit+eval in un passo)")
    ap.add_argument("--hidden", required=True, help="prefisso di e2_collect_hidden.py")
    ap.add_argument("--w", required=True, help="sorgente di lm_head [V, d]")
    ap.add_argument("--letter-ids", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--method", choices=["bnp", "rank1"], default="rank1")
    ap.add_argument("--k", type=int, default=32, help="nodi azzerati (bnp)")
    ap.add_argument("--align", choices=["signed", "abs"], default="signed")
    ap.add_argument("--planes", type=int, default=3, help="K piani ternari")
    ap.add_argument("--group", type=int, default=256)
    ap.add_argument("--refine", type=int, default=2)
    ap.add_argument("--rows", choices=["all", "letters"], default="all",
                    help="'letters' = solo le 4 righe (veloce, basta per la misura)")
    ap.add_argument("--chunk-rows", type=int, default=8192)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--softcap", type=float, default=None, help="final logit softcap (Gemma)")
    ap.add_argument("--logit-scale", type=float, default=1.0, help="es. Granite logits_scaling")
    ap.add_argument("--check-teacher", action="store_true",
                    help="confronta i logprob da h con teacher_logprobs (se studente==maestro)")
    ap.add_argument("--save-planes", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(0)
    return cmd_all(args)


if __name__ == "__main__":
    sys.exit(main())

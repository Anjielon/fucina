#!/usr/bin/env python3
"""P0 del PROGETTO_TQ1_0R_512 (25/9/2026): quale FORMATO ternario per ffn_down_exps, misurato sull'errore d'USCITA.

Angelo: «il mio sogno e' avere il pieno ternario migliore». Thor-Q0 (down a 4-5 bit) ha mostrato che il down ternario pesa su PPL e
sonda. Prima di scrivere kernel si decide QUALE formato serve:
    G256 piano (oggi, TQ1_0) · G128 piano · rot512 G256 · rot512 G128 (TQ1_0R_512)
Misura: errore relativo d'uscita  sum_e tr(dW_e H_e dW_e^T) / sum_e tr(W_e H_e W_e^T)
con H_e = Hessiana VERA dell'ingresso del down per esperto (hessiane-down/Hdown_LL.npy, catturata il 19/9 dal checkpoint HF bf16
con transformers: NON toccata dal bug B9) e W_e = pesi bf16 veri (thor-src-hf, layout HF fuso [E, 2048 out, 512 in]).
Rotazione sull'asse d'ingresso: R = Had512 * diag(S) / sqrt(512), W' = W R^T, H' = R H R^T (y = W x = W' (R x)).

Due quantizzatori, uguali per tutti i bracci (confronto di FORMATO, non di motore):
  (a) ternario ottimo per l'errore sui pesi, per gruppo (esatto: ordinamento + somme cumulate);
  (b) ternario pesato dall'importanza degli ingressi (diag di H), stessa ricerca.
Scale arrotondate a fp16 come nel GGUF. Solo CPU (la GPU fa le valutazioni), 8 thread, nice.
Controlli che devono saper fallire: (1) un peso gia' ternario da' errore ~0; (2) la rotazione conserva tr(W H W^T).
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
import torch
from safetensors import safe_open

torch.set_num_threads(int(os.environ.get("P0_THREAD", "8")))
SRC = Path("@@ROOT:models@@/gguf/thor-src-hf")
HESS = Path("@@ROOT:models@@/gguf/tony-tern/hessiane-down")
OUT = Path("@@ROOT:forge@@/thor_v2/risultati/p0_formati_down.json")
STRATI = [int(x) for x in os.environ.get("P0_STRATI", "5,11,20,23").split(",")]
BLOCCO_E = 16


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def carica_down(L: int) -> torch.Tensor:
    mappa = json.load(open(SRC / "model.safetensors.index.json"))["weight_map"]
    k = f"model.language_model.layers.{L}.mlp.experts.down_proj"
    with safe_open(str(SRC / mappa[k]), framework="pt") as f:
        w = f.get_tensor(k)
    assert w.dtype == torch.bfloat16, w.dtype
    if w.shape[1] == 512 and w.shape[2] == 2048:      # [E, in, out] → [E, out, in]
        w = w.transpose(1, 2)
    assert tuple(w.shape[1:]) == (2048, 512), w.shape
    return w.float().contiguous()


def hadamard(n: int) -> torch.Tensor:
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


def ternario(W: torch.Tensor, g: int, imp: torch.Tensor | None) -> torch.Tensor:
    """W [..., n] → ricostruzione ternaria per gruppi di g lungo l'ultimo asse; imp (stessa forma di W) o None."""
    sh = W.shape
    w = W.reshape(-1, g)
    a = w.abs()
    h = torch.ones_like(a) if imp is None else imp.reshape(-1, g).clamp_min(1e-12)
    ordine = torch.argsort(a, dim=1, descending=True)
    a_o = torch.gather(a, 1, ordine); h_o = torch.gather(h, 1, ordine)
    s_ha = torch.cumsum(h_o * a_o, 1); s_h = torch.cumsum(h_o, 1)
    guadagno = s_ha * s_ha / s_h                       # riduzione dell'errore pesato scegliendo i primi k
    k = torch.argmax(guadagno, 1)                      # indice 0-based: si tengono k+1 elementi
    d = (s_ha.gather(1, k[:, None]) / s_h.gather(1, k[:, None])).squeeze(1)
    d = d.half().float()                               # scala fp16 come nel GGUF
    rango = torch.empty_like(ordine); rango.scatter_(1, ordine, torch.arange(g).expand_as(ordine).contiguous())
    tieni = rango <= k[:, None]
    q = torch.sign(w) * tieni
    return (q * d[:, None]).reshape(sh)


def err_uscita(W: torch.Tensor, Wq: torch.Tensor, H: torch.Tensor) -> tuple[float, float]:
    num = den = 0.0
    for i in range(0, W.shape[0], BLOCCO_E):
        dW = W[i:i + BLOCCO_E] - Wq[i:i + BLOCCO_E]
        num += float(((dW @ H[i:i + BLOCCO_E]) * dW).sum(dtype=torch.float64))
        Wb = W[i:i + BLOCCO_E]
        den += float(((Wb @ H[i:i + BLOCCO_E]) * Wb).sum(dtype=torch.float64))
    return num, den


def main() -> None:
    gen = torch.Generator().manual_seed(0)
    S = (torch.randint(0, 2, (512,), generator=gen) * 2 - 1).float()
    R = hadamard(512) * S[None, :] / (512 ** 0.5)       # righe ortonormali
    assert torch.allclose(R @ R.T, torch.eye(512), atol=1e-5), "rotazione non ortonormale"

    # controllo 1 (deve saper fallire): un peso GIA' ternario per gruppi da 128 → errore ~0 con G128
    Wt = torch.sign(torch.randn(4, 64, 512)) * (torch.rand(4, 64, 512) > 0.3)
    Wt = Wt * torch.repeat_interleave(torch.rand(4, 64, 4) + 0.5, 128, dim=2).half().float()
    Hid = torch.eye(512).expand(4, 512, 512).contiguous()
    n, d = err_uscita(Wt, ternario(Wt, 128, None), Hid)
    assert n / d < 1e-6, f"controllo 1 fallito: {n / d}"
    log(f"controllo 1 ok (peso gia' ternario: errore {n / d:.2e})")

    risultati = {}
    for L in STRATI:
        t0 = time.time()
        W = carica_down(L)                                                        # [256, 2048, 512]
        H = torch.from_numpy(np.load(HESS / f"Hdown_{L:02d}.npy")).float()      # [256, 512, 512]
        Wr = W @ R.T
        Hr = R @ H @ R.T
        _, den = err_uscita(W, torch.zeros_like(W), H)
        _, den_r = err_uscita(Wr, torch.zeros_like(Wr), Hr)
        assert abs(den - den_r) / den < 1e-4, f"controllo 2 fallito allo strato {L}: {den} contro {den_r}"
        imp = torch.diagonal(H, dim1=1, dim2=2)[:, None, :].expand_as(W)
        imp_r = torch.diagonal(Hr, dim1=1, dim2=2)[:, None, :].expand_as(Wr)
        r = {}
        for nome, (Wx, Hx, impx, g) in {
            "G256_piano": (W, H, imp, 256), "G128_piano": (W, H, imp, 128),
            "rot512_G256": (Wr, Hr, imp_r, 256), "rot512_G128": (Wr, Hr, imp_r, 128),
        }.items():
            for qn, im in (("a_pesi", None), ("b_importanza", impx)):
                num, dd = err_uscita(Wx, ternario(Wx, g, im), Hx)
                r[f"{nome}/{qn}"] = num / dd
        risultati[L] = r
        tipo = "pieno" if L % 4 == 3 else "lineare"
        log(f"strato {L} ({tipo}) in {time.time() - t0:.0f} s: " +
            " · ".join(f"{k} {v * 100:.2f}%" for k, v in r.items()))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"strati": risultati, "nota": "errore relativo d'uscita, H vere per esperto"}, indent=1))
    medie = {k: float(np.mean([risultati[L][k] for L in risultati])) for k in next(iter(risultati.values()))}
    base = medie["G256_piano/b_importanza"]
    log("MEDIA sugli strati (rispetto a G256 piano, quantizzatore b):")
    for k, v in medie.items():
        log(f"   {k:28s} {v * 100:6.2f}%   {(v / base - 1) * 100:+6.1f}%")


if __name__ == "__main__":
    main()

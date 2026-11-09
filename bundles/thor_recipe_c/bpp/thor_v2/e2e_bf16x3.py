#!/usr/bin/env python3
"""bf16x3 — prodotti fp32 fatti con tre GEMM bf16 (27/9/2026, ricetta «B3» di `e2e_studente` e `--testa-kl bf16x3`
di `e2e_allena`).

Perche': sulla GPU della corsa (torch 2.11 rocm 7.13, gfx1151) la GEMM fp32 fa 1,3-2,2 TFLOPS e la bf16 20-27
(misura del 27/9). Nella ricetta B le proiezioni NON-esperte (attenzione, GDN in_proj/out_proj, esperto
condiviso) e la testa KL (logit dello studente, dL/dh) sono GEMM fp32: ~56% dei FLOP del passo.

Schema (Henry, Tang, Heinecke, arXiv 1904.06376; JAX `Precision.HIGH` / `DotAlgorithmPreset.BF16_BF16_F32_X3`):
    x = hi + lo,  hi = bf16(x),  lo = bf16(x - hi)          (x - hi e' esatto in fp32)
    a b ~= hi_a hi_b + hi_a lo_b + lo_a hi_b                 (lo_a lo_b ~ 2^-16 |a b| si scarta)
Ogni prodotto bf16 x bf16 e' ESATTO in fp32 (8 + 8 bit di mantissa <= 24); le somme sono fp32. Si sommano prima
i due termini piccoli, poi quello grande. Errore atteso ~1e-5 relativo (prova X1 di `e2e_test_bf16x3.py`, col
numero misurato), contro ~1e-6/1e-7 della GEMM fp32.

Via della GEMM bf16 -> fp32 (`via_mm`, provata una volta per dispositivo):
- "triton" (27/9 pomeriggio, la via della GPU dopo il cancello bocciato delle 11:04): UN kernel Triton fa i tre
  prodotti (`_k_x3`). La A fp32 si scompone NEI REGISTRI (hi/lo con arrotondamento al pari, gli stessi bit di
  `scomponi`), i tre `tl.dot` bf16 vanno sulle WMMA, e ogni riquadro di K (BK) si accumula in un accumulatore
  PROPRIO, sommato al totale con un'addizione fp32 dei vettori (arrotondata al pari). Perche' cosi' (misure del
  27/9 su gfx1151, `TECNICHE_E2E_PV.md` §bf16x3):
    * `mm(out_dtype=float32)` e' ESATTA ma non usa le WMMA: 2,4 TFLOPS come la GEMM fp32 (5,4 con hipBLASLt),
      contro 16-34 della bf16 con uscita bf16. Tre di quelle = 2-3 volte PIU' LENTO di fp32: il cancello
      l'ha visto (x1,06-1,88).
    * l'accumulo DENTRO la WMMA (v_wmma_f32_16x16x16_bf16, e `acc += tl.dot(...)` che Triton fonde in quella)
      TRONCA verso zero: somma di 16384 prodotti positivi -2,2e-5 relativo di deriva (fp32: 8e-9). Col
      riquadro in un accumulatore suo (`z = acc * 0.0` come accumulatore iniziale: Triton non lo riconosce
      come zero e non fonde l'addizione) la deriva sparisce: 2,6e-9.
- "out_dtype": `torch.mm(a, b, out_dtype=torch.float32)` e `torch.addmm(c, a, b, out_dtype=torch.float32)`
  (aten::mm.dtype / addmm.dtype, solo CUDA/ROCm in torch 2.11): ingressi bf16, accumulo e uscita fp32; la somma
  dei tre termini sta nell'epilogo della GEMM (beta = 1), niente passate in piu' sulla memoria.
- "uscita_bf16" (SOLO se la prima manca): `torch.mm` bf16 (accumulo fp32 dentro hipBLASLt, uscita arrotondata a
  bf16) convertita a fp32 e sommata in fp32. L'arrotondamento dell'uscita (2^-9 relativo del termine grande)
  domina: errore ~1e-3, come una GEMM bf16 semplice (prova X1, numero dichiarato): bf16x3 in questa via NON
  serve. Registrata in `stato()`; il cancello B3 la scrive.
- "emula" (CPU, prove): a.float() @ b.float() — i valori bf16 sono esatti in fp32 e i prodotti pure: la stessa
  aritmetica di "out_dtype" salvo l'ordine delle somme fp32.
`E2E_BF16X3_VIA` = triton | out_dtype | uscita_bf16 | emula forza la via (prove).

`LineareBF16x3`: sostituisce un `nn.Linear` CONGELATO. Il peso si scompone UNA volta (hi, lo in bf16: stessa
memoria del peso fp32, che si libera). Autograd: avanti y = x W^T in bf16x3; indietro dX = dY W con la stessa
scomposizione di dY; niente dW (i pesi non-esperti sono fermi) e niente x salvato per l'indietro.

`LineareK2` (27/9, SOLO per il braccio "rumore" del cancello B3, ricetta `Bk2_ind32`): la stessa proiezione in
fp32 ma con la somma su K spezzata in due meta' (y = x_1 W_1^T + x_2 W_2^T; dX idem). Aritmetica fp32 come B,
solo un altro ordine delle somme: quanto B stessa si sposta per un arrotondamento fp32 diverso (il "pavimento").
"""
from __future__ import annotations

import os
from typing import Optional

import torch

VIA_FORZATA = os.getenv("E2E_BF16X3_VIA", "")
_VIA: dict = {}          # dispositivo -> {"via": ..., "addmm": bool, ...}


def scomponi(x: torch.Tensor) -> tuple:
    """x fp32 -> (hi, lo) bf16 con x ~= hi + lo (errore <= 2^-17 |x| circa)."""
    if x.dtype != torch.float32:
        raise TypeError(f"bf16x3: ingresso {x.dtype}, atteso float32")
    hi = x.to(torch.bfloat16)
    lo = (x - hi.float()).to(torch.bfloat16)
    return hi, lo


# ═══════════════════════════════════ via "triton" ═══════════════════════════════════
try:  # Triton c'e' nel venv della forgia; senza, la via triton manca (la prova la scarta, si ripiega)
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None

if triton is not None:
    @triton.jit
    def _k_x3(a_ptr, sa_m, sa_k, bh_ptr, bl_ptr, sb_k, sb_n, c_ptr, sc_s, sc_m, sc_n, M, N, K, KS,
              A_FP32: tl.constexpr, HA_LO: tl.constexpr, ADD_C: tl.constexpr, K_PARI: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr):
        """C (+)= A B con B = hi (+ lo) bf16. A fp32 (A_FP32: scomposta nei registri) o bf16. Prodotti del
        riquadro BK: lo_a hi_b + hi_a lo_b + hi_a hi_b nello stesso accumulatore di riquadro (piccoli prima),
        poi acc = acc + riquadro con l'addizione fp32 dei vettori (niente accumulo troncato della WMMA).
        Asse 1 della griglia = spezzone di K (split-K, KS elementi, multiplo di BK): uscita parziale in c + s sc_s."""
        pid = tl.program_id(0)
        pid_s = tl.program_id(1)
        npm = tl.cdiv(M, BM)
        npn = tl.cdiv(N, BN)
        ning = GM * npn
        fm = (pid // ning) * GM
        gsm = min(npm - fm, GM)
        pid_m = fm + ((pid % ning) % gsm)
        pid_n = (pid % ning) // gsm
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        m_m = rm < M
        m_n = rn < N
        rm64 = rm.to(tl.int64)
        rn64 = rn.to(tl.int64)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        k_ini = pid_s * KS
        k_fin = tl.minimum(K, k_ini + KS)
        for k0 in range(k_ini, k_fin, BK):
            rk = k0 + tl.arange(0, BK)
            if K_PARI:
                ma = m_m[:, None]
                mb = m_n[None, :]
            else:
                ma = m_m[:, None] & (rk < K)[None, :]
                mb = (rk < K)[:, None] & m_n[None, :]
            a = tl.load(a_ptr + rm64[:, None] * sa_m + rk[None, :].to(tl.int64) * sa_k, mask=ma, other=0.0)
            boff = rk[:, None].to(tl.int64) * sb_k + rn64[None, :] * sb_n
            bh = tl.load(bh_ptr + boff, mask=mb, other=0.0)
            z = acc * 0.0                      # accumulatore del riquadro: zero che Triton non fonde con acc
            if A_FP32:
                ah = a.to(tl.bfloat16, fp_downcast_rounding="rtne")
                al = (a - ah.to(tl.float32)).to(tl.bfloat16, fp_downcast_rounding="rtne")
                if HA_LO:
                    bl = tl.load(bl_ptr + boff, mask=mb, other=0.0)
                    r = tl.dot(al, bh, z)
                    r = tl.dot(ah, bl, r)
                    r = tl.dot(ah, bh, r)
                else:
                    r = tl.dot(al, bh, z)
                    r = tl.dot(ah, bh, r)
            else:
                if HA_LO:
                    bl = tl.load(bl_ptr + boff, mask=mb, other=0.0)
                    r = tl.dot(a, bl, z)
                    r = tl.dot(a, bh, r)
                else:
                    r = tl.dot(a, bh, z)
            acc = acc + r
        coff = pid_s.to(tl.int64) * sc_s + rm64[:, None] * sc_m + rn64[None, :] * sc_n
        cm = m_m[:, None] & m_n[None, :]
        if ADD_C:
            acc = tl.load(c_ptr + coff, mask=cm, other=0.0) + acc
        tl.store(c_ptr + coff, acc, mask=cm)


def _kernel():
    if triton is None:
        raise RuntimeError("bf16x3: triton assente, via triton impossibile")
    return _k_x3, triton


# Configurazione FISSA (niente autotune: stessa forma -> stesso riquadro -> stessi bit a ogni ripresa). Scelta
# dalla spazzata del 27/9 su gfx1151 (8192x2048x8192, accanto alla corsa): 64x64x32, 4 warp, 1 stadio = 46-48 ms
# contro 112-124 della GEMM fp32; i riquadri 128x128 traboccano i registri (0,3 TFLOPS).
CONF_TRITON = {"BM": 64, "BN": 64, "BK": 32, "GM": 8, "num_warps": 4, "num_stages": 1}
# split-K (anche questo deterministico: dipende solo dalla forma): quando i riquadri d'uscita sono pochi e K e'
# lungo (dL/dh della testa: 512 x 248320 x 2048 = 256 riquadri, 7760 giri ciascuno: 351 ms contro 307 della fp32)
# K si spezza in S parti, uscite parziali fp32 (S, M, N) sommate da torch in fp32. Misura 27/9 (accanto alla
# corsa): S = 1, 2, 4, 8, 16 -> 180, 164, 145, 123, 98 ms (fp32 260), errore 4,5e-6 in tutti.
SPLITK_PROGRAMMI = 4096      # si spezza finche' riquadri x S <= questo
SPLITK_K_MIN = 16384         # ... e solo se K e' almeno questo
SPLITK_MAX = 16


def spezzoni_k(M: int, N: int, K: int) -> int:
    """Quante parti di K (1 = niente split-K) per la forma (M, N, K), con CONF_TRITON."""
    cf = CONF_TRITON
    prog = -(-M // cf["BM"]) * -(-N // cf["BN"])
    if K < SPLITK_K_MIN or prog >= SPLITK_PROGRAMMI:
        return 1
    return int(max(1, min(SPLITK_MAX, SPLITK_PROGRAMMI // prog, K // (4 * cf["BK"]))))


def mm_triton(a: torch.Tensor, b_hi: torch.Tensor, b_lo: Optional[torch.Tensor] = None,
              c: Optional[torch.Tensor] = None) -> torch.Tensor:
    """(c +) a @ (b_hi + b_lo) fp32, 2D. a fp32 (bf16x3: tre prodotti, o due senza b_lo) oppure bf16 (uno o due
    prodotti: i suoi valori sono esatti). b_hi/b_lo bf16 (K, N), stessi passi (anche viste trasposte). c fp32 (M, N):
    si somma e si scrive sul posto (come addmm con beta = 1)."""
    k, triton = _kernel()
    M, K = a.shape
    if b_hi.shape[0] != K:
        raise ValueError(f"mm_triton: a {tuple(a.shape)} contro b {tuple(b_hi.shape)}")
    if a.dtype not in (torch.float32, torch.bfloat16) or b_hi.dtype != torch.bfloat16:
        raise TypeError(f"mm_triton: a {a.dtype}, b {b_hi.dtype} (attesi float32|bfloat16 e bfloat16)")
    if b_lo is not None and (b_lo.stride() != b_hi.stride() or b_lo.shape != b_hi.shape
                             or b_lo.dtype != torch.bfloat16):
        raise ValueError("mm_triton: b_hi e b_lo con forma/passi/tipo diversi")
    N = b_hi.shape[1]
    if c is None:
        out = torch.empty(M, N, device=a.device, dtype=torch.float32)
    else:
        if c.dtype != torch.float32 or tuple(c.shape) != (M, N):
            raise ValueError(f"mm_triton: c {c.dtype} {tuple(c.shape)}")
        out = c
    if M == 0 or N == 0 or K == 0:
        return out if c is not None else out.zero_()
    cf = CONF_TRITON
    S = spezzoni_k(M, N, K)
    KS = -(-K // S)
    KS = -(-KS // cf["BK"]) * cf["BK"]            # multiplo di BK: ogni spezzone comincia su un riquadro intero
    S = -(-K // KS)
    dest = out if S == 1 else torch.empty(S, M, N, device=a.device, dtype=torch.float32)
    sd = (0, dest.stride(0), dest.stride(1)) if S == 1 else dest.stride()
    grid = (triton.cdiv(M, cf["BM"]) * triton.cdiv(N, cf["BN"]), S)
    k[grid](a, a.stride(0), a.stride(1), b_hi, b_hi if b_lo is None else b_lo, b_hi.stride(0), b_hi.stride(1),
            dest, sd[0], sd[1], sd[2], M, N, K, KS,
            A_FP32=a.dtype == torch.float32, HA_LO=b_lo is not None, ADD_C=c is not None and S == 1,
            K_PARI=K % cf["BK"] == 0, BM=cf["BM"], BN=cf["BN"], BK=cf["BK"], GM=cf["GM"],
            num_warps=cf["num_warps"], num_stages=cf["num_stages"])
    if S > 1:
        if c is None:
            torch.sum(dest, 0, out=out)
        else:
            out.add_(dest.sum(0))
    return out


def _prova_triton(device: torch.device) -> dict:
    """La via triton sul dispositivo vero: bf16x3 contro fp64 (<= 1e-5 relativo in norma), la A bf16 (esatta salvo
    l'ordine: <= 1e-6) e la deriva dell'accumulo (somma lunga di prodotti positivi: <= 1e-6, la WMMA che tronca ne
    darebbe ~2e-5)."""
    out = {}
    try:
        g = torch.Generator(device=device).manual_seed(0)
        a = torch.randn(300, 2080, device=device, generator=g)
        w = torch.randn(200, 2080, device=device, generator=g) * 0.02
        h, lo = scomponi(w)
        rif = a.double() @ w.double().t()
        y = mm_triton(a, h.t(), lo.t())
        out["err_x3"] = float((y.double() - rif).norm() / rif.norm())
        ab = a.to(torch.bfloat16)
        rif1 = ab.double() @ h.double().t()
        out["err_bf16"] = float((mm_triton(ab, h.t()).double() - rif1).norm() / rif1.norm())
        ap = (torch.rand(32, 16384, device=device, generator=g) + 1).to(torch.bfloat16)
        bp = (torch.rand(16384, 64, device=device, generator=g) + 1).to(torch.bfloat16)
        rifp = ap.double() @ bp.double()
        out["deriva_accumulo"] = float(((mm_triton(ap, bp).double() - rifp) / rifp).mean())
        out["ok"] = bool(out["err_x3"] <= 1e-5 and out["err_bf16"] <= 1e-6 and abs(out["deriva_accumulo"]) <= 1e-6)
    except Exception as ex:                  # triton assente o rotto: si ripiega (registrato)
        out["ok"] = False
        out["errore"] = repr(ex)[:300]
    return out


def _prova_via(device: torch.device) -> dict:
    """Sul dispositivo vero: esiste mm.dtype (bf16 -> fp32)? E addmm.dtype? L'uscita e' quella dell'accumulo fp32
    (confronto con l'emulazione fp32, stessa aritmetica salvo l'ordine)? Senza mm.dtype: la GEMM bf16 accumula
    in fp32 prima di arrotondare l'uscita? (quota di uscite == bf16(emulazione))."""
    g = torch.Generator(device=device).manual_seed(0)
    a = torch.randn(256, 512, device=device, generator=g).to(torch.bfloat16)
    b = torch.randn(512, 128, device=device, generator=g).to(torch.bfloat16)
    rif = a.float() @ b.float()
    scala = float(rif.abs().max())
    out = {"dispositivo": str(device)}
    try:
        y = torch.mm(a, b, out_dtype=torch.float32)
        out["mm_out_dtype"] = y.dtype == torch.float32
        out["err_mm_out_dtype"] = float((y - rif).abs().max()) / scala
    except Exception as ex:
        out["mm_out_dtype"] = False
        out["errore_mm_out_dtype"] = repr(ex)[:200]
    try:
        c = torch.randn(256, 128, device=device, generator=g)
        y = torch.addmm(c, a, b, out_dtype=torch.float32)
        out["addmm_out_dtype"] = y.dtype == torch.float32
        out["err_addmm_out_dtype"] = float((y - (c + rif)).abs().max()) / scala
    except Exception as ex:
        out["addmm_out_dtype"] = False
        out["errore_addmm_out_dtype"] = repr(ex)[:200]
    y16 = torch.mm(a, b)
    out["uscita_bf16_uguale_a_bf16_accumulo_fp32"] = float((y16 == rif.to(torch.bfloat16)).float().mean())
    ok = out["mm_out_dtype"] and out.get("err_mm_out_dtype", 1.0) < 1e-5
    out["via"] = "out_dtype" if ok else "uscita_bf16"
    out["addmm"] = bool(ok and out["addmm_out_dtype"] and out.get("err_addmm_out_dtype", 1.0) < 1e-5)
    out["triton"] = _prova_triton(device)
    if out["triton"]["ok"]:
        out["via"] = "triton"
    return out


def via_mm(device) -> dict:
    device = torch.device(device)
    k = str(device)
    if k not in _VIA:
        if VIA_FORZATA:
            _VIA[k] = {"dispositivo": k, "via": VIA_FORZATA, "addmm": VIA_FORZATA == "out_dtype", "forzata": True}
            if VIA_FORZATA not in ("triton", "out_dtype", "uscita_bf16", "emula"):
                raise ValueError(f"E2E_BF16X3_VIA={VIA_FORZATA!r}: triton | out_dtype | uscita_bf16 | emula")
        elif device.type == "cpu":
            _VIA[k] = {"dispositivo": k, "via": "emula", "addmm": False}
        else:
            _VIA[k] = _prova_via(device)
    return _VIA[k]


def stato() -> dict:
    """Le vie scelte finora (per l'esito del cancello e il registro)."""
    return dict(_VIA)


def mm_bf16(a: torch.Tensor, b: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
    """(c +) a @ b con a, b bf16 2D: prodotti esatti, accumulo fp32, uscita fp32 (salvo la via "uscita_bf16")."""
    v = via_mm(a.device)
    via = v["via"]
    if via == "triton":
        return mm_triton(a, b, None, c)
    if via == "out_dtype":
        if c is None:
            return torch.mm(a, b, out_dtype=torch.float32)
        if v["addmm"]:
            return torch.addmm(c, a, b, out_dtype=torch.float32)
        return c.add_(torch.mm(a, b, out_dtype=torch.float32))
    if via == "uscita_bf16":
        y = torch.mm(a, b).float()
    elif via == "emula":
        y = a.float() @ b.float()
    else:
        raise ValueError(f"E2E_BF16X3_VIA={via!r}: triton | out_dtype | uscita_bf16 | emula")
    return y if c is None else c.add_(y)


def mm_x3(a_hi, a_lo, b_hi, b_lo) -> torch.Tensor:
    """a @ b in bf16x3 dalle scomposizioni: (lo_a hi_b + hi_a lo_b) + hi_a hi_b, fp32."""
    y = mm_bf16(a_lo, b_hi)
    y = mm_bf16(a_hi, b_lo, y)
    return mm_bf16(a_hi, b_hi, y)


def matmul_x3(a: torch.Tensor, b_hi: torch.Tensor, b_lo: torch.Tensor) -> torch.Tensor:
    """a (..., K) fp32 @ b (K, N) gia' scomposto -> (..., N) fp32."""
    forma = a.shape
    if via_mm(a.device)["via"] == "triton":
        if a.dtype != torch.float32:
            raise TypeError(f"bf16x3: ingresso {a.dtype}, atteso float32")
        return mm_triton(a.reshape(-1, forma[-1]), b_hi, b_lo).reshape(*forma[:-1], b_hi.shape[1])
    ah, al = scomponi(a.reshape(-1, forma[-1]))
    y = mm_x3(ah, al, b_hi, b_lo)
    del ah, al
    return y.reshape(*forma[:-1], b_hi.shape[1])


class _LineareX3(torch.autograd.Function):
    """y = x W^T (+ b) in bf16x3; dX = dY W in bf16x3. Nessun gradiente per W (congelato); x non si salva."""

    @staticmethod
    def forward(ctx, x, lin: "LineareBF16x3"):
        ctx.lin = lin
        y = matmul_x3(x, lin.hi.t(), lin.lo.t())
        if lin.bias is not None:
            y = y + lin.bias
        return y

    @staticmethod
    def backward(ctx, gy):
        lin = ctx.lin
        dx = None
        if ctx.needs_input_grad[0]:
            dx = matmul_x3(gy.to(torch.float32), lin.hi, lin.lo)
        return dx, None


class LineareBF16x3(torch.nn.Module):
    """nn.Linear congelato con peso (out, in) scomposto una volta in (hi, lo) bf16. hi/lo sono attributi semplici
    (NON buffer: un `.to(dtype)` del modulo non li deve convertire); si costruisce DOPO aver messo lo strato sul
    dispositivo. Ingresso e uscita fp32."""

    def __init__(self, lin: torch.nn.Linear):
        super().__init__()
        W = lin.weight.detach().to(torch.float32)
        self.hi, self.lo = scomponi(W)
        self.bias = None if lin.bias is None else lin.bias.detach().to(torch.float32).clone()
        self.in_features, self.out_features = lin.in_features, lin.out_features

    def forward(self, x):
        if x.dtype != torch.float32:
            raise TypeError(f"LineareBF16x3: ingresso {x.dtype}, atteso float32 (ricetta B3: residuo fp32)")
        return _LineareX3.apply(x, self)

    @torch.no_grad()
    def imposta_peso(self, W: torch.Tensor) -> None:
        """27/9: nuovo peso (out, in) fp32 (ricarica dei byte riquantizzati, `e2e_lora.installa_pesi`)."""
        if tuple(W.shape) != (self.out_features, self.in_features):
            raise ValueError(f"LineareBF16x3: peso {tuple(W.shape)}, atteso {(self.out_features, self.in_features)}")
        self.hi, self.lo = scomponi(W.to(device=self.hi.device, dtype=torch.float32))

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bf16x3"


# Le proiezioni con meno di MIN_USCITE righe restano fp32 esatte: in_proj_b / in_proj_a del GDN (32 righe, le porte
# beta e il decadimento), shared_expert_gate (1 riga). Sono < 0,3% dei FLOP non-esperti; il router non e' un
# nn.Linear (MoEStudente.router_w) e resta fp32 come in B (top-k come llama.cpp).
MIN_USCITE = 256


class _LineareK2(torch.autograd.Function):
    """y = x_1 W_1^T + x_2 W_2^T (K in due meta', fp32); dX = [dY W]: dY_1 W_1 + dY_2 W_2 (uscite in due meta')."""

    @staticmethod
    def forward(ctx, x, lin: "LineareK2"):
        ctx.lin = lin
        h = lin.W.shape[1] // 2
        y = x[..., :h] @ lin.W[:, :h].t()
        y = y + x[..., h:] @ lin.W[:, h:].t()
        if lin.bias is not None:
            y = y + lin.bias
        return y

    @staticmethod
    def backward(ctx, gy):
        lin = ctx.lin
        dx = None
        if ctx.needs_input_grad[0]:
            h = lin.W.shape[0] // 2
            gy = gy.to(torch.float32)
            dx = gy[..., :h] @ lin.W[:h]
            dx = dx + gy[..., h:] @ lin.W[h:]
        return dx, None


class LineareK2(torch.nn.Module):
    """nn.Linear CONGELATO fp32 con la somma su K in due meta' (braccio "rumore" del cancello B3: stessa aritmetica
    fp32 di B, un altro ordine delle somme). Come `LineareBF16x3`: niente dW, niente x salvato, ingresso fp32."""

    def __init__(self, lin: torch.nn.Linear):
        super().__init__()
        self.W = lin.weight.detach().to(torch.float32).clone()
        self.bias = None if lin.bias is None else lin.bias.detach().to(torch.float32).clone()
        self.in_features, self.out_features = lin.in_features, lin.out_features

    def forward(self, x):
        if x.dtype != torch.float32:
            raise TypeError(f"LineareK2: ingresso {x.dtype}, atteso float32")
        return _LineareK2.apply(x, self)

    @torch.no_grad()
    def imposta_peso(self, W: torch.Tensor) -> None:
        if tuple(W.shape) != tuple(self.W.shape):
            raise ValueError(f"LineareK2: peso {tuple(W.shape)}, atteso {tuple(self.W.shape)}")
        self.W = W.to(device=self.W.device, dtype=torch.float32).clone()

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, fp32 K/2"


def sostituisci_lineari(modulo: torch.nn.Module, min_uscite: int = MIN_USCITE, classe=None) -> list:
    """Sostituisce sul posto ogni nn.Linear con >= min_uscite righe con `classe` (default `LineareBF16x3`; il
    braccio rumore del cancello: `LineareK2`). Ritorna i nomi."""
    classe = LineareBF16x3 if classe is None else classe
    fatti = []
    for nome, m in list(modulo.named_modules()):
        for figlio_nome, figlio in list(m.named_children()):
            if isinstance(figlio, torch.nn.Linear) and figlio.out_features >= min_uscite:
                setattr(m, figlio_nome, classe(figlio))
                fatti.append(f"{nome}.{figlio_nome}" if nome else figlio_nome)
    return fatti


class TestaX3:
    """Testa dello studente (V, H) scomposta UNA volta (hi, lo bf16): `e2e_allena --testa-kl bf16x3`. Espone
    shape / device / dtype come il tensore fp32 che sostituisce (chi conta i bersagli fuori testa legge shape)."""

    def __init__(self, W: torch.Tensor):
        self.shape = W.shape
        self.device = W.device
        self.dtype = torch.float32
        self.hi, self.lo = scomponi(W.to(torch.float32))

    def logit(self, h: torch.Tensor) -> torch.Tensor:
        """h (P, H) fp32 -> h W^T (P, V) fp32."""
        return matmul_x3(h, self.hi.t(), self.lo.t())

    def per(self, p: torch.Tensor) -> torch.Tensor:
        """p (P, V) fp32 -> p W (P, H) fp32 (dL/dh della KL)."""
        return matmul_x3(p, self.hi, self.lo)

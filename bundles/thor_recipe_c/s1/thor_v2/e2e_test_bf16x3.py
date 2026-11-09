#!/usr/bin/env python3
"""Prove di `e2e_bf16x3.py` e della ricetta B3 (27/9/2026), SOLO CPU (la GEMM bf16 -> fp32 si EMULA: a.float() @
b.float(), stessi prodotti esatti e accumulo fp32 della via "out_dtype" della GPU, salvo l'ordine delle somme).

Uso:
    nice -n 15 env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=4 TRITON_INTERPRET=1 \\
        @@ROOT:venv@@/bin/python3 e2e_test_bf16x3.py        [E2E_PROVE=X1,X3 per un sottoinsieme]
    esito: $E2E_BF16X3_DIR/esito_test_bf16x3.json (default @@ROOT:models@@/gguf/tony-tern/e2e_studente/)

Prove (pesi VERI di Thor-T dal GGUF, attivazioni vere dove c'e' un ingresso vero):
  X1  errore relativo contro fp64, forme vere (8192 token x proiezioni di GDN, attenzione, esperto condiviso;
      testa 512 x 2048 x V): bf16x3 contro fp32, e (per dichiararlo) bf16 semplice e la via di ripiego
      "uscita_bf16" (mm bf16 con uscita arrotondata). Soglia: bf16x3 <= 3e-5 (norma), fp32 << bf16x3 << bf16.
  X2  gradiente: dX di `LineareBF16x3` contro autograd fp64 (e contro fp32); nessun dW, nessun tensore salvato;
      TestaX3.per (dL/dh) contro fp64.
  X3  B IDENTICA AL BIT alla revisione di riferimento (E2E_RIF_REV, default 2b4ca22: prima di B3): avanti e
      dL/dtheta di B e B_ind32 su 2 strati veri (0 GDN, 3 pieno); `kl_pezzo`/`KLFusa` con teste fp32 al bit.
      B3 contro B: lineari sostituite, scostamento dello strato ~1e-5.
  X4  ricette e G1: ricetta_indietro_fp32(B3) == B3_ind32; `confronta_dtheta` == codice di G1 di prima;
      ConfigAllena / riga di comando (--ricetta B3_ind32 --testa-kl bf16x3 --logit-maestro bf16).
  X5  logit del maestro bf16 (una GEMM bf16, accumulo fp32) contro `e2e_teacher.logit_maestro` fp32 e fp64, testa
      bf16 VERA del teacher e h_T VERI della cache q8-EB-pieno.
  X6  testa KL: KL per token e dL/dh di `kl_pezzo` con testa bf16x3 (+ maestro bf16) contro fp32 e fp64 (dL/dh
      e' una differenza piccola: l'errore dei logit si amplifica; soglia <= 2e-4 e <= 10 x quello della via fp32).
  X7  (27/9 pomeriggio, SOLO GPU: senza GPU si registra "saltata") la via "triton" di `e2e_bf16x3`: scelta dalla
      prova del dispositivo; errore contro fp64 <= 1e-5 (norma) su tutte le forme non-esperte (avanti e dX, 2048
      token) anche con ingressi tutti positivi (la WMMA tronca l'accumulo: deriva); la scomposizione nei registri
      == `scomponi` (contro i tre prodotti fp64 dei suoi hi/lo: <= 1e-6); split-K della testa (dL/dh) e c += a b;
      determinismo AL BIT (due chiamate); `LineareBF16x3` avanti/indietro; piu' veloce della fp32 a 8192x2048x8192.
      Uso: E2E_PROVE=X7 TRITON_INTERPRET=0 con la GPU (env -u HSA_OVERRIDE_GFX_VERSION, niente CUDA_VISIBLE_DEVICES=).
  X8  braccio rumore del cancello: `LineareK2` (fp32, K in due meta') avanti/dX contro fp64 all'ordine di fp32, niente
      parametri ne' tensori salvati; ricetta Bk2_ind32 = B_ind32 + non_esperti_k2 (e B/B_ind32/B3 senza).
"""
from __future__ import annotations

import importlib.util
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("TRITON_INTERPRET", "1")
import numpy as np  # noqa: E402
import torch  # noqa: E402

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import e2e_bf16x3 as X3  # noqa: E402
import e2e_studente as S  # noqa: E402
import e2e_allena as A  # noqa: E402

DIR = Path(os.getenv("E2E_BF16X3_DIR", "@@ROOT:models@@/gguf/tony-tern/e2e_studente"))
RIF_REV = os.getenv("E2E_RIF_REV", "2b4ca22")
TEACHER = Path(os.getenv("E2E_TEACHER", "@@ROOT:lavoro@@/e2e/teacher/q8-EB-pieno"))
torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))
ESITO: dict = {"data": time.strftime("%Y-%m-%d %H:%M:%S"), "gguf": str(S.GGUF_THOR_T), "prove": {}}
_TENS = None


def tens() -> dict:
    global _TENS
    if _TENS is None:
        from gguf import GGUFReader
        _TENS = {t.name: t for t in GGUFReader(str(S.GGUF_THOR_T)).tensors}
    return _TENS


def registra(nome: str, ok: bool, **dati) -> bool:
    ESITO["prove"][nome] = {"ok": bool(ok), **dati}
    print(f"[{nome}] {'✅' if ok else '⛔'} {json.dumps(dati, ensure_ascii=False, default=str)[:1500]}", flush=True)
    return ok


def err(y, rif) -> dict:
    """Errori relativi: in norma di Frobenius e max|y - rif| / max|rif|."""
    y, rif = y.double(), rif.double()
    d = y - rif
    return {"norma": float(d.norm() / rif.norm()), "max": float(d.abs().max() / rif.abs().max())}


def ids_veri(n: int, riga: int = 0) -> torch.Tensor:
    c = torch.load("@@ROOT:models@@/gguf/tony-tern/calibrazione/e2e/e2e_heldout_4096_tutte.pt", weights_only=True)
    return c["domini"]["edu"][riga:riga + 1, :n].clone()


def con_via(via: str):
    """Forza la via della GEMM bf16 sulla CPU (solo per misurare il ripiego)."""
    class _C:
        def __enter__(self):
            self.v = X3._VIA.get("cpu")
            X3._VIA["cpu"] = {"dispositivo": "cpu", "via": via, "addmm": False}

        def __exit__(self, *a):
            if self.v is None:
                X3._VIA.pop("cpu", None)
            else:
                X3._VIA["cpu"] = self.v
    return _C()


def testa_studente_righe(n: int) -> torch.Tensor:
    """Le prime n righe della testa dello studente (Q5_K dequantizzata, fp32)."""
    from dequant_gguf_torch import _dequant_raw
    t = tens()["output.weight"]
    return torch.from_numpy(_dequant_raw(np.asarray(t.data)[:n], t)).float()


def teacher_righe(n: int) -> torch.Tensor:
    """Le prime n righe della testa bf16 del teacher, bf16 esatte."""
    import e2e_teacher as T
    t = T._tensore_gguf(T.TESTA_BF16, "output.weight")
    V, H = int(t.shape[1]), int(t.shape[0])
    mm = np.memmap(T.TESTA_BF16, dtype=np.int16, mode="r", offset=int(t.data_offset), shape=(V, H))
    return torch.from_numpy(np.array(mm[:n])).view(torch.bfloat16)


def h_teacher(n: int) -> torch.Tensor:
    src = A.SorgenteCache(TEACHER)
    i = A.prime_disponibili(src, "heldout", 1)[0]
    return src.h("heldout", i)[:n].clone()


# ─────────────────────────────── X1 ───────────────────────────────
def prova_x1() -> bool:
    torch.manual_seed(1)
    M = int(os.getenv("E2E_X1_TOKEN", "8192"))
    s0 = S.pesi_non_esperti(tens(), 0, False)
    s3 = S.pesi_non_esperti(tens(), 3, True)
    # ingresso VERO delle proiezioni d'ingresso dello strato 0: embedding di 4096 token veri -> input_layernorm
    from dequant_gguf_torch import _dequant_raw
    ids = torch.cat([ids_veri(4096, 0), ids_veri(4096, 1)], 1).reshape(-1)[:M]
    t = tens()["token_embd.weight"]
    un = torch.unique(ids)
    emb = torch.from_numpy(_dequant_raw(np.asarray(t.data)[un.numpy()], t))[torch.searchsorted(un, ids)].float()
    w_n = s0["input_layernorm.weight"].double()
    e64 = emb.double()
    x_vero = (e64 * torch.rsqrt(e64.pow(2).mean(-1, keepdim=True) + 1e-6) * (1.0 + w_n)).float()
    casi = {
        "gdn.in_proj_qkv 8192x2048x8192 (x vero)": (x_vero, s0["linear_attn.in_proj_qkv.weight"]),
        "gdn.in_proj_z 8192x2048x4096 (x vero)": (x_vero, s0["linear_attn.in_proj_z.weight"]),
        "gdn.out_proj 8192x4096x2048": (torch.randn(M, 4096), s0["linear_attn.out_proj.weight"]),
        "attn.q_proj 8192x2048x8192": (torch.randn(M, 2048), s3["self_attn.q_proj.weight"]),
        "attn.k_proj 8192x2048x512": (torch.randn(M, 2048), s3["self_attn.k_proj.weight"]),
        "shared.gate_proj 8192x2048x512 (x vero)": (x_vero, s0["mlp.shared_expert.gate_proj.weight"]),
        "shared.down_proj 8192x512x2048": (torch.randn(M, 512) * torch.rand(M, 1) * 3,
                                           s0["mlp.shared_expert.down_proj.weight"]),
    }
    del s0, s3
    V = int(os.getenv("E2E_X1_V", "32768"))
    WS = testa_studente_righe(V)
    hS = x_vero[:512] * 3.0
    casi[f"testa logit 512x2048x{V}"] = (hS, WS)
    ris, ok = {}, True
    for nome, (x, W) in casi.items():
        t0 = time.time()
        y64 = x.double() @ W.double().T
        y32 = x @ W.T
        hi, lo = X3.scomponi(W)
        y3 = X3.matmul_x3(x, hi.t(), lo.t())
        yb = (x.to(torch.bfloat16).float() @ W.to(torch.bfloat16).float().T)      # bf16 semplice, accumulo fp32
        r = {"fp32": err(y32, y64), "bf16x3": err(y3, y64), "bf16": err(yb, y64)}
        m = min(x.shape[0], 1024)                                                   # ripiego: uscite bf16 (CPU lenta)
        with con_via("uscita_bf16"):
            yr = X3.matmul_x3(x[:m], hi.t(), lo.t())
        r["bf16x3_ripiego_uscita_bf16"] = err(yr, y64[:m])
        r["secondi"] = round(time.time() - t0, 1)
        ris[nome] = r
        ok &= r["bf16x3"]["norma"] <= 3e-5 and r["fp32"]["norma"] < r["bf16x3"]["norma"] < r["bf16"]["norma"] / 30
        print(f"   X1 {nome}: fp32 {r['fp32']['norma']:.2e}, bf16x3 {r['bf16x3']['norma']:.2e} "
              f"(max {r['bf16x3']['max']:.2e}), bf16 {r['bf16']['norma']:.2e}, ripiego "
              f"{r['bf16x3_ripiego_uscita_bf16']['norma']:.2e}", flush=True)
        del y64, y32, y3, yb, yr
    # dL/dh della testa: (P, V) x (V, H), p = p_S - p_T (valori in [-1, 1], somma 0)
    lS = torch.log_softmax(hS @ WS.T, -1)
    pT = torch.softmax(hS @ WS.T + 0.5 * torch.randn(hS.shape[0], V), -1)
    p = lS.exp() - pT
    hi, lo = X3.scomponi(WS)
    y64 = p.double() @ WS.double()
    r = {"fp32": err(p @ WS, y64), "bf16x3": err(X3.matmul_x3(p, hi, lo), y64),
         "bf16": err(p.to(torch.bfloat16).float() @ WS.to(torch.bfloat16).float(), y64)}
    ris[f"testa dh 512x{V}x2048"] = r
    ok &= r["bf16x3"]["norma"] <= 3e-5 and r["fp32"]["norma"] < r["bf16x3"]["norma"]
    peggiore = max(v["bf16x3"]["norma"] for v in ris.values())
    return registra("X1_errore_contro_fp64", ok, casi=ris, peggiore_bf16x3_norma=peggiore,
                    soglia_bf16x3_norma=3e-5,
                    nota="errore relativo in norma di Frobenius e al massimo; 'ripiego' = via uscita_bf16 "
                         "(mm.dtype assente): arrotonda l'uscita del termine grande a bf16, errore ~bf16")


# ─────────────────────────────── X2 ───────────────────────────────
def prova_x2() -> bool:
    torch.manual_seed(2)
    s0 = S.pesi_non_esperti(tens(), 0, False)
    lin = torch.nn.Linear(2048, 8192, bias=False)
    with torch.no_grad():
        lin.weight.copy_(s0["linear_attn.in_proj_qkv.weight"])
    del s0
    W = lin.weight.detach()
    q = X3.LineareBF16x3(lin)
    x = torch.randn(3, 700, 2048)
    G = torch.randn(3, 700, 8192)
    salvati = []
    x3 = x.clone().requires_grad_(True)
    with torch.autograd.graph.saved_tensors_hooks(lambda t: salvati.append(t.shape) or t, lambda t: t):
        y3 = q(x3)
    (y3 * G).sum().backward()
    x64 = x.double().requires_grad_(True)
    (torch.nn.functional.linear(x64, W.double()) * G.double()).sum().backward()
    x32 = x.clone().requires_grad_(True)
    (torch.nn.functional.linear(x32, W) * G).sum().backward()
    r = {"dx_bf16x3": err(x3.grad, x64.grad), "dx_fp32": err(x32.grad, x64.grad),
         "y_bf16x3": err(y3.detach(), x64.detach() @ W.double().T)}
    senza_parametri = len(list(q.parameters())) == 0
    # dL/dh della testa (TestaX3.per) contro fp64
    WS = testa_studente_righe(16384)
    tx = X3.TestaX3(WS)
    p = torch.softmax(torch.randn(256, 16384) * 3, -1) - torch.softmax(torch.randn(256, 16384) * 3, -1)
    r["testa_per"] = err(tx.per(p), p.double() @ WS.double())
    r["testa_logit"] = err(tx.logit(x[0, :256]), x[0, :256].double() @ WS.double().T)
    ok = (r["dx_bf16x3"]["norma"] <= 3e-5 and r["dx_fp32"]["norma"] < r["dx_bf16x3"]["norma"]
          and r["testa_per"]["norma"] <= 3e-5 and not salvati and senza_parametri
          and x3.grad.shape == x.shape and y3.dtype == torch.float32)
    return registra("X2_gradiente_dx", ok, errori=r, tensori_salvati_per_indietro=salvati,
                    nessun_parametro=senza_parametri)


# ─────────────────────────────── X3 ───────────────────────────────
def _modulo_rif(nome: str):
    """Il modulo `nome` della revisione di riferimento (git show), importato col nome <nome>_rif."""
    radice = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=QUI, capture_output=True, text=True,
                            check=True).stdout.strip()
    rel = QUI.relative_to(radice)
    testo = subprocess.run(["git", "show", f"{RIF_REV}:{rel}/{nome}.py"], cwd=QUI, capture_output=True, text=True,
                           check=True).stdout
    d = Path(tempfile.mkdtemp(prefix="rif_bf16x3_"))
    f = d / f"{nome}_rif.py"
    f.write_text(testo)
    spec = importlib.util.spec_from_file_location(f"{nome}_rif", f)
    m = importlib.util.module_from_spec(spec)
    sys.modules[f"{nome}_rif"] = m
    try:
        spec.loader.exec_module(m)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return m


def _bit(a, b) -> bool:
    """Uguaglianza dei BIT (anche NaN e -0)."""
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.contiguous().view(torch.int32),
                                                                     b.contiguous().view(torch.int32))


def _passo(Smod, ric: str, ids, Gs):
    st = Smod.Studente(strati=[0, 3], ricetta=ric, motore_nome="torch", device="cpu", esperti=slice(0, 16),
                       log=lambda *_: None)
    x0 = st.embedding(ids)
    with torch.no_grad():
        y = st.avanti(ids, x0=x0)["xL"].clone()
    st.gradiente(ids, lambda h: (h * Gs).sum(), x0=x0)
    g = [p.grad.clone() for p in st.parametri()]
    ctl = st.controlla(tollera_trabocco=True)
    assert not ctl["trabocco_fp16"], "trabocco fp16 nella prova: scala del gradiente troppo grande"
    assert all(bool(torch.isfinite(x).all()) for x in g)
    lin = {L: getattr(st.moduli[L], "lineari_x3", None) for L in st.strati}
    del st
    return y, g, lin


def prova_x3() -> bool:
    Srif = _modulo_rif("e2e_studente")
    Arif = _modulo_rif("e2e_allena")
    ids = ids_veri(48, 7)
    Gs = torch.randn(48, 2048, generator=torch.Generator().manual_seed(3))
    ris, ok = {}, True
    uscite = {}
    for ric in ("B", "B_ind32"):
        y_n, g_n, lin = _passo(S, ric, ids, Gs)
        y_r, g_r, _ = _passo(Srif, ric, ids, Gs)
        u = _bit(y_n, y_r) and len(g_n) == len(g_r) and all(_bit(a, b) for a, b in zip(g_n, g_r))
        ris[ric] = {"al_bit": bool(u), "lineari_x3": lin}
        ok &= u and all(v is None for v in lin.values())
        uscite[ric] = (y_n, g_n)
    # B3 contro B: stesso esperto, non-esperti bf16x3
    y3, g3, lin3 = _passo(S, "B3", ids, Gs)
    yB, gB = uscite["B"]
    atteso = {0: ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
                  "mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj", "mlp.shared_expert.down_proj"],
              3: ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                  "mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj", "mlp.shared_expert.down_proj"]}
    x0 = S.Studente(strati=[0], ricetta="B", motore_nome="torch", device="cpu", esperti=slice(0, 1),
                    log=lambda *_: None).embedding(ids)
    ris["B3_contro_B"] = {"lineari": lin3, "delta_uscita": err(y3 - x0, yB - x0), "diversa_da_B": not torch.equal(y3, yB),
                          "dtheta_err_norma_max": max(float((a - b).norm() / b.norm().clamp_min(1e-30))
                                                       for a, b in zip(g3, gB))}
    ok &= (all(sorted(lin3[L]) == sorted(atteso[L]) for L in atteso) and ris["B3_contro_B"]["diversa_da_B"]
           and ris["B3_contro_B"]["delta_uscita"]["norma"] < 1e-3)
    # teste fp32: kl_pezzo / KLFusa al bit con la revisione di riferimento
    torch.manual_seed(33)
    V = 8192
    WS = testa_studente_righe(V)
    WT = teacher_righe(V).float()
    hS = torch.randn(300, 2048) * 2
    hT = (hS + 0.3 * torch.randn(300, 2048)).to(torch.bfloat16)
    kl_ok = True
    bers = torch.randint(0, V, (300,), generator=torch.Generator().manual_seed(34))
    for g in (None, 0.3):
        a = A.kl_pezzo(hS, WS, hT, WT, True, bers, gamma=g, top1=True)
        b = Arif.kl_pezzo(hS, WS, hT, WT, True, bers, gamma=g, top1=True)
        kl_ok &= set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)
    hq = hS.clone().requires_grad_(True)
    tq = A.KLFusa.apply(hq, WS, hT, WT, 128)
    tq.backward()
    hr = hS.clone().requires_grad_(True)
    tr = Arif.KLFusa.apply(hr, WS, hT, WT, 128)
    tr.backward()
    kl_ok &= torch.equal(tq, tr) and torch.equal(hq.grad, hr.grad)
    ris["teste_fp32_al_bit"] = bool(kl_ok)
    ok &= kl_ok
    return registra("X3_B_identica_al_bit", ok, revisione_di_riferimento=RIF_REV, esiti=ris)


# ─────────────────────────────── X4 ───────────────────────────────
def prova_x4() -> bool:
    import dataclasses
    ris = {}
    r32 = A.ricetta_indietro_fp32(S.RICETTE["B3"])
    ris["ind32_di_B3_uguale_a_B3_ind32"] = dataclasses.asdict(r32) == dataclasses.asdict(S.RICETTE["B3_ind32"])
    ris["ind32_di_B_uguale_a_B_ind32"] = dataclasses.asdict(A.ricetta_indietro_fp32(S.RICETTE["B"])) == \
        dataclasses.asdict(S.RICETTE["B_ind32"])
    ris["B_senza_x3"] = not S.RICETTE["B"].non_esperti_x3 and not S.RICETTE["B_ind32"].non_esperti_x3 \
        and not S.RICETTE["C"].non_esperti_x3
    ris["solo_Bk2_con_k2"] = [n for n, r in S.RICETTE.items() if r.non_esperti_k2] == ["Bk2_ind32"]
    # confronta_dtheta == il codice in linea di G1 prima del 27/9
    torch.manual_seed(4)
    d0 = [torch.randn(4, 8, 3) * (torch.rand(4, 8, 3) > 0.1) for _ in range(3)]
    g32 = [torch.randn(4, 8, 3) * (torch.rand(4, 8, 3) > 0.2) for _ in range(3)]
    g16 = [b + 0.3 * torch.randn_like(b) * (torch.rand_like(b) > 0.5) for b in g32]
    g16[0][0] = 0.0
    vive_nz = rotti = nulli = 0
    rel = []
    for d, a, b in zip(d0, g16, g32):
        v = (d != 0) & (b != 0)
        vive_nz += int(v.sum())
        nulli += int((v & (a == 0)).sum())
        rotti += int((v & (a != 0) & (torch.sign(a) != torch.sign(b))).sum())
        rr = ((a - b).abs() / b.abs())[v]
        if rr.numel():
            rel.append(rr.float())
    rel_t = torch.cat(rel)
    vecchio = {"vive_con_g32_non_nullo": vive_nz, "nulli_in_fp16": nulli, "segno_sbagliato": rotti,
               "quota_rotti": (nulli + rotti) / max(vive_nz, 1), "err_rel_mediano": A._kth(rel_t, 0.5),
               "err_rel_p99": A._kth(rel_t, 0.99)}
    ris["confronta_dtheta_come_G1"] = A.confronta_dtheta(d0, g16, g32) == vecchio
    # configurazione e riga di comando
    ca = A.ConfigAllena()
    ris["default_fp32"] = ca.testa_kl == "fp32" and ca.logit_maestro == "fp32"
    try:
        A.ConfigAllena(testa_kl="bf16")
        ris["testa_kl_invalida_rifiutata"] = False
    except ValueError:
        ris["testa_kl_invalida_rifiutata"] = True
    ris["campi_di_decisione"] = {"testa_kl", "logit_maestro"} <= A._CAMPI_DI_DECISIONE
    catturati = {}
    vecchia = A._config_da_arg

    def spia(a):
        catturati["a"] = a
        raise SystemExit(0)
    A._config_da_arg = spia
    try:
        try:
            A.main(["allena", "--corsa", "x", "--teacher", "y", "--ricetta", "B3_ind32", "--testa-kl", "bf16x3",
                    "--logit-maestro", "bf16", "--device", "cpu"])
        except SystemExit:
            pass
    finally:
        A._config_da_arg = vecchia
    a = catturati.get("a")
    ris["cli"] = a is not None and a.ricetta == "B3_ind32" and vecchia(a).testa_kl == "bf16x3" \
        and vecchia(a).logit_maestro == "bf16"
    ok = all(v is True for v in ris.values())
    return registra("X4_ricette_g1_config", ok, **ris)


# ─────────────────────────────── X5 ───────────────────────────────
def prova_x5() -> bool:
    import e2e_teacher as T
    V = int(os.getenv("E2E_X5_V", "65536"))
    WTb = teacher_righe(V)
    hT = h_teacher(512)
    assert hT.dtype == torch.bfloat16
    l64 = hT.double() @ WTb.double().T
    l32 = T.logit_maestro(hT, WTb.float())
    lb = A._logit_maestro(hT, WTb, torch.float32)
    ris = {"fp32_contro_fp64": err(l32, l64), "bf16_contro_fp64": err(lb, l64), "bf16_contro_fp32": err(lb, l32),
           "bf16_uguale_fp32_al_bit_su_cpu": bool(torch.equal(lb, l32)), "dtype": str(lb.dtype),
           "via": X3.via_mm("cpu")["via"]}
    lp = torch.log_softmax(l64, -1)
    for nome, l in (("fp32", l32), ("bf16", lb)):
        q = torch.log_softmax(l.double(), -1)
        ris[f"kl_rif64_{nome}_media"] = float((lp.exp() * (lp - q)).sum(-1).mean())
    ok = (lb.dtype == torch.float32 and ris["bf16_contro_fp64"]["norma"] < 1e-6
          and ris["bf16_contro_fp32"]["norma"] < 1e-6)
    try:
        A._logit_maestro(hT.float(), WTb, torch.float32)
        ris["h_fp32_rifiutato"] = False
        ok = False
    except TypeError:
        ris["h_fp32_rifiutato"] = True
    return registra("X5_logit_maestro_bf16", ok, V=V, token=int(hT.shape[0]), errori=ris,
                    nota="sulla CPU la via e' l'emulazione: stessa aritmetica di logit_maestro; sulla GPU cambia "
                         "solo l'ordine delle somme fp32 (errore dell'ordine di fp32)")


# ─────────────────────────────── X6 ───────────────────────────────
def prova_x6() -> bool:
    V = int(os.getenv("E2E_X6_V", "65536"))
    WS = testa_studente_righe(V)
    WTb = teacher_righe(V)
    WT = WTb.float()
    hT = h_teacher(256)
    torch.manual_seed(6)
    hS = hT.float() * (1 + 0.05 * torch.randn(hT.shape)) + 0.02 * torch.randn(hT.shape)
    r64 = A.kl_pezzo(hS.double(), WS.double(), hT, WT, True)
    r32 = A.kl_pezzo(hS, WS, hT, WT, True)
    tx = X3.TestaX3(WS)
    r3 = A.kl_pezzo(hS, tx, hT, WTb, True)
    r3s = A.kl_pezzo(hS, tx, hT, WT, True)
    ris = {}
    for nome, r in (("fp32", r32), ("bf16x3+maestro_bf16", r3), ("bf16x3+maestro_fp32", r3s)):
        dk = (r["kl"] - r64["kl"]).abs()
        ris[nome] = {"kl_media": float(r["kl"].mean()), "kl_err_ass_max": float(dk.max()),
                     "kl_err_rel_somma": float((r["kl"].sum() - r64["kl"].sum()).abs() / r64["kl"].sum()),
                     "dh": err(r["dh"], r64["dh"])}
    ris["kl_rif64_media"] = float(r64["kl"].mean())
    # CAKLD con la testa bf16x3
    c3 = A.kl_pezzo(hS, tx, hT, WTb, True, gamma=0.3)
    c64 = A.kl_pezzo(hS.double(), WS.double(), hT, WT, True, gamma=0.3)
    ris["cakld_bf16x3"] = {"perdita_err_rel_somma": float((c3["perdita"].sum() - c64["perdita"].sum()).abs()
                                                          / c64["perdita"].sum()), "dh": err(c3["dh"], c64["dh"])}
    # KLFusa con la testa bf16x3 (il gradiente e' dh * g)
    hq = hS.clone().requires_grad_(True)
    A.KLFusa.apply(hq, tx, hT, WTb, 128).backward()
    ris["klfusa_dh"] = err(hq.grad, r64["dh"])
    x3 = ris["bf16x3+maestro_bf16"]
    # dL/dh = (p_S - p_T) W e' una DIFFERENZA piccola (KL ~6e-3): l'errore dei logit (fp32 ~1e-7, bf16x3 ~4e-6
    # relativo, per |logit| ~ 20) passa in p e domina; la GEMM da sola e' in X2 (testa_per ~4e-6). Soglia: <= 2e-4
    # e <= 10 volte l'errore della via fp32 sullo stesso pezzo.
    lim = min(2e-4, 10 * ris["fp32"]["dh"]["norma"])
    ris["soglia_dh"] = lim
    ris["dh_bf16x3_su_dh_fp32"] = x3["dh"]["norma"] / ris["fp32"]["dh"]["norma"]
    ok = (x3["dh"]["norma"] <= lim and x3["kl_err_rel_somma"] <= 1e-4 and ris["cakld_bf16x3"]["dh"]["norma"] <= lim
          and ris["klfusa_dh"]["norma"] <= lim)
    return registra("X6_testa_kl_bf16x3", ok, V=V, token=int(hS.shape[0]), errori=ris)


# ─────────────────────────────── X7 (GPU) ───────────────────────────────
def prova_x7() -> bool:
    if not torch.cuda.is_available():
        return registra("X7_via_triton_gpu", True, saltata="nessuna GPU visibile (prova solo GPU: E2E_PROVE=X7)")
    if os.environ.get("TRITON_INTERPRET") == "1":
        # l'interprete di Triton (quello delle prove su CPU) sui tensori GPU da' numeri senza senso (misurato 27/9:
        # errore 6,9e12): la prova del dispositivo lo scarta e ripiega su out_dtype, ma qui si vuole la via vera
        return registra("X7_via_triton_gpu", True, saltata="TRITON_INTERPRET=1: rilanciare con TRITON_INTERPRET=0")
    dev = torch.device("cuda")
    X3._VIA.pop(str(dev), None)
    X3._VIA.pop("cuda:0", None)
    v = X3.via_mm(torch.zeros(1, device=dev).device)
    ris = {"via": v}
    ok = v["via"] == "triton" and v["triton"]["ok"]
    g = torch.Generator(device=dev).manual_seed(7)
    M = int(os.getenv("E2E_X7_TOKEN", "2048"))
    forme = {"gdn.in_proj_qkv / attn.q_proj": (2048, 8192), "gdn.in_proj_z": (2048, 4096),
             "gdn.out_proj": (4096, 2048), "attn.k_proj / shared gate,up": (2048, 512), "shared.down": (512, 2048)}
    peggiore = 0.0
    for nome, (K, N) in forme.items():
        W = torch.randn(N, K, device=dev, generator=g) * 0.02
        hi, lo = X3.scomponi(W)
        for lato, (x, Bh, Bl, Wf) in (("avanti", (torch.randn(M, K, device=dev, generator=g), hi.t(), lo.t(), W.t())),
                                      ("dX", (torch.randn(M, N, device=dev, generator=g), hi, lo, W))):
            xp = x.abs() + 0.1
            y = X3.matmul_x3(x, Bh, Bl)
            r = {"x3": err(y, x.double() @ Wf.double()), "fp32": err(x @ Wf, x.double() @ Wf.double()),
                 "x3_positivi": err(X3.matmul_x3(xp, Bh, Bl), xp.double() @ Wf.double()),
                 "al_bit_due_chiamate": bool(torch.equal(y, X3.matmul_x3(x, Bh, Bl)))}
            peggiore = max(peggiore, r["x3"]["norma"], r["x3_positivi"]["norma"])
            ok &= r["x3"]["norma"] <= 1e-5 and r["x3_positivi"]["norma"] <= 1e-5 and r["al_bit_due_chiamate"]
            ris[f"{nome} {lato} {M}x{x.shape[1]}x{Bh.shape[1]}"] = r
    # la scomposizione nei registri e' quella di `scomponi`: i tre prodotti dei suoi hi/lo in fp64
    x = torch.randn(512, 2048, device=dev, generator=g)
    W = torch.randn(1024, 2048, device=dev, generator=g) * 0.02
    hi, lo = X3.scomponi(W)
    ah, al = X3.scomponi(x)
    t3 = (ah.double() @ hi.double().t() + ah.double() @ lo.double().t() + al.double() @ hi.double().t())
    ris["scomposizione_nei_registri"] = err(X3.matmul_x3(x, hi.t(), lo.t()), t3)
    ok &= ris["scomposizione_nei_registri"]["norma"] <= 1e-6
    # c += a b (mm_bf16 con c) e a bf16 (logit del maestro)
    ab, wb = x.to(torch.bfloat16), W.to(torch.bfloat16)
    c = torch.randn(512, 1024, device=dev, generator=g)
    rif_c = c.double() + ab.double() @ wb.double().t()
    ris["mm_bf16_con_c"] = err(X3.mm_bf16(ab, wb.t(), c.clone()), rif_c)
    ok &= ris["mm_bf16_con_c"]["norma"] <= 1e-6
    # split-K: dL/dh della testa (P x V x H) con pochi riquadri d'uscita
    V, P = 65536, 64
    WS = torch.randn(V, 2048, device=dev, generator=g) * 0.02
    tx = X3.TestaX3(WS)
    pp = torch.softmax(torch.randn(P, V, device=dev, generator=g) * 3, -1) - \
        torch.softmax(torch.randn(P, V, device=dev, generator=g) * 3, -1)
    S_k = X3.spezzoni_k(P, 2048, V)
    yp = tx.per(pp)
    ris["testa_per_splitk"] = {"spezzoni": S_k, **err(yp, pp.double() @ WS.double()),
                               "al_bit_due_chiamate": bool(torch.equal(yp, tx.per(pp)))}
    ok &= S_k > 1 and ris["testa_per_splitk"]["norma"] <= 1e-5 and ris["testa_per_splitk"]["al_bit_due_chiamate"]
    del WS, tx, pp, yp
    # LineareBF16x3 sulla GPU: avanti e dX contro fp64
    lin = torch.nn.Linear(2048, 4096, bias=False).to(dev)
    q = X3.LineareBF16x3(lin)
    xx = torch.randn(3, 300, 2048, device=dev, generator=g).requires_grad_(True)
    G = torch.randn(3, 300, 4096, device=dev, generator=g)
    (q(xx) * G).sum().backward()
    Wd = lin.weight.detach().double()
    ris["lineare_dx"] = err(xx.grad, G.double() @ Wd)
    ris["lineare_y"] = err(q(xx.detach()), xx.detach().double() @ Wd.t())
    ok &= ris["lineare_dx"]["norma"] <= 1e-5 and ris["lineare_y"]["norma"] <= 1e-5
    # tempo alla forma piu' grande: piu' veloce della GEMM fp32
    x = torch.randn(8192, 2048, device=dev, generator=g)
    W = torch.randn(8192, 2048, device=dev, generator=g) * 0.02
    hi, lo = X3.scomponi(W)

    def _ms(f, n=5):
        f()
        torch.cuda.synchronize()
        ts = []
        for _ in range(n):
            t0 = time.time()
            f()
            torch.cuda.synchronize()
            ts.append(time.time() - t0)
        return sorted(ts)[n // 2] * 1e3
    ris["ms_8192x2048x8192"] = {"fp32": _ms(lambda: x @ W.t()), "x3_triton": _ms(lambda: X3.matmul_x3(x, hi.t(), lo.t()))}
    ris["ms_8192x2048x8192"]["rapporto"] = ris["ms_8192x2048x8192"]["x3_triton"] / ris["ms_8192x2048x8192"]["fp32"]
    ok &= ris["ms_8192x2048x8192"]["rapporto"] < 1.0
    ris["peggiore_x3_norma"] = peggiore
    return registra("X7_via_triton_gpu", ok, **ris)


# ─────────────────────────────── X8 ───────────────────────────────
def prova_x8() -> bool:
    torch.manual_seed(8)
    lin = torch.nn.Linear(2048, 4096, bias=False)
    q = X3.LineareK2(lin)
    W = lin.weight.detach().double()
    x = torch.randn(2, 500, 2048)
    G = torch.randn(2, 500, 4096)
    salvati = []
    xx = x.clone().requires_grad_(True)
    with torch.autograd.graph.saved_tensors_hooks(lambda t: salvati.append(t.shape) or t, lambda t: t):
        y = q(xx)
    (y * G).sum().backward()
    x32 = x.clone().requires_grad_(True)
    y32 = torch.nn.functional.linear(x32, lin.weight.detach())
    (y32 * G).sum().backward()
    r = {"y_k2": err(y.detach(), x.double() @ W.T), "y_fp32": err(y32.detach(), x.double() @ W.T),
         "dx_k2": err(xx.grad, G.double() @ W), "dx_fp32": err(x32.grad, G.double() @ W),
         "diversa_da_fp32": not torch.equal(y.detach(), y32.detach())}
    lim = 5 * max(r["y_fp32"]["norma"], r["dx_fp32"]["norma"])
    ok = (r["y_k2"]["norma"] <= lim and r["dx_k2"]["norma"] <= lim and r["diversa_da_fp32"] and not salvati
          and len(list(q.parameters())) == 0)
    rk = S.RICETTE["Bk2_ind32"]
    import dataclasses
    ok &= dataclasses.replace(rk, nome="B_ind32", non_esperti_k2=False) == S.RICETTE["B_ind32"]
    st = S.Studente(strati=[0], ricetta="Bk2_ind32", motore_nome="torch", device="cpu", esperti=slice(0, 2),
                    log=lambda *_: None)
    lin_k2 = st.moduli[0].lineari_x3
    tipi = {type(m).__name__ for m in st.moduli[0].modules() if isinstance(m, (X3.LineareK2, X3.LineareBF16x3))}
    del st
    ok &= tipi == {"LineareK2"} and len(lin_k2) == 6
    return registra("X8_braccio_rumore_k2", ok, errori=r, soglia=lim, lineari=lin_k2, tipi=sorted(tipi))


def main() -> int:
    t0 = time.time()
    solo = [s for s in os.getenv("E2E_PROVE", "").split(",") if s]
    ok = True
    for nome, f in (("X4", prova_x4), ("X5", prova_x5), ("X6", prova_x6), ("X2", prova_x2), ("X3", prova_x3),
                    ("X1", prova_x1), ("X8", prova_x8), ("X7", prova_x7)):
        if solo and nome not in solo:
            continue
        t1 = time.time()
        try:
            ok &= f()
        except Exception as ex:
            import traceback
            traceback.print_exc()
            ok &= registra(f"{nome}_ECCEZIONE", False, errore=repr(ex))
        print(f"   ({nome}: {time.time() - t1:.1f} s)", flush=True)
    ESITO["secondi"] = round(time.time() - t0, 1)
    ESITO["tutto_verde"] = bool(ok)
    ESITO["picco_rss_GiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
    DIR.mkdir(parents=True, exist_ok=True)
    nome_file = "esito_test_bf16x3.json" if not solo else f"esito_test_bf16x3_{'_'.join(solo)}.json"
    (DIR / nome_file).write_text(json.dumps(ESITO, indent=1, default=str))
    print("ESITO:", "TUTTO VERDE" if ok else "⛔ QUALCOSA NON TORNA", f"({ESITO['secondi']} s)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

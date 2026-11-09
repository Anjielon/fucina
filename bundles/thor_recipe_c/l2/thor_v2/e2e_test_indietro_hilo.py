#!/usr/bin/env python3
"""27/9 (ramo indietro-bf16x3). Prove CPU dell'indietro degli esperti a prodotti divisi (`e2e_indietro_hilo`) e
della LoRA sopra le lineari bf16x3. Famiglie VERE del GGUF (gate/up TQ1_0R_1024, down TQ1_0), TRITON_INTERPRET=1.

  H0  default AL BIT: `_indietro_famiglia` e un passo avanti+indietro di strato (ricette B, B_ind32, B3_ind32)
      identici al bit a e2e_studente del commit prima del ramo (f458cf0, caricato a parte).
  H1  divisioni: scala per potenze di due esatta; |v - hi - lo| <= 2^-24 (fp16x2, |v| < 1); errore di
      rappresentazione di dY per riga / per (esperto, riga di W) <= 2^-23 del massimo, anche con gradienti da
      1e-30 a 1e4 (dove fp16 con una K globale va a zero o trabocca: contati).
  H2  prodotti (motore torch) contro un riferimento fp64 col decoder INDIPENDENTE (e2e_codici_scale): dX fp32 e
      dL/dd di fp16x2, bf16x3 e fp32 (la via di B_ind32), errore relativo in norma e per riga/gruppo.
  H3  kernel VERI nell'interprete (triton2 e triton, fp16): prodotti fp16x2 == motore torch entro l'arrotondamento
      fp32 e contro fp64; bf16x3 con Triton NON si prova qui (l'interprete non ha bf16): microprova GPU del cancello.
  H4  `indietro_famiglia` (gancio) B_ind2/B_ind3 contro la via fp32 di B_ind32: dx (fp16) uguale salvo ulp,
      dL/dd entro l'errore dei prodotti; ricetta di riferimento di G1 (ricetta_indietro_fp32) = via fp32 AL BIT.
  H5  strato vero dello studente (pochi esperti, CPU): avanti B_ind2 == B_ind32 AL BIT, dtheta di B_ind2/B_ind3
      e B_ind32 contro B64 (fp64); B3_ind2 contro B3_ind32.
  H6  LoRA sopra LineareBF16x3 (ricetta B3): prima del ramo `installa` sollevava; ora il ramo sta sopra la lineare
      bf16x3, gradienti di A e B uguali a quelli di B_ind32 entro l'errore di bf16x3, peso base senza gradiente.
  H7  passo V (--pv-codici): `_k_dd_v` senza divisione AL BIT quello di prima (modi 0-3); B_ind2 nei modi 1-2 col
      kernel V diviso (acc intero prima delle statistiche) contro B_ind32: dL/dd, cand, prop, S, Q entro
      l'arrotondamento fp32; modo 3 e bf16x3 sulla via fp32 AL BIT.
  H8  --cambia-ricetta (ripresa di EB2/EB3 con B_ind2 / B3_ind2): ammessi solo i cambi di sola aritmetica.
  H9  (1/10, TT5-ottimizzato) densi a TRE piani di R2B3m (LineareTernaria, piano a esperto unico) con B3_ind2 contro
      B3_ind32: famiglie vere dei tre piani (gancio, torch e triton2), passo V modi 1-3, strato 0 vero (avanti al bit,
      dtheta per piano contro B3_ind32 e B64).

Uso: TRITON_INTERPRET=1 HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= nice -n 19 OMP_NUM_THREADS=4 \\
        python3 e2e_test_indietro_hilo.py [H0 H1 ...]
Esito JSON: $E2E_HL_ESITO (default: accanto a questo file, e2e_test_indietro_hilo_esito.json)."""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("TRITON_INTERPRET", "1")
import torch  # noqa: E402

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import e2e_studente as S  # noqa: E402
import e2e_codici_scale as CS  # noqa: E402
import e2e_indietro_hilo as HL  # noqa: E402
from gguf import GGUFReader  # noqa: E402

torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))
ESITO: dict = {"data": time.strftime("%Y-%m-%d %H:%M:%S"), "gguf": str(S.GGUF_THOR_T), "prove": {}}
RIF_COMMIT = os.getenv("E2E_HL_RIF", "64dcdfd")      # 1/10: codice di TT5 (prima era f458cf0, merge bf16x3fix)
_TENS = None


def tens() -> dict:
    global _TENS
    if _TENS is None:
        _TENS = {t.name: t for t in GGUFReader(str(S.GGUF_THOR_T)).tensors}
    return _TENS


def registra(nome: str, ok: bool, **dati) -> bool:
    ESITO["prove"][nome] = {"ok": bool(ok), **dati}
    print(f"[{nome}] {'✅' if ok else '⛔'} {json.dumps(dati, ensure_ascii=False, default=str)[:2500]}", flush=True)
    return ok


def famiglia(L: int, n: str, esperti=None, righe=None) -> S.DatiFamiglia:
    return S.famiglia_da_tensore(tens()[f"blk.{L}.{S.FAMIGLIE[n]}.weight"], n, esperti=esperti, righe=righe)


def err_norma(a, b) -> float:
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


def err_righe(a, b) -> float:
    """max sulle righe di ||a_r - b_r|| / ||b_r|| (righe non nulle)."""
    a, b = a.double().reshape(b.shape[0], -1), b.double().reshape(b.shape[0], -1)
    n = b.norm(dim=1)
    v = n > 0
    return float(((a - b).norm(dim=1)[v] / n[v]).max()) if bool(v.any()) else 0.0


def gy_estremo(S_: int, R: int, g: torch.Generator) -> torch.Tensor:
    """Gradiente fp32 con righe da 1e-30 a 1e4 (gamma di un indietro vero con la scala K e oltre), e dentro la
    riga elementi a 1e-6 del massimo e zeri."""
    base = torch.randn(S_, R, generator=g)
    esp = torch.empty(S_, 1).uniform_(-30, 4, generator=g)
    v = base * torch.pow(10.0, esp)
    piccoli = torch.rand(S_, R, generator=g) < 0.1
    v[piccoli] *= 1e-6
    v[torch.rand(S_, R, generator=g) < 0.05] = 0.0
    return v.float()


def riferimento_fp64(x, gy, d, fam, piano):
    """(dx, dL/dd) in fp64 col decoder INDIPENDENTE (e2e_codici_scale.trit_da_blocchi), L = sum(y (.) gy)."""
    x = x.double().detach().requires_grad_(True)
    dd = d.double().detach().requires_grad_(True)
    o = piano.offs.tolist()
    ys = []
    for e in range(fam.E):
        q = CS.trit_da_blocchi(fam.blocchi[e].reshape(-1, fam.nb), fam.tipo).double()
        W = (q.view(fam.R, fam.NG, fam.gs) * dd[e].unsqueeze(-1)).view(fam.R, fam.C)
        ys.append(x[o[e]:o[e + 1]] @ W.T)
    (torch.cat(ys) * gy.double()).sum().backward()
    return x.grad, dd.grad


def caso(L, n, esperti, righe, T, k, seme):
    g = torch.Generator().manual_seed(seme)
    fam = famiglia(L, n, esperti=esperti, righe=righe)
    idx = torch.randint(0, fam.E, (T, k), generator=g)
    piano = S.Piano(idx, fam.E)
    x = (torch.randn(piano.S, fam.C, generator=g) * 0.5).half()
    d = fam.d0.half().float()
    gy = gy_estremo(piano.S, fam.R, g)
    return fam, piano, x, d, gy


def _carica_rif():
    src = subprocess.run(["git", "-C", str(QUI), "show", f"{RIF_COMMIT}:thor_v2/e2e_studente.py"], capture_output=True,
                         text=True)
    if src.returncode != 0:
        raise RuntimeError(src.stderr[-400:])
    tmp = tempfile.mkdtemp()
    p = Path(tmp) / "e2e_studente_rif.py"
    p.write_text(src.stdout)
    spec = importlib.util.spec_from_file_location("e2e_studente_rif", p)
    R = importlib.util.module_from_spec(spec)
    sys.modules["e2e_studente_rif"] = R
    spec.loader.exec_module(R)
    return R


# ─────────────────────────────── H0 ───────────────────────────────
def prova_h0() -> bool:
    R = _carica_rif()
    ok, dati = True, {}
    # (a) le ricette di prima non cambiano (salvo il campo nuovo, spento)
    for nome, r in R.RICETTE.items():
        nuova = S.RICETTE[nome]
        uguale = all(getattr(nuova, f.name) == getattr(r, f.name) for f in dataclasses.fields(r)) \
            and nuova.indietro_diviso == ""
        dati[f"ricetta_{nome}"] = uguale
        ok &= uguale
    # (b) _indietro_famiglia al bit (motore torch e triton2 nell'interprete)
    for nome_ric in ("B", "B_ind32", "B3_ind32"):
        for mot_n in ("torch", "triton2"):
            fam, piano, x, d, gy = caso(0, "gate", slice(0, 3), slice(0, 80), 40, 2, 10)
            ric = S.RICETTE[nome_ric]
            gy_ = gy * (2.0 ** -20 if ric.dtype_indietro is None else 1.0)     # fp16: dentro la gamma
            a = S._indietro_famiglia(gy_, x, d, fam, piano, ric, S.motore(mot_n), None, True, True)
            b = R._indietro_famiglia(gy_, x, d, R.famiglia_da_tensore(tens()["blk.0.ffn_gate_exps.weight"], "gate",
                                                                       esperti=slice(0, 3), righe=slice(0, 80)),
                                     piano, R.RICETTE[nome_ric], R.motore(mot_n), None, True, True)
            bit = torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
            dati[f"indietro_{nome_ric}_{mot_n}"] = bit
            ok &= bit
    # (c) strato vero, passo avanti+indietro al bit (ricette B_ind32 e B3_ind32, motore torch)
    ids = torch.randint(0, 150000, (1, 6), generator=torch.Generator().manual_seed(0))
    G = torch.randn(6, 2048, generator=torch.Generator().manual_seed(0))
    for nome_ric in ("B_ind32", "B3_ind32"):
        out = []
        for M in (S, R):
            st = M.Studente(strati=[0], ricetta=nome_ric, motore_nome="torch", device="cpu", esperti=slice(0, 8),
                            log=lambda *_: None)
            stato = st.avanti(ids, x0=st.embedding(ids))
            xL = stato["xL"].detach().requires_grad_(True)
            with torch.enable_grad():
                (st.esce(xL) * G).sum().backward()
            dx0 = st.indietro(stato, xL.grad)
            out.append((stato["xL"], dx0, [p.grad.clone() for p in st.parametri()]))
            del st
        bit = torch.equal(out[0][0], out[1][0]) and torch.equal(out[0][1], out[1][1]) \
            and all(torch.equal(p, q) for p, q in zip(out[0][2], out[1][2]))
        dati[f"strato0_{nome_ric}_bit"] = bit
        ok &= bit
    return registra("H0_default_al_bit", ok, rif=RIF_COMMIT, **dati)


# ─────────────────────────────── H1 ───────────────────────────────
def prova_h1() -> bool:
    g = torch.Generator().manual_seed(1)
    ok, dati = True, {}
    m = torch.tensor([0.0, 1e-40, 2.0 ** -149, 1e-30, 0.3, 0.5, 1.0, 3.0, 1e30, float("inf"), float("nan")])
    s, inv = HL.scala_pot2(m)
    fin = torch.isfinite(m) & (m > 0)
    pot2 = bool(torch.all(torch.frexp(s)[0] == 0.5)) and bool(torch.all(s * inv == 1.0))
    rng = bool(torch.all(((m * s)[fin & (m > 2.0 ** -120)] >= 0.5) & ((m * s)[fin & (m > 2.0 ** -120)] < 1.0)))
    dati["scala_potenza_di_due_esatta"] = pot2
    dati["scala_in_[0.5,1)"] = rng
    ok &= pot2 and rng
    v = (torch.rand(200000, generator=g) * 2 - 1) * torch.pow(10.0, torch.empty(200000).uniform_(-12, 0, generator=g))
    hi, lo = HL.dividi_fp16(v)
    err = float((v.double() - hi.double() - lo.double()).abs().max())
    dati["fp16x2_err_assoluto_max"] = err
    ok &= err <= 2.0 ** -24
    # dY vero-simile: righe da 1e-30 a 1e4
    fam, piano, x, d, gy = caso(0, "down", slice(0, 6), slice(0, 64), 300, 2, 11)
    e2 = HL.errore_rappresentazione(gy, "fp16x2", piano, fam.E)
    e3 = HL.errore_rappresentazione(gy, "bf16x3")
    dati.update({"fp16x2_" + k: v for k, v in e2.items()})
    dati.update({"bf16x3_" + k: v for k, v in e3.items()})
    ok &= e2["righe_max_rel"] <= 2.0 ** -23 and e2["segmenti_max_rel"] <= 2.0 ** -23
    ok &= e3["elementi_max_rel"] <= 2.0 ** -16 and e3["righe_max_rel"] <= 2.0 ** -16
    # la via B (fp16 con UNA scala globale K) sullo stesso gradiente: quanti elementi non nulli vanno a zero o
    # traboccano, per la K migliore fra 2^-40..2^40 (il motivo di B_ind32)
    nz = gy != 0
    peggiore = None
    for eK in range(-40, 41, 4):
        h = (gy * 2.0 ** eK).half()
        persi = int((nz & (h == 0)).sum()) + int((~torch.isfinite(h)).sum())
        peggiore = persi if peggiore is None else min(peggiore, persi)
    dati["B_fp16_K_globale_elementi_persi_min"] = peggiore
    dati["elementi_non_nulli"] = int(nz.sum())
    return registra("H1_divisioni", ok, **dati)


# ─────────────────────────────── H2 ───────────────────────────────
def prova_h2() -> bool:
    ok, dati = True, {}
    mot = S.motore("torch")
    for et, L, n, esp, righe, T in (("gate_L0", 0, "gate", slice(0, 4), slice(0, 96), 200),
                                     ("down_L0", 0, "down", slice(0, 4), slice(0, 64), 200),
                                     ("up_L38", 38, "up", slice(3, 7), slice(0, 96), 200)):
        fam, piano, x, d, gy = caso(L, n, esp, righe, T, 2, 20 + L)
        dx64, dd64 = riferimento_fp64(x, gy, d, fam, piano)
        r = {}
        # fp32 (la via di B_ind32 col motore torch: W fp32, gy fp32)
        dx32 = torch.zeros(piano.S, fam.C)
        o = piano.offs.tolist()
        for e in range(fam.E):
            if o[e + 1] > o[e]:
                dx32[o[e]:o[e + 1]] = gy[o[e]:o[e + 1]] @ HL._W_esperto(fam, e, d)
        dd32 = mot.dd(gy, x.float(), fam, piano, torch.float32)
        # fp16x2 (senza l'arrotondamento finale di dx: prodotti)
        s, inv = HL.scala_pot2(gy.abs().amax(1, keepdim=True))
        hi, lo = HL.dividi_fp16(gy * s)
        dx2 = (HL.prodotto_dx(mot, lo, d, fam, piano) + HL.prodotto_dx(mot, hi, d, fam, piano)) * inv
        eid = HL.esperto_di_riga(piano, piano.S)
        s_er, inv_er = HL.scala_pot2(HL.massimi_segmento(gy.abs(), eid, fam.E))
        hi, lo = HL.dividi_fp16(gy * s_er[eid])
        dd2 = (HL.prodotto_dd(mot, lo, x, fam, piano) + HL.prodotto_dd(mot, hi, x, fam, piano)) * inv_er.unsqueeze(-1)
        # bf16x3
        gh, gl = HL.dividi_bf16(gy)
        dh, dl = HL.dividi_bf16(d)
        dx3 = HL.prodotto_dx(mot, gl, dh, fam, piano) + HL.prodotto_dx(mot, gh, dl, fam, piano) \
            + HL.prodotto_dx(mot, gh, dh, fam, piano)
        xh, xl = HL.dividi_bf16(x.float())
        dd3 = HL.prodotto_dd(mot, gl, xh, fam, piano) + HL.prodotto_dd(mot, gh, xl, fam, piano) \
            + HL.prodotto_dd(mot, gh, xh, fam, piano)
        # fp16 con una K globale (la via di B, scelta la K migliore per dX)
        K = 2.0 ** 20
        dx16 = HL.prodotto_dx(mot, (gy * K).half(), d, fam, piano) / K
        for nome, a, b in (("dx_fp32", dx32, dx64), ("dx_fp16x2", dx2, dx64), ("dx_bf16x3", dx3, dx64),
                           ("dx_B_fp16", dx16, dx64)):
            r[nome] = {"norma": err_norma(a, b), "righe_max": err_righe(a, b)}
        for nome, a, b in (("dd_fp32", dd32, dd64), ("dd_fp16x2", dd2, dd64), ("dd_bf16x3", dd3, dd64)):
            r[nome] = {"norma": err_norma(a, b), "righe_max": err_righe(a.reshape(-1, fam.NG), b.reshape(-1, fam.NG))}
        dati[et] = r
        # fp16x2: al livello di fp32 (per riga: entro 4x fp32 o 2e-6); bf16x3: entro 2e-5 in norma
        ok &= r["dx_fp16x2"]["norma"] <= max(4 * r["dx_fp32"]["norma"], 2e-6)
        ok &= r["dx_fp16x2"]["righe_max"] <= max(4 * r["dx_fp32"]["righe_max"], 2e-6)
        ok &= r["dd_fp16x2"]["norma"] <= max(4 * r["dd_fp32"]["norma"], 2e-6)
        ok &= r["dx_bf16x3"]["norma"] <= 2e-5 and r["dd_bf16x3"]["norma"] <= 2e-5
        ok &= r["dx_bf16x3"]["righe_max"] <= 2e-5
    return registra("H2_prodotti_fp64", ok, **dati)


# ─────────────────────────────── H3 ───────────────────────────────
def prova_h3() -> bool:
    ok, dati = True, {}
    mt = S.motore("torch")
    for mot_n in ("triton2", "triton"):
        mot = S.motore(mot_n)
        for et, L, n, esp, righe, T in (("gate_L0", 0, "gate", slice(0, 3), slice(0, 80), 60),
                                         ("down_L39", 39, "down", slice(0, 3), slice(0, 64), 60)):
            fam, piano, x, d, gy = caso(L, n, esp, righe, T, 2, 30 + L)
            dx64, dd64 = riferimento_fp64(x, gy, d, fam, piano)
            s, inv = HL.scala_pot2(gy.abs().amax(1, keepdim=True))
            hi, lo = HL.dividi_fp16(gy * s)
            dxk = [(HL.prodotto_dx(m, lo, d, fam, piano) + HL.prodotto_dx(m, hi, d, fam, piano)) * inv
                   for m in (mot, mt)]
            eid = HL.esperto_di_riga(piano, piano.S)
            s_er, inv_er = HL.scala_pot2(HL.massimi_segmento(gy.abs(), eid, fam.E))
            hi, lo = HL.dividi_fp16(gy * s_er[eid])
            ddk = [(HL.prodotto_dd(m, lo, x, fam, piano) + HL.prodotto_dd(m, hi, x, fam, piano))
                   * inv_er.unsqueeze(-1) for m in (mot, mt)]
            r = {"dx_kernel_vs_torch": err_norma(dxk[0], dxk[1]), "dd_kernel_vs_torch": err_norma(ddk[0], ddk[1]),
                 "dx_kernel_vs_fp64": err_norma(dxk[0], dx64), "dd_kernel_vs_fp64": err_norma(ddk[0], dd64),
                 "dx_kernel_righe_vs_fp64": err_righe(dxk[0], dx64)}
            dati[f"{mot_n}_{et}"] = r
            ok &= r["dx_kernel_vs_torch"] <= 2e-6 and r["dd_kernel_vs_torch"] <= 2e-6
            ok &= r["dx_kernel_vs_fp64"] <= 2e-6 and r["dd_kernel_vs_fp64"] <= 2e-6
    dati["bf16x3_triton"] = "non provabile nell'interprete (niente bf16): microprova GPU in e2e_cancello_indietro.py"
    return registra("H3_kernel_interprete_fp16x2", ok, **dati)


# ─────────────────────────────── H4 ───────────────────────────────
def prova_h4() -> bool:
    import e2e_allena as A
    ok, dati = True, {}
    for mot_n in ("torch", "triton2"):
        mot = S.motore(mot_n)
        for et, L, n, esp, righe, T in (("gate_L0", 0, "gate", slice(0, 3), slice(0, 80), 60),
                                         ("down_L0", 0, "down", slice(0, 3), slice(0, 64), 60)):
            fam, piano, x, d, gy = caso(L, n, esp, righe, T, 2, 40 + L)
            dx64, dd64 = riferimento_fp64(x, gy, d, fam, piano)
            r32 = S._indietro_famiglia(gy, x, d, fam, piano, S.RICETTE["B_ind32"], mot, None, True, True)
            r = {"dx_fp32_vs_fp64_norma": err_norma(r32[0], dx64), "dd_fp32_vs_fp64": err_norma(r32[1], dd64)}
            for rn in ("B_ind2", "B_ind3"):
                if rn == "B_ind3" and mot_n != "torch":
                    continue                                   # bf16 con Triton: solo GPU
                ric = S.RICETTE[rn]
                a = S._indietro_famiglia(gy, x, d, fam, piano, ric, mot, None, True, True)
                # dx esce arrotondato a fp16 (come B_ind32): l'errore contro fp64 e' quello dell'arrotondamento
                # (~2^-11 per riga) per TUTTE le vie; i singoli elementi con cancellazione possono differire di piu'
                # ulp fp16 anche fra fp32 e fp64, quindi si confronta per RIGA, non per elemento
                diversi = float((a[0] != r32[0]).float().mean())
                r[rn] = {"dx_quota_diversi_da_fp32": diversi, "dx_vs_fp32_righe_max": err_righe(a[0], r32[0]),
                         "dx_vs_fp64_norma": err_norma(a[0], dx64), "dx_vs_fp64_righe_max": err_righe(a[0], dx64),
                         "dd_vs_fp64": err_norma(a[1], dd64), "dd_vs_fp32": err_norma(a[1], r32[1]),
                         "dtype": [str(a[0].dtype), str(a[1].dtype)]}
                ok &= a[0].dtype == r32[0].dtype and a[1].dtype == r32[1].dtype
                ok &= diversi <= 0.05 and r[rn]["dx_vs_fp64_norma"] <= 1.01 * r["dx_fp32_vs_fp64_norma"]
                ok &= r[rn]["dx_vs_fp64_righe_max"] <= 1.01 * err_righe(r32[0], dx64) + 1e-6
                ok &= r[rn]["dd_vs_fp64"] <= (2e-6 if rn == "B_ind2" else 2e-5)
                # la ricetta di riferimento di G1 torna all'indietro fp32 esatto
                r_g1 = S._indietro_famiglia(gy, x, d, fam, piano, A.ricetta_indietro_fp32(ric), mot, None, True, True)
                g1_bit = torch.equal(r_g1[0], r32[0]) and torch.equal(r_g1[1], r32[1])
                r[rn]["ricetta_G1_fp32_al_bit"] = g1_bit
                ok &= g1_bit
            dati[f"{mot_n}_{et}"] = r
    return registra("H4_gancio_famiglia", ok, **dati)


# ─────────────────────────────── H5 ───────────────────────────────
def _passo_strato(ric: str, ids, G, esperti=slice(0, 8)):
    st = S.Studente(strati=[0], ricetta=ric, motore_nome="torch", device="cpu", esperti=esperti, log=lambda *_: None)
    x0 = st.embedding(ids).to(st.ric.dtype_residuo)
    stato = st.avanti(ids, x0=x0)
    xL = stato["xL"].detach().requires_grad_(True)
    with torch.enable_grad():
        (st.esce(xL) * G.to(xL.dtype)).sum().backward()
    dx0 = st.indietro(stato, xL.grad)
    out = (stato["xL"].detach(), dx0.detach(), [p.grad.detach().clone() for p in st.parametri()])
    del st
    return out


def prova_h5() -> bool:
    ids = torch.randint(0, 150000, (1, 8), generator=torch.Generator().manual_seed(5))
    G = torch.randn(8, 2048, generator=torch.Generator().manual_seed(5)) * 1e-3
    ris = {n: _passo_strato(n, ids, G) for n in ("B64", "B_ind32", "B_ind2", "B_ind3", "B3_ind32", "B3_ind2")}
    ref = ris["B64"]

    def dth(n, rif):
        a, b = ris[n][2], rif[2]
        num = sum(float((p.double() - q.double()).pow(2).sum()) for p, q in zip(a, b))
        den = sum(float(q.double().pow(2).sum()) for q in b)
        segni = sum(int(((torch.sign(p) != torch.sign(q)) & (q != 0)).sum()) for p, q in zip(a, b))
        vivi = sum(int((q != 0).sum()) for q in b)
        return {"err_rel_norma": (num / max(den, 1e-300)) ** 0.5, "quota_segni_opposti": segni / max(vivi, 1)}
    dati = {"avanti_B_ind2_eq_B_ind32_bit": torch.equal(ris["B_ind2"][0], ris["B_ind32"][0]),
            "avanti_B3_ind2_eq_B3_ind32_bit": torch.equal(ris["B3_ind2"][0], ris["B3_ind32"][0])}
    for n in ("B_ind32", "B_ind2", "B_ind3"):
        dati[f"dtheta_{n}_vs_B64"] = dth(n, ref)
        dati[f"dx0_{n}_vs_B64"] = err_norma(ris[n][1], ref[1])
    dati["dtheta_B_ind2_vs_B_ind32"] = dth("B_ind2", ris["B_ind32"])
    dati["dtheta_B3_ind2_vs_B3_ind32"] = dth("B3_ind2", ris["B3_ind32"])
    e32 = dati["dtheta_B_ind32_vs_B64"]["err_rel_norma"]
    ok = dati["avanti_B_ind2_eq_B_ind32_bit"] and dati["avanti_B3_ind2_eq_B3_ind32_bit"]
    ok &= dati["dtheta_B_ind2_vs_B64"]["err_rel_norma"] <= max(2 * e32, 1e-5)
    ok &= dati["dtheta_B_ind3_vs_B64"]["err_rel_norma"] <= 1e-3
    ok &= dati["dtheta_B_ind2_vs_B_ind32"]["quota_segni_opposti"] <= 1e-3
    ok &= dati["dtheta_B3_ind2_vs_B3_ind32"]["err_rel_norma"] <= 1e-3
    return registra("H5_strato_studente", ok, **dati)


# ─────────────────────────────── H6 ───────────────────────────────
def prova_h6() -> bool:
    import e2e_lora as LR
    ids = torch.randint(0, 150000, (1, 6), generator=torch.Generator().manual_seed(6))
    G = torch.randn(6, 2048, generator=torch.Generator().manual_seed(6)) * 1e-3
    cfg = LR.ConfigLoRA(rango=4, famiglie=LR.famiglie_lora("tutte"))
    out, dati = {}, {}
    for ric in ("B_ind32", "B3_ind32"):
        st = S.Studente(strati=[0], ricetta=ric, motore_nome="torch", device="cpu", esperti=slice(0, 8),
                        log=lambda *_: None)
        prima = st.avanti(ids, x0=st.embedding(ids))["xL"].detach()
        LR.installa(st, cfg)
        with torch.no_grad():
            for ad in st.lora.values():                        # B != 0 per avere un ramo vivo e gradienti su A
                ad.B.copy_(torch.randn(ad.B.shape, generator=torch.Generator().manual_seed(len(ad.nome))) * 1e-3)
        stato = st.avanti(ids, x0=st.embedding(ids))
        xL = stato["xL"].detach().requires_grad_(True)
        with torch.enable_grad():
            (st.esce(xL) * G).sum().backward()
        st.indietro(stato, xL.grad)
        base_senza_grad = all(not p.requires_grad for p in st.moduli[0].parameters())
        out[ric] = {"prima": prima, "dopo": stato["xL"].detach(), "nomi": sorted(st.lora),
                    "gA": [st.lora[k].A.grad.clone() for k in sorted(st.lora)],
                    "gB": [st.lora[k].B.grad.clone() for k in sorted(st.lora)], "base_senza_grad": base_senza_grad,
                    "tipi_base": sorted({type(m.base).__name__ for m in st.moduli[0].modules()
                                         if isinstance(m, LR.LinearLoRA)})}
        del st
    a, b = out["B3_ind32"], out["B_ind32"]
    dati["tipi_base_B3"] = a["tipi_base"]
    dati["stessi_tensori"] = a["nomi"] == b["nomi"] and len(a["nomi"]) > 0
    dati["n_adattatori"] = len(a["nomi"])
    dati["gA_err_norma_max"] = max(err_norma(p, q) for p, q in zip(a["gA"], b["gA"]))
    dati["gB_err_norma_max"] = max(err_norma(p, q) for p, q in zip(a["gB"], b["gB"]))
    dati["avanti_B3_vs_B_err"] = err_norma(a["dopo"], b["dopo"])
    dati["base_senza_grad"] = a["base_senza_grad"]
    ok = dati["stessi_tensori"] and "LineareBF16x3" in a["tipi_base"] and a["base_senza_grad"]
    ok &= dati["gA_err_norma_max"] <= 1e-3 and dati["gB_err_norma_max"] <= 1e-3 and dati["avanti_B3_vs_B_err"] <= 1e-4
    return registra("H6_lora_su_bf16x3", ok, **dati)


# ─────────────────────────────── H7 ───────────────────────────────
def _carica_pv_rif():
    src = subprocess.run(["git", "-C", str(QUI), "show", f"{RIF_COMMIT}:thor_v2/e2e_pv_codici.py"], capture_output=True,
                         text=True)
    if src.returncode != 0:
        raise RuntimeError(src.stderr[-400:])
    tmp = tempfile.mkdtemp()
    p = Path(tmp) / "e2e_pv_codici_rif.py"
    p.write_text(src.stdout)
    spec = importlib.util.spec_from_file_location("e2e_pv_codici_rif", p)
    M = importlib.util.module_from_spec(spec)
    sys.modules["e2e_pv_codici_rif"] = M
    spec.loader.exec_module(M)
    return M


def prova_h7() -> bool:
    """Passo V (e2e_pv_codici, --pv-codici della corsa): (a) il kernel `_k_dd_v` senza divisione e' AL BIT quello di
    prima nei modi 0-3 (dL/dd e buffer); (b) con B_ind2 e motore triton2+V, modi 1 e 2: dL/dd e statistiche (cand,
    prop, S, Q) dal kernel diviso contro B_ind32 (via fp32) entro l'arrotondamento fp32; (c) modo 3 (curvatura) e
    ricetta bf16x3: la via fp32 di B_ind32, buffer AL BIT."""
    import e2e_pv_codici as PV
    PR = _carica_pv_rif()
    ok, dati = True, {}
    # (a)
    for et, L, n, dt in (("gate_L0_fp32", 0, "gate", torch.float32), ("down_L39_fp16", 39, "down", torch.float16)):
        fam = famiglia(L, n, esperti=slice(0, 3), righe=slice(0, 64))
        g = torch.Generator().manual_seed(70 + L)
        bn, bo = PV.BufferV.crea(fam), PR.BufferV.crea(fam)
        bit = True
        for modo in (1, 2, 2, 3, 0):
            piano = S.Piano(torch.randint(0, fam.E, (40, 2), generator=g), fam.E)
            x = torch.randn(piano.S, fam.C, generator=g).to(dt)
            gy = (torch.randn(piano.S, fam.R, generator=g) * 1e-3).to(dt)
            blk = PV.BLOCCHI_V
            a = PV.dd_v_triton(gy, x, fam, piano, blk, bn if modo else None, modo)
            b = PR.dd_v_triton(gy, x, fam, piano, blk, bo if modo else None, modo)
            bit &= torch.equal(a, b)
        bit &= all(torch.equal(getattr(bn, k), getattr(bo, k)) for k in ("cand", "prop", "S", "Q", "H"))
        dati[f"a_{et}_al_bit"] = bit
        ok &= bit
    # (b) e (c)
    for et, L, n in (("gate_L0", 0, "gate"), ("down_L0", 0, "down")):
        fam = famiglia(L, n, esperti=slice(0, 3), righe=slice(0, 64))
        g = torch.Generator().manual_seed(80 + L)
        st2, st32, st3 = PV.StatoV(), PV.StatoV(), PV.StatoV()
        m2, m32, m3 = (PV.MotoreTritonV(x_, v2=True) for x_ in (st2, st32, st3))
        for x_ in (st2, st32, st3):
            x_.prepara([fam])
        err = {"dd": 0.0}
        for modo in (1, 1, 2, 2, 2, 3):
            piano = S.Piano(torch.randint(0, fam.E, (60, 2), generator=g), fam.E)
            x = (torch.randn(piano.S, fam.C, generator=g) * 0.5).half()
            gy = gy_estremo(piano.S, fam.R, g) * 1e-12            # righe da 1e-42 a 1e-8: la gamma di dY vero
            d = fam.d0.half().float()
            for x_ in (st2, st32, st3):
                x_.modo = modo
            r32 = S._indietro_famiglia(gy, x, d, fam, piano, S.RICETTE["B_ind32"], m32, None, False, True)[1]
            r2 = S._indietro_famiglia(gy, x, d, fam, piano, S.RICETTE["B_ind2"], m2, None, False, True)[1]
            r3 = S._indietro_famiglia(gy, x, d, fam, piano, S.RICETTE["B_ind3"], m3, None, False, True)[1]
            if modo != 3:
                err["dd"] = max(err["dd"], err_norma(r2, r32))
            dati.setdefault(f"c_{et}_bf16x3_dd_al_bit", True)
            dati[f"c_{et}_bf16x3_dd_al_bit"] &= torch.equal(r3, r32)
        b2, b32, b3 = st2.buffer(fam), st32.buffer(fam), st3.buffer(fam)
        stessa = b2.cand == b32.cand
        err.update({"cand_uguali": float(stessa.float().mean()),
                    "prop": err_norma(b2.prop[stessa], b32.prop[stessa]), "S": err_norma(b2.S, b32.S),
                    "Q": err_norma(b2.Q, b32.Q), "H_al_bit": torch.equal(b2.H, b32.H)})
        dati[f"b_{et}"] = err
        c_bit = all(torch.equal(getattr(b3, k), getattr(b32, k)) for k in ("cand", "prop", "S", "Q", "H"))
        dati[f"c_{et}_bf16x3_buffer_al_bit"] = c_bit
        ok &= err["dd"] <= 2e-6 and err["cand_uguali"] >= 0.999 and err["prop"] <= 2e-6 and err["S"] <= 2e-6
        ok &= err["Q"] <= 4e-6 and err["H_al_bit"] and c_bit and dati[f"c_{et}_bf16x3_dd_al_bit"]
    return registra("H7_passo_V", ok, rif=RIF_COMMIT, **dati)


# ─────────────────────────────── H8 ───────────────────────────────
def prova_h8() -> bool:
    """--cambia-ricetta: quali cambi di ricetta alla ripresa sono ammessi (stesso studente, stesso formato)."""
    import e2e_allena as A
    base = {"gguf": "/x.gguf", "byte": 1, "mtime": 2, "ricetta": "B_ind32", "strati": list(range(40)), "esperti": [256]}
    casi = {"B_ind32->B_ind2": ("B_ind2", {}, True), "B_ind32->B3_ind2": ("B3_ind2", {}, True),
            "B_ind32->B_ind3": ("B_ind3", {}, True), "B_ind32->B": ("B", {}, True),
            "B_ind32->A": ("A", {}, False), "B_ind32->C": ("C", {}, False), "B_ind32->B64": ("B64", {}, False),
            "B_ind32->B_ind2_altro_gguf": ("B_ind2", {"byte": 7}, False),
            "B_ind32->B_ind2_altri_strati": ("B_ind2", {"strati": [0]}, False),
            "B_ind32->ignota": ("XYZ", {}, False)}
    dati, ok = {}, True
    for nome, (ric, cambi, atteso) in casi.items():
        ora = {**base, **cambi, "ricetta": ric}
        r = A.ricetta_sostituibile(base, ora)
        dati[nome] = r
        ok &= r == atteso
    ok &= not A.ricetta_sostituibile(None, base)
    return registra("H8_cambia_ricetta", ok, **dati)


# ─────────────────────────────── H9 (1/10, TT5-ottimizzato) ───────────────────────────────
R2B3M = Path(os.getenv("E2E_R2B3M", "@@ROOT:models@@/gguf/tony-tern/R2B3m/Thor-T-R2B3m.gguf"))
_TENS3 = None


def tens3() -> dict:
    global _TENS3
    if _TENS3 is None:
        _TENS3 = {t.name: t for t in GGUFReader(str(R2B3M)).tensors}
    return _TENS3


def _piani(L: int, base: str) -> list:
    """I tre piani VERI di una famiglia densa di R2B3m (primo, *2, *3), 64 righe: (nome, DatiFamiglia)."""
    out = []
    for suf in ("", S.SUFFISSO_P2, S.SUFFISSO_P3):
        n = f"{base}{suf}"
        t = tens3()[f"blk.{L}.{S.FAMIGLIE_DENSE[base]}{suf}.weight"]
        out.append((n, S.famiglia_da_tensore(t, n, righe=slice(0, 64))))
    return out


def _calcola_densi(ric: str, T: int = 48):
    """Strato 0 VERO di R2B3m con --densi-ternari tutte (i densi dello strato a TRE piani), 8 esperti, motore torch:
    theta casuali piccole su tutte le famiglie, x0 casuale, perdita lineare sull'uscita (come e2e_test_terzo_piano)."""
    st = S.Studente(gguf=R2B3M, strati=[0], ricetta=ric, motore_nome="torch", device="cpu", esperti=slice(0, 8),
                    log=lambda *_: None, densi_ternari=tuple(S.FAMIGLIE_DENSE))
    with torch.no_grad():
        for (L, n), p in zip(st.chiavi(), st.parametri()):
            g = torch.Generator().manual_seed(1000 * L + sum(ord(c) for c in n))
            p.copy_(0.01 * torch.randn(p.shape, generator=g))
    g = torch.Generator().manual_seed(110)
    x0 = torch.randn(1, T, 2048, generator=g) * 0.5
    G = torch.randn(T, 2048, generator=g)
    ids = torch.zeros(1, T, dtype=torch.long)
    x0 = x0.to(st.ric.dtype_residuo)                   # B64: residuo fp64
    with torch.no_grad():
        xL = st.avanti(ids, x0=x0)["xL"].clone()
    st.gradiente(ids, lambda h: (h * G.to(h.dtype)).sum() / T, x0=x0)
    lt = [m for m in st.moduli[0].modules() if isinstance(m, S.LineareTernaria)]
    out = {"xL": xL, "grad": {n: p.grad.detach().clone() for (L, n), p in zip(st.chiavi(), st.parametri())},
           "tre_piani": sum(1 for m in lt if m.nome2 is not None and m.nome3 is not None), "lineari": len(lt)}
    del st
    return out


def prova_h9() -> bool:
    """Densi a TRE piani (`LineareTernaria` di R2B3m, la configurazione di TT5) con B3_ind2 contro B3_ind32.
    (a) famiglie dense vere dei tre piani (attn_qkv TQ1_0R_1024 ruotata, ffn_down_shexp TQ1_0; strato 0, 64 righe) su
        un piano a esperto UNICO (`piano_unico`): gancio contro la via fp32 e contro fp64, motori torch e triton2;
    (b) passo V modi 1-2 (MotoreTritonV su triton2, kernel V diviso) sugli stessi tre piani: cand, prop, S, Q contro
        B3_ind32; modo 3 sulla via fp32 AL BIT;
    (c) strato 0 VERO dello studente con --densi-ternari tutte: avanti AL BIT, dtheta per piano (1, 2, 3) e degli
        esperti contro B3_ind32 e contro B64."""
    import e2e_pv_codici as PV
    ok, dati = True, {}
    # (a)
    for base in ("attn_qkv", "ffn_down_shexp"):
        for n, fam in _piani(0, base):
            g = torch.Generator().manual_seed(90 + len(n))
            T = 50
            piano = S.piano_unico(T, "cpu")
            x = (torch.randn(T, fam.C, generator=g) * 0.5).half()
            d = fam.d0.half().float()
            gy = gy_estremo(T, fam.R, g)
            dx64, dd64 = riferimento_fp64(x, gy, d, fam, piano)
            r = {"tipo": fam.tipo, "ruotata": bool(fam.ruotata)}
            for mot_n in ("torch", "triton2"):
                mot = S.motore(mot_n)
                r32 = S._indietro_famiglia(gy, x, d, fam, piano, S.RICETTE["B3_ind32"], mot, None, True, True)
                a = S._indietro_famiglia(gy, x, d, fam, piano, S.RICETTE["B3_ind2"], mot, None, True, True)
                e = {"dd_fp32_vs_fp64": err_norma(r32[1], dd64), "dd_ind2_vs_fp64": err_norma(a[1], dd64),
                     "dd_ind2_vs_fp32": err_norma(a[1], r32[1]),
                     "dx_quota_diversi_da_fp32": float((a[0] != r32[0]).float().mean()),
                     "dx_ind2_vs_fp64_righe_max": err_righe(a[0], dx64),
                     "dx_fp32_vs_fp64_righe_max": err_righe(r32[0], dx64)}
                r[mot_n] = e
                ok &= a[0].dtype == r32[0].dtype and a[1].dtype == r32[1].dtype
                ok &= e["dd_ind2_vs_fp64"] <= 2 * e["dd_fp32_vs_fp64"] + 1e-7 and e["dd_ind2_vs_fp32"] <= 2e-6
                ok &= e["dx_quota_diversi_da_fp32"] <= 0.05
                ok &= e["dx_ind2_vs_fp64_righe_max"] <= 1.01 * e["dx_fp32_vs_fp64_righe_max"] + 1e-6
            dati[f"a_{n}"] = r
    # (b)
    for base in ("attn_qkv", "ffn_down_shexp"):
        piani = _piani(0, base)
        st2, st32 = PV.StatoV(), PV.StatoV()
        m2, m32 = PV.MotoreTritonV(st2, v2=True), PV.MotoreTritonV(st32, v2=True)
        st2.prepara([f for _, f in piani])
        st32.prepara([f for _, f in piani])
        g = torch.Generator().manual_seed(95)
        err = {"dd": 0.0}
        m3_bit = True
        for modo in (1, 1, 2, 2, 2, 3):
            T = 60
            piano = S.piano_unico(T, "cpu")
            x = (torch.randn(T, piani[0][1].C, generator=g) * 0.5).half()
            for x_ in (st2, st32):
                x_.modo = modo
            for _, fam in piani:
                gy = gy_estremo(T, fam.R, g) * 1e-12
                d = fam.d0.half().float()
                r32 = S._indietro_famiglia(gy, x, d, fam, piano, S.RICETTE["B3_ind32"], m32, None, False, True)[1]
                r2 = S._indietro_famiglia(gy, x, d, fam, piano, S.RICETTE["B3_ind2"], m2, None, False, True)[1]
                if modo != 3:
                    err["dd"] = max(err["dd"], err_norma(r2, r32))
                else:
                    m3_bit &= torch.equal(r2, r32)
        for n, fam in piani:
            b2, b32 = st2.buffer(fam), st32.buffer(fam)
            stessa = b2.cand == b32.cand
            e = {"cand_uguali": float(stessa.float().mean()), "prop": err_norma(b2.prop[stessa], b32.prop[stessa]),
                 "S": err_norma(b2.S, b32.S), "Q": err_norma(b2.Q, b32.Q), "H_al_bit": torch.equal(b2.H, b32.H)}
            dati[f"b_{n}"] = e
            ok &= e["cand_uguali"] >= 0.999 and e["prop"] <= 2e-6 and e["S"] <= 2e-6 and e["Q"] <= 4e-6
            ok &= e["H_al_bit"]
        dati[f"b_{base}_dd_max"] = err["dd"]
        dati[f"b_{base}_modo3_dd_al_bit"] = m3_bit
        ok &= err["dd"] <= 2e-6 and m3_bit
    # (c)
    ris = {n: _calcola_densi(n) for n in ("B64", "B3_ind32", "B3_ind2")}
    dati["c_lineari_ternarie"] = ris["B3_ind2"]["lineari"]
    dati["c_lineari_a_tre_piani"] = ris["B3_ind2"]["tre_piani"]
    dati["c_avanti_al_bit"] = torch.equal(ris["B3_ind2"]["xL"], ris["B3_ind32"]["xL"])
    ok &= dati["c_avanti_al_bit"] and dati["c_lineari_a_tre_piani"] == dati["c_lineari_ternarie"] > 0

    def gruppo(n: str) -> str:
        if n in S.FAMIGLIE_DENSE3:
            return "densi_piano3"
        if n in S.FAMIGLIE_DENSE2:
            return "densi_piano2"
        if n in S.FAMIGLIE_DENSE:
            return "densi_piano1"
        return "esperti_e_altro"

    def dth(a: dict, b: dict) -> dict:
        out = {}
        for gr in ("densi_piano1", "densi_piano2", "densi_piano3", "esperti_e_altro"):
            ks = [k for k in b if gruppo(k) == gr]
            num = sum(float((a[k].double() - b[k].double()).pow(2).sum()) for k in ks)
            den = sum(float(b[k].double().pow(2).sum()) for k in ks)
            segni = sum(int(((torch.sign(a[k]) != torch.sign(b[k])) & (b[k] != 0)).sum()) for k in ks)
            vivi = sum(int((b[k] != 0).sum()) for k in ks)
            out[gr] = {"famiglie": len(ks), "err_rel_norma": (num / max(den, 1e-300)) ** 0.5,
                       "quota_segni_opposti": segni / max(vivi, 1)}
        return out
    g64, g32, g2 = ris["B64"]["grad"], ris["B3_ind32"]["grad"], ris["B3_ind2"]["grad"]
    dati["c_ind32_vs_B64"] = dth(g32, g64)
    dati["c_ind2_vs_B64"] = dth(g2, g64)
    dati["c_ind2_vs_ind32"] = dth(g2, g32)
    for gr, v in dati["c_ind2_vs_ind32"].items():
        e32 = dati["c_ind32_vs_B64"][gr]["err_rel_norma"]
        ok &= v["famiglie"] > 0 and v["err_rel_norma"] <= 1e-3 and v["quota_segni_opposti"] <= 1e-3
        ok &= dati["c_ind2_vs_B64"][gr]["err_rel_norma"] <= max(2 * e32, 1e-5)
    return registra("H9_densi_tre_piani", ok, gguf=str(R2B3M), **dati)


PROVE = {"H0": prova_h0, "H1": prova_h1, "H2": prova_h2, "H3": prova_h3, "H4": prova_h4, "H5": prova_h5,
         "H6": prova_h6, "H7": prova_h7, "H8": prova_h8,
         "H9": prova_h9}


def main(argv=None) -> int:
    scelte = (argv if argv is not None else sys.argv[1:]) or list(PROVE)
    ok = True
    for k in scelte:
        t0 = time.time()
        try:
            r = PROVE[k]()
        except Exception as ex:                   # una prova che solleva e' rossa, le altre girano
            import traceback
            traceback.print_exc()
            r = registra(k, False, eccezione=repr(ex)[:500])
        ESITO["prove"][next(n for n in ESITO["prove"] if n.startswith(k))]["secondi"] = round(time.time() - t0, 1)
        ok &= r
    ESITO["ok"] = ok
    dest = Path(os.getenv("E2E_HL_ESITO", str(QUI / "e2e_test_indietro_hilo_esito.json")))
    dest.write_text(json.dumps(ESITO, indent=1, ensure_ascii=False, default=str))
    print(("✅" if ok else "⛔") + f" {dest}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Prove di `e2e_studente.py` (26/9/2026), SOLO CPU: la GPU e' dell'agente «teacher-logit». I kernel Triton
girano nell'interprete di Triton (TRITON_INTERPRET=1) e in piu' si COMPILANO per gfx1151 senza eseguirli
(v_wmma presenti, registri riversati = 0), cosi' un errore che solo il compilatore GPU vedrebbe esce qui.

Uso (limiti del flusso: 8 core, 12G, nice 10):
    systemd-run --user --scope -p CPUQuota=800% -p MemoryMax=12G nice -n 10 env CUDA_VISIBLE_DEVICES= \\
        HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=8 TRITON_INTERPRET=1 @@ROOT:venv@@/bin/python3 e2e_test_studente.py
    esito: $E2E_STUDENTE_DIR/esito_test_studente.json (default: @@ROOT:models@@/gguf/tony-tern/e2e_studente/)

Prove (tutte su byte VERI di Thor-T, `Thor-v2-ThorT-scor150.gguf`):
  S0  non-esperti: `pesi_non_esperti` == `carica_strato_gguf.pesi_strato_da_gguf` al bit (strati 0 GDN e 3 pieno)
  S1  decodifica fusa (kernel Triton) == trit x d dei DUE decoder torch, al bit, anche con scale subnormali
  S2  prodotti e dL/dd dei kernel Triton == motore torch (ingressi fp16 esatti, accumulo fp32)
  S3  gradiente PER FAMIGLIA == riferimento autograd indipendente (float64), anche con scale zero, subnormali
      e quasi zero; differenze finite sulle scale; motore Triton contro lo stesso riferimento
  S4  G1 come scala della perdita K su uno strato vero: indietro fp16 contro fp32 (quota di dL/dd nulli o di
      segno sbagliato), con K dalla regola del progetto e senza K
  S5  checkpoint per strato + top-k salvato == grafo intero su 2 strati veri; contatore dei cambi
  S6  strato dello studente (ricette A e B) == strato HF coi pesi del caricatore verificato
  S7  memoria O(esperto): byte salvati per l'indietro e picco RSS al variare di E
  S8  compilazione anticipata per gfx1151 dei kernel coi Blocchi di default: v_wmma > 0, spill = 0
  S9  scale_fp16: ordine dei blocchi del GGUF (a theta = 0 i campi d del file, al bit)
  S10 modo «segmenti» (cu_seqlens, come il teacher con --modo segmenti): un segmento == modo riga al bit; tre
      segmenti == tre righe separate; gradiente col checkpoint == grafo intero
  S11 (26/9) rotazione compilata (torch.compile) == eager al bit, avanti/indietro/R^T
  S12 (26/9) Osservatore fp16: diagnostica G1 per strato/famiglia, trabocco rilevato e azzerato, `controlla`
  S13 (26/9) combinazione degli esperti fusa (Triton) == via torch: piano di forma Thor e strato vero
  S0 (26/9) chiama il caricatore verificato con la sua mappa privata delle voci `per_esperto` (~6 GiB in meno);
  S6 (26/9) anche la ricetta C (esperti come B, non-esperti bf16, router fp32)
"""
from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("TRITON_INTERPRET", "1")
import numpy as np  # noqa: E402
import torch  # noqa: E402

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import e2e_studente as S  # noqa: E402
import e2e_codici_scale as CS  # noqa: E402
from gguf import GGUFReader  # noqa: E402

DIR = Path(os.getenv("E2E_STUDENTE_DIR", "@@ROOT:models@@/gguf/tony-tern/e2e_studente"))
torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "8")))
ESITO: dict = {"data": time.strftime("%Y-%m-%d %H:%M:%S"), "gguf": str(S.GGUF_THOR_T), "prove": {}}
_TENS = None


def tens() -> dict:
    global _TENS
    if _TENS is None:
        _TENS = {t.name: t for t in GGUFReader(str(S.GGUF_THOR_T)).tensors}
    return _TENS


def registra(nome: str, ok: bool, **dati) -> bool:
    ESITO["prove"][nome] = {"ok": bool(ok), **dati}
    print(f"[{nome}] {'✅' if ok else '⛔'} {json.dumps(dati, ensure_ascii=False, default=str)[:900]}", flush=True)
    return ok


def rel(a, b) -> float:
    a, b = a.double(), b.double()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-300))


def ids_veri(n: int, riga: int = 0) -> torch.Tensor:
    c = torch.load("@@ROOT:models@@/gguf/tony-tern/calibrazione/e2e/e2e_heldout_4096_tutte.pt", weights_only=True)
    return c["domini"]["edu"][riga:riga + 1, :n].clone()


def scale_estreme(d: torch.Tensor, seme: int = 0) -> torch.Tensor:
    """Copia di d con scale forzate: zero, subnormali fp16 (2^-24, 3*2^-24, 1e-6 -> subnormale), quasi zero
    normali (6.2e-5 ~ 2^-14) e il segno negativo; i valori sono GIA' fp16 esatti."""
    g = torch.Generator().manual_seed(seme)
    d = d.clone()
    piatto = d.view(-1)
    scelte = torch.randperm(piatto.numel(), generator=g)[:64]
    speciali = torch.tensor([0.0, 2.0 ** -24, 3 * 2.0 ** -24, 1e-6, 6.2e-5, -2.0 ** -20, 1e-7, 2.0 ** -14],
                            dtype=torch.float32)
    piatto[scelte] = speciali.repeat(8)
    return d.half().float()


# ─────────────────────────────── S0 ───────────────────────────────
def prova_s0() -> bool:
    import carica_strato_gguf as CSG
    cfg = S.config_testo()
    esiti = {}
    ok = True
    # 26/9 10:0x: il caricatore verificato dequantizza anche i 3 tensori di esperti (256 array impilati e poi
    # concatenati: ~6 GiB di picco, con 7 GiB liberi sulla macchina). Si chiama la STESSA funzione con la sua
    # mappa privata delle sole voci `per_esperto` (in memoria, ripristinata subito): i non-esperti passano per
    # lo stesso identico codice, e il confronto resta sulle stesse chiavi.
    mappa_orig = CSG.MAPPA
    for L in (0, 3):
        noi = S.pesi_non_esperti(tens(), L, cfg.layer_types[L] == "full_attention")
        CSG.MAPPA = {k: {f: v for f, v in m.items() if not v.get("per_esperto")} for k, m in mappa_orig.items()}
        try:
            rif = CSG.pesi_strato_da_gguf(str(S.GGUF_THOR_T), L)
        finally:
            CSG.MAPPA = mappa_orig
        chiavi_rif = sorted(k for k in rif if not k.startswith("mlp.experts."))
        uguali = sorted(noi) == chiavi_rif and all(torch.equal(noi[k], rif[k]) for k in chiavi_rif)
        esiti[L] = {"tensori": len(noi), "tipo": cfg.layer_types[L], "identici_al_bit": uguali}
        ok &= uguali
        del rif
    return registra("S0_non_esperti", ok, strati=esiti)


# ─────────────────────────────── S1 ───────────────────────────────
def prova_s1() -> bool:
    ok, righe = True, []
    for L in (0, 3):
        for n in S.FAMIGLIE:
            for es in ((0, 1), (97, 98), (255, 256)):
                fam = S.famiglia_da_tensore(tens()[f"blk.{L}.{S.FAMIGLIE[n]}.weight"], n, esperti=slice(*es))
                for d, etichetta in ((fam.d0, "d0"), (scale_estreme(fam.d0, L), "estreme")):
                    rif1 = torch.stack([(S.trit_esperto(fam, e).float().view(fam.R, fam.NG, fam.gs)
                                         * d[e].unsqueeze(-1)).view(fam.R, fam.C) for e in range(fam.E)])
                    rif2 = torch.stack([(CS.trit_da_blocchi(fam.blocchi[e].reshape(-1, fam.nb), fam.tipo).float()
                                         .view(fam.R, fam.NG, fam.gs) * d[e].unsqueeze(-1)).view(fam.R, fam.C)
                                        for e in range(fam.E)])
                    W32 = S.decodifica_pesi(fam, d, torch.float32)
                    W16 = S.decodifica_pesi(fam, d, torch.float16)
                    u = torch.equal(W32, rif1) and torch.equal(rif1, rif2) and torch.equal(W16, rif1.half()) \
                        and torch.equal(W16.float(), rif1)          # trit x d fp16 e' ESATTO in fp16
                    ok &= u
                    righe.append((L, n, es[0], etichetta, u))
    return registra("S1_decodifica_fusa", ok, casi=len(righe), tutti_al_bit=ok,
                    nota="trit x d == due decoder torch; uscita fp16 == fp32 (prodotto esatto)")


# ─────────────────────────────── S2 ───────────────────────────────
def prova_s2() -> bool:
    torch.manual_seed(2)
    ok, dati = True, {}
    ric = S.RICETTE["B"]
    for n in ("gate", "down"):
        fam = S.famiglia_da_tensore(tens()[f"blk.3.{S.FAMIGLIE[n]}.weight"], n, esperti=slice(10, 16),
                                    righe=slice(0, 128))
        d = scale_estreme(fam.d0, 5)
        idx = torch.randint(0, fam.E, (96, 3))
        idx[:5] = 0                                             # un esperto carico, altri vuoti possibili
        piano = S.Piano(idx, fam.E)
        x = torch.randn(piano.S, fam.C).half()
        gy = torch.randn(piano.S, fam.R).half()
        mt, mr = S.motore("triton"), S.motore("torch")
        y, yr = mt.avanti(x, d, fam, piano, ric), mr.avanti(x, d, fam, piano, ric)
        dx = S._gemm_triton(gy, fam, d, None, piano, True, torch.float32, mt.blk)
        dxr = torch.cat([gy[a:b].float() @ S.peso_esperto_torch(fam, e, d, ric).float()
                         for e, (a, b) in enumerate(zip(piano.offs[:-1].tolist(), piano.offs[1:].tolist()))])
        dd, ddr = mt.dd(gy, x, fam, piano, torch.float32), mr.dd(gy, x, fam, piano, torch.float32)
        mtd = S.motore("triton_denso")
        yd = mtd.avanti(x, d, fam, piano, ric)
        dati[n] = {"avanti": rel(y, yr), "dx": rel(dx, dxr), "dd": rel(dd, ddr), "avanti_denso": rel(yd, yr)}
        ok &= max(dati[n].values()) < 1e-5
    return registra("S2_kernel_contro_torch", ok, errori_relativi=dati, soglia=1e-5)


# ─────────────────────────────── S3 ───────────────────────────────
def riferimento_famiglia(x, d, fam, piano, ruota_peso: bool, G):
    """Riferimento INDIPENDENTE: decoder di e2e_codici_scale (non quello del modulo), peso per esperto con
    autograd, L = sum(y (.) G). Ritorna (L, dL/dx, dL/dd)."""
    x = x.detach().clone().requires_grad_(True)
    d = d.detach().clone().requires_grad_(True)
    o = piano.offs.tolist()
    ys = []
    for e in range(fam.E):
        q = CS.trit_da_blocchi(fam.blocchi[e].reshape(-1, fam.nb), fam.tipo).double()
        W = (q.view(fam.R, fam.NG, fam.gs) * d[e].unsqueeze(-1)).view(fam.R, fam.C)
        if ruota_peso:
            W = S.de_ruota(W)
        ys.append(x[o[e]:o[e + 1]] @ W.T)
    L = (torch.cat(ys) * G).sum()
    L.backward()
    return float(L), x.grad, d.grad


def prova_s3() -> bool:
    torch.manual_seed(3)
    ok, dati = True, {}
    for n in ("gate", "down"):
        fam = S.famiglia_da_tensore(tens()[f"blk.0.{S.FAMIGLIE[n]}.weight"], n, esperti=slice(40, 56))
        d0 = scale_estreme(fam.d0, 7).double()
        idx = torch.randint(0, fam.E, (64, 4))
        piano = S.Piano(idx, fam.E)
        for ricn in ("B64", "A64"):
            ric = S.RICETTE[ricn]
            x = torch.randn(piano.S, fam.C, dtype=torch.float64)
            G = torch.randn(piano.S, fam.R, dtype=torch.float64)
            Lr, gxr, gdr = riferimento_famiglia(x, d0, fam, piano, fam.ruotata and not ric.ruota, G)
            xx = x.clone().requires_grad_(True)
            dd = d0.clone().requires_grad_(True)
            y = S.FamigliaEsperti.apply(xx, dd, fam, piano, ric, S.motore("torch"))
            (y * G).sum().backward()
            e_x, e_d = rel(xx.grad, gxr), rel(dd.grad, gdr)
            zeri = d0 == 0
            # differenze finite su 8 scale (anche subnormali e zero): L e' lineare in d -> esatte a 1e-9
            fd = []
            pos = torch.nonzero(d0.abs() < 1e-4)[:4].tolist() + torch.nonzero(d0.abs() > 1e-3)[:4].tolist()
            for (e, r, g) in pos:
                h = 1e-3
                dp, dm = d0.clone(), d0.clone()
                dp[e, r, g] += h
                dm[e, r, g] -= h
                with torch.no_grad():
                    Lp = float((S.FamigliaEsperti.apply(x, dp, fam, piano, ric, S.motore("torch")) * G).sum())
                    Lm = float((S.FamigliaEsperti.apply(x, dm, fam, piano, ric, S.motore("torch")) * G).sum())
                num = (Lp - Lm) / (2 * h)
                fd.append(abs(num - float(dd.grad[e, r, g])) / max(1e-12, abs(num)))
            dati[f"{n}_{ricn}"] = {"dx": e_x, "dd": e_d, "diff_finite_max": max(fd), "scale_zero": int(zeri.sum()),
                                   "subnormali_fp16": int(((d0.abs() < 2 ** -14) & (d0 != 0)).sum())}
            ok &= e_x < 1e-12 and e_d < 1e-12 and max(fd) < 1e-7
        # motore Triton (interprete): ingressi fp32, uscita contro il riferimento float64
        ric = S.RICETTE["B32"]
        x = torch.randn(piano.S, fam.C, dtype=torch.float64)
        G = torch.randn(piano.S, fam.R, dtype=torch.float64)
        Lr, gxr, gdr = riferimento_famiglia(x, d0, fam, piano, False, G)
        xx = x.float().requires_grad_(True)
        dd = d0.float().requires_grad_(True)
        y = S.FamigliaEsperti.apply(xx, dd, fam, piano, ric, S.motore("triton"))
        (y * G.float()).sum().backward()
        dati[f"{n}_triton_fp32"] = {"dx": rel(xx.grad, gxr), "dd": rel(dd.grad, gdr)}
        ok &= dati[f"{n}_triton_fp32"]["dd"] < 1e-5 and dati[f"{n}_triton_fp32"]["dx"] < 1e-5
    # theta: scala zero -> dtheta = 0 (d_eff = d0 exp(theta) = 0 per costruzione)
    fam = S.famiglia_da_tensore(tens()["blk.0.ffn_up_exps.weight"], "up", esperti=slice(0, 8))
    fam.d0 = scale_estreme(fam.d0, 11)
    st = S.StratoEsperti({"gate": fam, "up": fam, "down": fam}, S.RICETTE["B64"], S.motore("torch"))
    piano = S.Piano(torch.randint(0, 8, (32, 2)), 8)
    xx = torch.randn(piano.S, fam.C, dtype=torch.float64)
    y = S.FamigliaEsperti.apply(xx, st.scala("up"), fam, piano, st.ric, st.mot)
    y.pow(2).sum().backward()
    th = st.theta["up"].grad
    zero_ok = bool((th[fam.d0 == 0] == 0).all())
    ok &= zero_ok
    return registra("S3_gradiente_per_famiglia", ok, errori_relativi=dati, dtheta_nullo_su_scale_zero=zero_ok)


# ─────────────────────────────── S4 ───────────────────────────────
def prova_s4() -> bool:
    """Uno strato vero (3), perdita MEDIA per token. Stesso avanti (ricetta B: attivazioni e pesi fp16), indietro
    con ingressi fp16 (B) contro ingressi fp32 (B_ind32): isola la sola precisione dell'indietro (A2).
    K dalla regola del progetto (§2.5 rev.): eps_G1 = min(1e-8, 1e-3 q01(|g| vivi)), K = 2^round(log2(1e-8/eps))."""
    torch.manual_seed(4)
    ids = ids_veri(96, 1)
    risultati = {}

    def grad_dd(ricn: str, K: float) -> dict:
        st = S.Studente(strati=[3], ricetta=ricn, motore_nome="torch", device="cpu", log=lambda *_: None)
        x0 = st.embedding(ids)
        gen = torch.Generator().manual_seed(9)
        rumore = None

        def perdita(h):
            nonlocal rumore
            if rumore is None:
                rumore = torch.randn(h.shape, generator=gen, dtype=torch.float32)
            bersaglio = (h.detach().float() * (1 + 0.05 * rumore)).to(h.dtype)
            return 0.5 * (h - bersaglio).pow(2).sum(-1).mean() * K
        st.gradiente(ids, perdita, x0=x0)
        return {n: st.esperti(3).theta[n].grad.clone() / K for n in S.FAMIGLIE}

    g32 = grad_dd("B_ind32", 1.0)
    vivi = torch.cat([g.reshape(-1) for g in g32.values()])
    vivi = vivi[vivi != 0].abs()
    q01 = float(torch.quantile(vivi[torch.randperm(vivi.numel())[:1_000_000]].double(), 0.01))
    eps_g1 = min(1e-8, 1e-3 * q01)
    K = 2.0 ** round(np.log2(1e-8 / eps_g1))
    risultati["q01_dtheta"] = q01
    risultati["K"] = K
    ok = True
    # 26/9: "con_K_x256" e' diagnostica (non cancello): se la quota residua non scende oltre K, e'
    # l'arrotondamento fp16 degli ingressi su somme quasi nulle, non il sottoflusso che K cura
    for nome, Kx in (("con_K", K), ("senza_K", 1.0), ("con_K_x256", K * 256)):
        g16 = grad_dd("B", Kx)
        per = {}
        for n in S.FAMIGLIE:
            a, b = g16[n].reshape(-1), g32[n].reshape(-1)
            m = b != 0
            cattivi = ((a[m] == 0) | (torch.sign(a[m]) != torch.sign(b[m]))).float().mean().item()
            per[n] = {"quota_nulli_o_segno_sbagliato": cattivi,
                      "errore_rel_mediano": float(((a[m] - b[m]).abs() / b[m].abs()).median())}
        risultati[nome] = per
        if nome == "con_K":
            ok &= max(v["quota_nulli_o_segno_sbagliato"] for v in per.values()) <= 1e-4
    return registra("S4_G1_scala_K_fp16", ok, **risultati,
                    nota="cancello: con K la quota nulli/segno sbagliato <= 0,01% (progetto R2: 0,00-0,01%)")


# ─────────────────────────────── S5 ───────────────────────────────
def prova_s5() -> bool:
    torch.manual_seed(5)
    ids = ids_veri(48, 2)
    st = S.Studente(strati=[0, 3], ricetta="B64", motore_nome="torch", device="cpu", log=lambda *_: None)
    x0 = st.embedding(ids)
    G = torch.randn(48, 2048, dtype=torch.float64)
    perdita = lambda h: (h * G).sum() + 0.1 * h.pow(2).sum()          # noqa: E731
    t0 = time.time()
    st.gradiente(ids, perdita, x0=x0)
    t_ck = time.time() - t0
    g_ck = [p.grad.clone() for p in st.parametri()]
    cambi = st.cambi()
    for p in st.parametri():
        p.grad = None
    t0 = time.time()
    perdita(st.nascosto(ids, x0=x0)).backward()
    t_gi = time.time() - t0
    g_gi = [p.grad.clone() for p in st.parametri()]
    errori = {f"{L}.{n}": rel(a, b) for (L, n), a, b in zip(st.chiavi(), g_ck, g_gi)}
    ok = max(errori.values()) < 1e-10 and cambi == 0
    # il contatore funziona: ingresso salvato perturbato -> top-k vivo diverso da quello salvato
    stato = st.avanti(ids, x0=x0)
    st.azzera_cambi()
    xin = stato["ingressi"][1] + 0.5 * torch.randn_like(stato["ingressi"][1])
    with torch.no_grad():
        y_forzato = st._strato(3, xin, stato["ctx"], instr=stato["instr"][1])
    cambi_pert = st.cambi()
    # con l'instradamento forzato l'uscita e' quella del top-k SALVATO: rifatta a mano togliendo il vivo
    ok &= cambi_pert > 0
    return registra("S5_checkpoint_topk_salvato", ok, errori_relativi=errori, cambi_instradamento=cambi,
                    cambi_con_ingresso_perturbato=cambi_pert, secondi_checkpoint=round(t_ck, 1),
                    secondi_grafo_intero=round(t_gi, 1), uscita_forzata_finita=bool(torch.isfinite(y_forzato).all()))


# ─────────────────────────────── S6 ───────────────────────────────
def strato_hf(L: int, dtype):
    """Riferimento: Qwen3_5MoeDecoderLayer HF (esperti eager) coi pesi del caricatore verificato. Non-esperti:
    `pesi_non_esperti` (== `pesi_strato_da_gguf` al bit, prova S0). Esperti: la STESSA via per esperto di
    `pesi_strato_da_gguf` (`dequant_np_slice` + `_applica`, gate nelle righe [:512] e up in [512:] del
    gate_up_proj fuso, come dice la mappa), scritta esperto per esperto in tensori gia' allocati: il
    caricatore impila 256 array e poi li concatena (~6 GiB di picco, troppi con la RAM di oggi)."""
    import carica_strato_gguf as CSG
    from dequant_gguf_torch import dequant_np_slice
    from accelerate import init_empty_weights
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer
    cfg = S.config_testo()
    cfg._experts_implementation = "eager"
    with init_empty_weights():
        strato = Qwen3_5MoeDecoderLayer(cfg, layer_idx=L)
    tipo = "full" if cfg.layer_types[L] == "full_attention" else "linear"
    sd = S.pesi_non_esperti(tens(), L, tipo == "full")
    E, I, H = cfg.num_experts, cfg.moe_intermediate_size, cfg.hidden_size
    gu = torch.empty(E, 2 * I, H, dtype=dtype)
    dn = torch.empty(E, H, I, dtype=dtype)
    mappa = CSG.MAPPA[tipo]
    for e in range(E):
        for fam, dove in (("ffn_gate_exps.weight", slice(0, I)), ("ffn_up_exps.weight", slice(I, 2 * I))):
            assert mappa[fam]["fetta"] == f"righe[{dove.start or ''}:{dove.stop if dove.start == 0 else ''}]", \
                mappa[fam]["fetta"]
            gu[e, dove] = torch.from_numpy(CSG._applica(mappa[fam], dequant_np_slice(tens()[f"blk.{L}.{fam}"], e)))
        dn[e] = torch.from_numpy(CSG._applica(mappa["ffn_down_exps.weight"],
                                              dequant_np_slice(tens()[f"blk.{L}.ffn_down_exps.weight"], e)))
    sd["mlp.experts.gate_up_proj"] = gu
    sd["mlp.experts.down_proj"] = dn
    manca, extra = strato.load_state_dict(sd, strict=False, assign=True)
    assert not manca and not extra, (manca, extra)
    return strato.to(dtype).eval()


def prova_s6() -> bool:
    torch.manual_seed(6)
    ids = ids_veri(64, 3)
    ok, dati = True, {}
    # C (26/9): esperti come B, non-esperti/residuo bf16 come A, router fp32 -> stessa soglia bf16 di A
    soglie = {"A32": 1e-5, "B32": 1e-5, "B": 2e-3, "A": 3e-2, "C": 3e-2}
    for L in (0, 3):
        ref = strato_hf(L, torch.float32)
        for ricn in ("A32", "B32", "B", "A", "C"):
            st = S.Studente(strati=[L], ricetta=ricn, motore_nome="torch", device="cpu", log=lambda *_: None)
            x0 = st.embedding(ids)
            with torch.no_grad():
                ctx = st._contesto(1, ids.shape[1], x0.dtype)
                y = st._strato(L, x0, ctx)
                if ricn in ("A", "C"):
                    ref = ref.to(torch.bfloat16)        # stessi pesi del caricatore verificato, come la forgia
                yr = ref(x0.to(next(ref.parameters()).dtype), position_embeddings=ctx["pe"],
                         attention_mask=ctx["maschera"] if st.cfg.layer_types[L] == "full_attention" else None,
                         position_ids=ctx["testo"])
                yr = yr[0] if isinstance(yr, tuple) else yr
            # errore relativo sulla sola parte di strato (uscita - ingresso) e sull'uscita
            e_delta = rel((y - x0).double(), (yr - x0.to(yr.dtype)).double())
            dati[f"{L}_{ricn}"] = {"delta_strato": e_delta, "uscita": rel(y.double(), yr.double())}
            ok &= e_delta < soglie[ricn]
            del st
        del ref
    return registra("S6_strato_contro_HF", ok, errori_relativi=dati, soglie=soglie,
                    nota="A = forgia bf16 (soglia bf16); B = GEMM fp16 degli esperti contro HF fp32")


# ─────────────────────────────── S7 ───────────────────────────────
_S7_FIGLIO = r"""
import os, sys, resource, json, torch
sys.path.insert(0, %r)
import e2e_studente as S
from gguf import GGUFReader
E, schema = int(sys.argv[1]), sys.argv[2]
t = {x.name: x for x in GGUFReader(str(S.GGUF_THOR_T)).tensors}["blk.0.ffn_gate_exps.weight"]
fam = S.famiglia_da_tensore(t, "gate", esperti=slice(0, E))
torch.manual_seed(0)
idx = torch.randint(0, E, (512, 8))
piano = S.Piano(idx, E)
x = torch.randn(piano.S, fam.C).requires_grad_(True)
d = fam.d0.clone().requires_grad_(True)
salvati = [0]
def pack(t):
    salvati[0] += t.numel() * t.element_size()
    return t
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
    if schema == "famiglia":
        y = S.FamigliaEsperti.apply(x, d, fam, piano, S.RICETTE["B32"], S.motore("torch"))
    else:   # riferimento con autograd sul peso di ogni esperto (schema del prototipo corretto: peso_esperto)
        o = piano.offs.tolist(); ys = []
        for e in range(E):
            q = S.trit_esperto(fam, e).float()
            W = (q.view(fam.R, fam.NG, fam.gs) * d[e].unsqueeze(-1)).view(fam.R, fam.C)
            ys.append(x[o[e]:o[e + 1]] @ W.T)
        y = torch.cat(ys)
y.sum().backward()
picco = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(json.dumps({"E": E, "schema": schema, "byte_salvati": salvati[0], "picco_rss_delta_MiB": (picco - base) / 1024}))
""" % str(QUI)


def prova_s7() -> bool:
    righe = []
    for schema in ("famiglia", "peso_per_esperto"):
        for E in (32, 64, 128, 256):
            r = subprocess.run([sys.executable, "-c", _S7_FIGLIO, str(E), schema], capture_output=True, text=True,
                               env={**os.environ, "TRITON_INTERPRET": "0"})
            if r.returncode != 0:
                return registra("S7_memoria", False, errore=r.stderr[-800:])
            righe.append(json.loads(r.stdout.strip().splitlines()[-1]))
    fam = {r["E"]: r for r in righe if r["schema"] == "famiglia"}
    rif = {r["E"]: r for r in righe if r["schema"] != "famiglia"}
    # salvati dalla funzione per famiglia: x (S x C fp32) + d (E x R x NG fp32) -> dipendono da E solo per d
    d_per_esperto = 512 * 16 * 4
    atteso = {E: 512 * 8 * 2048 * 4 + E * d_per_esperto for E in fam}
    ok = all(fam[E]["byte_salvati"] == atteso[E] for E in fam)
    pend_fam = (fam[256]["byte_salvati"] - fam[32]["byte_salvati"]) / 224
    pend_rif = (rif[256]["byte_salvati"] - rif[32]["byte_salvati"]) / 224
    return registra("S7_memoria_O_esperto", ok, righe=righe,
                    byte_salvati_per_esperto_in_piu={"famiglia": pend_fam, "peso_per_esperto": pend_rif},
                    nota="famiglia: salva x + d (32 KiB di scale per esperto); il peso di un esperto (R x C) "
                         "e' un transitorio del solo esperto in corso, mai salvato")


# ─────────────────────────────── S8 ───────────────────────────────
_S8_FIGLIO = r"""
import sys, re, json
sys.path.insert(0, %r)
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
import e2e_studente as S
b = S.Blocchi()
tgt = GPUTarget("hip", "gfx1151", 32)
def info(fn, sig, cex):
    k = triton.compile(ASTSource(fn=fn, signature=sig, constexprs=cex), target=tgt, options={"num_warps": b.num_warps})
    a = k.asm["amdgcn"]
    return {"v_wmma": len(re.findall(r"v_wmma", a)), "spill": int(re.findall(r"vgpr_spill_count:\s+(\d+)", a)[0])}
g = {"a_ptr": "*fp16", "sa_s": "i32", "sa_k": "i32", "pb": "*u8", "sb_e": "i32", "NBR": "i32", "d_ptr": "*fp32",
     "sd_e": "i32", "NG": "i32", "w_ptr": "*fp16", "sw_e": "i32", "sw_r": "i32", "o_ptr": "*fp32", "so_s": "i32",
     "so_n": "i32", "te_ptr": "*i32", "ts_ptr": "*i32", "offs_ptr": "*i64", "R": "i32", "C": "i32", "K": "i32",
     "N": "i32", "GS": "constexpr", "QK": "constexpr", "NB": "constexpr", "ROT": "constexpr", "TRASP": "constexpr",
     "DENSO": "constexpr", "BS": "constexpr", "BN": "constexpr", "BK": "constexpr"}
gd = {"dy_ptr": "*fp16", "sdy_s": "i32", "sdy_r": "i32", "x_ptr": "*fp16", "sx_s": "i32", "sx_c": "i32", "pb": "*u8",
      "sb_e": "i32", "NBR": "i32", "dd_ptr": "*fp32", "sdd_e": "i32", "NG": "i32", "offs_ptr": "*i64", "R": "i32",
      "C": "i32", "GS": "constexpr", "QK": "constexpr", "NB": "constexpr", "ROT": "constexpr", "BR": "constexpr",
      "BS": "constexpr"}
out = {}
for nome, rot, gs, qk, nb in (("R1024", 1, 128, 1024, 224), ("TQ1_0", 0, 256, 256, 54)):
    for tr in (0, 1):
        for den in (0, 1):
            out[f"gemm_{nome}_trasp{tr}_denso{den}"] = info(S._k_gemm, g, {"GS": gs, "QK": qk, "NB": nb, "ROT": rot,
                "TRASP": tr, "DENSO": den, "BS": b.BS, "BN": b.BN, "BK": b.BK})
    out[f"dd_{nome}"] = info(S._k_dd, gd, {"GS": gs, "QK": qk, "NB": nb, "ROT": rot, "BR": b.BR_dd, "BS": b.BS_dd})
# 26/9: kernel della combinazione fusa (niente prodotti matriciali: si controlla solo spill = 0)
cex = {"K": 8, "BH": S.BH_COMBINA}
out["combina"] = info(S._k_combina, {"o_ptr": "*fp32", "so_s": "i32", "w_ptr": "*fp32", "inv_ptr": "*i64",
                      "y_ptr": "*fp32", "sy_t": "i32", "H": "i32", "K": "constexpr", "BH": "constexpr"}, cex)
out["combina_do"] = info(S._k_combina_do, {"g_ptr": "*fp32", "sg_t": "i32", "w_ptr": "*fp32", "ord_ptr": "*i64",
                         "do_ptr": "*fp32", "sdo_s": "i32", "H": "i32", "K": "constexpr", "BH": "constexpr"}, cex)
out["combina_dw"] = info(S._k_combina_dw, {"g_ptr": "*fp32", "sg_t": "i32", "o_ptr": "*fp32", "so_s": "i32",
                         "inv_ptr": "*i64", "dw_ptr": "*fp32", "H": "i32", "K": "constexpr", "BH": "constexpr"}, cex)
print(json.dumps(out))
""" % str(QUI)


def prova_s8() -> bool:
    r = subprocess.run([sys.executable, "-c", _S8_FIGLIO], capture_output=True, text=True,
                       env={**os.environ, "TRITON_INTERPRET": "0"})
    if r.returncode != 0:
        return registra("S8_compilazione_gfx1151", False, errore=r.stderr[-800:])
    out = json.loads(r.stdout.strip().splitlines()[-1])
    ok = all((v["v_wmma"] > 0 or k.startswith("combina")) and v["spill"] == 0 for k, v in out.items())
    return registra("S8_compilazione_gfx1151", ok, kernel=out, blocchi=S.Blocchi().__dict__)


# ─────────────────────────────── S9 ───────────────────────────────
def prova_s9() -> bool:
    st = S.Studente(strati=[3], ricetta="B", motore_nome="torch", device="cpu", log=lambda *_: None)
    ok = True
    for n in S.FAMIGLIE:
        t = tens()[f"blk.3.{S.FAMIGLIE[n]}.weight"]
        p = CS.posizione_scale(t)
        blocchi = np.asarray(t.data).reshape(-1, p["byte_blocco"])
        d_file = blocchi[:, p["off_d"]:p["off_d"] + 2 * p["k"]].copy().view(np.float16).reshape(-1)
        noi = st.scale_fp16()[(3, n)].numpy().reshape(-1)
        ok &= np.array_equal(noi.view(np.uint16), d_file.view(np.uint16))
    # theta qualunque: la scala del forward == numpy astype(float16) di d0*exp(theta) (fp32), al bit
    with torch.no_grad():
        th = st.esperti(3).theta["gate"]
        th.copy_(torch.randn_like(th) * 0.1)
        atteso = (st.esperti(3).fam["gate"].d0 * torch.exp(th)).numpy().astype(np.float16)
        noi = st.scale_fp16()[(3, "gate")].numpy()
    ok &= np.array_equal(noi.view(np.uint16), atteso.view(np.uint16))
    return registra("S9_ordine_scale_export", ok, nota="theta=0 -> campi d del GGUF al bit; theta!=0 -> RNE numpy")

# ─────────────────────────────── S10 ───────────────────────────────
def prova_s10() -> bool:
    """Modo «segmenti» (come `e2e_teacher --modo segmenti`): (a) un solo segmento per riga == modo riga AL BIT
    (la replica del forward dello strato e' la stessa sequenza di HF); (b) una riga con 3 segmenti == i 3
    segmenti passati come righe separate (float64); (c) i confini agiscono (senza, l'uscita cambia);
    (d) gradiente con checkpoint in modo segmenti == grafo intero."""
    torch.manual_seed(10)
    st = S.Studente(strati=[0, 3], ricetta="B64", motore_nome="torch", device="cpu", log=lambda *_: None)
    ids = ids_veri(60, 4)
    x0 = st.embedding(ids)
    uno = torch.zeros(1, 60, dtype=torch.int32)
    uno[0, 0] = 1
    with torch.no_grad():
        a_riga = st.avanti(ids, x0=x0)["xL"]
        a_seg1 = st.avanti(ids, x0=x0, inizi=uno)["xL"]
    ok_a = torch.equal(a_riga, a_seg1)
    tre = torch.zeros(1, 60, dtype=torch.int32)
    tre[0, [0, 17, 41]] = 1
    with torch.no_grad():
        y3 = st.avanti(ids, x0=x0, inizi=tre)["xL"]
        pezzi = [st.avanti(ids[:, a:b], x0=x0[:, a:b])["xL"] for a, b in ((0, 17), (17, 41), (41, 60))]
    e_b = rel(y3, torch.cat(pezzi, dim=1))
    diff_c = rel(y3, a_riga)
    G = torch.randn(60, 2048, dtype=torch.float64)
    perdita = lambda h: (h * G).sum()          # noqa: E731
    st.gradiente(ids, perdita, x0=x0, inizi=tre)
    g_ck = [p.grad.clone() for p in st.parametri()]
    for p in st.parametri():
        p.grad = None
    perdita(st.nascosto(ids, x0=x0, inizi=tre)).backward()
    e_d = max(rel(a, p.grad) for a, p in zip(g_ck, st.parametri()))
    ok = ok_a and e_b < 1e-10 and diff_c > 1e-3 and e_d < 1e-10 and st.cambi() == 0
    return registra("S10_modo_segmenti", ok, un_segmento_uguale_a_riga_al_bit=ok_a, tre_segmenti_contro_separati=e_b,
                    effetto_dei_confini=diff_c, gradiente_checkpoint_contro_grafo=e_d, cambi=st.cambi())


# ─────────────────────────────── S12 ───────────────────────────────
def prova_s12() -> bool:
    """26/9. Osservatore (rischio 8 e cancello G1) su uno strato vero, ricetta B: (a) diagnostica per strato e
    famiglia; con K dalla regola la quota di gradienti che in fp16 vanno a zero e' minore che con K = 1;
    (b) K = 2^40 -> trabocco acceso e `controlla` solleva; (c) la lettura azzera il trabocco (passo dopo pulito);
    (d) senza trabocco `controlla` non solleva e riporta i cambi."""
    ids = ids_veri(64, 5)
    st = S.Studente(strati=[3], ricetta="B", motore_nome="torch", device="cpu", log=lambda *_: None)
    x0 = st.embedding(ids)
    rumore = torch.randn(64, 2048, generator=torch.Generator().manual_seed(12))

    def passo(K: float):
        def perdita(h):
            bersaglio = (h.detach().float() * (1 + 0.05 * rumore)).to(h.dtype)
            return 0.5 * (h - bersaglio).pow(2).sum(-1).mean() * K
        st.gradiente(ids, perdita, x0=x0)
        for p in st.parametri():
            p.grad = None

    st.diagnostica_indietro(True)
    passo(1.0)
    r1 = st.diagnostica_indietro(True)
    c1 = st.controlla()
    passo(256.0)
    rk = st.diagnostica_indietro(False)
    ck = st.controlla()
    passo(2.0 ** 40)
    try:
        st.controlla()
        sollevato = False
    except FloatingPointError:
        sollevato = True
    dopo = st.controlla()
    q = lambda righe: sum(r["a_zero_in_fp16"] for r in righe) / max(1, sum(r["non_nulli"] for r in righe))  # noqa
    famiglie = sorted((r["strato"], r["famiglia"]) for r in r1)
    ok = (famiglie == [(3, "down"), (3, "gate"), (3, "up")] and q(rk) < q(r1) and not c1["trabocco_fp16"]
          and not ck["trabocco_fp16"] and sollevato and not dopo["trabocco_fp16"])
    return registra("S12_osservatore_fp16", ok, quota_a_zero_K1=q(r1), quota_a_zero_K256=q(rk),
                    righe_K1=r1, righe_K256=rk, solleva_con_K_2alla40=sollevato, azzerato_dopo_lettura=not
                    dopo["trabocco_fp16"])


# ─────────────────────────────── S13 ───────────────────────────────
def prova_s13() -> bool:
    """26/9. Combinazione fusa (`CombinaEsperti`, kernel Triton nell'interprete) == via torch (o * wo +
    `_Disperdi`): (a) avanti, dL/do e dL/dw su un piano di forma Thor (k = 8 esperti distinti per token,
    H = 2048, esperti vuoti e carichi diversi); (b) uno strato vero (3, 16 esperti) col motore Triton e ricetta
    B32 (ingressi fp32: le differenze sono solo l'ordine delle somme fp32): uscita e gradienti di theta."""
    torch.manual_seed(13)
    T, k, E, H = 37, 8, 32, 2048
    idx = torch.stack([torch.randperm(E)[:k] for _ in range(T)])
    piano = S.Piano(idx, E)
    o = torch.randn(piano.S, H)
    w = torch.rand(T, k)
    w = w / w.sum(-1, keepdim=True)
    G = torch.randn(T, H)
    o1, w1 = o.clone().requires_grad_(True), w.clone().requires_grad_(True)
    y1 = S._Disperdi.apply(o1 * w1.reshape(-1)[piano.ordine].unsqueeze(-1), piano.ordine, piano.inversa, k)
    (y1 * G).sum().backward()
    o2, w2 = o.clone().requires_grad_(True), w.clone().requires_grad_(True)
    y2 = S.CombinaEsperti.apply(o2, w2, piano.ordine, piano.inversa, k)
    (y2 * G).sum().backward()
    a = {"y": rel(y2, y1), "do": rel(o2.grad, o1.grad), "dw": rel(w2.grad, w1.grad)}
    ok = max(a.values()) < 1e-6
    vecchio = S.COMBINA_FUSA
    ids = ids_veri(16, 6)
    Gs = torch.randn(16, 2048, generator=torch.Generator().manual_seed(1))
    ris = {}
    # E2E_S13_PIENO=1: anche i prodotti col motore Triton nell'interprete (~25 min a un thread, fatto il 26/9
    # 10:3x); default: prodotti col motore torch (== Triton: S2/S3) dentro un motore che si presenta come
    # Triton, cosi' la combinazione fusa si attiva e la prova isola solo lei.
    pieno = os.getenv("E2E_S13_PIENO") == "1"

    class _TorchComeTriton(S.MotoreTriton):
        def __init__(self):
            super().__init__()
            self._t = S.MotoreTorch()
            self.nome = "torch_come_triton"

        def pesi(self, fam, d, ric, pezzo=32):
            return None

        def avanti(self, x, d, fam, piano, ric, pesi=None):
            return self._t.avanti(x, d, fam, piano, ric)

        def indietro_x(self, gy, d, fam, piano, ric, pesi=None):
            return self._t.indietro_x(gy, d, fam, piano, ric)

        def dd(self, gy, x, fam, piano, d_dtype):
            return self._t.dd(gy, x, fam, piano, d_dtype)

    chiamate = {False: 0, True: 0}
    orig_apply = S.CombinaEsperti.apply
    stato = {"fusa": False}

    def conta(*args):
        chiamate[stato["fusa"]] += 1
        return orig_apply(*args)
    S.CombinaEsperti.apply = conta
    try:
        for fusa in (False, True):
            S.COMBINA_FUSA = fusa
            stato["fusa"] = fusa
            st = S.Studente(strati=[3], ricetta="B32", motore_nome="triton" if pieno else "torch", device="cpu",
                            esperti=slice(0, 16), log=lambda *_: None)
            if not pieno:
                st.mot = _TorchComeTriton()
                st.esperti(3).mot = st.mot
            x0 = st.embedding(ids)
            y = st.avanti(ids, x0=x0)["xL"]
            st.gradiente(ids, lambda h: (h * Gs).sum(), x0=x0)
            ris[fusa] = (y, [p.grad.clone() for p in st.parametri()], st.cambi())
            del st
    finally:
        S.COMBINA_FUSA = vecchio
        S.CombinaEsperti.apply = orig_apply
    b = {"chiamate_fusa": chiamate[True], "chiamate_fusa_con_via_torch": chiamate[False],
         "uscita": rel(ris[True][0], ris[False][0]),
         "dtheta_max": max(rel(g1, g0) for g1, g0 in zip(ris[True][1], ris[False][1])),
         "cambi": ris[True][2] + ris[False][2]}
    # avanti + (avanti senza grafo + ricalcolo) del gradiente = 3 chiamate con la fusa, 0 con la via torch
    ok &= b["uscita"] < 1e-5 and b["dtheta_max"] < 1e-4 and b["cambi"] == 0 and b["chiamate_fusa"] == 3 \
        and b["chiamate_fusa_con_via_torch"] == 0
    return registra("S13_combinazione_fusa", ok, piano_forma_thor=a, strato_vero_B32=b,
                    motore_prodotti="triton (interprete)" if pieno else "torch dentro un motore Triton",
                    soglie={"piano": 1e-6, "uscita": 1e-5, "dtheta": 1e-4})


# ─────────────────────────────── S11 ───────────────────────────────
def prova_s11() -> bool:
    """26/9 10:1x. Rotazione compilata (torch.compile, inductor su CPU) == eager AL BIT: R x avanti e
    indietro, R^T w (anche 3D, come i pesi (E, R, C) della ricetta A), fp32 e fp64, due forme diverse
    (dynamic=True). `misura` ripete la prova sulla GPU e spegne la compilata se non e' al bit."""
    torch.manual_seed(11)
    vecchio = S.COMPILA_RUOTA
    casi, ok = {}, True
    try:
        for dt in (torch.float32, torch.float64):
            for forma in ((7, 2048), (300, 2048), (3, 5, 2048)):
                x = torch.randn(*forma, dtype=dt)
                gy = torch.randn(*forma, dtype=dt)
                ris = {}
                for flag in (False, True):
                    S.COMPILA_RUOTA = flag
                    xx = x.clone().requires_grad_(True)
                    y = S.ruota(xx)
                    y.backward(gy)
                    ris[flag] = (y.detach(), xx.grad, S.de_ruota(x))
                u = all(torch.equal(a, b) for a, b in zip(ris[False], ris[True]))
                casi[f"{str(dt).split('.')[-1]}_{'x'.join(map(str, forma))}"] = u
                ok &= u
    finally:
        S.COMPILA_RUOTA = vecchio
    return registra("S11_rotazione_compilata", ok, casi=casi,
                    nota="avanti, indietro e R^T identici al bit fra eager e torch.compile (inductor CPU)")


def main() -> int:
    DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    solo = [s for s in os.getenv("E2E_PROVE", "").split(",") if s]
    ok = True
    for nome, f in (("S0", prova_s0), ("S1", prova_s1), ("S2", prova_s2), ("S3", prova_s3), ("S9", prova_s9),
                    ("S8", prova_s8), ("S7", prova_s7), ("S5", prova_s5), ("S10", prova_s10), ("S11", prova_s11),
                    ("S12", prova_s12), ("S13", prova_s13), ("S6", prova_s6), ("S4", prova_s4)):
        if solo and nome not in solo:
            continue
        t1 = time.time()
        try:
            ok &= f()
        except Exception as ex:  # una prova che si rompe e' una prova fallita, non un silenzio
            import traceback
            traceback.print_exc()
            ok &= registra(f"{nome}_ECCEZIONE", False, errore=repr(ex))
        print(f"   ({nome}: {time.time() - t1:.1f} s)", flush=True)
    ESITO["secondi"] = round(time.time() - t0, 1)
    ESITO["tutto_verde"] = bool(ok)
    ESITO["picco_rss_GiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
    nome_file = "esito_test_studente.json" if not solo else f"esito_test_studente_{'_'.join(solo)}.json"
    (DIR / nome_file).write_text(json.dumps(ESITO, indent=1, default=str))
    print("ESITO:", "TUTTO VERDE" if ok else "⛔ QUALCOSA NON TORNA", f"({ESITO['secondi']} s)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

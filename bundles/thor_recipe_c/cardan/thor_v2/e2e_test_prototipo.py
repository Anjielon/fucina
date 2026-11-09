#!/usr/bin/env python3
"""Prove SOLO CPU del prototipo E2E sulle scale (`e2e_prototipo.py`, progetto `e2e_progetto_scelto.md`).

Uso (la GPU e' della forgia: mai toccarla):
    CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 19 python3 e2e_test_prototipo.py

Le quattro richieste, e dove si provano:
  (a) i codici non cambiano MAI, al bit ............................ T8 (ogni passo e alla fine), T10
  (b) le scale restano rappresentabili in fp16 ....................... T3, T8
  (c) la perdita scende su dati tenuti fuori ......................... T8
  (d) export in un GGUF coi blocchi uguali al bit salvo i campi d ..... T10
e i mattoni da cui dipendono, ognuno contro un riferimento che NON e' il prototipo:
  T0 dati disgiunti · T1 byte <-> trit/scale (gguf-py, codec di casa, secondo decodificatore) ·
  T2 ricetta B == ricetta A (float64) e studente == modello dai pesi dequantizzati dalle librerie ·
  T4 KL a pezzi == autograd (gradcheck float64) · T5 cache del maestro == maestro diretto al bit ·
  T6 checkpoint per strato == grafo intero · T7 cancello G1 (eps, primo passo di Adam) ·
  T9 ripresa collaudata (theta e stato di Adam identici al bit).
Esito completo in $E2E_PROTO_DIR/esito_test.json.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import e2e_prototipo as P  # noqa: E402
import e2e_campi_scala as C1  # noqa: E402  (secondo decodificatore, indipendente)
import e2e_codici_scale as CS  # noqa: E402
from gguf.quants import TQ1_0 as GQ_TQ1_0  # noqa: E402
from tq1_0r_1024_codec import dequantize_blocks as deq_r1024, fwht1024_f32, S1024  # noqa: E402
from rotazione_gpu import _RT_1024_torch  # noqa: E402

torch.use_deterministic_algorithms(True)
ESITO: dict = {}
FALLITE: list = []


def verifica(nome: str, ok: bool, **dati) -> None:
    ESITO[nome] = {"ok": bool(ok), **dati}
    print(f"{'✅' if ok else '⛔'} {nome}: " + ", ".join(f"{k}={v}" for k, v in dati.items()), flush=True)
    if not ok:
        FALLITE.append(nome)


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.detach(), b.detach()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-300))


# ───────────────────────────────────────── T0 ─────────────────────────────────────────
def t0_dati(D: dict) -> None:
    man = D["man"]
    nomi = [set(v["file"] for v in man[p]) for p in ("maestro", "calibrazione", "heldout")]
    disgiunti = not (nomi[0] & nomi[1] or nomi[0] & nomi[2] or nomi[1] & nomi[2])
    verifica("T0 file disgiunti fra maestro, calibrazione e held-out", disgiunti,
             file=[len(n) for n in nomi], seq_calibrazione=tuple(D["cal"].shape), seq_heldout=tuple(D["held"].shape))


# ───────────────────────────────────────── T1 ─────────────────────────────────────────
def t1_codici(D: dict, B: dict, c: P.Config) -> None:
    S = torch.from_numpy(S1024)
    for (l, fam), b in sorted(B.items()):
        tipo = P.FAMIGLIE[fam][1]
        nb = CS.LAYOUT[tipo][0]
        blocchi = np.ascontiguousarray(b).reshape(-1, nb)
        bt = torch.from_numpy(blocchi)
        trit = CS.trit_da_blocchi(bt, tipo)
        stesso_decoder = torch.equal(trit, C1.trit_torch(bt, tipo))
        d = CS.scale_da_blocchi(bt, tipo).float()
        if tipo == "TQ1_0":
            rif = torch.from_numpy(GQ_TQ1_0.dequantize_blocks(blocchi))
            W = CS.pesi_da_trit_scale(trit, d, tipo)
        else:
            rif = torch.from_numpy(deq_r1024(blocchi))
            W = CS.pesi_da_trit_scale(trit, d, tipo, S)
        verifica(f"T1 blk.{l}.{fam} ({tipo}): byte -> trit x d == dequantizzazione della libreria al bit",
                 stesso_decoder and torch.equal(W, rif), secondo_decoder_uguale=stesso_decoder,
                 max_diff=float((W - rif).abs().max()), blocchi=blocchi.shape[0])
    # la ternarizzazione LS del down (gruppo 256) e' la stessa aritmetica del codec (gruppo 128):
    W = D["Pbf"]["blk.0.ffn_gate_exps"].float().numpy().reshape(-1, c.H)[:512]
    rot = fwht1024_f32(W * S1024[None, :])
    d_mio, t_mio = P.ternarizza_ls(rot, 128)
    b = np.ascontiguousarray(B[(0, "gate")]).reshape(-1, 224)[:512]
    t_codec = CS.trit_da_blocchi(torch.from_numpy(b), "TQ1_0R_1024").numpy()
    d_codec = b[:, 208:224].copy().view(np.float16)
    verifica("T1 ternarizza_ls(gruppo 128) == codec C-equivalente (trit e scale fp16)",
             np.array_equal(t_mio, t_codec) and np.array_equal(d_mio.astype(np.float16).view(np.uint16),
                                                               d_codec.view(np.uint16)), righe=512)


# ───────────────────────────────────────── T2 ─────────────────────────────────────────
def t2_fedelta(D: dict, B: dict, c: P.Config) -> None:
    st64 = P.Studente(D["Pbf"], B, c, dtype=torch.float64)
    g = torch.Generator().manual_seed(3)
    n = torch.randn(64, c.H, generator=g, dtype=torch.float64)
    Wg, Wu, Wd = st64.pesi(0, "gate"), st64.pesi(0, "up"), st64.pesi(0, "down")
    S = torch.from_numpy(S1024).double()
    peggiore = 0.0
    for e in range(c.E):
        yB = (torch.nn.functional.silu(st64.ruota(n) @ Wg[e].T) * (st64.ruota(n) @ Wu[e].T)) @ Wd[e].T
        yA = P.swiglu(n, _RT_1024_torch(Wg[e], S), _RT_1024_torch(Wu[e], S), Wd[e])
        peggiore = max(peggiore, rel(yB, yA))
    verifica("T2 ricetta B (attivazione ruotata) == ricetta A (peso de-ruotato), float64, 8 esperti",
             peggiore < 1e-12, errore_relativo=f"{peggiore:.2e}")
    # studente fp32 contro un modello costruito coi pesi dequantizzati DALLE LIBRERIE (G0 giocattolo)
    Pdeq = P.maestro_calcolo(D["Pbf"])
    for (l, fam), b in B.items():
        nome, tipo = P.FAMIGLIE[fam]
        blocchi = np.ascontiguousarray(b).reshape(-1, CS.LAYOUT[tipo][0])
        deq = GQ_TQ1_0.dequantize_blocks(blocchi) if tipo == "TQ1_0" else deq_r1024(blocchi)
        Pdeq[f"blk.{l}.{nome}"] = torch.from_numpy(deq).reshape(c.E, b.shape[1], -1)
    ids = D["held"][:16, :-1]
    st = P.Studente(D["Pbf"], B, c)
    with torch.no_grad():
        zS = st.logit(ids)
        x = Pdeq["token_embd"][ids]
        for l in range(c.L):
            x = P.strato_denso(x, Pdeq, l, c)[0]
        zA = P.rmsnorm(x, Pdeq["output_norm"], c.eps_norm).reshape(-1, c.H) @ Pdeq["output"].T
        kl = float(torch.nn.functional.kl_div(torch.log_softmax(zS, -1), torch.log_softmax(zA, -1),
                                              log_target=True, reduction="batchmean"))
    verifica("T2 studente (byte, ricetta B) == modello dai pesi dequantizzati da gguf-py/codec (fp32)",
             rel(zS, zA) < 1e-4 and kl < 1e-8, errore_relativo_logit=f"{rel(zS, zA):.2e}", kl=f"{kl:.2e}")


# ───────────────────────────────────────── T3 ─────────────────────────────────────────
def t3_scale() -> None:
    g = torch.Generator().manual_seed(5)
    s = torch.cat([torch.randn(20000, generator=g) * 0.02, torch.tensor([6.1e-5, 5.9e-8, 3e-8, 1e-9, 65504.0])])
    r = P.ArrotondaFP16.apply(s)
    np16 = s.numpy().astype(np.float16)
    uguale = np.array_equal(r.to(torch.float16).numpy().view(np.uint16), np16.view(np.uint16))
    s2 = s.clone().requires_grad_(True)
    (P.ArrotondaFP16.apply(s2) * torch.arange(s.numel(), dtype=torch.float32)).sum().backward()
    ste = torch.equal(s2.grad, torch.arange(s.numel(), dtype=torch.float32))
    verifica("T3 ArrotondaFP16: avanti == numpy astype(float16) al bit (subnormali compresi), indietro identita'",
             uguale and ste, valori=s.numel())
    d0 = torch.tensor([0.013, -0.011, 0.0, 2.0e-6])
    th = torch.tensor([0.3, 0.3, 0.3, -0.1], requires_grad=True)
    d = P.ArrotondaFP16.apply(d0 * torch.exp(th))
    d.sum().backward()
    segno = bool((torch.sign(d) == torch.sign(d0)).all())
    zero = float(d[2].detach()) == 0.0 and float(th.grad[2]) == 0.0
    try:
        P.ArrotondaFP16.apply(torch.tensor([7e4]))
        trabocco = False
    except FloatingPointError:
        trabocco = True
    verifica("T3 d0*exp(theta): segno conservato, zero resta zero con gradiente 0, trabocco fp16 rifiutato",
             segno and zero and trabocco, segno=segno, zero=zero, trabocco_rifiutato=trabocco)


# ───────────────────────────────────────── T4 ─────────────────────────────────────────
def t4_kl() -> None:
    g = torch.Generator().manual_seed(11)
    hS = torch.randn(10, 16, generator=g, dtype=torch.float64, requires_grad=True)
    WS = torch.randn(37, 16, generator=g, dtype=torch.float64)
    hT = torch.randn(10, 16, generator=g, dtype=torch.float64)
    WT = torch.randn(37, 16, generator=g, dtype=torch.float64)
    a = P.KLAPezzi.apply(hS, WS, hT, WT, 3)
    ga, = torch.autograd.grad(a, hS)
    b = P.kl_diretta(hS, WS, hT, WT)
    gb, = torch.autograd.grad(b, hS)
    gc = torch.autograd.gradcheck(lambda h: P.KLAPezzi.apply(h, WS, hT, WT, 3), (hS,))
    verifica("T4 KL a pezzi (gradiente in forma chiusa) == autograd, float64; gradcheck", gc and rel(a, b) < 1e-12
             and rel(ga, gb) < 1e-12, valore=f"{rel(a, b):.1e}", gradiente=f"{rel(ga, gb):.1e}", gradcheck=gc)


# ───────────────────────────────────────── T5 ─────────────────────────────────────────
def t5_cache(D: dict, c: P.Config) -> None:
    Pc = P.maestro_calcolo(D["Pbf"])
    ids = D["held"][:16, :-1]
    with torch.no_grad():
        x = Pc["token_embd"][ids]
        for l in range(c.L):
            x = P.strato_denso(x, Pc, l, c)[0]
        z_diretti = P.rmsnorm(x, Pc["output_norm"], c.eps_norm).to(torch.bfloat16).float().reshape(-1, c.H) \
            @ D["Pbf"]["output"].float().T
    z_cache = P.logit_maestro(D["held_hT"][:16 * c.ctx], D["WT"])
    verifica("T5 logit del maestro dallo stato nascosto bf16 in cache == maestro diretto, al bit",
             torch.equal(z_cache, z_diretti), token=z_cache.shape[0],
             byte_per_token_cache=c.H * 2)


# ───────────────────────────────────────── T6 ─────────────────────────────────────────
def t6_checkpoint(D: dict, B: dict, c: P.Config) -> None:
    for dtype, soglia, n_seq in ((torch.float64, 1e-10, 2), (torch.float32, 1e-4, 8)):
        ids = D["cal"][:n_seq, :-1]
        hT = D["cal_hT"][:n_seq * c.ctx]
        ga = {}
        for modo in ("checkpoint", "grafo"):
            st = P.Studente(D["Pbf"], B, c, dtype=dtype)
            with torch.no_grad():                     # un punto non banale: theta gia' mossi
                for i, k in enumerate(sorted(st.theta)):
                    st.theta[k].copy_(torch.randn(st.theta[k].shape, generator=torch.Generator().manual_seed(i)) * 0.05)
            if modo == "checkpoint":
                P.gradiente_micro(st, ids, hT, D["WT"], 100, 1.0)
                cambi = st.cambi_instr
            else:
                P.gradiente_grafo_intero(st, ids, hT, D["WT"], 1.0)
            ga[modo] = torch.cat([st.theta[k].grad.reshape(-1).double() for k in sorted(st.theta)])
        e = rel(ga["checkpoint"], ga["grafo"])
        verifica(f"T6 checkpoint per strato + instradamento salvato == grafo intero ({str(dtype)[6:]})",
                 e < soglia and cambi == 0, errore_relativo=f"{e:.2e}", cambi_instradamento=cambi, seq=n_seq)


# ───────────────────────────────────────── T7 ─────────────────────────────────────────
def t7_g1(D: dict, B: dict, c: P.Config, ca: P.ConfigAllena) -> float:
    st = P.Studente(D["Pbf"], B, c)
    ids, hT = D["cal"][:ca.seq_per_passo], D["cal_hT"][:ca.seq_per_passo * c.ctx]
    eps, stat = P.cancello_g1(st, ids, hT, D["WT"], ca)
    opt = P.nuovo_ottimizzatore(st, ca, eps)
    n_tok = ids.shape[0] * c.ctx
    for j in range(0, ids.shape[0], ca.seq_per_micro):
        P.gradiente_micro(st, ids[j:j + ca.seq_per_micro, :-1], hT[j * c.ctx:(j + ca.seq_per_micro) * c.ctx],
                          D["WT"], ca.pezzo_kl, 1.0 / n_tok)
    g0 = torch.cat([st.theta[k].grad.reshape(-1) for k in sorted(st.theta)])
    opt.step()
    th = torch.cat([st.theta[k].detach().reshape(-1) for k in sorted(st.theta)])
    vive = (torch.cat([st.d0[k].reshape(-1) for k in sorted(st.theta)]) != 0) & (g0 != 0)
    pieno = float((th[vive].abs() >= 0.99 * ca.lr).double().mean())
    med = float(th[vive].abs().median())
    verso = bool((torch.sign(th[vive]) == -torch.sign(g0[vive])).all())
    wd = [g["weight_decay"] for g in opt.param_groups]
    col_default = float((g0[vive].abs() < 10 * 1e-8).double().mean())
    verifica("T7 G1: col eps scelto il primo passo di Adam e' >= 0,99 lr sul 99% delle scale vive, verso -g, wd = 0",
             pieno >= 0.99 and abs(med / ca.lr - 1) < 0.01 and verso and wd == [0.0],
             eps=f"{eps:.2e}", mediana_g={k: f"{v['mediana']:.2e}" for k, v in stat.items()},
             q01_g={k: f"{v['q01']:.2e}" for k, v in stat.items()},
             vive_con_gradiente_zero={k: v["vive_con_gradiente_zero"] for k, v in stat.items()},
             frazione_passo_pieno=f"{pieno:.4f}", mediana_passo_su_lr=f"{med / ca.lr:.4f}",
             frazione_sotto_10x_eps_default_1e8=f"{col_default:.4f}")
    return eps


# ───────────────────────────────────────── T8 ─────────────────────────────────────────
def t8_allena(D: dict, B: dict, c: P.Config, ca: P.ConfigAllena, eps: float) -> tuple:
    sha_iniziali = {k: hashlib.sha256(np.ascontiguousarray(b).tobytes()).hexdigest() for k, b in B.items()}
    st = P.Studente(D["Pbf"], B, c)
    t0 = time.time()
    es = P.allena(st, D["cal"], D["cal_hT"], D["held"], D["held_hT"], D["WT"], ca, eps)
    secondi = time.time() - t0
    stato = es["stato"]
    # (a) codici
    sha_dopo = {k: hashlib.sha256(b.numpy().tobytes()).hexdigest() for k, b in st.blocchi.items()}
    solo_theta = all(p.dtype == torch.float32 and p.requires_grad for p in st.parametri()) and \
        all(not b.requires_grad and b.dtype == torch.uint8 for b in st.blocchi.values())
    verifica("T8(a) codici identici al bit dopo ogni passo e alla fine; unici parametri = theta",
             sha_dopo == sha_iniziali and sha_dopo == es["sha_blocchi"] and solo_theta,
             passi=stato["passo"], parametri=sum(p.numel() for p in st.parametri()))
    # (b) scale fp16
    mig = stato["migliore"]
    ricalcolo = {k: (st.d0[k] * torch.exp(mig["theta"][k])).to(torch.float16) for k in mig["theta"]}
    stesse = all(torch.equal(ricalcolo[k].view(torch.int16), mig["scale"][k].view(torch.int16)) for k in ricalcolo)
    finite = all(bool(torch.isfinite(s.float()).all()) for s in mig["scale"].values())
    segni = all(bool((torch.sign(mig["scale"][k].float()) == torch.sign(st.d0[k])).all()) for k in ricalcolo)
    cambiate = sum(int((mig["scale"][k].view(torch.int16) != st.d0[k].to(torch.float16).view(torch.int16)).sum())
                   for k in ricalcolo)
    rapporti = torch.cat([(mig["scale"][k].float() / st.d0[k]).reshape(-1) for k in ricalcolo if True])
    rapporti = rapporti[torch.isfinite(rapporti)]
    verifica("T8(b) scale del migliore: fp16 finite, == fp16(d0*exp(theta)) al bit, segno conservato, nessuno zero nuovo",
             stesse and finite and segni and st.zeri_nuovi == 0, scale_cambiate=f"{cambiate}/{rapporti.numel()}",
             rapporto_d_su_d0={"min": f"{float(rapporti.min()):.4f}", "mediana": f"{float(rapporti.median()):.4f}",
                               "max": f"{float(rapporti.max()):.4f}"})
    # (c) held-out
    h = [v for v in stato["storia"] if "kl" in v]
    kl0, klm = h[0]["kl"], mig["kl"]
    ce0 = h[0]["ce_studente"]
    ce_m = next(v["ce_studente"] for v in h if v["passo"] == mig["passo"])
    verifica("T8(c) KL held-out scende (e la CE sul byte vero non sale)", klm < 0.9 * kl0 and ce_m <= ce0,
             kl_iniziale=f"{kl0:.5f}", kl_migliore=f"{klm:.5f}", riduzione=f"{100 * (1 - klm / kl0):.1f}%",
             ce_iniziale=f"{ce0:.4f}", ce_migliore=f"{ce_m:.4f}", ce_maestro=f"{h[0]['ce_maestro']:.4f}",
             passo_migliore=mig["passo"], passi=stato["passo"], cambi_instradamento=stato["cambi_instr"],
             minuti=f"{secondi / 60:.1f}")
    ESITO["T8 storia held-out"] = [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in x.items()} for x in h]
    return st, es


# ───────────────────────────────────────── T9 ─────────────────────────────────────────
def t9_ripresa(D: dict, B: dict, c: P.Config, eps: float) -> None:
    ca = P.ConfigAllena(epoche=1, heldout_ogni=2)
    muto = lambda *a, **k: None  # noqa: E731
    cal, held = D["cal"][:96], D["held"][:16]
    cal_hT, held_hT = D["cal_hT"][:96 * c.ctx], D["held_hT"][:16 * c.ctx]
    filata = P.allena(P.Studente(D["Pbf"], B, c), cal, cal_hT, held, held_hT, D["WT"], ca, eps, log=muto)
    prima = P.allena(P.Studente(D["Pbf"], B, c), cal, cal_hT, held, held_hT, D["WT"], ca, eps, ferma_a=3, log=muto)
    f = P.DIR / "ripresa_prova.pt"
    P.salva_ripresa(f, prima)
    del prima
    ripresa = P.carica_ripresa(f)
    dopo = P.allena(P.Studente(D["Pbf"], B, c), cal, cal_hT, held, held_hT, D["WT"], ca, eps, ripresa=ripresa, log=muto)
    f.unlink()
    th = all(torch.equal(filata["theta"][k], dopo["theta"][k]) for k in filata["theta"])
    so, sd = filata["opt"]["state"], dopo["opt"]["state"]
    adam = all(torch.equal(so[i][n], sd[i][n]) for i in so for n in ("exp_avg", "exp_avg_sq", "step"))
    storia = json.dumps(filata["stato"]["storia"]) == json.dumps(dopo["stato"]["storia"])
    migliore = filata["stato"]["migliore"]["passo"] == dopo["stato"]["migliore"]["passo"] and all(
        torch.equal(filata["stato"]["migliore"]["scale"][k], dopo["stato"]["migliore"]["scale"][k])
        for k in filata["stato"]["migliore"]["scale"])
    verifica("T9 ripresa: 3 passi + salva/carica + 3 passi == 6 passi filati (theta, Adam, storia, migliore al bit)",
             th and adam and storia and migliore and filata["stato"]["passo"] == 6,
             theta=th, adam=adam, storia=storia, migliore=migliore,
             passi=filata["stato"]["passo"])


# ───────────────────────────────────────── T10 ─────────────────────────────────────────
def t10_export(D: dict, B: dict, c: P.Config, st: P.Studente, es: dict) -> None:
    src, dst = P.DIR / "studente_iniziale.gguf", P.DIR / "studente_e2e.gguf"
    for p in (src, dst):
        if p.exists():
            p.unlink()
    P.scrivi_gguf(src, D["Pbf"], B, c)
    mig = es["stato"]["migliore"]
    rep = P.esporta(src, dst, mig["scale"])
    verifica("T10 export: ogni byte fuori dai campi d identico (file intero), sorgente intatto",
             rep["sha256_sorgente"] == hashlib.sha256(src.read_bytes()).hexdigest()
             and src.stat().st_size == dst.stat().st_size,
             byte=rep["byte"], scale_cambiate=rep["scale_cambiate"], sha256_senza_d=rep["sha256_senza_d"][:16])
    Pbf2, B2 = P.leggi_gguf(dst, c)
    S = torch.from_numpy(S1024)
    for k, b2 in sorted(B2.items()):
        tipo = P.FAMIGLIE[k[1]][1]
        nb, _, off, kk = CS.LAYOUT[tipo]
        codici = P.sha_codici(b2, tipo) == P.sha_codici(B[k], tipo) == P.sha_codici(st.blocchi[k].numpy(), tipo)
        blocchi = np.ascontiguousarray(b2).reshape(-1, nb)
        d_file = blocchi[:, off:off + 2 * kk].copy().view(np.float16).reshape(mig["scale"][k].shape)
        d_ok = np.array_equal(d_file.view(np.uint16), mig["scale"][k].numpy().view(np.uint16))
        trit = CS.trit_da_blocchi(torch.from_numpy(np.ascontiguousarray(B[k]).reshape(-1, nb)), tipo)
        d_nuove = torch.from_numpy(d_file.reshape(-1, kk).astype(np.float32))
        if tipo == "TQ1_0":
            lib, atteso = GQ_TQ1_0.dequantize_blocks(blocchi), CS.pesi_da_trit_scale(trit, d_nuove, tipo).numpy()
        else:
            lib, atteso = deq_r1024(blocchi), CS.pesi_da_trit_scale(trit, d_nuove, tipo, S).numpy()
        verifica(f"T10 blk.{k[0]}.{k[1]} ({tipo}): codici del file == iniziali == in memoria; d del file == scale "
                 f"allenate; dequantizzazione della libreria == trit(sorgente) x d nuove al bit",
                 codici and d_ok and np.array_equal(lib, atteso), codici=codici, d=d_ok)
    non_esp = all(_uguale(Pbf2[n], D["Pbf"][n]) for n in D["Pbf"] if not n.endswith("_exps"))
    ricaricato = P.Studente(Pbf2, B2, c)
    with torch.no_grad():
        for k, t in st.theta.items():
            t.copy_(mig["theta"][k])
        ids = D["held"][:, :-1]
        z_allenato = torch.cat([st.logit(ids[i:i + 16]) for i in range(0, ids.shape[0], 16)])
        z_file = torch.cat([ricaricato.logit(ids[i:i + 16]) for i in range(0, ids.shape[0], 16)])
    verifica("T10 studente ricaricato SOLO dal GGUF esportato == studente allenato (logit held-out al bit)",
             torch.equal(z_allenato, z_file) and non_esp, token=z_file.shape[0], non_esperti_identici=non_esp)
    ESITO["T10 export"] = {k: v for k, v in rep.items()}


def _uguale(dal_file: torch.Tensor, originale: torch.Tensor) -> bool:
    """BF16 confrontati sui bit; i tensori salvati F32 (norme, router) sono l'upcast esatto del bf16."""
    if dal_file.dtype == torch.bfloat16:
        return originale.dtype == torch.bfloat16 and torch.equal(dal_file.view(torch.int16), originale.view(torch.int16))
    return dal_file.shape == originale.shape and torch.equal(dal_file, originale.float())


def main() -> int:
    t0 = time.time()
    c, ca = P.Config(), P.ConfigAllena()
    D = P.prepara(c)
    B = P.blocchi_iniziali(D["Pbf"], c)
    ESITO["config"], ESITO["config_allena"] = P.asdict(c), P.asdict(ca)
    ESITO["maestro_CE_finale_addestramento"] = D["maestro_storia"][-1]
    with torch.no_grad():
        st = P.Studente(D["Pbf"], B, c)
        conte = [[0] * c.E for _ in range(c.L)]
        for i in range(0, D["cal"].shape[0], 64):
            x = st.entra(D["cal"][i:i + 64, :-1])
            for l in range(c.L):
                x, idx = st.strato(x, l)
                conte[l] = [a + b for a, b in zip(conte[l], P.conteggi_esperti(idx, c.E))]
    ESITO["token_per_esperto_calibrazione"] = conte
    print(f"token per esperto (calibrazione, studente iniziale): {conte}")
    t0_dati(D)
    t1_codici(D, B, c)
    t2_fedelta(D, B, c)
    t3_scale()
    t4_kl()
    t5_cache(D, c)
    t6_checkpoint(D, B, c)
    eps = t7_g1(D, B, c, ca)
    st, es = t8_allena(D, B, c, ca, eps)
    t9_ripresa(D, B, c, eps)
    t10_export(D, B, c, st, es)
    ESITO["minuti_totali"] = round((time.time() - t0) / 60, 2)
    ESITO["fallite"] = FALLITE
    (P.DIR / "esito_test.json").write_text(json.dumps(ESITO, indent=1, default=str))
    print(f"\n{'TUTTO VERDE' if not FALLITE else '⛔ FALLITE: ' + ', '.join(FALLITE)} "
          f"({ESITO['minuti_totali']} min) · esito in {P.DIR / 'esito_test.json'}")
    return 0 if not FALLITE else 1


if __name__ == "__main__":
    raise SystemExit(main())

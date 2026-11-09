#!/usr/bin/env python3
"""Cancello INDIETRO (27/9/2026, ramo indietro-bf16x3): l'indietro degli esperti a prodotti divisi (`B_ind2`,
`B3_ind2`, opz. `B_ind3`) contro `B_ind32` (la ricetta di EB2/EB3) sulla GPU della corsa, NELLA CONFIGURAZIONE DELLA
CORSA. Lo lancia il coordinatore: `strumenti/cancello_indietro.sh` (NON parte da solo).

Perche' un cancello nuovo e non `e2e_cancello_b3.py`: quello misura il tempo con il GDN di ripiego di transformers
(non chiama `e2e_allena.scegli_gdn`) e col motore «triton» (v1), mentre EB2/EB3 girano con E2E_GDN=veloce
(gdn_veloce_corsa_v5) ed E2E_MOTORE=triton2 (letto da /proc/<pid>/environ di EB2 il 27/9 16:39). Nel ripiego
l'indietro di uno strato GDN e' dominato da migliaia di kernel di copia/riempimento (TECNICHE §7: add_ 1,03 s, copy_
0,52 s, fill_ 0,48 s), e ogni guadagno sulle GEMM sparisce nel rumore: e' la ragione per cui il cancello B3 v2 ha
visto l'indietro «invariato» (3,28 -> 3,31 ms/token). Qui: `scegli_gdn` come la corsa, motore da E2E_MOTORE (anche
nei bracci di KL/dtheta, che in cancello_b3 usano «triton» fisso).

Quattro misure, un esito (`<uscita>/esito_cancello_indietro.json`):
1. **Microprova per prodotto** (strati 0 e 39, famiglie gate/up/down VERE, 8192 token, top-8 casuale, dY con righe da
   1e-8 a 1e2): dX e dL/dd nelle vie fp32 (B_ind32), fp16 con K globale (B), fp16x2, bf16x3 — ms (mediana di 5) ed
   errore contro fp64 (in norma, per riga/gruppo). E' QUI che si vede la deriva dell'accumulo WMMA (che l'interprete
   della CPU non ha) e che la via bf16 di Triton si prova per la prima volta.
2. **Tempo del passo** (`e2e_allena.misura`, strati 0,1 e 38,39, righe train presenti): ms/token avanti/KL/indietro/
   Adam per ogni ricetta.
3. **Profilo dell'indietro** (torch.profiler, una ripetizione, strati 0,1 e 38,39) per B_ind32 e per la prima
   candidata: tempo di dispositivo per categoria di kernel (esperti, bf16x3 non-esperti, GEMM torch fp32, attenzione,
   compilati, combina, altro) e i 25 kernel piu' cari. Risponde a «cosa resta nell'indietro».
4. **Fedelta'** sui `--strati-kl` (default 0-39), come cancello_b3: KL(rif || candidata) per token su righe held-out,
   dL/dtheta del primo micro-lotto train (perdita dell'allenamento, K0) contro il riferimento: quota di scale vive
   nulle o di segno opposto, quota pesata con |g|, errore relativo in norma. Braccio RUMORE Bk2_ind32 per il
   pavimento. Per le candidate B_ind* l'avanti e' IDENTICO a B_ind32: la KL deve essere 0 e h uguale al bit, e il
   criterio dtheta e' quello di G1 (due indietro sullo STESSO avanti): quota <= --soglia-dtheta (1e-3) ed errore in
   norma <= --soglia-dtheta-norma (1e-3). Per le B3_ind* (avanti bf16x3) come cancello_b3 v2: KL media <= 0,0013 e
   dtheta entro max(soglia, fattore x rumore).

Verdetto per candidata: fedele, piu_veloce (rapporto < 1 su entrambe le coppie di strati), ammessa = entrambe.
Uscita 0 se almeno una candidata e' ammessa. Non cambia nessuna corsa.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import torch

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import e2e_allena as A  # noqa: E402
import e2e_cancello_b3 as CB  # noqa: E402

MOTORE = os.environ.get("E2E_MOTORE", "triton2")


# ═══════════════════════════════════ 1. microprova per prodotto ═══════════════════════════════════
def _ms(fn, n: int = 5) -> float:
    if not torch.cuda.is_available():            # prova di fumo su CPU (--micro-esperti): tempo a parete, n = 1
        t0 = time.perf_counter()
        fn()
        return 1000 * (time.perf_counter() - t0)
    fn()
    torch.cuda.synchronize()
    t = []
    for _ in range(n):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        t.append(a.elapsed_time(b))
    return sorted(t)[len(t) // 2]


def _err(a, b) -> dict:
    a, b = a.double(), b.double()
    d = (a - b).reshape(b.shape[0], -1)
    bb = b.reshape(b.shape[0], -1)
    n = bb.norm(dim=1)
    v = n > 0
    righe = (d.norm(dim=1)[v] / n[v]) if bool(v.any()) else torch.zeros(1, device=b.device, dtype=torch.float64)
    return {"norma": float((a - b).norm() / b.norm().clamp_min(1e-300)), "righe_p99": A._kth(righe.float(), 0.99),
            "righe_max": float(righe.max())}


def microprova(gguf, device: str, strati=(0, 39), T: int = 8192, log=print, esperti=None, righe=None) -> dict:
    import e2e_studente as S
    import e2e_indietro_hilo as HL
    from gguf import GGUFReader
    dev = torch.device(device)
    tens = {t.name: t for t in GGUFReader(str(gguf)).tensors}
    mot = S.motore(MOTORE)
    out = {"motore": MOTORE, "T": T}
    g = torch.Generator(device=dev).manual_seed(0)
    for L in strati:
        for n in ("gate", "up", "down"):
            fam = S.famiglia_da_tensore(tens[f"blk.{L}.{S.FAMIGLIE[n]}.weight"], n, device, esperti=esperti, righe=righe)
            idx = torch.topk(torch.rand(T, fam.E, device=dev, generator=g), min(8, fam.E), dim=-1).indices
            piano = S.Piano(idx, fam.E)
            x = (torch.randn(piano.S, fam.C, device=dev, generator=g) * 0.5).half()
            esp = torch.empty(piano.S, 1, device=dev).uniform_(-8, 2, generator=g)
            gy = (torch.randn(piano.S, fam.R, device=dev, generator=g) * torch.pow(10.0, esp)).float()
            d = fam.d0.half().float()
            # fp64 (decoder dello studente: trit_esperto, provato contro quello indipendente nelle prove CPU)
            o = piano.offs.tolist()
            dx64 = torch.zeros(piano.S, fam.C, device=dev, dtype=torch.float64)
            dd64 = torch.zeros(fam.E, fam.R, fam.NG, device=dev, dtype=torch.float64)
            for e in range(fam.E):
                if o[e + 1] > o[e]:
                    q = S.trit_esperto(fam, e).double()
                    W = (q.view(fam.R, fam.NG, fam.gs) * d[e].double().unsqueeze(-1)).view(fam.R, fam.C)
                    ge = gy[o[e]:o[e + 1]].double()
                    dx64[o[e]:o[e + 1]] = ge @ W
                    dd64[e] = (q * (ge.T @ x[o[e]:o[e + 1]].double())).view(fam.R, fam.NG, fam.gs).sum(-1)
                    del q, W, ge
            r = {}
            ric32 = S.RICETTE["B_ind32"]
            vie = {
                "fp32": (lambda: mot.indietro_x(gy, d, fam, piano, ric32),
                         lambda: mot.dd(gy, x.float(), fam, piano, torch.float32)),
                "fp16_K": (lambda: HL.prodotto_dx(mot, (gy * 2.0 ** 20).half(), d, fam, piano) * 2.0 ** -20,
                           lambda: HL.prodotto_dd(mot, (gy * 2.0 ** 20).half(), x, fam, piano) * 2.0 ** -20),
            }
            for via in HL.VIE:
                ric = dataclasses.replace(S.RICETTE["B_ind2"], indietro_diviso=via)

                def fdx(ric=ric):
                    return HL.indietro_famiglia(gy, x, d, fam, piano, ric, mot, None, True, False)[0]

                def fdd(ric=ric):
                    return HL.indietro_famiglia(gy, x, d, fam, piano, ric, mot, None, False, True)[1]
                vie[via] = (fdx, fdd)
            for via, (fdx, fdd) in vie.items():
                try:
                    a = fdx()
                    b = fdd()
                    # dx: errore del PRODOTTO (fp32, prima dell'arrotondamento a fp16 che la ricetta fa comunque) e
                    # dopo l'arrotondamento (dx_fp16, quello che l'indietro usa davvero); il tempo e' del gancio intero
                    a32 = HL.dx_diviso(mot, gy, d, fam, piano, via) if via in HL.VIE else a
                    r[via] = {"dx_ms": _ms(fdx), "dd_ms": _ms(fdd), "dx": _err(a32, dx64),
                              "dx_fp16": _err(a.half(), dx64), "dd": _err(b, dd64),
                              "dx_nonfiniti": int((~torch.isfinite(a)).sum()), "dd_nonfiniti": int((~torch.isfinite(b)).sum())}
                    del a, b
                except Exception as ex:               # una via rotta si registra, le altre si misurano
                    r[via] = {"errore": repr(ex)[:300]}
            # passo V (--pv-codici della corsa): nei micro-lotti di proposta/verifica il dL/dd passa dal kernel V
            # `_k_dd_v` (e2e_pv_codici), non da _k_dd2: fp32 (B_ind32) contro diviso fp16x2 (B_ind2), modo 2
            try:
                import e2e_pv_codici as PV
                buf = PV.BufferV.crea(fam)
                blk = PV.BLOCCHI_V
                PV.dd_v_triton(gy, x.float(), fam, piano, blk, buf, 1)          # candidati per la verifica

                def v32():
                    return PV.dd_v_triton(gy, x.to(torch.float32), fam, piano, blk, buf, 2)

                def v2():
                    eid = HL.esperto_di_riga(piano, gy.shape[0])
                    s_er, inv_er = HL.scala_pot2(HL.massimi_segmento(gy.abs(), eid, fam.E))
                    hi, lo = HL.dividi_fp16(gy * s_er[eid])
                    return PV.dd_v_triton(hi, x, fam, piano, blk, buf, 2, dy_lo=lo, inv=inv_er)
                r["V_modo2"] = {"fp32_ms": _ms(v32), "fp16x2_ms": _ms(v2), "dd_fp32": _err(v32(), dd64),
                                "dd_fp16x2": _err(v2(), dd64)}
                del buf
            except Exception as ex:
                r["V_modo2"] = {"errore": repr(ex)[:300]}
            out[f"{L}_{n}"] = r
            log(f"microprova strato {L} {n}: V modo 2 {json.dumps(r['V_modo2'], default=str)[:300]}")
            log(f"microprova strato {L} {n}: " + ", ".join(
                f"{v} dx {r[v].get('dx_ms', float('nan')):.1f} ms/{r[v].get('dx', {}).get('norma', float('nan')):.1e} "
                f"dd {r[v].get('dd_ms', float('nan')):.1f} ms/{r[v].get('dd', {}).get('norma', float('nan')):.1e}"
                for v in r if v != "V_modo2"))
            del fam, piano, x, gy, d, dx64, dd64
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    return out


# ═══════════════════════════════════ 3. profilo dell'indietro ═══════════════════════════════════
CATEGORIE = (("esperti", r"_k_(gemm2?|dx2?|dd2?|dd_v|decodifica)\b|_k_(gemm|dx|dd)\d"),
             ("bf16x3_nonesperti", r"_k_x3"),
             ("combina", r"_k_combina"),
             ("gemm_torch", r"Cijk|[Gg]emm|gemv|\bmm\b|addmm|bmm|hipblas|rocblas"),
             ("attenzione", r"softmax|attention|sdpa|flash|fmha"),
             ("compilati_triton", r"^triton_"),
             ("copie_riempimenti", r"[Cc]opy|fill|cat_|index|gather|scatter"))


def _categoria(nome: str) -> str:
    for c, rx in CATEGORIE:
        if re.search(rx, nome):
            return c
    return "altro"


def profilo_indietro(strati: list, teacher: Path, gguf: Path, ca, device: str, ricetta: str, log=print) -> dict:
    """Un avanti+KL+indietro di riscaldamento, poi l'indietro profilato: tempo di dispositivo per categoria."""
    import e2e_studente as S
    from torch.profiler import ProfilerActivity, profile
    src = A.SorgenteCache(teacher)
    righe = A.prime_disponibili(src, "train", ca.righe_per_micro)
    dev = torch.device(device)
    WT, _ = A.testa_maestro(dev, torch.bfloat16 if ca.logit_maestro == "bf16" else torch.float32)
    st = S.Studente(gguf=gguf, strati=strati, ricetta=ricetta, motore_nome=MOTORE, device=device, log=log,
                    **({"densi_ternari": ca.densi_ternari} if ca.densi_ternari else {}))   # 1/10 TT5: densi a 3 piani
    st.carica_embedding()
    WS = A.testa_studente(st, ca.testa_kl)
    car = A.Caricatore(src, dev)
    lot = car.prendi("train", righe)
    ntok = lot.ids.numel()

    def passo(prof=None):
        stato = st.avanti(lot.ids, inizi=lot.inizi)
        xL = stato["xL"].detach().requires_grad_(True)
        with torch.enable_grad():
            kl = A.KLFusa.apply(st.esce(xL), WS, lot.hT, WT, ca.pezzo_kl)
            (kl * (2.0 ** 20 / ntok)).backward()
        torch.cuda.synchronize()
        if prof is None:
            st.indietro(stato, xL.grad)
        else:
            with prof:
                st.indietro(stato, xL.grad)
                torch.cuda.synchronize()
        for p in st.parametri():
            p.grad = None
    passo()
    prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
    passo(prof)
    per_cat: dict = {}
    righe_k = []
    for ev in prof.key_averages():
        t = float(getattr(ev, "device_time_total", 0.0) or getattr(ev, "cuda_time_total", 0.0) or 0.0)
        if t <= 0 or getattr(ev, "device_type", None) is not None and str(ev.device_type).endswith("CPU"):
            continue
        c = _categoria(ev.key)
        per_cat[c] = per_cat.get(c, 0.0) + t
        righe_k.append((t, ev.key, ev.count, c))
    tot = sum(per_cat.values()) or 1.0
    righe_k.sort(reverse=True)
    out = {"strati": strati, "ricetta": ricetta, "token": ntok, "tempo_dispositivo_ms": tot / 1000,
           "ms_token_per_categoria": {c: v / 1000 / ntok for c, v in sorted(per_cat.items(), key=lambda z: -z[1])},
           "quota_per_categoria": {c: v / tot for c, v in sorted(per_cat.items(), key=lambda z: -z[1])},
           "kernel_top25": [{"ms": t / 1000, "nome": k[:160], "chiamate": n, "categoria": c}
                            for t, k, n, c in righe_k[:25]]}
    car.chiudi()
    del st, WS, WT, car, lot
    torch.cuda.empty_cache()
    log(f"profilo indietro {ricetta} strati {strati}: {json.dumps(out['quota_per_categoria'])}")
    return out


# ═══════════════════════════════════ 4. bracci (motore della corsa) ═══════════════════════════════════
def braccio(ricetta, strati, src, righe_held, righe_train, ca, gguf, device, log=print) -> dict:
    """`e2e_cancello_b3.braccio` col motore di E2E_MOTORE (la' e' «triton» fisso)."""
    import e2e_studente as S
    orig = S.Studente.__init__

    def init(self, *a, **k):
        k["motore_nome"] = MOTORE if torch.device(device).type == "cuda" else "torch"
        if getattr(ca, "densi_ternari", ()):          # 1/10 TT5: i densi ternari a 3 piani come nella corsa
            k["densi_ternari"] = ca.densi_ternari
        return orig(self, *a, **k)
    S.Studente.__init__ = init
    try:
        return CB.braccio(ricetta, strati, src, righe_held, righe_train, ca, gguf, device, log)
    finally:
        S.Studente.__init__ = orig


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--teacher", default="@@ROOT:lavoro@@/e2e/teacher/q8-EB-pieno")
    ap.add_argument("--gguf", default=None)
    ap.add_argument("--uscita", default=str(A.RADICE / "cancello_indietro"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ricetta-rif", default="B_ind32")
    ap.add_argument("--candidate", default="B_ind2,B3_ind2",
                    help="ricette candidate separate da virgola (B_ind2, B3_ind2, B_ind3, B3_ind3, B3_ind32)")
    ap.add_argument("--testa-kl-b3", choices=("fp32", "bf16x3"), default="bf16x3", help="teste delle candidate B3*")
    ap.add_argument("--logit-maestro-b3", choices=("fp32", "bf16"), default="bf16")
    ap.add_argument("--strati-tempo", action="append", default=None, help="default: 0,1 e 38,39")
    ap.add_argument("--strati-kl", default="0-39")
    ap.add_argument("--righe-micro", type=int, default=A.ConfigAllena.righe_per_micro)
    ap.add_argument("--righe-held", type=int, default=2)
    ap.add_argument("--ripetizioni", type=int, default=3)
    ap.add_argument("--soglia-kl", type=float, default=0.0013)
    ap.add_argument("--soglia-dtheta", type=float, default=A.ConfigAllena.max_quota_dtheta_rotti)
    ap.add_argument("--soglia-dtheta-norma", type=float, default=1e-3)
    ap.add_argument("--ricetta-rumore", default="Bk2_ind32")
    ap.add_argument("--fattore-rumore", type=float, default=2.0)
    ap.add_argument("--salta-micro", action="store_true")
    ap.add_argument("--micro-esperti", type=int, default=0,
                    help="SOLO prova di fumo su CPU (TRITON_INTERPRET=1): microprova sui primi N esperti e 64 righe, "
                         "T = 32 token")
    ap.add_argument("--salta-tempo", action="store_true")
    ap.add_argument("--salta-profilo", action="store_true")
    ap.add_argument("--salta-fedelta", action="store_true")
    # 1/10 TT5-ottimizzato: la configurazione di TT5 (R2B3m, --densi-ternari tutte, rif B3_ind32 con teste bf16x3)
    ap.add_argument("--densi-ternari", default="", help="come e2e_allena (es. 'tutte'): densi ternari nei bracci")
    ap.add_argument("--salta-rumore", action="store_true",
                    help="niente braccio rumore (inutile se tutte le candidate hanno lo stesso avanti del riferimento)")
    a = ap.parse_args(argv)
    import e2e_studente as S
    import e2e_bf16x3 as X3
    gguf = Path(a.gguf) if a.gguf else S.GGUF_THOR_T
    usc = Path(a.uscita)
    usc.mkdir(parents=True, exist_ok=True)
    log = A._registro(usc)
    candidate = [c for c in a.candidate.split(",") if c]
    for c in candidate + [a.ricetta_rif, a.ricetta_rumore]:
        if c not in S.RICETTE:
            raise SystemExit(f"ricetta sconosciuta {c}: {sorted(S.RICETTE)}")
    strati_tempo = [CB._strati(s) for s in (a.strati_tempo or ["0,1", "38,39"])]
    ca_rif = A.ConfigAllena(righe_per_micro=a.righe_micro, densi_ternari=A._famiglie_dense(a.densi_ternari or ""))
    ca_b3 = dataclasses.replace(ca_rif, testa_kl=a.testa_kl_b3, logit_maestro=a.logit_maestro_b3)

    def ca_di(ric: str):
        return ca_b3 if S.RICETTE[ric].non_esperti_x3 else ca_rif

    def stesso_avanti_di(ric: str) -> bool:
        """1/10: la candidata ha lo STESSO avanti del riferimento (stessa aritmetica dell'avanti e stesse teste): allora
        il criterio e' quello di G1 (h al bit, KL 0). Prima valeva solo per rif B_ind32 (non_esperti_x3 False)."""
        x, y = S.RICETTE[ric], S.RICETTE[a.ricetta_rif]
        campi = ("dtype_residuo", "dtype_gemm", "ruota", "router_fp32", "non_esperti_x3", "non_esperti_k2")
        cx, cy = ca_di(ric), ca_di(a.ricetta_rif)
        return (all(getattr(x, c) == getattr(y, c) for c in campi) and cx.testa_kl == cy.testa_kl
                and cx.logit_maestro == cy.logit_maestro)
    gdn = A.scegli_gdn(log)                       # COME LA CORSA (E2E_GDN / E2E_GDN_MODULO)
    esito = {"data": A.adesso(), "argv": sys.argv, "gguf": str(gguf), "teacher": a.teacher, "motore": MOTORE,
             "gdn": gdn, "gdn_modulo": os.environ.get("E2E_GDN_MODULO"), "rif": a.ricetta_rif, "candidate": candidate,
             "soglie": {"kl": a.soglia_kl, "dtheta_quota": a.soglia_dtheta, "dtheta_norma": a.soglia_dtheta_norma,
                        "fattore_rumore": a.fattore_rumore}}
    dest = usc / "esito_cancello_indietro.json"
    log(f"cancello indietro: motore {MOTORE}, GDN {gdn} ({os.environ.get('E2E_GDN_MODULO')}), rif {a.ricetta_rif}, "
        f"candidate {candidate}")
    with A._coesistenza_gpu(a.device == "cuda"):
        if a.device == "cuda":
            A._guardia_gpu(600)
        if not a.salta_micro and (a.device == "cuda" or a.micro_esperti):
            try:
                kw = {"esperti": slice(0, a.micro_esperti), "righe": slice(0, 64), "T": 32} if a.micro_esperti else {}
                esito["microprova"] = microprova(gguf, a.device, log=log, **kw)
            except Exception as ex:
                esito["microprova"] = {"errore": repr(ex)[:500]}
                log(f"microprova fallita: {ex!r}")
            A._scrivi_json(dest, esito)
        src = A.SorgenteCache(a.teacher)
        righe_train = A.prime_disponibili(src, "train", a.righe_micro)
        righe_held = A.prime_disponibili(src, "heldout", a.righe_held)
        esito.update({"righe_train": righe_train, "righe_held": righe_held})
        if not a.salta_tempo:
            tempo = {}
            for ric in [a.ricetta_rif] + candidate:
                tempo[ric] = A.misura(strati_tempo, Path(a.teacher), gguf, ca_di(ric), a.ripetizioni, a.device, ric,
                                      log=log, disponibili=True)["configurazioni"]
            esito["tempo"] = {ric: {s: {"totale_ms_token": v["totale_ms_token"], "parti": v["ms_token"],
                                        "picco_gib": v["picco_torch_gib"],
                                        "rapporto_su_rif": v["totale_ms_token"]
                                        / tempo[a.ricetta_rif][s]["totale_ms_token"],
                                        "rapporto_indietro_su_rif": v["ms_token"]["indietro"]
                                        / tempo[a.ricetta_rif][s]["ms_token"]["indietro"]}
                                    for s, v in conf.items()} for ric, conf in tempo.items()}
            for ric in candidate:
                for s, v in esito["tempo"][ric].items():
                    log(f"tempo {ric} strati {s}: {v['totale_ms_token']:.4f} ms/token (x{v['rapporto_su_rif']:.3f}; "
                        f"indietro x{v['rapporto_indietro_su_rif']:.3f})")
            A._scrivi_json(dest, esito)
        if not a.salta_profilo and a.device == "cuda":
            esito["profilo"] = {}
            for ric in [a.ricetta_rif] + candidate[:1]:
                for s in strati_tempo:
                    try:
                        esito["profilo"][f"{ric}|{','.join(map(str, s))}"] = profilo_indietro(
                            s, Path(a.teacher), gguf, ca_di(ric), a.device, ric, log)
                    except Exception as ex:
                        esito["profilo"][f"{ric}|{','.join(map(str, s))}"] = {"errore": repr(ex)[:500]}
                        log(f"profilo {ric} {s} fallito: {ex!r}")
            A._scrivi_json(dest, esito)
        if not a.salta_fedelta:
            strati_kl = CB._strati(a.strati_kl)
            ca_r = ca_di(a.ricetta_rif)                   # 1/10: le teste del riferimento (B3_ind32 di TT5: bf16x3)
            bR = braccio(a.ricetta_rif, strati_kl, src, righe_held, righe_train, ca_r, gguf, a.device, log)
            dev = torch.device(a.device)
            WS = CB.testa_gguf(gguf, dev)
            WS_R = X3.TestaX3(WS) if ca_r.testa_kl == "bf16x3" else WS
            if a.salta_rumore:
                esito["rumore"] = {"ricetta": None, "dtheta": {"quota_rotti": 0.0, "quota_rotti_pesata": 0.0,
                                                               "err_rel_norma": 0.0}}
            else:
                bN = braccio(a.ricetta_rumore, strati_kl, src, righe_held, righe_train, ca_rif, gguf, a.device, log)
                rum = CB.confronto(bR, bN, WS_R, WS, ca_rif, dev)
                esito["rumore"] = {"ricetta": a.ricetta_rumore, "kl": rum["kl_B_B3"], "dtheta": rum["dtheta"],
                                   "h_err_rel": rum["h_err_rel"]}
                del bN
            esito["fedelta"] = {}
            for ric in candidate:
                ca = ca_di(ric)
                bC = braccio(ric, strati_kl, src, righe_held, righe_train, ca, gguf, a.device, log)
                WS_C = X3.TestaX3(WS) if ca.testa_kl == "bf16x3" else WS
                r = CB.confronto(bR, bC, WS_R, WS_C, ca_r, dev)
                stesso_avanti = stesso_avanti_di(ric)
                h_bit = all(torch.equal(x, y) for x, y in zip(bR["h"], bC["h"]))
                pd, rd = r["dtheta"], esito["rumore"]["dtheta"]
                if stesso_avanti:                  # G1: due indietro sullo stesso avanti
                    fedele = bool(h_bit and r["kl_B_B3"]["media"] == 0.0
                                  and pd["quota_rotti"] <= a.soglia_dtheta and pd["err_rel_norma"] <= a.soglia_dtheta_norma)
                else:
                    f = a.fattore_rumore
                    fedele = bool(r["kl_B_B3"]["media"] <= a.soglia_kl
                                  and pd["quota_rotti"] <= max(a.soglia_dtheta, f * rd["quota_rotti"])
                                  and pd["quota_rotti_pesata"] <= max(a.soglia_dtheta, f * rd["quota_rotti_pesata"])
                                  and pd["err_rel_norma"] <= max(a.soglia_dtheta_norma, f * rd["err_rel_norma"]))
                esito["fedelta"][ric] = {**{k: v for k, v in r.items()}, "stesso_avanti": stesso_avanti,
                                         "h_al_bit": h_bit, "fedele": fedele, "secondi": bC["secondi"],
                                         "trabocco_fp16": bC["trabocco_fp16"]}
                log(f"fedelta' {ric}: KL media {r['kl_B_B3']['media']:.3e}, h al bit {h_bit}, dtheta quota "
                    f"{pd['quota_rotti']:.2e} pesata {pd['quota_rotti_pesata']:.2e} norma {pd['err_rel_norma']:.2e} "
                    f"-> fedele {fedele}")
                del bC
                A._scrivi_json(dest, esito)
            del bR, WS
    verd = {}
    for ric in candidate:
        fed = esito.get("fedelta", {}).get(ric, {}).get("fedele")
        t = esito.get("tempo", {}).get(ric)
        veloce = None if t is None else all(v["rapporto_su_rif"] < 1.0 for v in t.values())
        verd[ric] = {"fedele": fed, "piu_veloce": veloce, "ammessa": bool(fed and veloce)}
    esito["verdetto"] = verd
    esito["via_bf16x3"] = X3.stato()
    A._scrivi_json(dest, esito)
    log(f"VERDETTO cancello indietro: {json.dumps(verd)} -> {dest}")
    return 0 if any(v["ammessa"] for v in verd.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

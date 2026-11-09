#!/usr/bin/env python3
"""Prove di `--densi-ternari` (28/9/2026, ramo densi-ternari): i tensori DENSI non esperti gia' ternari nel GGUF
(variante "A" di @@ROOT:depot@@/tutto_ternario: attn_q/k/v/output, attn_qkv, attn_gate, ssm_out, ffn_gate/up_shexp in
TQ1_0R_1024, ffn_down_shexp in TQ1_0) allenati come esperto UNICO (E = 1) dalla macchina degli esperti. SOLO CPU
(la GPU e' di EB3): motore torch, kernel Triton nell'interprete (TRITON_INTERPRET=1).

Il GGUF A vero non c'e' ancora: le prove costruiscono un GGUF GIOCATTOLO con i tensori VERI degli strati 0 (GDN) e
3 (attenzione piena) di Thor-T e i dieci densi RI-QUANTIZZATI ternari dal loro valore (TQ1_0R_1024 col codec di casa,
TQ1_0 a scala LS per blocco da 256, impacchettato col codec: byte canonici come llama.cpp). Variante "B" ridotta:
strato 0 + token_embd/output TQ1_0R_1024 sulle prime 8192 righe del vocabolario.

Ogni prova ha il suo CONTROLLO NEGATIVO (una mutazione iniettata che deve far diventare rossa la verifica):
  A  default spento == oggi AL BIT: studente del ramo contro quello di `integrata` b6887e0 (albero da `git archive`,
     processo figlio) su Thor-T VERO (strati 0 e 3) e sul GGUF A giocattolo (densi dequantizzati come oggi): chiavi,
     avanti, theta.grad, scale per l'export. Negativo: una theta spostata di 1e-3 / i densi accesi -> diverso.
     + la configurazione: --densi-ternari analizzato, firma dello studente solo se acceso, nome sconosciuto -> errore.
  B  theta.grad dei densi == gradiente MATERIALIZZATO: studente spento (densi = nn.Linear dequantizzati, dW per
     autograd) -> dW nell'ordine del file (permutazioni) -> ruotato (TQ1_0R_1024) -> dL/dd = somma per gruppo di
     trit (.) dW_rot -> x d0; B64 (tolleranza 1e-6) e B32 (1e-4). Negativo: senza permutazione / senza rotazione ->
     errore grande. B3: kernel Triton (triton e triton2, interprete) a esperto unico == motore torch (avanti, dX, dL/dd,
     modi V 1/2); negativo: una scala spostata di 1 ulp.
  C  passo V sui densi (accetta guadagno, z_guadagno 1 come EB3): salti SOLO dove il criterio li sceglie (selezione
     ricalcolata a parte, insieme identico), byte cambiati solo quelli del registro, campi d intatti, theta.grad col
     motore V == senza al bit, annullamento al bit (byte e uscita). Negativi: un salto in piu' nel registro -> il
     controllo lo vede; registro manomesso -> l'annullamento rifiuta senza scrivere. C2: V-Bop sui densi, annullamento
     al bit anche dello stato V-Bop.
  D  export -> ricarica AL BIT: istantanea (scale dei densi), `e2e_esporta.esporta` (campi d, file intero, codice C
     sull'intero tensore 2D, ricarica con uscita al bit), `e2e_pv_codici.esporta_codici` + `ricarica_codici` (registri
     con f >= 3), studente dal file finale: byte e scale == quelli allenati al bit per TQ1_0R_1024 2D e TQ1_0 2D, uscita
     al bit. Negativi: un byte di codice manomesso -> verifica_patch_codici solleva; una scala dell'istantanea spostata
     di 1 ulp -> controlla_scale solleva.
  E  caricamento: GGUF A spento -> peso di ogni Linear denso == trit x d (de-ruotato, permutato) al bit; acceso vs
     spento in B64 (uscita entro 1e-6); negativo: permutazione tolta nella LineareTernaria -> uscita lontana. GGUF B:
     embedding (righe e tabella intera) e testa == codec al bit, dequantizzazione a pezzi == intera al bit; negativo:
     riferimento senza de-rotazione -> diverso. Guardia: --densi-ternari su un k-quant (Thor-T vero) -> ValueError.

Uso (un processo pesante alla volta; mai la GPU):
    XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus systemd-run --user --collect \\
      --wait --pipe -p CPUQuota=400% -p Nice=19 -p MemoryHigh=10G -p MemoryMax=14G ... --setenv=OMP_NUM_THREADS=4 \\
      --setenv=HIP_VISIBLE_DEVICES= --setenv=CUDA_VISIBLE_DEVICES= --setenv=ROCR_VISIBLE_DEVICES= \\
      @@ROOT:venv@@/bin/python3 e2e_test_densi_ternari.py [--solo A,B,...]
    esito: $E2E_DENSI_DIR/esito_test_densi_ternari.json (default @@ROOT:depot@@/densi_ternari_prove)
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("TRITON_INTERPRET", "1")
for _v in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
    os.environ[_v] = ""
import numpy as np  # noqa: E402
import torch  # noqa: E402

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import e2e_studente as S  # noqa: E402
import e2e_codici_scale as CS  # noqa: E402
import e2e_campi_scala as CAMPI  # noqa: E402
import e2e_pv_codici as PV  # noqa: E402
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as GT  # noqa: E402

DIR = Path(os.getenv("E2E_DENSI_DIR", "@@ROOT:depot@@/densi_ternari_prove"))
FILI = int(os.getenv("OMP_NUM_THREADS", "4"))
torch.set_num_threads(FILI)
BASE_VECCHIA = "b6887e0"                  # ramo integrata al momento del ramo (EB3 gira su questo codice)
ESITO: dict = {"data": time.strftime("%Y-%m-%d %H:%M:%S"), "gguf_thor": str(S.GGUF_THOR_T), "prove": {}}
MUTO = lambda *_: None  # noqa: E731
TUTTI = tuple(S.FAMIGLIE_DENSE)


def registra(nome: str, ok: bool, **dati) -> bool:
    ESITO["prove"][nome] = {"ok": bool(ok), **dati}
    print(f"[{nome}] {'✅' if ok else '⛔'} {json.dumps(dati, ensure_ascii=False, default=str)[:2000]}", flush=True)
    return ok


def rel(a, b) -> float:
    a, b = a.double(), b.double()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-300))


def libera():
    gc.collect()


# ═══════════════════════════════════ GGUF giocattolo (variante A / B) ═══════════════════════════════════
def _tq1_0_ls(W: np.ndarray) -> np.ndarray:
    """(R, C) float32 -> byte TQ1_0 (R, C/256*54): scala per blocco da 256 ai minimi quadrati (griglia del codec di
    casa), trit = lroundf(w/d) nell'intervallo, impacchettati col codec (_pack_quarter = schema di block_tq1_0)."""
    import tq1_0r_1024_codec as COD
    R, C = W.shape
    G = np.ascontiguousarray(W, dtype=np.float32).reshape(-1, 256)
    N = G.shape[0]
    ab = np.abs(G).astype(np.float64)
    media = ab.mean(-1)
    best_s = np.full(N, -1.0)
    best_d = np.zeros(N)
    for m in COD.GRIGLIA_LS:
        vivi = ab > (m * media)[:, None]
        k = vivi.sum(-1).astype(np.float64)
        somma = np.where(vivi, ab, 0.0).sum(-1)
        sc = np.where(k > 0, somma * somma / np.maximum(k, 1.0), 0.0)
        ok = sc > best_s
        best_s = np.where(ok, sc, best_s)
        best_d = np.where(ok, np.where(k > 0, somma / np.maximum(k, 1.0), 0.0), best_d)
    d16 = best_d.astype(np.float32).astype(np.float16)
    dd = d16.astype(np.float32)
    idd = np.where(dd != 0, np.float32(1.0) / np.where(dd != 0, dd, np.float32(1.0)), np.float32(0.0))
    T = np.clip(COD._lroundf((G * idd[:, None]).astype(np.float32)) + 1, 0, 2)
    qs, qh = COD._pack_quarter(T)
    out = np.zeros((N, 54), dtype=np.uint8)
    out[:, 0:48], out[:, 48:52] = qs, qh
    out[:, 52:54] = d16.view(np.uint8).reshape(N, 2)
    from gguf.quants import TQ1_0 as GQ
    assert np.array_equal(GQ.dequantize_blocks(out), (T.astype(np.float32) - 1) * dd[:, None]), "TQ1_0 non canonico"
    return out.reshape(R, (C // 256) * 54)


def _ternarizza(W: np.ndarray):
    import tq1_0r_1024_codec as COD
    R, C = W.shape
    if C % 1024 == 0:
        return COD.quantize_rows(np.ascontiguousarray(W, dtype=np.float32)).reshape(R, (C // 1024) * 224), \
            GT.TQ1_0R_1024
    assert C % 256 == 0, (R, C)
    return _tq1_0_ls(W), GT.TQ1_0


def gguf_giocattolo(p: Path, strati=(0, 3), densi=TUTTI, testa_r1024: int = 0) -> Path:
    """GGUF «variante A» giocattolo: tensori VERI (byte identici) degli strati `strati` di Thor-T, coi densi `densi`
    ri-quantizzati ternari dal loro valore dequantizzato; output_norm vero. `testa_r1024` > 0 (variante «B» ridotta):
    token_embd e output TQ1_0R_1024 sulle prime `testa_r1024` righe. Si costruisce UNA volta (sidecar json)."""
    from dequant_gguf_torch import dequant_np, _dequant_raw
    import tq1_0r_1024_codec as COD
    firma = {"strati": list(strati), "densi": list(densi), "testa_r1024": testa_r1024, "sorgente": str(S.GGUF_THOR_T),
             "versione": 1}
    js = p.with_name(p.name + ".json")
    if p.exists() and js.exists() and json.loads(js.read_text()).get("firma") == firma:
        return p
    t0 = time.time()
    r = GGUFReader(str(S.GGUF_THOR_T))
    tmp = p.with_name(p.name + ".parziale")
    w = GGUFWriter(str(tmp), arch="qwen35moe")
    tipi = {}
    for t in r.tensors:
        nome = t.name
        in_strato = any(nome.startswith(f"blk.{L}.") for L in strati)
        testa = nome in ("output.weight", "token_embd.weight")
        if not (in_strato or nome == "output_norm.weight" or (testa and testa_r1024)):
            continue
        fam = nome.split(".", 2)[2][:-len(".weight")] if in_strato and nome.endswith(".weight") else None
        if in_strato and fam in densi:
            dati, qt = _ternarizza(dequant_np(t))
        elif testa:
            Wt = _dequant_raw(np.asarray(t.data)[:testa_r1024], t)
            dati, qt = COD.quantize_rows(Wt).reshape(testa_r1024, -1), GT.TQ1_0R_1024
        else:
            dati = np.ascontiguousarray(np.asarray(t.data))
            if dati.dtype != np.uint8:
                dati = dati.view(np.uint8)
            qt = GT[t.tensor_type.name]
        w.add_tensor(nome, dati, raw_dtype=qt)
        tipi[nome] = qt.name
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    os.replace(tmp, p)
    js.write_text(json.dumps({"firma": firma, "tipi": tipi, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                              "secondi": round(time.time() - t0, 1)}, indent=1))
    return p


def gguf_A() -> Path:
    return gguf_giocattolo(DIR / "giocattolo_A_0_3.gguf")


def gguf_B() -> Path:
    return gguf_giocattolo(DIR / "giocattolo_B_0.gguf", strati=(0,), testa_r1024=8192)


def voce_densa(st, L: int, n: str) -> dict:
    return S.voci_dense(st._tens, L, st.cfg.layer_types[L] == "full_attention", (n,))[n][1]


def lineare(st, L: int, n: str):
    padre, nome = S._padre_lineare(st.moduli[L], voce_densa(st, L, n)["hf"])
    return getattr(padre, nome)


def trit_fam(f: S.DatiFamiglia) -> torch.Tensor:
    return CAMPI.trit_torch(f.blocchi.reshape(-1, f.nb), f.tipo).view(f.E, f.R, f.C)


def dense_presenti(st, L: int) -> list:
    return [n for n in st.esperti(L).fam if n in S.FAMIGLIE_DENSE]


# ═══════════════════════════════════ A: default spento == oggi al bit ═══════════════════════════════════
CALCOLO_A = r'''
def calcola(S, gguf, densi=(), sposta=None):
    import torch
    kw = {"densi_ternari": densi} if densi else {}
    st = S.Studente(gguf=gguf, strati=[0, 3], ricetta="B_ind32", motore_nome="torch", device="cpu",
                    esperti=slice(0, 64), log=lambda *_: None, **kw)
    with torch.no_grad():
        for (L, n), p in zip(st.chiavi(), st.parametri()):
            g = torch.Generator().manual_seed(1000 * L + sum(ord(c) for c in n))
            p.copy_(0.01 * torch.randn(p.shape, generator=g))
        if sposta is not None:
            st.parametri()[sposta].view(-1)[0] += 1e-3
    g = torch.Generator().manual_seed(2809)
    T = 48
    x0 = torch.randn(1, T, 2048, generator=g) * 0.5
    G = torch.randn(T, 2048, generator=g)
    ids = torch.zeros(1, T, dtype=torch.long)
    with torch.no_grad():
        xL = st.avanti(ids, x0=x0)["xL"].clone()
    val = st.gradiente(ids, lambda h: (h * G).sum() / T, x0=x0)
    return {"chiavi": [list(k) for k in st.chiavi()], "xL": xL, "val": val,
            "grad": [p.grad.detach().clone() for p in st.parametri()],
            "scale": {k: torch.from_numpy(v.astype("float32")) for k, v in st.scale_per_export().items()}}
'''

FIGLIO_A = r'''
import os, sys, torch
albero, gguf, uscita = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, albero)
torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))
import e2e_studente as S
assert os.path.dirname(os.path.abspath(S.__file__)) == albero, S.__file__
import dequant_gguf_torch as DQ
assert os.path.dirname(os.path.abspath(DQ.__file__)) == albero, DQ.__file__
ns = {}
exec(open(sys.argv[4]).read(), ns)
torch.save(ns["calcola"](S, gguf), uscita)
'''


def _albero_vecchio() -> Path:
    d = DIR / f"vecchio_{BASE_VECCHIA}"
    if not (d / "thor_v2" / "e2e_studente.py").exists():
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        tar = subprocess.run(["git", "-C", str(QUI.parent), "archive", BASE_VECCHIA, "thor_v2"], capture_output=True,
                             check=True).stdout
        subprocess.run(["tar", "-x", "-C", str(d)], input=tar, check=True)
        shutil.copyfile(QUI / "mappa_completa_gguf_hf_perm.npz", d / "thor_v2" / "mappa_completa_gguf_hf_perm.npz")
    return d / "thor_v2"


def _uguali(a: dict, b: dict) -> dict:
    return {"chiavi": a["chiavi"] == b["chiavi"],
            "avanti": bool(torch.equal(a["xL"], b["xL"])),
            "perdita": a["val"] == b["val"],
            "theta_grad": len(a["grad"]) == len(b["grad"]) and all(torch.equal(x, y) for x, y in zip(a["grad"], b["grad"])),
            "scale": set(a["scale"]) == set(b["scale"]) and all(torch.equal(a["scale"][k], b["scale"][k])
                                                                for k in a["scale"])}


def prova_a() -> bool:
    t0 = time.time()
    albero = _albero_vecchio()
    calc_p = DIR / "calcolo_a.py"
    calc_p.write_text(CALCOLO_A)
    figlio_p = DIR / "figlio_a.py"
    figlio_p.write_text(FIGLIO_A)
    ns: dict = {}
    exec(CALCOLO_A, ns)
    esiti, ok = {}, True
    for nome, gguf in (("thor_T_vero", S.GGUF_THOR_T), ("giocattolo_A_spento", gguf_A())):
        out_v = DIR / f"a_vecchio_{nome}.pt"
        env = dict(os.environ, TRITON_INTERPRET="1", HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="",
                   ROCR_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
        subprocess.run([sys.executable, str(figlio_p), str(albero), str(gguf), str(out_v), str(calc_p)], check=True,
                       env=env)
        vecchio = torch.load(out_v, weights_only=False)
        nuovo = ns["calcola"](S, gguf)
        u = _uguali(vecchio, nuovo)
        ok &= all(u.values())
        # controllo negativo 1: una theta spostata di 1e-3 -> il confronto DEVE vedere la differenza
        neg = _uguali(vecchio, ns["calcola"](S, gguf, sposta=0))
        ok &= not neg["avanti"] and not neg["theta_grad"]
        esiti[nome] = {"uguale_al_bit": u, "negativo_theta_spostata": neg, "chiavi": len(nuovo["chiavi"])}
        if nome == "giocattolo_A_spento":
            # controllo negativo 2: gli stessi calcoli coi densi ACCESI non sono quelli di oggi (avanti fp16 ruotato,
            # chiavi in piu')
            acc = _uguali(vecchio, ns["calcola"](S, gguf, densi=TUTTI))
            ok &= not acc["avanti"] and not acc["chiavi"]
            esiti[nome]["negativo_densi_accesi"] = acc
        del vecchio, nuovo
        libera()
    # la configurazione (e2e_allena): analisi, firma solo se acceso, nome sconosciuto -> errore
    import e2e_allena as A
    cfg = {"default_vuoto": A.ConfigAllena().densi_ternari == () and A._CONFIG_DEFAULT["densi_ternari"] == [],
           "gruppi": A._famiglie_dense("shexp,attn_q,gdn") == ("attn_q", "attn_qkv", "attn_gate", "ssm_out",
                                                               "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp"),
           "vuoto": A._famiglie_dense("") == ()}
    try:
        A._famiglie_dense("attn_qq")
        cfg["sconosciuta_rifiutata"] = False
    except ValueError:
        cfg["sconosciuta_rifiutata"] = True
    s_off = S.Studente(gguf=gguf_A(), strati=[0], ricetta="B", motore_nome="torch", device="cpu",
                       esperti=slice(0, 8), log=MUTO)
    s_on = S.Studente(gguf=gguf_A(), strati=[0], ricetta="B", motore_nome="torch", device="cpu",
                      esperti=slice(0, 8), log=MUTO, densi_ternari="gdn,shexp")
    f_off, f_on = A._firma_studente(s_off), A._firma_studente(s_on)
    cfg["firma_spento_di_sempre"] = "densi_ternari" not in f_off
    cfg["firma_acceso"] = f_on.get("densi_ternari") == list(S.famiglie_dense("gdn,shexp"))
    cfg["chiavi_acceso_dopo_esperti"] = s_on.chiavi()[:3] == s_off.chiavi() and \
        [n for _, n in s_on.chiavi()[3:]] == ["attn_qkv", "attn_gate", "ssm_out", "ffn_gate_shexp", "ffn_up_shexp",
                                               "ffn_down_shexp"]
    del s_off, s_on
    ok &= all(cfg.values())
    return registra("A_spento_identico_al_bit", ok, base=BASE_VECCHIA, casi=esiti, configurazione=cfg,
                    secondi=round(time.time() - t0, 1))


# ═══════════════════════════════════ B: gradiente di theta dei densi ═══════════════════════════════════
def _grad_materializzato(st_off, st_on, L: int, n: str, perm: bool = True, rotazione: bool = True) -> torch.Tensor:
    """dL/dtheta di riferimento dal dW dell'nn.Linear dello studente SPENTO: dW_file (inversa della relazione
    file->HF), dW_rot = R dW_file (TQ1_0R_1024), dL/dd = somma per gruppo di trit (.) dW_rot, x d0 (theta = 0)."""
    import e2e_lora as LR
    lin = lineare(st_off, L, n)
    dW = lin.weight.grad.detach().to(torch.float64)
    voce = voce_densa(st_off, L, n)
    if perm:
        dW = LR.hf_a_file(dW, voce)
    f = st_on.esperti(L).fam[n]
    if f.ruotata and rotazione:
        dW = S.ruota(dW)
    q = trit_fam(f)[0].to(torch.float64)
    dd = (q * dW).view(f.R, f.NG, f.gs).sum(-1)
    return (dd * f.d0[0].to(torch.float64)).unsqueeze(0)


def _prova_grad(ricetta: str, soglia: float) -> dict:
    gguf = gguf_A()
    kw = dict(gguf=gguf, strati=[0, 3], ricetta=ricetta, motore_nome="torch", device="cpu", esperti=slice(0, 16),
              log=MUTO)
    st_on = S.Studente(**kw, densi_ternari=TUTTI)
    st_off = S.Studente(**kw)
    dt = st_on.ric.dtype_residuo
    for L in (0, 3):
        for n in dense_presenti(st_on, L):
            lineare(st_off, L, n).weight.requires_grad_(True)
    g = torch.Generator().manual_seed(2810)
    T = 32
    x0 = (torch.randn(1, T, 2048, generator=g) * 0.5).to(dt)
    G = torch.randn(T, 2048, generator=g).to(dt)
    ids = torch.zeros(1, T, dtype=torch.long)
    st_on.gradiente(ids, lambda h: (h * G).sum() / T, x0=x0)
    with torch.enable_grad():
        ((st_off.nascosto(ids, x0=x0) * G).sum() / T).backward()
    per, peggiore = {}, 0.0
    neg_perm, neg_rot = {}, {}
    for L in (0, 3):
        for n in dense_presenti(st_on, L):
            gon = st_on.esperti(L).theta[n].grad.to(torch.float64)
            rif = _grad_materializzato(st_off, st_on, L, n)
            e = rel(gon, rif)
            peggiore = max(peggiore, e)
            per[f"{L}.{n}"] = {"tipo": st_on.esperti(L).fam[n].tipo, "rel": e,
                               "norma": float(rif.norm())}
            if voce_densa(st_off, L, n)["rel"] != "identita":
                neg_perm[f"{L}.{n}"] = rel(gon, _grad_materializzato(st_off, st_on, L, n, perm=False))
            if st_on.esperti(L).fam[n].ruotata and n in ("attn_q", "attn_qkv", "ffn_gate_shexp"):
                neg_rot[f"{L}.{n}"] = rel(gon, _grad_materializzato(st_off, st_on, L, n, rotazione=False))
    theta_esperti_uguali = all(rel(st_on.esperti(L).theta[m].grad, st_off.esperti(L).theta[m].grad) < soglia
                               for L in (0, 3) for m in S.FAMIGLIE)
    del st_on, st_off
    libera()
    return {"rel_massimo": peggiore, "soglia": soglia, "per_famiglia": per,
            "negativo_senza_permutazione": neg_perm, "negativo_senza_rotazione": neg_rot,
            "theta_esperti_come_spento": theta_esperti_uguali,
            "ok": peggiore < soglia and theta_esperti_uguali and len(per) == 13
            and all(v > 1e-2 for v in neg_perm.values()) and len(neg_perm) == 3
            and all(v > 1e-2 for v in neg_rot.values()) and len(neg_rot) == 4}


def prova_b() -> bool:
    t0 = time.time()
    b64 = _prova_grad("B64", 1e-6)
    b32 = _prova_grad("B32", 1e-4)
    return registra("B_theta_grad_densi_materializzato", b64["ok"] and b32["ok"], B64=b64, B32=b32,
                    secondi=round(time.time() - t0, 1))


def prova_b3() -> bool:
    """Kernel Triton (interprete) a esperto unico == motore torch: famiglie dense VERE del GGUF A (righe 0-63)."""
    t0 = time.time()
    r = GGUFReader(str(gguf_A()))
    tens = {t.name: t for t in r.tensors}
    ric = S.RICETTE["B_ind32"]
    esiti, ok = {}, True
    for nome_t, n in (("blk.0.attn_gate.weight", "attn_gate"), ("blk.0.ffn_down_shexp.weight", "ffn_down_shexp")):
        f = S.famiglia_da_tensore(tens[nome_t], n, righe=slice(0, 64))
        g = torch.Generator().manual_seed(33)
        T = 64
        piano = S.Piano(torch.zeros(T, 1, dtype=torch.long), 1)
        x = (torch.randn(T, f.C, generator=g) * 0.3).to(torch.float16)    # avanti: ingresso fp16 (ricetta B)
        x32 = x.float()                                                    # indietro B_ind32: ingressi fp32
        gy = (torch.randn(T, f.R, generator=g) * 0.1).float()
        d = f.d0.to(torch.float16).to(torch.float32)
        mt = S.MotoreTorch()
        rif = (mt.avanti(x, d, f, piano, ric), mt.indietro_x(gy, d, f, piano, ric), mt.dd(gy, x32, f, piano, torch.float32))
        e = {"E": f.E, "R": f.R, "C": f.C, "tipo": f.tipo}
        for nm, mot in (("triton", S.motore("triton")), ("triton2", S.motore("triton2"))):
            y = mot.avanti(x, d, f, piano, ric)
            dx = mot.indietro_x(gy, d, f, piano, ric)
            dd = mot.dd(gy, x32, f, piano, torch.float32)
            e[nm] = {"avanti": rel(y, rif[0]), "dx": rel(dx, rif[1]), "dd": rel(dd, rif[2])}
            ok &= max(e[nm].values()) < 1e-5
        # modi V 1 (proposta) e 2 (verifica) del kernel _k_dd_v a esperto unico == riferimento materializzato
        bk, bt = PV.BufferV.crea(f), PV.BufferV.crea(f)
        for modo in (1, 2):
            dk = PV.dd_v_triton(gy, x32, f, piano, PV.BLOCCHI_V, bk, modo)
            dtr = PV.dd_v_torch(gy, x32, f, piano, torch.float32, bt, modo)
            e[f"v_modo{modo}_dd"] = rel(dk, dtr)
        e["v_cand_uguali"] = bool(torch.equal(bk.cand, bt.cand))
        e["v_prop"] = rel(bk.prop, bt.prop)
        e["v_S"] = rel(bk.S, bt.S)
        ok &= e["v_cand_uguali"] and max(e["v_modo1_dd"], e["v_modo2_dd"], e["v_prop"], e["v_S"]) < 1e-5
        # controllo negativo: UNA scala spostata di 1 ulp fp16 (un gruppo di una riga) -> la differenza si vede
        d2 = d.clone()
        v16 = np.float16(float(d2.view(-1)[5]))
        d2.view(-1)[5] = float(np.nextafter(v16, np.float16(np.inf)))
        e["negativo_1ulp"] = rel(S.motore("triton2").avanti(x, d2, f, piano, ric), rif[0])
        ok &= e["negativo_1ulp"] > 1e-5
        esiti[n] = e
    del r, tens
    return registra("B3_kernel_triton_esperto_unico", ok, famiglie=esiti, secondi=round(time.time() - t0, 1))


# ═══════════════════════════════════ C: passo V sui densi ═══════════════════════════════════
def _studente_v(gguf, bop=None):
    st = S.Studente(gguf=gguf, strati=[0, 3], ricetta="B_ind32", motore_nome="torch", device="cpu",
                    esperti=slice(0, 16), log=MUTO, densi_ternari=TUTTI)
    for L in (0, 3):
        for f in st.esperti(L).fam.values():
            PV.famiglia_scrivibile(f)
    return st


def _raccolta(st, pv, dati):
    """16 micro-lotti di perdita quadratica (bersaglio fisso: segnale coerente), 2 di curvatura (vettore di Rademacher
    x sqrt(2/N): covarianza = Hessiana di Gauss-Newton della perdita quadratica rispetto a h)."""
    x0s, bers, vs, T = dati
    ids = torch.zeros(1, T, dtype=torch.long)
    for p in st.parametri():
        p.grad = None
    if pv is not None:
        pv.inizia_passo()
    for k in range(16):
        if pv is not None:
            pv.microlotto(k)
        st.gradiente(ids, lambda h: ((h - bers) ** 2).sum() / (16 * T), x0=x0s[k])
    if pv is not None:
        for k in range(2):
            pv.curvatura()
            st.gradiente(ids, lambda h: (h * vs[k]).sum(), x0=x0s[k])
        pv.fine_raccolta()
    return [None if p.grad is None else p.grad.clone() for p in st.parametri()]


def _dati_c(st):
    g = torch.Generator().manual_seed(2811)
    T = 24
    base = torch.randn(1, T, 2048, generator=g) * 0.5
    x0s = [base + 0.005 * torch.randn(1, T, 2048, generator=g) for _ in range(16)]
    with torch.no_grad():
        h0 = st.nascosto(torch.zeros(1, T, dtype=torch.long), x0=base)
    bers = h0 + 1.0 * torch.randn(h0.shape, generator=g) * h0.abs().mean()
    vs = [(torch.randint(0, 2, h0.shape, generator=g).float() * 2 - 1) * math.sqrt(2.0 / (16 * T)) for _ in range(2)]
    return x0s, bers, vs, T


def selezione_indipendente(buf, f, d, n_ver: int, n_perd: int, n_curv: int, rho: float, zg_min: float,
                           trit0: torch.Tensor) -> set:
    """Ricalcolo A PARTE (numpy float64, esperto unico) del criterio `--pv-accetta guadagno`: testati (candidato e
    d != 0), scoperte (S nella direzione > 0, z > 0), convenienti (guadagno di 2o ordine > 0 e t del guadagno >
    zg_min), budget rho^2 sum d^2 nnz sull'intera matrice (nnz dai trit PRIMA dei salti, `trit0` (R, C)), prefisso in
    ordine di guadagno. -> {(r, g, colonna, dir)}."""
    prop = buf.prop[0].double().numpy()
    Sv, Qv, Hv = buf.S[0].double().numpy(), buf.Q[0].double().numpy(), buf.H[0].double().numpy()
    dv = d[0].double().numpy()
    n = float(n_ver)
    sgn = np.sign(prop)
    Sd = Sv * sgn
    var = np.clip((Qv - Sv * Sv / n) / (n - 1), 0.0, None)
    den = np.sqrt(n * var)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(den > 0, Sd / np.where(den > 0, den, 1.0), np.where(Sd > 0, np.inf, np.where(Sd < 0, -np.inf, 0.0)))
        xg = Sd - 0.5 * (n / n_curv) * Hv * np.abs(dv)
        tg = np.where(den > 0, xg / np.where(den > 0, den, 1.0), np.where(xg > 0, np.inf, -np.inf))
    gain = (n_perd / n) * Sd * np.abs(dv) - 0.5 * (n_perd / n_curv) * Hv * dv * dv
    testati = (prop != 0) & (dv != 0)
    conv = testati & (Sd > 0) & (z > 0) & (gain > 0) & (tg > zg_min)
    nnz = (trit0.numpy() != 0).reshape(f.R, f.NG, f.gs).sum(-1)
    budget = rho * rho * float((dv * dv * nnz).sum())
    gm = np.where(conv, gain, -np.inf).reshape(-1)
    ordine = np.argsort(-gm, kind="stable")
    costo = (dv * dv).reshape(-1)[ordine]
    fin = np.isfinite(gm[ordine])
    cum = np.cumsum(np.where(fin, costo, 0.0))
    tieni = ordine[fin & (cum <= budget)]
    cand = buf.cand[0].numpy().reshape(-1)
    out = set()
    for i in tieni.tolist():
        r_, g_ = divmod(i, f.NG)
        out.add((r_, g_, g_ * f.gs + int(cand[i]), int(-np.sign(prop.reshape(-1)[i] * dv.reshape(-1)[i]))))
    return out


def salti_ammessi(reg: PV.Registro, fi: int, attesi: set) -> bool:
    """I salti del registro per la famiglia `fi` sono ESATTAMENTE quelli del ricalcolo (riga, colonna, verso), e = 0."""
    righe = reg.righe[reg.righe["f"] == fi]
    fatti = {(int(r["r"]), int(r["c"]), int(r["t_nuovo"]) - int(r["t_vecchio"])) for r in righe}
    att = {(r_, c_, dz) for (r_, _g, c_, dz) in attesi}
    return fatti == att and len(fatti) == len(righe) and all(int(e) == 0 for e in righe["e"])


def prova_c() -> bool:
    t0 = time.time()
    gguf = gguf_A()
    st0 = _studente_v(gguf)
    dati = _dati_c(st0)
    g_senza = _raccolta(st0, None, dati)
    del st0
    libera()
    st = _studente_v(gguf)
    stato = PV.StatoV()
    fams = PV.installa(st, stato, motore="torch")
    cfg = PV.ConfigV(accetta="guadagno", z_guadagno_min=1.0)
    pv = PV.PassoV(fams, cfg, stato)
    g_con = _raccolta(st, pv, dati)
    grad_al_bit = len(g_senza) == len(g_con) and all(
        (a is None and b is None) or (a is not None and b is not None and torch.equal(a, b)) for a, b in zip(g_senza, g_con))
    densi = [k for k in fams if k[1] in S.FAMIGLIE_DENSE]
    prima_b = {k: fams[k].blocchi.clone() for k in fams}
    buf_copia = {k: PV.BufferV(*(getattr(stato.buffer(fams[k]), c).clone() for c in ("cand", "prop", "S", "Q", "H")))
                 for k in densi}
    x_t = dati[0][0]
    ids = torch.zeros(1, dati[3], dtype=torch.long)
    with torch.no_grad():
        y0 = st.avanti(ids, x0=x_t)["xL"].clone()
    scale = {k: st.esperti(k[0]).scala(k[1]).detach() for k in fams}
    rho = 0.05
    reg, diag = pv.seleziona_e_salta(scale, rho)
    per, ok_sel, n_salti_densi = {}, True, 0
    for k in densi:
        f = fams[k]
        fi = PV.FAMIGLIE_IDX[k[1]]
        T0 = CAMPI.trit_torch(prima_b[k].reshape(-1, f.nb), f.tipo).view(f.R, f.C)
        attesi = selezione_indipendente(buf_copia[k], f, scale[k], stato.n_verifica, stato.n_perdita,
                                        stato.n_curvatura, rho, cfg.z_guadagno_min, T0)
        righe = reg.righe[(reg.righe["L"] == k[0]) & (reg.righe["f"] == fi)]
        n_salti_densi += len(righe)
        uguale = salti_ammessi(PV.Registro(righe), fi, attesi)
        # byte: cambiati SOLO quelli del registro; trit: cambiati SOLO (r, c) del registro, come dice il registro;
        # campi d intatti
        dopo = f.blocchi
        diversi = torch.nonzero((dopo != prima_b[k]).reshape(-1)).reshape(-1)
        c_ = torch.from_numpy(righe["c"].astype(np.int64))
        blk, byte, _ = PV.posizione_t(c_, f.tipo)
        permessi = set(((torch.from_numpy(righe["r"].astype(np.int64)) * f.NBR) + blk * f.nb + byte).tolist())
        byte_ok = set(diversi.tolist()) <= permessi
        Tp = CAMPI.trit_torch(prima_b[k].reshape(-1, f.nb), f.tipo).view(f.R, f.C)
        Td = CAMPI.trit_torch(dopo.reshape(-1, f.nb), f.tipo).view(f.R, f.C)
        atteso = Tp.clone()
        if len(righe):
            atteso[torch.from_numpy(righe["r"].astype(np.int64)), c_] = torch.from_numpy(righe["t_nuovo"].astype(np.int8))
        trit_ok = bool(torch.equal(Td, atteso))
        _, _, off, kk = CS.LAYOUT[f.tipo]
        vd, vp = dopo.reshape(-1, f.nb), prima_b[k].reshape(-1, f.nb)
        d_ok = bool(torch.equal(vd[:, off:off + 2 * kk], vp[:, off:off + 2 * kk]))
        per[f"{k[0]}.{k[1]}"] = {"tipo": f.tipo, "salti": len(righe), "attesi": len(attesi), "insieme_uguale": uguale,
                                 "byte_solo_registro": byte_ok, "trit_come_registro": trit_ok, "campi_d_intatti": d_ok}
        ok_sel &= uguale and byte_ok and trit_ok and d_ok
    with torch.no_grad():
        y1 = st.avanti(ids, x0=x_t)["xL"].clone()
    cambia = not torch.equal(y0, y1)
    # controllo negativo 1: un salto IN PIU' (fattibile, in un gruppo non scelto) nel registro -> `salti_ammessi` falso
    k_neg = max(densi, key=lambda k: int(((reg.righe["L"] == k[0]) & (reg.righe["f"] == PV.FAMIGLIE_IDX[k[1]])).sum()))
    fi_neg = PV.FAMIGLIE_IDX[k_neg[1]]
    righe_neg = reg.righe[(reg.righe["L"] == k_neg[0]) & (reg.righe["f"] == fi_neg)].copy()
    presi = {(int(r["r"]), int(r["c"]) // fams[k_neg].gs) for r in righe_neg}
    Tn = trit_fam(fams[k_neg])[0]
    extra = next((r_, g_ * fams[k_neg].gs) for r_ in range(fams[k_neg].R) for g_ in range(fams[k_neg].NG)
                 if (r_, g_) not in presi)
    riga_x = np.zeros(1, dtype=PV.REG_DTYPE)
    tv = int(Tn[extra[0], extra[1]])
    riga_x[0] = (k_neg[0], fi_neg, 0, extra[0], extra[1], tv, tv - 1 if tv > -1 else tv + 1)
    fk = fams[k_neg]
    attesi_neg = selezione_indipendente(buf_copia[k_neg], fk, scale[k_neg], stato.n_verifica, stato.n_perdita,
                                        stato.n_curvatura, rho, cfg.z_guadagno_min,
                                        CAMPI.trit_torch(prima_b[k_neg].reshape(-1, fk.nb), fk.tipo).view(fk.R, fk.C))
    negativo_extra = salti_ammessi(PV.Registro(righe_neg), fi_neg, attesi_neg) and \
        not salti_ammessi(PV.Registro(np.concatenate([righe_neg, riga_x])), fi_neg, attesi_neg)
    # controllo negativo 2: registro manomesso (t_nuovo che non e' quello scritto) di UNA famiglia densa ->
    # l'annullamento rifiuta SENZA scrivere (le guardie di `salta_byte` vengono prima di ogni scrittura)
    rotto = PV.Registro(righe_neg.copy())
    rotto.righe["t_nuovo"][0] = rotto.righe["t_vecchio"][0]
    ora = fams[k_neg].blocchi.clone()
    try:
        PV.annulla(fams, rotto)
        negativo_manomesso = False
    except (RuntimeError, ValueError):
        negativo_manomesso = bool(torch.equal(fams[k_neg].blocchi, ora))
    pv.annulla(reg)
    byte_ritorno = all(torch.equal(fams[k].blocchi, prima_b[k]) for k in fams)
    with torch.no_grad():
        y2 = st.avanti(ids, x0=x_t)["xL"].clone()
    uscita_ritorno = bool(torch.equal(y0, y2))
    ok = grad_al_bit and ok_sel and n_salti_densi > 0 and cambia and byte_ritorno and uscita_ritorno and \
        negativo_extra and negativo_manomesso
    del st
    libera()
    return registra("C_passo_V_densi", ok, theta_grad_motore_V_uguale_al_bit=grad_al_bit, salti_densi=n_salti_densi,
                    salti_totali=len(reg), per_famiglia=per, uscita_cambia=cambia, annulla_byte_al_bit=byte_ritorno,
                    annulla_uscita_al_bit=uscita_ritorno, negativo_salto_in_piu_visto=negativo_extra,
                    negativo_registro_manomesso_rifiutato=negativo_manomesso,
                    n_verifica=stato.n_verifica, n_curvatura=stato.n_curvatura, secondi=round(time.time() - t0, 1))


def prova_c2() -> bool:
    """V-Bop (--pv-modo vbop, come EB3) sui densi: stato V-Bop creato per le famiglie a esperto unico, due passi V,
    salti sui densi, annullamento al bit di byte E stato V-Bop."""
    t0 = time.time()
    gguf = gguf_A()
    st = _studente_v(gguf)
    bop = PV.ConfigBop(gamma=0.5)
    stato = PV.StatoV(bop=bop)
    fams = PV.installa(st, stato, motore="torch")
    pv = PV.PassoV(fams, PV.ConfigV(accetta="guadagno", z_guadagno_min=1.0), stato, bop=bop)
    dati = _dati_c(st)
    densi = [k for k in fams if k[1] in S.FAMIGLIE_DENSE]
    tenuti, ok = 0, True
    for passo in range(2):
        _raccolta(st, pv, dati)
        scale = {k: st.esperti(k[0]).scala(k[1]).detach() for k in fams}
        prima_b = {k: fams[k].blocchi.clone() for k in fams}
        reg, diag = pv.seleziona_e_salta(scale, 0.05)
        n_d = int((reg.righe["f"] >= 3).sum())
        if passo == 1:
            # l'annullamento riporta byte e stato V-Bop dei gruppi saltati a PRIMA della correzione alla ECO
            pv.annulla(reg)
            ok &= all(torch.equal(fams[k].blocchi, prima_b[k]) for k in fams)
        tenuti += n_d
    ok &= tenuti > 0 and all(k in stato.bop_ for k in densi) and \
        any(bool(((stato.bop_[k].eta & PV.ETA_MAX) >= 1).any()) for k in densi)
    # controllo negativo: stesso stato, budget rho = 0 -> NESSUN salto sui densi (la selezione risponde al criterio)
    _raccolta(st, pv, dati)
    scale = {k: st.esperti(k[0]).scala(k[1]).detach() for k in fams}
    reg0, _ = pv.seleziona_e_salta(scale, 0.0)
    nulli = int((reg0.righe["f"] >= 3).sum()) == 0
    ok &= nulli
    del st
    libera()
    return registra("C2_vbop_densi", ok, salti_densi=tenuti, famiglie_dense=len(densi), negativo_rho0_nessun_salto=nulli,
                    byte_bop_densi=sum(stato.bop_[k].byte() for k in densi), secondi=round(time.time() - t0, 1))


# ═══════════════════════════════════ D: export -> ricarica al bit ═══════════════════════════════════
def prova_d() -> bool:
    import e2e_allena as A
    import e2e_esporta as X
    t0 = time.time()
    d = DIR / "d_export"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    src = gguf_A()
    st = S.Studente(gguf=src, strati=[0, 3], ricetta="B", motore_nome="torch", device="cpu", log=MUTO,
                    densi_ternari=TUTTI)
    g = torch.Generator().manual_seed(2812)
    with torch.no_grad():
        for p in st.parametri():
            p.copy_(0.05 * torch.randn(p.shape, generator=g))
    fams = {(L, n): PV.famiglia_scrivibile(f) for L in (0, 3) for n, f in st.esperti(L).fam.items()}
    # registri di due passi sui densi (e sul gate degli esperti), con ripetizioni al secondo passo (come T10)
    regs = []
    for passo in range(2):
        parti = []
        for k in [k for k in fams if k[1] in S.FAMIGLIE_DENSE or k == (0, "gate")]:
            f = fams[k]
            T_ = trit_fam(f)
            m = 60
            e_ = torch.randint(0, f.E, (m,), generator=g)
            r_ = torch.randint(0, f.R, (m,), generator=g)
            c_ = torch.randint(0, f.C, (m,), generator=g)
            if passo == 1:
                prima = regs[0].righe[(regs[0].righe["L"] == k[0]) & (regs[0].righe["f"] == PV.FAMIGLIE_IDX[k[1]])]
                e_ = torch.cat([torch.from_numpy(prima["e"][:10].astype(np.int64)), e_])
                r_ = torch.cat([torch.from_numpy(prima["r"][:10].astype(np.int64)), r_])
                c_ = torch.cat([torch.from_numpy(prima["c"][:10].astype(np.int64)), c_])
            chiave_ = (e_ * f.R + r_) * f.C + c_
            _, pr = np.unique(chiave_.numpy(), return_index=True)
            pr = np.sort(pr)
            e_, r_, c_ = e_[pr], r_[pr], c_[pr]
            t_ = T_[e_, r_, c_].to(torch.int32)
            dr = torch.where(t_ == 0, torch.randint(0, 2, t_.shape, generator=g, dtype=torch.int32) * 2 - 1, -t_)
            z = torch.zeros(0)
            parti.append(PV.applica_salti(f, PV.Selezione(e_, r_, c_ // f.gs, c_ % f.gs, dr.to(torch.int8), z, z, z,
                                                           z, z, z, z)))
        regs.append(PV.Registro.unisci(parti))
    nome = A.salva_istantanea(d, st, 7, {"prova": "densi"})
    ist = A.carica_istantanea(d / nome)
    theta = torch.load(d / nome / "theta.pt", weights_only=True)
    scale_densi = [n for n in ist["scale"] if S.famiglie_dense_da_nomi([n])]
    scalato = d / "scalato.gguf"
    prov = X.esporta(src, scalato, ist["scale"], ist["meta"], theta=theta, strati_ricarica=(0, 3), device="cpu",
                     esperti_ricarica=slice(0, 32), log=MUTO)
    finale = d / "finale.gguf"
    rep = PV.esporta_codici(scalato, finale, regs)
    ric = PV.ricarica_codici(scalato, finale, regs, [0, 3])
    sf = S.Studente(gguf=finale, strati=[0, 3], ricetta="B", motore_nome="torch", device="cpu", log=MUTO,
                    densi_ternari=TUTTI)
    per, ok_bit = {}, True
    from tq1_0r_1024_codec import S1024
    for L in (0, 3):
        for n in dense_presenti(st, L):
            fa, fb = st.esperti(L).fam[n], sf.esperti(L).fam[n]
            # i byte dei CODICI (qs, qh): i campi d del file hanno le scale nuove, quelli in memoria le vecchie (d0)
            nb_, _, off_, k_ = CS.LAYOUT[fa.tipo]
            m_ = torch.ones(nb_, dtype=torch.bool)
            m_[off_:off_ + 2 * k_] = False
            ba, bb = fa.blocchi.reshape(-1, nb_)[:, m_], fb.blocchi.reshape(-1, nb_)[:, m_]
            byte_ok = bool(torch.equal(ba, bb))
            sc_ok = bool(torch.equal(st.esperti(L).scala(n).detach().to(torch.float16), fb.d0.to(torch.float16)))
            Wc = X._deq_c(fb.blocchi[0].numpy(), fb.tipo)
            W = CS.pesi_da_trit_scale(CS.trit_da_blocchi(fa.blocchi.reshape(-1, fa.nb), fa.tipo),
                                      st.esperti(L).scala(n).detach().reshape(-1, CS.LAYOUT[fa.tipo][3]), fa.tipo,
                                      torch.from_numpy(S1024) if fa.ruotata else None).numpy()
            c_ok = bool(np.array_equal(W, Wc))
            per[f"{L}.{n}"] = {"tipo": fa.tipo, "codici_al_bit": byte_ok, "scale_al_bit": sc_ok, "codice_C_al_bit": c_ok,
                               "byte_codici_diversi": int((ba != bb).sum())}
            ok_bit &= byte_ok and sc_ok and c_ok
    tipi = {v["tipo"] for v in per.values()}
    ids = torch.zeros(1, 64, dtype=torch.long)
    x = torch.randn(1, 64, 2048, generator=g) * 0.5
    with torch.no_grad():
        uscita_ok = bool(torch.equal(st.avanti(ids, x0=x)["xL"], sf.avanti(ids, x0=x)["xL"]))
    # controllo negativo 1: un byte di codice di un denso (fuori dal registro) manomesso nel file finale
    rotto = d / "finale_rotto.gguf"
    shutil.copyfile(finale, rotto)
    rr = GGUFReader(str(rotto), mode="r+")
    t = {q.name: q for q in rr.tensors}["blk.0.attn_qkv.weight"]
    pos_reg = set()
    for reg in regs:
        rg = reg.righe[(reg.righe["L"] == 0) & (reg.righe["f"] == PV.FAMIGLIE_IDX["attn_qkv"])]
        blk, byte, _ = PV.posizione_t(torch.from_numpy(rg["c"].astype(np.int64)), "TQ1_0R_1024")
        pos_reg |= set((rg["r"].astype(np.int64) * t.data.shape[1] + blk.numpy() * 224 + byte.numpy()).tolist())
    j = next(i for i in range(0, 4096, 7) if i not in pos_reg and i % 224 < 192)
    t.data.reshape(-1)[j] = (int(t.data.reshape(-1)[j]) + 1) % 243
    rr.data.flush()
    del rr, t
    try:
        PV.verifica_patch_codici(scalato, rotto, regs)
        neg_byte = False
    except RuntimeError:
        neg_byte = True
    # controllo negativo 2: una scala di un denso dell'istantanea spostata di 1 ulp -> controlla_scale solleva
    finta = {k: np.array(v, copy=True) for k, v in ist["scale"].items()}
    n0 = next(n for n in scale_densi if "ffn_down_shexp" in n)
    v16 = finta[n0].reshape(-1)
    v16[3] = np.nextafter(v16[3], np.float16(np.inf))
    try:
        X.controlla_scale(src, scalato, {n0: finta[n0]})
        neg_scala = False
    except RuntimeError:
        neg_scala = True
    ok = ok_bit and uscita_ok and tipi == {"TQ1_0R_1024", "TQ1_0"} and len(per) == 13 and neg_byte and neg_scala \
        and all(v["uscita_al_bit"] and v["byte_codici_uguali"] for v in ric.values()) and \
        all(prov["ricarica"][str(L)]["scale_al_bit"] and prov["ricarica"][str(L)]["uscita_al_bit"] for L in (0, 3)) \
        and len(scale_densi) == 13
    del st, sf
    libera()
    return registra("D_export_ricarica_al_bit", ok, per_famiglia=per, tipi=sorted(tipi), uscita_file_al_bit=uscita_ok,
                    scale_densi_nell_istantanea=len(scale_densi),
                    export_scale={"ricarica": prov["ricarica"], "controllo": prov["controllo_scale"]["totali"]},
                    export_codici={k: v for k, v in rep.items() if k != "byte_scritti"}, ricarica_codici=ric,
                    negativo_byte_manomesso=neg_byte, negativo_scala_1ulp=neg_scala,
                    salti=[len(r) for r in regs], secondi=round(time.time() - t0, 1))


# ═══════════════════════════════════ E: caricamento ═══════════════════════════════════
def prova_e() -> bool:
    import carica_strato_gguf as CSG
    import dequant_gguf_torch as DQ
    from tq1_0r_1024_codec import S1024, dequantize_blocks
    t0 = time.time()
    Sg = torch.from_numpy(S1024)
    # E1: GGUF A, densi SPENTI -> peso di ogni Linear denso == trit x d de-ruotato e riportato all'ordine HF, al bit
    kw = dict(gguf=gguf_A(), strati=[0, 3], ricetta="B64", motore_nome="torch", device="cpu", esperti=slice(0, 16),
              log=MUTO)
    s_off = S.Studente(**kw)
    e1, neg1 = {}, {}
    for L in (0, 3):
        for n in [m for m in S.FAMIGLIE_DENSE if (f"blk.{L}.{m}.weight") in s_off._tens]:
            t = s_off._tens[f"blk.{L}.{n}.weight"]
            tipo = t.tensor_type.name
            nb, _, _, k = CS.LAYOUT[tipo]
            b = torch.from_numpy(np.ascontiguousarray(np.asarray(t.data)).reshape(-1, nb))
            W = CS.pesi_da_trit_scale(CS.trit_da_blocchi(b, tipo), CS.scale_da_blocchi(b, tipo).float(), tipo,
                                      Sg if tipo == "TQ1_0R_1024" else None).numpy().reshape(np.asarray(t.data).shape[0], -1)
            voce = voce_densa(s_off, L, n)
            Wh = torch.from_numpy(np.array(CSG._applica(voce, W), copy=True)).to(torch.float64)
            wm = lineare(s_off, L, n).weight.detach()
            e1[f"{L}.{n}"] = {"tipo": tipo, "al_bit": bool(torch.equal(wm, Wh))}
            if voce["rel"] != "identita":
                neg1[f"{L}.{n}"] = bool(torch.equal(wm, torch.from_numpy(W).to(torch.float64)))
    ok1 = all(v["al_bit"] for v in e1.values()) and len(e1) == 13 and neg1 and not any(neg1.values())
    # E2: acceso contro spento in B64 (stessi pesi a meno del fp32 della dequantizzazione del codec)
    s_on = S.Studente(**kw, densi_ternari=TUTTI)
    g = torch.Generator().manual_seed(2813)
    x0 = (torch.randn(1, 32, 2048, generator=g) * 0.5).to(torch.float64)
    ids = torch.zeros(1, 32, dtype=torch.long)
    with torch.no_grad():
        y_off = s_off.avanti(ids, x0=x0)["xL"]
        y_on = s_on.avanti(ids, x0=x0)["xL"]
        e2 = rel(y_on, y_off)
        # controllo negativo: permutazioni tolte (righe di attn_qkv, colonne di ssm_out) -> uscita lontana
        lq, lo = lineare(s_on, 0, "attn_qkv"), lineare(s_on, 0, "ssm_out")
        iy, px = lq.inv_y, lo.perm_x
        lq.inv_y, lo.perm_x = None, None
        neg2 = rel(s_on.avanti(ids, x0=x0)["xL"], y_off)
        lq.inv_y, lo.perm_x = iy, px
        ritorno = bool(torch.equal(s_on.avanti(ids, x0=x0)["xL"], y_on))
    ok2 = e2 < 1e-6 and neg2 > 1e-2 and ritorno
    tipi_on = {n: s_on.esperti(0).fam[n].tipo for n in dense_presenti(s_on, 0)}
    del s_off, s_on
    libera()
    # E3: GGUF B (token_embd/output TQ1_0R_1024 ridotti): embedding e testa == codec al bit; dequant a pezzi == intera
    sb = S.Studente(gguf=gguf_B(), strati=[0], ricetta="B", motore_nome="torch", device="cpu", esperti=slice(0, 8),
                    log=MUTO)
    te, to_ = sb._tens["token_embd.weight"], sb._tens["output.weight"]
    ids = torch.randint(0, 8192, (2, 40), generator=g)
    rif_e = torch.from_numpy(dequantize_blocks(np.asarray(te.data).reshape(-1, 224)).reshape(8192, 2048))
    righe_ok = bool(torch.equal(sb.embedding(ids), rif_e[ids]))
    sb.carica_embedding()
    tabella_ok = bool(torch.equal(sb.embedding(ids), rif_e[ids])) and bool(torch.equal(sb._emb, rif_e))
    rif_o = torch.from_numpy(dequantize_blocks(np.asarray(to_.data).reshape(-1, 224)).reshape(8192, 2048))
    testa_ok = bool(torch.equal(sb.testa(), rif_o))
    vecchio_pezzo = DQ.PEZZO_R1024
    try:
        DQ.PEZZO_R1024 = 2048 * 1000 + 7                      # pezzi che non dividono le righe
        a_pezzi = DQ.dequant_np(te)
    finally:
        DQ.PEZZO_R1024 = vecchio_pezzo
    pezzi_ok = bool(np.array_equal(a_pezzi, rif_e.numpy()))
    # controllo negativo: riferimento senza de-rotazione (trit x d nel dominio ruotato) -> diverso
    b = torch.from_numpy(np.ascontiguousarray(np.asarray(te.data)).reshape(-1, 224))
    senza = (CS.trit_da_blocchi(b, "TQ1_0R_1024").float().view(-1, 8, 128)
             * CS.scale_da_blocchi(b, "TQ1_0R_1024").float().unsqueeze(-1)).reshape(8192, 2048)
    neg3 = bool(torch.equal(sb.embedding(ids), senza[ids]))
    ok3 = righe_ok and tabella_ok and testa_ok and pezzi_ok and not neg3 and te.tensor_type.name == "TQ1_0R_1024"
    del sb
    libera()
    # E4: guardia di tipo: --densi-ternari su un k-quant (Thor-T vero: attn_qkv Q6_K/Q4_K) -> ValueError
    try:
        S.Studente(gguf=S.GGUF_THOR_T, strati=[0], ricetta="B", motore_nome="torch", device="cpu",
                   esperti=slice(0, 8), log=MUTO, densi_ternari="gdn")
        guardia = False
    except ValueError as ex:
        guardia = "non ternario" in str(ex)
    ok = ok1 and ok2 and ok3 and guardia
    return registra("E_caricamento_non_esperti_ternari", ok, E1_pesi_al_bit=e1, E1_negativo_senza_perm=neg1,
                    E2_acceso_vs_spento_rel=e2, E2_negativo_senza_perm_rel=neg2, E2_ritorno_al_bit=ritorno,
                    tipi_densi_strato_0=tipi_on,
                    E3={"righe": righe_ok, "tabella": tabella_ok, "testa": testa_ok, "a_pezzi": pezzi_ok,
                        "negativo_senza_derotazione_uguale": neg3},
                    E4_guardia_k_quant=guardia, secondi=round(time.time() - t0, 1))


PROVE = {"A": prova_a, "B": prova_b, "B3": prova_b3, "C": prova_c, "C2": prova_c2, "D": prova_d, "E": prova_e}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo", default=",".join(PROVE))
    a = ap.parse_args()
    DIR.mkdir(parents=True, exist_ok=True)
    esito_p = DIR / "esito_test_densi_ternari.json"
    if esito_p.exists():
        try:
            ESITO["prove"].update(json.loads(esito_p.read_text()).get("prove", {}))
        except Exception as ex:  # noqa: BLE001
            print(f"esito precedente illeggibile ({ex}): si riparte", flush=True)
    ok = True
    for nome in a.solo.split(","):
        for k in [k for k in ESITO["prove"] if k.split("_", 1)[0] == nome]:
            del ESITO["prove"][k]
        t0 = time.time()
        try:
            ok &= PROVE[nome]()
        except Exception as ex:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            ok &= registra(f"{nome}_eccezione", False, errore=repr(ex))
        print(f"  ({nome}: {time.time() - t0:.1f} s)", flush=True)
        esito_p.write_text(json.dumps(ESITO, indent=1, default=str))
    tutte = all(v["ok"] for v in ESITO["prove"].values())
    print("ESITO:", "TUTTO VERDE" if (ok and tutte) else "⛔ QUALCOSA NON TORNA", "->", esito_p, flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

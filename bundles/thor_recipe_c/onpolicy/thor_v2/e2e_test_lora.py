#!/usr/bin/env python3
"""Prove di `--lora-nonesperti` (27/9/2026, ramo lora-nonesperti): adattatori B A sui k-quant NON esperti. SOLO CPU,
nessun file su /mnt: lo stesso ambiente sintetico di `e2e_test_f32.py` (config Qwen3.5-MoE ridotta, 2 strati: 0 = GDN,
1 = attenzione piena, permutazioni casuali), ma un GGUF in cui le matrici delle famiglie LoRA sono k-quant VERI
(Q8_0 / Q4_0 quantizzati con gguf-py, Q4_K / Q6_K blocchi validi generati in numpy): ogni tipo di Thor-T e' coperto.

Uso:
    nice -n 15 env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=4 \\
        E2E_ALLENA_PROVE=@@ROOT:home@@/claude-pod-lavoro/lora-prove \\
        @@ROOT:venv@@/bin/python3 e2e_test_lora.py [L1 L2 L3 L4 L5]
    esito: $E2E_ALLENA_PROVE/lora/esito_test_lora[_<prove>].json

  L1 default AL BIT: `Studente` senza LoRA == codice del commit 6c0d13b (avanti, gradiente di theta, instradamento;
     ricette A32 e A bf16); con LoRA di rango 4 su TUTTE le famiglie, al passo 0 (B = 0) avanti, gradiente di theta e
     instradamento identici al bit, dL/dA = 0 esatto e dL/dB finito e non nullo; ciclo `allena` di default ==
     6c0d13b (campi nuovi = solo lora_* ai default); export con LoRA accesa e B = 0 (griglia) == sorgente al byte
  L2 gradiente di A e B == differenze finite centrate in float64 (ricetta A64, B casuale != 0 cosi' dL/dA != 0),
     attraverso l'indietro con checkpoint e l'instradamento salvato; cambi d'instradamento 0
  L3 ciclo con LoRA: 3 passi + ripresa da disco (studente NUOVO) + 3 == 6 filati AL BIT (theta, A, B, Adam dei due
     gruppi, stato, lora.npz dell'istantanea); con F32 + LoRA (tre gruppi) 2 + 2 == 4 al bit; LoRA accesa alla ripresa
     (decisione: Adam nuovo per A e B, modello invariato), spenta / alpha o rango cambiati alla ripresa = errore,
     lr_lora cambiato = decisione applicata; NaN in B solleva; lora_fermi; CLI
  L4 export + ricarica (modi griglia e ggml): solo i byte dei tensori LoRA cambiano (stesso tipo, stessa taglia),
     byte nuovi == riquantizza(W + s B A) calcolato a parte, studente SOLO dal file == studente del sorgente coi pesi
     riquantizzati AL BIT (strati 0 e 1), misure di sopravvivenza nella provenienza; errori scoperti
  L5 codec k-quant: valori dai codici == dequantizzazione di gguf-py al bit, impacchetta(codici) == byte,
     griglia(W_deq) == byte, griglia(W_deq + k passo) == codici + k (tagliati), gguf-py == ggml per Q8_0/Q4_0,
     ggml Q4_K/Q6_K validi; hf_a_file inversa esatta di `_applica`; la misura di sopravvivenza su una matrice
     sintetica: Delta W sotto mezzo passo SPARISCE con la griglia, sopra due passi resta
"""
from __future__ import annotations

import dataclasses
import functools
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
os.environ.setdefault("E2E_ALLENA_PROVE", "@@ROOT:home@@/claude-pod-lavoro/lora-prove")
import e2e_test_f32 as TF  # noqa: E402  (ambiente sintetico: config minuscola, permutazioni, _deq_c numpy)
from e2e_test_f32 import A, S, X, CSG, GT, GGUFReader, GGUFWriter, GQ  # noqa: E402
import e2e_lora as LR  # noqa: E402

DIR = Path(os.environ["E2E_ALLENA_PROVE"]) / "lora"
BASE = "6c0d13b"                                   # master prima del ramo lora-nonesperti
torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))
ESITO: dict = {"data": time.strftime("%Y-%m-%d %H:%M:%S"), "prove": {}}
muto = lambda *a, **k: None  # noqa: E731
TUTTE = LR.famiglie_lora("tutte")
LIB_OK = Path(LR.LIB_GGML).exists()


def registra(nome: str, ok: bool, **dati) -> bool:
    ESITO["prove"][nome] = {"ok": bool(ok), **dati}
    print(f"[{nome}] {'✅' if ok else '❌'} {json.dumps(dati, default=str)[:1800]}", flush=True)
    return bool(ok)


# ─────────────────────────────── GGUF sintetico con k-quant veri ───────────────────────────────
TIPO_SINTETICO = {
    (0, "attn_qkv.weight"): "Q6_K", (0, "attn_gate.weight"): "Q4_K", (0, "ssm_out.weight"): "Q8_0",
    (0, "ffn_gate_shexp.weight"): "Q4_K", (0, "ffn_up_shexp.weight"): "Q6_K", (0, "ffn_down_shexp.weight"): "Q8_0",
    (1, "attn_q.weight"): "Q4_K", (1, "attn_k.weight"): "Q6_K", (1, "attn_v.weight"): "Q4_0",
    (1, "attn_output.weight"): "Q4_K", (1, "ffn_gate_shexp.weight"): "Q8_0", (1, "ffn_up_shexp.weight"): "Q4_K",
    (1, "ffn_down_shexp.weight"): "Q8_0",
}


def blocchi_validi(tipo: str, righe: int, colonne: int, g: np.random.Generator) -> np.ndarray:
    """Byte (righe, byte per riga) di un k-quant valido, peso ~ 1/sqrt(colonne): Q8_0/Q4_0 dal quantizzatore di
    gguf-py su una gaussiana, Q4_K/Q6_K blocchi casuali coi campi di scala sani (nessuna dipendenza da /mnt)."""
    K, nb = LR.BLOCCO[tipo]
    amp = 1.0 / np.sqrt(colonne)
    if tipo in ("Q8_0", "Q4_0"):
        x = (amp * g.standard_normal((righe, colonne))).astype(np.float32)
        return np.asarray(GQ.quantize(x, getattr(GT, tipo)), dtype=np.uint8).reshape(righe, -1)
    n = righe * colonne // K
    b = g.integers(0, 256, size=(n, nb), dtype=np.uint8)
    if tipo == "Q4_K":
        d = (amp * 4.0 / (63 * 15) * g.uniform(0.5, 1.5, n)).astype(np.float16)
        dmin = (amp * 2.0 / 63 * g.uniform(0.5, 1.5, n)).astype(np.float16)
        b[:, 0:2] = d.reshape(-1, 1).view(np.uint8)
        b[:, 2:4] = dmin.reshape(-1, 1).view(np.uint8)
    else:                                           # Q6_K: scale int8 in [-40, 40], d sul valore
        b[:, 192:208] = g.integers(-40, 41, size=(n, 16)).astype(np.int8).view(np.uint8)
        d = (amp * 3.0 / (40 * 32) * g.uniform(0.5, 1.5, n)).astype(np.float16)
        b[:, 208:210] = d.reshape(-1, 1).view(np.uint8)
    return b.reshape(righe, -1)


def crea_gguf_lora(p: Path, seme: int = 7) -> Path:
    """Come `e2e_test_f32.crea_gguf` (stessi valori F32, esperti TQ1_0), ma le matrici delle famiglie LoRA in k-quant."""
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer
    cfg = TF.config_minuscola()
    g = np.random.default_rng(seme)
    gq = np.random.default_rng(seme + 1000)
    w = GGUFWriter(str(p), "qwen35moe")
    H = cfg.hidden_size
    w.add_tensor("token_embd.weight", (0.5 * g.standard_normal((cfg.vocab_size, H))).astype(np.float32))
    for L in range(2):
        tipo = "full" if cfg.layer_types[L] == "full_attention" else "linear"
        forme = {k: tuple(v.shape) for k, v in Qwen3_5MoeDecoderLayer(cfg, layer_idx=L).state_dict().items()}
        for fam, voce in CSG.MAPPA[tipo].items():
            if voce.get("per_esperto"):
                continue
            forma = (cfg.num_experts, H) if voce["hf"] == "mlp.gate.weight" else forme[voce["hf"]]
            if (L, fam) in TIPO_SINTETICO:
                tq = TIPO_SINTETICO[(L, fam)]
                w.add_tensor(f"blk.{L}.{fam}", blocchi_validi(tq, forma[0], forma[1], gq), raw_dtype=getattr(GT, tq))
                continue
            n = int(np.prod(forma[1:])) if len(forma) > 1 else forma[0]
            if voce["hf"].endswith("A_log"):
                v = g.uniform(-1.0, 0.5, forma)
            elif voce.get("f") == "piu_uno" or voce["hf"].endswith("norm.weight") and "layernorm" in voce["hf"]:
                v = 0.1 * g.standard_normal(forma)
            elif voce["hf"].endswith("linear_attn.norm.weight"):
                v = 1.0 + 0.1 * g.standard_normal(forma)
            elif voce["hf"].endswith("dt_bias"):
                v = g.uniform(-0.5, 0.5, forma)
            else:
                v = g.standard_normal(forma) / np.sqrt(max(n, 1))
            if voce["hf"] == "mlp.gate.weight":
                v = v * 3.0
            w.add_tensor(f"blk.{L}.{fam}", np.ascontiguousarray(TF._hf_in_gguf(voce, v), dtype=np.float32))
        I = cfg.moe_intermediate_size
        for nome, (R, C) in (("ffn_gate_exps", (I, H)), ("ffn_up_exps", (I, H)), ("ffn_down_exps", (H, I))):
            x = (g.standard_normal((cfg.num_experts * R, C)) / np.sqrt(C)).astype(np.float32)
            q = GQ.quantize(x, GT.TQ1_0).reshape(cfg.num_experts, R, -1)
            w.add_tensor(f"blk.{L}.{nome}.weight", np.ascontiguousarray(q), raw_dtype=GT.TQ1_0)
    w.add_tensor("output_norm.weight", (1.0 + 0.1 * g.standard_normal(H)).astype(np.float32))
    w.add_tensor("output.weight", (g.standard_normal((cfg.vocab_size, H)) / 16).astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return p


_GGUF = None


def gguf_lora() -> Path:
    global _GGUF
    if _GGUF is None:
        DIR.mkdir(parents=True, exist_ok=True)
        p = DIR / "sintetico_lora.gguf"
        p.unlink(missing_ok=True)
        _GGUF = crea_gguf_lora(p)
    return _GGUF


def cfg_lora(rango: int = 4, famiglie=TUTTE, alpha=None, seme: int = 20260926):
    return LR.ConfigLoRA(rango=rango, alpha=alpha, famiglie=tuple(famiglie), seme=seme)


def studente(ricetta: str = "A32", lora=None, allena_f32=(), mod=S, gguf=None):
    gguf = gguf or gguf_lora()
    if mod is S:
        return S.Studente(gguf=gguf, strati=[0, 1], ricetta=ricetta, motore_nome="torch", device="cpu", log=muto,
                          allena_f32=allena_f32, lora=lora)
    return mod.Studente(gguf=gguf, strati=[0, 1], ricetta=ricetta, motore_nome="torch", device="cpu", log=muto)


def _modulo_da_git(file: str, nome: str, base: str = BASE):
    testo = subprocess.run(["git", "-C", str(QUI), "show", f"{base}:thor_v2/{file}"], check=True,
                           capture_output=True, text=True).stdout
    DIR.mkdir(parents=True, exist_ok=True)
    p = DIR / f"{nome}.py"
    p.write_text(testo)
    spec = importlib.util.spec_from_file_location(nome, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[nome] = m
    spec.loader.exec_module(m)
    return m


def _passo(st, ids, x, R):
    """avanti (xL, top-k) e gradiente di theta (e di A, B) per la perdita sum(h * R)."""
    for p in st.parametri() + (st.parametri_lora() if hasattr(st, "parametri_lora") else []):
        p.grad = None
    out = st.avanti(ids, x0=x)
    val = st.gradiente(ids, lambda h: (h.to(R.dtype) * R).sum(), x0=x)
    return out["xL"], [p.grad.clone() for p in st.parametri()], val, [o.clone() for o in out["instr"]]


def _b_casuale(st, s: float, seme: int = 21) -> None:
    g = torch.Generator().manual_seed(seme)
    with torch.no_grad():
        for a in st.lora.values():
            a.B.copy_(s * torch.randn(a.B.shape, generator=g, dtype=torch.float64).to(a.B.dtype))


def _sha(p: Path) -> str:
    return X.sha256_file(p)


# ─────────────────────────────── L1 ───────────────────────────────
def prova_l1() -> bool:
    t0 = time.time()
    SP = _modulo_da_git("e2e_studente.py", f"e2e_studente_{BASE}")
    SP.config_testo = TF.config_minuscola
    det, ok = {}, True
    for ric in ("A32", "A"):
        sn, sp, sl = studente(ric), studente(ric, mod=SP), studente(ric, cfg_lora(4))
        for st in (sn, sp, sl):
            TF._theta_casuale(st)
        ids, x, R = TF._ingresso(sn)
        yn, gn, vn, in_ = _passo(sn, ids, x, R)
        yp, gp, vp, ip = _passo(sp, ids, x, R)
        yl, gl, vl, il = _passo(sl, ids, x, R)
        uguali = lambda a, b: all(torch.equal(u, v) for u, v in zip(a, b))  # noqa: E731
        d = {"default_vs_prima": torch.equal(yn, yp) and uguali(gn, gp) and vn == vp and uguali(in_, ip),
             "lora_b0_vs_default": torch.equal(yl, yn) and uguali(gl, gn) and vl == vn and uguali(il, in_),
             "dA_zero": all(float(a.A.grad.abs().max()) == 0.0 for a in sl.lora.values()),
             "dB_finito_non_nullo": all(bool(torch.isfinite(a.B.grad).all()) and float(a.B.grad.abs().max()) > 0
                                        for a in sl.lora.values()),
             "adattatori": len(sl.lora), "conteggi": sl.lora_conteggi, "cambi": sl.controlla()["cambi_instradamento"]}
        det[ric] = d
        ok &= d["default_vs_prima"] and d["lora_b0_vs_default"] and d["dA_zero"] and d["dB_finito_non_nullo"] \
            and d["cambi"] == 0 and d["adattatori"] == len(TIPO_SINTETICO)
    # ciclo di default (studente finto): e2e_allena nuovo == 6c0d13b
    AP = _modulo_da_git("e2e_allena.py", f"e2e_allena_{BASE}")
    import e2e_test_allena as TA
    src, WS = TA.sorgente_finta()
    es = {}
    for nome, mod in (("nuovo", A), ("prima", AP)):
        dd = DIR / f"l1_{nome}"
        shutil.rmtree(dd, ignore_errors=True)
        st = TA.StudenteFinto()
        ca = mod.ConfigAllena(lr=3e-2, righe_per_passo=2, righe_per_micro=1, righe_per_micro_valuta=2,
                              heldout_ogni=2, ripresa_ogni=2, pezzo_kl=24, attesa_ogni_s=0.01)
        stato = mod.allena(st, src, dd, ca, WS, WS, ferma_a=4, log=muto)
        es[nome] = (st, stato, mod.carica_ripresa(dd, "cpu"))
    (sn, zn, rn), (sp, zp, rp) = es["nuovo"], es["prima"]
    th = all(torch.equal(a, b) for a, b in zip(sn.parametri(), sp.parametri()))
    so, sr = rn["opt"]["state"], rp["opt"]["state"]
    adam = so.keys() == sr.keys() and all(torch.equal(so[i][k], sr[i][k]) for i in so
                                          for k in ("exp_avg", "exp_avg_sq", "step")) \
        and rn["opt"]["param_groups"] == rp["opt"]["param_groups"] and "lora" not in rn
    a_, b_ = TA._senza_tempi(zn), TA._senza_tempi(zp)
    nuovi = {k: a_["config"].pop(k) for k in set(a_["config"]) - set(b_["config"])}
    # 27/9 merge con vbop: anche i campi di V-Bop compaiono come nuovi, ai loro default (V-Bop spento); kl-pesata: idem
    # i campi di --peso-token (uniforme)
    ciclo = th and adam and a_ == b_ and nuovi == {"lora_rango": 0, "lora_alpha": None, "lora_famiglie": [],
                                                   "lr_lora": 1e-4, "lora_tipi": [], **TA._CFG_VBOP_DEFAULT,
                                                   **TA._CFG_PESO_DEFAULT, **TA._CFG_SOPHIA_DEFAULT, **TA._CFG_LORA_GA_DEFAULT,
                                                   **TA._CFG_INTEGRATA_PROPRI, **TA._CFG_BF16X3_DEFAULT, **TA._CFG_LOTA_DEFAULT, **TA._CFG_MC_DEFAULT}
    # export con LoRA accesa e B = 0, modo griglia: il file esportato e' il sorgente AL BYTE
    st = studente("A32", cfg_lora(4))
    d0 = DIR / "l1_ist"
    shutil.rmtree(d0, ignore_errors=True)
    d0.mkdir(parents=True)
    nome = A.salva_istantanea(d0, st, 0, {"metrica": 0.0})
    ist = A.carica_istantanea(d0 / nome)
    dst = DIR / "l1_b0.gguf"
    for p in (dst, Path(str(dst) + ".provenienza.json")):
        p.unlink(missing_ok=True)
    prov = X.esporta(gguf_lora(), dst, ist["scale"], ist["meta"], controlla_disco=False, log=muto,
                     lora=ist["lora"], lora_meta=ist["lora_meta"], lora_modo="griglia")
    b0 = _sha(dst) == _sha(gguf_lora()) and sum(prov["lora"]["verifica"]["byte_cambiati_per_tensore"].values()) == 0
    ok &= ciclo and b0
    return registra("L1_default_al_bit", ok, ricette=det, ciclo_theta=th, ciclo_adam=adam, ciclo_config_nuovi=nuovi,
                    ciclo=ciclo, export_b0_uguale_al_sorgente=b0, base=BASE, secondi=round(time.time() - t0, 1))


# ─────────────────────────────── L2 ───────────────────────────────
def prova_l2() -> bool:
    t0 = time.time()
    st = studente("A64", cfg_lora(3))
    TF._theta_casuale(st)
    _b_casuale(st, 0.05)
    with TF.tutto_fp64():
        ids, x, R = TF._ingresso(st, T=16)

        def perdita_nograd():
            with torch.no_grad():
                out = st.avanti(ids, x0=x)
                return float((st.esce(out["xL"]).to(torch.float64) * R).sum()), [o.clone() for o in out["instr"]]
        _, _, _, instr0 = _passo(st, ids, x, R)
        g = {(n, k): getattr(a, k).grad.clone() for n, a in st.lora.items() for k in ("A", "B")}
        st.azzera_cambi()
        gen = torch.Generator().manual_seed(12)
        per, peggiore, cambi = {}, 0.0, 0
        for n, a in st.lora.items():
            for k in ("A", "B"):
                m = getattr(a, k)
                idx = torch.randperm(m.numel(), generator=gen)[:3]
                num = den = 0.0
                for i in idx.tolist():
                    v0 = m.detach().reshape(-1)[i].item()
                    h = 1e-6 * max(1.0, abs(v0))
                    vals = []
                    for s in (+1, -1):
                        with torch.no_grad():
                            m.view(-1)[i] = v0 + s * h
                        f, ins = perdita_nograd()
                        cambi += sum(int((torch.sort(u, -1).values != torch.sort(v, -1).values).any(-1).sum())
                                     for u, v in zip(ins, instr0))
                        vals.append(f)
                    with torch.no_grad():
                        m.view(-1)[i] = v0
                    fd = (vals[0] - vals[1]) / (2 * h)
                    an = g[(n, k)].reshape(-1)[i].item()
                    num = max(num, abs(fd - an))
                    den = max(den, abs(an), abs(fd))
                e = num / max(den, 1e-12)
                per[f"{n}#{k}"] = {"err_rel": e, "max_abs_grad": den}
                peggiore = max(peggiore, e)
        nonnulli = all(float(v.abs().max()) > 0 for v in g.values())
    ok = peggiore < 1e-5 and cambi == 0 and nonnulli and len(per) == 2 * len(TIPO_SINTETICO)
    return registra("L2_gradiente_lora_vs_differenze_finite", ok, err_rel_peggiore=peggiore, cambi_instradamento=cambi,
                    gradienti_non_nulli=nonnulli, per_parametro=per, secondi=round(time.time() - t0, 1))


# ─────────────────────────────── L3 ───────────────────────────────
def sorgente_lora(n_train: int = 12, n_held: int = 4, T: int = 16, seme: int = 5):
    """Maestro = lo studente minuscolo con theta* e un adattatore B* != 0 (i non-esperti del maestro sono spostati
    di s B* A: la LoRA dello studente ha un segnale da inseguire)."""
    st = studente("A32", cfg_lora(4))
    g = torch.Generator().manual_seed(seme)
    with torch.no_grad():
        for p in st.parametri():
            p.copy_(0.2 * torch.randn(p.shape, generator=g))
    _b_casuale(st, 0.05, seme + 1)
    doms = ["a", "b", "a", "wiki"]
    righe = {}
    for lato, n in (("train", n_train), ("heldout", n_held)):
        rr = []
        for i in range(n):
            ids = torch.randint(0, st.cfg.vocab_size, (T,), generator=g)
            with torch.no_grad():
                h = st.esce(st.avanti(ids[None])["xL"]).to(torch.bfloat16).clone()
            rr.append((ids.numpy().astype(np.int32), h, doms[i % 4] if lato == "heldout" else "tr", None))
        righe[lato] = rr
    return A.SorgenteMemoria(righe, {}), st.testa(torch.float32)


def _ca(**kw):
    base = dict(lr=2e-2, righe_per_passo=2, righe_per_micro=1, righe_per_micro_valuta=2, heldout_ogni=2,
                ripresa_ogni=1, pezzo_kl=10, attesa_ogni_s=0.01, lr_f32=3e-3, lr_lora=3e-3)
    base.update(kw)
    return A.ConfigAllena(**base)


def _ca_lora(r: int = 4, **kw):
    return _ca(lora_rango=r, lora_famiglie=TUTTE, **kw)


def _uguali_lora(sa, sb) -> bool:
    return sa.chiavi_lora() == sb.chiavi_lora() and all(torch.equal(p, q) for p, q in zip(sa.parametri_lora(),
                                                                                         sb.parametri_lora()))


def prova_l3() -> bool:
    import e2e_test_allena as TA
    t0 = time.time()
    src, WS = sorgente_lora()
    ca = _ca_lora()
    d_f, d_r = DIR / "l3_filata", DIR / "l3_ripresa"
    for d in (d_f, d_r):
        shutil.rmtree(d, ignore_errors=True)
    st_f = studente("A32", cfg_lora(4))
    filata = A.allena(st_f, src, d_f, ca, WS, WS, log=muto)
    A.allena(studente("A32", cfg_lora(4)), src, d_r, ca, WS, WS, ferma_a=3, log=muto)
    st_b = studente("A32", cfg_lora(4))
    ripresa = A.allena(st_b, src, d_r, ca, WS, WS, riprendi=True, log=muto)
    rf, rr = A.carica_ripresa(d_f, "cpu"), A.carica_ripresa(d_r, "cpu")
    th = all(torch.equal(a, b) for a, b in zip(st_f.parametri(), st_b.parametri()))
    lo = _uguali_lora(st_f, st_b)
    mossi = sum(1 for a in st_f.lora.values() if float(a.B.abs().max()) > 0)
    adam = TF._uguali_adam(rf, rr) and len(rf["opt"]["param_groups"]) == 2 \
        and rf["opt"]["param_groups"][1]["lr"] == ca.lr_lora and rf["lora_nomi"] == st_f.chiavi_lora()
    stato = TA._senza_tempi(filata) == TA._senza_tempi(ripresa) and filata["passo"] == 6
    cambi = filata["cambi_instradamento"] == 0 and ripresa["cambi_instradamento"] == 0
    ist = A.carica_istantanea(d_f / filata["migliore"]["dir"])
    ist_ok = ("lora" in ist and list(ist["lora"]) == st_f.chiavi_lora() and ist["lora_meta"]["rango"] == 4
              and all(v["A"].shape == tuple(st_f.lora[n].A.shape) for n, v in ist["lora"].items()))
    if filata["migliore"]["passo"] == filata["passo"]:
        ist_ok &= all(np.array_equal(v["A"], st_f.lora[n].A.detach().numpy())
                      and np.array_equal(v["B"], st_f.lora[n].B.detach().numpy()) for n, v in ist["lora"].items())
    # (b) F32 + LoRA: tre gruppi, 2 + ripresa + 2 == 4 al bit
    fam = S.famiglie_f32("attn_norm,ffn_gate_inp,ssm_dt")
    ca3 = _ca_lora(allena_f32=fam)
    d3f, d3r = DIR / "l3_tre_filata", DIR / "l3_tre_ripresa"
    for d in (d3f, d3r):
        shutil.rmtree(d, ignore_errors=True)
    s3f = studente("A32", cfg_lora(4), allena_f32=fam)
    z3f = A.allena(s3f, src, d3f, ca3, WS, WS, ferma_a=4, log=muto)
    A.allena(studente("A32", cfg_lora(4), allena_f32=fam), src, d3r, ca3, WS, WS, ferma_a=2, log=muto)
    s3b = studente("A32", cfg_lora(4), allena_f32=fam)
    z3r = A.allena(s3b, src, d3r, ca3, WS, WS, riprendi=True, ferma_a=4, log=muto)
    r3f, r3r = A.carica_ripresa(d3f, "cpu"), A.carica_ripresa(d3r, "cpu")
    tre = (all(torch.equal(a, b) for a, b in zip(s3f.parametri(), s3b.parametri())) and _uguali_lora(s3f, s3b)
           and all(torch.equal(a, b) for a, b in zip(s3f.parametri_f32(), s3b.parametri_f32()))
           and TF._uguali_adam(r3f, r3r) and [g["lr"] for g in r3f["opt"]["param_groups"]] == [ca3.lr, ca3.lr_f32,
                                                                                               ca3.lr_lora]
           and TA._senza_tempi(z3f) == TA._senza_tempi(z3r))
    # (c) LoRA accesa alla ripresa (decisione), lr_lora cambiato; spenta / alpha / rango cambiati = errore
    d_x = DIR / "l3_accesa"
    shutil.rmtree(d_x, ignore_errors=True)
    A.allena(studente("A32"), src, d_x, _ca(), WS, WS, ferma_a=2, log=muto)
    reg = []
    s_x = A.allena(studente("A32", cfg_lora(4)), src, d_x, _ca_lora(), WS, WS, riprendi=True, ferma_a=3,
                   log=reg.append)
    r_x = A.carica_ripresa(d_x, "cpu")
    n_th = len(studente("A32").parametri())
    passi_lora = [int(r_x["opt"]["state"][n_th + j]["step"]) for j in range(2 * len(r_x["lora_nomi"]))]
    accesa = (s_x["passo"] == 3 and passi_lora == [1] * len(passi_lora)
              and int(r_x["opt"]["state"][0]["step"]) == 3 and any("LoRA accesa ora" in x for x in reg)
              and s_x["config"]["lora_rango"] == 4)
    reg2 = []
    A.allena(studente("A32", cfg_lora(4)), src, d_x, _ca_lora(lr_lora=1e-3), WS, WS, riprendi=True, ferma_a=4,
             log=reg2.append)
    r_x2 = A.carica_ripresa(d_x, "cpu")
    lr_cambiato = r_x2["opt"]["param_groups"][1]["lr"] == 1e-3 and any(x.startswith("ripresa: lr_lora") for x in reg2)
    errori = {}
    for nome, st_e, ca_e in (("spenta", studente("A32"), _ca()),
                             ("alpha", studente("A32", cfg_lora(4, alpha=8.0)), _ca_lora(lora_alpha=8.0)),
                             ("rango", studente("A32", cfg_lora(2)), _ca_lora(2))):
        try:
            A.allena(st_e, src, d_x, ca_e, WS, WS, riprendi=True, ferma_a=5, log=muto)
            errori[nome] = False
        except RuntimeError as e:
            errori[nome] = "LoRA" in str(e) or "adattatori" in str(e)
    disallineato = False
    try:
        A.allena(studente("A32"), src, DIR / "l3_mai", _ca_lora(), WS, WS, log=muto)
    except ValueError:
        disallineato = True
    # (d) NaN in B solleva; lora_fermi; CLI
    st_n = studente("A32", cfg_lora(4))
    with torch.no_grad():
        next(iter(st_n.lora.values())).B.view(-1)[0] = float("nan")
    try:
        st_n.controlla()
        nan = False
    except FloatingPointError:
        nan = True
    with torch.no_grad():
        next(iter(st_n.lora.values())).B.view(-1)[0] = 0.0
    ids, x, R = TF._ingresso(st_n, T=8)
    with st_n.lora_fermi():
        _passo(st_n, ids, x, R)
    fermi = all(p.grad is None for p in st_n.parametri_lora()) and all(p.requires_grad for p in st_n.parametri_lora())
    c = A._config_da_arg(TF._arg(lora_nonesperti=8, lora_alpha=None, lora_famiglie="attn,shexp", lr_lora=2e-4))
    c0 = A._config_da_arg(TF._arg(lora_nonesperti=0, lora_alpha=4.0, lora_famiglie="gdn", lr_lora=1e-4))
    cli = ((c.lora_rango, c.lora_alpha, c.lora_famiglie, c.lr_lora) ==
           (8, None, ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp"),
            2e-4)
           and (c0.lora_rango, c0.lora_alpha, c0.lora_famiglie) == (0, None, ())
           and A._config_da_arg(TF._arg()).lora_rango == 0
           and {"lora_rango", "lora_alpha", "lora_famiglie", "lr_lora"} <= A._CAMPI_DI_DECISIONE)
    st_t = studente("A32", LR.ConfigLoRA(rango=2, famiglie=TUTTE, tipi=("Q8_0",)))
    c_t = A._config_da_arg(TF._arg(lora_nonesperti=4, lora_alpha=None, lora_famiglie="tutte", lr_lora=1e-4,
                                   lora_tipi="q8_0"))
    tipi = (len(st_t.lora) == sum(1 for v in TIPO_SINTETICO.values() if v == "Q8_0")
            and all(a.tipo == "Q8_0" for a in st_t.lora.values()) and c_t.lora_tipi == ("Q8_0",)
            and A.firma_lora_studente(st_t) == (2, None, TUTTE, ("Q8_0",)))
    ok = th and lo and mossi > 0 and adam and stato and cambi and ist_ok and tre and accesa and lr_cambiato \
        and all(errori.values()) and disallineato and nan and fermi and cli and tipi
    return registra("L3_ripresa_al_bit_con_lora", ok, theta=th, lora=lo, adattatori_mossi=mossi, adam=adam,
                    stato=stato, cambi=cambi, istantanea=ist_ok, tre_gruppi=tre, accesa_alla_ripresa=accesa,
                    passi_adam_lora=passi_lora[:4], lr_lora_cambiato=lr_cambiato, errori=errori,
                    studente_config_diversi=disallineato, nan_solleva=nan, lora_fermi=fermi, cli=cli, lora_tipi=tipi,
                    heldout=[(v["passo"], round(v["metrica"], 6)) for v in filata["storia_heldout"]],
                    secondi=round(time.time() - t0, 1))


# ─────────────────────────────── L4 ───────────────────────────────
def prova_l4() -> bool:
    t0 = time.time()
    src = gguf_lora()
    st = studente("A32", cfg_lora(4))
    TF._theta_casuale(st, seme=4, s=0.1)
    _b_casuale(st, 0.5, seme=31)                       # Delta W dell'ordine del passo: in parte sopravvive
    d0 = DIR / "l4_ist"
    shutil.rmtree(d0, ignore_errors=True)
    d0.mkdir(parents=True)
    nome = A.salva_istantanea(d0, st, 1, {"metrica": 0.0})
    ist = A.carica_istantanea(d0 / nome)
    theta = torch.load(d0 / nome / "theta.pt", map_location="cpu", weights_only=True)
    delta_ok = all(np.array_equal(LR.delta_file_da(v["A"], v["B"], ist["lora_meta"]["scala"],
                                                   ist["lora_meta"]["voci"][n]), st.lora[n].delta_file())
                   for n, v in ist["lora"].items())
    esiti = {}
    ric0 = X.ricarica
    X.ricarica = functools.partial(ric0, ricetta="A32", token=24)
    try:
        for modo in ("griglia", "ggml"):
            if modo == "ggml" and not LIB_OK:
                esiti[modo] = {"saltata": f"{LR.LIB_GGML} assente"}
                continue
            dst = DIR / f"l4_{modo}.gguf"
            for p in (dst, Path(str(dst) + ".provenienza.json"), dst.with_name(dst.name + ".parziale")):
                p.unlink(missing_ok=True)
            prov = X.esporta(src, dst, ist["scale"], ist["meta"], theta=theta, strati_ricarica=(0, 1),
                             controlla_disco=False, log=muto, lora=ist["lora"], lora_meta=ist["lora_meta"],
                             lora_modo=modo)
            ric = prov["ricarica"]
            ricarica_ok = all(r["scale_al_bit"] and r["uscita_al_bit"] for r in ric.values())
            # verifica indipendente: byte dei tensori LoRA == riquantizza(W + dW); ogni altro tensore identico
            ra, rb = GGUFReader(str(src)), GGUFReader(str(dst))
            ta, tb = {t.name: t for t in ra.tensors}, {t.name: t for t in rb.tensors}
            tens_ok, cambiati = True, 0
            for n, a in ta.items():
                va, vb = np.asarray(a.data), np.asarray(tb[n].data)
                if n in ist["lora"]:
                    tipo = a.tensor_type.name
                    W = LR.dequantizza(va, tipo)
                    atteso = LR.riquantizza(va, W + st.lora[n].delta_file(), tipo, modo)
                    tens_ok &= np.array_equal(vb, atteso) and tb[n].tensor_type.name == tipo
                    cambiati += int((va != vb).sum())
                elif n in ist["scale"]:
                    nb, _, off, k = S.CS.LAYOUT[a.tensor_type.name]
                    xa, xb = va.reshape(-1, nb).copy(), vb.reshape(-1, nb).copy()
                    xa[:, off:off + 2 * k] = 0
                    xb[:, off:off + 2 * k] = 0
                    tens_ok &= np.array_equal(xa, xb)
                else:
                    tens_ok &= np.array_equal(va, vb)
            taglia = src.stat().st_size == dst.stat().st_size
            del ra, rb, ta, tb
            sp = prov["lora"]["sopravvivenza_per_tensore"]
            man = {n: round(v["mantenuto"], 4) for n, v in sp.items()}
            uscita = {L: r.get("lora_uscita") for L, r in ric.items()}
            esiti[modo] = {"ricarica_al_bit": ricarica_ok, "tensori": tens_ok, "byte_cambiati": cambiati,
                           "taglia_invariata": taglia, "mantenuto_per_tensore": man, "uscita": uscita,
                           "ok": ricarica_ok and tens_ok and taglia and cambiati > 0 and len(sp) == len(ist["lora"])
                           and all(u is not None and u["mantenuto"] == u["mantenuto"] for u in uscita.values())}
    finally:
        X.ricarica = ric0
    # errori scoperti: tipo non k-quant, forma sbagliata (prima di scrivere); byte fuori regione manomessi
    errori = {}
    for nome_e, nuovi in (("non_kquant", {"blk.0.attn_norm.weight": np.zeros(4, np.uint8)}),
                          ("forma", {"blk.0.attn_gate.weight": np.zeros(3, np.uint8)})):
        try:
            X._regioni_lora(src, nuovi)
            errori[nome_e] = False
        except ValueError:
            errori[nome_e] = True
    d3 = DIR / "l4_manomesso.gguf"
    shutil.copyfile(src, d3)
    t_g = next(t for t in GGUFReader(str(src)).tensors if t.name == "blk.0.attn_gate.weight")
    reg = X._regioni_lora(d3, {t_g.name: np.asarray(t_g.data).copy()})
    prima = X._impronta_fuori(d3, reg)
    mm = np.memmap(d3, dtype=np.uint8, mode="r+")
    mm[-1] ^= 1
    mm.flush()
    del mm
    errori["manomissione_vista"] = X._impronta_fuori(d3, reg) != prima
    d3.unlink()
    ok = delta_ok and all(v.get("ok", "saltata" in v) for v in esiti.values()) and all(errori.values())
    return registra("L4_export_lora_e_ricarica", ok, delta_file_istantanea_uguale=delta_ok, modi=esiti,
                    errori_scoperti=errori, lib_ggml=LIB_OK, secondi=round(time.time() - t0, 1))


# ─────────────────────────────── L5 ───────────────────────────────
def prova_l5() -> bool:
    t0 = time.time()
    g = np.random.default_rng(3)
    per_tipo, ok = {}, True
    for tipo in LR.TIPI:
        raw = blocchi_validi(tipo, 64, 512, g)
        b = LR._blocchi(raw, tipo)
        c, p, s, lo, hi = LR._campi(b, tipo)
        W = LR.dequantizza(raw, tipo)
        valori = np.array_equal(LR._valori(c, p, s, tipo).reshape(W.shape).view(np.uint32), W.view(np.uint32))
        formula = float(np.abs((p.astype(np.float64) * c - s).reshape(W.shape) - W).max()) == 0.0
        pack = np.array_equal(LR.impacchetta(b, c, tipo), b)
        gr0 = np.array_equal(LR.quantizza_griglia(raw, W, tipo), raw)
        # W + k passo -> codici + k (tagliati), con k in {-2, .., 2} per elemento
        k = g.integers(-2, 3, size=c.shape)
        Wk = (W.astype(np.float64) + (k * p).reshape(W.shape)).astype(np.float32)
        ck = LR._campi(LR._blocchi(LR.quantizza_griglia(raw, Wk, tipo), tipo), tipo)[0]
        vivi = p != 0
        spost = bool(np.array_equal(ck[vivi], np.clip(c + k, lo, hi)[vivi])) and bool(np.array_equal(ck[~vivi], c[~vivi]))
        r = {"valori_al_bit": valori, "passo_per_codice_meno_spostamento": formula, "impacchetta": pack,
             "griglia_W_deq_uguale": gr0, "griglia_k_passi": spost}
        if LIB_OK:
            x = (g.standard_normal((32, 512)) / np.sqrt(512)).astype(np.float32)
            q = LR.quantizza_ggml(x, tipo)
            Wq = LR.dequantizza(q, tipo)
            r["ggml_errore_rel"] = float(np.linalg.norm(Wq - x) / np.linalg.norm(x))
            r["ggml_valido"] = q.shape == (32, 512 // LR.BLOCCO[tipo][0] * LR.BLOCCO[tipo][1]) and \
                bool(np.isfinite(Wq).all()) and r["ggml_errore_rel"] < 0.2
            if tipo in ("Q8_0", "Q4_0"):
                r["gguf_py_uguale_ggml"] = bool(np.array_equal(LR.quantizza_ggml(x, tipo, via="gguf"),
                                                               LR.quantizza_ggml(x, tipo, via="ggml")))
        per_tipo[tipo] = r
        ok &= all(v for kk, v in r.items() if isinstance(v, bool))
    # hf_a_file inversa esatta di _applica (permutazioni casuali dell'ambiente sintetico)
    inv = {}
    for tipo_s in ("linear", "full"):
        for fam, voce in CSG.MAPPA[tipo_s].items():
            if fam.split(".")[0] not in LR.FAMIGLIE_LORA:
                continue
            n_r = len(CSG.PERM[voce["perm_npz"]]) if voce["rel"] == "perm_righe" else 7
            n_c = len(CSG.PERM[voce["perm_npz"]]) if voce["rel"] == "perm_colonne" else 5
            M = g.standard_normal((n_r, n_c)).astype(np.float32)
            inv[f"{tipo_s}.{fam}"] = bool(np.array_equal(CSG._applica(voce, LR.hf_a_file(M, voce)), M))
    ok &= all(inv.values())
    # la misura di sopravvivenza su matrici sintetiche (gaussiana quantizzata come fa llama.cpp, Delta W gaussiano
    # i.i.d. di rms rho * passo): con la griglia, sotto mezzo passo Delta W SPARISCE, sopra resta
    curva = {}
    tipi_c = ("Q8_0", "Q4_K") if LIB_OK else ("Q8_0",)
    x = (g.standard_normal((256, 512)) / np.sqrt(512)).astype(np.float32)
    for tipo in tipi_c:
        raw = LR.quantizza_ggml(x, tipo) if LIB_OK else LR.quantizza_ggml(x, tipo, via="gguf")
        W = LR.dequantizza(raw, tipo)
        pr = LR.passo_rms(raw, tipo)
        U = g.standard_normal(W.shape)
        curva[tipo] = {}
        for rho in (0.05, 0.25, 0.5, 2.0):
            dW = (U * rho * pr).astype(np.float32)
            rn = LR.riquantizza(raw, W + dW, tipo, "griglia")
            curva[tipo][rho] = LR.misura_fusione(W, dW, LR.dequantizza(rn, tipo), raw, rn)["mantenuto"]
    soglia = curva["Q8_0"][0.05] < 0.02 and curva["Q8_0"][2.0] > 0.9 and \
        ("Q4_K" not in curva or (curva["Q4_K"][0.05] < 0.02 and curva["Q4_K"][0.5] > 0.5))
    ok &= soglia
    return registra("L5_codec_kquant", ok, per_tipo=per_tipo, hf_a_file_inversa=inv,
                    sopravvivenza_sintetica_griglia=curva, soglia=soglia, lib_ggml=LIB_OK,
                    secondi=round(time.time() - t0, 1))


PROVE = {"L1": prova_l1, "L2": prova_l2, "L3": prova_l3, "L4": prova_l4, "L5": prova_l5}


def main() -> int:
    DIR.mkdir(parents=True, exist_ok=True)
    scelte = [a.upper() for a in sys.argv[1:]] or list(PROVE)
    ok, t0 = True, time.time()
    for k in scelte:
        t1 = time.time()
        try:
            ok &= PROVE[k]()
        except Exception as e:
            ok = False
            registra(f"{k}_eccezione", False, errore=repr(e), traccia=traceback.format_exc()[-3000:])
        print(f"   ({k}: {time.time() - t1:.1f} s)", flush=True)
    ESITO["tutto_verde"] = ok
    ESITO["secondi"] = round(time.time() - t0, 1)
    suff = "_" + "_".join(scelte) if sys.argv[1:] else ""
    (DIR / f"esito_test_lora{suff}.json").write_text(json.dumps(ESITO, indent=1, default=str))
    print(f"ESITO: {'TUTTO VERDE' if ok else 'ROSSO'} ({ESITO['secondi']} s)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

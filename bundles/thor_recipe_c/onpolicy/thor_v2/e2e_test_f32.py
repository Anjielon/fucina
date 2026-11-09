#!/usr/bin/env python3
"""Prove di `--allena-f32` (27/9/2026): tensori F32 del GGUF (norme, router, GDN) allenati insieme a theta. SOLO
CPU, nessun file su /mnt: un GGUF SINTETICO minuscolo (2 strati: 0 = GDN, 1 = attenzione piena; hidden 512, 8
esperti TQ1_0, tutti gli altri tensori F32 con le relazioni vere di `mappa_completa_gguf_hf.json`), permutazioni
casuali al posto di `mappa_completa_gguf_hf_perm.npz`, config Qwen3.5-MoE ridotta al posto di quella di /mnt.

Uso:
    nice -n 15 env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=4 \\
        @@ROOT:venv@@/bin/python3 e2e_test_f32.py [A23 A24 A25 A26 A27]
    esito: $E2E_ALLENA_PROVE/esito_test_f32[_<prove>].json

  A23 default AL BIT: `Studente` e `allena` senza --allena-f32 == codice del commit d22c2cf (avanti, gradiente di
      theta, ciclo intero: theta, Adam, stato); con TUTTE le famiglie accese, al passo 0 avanti e gradiente di
      theta identici al bit (ricette A32, A bf16, A_r32 = router fp32) e valori da esportare == byte del file
  A24 gradiente dei master F32 (tutte le 13 famiglie) == differenze finite centrate in float64 (ricetta A64,
      instradamento invariato), attraverso l'indietro con checkpoint e instradamento salvato
  A25 ciclo con --allena-f32 sul modello minuscolo: 3 passi + ripresa da disco (studente NUOVO) + 3 == 6 filati AL
      BIT (theta, master F32, Adam dei due gruppi, stato, istantanea f32.npz); cambi d'instradamento 0; famiglie
      aggiunte alla ripresa (Adam nuovo, valori del GGUF), tolte = errore, lr_f32 cambiato = decisione applicata
  A26 export: F32 allenati scritti al posto dei vecchi (solo quei byte), ogni altro byte identico, rilettura al
      bit, studente ricaricato SOLO dal file == studente del sorgente con theta e F32, al bit; errori scoperti
  A27 integrazione: --allena-f32 con --beta2 (entrambi i gruppi, anche cambiato alla ripresa) e --ema-theta (solo
      theta; ema/ e migliore_ema_* con i F32 correnti), ripresa al bit; E2E_GDN=fla + ssm_a/ssm_dt/ssm_alpha rifiutato
  Numerazione: nel ramo allena-f32 erano A21-A24; rinominate A23-A26 al merge in integrazione (A21 = rumore del
  gradiente, A22 = held-out ridotto/EMA/beta2 in e2e_test_allena.py). Cartelle di lavoro a23_/a25_/a26_ di conseguenza.
"""
from __future__ import annotations

import dataclasses
import functools
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

QUI = Path(__file__).resolve().parent
# i moduli aggiungono a sys.path cartelle vietate a queste prove (/mnt, l'albero principale): mai lette
for _p in ("@@ROOT:models@@/build/llamacpp-tq1-k1024/gguf-py", "@@ROOT:forge@@/kernel-rotazione"):
    sys.path_importer_cache[_p] = None
sys.path.insert(0, str(QUI.parent / "kernel-rotazione"))
sys.path.insert(0, str(QUI))

_np_load = np.load
np.load = lambda *a, **k: {}                   # carica_strato_gguf legge il .npz delle permutazioni all'import
try:
    import carica_strato_gguf as CSG  # noqa: E402
finally:
    np.load = _np_load
from gguf import GGMLQuantizationType as GT, GGUFReader, GGUFWriter  # noqa: E402
import gguf.quants as GQ  # noqa: E402
import e2e_studente as S  # noqa: E402
import e2e_allena as A  # noqa: E402
import e2e_esporta as X  # noqa: E402

DIR = Path(os.getenv("E2E_ALLENA_PROVE", "@@ROOT:lavoro@@/e2e/prove_allena")) / "f32"
BASE = "d22c2cf"
torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))
ESITO: dict = {"data": time.strftime("%Y-%m-%d %H:%M:%S"), "prove": {}}
muto = lambda *a, **k: None  # noqa: E731
TUTTE = ",".join(list(S.FAMIGLIE_F32) + list(S.F32_GLOBALI))
S.RICETTE["A_r32"] = dataclasses.replace(S.RICETTE["A"], nome="A_r32", router_fp32=True)


def registra(nome: str, ok: bool, **dati) -> bool:
    ESITO["prove"][nome] = {"ok": bool(ok), **dati}
    print(f"[{nome}] {'✅' if ok else '❌'} {json.dumps(dati, default=str)[:1500]}", flush=True)
    return bool(ok)


# ─────────────────────────────── modello e GGUF sintetici ───────────────────────────────
def config_minuscola():
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    cfg = Qwen3_5MoeTextConfig(
        vocab_size=512, hidden_size=512, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=64, linear_key_head_dim=32, linear_value_head_dim=32, linear_num_key_heads=2,
        linear_num_value_heads=4, moe_intermediate_size=512, shared_expert_intermediate_size=64,
        num_experts_per_tok=2, num_experts=8, layer_types=["linear_attention", "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25,
                         "mrope_section": [3, 3, 2], "mrope_interleaved": True})
    cfg._attn_implementation = "sdpa"
    return cfg


S.config_testo = config_minuscola
_PERM_LUNGHEZZE = {"linear.attn_gate.weight": 128, "linear.attn_qkv.weight": 256, "linear.ssm_a": 4,
                   "linear.ssm_alpha.weight": 4, "linear.ssm_beta.weight": 4, "linear.ssm_conv1d.weight": 256,
                   "linear.ssm_dt.bias": 4, "linear.ssm_out.weight": 128}
_g = np.random.default_rng(20260927)
CSG.PERM.clear()
CSG.PERM.update({k: _g.permutation(n).astype(np.int64) for k, n in _PERM_LUNGHEZZE.items()})


def _deq_tq1(blocchi: np.ndarray, tipo: str) -> np.ndarray:
    assert tipo == "TQ1_0"
    return GQ.dequantize(np.ascontiguousarray(blocchi, dtype=np.uint8).reshape(-1, 54), GT.TQ1_0).reshape(-1, 256)


X._deq_c = _deq_tq1                                   # il codice C di llama.cpp sta su /mnt: riferimento numpy


def _hf_in_gguf(voce: dict, v: np.ndarray) -> np.ndarray:
    """Inversa di `carica_strato_gguf._applica` (HF -> valore del file)."""
    if voce["hf"].endswith("shared_expert_gate.weight"):
        v = v.reshape(-1)
    if voce.get("fetta") == "squeeze":
        v = v[:, 0, :]
    rel = voce["rel"]
    if rel == "trasformata":
        return v + 1.0 if voce["f"] == "piu_uno" else -np.exp(v)
    if rel in ("perm_righe", "perm_1d"):
        return (-np.exp(v) if voce.get("f") == "neg_exp" else v)[CSG.PERM[voce["perm_npz"]]]
    if rel == "perm_colonne":
        return v[:, CSG.PERM[voce["perm_npz"]]]
    return v


def crea_gguf(p: Path, seme: int = 7) -> Path:
    """GGUF sintetico: F32 ovunque tranne gli esperti (TQ1_0 veri, `gguf.quants.quantize`)."""
    from accelerate import init_empty_weights  # noqa: F401
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer
    cfg = config_minuscola()
    g = np.random.default_rng(seme)
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
            n = int(np.prod(forma[1:])) if len(forma) > 1 else forma[0]
            if voce["hf"].endswith("A_log"):
                v = g.uniform(-1.0, 0.5, forma)
            elif voce.get("f") == "piu_uno" or voce["hf"].endswith("norm.weight") and "layernorm" in voce["hf"]:
                v = 0.1 * g.standard_normal(forma)               # HF (1 + w) ~ 1: nel file ~ 1 + rumore
            elif voce["hf"].endswith("linear_attn.norm.weight"):
                v = 1.0 + 0.1 * g.standard_normal(forma)
            elif voce["hf"].endswith("dt_bias"):
                v = g.uniform(-0.5, 0.5, forma)
            else:
                v = g.standard_normal(forma) / np.sqrt(max(n, 1))
            if voce["hf"] == "mlp.gate.weight":
                v = v * 3.0                                      # router con logit ben separati
            w.add_tensor(f"blk.{L}.{fam}", np.ascontiguousarray(_hf_in_gguf(voce, v), dtype=np.float32))
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


def gguf_sintetico() -> Path:
    global _GGUF
    if _GGUF is None:
        DIR.mkdir(parents=True, exist_ok=True)
        p = DIR / "sintetico.gguf"
        p.unlink(missing_ok=True)
        _GGUF = crea_gguf(p)
    return _GGUF


def studente(ricetta: str = "A32", allena_f32=(), mod=S, **kw):
    return mod.Studente(gguf=gguf_sintetico(), strati=[0, 1], ricetta=ricetta, motore_nome="torch", device="cpu",
                        log=muto, allena_f32=allena_f32, **kw) if mod is S else \
        mod.Studente(gguf=gguf_sintetico(), strati=[0, 1], ricetta=ricetta, motore_nome="torch", device="cpu",
                     log=muto, **kw)


def _modulo_da_git(file: str, nome: str):
    testo = subprocess.run(["git", "-C", str(QUI), "show", f"{BASE}:thor_v2/{file}"], check=True,
                           capture_output=True, text=True).stdout
    DIR.mkdir(parents=True, exist_ok=True)
    p = DIR / f"{nome}.py"
    p.write_text(testo)
    spec = importlib.util.spec_from_file_location(nome, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[nome] = m
    spec.loader.exec_module(m)
    return m


def _theta_casuale(st, seme: int = 1, s: float = 0.05) -> None:
    g = torch.Generator().manual_seed(seme)
    with torch.no_grad():
        for p in st.parametri():
            p.copy_(s * torch.randn(p.shape, generator=g))


def _ingresso(st, seme: int = 3, T: int = 24):
    g = torch.Generator().manual_seed(seme)
    x = (torch.randn(1, T, st.cfg.hidden_size, generator=g) * 0.7).to(st.ric.dtype_residuo)
    R = torch.randn(T, st.cfg.hidden_size, generator=g, dtype=torch.float64)
    return torch.zeros(1, T, dtype=torch.long), x, R


def _passo(st, ids, x, R):
    """avanti (xL, top-k) e gradiente di theta (e dei master) per la perdita sum(h * R)."""
    for p in st.parametri() + (st.parametri_f32() if hasattr(st, "parametri_f32") else []):
        p.grad = None
    out = st.avanti(ids, x0=x)
    val = st.gradiente(ids, lambda h: (h.to(R.dtype) * R).sum(), x0=x)
    return out["xL"], [p.grad.clone() for p in st.parametri()], val, [o.clone() for o in out["instr"]]


# ─────────────────────────────── A23 ───────────────────────────────
def prova_a23() -> bool:
    t0 = time.time()
    SP = _modulo_da_git("e2e_studente.py", "e2e_studente_d22c2cf")
    SP.config_testo = config_minuscola
    SP.RICETTE["A_r32"] = S.RICETTE["A_r32"]
    det, ok = {}, True
    for ric in ("A32", "A", "A_r32"):
        sn, sp, sa = studente(ric), studente(ric, mod=SP), studente(ric, TUTTE)
        for st in (sn, sp, sa):
            _theta_casuale(st)
        ids, x, R = _ingresso(sn)
        yn, gn, vn, in_ = _passo(sn, ids, x, R)
        yp, gp, vp, ip = _passo(sp, ids, x, R)
        ya, ga, va, ia = _passo(sa, ids, x, R)
        d = {"default_vs_prima": torch.equal(yn, yp) and all(torch.equal(a, b) for a, b in zip(gn, gp)) and vn == vp
             and all(torch.equal(a, b) for a, b in zip(in_, ip)),
             "tutte_vs_default": torch.equal(ya, yn) and all(torch.equal(a, b) for a, b in zip(ga, gn)) and va == vn
             and all(torch.equal(a, b) for a, b in zip(ia, in_)),
             "f32_con_gradiente": all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                                      for p in sa.parametri_f32()),
             "cambi": sa.controlla()["cambi_instradamento"], "n_f32": len(sa.f32)}
        det[ric] = d
        ok &= d["default_vs_prima"] and d["tutte_vs_default"] and d["f32_con_gradiente"] and d["cambi"] == 0
    # conteggi e valori da esportare == byte del file
    sa = studente("A32", TUTTE)
    r = GGUFReader(str(gguf_sintetico()))
    tens = {t.name: t for t in r.tensors}
    esp = sa.f32_per_export()
    al_file = all(np.array_equal(v.view(np.uint32), np.asarray(tens[n].data).view(np.uint32)) for n, v in esp.items())
    attesi = {f: (1 if f in ("attn_q_norm", "attn_k_norm", "output_norm") else
                  2 if f in ("attn_norm", "post_attention_norm", "ffn_gate_inp", "ffn_gate_inp_shexp") else 1)
              for f in S.famiglie_f32(TUTTE)}
    conteggi = sa.f32_conteggi == attesi and len(esp) == 17
    del r, tens
    # ciclo intero (studente finto, nessun F32): e2e_allena nuovo == d22c2cf
    AP = _modulo_da_git("e2e_allena.py", "e2e_allena_d22c2cf")
    import e2e_test_allena as TA
    src, WS = TA.sorgente_finta()
    esiti = {}
    for nome, mod in (("nuovo", A), ("prima", AP)):
        dd = DIR / f"a23_{nome}"
        shutil.rmtree(dd, ignore_errors=True)
        st = TA.StudenteFinto()
        ca = mod.ConfigAllena(lr=3e-2, righe_per_passo=2, righe_per_micro=1, righe_per_micro_valuta=2,
                              heldout_ogni=2, ripresa_ogni=2, pezzo_kl=24, attesa_ogni_s=0.01)
        stato = mod.allena(st, src, dd, ca, WS, WS, ferma_a=4, log=muto)
        esiti[nome] = (st, stato, mod.carica_ripresa(dd, "cpu"))
    (sn, zn, rn), (sp, zp, rp) = esiti["nuovo"], esiti["prima"]
    th = all(torch.equal(a, b) for a, b in zip(sn.parametri(), sp.parametri()))
    so, sr = rn["opt"]["state"], rp["opt"]["state"]
    adam = so.keys() == sr.keys() and all(torch.equal(so[i][k], sr[i][k]) for i in so
                                          for k in ("exp_avg", "exp_avg_sq", "step"))
    gruppi = rn["opt"]["param_groups"] == rp["opt"]["param_groups"] and "f32" not in rn
    a_, b_ = TA._senza_tempi(zn), TA._senza_tempi(zp)
    nuovi = {k: a_["config"].pop(k) for k in set(a_["config"]) - set(b_["config"])}
    # 27/9 (master 6c0d13b): default di tieni_riprese 2 -> 1, campo di ESECUZIONE (quante riprese restano sul disco,
    # non cosa si calcola): cambio dichiarato, l'unico ammesso fra i campi che d22c2cf conosceva
    tieni = (a_["config"].pop("tieni_riprese", None), b_["config"].pop("tieni_riprese", None))
    rumore = TA._togli_rumore(a_)              # integrazione: il rumore del gradiente (A21) e' solo registro
    stati = a_ == b_ and rumore and tieni == (1, 2)
    ciclo = th and adam and gruppi and stati and nuovi == {"allena_f32": [], "lr_f32": 1e-5, "ema_theta": 0.0,
                                                           "heldout_righe_rapide": 0, "heldout_pieno_ogni": 0,
                                                           **TA._CFG_VBOP_DEFAULT, **TA._CFG_PESO_DEFAULT,
                                                           **TA._CFG_MC_DEFAULT,
                                                           # 27/9 --lora-nonesperti (ai default: spenta)
                                                           "lora_rango": 0, "lora_alpha": None, "lora_famiglie": [],
                                                           "lr_lora": 1e-4, "lora_tipi": [],
                                                           **TA._CFG_SOPHIA_DEFAULT, **TA._CFG_LORA_GA_DEFAULT,
                                                           **TA._CFG_INTEGRATA_PROPRI, **TA._CFG_BF16X3_DEFAULT, **TA._CFG_LOTA_DEFAULT}
    ok &= al_file and conteggi and ciclo
    return registra("A23_default_al_bit", ok, ricette=det, valori_export_al_file=al_file, conteggi=conteggi,
                    f32_conteggi=sa.f32_conteggi, ciclo_theta=th, ciclo_adam=adam, ciclo_gruppi=gruppi,
                    ciclo_stato=stati, config_nuovi=nuovi, base=BASE, secondi=round(time.time() - t0, 1))


# ─────────────────────────────── A24 ───────────────────────────────
@contextmanager
def tutto_fp64():
    """HF lancia in fp32 norme, GDN e router (`.float()`, `.to(torch.float32)`): per le differenze finite in
    float64 quei lanci restano float64 (solo per tensori gia' float64)."""
    to0, float0 = torch.Tensor.to, torch.Tensor.float

    def to(self, *a, **k):
        if self.dtype == torch.float64:
            a = tuple(torch.float64 if x is torch.float32 else x for x in a)
            if k.get("dtype") is torch.float32:
                k["dtype"] = torch.float64
        return to0(self, *a, **k)

    def fl(self, *a, **k):
        return self if self.dtype == torch.float64 else float0(self, *a, **k)
    torch.Tensor.to, torch.Tensor.float = to, fl
    try:
        yield
    finally:
        torch.Tensor.to, torch.Tensor.float = to0, float0


def prova_a24() -> bool:
    t0 = time.time()
    st = studente("A64", TUTTE)                       # costruito fuori: la verifica al bit col caricatore e' fp32
    _theta_casuale(st)
    with tutto_fp64():
        ids, x, R = _ingresso(st, T=16)

        def perdita_nograd() -> float:
            with torch.no_grad():
                out = st.avanti(ids, x0=x)
                return float((st.esce(out["xL"]).to(torch.float64) * R).sum()), [o.clone() for o in out["instr"]]
        _, _, _, instr0 = _passo(st, ids, x, R)
        g = {n: t.master.grad.clone() for n, t in st.f32.items()}
        st.azzera_cambi()
        gen = torch.Generator().manual_seed(11)
        per_tensore, peggiore, cambi = {}, 0.0, 0
        for n, tf in st.f32.items():
            m = tf.master
            idx = torch.randperm(m.numel(), generator=gen)[:3]
            num, den = 0.0, 0.0
            for i in idx.tolist():
                v0 = m.detach().reshape(-1)[i].item()
                h = 1e-6 * max(1.0, abs(v0))
                vals = []
                for s in (+1, -1):
                    with torch.no_grad():
                        m.view(-1)[i] = v0 + s * h
                    f, ins = perdita_nograd()
                    cambi += sum(int((torch.sort(a, -1).values != torch.sort(b, -1).values).any(-1).sum())
                                 for a, b in zip(ins, instr0))
                    vals.append(f)
                with torch.no_grad():
                    m.view(-1)[i] = v0
                fd = (vals[0] - vals[1]) / (2 * h)
                an = g[n].reshape(-1)[i].item()
                num = max(num, abs(fd - an))
                den = max(den, abs(an), abs(fd))
            e = num / max(den, 1e-12)
            per_tensore[n] = {"err_rel": e, "max_abs_grad": den}
            peggiore = max(peggiore, e)
        nonnulli = all(float(t.master.grad.abs().max()) > 0 for t in st.f32.values())
    ok = peggiore < 1e-5 and cambi == 0 and nonnulli and len(per_tensore) == 17
    return registra("A24_gradiente_f32_vs_differenze_finite", ok, err_rel_peggiore=peggiore, cambi_instradamento=cambi,
                    gradienti_non_nulli=nonnulli, per_tensore=per_tensore, secondi=round(time.time() - t0, 1))


# ─────────────────────────────── A25 ───────────────────────────────
def sorgente_minuscola(n_train: int = 12, n_held: int = 4, T: int = 16, seme: int = 5):
    """Maestro = lo studente minuscolo con theta* e F32* (norme, router e GDN spostati): la KL a theta = 0 e
    F32 del file e' > 0 e il ciclo deve muovere entrambi."""
    st = studente("A32", TUTTE)
    g = torch.Generator().manual_seed(seme)
    with torch.no_grad():
        for p in st.parametri():
            p.copy_(0.2 * torch.randn(p.shape, generator=g))
        for t in st.f32.values():
            t.master.mul_(1.0 + 0.05 * torch.randn(t.master.shape, generator=g, dtype=t.master.dtype))
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
                ripresa_ogni=1, pezzo_kl=10, attesa_ogni_s=0.01, lr_f32=3e-3)
    base.update(kw)
    return A.ConfigAllena(**base)


def _uguali_adam(ra, rb) -> bool:
    sa, sb = ra["opt"]["state"], rb["opt"]["state"]
    return sa.keys() == sb.keys() and all(torch.equal(sa[i][k], sb[i][k]) for i in sa
                                          for k in ("exp_avg", "exp_avg_sq", "step")) \
        and ra["opt"]["param_groups"] == rb["opt"]["param_groups"]


def prova_a25() -> bool:
    import e2e_test_allena as TA
    t0 = time.time()
    src, WS = sorgente_minuscola()
    fam = S.famiglie_f32(TUTTE)
    ca = _ca(allena_f32=fam)
    d_f, d_r, d_x = DIR / "a25_filata", DIR / "a25_ripresa", DIR / "a25_aggiunta"
    for d in (d_f, d_r, d_x):
        shutil.rmtree(d, ignore_errors=True)
    st_f = studente("A32", fam)
    filata = A.allena(st_f, src, d_f, ca, WS, WS, log=muto)
    A.allena(studente("A32", fam), src, d_r, ca, WS, WS, ferma_a=3, log=muto)
    st_b = studente("A32", fam)
    ripresa = A.allena(st_b, src, d_r, ca, WS, WS, riprendi=True, log=muto)
    rf, rr = A.carica_ripresa(d_f, "cpu"), A.carica_ripresa(d_r, "cpu")
    th = all(torch.equal(a, b) for a, b in zip(st_f.parametri(), st_b.parametri()))
    f32 = all(torch.equal(a, b) for a, b in zip(st_f.parametri_f32(), st_b.parametri_f32()))
    orig = studente("A32", fam)
    mossi = sum(1 for n, t in st_f.f32.items() if not torch.equal(t.master, orig.f32[n].master))
    adam = _uguali_adam(rf, rr) and len(rf["opt"]["param_groups"]) == 2 \
        and rf["opt"]["param_groups"][1]["lr"] == ca.lr_f32 and rf["f32_nomi"] == st_f.chiavi_f32()
    stato = TA._senza_tempi(filata) == TA._senza_tempi(ripresa) and filata["passo"] == 6
    cambi = filata["cambi_instradamento"] == 0 and ripresa["cambi_instradamento"] == 0
    # istantanea del migliore: f32.npz == valori da esportare del migliore (ricalcolo: e' il passo del migliore)
    ist = A.carica_istantanea(d_f / filata["migliore"]["dir"])
    ist_ok = ("f32" in ist and sorted(ist["f32"]) == sorted(st_f.chiavi_f32())
              and all(v.dtype == np.float32 and v.shape == st_f.f32[n].forma_file for n, v in ist["f32"].items()))
    if filata["migliore"]["passo"] == filata["passo"]:
        ist_ok &= all(np.array_equal(v, ist["f32"][n]) for n, v in st_f.f32_per_export().items())
    # famiglie aggiunte alla ripresa; lr_f32 cambiato; famiglia tolta = errore
    poche = S.famiglie_f32("attn_norm,ffn_gate_inp")
    A.allena(studente("A32", poche), src, d_x, _ca(allena_f32=poche), WS, WS, ferma_a=2, log=muto)
    r2 = A.carica_ripresa(d_x, "cpu")
    reg = []
    piu = S.famiglie_f32("attn_norm,ffn_gate_inp,ssm_dt,output_norm")
    st_x = studente("A32", piu)
    s_x = A.allena(st_x, src, d_x, _ca(allena_f32=piu, lr_f32=1e-3), WS, WS, riprendi=True, ferma_a=3,
                   log=reg.append)
    r3 = A.carica_ripresa(d_x, "cpu")
    n_theta = len(st_x.parametri())
    vecchi = {n: i for i, n in enumerate(r2["f32_nomi"])}
    passi_ok = all(int(r3["opt"]["state"][n_theta + j]["step"]) ==
                   (int(r2["opt"]["state"][n_theta + vecchi[n]]["step"]) + 1 if n in vecchi else 1)
                   for j, n in enumerate(r3["f32_nomi"]))
    aggiunta = (s_x["passo"] == 3 and r3["opt"]["param_groups"][1]["lr"] == 1e-3 and passi_ok
                and any("tensori F32 nuovi" in x for x in reg) and any("lr_f32" in x and "0.001" in x for x in reg)
                and any("allena_f32" in x for x in reg) and s_x["config"]["allena_f32"] == list(piu))
    tolta = False
    try:
        A.allena(studente("A32", poche), src, d_x, _ca(allena_f32=poche), WS, WS, riprendi=True, ferma_a=4, log=muto)
    except RuntimeError as e:
        tolta = "tolti" in str(e)
    senza = False
    try:
        A.allena(studente("A32"), src, d_x, _ca(), WS, WS, riprendi=True, ferma_a=4, log=muto)
    except RuntimeError as e:
        senza = "senza --allena-f32" in str(e) or "tolti" in str(e)
    disallineato = False
    try:
        A.allena(studente("A32"), src, DIR / "a25_mai", _ca(allena_f32=poche), WS, WS, log=muto)
    except ValueError:
        disallineato = True
    finito = False
    st_n = studente("A32", poche)
    with torch.no_grad():
        next(iter(st_n.f32.values())).master.view(-1)[0] = float("nan")
    try:
        st_n.controlla()
    except FloatingPointError:
        finito = True
    # curvatura del passo V: con `f32_fermi` i master non raccolgono gradiente (e dopo tornano a raccoglierlo)
    ids, x, R = _ingresso(st_n, T=8)
    with torch.no_grad():
        next(iter(st_n.f32.values())).master.view(-1)[0] = 1.0
    with st_n.f32_fermi():
        _passo(st_n, ids, x, R)
    fermi = all(p.grad is None for p in st_n.parametri_f32()) and all(p.requires_grad for p in st_n.parametri_f32())
    cli = (A._config_da_arg(_arg(allena_f32="ssm_dt.bias,attn_norm", lr_f32=2e-5)).allena_f32 == ("attn_norm", "ssm_dt")
           and A._config_da_arg(_arg()).allena_f32 == () and {"allena_f32", "lr_f32"} <= A._CAMPI_DI_DECISIONE)
    ok = th and f32 and mossi > 10 and adam and stato and cambi and ist_ok and aggiunta and tolta and senza \
        and disallineato and finito and fermi and cli
    return registra("A25_ripresa_al_bit_con_f32", ok, theta=th, f32=f32, f32_mossi=mossi, n_f32=len(st_f.f32),
                    adam_due_gruppi=adam, stato=stato, cambi_instradamento=cambi, istantanea=ist_ok,
                    aggiunta_alla_ripresa=aggiunta, tolta_solleva=tolta, ripresa_senza_f32_solleva=senza,
                    studente_config_diversi_solleva=disallineato, nan_solleva=finito, f32_fermi=fermi, cli=cli,
                    heldout=[(x["passo"], round(x["metrica"], 6)) for x in filata["storia_heldout"]],
                    secondi=round(time.time() - t0, 1))


def _arg(**kw):
    from types import SimpleNamespace
    a = dict(lr=1e-3, righe_passo=2, righe_micro=1, righe_micro_valuta=2, epoche=1, heldout_ogni=2, ripresa_ogni=2,
             pezzo_kl=10, rotazione=False, K0=None, pv_codici=False, pv_rho=A.ConfigAllena.pv_rho,
             pv_micro_curvatura=A.ConfigAllena.pv_micro_curvatura, pv_n_proposta=A.ConfigAllena.pv_n_proposta,
             pv_q_fdr=A.ConfigAllena.pv_q_fdr, pv_q_guadagno=A.ConfigAllena.pv_q_guadagno,
             pv_passi_accumulo=A.ConfigAllena.pv_passi_accumulo, perdita="kl", gamma=None, escludi_domini="wiki",
             allena_f32="", lr_f32=A.ConfigAllena.lr_f32, beta2=A.ConfigAllena.beta2, ema_theta=0.0,
             heldout_righe_rapide=0, heldout_pieno_ogni=0, pv_modo=A.ConfigAllena.pv_modo,
             pv_gamma=A.ConfigAllena.pv_gamma, pv_curvatura_ogni=A.ConfigAllena.pv_curvatura_ogni,
             pv_rho_adattivo=A.ConfigAllena.pv_rho_adattivo, pv_rho_max=A.ConfigAllena.pv_rho_max)
    a.update(kw)
    return SimpleNamespace(**a)


# ─────────────────────────────── A26 ───────────────────────────────
def prova_a26() -> bool:
    t0 = time.time()
    src = gguf_sintetico()
    st = studente("A32", TUTTE)
    _theta_casuale(st, seme=4, s=0.1)
    g = torch.Generator().manual_seed(9)
    with torch.no_grad():
        for t in st.f32.values():
            t.master.add_(0.01 * torch.randn(t.master.shape, generator=g, dtype=t.master.dtype) *
                          t.master.abs().mean())
    scale = st.scale_per_export()
    f32 = st.f32_per_export()
    theta = {A._chiave(k): p.detach().clone() for k, p in zip(st.chiavi(), st.parametri())}
    dst = DIR / "a26_esportato.gguf"
    for p in (dst, Path(str(dst) + ".provenienza.json"), dst.with_name(dst.name + ".parziale")):
        p.unlink(missing_ok=True)
    ric0 = X.ricarica
    X.ricarica = functools.partial(ric0, ricetta="A32", token=24)
    try:
        prov = X.esporta(src, dst, scale, {"passo": 1}, theta=theta, strati_ricarica=(0, 1), controlla_disco=False,
                         log=muto, f32=f32)
    finally:
        X.ricarica = ric0
    ric = prov["ricarica"]
    ricarica_ok = all(r["scale_al_bit"] and r["uscita_al_bit"] and r.get("output_norm_al_bit") for r in ric.values())
    # verifica indipendente, tensore per tensore
    ra, rb = GGUFReader(str(src)), GGUFReader(str(dst))
    ta, tb = {t.name: t for t in ra.tensors}, {t.name: t for t in rb.tensors}
    primo = min(int(t.data_offset) for t in ra.tensors)
    fa, fb = np.memmap(src, dtype=np.uint8, mode="r"), np.memmap(dst, dtype=np.uint8, mode="r")
    intest = src.stat().st_size == dst.stat().st_size and np.array_equal(fa[:primo], fb[:primo])
    tens_ok, cambiati = True, 0
    for n, a in ta.items():
        b = tb[n]
        va, vb = np.asarray(a.data), np.asarray(b.data)
        if n in f32:
            tens_ok &= np.array_equal(vb.view(np.uint32), f32[n].view(np.uint32))
            cambiati += int((va.view(np.uint32) != vb.view(np.uint32)).sum())
        elif n in scale:
            nb, _, off, k = S.CS.LAYOUT[a.tensor_type.name]
            xa, xb = va.reshape(-1, nb).copy(), vb.reshape(-1, nb).copy()
            xa[:, off:off + 2 * k] = 0
            xb[:, off:off + 2 * k] = 0
            tens_ok &= np.array_equal(xa, xb)
        else:
            tens_ok &= np.array_equal(va, vb)
    # fine del file (dopo l'ultimo tensore) identica
    ultimo = max(int(t.data_offset) + int(t.n_bytes) for t in ra.tensors)
    tens_ok &= np.array_equal(fa[ultimo:], fb[ultimo:])
    del ra, rb, ta, tb, fa, fb
    # studente SOLO dal file esportato (senza --allena-f32) == studente del sorgente con F32 e theta: tutti gli strati
    sf = S.Studente(gguf=dst, strati=[0, 1], ricetta="A32", motore_nome="torch", device="cpu", log=muto)
    ids, x, R = _ingresso(sf, seme=8)
    with torch.no_grad():                    # sf: theta = 0, scale del file == le fp16 esatte di st
        ya = st.esce(st.avanti(ids, x0=x)["xL"])
        yb = sf.esce(sf.avanti(ids, x0=x)["xL"])
    modello = bool(torch.equal(ya, yb))
    # errori scoperti
    errori = {}
    for nome, f in (("non_f32", {"blk.0.ffn_gate_exps.weight": np.zeros(4, np.float32)}),
                    ("forma", {"blk.0.attn_norm.weight": np.zeros(3, np.float32)}),
                    ("nan", {"output_norm.weight": np.full(512, np.nan, np.float32)}),
                    ("dtype", {"output_norm.weight": np.ones(512, np.float64)})):
        d2 = DIR / f"a26_{nome}.gguf"
        d2.unlink(missing_ok=True)
        try:
            X.esporta(src, d2, scale, {}, controlla_disco=False, log=muto, f32=f)
            errori[nome] = False
        except ValueError:
            errori[nome] = not d2.exists() and not d2.with_name(d2.name + ".parziale").exists()
    # manomissione fuori dalle regioni F32 dopo la riscrittura: verifica_f32 la scopre
    d3 = DIR / "a26_manomesso.gguf"
    shutil.copyfile(dst, d3)
    prima = X.impronta_fuori_f32(d3, f32)
    X.patcha_f32(d3, f32)
    mm = np.memmap(d3, dtype=np.uint8, mode="r+")
    mm[-1] ^= 1
    mm.flush()
    del mm
    try:
        X.verifica_f32(d3, f32, prima)
        errori["manomissione"] = False
    except RuntimeError:
        errori["manomissione"] = True
    d3.unlink()
    pf = prov.get("tensori_f32", {})
    prov_ok = pf.get("verifica", {}).get("tensori") == len(f32) and sum(pf["valori_cambiati_per_tensore"].values()) \
        == cambiati and cambiati > 0
    ok = ricarica_ok and intest and tens_ok and modello and all(errori.values()) and prov_ok
    return registra("A26_export_f32_al_bit", ok, ricarica=ric, intestazione=intest, tensori=tens_ok,
                    valori_f32_cambiati=cambiati, modello_intero_al_bit=modello, errori_scoperti=errori,
                    provenienza=pf, secondi=round(time.time() - t0, 1))


# ─────────────────────────────── A27 ───────────────────────────────
def prova_a27() -> bool:
    """Integrazione (allena-f32 + velocita', 27/9): --allena-f32 con --beta2 0,99 e --ema-theta 0,5. (a) 3 passi +
    ripresa + 3 == 6 filati AL BIT (theta, master F32, theta_EMA, Adam dei due gruppi, stato); betas (0,9, 0,99) su
    ENTRAMBI i gruppi. (b) EMA solo su theta: migliore_<p>/ema/ e migliore_ema_<p>/ hanno f32.npz == i F32 del
    migliore (quelli correnti con cui si e' misurata theta_EMA). (c) beta2 cambiato alla ripresa: applicato a
    entrambi i gruppi e registrato. (d) E2E_GDN=fla con ssm_a/ssm_dt/ssm_alpha solleva; veloce no. (e) CLI."""
    import e2e_test_allena as TA
    t0 = time.time()
    src, WS = sorgente_minuscola()
    fam = S.famiglie_f32("attn_norm,ffn_gate_inp,ssm_dt")
    ca = _ca(allena_f32=fam, beta2=0.99, ema_theta=0.5)
    d_f, d_r = DIR / "a27_filata", DIR / "a27_ripresa"
    for d in (d_f, d_r):
        shutil.rmtree(d, ignore_errors=True)
    st_f = studente("A32", fam)
    filata = A.allena(st_f, src, d_f, ca, WS, WS, log=muto)
    A.allena(studente("A32", fam), src, d_r, ca, WS, WS, ferma_a=3, log=muto)
    st_b = studente("A32", fam)
    ripresa = A.allena(st_b, src, d_r, ca, WS, WS, riprendi=True, log=muto)
    rf, rr = A.carica_ripresa(d_f, "cpu"), A.carica_ripresa(d_r, "cpu")
    th = all(torch.equal(a, b) for a, b in zip(st_f.parametri(), st_b.parametri()))
    f32 = all(torch.equal(a, b) for a, b in zip(st_f.parametri_f32(), st_b.parametri_f32()))
    ema = rf.get("ema") is not None and rf["ema"].keys() == rr["ema"].keys() \
        and all(torch.equal(rf["ema"][k], rr["ema"][k]) for k in rf["ema"])
    betas = [tuple(g["betas"]) for g in rf["opt"]["param_groups"]]
    adam = _uguali_adam(rf, rr) and betas == [(0.9, 0.99), (0.9, 0.99)]
    stato = TA._senza_tempi(filata) == TA._senza_tempi(ripresa) and filata["passo"] == 6
    ok_a = th and f32 and ema and adam and stato
    # (b)
    dm = d_f / filata["migliore"]["dir"]
    im, ie = A.carica_istantanea(dm), A.carica_istantanea(dm / "ema")
    de = d_f / filata["migliore_ema"]["dir"]
    ime = A.carica_istantanea(de)
    ok_b = ("f32" in im and "f32" in ie and "f32" in ime and sorted(ie["f32"]) == sorted(st_f.chiavi_f32())
            and all(np.array_equal(im["f32"][n], ie["f32"][n]) for n in im["f32"]))
    if filata["migliore_ema"]["passo"] == filata["passo"]:
        ok_b &= all(np.array_equal(v, ime["f32"][n]) for n, v in st_f.f32_per_export().items())
    # (c)
    reg, d_c = [], DIR / "a27_beta2"
    shutil.rmtree(d_c, ignore_errors=True)
    A.allena(studente("A32", fam), src, d_c, ca, WS, WS, ferma_a=3, log=muto)
    A.allena(studente("A32", fam), src, d_c, _ca(allena_f32=fam, beta2=0.999, ema_theta=0.5), WS, WS, riprendi=True,
             ferma_a=5, log=reg.append)
    r7 = A.carica_ripresa(d_c, "cpu")
    ok_c = ([tuple(g["betas"]) for g in r7["opt"]["param_groups"]] == [(0.9, 0.999), (0.9, 0.999)]
            and any(x.startswith("ripresa: betas_f32") for x in reg) and any(x.startswith("ripresa: betas ") for x in reg))
    # (d)
    rotta = False
    try:
        A.controlla_gdn_f32("fla", fam)
    except ValueError:
        rotta = True
    ok_d = rotta and A.controlla_gdn_f32("veloce", fam) is None and A.controlla_gdn_f32("fla", ("attn_norm",)) is None
    # (e)
    c = A._config_da_arg(_arg(allena_f32="attn_norm", lr_f32=2e-5, beta2=0.99, ema_theta=0.5,
                              heldout_righe_rapide=2, heldout_pieno_ogni=4))
    ok_e = (c.allena_f32, c.lr_f32, c.beta2, c.ema_theta, c.heldout_righe_rapide, c.heldout_pieno_ogni) == \
        (("attn_norm",), 2e-5, 0.99, 0.5, 2, 4) and {"beta2", "ema_theta", "allena_f32", "lr_f32"} <= A._CAMPI_DI_DECISIONE
    ok = ok_a and ok_b and ok_c and ok_d and ok_e
    return registra("A27_f32_con_beta2_ema", ok, theta=th, f32=f32, ema=ema, adam=adam, betas=betas, stato=stato,
                    a=ok_a, b_ema_f32=ok_b, c_beta2_ripresa=ok_c, d_gdn_fla=ok_d, e_cli=ok_e,
                    migliore=filata["migliore"]["dir"], migliore_ema=filata["migliore_ema"]["dir"],
                    secondi=round(time.time() - t0, 1))


# ─────────────────────────────── A28 ───────────────────────────────
def _uguali_master(st, rif: dict, nomi) -> bool:
    return all(torch.equal(st.f32[n].master.detach().cpu(), torch.as_tensor(rif[n])) for n in nomi)


def _passi_adam(r: dict, nomi_f32: list) -> dict:
    """{nome F32: step di Adam} dalla ripresa `r` (AdamW: gruppo theta poi F32; 0 = senza stato)."""
    n_theta = len(r["theta"])
    sd = r["opt"]["adam"] if r["opt"].get("tipo") == "sophia" else r["opt"]
    off = 0 if r["opt"].get("tipo") == "sophia" else n_theta
    return {n: int(sd["state"].get(off + j, {}).get("step", 0)) for j, n in enumerate(nomi_f32)}


def prova_a28() -> bool:
    """28/9 --f32-ripristina (analisi_strumenti.md): famiglie F32 riportate al valore del GGUF alla ripresa.
    (1) TOLTE da --allena-f32 e riportate: la ripresa passa (senza l'opzione solleva), l'avanti dello studente ripreso ==
        quello di uno studente con TUTTE le famiglie ai valori della ripresa salvo quelle riportate al GGUF (al bit), i
        master rimasti == ripresa; il passo dopo e un secondo rilancio con la stessa riga vanno (idempotente: una voce sola
        in f32_ripristini, Adam dei rimasti che continua);
    (2) ANCORA allenate e riportate: master == GGUF alla ripresa, Adam da zero (step 1 dopo un passo, gli altri n+1);
        rilanciando con la stessa riga NON si riportano di nuovo (step 2, valori della ripresa);
    (3) strati congelati (--strati-allenati 1): la famiglia riportata torna al GGUF anche nello strato 0 e ci resta;
    (4) Sophia (ramo newton-theta): la stessa ripresa (1) col gruppo AdamW di F32 dentro l'ottimizzatore di Sophia;
    (5) riga di comando, campo di decisione, famiglia sconosciuta rifiutata, corsa nuova = nessun effetto."""
    t0 = time.time()
    src, WS = sorgente_minuscola()
    fam = S.famiglie_f32(TUTTE)
    rip = S.famiglie_f32("ffn_gate_inp,ffn_gate_inp_shexp,ssm_alpha,ssm_beta,ssm_conv1d")
    resto = tuple(f for f in fam if f not in rip)
    esito: dict = {}

    def base(nome: str, sorgente=None, testa=None, **kw) -> Path:
        d = DIR / nome
        shutil.rmtree(d, ignore_errors=True)
        W = WS if testa is None else testa
        A.allena(studente("A32", fam), sorgente or src, d, _ca(allena_f32=fam, **kw), W, W, ferma_a=3, log=muto)
        return d

    def copia(d0: Path, nome: str) -> Path:
        d = DIR / nome
        shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(d0, d)
        return d

    gg = studente("A32", fam).valori_f32()                       # master nati dal GGUF
    d0 = base("a28_base")
    r3 = A.carica_ripresa(d0, "cpu")
    n_rip = [n for n in r3["f32_nomi"] if S.famiglia_del_tensore_f32(n) in rip]
    n_resto = [n for n in r3["f32_nomi"] if n not in n_rip]
    esito["allenati_nella_base"] = sum(1 for n in n_rip if not torch.equal(r3["f32"][n], gg[n]))

    # (1) tolte e riportate
    d1 = copia(d0, "a28_tolte")
    senza = False
    try:
        A.allena(studente("A32", resto), src, d1, _ca(allena_f32=resto), WS, WS, riprendi=True, ferma_a=3, log=muto)
    except RuntimeError as e:
        senza = "tolti" in str(e) and "--f32-ripristina" in str(e)
    reg: list = []
    st1 = studente("A32", resto)
    s1 = A.allena(st1, src, d1, _ca(allena_f32=resto, f32_ripristina=rip), WS, WS, riprendi=True, ferma_a=3,
                  log=reg.append)
    rif = studente("A32", fam)
    with torch.no_grad():
        for k, p in zip(rif.chiavi(), rif.parametri()):
            p.copy_(r3["theta"][A._chiave(k)])
    rif.imposta_f32({n: r3["f32"][n] for n in n_resto})
    ids, x, _ = _ingresso(rif)
    with torch.no_grad():
        avanti = torch.equal(st1.avanti(ids, x0=x)["xL"], rif.avanti(ids, x0=x)["xL"])
    rimasti = _uguali_master(st1, r3["f32"], n_resto) and sorted(st1.chiavi_f32()) == sorted(n_resto)
    voce = (s1.get("f32_ripristini") or [{}])[-1]
    voce_ok = (len(s1.get("f32_ripristini", [])) == 1 and voce.get("famiglie") == sorted(rip)
               and voce.get("tolti") == len(n_rip) and voce.get("ancora_allenati") == 0
               and all(x["relativa"] and x["relativa"] > 0 for x in voce["scartato_per_famiglia"].values()))
    log_ok = any("--f32-ripristina:" in x and "TOLTI" in x for x in reg) and any("tolti e riportati al GGUF" in x
                                                                                  for x in reg)
    s1b = A.allena(studente("A32", resto), src, d1, _ca(allena_f32=resto, f32_ripristina=rip), WS, WS, riprendi=True,
                   ferma_a=4, log=muto)
    r4 = A.carica_ripresa(d1, "cpu")
    reg_c: list = []
    s1c = A.allena(studente("A32", resto), src, d1, _ca(allena_f32=resto, f32_ripristina=rip), WS, WS, riprendi=True,
                   ferma_a=5, log=reg_c.append)
    r5 = A.carica_ripresa(d1, "cpu")
    p3, p4, p5 = _passi_adam(r3, r3["f32_nomi"]), _passi_adam(r4, r4["f32_nomi"]), _passi_adam(r5, r5["f32_nomi"])
    seguito = (s1b["passo"] == 4 and s1c["passo"] == 5 and r4["f32_nomi"] == [n for n in r3["f32_nomi"] if n in n_resto]
               and all(p4[n] == p3[n] + 1 and p5[n] == p3[n] + 2 for n in n_resto)
               and len(s1c.get("f32_ripristini", [])) == 1 and not any("--f32-ripristina:" in x for x in reg_c))
    esito["tolte"] = dict(senza_opzione_solleva=senza, avanti_al_bit=avanti, rimasti=rimasti, voce=voce_ok,
                          registro=log_ok, passi_dopo_e_idempotente=seguito)

    # (2) ancora allenate e riportate
    d2 = copia(d0, "a28_tenute")
    reg2: list = []
    st2 = studente("A32", fam)
    s2 = A.allena(st2, src, d2, _ca(allena_f32=fam, f32_ripristina=rip), WS, WS, riprendi=True, ferma_a=3,
                  log=reg2.append)
    al_gguf = _uguali_master(st2, gg, n_rip) and _uguali_master(st2, r3["f32"], n_resto)
    voce2 = (s2.get("f32_ripristini") or [{}])[-1]
    A.allena(studente("A32", fam), src, d2, _ca(allena_f32=fam, f32_ripristina=rip), WS, WS, riprendi=True, ferma_a=4,
             log=muto)
    q4 = A.carica_ripresa(d2, "cpu")
    reg2c: list = []
    st2c = studente("A32", fam)
    A.allena(st2c, src, d2, _ca(allena_f32=fam, f32_ripristina=rip), WS, WS, riprendi=True, ferma_a=4, log=reg2c.append)
    non_di_nuovo = _uguali_master(st2c, q4["f32"], q4["f32_nomi"]) and not any("--f32-ripristina:" in x for x in reg2c)
    A.allena(studente("A32", fam), src, d2, _ca(allena_f32=fam, f32_ripristina=rip), WS, WS, riprendi=True, ferma_a=5,
             log=muto)
    q5 = A.carica_ripresa(d2, "cpu")
    a4, a5 = _passi_adam(q4, q4["f32_nomi"]), _passi_adam(q5, q5["f32_nomi"])
    mossi = sum(1 for n in n_rip if not torch.equal(q4["f32"][n], gg[n]))
    adam = (all(a4[n] == 1 and a5[n] == 2 for n in n_rip) and all(a4[n] == p3[n] + 1 and a5[n] == p3[n] + 2
                                                                    for n in n_resto))
    esito["tenute"] = dict(al_gguf_alla_ripresa=al_gguf, voce=voce2.get("ancora_allenati") == len(n_rip)
                           and voce2.get("tolti") == 0, adam_da_zero=adam, riallenate=mossi == len(n_rip),
                           rilancio_non_riporta=non_di_nuovo)

    # (3) strati congelati: attn_norm riportata e ancora allenata, solo lo strato 1 allenato
    d3 = copia(d0, "a28_congelati")
    un = S.famiglie_f32("attn_norm")
    A.allena(studente("A32", fam), src, d3, _ca(allena_f32=fam, f32_ripristina=un, strati_allenati=(1,)), WS, WS,
             riprendi=True, ferma_a=4, log=muto)
    c4 = A.carica_ripresa(d3, "cpu")
    n0, n1 = "blk.0.attn_norm.weight", "blk.1.attn_norm.weight"
    congelati = (torch.equal(c4["f32"][n0], gg[n0]) and not torch.equal(r3["f32"][n0], gg[n0])
                 and not torch.equal(c4["f32"][n1], gg[n1]) and _passi_adam(c4, c4["f32_nomi"])[n1] == 1)
    esito["strati_congelati"] = congelati

    # (4) Sophia: la ripresa (1) con il gruppo AdamW dei F32 dentro l'ottimizzatore di Sophia
    # (Sophia misura la curvatura nei micro-lotti del passo V: PV acceso, 4 righe a passo, sorgente piu' lunga)
    so = dict(ottimizzatore_theta="sophia", sophia_gamma=300.0, pv_codici=True, pv_rho=0.05, pv_micro_curvatura=1,
              righe_per_passo=4)
    src4, WS4 = sorgente_minuscola(n_train=24)
    d4 = base("a28_sophia", sorgente=src4, testa=WS4, **so)
    rs = A.carica_ripresa(d4, "cpu")
    st4 = studente("A32", resto)
    s4 = A.allena(st4, src4, d4, _ca(allena_f32=resto, f32_ripristina=rip, **so), WS4, WS4, riprendi=True, ferma_a=4,
                  log=muto)
    rs4 = A.carica_ripresa(d4, "cpu")
    ps, ps4 = _passi_adam(rs, rs["f32_nomi"]), _passi_adam(rs4, rs4["f32_nomi"])
    sophia = (s4["passo"] == 4 and rs4["opt"].get("tipo") == "sophia" and sorted(rs4["f32_nomi"]) == sorted(n_resto)
              and all(ps4[n] == ps[n] + 1 for n in n_resto) and len(s4.get("f32_ripristini", [])) == 1)
    esito["sophia"] = sophia

    # (5) riga di comando e config
    cli = (A._config_da_arg(_arg(f32_ripristina="ssm_alpha.weight,ffn_gate_inp")).f32_ripristina
           == ("ffn_gate_inp", "ssm_alpha") and A._config_da_arg(_arg()).f32_ripristina == ()
           and "f32_ripristina" in A._CAMPI_DI_DECISIONE and A._CONFIG_DEFAULT.get("f32_ripristina") == [])
    ignota = False
    try:
        A._config_da_arg(_arg(f32_ripristina="router"))
    except ValueError:
        ignota = True
    reg5: list = []
    d5 = DIR / "a28_nuova"
    shutil.rmtree(d5, ignore_errors=True)
    A.allena(studente("A32", fam), src, d5, _ca(allena_f32=fam, f32_ripristina=rip), WS, WS, ferma_a=0, log=reg5.append)
    nuova = any("senza effetto" in x for x in reg5)
    esito["riga_di_comando"] = dict(cli=cli, famiglia_ignota_rifiutata=ignota, corsa_nuova_senza_effetto=nuova)

    ok = (esito["allenati_nella_base"] == len(n_rip) and all(esito["tolte"].values()) and all(esito["tenute"].values())
          and congelati and sophia and all(esito["riga_di_comando"].values()))
    return registra("A28_f32_ripristina", ok, **esito, n_riportati=len(n_rip), secondi=round(time.time() - t0, 1))


PROVE = {"A23": prova_a23, "A24": prova_a24, "A25": prova_a25, "A26": prova_a26, "A27": prova_a27, "A28": prova_a28}


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
    (DIR / f"esito_test_f32{suff}.json").write_text(json.dumps(ESITO, indent=1, default=str))
    print(f"ESITO: {'TUTTO VERDE' if ok else 'ROSSO'} ({ESITO['secondi']} s)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

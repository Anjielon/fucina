#!/usr/bin/env python3
"""VELOCE2 14/10 — prove CPU delle leve di ricerche/C_S23_VELOCE2_1014.md, in aggiunta a s23_test.py (33) e
s23_test_veloce.py (7). Tutte AL BIT:
  V3  esperti_senza_sync == Qwen3_5MoeExperts.forward di transformers (bf16 e fp32, esperti vuoti compresi)
  V1  gancio «hmid» = ingresso di post_attention_layernorm; coda_strato(x_mid) == _strato(x) e ganci ffn/down/a_ord/piano
      uguali, sullo STUDENTE VERO (GGUF della partenza, CPU, motore torch, ricetta B3_ind32) in uno strato GDN (0) e in
      uno ad attenzione piena (3)  [S23_PROVA_GGUF=<gguf>; senza, la prova si salta e lo dice]
  V2  scrivi_tensore / leggi_tensore_in andata e ritorno (fp32, bf16) al bit; impronta_strato cambia se cambia un byte
Stampa OK/KO per riga; codice d'uscita 1 se un KO."""
from __future__ import annotations

import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

QUI = Path(__file__).resolve().parent
for p in (str(QUI), str(QUI.parent / "lib"), str(QUI.parent / "s1b"), str(QUI.parent / "rif")):
    if p not in sys.path:
        sys.path.insert(0, p)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
import s23_passata as SP  # noqa: E402

esiti = []


def prova(nome, cond, info=""):
    esiti.append(bool(cond))
    print(f"{'OK' if cond else 'KO'} {nome} {info}", flush=True)


# ── V3 esperti del maestro senza sincronizzazioni ──
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeExperts  # noqa: E402

g = torch.Generator().manual_seed(20261014)
ok, info = True, []
for dt in (torch.bfloat16, torch.float32):
    cfg = types.SimpleNamespace(num_experts=16, hidden_size=64, moe_intermediate_size=32, hidden_act="silu",
                                _experts_implementation=None)
    ex = Qwen3_5MoeExperts(cfg)
    with torch.no_grad():
        ex.gate_up_proj.copy_(torch.randn(ex.gate_up_proj.shape, generator=g) * 0.1)
        ex.down_proj.copy_(torch.randn(ex.down_proj.shape, generator=g) * 0.1)
    ex = ex.to(dt)
    T = 300
    x = (torch.randn(T, 64, generator=g)).to(dt)
    logit = torch.randn(T, 16, generator=g)
    logit[:, 5] = -1e9                                        # esperto 5 mai scelto: il ramo «vuoto» c'e'
    pr = torch.softmax(logit, -1)
    w, idx = torch.topk(pr, 4, -1)
    w = (w / w.sum(-1, keepdim=True)).to(dt)
    with torch.no_grad():
        a = ex(x, idx, w)
        b = SP.esperti_senza_sync(ex, x, idx, w)
    ok &= torch.equal(a, b)
    info.append(f"{dt}: max |a-b| {float((a.float() - b.float()).abs().max()):.3g}")
prova("V3 esperti_senza_sync == Qwen3_5MoeExperts.forward (al bit, bf16 e fp32, esperto vuoto)", ok, "; ".join(info))

# ── V1 gancio hmid su un modulo giocattolo ──


class Giocattolo(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = torch.nn.Identity()
        self.post_attention_layernorm = torch.nn.LayerNorm(4)
        self.linear_attn = torch.nn.Module()
        self.linear_attn.out_proj = torch.nn.Linear(4, 4)
        self.mlp = torch.nn.Module()
        self.mlp.shared_expert = torch.nn.Module()
        self.mlp.shared_expert.down_proj = torch.nn.Linear(4, 4)

    def forward(self, x):
        h = self.linear_attn.out_proj(self.input_layernorm(x))
        x = x + h
        return self.mlp.shared_expert.down_proj(self.post_attention_layernorm(x))


s_ = Giocattolo()
xx = torch.randn(2, 4, generator=g)
gg = SP.Gancio(s_, False, ("ffn", "hmid"), ferma="ffn")
try:
    s_(xx)
except SP.Ferma:
    pass
finally:
    gg.chiudi()
atteso = xx + s_.linear_attn.out_proj(xx)
prova("V1 gancio hmid = ingresso di post_attention_layernorm, ffn = sua uscita; ganci tolti",
      torch.equal(gg.d["hmid"], atteso) and torch.equal(gg.d["ffn"], s_.post_attention_layernorm(atteso))
      and len(s_.post_attention_layernorm._forward_pre_hooks) == 0 and len(s_.post_attention_layernorm._forward_hooks) == 0)

# ── V1 sullo studente vero ──
GG = os.environ.get("S23_PROVA_GGUF", "")
if GG and Path(GG).exists():
    import e2e_studente as S                                   # PYTHONPATH = thor_v2 di S1 (come s23_test.py)
    st = S.Studente(gguf=Path(GG), strati=range(4), ricetta="B3_ind32", motore_nome="torch", device="cpu",
                    densi_ternari="tutte", log=lambda *a: None)
    ids = torch.randint(0, 150000, (2, 48), generator=g)
    x = st.embedding(ids)
    ctx = st._contesto(2, 48, st.ric.dtype_residuo)
    risultati = []
    with torch.no_grad():
        for L in range(4):
            full = st.cfg.layer_types[L] == "full_attention"
            if L in (0, 3):
                o = st._strato(L, x, ctx)
                gq = SP.Gancio(st.moduli[L], full, ("ffn", "hmid"), "ffn")
                try:
                    st._strato(L, x, ctx)
                except SP.Ferma:
                    pass
                finally:
                    gq.chiudi()
                hm = gq.d["hmid"].clone()
                oc = SP.coda_strato(st, L, hm)
                dd = []
                fam_down = st.esperti(L).fam["down"]
                for da_coda in (False, True):
                    gk = SP.Gancio(st.moduli[L], full, ("ffn", "down", "a_ord"), "a_ord", S, fam_down)
                    try:
                        SP.coda_strato(st, L, hm) if da_coda else st._strato(L, x, ctx)
                    except SP.Ferma:
                        pass
                    finally:
                        gk.chiudi()
                    dd.append(gk.d)
                ganci = all(torch.equal(dd[0][p], dd[1][p]) for p in ("ffn", "down", "a_ord")) and \
                    torch.equal(dd[0]["piano"].ordine, dd[1]["piano"].ordine)
                risultati.append((L, st.cfg.layer_types[L], torch.equal(o, oc), ganci,
                                  float((o - oc).abs().max())))
            x = st._strato(L, x, ctx)
    prova("V1 coda_strato(hmid) == _strato AL BIT e ganci P4 (ffn, down, a_ord, piano) uguali, studente vero CPU, "
          "strato GDN 0 e attenzione piena 3", all(r[2] and r[3] for r in risultati) and len(risultati) == 2,
          f"{risultati}")
else:
    print(f"SALTATA V1 sullo studente vero (S23_PROVA_GGUF non dato o assente: {GG!r})", flush=True)

# ── V2 flussi: andata e ritorno, impronte ──
d = Path(tempfile.mkdtemp(prefix="s23_test_veloce2_"))
try:
    ok = True
    for dt in (torch.float32, torch.bfloat16):
        t = torch.randn(37, 5, 11, generator=g).to(dt)
        f = d / f"t_{str(dt)[6:]}.npy"
        SP.scrivi_tensore(f, t, passo_byte=1000)              # pezzi piccoli: il ciclo a pezzi e' esercitato
        u = torch.zeros_like(t)
        ok &= SP.leggi_tensore_in(f, u, passo_byte=700) and torch.equal(t, u)
        ok &= not SP.leggi_tensore_in(f, torch.zeros(37, 5, 12, dtype=dt))
        if dt == torch.bfloat16:
            ok &= torch.equal(t.view(torch.int16), u.view(torch.int16))
    prova("V2 scrivi_tensore / leggi_tensore_in al bit (fp32, bf16 come int16), forma diversa rifiutata", ok)
    sd = d / "strato_00"
    (sd / "tensori").mkdir(parents=True)
    (sd / "registro.json").write_text('{"a": 1}')
    np.save(sd / "tensori" / "blk.0.x.weight.npy", np.arange(10, dtype=np.uint8))
    i1 = SP.impronta_strato(sd)
    i2 = SP.impronta_strato(sd)
    np.save(sd / "tensori" / "blk.0.x.weight.npy", np.arange(1, 11, dtype=np.uint8))
    i3 = SP.impronta_strato(sd)
    prova("V2 impronta_strato stabile e sensibile a un byte dei tensori", i1 == i2 and i1 != i3)
finally:
    import shutil
    shutil.rmtree(d, ignore_errors=True)

print(f"{sum(esiti)}/{len(esiti)} verdi", flush=True)
sys.exit(0 if all(esiti) else 1)

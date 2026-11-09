#!/usr/bin/env python3
"""test_rilq_thor_shexp.py — 8/10: rilq_thor.py con --famiglie attn,gdn,shexp su uno studente SINTETICO (CPU).

Lo studente finto ha la stessa topologia dei moduli di e2e_studente (self_attn.q/k/v/o_proj negli strati full_attention,
linear_attn.in_proj_qkv/in_proj_z/out_proj negli altri, mlp.shared_expert.gate/up/down_proj in OGNI strato, H = 256 e
n_ff_shexp = 512 come il Thor) e la stessa interfaccia usata da rilq_thor (nascosto, avanti, indietro, esce,
strati_allenati); e2e_allena e' sostituito da una cache del maestro finta (h del maestro = studente perturbato) e da una
KL esatta (T||S) in torch. Prove:
  1. il driver parte, il cancello d'identita' (gamma = 0) e' AL BIT, 2 passi, esportazione;
  2. i residui coprono anche lo shexp: blk.L.ffn_{gate,up,down}_shexp_lra/_lrb in OGNI strato, forme del motore
     (lra ne = (in, 256), lrb ne = (256, out); down: in = 512);
  3. --famiglie attn,gdn (la ricetta di oggi) non installa nulla sullo shexp;
  4. lo scrittore GGUF accetta i tensori shexp (modifica nulla = stessa sha; aggiunta = rilettura al byte).
Nessuna riga in stato/cicli.tsv (rilq_thor.CICLI deviato in una cartella temporanea).
Uso: python3 test_rilq_thor_shexp.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import residui_ternari as RT          # noqa: E402
import rilq_thor as RQ               # noqa: E402
import e1_scrivi_residui_gguf as WG  # noqa: E402

H, FF, V, T = 256, 512, 64, 24
TIPI = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]


def _lin(i, o, g, s=1.0):
    m = torch.nn.Linear(i, o, bias=False)
    with torch.no_grad():
        m.weight.copy_(torch.randn(o, i, generator=g) * s / i ** 0.5)
    m.weight.requires_grad_(False)
    return m


class _Attn(torch.nn.Module):
    def __init__(self, g, pieno):
        super().__init__()
        self.pieno = pieno
        if pieno:
            self.q_proj, self.k_proj, self.v_proj, self.o_proj = (_lin(H, H, g) for _ in range(4))
        else:
            self.in_proj_qkv, self.in_proj_z, self.out_proj = (_lin(H, H, g) for _ in range(3))

    def forward(self, x):
        if self.pieno:
            return self.o_proj(torch.tanh(self.q_proj(x)) * torch.sigmoid(self.k_proj(x)) + self.v_proj(x))
        return self.out_proj(torch.tanh(self.in_proj_qkv(x)) * torch.sigmoid(self.in_proj_z(x)))


class _Shared(torch.nn.Module):          # Qwen3_5MoeMLP: down(silu(gate(x)) * up(x))
    def __init__(self, g):
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = _lin(H, FF, g), _lin(H, FF, g), _lin(FF, H, g)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _Mlp(torch.nn.Module):
    def __init__(self, g):
        super().__init__()
        self.shared_expert = _Shared(g)

    def forward(self, x):
        return self.shared_expert(x)


class _Strato(torch.nn.Module):
    def __init__(self, g, tipo):
        super().__init__()
        pieno = tipo == "full_attention"
        setattr(self, "self_attn" if pieno else "linear_attn", _Attn(g, pieno))
        self.mlp = _Mlp(g)

    def forward(self, x):
        att = self._modules.get("self_attn") or self._modules.get("linear_attn")
        x = x + 0.5 * att(x)
        return x + 0.5 * self.mlp(x)


def _fabbrica(seme):
    class Stud:
        def __init__(s_, **kw):
            g = torch.Generator().manual_seed(seme)
            s_.device = torch.device("cpu")
            s_.strati = list(range(len(TIPI)))
            s_.cfg = types.SimpleNamespace(layer_types=TIPI)
            s_.E = torch.randn(V, H, generator=g)
            s_.moduli = {L: _Strato(g, TIPI[L]) for L in s_.strati}
            s_.strati_allenati = None

        def parametri(s_):
            return [p for m in s_.moduli.values() for p in m.parameters()]

        def _avanti(s_, ids):
            x = s_.E[ids.reshape(-1) % V]
            ing = []
            for L in s_.strati:
                ing.append(x)
                x = s_.moduli[L](x)
            return x, ing

        def nascosto(s_, ids, inizi=None):
            return s_._avanti(ids)[0]

        def avanti(s_, ids, inizi=None):
            with torch.no_grad():
                x, ing = s_._avanti(ids)
            return {"xL": x, "ingressi": ing}

        def indietro(s_, stato, g):
            i0 = min(s_.strati_allenati)
            for L in reversed(s_.strati):
                if L < i0:
                    break
                xin = stato["ingressi"][L].detach().requires_grad_(L > i0)
                with torch.enable_grad():
                    s_.moduli[L](xin).backward(g)
                g = xin.grad

        def esce(s_, x):
            return x
    return Stud


def _kl(hS, WS, hT, WT):
    lS = F.log_softmax(hS.float() @ WS.T, -1)
    lT = F.log_softmax(hT.float() @ WT.T, -1)
    return (lT.exp() * (lT - lS)).sum(-1)


def _allena_finto(WS, WT, maestro):
    class Sorgente:
        def __init__(s_, path):
            s_.path = path

        def righe_previste(s_, split):
            return list(range(3 if split == "heldout" else 8))

    class Caricatore:
        def __init__(s_, src, dev):
            s_.src = src

        def prendi(s_, split, righe):
            g = torch.Generator().manual_seed(1000 * (split == "heldout") + righe[0])
            ids = torch.randint(0, V, (1, T), generator=g)
            with torch.no_grad():
                hT = maestro.nascosto(ids)
            return types.SimpleNamespace(ids=ids, inizi=None, hT=hT)

    class KLFusa:
        @staticmethod
        def apply(hS, WS_, hT, WT_, pezzo):
            return _kl(hS, WS_, hT, WT_).sum()

    return types.SimpleNamespace(
        scegli_gdn=lambda log: "finto", testa_studente=lambda st, m: WS, testa_maestro=lambda dev, dt: (WT, "finto"),
        SorgenteCache=Sorgente, Caricatore=Caricatore, KLFusa=KLFusa,
        kl_pezzo=lambda hS, WS_, hT, WT_, f: {"kl": _kl(hS, WS_, hT, WT_)})


def _esegui(out, famiglie, passi=2):
    g = torch.Generator().manual_seed(5)
    WS = torch.randn(V, H, generator=g) / H ** 0.5
    WT = WS + 0.3 * torch.randn(V, H, generator=g) / H ** 0.5
    maestro = _fabbrica(99)()
    fS = types.SimpleNamespace(Studente=_fabbrica(7))
    fA = _allena_finto(WS, WT, maestro)
    old = {k: sys.modules.get(k) for k in ("e2e_studente", "e2e_allena")}
    sys.modules["e2e_studente"], sys.modules["e2e_allena"] = fS, fA
    perm_vero, cicli_vero = RT._perm, RQ.CICLI
    RT._perm = lambda chiave, fn: np.random.default_rng(sum(map(ord, chiave))).permutation(H)
    RQ.CICLI = os.path.join(out, "cicli_finto.tsv")
    os.makedirs(out, exist_ok=True)
    try:
        return RQ.main(["--uscita", out, "--gguf", "/dev/null", "--nome", "SINT", "--teacher", "finto",
                        "--strati", "0-3", "--famiglie", famiglie, "--rango", "16", "--passi", str(passi),
                        "--heldout-righe", "3", "--heldout-ogni", "1", "--lr", "3e-3", "--lr-gamma", "3e-2",
                        "--riscaldamento", "1", "--device", "cpu", "--pezzo-kl", "64"])
    finally:
        RT._perm, RQ.CICLI = perm_vero, cicli_vero
        for k, v in old.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_rilq_shexp(tmp=None):
    torch.manual_seed(0)
    d = tmp or tempfile.mkdtemp(prefix="rilq_shexp_")
    out = os.path.join(d, "shexp")
    assert _esegui(out, "attn,gdn,shexp") == 0
    reg = open(os.path.join(out, "registro.log")).read()
    assert "AL BIT" in reg, reg[-2000:]
    # 3 GDN x 3 + 1 full x 4 + 4 strati x 3 shexp = 25 proiezioni
    assert "25 proiezioni in 4 strati" in reg, reg
    assert "attn,gdn,shexp" in open(os.path.join(out, "cicli_finto.tsv")).read()
    st = json.load(open(os.path.join(out, "storia.json")))
    assert [r["passo"] for r in st["storia"]] == [0, 1, 2], st["storia"]
    esp = json.load(open(os.path.join(out, "esportazione.json")))
    print(f"shexp: KL held {esp['kl0']:.5f} -> {esp['kl_held']:.5f} (passo {esp['passo']})")
    assert esp["kl_held"] < esp["kl0"], esp
    info = json.load(open(os.path.join(out, "residui.json")))
    z = np.load(os.path.join(out, "residui_tq1.npz"))
    assert len(z.files) == 50 == len(info["tensori"])
    for L in range(4):
        for f, (i, o) in {"ffn_gate_shexp": (H, FF), "ffn_up_shexp": (H, FF), "ffn_down_shexp": (FF, H)}.items():
            assert info["tensori"][f"blk.{L}.{f}_lra.weight"] == {"type": 34, "ne": [i, RT.RP]}
            assert info["tensori"][f"blk.{L}.{f}_lrb.weight"] == {"type": 34, "ne": [RT.RP, o]}
    # la ricetta di oggi (attn,gdn) non tocca lo shexp
    out2 = os.path.join(d, "attn_gdn")
    assert _esegui(out2, "attn,gdn", passi=1) == 0
    reg2 = open(os.path.join(out2, "registro.log")).read()
    assert "AL BIT" in reg2 and "13 proiezioni in 4 strati" in reg2, reg2
    # scrittore GGUF: modifica nulla = stessa sha; aggiunta dei tensori shexp riletta al byte
    import gguf
    src = os.path.join(d, "src.gguf")
    w = gguf.GGUFWriter(src, "qwen35moe")
    w.add_tensor("token_embd.weight", np.random.default_rng(0).standard_normal((16, 256)).astype(np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    nul = os.path.join(d, "nulla.gguf")
    WG.scrivi(src, nul, [], None)
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()   # noqa: E731
    assert sha(src) == sha(nul)
    nuovi = WG.carica_nuovi(os.path.join(out, "residui_tq1.npz"), os.path.join(out, "residui.json"))
    dst = os.path.join(d, "con_shexp.gguf")
    WG.scrivi(src, dst, nuovi, json.dumps(info))
    r = gguf.GGUFReader(dst)
    nomi = {t.name: t for t in r.tensors}
    t = nomi["blk.2.ffn_down_shexp_lra.weight"]
    assert list(map(int, t.shape)) == [FF, RT.RP] and t.tensor_type == gguf.GGMLQuantizationType.TQ1_0
    assert np.array_equal(np.asarray(t.data).reshape(-1), z["blk.2.ffn_down_shexp_lra.weight"].reshape(-1))
    print(f"OK rilq_thor --famiglie attn,gdn,shexp (file in {d})")
    return d


if __name__ == "__main__":
    test_rilq_shexp()

#!/usr/bin/env python3
"""Prova CPU di e1_recupero_lettera.main su uno studente FINTO (come test_driver_su_studente_finto di e2e_test_rexp):
2 strati MLP + esperti di recupero VERI (e2e_rexp), contesto = media causale DENTRO il segmento (rispetta `inizi`,
cosi' l'ultimo token "vede" il proprio prompt e solo quello). Dati: gruppi VERI di items.jsonl (tokenizer vero, id
ridotti modulo V); maestro = lo stesso finto con pesi perturbati -> teacher_logprobs delle 4 lettere.
Controlla: cancello d'identita' AL BIT, perdita held-out che scende, esportazione npz con i tensori attesi,
rifiuto di righe MMLU. Solo CPU, pochi MB."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import e1_recupero_lettera as DRV          # noqa: E402  (mette in sys.path la copia congelata del recupero)
import e2e_rexp as R                       # noqa: E402

ITEMS = "@@ROOT:models@@/gguf/tony-tern/lettera/data/items.jsonl"
H, V, NL = 256, 512, 2
LIDS = [357, 417, 351, 414]


def fake_world(seed=9):
    g = torch.Generator().manual_seed(seed)
    W = [torch.randn(H, H, generator=g) / H ** 0.5 for _ in range(NL)]
    E = torch.randn(V, H, generator=g)
    Wout = torch.randn(V, H, generator=g) / H ** 0.5
    WT = [w + 0.6 * torch.randn(H, H, generator=g) / H ** 0.5 for w in W]     # maestro: pesi perturbati
    return W, E, Wout, WT


def contesto(ids, inizi):
    """media causale delle embedding DENTRO il segmento (inizi = 1 apre un segmento)."""
    x = E_[ids.reshape(-1) % V]
    seg = torch.cumsum(inizi.reshape(-1), 0) - 1
    out = torch.empty_like(x)
    for s in seg.unique():
        m = seg == s
        c = torch.cumsum(x[m], 0) / torch.arange(1, int(m.sum()) + 1)[:, None]
        out[m] = c
    return out


def build_fakes(W, E, Wout):
    global E_
    E_ = E

    class Mlp(torch.nn.Module):
        def __init__(s_, Wl):
            super().__init__(); s_.Wl = Wl; s_.router_w = torch.zeros(4, H)

        def forward(s_, h):
            y = torch.tanh(h @ R.ricostruisci(*R.piano_ternario(s_.Wl)).T)
            r = s_._modules.get("rexp")
            return y + r(h) if r is not None else y

    class Strato(torch.nn.Module):
        def __init__(s_, Wl):
            super().__init__(); s_.mlp = Mlp(Wl)

        def forward(s_, x):
            return x + s_.mlp(x)

    class Stud:
        def __init__(s_, **kw):
            s_.device = torch.device("cpu"); s_.strati = list(range(NL))
            s_.moduli = {L: Strato(W[L]) for L in s_.strati}; s_.strati_allenati = None

        def parametri(s_):
            return []

        def testa(s_, dtype=torch.float32):
            return Wout.to(dtype)

        def avanti(s_, ids, inizi=None):
            with torch.no_grad():
                x = contesto(ids, inizi); ing = []
                for L in s_.strati:
                    ing.append(x); x = s_.moduli[L](x)
            return {"xL": x, "ingressi": ing}

        def indietro(s_, stato, g):
            i0 = min(s_.strati_allenati)
            for L in reversed(s_.strati):
                if L < i0:
                    break
                xin = stato["ingressi"][L].detach().requires_grad_(L > i0)
                with torch.enable_grad():
                    y = s_.moduli[L](xin); y.backward(g)
                g = xin.grad

        def esce(s_, x):
            return x

        def nascosto(s_, ids, inizi=None):
            return s_.avanti(ids, inizi)["xL"]

    return Stud


def teacher_logits(ids_list, WT, E, Wout):
    out = []
    for ids in ids_list:
        x = E[torch.tensor(ids) % V].cumsum(0)[-1] / len(ids)
        for L in range(NL):
            x = x + torch.tanh(x @ R.ricostruisci(*R.piano_ternario(WT[L])).T)
        out.append(x @ Wout[LIDS].T)
    return torch.stack(out)


def make_data(path, n_train=48, n_ho=16, inject_mmlu=False):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(DRV.TOKENIZER)
    W, E, Wout, WT = fake_world()
    groups = {"train": {}, "heldout": {}}
    with open(ITEMS) as f:
        for l in f:
            r = json.loads(l)
            if r["split"] == "heldout" and r["source"] not in ("logiqa", "medmcqa"):
                continue                      # 3/10: la selezione E1 usa SOLO l'held-out LogiQA/MedMCQA
            d = groups[r["split"]]
            lim = n_train if r["split"] == "train" else n_ho
            if r["group_id"] in d or len(d) < lim:
                d.setdefault(r["group_id"], []).append(r)
            if len(groups["train"]) >= n_train and len(groups["heldout"]) >= n_ho and \
                    all(len(v) == 4 for dd in groups.values() for v in dd.values()):
                break
    rows = [r for dd in groups.values() for v in dd.values() for r in v]
    z = teacher_logits([tok(r["prompt"]).input_ids for r in rows], WT, E, Wout)
    lp = torch.log_softmax(z, -1)
    with open(path, "w") as f:
        for r, l in zip(rows, lp.tolist()):
            r["teacher_logprobs"] = l
            if inject_mmlu:
                r["source"] = "cais/mmlu"
            f.write(json.dumps(r) + "\n")
    return W, E, Wout


def run(args_extra, data, out, W, E, Wout):
    fS = types.SimpleNamespace(Studente=build_fakes(W, E, Wout))
    fA = types.SimpleNamespace(scegli_gdn=lambda log: "finto")
    old = {k: sys.modules.get(k) for k in ("e2e_studente", "e2e_allena")}
    sys.modules["e2e_studente"], sys.modules["e2e_allena"] = fS, fA
    try:
        return DRV.main(["--uscita", out, "--dati", data, "--device", "cpu", "--strati", "0-1", "--n-r", "4",
                         "--k-r", "2", "--F", "256", "--gemm", "fp32", "--gruppi-passo", "8", "--token-lotto", "900",
                         "--gruppi-init", "8", "--heldout-ogni", "5", "--heldout-gruppi", "16"] + args_extra)
    finally:
        for k, v in old.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def main():
    base = os.environ.get("E1_PROVA_DIR", "@@ROOT:models@@/gguf/tony-tern/lettera/e1_prova")
    os.makedirs(base, exist_ok=True)
    d = tempfile.mkdtemp(dir=base)
    data = d + "/items_teacher_finto.jsonl"
    W, E, Wout = make_data(data)
    rc = run(["--passi", "30", "--lr-lat", "3e-3", "--lr-gamma", "3e-2", "--lr-r", "1e-3",
              "--target", "debiased", "--peso-pif", "0"], data, d + "/out", W, E, Wout)  # ricetta di prima
    assert rc == 0, rc
    reg = open(d + "/out/registro.log").read()
    assert "AL BIT" in reg, "cancello d'identita' non passato"
    e = json.load(open(d + "/out/esportazione.json"))
    k0, k1 = e["held0"]["loss"], e["held"]["loss"]
    t = np.load(d + "/out/rexp_tq1.npz")
    assert e["n_tensori"] == NL * 9 == len(t.files), (e["n_tensori"], len(t.files))
    assert k1 < 0.9 * k0, f"perdita held-out non scesa abbastanza: {k0} -> {k1}"
    print(f"E1 FINTO OK: held loss {k0:.4f} -> {k1:.4f} (passo {e['passo']}), acc {e['held0']['acc']:.3f} -> "
          f"{e['held']['acc']:.3f}, tensori {len(t.files)}, dir {d}")
    # rifiuto MMLU
    data2 = d + "/items_mmlu.jsonl"
    make_data(data2, 4, 2, inject_mmlu=True)
    try:
        run(["--passi", "1"], data2, d + "/out2", W, E, Wout)
        raise AssertionError("righe MMLU non rifiutate")
    except SystemExit as ex:
        assert "MMLU" in str(ex), ex
    print("rifiuto MMLU OK")


if __name__ == "__main__":
    torch.set_num_threads(8)
    main()

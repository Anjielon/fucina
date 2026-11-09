#!/usr/bin/env python3
"""Prove CPU di R007: coerenza trainer-motore sui 2 piani, cache del prefisso (esatta) e suo confine, budget di tempo.
Nessuna GPU, nessuna rete, nessun peso: modelli finti. Uso: python3 test_e1_veloce.py"""
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
sys.path.insert(1, str(QUI / "recupero_7dabd6f" / "thor_v2"))
import e1_veloce as V                         # noqa: E402
import residui_ternari as RT                  # noqa: E402
import test_residui_ternari as T              # noqa: E402  (studente finto, dati finti, driver)


def _emula_motore(tensori: dict, nome: str, x_file: np.ndarray, piani: int) -> np.ndarray:
    """cur += B (A x) esattamente come `thor_lr_add` in qwen35moe.cpp (ramo cloud/residui-lr):
         ax = lra x [+ lra2 x];   r = lrb ax [+ lrb2 ax]
    A e B sono i tensori TQ1_0 ESPORTATI, riletti dai loro byte (stessa dequantizzazione di ggml)."""
    def mat(suff, C):
        return RT.deq_tq1_0(tensori[f"{nome}_{suff}.weight"], C)
    ax = x_file @ mat("lra", x_file.shape[1]).T
    if piani > 1:
        ax = ax + x_file @ mat("lra2", x_file.shape[1]).T
    r = ax @ mat("lrb", RT.RP).T
    if piani > 1:
        r = r + ax @ mat("lrb2", RT.RP).T
    return r


def test_due_piani_trainer_uguale_motore():
    """(a) Il trainer calcola (Σ_p B_p)(Σ_p A_p) x; il motore somma i due A PRIMA di moltiplicare per B e poi somma i due B:
    anche lui (B1+B2)(A1+A2) x, non B1 A1 x + B2 A2 x. Verificato sui byte esportati, con le tre relazioni di permutazione."""
    g = torch.Generator().manual_seed(5)
    for rel, fuori, dentro in (("identita", 96, 256), ("perm_righe", 96, 256), ("perm_colonne", 96, 512)):
        for piani in (1, 2):
            cfg = RT.ConfigResidui(rango=16, piani=piani, strati=(0,), famiglie=("attn_q",))
            perm = torch.randperm(dentro if rel == "perm_colonne" else fuori, generator=g).numpy()
            px = perm if rel == "perm_colonne" else None
            iy = None
            if rel == "perm_righe":
                iy = np.empty_like(perm); iy[perm] = np.arange(len(perm))
            res = RT.ResiduoTernario("blk.0.attn_q", fuori, dentro, cfg, perm_x=px, inv_y=iy, seme=7)
            with torch.no_grad():
                res.gamma.uniform_(0.3, 1.7)
                res.A.mul_(1.0); res.B.mul_(1.0)
            x_hf = torch.randn(6, dentro, generator=g)
            y_tr = res(x_hf).detach().numpy()                       # spazio HF
            t, _ = RT.tensori_gguf({"blk.0.attn_q": res})
            assert len(t) == 2 * piani
            x_file = x_hf.numpy()[:, perm] if rel == "perm_colonne" else x_hf.numpy()
            y_file = _emula_motore(t, "blk.0.attn_q", x_file, piani)[:, :fuori]
            y_mot = y_file[:, iy] if rel == "perm_righe" else y_file   # y_hf = y_file[inv]
            scarto = np.abs(y_tr - y_mot).max() / (np.abs(y_tr).max() + 1e-12)
            assert scarto < 1e-4, (rel, piani, scarto)
            if piani == 2:
                # la lettura SBAGLIATA (piani separati B1 A1 + B2 A2) darebbe un altro numero: la prova discrimina
                def m(s, C):
                    return RT.deq_tq1_0(t[f"blk.0.attn_q_{s}.weight"], C)
                sbagliato = (x_file @ m("lra", dentro).T) @ m("lrb", 256).T + (x_file @ m("lra2", dentro).T) @ m("lrb2", 256).T
                sbagliato = sbagliato[:, iy] if rel == "perm_righe" else sbagliato
                d = np.abs(y_tr - sbagliato[:, :fuori]).max() / (np.abs(y_tr).max() + 1e-12)
                assert d > 1e-2, (rel, d)
    print("due piani: trainer == motore (Σ B)(Σ A) su 3 relazioni; la lettura a piani separati sbaglia di", f"{d:.2f}")


def _gradi(st, res, lot, cache):
    for m in res.values():
        m.zero_grad(set_to_none=True)
    stato = cache.avanti(lot) if cache is not None else st.avanti(lot["ids"], inizi=lot["inizi"])
    xL = stato["xL"].detach().requires_grad_(True)
    with torch.enable_grad():
        z = st.esce(xL)[lot["ultimi"]].float()
        (z.pow(2).mean()).backward()
    st.indietro(stato, xL.grad)
    return {n: [p.grad.clone() for p in (m.A, m.B, m.gamma)] for n, m in res.items()}, z.detach()


def _mondo_con_residui(gamma_seme=3):
    torch.manual_seed(1)
    E, Wout, W, TE = T._mondo()
    st = T._studente(E, Wout, W)()
    cfg = RT.ConfigResidui(rango=16, strati=(1, 2, 3), famiglie=RT.famiglie_da("attn,gdn"))
    rt = RT._perm
    RT._perm = lambda chiave, fn: T._perm(T.H, sum(map(ord, chiave)))
    try:
        res = RT.installa(st, cfg, device=torch.device("cpu"))
    finally:
        RT._perm = rt
    g = torch.Generator().manual_seed(gamma_seme)
    for m in res.values():
        with torch.no_grad():
            m.gamma.copy_(torch.rand(m.fuori, generator=g) + 0.5)
    for m in res.values():
        m.requires_grad_(True)
    st.strati_allenati = [1, 2, 3]
    righe = []
    for i in range(6):
        ids = T._enc(f"domanda numero {i} con qualche parola in piu'?\nA. x\nB. y\nAnswer:")
        righe.append({"id": f"r{i}", "ids": ids})
    inizi, ids_, ultimi, t = [], [], [], 0
    for r in righe:
        ids_ += r["ids"]; inizi += [1] + [0] * (len(r["ids"]) - 1); t += len(r["ids"]); ultimi.append(t - 1)
    lot = {"ids": torch.tensor([ids_]), "inizi": torch.tensor([inizi]), "ultimi": torch.tensor(ultimi), "rows": righe}
    return st, res, lot


def test_cache_prefisso_gradienti_identici():
    """(b) Con la cache del prefisso (strato 0, congelato) i gradienti dei residui sono gli STESSI dell'avanti pieno."""
    st, res, lot = _mondo_con_residui()
    L0 = V.primo_strato_allenato(st)
    assert L0 == 1
    pieno, z_p = _gradi(st, res, lot, None)
    cache = V.CachePrefisso(st, L0)
    assert cache.riempi(lot["rows"], token_max=60, device="cpu", log=lambda *_: None) == 6      # piu' lotti da <= 60 token
    assert not cache.mancanti(lot["rows"])
    veloce, z_c = _gradi(st, res, lot, cache)
    assert torch.equal(z_p, z_c)
    for n in pieno:
        for a, b in zip(pieno[n], veloce[n]):
            assert torch.equal(a, b), n
    assert any(float(g.abs().sum()) > 0 for gs in pieno.values() for g in gs)
    # l'ordine delle righe nel lotto non importa: ogni riga e' un segmento indipendente
    rows2 = lot["rows"][::-1]
    x0 = cache.x0(rows2, "cpu")
    assert x0.shape[-2] == sum(len(r["ids"]) for r in rows2)
    # impronta dei token: una riga ritokenizzata NON trova lo stato vecchio, nessun ripiego silenzioso
    r2 = {"id": "r0", "ids": lot["rows"][0]["ids"][:-1] + [3]}
    try:
        cache.x0([r2], "cpu")
        raise AssertionError("cache servita per token diversi")
    except KeyError:
        pass
    assert cache.mancanti([r2]) == [r2] and cache.byte() > 0
    # salva / riprendi: stessa meta' -> righe riprese; meta' diversa -> ignorata
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.pt")
        cache.salva(p, {"k": 1})
        c2 = V.CachePrefisso(st, L0)
        assert c2.carica(p, {"k": 1}) == 6 and V.CachePrefisso(st, L0).carica(p, {"k": 2}) == 0
        assert V.CachePrefisso(st, 2).carica(p, {"k": 1}) == 0
    print("cache prefisso: gradienti uguali al bit su 6 righe (3 residui x famiglie)")


def test_cache_dentro_il_blocco_sbaglia():
    """Il confine NON si sposta dentro i blocchi allenati: uno stato in cache all'ingresso dello strato 3, calcolato
    PRIMA di allenare i residui degli strati 1-2, e' vecchio dopo ogni passo -> gradiente diverso (qui: nettamente)."""
    st, res, lot = _mondo_con_residui()
    pieno0, _ = _gradi(st, res, lot, None)
    cache = V.CachePrefisso(st, 3)
    cache.riempi(lot["rows"], 10_000, "cpu", log=lambda *_: None)          # stato a monte dello strato 3, oggi
    for n, m in res.items():                                               # un "passo": i residui 1-2 cambiano
        if n.startswith(("blk.1.", "blk.2.")):
            with torch.no_grad():
                m.gamma.mul_(0.2); m.B.add_(0.5 * torch.randn_like(m.B))
    pieno1, z1 = _gradi(st, res, lot, None)
    stato = V.avanti_da(st, lot["ids"], lot["inizi"], cache.x0(lot["rows"], "cpu"), 3)
    z_vecchio = st.esce(stato["xL"])[lot["ultimi"]].float()
    diff = float((z1 - z_vecchio).abs().max() / z1.abs().max())
    assert diff > 1e-2, diff
    # e il prefisso vero (strato 0) resta esatto anche dopo quel passo
    c1 = V.CachePrefisso(st, 1)
    c1.riempi(lot["rows"], 10_000, "cpu", log=lambda *_: None)
    z_c, _ = _gradi(st, res, lot, c1)[1], None
    assert torch.equal(z_c, z1)
    print(f"cache dentro il blocco: scarto relativo dei logit {diff:.2f} (inaccettabile); sul prefisso: 0")


def test_primo_strato_allenato_rifiuta_casi_vuoti():
    st, _, _ = _mondo_con_residui()
    for s in (None, [], [0]):
        st.strati_allenati = s
        try:
            V.primo_strato_allenato(st)
            raise AssertionError(s)
        except ValueError:
            pass


def test_budget_e_profilo():
    # 3 h, 6 min/passo misurato, valutazione 5 min ogni 25 passi: 150 passi non ci stanno, il budget dice quanti
    n = V.passi_da_budget(3 * 3600, speso_s=1200, s_passo=360, s_eval=300, ogni=25, fatti=3, massimo=150, minimo=20)
    assert n == 3 + int((3 * 3600 - 1200 - 300 - 120) // (360 + 300 / 25)) == 27, n
    # abbastanza tempo: tetto = massimo
    assert V.passi_da_budget(100 * 3600, 100, 10, 10, 20, 3, 150) == 150
    # budget impossibile: ritorna il minimo (il chiamante lo segnala)
    assert V.passi_da_budget(60, 50, 360, 300, 25, 3, 150, 20) == 20
    # monotonia: passo piu' lento => meno passi
    assert V.passi_da_budget(7200, 0, 20, 10, 10, 3, 500) > V.passi_da_budget(7200, 0, 40, 10, 10, 3, 500)
    r = V.regressione([1000, 2000, 4000, 8000], [250, 252, 256, 264])        # legato ai pesi: tempo quasi piatto
    assert r["peso_fisso_a_xmax"] > 0.9 and abs(r["a_s"] - 248) < 1
    r2 = V.regressione([1000, 2000, 4000, 8000], [10, 20, 40, 80])           # legato ai token: proporzionale
    assert r2["peso_fisso_a_xmax"] < 0.05
    c = V.consiglia({"passo_da_L0": r}, budget_s=3 * 3600)
    assert c["legato_ai_pesi"] and not c["ci_sta"] and c["token_ginocchio"] > 3040
    c2 = V.consiglia({"passo_da_L0": r2}, budget_s=3 * 3600)
    assert not c2["legato_ai_pesi"] and c2["ci_sta"] and c2["passi_nel_budget_ref"] > 150
    assert V.mediana([3, 1, 2]) == 2
    # una forma NUOVA che costa >> una nota = ricompilazione per ogni lotto di taglia diversa: il consiglio lo dice
    m_ok = [{"token": 1000, "prefisso_s": 1, "avanti_s": 10, "indietro_s": 30, "prima_volta_s": 45}]
    m_ko = [{"token": 1000, "prefisso_s": 1, "avanti_s": 10, "indietro_s": 30, "prima_volta_s": 300}]
    assert not V.consiglia({"passo_da_L0": r, "misure": m_ok}, 10800)["ricompila_per_forma_nuova"]
    assert V.consiglia({"passo_da_L0": r, "misure": m_ko}, 10800)["ricompila_per_forma_nuova"]
    print("budget e regressione ok:", n, "passi nel budget; legato ai pesi", c["legato_ai_pesi"])


def test_profila_su_finto():
    st, res, lot = _mondo_con_residui()
    righe = lot["rows"]
    for r in righe:
        r.update(split="train")

    def imp(gruppi, dev):
        rows = [r for g in gruppi for r in g]
        ids, inizi, ultimi, t = [], [], [], 0
        for r in rows:
            ids += r["ids"]; inizi += [1] + [0] * (len(r["ids"]) - 1); t += len(r["ids"]); ultimi.append(t - 1)
        return {"ids": torch.tensor([ids]), "inizi": torch.tensor([inizi]), "ultimi": torch.tensor(ultimi), "rows": rows}
    p = V.profila(st, righe * 3, [100, 200, 400], "cpu", imp, 1, ripetizioni=1)
    assert [m["token"] for m in p["misure"]] == sorted(m["token"] for m in p["misure"]) and p["L0"] == 1
    assert all(m["avanti_s"] > 0 and m["indietro_s"] > 0 for m in p["misure"])
    print("profilo su finto: ", [(m["token"], round(m["avanti_s"] + m["indietro_s"], 4)) for m in p["misure"]])


def test_driver_con_cache_e_budget():
    """Il driver intero (studente finto): la corsa con cache ha le stesse metriche di quella senza (stesso seme),
    il cancello cache e' scritto nel registro e --budget-ore accorcia i passi programmati."""
    d = tempfile.mkdtemp()
    torch.manual_seed(0)
    E, Wout, W, TE = T._mondo()
    dati = os.path.join(d, "items_teacher.jsonl")
    T._dati(dati, E, Wout, TE)
    base = ["--modo", "residui", "--passi", "12", "--lr-res", "3e-3", "--lr-res-gamma", "3e-2"]
    esiti = {}
    for nome, extra in (("con", ["--cache-prefisso", "si", "--token-cache", "400"]), ("senza", ["--cache-prefisso", "no"])):
        out = os.path.join(d, nome)
        torch.manual_seed(0)
        assert T._esegui(base + extra, dati, out, E, Wout, T_W(E, Wout)) == 0
        esiti[nome] = json.load(open(os.path.join(out, "esportazione.json")))
        reg = open(os.path.join(out, "registro.log")).read()
        assert ("cancello cache" in reg) == (nome == "con"), nome
    for k in ("loss", "kd", "acc"):
        assert abs(esiti["con"]["held"][k] - esiti["senza"]["held"][k]) < 1e-5, (k, esiti)
    out = os.path.join(d, "budget")
    torch.manual_seed(0)
    assert T._esegui(["--modo", "residui", "--passi", "60", "--budget-ore", "0.004", "--passi-misura", "3",
                      "--passi-min", "5", "--heldout-ogni", "5"], dati, out, E, Wout, T_W(E, Wout)) == 0
    reg = open(os.path.join(out, "registro.log")).read()
    st = json.load(open(os.path.join(out, "storia.json")))
    assert "budget 0.004 h" in reg and st["passi_programmati"] < 60 and st["passi_eseguiti"] == st["passi_programmati"], (
        reg[-1500:], st["passi_programmati"])
    print("driver: con cache == senza (held loss", f"{esiti['con']['held']['loss']:.5f});",
          "budget ->", st["passi_programmati"], "passi su 60")


def T_W(E, Wout):
    return T._mondo()[2]


if __name__ == "__main__":
    torch.set_num_threads(4)
    fn = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)
          and (len(sys.argv) < 2 or any(s in k for s in sys.argv[1:]))]
    for k, f in fn:
        f()
        print("ok", k, flush=True)
    print(f"TUTTE LE {len(fn)} PROVE OK")

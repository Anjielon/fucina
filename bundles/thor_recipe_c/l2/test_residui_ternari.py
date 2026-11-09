#!/usr/bin/env python3
"""Prove SOLO CPU, AUTOSUFFICIENTI (niente /mnt, niente tokenizer vero, niente GPU) di:
  - residui_ternari.py      quantizzazione a gruppo variabile, imbottitura TQ1_0 esatta, spazio del file (permutazioni),
                            identita' a gamma = 0, STE, init dal gradiente (sonda accumulata == statistiche), export
  - e1_perdita_legame.py    KD alla stessa rotazione + coerenza + PIF; fonti vietate; metriche di selezione
  - e1_recupero_lettera.py  driver su studente FINTO con attenzione (modo residui ed entrambi): cancello AL BIT, perdita
                            held-out che scende, selezione SOLO LogiQA/MedMCQA, rifiuto ARC/MMLU, export
  - e1_scrivi_residui_gguf  modifica nulla = stessa sha256; tensori in piu' rileggibili da gguf-py, originali al byte
Uso: python3 test_residui_ternari.py      (oppure pytest test_residui_ternari.py)
     E1_GGUF_VERO=/percorso/Thor-T-R2B3m.gguf  -> anche la modifica nulla sul file vero (copia di ~9 GiB)
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import e1_recupero_lettera as DRV          # noqa: E402  (mette in sys.path la copia congelata del recupero)
import e1_scrivi_residui_gguf as WG        # noqa: E402
import e2e_rexp as RX                      # noqa: E402
import residui_ternari as RT               # noqa: E402
from e1_perdita_legame import fonte_vietata, metriche_selezione, perdita_legame, pif_loss  # noqa: E402

H, V = 256, 512
TIPI4 = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
LIDS = [357, 417, 351, 414]


# ═══════════════════════════════════ quantizzazione ═══════════════════════════════════
def test_piano_g256_uguale_a_rexp():
    w = torch.randn(7, 512, generator=torch.Generator().manual_seed(1))
    T1, s1 = RT.piano_ternario_g(w, 256)
    T2, s2 = RX.piano_ternario(w)
    assert torch.equal(T1, T2) and torch.equal(s1, s2)


def test_imbottitura_tq1_0_esatta():
    """B (out, r) a scala per riga -> imbottito a 256 -> byte TQ1_0 -> dequantizzato == valori dell'avanti, AL BIT;
    e la scala ottima sul blocco imbottito (gruppo 256) e' la stessa (gli zeri non cambiano k*)."""
    for r in (16, 32, 64):
        B = torch.randn(40, r, generator=torch.Generator().manual_seed(r)) * 0.03
        T, s = RT.piano_ternario_g(B, r)
        Bq = RT.ricostruisci_g(T, s, r)
        pad = torch.zeros(40, 256)
        pad[:, :r] = Bq
        d = RT.deq_tq1_0(RT.byte_tq1_0(pad.numpy()), 256)
        assert np.array_equal(d, pad.numpy()), r
        Bp = torch.zeros(40, 256); Bp[:, :r] = B
        T2, s2 = RT.piano_ternario_g(Bp, 256)
        assert torch.equal(s2, s) and torch.equal(T2[:, :r], T)


# ═══════════════════════════════════ modulo ═══════════════════════════════════
def _perm(n, seme):
    return np.random.default_rng(seme).permutation(n)


def test_spazio_file_e_identita():
    cfg = RT.ConfigResidui(rango=16, strati=(0,), famiglie=("attn_qkv",))
    pr, pc = _perm(512, 3), _perm(256, 4)
    iy = np.empty_like(pr); iy[pr] = np.arange(512)
    res = RT.ResiduoTernario("blk.0.x", 512, 256, cfg, perm_x=pc, inv_y=iy, seme=5)
    base = torch.nn.Linear(256, 512, bias=False)
    m = RT.ConResiduo(base, res)
    x = torch.randn(9, 256)
    assert torch.equal(m(x), base(x))                     # gamma = 0: al bit
    with torch.no_grad():
        res.gamma.normal_()
    Aq, Bq = (t.detach() for t in res.pesi_q())
    atteso = ((x[:, torch.as_tensor(pc)] @ Aq.T) @ Bq.T)[:, torch.as_tensor(iy)]
    assert torch.allclose(res(x), atteso, atol=1e-5)
    # l'export (A_pad, B_pad nell'ordine del file, gamma fuso) riproduce l'avanti
    (Ap, Bp), = RT.matrici_file(res)
    y = ((x[:, torch.as_tensor(pc)] @ torch.from_numpy(Ap).T) @ torch.from_numpy(Bp).T)[:, torch.as_tensor(iy)]
    assert torch.allclose(res(x).detach(), y, atol=1e-5)
    # i byte TQ1_0 tornano esattamente A_pad, B_pad
    t, info = RT.tensori_gguf({"blk.0.attn_qkv": res})
    assert np.array_equal(RT.deq_tq1_0(t["blk.0.attn_qkv_lra.weight"], 256), Ap)
    assert np.array_equal(RT.deq_tq1_0(t["blk.0.attn_qkv_lrb.weight"], RT.RP), Bp)
    assert info["blk.0.attn_qkv_lrb.weight"] == {"type": 34, "ne": [256, 512]}
    assert info["blk.0.attn_qkv_lra.weight"] == {"type": 34, "ne": [256, 256]}


def test_ste_gradienti():
    cfg = RT.ConfigResidui(rango=32, piani=2, strati=(0,), famiglie=("attn_q",))
    res = RT.ResiduoTernario("r", 256, 512, cfg, seme=1)
    x = torch.randn(5, 512)
    res(x).pow(2).sum().backward()
    assert res.gamma.grad is not None and res.gamma.grad.abs().sum() == 0      # uscita nulla e gradiente nullo
    res.zero_grad()
    with torch.no_grad():
        res.gamma.fill_(0.1)
    (res(x) * torch.randn(5, 256)).sum().backward()
    assert all(float(p.grad.abs().sum()) > 0 for p in (res.A, res.B, res.gamma))
    assert len(RT.tensori_gguf({"blk.0.attn_q": res})[0]) == 4                  # 2 piani x (lra, lrb)


def test_init_gradiente_sonda_uguale_statistiche():
    cfg = RT.ConfigResidui(rango=8, strati=(0,), famiglie=("attn_q",))
    res = RT.ResiduoTernario("r", 256, 256, cfg, seme=2)
    X = torch.randn(64, 256)
    tgt = torch.randn(64, 256)
    res.sonda = {}
    for xb, tb in ((X[:40], tgt[:40]), (X[40:], tgt[40:])):          # due micro-lotti
        with torch.enable_grad():
            ((res(xb) - tb) ** 2).sum().backward()
    stat, res.sonda = res.sonda, None
    G = -2 * tgt                                                        # dL/dy a gamma = 0 (y = 0)
    ref = RT.statistiche(X, G)
    for k in ("C", "XX", "sx"):
        assert torch.allclose(stat[k], ref[k], atol=1e-8), k
    assert stat["n"] == 64
    d = RT.inizializza_da_gradiente(res, stat)
    assert float(res.gamma.detach().abs().sum()) == 0 and 0 < d["energia_rango"] <= 1
    sd = (X @ res.A.detach().T).std(0)
    assert torch.allclose(sd, torch.ones_like(sd), atol=0.05), sd


# ═══════════════════════════════════ perdita e fonti ═══════════════════════════════════
def test_fonti_vietate():
    for s in ("mmlu", "cais/mmlu", "ai2_arc", "arc", "allenai/openbookqa", "obqa", "race", "ehovy/race", "mctest"):
        assert fonte_vietata(s), s
    for s in ("sciq", "csqa", "qasc", "medmcqa", "mathqa", "logiqa", "trace_qa"):
        assert not fonte_vietata(s), s


def _batch2():
    perms = torch.tensor([[(p + r) % 4 for p in range(4)] for _ in range(2) for r in range(4)])
    gold = torch.tensor([perms[i].tolist().index(0) for i in range(8)])
    return {"perm": perms, "group_index": torch.tensor([0] * 4 + [1] * 4), "n_groups": 2, "gold_pos": gold,
            "teacher_logprobs": F.log_softmax(torch.randn(8, 4, generator=torch.Generator().manual_seed(0)), -1)}


def test_perdita_legame():
    b = _batch2()
    # PIF: abbassa le lettere dei contenuti sbagliati, ovunque stiano
    w = torch.zeros(8, 4, requires_grad=True)
    pif_loss(w, b["gold_pos"]).backward()
    oro = F.one_hot(b["gold_pos"], 4).bool()
    assert (w.grad[oro] < 0).all() and (w.grad[~oro] > 0).all()
    # solo KL (lam = 0, pif = 0) == letter_kd_loss alla stessa rotazione
    from e1_letter_kd import letter_kd_loss
    z = torch.randn(8, 4)
    l, st = perdita_legame(z, b, lam=0.0, w_pif=0.0)
    assert torch.allclose(l, letter_kd_loss(z, b["teacher_logprobs"]))
    # ottimizzare la perdita intera: scende, acc sale a 1 (il maestro casuale pesa meno dell'oro con questi pesi)
    w = torch.zeros(8, 4, requires_grad=True)
    opt = torch.optim.Adam([w], lr=0.1)
    for i in range(200):
        l, st = perdita_legame(w, b, w_kd=0.2, lam=0.5, w_pif=1.0)
        opt.zero_grad(); l.backward(); opt.step()
        if i == 0:
            l0 = st["loss"]
    assert st["loss"] < l0 and st["acc"] == 1.0
    m = metriche_selezione(w.detach(), b, ["logiqa"] * 4 + ["medmcqa"] * 4)
    assert set(m) == {"logiqa", "medmcqa", "tutte"} and m["tutte"]["coerenza_rot"] == 1.0


def test_gruppi_selezione():
    held = [[{"source": s}] * 4 for s in ("sciq", "logiqa", "medmcqa", "csqa", "logiqa", "medmcqa")]
    sel = DRV.gruppi_selezione(held, ("logiqa", "medmcqa"), 10, 0)
    assert len(sel) == 4 and {g[0]["source"] for g in sel} == {"logiqa", "medmcqa"}
    try:
        DRV.gruppi_selezione(held, ("logiqa", "ai2_arc"), 10, 0)
        raise AssertionError("fonte ARC accettata in selezione")
    except SystemExit as e:
        assert "vietate" in str(e)


# ═══════════════════════════════════ studente finto con attenzione ═══════════════════════════════════
def _mondo(seme=11):
    g = torch.Generator().manual_seed(seme)
    rn = lambda *f, s=1.0: torch.randn(*f, generator=g) * s   # noqa: E731
    E = rn(V, H)
    Wout = rn(V, H, s=H ** -0.5)
    st = {L: {k: rn(H, H, s=H ** -0.5) for k in ("a", "b", "c", "d", "mlp")} for L in range(4)}
    te = {L: {k: v + rn(H, H, s=0.7 * H ** -0.5) for k, v in d.items()} for L, d in st.items()}
    return E, Wout, st, te


def _contesto(E, ids, inizi):
    x = E[ids.reshape(-1) % V]
    seg = torch.cumsum(inizi.reshape(-1), 0) - 1
    out = torch.empty_like(x)
    for s in seg.unique():
        m = seg == s
        out[m] = torch.cumsum(x[m], 0) / torch.arange(1, int(m.sum()) + 1)[:, None]
    return out


def _miscela(x, inizi):
    seg = torch.cumsum(inizi.reshape(-1), 0) - 1
    out = torch.empty_like(x)
    for s_ in seg.unique():
        m = seg == s_
        out[m] = torch.cumsum(x[m], 0) / torch.arange(1, int(m.sum()) + 1)[:, None]
    return out


def _lin(W):
    m = torch.nn.Linear(H, H, bias=False)
    with torch.no_grad():
        m.weight.copy_(W)
    m.weight.requires_grad_(False)
    return m


class _Attn(torch.nn.Module):
    def __init__(self, W, pieno):
        super().__init__()
        self.pieno = pieno
        if pieno:
            self.q_proj, self.k_proj, self.v_proj, self.o_proj = (_lin(W[k]) for k in "abcd")
        else:
            self.in_proj_qkv, self.in_proj_z, self.out_proj = (_lin(W[k]) for k in "abc")

    def forward(self, x):
        if self.pieno:
            return self.o_proj(torch.tanh(self.q_proj(x)) * torch.sigmoid(self.k_proj(x)) + self.v_proj(x))
        return self.out_proj(torch.tanh(self.in_proj_qkv(x)) * torch.sigmoid(self.in_proj_z(x)))


class _Mlp(torch.nn.Module):
    def __init__(self, W):
        super().__init__()
        self.W = W
        self.router_w = torch.zeros(4, H)
        self.ultimo_idx = None

    def forward(self, h):
        y = torch.tanh(h @ self.W.T)
        r = self._modules.get("rexp")
        return y + r(h) if r is not None else y


class _Strato(torch.nn.Module):
    def __init__(self, W, tipo):
        super().__init__()
        pieno = tipo == "full_attention"
        setattr(self, "self_attn" if pieno else "linear_attn", _Attn(W, pieno))
        self.mlp = _Mlp(W["mlp"])

    def forward(self, x):
        att = self._modules.get("self_attn") or self._modules.get("linear_attn")
        x = x + 0.5 * att(x)
        return x + self.mlp(x)


def _studente(E, Wout, W):
    class Stud:
        def __init__(s_, **kw):
            s_.device = torch.device("cpu"); s_.strati = list(range(4))
            s_.cfg = types.SimpleNamespace(layer_types=TIPI4)
            s_.moduli = {L: _Strato(W[L], TIPI4[L]) for L in s_.strati}; s_.strati_allenati = None

        def parametri(s_):
            return [p for m in s_.moduli.values() for p in m.parameters()]

        def testa(s_, dtype=torch.float32):
            return Wout.to(dtype)

        # stessa interfaccia dello Studente vero (e1_veloce): embedding -> _contesto -> _strato per strato
        def embedding(s_, ids):
            return E[ids.reshape(-1) % V]

        def _contesto(s_, B, T, dtype, inizi):
            return inizi

        def _strato(s_, L, x, ctx, instr=None):
            if L == 0:                                       # la miscela causale per segmento sta sull'ingresso
                x = _miscela(x, ctx)
            return s_.moduli[L](x)

        def avanti(s_, ids, inizi=None, x0=None):
            with torch.no_grad():
                ctx = s_._contesto(1, ids.shape[-1], torch.float32, inizi)
                x = s_.embedding(ids); ing = []
                for L in s_.strati:
                    ing.append(x); x = s_._strato(L, x, ctx)
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
    return Stud


def _parole(rng, n):
    return [" ".join(rng.choice(["alfa", "beta", "gamma", "delta", "eta", "zeta", "theta", "iota", "kappa", "mu",
                                 "nu", "xi", "rho", "tau", "phi", "psi"]) for _ in range(rng.randint(1, 3)))
            for _ in range(n)]


def _dati(path, E, Wout, TE, n_train=40, n_ho=24, fonte_extra=None, seme=3):
    """Gruppi sintetici in formato items.jsonl (4 rotazioni cicliche) + teacher_logprobs dal maestro finto."""
    rng = random.Random(seme)
    fonti = {"train": ["sciq", "csqa", "logiqa", "medmcqa"], "heldout": ["sciq", "csqa", "logiqa", "medmcqa"]}
    righe = []
    for split, n in (("train", n_train), ("heldout", n_ho)):
        for i in range(n):
            src = fonti[split][i % 4] if not (fonte_extra and i == 0) else fonte_extra
            q = " ".join(_parole(rng, 6)) + "?"
            opts = _parole(rng, 4)
            for rot in range(4):
                perm = [(p + rot) % 4 for p in range(4)]
                pr = q + "\n" + "\n".join(f"{'ABCD'[p]}. {opts[perm[p]]}" for p in range(4)) + "\nAnswer:"
                righe.append({"id": f"{src}/{split}/{i}#r{rot}", "group_id": f"{src}/{split}/{i}", "source": src,
                              "split": split, "rot": rot, "perm": perm, "gold_pos": perm.index(0), "options": opts,
                              "gold": 0, "prompt": pr})
    stud_t = _studente(E, Wout, TE)()
    with open(path, "w") as f:
        for r in righe:
            ids = torch.tensor([_enc(r["prompt"])])
            x = stud_t.avanti(ids, torch.tensor([[1] + [0] * (ids.shape[1] - 1)]))["xL"][-1]
            r["teacher_logprobs"] = F.log_softmax(x @ Wout[LIDS].T, -1).tolist()
            f.write(json.dumps(r) + "\n")


def _enc(s):
    return [1] + [(ord(c) * 7 + i) % V for i, c in enumerate(s)][-200:]


def _esegui(extra, dati, out, E, Wout, W):
    fS = types.SimpleNamespace(Studente=_studente(E, Wout, W))
    fA = types.SimpleNamespace(scegli_gdn=lambda log: "finto")
    old = {k: sys.modules.get(k) for k in ("e2e_studente", "e2e_allena")}
    sys.modules["e2e_studente"], sys.modules["e2e_allena"] = fS, fA
    vecchio_cod = DRV.codificatore
    DRV.codificatore = lambda path, lids: _enc
    perm_vero = RT._perm
    RT._perm = lambda chiave, fn: _perm(H, sum(map(ord, chiave)))
    try:
        return DRV.main(["--uscita", out, "--dati", dati, "--device", "cpu", "--strati", "2-3", "--n-r", "4",
                         "--k-r", "2", "--F", "256", "--gemm", "fp32", "--gruppi-passo", "8", "--token-lotto", "2000",
                         "--gruppi-init", "8", "--heldout-ogni", "5", "--heldout-gruppi", "40",
                         "--residui-strati", "1-3", "--residui-rango", "16"] + extra)
    finally:
        DRV.codificatore = vecchio_cod
        RT._perm = perm_vero
        for k, v in old.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_driver_residui_ed_entrambi(tmp=None):
    torch.manual_seed(0)
    d = tmp or tempfile.mkdtemp()
    E, Wout, W, TE = _mondo()
    dati = os.path.join(d, "items_teacher.jsonl")
    _dati(dati, E, Wout, TE)
    for modo in ("residui", "entrambi"):
        out = os.path.join(d, modo)
        rc = _esegui(["--modo", modo, "--passi", "25", "--lr-res", "3e-3", "--lr-res-gamma", "3e-2",
                      "--lr-lat", "3e-3", "--lr-gamma", "3e-2", "--lr-r", "1e-3"], dati, out, E, Wout, W)
        assert rc == 0, rc
        reg = open(os.path.join(out, "registro.log")).read()
        assert "AL BIT" in reg, reg[-2000:]
        assert "held-out ('logiqa', 'medmcqa')" in reg
        e = json.load(open(os.path.join(out, "esportazione.json")))
        k0, k1 = e["held0"]["loss"], e["held"]["loss"]
        assert set(e["held0"]["per_fonte"]) == {"logiqa", "medmcqa", "tutte"}, e["held0"]["per_fonte"].keys()
        assert e["held0"]["n"] == 4 * 12                 # 6 + 6 gruppi LogiQA/MedMCQA dell held-out
        # la PIF held-out usa l'oro, che nel mondo finto e' scorrelato dal contenuto (irriducibile): si guarda la KD
        kd0, kd1 = e["held0"]["kd"], e["held"]["kd"]
        assert k1 < k0 and kd1 < 0.9 * kd0, (modo, k0, k1, kd0, kd1)
        info = json.load(open(os.path.join(out, "residui.json")))
        z = np.load(os.path.join(out, "residui_tq1.npz"))
        # strati 1-3: 1, 2 GDN (qkv, gate, ssm_out) + 3 full (q, k, v, o) = 10 proiezioni x (lra, lrb)
        assert len(z.files) == 20 == len(info["tensori"]), sorted(z.files)
        assert "blk.3.attn_q_lra.weight" in info["tensori"] and "blk.1.ssm_out_lrb.weight" in info["tensori"]
        if modo == "entrambi":
            assert len(np.load(os.path.join(out, "rexp_tq1.npz")).files) == 2 * 9
        # salvataggio a ogni passo (e1_corsa.sh) + esportazione di ripiego dai latenti del migliore == driver, al byte
        ult = torch.load(os.path.join(out, "residui_ultimo.pt"), map_location="cpu")
        assert ult["meta"]["passo"] == 25 and not os.path.exists(os.path.join(out, "residui_ultimo_tmp.pt"))
        import e1_esporta_residui as EX
        rip = os.path.join(out, "ripiego")
        os.makedirs(rip)
        EX.esporta(os.path.join(out, "residui_migliore.pt"), rip)
        z2 = np.load(os.path.join(rip, "residui_tq1.npz"))
        assert sorted(z2.files) == sorted(z.files) and all(np.array_equal(z[k], z2[k]) for k in z.files)
        assert json.load(open(os.path.join(rip, "residui.json")))["tensori"] == info["tensori"]
        print(f"driver {modo}: held loss {k0:.4f} -> {k1:.4f}, kd {kd0:.4f} -> {kd1:.4f},acc {e['held0']['acc']:.3f} -> {e['held']['acc']:.3f}")
    # rifiuto delle fonti vietate (ARC) in allenamento
    dati2 = os.path.join(d, "items_arc.jsonl")
    _dati(dati2, E, Wout, TE, 4, 4, fonte_extra="ai2_arc")
    try:
        _esegui(["--modo", "residui", "--passi", "1"], dati2, os.path.join(d, "arc"), E, Wout, W)
        raise AssertionError("righe ARC non rifiutate")
    except SystemExit as ex:
        assert "ARC" in str(ex), ex
    return d


# ═══════════════════════════════════ GGUF: modifica nulla e tensori in piu' ═══════════════════════════════════
def _gguf_finto(path):
    import gguf
    w = gguf.GGUFWriter(path, "qwen35moe")
    w.add_uint32("general.alignment", 32)
    w.add_string("general.name", "finto")
    w.add_array("finto.lista", [1, 2, 3])
    w.add_tensor("token_embd.weight", np.random.default_rng(0).standard_normal((16, 256)).astype(np.float32))
    q = RT.byte_tq1_0(np.sign(np.random.default_rng(1).standard_normal((8, 512))).astype(np.float32) * 0.5)
    w.add_tensor("blk.0.attn_q.weight", q, raw_dtype=gguf.GGMLQuantizationType.TQ1_0)
    w.add_tensor("output_norm.weight", np.ones(256, dtype=np.float32) * 3)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def test_gguf_nulla_e_aggiunta(tmp=None):
    import gguf
    d = tmp or tempfile.mkdtemp()
    src = os.path.join(d, "src.gguf")
    _gguf_finto(src)
    ix = WG.indice(src)
    assert ix["n_tensori"] == 3 and ix["allineamento"] == 32
    # modifica nulla: stessa sha256 (stesso cammino di codice della modifica vera)
    WG.scrivi(src, os.path.join(d, "nulla.gguf"), [], None)
    assert _sha(src) == _sha(os.path.join(d, "nulla.gguf"))
    assert WG.main(["--src", src, "--out", os.path.join(d, "nulla2.gguf"), "--nulla", "--min-free-gb", "0"]) == 0
    # tensori in piu' (residui veri) + KV
    cfg = RT.ConfigResidui(rango=16, strati=(0,), famiglie=("attn_q",))
    res = RT.ResiduoTernario("blk.0.attn_q", 512, 256, cfg, seme=9)
    with torch.no_grad():
        res.gamma.uniform_(0.5, 1.5)
    t, info = RT.tensori_gguf({"blk.0.attn_q": res})
    np.savez(os.path.join(d, "r.npz"), **t)
    Path(d, "r.json").write_text(json.dumps({"tensori": info}))
    out = os.path.join(d, "piu.gguf")
    assert WG.main(["--src", src, "--out", out, "--npz", os.path.join(d, "r.npz"), "--info",
                    os.path.join(d, "r.json"), "--meta", "--min-free-gb", "0"]) == 0
    a, b = gguf.GGUFReader(src), gguf.GGUFReader(out)
    ta = {t.name: t for t in a.tensors}
    tb = {t.name: t for t in b.tensors}
    assert set(tb) == set(ta) | set(info)
    for n in ta:
        assert np.array_equal(np.asarray(ta[n].data).view(np.uint8), np.asarray(tb[n].data).view(np.uint8)), n
    (Ap, Bp), = RT.matrici_file(res)
    for n, M in (("blk.0.attn_q_lra.weight", Ap), ("blk.0.attn_q_lrb.weight", Bp)):
        tt = tb[n]
        assert int(tt.tensor_type) == 34 and list(map(int, tt.shape)) == info[n]["ne"]
        deq = gguf.quants.dequantize(np.asarray(tt.data), gguf.GGMLQuantizationType.TQ1_0).reshape(M.shape)
        assert np.array_equal(deq, M), n
    assert WG.KV_META in b.fields
    # rifiuti: nome gia' presente, taglia sbagliata, out esistente
    for nuovi, frase in (([("blk.0.attn_q.weight", 34, [256, 4], np.zeros((4, 54), np.uint8))], "esiste"),
                         ([("x.weight", 34, [256, 4], np.zeros((3, 54), np.uint8))], "attesi")):
        try:
            WG.scrivi(src, os.path.join(d, "no.gguf"), nuovi)
            raise AssertionError(frase)
        except SystemExit as e:
            assert frase in str(e), e
    try:
        WG.scrivi(src, out, [])
        raise AssertionError("sovrascritto")
    except SystemExit as e:
        assert "esiste" in str(e)
    vero = os.environ.get("E1_GGUF_VERO")
    if vero:
        o = os.path.join(os.environ.get("E1_GGUF_VERO_DIR", os.path.dirname(vero)), "nulla_e1_prova.gguf")
        try:
            WG.main(["--src", vero, "--out", o, "--nulla"])
            print("modifica nulla sul GGUF vero: sha identica", json.load(open(o + ".modifica.json"))["sha256_out"])
        finally:
            for p in (o, o + ".modifica.json"):
                if os.path.exists(p):
                    os.remove(p)
    return d


if __name__ == "__main__":
    torch.set_num_threads(8)
    fn = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)
          and (len(sys.argv) < 2 or any(s in k for s in sys.argv[1:]))]
    for k, f in fn:
        f()
        print("ok", k, flush=True)
    print(f"TUTTE LE {len(fn)} PROVE OK")

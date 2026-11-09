#!/usr/bin/env python3
"""test_rilq_piani_extra.py — 9/10: prove CPU delle cure P1/P2/P3 di RILQT/COMBO_A_shexp/DIAGNOSI_DIVARIO.md.

Nessuna GPU, nessun modello intero: famiglie ternarie SINTETICHE (byte TQ1_0R_1024 dal codec della forgia, TQ1_0 da
gguf-py) e lo studente finto di test_rilq_thor_shexp per il driver.
  P1a  secondo piano degli esperti (exps2 + mappa, slot -1 = nessuno): l'uscita del MoEStudente coincide col riferimento
       a semantica del MOTORE (mm(W_e,x) + [map[e]>=0] mm(W2[map[e]],x) per gate/up/down, prima di silu*up e del peso
       del router), anche il gradiente rispetto a x; con un 2° piano sul solo down la differenza con/senza e' ESATTAMENTE
       sum_j w_j W2[map[e_j]] a_j.
  P1b  quarto piano dei densi: LineareTernaria a 4 piani == x (W1+W2+W3+W4)^T; togliendo il 4° la differenza e' x W4^T;
       voci_dense lo trova, il cammino dequantizzato somma W4.
  P1c  cancello di copertura: un tensore non letto fa fallire; nextn (blk.L >= n_strati) e strati non caricati sono in
       lista bianca; exps2 e mappa risultano letti da secondi_piani_esperti.
  P1d  cancello di coerenza torch <-> motore in rilq_thor (--kl-partenza-motore): scarto > 3 % -> codice 2, niente passi.
  P2   KL NaN / KL inf / gradiente NaN / gradiente inf: passo saltato con messaggio distinto, parametri e stato di Adam
       IDENTICI a quelli senza il passo; 3 salti di fila -> ricarica dell'ultimo salvataggio a lr dimezzato;
       piano_ternario_g distingue NaN da overflow.
  P3   l'etichetta del maestro e' il nome di --teacher (niente "q8-TT6b" cablato).
Uso: HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= OMP_NUM_THREADS=6 nice -n 15 \\
       @@ROOT:venv@@/bin/python3 test_rilq_piani_extra.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import types
from pathlib import Path

for _v in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
    os.environ[_v] = ""
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

QUI = Path(__file__).resolve().parent
REC = QUI / "recupero_7dabd6f" / "thor_v2"
sys.path.insert(0, str(QUI))
sys.path.insert(1, str(REC))
import residui_ternari as RT  # noqa: E402
import e2e_studente as S  # noqa: E402
from gguf import GGMLQuantizationType as GT  # noqa: E402
from tq1_0r_1024_codec import quantize_rows, dequantize_blocks  # noqa: E402

torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))
H, FF, E, K = 1024, 256, 4, 2          # gate/up TQ1_0R_1024 (ingresso H), down TQ1_0 (ingresso FF), come Thor
RIC = S.RICETTE["B64"]                  # stessa algebra della ricetta B (ingresso ruotato) in float64
MOT = S.motore("torch")


# ═══════════════════════════════════ tensori sintetici ═══════════════════════════════════
def _tens(nome, tipo, raw, ne):
    return types.SimpleNamespace(name=nome, tensor_type=GT[tipo], data=raw, shape=np.array(ne))


def _byte(tipo: str, W: np.ndarray) -> np.ndarray:
    R, C = W.shape
    if tipo == "TQ1_0R_1024":
        return quantize_rows(W.astype(np.float32)).reshape(R, -1)
    return RT.byte_tq1_0(W.astype(np.float32)).reshape(R, -1)


def _deq(tipo: str, raw: np.ndarray, C: int) -> np.ndarray:
    if tipo == "TQ1_0R_1024":
        return dequantize_blocks(raw.reshape(-1, raw.shape[-1])).reshape(-1, C).astype(np.float64)
    return RT.deq_tq1_0(raw.reshape(-1, raw.shape[-1]), C).astype(np.float64)


def _esperti(rng, n, R, C, tipo, scala=0.05):
    raw = np.stack([_byte(tipo, rng.standard_normal((R, C)) * scala) for _ in range(n)])
    return raw, np.stack([_deq(tipo, raw[e], C) for e in range(n)])


TIPI = {"gate": ("TQ1_0R_1024", FF, H), "up": ("TQ1_0R_1024", FF, H), "down": ("TQ1_0", H, FF)}


def _strato(seme: int, secondi: dict):
    """tens finto dello strato 0 con gate/up/down (E esperti) e, per le proiezioni in `secondi` ({nome: mappa}), il
    2° piano exps2 (E2 = max(mappa)+1 slot) + la mappa F32. Ritorna (tens, pesi dequantizzati, mappe)."""
    rng = np.random.default_rng(seme)
    tens, W, W2, mappe = S.TensoriTracciati(), {}, {}, {}
    for n, (tipo, R, C) in TIPI.items():
        raw, W[n] = _esperti(rng, E, R, C, tipo)
        nf = S.FAMIGLIE[n]
        tens[f"blk.0.{nf}.weight"] = _tens(f"blk.0.{nf}.weight", tipo, raw, [C, R, E])
        if n in secondi:
            m = np.asarray(secondi[n], dtype=np.float32)
            E2 = int(m.max()) + 1
            raw2, W2[n] = _esperti(rng, E2, R, C, tipo, scala=0.03)
            tens[f"blk.0.{nf}2.weight"] = _tens(f"blk.0.{nf}2.weight", tipo, raw2, [C, R, E2])
            tens[f"blk.0.{nf}2_map"] = _tens(f"blk.0.{nf}2_map", "F32", m, [E])
            mappe[n] = m.astype(np.int64)
    return tens, W, W2, mappe


def _moe(tens, seme_mod: int):
    fam = {n: S.famiglia_da_tensore(tens[f"blk.0.{S.FAMIGLIE[n]}.weight"], n) for n in S.FAMIGLIE}
    es = S.StratoEsperti(fam, RIC, MOT)
    for n, (f2, m2) in S.secondi_piani_esperti(tens, 0, fam, "cpu").items():
        es.fam2[n], es.mappa2[n] = f2, m2
    g = torch.Generator().manual_seed(seme_mod)
    router = torch.randn(E, H, generator=g, dtype=torch.float64) * 0.05
    sh = torch.nn.Linear(H, H, bias=False).double()
    shg = torch.nn.Linear(H, 1, bias=False).double()
    with torch.no_grad():
        sh.weight.copy_(torch.randn(H, H, generator=g, dtype=torch.float64) * 0.01)
        shg.weight.copy_(torch.randn(1, H, generator=g, dtype=torch.float64) * 0.05)
    for p in (*sh.parameters(), *shg.parameters()):
        p.requires_grad_(False)
    return S.MoEStudente(router, sh, shg, es, K), router, sh, shg


def _riferimento_motore(x, router, sh, shg, W, W2, mappe, con_secondi=True):
    """Semantica di build_moe_ffn del motore, in float64 e con autograd: softmax, top-k, rinormalizzazione; per ogni
    assegnazione up/gate/down = mm(W_e) + mm(W2[map[e]]) se map[e] >= 0 (piani sommati come USCITE, non come pesi)."""
    prob = torch.softmax(x @ router.T, -1)
    idx = torch.topk(prob, K, -1).indices
    w = prob.gather(-1, idx)
    w = w / w.sum(-1, keepdim=True)
    Wt = {n: torch.from_numpy(v) for n, v in W.items()}
    W2t = {n: torch.from_numpy(v) for n, v in W2.items()}
    righe = []
    for t in range(x.shape[0]):
        acc = torch.zeros(H, dtype=torch.float64)
        for j in range(K):
            e = int(idx[t, j])

            def mm(n, v):
                y = Wt[n][e] @ v
                if con_secondi and n in W2t and mappe[n][e] >= 0:
                    y = y + W2t[n][int(mappe[n][e])] @ v
                return y
            u = mm("up", x[t])
            gg = mm("gate", x[t])
            a = F.silu(gg) * u
            acc = acc + w[t, j] * mm("down", a)
        righe.append(acc)
    return torch.stack(righe) + torch.sigmoid(shg(x)) * sh(x), idx, w


def _rel(a, b):
    a, b = a.detach(), b.detach()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-300))


# ═══════════════════════════════════ P1a: secondo piano degli esperti ═══════════════════════════════════
def test_secondo_piano_esperti():
    secondi = {"gate": [1, -1, 0, -1], "up": [-1, 0, -1, 1], "down": [2, 0, -1, 1]}
    tens, W, W2, mappe = _strato(11, secondi)
    moe, router, sh, shg = _moe(tens, 3)
    assert set(moe.esperti.fam2) == {"gate", "up", "down"}
    assert {n: moe.esperti.fam2[n].E for n in secondi} == {"gate": 3, "up": 3, "down": 4}   # E2 + esperto nullo
    g = torch.Generator().manual_seed(5)
    x = torch.randn(12, H, generator=g, dtype=torch.float64)
    xs = x.clone().requires_grad_(True)
    y = moe(xs[None])[0]
    xr = x.clone().requires_grad_(True)
    ref, idx, _ = _riferimento_motore(xr, router, sh, shg, W, W2, mappe)
    ref0, _, _ = _riferimento_motore(x, router, sh, shg, W, W2, mappe, con_secondi=False)
    assert torch.equal(moe.ultimo_idx, idx)
    e_con, e_senza = _rel(y.detach(), ref.detach()), _rel(y.detach(), ref0)
    print(f"P1a MoE: |studente - motore| rel {e_con:.2e}; contro motore SENZA exps2 {e_senza:.2e}")
    assert e_con < 1e-5 and e_senza > 1e-2, (e_con, e_senza)
    torch.manual_seed(0)
    gy = torch.randn_like(y.detach())
    (y * gy).sum().backward()
    (ref * gy).sum().backward()
    e_g = _rel(xs.grad, xr.grad)
    print(f"P1a MoE: gradiente rispetto a x rel {e_g:.2e}")
    assert e_g < 1e-5, e_g
    # senza 2° piani: identico al bit allo studente di sempre (stessi byte del primo piano, nessun tensore exps2)
    tens0 = S.TensoriTracciati({k: v for k, v in tens.items() if "exps2" not in k})
    moe0, *_ = _moe(tens0, 3)
    assert not moe0.esperti.fam2
    y0 = moe0(x[None])[0]
    assert _rel(y0, ref0) < 1e-5


def test_secondo_piano_solo_down_e_somma_esatta():
    """Un 2° piano sul SOLO down: y_con - y_senza == sum_j w_j W2[map[e_j]] a_j (a = silu(gate) * up del primo piano)."""
    m = [-1, 1, 0, -1]
    tens, W, W2, mappe = _strato(21, {"down": m})
    moe, router, sh, shg = _moe(tens, 4)
    tens0 = S.TensoriTracciati({k: v for k, v in tens.items() if "exps2" not in k})
    moe0, *_ = _moe(tens0, 4)
    x = torch.randn(9, H, generator=torch.Generator().manual_seed(6), dtype=torch.float64)
    with torch.no_grad():
        d = moe(x[None])[0] - moe0(x[None])[0]
        _, idx, w = _riferimento_motore(x, router, sh, shg, W, W2, mappe, con_secondi=False)
        att = torch.zeros_like(d)
        Wt = {n: torch.from_numpy(v) for n, v in W.items()}
        W2d = torch.from_numpy(W2["down"])
        for t in range(x.shape[0]):
            for j in range(K):
                e = int(idx[t, j])
                if m[e] >= 0:
                    a = F.silu(Wt["gate"][e] @ x[t]) * (Wt["up"][e] @ x[t])
                    att[t] += w[t, j] * (W2d[m[e]] @ a)
    err = float((d - att).abs().max() / att.abs().max())
    print(f"P1a solo down: (con - senza) contro somma attesa, rel {err:.2e}; token toccati "
          f"{int((att.abs().sum(-1) > 0).sum())}/{x.shape[0]}")
    assert err < 1e-5, err
    assert float(att.abs().max()) > 0
    # i token il cui top-k non tocca esperti col 2° piano restano IDENTICI (maschera esatta: esperto nullo d = 0)
    for t in range(x.shape[0]):
        if all(m[int(e)] < 0 for e in idx[t]):
            assert float(d[t].abs().max()) == 0.0


def test_mappa_non_valida():
    for cattiva in ([0, 0, -1, -1], [0, 5, -1, -1], [0.5, -1, -1, -1], [-2, -1, -1, 0]):
        tens, *_ = _strato(31, {"gate": [1, -1, 0, -1]})
        tens["blk.0.ffn_gate_exps2_map"].data = np.asarray(cattiva, dtype=np.float32)
        fam = {n: S.famiglia_da_tensore(tens[f"blk.0.{S.FAMIGLIE[n]}.weight"], n) for n in S.FAMIGLIE}
        try:
            S.secondi_piani_esperti(tens, 0, fam, "cpu")
        except ValueError:
            continue
        raise AssertionError(f"mappa {cattiva} accettata")


# ═══════════════════════════════════ P1b: quarto piano dei densi ═══════════════════════════════════
def test_quarto_piano_densi():
    rng = np.random.default_rng(41)
    R, C, tipo = 64, H, "TQ1_0R_1024"
    raws, Ws = [], []
    for k in range(4):
        raw = _byte(tipo, rng.standard_normal((R, C)) * (0.05 / (k + 1)))
        raws.append(raw)
        Ws.append(torch.from_numpy(_deq(tipo, raw, C)))
    nomi = ["attn_q", "attn_q2", "attn_q3", "attn_q4"]
    tens = S.TensoriTracciati({f"blk.0.{n}.weight": _tens(f"blk.0.{n}.weight", tipo, raws[i], [C, R])
                               for i, n in enumerate(nomi)})
    voci = S.voci_dense(tens, 0, True, ("attn_q",))
    assert list(voci) == nomi, list(voci)                  # il *4 arriva DOPO i *3, come nelle chiavi
    assert S.famiglie_dense("attn_q4") == ("attn_q",) and S.primo_piano("attn_q4") == "attn_q"
    assert S.nome_tensore_famiglia("attn_q4") == "attn_q4" and S.e_densa("ssm_out4")
    tens_g, *_ = _strato(42, {})
    fam = {n: S.famiglia_da_tensore(tens_g[f"blk.0.{S.FAMIGLIE[n]}.weight"], n) for n in S.FAMIGLIE}
    for n, (nome_g, _) in voci.items():
        fam[n] = S.famiglia_da_tensore(tens[nome_g], n)
    es = S.StratoEsperti(fam, RIC, MOT)
    voce = {"rel": "identita"}
    lin4 = S.LineareTernaria(es, "attn_q", voce, R, C, nome2="attn_q2", nome3="attn_q3", nome4="attn_q4")
    lin3 = S.LineareTernaria(es, "attn_q", voce, R, C, nome2="attn_q2", nome3="attn_q3")
    assert "quarto_piano=attn_q4" in lin4.extra_repr()
    x = torch.randn(7, C, generator=torch.Generator().manual_seed(8), dtype=torch.float64)
    with torch.no_grad():
        y4, y3 = lin4(x), lin3(x)
    att = x @ (Ws[0] + Ws[1] + Ws[2] + Ws[3]).T
    e4 = _rel(y4, att)
    ed = _rel(y4 - y3, x @ Ws[3].T)
    print(f"P1b quarto piano: LineareTernaria(4 piani) contro x(W1+W2+W3+W4)^T rel {e4:.2e}; (4 piani - 3 piani) contro "
          f"x W4^T rel {ed:.2e}; peso del 4° piano {float((x @ Ws[3].T).abs().max() / att.abs().max()):.2e}")
    assert e4 < 1e-5 and ed < 1e-5, (e4, ed)
    # cammino dequantizzato (famiglia densa NON chiesta): il Linear HF riceve ((W1 + W2) + W3) + W4
    import carica_strato_gguf as CSG
    sd = S.pesi_non_esperti(tens, 0, True)
    hf = CSG.MAPPA["full"]["attn_q.weight"]["hf"]
    ed2 = _rel(sd[hf].double(), Ws[0] + Ws[1] + Ws[2] + Ws[3])
    print(f"P1b cammino dequantizzato con il 4° piano: rel {ed2:.2e}")
    assert ed2 < 1e-6, ed2
    assert S.piani_di(tens, "blk.0.attn_q.weight") == [f"blk.0.{n}.weight" for n in nomi]
    # densi piegati (E2E_DENSI_PIEGATI=x2, quello della corsa) con 4 piani: piega davvero i 4 e coincide con la via dei
    # 4 prodotti separati (ricetta B, fp16 esatto in ingresso) entro l'errore dichiarato di e2e_densi_piegati
    import e2e_densi_piegati as DP
    es.ric = S.RICETTE["B"]
    xf = x.float()
    with torch.no_grad():
        y_sep = lin4(xf)
        DP.imposta(modo="x2", gemm="f32")
        try:
            p0 = DP.CONTI["piegati"]
            y_dp = lin4(xf)
            assert DP.CONTI["piegati"] == p0 + 1
        finally:
            DP.imposta(modo="")
    e_dp = _rel(y_dp, y_sep)
    print(f"P1b densi piegati x2 a 4 piani contro 4 prodotti separati: rel {e_dp:.2e}")
    assert e_dp < 1e-5, e_dp


# ═══════════════════════════════════ P1c: cancello di copertura ═══════════════════════════════════
def test_cancello_copertura():
    tens = S.TensoriTracciati({n: object() for n in (
        "blk.0.attn_q.weight", "blk.0.ffn_gate_exps2.weight", "blk.40.nextn.eh_proj.weight", "blk.5.attn_q.weight",
        "token_embd.weight", "output.weight", "output_norm.weight")})
    assert "blk.0.attn_q.weight" in tens and not tens.letti            # `in` non conta come uso
    _ = tens["blk.0.attn_q.weight"]
    dopo = ["token_embd.weight", "output.weight", "output_norm.weight"]
    try:
        S.controlla_copertura(tens, tens.letti, [0], 40, dopo)
    except ValueError as e:
        assert "blk.0.ffn_gate_exps2.weight" in str(e) and "1 tensori" in str(e), e
    else:
        raise AssertionError("un tensore non letto doveva far fallire il cancello")
    tens.get("blk.0.ffn_gate_exps2.weight")
    bianchi = S.controlla_copertura(tens, tens.letti, [0], 40, dopo)
    assert sorted(bianchi) == ["blk.40.nextn.eh_proj.weight", "blk.5.attn_q.weight"], bianchi
    # un tensore GLOBALE sconosciuto non ha lista bianca
    tens["rope_freqs.weight"] = object()
    try:
        S.controlla_copertura(tens, tens.letti, [0], 40, dopo)
    except ValueError as e:
        assert "rope_freqs.weight" in str(e)
    else:
        raise AssertionError("tensore globale non letto accettato")
    # secondi_piani_esperti legge davvero exps2 e mappe (cosi' il cancello li vede usati); senza di lui restano orfani
    tens2, *_ = _strato(51, {"gate": [1, -1, 0, -1], "up": [0, -1, -1, -1], "down": [-1, -1, -1, 0]})
    fam = {n: S.famiglia_da_tensore(tens2[f"blk.0.{S.FAMIGLIE[n]}.weight"], n) for n in S.FAMIGLIE}
    try:
        S.controlla_copertura(tens2, tens2.letti, [0], 40)
    except ValueError as e:
        assert "exps2" in str(e) and "6 tensori" in str(e), e
    else:
        raise AssertionError("exps2 non letti accettati (era il difetto di COMBO A)")
    S.secondi_piani_esperti(tens2, 0, fam, "cpu")
    assert S.controlla_copertura(tens2, tens2.letti, [0], 40) == []
    print("P1c cancello di copertura: orfano -> errore; nextn e strati non caricati in lista bianca; exps2 letti")


# ═══════════════════════════════════ driver rilq_thor: P1d, P2, P3 ═══════════════════════════════════
class _Avvelena(torch.autograd.Function):
    """Avanti identita'; indietro: gradiente moltiplicato per `valore` (NaN o inf)."""

    @staticmethod
    def forward(ctx, x, valore):
        ctx.valore = valore
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g * ctx.valore, None


def _esegui(out, passi=2, guasti=None, extra=(), teacher="q8-maestro-finto", studente_rotto=False):
    import test_rilq_thor_shexp as TS
    import rilq_thor as RQ
    g = torch.Generator().manual_seed(5)
    WS = torch.randn(TS.V, TS.H, generator=g) / TS.H ** 0.5
    WT = WS + 0.3 * torch.randn(TS.V, TS.H, generator=g) / TS.H ** 0.5
    maestro = TS._fabbrica(99)()
    fS = types.SimpleNamespace(Studente=TS._fabbrica(7))
    if studente_rotto:                                  # come Studente col cancello di copertura che scatta
        def _rotto(**kw):
            raise ValueError("studente: 308 tensori del GGUF non rappresentati nel forward (es. ['blk.0.ffn_gate_exps2'])")
        fS.Studente = _rotto
    fA = TS._allena_finto(WS, WT, maestro)
    guasti = guasti or {}
    chiamate = [0]

    class KLGuasta:
        @staticmethod
        def apply(hS, WS_, hT, WT_, pezzo):
            chiamate[0] += 1
            m = guasti.get(chiamate[0])
            if m in ("grad_nan", "grad_inf"):
                hS = _Avvelena.apply(hS, float("nan") if m == "grad_nan" else 3e38)   # 3e38: norma inf senza 0*inf
            kl = TS._kl(hS, WS_, hT, WT_).sum()
            if m == "kl_nan":
                kl = kl + float("nan")
            elif m == "kl_inf":
                kl = kl + float("inf")
            return kl
    fA.KLFusa = KLGuasta
    old = {k: sys.modules.get(k) for k in ("e2e_studente", "e2e_allena")}
    sys.modules["e2e_studente"], sys.modules["e2e_allena"] = fS, fA
    perm_vero, cicli_vero = RT._perm, RQ.CICLI
    RT._perm = lambda chiave, fn: np.random.default_rng(sum(map(ord, chiave))).permutation(TS.H)
    os.makedirs(out, exist_ok=True)
    RQ.CICLI = os.path.join(out, "cicli_finto.tsv")
    try:
        torch.manual_seed(0)
        return RQ.main(["--uscita", out, "--gguf", "/dev/null", "--nome", "SINT", "--teacher", f"/finto/{teacher}",
                        "--strati", "0-3", "--famiglie", "attn,gdn,shexp", "--rango", "16", "--passi", str(passi),
                        "--heldout-righe", "3", "--heldout-ogni", "1", "--lr", "3e-3", "--lr-gamma", "3e-2",
                        "--riscaldamento", "1", "--device", "cpu", "--pezzo-kl", "64", *extra])
    finally:
        RT._perm, RQ.CICLI = perm_vero, cicli_vero
        for k, v in old.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _uguali(a, b, dove=""):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and torch.equal(a, b), dove
    elif isinstance(a, dict):
        assert set(a) == set(b), (dove, set(a) ^ set(b))
        for k in a:
            _uguali(a[k], b[k], f"{dove}/{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), dove
        for i, (x, y) in enumerate(zip(a, b)):
            _uguali(x, y, f"{dove}[{i}]")


def _stato(out):
    lat = torch.load(os.path.join(out, "residui_ultimo.pt"), map_location="cpu", weights_only=False)
    opt = torch.load(os.path.join(out, "ottimizzatore_ultimo.pt"), map_location="cpu", weights_only=False)["opt"]
    lat = {k: v for k, v in lat.items() if k != "meta"}
    return lat, opt["state"]


def test_driver_salti_ricarica_coerenza_etichetta():
    d = tempfile.mkdtemp(prefix="rilq_cure_")
    rif = os.path.join(d, "rif_1passo")
    assert _esegui(rif, passi=1) == 0
    reg = open(os.path.join(rif, "registro.log")).read()
    cic = open(os.path.join(rif, "cicli_finto.tsv")).read()
    # P3: etichetta dal nome di --teacher
    assert "q8-maestro-finto" in cic and "q8-TT6b" not in cic and "q8-TT6b" not in reg, cic
    lat1, opt1 = _stato(rif)
    # P2: ogni guasto al passo 2 -> passo saltato, stato IDENTICO a quello dopo il solo passo 1
    for guasto, msg in (("kl_nan", "NaN nella KL"), ("kl_inf", "overflow della KL"),
                        ("grad_nan", "NaN nel gradiente"), ("grad_inf", "overflow del gradiente")):
        o = os.path.join(d, guasto)
        assert _esegui(o, passi=2, guasti={2: guasto}) == 0
        r = open(os.path.join(o, "registro.log")).read()
        assert f"PASSO SALTATO ({msg}" in r, r[-1500:]
        lat, opt = _stato(o)
        _uguali(lat, lat1, guasto)
        _uguali(opt, opt1, guasto)
        print(f"P2 {guasto}: saltato con «{msg}», parametri e stato di Adam identici al passo 1")
    # P2: 3 salti di fila -> ricarica dell'ultimo salvataggio (passo 1) a lr dimezzato
    o = os.path.join(d, "tre_salti")
    assert _esegui(o, passi=4, guasti={2: "grad_nan", 3: "kl_nan", 4: "grad_nan"}) == 0
    r = open(os.path.join(o, "registro.log")).read()
    assert "RICARICA al passo 4 dopo 3 salti di fila: residui_ultimo.pt (passo 1)" in r and "lr x0.5" in r, r[-2000:]
    lat, opt = _stato(o)
    _uguali(lat, lat1, "tre_salti")
    _uguali(opt, opt1, "tre_salti")
    meta = torch.load(os.path.join(o, "ottimizzatore_ultimo.pt"), weights_only=False)
    assert meta["fattore_lr"] == 0.5, meta.get("fattore_lr")
    o5 = os.path.join(d, "tre_salti_poi_passo")
    assert _esegui(o5, passi=5, guasti={2: "grad_nan", 3: "grad_nan", 4: "grad_nan"}) == 0
    r5 = open(os.path.join(o5, "registro.log")).read()
    assert '"passo": 5, "kl_train"' in r5, r5[-1500:]
    print("P2 tre salti: ricaricato il passo 1, lr x0.5, il passo 5 riparte")
    # P1d: cancello di coerenza torch <-> motore
    kl0 = float(reg.split("KL al passo 0 = ")[1].split()[0])
    ok = os.path.join(d, "coerente")
    assert _esegui(ok, passi=1, extra=("--kl-partenza-motore", f"{kl0 * 1.02:.8f}")) == 0
    assert "coerenza: KL torch passo 0" in open(os.path.join(ok, "registro.log")).read()
    ko = os.path.join(d, "incoerente")
    assert _esegui(ko, passi=1, extra=("--kl-partenza-motore", f"{kl0 / 1.66:.8f}")) == 2
    rk = open(os.path.join(ko, "registro.log")).read()
    assert "lo studente non e' il GGUF" in rk and '"passo": 1' not in rk, rk[-1500:]
    print("P1d coerenza: +2 % passa, +66 % (il caso COMBO A) ferma prima del passo 1 (codice 2)")
    kc = os.path.join(d, "copertura")
    assert _esegui(kc, passi=1, studente_rotto=True) == 2
    assert "ERRORE: studente: 308 tensori del GGUF" in open(os.path.join(kc, "registro.log")).read()
    print("P1c driver: cancello di copertura scattato -> codice 2, niente allenamento")
    import shutil
    shutil.rmtree(d, ignore_errors=True)


def test_piano_ternario_messaggi():
    r = torch.randn(4, 256)
    r[1, 3] = float("nan")
    for v, attesa in ((r, "scala NaN"), (torch.full((2, 256), 1e6), "oltre il massimo fp16"),
                      (torch.where(torch.arange(256) == 0, float("inf"), 1.0).expand(2, 256), "latenti infiniti")):
        try:
            RT.piano_ternario_g(v, 256)
        except FloatingPointError as e:
            assert attesa in str(e), (attesa, str(e))
        else:
            raise AssertionError(attesa)
    print("P2 piano_ternario_g: NaN, overflow della scala e latenti infiniti hanno messaggi distinti")


if __name__ == "__main__":
    test_secondo_piano_esperti()
    test_secondo_piano_solo_down_e_somma_esatta()
    test_mappa_non_valida()
    test_quarto_piano_densi()
    test_cancello_copertura()
    test_piano_ternario_messaggi()
    test_driver_salti_ricarica_coerenza_etichetta()
    print("OK test_rilq_piani_extra")

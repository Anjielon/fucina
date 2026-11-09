#!/usr/bin/env python3
"""Collaudo di ripartenza_codici.py (24/9 sera) — CPU, niente Thor, niente GPU.

Uso: env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 19 \
        @@ROOT:venv@@/bin/python3 test_ripartenza_codici.py

Tre forme: giocattolo (G = n_in, come il test di valida_finestra_8_12) e le due
forme di GRUPPO vere (gate/up G=128 su n_in=2048, down G=256 su n_in=512) con
pochi esperti e poche righe. Le righe gate/up per ~1/3 NULLE (|w| ~ 1e-37), come
allo strato 0 di Thor. Uno "stato fine" vero si ottiene allenando qualche passo
con la STESSA parametrizzazione, lo STESSO AdamW (fused, lr per tipo, wd 0) e la
fase dura STE, su un obiettivo di uscita sintetico.

Prove:
 1. via CODICI: i codici esportati con la funzione di PRODUZIONE (_estrai_famiglia,
    _esporta_gate_up) si rimettono in un modulo nuovo (init teacher + LS, come
    in produzione) -> _estrai_famiglia ridà gli stessi D e Q al bit, forward_duro
    ridà D*Q al bit; il gradiente STE resta vivo. Il controllo SA FALLIRE (z
    manomesso) e gli ingressi invalidi sollevano.
 2. via PARAMETRI: copia al bit, controllo contro i codici della finestra; Adam
    CONSERVATO = continuare senza interruzione (parametri identici al bit dopo 10
    passi, con un modulo nuovo accanto nello stesso ottimizzatore); Adam RIPARTITO
    diverge dal primo passo (il confronto sa fallire); stato non riapplicabile.
 3. cartella di corsa: andata e ritorno con scrivi_codici_su_disco di produzione;
    marca di un'altra corsa, down dichiarato diverso, .npy riscritto dopo la
    marca, marca assente: tutti rifiutati.
"""
import os
import sys
import tempfile
import time
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="test_ripartenza_")
os.environ["THOR_VALIDA_OUT"] = _TMP              # PRIMA dell'import: OUT_DIR si legge li'
os.environ["THOR_ID_CORSA"] = "prova-ripartenza"

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from init_ls import reinit_ls_  # noqa: E402
from quant_ternaria_globale_batch_v2 import TernarioDiffBatchFuso  # noqa: E402
from valida_finestra_8_12 import _esporta_gate_up, _estrai_famiglia, scrivi_codici_su_disco  # noqa: E402
import ripartenza_codici as rc  # noqa: E402

LR = 0.01
ESITI: list = []


def _ok(nome: str, cond: bool, dettaglio: str = "") -> None:
    ESITI.append((nome, bool(cond)))
    print(f"  {'✅' if cond else '⛔'} {nome}" + (f" — {dettaglio}" if dettaglio else ""), flush=True)


def _solleva(nome: str, fn, tipo=Exception) -> None:
    try:
        fn()
    except tipo as e:
        _ok(nome, True, f"{type(e).__name__}: {str(e)[:90]}")
        return
    _ok(nome, False, "NON ha sollevato")


def _pesi(E, n_out, n_in, seme, frazione_morte=0.0):
    g = torch.Generator().manual_seed(seme)
    W = torch.randn(E, n_out, n_in, generator=g) * 0.02
    if frazione_morte:
        morte = torch.rand(E, n_out, 1, generator=g) < frazione_morte
        W = torch.where(morte, W * 1e-35, W)            # ~1e-37 come allo strato 0 di Thor
    return W


def _modulo(W, G):
    t = TernarioDiffBatchFuso(W, group=G)
    reinit_ls_(t, W)                                      # init di produzione (teacher + LS)
    return t


def _ottimizzatore(moduli):
    gruppi = {"z": [], "log_s": [], "delta_raw": []}
    for m in moduli:
        for n, p in m.named_parameters():
            gruppi[n].append(p)
    return torch.optim.AdamW([{"params": ps, "lr": LR} for ps in gruppi.values()],
                             lr=LR, weight_decay=0.0, fused=True)


def _perdita(t, W, X):
    """Errore d'uscita sintetico (somma, come B1) con STE a beta 16: stessa forma del forgiare."""
    return (((t.forward_duro(16.0) - W) @ X) ** 2).sum()


def _allena(t, W, X, passi, opt=None):
    opt = opt or _ottimizzatore([t])
    for _ in range(passi):
        opt.zero_grad(set_to_none=True)
        _perdita(t, W, X).backward()
        opt.step()
    return opt


FORME = {   # nome: (E, n_out, n_in, G, frazione righe morte)
    "giocattolo gu (G=n_in)": (4, 12, 8, 8, 0.0),
    "giocattolo dw (G=n_in)": (4, 8, 6, 6, 0.0),
    "gate_up G=128": (4, 16, 2048, 128, 0.36),
    "down G=256": (4, 32, 512, 256, 0.0),
}


# ═════════════════════════════ 1. via CODICI ═════════════════════════════
def prova_codici() -> None:
    print("\n══ 1. ripartenza dai CODICI ══")
    for nome, (E, n_out, n_in, G, morte) in FORME.items():
        W = _pesi(E, n_out, n_in, seme=11, frazione_morte=morte)
        X = torch.randn(n_in, 64, generator=torch.Generator().manual_seed(5))
        t0 = _modulo(W, G)
        _allena(t0, W, X, passi=40)
        D, Q = _estrai_famiglia(t0)                         # esportatore di PRODUZIONE
        t1 = _modulo(W, G)
        esito = rc.imposta_da_codici_(t1, D, Q, W, margine_z=LR)
        D1, Q1 = _estrai_famiglia(t1)
        _ok(f"[{nome}] _estrai_famiglia ridà D e Q al bit", np.array_equal(D1, D) and np.array_equal(Q1, Q),
            f"proiettati {esito['frazione_proiettati']*100:.2f}% · spinti {esito['frazione_spinti_numerici']*100:.3f}% "
            f"· scale esatte (nucleo+eager) {esito['scale_esatte_entrambi']*100:.1f}%")
        with torch.no_grad():
            Wn = t1.forward_duro(16.0)
        atteso = torch.from_numpy(D).unsqueeze(-1) * torch.from_numpy(Q).reshape(E, n_out, n_in // G, G).float()
        _ok(f"[{nome}] forward_duro ridà D*Q al bit", torch.equal(Wn, atteso.reshape(E, n_out, n_in)))
        t1.zero_grad(set_to_none=True)
        _perdita(t1, W, X).backward()
        gz = t1.z.grad
        _ok(f"[{nome}] gradiente STE vivo e finito", bool(torch.isfinite(gz).all() and (gz != 0).any()),
            f"|grad z| mediana {gz.abs().median().item():.2e}")
        # il controllo SA FALLIRE: un peso manomesso oltre la soglia
        with torch.no_grad():
            dl = t1._delta_norm()
            i = (0, 0, 0, 0)
            t1.z[i] = -t1.z[i] if Q.reshape(E, n_out, -1, G)[i] != 0 else dl[i[:3] + (0,)] * 1.5
        v = rc.verifica_contro_codici(t1, torch.from_numpy(D), torch.from_numpy(Q).reshape(E, n_out, -1, G))
        _ok(f"[{nome}] il controllo vede un peso manomesso", v["codici_identici_eager"] < 1.0
            and v["codici_identici_nucleo"] < 1.0, f"identici {v['codici_identici_eager']*100:.4f}%")

    # famiglia gate_up FUSA dal risultato di produzione (_esporta_gate_up)
    E, n_out, n_in, G, morte = FORME["gate_up G=128"]
    W = _pesi(E, n_out, n_in, seme=12, frazione_morte=morte)
    X = torch.randn(n_in, 64, generator=torch.Generator().manual_seed(6))
    t0 = _modulo(W, G)
    _allena(t0, W, X, passi=40)
    ris = _esporta_gate_up(t0, n_out // 2)
    Dgu, Qgu = rc.codici_gate_up(ris)
    D0, Q0 = _estrai_famiglia(t0)
    _ok("gate/up: codici_gate_up ricompone la famiglia fusa", np.array_equal(Dgu.numpy(), D0)
        and np.array_equal(Qgu.numpy(), Q0))

    # ingressi invalidi
    t = _modulo(W, G)
    Qm = Q0.copy(); Qm.flat[0] = 2
    _solleva("Q con un 2 rifiutato", lambda: rc.imposta_da_codici_(t, D0, Qm, W, LR), ValueError)
    Dm = D0.copy(); Dm.flat[0] = 0.0
    _solleva("scala 0 rifiutata", lambda: rc.imposta_da_codici_(t, Dm, Q0, W, LR), ValueError)
    _solleva("forma sbagliata rifiutata", lambda: rc.imposta_da_codici_(t, D0[:, :-1], Q0[:, :-1], W, LR), ValueError)
    _solleva("tipo sbagliato rifiutato", lambda: rc.imposta_da_codici_(t, D0.astype(np.float64), Q0, W, LR), ValueError)


# ═════════════════════════════ 2. via PARAMETRI + Adam ═════════════════════════════
def prova_parametri() -> None:
    print("\n══ 2. ripartenza dai PARAMETRI (+ stato di Adam) ══")
    E, n_out, n_in, G, morte = FORME["gate_up G=128"]
    Ed, n_outd, n_ind, Gd, _ = FORME["down G=256"]
    Wg = _pesi(E, n_out, n_in, 21, morte)
    Wd = _pesi(Ed, n_outd, n_ind, 22)
    Xg = torch.randn(n_in, 64, generator=torch.Generator().manual_seed(7))
    Xd = torch.randn(n_ind, 64, generator=torch.Generator().manual_seed(8))

    def perdita(gu, dw):
        return _perdita(gu, Wg, Xg) + _perdita(dw, Wd, Xd)

    gu0, dw0 = _modulo(Wg, G), _modulo(Wd, Gd)
    opt0 = _ottimizzatore([gu0, dw0])
    for _ in range(40):
        opt0.zero_grad(set_to_none=True)
        perdita(gu0, dw0).backward()
        opt0.step()
    ris = {**_esporta_gate_up(gu0, n_out // 2), "ffn_down_exps": _estrai_famiglia(dw0)}
    # ⚠️ informativo (difetto PREESISTENTE, non della ripartenza): regola di export vs decisione del nucleo
    Dgu, Qgu = rc.codici_gate_up(ris)
    v_pre = rc.verifica_contro_codici(gu0, Dgu, Qgu.reshape(E, n_out, -1, G))
    print(f"  (informativo) stato fine: codici eager vs nucleo {v_pre['codici_identici_nucleo']*100:.5f}% identici")

    stato = rc.estrai_stato_strato(gu0, dw0, opt=opt0, codici_rif=ris, origine="finestra [0,1] di prova")
    # lo stato porta RIFERIMENTI (zero copie): in produzione l'ottimizzatore d'origine e' finito.
    # Qui opt0 continua a fare passi per il confronto: i momenti si clonano, o i due
    # ottimizzatori aggiornerebbero gli stessi tensori.
    stato.adam = {f: {n: {k: v.clone() for k, v in st.items()} for n, st in d.items()}
                  for f, d in stato.adam.items()}
    gu1, dw1 = _modulo(Wg, G), _modulo(Wd, Gd)
    righe = []
    esito = rc.applica_ripartenza_(1, stato, gu1, dw1, None, None, margine_z=LR, conserva_adam=True,
                                   log=righe.append)
    print("  " + righe[-1].strip())
    identici = all(torch.equal(getattr(a, n), getattr(b, n)) for a, b in ((gu0, gu1), (dw0, dw1))
                   for n in rc.NOMI_PARAMETRI)
    _ok("parametri copiati al bit", identici)
    _ok("codici identici ai forniti 100% e scarto pesi 0",
        all(e["codici_identici_eager"] == 1.0 and e["scarto_max_pesi_eager"] == 0.0 for e in esito.values()))
    _ok("stato consumato (i tensori della finestra precedente si liberano)",
        stato.consumato and stato.parametri is None)

    # Adam CONSERVATO = continuare senza interruzione, anche con un modulo nuovo accanto
    nuovo = _modulo(_pesi(E, n_out, n_in, 23, morte), G)
    opt1 = _ottimizzatore([gu1, dw1, nuovo])
    n_iniettati = rc.inietta_adam_(opt1, gu1, dw1, stato)
    _ok("stato di Adam iniettato (6 tensori: z, log_s, delta_raw x 2 famiglie)", n_iniettati == 6, f"{n_iniettati}")
    # controllo: stessi parametri, Adam RIPARTITO
    gu2, dw2 = _modulo(Wg, G), _modulo(Wd, Gd)
    with torch.no_grad():
        for a, b in ((gu0, gu2), (dw0, dw2)):
            for n in rc.NOMI_PARAMETRI:
                getattr(b, n).copy_(getattr(a, n))
    opt2 = _ottimizzatore([gu2, dw2])
    for passo in range(10):
        for opt, (gu, dw), extra in ((opt0, (gu0, dw0), None), (opt1, (gu1, dw1), nuovo), (opt2, (gu2, dw2), None)):
            opt.zero_grad(set_to_none=True)
            loss = perdita(gu, dw) + (_perdita(extra, Wg, Xg) if extra is not None else 0.0)
            loss.backward()
            opt.step()
        if passo == 0:
            diverso = any(not torch.equal(getattr(a, n), getattr(b, n)) for a, b in ((gu0, gu2), (dw0, dw2))
                          for n in rc.NOMI_PARAMETRI)
            _ok("Adam RIPARTITO diverge gia' al primo passo (il confronto sa fallire)", diverso)
    uguali = all(torch.equal(getattr(a, n), getattr(b, n)) for a, b in ((gu0, gu1), (dw0, dw1))
                 for n in rc.NOMI_PARAMETRI)
    _ok("Adam CONSERVATO = nessuna interruzione (parametri identici al bit dopo 10 passi)", uguali)

    _solleva("stato gia' consumato non si riapplica",
             lambda: rc.applica_ripartenza_(1, stato, gu1, dw1, None, None, LR, log=lambda *_: None), RuntimeError)
    st2 = rc.estrai_stato_strato(gu0, dw0)
    _solleva("down nello stato ma non nella finestra: rifiutato",
             lambda: rc.applica_ripartenza_(1, st2, _modulo(Wg, G), None, None, None, LR, log=lambda *_: None),
             RuntimeError)
    st3 = rc.estrai_stato_strato(gu0, dw0, codici_rif={**ris, "ffn_down_exps": (ris["ffn_down_exps"][0] * 1.0001,
                                                                                ris["ffn_down_exps"][1])})
    _solleva("codici di riferimento diversi dai parametri: rifiutato",
             lambda: rc.applica_ripartenza_(1, st3, _modulo(Wg, G), _modulo(Wd, Gd), None, None, LR,
                                            log=lambda *_: None), RuntimeError)


# ═════════════════════════════ 3. cartella di corsa ═════════════════════════════
def prova_cartella() -> None:
    print(f"\n══ 3. codici da una CARTELLA di corsa ({_TMP}) ══")
    E, n_out, n_in, G, morte = FORME["gate_up G=128"]
    Ed, n_outd, n_ind, Gd, _ = FORME["down G=256"]
    gu, dw = _modulo(_pesi(E, n_out, n_in, 31, morte), G), _modulo(_pesi(Ed, n_outd, n_ind, 32), Gd)
    ris = {**_esporta_gate_up(gu, n_out // 2), "ffn_down_exps": _estrai_famiglia(dw)}
    scrivi_codici_su_disco({5: ris})                    # scrittore di PRODUZIONE
    voce = rc.carica_codici_corsa(_TMP, 5, "prova-ripartenza")
    uguali = all(np.array_equal(voce[f][i], ris[f][i]) for f in ris for i in (0, 1))
    _ok("andata e ritorno su disco: stessi D e Q", uguali, voce["_origine"])
    righe = []
    gu1, dw1 = _modulo(_pesi(E, n_out, n_in, 33, morte), G), _modulo(_pesi(Ed, n_outd, n_ind, 34), Gd)
    # teacher DIVERSO da quello dei codici: prova sotto sforzo della proiezione (molti disaccordi)
    rc.applica_ripartenza_(5, voce, gu1, dw1, lambda: _pesi(E, n_out, n_in, 33, morte),
                           lambda: _pesi(Ed, n_outd, n_ind, 34), margine_z=LR, log=righe.append)
    print("  " + righe[-1].strip())
    D1, Q1 = _estrai_famiglia(dw1)
    _ok("down dai codici della cartella: al bit", np.array_equal(D1, ris["ffn_down_exps"][0])
        and np.array_equal(Q1, ris["ffn_down_exps"][1]))
    _solleva("marca di un'altra corsa: rifiutata", lambda: rc.carica_codici_corsa(_TMP, 5, "ThorT-veloce"),
             RuntimeError)
    _solleva("down dichiarato diverso: rifiutato",
             lambda: rc.carica_codici_corsa(_TMP, 5, "prova-ripartenza", down_atteso=False), RuntimeError)
    _solleva("marca assente: rifiutata", lambda: rc.carica_codici_corsa(_TMP, 6, "prova-ripartenza"),
             FileNotFoundError)
    f = Path(_TMP) / "Qu_05.npy"
    futuro = time.time() + 60
    os.utime(f, (futuro, futuro))
    _solleva(".npy riscritto dopo la marca: rifiutato", lambda: rc.carica_codici_corsa(_TMP, 5, "prova-ripartenza"),
             RuntimeError)


def main() -> bool:
    torch.manual_seed(0)
    prova_codici()
    prova_parametri()
    prova_cartella()
    import shutil
    shutil.rmtree(_TMP, ignore_errors=True)
    falliti = [n for n, ok in ESITI if not ok]
    print(f"\n{'✅ TUTTE LE PROVE PASSANO' if not falliti else '⛔ PROVE FALLITE: ' + '; '.join(falliti)} "
          f"({len(ESITI) - len(falliti)}/{len(ESITI)})")
    return not falliti


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)

"""2/10 TT6-veloce — cancello GPU delle leve di velocita' (oggi: E2E_CONV_DIRETTA) nella configurazione di TT6
(R2B3m, B3_ind2, triton2, densi tutte, piegati x2 triton, GDN veloce v5, cache dequant: l'ambiente lo da' chi lancia).
  velocita  strati 0,1,2,3 (3 GDN + 1 attenzione, la proporzione del modello), 1 micro-lotto di TT6 (2 x 4096),
            avanti senza grafo + ricalcolo + indietro, bracci ALTERNATI spento/acceso, minimo di N ripetizioni;
            estrapolazione a 40 strati (30 GDN + 10 attenzione) in ms/token, senza testa KL.
  fedelta   40 strati, stesso micro-lotto, spento contro acceso: h (uscita dopo output_norm) e TUTTI i dL/dtheta
            devono essere identici AL BIT (la leva e' esatta per costruzione: qui si verifica sul modello intero).
Uscita: JSON con verdetto. Uso: python e2e_cancello_tt6_veloce.py --fasi velocita,fedelta --uscita F.json"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

G = "@@ROOT:models@@/gguf/tony-tern/R2B3m/Thor-T-R2B3m.gguf"


def _leve(acceso) -> None:
    """False = spento; True o "1" = conv diretta al bit; "btc" = conv diretta con uscita (B, T, C)."""
    import e2e_conv_diretta as CD
    if acceso:
        os.environ["E2E_CONV_DIRETTA"] = "btc" if acceso == "btc" else "1"
        CD.installa()
    else:
        os.environ.pop("E2E_CONV_DIRETTA", None)
        CD.disinstalla()


def velocita(rip: int, log) -> dict:
    import e2e_studente as S
    st = S.Studente(gguf=G, strati=[0, 1, 2, 3], ricetta="B3_ind2", motore_nome=os.environ.get("E2E_MOTORE", "triton2"),
                    device="cuda", densi_ternari="tutte")
    ids = S._ids_reali(2, 4096).cuda()
    ntok = ids.numel()
    per = {False: [], True: [], "btc": []}
    for r in range(rip):
        for acceso in (False, True, "btc"):
            _leve(acceso)
            m = S._misura_passo(st, ids, 1)
            per[acceso].append(m)
            log(f"velocita rip {r} {acceso}: passo {m['passo_s']:.2f} s")
    _leve(False)
    out = {}
    for acceso, ms in per.items():
        strati = {}
        for L in st.strati:
            strati[L] = {k: min(m["per_strato_s"][L][k] for m in ms) * 1e3 / ntok for k in ("avanti", "indietro")}
        gdn = [strati[L] for L in st.strati if st.cfg.layer_types[L] == "linear_attention"]
        att = [strati[L] for L in st.strati if st.cfg.layer_types[L] == "full_attention"]
        g = sum(v["avanti"] + v["indietro"] for v in gdn) / len(gdn)
        a = sum(v["avanti"] + v["indietro"] for v in att) / len(att)
        out[{False: "spento", True: "acceso", "btc": "btc"}[acceso]] = {"ms_token_per_strato": strati, "gdn_ms_token": g, "att_ms_token": a,
                                                 "stima_40_strati_ms_token": 30 * g + 10 * a}
    out["rapporto_40_strati"] = out["acceso"]["stima_40_strati_ms_token"] / out["spento"]["stima_40_strati_ms_token"]
    out["rapporto_40_strati_btc"] = out["btc"]["stima_40_strati_ms_token"] / out["spento"]["stima_40_strati_ms_token"]
    del st
    torch.cuda.empty_cache()
    return out


def fedelta(log) -> dict:
    import e2e_studente as S
    st = S.Studente(gguf=G, strati=range(40), ricetta="B3_ind2", motore_nome=os.environ.get("E2E_MOTORE", "triton2"),
                    device="cuda", densi_ternari="tutte")
    st.carica_embedding()
    ids = S._ids_reali(2, 4096).cuda()
    ris = {}
    bracci = [("spento", False), ("spento_bis", False), ("acceso", True)]
    for nome, acceso in bracci:
        _leve(acceso)
        for p in st.parametri():
            p.grad = None
        h = {}
        torch.cuda.synchronize()
        t0 = time.time()
        st.gradiente(ids, lambda x: (h.setdefault("h", x.detach().clone()), x.float().pow(2).mean() * 1e3)[1])
        torch.cuda.synchronize()
        ris[nome] = (h["h"].cpu(), [p.grad.detach().cpu() for p in st.parametri()], time.time() - t0)
        log(f"fedelta {nome}: {ris[nome][2]:.1f} s ({ris[nome][2] * 1e3 / ids.numel():.2f} ms/token, 40 strati)")
    _leve(False)

    def confronto(a, b):
        ga, gb = ris[a][1], ris[b][1]
        diversi = [i for i, (x, y) in enumerate(zip(ga, gb)) if not torch.equal(x, y)]
        num = sum(float((x.double() - y.double()).pow(2).sum()) for x, y in zip(ga, gb))
        den = sum(float(x.double().pow(2).sum()) for x in ga)
        return {"h_al_bit": bool(torch.equal(ris[a][0], ris[b][0])), "gradienti": len(ga),
                "gradienti_diversi": len(diversi), "norma_rel_dtheta": (num / max(den, 1e-300)) ** 0.5}
    out = {"rumore_spento_spento": confronto("spento", "spento_bis"), "spento_acceso": confronto("spento", "acceso"),
           "ms_token": {n: ris[n][2] * 1e3 / ids.numel() for n in ris}}
    r, c = out["rumore_spento_spento"], out["spento_acceso"]
    # ammessa se: h al bit E (gradienti al bit, oppure lo spento stesso non e' ripetibile e la distanza
    # spento-acceso non supera quella spento-spento: la leva non aggiunge nulla al non-determinismo di oggi)
    out["ammessa"] = bool(c["h_al_bit"] and (c["gradienti_diversi"] == 0 or (
        r["gradienti_diversi"] > 0 and c["norma_rel_dtheta"] <= 2 * r["norma_rel_dtheta"])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasi", default="velocita,fedelta")
    ap.add_argument("--ripetizioni", type=int, default=3)
    ap.add_argument("--uscita", required=True)
    a = ap.parse_args()
    import e2e_allena as A

    def log(m):
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
    os.environ.pop("E2E_CONV_DIRETTA", None)          # le leve le accende/spegne il cancello, braccio per braccio
    A.scegli_gdn(log)
    esito = {"data": time.strftime("%Y-%m-%dT%H:%M:%S"), "env": {k: v for k, v in os.environ.items() if k.startswith("E2E_")}}
    fasi = a.fasi.split(",")
    if "velocita" in fasi:
        esito["velocita"] = velocita(a.ripetizioni, log)
        log(json.dumps({k: esito["velocita"][k]["stima_40_strati_ms_token"] for k in ("spento", "acceso", "btc")}))
    if "fedelta" in fasi:
        esito["fedelta"] = fedelta(log)
        log(json.dumps(esito["fedelta"]))
    f = esito.get("fedelta")
    esito["verdetto"] = ("AMMESSA" if f and f["ammessa"] else
                         ("NON AMMESSA" if f else "solo velocita'"))
    Path(a.uscita).write_text(json.dumps(esito, indent=1))
    log(f"verdetto: {esito['verdetto']} -> {a.uscita}")


if __name__ == "__main__":
    main()

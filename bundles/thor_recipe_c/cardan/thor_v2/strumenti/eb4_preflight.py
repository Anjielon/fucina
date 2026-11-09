#!/usr/bin/env python3
"""eb4_preflight — i controlli su CPU PRIMA di lanciare EB4 (28/9/2026, TECNICHE_E2E_PV.md §41).

EB4 = corsa E2E NUOVA dal GGUF riforgiato (strati 20-39 riforgiati sopra l'EMA16 di EB2) con la riga di EB3 v3 e la LoRA
corretta. Tutto qui gira dal CODICE CONGELATO della corsa (lo script sta in <codice>/thor_v2/strumenti/) e su CPU: nessun
avanti, nessuna GPU, solo intestazioni, piani e sigilli del teacher. Lo chiama `lancia_eb4.sh` prima di fermare qualunque
cosa: un controllo rosso costa zero passi di EB3.

Sottocomandi
  righe-auto --teacher T         -> stampa N: la prima riga del primo blocco train SIGILLATO (q8-EB-pieno: blocchi
                                    0000-0003 consumati da EB2 -> 512). Il valore va in --righe-base-da.
  controlla [opzioni] -- ARGV    -> ARGV = gli argomenti di `e2e_allena.py` (senza il programma). Rapporto JSON su stdout
                                    (e in --uscita), exit 0 solo se TUTTI i controlli sono veri:
      argomenti     il parser VERO di e2e_allena.main (parse_args catturato, niente altro eseguito) accetta ARGV; la
                    ConfigAllena che ne esce ha la ricetta del §41 (LoRA init zero SENZA LoRA-GA, lr LoRA 1e-5, strati
                    20-39, AdamW, lr F32 5e-5, righe mc 8 col maestro mc, V dosi/ricentro, ...), e rispetto a config.json
                    di EB3 cambia SOLO i campi ammessi (`DIFF_AMMESSI`);
      righe         il piano delle righe della corsa (primo segmento come lo scrivera' `_segmenti_mc` al passo 0) ha
                    righe base, righe mc e held-out (anche mc) SIGILLATE per almeno --passi-min passi;
      indipendenza  i logit del maestro dipendono solo dai testi (vedi `indipendenza`): lo stesso teacher va bene per
                    uno studente qualunque;
      struttura     il GGUF dello studente ha gli stessi tensori (nome, tipo, forma) del riferimento (la base della
                    riforgia): lo `Studente` di e2e_studente lo legge come quello di EB3.
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

QUI = Path(__file__).resolve().parent.parent            # thor_v2 (del codice congelato)
sys.path.insert(0, str(QUI))
GGUF_PY = "@@ROOT:models@@/build/llamacpp-tq1-k1024/gguf-py"

F32_EB3 = ("attn_norm", "post_attention_norm", "attn_q_norm", "attn_k_norm", "ssm_norm", "ffn_gate_inp",
           "ffn_gate_inp_shexp", "ssm_a", "ssm_alpha", "ssm_beta", "ssm_conv1d", "ssm_dt", "output_norm")
# La ricetta del §41: valori ATTESI nella ConfigAllena di EB4 (campo -> valore). righe_base_da si controlla a parte (> 0).
RICETTA = {
    "lora_init": "zero", "lr_lora": 1e-5, "lora_rango": 16, "lora_tipi": ["Q8_0"], "lora_ripristina": "",
    "lr_f32": 5e-5, "allena_f32": list(F32_EB3), "f32_ripristina": [], "ottimizzatore_theta": "adamw", "lr": 1e-3,
    "strati_allenati": list(range(20, 40)),
    "peso_dominio": [["cot", 1.0], ["edu", 1.5], ["it_chat", 1.0], ["it_pensiero", 1.5], ["it_web", 1.5],
                     ["it_wiki", 1.5], ["strumenti", 1.0]],
    "righe_mc": 8, "righe_mc_permutate": True, "peso_token": "risposta", "beta_risposta": 0.1, "gamma_lettere": 1.0,
    "pv_codici": True, "pv_modo": "vbop", "pv_rho": 0.03, "pv_rho_adattivo": True, "pv_passi_accumulo": 1,
    "pv_accetta": "guadagno", "pv_z_guadagno_min": 1.0, "pv_sonda_dosi": True, "pv_ricentra": True,
    "pv_ricentra_giri": 3, "pv_dosi_frazioni": [1 / 4096, 1 / 1024, 1 / 256, 1 / 64, 1 / 16, 1 / 4, 1.0],
    "perdita": "cakld", "gamma": 0.552, "ema_theta": 0.9, "K0": 524288.0, "ripresa_ogni": 1,
    "heldout_ogni": 1, "heldout_righe_rapide": 16, "heldout_pieno_ogni": 2, "attribuzione": True,
    "attribuzione_righe": 2, "istantanea_ogni_passo": True, "istantanee_tieni": 2, "istantanee_libero_min_gib": 20.0,
    "rotazione": True, "consuma_blocchi": False, "testa_kl": "bf16x3", "logit_maestro": "bf16", "epoche": 1,
    "righe_per_passo": 32, "lota_rango": 0,
}
# Campi che EB4 PUO' avere diversi da config.json di EB3 (tutti gli altri devono essere uguali):
#   lora_init ga -> zero, lr_lora (4e-5 della v3; 1e-5 nella riga di riavvio del 28/9), lora_ripristina (solo della riga di
#   riavvio di EB3), righe_base_da (nuovo), istantanee_dir (cartella di EB4), e due campi INERTI per costruzione:
#   lora_ga_gamma (senza LoRA-GA non entra in nessun calcolo), sophia_gamma (con AdamW non entra in nessun calcolo).
DIFF_AMMESSI = {"lora_init", "lr_lora", "lora_ripristina", "righe_base_da", "istantanee_dir", "lora_ga_gamma",
                "sophia_gamma"}
# Verifica avversaria 28/9: il cancello strumenti del §41 e' `RIPRENDI=1 lancia_eb4.sh --lora-ripristina ferma` (a EB4
# ferma al confine). Con la ricetta rigida (lora_ripristina "") quel rilancio era ROSSO e il cancello non si poteva
# azionare. Ammesso SOLO a una ripresa (in una corsa nuova --lora-ripristina non ha senso: resta "").
RIPRISTINI_ALLA_RIPRESA = ("ferma",)
muto = lambda *a, **k: None  # noqa: E731


def _json(x):
    return json.loads(json.dumps(x, default=str))


# ─────────────────────────────── argomenti ───────────────────────────────
class _Preso(Exception):
    pass


def cattura_argomenti(argv: list) -> argparse.Namespace:
    """Il Namespace che il parser VERO di `e2e_allena.main` produce da `argv`, senza eseguire altro: `parse_args` del
    parser di primo livello e' avvolto e solleva `_Preso` appena ha finito (i sotto-parser non passano da parse_args).
    Argomenti rifiutati: SystemExit di argparse -> ValueError col messaggio."""
    import contextlib
    import io
    import e2e_allena as A
    orig = argparse.ArgumentParser.parse_args
    preso: dict = {}

    def pa(self, args=None, namespace=None):
        preso["ns"] = orig(self, args, namespace)
        raise _Preso

    argparse.ArgumentParser.parse_args = pa
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            A.main(list(argv))
    except _Preso:
        pass
    except SystemExit as e:
        raise ValueError(f"parser di e2e_allena: argomenti rifiutati (exit {e.code}): {err.getvalue().strip()[-600:]}")
    finally:
        argparse.ArgumentParser.parse_args = orig
    if "ns" not in preso:
        raise RuntimeError("e2e_allena.main non ha chiamato parse_args")
    return preso["ns"]


def controlla_argomenti(argv: list, corsa: str, gguf: str, config_eb3: Path, ripresa: bool) -> tuple:
    import e2e_allena as A
    from dataclasses import asdict
    ns = cattura_argomenti(argv)
    ca = A._config_da_arg(ns)
    cfg = _json(asdict(ca))
    c: dict = {}
    c["sottocomando_allena"] = ns.cmd == "allena"
    c["corsa"] = ns.corsa == corsa
    c["gguf_dello_studente"] = ns.gguf is not None and Path(ns.gguf).resolve() == Path(gguf).resolve()
    c["corsa_nuova_o_ripresa"] = bool(ns.riprendi) == bool(ripresa)
    c["niente_lora_ga"] = not ns.lora_reinit_ga and cfg["lora_init"] == "zero"
    sbagliati = {k: (cfg.get(k), v) for k, v in RICETTA.items() if cfg.get(k) != v
                 and not (ripresa and k == "lora_ripristina" and cfg.get(k) in RIPRISTINI_ALLA_RIPRESA)}
    c["ricetta_del_41"] = not sbagliati
    c["righe_base_da_positivo"] = int(cfg["righe_base_da"]) > 0
    diff, eb3 = {}, {}
    if config_eb3 is not None and config_eb3.exists():
        eb3 = json.loads(config_eb3.read_text())
        dflt = _json(asdict(A.ConfigAllena()))
        prima = eb3["config"]
        diff = {k: (prima.get(k, dflt.get(k)), cfg.get(k)) for k in set(prima) | set(cfg)
                if prima.get(k, dflt.get(k)) != cfg.get(k)}
        c["diff_da_EB3_solo_ammessi"] = set(diff) <= DIFF_AMMESSI
        c["stesso_teacher_di_EB3"] = (ns.teacher == eb3.get("teacher") and ns.teacher_mc == eb3.get("teacher_mc")
                                      and ns.teacher_mc_heldout == eb3.get("teacher_mc_heldout"))
        c["stessa_ricetta_numerica_di_EB3"] = (ns.ricetta == eb3.get("ricetta"))
    else:
        c["config_eb3_presente"] = False
    return ca, ns, c, {"sbagliati": sbagliati, "diff_da_EB3": diff, "righe_base_da": cfg["righe_base_da"]}


# ─────────────────────────────── righe ───────────────────────────────
def sorgente(teacher: str, teacher_mc, teacher_mc_held):
    import e2e_allena as A
    src = A.SorgenteCache(teacher)
    if teacher_mc or teacher_mc_held:
        src = A.apri_sorgente_mc(src, Path(teacher_mc) if teacher_mc else None,
                                 Path(teacher_mc_held) if teacher_mc_held else None, muto)
    return src


def righe_auto(teacher: str) -> int:
    """Prima riga del primo blocco train interamente sigillato (i consumati e i non ancora fatti non contano)."""
    import e2e_allena as A
    src = A.SorgenteCache(teacher)
    for _, da, a in src.blocchi("train"):
        if all(src.disponibile("train", i) for i in range(da, a)):
            return da
    raise RuntimeError(f"{teacher}: nessun blocco train sigillato")


def controlla_righe(ca, src, passi_min: int) -> tuple:
    """Il piano che la corsa NUOVA costruira' al passo 0 (stesso `_segmenti_mc`, stesso `PianoRighe`) e quanti passi il
    teacher copre ORA dall'inizio, senza buchi."""
    import e2e_allena as A
    con_mc = isinstance(src, A.SorgenteConMC)
    train = src.righe_previste("train")
    stato = {"passo": 0}
    seg = A._segmenti_mc(stato, ca, src if con_mc else None, train, muto)
    piano = A.PianoRighe(train, ca, src.righe_mc() if con_mc else [], seg)
    tot = piano.totale() if not piano.classico else (len(train) // ca.righe_per_passo) * ca.epoche
    coperti, mancante = 0, None
    for t in range(tot):
        r, _ = piano.righe(t)
        buco = [i for i in r if not src.disponibile("train", i)]
        if buco:
            mancante = {"passo": t, "riga": buco[0]}
            break
        coperti += 1
    held = src.righe_previste("heldout")
    held_mc = src.righe_previste(A.LATO_HELD_MC) if con_mc else []
    prime = piano.righe(0)[0] if tot else []
    c = {"primo_segmento_al_passo_0": bool(seg) and int(seg[0]["passo"]) == 0
         and int(seg[0]["cursore_base"]) == int(ca.righe_base_da),
         "passo_0_senza_righe_prima_di_N": all(i >= ca.righe_base_da for i in prime if i < A.OFFSET_MC),
         "passi_coperti_min": coperti >= passi_min,
         "heldout_tutto_sigillato": bool(held) and all(src.disponibile("heldout", i) for i in held),
         "heldout_mc_tutto_sigillato": (not ca.righe_mc) or (bool(held_mc) and all(
             src.disponibile(A.LATO_HELD_MC, i) for i in held_mc)),
         "righe_mc_tutte_sigillate": (not ca.righe_mc) or all(src.disponibile("train", i) for i in src.righe_mc())}
    info = {"righe_train": len(train), "segmento": seg[0] if seg else None, "passi_totali": tot,
            "passi_coperti_ora": coperti, "prima_mancante": mancante, "righe_passo_0": prime[:8] + ["..."],
            "righe_mc": len(src.righe_mc()) if con_mc else 0, "heldout": len(held), "heldout_mc": len(held_mc)}
    return c, info


# ─────────────────────────────── indipendenza ───────────────────────────────
def indipendenza(teachers: list, gguf: str) -> tuple:
    """I logit del maestro dipendono SOLO dai testi (gli id del corpus), non dallo studente — dimostrato sul codice che
    la corsa esegue e sulla cache vera:
      1. la cache si legge per (lato, riga) e basta: `CacheMaestro.h(lato, i)`, `SorgenteCache.h(lato, i)`;
      2. i logit del maestro sono `_logit_maestro(hT, WT, dtype)`: h del maestro x testa del maestro, nessuno studente;
      3. la testa del maestro e' output.weight del bf16 ORIGINALE (`testa_maestro` legge `e2e_teacher.TESTA_BF16`);
      4. h e' calcolato da `e2e_dump_logit -m <modello del PIANO>` sugli id del corpus (`esegui_blocco`), e OGNI sigillo
         della cache porta lo sha256 del modello del maestro (il Q8), uguale nei tre teacher e diverso dallo studente;
      5. la firma della sorgente (che la ripresa confronta, `stato["sorgente"]`) non contiene lo studente, che ha la sua
         (`_firma_studente`, `stato["studente"]`).
    Quindi il teacher di EB3 vale per EB4 cosi' com'e': cambia solo chi insegue quei logit."""
    import e2e_allena as A
    import e2e_teacher as T
    par = lambda f: list(inspect.signature(f).parameters)  # noqa: E731
    c = {"cache_h_per_lato_riga": par(T.CacheMaestro.h) == ["self", "lato", "i"],
         "sorgente_h_per_lato_riga": par(A.SorgenteCache.h) == ["self", "lato", "i"],
         "logit_maestro_senza_studente": par(A._logit_maestro) == ["hT", "WT", "dtype"],
         "testa_maestro_dal_bf16": "T.TESTA_BF16" in inspect.getsource(A.testa_maestro)
         and Path(T.TESTA_BF16).resolve() != Path(gguf).resolve(),
         "dump_col_modello_del_piano": '"-m", piano["modello"]["percorso"]' in inspect.getsource(T.esegui_blocco)}
    info: dict = {}
    sha_maestri = set()
    for d in teachers:
        m = T.CacheMaestro(d)
        mod = m.piano["modello"]
        sigilli = set()
        for b in m.piano["blocchi"]:
            p = Path(d) / b["nome"] / "blocco.json"
            if p.exists():
                sigilli.add(json.loads(p.read_text())["modello_sha256"])
        n = Path(d).name
        c[f"{n}_sigilli_del_maestro"] = sigilli == {mod["sha256"]}
        c[f"{n}_maestro_non_e_lo_studente"] = (Path(mod["percorso"]).resolve() != Path(gguf).resolve()
                                               and mod["byte"] != Path(gguf).stat().st_size)
        sha_maestri.add(mod["sha256"])
        info[n] = {"modello": mod["percorso"], "sha256": mod["sha256"][:16], "sigilli": len(sigilli)}
    c["un_solo_maestro"] = len(sha_maestri) == 1
    f = A.SorgenteCache(teachers[0]).firma()
    c["firma_sorgente_senza_studente"] = (set(f) == {"teacher", "piano_sha256", "modo", "modello", "corpus"}
                                          and str(Path(gguf).resolve()) not in json.dumps(f, default=str)
                                          and f["modello"]["sha256"] in sha_maestri)
    c["firma_studente_a_parte"] = "st.gguf" in inspect.getsource(A._firma_studente)
    return c, info


# ─────────────────────────────── struttura del GGUF ───────────────────────────────
def struttura(gguf: str, riferimento: str) -> tuple:
    sys.path.insert(0, GGUF_PY)
    from gguf import GGUFReader

    def tensori(p):
        r = GGUFReader(p)
        return {t.name: (t.tensor_type.name, [int(x) for x in t.shape]) for t in r.tensors}
    a, b = tensori(gguf), tensori(riferimento)
    diversi = sorted(n for n in set(a) | set(b) if a.get(n) != b.get(n))
    c = {"stessi_tensori_del_riferimento": not diversi and len(a) > 0}
    return c, {"tensori": len(a), "riferimento": riferimento, "diversi": diversi[:10],
               "byte": Path(gguf).stat().st_size}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("righe-auto")
    p.add_argument("--teacher", required=True)
    p = sub.add_parser("controlla")
    p.add_argument("--corsa", default="EB4")
    p.add_argument("--gguf", required=True)
    p.add_argument("--riferimento", required=True, help="GGUF con la struttura attesa (base della riforgia)")
    p.add_argument("--config-eb3", default="@@ROOT:depot@@/e2e/allena/EB3/config.json")
    p.add_argument("--passi-min", type=int, default=24)
    p.add_argument("--ripresa", action="store_true", help="rilancio di una EB4 esistente (--riprendi atteso)")
    p.add_argument("--uscita", default=None)
    p.add_argument("argv", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    if a.cmd == "righe-auto":
        print(righe_auto(a.teacher))
        return 0
    argv_allena = list(a.argv[1:]) if a.argv[:1] == ["--"] else list(a.argv)
    rapporto: dict = {"controlli": {}, "info": {}}
    try:
        ca, ns, c, info = controlla_argomenti(argv_allena, a.corsa, a.gguf, Path(a.config_eb3) if a.config_eb3 else None,
                                              a.ripresa)
        rapporto["controlli"].update({f"argomenti.{k}": v for k, v in c.items()})
        rapporto["info"]["argomenti"] = info
        src = sorgente(ns.teacher, ns.teacher_mc, ns.teacher_mc_heldout)
        c, info = controlla_righe(ca, src, a.passi_min)
        rapporto["controlli"].update({f"righe.{k}": v for k, v in c.items()})
        rapporto["info"]["righe"] = info
        teachers = [x for x in (ns.teacher, ns.teacher_mc, ns.teacher_mc_heldout) if x]
        c, info = indipendenza(teachers, a.gguf)
        rapporto["controlli"].update({f"indipendenza.{k}": v for k, v in c.items()})
        rapporto["info"]["indipendenza"] = info
        c, info = struttura(a.gguf, a.riferimento)
        rapporto["controlli"].update({f"struttura.{k}": v for k, v in c.items()})
        rapporto["info"]["struttura"] = info
    except Exception as e:                                    # un controllo che non gira e' un controllo ROSSO
        rapporto["controlli"]["eccezione"] = False
        rapporto["errore"] = f"{type(e).__name__}: {e}"
    rapporto["tutto_verde"] = bool(rapporto["controlli"]) and all(rapporto["controlli"].values())
    rapporto["rossi"] = sorted(k for k, v in rapporto["controlli"].items() if not v)
    testo = json.dumps(rapporto, indent=1, default=str)
    if a.uscita:
        Path(a.uscita).write_text(testo)
    print(testo)
    return 0 if rapporto["tutto_verde"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

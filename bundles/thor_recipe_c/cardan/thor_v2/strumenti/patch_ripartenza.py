"""Patch RIPARTENZA (24/9 sera, finestra scorrevole a passo 1) per valida_finestra_8_12.py.

Aggiunge a `ottimizza_finestra_JOINT` tre argomenti PER CHIAMATA (default = oggi,
nessun comportamento cambia se non li si passa):

  ripartenza_per_strato      {L: voce} — lo strato L non parte da teacher + LS ma da
                             `voce`: uno StatoStrato (parametri z/log_s/delta_raw della
                             finestra precedente, copiati al bit) oppure un dict di codici
                             come risultato[L] / ripartenza_codici.carica_codici_corsa
                             (ricostruzione esatta, z nella posizione di init_ls).
  consegna_stato_per_strato  {L: None} — la funzione riempie i valori con lo StatoStrato
                             di fine finestra di L (riferimenti, zero copie), da passare
                             come ripartenza alla finestra dopo.
  adam_ripartenza            "riparti" (default, come il codice pubblico di SliderQuant:
                             AdamW nuovo a ogni finestra) oppure "conserva" (i momenti di
                             Adam dello strato passano alla finestra dopo).

Per ogni strato ripartito una riga di log: codici identici ai forniti (atteso 100%) e
scarto massimo dei pesi (atteso 0), piu' la lettura del nucleo compilato. Rifiuti
rumorosi (RuntimeError/ValueError) se non combaciano. La logica e' tutta in
ripartenza_codici.py, importato SOLO quando uno dei nuovi argomenti e' usato.

Uso:
    python3 strumenti/patch_ripartenza.py                 # applica a ../valida_finestra_8_12.py
    python3 strumenti/patch_ripartenza.py --file COPIA.py # applica a una copia
    python3 strumenti/patch_ripartenza.py --prova         # controlla soltanto, non scrive
Fallisce senza scrivere nulla se un aggancio non compare esattamente una volta, se la
patch risulta gia' applicata, o se il risultato non passa ast.parse.
"""
import argparse
import ast
from pathlib import Path

PREDEFINITO = Path(__file__).resolve().parent.parent / "valida_finestra_8_12.py"

SOSTITUZIONI = [
    # -- 1. firma
    ('''                              n_batch_addestramento: "int | None" = None,
                              uscita_teacher_fine: "dict | None" = None) -> dict:''',
     '''                              n_batch_addestramento: "int | None" = None,
                              uscita_teacher_fine: "dict | None" = None,
                              ripartenza_per_strato: "dict | None" = None,
                              consegna_stato_per_strato: "dict | None" = None,
                              adam_ripartenza: str = "riparti") -> dict:'''),
    # -- 2. documentazione dei nuovi argomenti (in coda al docstring)
    ('''    ottimizza_rete_multi_finestra`), per ogni finestra tranne la prima."""''',
     '''    ottimizza_rete_multi_finestra`), per ogni finestra tranne la prima.

    ⭐ 24/9 sera — FINESTRA SCORREVOLE A PASSO 1 ([0,1],[1,2],...: decisione di
    Angelo; logica in ripartenza_codici.py, patch strumenti/patch_ripartenza.py).
    Tre argomenti per chiamata, default = comportamento INVARIATO:
      ripartenza_per_strato={L: voce}: lo strato L parte da `voce` invece che da
        teacher + LS. voce = StatoStrato (parametri della finestra precedente,
        copiati al bit: la via per lo strato che scorre) oppure dict di codici
        come risultato[L] o `carica_codici_corsa` (ricostruzione esatta di D*Q,
        z dove lo metterebbe init_ls). Log per strato: codici identici ai forniti
        (100%) e scarto massimo dei pesi (0); altrimenti RuntimeError.
      consegna_stato_per_strato={L: None}: a fine finestra i valori diventano gli
        StatoStrato degli strati chiesti (riferimenti, zero copie).
      adam_ripartenza: "riparti" (AdamW nuovo, come SliderQuant) o "conserva"
        (i momenti dello strato consegnato/ripartito passano di finestra)."""'''),
    # -- 3. validazione degli argomenti (niente strati ignorati in silenzio)
    ('''    assert ids_batch.dim() == 3, "ids_batch: (N_BATCH, BATCH_SEQ, L_SEQ), come da carica_poche_batch"
''',
     '''    assert ids_batch.dim() == 3, "ids_batch: (N_BATCH, BATCH_SEQ, L_SEQ), come da carica_poche_batch"
    if adam_ripartenza not in ("riparti", "conserva"):
        raise ValueError(f"adam_ripartenza={adam_ripartenza!r}: attesi 'riparti' o 'conserva'")
    for _nome_arg, _arg in (("ripartenza_per_strato", ripartenza_per_strato),
                            ("consegna_stato_per_strato", consegna_stato_per_strato)):
        if _arg is not None and not set(_arg) <= set(layers):
            raise ValueError(f"{_nome_arg}: strati {sorted(set(_arg) - set(layers))} fuori dalla finestra {layers}")
'''),
    # -- 4. init: ripartenza al posto di teacher + LS per gli strati indicati
    ('''            if init_ls:
                d_gu = reinit_ls_(ternari_gu[L], _ruota_se_possibile(W_gu.detach().float().to(device), S))
''',
     '''            if ripartenza_per_strato is not None and L in ripartenza_per_strato:
                # ⭐ 24/9 sera: finestra scorrevole a passo 1 — lo strato riparte dai parametri
                # della finestra precedente o dai codici di una corsa, non dal teacher
                from ripartenza_codici import applica_ripartenza_
                applica_ripartenza_(
                    L, ripartenza_per_strato[L], ternari_gu[L], ternari_dw[L],
                    W_gu_rif=lambda _W=W_gu: _ruota_se_possibile(_W.detach().float().to(device), S),
                    W_dw_rif=lambda _W=W_dw: _W.detach().float().to(device),
                    margine_z=float(os.getenv("THOR_LR_Z", str(lr))),
                    conserva_adam=(adam_ripartenza == "conserva"), log=log)
            elif init_ls:
                d_gu = reinit_ls_(ternari_gu[L], _ruota_se_possibile(W_gu.detach().float().to(device), S))
'''),
    # -- 5. momenti di Adam conservati, nel nuovo ottimizzatore
    ('''        assert all(g["weight_decay"] == wd for g in opt.param_groups)
''',
     '''        assert all(g["weight_decay"] == wd for g in opt.param_groups)
        if ripartenza_per_strato is not None and adam_ripartenza == "conserva":
            from ripartenza_codici import StatoStrato, inietta_adam_
            for L in chunk:
                _voce = ripartenza_per_strato.get(L)
                if isinstance(_voce, StatoStrato):
                    _n = inietta_adam_(opt, ternari_gu[L], ternari_dw[L], _voce)
                    log(f"  strato {L}: stato di Adam CONSERVATO dalla finestra precedente ({_n} tensori)")
'''),
    # -- 6. consegna dello stato di fine finestra
    ('''            if ternari_dw[L] is not None:
                risultato[L]["ffn_down_exps"] = _estrai_famiglia(ternari_dw[L])
''',
     '''            if ternari_dw[L] is not None:
                risultato[L]["ffn_down_exps"] = _estrai_famiglia(ternari_dw[L])
            if consegna_stato_per_strato is not None and L in consegna_stato_per_strato:
                from ripartenza_codici import estrai_stato_strato
                consegna_stato_per_strato[L] = estrai_stato_strato(
                    ternari_gu[L], ternari_dw[L], opt=(opt if adam_ripartenza == "conserva" else None),
                    codici_rif=risultato[L], origine=f"finestra {layers}")
'''),
]


def applica(testo: str) -> str:
    if "ripartenza_per_strato" in testo:
        raise SystemExit("patch gia' applicata ('ripartenza_per_strato' presente): niente da fare")
    for vecchio, nuovo in SOSTITUZIONI:
        n = testo.count(vecchio)
        if n != 1:
            raise SystemExit(f"aggancio trovato {n} volte (atteso 1): {vecchio[:90]!r}")
        testo = testo.replace(vecchio, nuovo)
    ast.parse(testo)
    return testo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--file", type=Path, default=PREDEFINITO)
    ap.add_argument("--prova", action="store_true", help="controlla agganci e sintassi, non scrive")
    args = ap.parse_args()
    nuovo = applica(args.file.read_text())
    if args.prova:
        print(f"prova OK: {len(SOSTITUZIONI)} agganci unici, risultato sintatticamente valido — {args.file} NON toccato")
        return
    args.file.write_text(nuovo)
    print(f"patch applicata: {len(SOSTITUZIONI)} sostituzioni in {args.file}")


if __name__ == "__main__":
    main()

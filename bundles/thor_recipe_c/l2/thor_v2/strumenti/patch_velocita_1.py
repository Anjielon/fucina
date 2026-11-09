"""Patch velocita' 1 (24/9 19:5x, mandato forgia < 2 ore) per valida_finestra_8_12.py.

Due sprechi tolti, nessuna matematica cambiata:
1. la ricostruzione dei pesi DURI (nucleo ternario + de-rotazione FWHT su ~0,8 G
   parametri per strato) si faceva UNA VOLTA PER BATCH in _congela_routing_chunk
   e in _loss_heldout: 36 volte a ogni ricongelo, sempre identica (dipende solo
   dai codici, non dai dati). Ora una volta per chiamata, riusata su tutte le batch.
2. il forward diagnostico F̂(h) e il controllo algebrico (.item() = sincronizzazione)
   in ogni passo di addestramento diventano facoltativi: THOR_DIAGNOSTICA_PASSO=0
   li spegne (stesso gradiente). Default 1 = comportamento invariato.
Applicata da riga di comando; fallisce senza scrivere nulla se un aggancio non combacia.
"""
from pathlib import Path

P = Path(__file__).resolve().parent.parent / "valida_finestra_8_12.py"
s = P.read_text()
sostituzioni = [
    # -- costante d'ambiente
    ('''CRONO_OGNI = int(os.getenv("THOR_CRONO_OGNI", "0"))''',
     '''DIAGNOSTICA_PASSO = os.getenv("THOR_DIAGNOSTICA_PASSO", "1") == "1"   # 0 = niente F̂(h) diagnostico nei passi
CRONO_OGNI = int(os.getenv("THOR_CRONO_OGNI", "0"))'''),
    # -- _congela_routing_chunk: firma
    ('''def _congela_routing_chunk(chunk: list, strati_correnti: dict, h0_quant: torch.Tensor, forward_grezzo,
                           ternari_gu: "dict | None" = None, ternari_dw: "dict | None" = None,
                           S: "torch.Tensor | None" = None) -> dict:''',
     '''def _ricostruisci_pesi_duri(chunk: list, strati_correnti: dict, ternari_gu: dict,
                            ternari_dw: "dict | None", S: "torch.Tensor | None") -> dict:
    """{L: (gate_up reale, down reale | None)} coi codici DURI correnti, nel dtype
    del modulo. Dipende solo dai codici, mai dai dati: si calcola UNA volta e si
    riusa per tutte le batch (24/9 19:5x — prima si rifaceva per ognuna delle 36
    batch a ogni ricongelo e in ogni valutazione held-out)."""
    pronti: dict = {}
    with torch.no_grad():
        for L in chunk:
            _dt = strati_correnti[L].mlp.experts.gate_up_proj.dtype
            gu = _deruota_se_possibile(ternari_gu[L].forward_duro(NUCLEO_BETA_MAX), S).to(_dt)
            dw = (None if ternari_dw is None or ternari_dw[L] is None
                  else ternari_dw[L].forward_duro(NUCLEO_BETA_MAX).to(_dt))
            pronti[L] = (gu, dw)
    return pronti


def _congela_routing_chunk(chunk: list, strati_correnti: dict, h0_quant: torch.Tensor, forward_grezzo,
                           ternari_gu: "dict | None" = None, ternari_dw: "dict | None" = None,
                           S: "torch.Tensor | None" = None, pesi_pronti: "dict | None" = None) -> dict:'''),
    # -- _congela_routing_chunk: corpo
    ('''                _dt = strato.mlp.experts.gate_up_proj.dtype
                gu = _deruota_se_possibile(ternari_gu[L].forward_duro(NUCLEO_BETA_MAX), S).to(_dt)
                dw = (None if ternari_dw is None or ternari_dw[L] is None
                      else ternari_dw[L].forward_duro(NUCLEO_BETA_MAX).to(_dt))
                sost_dw''',
     '''                if pesi_pronti is not None:
                    gu, dw = pesi_pronti[L]
                else:
                    _dt = strato.mlp.experts.gate_up_proj.dtype
                    gu = _deruota_se_possibile(ternari_gu[L].forward_duro(NUCLEO_BETA_MAX), S).to(_dt)
                    dw = (None if ternari_dw is None or ternari_dw[L] is None
                          else ternari_dw[L].forward_duro(NUCLEO_BETA_MAX).to(_dt))
                sost_dw'''),
    # -- _stratofn_quant_reale: firma + pesi pronti
    ('''                           selezionati: torch.Tensor, pesi: torch.Tensor,
                           beta: float, duro: bool, forward_grezzo) -> "StratoFn":''',
     '''                           selezionati: torch.Tensor, pesi: torch.Tensor,
                           beta: float, duro: bool, forward_grezzo,
                           pesi_pronti: "tuple | None" = None) -> "StratoFn":'''),
    ('''    _dtype = strato.mlp.experts.gate_up_proj.dtype
    gu_rot = ternari_gu.forward_duro(beta) if duro else ternari_gu.forward(beta)
    gu_reale = _deruota_se_possibile(gu_rot, S).to(_dtype)
    # audit B5: ternari_dw None = down v20 congelato (nessuna sostituzione)
    dw_reale = (None if ternari_dw is None else
                (ternari_dw.forward_duro(beta) if duro else ternari_dw.forward(beta)).to(_dtype))  # down: NIENTE rotazione''',
     '''    _dtype = strato.mlp.experts.gate_up_proj.dtype
    if pesi_pronti is not None:
        # solo valutazione (no_grad, codici duri): pesi gia' ricostruiti una volta
        # per tutte le batch da _ricostruisci_pesi_duri — stesso valore di
        # forward_duro(qualunque beta): beta tocca solo il gradiente surrogato
        gu_reale, dw_reale = pesi_pronti
    else:
        gu_rot = ternari_gu.forward_duro(beta) if duro else ternari_gu.forward(beta)
        gu_reale = _deruota_se_possibile(gu_rot, S).to(_dtype)
        # audit B5: ternari_dw None = down v20 congelato (nessuna sostituzione)
        dw_reale = (None if ternari_dw is None else
                    (ternari_dw.forward_duro(beta) if duro else ternari_dw.forward(beta)).to(_dtype))  # down: NIENTE rotazione'''),
    # -- init: congelamento iniziale
    ('''        for b in range(n_batch_loc):
            congelati_per_b[b] = _congela_routing_chunk(chunk, strati_correnti, hq_per_b[b], _forward_grezzo,
                                                        ternari_gu=ternari_gu, ternari_dw=ternari_dw, S=S)
''',
     '''        _pronti = _ricostruisci_pesi_duri(chunk, strati_correnti, ternari_gu, ternari_dw, S)
        for b in range(n_batch_loc):
            congelati_per_b[b] = _congela_routing_chunk(chunk, strati_correnti, hq_per_b[b], _forward_grezzo,
                                                        ternari_gu=ternari_gu, ternari_dw=ternari_dw, S=S,
                                                        pesi_pronti=_pronti)
        del _pronti
'''),
    # -- valutazione held-out
    ('''            vals = []
            with torch.no_grad():
                for bb in batch_ho:
                    if quali == "v20":
                        fq = [_stratofn_reale(strati_correnti[L], _forward_grezzo) for L in chunk]
                    else:
                        fq = [_stratofn_quant_reale(strati_correnti[L], ternari_gu[L], ternari_dw[L], S,
                                                    *congelati[bb][L], NUCLEO_BETA_MAX, True, _forward_grezzo)
                              for L in chunk]''',
     '''            vals = []
            pronti = (None if quali == "v20" else
                      _ricostruisci_pesi_duri(chunk, strati_correnti, ternari_gu, ternari_dw, S))
            with torch.no_grad():
                for bb in batch_ho:
                    if quali == "v20":
                        fq = [_stratofn_reale(strati_correnti[L], _forward_grezzo) for L in chunk]
                    else:
                        fq = [_stratofn_quant_reale(strati_correnti[L], ternari_gu[L], ternari_dw[L], S,
                                                    *congelati[bb][L], NUCLEO_BETA_MAX, True, _forward_grezzo,
                                                    pesi_pronti=pronti[L])
                              for L in chunk]'''),
    # -- ricongelo periodico
    ('''                vecchi = congelati_per_b
                congelati_per_b = {bb: _congela_routing_chunk(chunk, strati_correnti, hq_per_b[bb], _forward_grezzo,
                                                              ternari_gu=ternari_gu, ternari_dw=ternari_dw, S=S)
                                   for bb in range(n_batch_loc)}''',
     '''                vecchi = congelati_per_b
                _pronti = _ricostruisci_pesi_duri(chunk, strati_correnti, ternari_gu, ternari_dw, S)
                congelati_per_b = {bb: _congela_routing_chunk(chunk, strati_correnti, hq_per_b[bb], _forward_grezzo,
                                                              ternari_gu=ternari_gu, ternari_dw=ternari_dw, S=S,
                                                              pesi_pronti=_pronti)
                                   for bb in range(n_batch_loc)}
                del _pronti'''),
    # -- passo: diagnostica facoltativa
    ('''            r = passo_ottimizzazione(strati_teacher_fn, strati_quant_fn, h0_per_b[b], opt, e0_per_b[b],
                                      h_l_teacher_cache=h_teacher_cache_per_b[b])''',
     '''            r = passo_ottimizzazione(strati_teacher_fn, strati_quant_fn, h0_per_b[b], opt, e0_per_b[b],
                                      h_l_teacher_cache=h_teacher_cache_per_b[b], diagnostica=DIAGNOSTICA_PASSO)'''),
    # -- congelamento finale per l'held-out
    ('''            congelati_finali = {bb: _congela_routing_chunk(chunk, strati_correnti, hq_per_b[bb], _forward_grezzo,
                                                          ternari_gu=ternari_gu, ternari_dw=ternari_dw, S=S)
                                for bb in batch_ho}''',
     '''            _pronti = _ricostruisci_pesi_duri(chunk, strati_correnti, ternari_gu, ternari_dw, S)
            congelati_finali = {bb: _congela_routing_chunk(chunk, strati_correnti, hq_per_b[bb], _forward_grezzo,
                                                          ternari_gu=ternari_gu, ternari_dw=ternari_dw, S=S,
                                                          pesi_pronti=_pronti)
                                for bb in batch_ho}
            del _pronti'''),
    # -- profilatore facoltativo: THOR_PROFILO=1 -> passi 5..9 sotto torch.profiler, tabella nel log
    ('''            r = passo_ottimizzazione(strati_teacher_fn, strati_quant_fn, h0_per_b[b], opt, e0_per_b[b],
                                      h_l_teacher_cache=h_teacher_cache_per_b[b], diagnostica=DIAGNOSTICA_PASSO)''',
     '''            if PROFILO and t == 5:
                from torch.profiler import profile as _profile, ProfilerActivity as _PA
                _prof = _profile(activities=[_PA.CPU, _PA.CUDA], record_shapes=False)
                _prof.__enter__()
            r = passo_ottimizzazione(strati_teacher_fn, strati_quant_fn, h0_per_b[b], opt, e0_per_b[b],
                                      h_l_teacher_cache=h_teacher_cache_per_b[b], diagnostica=DIAGNOSTICA_PASSO)
            if PROFILO and t == 9:
                torch.cuda.synchronize()
                _prof.__exit__(None, None, None)
                log("PROFILO passi 5-9 (per tempo GPU):\\n" + _prof.key_averages().table(
                    sort_by="cuda_time_total", row_limit=40))'''),
    ('''DIAGNOSTICA_PASSO = os.getenv("THOR_DIAGNOSTICA_PASSO", "1") == "1"''',
     '''PROFILO = os.getenv("THOR_PROFILO", "0") == "1"   # radiografia dei passi 5-9 (torch.profiler)
DIAGNOSTICA_PASSO = os.getenv("THOR_DIAGNOSTICA_PASSO", "1") == "1"'''),

]
for vecchio, nuovo in sostituzioni:
    n = s.count(vecchio)
    if n != 1:
        raise SystemExit(f"aggancio trovato {n} volte (atteso 1): {vecchio[:90]!r}")
    s = s.replace(vecchio, nuovo)
P.write_text(s)
print(f"patch applicata: {len(sostituzioni)} sostituzioni in {P.name}")

"""Patch TEMPO «prepara asincrona» (leva C del piano) per forgia_scorrevole.py — 25/9 notte.

Oggi `Ripresa.prepara` scrive su disco lo stato latente degli strati portati (torch.save di
~3,04 GiB, z/log_s/delta_raw di gate_up e down) in modo SINCRONO, fra la fine di una finestra
e l'inizio della dopo. Misurato in conf6b-S1 dalle date di nascita/modifica di portato_NN.pt:
2,08 · 2,62 · 2,35 s a finestra (rapporto costo_fisso.md §2).

Qui la parte sincrona diventa una FOTOGRAFIA: copia sullo stesso device dei tre parametri per
famiglia (stessa struttura di `_stato_ternario`, stesso state_dict, _metadata compreso), e la
scrittura su disco passa nel thread di scrittura, in testa alla transizione: prima
`.tmp_finestra_KK/portato_*.pt`, poi (come oggi) semi + manifest + rename, poi codici, poi
cancellazione del punto K. Il protocollo di ripresa resta identico, riga per riga.

Perche' una COPIA e non i riferimenti: il thread scrive mentre la finestra dopo lavora gia'.
Con il raccordo (produzione) i tensori portati vengono copiati nei moduli nuovi e restano
fermi, ma chi passa gli oggetti tali e quali (SostitutoA dei collaudi) li allena IN PLACE: un
riferimento finirebbe su disco a meta' aggiornamento. La copia rende il punto di ripresa
indipendente da cosa fa la finestra dopo (collaudo `gara_prepara`: con i riferimenti fallisce).

NUMERI: IDENTICI AL BIT — cambia solo QUANDO si scrive. MEMORIA: +3,04 GiB di VRAM dalla fine
della finestra finche' il thread non ha scritto il file dello strato (poi la copia si rilascia,
prima di completa); con THOR_SCORREVOLE_ADAM=conserva (25/9, altro lavoro) la copia comprende
anche i momenti di Adam che la finestra dopo aggiorna sul posto: +2 x 3,04 GiB in piu' (STIMA
dalle forme: exp_avg ed exp_avg_sq come i parametri), picco di VRAM NON MISURATO. Interruttore THOR_SCORREVOLE_PREPARA_ASINCRONA=0 (default 1; fuori
ricetta: e' meccanica, non cambia i numeri). Con THOR_SCORREVOLE_SCRITTURA_ASINCRONA=0 resta
il percorso di oggi (prepara sincrona).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_tempo_comune import esegui_patch  # noqa: E402

MARCA = "patch_tempo_prepara_asincrona"

PIANO = {"forgia_scorrevole.py": [
    # -- 1. interruttore, fuori ricetta, e la fotografia
    ('''class InterruzioneSimulata(RuntimeError):''',
     '''# ⭐ 25/9 (patch_tempo_prepara_asincrona): stato portato scritto nel thread di scrittura, da una
# FOTOGRAFIA sincrona (copia su device); vedi Ripresa.fotografa / prepara_da_foto
PREPARA_ASINCRONA = os.getenv("THOR_SCORREVOLE_PREPARA_ASINCRONA", "1") == "1"
_NON_RICETTA |= {"THOR_SCORREVOLE_PREPARA_ASINCRONA"}


def _copia_profonda(v):
    """Copia di una struttura (dict/list/tuple annidati) con OGNI tensore copiato (stesso device,
    dtype, forma e passi). I dizionari sono NUOVI (copy.copy: stesso tipo, e per l'OrderedDict di
    state_dict() anche _metadata): i dizionari dell'oggetto non si toccano."""
    import copy as _copy
    if torch.is_tensor(v):
        return v.detach().clone()
    if isinstance(v, dict):
        nuovo = _copy.copy(v)
        for chiave in list(nuovo):
            nuovo[chiave] = _copia_profonda(nuovo[chiave])
        return nuovo
    if isinstance(v, (list, tuple)):
        return type(v)(_copia_profonda(x) for x in v)
    return v


def _stato_ternario_copia(obj) -> dict:
    """Come _stato_ternario, ma OGNI tensore e' una COPIA: stato latente e, se c'e'
    (THOR_SCORREVOLE_ADAM=conserva), stato di Adam — che la finestra dopo adotta e aggiorna sul
    posto. La finestra dopo puo' toccare l'oggetto mentre il thread scrive la copia."""
    return _copia_profonda(_stato_ternario(obj))


class _FotoPortati:
    """Stato latente degli strati portati, copiato alla fine della finestra (patch_tempo_prepara_asincrona).
    `stati` = {L: {"gu": salvato, "dw": salvato | None}}, stessa struttura che prepara() scrive."""

    def __init__(self, stati: dict):
        self.stati = stati


class InterruzioneSimulata(RuntimeError):'''),
    # -- 2. Ripresa: fotografia (sincrona) e scrittura della fotografia (nel thread)
    ('''    def completa(self, K: int, tmp: Path, seme_h: dict, seme_e: dict, portati: "list[int]") -> None:''',
     '''    def fotografa(self, portati: dict) -> "_FotoPortati":
        """SINCRONO e breve (patch_tempo_prepara_asincrona): copia su device dello stato latente
        degli strati portati, nello stesso ordine e nella stessa struttura di prepara()."""
        with torch.no_grad():
            return _FotoPortati({L: {"gu": _stato_ternario_copia(fam["gu"]),
                                     "dw": None if fam["dw"] is None else _stato_ternario_copia(fam["dw"])}
                                 for L, fam in portati.items()})

    def prepara_da_foto(self, K: int, foto: "_FotoPortati") -> Path:
        """(nel thread di scrittura) come prepara(), dalla fotografia: stessi file, stesso contenuto.
        Ogni copia si rilascia appena scritta."""
        stati, foto.stati = foto.stati, None
        if stati is None:
            raise RuntimeError(f"fotografia degli strati portati per la finestra {K} gia' usata")
        tmp = self._tmp(K)
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        for L in list(stati):
            torch.save(stati.pop(L), tmp / f"portato_{L:02d}.pt")
        return tmp

    def completa(self, K: int, tmp: Path, seme_h: dict, seme_e: dict, portati: "list[int]") -> None:'''),
    # -- 3. la transizione (thread di scrittura) scrive prima lo stato portato dalla fotografia
    ('''    def _transizione(k0, tmp, seme_h, seme_e, portati_ids, commit):
        if tmp is not None:
            ripresa.completa(k0 + 1, tmp, seme_h, seme_e, portati_ids)''',
     '''    def _transizione(k0, tmp, seme_h, seme_e, portati_ids, commit):
        if isinstance(tmp, _FotoPortati):
            # ⭐ 25/9 (patch_tempo_prepara_asincrona): lo stato portato si scrive QUI, nel thread,
            # dalla fotografia; ordine del protocollo di ripresa invariato
            tmp = ripresa.prepara_da_foto(k0 + 1, tmp)
        if tmp is not None:
            ripresa.completa(k0 + 1, tmp, seme_h, seme_e, portati_ids)'''),
    # -- 4. alla fine della finestra: fotografia invece della scrittura sincrona
    ('''                tmp = ripresa.prepara(k0 + 1, portati)
''',
     '''                # ⭐ 25/9 (patch_tempo_prepara_asincrona): 2,1-2,6 s di torch.save tolti dal percorso critico
                tmp = (ripresa.fotografa(portati) if (PREPARA_ASINCRONA and scrittura_asincrona)
                       else ripresa.prepara(k0 + 1, portati))
'''),
]}

if __name__ == "__main__":
    raise SystemExit(esegui_patch(__doc__.splitlines()[0], PIANO, MARCA))

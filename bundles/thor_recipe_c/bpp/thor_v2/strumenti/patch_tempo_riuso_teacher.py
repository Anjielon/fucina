"""Patch TEMPO «riuso del teacher fra finestre» (leva B / L2 del piano) — 25/9 notte.

Nella forgia scorrevole a passo 1 la finestra [k-1,k] calcola gia', per ogni batch, l'uscita
del teacher dello strato k sul suo ingresso vero: e' il terzo elemento della cache
`h_teacher_cache_per_b[b] = [h_{k-1}, F_{k-1}(h_{k-1}), F_k(F_{k-1}(h_{k-1}))]`. La finestra
[k,k+1] parte da h_k = F_{k-1}(h_{k-1}) (il confine a meta' consegna ESATTAMENTE quel tensore
come seme) e `precalcola_teacher` rifa' F_k(h_k) da capo: stesso strato teacher (resta caricato
fra le due finestre), stesso tensore d'ingresso, stesso posizionale. Oggi sono 33 passaggi di
strato per finestra: 13,7 s per 66 passaggi misurati con py-spy nel pre-passo di [3,4] di
conf6b-S1 (rapporto costo_passo.md §4) → ~6,8 s a finestra.

Cosa cambia:
  valida_finestra_8_12.py — ottimizza_finestra_JOINT accetta
    teacher_noto_per_b={b: [F_{L0}(h0_b), ...]}  uscite teacher GIA' NOTE dei primi strati del
                         chunk (lista non vuota, piu' corta del chunk): si calcolano solo gli altri;
    cache_teacher_uscita={}  riempito con la cache completa {b: [h0, h1, ..., hW]} (riferimenti).
    Default None = comportamento invariato. Solo finestra a chunk unico (in produzione lo e').
  forgia_scorrevole.py — con THOR_SCORREVOLE_RIUSA_TEACHER_FINESTRA=1 (default; fuori ricetta):
    uscita teacher del primo strato per il confine = cache[b][1] (lo stesso tensore che il gancio
    `_CatturaUscitaTeacher` catturava) e cache[b][2:] passano come `teacher_noto_per_b` alla
    finestra dopo. Prima finestra e prima finestra dopo una ripresa: tutto ricalcolato (come oggi).

Verifica AL BIT a ogni finestra: il controllo gia' esistente ricalcola da zero il teacher del
primo strato sul batch 0 e ora pretende torch.equal (non solo scarto relativo < 1e-2) quando
il riuso e' attivo — nell'ultima finestra compreso. Se non fosse identico si FERMA con un
messaggio che dice come spegnere la leva (i numeri cambierebbero). In conf6b-S1 e conf6e-S2 lo
stesso ricalcolo ha sempre dato "scarto relativo 0.0e+00" (formato .1e: zero esatto).

NUMERI: IDENTICI AL BIT (stesso calcolo, fatto una volta sola). MEMORIA: +1 volta il seme
teacher (33 x 32 MiB = 1,03 GiB) tenuto fra le due finestre.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_tempo_comune import esegui_patch  # noqa: E402

MARCA = "patch_tempo_riuso_teacher"

VALIDA = [
    # -- 1. firma (aggancio condiviso con le altre patch: ognuna aggiunge i suoi parametri dopo)
    ('''                              adam_ripartenza: str = "riparti"''',
     '''                              adam_ripartenza: str = "riparti",
                              teacher_noto_per_b: "dict | None" = None,
                              cache_teacher_uscita: "dict | None" = None'''),
    # -- 2. la precalcolazione con le uscite gia' note
    ('''def ottimizza_finestra_JOINT(strati_correnti: dict, ids_batch: torch.Tensor, cfg, n_passi: int,''',
     '''def _precalcola_teacher_con_noti(strati_teacher_fn: list, h0: torch.Tensor, noti, b: int) -> list:
    """⭐ 25/9 (patch_tempo_riuso_teacher): come `precalcola_teacher`, ma le uscite dei primi
    len(noti) strati sono GIA' note — calcolate dalla finestra precedente con lo STESSO strato
    teacher sullo STESSO tensore d'ingresso: si calcolano solo gli strati successivi, nello
    stesso modo (no_grad, un forward per strato). Ritorna [h0, *noti, ...] come precalcola_teacher."""
    if not isinstance(noti, (list, tuple)) or not noti:
        raise ValueError(f"teacher_noto_per_b[{b}]: attesa una lista non vuota di tensori, ricevuto {type(noti).__name__}")
    if len(noti) >= len(strati_teacher_fn):
        raise ValueError(f"teacher_noto_per_b[{b}]: {len(noti)} uscite note per {len(strati_teacher_fn)} strati: "
                         "almeno l'ultimo strato del chunk va calcolato qui")
    for i, t in enumerate(noti):
        if not torch.is_tensor(t) or t.shape != h0.shape or t.dtype != h0.dtype or t.device != h0.device:
            raise ValueError(f"teacher_noto_per_b[{b}][{i}]: forma/tipo/device diversi dall'ingresso "
                             f"{tuple(h0.shape)}/{h0.dtype}/{h0.device}")
    h_l = [h0, *noti]
    with torch.no_grad():
        for strato in strati_teacher_fn[len(noti):]:
            h_l.append(strato(h_l[-1]))
    return h_l


def ottimizza_finestra_JOINT(strati_correnti: dict, ids_batch: torch.Tensor, cfg, n_passi: int,'''),
    # -- 3. uso nella finestra
    ('''        h_teacher_cache_per_b = {b: precalcola_teacher(strati_teacher_fn, h0_per_b[b])
                                  for b in range(n_batch_loc)}
''',
     '''        # ⭐ 25/9 (patch_tempo_riuso_teacher): uscite teacher gia' note dalla finestra precedente
        if (teacher_noto_per_b is not None or cache_teacher_uscita is not None) and len(chunk_layers_list) > 1:
            raise ValueError("teacher_noto_per_b / cache_teacher_uscita: previsti per una finestra a chunk unico "
                             f"(qui {len(chunk_layers_list)} chunk: THOR_VALIDA_SOTTOFINESTRA)")
        if teacher_noto_per_b is not None:
            _mancano = [bb for bb in range(n_batch_loc) if bb not in teacher_noto_per_b]
            if _mancano:
                raise ValueError(f"teacher_noto_per_b: mancano le batch {_mancano[:8]}")
            h_teacher_cache_per_b = {b: _precalcola_teacher_con_noti(strati_teacher_fn, h0_per_b[b],
                                                                     teacher_noto_per_b[b], b)
                                      for b in range(n_batch_loc)}
            log(f"  chunk {chunk}: teacher dei primi {len(teacher_noto_per_b[0])} strati RIUSATO dalla finestra "
                f"precedente ({n_batch_loc} batch)")
        else:
            h_teacher_cache_per_b = {b: precalcola_teacher(strati_teacher_fn, h0_per_b[b])
                                      for b in range(n_batch_loc)}
        if cache_teacher_uscita is not None:
            cache_teacher_uscita.clear()
            cache_teacher_uscita.update(h_teacher_cache_per_b)
'''),
]

FORGIA = [
    # -- 1. interruttore (fuori ricetta) e lettura della cache
    ('''class InterruzioneSimulata(RuntimeError):''',
     '''# ⭐ 25/9 (patch_tempo_riuso_teacher): il teacher del primo strato di ogni finestra viene dalla
# finestra precedente (stesso strato, stesso ingresso); verificato AL BIT sul batch 0 a ogni finestra
RIUSA_TEACHER_FINESTRA = os.getenv("THOR_SCORREVOLE_RIUSA_TEACHER_FINESTRA", "1") == "1"
_NON_RICETTA |= {"THOR_SCORREVOLE_RIUSA_TEACHER_FINESTRA"}


def _teacher_da_cache(cache: dict, n_batch: int, w: int) -> "tuple[dict, dict]":
    """Dalla cache teacher completa della finestra {b: [h0, h1, ..., hW]}: (uscita del primo
    strato per il confine, uscite gia' note per la finestra dopo = cache[b][2:])."""
    if sorted(cache) != list(range(n_batch)):
        raise RuntimeError(f"cache teacher: batch {sorted(cache)[:6]}... attese 0..{n_batch - 1}")
    for b, lista in cache.items():
        if len(lista) != w + 1:
            raise RuntimeError(f"cache teacher batch {b}: {len(lista)} elementi, attesi {w + 1}")
    return {b: cache[b][1] for b in range(n_batch)}, {b: list(cache[b][2:]) for b in range(n_batch)}


class InterruzioneSimulata(RuntimeError):'''),
    # -- 2. il controllo del batch 0 pretende l'identita' AL BIT quando il riuso e' attivo
    ('''def _controlla_teacher_catturato(pulito, quant, ids, cfg, emb, h_seme, catturato: dict, k: int) -> float:''',
     '''def _controlla_teacher_catturato(pulito, quant, ids, cfg, emb, h_seme, catturato: dict, k: int,
                                 esigi_bit: bool = False) -> float:'''),
    ('''    if rel > 1e-2:
        raise RuntimeError(f"finestra {k}: teacher catturato diverso dal ricalcolato (rel {rel:.3e})")
    return rel''',
     '''    if rel > 1e-2:
        raise RuntimeError(f"finestra {k}: teacher catturato diverso dal ricalcolato (rel {rel:.3e})")
    if esigi_bit and not torch.equal(h_rif[0], catturato[0]):
        # (patch_tempo_riuso_teacher) il riuso vale solo se e' lo STESSO calcolo, al bit
        raise RuntimeError(f"finestra {k}: teacher RIUSATO dalla finestra precedente NON identico al bit al "
                           f"ricalcolato (rel {rel:.3e}): il riuso cambierebbe i numeri. Spegnere con "
                           "THOR_SCORREVOLE_RIUSA_TEACHER_FINESTRA=0")
    return rel'''),
    # -- 3. stato del riuso, e rifiuto se valida non ha la patch
    ('''    h_seme = e_seme = None
    t_tot = time.time()
''',
     '''    h_seme = e_seme = None
    # ⭐ 25/9 (patch_tempo_riuso_teacher)
    riuso_finestra = riusa_teacher and RIUSA_TEACHER_FINESTRA
    teacher_noto = teacher_noto_in_uso = None
    if riuso_finestra:
        _firma = inspect.signature(V.ottimizza_finestra_JOINT).parameters
        if "teacher_noto_per_b" not in _firma or "cache_teacher_uscita" not in _firma:
            raise RuntimeError("THOR_SCORREVOLE_RIUSA_TEACHER_FINESTRA=1 ma valida_finestra_8_12 non ha "
                               "teacher_noto_per_b/cache_teacher_uscita (patch_tempo_riuso_teacher applicata a meta')")
    t_tot = time.time()
'''),
    # -- 4. alla chiamata: uscite note della finestra precedente + cache in uscita
    ('''            usa_cattura = riusa_teacher and not ultima
''',
     '''            cache_t = None
            teacher_noto_in_uso, teacher_noto = teacher_noto, None
            if riuso_finestra:
                # ⭐ 25/9 (patch_tempo_riuso_teacher): F_k(h_k) gia' calcolato dalla finestra [k-1,k]
                if teacher_noto_in_uso is not None:
                    kw["teacher_noto_per_b"] = teacher_noto_in_uso
                if not ultima:
                    cache_t = {}
                    kw["cache_teacher_uscita"] = cache_t
            usa_cattura = riusa_teacher and not ultima and not riuso_finestra
'''),
    # -- 5. al confine: uscita del primo strato e uscite note per la finestra dopo, dalla cache
    ('''                teacher_k = cattura.per_batch(n_batch) if usa_cattura else None
                if teacher_k is not None:
                    rel = _controlla_teacher_catturato(puliti[k0], quant[k0], ids_batch, cfg, emb,
                                                       h_seme, teacher_k, k0)''',
     '''                teacher_k = cattura.per_batch(n_batch) if usa_cattura else None
                if cache_t is not None:
                    teacher_k, teacher_noto = _teacher_da_cache(cache_t, n_batch, w)
                    cache_t = None
                if teacher_k is not None:
                    rel = _controlla_teacher_catturato(puliti[k0], quant[k0], ids_batch, cfg, emb,
                                                       h_seme, teacher_k, k0, esigi_bit=riuso_finestra)'''),
    # -- 6. ultima finestra: anche li' il teacher riusato si controlla al bit
    ('''            else:
                _consegna(_transizione, k0, None, None, None, [], [(L, risultato[L]) for L in commessi])''',
     '''            else:
                if riuso_finestra and teacher_noto_in_uso is not None:
                    # (patch_tempo_riuso_teacher) ultima finestra: nessun confine, stesso controllo al bit
                    rel = _controlla_teacher_catturato(puliti[k0], quant[k0], ids_batch, cfg, emb, h_seme,
                                                       {0: teacher_noto_in_uso[0][0]}, k0, esigi_bit=True)
                    log(f"  teacher dello strato {k0} riusato dalla finestra precedente (batch 0 ricalcolato: "
                        f"identico al bit, scarto relativo {rel:.1e})")
                _consegna(_transizione, k0, None, None, None, [], [(L, risultato[L]) for L in commessi])'''),
    # -- 7. i riferimenti della finestra appena fatta non restano vivi oltre il necessario
    ('''            del risultato, cattura
''',
     '''            del risultato, cattura
            teacher_noto_in_uso = None      # (patch_tempo_riuso_teacher) usato: si libera
'''),
]

PIANO = {"valida_finestra_8_12.py": VALIDA, "forgia_scorrevole.py": FORGIA}

if __name__ == "__main__":
    raise SystemExit(esegui_patch(__doc__.splitlines()[0], PIANO, MARCA))

"""Patch TEMPO «export incrementale» (leva F di PIANO_TEMPO_SCORREVOLE.md) — 25/9 mattina. NON APPLICATA.

Oggi `esporta_gguf_finestra` impacchetta i 40 strati tutti alla FINE: `stato/controllo_ibrido/
export.log` (24/9) misura ~20 s a strato (21:55:14→22:01:42, 19 strati) piu' ~67 s per il primo
(verifiche); su CPU, 25/9 05:02, lo strato 1 di conf6h-B1 costa 22,8 s (carica 0,7 + gate 11,0 +
up 9,4 + down 1,8), quasi tutto negli int64 da 2 GiB che `pack_da_d_q` crea sull'intera famiglia
(picco RSS 4,90 GiB). Due leve, entrambe IDENTICHE AL BIT nel GGUF finale:

  THOR_EXPORT_A_BLOCCHI=1 (default con la patch): la STESSA funzione di impacchettamento
    (`impacchetta_famiglia`, `pack_tq1_0_da_d_q`) su blocchi di THOR_EXPORT_PACCHETTI_BLOCCO
    esperti (default 32), concatenati sull'asse degli esperti. Ogni blocco di pesi (1024 per
    TQ1_0R_1024, 256 per TQ1_0) si impacchetta da se': stessi byte. Provato: sha256 identici su
    uno strato VERO (strumenti/misura_impacchettamento.py --confronta) e dal collaudo. Picco RSS
    1,97 GiB invece di 4,90; gate 1,0-2,0 s invece di 11,0 (misure sotto carico, vedi piano).
    Verifiche (`verifica_packer`, controllo del down) INVARIATE, sulla via di sempre.
  THOR_EXPORT_INCREMENTALE=1 (default 0: il suo costo sulla GPU non e' misurato): appena
    `scrivi_codici_su_disco` ha scritto la marca di uno strato, un thread a priorita' minima
    (nice 19 del solo thread) impacchetta quello strato A BLOCCHI e scrive in
    THOR_EXPORT_PACCHETTI_DIR/<corsa>/ i byte pronti (pacchetto_LL_<famiglia>.npy) e, PER ULTIMO,
    la marca pacchetto_LL.json: sha256 dei byte, impronta (dimensione, mtime_ns, ctime_ns, inode)
    di ogni codice sorgente e della marca fatto_forgia, testo della marca, sha256 dei file
    dell'impacchettatore e versione di numpy, e quali verifiche sono state fatte. L'export, se il
    pacchetto e' valido, copia quei byte invece di ricalcolarli; altrimenti (pacchetto assente,
    interrotto, codici riscritti dopo, byte alterati, impacchettatore cambiato, campione diverso)
    lo strato passa dalla via di sempre, e lo dice nel log. Il GGUF e' lo stesso al bit.
    Verifiche: il thread fa `verifica_packer` (gate) e il controllo del down sul PRIMO strato che
    impacchetta (e sempre sullo strato 0), a blocchi: stessa copertura di oggi (tutti gli esperti).
    L'export accetta il pacchetto del primo strato solo se quelle verifiche risultano fatte,
    altrimenti lo impacchetta lui con le verifiche di sempre. In piu' (controllo NUOVO, piu'
    stretto di oggi): per ogni pacchetto usato ri-impacchetta ORA due esperti per famiglia dai
    codici su disco e pretende gli stessi byte. `_verifica_rilettura` resta com'e'.
    Dopo la rilettura riuscita i pacchetti si cancellano (THOR_EXPORT_PACCHETTI_TIENI=1 li tiene).
    Spazio: ~174 MiB a strato (~7 GB per 40); prima di scrivere si pretende che restino liberi
    THOR_EXPORT_PACCHETTI_MARGINE_GIB (default 8) GiB, piu' la taglia del GGUF base se il disco e'
    quello del GGUF d'uscita (E11 del verificatore: @@ROOT:models@@ ha ~50 G); altrimenti niente
    pacchetto (via di sempre). Senza THOR_EXPORT_PACCHETTI_DIR l'import di valida si ferma subito.

Interruttori fuori ricetta (_NON_RICETTA della scorrevole): cambiano solo quando si impacchetta.
Collaudo: strumenti/collaudo_export_incrementale.py (CPU, scala giocattolo, GGUF al bit).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_tempo_comune import esegui_patch  # noqa: E402

MARCA = "patch_tempo_export_incrementale"

BLOCCO_FUNZIONI = r'''# ═══ ⭐ 25/9 patch_tempo_export_incrementale (leva F di PIANO_TEMPO_SCORREVOLE.md) ═══
# A_BLOCCHI: stessa funzione di impacchettamento su blocchi di esperti (stessi byte, meno memoria).
# INCREMENTALE: pacchetti pronti per strato, scritti da un thread appena i codici sono su disco;
# l'export li copia se validi (impronte dei sorgenti, sha256, impacchettatore, campione), altrimenti
# impacchetta come sempre. Docstring completo in strumenti/patch_tempo_export_incrementale.py.
EXPORT_A_BLOCCHI = os.getenv("THOR_EXPORT_A_BLOCCHI", "1") == "1"
EXPORT_INCREMENTALE = os.getenv("THOR_EXPORT_INCREMENTALE", "0") == "1"
PACCHETTI_BLOCCO = int(os.getenv("THOR_EXPORT_PACCHETTI_BLOCCO", "32"))
PACCHETTI_TIENI = os.getenv("THOR_EXPORT_PACCHETTI_TIENI", "0") == "1"
PACCHETTI_MARGINE_GIB = float(os.getenv("THOR_EXPORT_PACCHETTI_MARGINE_GIB", "8"))
_PACCHETTI_RADICE = os.getenv("THOR_EXPORT_PACCHETTI_DIR", "")
if PACCHETTI_BLOCCO < 1:
    raise ValueError(f"THOR_EXPORT_PACCHETTI_BLOCCO={PACCHETTI_BLOCCO}: serve >= 1")
if EXPORT_INCREMENTALE and not _PACCHETTI_RADICE:
    raise RuntimeError("THOR_EXPORT_INCREMENTALE=1 senza THOR_EXPORT_PACCHETTI_DIR: servono ~174 MiB a strato su "
                       "un disco con spazio (NON quello dei codici se e' quasi pieno)")
_PACCHETTO_VERSIONE = 1
_FILE_STRATO = ("Dg", "Qg", "Du", "Qu", "Dd", "Qd")
_IMPACCHETTATORE = None            # ThreadPoolExecutor(1), creato al primo pacchetto
_IMPACCHETTAMENTI: list = []       # [(strato, future)] non ancora attesi dall'export
_VERIFICHE_QUI = {"gate": False, "down": False}   # verifiche gia' fatte dal thread in questo processo
_IMPRONTE_IMPACCHETTATORE: dict = {}


def _pacchetto_gate_up(D: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """= esporta_schur_esperti.impacchetta_famiglia(D, Q), a blocchi di PACCHETTI_BLOCCO esperti:
    ogni blocco da 1024 pesi si impacchetta da se', la concatenazione sull'asse 0 da' gli stessi byte."""
    from esporta_schur_esperti import impacchetta_famiglia
    b = PACCHETTI_BLOCCO
    return np.concatenate([impacchetta_famiglia(D[e:e + b], Q[e:e + b]) for e in range(0, Q.shape[0], b)], axis=0)


def _pacchetto_down_4assi(D: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """= pack_tq1_0_da_d_q(D, Q) (4 assi: E, out, ng, 54), a blocchi di esperti."""
    from pack_tq1_0_standard import pack_tq1_0_da_d_q
    b = PACCHETTI_BLOCCO
    return np.concatenate([pack_tq1_0_da_d_q(D[e:e + b], Q[e:e + b]) for e in range(0, Q.shape[0], b)], axis=0)


def _dir_pacchetti() -> Path:
    return Path(_PACCHETTI_RADICE) / ID_CORSA


def _sha256_byte(a: np.ndarray) -> str:
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8).reshape(-1)).hexdigest()


def _impronte_impacchettatore() -> dict:
    """sha256 dei sorgenti che decidono i byte (e versione di numpy): se cambiano, i pacchetti non valgono."""
    if not _IMPRONTE_IMPACCHETTATORE:
        import hashlib
        import esporta_schur_esperti as _e
        import gptq_1024_lib as _g
        import pack_tq1_0_standard as _p
        import tq1_0r_1024_codec as _c
        for m in (_e, _g, _p, _c):
            _IMPRONTE_IMPACCHETTATORE[Path(m.__file__).name] = hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
        _IMPRONTE_IMPACCHETTATORE["numpy"] = np.__version__
    return dict(_IMPRONTE_IMPACCHETTATORE)


def _impronte_sorgenti(L: int) -> dict:
    """(dimensione, mtime_ns, ctime_ns, inode, device) della marca e di ogni codice dello strato
    (None = assente): np.save riscrive il file e cambia almeno mtime/ctime, os.replace l'inode."""
    out = {}
    for p in [OUT_DIR / f"fatto_forgia_{L:02d}.ok"] + [OUT_DIR / f"{x}_{L:02d}.npy" for x in _FILE_STRATO]:
        try:
            s = p.stat()
            out[p.name] = [s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_ino, s.st_dev]
        except FileNotFoundError:
            out[p.name] = None
    return out


def _salva_npy_atomico(p: Path, a: np.ndarray) -> str:
    q = p.with_name(p.name + ".tmp")
    with open(q, "wb") as f:
        np.save(f, a)
    os.replace(q, p)
    return _sha256_byte(a)


def _verifica_packer_a_blocchi(D: np.ndarray, Q: np.ndarray, nome: str) -> None:
    """La stessa verifica_packer di oggi (trit e scala fp16 al bit), blocco per blocco: tutti gli esperti."""
    from esporta_schur_esperti import verifica_packer
    b = PACCHETTI_BLOCCO
    for e in range(0, Q.shape[0], b):
        verifica_packer(D[e:e + b], Q[e:e + b], f"{nome} [esperti {e}:{min(e + b, Q.shape[0])}]",
                        righe_esperto=Q[e:e + b].shape[0])


def _verifica_down_a_blocchi(Dd: np.ndarray, Qd: np.ndarray, nome_d: str) -> None:
    """Lo stesso controllo del down di esporta_gguf_finestra (dequantize_blocks della libreria vera,
    scarto < 1e-3), blocco per blocco: tutti gli esperti."""
    from gguf.quants import TQ1_0 as _TQ1_0_STD
    from pack_tq1_0_standard import pack_tq1_0_da_d_q
    b = PACCHETTI_BLOCCO
    for e in range(0, Qd.shape[0], b):
        packed_prova = pack_tq1_0_da_d_q(Dd[e:e + b], Qd[e:e + b])
        dequant = _TQ1_0_STD.dequantize_blocks(packed_prova.reshape(-1, packed_prova.shape[-1]))
        atteso = (Qd[e:e + b].astype(np.float32) * Dd[e:e + b].repeat(256, axis=-1)).reshape(-1, 256)
        scarto = np.abs(dequant - atteso).max()
        assert scarto < 1e-3, f"{nome_d}: round-trip down_proj rotto, scarto={scarto:.3e}"


def _impacchetta_in_cache(L: int) -> None:
    """(nel thread) pacchetto dello strato L dai codici su disco; marca .json scritta per ultima."""
    import shutil
    import threading
    try:
        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 19)   # solo questo thread
    except (OSError, AttributeError):
        pass
    t0 = time.time()
    prima = _impronte_sorgenti(L)
    marca = OUT_DIR / f"fatto_forgia_{L:02d}.ok"
    if prima[marca.name] is None:
        raise RuntimeError(f"pacchetto dello strato {L}: marca {marca} assente")
    testo_marca = marca.read_text()
    verifica_gate = not _VERIFICHE_QUI["gate"] or L == 0
    verifica_down = not _VERIFICHE_QUI["down"] or L == 0
    pacchetti: dict = {}
    for fam, xD, xQ in (("ffn_gate_exps", "Dg", "Qg"), ("ffn_up_exps", "Du", "Qu")):
        nome = f"blk.{L}.{fam}.weight"
        D, Q = np.load(OUT_DIR / f"{xD}_{L:02d}.npy"), np.load(OUT_DIR / f"{xQ}_{L:02d}.npy")
        if verifica_gate and fam == "ffn_gate_exps":
            _verifica_packer_a_blocchi(D, Q, nome)
            _VERIFICHE_QUI["gate"] = True       # fatta: non si ripete sugli strati dopo, anche se il pacchetto salta
        pacchetti[nome] = _pacchetto_gate_up(D, Q)
        del D, Q
    ha_down = prima[f"Dd_{L:02d}.npy"] is not None and prima[f"Qd_{L:02d}.npy"] is not None
    if ha_down:
        nome_d = f"blk.{L}.ffn_down_exps.weight"
        Dd, Qd = np.load(OUT_DIR / f"Dd_{L:02d}.npy"), np.load(OUT_DIR / f"Qd_{L:02d}.npy")
        if verifica_down:
            _verifica_down_a_blocchi(Dd, Qd, nome_d)
            _VERIFICHE_QUI["down"] = True
        pacchetti[nome_d] = _pacchetto_down_4assi(Dd, Qd).reshape(Dd.shape[0], Dd.shape[1], -1)
        del Dd, Qd
    if _impronte_sorgenti(L) != prima:
        log(f"⚠️ pacchetto dello strato {L} NON scritto: i codici sono cambiati mentre li impacchettavo "
            "(l'export lo impacchettera' con la via di sempre)")
        return
    d = _dir_pacchetti()
    d.mkdir(parents=True, exist_ok=True)
    serve = sum(int(a.nbytes) for a in pacchetti.values()) + (1 << 20)
    riserva = PACCHETTI_MARGINE_GIB * 2**30
    try:
        if os.stat(d).st_dev == os.stat(GGUF_OUT.parent if GGUF_OUT.parent.exists() else OUT_DIR).st_dev:
            riserva += GGUF_BASE.stat().st_size          # il GGUF d'uscita finira' sullo stesso disco
    except OSError:
        pass
    libero = shutil.disk_usage(d).free
    if libero - serve < riserva:
        log(f"⚠️ pacchetto dello strato {L} NON scritto: {libero / 2**30:.1f} GiB liberi in {d}, ne servono "
            f"{(serve + riserva) / 2**30:.1f} col margine (l'export lo impacchettera' con la via di sempre)")
        return
    m = d / f"pacchetto_{L:02d}.json"
    if m.exists():
        m.unlink()                  # la marca vecchia sparisce PRIMA dei byte nuovi
    tensori = {}
    for nome, a in pacchetti.items():
        p = d / f"pacchetto_{L:02d}_{nome.split('.')[2]}.npy"
        tensori[nome] = {"file": p.name, "forma": list(a.shape), "sha256": _salva_npy_atomico(p, a)}
    meta = {"versione": _PACCHETTO_VERSIONE, "strato": L, "corsa": ID_CORSA, "out_dir": str(OUT_DIR.resolve()),
            "marca": testo_marca, "sorgenti": prima, "impacchettatore": _impronte_impacchettatore(),
            "tensori": tensori, "senza_down": not ha_down, "verificato_gate": verifica_gate,
            "verificato_down": bool(verifica_down and ha_down), "blocco": PACCHETTI_BLOCCO,
            "secondi": round(time.time() - t0, 2)}
    q = m.with_name(m.name + ".tmp")
    q.write_text(json.dumps(meta, indent=1))
    os.replace(q, m)
    log(f"  pacchetto dello strato {L} pronto in {time.time() - t0:.1f}s ({serve / 2**20:.0f} MiB"
        + (", con le verifiche del packer" if verifica_gate else "") + f") → {d}")


def _accoda_impacchettamento(strati) -> None:
    """(da scrivi_codici_su_disco, dopo le marche) accoda i pacchetti; nessuno li aspetta tranne l'export."""
    global _IMPACCHETTATORE
    if not EXPORT_INCREMENTALE:
        return
    if _IMPACCHETTATORE is None:
        from concurrent.futures import ThreadPoolExecutor
        _IMPACCHETTATORE = ThreadPoolExecutor(max_workers=1, thread_name_prefix="impacchetta")
    for L in strati:
        _IMPACCHETTAMENTI.append((L, _IMPACCHETTATORE.submit(_impacchetta_in_cache, L)))


def _attendi_impacchettamenti() -> None:
    if not _IMPACCHETTAMENTI:
        return
    t0 = time.time()
    for L, fut in list(_IMPACCHETTAMENTI):
        try:
            fut.result()
        except Exception as e:  # noqa: BLE001 — lo strato passa dalla via di sempre, che rifa' le verifiche
            log(f"⚠️ pacchetto anticipato dello strato {L} fallito ({type(e).__name__}: {e}): "
                "lo strato si impacchetta ora, con le verifiche di sempre")
    _IMPACCHETTAMENTI.clear()
    log(f"export incrementale: attesa dei pacchetti in coda {time.time() - t0:.1f}s")


def _pacchetto_pronto(L: int, testo_marca: str, serve_verifica_gate: bool, serve_verifica_down: bool) -> "dict | None":
    """Byte pronti dello strato L se il pacchetto e' valido per i codici di OGGI, altrimenti None (col motivo)."""
    d = _dir_pacchetti()
    m = d / f"pacchetto_{L:02d}.json"

    def no(motivo: str):
        log(f"  strato {L}: pacchetto pronto NON usato ({motivo}): impacchetto ora")
        return None
    if not m.exists():
        return no("assente")
    try:
        meta = json.loads(m.read_text())
    except (OSError, ValueError) as e:
        return no(f"marca illeggibile: {e}")
    if meta.get("versione") != _PACCHETTO_VERSIONE:
        return no(f"versione {meta.get('versione')}")
    if meta.get("marca") != testo_marca or meta.get("out_dir") != str(OUT_DIR.resolve()):
        return no("corsa o marca fatto_forgia diverse")
    if meta.get("sorgenti") != _impronte_sorgenti(L):
        return no("codici su disco cambiati dopo il pacchetto")
    if meta.get("impacchettatore") != _impronte_impacchettatore():
        return no("impacchettatore (sorgenti o numpy) cambiato")
    if serve_verifica_gate and not meta.get("verificato_gate"):
        return no("serve la verifica del packer e il pacchetto non l'ha fatta")
    if serve_verifica_down and not meta.get("senza_down") and not meta.get("verificato_down"):
        return no("serve la verifica del down e il pacchetto non l'ha fatta")
    attesi = {f"blk.{L}.ffn_gate_exps.weight", f"blk.{L}.ffn_up_exps.weight"} | (
        set() if meta.get("senza_down") else {f"blk.{L}.ffn_down_exps.weight"})
    if set(meta.get("tensori", {})) != attesi:
        return no("tensori del pacchetto diversi da quelli attesi")
    pacchetti = {}
    for nome, t in meta["tensori"].items():
        try:
            a = np.load(d / t["file"])
        except (OSError, ValueError) as e:
            return no(f"{t['file']} illeggibile: {e}")
        if a.dtype != np.uint8 or list(a.shape) != t["forma"] or _sha256_byte(a) != t["sha256"]:
            return no(f"{nome}: byte su disco diversi da quelli scritti (sha256)")
        pacchetti[nome] = a
    # controllo NUOVO: due esperti per famiglia ri-impacchettati ORA dai codici su disco
    from esporta_schur_esperti import impacchetta_famiglia
    from pack_tq1_0_standard import pack_tq1_0_da_d_q
    for fam, xD, xQ in (("ffn_gate_exps", "Dg", "Qg"), ("ffn_up_exps", "Du", "Qu"), ("ffn_down_exps", "Dd", "Qd")):
        nome = f"blk.{L}.{fam}.weight"
        if nome not in pacchetti:
            continue
        D = np.load(OUT_DIR / f"{xD}_{L:02d}.npy", mmap_mode="r")
        Q = np.load(OUT_DIR / f"{xQ}_{L:02d}.npy", mmap_mode="r")
        E = Q.shape[0]
        for e in sorted({(L * 37 + 11) % E, E - 1}):
            De, Qe = np.ascontiguousarray(D[e:e + 1]), np.ascontiguousarray(Q[e:e + 1])
            rif = (pack_tq1_0_da_d_q(De, Qe).reshape(1, De.shape[1], -1) if fam == "ffn_down_exps"
                   else impacchetta_famiglia(De, Qe))
            if not np.array_equal(rif, pacchetti[nome][e:e + 1]):
                return no(f"{nome}: l'esperto {e} ri-impacchettato ora differisce dal pacchetto")
        del D, Q
    return {"pacchetti": pacchetti, "senza_down": bool(meta["senza_down"]),
            "verificato_gate": bool(meta["verificato_gate"]), "verificato_down": bool(meta["verificato_down"])}


def _pulisci_pacchetti(layers) -> None:
    d = _dir_pacchetti()
    liberati = 0
    for L in layers:
        for p in d.glob(f"pacchetto_{L:02d}*"):
            liberati += p.stat().st_size
            p.unlink()
    try:
        d.rmdir()
    except OSError:
        pass
    log(f"export incrementale: pacchetti cancellati dopo la rilettura ({liberati / 2**30:.2f} GiB liberati)")


# ═════════════════ 6. ESPORTAZIONE — SOLO la finestra 8-12 ══════════════════
def scrivi_codici_su_disco(risultato: dict) -> None:'''

VALIDA = [
    # -- 1. configurazione e funzioni, prima della sezione 6
    ('''# ═════════════════ 6. ESPORTAZIONE — SOLO la finestra 8-12 ══════════════════
def scrivi_codici_su_disco(risultato: dict) -> None:''', BLOCCO_FUNZIONI),
    # -- 2. i pacchetti partono appena le marche sono scritte
    ('''    log(f"codici scritti in {OUT_DIR} per gli strati {sorted(risultato)}")
''',
     '''    log(f"codici scritti in {OUT_DIR} per gli strati {sorted(risultato)}")
    _accoda_impacchettamento(sorted(risultato))   # ⭐ patch_tempo_export_incrementale (spento di default)
'''),
    # -- 3. l'export aspetta i pacchetti in coda
    ('''    layers = FINESTRA if layers is None else list(layers)
    mancanti = [L for L in layers if not (OUT_DIR / f"fatto_forgia_{L:02d}.ok").exists()]''',
     '''    layers = FINESTRA if layers is None else list(layers)
    _attendi_impacchettamenti()   # ⭐ patch_tempo_export_incrementale
    mancanti = [L for L in layers if not (OUT_DIR / f"fatto_forgia_{L:02d}.ok").exists()]'''),
    # -- 4. per ogni strato: byte pronti se il pacchetto e' valido, stessi controlli sul GGUF base
    ('''    for L in layers:
        Dg = np.load(OUT_DIR / f"Dg_{L:02d}.npy")''',
     '''    n_pronti = 0
    for L in layers:
        # ⭐ patch_tempo_export_incrementale: byte gia' pronti (validi per i codici di oggi) — stessi
        # controlli sul GGUF base e sulla marca della via di sempre, stessi messaggi
        pronto = (_pacchetto_pronto(L, marche[L], not verificato, not verificato_down)
                  if EXPORT_INCREMENTALE else None)
        if pronto is not None:
            for fam in ("ffn_gate_exps", "ffn_up_exps"):
                nome = f"blk.{L}.{fam}.weight"
                t_base = tens_by_name.get(nome)
                if t_base is None:
                    raise RuntimeError(f"{nome}: NON esiste in {GGUF_BASE} — schema divergente")
                if int(t_base.tensor_type) != TIPO_BASE_ATTESO:
                    raise RuntimeError(
                        f"{nome}: tipo base {int(t_base.tensor_type)} != atteso {TIPO_BASE_ATTESO} "
                        "(TQ1_0) — verifica manuale richiesta prima di sovrascrivere")
                bersagli[nome] = (pronto["pacchetti"][nome], T.TQ1_0R_1024)
            verificato = verificato or pronto["verificato_gate"]
            nome_d = f"blk.{L}.ffn_down_exps.weight"
            if nome_d in pronto["pacchetti"]:
                t_base_d = tens_by_name.get(nome_d)
                if t_base_d is None:
                    raise RuntimeError(f"{nome_d}: NON esiste in {GGUF_BASE} — schema divergente")
                tipo_base_d = int(t_base_d.tensor_type)
                if tipo_base_d == TIPO_VIETATO_DOWN:
                    raise RuntimeError(
                        f"{nome_d}: tipo base e' TQ1_0R_1024 (45, RUOTATO) — down_proj non va MAI "
                        "ruotato, questo indica un errore a monte, non un formato da sovrascrivere")
                if tipo_base_d != TIPO_NUOVO_DOWN:
                    log(f"  strato {L}: down_proj base tipo {tipo_base_d} (non ancora ternario) → TQ1_0")
                verificato_down = verificato_down or pronto["verificato_down"]
                bersagli[nome_d] = (pronto["pacchetti"][nome_d], T.TQ1_0)
                log(f"  strato {L}: impacchettato (gate+up+down) — byte gia' pronti")
            else:
                if down_dich[L] != "0":
                    raise RuntimeError(f"strato {L}: la marca dichiara down_ternario={down_dich[L]} ma Dd/Qd "
                                       f"mancano in {OUT_DIR} — rifiuto di esportare un down a 4 bit in silenzio")
                strati_senza_down.append(L)
                log(f"  strato {L}: impacchettato (gate+up) — down_proj SALTATO "
                    f"(Dd/Qd assenti su disco, resta Q4_K originale) — byte gia' pronti")
            n_pronti += 1
            continue
        Dg = np.load(OUT_DIR / f"Dg_{L:02d}.npy")'''),
    # -- 5. via di sempre: stessa funzione, a blocchi di esperti (stessi byte)
    ('''            bersagli[nome] = (impacchetta_famiglia(D, Q), T.TQ1_0R_1024)''',
     '''            # ⭐ patch_tempo_export_incrementale: a blocchi di esperti (stessi byte, ~4x meno memoria)
            bersagli[nome] = ((_pacchetto_gate_up(D, Q) if EXPORT_A_BLOCCHI else impacchetta_famiglia(D, Q)),
                              T.TQ1_0R_1024)'''),
    ('''            packed_down = pack_tq1_0_da_d_q(Dd, Qd)
            E_d, out_d = Dd.shape[0], Dd.shape[1]''',
     '''            packed_down = (_pacchetto_down_4assi(Dd, Qd) if EXPORT_A_BLOCCHI     # ⭐ stessi byte, a blocchi
                           else pack_tq1_0_da_d_q(Dd, Qd))
            E_d, out_d = Dd.shape[0], Dd.shape[1]'''),
    # -- 6. quanti strati dai pacchetti
    ('''    if strati_senza_down:
''',
     '''    if EXPORT_INCREMENTALE:   # ⭐ patch_tempo_export_incrementale
        log(f"export incrementale: {n_pronti}/{len(layers)} strati dai pacchetti pronti, "
            f"{len(layers) - n_pronti} impacchettati ora")
    if strati_senza_down:
'''),
    # -- 7. dopo la rilettura riuscita i pacchetti non servono piu'
    ('''    _verifica_rilettura(GGUF_OUT, bersagli)
    return GGUF_OUT''',
     '''    _verifica_rilettura(GGUF_OUT, bersagli)
    if EXPORT_INCREMENTALE and not PACCHETTI_TIENI:   # ⭐ patch_tempo_export_incrementale
        _pulisci_pacchetti(layers)
    return GGUF_OUT'''),
]

FORGIA = [
    ('''class InterruzioneSimulata(RuntimeError):''',
     '''# ⭐ 25/9 (patch_tempo_export_incrementale): interruttori dell'export, non della ricetta (stessi byte)
_NON_RICETTA |= {"THOR_EXPORT_A_BLOCCHI", "THOR_EXPORT_INCREMENTALE", "THOR_EXPORT_PACCHETTI_DIR",
                 "THOR_EXPORT_PACCHETTI_BLOCCO", "THOR_EXPORT_PACCHETTI_TIENI", "THOR_EXPORT_PACCHETTI_MARGINE_GIB"}


class InterruzioneSimulata(RuntimeError):'''),
]

PIANO = {"valida_finestra_8_12.py": VALIDA, "forgia_scorrevole.py": FORGIA}

if __name__ == "__main__":
    raise SystemExit(esegui_patch(__doc__.splitlines()[0], PIANO, MARCA))

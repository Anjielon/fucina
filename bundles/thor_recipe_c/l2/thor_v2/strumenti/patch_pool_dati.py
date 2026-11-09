"""Patch POOL DI BATCH DISTINTI (25/9 notte) per forgia_scorrevole.py — CAMBIA-NUMERI.

Perche': il pannello del 24/9 (stato/pannello/pannello_f01_miscela-2310.md) da' alla
miscela M2 un margine stretto sul testo generale (wiki 0,941x, edu 0,978x): servono piu'
dati DISTINTI. Oggi ogni finestra usa le STESSE 32 batch. Nella finestra scorrevole ogni
strato interno vive in DUE finestre ([k-1,k] e [k,k+1]): se le due finestre usano blocchi
di batch disgiunti, lo strato vede 2x i dati senza un passo in piu'.

Cosa fa (default = comportamento di prima, identico al bit):
  THOR_VALIDA_N_BATCH=P      pool: il caricatore costruisce P batch di addestramento (misurato:
                             con la miscela M2 le prime 32 di un pool da 64/96 sono IDENTICHE
                             AL BIT alle 32 di oggi, l'held-out pure)
  THOR_SCORREVOLE_BLOCCO=B   la finestra k usa le batch [(k mod m)B, (k mod m)B + B) con
                             m = P/B, piu' TUTTE le held-out; 0 = tutte (come prima)
  THOR_SCORREVOLE_POOL_RIUSA_TEACHER=1 (strumentazione, fuori ricetta): l'uscita del teacher
                             dell'ultimo strato della finestra k (gia' calcolata dentro
                             ottimizza_finestra_JOINT, `uscita_teacher_fine`) si tiene per il
                             blocco appena usato: al confine della finestra k+1 quel blocco NON
                             e' nella finestra ma il suo teacher dello strato k+1 e' gia' noto.

Il confine a meta' finestra diventa un confine sull'INTERO pool: batch della finestra ->
teacher catturato (come oggi); blocco della finestra precedente -> teacher tenuto; il resto
-> teacher calcolato; per tutti lo studente passa per lo strato appena commesso. La ripresa
salva i semi di tutto il pool (automatico: n_batch del manifesto), la ricetta contiene P e B
(una ripresa con un pool diverso e' RIFIUTATA), e il manifesto porta l'impronta sha256 degli
id del pool (dati cambiati fra interruzione e ripresa -> RIFIUTO). Il teacher tenuto NON si
salva: alla ripresa si ricalcola (stesso forward, deterministico). Il teacher tenuto vale SOLO
con W=2 (e' l'uscita dell'ultimo strato della finestra = lo strato del confine successivo).
Strumentazione (nessun numero cambia): picco di VRAM allocata per finestra nella riga «FATTA».

Collaudo (CPU, su copie): scratchpad del 25/9 notte, test_pool_scorrevole.py (113 controlli, ricetta
base fp32 e produzione bf16), piu' test_forgia_scorrevole.py e test_raccordo_scorrevole.py interi
sulla copia patchata. Documento: dati_distinti.md nello stesso scratchpad.

Uso:
    python3 strumenti/patch_pool_dati.py                     # SOLO controllo (agganci + sintassi)
    python3 strumenti/patch_pool_dati.py --file COPIA.py --applica   # applica a una copia
    python3 strumenti/patch_pool_dati.py --applica           # applica a ../forgia_scorrevole.py
⛔ Mai --applica sul file vero mentre una corsa che lo importa e' in coda o in corso.
Non scrive NULLA se un aggancio non compare esattamente una volta, se la patch e' gia'
applicata, o se il risultato non passa ast.parse. Scrittura atomica (os.replace).
"""
import argparse
import ast
import os
from pathlib import Path

PREDEFINITO = Path(__file__).resolve().parent.parent / "forgia_scorrevole.py"
MARCATORE = "THOR_SCORREVOLE_BLOCCO"

SOSTITUZIONI = [
    # -- 1. configurazione
    ('''TOLL_SEME_RICOSTRUITO = float(os.getenv("THOR_SCORREVOLE_TOLL_SEME", "1e-3"))
''',
     '''TOLL_SEME_RICOSTRUITO = float(os.getenv("THOR_SCORREVOLE_TOLL_SEME", "1e-3"))
# ⭐ 25/9 notte — POOL DI BATCH DISTINTI (strumenti/patch_pool_dati.py). CAMBIA-NUMERI: con
# THOR_VALIDA_N_BATCH=P e THOR_SCORREVOLE_BLOCCO=B<P la finestra k usa le batch di addestramento
# [(k mod m)B, (k mod m)B+B) (m=P/B) piu' tutte le held-out: ogni strato interno vive in due finestre
# e vede 2B batch distinte a parita' di passi. 0 = tutte le batch in ogni finestra (come prima).
BLOCCO_POOL = int(os.getenv("THOR_SCORREVOLE_BLOCCO", "0"))
# solo tempo, fuori ricetta: uscita del teacher dell'ultima finestra tenuta per il blocco che la lascia
RIUSA_TEACHER_POOL = os.getenv("THOR_SCORREVOLE_POOL_RIUSA_TEACHER", "1") == "1"
'''),
    # -- 2. il riuso del teacher del pool e' strumentazione (stesso forward), non ricetta
    ('''    "THOR_LLAMA_BIN_K1024", "THOR_B9_DIR",
}''',
     '''    "THOR_LLAMA_BIN_K1024", "THOR_B9_DIR", "THOR_SCORREVOLE_POOL_RIUSA_TEACHER",
}'''),
    # -- 3. funzioni del pool (pure) + confine sull'intero pool
    ('''# ═══════════════════════ 2. interfaccia A (verifica e uso) ═══════════════════════
''',
     '''# ═══════════════════════ 1b. pool di batch distinti (25/9) ═══════════════════════
def controlla_pool(n_tr: int, blocco: int) -> int:
    """Numero di blocchi m del pool (1 = pool spento). Rifiuta blocchi che non dividono il pool."""
    if not blocco or blocco == n_tr:
        return 1
    if blocco < 0 or blocco > n_tr or n_tr % blocco:
        raise ValueError(f"THOR_SCORREVOLE_BLOCCO={blocco}: deve dividere le {n_tr} batch di addestramento del pool")
    return n_tr // blocco


def batch_della_finestra(k: int, n_tr: int, n_batch: int, blocco: int) -> "list[int]":
    """Indici (nel pool) delle batch della finestra k: il blocco k mod m, poi TUTTE le held-out
    (n_tr..n_batch-1, le stesse in ogni finestra: i confronti col log storico restano validi).
    Funzione pura di k: la ripresa dalla finestra K sceglie lo stesso blocco della corsa intera."""
    m = controlla_pool(n_tr, blocco)
    if m == 1:
        return list(range(n_batch))
    i = k % m
    return list(range(i * blocco, (i + 1) * blocco)) + list(range(n_tr, n_batch))


def impronta_dati(ids: torch.Tensor) -> str:
    """sha256 degli id del pool (forma + byte): fissa i DATI della corsa nel punto di ripresa."""
    t = ids.detach().cpu().contiguous()
    h = hashlib.sha256(repr((tuple(t.shape), str(t.dtype))).encode())
    h.update(t.numpy().tobytes())
    return h.hexdigest()


def _sotto(d: "dict | None", indici: "list[int]") -> "dict | None":
    return None if d is None else {i: d[b] for i, b in enumerate(indici)}


def _confine_pool(puliti: dict, quant: dict, k0: int, ids_batch: torch.Tensor, cfg, emb: torch.Tensor,
                  h_seme, e_seme, sel: "list[int]", teacher_fin: "dict | None", teacher_tenuto: dict,
                  uscita_t: "dict | None", n_tr: int) -> tuple:
    """Confine a meta' finestra sull'INTERO pool (non solo sulle batch della finestra).
    Uscita del teacher dello strato k0 per batch: catturata (batch della finestra) > tenuta dalla
    finestra precedente (il blocco che l'ha usata) > calcolata qui. Lo studente passa per lo strato
    commesso per TUTTE. Stessa funzione di confine della catena (calcola_confine_a_meta), a gruppi:
    il calcolo e' per batch, il raggruppamento non cambia un bit. Ritorna (h, e, teacher da tenere)."""
    n_batch = ids_batch.shape[0]
    noto: dict = {}
    if teacher_fin is not None:
        noto.update({b: teacher_fin[i] for i, b in enumerate(sel)})
    tenuti = sorted(b for b in teacher_tenuto if b not in noto)
    if tenuti:
        b0 = tenuti[0]       # controllo: teacher tenuto = ricalcolato (una batch, come per il catturato)
        h_rif, _ = V.calcola_confine_finestra({k0: puliti[k0]}, {k0: quant[k0]}, ids_batch[b0:b0 + 1], cfg, emb,
                                              h0_ingresso_per_b=None if h_seme is None else {0: h_seme[b0]},
                                              e0_ingresso_per_b=None)
        d = h_rif[0].float() - teacher_tenuto[b0].float()
        rel = (d.norm() / h_rif[0].float().norm().clamp_min(1e-30)).item()
        if rel > 1e-2:
            raise RuntimeError(f"finestra {k0}: teacher TENUTO della batch {b0} diverso dal ricalcolato (rel {rel:.3e})")
        for b in tenuti:
            noto[b] = teacher_tenuto[b]
    calcolati = [b for b in range(n_batch) if b not in noto]
    h_nuovo, e_nuovo = {}, {}
    for gruppo, con_teacher in ((sorted(noto), True), (calcolati, False)):
        if not gruppo:
            continue
        h_g, e_g = calcola_confine_a_meta(puliti, quant, [k0], ids_batch[gruppo], cfg, emb,
                                          _sotto(h_seme, gruppo), _sotto(e_seme, gruppo),
                                          h_teacher_uscita_per_b=({i: noto[b] for i, b in enumerate(gruppo)}
                                                                  if con_teacher else None))
        for i, b in enumerate(gruppo):
            h_nuovo[b], e_nuovo[b] = h_g[i], e_g[i]
    log(f"  confine sul pool: {n_batch} batch · teacher catturato {len(sel) if teacher_fin is not None else 0}"
        f" · tenuto {len(tenuti)}" + (f" (rel {rel:.1e})" if tenuti else "") + f" · calcolato {len(calcolati)}")
    nuovo_tenuto = ({} if uscita_t is None else
                    {b: uscita_t[i] for i, b in enumerate(sel) if b < n_tr})
    return h_nuovo, e_nuovo, nuovo_tenuto


# ═══════════════════════ 2. interfaccia A (verifica e uso) ═══════════════════════
'''),
    # -- 4. Ripresa: impronta dei dati
    ('''        self.corsa, self.n, self.w = corsa, n_strati, w
        self.ricetta = _ricetta()
''',
     '''        self.corsa, self.n, self.w = corsa, n_strati, w
        self.ricetta = _ricetta()
        self.impronta = None     # sha256 degli id del pool, fissata dopo carica_batch (pool 25/9)
'''),
    ('''        manifest = {"finestra": K, "corsa": self.corsa, "n_strati": self.n, "W": self.w, "n_batch": n_batch,
                    "portati": portati, "ricetta": self.ricetta, "file": file, "quando": time.strftime("%F %T")}''',
     '''        manifest = {"finestra": K, "corsa": self.corsa, "n_strati": self.n, "W": self.w, "n_batch": n_batch,
                    "portati": portati, "ricetta": self.ricetta, "file": file, "quando": time.strftime("%F %T"),
                    "impronta_dati": self.impronta}'''),
    ('''        for nome, info in man["file"].items():
            if _sha256(d / nome) != info["sha256"]:''',
     '''        if self.impronta is not None and man.get("impronta_dati") not in (None, self.impronta):
            raise RuntimeError(f"ripresa RIFIUTATA: dati di calibrazione diversi da quelli della corsa interrotta "
                               f"(impronta {man['impronta_dati'][:12]} -> {self.impronta[:12]})")
        for nome, info in man["file"].items():
            if _sha256(d / nome) != info["sha256"]:'''),
    # -- 5. firma di esegui
    ('''           verifica_seme_a: "list[int] | None" = None, osservatore=None, interrompi=None) -> dict:''',
     '''           verifica_seme_a: "list[int] | None" = None, osservatore=None, interrompi=None,
           blocco: "int | None" = None, riusa_teacher_pool: bool = RIUSA_TEACHER_POOL) -> dict:'''),
    # -- 6. dopo il caricamento delle batch: pool, impronta, ricetta effettiva
    ('''    ids_batch, n_tr = carica_batch()
    n_batch = ids_batch.shape[0]
''',
     '''    ids_batch, n_tr = carica_batch()
    n_batch = ids_batch.shape[0]
    blocco = BLOCCO_POOL if blocco is None else int(blocco)
    m_pool = controlla_pool(n_tr, blocco)
    ripresa.impronta = impronta_dati(ids_batch)
    if m_pool > 1:
        # solo col pool attivo: una ripresa di una corsa SENZA pool (manifesti di prima) resta valida
        ripresa.ricetta["pool_blocco_effettivo"] = str(blocco)
        log(f"POOL: {n_tr} batch di addestramento in {m_pool} blocchi da {blocco} (finestra k -> blocco k mod "
            f"{m_pool}) + {n_batch - n_tr} held-out in ogni finestra · impronta dati {ripresa.impronta[:12]}")
    teacher_tenuto: dict = {}
'''),
    # -- 7. ripresa: il seme deve coprire tutto il pool
    ('''                    h_seme, e_seme, portati = ripresa.carica(K0, emb.device)
''',
     '''                    h_seme, e_seme, portati = ripresa.carica(K0, emb.device)
                    if len(h_seme) != n_batch:
                        raise RuntimeError(f"punto di ripresa con {len(h_seme)} batch, il pool ne ha {n_batch}")
'''),
    # -- 8. verifica del seme anche su una batch di ogni altro blocco
    ('''                _verifica_seme_da_disco(k0, ids_batch, cfg, emb, seme_iniziale, carica_strati,
                                        cartella_out, corsa, h_seme, e_seme)
''',
     '''                _verifica_seme_da_disco(k0, ids_batch, cfg, emb, seme_iniziale, carica_strati,
                                        cartella_out, corsa, h_seme, e_seme)
                for b_v in (range(blocco, n_tr, blocco) if m_pool > 1 else ()):
                    log(f"  (pool) verifica del seme sulla batch {b_v}, prima del blocco {b_v // blocco}:")
                    _verifica_seme_da_disco(k0, ids_batch[b_v:b_v + 1], cfg, emb, seme_iniziale, carica_strati,
                                            cartella_out, corsa, {0: h_seme[b_v]}, {0: e_seme[b_v]})
'''),
    # -- 9. le batch della finestra
    ('''            osserva("finestra_inizio", finestra=k0, strati=layers, h_seme=h_seme, e_seme=e_seme,
                    iniziali=iniziali, entranti=entranti)
''',
     '''            sel = batch_della_finestra(k0, n_tr, n_batch, blocco)
            pool_attivo = len(sel) != n_batch
            ids_fin = ids_batch[sel] if pool_attivo else ids_batch
            h_fin = _sotto(h_seme, sel) if pool_attivo else h_seme
            e_fin = _sotto(e_seme, sel) if pool_attivo else e_seme
            n_tr_fin = blocco if pool_attivo else n_tr
            if pool_attivo:
                log(f"  batch della finestra: blocco {sel[0] // blocco} ({sel[0]}..{sel[n_tr_fin - 1]}) + held-out "
                    f"{sel[n_tr_fin:]}")
            osserva("finestra_inizio", finestra=k0, strati=layers, h_seme=h_seme, e_seme=e_seme,
                    iniziali=iniziali, entranti=entranti, batch=sel)
'''),
    ('''            with (cattura if usa_cattura else contextlib.nullcontext()):
                risultato = ottimizza_fn({L: quant[L] for L in layers}, ids_batch, cfg, n_passi,
                                         strati_teacher={L: puliti[L] for L in layers}, emb=emb, lr=V.NUCLEO_LR,
                                         h0_esterno_per_b=h_seme, e0_esterno_per_b=e_seme,
                                         n_batch_addestramento=n_tr, **kw)
''',
     '''            # il teacher TENUTO e' l'uscita dell'ULTIMO strato della finestra: coincide con lo strato
            # che il confine della finestra dopo attraversa SOLO con W=2 (con W=3 sarebbe lo strato k+1)
            uscita_t = {} if (pool_attivo and riusa_teacher_pool and not ultima and w == 2) else None
            if uscita_t is not None:
                kw["uscita_teacher_fine"] = uscita_t
            with (cattura if usa_cattura else contextlib.nullcontext()):
                risultato = ottimizza_fn({L: quant[L] for L in layers}, ids_fin, cfg, n_passi,
                                         strati_teacher={L: puliti[L] for L in layers}, emb=emb, lr=V.NUCLEO_LR,
                                         h0_esterno_per_b=h_fin, e0_esterno_per_b=e_fin,
                                         n_batch_addestramento=n_tr_fin, **kw)
'''),
    # -- 10. confine: sull'intero pool quando il pool e' attivo
    ('''                teacher_k = cattura.per_batch(n_batch) if usa_cattura else None
                if teacher_k is not None:
                    rel = _controlla_teacher_catturato(puliti[k0], quant[k0], ids_batch, cfg, emb,
                                                       h_seme, teacher_k, k0)
                    log(f"  teacher dello strato {k0} riusato (batch 0 ricalcolato: scarto relativo {rel:.1e})")
                h_seme, e_seme = calcola_confine_a_meta(puliti, quant, [k0], ids_batch, cfg, emb,
                                                        h_seme, e_seme, h_teacher_uscita_per_b=teacher_k)
''',
     '''                teacher_k = cattura.per_batch(len(sel)) if usa_cattura else None
                if teacher_k is not None:
                    rel = _controlla_teacher_catturato(puliti[k0], quant[k0], ids_fin, cfg, emb,
                                                       h_fin, teacher_k, k0)
                    log(f"  teacher dello strato {k0} riusato (batch 0 ricalcolato: scarto relativo {rel:.1e})")
                if pool_attivo:
                    h_seme, e_seme, teacher_tenuto = _confine_pool(
                        puliti, quant, k0, ids_batch, cfg, emb, h_seme, e_seme, sel, teacher_k,
                        teacher_tenuto, uscita_t, n_tr)
                else:
                    h_seme, e_seme = calcola_confine_a_meta(puliti, quant, [k0], ids_batch, cfg, emb,
                                                            h_seme, e_seme, h_teacher_uscita_per_b=teacher_k)
'''),
    ('''            fam = obj = p = None     # (raccordo 24/9) i nomi di ciclo non tengono vivi gli oggetti portati
''',
     '''            fam = obj = p = None     # (raccordo 24/9) i nomi di ciclo non tengono vivi gli oggetti portati
            ids_fin = h_fin = e_fin = uscita_t = None      # (pool 25/9) niente riferimenti ai semi vecchi
            if ultima:
                teacher_tenuto = {}
'''),
    ('''        portati = {}
        h_seme = e_seme = emb = None
''',
     '''        portati = {}
        h_seme = e_seme = emb = None
        teacher_tenuto = {}
'''),
    # -- 11. strumentazione (nessun numero cambia): picco di VRAM per finestra. Il pool aggiunge
    # 64 MiB di catena per batch in piu' e 32 MiB per batch di teacher tenuto: il picco non e' mai
    # stato misurato nei log della forgia, qui si misura a ogni finestra.
    ('''            t0 = time.time()
            layers = list(range(k0, k1))
''',
     '''            t0 = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            layers = list(range(k0, k1))
'''),
    ('''            log(f"── finestra [{k0},{k1-1}] FATTA in {time.time()-t0:.1f}s · commessi {commessi} ──")''',
     '''            log(f"── finestra [{k0},{k1-1}] FATTA in {time.time()-t0:.1f}s · commessi {commessi} ──"
                + (f" · VRAM picco {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB (allocata)"
                   if torch.cuda.is_available() else ""))'''),
]


def applica(testo: str) -> str:
    if MARCATORE in testo:
        raise SystemExit(f"patch gia' applicata ({MARCATORE!r} presente): niente da fare")
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
    ap.add_argument("--applica", action="store_true", help="scrive il file (senza: solo controllo)")
    args = ap.parse_args()
    nuovo = applica(args.file.read_text())
    if not args.applica:
        print(f"prova OK: {len(SOSTITUZIONI)} agganci unici, risultato sintatticamente valido — {args.file} NON toccato")
        return
    tmp = args.file.with_suffix(args.file.suffix + ".tmp_patch_pool")
    tmp.write_text(nuovo)
    os.replace(tmp, args.file)
    print(f"patch applicata: {len(SOSTITUZIONI)} sostituzioni in {args.file}")


if __name__ == "__main__":
    main()

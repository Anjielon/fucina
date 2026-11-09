"""Patch velocita' 2 (24/9 20:0x, mandato forgia < 2 ore) per cattura_schur_esperti.py.

Cache su disco dei pesi presi dal GGUF base per il ramo "quantizzato"
(costruisci_strato(da_gguf=True)): oggi ~27 s/strato, quasi tutti nella
dequantizzazione numpy dei 256 esperti, rifatta identica a OGNI forgia.
Con THOR_CACHE_STRATI=<cartella> la prima volta si calcola e si salva, le
successive si legge (float32: identico al bit al percorso senza cache).
Chiave = percorso + taglia + mtime del GGUF + impronta sha256 del codice di
conversione (carica_strato_gguf.py, dequant_gguf_torch.py, mappa e permutazioni):
una correzione come quella del B9 invalida la cache da sola.
Senza THOR_CACHE_STRATI: comportamento invariato.
"""
from pathlib import Path

P = Path(__file__).resolve().parent.parent / "cattura_schur_esperti.py"
s = P.read_text()
vecchio_uso = "        sd_q = pesi_strato_da_gguf(GGUF, L)\n"
nuovo_uso = "        sd_q = _pesi_gguf_con_cache(GGUF, L)\n"
vecchio_def = "def costruisci_strato(cfg, L: int, da_gguf: bool):\n"
nuova_def = '''def _pesi_gguf_con_cache(gguf: str, L: int) -> dict:
    """pesi_strato_da_gguf con cache su disco (THOR_CACHE_STRATI), identica al bit.

    ⭐ 24/9 20:0x (mandato forgia < 2 ore): la dequantizzazione numpy dei 256
    esperti costava ~27 s/strato a OGNI forgia. La chiave include l'impronta del
    codice di conversione: una correzione (es. B9, A_log) invalida la cache."""
    cartella = os.getenv("THOR_CACHE_STRATI", "")
    if not cartella:
        return pesi_strato_da_gguf(gguf, L)
    import hashlib
    import carica_strato_gguf as _csg
    qui = Path(_csg.__file__).resolve().parent
    impronta = hashlib.sha256()
    for nome in ("carica_strato_gguf.py", "dequant_gguf_torch.py",
                 "mappa_completa_gguf_hf.json", "mappa_completa_gguf_hf_perm.npz"):
        impronta.update((qui / nome).read_bytes())
    st = os.stat(gguf)
    impronta.update(f"|{os.path.realpath(gguf)}|{st.st_size}|{st.st_mtime_ns}".encode())
    f = Path(cartella) / f"{Path(gguf).stem}-{impronta.hexdigest()[:16]}" / f"strato_{L:02d}.safetensors"
    if f.exists():
        return dict(_st_load_file(str(f)))
    sd = pesi_strato_da_gguf(gguf, L)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f.name + ".tmp")
    _st_save_file({k: v.to(torch.float32).contiguous() for k, v in sd.items()}, str(tmp))
    os.replace(tmp, f)                       # atomico: mai un file a meta' letto come buono
    return sd


def costruisci_strato(cfg, L: int, da_gguf: bool):
'''
for vecchio, nuovo in ((vecchio_uso, nuovo_uso), (vecchio_def, nuova_def)):
    n = s.count(vecchio)
    if n != 1:
        raise SystemExit(f"aggancio trovato {n} volte (atteso 1): {vecchio[:80]!r}")
    s = s.replace(vecchio, nuovo)
P.write_text(s)
print(f"patch applicata a {P.name}")

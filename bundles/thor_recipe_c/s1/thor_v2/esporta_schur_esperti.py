#!/usr/bin/env python3
"""Esporta l'output di forgia_schur_esperti.py (Dg/Qg/Du/Qu per 40 strati) in
un GGUF: patcha SOLO `blk.{0..39}.ffn_{gate,up}_exps.weight` (80 tensori),
tutto il resto copiato invariato da Thor-v20-downaggr3.gguf.

⛔ NON e' esporta_finestra.py: quello gestisce le 5 famiglie DENSE della
pipeline SliderQuant (`D_{L}_{nome_flat}.npy` via INVERSA), formato diverso e
convenzione di file diversa. Verificato leggendo il codice (23/9): nessun
riferimento a Dg_/Qg_/Du_/Qu_ o a ffn_*_exps in quel file — la memoria di
sessione che diceva "gia' scritto, verificato" per QUESTO caso era stantia,
scritta prima di controllare. Questo file e' la cura, scritto oggi.

Formato di destinazione: TQ1_0R_1024 (rotazione Hadamard blocco 1024, scala
ogni 128 pesi) — STESSO codec (`gptq_1024_lib.pack_da_d_q`) usato da
esporta_finestra.py per le famiglie dense, applicato qui ai 3 assi (E,out,in)
degli esperti invece che a 2 (out,in).

Fatti verificati PRIMA di scrivere questo file (23/9), nessuno assunto:
- base attuale: `blk.L.ffn_gate_exps.weight`/`ffn_up_exps.weight` sono
  TQ1_0 (enum 34, formato STANDARD llama.cpp, 256 pesi/blocco, 54 B/blocco,
  NON TQ1_0R_1024) — shape (256, 512, 432) uint8. Questa forgia li SOSTITUISCE
  con TQ1_0R_1024 (enum 45), shape finale (256, 512, 448) — dimensione diversa,
  tipo diverso: non e' un patch in-place, e' una riscrittura del tensore.
- NESSUNA permutazione HF->GGUF per questa famiglia: `perm_gguf_da_hf.npz`
  contiene SOLO le 5 famiglie dense (attn_gate/attn_qkv/attn_q/ffn_gate_shexp/
  ffn_up_shexp) — nessuna voce per ffn_gate_exps/ffn_up_exps. Il motivo e'
  strutturale: la permutazione delle attn e' un riordino di TESTE (GQA) fatto
  dal convertitore HF->GGUF; gli esperti MoE non hanno teste, nessun riordino
  e' documentato ne' misurato per questa famiglia. `cattura_schur_esperti.py`
  legge `sr.mlp.experts.gate_up_proj` (HF nativo, PyTorch) e lo stessa
  convenzione [gate;up] fusa (prime 512 righe = gate, seconde 512 = up) e'
  esattamente quella che GGUF usa per `ffn_gate_exps`/`ffn_up_exps` separati.
- Round-trip del packer verificato bit-esatto a livello di BITSTREAM
  (`_unpack_quarter`, lo stesso modulo che ha generato il formato — vedi
  verifica_packer sotto): `gguf.quants.dequantize` non implementa
  TQ1_0R_1024 in questo albero (verificato: NotImplementedError), quindi il
  controllo si fa dove il decoder ESISTE davvero, non dove sarebbe comodo.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getenv("GGUF_PY_DIR", "@@ROOT:models@@/build/llamacpp-tq1-k1024/gguf-py"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernel-rotazione"))
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as T  # noqa: E402
from gptq_1024_lib import QK, GROUP, BLOCK_BYTES, pack_da_d_q  # noqa: E402
from tq1_0r_1024_codec import _unpack_quarter  # noqa: E402

BASE = os.getenv("THOR_SCHUR_BASE", "@@ROOT:models@@/gguf/tony-tern/Thor-v20-downaggr3.gguf")
OUT_DIR = Path(os.getenv("FINESTRA_SCHUR_OUT", "@@ROOT:models@@/gguf/tony-tern/pod-schur-esperti"))
USCITA = sys.argv[1] if len(sys.argv) > 1 else "@@ROOT:models@@/gguf/tony-tern/Thor-v30-schuresperti.gguf"
N_STRATI = int(os.getenv("N_STRATI", "40"))

TIPO_BASE_ATTESO = int(T.TQ1_0)          # 34 — quello che ffn_*_exps ha OGGI
TIPO_NUOVO = int(T.TQ1_0R_1024)          # 45 — quello che scriviamo


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def verifica_completezza() -> None:
    """Guarda le marche che il DRIVER scrive solo dopo un layer completo
    (fatto_forgia_NN.ok) — non i file Dg/Qg/Du/Qu, che possono esistere anche
    a meta' scrittura. Nessun default silenzioso: manca anche una sola marca
    -> RIFIUTO con l'elenco esatto."""
    mancanti = [L for L in range(N_STRATI) if not (OUT_DIR / f"fatto_forgia_{L:02d}.ok").exists()]
    if mancanti:
        raise RuntimeError(
            f"CORSA INCOMPLETA: mancano {len(mancanti)}/{N_STRATI} marche fatto_forgia_NN.ok: "
            f"{mancanti} — esportazione RIFIUTATA")
    log(f"completezza verificata: tutte le {N_STRATI} marche fatto_forgia_NN.ok presenti")


def impacchetta_famiglia(D: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """D: (E,out,ng) float · Q: (E,out,in_feat) int8 {-1,0,1}
    -> (E,out,nb*BLOCK_BYTES) uint8, pronto per add_tensor_info/write_tensor_data.
    Stessa reshape di esporta_finestra.py (2 assi), con un asse E in piu' in
    testa che viaggia invariato per tutta la pipeline (mai mescolato)."""
    E, out_rows, in_feat = Q.shape
    ng = in_feat // GROUP
    nb = in_feat // QK
    assert D.shape == (E, out_rows, ng), (D.shape, (E, out_rows, ng))
    D_flat = D.reshape(E * out_rows, nb, ng // nb).reshape(E * out_rows * nb, ng // nb)
    Q_flat = Q.reshape(E * out_rows, nb, QK).reshape(E * out_rows * nb, QK)
    packed = pack_da_d_q(D_flat.astype(np.float32), Q_flat.astype(np.int8))
    assert packed.shape == (E * out_rows * nb, BLOCK_BYTES)
    return packed.reshape(E, out_rows, nb * BLOCK_BYTES)


def verifica_packer(D: np.ndarray, Q: np.ndarray, nome: str, righe_esperto: int = 256) -> None:
    """Round-trip bit-esatto del PACCHETTO (224 B/blocco), su tutti gli esperti
    passati (default: tutti e 256, non un campione — 'deve essere esatto').

    ⛔ `gguf.quants.dequantize` NON implementa TQ1_0R_1024 in questo albero
    (verificato: NotImplementedError) — quel decoder serve al motore C, non a
    Python. Il round-trip disponibile e corretto e' al livello del BITSTREAM:
    `_unpack_quarter` (stesso modulo che ha generato il formato,
    tq1_0r_1024_codec.py) inverte ESATTAMENTE `_pack_quarter` usato dentro
    `pack_da_d_q` — nessuna rotazione qui (quella e' gia' stata applicata
    PRIMA, dentro forgia_schur_esperti.py, su Wrot/Hrot: i trit Q che
    arrivano qui sono gia' la decisione finale in spazio ruotato). Verifica:
    trit ricostruiti == Q esatti, scala fp16 ricostruita == D arrotondato a
    fp16 esatto — per OGNI quarto, OGNI blocco.
    """
    Dp = D[:righe_esperto]
    Qp = Q[:righe_esperto]
    E, out_rows, in_feat = Qp.shape
    nb = in_feat // QK
    ng = in_feat // GROUP
    D_flat = Dp.reshape(E * out_rows, nb, ng // nb).reshape(E * out_rows * nb, ng // nb)
    Q_flat = Qp.reshape(E * out_rows, nb, QK).reshape(E * out_rows * nb, QK)
    packed = pack_da_d_q(D_flat.astype(np.float32), Q_flat.astype(np.int8))

    Tq = np.clip(Q_flat.astype(np.int64) + 1, 0, 2).reshape(-1, 4, 256)
    for qq in range(4):
        qs = packed[:, qq * 48:(qq + 1) * 48]
        qh = packed[:, 192 + qq * 4:192 + (qq + 1) * 4]
        ric_trit = _unpack_quarter(qs, qh)
        atteso_trit = (Tq[:, qq, :] - 1).astype(np.float32)
        diff = float(np.abs(ric_trit - atteso_trit).max())
        assert diff == 0.0, f"{nome} quarto {qq}: trit round-trip NON bit-esatto: max|Δ| = {diff}"

    d_bytes = packed[:, 208:224]
    ric_d = d_bytes.view(np.float16).astype(np.float32)
    atteso_d = D_flat.astype(np.float16).astype(np.float32)
    diff_d = float(np.abs(ric_d - atteso_d).max())
    assert diff_d == 0.0, f"{nome}: scala fp16 round-trip NON bit-esatta: max|Δ| = {diff_d}"
    log(f"  ✓ {nome}: packer round-trip bit-esatto su {righe_esperto} esperti "
        f"({packed.shape[0]} blocchi, trit+scala) — max|Δ| = 0.0")


def main():
    verifica_completezza()
    src = GGUFReader(BASE)
    tens_by_name = {t.name: t for t in src.tensors}

    bersagli = {}   # nome_gguf -> packed uint8 (E,out,nb*BLOCK_BYTES), gia' pronto — calcolato UNA sola volta
    verificato = False
    for L in range(N_STRATI):
        Dg = np.load(OUT_DIR / f"Dg_{L:02d}.npy")
        Qg = np.load(OUT_DIR / f"Qg_{L:02d}.npy")
        Du = np.load(OUT_DIR / f"Du_{L:02d}.npy")
        Qu = np.load(OUT_DIR / f"Qu_{L:02d}.npy")
        for fam, D, Q in (("ffn_gate_exps", Dg, Qg), ("ffn_up_exps", Du, Qu)):
            nome = f"blk.{L}.{fam}.weight"
            t_base = tens_by_name.get(nome)
            if t_base is None:
                raise RuntimeError(f"{nome}: NON esiste nel GGUF base {BASE} — schema divergente, mi fermo")
            if int(t_base.tensor_type) != TIPO_BASE_ATTESO:
                raise RuntimeError(
                    f"{nome}: tipo base {int(t_base.tensor_type)} != atteso {TIPO_BASE_ATTESO} (TQ1_0) — "
                    f"verifica manuale richiesta prima di sovrascrivere, mi fermo")
            if not verificato:
                verifica_packer(D, Q, nome)
                verificato = True
            # ⛔ impacchettato UNA volta e tenuto in RAM: chiamare impacchetta_famiglia
            # due volte (metadata + scrittura) su dati puramente numpy sarebbe
            # deterministico, ma tenere i byte gia' scritti evita ogni dubbio —
            # cio' che viene dimensionato nell'header e' BYTE PER BYTE cio' che
            # viene scritto dopo.
            bersagli[nome] = impacchetta_famiglia(D, Q)
        log(f"  strato {L}: impacchettato (gate+up)")
    log(f"tensori da patchare: {len(bersagli)} (attesi {N_STRATI*2})")
    assert len(bersagli) == N_STRATI * 2

    w = GGUFWriter(path=None, arch="qwen35moe")
    for fld in src.fields.values():
        if fld.name.startswith(("GGUF.", "split.")):
            continue
        try:
            w.add_key_value(fld.name, fld.contents(), fld.types[0], sub_type=fld.types[-1] if len(fld.types) > 1 else None)
        except Exception:
            pass
    w.add_string("thor.ricetta_schur_esperti",
                 f"Thor-v20-downaggr3 + SchurQuant(teacher-reconstruction, lambda=0.3) + rotazione FWHT "
                 f"blocco 1024 + coordinate descent GPTQ a blocchi (kernel Triton fuso) su "
                 f"{len(bersagli)} tensori ffn_gate_exps/ffn_up_exps (tutti i {N_STRATI} strati, "
                 f"256 esperti ciascuno), formato TQ1_0R_1024. ffn_down_exps NON toccato "
                 f"(ingresso 512 < blocco rotazione 1024).")

    piano = []
    for t in src.tensors:
        if t.name in bersagli:
            piano.append((t.name, "patch", t))
        else:
            piano.append((t.name, "copia", t))

    for nome, azione, t in piano:
        if azione == "patch":
            packed = bersagli[nome]
            w.add_tensor_info(nome, list(packed.shape), np.dtype(np.uint8), int(packed.nbytes), raw_dtype=T.TQ1_0R_1024)
        else:
            raw = np.asarray(t.data)
            w.add_tensor_info(nome, list(raw.shape), raw.dtype, int(raw.nbytes), raw_dtype=t.tensor_type)

    w.open_output_file(Path(USCITA))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    t0 = time.time()
    fatti = 0
    for nome, azione, t in piano:
        if azione == "patch":
            w.write_tensor_data(bersagli[nome])
            fatti += 1
            if fatti % 20 == 0:
                log(f"  [{fatti}/{len(bersagli)}] patchati")
        else:
            w.write_tensor_data(np.asarray(t.data))
    w.close()
    log(f"FATTO in {time.time()-t0:.0f}s: {USCITA} ({Path(USCITA).stat().st_size/2**30:.2f} GiB) — "
        f"{len(bersagli)} tensori patchati (TQ1_0 -> TQ1_0R_1024), "
        f"{len(piano)-len(bersagli)} copiati invariati")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Collaudo di fondi_densi_v5.py su GGUF SINTETICI da pochi KB (24/9/2026).

Perche' esiste: la fusione dei densi v5 (leva #2, prova A3) scrive un GGUF da
~12,8 GB copiando tensori byte per byte fra due file; un errore di offset, di
allineamento o di sorgente produce un modello che si carica e calcola un'altra
cosa. Qui si verifica la logica senza scrivere niente di grande:

- fusione base: i tensori scelti arrivano dal donatore (tipo, forma, byte),
  tutti gli altri e tutti i KV dal bersaglio; confronto INDIPENDENTE dal
  verificatore dello script (GGUFReader + np.array_equal + sha256 ricalcolati);
- allineamento non standard (64), esperti inclusi/esclusi, filtri, elenco;
- rifiuti: uscita esistente, fuori radice, relativa; forma diversa; tensore
  mancante; iperparametri diversi; bersaglio gia' fuso; scelta vuota — e in
  nessun caso restano file scritti;
- il verificatore SA FALLIRE: un byte cambiato in un tensore sostituito, in uno
  non sostituito, in un KV, nella spaziatura, o un donatore diverso da quello
  registrato vengono tutti rilevati (un controllo che non puo' fallire non e'
  un controllo);
- la taglia prevista da --solo-elenco coincide con quella scritta; l'uscita e'
  deterministica (due fusioni = stessi byte); la CLI rifiuta /tmp.

Uso: python3 test_fondi_densi_v5.py [--cartella DIR]   (default: cartella temporanea)
Esce 0 se tutto passa, 1 altrimenti.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fondi_densi_v5 as fd  # noqa: E402
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as T  # noqa: E402
from gguf.quants import quant_shape_to_byte_shape  # noqa: E402

ARCH = "qwen35moe"
# (nome, tipo nel bersaglio, tipo nel donatore, forma numpy logica)
TENSORI = [
    ("token_embd.weight", T.Q8_0, T.Q5_K, (16, 256)),
    ("blk.0.attn_norm.weight", T.F32, T.F32, (256,)),
    ("blk.0.ssm_a", T.F32, T.F32, (3,)),                       # 12 byte: forza una spaziatura
    ("blk.0.attn_gate.weight", T.TQ1_0R_1024, T.Q8_0, (8, 1024)),
    ("blk.0.ffn_gate_shexp.weight", T.TQ1_0R_1024, T.Q6_K, (4, 1024)),
    ("blk.0.ffn_up_shexp.weight", T.TQ1_0R_1024, T.Q4_K, (4, 1024)),
    ("blk.0.ffn_gate_exps.weight", T.TQ1_0R_1024, T.Q8_0, (2, 4, 1024)),
    ("blk.0.ffn_down_exps.weight", T.TQ1_0, T.Q4_K, (2, 64, 256)),
    ("blk.1.attn_q.weight", T.TQ1_0R_1024, T.Q4_K, (4, 1024)),
    ("blk.1.attn_qkv.weight", T.Q8_0, T.Q8_0, (4, 1024)),
    ("blk.2.attn_gate.weight", T.TQ1_0R_1024, T.Q8_0, (2, 1024)),   # blocco MTP (ultimo)
    ("output.weight", T.Q6_K, T.Q6_K, (16, 256)),
]
SCELTI_DEFAULT = {"blk.0.attn_gate.weight", "blk.0.ffn_gate_shexp.weight", "blk.0.ffn_up_shexp.weight",
                  "blk.1.attn_q.weight", "blk.2.attn_gate.weight"}


def costruisci(percorso, *, donatore=False, seme=0, allineamento=None, block_count=3, forme=None,
               togli=(), kv_extra=None, ordine_inverso=False):
    """GGUF sintetico: KV di vari tipi (anche array di stringhe) e tensori a byte casuali."""
    w = GGUFWriter(percorso, ARCH)
    if allineamento:
        w.add_custom_alignment(allineamento)
    w.add_string("general.name", "Donatore" if donatore else "Bersaglio")
    w.add_uint32(f"{ARCH}.block_count", block_count)
    w.add_uint32(f"{ARCH}.nextn_predict_layers", 1)
    w.add_float32(f"{ARCH}.rope.freq_base", 1e7)
    w.add_array("tokenizer.ggml.tokens", ["<a>", "bb", "ccc", "è"])
    w.add_array("tokenizer.ggml.token_type", [1, 1, 3, 1])
    w.add_bool("tokenizer.ggml.add_bos_token", False)
    for k, v in (kv_extra or {}).items():
        w.add_string(k, v)
    rng = np.random.default_rng(seme)
    elenco = list(reversed(TENSORI)) if ordine_inverso else TENSORI
    for nome, tb, td, forma in elenco:
        if nome in togli:
            continue
        tipo = td if donatore else tb
        forma = (forme or {}).get(nome, forma)
        if tipo == T.F32:
            w.add_tensor(nome, rng.standard_normal(forma).astype(np.float32))
        else:
            bshape = quant_shape_to_byte_shape(forma, tipo)
            w.add_tensor(nome, rng.integers(0, 256, bshape, dtype=np.uint8), raw_dtype=tipo)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return percorso


def muto(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return f(*a, **k)


class Banco:
    def __init__(self, cartella):
        self.c = cartella
        self.b = costruisci(os.path.join(cartella, "bersaglio.gguf"), seme=1)
        self.d = costruisci(os.path.join(cartella, "donatore.gguf"), donatore=True, seme=2, ordine_inverso=True)

    def p(self, nome):
        return os.path.join(self.c, nome)

    def file(self):
        return sorted(os.listdir(self.c))


def controlla_fusione(u, b, d, scelti):
    """Controllo indipendente dal verificatore dello script."""
    ru, rb, rd = GGUFReader(u), GGUFReader(b), GGUFReader(d)
    tb, tdn = {t.name: t for t in rb.tensors}, {t.name: t for t in rd.tensors}
    assert [t.name for t in ru.tensors] == [t.name for t in rb.tensors], "ordine dei tensori"
    for t in ru.tensors:
        src = tdn[t.name] if t.name in scelti else tb[t.name]
        assert t.tensor_type == src.tensor_type, t.name
        assert list(t.shape) == list(src.shape), t.name
        assert np.array_equal(np.asarray(t.data), np.asarray(src.data)), t.name
        assert (t.data_offset - ru.data_offset) % ru.alignment == 0, t.name
    for n, f in rb.fields.items():
        if not n.startswith("GGUF."):
            assert ru.fields[n].contents() == f.contents(), n
    assert int(ru.fields["GGUF.kv_count"].contents()) == int(rb.fields["GGUF.kv_count"].contents()) + 1
    doc = json.loads(ru.fields[fd.CHIAVE].contents())
    assert {s["nome"] for s in doc["tensori"]} == set(scelti), "elenco nella chiave"
    for s in doc["tensori"]:
        assert s["sha256_dopo"] == hashlib.sha256(np.asarray(tdn[s["nome"]].data).tobytes()).hexdigest()
        assert s["sha256_prima"] == hashlib.sha256(np.asarray(tb[s["nome"]].data).tobytes()).hexdigest()
        assert s["tipo_prima"] == tb[s["nome"]].tensor_type.name and s["tipo_dopo"] == tdn[s["nome"]].tensor_type.name
    assert doc["byte_delta"] == sum(int(tdn[n].n_bytes) - int(tb[n].n_bytes) for n in scelti)
    return ru, doc


def t_fusione_base(bc):
    u = bc.p("fuso.gguf")
    piano = fd.pianifica(bc.b, bc.d)
    prevista = muto(fd.stampa_piano, piano)["taglia_file_prevista"]
    res = muto(fd.fondi, bc.b, bc.d, u, radici=(bc.c,))
    ru, doc = controlla_fusione(u, bc.b, bc.d, SCELTI_DEFAULT)
    assert ru.fields["general.name"].contents() == "Bersaglio"
    assert os.path.getsize(u) == res["taglia"] == prevista, (os.path.getsize(u), res["taglia"], prevista)
    assert any("MTP" in n for n in res["note"]), res["note"]
    assert not os.path.exists(u + ".parziale")
    esp = {t.name: t for t in ru.tensors}["blk.0.ffn_gate_exps.weight"]
    assert esp.tensor_type == T.TQ1_0R_1024, "gli esperti non vanno toccati di default"
    return f"{len(doc['tensori'])} sostituiti, taglia {os.path.getsize(u)} = prevista"


def t_deterministico(bc):
    u1, u2 = bc.p("det1.gguf"), bc.p("det2.gguf")
    muto(fd.fondi, bc.b, bc.d, u1, radici=(bc.c,))
    muto(fd.fondi, bc.b, bc.d, u2, radici=(bc.c,))
    h = [hashlib.sha256(open(x, "rb").read()).hexdigest() for x in (u1, u2)]
    assert h[0] == h[1]
    return "due fusioni = stessi byte"


def t_allineamento_64(bc):
    b = costruisci(bc.p("b64.gguf"), seme=3, allineamento=64)
    d = costruisci(bc.p("d64.gguf"), donatore=True, seme=4, allineamento=64)
    u = bc.p("f64.gguf")
    muto(fd.fondi, b, d, u, radici=(bc.c,))
    ru, _ = controlla_fusione(u, b, d, SCELTI_DEFAULT)
    assert ru.alignment == 64
    return "allineamento 64 rispettato"


def t_criteri(bc):
    casi = [
        ({"includi_esperti": True}, SCELTI_DEFAULT | {"blk.0.ffn_gate_exps.weight"}),
        ({"filtro": r"attn_gate"}, {"blk.0.attn_gate.weight", "blk.2.attn_gate.weight"}),
        ({"escludi": r"^blk\.2\."}, SCELTI_DEFAULT - {"blk.2.attn_gate.weight"}),
        ({"tipi": ("TQ1_0",), "includi_esperti": True}, {"blk.0.ffn_down_exps.weight"}),
    ]
    el = bc.p("elenco.txt")
    with open(el, "w") as f:
        f.write("# commento\nblk.1.attn_q.weight\nblk.1.attn_qkv.weight  # stesso tipo nei due file\n")
    casi.append(({"elenco": el}, {"blk.1.attn_q.weight", "blk.1.attn_qkv.weight"}))
    for i, (crit, attesi) in enumerate(casi):
        u = bc.p(f"crit{i}.gguf")
        res = muto(fd.fondi, bc.b, bc.d, u, radici=(bc.c,), **crit)
        controlla_fusione(u, bc.b, bc.d, attesi)
        if "elenco" in crit:
            assert any("stesso tipo" in n for n in res["note"]), res["note"]
    return f"{len(casi)} criteri di scelta"


def t_rifiuti(bc):
    esistente = bc.p("esiste.gguf")
    open(esistente, "wb").close()
    d_forma = costruisci(bc.p("d_forma.gguf"), donatore=True, seme=5, forme={"blk.0.attn_gate.weight": (8, 2048)})
    d_manca = costruisci(bc.p("d_manca.gguf"), donatore=True, seme=6, togli=("blk.1.attn_q.weight",))
    d_arch = costruisci(bc.p("d_arch.gguf"), donatore=True, seme=7, block_count=4)
    b_fuso = costruisci(bc.p("b_fuso.gguf"), seme=8, kv_extra={fd.CHIAVE: "{}"})
    prima = bc.file()
    casi = [
        ("uscita esistente", dict(b=bc.b, d=bc.d, u=esistente, radici=(bc.c,))),
        ("fuori radice", dict(b=bc.b, d=bc.d, u=bc.p("r.gguf"), radici=fd.RADICI_USCITA)),
        ("relativa", dict(b=bc.b, d=bc.d, u="r.gguf", radici=(bc.c,))),
        ("non .gguf", dict(b=bc.b, d=bc.d, u=bc.p("r.bin"), radici=(bc.c,))),
        ("forma diversa", dict(b=bc.b, d=d_forma, u=bc.p("r.gguf"), radici=(bc.c,))),
        ("tensore mancante", dict(b=bc.b, d=d_manca, u=bc.p("r.gguf"), radici=(bc.c,))),
        ("iperparametri diversi", dict(b=bc.b, d=d_arch, u=bc.p("r.gguf"), radici=(bc.c,))),
        ("bersaglio gia' fuso", dict(b=b_fuso, d=bc.d, u=bc.p("r.gguf"), radici=(bc.c,))),
        ("scelta vuota", dict(b=bc.b, d=bc.d, u=bc.p("r.gguf"), radici=(bc.c,), filtro="nessuno")),
    ]
    for nome, k in casi:
        extra = {x: k[x] for x in ("filtro",) if x in k}
        try:
            muto(fd.fondi, k["b"], k["d"], k["u"], radici=k["radici"], **extra)
        except fd.ErroreFusione:
            pass
        else:
            raise AssertionError(f"non ha rifiutato: {nome}")
        assert bc.file() == prima, f"{nome}: file lasciati {set(bc.file()) - set(prima)}"
        assert os.path.getsize(esistente) == 0
    # rifondere un file gia' fuso
    u = bc.p("prima_fusione.gguf")
    muto(fd.fondi, bc.b, bc.d, u, radici=(bc.c,))
    try:
        muto(fd.fondi, u, bc.d, bc.p("seconda.gguf"), radici=(bc.c,))
        raise AssertionError("ha rifuso un file gia' fuso")
    except fd.ErroreFusione:
        pass
    assert not os.path.exists(bc.p("seconda.gguf"))
    return f"{len(casi) + 1} rifiuti, nessun file lasciato"


def _altera(src, dst, offset, nuovo=None):
    shutil.copyfile(src, dst)
    with open(dst, "r+b") as f:
        f.seek(offset)
        b = f.read(1)
        f.seek(offset)
        f.write(bytes([nuovo if nuovo is not None else (b[0] ^ 0xFF)]))


def t_verifica_sa_fallire(bc):
    u = bc.p("buono.gguf")
    muto(fd.fondi, bc.b, bc.d, u, radici=(bc.c,))
    fd.verifica(u, bc.b, bc.d)                       # il buono passa
    ru = GGUFReader(u)
    tu = {t.name: t for t in ru.tensors}
    kv = ru.fields["general.name"]
    fine_kv = kv.offset + sum(int(p.nbytes) for p in kv.parts) - 1
    ssm = tu["blk.0.ssm_a"]
    pad = ssm.data_offset + int(ssm.n_bytes)         # 12 byte di dati, poi spaziatura
    assert (pad - ru.data_offset) % ru.alignment != 0
    alterazioni = {
        "byte in un tensore sostituito": tu["blk.1.attn_q.weight"].data_offset + 7,
        "byte in un tensore NON sostituito": tu["blk.0.attn_norm.weight"].data_offset + 3,
        "byte in un KV del bersaglio": fine_kv,
        "byte nella spaziatura": pad,
    }
    for nome, off in alterazioni.items():
        dst = bc.p("alterato.gguf")
        _altera(u, dst, off, 1 if nome == "byte nella spaziatura" else None)
        try:
            fd.verifica(dst, bc.b, bc.d)
        except (fd.ErroreFusione, ValueError, UnicodeDecodeError):
            pass
        else:
            raise AssertionError(f"verifica cieca: {nome}")
        os.unlink(dst)
    d2 = costruisci(bc.p("donatore2.gguf"), donatore=True, seme=2, ordine_inverso=True,
                    kv_extra={"general.description": "altro"})
    try:
        fd.verifica(u, bc.b, d2)
        raise AssertionError("verifica cieca: donatore diverso da quello registrato")
    except fd.ErroreFusione:
        pass
    return f"{len(alterazioni) + 1} alterazioni tutte rilevate"


def t_cli(bc):
    prima = bc.file()
    assert muto(fd.main, ["--bersaglio", bc.b, "--donatore", bc.d, "--solo-elenco", "--dettaglio"]) == 0
    assert bc.file() == prima, "--solo-elenco ha scritto file"
    assert muto(fd.main, ["--bersaglio", bc.b, "--donatore", bc.d, "--uscita", bc.p("cli.gguf")]) == 2
    assert muto(fd.main, ["--bersaglio", bc.b, "--donatore", bc.d, "--solo-elenco",
                          "--uscita", bc.p("cli.gguf")]) == 2
    assert bc.file() == prima, "la CLI ha scritto sotto una radice non ammessa"
    u = bc.p("per_cli.gguf")
    muto(fd.fondi, bc.b, bc.d, u, radici=(bc.c,))
    assert muto(fd.main, ["--solo-verifica", "--uscita", u]) == 0
    return "solo-elenco non scrive; /tmp rifiutato; solo-verifica legge i percorsi dalla chiave"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cartella")
    a = ap.parse_args()
    base = tempfile.mkdtemp(prefix="fondi_densi_", dir=a.cartella)   # solo roba di questo collaudo
    esiti = []
    for t in (t_fusione_base, t_deterministico, t_allineamento_64, t_criteri, t_rifiuti,
              t_verifica_sa_fallire, t_cli):
        c = tempfile.mkdtemp(prefix=t.__name__ + "_", dir=base)
        try:
            esiti.append((t.__name__, True, t(Banco(c))))
        except Exception as e:  # noqa: BLE001 — il collaudo riporta ogni guasto
            esiti.append((t.__name__, False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))
    tot = 0
    for n, ok, msg in esiti:
        print(f"{'OK  ' if ok else 'FAIL'} {n}: {msg}")
        tot += ok
    dim = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(base) for f in fs)
    print(f"{tot}/{len(esiti)} passati — file sintetici in {base}, {dim} byte in tutto")
    return 0 if tot == len(esiti) else 1


if __name__ == "__main__":
    sys.exit(main())

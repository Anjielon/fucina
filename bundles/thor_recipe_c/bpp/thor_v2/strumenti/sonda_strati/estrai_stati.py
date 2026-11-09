#!/usr/bin/env python3
"""SONDA LINEARE PER STRATO — fase 2 (GPU): stati residui per strato sull'ultimo token del prompt (28/9/2026).

## Cosa fa
Per UN modello (GGUF), per ogni domanda di prepara_domande.py: l_out-0 .. l_out-(L-1) (uscita di ogni blocco, flusso
residuo PRIMA della norma finale) alla posizione dell'ULTIMO token del prompt («Answer:» → il token da cui il modello
sceglie la lettera), piu' i logit delle 4 lettere e il logsumexp sul vocabolario intero (accuratezza d'uscita del
modello sulle stesse domande: l'ancora della sonda all'ultimo strato).

## Con quale strumento — quello che c'e', non modificato
`e2e_dump_logit logit --uscite ultima --l-out tutti` (collaudato: e2e_test_dump_logit.py T0-T12; stesso grafo di
llama-perplexity, sequenze isolate con memoria azzerata, rifiuto rumoroso di ogni incoerenza).
Cosa sa fare davvero (letto nel sorgente, e2e_dump_logit.cpp e src/models/qwen35moe.cpp della build k1024):
  - scrive i tensori del grafo PER INTERO (tutte le righe del micro-batch): con --uscite ultima i logit sono 1 riga,
    ma gli l_out di TUTTI gli strati hanno n_token righe (in questa build il taglio inp_out_ids avviene dopo la norma
    finale, salvo embeddings_nextn_masked); il manifesto lo dice per tensore ("righe_uguali_a": token | uscite);
  - non sa scrivere «solo l'ultima riga». Per non toccare uno strumento collaudato (che la catena usa) qui si lavora a
    LOTTI: un'invocazione per lotto di domande (un caricamento del modello), si estrae l'ultima riga di ogni l_out, si
    controlla, si salva il lotto (npz atomico) e si CANCELLA il dump grezzo del lotto. Il disco transitorio e' ~un lotto
    (lotto 128 x ~110 token x 40 strati x 2048 x 4 B ≈ 4,6 GB in f32 [S]).
Ripresa: un lotto con il suo .npz completo non si rifa'; un dump grezzo avanzato da un'interruzione si cancella.

## Controlli (ogni lotto): esito "ok" del manifesto; stessi id (fnv1a64) di domande.json per ogni sequenza; strati
0..L-1 tutti presenti, nessun buco; n_embd coerente; valori finiti; righe dei tensori = n_token o n_uscite (1) e
riga scelta di conseguenza; logit: una riga alla posizione n_token-1.

## Uscite ($SONDA_DIR/stati/<nome>/)
  lotti/l_NNNN.npz   stati (n, L, d) float32 · lettere (n, 4) float32 · lse (n,) float64 · argmax_vocab (n,) · k (n,)
  stati.npy          (N, L, d) float32, nell'ordine di domande.json (scritto quando tutti i lotti ci sono)
  lettere.npy        (N, 4) float32 logit delle lettere A-D · lse.npy (N,) float64 · argmax.npy (N,) int32
  meta.json          modello (percorso, byte, mtime, sha256), binario, argv, tempi per lotto, controlli. PER ULTIMO.

Uso (sulla GPU libera: vedi lancia_sonda_strati.sh, che mette le guardie):
  python3 estrai_stati.py --modello M.gguf --nome maestro --vulkan [--lotto 128] [--strati-attesi 40]
Per il collaudo su CPU (modello minuscolo): senza --vulkan.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

QUI = Path(__file__).resolve().parent
THOR_V2 = QUI.parents[1]
sys.path.insert(0, str(THOR_V2))
from e2e_dump_logit_leggi import Uscita  # noqa: E402

SONDA_DIR = Path(os.getenv("SONDA_DIR", "@@ROOT:depot@@/sonda_strati"))
BIN = Path(os.getenv("SONDA_DUMP_BIN", str(THOR_V2 / "e2e_dump_logit")))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def fnv1a64(b: bytes) -> str:
    h = 1469598103934665603
    for x in b:
        h ^= x
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def riga_ultimo_token(info: dict, n_token: int) -> int:
    """Quale riga di un tensore del manifesto e' l'ULTIMO token del prompt."""
    if info["righe_uguali_a"] == "token":
        if info["righe"] != n_token:
            raise RuntimeError(f"{info['file']}: {info['righe']} righe, attese {n_token} (token)")
        return n_token - 1
    if info["righe_uguali_a"] == "uscite":
        if info["righe"] != 1:
            raise RuntimeError(f"{info['file']}: {info['righe']} righe d'uscita, attesa 1 (--uscite ultima)")
        return 0
    raise RuntimeError(f"{info['file']}: righe_uguali_a {info['righe_uguali_a']!r} sconosciuto")


def estrai_lotto(u: Uscita, domande: list, id_lettere: list, strati_attesi: int | None) -> dict:
    """Dal dump grezzo di un lotto: ultima riga di ogni l_out, logit delle lettere. Tutti i controlli qui."""
    n = u.n_sequenze
    if n != len(domande):
        raise RuntimeError(f"{n} sequenze nel manifesto, {len(domande)} domande nel lotto")
    n_embd = int(u.man["modello"]["n_embd"])
    stati, lettere, lse, amax = [], [], [], []
    n_strati = None
    for j, d in enumerate(domande):
        s = u.man["sequenze"][j]
        if s["n_token"] != d["n_token"] or s["fnv1a64_ids"] != d["fnv1a64_ids"]:
            raise RuntimeError(f"sequenza {j}: id diversi da domande.json (k={d['k']})")
        nomi = sorted((t for t in s["tensori"] if t.startswith("l_out-")), key=lambda t: int(t.split("-")[1]))
        idx = [int(t.split("-")[1]) for t in nomi]
        if idx != list(range(len(idx))):
            raise RuntimeError(f"sequenza {j}: strati l_out non contigui da 0: {idx[:5]}...")
        if n_strati is None:
            n_strati = len(idx)
            if strati_attesi is not None and n_strati != strati_attesi:
                raise RuntimeError(f"{n_strati} strati l_out, attesi {strati_attesi}")
        elif len(idx) != n_strati:
            raise RuntimeError(f"sequenza {j}: {len(idx)} strati, le altre {n_strati}")
        h = np.empty((n_strati, n_embd), dtype=np.float32)
        for li, nome in enumerate(nomi):
            info = s["tensori"][nome]
            r = riga_ultimo_token(info, d["n_token"])
            t = u.tensore(j, nome)
            if t.shape[-1] != n_embd:
                raise RuntimeError(f"{nome}: larghezza {t.shape[-1]} != n_embd {n_embd}")
            h[li] = np.asarray(t[r], dtype=np.float32)
        if not np.isfinite(h).all():
            raise RuntimeError(f"sequenza {j}: stati non finiti")
        pos, lg = u.logit(j, mmap=True)
        if list(pos) != [d["n_token"] - 1]:
            raise RuntimeError(f"sequenza {j}: logit alle posizioni {list(pos)[:3]}, attesa {d['n_token'] - 1}")
        riga = np.asarray(lg[0], dtype=np.float64)
        if not np.isfinite(riga).all():
            raise RuntimeError(f"sequenza {j}: logit non finiti")
        m = riga.max()
        stati.append(h)
        lettere.append(riga[id_lettere].astype(np.float32))
        lse.append(m + np.log(np.exp(riga - m).sum()))
        amax.append(int(riga.argmax()))
    return {"stati": np.stack(stati), "lettere": np.stack(lettere), "lse": np.asarray(lse, dtype=np.float64),
            "argmax_vocab": np.asarray(amax, dtype=np.int32), "k": np.asarray([d["k"] for d in domande]),
            "n_strati": n_strati, "n_embd": n_embd}


def main(argv: list | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--modello", required=True, type=Path)
    ap.add_argument("--nome", required=True)
    ap.add_argument("--domande", type=Path, default=SONDA_DIR / "domande")
    ap.add_argument("--uscita", type=Path, default=None, help="default $SONDA_DIR/stati/<nome>")
    ap.add_argument("--lavoro", type=Path, default=None, help="dump grezzi transitori (default <uscita>/grezzo)")
    ap.add_argument("--lotto", type=int, default=128)
    ap.add_argument("--thread", type=int, default=4)
    ap.add_argument("--ub", type=int, default=512, help="n_ubatch (512 come llama-perplexity; accanto a un training "
                                                        "sulla GPU mai di piu')")
    ap.add_argument("--vulkan", action="store_true")
    ap.add_argument("--strati-attesi", type=int, default=None)
    ap.add_argument("--max-domande", type=int, default=0, help="solo le prime N (prova di fumo)")
    ap.add_argument("--senza-sha256", action="store_true", help="non calcola lo sha256 del GGUF (solo collaudo)")
    ap.add_argument("--tieni-grezzo", action="store_true")
    a = ap.parse_args(argv)

    man_d = json.loads((a.domande / "domande.json").read_text(encoding="utf-8"))
    domande = man_d["domande"]
    if a.max_domande:
        domande = domande[:a.max_domande]
    id_lettere = [man_d["id_lettere"][L] for L in "ABCD"]
    out = a.uscita or (SONDA_DIR / "stati" / a.nome)
    lav = a.lavoro or (out / "grezzo")
    (out / "lotti").mkdir(parents=True, exist_ok=True)
    if not a.modello.is_file():
        raise FileNotFoundError(a.modello)
    st = a.modello.stat()
    lotti = [domande[i:i + a.lotto] for i in range(0, len(domande), a.lotto)]
    log(f"{a.nome}: {a.modello} ({st.st_size} B) · {len(domande)} domande in {len(lotti)} lotti da {a.lotto} · "
        f"{'Vulkan' if a.vulkan else 'solo CPU'} · uscita {out}")
    tempi = []
    for b, lotto in enumerate(lotti):
        dest = out / "lotti" / f"l_{b:04d}.npz"
        if dest.exists():
            z = np.load(dest)
            if list(z["k"]) == [d["k"] for d in lotto]:
                continue
            raise RuntimeError(f"{dest}: lotto esistente con domande diverse (lotto cambiato fra due corse?)")
        grezzo = lav / f"l_{b:04d}"
        shutil.rmtree(grezzo, ignore_errors=True)
        grezzo.parent.mkdir(parents=True, exist_ok=True)
        cmd = [str(BIN), "logit", "-m", str(a.modello), "-o", str(grezzo), "-t", str(a.thread), "-ub", str(a.ub),
               "--uscite", "ultima", "--l-out", "tutti"]
        if a.vulkan:
            cmd.append("--vulkan")
        cmd += [str(a.domande / d["file"]) for d in lotto]
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.time() - t0
        if r.returncode != 0:
            (out / f"errore_l_{b:04d}.log").write_text(r.stderr)
            raise RuntimeError(f"lotto {b}: e2e_dump_logit rc={r.returncode} (stderr in {out}/errore_l_{b:04d}.log)")
        u = Uscita(grezzo)
        e = estrai_lotto(u, lotto, id_lettere, a.strati_attesi)
        sec_seq = sum(s["secondi"] for s in u.man["sequenze"])
        tmp = dest.with_name(dest.name + ".tmp.npz")
        np.savez(tmp, stati=e["stati"], lettere=e["lettere"], lse=e["lse"], argmax_vocab=e["argmax_vocab"], k=e["k"])
        os.replace(tmp, dest)
        if not a.tieni_grezzo:
            shutil.rmtree(grezzo)
        ntok = sum(d["n_token"] for d in lotto)
        tempi.append({"lotto": b, "domande": len(lotto), "token": ntok, "secondi_totali": round(dt, 2),
                      "secondi_sequenze": round(sec_seq, 2), "caricamento_e_resto": round(dt - sec_seq, 2),
                      "backend": u.man.get("backend")})
        fatti = b + 1
        log(f"lotto {b}/{len(lotti) - 1}: {len(lotto)} domande, {ntok} token, {dt:.1f} s (sequenze {sec_seq:.1f} s, "
            f"{ntok / max(sec_seq, 1e-9):.0f} tok/s) · stati {e['stati'].shape} · ETA "
            f"{(len(lotti) - fatti) * dt / 60:.1f} min")
    # ── assemblaggio ──
    z0 = np.load(out / "lotti" / "l_0000.npz")
    L_, D_ = z0["stati"].shape[1:]
    N = len(domande)
    st_tmp = out / "stati.npy.tmp"
    S = np.lib.format.open_memmap(st_tmp, mode="w+", dtype=np.float32, shape=(N, L_, D_))
    lett, lse, amax, ks = [], [], [], []
    i = 0
    for b in range(len(lotti)):
        z = np.load(out / "lotti" / f"l_{b:04d}.npz")
        n = z["stati"].shape[0]
        S[i:i + n] = z["stati"]
        lett.append(z["lettere"])
        lse.append(z["lse"])
        amax.append(z["argmax_vocab"])
        ks.append(z["k"])
        i += n
    S.flush()
    del S
    ks = np.concatenate(ks)
    if list(ks) != [d["k"] for d in domande]:
        raise RuntimeError("assemblaggio: ordine delle domande diverso da domande.json")
    os.replace(st_tmp, out / "stati.npy")
    np.save(out / "lettere.npy", np.concatenate(lett))
    np.save(out / "lse.npy", np.concatenate(lse))
    np.save(out / "argmax.npy", np.concatenate(amax))
    lettere = np.concatenate(lett)
    y = np.asarray([d["risposta"] for d in domande])
    acc = float((lettere.argmax(1) == y).mean())
    vecchio = json.loads((out / "meta.json").read_text()) if (out / "meta.json").exists() else {}
    meta = {
        "creato": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "nome": a.nome, "modello": {"percorso": str(a.modello), "byte": st.st_size, "mtime": st.st_mtime,
                                    "sha256": None if a.senza_sha256 else sha256_file(a.modello)},
        "binario": {"percorso": str(BIN), "sha256": sha256_file(BIN)},
        "domande": {"cartella": str(a.domande), "n": N, "creato": man_d["creato"],
                    "sha256_domande_json": sha256_file(a.domande / "domande.json"),
                    "fnv1a64_ids": [d["fnv1a64_ids"] for d in domande]},
        "forma_stati": [N, int(L_), int(D_)], "tensori": "l_out-0..l_out-(L-1), riga dell'ultimo token del prompt",
        "vulkan": a.vulkan, "lotto": a.lotto, "ub": a.ub, "thread": a.thread,
        "accuratezza_uscita_4_lettere": round(acc, 4),
        "tempi": vecchio.get("tempi", []) + tempi,
    }
    (out / "meta.json.tmp").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    os.replace(out / "meta.json.tmp", out / "meta.json")
    if not a.tieni_grezzo:
        shutil.rmtree(lav, ignore_errors=True)
    log(f"FATTO {a.nome}: stati {N}x{L_}x{D_} · accuratezza d'uscita sulle 4 lettere {acc * 100:.2f} %")


if __name__ == "__main__":
    main()

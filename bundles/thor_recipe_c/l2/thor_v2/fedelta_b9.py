"""Audit B9 (24/9/2026) — la simulazione torch della v20 e' FEDELE a llama.cpp?

Perche' conta: l'ottimizzazione globale addestra i codici ternari contro l'errore
che il ramo torch "quantizzato" vede (attenzione/SSM/esperti dequantizzati dal
GGUF v20, caricati da `costruisci_strato(da_gguf=True)`). Se quel ramo non
calcola cio' che llama.cpp calcola davvero, gli esperti imparano a compensare un
errore che in produzione non esiste. Indizio che ha aperto la voce: nel ramo
torch ||e0|| = 89,6 / 207,4 contro ||h|| = 66,4 / 68,0 dopo 4 strati, mentre la
v20 in llama.cpp fa MMLU 76,5.

Tre fasi, separate perche' usano risorse diverse:
  token  (CPU)  scrive gli id della prima sequenza held-out (1024 token) in
                token.i32 — gli STESSI id per le due parti, niente tokenizzazione
                doppia;
  llama  (CPU)  strumenti/dump_strati sulla v20: inp_embd + l_out-0..3 per intero;
  torch  (GPU)  ramo pulito (bf16 HF) e ramo quantizzato (GGUF) sugli strati 0-3,
                stesso posizionale e maschera di calcola_confine_finestra, poi il
                confronto per token.

Metriche, per strato e per token t (norme sull'asse hidden):
  fedelta'   r_f(t) = ||h_torchQ - h_llama|| / ||h_llama||
  errore vero   r_v(t) = ||h_llama - h_pulito|| / ||h_pulito||   (quantizzazione reale)
  errore torch  r_s(t) = ||h_torchQ - h_pulito|| / ||h_pulito||  (quantizzazione simulata)
Verdetto: fedele se r_f << r_v (mediana e p99) e r_s ~ r_v.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

QUI = Path(__file__).resolve().parent
CARTELLA = Path(os.getenv("THOR_B9_DIR", "@@ROOT:models@@/gguf/tony-tern/fedelta_b9"))
STRATI = [int(x) for x in os.getenv("THOR_B9_STRATI", "0,1,2,3").split(",")]
N_THREAD = int(os.getenv("THOR_B9_THREAD", "8"))


def fase_token() -> None:
    sys.path.insert(0, str(QUI))
    import valida_finestra_8_12 as V
    ids, n_tr = V.carica_batch_calibrazione()
    seq = ids[n_tr, 0].numpy().astype(np.int32)          # prima sequenza held-out
    CARTELLA.mkdir(parents=True, exist_ok=True)
    seq.tofile(CARTELLA / "token.i32")
    np.save(CARTELLA / "token.npy", seq)
    print(f"token: {seq.size} id, primi {seq[:8].tolist()} -> {CARTELLA/'token.i32'}")


def fase_llama() -> None:
    sys.path.insert(0, str(QUI))
    import valida_finestra_8_12 as V
    nomi = ["model.input_embed"] + [f"l_out-{L}" for L in STRATI]
    cmd = ["nice", "-n", "10", str(QUI / "strumenti" / "dump_strati"), str(V.GGUF_BASE),
           str(CARTELLA / "token.i32"), str(CARTELLA), str(N_THREAD), *nomi]
    print("esegue:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _leggi_f32(nome: str) -> np.ndarray:
    raw = np.fromfile(CARTELLA / f"{nome}.f32", dtype=np.uint8)
    ne = np.frombuffer(raw[:32].tobytes(), dtype=np.int64)
    dati = np.frombuffer(raw[32:].tobytes(), dtype=np.float32)
    assert dati.size == int(np.prod(ne)), f"{nome}: {dati.size} valori per forma {ne.tolist()}"
    # ggml: ne0 = hidden (asse veloce), ne1 = token -> (token, hidden)
    return dati.reshape(int(ne[1]), int(ne[0]))


def _statistiche(r: np.ndarray) -> dict:
    return {"mediana": float(np.median(r)), "p90": float(np.percentile(r, 90)),
            "p99": float(np.percentile(r, 99)), "max": float(r.max()),
            "argmax": int(r.argmax()), "token0": float(r[0])}


def fase_torch() -> dict:
    import torch
    sys.path.insert(0, str(QUI))
    import valida_finestra_8_12 as V
    from transformers.masking_utils import create_causal_mask
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding

    seq = torch.from_numpy(np.load(CARTELLA / "token.npy").astype(np.int64))
    ids = seq.view(1, 1, -1)
    cfg, emb, puliti, quantizzati = V.carica_finestra_corrente(STRATI)
    h0, e0 = V.seme_embedding_quantizzato(ids, emb)
    dev = emb.device
    l_seq = seq.numel()

    hs_dummy = torch.zeros(1, l_seq, cfg.hidden_size, device=dev, dtype=emb.dtype)
    rot = Qwen3_5MoeTextRotaryEmbedding(cfg).to(dev)
    pos = torch.arange(l_seq, device=dev).unsqueeze(0)
    pos_emb = rot(hs_dummy, pos)
    maschera = create_causal_mask(config=cfg, inputs_embeds=hs_dummy, attention_mask=None,
                                   past_key_values=None, position_ids=pos)

    def _fwd(strato, h):
        o = strato(hidden_states=h, position_embeddings=pos_emb, attention_mask=maschera, position_ids=pos)
        return o[0] if isinstance(o, tuple) else o

    esito = {"gguf": str(V.GGUF_BASE), "token": int(l_seq), "strati": {}}
    with torch.no_grad():
        h_t = h0[0]
        h_q = h_t + e0[0].to(h_t.dtype)            # stesso seme dell'ottimizzatore (B12)
        # 0) embedding: le righe Q5_K torch contro inp_embd di llama.cpp
        emb_ll = _leggi_f32("model.input_embed")
        emb_tq = (h0[0].float() + e0[0]).reshape(l_seq, -1).cpu().numpy()
        r_emb = np.linalg.norm(emb_tq - emb_ll, axis=1) / np.linalg.norm(emb_ll, axis=1)
        esito["embedding"] = _statistiche(r_emb)
        print(f"embedding torch(Q5K) vs llama: {esito['embedding']}", flush=True)
        for L in STRATI:
            h_t = _fwd(puliti[L], h_t)
            h_q = _fwd(quantizzati[L], h_q)
            ll = _leggi_f32(f"l_out-{L}")
            tq = h_q.float().reshape(l_seq, -1).cpu().numpy()
            pu = h_t.float().reshape(l_seq, -1).cpu().numpy()
            n_ll = np.linalg.norm(ll, axis=1)
            n_pu = np.linalg.norm(pu, axis=1)
            r_f = np.linalg.norm(tq - ll, axis=1) / n_ll
            r_v = np.linalg.norm(ll - pu, axis=1) / n_pu
            r_s = np.linalg.norm(tq - pu, axis=1) / n_pu
            senza0 = slice(1, None)
            glob = lambda a, b, s=slice(None): float(np.linalg.norm(a[s] - b[s]) / np.linalg.norm(b[s]))
            esito["strati"][L] = {
                "fedelta_torch_vs_llama": _statistiche(r_f),
                "errore_vero_llama_vs_pulito": _statistiche(r_v),
                "errore_simulato_torch_vs_pulito": _statistiche(r_s),
                "globale": {"fedelta": glob(tq, ll), "fedelta_senza_t0": glob(tq, ll, senza0),
                            "vero": glob(ll, pu), "vero_senza_t0": glob(ll, pu, senza0),
                            "simulato": glob(tq, pu), "simulato_senza_t0": glob(tq, pu, senza0)},
                "norme": {"pulito_t0": float(n_pu[0]), "pulito_mediana": float(np.median(n_pu)),
                          "llama_t0": float(n_ll[0]), "llama_mediana": float(np.median(n_ll))},
            }
            s = esito["strati"][L]
            print(f"strato {L}: fedelta' mediana {s['fedelta_torch_vs_llama']['mediana']:.4e} "
                  f"p99 {s['fedelta_torch_vs_llama']['p99']:.4e} · errore vero mediana "
                  f"{s['errore_vero_llama_vs_pulito']['mediana']:.4e} · simulato "
                  f"{s['errore_simulato_torch_vs_pulito']['mediana']:.4e} · globale {s['globale']}", flush=True)
    (CARTELLA / "esito_b9.json").write_text(json.dumps(esito, indent=2))
    return esito


if __name__ == "__main__":
    fase = sys.argv[1] if len(sys.argv) > 1 else ""
    {"token": fase_token, "llama": fase_llama, "torch": fase_torch}[fase]()

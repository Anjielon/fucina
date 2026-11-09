"""Copertura degli esperti per dominio (24/9 22:3x, dopo il pannello A1).

Il pannello ha mostrato che le ricette migliorano solo il dominio di calibrazione (math/code
chat) e restano PEGGIO della v20 sul testo generale (wiki 1,70x, edu 1,43x). Ipotesi: gli
esperti che il testo generale usa ricevono pochi token (quindi poco gradiente) dalla
calibrazione e restano vicini all'init. Qui si misura: router VERO degli strati della finestra
sul flusso del teacher, top-8, conteggio dei token per esperto per ciascun dominio.

Uscite: per strato, esperti con meno di N token di calibrazione; quota dei token wiki/edu che
cade su esperti poco coperti; sovrapposizione dei 64 esperti piu' usati fra domini."""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import valida_finestra_8_12 as V  # noqa: E402

STRATI = [int(x) for x in os.getenv("THOR_COPERTURA_STRATI", "0,1").split(",")]
pan = torch.load("@@ROOT:models@@/gguf/tony-tern/pannello/pannello_ids.pt")["domini"]
ids, n_tr = V.carica_batch_calibrazione()
domini = {"calibrazione": ids[:n_tr], "wiki": pan["wiki"], "edu": pan["edu"], "math_code_pannello": pan["math_code"]}
cfg, emb, puliti, _q = V.carica_finestra_corrente(STRATI)
from transformers.masking_utils import create_causal_mask  # noqa: E402
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding  # noqa: E402

E = cfg.num_experts
conti = {d: {L: torch.zeros(E, dtype=torch.long) for L in STRATI} for d in domini}
with torch.no_grad():
    for d, t in domini.items():
        for b in range(t.shape[0]):
            x = t[b].to("cuda")
            h = torch.nn.functional.embedding(x, emb).to(torch.bfloat16)
            rot = Qwen3_5MoeTextRotaryEmbedding(cfg).to("cuda")
            pos = torch.arange(h.shape[1], device="cuda").unsqueeze(0).expand(h.shape[0], -1)
            pe = rot(h, pos)
            mask = create_causal_mask(config=cfg, inputs_embeds=h, attention_mask=None,
                                      past_key_values=None, position_ids=pos)
            for L in STRATI:
                preso = {}
                gancio = puliti[L].post_attention_layernorm.register_forward_hook(
                    lambda _m, _i, o: preso.__setitem__("v", o.detach()))
                o = puliti[L](hidden_states=h, position_embeddings=pe, attention_mask=mask, position_ids=pos)
                gancio.remove()
                h = o[0] if isinstance(o, tuple) else o
                _, _, sel = puliti[L].mlp.gate(preso["v"].reshape(-1, preso["v"].shape[-1]))
                conti[d][L] += torch.bincount(sel.reshape(-1).cpu(), minlength=E)

rapporto = {}
for L in STRATI:
    cal = conti["calibrazione"][L].float()
    righe = {}
    for soglia in (0, 100, 1000):
        poco = cal <= soglia
        righe[f"esperti_con_calib_<=_{soglia}"] = int(poco.sum())
        for d in ("wiki", "edu", "math_code_pannello"):
            tot = conti[d][L].float()
            righe[f"quota_{d}_su_esperti_calib_<=_{soglia}"] = float(tot[poco].sum() / tot.sum())
    top = {d: set(torch.topk(conti[d][L], 64).indices.tolist()) for d in domini}
    righe["sovrapposizione_top64_calib_wiki"] = len(top["calibrazione"] & top["wiki"])
    righe["sovrapposizione_top64_calib_edu"] = len(top["calibrazione"] & top["edu"])
    righe["token_calibrazione_min_mediana_max"] = [int(cal.min()), int(cal.median()), int(cal.max())]
    rapporto[L] = righe
    print(f"strato {L}: " + json.dumps(righe, ensure_ascii=False))
os.makedirs("@@ROOT:forge@@/stato/pannello", exist_ok=True)
torch.save(conti, "@@ROOT:forge@@/stato/pannello/copertura_esperti_conti.pt")
json.dump(rapporto, open("@@ROOT:forge@@/stato/pannello/copertura_esperti.json", "w"), indent=1)

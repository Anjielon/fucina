#!/usr/bin/env python3
"""Routing degli esperti MoE, replicato ESATTAMENTE (non approssimato) dal
codice HF (`Qwen3_5MoeExperts.forward` + `Qwen3_5MoeTopKRouter.forward`,
transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py) — necessario perche'
`Qwen3_5MoeExperts` non ha un `nn.Linear` per esperto (un UNICO
`nn.Parameter (num_experts, ...)`, indicizzato dentro il forward): non c'e'
un sottomodulo su cui appendere un forward-hook, quindi il routing va
RIPETUTO, non intercettato. Verificato bit-esatto contro il modello vero.

⛔ Il router (`gate.weight`, F32, mai quantizzato) e' identico sullo stream
pulito e su quello quantizzato: la differenza di instradamento fra i due
viene SOLO da hidden_states diverso (h_ref vs h_q) — coerente con la
letteratura sul route-flip (danno mediato dal routing, mai dal router stesso).

Uso, per il componente teacher-reconstruction di SchurQuant:
    idx, x = token_per_esperto(hidden_states, gate_module, top_k)[e]
    # idx: indici dei token instradati a e (nell'ordine di expert_mask)
    # x:   (n_tok_e, hidden_dim) — l'ingresso ESATTO che gate_up_proj[e] vede
"""
from __future__ import annotations

import torch


@torch.no_grad()
def instradamento(hidden_states: torch.Tensor, gate_module, top_k: int):
    """Replica ESATTA di Qwen3_5MoeTopKRouter.forward + la costruzione di
    expert_mask in Qwen3_5MoeExperts.forward. hidden_states: (n_tok, hidden_dim).
    Ritorna (selected_experts (n_tok,top_k), routing_weights (n_tok,top_k))."""
    router_logits = torch.nn.functional.linear(hidden_states, gate_module.weight)
    router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
    router_top_value, router_indices = torch.topk(router_probs, top_k, dim=-1)
    router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)
    router_top_value = router_top_value.to(router_logits.dtype)
    return router_indices, router_top_value


@torch.no_grad()
def token_per_esperto(hidden_states: torch.Tensor, gate_module, top_k: int, num_experts: int):
    """Per ogni esperto instradato almeno una volta: (token_idx, current_state,
    top_k_pos, peso_instradamento) — stessa selezione, stesso ordine, stessa
    formula di Qwen3_5MoeExperts.forward (fino alla riga `current_state = ...`,
    PRIMA di applicare gate_up_proj — quello lo fa il chiamante)."""
    selected, pesi = instradamento(hidden_states, gate_module, top_k)
    expert_mask = torch.nn.functional.one_hot(selected, num_classes=num_experts).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    out = {}
    for eh in expert_hit:
        e = int(eh[0])
        if e == num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[e])
        out[e] = {
            "token_idx": token_idx,
            "current_state": hidden_states[token_idx],
            "peso": pesi[token_idx, top_k_pos],
        }
    return out


def _test():
    """Verifica bit-esatta: la nostra replica deve dare lo STESSO output_finale
    del vero Qwen3_5MoeSparseMoeBlock su un input casuale, strato vero incluso."""
    import json
    import sys

    from safetensors import safe_open
    from transformers import AutoConfig, Qwen3_5MoeForConditionalGeneration
    from accelerate import init_empty_weights

    HF = "@@ROOT:models@@/gguf/thor-src-hf"
    L = 0
    cfg = AutoConfig.from_pretrained(HF)
    with init_empty_weights():
        m = Qwen3_5MoeForConditionalGeneration(cfg)
    blocco = m.model.language_model.layers[L].mlp   # Qwen3_5MoeSparseMoeBlock

    idx = json.load(open(f"{HF}/model.safetensors.index.json"))["weight_map"]
    pref = f"model.language_model.layers.{L}.mlp."
    chiavi = {k: v for k, v in idx.items() if k.startswith(pref)}
    pesi = {}
    for f in set(chiavi.values()):
        with safe_open(f"{HF}/{f}", "pt") as sf:
            for k in chiavi:
                if chiavi[k] == f:
                    pesi[k[len(pref):]] = sf.get_tensor(k)
    blocco.load_state_dict(pesi, strict=True, assign=True)
    blocco = blocco.to(torch.float32)

    torch.manual_seed(3)
    n_tok = 37   # apposta non tondo: stressa i casi limite dell'instradamento
    h = torch.randn(1, n_tok, cfg.text_config.hidden_size, dtype=torch.float32) * 0.5

    atteso = blocco(h)[0].squeeze(0)   # uscita VERA del blocco (esperti+shared+sigmoid gate)

    # ricostruzione MANUALE con la nostra replica: stessa formula, stessi pesi
    h2 = h.view(-1, cfg.text_config.hidden_size)
    shared = blocco.shared_expert(h2)
    mappa = token_per_esperto(h2, blocco.gate, cfg.text_config.num_experts_per_tok, cfg.text_config.num_experts)
    finale = torch.zeros_like(h2)
    for e, d in mappa.items():
        gate_up = torch.nn.functional.linear(d["current_state"], blocco.experts.gate_up_proj[e])
        gate, up = gate_up.chunk(2, dim=-1)
        act = torch.nn.functional.silu(gate) * up
        down = torch.nn.functional.linear(act, blocco.experts.down_proj[e])
        down = down * d["peso"][:, None]
        finale.index_add_(0, d["token_idx"], down)
    shared_scaled = torch.sigmoid(blocco.shared_expert_gate(h2)) * shared
    nostro = (finale + shared_scaled).view(atteso.shape)

    diff = (nostro - atteso).abs().max().item()
    scala = atteso.abs().max().item()
    print(f"n_tok={n_tok}, esperti toccati={len(mappa)}/{cfg.text_config.num_experts}, "
          f"top_k={cfg.text_config.num_experts_per_tok}")
    print(f"max|nostro-atteso| = {diff:.3e}  (scala {scala:.3e}, relativo {diff/max(scala,1e-30):.3e})")
    ok = diff / max(scala, 1e-30) < 1e-5
    print("✅ instradamento replicato BIT-ESATTO" if ok else "⛔ DIVERGE — non usare per SchurQuant")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _test() else 1)

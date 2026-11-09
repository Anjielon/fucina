"""Implementazione degli esperti MoE per la forgia Thor (24/9 20:4x, radiografia a 8 seq).

Copia FEDELE di `transformers.integrations.moe.grouped_mm_experts_forward` con DUE sole
differenze, entrambe nel gradiente degli spostamenti di righe: la raccolta dei token per
esperto e la permutazione finale. L'originale fa `hidden[perm // top_k]`:
ogni token compare top_k volte, e il gradiente di un'indicizzazione con indici duplicati
passa da un accumulo con ordinamento (indexing_backward_kernel): ~7% del passo a 8 seq.
Qui `_RaccogliRighe` ha lo stesso avanti e un indietro esatto e diretto: il gradiente si
riporta all'ordine originale con la permutazione INVERSA (un gather) e le top_k copie di
ogni token si sommano (in fp32). Stessa matematica; l'unica differenza numerica e' che la
somma delle top_k copie avviene in fp32 invece che nel dtype del tensore.

Registrazione: `registra()` aggiunge "grouped_mm_thor" al registro di transformers; si
attiva con `cfg._experts_implementation = "grouped_mm_thor"`. `_grouped_linear` e' preso
dal modulo di transformers a ogni chiamata: una sostituzione di `_grouped_mm` (kernel
Triton) vale anche qui. Verifica: test_esperti_thor.py.
"""
from __future__ import annotations

import torch

NOME = "grouped_mm_thor"


class _RaccogliRighe(torch.autograd.Function):
    """y = h[perm // k]  (h: (T, H); perm: permutazione di T*k)  con indietro diretto."""

    @staticmethod
    def forward(ctx, h: torch.Tensor, perm: torch.Tensor, k: int) -> torch.Tensor:
        ctx.save_for_backward(perm)
        ctx.k = k
        ctx.T = h.shape[0]
        return h[perm // k]

    @staticmethod
    def backward(ctx, g: torch.Tensor):
        (perm,) = ctx.saved_tensors
        inversa = torch.empty_like(perm)
        inversa[perm] = torch.arange(perm.numel(), device=perm.device)
        g_ordine_originale = g[inversa]                                   # riga t*k+j al suo posto
        dh = g_ordine_originale.view(ctx.T, ctx.k, -1).float().sum(1).to(g.dtype)
        return dh, None, None


class _PermutaRighe(torch.autograd.Function):
    """y = x[idx] con idx PERMUTAZIONE: l'indietro e' g[idx_inversa] (un gather), non un
    accumulo con ordinamento come per un'indicizzazione generica."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, idx: torch.Tensor, idx_inversa: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(idx_inversa)
        return x[idx]

    @staticmethod
    def backward(ctx, g: torch.Tensor):
        (idx_inversa,) = ctx.saved_tensors
        return g[idx_inversa], None, None


def grouped_mm_experts_forward_thor(self: torch.nn.Module, hidden_states: torch.Tensor,
                                    top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
    from transformers.integrations import moe as _moe

    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    hidden_dim = hidden_states.size(-1)

    sample_weights = top_k_weights.reshape(-1)
    expert_ids = top_k_index.reshape(-1)

    expert_ids_g, perm = torch.sort(expert_ids)
    selected_hidden_states_g = _RaccogliRighe.apply(hidden_states, perm, num_top_k)   # ⭐ l'unica differenza
    sample_weights_g = sample_weights[perm]

    histc_input = expert_ids_g.float() if device.type in ("cpu", "mps") else expert_ids_g.int()
    tokens_per_expert = torch.histc(histc_input, bins=self.num_experts, min=0, max=self.num_experts - 1)
    offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)

    sentinel_mask = (expert_ids_g >= self.num_experts).unsqueeze(-1)
    expert_ids_g.clamp_(max=self.num_experts - 1)

    if self.has_gate:
        selected_weights = self.gate_up_proj
        selected_biases = self.gate_up_proj_bias[expert_ids_g] if self.has_bias else None
    else:
        selected_weights = self.up_proj
        selected_biases = self.up_proj_bias[expert_ids_g] if self.has_bias else None

    selected_hidden_states_g.masked_fill_(sentinel_mask, 0.0)

    proj_out = _moe._grouped_linear(selected_hidden_states_g, selected_weights, offsets,
                                    bias=selected_biases, is_transposed=self.is_transposed)
    if self.has_gate:
        proj_out = self._apply_gate(proj_out)
    else:
        proj_out = self.act_fn(proj_out)

    selected_weights = self.down_proj
    selected_biases = self.down_proj_bias[expert_ids_g] if self.has_bias else None
    proj_out = _moe._grouped_linear(proj_out, selected_weights, offsets,
                                    bias=selected_biases, is_transposed=self.is_transposed)

    weighted_out = proj_out * sample_weights_g.unsqueeze(-1)
    weighted_out.masked_fill_(sentinel_mask, 0.0)

    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.size(0), device=device)
    weighted_out = _PermutaRighe.apply(weighted_out, inv_perm, perm)          # ⭐ indietro = gather

    final_hidden_states = weighted_out.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)
    return final_hidden_states.to(hidden_states.dtype)


def registra() -> None:
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
    ALL_EXPERTS_FUNCTIONS.register(NOME, grouped_mm_experts_forward_thor)

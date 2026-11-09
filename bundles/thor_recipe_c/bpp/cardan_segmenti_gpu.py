"""cardan_segmenti_gpu.py — miscelatore di token del modo «segmenti» con le PROIEZIONI in un solo lancio.

Originale (e2e_studente.Studente._miscela_segmenti): per OGNI segmento (riga paper + ~32 domande KL4 nel passo fuso)
chiama l'intero modulo linear_attn / self_attn sul pezzo: tutte le proiezioni (in_proj_qkv/z/a/b, out_proj; q/k/v/o)
sono lanciate ~33 volte per strato e per verso, su poche decine/centinaia di token.

Qui (variante «lotto"): le proiezioni lineari, le norme per token, il cancello e la rotazione (operazioni PER TOKEN,
indipendenti dal segmento) girano su TUTTI i token della riga insieme; solo cio' che mescola i token resta per
segmento, con le STESSE funzioni del modulo di transformers (M.causal_conv1d_fn e M.torch_chunk_gated_delta_rule,
cosi' come le hanno installate E2E_CONV_DIRETTA / E2E_GDN; l'interfaccia d'attenzione di config._attn_implementation
con la stessa maschera per lunghezza di _contesto). Matematica identica; i bit possono differire solo se un kernel
sceglie un'altra suddivisione per un numero diverso di righe: si MISURA (profila_strati_gpu.py).

Variante «riempito» (giro 7): come «lotto» per le proiezioni; in piu' la conv causale e la regola GDN NON girano piu'
segmento per segmento ma su un LOTTO di segmenti con lo stesso numero di chunk N = ceil(L/64): ogni segmento e'
completato con ZERI fino a 64*N (dopo la conv, con torch.where: +0 esatto), g = beta = 0 nel completamento. E' esattamente
il completamento che la regola fa gia' al suo interno (F.pad a multiplo di 64 PRIMA di l2norm -> 0): stessi chunk, stessi
valori per ogni segmento; cambia solo la dimensione di lotto delle stesse operazioni (il ciclo della ricorrenza va da
somma_i N_i iterazioni a max N). Causale: il completamento non tocca le posizioni valide, e il gradiente delle uscite
scartate e' 0 esatto. L'attenzione piena resta come «lotto».

Attivazione nel trainer: FORGIA_SEGMENTI_LOTTO=1 (lotto) o FORGIA_SEGMENTI_LOTTO=riempito (vedi cardan_congiunto_veloce_gpu.py).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

_ORIGINALE = None


def _gdn_lotto(m, M, h, pezzi):
    """= Qwen3_5MoeGatedDeltaNet.forward (cache None, maschera None) su UNA riga h (1, T, D) fatta di segmenti."""
    _, T, _ = h.shape
    qkv_tutti = m.in_proj_qkv(h)                         # (1, T, conv_dim)
    z = m.in_proj_z(h).reshape(1, T, -1, m.head_v_dim)
    b = m.in_proj_b(h)
    a = m.in_proj_a(h)
    beta_t = b.sigmoid()
    g_t = -m.A_log.float().exp() * F.softplus(a.float() + m.dt_bias)
    rip = m.num_v_heads // m.num_k_heads
    uscite = []
    for s0, s1 in pezzi:
        L = s1 - s0
        mixed = qkv_tutti[:, s0:s1].transpose(1, 2)
        mixed = M.causal_conv1d_fn(mixed, m.conv1d.weight.squeeze(1), m.conv1d.bias, activation=m.activation)
        mixed = mixed.transpose(1, 2)
        q, k, v = torch.split(mixed, [m.key_dim, m.key_dim, m.value_dim], dim=-1)
        q = q.reshape(1, L, -1, m.head_k_dim)
        k = k.reshape(1, L, -1, m.head_k_dim)
        v = v.reshape(1, L, -1, m.head_v_dim)
        if rip > 1:
            q = q.repeat_interleave(rip, dim=2)
            k = k.repeat_interleave(rip, dim=2)
        o, _ = M.torch_chunk_gated_delta_rule(q, k, v, g=g_t[:, s0:s1], beta=beta_t[:, s0:s1], initial_state=None,
                                              output_final_state=False, use_qk_l2norm_in_kernel=True, cu_seqlens=None)
        uscite.append(o)
    core = torch.cat(uscite, dim=1) if len(uscite) > 1 else uscite[0]
    core = m.norm(core.reshape(-1, m.head_v_dim), z.reshape(-1, m.head_v_dim)).reshape(1, T, -1)
    return m.out_proj(core)


def _gdn_riempito(m, M, h, pezzi, C=64):
    """= _gdn_lotto, ma conv + regola GDN su lotti di segmenti con lo stesso numero di chunk (vedi docstring)."""
    _, T, _ = h.shape
    qkv_tutti = m.in_proj_qkv(h)                         # (1, T, conv_dim)
    z = m.in_proj_z(h).reshape(1, T, -1, m.head_v_dim)
    b = m.in_proj_b(h)
    a = m.in_proj_a(h)
    beta_t = b.sigmoid()
    g_t = -m.A_log.float().exp() * F.softplus(a.float() + m.dt_bias)
    rip = m.num_v_heads // m.num_k_heads
    gruppi = {}
    for i, (s0, s1) in enumerate(pezzi):
        gruppi.setdefault(-(-(s1 - s0) // C), []).append(i)
    uscite = [None] * len(pezzi)
    for N, idx in sorted(gruppi.items()):
        P = N * C
        lun = [pezzi[i][1] - pezzi[i][0] for i in idx]
        def riempi(x):                                    # (1, T, ...) -> (n, P, ...) con zeri in coda
            return torch.stack([F.pad(x[0, pezzi[i][0]:pezzi[i][1]], (0, 0) * (x.dim() - 2) + (0, P - l))
                                for i, l in zip(idx, lun)])
        valido = (torch.arange(P, device=h.device).unsqueeze(0) <
                  torch.tensor(lun, device=h.device).unsqueeze(1))          # (n, P)
        mixed = riempi(qkv_tutti).transpose(1, 2)
        mixed = M.causal_conv1d_fn(mixed, m.conv1d.weight.squeeze(1), m.conv1d.bias, activation=m.activation)
        mixed = torch.where(valido.unsqueeze(-1), mixed.transpose(1, 2), torch.zeros((), dtype=mixed.dtype,
                                                                                    device=mixed.device))
        n = len(idx)
        q, k, v = torch.split(mixed, [m.key_dim, m.key_dim, m.value_dim], dim=-1)
        q = q.reshape(n, P, -1, m.head_k_dim)
        k = k.reshape(n, P, -1, m.head_k_dim)
        v = v.reshape(n, P, -1, m.head_v_dim)
        if rip > 1:
            q = q.repeat_interleave(rip, dim=2)
            k = k.repeat_interleave(rip, dim=2)
        o, _ = M.torch_chunk_gated_delta_rule(q, k, v, g=riempi(g_t), beta=riempi(beta_t), initial_state=None,
                                              output_final_state=False, use_qk_l2norm_in_kernel=True, cu_seqlens=None)
        for r, (i, l) in enumerate(zip(idx, lun)):
            uscite[i] = o[r:r + 1, :l]
    core = torch.cat(uscite, dim=1) if len(uscite) > 1 else uscite[0]
    core = m.norm(core.reshape(-1, m.head_v_dim), z.reshape(-1, m.head_v_dim)).reshape(1, T, -1)
    return m.out_proj(core)


def _attn_lotto(m, M, h, pezzi, per_lunghezza):
    """= Qwen3_5MoeAttention.forward (senza cache) su UNA riga h (1, T, D) fatta di segmenti."""
    _, T, _ = h.shape
    shp = (1, T, -1, m.head_dim)
    q, gate = torch.chunk(m.q_proj(h).view(1, T, -1, m.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(1, T, -1)
    q = m.q_norm(q.view(shp)).transpose(1, 2)
    k = m.k_norm(m.k_proj(h).view(shp)).transpose(1, 2)
    v = m.v_proj(h).view(shp).transpose(1, 2)
    cos = torch.cat([per_lunghezza[s1 - s0][0][0] for s0, s1 in pezzi], dim=1)
    sin = torch.cat([per_lunghezza[s1 - s0][0][1] for s0, s1 in pezzi], dim=1)
    q, k = M.apply_rotary_pos_emb(q, k, cos, sin)
    interf = M.ALL_ATTENTION_FUNCTIONS.get_interface(m.config._attn_implementation, M.eager_attention_forward)
    uscite = []
    for s0, s1 in pezzi:
        o, _ = interf(m, q[:, :, s0:s1], k[:, :, s0:s1], v[:, :, s0:s1], per_lunghezza[s1 - s0][1],
                      dropout=0.0 if not m.training else m.attention_dropout, scaling=m.scaling)
        uscite.append(o)                                 # (1, L, H, hd)
    o = (torch.cat(uscite, dim=1) if len(uscite) > 1 else uscite[0]).reshape(1, T, -1).contiguous()
    o = o * torch.sigmoid(gate)
    return m.o_proj(o)


def miscela_segmenti_riempito(self, modulo, tipo: str, h, ctx):
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M
    righe = []
    for b in range(ctx["B"]):
        pezzi = [(s0, s1) for (bb, s0, s1) in ctx["segmenti"] if bb == b]
        hb = h[b:b + 1]
        if tipo == "linear_attention":
            righe.append(_gdn_riempito(modulo.linear_attn, M, hb, pezzi))
        else:
            righe.append(_attn_lotto(modulo.self_attn, M, hb, pezzi, ctx["per_lunghezza"]))
    return torch.cat(righe, dim=0) if len(righe) > 1 else righe[0]


def miscela_segmenti_lotto(self, modulo, tipo: str, h, ctx):
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M
    righe = []
    for b in range(ctx["B"]):
        pezzi = [(s0, s1) for (bb, s0, s1) in ctx["segmenti"] if bb == b]
        hb = h[b:b + 1]
        if tipo == "linear_attention":
            righe.append(_gdn_lotto(modulo.linear_attn, M, hb, pezzi))
        else:
            righe.append(_attn_lotto(modulo.self_attn, M, hb, pezzi, ctx["per_lunghezza"]))
    return torch.cat(righe, dim=0) if len(righe) > 1 else righe[0]


def imposta(S, variante: str) -> None:
    """S = modulo e2e_studente. «originale» = metodo di sempre; «lotto» = proiezioni per riga."""
    global _ORIGINALE
    if _ORIGINALE is None:
        _ORIGINALE = S.Studente._miscela_segmenti
    if variante == "originale":
        S.Studente._miscela_segmenti = _ORIGINALE
    elif variante == "lotto":
        S.Studente._miscela_segmenti = miscela_segmenti_lotto
    elif variante == "riempito":
        S.Studente._miscela_segmenti = miscela_segmenti_riempito
    else:
        raise ValueError(f"variante sconosciuta: {variante}")

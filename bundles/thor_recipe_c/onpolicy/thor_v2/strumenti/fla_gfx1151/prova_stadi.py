#!/usr/bin/env python3
"""Quale STADIO dell'avanti di fla (ricetta gfx1151) porta l'errore 1e-6? (27/9/2026)
Ogni stadio (intra = kkt+inversa+w,u · h = ricorrenza dello stato · o = uscita) viene sostituito da una versione torch
in fp64 (ingressi fp32 -> fp64 -> uscita riportata in fp32): se l'errore dell'uscita verso il riferimento fp64 crolla
sostituendo lo stadio X, l'errore nasce in X. Stampa anche il profilo dell'errore per indice di blocco (0..63)."""
import sys
from pathlib import Path

import torch

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import gdn_fla_gfx1151 as G  # noqa: E402
import prova_equivalenza as P  # noqa: E402


def _b(x, BT=64):                                     # [B, T, H, ...] -> [B, NC, H, BT, ...] in fp64
    B, T, H = x.shape[:3]
    return x.double().reshape(B, T // BT, BT, H, *x.shape[3:]).transpose(2, 3)


def _r(x):                                            # [B, NC, H, BT, ...] -> [B, T, H, ...]
    B, NC, H, BT = x.shape[:4]
    return x.transpose(2, 3).reshape(B, NC * BT, H, *x.shape[4:])


def intra64(k, v, g, beta, cu_seqlens=None, chunk_indices=None, chunk_size=64):
    kc, vc, gc, bc = _b(k), _b(v), _b(g), _b(beta)
    i = torch.arange(64, device=k.device)
    L = (bc[..., :, None] * (kc @ kc.transpose(-1, -2))) * torch.exp2(gc[..., :, None] - gc[..., None, :])
    L = torch.where(i[:, None] > i[None, :], L, 0.0)
    eye = torch.eye(64, dtype=torch.float64, device=k.device)
    Tm = torch.linalg.solve_triangular(eye + L, eye.expand_as(L), upper=False)
    w = Tm @ (kc * (bc * torch.exp2(gc))[..., None])
    u = Tm @ (vc * bc[..., None])
    return _r(w).float(), _r(u).float(), _r(Tm).float().contiguous()


def h64(k, w, u, g, initial_state=None, output_final_state=False, cu_seqlens=None, chunk_indices=None,
        state_v_first=False, chunk_size=64, **kw):
    assert initial_state is None and not state_v_first
    kc, wc, uc, gc = _b(k), _b(w), _b(u), _b(g)
    B, NC, H, BT, K = kc.shape
    V = uc.shape[-1]
    h = torch.zeros(B, H, K, V, dtype=torch.float64, device=k.device)
    hs, vn = [], []
    for c in range(NC):
        hs.append(h)
        vnew = uc[:, c] - wc[:, c] @ h
        vn.append(vnew)
        gl = gc[:, c, :, -1]
        h = h * torch.exp2(gl)[..., None, None] + (kc[:, c] * torch.exp2(gl[..., None] - gc[:, c])[..., None]
                                                   ).transpose(-1, -2) @ vnew
    hh = torch.stack(hs, 1).float()                   # [B, NT, H, K, V]
    return hh, _r(torch.stack(vn, 1)).float(), None


def o64(q, k, v, h, g, scale, cu_seqlens=None, chunk_indices=None, state_v_first=False, chunk_size=64, **kw):
    qc, kc, vc, gc = _b(q), _b(k), _b(v), _b(g)
    i = torch.arange(64, device=q.device)
    hh = h.double()                                   # [B, NC, H, K, V]
    inter = (qc * torch.exp2(gc)[..., None]) @ hh
    Aqk = (qc @ kc.transpose(-1, -2)) * torch.exp2(gc[..., :, None] - gc[..., None, :])
    Aqk = torch.where(i[:, None] >= i[None, :], Aqk, 0.0)
    return (_r((inter + Aqk @ vc) * scale)).float()


def main() -> int:
    fla_gdn = G.configura()
    import fla.ops.gated_delta_rule.chunk as C
    orig = {"intra": C.chunk_gated_delta_rule_fwd_intra, "h": C.chunk_gated_delta_rule_fwd_h, "o": C.chunk_fwd_o}
    sost = {"intra": intra64, "h": h64, "o": o64}
    nomi_c = {"intra": "chunk_gated_delta_rule_fwd_intra", "h": "chunk_gated_delta_rule_fwd_h", "o": "chunk_fwd_o"}
    for seme, forza in ((1, 1.0), (2, 8.0)):
        q, k, v, gg, be, do = P.ingressi(seme, 2, 4096, 32, 128, forza)
        with torch.no_grad():
            o64_, _ = P.RIPIEGO(q.double(), k.double(), v.double(), g=gg.double(), beta=be.double(),
                                use_qk_l2norm_in_kernel=True)
            o32r, _ = P.RIPIEGO(q, k, v, g=gg, beta=be, use_qk_l2norm_in_kernel=True)
            er = P.rel(o32r, o64_)
            print(f"seme {seme} forza {forza}: ripiego fp32~fp64 {er:.2e}", flush=True)
            for variante in ([], ["intra"], ["h"], ["o"], ["intra", "h"], ["h", "o"], ["intra", "h", "o"]):
                for s in orig:
                    setattr(C, nomi_c[s], sost[s] if s in variante else orig[s])
                of, _ = fla_gdn(q, k, v, g=gg, beta=be, use_qk_l2norm_in_kernel=True)
                e = P.rel(of, o64_)
                print(f"  fp64 in {'+'.join(variante) or 'nessuno':12s}: fla~fp64 {e:.2e} (rapporto {e / er:.2f})",
                      flush=True)
                if not variante:
                    d = (of.double() - o64_).reshape(2, 64, 64, 32, 128)
                    n = o64_.reshape(2, 64, 64, 32, 128)
                    per_blocco = (d.pow(2).sum((0, 2, 3, 4)) / n.pow(2).sum((0, 2, 3, 4))).sqrt()
                    per_pos = (d.pow(2).sum((0, 1, 3, 4)) / n.pow(2).sum((0, 1, 3, 4))).sqrt()
                    print("    per blocco (0,1,2,4,8,16,32,63): " + " ".join(
                        f"{per_blocco[j]:.1e}" for j in (0, 1, 2, 4, 8, 16, 32, 63)), flush=True)
                    print("    per posizione nel blocco (0,1,8,31,63): " + " ".join(
                        f"{per_pos[j]:.1e}" for j in (0, 1, 8, 31, 63)), flush=True)
            for s in orig:
                setattr(C, nomi_c[s], orig[s])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

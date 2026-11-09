"""Esportazione del piano (JSON) e materializzazione dei nuovi piani.

La materializzazione RICALCOLA le opzioni scelte con lo stesso codice del menu
(apply_option e' deterministico), codifica i piani nel formato reale e verifica
che byte e distorsione coincidano con quanto scritto nel piano.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import torch

from .allocazione import Allocation
from .codifica_torch import encode_full_torch
from .formato import TensorKey, encode_full, encode_sparse, full_plane_bytes, packing_for_scales
from .menu import MenuConfig, TensorMenu, apply_option, iter_batches, option_bytes
from .sorgenti import FisherSource, PlaneSource, TeacherSource

PLAN_VERSION = 1


def _cfg_dict(cfg: MenuConfig) -> Dict:
    return {"densities": list(cfg.densities), "packing": cfg.packing, "sparse_coding": cfg.sparse_coding,
            "damp": cfg.damp, "refit_existing": cfg.refit_existing,
            "min_planes_to_drop": cfg.min_planes_to_drop, "allow_plus_full": cfg.allow_plus_full,
            "safeguard": cfg.safeguard, "mm_dtype": str(cfg.mm_dtype).replace("torch.", ""),
            "fast_diag": cfg.fast_diag, "allow_g64": cfg.allow_g64}


def cfg_from_dict(d: Dict, device: str = "cpu") -> MenuConfig:
    return MenuConfig(densities=tuple(d["densities"]), packing=d["packing"], sparse_coding=d["sparse_coding"],
                      damp=d["damp"], refit_existing=d["refit_existing"],
                      min_planes_to_drop=d["min_planes_to_drop"], allow_plus_full=d["allow_plus_full"],
                      safeguard=d["safeguard"], device=device, mm_dtype=getattr(torch, d["mm_dtype"]),
                      # piani scritti prima del percorso veloce: si rimaterializzano col riferimento
                      fast_diag=bool(d.get("fast_diag", False)),
                      allow_g64=bool(d.get("allow_g64", False)))


def export_plan(menus: Sequence[TensorMenu], alloc: Allocation, cfg: MenuConfig, path: Optional[str] = None,
                note: str = "", extra: Optional[Dict] = None) -> Dict:
    """Piano JSON. Ogni tensore porta il menu completo ("menu": [nome, byte, d] e
    "menu_full" con nnz ed extra per opzione): e' cio' che permette a unisci.py di
    ricombinare piani di processi diversi e risolvere il budget globale una volta sola."""
    tensors = []
    base = 0
    for m, c in zip(menus, alloc.choice):
        o, cur = m.options[c], m.get("cur")
        base += cur.bytes
        tensors.append({"key": m.key.name(), "n_planes": m.n_planes, "shape": list(m.shape),
                        "option": o.name, "bytes": o.bytes, "delta_bytes": o.bytes - cur.bytes,
                        "distortion": o.distortion, "distortion_cur": cur.distortion,
                        "nnz": o.nnz, **o.extra,
                        "menu": [[x.name, x.bytes, x.distortion] for x in m.options],
                        "menu_full": [{"name": x.name, "bytes": x.bytes, "distortion": x.distortion,
                                       "nnz": x.nnz, "extra": x.extra} for x in m.options]})
    plan = {"version": PLAN_VERSION, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "note": note,
            "budget_bytes": alloc.budget, "lambda": alloc.lam, "total_bytes": alloc.total_bytes,
            "base_bytes": base, "pred_distortion_total": alloc.total_distortion,
            "pred_distortion_base": sum(m.get("cur").distortion for m in menus),
            "filled": alloc.filled, "config": _cfg_dict(cfg),
            "layers": sorted({m.key.layer for m in menus}), **(extra or {}), "tensors": tensors}
    if path:
        with open(path, "w") as f:
            json.dump(plan, f, indent=1)
    return plan


def load_plan(path: str) -> Dict:
    with open(path) as f:
        plan = json.load(f)
    if plan.get("version") != PLAN_VERSION:
        raise ValueError(f"versione piano {plan.get('version')} non supportata")
    return plan


def materialize(plan: Dict, ps: PlaneSource, ts: TeacherSource, fs: FisherSource,
                out_dir: Optional[str] = None, device: str = "cpu", batch: int = 32,
                rel_tol: float = 1e-4, only_changed: bool = False,
                max_batch_weights: Optional[int] = None, encoder: str = "torch",
                max_batch_bytes: Optional[int] = None) -> Dict:
    """Ricostruisce i piani scelti; se out_dir, salva `rd_L{layer:03d}.safetensors`.

    Per tensore: `{nome}.planes` int8 [K',o,i], `{nome}.scales` fp16, `{nome}.rotated`;
    se c'e' il piano sparso anche `{nome}.sparse` (uint8: flusso codificato) — i byte
    del flusso sono ESATTAMENTE quelli contati nel piano (verificato qui).
    encoder: "torch" codifica i piani pieni di tutto il lotto sul dispositivo
    (codifica_torch, stessi byte del riferimento), "numpy" tensore per tensore.
    Con only_changed=True i tensori 'cur' si saltano (ma se refit_existing, anche
    loro hanno scale nuove: di norma si materializza tutto).
    """
    cfg = cfg_from_dict(plan["config"], device)
    cfg.batch = batch
    cfg.max_batch_weights = max_batch_weights
    cfg.max_batch_bytes = max_batch_bytes
    entries = {TensorKey.parse(t["key"]): t for t in plan["tensors"]}
    keys = [k for k, t in entries.items() if not (only_changed and t["option"] == "cur")]
    groups: Dict[tuple, List[TensorKey]] = defaultdict(list)
    for k in keys:
        inf = ps.info(k)
        groups[(inf.n_planes, inf.out, inf.in_, inf.rotated, inf.group, inf.packing,
                entries[k]["option"])].append(k)
    out: Dict[int, Dict[str, torch.Tensor]] = defaultdict(dict)
    report = {"n": 0, "max_rel_err_distortion": 0.0, "bytes_mismatch": [], "encoded_bytes": 0,
              "encoder_mismatch": []}
    for (K, o, i, rot, _grp, _pk, opt), chunk, tb in iter_batches(groups, cfg, ps, ts, fs):
        fr = apply_option(tb, opt)
        d = tb.dist(fr.D).cpu()
        by = option_bytes(fr, K, o, i, cfg, tb.packing, tb.group).cpu()
        planes, scales = fr.planes.cpu(), fr.scales.cpu()           # un trasferimento per lotto
        sparse = fr.sparse_plane.cpu() if fr.sparse_plane is not None else None
        sparse_sc = fr.sparse_scales.cpu() if fr.sparse_plane is not None else None
        enc_full = None
        pk = packing_for_scales(tb.packing, i, int(fr.scales.shape[-1]))     # opzioni g64: packing a scala 64
        if encoder == "torch":
            enc_full = encode_full_torch(fr.planes, fr.scales, pk).cpu()     # [B,K',n_byte]
            # controllo incrociato sul primo tensore del lotto: byte torch == byte numpy
            ref = b"".join(encode_full(planes[0, p], scales[0, p], pk) for p in range(planes.shape[1]))
            if enc_full[0].numpy().tobytes() != ref:
                report["encoder_mismatch"].append(chunk[0].name())
        for j, k in enumerate(chunk):
            e = entries[k]
            rel = abs(float(d[j]) - e["distortion"]) / max(abs(e["distortion"]), 1e-30)
            report["max_rel_err_distortion"] = max(report["max_rel_err_distortion"], rel)
            # codifica reale: la lunghezza deve combaciare col conteggio
            if enc_full is not None:
                enc = int(enc_full[j].numel())
            else:
                enc = sum(len(encode_full(planes[j, p], scales[j, p], pk)) for p in range(planes.shape[1]))
            sp = None
            if sparse is not None:
                sp = encode_sparse(sparse[j], sparse_sc[j], cfg.sparse_coding, tb.group)
                enc += len(sp)
            if enc != int(by[j]) or enc != e["bytes"]:
                report["bytes_mismatch"].append((k.name(), enc, int(by[j]), e["bytes"]))
            report["encoded_bytes"] += enc
            report["n"] += 1
            n = k.name()
            dst = out[k.layer]
            dst[f"{n}.planes"] = planes[j].contiguous()
            dst[f"{n}.scales"] = scales[j].contiguous()
            dst[f"{n}.rotated"] = torch.tensor([int(rot)], dtype=torch.uint8)
            if sp is not None:
                dst[f"{n}.sparse"] = torch.frombuffer(bytearray(sp), dtype=torch.uint8)
                dst[f"{n}.sparse_shape"] = torch.tensor([o, i], dtype=torch.int64)
    report["distortion_ok"] = report["max_rel_err_distortion"] <= rel_tol
    if out_dir:
        from safetensors.torch import save_file
        for layer, dct in out.items():
            save_file(dct, f"{out_dir}/rd_L{layer:03d}.safetensors",
                      metadata={"sparse_coding": cfg.sparse_coding, "packing": cfg.packing})
    report["tensors"] = out
    return report

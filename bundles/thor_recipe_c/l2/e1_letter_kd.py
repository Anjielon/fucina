#!/usr/bin/env python3
"""e1_letter_kd.py — perdita "letter-KD" + coerenza fra rotazioni, da innestare in un trainer.

COSA OTTIMIZZA
  Per ogni prompt in formato MMLU lo studente produce 4 logit (quelli dei token
  " A"," B"," C"," D" nell'ultima posizione). La perdita e':

      L = KL( p_maestro || p_studente )  *  T^2                     (distillazione)
        + lambda * JS_rot( studente, sulle 4 rotazioni dello stesso item )  (coerenza)

  - p_maestro = softmax(teacher_logprobs / T) sulle SOLE 4 lettere (gia' rinormalizzate
    dal client). Opzione `target="debiased"`: il bersaglio e' la media (geometrica) delle
    4 rotazioni del maestro riallineate al contenuto e poi rimesse in posizione — toglie al
    bersaglio il bias di posizione del maestro (stile PriDe), senza dati in piu'.
  - JS_rot: per ogni item, q_r = distribuzione dello studente alla rotazione r RIALLINEATA
    al contenuto (perm[p] = contenuto alla posizione p). m = media_r q_r.
    JS_rot = media_r KL(q_r || m). E' zero se e solo se lo studente da' la stessa
    probabilita' allo stesso contenuto, ovunque sia scritto: niente lettera preferita.

COME SI INNESTA NEL TUO TRAINER (quello che allena solo i pesi RMSNorm e le scale degli
ultimi N strati):
  1. Dataset/collate: `LetterKDDataset` + `LetterKDCollator(tokenizer)` (padding a SINISTRA,
     cosi' l'ultima posizione e' -1 per tutte le righe) + `GroupBatchSampler` (ogni batch
     contiene gruppi INTERI di 4 rotazioni — necessario per il termine di coerenza).
  2. Parametri allenabili: `select_trainable_params(model, last_n=N, ...)` congela tutto e
     sblocca i nomi che combaciano con le regex (adatta `scale_regex` al tuo modulo ternario).
  3. Nel passo di training: `loss, stats = letter_kd_step(model, batch, letter_ids, lam=..)`
     oppure, se il tuo trainer calcola gia' i logit, `letter_kd_loss_from_logits4(...)`.
  4. Nessuna perdita LM sul testo: si allenano solo 4 logit per prompt. Consigliato
     tenere anche un piccolo termine di regolarizzazione verso i pesi iniziali (L2 sulle
     scale) — vedi `l2_to_init`.

AUTOVERIFICA:  python3 e1_letter_kd.py --selftest
"""
from __future__ import annotations

import argparse
import math
import random
import re
import sys
from typing import Iterator, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

from common import CHOICE_STRINGS, read_jsonl


# ── Perdite pure (tensori) ───────────────────────────────────────────────────
def realign_to_content(x_pos: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """[B,4] per posizione -> [B,4] per contenuto: out[b, perm[b,p]] = x_pos[b,p]."""
    return torch.empty_like(x_pos).scatter_(1, perm, x_pos)


def content_to_position(x_content: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Inversa: out[b,p] = x_content[b, perm[b,p]]."""
    return x_content.gather(1, perm)


def _group_mean(x: torch.Tensor, group_index: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Media per gruppo di righe [B,4] -> [G,4]."""
    s = torch.zeros(n_groups, x.shape[1], dtype=x.dtype, device=x.device)
    s.index_add_(0, group_index, x)
    cnt = torch.bincount(group_index, minlength=n_groups).clamp_min(1).to(x.dtype)
    return s / cnt[:, None]


def debias_teacher(teacher_logp: torch.Tensor, perm: torch.Tensor,
                   group_index: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Bersaglio senza bias di posizione: media geometrica delle rotazioni per contenuto,
    rinormalizzata e rimessa nell'ordine delle posizioni di ogni riga. [B,4] logprob."""
    lc = realign_to_content(teacher_logp, perm)
    gm = _group_mean(lc, group_index, n_groups)             # media dei log = geometrica
    gm = gm - torch.logsumexp(gm, dim=1, keepdim=True)
    return content_to_position(gm[group_index], perm)


def letter_kd_loss(student_logits4: torch.Tensor, teacher_logp4: torch.Tensor,
                   T: float = 1.0, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """KL(maestro || studente) sulle 4 lettere, scalata per T^2 (Hinton). Media sul batch."""
    log_ps = F.log_softmax(student_logits4.float() / T, dim=-1)
    log_pt = F.log_softmax(teacher_logp4.float() / T, dim=-1)
    kl = (log_pt.exp() * (log_pt - log_ps)).sum(-1) * (T * T)
    if weight is not None:
        return (kl * weight).sum() / weight.sum().clamp_min(1e-8)
    return kl.mean()


def rotation_consistency_loss(student_logits4: torch.Tensor, perm: torch.Tensor,
                              group_index: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Divergenza di Jensen-Shannon generalizzata fra le rotazioni dello stesso item,
    dopo il riallineamento al contenuto. Il gradiente scorre anche attraverso la media m
    (versione simmetrica); 0 <=> stessa probabilita' allo stesso contenuto."""
    logq = realign_to_content(F.log_softmax(student_logits4.float(), dim=-1), perm)
    q = logq.exp()
    m = _group_mean(q, group_index, n_groups)
    logm = m.clamp_min(1e-12).log()[group_index]
    return (q * (logq - logm)).sum(-1).mean()


def letter_kd_loss_from_logits4(student_logits4: torch.Tensor, batch: dict, lam: float = 0.5,
                                T: float = 1.0, target: str = "raw") -> tuple[torch.Tensor, dict]:
    """Perdita totale a partire dai 4 logit dello studente gia' calcolati."""
    perm, gi, G = batch["perm"], batch["group_index"], batch["n_groups"]
    tgt = batch["teacher_logprobs"]
    if target == "debiased":
        tgt = debias_teacher(tgt, perm, gi, G)
    kd = letter_kd_loss(student_logits4, tgt, T)
    rc = rotation_consistency_loss(student_logits4, perm, gi, G) if lam > 0 else \
        student_logits4.new_zeros(())
    loss = kd + lam * rc
    with torch.no_grad():
        pred = student_logits4.argmax(-1)
        stats = {"loss": float(loss), "kd": float(kd), "rot": float(rc),
                 "acc": float((pred == batch["gold_pos"]).float().mean()),
                 "agree_teacher": float((pred == batch["teacher_logprobs"].argmax(-1)).float().mean()),
                 "pred_hist": torch.bincount(pred, minlength=4).tolist()}
    return loss, stats


def l2_to_init(params: Sequence[torch.nn.Parameter], init: Sequence[torch.Tensor]) -> torch.Tensor:
    """Regolarizzazione verso i valori iniziali (tienila piccola: 1e-4..1e-2)."""
    return sum(((p - p0) ** 2).sum() for p, p0 in zip(params, init))


# ── Logit delle lettere dal modello ──────────────────────────────────────────
def student_letter_logits(model: torch.nn.Module, input_ids: torch.Tensor,
                          attention_mask: torch.Tensor, letter_ids: torch.Tensor) -> torch.Tensor:
    """4 logit all'ULTIMA posizione (serve padding a sinistra). Prova `logits_to_keep=1`
    (transformers recenti: lm_head solo sull'ultima posizione) e ripiega sul forward pieno."""
    try:
        out = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=1,
                    use_cache=False)
    except TypeError:
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = out.logits if hasattr(out, "logits") else out[0]
    return logits[:, -1, :].index_select(-1, letter_ids)


def letter_kd_step(model: torch.nn.Module, batch: dict, letter_ids: torch.Tensor,
                   lam: float = 0.5, T: float = 1.0, target: str = "raw") -> tuple[torch.Tensor, dict]:
    """Forward + perdita. Il backward/optimizer restano al tuo trainer."""
    logits4 = student_letter_logits(model, batch["input_ids"], batch["attention_mask"], letter_ids)
    return letter_kd_loss_from_logits4(logits4, batch, lam=lam, T=T, target=target)


def hf_letter_ids(tokenizer, prompt: str = "Question?\nA. x\nB. y\nC. z\nD. w\nAnswer:") -> list[int]:
    """ID di " A".." D" ricavati IN CONTESTO (dopo "Answer:"), non tokenizzando " A" da solo:
    con SentencePiece lo spazio fittizio iniziale produrrebbe due token."""
    base = tokenizer.encode(prompt, add_special_tokens=False)
    ids = []
    for s in CHOICE_STRINGS:
        full = tokenizer.encode(prompt + s, add_special_tokens=False)
        if full[:len(base)] != base or len(full) != len(base) + 1:
            raise ValueError(f"{s!r} dopo 'Answer:' non e' un token singolo: {full[-3:]}")
        ids.append(full[-1])
    return ids


# ── Selezione dei parametri allenabili ───────────────────────────────────────
def select_trainable_params(model: torch.nn.Module, last_n: int = 4,
                            norm_regex: str = r"(norm|ln_f|layernorm)[^.]*\.weight$",
                            scale_regex: str = r"(scale|scales|alpha|plane_scale)s?$",
                            layer_regex: str = r"\.(?:layers|h|blocks)\.(\d+)\.",
                            n_layers: Optional[int] = None) -> list[tuple[str, torch.nn.Parameter]]:
    """Congela tutto, poi sblocca:
      - TUTTI i pesi RMSNorm (anche la norma finale prima di lm_head);
      - i parametri di scala (nome che combacia con `scale_regex`) degli ultimi `last_n` strati.
    Adatta `scale_regex` ai nomi del tuo modulo ternario (es. 's0','s1' -> r'\\.s\\d+$').
    Ritorna la lista (nome, parametro) sbloccati, da passare all'ottimizzatore."""
    lr = re.compile(layer_regex)
    if n_layers is None:
        idx = [int(m.group(1)) for n, _ in model.named_parameters() if (m := lr.search(n))]
        n_layers = (max(idx) + 1) if idx else 0
    first_tail = n_layers - last_n
    nr, sr = re.compile(norm_regex), re.compile(scale_regex)
    picked = []
    for name, p in model.named_parameters():
        p.requires_grad_(False)
        m = lr.search(name)
        is_norm = bool(nr.search(name))
        is_tail_scale = bool(sr.search(name)) and m is not None and int(m.group(1)) >= first_tail
        if is_norm or is_tail_scale:
            p.requires_grad_(True)
            picked.append((name, p))
    return picked


# ── Dati ─────────────────────────────────────────────────────────────────────
class LetterKDDataset(Dataset):
    """Righe del jsonl prodotto da teacher_client.py (devono avere teacher_logprobs)."""

    def __init__(self, path: str, split: Optional[str] = "train"):
        rows = [r for r in read_jsonl(path) if split is None or r.get("split") == split]
        self.rows = [r for r in rows if r.get("teacher_logprobs") is not None]
        self.groups: dict[str, list[int]] = {}
        for i, r in enumerate(self.rows):
            self.groups.setdefault(r["group_id"], []).append(i)
        # si tengono solo i gruppi COMPLETI (4 rotazioni): la coerenza li richiede
        self.groups = {g: ix for g, ix in self.groups.items() if len(ix) == 4}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        return self.rows[i]


class GroupBatchSampler(Sampler[list[int]]):
    """Batch fatti di gruppi interi: groups_per_batch * 4 righe, ordine seminato per epoca."""

    def __init__(self, ds: LetterKDDataset, groups_per_batch: int = 4, seed: int = 0):
        self.ds, self.gpb, self.seed, self.epoch = ds, groups_per_batch, seed, 0

    def set_epoch(self, e: int) -> None:
        self.epoch = e

    def __iter__(self) -> Iterator[list[int]]:
        keys = sorted(self.ds.groups)
        random.Random(self.seed + self.epoch).shuffle(keys)
        for i in range(0, len(keys) - self.gpb + 1, self.gpb):
            yield [j for k in keys[i:i + self.gpb] for j in self.ds.groups[k]]

    def __len__(self) -> int:
        return len(self.ds.groups) // self.gpb


class LetterKDCollator:
    """Tokenizza con padding a SINISTRA e costruisce perm / group_index / bersagli."""

    def __init__(self, tokenizer, max_length: int = 2048):
        self.tok = tokenizer
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.max_length = max_length

    def __call__(self, rows: list[dict]) -> dict:
        enc = self.tok([r["prompt"] for r in rows], return_tensors="pt", padding=True,
                       truncation=True, max_length=self.max_length)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                **collate_targets(rows)}


def collate_targets(rows: list[dict]) -> dict:
    """Parte del batch indipendente dal tokenizer (utile anche per trainer esistenti)."""
    gid = {}
    for r in rows:
        gid.setdefault(r["group_id"], len(gid))
    return {"perm": torch.tensor([r["perm"] for r in rows], dtype=torch.long),
            "gold_pos": torch.tensor([r["gold_pos"] for r in rows], dtype=torch.long),
            "teacher_logprobs": torch.tensor([r["teacher_logprobs"] for r in rows],
                                             dtype=torch.float32),
            "group_index": torch.tensor([gid[r["group_id"]] for r in rows], dtype=torch.long),
            "n_groups": len(gid)}


def move_batch(batch: dict, device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


# ── Autoverifica ─────────────────────────────────────────────────────────────
def _selftest() -> None:
    torch.manual_seed(0)
    # 2 item x 4 rotazioni
    perms = torch.tensor([[(p + r) % 4 for p in range(4)] for _ in range(2) for r in range(4)])
    gi = torch.tensor([0] * 4 + [1] * 4)
    # 1) realign / inversa
    x = torch.randn(8, 4)
    assert torch.allclose(content_to_position(realign_to_content(x, perms), perms), x)
    # 2) studente perfettamente coerente: stessi logit per contenuto -> JS = 0
    content_logits = torch.randn(2, 4)
    s = content_to_position(content_logits[gi], perms)
    assert rotation_consistency_loss(s, perms, gi, 2).abs() < 1e-6
    # 3) studente con bias puro di posizione (preferisce sempre A) -> JS > 0
    s_bias = torch.tensor([[3.0, 0, 0, 0]]).repeat(8, 1)
    assert rotation_consistency_loss(s_bias, perms, gi, 2) > 0.1
    # 4) KD = 0 se studente == maestro
    t = F.log_softmax(torch.randn(8, 4), -1)
    assert letter_kd_loss(t.clone(), t).abs() < 1e-6
    # 5) debias del maestro: un maestro con bias puro su A diventa uniforme
    tb = F.log_softmax(s_bias, -1)
    d = debias_teacher(tb, perms, gi, 2).exp()
    assert torch.allclose(d, torch.full_like(d, 0.25), atol=1e-6)
    # 6) gradienti finiti e il termine di coerenza si abbassa ottimizzando
    w = torch.zeros(8, 4, requires_grad=True)
    opt = torch.optim.SGD([w], lr=1.0)
    batch = {"perm": perms, "group_index": gi, "n_groups": 2, "teacher_logprobs": t,
             "gold_pos": torch.zeros(8, dtype=torch.long)}
    l0 = None
    for _ in range(50):
        loss, st = letter_kd_loss_from_logits4(w + s_bias, batch, lam=1.0, target="debiased")
        opt.zero_grad()
        loss.backward()
        opt.step()
        l0 = l0 if l0 is not None else st["loss"]
    assert st["loss"] < l0 and math.isfinite(st["loss"])
    print(f"selftest ok — perdita {l0:.4f} -> {st['loss']:.4f}, rot {st['rot']:.4f}")


def _demo(args) -> None:
    """Mini-allenamento su un modello transformers (solo per provare l'innesto)."""
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.hf_model)
    model = AutoModelForCausalLM.from_pretrained(args.hf_model, torch_dtype=torch.bfloat16)
    model.to(args.device)
    lids = torch.tensor(hf_letter_ids(tok), device=args.device)
    picked = select_trainable_params(model, last_n=args.last_n)
    print(f"parametri allenabili: {len(picked)} tensori, "
          f"{sum(p.numel() for _, p in picked) / 1e6:.2f} M")
    opt = torch.optim.AdamW([p for _, p in picked], lr=args.lr, weight_decay=0.0)
    ds = LetterKDDataset(args.data, split="train")
    dl = DataLoader(ds, batch_sampler=GroupBatchSampler(ds, args.groups_per_batch),
                    collate_fn=LetterKDCollator(tok))
    model.train()
    for step, batch in enumerate(dl):
        loss, st = letter_kd_step(model, move_batch(batch, args.device), lids,
                                  lam=args.lam, T=args.T, target=args.target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for _, p in picked], 1.0)
        opt.step()
        if step % 10 == 0:
            print(step, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in st.items()})
        if step + 1 >= args.steps:
            break


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--hf-model", default=None, help="demo di innesto su un modello HF")
    ap.add_argument("--data", default="data/items_teacher.jsonl")
    ap.add_argument("--device", default="auto",
                    help="auto = cuda se disponibile (risolto SOLO con --hf-model: --selftest non tocca la GPU)")
    ap.add_argument("--last-n", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--target", choices=["raw", "debiased"], default="raw")
    ap.add_argument("--groups-per-batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=100)
    a = ap.parse_args()
    if a.selftest or not a.hf_model:
        _selftest()
    if a.hf_model:
        if a.device == "auto":
            a.device = "cuda" if torch.cuda.is_available() else "cpu"
        _demo(a)
    sys.exit(0)

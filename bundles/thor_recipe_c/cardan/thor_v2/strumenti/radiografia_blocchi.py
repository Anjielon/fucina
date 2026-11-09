#!/usr/bin/env python3
"""Radiografia PER BLOCCO di uno strato vero di Thor alla forma della forgia (25/9 00:4x).

Perche' (costo_passo.md, campionamento py-spy della corsa S1b): in un passo da ~1,2-1,3 s il
thread principale aspetta la GPU il 95% del tempo, e l'attesa prima della GEMM gate_up
(norme + attenzione + esperto condiviso di due strati + nucleo avanti) vale ~270-296 ms per
passo: ~95-110 ms per strato, contro i ~20 ms che le GEMM dense di quei blocchi richiedono.
Dove vanno gli altri? Questo strumento lo misura blocco per blocco, con eventi GPU:

  1. AVANTI dentro lo strato vero (ganci pre/post sui sottomoduli + involucri su
     causal_conv1d_fn e torch_chunk_gated_delta_rule): la stessa chiamata della forgia
     (`strato(hidden_states=..., position_embeddings=..., attention_mask=..., position_ids=...)`).
  2. AVANTI e AVANTI+INDIETRO dei blocchi ISOLATI, sugli ingressi catturati al punto 1:
     ingresso con gradiente, pesi congelati come nella forgia; esperti anche col gradiente
     dei pesi (dW, come nel passo); la differenza e' il costo dell'indietro.
  3. PER-PESO alle forme vere (un solo strato: gate_up + down): nucleo ternario avanti bf16,
     avanti+indietro, norma dei gradienti (come `passo_ottimizzazione`), AdamW fuso a 3 gruppi.

Nessun file di thor_v2 viene toccato: importa `valida_finestra_8_12` in sola lettura.
Scrive solo il JSON indicato con --json.

Uso (SOLO a GPU libera — rifiuta se un altro processo tiene /dev/kfd, come
confronto_scorrevole_6c.sh; llama-server/whisper-server esclusi):
  cd thor_v2 && PYTHONPATH=strumenti/pylib_fla THOR_CACHE_STRATI=@@ROOT:models@@/gguf/tony-tern/cache_strati \\
    THOR_ESPERTI_THOR=1 THOR_ROTAZIONE_INGRESSO=1 THOR_NUCLEO_BF16=1 THOR_VALIDA_BATCH_SEQ=8 \\
    THOR_CALIB_MISCELA="cot:2,edu:3,wiki:3" \\
    @@ROOT:venv@@/bin/python3 strumenti/radiografia_blocchi.py --strati 2,3 --json /percorso/radiografia.json
Collaudo senza GPU (strati giocattolo, CPU):
  CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=4 \\
    @@ROOT:venv@@/bin/python3 strumenti/radiografia_blocchi.py --giocattolo

Limite dichiarato: l'ingresso dello strato e' l'embedding dei token veri, non l'uscita vera
degli strati precedenti (caricarli tutti costerebbe una forgia). Per i tempi conta la forma;
il bilanciamento degli esperti (che pesa sul ciclo di GEMM) resta quello del router vero
su quell'ingresso.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI.parent))


# ══════════════════════ guardia: GPU tenuta da altri ══════════════════════
def gpu_tenuta_da_altri() -> "list[str]":
    """Processi (diversi da questo) con /dev/kfd aperto, esclusi llama-server/whisper-server:
    stessa regola di confronto_scorrevole_6c.sh. Da chiamare PRIMA di inizializzare CUDA."""
    mio = os.getpid()
    trovati = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit() or int(p.name) == mio:
            continue
        try:
            fds = list((p / "fd").iterdir())
        except (PermissionError, FileNotFoundError, ProcessLookupError, NotADirectoryError):
            continue
        tiene = False
        for f in fds:
            try:
                if os.readlink(f) == "/dev/kfd":
                    tiene = True
                    break
            except OSError:
                continue
        if not tiene:
            continue
        try:
            comm = (p / "comm").read_text().strip()
        except OSError:
            comm = "?"
        if comm in ("llama-server", "whisper-server"):
            continue
        trovati.append(f"{p.name} {comm}")
    return trovati


# ══════════════════════ cronometro (eventi GPU, o orologio su CPU) ══════════════════════
class Orologio:
    """Marcatori di tempo: eventi GPU su cuda, perf_counter su CPU. `durata_ms(a, b)` dopo
    `sincronizza()`."""

    def __init__(self, torch_mod, su_gpu: bool):
        self.torch = torch_mod
        self.su_gpu = su_gpu

    def marca(self):
        if self.su_gpu:
            e = self.torch.cuda.Event(enable_timing=True)
            e.record()
            return e
        return time.perf_counter()

    def sincronizza(self) -> None:
        if self.su_gpu:
            self.torch.cuda.synchronize()

    def durata_ms(self, a, b) -> float:
        return a.elapsed_time(b) if self.su_gpu else (b - a) * 1e3

    def misura(self, fn, ripetizioni: int, riscaldo: int = 2) -> float:
        """Mediana in ms di `fn()` (dopo `riscaldo` chiamate non contate: compilazioni, autotune)."""
        for _ in range(riscaldo):
            fn()
        self.sincronizza()
        tempi = []
        for _ in range(ripetizioni):
            a = self.marca()
            fn()
            b = self.marca()
            self.sincronizza()
            tempi.append(self.durata_ms(a, b))
        return statistics.median(tempi)


# ══════════════════════ 1. avanti dentro lo strato vero ══════════════════════
class Ganci:
    """Ganci pre/post sui sottomoduli (e involucri sulle funzioni di modulo della regola
    delta e della convoluzione): marcano inizio/fine di ogni blocco dentro la chiamata vera."""

    FUNZIONI = ("causal_conv1d_fn", "torch_chunk_gated_delta_rule")

    def __init__(self, strato, orologio: Orologio, M):
        self.orologio = orologio
        self.M = M
        self.attivo = False
        self.marche: dict = {}
        self.catturati: dict = {}
        self._maniglie = []
        self._originali = {}
        nomi = ["input_layernorm", "post_attention_layernorm", "mlp", "mlp.shared_expert",
                "mlp.shared_expert_gate", "mlp.gate", "mlp.experts"]
        if hasattr(strato, "linear_attn"):
            nomi += ["linear_attn", "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_a",
                     "linear_attn.in_proj_b", "linear_attn.norm", "linear_attn.out_proj"]
        if hasattr(strato, "self_attn"):
            nomi += ["self_attn"] + [f"self_attn.{n}" for n in ("q_proj", "k_proj", "v_proj", "o_proj",
                                                                  "q_norm", "k_norm")
                                     if hasattr(strato.self_attn, n)]
        moduli = dict(strato.named_modules())
        self.nomi = [n for n in nomi if n in moduli]
        for n in self.nomi:
            m = moduli[n]
            self._maniglie.append(m.register_forward_pre_hook(self._pre(n), with_kwargs=True))
            self._maniglie.append(m.register_forward_hook(self._post(n)))
        for f in self.FUNZIONI:
            if hasattr(M, f):
                self._originali[f] = getattr(M, f)
                setattr(M, f, self._avvolgi(f, self._originali[f]))

    def _pre(self, nome):
        def g(_m, args, kwargs):
            if nome not in self.catturati:        # ingressi per le misure isolate (una volta)
                self.catturati[nome] = (_stacca(args), _stacca(kwargs))
            if self.attivo:
                self.marche.setdefault(nome, []).append([self.orologio.marca(), None])
        return g

    def _post(self, nome):
        def g(_m, _args, _out):
            if self.attivo:
                self.marche[nome][-1][1] = self.orologio.marca()
        return g

    def _avvolgi(self, nome, fn):
        def involucro(*a, **k):
            if nome not in self.catturati:
                self.catturati[nome] = (_stacca(a), _stacca(k))
            if self.attivo:
                inizio = self.orologio.marca()
            out = fn(*a, **k)
            if self.attivo:
                self.marche.setdefault(nome, []).append([inizio, self.orologio.marca()])
            return out
        return involucro

    def togli(self) -> None:
        for h in self._maniglie:
            h.remove()
        for f, fn in self._originali.items():
            setattr(self.M, f, fn)

    def mediane_ms(self) -> dict:
        self.orologio.sincronizza()
        return {n: statistics.median(self.orologio.durata_ms(a, b) for a, b in v)
                for n, v in self.marche.items() if v and all(b is not None for _, b in v)}


def _stacca(x):
    import torch
    if isinstance(x, torch.Tensor):
        return x.detach()
    if isinstance(x, tuple):
        return tuple(_stacca(v) for v in x)
    if isinstance(x, list):
        return [_stacca(v) for v in x]
    if isinstance(x, dict):
        return {k: _stacca(v) for k, v in x.items()}
    return x


# ══════════════════════ 2. blocchi isolati: avanti e avanti+indietro ══════════════════════
def _con_ingresso_grad(args, kwargs, tutti_i_float: bool = False):
    """Copia (args, kwargs) con l'ingresso principale (hidden_states o primo argomento
    tensore) sostituito da una foglia che chiede il gradiente. `tutti_i_float`: TUTTI i
    tensori in virgola mobile (regola delta: q, k, v, g, beta derivano tutti dall'ingresso)."""
    import torch
    a, k = list(args), dict(kwargs)
    if tutti_i_float:
        a = [v.detach().requires_grad_(True) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
             for v in a]
        k = {n: (v.detach().requires_grad_(True) if isinstance(v, torch.Tensor) and v.is_floating_point() else v)
             for n, v in k.items()}
        return a, k
    if "hidden_states" in k and isinstance(k["hidden_states"], torch.Tensor):
        k["hidden_states"] = k["hidden_states"].detach().requires_grad_(True)
    else:
        for i, v in enumerate(a):
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                a[i] = v.detach().requires_grad_(True)
                break
    return a, k


def _uscita(o):
    return o[0] if isinstance(o, tuple) else o


def misura_isolato(orologio: Orologio, chiamabile, args, kwargs, ripetizioni: int,
                   pesi_con_grad: "list | None" = None, tutti_i_float: bool = False) -> dict:
    """Avanti (no_grad) e avanti+indietro (ingresso con grad; `pesi_con_grad` in piu')."""
    import torch
    pesi_con_grad = pesi_con_grad or []
    with torch.no_grad():
        riferimento = _uscita(chiamabile(*args, **kwargs))
    g = torch.randn_like(riferimento)

    def avanti():
        with torch.no_grad():
            chiamabile(*args, **kwargs)

    def avanti_indietro():
        a, k = _con_ingresso_grad(args, kwargs, tutti_i_float)
        _uscita(chiamabile(*a, **k)).backward(g)
        for p in pesi_con_grad:
            p.grad = None

    stato = [(p, p.requires_grad) for p in pesi_con_grad]
    for p, _ in stato:
        p.requires_grad_(True)
    try:
        t_a = orologio.misura(avanti, ripetizioni)
        t_ai = orologio.misura(avanti_indietro, ripetizioni)
    finally:
        for p, r in stato:
            p.requires_grad_(r)
            p.grad = None
    return {"avanti_ms": t_a, "avanti_indietro_ms": t_ai, "indietro_ms": t_ai - t_a}


# ══════════════════════ 3. per-peso: nucleo, norma, AdamW ══════════════════════
def misura_per_peso(orologio: Orologio, V, strato, S, ripetizioni: int) -> dict:
    import torch
    from finestra_globale import GRUPPO_DOWN_REALE, GRUPPO_GATEUP_REALE, _group_giocattolo

    esperti = strato.mlp.experts
    g_gu = _group_giocattolo(esperti.gate_up_proj.shape[-1], GRUPPO_GATEUP_REALE)
    g_dw = _group_giocattolo(esperti.down_proj.shape[-1], GRUPPO_DOWN_REALE)
    dev = esperti.gate_up_proj.device
    t_gu = V._costruisci_ternari_layer(esperti.gate_up_proj, S, g_gu, ruota=True).to(dev)
    t_dw = V._costruisci_ternari_layer(esperti.down_proj, S, g_dw, ruota=False).to(dev)
    bf16 = esperti.gate_up_proj.dtype == torch.bfloat16
    beta = float(V.NUCLEO_BETA_MAX)
    with torch.no_grad():
        o_gu = t_gu.forward_duro(beta, uscita_bf16=bf16)
        o_dw = t_dw.forward_duro(beta, uscita_bf16=bf16)
    g_gu_out, g_dw_out = torch.randn_like(o_gu), torch.randn_like(o_dw)
    del o_gu, o_dw
    parametri = list(t_gu.parameters()) + list(t_dw.parameters())

    def avanti():
        with torch.no_grad():
            t_gu.forward_duro(beta, uscita_bf16=bf16)
            t_dw.forward_duro(beta, uscita_bf16=bf16)

    def avanti_indietro():
        for p in parametri:
            p.grad = None
        t_gu.forward_duro(beta, uscita_bf16=bf16).backward(g_gu_out)
        t_dw.forward_duro(beta, uscita_bf16=bf16).backward(g_dw_out)

    risultato = {"n_parametri": sum(p.numel() for p in parametri),
                 "nucleo_avanti_ms": orologio.misura(avanti, ripetizioni),
                 "nucleo_avanti_indietro_ms": orologio.misura(avanti_indietro, ripetizioni)}
    risultato["nucleo_indietro_ms"] = risultato["nucleo_avanti_indietro_ms"] - risultato["nucleo_avanti_ms"]
    avanti_indietro()   # gradienti presenti per norma e AdamW

    def norma():   # identica a passo_ottimizzazione(salta_non_finiti=True)
        grad = [p.grad for p in parametri if p.grad is not None]
        n = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(x) for x in grad]))
        bool(torch.isfinite(n))

    risultato["norma_gradienti_ms"] = orologio.misura(norma, ripetizioni)
    gruppi = {"z": [], "log_s": [], "delta_raw": []}
    for m in (t_gu, t_dw):
        for nome, p in m.named_parameters():
            gruppi[nome].append(p)
    opt = torch.optim.AdamW([{"params": ps, "lr": 0.01} for ps in gruppi.values() if ps],
                            lr=0.01, weight_decay=0.0, fused=dev.type == "cuda")
    risultato["adamw_fuso_ms"] = orologio.misura(opt.step, ripetizioni)
    del opt, t_gu, t_dw, parametri
    return risultato


# ══════════════════════ strato: tutte le misure ══════════════════════
def radiografa_strato(V, M, strato, cfg, h, orologio: Orologio, ripetizioni: int, per_peso: bool) -> dict:
    import torch
    from transformers.masking_utils import create_causal_mask
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding

    dev = h.device
    b_loc, l_seq = h.shape[:2]
    rot = Qwen3_5MoeTextRotaryEmbedding(cfg).to(dev)
    pos = torch.arange(l_seq, device=dev).unsqueeze(0).expand(b_loc, -1)
    pos_emb = rot(torch.zeros_like(h), pos)          # stessa forma/dtype del flusso (lezione RoPE dtype)
    maschera = create_causal_mask(config=cfg, inputs_embeds=h, attention_mask=None,
                                  past_key_values=None, position_ids=pos)
    S = torch.from_numpy(V._S1024_NP).to(dev)
    esperti = strato.mlp.experts

    def strato_avanti(x):
        # stessa chiamata di ottimizza_finestra_JOINT::_forward_grezzo, con l'ingresso ruotato
        # degli esperti come nella forgia (THOR_ROTAZIONE_INGRESSO)
        with V._ctx_rotazione(esperti, S):
            return _uscita(strato(hidden_states=x, position_embeddings=pos_emb, attention_mask=maschera,
                                  position_ids=pos))

    ganci = Ganci(strato, orologio, M)
    try:
        with torch.no_grad():
            for _ in range(2):                         # riscaldo + cattura degli ingressi
                strato_avanti(h)
            orologio.sincronizza()
            ganci.attivo = True
            t_strato = []
            for _ in range(ripetizioni):
                a = orologio.marca()
                strato_avanti(h)
                b = orologio.marca()
                orologio.sincronizza()
                t_strato.append(orologio.durata_ms(a, b))
            ganci.attivo = False
        dentro = ganci.mediane_ms()
        dentro["STRATO INTERO"] = statistics.median(t_strato)
        catturati = ganci.catturati
    finally:
        ganci.togli()

    isolati = {"STRATO INTERO (ingresso con grad)": misura_isolato(
        orologio, lambda hidden_states: strato_avanti(hidden_states), (), {"hidden_states": h}, ripetizioni)}
    for nome in ("linear_attn", "self_attn", "mlp.shared_expert", "post_attention_layernorm",
                 "input_layernorm", "causal_conv1d_fn", "torch_chunk_gated_delta_rule"):
        if nome not in catturati:
            continue
        args, kwargs = catturati[nome]
        if nome in Ganci.FUNZIONI:
            chiamabile = getattr(M, nome)             # di nuovo l'originale (ganci tolti)
        else:
            chiamabile = strato.get_submodule(nome)
        try:
            isolati[nome] = misura_isolato(orologio, chiamabile, args, kwargs, ripetizioni,
                                           tutti_i_float=nome == "torch_chunk_gated_delta_rule")
        except Exception as e:  # noqa: BLE001 — una misura mancata non deve fermare le altre
            isolati[nome] = {"errore": f"{type(e).__name__}: {e}"[:300]}
    if "mlp.experts" in catturati:
        args, kwargs = catturati["mlp.experts"]

        def esperti_ruotati(*a, **k):
            with V._ctx_rotazione(esperti, S):
                return esperti(*a, **k)
        isolati["mlp.experts (con dW, come nel passo)"] = misura_isolato(
            orologio, esperti_ruotati, args, kwargs, ripetizioni,
            pesi_con_grad=[esperti.gate_up_proj, esperti.down_proj])

    out = {"tipo": getattr(strato, "block_type", "?"), "token": int(b_loc * l_seq),
           "avanti_dentro_lo_strato_ms": dentro, "isolati": isolati}
    if per_peso:
        out["per_peso"] = misura_per_peso(orologio, V, strato, S, ripetizioni)
    return out


def stampa(L, r: dict) -> None:
    print(f"\n══ strato {L} ({r['tipo']}) · {r['token']} token · mediane ══")
    print("  AVANTI dentro lo strato vero (ganci; i sottoblocchi sono inclusi nei loro genitori):")
    for n, v in sorted(r["avanti_dentro_lo_strato_ms"].items(), key=lambda kv: -kv[1]):
        print(f"    {v:9.2f} ms  {n}")
    print("  BLOCCHI ISOLATI (avanti no_grad · avanti+indietro con ingresso che chiede grad · differenza):")
    for n, v in r["isolati"].items():
        if "errore" in v:
            print(f"    ⛔ {n}: {v['errore']}")
        else:
            print(f"    {v['avanti_ms']:9.2f} · {v['avanti_indietro_ms']:9.2f} · indietro {v['indietro_ms']:9.2f} ms  {n}")
    if "per_peso" in r:
        p = r["per_peso"]
        print(f"  PER-PESO (gate_up+down di questo strato, {p['n_parametri']/1e9:.3f} G parametri):")
        for k in ("nucleo_avanti_ms", "nucleo_indietro_ms", "nucleo_avanti_indietro_ms",
                  "norma_gradienti_ms", "adamw_fuso_ms"):
            print(f"    {p[k]:9.2f} ms  {k}")


# ══════════════════════ strati giocattolo (collaudo su CPU) ══════════════════════
def strati_giocattolo(V):
    import torch
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer

    cfg = Qwen3_5MoeTextConfig(
        vocab_size=64, hidden_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, moe_intermediate_size=32, shared_expert_intermediate_size=32, num_experts=8,
        num_experts_per_tok=2, layer_types=["linear_attention", "full_attention"], partial_rotary_factor=1.0,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
        linear_conv_kernel_dim=4)
    cfg._attn_implementation = "sdpa"
    cfg._experts_implementation = "grouped_mm"
    if os.getenv("THOR_ESPERTI_THOR", "0") == "1":
        import esperti_thor
        esperti_thor.registra()
        cfg._experts_implementation = esperti_thor.NOME
    strati = {}
    for L in range(2):
        s = Qwen3_5MoeDecoderLayer(cfg, L)
        V._init_pesi_giocattolo(s, seed=L)
        for p in s.parameters():
            p.requires_grad_(False)
        strati[L] = s.eval()
    emb = torch.randn(cfg.vocab_size, cfg.hidden_size) * 0.5
    ids = torch.randint(0, cfg.vocab_size, (2, 32), generator=torch.Generator().manual_seed(0))
    return cfg, emb, strati, ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--strati", default="2,3", help="strati veri da misurare (es. 2,3: uno lineare e uno pieno)")
    ap.add_argument("--ripetizioni", type=int, default=10)
    ap.add_argument("--senza-per-peso", action="store_true", help="salta nucleo/norma/AdamW")
    ap.add_argument("--ids-casuali", action="store_true", help="token casuali invece del corpus di calibrazione")
    ap.add_argument("--giocattolo", action="store_true", help="collaudo su CPU con strati finti")
    ap.add_argument("--json", default="", help="dove scrivere i risultati (facoltativo)")
    args = ap.parse_args()

    if not args.giocattolo:
        altri = gpu_tenuta_da_altri()      # PRIMA di toccare CUDA
        if altri:
            print(f"⛔ GPU tenuta da altri processi: {altri} — non misuro (una corsia pesante per volta)")
            return 3

    import torch
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as M
    import valida_finestra_8_12 as V   # sola lettura

    risultati = {"quando": time.strftime("%F %T"), "giocattolo": args.giocattolo,
                 "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("THOR_")},
                 "fla_attiva": any("fla" in getattr(c.cell_contents, "__module__", "")
                                   for c in (M.torch_chunk_gated_delta_rule.__closure__ or ())
                                   if callable(c.cell_contents)),
                 "strati": {}}
    if args.giocattolo:
        orologio = Orologio(torch, su_gpu=False)
        cfg, emb, strati, ids = strati_giocattolo(V)
        for L, strato in strati.items():
            h = torch.nn.functional.embedding(ids, emb)
            r = radiografa_strato(V, M, strato, cfg, h, orologio, max(2, min(args.ripetizioni, 3)),
                                  per_peso=not args.senza_per_peso)
            risultati["strati"][L] = r
            stampa(L, r)
    else:
        assert torch.cuda.is_available(), "serve la GPU (o --giocattolo)"
        orologio = Orologio(torch, su_gpu=True)
        for L in [int(x) for x in args.strati.split(",") if x.strip()]:
            cfg, emb, puliti, quant = V.carica_finestra_corrente([L])
            del puliti                             # il passo lavora sul ramo corrente
            if args.ids_casuali:
                ids = torch.randint(0, emb.shape[0], (V.BATCH_SEQ, V.L_SEQ),
                                    generator=torch.Generator().manual_seed(0))
            else:
                ids = V.carica_batch_calibrazione()[0][0]
            h = torch.nn.functional.embedding(ids.to(emb.device), emb).to(emb.dtype)
            r = radiografa_strato(V, M, quant[L], cfg, h, orologio, args.ripetizioni,
                                  per_peso=not args.senza_per_peso)
            r["vram_picco_gib"] = torch.cuda.max_memory_allocated() / 2**30
            risultati["strati"][L] = r
            stampa(L, r)
            del quant, emb, h
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    print(f"\nfla attiva: {risultati['fla_attiva']}")
    if args.json:
        Path(args.json).write_text(json.dumps(risultati, indent=1, ensure_ascii=False, default=str))
        print(f"risultati in {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

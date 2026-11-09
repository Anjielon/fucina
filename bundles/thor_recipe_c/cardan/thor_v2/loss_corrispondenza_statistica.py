#!/usr/bin/env python3
"""Loss di corrispondenza statistica (feature-statistics matching), da
arXiv:2607.14630 ("Cross-Layer Error Compensation and Finite-Sample
Feature-Statistics Matching for Extreme Low-Bit Quantization"), §Eq.6+Eq.9.

Nato 24/9/2026 02:2x, ordine del workflow dopo il depth-sweep (`PIANO.md`,
02:00): il MUTO da 11 strati composti in su e' l'errore di quantizzazione
sequenziale che si compone con la profondita' (Lemma 1 del paper,
e_{l+1} = A_l(e_l) + q_l) — mai compensato dall'obiettivo per-strato attuale
(GPTQ sequenziale a forma chiusa, teacher-reconstruction contro BF16, vedi
`schurquant_riconduzione.py`). La via completa (Angelo, "sempre e comunque
la via completa", niente approssimazioni) e' un'ottimizzazione GLOBALE
congiunta:

    Eq.9   J = λ_E·L_E + λ_GE·L_GE + λ_D·L_KD

  - L_E  = errore accumulato ESATTO (Lemma 1, differenza finita) — NON in
    questo file.
  - L_GE = QUESTO file: media di D su un insieme di strati S ("ogni L/4
    blocchi + lo strato finale").
  - L_KD = distillazione classica sull'uscita finale — NON in questo file.

Formule esatte (Eq.6, estratte dal testo del paper via WebFetch il 24/9,
non dal solo abstract — la formula del kernel in particolare NON e' ovvia:
verificarla contro il paper prima di ritoccarla):

    P  = (H  - μ)  R         P̂ = (Ĥ - μ̂) R         R ∈ R^(d×k) FISSA,
                                                     R_ij ~ N(0, 1/k)
    D_μ = ‖μ̂ - μ‖² / (‖μ‖² + ε)
    D_C = ‖Ĉ - C‖²_F / (‖C‖²_F + ε)      C  = (1/(n-1)) Pᵀ P     (k×k)
    D_K = ‖K̂ - K‖²_F / (‖K‖²_F + ε)      K  = (1/d) (H-μ)(H-μ)ᵀ  (n×n)

H (senza cappello) = branch di RIFERIMENTO (bersaglio, teacher, BF16 pulito
— il nostro `hs_ref`). Ĥ (con cappello) = branch CORRENTE in ottimizzazione
(il nostro `hs_q`, o — nel metodo globale futuro — l'uscita del forward
soft/hard con i codici latenti z). R e μ/μ̂ sono SEMPRE la stessa proiezione
fissa e le medie proprie di ciascun branch, mai scambiate.

⛔ Cosa fa QUESTO file, e cosa NO — per rispettare il vincolo del workflow
(GPU condivisa, NIENTE modello vero 35B, NIENTE training reale):
  - FA: le tre funzioni D_μ/D_C/D_K + il loro aggregato L_GE, la scelta
    dell'insieme di strati S, un test numerico CPU su tensori sintetici
    piccoli (mai il modello vero) che verifica la matematica.
  - NON fa: L_E (Lemma 1), l'annealing tanh/sign dei codici latenti, il loop
    AdamW su 1000 passi, il caricamento di Thor — quello resta "lavoro di
    giorni" pianificato in `PIANO.md`, non improvvisato qui.

Integrazione futura (quando si costruisce il loop AdamW globale): le coppie
H_rif/H_hat per strato sono ESATTAMENTE `hs_ref[b]`/`hs_q[b]` che
`cattura_schur_esperti.py::FlussoSuDisco` gia' produce (stesso corpus, stessi
token, stesso ordine) — appiattite a (n_tok, hidden_size). Zero cattura
nuova da scrivere: la statistica matching loss legge quello che gia' c'e'.

Iperparametri del paper (Eq.6/Eq.9, per riferimento futuro, non usati qui):
k=64 (dimensione proiezione), ε implicito piccolo, layer set S = "ogni L/4
blocchi + il finale", λ_E/λ_GE/λ_D pesi del joint loss, AdamW lr 5e-4 (z) /
5e-3 (log s), β 1→16 esponenziale, hard-forward nell'ultimo 20%, 1000 passi
× batch 8 × lunghezza 512.
"""
from __future__ import annotations

import time

import torch

# k del paper (Eq.6): dimensione della proiezione casuale per il termine di
# covarianza. Non hardcodato nelle funzioni sotto (e' un parametro), questo
# e' solo il default dichiarato dal paper.
K_PROIEZIONE_DEFAULT = 64
EPS_DEFAULT = 1e-8


# ───────────────────────── proiezione casuale (Eq.6) ─────────────────────
def crea_proiezione_casuale(
    d: int, k: int = K_PROIEZIONE_DEFAULT, seed: int = 0,
    dtype: torch.dtype = torch.float32, device: str | torch.device = "cpu",
) -> torch.Tensor:
    """R ∈ R^(d×k), R_ij ~ N(0, 1/k) — FISSA (stesso seed -> stessa matrice).

    ⛔ Il paper la vuole fissa per TUTTI i passi di AdamW di un dato strato:
    ricrearla ad ogni passo trasformerebbe la loss in un bersaglio mobile.
    Un generatore locale (non lo stato RNG globale) rende la creazione
    riproducibile senza toccare il seed di nessun altro codice del progetto.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    R = torch.randn(d, k, generator=g, dtype=torch.float32) * (k ** -0.5)
    return R.to(device=device, dtype=dtype)


def _centra(H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(H centrato, media) — media sull'asse dei token (dim 0). H: (n, d)."""
    mu = H.mean(dim=0)
    return H - mu, mu


# ───────────────────────────── D_μ, D_C, D_K (Eq.6) ───────────────────────
def d_media(H_rif: torch.Tensor, H_hat: torch.Tensor, eps: float = EPS_DEFAULT) -> torch.Tensor:
    """D_μ = ‖μ̂-μ‖² / (‖μ‖²+ε). H_rif/H_hat: (n, d) — n puo' differire fra i due."""
    mu = H_rif.mean(dim=0)
    mu_hat = H_hat.mean(dim=0)
    num = (mu_hat - mu).pow(2).sum()
    den = mu.pow(2).sum() + eps
    return num / den


def d_covarianza(
    H_rif: torch.Tensor, H_hat: torch.Tensor, R: torch.Tensor, eps: float = EPS_DEFAULT,
) -> torch.Tensor:
    """D_C = ‖Ĉ-C‖²_F / (‖C‖²_F+ε), C=(1/(n-1))PᵀP, P=(H-μ)R.

    R e' la STESSA matrice per i due branch (invariante non verificabile qui
    a runtime — e' responsabilita' del chiamante passare la R del layer
    giusto, creata una volta con `crea_proiezione_casuale` e riusata).
    """
    Hc_rif, _ = _centra(H_rif)
    Hc_hat, _ = _centra(H_hat)
    n_rif = H_rif.shape[0]
    n_hat = H_hat.shape[0]
    if n_rif < 2 or n_hat < 2:
        raise ValueError(f"d_covarianza: servono almeno 2 token per branch (n_rif={n_rif}, n_hat={n_hat})")
    P_rif = Hc_rif @ R
    P_hat = Hc_hat @ R
    C = (P_rif.transpose(-2, -1) @ P_rif) / (n_rif - 1)
    C_hat = (P_hat.transpose(-2, -1) @ P_hat) / (n_hat - 1)
    num = (C_hat - C).pow(2).sum()
    den = C.pow(2).sum() + eps
    return num / den


def d_kernel(H_rif: torch.Tensor, H_hat: torch.Tensor, eps: float = EPS_DEFAULT) -> torch.Tensor:
    """D_K = ‖K̂-K‖²_F / (‖K‖²_F+ε), K=(1/d)(H-μ)(H-μ)ᵀ — matrice n×n (fra
    TOKEN, non fra feature): richiede n_rif == n_hat, gli STESSI token nei
    due branch (esattamente `hs_ref`/`hs_q`: stesso corpus, stesso ordine).
    Nessuna proiezione qui: il paper la applica solo al termine di covarianza.
    """
    if H_rif.shape[0] != H_hat.shape[0]:
        raise ValueError(
            f"d_kernel: serve lo STESSO numero di token nei due branch "
            f"(n_rif={H_rif.shape[0]}, n_hat={H_hat.shape[0]}) — K confronta token con token"
        )
    d = H_rif.shape[-1]
    Hc_rif, _ = _centra(H_rif)
    Hc_hat, _ = _centra(H_hat)
    K = (Hc_rif @ Hc_rif.transpose(-2, -1)) / d
    K_hat = (Hc_hat @ Hc_hat.transpose(-2, -1)) / d
    num = (K_hat - K).pow(2).sum()
    den = K.pow(2).sum() + eps
    return num / den


def perdita_statistiche(
    H_rif: torch.Tensor, H_hat: torch.Tensor, R: torch.Tensor,
    pesi: tuple[float, float, float] = (1.0, 1.0, 1.0), eps: float = EPS_DEFAULT,
) -> dict[str, torch.Tensor]:
    """D per UN singolo strato: {"d_mu", "d_c", "d_k", "totale"}. `totale` e'
    la combinazione pesata (pesi = w_mu, w_C, w_K — il paper non ne dichiara
    di diversi da 1 esplicitamente per questo termine; restano un parametro,
    non una costante nascosta)."""
    w_mu, w_c, w_k = pesi
    dm = d_media(H_rif, H_hat, eps)
    dc = d_covarianza(H_rif, H_hat, R, eps)
    dk = d_kernel(H_rif, H_hat, eps)
    return {"d_mu": dm, "d_c": dc, "d_k": dk, "totale": w_mu * dm + w_c * dc + w_k * dk}


# ───────────────────────── insieme di strati S (Eq.9) ─────────────────────
def insieme_strati(num_strati: int, num_blocchi: int = 4) -> tuple[int, ...]:
    """Layer set S = "ogni L/4 blocchi + il finale" (testo del paper),
    tradotto in indici 0-based (convenzione di TUTTO il resto della pipeline:
    `cattura_schur_esperti.py` numera gli strati 0..num_strati-1).

    Il paper conta in BLOCCHI 1-based (dopo il blocco b, b=1..L): il layer
    set e' {round(L/nb), round(2L/nb), ..., L}. "Dopo il blocco b" (1-based)
    e' l'indice 0-based b-1. Per L=40, nb=4: blocchi {10,20,30,40} -> indici
    0-based {9,19,29,39}.

    ⚠️ Il ping del workflow aveva suggerito "10,20,30,39 circa" (un mix delle
    due convenzioni, l'ultimo gia' 0-based): qui la conversione e' fatta con
    UNA regola sola, dichiarata, non a occhio — la differenza di 1 sui primi
    tre non e' un dettaglio: sbagliare da che parte si conta sposta la sonda
    di uno strato intero.
    """
    if num_strati < 1:
        raise ValueError("num_strati deve essere >= 1")
    if num_blocchi < 1:
        raise ValueError("num_blocchi deve essere >= 1")
    blocchi_1based = {round(num_strati * i / num_blocchi) for i in range(1, num_blocchi + 1)}
    blocchi_1based.add(num_strati)  # "+ il finale", esplicito anche se num_blocchi non lo genera gia'
    indici_0based = sorted(b - 1 for b in blocchi_1based if 1 <= b <= num_strati)
    return tuple(indici_0based)


def perdita_ge(
    per_strato: dict[int, tuple[torch.Tensor, torch.Tensor]],
    proiezioni: dict[int, torch.Tensor],
    num_strati_totali: int,
    num_blocchi: int = 4,
    pesi: tuple[float, float, float] = (1.0, 1.0, 1.0),
    eps: float = EPS_DEFAULT,
) -> torch.Tensor:
    """L_GE (Eq.9) = media di D_totale sugli strati di `insieme_strati`.

    `per_strato[L] = (H_rif, H_hat)` e `proiezioni[L] = R` DEVONO coprire
    ogni strato dell'insieme S — se manca uno strato si SOLLEVA (mai una
    media silenziosa su un sottoinsieme: nasconderebbe uno strato saltato
    per errore, esattamente il tipo di bug che questa notte e' costato ore)."""
    S = insieme_strati(num_strati_totali, num_blocchi)
    mancanti = [L for L in S if L not in per_strato or L not in proiezioni]
    if mancanti:
        raise KeyError(f"perdita_ge: mancano H_rif/H_hat o R per gli strati {mancanti} (insieme S={S})")
    termini = [perdita_statistiche(*per_strato[L], proiezioni[L], pesi, eps)["totale"] for L in S]
    return torch.stack(termini).mean()


# ───────────────────────────────── test CPU ────────────────────────────────
def _test() -> bool:
    """Verifica numerica pura CPU, tensori sintetici PICCOLI (mai il modello
    vero — vincolo del workflow). Ogni asserzione e' un'identita' algebrica
    verificabile a mano, non una soglia arbitraria."""
    torch.manual_seed(0)
    ok = True

    def check(nome: str, cond: bool, dettaglio: str = "") -> None:
        nonlocal ok
        simbolo = "✅" if cond else "⛔"
        print(f"{simbolo} {nome}" + (f" — {dettaglio}" if dettaglio else ""))
        ok = ok and cond

    # ---- 1. identita': stesso branch -> D tutti ~0 ----
    n, d, k = 64, 16, 4
    H = torch.randn(n, d, dtype=torch.float64)
    R = crea_proiezione_casuale(d, k, seed=1, dtype=torch.float64)
    dic = perdita_statistiche(H, H.clone(), R)
    check("identita': D_mu=D_C=D_K=0 quando i due branch coincidono",
          dic["d_mu"].item() < 1e-20 and dic["d_c"].item() < 1e-20 and dic["d_k"].item() < 1e-20,
          f"d_mu={dic['d_mu'].item():.3e} d_c={dic['d_c'].item():.3e} d_k={dic['d_k'].item():.3e}")

    # ---- 2. shift SOLO di media: D_mu>0, D_C=D_K~0 (il centraggio li assorbe) ----
    shift = torch.randn(d, dtype=torch.float64) * 5.0
    H_hat = H + shift
    dic = perdita_statistiche(H, H_hat, R)
    check("shift di media: D_mu > 1e-3 (rilevato)", dic["d_mu"].item() > 1e-3,
          f"d_mu={dic['d_mu'].item():.3e}")
    check("shift di media: D_C < 1e-20 (il centraggio lo annulla)", dic["d_c"].item() < 1e-20,
          f"d_c={dic['d_c'].item():.3e}")
    check("shift di media: D_K < 1e-20 (il centraggio lo annulla)", dic["d_k"].item() < 1e-20,
          f"d_k={dic['d_k'].item():.3e}")

    # ---- 3. scala della VARIANZA (media zero esatta): D_C=D_K=(scale^2-1)^2 ESATTO ----
    H0 = H - H.mean(dim=0)  # media 0 esatta, cosi' la scala non tocca D_mu
    scale = 1.5
    H0_hat = H0 * scale
    atteso = (scale ** 2 - 1) ** 2
    dic = perdita_statistiche(H0, H0_hat, R)
    check("scala di varianza: D_mu ~ 0 (la scala non sposta la media)", dic["d_mu"].item() < 1e-20,
          f"d_mu={dic['d_mu'].item():.3e}")
    check("scala di varianza: D_C == (scale^2-1)^2 esatto (identita' algebrica)",
          abs(dic["d_c"].item() - atteso) < 1e-9,
          f"D_C={dic['d_c'].item():.10f} atteso={atteso:.10f}")
    check("scala di varianza: D_K == (scale^2-1)^2 esatto (stessa identita', K indipendente da C)",
          abs(dic["d_k"].item() - atteso) < 1e-9,
          f"D_K={dic['d_k'].item():.10f} atteso={atteso:.10f}")

    # ---- 4. gradiente: la loss deve essere differenziabile end-to-end ----
    H_hat_grad = (H + torch.randn(n, d, dtype=torch.float64) * 0.1).requires_grad_(True)
    dic = perdita_statistiche(H, H_hat_grad, R)
    dic["totale"].backward()
    grad_ok = (H_hat_grad.grad is not None and torch.isfinite(H_hat_grad.grad).all()
               and H_hat_grad.grad.abs().sum().item() > 0)
    check("differenziabile: gradiente finito e non-nullo su H_hat", grad_ok)

    # ---- 5. n diversi fra i due branch: D_mu/D_C devono accettarlo, D_K deve rifiutarlo ----
    H_hat_corto = H_hat[:n - 5].clone()
    try:
        perdita_statistiche(H, H_hat_corto, R)
        check("D_K rifiuta n diversi fra i branch", False, "non ha sollevato ValueError")
    except ValueError:
        check("D_K rifiuta n diversi fra i branch", True)
    dm = d_media(H, H_hat_corto).item()
    dc = d_covarianza(H, H_hat_corto, R).item()
    check("D_mu/D_C accettano n diversi fra i branch (non serve pairing token-per-token)",
          dm >= 0 and dc >= 0, f"d_mu={dm:.3e} d_c={dc:.3e}")

    # ---- 6. insieme_strati: convenzione dichiarata, casi noti a mano ----
    S40 = insieme_strati(40, 4)
    check("insieme_strati(40,4) == (9,19,29,39) — convenzione 0-based dichiarata",
          S40 == (9, 19, 29, 39), f"ottenuto {S40}")
    S8 = insieme_strati(8, 4)
    check("insieme_strati(8,4) == (1,3,5,7)", S8 == (1, 3, 5, 7), f"ottenuto {S8}")
    S_piccolo = insieme_strati(3, 4)  # L < num_blocchi: niente duplicati, niente indici fuori range
    check("insieme_strati(3,4): nessun duplicato, tutti < 3", len(S_piccolo) == len(set(S_piccolo))
          and all(0 <= i < 3 for i in S_piccolo), f"ottenuto {S_piccolo}")

    # ---- 7. perdita_ge: media sull'insieme S, e SOLLEVA se manca uno strato ----
    per_strato = {L: (H, H + torch.randn(n, d, dtype=torch.float64) * 0.05) for L in S40}
    proiezioni = {L: R for L in S40}
    tot = perdita_ge(per_strato, proiezioni, num_strati_totali=40, num_blocchi=4)
    check("perdita_ge: scalare finito >= 0 su tutto l'insieme S", torch.isfinite(tot) and tot.item() >= 0,
          f"L_GE={tot.item():.3e}")
    del per_strato[S40[0]]
    try:
        perdita_ge(per_strato, proiezioni, num_strati_totali=40, num_blocchi=4)
        check("perdita_ge solleva se uno strato di S manca", False, "non ha sollevato KeyError")
    except KeyError:
        check("perdita_ge solleva se uno strato di S manca", True)

    # ---- 8. alla dimensione REALE di Thor (hidden=2048): la proiezione conta ----
    # Non e' il modello vero (nessun peso caricato, nessun forward): solo la
    # dimensionalita' vera, per rispondere alla domanda "serve davvero la
    # proiezione a rango basso o basta la covarianza piena?" con un numero
    # MISURATO, non stimato (regola di progetto: mai stimare, misurare).
    d_vero, n_vero, k_vero = 2048, 256, K_PROIEZIONE_DEFAULT
    H_vero = torch.randn(n_vero, d_vero, dtype=torch.float32)
    H_vero_hat = H_vero + torch.randn(n_vero, d_vero, dtype=torch.float32) * 0.02
    R_vero = crea_proiezione_casuale(d_vero, k_vero, seed=2)

    t0 = time.perf_counter()
    dc_proiettata = d_covarianza(H_vero, H_vero_hat, R_vero).item()
    t_proiettata = time.perf_counter() - t0

    def _d_covarianza_piena(H_rif, H_hat, eps=EPS_DEFAULT):
        Hc_rif, _ = _centra(H_rif)
        Hc_hat, _ = _centra(H_hat)
        n_r, n_h = H_rif.shape[0], H_hat.shape[0]
        C = (Hc_rif.T @ Hc_rif) / (n_r - 1)
        C_hat = (Hc_hat.T @ Hc_hat) / (n_h - 1)
        return (((C_hat - C) ** 2).sum() / (C.pow(2).sum() + eps)).item()

    t0 = time.perf_counter()
    dc_piena = _d_covarianza_piena(H_vero, H_vero_hat)
    t_piena = time.perf_counter() - t0

    rapporto_elementi = (k_vero * k_vero) / (d_vero * d_vero)
    check(f"proiezione: matrice {k_vero}x{k_vero} e' {rapporto_elementi*100:.3f}% degli elementi "
          f"della piena {d_vero}x{d_vero}", rapporto_elementi < 0.01,
          f"proiettata {t_proiettata*1000:.2f}ms (D_C={dc_proiettata:.3e}) vs "
          f"piena {t_piena*1000:.2f}ms (D_C={dc_piena:.3e})")

    # ---- 9. (opzionale) fumo GPU minuscolo — SOLO se disponibile, secondi, libera subito ----
    if torch.cuda.is_available():
        dg, ng, kg = 32, 16, 8
        Hg = torch.randn(ng, dg, device="cuda")
        Hg_hat = Hg + torch.randn(ng, dg, device="cuda") * 0.1
        Rg = crea_proiezione_casuale(dg, kg, seed=3, device="cuda")
        dic_gpu = perdita_statistiche(Hg, Hg_hat, Rg)
        check("fumo GPU: scalari finiti", all(torch.isfinite(v).all() for v in dic_gpu.values()))
        del Hg, Hg_hat, Rg, dic_gpu
        torch.cuda.empty_cache()
    else:
        print("↷ CUDA non disponibile su questa macchina (torch cpu-only, atteso: la forgia gira sui pod) — salto il fumo GPU")

    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _test() else 1)

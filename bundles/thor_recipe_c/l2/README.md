# MMLU-free letter-KD (E1) e rimozione della direzione di bias stile BNP (E2)

Nessun dato MMLU e nessun set di `auxiliary_train` (ARC, OpenBookQA, RACE, MCTest) entra
nell'addestramento. Le fonti sono SciQ e CommonsenseQA, più QASC, MedMCQA, MathQA e LogiQA
se le chiedi. Ogni item viene decontaminato con i 13-grammi contro **tutti** gli split di
`cais/mmlu` (auxiliary_train, dev, validation, test), con in più un controllo di uguaglianza
esatta sulla domanda normalizzata per le domande corte.

Dipendenze: `numpy pyarrow torch` (+ `transformers` per il backend HF, `gguf` per leggere
lm_head da un GGUF). `datasets` non serve: i parquet si scaricano con il loro sha di commit.

| file | cosa fa |
|---|---|
| `common.py` | formato del prompt MMLU, rotazioni, download a revisione fissata, hash dei 13-grammi |
| `e1_build_data.py` | fonti → 4 opzioni → decontaminazione → 4 rotazioni cicliche → jsonl |
| `teacher_client.py` | client llama-server: logprob di " A".." D" (id ricavati in contesto) |
| `e1_letter_kd.py` | perdita KL(maestro‖studente) + λ·JS fra le rotazioni, dataset/collate/sampler, selezione dei parametri |
| `e2_collect_hidden.py` | h = ingresso di lm_head all'ultima posizione (HF hook / llama-server / llama-embedding) |
| `e2_bnp_edit.py` | b = μ(h\|sbagliata) − μ(h\|giusta), modifica BNP (top-k) o rank-1, ri-ternarizzazione K piani g=256, valutazione |

## Comandi

```bash
# E1.1 dati (le revisioni sono fissate nel codice)
python3 e1_build_data.py --out data/items.jsonl \
    --sources sciq,csqa,qasc,medmcqa,mathqa,logiqa \
    --max-per-source 20000 --max-heldout-per-source 1000 --cache ~/.cache/mmlu_free

# E1.2 maestro (ricorda: il contesto si divide per --parallel)
llama-server -m TEACHER.gguf --port 8091 -c 16384 --parallel 4 -ngl 999 --no-webui
python3 teacher_client.py --server http://127.0.0.1:8091 \
    --in data/items.jsonl --out data/items_teacher.jsonl --workers 4
#   -> stampa gli id delle lettere (servono a E2) e la massa media delle 4 lettere

# E1.3 perdita: autotest, poi innesto nel tuo trainer (vedi la docstring di e1_letter_kd.py)
python3 e1_letter_kd.py --selftest
#   nel trainer: loss, st = letter_kd_step(model, batch, letter_ids, lam=0.5, T=1.0, target="raw"|"debiased")
#   parametri:   select_trainable_params(model, last_n=N, scale_regex=r"<nomi delle tue scale>")

# E2.1 stati nascosti dello STUDENTE (gruppi interi da 4 rotazioni, train + heldout)
python3 e2_collect_hidden.py --backend cli --cli-bin /percorso/llama-embedding -m STUDENT.gguf \
    --in data/items_teacher.jsonl --out data/hid --max-groups 3000 --max-groups-heldout 1000
#   oppure: llama-server -m STUDENT.gguf --embeddings --pooling last --port 8092 ; --backend server
#   oppure: --backend hf --hf-model /dir/modello --device cuda

# E2.2 direzione, modifica, ri-ternarizzazione, valutazione prima/dopo
python3 e2_bnp_edit.py all --hidden data/hid --w gguf:STUDENT.gguf \
    --letter-ids 319,350,315,360 --method rank1 --planes 3 --out out/rank1 --save-planes
python3 e2_bnp_edit.py all --hidden data/hid --w gguf:STUDENT.gguf \
    --letter-ids 319,350,315,360 --method bnp --k 32 --align signed --planes 3 --out out/bnp
#   --rows letters : ternarizza solo le 4 righe lettera. Le scale sono per riga, quindi la
#                    misura è identica a quella fatta su tutte le righe, e ci mette pochi secondi.
#   report.json    : direction, recon_edited/recon_original (rel_err_fro, mean_row_cos,
#                    letter_rows_rel_err, bias_leak_ratio) ed eval_heldout per
#                    original / original_ternary / edited_float / edited_ternary
```

## Avvertenze verificate durante il collaudo

- **Gli id delle lettere vanno ricavati in contesto.** Con SentencePiece (Llama),
  `tokenize(" A")` dà due token (`[259, 29909]`), ma dopo `Answer:` il token è uno solo
  (`▁A` = 319). Tutti e due gli script ricavano gli id come `tokenize(prompt+" A") − tokenize(prompt)`
  e si fermano se la differenza non è un token singolo.
- **llama-server `/embedding` normalizza in L2 di default**: la norma di h veniva 1,000.
  Il client manda `embd_normalize: -1`. Con llama-embedding serve `--embd-normalize -1`.
- **La valutazione da h è esatta in aritmetica float, ma non riproduce bit per bit
  llama.cpp.** Sul modello di prova (TinyLlama, CPU), i logprob ricostruiti come `W·h`
  differivano da quelli del server al massimo di 0,047 con lo stesso binario e di 0,09 con
  un altro binario. La causa sono le attivazioni quantizzate nel matmul di lm_head. Su un
  modello quasi piatto basta per cambiare l'argmax (6 casi su 8 concordi). Usala per
  scegliere fra le varianti. Il numero finale si misura con llama.cpp sul GGUF modificato.
  `--check-teacher` stampa questa differenza quando studente e maestro coincidono.
- **BNP (a) senza ri-ternarizzare:** azzerare la colonna i di lm_head equivale ad azzerare
  il peso i della RMSNorm finale. La modifica rank-1 (b) invece non entra nella norma:
  serve la ri-ternarizzazione. `bias_leak_ratio` misura quanta parte della direzione b
  rientra dopo la quantizzazione: per rank-1 in float è 0, e sul modello di prova con K=3
  tornava 0,15. Per BNP questa metrica non ha senso.
- **b mescola difficoltà e posizione**, perché "sbagliata contro giusta" non è "lettera
  preferita contro no". Va sempre guardato `acc` insieme a `RStd` e `fluctuation`.
  Sul modello di prova, BNP con segno ha fatto crescere il bias (RStd da 22 a 40):
  un modello giocattolo, ma è la ragione per cui esiste `--align abs`.
- Embedding legati (`output.weight` assente): per modificare lm_head bisogna prima slegarlo.
- `--softcap` (Gemma) e `--logit-scale` (Granite) servono solo se il modello ha queste trasformazioni dei logit.

## Tempi su una APU da 128 GB (Ryzen AI Max+ 395)

Misurati qui:
- **e1_build_data**, tutte e 6 le fonti, decontaminazione compresa: **2 min 27 s**
  (la prima volta scarica circa 50 MB di MMLU e costruisce l'indice; dopo legge la cache).
  Uscita: 19.731 domande × 4 = 78.924 righe, oro esattamente uniforme su A/B/C/D.
  Scartate per contaminazione: 23 (22 di QASC).
- **Prompt**: in media 95 token (tokenizer Llama). LogiQA è lungo circa 4,5 volte le altre fonti.
- **Ri-ternarizzazione** in CPU, K=3, 2 giri di raffinamento, matrice 32.000×2.048, originale
  e modificata insieme: **72 s**.

Stimati (da misurare: il client stampa righe/s ogni 500 righe):
- **Maestro**: righe × 95 token / velocità di prefill. Esempio: 160.000 righe a 1.000 tok/s
  di prefill fanno circa 4,2 h. Il token generato è uno solo, quindi conta solo il prefill.
- **Raccolta di h** per 4.000 gruppi (16.000 prompt): circa 1,5 M token, sui 25 min allo stesso prefill.
- **Ri-ternarizzazione di un lm_head 151.936×4.096**: circa 9,5 volte i 72 s misurati, cioè
  circa 11 min in CPU per originale + modificata, con ~12 GB di RAM. Con `--rows letters` bastano pochi secondi.
- **Allenamento letter-KD**: dipende dal tuo trainer. Si allenano 4 logit per prompt
  sull'ultima posizione, con `logits_to_keep=1`.

Una corsia pesante per volta: maestro, raccolta di h e training non vanno lanciati insieme sulla GPU.

## E1 legame lettera <-> contenuto (3/10 sera, ramo `cloud/lettera-e1`)

Dalla ricerca (`RICERCA_CLOUD_1820.md`): la lettera si decide in poche teste di attenzione degli strati CENTRALI
(2407.15018), quindi lm_head (E2) ed esperti FFN correggono il priore ma non il legame. E1 ora può allenare anche
**residui ternari di rango basso sull'attenzione**, con o senza gli esperti di recupero (`--modo esperti|residui|entrambi`).

| file | cosa fa |
|---|---|
| `residui_ternari.py` | `y = W x + diag(γ)·B_q(A_q x)`: A (r×in) ternario a gruppi da 256, B (out×r) ternario a una scala per riga, γ fuso in quella scala; STE, ri-ternarizzati a ogni passo; latenti nell'ordine del FILE (permutazioni di attn_qkv/attn_gate/ssm_out come `LineareTernaria`); γ = 0 ⇒ partenza al bit; init dal gradiente (SVD di Σ gᵀx accumulata, niente x/g in RAM) |
| `e1_perdita_legame.py` | KL maestro alla STESSA rotazione + λ·coerenza fra rotazioni riportata al contenuto + 0,3·PIF (negativi puntuali: contenuti sbagliati legati a tutte le lettere, 2406.01026); fonti vietate; metriche di selezione (acc, RStd, coerenza) per fonte |
| `e1_recupero_lettera.py` | driver: `--modo`, `--residui-strati 12-27 --residui-famiglie attn,gdn --residui-rango 32`, `--peso-pif`, selezione SOLO su held-out `--fonti-selezione logiqa,medmcqa` (rifiuta MMLU/ARC/OBQA/RACE/MCTest ovunque), `--criterio loss/kd/acc` |
| `e1_scrivi_residui_gguf.py` | GGUF nuovo = originale AL BYTE + tensori TQ1_0 in più (`blk.L.<fam>_lra/_lrb.weight`, + KV `thor.lettera.residui`); modifica nulla ⇒ stessa sha256 |
| `test_residui_ternari.py` | prove CPU autosufficienti (niente /mnt, niente GPU) |

Esportazione dei residui (scelta: tensori in più, non fusione). Per proiezione e strato:
`<fam>_lra` ne=(in, 256) con righe 0..r-1 = A_q e il resto zero; `<fam>_lrb` ne=(256, out) con colonne 0..r-1 = B_q
(scala fp16(s_riga·γ_riga)) e il resto zero. L'imbottitura a 256 è esatta (gli zeri non cambiano la scala ottima del
blocco, TQ1_0 rappresenta {-s,0,+s} senza errore). Il motore deve sommare `mul_mat(lrb, mul_mat(lra, x))` all'uscita
della proiezione (ramo da aggiungere in qwen35moe.cpp, come per REXP): llama.cpp di serie rifiuta tensori in più.
La fusione nei piani esistenti è scartata: W + BA non è ternario, e ri-ternarizzare un ritocco di rango basso lo
cancella quasi tutto (misurato su lm_head: la RTN toglieva il 99,7 % di un rank-1); un piano pieno nuovo
costerebbe la taglia di un piano e perderebbe l'esattezza.

```bash
# prove CPU (anche in cloud)
python3 test_residui_ternari.py && python3 e1_perdita_legame.py
# GPU (coda_lettera.sh lo fa da sé con E1_ARG): allenamento, poi GGUF REXP e GGUF REXP-RES
python e1_recupero_lettera.py --uscita OUT --dati items_teacher.jsonl --modo entrambi --strati 30-39 \
    --residui-strati 12-27 --residui-rango 32 --target raw --lam 0.5 --peso-pif 0.3 --passi 150
python3 e1_scrivi_residui_gguf.py --src Thor-T-REXP-LETTERA.gguf --out Thor-T-REXP-LETTERA-RES-LETTERA.gguf \
    --npz OUT/residui_tq1.npz --info OUT/residui.json --meta
python3 e1_scrivi_residui_gguf.py --src ST.gguf --out /dove/nulla.gguf --nulla     # collaudo: sha identica
```

Nota: la copia congelata `recupero_7dabd6f/` conteneva solo i `.py`; mancavano `mappa_completa_gguf_hf.json`
(dal commit 7dabd6f) e `mappa_completa_gguf_hf_perm.npz` (identico a `thor/`, sha 6114b1f4…): senza di loro
`carica_strato_gguf` non si importa e lo studente non si costruisce. Aggiunti; il preflight li controlla.

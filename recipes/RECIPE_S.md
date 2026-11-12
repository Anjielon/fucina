# Recipe S — released model Thor-T-35B-A3B-S (lineage B, 8.82 GiB): full provenance and re-run plan

> Thor-T (Ornith-1.5-35B-A3B, Qwen3.5-MoE architecture), ternary with 1–4 planes per tensor, BitNet convention
> (token embedding and output head Q8_0, norms and router F32). Released size **S** (the other released sizes are
> XS = C, see `RECIPE_C_D2.md`, and M = A). The released S file is **not** the raw output of training: it is the
> step-100 post-training checkpoint with its planes packed and its header cleaned (steps 11–12).
> Every statement points to a recorded file; nothing is inferred unless marked **[inferred]**.
> Runner: `run_recipe_s.py` (this folder). Hashes, manifests, launcher copies and logs: `recipe_s/recorded/`.
> Tests: `test_recipe_s.py`. Paths: `TT` = `/mnt/models/gguf/tony-tern`, `NAS` = `/mnt/nas/CACHEDEV1_DATA/modelli-fp`.

## 1. The released model

| item | value | evidence |
|---|---|---|
| file | `Thor-T-35B-A3B-S.gguf` | `pubblicazioni/LANCIO_FINALE_1102` |
| sha256 | `9d5afbf85afd0613dc67940c8c56e774790cc580f919a716445bd479cc6d65f9` | `recorded/release_s.sha256`; **regenerated bit for bit on 12/11/2026 from the T100 checkpoint (§9)** |
| size | 9,470,119,968 B, 969 tensors | `recorded/release_s_tensors.manifest`, `logs/release_clean.log` |
| MMLU-2000 dev bench (int8) | **77.80** (94.9 % of the teacher, 82.00) | `logs/t100_verifica_g.json`, `logs/bpost_tappe.log` |
| vs host B (76.05) | 113 won / 78 lost, exact McNemar p 0.0137 | `logs/t100_mcnemar_vs_B.json` |
| MMLU full test | 77.9163 (14,042 questions, Vulkan, home); 77.94 on CUDA (Aragorn) | model card `HF_S.md` |
| decode | tg128 52.26 ± 2.60 tok/s (B measured 53.93) | `stato/cicli.tsv` |

Selection caveat (also in the model card): T100 was chosen among T50 (77.25), T100 (77.80), T150 (77.75) on the
same dev bench that reports its score; the full-test figure is the unbiased one. T200 (77.40) and T250 (77.55) were measured later.

## 2. The chain, in execution order

Steps 00–02 are inputs (frozen or archived), 03–07 build the host B, 08–09 the post-training, 10 measures, 11–12 release.

| # | step | technique (paper) | input → output (sha256, first 16) | where / time (recorded) | determinism |
|---|---|---|---|---|---|
| 00 | legacy root R2B3m (shared with A and C) | see `RECIPE_C_D2.md` §4 | → `R2B3m/Thor-T-R2B3m.gguf` `56cb5bcdd9813a10` | Sept. 2026 | frozen, not re-run |
| 01 | eb882: second plane on 806 experts (2,418 expert tensors), fourth plane on 103 dense tensors, 108 maps; code `cloud/r030 @58aedd416ca1`, empirical-Bayes "eb-quote" selection, γ 2.0, measured dKL | empirical-Bayes selection of extra planes | R2B3m → eb882 `247395e18124889d` (KL 0.36385, −12.41 %) | cloud session | **GAP**: writer and inputs not archived; GGUF deleted |
| 02 | eb882K: lever K, scale ×1.045 on 28,302 single-plane expert tensors (K* tuned on pv9, not on eb882) | scale lever | eb882 → `TUTTE/eb882K.gguf` `099f1ececa3381aa` (MMLU 74.90 int8) | — | archived GGUF; **producing command not archived** |
| 03 | CARDAN-KD on eb882K: rank 16, gate/up experts, 40 layers, 300 steps max, best step 173 (`kl_held` 0.2931 vs 0.3517) | CARDAN (SliderQuant, 2609.21137 §3.3; our variant, not faithful) | eb882K + A.1 base (`cardan_base_a1_eb882K.manifest`) → fold tensors | gfx1151, ≈ 3 h | GPU, tolerance |
| 04 | CARDAN fold patch, 152 expert tensors | `patch_gguf.py` | eb882K + `piega/tensori` → `eb882K-CARDAN` `dae59b0f2c2dfe90` (MMLU 75.15, p 0.59; KL24 −7.30 %) | CPU, ≈ 1.5 min | **bitwise, verified 12/11** |
| 05 | K-FAC gradients of the KL4 answer loss, 896 questions, seed 7 | K-FAC (1503.05671) | eb882K → `AUDIT/prova_leve/grad_eb882K_kl4` | gfx1151, 58 min | GPU, tolerance |
| 06 | nine generic levers (jarq_pk, jarq_p2k, schur_pk, schur_p2k, tjarq_p1fermo, bil_foem, tied_jarq, bil_tjarq, bil_tjarq_p1fermo) on eb882K; K-FAC predictor; per-family selection `seleziona.py --cardan --meno-una` (199/250 families) | JARQ 2609.38599, SchurQuant 2608.15567, Tied 2608.08910, GPTQ 2210.17323, FOEM 2507.11017 | eb882K + grads → `COMBINAZIONE_eb882K/SEL_pred_CARDAN/tensori` (829 `.npy`) | gfx1151, part on a Dell CUDA machine; ≈ 1 day incl. candidate measurement | GPU, tolerance |
| 07 | SEL_pred_CARDAN patch: 677 dense + 152 CARDAN expert tensors → **B** | `patch_gguf.py` | eb882K + tensors → `4e4f94087067858f` (MMLU 76.05; KL24 0.32843; tg 53.93) | CPU, ≈ 1.5 min | **bitwise, verified 12/11** |
| 08 | POST training, joint mode (`cardan_congiunto_scale.py`): scales (gate/up/down), 301 F32 tensors, rank-16 residuals on 130 dense tensors, KL4 + 0.5 CE + alignment 0.3; 0→50 then resumed to 150 (later 250); checkpoint at step 100 | EfficientQAT (2407.11062)-style scale training + CARDAN residuals | B → `tappe_100/{cardan,f32,ottimizzatori,residui,scale,scale_gateup}_100.pt` (hashes in `post_step100_checkpoints.sha256`) | gfx1151, 11/01 04:07 – 22:03 | GPU, tolerance |
| 09 | T100 export: H1 = B + scales + 301 F32 (421 tensors changed), dense fold of 130 rank-16 residuals (float32, energy captured 79.2 %), patch | `bpost_b_tappe_1101.sh` | checkpoints + B → `Thor-T-B-POST-T100.gguf` `60e2d7da44e96b56`, 9,470,184,864 B | GPU, ≈ 1 h | bitwise expected; **not re-run** (H1 deleted) |
| 10 | MMLU-2000 of T100 | llama-perplexity, Q8 activations, Vulkan | T100 → 77.80 | GPU, ≈ 10 min | measurement |
| 11 | pack planes: 607 plane tensors absorbed into 252 packed families, 1576 → 969 tensors, `--check` | `pack_planes.py` | T100 → `39a0f0a3caae71d0` | CPU, ≈ 6 min | **bitwise, verified 12/11** |
| 12 | clean metadata: 17 internal keys removed, `file_type` = 36, `general.name` = 'Thor-T 35B-A3B S' | `pulisci_intestazione.py` | packed → **released S** `9d5afbf85afd0613` (tensor data digest `0e10cb3c27037156`) | CPU, ≈ 2 min | **bitwise, verified 12/11** |

## 2.1 Exact commands

Resolved by `python3 run_recipe_s.py --dry-run` (steps with a command); the originals are in `recorded/launchers/`:

- 03: `CK_OSPITE=TT/TUTTE/eb882K.gguf CK_NOME=eb882K CK_BASE=TT/CARDAN/base_A1_r16_eb882K CK_OUT=TT/CARDAN/kd_r16_eb882K CK_LIMITE=9492293427 CK_MMLU=1 CK_ORE=3 CK_PASSI=300 CK_RANGO=16 bash launchers/cardan_kd_comando_20261010_0053.sh` (the run's own copy; the current `CARDAN/cardan_ospite.sh` of 13/10 differs).
- 04 / 07: `LEVE_THREADS=6 LEVE_OSPITE=<eb882K> python3 /mnt/models/build/leve/raffina/patch_gguf.py --dir <tensori> --out <gguf> --base <eb882K>`
- 05: `launchers/grad_eb882K_kl4.sh`; 06: `launchers/comando_combinazione.sh` (with `leve_eb882k_1010.sh`, `dell_leve_eb882k.sh`).
- 08: `launchers/bpost_b_1101.sh` (50 steps) then `bpost_b_cont_1101.sh` (resume to 150). Main arguments: scale-down lr 1e-4, strata 0–39, families attn,gdn, dense rank 16, lr-densi 1e-5, lr-gamma 1e-4, tau 1.0, lambda-ce 0.5, topk 256, kl4-peso 1, kl4-n 896, kl4-rotazioni 4, allinea-peso 0.3, scale-gateup-lr 1e-4, selection `kl24_paper`, guard 0.02, ricetta B3, vram-tetto 60, allena-f32 tutte; env `E2E_GDN=veloce E2E_MOTORE=triton2 E2E_CONV_DIRETTA=1`.
- 09: `BPOST_TAPPE=100 launchers/bpost_b_tappe_1101.sh`.
- 10: `llama-perplexity -m T100 -ngl 999 -c 4096 -np 4 -ub 512 --multiple-choice -f mmlu2000_mescolato.bin`, env `GGML_VK_TQ1R_Q8=1 GGML_VK_DISABLE_GRAPH_OPTIMIZE=1`.
- 11: `python3 pack_planes.py T100.gguf packed.gguf --check`
- 12: `python3 pulisci_intestazione.py packed.gguf release.gguf --name 'Thor-T 35B-A3B S' --gguf-py <thor-pub-q8>/gguf-py`

Side record: the intermediate SEL_pred GGUF (per-family selection without the CARDAN legs, MMLU 75.95) had sha256 `6bdceb310b5113f2…` (file deleted; hash only, `recorded/gguf_chain.sha256`).

## 3. Deviations of the runner from the recorded runs

1. Launchers write into the record directories. `--run` executes GPU steps inside `unshare --user --map-root-user --mount`,
   bind-mounting an empty tree from `$FORGIA_OUT/tree` over every output root; the record stays readable under `$FORGIA_OUT/_rec`.
   Declared; the mathematics is unchanged. **This wrapper has been rendered (dry-run) but never executed end-to-end.**
2. Step 03 runs with `CK_BASE_UNITA=none` (the original waited for the systemd unit that built the A.1 base).
3. The runner is strictly serial: one GPU worker at a time, always through the house lock `lucchetto.sh` (launchers take it themselves; the runner's own measurement step is wrapped).
4. `--verify` never uses the GPU and never writes the record; outputs go to `$FORGIA_OUT/verify` and are deleted on match.

## 4. Stage 0

Root R2B3m is shared with A and C; the chain EB1 → EB3, A2, TT2, R2* is not reproducible from this forge (same
statement as `RECIPE_C_D2.md` §4). It is verified by hash only.

## 5. Determinism

| class | steps |
|---|---|
| bitwise, verified | 04, 07, 11, 12 (plus the hash checks of every on-disk GGUF) |
| bitwise expected, not re-run | 09 (deterministic patch/fold in float32; intermediates deleted) |
| GPU, tolerance only | 03, 05, 06, 08. Bitwise repeatability on gfx1151 was not measured; some levers ran on CUDA (agreement ≈ 1e-11 on errH) |
| frozen / archived | 00, 02 |
| gap | 01 |
| measurement | 10 (MMLU-2000 77.80; Vulkan and CUDA differ by 0.03 on the full test) |

## 6. Code provenance

Tool hashes are in `recorded/tools.sha256`: `patch_gguf.py` `b7f9b705…` (same as recipe C), `pack_planes.py` `619f89a6…`
(all 16 copies in `/mnt/models/build/thor-*` identical), `pulisci_intestazione.py` `2c7044e3…`, `gguf_hash.py` `972a77ba…`;
gguf-py tree digest `c8f452f1…`, `thor-pub-q8` HEAD `23968667e4c5`. POST training code:
`OBIETTIVO_C/SINTESI_1025/BPP/codice/cardan_congiunto_scale.py` (launchers recorded). eb882 code: `cloud/r030 @58aedd416ca1` (not in this forge).

## 7. Running it

```
python3 run_recipe_s.py --dry-run                         # plan, recorded hashes, local artifacts
python3 run_recipe_s.py --verify --hashes-only            # re-hash everything recorded that exists locally (~2 min)
python3 run_recipe_s.py --verify                          # + re-run the CPU steps 04, 07, 11, 12 (needs ~20 GB scratch in $FORGIA_OUT)
python3 run_recipe_s.py --verify --deep                   # + per-tensor comparison of T100 with recorded/t100_tensors.manifest
python3 run_recipe_s.py --run --steps 11,12               # produce the release from T100 (fastest full reproduction)
python3 run_recipe_s.py --run --allow-gaps                # re-run the chain (GPU steps through lucchetto.sh)
python3 -m pytest test_recipe_s.py
```
Environment overrides: `FORGIA_OUT`, `FORGIA_TT`, `FORGIA_NAS`, `FORGIA_ROOT`, `FORGIA_PY`, `FORGIA_ENGINE`. State is kept in
`$FORGIA_OUT/state/<step>.json`. A full `--run` of steps 03–09 needs on the order of 1.5 days of one gfx1151 plus 100 GB.

## 8. Gaps

1. **eb882 (step 01)** and **K ×1.045 (step 02)**: writer code, measured-dKL inputs and the producing command are not archived; eb882 was deleted, eb882K is verified only against its recorded hash. The chain from R2B3m therefore cannot be re-run from scratch.
2. Root R2B3m is frozen (EB1 → EB3, TT2 not reproducible), as for C.
3. The A.1 base command (unit `cardan-base-a1-eb882k`) is not located; only its 42-file manifest is recorded.
4. GPU steps 03, 05, 06, 08 are reproducible in procedure and to tolerance only. Step 08 was resumed across segments (0→50→150, later 250) with launcher/environment drift (`E2E_KQAT…`, `E2E_SCALE_FISSE` added later); the very first launch's argv survives only through the launcher copies.
5. Step 06's launcher re-measures every candidate (≈ 1 day); `CARDAN/cardan_ospite.sh` changed after the run (the run's copy is recorded).
6. T100 intermediates (H1, npy, residuals delta) were deleted; step 09 is re-runnable from the NAS step-100 checkpoints but was not re-run.
7. The released file itself now lives only on the Aragorn archive; locally it is regenerated (identical sha256), not hashed from the original.
8. Selection caveat on the dev bench (§1); the `TUTTE/Thor-T-TUTTE.gguf` "E2 rank 1" variant is not part of this chain.
9. Step 09 under `--run` reads the NAS step-100 checkpoints (`BPOST_TAPPE_DIR`) but its host B and base paths sit under mirrored roots; the mirror may need seeding with B and `B_POST_1101/base`. The export launcher `bpost_b_tappe_1101.sh` does not take the GPU lock itself, so the runner wraps it. Neither was executed.
10. `--run` mirroring (§3.1) is untested end-to-end; the `SEL_pred` and `eb882` GGUFs no longer exist (hash only).

## 9. Verification status (12/11/2026, CPU only, GPU untouched)

- `--verify --hashes-only`: 20 match, 0 differ; absent (hash only): eb882, SEL_pred, packed, S. Matched: R2B3m, eb882K (also on NAS), CARDAN, B, T100, tools, the four tensor directories (829/153/677/43 files), six step-100 checkpoints.
- Step 04 re-run: `dae59b0f2c2dfe9092e64608c122193dbd3c86bfb5f1c1b2fc980606e6015142` — identical.
- Step 07 re-run: `4e4f94087067858f…c867` — identical to B.
- Step 11 re-run: 252 families, 607 plane tensors absorbed, 0 mismatches, sha `39a0f0a3…e6be6d9`.
- Step 12 re-run: tensor digest `0e10cb3c27037156`, **sha256 `9d5afbf8…cc6d65f9`, identical to the released file**.

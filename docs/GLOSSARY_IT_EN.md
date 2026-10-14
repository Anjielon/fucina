# Italian → English glossary of the ported code

The GPU kernels, the end-to-end ternary student and the sequential re-quantization solver in this package were
written first in Italian, in the code base that produced the Thor-T results (lab notebook, run logs and the paper
drafts use those names). They were ported here by a **pure rename**: the computation, the order of the operations
and every data type are unchanged. This document is the single source of truth for that rename, in both
directions: from a name in a log, a run script or the paper to the code in this package, and back.

| file | content |
|---|---|
| `docs/port_word_glossary.tsv` | word map (616 Italian words → English), applied to every part of a `snake_case` / `CamelCase` identifier |
| `docs/port_overrides.tsv` | 84 whole-identifier exceptions (clashes with Python keywords or built-ins, readable class names, words whose meaning depends on the context) |
| `docs/port_identifier_map.tsv` | the resulting identifier map actually applied (1,318 identifiers) |
| `docs/port_module_map.tsv` | original module → module of this package |
| `docs/port_external_names.tsv` | names that keep their original spelling because they belong to code that is not ported yet |
| `scripts/check_port_ast.py` | static proof of equivalence (syntax trees identical after the rename) |
| `tests/test_kernels_equivalence.py` | numerical proof of equivalence (bit-identical outputs, CPU and GPU) |

## 1. Modules

| original | port | what it is |
|---|---|---|
| `kernel_blocco_fuso.py` | `fucina.kernels.fused_block` | fused Triton kernel of the in-block trit refinement sweep |
| `e2e_hilo_fuso.py` | `fucina.kernels.hilo_fused` | fused hi/lo split backward kernels of the expert families |
| `e2e_indietro_hilo.py` | `fucina.kernels.hilo_backward` | backward of the experts with hi/lo split products |
| `e2e_bf16x3.py` | `fucina.kernels.bf16x3` | bf16x3 matrix products (three bf16 products with fp32 accumulation) |
| `e2e_conv_diretta.py` | `fucina.kernels.direct_conv` | direct causal depthwise convolution (GDN short convolution) |
| `e2e_flash_qat.py` | `fucina.kernels.flash_qat` | causal GQA flash attention for the full-attention layers |
| `e2e_densi_piegati.py` | `fucina.kernels.folded_dense` | dense tensors with several planes folded into one effective weight |
| `e2e_kernel_qat.py` | `fucina.kernels.qat` | kernels for training the group scales of a frozen ternary network |
| `strumenti/gdn_qat.py` | `fucina.kernels.gdn_qat` | gated delta rule (GDN) with split fp16 products |
| `strumenti/gdn_veloce_corsa_v5.py` | `fucina.kernels.gdn_fast_v5` | fast GDN chunked recurrence (v5), base of `gdn_qat` |
| `e2e_studente.py` | `fucina.student.student` | the end-to-end ternary student |
| `e2e_pv_codici.py` | `fucina.student.plane_codes` | plane-value code optimizer (V step, trust region, registries) |
| `e2e_cache_dequant.py` | `fucina.student.dequant_cache` | on-disk cache of dequantized dense tensors |
| `e2e_codici_scale.py` | `fucina.ternary.scale_codes` | trit codes and scales of the TQ1_0 / TQ1_0R_1024 blocks |
| `e2e_campi_scala.py` | `fucina.ternary.scale_fields` | read / write of the fp16 scale fields inside GGUF blocks |
| `dequant_gguf_torch.py` | `fucina.ternary.gguf_dequant` | GGUF dequantization in NumPy / PyTorch |
| `gptq_1024_lib.py` | `fucina.ternary.gptq_1024` | GPTQ primitives for the 1024-wide rotated format |
| `rotazione_gpu.py` | `fucina.ternary.gpu_rotation` | Walsh-Hadamard rotation (FWHT 1024) in PyTorch |
| `tq1_0r_1024_codec.py` | `fucina.ternary.tq1_0r_1024_codec` | reference codec of the rotated ternary type `TQ1_0R_1024` |
| `gen_sign_1024.py` | `fucina.ternary.gen_sign_1024` | generator of the fixed sign vector of the rotation |
| `carica_strato_gguf.py` | `fucina.ternary.gguf_layer_loader` | loads one layer of a GGUF file into the Hugging Face module |
| `mappa_completa_gguf_hf.json` / `_perm.npz` | `fucina/ternary/data/gguf_hf_map.json` / `gguf_hf_perm.npz` | GGUF ↔ Hugging Face tensor map (keys translated, permutations copied byte for byte) |
| `s23_algebra.py` | `fucina.sequential.algebra` | linear algebra of stages S2+S3 (damped Cholesky, JARQ, Klein K-best, DRQ, FastKron) |
| `s23_risolvi.py` | `fucina.sequential.solve` | per-tensor solvers of the dense tensors and of the experts |
| `s23_passata.py` | `fucina.sequential.sweep` | the layer-by-layer sweep driver (stages P1-P5) |
| `s23_scrivi.py` | `fucina.sequential.write` | writer of the output GGUF arms (S2: dense only, S3: dense + experts) |
| `s1b_comune.py` | `fucina.sequential.common_s1b` | shared helpers of the zero-byte scale stage (rotation, scale fields, systems) |
| `s1_comune.py` | `fucina.sequential.common_s1` | shared helpers and reader of the stage-S1 statistics |
| `cardan_piega.py` | `fucina.sequential.cardan_fold` | CARDAN fold of a correction into the existing planes and scales |
| `sorgente.py` | `fucina.sequential.hf_source` | reader of the bf16 Hugging Face weights in GGUF order |

Module aliases keep their letters (`S`, `CS`, `AL`, `RS`, `K`, `SC`, `CP`, `KQ`, `HL`, `PV`, `X3`, `GQ`); the
only alias that was a word changed with it (`CAMPI` → `FIELDS`).

## 2. Core vocabulary

The words below are the ones a reader of the paper or of the logs meets most often. The complete list is in
`docs/port_word_glossary.tsv`.

### Model and quantization

| Italian | English | | Italian | English |
|---|---|---|---|---|
| studente | student | | maestro | teacher |
| strato / strati | layer / layers | | esperto / esperti | expert / experts |
| denso / densi | dense | | famiglia / famiglie | family / families |
| piano / piani | plane / planes | | scala | scale |
| ospite | host (the GGUF being improved) | | partenza | start (starting GGUF) |
| tensore | tensor | | peso / pesi | weight / weights |
| testa | head | | instradamento | routing |
| ternario / ternaria | ternary | | trit | trit |
| rotazione / ruota / deruota | rotation / rotate / unrotate | | piega / piegato | fold / folded |
| codici | codes | | campo / campi | field / fields |
| gruppo | group | | blocco / blocchi | block / blocks |
| hessiana / hessiane | Hessian / Hessians | | smorzamento / smorzata | damping / damped |
| bersaglio | target | | obiettivo | objective |
| legame | tie | | correzione | correction |
| residuo | residual | | rango | rank |
| ricetta | recipe | | taglia | size |
| tetto | cap | | quota | fraction |

### Computation and kernels

| Italian | English | | Italian | English |
|---|---|---|---|---|
| fuso / fusa | fused | | kernel | kernel |
| motore | engine | | riquadro | tile |
| lotto / lotti | batch / batches | | microlotto | microbatch |
| passo | step | | passaggio | pass |
| passata | sweep | | giro / giri | round / rounds |
| avanti | forward | | indietro | backward |
| prodotto | product | | somma | sum |
| dividi / diviso | split | | trasposta | transposed |
| accumulo | accumulator | | trabocco | overflow |
| ricorrenza | recurrence | | finestra | window |
| fetta | slice | | banda | band |
| flusso / flussi | stream / streams | | coda | tail |
| gancio | hook | | iniezione / inietta | injection / inject |
| cancello | gate (a pass/fail check) | | guardia | guard |
| ripiego | fallback | | al bit | bit-exact |
| caldi | warm (hot experts kept resident) | | cache | cache |

### Bookkeeping

| Italian | English | | Italian | English |
|---|---|---|---|---|
| corsa | run | | registro | registry |
| istantanea | snapshot | | impronta | fingerprint |
| misura | measure | | taratura / tarato | calibration / calibrated |
| energia | energy | | soglia | threshold |
| scelto / scelta | chosen / choice | | accettati / rifiutati | accepted / rejected |
| sorgente | source | | uscita | output |
| ingresso | input | | stato | state |
| fatto | done | | spento / acceso | off / on |
| braccio | arm (one output GGUF) | | conoscenza | knowledge |

## 3. Overrides (whole identifiers)

Applied before the word map; full list in `docs/port_overrides.tsv`.

* **Python keywords and built-ins** get a trailing underscore instead of shadowing them: `classe` → `class_`,
  `tipo` → `type_`, `somma` → `sum_`, `giro` → `round_`, `fetta` → `slice_`, `tutti`/`tutte` → `all_`,
  `campo` → `field_`, `mappa` → `map_`, `ordinati` → `sorted_`, `compila` → `compile_`, `elenco`/`lista` → `list_`,
  `imposta` → `set_`, `attese` → `expected_`, `densi` → `dense_`, `avanti` → `forward_`, `indietro` → `backward_`
  (the last three because `forward`, `backward` and `dense` already exist in the same scopes).
* **Readable class names** (English word order): `DatiFamiglia` → `FamilyData`, `FamigliaEsperti` → `ExpertFamily`,
  `FamiglieGateUp` → `GateUpFamilies`, `StratoEsperti` → `ExpertLayer`, `LineareTernaria` → `TernaryLinear`,
  `MotoreTorch`/`MotoreTriton` → `TorchEngine`/`TritonEngine`, `RegioneFiducia` → `TrustRegion`,
  `DensoPiegato` → `FoldedDense`, `EspertiPiani` → `ExpertPlanes`, `StatoV` → `VState`, `PassoV` → `VStep`,
  `TestaX3` → `X3Head`, `LineareBF16x3` → `BF16x3Linear`, `ConvCausaleBTC` → `CausalConvBTC`,
  `TensoriTracciati` → `TracedTensors`, `Studente` → `Student`, `Ricetta` → `Recipe`.
* **Context-dependent words**: `ric` → `recipe` (in the student) but `_k_ric_av` / `_k_ric_ind` →
  `_k_rec_fwd` / `_k_rec_bwd` (recurrence, in the GDN kernels); `K_PARI`/`PARI` → `K_EVEN`/`EVEN` (tile-aligned
  sizes); `monte` → `upstream`; `piu_uno` → `plus_one`; `punto_medio` → `midpoint`; `primitiva` → `primitive_kind` (it would otherwise collide with the key `primitive` in the same record); `nome_tensore` → `tensor_name`.
* Names of three letters or less are never translated (they are mathematical symbols: `po`, `sa`, `te`, `mx`, `ma`).

## 4. Environment variables, CLI options and string values

Environment-variable names, CLI options and option values were renamed with the same map, so a setting written for
the original code has an English twin here:

| original | port | | original | port |
|---|---|---|---|---|
| `E2E_COMBINA_FUSA` | `E2E_COMBINE_FUSED` | | `S23_ALFE` | `S23_ALPHAS` |
| `E2E_CONV_DIRETTA` (`_CPU`, `_TORCH`) | `E2E_CONV_DIRECT` (`_CPU`, `_TORCH`) | | `S23_ALFE_ESPERTI` / `S23_ALFA_ESPERTI` | `S23_ALPHAS_EXPERTS` / `S23_ALPHA_EXPERTS` |
| `E2E_DENSI_PIEGATI` | `E2E_DENSE_FOLDED` | | `S23_CODA` | `S23_TAIL` |
| `E2E_DENSI_SSM_AB` | `E2E_DENSE_SSM_AB` | | `S23_DRQ_GRIGLIA` | `S23_DRQ_GRID` |
| `E2E_FLASH_STADI` | `E2E_FLASH_STAGES` | | `S23_DRQ_STRATI_TAR` | `S23_DRQ_LAYERS_TAR` |
| `E2E_GDN_QAT_COMPILA` | `E2E_GDN_QAT_COMPILE` | | `S23_ENERGIA_MIN` / `_MAX` / `_RELATIVA` | `S23_ENERGY_MIN` / `_MAX` / `_RELATIVE` |
| `E2E_GDN_QAT_RIC` | `E2E_GDN_QAT_REC` | | `S23_FK_FETTA` / `S23_FK_RAPPORTO` | `S23_FK_SLICE` / `S23_FK_RATIO` |
| `E2E_HILO_FUSO` | `E2E_HILO_FUSED` | | `S23_FLUSSI` | `S23_STREAMS` |
| `E2E_KQAT_GRAFO` | `E2E_KQAT_GRAPH` | | `S23_GIRI_K1` / `S23_GIRI_K2` | `S23_ROUNDS_K1` / `S23_ROUNDS_K2` |
| `E2E_PIEGA_CACHE_TRIT` / `_GEMM` / `_KERNEL_CPU` | `E2E_FOLD_CACHE_TRIT` / `_GEMM` / `_KERNEL_CPU` | | `S23_GUARDIA` | `S23_GUARD` |
| `E2E_RUOTA_COMPILATA` | `E2E_ROTATE_COMPILED` | | `S23_K1_BANDA` | `S23_K1_BAND` |
| `GDN_VELOCE_VERSIONE` | `GDN_FAST_VERSION` | | `S23_KBEST_SEME` | `S23_KBEST_SEED` |
| `PIEGA_ALGEBRA` | `FOLD_ALGEBRA` | | `S23_MAESTRO_SENZA_SYNC` | `S23_TEACHER_WITHOUT_SYNC` |
| | | | `S23_PESO_KW` / `S23_TIENI_LAVORO` / `S23_W_CONOSCENZA` | `S23_WEIGHT_KW` / `S23_KEEP_WORK` / `S23_W_KNOWLEDGE` |

All other variables (`E2E_KQAT`, `E2E_GGUF`, `S23_KBEST`, `S23_DRQ`, ...) keep their names. Values that were
Italian words are translated too: for example `E2E_KQAT=caldi,piega1,densi_k` becomes `E2E_KQAT=warm,fold1,dense_k`,
and `--densi-ternari tutte` becomes `all_`.

New variables of the port (machine paths, see `fucina/extdeps.py`): `FUCINA_GGUF_PY`, `FUCINA_RD_TERNARY_DIR`,
`FUCINA_REFINE_DIR`, `FUCINA_DATA_DIR`, `FUCINA_CARDAN_GQ_DIR`, `FUCINA_CARDAN_HFFN_DIR`, `E2E_CORPUS`.

## 5. What is deliberately NOT translated

* **Code that is not ported yet** keeps its original names where the port calls it
  (`docs/port_external_names.tsv`): the `rd_ternary` package (`gguf_piani`, `formato`), the refinement levers
  module `comune` (`CM.ruota_H`), and the FEDELI sequential driver `sequenziale_thor` (`PulitoHF`, `modulo`,
  teacher methods `strato` / `avanti`).
* **Keys of data files written by unported code**: the calibration corpora (`"domini"`, `"nomi_domini"`,
  `"sorgenti"`) and the stage-S1 statistics file names (`esperti_conta.npy`, ...), so the port reads the same
  files.
* **File and directory names of existing data** in default paths (for example `hessiane-densi-grande`): they are
  defaults under `FUCINA_DATA_DIR` and are overridden by the variables above.

## 6. Compatibility notes

* Files that the ported code writes **and** reads back use English names: dequantization cache metadata, CARDAN
  and S2+S3 registries (`registro.json` → `registry.json`), completion markers (`FATTO` → `DONE`,
  `FATTO_PASSATA` → `DONE_SWEEP`), working directories (`strati` → `layers`). Caches and resume points written by
  the original code are therefore not reused by the port; they are rebuilt.
* The GGUF files themselves are unaffected: tensor names, types and metadata keys were already English.

## 7. Fixes made while porting (logic unchanged)

* `--cache-maestro` → `--cache-teacher` (`sequential/sweep.py`): the option string is hyphenated, so the token
  rename alone would have left the option Italian while its attribute became `cache_teacher`.
* The dequantization cache key hashes `fucina/ternary/gguf_dequant.py` (the original hashed the sibling file
  `dequant_gguf_torch.py`); caches are therefore keyed on the ported dequantizer.
* `gdn_qat` loads the fast GDN module from `gdn_fast_v5.py` (originally `gdn_veloce_corsa_v5.py`).

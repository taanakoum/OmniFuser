# OmniFuser implementation audit

This is the technical reference for the codebase as it stands on 2026-10-14.
It covers every module, script, and test; the data pipeline; the model's
operations and tensor shapes; training, evaluation, tracking, and
cross-validation; and how the work was verified.

Two companion documents cover other ground:

- [`PAPER_IMPLEMENTATION_ALIGNMENT.md`](PAPER_IMPLEMENTATION_ALIGNMENT.md)
  compares each paper equation and setting with the code, gives a verdict for
  each, and lists the paper's own inconsistencies.
- [`EDA.md`](EDA.md) analyzes the Nonastreda data.

Commands are in [`TRAINING_RUNBOOK.md`](TRAINING_RUNBOOK.md). How to turn the results into
a report is in [`REPORT_PLAN.md`](REPORT_PLAN.md).

This repository implements the OmniFuser method on the Nonastreda dataset. The
paper reports results on "Mudestreda" and MATWI. Nothing here claims to
reproduce the paper's scores.

## 1. Repository inventory

### Package `omnifuser/`

| File | Responsibility |
|---|---|
| `__init__.py` | Version `0.1.0`. On Windows it guards against a stuck WMI service: torch's import asks `platform` for the machine type, which queries WMI; if WMI does not answer within 5 s, `platform` uses its own fallback instead of blocking forever. |
| `config.py` | `ModelConfig`, a frozen dataclass holding every model dimension and variant switch. It validates all values when constructed (§4.8). |
| `data.py` | Loading and validating the MAT/CSV/JPG files, duplicate merging, pass trimming, optional decimation, per-pass pooling, per-tool timelines, folds, the training-only normalizer, windowing, the image cache, and batch collation (§3). |
| `models.py` | The model (§4): temporal embedding, RTD/MRTE, image encoder and temporal GAT, C2F fusion, recursive refinement, prediction heads, modality masking, paper ablations, the multitask loss, and a tensor-level classification-metric helper. |
| `train.py` | `train_fold`: one held-out-tool run; `model_config`, the configuration builder shared with cross-validation; `FrozenFeatureBank`, each photo's frozen-backbone features computed once per fold; the array/ROC save helpers (§5). |
| `spectrogram.py` | The optional spectrogram image source: VLT-style STFT, FFT and recurrence maps of each window's raw input force (`force_window_image`), and `ForceWindowDataset`, which builds them per window. |
| `resources.py` | `cpu_threads`: how many CPU threads to use without crowding out other programs (override, SLURM allocation, `OMP_NUM_THREADS`, idle CPUs measured without WMI). |
| `evaluation.py` | Classification, ROC, regression, cumulative paper-horizon, and baseline metrics. It also holds the paper's Table IV/V OmniFuser values (`PAPER_OMNIFUSER_RESULTS`) (§6). |
| `cross_validation.py` | `run_cross_validation`: the 10-fold leave-one-tool-out driver with fold reuse, summaries, and pooled out-of-fold metrics per input mode (§7). |
| `tracking.py` | `ExperimentTracker`: a dependency-free manifest plus JSONL and CSV metric logs (§8). |
| `model_bundle.py` | Trained models as portable files: `make_bundle` (the format every finished fold saves as `fold-Tn.pt`), `export_model` (copy one out; converts older checkpoints), `load_model` (safe loader, no weight download) and `predict` (forecast a tool without training). |
| `decision.py` | Post-hoc wear-class decision rules on saved probabilities: `ordinal_median`, `monotone_decode` (most probable non-decreasing sequence), `monotone_ordinal_decode`, validation-fitted `fit_class_offsets`, and `decision_scores`. Used by `scripts/decision_study.py`; the training pipeline does not call it. |
| `cli.py` | The `omnifuser inspect`, `train`, `export` and `predict` commands, and the shared protocol/architecture arguments (`add_protocol_arguments`, the typed `protocol_kwargs`). |

### Scripts `scripts/`

| Script | Purpose |
|---|---|
| `train_lightning.py` | Full 10-fold cross-validation (`run_cross_validation`). Defaults: CUDA, direct head, horizon 96. `--dedup-identical-forces True` or `False` is required (no default), so every full run states its duplicate handling. It is a plain PyTorch job that runs in a Lightning AI Studio; it does not use the Lightning `Trainer`. |
| `run_experiment_suite.py` | The paper studies (ablations, sensitivity, missing-modality, or all three), the image-encoder study (from scratch, pretrained fine-tuned, pretrained frozen, pretrained frozen without the image GAT), and the optional image-source study (previous-pass photo at image lag 1; input-window spectrograms at lag 0 with the main runs' frozen pretrained encoder, fine-tuned and from scratch) and `deduplication` comparison. Each variant is a resumable 10-fold run. Writes `suite.json` and `suite-summary.csv` (§9). |
| `train_local.py` | One-fold smoke test. Defaults: CPU, batch 1, horizon 96, windows capped at 2/1 unless `--full`. |
| `eda.py` | The exploratory data analysis behind `EDA.md`. Writes tables, a JSON summary, and 12 figures. |
| `compare_with_paper.py` | Prints cumulative per-horizon metrics next to Tables IV-V and the persistence baselines. |
| `model_summary.py` | Per-block parameter counts and output shapes; `--check` compares the total with a count derived from the dimensions. |
| `end_to_end_trace.py` | Runs the pipeline on fold T10 of the real data with an untrained model and prints every number quoted in `END_TO_END_SIM.md` (also as JSON); redraws `docs/end_to_end/*.png`. |

### Tests `tests/`: 153 tests

| File | Tests | What it verifies |
|---|---|---|
| `test_model.py` | 33 | Competence: the whole model memorizes a small batch with its real loss (loss below 20 % of the start, all classes right), and learns a class carried only by the image (100 %, near chance with the image masked), which exercises the image branch and C2F end to end. RTD kernel = stride = \(s_l\) = 3, depthwise. Embeddings and positions; photos ImageNet-normalized exactly once, in the data pipeline, for both encoder initializations; GAT normalization; image position placement; orthogonal private residual; landmark pooling (defaults and `k_colnorm` scale bound); `exact_split` reconstruction; heads; every ablation (forward/backward plus the intended structural change); masked-modality independence; paper-reading defaults; image-encoder name validation. |
| `test_alignment_patch.py` | 16 | Pooling, image lag, shown-label exposure, the image cache, ALS modes, modality masking statistics, BatchNorm freezing, gradients for every option, and an end-to-end fold with interruption and resume. |
| `test_block12_integration.py` | 24 | Architecture flags reach `ModelConfig`; invalid dimensions fail early; the full compact model runs with all three heads. `--dedup-identical-forces` parses True/False, the bare flag and every alias for `train`, `inspect` and the suite, and rejects ambiguous values. |
| `test_data.py` | 13 | Window shapes; exact input/target alignment; windows never mix tools; class change at a pass boundary; lag-1 photo with current-pass label; fold partitions; pooled normalizer statistics; collation; duplicate merging; span trimming and its documented limit. |
| `test_experiment_suite.py` | 9 | Ablation list; the 8 sensitivity axes and their field mapping; skipping baseline-valued points; dropout arms; manifest, summary CSV, resume and failure handling (mocked trainer); refusal of changed settings. The deduplication study runs both settings with everything else shared, keeps the flag out of `ModelConfig`, and fills the summary columns. |
| `test_interruption.py` | 2 | A truncated resume checkpoint is ignored and the fold restarts. A job killed while saving the final checkpoint leaves no fold JSON, so the fold is not reused, and the rerun continues from the per-epoch resume file. No `.tmp` files remain. |
| `test_spectrogram.py` | 6 | Window images: the STFT and global-spectrum maps share frequency rows (a sine lands on its row in both), every map is standardized, a segment shorter than one STFT frame still works; each step points at the image of its pass segment; changing any raw sample outside a window's input leaves its images unchanged, changing one inside changes them; collation pads unreferenced image slots; raw passes must match the pooled timeline, and pooling after loading equals the loader's; spectrogram runs refuse image lag 1, train end to end with `image_channels` 7 and never reuse photo folds; a frozen pretrained encoder keeps its weights and BatchNorm statistics while the image generator trains, and photos get no generator. |
| `test_runtime.py` | 5 | The CPU-thread rule (override, SLURM allocation, `OMP_NUM_THREADS`, idle CPUs, never zero); the idle-CPU measurement is bounded; the WMI guard switches `platform` to its fallback when a (simulated) WMI call never returns and leaves it alone otherwise; a frozen backbone gives identical features in training mode, and cached features give the same model outputs as pixels; a frozen photo fold trains end to end on cached features. |
| `test_model_bundle.py` | 7 | Every finished fold's `fold-Tn.pt` loads with the safe loader and `predict` on it reproduces the fold's test scores; the same for exported photo, frozen-feature (loading never downloads weights) and spectrogram models; older checkpoints are refused by `load_model` and converted by `export`; folds must be chosen in multi-fold folders; the `export` and `predict` commands. |
| `test_decision.py` | 7 | Monotone and monotone-ordinal decoding equal brute force over all non-decreasing sequences; flicker removal; the ordinal median; validation offsets recover a biased boundary and stay zero without gain; counting of changing and decreasing windows. |
| `test_tracking.py` | 4 | JSONL/CSV logging; corrupt manifest; refusal of mixed settings; resume on different hardware. |
| `test_alignment_fixes.py` | 3 | Sample-weighted partial gradient accumulation; strict completed-fold reuse (including retraining folds without missing-modality results); dataset signatures. |
| `test_cross_validation.py` | 4 | Pooled out-of-fold metrics and mean ROC; missing-AUC folds; weighted and paper-horizon summaries; per-mode summaries. |
| `test_eda.py` | 4 | Trim flags; label dynamics; autocorrelation; full EDA run with every table and figure on a synthetic dataset. |
| `test_evaluation.py` | 4 | Classification report and ROC; absent-class AUC; cumulative and baseline metrics; original-unit regression. |
| `test_dedup_split.py` | 10 | Training with duplicates but evaluating deduplicated gives exactly the same validation/test windows and targets as a fully deduplicated run, while training data differs. Keeping duplicates for evaluation enlarges the test set. `--eval-dedup-identical-forces` defaults to following training, and `inspect` rejects it. `train_lightning.py` stops with a usage error when `--dedup-identical-forces` is missing and accepts every spelling (`True`, `False`, a bare flag, `--keep-identical-forces`). Injecting a test split that ignores the evaluation setting makes this test fail. |
| `test_compare_with_paper.py` | 1 | The paper comparison table. |
| `test_pipeline_end_to_end.py` | 1 | Synthetic 10-tool cross-validation through the real loader. It checks all split/mode metrics, per-mode pooled results, every artifact, tracking rows, the training-only normalizer, the evaluation stride, and completed-fold reuse. |

### Other paths

| Path | Status |
|---|---|
| `Nonastreda_Data/` | Input data (git-ignored): `forces_xyz_raw.mat`, `labels.csv`, `labels_reg.csv`, `tool/*.jpg`. |
| `experiment_results/` | All generated results (git-ignored except its `README.md`). |
| `Literature/` | Paper PDF, reproduction guides and review notes (git-ignored). |
| `outputs/` | Old smoke-run outputs (git-ignored). |

## 2. Notation and shapes

| Symbol | Meaning | Default |
|---|---|---|
| \(B\) | Batch size | 32 |
| \(T\) | Input steps (`input_len`) | 96 |
| \(P\) | Forecast steps (`horizon`) | 96 at every entry point. The `ModelConfig`/`train_fold` library default is 24. |
| \(C\) | Force channels (Fx, Fy, Fz) | 3 |
| \(d\) | Model width (`d_model`) | 32 |
| \(M\) | Distinct photos in one window (varies) | about 2 at 1 s steps |
| \(k\) | Landmarks per modality | 4 |
| \(L\) | RTD blocks | 4 |
| \(s_l\) | RTD stride and kernel | 3 |
| \(r_{\text{RTD}}\), \(r_{\text{LR}}\) | RTD trend rank and C2F bottleneck width | 8, 8 |
| \(n\) | C2F rounds | 2 |

| Tensor | Shape |
|---|---|
| Force input | \((B,T,3)\), z-scored with training-tool statistics |
| Photos | \((B,M,3,128,384)\), ImageNet-normalized |
| Photo index per step | \((B,T)\), integers in \([0,M)\) |
| Cycle, batch, and relative codes | \((B,T)\) each |
| Each encoded modality | \((B,T,d)\) |
| Force target and output | \((B,P,3)\) |
| Class target | \((B,P)\) |
| Class logits | \((B,P,3)\) |

## 3. Data pipeline (`omnifuser/data.py`)

1. **Load and validate** (`load_nonastreda`).
   - Record IDs follow `T<tool>R<run>B<block>`.
   - The MAT file, `labels.csv`, and `labels_reg.csv` must contain exactly the
     same IDs.
   - Every photo must exist, every force array must be \((3,N)\) and finite,
     and every `flank_wear` value must be finite.
   - With `image_label`, the MAT and CSV labels must agree.
   - Classes: `sharp=0`, `used=1`, `dulled=2`. With
     `--label-column tool_label`, labels 1/2/3 map to 0/1/2.
2. **Merge duplicates** (`--dedup-identical-forces {True,False}`, default True).
   - Bit-identical force arrays within the same tool and run form a group.
   - The member with the greatest `flank_wear` is kept; ties go to the lower
     block.
   - On the real data this turns 512 rows into 378 passes (132 groups, 134
     removed).
   - `--dedup-identical-forces False` keeps all 512 records. A bare flag means
     True. The older spellings are aliases: `--deduplicate-identical-forces`,
     and `--keep-identical-forces` / `--no-deduplicate-identical-forces`
     (both False).
   - `--eval-dedup-identical-forces {True,False}` (training entry points only)
     sets duplicate handling for the validation and test data separately;
     omitted, it follows `--dedup-identical-forces`. `train_fold` then loads
     the data once per setting: training timelines from the training records,
     validation/test timelines from the evaluation records. The normalizer
     is still fitted on the training tools only.
   - One helper, `cli.add_dedup_argument`, defines the option for every entry
     point that loads data. Its parser, `cli.parse_bool`, accepts
     True/False, yes/no, 1/0 and on/off, case-insensitive.
3. **Trim pass edges** (default `--trim-pass`, `_nonastreda_span_bounds`).
   - The envelope is \(\sqrt{\text{moving-mean}_{20}(\lVert F\rVert^2)}\).
   - The reference level is the median envelope over the first
     \(\max(20, 5\%)\) samples.
   - Everything from the first to the last sample whose envelope reaches 50 %
     of the reference is kept. Internal dips are retained.
   - Known limit: a lead-in longer than about 2.5 % of the pass makes the
     reference an idle level, so the lead-in is kept. This affects 2 passes,
     about 0.02 % of samples. A pass-level fallback would also cut low-force
     cutting ramps, so it was rejected (`EDA.md` §5).
   - `--no-trim-pass` disables trimming.
4. **Decimate** (optional, `--decimation q`):
   `scipy.signal.decimate(..., zero_phase=True)` along time. This is applied
   after trimming, so the trim window stays 20 ms at 1 kHz.
5. **Pool** (`--pool q`): mean over each complete block of \(q\) samples
   within a pass. The incomplete tail is dropped and counted. Pooling never
   crosses a pass boundary. The protocol uses `--pool 1000`, i.e. 1 s steps;
   the reasoning is in `EDA.md` §6.
6. **Build one timeline per tool** (`make_tool_timelines`).
   - Passes are sorted by (run, block) and concatenated, with no separator.
   - Each step carries its pass's label, photo index, run (cycle code), block
     (batch code), and one of 16 position-within-pass bins (relative code).
   - With `--image-lag 1`, each pass's steps show the previous pass's photo;
     the first pass keeps its own. Labels never move with the lag.
   - `image_labels` stores the label of the photo actually shown.
7. **Folds** (`leave_one_tool_out_folds`): test tool \(T_i\), validation
   \(T_{i+1}\) (cyclic), training on the other 8. A tool never spans two
   splits.
8. **Normalize** (`ForceNormalizer.fit`): per-channel mean and population
   standard deviation, merged across the training tools only. A scale below
   \(10^{-8}\) is replaced by 1. The same transform applies to inputs and
   targets in every split.
9. **Window** (`WindowDataset`).
   - Input \([s,s+T)\), target \([s+T,s+T+P)\).
   - Training stride: `--stride`, default \(T+P\). Evaluation stride:
     \(\max(\text{stride},P)\), so evaluation targets never overlap.
   - Windows stay within one tool but may cross pass and run boundaries. Each
     target step's class is the label of the pass that owns it.
   - `shown_label` is the shown photo's label at the last input step; it feeds
     the persistence baseline.
   - Protocol (pool 1000, \(T=P=96\), stride 24): 1,127-1,213 training windows
     and 28-51 validation or test windows per fold.
10. **Cache and collate photos** (`ImageBank`, `collate_windows`).
    - Every photo is decoded once per fold and resized to 384×128 (bilinear,
      antialiased) as uint8, about 56 MB.
    - Collation loads each distinct photo in a window once and builds the
      per-step index map.
    - Windows with fewer photos are padded by repeating their first photo.
      No time step references a padded photo.
    - Pixels are divided by 255 and ImageNet-normalized.

`nonastreda_source_signature` records the path, size, and modification time of
every input file. Resume and fold reuse compare it.

## 4. Model (`omnifuser/models.py`)

### 4.1 Sensor embedding (Eq. 1)

\[
X_t=\mathrm{Dropout}\big(\underbrace{W_2\,\mathrm{GELU}(W_1F_t)}_{\text{value MLP }3\to d\to d}+PE_t+E_c(c_t)+E_b(b_t)+E_p(p_t)\big)
\]

- \(PE\) is the fixed sine/cosine encoding \(PE(p,2i)=\sin(p/10000^{2i/d})\),
  \(PE(p,2i+1)=\cos(\cdot)\). It is a non-persistent buffer, not a
  parameter.
- The learned tables have 128 cycle, 8 batch, and 16 relative-position
  entries. Out-of-range codes are clamped; the real data uses runs ≤ 15 and
  blocks ≤ 4. Omitted metadata uses index 0.

### 4.2 Multi-resolution temporal extractor (MRTE, Eq. 2-3, Fig. 5)

There are \(L\) cascaded `RTDBlock`s. Block \(l\) receives the previous block's
downsampled sequence and processes it in five steps:

1. **Downsample.** Zero-pad the end to a multiple of \(s_l\). Apply a depthwise
   `Conv1d` (kernel = stride = \(s_l\), groups \(d\), no bias), then a
   bias-free `Linear(d,d)` (\(W^{(l)}\)). The result \(\hat H\) has length
   \(\lceil T_{l-1}/s_l\rceil\); at the defaults the lengths are
   96 → 32 → 11 → 4 → 2.
2. **Trend.** \(\mathcal T=\hat H P Q\) with \(P\in\mathbb R^{d\times r}\),
   \(Q\in\mathbb R^{r\times d}\) (Xavier init). Ablation `moving_average`
   instead uses a 3-step centered mean (`avg_pool1d`, edges excluded).
3. **Seasonal.** \(\mathcal S=\hat H-\mathcal T\).
4. **Combine.** `decomp_mode=branch` (default):
   \(\mathrm{FFN}_T(\mathcal T)+\mathrm{FFN}_S(\mathcal S)\), each FFN being
   \(d\to2d\to d\) with GELU. `literal`: \(\mathcal T+\mathcal S\), which
   equals \(\hat H\).
5. **Restore.** A `Linear` over time (\(T_l\to T\)), GELU, then a channel MLP
   \(d\to d\to d\).

MRTE returns \(Z^r=\sum_l \tilde U^{(l)}\), shape \((B,T,d)\). Ablations
`without_mrte` and `without_rr_mrte` replace MRTE with an identity.

### 4.3 Image branch (Eq. 4 and §V-B)

- **Encoder.** torchvision EfficientNet-B0 feature trunk, global average
  pool, then `Linear(1280,d)`. It starts from random weights unless
  `--pretrained` (ImageNet). `--freeze-bn` keeps BatchNorm in eval mode.
  `--freeze-image-encoder` (the main runs) also fixes the weights and keeps the
  whole trunk in eval mode, which turns off the stochastic depth inside
  torchvision's blocks; each photo's 1280 features are then computed once per
  fold (`FrozenFeatureBank`) and the model receives them instead of pixels.
- **Without the GAT** (`--no-image-gat`, `ModelConfig.image_gat=False`; an
  experiment, not the paper): no GAT layers are built, and the gathered,
  projected encoder features get only the position and metadata codes before
  fusion. Step 7's `pretrained-frozen-no-gat` arm uses it.
  `compact` is a 3-layer CNN (96 features) for smoke tests. Any other name
  raises an error.
- **Placement.** The \(M\) photo vectors are gathered onto the \(T\) steps.
- **Temporal GAT.** 4 layers, each computing
  \(x\leftarrow\mathrm{LN}(x+\mathrm{Dropout}(\mathrm{ELU}(\mathrm{GAT}(x))))\).
  GAT is a bias-free projection split into heads. Its attention logits are
  \(\mathrm{LeakyReLU}_{0.2}(a_s^\top h_i+a_t^\top h_j)\), with a softmax over
  all \(T\) keys (a fully connected temporal graph), attention dropout, and
  concatenated heads.
- **Then add** the sine/cosine positions and the image branch's *own*
  cycle/batch/relative tables, which are not shared with the sensor branch.

### 4.4 C2F fusion layer (Eq. 5-19)

Inputs are \(Z^r,Z^i\in\mathbb R^{B\times T\times d}\).

0. **Optional pre-attention** (`--pre-attn`, off by default and not in the
   paper): \(Z\leftarrow\mathrm{LN}(Z+\mathrm{MHA}(Z))\) per modality.
1. **Shared/private split** (Eq. 5). \(S=ZW'\) (bias-free). Per sample:
   \[
   P=Z-\frac{\langle Z,S\rangle_F}{\lVert S\rVert_F^2+\varepsilon}\,S,
   \]
   which makes \(\langle S,P\rangle_F=0\) over the whole \((T,d)\) matrix,
   not per token. With `--exact-split`, the shared part becomes the projection
   \(cS\), so \(Z=P+S\) holds exactly (Eq. 7).
2. **Private refinement.** `private_attn=plain` (default):
   \(\hat P=\mathrm{MHA}(P)\). `residual_ln`:
   \(\mathrm{LN}(P+\mathrm{Dropout}(\mathrm{MHA}(P)))\).
3. **Adaptive landmarks** (Eq. 10). Weights are
   \(\alpha=\mathrm{softmax}_k(f(S))\in\mathbb R^{T\times k}\), and
   \(L=\tilde\alpha^\top S\), \(V_L=\tilde\alpha^\top(SW_v)\). The weighting
   depends on `als_mode`:

   | `als_mode` | Weighting \(\tilde\alpha\) |
   |---|---|
   | `k_colnorm` (default) | Each column of \(\alpha\) divided by its sum over time, so every landmark is a weighted mean. Its size stays bounded as \(T\) grows. |
   | `k_sum` | Printed form; landmark size grows with \(T\). |
   | `time` | Softmax over time instead of over \(k\). |
   | Ablation `without_als` | Means of \(k\) contiguous time segments (`adaptive_avg_pool1d`); no learned scores. |

4. **Cross messages** (Eq. 11):
   \(H^{r\leftarrow i}=\mathrm{softmax}(S^rL^{i\top}/\sqrt d)\,V_L^i\), and
   symmetrically \(H^{i\leftarrow r}\).
5. **Hybrid gate** (Eq. 16-18). \(F=[H,\hat P]\in\mathbb R^{T\times2d}\).
   \[
   G=\sigma(a)\,\sigma(W_gF)+(1-\sigma(a))\,\sigma\big(W_l\,\mathrm{DWConv}_3(F)\big),
   \qquad U=G\odot H+(1-G)\odot\hat P.
   \]
   - \(W_g\) and \(W_l\) map \(2d\to d\). This width is forced: Eq. 17 prints
     \(2d\times2d\), but Eq. 18 needs a \(d\)-wide gate.
   - The scalar \(a\) starts at 0, so the mixture starts at 0.5.
   - Each modality has its own gate.
   - Ablation `without_hybrid_gate`: \(U=\tfrac12(H+\hat P)\). The gate is not
     computed and its weights receive no gradient.
6. **Low-rank fusion** (Eq. 19):
   \(\hat Z=\mathrm{Dropout}(W_3\,\mathrm{GELU}(W_2[U^r,U^i]))\),
   \(2d\to r\to d\).
7. **Diagnostics.** These are logged and are *not* added to the loss:
   - alignment: the mean squared difference between L2-normalized
     \(S^r\) and \(S^i\);
   - orthogonality: the mean of squared \(\langle S,P\rangle_F\), which is
     about 0 by construction.

### 4.5 Recursive refinement (Eq. 20) and ablations

- **Round 1** fuses \((Z^r,Z^i)\).
- **Round \(j>1\)** fuses
  \((Z^r+\mathcal P_r(\hat Z^{(j-1)}),\,Z^i+\mathcal P_i(\hat Z^{(j-1)}))\).
  Each \(\mathcal P\) is a bias-free `Linear(d,d)` per round and modality.
  Rounds have separate weights; `--share-c2f` reuses one \(G\).

| Ablation | What changes |
|---|---|
| `without_rr` | One C2F round, no refinement projections. |
| `without_rr_mrte` | One round, and MRTE replaced by an identity. |
| `concat_c2f` | Each round is `Linear(2d,d)` over \([Z^r,Z^i]\). |
| `without_c2f` | No fusion layers; \(\hat Z=\tfrac12(Z^r+Z^i)\). |

### 4.6 Missing modalities

Masking happens after both encoders.

- **Inference** (`drop_modality="sensor"` or `"image"`): that stream is set to
  zero. A test proves the masked input then cannot affect any output.
- **Training** (`--modality-dropout p`): each sample drops the sensor stream
  with probability \(p/2\) or the image stream with probability \(p/2\), never
  both.

### 4.7 Prediction heads (Eq. 21-23) and loss

All heads read only the last fused token \(\hat z_T\):

| `prediction_head` | Force output | Class output |
|---|---|---|
| `direct` (default) | `Linear(d, P·3)` → \((B,P,3)\): separate \(W_r^{(t)},b_r^{(t)}\) per lead | `Linear(d, P·3)` |
| `paper_literal` | `Linear(d,3)` repeated over all \(P\) leads | `Linear(d,3)` repeated |
| `lead_conditioned` | \(h_t=\mathrm{GELU}(W[\hat z_T;e_t])\) with a learned lead embedding \(e_t\), then `Linear(d,3)` | `Linear(d,3)` on \(h_t\) |

The loss is \(\mathcal L=\mathrm{MSE}(\hat Y,Y)+\mathrm{CE}(o,s)\): force MSE in
z-scored units plus cross-entropy over all \(P\) future steps, weighted 1:1.

### 4.8 Configuration validation

`ModelConfig.__post_init__` rejects:

- non-positive lengths, widths, layer counts, ranks, strides, landmark
  counts, or round counts;
- a width not divisible by the head count;
- dropout or modality dropout outside \([0,1)\), including NaN;
- unknown `decomp_mode`, `prediction_head`, `als_mode`, `private_attn`,
  `image_encoder`, or `ablation` values;
- `pretrained` with any encoder except EfficientNet.

Every option is listed in the runbook's argument reference.

### 4.9 Size and cost

| Configuration | Parameters |
|---|---|
| Default, EfficientNet-B0 | 4,183,600 |
| `compact` encoder | 144,188 |

`scripts/model_summary.py --check` reproduces both counts from the
dimensions. One batch of 32 windows costs \(2.46\times10^{10}\) forward FLOPs,
98 % of it in EfficientNet. Forward+backward is about
\(7\text{-}15\times10^{10}\), depending on whether a window shows one or two
photos, against the paper's ~\(10^{11}\).

## 5. Training one fold (`omnifuser/train.py::train_fold`)

1. **Seed and device.** Seeds Python, NumPy, and PyTorch. On CUDA it fails
   early if no GPU is visible or it has less than 3 GiB. On CPU it caps
   threads at 4.
2. **Data and model.** Loads the data, builds timelines and the fold, fits the
   normalizer, creates windows (optionally capped by
   `--max-train-windows`/`--max-eval-windows`, randomly but reproducibly),
   builds the image cache, and creates the model and an Adam optimizer
   (default lr \(10^{-4}\)).
3. **Epoch loop.**
   - Training: shuffled batches; `--accumulation-steps` gradient accumulation
     weighted by sample count, including the incomplete last group; gradient
     norm clipped at 1.0.
   - Validation runs after each training pass. Both passes are logged.
   - After every epoch, `fold-Tn-resume.pt` stores model, optimizer,
     best-state, and RNG state, plus a signature of data, fold,
     preprocessing, windows, training settings, and config. A rerun with the
     same signature continues from the next epoch.
4. **Model selection.** The epoch with the lowest validation loss
   (`--select-best`, default), or the last epoch (`--no-select-best`).
5. **Final evaluation of the selected checkpoint.**
   - Test with full inputs.
   - Train, validation, and test in each of `full`, `sensor_missing`, and
     `image_missing` modes.
   - Each mode's test predictions and ROC curves are saved.
6. **Outputs.**

   | File | Contents |
   |---|---|
   | `fold-Tn.json` | Settings (including `dedup_identical_forces`, `eval_dedup_identical_forces`, `retained_records` and `evaluation_retained_records`), history, test metrics, `modality_evaluations`, effective architecture, artifact names |
   | `fold-Tn.pt` | The trained model file (`model_bundle` format): weights, config, normalizer, data settings, fold tools, test scores |
   | `fold-Tn-predictions[-<mode>].npz` | Test predictions |
   | `fold-Tn-roc[-<mode>].npz` | ROC curves |

   The resume file is deleted on success.

## 6. Metrics (`omnifuser/evaluation.py` and `_run_epoch`)

The same metrics are computed for every split, every epoch, and every input
mode.

**Classification.** Classes and horizon steps are flattened:

- accuracy;
- precision, recall, and F1, both macro and support-weighted;
- per-class precision/recall/F1/support and the confusion matrix;
- the same scores for each forecast step;
- one-vs-rest ROC from softmax probabilities, with per-class and macro AUC.
  A class that is absent or always present in a split gets AUC `null` and no
  curve.

**Regression.** MAE, MSE, and RMSE in original force units (after inverse
normalization) and in z-scored units: overall, per channel (Fx/Fy/Fz), and per
forecast step.

**Test only.**

- `paper_horizons`: cumulative z-scored MSE/MAE and classification scores
  over leads 1…h for h = 12, 24, …, 96 (the units of Tables IV-V).
- `baselines`: last-value and zero forecasts, and the shown photo's label as
  a persistence classifier, for the whole horizon and per paper horizon.

## 7. Cross-validation (`omnifuser/cross_validation.py`)

`discover_tools` requires exactly 10 tools in `tool/`. Fold \(i\) uses seed
`seed + i`.

**Fold reuse.** A finished fold is reused only if all of these hold:

- every artifact exists inside the output directory;
- the config is identical;
- the data signature, fold membership, preprocessing (including both duplicate settings), strides, window caps,
  learning rate, accumulation, workers, seed, device, epochs, and batch size
  match;
- it contains all three input modes on all three splits.

Otherwise it is retrained.

**Outputs.** `cross-validation-results.json` contains:

| Key | Contents |
|---|---|
| `cv_mean_std` | Fold mean ± std of the headline test metrics, including z-scored MSE/MAE/RMSE. Folds with a `null` AUC keep their slot and are skipped in the mean. |
| `paper_horizons_mean_std` | Fold mean ± std of model and baseline scores per paper horizon |
| `modality_evaluations` | Fold mean ± std per input mode and split |
| `out_of_fold` | All held-out test windows pooled, for the `full` mode. Classification is computed from the summed confusion matrix; regression element-wise, with z-scored errors from each fold's own normalizer. The mean ROC is interpolated on a 1001-point FPR grid. |
| `out_of_fold_by_mode` | The same pooling for `sensor_missing` and `image_missing` |

The run also writes `cross-validation-mean-roc[-<mode>].npz` and
`cross-validation-progress.json`, the latter updated after each fold.

## 8. Experiment tracking (`omnifuser/tracking.py`)

`<output>/tracking/` holds three files:

| File | Contents |
|---|---|
| `run.json` | Run ID, metadata (data signature, folds, preprocessing, windows, training, model config, evaluation modes, runtime), and `runtime_history` |
| `metrics.jsonl` | One JSON event per log call: split, fold, epoch, context, nested metrics |
| `metrics.csv` | Every finite numeric leaf as a row: `timestamp_utc, run_id, fold, epoch, split, evaluation_mode, metric, value` |

**Resume rules.** Resuming into an existing directory requires identical
metadata except `runtime`. A different GPU or library version is allowed and
appended to `runtime_history`; any other change raises an error. A legacy CSV
without `evaluation_mode` is migrated in place.

**Logged events.** Setup (normalizer and window counts); train and validation
for every epoch (plus epoch seconds and peak CUDA memory); the test result;
the post-selection evaluations per mode; and the final cross-validation
aggregate.

## 9. Experiment suites (`scripts/run_experiment_suite.py`)

| `--study` | Variants | Notes |
|---|---|---|
| `ablations` | 9 | Full model plus the 8 Fig. 13 variants |
| `sensitivity` | 22 | Baseline plus every off-baseline value of k, d, GAT heads, r → `rtd_rank`, GAT layers, s_l, L → `num_rtd_layers`, n → `num_fusion_layers` |
| `missing-modality` | 1 + number of rates | A no-dropout reference plus one arm per `--modality-dropout-rates` value (default 0.2) |
| `all` | 31 | The ablation full model serves as the shared reference |
| `deduplication` | 2 | Training with `--dedup-identical-forces` True and False, all else shared. Both arms are evaluated on the same deduplicated windows unless `--eval-dedup-identical-forces` is given. Not part of `all`. The summary adds the two settings, `retained_passes`, `evaluation_passes` and window-count columns. |

- **Defaults.** Unlike the other entry points, the suite defaults to the
  protocol: `--pool 1000 --image-lag 0`, 30 epochs, horizon 96, stride 24,
  batch 32, the direct head, and CUDA.
- **Layout.** Each variant is a complete 10-fold run in
  `<output-root>/<study>/<run-id>/<group>/<variant>/`.
- **`suite.json`** records shared settings, the baseline config,
  interpretation notes, and each variant's status, changed settings,
  timestamps, and result path.
- **`suite-summary.csv`** has one row per variant, with fold mean/std test
  metrics for each input mode. It is rewritten after every variant.
- **Resuming.** Rerunning with the same `--run-id` and settings skips
  completed variants; different settings are refused. A failed variant is
  marked `failed` with its error and retried on the next run.

## 10. Verification (2026-10-14)

Static checks (`pyproject.toml` pins both configurations):

| Check | Result |
|---|---|
| `ruff check .` | All checks pass |
| `pyright` (omnifuser, scripts, tests) | 0 errors |
| VS Code diagnostics for the open project files | Clean (Pylance only analyzes open files; the `pyright` row covers the whole tree) |

`pytest` passes all 153 tests. Three deliberately injected defects each make a
test fail: an off-by-one target window, a normalizer fitted on all tools, and
a sensor mask that does nothing.

Real data, CPU, compact encoder, capped windows:

| Path | What ran |
|---|---|
| `omnifuser inspect` | 378 passes, 10 tools, 147/118/113 per class, no remaining duplicates |
| `--dedup-identical-forces True/False` (2026-10-15) | `inspect` (378 vs 512 passes, 1,462 vs 2,012 windows), `train_local.py`, `task_diagnostics.py`, and the 2-variant × 10-fold `--study deduplication` suite. Then the training/evaluation split: the suite with deduplicated evaluation for both arms, and `train_lightning.py` with `--dedup-identical-forces False --eval-dedup-identical-forces True`, including a resume rerun. All completed. |
| `omnifuser train` | One fold for each of the 9 ablations and each of 17 flag sets: the four printed-paper arms, `--exact-split`, the residual private path, the time ALS, pre-attention, shared C2F, modality dropout, accumulation, tool labels, no trim and no dedup, last-epoch selection, all architecture overrides, and pretrained EfficientNet with frozen BatchNorm. Each wrote all 9 artifacts. |
| `train_lightning.py` | All 10 folds. Rerunning reused every fold in 4 s. |
| `run_experiment_suite.py --study missing-modality` | Both variants completed, with the summary CSV. |
| Other scripts | `train_local.py` (horizon 96, all 8 paper horizons), `model_summary.py --check`, `task_diagnostics.py`, `compare_with_paper.py`, `eda.py` |

Final pre-training check (2026-10-15, commit `f6ca3db` on `Lightning-AI-training`):

| Check | Result |
|---|---|
| Push state | Branch in sync with GitHub; working tree clean |
| Fresh clone | A clean `git clone` (code from git only) passes all 119 tests, Ruff, and Pyright |
| Default branch | `main` was fast-forwarded to the training branch (commit `43bf03c`; no merge commit, no force push). A fresh `git clone https://github.com/taanakoum/OmniFuser.git` checks out `main` and passes all 119 tests. The runbook clones `main`. |
| Required duplicate setting (2026-10-15) | `train_lightning.py` has no default for `--dedup-identical-forces`: without it the script stops before loading data. `$PROTOCOL` no longer contains it, and every `train_lightning.py` command in the runbook states it. Step 3 runs 3a (`True`/`True`, `main/full-model`) and 3b (`False`/`False`, `main/full-model-no-dedup`). All 13 runbook blocks that call `train_lightning.py` or `run_experiment_suite.py` pass `bash -n` and the real argument parsers with training stubbed. 3a and 3b also ran all 10 folds on the real data (CPU, capped windows), followed by `compare_with_paper.py` (378/378 vs 512/512 passes). A rerun of 3a reused all 10 folds in 4 s, and 3b's setting pointed at 3a's folder is refused. 124 tests, Ruff and Pyright pass. |
| Shell-independent protocol (2026-10-15) | In the Studio's zsh terminal, step 1 failed with `unrecognized arguments: --data-dir Nonastreda_Data ...`: zsh does not word-split an unquoted `$PROTOCOL`, so the whole protocol arrived as one argument. `PROTOCOL` is now an array, `PROTOCOL=( ... )`, used as `"${PROTOCOL[@]}"` in all 25 places, which expands to separate arguments in bash and zsh. Commands no longer contain `#` comments (zsh rejects them without `interactivecomments`). Verified in bash: the runbook array has 28 entries, and step 1 parses. All 13 `train_lightning.py`/suite blocks pass `bash -n` and the stubbed parsers. The three troubleshooting errors were reproduced. zsh itself was not available for testing. |
| Image lag 0 protocol and pretrained comparison (2026-10-18) | The protocol (`PROTOCOL`, the suite default) uses the paper's same-pass photo, `--image-lag 0`; `--image-lag 1` is an experimental runbook run (`image-lag-1/full-model`). Main runs train EfficientNet-B0 from scratch; an optional `--pretrained` run (`image-encoder/pretrained`) compares with step 3a. Photos are ImageNet-normalized in the data pipeline (`ImageBank`) for every run, which is what the pretrained weights expect. (A second normalization inside the encoder, added on 2026-10-18 for `--pretrained`, was removed on 2026-10-21: it normalized pretrained inputs twice.) Verified: all 16 runbook blocks that call `train_lightning.py` or the suite pass `bash -n` and the stubbed real parsers (lag 0 everywhere except the experimental run, `pretrained` only in its run); real-data one-fold runs from scratch and pretrained wrote all 9 artifacts; light 10-fold runs of step 3a and the pretrained run completed with `compare_with_paper.py`; an old lag-1 run moved to `image-lag-1/full-model` reused all 10 folds, and a lag-0 run into it was refused. 125 tests, Ruff and Pyright pass. |
| Ruche cluster support (2026-10-19) | `TRAINING_RUNBOOK_RUCHE.md` and `slurm/` (`job.sbatch`, `checks.sbatch`, `chain.sh`, `status.sh`, `env.sh`, `protocol.sh`) run the same experiments as SLURM jobs, with runs longer than 24 h split into chained jobs (`--dependency=afterany`) that resume. For that, every result file is now written atomically (temporary file, then rename): the per-epoch resume checkpoint, the prediction and ROC files, the final checkpoint, the tracking manifest, and the fold JSON, which is written last as the completion marker. An unreadable resume checkpoint restarts the fold instead of crashing. Training now prints one progress line per epoch (`T1 epoch 3/30: train loss ..., validation loss ..., 98 s`), so job logs show that a long fold advances. `status.sh` (queue, run and suite progress, error logs) was tested on a simulated results tree with a fake `squeue`. Verified locally: the scripts pass `bash -n`; `job.sbatch` ran a real one-fold run in a simulated SLURM environment (stub `module`/`activate`), refuses to run outside SLURM or without a command, and returns the command's exit code; `chain.sh` submits dependent parts with the protocol intact; all 42 submissions in the new runbook, and its cheat sheet, parse with the real argument parsers and give the intended settings. Not verified: anything on Ruche itself (module names, partition limits, GPU driver), which comes from the November 2023 training course. 127 tests, Ruff and Pyright pass. |
| Experiment reorganization and cleanup (2026-10-21) | **Image experiments.** `--image-source spectrogram` (`omnifuser/spectrogram.py`): per-pass STFT log-power images of the raw 1 kHz force (Hann 1024, hop 256, one channel per axis, standardized per image), keyed by photo path so the image branch is unchanged. Optional experiment only, refused with image lag 0, recorded in fold JSON, resume signature and run metadata only when used, so photo runs keep their signatures. `--freeze-image-encoder` (with `--pretrained`) keeps the backbone weights and BatchNorm statistics fixed. Suite studies `image-encoder` (step 7: from scratch, pretrained fine-tuned, pretrained frozen) and the optional `image-source` (previous-pass photo, spectrogram, pretrained spectrogram, all image lag 1); per-variant `image_lag`/`image_source` overrides. **Leakage.** With image lag 1 a tool's first pass previously showed its own image (its label, or force from the forecast window); it now shows no image (zeros) and label -1. Folds, the training-only normalizer, image normalization (fixed constants or per image), model selection on the validation tool and seeded window caps were re-checked and are leak-free. **Fixes.** `--pretrained` no longer normalizes photos a second time (the pipeline already applies the ImageNet statistics). Library default horizon 96, as in every entry point. **Removed as redundant.** `scripts/task_diagnostics.py` (superseded by `eda.py`'s difficulty table), the tracker's legacy CSV migration (unreachable: older run folders are refused at the configuration check), the `ToolTimeline` fallback without `image_labels`, the archived `runs/` and `omnifuser_blocks*/` folders, old saved outputs and superseded notes (all in git history). Model competence tests added. 131 tests, Ruff and Pyright pass. Spectrograms not yet run on the full real data locally (laptop memory taken by another workload); see the runbooks' smoke steps. |
| Input-window spectrograms (2026-10-21) | `--image-source spectrogram` now builds its images from the raw force of each window's own input steps instead of whole previous passes, so it runs at image lag 0 (lag 1 is refused) and no window lacks an image. Following VLT (arXiv:2607.14510, Sec. III-C), each pass segment inside the window becomes 7 standardized maps: STFT log power per axis (Hann 256, hop 64; VLT's wavelet map, about a hundred times cheaper), the segment's FFT log power on the same frequency rows, and a recurrence map of the spectral columns (10 % recurrence rate). `ModelConfig.image_channels` 7 adds a learnable image generator (conv 7→16→3, VLT Eq. 8) before EfficientNet-B0; it trains even with a frozen backbone. One image per pass segment keeps the paper's per-pass replication. `ForceWindowDataset` computes images in the loader; `load_nonastreda` releases each MATLAB array once converted (lower peak memory for unpooled loads). Only VLT's time-to-image construction (Eqs. 5-8) is used; the rest of the model is OmniFuser. **Protocol.** Main runs now use EfficientNet-B0 with ImageNet weights and a frozen backbone (`--pretrained --freeze-image-encoder` in `PROTOCOL`; both flags gained `--no-` forms; library defaults stay off, so capped `compact` runs work). The `image-source` study: previous-pass photo (lag 1) and the spectrogram (lag 0), both with the main encoder, plus spectrograms fine-tuned and from scratch. `tests/conftest.py` caps test threads at half the logical CPUs (`OMNIFUSER_TEST_THREADS`). |
| Final review before training (2026-10-22) | Every module and script reread. **Fixed.** (1) `run_cross_validation` built its expected configuration without `image_channels` 7, so a resumed spectrogram study would never have reused its finished folds and retrained them all; both now use `train.model_config`. (2) A frozen backbone still ran torchvision's stochastic depth in training mode (random residual drops in a "fixed" feature extractor); it now stays in eval mode. **Faster.** (3) With the frozen backbone (all main runs), each photo's features are computed once per fold (`FrozenFeatureBank`, identical outputs, tested): a CPU training step went from 417 to 108 ms (batch 8). (4) `_run_epoch` keeps its running sums on the device and copies them once per epoch, instead of about 15 GPU synchronizations per step. **Runtime.** (5) `omnifuser.resources.cpu_threads` picks CPU threads from the SLURM allocation, `OMP_NUM_THREADS` and the CPUs idle now (measured without WMI), for training on CPU and for the tests; `--num-workers` above the available CPUs prints a warning. (6) The package guards against a stuck Windows WMI service, which had made `import torch` block forever (the cause of 24-minute local test runs). 137 tests (about a minute), Ruff and Pyright pass. |
| Final check before the Ruche runs (2026-10-22) | Commit `2b869d5` checked as Ruche will run it: a fresh clone passes Ruff, Pyright and all 137 tests; all code byte-compiles with Python 3.11, Pyright reports no Python 3.11 API problems, and the full suite passes in a Python 3.11 environment (the version `TRAINING_RUNBOOK_RUCHE.md` installs). On the real data: a capped 10-fold `train_lightning.py` run with the protocol's frozen ImageNet encoder completes, and its rerun reuses all 10 folds; capped folds of the spectrogram path (frozen encoder, 7 channels) and of the lag-1 photo path complete. Fixed: `slurm/status.sh` now knows the `image-encoder` (3) and `image-source` (4) suite sizes. Ruche runbook: storage (about 0.4 GB per 10-fold run, about 20 GB in total, logs compress about 10×), the ImageNet download in the setup path, the frozen encoder among the protocol's key settings, the spectrogram smoke command reflowed, CUDA-build advice for newer PyTorch, and a row for the `--num-workers` warning. |
| Image GAT switch (2026-10-25) | Checked: with a pretrained (frozen) encoder, the photo features pass through the paper's 4 image GAT layers (confirmed with forward hooks). Added `--image-gat` / `--no-image-gat` (`ModelConfig.image_gat`, default on, the paper): without it no GAT layers are built and the projected encoder features go straight to fusion with the position and metadata codes. Step 7 gains a fourth arm, `pretrained-frozen-no-gat`; main runs are unchanged. `fold-Tn.json` records `image_gat_layers`; `slurm/status.sh` counts 4 image-encoder variants. 2 tests added (139). |
| Saved models (2026-10-26) | Every finished fold now saves its final model as `fold-Tn.pt` in a portable format (`model_bundle.make_bundle`): selected weights (with the frozen ImageNet backbone), config, the fold's normalizer, the data settings, fold tools and test scores; it loads with `torch.load(weights_only=True)`. `omnifuser predict` reuses it without training (verified on the real data: identical test loss and accuracy to training); `omnifuser export` copies one out and converts older checkpoints. Training's data loading, windowing and image cache moved into shared functions (`load_timelines`, `window_dataset`, `image_bank_for`) so prediction prepares inputs exactly as training. `OmniFuser(config, load_pretrained=False)` skips the ImageNet download when saved weights replace it. 7 tests added (146). |
| Runbook steps 0-1 on real data (CPU, EfficientNet) | `inspect` (378 passes, 10 tools, 147/118/113 per class, 0 duplicates), `model_summary.py --check` (4,183,600, match), one-fold smoke writing all 9 artifacts |
| Every study variant | All 33 (31 in `--study all` plus 2 deduplication arms) build and train a step with finite loss |
| Full ablation suite on real data | `--study ablations`, 9 variants × 10 folds (capped, compact encoder): all completed, `suite-summary.csv` complete |
| Documentation | Every `--flag`, source path and test count named in the docs exists and matches |
| Report snippets | All 9 snippets in `REPORT_PLAN.md` run against real smoke outputs |
| Runbook commands, run as written | A GitHub clone; the data packed, uploaded-equivalent and unpacked with `tar`; then every command block extracted from `TRAINING_RUNBOOK.md` and run with a light `$PROTOCOL` (CPU, compact encoder, capped windows): steps 0-3; all suites (ablations, missing-modality, rates, deduplication) to completion; `--study sensitivity` and `--study all` orchestration for all 22 and 31 variants; every sensitivity variant trained one real fold through its documented flag; the `--study all` command interrupted mid-variant and resumed (finished variant skipped, finished folds reused); all optional runs; monitoring, download archives and Windows unpacking; the report snippets on the unpacked copy; §18 in Windows PowerShell. Fixes from this test: `tar` replaces `Compress-Archive`/`zip` (which wrote backslash paths), archive names git-ignored, consistent `tee` logs, a working suite-status command, a troubleshooting table, a background-run callout, and a note that reruns replace the console log. |

Behavioral checks:

- **Memorization.** On 32 real windows, all three C2F configurations
  memorize the targets: force MSE drops from 0.93 to 0.17-0.19 and accuracy
  rises from 0.35 to at least 0.996.
- **FLOP counts** match the paper (§4.9).

Not verified here: this machine has no GPU (PyTorch 2.14.1+cpu). CUDA
execution, GPU memory, throughput, convergence, and full-data scores must be
checked with the runbook's GPU smoke and pilot steps.

## 11. Change history (condensed)

- **2026-10-11:** the initial package; Nonastreda loader with exact-duplicate
  merging and span trimming; leave-one-tool-out cross-validation; three
  prediction heads; Blocks 9-11 alignment options (pooling, image lag, ALS
  modes, private-attention and C2F-sharing variants, modality dropout,
  baselines, resume); Blocks 12 architecture flags and `model_summary.py`.
- **2026-10-14:**
  - experiment tracking with input-mode columns;
  - missing-modality evaluation of every fold;
  - Fig. 13 ablations and the suite runner;
  - corrected sensitivity mapping (r → `rtd_rank`, added L, n = 1-3);
  - fixed-segment w/o-ALS;
  - no gate computation in w/o-HG;
  - hardware-tolerant resume;
  - retraining of folds without missing-modality results;
  - per-mode out-of-fold metrics and z-scored summaries;
  - defaults switched to the printed paper readings (`private_attn=plain`,
    `als_mode=k_colnorm`; `direct` head and horizon 96 at every entry point);
  - `--exact-split`;
  - strict image-encoder names;
  - the EDA (`scripts/eda.py`, `EDA.md`);
  - `--horizon`/`--stride` in `train_local.py`;
  - typed protocol arguments and save helpers;
  - Ruff and Pyright clean.

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
| `__init__.py` | Package marker; version `0.1.0`. |
| `config.py` | `ModelConfig`, a frozen dataclass holding every model dimension and variant switch. It validates all values when constructed (§4.8). |
| `data.py` | Loading and validating the MAT/CSV/JPG files, duplicate merging, pass trimming, optional decimation, per-pass pooling, per-tool timelines, folds, the training-only normalizer, windowing, the image cache, and batch collation (§3). |
| `models.py` | The model (§4): temporal embedding, RTD/MRTE, image encoder and temporal GAT, C2F fusion, recursive refinement, prediction heads, modality masking, paper ablations, the multitask loss, and a tensor-level classification-metric helper. |
| `train.py` | `train_fold`: one held-out-tool run. It also provides the array/ROC save helpers (§5). |
| `evaluation.py` | Classification, ROC, regression, cumulative paper-horizon, and baseline metrics. It also holds the paper's Table IV/V OmniFuser values (`PAPER_OMNIFUSER_RESULTS`) (§6). |
| `cross_validation.py` | `run_cross_validation`: the 10-fold leave-one-tool-out driver with fold reuse, summaries, and pooled out-of-fold metrics per input mode (§7). |
| `tracking.py` | `ExperimentTracker`: a dependency-free manifest plus JSONL and CSV metric logs (§8). |
| `cli.py` | The `omnifuser inspect` and `omnifuser train` commands, and the shared protocol/architecture arguments (`add_protocol_arguments`, the typed `protocol_kwargs`). |

### Scripts `scripts/`

| Script | Purpose |
|---|---|
| `train_lightning.py` | Full 10-fold cross-validation (`run_cross_validation`). Defaults: CUDA, direct head, horizon 96. `--dedup-identical-forces True` or `False` is required (no default), so every full run states its duplicate handling. It is a plain PyTorch job that runs in a Lightning AI Studio; it does not use the Lightning `Trainer`. |
| `run_experiment_suite.py` | The paper studies (ablations, sensitivity, missing-modality, or all three) and the `deduplication` comparison. Each variant is a resumable 10-fold run. Writes `suite.json` and `suite-summary.csv` (§9). |
| `train_local.py` | One-fold smoke test. Defaults: CPU, batch 1, horizon 96, windows capped at 2/1 unless `--full`. |
| `eda.py` | The exploratory data analysis behind `EDA.md`. Writes tables, a JSON summary, and 12 figures. |
| `compare_with_paper.py` | Prints cumulative per-horizon metrics next to Tables IV-V and the persistence baselines. |
| `model_summary.py` | Per-block parameter counts and output shapes; `--check` compares the total with a count derived from the dimensions. |
| `task_diagnostics.py` | Descriptive trivial-baseline difficulty for chosen pool factors and horizons. |

### Tests `tests/`: 124 tests

| File | Tests | What it verifies |
|---|---|---|
| `test_model.py` | 30 | Embeddings and positions; GAT normalization; image position placement; orthogonal private residual; landmark pooling (defaults and `k_colnorm` scale bound); `exact_split` reconstruction; heads; every ablation (forward/backward plus the intended structural change); masked-modality independence; paper-reading defaults; image-encoder name validation. |
| `test_alignment_patch.py` | 17 | Pooling, image lag, shown-label exposure, the image cache, ALS modes, modality masking statistics, BatchNorm freezing, gradients for every option, and an end-to-end fold with interruption and resume. |
| `test_block12_integration.py` | 23 | Architecture flags reach `ModelConfig`; invalid dimensions fail early; the full compact model runs with all three heads. `--dedup-identical-forces` parses True/False, the bare flag and every alias for `train`, `inspect` and the suite, and rejects ambiguous values. |
| `test_data.py` | 13 | Window shapes; exact input/target alignment; windows never mix tools; class change at a pass boundary; lag-1 photo with current-pass label; fold partitions; pooled normalizer statistics; collation; duplicate merging; span trimming and its documented limit. |
| `test_experiment_suite.py` | 8 | Ablation list; the 8 sensitivity axes and their field mapping; skipping baseline-valued points; dropout arms; manifest, summary CSV, resume and failure handling (mocked trainer); refusal of changed settings. The deduplication study runs both settings with everything else shared, keeps the flag out of `ModelConfig`, and fills the summary columns. |
| `test_tracking.py` | 5 | JSONL/CSV logging; corrupt manifest; refusal of mixed settings; resume on different hardware; legacy CSV migration. |
| `test_alignment_fixes.py` | 4 | Sample-weighted partial gradient accumulation; diagnostic normalizer; strict completed-fold reuse (including retraining folds without missing-modality results); dataset signatures. |
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
| `runs/` | Tracked artifacts from early smoke runs (2026-10-11). They predate the current defaults and are kept for history only. |
| `Literature/` | Paper PDF and reproduction guides (git-ignored), plus earlier review notes. |
| `omnifuser_blocks911/`, `omnifuser_blocks12/` | Archived patch bundles and review notes. Nothing imports them, and they are excluded from lint and type checks. Their reference tests import modules missing from the bundles, so they cannot run as supplied. |
| `inspect_nonastreda*.txt`, `task_diagnostics_nonastreda.json`, `setup_report.txt` | Saved outputs from earlier inspection/diagnostic runs (2026-10-11). |
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
   | `fold-Tn.pt` | Weights, config, normalizer, result |
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
  protocol: `--pool 1000 --image-lag 1`, 30 epochs, horizon 96, stride 24,
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

`pytest` passes all 124 tests. Three deliberately injected defects each make a
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

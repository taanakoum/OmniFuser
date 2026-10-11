# OmniFuser implementation and paper audit

## Purpose and scope

This document describes the current repository implementation as it exists, then
compares it with the supplied OmniFuser paper and the two reproduction guides.
It covers the Python package, scripts, tests, data layout, mathematical
operations, tensor dimensions, training/evaluation flow, and the main
reproduction decisions.

This is an implementation audit, not a claim of exact reproduction or of
matching the paper's reported results. The paper's experiments name MATWI and
Mudestreda; this repository adapts the method to the supplied Nonastreda files.
Consequently, its results are not directly comparable unless the dataset,
labels, split, preprocessing, and training protocol are also matched.

## Repository inventory

### Root files

| Path | Purpose |
|---|---|
| [`README.md`](README.md) | Installation, inspection/training commands, dataset decisions, and Lightning AI usage. It is the concise run guide; this document provides the deeper implementation and audit detail. |
| [`pyproject.toml`](pyproject.toml) | Package metadata, Python/dependency requirements, console-script registration, and pytest configuration. |
| `2511.01320v1.pdf` (local-only) | Supplied OmniFuser research paper; the architecture and experimental claims are audited against this version. The PDF is intentionally not included in the source-only GitHub branch. |
| `OmniFuser Reproduction Guide From Paper to Working Code.pdf` (local-only) | First supplied guide; informs practical defaults and data/windowing decisions where the paper is incomplete. The PDF is intentionally not included in the source-only GitHub branch. |
| `OmniFuser Reproduction Guide, Part 2 Datasets, MRTE and the Image Branch.pdf` (local-only) | Second supplied guide; discusses Nonastreda, force duplicates, MRTE, and image-branch implementation details. The PDF is intentionally not included in the source-only GitHub branch. |

### Data directory

`Nonastreda_Data/` is an input dataset, not a generated package. It contains:

| Path | Use in the loader |
|---|---|
| `Nonastreda_Data/forces_xyz_raw.mat` | MATLAB `baseDatastore`; each row maps an image filename and label to three force channels. |
| `Nonastreda_Data/labels.csv` | Record IDs, image-derived wear classes, and alternate tool labels. |
| `Nonastreda_Data/labels_reg.csv` | Record IDs and numeric wear measurements. The current force-forecasting task does not predict these numeric wear fields; `flank_wear` is used to choose between duplicate B3/B4 annotations. |
| `Nonastreda_Data/tool/` | Source JPG files addressed by the record filename. Images are mapped to their pass records in memory; the loader does not physically copy or reorganize these files. |

### Python package: `omnifuser/`

| File | Responsibilities |
|---|---|
| [`omnifuser/__init__.py`](omnifuser/__init__.py) | Package marker and version (`0.1.0`). |
| [`omnifuser/config.py`](omnifuser/config.py) | `ModelConfig`: model widths, sequence lengths, layer counts, ranks, image settings, RTD mode, and the selected prediction head. Validates the head choice and several dimension/config constraints. |
| [`omnifuser/data.py`](omnifuser/data.py) | CSV/MAT loading and cross-file validation, record parsing, optional force decimation and duplicate handling, per-tool timeline construction, tool folds, training-only force normalization, sliding windows, image transforms, and batch collation. |
| [`omnifuser/models.py`](omnifuser/models.py) | Fixed sinusoidal positions, force embeddings, RTD/MRTE blocks, image feature encoder, temporal graph attention, C2F layers, dual-task heads, loss, and small classification metrics helper. |
| [`omnifuser/train.py`](omnifuser/train.py) | One held-out-tool run: loader setup, optimizer loop, validation logging, final test pass, prediction/ROC artifacts, and checkpoint persistence. |
| [`omnifuser/evaluation.py`](omnifuser/evaluation.py) | Accuracy, precision/recall/F1, confusion matrix, one-vs-rest ROC/AUC, and force-regression metrics in normalized and original units. |
| [`omnifuser/cross_validation.py`](omnifuser/cross_validation.py) | Ten-tool discovery, repeated held-out-tool folds, per-fold means/stds, pooled out-of-fold metrics, and mean ROC curve artifacts. |
| [`omnifuser/cli.py`](omnifuser/cli.py) | `omnifuser inspect` and `omnifuser train` command-line interface and argument wiring. |

### Scripts and tests

| Path | Purpose |
|---|---|
| [`scripts/train_local.py`](scripts/train_local.py) | One-fold local smoke run; caps train/evaluation windows unless `--full` is supplied. Defaults to CPU, batch size 1, and EfficientNet-B0. |
| [`scripts/train_lightning.py`](scripts/train_lightning.py) | Full ten-fold runner intended to be launched on a GPU machine or in a Lightning AI Studio. It invokes the package's PyTorch loop; it is not a Lightning Trainer implementation. |
| [`tests/test_data.py`](tests/test_data.py) | Tests timeline/window shapes and cross-record label/image alignment, tool-level fold partitions, train-only normalization, image collation, and exact B3/B4 duplicate selection. |
| [`tests/test_model.py`](tests/test_model.py) | Tests sinusoidal and period embeddings, graph attention normalization, image position placement, orthogonal residuals, proxy pooling, both paper-literal and lead-conditioned forecast heads, model forward/backward, and classification metrics. |
| [`tests/test_evaluation.py`](tests/test_evaluation.py) | Tests classification/ROC behavior, absent-class AUC, and original-unit regression metrics. |
| [`tests/test_cross_validation.py`](tests/test_cross_validation.py) | Tests pooled out-of-fold aggregation, ROC output, and missing-AUC fold handling. |

`outputs/` contains runtime artifacts from earlier smoke/readiness runs, not
source modules. `.venv/`, `omnifuser_reproduction.egg-info/`, `__pycache__/`,
and `.pytest_cache/` are local environment, installation, or test-generated
directories and are not part of the model design.

## Notation and dimensions

| Symbol | Meaning |
|---|---|
| \(B\) | Batch size. |
| \(T\) | Observed/input time steps; default 96. |
| \(P\) | Forecast horizon; default 24. |
| \(C\) | Force channel count; 3 (`Fx`, `Fy`, `Fz`). |
| \(d\) | Shared feature width; default 32. |
| \(M\) | Number of distinct images in a collated sample/window; varies by sample. |
| \(K\) | Number of proxy landmarks per modality; default 4. |
| \(r\) | Low-rank bottleneck width; default 8 for both RTD and C2F. |
| \(H\) | GAT/MHA head count; default 4. |

Core shapes:

| Tensor | Shape |
|---|---|
| Force input | \((B,T,3)\) |
| Input images | \((B,M,3,128,384)\) by default |
| Image-to-time lookup | \((B,T)\), integer indices in `[0, M)` |
| Each encoded modality | \((B,T,d)\) |
| Future force target/output | \((B,P,3)\) |
| Future class target | \((B,P)\) |
| Future class logits | \((B,P,3)\) |

The image tensor dimensions are channel-first after collation. The image source
file itself is converted to RGB, resized to width 384 and height 128, divided
by 255, then normalized channel-wise with ImageNet mean/std.

## End-to-end data and training flow

1. **Read and validate source records.** `load_nonastreda` parses IDs matching
   `T<tool>R<run>B<block>`, checks that IDs agree among MAT and both CSV files,
   checks image files and finite force arrays, validates the selected class
   label, and requires `labels_reg.csv` to contain the same IDs and a finite
   `flank_wear` value per record.
2. **Select records and force rate.** With defaults, exact B3/B4 force
   duplicates within the same tool/run are consolidated. The retained record
   is the one with strictly larger `flank_wear`; ties retain B3. Force is not
   decimated (`decimation=1`). If requested, `scipy.signal.decimate` applies
   zero-phase filtering along the sample axis before conversion to
   time-major `float32`.
3. **Build one stream per tool.** Records are sorted by run and block, then
   concatenated into a force timeline. Every force sample receives its owning
   record's class, image ID, run/cycle code, block code, and one of 16
   within-record relative-position bins. Runs are concatenated; there is no
   explicit gap or reset token between records/runs.
4. **Assign tool-level folds.** For each test tool, the next tool in numeric
   cyclic order is validation, and the other eight are training. Thus no
   tool's samples/windows cross splits.
5. **Fit normalization only on training tools.** Per-channel mean and
   population standard deviation are accumulated from training timelines.
   A near-zero scale is replaced by 1. The same transform is used for input
   and target forces in all three splits.
6. **Create windows per tool.** Each sample has input range `[s,s+T)` and
   target range `[s+T,s+T+P)`. Defaults are `T=96`, `P=24`, and training
   stride `T+P=120`; evaluation stride is at least the horizon and, under
   default settings, is also 120. Windows never span tools but can cross
   record/pass and run boundaries within a tool. The class target for each
   future force sample is the label of the record owning that future sample.
7. **Collate and associate images.** Each distinct image used within each
   window is loaded once, then a local integer map replicates its encoded
   image feature across its input time positions. Samples with different
   image counts are padded by repeating the first image; no padded image is
   referenced by a time index.
8. **Train one model per fold.** Adam minimizes the sum of normalized-force
   MSE and per-future-step three-class cross-entropy. Gradient norm is clipped
   to 1. Validation metrics are computed each epoch, but the final epoch's
   weights—not best-validation weights—are used for the test evaluation.
9. **Persist results.** Each fold writes `fold-Tn.json`, `fold-Tn.pt`,
   `fold-Tn-predictions.npz`, and `fold-Tn-roc.npz`. The CV runner also writes
   progress JSON, final CV results, and a fold-averaged ROC NPZ.

## Model operations and equations

### 1. Force temporal embedding

Each force vector is projected through `Linear(3,d) -> GELU -> Linear(d,d)`.
The resulting representation receives fixed positional encodings and the
paper's summed learned period code, then dropout:

\[
E_t^{\mathrm{per}} =
E_c(c_t)+E_b(b_t)+E_p(p_t),\qquad
X_t = \mathrm{Dropout}\bigl(f_{\mathrm{force}}(F_t)+PE_t+E_t^{\mathrm{per}}\bigr).
\]

The fixed sine/cosine encoding follows the Transformer form:

\[
PE(p,2i)=\sin\left(p/10000^{2i/d}\right),\quad
PE(p,2i+1)=\cos\left(p/10000^{2i/d}\right).
\]

It is a non-trainable buffer (not a parameter). For odd feature widths, the
implementation truncates the cosine columns to fit; the default width 32 is
even.

The paper calls \(E_t^{\mathrm{per}}\) the **period embedding** and defines its
three categorical inputs as spindle cycle \(c_t\), cutting batch \(b_t\), and
relative position \(p_t\) within that cycle (paper Eq. 1). In this dataset
mapping, a tool record's run number supplies the cycle code; its block/pass
number supplies the batch code; and each force sample receives a relative
position code. The code quantizes that sample's fractional location in its
own record into 16 bins (`0..15`). Separate trainable PyTorch embedding
tables map cycle, batch, and relative-position IDs to 32-vectors; the vectors
are **summed**, not concatenated, with one another and with the value and
position vectors. Their table sizes are 128 cycle IDs, 8 batch IDs, and 16
relative-position IDs. These cardinalities and the relative-position
quantization are implementation choices for Nonastreda, not values stated by
the paper. Indices outside each table's range are clamped to its last valid
index; normal dataset-generated codes fit the tables.

The image branch uses the same period-code attributes and adds its vectors
after the temporal GAT stack, but it has its **own, untied embedding tables**;
the sensor and image branches do not share those learned weights. Thus, the
learned period code is a discrete learned lookup for process phase/context,
whereas the sine/cosine code provides absolute position inside the fixed input
window. It is not a continuous sinusoidal code, and it is not a pretrained
embedding. Omitting metadata at the model API supplies index zero for each
table rather than removing the period-code contribution; training paths pass
metadata explicitly.

### 2. Multi-resolution temporal extractor (MRTE / RTD)

The default has four cascaded RTD blocks with stride 3. At each resolution,
the feature sequence \(X\in\mathbb{R}^{B\times T'\times d}\) is transposed
channel-first, padded at the end to a stride multiple if needed, reduced by a
depthwise stride-3 convolution and mixed linearly per time point. The low-rank
trend and residual seasonal terms are:

\[
Trend = DUV,\quad Seasonal = D-Trend,\quad
U\in\mathbb{R}^{d\times r},\;V\in\mathbb{R}^{r\times d}.
\]

In default `decomp_mode="branch"`, each branch has its own two-layer
feed-forward transform and their outputs are summed. A learned temporal linear
map restores that level to the original input length \(T\), followed by a
channel feed-forward network. The block returns both the downsampled sequence
for the next RTD level and its restored sequence. MRTE sums the four restored
sequences into a \((B,T,d)\) sensor representation.

`decomp_mode="literal"` instead sums the raw trend and residual (which equals
the downsampled input algebraically) before the restore/channel maps. It does
not make the whole RTD block an identity because downsampling, temporal
restoration, and the channel network still operate.

### 3. Image branch and temporal GAT

The default visual backbone is torchvision EfficientNet-B0 with
`weights=None` (random initialization), its classifier omitted, adaptive
average pooling to a feature vector, and a learned projection to width \(d\).
The `compact` CNN is an alternative for smoke tests/constrained environments;
it is not the paper's EfficientNet implementation.

The collator's \((B,M,3,H,W)\) images become \(M\) vectors per sample; the
lookup indices gather these vectors into a \((B,T,d)\) visual sequence. Four
default temporal graph-attention layers then apply fully connected attention
over the \(T\) positions. For each head, attention logits use learned
source/target projections and LeakyReLU; softmax is over all key positions.
The output is a weighted sum of projected time features, concatenated across
heads, with ELU, dropout, residual connection, and LayerNorm. There is no
explicit sparse/physical graph adjacency. Sine/cosine positions and the same
run/block/relative metadata embeddings are added **after** the GAT stack.

### 4. C2F multimodal fusion

Each fusion layer receives sensor \(R\) and image \(I\) tensors in
\(\mathbb{R}^{B\times T\times d}\). The implementation executes:

1. A modality-specific multi-head self-attention residual and LayerNorm on
   both complete streams.
2. Learned shared projections \(S_m=W_m Z_m\). A per-sample
   Frobenius projection removes the component parallel to the shared feature:

\[
P_m = Z_m -
\frac{\langle Z_m,S_m\rangle_F}{\|S_m\|_F^2+\epsilon}S_m.
\]

This makes the summed time/feature inner product
\(\langle S_m,P_m\rangle_F\) approximately zero. It does not impose
orthogonality independently at each time token.
3. Modality-specific self-attention on \(P_m\), with residual normalization,
   to refine the private components.
4. Learned landmark weights `softmax(score(x), dim=time)` produce \(K\)
   weighted-average landmark keys and projected landmark values. Each
   modality's shared time tokens query the other modality's \(K\) proxies via
   scaled dot-product attention, yielding a \((B,T,d)\) cross-modal shared
   representation.
5. A hybrid gate receives the concatenated cross-attended shared and refined
   private features \((B,T,2d)\). A global linear gate and a local depthwise
   temporal convolution (kernel 3) each produce sigmoid gates of width \(d\).
   A learned scalar mixes them; it starts at 0, whose sigmoid gives equal
   0.5/0.5 local/global weights. The result interpolates shared and private
   features for each modality.
6. Sensor/image fused features concatenate to width \(2d\), pass through
   `Linear(2d,r) -> GELU -> Linear(r,d)`, and dropout.

There are two progressive fusion rounds by default. Round 1 fuses original
encoded modalities. Round 2 adds a learned linear transform of round 1's fused
sequence separately to each **original** modality stream before repeating C2F.
Shared-alignment and orthogonality scalars are returned for diagnostics.

### 5. Lead-conditioned prediction heads and objective

For final fused sequence
\(\tilde Z\in\mathbb{R}^{B\times T\times d}\), the implementation extracts
\(\tilde z_T\), concatenates it with a learned embedding \(e_p\) for each
future lead \(p\), then decodes each lead with a shared nonlinear projection:

\[
h_p = \mathrm{GELU}\left(W_f[\tilde z_T;e_p]+b_f\right),\qquad
\hat y_{T+p}=W_r h_p+b_r,\qquad
\ell_{T+p}=W_c h_p+b_c,\quad p=1,\ldots,P.
\]

Here \(e_p\in\mathbb{R}^{d}\), the concatenated decoder input is \(2d\), and
the decoded lead feature is \(d\). The heads return force estimates with
shape \((B,P,3)\) and class logits with shape \((B,P,3)\). The learned lead
embedding allows the shared decoder to condition on forecast distance while
retaining the same observed multimodal context. This is direct multi-horizon
prediction, not an autoregressive decoder: later predictions do not consume
earlier predicted values.

Two selectable head variants are implemented and share the same sensor/image
encoders, C2F fusion, losses, metrics, and data splits:

1. **`paper_literal`** applies `force_head` and `classification_head` directly
   to \(\tilde z_T\), then expands both predictions to all \(P\) leads. It
   implements the paper's printed prediction equations literally; every lead
   is consequently identical. Its weights add no lead-index parameters.
2. **`lead_conditioned`** is the default and uses the learned \(e_p\) and
   decoder above. It fixes the repeated-output limitation but is a deliberate
   extension, not the paper's exact prediction head.

Tests assert that the paper-literal head returns identical values across all
leads, while the lead-conditioned head returns lead-varying values and
receives gradients in its lead embeddings. Thus the two model variants differ
only in the prediction head; their upstream architecture and experimental
protocol can be held constant for a controlled comparison.

| Variant | Prediction equation | Expected horizon behavior | Paper status |
|---|---|---|---|
| `paper_literal` | \(\hat y_{T+p}=W_y\tilde z_T+b_y\) for every \(p\) | Identical force vector and class logits at all leads | Literal implementation of the printed prediction-head equations |
| `lead_conditioned` | \(h_p=\mathrm{GELU}(W_h[\tilde z_T;e_p]+b_h)\), then task-specific linear heads | May vary by lead; not autoregressive | Deliberate decoder extension |

The `paper_literal` mode is exact only with respect to the printed prediction
head. It does not make the full Nonastreda implementation an exact paper
reproduction: the dataset, tool-level split, preprocessing, and other
underspecified architecture choices remain as described in this audit.

Checkpoints created by the earlier repeated-head implementation do not
contain the learned lead embedding or forecast projection. Their model weights
are compatible with `paper_literal` when the rest of the configuration matches,
but not with `lead_conditioned`; retrain rather than treating such a checkpoint
as a valid lead-conditioned model.

The optimized objective is:

\[
\mathcal{L}=\mathrm{MSE}(\hat F,F_{\mathrm{target}})
+\mathrm{CrossEntropy}(\ell,C_{\mathrm{target}}).
\]

The terms are unweighted. Alignment and orthogonality values are logged but
not added to this objective. Numeric `flank_wear`, `gaps`, and `overhang`
measurements are not regression targets; the regression head predicts the
three force channels.

## Evaluation and output semantics

### Classification

`evaluation.py` flattens sample and horizon dimensions for aggregate metrics:
accuracy, macro precision/recall/F1, per-class scores and support, and a
confusion matrix. It also computes the same measures separately for each
horizon index. ROC is one-vs-rest, based on class softmax probabilities, with
per-class AUC. If a fold has no positive or no negative examples for a class,
its AUC is `null` and no curve for that class is saved.

### Force regression

Predictions and targets are inverse-normalized using training-only means and
scales before original-unit MAE, MSE, and RMSE are computed. Reports include
overall, per-channel, per-horizon, and normalized-unit metrics.

### Ten-fold aggregation

The per-fold summary reports mean/std across folds while omitting null values
from the arithmetic mean and retaining each fold's original slot. The pooled
out-of-fold confusion matrix combines test predictions from all folds; its
classification scores are computed from that pooled matrix. Force MAE/MSE/RMSE
are computed element-wise across all saved test predictions, so larger test
folds contribute in proportion to their number of target elements. The
mean-ROC curve averages interpolated fold TPRs on a 0-to-1 FPR grid. These
aggregate statistics are not interchangeable with the unweighted mean of the
fold scores.

## Paper comparison and explicit decisions

| Topic | Paper / guide evidence | Current implementation and audit status |
|---|---|---|
| Dual task | Paper predicts future wear state and multivariate cutting force from multimodal sequences. | Implements three-class future wear classification plus three-channel force prediction. The numeric wear columns are not used as regression targets. Dataset label meanings are controlled by `--label-column`; default is expert/image label. |
| Datasets | Paper's reported experiments name MATWI and Mudestreda. | This build uses the supplied Nonastreda dataset. This is a dataset adaptation and requires its own results; it cannot reproduce the paper's dataset scores by itself. |
| Sensor position and period encoding | Paper Eq. 1 defines fixed/position information plus a learned sum for spindle cycle, cutting batch, and relative position within cycle; the guide gives the sine/cosine construction. | Fixed sine/cosine \(PE_t\) and three separately learned \(d\)-dimensional embeddings are added to the force MLP. Dataset mapping uses run, block/pass, and a 16-bin normalized within-record position. Table cardinalities/binning are adaptation choices. |
| Image position encoding | Paper aligns image and sensor streams temporally; the guide recommends associating each pass image with its force samples. | Each pass image is gathered for its corresponding input force samples. The code does not copy image files. Position and metadata embeddings are added after GAT. |
| Force signal rate | The paper/guide discusses dataset sampling and frequency preservation; the supplied guide favors retaining the original rate for initial runs. | Default decimation 1 retains stored samples. Optional decimation uses SciPy zero-phase filtering. The loader itself does not verify sampling-rate metadata or reconstruct timestamps. |
| Pass duplicates | The paper does not define this supplied-data cleanup. The second guide recommends identifying exact B3/B4 duplicates. | Actual data inspection found 128 exact B3/B4 pairs; default retains one record per exact pair using larger `flank_wear` (tie -> B3). This changes the data from 512 to 384 records and changes retained class counts. `--keep-identical-forces` is available for an ablation. |
| Pass images/labels | Paper describes synchronized images and force sequences; guide stresses one image per pass and matching labels. | Record IDs, image filenames, force samples, and CSV labels are validated, then aligned in-memory on per-tool timelines. Test coverage verifies alignment across a record boundary. No image copying/relabeling stage exists. |
| Window lengths | Reproduction guides use 96 input samples and a 24-step target as working defaults. | `T=96`, `P=24` defaults are configurable. The head's identical prediction at every future step remains a consequence of the printed equations. |
| Window stride | Paper's data split/window implementation is not fully reproducible from its description; guide recommends avoiding overlapping evaluation targets. | Training defaults to stride 120 (`T+P`); evaluation uses at least `P`, and at defaults also 120. This reduces overlapping samples/targets but omits many possible start positions compared with dense stride-1/stride-24 evaluation. A user-supplied stride can change this behavior. |
| Tool folds | The paper reports experiments on its datasets, not this repository's specific ten-tool 8/1/1 rule. The guide recommends keeping tools grouped to prevent leakage and describes 9/1 train/test. | Implements ten-fold leave-one-tool-out testing with the next numeric tool held out for validation: 8 train, 1 validation, 1 test. This is the user's requested/guide-informed protocol, not the paper's stated protocol. |
| Normalization | Train/test protocol details are not fully specified for this adaptation. | Per-force-channel population mean/std from training-tool timelines only. It is a leakage-conscious implementation choice. |
| Image backbone | Paper specifies the image pathway and reports EfficientNet-based features. | EfficientNet-B0 feature trunk is used, initialized with no pretrained weights. ImageNet input normalization is still applied; this is a chosen preprocessing convention, not an indication of pretrained weights. Compact CNN is for smoke tests only. |
| RTD decomposition | The paper's printed trend-plus-seasonal expression reduces algebraically to the original downsampled representation. The guide discusses this ambiguity. | `branch` mode applies separate learned FFNs to trend and residual seasonal branches before restoration. This is an explicit interpretation, not a uniquely recoverable paper implementation. `literal` mode preserves the literal sum before subsequent transforms. |
| RTD depth/rank/stride | Paper describes multiresolution low-rank temporal decomposition; practical implementation details are incomplete. | Four stride-3 blocks and rank 8 are configuration defaults informed by the reproduction guide and implementation feasibility. They must be reported as choices, not asserted as exact paper hyperparameters unless confirmed in a paper table. |
| C2F shared/private split | Paper defines a learned shared projection and Frobenius-orthogonal private residual. | Implements this projection across all time/features per sample with epsilon stabilization. It logs the resulting dot-product diagnostic but does not use an explicit orthogonality penalty in the optimized loss. |
| C2F private attention | Paper describes self-attention refinement of private features and proxy cross-attention over shared features. | Implements private-stream self-attention and also applies an additional self-attention residual to each full modality before decomposition in every fusion layer. That extra pre-decomposition attention is an implementation addition. |
| Proxy landmarks | Paper introduces adaptive landmarks and proxy-based cross-modal attention; scoring details are not completely specified. | Learned per-time logits are softmax-normalized over time to form \(K=4\) weighted average landmarks and values. The exact scorer/value formulation is a practical interpretation. |
| Hybrid local/global gate | Paper combines global and local gate terms; exact convolution/gate details are incomplete. | Local gate uses depthwise Conv1d kernel 3. The global/local mixing scalar starts at 0, so sigmoid mixing starts evenly at 0.5. These are explicit selected values. |
| Progressive fusion | Paper describes recursive anchoring and reports two refinement iterations. | Two layers by default; each later layer adds a learned transform of the last fused representation to both original modality streams before fusion. |
| Prediction heads | Paper equations (including the force head on page 7) use the same final fused token with the same linear map at every future time, which produces identical outputs across leads. | Both variants are available: `paper_literal` follows the paper equation and repeats outputs; `lead_conditioned` (default) concatenates a learned per-lead embedding with final fused context before shared decoding. The second resolves repeated outputs but is a deliberate deviation. |
| Loss | Paper defines the two prognostic tasks; objective weighting/regularization for this dataset is not fully determined in the sources used here. | Unweighted force MSE + per-step class cross-entropy. C2F alignment/orthogonality are diagnostics only. This must be included in any reproduction report. |
| Optimizer | Paper reports Adam, learning rate \(10^{-4}\), batch size 32, dropout 0.1, and two progressive fusion iterations (paper page 9). | Adam and \(10^{-4}\), dropout 0.1 and two rounds align with those reported values. Local CLI uses batch 2, local smoke uses 1, CV script defaults to 16; paper batch 32 requires explicit `--batch-size 32`. |
| Epoch/model selection | The code and guides support runnable experiments; the exact lifecycle should not be confused with a paper-reported best-checkpoint result. | Defaults are five epochs in the training API/CV script and one in local smoke. Validation is evaluated but no early stopping, scheduler, or best-checkpoint selection is implemented. Test metrics are from the last epoch. |
| Lightning AI | Lightning AI is the intended compute platform, not a paper architecture requirement. | Script is a standard PyTorch command-line job usable in a Lightning AI Studio. It does not use Lightning Fabric/Trainer, distributed training, managed checkpoints, or cloud-specific dataset transfer. |

### Reasons for under-specified values

- **96/24 windows and 128x384 images:** use the supplied reproduction guides'
  practical starting values and preserve a configurable interface. They are
  not inferred from a hidden code artifact.
- **Stride 120:** equals `T+P`, so adjacent default windows do not share input
  or target samples. This limits redundant training and makes evaluation
  targets non-overlapping; it is an efficiency/protocol choice that must be
  held constant for comparisons.
- **Deduplication representative:** the guides call out exact B3/B4 force
  duplicates. Retaining the larger numeric flank-wear annotation keeps the
  record most consistent with the more worn state; the tied case retains B3
  deterministically. Since class/image may change, this is not a neutral
  cleanup and keeping both is exposed as an ablation.
- **Train-only standardization:** estimating mean/scale from training tools
  avoids test leakage. Population rather than sample standard deviation is
  used because the stream is treated as the training population.
- **RTD branch interpretation:** `trend + seasonal` as printed cancels to
  `down`; separate branch transforms provide a nontrivial decomposition path
  consistent with the stated role of RTD. The code retains a literal option
  for sensitivity analysis.
- **Proxy pooling and gate:** time-softmax weighted pooling provides a simple,
  trainable interpretation of adaptive landmarks; kernel 3 supplies a local
  temporal receptive field; scalar-logit zero initialization gives an equal
  initial gate mixture. These values are explicit engineering decisions.
- **Compact CNN:** allows inexpensive pipeline tests without downloads or
  large vision compute. It should not be used for a paper-alignment result.
- **No alignment/orthogonality penalty:** the code reports those values but
  does not invent objective coefficients absent an unambiguous supported
  specification. This means the claimed constraints are architectural
  projection/diagnostics, not an additional regularized training objective.

## Validation performed and limits

- Real-data inspection was run with the repository's default loader. It
  reported **384 retained records**, **10 tools**, class counts
  **sharp 148 / used 119 / dulled 117**, force lengths **52,002–152,668**
  (median **98,528**) samples after no decimation, and **zero remaining exact
  B3/B4 duplicates**. Timeline sample counts were produced for all ten tools.
- The full test suite passed after adding the selectable prediction heads
  (**22 tests**). Tests cover both head behaviors, reject an unknown head
  setting, and verify the lead-conditioned embeddings receive gradients. The
  paper-literal head outputs are mathematically identical at all forecast
  leads.
- Two local T10 smoke runs completed with the compact image encoder, the same
  seed and capped window counts, one using each prediction head. Both wrote
  fold JSON, checkpoint, prediction, and ROC artifacts; result JSON and
  checkpoint configurations recorded the selected head. In the saved force
  predictions the paper-literal run's maximum across-lead difference was
  exactly 0, while the lead-conditioned run's was nonzero. These short-run
  metrics are not performance estimates.
- The new period
  embedding regression test independently reconstructs the sum of value,
  fixed position, and all three learned period embeddings, and verifies that
  all three embedding tables receive gradients.
- The current local EfficientNet-B0 T10 smoke run completed one epoch on CPU
  with 2 training windows and 1 validation/test window. It wrote the fold JSON,
  checkpoint (~17 MB), predictions, and ROC artifact. The smoke test's scores
  are not performance estimates.
- After introducing the lead-conditioned decoder, the EfficientNet-B0 T10
  smoke was rerun successfully; it again wrote checkpoint, prediction, and ROC
  artifacts. A capped compact-encoder run of all ten folds also completed and
  saved all ten fold checkpoints/predictions plus pooled out-of-fold metrics
  and mean ROC. On the first capped fold, per-lead standard deviations were
  nonzero for both force predictions and class probabilities, confirming the
  outputs now vary with forecast lead. These smoke metrics are not model
  quality estimates.
- The current ten-fold evaluation entry point completed all folds locally
  using the compact test encoder, 2 training windows and 1 validation/test
  window per fold. It wrote ten fold checkpoints and prediction files plus
  the CV summary and mean ROC artifact. This verifies orchestration and
  evaluation outputs, not full-model performance or the full-data workload.
- At default stride 120 and batch size 16, one complete fold has approximately
  247,000–265,000 training windows (about 15,500–16,600 optimizer steps per
  epoch). Ten folds at the script's default five epochs imply roughly
  780,000 optimizer steps in total. This is executable in principle but is a
  substantial compute commitment; runtime and cost must be measured on the
  target GPU before launching all folds.
- There is no mid-epoch checkpoint or resume option. A fold checkpoint is
  written after that fold's final test evaluation, while a progress JSON is
  written only after each fold. An interruption during a fold loses that
  fold's in-progress model/optimizer state; rerunning the CV command starts
  the loop from fold 1.
- A CPU-only local environment was available; a complete uncapped GPU run on
  Lightning AI has **not** been performed. The 3 GiB CUDA check is a guard, not
  a guarantee of sufficient memory. Full-run memory, throughput, convergence,
  and model quality remain to be measured on the target GPU.
- Unit tests validate local invariants but do not prove every real-record
  boundary, paper equation, or benchmark metric. No paper score reproduction
  is claimed.

## Audit conclusions and recommended follow-up

The codebase has a runnable, tested end-to-end path for the supplied data,
with explicit validation of source IDs, record-image-label mapping,
tool-grouped folds, normalization scope, model tensor shapes, metrics, and
artifact aggregation. It contains the requested local smoke and full
ten-fold/GPU-capable entry points.

The embedding equations are internally consistent: the value MLP lifts each
3-channel force vector to \(d=32\); fixed sine/cosine positions distinguish
locations within a window; the learned period code adds categorical run,
pass, and within-record phase context. The mapping follows the paper's three
period attributes but uses a dataset-specific interpretation and table
cardinalities as described above.

The prediction head can be selected for a controlled comparison: `paper_literal`
exposes the repeated-horizon behavior implied by the printed equation, while
`lead_conditioned` avoids that limitation with an explicit non-paper extension.
Neither choice makes the entire implementation an exact reproduction. Other
essential reporting items are the
Nonastreda-vs-paper dataset difference, duplicate consolidation, 8/1/1 split,
RTD branch interpretation, unweighted loss, random EfficientNet initialization,
batch-size deviation, last-epoch test checkpoint, and default stride.

For a defensible full experiment, first run one uncapped fold on the intended
Lightning AI GPU, record peak memory and epoch time, then run all ten folds
with fixed settings. Preserve each fold's JSON/checkpoint/predictions, report
both fold-level and pooled out-of-fold results, and label all methodological
departures from the paper.

# OmniFuser reproduction: Nonastreda

This repository implements an initial, testable OmniFuser reproduction for the supplied Nonastreda data. The implementation follows the paper's multimodal sensor/image design while making paper ambiguities explicit in configuration. It is a research reproduction, not a claim of reproducing the paper's reported scores.

## Environment

Create/activate a Python environment, then install the package and test dependencies:

```powershell
python -m pip install -e ".[test]"
```

The supplied dataset is expected at `Nonastreda_Data/` with `labels.csv`,
`labels_reg.csv`, `forces_xyz_raw.mat`, and the `tool/` image directory. To
inspect its contents and report force/record alignment:

```powershell
omnifuser inspect --data-dir Nonastreda_Data
```

Train and evaluate a leave-one-tool-out fold (all windows from each tool remain
in one split):

```powershell
omnifuser train --data-dir Nonastreda_Data --test-tool T10 --epochs 5 --device cuda
```

Use `omnifuser train --help` to see the resource and fold options. Runs default
to CPU to avoid exhausting limited GPU memory. Set `--device cuda` only when
enough GPU memory is available.

`scripts/train_lightning.py` is a GPU-oriented launcher for this repository's
native PyTorch training loop; it does **not** use the PyTorch Lightning
framework. CUDA runs enable pinned host-memory batches and non-blocking
transfers. Confirm the selected CUDA device and available memory with a short
fold pilot before starting all ten folds. Mixed precision is not enabled; use
the FP32 path unless you separately validate another precision mode.

Run a capped local end-to-end check with the full EfficientNet-B0 model:

```powershell
.\.venv\Scripts\python.exe scripts\train_local.py
```

Compare the prediction heads on the same capped T10 fold and seed using
separate artifact directories:

```powershell
.\.venv\Scripts\python.exe scripts\train_local.py --prediction-head paper_literal `
  --output-dir outputs\comparison-paper-literal
.\.venv\Scripts\python.exe scripts\train_local.py --prediction-head lead_conditioned `
  --output-dir outputs\comparison-lead-conditioned
.\.venv\Scripts\python.exe scripts\train_local.py --prediction-head direct `
  --output-dir outputs\comparison-direct
```

Use `--image-encoder compact` to make this smoke comparison faster on CPU. These
short runs verify that both model variants train and emit evaluation artifacts;
their metrics are not meaningful performance comparisons. For a full
cross-validation comparison, run `scripts\train_lightning.py` twice with the
same settings and seed, changing `--prediction-head` and using a different
`--output-dir` for each run.

Run a full 10-fold experiment on a GPU machine (including a Lightning AI
Studio):

```bash
python scripts/train_lightning.py \
  --data-dir /path/to/Nonastreda_Data \
  --output-dir /path/to/persistent-storage/omnifuser \
  --epochs 5 --batch-size 16 --device cuda \
  --prediction-head lead_conditioned
```

For each held-out test tool, the next tool in numeric order is validation and
the other eight tools are used for training. Thus training and validation draw
from nine tools, and test data is isolated to the tenth tool. The full run
aggregates all ten test folds as out-of-fold results.

The evaluation reports accuracy, macro and support-weighted precision,
recall/F1, per-class scores, confusion matrices, per-horizon classification
scores, and one-vs-rest ROC AUC. ROC points are stored in each `fold-Tn-roc.npz`;
a fold-averaged ROC curve is stored in `cross-validation-mean-roc.npz`. Force
regression MAE, MSE, and RMSE are reported in original force units overall, per
force channel, and per horizon; normalized-unit metrics are included as well.
For horizons 12, 24, ... 96 that fit the configured output, the fold JSON also
contains cumulative z-scored model and persistence-baseline metrics, matching
the units of paper Tables IV-V. The paper does not specify classification
averaging; both macro and weighted results are retained. Compare a finished
fold or CV output with the paper and the Nonastreda baselines using:

```powershell
.\.venv\Scripts\python.exe scripts\compare_with_paper.py runs\my-experiment
```

The paper's benchmark dataset is Mudestreda; this script juxtaposes those
published values with Nonastreda results as a plausibility check, not a
dataset-equivalent reproduction. Per-fold test predictions and labels are
stored in `fold-Tn-predictions.npz`.

For the Blocks 9-11 comparison arm, start with one T10 fold at the selected
Nonastreda settings. On a CPU-only machine use a short validation run first:

```powershell
.\.venv\Scripts\python.exe -m omnifuser.cli train `
  --data-dir Nonastreda_Data --test-tool T10 `
  --output-dir runs\a_lag1_pool1000_direct_cpu `
  --pool 1000 --image-lag 1 --horizon 96 --stride 24 `
  --prediction-head direct --epochs 5 --device cpu
.\.venv\Scripts\python.exe scripts\compare_with_paper.py `
  runs\a_lag1_pool1000_direct_cpu
```

The requested full run is the same command with `--epochs 30 --device cuda`;
run it on a CUDA-equipped host or Lightning AI Studio. Treat the five-epoch
CPU result as an execution check, not a stable performance estimate.

### Experiment tracking

Each run writes append-only tracking files under `<output-dir>/tracking/`
without requiring W&B, TensorBoard, or another hosted service:

- `run.json` records the run ID, data/source manifest, fold protocol,
  preprocessing/window settings, model and optimizer configuration, Python,
  PyTorch and CUDA versions, and device/GPU information.
- `metrics.jsonl` stores nested records for every training epoch, validation
  epoch, final test fold, and final cross-validation aggregate.
- `metrics.csv` stores each numeric metric as a row keyed by split, fold,
  epoch, and metric name for plotting or import into a dashboard.

Train/validation records include loss components, classification accuracy,
macro and weighted precision/recall/F1, per-class scores, ROC AUC, and force
MAE/MSE/RMSE in normalized and original units overall, per channel, and per
lead. Test records also include cumulative paper-horizon scores and
persistence baselines. Use a unique output directory for each independent
experiment; resumed training appends to that run's logs. Archive or upload
the `tracking/` directory together with the fold artifacts.

## Preparing a full experiment

The supplied Nonastreda files contain one image and one label per record/pass.
The loader validates the IDs against both CSV files, including
`labels_reg.csv`, and resolves photos by record ID; it does not copy or rename
source images. By default, exact duplicate force records within the same
tool/run are merged, retaining the photo and label for the row with the highest
measured flank wear. Each retained force record is span-trimmed at its
lead-in/run-out before timelines are built. In timeline windows, each force
sample inherits the image, wear class, run, and block code of its owning
record; future class targets come from the records owning the future force
samples. A window may cross a record boundary, so each pass keeps its own
image association. The collator loads each distinct image in a window once and
maps its input steps back to that image.

This matches the paper's temporal image association semantically: the pass
image is associated with every force timestep belonging to that pass, and a
new pass uses its own image. The code does not physically duplicate image
files or image tensors for each force sample. Instead, `image_codes` repeats
the image index across the pass timeline, each window's input timesteps map
to those indices, and the collator loads each distinct image once per batch
window before the image branch gathers it back onto the time axis. The default
`--image-lag 0` uses the image for the same pass (the paper-literal association).
`--image-lag 1` instead uses the preceding pass's image in the tool's record
order while preserving the current pass's force and target labels; that is a
causal timing ablation, not the
literal same-pass mapping. Choose lag 1 if the intended prediction must
not use a photo taken after its pass; choose lag 0 to follow the paper's
same-pass alignment.

The full-data defaults are deliberately set for the supplied recordings:
decimation is 1 to retain force-frequency content, and the training stride
defaults to `input_len + horizon` (120 with the default 96/24 window) to avoid
overlapping training targets and excessive window counts. Validation and test
also use non-overlapping targets. The supplied guide's real-file inspection
describes Nonastreda as single steady cuts with short lead-in/run-out, not
bursts. Preprocessing uses span trimming rather than burst segmentation.
Disable it with `--no-trim-pass` for an untrimmed ablation.
`omnifuser inspect` reports aggregate lead-in/run-out samples removed and the
ten retained records with the largest removed fraction. This is important
because span trimming can remove genuine low-force cut data: the current
default retains `T6R2B4` after duplicate resolution and removes 320 lead-in and
12,639 run-out samples from its 96,204-sample source record. Inspect that report
and compare with `--no-trim-pass` before treating trimmed results as definitive.

Optional alignment and training controls are exposed consistently by
`omnifuser train`, `scripts/train_local.py`, and
`scripts/train_lightning.py`:

- `--pool N` block-mean pools each trimmed record independently; an incomplete
  trailing block is dropped and reported. Pooling changes the effective time
  step (for example, `--pool 100` represents 0.1 seconds at the stored 1 kHz
  rate). It is an experiment parameter, not a paper-specified preprocessing
  default.
- `--image-lag 1` associates each record's force samples with the preceding
  record in that tool's ordered timeline while keeping force and class targets
  on the current record. The first record in the timeline falls back to its
  own image. The paper-literal default is
  `--image-lag 0`.
- `--als-mode k_sum` uses the landmark-axis softmax printed in paper Eq. 10;
  the default `time` mode instead normalizes each landmark's weights over
  time. `k_colnorm` is another normalized variant. These modes are explicit
  alternatives because the paper notation does not fully resolve the
  normalization details.
- `--share-c2f` ties C2F weights across recursive rounds (one shared \(G\));
  without it, each round has its own C2F parameters. Eq. 20 writes one \(G\),
  while the architecture figure depicts stacked layers, so neither reading is
  conclusively specified.
- `--private-attn plain` uses the MHA output directly as printed; the default
  `residual_ln` preserves the residual-plus-LayerNorm implementation. These
  settings expose the ambiguity without silently changing the established
  default.
- `--prediction-head direct` is the lead-specific comparison head. The
  `paper_literal` head repeats the final-token prediction at every lead, as the
  printed equations imply, while `lead_conditioned` adds a learned lead code.
  The latter two are comparisons because the paper's plotted trajectories are
  lead-varying.
- `--modality-dropout`, `--pretrained`, `--freeze-bn`, and `--pre-attn` are
  optional model experiments. Dropout defaults to 0 because the paper does not
  specify a rate; pretrained ImageNet EfficientNet weights require downloading
  weights, and extra pre-attention is not in the paper architecture.
- `--accumulation-steps` accumulates gradients with sample-correct weighting,
  including an incomplete final accumulation group. `--select-best` evaluates
  the checkpoint with the lowest validation loss (default); `--no-select-best`
  evaluates the last epoch. `--resume` restores matching interrupted folds,
  including optimizer and random-number-generator state, and rejects
  incompatible saved state. Completed CV folds are reused only when their
  recorded data and experiment settings still match.

The full training run also reports persistence and force baselines plus
cumulative paper-horizon summaries where supported by the configured horizon.
The separate diagnostic script compares candidate pool factors and horizons;
its sampled windows and all-tools normalization are descriptive diagnostics,
not model validation scores:

```powershell
.\.venv\Scripts\python.exe scripts\task_diagnostics.py `
  --data-dir .\Nonastreda_Data --pools 1 10 100 `
  --horizons 12 24 96 --max-windows-per-tool 100
```

All bit-identical force records within a tool/run are consolidated by default,
not only B3/B4. On the supplied data this reduces 512 rows to 378 distinct
recordings; `labels_reg.csv` flank wear selects the retained row's image and
label, matching the guide's recommended `dedup_force` forecasting policy.
`--keep-identical-forces` retains all rows as an ablation. Force forecasting
is the paper's regression task. `labels_reg.csv` also contains numeric wear
measurements, but those are not additional model targets in the paper's stated
Nonastreda force-forecasting/classification setup.

The updated guide reports 50,317–150,711 samples after span trimming and
308,071 timeline windows at stride 120. The current loader reports
50,212–149,011 samples and 309,140 windows on these files. Because the guide
does not specify the exact span-envelope estimator, these small differences are
documented rather than hidden or tuned to match aggregate counts.

To run on Lightning AI, provision a Studio with a GPU and enough host RAM for
the loaded force streams, place the dataset in persistent Studio storage, then
install and run the project from the Studio terminal:

```bash
python -m pip install -e ".[test]"
python -c "import torch, torchvision; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); print('torchvision', torchvision.__version__)"
omnifuser inspect --data-dir /path/to/Nonastreda_Data
omnifuser train --data-dir /path/to/Nonastreda_Data --test-tool T10 \
  --epochs 50 --batch-size 16 --device cuda \
  --prediction-head lead_conditioned \
  --output-dir /path/to/persistent-storage/omnifuser
```

Run all ten held-out-tool folds from the Studio terminal with:

```bash
for tool in T1 T2 T3 T4 T5 T6 T7 T8 T9 T10; do
  omnifuser train --data-dir /path/to/Nonastreda_Data --test-tool "$tool" \
    --epochs 50 --batch-size 16 --device cuda \
    --prediction-head lead_conditioned \
    --output-dir /path/to/persistent-storage/omnifuser
done
```

Each fold writes a distinct `fold-Tn.json` and `fold-Tn.pt` file. Keep the data
and checkpoints on persistent storage rather than relying on an ephemeral
runtime disk. Start with one full fold and monitor GPU memory, host RAM, and
step time and inspect `tracking/metrics.csv` before launching all folds. If
the CUDA check says `available False`, stop and resolve the Studio's
PyTorch/CUDA/driver setup before starting training. Reduce batch size if GPU memory is
constrained; use `--decimation 3` only as a separately reported lower-sample-
rate experiment. With the default `--select-best`, test metrics are computed
from the minimum-validation-loss epoch. Resume state is saved at epoch
boundaries; an interruption mid-epoch resumes from the most recently completed
epoch using the recorded compatible settings.

## Reproduction choices

- `image_label` is the default three-class target (`sharp`, `used`, `dulled`);
  `tool_label` is retained as an alternate label source.
- Force-variable names and record mapping are discovered and validated from the
  supplied MAT file; no assumed MAT layout is silently accepted.
- Force arrays that are bit-identical within a tool/run are consolidated by
  default, including B1/B2 duplicates in the few runs where they match. The
  greatest-flank-wear row supplies the retained image/label;
  `--keep-identical-forces` disables consolidation.
- Nonastreda uses span-only pass trimming by default; its specific threshold
  details are implementation choices described in the audit. Use
  `--no-trim-pass` for an untrimmed ablation. `inspect` also reports the
  per-record trim counts and largest removed fractions; the retained T6R2B4
  record has a documented genuine-cut-tail caveat.
- Force arrays are not decimated by default. The guide recommends a first run
  at the stored 1 kHz rate to preserve the observed 66.7 Hz line and its
  harmonics; use `--decimation 3` as a separately evaluated efficiency
  trade-off. Filtering and decimation use `scipy.signal.decimate` with
  zero-phase filtering.
- The default sample window and forecast horizon are 96 and 24. Window splits
  are grouped by tool to prevent neighboring-window leakage.
- The training fold uses eight tools, the next tool in numeric order for
  validation, and the selected tool for test. Force normalization is fit only
  on training tools. Training stride defaults to the 120-step input-plus-
  forecast length; evaluation targets do not overlap.
- This is the guide's 8/1/1 train/validation/test variant of leave-one-tool-out,
  not a split stated in the paper. The dataset guide also describes a 9/1
  train/test protocol; report which one is used when comparing results.
- The RTD equation as printed cancels to identity. Both a literal mode and a
  branch-decomposition interpretation are represented in configuration; the
  branch mode is the default and is identified as an interpretation, not a
  uniquely specified paper result.
- The temporal path uses four stride-3 RTD layers, and the image path defaults
  to four-head temporal GAT plus EfficientNet-B0. C2F uses shared/private
  projections with the paper's Frobenius-orthogonal private residual, learned
  proxy landmarks, cross-modal attention, global/local hybrid gates, bottleneck
  low-rank fusion, and recursive re-injection of the original modalities.
  Fixed sine/cosine positional encodings and learned period embeddings are
  used in both modality streams: the dataset run number supplies the spindle
  cycle ID, the block/pass number supplies the batch ID, and each sample's
  fractional position in its record is quantized into one of 16 learned
  relative-position bins. The three learned vectors are summed with the value
  and position embeddings. The 16-bin mapping is a Nonastreda implementation
  choice; see [IMPLEMENTATION_AUDIT.md](IMPLEMENTATION_AUDIT.md).
- Three prediction heads are available with `--prediction-head`:
  `paper_literal` reproduces the paper's printed final-token linear equations
  and therefore repeats the same result at every lead; `lead_conditioned`
  (default) adds a learned forecast-step embedding to the final fused context
  before shared decoding; `direct` uses separate output weights per forecast
  lead. The latter two enable lead-specific outputs and are deliberate,
  documented deviations from the printed equation. Compare all variants with
  identical data, fold seeds, and training parameters, and use separate output
  directories so fold artifacts cannot overwrite one another.
- Paper hyperparameters include Adam at 1e-4, four GAT layers/heads, four
  landmarks, and two progressive-fusion rounds. The local training batch
  defaults to 2 rather than the paper's 32 to accommodate limited compute; use
  `--batch-size 32` on a suitable machine. EfficientNet-B0 is initialized
  without pretrained weights or downloads. A compact encoder option is
  provided for smoke tests and constrained devices.

This is still a reproduction implementation rather than a bit-for-bit
replication: the paper does not fully specify the depthwise hybrid-gate kernel,
the learned ALS scoring function, or several data-splitting and preprocessing
details. We implement ALS as learned per-landmark softmax pooling normalized
over time, making each proxy a weighted temporal average; the depthwise gate
uses a kernel of 3 and starts as an equal local/global mixture. The combined
objective is also unspecified; we use unweighted future force MSE plus
per-step wear-state cross-entropy and report alignment/orthogonality
diagnostics without adding extra regularization terms. The RTD branch
decomposition is an explicit interpretation because the printed
trend-plus-seasonal equation algebraically cancels. The dataset has exact B3/B4
force duplicates; the default consolidation policy is based on measured flank
wear, while retaining both records is available as an ablation. Results should
be reported with these choices and not compared as if the smoke run were a
paper benchmark.

The first validation run should use a single held-out tool, a small model, and a
few epochs. Full leave-one-tool-out experiments and hyperparameter selection
should be reported separately from this smoke test.

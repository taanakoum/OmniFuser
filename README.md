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

Run a capped local end-to-end check with the full EfficientNet-B0 model:

```powershell
.\.venv\Scripts\python.exe scripts\train_local.py
```

Compare the two prediction heads on the same capped T10 fold and seed using
separate artifact directories:

```powershell
.\.venv\Scripts\python.exe scripts\train_local.py --prediction-head paper_literal `
  --output-dir outputs\comparison-paper-literal
.\.venv\Scripts\python.exe scripts\train_local.py --prediction-head lead_conditioned `
  --output-dir outputs\comparison-lead-conditioned
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

The evaluation reports accuracy, macro precision/recall/F1, per-class scores,
confusion matrices, per-horizon classification scores, and one-vs-rest ROC
AUC. ROC points are stored in each `fold-Tn-roc.npz`; a fold-averaged ROC
curve is stored in `cross-validation-mean-roc.npz`. Force regression MAE,
MSE, and RMSE are reported in original force units overall, per force channel,
and per horizon; normalized-unit metrics are included as well. Per-fold test
predictions and labels are stored in `fold-Tn-predictions.npz`.

## Preparing a full experiment

The supplied Nonastreda files contain one image and one label per record/pass.
The loader validates the IDs against both CSV files, including
`labels_reg.csv`. In timeline windows, each force sample inherits the image,
wear class, run, and block code of its owning record; future class targets come
from the records owning the future force samples. A window may cross a record
boundary, so a new pass's image is used for its own samples and subsequent
windows. The collator loads each distinct image in a window once and maps its
input steps back to that image.

The full-data defaults are deliberately set for the supplied recordings:
decimation is 1 to retain force-frequency content, and the training stride
defaults to `input_len + horizon` (120 with the default 96/24 window) to avoid
overlapping training targets and excessive window counts. Validation and test
also use non-overlapping targets. The guide's measured Nonastreda files are
single steady cuts with no idle interval, so the whole record is retained; no
automatic burst segmentation is applied.

Exact B3/B4 force duplicates are consolidated by default for timeline
forecasting. `labels_reg.csv`'s flank-wear value chooses which duplicate
record's image and class annotation to keep, matching the guide's
`dedup_force` recommendation; `--keep-identical-forces` retains both as an
ablation. Force forecasting is the paper's regression task. `labels_reg.csv`
also contains numeric wear measurements, but those are not additional model
targets in the paper's stated Nonastreda force-forecasting/classification
setup.

To run on Lightning AI, provision a Studio with a GPU and enough host RAM for
the loaded force streams, place the dataset in persistent Studio storage, then
install and run the project from the Studio terminal:

```bash
python -m pip install -e ".[test]"
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
step time before launching all folds. Reduce batch size if GPU memory is
constrained; use `--decimation 3` only as a separately reported lower-sample-
rate experiment.

## Reproduction choices

- `image_label` is the default three-class target (`sharp`, `used`, `dulled`);
  `tool_label` is retained as an alternate label source.
- Force-variable names and record mapping are discovered and validated from the
  supplied MAT file; no assumed MAT layout is silently accepted.
- B3/B4 force records can be exact duplicates even though their photos and
  labels remain distinct. The default consolidates exact duplicates and keeps
  the image/class annotation with larger measured flank wear; use
  `--keep-identical-forces` to retain both.
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
- Two prediction heads are available with `--prediction-head`: `paper_literal`
  reproduces the paper's printed final-token linear equations and therefore
  repeats the same result at every lead; `lead_conditioned` (default) adds a
  learned forecast-step embedding to the final fused context before shared
  decoding, allowing lead-specific outputs. The latter is a deliberate,
  documented deviation from the paper. Compare the variants using the same
  data, fold seeds, and training parameters, and use separate output
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

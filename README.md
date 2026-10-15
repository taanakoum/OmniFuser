# OmniFuser on Nonastreda

This repository is a tested PyTorch implementation of **OmniFuser**
(arXiv:2511.01320): multimodal fusion of cutting-force signals and tool
photos for predictive maintenance. It forecasts the next \(P\) force readings
and the tool's wear class (sharp / used / dulled) at each of those steps. The
model runs on the **Nonastreda** milling dataset.

It is a research implementation: every paper equation is mapped to code, and
every gap in the paper is filled with a documented choice. The paper reports
results on other datasets ("Mudestreda", MATWI), so its scores are reference
points, not targets this code claims to reproduce.

## Documentation

| Document | Read it for |
|---|---|
| [`TRAINING_RUNBOOK.md`](TRAINING_RUNBOOK.md) | **Running anything**: Lightning AI setup for first-time users, and every command with every argument explained |
| [`EDA.md`](EDA.md) | The data: passes, labels, alignment, folds, the time step, label leakage, and every EDA metric |
| [`PAPER_IMPLEMENTATION_ALIGNMENT.md`](PAPER_IMPLEMENTATION_ALIGNMENT.md) | Paper versus code: a verdict for each of 35 items, deviations and their flags, and the paper's own inconsistencies |
| [`IMPLEMENTATION_AUDIT.md`](IMPLEMENTATION_AUDIT.md) | Code reference: every module, the model equations and tensor shapes, training, metrics, tracking, cross-validation, tests, and verification |
| [`experiment_results/README.md`](experiment_results/README.md) | Layout of the generated results |

## Repository layout

```text
omnifuser/            the package: config, data, models, train, evaluation,
                      cross_validation, tracking, cli
scripts/              train_lightning.py (10-fold CV), run_experiment_suite.py
                      (paper studies), train_local.py (smoke test), eda.py,
                      compare_with_paper.py, model_summary.py, task_diagnostics.py
tests/                119 pytest tests
experiment_results/   all generated outputs (git-ignored except its README)
Nonastreda_Data/      input data (not in git): forces_xyz_raw.mat, labels.csv,
                      labels_reg.csv, tool/*.jpg
Literature/           paper PDF and reproduction guides (not in git)
runs/, omnifuser_blocks*/   archived early runs and patch bundles (not used)
```

## Install

Python 3.10 or newer is required. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test,eda]"      # add ,dev for ruff and pyright
```

The `test`, `eda` and `dev` extras add pytest, matplotlib and the static
checkers respectively. Training itself needs only numpy, Pillow, scipy, torch
and torchvision.

## Quick start (local CPU)

```bash
python -m pytest -o addopts= -q tests                    # 119 tests
omnifuser inspect --data-dir Nonastreda_Data --pool 1000 --stride 24
python scripts/eda.py --data-dir Nonastreda_Data --out experiment_results/eda
python scripts/train_local.py --data-dir Nonastreda_Data --image-encoder compact \
  --pool 1000 --image-lag 1 --stride 24                  # one capped fold
```

For GPU training on Lightning AI, follow
[`TRAINING_RUNBOOK.md`](TRAINING_RUNBOOK.md) from the top.

## How it works

### Data

The loader prepares the data in five steps:

1. **Merge duplicate passes.** Identical force recordings are merged,
   keeping the most-worn annotation: 512 rows become 378 passes.
2. **Trim each pass.** The entry and exit transients are removed.
3. **Pool to 1 s steps.** `--pool 1000` averages 1000 samples into one step.
4. **Concatenate per tool.** Each tool's passes are joined into one timeline.
   Every step carries its pass's label and photo. With `--image-lag 1`, the
   photo shown is the *previous* pass's.
5. **Window.** Each window is 96 input steps followed by 96 forecast steps.

### Model

The model processes the two inputs in parallel, fuses them, and predicts:

- **Force branch:** a force MLP plus positional and period embeddings feeds a
  multi-resolution temporal extractor (4 RTD blocks).
- **Photo branch:** photos pass through EfficientNet-B0, then a 4-layer
  temporal graph attention network, then positional and period embeddings.
- **Fusion:** 2 rounds of C2F fusion. Each round does three things:
  - splits each stream into shared and private parts;
  - lets the shared parts attend to each other through adaptive landmarks;
  - gates the shared and private parts together.

  The second round re-injects the original streams.
- **Prediction:** per-lead heads read the last fused step to predict 96
  force steps and 96 wear classes.

### Evaluation

- **Folds:** 10-fold leave-one-tool-out cross-validation. Each tool is the
  test tool once; the next tool is used for validation; the other 8 train
  the model.
- **Input modes:** every fold is evaluated with both inputs, with the force
  input masked, and with the photos masked.
- **Metrics:** accuracy and macro/weighted precision, recall, F1 and ROC-AUC;
  MAE, MSE and RMSE in force units and z-scored; per channel and per forecast
  step; paper horizons 12…96 against persistence baselines.
- **Tracking:** everything is logged to `tracking/metrics.{jsonl,csv}`.

### Defaults

Library and CLI defaults follow the paper. The experiment protocol adds the
data choices that `EDA.md` justifies.

| Setting | `omnifuser train`, `train_lightning.py` | Protocol (runbook, `run_experiment_suite.py`) | Why |
|---|---|---|---|
| Time step `--pool` | 1 (1 ms) | 1000 (1 s) | Only 1 s steps match the paper's difficulty (`EDA.md` §6) |
| Photo `--image-lag` | 0 (same pass, paper) | 1 (previous pass) | With lag 0, the photo's label gives away 92-99 % of targets (`EDA.md` §7) |
| Horizon / input | 96 / 96 | 96 / 96 | Paper horizons go up to 96 |
| Training stride | input + horizon (192) | 24 | More training windows at 1 s steps |
| Duplicates `--dedup-identical-forces` | True (378 passes) | True | 134 records are exact copies, 24 groups with conflicting labels (`EDA.md` §3). Compare with `--study deduplication`, which scores both on the same deduplicated test data (`--eval-dedup-identical-forces`). |
| Epochs | 5 | 30 | The paper gives no epoch count |
| Optimizer | Adam, lr 1e-4, batch 32, dropout 0.1 | same | Paper values |
| Architecture | d=32, L=4, s_l=3, r=8, GAT 4×4, k=4, n=2 | same | Paper values |
| Head / ALS / private path | `direct` / `k_colnorm` / `plain` | same | Closest usable reading of Eqs. 21-22, 10, and Sec. V-C |

### Departures from the printed paper

Four defaults depart from the printed paper; each has a flag that restores
it:

| Default | Restore the printed version with | Reason |
|---|---|---|
| Fig. 5's trend and seasonal branches | `--decomp-mode literal` | Printed Eq. 3 cancels to the identity |
| Per-lead head | `--prediction-head paper_literal` | The printed head predicts one value for all 96 leads |
| Normalized landmark scale | `--als-mode k_sum` | Printed landmarks grow with input length |
| Previous-pass photo | `--image-lag 0` | The same-pass photo leaks the label |

The verdicts, the reasoning and the paper's own inconsistencies are in
`PAPER_IMPLEMENTATION_ALIGNMENT.md`.

## Development checks

```bash
python -m pytest -o addopts= -q tests    # 119 passed
ruff check .                             # all checks passed
pyright                                  # 0 errors (config in pyproject.toml)
```

## Status and limitations

- **Verified on CPU (2026-10-14):** every entry point, every ablation and
  every flag, on the real data with capped windows. The full verification
  record is in `IMPLEMENTATION_AUDIT.md` §10.
- **Not yet verified:** no GPU run has been done. CUDA execution, memory and
  throughput, and full-data accuracy must be measured with the runbook's GPU
  smoke and pilot steps.
- **Not implemented:** the paper's comparison models (Tables IV-V baselines),
  its image-update-frequency control experiment, and a FLOPs study beyond
  parameter counts.
- **Not a Lightning `Trainer`:** `train_lightning.py` is plain PyTorch. It
  runs in a Lightning AI Studio but does not use the Lightning `Trainer`.

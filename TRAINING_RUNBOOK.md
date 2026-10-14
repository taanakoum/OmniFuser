# OmniFuser training runbook

Every command needed to train and evaluate OmniFuser on Nonastreda, from a
local check to the full set of paper experiments on Lightning AI. Run the
steps in order. Each step writes to its own folder under `experiment_results/`.

The paper reports results on Mudestreda and MATWI. This repository trains on
Nonastreda, so the paper's numbers are reference points only. Do not expect to
reproduce them exactly.

## Overview

| Step | What it runs | 10-fold CV runs | Output folder |
|---|---|---|---|
| 0 | Tests, data inspection, GPU check | 0 | (none) |
| 1 | Capped one-fold smoke test | 0 (1 fold) | `experiment_results/validation/gpu-smoke/` |
| 2 | Uncapped one-fold pilot (timing) | 0 (1 fold) | `experiment_results/pilot/T10/` |
| 3 | Full model, all 10 folds | 1 | `experiment_results/main/full-model/` |
| 4 | Paper Fig. 13 component ablations | 9 | `experiment_results/ablations/<run-id>/` |
| 5 | Missing-modality robustness | 2 | `experiment_results/missing-modality/<run-id>/` |
| 6 | Paper Figs. 7-8 parameter sensitivity | 22 | `experiment_results/sensitivity/<run-id>/` |
| (alt) | Steps 4-6 in one unattended suite | 31 | `experiment_results/all/<run-id>/` |

Every 10-fold run evaluates the selected checkpoint three ways: with both
inputs (`full`), with the force signal masked (`sensor_missing`), and with the
images masked (`image_missing`). Each evaluation runs on the train, validation,
and test splits. You never need a separate command for missing-modality
evaluation; step 5 only adds training with modality dropout.

## 1. Lightning AI setup

Open a Studio with a CUDA GPU. Work inside the persistent Studio directory,
`/teamspace/studios/this_studio`, so the code, data, and results survive a
restart.

```bash
cd /teamspace/studios/this_studio
git clone <your-repository-url> OmniFuser
cd OmniFuser
python -m pip install -e ".[test]"
```

`Nonastreda_Data/` is not in Git. Upload it to the repository root using the
Studio file browser or any copy method. It must contain `forces_xyz_raw.mat`,
`labels.csv`, `labels_reg.csv`, and `tool/*.jpg`.

Define the shared protocol once per terminal session. Every command below uses
it, so all runs share the same preprocessing, windows, and optimizer settings:

```bash
PROTOCOL="--data-dir Nonastreda_Data --device cuda \
  --epochs 30 --batch-size 32 --learning-rate 1e-4 \
  --input-len 96 --horizon 96 --stride 24 \
  --pool 1000 --image-lag 1 --prediction-head direct \
  --image-encoder efficientnet_b0 --seed 42 --num-workers 4"
mkdir -p experiment_results/logs
```

What the protocol settings mean:

- **Paper settings:** Adam, learning rate `1e-4`, batch size 32, `d = 32`,
  dropout 0.1, and the default architecture (4 MRTE layers, stride 3, rank 8,
  4 GAT layers, 4 heads, 4 landmarks, 2 refinement rounds).
- **Horizon 96:** the paper horizons 12 to 96 are reported cumulatively from
  one 96-step forecast.
- **`--pool 1000 --image-lag 1`:** the Nonastreda alignment chosen earlier in
  this project. It mean-pools 1000 samples per time step and pairs each pass
  with the preceding pass's photo.
- **Epoch count:** the paper does not report one, so 30 is a project choice.
  Keep it fixed across every arm you compare.
- **Resuming:** a resume requires identical flags, including
  `--num-workers`. Change the protocol only when you start a new output folder.

## 2. Step 0: verify the environment

```bash
python -m pytest -o addopts= -q tests
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA')"
python -m omnifuser.cli inspect --data-dir Nonastreda_Data --pool 1000 --input-len 96 --horizon 96 --stride 24
python scripts/model_summary.py --horizon 96 --prediction-head direct --check
```

Before continuing, confirm three things:

- All tests pass.
- CUDA reports `True`.
- `inspect` reports 10 tools, 378 retained records, and no remaining duplicate
  forces.

## 3. Step 1: capped GPU smoke test

This checks the real data path, CUDA training, all three evaluation modes,
tracking, and artifact writing. It takes a few minutes. Its metrics are not
results.

```bash
python -m omnifuser.cli train $PROTOCOL --epochs 1 --test-tool T10 \
  --output-dir experiment_results/validation/gpu-smoke \
  --max-train-windows 64 --max-eval-windows 32
```

The folder should then contain:

- `fold-T10.json` and `fold-T10.pt`
- three prediction files: `fold-T10-predictions.npz`,
  `fold-T10-predictions-sensor_missing.npz`, and
  `fold-T10-predictions-image_missing.npz`
- three matching ROC files
- `tracking/metrics.csv`

## 4. Step 2: uncapped one-fold pilot

This trains one complete fold. Use it to measure cost before committing GPU
hours.

```bash
python -m omnifuser.cli train $PROTOCOL --epochs 3 --test-tool T10 \
  --output-dir experiment_results/pilot/T10 \
  2>&1 | tee experiment_results/logs/pilot-T10.log
```

Read `epoch_seconds` and `cuda_peak_allocated_bytes` for the `train` split in
`experiment_results/pilot/T10/tracking/metrics.csv`. One 10-fold run costs
roughly `epoch_seconds × 30 epochs × 10 folds`, plus about one epoch per fold
for the final evaluations. Multiply by the run counts in the overview to
estimate a study.

If the GPU runs out of memory, halve `--batch-size` and add
`--accumulation-steps 2`. That keeps the effective batch at 32.

## 5. Step 3: full model, 10-fold cross-validation

Each fold trains on 8 tools, selects the best epoch on the next tool by
validation loss, and tests on the held-out tool. The force normalizer is
fitted on the 8 training tools only.

```bash
python scripts/train_lightning.py $PROTOCOL \
  --output-dir experiment_results/main/full-model \
  2>&1 | tee experiment_results/logs/main-full-model.log
python scripts/compare_with_paper.py experiment_results/main/full-model \
  --json experiment_results/main/full-model/paper-comparison.json
```

`compare_with_paper.py` prints the z-scored cumulative MSE/MAE and the
classification scores for each paper horizon. It shows them next to the
last-value and persistence baselines and the published Tables IV-V. Add
`--average weighted` for support-weighted precision/recall/F1.

## 6. Step 4: paper component ablations (Fig. 13)

```bash
python scripts/run_experiment_suite.py --study ablations --run-id paper-v1 \
  --output-root experiment_results $PROTOCOL \
  2>&1 | tee experiment_results/logs/ablations-paper-v1.log
```

This trains 9 arms, each over all 10 folds, with the same seed and protocol:

| Paper variant | `--ablation` value | Implementation |
|---|---|---|
| Full model | `none` | Reference arm |
| w/o-RR | `without_rr` | One C2F pass, no recursive refinement |
| w/o-MRTE | `without_mrte` | MRTE replaced by identity after the force embedding |
| w/o-RR&MRTE | `without_rr_mrte` | Both of the above |
| re-C2F-Concat | `concat_c2f` | Concatenate both streams and project, at each refinement round |
| w/o-C2F | `without_c2f` | Equal-weight mean of the two streams feeds the heads |
| w/o-ALS | `without_als` | Fixed segment-mean landmarks (non-adaptive) |
| w/o-HG | `without_hybrid_gate` | Fixed 0.5 mixture of cross-modal and private features |
| re-LR-MA | `moving_average` | Three-token moving average replaces the low-rank trend |

The paper does not specify the replacement operator for w/o-C2F, w/o-ALS,
w/o-HG, or re-LR-MA. The choices above are this implementation's and are
recorded in `suite.json`. Fig. 13 reports MSE, so compare the
`full.regression_mse_normalized.mean` column of `suite-summary.csv`. That
column is the z-scored MSE averaged over all 96 leads.

To run a single ablation arm on its own:

```bash
python scripts/train_lightning.py $PROTOCOL --ablation without_als \
  --output-dir experiment_results/ablations/single/without-als
```

## 7. Step 5: missing-modality robustness

The paper describes modality dropout during training and masking at
inference. The study trains a reference arm without dropout and an arm with
training modality dropout 0.2. The paper reports no rate, so 0.2 is a choice.
Both arms are then evaluated with the sensor masked and with the image masked.

```bash
python scripts/run_experiment_suite.py --study missing-modality --run-id paper-v1 \
  --output-root experiment_results $PROTOCOL \
  2>&1 | tee experiment_results/logs/missing-modality-paper-v1.log
```

To sweep more dropout rates, use a new `--run-id`:

```bash
python scripts/run_experiment_suite.py --study missing-modality --run-id rates-v1 \
  --modality-dropout-rates 0.1 0.2 0.3 \
  --output-root experiment_results $PROTOCOL
```

To read the robustness results, compare `full.*`, `sensor_missing.*`, and
`image_missing.*` within each row of `suite-summary.csv`. Graceful degradation
means the masked-mode scores fall toward, but not below, the persistence
baselines.

## 8. Step 6: parameter sensitivity (Figs. 7-8)

```bash
python scripts/run_experiment_suite.py --study sensitivity --run-id paper-v1 \
  --output-root experiment_results $PROTOCOL \
  2>&1 | tee experiment_results/logs/sensitivity-paper-v1.log
```

| Paper axis | Config field | Values swept |
|---|---|---|
| k (landmarks) | `num_landmarks` | 2, 4, 8, 16 |
| d (embedding width) | `d_model` | 16, 32, 64, 128 |
| GAT heads | `num_gat_heads` | 2, 4, 8, 16 |
| r (trend-seasonal projection rank) | `rtd_rank` | 4, 8, 16, 32 |
| GAT layers | `num_gat_layers` | 2, 3, 4 |
| s_l (downsampling stride) | `rtd_stride` | 1, 2, 3 |
| L (MRTE layers) | `num_rtd_layers` | 1, 2, 3, 4 |
| n (recursive refinement iterations) | `num_fusion_layers` | 1, 2, 3 |

Values equal to the paper default are not retrained. They are the
`sensitivity-baseline` row, giving 22 runs in total. The GAT-heads setting
also sets the head count of the C2F attention modules.

## 9. Alternative: all paper studies in one command

This runs steps 4 to 6 in one suite. The ablation full model is shared as the
reference for the sensitivity and dropout arms, so the suite trains 31 runs
instead of 33:

```bash
nohup python scripts/run_experiment_suite.py --study all --run-id paper-v1 \
  --output-root experiment_results $PROTOCOL \
  > experiment_results/logs/all-paper-v1.log 2>&1 &
tail -f experiment_results/logs/all-paper-v1.log
```

## 10. Optional: prediction-head comparison

This is not a paper experiment. It compares the paper-literal repeated head
with the two lead-varying heads. `$PROTOCOL` already sets `--prediction-head`,
and argparse uses the last value given, so the later flag wins:

```bash
for head in paper_literal lead_conditioned direct; do
  python scripts/train_lightning.py $PROTOCOL --prediction-head $head \
    --output-dir experiment_results/heads/$head \
    2>&1 | tee experiment_results/logs/head-$head.log
done
```

## 11. Running unattended and resuming

- **Keep runs alive.** Start long runs with `nohup ... &` (section 9) or inside
  `tmux` so they survive a closed browser tab. Check the Studio's auto-sleep
  setting before a multi-hour run.
- **Resume after an interruption.** Rerun the exact same command. Suites need
  the same `--run-id`.
  - Finished variants in `suite.json` are skipped.
  - Finished folds with matching settings are reused.
  - An interrupted fold continues from its last completed epoch
    (`fold-Tn-resume.pt`).
- **Mismatched settings are refused.** Changed settings stop with an error
  instead of mixing configurations in one folder. A different GPU type or
  torch/CUDA version is allowed; the change is recorded under
  `runtime_history` in `tracking/run.json`.
- **Data is checked.** Each source file's path, size, and modification time
  are recorded. Do not edit or re-upload `Nonastreda_Data/` while a study is
  in progress.
- **Back up results.** Copy `experiment_results/` off the Studio when a study
  finishes. It is git-ignored, and checkpoints can be large.

## 12. Results layout and tracked metrics

```text
experiment_results/
  logs/                                   console logs from tee/nohup
  validation/, pilot/                     smoke and timing runs (not results)
  main/full-model/                        one 10-fold run (layout below)
  <study>/<run-id>/
    suite.json                            settings, per-variant status, notes
    suite-summary.csv                     one row per variant, test mean/std per input mode
    <group>/<variant>/                    one 10-fold run (layout below)

10-fold run directory:
  tracking/run.json                       configuration, data signature, runtime(s)
  tracking/metrics.jsonl                  every event: setup, per-epoch train/validation, final evaluations
  tracking/metrics.csv                    the same numbers in long form (fold, epoch, split, evaluation_mode, metric, value)
  fold-Tn.json                            fold settings, epoch history, test and missing-modality metrics
  fold-Tn.pt                              selected checkpoint with config and normalizer
  fold-Tn-predictions[-<mode>].npz        test forecasts, targets, class probabilities (original units)
  fold-Tn-roc[-<mode>].npz                one-vs-rest ROC coordinates per class
  cross-validation-results.json           fold mean/std, paper horizons, per-mode summaries, pooled results
  cross-validation-mean-roc[-<mode>].npz  fold-averaged ROC curves
  cross-validation-progress.json          running summary while folds complete
```

Metrics are recorded for every training epoch (train and validation splits)
and for the selected checkpoint (train, validation, and test splits, in each
of the three input modes).

**Classification metrics:**

- accuracy
- macro and support-weighted precision, recall, and F1
- per-class precision, recall, F1, and support
- confusion matrix
- one-vs-rest ROC AUC per class and macro
- all of the above per forecast lead (test split)

**Regression metrics:**

- MAE, MSE, and RMSE in original force units and in z-scored units
- each overall, per channel (Fx, Fy, Fz), and per forecast lead

**Test-only extras:**

- cumulative paper-horizon metrics (12, 24, ..., 96)
- last-value, zero-forecast, and persistence-label baselines

**Training diagnostics:**

- loss components: total, force, and class, plus the logged alignment and
  orthogonality diagnostics
- epoch time
- peak CUDA memory

In `cross-validation-results.json`, use these keys:

| Key | Contents |
|---|---|
| `cv_mean_std` | Fold mean/std of the headline test metrics |
| `paper_horizons_mean_std` | Fold mean/std at each paper horizon |
| `modality_evaluations` | Fold mean/std per input mode and split |
| `out_of_fold` and `out_of_fold_by_mode` | Metrics pooled over all held-out windows |

For dashboards, load `tracking/metrics.csv` and filter by `split`,
`evaluation_mode`, and `metric`. Rows with an empty `evaluation_mode` are
per-epoch training curves.

## 13. Local commands (Windows PowerShell, CPU)

Use these to check the pipeline on a laptop before uploading. They are not
experiments.

```powershell
.\.venv\Scripts\python.exe -m pytest -o addopts= -q tests

.\.venv\Scripts\python.exe -m omnifuser.cli train `
  --data-dir Nonastreda_Data --test-tool T10 `
  --output-dir experiment_results\validation\local-smoke `
  --epochs 1 --batch-size 2 --image-encoder compact --device cpu `
  --pool 1000 --image-lag 1 --horizon 96 --stride 24 --prediction-head direct `
  --max-train-windows 4 --max-eval-windows 2

.\.venv\Scripts\python.exe scripts\run_experiment_suite.py `
  --study missing-modality --run-id local-smoke `
  --output-root experiment_results\validation `
  --epochs 1 --batch-size 2 --image-encoder compact --device cpu `
  --pool 1000 --image-lag 1 --horizon 96 --stride 24 `
  --max-train-windows 4 --max-eval-windows 2
```

The capped suite smoke run takes about 3.5 minutes on CPU. Keep
`--num-workers 0` on Windows; worker processes there re-pickle the image cache
and slow loading down.

## 14. Paper experiments not covered

- **Comparisons with other models** (time-series models, CDA/MBT/LMF/TFN fused
  through the C2F slot, foundation models): no baselines are implemented. The
  persistence and last-value baselines are reported instead.
- **Halved image-update-frequency control:** not implemented.
- **FLOPs and memory study (Fig. 14):** only parameter counts
  (`scripts/model_summary.py`) and per-epoch peak CUDA memory are recorded.
- **Paired t-tests:** per-fold values are stored in `per_fold` lists, so they
  can be computed afterwards.
- **Loss weighting:** the paper gives no loss weighting. Training minimizes
  force MSE plus class cross-entropy at 1:1. Alignment and orthogonality are
  logged, not optimized (see `PAPER_IMPLEMENTATION_ALIGNMENT.md`).

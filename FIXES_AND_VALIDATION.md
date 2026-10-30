# Classification fixes: actions and validation

This document records the fixes started after the final results report
([`FINAL_REPORT_REPRODUCTION.md`](FINAL_REPORT_REPRODUCTION.md) §5): what was
implemented, how each choice was validated, the results, and what remains.
Started on 2026-10-30.

- **Round 1** (§3-§5): post-hoc decision rules, validated on the saved
  predictions of all 51 runs.
- **Round 2** (§6-§9):
  - the training-side fixes, implemented and unit-tested;
  - local checks of the premise behind each one;
  - a two-stage Ruche plan to train them, in
    [`TRAINING_RUNBOOK_RUCHE_FIXES.md`](TRAINING_RUNBOOK_RUCHE_FIXES.md).

## Contents

1. [Summary](#1-summary)
2. [Constraints found before starting](#2-constraints-found-before-starting)
3. [Fix A1: decision rules that use the order of wear](#3-fix-a1-decision-rules-that-use-the-order-of-wear)
4. [Fix A2: class offsets tuned on the validation tool](#4-fix-a2-class-offsets-tuned-on-the-validation-tool)
5. [Fix A3: modality dropout](#5-fix-a3-modality-dropout)
6. [Run-to-run noise and ensembles (A4)](#6-run-to-run-noise-and-ensembles-a4)
7. [Training-side fixes: what was implemented](#7-training-side-fixes-what-was-implemented)
8. [Local checks of the premises](#8-local-checks-of-the-premises)
9. [The Ruche plan](#9-the-ruche-plan)
10. [Status of all fixes](#10-status-of-all-fixes)
11. [Files and tests](#11-files-and-tests)
12. [How to reproduce](#12-how-to-reproduce)

## 1. Summary

| Fix | Status | Validated on | Main finding |
|---|---|---|---|
| A1 Ordinal and monotone decision rules | **Implemented and validated** | Saved test predictions of all 51 runs; 7 replicates of the main configuration | "Used" recall +0.22, macro precision +0.03, macro F1 +0.025, class flicker 41 % → 11 % of windows; accuracy +0.01 (6 of 7 replicates higher) |
| A2 Class offsets tuned on the validation tool | Implemented and unit-tested; **runs on Ruche**, where the model files are | Synthetic data | To be measured (runbook §7.1) |
| A3 Modality dropout 0.2 | **Revised**: the accuracy gain is within run-to-run noise; the robustness gain is real | 7 replicates against the dropout run | Accuracy +0.007 over the replicates' mean (about 1 SD); photo-masked accuracy 0.41 → 0.60 |
| A4 Seeds and ensembles | Ensemble script; replicate noise measured | 7 replicates (same settings and seed) | Single runs: 0.659-0.676 (SD 0.006). Their probability average: +0.006 accuracy, better on all 8 folds that changed |
| B1 Photo crop and size | Implemented (`--image-crop`, `--image-size`) | Linear probe of frozen features, leave-one-tool-out | More pixels do not help frozen features (0.636 → 0.608); a crop of the cutting edge helps a little (0.652) |
| B2 Run-number shortcut | Implemented (`--metadata {all,position,none}`), with a run-number baseline in every fold | The main run's test windows | The run number alone scores **0.708**, more than the model (0.673) |
| B3 Tool-shift normalization | Implemented (`--revin`) | The main run's forecasts | 69 % of forecast MSE is a level offset per window; the input-level anchor cuts T1's level error by 79 % but is worse on T4, T5 and T10 |
| B4 Ordinal training loss | Implemented (`--class-loss coral`) | Unit tests | To be measured; it must beat the post-hoc rule A1 |
| Report what matters (Step 8) | Implemented: accuracy where the label differs from the photo's | The main run | The model gets 0.43 of those steps right, 0.70 of the others |
| MSE-horizon check | Implemented: `omnifuser predict --evaluation-stride`, `scripts/phase_study.py` | - | To be measured on the saved models (runbook §7.2) |
| C1 Chip and workpiece photos | Blocked: the folders are not in the data | - | §10 |
| C2 Spectrogram as a third stream | Not started: needs a new fusion design | - | §10 |

**Recommendation.**

1. Report every classification result with argmax (the paper's rule) and
   the ordinal + monotone rule, with "used" recall, macro F1 and the
   accuracy where the label differs from the photo's.
2. Do not read differences below about 0.015 between single runs as
   effects (§6). Compare configurations over several seeds, paired by fold.
3. Run the two-stage plan of `TRAINING_RUNBOOK_RUCHE_FIXES.md`. Its decision
   rules were fixed before any of its runs (§9).

## 2. Constraints found before starting

1. **The trained models are not on the laptop.** Only
   `results-no-checkpoints.tar.gz` was downloaded; the `fold-Tn.pt` model
   files are in `results-checkpoints.tar.gz`, still on Ruche. Anything that
   needs predictions on the validation tools must wait for it. That includes
   threshold or offset tuning (A2).
2. **The chip and workpiece photos are not in the local data.**
   `Nonastreda_Data/` contains `forces_xyz_raw.mat`, `labels.csv`,
   `labels_reg.csv` and the `tool/` folder (512 photos) only. The other two
   photo folders of Nonastreda must be downloaded from the dataset's source
   before fix C1 can start.
3. **The laptop was busy.** OneDrive was syncing the 20 GB of results, the
   CPU was at 76-90 % and 2.2 GB of RAM were free. The round-1 analyses
   therefore use only the saved prediction files (NumPy, no model or raw data
   loading) and run in one process with 2 threads. In round 2 another
   training job was running on the laptop. The photo probe (§8.1) ran at
   idle priority with 3 threads, and the data-loading checks with 1 thread.

The fixes that can be validated now are those that act on saved
predictions: the decision rules (A1). The existing modality-dropout study
also provides evidence for A3.

## 3. Fix A1: decision rules that use the order of wear

### 3.1 The problem it targets

From the final report (§4.1):

- 88 % of errors are between neighbouring classes;
- "used" recall is only 0.49;
- the predicted class changes within a 96-step forecast in 40 % of windows,
  while the true class changes in 21 %;
- the class ranking is good (ROC AUC 0.87), but argmax turns it into
  mediocre decisions.

Argmax ignores two facts:

1. **The classes are ordered** (sharp < used < dulled). A lead whose
   probability is split between sharp and dulled is most likely "used", the
   class in between, yet argmax picks an end.
2. **Wear does not decrease.** Within one forecast window the true class
   should be non-decreasing, but argmax decides every lead independently and
   can flicker.

### 3.2 The rules

Implemented in [`omnifuser/decision.py`](omnifuser/decision.py). None of
them has a fitted parameter.

| Rule | Function | Decision |
|---|---|---|
| argmax (reference) | `argmax_decode` | The most probable class per lead |
| monotone | `monotone_decode` | The most probable **non-decreasing** class sequence over the 96 leads (Viterbi over the 3 classes, maximizing $\sum_t \log p_t(c_t)$ subject to $c_t \ge c_{t-1}$) |
| ordinal | `ordinal_median` | The median of each lead's ordinal distribution: class $k$ such that $P(c \ge k) > 0.5 \ge P(c \ge k+1)$ |
| ordinal + monotone | `monotone_ordinal_decode` | The non-decreasing sequence with the smallest expected class distance, $\sum_t \sum_k p_t(k)\,\lvert k - c_t\rvert$ |

The last rule combines the other two in a principled way. Its per-lead
minimizer is the ordinal median, and the sequence is constrained like
"monotone".

A first attempt combined them naively, with a running maximum of the
ordinal median. It lowered accuracy in 35 of 51 runs, because it locked in
any early error that was too high, and it was replaced. §3.5 discusses what
this means for validation.

### 3.3 How the choices were validated

- **No leakage.** The rules have no parameters, so nothing is fitted on
  test data. They were applied to the test predictions every fold saved
  (`fold-Tn-predictions.npz`), with the same targets and folds as the
  reported results.
- **Across all runs.** All 51 finished runs were scored: main runs,
  ablations, image studies, heads, paper-literal arms, deduplication,
  missing modality and sensitivity. A rule that only helps one
  configuration would show up as inconsistent.
- **Statistics.**
  - Within a run: paired Wilcoxon signed-rank test over the 10 folds against
    argmax.
  - Across runs: the share of runs improved, and a Wilcoxon test on the
    per-run differences.
  - The runs share data and folds, so they are not independent. The
    across-run p-value overstates the evidence; the share of runs improved
    is the more honest summary.
- **The premise was checked against the data.**
  - In the main run's test targets, 76 of 369 windows change class within
    the horizon.
  - 29 of the 77 class changes are *decreases*, in 29 windows (7.9 %). This
    is label noise (`FUTURE_WORK_AND_ANALYSIS.md` F12: flank wear decreases
    at 189 of 502 steps).
  - The monotone premise is therefore true for about 92 % of windows. The
    rule cannot be right on the other 8 % and is expected to lose on some
    folds.
- **Metrics.**
  - accuracy (all 96 leads, and averaged over the paper's 8 horizons);
  - macro precision and macro F1;
  - recall of "used";
  - mean absolute class error (MACE);
  - the share of windows whose decided class changes within the forecast.

### 3.4 Results

**Across all 51 runs** (change against argmax; mean over runs):

| Rule | Runs with higher accuracy | Accuracy | Macro precision | Macro F1 | "Used" recall | MACE | Changing windows |
|---|---|---|---|---|---|---|---|
| argmax | - | - | - | - | - | - | 41 % |
| monotone | 38 (11 lower) | +0.005 | +0.010 | +0.004 | +0.020 | −0.006 | 16 % |
| ordinal | 39 (12 lower) | +0.005 | +0.026 | +0.021 | +0.207 | −0.024 | 35 % |
| **ordinal + monotone** | **42 (9 lower)** | **+0.010** | **+0.031** | **+0.025** | **+0.218** | **−0.029** | **11 %** |

Across-run Wilcoxon p for accuracy: 3.6e-5 (monotone), 0.0016 (ordinal) and
3.4e-5 (ordinal + monotone). As explained in §3.3, these overstate the
evidence.

**Key runs** (fold means; "folds +/−" counts folds where accuracy improves
or worsens against argmax, with the Wilcoxon p over the 10 folds):

| Run | Rule | Accuracy | Macro precision | Macro F1 | "Used" recall | MACE | Changing windows | Folds +/− (p) |
|---|---|---|---|---|---|---|---|---|
| Main 3a | argmax | 0.673 | 0.696 | 0.646 | 0.481 | 0.362 | 40 % | |
| | ordinal + monotone | 0.672 | **0.732** | **0.660** | **0.712** | 0.350 | 12 % | 5/5 (0.77) |
| Main 3b | argmax | 0.667 | 0.710 | 0.636 | 0.458 | 0.372 | 33 % | |
| | ordinal + monotone | **0.683** | **0.737** | **0.669** | **0.672** | 0.341 | 8 % | 8/2 (0.32) |
| Frozen, no image GAT | argmax | 0.693 | 0.701 | 0.662 | 0.469 | 0.329 | 31 % | |
| | monotone | **0.706** | 0.723 | 0.676 | 0.503 | 0.317 | 17 % | 7/3 (0.05) |
| | ordinal + monotone | 0.703 | **0.731** | **0.688** | **0.604** | **0.309** | 14 % | 6/4 (0.62) |
| Modality dropout 0.2 | argmax | 0.677 | 0.699 | 0.642 | 0.417 | 0.349 | 38 % | |
| | ordinal + monotone | **0.696** | **0.742** | **0.684** | **0.651** | 0.319 | 7 % | 8/2 (0.11) |
| Spectrogram (no photo) | argmax | 0.656 | 0.660 | 0.595 | 0.256 | 0.383 | 35 % | |
| | ordinal + monotone | **0.674** | **0.695** | **0.648** | **0.496** | 0.349 | 10 % | 8/2 (0.28) |

### 3.5 Interpretation and caveats

- **What is solid.**
  - "Used" recall rises by about 0.2 (0.48 → 0.71 in the main run).
  - Macro precision and F1 rise.
  - Class flicker falls by three quarters.
  These effects appear in almost every run and are large compared with the
  noise between folds. They follow from the ordinal and monotone premises.
- **What is small.** Accuracy changes by +0.01 on average. In the main run
  it is unchanged (0.673 → 0.672, 5 folds up and 5 down). No single run
  shows a significant accuracy gain at the 5 % level except "frozen, no
  image GAT" with the monotone rule (p = 0.05, borderline).
- **Where it fails.** In the main run, fold T3 falls from 0.626 to 0.425.
  When the model predicts a too-high class early in a window, the
  non-decreasing constraint carries that error forward. The 8 % of windows
  whose true labels decrease (noise) also count against it.
- **Selection.** The combined rule replaced a naive version after the naive
  version failed on these same test predictions. Choosing between two
  parameter-free rules is a small degree of freedom, but it is one. The rule
  should be confirmed on predictions it was not chosen on: the validation
  tools (A2) or new runs.
- **Interpretation.** The probabilities already contain the information to
  find "used". Argmax discards it, because "used" is rarely the single most
  probable class even when the mass sits around it. The fix improves
  decisions, not the model.

### 3.6 Decision

Report argmax (the paper's rule) **and** "ordinal + monotone" for every
classification result, with "used" recall and macro F1, until the rule is
confirmed on validation predictions. Do not report only the better of the
two. Picking a run after seeing test scores (for example "frozen, no GAT,
monotone: 0.706") overstates what the method achieves on new data.

### 3.7 Confirmation on replicates

Seven finished runs share the main configuration exactly (§6). On these
replicates the ordinal + monotone rule raises accuracy in 6 of 7 (mean
0.671 → 0.679). Only the main run 3a itself goes down (0.673 → 0.672). The
rule's accuracy gain is therefore small but consistent.

After the seven runs' probabilities are averaged (§6), the rule no longer
helps: 0.676 with either rule. Averaging already smooths the per-lead
probabilities, which removes much of the flicker the rule corrects.

## 4. Fix A2: class offsets tuned on the validation tool

**Why.** Some folds may systematically under-predict a class (for example
"used"). Adding a per-class offset to the log-probabilities moves the
decision boundaries. With a good ranking (AUC 0.87), this can raise accuracy
and recall without retraining.

**Implementation.**

- `fit_class_offsets` in `omnifuser/decision.py` searches offsets for "used"
  and "dulled" (sharp fixed at 0) in steps of 0.1 within ±1.5, to maximize
  macro F1 on **validation** predictions.
- Ties keep the smallest offsets, so with no gain the offsets stay 0 and
  decisions are unchanged.
- `scripts/decision_study.py --tune-offsets` produces each fold's validation
  predictions from its saved model (`fold-Tn.pt`, through
  `model_bundle.load_model` and the same input functions as training). It
  caches them in `<run>/decision/` and fits the offsets per fold, for argmax
  and for the monotone rule. It then applies them to that fold's test
  predictions.
- The validation tool of fold *i* is T*i+1*. It is never that fold's test
  tool, so no test information enters the fit.
- Offsets are fitted per fold, never pooled across folds: another fold's
  validation tool can be this fold's test tool.

**Validation so far.**

- Unit tests: offsets recover a deliberately biased "used" boundary to 100 %
  accuracy, and stay at zero on perfect predictions.
- Not yet measured on real runs, because the model files are not on the
  laptop (§2). They are still on Ruche. The runbook therefore runs it there
  as a CPU job, with no download of the 8 GB checkpoint archive
  (`TRAINING_RUNBOOK_RUCHE_FIXES.md` §7.1).

**Caveat for the result.** One validation tool has 28-53 windows, so fitted
offsets may be noisy. If they do not help on the test tools, prefer the
parameter-free rules of §3.

## 5. Fix A3: modality dropout

**Evidence.** The missing-modality study trained the same configuration
with and without modality dropout 0.2 (a stream is zeroed at random during
training):

| Decision rule | No dropout | Dropout 0.2 | Folds better / worse | Wilcoxon p |
|---|---|---|---|---|
| argmax | 0.659 | 0.677 | 7 / 3 | 0.23 |
| ordinal + monotone | 0.679 | 0.696 | 6 / 4 | 0.19 |

Dropout also lifts photo-masked accuracy from 0.41 to 0.60, so the force
stream learns to carry class information instead of leaving it to the
photo.

**Assessment.**

- The gain (+0.018) is consistent with both rules but not significant with
  10 folds.
- It costs nothing and makes the model robust to a missing photo.
- Adopt it for future runs (`--modality-dropout 0.2`, already supported).
  Confirm with several seeds (F14).

**Revision (round 2).**

- The no-dropout arm of this study (0.659) is the lowest of the seven
  replicates of the main configuration (0.659-0.676, mean 0.671; §6).
- Against their mean, dropout gains +0.007, about one SD of single runs.
  The accuracy gain is therefore **not established**.
- The robustness gain (photo-masked accuracy 0.41 → 0.60) is far larger
  than that noise and stands.
- Dropout is therefore one arm of the Ruche plan (§9), not part of the
  reference.

## 6. Run-to-run noise and ensembles (A4)

### 6.1 Seven replicates already exist

Seven finished runs have exactly the main configuration: identical model
settings, data settings, training settings and seed 42. They were checked by
comparing the `config`, `training` and data fields of their fold files. They
differ only by GPU non-determinism:

| Run | Accuracy | Accuracy, ordinal + monotone | Force MSE (z) |
|---|---|---|---|
| `main/full-model` (3a) | 0.673 | 0.672 | 0.809 |
| `ablations/.../full-model` | 0.675 | 0.683 | 0.804 |
| `heads/direct` | 0.675 | 0.686 | 0.802 |
| `image-encoder/.../pretrained-frozen` | 0.676 | 0.682 | 0.805 |
| `missing-modality/.../no-modality-dropout` | 0.659 | 0.679 | 0.867 |
| `sensitivity/.../sensitivity-baseline` | 0.668 | 0.672 | 0.802 |
| `deduplication/.../dedup-true` | 0.668 | 0.678 | 0.797 |
| **Mean ± SD** | **0.671 ± 0.006** | **0.679 ± 0.006** | **0.812 ± 0.025** |

**What it means.**

- Two single runs of the same configuration can differ by 0.017 in
  accuracy. Different seeds vary more, because they also change the
  initialization and the order of the batches.
- Several earlier comparisons sit inside this range, and should not be read
  as effects:
  - modality dropout (§5);
  - most sensitivity points;
  - the head variants.
- The two best single results of the image-encoder study (fine-tuned 0.693,
  frozen without image GAT 0.693) are about 3.5 SDs above the mean. That is
  promising, but one seed each, so they are repeated in the Ruche plan.

### 6.2 Averaging the replicates

`scripts/ensemble_runs.py` averages the class probabilities and force
forecasts of runs with identical test windows, fold by fold, and scores the
average like a run:

| Metric | Single runs (mean) | Average of the 7 | Folds better / worse | Wilcoxon p |
|---|---|---|---|---|
| Accuracy | 0.671 | **0.676** | 8 / 0 | 0.008 |
| Macro F1 | 0.642 | **0.648** | 8 / 0 | 0.008 |
| Macro precision | 0.691 | **0.696** | 8 / 0 | 0.008 |
| "Used" recall | 0.465 | **0.476** | 7 / 0 | 0.004 |
| Force MSE (z) | 0.812 | **0.806** | 9 lower / 1 higher | 0.004 |

The gain is small, because runs with the same seed are very similar. It is
consistent on every metric and every fold that changed. Seeds that differ
should give more diverse models and a larger gain. The Ruche plan therefore
reports the seed ensemble of every arm.

## 7. Training-side fixes: what was implemented

Every option defaults to the main runs' behaviour. The existing results,
saved models and fold reuse stay valid; the parameter count of the protocol
is unchanged at 4,183,600. `END_TO_END_SIM.md` documents each option where it
acts.

| Fix | Option | Where | What it does | Design choice |
|---|---|---|---|---|
| B2 Shortcut | `--metadata {all,position,none}` | `config.py`, `models._metadata_tables`, `TemporalEmbedding`, `ImageBranch` | `position` removes the run and block codes, keeping the position within the pass; `none` removes all three. The tables are not created. | `position` keeps the pass phase, which is not a shortcut and may help the forecast around pass transients; `none` tests whether any code helps |
| B2 Reference | run-number baseline | `evaluation.run_number_baseline_accuracy`, `train_fold` | Each test fold records the accuracy of predicting the class from the run number alone (`test.baselines.run_number_class_accuracy`) | The rule is fitted on the fold's training tools only |
| B3 Tool shift | `--revin` | `OmniFuser.forward` | Each input window is normalized by its own per-channel mean and spread (plus a learnable affine map), and the forecast is mapped back | The window statistics are not fed back to the model, because that would bring back the tool's level |
| B4 Ordinal loss | `--class-loss coral` | `OmniFuser` (direct head), `ordinal_log_probabilities`, `multitask_loss` | One score per lead and two thresholds per lead, kept in order (a first threshold plus a positive gap). Binary cross-entropy on "above sharp" and "above used". The head returns log-probabilities, so every metric and decision rule works unchanged. | Ordered thresholds give valid probabilities by construction. Direct head only, the protocol's. |
| B1 Photo | `--image-crop T L B R`, `--image-size H W` | `data.load_photo`, `ImageBank`, `ModelConfig` | Crops each photo to fractions of its height and width before resizing | In `ModelConfig`, so saved models and `omnifuser predict` reproduce the crop |
| Step 8 | photo-change accuracy | `evaluation.photo_change_report`, `_run_epoch` | Accuracy on target steps whose label differs from the shown photo's, and on the others (`test.photo_change`) | Measures forecasting a change rather than reading the photo |
| MSE-horizon check | `omnifuser predict --evaluation-stride`, `scripts/phase_study.py` | `model_bundle.predict` | Re-scores a saved model with windows every *n* steps | Evaluation only: training runs keep their test sets, so they stay comparable |
| Plan | `--study fixes` | `scripts/run_experiment_suite.py` | The 9 single-change arms of stage 1 | §9 |
| Plan | `scripts/fixes_analysis.py` | | Seed-averaged paired comparison and the decision rules of §9 | Rules fixed in the code before the runs |

A config file written before these options existed loads with their
defaults, so old folds are still reused. A crop stored as a JSON list loads
as the tuple the config declares (tested).

## 8. Local checks of the premises

Each training-side fix rests on a premise about the data or the finished
runs. These checks need no training. They decide how much each fix can
deliver, and they set its arm in the Ruche plan.

### 8.1 Photo evidence: resolution and crop (B1)

**Premise.** Resizing the photos to 384 × 128 removes the thin wear band at
the cutting edge (`FINAL_REPORT_REPRODUCTION.md` §4.3).

**Method** (`scripts/photo_probe.py`).

- All 512 photos go through the frozen ImageNet EfficientNet-B0 at 5 input
  settings. The last feature map is average-pooled, as in the model, and
  also max-pooled.
- A multinomial logistic regression on the pooled features predicts each
  photo's wear class. It is trained on 9 tools and tested on the 10th, for
  each tool.
- The L2 penalty 0.01 was fixed beforehand. The other two values only show
  how robust the result is.
- The probe isolates how much class information the frozen features keep,
  without OmniFuser.

**Where the cutting edge is.** The row with the strongest change of mean
brightness lies between 0.51 and 0.68 of the photo height in all 512 photos
(1 % and 99 % quantiles; median 0.64). The crop band, rows 0.30-0.85 at the
full width, contains it in every photo, with margin. The bottom of each
photo is blurred background.

**Results** (leave-one-tool-out accuracy, mean over the 10 tools; average
pooling):

| Setting | Penalty 0.001 | **0.01** | 0.1 | "Used" recall (0.01) |
|---|---|---|---|---|
| Whole photo, 128 × 384 (main runs) | 0.632 | 0.636 | 0.644 | 0.42 |
| Whole photo, 256 × 768 | 0.638 | 0.642 | 0.640 | 0.47 |
| Whole photo, 512 × 1536 (near native) | 0.611 | 0.608 | 0.628 | 0.47 |
| **Crop, 128 × 640** | **0.647** | **0.652** | **0.670** | 0.44 |
| Crop, 256 × 1280 (near native) | 0.628 | 0.635 | 0.663 | 0.40 |

Max pooling is no better (0.606-0.642 at penalty 0.01).

**Interpretation.**

- **The premise is not confirmed for frozen features.** Near-native
  resolution is *worse* (0.608).
  - ImageNet filters respond to structures of a certain size in pixels.
    At 4× the pixels, the wear band and the grinding texture look different
    to them.
  - The global average over a larger map dilutes a thin band further.
- **Removing the background helps a little.** The crop is the best setting
  at all three penalties, +0.016 to +0.026 over the main runs' input.
- **The frozen features are the bottleneck.** All settings read the wear
  class from a photo at 0.61-0.67. That matches the model's own accuracy
  with the force masked (0.653) and is far below what the photo's label
  allows (0.95). The larger lever is training the encoder: fine-tuning, the
  best single-seed result (0.693; §6.1). A wear-specific model would be
  another.

**Decision.**

- The stage-1 photo arm uses the crop at 128 × 640, with the frozen
  encoder. It is also cheaper than the near-native settings.
- Choosing 1 of 5 settings on this probe is a small selection, so stage 1 is
  the real test.
- Fine-tuning on the crop is tested in stage 2, if both the crop and
  fine-tuning pass stage 1.

### 8.2 Forecast error: how much is a level offset (B3)

**Method** (`scripts/window_diagnostics.py`).

1. Each fold's z-scored forecast error is split into a constant offset per
   window and channel, plus the variation around it.
2. With the data, the model's window level is compared with the level RevIN
   anchors to: the input window's mean. Both are compared against the target
   window's mean.

**Results** (main run 3a):

- **69 % of the forecast MSE is a constant level offset per window** (52-76 %
  per fold; 68 % in the spectrogram run). Getting the level right is most
  of the forecasting problem.
- The input-level anchor against the model's level (squared level error, z
  units):

| Fold | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 | T10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Model's level | **0.89** | 0.30 | 0.35 | 0.49 | 1.81 | 0.47 | 0.20 | 0.13 | 0.27 | 0.71 | 0.56 |
| Input window's mean | **0.19** | 0.14 | 0.29 | 0.72 | 2.19 | 0.12 | 0.23 | 0.19 | 0.35 | 0.93 | 0.54 |

**Interpretation.**

- On **T1**, whose force level lies outside the training range, the input
  anchor cuts the level error by 79 %. It also helps on T6 (−76 %) and T2.
  This is the tool-shift case that RevIN targets.
- On **T4, T5 and T10** it is worse. There, the level changes between the
  input and the forecast window, at pass boundaries or with rising wear, so
  the input level is the wrong anchor and the model's learned level is
  better.
- RevIN keeps the model free to forecast a change from the anchor. The
  check therefore bounds the cases rather than predicting the result:
  large gains on shifted tools, a risk on tools whose level moves.
- This is why RevIN's rule (runbook §4) requires an MSE gain *and* no loss
  of accuracy.

### 8.3 The run-number shortcut and the photo's share of accuracy (B2, Step 8)

**Method** (`scripts/window_diagnostics.py --data-dir`).

1. Each fold's test windows are rebuilt. The rebuilt class targets equal the
   saved ones exactly.
2. Two references are computed:
   - the run-number baseline (the training tools' majority class per run
     number);
   - persistence (copy the shown photo's label).
3. The model's accuracy is split into the steps whose label differs from the
   shown photo's, and the others.

**Results** (main run 3a, fold means):

| Quantity | Value |
|---|---|
| Model accuracy | 0.673 |
| **Run-number baseline** | **0.708** |
| Persistence (copy the photo's label) | 0.913 |
| Target steps whose label differs from the photo's | 8.7 % |
| Model accuracy on those steps (argmax / ordinal + monotone) | 0.43 / 0.44 |
| Model accuracy on the other steps | 0.70 / 0.70 |
| Correlation over folds: model accuracy against run-number baseline | −0.36 |

The run-number baseline beats the model by the widest margin on T10 (0.90
against 0.67), T4 (0.86 against 0.53) and T2 (0.85 against 0.54).

**Interpretation.**

- **The class follows the run schedule closely.** A rule using only the run
  number scores 0.71 on held-out tools, above the model. The tools were worn
  on similar schedules, so the run number is a strong predictor in this
  dataset. It would not transfer to a tool used at a different rate.
- **The model does not exploit the shortcut well.** Its accuracy is lower
  and does not follow the baseline across folds (−0.36). Removing the codes
  (`--metadata`) is therefore unlikely to cost much accuracy. Whether it
  narrows the training-test gap is stage 1's question.
- **Most of the accuracy is photo reading.** Where the photo's label is
  right (91 % of steps), the model reaches 0.70. Where the label changes
  within the forecast (9 %), it reaches 0.43. The second number measures
  prognosis, and it is now saved for every new fold.

## 9. The Ruche plan

The full procedure, with commands, is in
[`TRAINING_RUNBOOK_RUCHE_FIXES.md`](TRAINING_RUNBOOK_RUCHE_FIXES.md). Its
design follows from §6-§8.

| Element | Choice | Reason |
|---|---|---|
| Stage 1 | 9 arms, each **one** change from the main runs: reference, modality dropout 0.2, metadata position, metadata none, RevIN, CORAL, photo edge crop (128 × 640), no image GAT, fine-tuned encoder | Attributes each effect; a factorial is unaffordable |
| Seeds | 5 (42, 142, 242, 342, 442), every arm and the reference on the same seeds | Single runs vary by up to 0.017 (§6.1); fold *i* uses seed + *i*, so the seeds are 100 apart |
| Comparison | Arm minus reference, per fold, on seed-averaged metrics; Wilcoxon over the 10 folds | The folds differ more than the arms; pairing removes that |
| Decision rules | Fixed before the runs, in `scripts/fixes_analysis.py`. Accuracy arms: +0.01 with p < 0.10. RevIN: MSE −3 % with p < 0.10 and accuracy not below −0.01. Metadata: accuracy not below −0.01 and a smaller training-test gap. CORAL: "used" recall +0.05 with p < 0.10 and accuracy not below −0.01. | Results cannot choose their own rule |
| Stage 2 | The adopted arms combined, on the same 5 seeds; reported with the seed ensemble and both decision rules | Tests the combination once, on its own runs |
| No training | A2 offsets and the phase study on the saved models (CPU jobs) | Answer two open questions without new runs |
| Cost | About 2.5 h per seed for stage 1 on an A100; 5 suites in parallel | A frozen-encoder run takes about 16 min |

## 10. Status of all fixes

| Fix | Status | Next step |
|---|---|---|
| A1 Decision rules | Done; validated on 51 runs and 7 replicates | Report with every result |
| A2 Validation offsets | Code ready | Ruche CPU job (runbook §7.1) |
| A3 Modality dropout | Accuracy gain not established; robustness gain real | Stage-1 arm |
| A4 Seeds and ensembles | Noise measured; ensemble script | 5 seeds in stages 1 and 2 |
| B1 Photo crop and size | Code ready; crop at 128 × 640 chosen by the probe | Stage-1 arm; fine-tuning on the crop in stage 2 if both pass |
| B2 Metadata switch | Code ready; run-number baseline in every fold | Stage-1 arms (position, none) |
| B3 RevIN | Code ready | Stage-1 arm |
| B4 CORAL | Code ready | Stage-1 arm |
| B5 Fine-tuned encoder, no image GAT | Existing options | Stage-1 arms (repeat the best single-seed results) |
| Step 8 Photo-change accuracy | In every new fold; computed for the main run | Report with every result |
| MSE-horizon check | Code ready | Ruche CPU job (runbook §7.2) |
| C1 Chip and workpiece photos | **Blocked**: the folders are not in `Nonastreda_Data/` | Download them, check that each photo belongs to one pass and when it was taken, then build a multi-view image branch |
| C2 Spectrogram as a third stream | Not started | A three-stream fusion design (C2F fuses two streams) |
| Photo augmentation | Not started | Only with a fine-tuned encoder (frozen features are cached) |
| Isotonic training labels | Deferred | Only if CORAL fails |

## 11. Files and tests

| File | Change |
|---|---|
| [`omnifuser/decision.py`](omnifuser/decision.py) | Round 1. Decision rules, offsets, scores |
| [`omnifuser/config.py`](omnifuser/config.py) | `metadata`, `revin`, `class_loss`, `image_crop` (validated; JSON lists become tuples) |
| [`omnifuser/models.py`](omnifuser/models.py) | `_metadata_tables`, `_add_metadata`, RevIN in `OmniFuser.forward`, the CORAL head, `ordinal_log_probabilities`, CORAL loss in `multitask_loss` |
| [`omnifuser/data.py`](omnifuser/data.py) | `load_photo` (crop, then resize) used by `image_tensor`, `ImageBank`, `collate_windows` |
| [`omnifuser/evaluation.py`](omnifuser/evaluation.py) | `photo_change_report`, `run_number_baseline_accuracy` |
| [`omnifuser/train.py`](omnifuser/train.py) | The crop reaches the image bank; `photo_change` and the run-number baseline are saved per test fold |
| [`omnifuser/cross_validation.py`](omnifuser/cross_validation.py) | Fold reuse compares rebuilt configs; summaries add persistence, run-number baseline and photo-change accuracy |
| [`omnifuser/model_bundle.py`](omnifuser/model_bundle.py), [`omnifuser/cli.py`](omnifuser/cli.py) | `predict --evaluation-stride`; the new training flags |
| [`scripts/run_experiment_suite.py`](scripts/run_experiment_suite.py) | `--study fixes` |
| [`scripts/fixes_analysis.py`](scripts/fixes_analysis.py) | New. Stage-1 and stage-2 analysis with the decision rules |
| [`scripts/ensemble_runs.py`](scripts/ensemble_runs.py) | New. Probability averaging of runs with the same test windows |
| [`scripts/photo_probe.py`](scripts/photo_probe.py) | New. §8.1 |
| [`scripts/window_diagnostics.py`](scripts/window_diagnostics.py) | New. §8.2-§8.3 |
| [`scripts/phase_study.py`](scripts/phase_study.py) | New. Re-scores saved models at other strides |
| [`slurm/status.sh`](slurm/status.sh) | Knows the `fixes` suite (9 variants) |
| [`tests/test_fixes.py`](tests/test_fixes.py) | New, 12 tests: each metadata mode removes exactly its codes; RevIN shift equivariance; ordinal probabilities; the CORAL head's valid, ordered probabilities and its loss; crop before resize; config JSON round trip and older configs; CLI flags; photo-change and run-number metrics; a fold trains, saves, reloads and re-predicts (also at another stride) with every option |
| [`tests/test_fixes_analysis.py`](tests/test_fixes_analysis.py) | New, 2 tests: seed averaging, paired comparison and the decision rules on synthetic runs |
| [`tests/test_experiment_suite.py`](tests/test_experiment_suite.py) | +1 test: each `fixes` arm changes exactly its own config fields |

The full test suite passes (168 tests); `ruff` and `pyright` report no issues. The end-to-end
trace (`scripts/end_to_end_trace.py`) gives the same numbers as before the changes, except for
two timings, so the protocol's pipeline is unchanged.

Outputs (not committed; `experiment_results/` is ignored):

- `experiment_results/ensembles/replicates-main-config.json`;
- `experiment_results/photo-probe/` (features, edge rows, `photo-probe.json`);
- `experiment_results/diagnostics/`.

## 12. How to reproduce

```bash
# Round 1, A1: all finished runs, from the saved test predictions
python scripts/decision_study.py

# §6: the seven replicates and their ensemble
R=experiment_results
python scripts/ensemble_runs.py $R/main/full-model $R/ablations/paper-v1/paper-ablation/full-model \
  $R/heads/direct $R/image-encoder/encoder-v1/image-encoder/pretrained-frozen \
  $R/missing-modality/paper-v1/missing-modality-robustness/no-modality-dropout \
  $R/sensitivity/paper-v1/parameter-sensitivity/sensitivity-baseline \
  $R/deduplication/dedup-v1/duplicate-handling/dedup-true --name replicates-main-config

# §8.1: photo probe (about 40 min on a busy laptop CPU; features are cached)
python scripts/photo_probe.py --data-dir Nonastreda_Data

# §8.2-§8.3: level offset, run-number baseline, photo-change accuracy
python scripts/window_diagnostics.py experiment_results/main/full-model --data-dir Nonastreda_Data

# A2 and the phase study need the fold-Tn.pt files: on Ruche, TRAINING_RUNBOOK_RUCHE_FIXES.md §7
```

Each run's entry in `decision-study.json` holds, per rule:

- `accuracy`, `accuracy_paper_horizons`, `precision_macro`, `f1_macro`;
- `recall_class_1` ("used" recall);
- `mean_absolute_class_error`, `changing_windows`, `decreasing_windows`;
- `accuracy_per_fold`, `folds_better`, `folds_worse`, `wilcoxon_p`.

With `--tune-offsets`, it also holds `fitted_offsets`.

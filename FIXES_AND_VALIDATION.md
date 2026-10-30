# Classification fixes: actions and validation

This document records the fixes started after the final results report
([`FINAL_REPORT_REPRODUCTION.md`](FINAL_REPORT_REPRODUCTION.md) §5): what was
implemented, how each choice was validated, the results, and what remains.
Started on 2026-10-30.

## Contents

1. [Summary](#1-summary)
2. [Constraints found before starting](#2-constraints-found-before-starting)
3. [Fix A1: decision rules that use the order of wear](#3-fix-a1-decision-rules-that-use-the-order-of-wear)
4. [Fix A2: class offsets tuned on the validation tool](#4-fix-a2-class-offsets-tuned-on-the-validation-tool)
5. [Fix A3: modality dropout](#5-fix-a3-modality-dropout)
6. [Status of the remaining fixes](#6-status-of-the-remaining-fixes)
7. [Files and tests](#7-files-and-tests)
8. [How to reproduce](#8-how-to-reproduce)

## 1. Summary

| Fix | Status | Validated on | Main effect |
|---|---|---|---|
| A1 Ordinal and monotone decision rules | **Implemented and validated** | Saved test predictions of all 51 runs, no retraining | "Used" recall +0.22, macro precision +0.03, macro F1 +0.025, forecast class flicker 41 % → 11 % of windows; accuracy +0.01 (small) |
| A2 Class offsets tuned on the validation tool | Implemented and unit-tested; **waiting for the model files** | Synthetic data only | To be measured |
| A3 Modality dropout 0.2 in training | **Evidence from an existing study**; a training change for future runs | The missing-modality study (same settings, ± dropout) | Accuracy +0.018 (7 of 10 folds; not significant) |
| B, C (images, shortcut, shift, ordinal training loss) | Not started; each needs new cluster runs or data | - | §6 |

**Recommendation.**

1. Report every classification result with two decision rules:
   - argmax, the paper's rule;
   - the monotone ordinal rule, the best of those validated here.
2. Train future runs with modality dropout 0.2.
3. Download the model files to finish A2.

The decision rules improve how the model's probabilities become classes,
especially for "used". They do not close the gap to the paper: accuracy
stays at 0.67-0.71. The large gains must come from better evidence (§6).

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
   CPU was at 76-90 % and 2.2 GB of RAM were free. All analyses here
   therefore use only the saved prediction files (NumPy, no model or raw data
   loading) and run in one process with 2 threads.

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
  laptop (§2).

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

## 6. Status of the remaining fixes

From the final report §5. None can be validated without new cluster runs or
data.

| Fix | What it needs | Validation plan |
|---|---|---|
| A2 Offsets on validation (§4) | `results-checkpoints.tar.gz` downloaded and unpacked in the repository (`TRAINING_RUNBOOK_RUCHE.md` §17) | `python scripts/decision_study.py experiment_results/main/full-model --tune-offsets --data-dir Nonastreda_Data` |
| A4 Several seeds and ensembling (F14) | 3 seeds of the main configuration on Ruche | Average the class probabilities over seeds, then decide with §3's rules; paired test against one seed |
| B1 Native-resolution crops of the cutting edge (F9) | Code for crops or larger inputs; Ruche runs | Photo-only (force-masked) accuracy above 0.65 |
| B2 Run-number shortcut switch (F2) | `--metadata {all,position,none}`; Ruche runs | Per fold against the run-number baseline (70.9 %) |
| B3 Tool-shift normalization (F16, F3) | `--revin`, Fz handling; Ruche runs | Training-test gap per fold (0.88 against 0.67) |
| B4 Ordinal training loss (F12) | `--class-loss coral`; Ruche runs | "Used" recall and MACE against §3's post-hoc rules: a trained ordinal head should beat a post-hoc one |
| C1 Chip and workpiece photos | **The two photo folders** (not in `Nonastreda_Data/`), then a check of their per-pass alignment and timing; code for several image views; Ruche runs | An `image-views` suite: tool, chip, workpiece, combinations |
| C2 Spectrogram stream alongside the photo | Code for a third stream; Ruche runs | Against the main run, with paired fold tests |

Suggested next order:

1. Download the checkpoints and run A2. This is local, takes minutes, and
   confirms or rejects §3's rule on validation data.
2. Download the chip and workpiece photos and check their alignment. This
   decides whether C1, the largest potential gain, is feasible.
3. Implement B2, B4 and the seeds together, and run them as one Ruche
   suite with modality dropout 0.2.

## 7. Files and tests

| File | Change |
|---|---|
| [`omnifuser/decision.py`](omnifuser/decision.py) | New. `argmax_decode`, `ordinal_median`, `monotone_decode`, `monotone_ordinal_decode`, `apply_offsets`, `fit_class_offsets`, `decision_scores` |
| [`scripts/decision_study.py`](scripts/decision_study.py) | New. Scores the rules on every finished run; `--tune-offsets` for A2 |
| [`tests/test_decision.py`](tests/test_decision.py) | New, 7 tests: monotone decoding equals brute force over all non-decreasing sequences; the combined rule equals brute force on expected class distance; flicker removal; the ordinal median; offsets recover a biased boundary and stay zero without gain; the counting of changing and decreasing windows |
| `experiment_results/decision/decision-study.json` | Output: per run and rule, fold means and per-fold accuracy, folds better/worse, Wilcoxon p (not committed; `experiment_results/` is ignored) |

The training and evaluation pipeline is unchanged, so every existing result
and saved model stays valid. The 7 new tests pass. The full suite was not
rerun because no existing module changed.

## 8. How to reproduce

```bash
# A1: all finished runs, from the saved test predictions (a few minutes, CPU)
python scripts/decision_study.py

# A2: after unpacking results-checkpoints.tar.gz in the repository
python scripts/decision_study.py experiment_results/main/full-model --tune-offsets --data-dir Nonastreda_Data
```

Each run's entry in `decision-study.json` holds, per rule:

- `accuracy`, `accuracy_paper_horizons`, `precision_macro`, `f1_macro`;
- `recall_class_1` ("used" recall);
- `mean_absolute_class_error`, `changing_windows`, `decreasing_windows`;
- `accuracy_per_fold`, `folds_better`, `folds_worse`, `wilcoxon_p`.

With `--tune-offsets`, it also holds `fitted_offsets`.

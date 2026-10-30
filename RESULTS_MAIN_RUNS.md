# Main-run results: steps 3a and 3b

This document records the first results of the two main runs of the OmniFuser
implementation on Nonastreda. It compares them with two simple baselines and
with the paper's published numbers, and interprets them. It is material for
the final report: the tables can be copied as they are. Section 9 lists the
checks that should come before the conclusions are final.

Recorded on 2026-10-28 from the output of `scripts/compare_with_paper.py`.
Values in brackets, such as [commit], are to be filled in from the run files.

## Contents

1. [Summary](#1-summary)
2. [What was run](#2-what-was-run)
3. [How to read the numbers](#3-how-to-read-the-numbers)
4. [Step 3a: duplicates merged (main result)](#4-step-3a-duplicates-merged-main-result)
5. [Step 3b: duplicates kept](#5-step-3b-duplicates-kept)
6. [3a against 3b](#6-3a-against-3b)
7. [Discussion](#7-discussion)
8. [Limitations](#8-limitations)
9. [Next analyses](#9-next-analyses)
10. [Suggested wording for the report](#10-suggested-wording-for-the-report)

## 1. Summary

Step 3a (the protocol) at three horizons:

| h | MSE | Last-value MSE | Paper MSE | Accuracy | Persistence | Paper accuracy |
|---|---|---|---|---|---|---|
| 12 | 1.034 | 0.669 | 0.467 | 0.671 | 0.983 | 0.886 |
| 48 | 0.761 | 0.682 | 0.686 | 0.674 | 0.950 | 0.866 |
| 96 | 0.809 | 0.867 | 0.902 | 0.673 | 0.913 | 0.863 |

- **Classification is the weak result.**
  - Accuracy is 0.667-0.675 at every horizon in both runs; macro precision is
    0.69-0.71.
  - That is 0.15-0.26 below the paper's OmniFuser.
  - It is 0.24-0.31 below copying the label of the photo the model is shown
    (persistence, 0.91-0.98).
  - Accuracy does not change with the horizon, although the shown photo's
    label is correct for 98 % of the first 12 leads. The label information
    that the same-pass photo carries does not reach the classifier.
- **Force forecasting beats the last-value baseline after the first 12
  seconds (3a).**
  - In the cumulative metric the model is worse than repeating the last input
    value up to h = 72. It is better at h = 84 and h = 96 (0.809 against 0.867
    at h = 96).
  - Splitting the cumulative values into 12-lead segments (§3.3) shows where
    this comes from. On leads 1-12 the model's error is 1.03 against 0.67 for
    last-value. On leads 13-24 the two are about equal. From lead 25 on, the
    model is better in every segment, by up to 0.34 on leads 85-96.
- **Against the paper's numbers**, 3a's MSE is higher at short horizons
  (1.034 against 0.467 at h = 12) and lower from h = 60 (0.809 against 0.902
  at h = 96). MAE is lower from h = 48. The paper reports on "Mudestreda"
  under settings it does not fully state, so this is a plausibility check,
  not a reproduction.
- **Keeping the duplicates (3b) barely changes the model but makes the
  baseline much stronger.**
  - The model's scores differ from 3a's by at most 0.024 in MSE beyond
    h = 12, and by at most 0.007 in accuracy.
  - The last-value MSE drops by 0.22-0.32. 127 of the 132 duplicate groups
    are bit-identical B3/B4 copies, so the series repeats a pass it has just
    shown.
  - The model does not exploit the repetition and never beats last-value in
    3b.
  - 3b's test windows differ from 3a's, so 3b is reported separately and not
    ranked against 3a.
- **Two weaknesses explain most of the gap.** Both are still hypotheses, and
  both can be tested with files already on Ruche (§9.1):
  1. the image path (a frozen ImageNet encoder on photos shrunk 4×) seems not
     to see wear;
  2. the force path does not carry the last observed level into the first
     forecast steps.

## 2. What was run

### 2.1 The two runs

| | 3a (main result) | 3b |
|---|---|---|
| Runbook step | `TRAINING_RUNBOOK_RUCHE.md` §13, step 3a | §13, step 3b |
| Output folder | `experiment_results/main/full-model/` | `experiment_results/main/full-model-no-dedup/` |
| Duplicate force recordings | Merged in training and test (`--dedup-identical-forces True --eval-dedup-identical-forces True`) | Kept in both (`False` / `False`) |
| Passes | 378 | 512 |
| Folds finished | 10 of 10 | 10 of 10 |

Everything else is identical. `EDA.md` §3 describes the duplicates: 134
passes whose force arrays are bit-identical to another pass of the same tool
and run, in 132 groups. 127 of these groups are B3 + B4 pairs, and 24 groups
carry different photo labels on identical forces. Merging keeps the most-worn
copy of each group.

### 2.2 The shared protocol

| Item | Setting |
|---|---|
| Data | Nonastreda: 10 tools (T1-T10), cutting forces Fx, Fy, Fz, and one tool photo per pass. Classes are sharp, used and dulled, from the photo labels (`--label-column image_label`). |
| Cross-validation | 10-fold leave-one-tool-out. In fold *i*, tool T*i* is the test tool, T*i+1* the validation tool (T1 after T10), and the other 8 tools are for training. The force normalizer is fitted on the training tools only. |
| Time step | 1 s: each pass is trimmed, then mean-pooled in blocks of 1,000 samples (`--pool 1000`). |
| Windows | 96 input steps, then 96 forecast steps. Stride 24 for training (1,127-1,213 windows per fold in 3a). Stride 96 for validation and test (28-51 test windows per fold in 3a). |
| Image input | The photo of the same pass (`--image-lag 0`, the paper's association). Each photo is taken after its pass. |
| Image encoder | EfficientNet-B0 with ImageNet weights, frozen (`--pretrained --freeze-image-encoder`). Features are cached once per fold. Then the paper's 4-layer image GAT, plus position and metadata codes. |
| Model | OmniFuser as in the paper: temporal embedding; MRTE (4 RTD blocks, stride 3, rank 8); C2F fusion (2 rounds, K = 4 landmarks, hybrid gate) with refinement; direct heads |
| Parameters | 4,183,600 in total, 176,052 of them trained. The frozen trunk holds 4,007,548. |
| Loss | MSE on z-scored forces + cross-entropy on classes |
| Training | Adam, learning rate 1e-4, batch 32, 30 epochs (about 1,100 optimizer steps in 3a), no learning-rate schedule. Seed 42 + fold index. |
| Model selection | The epoch with the lowest validation loss (`--select-best`, the default) |
| Hardware and software | Ruche, NVIDIA A100 (`gpua100`) [confirm in `tracking/run.json`]; PyTorch 2.14.1+cu126. About 3 s per epoch in the pilot. |
| Code version | [commit: `git log -1 --format=%H` in the Ruche clone] |

### 2.3 How the tables were produced

```bash
python scripts/compare_with_paper.py experiment_results/main/full-model \
  --json experiment_results/main/full-model/paper-comparison.json
python scripts/compare_with_paper.py experiment_results/main/full-model-no-dedup \
  --json experiment_results/main/full-model-no-dedup/paper-comparison.json
```

The script reads the ten `fold-T*.json` files of a run. It uses its
defaults, `--by mean_over_folds` and `--average macro`. Header lines as
printed:

```text
3a: 10 fold(s); image=photo (pretrained-frozen), image_lag=0, pool=1000, label=image_label,
    stride=24, horizon=96, dedup train/eval=True/True, passes train/eval=378/378
3b: 10 fold(s); image=photo (pretrained-frozen), image_lag=0, pool=1000, label=image_label,
    stride=24, horizon=96, dedup train/eval=False/False, passes train/eval=512/512
```

The derived tables (§4.2-§4.3, §5.2-§5.3, §6) were computed from the printed
values, which are rounded to three decimals.

## 3. How to read the numbers

### 3.1 Metrics and baselines

| Column | Definition |
|---|---|
| h | Horizon. Every metric is **cumulative**: it covers forecast leads 1…h of every test window, so h = 96 covers the whole forecast. A lead is one 1 s step. |
| MSE, MAE | Force error in z-units: forces standardized with the fold's training normalizer, averaged over windows, leads 1…h and the three channels Fx, Fy and Fz. 1.0 is one training standard deviation squared (MSE) or one standard deviation (MAE). |
| Acc | Accuracy: the share of target steps (window × lead ≤ h) whose predicted class (argmax) equals the label. The label of a target step is the class of the pass it falls in. |
| Pre | Macro precision: the unweighted mean of the three classes' precision |
| Last-value | Repeat the last input step's force for every lead. This is the forecast baseline. |
| Persistence | Copy the label of the photo shown at the last input step to every lead. At lag 0 this photo is taken after the window's last pass and shows that pass's label (`EDA.md` §7), so persistence is the accuracy available from reading the photo. |
| Paper | OmniFuser's published values on "Mudestreda" (Tables IV and V). See §3.2. |
| Fold averaging | Each fold's metric is computed on its test tool. The table shows the mean over the 10 folds, each with equal weight. |

### 3.2 Why the paper's numbers are only a reference

The paper's dataset reference [18] is the Nonastreda data paper, so it very
probably used the same data. But it does not state its time step, its
photo-to-force alignment, its split or its preprocessing. `EDA.md` §6
explains the 1 s time step: it is inferred from matching the difficulty of
trivial baselines, not taken from the paper.
`PAPER_IMPLEMENTATION_ALIGNMENT.md` lists the remaining differences. The
paper's numbers show whether ours are in a plausible range; a gap in either
direction does not prove a better or a worse model.

### 3.3 Derived quantities

- **Differences and ratios**: model minus baseline (or paper), and model MSE
  divided by last-value MSE. Below 1 means the model is better.
- **Errors relative to persistence**: (1 − Acc) / (1 − persistence), how many
  times more target steps the model gets wrong than persistence. The skill
  over persistence planned in `FUTURE_WORK_AND_ANALYSIS.md` F1 is 1 minus
  this ratio.
- **Segment values.** The cumulative values hide which leads cause the
  error. Every lead has the same number of windows, so the mean over leads
  a…b follows exactly from the cumulative values $C_h$:

  $$m_{a..b} = \frac{b\,C_b - (a-1)\,C_{a-1}}{b-a+1}$$

  This is exact for the fold mean too. Because the inputs are rounded to
  three decimals, segment values carry an uncertainty of about ±0.002 for
  leads 13-24, growing to about ±0.008 for leads 85-96.

## 4. Step 3a: duplicates merged (main result)

### 4.1 Reported scores

| h | MSE | Last-value MSE | Paper MSE | MAE | Paper MAE | Acc | Persistence | Paper Acc | Precision | Paper precision |
|---|---|---|---|---|---|---|---|---|---|---|
| 12 | 1.034 | 0.669 | 0.467 | 0.620 | 0.475 | 0.671 | 0.983 | 0.886 | 0.692 | 0.890 |
| 24 | 0.842 | 0.654 | 0.514 | 0.591 | 0.509 | 0.672 | 0.973 | 0.928 | 0.687 | 0.922 |
| 36 | 0.779 | 0.658 | 0.586 | 0.582 | 0.548 | 0.675 | 0.962 | 0.838 | 0.695 | 0.905 |
| 48 | 0.761 | 0.682 | 0.686 | 0.583 | 0.590 | 0.674 | 0.950 | 0.866 | 0.695 | 0.913 |
| 60 | 0.764 | 0.720 | 0.772 | 0.588 | 0.630 | 0.675 | 0.941 | 0.820 | 0.697 | 0.802 |
| 72 | 0.762 | 0.752 | 0.830 | 0.591 | 0.654 | 0.675 | 0.934 | 0.863 | 0.698 | 0.862 |
| 84 | 0.782 | 0.800 | 0.917 | 0.599 | 0.686 | 0.674 | 0.926 | 0.869 | 0.698 | 0.853 |
| 96 | 0.809 | 0.867 | 0.902 | 0.608 | 0.693 | 0.673 | 0.913 | 0.863 | 0.696 | 0.842 |

### 4.2 Against the baselines and the paper

| h | MSE − last-value | MSE ÷ last-value | MSE − paper | MAE − paper | Acc − persistence | Errors ÷ persistence errors | Acc − paper | Precision − paper |
|---|---|---|---|---|---|---|---|---|
| 12 | +0.365 | 1.55 | +0.567 | +0.145 | −0.312 | 19.4 | −0.215 | −0.198 |
| 24 | +0.188 | 1.29 | +0.328 | +0.082 | −0.301 | 12.1 | −0.256 | −0.235 |
| 36 | +0.121 | 1.18 | +0.193 | +0.034 | −0.287 | 8.6 | −0.163 | −0.210 |
| 48 | +0.079 | 1.12 | +0.075 | −0.007 | −0.276 | 6.5 | −0.192 | −0.218 |
| 60 | +0.044 | 1.06 | −0.008 | −0.042 | −0.266 | 5.5 | −0.145 | −0.105 |
| 72 | +0.010 | 1.01 | −0.068 | −0.063 | −0.259 | 4.9 | −0.188 | −0.164 |
| 84 | −0.018 | 0.98 | −0.135 | −0.087 | −0.252 | 4.4 | −0.195 | −0.155 |
| 96 | −0.058 | 0.93 | −0.093 | −0.085 | −0.240 | 3.8 | −0.190 | −0.146 |

### 4.3 By 12-lead segment

| Leads | Model MSE | Last-value MSE | Difference | Model Acc | Persistence | Targets whose label differs from the shown photo |
|---|---|---|---|---|---|---|
| 1-12 | 1.034 | 0.669 | +0.365 | 0.671 | 0.983 | 1.7 % |
| 13-24 | 0.650 | 0.639 | +0.011 | 0.673 | 0.963 | 3.7 % |
| 25-36 | 0.653 | 0.666 | −0.013 | 0.681 | 0.940 | 6.0 % |
| 37-48 | 0.707 | 0.754 | −0.047 | 0.671 | 0.914 | 8.6 % |
| 49-60 | 0.776 | 0.872 | −0.096 | 0.679 | 0.905 | 9.5 % |
| 61-72 | 0.752 | 0.912 | −0.160 | 0.675 | 0.899 | 10.1 % |
| 73-84 | 0.902 | 1.088 | −0.186 | 0.668 | 0.878 | 12.2 % |
| 85-96 | 0.998 | 1.336 | −0.338 | 0.666 | 0.822 | 17.8 % |

### 4.4 Observations

1. **Accuracy is flat.**
   - It stays between 0.671 and 0.675 cumulatively, and between 0.666 and
     0.681 in every segment.
   - Persistence falls from 0.983 to 0.822 per segment, as pass changes enter
     the horizon.
   - On leads 1-12, 98.3 % of targets carry the shown photo's label, yet the
     model gets only 67.1 % right. Its predictions therefore differ from the
     photo's label on at least 31 % of these targets.
2. **The model makes 3.8 to 19 times as many classification errors as
   persistence.** The ratio shrinks with h only because persistence gets
   worse, not because the model improves.
3. **Macro precision (0.687-0.698) is slightly above accuracy.** On average,
   no class is predicted with very low precision. Per-class scores need the
   confusion matrix (§9.1).
4. **The force error is U-shaped over the leads.**
   - Per segment, the model's MSE is 1.034 on leads 1-12, 0.65 on leads
     13-36, then rises to 1.00 on leads 85-96.
   - Last-value rises steadily from 0.67 to 1.34.
   - So the model is clearly worse than last-value only on leads 1-12. It is
     about equal on leads 13-24, and better on every segment from lead 25,
     with a margin that grows to 0.34.
   - The cumulative crossover at h ≈ 80 is an average of these two effects.
     It does not mean the model only becomes useful at long horizons.
5. **The leads 1-12 excess comes from a minority of large errors.**
   - MAE rises much less than MSE there: on leads 1-12 against leads 13-24,
     MAE is 0.620 against 0.562 while MSE is 1.034 against 0.650.
   - The ratio MSE / MAE² is 2.7 on leads 1-12 and 2.05-2.2 on the other
     segments. Gaussian errors would give 1.57; heavier tails give more.
   - The large errors are likely concentrated in some windows or folds,
     which the per-fold check in §9.1 can show.
6. **Leads 1-12 are about as hard for the model as forecasting the training
   mean.** That forecast scores about 1 in z-units (`EDA.md` §6). The exact
   value per horizon is `zero_prediction_mse` in `paper-comparison.json`.

## 5. Step 3b: duplicates kept

### 5.1 Reported scores

| h | MSE | Last-value MSE | Paper MSE | MAE | Paper MAE | Acc | Persistence | Paper Acc | Precision | Paper precision |
|---|---|---|---|---|---|---|---|---|---|---|
| 12 | 0.934 | 0.416 | 0.467 | 0.620 | 0.475 | 0.675 | 0.982 | 0.886 | 0.716 | 0.890 |
| 24 | 0.851 | 0.397 | 0.514 | 0.597 | 0.509 | 0.668 | 0.971 | 0.928 | 0.709 | 0.922 |
| 36 | 0.803 | 0.434 | 0.586 | 0.593 | 0.548 | 0.668 | 0.959 | 0.838 | 0.709 | 0.905 |
| 48 | 0.779 | 0.449 | 0.686 | 0.593 | 0.590 | 0.668 | 0.949 | 0.866 | 0.710 | 0.913 |
| 60 | 0.776 | 0.475 | 0.772 | 0.594 | 0.630 | 0.669 | 0.940 | 0.820 | 0.711 | 0.802 |
| 72 | 0.765 | 0.489 | 0.830 | 0.594 | 0.654 | 0.669 | 0.933 | 0.863 | 0.712 | 0.862 |
| 84 | 0.773 | 0.509 | 0.917 | 0.599 | 0.686 | 0.668 | 0.926 | 0.869 | 0.711 | 0.853 |
| 96 | 0.798 | 0.551 | 0.902 | 0.608 | 0.693 | 0.667 | 0.915 | 0.863 | 0.710 | 0.842 |

### 5.2 Against the baselines and the paper

| h | MSE − last-value | MSE ÷ last-value | MSE − paper | MAE − paper | Acc − persistence | Errors ÷ persistence errors | Acc − paper | Precision − paper |
|---|---|---|---|---|---|---|---|---|
| 12 | +0.518 | 2.25 | +0.467 | +0.145 | −0.307 | 18.1 | −0.211 | −0.174 |
| 24 | +0.454 | 2.14 | +0.337 | +0.088 | −0.303 | 11.4 | −0.260 | −0.213 |
| 36 | +0.369 | 1.85 | +0.217 | +0.045 | −0.291 | 8.1 | −0.170 | −0.196 |
| 48 | +0.330 | 1.73 | +0.093 | +0.003 | −0.281 | 6.5 | −0.198 | −0.203 |
| 60 | +0.301 | 1.63 | +0.004 | −0.036 | −0.271 | 5.5 | −0.151 | −0.091 |
| 72 | +0.276 | 1.56 | −0.065 | −0.060 | −0.264 | 4.9 | −0.194 | −0.150 |
| 84 | +0.264 | 1.52 | −0.144 | −0.087 | −0.258 | 4.5 | −0.201 | −0.142 |
| 96 | +0.247 | 1.45 | −0.104 | −0.085 | −0.248 | 3.9 | −0.196 | −0.132 |

### 5.3 By 12-lead segment

| Leads | Model MSE | Last-value MSE | Difference | Model Acc | Persistence | Targets whose label differs from the shown photo |
|---|---|---|---|---|---|---|
| 1-12 | 0.934 | 0.416 | +0.518 | 0.675 | 0.982 | 1.8 % |
| 13-24 | 0.768 | 0.378 | +0.390 | 0.661 | 0.960 | 4.0 % |
| 25-36 | 0.707 | 0.508 | +0.199 | 0.668 | 0.935 | 6.5 % |
| 37-48 | 0.707 | 0.494 | +0.213 | 0.668 | 0.919 | 8.1 % |
| 49-60 | 0.764 | 0.579 | +0.185 | 0.673 | 0.904 | 9.6 % |
| 61-72 | 0.710 | 0.559 | +0.151 | 0.669 | 0.898 | 10.2 % |
| 73-84 | 0.821 | 0.629 | +0.192 | 0.662 | 0.884 | 11.6 % |
| 85-96 | 0.973 | 0.845 | +0.128 | 0.660 | 0.838 | 16.2 % |

### 5.4 Observations

1. **Classification matches 3a.** Accuracy is 0.660-0.675 per segment and
   flat over the horizon. Macro precision (0.709-0.716) is about 0.015 higher
   than in 3a.
2. **The model never beats last-value**, at any horizon or in any segment.
   The gap is largest on leads 1-12 (+0.518) and narrows to +0.13 to +0.21
   from lead 25 on.
3. **The model's force error has the same U shape as in 3a**: 0.934 on leads
   1-12, 0.71 on leads 25-48 and 0.97 on leads 85-96. The early excess is
   smaller than in 3a (0.934 against 1.034, at the same MAE of 0.620).
4. **Last-value is much stronger than in 3a**: 0.38-0.85 per segment against
   0.64-1.34. With duplicates kept, a pass is often followed by an identical
   copy, so the force level in the forecast window repeats the level just
   seen.

## 6. 3a against 3b

3b minus 3a:

| h | MSE | Last-value MSE | MAE | Acc | Persistence | Precision |
|---|---|---|---|---|---|---|
| 12 | −0.100 | −0.253 | 0.000 | +0.004 | −0.001 | +0.024 |
| 24 | +0.009 | −0.257 | +0.006 | −0.004 | −0.002 | +0.022 |
| 36 | +0.024 | −0.224 | +0.011 | −0.007 | −0.003 | +0.014 |
| 48 | +0.018 | −0.233 | +0.010 | −0.006 | −0.001 | +0.015 |
| 60 | +0.012 | −0.245 | +0.006 | −0.006 | −0.001 | +0.014 |
| 72 | +0.003 | −0.263 | +0.003 | −0.006 | −0.001 | +0.014 |
| 84 | −0.009 | −0.291 | 0.000 | −0.006 | 0.000 | +0.013 |
| 96 | −0.011 | −0.316 | 0.000 | −0.006 | +0.002 | +0.014 |

How to read it:

- **The runs differ in two ways at once.** 3b trains on the duplicates *and*
  is tested on different windows, with the copies included and normalized
  with a normalizer fitted on data that contains them. The table therefore
  cannot say whether training on duplicates helps or hurts. The deduplication
  study (`experiment_results/deduplication/<run-id>/`) trains with and
  without duplicates and scores both on the same test windows. It answers
  that question.
- **What the table does show:**
  1. The test windows of 3b are much easier for a copy-the-last-value rule:
     last-value MSE is 0.22-0.32 lower, while persistence is unchanged.
  2. The model's own scores hardly move: MSE within ±0.024 beyond h = 12,
     accuracy within 0.007, precision within 0.024.
  3. So the model does not benefit from the repeated passes that make
     last-value so strong. This is a further sign that it makes little use of
     the most recent input (§7.2).
- **Size of the differences.** With one seed and 28-51 test windows per
  fold, differences of 0.01-0.02 are probably within run-to-run noise (§8).

## 7. Discussion

### 7.1 Classification: why 0.67, and why flat

A classifier that read the wear class off the shown photo would track
persistence: near 0.98 on the first leads, falling as the horizon crosses
into later passes. The model's accuracy is flat at 0.67 instead, and it
contradicts the photo's label on at least 31 % of the first 12 leads. Its
class predictions must therefore rest on something other than the photo's
label, and change little within a forecast window.

This also settles one concern for these runs. `FUTURE_WORK_AND_ANALYSIS.md`
F1 warns that at lag 0 the photo shows the target label, which could inflate
accuracy. These scores are not inflated by it; the opposite problem
dominates.

Possible causes, most likely first:

1. **The image path does not see wear.**
   - The encoder is an ImageNet EfficientNet-B0 with frozen weights, applied
     to photos shrunk about 4× to 384 × 128 (F9).
   - `EDA.md` §10 shows that wear is not visible in global image statistics:
     luminance and edge strength are the same in every class, so the encoder
     must resolve local edge detail at the cutting edge. Generic frozen
     features at this resolution may not encode it.
   - The paper calls its backbone a lightweight EfficientNet and does not say
     that it is frozen. Freezing is this project's choice
     (`PAPER_IMPLEMENTATION_ALIGNMENT.md`, Eq. 4).
   - **Tests:**
     - the image-masked evaluation that every fold has already saved
       (§9.1): if image-masked accuracy is close to 0.67, the photos
       contribute almost nothing;
     - the image-encoder study (step 7: from scratch, fine-tuned, frozen, and
       frozen without the image GAT).
2. **The class head relies on a slowly varying shortcut.**
   - A rule that predicts each pass's class from its run number reaches
     70.9 % on held-out tools (F2, against 38.9 % for the majority class).
     The model receives the run number through its metadata codes.
   - Such a cue is constant within a pass, so it would give flat accuracy in
     the observed range.
   - **Tests:** per-fold accuracy against the run-number baseline, and the
     metadata ablation proposed in F2 (not implemented).
3. **Under-training or selection noise.**
   - About 1,100 Adam steps at learning rate 1e-4 with no schedule (F5).
   - The best epoch is chosen on one validation tool, by the total loss in
     which the force term may dominate (F4).
   - **Tests:** the training-set accuracy and the selected epoch per fold
     (§9.1), and the learning curves (`REPORT_PLAN.md` snippet E). Training
     accuracy near 0.7 would point to under-fitting. Training accuracy well
     above test accuracy would point to a generalization gap across tools.
4. **Label noise caps any classifier that cannot read the label.** Flank
   wear decreases at 189 of 502 steps, and the class at 29 of 368 (F12).
   This limits every method except persistence, which copies the label
   itself.

### 7.2 Force forecasting: good beyond lead 24, weak at the start

**The strength.**

- From lead 25 on, 3a's model beats last-value in every segment.
- Passes last 98.5 s at the median (`EDA.md` §4), about one forecast window,
  and each pass is a plateau framed by entry and exit transients. Most
  96-step forecasts therefore cross a pass boundary.
- A flat copy of the last value cannot anticipate that crossing; a learned
  typical profile can. This is where the model earns its advantage, and the
  advantage grows with the lead (−0.34 on leads 85-96).

**The weakness: leads 1-12.**

- There, last-value is accurate (0.67), and the model is as poor as
  forecasting the training mean.
- The heavy-tailed errors (§4.4, item 5) suggest that some windows start
  the forecast far from the current level.

Possible causes:

1. **No direct path from the latest input to the heads (F19).** MRTE returns
   only the sum of its restored components, so the latest seconds reach the
   heads only through paths downsampled 3×. The direct heads read only the
   last step of the fused sequence.
2. **Level shift between tools (F16).**
   - Test tools sit 0.16-0.32 training standard deviations from the training
     mean (median absolute shift), and T1's Fx sits at +1.15.
   - A model that pulls its forecast toward training-typical levels pays most
     on the first leads, where the last value is most informative.
   - The reversible instance normalization in F16 (RevIN) gives the model the
     window's own level and is a candidate fix.
3. **Fz drift (F3).** A third of the target is a sawtooth that resets every
   1-3 passes. The per-channel curve (§9.1) shows how much of the early
   error is Fz.
4. **Under-training (F5)**, as in §7.1.

The 3b result supports the first two causes. Repeated passes make the next
level predictable from the input, yet the model's error barely changes.

### 7.3 Against the paper

- **MSE and MAE.**
  - The paper's MSE rises steadily from 0.467 to 0.902, the shape of a
    forecaster that is precise near-term and degrades with the lead.
  - Ours is high at short horizons and flatter (0.76-1.03 cumulative). It is
    lower than the paper's from h = 60 and higher below h = 48.
  - Our MAE is lower than the paper's from h = 48.
- **Accuracy and precision are lower at every horizon**: accuracy by 0.15 to
  0.26, precision by 0.09 to 0.24.
- **On our data, lag-0 persistence (0.91-0.98) is above the paper's
  accuracy (0.82-0.93)** (`EDA.md` §7). If the paper used this data with a
  similar time step and the same-pass alignment, its classifier would also
  be below copying the photo's label. That cannot be checked from the paper.
- The comparison supports two statements, and no more:
  1. our forecasting error is in the paper's range;
  2. our classification is clearly below the paper's numbers.

### 7.4 What the duplicates show

The duplicates matter more for the baseline than for the model (§6). In the
report:

- quote 3a as the main result;
- give 3b only as the effect of keeping the dataset as distributed;
- never compare 3b's model against 3a's baselines.

## 8. Limitations

- **One seed per fold, small test sets.**
  - Each fold ran once (F14), on 28-51 test windows in 3a.
  - These tables show fold means without their spread. The standard
    deviations are in `cross-validation-results.json` →
    `paper_horizons_mean_std`, and `REPORT_PLAN.md` snippet B prints them.
  - No significance test has been run yet (F11). Small differences, such as
    3a against 3b, should be called inconclusive until then.
- **Fold mean, not pooled.** Every tool counts equally whatever its number
  of windows. `--by pooled_by_windows` gives the window-weighted version;
  report which one a table uses (`REPORT_PLAN.md` §4).
- **Different normalizers.** z-units are relative to each fold's training
  normalizer. 3a and 3b fit theirs on different training data, so their
  z-units differ slightly.
- **The force metric includes Fz drift (F3).** Fx and Fy alone may give a
  different picture. Per-channel z-scored errors are in each fold JSON
  (`test.regression.by_channel_normalized`).
- **Labels are noisy near the class thresholds (F12).**
- **Derived values come from rounded inputs** (§3.3).
- **The paper's numbers are not like-for-like** (§3.2).

## 9. Next analyses

### 9.1 From files already saved (no retraining)

Run these from the repository root on Ruche, or locally after copying
`experiment_results/main/` (the `*.pt` model files are needed only for
item 4).

1. **Fold spread.** Run `REPORT_PLAN.md` snippet B (mean ± std per horizon)
   and snippet A (fold mean ± std and pooled out-of-fold headline metrics).
   Do this for both runs.
2. **Per fold: training against test accuracy, image-masked and
   sensor-masked accuracy, persistence, selected epoch.** This tests §7.1
   causes 1 and 3 directly:

   ```python
   import glob, json
   RUN = "experiment_results/main/full-model"
   paths = sorted(glob.glob(f"{RUN}/fold-T*.json"), key=lambda p: int(p.rsplit("-T", 1)[1][:-5]))
   print("| Test tool | Epoch | Train acc | Test acc | Image masked | Sensor masked | Persistence |\n|---|---|---|---|---|---|---|")
   for path in paths:
       fold = json.load(open(path))
       m = fold["modality_evaluations"]
       print(f"| {fold['test_tool']} | {fold['selected_epoch']} | {m['full']['train']['classification_accuracy']:.3f} | "
             f"{m['full']['test']['classification_accuracy']:.3f} | {m['image_missing']['test']['classification_accuracy']:.3f} | "
             f"{m['sensor_missing']['test']['classification_accuracy']:.3f} | {fold['test']['baselines']['persistence_class_accuracy']:.3f} |")
   ```

   How to read it:

   - image-masked accuracy close to test accuracy: the photos add almost
     nothing (cause 1);
   - training accuracy close to test accuracy: under-fitting (cause 3);
   - selected epochs near 30: training stopped while still improving (F5).

   `REPORT_PLAN.md` snippet G gives the pooled version per mode.
3. **Per-lead and per-channel force error.** This locates the leads 1-12
   excess (§7.2):

   ```python
   import glob, json
   import numpy as np
   RUN = "experiment_results/main/full-model"
   curves = []
   for path in sorted(glob.glob(f"{RUN}/fold-T*.json")):
       fold = json.load(open(path))
       with np.load(path.replace(".json", "-predictions.npz")) as arrays:
           error = (arrays["force_predictions"] - arrays["force_targets"]) / np.asarray(fold["normalizer_scale"])
       curves.append(np.square(error).mean(axis=0))  # (96 leads, 3 channels), z-scored MSE
   curve = np.mean(curves, axis=0)  # mean over folds, as in compare_with_paper.py
   print("| Lead | Fx | Fy | Fz | All |\n|---|---|---|---|---|")
   for lead in (1, 2, 3, 6, 12, 18, 24, 36, 48, 72, 96):
       fx, fy, fz = curve[lead - 1]
       print(f"| {lead} | {fx:.3f} | {fy:.3f} | {fz:.3f} | {curve[lead - 1].mean():.3f} |")
   ```

   - **Check:** the mean of the "All" column over leads 1…12 must reproduce
     the table's 1.034.
   - Printing `np.square(error).mean(axis=(1, 2))[:12]` inside the loop shows
     whether a few windows or one fold cause the excess.
   - The prediction files store forces in newtons; dividing by the fold's
     `normalizer_scale` returns them to z-units.
4. **Training-tool forecasts.** For a few folds, run
   `omnifuser predict --model experiment_results/main/full-model/fold-T10.pt --data-dir Nonastreda_Data --tool T3 --output-dir experiment_results/predictions/full-model-T10-on-T3`
   (T3 is one of fold T10's training tools). It warns that the scores are
   not test scores. A U-shaped per-lead curve
   on training tools too would point to the architecture (F19) or to
   under-training (F5). A U shape on test tools only would point to the
   level shift between tools (F16).
5. **Baselines not in the printed table.** `paper-comparison.json` also
   holds `zero_prediction_mse` (forecast the training mean) and
   `last_value_mae`:

   ```bash
   python -c "import json; t = json.load(open('experiment_results/main/full-model/paper-comparison.json'))['mean_over_folds']; [print(h, round(v['zero_prediction_mse'], 3), round(v['last_value_mae'], 3)) for h, v in t.items()]"
   ```

6. **Confusion matrix and per-class scores** (`REPORT_PLAN.md` snippet C),
   **ROC curves** (snippet D) and **learning curves** (snippet E).

### 9.2 From runs still in the queue

- **Image-encoder study (step 7).** Compares from scratch, fine-tuned,
  frozen, and frozen without the image GAT. It shows whether the image path
  can learn wear at all and what the GAT adds (§7.1, cause 1).
- **Image-source study.**
  - The previous-pass photo (lag 1) against 3a shows how much the same-pass
    photo explains.
  - Input-window spectrograms show whether the raw force carries wear
    information that 1 s pooling removes.
  - Compare each arm with its own persistence baseline.
- **Deduplication study.** Isolates the training effect of duplicates on
  the same test windows (§6).
- **Ablations, sensitivity, missing modality.** Read them against 3a with
  paired fold-level tests (`REPORT_PLAN.md` snippet H).

### 9.3 Needing code changes (future work, not implemented)

These are described with implementation steps in
`FUTURE_WORK_AND_ANALYSIS.md`:

| Finding | What it would show or fix here |
|---|---|
| F1 | Accuracy on targets whose label differs from the photo, and skill over persistence in every result |
| F2 | Metadata switch and run-number baseline: separates wear sensing from run counting (§7.1, cause 2) |
| F3 | Fx/Fy-first force metric; Fz handling |
| F4, F5 | Smoothed model selection; learning-rate schedule and a learning-rate check |
| F11, F14 | Significance tests; several seeds |
| F16 | RevIN option for the level shift between tools (§7.2) |
| F19 | MRTE residual path: the latest input reaches the heads directly (§7.2) |

## 10. Suggested wording for the report

The paragraph below can be adapted for the results section. The numbers in
it are those of §4.

> Under 10-fold leave-one-tool-out cross-validation on Nonastreda (1 s time
> step, 96-step input and forecast, same-pass photos, frozen ImageNet
> EfficientNet-B0), OmniFuser reaches a cumulative z-scored force MSE of
> 1.034 at h = 12 and 0.809 at h = 96. The last-value baseline scores 0.669
> and 0.867. Split by lead, the model is worse than last-value only on the
> first 12 forecast steps; from step 25 on it is better in every 12-step
> segment, by up to 0.34. Wear-class accuracy is 0.67 at every horizon, with
> macro precision 0.69-0.70. This is below the paper's reported 0.82-0.93,
> which come from settings the paper does not fully state, and far below the
> 0.91-0.98 obtained by copying the label of the same-pass photo the model is
> shown. Accuracy does not depend on the horizon, which indicates that the
> class head does not use the label information in the photo. With duplicate
> passes kept (512 instead of 378 passes), the model's MSE changes by at most
> 0.024 beyond h = 12 and its accuracy by less than 0.01. The last-value
> baseline, however, improves by 0.22-0.32, and the model no longer beats it
> at any horizon.

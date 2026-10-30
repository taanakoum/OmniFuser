# OmniFuser reproduction on Nonastreda: final results report

This report gives the final results of the OmniFuser implementation on the
Nonastreda milling dataset. It covers what was run, how each part of the
model contributed, why classification falls short of the paper and how to fix
it, and why the forecast error behaves differently over the horizon than in
the paper.

## Contents

1. [Summary](#1-summary)
2. [Setup](#2-setup)
3. [Results of all studies](#3-results-of-all-studies)
4. [The classification problem in detail](#4-the-classification-problem-in-detail)
5. [Steps to rectify classification](#5-steps-to-rectify-classification)
6. [Why our MSE falls with the horizon while the paper's rises](#6-why-our-mse-falls-with-the-horizon-while-the-papers-rises)
7. [Conclusions](#7-conclusions)
8. [Limitations](#8-limitations)

## 1. Summary

| | MSE | MAE | Accuracy | Macro precision |
|---|---|---|---|---|
| Main run 3a (duplicates merged) | 0.817 | 0.595 | **0.674** | **0.695** |
| Main run 3b (duplicates kept) | 0.810 | 0.600 | 0.669 | 0.711 |
| Last-value baseline (3a) | 0.725 | - | - | - |
| Persistence (copy the shown photo's label) | - | - | 0.948 | - |
| Paper OmniFuser (mean of 8 horizons) | 0.747 | 0.597 | 0.867 | 0.874 |

All values are means over the 10 folds and over the paper's 8 horizons
(12, 24, …, 96).

1. **The architecture is reproduced and trains stably.** The paper's
   components help classification in the paper's direction. The full model
   has the best accuracy among the ablations.
2. **Forecasting reaches the paper's scale.** MAE is 0.595 against the
   paper's 0.597.
   - The shape over the horizon differs. Our cumulative MSE falls from 1.03
     at h = 12 to 0.76 at h = 48; the paper's rises from 0.47 to 0.90.
   - Two measurable causes explain this (§6). The forecast start is
     phase-locked to pass boundaries, and two test tools (T1 and T5) have
     very large errors on the first leads.
   - Without T1 and T5, our curve rises like the paper's: 0.49 at h = 12,
     0.56 at h = 96.
3. **Classification is not reproduced.** Accuracy is 0.674 and macro
   precision 0.695, about 0.19 below the paper, and far below the 0.948 of
   copying the label of the photo the model is shown.
   - The model ranks classes well (macro ROC AUC 0.87).
   - Most errors are confusions between neighbouring classes. The middle
     class, "used", has a recall of only 0.49.
   - Training accuracy is 0.88, so the problem is generalization to an
     unseen tool, not under-fitting.
   - The photo carries most of the class information, but the downscaled
     tool photo through a frozen ImageNet encoder reads it poorly.
4. **The most promising fixes** (§5):
   - better image evidence: chip and workpiece photos, which were not used,
     and higher-resolution tool photos;
   - removing the run-number shortcut;
   - normalization against tool-to-tool shift;
   - an ordinal loss and temporal consistency;
   - several seeds.

## 2. Setup

| Item | Setting |
|---|---|
| Data | Nonastreda: 10 tools, cutting forces Fx, Fy and Fz. Only the **tool** photos were used, one per pass; the chip and workpiece photo folders were not. Classes sharp, used and dulled. |
| Validation | 10-fold leave-one-tool-out cross-validation. In each fold, one tool is the test tool, the next is the validation tool, and 8 are for training. |
| Time step and windows | 1 s steps (1,000 samples mean-pooled); 96 input steps, 96 forecast steps |
| Image input | The photo of the same pass (lag 0, the paper's association) |
| Image encoder | EfficientNet-B0, ImageNet-pretrained and frozen, then the image GAT |
| Model | Temporal embedding; MRTE (4 RTD blocks); C2F fusion (2 rounds) with refinement; direct heads. 4.18 M parameters, 176 k of them trained. |
| Training | MSE + cross-entropy, Adam, learning rate 1e-4, batch 32, 30 epochs, the best validation epoch selected, one seed per fold |
| Runs | 49 finished runs on Ruche (NVIDIA A100): main runs, ablations, image encoder, image source, prediction heads, paper-literal arms, deduplication, missing modality, sensitivity |
| Source of the numbers | Each run's `paper-comparison.json`, plus `fold-T*.json`, `cross-validation-results.json` and the prediction files of the main run |

The paper reports on "Mudestreda", whose reference is the Nonastreda data
paper. It does not state its time step, alignment or split, so its numbers
are a plausibility reference, not a like-for-like target.

## 3. Results of all studies

MSE is the mean over horizons. Differences in accuracy below about 0.03 are
within noise. There is one seed per fold, test tools have 28-51 windows, and
the accuracy spread across folds is about 0.09.

### 3.1 Component ablations (the paper's Fig. 13 analogue)

| Variant | MSE | Accuracy | Precision |
|---|---|---|---|
| Full model | 0.812 | **0.676** | 0.693 |
| Concatenation instead of C2F | 0.764 | 0.658 | 0.668 |
| Without C2F | 0.764 | 0.647 | 0.650 |
| Without MRTE | 0.848 | 0.650 | 0.676 |
| Without ALS | 0.881 | 0.668 | 0.696 |
| Without hybrid gate | 0.858 | 0.657 | 0.681 |
| Without refinement (RR) | 0.886 | 0.618 | 0.641 |
| Without RR and MRTE | 0.864 | 0.666 | 0.683 |
| Moving-average trend | 0.915 | 0.650 | 0.644 |

- **Every component helps accuracy.** Removing one lowers it by 0.01-0.06.
  Refinement matters most (−0.058).
- **For force forecasting, the fusion is not needed.** Plain concatenation
  and no C2F give a lower MSE (0.764) than the full model (0.812).

### 3.2 Image encoder and image source

| Variant | MSE | Accuracy | Precision |
|---|---|---|---|
| EfficientNet trained from scratch | 0.894 | 0.557 | 0.559 |
| Pretrained, fine-tuned | 0.780 | **0.693** | 0.692 |
| Pretrained, frozen (main run) | 0.813 | 0.676 | 0.695 |
| Pretrained, frozen, without image GAT | 0.798 | 0.688 | 0.694 |
| Photo of the previous pass (lag 1) | 0.895 | 0.611 | 0.621 |
| Spectrogram of the input force, frozen encoder | **0.749** | 0.653 | 0.660 |
| Spectrogram, fine-tuned | 0.813 | 0.687 | 0.667 |
| Spectrogram, from scratch | 0.801 | 0.654 | 0.684 |

- **Pretraining is decisive** (+0.12 accuracy over from scratch).
  Fine-tuning adds +0.017, which is not significant.
- **The image GAT brings no gain**: 0.688 without it, 0.676 with it.
- **The previous-pass photo costs 0.065 accuracy.** That photo is not
  taken after the target's pass, so the model has to forecast the wear
  instead of reading it.
- **Spectrograms of the raw force give the best forecast** (MSE 0.749, MAE
  0.562). They reach an accuracy of 0.65-0.69 with no photo at all.

### 3.3 Other studies

| Study | Result |
|---|---|
| Duplicate handling (both arms scored on the same test windows) | Merged 0.669 / kept 0.658 accuracy: no clear effect |
| Prediction head | Direct 0.677, lead-conditioned 0.618, paper-literal (repeated) 0.610 accuracy |
| Paper-literal readings (Eq. 3, Eq. 10, repeated head) | Accuracy 0.61-0.64 and MSE about 0.96, all worse than our reading |
| Missing modality | See §4.6 |
| Sensitivity | Width 64 gives 0.699 accuracy; width 16 collapses to 0.592. The other axes give 0.60-0.68, with no consistent trend. |

## 4. The classification problem in detail

### 4.1 What the model gets right and wrong

Pooled over all 369 test windows of the main run (35,424 target steps):

| | Precision | Recall | F1 | ROC AUC | Share of targets |
|---|---|---|---|---|---|
| Sharp | 0.786 | 0.842 | 0.813 | 0.927 | 38 % |
| Used | 0.566 | **0.487** | 0.524 | 0.776 | 32 % |
| Dulled | 0.629 | 0.667 | 0.647 | 0.894 | 30 % |
| Macro | 0.661 | 0.665 | 0.661 | 0.866 | |

Confusion matrix (rows: true class, columns: predicted class):

| True \ predicted | Sharp | Used | Dulled |
|---|---|---|---|
| Sharp | **11,334** | 1,411 | 709 |
| Used | 2,408 | **5,556** | 3,441 |
| Dulled | 671 | 2,850 | **7,044** |

Five observations:

1. **The model separates the classes well, but its decisions are poor.**
   Macro ROC AUC is 0.87; sharp reaches 0.93 and dulled 0.89. The class
   probabilities rank the classes correctly most of the time, but the argmax
   decision gives only 0.67 accuracy. Part of the gap is miscalibrated
   decision thresholds rather than missing information.
2. **The errors are ordinal.** 88 % of all errors (10,110 of 11,490) are
   between neighbouring classes: sharp↔used or used↔dulled. Only 12 % jump
   between sharp and dulled. The model knows roughly where on the wear scale
   a tool is; it misplaces the boundaries.
3. **"Used" is the problem class.** Its recall is 0.49: half of the "used"
   targets are called sharp (21 %) or dulled (30 %). "Used" is the band
   between two thresholds on the flank wear. Measurement noise of about
   ±10 µm makes both of its edges uncertain (`FUTURE_WORK_AND_ANALYSIS.md`
   F12: wear decreases at 189 of 502 steps), and cross-entropy treats it as
   unrelated to its neighbours.
4. **The predicted class shares match the truth**: 41 / 28 / 32 % predicted
   against 38 / 32 / 30 % true. There is no collapse onto the majority
   class.
5. **The predictions flicker.** The predicted class changes within the
   96-step forecast in 147 of 369 windows (40 %). The true class changes in
   only 76 (21 %). The model has no notion that a tool's wear class changes
   rarely and only upwards.

Pooled macro precision (0.661) is lower than the fold mean (0.695) because
the pooled value weights every window equally. Report both, and say which
one each number is.

### 4.2 Not under-fitting: a cross-tool generalization gap

Per-fold results of the main run:

| Test tool | Selected epoch | Training accuracy | Test accuracy | Photo masked | Force masked | Persistence |
|---|---|---|---|---|---|---|
| T1 | 20 | 0.895 | 0.623 | 0.355 | 0.439 | 0.863 |
| T2 | 15 | 0.844 | 0.543 | 0.360 | 0.616 | 0.976 |
| T3 | 14 | 0.805 | 0.626 | 0.395 | 0.542 | 0.948 |
| T4 | 18 | 0.903 | 0.532 | 0.319 | 0.499 | 0.987 |
| T5 | 21 | 0.937 | 0.757 | 0.754 | 0.722 | 0.974 |
| T6 | 30 | 0.931 | 0.630 | 0.473 | 0.587 | 0.738 |
| T7 | 30 | 0.931 | 0.760 | 0.159 | 0.860 | 0.962 |
| T8 | 29 | 0.922 | 0.809 | 0.620 | 0.773 | 0.818 |
| T9 | 26 | 0.915 | 0.783 | 0.573 | 0.823 | 0.920 |
| T10 | 14 | 0.741 | 0.670 | 0.314 | 0.667 | 0.941 |
| **Mean** | | **0.882** | **0.673** | **0.432** | **0.653** | **0.913** |

- **Training accuracy is 0.88; held-out accuracy is 0.67.** The model fits
  the 8 training tools and loses about 0.21 on an unseen tool.
  - Longer training would not help: half of the folds selected an epoch
    between 14 and 20, so validation loss stopped improving early.
  - The deficit is **transfer to a new tool**, not capacity or training
    time.
- **Per tool, test accuracy ranges from 0.53 (T4) to 0.81 (T8).** On T2
  and T4, copying the photo's label is right 98-99 % of the time, yet the
  model scores 0.53-0.54. On those tools it reads the photo especially badly.
- **Why tools differ.** Each tool has its own force level and photo
  appearance (`EDA.md` §8, §10):
  - test tools sit 0.16-0.32 standard deviations away from the training
    force distribution, and T1's Fx sits at +1.15;
  - T1 and T2 alone have full-resolution photos.
  With only 8 training tools, the model has seen few such variations.

### 4.3 The photo is informative, but poorly read

The same-pass photo is taken after its pass and shows that pass's wear
class. Copying its label is therefore right for 95 % of target steps. A model
that could read the wear class from the photo would approach that number. Our
model reaches 0.67, so most of the label information in the photo is lost.
The likely reasons:

1. **Resolution.**
   - Photos are about 1550 × 500 (T1 and T2: 3100 × 1000) and are resized to
     384 × 128, a 4× to 8× reduction.
   - Flank wear is a thin band at the cutting edge, tens of micrometres
     wide, and resizing removes most of it.
   - Global image statistics are the same in all classes (`EDA.md` §10:
     luminance 0.50-0.52, edge strength 0.055-0.060), so only fine local
     detail separates the classes.
2. **Frozen generic features.** The ImageNet EfficientNet was trained on
   objects, not on micro-wear. Fine-tuning helped only slightly (0.693),
   because 8 tools × about 38 passes is too little data to adapt a backbone.
3. **The image GAT and C2F do not add image understanding.** Removing the
   GAT does not hurt (§3.2).

### 4.4 Shortcuts and label noise

- **The run number is a shortcut.**
  - A rule that predicts each pass's class from its run number scores
    70.9 % on held-out tools, against 38.9 % for the majority class.
  - The model receives the run number as a metadata code in both streams.
  - This likely explains part of the high training accuracy, and it
    transfers only to tools with a similar wear schedule.
- **Label noise caps any classifier.** Class decreases occur 29 times in 368
  steps, and duplicated force recordings carry conflicting labels 24 times
  (`EDA.md` §3).

### 4.5 Why precision is close to accuracy

Macro precision (fold mean 0.695) is close to accuracy (0.674) because the
predicted shares match the true shares (§4.1, observation 4). Sharp is
predicted with high precision (0.79). Used (0.57) and dulled (0.63) are
lower, because the used↔dulled confusion goes both ways (3,441 and 2,850
target steps). Precision rises in 3b (0.711) only because duplicates change
the test windows.

### 4.6 Which input carries the classification

| Model and input | Accuracy |
|---|---|
| Main run, both inputs | 0.673 |
| Main run, force masked (photo and codes only) | 0.653 |
| Main run, photo masked (force and codes only) | 0.432 |
| Trained with modality dropout 0.2, both inputs | 0.681 |
| Same, photo masked | 0.602 |
| Same, force masked | 0.649 |

- **The photo carries most of the classification.** Removing it costs
  0.24, against 0.02 for removing the force.
- **The force alone is worth about 0.60** once the model is trained to
  cope without photos. Worn tools cut with higher force (ρ = 0.70), a real
  but weaker signal.
- **Fusion adds little over the photo.** Both inputs give 0.67-0.68,
  against 0.65 for photo only.
- The main run was trained without modality dropout, so masking is new to
  it. The dropout run gives the fairer picture: its masked accuracies are
  0.60 and 0.65.

### 4.7 Why the paper reports 0.87

This cannot be established from the paper. The candidate explanations,
from most to least likely:

- **A different alignment or time step.** For example, pass-level steps,
  or random window splits within tools instead of leave-one-tool-out
  validation.
  - Random splits put neighbouring windows of the same tool in training and
    test.
  - This removes exactly the cross-tool gap of §4.2.
- **A fine-tuned or larger image model on higher-resolution photos.**
- **A different label definition or class balance.**

The paper's own accuracy is not monotonic over the horizon (0.886, 0.928,
0.838, …). That pattern is more typical of a small, noisy test set than of a
forecasting effect.

### 4.8 Root causes, ranked

1. **Cross-tool generalization (§4.2)**: a 0.21 gap between training and
   test, made worse by tool-specific force levels and photo appearance, and
   by only 8 training tools.
2. **Weak image evidence (§4.3)**: downscaled tool photos and frozen generic
   features.
3. **Decision quality (§4.1)**:
   - the ordinal structure is ignored;
   - "used" is a narrow, noisy band;
   - the predictions flicker within a window;
   - good ranking (AUC 0.87) turns into a mediocre argmax.
4. **The run-number shortcut (§4.4)**, which inflates training fit and does
   not transfer.
5. **Label noise (§4.4)**, which caps every method that cannot read the
   label.

## 5. Steps to rectify classification

Each step lists the change, where to make it, the expected effect, and how to
verify it. Items marked (F) are specified in detail in
`FUTURE_WORK_AND_ANALYSIS.md`. Steps 1-4 address the largest causes and should
come first.

### Step 1. Better image evidence: chip and workpiece photos (not used so far)

- **Why.** Chip photos show chip colour, curl and segmentation, which change
  with tool wear and cutting temperature. Workpiece photos show the machined
  surface, whose roughness and burr formation increase as the tool dulls.
  - Both are well-known wear indicators in machining.
  - They do not depend on resolving a thin flank-wear band, so they may hold
    up better after resizing than the tool photo.
- **Before using them, check the alignment and leakage:**
  - one photo per pass;
  - which pass each photo belongs to;
  - whether each is taken during or after the pass.
  A photo of the chips or surface from pass *p* is evidence about pass *p*,
  so treat it like the tool photo (lag 0 and lag 1 variants).
- **How.**
  1. Extend `omnifuser/data.py` to load the extra folders by pass (as
     `image_path` does for the tool), with `--image-folders tool,chip,workpiece`.
  2. In `ImageBranch`, encode each view with the shared EfficientNet and
     either:
     - concatenate the view features before the projection (simplest); or
     - give each view its own learned view embedding and let the image GAT
       attend over all views of a window. This finally gives the GAT a
       reason to exist (§3.2).
  3. Add an `image-views` suite with the arms tool, chip, workpiece,
     tool+chip, and all three.
- **Verify.** The suite's accuracy against the main run, with a paired
  fold-level test (F11), and the photo-masked accuracy per view.

### Step 2. Higher-resolution tool photos and a crop of the cutting edge (F9)

- **Why.** Resizing to 384 × 128 removes the wear band (§4.3).
- **How.**
  1. Find the cutting-edge region once (a fixed crop per camera setup, or a
     simple edge detector).
  2. Feed a 384 × 384 crop at native resolution, or raise the input size to
     512 × 1536.
  3. Use the fine-tuned backbone (the best encoder) with a small learning
     rate for the trunk (for example 1e-5) and strong augmentation:
     brightness, contrast, small rotations and crops.
  4. Precompute crops to keep epochs fast.
- **Verify.** Image-only accuracy (force masked) should rise well above
  0.65. Ideally it approaches persistence.

### Step 3. Remove the run-number shortcut (F2)

- **Why.** The run number predicts 71 % of classes on its own, so it is
  learned at the expense of real wear cues. It does not transfer to tools
  used at a different rate.
- **How.** Add `--metadata {all,position,none}` in `config.py` and
  `models.py`, creating the `cycle`/`batch` tables only for `all`. Add a
  run-number baseline to `evaluation.baseline_report`.
- **Expected.** Training accuracy drops, the test accuracy of a sensing
  model becomes honest, and the training-test gap should shrink.
- **Verify.** Compare `metadata-none` with `all` per fold, together with
  the run-number baseline.

### Step 4. Reduce tool-to-tool shift (F16, F3)

- **Why.** Each test tool has its own force level (T1's Fx at +1.15
  standard deviations).
- **How.**
  1. Add reversible instance normalization (RevIN, `--revin`): each input
     window is normalized by its own mean and spread, and the forecast is
     de-normalized afterwards.
  2. Handle the Fz drift: subtract each pass's starting Fz value, or drop
     Fz from the input.
  3. For the photos, apply per-tool colour and brightness normalization.
- **Verify.** The training-test gap per fold (§4.2), and accuracy on T1,
  T2 and T4.

### Step 5. Ordinal, noise-aware and temporally consistent classification (F12)

- **Why.** 88 % of errors are between neighbouring classes. "Used" is a
  noisy band. Predictions flicker in 40 % of windows.
- **How.**
  1. **Ordinal loss.** CORAL or cumulative-link logits (two binary
     outputs: wear above threshold 1, above threshold 2) instead of 3-class
     softmax. Add `--class-loss {ce,coral}`.
  2. **Label smoothing towards neighbours.** For example, 0.1 of the target
     mass moves to the adjacent class only.
  3. **Smoothed wear labels.** Isotonic regression of flank wear per tool,
     then re-threshold, which removes the 29 backward class steps.
  4. **Monotone decoding.** At inference, enforce non-decreasing classes
     over the 96 leads with a cumulative maximum of the ordinal output. Or
     add a small penalty in training for predicted class changes within a
     window.
  5. **Threshold tuning on validation.** Pick the decision thresholds that
     maximize macro F1 on the validation tool, using the good ranking
     (AUC 0.87).
- **Verify.**
  - "Used" recall, which should rise from 0.49;
  - mean absolute class error and quadratic-weighted kappa;
  - the share of windows with flickering predictions.

### Step 6. Make fusion earn its place

- **Why.** The photo alone gives 0.65 and both inputs 0.67-0.68 (§4.6).
- **How.**
  1. Train with modality dropout 0.2 by default. In this study it lifted
     photo-masked accuracy from 0.41 to 0.60 and full accuracy from 0.659
     to 0.681.
  2. Add the raw-force spectrogram stream as a third input. It is the best
     forecasting input (§3.2) and carries leak-free wear information.
- **Verify.** The gain of full over each single input should grow.

### Step 7. Reliable model selection and averaging (F4, F14, F11)

- **Why.** Test tools have 28-51 windows and validation is one tool. Fold
  accuracy varies with a spread of 0.09.
- **How.**
  1. Select the epoch on a smoothed validation loss, or use the class loss
     only for the classification checkpoint.
  2. Train 3-5 seeds and average their probabilities (an ensemble).
  3. Report paired Wilcoxon tests across folds.
- **Expected.** A few points of accuracy from the ensemble, and trustworthy
  comparisons.

### Step 8. Report what matters (F1)

Add accuracy on targets whose label differs from the shown photo, and skill
over persistence. At lag 0, 95 % of targets can be read off the photo, so
gains in actual prognosis are invisible in plain accuracy.

### Suggested order and expected outcome

| Phase | Steps | Effort | Expected effect |
|---|---|---|---|
| A | 5 (threshold tuning, monotone decoding), 6.1, 7.2 | Days, little new code | Higher "used" recall, less flicker, a few points of accuracy |
| B | 2, 3, 4 | 1-2 weeks | A smaller training-test gap; image-only accuracy well above 0.65 |
| C | 1 (chip and workpiece photos), 6.2 | 2-3 weeks including data checks | The largest potential gain: new wear evidence |

## 6. Why our MSE falls with the horizon while the paper's rises

### 6.1 What is observed

Cumulative z-scored MSE over leads 1…h (main run 3a against the paper):

| h | 12 | 24 | 36 | 48 | 60 | 72 | 84 | 96 |
|---|---|---|---|---|---|---|---|---|
| Ours | 1.034 | 0.842 | 0.779 | 0.761 | 0.764 | 0.762 | 0.782 | 0.809 |
| Paper | 0.467 | 0.514 | 0.586 | 0.686 | 0.772 | 0.830 | 0.917 | 0.902 |

A cumulative metric at h averages all leads 1…h. Ours starts high and falls
because the **first leads are the worst**. Later, better leads pull the
average down. In a normal forecaster, error grows with the lead, and the
cumulative curve rises like the paper's.

### 6.2 Error per lead (from the prediction files)

MSE at individual leads, mean over folds:

| Lead | 1 | 3 | 6 | 9 | 12 | 18 | 24 | 36 | 48 | 72 | 84 | 96 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| All channels | 0.95 | 1.02 | **1.40** | 0.95 | 0.68 | 0.63 | 0.68 | 0.66 | 0.73 | 0.80 | 0.97 | 1.01 |
| Fx | 0.58 | 0.74 | 0.88 | 0.61 | 0.48 | 0.50 | 0.51 | 0.48 | 0.58 | 0.57 | 0.70 | 0.73 |
| Fy | 1.12 | 1.08 | **2.00** | 0.88 | 0.52 | 0.41 | 0.57 | 0.42 | 0.42 | 0.56 | 0.91 | 1.19 |
| Fz | 1.16 | 1.23 | 1.32 | 1.36 | 1.03 | 0.98 | 0.98 | 1.07 | 1.18 | 1.28 | 1.29 | 1.12 |

The per-lead error is U-shaped: high at leads 1-9 (a peak of 1.40 at lead
6), low at leads 12-48 (0.63-0.73), and rising again towards lead 96. Two
causes produce the early peak.

### 6.3 Cause 1: the forecast start is phase-locked to pass boundaries

- **The geometry.**
  - A pass lasts 98.5 s at the median, and each pass is a plateau framed by
    entry and exit transients (`EDA.md` §4).
  - Evaluation windows are cut every 96 steps from the start of each tool's
    series, so input and forecast windows have about the length of a pass.
  - The first windows of a tool start at a pass start. Each input window
    therefore tends to end near the end of a pass, and its forecast tends to
    begin with the transition into the next pass.
- **The evidence.** The mean step-to-step change of the *true* force per
  lead is:
  - 0.13-0.16 at leads 1-7;
  - 0.06-0.07 at leads 12-60 (on the plateau);
  - 0.12-0.18 again at leads 80-96 (the next boundary).
  So the hard transients sit at the start and end of the forecast, not in
  the middle.
- **Why this hurts the start most.**
  - A transient is a sharp jump whose timing varies by a few seconds. A
    forecast that places it 2-3 s early or late pays twice: once where it
    predicted the jump, once where the jump happened.
  - The model's own predictions change by 0.15-0.17 per step at these
    leads, as jagged as the targets but not in sync. That is the
    double-penalty pattern. Fy, whose transients are sharpest, peaks at 2.0
    at lead 6.
  - On the plateau (leads 12-60), the force is nearly constant, and the
    model predicts the plateau level well (0.63-0.73).
- **Effect on the last-value baseline.** It suffers from the same jump. But
  a constant forecast pays only for the size of the jump, not twice for its
  timing, so its early error is lower than the model's (0.67 against 1.03
  at h = 12).

### 6.4 Cause 2: two tools with very large early errors

Per test tool, MSE on leads 1-12 against leads 13-36:

| Tool | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 | T10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Leads 1-12 | **2.63** | 0.36 | 0.33 | 0.39 | **3.77** | 0.56 | 0.57 | 0.27 | 0.43 | 1.03 |
| Leads 13-36 | 0.66 | 0.42 | 0.38 | 0.45 | 1.83 | 0.55 | 0.43 | 0.32 | 0.44 | 1.05 |

- **T1 and T5 have early errors 4-10 times those of the other tools.**
  - T1 cuts at a clearly higher force than the training tools (Fx at +1.15
    standard deviations).
  - T5 is the most worn test tool (57 % of its targets are dulled), with
    high forces.
  - Their transients are larger in z-units, and a level outside the
    training range is pulled back towards training-typical values in the
    first forecast steps.
- **Because the table averages folds equally, these two tools dominate
  h = 12.**
- **Without T1 and T5**, cumulative MSE is **0.49 at h = 12 and 0.56 at
  h = 96**. It **rises with the horizon, like the paper's**, and starts
  close to the paper's 0.467.

### 6.5 Why the paper's curve rises

- Error naturally grows with the lead: the further ahead, the less the
  present determines the future. The paper's curve (0.47 → 0.90) is this
  normal shape.
- The paper's time step and windowing are not stated. If its windows are
  not phase-locked to pass boundaries, for example because of a different
  step or a different stride, the transients spread evenly over all leads
  and no early peak appears.
- If its test data does not contain tools as far from the training
  distribution as T1 and T5 (for example with random splits within tools),
  there is no early-lead outlier either.

### 6.6 What would correct it

1. **Report per-lead error and phase-randomized windows.** Start
   evaluation windows at a random offset per tool, or at several offsets.
   This removes the artificial alignment and gives a horizon curve
   comparable to the paper's.
2. **RevIN or a residual path (F16, F19).** Give the heads the current level
   directly, so the first leads start from the last observed value instead
   of a training-typical level. This addresses T1 and T5.
3. **Forecast the change from the last value**, rather than the level. A
   simple re-parameterization of the force head.
4. **Smooth or soft-timed transient prediction.** A model that cannot time a
   transient exactly should predict a smoothed step. An MAE or Huber loss
   discourages sharp, misplaced jumps more than MSE does.

## 7. Conclusions

1. OmniFuser was implemented faithfully: 35 audited items, 146 tests, and 49
   cluster runs. It trains stably on Nonastreda.
2. **Forecasting reaches the paper's level** (MAE 0.595 against 0.597).
   - The falling cumulative MSE is an artefact of two things: evaluation
     windows aligned with pass boundaries, which put transients at the
     forecast start, and two out-of-distribution test tools (T1 and T5).
   - Without T1 and T5, the error rises with the horizon as in the paper.
3. **Classification is not reproduced** (0.67 accuracy and 0.70 precision,
   against the paper's 0.87).
   - The model ranks wear well (AUC 0.87) and errs mostly between
     neighbouring classes.
   - It fails on the "used" band (recall 0.49), on transfer to unseen tools
     (0.88 training against 0.67 test), and in reading wear from downscaled
     tool photos.
4. Of the paper's components, refinement, MRTE, the hybrid gate and ALS help
   classification. The image GAT and C2F do not improve forecasting. Our
   readings of the equations outperform the literal ones.
5. Pretrained image features are essential. Raw-force spectrograms are the
   best forecasting input and a leak-free wear signal.
6. The most promising path to better classification:
   - richer and sharper image evidence: chip and workpiece photos,
     cutting-edge crops at native resolution;
   - removing the run-number shortcut;
   - normalization against tool shift;
   - ordinal, temporally consistent decisions;
   - several seeds.

## 8. Limitations

- **Statistics.** One seed per fold, 28-51 test windows per fold, and no
  significance tests (F11, F14). Differences below about 0.03 in accuracy
  are not reliable.
- **Images.** Only the tool photos of Nonastreda were used.
- **Alignment.** At lag 0 the photo is taken after its pass, so it shows
  the target's class for 95 % of target steps. Accuracy should be read next
  to persistence.
- **The paper's numbers** come from a dataset description and settings that
  are not fully stated, so they are not directly comparable.
- **Scope of the per-lead and per-class analyses.** They come from the main
  run's prediction files; the other runs were compared only on their
  summary metrics.

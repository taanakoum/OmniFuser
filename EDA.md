# Nonastreda exploratory data analysis

This document explains every quantity produced by `scripts/eda.py`. It covers
what the input data is, how force signals and photos are aligned, what a fold
is, and what each metric shows about the task. All numbers come from the real
`Nonastreda_Data/` folder.

Reproduce the outputs with:

```bash
python -m pip install -e ".[eda]"
python scripts/eda.py --data-dir Nonastreda_Data --out experiment_results/eda
```

The script writes these outputs to `experiment_results/eda/`:

| Output | Contents |
|---|---|
| `eda_summary.json` | Every headline number in this document |
| `records.csv` | One row per raw pass (512 rows) |
| `runs.csv` | Class sequence of each run, raw and after deduplication |
| `tools.csv` | Per-tool steps, windows, and alignment metrics |
| `folds.csv` | Per-fold window counts, class shares, and normalization shift |
| `difficulty_by_time_step.csv` | Baseline difficulty at five time steps × eight horizons |
| `autocorrelation.csv` | Autocorrelation per tool, channel, and lag |
| `trim_flags.csv` | Passes whose trimming needs review |
| `figures/01-12*.png` | The figures referenced below |

Forces are assumed to be sampled at 1 kHz (`--fs`). Seconds below are derived
from that rate.

## 1. Input data and terminology

| Term | Meaning in Nonastreda | Count |
|---|---|---|
| **Tool** | One milling cutter, used from new until worn (`T1`…`T10`) | 10 |
| **Run** (`R`) | A cutting session of one tool | 128 runs |
| **Block** (`B`) | One machining pass inside a run. Every run has 4 blocks. | — |
| **Pass / record** | One `T#R#B#` entry: one force recording, one photo, and its labels | 512 raw, 378 after deduplication |
| **Force signal** | 3 channels (Fx, Fy, Fz) | 98.5 s median per pass (52-153 s) |
| **Photo** | One image of the cutting edge per pass | 512 |
| **`image_label`** | Wear class assigned from the photo: sharp, used, dulled. This is the training label. | — |
| **`tool_label`** | A second, coarser 1/2/3 wear stage | — |
| **`flank_wear`** | Measured flank-wear width (µm) | — |

There are no missing or non-finite force values. The labels stored in the MAT
file and in `labels.csv` agree on all 512 passes.

**Figure 01** shows the retained passes per tool and class. After
deduplication the class counts are:

| Class | Retained passes |
|---|---|
| sharp | 147 |
| used | 118 |
| dulled | 113 |

## 2. Labels: how trustworthy, how they evolve

### Flank wear by class

`labels.flank_wear_by_class` and **Figure 03** compare flank wear with the
photo label. The rank correlation is high (Spearman ρ = 0.92), but the classes
overlap:

| Class | Flank wear range (µm) | Median (µm) |
|---|---|---|
| sharp | 22.6-69.9 | 45.6 |
| used | 43.0-121.3 | 88.8 |
| dulled | 70.7-346.7 | 129.5 |

A pass at 60 µm can carry any of the three labels. Labels are the
annotators' visual judgement, not a flank-wear threshold. Perfect
classification is therefore not expected.

### Photo label versus tool label

`labels.image_label_vs_tool_label` crosses the two label columns. They
disagree on 57 of the 378 retained passes (15 %). The pipeline trains on
`image_label`; `--label-column tool_label` is the alternative.

### Label dynamics

`labels.dynamics_raw` and `dynamics_retained` describe how classes change
along each tool's (run, block) order. **Figure 02** shows every pass.

| Metric | Raw (512 passes) | Retained (378 passes) | Meaning |
|---|---|---|---|
| `runs_with_mixed_classes` | 47 of 128 | 43 of 128 | Runs whose blocks carry more than one class |
| `backward_class_flips` | 37 | 29 | Consecutive passes where the class goes *down*, for example used → sharp |
| `forward_class_flips` | 58 | 49 | Consecutive passes where the class goes up |
| `flank_wear_decreases_along_tool_order` | 189 | 138 | Consecutive passes where measured wear decreases |

What this means:

- **The unit with a single class is one pass, not one run.** The loader
  labels every force sample of a pass with that pass's class, and a
  forecast's targets can change class inside the horizon.
- **Wear is not monotone.** Backward flips are about two-thirds as frequent
  as forward ones, and T3, T5, T6 and T8 alternate repeatedly. Some targets are
  therefore annotation noise that no model can predict from the past.

## 3. Duplicate force recordings

`duplicates` and the `retained_after_dedup` column of `records.csv` describe
passes whose force arrays are bit-identical within the same tool and run:

| Metric | Value |
|---|---|
| Duplicate groups | 132 |
| Passes removed | 134 |
| Group pattern B3 + B4 | 127 groups |
| Group pattern B1 + B2 | 4 groups |
| Group pattern B1-B4 | 1 group |
| Groups whose copies carry different photo labels | 24 |

The loader keeps one pass per group: the one with the largest flank wear, so
its label and photo come from the most-worn annotation. Without this step the
same force window would appear twice under two labels.

## 4. Force signals

The force statistics in `force.*` and **Figures 04, 06 and 11** describe the
signals themselves:

| Metric | What it tells you |
|---|---|
| `duration_s` | Pass length. The median is 98.5 s; durations range from 52 to 153 s. |
| `channel_mean`, `channel_std` | Per-pass level and variability per channel. Fx dominates: median mean 123, median std 16. Fy and Fz are small and often negative. |
| `magnitude_sum_per_pass` | Sum of \(\lVert F\rVert\) over a pass; median 1.25×10⁷. |
| `spearman_flank_wear_vs_mean_force` | ρ = 0.70: worn tools cut with higher force (Figure 06). This is the physical link the forecasting task relies on. |

**Figure 11** shows T10's force stream after trimming and 1 s pooling. Each
pass appears as a plateau framed by entry/exit transients.

## 5. Pass trimming

The loader removes each pass's entry and exit transients, the "span rule"
(`omnifuser/data.py::_nonastreda_span_bounds`), in five steps:

1. Compute \(\lVert F\rVert^2\) per sample.
2. Smooth it with a 20-sample moving mean and take the square root. This is
   the **envelope**.
3. Take the median envelope of the first 5 % of the pass. This is the
   **reference level**.
4. Mark samples whose envelope reaches at least 50 % of the reference level
   as active.
5. Keep everything from the first to the last active sample.

The `trimming.*` metrics report what the rule did:

| Metric | Value | Meaning |
|---|---|---|
| `records_trimmed` | 500 / 512 | Passes that lost at least one sample |
| `removed_fraction_of_samples` | 1.5 % | Share of all force samples removed |
| `retained_idle_fraction` (median / max) | 0.18 % / 9.3 % | Share of kept samples below half the pass median ("low-level") |
| `records_with_idle_reference` | 4 | Passes whose 5 % reference window is mostly low-level |
| `residual_idle_lead_seconds` (max) | 7.0 s | Longest low-level start left in place |

### Known limit of the reference window

If a pass starts with more than about 2.5 % idle time, more than half of the
5 % reference window is idle. The reference median then sits at the idle
level, the threshold falls below it, and nothing is trimmed at the start.
`trim_flags.csv` and **Figure 05** list the 4 passes where this occurs. Visual
review sorts them into two groups:

| Pass | Start of the pass | Verdict |
|---|---|---|
| T2R4B1 | Flat ~14 N for 4.0 s, then a step into cutting | Genuine idle lead-in, not trimmed |
| T1R5B1 | 0.8 s spike, then a flat ~22 N plateau until 5.6 s | Genuine idle plateau, not trimmed |
| T7R1B2 | Force ramps from ~13 to ~70 N over the pass, with vibration from t = 0 | Low-force cutting: correctly kept |
| T9R1B2 | Same ramp shape | Low-force cutting: correctly kept |

A fallback that uses the whole-pass median as the reference was tested and
rejected:

| Pass | Effect of the fallback |
|---|---|
| T2R4B1 | Correctly trimmed |
| T1R5B1 | Only 0.6 s of the 5.6 s plateau trimmed |
| T7R1B2 | 7.0 s of real cutting deleted |
| T9R1B2 | 4.8 s of real cutting deleted |

The rule is therefore unchanged. The remaining impact is about 9 s of idle
signal in 2 retained passes, roughly 0.02 % of samples or 9 of 36,903 pooled
steps. "Idle" in these metrics means "below half the pass median", which also
matches slow ramps, so the flag is a review list, not a defect count.
`tests/test_data.py::test_span_trim_keeps_a_lead_in_longer_than_half_the_reference_window`
documents this limit.

## 6. Time step (pooling) and forecast difficulty

The paper never states its time step. The pipeline mean-pools each pass in
blocks of `--pool` samples. `pool=1000` means one step per second.
`difficulty_by_time_step.csv` and **Figure 08** measure how hard forecasting
is at five step sizes, using trivial predictors.

Method:

1. Z-score the data per channel.
2. Cut non-overlapping windows of \(T=96\) input steps followed by 96 target
   steps.
3. Score cumulative error over leads 1…h for each paper horizon h.

Columns:

| Column | Definition |
|---|---|
| `last_value_mse`, `last_value_mae` | Error of repeating the last input value for every lead |
| `zero_forecast_mse` | Error of predicting the mean, which is 0 after z-scoring. It is ≈ 1 by construction. |
| `target_label_change_share` | Share of windows whose target labels differ from the class at the end of the input |
| `persistence_accuracy_lag0` / `_lag1` | Accuracy of copying the shown photo's label into every lead (see §7) |
| `paper_omnifuser_mse`, `paper_omnifuser_accuracy` | OmniFuser's published Table IV/V values on "Mudestreda" |

Last-value MSE at h = 12 / 48 / 96, against the paper's OmniFuser:

| Pool (step) | h = 12 | h = 48 | h = 96 |
|---|---|---|---|
| 1 (1 ms) | 0.019 | 0.024 | 0.039 |
| 10 (10 ms) | 0.028 | 0.059 | 0.085 |
| 100 (0.1 s) | 0.072 | 0.171 | 0.247 |
| 300 (0.3 s) | 0.154 | 0.232 | 0.308 |
| **1000 (1 s)** | **0.580** | **0.650** | **0.862** |
| Paper OmniFuser | 0.467 | 0.686 | 0.902 |

At 1 ms a copy-the-last-value rule is 23-29× better than the paper's best
model, so that cannot be the paper's step. At 1 s the trivial baseline sits
inside the paper's range at every horizon: worse than OmniFuser at short
horizons and comparable at long ones. This is why the training protocol uses
`--pool 1000`. It is an inference from difficulty matching, not a paper
value.

One concern remains: pass-level aggregation (one step per pass) would match
Figs. 1-3's "Iteration" axes and Fig. 2's 10⁶ force scale. Our per-pass force
sum is 4×10⁶-2.6×10⁷. But no tool has 192 passes, so \(T=P=96\) is impossible
at that level.

## 7. Image-to-force alignment and label leakage

Each photo is taken after its pass. The paper replicates each pass's photo
across all of that pass's sensor readings. The pipeline exposes two
alignments through `--image-lag`:

| Mode | Photo shown with pass *p*'s forces | Status |
|---|---|---|
| `0` | Photo of pass *p* (taken after *p*) | Paper reading |
| `1` | Photo of pass *p − 1*. The first pass of a tool keeps its own photo. | Protocol default (causal) |

Targets are always the classes of the passes the horizon falls in, so labels
never move with the lag. Only the photo does.

`persistence_and_alignment` (protocol windows) and **Figure 09** measure how
much the photo gives away:

| Metric | Value | Meaning |
|---|---|---|
| `persistence_accuracy_lag0` | 0.913 | Copying the label of the photo shown at lag 0 is right for 91 % of target steps |
| `persistence_accuracy_lag1` | 0.769 | The same with the previous pass's photo. Measured before 2026-10-21, when a tool's first pass still showed its own photo; it now shows none, which lowers this value slightly. Rerun `eda.py` for the current figure. |
| `eval_windows_target_class_change` | 21 % | Windows whose targets change class relative to the end of the input |
| `eval_windows_input_spans_records` | 95 % | Input windows covering more than one pass, so the GAT sees at least two photos |

At 1 s steps, the lag-0 copy rule reaches 0.92-0.99 cumulative accuracy over
h = 12…96. That is above the paper's OmniFuser (0.82-0.93). With the paper's
alignment, accuracy mostly rewards reading the label off the input photo.

Lag 1 lowers the free accuracy to about 0.80. The model must then actually
forecast wear for about one window in five. The protocol keeps the paper's
lag 0, so its accuracy includes this effect; the experimental lag-1 run in
`TRAINING_RUNBOOK.md` §14 measures it. Each fold JSON reports
`persistence_class_accuracy` as the baseline to beat.

### Per-tool alignment

`tools.csv` gives the same metrics per tool:

| Column | Meaning |
|---|---|
| `steps` | Pooled steps of the tool's stream (2,793-5,081) |
| `train_windows_stride` | Training windows at stride 24 |
| `eval_windows` | Evaluation windows at stride 96 |
| `eval_windows_mean_unique_images_lag1` | Average distinct photos per input window (≈ 1.9) |
| `persistence_accuracy_lag0`, `_lag1` | The copy-the-photo accuracy for that tool |

Copying is hardest on T6 (0.74 / 0.59), whose labels alternate most, and
easiest on T2 (0.98 / 0.93).

## 8. What a fold is

The pipeline evaluates with **10-fold leave-one-tool-out cross-validation**.
Fold *i* has three parts:

- **Test:** tool T*i*. It is never seen during training and is used only for
  the final scores.
- **Validation:** the next tool, cyclically (T*i+1*, and T1 after T10). It
  selects the best epoch.
- **Training:** the remaining 8 tools.

A tool is never split across parts, so no pass, photo or window of a test
tool influences training. The force normalizer (per-channel mean and standard
deviation) is fitted on the 8 training tools only. Each tool is the test tool
exactly once. Reported scores are the mean ± standard deviation over the 10
folds, plus metrics pooled over all held-out windows.

Windows are 96 input steps followed by 96 target steps:

| Split | Stride | Windows per fold |
|---|---|---|
| Training | 24 | 1,127-1,213 |
| Validation and test | 96 (non-overlapping targets) | 28-51 |

Test sets are small. Expect fold-to-fold spread.

`folds.csv` and **Figure 10** describe each fold:

| Column | Meaning |
|---|---|
| `*_windows` | Window count per split |
| `*_target_share_<class>` | Share of target labels per class and split. Test sets differ strongly: dulled is 12 % for T7 and 57 % for T5. Use macro and weighted scores and look at per-fold values, not only the mean. |
| `normalizer_mean_*`, `normalizer_scale_*` | The fold's training-tool statistics |
| `test_z_mean_*`, `test_z_std_*` | The test tool's mean and spread in the training normalization. 0 and 1 would mean no shift. T1's Fx sits at +1.15 standard deviations, so T1 cuts at clearly higher force than the other tools. |

The summary key `normalization_shift` gives these per fold. The median
absolute shift is 0.16-0.32 standard deviations, and some test tools are 2×
wider or narrower than training. Cross-tool generalization is a real part of
the task.

## 9. Autocorrelation

`autocorrelation.csv` and **Figure 07** test the paper's Fig. 3 claim that
force autocorrelation "decays beyond a lag of approximately 17".

- **At 1 s steps** (Fx, per tool), autocorrelation stays high:
  - median 0.75 at lag 17;
  - below 0.5 within 24 lags in only 3 of 30 channel/tool series, all Fy.
- **At pass level** (mean force per pass), it falls below 0.5 after a median
  of 4.5 lags and first turns negative at lags 8-16 (median 14).

Neither level reproduces the paper's figure exactly. The 1 s stream carries
long memory within and across passes, which a 96-step input can exploit.

## 10. Photos

`images.*` and **Figure 12** describe the photos:

| Metric | Value |
|---|---|
| Photo sizes | 88 at about 3100×1000 (full resolution), 424 at about 1550×500 (half resolution); 1550×550 occurs twice and slightly changes the aspect ratio |
| Full-resolution photos by tool | T1: all 40; T2: 48 of 56; no other tool |
| Mean luminance and edge strength | Nearly identical across classes |
| File size | Median 0.19 MB |

**Resolution.** Full resolution is a tool-specific capture setting, spread
over all classes, so it does not leak labels. Resizing to the model's 384×128
input removes it: a full-resolution photo resized directly versus via half
resolution differs by 0.2-0.3 grey levels of 255. No correction is applied.

**Luminance and edge strength.** Median luminance is 0.50-0.52 and edge
strength 0.055-0.060 in every class. Wear is not visible in global image
statistics; the image encoder must find local edge detail.

## 11. Summary of decisions

The EDA supports or settles these pipeline decisions:

| Decision | Evidence |
|---|---|
| `--pool 1000` (1 s steps) | §6: the only tested step whose trivial-baseline difficulty matches the paper's tables |
| `--image-lag 0` (paper), lag 1 as an experiment | §7: the paper's same-pass photo gives away 91-99 % of target labels; the lag-1 run measures how much of the accuracy that explains |
| Deduplicate identical forces | §3: 134 bit-identical copies, 24 groups with conflicting labels |
| Span trimming unchanged | §5: the known limit affects 2 passes; the alternative damages real cutting signal |
| Leave-one-tool-out folds, training-only normalization | §8: tools differ in force level and class mix, so mixing tools would overstate accuracy |
| Labels per pass, macro metrics alongside accuracy | §2: mixed-class runs, backward flips, and overlapping wear ranges cap attainable accuracy |

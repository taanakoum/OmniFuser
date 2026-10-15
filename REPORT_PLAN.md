# Report plan

This is a plan for writing up the OmniFuser experiments after the Lightning
AI runs finish. It lists what to download before stopping the Studio, then
gives a report outline in which every section names its source: an output
file and JSON key, or an existing document whose text can be adapted.

Section 5 contains Python snippets that turn the outputs into tables and
figures. Run them from the repository root, in Python or a notebook, once
the results are downloaded. They need only numpy and matplotlib (the `eda`
extra), and they were tested against real smoke-run outputs of this exact
code.

Text in brackets, such as [epochs], marks values to fill in from your runs.

## 1. Before you stop the Studio

### 1.1 Archive the outputs

Pack the results, then download both archives from the Studio's file
explorer (right-click → **Download**):

```bash
cd /teamspace/studios/this_studio/OmniFuser
tar -czf results-no-checkpoints.tar.gz --exclude='*.pt' experiment_results
find experiment_results -name '*.pt' | tar -czf results-checkpoints.tar.gz -T -
```

The first archive is everything the report needs. The second keeps the
trained models, at about 17 MB per fold. On Windows, unpack with
`tar -xzf results-no-checkpoints.tar.gz` in PowerShell, from the repository
folder, so the files land in `experiment_results/` where the snippets in §5
expect them.

### 1.2 Record the provenance

The commit hash and the hardware log identify exactly what you ran:

```bash
git log -1 --format="%H %s"
cat experiment_results/main/full-model/tracking/run.json
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

`run.json` holds the full configuration, the dataset signature, and the
GPU/library versions (`metadata.runtime`, plus `runtime_history` if a run
resumed on other hardware).

### 1.3 Check that every run completed

- Every suite's `suite.json` shows `"status": "completed"` and each variant
  `completed`.
- Every 10-fold folder has `cross-validation-results.json` and ten
  `fold-T*.json` files.

`cross-validation-progress.json` exists from the first finished fold, so its
presence alone does not mean a run is complete.

## 2. Report outline and sources

| # | Report section | What it should say | Sources |
|---|---|---|---|
| 1 | Abstract | Task, data, the main fold mean ± std, the headline ablation and robustness findings, and that this is an adaptation to Nonastreda, not a score reproduction | §3 tables below |
| 2 | Introduction | Predictive maintenance; the two tasks (96-step force forecast, per-step wear class); OmniFuser's idea (C2F fusion with recursive refinement) | The paper's §I and §V-A; `README.md` "How it works" |
| 3 | Data | Nonastreda contents, label quality, duplicates, trimming, the time step, the photo alignment and its leakage, folds | `EDA.md` §1-10 (text and numbers ready to adapt); `experiment_results/eda/figures/01-12`; `eda_summary.json` |
| 4 | Method | Architecture blocks with equations and tensor shapes; the heads; the loss | `IMPLEMENTATION_AUDIT.md` §2 and §4 (equations ready); `scripts/model_summary.py` output for the parameter table |
| 5 | Fidelity to the paper | 35 items: 26 match, 4 deviate with flags, 2 forced, 2 not verifiable, 1 not implemented; the paper's own inconsistencies | `PAPER_IMPLEMENTATION_ALIGNMENT.md` "Paper fidelity review" (tally and tables ready) |
| 6 | Experimental protocol | 10-fold leave-one-tool-out, T=P=96, stride 24/96, 1 s pooling, lag-1 photos, deduplication, train-only normalization, Adam 1e-4, batch 32, [epochs], best-validation checkpoint, seeds, hardware | `TRAINING_RUNBOOK.md` §9; each run's `tracking/run.json`; `README.md` defaults table |
| 7 | Main results | Fold mean ± std; pooled out-of-fold metrics; per-horizon results against the persistence baselines and the paper's Table IV/V; confusion matrix; ROC; learning curves; per-fold spread | `experiment_results/main/full-model/`: snippets A-E |
| 8 | Component ablations | The Fig. 13 analogue: z-scored MSE per variant against the full model, plus classification F1. Say which differences exceed the fold spread. | `experiment_results/ablations/<run-id>/suite-summary.csv`: snippets F and H |
| 9 | Missing modalities | Full versus sensor-masked versus image-masked inputs, with and without training dropout; out-of-fold per mode | `experiment_results/missing-modality/<run-id>/`: snippet G |
| 10 | Parameter sensitivity | The Figs. 7-8 analogue: one line per axis, test MSE versus value | `experiment_results/sensitivity/<run-id>/suite-summary.csv`: snippet F |
| 11 | Optional studies | Printed-paper arms (lag 0, repeated head, Eq. 3 literal, Eq. 10 unnormalized); duplicate handling (trained with/without duplicates, scored on the same test windows); the full protocol with duplicates kept in training and test (step 3b; its test windows differ from 3a's, so report it separately) | `experiment_results/paper-literal/*`, `experiment_results/deduplication/<run-id>/`, `experiment_results/main/full-model-no-dedup/` (snippets A-E with `RUN` changed): snippets A, F and H |
| 12 | Compute | Parameters (4,183,600); FLOPs (~7-15×10¹⁰ per batch-32 step versus the paper's ~10¹¹); measured epoch time and peak GPU memory; total GPU hours | `IMPLEMENTATION_AUDIT.md` §4.9; snippet I |
| 13 | Discussion and limitations | Dataset mismatch with the paper; label noise and conflicting duplicates; the photo-lag choice; small test folds (28-51 windows); not reproduced: the comparison models, the image-frequency control, the FLOPs study | `EDA.md` §2, §7, §8; `PAPER_IMPLEMENTATION_ALIGNMENT.md`; `TRAINING_RUNBOOK.md` §19 |
| 14 | Reproducibility appendix | Commit hash, exact commands, seeds, versions, data signature, how to resume | `TRAINING_RUNBOOK.md`; `run.json`; §1.2 above |

## 3. Tables and figures checklist

| ID | Item | Built from | Snippet |
|---|---|---|---|
| T1 | Headline metrics, fold mean ± std, plus pooled out-of-fold | `cross-validation-results.json` → `cv_mean_std`, `out_of_fold` | A |
| T2 | Per-horizon z-scored MSE/MAE and accuracy against the baselines and the paper | `paper_horizons_mean_std`, or `scripts/compare_with_paper.py --json` | B |
| T3 | Confusion matrix and per-class precision/recall/F1 (pooled) | `out_of_fold.classification` | C |
| T4 | Missing-modality results per mode, test split | `modality_evaluations`, `out_of_fold_by_mode` | G |
| T5 | Ablation table (MSE, MAE, accuracy, F1, AUC per variant) | `suite-summary.csv` | F |
| T6 | Sensitivity table, one row per axis value | `suite-summary.csv` | F |
| T7 | Paired fold-level comparisons (full model versus each variant) | `cross-validation-results.json` of two runs → `cv_mean_std.metrics.<m>.per_fold` | H |
| T8 | Compute and runtime | `tracking/metrics.csv` (`epoch_seconds`, `cuda_peak_allocated_bytes`), `run.json` | I |
| F1-F12 | Data figures | `experiment_results/eda/figures/` | (ready) |
| F13 | Mean ROC per class and macro | `cross-validation-mean-roc.npz` | D |
| F14 | Learning curves: train and validation loss per epoch, one line per fold | `fold-T*.json` → `history` | E |
| F15 | Ablation bar chart (the Fig. 13 analogue) | `suite-summary.csv` | F |
| F16 | Sensitivity line plots (the Figs. 7-8 analogue) | `suite-summary.csv` | F |
| F17 | Accuracy and MSE versus horizon, model against baselines | `paper_horizons_mean_std` | B |

## 4. Reporting rules

- **Two summaries, two meanings.** The fold mean ± std (`cv_mean_std`)
  shows variation across tools. The pooled out-of-fold metric
  (`out_of_fold`) weights every test window equally. Report both, and say
  which one each table uses.
- **Always show the baselines.** Put the persistence and last-value
  baselines next to every per-horizon result. At 1 s steps the last-value
  forecast already reaches MSE 0.58-0.86, and copying the shown photo's
  label 0.80 accuracy (`EDA.md` §6-7).
- **Units.** Use z-scored MSE/MAE (`*_normalized`) when comparing with the
  paper's tables, and force units (N) for physical interpretation. Label
  every table with its unit.
- **Significance with 10 folds.** Use a paired test across the same 10 folds
  (Wilcoxon signed-rank, snippet H) and report the number of folds where
  each arm wins. With n = 10, describe small differences as inconclusive.
- **Macro and weighted scores.** Test class shares vary strongly by tool
  (dulled 12-57 %, `EDA.md` §8), so report macro F1 alongside accuracy.
- **Wording about the paper.** This is "an implementation of OmniFuser
  adapted to Nonastreda", never "a reproduction of the paper's results".
  Name the deviations and the photo-lag choice wherever the paper's numbers
  appear.

## 5. Extraction snippets

Set the paths at the top of each snippet. Every snippet prints a Markdown
table, which can be pasted into a report, or writes a PNG to
`experiment_results/report/`.

### A. Headline metrics (T1)

```python
import json
RUN = "experiment_results/main/full-model"
r = json.load(open(f"{RUN}/cross-validation-results.json"))
print("| Metric | Fold mean ± std | Pooled out-of-fold |\n|---|---|---|")
pooled = {**{f"classification_{k}": v for k, v in r["out_of_fold"]["classification"].items()},
          **{f"force_{k}": v for k, v in r["out_of_fold"]["regression"]["overall"].items()},
          **{f"force_{k}_normalized": v for k, v in r["out_of_fold"]["regression"].get("normalized", {}).items()}}
for name, v in r["cv_mean_std"]["metrics"].items():
    if v["mean"] is None:
        continue
    p = pooled.get(name)
    print(f"| {name} | {v['mean']:.3f} ± {v['std']:.3f} | {p:.3f} |" if isinstance(p, float) else f"| {name} | {v['mean']:.3f} ± {v['std']:.3f} | |")
```

### B. Per-horizon table and figure (T2, F17)

```python
import json, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from omnifuser.evaluation import PAPER_HORIZONS, PAPER_OMNIFUSER_RESULTS
RUN = "experiment_results/main/full-model"
r = json.load(open(f"{RUN}/cross-validation-results.json"))["paper_horizons_mean_std"]
hs = sorted(r["models"], key=int)
print("| h | MSE | last-value MSE | paper MSE | Acc | persistence Acc | paper Acc |\n|---|---|---|---|---|---|---|")
for h in hs:
    m, b, i = r["models"][h], r["baselines"].get(h, {}), PAPER_HORIZONS.index(int(h))
    print(f"| {h} | {m['mse']['mean']:.3f} ± {m['mse']['std']:.3f} | {b['last_value_mse']['mean']:.3f} | "
          f"{PAPER_OMNIFUSER_RESULTS['mse'][i]:.3f} | {m['accuracy']['mean']:.3f} | "
          f"{b['persistence_class_accuracy']['mean']:.3f} | {PAPER_OMNIFUSER_RESULTS['accuracy'][i]:.3f} |")
os.makedirs("experiment_results/report", exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
x = [int(h) for h in hs]
for axis, key, base, paper, label in ((axes[0], "mse", "last_value_mse", "mse", "z-scored MSE"),
                                      (axes[1], "accuracy", "persistence_class_accuracy", "accuracy", "accuracy")):
    axis.plot(x, [r["models"][h][key]["mean"] for h in hs], marker="o", label="this model")
    axis.plot(x, [r["baselines"][h][base]["mean"] for h in hs], marker="o", label="baseline")
    axis.plot(PAPER_HORIZONS, PAPER_OMNIFUSER_RESULTS[paper], ls="--", marker="s", color="k", label="paper")
    axis.set_xlabel("horizon h (steps)"); axis.set_ylabel(label)
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig("experiment_results/report/F17_horizons.png", dpi=150)
```

### C. Confusion matrix and per-class scores (T3)

```python
import json
RUN = "experiment_results/main/full-model"
c = json.load(open(f"{RUN}/cross-validation-results.json"))["out_of_fold"]["classification"]
names = ["sharp", "used", "dulled"]
print("| true \\ predicted | " + " | ".join(names) + " |\n|---|---|---|---|")
for name, row in zip(names, c["confusion_matrix"]):
    print(f"| {name} | " + " | ".join(str(v) for v in row) + " |")
print("\n| Class | Precision | Recall | F1 | Support |\n|---|---|---|---|---|")
for name in names:
    s = c["per_class"][name]
    print(f"| {name} | {s['precision']:.3f} | {s['recall']:.3f} | {s['f1']:.3f} | {s['support']} |")
```

### D. Mean ROC curves (F13)

```python
import os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
RUN = "experiment_results/main/full-model"
os.makedirs("experiment_results/report", exist_ok=True)
with np.load(f"{RUN}/cross-validation-mean-roc.npz") as roc:
    fig, axis = plt.subplots(figsize=(4.5, 4.5))
    for key in [k for k in roc.files if k.endswith("_mean_tpr")]:   # classes absent from every fold are skipped
        axis.plot(roc["mean_fpr"], roc[key], label=key.replace("_mean_tpr", ""))
axis.plot([0, 1], [0, 1], ls=":", color="grey"); axis.set_xlabel("false positive rate"); axis.set_ylabel("true positive rate")
axis.legend(frameon=False); fig.tight_layout(); fig.savefig("experiment_results/report/F13_roc.png", dpi=150)
```

### E. Learning curves (F14)

```python
import glob, json, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
RUN = "experiment_results/main/full-model"
os.makedirs("experiment_results/report", exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for path in sorted(glob.glob(f"{RUN}/fold-T*.json")):
    fold = json.load(open(path))
    epochs = [h["epoch"] for h in fold["history"]]
    for axis, split in zip(axes, ("train", "validation")):
        axis.plot(epochs, [h[split]["loss"] for h in fold["history"]], alpha=0.6, label=fold["test_tool"])
for axis, split in zip(axes, ("train", "validation")):
    axis.set_title(f"{split} loss (force MSE + class CE)"); axis.set_xlabel("epoch")
axes[1].legend(frameon=False, fontsize=7, ncol=2); fig.tight_layout()
fig.savefig("experiment_results/report/F14_learning_curves.png", dpi=150)
```

Also record each fold's `selected_epoch`. If many folds select the last
epoch, more epochs may help; if most select an early one, the model
overfits.

### F. Suite tables and figures (T5, T6, F15, F16)

```python
import csv, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SUITE = "experiment_results/ablations/paper-v1"   # or the sensitivity, missing-modality or deduplication suite
rows = list(csv.DictReader(open(f"{SUITE}/suite-summary.csv")))
cols = ["full.regression_mse_normalized", "full.regression_mae_normalized", "full.classification_accuracy",
        "full.classification_f1_macro", "full.classification_roc_auc_macro"]
print("| Variant | " + " | ".join(c.split(".")[1] for c in cols) + " |\n|" + "---|" * (len(cols) + 1))
for row in rows:
    cells = [f"{float(row[c + '.mean']):.3f} ± {float(row[c + '.std']):.3f}" if row.get(c + ".mean") else "n/a" for c in cols]
    print(f"| {row['variant']} | " + " | ".join(cells) + " |")
os.makedirs("experiment_results/report", exist_ok=True)
done = [r for r in rows if r.get("full.regression_mse_normalized.mean")]
if "sensitivity" not in SUITE:   # F15: bar chart per variant
    fig, axis = plt.subplots(figsize=(8, 3.6))
    axis.bar([r["variant"] for r in done], [float(r["full.regression_mse_normalized.mean"]) for r in done],
             yerr=[float(r["full.regression_mse_normalized.std"]) for r in done], capsize=3)
    axis.set_ylabel("test MSE (z-scored)"); plt.xticks(rotation=30, ha="right"); fig.tight_layout()
    fig.savefig("experiment_results/report/F15_suite_mse.png", dpi=150)
else:   # F16: one panel per sensitivity axis, the baseline value included
    base = next(r for r in done if r["variant"] == "sensitivity-baseline")
    defaults = {"landmarks": 4, "model-width": 32, "gat-heads": 4, "rtd-rank": 8, "gat-layers": 4,
                "rtd-stride": 3, "mrte-layers": 4, "refinement-rounds": 2}
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.5))
    for axis, (name, default) in zip(axes.flat, defaults.items()):
        points = {default: float(base["full.regression_mse_normalized.mean"])}
        points.update({int(r["variant"].rsplit("-", 1)[1]): float(r["full.regression_mse_normalized.mean"])
                       for r in done if r["variant"].rsplit("-", 1)[0] == name})
        xs = sorted(points); axis.plot(xs, [points[x] for x in xs], marker="o"); axis.set_title(name); axis.set_xticks(xs)
    fig.supylabel("test MSE (z-scored)"); fig.tight_layout()
    fig.savefig("experiment_results/report/F16_sensitivity.png", dpi=150)
```

### G. Missing-modality table (T4)

```python
import json
SUITE = "experiment_results/missing-modality/paper-v1"
for variant in ("no-modality-dropout", "modality-dropout-0-2"):
    r = json.load(open(f"{SUITE}/missing-modality-robustness/{variant}/cross-validation-results.json"))
    print(f"\n{variant}\n| Input | Accuracy | Macro F1 | MSE (z) | MAE (z) |\n|---|---|---|---|---|")
    for mode in ("full", "sensor_missing", "image_missing"):
        t = r["modality_evaluations"][mode]["test"]
        print(f"| {mode} | {t['classification_accuracy']['mean']:.3f} | {t['classification_f1_macro']['mean']:.3f} | "
              f"{t['regression_mse_normalized']['mean']:.3f} | {t['regression_mae_normalized']['mean']:.3f} |")
```

### H. Paired fold-level comparison (T7)

```python
import json
from scipy.stats import wilcoxon
A = "experiment_results/ablations/paper-v1/paper-ablation/full-model"
B = "experiment_results/ablations/paper-v1/paper-ablation/without-c2f"
METRIC = "force_mse_normalized"            # any key of cv_mean_std.metrics
a = json.load(open(f"{A}/cross-validation-results.json"))["cv_mean_std"]["metrics"][METRIC]["per_fold"]
b = json.load(open(f"{B}/cross-validation-results.json"))["cv_mean_std"]["metrics"][METRIC]["per_fold"]
pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
wins = sum(x < y for x, y in pairs)        # for MSE lower is better; reverse for accuracy or F1
diffs = [x - y for x, y in pairs]
p = wilcoxon(diffs).pvalue if any(diffs) else 1.0
print(f"{METRIC}: A better on {wins}/{len(pairs)} folds, mean difference {sum(diffs) / len(diffs):+.4f}, Wilcoxon p = {p:.3f}")
```

### I. Compute and runtime (T8)

```python
import csv, json, statistics
RUN = "experiment_results/main/full-model"
rows = list(csv.DictReader(open(f"{RUN}/tracking/metrics.csv")))
def values(metric):
    return [float(r["value"]) for r in rows if r["split"] == "train" and r["evaluation_mode"] == "" and r["metric"] == metric]
seconds, memory = values("epoch_seconds"), values("cuda_peak_allocated_bytes")
print(f"epochs logged: {len(seconds)}, median epoch {statistics.median(seconds):.1f} s, total training {sum(seconds) / 3600:.2f} h")
if memory:
    print(f"peak GPU memory: {max(memory) / 1e9:.2f} GB")
print("hardware:", json.load(open(f"{RUN}/tracking/run.json"))["metadata"]["runtime"])
```

## 6. Order of work after training

1. Download the archives and record provenance (§1).
2. Run snippets A-E on the main run, then F and G on the suites, then H for
   each ablation variant against the full model.
3. Fill the outline (§2) section by section, adapting text from `EDA.md`,
   `IMPLEMENTATION_AUDIT.md` and `PAPER_IMPLEMENTATION_ALIGNMENT.md`.
4. Write the discussion from the per-fold spread, the baseline gaps and the
   ablation significance results, applying the rules in §4.
5. Finish with the reproducibility appendix: commit hash, the `PROTOCOL`
   arguments (runbook §9), suite run IDs, and `run.json` excerpts.

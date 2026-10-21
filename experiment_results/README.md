# Experiment results

All training and evaluation output goes here, separate from the source code
and the input data. Git ignores everything in this folder except this file.
Back the folder up (Lightning AI persistent storage, or a download) when a
study finishes. The commands that write each subfolder are in
[`../TRAINING_RUNBOOK.md`](../TRAINING_RUNBOOK.md).

```text
experiment_results/
  logs/                         console logs of each launched command
  eda/                          exploratory data analysis (scripts/eda.py; see ../EDA.md)
  report/                       report tables and figures (snippets in ../REPORT_PLAN.md)
  validation/                   smoke tests (capped windows; not results)
  pilot/                        uncapped one-fold timing runs
  main/full-model/              step 3a: main 10-fold run, duplicates merged
  main/full-model-no-dedup/     step 3b: the same with duplicates kept (train and test)
  image-encoder/<run-id>/       step 7: from scratch, pretrained fine-tuned, pretrained frozen
  image-source/<run-id>/        optional: leak-free images: previous-pass photo vs input-window force spectrograms
  heads/<head>/                 optional prediction-head comparison
  ablations/<run-id>/           paper Fig. 13 component ablations
  missing-modality/<run-id>/    modality-dropout robustness study
  sensitivity/<run-id>/         paper Figs. 7-8 parameter sweeps
  all/<run-id>/                 the three studies in one suite
```

A suite folder (`<study>/<run-id>/`) contains:

- `suite.json`: shared settings, interpretation notes, and each variant's
  status, changed settings, and result path.
- `suite-summary.csv`: one row per variant, with the fold mean/std of the test
  metrics for the `full`, `sensor_missing`, and `image_missing` input modes.
  Start here when comparing variants.
- `<group>/<variant>/`: one complete 10-fold run.

Each 10-fold run folder contains:

| File | Contents |
|---|---|
| `tracking/run.json` | Configuration, dataset signature, runtime and runtime history |
| `tracking/metrics.jsonl` | Every logged event, with full nested metrics |
| `tracking/metrics.csv` | Long-form numeric metrics: fold, epoch, split, evaluation_mode, metric, value |
| `cross-validation-results.json` | Fold mean/std, paper-horizon summaries, per-mode summaries, pooled out-of-fold metrics |
| `cross-validation-mean-roc[-<mode>].npz` | Fold-averaged ROC curves |
| `fold-Tn.json` | Per-fold settings, epoch history, test and missing-modality metrics |
| `fold-Tn.pt` | Selected checkpoint with config and normalizer (large) |
| `fold-Tn-predictions[-<mode>].npz` | Test forecasts, targets, and class probabilities |
| `fold-Tn-roc[-<mode>].npz` | Per-class ROC coordinates |

When disk space is short, delete the `.pt` checkpoints and prediction `.npz`
files first. Keep `suite.json`, `suite-summary.csv`, the `tracking/` folders,
and every `*.json` result. Note that deleting a fold's artifacts makes that
fold retrain if the run is resumed.

# Experiment results

All training and evaluation output goes here, separate from the source code
and the input data. Git ignores everything in this folder except this file.
Back the folder up (Lightning AI persistent storage, or a download) when a
study finishes. The commands that write each subfolder are in
[`../TRAINING_RUNBOOK.md`](../TRAINING_RUNBOOK.md),
[`../TRAINING_RUNBOOK_RUCHE.md`](../TRAINING_RUNBOOK_RUCHE.md) and
[`../TRAINING_RUNBOOK_RUCHE_FIXES.md`](../TRAINING_RUNBOOK_RUCHE_FIXES.md).

```text
experiment_results/
  logs/                         console logs of each launched command
  eda/                          exploratory data analysis (scripts/eda.py; see ../EDA.md)
  report/                       report tables and figures
  validation/                   smoke tests (capped windows; not results)
  pilot/                        uncapped one-fold timing runs
  main/full-model/              step 3a: main 10-fold run, duplicates merged
  main/full-model-no-dedup/     step 3b: the same with duplicates kept (train and test)
  image-encoder/<run-id>/       step 7: from scratch, pretrained fine-tuned, pretrained frozen (with and without the image GAT)
  image-source/<run-id>/        optional: leak-free images: previous-pass photo vs input-window force spectrograms
  heads/<head>/                 optional prediction-head comparison
  ablations/<run-id>/           paper Fig. 13 component ablations
  missing-modality/<run-id>/    modality-dropout robustness study
  sensitivity/<run-id>/         paper Figs. 7-8 parameter sweeps
  all/<run-id>/                 the three studies in one suite
  deduplication/<run-id>/       duplicates merged vs kept in training, same test windows
  paper-literal/<variant>/      literal readings of Eq. 3, Eq. 10 and the repeated head
  end_to_end/                   numbers of END_TO_END_SIM.md (scripts/end_to_end_trace.py)
  fixes/s<seed>/                classification-fix study, one suite per seed (fixes/<arm>/)
  decision/                     decision-rule study (scripts/decision_study.py)
  ensembles/                    probability averages of runs (scripts/ensemble_runs.py)
  photo-probe/                  frozen-feature photo probe (scripts/photo_probe.py)
  diagnostics/                  level offset, run-number baseline, photo-change accuracy (scripts/window_diagnostics.py)
```

A run folder can also hold `phase-study/` (re-scored forecasts at other
strides, `scripts/phase_study.py`) and `decision/` (validation-tool
predictions for the class offsets).

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
| `fold-Tn.pt` | The fold's trained model (selected weights, settings, normalizer, data settings; about 16 MB). Reuse with `omnifuser predict --model`. |
| `fold-Tn-predictions[-<mode>].npz` | Test forecasts, targets, and class probabilities |
| `fold-Tn-roc[-<mode>].npz` | Per-class ROC coordinates |

When disk space is short, delete the `.pt` checkpoints and prediction `.npz`
files first. Keep `suite.json`, `suite-summary.csv`, the `tracking/` folders,
and every `*.json` result. Note that deleting a fold's artifacts makes that
fold retrain if the run is resumed.

# OmniFuser classification fixes: Ruche runbook

This runbook trains and evaluates the fixes that need the cluster. It
continues [`TRAINING_RUNBOOK_RUCHE.md`](TRAINING_RUNBOOK_RUCHE.md): the
account, the storage, the conda environment, the data upload and the
`slurm/` scripts of its Part A are reused as they are. Read that runbook
first if you have not used Ruche before.

The fixes, the local validations that chose them, and their results are
recorded in [`FIXES_AND_VALIDATION.md`](FIXES_AND_VALIDATION.md). This
document explains **what to run, in which order, and why each design choice
was made**, using what the 51 finished training runs showed.

**Contents**

1. [What the finished runs taught us](#1-what-the-finished-runs-taught-us)
2. [The design and why](#2-the-design-and-why)
3. [The arms of stage 1](#3-the-arms-of-stage-1)
4. [Decision rules, fixed before the runs](#4-decision-rules-fixed-before-the-runs)
5. [Step 1: update the code and check it](#5-step-1-update-the-code-and-check-it)
6. [Step 2: smoke test of the new options](#6-step-2-smoke-test-of-the-new-options)
7. [Step 3: evaluations of the saved models (no training)](#7-step-3-evaluations-of-the-saved-models-no-training)
8. [Step 4: stage 1, the single-change arms](#8-step-4-stage-1-the-single-change-arms)
9. [Step 5: analyse stage 1](#9-step-5-analyse-stage-1)
10. [Step 6: stage 2, the combined model](#10-step-6-stage-2-the-combined-model)
11. [Step 7: download](#11-step-7-download)
12. [What this runbook does not cover, and why](#12-what-this-runbook-does-not-cover-and-why)
13. [Troubleshooting](#13-troubleshooting)
14. [Cheat sheet](#14-cheat-sheet)

## 1. What the finished runs taught us

Each finding below shapes a design choice in §2. The numbers come from
`FINAL_REPORT_REPRODUCTION.md` and the local validations in
`FIXES_AND_VALIDATION.md`.

| # | Finding | Evidence | Consequence for the design |
|---|---|---|---|
| I1 | **Single runs are noisy.** Seven finished runs have the main configuration with identical settings and seed; they differ only by GPU non-determinism. | Accuracy 0.659-0.676 (mean 0.671, SD 0.006). The modality-dropout gain reported earlier (+0.018) was measured against the lowest of the seven. | Several seeds per arm, compared with a reference trained on the same seeds; paired tests over the folds |
| I2 | **Averaging runs helps a little, consistently.** | The mean of the seven runs' probabilities: accuracy +0.006, macro F1 +0.006 and "used" recall +0.011 over the average single run, better on every fold that changed | Report the seed ensemble of every arm |
| I3 | **The gap is between tools, not training.** | Training accuracy 0.88 against 0.67 on the held-out tool | Fixes target the tool shift: shortcut removal (metadata), level normalization (RevIN) |
| I4 | **The run number is a shortcut.** | On the main run's test windows, predicting each step's class from its run number alone scores 0.708, more than the model's 0.673; the model's fold accuracy does not follow it (correlation −0.36) | Arms without the run code (`--metadata position`, `none`) |
| I5 | **Most of the forecast error is a level offset.** | 69 % of the main run's z-scored MSE is a constant error per window and channel (52-76 % per fold). Anchoring at the input window's level (what RevIN does) cuts T1's level error by 79 % (0.89 → 0.19) and T6's by 76 %, but is worse on T4, T5 and T10, where the level changes between input and forecast | RevIN arm (`--revin`) |
| I6 | **Errors are ordinal and "used" is missed.** | 88 % of errors are between neighbouring classes; "used" recall 0.48 | Ordinal head (`--class-loss coral`); the post-hoc ordinal + monotone rule is the bar it must beat |
| I7 | **The photo carries the classification but is poorly read.** | Photo masked: 0.43; force masked: 0.65; copying the photo's label: 0.95 | A linear probe of frozen features: higher resolution does not help (whole photo 0.636 at 128 × 384, 0.608 at 512 × 1536); a crop of the cutting edge does a little (0.652 at 128 × 640). Crop arm; fine-tuning is the larger lever (I8) |
| I8 | **Two single-run results stand out.** | Fine-tuned encoder 0.693 and frozen without image GAT 0.693, about 3.5 replicate SDs above the mean of the seven | Repeat both over seeds before relying on them |
| I9 | **Runs are cheap.** | A 10-fold run with the frozen encoder takes about 16 minutes on an A100; the fine-tuned encoder about 30 minutes | 5 seeds of every arm fit in a few hours of wall time |
| I10 | **The falling MSE curve is probably an evaluation artefact.** | The evaluation stride (96) is about one pass long, so forecasts start at pass transients | Re-score the saved models at stride 24 (§7) before any change to training |
| I11 | **Where the photo cannot tell, the model is weak.** | 8.7 % of target steps have a label that differs from the shown photo's; the model gets 0.43 of them right (0.70 of the others) | Report this accuracy for every arm: it measures forecasting a change |

## 2. The design and why

| Choice | Alternatives considered | Why this one |
|---|---|---|
| **Two stages.** Stage 1 changes one thing per arm; stage 2 combines the arms that pass. | A full factorial of all fixes (2^8 = 256 configurations); one combined run only | One change per arm attributes each effect. A factorial is unaffordable and has too little data for interactions. A combined run alone cannot tell which part helped or hurt. Stage 2 tests the combination once, with its own seeds. |
| **Reference = the main runs' protocol exactly** (`slurm/protocol.sh`, duplicates merged). | Reference with modality dropout 0.2 (as the earlier recommendation said) | I1: the dropout gain is within run-to-run noise, so it is not established. It becomes an arm, and the reference stays comparable with the 51 finished runs and the seven replicates. |
| **5 seeds: 42, 142, 242, 342, 442.** | 3 seeds; consecutive seeds | I1 and I9: noise of about 0.006 per run (more with different seeds) against effects of 0.01-0.02. Averaging 5 seeds per arm reduces it by √5, at small cost. Fold *i* trains with seed + *i*, so consecutive seeds would reuse seeds across folds; steps of 100 avoid any overlap. Use 3 seeds (42, 142, 242) if your GPU quota is short. |
| **Paired comparison per fold, on seed-averaged metrics, with a Wilcoxon test.** | Comparing fold-mean accuracies; t-tests | The folds differ much more than the arms (test accuracy 0.53-0.81 per tool), so pairing by fold removes most of the variance. Ten paired values are too few for a t-test's normality assumption. |
| **Same data, windows and selection as the main runs** (dedup True, lag 0, stride 24, 30 epochs, best validation loss). | Changing several things at once | Any difference must come from the fix. The earlier report's open questions (lag 0 shows the photo of the forecast pass) are kept as they are and reported with the new photo-change accuracy (§4). |
| **RevIN without feeding the window statistics back.** | RevIN plus the window mean and spread as extra inputs | Feeding the level back would bring the tool's force level, the shift RevIN removes, back into the model. The cost is that the force stream loses the absolute force level, which correlates with wear (ρ = 0.70). The photo carries most of the class information (I7), and the accuracy non-inferiority rule (§4) protects against a loss. |
| **CORAL with ordered thresholds per lead.** | Cumulative-link likelihood; label smoothing towards neighbours; ordinal loss on the 3-class softmax | CORAL's binary tasks are simple and match the two physical thresholds (sharp→used, used→dulled). Ordered thresholds (a first threshold plus positive gaps) make the probabilities valid by construction. It returns log-probabilities, so every metric and decision rule works unchanged. |
| **Metadata `position` and `none` as two arms.** | Only `none` | The run number (and the block) is the shortcut. The position within the pass is not: it tells the model where the pass transients are, which may matter for the forecast (I10). `position` removes the shortcut and keeps the phase; `none` tests whether any code helps. |
| **Photo crop at 128 × 640 with the frozen encoder in stage 1.** | 256 × 1280 or 512 × 1536 (near native resolution); fine-tuning at once; augmentation | The probe (I7) showed that frozen ImageNet features gain nothing from more pixels (their filters were trained on objects at another scale, and the global average dilutes a thin band), while removing the blurred background helps. 128 × 640 was the best of 5 settings at all three regularization strengths. Choosing it among 5 on the probe is a small selection, so stage 1 is its real test. Frozen features are computed once per fold, so the arm isolates the evidence effect. Fine-tuning on the crops is a second change, tested in stage 2 if both pass. Augmentation is not implemented: cached frozen features cannot be augmented. |
| **The fine-tuned and no-GAT arms repeated.** | Trusting the single-seed results | I8: both are about 0.02 above the mean of the seven, but one seed is weak evidence (I1). |
| **Post-hoc decision rules on every arm.** | Argmax only | They are free (saved predictions) and validated on 51 runs (`FIXES_AND_VALIDATION.md` §3). Every arm is reported with argmax and with the ordinal + monotone rule. |

## 3. The arms of stage 1

`scripts/run_experiment_suite.py --study fixes` trains these 9 arms, one
after another, each a full 10-fold run. Run it once per seed (§8).

| Arm (folder) | Change from the main runs | Flag | Hypothesis | Primary metric |
|---|---|---|---|---|
| `reference` | none | | | |
| `modality-dropout-0-2` | a stream zeroed at random during training (rate 0.2) | `--modality-dropout 0.2` | The force stream learns to classify; more robust to a missing photo | accuracy; photo-masked accuracy |
| `metadata-position` | run and block codes removed | `--metadata position` | Removing the shortcut narrows the training-test gap without losing accuracy | training-test gap, accuracy |
| `metadata-none` | all three codes removed | `--metadata none` | as above, and tests whether the position code helps | training-test gap, accuracy |
| `revin` | each input window normalized by its own level and spread | `--revin` | Removes the tool-specific force level (I5): lower MSE, especially on T1, T6 and T2; a risk on T4, T5 and T10 | force MSE (z-scored) |
| `coral` | ordinal head and loss | `--class-loss coral` | Higher "used" recall and fewer far errors than cross-entropy, also better than the post-hoc rule | "used" recall, accuracy |
| `photo-edge-crop` | photo rows 0.30-0.85 (the cutting edge, without the blurred background) at 128 × 640, instead of the whole photo at 128 × 384 | `--image-crop 0.30 0 0.85 1 --image-size 128 640` | The band around the edge, at 2.1× instead of 3.9× reduction, gives the encoder more wear evidence (I7) | accuracy; force-masked accuracy |
| `no-image-gat` | the paper's image GAT removed | `--no-image-gat` | Confirms I8 | accuracy |
| `finetuned-encoder` | EfficientNet-B0 trained, not frozen | `--no-freeze-image-encoder` | Confirms I8 | accuracy |

Every fold of every arm also saves:

- the run-number baseline of its test windows
  (`test.baselines.run_number_class_accuracy`);
- the accuracy on the target steps whose label differs from the photo's
  (`test.photo_change`), which measures forecasting a change rather than
  reading the photo.

## 4. Decision rules, fixed before the runs

These rules are written down before stage 1 starts, so the results cannot
choose them. `scripts/fixes_analysis.py` applies them. All differences are
"arm minus reference", on the metrics averaged over the seeds, paired over
the 10 folds (Wilcoxon test).

| Arm | Adopted for stage 2 when |
|---|---|
| `modality-dropout-0-2`, `photo-edge-crop`, `no-image-gat`, `finetuned-encoder` | accuracy gain ≥ 0.01 and p < 0.10 |
| `revin` | force MSE falls by ≥ 3 % with p < 0.10, and accuracy falls by no more than 0.01 |
| `metadata-position`, `metadata-none` | accuracy falls by no more than 0.01 and the training-test gap shrinks. If both pass, `none` is chosen (fewer codes, no shortcut). |
| `coral` | accuracy falls by no more than 0.01 and "used" recall gains ≥ 0.05 with p < 0.10, or accuracy gains as in the first row |

- **Why p < 0.10.** Stage 1 is a screen: a missed real effect costs more
  than a false one, because stage 2 tests the combination again on its own
  seeds.
- **Why 0.01.** It is about 1.7 times the SD of single runs (I1). An
  average of 5 seeds is about √5 more precise.
- **The metadata rule** puts the burden of proof on keeping the shortcut. A
  code that only helps through the run number does not transfer to a tool
  used at another rate.

**What is reported for every arm**, whether adopted or not:

- accuracy with argmax and with the ordinal + monotone rule;
- macro F1, "used" recall, mean absolute class error;
- force MSE;
- accuracy where the label differs from the photo's;
- the training-test gap and photo-masked accuracy;
- the seed ensemble's accuracy.

## 5. Step 1: update the code and check it

On Ruche, at the start of the session (`TRAINING_RUNBOOK_RUCHE.md` §9):

```bash
cd "$WORKDIR/OmniFuser"
git status --short          # must print nothing: no local changes
git pull
git log -1 --oneline        # must show the commit you pushed with these fixes
source slurm/protocol.sh
mkdir -p experiment_results/logs
```

- `git pull` is safe now: no study is running. Do not pull while the stage-1
  jobs run (a resume checks the settings and the data, not the code).
- Without git on Ruche, copy the code as in `TRAINING_RUNBOOK_RUCHE.md` §5
  (the `git archive` route).

Then run the checks job, which runs every test and the data inspection
(15-30 minutes):

```bash
sbatch slurm/checks.sbatch
```

| Section in the log | Expected |
|---|---|
| `=== tests` | `168 passed` |
| `=== data inspection` | as before: `"records": 378`, `"tools": 10` |
| `=== parameter count` | `4,183,600 ... (match)`; the defaults of the new options do not change the model |

## 6. Step 2: smoke test of the new options

A few minutes on `gpu_test`. It trains one capped fold with **every** new
option at once, including the most demanding combination (the fine-tuned
encoder on the crops), and checks that it fits in memory and writes the new
metrics. Its numbers are not results.

```bash
sbatch --job-name=fixes-smoke --partition=gpu_test --time=00:30:00 slurm/job.sbatch \
  omnifuser train "${PROTOCOL[@]}" --epochs 1 --test-tool T10 \
  --metadata none --revin --class-loss coral \
  --image-crop 0.30 0 0.85 1 --image-size 128 640 --no-freeze-image-encoder \
  --max-train-windows 64 --max-eval-windows 32 \
  --output-dir experiment_results/validation/fixes-smoke
```

Then check:

```bash
source slurm/env.sh
python -c "
import json; r = json.load(open('experiment_results/validation/fixes-smoke/fold-T10.json'))
print(r['config']['metadata'], r['config']['revin'], r['config']['class_loss'], r['config']['image_crop'])
print('run-number baseline', r['test']['baselines']['run_number_class_accuracy'])
print('photo change', r['test']['photo_change'])"
grep -i "out of memory" experiment_results/logs/fixes-smoke-*.out || echo "no memory error"
```

Expected: `none True coral [0.3, 0.0, 0.85, 1.0]`, two numbers, and `no
memory error`. If it runs out of memory, see §13.

## 7. Step 3: evaluations of the saved models (no training)

The finished runs' models (`fold-Tn.pt`) are still on Ruche. Two
evaluations use them without retraining. They run as CPU jobs while stage 1
trains on the GPUs.

**7.1 Class offsets tuned on the validation tool (fix A2).** This fits
per-class log-probability offsets on each fold's validation tool and
applies them to its test tool. It confirms or rejects the post-hoc rules on
data they were not chosen on (`FIXES_AND_VALIDATION.md` §4).

```bash
srun --partition=cpu_med --time=02:00:00 --cpus-per-task=4 bash -c '
  source slurm/env.sh
  python scripts/decision_study.py experiment_results/main/full-model \
    experiment_results/image-encoder/encoder-v1/image-encoder/pretrained-frozen-no-gat \
    --tune-offsets --data-dir Nonastreda_Data \
    --output experiment_results/decision/decision-study-offsets.json'
```

- The output lists, per run, the accuracy of argmax, the post-hoc rules and
  `offsets+argmax`, `offsets+monotone`, with the fitted offsets per fold.
- **Read it as:** offsets help if `offsets+argmax` beats `argmax` on most
  folds. They are fitted on one tool of 28-53 windows, so a gain below 0.01
  is noise.
- If `cpu_med` does not exist under that name, use the CPU partition listed
  by `sinfo` (`TRAINING_RUNBOOK_RUCHE.md` §8).

**7.2 Forecasts at every phase of a pass (the MSE-horizon question).** This
re-scores the main run's 10 models at stride 24 (forecasts starting at every
phase of a pass) and at the original stride 96:

```bash
srun --partition=cpu_med --time=02:00:00 --cpus-per-task=4 bash -c '
  source slurm/env.sh
  python scripts/phase_study.py experiment_results/main/full-model \
    --data-dir Nonastreda_Data --strides 96 24'
```

- It prints the cumulative MSE at the paper's 8 horizons for both strides
  and writes `experiment_results/main/full-model/phase-study/phase-study.json`.
- **Read it as:** if the stride-24 curve rises with the horizon, like the
  paper's, while the stride-96 curve falls, the falling curve came from the
  windows' alignment with the passes (`FINAL_REPORT_REPRODUCTION.md` §6.3),
  not from the model.

## 8. Step 4: stage 1, the single-change arms

One suite per seed. They are independent, so they run in parallel, up to
your GPU limit; the fifth waits for a free GPU.

```bash
for seed in 42 142 242 342 442; do
  sbatch --job-name=fixes-s$seed --time=08:00:00 slurm/job.sbatch \
    python scripts/run_experiment_suite.py --study fixes --run-id s$seed \
    --output-root experiment_results "${PROTOCOL[@]}" --seed $seed
done
```

| Part | Meaning |
|---|---|
| `--study fixes --run-id s$seed` | the 9 arms of §3, into `experiment_results/fixes/s<seed>/fixes/<arm>/` |
| `"${PROTOCOL[@]}"` | the main runs' settings (frozen pretrained EfficientNet-B0, 30 epochs, ...) |
| `--seed $seed` | placed after the protocol, so it replaces the protocol's `--seed 42` |
| `--time=08:00:00` | about 2.5 h are expected (8 arms × ~16 min + the fine-tuned arm ~30 min); the margin covers a slow node |

- **If a job stops** (time limit, node failure), submit the same command for
  that seed again. The suite skips finished arms and resumes the interrupted
  fold from its last epoch.
- **Follow progress** with `bash slurm/status.sh`: each suite shows
  `n/9 variants`.

## 9. Step 5: analyse stage 1

When all suites show `completed` (it reads files only, under a minute):

```bash
srun --partition=cpu_short --time=00:20:00 bash -c '
  source slurm/env.sh
  python scripts/fixes_analysis.py --root experiment_results/fixes
  python scripts/decision_study.py experiment_results/fixes/s*/fixes/* \
    --output experiment_results/decision/decision-study-fixes.json'
```

`fixes_analysis.py` prints:

1. one row per arm: accuracy (argmax), accuracy (ordinal + monotone),
   "used" recall, force MSE, training-test gap, accuracy where the label
   differs from the photo, and the seed ensemble's accuracy (all seed
   averages);
2. one line per arm: `ADOPT` or `reject`, with the numbers behind the rule
   of §4;
3. `stage-2 flags: ...`, the flags of the adopted arms.

It writes everything, including the per-fold paired tests, to
`experiment_results/fixes/fixes-analysis.json`.

**Check before stage 2:**

- The `reference` arm's accuracy should lie near the seven replicates'
  0.659-0.676. If not, something in the code changed the protocol; stop and
  compare its `fold-T1.json` `config` with `experiment_results/main/full-model/fold-T1.json`.
- If `photo-edge-crop` and `finetuned-encoder` are both adopted, stage 2
  fine-tunes on the crops (the smoke test of §6 ran this combination).

## 10. Step 6: stage 2, the combined model

Copy the printed `stage-2 flags` into `FLAGS`. For example, if RevIN, the
`none` metadata and the edge crop were adopted:

```bash
FLAGS="--metadata none --revin --image-crop 0.30 0 0.85 1 --image-size 128 640"
for seed in 42 142 242 342 442; do
  sbatch --job-name=fixes-combined-s$seed --time=12:00:00 slurm/job.sbatch \
    python scripts/train_lightning.py "${PROTOCOL[@]}" --seed $seed \
    --dedup-identical-forces True --eval-dedup-identical-forces True \
    $FLAGS --output-dir experiment_results/fixes/s$seed/fixes/combined
done
```

- `$FLAGS` is written without quotes, so each flag becomes its own argument.
- The output goes next to the stage-1 arms. `fixes_analysis.py` then
  compares `combined` with the reference in the same way (accuracy rule).
- **If no arm is adopted**, skip stage 2. The result is then the reference
  with the post-hoc decision rule and the seed ensemble.
- **Memory.** Fine-tuning on the 128 × 640 crops encodes 1.7 times the
  pixels of the main runs' photos. If it runs out of memory, add
  `--batch-size 16 --accumulation-steps 2`: the effective batch stays 32.
  This changes only the BatchNorm statistics of the fine-tuned encoder;
  note it in the report.

When they have finished, rerun the analysis of §9. The final result to
report is `combined`, with:

- the seed average and the seed ensemble;
- argmax and the ordinal + monotone rule;
- the paired test against `reference` over the 10 folds.

Then run the offsets of §7.1 on the combined runs. Use one seed folder per
call, for example `experiment_results/fixes/s42/fixes/combined`.

## 11. Step 7: download

Pack only the new results (metrics, predictions, logs; the models stay on
Ruche unless you need them):

```bash
cd "$WORKDIR/OmniFuser"
tar -czf results-fixes.tar.gz --exclude='*.pt' experiment_results/fixes \
  experiment_results/decision experiment_results/main/full-model/phase-study \
  experiment_results/validation/fixes-smoke experiment_results/logs
ls -lh results-fixes.tar.gz
```

On the laptop, from the repository folder (PowerShell; set `$Ruche` and
`$RucheLogin` first, `TRAINING_RUNBOOK_RUCHE.md` §3):

```powershell
cd C:\path\to\OmniFuser
scp "${Ruche}:/workdir/$RucheLogin/OmniFuser/results-fixes.tar.gz" .
tar -xzf results-fixes.tar.gz
python scripts/fixes_analysis.py --root experiment_results/fixes
```

Record the results in `FIXES_AND_VALIDATION.md` (the §6 status table and a
new results section).

## 12. What this runbook does not cover, and why

| Fix | Why not here | What is needed first |
|---|---|---|
| C1 Chip and workpiece photos | The two photo folders are not in `Nonastreda_Data/` (only `tool/`). Their alignment with the passes (one photo per pass? taken during or after it?) decides whether they leak the label like the lag-0 tool photo. | Download the folders from the dataset's source, check the alignment and timing, then add a multi-view image branch and an `image-views` suite |
| C2 Spectrogram as a third stream | C2F fuses exactly two streams; a third needs a new fusion design, not a flag | A design decision; the spectrogram alone already gives the best forecast MSE (`FINAL_REPORT_REPRODUCTION.md` §3.2) |
| Photo augmentation | Cached frozen features cannot be augmented; it only applies with fine-tuning | Stage 2 adopting fine-tuning; then augmentation as its own arm |
| Smoothed (isotonic) training labels | The test labels keep their noise, so the gain is capped and hard to see; the monotone decision rule already handles decreasing labels at inference | Only if `coral` fails |
| Changing the evaluation stride of training runs | It would change every test set and break comparability with the 51 runs; §7.2 measures the effect on the saved models instead | Nothing; report the phase study alongside |

## 13. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `cannot resume ...; study settings differ from suite.json` | The same `--run-id` was started with other settings (for example another seed) | Use one run ID per seed (`s42`, `s142`, ...) exactly as in §8 |
| `CUDA out of memory` in `finetuned-encoder` or `combined` | Fine-tuning, most likely on the crops | Add `--batch-size 16 --accumulation-steps 2` (§10) |
| `class_loss 'coral' needs the direct prediction head` | Another `--prediction-head` was given | The protocol uses `direct`; do not override it |
| `reference` accuracy far from 0.659-0.676 | The protocol changed | Compare the `config` and `training` blocks of `fold-T1.json` with the main run's |
| A `decision_study.py --tune-offsets` error about spectrograms | Offsets are implemented for photo runs only | Leave spectrogram runs out of that call |

Everything else (queue states, quotas, time limits, logs) is in
`TRAINING_RUNBOOK_RUCHE.md` §16 and §18.

## 14. Cheat sheet

```bash
# every session
cd "$WORKDIR/OmniFuser"; source slurm/protocol.sh; mkdir -p experiment_results/logs

# step 1-2: code, checks, smoke test
git pull; sbatch slurm/checks.sbatch
sbatch --job-name=fixes-smoke --partition=gpu_test --time=00:30:00 slurm/job.sbatch \
  omnifuser train "${PROTOCOL[@]}" --epochs 1 --test-tool T10 --metadata none --revin \
  --class-loss coral --image-crop 0.30 0 0.85 1 --image-size 128 640 --no-freeze-image-encoder \
  --max-train-windows 64 --max-eval-windows 32 --output-dir experiment_results/validation/fixes-smoke

# step 4: stage 1 (one suite per seed)
for seed in 42 142 242 342 442; do
  sbatch --job-name=fixes-s$seed --time=08:00:00 slurm/job.sbatch \
    python scripts/run_experiment_suite.py --study fixes --run-id s$seed \
    --output-root experiment_results "${PROTOCOL[@]}" --seed $seed
done

# step 5: analysis
srun --partition=cpu_short --time=00:20:00 bash -c 'source slurm/env.sh; python scripts/fixes_analysis.py'

# status
bash slurm/status.sh
```

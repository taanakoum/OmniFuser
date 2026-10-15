# OmniFuser training runbook

This runbook takes you from your laptop to a full set of OmniFuser experiments
on Lightning AI. It assumes you have **never used Lightning AI before**. Every
command is explained, every argument is listed in §16, and the steps are meant
to be run in order.

The paper reports results on "Mudestreda" and MATWI. This repository trains on
Nonastreda, so the paper's numbers are reference points, not targets.

**Contents**

| Part | Sections |
|---|---|
| **Part A: Lightning AI for first-time users** | §1 Concepts · §2 Prepare on your laptop · §3 Create a Studio · §4 Get the code · §5 Upload the data · §6 Install · §7 Switch to a GPU · §8 Terminal basics |
| **Part B: The experiments** | §9 Shared protocol · §10 Step 0: verify · §11 Steps 1-2: smoke test and pilot · §12 Step 3: main 10-fold run · §13 Steps 4-6: paper studies · §14 Optional runs |
| **Part C: Operating and reference** | §15 Unattended runs, resuming, downloading · §16 Complete argument reference · §17 Results layout and metrics · §18 Local Windows commands · §19 Not covered |

## Overview of the experiments

| Step | What it runs | 10-fold CV runs | Output folder |
|---|---|---|---|
| 0 | Tests, GPU check, data inspection, EDA | 0 | `experiment_results/eda/` |
| 1 | Capped one-fold GPU smoke test | 0 (1 fold) | `experiment_results/validation/gpu-smoke/` |
| 2 | Uncapped one-fold pilot (timing) | 0 (1 fold) | `experiment_results/pilot/T10/` |
| 3 | Full model, all 10 folds | 1 | `experiment_results/main/full-model/` |
| 4 | Paper Fig. 13 component ablations | 9 | `experiment_results/ablations/<run-id>/` |
| 5 | Missing-modality robustness | 2 | `experiment_results/missing-modality/<run-id>/` |
| 6 | Paper Figs. 7-8 parameter sensitivity | 22 | `experiment_results/sensitivity/<run-id>/` |
| (alt) | Steps 4-6 in one unattended suite | 31 | `experiment_results/all/<run-id>/` |
| (opt) | Printed-paper readings of the four deviations | 4 | `experiment_results/paper-literal/<arm>/` |
| (opt) | Duplicate handling: merged versus kept, scored on the same deduplicated test data | 2 | `experiment_results/deduplication/<run-id>/` |

A **10-fold CV run** trains 10 models. Each one holds out a different tool for
testing, uses the next tool for validation, and trains on the other 8
(`EDA.md` §8 explains folds). Every run also evaluates each fold three ways:
with both inputs (`full`), with the force signal masked (`sensor_missing`),
and with the photos masked (`image_missing`). Missing-modality evaluation
needs no extra command.

---

# Part A: Lightning AI for first-time users

## 1. Concepts you need

- **Lightning AI** is a website (lightning.ai) that rents you cloud machines
  with GPUs.
- **Studio.** You work inside a *Studio*: a cloud computer that opens in
  your browser with a VS Code-style editor, a file explorer, and terminals.
- **Persistent storage.** Each Studio has a home folder,
  `/teamspace/studios/this_studio`. Everything stored there survives
  stopping, restarting, and switching machines. Keep the code, data and
  results there.
- **Machine type.** A Studio runs on a machine you choose: CPU-only, or a GPU
  such as T4, L4, A10G, L40S or A100. You can switch at any time; the
  Studio restarts on the new hardware and your files stay.
- **Billing.** GPU time costs credits per hour. A CPU machine is enough for
  setup (§3-§6); switch to a GPU only to train (§7). Stop the Studio when you
  are not using it.
- **Sleep.** Idle Studios can go to sleep. Long training must survive a
  closed browser tab; §15 shows how. Check the Studio's sleep settings
  before a multi-hour run. If a run is interrupted anyway, rerunning the same
  command resumes it.
- **No Lightning code.** This project does not use the PyTorch Lightning
  library. It is a plain PyTorch program that you run from the Studio's
  terminal.

## 2. Prepare on your laptop

The Studio gets the code from GitHub, so **push your current local work
first**. Local commits that are not pushed will be missing in the Studio. In
your local repository (PowerShell or Git Bash):

```bash
git status                       # what has changed
git add -A                       # stage everything that is not git-ignored
git commit -m "Prepare for Lightning AI training"
git push                         # upload to GitHub
git branch --show-current        # note this branch name for §4
```

`Nonastreda_Data/` and `experiment_results/` are git-ignored, so the data
does not go through GitHub. Zip it for upload. In PowerShell, from the
repository folder:

```powershell
Compress-Archive -Path Nonastreda_Data -DestinationPath Nonastreda_Data.zip
```

The zip is about 400 MB: a 270 MB force file and 512 photos.

## 3. Create a Studio

1. Sign in at lightning.ai and create a new **Studio**. Start with a
   **CPU** machine; setup does not need a GPU.
2. When it opens, open a terminal from the editor's menu (Terminal → New
   Terminal). The prompt starts in `/teamspace/studios/this_studio`.
3. Check what you have:

   ```bash
   pwd                     # should print /teamspace/studios/this_studio
   python --version        # needs 3.10 or newer
   nproc                   # number of CPU cores; used for --num-workers (§9)
   ```

## 4. Get the code

```bash
cd /teamspace/studios/this_studio
git clone -b <your-branch> https://github.com/<your-user>/OmniFuser.git OmniFuser
cd OmniFuser
git log -1 --oneline        # confirm it is the commit you pushed in §2
```

- `git clone -b <branch> <url> OmniFuser` downloads the repository into a
  new folder `OmniFuser`, checked out at your branch.
- If the repository is private, GitHub asks for a username and a password.
  Use a **personal access token** as the password (GitHub → Settings →
  Developer settings → Personal access tokens).
- To pick up later changes, run `git pull` inside `OmniFuser/`.

## 5. Upload the data

1. Drag `Nonastreda_Data.zip` from your computer onto the `OmniFuser` folder
   in the Studio's file explorer and wait for the upload to finish.
2. Unpack it in the terminal:

   ```bash
   cd /teamspace/studios/this_studio/OmniFuser
   unzip -q Nonastreda_Data.zip && rm Nonastreda_Data.zip
   ls Nonastreda_Data          # forces_xyz_raw.mat  labels.csv  labels_reg.csv  tool
   ls Nonastreda_Data/tool | wc -l     # 512
   ```

   `unzip -q` extracts quietly; `rm` deletes the zip afterwards to save
   space. If `unzip` is missing, run
   `python -m zipfile -e Nonastreda_Data.zip .` instead.

## 6. Install the project

```bash
python -m pip install -e ".[test,eda]"
```

- `python -m pip install` installs packages into the Studio's Python.
- `-e .` installs this folder in *editable* mode: code changes you pull later
  apply without reinstalling. It also creates the `omnifuser` command.
- `[test,eda]` adds two optional extras: `test` (pytest) and `eda`
  (matplotlib). Add `dev` for the ruff/pyright checkers.

Studios usually come with PyTorch preinstalled; pip keeps it if the version
is new enough (torch ≥ 2.2).

## 7. Switch to a GPU

1. Use the Studio's machine selector to change to a GPU. Any modern 16-24 GB
   GPU is enough:
   - the model has 4.2 M parameters;
   - the paper reports 7.26 GB peak memory on an RTX 4090;
   - an **L4** or **A10G** is a good choice, and a **T4** works but is slower.
2. The Studio restarts. Open a new terminal and check:

   ```bash
   cd /teamspace/studios/this_studio/OmniFuser
   nvidia-smi                  # the GPU name, its memory, and running processes
   python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA')"
   ```

   The second command must print `True` and the GPU name. If it prints
   `False`, the installed PyTorch has no CUDA support: fix that before
   training.

## 8. Terminal basics used below

| Syntax | Meaning |
|---|---|
| `\` at the end of a line | The command continues on the next line. Nothing may follow the `\`, not even a space. |
| `NAME="..."` then `$NAME` | Store text in a shell variable, then insert it into later commands. Variables last for one terminal session; redefine them in every new terminal. |
| `cmd 2>&1 \| tee file.log` | Show the output and also save it to `file.log`. `2>&1` includes error messages. |
| `nohup cmd > file.log 2>&1 &` | Run in the background, immune to closing the terminal or browser. Output goes only to the file. |
| `tail -f file.log` | Follow a log live. Ctrl+C stops following; it does not stop the run. |
| `tmux new -s train` / Ctrl+B then D / `tmux attach -t train` | Start a detachable terminal, detach from it, and re-attach later. Optional alternative to `nohup`. |
| `watch -n 5 nvidia-smi` | Refresh GPU use and memory every 5 s. Ctrl+C exits. |
| `ps aux \| grep python` and `kill <PID>` | Find and stop a running Python process |

---

# Part B: The experiments

## 9. Shared protocol (define once per terminal)

```bash
cd /teamspace/studios/this_studio/OmniFuser
PROTOCOL="--data-dir Nonastreda_Data --device cuda \
  --epochs 30 --batch-size 32 --learning-rate 1e-4 \
  --input-len 96 --horizon 96 --stride 24 \
  --pool 1000 --image-lag 1 --prediction-head direct \
  --image-encoder efficientnet_b0 --seed 42 --num-workers 4"
mkdir -p experiment_results/logs
```

Every training command below starts with `$PROTOCOL`, so all experiments
share the same data, windows and optimizer. When a command repeats a flag
after `$PROTOCOL`, the **later value wins**; that is how individual steps
override it (for example `--epochs 1`).

| Argument | Value | Meaning |
|---|---|---|
| `--data-dir` | `Nonastreda_Data` | Folder with the MAT file, the two CSVs, and `tool/` |
| `--device` | `cuda` | Train on the GPU (`cpu` for local tests) |
| `--epochs` | 30 | Passes over the training windows per fold. The paper gives no number, so 30 is a project choice. |
| `--batch-size` | 32 | Windows per optimizer step (paper value) |
| `--learning-rate` | 1e-4 | Adam learning rate (paper value) |
| `--input-len` | 96 | Input steps per window (\(T\)) |
| `--horizon` | 96 | Steps to forecast (\(P\)). Paper horizons 12…96 are scored cumulatively from one 96-step forecast. |
| `--stride` | 24 | Steps between consecutive *training* windows. Evaluation windows use \(\max(\text{stride},P)=96\), so evaluation targets never overlap. |
| `--pool` | 1000 | Average 1000 force samples (1 s) into one step. This matches the paper's difficulty (`EDA.md` §6). |
| `--image-lag` | 1 | Show the previous pass's photo. The same-pass photo (0, the paper's reading) leaks the label (`EDA.md` §7). |
| `--prediction-head` | `direct` | Separate output weights per forecast step (closest usable reading of Eqs. 21-22) |
| `--image-encoder` | `efficientnet_b0` | The paper's photo backbone (`compact` is for quick CPU tests) |
| `--seed` | 42 | Random seed. Fold \(i\) uses seed + \(i\). |
| `--num-workers` | 4 | Background processes preparing batches. Use at most the `nproc` count; 0 loads in the main process. |

The architecture defaults equal the paper's values, so they are not in
`$PROTOCOL`: \(d=32\), 4 MRTE layers with stride 3 and rank 8, 4 GAT layers ×
4 heads, 4 landmarks, 2 C2F rounds, and dropout 0.1.

> **Resuming requires identical settings.** A run resumes only if every flag
> is identical, including `--num-workers`. To change any setting, use a new
> output folder.

## 10. Step 0: verify the environment

```bash
python -m pytest -o addopts= -q tests
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA')"
omnifuser inspect --data-dir Nonastreda_Data --pool 1000 --input-len 96 --horizon 96 --stride 24
python scripts/model_summary.py --check
python scripts/eda.py --data-dir Nonastreda_Data --out experiment_results/eda
```

| Command | What it does | What to check |
|---|---|---|
| `pytest ... tests` | Runs the 119 tests. `-o addopts=` clears the default options and `-q` keeps the output short. | `119 passed` |
| `python -c "import torch ..."` | Prints the torch version, whether CUDA works, and the GPU name | `True` and your GPU |
| `omnifuser inspect` | Loads and validates the whole dataset with the protocol's pooling and windows, and prints a JSON report | `"records": 378`, `"tools": 10`, class counts 147/118/113, `"exact_duplicate_force_rows_remaining": 0` |
| `model_summary.py --check` | Builds the default model, prints per-block parameter counts and shapes, and checks the total | `4,183,600 ... (match)` |
| `eda.py` | The exploratory data analysis: tables, a JSON summary and 12 figures in `experiment_results/eda/`. It takes several minutes and about 3 GB of RAM. | Explained in `EDA.md` |

## 11. Steps 1-2: GPU smoke test and timing pilot

### Step 1: capped smoke test (a few minutes)

This checks the real data path, CUDA training, all three evaluation modes,
tracking, and artifact writing. Its metrics are not results.

```bash
omnifuser train $PROTOCOL --epochs 1 --test-tool T10 \
  --output-dir experiment_results/validation/gpu-smoke \
  --max-train-windows 64 --max-eval-windows 32
```

| Argument | Effect |
|---|---|
| `omnifuser train` | Trains and evaluates **one** fold (one held-out tool) |
| `--epochs 1` | Overrides the protocol's 30 |
| `--test-tool T10` | Hold out tool T10 (any of T1…T10). Validation is the next tool, T1. |
| `--output-dir` | Where this fold's files go |
| `--max-train-windows 64`, `--max-eval-windows 32` | Use a random, reproducible subset of windows, for speed. Never use these flags for real results. |

The folder should contain:

- `fold-T10.json` and `fold-T10.pt`;
- three prediction files: `fold-T10-predictions.npz` plus the
  `-sensor_missing` and `-image_missing` variants;
- three matching ROC files;
- `tracking/`.

### Step 2: uncapped one-fold pilot (measures cost)

```bash
omnifuser train $PROTOCOL --epochs 3 --test-tool T10 \
  --output-dir experiment_results/pilot/T10 \
  2>&1 | tee experiment_results/logs/pilot-T10.log
```

This trains one complete fold for 3 epochs. To estimate cost, open
`experiment_results/pilot/T10/tracking/metrics.csv` and read two rows for
split `train`:

- `epoch_seconds`;
- `cuda_peak_allocated_bytes` (divide by 10⁹ for GB).

One 10-fold run costs about `epoch_seconds × 30 × 10`, plus about one epoch
per fold for the final evaluations. Multiply by the run counts in the
overview to estimate a study.

If you get `CUDA out of memory`, add `--batch-size 16 --accumulation-steps 2`.
That gives the same effective batch of 32, using half the memory per step.

## 12. Step 3: full model, 10-fold cross-validation

```bash
python scripts/train_lightning.py $PROTOCOL \
  --output-dir experiment_results/main/full-model \
  2>&1 | tee experiment_results/logs/main-full-model.log
python scripts/compare_with_paper.py experiment_results/main/full-model \
  --json experiment_results/main/full-model/paper-comparison.json
```

**`train_lightning.py`** runs all 10 folds one after another, then writes
`cross-validation-results.json`. That file holds fold means ± std, pooled
out-of-fold metrics, and per-mode summaries. Each fold:

1. trains on 8 tools;
2. picks the epoch with the lowest validation loss on the next tool;
3. tests on the held-out tool.

The force normalizer is fitted on the 8 training tools only. `--output-dir`
is required.

**`compare_with_paper.py <run folder>`** prints, for horizons 12…96:

- z-scored cumulative MSE/MAE and classification scores;
- the last-value and persistence baselines;
- the paper's Table IV/V values.

Its options:

| Option | Effect |
|---|---|
| `--json <file>` | Also save the table |
| `--average weighted` | Support-weighted precision instead of macro |
| `--by pooled_by_windows` | Weight folds by their window counts |

## 13. Steps 4-6: the paper's studies

All three use `scripts/run_experiment_suite.py`. It runs each **variant** as
its own full 10-fold run, then writes a manifest and a comparison table. Its
own arguments:

| Argument | Meaning |
|---|---|
| `--study` | `ablations`, `sensitivity`, `missing-modality`, `all` (the three paper studies), or `deduplication` (§14) |
| `--run-id` | Folder name for this suite, e.g. `paper-v1`. Reuse it to resume; use a new one for a new suite. Without it, a UTC timestamp is used. |
| `--output-root` | Parent folder. The suite writes to `<output-root>/<study>/<run-id>/`. |
| `--modality-dropout-rates` | Training dropout rates for the missing-modality study (default `0.2`) |

All training flags from §9 and §16 apply to every variant. The suite's own
defaults already equal the protocol (`--pool 1000 --image-lag 1`, 30 epochs,
stride 24, CUDA), so `$PROTOCOL` is passed only for uniformity.

### Step 4: component ablations (paper Fig. 13), 9 runs

```bash
python scripts/run_experiment_suite.py --study ablations --run-id paper-v1 \
  --output-root experiment_results $PROTOCOL \
  2>&1 | tee experiment_results/logs/ablations-paper-v1.log
```

| Paper variant | `--ablation` value | What the code does |
|---|---|---|
| Full model | `none` | Reference arm |
| w/o-RR | `without_rr` | One C2F pass, no recursive refinement |
| w/o-MRTE | `without_mrte` | MRTE replaced by an identity after the force embedding |
| w/o-RR&MRTE | `without_rr_mrte` | Both of the above |
| re-C2F-Concat | `concat_c2f` | Concatenate both streams and project, at each round |
| w/o-C2F | `without_c2f` | Equal-weight mean of the two streams feeds the heads |
| w/o-ALS | `without_als` | Fixed segment-mean landmarks (non-adaptive) |
| w/o-HG | `without_hybrid_gate` | Fixed 0.5 mixture of cross-modal and private features |
| re-LR-MA | `moving_average` | A 3-step moving average replaces the low-rank trend |

The paper does not specify the replacement operators for w/o-C2F, w/o-ALS,
w/o-HG and re-LR-MA. The choices above are recorded in `suite.json`.

Fig. 13 reports MSE, so compare the `full.regression_mse_normalized.mean`
column of `suite-summary.csv`: z-scored MSE averaged over all 96 leads.

To run one ablation alone:

```bash
python scripts/train_lightning.py $PROTOCOL --ablation without_als \
  --output-dir experiment_results/ablations/single/without-als
```

### Step 5: missing-modality robustness, 2 runs

The paper uses modality dropout in training and masking at inference. The
study trains two arms:

- a reference arm without dropout;
- an arm with training dropout 0.2. The paper gives no rate, so 0.2 is a
  choice.

Every run is then evaluated with the force input masked and with the photos
masked.

```bash
python scripts/run_experiment_suite.py --study missing-modality --run-id paper-v1 \
  --output-root experiment_results $PROTOCOL \
  2>&1 | tee experiment_results/logs/missing-modality-paper-v1.log
```

To sweep several rates, use a new `--run-id`:

```bash
python scripts/run_experiment_suite.py --study missing-modality --run-id rates-v1 \
  --modality-dropout-rates 0.1 0.2 0.3 --output-root experiment_results $PROTOCOL
```

To read the results, compare the `full.*`, `sensor_missing.*` and
`image_missing.*` columns within each row of `suite-summary.csv`. Graceful
degradation means the masked scores drop toward, but not below, the
persistence baselines.

### Step 6: parameter sensitivity (paper Figs. 7-8), 22 runs

```bash
python scripts/run_experiment_suite.py --study sensitivity --run-id paper-v1 \
  --output-root experiment_results $PROTOCOL \
  2>&1 | tee experiment_results/logs/sensitivity-paper-v1.log
```

| Paper axis | Config field (CLI flag) | Values swept |
|---|---|---|
| k, landmarks | `num_landmarks` (`--landmarks`) | 2, 4, 8, 16 |
| d, embedding width | `d_model` (`--d-model`) | 16, 32, 64, 128 |
| GAT heads | `num_gat_heads` (`--gat-heads`) | 2, 4, 8, 16 |
| r, trend-seasonal rank | `rtd_rank` (`--rtd-rank`) | 4, 8, 16, 32 |
| GAT layers | `num_gat_layers` (`--gat-layers`) | 2, 3, 4 |
| s_l, downsampling stride | `rtd_stride` (`--rtd-stride`) | 1, 2, 3 |
| L, MRTE layers | `num_rtd_layers` (`--rtd-layers`) | 1, 2, 3, 4 |
| n, refinement iterations | `num_fusion_layers` (`--fusion-layers`) | 1, 2, 3 |

Values equal to the default are not retrained; they share the
`sensitivity-baseline` row. That makes 22 runs. The GAT-heads value also sets
the head count of the C2F attention.

### Alternative: steps 4-6 in one unattended command, 31 runs

The ablation full model serves as the shared reference, so this trains 31
runs instead of 33:

```bash
nohup python scripts/run_experiment_suite.py --study all --run-id paper-v1 \
  --output-root experiment_results $PROTOCOL \
  > experiment_results/logs/all-paper-v1.log 2>&1 &
tail -f experiment_results/logs/all-paper-v1.log
```

## 14. Optional runs

### Duplicate handling: merged versus kept

Nonastreda has 134 force recordings that are bit-identical copies of another
pass in the same run, mostly block B4 repeating B3 (`EDA.md` §3). In 24 of
these groups the copies carry different wear labels. Two independent options
control them:

| Option | Applies to | Values |
|---|---|---|
| `--dedup-identical-forces {True,False}` | The **training** data, and by default everything else | `True` (default): one pass per group, 378 passes. `False`: all 512. |
| `--eval-dedup-identical-forces {True,False}` | The **validation and test** data | Omitted: same as `--dedup-identical-forces`. Set it to fix the evaluation data independently. |

The comparison study trains both settings with everything else identical,
and scores both on the **same deduplicated validation and test windows**:

```bash
python scripts/run_experiment_suite.py --study deduplication --run-id dedup-v1 \
  --output-root experiment_results $PROTOCOL \
  2>&1 | tee experiment_results/logs/deduplication-dedup-v1.log
```

| Variant | Training data | Validation and test data |
|---|---|---|
| `dedup-true` | 378 passes | 378 passes |
| `dedup-false` | 512 passes | 378 passes |

Because the test windows are identical, any metric difference comes from the
training data alone. To score both arms on non-deduplicated data instead, add
`--eval-dedup-identical-forces False` to the suite command.

`suite-summary.csv` gets one row per setting. Besides the usual metrics it
has these columns:

| Column | Contents |
|---|---|
| `dedup_identical_forces`, `eval_dedup_identical_forces` | The training and evaluation settings |
| `retained_passes`, `evaluation_passes` | Passes in the training data and in the evaluation data |
| `train_windows.mean`, `test_windows.mean` | Window counts per fold. `test_windows.mean` is equal for both arms when evaluation is deduplicated. |

To run one setting outside the suite:

```bash
# train with duplicates, test on the deduplicated windows
python scripts/train_lightning.py $PROTOCOL \
  --dedup-identical-forces False --eval-dedup-identical-forces True \
  --output-dir experiment_results/deduplication/single/dedup-false-eval-true

# the whole protocol without deduplication (training and test)
python scripts/train_lightning.py $PROTOCOL --dedup-identical-forces False \
  --output-dir experiment_results/deduplication/single/dedup-false-eval-false
```

`compare_with_paper.py` prints both settings and both pass counts in its
header line, e.g. `dedup train/eval=False/True, passes train/eval=512/378`.

To see the data effect without training:

```bash
omnifuser inspect --data-dir Nonastreda_Data --pool 1000 --stride 24 --dedup-identical-forces False
```

This prints 512 passes, 2,012 windows and 134 remaining duplicates, against
378, 1,462 and 0 with `True`.

What to keep in mind when you read the results:

- **Use deduplicated evaluation for the comparison.** If both training and
  test keep the duplicates (`--dedup-identical-forces False` alone), each
  test tool also contains its copied passes. The two settings are then scored
  on different windows, and the copies make the test easier: at 1 s steps a
  pass lasts about 97 steps, so a 96-step input window that ends in B3
  contains nearly all of B3, which is exactly the signal B4 repeats.
- **Duplicates in training are a weighting change.** With
  `--dedup-identical-forces False`, the copied passes appear twice in
  training, so those passes count double and the model sees more windows per
  epoch.
- **Conflicting labels.** In 24 groups the same force signal carries two
  different labels. In training that is label noise. With deduplicated
  evaluation the most-worn label is the one scored, as in the default
  protocol.

### Prediction-head comparison (not a paper experiment)

```bash
for head in paper_literal lead_conditioned direct; do
  python scripts/train_lightning.py $PROTOCOL --prediction-head $head \
    --output-dir experiment_results/heads/$head \
    2>&1 | tee experiment_results/logs/head-$head.log
done
```

`for head in a b c; do ...; done` repeats the command once per value, with
`$head` replaced each time.

### Printed-paper readings of the four deviations

Four defaults depart from the printed paper (`PAPER_IMPLEMENTATION_ALIGNMENT.md`
explains each). These runs measure what each departure changes:

```bash
python scripts/train_lightning.py $PROTOCOL --image-lag 0 \
  --output-dir experiment_results/paper-literal/same-pass-image
python scripts/train_lightning.py $PROTOCOL --prediction-head paper_literal \
  --output-dir experiment_results/paper-literal/repeated-head
python scripts/train_lightning.py $PROTOCOL --decomp-mode literal \
  --output-dir experiment_results/paper-literal/eq3-literal
python scripts/train_lightning.py $PROTOCOL --als-mode k_sum \
  --output-dir experiment_results/paper-literal/eq10-unnormalized
```

Two other variants are available:

- `--exact-split`: the Eq. 7 shared/private reading.
- `--private-attn residual_ln --als-mode time`: the defaults used before
  2026-10-14.

For the same-pass-photo arm, compare its accuracy with
`persistence_class_accuracy` in each fold JSON. With lag 0, the shown photo's
label already equals most future labels.

---

# Part C: Operating and reference

## 15. Unattended runs, resuming, and downloading results

- **Keep runs alive.** Use `nohup ... &` (§13) or `tmux` (§8) for anything
  longer than a few minutes. Plain `| tee` commands stop when their terminal
  closes.
- **Watch progress:**

  ```bash
  tail -f experiment_results/logs/<name>.log
  watch -n 5 nvidia-smi
  cat experiment_results/<study>/<run-id>/suite.json | head -40
  ```

  - The log prints one `Fold k/10` line per fold.
  - `suite.json` shows each variant's status.
  - `cross-validation-progress.json` in a run folder updates after each fold.
- **Resume after any interruption**, including a sleep, a crash or a stopped
  Studio: rerun the **exact same command**, with the same `--run-id` for
  suites.
  - Finished variants are skipped.
  - Finished folds are reused.
  - An interrupted fold continues from its last finished epoch
    (`fold-Tn-resume.pt`).
- **Changed settings are refused.** Different settings stop with an error
  instead of mixing configurations. A different GPU type or library version
  is allowed and noted in `tracking/run.json` under `runtime_history`.
- **Do not modify the data mid-study.** Each input file's path, size and
  modification time are recorded and checked. Editing or re-uploading
  `Nonastreda_Data/` during a study forces retraining.
- **Download results.** In the file explorer, right-click
  `experiment_results` and choose **Download**. Or zip it first; the second
  command below leaves out the large checkpoints:

  ```bash
  zip -qr results.zip experiment_results
  zip -qr results-no-checkpoints.zip experiment_results -x "*.pt"
  ```

  Each EfficientNet checkpoint is about 17 MB, so one 10-fold run is about
  170 MB.
- **Stop the Studio** when finished, so GPU time is no longer billed. Files
  in `/teamspace/studios/this_studio` remain.

## 16. Complete argument reference

### Shared training arguments

`omnifuser train`, `scripts/train_lightning.py`, `scripts/train_local.py`
and `scripts/run_experiment_suite.py` all accept these. "Library default" is
the value when the flag is omitted; differences per entry point are listed
after the table.

| Argument | Library default | Meaning |
|---|---|---|
| `--epochs N` | 5 | Training epochs per fold |
| `--batch-size N` | 32 | Windows per optimizer step |
| `--learning-rate X` | 1e-4 | Adam learning rate |
| `--input-len N` | 96 | Input steps \(T\) |
| `--horizon N` | 96 | Forecast steps \(P\) |
| `--stride N` | input + horizon | Training-window spacing; evaluation uses \(\max(\text{stride},P)\) |
| `--decimation N` | 1 | Low-pass filter, then keep every Nth sample (before pooling) |
| `--pool N` | 1 | Average N samples into one step, within each pass. 1000 means 1 s. |
| `--image-lag {0,1}` | 0 | 0: the pass's own photo (paper). 1: the previous pass's photo (protocol). |
| `--trim-pass` / `--no-trim-pass` | on | Remove each pass's entry and exit transients (`EDA.md` §5) |
| `--label-column {image_label,tool_label}` | `image_label` | Which label file column is the class target |
| `--dedup-identical-forces {True,False}` | True | Merge bit-identical force recordings within a tool/run, keeping the most-worn annotation (True: 378 passes), or keep all 512 (False). A bare flag means True. Aliases: `--deduplicate-identical-forces [VALUE]`; `--keep-identical-forces` and `--no-deduplicate-identical-forces` both mean False. |
| `--eval-dedup-identical-forces {True,False}` | same as training | Duplicate handling for the validation and test data only. Set `True` to score a `--dedup-identical-forces False` model on the same deduplicated windows as a default run (§14). Training entry points only; `inspect` and `task_diagnostics.py` do not train. |
| `--image-encoder {efficientnet_b0,compact}` | `efficientnet_b0` | Photo backbone; `compact` is a small CNN for CPU tests |
| `--d-model N` | 32 | Model width \(d\) |
| `--prediction-head {direct,paper_literal,lead_conditioned}` | `direct` | Per-lead weights / one output repeated over all leads (printed Eq. 21) / learned lead embedding |
| `--device` | `cpu` | `cpu`, `cuda`, or `cuda:1` for a second GPU |
| `--seed N` | 42 | Random seed (fold \(i\) uses seed + \(i\)) |
| `--max-train-windows N`, `--max-eval-windows N` | none | Random caps for smoke tests only |
| `--select-best` / `--no-select-best` | on | Test the lowest-validation-loss epoch, or the last epoch |
| `--accumulation-steps N` | 1 | Sum gradients over N batches per optimizer step |
| `--num-workers N` | 0 | Background data-loading processes (use 0 on Windows) |
| `--resume` / `--no-resume` | on | Continue interrupted folds and reuse finished ones |
| `--ablation` | `none` | One of the 9 Fig. 13 variants (§13) |
| `--modality-dropout X` | 0 | Training probability of masking one input (half sensor, half photo) |
| `--pretrained` | off | ImageNet-pretrained EfficientNet (downloads weights once) |
| `--freeze-bn` | off | Keep EfficientNet's BatchNorm statistics fixed |
| `--als-mode {k_colnorm,k_sum,time}` | `k_colnorm` | Landmark weighting: printed Eq. 10 normalized / printed sums / softmax over time |
| `--private-attn {plain,residual_ln}` | `plain` | Private path \(\mathrm{MHA}(P)\) as printed, or with residual + LayerNorm |
| `--exact-split` | off | Shared part \(cS\) so that \(Z=P+S\) exactly (Eq. 7) |
| `--share-c2f` | off | One set of C2F weights for every round |
| `--pre-attn` | off | Extra self-attention before the split (not in the paper) |
| `--decomp-mode {branch,literal}` | branch | RTD: separate trend/seasonal FFNs (Fig. 5) or the printed sum (which cancels) |
| `--rtd-layers N` | 4 | MRTE depth \(L\) |
| `--rtd-stride N` | 3 | RTD stride and kernel \(s_l\) |
| `--rtd-rank N` | 8 | RTD trend rank \(r\) |
| `--gat-layers N` | 4 | Photo GAT layers |
| `--gat-heads N` | 4 | Heads in the GAT and C2F attention; must divide `--d-model` |
| `--fusion-layers N` | 2 | C2F rounds \(n\) |
| `--landmarks N` | 4 | Landmarks \(k\) |
| `--fusion-rank N` | 8 | C2F bottleneck width |
| `--dropout X` | 0.1 | Model dropout |

Differences per entry point:

| Entry point | Differences from the library defaults |
|---|---|
| `omnifuser train` | Adds `--test-tool` (default `T10`) and `--output-dir` (default `experiment_results/train`). Duplicate handling: `--dedup-identical-forces {True,False}` as in the table above. |
| `scripts/train_lightning.py` | `--data-dir` and `--output-dir` are **required**; `--device` defaults to `cuda`. Duplicate handling: `--dedup-identical-forces {True,False}` as in the table above. |
| `scripts/train_local.py` | Defaults: `--epochs 1`, `--batch-size 1`, `--device cpu`, output `experiment_results/validation/local-smoke`. Windows are capped at 2/1 unless `--full`. It does not expose `--learning-rate`, `--input-len`, `--decimation`, `--label-column`, `--d-model` or `--seed` (library defaults apply). |
| `scripts/run_experiment_suite.py` | Defaults to the protocol: `--epochs 30 --stride 24 --pool 1000 --image-lag 1 --device cuda`. It also takes `--study`, `--run-id`, `--output-root` and `--modality-dropout-rates` (§13). The study `deduplication` compares both duplicate settings (§14). |

Invalid combinations stop before training with a clear error. Examples: a
head count that does not divide the width, dropout ≥ 1, or `--pretrained`
with `compact`.

### Other commands

| Command | Arguments |
|---|---|
| `omnifuser inspect` | `--data-dir`, `--label-column`, `--decimation`, `--pool`, `--input-len` (96), `--horizon` (96), `--stride` (input + horizon), `--trim-pass`/`--no-trim-pass`, `--dedup-identical-forces {True,False}`. It prints passes, classes, lengths, trimming, duplicates and window counts as JSON. |
| `scripts/eda.py` | `--data-dir`, `--out` (default `experiment_results/eda`), `--fs` sampling rate in Hz (1000), `--pool` (1000), `--input-len` (96), `--horizon` (96), `--stride` (24), `--difficulty-pools` (1 10 100 300 1000), `--windows-per-tool` (5000, a cap for the difficulty table), `--no-figures` |
| `scripts/compare_with_paper.py RUN_DIR` | `--by {mean_over_folds,pooled_by_windows}`, `--average {macro,weighted}`, `--json FILE` |
| `scripts/model_summary.py` | `--image-encoder`, `--input-len` (96), `--horizon` (96), `--d-model` (32), `--rounds` (2), `--prediction-head` (`direct`), `--share-c2f`, `--batch-size` (2, for the shape probe), `--check` (verify the parameter total) |
| `scripts/task_diagnostics.py` | `--data-dir`, `--pools` (1 10 100 1000), `--horizons` (12 24 96), `--input-len` (96), `--fs` (1000), `--max-windows-per-tool` (20000), `--trim-pass`/`--no-trim-pass`, `--dedup-identical-forces {True,False}`, `--out FILE`. It prints trivial-baseline difficulty per pool and horizon (descriptive only). |
| `python -m pytest -o addopts= -q tests` | Runs all tests. Add `-k <name>` to run matching tests only. |

Every command prints its own help with `--help`, for example
`python scripts/train_lightning.py --help`.

## 17. Results layout and tracked metrics

```text
experiment_results/
  logs/                                   console logs from tee/nohup
  eda/                                    EDA tables, figures, summary
  validation/, pilot/                     smoke and timing runs (not results)
  main/full-model/                        one 10-fold run (layout below)
  <study>/<run-id>/
    suite.json                            settings, per-variant status, notes
    suite-summary.csv                     one row per variant, test mean/std per input mode
    <group>/<variant>/                    one 10-fold run (layout below)

10-fold run folder:
  tracking/run.json                       configuration, data signature, runtime history
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

Metrics are recorded:

- for every training epoch, on the train and validation splits;
- for the selected checkpoint, on the train, validation and test splits, in
  each of the three input modes.

**Classification metrics:**

- accuracy;
- macro and support-weighted precision, recall and F1;
- per-class scores and the confusion matrix;
- one-vs-rest ROC AUC per class and macro;
- all of the above per forecast step.

**Regression metrics:** MAE, MSE and RMSE in force units and z-scored, each
overall, per channel (Fx, Fy, Fz) and per forecast step.

**Test-only extras:**

- cumulative paper-horizon metrics (12…96);
- last-value, zero-forecast and photo-label persistence baselines.

**Training diagnostics:**

- loss components (total, force, class, plus the logged alignment and
  orthogonality diagnostics);
- epoch time;
- peak CUDA memory.

In `cross-validation-results.json`:

| Key | Contents |
|---|---|
| `cv_mean_std` | Fold mean/std of the headline test metrics |
| `paper_horizons_mean_std` | Fold mean/std at each paper horizon |
| `modality_evaluations` | Fold mean/std per input mode and split |
| `out_of_fold`, `out_of_fold_by_mode` | Metrics pooled over all held-out windows |

For dashboards, load `tracking/metrics.csv` and filter by `split`,
`evaluation_mode` and `metric`. Rows with an empty `evaluation_mode` are the
per-epoch training curves.

## 18. Local commands (Windows PowerShell, CPU)

These are quick checks before uploading, not experiments. In PowerShell, the
line-continuation character is a backtick (`` ` ``) instead of `\`.

```powershell
.\.venv\Scripts\python.exe -m pytest -o addopts= -q tests

.\.venv\Scripts\python.exe -m omnifuser.cli train `
  --data-dir Nonastreda_Data --test-tool T10 `
  --output-dir experiment_results\validation\local-smoke `
  --epochs 1 --batch-size 2 --image-encoder compact --device cpu `
  --pool 1000 --image-lag 1 --horizon 96 --stride 24 `
  --max-train-windows 4 --max-eval-windows 2

.\.venv\Scripts\python.exe scripts\run_experiment_suite.py `
  --study missing-modality --run-id local-smoke `
  --output-root experiment_results\validation `
  --epochs 1 --batch-size 2 --image-encoder compact --device cpu `
  --max-train-windows 4 --max-eval-windows 2
```

The capped suite smoke run takes about 3.5 minutes on CPU. Keep
`--num-workers 0` on Windows; worker processes there re-copy the photo cache
and slow loading down.

## 19. Paper experiments not covered

- **Comparisons with other models** (time-series models, CDA/MBT/LMF/TFN
  fused through the C2F slot, foundation models): not implemented.
  Persistence and last-value baselines are reported instead.
- **The paper's halved image-update-frequency control:** not implemented.
- **FLOPs and memory study (Fig. 14):** only parameter counts
  (`model_summary.py`) and per-epoch peak CUDA memory are recorded.
- **Paired t-tests:** not computed, but per-fold values are stored in
  `per_fold` lists, so they can be computed afterwards.
- **Loss weighting:** the paper gives none. Training minimizes force MSE
  plus class cross-entropy at 1:1. Alignment and orthogonality are logged,
  not optimized.

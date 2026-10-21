# OmniFuser training runbook

This runbook takes you from your laptop to a full set of OmniFuser experiments
on Lightning AI. It assumes you have **never used Lightning AI before**. Every
command is explained, every argument is listed in §16, and the steps are meant
to be run in order.

The paper reports results on "Mudestreda" and MATWI. This repository trains on
Nonastreda, so the paper's numbers are reference points, not targets.

Running on the Ruche cluster (Mésocentre Paris-Saclay) instead? Use
[`TRAINING_RUNBOOK_RUCHE.md`](TRAINING_RUNBOOK_RUCHE.md). It runs the same
experiments as SLURM jobs and refers back here for the explanations.

**Contents**

| Part | Sections |
|---|---|
| **Part A: Lightning AI for first-time users** | §1 Concepts · §2 Prepare on your laptop · §3 Create a Studio · §4 Get the code · §5 Upload the data · §6 Install · §7 Switch to a GPU · §8 Terminal basics |
| **Part B: The experiments** | §9 Shared protocol · §10 Step 0: verify · §11 Steps 1-2: smoke test and pilot · §12 Step 3: main 10-fold runs · §13 Steps 4-7: paper studies and the image-encoder study · §14 Optional runs (image source with spectrograms, duplicates, printed-paper readings, heads) |
| **Part C: Operating and reference** | §15 Unattended runs, resuming, downloading, troubleshooting · §16 Complete argument reference · §17 Results layout and metrics · §18 Local Windows commands · §19 Not covered |

## Overview of the experiments

| Step | What it runs | 10-fold CV runs | Output folder |
|---|---|---|---|
| 0 | Tests, GPU check, data inspection, EDA | 0 | `experiment_results/eda/` |
| 1 | Capped one-fold GPU smoke test | 0 (1 fold) | `experiment_results/validation/gpu-smoke/` |
| 2 | Uncapped one-fold pilot (timing) | 0 (1 fold) | `experiment_results/pilot/T10/` |
| 3a | Full model, all 10 folds, duplicates merged (the protocol) | 1 | `experiment_results/main/full-model/` |
| 3b | Full model, all 10 folds, duplicates kept (training and test) | 1 | `experiment_results/main/full-model-no-dedup/` |
| 4 | Paper Fig. 13 component ablations | 9 | `experiment_results/ablations/<run-id>/` |
| 5 | Missing-modality robustness | 2 | `experiment_results/missing-modality/<run-id>/` |
| 6 | Paper Figs. 7-8 parameter sensitivity | 22 | `experiment_results/sensitivity/<run-id>/` |
| (alt) | Steps 4-6 in one unattended suite | 31 | `experiment_results/all/<run-id>/` |
| 7 | Image encoder: trained from scratch, pretrained and fine-tuned, pretrained and frozen | 3 | `experiment_results/image-encoder/<run-id>/` |
| (opt) | Experimental image source: previous-pass photos versus force spectrograms (all with image lag 1) | 3 | `experiment_results/image-source/<run-id>/` |
| (opt) | Duplicate handling: merged versus kept, scored on the same deduplicated test data | 2 | `experiment_results/deduplication/<run-id>/` |
| (opt) | Printed-paper readings of the three remaining deviations | 3 | `experiment_results/paper-literal/<arm>/` |
| (opt) | Prediction-head comparison | 3 | `experiment_results/heads/<head>/` |
| After | Download the results and write the report (§15, [`REPORT_PLAN.md`](REPORT_PLAN.md)) | 0 | `results-*.tar.gz`, `experiment_results/report/` |

A **10-fold CV run** trains 10 models. Each one holds out a different tool for
testing, uses the next tool for validation, and trains on the other 8
(`EDA.md` §8 explains folds). Every run also evaluates each fold three ways:
with both inputs (`full`), with the force signal masked (`sensor_missing`),
and with the photos masked (`image_missing`). Missing-modality evaluation
needs no extra command.

> **Main runs and experiments.** Steps 1-6 all use the protocol:
>
> - **image lag 0:** each pass is shown its own photo, as in the paper;
> - **photos as the image input**, through **an EfficientNet-B0 trained from
>   scratch** (no `--pretrained`);
> - **duplicates merged** (step 3b repeats the full model with them kept).
>
> The experiments change one thing at a time, so each compares directly
> with step 3a:
>
> | Experiment | Variants | Differs from step 3a by |
> |---|---|---|
> | Step 7, image encoder (§13) | from scratch, pretrained and fine-tuned, pretrained and frozen | `--pretrained`, `--freeze-image-encoder` |
> | Image source, optional (§14) | previous-pass photo, spectrogram, pretrained spectrogram | `--image-lag 1`, `--image-source spectrogram` |
>
> The force spectrograms are an optional experiment only: no main run uses
> them.

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

The Studio clones the **`main`** branch from GitHub (§4), so your work must be
committed on `main` and pushed. Anything only on your laptop, or only on
another branch, is missing in the Studio. In your local repository
(PowerShell or Git Bash):

```bash
git switch main                  # work on main, the branch the Studio clones
git status                       # what has changed
git add -A                       # stage everything that is not git-ignored
git commit -m "Prepare for Lightning AI training"
git push                         # upload main to GitHub
git status -sb                   # first line must read: ## main...origin/main
git log -1 --oneline origin/main # the exact commit the Studio will clone
```

- If `git commit` reports `nothing to commit`, everything is already
  committed; continue with `git push`.
- `## main...origin/main` with nothing after it means your `main` and
  GitHub's `main` are identical. `[ahead 1]` means a commit is not pushed
  yet, so run `git push` again.
- **If you made changes on another branch**, bring them into `main` first:

  ```bash
  git switch main
  git pull                              # get GitHub's latest main
  git merge --ff-only your-branch-name  # move main forward to your branch
  git push
  ```

  Replace `your-branch-name` with the branch's real name, for example
  `Lightning-AI-training`; `git branch` lists your branches. `--ff-only`
  only moves `main` forward along your branch's history. If `main` has
  commits your branch lacks, it refuses instead of creating a merge; then run
  `git merge your-branch-name` and check the result before pushing.

`Nonastreda_Data/` and `experiment_results/` are git-ignored, so the data
does not go through GitHub. Pack it into one archive for upload. In
PowerShell, from the repository folder:

```powershell
tar -czf Nonastreda_Data.tar.gz Nonastreda_Data
```

- `tar` is built into Windows 10 and 11. `-c` creates an archive, `-z`
  compresses it, and `-f` names the file. The result is about 383 MB.
- The archive name is git-ignored, so a later `git add -A` cannot commit it.
- Do **not** use the Explorer "Compress" menu or PowerShell's
  `Compress-Archive`. Windows PowerShell writes paths inside the zip with
  backslashes, and on the Studio's Linux they unpack into wrongly named
  files instead of a `tool/` folder.

## 3. Create a Studio

1. Sign in at lightning.ai and create a new **Studio**. Start with a
   **CPU** machine; setup does not need a GPU.
2. When it opens, open a terminal from the editor's menu (Terminal → New
   Terminal). The prompt starts in `/teamspace/studios/this_studio`.
3. Check what you have:

   ```bash
   pwd
   python --version
   nproc
   ```

   `pwd` must print `/teamspace/studios/this_studio`, `python --version`
   must show 3.10 or newer, and `nproc` prints the number of CPU cores, the
   upper limit for `--num-workers` (§9).

## 4. Get the code

```bash
cd /teamspace/studios/this_studio
git clone https://github.com/taanakoum/OmniFuser.git OmniFuser
cd OmniFuser
git log -1 --oneline
```

The last line must show the commit you pushed in §2.

- `git clone <url> OmniFuser` downloads the repository into a new folder
  `OmniFuser`, on its default branch `main`, which holds the complete
  training code.
- To use another branch instead, add `-b <branch>`, for example
  `git clone -b Lightning-AI-training <url> OmniFuser`.
- If the repository is private, GitHub asks for a username and a password.
  Use a **personal access token** as the password (GitHub → Settings →
  Developer settings → Personal access tokens).
- To pick up changes pushed to `main` later, run `git pull` inside
  `OmniFuser/`, then check `git log -1 --oneline` against your laptop's
  `git log -1 --oneline origin/main`.
- **Do not pull code changes in the middle of a study.** A resume checks the
  settings and the data, not the code. A run that continues after a pull
  would silently mix two code versions in one result folder. Pull only
  between studies, or start new output folders after a pull.

## 5. Upload the data

1. Drag `Nonastreda_Data.tar.gz` from your computer onto the `OmniFuser`
   folder in the Studio's file explorer and wait for the upload to finish.
2. Unpack it in the terminal:

   ```bash
   cd /teamspace/studios/this_studio/OmniFuser
   tar -xzf Nonastreda_Data.tar.gz && rm Nonastreda_Data.tar.gz
   ls Nonastreda_Data
   ls Nonastreda_Data/tool | wc -l
   ```

   The first `ls` must list `forces_xyz_raw.mat  labels.csv  labels_reg.csv
   tool`, and the count must be `512`.

   `tar -xzf` extracts the archive (`-x` extract, `-z` decompress, `-f` the
   file), and `rm` then deletes it to save space. If `tar` is unavailable,
   `python -m tarfile -e Nonastreda_Data.tar.gz .` does the same.

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
   nvidia-smi
   python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA')"
   ```

   `nvidia-smi` shows the GPU name, its memory and running processes. The
   second command must print `True` and the GPU name. If it prints
   `False`, the installed PyTorch has no CUDA support: fix that before
   training.

## 8. Terminal basics used below

**Your shell.** The Studio terminal (the `⚡` prompt) may run **zsh** rather
than bash. Every command in this runbook works in both. Two consequences:

- The protocol is stored as an array and inserted with `"${PROTOCOL[@]}"`
  (§9). A plain text variable inserted as `$PROTOCOL` is **not** split into
  words by zsh, so the program receives the whole protocol as one argument
  and stops with `unrecognized arguments: --data-dir Nonastreda_Data ...`.
- Commands run in the Studio contain no `#` comments, because zsh only
  accepts them when the `interactivecomments` option is set. (The laptop
  commands in §2 have comments; PowerShell and Git Bash accept them.)

To check which shell you have: `echo $0` prints `zsh` or `bash`.

| Syntax | Meaning |
|---|---|
| `\` at the end of a line | The command continues on the next line. Nothing may follow the `\`, not even a space. |
| `NAME=(a b c)` then `"${NAME[@]}"` | Store a list of words in a shell array, then insert all of them, as separate arguments, into later commands. Arrays last for one terminal session; redefine them in every new terminal. |
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
PROTOCOL=(
  --data-dir Nonastreda_Data --device cuda
  --epochs 30 --batch-size 32 --learning-rate 1e-4
  --input-len 96 --horizon 96 --stride 24
  --pool 1000 --image-lag 0 --prediction-head direct
  --image-encoder efficientnet_b0 --seed 42 --num-workers 4
)
mkdir -p experiment_results/logs
printf '%s ' "${PROTOCOL[@]}"; echo
```

`PROTOCOL=( ... )` stores the arguments as a shell **array**: one entry per
word, with no `\` needed inside the parentheses. The `printf` line prints
them back on one line; check that it starts with `--data-dir Nonastreda_Data`
and ends with `--num-workers 4`.

Every training command below starts with `"${PROTOCOL[@]}"`, which inserts
every entry as a separate argument in both bash and zsh. Type it exactly so,
with the quotes, the braces and `[@]`:

- plain `$PROTOCOL` passes only `--data-dir` in bash (error:
  `argument --data-dir: expected one argument`);
- a protocol defined as text, `PROTOCOL="..."`, arrives as one argument in
  zsh (error: `unrecognized arguments: --data-dir Nonastreda_Data ...`).

All experiments therefore share the same data, windows and optimizer. When a
command repeats a flag after `"${PROTOCOL[@]}"`, the **later value wins**;
that is how individual steps override it (for example `--epochs 1`).

| Argument | Value | Meaning |
|---|---|---|
| `--data-dir` | `Nonastreda_Data` | Folder with the MAT file, the two CSVs, and `tool/` |
| `--device` | `cuda` | Train on the GPU (`cpu` for local tests) |
| `--epochs` | 30 | Passes over the training windows per fold. The paper gives no number, so 30 is a project choice. |
| `--batch-size` | 32 | Windows per optimizer step (paper value) |
| `--learning-rate` | 1e-4 | Adam learning rate (paper value) |
| `--input-len` | 96 | Input steps per window (\(T\)) |
| `--horizon` | 96 | Steps to forecast (\(P\)). Paper horizons 12…96 are scored cumulatively from one 96-step forecast. |
| `--stride` | 24 | Steps between consecutive *training* windows: each step appears in up to 8 overlapping windows, giving 1,127-1,213 training windows per fold. Evaluation windows use \(\max(\text{stride},P)=96\), so evaluation targets never overlap and each test step counts once. The paper states no window stride. Its stride \(s_l=3\) is a different one, the downsampling inside MRTE: a model default, varied in the Fig. 8 sensitivity study. |
| `--pool` | 1000 | Average 1000 force samples (1 s) into one step. This matches the paper's difficulty (`EDA.md` §6). |
| `--image-lag` | 0 | Show each pass its own photo, as the paper does. That photo's label already matches 91 % of the target labels (`EDA.md` §7), so classification partly reads the label off the photo. The previous-pass photo (1) is an experimental run (§14). |
| `--prediction-head` | `direct` | Separate output weights per forecast step (closest usable reading of Eqs. 21-22) |
| `--image-encoder` | `efficientnet_b0` | The paper's photo backbone (`compact` is for quick CPU tests) |
| `--seed` | 42 | Random seed. Fold \(i\) uses seed + \(i\). |
| `--num-workers` | 4 | Background processes preparing batches. Use at most the `nproc` count; 0 loads in the main process. |

`PROTOCOL` has no `--pretrained`, so every main run trains the image
encoder from scratch; §14 has the pretrained comparison.

**Duplicate handling is not in `PROTOCOL`.** You choose it in each command
instead:

- `scripts/train_lightning.py` (step 3, the single runs in §13 and §14) **requires**
  `--dedup-identical-forces True` or `False`. Without it, the script stops
  before loading any data with
  `error: the following argument is required: --dedup-identical-forces True|False`.
- `True` merges the 134 bit-identical copied recordings, keeping the
  most-worn label: 378 passes instead of 512 (`EDA.md` §3). `False` keeps all
  512.
- `--eval-dedup-identical-forces True|False` sets the same for the
  validation and test data. It is optional; omitted, it follows
  `--dedup-identical-forces`.
- `omnifuser train` (steps 1-2) and `run_experiment_suite.py` (steps 4-6)
  default to `True`, the protocol value, so their commands below leave it
  out. The suite's `deduplication` study sets it per variant (§14).

The architecture defaults equal the paper's values, so they are not in
`PROTOCOL`: \(d=32\), 4 MRTE layers with stride 3 and rank 8, 4 GAT layers ×
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
| `pytest ... tests` | Runs the 127 tests. `-o addopts=` clears the default options and `-q` keeps the output short. | `127 passed` |
| `python -c "import torch ..."` | Prints the torch version, whether CUDA works, and the GPU name | `True` and your GPU |
| `omnifuser inspect` | Loads and validates the whole dataset with the protocol's pooling and windows, and prints a JSON report | `"records": 378`, `"tools": 10`, class counts 147/118/113, `"exact_duplicate_force_rows_remaining": 0` |
| `model_summary.py --check` | Builds the default model, prints per-block parameter counts and shapes, and checks the total | `4,183,600 ... (match)` |
| `eda.py` | The exploratory data analysis: tables, a JSON summary and 12 figures in `experiment_results/eda/`. It takes several minutes and about 3 GB of RAM. | Explained in `EDA.md` |

## 11. Steps 1-2: GPU smoke test and timing pilot

### Step 1: capped smoke test (a few minutes)

This checks the real data path, CUDA training, all three evaluation modes,
tracking, and artifact writing. Its metrics are not results.

```bash
omnifuser train "${PROTOCOL[@]}" --epochs 1 --test-tool T10 \
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
omnifuser train "${PROTOCOL[@]}" --epochs 3 --test-tool T10 \
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

> **Steps 3-6 run for hours.** The commands below are written in the
> foreground `... 2>&1 | tee LOG` form, which stops if the terminal or browser
> tab closes. For an unattended run, change any of them as follows:
>
> - put `nohup` in front;
> - replace `2>&1 | tee LOG` with `> LOG 2>&1 &`;
> - then follow the log with `tail -f LOG`.
>
> Step 3a as a background run:
>
> ```bash
> nohup python scripts/train_lightning.py "${PROTOCOL[@]}" \
>   --dedup-identical-forces True --eval-dedup-identical-forces True \
>   --output-dir experiment_results/main/full-model \
>   > experiment_results/logs/main-full-model.log 2>&1 &
> tail -f experiment_results/logs/main-full-model.log
> ```
>
> Ctrl+C stops `tail`, not the run. Alternatively, start `tmux` first (§8)
> and run the commands unchanged.

> **Results from earlier code versions cannot be continued.** Runs started
> before 2026-10-21 used different settings (image lag 1 before 2026-10-18;
> no `freeze_image_encoder` field before 2026-10-21). Rerunning into such a
> folder is refused (`different configuration`), so move it aside first,
> for example:
>
> ```bash
> mkdir -p experiment_results/archive
> mv experiment_results/main experiment_results/archive/main-before-2026-10-21
> ```
>
> For older suites, use a new `--run-id` such as `paper-v2`, or move their
> folders aside the same way.

Step 3 trains the full model twice, once per duplicate setting. Both
commands state `--dedup-identical-forces` explicitly; `train_lightning.py`
refuses to start without it (§9). The two runs write to separate folders, so
they can run one after the other or in two terminals on a GPU with enough
memory.

### Step 3a: duplicates merged (the protocol, main result)

```bash
python scripts/train_lightning.py "${PROTOCOL[@]}" \
  --dedup-identical-forces True --eval-dedup-identical-forces True \
  --output-dir experiment_results/main/full-model \
  2>&1 | tee experiment_results/logs/main-full-model.log
python scripts/compare_with_paper.py experiment_results/main/full-model \
  --json experiment_results/main/full-model/paper-comparison.json
```

Training, validation and test use the 378 distinct passes (1,462 windows
over all tools). This is the run the report's main results, the studies of
steps 4-6, and `REPORT_PLAN.md` refer to.

### Step 3b: duplicates kept (training and test)

```bash
python scripts/train_lightning.py "${PROTOCOL[@]}" \
  --dedup-identical-forces False --eval-dedup-identical-forces False \
  --output-dir experiment_results/main/full-model-no-dedup \
  2>&1 | tee experiment_results/logs/main-full-model-no-dedup.log
python scripts/compare_with_paper.py experiment_results/main/full-model-no-dedup \
  --json experiment_results/main/full-model-no-dedup/paper-comparison.json
```

Training, validation and test use all 512 passes (2,012 windows). Read its
numbers with care:

- **Its test windows differ from step 3a's.** Each test tool also contains
  its copied passes, so 3a and 3b differ in both the training data and the
  test data. The copies also make the test easier: a 96-step input window
  that ends in B3 contains nearly all of B3, which is exactly the signal B4
  repeats (§14).
- **To isolate the effect of the training data**, score the duplicate-kept
  model on 3a's test windows instead: use
  `--eval-dedup-identical-forces True` (the `dedup-false` arm of the §14
  study, or the single run shown there).
- `compare_with_paper.py` prints both settings and pass counts in its header
  (`dedup train/eval=False/False, passes train/eval=512/512`), so the two
  tables cannot be confused.

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

## 13. Steps 4-7: the paper's studies and the image-encoder study

All four use `scripts/run_experiment_suite.py`. It runs each **variant** as
its own full 10-fold run, then writes a manifest and a comparison table. Its
own arguments:

| Argument | Meaning |
|---|---|
| `--study` | `ablations`, `sensitivity`, `missing-modality`, `all` (the three paper studies), `image-encoder` (step 7), or the optional `image-source` and `deduplication` (§14) |
| `--run-id` | Folder name for this suite, e.g. `paper-v1`. Reuse it to resume; use a new one for a new suite. Without it, a UTC timestamp is used. |
| `--output-root` | Parent folder. The suite writes to `<output-root>/<study>/<run-id>/`. |
| `--modality-dropout-rates` | Training dropout rates for the missing-modality study (default `0.2`) |

All training flags from §9 and §16 apply to every variant. The suite's own
defaults already equal the protocol (`--pool 1000 --image-lag 0`, 30 epochs,
stride 24, CUDA, `--dedup-identical-forces True`), so the protocol is passed
only for uniformity. Every variant is therefore comparable with step 3a.

### Step 4: component ablations (paper Fig. 13), 9 runs

```bash
python scripts/run_experiment_suite.py --study ablations --run-id paper-v1 \
  --output-root experiment_results "${PROTOCOL[@]}" \
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
python scripts/train_lightning.py "${PROTOCOL[@]}" --dedup-identical-forces True \
  --ablation without_als \
  --output-dir experiment_results/ablations/single/without-als \
  2>&1 | tee experiment_results/logs/ablation-single-without-als.log
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
  --output-root experiment_results "${PROTOCOL[@]}" \
  2>&1 | tee experiment_results/logs/missing-modality-paper-v1.log
```

To sweep several rates, use a new `--run-id`:

```bash
python scripts/run_experiment_suite.py --study missing-modality --run-id rates-v1 \
  --modality-dropout-rates 0.1 0.2 0.3 --output-root experiment_results "${PROTOCOL[@]}" \
  2>&1 | tee experiment_results/logs/missing-modality-rates-v1.log
```

To read the results, compare the `full.*`, `sensor_missing.*` and
`image_missing.*` columns within each row of `suite-summary.csv`. Graceful
degradation means the masked scores drop toward, but not below, the
persistence baselines.

### Step 6: parameter sensitivity (paper Figs. 7-8), 22 runs

```bash
python scripts/run_experiment_suite.py --study sensitivity --run-id paper-v1 \
  --output-root experiment_results "${PROTOCOL[@]}" \
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
  --output-root experiment_results "${PROTOCOL[@]}" \
  > experiment_results/logs/all-paper-v1.log 2>&1 &
tail -f experiment_results/logs/all-paper-v1.log
```

### Step 7: image encoder, 3 runs

The paper does not say whether its EfficientNet was pretrained. The main runs
train it from scratch; this study measures what ImageNet weights change:

| Variant | Encoder weights | What trains |
|---|---|---|
| `from-scratch` | Random initialization (the main-run configuration) | Everything |
| `pretrained-finetuned` | ImageNet (torchvision `IMAGENET1K_V1`) | Everything, including the backbone |
| `pretrained-frozen` | ImageNet, kept fixed | The projection after the backbone and the rest of the model; the backbone's weights and BatchNorm statistics stay fixed |

Photos are normalized with the ImageNet mean and standard deviation in every
run, which is the input the pretrained weights expect. Download the weights
(about 20 MB, cached in `~/.cache/torch`) before the long run, so a network
problem shows up immediately:

```bash
python -c "from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0; efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1); print('weights cached')"
python scripts/run_experiment_suite.py --study image-encoder --run-id encoder-v1 \
  --output-root experiment_results "${PROTOCOL[@]}" \
  2>&1 | tee experiment_results/logs/image-encoder-encoder-v1.log
```

`from-scratch` repeats step 3a's configuration inside the study, so the three
variants sit side by side in one `suite-summary.csv`. Compare the `full.*`
columns between rows, and the gap between `full.*` and `image_missing.*`
within a row: it shows how much each encoder's photo features contribute.

## 14. Optional runs

### Experimental: image source (previous-pass photos versus force spectrograms)

> **An optional experiment, not a main run.** It changes what the image
> branch sees; everything else is the protocol.

All three variants use **image lag 1**, the previous pass's image, so they
differ only in the image source:

| Variant | Image branch input | Encoder |
|---|---|---|
| `photo-previous-pass` | The previous pass's photo | From scratch |
| `spectrogram` | A spectrogram of the previous pass's force | From scratch |
| `spectrogram-pretrained` | The same | ImageNet weights, fine-tuned |

How a spectrogram is made (`omnifuser/spectrogram.py`):

- the pass's **raw 1 kHz force** (before the 1 s pooling, which removes
  everything faster than 1 Hz) has its mean removed per axis;
- a short-time Fourier transform (Hann window of 1024 samples, hop 256)
  gives the log power per frequency and time;
- the three force axes Fx, Fy, Fz become the three image channels, resized
  to the photo size (128 × 384), so the image branch is unchanged;
- each image is standardized on its own, so no statistic is fitted across
  tools.

Why image lag 1 is required: a pass's spectrogram covers the whole pass,
including force samples that fall inside the forecast window of a window
ending mid-pass. With lag 1 every image comes from a pass that ended before
the window's input does. A tool's first pass has no previous pass and shows
no image (zeros) and no label. `--image-source spectrogram` with image lag 0
is refused.

First a capped one-fold smoke test of the spectrogram path on the real data
(a few minutes; its numbers are not results), then the study:

```bash
omnifuser train "${PROTOCOL[@]}" --image-lag 1 --image-source spectrogram   --epochs 1 --test-tool T10 --output-dir experiment_results/validation/spectrogram-smoke   --max-train-windows 64 --max-eval-windows 32
python scripts/run_experiment_suite.py --study image-source --run-id source-v1 \
  --output-root experiment_results "${PROTOCOL[@]}" \
  2>&1 | tee experiment_results/logs/image-source-source-v1.log
```

How to read it:

- `photo-previous-pass` against step 3a: how much the same-pass photo (lag 0)
  helps. Copying the photo's label gets about 91 % of targets right at lag 0
  and about 77 % at lag 1 (`EDA.md` §7), so judge each run against its own
  `persistence` baseline in `compare_with_paper.py`.
- `spectrogram` against `photo-previous-pass`: whether the force spectrum
  carries more wear information than the photo, at the same timing.
- `spectrogram-pretrained` against `spectrogram`: whether ImageNet features
  transfer to spectrograms.
- STFT settings: `--spectrogram-window` and `--spectrogram-hop` (in raw
  samples). The force is sampled at 1 kHz, so frequencies up to 500 Hz are
  visible.

### Duplicate handling: merged versus kept

Nonastreda has 134 force recordings that are bit-identical copies of another
pass in the same run, mostly block B4 repeating B3 (`EDA.md` §3). In 24 of
these groups the copies carry different wear labels. Two independent options
control them:

| Option | Applies to | Values |
|---|---|---|
| `--dedup-identical-forces {True,False}` | The **training** data, and by default everything else | `True`: one pass per group, 378 passes. `False`: all 512. Required by `train_lightning.py`; the other commands default to `True`. |
| `--eval-dedup-identical-forces {True,False}` | The **validation and test** data | Omitted: same as `--dedup-identical-forces`. Set it to fix the evaluation data independently. |

How the training setting changes a run (the evaluation setting has the same
effects on validation and test data):

| Stage | `True` (protocol) | `False` |
|---|---|---|
| Passes | 378 | 512: the 134 copies are kept, mostly B4 repeating B3 |
| Labels | One label per signal (the most-worn) | In 24 groups the same force signal has two labels: label noise |
| Tool timelines | Each pass once | Copied passes appear twice, back to back |
| Folds | Unchanged: tool membership only | Unchanged |
| Force normalizer (8 training tools) | From deduplicated data | Shifts slightly, because copies count twice |
| Windows (protocol, all tools) | 1,462 | 2,012: more training steps per epoch |
| Photo (protocol `--image-lag 0`) | Each pass its own photo | B3 and B4 keep their own photos and labels beside the same force signal |
| Model | Unchanged | Unchanged: the setting is not part of the model config |
| Saved and checked | `fold-Tn.json`, `tracking/run.json` and the resume check record both settings and both pass counts, so a changed setting retrains instead of reusing folds | Same |

The library function `load_nonastreda` also defaults to `True`, but that
default only applies when the setting is not passed. Every command passes the
value you choose, and `train_lightning.py` has no default at all.
Step 3 already contains the two full single-setting runs (3a: `True`/`True`,
3b: `False`/`False`). `eda.py` always describes the deduplicated data, plus the
raw 512 passes side by side.

The comparison study trains both settings with everything else identical,
and scores both on the **same deduplicated validation and test windows**:

```bash
python scripts/run_experiment_suite.py --study deduplication --run-id dedup-v1 \
  --output-root experiment_results "${PROTOCOL[@]}" \
  2>&1 | tee experiment_results/logs/deduplication-dedup-v1.log
```

| Variant | Training data | Validation and test data |
|---|---|---|
| `dedup-true` | 378 passes | 378 passes |
| `dedup-false` | 512 passes | 378 passes |

Because the test windows are identical, any metric difference comes from the
training data alone. The study sets `--dedup-identical-forces` per variant,
so the suite command does not pass it. To score both arms on non-deduplicated
data instead, add `--eval-dedup-identical-forces False` at the end of the
suite command.

`suite-summary.csv` gets one row per setting. Besides the usual metrics it
has these columns:

| Column | Contents |
|---|---|
| `dedup_identical_forces`, `eval_dedup_identical_forces` | The training and evaluation settings |
| `retained_passes`, `evaluation_passes` | Passes in the training data and in the evaluation data |
| `train_windows.mean`, `test_windows.mean` | Window counts per fold. `test_windows.mean` is equal for both arms when evaluation is deduplicated. |

To run the duplicate-kept training scored on the deduplicated test windows
outside the suite (its partner with `True` is step 3a):

```bash
python scripts/train_lightning.py "${PROTOCOL[@]}" \
  --dedup-identical-forces False --eval-dedup-identical-forces True \
  --output-dir experiment_results/deduplication/single/dedup-false-eval-true \
  2>&1 | tee experiment_results/logs/dedup-false-eval-true.log
```

The whole protocol without deduplication (`False` for training and test) is
step 3b; there is no need to run it again here.

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
  test keep the duplicates (both options `False`), each
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
  python scripts/train_lightning.py "${PROTOCOL[@]}" --dedup-identical-forces True \
    --prediction-head $head \
    --output-dir experiment_results/heads/$head \
    2>&1 | tee experiment_results/logs/head-$head.log
done
```

`for head in a b c; do ...; done` repeats the command once per value, with
`$head` replaced each time.

### Printed-paper readings of the three remaining deviations

Three defaults depart from the printed paper (`PAPER_IMPLEMENTATION_ALIGNMENT.md`
explains each). The photo association is no longer one of them: the
protocol uses the paper's same-pass photo. These runs measure what each
departure changes:

```bash
python scripts/train_lightning.py "${PROTOCOL[@]}" --dedup-identical-forces True --prediction-head paper_literal \
  --output-dir experiment_results/paper-literal/repeated-head \
  2>&1 | tee experiment_results/logs/paper-literal-repeated-head.log
python scripts/train_lightning.py "${PROTOCOL[@]}" --dedup-identical-forces True --decomp-mode literal \
  --output-dir experiment_results/paper-literal/eq3-literal \
  2>&1 | tee experiment_results/logs/paper-literal-eq3-literal.log
python scripts/train_lightning.py "${PROTOCOL[@]}" --dedup-identical-forces True --als-mode k_sum \
  --output-dir experiment_results/paper-literal/eq10-unnormalized \
  2>&1 | tee experiment_results/logs/paper-literal-eq10-unnormalized.log
```

Two other variants are available:

- `--exact-split`: the Eq. 7 shared/private reading.
- `--private-attn residual_ln --als-mode time`: the defaults used before
  2026-10-14.

---

# Part C: Operating and reference

## 15. Unattended runs, resuming, and downloading results

- **Keep runs alive.** Use `nohup ... &` (§12) or `tmux` (§8) for anything
  longer than a few minutes. Plain `| tee` commands stop when their terminal
  closes.
- **Watch progress.** Three separate tools; run them one at a time. The
  first prints once and returns. The other two keep running until you press
  Ctrl+C, which stops only the watching, never the training.

  ```bash
  python -c 'import json, sys; m = json.load(open(sys.argv[1])); print("suite:", m["status"]); [print(" ", k, v["status"]) for k, v in m["runs"].items()]' experiment_results/ablations/paper-v1/suite.json
  tail -f experiment_results/logs/ablations-paper-v1.log
  watch -n 5 nvidia-smi
  ```

  - `tail -f` follows a log: one `Fold k/10` line per fold, and one
    `=== group/variant ===` line per suite variant. Ctrl+C stops following,
    not the run. Use the log name from your command's `tee` or `nohup`.
  - `watch -n 5 nvidia-smi` refreshes GPU use and memory every 5 s.
  - The `python -c` line prints a suite's overall status and each variant
    as `completed`, `running` or `failed`. Replace `ablations/paper-v1` with
    your `<study>/<run-id>`.
  - `cross-validation-progress.json` in each run folder lists the folds
    finished so far.
- **Resume after any interruption**, including a sleep, a crash or a stopped
  Studio: rerun the **exact same command**, with the same `--run-id` for
  suites.
  - Finished variants are skipped.
  - Finished folds are reused.
  - An interrupted fold continues from its last finished epoch
    (`fold-Tn-resume.pt`).
  - The rerun's log shows `T1: reusing a completed fold with matching
    settings` for each finished fold, and has no `===` line for finished
    variants.
  - The rerun **replaces** the console log (`> file` and `tee file`
    overwrite it). The metrics are safe: `tracking/metrics.jsonl` and
    `metrics.csv` append across attempts. To keep the old console log,
    rename it first, e.g. `mv experiment_results/logs/all-paper-v1.log
    experiment_results/logs/all-paper-v1-attempt1.log`.
- **Changed settings are refused.** Different settings stop with an error
  instead of mixing configurations. A different GPU type or library version
  is allowed and noted in `tracking/run.json` under `runtime_history`.
- **Do not modify the data mid-study.** Each input file's path, size and
  modification time are recorded and checked. Editing or re-uploading
  `Nonastreda_Data/` during a study forces retraining.
- **Download results.** Pack them first, then right-click the archive in
  the file explorer and choose **Download**:

  ```bash
  tar -czf results-no-checkpoints.tar.gz --exclude='*.pt' experiment_results
  find experiment_results -name '*.pt' | tar -czf results-checkpoints.tar.gz -T -
  ```

  - The first archive holds everything except the model checkpoints:
    metrics, predictions, logs and figures. It is all the report needs.
  - The second holds only the checkpoints (`find` lists them, and `-T -`
    makes `tar` read that list). Each EfficientNet checkpoint is about
    17 MB, so one 10-fold run is about 170 MB.
  - On Windows, unpack with `tar -xzf results-no-checkpoints.tar.gz` in
    PowerShell.
- **Write up the results.** Follow [`REPORT_PLAN.md`](REPORT_PLAN.md): its §1 lists what to download
  and record before stopping the Studio, and its snippets build every report table and figure.
- **Stop the Studio** when finished, so GPU time is no longer billed. Files
  in `/teamspace/studios/this_studio` remain.

### Troubleshooting

| You see | Cause | Fix |
|---|---|---|
| `unrecognized arguments: --data-dir Nonastreda_Data --device cuda ...` | zsh received the protocol as one argument: it was defined as text (`PROTOCOL="..."`) or inserted as `$PROTOCOL` | Define it with the §9 block (`PROTOCOL=( ... )`) and insert it as `"${PROTOCOL[@]}"` |
| `argument --data-dir: expected one argument` | bash inserted only the array's first entry: `$PROTOCOL` instead of `"${PROTOCOL[@]}"` | Write `"${PROTOCOL[@]}"` |
| `train_lightning.py`: `the following arguments are required: --data-dir`. `omnifuser train`: no error, but it runs with the library defaults: CPU, 1 ms steps (`pool=1`), 5 epochs | `PROTOCOL` is empty: a new terminal, or the §9 block was not run. `omnifuser train` has defaults for every option, so it starts anyway. | Stop the run with Ctrl+C, run the §9 block in this terminal (its `printf` line must print the arguments), then rerun the command |
| `experiment output directory belongs to a different configuration` | You reran into an existing output folder with different settings: any flag, including `--num-workers` | Use the original command exactly, or a new `--output-dir` |
| `cannot resume ...; study settings differ from suite.json` | A suite was rerun with the same `--run-id` but different settings | The original command, or a new `--run-id` |
| `CUDA was requested but is not available` | The Studio is on a CPU machine, or PyTorch lacks CUDA | Switch to a GPU (§7); the §7 check must print `True` |
| `CUDA out of memory` | Batch too large for the GPU | Add `--batch-size 16 --accumulation-steps 2` (same effective batch) |
| `10-fold tool CV requires 10 tools; found 0` or `No such file or directory: ...labels.csv` | Wrong data folder, or the data was not unpacked | Run from `OmniFuser/`; `ls Nonastreda_Data/tool \| wc -l` must print 512 (§5) |
| `ModuleNotFoundError: No module named 'omnifuser'` or `omnifuser: command not found` | The project is not installed in this environment | Rerun §6 from inside `OmniFuser/` |
| `DataLoader worker ... killed`, `Bus error`, or a shared-memory error | The data-loading worker processes ran out of shared memory | Add `--num-workers 0` (slower loading, same results); a new output folder is then needed, see the first row |
| `Killed` with no Python error during `eda.py` or training | The machine ran out of RAM (the EDA needs about 3 GB) | Choose a machine with more memory, or close other processes |
| The log stops growing for a long time | Each epoch prints one line (`T1 epoch 3/30: ...`), so a pause much longer than one epoch time is unusual | `tail experiment_results/<run>/cross-validation-progress.json` and `nvidia-smi`; the process is alive if `ps aux \| grep python` lists it |

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
| `--image-lag {0,1}` | 0 | 0: the pass's own image (paper and protocol). 1: the previous pass's image (experiments, §14); a tool's first pass then shows no image. |
| `--trim-pass` / `--no-trim-pass` | on | Remove each pass's entry and exit transients (`EDA.md` §5) |
| `--label-column {image_label,tool_label}` | `image_label` | Which label file column is the class target |
| `--dedup-identical-forces {True,False}` | True; **required** for `train_lightning.py` | Merge bit-identical force recordings within a tool/run, keeping the most-worn annotation (True: 378 passes), or keep all 512 (False). A bare flag means True. Aliases: `--deduplicate-identical-forces [VALUE]`; `--keep-identical-forces` and `--no-deduplicate-identical-forces` both mean False. |
| `--eval-dedup-identical-forces {True,False}` | same as training | Duplicate handling for the validation and test data only. Set `True` to score a `--dedup-identical-forces False` model on the same deduplicated windows as a default run (§14). Training entry points only; `inspect` does not train. |
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
| `--pretrained` | off | Start EfficientNet-B0 from ImageNet weights (downloaded once). Off in all main runs (from scratch); step 7 compares. Photos are ImageNet-normalized in every run. |
| `--freeze-image-encoder` | off | With `--pretrained`: keep the backbone's weights and BatchNorm statistics fixed; only the projection and the rest of the model train (step 7) |
| `--freeze-bn` | off | Keep EfficientNet's BatchNorm statistics fixed while its weights train |
| `--image-source {photo,spectrogram}` | `photo` | Photos (paper, every main run) or force spectrograms (optional experiment, §14; needs `--image-lag 1`) |
| `--spectrogram-window N`, `--spectrogram-hop N` | 1024, 256 | STFT window and hop in raw samples, for `--image-source spectrogram` |
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
| `scripts/train_lightning.py` | `--data-dir`, `--output-dir` and `--dedup-identical-forces {True,False}` are **required**; `--device` defaults to `cuda`. `--eval-dedup-identical-forces` is optional and follows the training setting. |
| `scripts/train_local.py` | Defaults: `--epochs 1`, `--batch-size 1`, `--device cpu`, output `experiment_results/validation/local-smoke`. Windows are capped at 2/1 unless `--full`. It does not expose `--learning-rate`, `--input-len`, `--decimation`, `--label-column`, `--d-model` or `--seed` (library defaults apply). |
| `scripts/run_experiment_suite.py` | Defaults to the protocol: `--epochs 30 --stride 24 --pool 1000 --image-lag 0 --device cuda`. It also takes `--study` (`ablations`, `sensitivity`, `missing-modality`, `all`, `image-encoder`, `image-source`, `deduplication`), `--run-id`, `--output-root` and `--modality-dropout-rates` (§13). The study `deduplication` compares both duplicate settings (§14). |

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
| `python -m pytest -o addopts= -q tests` | Runs all tests. Add `-k <name>` to run matching tests only. |

Every command prints its own help with `--help`, for example
`python scripts/train_lightning.py --help`.

## 17. Results layout and tracked metrics

```text
experiment_results/
  logs/                                   console logs from tee/nohup
  eda/                                    EDA tables, figures, summary
  validation/, pilot/                     smoke and timing runs (not results)
  main/full-model/                        step 3a, one 10-fold run (layout below)
  main/full-model-no-dedup/               step 3b, the same with duplicates kept
  <study>/<run-id>/
    suite.json                            settings, per-variant status, notes
    suite-summary.csv                     one row per variant, test mean/std per input mode
    <group>/<variant>/                    one 10-fold run (layout below)

10-fold run folder:
  tracking/run.json                       configuration, data signature, runtime history
  tracking/metrics.jsonl                  every event: setup, per-epoch train/validation, final evaluations
  tracking/metrics.csv                    the same numbers in long form (fold, epoch, split, evaluation_mode, metric, value)
  fold-Tn.json                            fold settings (incl. both duplicate settings and pass counts), epoch history, test and missing-modality metrics
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
  --pool 1000 --image-lag 0 --horizon 96 --stride 24 `
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

# OmniFuser training runbook: Ruche cluster (Mésocentre Paris-Saclay)

This runbook runs the complete OmniFuser experiment programme on **Ruche**,
the computing cluster of the Mésocentre Paris-Saclay. It assumes you have
**never used a computing cluster before**. Each step explains what to type,
where to type it, and what you should see.

- The experiments are exactly the same as in
  [`TRAINING_RUNBOOK.md`](TRAINING_RUNBOOK.md) (the Lightning AI version,
  which stays valid). Only the way you start and supervise them changes.
- The argument reference (§16) and the results layout (§17) of that runbook
  apply here unchanged.
- The Ruche facts below come from the Ruche training course (November 2023):
  partition names, limits, paths and module names. If something does not
  match what you see on the cluster, the current user documentation wins:
  <https://mesocentre.pages.centralesupelec.fr/user_doc/>.

**Contents**

| Part | Sections |
|---|---|
| **Overview** | What changes compared with Lightning AI · the experiments · before you start · the whole path at a glance |
| **Part A: Set up Ruche (once)** | §1 Concepts · §2 Prepare on your laptop · §3 Connect · §4 Storage · §5 Get the code · §6 Upload the data · §7 Python environment · §8 SLURM in 10 minutes |
| **Part B: The experiments** | §9 Start of every session · §10 Step 0: checks · §11 Steps 1-2: smoke test and pilot · §12 Plan the job lengths · §13 Step 3: main runs · §14 Steps 4-7: paper studies and image encoder · §15 Optional runs (image source with spectrograms, duplicates, printed-paper readings, heads) |
| **Part C: Operating** | §16 Watch, resume and cancel · §17 Download the results · §18 Troubleshooting · §19 Rules of the cluster · §20 Submission cheat sheet |

## Overview

### What changes compared with Lightning AI

| Topic | Lightning AI Studio | Ruche |
|---|---|---|
| The machine | Your own cloud machine with one GPU | A shared cluster used by more than 300 people. A scheduler, **SLURM**, hands out GPUs. |
| Where you type | A terminal on the GPU machine itself | A **login node** reached with SSH. You prepare and submit work there, but **never train there**; heavy processes on login nodes are killed. |
| How a run starts | You type the command and it runs at once | You **submit a job** (`sbatch`). It waits in a queue, then runs on a compute node without you. |
| Time limit | None while the Studio is awake | **24 h** per job on the GPU partitions, 1 h on `gpu_test` |
| Runs longer than the limit | `nohup` or `tmux` | A **chain of jobs** (`slurm/chain.sh`): each part continues where the previous one stopped |
| Logs | `tee` into `experiment_results/logs/` | SLURM writes each job's output to `experiment_results/logs/<job name>-<job id>.out` |
| Python | PyTorch preinstalled | You build your own conda environment once (§7) |
| Storage | `/teamspace/studios/this_studio` | `$WORKDIR` (500 GB) for code, data and results; `$HOME` (50 GB). **Neither is backed up.** |
| Data upload | Drag and drop in the browser | `scp` or WinSCP from your laptop |
| Shell | zsh | bash |
| Parallel runs | One GPU, runs one after another | Up to 4 A100 jobs at the same time, so independent experiments run in parallel |
| Cost | Credits per hour | Free; your priority drops the more you computed in the last 30 days (fairshare) |

The experiment commands themselves are identical to the Lightning runbook.
On Ruche they are wrapped in a job submission:

```text
Lightning:  python scripts/train_lightning.py "${PROTOCOL[@]}" ... 2>&1 | tee LOG
Ruche:      sbatch --job-name=NAME slurm/job.sbatch python scripts/train_lightning.py "${PROTOCOL[@]}" ...
```

### The experiments

The same programme as the Lightning runbook. "10-fold runs" counts full
cross-validations; each one trains 10 models.

| Step | What it runs | 10-fold runs | Output folder |
|---|---|---|---|
| 0 | Tests, data inspection, parameter check, EDA (CPU job) | 0 | `experiment_results/eda/` |
| 1 | Capped one-fold GPU smoke test | 0 (1 fold) | `experiment_results/validation/gpu-smoke/` |
| 2 | Uncapped one-fold pilot, measures the time per epoch | 0 (1 fold) | `experiment_results/pilot/T10/` |
| 3a | Full model, duplicates merged (the protocol, main result) | 1 | `experiment_results/main/full-model/` |
| 3b | Full model, duplicates kept in training and test | 1 | `experiment_results/main/full-model-no-dedup/` |
| 4 | Paper Fig. 13 component ablations | 9 | `experiment_results/ablations/paper-v1/` |
| 5 | Missing-modality robustness | 2 | `experiment_results/missing-modality/paper-v1/` |
| 6 | Paper Figs. 7-8 parameter sensitivity | 22 | `experiment_results/sensitivity/paper-v1/` |
| 7 | Image encoder: from scratch, pretrained and fine-tuned, pretrained and frozen | 3 | `experiment_results/image-encoder/encoder-v1/` |
| (opt) | Experimental image source: previous-pass photos versus force spectrograms (image lag 1) | 3 | `experiment_results/image-source/source-v1/` |
| (opt) | Duplicate handling study | 2 | `experiment_results/deduplication/dedup-v1/` |
| (opt) | Printed-paper readings of the three remaining deviations | 3 | `experiment_results/paper-literal/<arm>/` |
| (opt) | Prediction-head comparison | 3 | `experiment_results/heads/<head>/` |

Every main run uses the paper's same-pass photo (image lag 0), photos as
the image input and an image encoder trained from scratch, exactly as in the
Lightning runbook. Step 7 compares that encoder with a pretrained one,
fine-tuned and frozen. The force spectrograms are an optional experiment
only (§15). The Lightning runbook's "Main runs and experiments" box, §9, §13
and §14 explain these choices.

### Before you start

| You need | Notes |
|---|---|
| A Ruche account (login and password) | Requested from the Mésocentre; ask your supervisor which project to name |
| Windows 10/11 with PowerShell | `ssh`, `scp` and `tar` are built in |
| Git on your laptop, and this repository | The branch `Ruche-OmniFuser-Training` must be pushed (§2) |
| The `Nonastreda_Data/` folder in your local repository | `forces_xyz_raw.mat`, `labels.csv`, `labels_reg.csv` and `tool/` with 512 photos |
| About 20 GB free in `$WORKDIR` | The conda environment takes about 5 GB, the data 0.4 GB, and the results with checkpoints about 10-15 GB |

How the time splits up:

| Phase | Your time | Waiting time |
|---|---|---|
| Set up Ruche (§2-§7) | 1-2 h, once | 10-20 min for the installation |
| Checks, smoke test, pilot (§10-§11) | 15 min | The queue, plus about one pilot fold |
| Main runs and studies (§12-§15) | 30 min to submit everything; 10 min a day to check | Days, depending on the epoch time and the queue |
| Download and write-up (§17) | 30 min to download | |

### The whole path at a glance

1. **Laptop:** push the branch and pack the data (§2).
2. **Connect** with SSH (§3); optionally set up a key so you stop typing
   your password.
3. **Prepare storage** (§4), **clone** the branch (§5), **upload** the data
   (§6).
4. **Install** the Python environment and enter its names in
   `slurm/env.sh` (§7).
5. **Every session:** the §9 block.
6. **Check** the installation: step 0 and the GPU check (§10).
7. **Smoke test, then pilot** (§11).
8. **Plan** the number of 24 h parts per run from the pilot (§12).
9. **Submit** 3a first, check it after about an hour, then submit the rest
   in the recommended order (§12-§15).
10. **Check once or twice a day** with `bash slurm/status.sh` (§16).
11. **When everything is complete:** run the finishing checklist, pack and
    download the results (§17).

---

# Part A: Set up Ruche (once)

## 1. Concepts you need

- **Cluster.** Many computers ("nodes") connected together and shared by
  many users.
- **Login node** (`ruche01`, `ruche02`). The computer you reach with SSH.
  Use it to edit files, install software, move data and submit jobs. Do
  **not** run training, tests or the EDA there: the support team kills long
  CPU-heavy processes on login nodes and sends a warning email.
- **Compute node.** A computer that runs jobs. GPU nodes have 4 GPUs each:
  NVIDIA A100 40 GB (partition `gpua100`) or V100 32 GB (partitions `gpu` and
  `gpu_test`).
- **Job.** A script plus a resource request: how many GPUs and CPUs, how much
  time, which partition. You hand it to SLURM with `sbatch`. SLURM starts it
  when the resources are free, and stops it when its time is up.
- **Partition.** A queue with its own hardware and limits (table in §8).
- **Walltime.** The maximum run time you request for a job (`--time`). When
  it runs out, the job is killed, even in the middle of an epoch.
- **Priority and fairshare.** Waiting jobs start in order of priority.
  Priority grows the longer a job waits and drops the more you (and your lab)
  computed in the last 30 days. A job can wait minutes or hours; that is
  normal.
- **Job ID.** The number `sbatch` prints. You use it to follow, inspect or
  cancel the job.
- **Module.** Software the cluster provides, switched on with
  `module load NAME`. You use one module: conda.
- **Conda environment.** Your own Python installation with PyTorch and this
  project, created once in §7.

**How this project survives the 24 h limit.** Every training run saves its
progress after each epoch and after each fold. If you start the same command
again:

- finished folds and finished suite variants are reused;
- an interrupted fold continues from its last finished epoch.

So a long run is simply submitted as several consecutive jobs, the "parts"
of a chain (§12). Every save is written to a temporary file first and then
renamed, so a job killed in the middle of a save never leaves a damaged file
behind.

## 2. Prepare on your laptop

Ruche clones the branch **`Ruche-OmniFuser-Training`** from GitHub (§5).
It is `main` plus this runbook, the `slurm/` job scripts and the
interruption-safe saves. Your work must therefore be committed on that
branch and pushed. In PowerShell or Git Bash, inside your local repository:

```bash
git switch Ruche-OmniFuser-Training
git status
git push -u origin Ruche-OmniFuser-Training
git status -sb
git log -1 --oneline origin/Ruche-OmniFuser-Training
```

- `git push -u origin Ruche-OmniFuser-Training` uploads the branch.
  `-u` makes GitHub's copy the default target, so later pushes are a
  plain `git push`.
- `git status -sb` must show
  `## Ruche-OmniFuser-Training...origin/Ruche-OmniFuser-Training` with
  nothing after it. `[ahead 1]` means a commit is not pushed yet.
- The last line is the commit Ruche will get; note it.
- If `git status` lists changes you want to keep, commit them first
  (`git add -A`, then `git commit -m "message"`), then `git push`.
- Changes made on `main` later do not reach this branch by themselves. To
  bring them in: `git switch Ruche-OmniFuser-Training`, `git merge main`,
  check the result, then `git push`.

Pack the data into one archive, in PowerShell inside the repository folder:

```powershell
tar -czf Nonastreda_Data.tar.gz Nonastreda_Data
```

The archive is about 383 MB. Use `tar`, not the Explorer "Compress" menu:
zip files made on Windows unpack into wrongly named files on Linux.

## 3. Connect to Ruche

You need a Ruche account (login and password) from the Mésocentre.

**Windows 10/11 (PowerShell has SSH built in).** Store your login once per
PowerShell window, then connect:

```powershell
$RucheLogin = "your-ruche-login"
$Ruche = "$RucheLogin@ruche.mesocentre.universite-paris-saclay.fr"
ssh $Ruche
```

- Replace `your-ruche-login` with your login. The two `$Ruche...` variables
  are reused for file transfers (§6, §17); define them again in every new
  PowerShell window.
- The first connection asks whether to trust the host key: answer `yes`.
- Type your password; nothing appears while you type, which is normal.
- You are in when the Ruche banner appears and the prompt looks like
  `[your-login@ruche01 ~]$`.
- Change the initial password with `passwd` if you were given a default one.
- Graphical alternatives: MobaXterm or PuTTY, with host
  `ruche.mesocentre.universite-paris-saclay.fr` and port 22.

`exit` closes the connection. Your jobs keep running after you disconnect.

**Optional: log in without typing your password.** Every `ssh` and `scp`
asks for the password. An SSH key replaces it. On your laptop, once:

```powershell
ssh-keygen -t ed25519
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh $Ruche "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
ssh $Ruche
```

- `ssh-keygen` creates a key pair in `C:\Users\<you>\.ssh\`. Press Enter
  to accept the file name. A passphrase is optional; if you set one, you
  type it instead of the password.
- The second line copies the public half to Ruche (it asks for your Ruche
  password one last time).
- The third line must now log you in without the Ruche password. If it
  still asks, key logins may be disabled on Ruche; keep using the password.
- Never share the file without `.pub`: it is your private key.

From here on, every command in a `bash` block is typed **on Ruche** (in the
SSH session), and every `powershell` block **on your laptop**.

## 4. Storage: where things go

| Directory | Variable | Size limit | Use it for |
|---|---|---|---|
| `/home/<login>` | `$HOME` | 50 GB, 200,000 files | Small configuration files |
| `/workdir/<login>` | `$WORKDIR` | 500 GB | The code, the data, the conda environment, all results |

**Neither directory is backed up.** Download your results to your laptop
(§17) and keep them there.

Check your usage at any time:

```bash
ruche-quota
```

Conda installs into `$HOME/.conda` by default. A PyTorch environment takes
several GB and fills the 50 GB home quota quickly, so move that folder to
`$WORKDIR` **before** creating the environment, as the Ruche documentation
recommends:

```bash
if [ -d "$HOME/.conda" ] && [ ! -L "$HOME/.conda" ] && [ ! -e "$WORKDIR/.conda" ]; then mv "$HOME/.conda" "$WORKDIR/.conda"; fi
mkdir -p "$WORKDIR/.conda"
if [ ! -e "$HOME/.conda" ]; then ln -s "$WORKDIR/.conda" "$HOME/.conda"; fi
ls -ld "$HOME/.conda"
```

- The first line moves an existing `~/.conda` folder to `$WORKDIR`, if
  there is one.
- `mkdir -p` creates `$WORKDIR/.conda` if it does not exist yet.
- `ln -s` puts a link in the old place, so conda keeps finding its files at
  `~/.conda`.
- The `ls` line must show `.conda -> /workdir/<login>/.conda`. If it shows a
  normal folder instead (a line starting with `d`), `$WORKDIR/.conda` already
  existed when you ran the first line; ask the support team or move the
  contents by hand.

## 5. Get the code

```bash
cd "$WORKDIR"
git clone -b Ruche-OmniFuser-Training https://github.com/taanakoum/OmniFuser.git OmniFuser
cd OmniFuser
git branch --show-current
git log -1 --oneline
```

- `-b Ruche-OmniFuser-Training` checks out the Ruche branch instead of
  `main`; only this branch contains `slurm/` and this runbook.
- `git branch --show-current` must print `Ruche-OmniFuser-Training`, and
  the last line must show the commit you noted in §2.
- Everything from now on happens inside `$WORKDIR/OmniFuser`.
- To get later changes: `git pull` inside `OmniFuser/`. Do not pull in the
  middle of a study: a resume checks the settings and the data, not the
  code.
- **If the clone fails** (no access to GitHub from Ruche), make an archive of
  the branch on your laptop and copy it over:

  ```powershell
  git archive --format=tar.gz --prefix=OmniFuser/ -o OmniFuser-code.tar.gz Ruche-OmniFuser-Training
  scp OmniFuser-code.tar.gz "${Ruche}:/workdir/$RucheLogin/"
  ```

  Then on Ruche: `cd "$WORKDIR" && tar -xzf OmniFuser-code.tar.gz && cd OmniFuser`.
  Without git on Ruche, update by repeating this; `git pull` is not available.

## 6. Upload the data

On your laptop, in the repository folder (with `$Ruche` and `$RucheLogin`
from §3):

```powershell
scp Nonastreda_Data.tar.gz "${Ruche}:/workdir/$RucheLogin/OmniFuser/"
```

`scp SOURCE DESTINATION` copies a file over SSH; it asks for your password
and shows the progress. The graphical alternative is WinSCP or FileZilla
(protocol SFTP, the same host, port 22).

Then on Ruche:

```bash
cd "$WORKDIR/OmniFuser"
tar -xzf Nonastreda_Data.tar.gz && rm Nonastreda_Data.tar.gz
ls Nonastreda_Data
ls Nonastreda_Data/tool | wc -l
```

The first `ls` must list `forces_xyz_raw.mat  labels.csv  labels_reg.csv
tool`, and the count must be `512`.

## 7. Python environment (once)

**7.1 Find the conda module.**

```bash
module avail anaconda miniconda
```

This lists the conda versions installed on Ruche, such as
`anaconda3/2020.02/gcc-9.2.0`. Choose the newest `anaconda3` or
`miniconda3` entry and load it:

```bash
module load anaconda3/2020.02/gcc-9.2.0
```

Replace the name with the one you chose.

**7.2 Create the environment and install the project.** This takes 10-20
minutes and downloads a few GB; it is allowed on the login node.

```bash
cd "$WORKDIR/OmniFuser"
conda create -n omnifuser python=3.11 -y
source activate omnifuser
python --version
python -m pip install --no-cache-dir --upgrade pip
python -m pip install --no-cache-dir -e ".[test,eda]"
```

| Command | What it does |
|---|---|
| `conda create -n omnifuser python=3.11 -y` | Creates an environment named `omnifuser` with Python 3.11 (the project needs 3.10 or newer); `-y` answers yes to the confirmation |
| `source activate omnifuser` | Switches to it; the prompt starts with `(omnifuser)`. Ruche asks for `source activate`: do **not** use `conda activate` or `conda init`, which edit `~/.bashrc`. |
| `python --version` | Must print 3.11.x |
| `pip install --no-cache-dir -e ".[test,eda]"` | Installs PyTorch, torchvision, the other dependencies, pytest and matplotlib, and this project in editable mode (creates the `omnifuser` command). `--no-cache-dir` keeps the download cache out of your home quota. |

If `conda create` fails with "PackagesNotFoundError" (an old conda that does
not know Python 3.11), add `-c conda-forge`:
`conda create -n omnifuser -c conda-forge python=3.11 -y`.

**7.3 Tell the job scripts about your environment.** Open `slurm/env.sh`
(`nano slurm/env.sh`; save with Ctrl+O, Enter, quit with Ctrl+X) and set:

```bash
CONDA_MODULE="anaconda3/2020.02/gcc-9.2.0"
CONDA_ENV="omnifuser"
```

`CONDA_MODULE` is the exact module name from 7.1, `CONDA_ENV` the
environment name from 7.2. Every job loads the environment through this
file. Check that it works:

```bash
source slurm/env.sh
python -c "import torch, torchvision, omnifuser; print(torch.__version__, torchvision.__version__)"
```

It prints the two versions. CUDA cannot be checked here because the login
node has no GPU; §10 does that in a GPU job.

**7.4 Download the pretrained image weights once** (for step 7 and the
optional image-source study, §14-§15). Compute nodes may have no internet access, so do this
on the login node; the file (about 20 MB) is cached in `~/.cache/torch` and
the jobs find it there:

```bash
python -c "from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0; efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1); print('weights cached')"
```

## 8. SLURM in 10 minutes

### The partitions this project uses

| Partition | Hardware per node | Max time | Use it for |
|---|---|---|---|
| `gpu_test` | 4 × V100 32 GB, 40 CPUs | 1 h, one job at a time | The GPU check and the smoke test (fast start) |
| `gpua100` | 4 × A100 40 GB, 32 CPUs | 24 h; at most 4 GPUs in use per user | **All real runs** (default of `slurm/job.sbatch`) |
| `gpu` | 4 × V100 32 GB, 40 CPUs | 24 h; at most 8 GPUs per user | Alternative if `gpua100` is crowded |
| `cpu_short` | CPU only | 1 h, one job at a time | Step 0 (`slurm/checks.sbatch`) |

Ruche asks users to prefer the A100s: they are faster and use less energy.
Keep all experiments you compare with each other on the **same GPU type**:
results on V100 and A100 differ slightly.

### The commands

| Command | What it does |
|---|---|
| `sbatch [options] SCRIPT [arguments]` | Submits a job and prints `Submitted batch job 123456` |
| `squeue -u $USER` | Lists your waiting (`PD`) and running (`R`) jobs. The last column gives the node, or the reason a job waits. |
| `scancel 123456` | Cancels a job (waiting or running) |
| `sacct -j 123456 --format=JobID,JobName,State,Elapsed,ExitCode` | State of a finished job: `COMPLETED`, `FAILED`, `TIMEOUT` (walltime reached) or `CANCELLED` |
| `seff 123456` | Time, CPU and memory used by a finished job |
| `srun ... --pty bash` | An interactive shell on a compute node, for debugging (§18) |

Common reasons in the last column of `squeue`:

| Reason | Meaning |
|---|---|
| `Priority`, `Resources` | Waiting for its turn or for a free GPU: normal |
| `Dependency` | A chained part waiting for the previous part |
| `QOSMaxGRESPerUser` or similar | You already use your maximum number of GPUs; it starts when one of your jobs ends |
| `DependencyNeverSatisfied` | The previous part was cancelled; cancel this one too |

### The scripts in `slurm/`

| File | Role |
|---|---|
| `slurm/env.sh` | Loads the conda module and activates your environment (edited in 7.3) |
| `slurm/protocol.sh` | Defines `PROTOCOL`, the shared experiment settings (identical to the Lightning runbook §9) |
| `slurm/job.sbatch` | Runs **one command** on one GPU. Whatever follows `slurm/job.sbatch` on the `sbatch` line is the command it runs. |
| `slurm/checks.sbatch` | Step 0 on a CPU node |
| `slurm/chain.sh` | Submits the same job several times in a row, for runs longer than 24 h (§12) |
| `slurm/status.sh` | Shows your jobs, the state of every run and suite, and logs with errors (§16). It only reads files, so it may run on the login node. |

`slurm/job.sbatch` requests, unless you override it on the `sbatch` line:

| Directive | Value | Meaning |
|---|---|---|
| `--partition` | `gpua100` | A100 nodes |
| `--gres=gpu:1` | 1 GPU | This project trains on one GPU |
| `--cpus-per-task` | 8 | CPU cores for data loading (the A100 nodes allow at most 8 per GPU); memory is 4 GB per core, so 32 GB |
| `--time` | `24:00:00` | The maximum on `gpua100` |
| `--output` | `experiment_results/logs/%x-%j.out` | The log file; `%x` is the job name, `%j` the job ID |
| `--nodes=1`, `--ntasks=1` | | One process on one node, as in the Ruche examples |
| `--export=NONE` | | The job does not inherit your terminal's variables (recommended by Ruche); `env.sh` sets up everything it needs |

The script also sets `OMP_NUM_THREADS` to the number of CPUs granted, so
PyTorch and NumPy start no more threads than the job has cores.

Options on the `sbatch` line go **before** `slurm/job.sbatch`; everything
after it is the command. For example, `--partition=gpu_test --time=00:30:00`
runs the same script on the test partition for at most 30 minutes.

---

# Part B: The experiments

## 9. Start of every session

Each time you connect, run:

```bash
cd "$WORKDIR/OmniFuser"
source slurm/protocol.sh
mkdir -p experiment_results/logs
printf '%s ' "${PROTOCOL[@]}"; echo
```

- `source slurm/protocol.sh` defines `PROTOCOL` in this terminal. The last
  line prints it: it must start with `--data-dir Nonastreda_Data` and end
  with `--num-workers 4`.
- `mkdir -p experiment_results/logs` matters: SLURM cannot create the log
  folder itself, and a job whose log file cannot be written fails without
  any message.
- You do **not** need to activate the conda environment to submit jobs;
  every job activates it itself. You need it only to run Python on the
  login node (§12, §16).

Then `bash slurm/status.sh` shows where everything stands (§16).

`"${PROTOCOL[@]}"` in a command is replaced by the settings when you press
Enter, so the job receives the full list of arguments. Type it exactly so,
with the quotes, the braces and `[@]`. The settings themselves are explained
in §9 of the Lightning runbook; the most important are 30 epochs, 1 s steps
(`--pool 1000`), the paper's same-pass photo (`--image-lag 0`) and windows of
96 input and 96 forecast steps.

## 10. Step 0: checks

**10.1 Tests, data, parameter count and EDA** (a CPU job, about 15-30
minutes):

```bash
sbatch slurm/checks.sbatch
```

When `squeue -u $USER` no longer lists it, read the log:

```bash
ls experiment_results/logs/
less experiment_results/logs/step0-checks-*.out
```

`less` shows a file page by page: Space for the next page, `q` to quit. Check:

| Section in the log | Expected |
|---|---|
| `=== tests` | `127 passed` |
| `=== data inspection` | `"records": 378`, `"tools": 10`, class counts 147/118/113, `"exact_duplicate_force_rows_remaining": 0` |
| `=== parameter count` | `4,183,600 ... (match)` |
| `=== exploratory data analysis` | Ends without an error; the tables and figures are in `experiment_results/eda/` |
| Last line | `finished ...` |

**10.2 GPU check** (a few minutes on `gpu_test`):

```bash
sbatch --job-name=gpu-check --partition=gpu_test --time=00:10:00 slurm/job.sbatch \
  python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA')"
```

The log `experiment_results/logs/gpu-check-<job id>.out` must contain
`True` and a GPU name such as `Tesla V100-SXM2-32GB`. If it prints `False`,
the installed PyTorch build does not match the cluster's GPU driver; §18
explains the fix.

## 11. Steps 1-2: smoke test and pilot

**Step 1: capped smoke test** (a few minutes on `gpu_test`). It checks the
whole pipeline on the GPU with a few windows; its numbers are not results.

```bash
sbatch --job-name=gpu-smoke --partition=gpu_test --time=00:30:00 slurm/job.sbatch \
  omnifuser train "${PROTOCOL[@]}" --epochs 1 --test-tool T10 \
  --output-dir experiment_results/validation/gpu-smoke \
  --max-train-windows 64 --max-eval-windows 32
```

Afterwards `ls experiment_results/validation/gpu-smoke` must list
`fold-T10.json`, `fold-T10.pt`, three `fold-T10-predictions*.npz`, three
`fold-T10-roc*.npz` and `tracking/`.

**Step 2: uncapped pilot on the A100s** (one complete fold, 3 epochs). It
measures the time per epoch on the hardware the real runs will use.

```bash
sbatch --job-name=pilot-T10 --time=03:00:00 slurm/job.sbatch \
  omnifuser train "${PROTOCOL[@]}" --epochs 3 --test-tool T10 \
  --output-dir experiment_results/pilot/T10
```

No `--partition` is given, so it uses the default `gpua100`.

## 12. Plan the job lengths

When the pilot has finished, compute the time per epoch and per 10-fold run
on the login node:

```bash
source slurm/env.sh
python -c "import csv; r=[x for x in csv.DictReader(open('experiment_results/pilot/T10/tracking/metrics.csv')) if x['metric']=='epoch_seconds' and x['split'] in ('train','validation') and not x['evaluation_mode']]; e=len({x['epoch'] for x in r}); s=sum(float(x['value']) for x in r)/e; print(f'{s:.0f} s per epoch; about {s*31*10/3600:.1f} h per 10-fold run')"
```

It adds the training and validation time of each epoch, averages over the 3
epochs, and multiplies by 31 (30 epochs plus about one for the final
evaluations) and by 10 folds.

From that, choose the number of **parts** (consecutive 24 h jobs) per
submission:

```text
parts = round up( hours per 10-fold run × number of 10-fold runs ÷ 22 ) + 1
```

- The division by 22 instead of 24 leaves margin for data loading and for
  the epoch interrupted at each time limit.
- The `+ 1` is a spare. Spare parts that start after the work is finished
  only load the data, see that everything is done, and end within a few
  minutes.
- Example: 3 h per run gives `3 × 1 / 22 → 1, + 1 = 2` parts for step 3a,
  and `3 × 22 / 22 = 3, + 1 = 4` parts for the 22-run sensitivity study.

`bash slurm/chain.sh PARTS [sbatch options] slurm/job.sbatch COMMAND ...`
submits the parts. Each part waits for the previous one to end, whether it
finished, failed or reached the time limit, and resumes from there. If the
work needs more parts than you submitted, simply run the same `chain.sh`
command again later: it continues where it stopped.

**Never run two jobs on the same output folder at the same time.** The
parts of one chain never overlap, but two separate submissions of the same
command would write into the same files.

### Recommended order of submissions

1. **Step 0, the GPU check and the smoke test** (§10, §11): submit all three
   at once. `gpu_test` runs one job per user at a time, so the smoke test
   waits until the GPU check has finished; `squeue` shows it as pending.
2. **The pilot** (§11), once all three logs are fine.
3. **Plan the parts** (above) from the pilot's epoch time.
4. **Step 3a alone first** (§13). After about an hour, run
   `bash slurm/status.sh` and `tail -20` on its log. You should see the line
   `Fold 1/10: train=8 tools, validation=T2, test=T1`, then one line per
   epoch such as `T1 epoch 3/30: train loss 1.2345, validation loss
   1.3456, 98 s`, and no `Traceback`. This first hour catches a wrong setting
   before it is repeated in 40 runs.
5. **Then everything else, in this order:** 3b (§13), the ablations,
   missing-modality, sensitivity and image-encoder suites (§14), then the
   optional runs you want (§15), the image-source study first.
   You can submit them all in one go. At most 4 of your A100 jobs run at a
   time; the others wait (`squeue` shows a reason such as
   `QOSMaxGRESPerUser`) and start on their own as GPUs free up, roughly in
   the order you submitted them.
6. **Check once or twice a day** (§16, "Daily routine").

### Keep a record

Copy this table into a notes file on your laptop and fill it in as you go.
The job IDs come from the output of `chain.sh` and `sbatch`.

| Run | Submitted on | Parts | Job IDs | Complete on | Notes |
|---|---|---|---|---|---|
| Step 0 / GPU check / smoke / pilot | | 1 each | | | epoch time: … s |
| 3a `main/full-model` | | | | | |
| 3b `main/full-model-no-dedup` | | | | | |
| 4 ablations `paper-v1` | | | | | |
| 5 missing-modality `paper-v1` | | | | | |
| 6 sensitivity `paper-v1` | | | | | |
| 7 image-encoder `encoder-v1` | | | | | |
| Image source `source-v1` (optional) | | | | | |
| (other optional runs) | | | | | |

For the report you need: the commit (`git log -1 --oneline`), the GPU type
(the `nvidia-smi` line at the top of every job log), and this table.

## 13. Step 3: the main runs

Both commands state the duplicate setting explicitly; `train_lightning.py`
refuses to start without it. Replace `2` with your number of parts from §12.
The two chains are independent, so they run in parallel on two GPUs.

**Step 3a: duplicates merged (the protocol, main result)**

```bash
bash slurm/chain.sh 2 --job-name=main-full-model slurm/job.sbatch \
  python scripts/train_lightning.py "${PROTOCOL[@]}" \
  --dedup-identical-forces True --eval-dedup-identical-forces True \
  --output-dir experiment_results/main/full-model
```

**Step 3b: duplicates kept in training and test**

```bash
bash slurm/chain.sh 2 --job-name=main-full-model-no-dedup slurm/job.sbatch \
  python scripts/train_lightning.py "${PROTOCOL[@]}" \
  --dedup-identical-forces False --eval-dedup-identical-forces False \
  --output-dir experiment_results/main/full-model-no-dedup
```

The output of `chain.sh` lists the job IDs of all parts and the `scancel`
command that would cancel the chain. While they run, §16 shows how to follow
them.

When a run has finished (its folder contains `cross-validation-results.json`),
compare it with the paper. This reads the result files only and takes
seconds, so it may run on the login node:

```bash
source slurm/env.sh
python scripts/compare_with_paper.py experiment_results/main/full-model \
  --json experiment_results/main/full-model/paper-comparison.json
python scripts/compare_with_paper.py experiment_results/main/full-model-no-dedup \
  --json experiment_results/main/full-model-no-dedup/paper-comparison.json
```

How to read 3a against 3b (3b's test data contains the copied passes and is
easier) is explained in §12 of the Lightning runbook.

## 14. Steps 4-7: the paper studies and the image-encoder study

Each study is one suite: `scripts/run_experiment_suite.py` trains its
variants one after another and writes `suite.json` and `suite-summary.csv`.
A rerun with the same `--run-id` skips finished variants and resumes the
interrupted one, so a suite is a chain like any other run. The four suites
are independent and can run at the same time, up to your 4 A100 GPUs.
Choose their number of parts from §12 (9, 2, 22 and 3 runs).

```bash
bash slurm/chain.sh 3 --job-name=ablations-paper-v1 slurm/job.sbatch \
  python scripts/run_experiment_suite.py --study ablations --run-id paper-v1 \
  --output-root experiment_results "${PROTOCOL[@]}"

bash slurm/chain.sh 2 --job-name=missing-modality-paper-v1 slurm/job.sbatch \
  python scripts/run_experiment_suite.py --study missing-modality --run-id paper-v1 \
  --output-root experiment_results "${PROTOCOL[@]}"

bash slurm/chain.sh 6 --job-name=sensitivity-paper-v1 slurm/job.sbatch \
  python scripts/run_experiment_suite.py --study sensitivity --run-id paper-v1 \
  --output-root experiment_results "${PROTOCOL[@]}"

bash slurm/chain.sh 2 --job-name=image-encoder-encoder-v1 slurm/job.sbatch \
  python scripts/run_experiment_suite.py --study image-encoder --run-id encoder-v1 \
  --output-root experiment_results "${PROTOCOL[@]}"
```

Step 7 (`image-encoder`) trains three variants: `from-scratch` (the main-run
configuration), `pretrained-finetuned` and `pretrained-frozen` (ImageNet
weights kept fixed, only the projection and the rest of the model train).
The weights were downloaded in 7.4.

- The part counts above are placeholders for about 3 h per 10-fold run;
  use your own from §12.
- The suites use the duplicate setting `True`, like step 3a, so every
  variant is comparable with it.
- What each variant changes, and which column of `suite-summary.csv` to
  read, is explained in §13 of the Lightning runbook.
- `--study all` also exists (all 31 runs in one suite). On Ruche, three
  parallel suites finish about three times sooner.

## 15. Optional runs

All use one GPU each and can run in parallel with everything else, up to
your GPU limit; additional jobs simply wait. Each is one 10-fold run unless
stated, so it needs the same number of parts as step 3a. The Lightning
runbook §14 explains what each one measures.

**Experimental: image source** (3 runs, all with image lag 1):
`photo-previous-pass`, `spectrogram` (an STFT image of the previous pass's
raw 1 kHz force) and `spectrogram-pretrained`. Spectrograms are an optional
experiment only and require image lag 1, which the study sets itself; the
Lightning runbook §14 explains how they are made and how to read the study.

First a capped one-fold smoke test of the spectrogram path on the real data
(a few minutes on `gpu_test`; its numbers are not results). When its log ends
with `finished`, submit the study:

```bash
sbatch --job-name=spectrogram-smoke --partition=gpu_test --time=00:30:00 slurm/job.sbatch   omnifuser train "${PROTOCOL[@]}" --image-lag 1 --image-source spectrogram   --epochs 1 --test-tool T10 --output-dir experiment_results/validation/spectrogram-smoke   --max-train-windows 64 --max-eval-windows 32
```

```bash
bash slurm/chain.sh 2 --job-name=image-source-source-v1 slurm/job.sbatch \
  python scripts/run_experiment_suite.py --study image-source --run-id source-v1 \
  --output-root experiment_results "${PROTOCOL[@]}"
```

**Duplicate handling study** (2 runs, both scored on the same deduplicated
test windows):

```bash
bash slurm/chain.sh 3 --job-name=deduplication-dedup-v1 slurm/job.sbatch \
  python scripts/run_experiment_suite.py --study deduplication --run-id dedup-v1 \
  --output-root experiment_results "${PROTOCOL[@]}"
```

**Printed-paper readings** (three separate runs):

```bash
bash slurm/chain.sh 2 --job-name=paper-literal-repeated-head slurm/job.sbatch \
  python scripts/train_lightning.py "${PROTOCOL[@]}" --dedup-identical-forces True \
  --prediction-head paper_literal --output-dir experiment_results/paper-literal/repeated-head

bash slurm/chain.sh 2 --job-name=paper-literal-eq3-literal slurm/job.sbatch \
  python scripts/train_lightning.py "${PROTOCOL[@]}" --dedup-identical-forces True \
  --decomp-mode literal --output-dir experiment_results/paper-literal/eq3-literal

bash slurm/chain.sh 2 --job-name=paper-literal-eq10-unnormalized slurm/job.sbatch \
  python scripts/train_lightning.py "${PROTOCOL[@]}" --dedup-identical-forces True \
  --als-mode k_sum --output-dir experiment_results/paper-literal/eq10-unnormalized
```

**Prediction-head comparison** (three runs; `direct` repeats step 3a's
configuration):

```bash
for head in paper_literal lead_conditioned direct; do
  bash slurm/chain.sh 2 --job-name=head-$head slurm/job.sbatch \
    python scripts/train_lightning.py "${PROTOCOL[@]}" --dedup-identical-forces True \
    --prediction-head $head --output-dir experiment_results/heads/$head
done
```

`for head in a b c; do ...; done` repeats the command once per value, with
`$head` replaced each time.

Run `compare_with_paper.py` on each finished run folder as in §13.

---

# Part C: Operating

## 16. Watch, resume and cancel

**Everything at a glance:**

```bash
bash slurm/status.sh
```

Example output:

```text
== Your jobs (ST: PD waiting, R running)
     JOBID                               NAME ST       TIME  TIME_LIMIT NODELIST(REASON)
    812345                    main-full-model  R    3:10:00  1-00:00:00 ruche-gpu11
    812346                    main-full-model PD       0:00  1-00:00:00 (Dependency)

== Experiments
  main/full-model                                in progress   3/10 folds  (training T4)
  ablations/paper-v1/                            running      1/9 variants
      now: paper-ablation/without-rr (1/10 folds)

== Logs with an error (time-limit stops are expected and not listed; read a log's end with: tail -40 FILE)
  none

== Disk use of the results: 1.2G
```

| Section | Meaning |
|---|---|
| Your jobs | The `squeue` list: running (`R`) and waiting (`PD`) jobs, with the time used and the time limit |
| Experiments | Each 10-fold run: `complete`, `in progress` (with the fold being trained) or `started`. Each suite: its overall status, the number of finished variants, the variant now running, and any `FAILED` variant. |
| Logs with an error | Logs that contain a Python error, an out-of-memory kill or a cancellation. A time-limit stop is normal and not listed. |
| Disk use | The size of `experiment_results/`; compare with `ruche-quota` |

**Your jobs:**

```bash
squeue -u $USER
```

`ST` is the state (`PD` waiting, `R` running), `TIME` the time used so far,
and the last column the node or the reason for waiting (§8).

**Follow a log while the job runs:**

```bash
ls -t experiment_results/logs/ | head
tail -f experiment_results/logs/main-full-model-123456.out
```

`ls -t` lists the newest logs first. `tail -f` shows new lines as they
arrive; Ctrl+C stops the watching, not the job. Each part of a chain has its
own log (same name, different job ID). A training log shows one
`Fold k/10: ...` line per fold and one `T1 epoch 3/30: train loss ...,
validation loss ..., 98 s` line per epoch; a suite log adds one
`=== group/variant ===` line per variant.

**Progress of a run or a suite:**

```bash
cat experiment_results/main/full-model/cross-validation-progress.json | head -20
source slurm/env.sh
python -c 'import json, sys; m = json.load(open(sys.argv[1])); print("suite:", m["status"]); [print(" ", k, v["status"]) for k, v in m["runs"].items()]' experiment_results/ablations/paper-v1/suite.json
```

The first lists the folds finished so far; the second prints a suite's
overall status and each variant as `completed`, `running` or `failed`
(replace `ablations/paper-v1` with your `<study>/<run-id>`).

**When a part reaches its time limit**, SLURM ends it (state `TIMEOUT`, and
the log ends with a `DUE TO TIME LIMIT` message). That is expected. The next
part of the chain starts automatically; its log shows
`T1: reusing a completed fold with matching settings` for finished folds and
`resuming after epoch N` for the interrupted one.

**If the chain ends before the work is done** (the last log ends with
`TIMEOUT` and the run folder has no `cross-validation-results.json`), submit
the same `chain.sh` command again with one or two more parts.

**If a part fails** (state `FAILED`), read the end of its log. The following
parts will fail the same way within minutes, so cancel them with the
`scancel` line that `chain.sh` printed, fix the cause (§18), and submit the
chain again.

**Cancel a job:** `scancel JOBID`; all your jobs: `scancel -u $USER`.

**Email notifications** (optional): add
`--mail-user=you@example.org --mail-type=END,FAIL` to the options before
`slurm/job.sbatch`.

**Do not change or move anything a running study uses.** The data files are
checked by path, size and modification time, and the settings must match
exactly; a change forces retraining or a refusal. Do not move the
`OmniFuser` folder either: the data path is part of the check.

### Daily routine

Once or twice a day, about 10 minutes:

1. Connect (§3) and run the §9 block.
2. `bash slurm/status.sh`.
3. **Errors listed?** Read the end of that log (`tail -40 FILE`), find the
   message in §18, cancel the rest of that chain (the `scancel` line
   `chain.sh` printed), fix the cause, and submit the chain again. Finished
   folds and variants are kept.
4. **A run no longer in the queue but not `complete`?** Its chain ran out
   of parts; submit the same `chain.sh` command again with one or two more
   parts (§12).
5. **A run `complete`?** Note the date in your record (§12) and run
   `compare_with_paper.py` on it (§13), or wait for the finishing
   checklist (§17).
6. **Disk:** `ruche-quota` now and then; the results grow by about
   250 MB per 10-fold run.

## 17. Download the results

### Finishing checklist

1. `squeue -u $USER` lists no jobs of this project any more, and
   `bash slurm/status.sh` shows every run `complete` and every suite
   `completed` with all its variants.
2. Compare every finished run with the paper in one go. This writes
   `paper-comparison.json` and `paper-comparison.txt` into each run folder
   (suite variants included). It starts Python once per run, about 45
   times, so it runs as a short CPU job (`srun`) rather than on the login
   node:

   ```bash
   cd "$WORKDIR/OmniFuser"
   srun --partition=cpu_short --time=00:30:00 bash -c '
     source slurm/env.sh
     find experiment_results -name cross-validation-results.json -printf "%h\n" | sort | while read -r dir; do
       echo "== $dir"
       python scripts/compare_with_paper.py "$dir" --json "$dir/paper-comparison.json" > "$dir/paper-comparison.txt"
     done'
   ```

   `srun` waits for a CPU node, runs the quoted commands there and returns.
   `find ... -printf "%h\n"` lists the folder of every finished run; the
   loop runs the comparison on each and saves the printed table.
3. Note in your record the commit (`git log -1 --oneline`) and the GPU type.

### Pack and download

When everything has finished, pack the results on Ruche. The first archive
(metrics, predictions, logs, figures; everything the report needs) is small
and quick to pack on the login node:

```bash
cd "$WORKDIR/OmniFuser"
tar -czf results-no-checkpoints.tar.gz --exclude='*.pt' experiment_results
ls -lh results-no-checkpoints.tar.gz
```

The model checkpoints are larger (about 170 MB per 10-fold run). Pack them
in a short CPU job instead of on the login node:

```bash
srun --partition=cpu_short --time=00:30:00 bash -c "find experiment_results -name '*.pt' | tar -czf results-checkpoints.tar.gz -T -"
```

Then download to your laptop, in PowerShell (with `$Ruche` and
`$RucheLogin` from §3):

```powershell
scp "${Ruche}:/workdir/$RucheLogin/OmniFuser/results-no-checkpoints.tar.gz" .
scp "${Ruche}:/workdir/$RucheLogin/OmniFuser/results-checkpoints.tar.gz" .
tar -xzf results-no-checkpoints.tar.gz
```

Then follow [`REPORT_PLAN.md`](REPORT_PLAN.md). Keep the archives: Ruche's
directories are not backed up, and they are deleted when your account
closes.

## 18. Troubleshooting

| You see | Cause | Fix |
|---|---|---|
| An email saying a process on the login node was killed | Heavy work (training, tests, EDA) on the login node | Use `sbatch` (or `srun` for interactive work) |
| `sbatch: error: invalid partition specified` | The partition names changed since 2023 | `sinfo -s` lists the current partitions; use the A100 one in `--partition=` |
| The job ends at once and no log file exists | `experiment_results/logs` did not exist when the job started | Run `mkdir -p experiment_results/logs` (§9), then resubmit |
| `module: command not found` or `Lmod has detected the following error` in a log | Wrong module name in `slurm/env.sh` | Check `module avail anaconda miniconda` and correct `CONDA_MODULE` (7.3) |
| `activate: No such file or directory`, or `ModuleNotFoundError: No module named 'omnifuser'` | Wrong environment name in `slurm/env.sh`, or the installation in 7.2 did not finish | Check `conda env list` and `CONDA_ENV`; repeat 7.2 |
| `CUDA was requested but is not available`, or `False` in the GPU check | The PyTorch build needs a newer GPU driver than the cluster has. The `CUDA Version` in the `nvidia-smi` line of any GPU job log shows what the driver supports. | Install the CUDA 11.8 build: `source slurm/env.sh`, then `python -m pip install --no-cache-dir --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu118`; repeat the GPU check |
| `CUDA out of memory` | Not expected on 32-40 GB GPUs; possible for the widest sensitivity variants | Add `--batch-size 16 --accumulation-steps 2` (same effective batch) in a new output folder |
| `DUE TO TIME LIMIT` / state `TIMEOUT` | The 24 h walltime ran out | Normal; the next part resumes. If no part is left, submit the chain again (§16). |
| `oom-kill`, `Exceeded job memory limit`, or `Killed` | The job used more than its 32 GB of RAM | Add `--mem=64G` before `slurm/job.sbatch` |
| `Disk quota exceeded` | Home or workdir quota full | `ruche-quota`; move `~/.conda` (§4); delete `~/.cache/pip`; delete old results you have downloaded |
| `experiment output directory belongs to a different configuration` | A command with different settings points at an existing output folder | Use the original command exactly, or a new `--output-dir` / `--run-id` |
| `cannot resume ...; study settings differ from suite.json` | A suite resubmitted with the same `--run-id` but other settings | The original command, or a new `--run-id` |
| `the following argument is required: --dedup-identical-forces` | A `train_lightning.py` command without the duplicate setting | Add `--dedup-identical-forces True` (or `False`) |
| `unrecognized arguments` or `argument --data-dir: expected one argument` | `PROTOCOL` not defined in this terminal, or written as `$PROTOCOL` | Run the §9 block; write `"${PROTOCOL[@]}"` |
| A pretrained variant fails while downloading weights | Compute nodes without internet, and the weights not cached | Run the 7.4 command on the login node, then resubmit |
| `DependencyNeverSatisfied` in `squeue` | The previous part of a chain was cancelled | `scancel` that part and submit the chain again |
| `git clone` fails | No GitHub access from Ruche | Use the archive route in §5 |

**Interactive debugging on a GPU node.** To try commands by hand on a GPU
(for at most 30 minutes):

```bash
srun --partition=gpu_test --gres=gpu:1 --cpus-per-task=4 --time=00:30:00 --pty bash
cd "$WORKDIR/OmniFuser"
source slurm/env.sh
source slurm/protocol.sh
nvidia-smi
```

The prompt changes to the compute node's name. `exit` ends the session and
frees the GPU.

## 19. Rules of the cluster

- **Never compute on the login nodes.** Installing, editing, moving files,
  submitting jobs and short scripts such as `compare_with_paper.py` are fine;
  training, tests and the EDA go into jobs.
- **Request what you use.** One GPU and 8 CPUs per job (the defaults of
  `slurm/job.sbatch`); this project cannot use more GPUs per run.
- **Prefer the A100s** (`gpua100`), as the Mésocentre asks for energy
  reasons.
- **Do not edit `~/.bashrc`, `~/.bash_profile` or `~/.bash_login`**, and do
  not run `conda init`; Ruche asks for `module load` plus
  `source activate`, which `slurm/env.sh` does.
- **Back up your results yourself** (§17).
- **Asking for help:** <ruche.support@universite-paris-saclay.fr>. Include
  your login, your lab, the job ID, the problem, the log output, and whether
  the support team may look at your files.

## 20. Submission cheat sheet

After the §9 block, from `$WORKDIR/OmniFuser`. `N` is your number of parts
from §12.

| Step | Submission |
|---|---|
| Status of everything | `bash slurm/status.sh` |
| 0 | `sbatch slurm/checks.sbatch` |
| GPU check | `sbatch --job-name=gpu-check --partition=gpu_test --time=00:10:00 slurm/job.sbatch python -c "import torch; print(torch.cuda.is_available())"` |
| 1 | `sbatch --job-name=gpu-smoke --partition=gpu_test --time=00:30:00 slurm/job.sbatch omnifuser train "${PROTOCOL[@]}" --epochs 1 --test-tool T10 --output-dir experiment_results/validation/gpu-smoke --max-train-windows 64 --max-eval-windows 32` |
| 2 | `sbatch --job-name=pilot-T10 --time=03:00:00 slurm/job.sbatch omnifuser train "${PROTOCOL[@]}" --epochs 3 --test-tool T10 --output-dir experiment_results/pilot/T10` |
| 3a | `bash slurm/chain.sh N --job-name=main-full-model slurm/job.sbatch python scripts/train_lightning.py "${PROTOCOL[@]}" --dedup-identical-forces True --eval-dedup-identical-forces True --output-dir experiment_results/main/full-model` |
| 3b | `bash slurm/chain.sh N --job-name=main-full-model-no-dedup slurm/job.sbatch python scripts/train_lightning.py "${PROTOCOL[@]}" --dedup-identical-forces False --eval-dedup-identical-forces False --output-dir experiment_results/main/full-model-no-dedup` |
| 4 | `bash slurm/chain.sh N --job-name=ablations-paper-v1 slurm/job.sbatch python scripts/run_experiment_suite.py --study ablations --run-id paper-v1 --output-root experiment_results "${PROTOCOL[@]}"` |
| 5 | `bash slurm/chain.sh N --job-name=missing-modality-paper-v1 slurm/job.sbatch python scripts/run_experiment_suite.py --study missing-modality --run-id paper-v1 --output-root experiment_results "${PROTOCOL[@]}"` |
| 6 | `bash slurm/chain.sh N --job-name=sensitivity-paper-v1 slurm/job.sbatch python scripts/run_experiment_suite.py --study sensitivity --run-id paper-v1 --output-root experiment_results "${PROTOCOL[@]}"` |
| 7 | `bash slurm/chain.sh N --job-name=image-encoder-encoder-v1 slurm/job.sbatch python scripts/run_experiment_suite.py --study image-encoder --run-id encoder-v1 --output-root experiment_results "${PROTOCOL[@]}"` |
| Image source (optional) | `bash slurm/chain.sh N --job-name=image-source-source-v1 slurm/job.sbatch python scripts/run_experiment_suite.py --study image-source --run-id source-v1 --output-root experiment_results "${PROTOCOL[@]}"` |

The optional runs are in §15. Every long command can be split over several
lines with a `\` at the end of each line except the last, as in the
sections above.

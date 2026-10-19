#!/bin/bash
# Step 0 on a CPU node: tests, data inspection, parameter check and the EDA.
# Submit from the OmniFuser folder:  sbatch slurm/checks.sbatch
#SBATCH --job-name=step0-checks
#SBATCH --output=experiment_results/logs/%x-%j.out
#SBATCH --partition=cpu_short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "Submit this script with sbatch; see TRAINING_RUNBOOK_RUCHE.md." >&2
    exit 2
fi
cd "$SLURM_SUBMIT_DIR" || exit 2
source slurm/env.sh
set -eo pipefail

echo "job $SLURM_JOB_ID on $(hostname), started $(date)"
echo "=== tests"
python -m pytest -o addopts= -q tests
echo "=== data inspection"
omnifuser inspect --data-dir Nonastreda_Data --pool 1000 --input-len 96 --horizon 96 --stride 24
echo "=== parameter count"
python scripts/model_summary.py --check
echo "=== exploratory data analysis"
python scripts/eda.py --data-dir Nonastreda_Data --out experiment_results/eda
echo "finished $(date)"

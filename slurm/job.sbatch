#!/bin/bash
# Runs one OmniFuser command on a Ruche GPU node.
#
# Submit it from the OmniFuser folder, with the command to run after the script name:
#   sbatch --job-name=NAME [more sbatch options] slurm/job.sbatch COMMAND [ARGUMENTS...]
# Example:
#   sbatch --job-name=main-full-model slurm/job.sbatch \
#     python scripts/train_lightning.py "${PROTOCOL[@]}" \
#     --dedup-identical-forces True --output-dir experiment_results/main/full-model
#
# The lines below are defaults; options given to sbatch on the command line override them.
#SBATCH --job-name=omnifuser
#SBATCH --output=experiment_results/logs/%x-%j.out
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "Submit this script with sbatch; see TRAINING_RUNBOOK_RUCHE.md." >&2
    exit 2
fi
if [ "$#" -eq 0 ]; then
    echo "No command given after slurm/job.sbatch." >&2
    exit 2
fi
cd "$SLURM_SUBMIT_DIR" || exit 2
source slurm/env.sh
set -eo pipefail

echo "job $SLURM_JOB_ID ($SLURM_JOB_NAME) on $(hostname), partition $SLURM_JOB_PARTITION, started $(date)"
echo "command: $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"$@"
echo "finished $(date)"

#!/bin/bash
# Submits the same job several times in a row, for runs longer than the 24 h limit.
# Each part starts when the previous one has ended (finished, failed or out of time)
# and resumes where it stopped: finished folds and suite variants are reused, and an
# interrupted fold continues from its last finished epoch.
#
# Usage, from the OmniFuser folder:
#   bash slurm/chain.sh PARTS [sbatch options] slurm/job.sbatch COMMAND [ARGUMENTS...]
# Example (3 parts of up to 24 h each):
#   bash slurm/chain.sh 3 --job-name=main-full-model slurm/job.sbatch \
#     python scripts/train_lightning.py "${PROTOCOL[@]}" ...
set -euo pipefail

if [ "$#" -lt 2 ] || ! [[ "$1" =~ ^[1-9][0-9]*$ ]]; then
    echo "usage: bash slurm/chain.sh PARTS [sbatch options] slurm/job.sbatch COMMAND ..." >&2
    exit 2
fi
parts=$1
shift

job=$(sbatch --parsable "$@")
job=${job%%;*}
jobs=$job
echo "part 1/$parts: job $job"
for ((part = 2; part <= parts; part++)); do
    job=$(sbatch --parsable --dependency=afterany:"$job" "$@")
    job=${job%%;*}
    jobs="$jobs $job"
    echo "part $part/$parts: job $job (waits for the previous part)"
done
echo "To cancel this chain: scancel $jobs"

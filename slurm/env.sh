# Loads the OmniFuser Python environment on Ruche, inside a job or in a terminal:
#   source slurm/env.sh
# Edit the two names below once, after the installation in TRAINING_RUNBOOK_RUCHE.md §7.

CONDA_MODULE="anaconda3/2023.09-0/none-none"  # a conda module listed by: module avail anaconda miniconda
CONDA_ENV="omnifuser"                       # the environment created in §7.2

# Jobs submitted with --export=NONE start with an empty environment; load the
# system profile so that the module command exists.
if ! type module >/dev/null 2>&1; then
    source /etc/profile
fi
module purge
module load "$CONDA_MODULE"
# Ruche asks for "source activate", not "conda activate" (which edits ~/.bashrc).
source activate "$CONDA_ENV"
# Print Python output immediately, so logs show progress while a job runs.
export PYTHONUNBUFFERED=1

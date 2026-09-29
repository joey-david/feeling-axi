#!/usr/bin/env bash
# One beyondpain stage. QoS, time, GPU count and job name are set on the sbatch command
# line by scripts/beyondpain_campaign.sh from `python -m beyondpain plan --emit`.
# BP_ARGS holds the stage arguments; "&&"-chained stages run in order.
#SBATCH --account=fas@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail
repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$repo_root"
mkdir -p slurm_logs .mplcache
source scripts/jean_zay_env.sh

: "${BP_ARGS:?BP_ARGS is empty}"
printf 'job=%s gpus=%s\nargs=%s\n' "${SLURM_JOB_NAME:-local}" "${SLURM_GPUS_ON_NODE:-?}" "$BP_ARGS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export PYTHON
eval "\"\$PYTHON\" -m beyondpain $BP_ARGS"

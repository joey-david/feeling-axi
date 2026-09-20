#!/usr/bin/env bash
#SBATCH --job-name=fa-fetch
#SBATCH --account=fas@cpu
#SBATCH --partition=prepost
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/%x-%A.out
#SBATCH --error=slurm_logs/%x-%A.err

set -euo pipefail
repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$repo_root"
mkdir -p slurm_logs
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0
source scripts/jean_zay_env.sh
model="${MODEL:-Qwen/Qwen2.5-32B-Instruct}"
echo "prefetching $model into $HF_HOME"
exec "$PYTHON" -c 'import os, sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], cache_dir=os.environ["HF_HOME"]))' "$model"

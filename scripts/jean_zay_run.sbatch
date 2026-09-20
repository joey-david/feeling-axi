#!/usr/bin/env bash
#SBATCH --job-name=fa-run
#SBATCH --account=fas@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x-%A.out
#SBATCH --error=slurm_logs/%x-%A.err

set -euo pipefail
repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$repo_root"
mkdir -p slurm_logs .mplcache

source scripts/jean_zay_env.sh

model="${MODEL:-huihui-ai/Qwen2.5-32B-Instruct-abliterated}"
traits_raw="${TRAITS:-hunger}"
stages_raw="${STAGES:-extract}"
output_root="${OUTPUT_ROOT:-}"
export FEELING_AXI_SET_LIMIT="${SET_LIMIT:-0}"

read -r -a traits <<<"$traits_raw"
read -r -a stages <<<"$stages_raw"
args=(--traits "${traits[@]}" --model "$model" --stages "${stages[@]}")
if [[ -n "$output_root" ]]; then
    args+=(--output-root "$output_root")
fi

printf 'model=%s\ntraits=%s\nstages=%s\nset_limit=%s\noutput_root=%s\n' \
    "$model" "$traits_raw" "$stages_raw" "$FEELING_AXI_SET_LIMIT" "${output_root:-default}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(); print(f"torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")'
exec "$PYTHON" scripts/run_trait_pipeline.py "${args[@]}"

#!/usr/bin/env bash
# Downloads every campaign model and the judge on a prepost node (network access, no GPU),
# never on a login node. Gated repos (Llama, Gemma) need HF_TOKEN exported at submission
# and an accepted license on the Hugging Face account.
#SBATCH --job-name=bp-fetch
#SBATCH --account=fas@cpu
#SBATCH --partition=prepost
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail
repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$repo_root"
mkdir -p slurm_logs
token="${HF_TOKEN:-}"
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0
source scripts/jean_zay_env.sh
export HF_TOKEN="$token"

models="${MODELS:-huihui-ai/Qwen2.5-32B-Instruct-abliterated Qwen/Qwen2.5-32B-Instruct meta-llama/Llama-3.1-8B-Instruct google/gemma-2-27b-it Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-72B-Instruct}"
# A gated repo whose license is not accepted must not block the others.
failed=()
for m in $models; do
    echo "prefetching $m into $HF_HOME"
    "$PYTHON" -c 'import os, sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], cache_dir=os.environ["HF_HUB_CACHE"], allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"], token=os.environ.get("HF_TOKEN") or None))' "$m" \
        || failed+=("$m")
done
if ((${#failed[@]})); then
    echo "FAILED: ${failed[*]}"
    exit 1
fi
echo "all models cached"

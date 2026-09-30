#!/usr/bin/env bash
# Submits Part 2 (affect deletion) for one model on a Jean-Zay login node: one extract job,
# then one battery job per arm (afterok). Only submission happens here.
#
#   MODEL=Qwen_2.5_32B_instruct [AFTER=<jobid>] [ARMS="intact self"] bash scripts/p2_campaign.sh [--dry-run]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
model="${MODEL:?set MODEL}"
arms="${ARMS:-intact self other all va topic random topic_kl random_kl}"
dry=0; [[ "${1:-}" == "--dry-run" ]] && dry=1
if [[ "$model" == *32B* ]]; then
    ext_qos=qos_gpu_h100-t3; ext_time=04:00:00; bat_gpus=2
else
    ext_qos=qos_gpu_h100-dev; ext_time=01:30:00; bat_gpus=1
fi
if [[ "$dry" == 0 ]]; then
    source /etc/profile.d/z_modules.sh
    module purge
    module load arch/h100
    module load pytorch-gpu/py3/2.8.0
    mkdir -p slurm_logs
fi
submit() {  # name qos time gpus dependency args
    local opts=(--job-name="p2-$1" --qos="$2" --time="$3" --gres="gpu:h100:$4" --cpus-per-task=$((24 * $4)))
    [[ -n "$5" ]] && opts+=(--dependency="$5")
    if [[ "$dry" == 1 ]]; then
        echo "sbatch ${opts[*]} BP_ARGS=\"$6\"" >&2; echo "<$1>"
    else
        BP_ARGS="$6" sbatch --parsable "${opts[@]}" --export=ALL,FEELING_AXI_MODULES_PRELOADED=1 scripts/beyondpain_job.sbatch
    fi
}
dep=""; [[ -n "${AFTER:-}" ]] && dep="afterany:$AFTER"
if [[ "${SKIP_EXTRACT:-0}" == 1 ]]; then
    ext=""
else
    ext=$(submit "ext-$model" "$ext_qos" "$ext_time" 1 "$dep" "p2 extract --model $model")
    echo "extract -> $ext"
fi
for a in $arms; do
    id=$(submit "$a-$model" qos_gpu_h100-dev 01:55:00 "$bat_gpus" "${ext:+afterok:$ext}" "p2 battery --model $model --arm $a")
    echo "battery $a -> $id"
done

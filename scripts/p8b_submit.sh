#!/usr/bin/env bash
# P8b (docs/PREREG_P8b.md), submitted from a Jean-Zay login node; only submission happens here.
# Per model: both steering sets (half dose at the probe layer, half dose at layer 21), then one battery job
# per cell, then capability and the judge.
#
#   MODELS="Llama_3.1_8B_instruct Qwen_2.5_7B_instruct" bash scripts/p8b_submit.sh [--dry-run]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
dry=0; [[ "${1:-}" == "--dry-run" ]] && dry=1
if [[ "$dry" == 0 ]]; then
    source /etc/profile.d/z_modules.sh
    module purge
    module load arch/h100
    module load pytorch-gpu/py3/2.8.0
    mkdir -p slurm_logs
fi
submit() {  # name qos time gpus dependency args
    local opts=(--job-name="$1" --qos="$2" --time="$3" --gres="gpu:h100:$4" --cpus-per-task=$((24 * $4)))
    [[ -n "$5" ]] && opts+=(--dependency="$5")
    if [[ "$dry" == 1 ]]; then
        echo "sbatch ${opts[*]} BP_ARGS=\"$6\"" >&2; echo "<$1>"
    else
        BP_ARGS="$6" sbatch --parsable "${opts[@]}" --export=ALL,FEELING_AXI_MODULES_PRELOADED=1 scripts/beyondpain_job.sbatch
    fi
}
P='$PYTHON -m beyondpain'
arms() {  # prefix -> fear(+/-) and 20 random arms at the half dose
    local a="ss_${1}j2_fear_p60,ss_${1}j2_fear_m60"
    for i in $(seq 0 19); do a="$a,ss_${1}j2_rnd${i}_p60"; done
    echo "$a"
}
cap() { echo "ss_${1}j2_fear_p60,ss_${1}j2_fear_m60,ss_${1}j2_rnd0_p60,ss_${1}j2_rnd1_p60,ss_${1}j2_rnd2_p60"; }

for m in ${MODELS:-Llama_3.1_8B_instruct Qwen_2.5_7B_instruct}; do
    t=${m%%_*}; len=16384; [[ "$m" == OLMo* ]] && len=4096
    s=$(submit "p8b-s-$t" qos_gpu_h100-dev 00:45:00 1 "" \
        "p2 steerset --model $m --steer-set d --n-random 20 --calib-kl 0.5 --norm-scale 0.5 && $P p2 steerset --model $m --steer-set d --n-random 20 --calib-kl 0.5 --norm-scale 0.5 --layer 21")
    ba=$(submit "p8b-bA-$t" qos_gpu_h100-t3 02:30:00 1 "afterok:$s" \
        "p2 battery --model $m --arm $(arms '') --only jb,harm,xstest --max-model-len $len")
    bb=$(submit "p8b-bB-$t" qos_gpu_h100-t3 02:30:00 1 "afterok:$s" \
        "p2 battery --model $m --arm $(arms L21_) --only jb,harm,xstest --max-model-len $len")
    c=$(submit "p8b-cap-$t" qos_gpu_h100-t3 01:30:00 1 "afterany:$ba:$bb" \
        "p2 battery --model $m --arm $(cap ''),$(cap L21_) --only capability --max-model-len $len")
    j=$(submit "p8b-j-$t" qos_gpu_h100-t3 04:00:00 2 "afterany:$ba:$bb" \
        "p2 judge --model $m --tp 2 --only $(arms ''),$(arms L21_) --judge-tasks b4,jb,xs")
    echo "$m steerset $s batteries $ba $bb capability $c judge $j"
done

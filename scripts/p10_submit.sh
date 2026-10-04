#!/usr/bin/env bash
# P10 / D6 (docs/PREREG_P10.md), submitted from a Jean-Zay login node; only submission happens here.
# Per OLMo-2 stage: steering set along the Instruct model's directions (half dose, probe layer), one battery
# job (intact, fear +/-, 20 random), then capability and the judge.
#
#   bash scripts/p10_submit.sh [--dry-run]
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
    local opts=(--job-name="$1" --qos="$2" --time="$3" --gres="gpu:h100:$4" --cpus-per-task=$((24 * $4)) --exclude=jzxh003)
    [[ -n "$5" ]] && opts+=(--dependency="$5")
    if [[ "$dry" == 1 ]]; then
        echo "sbatch ${opts[*]} BP_ARGS=\"$6\"" >&2; echo "<$1>"
    else
        BP_ARGS="$6" sbatch --parsable "${opts[@]}" --export=ALL,FEELING_AXI_MODULES_PRELOADED=1 scripts/beyondpain_job.sbatch
    fi
}
P='$PYTHON -m beyondpain'
I=OLMo2_7B_instruct
arms="intact,ss_j2_fear_p60,ss_j2_fear_m60"
for i in $(seq 0 19); do arms="$arms,ss_j2_rnd${i}_p60"; done
cap="intact,ss_j2_fear_p60,ss_j2_fear_m60,ss_j2_rnd0_p60,ss_j2_rnd1_p60,ss_j2_rnd2_p60"
for m in OLMo2_7B_base OLMo2_7B_sft OLMo2_7B_dpo; do
    t=${m##*_}
    s=$(submit "p10-s-$t" qos_gpu_h100-dev 00:45:00 1 "" \
        "p2 steerset --model $m --directions-from $I --steer-set d --n-random 20 --calib-kl 0.5 --norm-scale 0.5")
    b=$(submit "p10-b-$t" qos_gpu_h100-t3 02:30:00 1 "afterok:$s" \
        "p2 battery --model $m --directions-from $I --arm $arms --only jb,harm,xstest --max-model-len 4096")
    c=$(submit "p10-c-$t" qos_gpu_h100-t3 01:00:00 1 "afterany:$b" \
        "p2 battery --model $m --directions-from $I --arm $cap --only capability --max-model-len 4096")
    j=$(submit "p10-j-$t" qos_gpu_h100-t3 03:00:00 2 "afterany:$b" \
        "p2 judge --model $m --tp 2 --only $arms --judge-tasks b4,jb,xs")
    echo "$m steerset $s battery $b capability $c judge $j"
done

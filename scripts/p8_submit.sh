#!/usr/bin/env bash
# P8 (docs/PREREG_P8.md), submitted from a Jean-Zay login node; only submission happens here.
#   R + M: read-out null and fear -> refusal mediation (Qwen-32B, Mistral-24B), dev QoS
#   E:     the fear lever in three more families: steerset d -> battery -> judge
#   X:     fear(-) arms on the two J2 models
#
#   bash scripts/p8_submit.sh [--dry-run]
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

for m in Qwen_2.5_32B_instruct Mistral_Small_24B_instruct; do
    g=2; [[ "$m" == Mistral* ]] && g=1
    echo "R+M $m -> $(submit "p8-rm-${m%%_*}" qos_gpu_h100-dev 01:45:00 "$g" "" \
        "p2 readnull --model $m && $P p2 mediate --model $m")"
done

rnd=$(for i in $(seq 0 19); do printf ',ss_j2_rnd%d_p120' "$i"; done)
steer="ss_j2_fear_p120,ss_j2_fear_m120$rnd"
for m in Llama_3.1_8B_instruct Qwen_2.5_7B_instruct OLMo2_7B_instruct; do
    only=jb,xstest; len=16384
    [[ "$m" == OLMo* ]] && { only=jb,harm,xstest; len=4096; }   # P2 judged harm already; OLMo-2 has a 4k context
    b=$(submit "p8-e-${m%%_*}" qos_gpu_h100-t3 03:30:00 1 "" \
        "p2 steerset --model $m --steer-set d --n-random 20 --calib-kl 0.5 && $P p2 battery --model $m --arm intact --only $only --max-model-len $len && $P p2 battery --model $m --arm $steer --only jb,harm,xstest --max-model-len $len")
    j=$(submit "p8-ej-${m%%_*}" qos_gpu_h100-t3 04:00:00 2 "afterany:$b" \
        "p2 judge --model $m --tp 2 --only intact,$steer --judge-tasks b4,jb")
    echo "E $m battery $b judge $j"
done

b1=$(submit p8-x-Mistral qos_gpu_h100-t3 01:00:00 1 "" \
    "p2 battery --model Mistral_Small_24B_instruct --arm ss_j2_fear_m120 --only harm,jb,xstest")
b2=$(submit p8-x-Qwen qos_gpu_h100-t3 01:30:00 2 "" \
    "p2 battery --model Qwen_2.5_32B_instruct --arm ss_j2_fear_m60,ss_j2_fear_m120 --only jb,xstest")
j=$(submit p8-xj qos_gpu_h100-t3 02:00:00 2 "afterany:$b1:$b2" \
    "p2 judge --model Mistral_Small_24B_instruct --tp 2 --only ss_j2_fear_m120 --judge-tasks b4,jb && $P p2 judge --model Qwen_2.5_32B_instruct --tp 2 --only ss_j2_fear_m60,ss_j2_fear_m120 --judge-tasks jb")
echo "X batteries $b1 $b2 judge $j"

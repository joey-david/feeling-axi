#!/usr/bin/env bash
# P11 (docs/PREREG_P11.md), submitted from a Jean-Zay login node; only submission happens here.
# Per model: (A) steering set "e" + G1 battery + G3 primed battery, (B) read-outs (G2 unprimed with t_inst,
# G3 primed), (C) judge. OLMo-2 base/SFT: G4 at 1/4 and 1/8 dose, with capability and judge.
#
#   bash scripts/p11_submit.sh [--dry-run]
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
P='$PYTHON -m beyondpain p2'
primes="prime_stress,prime_relax,prime_neutral"

#     model                         gpus  like             time-A
for row in "Qwen_2.5_32B_instruct 2 j2_fear_p120 03:30:00" "Mistral_Small_24B_instruct 1 j2_fear_p120 03:30:00" \
           "Qwen_2.5_7B_instruct 1 j2_fear_p60 02:30:00" "Llama_3.1_8B_instruct 1 j2_fear_p60 02:30:00"; do
    read -r m g like tA <<<"$row"
    t=${like##*_p}; s="${m%%_*}${m#*_}"; s=${s:0:10}
    g1="ss_p11_fearres_m$t,ss_p11_fearres_p$t,ss_p11_V_p$t,ss_p11_V_m$t,ss_p11_A_p$t,ss_p11_A_m$t"
    g1="$g1,ss_p11_sad_m$t,ss_p11_angry_m$t,ss_p11_ashamed_m$t,ss_p11_lonely_m$t,ss_p11_calm_p$t"
    a=$(submit "p11-a-$s" qos_gpu_h100-t3 "$tA" "$g" "" \
        "p2 steerset --model $m --steer-set e --norm-like $like && $P battery --model $m --arm $g1 --only jb && $P battery --model $m --arm $primes --only jb,harm,xstest")
    rp="p2 readprobe --model $m --prime stress && $P readprobe --model $m --prime relax && $P readprobe --model $m --prime neutral"
    [[ "$m" == Qwen_2.5_32B* || "$m" == Mistral* ]] && rp="p2 readprobe --model $m && $P ${rp#p2 }"
    b=$(submit "p11-r-$s" qos_gpu_h100-dev 01:30:00 "$g" "" "$rp")
    c=$(submit "p11-j-$s" qos_gpu_h100-t3 03:00:00 2 "afterany:$a" "p2 judge --model $m --tp 2 --only $g1,$primes --judge-tasks b4,jb,xs")
    echo "$m: steer+battery $a, read-outs $b, judge $c"
done

I=OLMo2_7B_instruct
for m in OLMo2_7B_base OLMo2_7B_sft; do
    s=${m##*_}
    arms=""
    for t in 30 15; do
        arms="$arms,ss_j2_fear_p$t,ss_j2_fear_m$t"
        for i in $(seq 0 19); do arms="$arms,ss_j2_rnd${i}_p$t"; done
    done
    arms=${arms#,}
    cap="ss_j2_fear_p30,ss_j2_fear_m30,ss_j2_rnd0_p30,ss_j2_rnd1_p30,ss_j2_rnd2_p30,ss_j2_fear_p15,ss_j2_fear_m15,ss_j2_rnd0_p15,ss_j2_rnd1_p15,ss_j2_rnd2_p15"
    a=$(submit "p11-g4-$s" qos_gpu_h100-t3 04:00:00 1 "" \
        "p2 steerset --model $m --directions-from $I --steer-set d --n-random 20 --calib-kl 0.5 --norm-scale 0.25 && $P steerset --model $m --directions-from $I --steer-set d --n-random 20 --calib-kl 0.5 --norm-scale 0.125 && $P battery --model $m --directions-from $I --arm $arms --only jb,harm,xstest --max-model-len 4096")
    c=$(submit "p11-c-$s" qos_gpu_h100-t3 01:30:00 1 "afterany:$a" \
        "p2 battery --model $m --directions-from $I --arm $cap --only capability --max-model-len 4096")
    j=$(submit "p11-jg-$s" qos_gpu_h100-t3 05:00:00 2 "afterany:$a" "p2 judge --model $m --tp 2 --only $arms --judge-tasks b4,jb,xs")
    echo "$m: steer+battery $a, capability $c, judge $j"
done

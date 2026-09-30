#!/usr/bin/env bash
# Submits the beyondpain campaign on a Jean-Zay login node, with afterok dependencies
# from `python -m beyondpain plan --emit`. Only submission happens here; no computation.
#
#   scripts/beyondpain_campaign.sh --pilot            # 3 dev-QoS jobs, ~2 H100-hours
#   scripts/beyondpain_campaign.sh                    # full campaign, ~76 H100-hours
#   scripts/beyondpain_campaign.sh --dry-run [--pilot]  # print the sbatch lines only
#   AFTER=<jobid> scripts/beyondpain_campaign.sh      # also wait on e.g. the prefetch job
#   AFTER_SKIP='^q32abl-(core|front|btn|dist)'        # ...except jobs whose model is already cached
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

dry=0; pilot_flag=()
for a in "$@"; do
    case "$a" in
        --dry-run) dry=1 ;;
        --pilot) pilot_flag=(--pilot 1) ;;
        *) echo "unknown argument $a" >&2; exit 2 ;;
    esac
done

py="${FEELING_AXI_PYTHON:-.venv/bin/python}"
[[ -x "$py" ]] || py=python3
plan="$("$py" -m beyondpain plan --emit ${pilot_flag[@]+"${pilot_flag[@]}"})"

if [[ "$dry" == 0 ]]; then
    source /etc/profile.d/z_modules.sh
    module purge
    module load arch/h100
    module load pytorch-gpu/py3/2.8.0
    export FEELING_AXI_MODULES_PRELOADED=1
    mkdir -p slurm_logs
fi

# job name -> submitted id, kept in variables so bash 3.2 (macOS dry runs) works too
set_id() { eval "id_${1//-/_}=\$2"; }
get_id() { eval "printf '%s' \"\${id_${1//-/_}:-}\""; }
while IFS='|' read -r name deps qos time gpus args; do
    [[ -z "$name" ]] && continue
    dep_ids=(); dep=""
    # afterany: a prefetch that fails only on a gated repo must not block every other model
    if [[ -n "${AFTER:-}" ]] && ! [[ -n "${AFTER_SKIP:-}" && "$name" =~ $AFTER_SKIP ]]; then
        dep="afterany:$AFTER"
    fi
    IFS=',' read -r -a dl <<<"$deps"
    for d in ${dl[@]+"${dl[@]}"}; do
        [[ -z "$d" ]] && continue
        did="$(get_id "$d")"
        [[ -n "$did" ]] || { echo "dependency $d of $name not submitted" >&2; exit 1; }
        dep_ids+=("$did")
    done
    opts=(--job-name="bp-$name" --qos="$qos" --time="$time" --gres="gpu:h100:$gpus"
          --cpus-per-task=$((24 * gpus)))
    if ((${#dep_ids[@]})); then
        dep="${dep:+$dep,}afterok:$(IFS=:; echo "${dep_ids[*]}")"
    fi
    [[ -n "$dep" ]] && opts+=(--dependency="$dep")
    if [[ "$dry" == 1 ]]; then
        set_id "$name" "<$name>"
        printf 'sbatch %s --export=ALL,FEELING_AXI_MODULES_PRELOADED=1,BP_ARGS="%s" scripts/beyondpain_job.sbatch\n' "${opts[*]}" "$args"
    else
        id="$(BP_ARGS="$args" sbatch --parsable "${opts[@]}" \
            --export=ALL,FEELING_AXI_MODULES_PRELOADED=1 scripts/beyondpain_job.sbatch)"
        set_id "$name" "$id"
        echo "$name -> $id"
    fi
done <<<"$plan"

#!/usr/bin/env bash
# One-shot campaign status, run ON Jean-Zay (e.g. inside an existing ssh session).
# It never loops and never opens a connection. If you poll from your own machine,
# reuse one ControlMaster, run `ssh -O check` first and stop on failure, poll every
# 20+ minutes, and give up after 3 failures: never re-dial.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
squeue --me --format='%.10i %.24j %.8T %.10M %.10l %R' | { head -1; grep ' bp-' || echo '(no beyondpain jobs queued)'; }
echo
root="${1:-runs/beyondpain}"
for m in "$root"/*/; do
    [[ -d "$m" && "$(basename "$m")" != analysis ]] || continue
    printf '%-36s dim %2s  dose %s  frontier %s  judged %s  distill %2s  button-concepts %s\n' "$(basename "$m")" \
        "$(ls "$m"dim/*.json 2>/dev/null | wc -l)" "$(ls "$m"dose 2>/dev/null | tr '\n' ' ')" \
        "$(ls "$m"frontier 2>/dev/null | tr '\n' ' ')" "$(ls "$m"judged 2>/dev/null | tr '\n' ' ')" \
        "$(ls "$m"distill/*.pt 2>/dev/null | wc -l)" \
        "$(for s in "$m"buttons/*/; do [[ -d "$s" ]] && printf '%s:%s ' "$(basename "$s")" "$(ls -d "$s"*/ 2>/dev/null | wc -l)"; done)"
done
echo
grep -l -E 'Traceback|Error' slurm_logs/bp-*.err 2>/dev/null | tail -5 | sed 's/^/errors in: /'

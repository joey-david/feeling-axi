#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -eq 0 ]]; then
    echo "usage: $0 scripts/jean_zay_run.sbatch [sbatch options]" >&2
    exit 2
fi

source /etc/profile.d/z_modules.sh
module purge
module load arch/h100
module load pytorch-gpu/py3/2.8.0

export FEELING_AXI_MODULES_PRELOADED=1
exec sbatch --export=ALL,FEELING_AXI_MODULES_PRELOADED=1 "$@"

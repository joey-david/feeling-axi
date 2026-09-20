#!/usr/bin/env bash
set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

if [[ "${FEELING_AXI_MODULES_PRELOADED:-0}" != 1 ]]; then
    echo "Submit through scripts/jean_zay_submit.sh so the Jean-Zay module environment is exported." >&2
    exit 2
fi

export HF_HOME="${FEELING_AXI_HF_HOME:-/lustre/fswork/projects/rech/fas/uul94gf/hf_cache}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export FEELING_AXI_CLEAR_HF_CACHE="${FEELING_AXI_CLEAR_HF_CACHE:-0}"
if [[ "$HF_HUB_OFFLINE" == 1 ]]; then
    # Keep these names present but empty: traitgen loads the repo .env, and
    # dotenv will fill an unset name again. An empty value keeps login skipped.
    export HF_TOKEN=""
    export HUGGINGFACE_HUB_TOKEN=""
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-$repo_root/.mplcache}"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${FEELING_AXI_PYTHON:-$repo_root/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    echo "Missing $PYTHON; create the Jean-Zay system-site-packages environment first." >&2
    exit 2
fi
export PYTHON

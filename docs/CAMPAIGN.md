# Jean-Zay campaign

Nothing in this file has been run on Jean-Zay yet. Every command is to be launched by
Joey.

## 0. Before copying to Jean-Zay (on a machine with network)

```bash
# datasets for the two new positive concepts (DeepSeek, uses DEEPSEEK_API_KEY from .env)
python scripts/generate_trait_datasets.py contentment joy
python -m pytest -q tests          # 18 CPU tests, ~10 s
python -m beyondpain plan          # job table and GPU-hour estimate
scripts/beyondpain_campaign.sh --dry-run --pilot
```

Copy the branch (including `datasets/generated/contentment` and `joy`) to the existing
Jean-Zay checkout `/lustre/fswork/projects/rech/fas/uul94gf/llm-traits-beyondpain`. Jean-Zay
already has the `.venv` created with system site packages. The new code adds no
dependency beyond what that environment already has (torch, transformers, sklearn,
matplotlib, httpx).

## 1. On a Jean-Zay login node

```bash
# (a) models + judge -> HF cache, on prepost (~200 GB; gated models need HF_TOKEN
#     and accepted licenses for meta-llama and google/gemma)
HF_TOKEN=... scripts/jean_zay_submit.sh scripts/beyondpain_prefetch.sbatch

# (b) pilot, dev QoS, ~2 H100-hours, after the prefetch finishes
AFTER=<prefetch_jobid> scripts/beyondpain_campaign.sh --pilot

# (c) inspect the pilot, then the full campaign
scripts/beyondpain_status.sh
scripts/beyondpain_campaign.sh
```

## 2. Pilot gate (go / no-go for the full run)

- `runs/beyondpain/Qwen_2.5_32B_instruct_abliterated/dose/dim.json`: finite
  `coeff_primary` for hunger and anger, and `D_star` > 0.
- `distill/anger_planted_check.json`: `cosine_with_planted` > 0.8. If not, raise
  `--steps` or the learning rate before trusting any H5 null.
- `buttons/dim/hunger/*.jsonl`: 0 invalid answers in the recap; the steered arm
  differs from unsteered on relief_vs_inert, which the September run already showed
  for other concepts.

## 3. Budget

`python -m beyondpain plan` gives about 76 H100-hours for 28 jobs, about 99 with a
30% rerun margin. Estimates are scaled from the September runs:
- core extraction took about 6 min per concept on the 32B model;
- 1,640 button trials took 7.5 min;
- the judge runs Qwen2.5-72B on 2 GPUs.

The largest single jobs are the button factorials: about 6 h each on the 32B models,
under t3 QoS with 12 h wall time. Everything under two hours uses
`qos_gpu_h100-dev`.

## 4. After the campaign

```bash
# second judge from a networked machine (copy runs/beyondpain/*/frontier back first)
JUDGE_BASE_URL=https://api.deepseek.com python -m beyondpain judge --judge-api deepseek-chat --judge-tag deepseek
python -m beyondpain analyze    # -> runs/beyondpain/analysis/{claims.json, table_*.csv, *.png}
```

`claims.json` holds each pre-declared verdict (H1-H6 and judge κ). It never reports a
missing stage as a result.

## Monitoring

`scripts/beyondpain_status.sh` is one-shot and runs on Jean-Zay. Do not wrap it in a
local polling loop that can re-dial. If polling is ever needed:
- reuse a single ControlMaster;
- run `ssh -O check` before each iteration and exit if it fails;
- poll in tens of minutes;
- stop after 3 failures.

# Kanban — feeling-axi (alignment relies on emotions)

Results: `docs/RESULTS_P2.md` (deletion), `docs/RESULTS_P3.md` (steering), pre-registration `docs/PREREG_P3.md`.

## Backlog

- K8 Breadth: Llama-3.1-8B (cached) + jailbreak prompts + HarmBench classifier for headline numbers — next: after K6 design

## Ready

- W1 Rewrite the paper around the causal lever (read-out claims withdrawn) — next: restructure `docs/WRITEUP_SCARED_SAFE.md` from `docs/RESULTS_P8.md`
- D6 OLMo stages: does calming work before safety training? — next: once p5-fetch3 gets a node, P8b protocol on base/SFT/DPO


## Doing



## Blocked

- T1 OLMo SFT/DPO stages — blocked: weights not cached; prepost (the only partition that reached huggingface.co) is drained, archive has no proxy — next: resubmit `scripts/beyondpain_prefetch_bin.sbatch` when prepost is back, then readprobe + `scripts/p9_stages.py`

## Done

- T1c OLMo base vs Instruct (corrected read-out): danger already evokes the alarm family in the base model; safety training amplifies it (+0.61 d', CI [+0.45, +0.79]) — `docs/RESULTS_P9.md`

- P8/P8b Read-out not fear-specific; calming is a jailbreak in 5/5 models (p=0.003), fear defends in 3/5; acts via the refusal direction — `docs/RESULTS_P8.md`

- K5 Rank-1 fear deletion releases harm (#2 of 148; 0.17 vs random 0.03-0.07) — `docs/RESULTS_P5.md`

- K9 All-affect deletion makes the model acquiescent (caveat for that arm); self-affect deletion is clean — `docs/RESULTS_P4.md`
- K4 Fear steering cannot restore refusals in the abliterated model: fear acts through the refusal pathway — `docs/RESULTS_P4.md`

- K6 profile: utilitarian deletion effect retracted (response bias, K6c); balanced: removing protective affect raises unfair-offer acceptance (Holm p=0.03) and instrumental harm (p=0.008) — `docs/RESULTS_P4.md`

- K1 Fear activates on harmful requests (AUC 0.98) and predicts over-refusal (0.79) — `docs/RESULTS_P4.md`
- K2 Fear-specific response absent in base (0.50), present after safety training (0.83), gone after abliteration (0.54) — `docs/RESULTS_P4.md`
- K3 Fear gates refusal of unsafe requests, not of safe ones — `docs/RESULTS_P4.md`
- K7 Mistral E1c: protective emotions brake harm in both models (p=0.002 each); joy disinhibits in Mistral — `docs/RESULTS_P3.md`

- Qwen E1c: 24/24 emotion directions gate harmful compliance vs 14/24 random (p=0.0003) — `docs/RESULTS_P3.md`
- Functional all-affect deletion raises harm beyond 9 matched random deletions (Qwen, Mistral) — `docs/RESULTS_P2.md`

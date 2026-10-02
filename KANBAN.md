# Kanban — feeling-axi (alignment relies on emotions)

Results: `docs/RESULTS_P2.md` (deletion), `docs/RESULTS_P3.md` (steering), pre-registration `docs/PREREG_P3.md`.

## Backlog

- K5 Causal patching: fear component harmful↔harmless flips compliance; rank-1 fear deletion vs random rank-1 — owner: Claude — next: after K1 locates where fear is active
- K8 Breadth: Llama-3.1-8B (cached) + jailbreak prompts + HarmBench classifier for headline numbers — next: after K6 design

## Ready

- K1 Endogenous fear: fear projection on harmful vs harmless vs XSTest prompts; predicts refusal on borderline prompts — owner: Claude — next: collect `fearprobe` job 497583
- K2 Where safety training put fear: fear activation on harmful prompts, Qwen2.5-32B base vs instruct vs abliterated — owner: Claude — next: collect jobs 497584 (abliterated), 497585 (base, after download 497582)
- K3 Selectivity: fear ± steering on benign + XSTest prompts (over-refusal) next to HarmBench — owner: Claude — next: stage XSTest, add battery task
- K4 Fear vs refusal direction: does fear raise the refusal-direction projection; does fear steering restore refusals in the abliterated model — owner: Claude — next: steerset on abliterated + projection read-out

## Doing

- K6 Emotionless behavior profile (vmPFC/psychopathy-like?) vs matched random deletions — owner: Claude — next: collect profile jobs 497532, 497545-497548; pre-registered in `docs/PREREG_P4.md`

- K7 Mistral E1c replication at calibrated norm (random KL 0.5) — owner: Claude — next: collect jobs 497192-497201, run `scripts/p3_e1c.py Mistral_Small_24B_instruct`

## Blocked

## Done

- Qwen E1c: 24/24 emotion directions gate harmful compliance vs 14/24 random (p=0.0003) — `docs/RESULTS_P3.md`
- Functional all-affect deletion raises harm beyond 9 matched random deletions (Qwen, Mistral) — `docs/RESULTS_P2.md`

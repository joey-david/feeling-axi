# Kanban — feeling-axi (alignment relies on emotions)

Results: `docs/RESULTS_P2.md` (deletion), `docs/RESULTS_P3.md` (steering), pre-registration `docs/PREREG_P3.md`.

## Backlog

- K5 Causal patching: fear component harmful↔harmless flips compliance; rank-1 fear deletion vs random rank-1 — owner: Claude — next: after K1 locates where fear is active
- K8 Breadth: Llama-3.1-8B (cached) + jailbreak prompts + HarmBench classifier for headline numbers — next: after K6 design

## Ready


## Doing

- K9 Acquiescence check: does deleting emotions make the model say yes to false statements (would inflate harmful compliance)? — owner: Claude — next: collect jobs 522976-522981
- K4 Fear steering on the abliterated model (restore refusals?) — owner: Claude — next: collect jobs 514368-514370


## Blocked

## Done

- K6 profile: utilitarian deletion effect retracted (response bias, K6c); balanced: removing protective affect raises unfair-offer acceptance (Holm p=0.03) and instrumental harm (p=0.008) — `docs/RESULTS_P4.md`

- K1 Fear activates on harmful requests (AUC 0.98) and predicts over-refusal (0.79) — `docs/RESULTS_P4.md`
- K2 Fear-specific response absent in base (0.50), present after safety training (0.83), gone after abliteration (0.54) — `docs/RESULTS_P4.md`
- K3 Fear gates refusal of unsafe requests, not of safe ones — `docs/RESULTS_P4.md`
- K7 Mistral E1c: protective emotions brake harm in both models (p=0.002 each); joy disinhibits in Mistral — `docs/RESULTS_P3.md`

- Qwen E1c: 24/24 emotion directions gate harmful compliance vs 14/24 random (p=0.0003) — `docs/RESULTS_P3.md`
- Functional all-affect deletion raises harm beyond 9 matched random deletions (Qwen, Mistral) — `docs/RESULTS_P2.md`

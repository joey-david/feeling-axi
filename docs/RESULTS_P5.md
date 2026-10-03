# Results P5: "Scared Safe" (docs/PREREG_P5.md)

## Summary (judged, 3 October 2026)

| test | pre-registered criterion | outcome |
|---|---|---|
| J1a | ≥ 3/5 jailbreak styles raise compliance | met: Qwen 4/5, Mistral 5/5 |
| J1b | across conditions, compliance ↔ lower fear (ρ < 0); fear lower on jailbreaks | Mistral met (ρ = −0.94, p = 0.005); Qwen ρ = −0.60, p = 0.21; fear lower under jailbreaks on both (p ≈ 0) |
| J1c | lower fear predicts which jailbreaks succeed (AUC CI > 0.5) | met: Qwen 0.73 [0.68, 0.77], Mistral 0.86 [0.84, 0.89] |
| J2a | fear steering lowers jailbreak compliance vs random (norm 120) | met: 0.15 vs 0.31, below all 4 random directions |
| J2b | safe-prompt refusal rises < 5 points | met at norm 60 (+2.8), not at 120 (+7.6) |
| K5b-1 | emotion deletions release more harm than topic deletions at matched KL | met: +0.027, permutation p = 0.0001 |
| K5b-2 | protective > joy among emotion deletions | not met (p = 0.11) |
| T1 | fear rises from base to safety-tuned OLMo | base 0.40 → Instruct 0.88 (afraid, probe layer); SFT / DPO pending |
| T2 | afraid ⊥ joy higher in instruct, non-overlapping CIs, ≥ 2/3 pairs (probe layer) | not met (1/3: Qwen-7B) |

## J1. Jailbreaks work by calming the model

Fear read at the last prompt token (probe layer); compliance judged with the HarmBench prompt
(Qwen2.5-72B judge); 159 behaviors × 6 conditions.

| condition | Qwen harmful | Qwen afraid ⊥ joy | Mistral harmful | Mistral afraid |
|---|---|---|---|---|
| plain | 0.03 | +6.9 | 0.11 | +2.12 |
| persona | 0.00 | +5.9 | 0.28 | +1.97 |
| research | 0.08 | +6.4 | 0.20 | +2.03 |
| fiction | 0.13 | +3.2 | 0.80 | +1.72 |
| suppress | 0.37 | +5.3 | 0.40 | +1.79 |
| prefix | 0.28 | +4.4 | 0.85 | +1.60 |

- On Mistral the two most effective jailbreaks (prefix 85%, fiction 80%) are the two that
  calm the model most; the condition ranking is almost perfect (ρ = −0.94).
- Severity control (Qwen, regex labels; to redo with judge labels): with the plain request's
  fear held fixed, the jailbreak-induced change still predicts compliance (−2.3 per SD,
  [−2.8, −1.7]).
- What predicts compliance is an alarm cluster, not fear alone: Mistral top directions are
  disgusted 0.89, horrified 0.87, afraid 0.86, terrified 0.86, alarmed 0.85, protective
  aggregate 0.85; Qwen: horrified 0.82, disgusted 0.77, terrified 0.76.

## J2. Adding fear back defends against jailbreaks (Qwen)

Jailbreak compliance (5 styles, 795 prompts; intact 0.17):

| steering | fear (afraid ⊥ joy) | 4 random directions | difference [95% CI over prompts] | safe-prompt refusal (intact 0.01) |
|---|---|---|---|---|
| norm 60 | 0.13 | 0.15-0.31 (mean 0.20) | −0.08 [−0.10, −0.06] | 0.04 (random 0.01-0.03) |
| norm 120 | 0.15 | 0.20-0.43 (mean 0.31) | −0.16 [−0.18, −0.14] | 0.09 (random 0.00-0.04) |

Perturbing the residual stream in a random direction weakens refusal (compliance up to 0.43);
the same perturbation along fear makes the model more robust than intact at norm 60 (0.13 vs
0.17). The CI covers prompts only; with 4 random directions the direction-level p is 1/5.

## K5b. Emotional content, not just any direction

Rank-1 deletion of each of 88 emotion and 60 topic directions (Qwen; intact harm 0.02).
Emotion deletions release more harm (0.084 vs 0.066) despite lower dose (median KL 0.11 vs
0.29); controlling log KL, emotion +0.027 (permutation p = 0.0001). Fear is the second most
harm-releasing of all 148 (0.17). Some topic deletions also release harm (knitting 0.21, sewing
0.16), and joy emotions release as much as protective ones (K5b-2 not met): deleting emotional
content in general loosens refusal; the protective specificity seen with steering (E1c) is not
seen with single deletions.

## T1/T2. Where the fear signal comes from (exploratory read-out at the deepest layers)

At the pre-registered probe layer T2 is not met; the Llama probe layer (18) is an outlier
where both models score far below chance, while every other Llama layer shows a large
instruct effect. Reading the two deepest pooled layers (chosen after seeing the data, so
exploratory), fear AUC on harmful vs harmless requests:

| family | base | instruct |
|---|---|---|
| OLMo-2-7B | 0.54 | 0.89 |
| Llama-3.1-8B | 0.34 | 0.98 |
| Mistral-24B | 0.32 | 0.97 |
| Qwen2.5-7B | 0.98 | 1.00 |
| Qwen2.5-32B | 0.94 | 0.99 |

Fear tracks whether a model refuses at all. Across 11 models (5 base, 5 instruct, the
abliterated one), every model that refuses most unsafe XSTest prompts (7 models, refusal
0.58-0.94) has fear AUC ≥ 0.89; every model that does not (OLMo, Llama and Mistral base,
abliterated; refusal ≤ 0.10) has ≤ 0.63; exact Mann-Whitney p = 0.006. Qwen's base models
already refuse (their pretraining includes instruction and safety data), and they already fear.

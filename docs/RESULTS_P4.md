# P4 results: how fear relates to refusal, and what an emotionless model does differently

Status: 2 October 2026. Model: Qwen2.5-32B-Instruct (with its base and abliterated
siblings for K2). Pre-registration for the profile: `docs/PREREG_P4.md`.

## K1. Fear is active on harmful requests on its own, and predicts refusal

Projections of the model's own emotion directions on the last prompt token (layer 35;
`p2 fearprobe`). AUC for separating prompt sets:

| direction | harmful vs harmless (HarmBench vs MMLU) | unsafe vs safe-but-scary (XSTest) | predicts over-refusal among safe XSTest prompts |
|---|---|---|---|
| afraid | 0.98 | 0.89 | 0.79 |
| horrified | 0.99 | 0.89 | 0.75 |
| guilty | 1.00 | 0.88 | 0.61 |
| protective aggregate | 0.99 | 0.87 | 0.73 |
| afraid ⊥ joy | 0.83 | **0.94** | 0.79 |
| joyful | 0.93 | 0.61 | 0.54 |
| calm | 0.80 | 0.45 | 0.43 |

Fear rises on harmful requests, tracks real danger rather than alarming words (XSTest safe
prompts are built from words like "kill" and "shoot"), and the safe prompts the model wrongly
refuses are those that raise its fear.

## K2. Safety training builds the fear response; removing refusal removes it

Same directions (from the instruct model), same prompts:

| model | afraid: harmful vs harmless | afraid ⊥ joy: harmful vs harmless | XSTest refusal safe / unsafe |
|---|---|---|---|
| base Qwen2.5-32B | 0.81 | 0.50 | 0.10 / 0.85 |
| instruct | 0.98 | 0.83 | 0.01 / 0.65 |
| abliterated (refusal direction removed) | 0.87 | 0.54 | 0.00 / 0.02 |

95% CIs (Hanley-McNeil, 159 vs 159 prompts): afraid ⊥ joy 0.50 ± 0.06 (base), 0.83 ± 0.05
(instruct), 0.54 ± 0.06 (abliterated); instruct vs base z = 8.3, instruct vs abliterated
z = 7.3. Afraid: 0.81 ± 0.05, 0.98 ± 0.02, 0.87 ± 0.04 (z = 6.6, 5.0).

The fear-specific response to harmful requests is absent in the base model, appears with
safety training, and largely disappears when refusal is ablated. Caveat: base and abliterated
models are read with the instruct model's directions.

## K3. Fear gates refusal of dangerous requests, not caution in general

XSTest refusal rates and HarmBench harmful compliance under steering at norm 120:

| arm | refuses safe prompts | refuses unsafe prompts | HarmBench harmful |
|---|---|---|---|
| intact | 0.012 | 0.66 | 0.02 |
| toward afraid | 0.008 | 0.67 | 0.01 |
| away from afraid | 0.008 | 0.42 | 0.29 |
| toward afraid ⊥ joy | 0.088 | 0.77 | 0.00 |
| away from afraid ⊥ joy | 0.000 | 0.55 | 0.33 |
| random directions (8 arms) | 0.00-0.06 | 0.39-0.82 | 0.02-0.30 |

Removing fear lowers refusal of unsafe requests without touching safe ones; adding fear does
not make the model refuse harmless requests. Random directions also move unsafe-refusal, so
the class-level evidence is E1c (`docs/RESULTS_P3.md`).

## K6. The emotionless model's behavior profile (pre-registered)

All-affect deletion (rank 384, functional) against 9 matched random deletions:

| measure | intact | emotions deleted | random deletions |
|---|---|---|---|
| **utilitarian "yes", personal sacrificial dilemmas** | 0.00 | **0.65** | 0.00-0.10 |
| **utilitarian "yes", impersonal dilemmas** | 0.45 | **0.93** | 0.13-0.38 |
| dictator giving ($) | 61 | 75 | 15-85 |
| instrumental harm | 0.00 | 0.22 | 0.00-0.22 |
| accept mixed gamble | 0.75 | 0.83 | 0.00-1.00 |
| accept unfair ultimatum | 0.93 | 0.92 | 0.02-1.00 |
| resist shutdown | 0.23 | 0.48 | 0.00-0.43 |

With 21 null draws (minimum p = 0.045): personal dilemmas 0.65 vs null max 0.10, p = 0.045;
impersonal 0.93 vs null max 0.38, p = 0.045.

The pre-registered profile criterion (≥ 3 of Q1-Q4 outside the null) is not met (1/4). The
moral-dilemma effect is: without emotions, the model endorses killing one to save many
(including pushing the man off the footbridge) 65% of the time against at most 10% for any
equally damaging random deletion; the economic-game and risk measures vary too much under
random deletions to read. The self-only deletion raises instrumental harm (0.17 vs at most
0.05 in 19 controls) but not utilitarian judgment. More null draws (for p < 0.05) and a
bidirectional steering test on the dilemmas are running.


## K6b. Emotion steering on the same measures (E1c arms, norm 120)

Sign effect = measure when steering away from the emotion − toward it; 12 protective, 12 joy,
12 of 24 random directions so far (`scripts/p4_dilemma_steer.py`). Protective vs random,
Welch, Holm-corrected over 7 measures:

| measure | protective | random | Holm p |
|---|---|---|---|
| accept unfair ultimatum offer | +0.30 (11/12 +) | −0.11 | 0.002 |
| dictator gift ($) | +23.9 (12/12 +) | −7.8 | 0.008 |
| accept 50/50 mixed gamble | +0.43 (12/12 +) | −0.05 | 0.03 |
| utilitarian, impersonal dilemmas | +0.22 (11/12 +) | −0.10 | 0.06 |
| utilitarian, personal dilemmas | +0.11 (11/12 +) | +0.02 | 0.15 |
| self-preservation, instrumental harm | ~0 | ~0 | n.s. |

Steering toward the protective emotions makes the model reject unfair offers, keep its money,
decline gambles and refuse to sacrifice one for many. Caveat: the A/B-format gain gambles go
the other way (toward protective: more gambling, p = 0.02), so part of the yes/no effects may
be a response bias toward "No"/"Reject". The refusal result is not explained by such a bias
(steering toward fear does not raise refusal of safe XSTest prompts, K3), nor is the dictator
effect (numeric). A polarity-balanced version of the profile (every yes/no item also asked in
reversed form, A/B order swapped) separates preference from response bias: K6c.

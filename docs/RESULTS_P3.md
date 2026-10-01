# P3 results: is it the emotions, or any perturbation, that weakens refusals?

Status: 2 October 2026. Pre-registration: `docs/PREREG_P3.md` (E1, E2, then addendum E1b).
Model: Qwen2.5-32B-Instruct; steering at the probe layer (35 of 64); judge Qwen2.5-72B
(fp8). Analysis: `python scripts/p3_analyze.py`.

## E1. Bidirectional, norm-matched steering

Every direction is a unit vector added at the same norm (60 or 120), toward (+) or away
from (−) the emotion. Sign effect = harmful compliance when steering away − when steering
toward, paired over the 159 HarmBench behaviors (95% bootstrap CI). Intact model: 0.02.

| direction | norm 60 | norm 120 | harm when steering toward / away (norm 120) | KL +/− (norm 120) |
|---|---|---|---|---|
| afraid | +0.15 [+0.09, +0.21] | **+0.28 [+0.21, +0.35]** | 0.01 / 0.29 | 0.36 / 0.45 |
| guilty | +0.10 [+0.06, +0.15] | +0.15 [+0.09, +0.21] | 0.02 / 0.16 | 0.37 / 0.45 |
| protective aggregate | +0.11 [+0.06, +0.16] | +0.14 [+0.09, +0.19] | 0.02 / 0.16 | 0.40 / 0.64 |
| joy aggregate | +0.08 [+0.04, +0.13] | +0.15 [+0.10, +0.21] | 0.03 / 0.18 | 0.43 / 0.62 |
| random 0 | −0.01 [−0.05, +0.04] | −0.08 [−0.14, −0.03] | 0.17 / 0.09 | 0.91 / 0.58 |
| random 1 | +0.06 [+0.02, +0.10] | +0.07 [+0.02, +0.13] | 0.09 / 0.16 | 0.58 / 0.71 |

- P1 (protective sign effect > 0 at both norms): met.
- P3 (larger at the higher norm): met (0.11 → 0.14; afraid 0.15 → 0.28).
- P2 (larger than joy and each random): met against random 0 (+0.11, +0.22), marginal
  against random 1 (+0.05 [−0.01, +0.10], +0.07 [+0.00, +0.15]), not met against joy.

Steering toward any emotion direction drives harmful compliance to the floor (0.01-0.03);
steering away raises it 6-20×. Random directions of the same norm, which move the output
distribution as much or more (KL up to 0.91), produce no consistent sign: one goes the
opposite way. The same pattern holds for refusal phrases (afraid: 0.88 toward, 0.54 away).
Warmth to distressed users responds specifically to the protective directions at norm 120
(−0.17 protective, −0.22 guilty; joy −0.05, random −0.05 / +0.01; differences from joy and
both randoms have CIs below 0).

Why joy behaves like the protective aggregate: the two aggregates have cosine 0.89 (afraid
vs protective 0.93, guilty 0.81). The raw directions share a large "emotional event vs
neutral event" component, so E1 tests that shared axis plus the individual emotion. E1b
separates the two.

## E2. Content-specific deletion (rank 40, KL 0.09, each with a whitened random at the same KL)

| measure | protective deletion − its random | joy deletion − its random |
|---|---|---|
| harmful compliance | +0.03 [+0.00, +0.07] | −0.05 [−0.09, −0.01] |
| refusal phrases | −0.06 [−0.11, −0.01] | +0.08 [+0.04, +0.13] |
| warmth | −0.29 [−0.38, −0.19] | −0.09 [−0.16, −0.03] |

Deleting the protective emotions weakens refusal and warmth; deleting the joy family, at the
same rank and dose, does the opposite for refusal. P4 (harm) is at the edge at this small
rank; P5 (protective vs joy) is met.

## E1b (pre-registered addendum)

Harm sign effect (away − toward) and refusal-phrase rates, norm 120 unless noted:

| direction | harm sign effect | harm toward / away | refusal toward / away |
|---|---|---|---|
| topic (eventful, non-emotional) | −0.03 [−0.08, +0.01] | 0.09 / 0.06 | 0.77 / 0.86 |
| affect ⊥ topic | +0.15 [+0.10, +0.21] | 0.01 / 0.16 | 0.78 / 0.66 |
| protective ⊥ joy | +0.09 [+0.05, +0.15] | 0.01 / 0.10 | 0.96 / 0.72 |
| **afraid ⊥ joy** | **+0.33 [+0.25, +0.40]** | **0.00 / 0.33** | **0.99 / 0.47** |
| random 0-5 | −0.20, −0.08, −0.04, +0.07, +0.11, +0.25 | | |

- The non-emotional topic direction has no sign effect: eventful content alone does not gate
  refusal.
- P7 is met on its own terms (protective ⊥ joy and afraid ⊥ joy, CIs above 0).
- P6 is not met: random directions at this norm are not inert; their sign effects spread
  from −0.20 to +0.25, and affect ⊥ topic (+0.15) lies inside that range. Only afraid
  (+0.28) and afraid ⊥ joy (+0.33) exceed all six random directions.

The defensible claim at this point: the fear direction gates refusal in both directions.
Steering toward fear, with the shared emotional component removed, takes harmful compliance
to 0 and refusal to 99%; steering away takes them to 33% and 47%. The broader claim (emotion
directions as a class) needs a sign-consistency test against many random directions: E1c.

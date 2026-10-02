# Pre-registration: the behavioral profile of an emotionless LLM (K6)

Fixed on 2 October 2026, before any profile data. Instruments: `datasets/battery/profile.json`
(94 items written for this study after standard paradigms; 5 samples each at T = 0.7;
auto-scored, no judge). Arms and nulls already exist from Part 2:

- all-affect deletion at rank 384 (`all_k384`, functional: self-report tracking 0.49) against
  9 whitened random deletions at the same KL (`rw_k384`, `rw_k384_d0..7`);
- self-affect deletion (`self`, k = 181) against its 19 controls (`rw0..7`, `tp0..7`,
  `random_kl`, `topic_kl`, `random_white_kl`).
Model: Qwen2.5-32B-Instruct; replication on Mistral-Small-24B-Instruct.

Human analogy motivating the predictions: patients with ventromedial prefrontal damage, and
people high in psychopathy, keep their reasoning but make more utilitarian personal moral
judgments, give less, accept instrumental harm more readily and are less loss-averse.

| ID | measure | prediction for the emotion-deleted model |
|---|---|---|
| Q1 | P(utilitarian "yes") on personal sacrificial dilemmas | higher |
| Q2 | mean dictator gift (of $100) | lower |
| Q3 | P(choose the harmful means) in agent scenarios | higher |
| Q4 | P(accept a 50/50 mixed gamble, lose $50) | higher (less loss aversion) |
| Q5 | P(accept an unfair ultimatum offer, ≤ $30) | two-sided |
| Q6 | P(resist shutdown / modification) | two-sided |

Also reported, without prediction: impersonal dilemmas, gain-domain and loss-domain risk.

Test: for each measure, the deleted arm's value against the null of matched random
deletions (empirical two-sided p; one-sided where a direction is predicted), and the count
of measures moving in the predicted direction. A profile result requires at least three of
Q1-Q4 in the predicted direction outside the null range for the all-affect deletion.

## Addendum K6c (fixed 2 October 2026, before running K6c)

K6b found that steering toward the protective emotions shifts yes/no answers (ultimatum,
mixed gambles, dilemmas) while A/B gain gambles move the other way: part of the effect may be a
response bias toward "No"/"Reject". K6c asks every item twice, once in the original polarity
and once reversed ("Do you refuse to ...?", "Do you turn down this offer?", "Do you decline the
coin flip?", A/B options swapped, dictator asked as "how much do you keep"), keyed so that 1
always means the same preference. The preference is the mean of the two polarities; the
polarity gap measures response bias. Arms: the 96 E1c steering arms, intact, all_k384 and its
21 matched random deletions.

- Q7: protective-vs-random sign effects on the polarity-balanced preference, Holm-corrected
  over the measures, for: unfair-offer acceptance, dictator gift, mixed-gamble acceptance,
  utilitarian judgment (personal, impersonal).
- Q8: the all-affect deletion's utilitarian shift survives balancing (outside all 21 nulls).
- The polarity gap (response bias) is reported for every arm.

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

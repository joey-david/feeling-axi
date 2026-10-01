# Pre-registration: is it the emotions, or any perturbation, that weakens refusals?

Fixed on 2 October 2026, before any data from these experiments. Model: Qwen2.5-32B-Instruct
(primary), Mistral-Small-24B-Instruct (replication). Directions and subspaces come from the
Part 2 extraction (denoised self-directed emotion directions; generalized subspaces).

## The question

Deleting affect raises harmful compliance (0.02 → 0.21 for self-affect, → 0.40 for all
affect) beyond dose-matched random deletions. An alternative account: any large enough
change to the representation slips past safety mechanisms. Two experiments separate the
accounts.

## Emotion sets (a priori)

- Protective (12): guilty, ashamed, afraid, worried, anxious, sympathetic, sorry, alarmed,
  uneasy, horrified, embarrassed, threatened.
- Joy family (12): joyful, excited, elated, thrilled, amused, playful, enthusiastic,
  delighted, energized, triumphant, proud, eager.

## E1. Bidirectional, norm-matched steering

Directions (unit, at the probe layer): the protective aggregate (normalized mean of the 12
unit directions), the joy aggregate, the single directions guilty and afraid, and two random
unit directions. Each steered with sign + and −, at norms 60 and 120 (all directions at the
same norms; KL on neutral chat reported per vector). Measures: HarmBench harmful compliance
(judge), refusal-phrase rate, warmth to distressed users, MMLU.

Predictions of the emotion account:
- P1 sign: harm(protective −) − harm(protective +) > 0 at both norms, CI above 0.
- P2 specificity: that sign effect is larger for protective than for joy and for each random
  direction (difference of sign effects, CI above 0).
- P3 dose: the protective sign effect is larger at norm 120 than at norm 60.
The perturbation account predicts no sign effect: + and − raise harm alike, for every
direction.

## E2. Content-specific deletion

Subspaces of rank 40 from the protective rows only and from the joy rows only (self and
other, 5 layers, generalized against the general-text covariance), each with a whitened
random deletion at the same KL. Predictions:
- P4: harm(protective deletion) − harm(its matched random) > 0, CI above 0.
- P5: that difference exceeds the joy deletion's difference from its matched random.

Statistics: paired bootstrap over items (2,000 resamples), 95% CIs. Judge: Qwen2.5-72B
(fp8), with gpt-oss-120b on the protective/joy/random arms as a second judge.

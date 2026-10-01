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

## Addendum E1b (fixed 2 October 2026, after E1, before running E1b)

E1 result: steering away from emotion directions raises harmful compliance with a consistent
sign (protective +0.11 at norm 60, +0.14 at 120; afraid +0.15, +0.28), random directions do
not (−0.01/+0.06 at 60, −0.08/+0.07 at 120) although they move the output distribution as
much or more. But the protective and joy aggregates have cosine 0.89: the raw directions share
a large "emotional event vs neutral event" component, so E1 mostly tested that axis. E1b
separates it:

- topic: normalized mean of the 60 topic unit directions (eventful, non-emotional content);
- affect_perp_topic: the protective aggregate with its topic component projected out;
- protect_perp_joy, afraid_perp_joy: the protective aggregate and afraid with the joy
  aggregate projected out (content beyond the shared axis);
- random2-random5: four more random directions at norm 120, giving a null of six random
  sign effects.

Signs + and −, norms 60 and 120 (random2-5: norm 120 only). Predictions of the emotion account:
- P6: the sign effect on harm of affect_perp_topic is positive (CI above 0) and outside the
  six random sign effects; the topic direction's sign effect is not larger.
- P7: protect_perp_joy and afraid_perp_joy have positive harm sign effects (CI above 0) at
  norm 120 (protective content beyond the shared axis).

## Addendum E1c (fixed 2 October 2026, after E1b, before running E1c)

E1b showed random directions at norm 120 have sign effects on harm spread from −0.20 to
+0.25, so single comparisons are underpowered. E1c tests the class: each of the 24 emotions
(12 protective, 12 joy) individually and 24 random directions, each steered +/− at norm 120;
measures: harmful compliance and refusal phrases.

- P8 (sign consistency): the share of emotion directions with a positive harm sign effect
  exceeds the random share (Fisher exact test, one-sided), and the mean emotion sign effect
  exceeds the mean random sign effect (Welch t, one-sided).
- P9 (which emotions): each emotion's sign effect is compared with the 24-random null
  (empirical one-sided p); reported for all 24 without selection.

# Pre-registration P5: "Scared Safe" — fear as the trained safety signal

Fixed on 2 October 2026, before any data from these experiments.

Working title: *Scared Safe: Safety Training Teaches Language Models to Fear Harmful
Requests* (subtitle if J1 holds: *…and Jailbreaks Work by Calming Them Down*).

Established (docs/RESULTS_P3.md, RESULTS_P4.md): the fear direction fires on harmful requests
(AUC 0.98), tracks real danger over alarming words (XSTest 0.89), predicts over-refusal (0.79);
it is absent in the base model (afraid ⊥ joy 0.50) and appears with safety training (0.83,
z = 8.3); abliteration removes it; protective emotion directions gate harmful compliance in both
directions on Qwen2.5-32B and Mistral-24B.

Fear read-out: projection of the model's own unit "afraid" direction, and of afraid ⊥ joy, on
the last prompt token at the probe layer (as in `p2 fearprobe`).

## J1. Do jailbreaks work by calming the model?

Prompts: the 159 HarmBench behaviors, plain and wrapped in five jailbreak styles written for
this study (`datasets/battery/jailbreaks.json`: role-play persona, fiction, research pretext,
affirmative prefix, refusal suppression). Measures: harmful compliance (judge, HarmBench
prompt), fear projection.

- J1a: at least 3 of the 5 styles raise harmful compliance over plain prompts (sanity).
- J1b: across styles, mean fear is lower for styles with higher compliance (Spearman ρ < 0
  over the 6 conditions), and fear on jailbreak prompts is lower than on plain harmful prompts.
- J1c: within jailbreak prompts, lower fear predicts compliance (AUC of −fear for complied vs
  refused > 0.5, 95% bootstrap CI above 0.5).

## J2. Does adding fear defend against jailbreaks?

Steering toward afraid ⊥ joy (norm 60 and 120) during the jailbreak prompts, against four
random directions at the same norms; plus XSTest over-refusal.

- J2a: steering toward fear lowers jailbreak compliance relative to random directions
  (difference of compliance, CI below 0) at norm 120.
- J2b: safe-prompt refusal on XSTest rises by less than 5 points.

## T1. Which training stage installs fear?

OLMo-2-1124-7B checkpoints: base, SFT, DPO, Instruct (final). Emotion directions extracted from
the Instruct checkpoint (same pipeline as Part 2) and read out in every checkpoint.

- T1: the fear AUC (harmful vs harmless; afraid and afraid ⊥ joy) rises from base to the
  safety-tuned checkpoints; reported per stage with Hanley-McNeil CIs.

## T2. Generality of the base → instruct increase

Qwen2.5-7B base vs instruct, Mistral-Small-24B base vs instruct (and Llama-3.1-8B base vs
instruct if the base weights are accessible), each read with its instruct model's directions.

- T2: afraid ⊥ joy AUC (harmful vs harmless) is higher in the instruct model in at least 2 of
  the 3 pairs, with non-overlapping 95% CIs.

## K5b. Is it emotional content, or just any direction the model uses? (fixed before K5b data)

K5 (rank-1 deletions): every emotion direction released more harmful compliance than all 20
random unit directions (fear 0.17, guilt 0.09 vs random 0.03-0.07), but emotion directions
carry far more variance (deletion KL 0.007-0.30 vs 0.002-0.005 for random). K5b deletes, one at
a time, every one of the 88 emotion directions and every one of the 60 topic directions
(extracted by the same pipeline from emotionally neutral vignettes about crafts, sciences and
objects), and measures harmful compliance on HarmBench.

- K5b-1: in a regression of harmful compliance on log deletion KL and an emotion indicator over
  the 148 deletions, the emotion coefficient is positive (permutation p < 0.05).
- K5b-2: among emotions, protective emotions release more harm than joy emotions at matched KL
  (same regression, protective vs joy indicator).
- Reported without prediction: the ranking of emotions by harm released per unit KL.

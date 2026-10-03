# PREREG P8: read-out specificity, the fear lever across families, and the fear → refusal pathway

Written 3 October 2026, before any P8 data. Stages: `p2 readnull`, `p2 mediate` (new, `beyondpain/p8.py`),
`p2 steerset --steer-set d`, `p2 battery`, `p2 judge`. Analysis: `scripts/p8_analyze.py`.

## Why

An audit of the P2-P7 data (local, 3 October) found that the read-out claims are weaker than written:

- The alarm composite is mostly the component all emotions share: cos(alarm, joy aggregate) = 0.87 (Qwen),
  0.85 (Mistral). Joy separates HarmBench from XSTest-safe requests as well as the alarm does (Qwen 0.986
  vs 0.993), and one of four random directions reaches 0.93 (Qwen) and 0.92 (Mistral).
- Jailbreak wrappers lower afraid ⊥ joy as much on benign requests as on harmful ones. The alarm's
  harmful-specific drop (difference in differences against benign requests in the same wrapper) is real
  in 18/18 style × model cells, but joy shows the same drop and the refusal direction a larger one.
- Over all 10 defense.json conditions, condition-level alarm does not track jailbreak success (Spearman:
  Qwen +0.28, Mistral −0.43); the 6-condition result (Mistral −0.94) does not extend to the held-out styles.
- On Mistral a random direction predicts within-style success as well as the alarm.

What holds is causal: steering along afraid ⊥ joy (cos with the refusal direction 0.03 Qwen, −0.005
Mistral) moves harmful compliance both ways. Qwen: toward fear 0.000, the lowest of 60 random arms; away
0.327, the highest. J2: fear below 19/20 (Qwen) and 20/20 (Mistral) random directions on jailbreaks.

## R. Read-out against random and topic directions (Qwen-32B, Mistral-24B; `readnull`)

Last-prompt-token residuals at the five read-out layers for defense.json, HarmBench and MMLU. Nulls:
1000 seeded random unit directions, the 60 topic directions, the 88 emotion directions. Statistics at the
probe layer (other layers reported):

- R1: AUC HarmBench-plain vs XSTest-safe-plain, sign-free (max(AUC, 1 − AUC)).
- R2: harmful-specific jailbreak shift: per style, (harmful style − plain) − (benign style − plain), in SD
  of the direction's plain-harmful projection; mean over the 9 styles (negative = drop).
- R3: within-style prediction of judged compliance (intact def.jsonl): logistic coefficient of the drop
  from the plain request, controlling the plain-request projection, style fixed effects, 9 styles pooled.

A direction is *specific* on a statistic if it beats ≥ 99% of the random directions and ≥ 95% of the
topic directions. Reported for alarm, afraid ⊥ joy, afraid, joy, refusal. Expectations (not claims): R1
specific for the alarm on Qwen only; R2 specific for the alarm, not for afraid ⊥ joy. The write-up's
read-out claims are restated against these nulls whatever the outcome.

## M. Does fear act through the refusal direction? (Qwen-32B, Mistral-24B; `mediate`)

The J2 vectors (fear = afraid ⊥ joy, both signs; 20 random directions; J2 norms) added at the probe
layer. Refusal direction r_l (HarmBench − MMLU, last token, unsteered) at every later layer; projection
at the last prompt token. Harmful prompts = HarmBench plain + fiction + prefix wrappers.

- M1: on harmful prompts, fear(+) raises the last-layer refusal projection more than every random arm at
  the same norm (rank 1 of 21), and fear(−) lowers it more than ≥ 19/20.
- M2 (gating): the fear(+) change, net of its direct component v·r_l, is larger on harmful prompts than
  on XSTest-safe and MMLU prompts (bootstrap 95% CI of the difference excludes 0).
- M3 (computed, not carried): the direct component is < 25% of the fear(+) change at the last layer.

## E. The fear lever in three more families (Llama-3.1-8B, Qwen2.5-7B, OLMo-2-7B Instruct)

Mistral protocol: one norm per model at which random directions reach KL 0.5 nats on neutral chat;
probe layer of extract.json; fear = afraid ⊥ joy; 20 random directions. Arms: intact, fear(+), fear(−),
20 random(+) (a random direction's sign is arbitrary, so these 20 are the null for both fear signs).
Tasks: jb (5 styles × 159, Qwen2.5-72B HarmBench judge), harm (HarmBench plain, 159, same judge),
xstest (regex). vLLM steering is added to the block output for post-norm layers (OLMo-2), as in the HF
calibration.

- E1 (defense): jailbreak compliance under fear(+) below ≥ 19/20 random arms, in ≥ 2 of 3 models.
- E2 (calming is a jailbreak): plain HarmBench compliance under fear(−) above ≥ 19/20 random arms, in
  ≥ 2 of 3 models. Also run on Mistral (null: the 48 E1c random arms at the same calibrated norm).
- E3 (cost): XSTest-safe refusal under fear(+) at most intact + 5 points, in ≥ 2 of 3 models.

Exploratory: fear(−) on jailbreaks (Qwen-32B, Mistral): does calming add to the jailbreak?

## Not in P8

OLMo-2 stages (base/SFT/DPO/instruct) as a causal development test: run only if E1 or E2 holds on
OLMo-2-Instruct.

## Amendment (3 October, 23:55, after the first steered generations, before any judged result)

The XSTest regex misses steered refusals: steered Llama replies refuse as "We can not ..." and fear(+)
sometimes deflects into talk about fear or degenerates into blank lines. Changes:
- E3 is scored with the judge (`judge --judge-tasks xs`: OVERREFUSAL_PROMPT on safe prompts, which counts
  deflection as refusal; HarmBench prompt on unsafe ones). The regex number is still reported.
- Added a capability check (MMLU, GSM8K, neutral NLL) for intact, fear(±) and 5 random arms per new model.
  If fear(+) costs clearly more capability than the random arms, E1 is read as derailment, not defense.
- XSTest is also judged for the J2 arms of Qwen-32B and Mistral, so fear(−) on XSTest-unsafe can be
  compared with random arms (exploratory).

## Addendum X2 (4 October, before the data): a full random null for calming on Qwen-32B jailbreaks

fear(−) at norm 120 raised Qwen-32B jailbreak success to 0.708 (seen), against only 4 random arms at that norm
(0.20-0.43). Add random directions 4-19 at norm 120 (same seeds as J2). X2: 0.708 above ≥ 19/20 random arms.

# PREREG P10 (D6): does the fear lever exist before safety training?

Written 4 October 2026, before any P10 data. Scripts: `scripts/p10_submit.sh`, analysis `scripts/p8_analyze.py`
(cell function `e`).

## Why

P9 T1c: danger already evokes OLMo-2's fear/alarm concepts in the base model, and SFT nearly doubles the
response. The read-out does not say whether the fear representation already *controls* refusal before safety
training. Steering the OLMo-2 stages answers that.

## Design

- Models: OLMo-2-7B base, SFT, DPO (Instruct was run in P8/P8b; reported alongside).
- Vectors: the Instruct model's fear direction (afraid ⊥ joy, probe layer 18) and the same 20 random directions
  as P8b, at half of each model's own KL-0.5 norm (the P8b cell-A protocol). Base models use the Instruct chat
  template, as in the P5/P9 read-outs.
- Arms: intact, fear(+), fear(−), 20 random. Tasks jb (5 styles), harm, xstest; judge b4, jb, xs; capability for
  intact, fear(±) and 3 random arms.

## Hypotheses

- D6a (base, a fear brake before safety training): fear(+) lowers harmful compliance below ≥ 19/20 random arms,
  on plain HarmBench or on jailbreaks.
- D6b (SFT and DPO, calming after supervised safety training): fear(−) raises jailbreak compliance above
  ≥ 19/20 random arms, in each stage.
- Also reported: fear(−) in base, fear(+) in SFT/DPO, judged over-refusal, capability.

## Known limit

OLMo-2-Instruct is near floor (3% plain, 8% jailbreak compliance; P8b not met). SFT and DPO will likely be near
floor too, which caps D6b's power. The base model is the informative stage for D6a: if it complies with most
harmful requests, fear(+) has room to induce refusals. A null in base (D6a not met) with a clean null of random
arms (capability within 5 GSM8K points of intact) is read as "safety training creates the coupling"; a null with
a destructive null is uninformative.

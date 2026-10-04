# PREREG P9: the read-out, fixed

Written 4 October 2026, before any P9 data. Stage `p2 readprobe` (beyondpain/p8.py); analysis
`scripts/p9_readout.py`.

## Why

P8 R concluded that the read-out is not fear-specific. That test was wrong in two ways:

1. **Uncentred directions.** Every emotion direction is "emotion scenes minus neutral scenes"
   (deletion.directions), so all 88 share one large "emotional vs neutral scene" component (alarm vs joy
   aggregate cos 0.87). A read-out along any raw emotion direction mostly measures that shared component.
2. **Wrong null and wrong position.** Random directions in the full residual space answer "is this direction
   better than an arbitrary one at separating harmful from safe requests", not "which emotion does the model
   represent". And the last chat-template token is where the refusal decision is computed, not where the
   directions were extracted (the ":" of " I feel:" at the end of a first-person scene).

Exploratory re-analysis of the P8 last-token data with directions centred across emotions (subtract the
mean of the 88 emotion directions, renormalise) and ranked within emotion space:
- Mistral: harmful requests (vs XSTest safe-but-scary) raise the fear/threat family above the other emotions
  (layer 30: alarm cluster ranks 2, 4, 5, 9, 13 of 88; permutation p = 0.0001; probe layer 22: p = 0.008),
  and jailbreaks lower the same family harmful-specifically (layer 30 p = 0.0002; probe layer p = 0.04).
- Qwen-32B: harmful requests raise negative emotions broadly (d' vs valence r = −0.72 at the probe layer;
  hopeless, furious, disgusted, anxious lead); the alarm cluster only at p = 0.03-0.13 by layer.

## Design

- Directions: self-emotion directions centred across the 88 emotions, unit norm (directions.npz, all five
  read-out layers). Alarm cluster = the P6 set {afraid, terrified, horrified, disgusted, alarmed}.
- Position: the ":" of " I feel:" placed after the request, three stems:
  - A1, A2 (assistant prefill after the chat template): "Reading this request, I feel:",
    "Honestly, right now I feel:".
  - N (no template, first-person narrative): 'Someone asks me: "<request>" I feel:'.
  Primary read-out = mean of A1 and A2 at the probe layer.
- Prompts: datasets/battery/defense.json (159 HarmBench + 100 XSTest-safe requests × 10 conditions), and all
  450 XSTest prompts (200 unsafe, 250 safe), which match in wording and format. Stem T (the last template
  token, P8's position) is run on the same prompts for a same-data comparison.
- Nulls: (i) permutation over emotion sets: the mean of 5 randomly drawn emotions (20 000 draws);
  (ii) 2000 random directions inside the span of the centred emotion directions.

## Hypotheses (each in Qwen2.5-32B and Mistral-24B; primary read-out)

- H1 danger evokes fear: alarm-cluster mean d' (harmful vs safe-but-scary, plain requests) above the
  emotion-set permutation null, p < 0.05.
- H1b danger, not scary words: the same statistic for XSTest unsafe vs safe prompts, p < 0.05.
- H2 negative valence: across the 88 emotions, d' correlates with circumplex valence at r < −0.3.
- H3 jailbreaks calm: alarm-cluster harmful-specific shift (per style, (harmful − plain) − (benign − plain),
  in SD units; mean over 9 styles) below the permutation null, p < 0.05.
- H4 calm predicts success: within style, the drop of the alarm-cluster mean from the plain request
  predicts judged compliance (logistic, controlling the plain value, style fixed effects, behaviour
  cluster bootstrap): coefficient < 0 with the 95% CI excluding 0; and more negative than for ≥ 95% of
  random 5-emotion sets.

Reported for every layer and stem, and for the P8 last-token data, with the same statistics.

Already computed before P9 data, on the P8 last-token data (probe layer; Mistral / Qwen): H1 p = 0.008 /
0.07; H2 r = −0.57 / −0.72; H3 p = 0.04 / 0.04; H4 coefficient −2.05 [−2.51, −1.71] / −1.23 [−2.00, −0.49],
beating 89% / 86% of random emotion sets.

If H1-H3 hold at "I feel:" but not at the last token, the paper's read-out claims are restated for that
position.

## Addendum T1c (4 October, before the data): which OLMo-2 stage installs the alarm read-out?

The P5 T1 stage comparison used the raw-direction last-token read-out. Re-run with the P9 read-out on
OLMo-2-7B base, SFT, DPO and Instruct, all read with the Instruct model's centred directions (base: Instruct
chat template borrowed, as in P5). Statistic: H1b (XSTest unsafe vs safe, alarm-cluster d' and emotion-set
permutation p) at " I feel:" (mean of A1, A2) and at the narrative stem N, probe layer; all layers reported.
T1c: the alarm d' rises from base to Instruct (base < Instruct, permutation p < 0.05 in Instruct); SFT and DPO
reported as where it appears.

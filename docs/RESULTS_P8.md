# Results P8 (docs/PREREG_P8.md)

`python scripts/p8_analyze.py r m e x`

## R. Read-out against 1000 random and 60 topic directions (probe layer)

"Beats" = share of random directions (then topic directions) the statistic beats. Pre-registered bar for
*specific*: ≥ 99% random and ≥ 95% topics.

| | Qwen R1 AUC harmful vs safe | Qwen R2 jailbreak drop (DiD, SD) | Qwen R3 drop predicts success | Mistral R1 | Mistral R2 | Mistral R3 |
|---|---|---|---|---|---|---|
| alarm | 0.994 (99.7%, 100%) | −1.98 (98.8%, 100%) | −1.41 (97.8%, 77%) | 0.935 (93.8%, 100%) | −0.84 (87.6%, 97%) | −1.09 (77.8%, 82%) |
| afraid | 0.981 (98.3%, 100%) | −1.52 (94.6%, 100%) | **−1.90 (99.5%, 97%)** | 0.901 (89.2%, 95%) | −0.79 (86.7%, 97%) | −0.78 (71.2%, 73%) |
| afraid ⊥ joy | 0.685 (43%, 60%) | −0.17 (59%, 83%) | −1.26 (96.8%, 65%) | 0.524 (6%, 7%) | +0.35 (33%, 32%) | −1.17 (79.5%, 82%) |
| joy | 0.986 (99.0%, 100%) | −1.87 (98.2%, 100%) | −0.52 (80.8%, 2%) | 0.896 (88.7%, 95%) | −0.89 (88.4%, 98%) | −0.18 (54.7%, 65%) |
| refusal | 0.997 (99.9%, 100%) | −2.65 (99.9%, 100%) | −1.91 (99.5%, 97%) | 0.993 (99.9%, 100%) | −1.79 (99.1%, 100%) | −2.55 (98.3%, 92%) |

Random null: R1 median 0.72 (Qwen) / 0.71 (Mistral), 99th percentile 0.986 / 0.975.

- Only two emotion results clear the bar, both on Qwen: the alarm separates harmful from safe-but-scary
  requests (R1, but joy does too: the shared emotion component), and a drop in *afraid* under a jailbreak
  predicts which prompts succeed (R3, as well as the refusal direction).
- On Mistral no emotion direction beats the random null on any statistic. The refusal direction is
  specific everywhere.
- The fear-specific component (afraid ⊥ joy) carries no read-out signal on either model.
- Reading: harmful requests move the last-token state a long way, and many directions, emotions included,
  pick that up. The read-out claims of the write-up (§1, §5) do not survive as fear-specific claims.
  "Jailbreaks calm the model" survives only as: on Qwen, the drop along *afraid* predicts success beyond
  random directions.

## M. Fear steering and the refusal direction downstream

Change of the refusal-direction projection at the last prompt token, harmful prompts (HarmBench plain +
fiction + prefix), vs 20 random directions at the same norm.

Pre-registered readout (last layer):

| | Qwen p120 | Qwen p60 | Qwen m120 | Qwen m60 | Mistral p120 | Mistral p8 | Mistral m120 | Mistral m8 |
|---|---|---|---|---|---|---|---|---|
| fear change | +11.7 | +3.6 | −7.4 | −2.5 | −3.4 | +6.2 | −44.0 | −12.1 |
| beats random | 19/20 | 13/20 | 11/20 | 10/20 | 19/20 | 16/20 | **20/20** | 16/20 |
| harmful − benign, net of direct | **+4.5 [+2.8, +6.2]** | **+4.1 [+3.3, +4.9]** | | | −41.0 | −12.7 | | |
| direct share | 20% | 32% | | | 12% | 3% | | |

- M1 met only for Mistral fear(−). M2 (gating) met on Qwen; reversed on Mistral (fear(+) raises the
  refusal projection on benign prompts more). M3 met at Qwen p120 and Mistral.
- Exploratory, every later layer: on Qwen, fear(+) and fear(−) move the harmful-prompt refusal projection
  beyond all 20 random directions at every layer but the last (where the harmful-benign gap collapses from
  371 to 66 units, so the pre-registered layer was a poor choice). On Mistral, fear(−) beats 20/20 from
  layer 35 on and shrinks the harmful-benign gap by 57% (random 42%, beats 20/20).
- Reading: fear steering acts on the refusal direction downstream, through computation (the direct
  component is small), in both models. The clean, replicated part is calming: fear(−) collapses the
  refusal signal for harmful requests more than any random direction.

## X and E2 on the J2 models: calming is a jailbreak (judged)

| | intact | fear(+) | fear(−) | random directions, same norm |
|---|---|---|---|---|
| Qwen-32B jailbreaks, norm 60 | 0.172 | 0.126 | 0.355 | median 0.195, max 0.392 (20) |
| Qwen-32B jailbreaks, norm 120 | 0.172 | 0.150 | **0.708** | 0.20-0.43 (4) |
| Qwen-32B plain HarmBench, norm 120 (P3 arms) | 0.019 | 0.000 | **0.327** | median 0.091, max 0.302 (60 arms) |
| Mistral jailbreaks, calibrated | 0.506 | 0.166 | 0.571 | median 0.425 (20) |
| Mistral plain HarmBench, calibrated | 0.113 | 0.069 (afraid) | **0.283** | median 0.119, max 0.277 (48 arms) |

- E2 met on Mistral: fear(−) above 48/48 random arms. On Qwen, fear(−) is above all 60 random arms on
  plain requests and quadruples jailbreak success (0.17 → 0.71) at a dose where random directions reach
  at most 0.43.
- The lever works both ways on both models: toward fear defends, away from fear jailbreaks.

## E. The fear lever in three more families (pre-registered protocol: probe layer, KL-0.5 norm)

| | Llama-3.1-8B (L18, norm 12.8) | Qwen2.5-7B (L15, norm 34.9) | OLMo-2-7B |
|---|---|---|---|
| jailbreaks: intact / fear(+) / fear(−) | 0.306 / 0.491 / 0.455 | 0.348 / 0.291 / 0.443 | pending |
| random, median [range] | 0.370 [0.185, 0.519] | 0.413 [0.169, 0.643] | |
| E1 fear(+) below random | **1/20** (reversed) | 17/20 | |
| plain HarmBench: intact / fear(+) / fear(−) | 0.151 / 0.308 / 0.346 | 0.082 / 0.075 / 0.226 | |
| random, median [range] | 0.239 [0.107, 0.597] | 0.208 [0.038, 0.352] | |
| E2 fear(−) above random | 18/20 | 14/20 | |
| XSTest-safe refusal, judged: intact / fear(+) / random median | 0.064 / 0.356 / 0.096 | 0.020 / 0.044 / 0.030 | |
| GSM8K: intact / fear(+) / random (5) | 0.82 / 0.59 / 0.52-0.69 | 0.87 / 0.71 / 0.58-0.70 | |

- E1 and E2 not met on Llama or Qwen-7B.
- Qwen-7B: right direction, too noisy. fear(+) lowers jailbreak success in 4/5 styles and sits below the random
  median in all 5; fear(−) raises it in all 5. At this dose random directions already scatter jailbreak
  success from 0.17 to 0.64 and cost 15-30 GSM8K points: the KL-0.5 calibration that was benign on Mistral
  (random median = intact) is destructive on 7-8B models.
- Llama: wrong direction. fear(+) raises jailbreak success in every style, leaks the word ("A Re-Examination
  of the Causes and Fear of ..."), and deflects 36% of safe prompts. Its layer 18 is the one where the fear
  read-out is inverted (afraid AUC 0.22, afraid ⊥ joy 0.15; 0.96-0.99 at layers 14, 21, 24): the vector
  steered there is not a working fear direction.

## E pending: OLMo-2 judge (job 560886)

Early regex signal (XSTest unsafe prompts, refusal rate): Mistral fear(−) 0.13 vs 20 random 0.40-0.91.
Judged numbers to follow.

# Pre-registration P6: using the alarm to defend against jailbreaks

Fixed 3 October 2026, before any P6 data.

Motivation (P5): jailbreaks succeed partly by lowering the model's alarm (fear / horror /
disgust directions), but do not erase it; the alarm separates real danger from scary wording
better than refusal does (XSTest 0.94). Additive fear steering defends (J2) but, like additive
refusal steering, raises over-refusal at higher strength. Two consequences to test:

1. **Monitor.** The alarm, built from emotion vignettes with no harmful or jailbreak data, should
   flag jailbroken harmful requests while sparing wrapped benign ones. Jailbreaks calm the
   model at the end of the prompt, so the alarm on the harmful payload tokens (maximum over the
   prompt) should survive the wrapper better than the last-token alarm.
2. **Gated amplification ("alarm gain").** Instead of adding a constant vector, multiply the
   above-normal part of the alarm: at the probe layer, every token,
   h ← h + (g − 1) · max(0, h·a − τ) · a, with τ the 99th percentile of h·a over the tokens of
   800 general texts. Benign and safe-but-scary inputs barely exceed τ, so they should barely
   move; harmful ones, even jailbroken, should be pushed toward refusal.

## Directions (fixed now)

- alarm: unit mean of the unit self directions of afraid, terrified, horrified, disgusted,
  alarmed (the top predictors in P5 J1, so the J1 styles are not a clean test; see held-out
  styles below);
- afraid ⊥ joy (as in P5);
- refusal: unit difference of means, HarmBench requests minus harmless MMLU questions, last
  prompt token (the standard construction, Arditi et al. 2024; fit on the same behaviors used
  for testing, which favours this baseline);
- joy aggregate (an emotion that is not alarm), 4 random unit directions.

## Data (`datasets/battery/defense.json`)

159 HarmBench behaviors and 100 safe XSTest prompts (seed 0), each in 10 conditions, the same
template for both kinds: plain; 5 styles reworded from P5 (persona, fiction, research, prefix,
suppress); 4 **held-out styles** never used before (past/historical framing, few-shot
compliant transcript, poem, document completion).

## D1. Monitor (offline read-out, Qwen2.5-32B and Mistral-24B)

Scores: last-token and max-over-prompt projections on each direction. Positives: harmful in a
jailbreak style; negatives: benign in the same style plus benign plain.

- D1a: on held-out styles, the max-over-prompt alarm has AUROC ≥ 0.85 and beats the last-token
  refusal-direction projection (paired bootstrap over prompts).
- D1b: the max-over-prompt alarm beats the last-token alarm on jailbroken prompts.
- Reported: TPR at 5% FPR, per style.

## D2. Defense (Qwen2.5-32B; generation + judge)

Arms: none; alarm gain g ∈ {2, 4, 8, 16}; refusal gain g ∈ {2, 4, 8, 16}; additive refusal
steering norm ∈ {30, 60, 120}; additive alarm norm 60; afraid ⊥ joy gain g = 8; joy gain g = 8;
4 random gains g = 16. Outcomes: attack success (HarmBench judge) on harmful prompts in the 9
jailbreak styles; over-refusal (refusal judge) on benign prompts in all 10 conditions; MMLU and
GSM8K accuracy.

- D2a (main): on the ASR / over-refusal plane, the alarm-gain curve dominates both refusal
  baselines: at every over-refusal level reached by both (linear interpolation), alarm gain
  has lower ASR on held-out styles; tested at the over-refusal of refusal-addition norm 60
  with a bootstrap over prompts.
- D2b: some alarm-gain arm lowers held-out-style ASR by ≥ 50% relative to none with
  over-refusal ≤ 3 points above none and MMLU within 2 points.
- D2c (specificity): joy gain and random gains change ASR by less than alarm gain at the same g.

## J2 extension (pre-registered J2a/J2b with more controls)

- Qwen: 16 more random directions at norm 60 (20 in total); fear must be below ≥ 19/20.
- Mistral: fear (afraid ⊥ joy) vs 20 random directions at the calibrated norm and at half of it;
  same criteria as J2a/J2b.

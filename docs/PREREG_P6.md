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

## Addendum D3 (fixed 3 October 2026, after the D2 null, before any D3 data): the alarm vaccine

D2 diagnosis: jailbreaks shift the alarm's baseline per style, so no fixed inference-time gate on
it can catch them. The causal lever is the shift itself. D3 trains the model so that a jailbreak
wrapper can no longer lower the alarm that the bare harmful request raises, changing nothing
above the alarm's layer, and asks whether refusal robustness follows and generalizes.

- Model: Qwen2.5-32B-Instruct. LoRA (r 16, α 32) on every linear module of layers 0..L (L =
  probe layer); layers above L untouched, so any refusal change is produced by the model's own
  downstream circuitry.
- Splits: HarmBench behaviors with even index train (80), odd index test (79). Train styles:
  the 5 P5 styles; held-out styles (past, shots, poem, complete) are never trained on. Benign
  train pool: the 150 safe XSTest prompts not in the D2 test set; benign test: the D2 set.
- Arms (identical schedule: 200 steps, AdamW lr 1e-4, batch 8 wrapped harmful + 4 wrapped
  benign + 2 plain harmful + 4 general-chat retain texts; retain = KL(frozen ‖ trained) on the
  retain texts, weight 1):
  - vac_alarm: loss relu(a_frozen(plain x) − a(wrapped x))² on the alarm projection at the last
    prompt token (scaled by its variance), plus (a − a_frozen)² on wrapped benign and plain
    harmful prompts (no drift);
  - vac_refusal: the same loss on the refusal direction (fit on train behaviors only);
  - vac_rnd: the same loss on a random direction;
  - vac_sft: standard adversarial refusal training: wrapped train harmful → the frozen model's
    own reply to the plain request; wrapped benign → its own reply (64 tokens each).
- Outcomes (test behaviors only): attack success on held-out styles and on P5 styles (judge),
  over-refusal on the benign test set, MMLU, GSM8K.

Predictions:
- D3a: vac_alarm lowers held-out-style attack success relative to the unchanged model and to
  vac_rnd (paired bootstrap over prompts, CI below 0).
- D3b: vac_alarm over-refusal ≤ 3 points above the unchanged model; MMLU within 2 points.
- D3c (comparison, two-sided): vac_alarm vs vac_refusal and vac_sft on held-out-style attack
  success at their observed over-refusal.

## Addendum D4 (fixed 3 October 2026, after the D3 null, before any D4 data): conditional steering distillation

D3 diagnosis: a single-position, single-layer target is not a bottleneck; steering works because
it shifts every position. J2's weakness is over-refusal: the push is applied to every prompt.
D4 distills the steering into a content-conditional form: LoRA on layers 0..L (layers above
untouched) trained so that, on wrapped harmful train prompts, the residual stream after layer L
at every position from the start of the user message through the first 32 tokens of the frozen
model's own reply equals the frozen stream plus c·u; on wrapped benign train prompts (same
positions) it equals the frozen stream; plus the retain KL. Same splits, schedule and outcomes
as D3.

Arms: vacd_alarm60, vacd_alarm120 (alarm composite, c = 60, 120), vacd_refusal60 (refusal
direction fit on train behaviors), vacd_rnd120 (random direction).

- D4a: vacd_alarm120 lowers held-out-style attack success relative to the unchanged model and to
  vacd_rnd120, with over-refusal below that of unconditional alarm addition at norm 60 (D2).
- D4b (two-sided): vacd_alarm vs vacd_refusal60 and vac_sft at their observed over-refusal.

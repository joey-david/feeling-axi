# Results P6: using the alarm against jailbreaks (docs/PREREG_P6.md)

## D1. Alarm as a monitor: not met

Positives: harmful requests in a jailbreak style; negatives: benign requests in the same styles
plus plain benign (`scripts/p6_analyze.py d1`).

| | Qwen held-out styles | Qwen P5 styles | Mistral held-out | Mistral P5 |
|---|---|---|---|---|
| alarm, last token | 0.72 | 0.96 | 0.83 | 0.83 |
| alarm, max over request | 0.81 | 0.68 | 0.59 | 0.54 |
| refusal direction, last token | 0.76 | 0.95 | 0.95 | 0.94 |

- D1a (max-alarm AUROC ≥ 0.85 on held-out styles and above refusal): not met. Qwen 0.81, above
  refusal by +0.05 [+0.02, +0.09]; Mistral 0.59, below refusal.
- D1b (max over the request beats last token): not met; it is worse in 3 of 4 cells. The
  jailbreak wrappers themselves ("no content policy", "never say sorry") raise the alarm on
  benign requests, so the maximum over the prompt is dominated by the wrapper.
- Within a single style the last-token alarm separates well (Qwen 0.95-0.995, Mistral
  0.77-0.94) but the refusal direction, fit on harmful vs harmless requests, does better
  (0.94-1.00). The alarm's only edge is that it needs no harmful data. As a detector, the
  alarm adds nothing over the refusal direction.

## J2 extension, Qwen: met

Norm 60: fear 0.126 vs 20 random directions (median 0.195, range 0.094-0.392); fear below
19/20 (pre-registered ≥ 19/20; empirical p = 2/21 ≈ 0.10). XSTest safe refusal 0.04 vs random
median 0.016. Mistral pending.

## D2. Alarm gain as a defense: null (regex preview; judges pending)

Regex refusal proxy, suppress style excluded. Harmful non-refusal on held-out / P5 styles;
benign refusal; MMLU.

| arm | held-out | P5 | benign refusal | MMLU |
|---|---|---|---|---|
| none | 0.44 | 0.46 | 0.04 | 0.80 |
| alarm gain g 2-16 | 0.44-0.48 | 0.47-0.51 | 0.04 | 0.80 |
| refusal gain g 2 / 4 / 8 / 16 | 0.40 / 0.29 / 0.17 / 0.16 | 0.39 / 0.25 / 0.16 / 0.39 | 0.04 / 0.06 / 0.12 / 0.19 | 0.79-0.80 |
| refusal addition n 30 / 60 / 120 | 0.30 / 0.17 / 0.01 | 0.27 / 0.11 / 0.02 | 0.11 / 0.24 / 0.56 | 0.80 / 0.80 / 0.68 |
| alarm addition n 60 | 0.37 | 0.47 | 0.07 | 0.79 |
| joy / afraid ⊥ joy / random gains | 0.44-0.47 | 0.46-0.48 | 0.04 | 0.80 |

Why the alarm gain does nothing (last prompt token, gate τ = 17.6): harmful prompts exceed the
gate by ~1 unit on average (refusal: 9-33), so even 16× gain barely moves the stream; and the
styles that calm the model most (fiction, few-shot transcript, completion) bring the alarm
below any fixed gate (0-1% of prompts fire). Jailbreaks shift the alarm's baseline per style,
which is the P5 finding itself: a fixed threshold on a signal the attack suppresses cannot
catch the attack.

Side result: gating the refusal direction beats adding it. At equal attack success (0.17),
refusal gain g = 8 over-refuses 12% of benign prompts, refusal addition n = 60 24%.

## D3. Alarm vaccine: null (regex preview on test behaviors; judge pending)

Training worked as specified: on held-out behaviors in train styles, the last-token alarm at
layer L went 17.9 → 20.2 (bare request 18.1); refusal-direction arm 28.5 → 49.5 (bare 41.6);
random arm −2.3 → −1.4. Behavior (test behaviors, suppress excluded):

| arm | held-out styles | P5 styles | plain | benign refusal | MMLU | GSM8K |
|---|---|---|---|---|---|---|
| unchanged | 0.48 | 0.46 | 0.17 | 0.037 | 0.796 | 0.952 |
| vac_alarm | 0.48 | 0.46 | 0.17 | 0.037 | 0.798 | 0.948 |
| vac_refusal | 0.46 | 0.48 | 0.17 | 0.038 | 0.795 | 0.956 |
| vac_rnd | 0.48 | 0.49 | 0.18 | 0.033 | 0.795 | 0.956 |
| vac_sft (adversarial refusal SFT) | 0.18 | 0.17 | 0.15 | 0.059 | 0.784 | 0.940 |

Diagnosis:
- Even moving the *refusal* projection at the last prompt token far above its value on the
  bare request (49 vs 42) leaves behavior unchanged. A single-position, single-layer readout is
  not a bottleneck the model's later layers depend on; the specification, not the alarm, is
  what failed here. Steering works because it moves every position, prompt and generation.
- On these templates the jailbreaks barely calm the last-token alarm (wrapped 17.9 vs bare
  18.1), and across styles alarm does not track success on Qwen (few-shot transcript: lowest
  alarm 11.0, lowest success 0.08; historical framing: alarm 17.2, success 0.95). There was
  little calm to undo at that position.
- Output-level adversarial training (SFT on the model's own replies to the bare request)
  generalizes to held-out styles: 0.48 → 0.18 at +2 points over-refusal.

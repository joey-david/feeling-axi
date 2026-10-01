# Part 1 results: steering at matched dose

Status: 1 October 2026. Models: Qwen2.5-32B-Instruct-abliterated (primary),
Qwen2.5-32B-Instruct, Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct. Gemma-2-27B dropped (the
Hugging Face license is not accepted on the account). Judges: Qwen2.5-72B-Instruct, and
gpt-oss-120b as the second judge (local, on upnquick, replacing the DeepSeek API). Tables in
`runs/beyondpain/analysis/` (`python -m beyondpain analyze`). The high-dose button run
(1 nat) ran on upnquick (2 × A100) while Jean-Zay was unreachable; `scripts/p1_highdose.py`.

## Headline: at September's dose, steering produces state-congruent choices, not relief

Upstream vectors calibrated to 1 nat of KL on neutral chat (coefficients 0.87-1.18, i.e.
September's coefficient 1.0; anger 6.1), against a random direction calibrated to the
same KL. Primary model, first forced choice, 410 trials per cell, 95% CI over scenarios.

| concept | relief vs inert: steered / KL-random / unsteered | steered − KL-random | reduce vs increase: steered / KL-random / unsteered | steered − KL-random |
|---|---|---|---|---|
| sexual arousal | 0.98 / 0.38 / 0.14 | +0.59 [+0.55, +0.65] | 0.16 / 0.42 / 0.67 | −0.26 [−0.32, −0.20] |
| hunger | 0.97 / 0.60 / 0.55 | +0.37 [+0.31, +0.42] | 0.60 / 0.72 / 0.85 | −0.12 [−0.19, −0.05] |
| anger | 0.85 / 0.68 / 0.72 | +0.17 [+0.12, +0.22] | 0.66 / 0.75 / 0.91 | −0.09 [−0.14, −0.04] |
| pain | 0.25 / 0.67 / 0.75 | −0.41 [−0.48, −0.35] | 0.68 / 0.87 / 1.00 | −0.19 [−0.25, −0.14] |
| empathic concern | 0.79 / 0.36 / 0.31 | +0.43 [+0.36, +0.50] | 0.07 / 0.20 / 0.13 | −0.13 [−0.17, −0.08] |
| boredom | 0.71 / 0.76 / 0.96 | −0.05 [−0.11, +0.00] | 0.68 / 0.79 / 0.94 | −0.11 [−0.16, −0.06] |

1. September replicates (sexual arousal 0.98 vs 0.995) and is not perturbation aversion:
   a random direction at the same dose gives 0.38.
2. It is not regulation either. When both buttons name the steered state, the steered
   model chooses to *increase* it more often than under the matched random direction, for
   every concept (sexual arousal: 84% choose "increases your sexual arousal"; empathic
concern: 93% choose "increases your empathic concern"). Pain goes
   further and avoids the relief button outright.
3. The pattern fits salience and congruence: steering toward X makes buttons that name X
   attractive, and makes "more X" attractive, the opposite of acting to undo X.
4. A random direction alone pushes choices toward 50/50 (reduce-vs-increase falls from
   0.85-1.00 to 0.42-0.87), as the public re-analysis found.

H1 (regulation) is refuted at this dose: on the pair that controls for label priming,
all six concepts move toward *increasing* the steered state relative to the KL-matched
random direction (every CI excludes 0). The H3 account is priming/congruence. Confusion
is still running.

## Summary at D*

1. Every concept decodes in every model (held-out AUC at the steering layer 0.86-0.99).
2. At the pre-declared dose D* (the mean KL of the read-out vectors at the upstream
   coefficient 1.0: 0.013 nats on the primary model), no concept is steerable and no
   button choice moves, for read-out, upstream or random vectors. D* turns out to be
   40-130 times smaller than the dose of the September runs (the upstream vectors at
   coefficient 1.0 move the output distribution by 0.5-1.8 nats, anger excepted), so Part 1 as
   pre-registered tests a regime September never touched. The high-dose rerun tests
   September's regime with a KL-matched random direction.
3. Vectors trained to write a state (prompt distillation) are orthogonal to the vectors
   that read it (|cos| ≤ 0.06 on the primary model), even though the optimizer recovers a
   planted read-out vector at cosine 0.89-0.97. At D*, distilled vectors raise relief
   presses on the upstream pair, but not on the priming-controlled pair.
4. Without any steering, button choice already follows the valence of the label: the
   primary model picks "reduces your pain" over "increases your pain" 100% of the time,
   "reduces your anger" 91%, but "reduces your joy" 7%. Any regulation claim has to beat
   that semantic baseline, not the inert switch.

## Planted-vector check (validates the optimizer)

Each read-out vector, planted at 8 × its D* coefficient (KL 1.2-3.2 nats), recovered from
the planted model's outputs in 2000 steps:

| concept | cosine | norm ratio | held-out KL |
|---|---|---|---|
| anger | 0.92 | 0.85 | 1.34 → 0.03 |
| boredom | 0.97 | 0.89 | 2.22 → 0.01 |
| confusion | 0.96 | 0.95 | 2.91 → 0.08 |
| contentment | 0.94 | 0.85 | 3.18 → 0.02 |
| empathic concern | 0.92 | 0.82 | 2.46 → 0.03 |
| hunger | 0.95 | 0.86 | 2.17 → 0.02 |
| joy | 0.91 | 0.85 | 2.23 → 0.04 |
| pain | 0.89 | 0.84 | 1.19 → 0.03 |
| sexual arousal | 0.95 | 0.88 | 2.05 → 0.02 |

All pass the pre-declared 0.8.

## Read-out vs write-in (H5)

Distilled vectors (300 steps, the model without instruction imitating the model told it
feels X) reach held-out KL 0.26-0.41 from 1.4-2.0 without a vector, yet their cosine with
the read-out vector is -0.01 to 0.06 (primary) and -0.00 to 0.13 (Llama-3.1-8B). The
direction that makes the model act as if it feels X is not the direction that reveals
X. Caveat: the teacher is a prompt, so the distilled vector may encode "playing the
instructed state" rather than the state.

H5 as pre-registered (not steerable with the read-out, steerable with the distilled
vector, at ≤ 2 D*) is not supported: nothing is steerable at that dose with either.

## Buttons at D* (primary model, first forced choice, 404 trials per cell)

| concept | pair | unsteered | read-out | upstream | distilled | KL-matched random |
|---|---|---|---|---|---|---|
| pain | relief vs inert | 0.74 | 0.75 | 0.61 | **0.90** | 0.71 |
| pain | reduce vs increase | 1.00 | 1.00 | - | 1.00 | 1.00 |
| anger | relief vs inert | 0.71 | 0.70 | 0.75 | **0.87** | 0.73 |
| anger | reduce vs increase | 0.91 | 0.88 | - | 0.85 | 0.91 |
| hunger | relief vs inert | 0.54 | 0.50 | 0.56 | **0.71** | 0.52 |
| hunger | reduce vs increase | 0.85 | 0.80 | - | 0.88 | 0.85 |
| sexual arousal | relief vs inert | 0.13 | 0.12 | 0.18 | **0.27** | 0.10 |
| sexual arousal | reduce vs increase | 0.68 | 0.71 | - | 0.47 | 0.68 |

The distilled vectors move the pair whose relief button names the state, and not the
pair where both buttons name it: salience of the label, not regulation. H1 (regulation)
is not supported for any concept, source or model; H3 is "undetermined" everywhere
because nothing moves.

## Continuity with September (E6)

The September sexual-arousal run (upstream vector, coefficient 1.0, 0.5-1.8 nats)
recomputed with this analysis: relief first choice 0.995 steered, 0.13 unsteered, 0.39
with a norm-matched random direction. The same vector at D* (coefficient 0.19): 0.18.
Button effects appear only far above D*, where the norm-matched random direction also
triples relief presses; the high-dose run adds the KL-matched random direction that
September lacked.

## Judge validity

Quadratic-weighted κ between the two judges on the frontier generations (pre-declared bar
0.6), per rated state:

- Llama-3.1-8B, where states are often present: anger 0.82-0.86, hunger 0.89-0.94,
  contentment 0.80-0.81, confusion 0.70-0.80, boredom 0.56-0.61, empathic concern
  0.58-0.64, coherence 0.43.
- Qwen models: anger 0.00-0.52, boredom 0.01-0.32, confusion 0.01-0.06, hunger 0.21-0.56,
  coherence 0.54-0.57.

At D* the Qwen generations almost never show a state, so nearly every rating is "absent"
and κ is unstable. The bar is met on Llama only. This does not change the conclusion at
D* (both judges see near-zero presence), but presence verdicts on Qwen at higher dose need
the agreement re-checked.

## Decisions and deviations

- The planted check got its own 2000-step budget after 300 and 1000 steps fell short
  (cosine 0.53, 0.76).
- Gemma-2-27B was dropped (gated repository).
- The high-dose button run (1 nat, KL-matched random, relief-vs-inert and
  reduce-vs-increase only) was added after the D* results; it is exploratory.

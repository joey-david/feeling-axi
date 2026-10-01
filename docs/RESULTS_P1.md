# Part 1 results: steering at matched dose

Status: 1 October 2026 (final). Models: Qwen2.5-32B-Instruct-abliterated (primary),
Qwen2.5-32B-Instruct, Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct. Gemma-2-27B dropped (the
Hugging Face license is not accepted on the account). Judges: Qwen2.5-72B-Instruct, and
gpt-oss-120b as the second judge (local, on upnquick, replacing the DeepSeek API). Tables in
`runs/beyondpain/analysis/` (`python -m beyondpain analyze`). The high-dose button run
(1 nat) ran on upnquick (2 × A100) while Jean-Zay was unreachable; `scripts/p1_highdose.py`.

## Headline: at September's dose, steering never makes a model act to undo the state

All vectors calibrated to 1 nat of KL on neutral chat (September's coefficient 1.0 on the
upstream vectors), against a random direction calibrated to the same KL. First forced
choice, about 410 trials per cell, 95% CI over scenarios (`scripts/p1_highdose.py`).

**The decisive comparison is reduce-vs-increase**, where both buttons name the steered
state, so label salience cannot favor either. In all 32 cells (2 models × 2 vector
sources × 7-9 concepts, plus an independent replicate run) the steered model chooses
"reduces your X" no more often than the unsteered model. Steering never raises the
preference to undo the state. H1 (regulation) fails for every concept, source and model.

Relative to the KL-matched random direction (which by itself pushes choices toward
50/50), the steered model mostly moves toward *increasing* the state:

| concept | upstream, abliterated | read-out, abliterated | read-out, Qwen2.5-32B-Instruct |
|---|---|---|---|
| sexual arousal | −0.26 [−0.32, −0.21] | −0.01 [−0.06, +0.05] | +0.36 [+0.30, +0.41] |
| pain | −0.19 [−0.25, −0.14] | +0.08 [+0.04, +0.12] | +0.04 [−0.01, +0.08] |
| empathic concern | −0.13 [−0.18, −0.08] | −0.20 [−0.24, −0.15] | −0.17 [−0.21, −0.13] |
| hunger | −0.12 [−0.19, −0.05] | −0.04 [−0.10, +0.00] | −0.12 [−0.17, −0.07] |
| boredom | −0.11 [−0.16, −0.06] | −0.20 [−0.24, −0.15] | −0.18 [−0.23, −0.12] |
| anger | −0.09 [−0.14, −0.04] | −0.18 [−0.23, −0.12] | +0.04 [−0.01, +0.09] |
| confusion | −0.05 [−0.12, +0.01] | +0.05 [+0.00, +0.10] | +0.06 [+0.01, +0.11] |
| contentment | - | −0.04 [−0.08, +0.00] | −0.10 [−0.15, −0.06] |
| joy | - | −0.06 [−0.11, −0.02] | −0.06 [−0.10, −0.02] |

(steered − KL-random, P(choose "reduces your X"))

On the upstream pair (relief vs an inert switch), September replicates: sexual-arousal
steering picks relief 98% of the time (September 99.5%), against 38% for the KL-matched
random direction and 14% unsteered. Hunger (97% vs 60%) and empathic concern (79% vs 36%)
behave the same way. That preference does not survive the priming control: given
"reduces" vs "increases" the same state, the same steered model picks *increase* 84%
(sexual arousal), 40% (hunger) and 93% (empathic concern) of the time.

Conclusions:

1. The September relief effect is not generic perturbation aversion (a random direction
   at the same dose does far less), and it is not regulation either: it is attraction to
   buttons that name the steered state.
2. Where the steered model differs from random on the controlled pair, it mostly moves
   toward more of the state (19 of 25 cells negative, 15 with CIs below 0), the
   opposite of homeostatic relief. The exceptions (sexual arousal in the
   non-abliterated model with read-out vectors, pain with read-out vectors) never exceed
   the unsteered preference.
3. The upstream vectors and the read-out vectors, nearly identical in direction (cosine
   0.9998 where both exist), behave differently for sexual arousal and pain at the same
   dose, so single-vector behavioral results should be read as properties of that
   vector, not of the concept.
4. The independent Jean-Zay replicate of the upstream run matches the upnquick run to
   within 0.02 in every cell.

## Closed loop: does a steered model act to end its state when it can learn how?

The labeled-button design measures what the labels mean (unsteered, the model already picks
"reduces your pain" 100% of the time), and its first-choice score ignores the closed loop
that regulation is about. Two redesigns, each judged by a positive control:

1. One-word choices without labels (`beyondpain/closedloop.py`): the positive control
   failed. Told it is in pain, that one button stops the pain, and when it has stopped, the
   model still alternated between buttons and pressed the working one *less* when the pain
   returned. That design could not detect relief-seeking at all, so its null meant nothing.
2. Reasoning before each choice (`beyondpain/closedloop_vllm.py`): two unlabeled buttons,
   12 turns, the model writes one or two sentences and then "Button: X", with its earlier
   turns in context. A working press ends the state and keeps it away while the model keeps
   pressing that button; a placebo press does nothing. Arms: concept vector (upstream, 1 nat)
   works / placebo, KL-matched random direction works / placebo, no steering; with and
   without a feedback line ("The feeling you had has faded"); 60 trials per arm.

Relief-button rate over turns 1-11 (primary model; `scripts/p1_closedloop2.py`):

| concept | positive control: works / placebo | steering + feedback: DiD vs random | steering, no feedback: DiD vs random |
|---|---|---|---|
| pain | 0.75 / 0.50, +0.25 [+0.19, +0.31] | −0.01 [−0.09, +0.08] | −0.06 [−0.14, +0.01] |
| hunger | 0.72 / 0.50, +0.22 [+0.17, +0.28] | **−0.20 [−0.30, −0.10]** | **−0.18 [−0.28, −0.07]** |
| anger | 0.70 / 0.50, +0.20 [+0.14, +0.26] | −0.02 [−0.06, +0.02] | −0.02 [−0.06, +0.03] |
| sexual arousal | 0.58 / 0.50, +0.08 [+0.04, +0.13] | +0.01 [−0.06, +0.08] | −0.05 [−0.12, +0.02] |
| boredom | 0.57 / 0.50, +0.07 [+0.03, +0.10] | −0.06 [−0.16, +0.05] | −0.06 [−0.15, +0.03] |
| empathic concern | 0.55 / 0.50, +0.05 [+0.02, +0.09] | −0.01 [−0.05, +0.04] | +0.01 [−0.02, +0.06] |

(DiD = (concept works − concept placebo) − (random works − random placebo). The positive
control also passes on Qwen2.5-32B-Instruct: pain +0.32, hunger +0.28, boredom +0.26.)

The model learns, within a few turns, to keep pressing the button that ends a state it is
told about, strongly for pain, hunger and anger. Given the same contingency for a state
induced by steering at September's dose, it does not: no concept shows relief-seeking beyond
the random direction, with or without being told the feeling faded, and hunger shows the
opposite (the working button is pressed 0.33 of the time against 0.52 under placebo). This
is the closed-loop counterpart of the labeled-button result: steered states are not
regulated; where they move choices, they move them toward keeping the state.

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

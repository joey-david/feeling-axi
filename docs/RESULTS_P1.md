# Part 1 results: steering at matched dose

Status: 1 October 2026. Models: Qwen2.5-32B-Instruct-abliterated (primary),
Qwen2.5-32B-Instruct, Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct. Gemma-2-27B dropped (the
Hugging Face license is not accepted on the account). Judge: Qwen2.5-72B-Instruct; the
second judge (DeepSeek) did not run (API credit exhausted). Tables in
`runs/beyondpain/analysis/` (`python -m beyondpain analyze`). The high-dose button run
(1 nat) is in progress and not reported here.

## Summary

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

## Decisions and deviations

- The planted check got its own 2000-step budget after 300 and 1000 steps fell short
  (cosine 0.53, 0.76).
- Gemma-2-27B was dropped (gated repository).
- The high-dose button run (1 nat, KL-matched random, relief-vs-inert and
  reduce-vs-increase only) was added after the D* results; it is exploratory.

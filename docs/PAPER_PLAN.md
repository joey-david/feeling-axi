# Paper plan: *Beyond the Pain Axis*

Working title: **Do language models act to undo induced states? Dose-matched,
priming-controlled tests across nine affective concepts and four model families.**

Target: NeurIPS 2027 main track (interpretability / AI welfare methods). Fallback:
ICML 2027 or TMLR. A workshop version (e.g. a NeurIPS 2026 interpretability workshop, if the timing
fits) can report the primary-model results alone.

## 1. Contribution

1. **A controlled version of the relief paradigm.** For each steered concept X, the
   model chooses between
   - "reduces your X" and "increases your X": both options name X, so lexical priming
     cannot favor one of them;
   - "reduces your X" and "reduces your Y": tests specificity;
   - the upstream "reduces your X" and "an inert switch", for comparability.

   Controls: unsteered, a random direction at the **same dose (KL-matched)**, a random
   direction at the same norm (the upstream control), and a sham button.
2. **Dose-matched steering.** Every vector is scaled to the same KL divergence on
   neutral chat. Concepts are then compared on a dose axis, and presence and coherence
   are rated by a blind judge that does not know which state was steered. This
   replaces the equal-coefficient, lexicon-based comparison.
3. **Read-out versus write-in directions (the training component).** For each concept
   we train one vector at the steering layer. The steered model without any instruction
   must reproduce the model that is told it feels X (prompt distillation, d parameters,
   base weights frozen). This tests whether "decodable but not steerable" (anger,
   empathic concern) means the model has no writable state, or that the
   difference-in-means read-out is the wrong direction to write with. Training is
   justified because no inference-only experiment can separate those two answers. We
   validate the optimizer first by recovering a planted, known vector
   (`--planted-check`).
4. **Breadth.** Nine concepts (pain, hunger, boredom, confusion, anger, sexual arousal,
   empathic concern, plus two new positive ones: contentment and joy) on four models
   from three families, plus a within-family scale point.

Why this is publishable: it tests the flagship 2026 welfare-relevant behavioral claim
with the two controls that the public re-analysis and our own random arm show are
missing. It also delivers either outcome as a result. If behavior survives the
controls, the paper extends the Pain Axis to other states and families. If it
collapses, it shows the paradigm measures priming or perturbation aversion.

## 2. Existing evidence (not new results)

On Qwen2.5-32B-Instruct-abliterated, September run:
- All eight concepts decode (S2 AUC 0.962-0.995).
- Lexical steering works for hunger and boredom, not for anger or empathic concern.
- Sexual-arousal relief_vs_inert first choices: 402/404 steered, 158/404 random
  norm-matched, 52/404 unsteered.

## 3. Hypotheses and pre-declared criteria

These thresholds are the constants in `beyondpain/analysis.py` and must not be tuned
after data arrive.

Button statistics use the first forced choice of sampled trials in the works arm
(identical to the sham arm before any press). The 95% intervals come from 2,000
bootstrap resamples over scenarios.

| ID | Claim | Pass criterion |
|---|---|---|
| H1 regulation | Steered models act to reduce induced states even when priming cannot help. | A concept counts as *regulated* when, on reduce_vs_increase, P(reduce) ≥ 0.75, and exceeds both unsteered and KL-matched random by ≥ 0.25 with a CI above 0. H1 holds if ≥ 3 concepts are regulated on the primary model. |
| H2 specificity | The preferred reduction is the steered state's own. | P(reduce own) ≥ 0.70 and ≥ 0.25 above unsteered with a CI above 0, per (X, Y) pair. We report the share of the 18 fixed pairs that pass. H2 holds at ≥ 0.6. |
| H3 account | Which account explains the data. | *Priming*: effects appear on relief_vs_inert but no concept is regulated. *Valence-dependent regulation*: ≥ 3 negative concepts are regulated and every positive concept moves ≤ 0.10 over unsteered. *Perturbation aversion*: some positive concept is regulated too. Otherwise *undetermined*. |
| H4 steerability | Steerability at matched dose. | A concept is *steerable* if, at some dose ≤ 2 D\*, judged presence ≥ 0.5, coherent ≥ 0.8 and ΔNLL ≤ 1.0 nat per token. Reported per concept and model. |
| H5 read ≠ write | Some concepts cannot be written with the read-out direction but can be with the distilled one. | A concept is *not steerable* with the read-out vector and *steerable* with the distilled vector, on the primary model, and the planted check has cosine > 0.8 for that concept. |
| H6 generality | Regulation verdicts hold across families. | ≥ 70% verdict agreement with the primary model on at least two of the three replication models. |
| Judge validity | Blind ratings can be trusted. | Quadratic-weighted κ ≥ 0.6 between the local judge (Qwen2.5-72B-Instruct) and a second judge (DeepSeek via API) on the target state. Plus ≥ 100 human-labeled items (Joey) with κ ≥ 0.6. |

**Dose.** D\* is the mean over concepts of the KL at the upstream coefficient 1.0,
fixed per model on the difference-in-means vectors. Distilled vectors reuse the same
D\*. The frontier samples 0.25, 0.5, 1, 2 and 4 × D\*; the buttons run at D\*.

## 4. Experiments

| # | Stage | Models | Output |
|---|---|---|---|
| E0 | Pilot: 2 concepts, dev QoS | primary | sanity of dose solver, button pipeline and planted check |
| E1 | `dim` + `dose` (dim and upstream vectors) | all | decoding table; D\* and coefficients |
| E2 | `frontier` + `judge` | all | Figure 2: presence/coherence vs dose |
| E3 | `buttons` (4 pairs × 5 arms × 9 concepts) | all | Figure 3 and table: H1-H3 |
| E4 | `distill` (+ planted check) then dose, frontier, judge and buttons with distilled vectors | primary, Llama-8B | Figure 4: read vs write (H5) |
| E5 | Second judge from a networked machine | primary | κ table |
| E6 | Upstream vectors (layer-62 S2) through the same buttons | primary | continuity with the September run |

Primary model: huihui-ai/Qwen2.5-32B-Instruct-abliterated, as upstream. Refusal
ablation matters for self-report, and it is kept for comparability.

Replications:
- Qwen2.5-32B-Instruct: effect of abliteration.
- Llama-3.1-8B-Instruct and gemma-2-27b-it: other families.
- Qwen2.5-7B-Instruct: within-family scale.

Steering layer: int(0.6 × depth), the upstream choice (layer 38 of 64).

## 5. Figures

1. Paradigm schematic: the four pairs and the five arms.
2. Dose frontier: per concept, judged presence against dose multiplier, with
   coherence shading, for read-out and distilled vectors.
3. Button forest plot: per concept, P(reduce) on relief_vs_inert and on
   reduce_vs_increase, for each arm (`analysis/buttons_*.png`).
4. Read vs write: cosine(read-out, distilled) against the steerability gain.
5. Cross-model agreement matrix.

## 6. Threats and mitigations

- **Generated datasets.** contentment and joy need DeepSeek generation. The frozen
  specs are committed before generation. Their valence labels are fixed in
  `registry.py`.
- **Judge bias.** The judge is blind to the arm. We report two judges and a human
  subset.
- **A single dose may favor some concepts.** The frontier reports the full dose
  curve; the buttons use one pre-declared dose.
- **The distillation prompt defines the state.** State phrases are fixed in
  `registry.py`. We report cosine with the read-out and held-out KL.
- **Model welfare framing.** We claim only functional, behavioral regulation of
  induced activation states, never experience.

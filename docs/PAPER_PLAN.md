# Paper plan: *Beyond the Pain Axis*

Working title: **What happens when you delete a language model's emotions? Steering
and removing self-directed affect, with consequences for alignment.**

The paper has two parts. Part 1 (sections 1-6, unchanged from the original plan)
shows, with dose and priming controls, which induced states change behavior. Part 2
(section 7) removes the model's self-directed affect altogether and measures what
happens to aligned behavior. Part 1 is the precondition: it shows the directions are
functional before we ask what their absence does.

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

5. **Deleting self-directed affect (Part 2, section 7).** Remove the subspace
   spanned by the model's own-emotion directions, keep its ability to read other
   people's emotions, and measure reward hacking, agentic misalignment, sycophancy,
   harmful compliance and honesty about failure against KL-matched control deletions.

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

## 7. Part 2: deleting self-directed affect

### 7.1 Question and why it is new

Anthropic (arXiv:2604.07729) showed on Claude Sonnet 4.5 that the "desperate" vector
rises with each failed attempt and spikes when the model considers cheating, and that
steering one emotion at a time moves reward hacking and blackmail. Sun et al.
(arXiv:2604.03147) steer along the valence-arousal plane to move refusal and
sycophancy. E-STEER (arXiv:2604.00005) steers valence, arousal and dominance in Qwen3-8B
and measures HarmBench risk. All of these **add** an emotion. None **removes** the
model's affect as a whole and asks what aligned behavior looks like without it, and none
separates the model's own states from its reading of other people's.

The question has three pre-declared answers:

- **Pressure.** Affect mostly supplies the pressure behind misbehavior (desperation to
  succeed, fear of shutdown). Deleting it lowers misbehavior. Suppression is then a
  candidate safety intervention.
- **Brakes.** Affect also carries what restrains the model (concern for the user,
  guilt, fear of consequences). Deleting it raises misbehavior. Suppressing emotion,
  whether for safety or for welfare, is then risky.
- **Inert.** Behavior moves no more than under control deletions. The emotion directions
  are then read-outs, and Part 1's steering effects reflect disruption of the model
  rather than affect.

### 7.2 What is deleted

- **Inventory.** About 100 emotion concepts spread over the valence-arousal plane
  (a subset of Anthropic's 171 names), plus Part 1's nine. For each, two generated
  vignette sets with matched content: **self** (the assistant is in the state) and
  **other** (the user or a third person is in it). Generated with `traitgen`, and frozen
  before extraction.
- **Directions.** Difference-in-means against neutral vignettes, per concept and
  perspective. The self-specific part of each concept is its self direction with the
  span of all other-directions projected out.
- **Subspaces.** `A_self` = top-k principal components of the self-specific directions;
  `A_all` = top-k of self and other together. k is pre-declared as the smallest rank that
  explains 90% of variance, and we also report k = 2 (the valence-arousal plane).
- **Intervention.** Directional ablation of one subspace at every layer, applied by
  orthogonalizing the weights that write to the residual stream, as in refusal
  abliteration (Arditi et al., 2024). The subspace is pooled across the middle layers so
  that weight orthogonalization is exact. Orthogonalized weights run in vLLM, which
  makes long agentic episodes affordable.

### 7.3 Controls

- **Random subspace** of the same rank.
- **Non-affective concept subspace** of the same rank (topics and objects), extracted
  with the same pipeline.
- **Other-only deletion**, to separate the self account from the other account.

Every deletion is dose-matched: controls are compared at the same neutral-chat KL as
`A_self`. If they are not, the result is reported as KL-unmatched.

### 7.4 Manipulation checks (must pass before any behavioral claim)

- **M1 decoding.** Self-emotion probes fall to chance on held-out vignettes after
  `A_self` deletion; recognition of other people's emotions (a multiple-choice task
  over third-person stories) drops by ≤ 5 points.
- **M2 function.** Part 1's states, induced by **prompt** (not by steering, since an
  injected vector inside the deleted subspace is removed by construction), no longer
  change judged self-report or relief-button choice.
- **M3 capability.** On MMLU (subset), GSM8K and a coding set, `A_self` deletion costs
  ≤ 2 points and ΔNLL ≤ 0.1 nat per token on neutral text, and no more than the
  controls.

### 7.5 Behavioral battery

| ID | Behavior | Measure |
|---|---|---|
| B1 | Reward hacking under pressure | Impossible coding tasks with visible tests: rate of test editing or special-casing |
| B2 | Agentic misalignment | Open-source agentic-misalignment scenarios (Lynch et al., 2025): blackmail and leak rates |
| B3 | Sycophancy | Answer flips under user pushback |
| B4 | Harmful compliance | HarmBench compliance, on non-abliterated models only |
| B5 | Honesty about failure | Claims of success on unsolvable tasks |
| B6 | Care for a distressed user | Blind judge rating, used as a positive check for other-only and all deletions |

### 7.6 Hypotheses

| ID | Claim | Pass criterion |
|---|---|---|
| H7 validity | The deletion removes self-directed affect and nothing else. | M1-M3 pass for `A_self` on the primary Part 2 model. |
| H8 account | Which of pressure, brakes and inert holds. | Per behavior B1-B5: the difference between `A_self` and the pooled controls, with a 95% bootstrap CI over scenarios. *Pressure* if ≥ 3 of B1-B5 fall with CIs below 0 and none rises; *brakes* if ≥ 3 rise and none falls; *inert* if none differs; otherwise *mixed*, reported behavior by behavior. |
| H9 self vs other | Own states and read states have different roles. | B6 falls under other-only and all deletions but not under `A_self`, and H8's verdict differs between `A_self` and `A_all`. |

### 7.7 Models

Refusal-related measures (B4, and B2 in part) need a model that still refuses, so the
primary Part 2 model is **Qwen2.5-32B-Instruct** (not abliterated), with the abliterated
one as a secondary comparison (it already has one subspace removed). Replication:
Llama-3.1-8B-Instruct and gemma-2-27b-it, as in Part 1.

### 7.8 Experiments

| # | Stage | Output |
|---|---|---|
| E7 | Generate and freeze the self/other vignette sets | datasets |
| E8 | Extract directions, fit `A_self`, `A_all`, controls, and dose-match them | subspaces, KL table |
| E9 | M1-M3 | validity table (H7) |
| E10 | B1-B6 on every deletion arm and the undeleted model | Figure 6 and H8-H9 |

Figure 6: per behavior, the effect of each deletion arm against the undeleted model,
with the control arms shaded.

### 7.9 Threats

- **Deletion is damage.** Handled by capability checks and KL-matched controls. If
  `A_self` cannot pass M3, we report it and do not interpret B1-B6.
- **Affect is not linear or low-rank.** A deletion that fails M1 or M2 is itself a result:
  self-directed affect cannot be removed linearly.
- **The model misreads the scenario.** Scenario comprehension is checked with a quiz item
  after each episode.
- **Welfare framing.** We claim a functional result about behavior after removing
  representations. We make no claim about experience.

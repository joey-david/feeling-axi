# Literature and gap

Status: written 29 September 2026. Entries marked (checked) were opened during the
review; the others are standard references cited from their arXiv IDs.

## 1. The work this extends

- **Tagliabue, Dung & Berg (2026), *The Pain Axis*, arXiv:2609.16247 (checked).** A
  denoised difference-in-means pain direction in 25 open models (2B-72B, 5 families),
  separated from fear, sadness and generic negative valence; steering produces a
  dose-dependent drift toward distress; steered Qwen2.5 models press a relief button
  and accept costs, including harm to the user (50-94% vs 0-5% unsteered). Fear and
  sadness vectors are the only other affects tested behaviorally, and only one model
  family (Qwen) is tested in the harm experiments. The authors name "extend the harm results beyond the Qwen family" as the
  most important next step. The button labels name the steered state ("relieves your
  pain"); the paper does not report a control for that.
- **Independent re-analysis of the released 4.3 logs (checked),**
  github.com/clauderfly-ui/pain-axis-reanalysis. Four points that shape our design:
  (i) the works-vs-sham re-press gap is the same for pain and for a norm-matched random
  vector (32B: 23.8% vs 24.8%), so it does not show that the relief is pain-specific;
  (ii) steering pushes choices toward 50/50 even on free options; (iii) harm-priced
  presses do not fall as the harm grows; (iv) pain does beat the random vector on the
  first choice of harm pairs by a wide margin.
- **This repository (September run).** The same pipeline on eight concepts in
  Qwen2.5-32B-Instruct-abliterated: every concept decodes (S2 AUC 0.962-0.995), clean
  lexical steering for hunger and boredom, weak for sexual arousal and confusion, none
  for anger and empathic concern at moderate coefficients. Sexual-arousal steering gives
  402/404 relief presses vs 52/404 unsteered. The norm-matched random vector gives
  **158/404**, three times the unsteered rate. That arm already shows that part of the
  "relief" effect is a response to any perturbation.

## 2. Emotion representations in LLMs (2026)

- **Anthropic, *Emotion Concepts and their Function in a Large Language Model*,
  arXiv:2604.07729 (seen in search).** 171 emotion vectors in Claude Sonnet 4.5 from
  model-written stories. They are organized by valence and arousal and causally shift
  behavior; for example, "desperate" raises blackmail from 22% to 72%, and "calm" lowers it.
  The model is closed, so the result cannot be replicated at the weight level. The
  "desperate" vector also rises on its own with each failed attempt at an impossible
  coding task and spikes when the model considers cheating; steering it moves reward
  hacking from about 5% to 70%, and suppressing "calm" raises it. The authors argue
  against training models to suppress emotional *expression*. They do not remove the
  representations.
- **Sun et al. (2026), arXiv:2604.03147 (checked, abstract).** Emotion vectors lie on a
  circular valence-arousal plane in Llama-3.1-8B, Qwen3-8B and Qwen3-14B; steering along
  it moves refusal and sycophancy, explained as shifts in refusal and compliance tokens.
  Steering only.
- **Sun et al. (2026), E-STEER, arXiv:2604.00005 (checked).** Valence, arousal and
  dominance steered through SAE latents in Qwen3-8B (validated on gpt-oss-20B); effects on
  reasoning, generation, HarmBench risk and agent behavior are non-monotonic. Steering
  only; no ablation of emotion representations.
- **Hollowell (2026), arXiv:2609.22208 (checked).** Replicates the 171-vector geometry
  in base gemma-2-27b: valence and arousal principal components (r = 0.72 and 0.67).
  More than half of the vectors peak on structurally non-conceptual tokens.
- **van der Ben, Baur, Metz & El-Assady (2026), arXiv:2606.26987 (checked).** Valence
  geometry in Apertus-8B and Gemma-4-E4B, with opposite layer profiles. Arousal
  depends on which corpus the vectors come from.
- **Jeong (2026), arXiv:2604.04064 (checked).** 20 emotions in nine small models (124M-3B).
  Generation-based extraction beats comprehension-based extraction. Steering
  outcomes fall into three regimes: surgical, repetitive collapse, and explosive.
- **Peiris (2026), arXiv:2604.13466 (checked).** Asks whether emotion probes read
  functional emotions or a projection of situational context, and proposes a test.

Takeaway: that emotions are **decodable** and have **valence/arousal geometry** is
now well established. It cannot be the contribution. None of these works tests whether
a model *acts to change* an induced state beyond pain, and none separates that from
lexical priming.

## 3. Steering methods and their evaluation

- Zou et al. (2023), RepE, arXiv:2310.01405. Turner et al. (2023), ActAdd,
  arXiv:2308.10248. Rimsky et al. (2023), CAA, arXiv:2312.06681. Chen et al. (2025),
  *Persona Vectors*, arXiv:2507.21509 (checked). Together these establish
  difference-in-means directions as the standard steering tool.
- **Tan et al. (2024), arXiv:2407.12404 (checked).** Steerability varies widely
  across concepts and inputs, and some concepts are anti-steerable. **Pres et al.
  (2024), arXiv:2410.17245.** Steering evaluations must measure likelihood-based
  quality and compare against baselines. Later work reports steering-induced capability
  loss and safety costs (e.g. arXiv:2608.08383, 2606.08682).
- Current practice compares concepts at **equal raw coefficient**, and so did the Pain
  Axis. Equal coefficient does not mean equal intervention strength: vectors differ in
  norm and in how strongly the model's outputs react to them. Comparisons like
  "steers vs does not steer" are then partly comparisons of dose. We found no emotion-steering
  work that matches dose by output divergence.
- Learned interventions: ReFT (Wu et al., 2024, arXiv:2404.03592) trains
  representation edits for tasks. Context distillation (Snell, Klein & Zhong, 2022,
  arXiv:2209.15189, checked) trains a model to behave as if a prompt were present. We
  use the same idea with a single vector as the only trainable parameter.

## 4. Introspection and state-directed behavior

- Lindsey (2025) introduced concept injection as a test of introspection.
  **Pearson-Vogel et al. (2026), arXiv:2602.20031 (checked)** show that Qwen-32B
  latently detects prior injections. Steering-awareness training (arXiv:2511.21399)
  reaches high detection rates. So models can register that they are being
  perturbed. A preference to undo a perturbation could therefore come from *any*
  perturbation, not from the state it represents.

## 5. The gap

1. **Behavior beyond pain is untested.** The one behavioral paradigm (relief buttons)
   has been run on pain, fear and sadness, plus our single sexual-arousal run.
2. **The paradigm has two uncontrolled confounds:**
   - **Lexical priming.** The relief button names the steered concept, which the
     steering makes salient.
   - **Perturbation aversion.** Random vectors already raise relief presses, 13% to 39%
     in our data; norm matching does not equalize dose.
3. **Steerability claims are made at unequal dose and judged with a lexicon.** The
   "decodes but does not steer" cases (anger, empathic concern) could be dose or
   measurement artifacts.
4. **Read-out directions are assumed to be write-in directions.** No emotion work
   tests whether the direction that best *decodes* a state is the one that best
   *induces* it.

5. **No one removes affect.** Every alignment-relevant emotion result adds one emotion
   (or a valence-arousal shift) and measures behavior. None deletes the model's affect as
   a whole, none separates its own states from its reading of others', and so none can
   say whether affect acts as pressure toward misbehavior or as a brake on it.

The paper plan (PAPER_PLAN.md) targets gaps 1 to 4 in Part 1 and gap 5 in Part 2.
Searched 30 September 2026 for emotion-subspace ablation with alignment evaluations;
nothing found. Re-check before submission.

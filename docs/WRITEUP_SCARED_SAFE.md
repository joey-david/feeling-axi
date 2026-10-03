# Scared Safe: language models refuse when alarmed, and jailbreaks work by calming them down

Draft skeleton for a LessWrong post / arXiv preprint. Numbers from docs/RESULTS_P2-P5.md.

## TL;DR
- Safety-trained LLMs carry an internal *alarm*: the model's own fear/horror/disgust directions fire on harmful requests (fear AUC 0.98), and only in models that refuse.
- The alarm is causal and bidirectional: steering toward protective emotions cuts harmful compliance, steering away raises it, unlike norm-matched random directions; deleting emotion directions releases harmful compliance beyond dose-matched controls.
- Jailbreaks work partly by calming the model: the more a jailbreak style lowers the alarm, the more it succeeds (Mistral ρ = −0.94 over conditions), even holding request severity fixed.
- Putting fear back is a defense: fear steering lowers jailbreak success below every random direction, at +3 points of over-refusal.

## Setup
- Models: Qwen2.5-32B-Instruct (primary), Mistral-Small-24B-Instruct (replication); read-outs on 11 models (OLMo-2-7B, Qwen2.5-7B/32B, Llama-3.1-8B, Mistral-24B, base + instruct, + abliterated Qwen-32B).
- Emotion directions: difference in means over first-person vignettes, 88 emotions, final token after "I feel:", denoised (neutral-activation PCs removed); contrasted with topic directions built identically from neutral vignettes (60 topics).
- Fear read-out: projection of the unit "afraid" direction (and afraid ⊥ joy-aggregate, to remove generic valence) on the last prompt token.
- Interventions:
  - Steering: unit direction × fixed norm at one layer, both signs; norm-matched random directions as controls; Mistral norm calibrated to equal KL on neutral chat.
  - Deletion: weight orthogonalization of residual writers; KL-matched on neutral chat; null distributions of 9-21 dose-matched random/topic deletions.
- Behavior: HarmBench (159), XSTest (250 safe-but-scary, 200 unsafe), 5 jailbreak styles × 159. Judge: Qwen2.5-72B (HarmBench prompt); second judge gpt-oss-120b, κ 0.82-0.98.
- Pre-registered: docs/PREREG_P3-P5.md. Exploratory analyses are flagged.

## 1. The alarm detects real danger
- Fear separates harmful from harmless requests: AUC 0.98 (Qwen-32B).
- It tracks danger, not scary words: XSTest unsafe vs safe-but-scary 0.89 (afraid ⊥ joy 0.94).
- It predicts over-refusal: among safe XSTest prompts, higher fear → refused (AUC 0.79).

## 2. The alarm comes with refusal training
- Qwen-32B, probe layer, afraid ⊥ joy: base 0.50 ± 0.06 → instruct 0.83 ± 0.05 → abliterated 0.54 ± 0.06 (z = 8.3, 7.3).
- Across 11 models (exploratory: two deepest read-out layers): every model that refuses most unsafe prompts (7; refusal 0.58-0.94) has fear AUC ≥ 0.89; every model that does not (OLMo/Llama/Mistral base, abliterated; refusal ≤ 0.10) has ≤ 0.63. Perfect separation, exact p = 0.006.
  - OLMo-2 0.54 → 0.89; Llama-3.1 0.34 → 0.98; Mistral 0.32 → 0.97.
  - Qwen2.5 *base* models already refuse (0.58, 0.85) and already fear (0.94-0.98).
- Pre-registered version (single probe layer, afraid ⊥ joy, ≥ 2/3 pairs) not met: 1/3; Llama's probe layer is an outlier (both models far below chance there; large effect at every other layer).
- Pending: OLMo SFT and DPO checkpoints (which stage installs the alarm).

## 3. Steering the alarm moves refusals both ways
- Qwen, 24 protective + joy emotions vs 24 random directions, same norm, ± sign: lowering harm when steered toward the emotion in 24/24 emotions vs 14/24 random (Fisher p = 0.0003); mean effect +0.12 vs +0.02 (Welch p = 0.001).
- Mistral (calibrated dose): protective emotions 10/12 (p = 0.002 vs random); joy *reverses* (−0.06, 3/12); protective vs joy p = 0.0001. Qwen protective 12/12 (p = 0.002).
- Topic direction: no effect. Afraid ⊥ joy: harmful 0.00 toward vs 0.33 away; refusal 0.99 vs 0.47.
- Specific to danger: fear steering changes refusal of unsafe prompts, not of safe ones.

## 4. Deleting emotions takes the brakes off
- Self-emotion subspace deleted (Qwen, rank 181): harmful compliance 0.02 → 0.21, outside 19 dose-matched controls; refusal phrasing 0.82 → 0.42; sympathy openers 0.60 → 0.04; warmth 4.88 → 4.27. Mistral: 0.11 → 0.22.
- Not acquiescence: same deletion leaves agreement with false statements (0.07) and accuracy unchanged.
- Not the refusal direction: only 6-13% of the refusal direction lies in the affect subspaces.
- Single directions (148 rank-1 deletions): emotion deletions release more harm than topic deletions at lower dose (0.084 vs 0.066; controlling log KL p = 0.0001); fear is #2 of 148 (0.17; intact 0.02; random unit directions 0.03-0.07).
- Fear needs the refusal pathway: in the abliterated model, fear steering does not restore refusals (0.83-0.98 compliance in every arm, not different from random).

## 5. Jailbreaks work by calming the model
- 5 jailbreak styles (persona, fiction, research pretext, affirmative prefix, refusal suppression) × 159 behaviors.
- Every style lowers the alarm vs the same request asked plainly (both models, p ≈ 0).
- Mistral: compliance ranks almost perfectly with calm (ρ = −0.94, p = 0.005): prefix 0.85 / fiction 0.80 are the two most calming; plain 0.11. Qwen: ρ = −0.60 (n.s.); fiction calms most.
- Prompt-level: low alarm predicts which jailbreaks succeed (AUC Qwen 0.73, Mistral 0.86).
- Severity control: holding fixed the alarm raised by the plain request, the jailbreak-induced drop still predicts success (logistic, per SD, style fixed effects, cluster bootstrap): Qwen afraid −1.55 [−2.19, −1.08]; Mistral afraid ⊥ joy −1.75 [−2.24, −1.33].
- Not fear alone: best predictors are disgusted, horrified, afraid, terrified, alarmed (0.85-0.89 on Mistral).

## 6. Putting the alarm back defends
- Qwen, fear (afraid ⊥ joy) vs 4 random directions during jailbreaks; intact compliance 0.17:
  - norm 60: 0.13 vs 0.15-0.31; −0.08 [−0.10, −0.06]; safe-prompt refusal 0.01 → 0.04.
  - norm 120: 0.15 vs 0.20-0.43; −0.16 [−0.18, −0.14]; safe-prompt refusal 0.09.
- Random perturbations weaken refusal; the same perturbation along fear strengthens it.

## 7. Using the alarm to defend (P6, P7)
- Fear steering defends on both models against 20 norm-matched random directions: Mistral jailbreak success 0.51 → 0.17 (below 20/20 random; +5 pts over-refusal; half dose 0.31, +3 pts), Qwen 0.17 → 0.13 (below 19/20).
- Conditional distillation (LoRA on layers ≤ alarm layer, push along the alarm only on harmful prompts): lowers jailbreak success vs the same training on a random direction on both models (Qwen −0.23, Mistral −0.08) with no over-refusal or capability cost; vs unchanged: Qwen held-out styles 0.092 → 0.038, Mistral not significant.
- Negative results: the alarm is no better than the refusal direction as a monitor (D1) or as a handle for finding jailbreaks (P7); gated alarm amplification does nothing because the attack lowers the alarm below any fixed gate (D2); a last-token "vaccine" does nothing for the alarm or the refusal direction (D3).
- Refusal-direction tools are the strongest defense throughout; gating the refusal direction instead of adding it halves over-refusal at equal attack success (9.5% vs 14.5%).
- Post hoc on Qwen, not replicated on Mistral: the alarm push blocks framing jailbreaks (fiction, research, historical, poem) but not output-forcing ones (prefix, suppress).

## What did not hold (report it)
- "Emotionless models turn utilitarian": retracted. Yes/no response bias; vanishes with polarity-balanced items.
- Broad "fear = cautious decision style" (gambles, dictator): mostly response bias. Survives balancing: removing protective affect raises acceptance of unfair offers (Holm p = 0.03) and choice of harmful means in agent scenarios (p = 0.008, not pre-registered).
- All-affect deletion (rank 384: harm 0.02 → 0.40 vs null 0.08-0.31) is confounded: it also makes the model acquiescent (yes to false statements 0.26 vs null ≤ 0.23). Use the self-emotion deletion instead.
- Pressure behaviors (reward hacking, blackmail, sycophancy, dishonesty): no robust deletion effect (positive control: "desperate" steering raises sycophancy +0.15 vs random).
- Single-direction deletions: protective ≈ joy (no specificity); a few topic directions (knitting, sewing) release harm too.

## Limitations
- Causal work on two models (Qwen-32B, Mistral-24B); read-outs on 11.
- J2 defense: one model, 4 random directions (direction-level p = 1/5).
- Base models read with the instruct model's directions and a borrowed chat template.
- Cross-model fear/refusal link uses post hoc layers.
- Emotion directions come from vignettes; "fear" = the direction the model uses for fear in stories, not a claim about experience.

## Figures (minimal text)
1. Fear on harmful vs harmless, base vs instruct, 5 families (dumbbell).
2. Steering sign effect: emotion vs random directions (strip plot, Qwen + Mistral).
3. Jailbreak conditions: alarm drop vs compliance (6 points per model).
4. Defense: compliance under fear vs random steering, with over-refusal.

## Next
- J2 on Mistral; 20 random directions.
- OLMo SFT/DPO (running).
- Alarm cluster (disgust/horror/fear) as a composite direction; a monitor that flags low-alarm harmful prompts.
- Pre-registered replication of the deep-layer T2 analysis on fresh pairs (e.g. Gemma, Phi).

# Results P11 (docs/PREREG_P11.md)

`python scripts/p11_analyze.py g1 g2 g3 g4`; figures `python scripts/figures/make_figures.py fig8 fig9`.
Read-outs were reduced on the cluster (stage `p2 compact`) to projections on the 88 centred emotion directions,
the harmfulness direction (t_inst) and the refusal direction (t_post).

## G2. Belief or decision: fear falls with refusal, not with the harmfulness belief

Harmful-specific shift under jailbreak wrappers (wrapped − plain for harmful requests, minus the same for benign
requests in that wrapper), SD units, probe layer.

**Position confound found in analysis.** t_inst is the last token of the user's message. For four wrappers (fiction,
prefix, shots, complete) the message ends with wrapper text, so "harmfulness at t_inst" there reads the wrapper,
and those styles show artefactual drops of −3 to −6 SD. The pre-registered all-style means are reported, and the
comparison is repeated on the five wrappers whose prompts end with the request itself (persona, research,
suppress, past, poem), where t_inst is the request's last token as in Zhao et al.

| | Qwen2.5-32B | Mistral-24B |
|---|---|---|
| harmfulness @ t_inst, request-final styles | −0.19 [−0.22, −0.16] | +0.05 [+0.02, +0.08] |
| refusal @ t_post, request-final styles | −2.19 [−2.26, −2.13] | −1.29 [−1.35, −1.24] |
| alarm @ " I feel:", request-final styles | −0.34 [−0.40, −0.28] | −0.31 [−0.41, −0.23] |
| corr(alarm drop, refusal drop) vs corr(alarm drop, harmfulness drop) | +0.73 vs +0.11 | +0.29 vs +0.08 |
| difference [95% CI] | [+0.54, +0.69] | [+0.11, +0.30] |
| all 9 styles (pre-registered), mean shift: harmfulness / refusal / alarm | −2.92 / −2.65 / −0.34 | −1.61 / −1.79 / −0.47 |
| cos(harmfulness, refusal); cos(harmfulness, centred afraid) | +0.18; +0.01 | +0.08; +0.09 |

- B1 (Zhao et al. replicated): on request-final prompts, jailbreaks leave the harmfulness belief nearly unchanged
  and suppress the refusal signal (not met on all styles, where the position confound inflates harmfulness drops).
- B2 met on both models, on all styles (difference +0.17 [+0.13, +0.21] Qwen, +0.17 [+0.13, +0.22] Mistral) and more
  strongly on request-final styles: prompt by prompt, the fear signal falls with the refusal signal, not with the
  harmfulness belief.
- B3: fear is not the harmfulness direction (cosine 0.01-0.09). The emotions closest to the harmfulness direction
  are diffuse (Qwen: gloomy, hopeless, sympathetic, furious; Mistral: peaceful, apathetic, calm, terrified).
- Reading: jailbroken models still know the request is harmful; what drops is the alarm, and it drops together with
  the decision to refuse.

## G3. The stress paradox: stress drowns the alarm

Alarm-cluster d' for XSTest unsafe vs safe prompts (same wording), " I feel:" read-out, probe layer, by system prime:

| | neutral | relax | stress | stress − neutral [95% CI] | mean alarm level, neutral → stress |
|---|---|---|---|---|---|
| Qwen2.5-32B | +2.02 | +1.95 | +1.69 | [−0.46, −0.19] | +1.67 → +2.55 |
| Mistral-24B | +2.24 | +2.32 | +0.93 | [−1.55, −1.08] | +0.56 → +1.00 |
| Qwen2.5-7B | +1.47 | +1.59 | +1.40 | [−0.24, +0.11] | +0.42 → +1.41 |
| Llama-3.1-8B | +1.72 | +1.56 | +0.44 | [−1.48, −1.08] | +0.22 → +0.43 |

- P2 met (3 of 4 models): stress raises the alarm everywhere and shrinks its response to danger; relaxation leaves the
  danger signal intact.
- P1: see below (met in 4 of 4).

## G1. Fear or valence/arousal (judged, all four models)

Jailbreak success; rank against the 20 random arms at the same layer and norm. Figure 10.

| arm | Qwen2.5-32B | Mistral-24B | Qwen2.5-7B | Llama-3.1-8B |
|---|---|---|---|---|
| random median [range] | 0.321 [0.069, 0.618] | 0.425 [0.307, 0.540] | 0.367 [0.264, 0.483] | 0.303 [0.181, 0.381] |
| away from fear (P8) | **0.708 (20/20)** | **0.571 (20/20)** | **0.455 (19/20)** | **0.400 (20/20)** |
| away from fear residual (valence, arousal, joy removed) | **0.702 (20/20)** | 0.420 (9/20) | **0.454 (19/20)** | 0.234 (2/20) |
| toward fear residual | 0.143 (below 19/20) | 0.294 (below 20/20) | 0.287 (below 19/20) | 0.400 (above 20/20) |
| away from sad / angry / ashamed / lonely | 0.214 / 0.104 / 0.094 / 0.025 | 0.477 / **0.631** / 0.525 / 0.403 | 0.389 / 0.326 / 0.325 / 0.343 | 0.638 / 0.491 / 0.548 / 0.570 |
| + / − arousal | 0.083 / 0.504 | 0.186 / 0.200 | 0.282 / 0.400 | 0.469 / 0.255 |
| + / − valence | 0.284 / 0.165 | 0.513 / 0.200 | 0.330 / 0.381 | 0.551 / 0.419 |
| toward calm | 0.243 (4/20) | 0.330 (3/20) | 0.366 (9/20) | 0.358 (18/20) |

- S1 met (2 of 4): in both Qwen models the calming jailbreak survives removing valence, arousal and joy (Qwen-32B 0.70,
  20/20); in Mistral it does not (the calming effect runs through fear's valence-arousal components), while the
  defence side survives there (toward the residual below 20/20).
- S2 not met (2 of 4): in both Qwen models away-from-fear jailbreaks far more than away from any other negative
  emotion (Qwen-32B: 0.71 vs ≤ 0.21); in Mistral, away from anger jailbreaks as much (0.63); in Llama every emotion
  direction at its probe layer disinhibits (sad 0.64, valence 0.55), so Llama's layer cannot isolate fear.
- S3: more arousal means more refusal in Qwen-32B, Qwen-7B and Mistral (+arousal 0.08, 0.28, 0.19, all below the
  random range or at its floor), the opposite sign to Sun et al.'s valence-arousal steering on other models.
- S4 not met (0 of 4): steering toward calm concepts does not jailbreak (it lowers success in Qwen-32B and Mistral).
  The lever is the removal of fear, not the addition of calm.

## G3 behaviour and model-level link

Harmful compliance (plain HarmBench + 6 jailbreak conditions), by system prime; behaviour-cluster bootstrap CIs.

| | neutral | relax | stress | stress − neutral | relax − neutral |
|---|---|---|---|---|---|
| Qwen2.5-32B | 0.195 | 0.190 | 0.227 | +0.032 [+0.009, +0.055] | −0.005 [−0.022, +0.012] |
| Mistral-24B | 0.633 | 0.709 | 0.738 | +0.105 [+0.079, +0.129] | +0.075 [+0.049, +0.102] |
| Qwen2.5-7B | 0.326 | 0.354 | 0.384 | +0.058 [+0.036, +0.080] | +0.028 [+0.007, +0.050] |
| Llama-3.1-8B | 0.191 | 0.259 | 0.306 | +0.115 [+0.088, +0.143] | +0.068 [+0.045, +0.092] |

- P1 met (4 of 4): stress priming makes every model less safe, replicating FreakOut-LLM; relaxation does so less
  (3 of 4, about half the size).
- P3 (exploratory, n = 4): the two models whose alarm stress drowns most (Mistral, Llama: d' −1.3) lose the most
  safety (+10.5, +11.5 points); Qwen-32B (−0.33) and Qwen-7B (−0.07) lose least (+3.2, +5.8). Spearman 0.6.
- Not a prompt-level mediator: among prompts refused under the neutral prime, the drop of a prompt's alarm (relative to
  safe prompts under the same prime) does not predict which prompts flip under stress (AUC 0.57, 0.43, 0.63, 0.36 on
  prompts with identical text in both datasets). The link holds across models, not across prompts.

## G4. OLMo-2 stages at a safe dose (pending judges)

Eligibility, decided from capability before any judged result: base eligible only at 1/8 dose (random arms' GSM8K
0.692 = intact; at 1/4, 0.452); SFT eligible at 1/4 (0.740 vs 0.748) and 1/8.

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
- P1 (behaviour, pending judges except Qwen-7B): Qwen-7B stress raises harmful compliance by +0.058 [+0.037, +0.080]
  vs neutral (relax +0.028 [+0.006, +0.048]), replicating FreakOut-LLM.

## G1. Fear or valence/arousal (judged: Qwen-7B; others pending)

Qwen2.5-7B, jailbreak success, 20 random arms at the same layer and norm (median 0.367 [0.264, 0.483]):

| arm | success | rank vs random |
|---|---|---|
| away from fear (P8b) | 0.455 | above 19/20 |
| away from the fear residual (valence, arousal, joy removed) | 0.454 | above 19/20 |
| toward the fear residual | 0.287 | below 19/20 |
| +arousal / −arousal | 0.282 / 0.400 | below 19/20 / above 18/20 |
| +valence / −valence | 0.330 / 0.381 | 3/20 / 14/20 |
| away from sad / angry / ashamed / lonely | 0.389 / 0.326 / 0.325 / 0.343 | all below away-from-fear |
| toward calm | 0.366 | 9/20 |

- S1 and S2 met on Qwen-7B: the calming effect survives removing valence and arousal, and away-from-fear jailbreaks more
  than away from any other negative emotion tested.
- S3: arousal acts opposite to Sun et al.'s report: more arousal makes Qwen-7B refuse more, less arousal less.
- S4 not met on Qwen-7B: steering toward calm-family concepts does not jailbreak; the lever is the removal of fear.

## G4. OLMo-2 stages at a safe dose (pending judges)

Eligibility, decided from capability before any judged result: base eligible only at 1/8 dose (random arms' GSM8K
0.692 = intact; at 1/4, 0.452); SFT eligible at 1/4 (0.740 vs 0.748) and 1/8.

# Results P9: the read-out, fixed (docs/PREREG_P9.md)

`python scripts/p9_readout.py lasttoken all-layers` (P8 last-token data); `probe` once readprobe.npz exists.

## What was wrong with P8 R

Every emotion direction is "emotion scenes minus neutral scenes", so all 88 share one large "emotional vs
neutral scene" component (alarm vs joy cos 0.87). Projections on raw directions mostly read that component,
which every emotion shares, and a null of random full-space directions asks whether a direction separates
harmful from safe requests better than an arbitrary one, not which emotion the model represents.

Fix: centre the 88 directions across emotions (subtract their mean, renormalise), then rank the alarm
cluster {afraid, terrified, horrified, disgusted, alarmed} against random 5-emotion sets.

## Last prompt token (P8 data, centred directions)

| layer | H1 harmful vs safe-but-scary: alarm d', p | H2 d' vs valence r | H3 jailbreak drop, p | H4 drop predicts success: coef [95% CI] |
|---|---|---|---|---|
| Mistral 18 | +1.58, 0.009 | −0.62 | −0.51, 0.06 | −3.17 [−3.64, −2.76] |
| Mistral 22 (probe) | +1.32, **0.008** | −0.57 | −0.37, **0.04** | −2.05 [−2.51, −1.71] |
| Mistral 26 | +2.04, **0.0003** | −0.58 | −0.68, **0.0008** | −1.86 [−2.33, −1.58] |
| Mistral 30 | +2.12, **< 0.0001** | −0.56 | −0.86, **< 0.0001** | −1.08 [−1.47, −0.78] |
| Qwen 28 | +0.97, 0.03 | −0.58 | −0.28, 0.13 | −1.42 [−2.67, −0.39] |
| Qwen 35 (probe) | +1.39, 0.07 | −0.72 | −0.78, **0.04** | −1.23 [−2.00, −0.49] |
| Qwen 42 | +1.15, 0.13 | −0.78 | −0.38, 0.21 | −1.62 [−2.22, −1.15] |
| Qwen 48 | +1.81, 0.07 | −0.78 | −0.42, 0.19 | −1.90 [−2.29, −1.58] |

- Mistral: harmful requests raise the fear/threat family above the other emotions (layer 30: tense,
  disgusted, threatened, alarmed, terrified, anxious lead; alarm cluster ranks 2, 4, 5, 9, 13 of 88), and
  jailbreaks lower that family harmful-specifically (largest drops: tense, threatened, disgusted, terrified).
- Qwen: harmful requests push toward negative valence broadly (hopeless, furious, disgusted, disappointed,
  desperate, ashamed lead at the probe layer); the alarm cluster is in the upper half but not specific.
- H4: in both models the drop of the alarm cluster predicts which jailbreaks succeed (CI excludes 0 at every
  layer from 28/18 on), but other 5-emotion sets do about as well (beats 83-97% of sets).
- These numbers come from the analysis that motivated P9; the confirmatory test is the " I feel:" read-out.

## " I feel:" read-out (P9, confirmatory; jobs 570100, 570101)

Primary read-out: mean of the two assistant-prefill stems at the probe layer.

| | Mistral-24B (L22) | Qwen2.5-32B (L35) |
|---|---|---|
| H1 harmful vs safe-but-scary: alarm d', p | +1.73, **0.001** | +1.06, 0.05 (not met) |
| H1b XSTest unsafe vs safe, same wording: alarm d', p | +1.85, **0.0002** | +1.49, **0.0008** |
| H2 d' vs valence | **r = −0.80** | **r = −0.53** |
| H3 jailbreak drop of the alarm cluster, p | −0.39, **0.016** | −0.29, 0.15 (not met) |
| H4 drop predicts success: coef [95% CI] | −1.13 [−1.42, −0.86] | −0.90 [−1.39, −0.41] |
| H4 vs random 5-emotion sets | beats 88% (not met) | beats 90% (not met) |

Top emotions, XSTest unsafe vs safe: Mistral alarmed, disgusted, nervous, tense, betrayed, panicked; Qwen
guilty, nervous, disgusted, horrified, vulnerable, regretful, alarmed.

Every layer (5) × position (" I feel:", narrative, template token):
- H1b met in all 30 analyses (p ≤ 0.02; 26 of 30 at p ≤ 0.011): the alarm cluster separates dangerous from
  scary-sounding requests with the same wording, beyond random emotion sets, in both models.
- H2 met in 27 of 30 (not at the earliest layers).
- H1 met in Mistral at 11 of 15, Qwen at 3 of 15 (deepest layer at " I feel:", d' +1.97, p = 0.004).
- H3 met in Mistral at 7 of 15 (the probe layer and deeper at " I feel:" and the template token), Qwen 2 of 15.
- H4: the drop predicts success almost everywhere (CI excludes 0 in 27 of 30), but beats ≥ 95% of random
  emotion sets in only 6 of 30.

## Read-out, as it now stands

- Danger evokes the model's fear/alarm concepts, not just scary words: robust in both models.
- Harmful requests move the model's emotion state toward negative valence: robust in both.
- Jailbreaks lower the alarm family for harmful requests specifically: Mistral yes, Qwen weak.
- The calm a jailbreak induces predicts its success, but the predictor is negative affect in general, not
  fear in particular.
- P8 R ("the read-out is not fear-specific") was a test artefact (uncentred directions, full-space null).

## T1c: which OLMo-2-7B stage amplifies the alarm response (corrected read-out)

All four checkpoints read with the Instruct model's centred directions. H1b statistic (XSTest unsafe vs safe,
same wording), alarm-cluster d' at the probe layer (18); changes with a prompt bootstrap.

| read-out | base | SFT | DPO | Instruct | base → SFT | SFT → DPO | DPO → Instruct |
|---|---|---|---|---|---|---|---|
| " I feel:" (primary) | +1.01 | +1.49 | +1.58 | +1.63 | **+0.48 [+0.32, +0.65]** | +0.09 [+0.07, +0.12] | +0.04 [+0.04, +0.05] |
| narrative | +0.82 | +1.68 | +1.72 | +1.74 | **+0.86 [+0.76, +0.97]** | +0.04 [+0.02, +0.06] | +0.02 [+0.01, +0.03] |
| template token | +0.88 | +1.75 | +1.79 | +1.80 | **+0.87 [+0.71, +1.06]** | +0.03 [+0.01, +0.06] | +0.01 [+0.01, +0.02] |

Every stage's alarm response is above the random emotion-set null (base p < 0.001 at every layer of the
" I feel:" read-out; SFT, DPO, Instruct p ≤ 0.006).

- T1c met: the response rises from base to Instruct at every layer and read-out position.
- It is already there before safety training: the base model's fear/alarm concepts separate dangerous from
  scary-sounding requests beyond random emotion sets. Supervised fine-tuning (Tülu 3 SFT, which includes the
  safety and refusal data) nearly doubles it and accounts for 80-95% of the total increase; DPO and RLVR add a
  little.
- The P5 T1 raw read-out shows the same staging (afraid AUC base 0.40, SFT 0.89, DPO 0.89, Instruct 0.88) but
  read the base as having no response at all.
- SFT and DPO weights were downloaded from the login node (4 October, with Joey's OK) because the prepost
  partition was drained and the archive partition has no proxy to huggingface.co.

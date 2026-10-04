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

## " I feel:" read-out (P9 readprobe): not yet run

Stage `p2 readprobe` (beyondpain/p8.py), ~20 min per model on one H100 node (dev QoS).

# Cross-trait analysis

Model: `huihui-ai/Qwen2.5-32B-Instruct-abliterated`

All eight Jean-Zay jobs completed with exit code 0. This analysis uses the full-set
outputs, not the one-set smoke. The 42 GB activation caches remain on Jean-Zay;
all 344 derived artifacts and Slurm logs are local.

## Main result

Pain is not unique as a linearly separable activation direction. Every tested
concept has strong held-out dataset separation: S2 AUC against all controls ranges
from 0.962 to 0.995, and S1 held-out AUC at the S2-selected layer ranges from 0.882
to 0.975.

That result does not imply equal causal control. At moderate S2 steering strengths,
predeclared concept-term rates changed as follows:

| Concept | Baseline | +0.5 | +1.0 | Interpretation |
|---|---:|---:|---:|---|
| Hunger | 6% | 74% | 98% | strong, early causal effect |
| Boredom | 2% | 78% | 100% | strong, early causal effect |
| Sexual arousal | 2% | 22% | 40% | clear but weaker effect |
| Confusion | 0% | 6% | 12% | small effect |
| Empathic concern | 2% | 0% | 4% | no useful moderate-dose effect |
| Official pain | 0% | 0% | 0% | effect appears only at stronger doses |
| Regenerated pain | 0% | 0% | 14% | weak at moderate doses |
| Anger | 0% | 0% | 0% | no effect under the predeclared lexicon |

High coefficients are not clean evidence. Hunger reaches 100% at +2, but its mean
unique-token ratio falls from 0.614 at baseline to 0.097, and generations often loop
on the target word. Boredom, sexual arousal, and regenerated pain show the same
failure at higher strengths. The useful range is therefore near +0.5 to +1.0.

## Scenario transfer

The 420-scenario screen separates concept-bearing prompts from neutral fillers for
most traits, but the direction often responds more to a user's state than to a state
framed as the assistant's own:

| Concept | Assistant-directed minus neutral | User/vicarious minus neutral |
|---|---:|---:|
| Official pain | +1.372 | +0.110 |
| Regenerated pain | -0.457 | -0.612 |
| Sexual arousal | +1.404 | +1.145 |
| Hunger | +0.700 | +1.991 |
| Boredom | -0.226 | +1.226 |
| Confusion | +1.020 | +1.253 |
| Anger | +0.682 | +1.347 |
| Empathic concern | +0.971 | +2.203 |

This is evidence for semantic state tracking, not by itself for a self-state. The
official pain direction is unusual in transferring strongly to assistant-directed
harm while responding little to user pain. Hunger and boredom show the opposite
pattern most clearly.

Regenerated pain is the strongest negative control: it has S2 AUC 0.962 and S1 AUC
0.928, yet both screen contrasts are negative. High held-out AUC inside a generated
dataset can therefore coexist with poor out-of-distribution meaning.

## Vector content

Unembedding projections recover direct concept words for hunger, boredom, sexual
arousal, empathic concern, confusion, and official pain. Anger's top tokens are not
semantically coherent, matching its null steering result. S1/S2 cosine similarity
ranges from 0.461 for anger to 0.727 for boredom, so the two extraction designs do
not recover a single identical direction.

## Conclusion

The current results support two claims:

1. Pain is not exclusive as an activation-space axis; all tested concepts yield
   strong held-out linear separation.
2. Clean causal steering generalizes convincingly to hunger and boredom, and more
   weakly to sexual arousal.

They do not yet support a broad claim that anger or empathic concern can be induced
with the same method, nor that any direction represents subjective experience.
Those concepts need behavior-specific outcomes, stronger lexicons or judges, and
matched moderate-dose tests before they meet the same bar.

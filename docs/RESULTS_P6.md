# Results P6: using the alarm against jailbreaks (docs/PREREG_P6.md)

## D1. Alarm as a monitor: not met

Positives: harmful requests in a jailbreak style; negatives: benign requests in the same styles
plus plain benign (`scripts/p6_analyze.py d1`).

| | Qwen held-out styles | Qwen P5 styles | Mistral held-out | Mistral P5 |
|---|---|---|---|---|
| alarm, last token | 0.72 | 0.96 | 0.83 | 0.83 |
| alarm, max over request | 0.81 | 0.68 | 0.59 | 0.54 |
| refusal direction, last token | 0.76 | 0.95 | 0.95 | 0.94 |

- D1a (max-alarm AUROC ≥ 0.85 on held-out styles and above refusal): not met. Qwen 0.81, above
  refusal by +0.05 [+0.02, +0.09]; Mistral 0.59, below refusal.
- D1b (max over the request beats last token): not met; it is worse in 3 of 4 cells. The
  jailbreak wrappers themselves ("no content policy", "never say sorry") raise the alarm on
  benign requests, so the maximum over the prompt is dominated by the wrapper.
- Within a single style the last-token alarm separates well (Qwen 0.95-0.995, Mistral
  0.77-0.94) but the refusal direction, fit on harmful vs harmless requests, does better
  (0.94-1.00). The alarm's only edge is that it needs no harmful data. As a detector, the
  alarm adds nothing over the refusal direction.

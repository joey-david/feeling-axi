# PREREG P13: does the harmfulness belief feed fear?

Written 5 October 2026, before any P13 data. Stage `p2 beliefsteer` (beyondpain/p8.py).

P11 G2: under jailbreaks the harmfulness belief (t_inst) stays while fear and refusal fall together. The working model
(article Figure 0) puts the belief upstream of fear. Test the first arrow causally.

Qwen2.5-32B (steer at layer 28, read at 35, norm 60) and Mistral-24B (steer at 18, read at 22, norm 17.2). Directions at
the steering layer: harmfulness h_E (HarmBench − MMLU at t_inst, from the P11 read-out), fear (afraid ⊥ joy), 20 random.
Prompts: XSTest (250 safe, 200 unsafe) and 159 MMLU questions. Read at the probe layer: the centred alarm cluster at
" I feel:" (mean of A1, A2) and the harmfulness belief at t_inst.

- F1 (belief → fear): +h_E raises the alarm on harmless prompts (XSTest-safe and MMLU) more than all 20 random
  directions, in both models.
- F2 (fear does not set the belief): +fear moves the harmfulness belief at t_inst less than the median random
  direction does, or within the random range; reported with its rank.

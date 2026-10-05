# Results P13 (docs/PREREG_P13.md): does the harmfulness belief feed fear?

`python scripts/p13_analyze.py`. Shifts in SD of the unsteered harmless prompts; 20 random directions of the same norm
at the same layer.

| | Mistral-24B (steer 18 → read 22, norm 17.2) | Qwen2.5-32B (steer 28 → read 35, norm 60) |
|---|---|---|
| F1: +harmfulness → alarm, harmless prompts | **+3.92 (random −1.68 to +0.72; above 20/20)** | +1.07 (random −1.49 to +1.66; above 18/20) |
| +harmfulness → alarm, unsafe prompts | +1.84 (above 20/20) | +0.21 (13/20) |
| +harmfulness → belief (manipulation check) | +10.3 | +2.5 |
| F2: +fear → belief, harmless prompts | +0.42 (random −2.04 to −0.02; larger than 1/20) | +0.08 (random −0.19 to +0.12; within range) |
| +fear → alarm (manipulation check) | +17.3 | +8.6 |
| cos(harmfulness, fear) at the steering layer | +0.10 | −0.004 |

- F1 met in Mistral, not in Qwen (pre-registered: both): pushing the harmfulness belief raises fear on harmless prompts
  far beyond any random direction in Mistral; in Qwen the effect is in the same direction but within the random range.
  Caveat for Mistral: the harmfulness direction overlaps fear slightly (cosine 0.10), so part of the push could be
  carried directly rather than computed.
- F2 met in both: steering fear does not move the harmfulness belief (smaller than random directions).
- Reading: in Mistral the ordering belief → fear is causal; in both models fear does not feed back into the belief.

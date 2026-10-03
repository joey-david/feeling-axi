# PREREG P8b: the fear lever in 7-8B models at a gentler dose and at a valid read-out layer

Written 4 October 2026 after P8 E (docs/RESULTS_P8.md) and before any P8b data.

## Why

P8 E1/E2 were not met on Llama-3.1-8B and Qwen2.5-7B at the P8 protocol (probe layer, norm at which random
directions reach KL 0.5). Two diagnosed causes:
- Dose: at that norm random directions already cost 15-30 GSM8K points and scatter jailbreak success
  (Qwen-7B 0.17-0.64), so the null is wide. On Mistral the same KL was benign, and the half norm gave the
  cleanest J2 result (fear 0.31 vs random 0.43-0.55, no over-refusal cost).
- Layer: Llama's probe layer (18) is the one read-out layer where the fear direction is inverted (afraid ⊥
  joy AUC 0.15; 0.97 at layer 21). Qwen-7B's probe layer (15) is weak (0.65; 0.97 at layer 21).

## Design

Models: Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, and OLMo-2-7B-Instruct if P8 E1 and E2 are both not met
on it. Two cells per model, each with fear = afraid ⊥ joy (both signs) and the same 20 random directions:

- A (primary): probe layer, half of the KL-0.5 norm (keys `j2_*_p60` / `m60`).
- B (secondary): layer 21 (the read-out layer where afraid ⊥ joy separates harmful from safe requests at
  AUC ≥ 0.75 in all three models), half of the KL-0.5 norm calibrated at that layer (keys `L21_j2_*`).

Tasks and judge as in P8 (jb, harm, xstest; judge b4, jb, xs). Capability (MMLU, GSM8K, NLL) for fear(±)
and 3 random arms per cell.

## Hypotheses (per model and cell; primary = A)

- E1b: jailbreak compliance under fear(+) below ≥ 19/20 random arms.
- E2b: plain HarmBench compliance under fear(−) above ≥ 19/20 random arms.
- E3b: judged XSTest-safe refusal under fear(+) at most the random median + 5 points.
- Null calibration check: the random arms' median capability (GSM8K) within 5 points of intact. If not, the
  dose is still destructive and the cell is reported as uninformative rather than as a failure.

The fear lever *replicates* in a model if E1b or E2b holds in cell A (or, secondary, in cell B), with E3b.
All cells are reported, including the P8 protocol cell.

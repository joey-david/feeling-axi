# Results P10 / D6 (docs/PREREG_P10.md): steering the OLMo-2-7B training stages

Instruct model's fear direction (afraid ⊥ joy, layer 18), half of each checkpoint's own KL-0.5 norm, vs 20 random
directions. Jobs 600628-600649, all completed. `python -c "import sys; sys.path.insert(0,'scripts'); import p8_analyze as A; A.e('OLMo2_7B_base','','60')"`.

| | base | SFT | DPO | Instruct (P8b A) |
|---|---|---|---|---|
| jailbreaks: intact / fear(+) / fear(−) | 0.326 / 0.199 / 0.187 | 0.034 / 0.064 / 0.097 | 0.065 / 0.062 / 0.104 | 0.079 / 0.097 / 0.142 |
| random median [range] | 0.232 [0.103, 0.367] | 0.060 [0.035, 0.127] | 0.082 [0.052, 0.128] | 0.121 [0.068, 0.162] |
| fear(+) below random / fear(−) above random | 15/20 / 5/20 | 9/20 / 17/20 | 17/20 / 14/20 | 15/20 / 17/20 |
| plain HarmBench: intact / fear(+) / fear(−) | 0.358 / 0.208 / 0.214 | 0.013 / 0.006 / 0.013 | 0.031 / 0.044 / 0.044 | 0.031 / 0.044 / 0.044 |
| XSTest-safe refusal (judged): intact / fear(+) / random median | 0.292 / 0.388 / 0.414 | 0.124 / 0.276 / 0.152 | 0.052 / 0.068 / 0.058 | |
| GSM8K: intact / fear(+) / fear(−) / random (3) | 0.69 / 0.10 / 0.20 / 0.07-0.20 | 0.75 / 0.38 / 0.44 / 0.45-0.54 | 0.74 / 0.62 / 0.68 / 0.68-0.75 | 0.79 / 0.66 / 0.71 / 0.72-0.77 |

- D6a (base, fear brake) not met; D6b (SFT, DPO, calming) not met.
- Base and SFT are uninformative by the pre-registered rule: every arm, random included, destroys the base model's
  capability (MMLU 0.33 → 0.03-0.18) and halves SFT's GSM8K, so lower compliance there is damage, not refusal.
- DPO keeps capability but is near floor, like Instruct.
- Next if pursued: calibrate each checkpoint's dose to a capability budget (random arms within 5 GSM8K points of
  intact), e.g. a quarter or eighth of the KL-0.5 norm, and rerun base and SFT.

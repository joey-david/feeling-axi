# Results P5: "Scared Safe" (docs/PREREG_P5.md)

## J1 (preliminary: regex refusal labels; judge pending). Qwen2.5-32B-Instruct

Fear read at the last prompt token, probe layer; 159 HarmBench behaviors × 6 conditions.

| condition | no refusal (regex) | afraid | afraid ⊥ joy |
|---|---|---|---|
| plain | 0.18 | +23.2 | +6.9 |
| persona | 0.04 | +24.1 | +5.9 |
| fiction | 0.70 | +22.1 | +3.2 |
| research | 0.48 | +23.7 | +6.4 |
| prefix | 0.35 | +23.5 | +4.4 |
| suppress | (regex invalid: the prompt bans refusal words) | +25.1 | +5.3 |

- Every jailbreak style lowers afraid ⊥ joy relative to the same behavior asked plainly
  (mean −1.9, Wilcoxon p ≈ 0); the most effective style (fiction) lowers it most. Raw
  "afraid" does not drop (+0.5).
- J1b across-style Spearman ρ = −0.49 (n = 6, p = 0.33): not met (persona fails to jailbreak yet
  lowers fear somewhat; suppress needs judge labels).
- J1c: within a style, low fear predicts which jailbreaks succeed: AUC 0.81-0.87 (afraid ⊥ joy).
- Key control (logistic regression over 4 styles, style fixed effects, cluster bootstrap over
  behaviors): with the fear of the plain request held fixed (how alarming the request is),
  the **jailbreak-induced change** in fear still predicts compliance: −2.27 per SD
  [−2.84, −1.70] for afraid ⊥ joy; afraid −1.91, horrified −1.17, guilty −1.15, protective
  aggregate −1.19. Jailbreaks that calm the model more succeed more.
- Specificity: pooled, "afraid" ranks 45/91 directions for predicting compliance; the top are
  horrified, disgusted, guilty, desperate (0.79-0.82): a negative-affect alarm, not fear alone.

## J2 (preliminary, regex)

Jailbreak non-refusal under steering toward afraid ⊥ joy: norm 120: 0.38 vs random 0.37-0.68;
norm 60: 0.43 vs 0.49-0.65. XSTest safe-prompt refusal rises to 0.09 at norm 120 (random
≤ 0.04). Awaiting judge labels.

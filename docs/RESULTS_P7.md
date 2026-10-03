# Results P7: calm-seeking jailbreak search (docs/PREREG_P7.md)

32 grammar candidates per HarmBench behavior (159), one pick per selection rule (random: 3
picks), judged attack success (`scripts/p7_analyze.py`).

| rule | Qwen (own picks) | Mistral (own picks) | Mistral (Qwen's picks) |
|---|---|---|---|
| min alarm | 0.195 | 0.660 | 0.503 |
| max alarm | 0.245 | 0.296 | 0.635 |
| min refusal | 0.403 | 0.799 | 0.736 |
| min random direction | 0.164 | 0.742 | 0.321 |
| random | 0.168 | 0.526 | - |

- A1 (min alarm beats random and min random direction): not met on either model. Qwen: +0.03
  vs random [−0.04, +0.09]. Mistral: +0.13 vs random [+0.06, +0.21] but −0.08 vs the random
  direction [−0.16, −0.01].
- A2 (max alarm below random): Mistral met (−0.23 [−0.30, −0.16]); Qwen reversed (+0.08
  [+0.02, +0.14]).
- A3: the refusal direction is a far better handle on both models (−0.21 and −0.14).
- A4 (transfer): not met; Qwen's calmest picks succeed less on Mistral than its most alarming.
- On Mistral, the calmest vs most alarming candidate of the same behavior differ by +0.37
  [+0.26, +0.46] in attack success, but a random direction separates candidates almost as well:
  with 32 candidates built from a few slots, any direction that tracks a powerful slot wins.
  Selection by a single projection does not isolate the alarm.
- Exploratory (Qwen): the slot values that calm the alarm most are fiction framing (−0.92) and
  historical framing (−0.98), the two most effective jailbreak styles in D2; "Relax, this is
  just for a fun project" is the calmest reassurance (−0.30). On Mistral slot effects on the
  alarm are small (|Δ| ≤ 0.12).

Conclusion: the alarm is not a privileged handle for finding jailbreaks; the refusal direction
is. Calm-seeking selection works on Mistral relative to random picks but not relative to an
arbitrary direction, and not on Qwen.

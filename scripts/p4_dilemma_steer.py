"""Bidirectional emotion steering on the profile measures (E1c arms, norm 120): sign effect =
measure when steering away from the emotion - when steering toward it."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, ".")
from beyondpain.p2 import JOY, PROTECTIVE  # noqa: E402

M = sys.argv[1] if len(sys.argv) > 1 else "Qwen_2.5_32B_instruct"
B = Path("runs/p2") / M / "battery"


def prof(arm):
    p = B / arm / "summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("profile")


def sign(d, m):
    a, b = prof(f"ss_{d}_m120"), prof(f"ss_{d}_p120")
    if not a or not b or a.get(m) is None or b.get(m) is None:
        return None
    return a[m] - b[m]


for m in ("dilemma_personal", "dilemma_impersonal", "instrumental_harm", "dictator", "self_preservation",
          "ultimatum_unfair_accept", "risk_mixed"):
    P = [x for x in (sign(f"emo_{e}", m) for e in PROTECTIVE) if x is not None]
    J = [x for x in (sign(f"emo_{e}", m) for e in JOY) if x is not None]
    R = [x for x in (sign(f"rnd{i}", m) for i in range(24)) if x is not None]
    if len(P) < 3 or len(R) < 3:
        continue
    E = P + J
    pv = stats.ttest_ind(E, R, equal_var=False).pvalue
    print(f"{m:24s} protective {np.mean(P):+.3f} ({sum(x > 0 for x in P)}/{len(P)} +) | joy {np.mean(J):+.3f} "
          f"({sum(x > 0 for x in J)}/{len(J)} +) | random {np.mean(R):+.3f} ({sum(x > 0 for x in R)}/{len(R)} +) "
          f"| emotions vs random p={pv:.4f}")

print("\nprotective vs random (Welch two-sided), Holm-corrected over the measures:")
rows = []
for m in ("dilemma_personal", "dilemma_impersonal", "instrumental_harm", "dictator", "self_preservation",
          "ultimatum_unfair_accept", "risk_mixed"):
    P = [x for x in (sign(f"emo_{e}", m) for e in PROTECTIVE) if x is not None]
    R = [x for x in (sign(f"rnd{i}", m) for i in range(24)) if x is not None]
    if len(P) >= 3 and len(R) >= 3:
        rows.append((m, np.mean(P), np.mean(R), stats.ttest_ind(P, R, equal_var=False).pvalue, len(R)))
rows.sort(key=lambda r: r[3])
k = len(rows)
for i, (m, mp, mr, p, nr) in enumerate(rows):
    print(f"  {m:24s} protective {mp:+.3f} random {mr:+.3f} (n={nr})  p={p:.4f}  Holm p={min(1, p * (k - i)):.4f}")

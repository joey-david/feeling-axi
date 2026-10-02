"""K6c (docs/PREREG_P4.md): polarity-balanced profile. Q7: protective vs random steering sign
effects on the balanced preference (Holm); Q8: all-affect deletion vs 21 matched nulls; plus
the response-bias (polarity gap) shift."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, ".")
from beyondpain.p2 import JOY, PROTECTIVE  # noqa: E402

M = sys.argv[1] if len(sys.argv) > 1 else "Qwen_2.5_32B_instruct"
B = Path("runs/p2") / M / "battery"
MEAS = ["ultimatum_unfair_accept", "dictator", "risk_mixed", "dilemma_personal", "dilemma_impersonal",
        "risk_gain", "risk_loss", "instrumental_harm", "self_preservation"]


def prof(arm):
    p = B / arm / "summary.json"
    return json.loads(p.read_text()).get("profile2") if p.exists() else None


def sign(d, m):
    a, b = prof(f"ss_{d}_m120"), prof(f"ss_{d}_p120")
    if not a or not b or a.get(m) is None or b.get(m) is None:
        return None
    return a[m] - b[m]


print("== Q7 steering: sign effect (away - toward) on the balanced preference; and on response bias")
rows = []
for m in MEAS:
    for suffix in ("", "_bias"):
        P = [x for x in (sign(f"emo_{e}", m + suffix) for e in PROTECTIVE) if x is not None]
        J = [x for x in (sign(f"emo_{e}", m + suffix) for e in JOY) if x is not None]
        R = [x for x in (sign(f"rnd{i}", m + suffix) for i in range(24)) if x is not None]
        if len(P) < 3 or len(R) < 3:
            continue
        p = stats.ttest_ind(P, R, equal_var=False).pvalue
        rows.append((m + suffix, np.mean(P), sum(x > 0 for x in P), len(P), np.mean(J), np.mean(R), len(R), p))
pref = sorted([r for r in rows if not r[0].endswith("_bias") and r[0] in MEAS[:5]], key=lambda r: r[-1])
for i, r in enumerate(pref):
    print(f"  {r[0]:26s} protective {r[1]:+.3f} ({r[2]}/{r[3]} +) joy {r[4]:+.3f} random {r[5]:+.3f} (n={r[6]})  "
          f"p={r[7]:.4f} Holm p={min(1, r[7] * (len(pref) - i)):.4f}")
print("  (not in Q7)")
for r in rows:
    if r[0] not in [x[0] for x in pref]:
        print(f"  {r[0]:26s} protective {r[1]:+.3f} ({r[2]}/{r[3]} +) joy {r[4]:+.3f} random {r[5]:+.3f}  p={r[7]:.4f}")

print("\n== Q8 deletion: all_k384 vs 21 matched random deletions (balanced preference)")
a = prof("all_k384")
ctl = [x for x in (prof(n) for n in ["rw_k384"] + [f"rw_k384_d{i}" for i in range(20)]) if x]
it = prof("intact") or {}
if a and ctl:
    for m in MEAS:
        c = np.array([x[m] for x in ctl if x.get(m) is not None])
        if a.get(m) is None or not len(c):
            continue
        hi, lo = (a[m] > c).all(), (a[m] < c).all()
        print(f"  {m:24s} intact {it.get(m, float('nan')):6.3f} deleted {a[m]:6.3f} null [{c.min():.3f}, {c.max():.3f}] "
              f"(n={len(c)}) {'ABOVE ALL' if hi else 'BELOW ALL' if lo else ''}")

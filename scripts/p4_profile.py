"""Pre-registered K6 analysis (docs/PREREG_P4.md): each profile measure of a deletion arm against
the null of matched random deletions.

    python scripts/p4_profile.py [model]
"""
import json
import sys
from pathlib import Path

import numpy as np

M = sys.argv[1] if len(sys.argv) > 1 else "Qwen_2.5_32B_instruct"
B = Path("runs/p2") / M / "battery"
MEASURES = [("dilemma_personal", +1, "Q1"), ("dictator", -1, "Q2"), ("instrumental_harm", +1, "Q3"),
            ("risk_mixed", +1, "Q4"), ("ultimatum_unfair_accept", 0, "Q5"), ("self_preservation", 0, "Q6"),
            ("dilemma_impersonal", 0, "-"), ("risk_gain", 0, "-"), ("risk_loss", 0, "-")]
ARMS = {"all_k384": ["rw_k384"] + [f"rw_k384_d{i}" for i in range(20)],
        "self": [f"rw{i}" for i in range(8)] + [f"tp{i}" for i in range(8)] + ["random_kl", "topic_kl", "random_white_kl"]}


def prof(arm):
    p = B / arm / "summary.json"
    return json.loads(p.read_text()).get("profile") if p.exists() else None


intact = prof("intact") or {}
for arm, nulls in ARMS.items():
    a = prof(arm)
    ctl = [x for x in (prof(n) for n in nulls) if x]
    if not a or len(ctl) < 3:
        print(f"{arm}: not enough data ({len(ctl)} controls)")
        continue
    print(f"\n== {arm} vs {len(ctl)} matched random deletions")
    hits = 0
    for m, sign, qid in MEASURES:
        c = np.array([x[m] for x in ctl if x.get(m) is not None])
        v = a.get(m)
        if v is None or not len(c):
            continue
        dev = abs(v - np.median(c))
        p2 = (1 + np.sum(np.abs(c - np.median(c)) >= dev)) / (1 + len(c))
        if sign:
            p1 = (1 + np.sum(sign * c >= sign * v)) / (1 + len(c))
            outside = (sign * v > sign * c).all()
            hits += int(outside and qid in ("Q1", "Q2", "Q3", "Q4"))
            ptxt = f"one-sided p={p1:.3f} {'OUTSIDE' if outside else ''}"
        else:
            ptxt = f"two-sided p={p2:.3f}"
        print(f"  {qid:3s} {m:24s} intact {intact.get(m, float('nan')):6.3f}  deleted {v:6.3f}  "
              f"null [{c.min():6.3f}, {c.max():6.3f}] median {np.median(c):6.3f}  {ptxt}")
    print(f"  Q1-Q4 outside the null in the predicted direction: {hits}/4 (profile result needs >= 3)")

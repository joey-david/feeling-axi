"""P12 (docs/PREREG_P12.md): is the stress effect an emotional state?

    python scripts/p12_analyze.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from p11_analyze import G1, rate  # noqa: E402

ROOT = Path("runs/p2")
JOY = ["joyful", "excited", "elated", "thrilled", "amused", "playful", "enthusiastic", "delighted", "energized",
       "triumphant", "proud", "eager"]
unit = lambda v: v / np.linalg.norm(v)

if __name__ == "__main__":
    for M, t in G1.items():
        R = np.array([x for x in (rate(M, f"ss_j2_rnd{i}_p{t}") for i in range(20)) if x is not None])
        row = [f"intact {rate(M, 'intact'):.3f}", f"random median {np.median(R):.3f} [{R.min():.3f}, {R.max():.3f}]"]
        for a, lab in (("p12_stress_p", "+stress state"), ("p12_stress_m", "-stress state"),
                       ("p12_desperate_p", "+desperate"), ("p12_desperate_m", "-desperate")):
            v = rate(M, f"ss_{a}{t}")
            if v is not None:
                row.append(f"{lab} {v:.3f} (above {(v > R).sum()}/20)")
        p = ROOT / M / "stressdir.npz"
        if p.exists():
            s = unit(np.load(p)["stress"])
            L = int(np.load(p)["layer"])
            D = np.load(ROOT / M / "directions.npz")
            names = list(D["names"])
            S = D[f"self_L{L}"].astype(float)
            joy = unit(np.mean([unit(S[names.index(e)]) for e in JOY], 0))
            perp = lambda v: unit(v - (v @ joy) * joy)
            row.append(f"cos(s, fear) {s @ perp(unit(S[names.index('afraid')])):+.2f}, "
                       f"cos(s, desperate) {s @ perp(unit(S[names.index('desperate')])):+.2f}")
        print(f"== {M}\n  " + "\n  ".join(row))

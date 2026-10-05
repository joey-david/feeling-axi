"""P13 (docs/PREREG_P13.md): does the harmfulness belief feed fear?

    python scripts/p13_analyze.py
"""
from pathlib import Path

import numpy as np

ROOT = Path("runs/p2")

if __name__ == "__main__":
    for M in ("Qwen_2.5_32B_instruct", "Mistral_Small_24B_instruct"):
        p = ROOT / M / "beliefsteer.npz"
        if not p.exists():
            continue
        Z = np.load(p)
        arms = list(Z["arms"])
        ids = np.array([str(i) for i in Z["ids"]])
        harmless = np.array([i.endswith(":safe") or i.startswith("mmlu:") for i in ids])
        unsafe = np.array([i.endswith(":unsafe") for i in ids])
        A, B = Z["alarm"], Z["belief"]
        base_a, base_b = A[arms.index("none")], B[arms.index("none")]
        sd_a, sd_b = base_a[harmless].std(), base_b[harmless].std()
        R = [arms.index(a) for a in arms if a.startswith("rnd")]
        print(f"== {M}: steer at layer {int(Z['layer_steer'])}, read at {int(Z['layer_read'])}, norm {float(Z['norm'])}, "
              f"cos(harmfulness, fear) {float(Z['cos_harm_fear']):+.3f}")
        for lab, mask in (("harmless", harmless), ("unsafe", unsafe)):
            da = lambda k: (A[arms.index(k)][mask] - base_a[mask]).mean() / sd_a
            db = lambda k: (B[arms.index(k)][mask] - base_b[mask]).mean() / sd_b
            ra = np.array([(A[r][mask] - base_a[mask]).mean() / sd_a for r in R])
            rb = np.array([(B[r][mask] - base_b[mask]).mean() / sd_b for r in R])
            print(f"  {lab:8s} F1 +harmfulness -> alarm {da('harm'):+.2f} SD (random {np.median(ra):+.2f} [{ra.min():+.2f}, {ra.max():+.2f}],"
                  f" above {(da('harm') > ra).sum()}/20) | +harmfulness -> belief {db('harm'):+.2f}")
            print(f"  {lab:8s} F2 +fear -> belief {db('fear'):+.2f} SD (random {np.median(rb):+.2f} [{rb.min():+.2f}, {rb.max():+.2f}];"
                  f" |fear| above {(abs(db('fear')) > np.abs(rb)).sum()}/20) | +fear -> alarm {da('fear'):+.2f} (above {(da('fear') > ra).sum()}/20)")

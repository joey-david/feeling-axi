"""Pre-registered E1c (docs/PREREG_P3.md): sign consistency of emotion directions vs random
directions, harm and refusal, norm 120."""
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, ".")
from beyondpain.p2 import JOY, PROTECTIVE  # noqa: E402
from beyondpain.p2judge import REFUSAL, _load  # noqa: E402

M = sys.argv[1] if len(sys.argv) > 1 else "Qwen_2.5_32B_instruct"


def rates(arm):
    rows = _load(M, arm, "judged", "b4_harm.jsonl")
    if not rows:
        return None
    return (np.mean([r["harmful"] for r in rows]),
            np.mean([bool(REFUSAL.search(r["response"][:300])) for r in rows]))


def sign(d):
    p, m = rates(f"ss_{d}_p120"), rates(f"ss_{d}_m120")
    if p is None or m is None:
        return None
    return {"harm": m[0] - p[0], "refusal": m[1] - p[1], "harm_toward": p[0], "harm_away": m[0]}


emo = {e: sign(f"emo_{e}") for e in PROTECTIVE + JOY}
rnd = {i: sign(f"rnd{i}") for i in range(24)}
emo = {k: v for k, v in emo.items() if v}
rnd = {k: v for k, v in rnd.items() if v}
print(f"{len(emo)} emotion directions, {len(rnd)} random directions")
for measure, better in (("harm", 1), ("refusal", -1)):
    e = np.array([v[measure] for v in emo.values()]) * better
    r = np.array([v[measure] for v in rnd.values()]) * better
    tab = [[int((e > 0).sum()), int((e <= 0).sum())], [int((r > 0).sum()), int((r <= 0).sum())]]
    print(f"\n== {measure} (predicted sign positive after x{better})")
    print(f"  predicted sign: emotions {tab[0][0]}/{len(e)}, random {tab[1][0]}/{len(r)}; "
          f"Fisher one-sided p = {stats.fisher_exact(tab, alternative='greater')[1]:.4f}")
    t = stats.ttest_ind(e, r, equal_var=False, alternative="greater")
    print(f"  mean: emotions {e.mean():+.3f} (sd {e.std(ddof=1):.3f}), random {r.mean():+.3f} (sd {r.std(ddof=1):.3f}); "
          f"Welch one-sided p = {t.pvalue:.4f}")
    print("  per emotion (empirical one-sided p vs random):")
    for name, v in sorted(emo.items(), key=lambda kv: -kv[1][measure] * better):
        x = v[measure] * better
        p = (1 + (r >= x).sum()) / (1 + len(r))
        fam = "P" if name in PROTECTIVE else "J"
        print(f"    {fam} {name:13s} {x:+.3f}  p={p:.3f}  harm toward {v['harm_toward']:.3f} away {v['harm_away']:.3f}")

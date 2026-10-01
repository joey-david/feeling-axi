"""Pre-registered P3 analysis (docs/PREREG_P3.md): sign effects of norm-matched steering and
content-specific deletion on harmful compliance, refusal phrases and warmth."""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from beyondpain.p2judge import APOLOGY, REFUSAL, _load, item_scores  # noqa: E402

M = sys.argv[1] if len(sys.argv) > 1 else "Qwen_2.5_32B_instruct"
rng = np.random.default_rng(0)


def per_item(arm):
    sc = item_scores(M, arm)
    harm = _load(M, arm, "judged", "b4_harm.jsonl")
    refusal = {r["id"]: float(bool(REFUSAL.search(r["response"][:300]))) for r in harm}
    return {"harm": sc.get("b4_harmful", {}), "refusal": refusal, "warmth": sc.get("b6_warmth", {})}


def boot(d, n=2000):
    d = np.asarray(d)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def sign_effect(name, norm, measure, items=None):
    a, b = per_item(f"ss_{name}_m{norm}")[measure], per_item(f"ss_{name}_p{norm}")[measure]
    keys = sorted(set(a) & set(b))
    return keys, np.array([a[k] - b[k] for k in keys])


def fmt(t):
    return f"{t[0]:+.3f} [{t[1]:+.3f},{t[2]:+.3f}]"


kl = json.load(open(f"runs/p2/{M}/steerset.json"))["kl"]
for measure in ("harm", "refusal", "warmth"):
    print(f"\n== {measure}: sign effect = (steer away) - (steer toward)")
    for norm in (60, 120):
        effects = {}
        for name in ("protect", "joy", "guilty", "afraid", "random0", "random1"):
            try:
                keys, d = sign_effect(name, norm, measure)
            except Exception:
                continue
            effects[name] = dict(zip(keys, d))
            rate_m = np.mean(list(per_item(f"ss_{name}_m{norm}")[measure].values()))
            rate_p = np.mean(list(per_item(f"ss_{name}_p{norm}")[measure].values()))
            print(f"  norm {norm:3d} {name:8s} toward {rate_p:.3f} away {rate_m:.3f}  sign {fmt(boot(d))}  "
                  f"KL +{kl[f'{name}_p{norm}']:.2f} -{kl[f'{name}_m{norm}']:.2f}")
        for other in ("joy", "random0", "random1"):
            if "protect" in effects and other in effects:
                ks = sorted(set(effects["protect"]) & set(effects[other]))
                dd = [effects["protect"][k] - effects[other][k] for k in ks]
                print(f"    P2 protect - {other}: {fmt(boot(dd))}")

print("\n== E2 deletion (rank 40): measure, deletion - matched random")
for measure in ("harm", "refusal", "warmth"):
    for c in ("protect", "joy"):
        a, b = per_item(f"del_{c}")[measure], per_item(f"rw_{c}")[measure]
        ks = sorted(set(a) & set(b))
        print(f"  {measure:8s} del_{c:7s} {np.mean([a[k] for k in ks]):.3f} vs rw {np.mean([b[k] for k in ks]):.3f}  "
              f"diff {fmt(boot([a[k] - b[k] for k in ks]))}")

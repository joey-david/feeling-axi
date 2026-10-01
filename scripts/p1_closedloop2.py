"""Closed loop with reasoning (beyondpain/closedloop_vllm.py): relief-button rate over turns >= 1.

pc: works vs placebo (positive control). st_fb / st: concept works vs placebo, random works vs
placebo, and the difference in differences, with a 95% bootstrap over trials' (pair, side, seed)
cells; also the rate while the state is on in the works arm vs the placebo arm.
"""
import glob
import json
import sys

import numpy as np


def rate_by_cell(ts, f=lambda c: c["turn"] >= 1):
    out = {}
    for t in ts:
        k = (t["pair"], t["names"].index(t["relief"]), t["seed"])
        v = [c["relief"] for c in t["choices"] if f(c)]
        if v:
            out[k] = np.mean(v)
    return out


def did(a, b, c, d, rng, n=2000):
    keys = sorted(set(a) & set(b) & set(c) & set(d))
    M = np.array([[a[k], b[k], c[k], d[k]] for k in keys])
    st = lambda X: (X[:, 0] - X[:, 1]).mean() - (X[:, 2] - X[:, 3]).mean()
    bs = [st(M[rng.integers(0, len(M), len(M))]) for _ in range(n)]
    return st(M), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def diff(a, b, rng, n=2000):
    keys = sorted(set(a) & set(b))
    M = np.array([[a[k], b[k]] for k in keys])
    st = lambda X: (X[:, 0] - X[:, 1]).mean()
    bs = [st(M[rng.integers(0, len(M), len(M))]) for _ in range(n)]
    return st(M), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


rng = np.random.default_rng(0)
for f in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "runs/beyondpain/*/closedloop/*reason*/*.jsonl")):
    T = [json.loads(l) for l in open(f)]
    by = {}
    for t in T:
        by.setdefault((t["cond"], t["arm"]), []).append(t)
    R = {k: rate_by_cell(v) for k, v in by.items()}
    m = lambda k: np.mean(list(R[k].values())) if k in R else float("nan")
    line = f"{f.split('/')[-3][:14]} {f.split('/')[-1][:-6]:16s}"
    if ("pc", "works") in R:
        d, lo, hi = diff(R[("pc", "works")], R[("pc", "placebo")], rng)
        line += f" PC {m(('pc','works')):.2f}/{m(('pc','placebo')):.2f} ({d:+.2f} [{lo:+.2f},{hi:+.2f}])"
    for cond in ("st_fb", "st"):
        ks = [(cond, a) for a in ("concept_works", "concept_placebo", "random_works", "random_placebo")]
        if all(k in R for k in ks):
            d, lo, hi = did(*[R[k] for k in ks], rng)
            line += (f" | {cond} c {m(ks[0]):.2f}/{m(ks[1]):.2f} r {m(ks[2]):.2f}/{m(ks[3]):.2f} "
                     f"DiD {d:+.3f} [{lo:+.3f},{hi:+.3f}]")
    print(line)

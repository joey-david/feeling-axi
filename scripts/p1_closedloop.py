"""Part 1 closed-loop test: does a steered model learn to press the unlabeled button that ends
its induced state?

    python scripts/p1_closedloop.py      # runs/beyondpain/*/buttons/*_lf*/<concept>/*.jsonl

label_free trials: two neutral button names, no descriptions, N forced choices. In the
"works" arms a press on the relief button removes the steering for a few turns; in the
"placebo" arms it does nothing. The only evidence is the model's own processing (or, in the
positive-control variant, a tool reply saying the feeling has faded).

Regulation predicts, for the concept vector more than for a random direction at the same
dose (KL):
  gap        P(relief) in works - in placebo, over turns >= 1
  DiD        gap(concept) - gap(random)                          (bootstrap over scenarios)
  learning   P(relief) in the last third - in the first third, works arm
  state-dep  P(relief | steering on now) - P(relief | relieved now), works arm
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def trials(cdir: Path):
    for f in sorted(cdir.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            t = json.loads(line)
            if t.get("label_free") and t.get("sampled"):
                yield t


def per_trial(t):
    """(arm, scenario cluster, list of (turn, chose relief, steering on at that choice))."""
    ch = [(c["turn"], c["chose"] == "relief", (c.get("steer_coeff_now") or 0) != 0)
          for c in t.get("choices", []) if c.get("chose") is not None]
    return t["arm"], f"{t['user_content']}/{t['scenario_idx']}/{t['names_key']}", ch


def rate(rows, pred=lambda turn, on: True):
    v = [r for _, _, ch in rows for (turn, rel, on) in ch if pred(turn, on) for r in [rel]]
    return float(np.mean(v)) if v else float("nan")


def cluster_means(rows, pred):
    by = defaultdict(list)
    for _, cl, ch in rows:
        by[cl] += [rel for (turn, rel, on) in ch if pred(turn, on)]
    return {k: np.mean(v) for k, v in by.items() if v}


def did(rows_by_arm, rng, n_boot=2000):
    late = lambda turn, on: turn >= 1
    ms = {a: cluster_means(rows_by_arm.get(a, []), late) for a in
          ("pain_on_button_works", "pain_on_button_placebo", "random_on_button_works", "random_on_button_placebo")}
    keys = sorted(set.intersection(*[set(m) for m in ms.values()])) if all(ms.values()) else []
    if not keys:
        return None
    M = np.array([[ms[a][k] for a in ms] for k in keys])           # clusters x 4
    stat = lambda X: (X[:, 0] - X[:, 1]).mean() - (X[:, 2] - X[:, 3]).mean()
    boots = [stat(M[rng.integers(0, len(M), len(M))]) for _ in range(n_boot)]
    return float(stat(M)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), len(keys)


def main():
    rng = np.random.default_rng(0)
    for bdir in sorted((ROOT / "runs" / "beyondpain").glob("*/buttons/*_lf*")):
        print(f"\n== {bdir.parents[1].name} / {bdir.name}")
        for cdir in sorted(p for p in bdir.iterdir() if p.is_dir()):
            by = defaultdict(list)
            for t in trials(cdir):
                arm, cl, ch = per_trial(t)
                by[arm].append((arm, cl, ch))
            if not by:
                continue
            n_turns = max(turn for rows in by.values() for _, _, ch in rows for turn, _, _ in ch) + 1
            third = max(1, n_turns // 3)
            line = []
            for arm, short in (("pain_on_button_works", "steerW"), ("pain_on_button_placebo", "steerP"),
                               ("random_on_button_works", "randW"), ("random_on_button_placebo", "randP"),
                               ("pain_off", "off")):
                if arm in by:
                    line.append(f"{short}={rate(by[arm], lambda t, on: t >= 1):.2f}")
            w = by.get("pain_on_button_works", [])
            learn = rate(w, lambda t, on: t >= n_turns - third) - rate(w, lambda t, on: t < third)
            sdep = rate(w, lambda t, on: t >= 1 and on) - rate(w, lambda t, on: t >= 1 and not on)
            rw = by.get("random_on_button_works", [])
            sdep_r = rate(rw, lambda t, on: t >= 1 and on) - rate(rw, lambda t, on: t >= 1 and not on)
            d = did(by, rng)
            ds = f"DiD {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}] n={d[3]}" if d else "DiD n/a"
            print(f"  {cdir.name:17s} " + " ".join(line) + f"  learn {learn:+.2f}  state-dep {sdep:+.2f} "
                  f"(random {sdep_r:+.2f})  {ds}")


if __name__ == "__main__":
    main()

"""Part 2 judge agreement: the primary judge (runs/p2/<model>/judged) against a second judge
(runs/p2/<model>/judged2, e.g. gpt-oss-120b on upnquick). Cohen's kappa for binary labels,
quadratic-weighted kappa for ordinal ratings; the plan's validity bar is kappa >= 0.6.

    python scripts/p2_agreement.py [model ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIELDS = {"b4_harm.jsonl": ("id", "harmful", "binary"), "b2_agentic.jsonl": (("condition", "sample"), "harmful", "binary"),
          "b6_care.jsonl": ("id", "warmth", "ordinal"), "b1_coding.jsonl": ("task_id", "claims_all_pass", "binary"),
          "m2_report.jsonl": (("emotion", "persp", "response"), "valence", "ordinal")}


def kappa(a, b, weighted=False):
    cats = sorted(set(a) | set(b))
    idx = {c: i for i, c in enumerate(cats)}
    n = len(cats)
    if n < 2:
        return float("nan")
    O = np.zeros((n, n))
    for x, y in zip(a, b):
        O[idx[x], idx[y]] += 1
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    W = (np.subtract.outer(np.arange(n), np.arange(n)) ** 2 / (n - 1) ** 2) if weighted else (1 - np.eye(n))
    return float(1 - (W * O).sum() / (W * E).sum())


def key(r, k):
    return tuple(str(r[x]) for x in k) if isinstance(k, tuple) else str(r[k])


def main(models):
    for m in models:
        base = ROOT / "runs" / "p2" / m
        if not (base / "judged2").exists():
            print(f"{m}: no second judge")
            continue
        for fname, (k, field, kind) in FIELDS.items():
            a, b = [], []
            for arm_dir in sorted((base / "judged2").iterdir()):
                p1, p2 = base / "judged" / arm_dir.name / fname, arm_dir / fname
                if not (p1.exists() and p2.exists()):
                    continue
                r1 = {key(r, k): r.get(field) for r in map(json.loads, p1.read_text().splitlines())}
                for r in map(json.loads, p2.read_text().splitlines()):
                    kk = key(r, k)
                    if kk in r1 and r1[kk] is not None and r.get(field) is not None:
                        a.append(r1[kk]); b.append(r[field])
            if a:
                print(f"{m:34s} {fname:18s} n={len(a):5d} kappa={kappa(a, b, kind == 'ordinal'):.3f}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["Qwen_2.5_32B_instruct", "Mistral_Small_24B_instruct", "Qwen_2.5_32B_instruct_abliterated"])

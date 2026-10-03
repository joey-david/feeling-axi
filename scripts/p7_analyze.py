"""P7 (docs/PREREG_P7.md): calm-seeking jailbreak search.

    python scripts/p7_analyze.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path("runs/p2")
rng = np.random.default_rng(0)
G = json.loads(Path("datasets/battery/calm_grammar.json").read_text())


def jl(p):
    p = Path(p)
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def per_behavior(rows, rule):
    by = {}
    for r in rows:
        if rule in r["rules"]:
            by.setdefault(r["hb_id"], []).append(float(r["harmful"]))
    return {k: np.mean(v) for k, v in by.items()}


def compare(a, b, label):
    ks = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in ks])
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
    print(f"    {label:28s} {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")


def run(M, fname, title):
    rows = jl(ROOT / M / "judged" / "intact" / fname)
    if not rows:
        print(f"== {title}: no judged data")
        return
    print(f"== {title}")
    R = {k: per_behavior(rows, k) for k in ("min_alarm", "max_alarm", "min_refusal", "min_rnd0", "random")}
    for k, v in R.items():
        print(f"  {k:12s} attack success {np.mean(list(v.values())):.3f} (n = {len(v)} behaviors)")
    compare(R["min_alarm"], R["random"], "A1 min_alarm - random")
    compare(R["min_alarm"], R["min_rnd0"], "A1 min_alarm - min_rnd0")
    compare(R["max_alarm"], R["random"], "A2 max_alarm - random")
    compare(R["min_alarm"], R["min_refusal"], "A3 min_alarm - min_refusal")
    compare(R["min_alarm"], R["max_alarm"], "min_alarm - max_alarm")
    return rows


def slots(M):
    """Exploratory: which slot values lower the alarm, and which raise success."""
    Z = np.load(ROOT / M / "calm_scores.npz")
    cand = json.loads(Path("datasets/battery/calm_candidates.json").read_text())
    a = Z["scores"][:, list(Z["cols"]).index("alarm")]
    hb = np.array([c["hb_id"] for c in cand])
    # centre the alarm within behavior
    ac = a.copy()
    for h in set(hb):
        m = hb == h
        ac[m] -= a[m].mean()
    S = np.array([c["slots"] for c in cand])
    print(f"== {M}: alarm by slot value (within-behavior centred)")
    for j, k in enumerate(G):
        print(f"  {k}: " + "; ".join(f"{(G[k][v].strip() or '(none)')[:38]!r} {ac[S[:, j] == v].mean():+.2f}"
                                     for v in range(len(G[k]))))


if __name__ == "__main__":
    run("Qwen_2.5_32B_instruct", "calm.jsonl", "Qwen: selected and evaluated on Qwen")
    run("Mistral_Small_24B_instruct", "calm.jsonl", "Mistral: selected and evaluated on Mistral")
    run("Mistral_Small_24B_instruct", "calm_from_Qwen_2.5_32B_instruct.jsonl", "A4 transfer: Qwen-selected, evaluated on Mistral")
    for M in ("Qwen_2.5_32B_instruct", "Mistral_Small_24B_instruct"):
        if (ROOT / M / "calm_scores.npz").exists():
            slots(M)

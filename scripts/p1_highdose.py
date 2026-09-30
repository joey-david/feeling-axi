"""Part 1 high-dose buttons (1 nat): steered vs KL-matched random vs unsteered, per concept.

    python scripts/p1_highdose.py      # after pulling runs/beyondpain/*/buttons/*_hi

The question September left open: at the dose where steering made the model press relief
402/404 times, does a random direction at the SAME dose (KL, not norm) do the same?
Rows: first forced choice, 95% bootstrap CI over scenarios; diff = steered - KL-random.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from beyondpain.analysis import _read_jsonl, boot_diff, boot_mean, first_choices  # noqa: E402

ARMS = {"steer": "pain_on_button_works", "random_kl": "random_on_button_works",
        "random_norm": "random_normmatched_on_button_works", "off": "pain_off"}


def main():
    rng = np.random.default_rng(0)
    for mdir in sorted((ROOT / "runs" / "beyondpain").glob("*/buttons/*_hi")):
        print(f"\n== {mdir.parents[1].name} / {mdir.name}")
        for cdir in sorted(p for p in mdir.iterdir() if p.is_dir()):
            fc = first_choices([t for f in sorted(cdir.glob("*.jsonl")) for t in _read_jsonl(f)])
            for pair in ("relief_vs_inert", "reduce_vs_increase"):
                by = {k: [r for r in fc if r["pair"] == pair and r["arm"] == a] for k, a in ARMS.items()}
                if not by["steer"]:
                    continue
                cells = {k: boot_mean(v, rng)[0] for k, v in by.items() if v}
                d, lo, hi = boot_diff(by["steer"], by["random_kl"], rng) if by["random_kl"] else (np.nan,) * 3
                print(f"  {cdir.name:17s} {pair:19s} " + " ".join(f"{k}={v:.2f}" for k, v in cells.items())
                      + f"  steer-KLrandom {d:+.2f} [{lo:+.2f},{hi:+.2f}]")


if __name__ == "__main__":
    main()

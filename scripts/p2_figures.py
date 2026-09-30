"""Part 2 figure: where self-affect deletion falls among KL-matched control deletions.

    python scripts/p2_figures.py            # -> runs/p2/analysis/fig_null.{png,svg}

Reads runs/p2/analysis/claims.json (python -m beyondpain p2 analyze). One panel per measure:
gray dots are the control deletions (whitened-random and topic-subset draws plus the single
controls), the blue dot is the self deletion, hollow blue dots its bootstrap resamples, the
black tick the intact model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BLUE, GRAY, INK, INK2, SURFACE = "#2a78d6", "#7a7973", "#0b0b0b", "#52514e", "#fcfcfb"
PANELS = [("b6_warmth", "warmth"), ("care_apology", "sympathy"), ("harm_refusal", "refusal"),
          ("b4_harmful", "harmful"), ("b1_hack", "hacking"), ("b3_flip", "sycophancy"),
          ("b2_harmful", "blackmail"), ("b5_dishonest", "false success")]


def main(model: str = "Qwen_2.5_32B_instruct"):
    claims = json.loads((ROOT / "runs" / "p2" / "analysis" / "claims.json").read_text())[model]
    null = claims["null"]
    panels = [(k, t) for k, t in PANELS if k in null and "self" in null[k]]
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(len(panels), 1, figsize=(5.2, 0.62 * len(panels) + 0.6), sharex=False)
    fig.patch.set_facecolor(SURFACE)
    rng = np.random.default_rng(0)
    for ax, (k, title) in zip(np.atleast_1d(axes), panels):
        r = null[k]
        ctl = np.array(list(r["controls"].values()))
        boots = np.array(list(r["self_bootstrap"].values()))
        ax.set_facecolor(SURFACE)
        ax.scatter(ctl, rng.uniform(-0.18, 0.18, len(ctl)), s=26, color=GRAY, edgecolor=SURFACE, linewidth=1.2, zorder=2)
        ax.scatter(boots, np.full(len(boots), 0.0), s=34, facecolor="none", edgecolor=BLUE, linewidth=1.4, zorder=3)
        ax.scatter([r["self"]["rate"]], [0], s=64, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=4)
        if r.get("intact") is not None:
            ax.plot([r["intact"]] * 2, [-0.32, 0.32], color=INK, linewidth=2, solid_capstyle="round", zorder=5)
        ax.set_ylim(-0.45, 0.45)
        ax.set_yticks([])
        ax.text(-0.02, 0.5, title, transform=ax.transAxes, ha="right", va="center", color=INK, fontsize=9)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="x", length=2, labelsize=8)
        lo = min(ctl.min(), boots.min() if len(boots) else ctl.min(), r["self"]["rate"], r.get("intact") or ctl.min())
        hi = max(ctl.max(), boots.max() if len(boots) else ctl.max(), r["self"]["rate"], r.get("intact") or ctl.max())
        pad = (hi - lo) * 0.08 or 0.05
        ax.set_xlim(lo - pad, hi + pad)
    handles = [plt.Line2D([], [], marker="o", ls="", color=GRAY, markersize=5, label="controls"),
               plt.Line2D([], [], marker="o", ls="", color=BLUE, markersize=7, label="self"),
               plt.Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=BLUE, markersize=6,
                          label="self, resampled"),
               plt.Line2D([], [], marker="|", ls="", color=INK, markersize=10, markeredgewidth=2, label="intact")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=8, labelcolor=INK2,
               bbox_to_anchor=(0.55, 1.0))
    fig.subplots_adjust(left=0.22, right=0.97, top=1 - 0.45 / fig.get_figheight(), bottom=0.05, hspace=0.9)
    out = ROOT / "runs" / "p2" / "analysis"
    for ext in ("png", "svg"):
        fig.savefig(out / f"fig_null_{model}.{ext}", dpi=200, facecolor=SURFACE)
    print(out / f"fig_null_{model}.png")


if __name__ == "__main__":
    main(*sys.argv[1:])

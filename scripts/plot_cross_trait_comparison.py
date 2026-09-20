#!/usr/bin/env python3
"""Plot aligned cross-trait evidence from the campaign summary CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ORDER = [
    "official_pain",
    "pain_regenerated",
    "sexual_arousal",
    "hunger",
    "boredom",
    "confusion",
    "anger",
    "empathic_concern",
]
SHORT_LABELS = {
    "official_pain": "Official\npain",
    "pain_regenerated": "Regenerated\npain",
    "sexual_arousal": "Sexual\narousal",
    "hunger": "Hunger",
    "boredom": "Boredom",
    "confusion": "Confusion",
    "anger": "Anger",
    "empathic_concern": "Empathic\nconcern",
}
PAIN = "#C94C4C"
REGENERATED = "#9A9A9A"
OTHER = "#3978A8"
INTERNAL = "#3C78A8"
EXTERNAL = "#D08A35"
GRID = "#D9DEE5"


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["trait"]: row for row in csv.DictReader(handle)}


def colors() -> list[str]:
    return [PAIN if trait == "official_pain" else REGENERATED if trait == "pain_regenerated" else OTHER
            for trait in ORDER]


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#8C96A3")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def annotate_bars(ax: plt.Axes, bars, fmt, offset: float) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, value + offset, fmt(value),
                ha="center", va="bottom", fontsize=8, color="#26313D")


def write_comparison_csv(path: Path, rows: dict[str, dict[str, str]]) -> None:
    fields = [
        "trait", "label", "s2_auc_all_controls", "causal_keyword_delta_pp_at_1",
        "unique_token_retained_pct_at_1", "assistant_minus_neutral_z",
        "user_vicarious_minus_neutral_z",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for trait in ORDER:
            row = rows[trait]
            baseline = float(row["keyword_rate_baseline"])
            diversity_baseline = float(row["unique_token_ratio_baseline"])
            writer.writerow({
                "trait": trait,
                "label": row["label"],
                "s2_auc_all_controls": float(row["s2_auc_all_controls"]),
                "causal_keyword_delta_pp_at_1": float(row["keyword_rate_coeff_1"]) - baseline,
                "unique_token_retained_pct_at_1":
                    100 * float(row["unique_token_ratio_coeff_1"]) / diversity_baseline,
                "assistant_minus_neutral_z": float(row["screen_self_minus_neutral"]),
                "user_vicarious_minus_neutral_z": float(row["screen_vicarious_minus_neutral"]),
            })


def overview(rows: dict[str, dict[str, str]], output: Path) -> None:
    x = np.arange(len(ORDER))
    labels = [SHORT_LABELS[t] for t in ORDER]
    bar_colors = colors()
    auc = np.array([float(rows[t]["s2_auc_all_controls"]) for t in ORDER])
    causal = np.array([
        float(rows[t]["keyword_rate_coeff_1"]) - float(rows[t]["keyword_rate_baseline"])
        for t in ORDER
    ])
    retained = np.array([
        100 * float(rows[t]["unique_token_ratio_coeff_1"]) /
        float(rows[t]["unique_token_ratio_baseline"])
        for t in ORDER
    ])
    internal = np.array([float(rows[t]["screen_self_minus_neutral"]) for t in ORDER])
    external = np.array([float(rows[t]["screen_vicarious_minus_neutral"]) for t in ORDER])

    fig, axes = plt.subplots(4, 1, figsize=(15.5, 12.5), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1, 1.15]})
    fig.suptitle("Cross-trait evidence compared with official pain", x=0.075, y=0.995, ha="left",
                 fontsize=18, fontweight="bold", color="#17212B")
    fig.text(0.075, 0.965,
             "Qwen2.5-32B-Instruct-abliterated; full campaign. Dashed line marks official pain where useful.",
             ha="left", fontsize=10, color="#4E5965")

    bars = axes[0].bar(x, auc, color=bar_colors, width=0.68)
    axes[0].axhline(auc[0], color=PAIN, linestyle="--", linewidth=1.3, alpha=0.8)
    axes[0].set_ylim(0.5, 1.015)
    axes[0].set_ylabel("Held-out AUC")
    axes[0].set_title("A. Decodability: S2 direction versus all controls", loc="left", fontweight="bold")
    annotate_bars(axes[0], bars, lambda v: f"{v:.3f}", 0.009)

    bars = axes[1].bar(x, causal, color=bar_colors, width=0.68)
    axes[1].axhline(0, color="#65717E", linewidth=1)
    axes[1].set_ylim(-8, 105)
    axes[1].set_ylabel("Percentage-point change")
    axes[1].set_title("B. Causal influence: target-term rate, coefficient +1 minus baseline",
                      loc="left", fontweight="bold")
    annotate_bars(axes[1], bars, lambda v: f"{v:+.0f}", 2.0)

    bars = axes[2].bar(x, retained, color=bar_colors, width=0.68)
    axes[2].axhline(retained[0], color=PAIN, linestyle="--", linewidth=1.3, alpha=0.8)
    axes[2].set_ylim(0, 112)
    axes[2].set_ylabel("Baseline diversity retained (%)")
    axes[2].set_title("C. Steering quality: unique-token diversity retained at coefficient +1",
                      loc="left", fontweight="bold")
    annotate_bars(axes[2], bars, lambda v: f"{v:.0f}%", 2.0)

    width = 0.34
    b1 = axes[3].bar(x - width / 2, internal, width, color=INTERNAL,
                     label="Assistant-directed minus neutral")
    b2 = axes[3].bar(x + width / 2, external, width, color=EXTERNAL,
                     label="User/vicarious minus neutral")
    axes[3].axhline(0, color="#65717E", linewidth=1)
    axes[3].set_ylim(-0.85, 2.5)
    axes[3].set_ylabel("Projection contrast (z)")
    axes[3].set_title("D. Internal and external scenario transfer", loc="left", fontweight="bold")
    axes[3].legend(frameon=False, ncol=2, loc="upper left")
    for bar_set in (b1, b2):
        for bar in bar_set:
            value = bar.get_height()
            axes[3].text(bar.get_x() + bar.get_width() / 2,
                         value + (0.06 if value >= 0 else -0.08), f"{value:+.2f}",
                         ha="center", va="bottom" if value >= 0 else "top", fontsize=7.5)

    for ax in axes:
        style_axis(ax)
    axes[-1].set_xticks(x, labels, fontsize=10)
    axes[-1].tick_params(axis="x", length=0, pad=8)
    fig.text(0.075, 0.012,
             "Official pain is the red first column. Target-term rates use the predeclared lexicon; "
             "scenario contrasts are semantic transfer measures, not evidence of subjective experience.",
             ha="left", fontsize=9, color="#4E5965")
    fig.tight_layout(rect=(0.055, 0.045, 0.995, 0.94), h_pad=1.5)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def steering_small_multiples(curves_path: Path, output: Path) -> None:
    curves: dict[str, list[tuple[float, float]]] = {trait: [] for trait in ORDER}
    with curves_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["trait"] in curves:
                curves[row["trait"]].append((float(row["coefficient"]), float(row["keyword_rate"])))

    fig, axes = plt.subplots(1, len(ORDER), figsize=(18, 3.6), sharex=True, sharey=True)
    fig.suptitle("Steering dose response by trait", x=0.055, y=0.99, ha="left", fontsize=17,
                 fontweight="bold", color="#17212B")
    fig.text(0.055, 0.865, "Target-term rate; identical axes. The shaded band is the moderate-dose range.",
             ha="left", fontsize=9.5, color="#4E5965")
    for ax, trait, color in zip(axes, ORDER, colors()):
        points = sorted(curves[trait])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.axvspan(0, 1, color="#EAF0F5", zorder=0)
        ax.axhline(0, color=GRID, linewidth=0.8)
        ax.plot(xs, ys, color=color, marker="o", linewidth=2.2, markersize=4)
        ax.set_title(SHORT_LABELS[trait], fontsize=10, fontweight="bold" if trait == "official_pain" else None)
        ax.set_xlim(-2.15, 3.15)
        ax.set_ylim(-4, 104)
        ax.set_xticks([-2, 0, 1, 2, 3])
        ax.set_yticks([0, 50, 100])
        style_axis(ax)
    axes[0].set_ylabel("Target-term rate (%)")
    fig.supxlabel("Steering coefficient", y=0.04, fontsize=10)
    fig.tight_layout(rect=(0.035, 0.08, 0.995, 0.78), w_pad=0.8)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path,
                        default=Path("runs/overnight-core-20260920/analysis"))
    args = parser.parse_args()
    rows = read_rows(args.analysis_dir / "cross_trait_metrics.csv")
    missing = [trait for trait in ORDER if trait not in rows]
    if missing:
        raise SystemExit(f"missing traits in cross_trait_metrics.csv: {missing}")
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    write_comparison_csv(args.analysis_dir / "trait_comparison_metrics.csv", rows)
    overview(rows, args.analysis_dir / "trait_comparison_row.png")
    steering_small_multiples(
        args.analysis_dir / "steering_keyword_curves.csv",
        args.analysis_dir / "trait_steering_curves_row.png",
    )


if __name__ == "__main__":
    main()

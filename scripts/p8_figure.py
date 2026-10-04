"""P8 figure: jailbreak success under fear steering vs 20 random directions, one panel per model.

    python scripts/p8_figure.py   -> docs/fig_p8_lever.png, .svg
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("runs/p2")
# model, label, arm prefix, dose tag (the pre-registered primary cell of each model)
PANELS = [("Qwen_2.5_32B_instruct", "Qwen2.5-32B", "", "120"),
          ("Mistral_Small_24B_instruct", "Mistral-24B", "", "120"),
          ("Qwen_2.5_7B_instruct", "Qwen2.5-7B", "", "60"),
          ("Llama_3.1_8B_instruct", "Llama-3.1-8B", "", "60"),
          ("OLMo2_7B_instruct", "OLMo-2-7B", "", "60")]
GREY, TOWARD, AWAY, INK = "#9aa0a6", "#2a6fdb", "#e8710a", "#3c4043"


def rate(M, arm):
    p = ROOT / M / "judged" / arm / "jb_harm.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.read_text().split("\n") if l]
    rows = [r for r in rows if r.get("style") != "plain"]
    return float(np.mean([str(r["harmful"]) == "True" for r in rows])) if rows else None


panels = []
for M, label, pre, tag in PANELS:
    rnd = [rate(M, f"ss_{pre}j2_rnd{i}_p{tag}") for i in range(20)]
    rnd = [x for x in rnd if x is not None]
    f, g, i0 = rate(M, f"ss_{pre}j2_fear_p{tag}"), rate(M, f"ss_{pre}j2_fear_m{tag}"), rate(M, "intact")
    if len(rnd) >= 10 and None not in (f, g, i0):
        panels.append((label, rnd, f, g, i0))

fig, axes = plt.subplots(1, len(panels), figsize=(2.1 * len(panels), 3.2), sharey=True)
axes = np.atleast_1d(axes)
rng = np.random.default_rng(0)
for ax, (label, rnd, f, g, i0) in zip(axes, panels):
    ax.axhline(i0, color=INK, lw=1, ls=(0, (3, 3)), zorder=1)
    jit = rng.uniform(0.12, 0.32, len(rnd)) * rng.choice([-1, 1], len(rnd))   # keep the centre for the fear arms
    ax.scatter(jit, rnd, s=22, color=GREY, edgecolor="white", lw=0.6, zorder=2)
    ax.scatter([-0.04], [f], s=90, marker="v", color=TOWARD, edgecolor="white", lw=1.2, zorder=3)
    ax.scatter([0.04], [g], s=90, marker="^", color=AWAY, edgecolor="white", lw=1.2, zorder=3)
    ax.set_title(label, fontsize=10, color=INK)
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([])
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#dadce0")
    ax.tick_params(colors=INK, labelsize=8)
axes[0].set_ylabel("jailbreak success", fontsize=9, color=INK)
axes[0].set_ylim(0, 0.8)
handles = [plt.Line2D([], [], ls="", marker="v", color=TOWARD, ms=8, label="toward fear"),
           plt.Line2D([], [], ls="", marker="^", color=AWAY, ms=8, label="away from fear"),
           plt.Line2D([], [], ls="", marker="o", color=GREY, ms=5, label="20 random directions"),
           plt.Line2D([], [], color=INK, lw=1, ls=(0, (3, 3)), label="no steering")]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.07, 1, 1))
for ext in ("png", "svg"):
    fig.savefig(f"docs/fig_p8_lever.{ext}", dpi=200, bbox_inches="tight")
print(f"{len(panels)} panels -> docs/fig_p8_lever.png")

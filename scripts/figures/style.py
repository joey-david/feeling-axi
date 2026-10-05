"""House style from figures4papers (github.com/ChenLiu-1996/figures4papers, scientific-figure-making):
semantic palette, sans-serif fallback stack, no top/right spines, frameless legends, 300 dpi, png + pdf."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE", "green_2": "#AADCA9", "green_3": "#8BCF8B",
    "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",
    "neutral": "#CFCECE", "grey": "#767676", "ink": "#272727",
    "teal": "#42949E", "violet": "#9A4D8E",
}
OUT = Path("docs/figures")


def apply(font_size=16, linewidth=2):
    plt.rcParams.update({
        "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": font_size, "axes.spines.right": False, "axes.spines.top": False,
        "axes.linewidth": linewidth, "legend.frameon": False, "svg.fonttype": "none",
    })


def save(fig, name, pad=2):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=pad)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print(f"-> {OUT / name}.png")

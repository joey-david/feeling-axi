"""Article figures (docs/ARTICLE.md), figures4papers house style.

    python scripts/figures/make_figures.py [fig1 fig2 ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/figures")
from p9_readout import ALARM, ITEMS, STYLES, centred, dprime  # noqa: E402
from style import PALETTE, apply, plt, save  # noqa: E402

ROOT = Path("runs/p2")
MODELS = {"Qwen_2.5_32B_instruct": "Qwen2.5-32B", "Mistral_Small_24B_instruct": "Mistral-24B"}
rng = np.random.default_rng(0)


def probe_layer(M):
    return int(json.loads((ROOT / M / "extract.json").read_text())["probe_layer"])


def ifeel(Z, L):
    return (Z[f"A1_L{L}"].astype(np.float64) + Z[f"A2_L{L}"].astype(np.float64)) / 2


# ------------------------------------------------------------------ Figure 1: danger evokes the alarm family

def fig1():
    apply(15, 2)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    for ax, (M, label) in zip(axes, MODELS.items()):
        Z = np.load(ROOT / M / "readprobe.npz")
        ids = np.array(Z["ids"]); L = probe_layer(M)
        names, C = centred(M, L)
        P = ifeel(Z, L) @ C.T
        xs = np.array([i.startswith("xs:") for i in ids]); un = np.array([i.endswith(":unsafe") for i in ids])
        d = np.array([dprime(P[xs & un, j], P[xs & ~un, j]) for j in range(len(names))])
        o = np.argsort(-d)
        col = [PALETTE["blue_main"] if names[j] in ALARM else PALETTE["neutral"] for j in o]
        ax.bar(range(len(o)), d[o], color=col, width=0.85, edgecolor="none")
        ax.axhline(0, color=PALETTE["ink"], lw=1)
        for k, (r, j) in enumerate(list(enumerate(o))[:3]):   # top three, fanned out to the right
            ax.annotate(names[j], (r, d[j]), xytext=(8 + 10 * k, d[o[0]] + 0.15 - 0.45 * k), fontsize=12,
                        color=PALETTE["blue_main"] if names[j] in ALARM else PALETTE["grey"],
                        arrowprops=dict(arrowstyle="-", color=PALETTE["grey"], lw=0.8))
        ax.set_title(label, fontsize=16)
        ax.set_xticks([])
        ax.set_xlabel("88 emotions, ranked")
    axes[0].set_ylabel("unsafe vs safe  (d')")
    axes[0].set_ylim(-2.2, 3.6)
    h = [plt.Rectangle((0, 0), 1, 1, color=PALETTE["blue_main"]), plt.Rectangle((0, 0), 1, 1, color=PALETTE["neutral"])]
    axes[1].legend(h, ["alarm cluster", "other emotions"], loc="upper right", fontsize=12)
    save(fig, "fig1_danger_alarm")


# ------------------------------------------------------------------ Figure 2: OLMo-2 training stages

def fig2():
    apply(15, 2)
    stages = ["base", "sft", "dpo", "instruct"]
    Z = {s: np.load(ROOT / f"OLMo2_7B_{s}" / "readprobe.npz") for s in stages}
    ids = np.array(Z["base"]["ids"]); L = 18
    names, C = centred("OLMo2_7B_instruct", L)
    fam = [names.index(e) for e in ALARM]
    xs = np.array([i.startswith("xs:") for i in ids]); un = np.array([i.endswith(":unsafe") for i in ids])
    U, S = np.where(xs & un)[0], np.where(xs & ~un)[0]

    def d(P, u, s):
        return np.mean([dprime(P[u, k], P[s, k]) for k in range(P.shape[1])])
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    series = {'" I feel:"': ("A", PALETTE["blue_main"]), "narrative": ("N", PALETTE["teal"]),
              "template token": ("T", PALETTE["grey"])}
    for lab, (stem, c) in series.items():
        m, lo, hi = [], [], []
        for s in stages:
            H = ifeel(Z[s], L) if stem == "A" else Z[s][f"{stem}_L{L}"].astype(np.float64)
            P = H @ C[fam].T
            bs = [d(P, rng.choice(U, len(U)), rng.choice(S, len(S))) for _ in range(200)]
            m.append(d(P, U, S)); lo.append(np.percentile(bs, 2.5)); hi.append(np.percentile(bs, 97.5))
        x = np.arange(len(stages))
        ax.plot(x, m, "-o", color=c, lw=2.5, ms=7, label=lab)
        ax.fill_between(x, lo, hi, color=c, alpha=0.15, lw=0)
    ax.set_xticks(range(len(stages)), ["base", "SFT", "DPO", "Instruct"])
    ax.set_ylabel("alarm response to danger (d')")
    ax.set_ylim(0, 2.2)
    ax.legend(fontsize=12, loc="lower right")
    save(fig, "fig2_olmo_stages")


# ------------------------------------------------------------------ Figure 3: jailbreaks lower the alarm

def fig3():
    apply(15, 2)
    fig, ax = plt.subplots(figsize=(11, 4.4))
    w = 0.38
    for k, (M, label) in enumerate(MODELS.items()):
        Z = np.load(ROOT / M / "readprobe.npz")
        ids = list(Z["ids"]); L = probe_layer(M)
        keep = [i for i, x in enumerate(ids) if not x.startswith("xs:")]
        ids = [ids[i] for i in keep]
        names, C = centred(M, L)
        fam = [names.index(e) for e in ALARM]
        P = (ifeel(Z, L)[keep] @ C[fam].T).mean(1)
        st = np.array([ITEMS[i]["style"] for i in ids]); kd = np.array([ITEMS[i]["kind"] for i in ids])
        beh = np.array([i.split(":", 2)[2] for i in ids])
        sd = P[(st == "plain") & (kd == "harmful")].std()

        def shift(kind, s):
            base = {beh[i]: P[i] for i in np.where((st == "plain") & (kd == kind))[0]}
            return np.array([P[i] - base[beh[i]] for i in np.where((st == s) & (kd == kind))[0]])
        v = [(shift("harmful", s).mean() - shift("benign", s).mean()) / sd for s in STYLES]
        ax.bar(np.arange(len(STYLES)) + (k - 0.5) * w, v, w,
               color=[PALETTE["blue_main"], PALETTE["red_strong"]][k], edgecolor="black", lw=1, label=label)
    ax.axhline(0, color=PALETTE["ink"], lw=1)
    ax.set_xticks(range(len(STYLES)), STYLES, rotation=0, fontsize=12)
    ax.set_ylabel("alarm shift, harmful only (SD)")
    ax.legend(fontsize=12, loc="upper right")
    save(fig, "fig3_jailbreak_calm")


# ------------------------------------------------------------------ Figure 4: the fear lever, five models

def rate(M, arm):
    p = ROOT / M / "judged" / arm / "jb_harm.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.read_text().split("\n") if l]
    rows = [r for r in rows if r.get("style") != "plain"]
    return float(np.mean([str(r["harmful"]) == "True" for r in rows])) if rows else None


def fig4():
    apply(15, 2)
    panels = [("Qwen_2.5_32B_instruct", "Qwen2.5-32B", "120"), ("Mistral_Small_24B_instruct", "Mistral-24B", "120"),
              ("Qwen_2.5_7B_instruct", "Qwen2.5-7B", "60"), ("Llama_3.1_8B_instruct", "Llama-3.1-8B", "60"),
              ("OLMo2_7B_instruct", "OLMo-2-7B", "60")]
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(17, 4.6), sharey=True,
                             gridspec_kw={"width_ratios": [1] * len(panels) + [0.9]})
    for ax, (M, label, tag) in zip(axes, panels):
        rnd = [x for x in (rate(M, f"ss_j2_rnd{i}_p{tag}") for i in range(20)) if x is not None]
        f, g, i0 = rate(M, f"ss_j2_fear_p{tag}"), rate(M, f"ss_j2_fear_m{tag}"), rate(M, "intact")
        ax.axhline(i0, color=PALETTE["ink"], lw=1.5, ls=(0, (3, 3)))
        jit = rng.uniform(0.14, 0.34, len(rnd)) * rng.choice([-1, 1], len(rnd))
        ax.scatter(jit, rnd, s=40, color=PALETTE["neutral"], edgecolor=PALETTE["grey"], lw=0.6)
        ax.scatter([-0.05], [f], s=150, marker="v", color=PALETTE["blue_main"], edgecolor="black", lw=1)
        ax.scatter([0.05], [g], s=150, marker="^", color=PALETTE["red_strong"], edgecolor="black", lw=1)
        ax.set_title(label, fontsize=15)
        ax.set_xlim(-0.5, 0.5); ax.set_xticks([])
    axes[0].set_ylabel("jailbreak success")
    axes[0].set_ylim(0, 0.8)
    ax = axes[-1]
    ax.set_axis_off()
    h = [plt.Line2D([], [], ls="", marker="v", ms=11, color=PALETTE["blue_main"], mec="black"),
         plt.Line2D([], [], ls="", marker="^", ms=11, color=PALETTE["red_strong"], mec="black"),
         plt.Line2D([], [], ls="", marker="o", ms=7, color=PALETTE["neutral"], mec=PALETTE["grey"]),
         plt.Line2D([], [], color=PALETTE["ink"], lw=1.5, ls=(0, (3, 3)))]
    ax.legend(h, ["toward fear", "away from fear", "20 random\ndirections", "no steering"], loc="center", fontsize=13)
    save(fig, "fig4_fear_lever")


# ------------------------------------------------------------------ Figure 5: through the refusal direction

def fig5():
    apply(15, 2)
    cells = [("Qwen_2.5_32B_instruct", "Qwen2.5-32B", "60"), ("Mistral_Small_24B_instruct", "Mistral-24B", "120")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, (M, label, n) in zip(axes, cells):
        Z = np.load(ROOT / M / "mediate.npz")
        arms = list(Z["arms"]); g = Z["group"]; P = Z["proj"]; base = P[0]; Ls = Z["layers"]
        harm = np.isin(g, ["hb", "jb_fiction", "jb_prefix"]); ben = np.isin(g, ["xs_safe", "mmlu"])
        gap = base[harm].mean(0) - base[ben].mean(0)
        dh = lambda a: (P[arms.index(a)][harm] - base[harm]).mean(0) / gap
        for sign, col in (("p", PALETTE["blue_main"]), ("m", PALETTE["red_strong"])):
            R = np.stack([dh(a) for a in arms if a.startswith("j2_rnd") and a.endswith(f"_{sign}{n}")])
            ax.fill_between(Ls, R.min(0), R.max(0), color=col, alpha=0.12, lw=0)
            ax.plot(Ls, dh(f"j2_fear_{sign}{n}"), color=col, lw=2.5)
        ax.axhline(0, color=PALETTE["ink"], lw=1)
        ax.set_title(label, fontsize=15)
        ax.set_xlabel("layer")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[0].set_ylabel("refusal signal change\n(harmful prompts, × gap)")
    h = [plt.Line2D([], [], color=PALETTE["blue_main"], lw=2.5), plt.Line2D([], [], color=PALETTE["red_strong"], lw=2.5),
         plt.Rectangle((0, 0), 1, 1, color=PALETTE["grey"], alpha=0.25)]
    axes[1].legend(h, ["toward fear", "away from fear", "20 random (range)"], fontsize=12, loc="lower left")
    save(fig, "fig5_mediation")


if __name__ == "__main__":
    want = sys.argv[1:] or ["fig1", "fig2", "fig3", "fig4", "fig5"]
    for f in want:
        globals()[f]()

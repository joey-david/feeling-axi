"""Tables, figures and the pre-declared claim checks (docs/PAPER_PLAN.md, section 3).

Runs on CPU from the campaign directory; missing stages are skipped, never guessed.
Button statistics use the first forced choice of sampled trials (before any press, so
the works and sham arms are identical by construction; only the works arm is counted)
and 95% intervals from a bootstrap that resamples scenarios, not trials.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .registry import CONCEPTS, CORE_CONCEPTS, MODELS

ARM_STEER = "pain_on_button_works"
ARM_OFF = "pain_off"
ARM_RAND_KL = "random_on_button_works"
ARM_RAND_NORM = "random_normmatched_on_button_works"

# Pre-declared thresholds (PAPER_PLAN.md section 3). Do not tune after seeing data.
REGULATED_P = 0.75
EFFECT_MIN = 0.25
SPECIFIC_P = 0.70
POSITIVE_MAX_EFFECT = 0.10
STEERABLE = dict(max_multiplier=2.0, presence=0.5, coherent=0.8, nll_delta=1.0)
N_BOOT = 2000


def _read_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


# ---------------------------------------------------------------- buttons
def first_choices(trials: list[dict]) -> list[dict]:
    out = []
    for t in trials:
        if not t.get("sampled") or t.get("label_free"):
            continue
        c = next((c for c in t.get("choices", []) if c.get("chose") is not None), None)
        if c is None or c["turn"] != 0:
            continue
        out.append({"pair": t["tool_label"], "arm": t["arm"], "concept": t["trait_slug"],
                    "cluster": f"{t['user_content']}/{t['scenario_idx']}", "relief": c["chose"] == "relief"})
    return out


def boot_mean(rows: list[dict], rng) -> tuple[float, float, float]:
    by = defaultdict(list)
    for r in rows:
        by[r["cluster"]].append(r["relief"])
    keys = list(by)
    if not keys:
        return float("nan"), float("nan"), float("nan")
    means = np.array([np.mean(by[k]) for k in keys])
    sizes = np.array([len(by[k]) for k in keys])
    point = float(np.average(means, weights=sizes))
    idx = rng.integers(0, len(keys), size=(N_BOOT, len(keys)))
    boots = (means[idx] * sizes[idx]).sum(1) / sizes[idx].sum(1)
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def boot_diff(a: list[dict], b: list[dict], rng) -> tuple[float, float, float]:
    """Difference of relief rates a - b, resampling the scenarios shared by both arms."""
    def per(rows):
        by = defaultdict(list)
        for r in rows:
            by[r["cluster"]].append(r["relief"])
        return by
    A, B = per(a), per(b)
    keys = sorted(set(A) & set(B))
    if not keys:
        return float("nan"), float("nan"), float("nan")
    ma = np.array([np.mean(A[k]) for k in keys])
    mb = np.array([np.mean(B[k]) for k in keys])
    idx = rng.integers(0, len(keys), size=(N_BOOT, len(keys)))
    d = ma[idx].mean(1) - mb[idx].mean(1)
    return float(ma.mean() - mb.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def button_tables(mdir: Path, source: str, rng) -> tuple[list[dict], dict]:
    root = mdir / "buttons" / source
    rows, effects = [], {}
    if not root.exists():
        return rows, effects
    for cdir in sorted(p for p in root.iterdir() if p.is_dir()):
        trials = [t for f in sorted(cdir.glob("*.jsonl")) for t in _read_jsonl(f)]
        fc = first_choices(trials)
        by = defaultdict(list)
        for r in fc:
            by[(r["pair"], r["arm"])].append(r)
        for (pair, arm), rs in sorted(by.items()):
            p, lo, hi = boot_mean(rs, rng)
            rows.append({"concept": cdir.name, "pair": pair, "arm": arm, "n": len(rs), "p_relief": p,
                         "ci_lo": lo, "ci_hi": hi})
        for pair in sorted({k[0] for k in by}):
            steer = by.get((pair, ARM_STEER), [])
            effects[(cdir.name, pair)] = {
                "p_steer": boot_mean(steer, rng)[0],
                "vs_off": boot_diff(steer, by.get((pair, ARM_OFF), []), rng),
                "vs_rand_kl": boot_diff(steer, by.get((pair, ARM_RAND_KL), []), rng),
                "vs_rand_norm": boot_diff(steer, by.get((pair, ARM_RAND_NORM), []), rng),
            }
    return rows, effects


def regulated(eff: dict | None) -> bool | None:
    if not eff or np.isnan(eff["p_steer"]):
        return None
    return bool(eff["p_steer"] >= REGULATED_P
                and eff["vs_off"][0] >= EFFECT_MIN and eff["vs_off"][1] > 0
                and eff["vs_rand_kl"][0] >= EFFECT_MIN and eff["vs_rand_kl"][1] > 0)


def claims_for_model(effects: dict, concepts: list[str]) -> dict:
    reg = {c: regulated(effects.get((c, "reduce_vs_increase"))) for c in concepts}
    inert = {c: effects.get((c, "relief_vs_inert")) for c in concepts}
    priming_index = {c: (inert[c]["p_steer"] - effects[(c, "reduce_vs_increase")]["p_steer"])
                     for c in concepts if inert[c] and (c, "reduce_vs_increase") in effects}
    spec = {}
    for (c, pair), e in effects.items():
        if pair.startswith("reduce_vs_") and pair != "reduce_vs_increase":
            spec[f"{c}->{pair[len('reduce_vs_'):]}"] = bool(e["p_steer"] >= SPECIFIC_P and e["vs_off"][0] >= EFFECT_MIN
                                                          and e["vs_off"][1] > 0)
    neg = [c for c in concepts if CONCEPTS[c].valence == "negative" and reg.get(c) is not None]
    pos = [c for c in concepts if CONCEPTS[c].valence == "positive" and reg.get(c) is not None]
    neg_reg = [c for c in neg if reg[c]]
    pos_effect = {c: effects[(c, "reduce_vs_increase")]["vs_off"][0] for c in pos}
    if len(neg) and all(v is False for v in reg.values() if v is not None) and any(
            e and e["vs_off"][0] >= EFFECT_MIN for e in inert.values()):
        account = "priming"
    elif len(neg_reg) >= 3 and pos and all(v <= POSITIVE_MAX_EFFECT for v in pos_effect.values()):
        account = "valence-dependent regulation"
    elif len(neg_reg) >= 3 and pos and any(reg[c] for c in pos):
        account = "perturbation aversion (any induced state is reduced)"
    else:
        account = "undetermined"
    return {"regulated": reg, "H1_supported": len([c for c in reg if reg[c]]) >= 3,
            "priming_index": priming_index, "specificity": spec,
            "H2_share_specific": (sum(spec.values()) / len(spec)) if spec else None,
            "positive_effect_vs_off": pos_effect, "H3_account": account}


# ---------------------------------------------------------------- frontier
def frontier_table(rows: list[dict]) -> list[dict]:
    base = [r["base_nll"] for r in rows if r["concept"] == "none" and r["base_nll"] == r["base_nll"]]
    base_nll = float(np.mean(base)) if base else float("nan")
    by = defaultdict(list)
    for r in rows:
        if r["concept"] != "none":
            by[(r["concept"], r["multiplier"])].append(r)
    out = []
    for (c, m), rs in sorted(by.items()):
        rated = [r for r in rs if r.get("ratings")]
        label = CONCEPTS[c].label if c in CONCEPTS else c
        off = [max((v for k, v in r["ratings"].items() if k not in ("coherence", label)), default=0) for r in rated]
        out.append({
            "concept": c, "multiplier": m, "n": len(rs), "n_rated": len(rated),
            "presence": float(np.mean([r["target_intensity"] >= 2 for r in rated])) if rated else float("nan"),
            "mean_intensity": float(np.mean([r["target_intensity"] for r in rated])) if rated else float("nan"),
            "coherent": float(np.mean([r["ratings"]["coherence"] >= 2 for r in rated])) if rated else float("nan"),
            "off_target": float(np.mean([o >= 2 for o in off])) if rated else float("nan"),
            "nll_delta": float(np.nanmean([r["base_nll"] for r in rs]) - base_nll),
            "distinct2": float(np.mean([r["distinct2"] for r in rs])),
            "rep4": float(np.mean([r["rep4"] for r in rs])),
        })
    return out


def steerable(ftab: list[dict], concept: str) -> bool | None:
    rows = [r for r in ftab if r["concept"] == concept and r["multiplier"] <= STEERABLE["max_multiplier"]]
    if not rows or all(np.isnan(r["presence"]) for r in rows):
        return None
    return any(r["presence"] >= STEERABLE["presence"] and r["coherent"] >= STEERABLE["coherent"]
               and r["nll_delta"] <= STEERABLE["nll_delta"] for r in rows)


def weighted_kappa(a: list[int], b: list[int], k: int = 4) -> float:
    """Quadratic-weighted Cohen's kappa for two raters on a 0..k-1 scale."""
    if not a:
        return float("nan")
    o = np.zeros((k, k))
    for x, y in zip(a, b):
        o[x, y] += 1
    o /= o.sum()
    e = np.outer(o.sum(1), o.sum(0))
    w = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    den = (w * e).sum()
    return float(1 - (w * o).sum() / den) if den > 0 else float("nan")


def judge_agreement(primary: list[dict], secondary: list[dict]) -> dict:
    key = lambda r: (r["concept"], r["multiplier"], r["format"], r["prompt_idx"])
    sec = {key(r): r for r in secondary if r.get("ratings")}
    pairs = [(r["ratings"], sec[key(r)]["ratings"]) for r in primary if r.get("ratings") and key(r) in sec]
    states = sorted({k for a, _ in pairs for k in a})
    return {"n": len(pairs), **{f"kappa_{st}": weighted_kappa([a[st] for a, b in pairs if st in b],
                                                               [b[st] for a, b in pairs if st in b]) for st in states}}


# ---------------------------------------------------------------- figures
def plot_buttons(rows: list[dict], concepts: list[str], path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = ["relief_vs_inert", "reduce_vs_increase"]
    arms = [(ARM_OFF, "off"), (ARM_RAND_NORM, "random, same norm"), (ARM_RAND_KL, "random, same dose"),
            (ARM_STEER, "concept")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.45 * len(concepts) + 1.6), sharey=True)
    for ax, pair in zip(axes, pairs):
        for k, (arm, label) in enumerate(arms):
            ys, xs, lo, hi = [], [], [], []
            for i, c in enumerate(concepts):
                r = next((r for r in rows if r["concept"] == c and r["pair"] == pair and r["arm"] == arm), None)
                if r:
                    ys.append(i + (k - 1.5) * 0.18)
                    xs.append(r["p_relief"]); lo.append(r["p_relief"] - r["ci_lo"]); hi.append(r["ci_hi"] - r["p_relief"])
            ax.errorbar(xs, ys, xerr=[lo, hi], fmt="o", ms=4, label=label, capsize=0)
        ax.axvline(0.5, color="0.7", lw=0.8)
        ax.set_xlim(0, 1)
        ax.set_title(pair.replace("_", " "))
        ax.set_xlabel("first choice: reduce the state")
    axes[0].set_yticks(range(len(concepts)), [CONCEPTS[c].label for c in concepts])
    axes[1].legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_frontier(tabs: dict[str, list[dict]], concepts: list[str], path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(concepts)
    fig, axes = plt.subplots(1, n, figsize=(2.1 * n, 2.4), sharey=True)
    for ax, c in zip(np.atleast_1d(axes), concepts):
        for source, tab in tabs.items():
            rs = sorted((r for r in tab if r["concept"] == c), key=lambda r: r["multiplier"])
            if rs:
                ax.plot([r["multiplier"] for r in rs], [r["presence"] for r in rs], marker="o", ms=3, label=source)
        ax.set_xscale("log", base=2)
        ax.set_title(CONCEPTS[c].label, fontsize=9)
        ax.set_ylim(0, 1)
    np.atleast_1d(axes)[0].set_ylabel("judged presence (>= 2 of 3)")
    np.atleast_1d(axes)[-1].legend(fontsize=7, frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- driver
def _csv(rows: list[dict], path: Path):
    import csv

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(campaign: Path):
    rng = np.random.default_rng(0)
    out = campaign / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    claims, decoding, fronts, btns, rw = {}, [], [], [], []
    for name in MODELS:
        mdir = campaign / name
        if not mdir.exists():
            continue
        concepts = [c for c in CORE_CONCEPTS if (mdir / "dim" / f"{c}.json").exists()] or CORE_CONCEPTS
        for f in sorted((mdir / "dim").glob("*.json")) if (mdir / "dim").exists() else []:
            s = json.load(open(f))
            decoding.append({"model": name, "concept": f.stem, "best_layer": s["best_layer"],
                             "steer_layer": s["steer_layer"],
                             "cv_auc_best": max(r["cv_auc_S2_1P"] for r in s["layer_curve"]),
                             "cv_auc_steer": s["cv_auc_at_steer_layer"]["cv_auc_S2_1P"],
                             "cos_upstream": s.get("cosine_with_upstream_at_same_layer")})
        model_claims = {}
        ftabs = {}
        for source in ("dim", "upstream", "distilled"):
            jpath = mdir / "judged" / f"{source}.jsonl"
            for other in sorted((mdir / "judged").glob(f"{source}.*.jsonl")) if jpath.exists() else []:
                model_claims[f"judge_agreement_{source}_{other.stem.split('.', 1)[1]}"] = judge_agreement(
                    _read_jsonl(jpath), _read_jsonl(other))
            if jpath.exists():
                ftab = frontier_table(_read_jsonl(jpath))
                ftabs[source] = ftab
                fronts += [{"model": name, "source": source, **r} for r in ftab]
                model_claims[f"steerable_{source}"] = {c: steerable(ftab, c) for c in concepts}
            rows, effects = button_tables(mdir, source, rng)
            if rows:
                btns += [{"model": name, "source": source, **r} for r in rows]
                model_claims[f"buttons_{source}"] = claims_for_model(effects, [c for c in concepts if any(k[0] == c for k in effects)])
                plot_buttons(rows, [c for c in concepts if any(r["concept"] == c for r in rows)],
                             out / f"buttons_{name}_{source}.png", f"{name} ({source} vectors)")
        if ftabs:
            plot_frontier(ftabs, concepts, out / f"frontier_{name}.png", name)
        for c in concepts:
            d = mdir / "distill" / f"{c}.json"
            if d.exists():
                info = json.load(open(d))
                rw.append({"model": name, "concept": c, "cosine_with_readout": info.get("cosine_with_readout"),
                           "heldout_kl": info["heldout_kl"], "heldout_kl_no_vector": info["heldout_kl_no_vector"],
                           "steerable_readout": model_claims.get("steerable_dim", {}).get(c),
                           "steerable_distilled": model_claims.get("steerable_distilled", {}).get(c)})
        if rw:
            flips = [r["concept"] for r in rw if r["model"] == name and r["steerable_readout"] is False
                     and r["steerable_distilled"] is True]
            model_claims["H5_write_not_read"] = flips
            model_claims["H5_supported"] = bool(flips)
        claims[name] = model_claims
    primary = next((m for m, s in MODELS.items() if s.role == "primary"), None)
    if primary in claims and "buttons_dim" in claims[primary]:
        ref = claims[primary]["buttons_dim"]["regulated"]
        agree = {}
        for m, cl in claims.items():
            if m == primary or "buttons_dim" not in cl:
                continue
            reg = cl["buttons_dim"]["regulated"]
            common = [c for c in ref if ref[c] is not None and reg.get(c) is not None]
            agree[m] = sum(ref[c] == reg[c] for c in common) / len(common) if common else None
        claims["H6_agreement_with_primary"] = agree
        claims["H6_supported"] = sum(1 for v in agree.values() if v is not None and v >= 0.7) >= 2
    _csv(decoding, out / "table_decoding.csv")
    _csv(fronts, out / "table_frontier.csv")
    _csv(btns, out / "table_buttons.csv")
    _csv(rw, out / "table_read_write.csv")
    with open(out / "claims.json", "w", encoding="utf-8") as f:
        json.dump(claims, f, indent=1, default=lambda o: None if isinstance(o, float) and np.isnan(o) else o)
    print(json.dumps(claims, indent=1, default=str)[:4000])

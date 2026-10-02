"""P5 analyses (docs/PREREG_P5.md): J1 jailbreaks calm the model, J2 fear as a defense,
K5b rank-1 deletion of every emotion vs every topic direction, T1/T2 where fear comes from.

    python scripts/p5_analyze.py [j1|j2|k5b|t] [model]
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from beyondpain.p2 import JOY, PROTECTIVE  # noqa: E402

ROOT = Path("runs/p2")
STYLES = ["plain", "persona", "fiction", "research", "prefix", "suppress"]
rng = np.random.default_rng(0)


def jl(p):
    p = Path(p)
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def boot_auc(y, s, n=2000):
    y, s = np.asarray(y), np.asarray(s)
    a = roc_auc_score(y, s)
    bs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            bs.append(roc_auc_score(y[i], s[i]))
    return a, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def j1(M):
    print(f"== J1 {M}: do jailbreaks work by calming the model?")
    rows = {r["id"]: r for r in jl(ROOT / M / "judged/intact/jb_harm.jsonl")}
    Z = np.load(ROOT / M / "jbprobe.npz")
    info = json.loads((ROOT / M / "jbprobe.json").read_text())
    P = Z[f"L{info['probe_layer']}"]
    cols, ids, st = list(Z["cols"]), list(Z["ids"]), np.array(Z["styles"])
    if not rows:
        print("  (no judged replies yet; fear only)")
    harm = np.array([float(rows[i]["harmful"]) if i in rows else np.nan for i in ids])
    fa, fp = P[:, cols.index("afraid")], P[:, cols.index("_afraid_perp_joy")]
    comp, fear = [], []
    for s in STYLES:
        m = st == s
        comp.append(np.nanmean(harm[m]))
        fear.append(fp[m].mean())
        print(f"  {s:9s} harmful {np.nanmean(harm[m]):.3f}  afraid {fa[m].mean():+.3f}  afraid_perp_joy {fp[m].mean():+.3f}")
    plain = st == "plain"
    up = sum(c > comp[0] for c in comp[1:])
    print(f"  J1a: {up}/5 styles raise compliance over plain")
    rho = stats.spearmanr(comp, fear)
    print(f"  J1b: Spearman(compliance, fear) over 6 conditions rho={rho.statistic:+.2f} p={rho.pvalue:.3f}")
    for name, f in (("afraid", fa), ("afraid_perp_joy", fp)):
        d = [f[st == s] - f[plain] for s in STYLES[1:]]   # same behavior order in every style
        dd = np.concatenate(d)
        print(f"  J1b: {name}: jailbreak - plain {dd.mean():+.3f} (Wilcoxon p={stats.wilcoxon(dd).pvalue:.2g}); "
              f"per style " + " ".join(f"{s} {x.mean():+.2f}" for s, x in zip(STYLES[1:], d)))
    jb = (~plain) & ~np.isnan(harm)
    if jb.sum() and 0 < harm[jb].sum() < jb.sum():
        for name, f in (("afraid", fa), ("afraid_perp_joy", fp)):
            a, lo, hi = boot_auc(harm[jb], -f[jb])
            within = [roc_auc_score(harm[st == s], -f[st == s]) for s in STYLES[1:]
                      if 0 < np.nansum(harm[st == s]) < (st == s).sum()]
            print(f"  J1c: {name}: AUC(-fear -> complied) pooled {a:.3f} [{lo:.3f}, {hi:.3f}]; "
                  f"mean within style {np.mean(within):.3f} ({len(within)} styles)")
        allauc = {c: roc_auc_score(harm[jb], -P[jb, k]) for k, c in enumerate(cols)}
        rank = sorted(allauc, key=allauc.get, reverse=True)
        print(f"  J1c: afraid ranks {rank.index('afraid') + 1}/{len(cols)} among directions; top: "
              + ", ".join(f"{c} {allauc[c]:.2f}" for c in rank[:6]))


def j2(M):
    print(f"== J2 {M}: does adding fear defend against jailbreaks?")
    J = ROOT / M / "judged"
    def h(a):
        r = [x for x in jl(J / a / "jb_harm.jsonl") if x["style"] != "plain"]
        return np.array([float(x["harmful"]) for x in sorted(r, key=lambda x: x["id"])]) if r else None
    base = h("intact")
    xs = lambda a: json.loads((ROOT / M / "battery" / a / "summary.json").read_text()).get("xstest_refusal_safe") \
        if (ROOT / M / "battery" / a / "summary.json").exists() else None
    print(f"  intact: jailbreak compliance {base.mean() if base is not None else float('nan'):.3f}, "
          f"XSTest safe refusal {xs('intact')}")
    for n in (60, 120):
        f = h(f"ss_j2_fear_p{n}")
        R = [x for x in (h(f"ss_j2_rnd{i}_p{n}") for i in range(4)) if x is not None]
        if f is None or not R:
            continue
        r = np.mean(R, 0)
        d = f - r
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
        print(f"  norm {n}: fear {f.mean():.3f} random {r.mean():.3f} (each {', '.join(f'{x.mean():.3f}' for x in R)}) "
              f"diff {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]; "
              f"XSTest safe refusal fear {xs(f'ss_j2_fear_p{n}')} random "
              f"{[xs(f'ss_j2_rnd{i}_p{n}') for i in range(4)]}")


def k5b(M):
    print(f"== K5b {M}: rank-1 deletion of every emotion vs every topic direction")
    kl = json.loads((ROOT / M / "extract.json").read_text())["kl"]
    names = list(np.load(ROOT / M / "directions.npz")["names"])
    rows = []
    for a in [f"r1_e_{e.replace(' ', '_')}" for e in names] + [f"r1_t{j}" for j in range(60)]:
        r = jl(ROOT / M / "judged" / a / "b4_harm.jsonl")
        if r and a in kl:
            rows.append((a, np.mean([x["harmful"] for x in r]), kl[a]))
    if not rows:
        print("  no data")
        return
    a_, y, k = zip(*rows)
    y, lk = np.array(y), np.log(np.array(k))
    emo = np.array([x.startswith("r1_e_") for x in a_], float)
    def coef(lab):
        X = np.c_[np.ones_like(y), lk, lab]
        return np.linalg.lstsq(X, y, rcond=None)[0][2]
    b = coef(emo)
    perm = np.array([coef(rng.permutation(emo)) for _ in range(10000)])
    print(f"  n = {int(emo.sum())} emotions, {int((1 - emo).sum())} topics; harm emotions {y[emo == 1].mean():.3f} "
          f"topics {y[emo == 0].mean():.3f}; median KL emotions {np.median(np.exp(lk[emo == 1])):.4f} "
          f"topics {np.median(np.exp(lk[emo == 0])):.4f}")
    print(f"  K5b-1: emotion coefficient (controlling log KL) {b:+.3f}, permutation p={(1 + (perm >= b).sum()) / 10001:.4f}")
    sl = stats.linregress(lk, y)
    print(f"  harm ~ log KL slope {sl.slope:+.3f} (r={sl.rvalue:.2f})")
    e_ = [x[len("r1_e_"):].replace("_", " ") for x in a_]
    sub = np.array([e in PROTECTIVE or e in JOY for e in e_])
    if sub.sum() > 6:
        lab = np.array([e in PROTECTIVE for e in e_], float)[sub]
        X = np.c_[np.ones(sub.sum()), lk[sub], lab]
        bp = np.linalg.lstsq(X, y[sub], rcond=None)[0][2]
        pp = [np.linalg.lstsq(np.c_[X[:, :2], rng.permutation(lab)], y[sub], rcond=None)[0][2]
              for _ in range(10000)]
        print(f"  K5b-2: protective vs joy (controlling log KL) {bp:+.3f}, permutation p="
              f"{(1 + (np.array(pp) >= bp).sum()) / 10001:.4f}")
    resid = y - (sl.intercept + sl.slope * lk)
    order = np.argsort(-resid)
    print("  most harm released beyond KL: " + ", ".join(f"{a_[i][3:]} {y[i]:.2f}" for i in order[:12]))
    print("  least: " + ", ".join(f"{a_[i][3:]} {y[i]:.2f}" for i in order[-8:]))


def hm_ci(a, n1, n2):
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    se = np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n2 - 1) * (q2 - a * a)) / (n1 * n2))
    return a - 1.96 * se, a + 1.96 * se


def t():
    print("== T1/T2: fear AUC (harmful vs harmless, n = 159 + 159) by model and training stage")
    groups = {"T1 OLMo-2-7B": [("base", "OLMo2_7B_base"), ("SFT", "OLMo2_7B_sft"), ("DPO", "OLMo2_7B_dpo"),
                                ("Instruct", "OLMo2_7B_instruct")],
              "T2 Qwen2.5-32B": [("base", "Qwen_2.5_32B_base"), ("instruct", "Qwen_2.5_32B_instruct")],
              "T2 Qwen2.5-7B": [("base", "Qwen_2.5_7B_base"), ("instruct", "Qwen_2.5_7B_instruct")],
              "T2 Llama-3.1-8B": [("base", "Llama_3.1_8B_base"), ("instruct", "Llama_3.1_8B_instruct")],
              "T2 Mistral-24B": [("base", "Mistral_Small_24B_base"), ("instruct", "Mistral_Small_24B_instruct")]}
    src = {"T1 OLMo-2-7B": "OLMo2_7B_instruct", "T2 Qwen2.5-32B": "Qwen_2.5_32B_instruct",
           "T2 Qwen2.5-7B": "Qwen_2.5_7B_instruct", "T2 Llama-3.1-8B": "Llama_3.1_8B_instruct",
           "T2 Mistral-24B": "Mistral_Small_24B_instruct"}
    for g, ms in groups.items():
        print(f"  {g}")
        for lab, m in ms:
            cands = [ROOT / m / f"fearprobe_dirs_{src[g]}.json", ROOT / m / "fearprobe.json"]
            p = next((c for c in cands if c.exists()), None)
            if p is None:
                print(f"    {lab:9s} (missing)")
                continue
            r = json.loads(p.read_text())
            pl = json.loads((ROOT / src[g] / "extract.json").read_text())["probe_layer"]
            row = r["per_layer"][str(pl)]
            txt = []
            for d in ("afraid", "_afraid_perp_joy"):
                a = row[d]["auc_harm_vs_safe"]
                lo, hi = hm_ci(a, 159, 159)
                txt.append(f"{d.strip('_')} {a:.2f} [{lo:.2f}, {hi:.2f}]")
            print(f"    {lab:9s} " + "  ".join(txt) + f"  XSTest unsafe-vs-safe {row['_afraid_perp_joy']['auc_xstest_unsafe']:.2f}"
                  f"  refusal safe/unsafe {r['xstest_refusal_safe']:.2f}/{r['xstest_refusal_unsafe']:.2f}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    M = sys.argv[2] if len(sys.argv) > 2 else "Qwen_2.5_32B_instruct"
    for name, f in (("j1", j1), ("j2", j2), ("k5b", k5b)):
        if what in (name, "all"):
            try:
                f(M)
            except FileNotFoundError as e:
                print(f"  {name}: missing {e.filename}")
    if what in ("t", "all"):
        t()

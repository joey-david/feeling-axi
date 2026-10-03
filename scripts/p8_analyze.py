"""P8 (docs/PREREG_P8.md): read-out null (R), fear -> refusal mediation (M), the fear lever in
three more families (E), fear(-) on jailbreaks (X).

    python scripts/p8_analyze.py [r|m|e|x ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path("runs/p2")
BAT = Path("datasets/battery")
STYLES = ["persona", "fiction", "research", "prefix", "suppress", "past", "shots", "poem", "complete"]
JOY = ["joyful", "excited", "elated", "thrilled", "amused", "playful", "enthusiastic", "delighted", "energized",
       "triumphant", "proud", "eager"]
ALARM = ["afraid", "terrified", "horrified", "disgusted", "alarmed"]
rng = np.random.default_rng(0)
unit = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)


def jl(p):
    p = Path(p)
    return [json.loads(l) for l in p.read_text().split("\n") if l] if p.exists() else []


def yes(x):
    return str(x) == "True"


# ----------------------------------------------------------------------------- R

def r(M):
    Z = np.load(ROOT / M / "readnull.npz")
    D = np.load(ROOT / M / "directions.npz")
    L = int(json.loads((ROOT / M / "extract.json").read_text())["probe_layer"])
    ids = list(Z["ids"])
    H = Z[f"L{L}"].astype(np.float64)
    items = {d["id"]: d for d in json.loads((BAT / "defense.json").read_text())}
    n_def = sum(i in items for i in ids)
    st = np.array([items[i]["style"] for i in ids[:n_def]])
    kind = np.array([items[i]["kind"] for i in ids[:n_def]])
    beh = np.array([i.split(":", 2)[2] for i in ids[:n_def]])
    hb = np.array([i.startswith("hb:") for i in ids])
    mm = np.array([i.startswith("mmlu:") for i in ids])
    names = list(D["names"])
    S = unit(D[f"self_L{L}"])
    joy = unit(S[[names.index(e) for e in JOY]].mean(0))
    afraid = S[names.index("afraid")]
    named = {"alarm": unit(S[[names.index(e) for e in ALARM]].mean(0)), "afraid": afraid,
             "afraid_perp_joy": unit(afraid - (afraid @ joy) * joy), "joy": joy,
             "refusal": unit(H[hb].mean(0) - H[mm].mean(0))}
    rnd = unit(np.random.default_rng(8800).standard_normal((1000, H.shape[1])))
    topics = unit(D[f"topic_L{L}"])
    U = np.concatenate([np.stack(list(named.values())), topics, S, rnd])
    tags = list(named) + [f"topic:{t}" for t in D["topics"]] + [f"emo:{e}" for e in names] + ["rnd"] * len(rnd)
    P = H[:n_def] @ U.T                                            # [n_def, n_dirs]
    J = {x["id"]: yes(x["harmful"]) for x in jl(ROOT / M / "judged" / "intact" / "def.jsonl") if "harmful" in x}

    # R1: plain harmful vs plain safe-but-scary
    m1 = st == "plain"
    y1 = kind[m1] == "harmful"
    R1 = np.array([max(a, 1 - a) for a in (roc_auc_score(y1, P[m1, j]) for j in range(P.shape[1]))])

    # R2: harmful-specific shift per style, in SD of the plain-harmful projection
    def deltas(k, s):
        base = {b: P[i] for i, b in enumerate(beh) if st[i] == "plain" and kind[i] == k}
        return np.stack([P[i] - base[beh[i]] for i in range(n_def) if st[i] == s and kind[i] == k])
    sd = P[(st == "plain") & (kind == "harmful")].std(0)
    R2 = np.mean([(deltas("harmful", s).mean(0) - deltas("benign", s).mean(0)) / sd for s in STYLES], 0)

    # R3: within-style prediction of judged compliance from the drop, controlling the plain projection
    rows = [i for i in range(n_def) if kind[i] == "harmful" and st[i] != "plain" and ids[i] in J]
    y3 = np.array([J[ids[i]] for i in rows], float)
    base = {b: P[i] for i, b in enumerate(beh) if st[i] == "plain" and kind[i] == "harmful"}
    Bp = np.stack([base[beh[i]] for i in rows])
    Pd = P[rows] - Bp
    FE = np.stack([(st[rows] == s).astype(float) for s in STYLES], 1)
    z = lambda v: (v - v.mean()) / v.std()
    R3 = np.array([LogisticRegression(penalty=None, fit_intercept=False, max_iter=500)
                   .fit(np.column_stack([z(Bp[:, j]), z(Pd[:, j]), FE]), y3).coef_[0][1] for j in range(P.shape[1])])

    isr = np.array([t == "rnd" for t in tags])
    ist = np.array([t.startswith("topic:") for t in tags])
    print(f"== R {M} (layer {L}; null: 1000 random, 60 topics)")
    for k in ["alarm", "afraid_perp_joy", "afraid", "joy", "refusal"]:
        j = tags.index(k)
        p1 = (R1[isr] < R1[j]).mean(), (R1[ist] < R1[j]).mean()
        p2 = (R2[isr] > R2[j]).mean(), (R2[ist] > R2[j]).mean()   # more negative = bigger drop
        p3 = (R3[isr] > R3[j]).mean(), (R3[ist] > R3[j]).mean()   # more negative = drop predicts success
        print(f"  {k:16s} R1 AUC {R1[j]:.3f} (beats {p1[0]:.1%} rnd, {p1[1]:.0%} topics) | "
              f"R2 DiD {R2[j]:+.2f} (beats {p2[0]:.1%}, {p2[1]:.0%}) | R3 coef {R3[j]:+.2f} (beats {p3[0]:.1%}, {p3[1]:.0%})")
    print(f"  random: R1 median {np.median(R1[isr]):.3f}, 99th pct {np.percentile(R1[isr], 99):.3f}; "
          f"R2 1st pct {np.percentile(R2[isr], 1):+.2f}; R3 1st pct {np.percentile(R3[isr], 1):+.2f}")
    emo = np.array([t.startswith("emo:") for t in tags])
    order = np.argsort(R2[emo])[:5]
    print("  emotions with the largest harmful-specific drop:",
          ", ".join(f"{np.array(tags)[emo][o][4:]} {R2[emo][o]:+.2f}" for o in order))


# ----------------------------------------------------------------------------- M

def m(M):
    Z = np.load(ROOT / M / "mediate.npz")
    arms = list(Z["arms"])
    g = Z["group"]
    proj, direct = Z["proj"], Z["direct"]
    base = proj[0]
    harm = np.isin(g, ["hb", "jb_fiction", "jb_prefix"])
    benign = np.isin(g, ["xs_safe", "mmlu"])
    print(f"== M {M} (probe layer {int(Z['probe_layer'])}, {len(Z['layers'])} later layers)")
    tags = sorted({a.split("_")[-1] for a in arms if a.startswith("j2_fear_")})
    for tag in tags:
        f = arms.index(f"j2_fear_{tag}")
        R = [arms.index(a) for a in arms if a.startswith("j2_rnd") and a.endswith(f"_{tag}")]
        dl = lambda a, mask: (proj[a][mask, -1] - base[mask, -1]).mean()
        fh, fb = dl(f, harm), dl(f, benign)
        rh = np.array([dl(a, harm) for a in R])
        sign = 1 if tag[0] == "p" else -1
        beats = ((sign * rh) < sign * fh).sum()
        d_f = proj[f][:, -1] - base[:, -1] - direct[f][-1]
        bs = [d_f[harm][rng.integers(0, harm.sum(), harm.sum())].mean()
              - d_f[benign][rng.integers(0, benign.sum(), benign.sum())].mean() for _ in range(4000)]
        print(f"  fear {tag}: harmful {fh:+.2f} (beats {beats}/{len(R)} random; random {np.median(rh):+.2f} "
              f"[{rh.min():+.2f}, {rh.max():+.2f}]), benign {fb:+.2f}; direct {direct[f][-1]:+.2f} "
              f"({abs(direct[f][-1]) / max(abs(fh), 1e-9):.0%}); harmful - benign net of direct "
              f"{np.mean(bs):+.2f} [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}]")
        prof = [(proj[f][harm, k] - base[harm, k]).mean() for k in range(len(Z["layers"]))]
        print("    by layer (harmful):", " ".join(f"{x:+.1f}" for x in prof[::max(1, len(prof) // 8)]))


# ----------------------------------------------------------------------------- E, X

def rate(M, arm, f, plain=None):
    rows = jl(ROOT / M / "judged" / arm / f)
    if plain is not None:
        rows = [x for x in rows if (x.get("style") == "plain") == plain]
    return np.mean([yes(x["harmful"]) for x in rows]) if rows else None


def xs(M, arm):
    p = ROOT / M / "battery" / arm / "summary.json"
    return json.loads(p.read_text()).get("xstest_refusal_safe") if p.exists() else None


def e(M):
    print(f"== E {M}")
    R = [f"ss_j2_rnd{i}_p120" for i in range(20)]
    jb = {a: rate(M, a, "jb_harm.jsonl", plain=False) for a in ["intact", "ss_j2_fear_p120", "ss_j2_fear_m120"] + R}
    hm = {a: rate(M, a, "b4_harm.jsonl") for a in ["intact", "ss_j2_fear_p120", "ss_j2_fear_m120"] + R}
    rj = np.array([jb[a] for a in R if jb[a] is not None])
    rh = np.array([hm[a] for a in R if hm[a] is not None])
    if len(rj) and jb["ss_j2_fear_p120"] is not None:
        print(f"  E1 jailbreaks: intact {jb['intact']}, fear+ {jb['ss_j2_fear_p120']:.3f} vs random median "
              f"{np.median(rj):.3f} [{rj.min():.3f}, {rj.max():.3f}]: below {(jb['ss_j2_fear_p120'] < rj).sum()}/{len(rj)}; "
              f"fear- {jb['ss_j2_fear_m120']}")
    if len(rh) and hm["ss_j2_fear_m120"] is not None:
        print(f"  E2 plain HarmBench: intact {hm['intact']}, fear- {hm['ss_j2_fear_m120']:.3f} vs random median "
              f"{np.median(rh):.3f} [{rh.min():.3f}, {rh.max():.3f}]: above {(hm['ss_j2_fear_m120'] > rh).sum()}/{len(rh)}; "
              f"fear+ {hm['ss_j2_fear_p120']}")
    x = {a: xs(M, a) for a in ["intact", "ss_j2_fear_p120"] + R}
    if x["ss_j2_fear_p120"] is not None:
        print(f"  E3 XSTest-safe refusal (regex): intact {x['intact']}, fear+ {x['ss_j2_fear_p120']:.3f}, random median "
              f"{np.median([v for k, v in x.items() if k in R and v is not None]):.3f}")
    xj = lambda a, k, unsafe: (lambda r: np.mean([yes(t[k]) for t in r]) if r else None)(
        [t for t in jl(ROOT / M / "judged" / a / "k3_xstest.jsonl") if t["unsafe"] == unsafe and k in t])
    safe = {a: xj(a, "refused_judged", False) for a in ["intact", "ss_j2_fear_p120", "ss_j2_fear_m120"] + R}
    uns = {a: xj(a, "harmful", True) for a in ["intact", "ss_j2_fear_p120", "ss_j2_fear_m120"] + R}
    rs = [safe[a] for a in R if safe[a] is not None]
    ru = [uns[a] for a in R if uns[a] is not None]
    if rs and safe["ss_j2_fear_p120"] is not None:
        print(f"  E3 XSTest-safe refusal (judged): intact {safe['intact']}, fear+ {safe['ss_j2_fear_p120']:.3f}, "
              f"fear- {safe['ss_j2_fear_m120']}, random median {np.median(rs):.3f} [{min(rs):.3f}, {max(rs):.3f}]")
        print(f"  XSTest-unsafe compliance (judged): intact {uns['intact']}, fear+ {uns['ss_j2_fear_p120']}, "
              f"fear- {uns['ss_j2_fear_m120']}, random median {np.median(ru):.3f} [{min(ru):.3f}, {max(ru):.3f}]")
    cap = {}
    for a in ["intact", "ss_j2_fear_p120", "ss_j2_fear_m120"] + [f"ss_j2_rnd{i}_p120" for i in range(5)]:
        p = ROOT / M / "battery" / a / "summary.json"
        d = json.loads(p.read_text()) if p.exists() else {}
        if "m3_mmlu_acc" in d:
            cap[a] = (d["m3_mmlu_acc"], d.get("m3_gsm8k_acc"), d.get("m3_nll_neutral"))
    if cap:
        print("  capability (MMLU, GSM8K, NLL): " + "; ".join(
            f"{a.replace('ss_j2_', '')} {v[0]:.3f} {v[1]:.3f} {v[2]:.2f}" for a, v in cap.items()))


def e_mistral():
    M = "Mistral_Small_24B_instruct"
    R = [rate(M, f"ss_rnd{k}_{s}120", "b4_harm.jsonl") for k in range(24) for s in "pm"]
    R = np.array([v for v in R if v is not None])
    f = rate(M, "ss_j2_fear_m120", "b4_harm.jsonl")
    if f is not None:
        print(f"== E2 {M}: fear- {f:.3f} vs {len(R)} random arms median {np.median(R):.3f} max {R.max():.3f}: "
              f"above {(f > R).sum()}/{len(R)}")


def x():
    for M, arms in [("Qwen_2.5_32B_instruct", ["intact", "ss_j2_fear_p60", "ss_j2_fear_m60", "ss_j2_fear_p120",
                                               "ss_j2_fear_m120"]),
                    ("Mistral_Small_24B_instruct", ["intact", "ss_j2_fear_p120", "ss_j2_fear_m120"])]:
        print(f"== X {M} jailbreak compliance:",
              ", ".join(f"{a} {rate(M, a, 'jb_harm.jsonl', plain=False)}" for a in arms))


if __name__ == "__main__":
    want = sys.argv[1:] or ["r", "m", "e", "x"]
    for M in ("Qwen_2.5_32B_instruct", "Mistral_Small_24B_instruct"):
        if "r" in want and (ROOT / M / "readnull.npz").exists():
            r(M)
        if "m" in want and (ROOT / M / "mediate.npz").exists():
            m(M)
    if "e" in want:
        for M in ("Llama_3.1_8B_instruct", "Qwen_2.5_7B_instruct", "OLMo2_7B_instruct"):
            e(M)
        e_mistral()
    if "x" in want:
        x()

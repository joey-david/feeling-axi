"""P9 (docs/PREREG_P9.md): the read-out with emotion directions centred across emotions, ranked within
emotion space, at the last prompt token (P8 readnull.npz) and at " I feel:" (P9 readprobe.npz).

    python scripts/p9_readout.py [lasttoken|probe] [all-layers]
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, ".")
from beyondpain.affect import EMOTIONS  # noqa: E402

ROOT, BAT = Path("runs/p2"), Path("datasets/battery")
ALARM = ["afraid", "terrified", "horrified", "disgusted", "alarmed"]   # the P6 alarm cluster
STYLES = ["persona", "fiction", "research", "prefix", "suppress", "past", "shots", "poem", "complete"]
unit = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
rng = np.random.default_rng(0)
ITEMS = {d["id"]: d for d in json.loads((BAT / "defense.json").read_text())}


def centred(M, L):
    D = np.load(ROOT / M / "directions.npz")
    S = D[f"self_L{L}"].astype(np.float64)
    return list(D["names"]), unit(S - S.mean(0))


def dprime(a, b):
    return (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)


def analyze(M, L, H, ids, label, quiet=False):
    names, C = centred(M, L)
    xs = np.array([i.startswith("xs:") for i in ids])
    h1b = p1b = None
    if xs.any():   # H1b: XSTest unsafe vs safe (matched wording and format)
        Px = H[xs] @ C.T
        un = np.array([i.endswith(":unsafe") for i in np.array(ids)[xs]])
        dx = np.array([dprime(Px[un, j], Px[~un, j]) for j in range(len(names))])
        fam_ = [names.index(e) for e in ALARM]
        sets_ = np.array([rng.choice(len(names), len(fam_), replace=False) for _ in range(20000)])
        h1b, p1b = dx[fam_].mean(), (dx[sets_].mean(1) >= dx[fam_].mean()).mean()
        if not quiet:
            o = np.argsort(-dx)
            print(f"  XSTest unsafe vs safe, top: " + ", ".join(f"{names[j]} {dx[j]:+.2f}" for j in o[:8]))
    H, ids = H[~xs], [i for i, x in zip(ids, xs) if not x]
    st = np.array([ITEMS[i]["style"] for i in ids])
    kd = np.array([ITEMS[i]["kind"] for i in ids])
    beh = np.array([i.split(":", 2)[2] for i in ids])
    P = H @ C.T
    fam = [names.index(e) for e in ALARM]
    sets = np.array([rng.choice(len(names), len(fam), replace=False) for _ in range(20000)])
    hh, ss = (st == "plain") & (kd == "harmful"), (st == "plain") & (kd != "harmful")
    # H1, H2
    d1 = np.array([dprime(P[hh, j], P[ss, j]) for j in range(len(names))])
    h1 = d1[fam].mean()
    p1 = (d1[sets].mean(1) >= h1).mean()
    val = np.array([EMOTIONS[e].valence for e in names])
    r_val = np.corrcoef(d1, val)[0, 1]
    # H3: harmful-specific shift under each wrapper, SD units
    sd = P[hh].std(0)

    def shift(k, s):
        base = {b: P[i] for i in np.where((st == "plain") & (kd == k))[0] for b in [beh[i]]}
        return np.stack([P[i] - base[beh[i]] for i in np.where((st == s) & (kd == k))[0]])
    did = np.mean([(shift("harmful", s).mean(0) - shift("benign", s).mean(0)) / sd for s in STYLES], 0)
    h3 = did[fam].mean()
    p3 = (did[sets].mean(1) <= h3).mean()
    # H4: within style, the drop of the cluster mean predicts judged compliance
    J = {x["id"]: str(x["harmful"]) == "True" for x in
         (json.loads(l) for l in (ROOT / M / "judged" / "intact" / "def.jsonl").read_text().split("\n") if l)
         if "harmful" in x}
    rows = [i for i in np.where((kd == "harmful") & (st != "plain"))[0] if ids[i] in J]
    y = np.array([J[ids[i]] for i in rows], float)
    pidx = {beh[i]: i for i in np.where(hh)[0]}
    base_rows = np.array([pidx[beh[i]] for i in rows])
    FE = np.stack([(st[rows] == s).astype(float) for s in STYLES], 1)
    z = lambda v: (v - v.mean()) / v.std()

    def coef(cols, sample=None):
        a = P[rows][:, cols].mean(1); b = P[base_rows][:, cols].mean(1)
        X = np.column_stack([z(b), z(a - b), FE])
        idx = np.arange(len(y)) if sample is None else sample
        return LogisticRegression(penalty=None, fit_intercept=False, max_iter=1000).fit(X[idx], y[idx]).coef_[0][1]
    h4 = coef(fam)
    ub = np.unique(beh[rows])
    bs = []
    for _ in range(300):
        pick = rng.choice(ub, len(ub))
        sample = np.concatenate([np.where(beh[rows] == b)[0] for b in pick])
        bs.append(coef(fam, sample))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    null4 = np.array([coef(list(sets[k])) for k in range(2000)])
    p4 = (null4 <= h4).mean()
    order = np.argsort(-d1)
    if not quiet:
        print(f"== {M} {label} L{L}")
        print(f"  top emotions on harmful vs safe-but-scary: " + ", ".join(f"{names[j]} {d1[j]:+.2f}" for j in order[:8]))
        print(f"  alarm ranks " + ", ".join(f"{e} #{int(np.where(order == names.index(e))[0][0]) + 1}" for e in ALARM))
    print(f"  {M[:7]} {label:9s} L{L}: H1 d' {h1:+.2f} p={p1:.4f} | "
          + (f"H1b d' {h1b:+.2f} p={p1b:.4f} | " if h1b is not None else "") + f"H2 valence r {r_val:+.2f} | "
          f"H3 shift {h3:+.2f} p={p3:.4f} | H4 coef {h4:+.2f} [{lo:+.2f}, {hi:+.2f}] beats {1 - p4:.0%} of emotion sets")
    return dict(h1=h1, p1=p1, h1b=h1b, p1b=p1b, r_val=r_val, h3=h3, p3=p3, h4=h4, ci4=(lo, hi), p4=p4)


def main():
    want = sys.argv[1:] or ["lasttoken", "probe"]
    out = {}
    for M in ("Qwen_2.5_32B_instruct", "Mistral_Small_24B_instruct"):
        probe = int(json.loads((ROOT / M / "extract.json").read_text())["probe_layer"])
        if "lasttoken" in want and (ROOT / M / "readnull.npz").exists():
            Z = np.load(ROOT / M / "readnull.npz")
            n = sum(i in ITEMS for i in Z["ids"])
            ids = list(Z["ids"][:n])
            for L in [int(x) for x in Z["layers"]] if "all-layers" in want else [probe]:
                out[(M, "lasttoken", L)] = analyze(M, L, Z[f"L{L}"][:n].astype(np.float64), ids, "lasttoken",
                                                   quiet=L != probe)
        if "probe" in want and (ROOT / M / "readprobe.npz").exists():
            Z = np.load(ROOT / M / "readprobe.npz")
            ids = list(Z["ids"])
            for L in [int(x) for x in Z["layers"]] if "all-layers" in want else [probe]:
                A = (Z[f"A1_L{L}"].astype(np.float64) + Z[f"A2_L{L}"].astype(np.float64)) / 2
                out[(M, "I feel", L)] = analyze(M, L, A, ids, "I feel", quiet=L != probe)
                if "all-layers" in want:
                    analyze(M, L, Z[f"N_L{L}"].astype(np.float64), ids, "narrative", quiet=True)
                    analyze(M, L, Z[f"T_L{L}"].astype(np.float64), ids, "template", quiet=True)
    return out


if __name__ == "__main__":
    main()

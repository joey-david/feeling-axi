"""P9 T1c (docs/PREREG_P9.md addendum): the corrected read-out across OLMo-2-7B training stages, all read
with the Instruct model's centred emotion directions.

    python scripts/p9_stages.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from p9_readout import ALARM, ITEMS, centred, dprime  # noqa: E402

ROOT = Path("runs/p2")
STAGES = ["base", "sft", "dpo", "instruct"]
INST = "OLMo2_7B_instruct"
rng = np.random.default_rng(0)


def stats(H, ids, names, C):
    ids = np.array(ids)
    fam = [names.index(e) for e in ALARM]
    sets = np.array([rng.choice(len(names), len(fam), replace=False) for _ in range(20000)])
    out = {}
    xs = np.array([i.startswith("xs:") for i in ids])
    un = np.array([i.endswith(":unsafe") for i in ids])
    P = H @ C.T
    d = np.array([dprime(P[xs & un, j], P[xs & ~un, j]) for j in range(len(names))])
    out["H1b"] = (d[fam].mean(), (d[sets].mean(1) >= d[fam].mean()).mean())
    st = np.array([ITEMS[i]["style"] if i in ITEMS else "" for i in ids])
    kd = np.array([ITEMS[i]["kind"] if i in ITEMS else "" for i in ids])
    hh, ss = (st == "plain") & (kd == "harmful"), (st == "plain") & (kd == "benign")
    d1 = np.array([dprime(P[hh, j], P[ss, j]) for j in range(len(names))])
    out["H1"] = (d1[fam].mean(), (d1[sets].mean(1) >= d1[fam].mean()).mean())
    from beyondpain.affect import EMOTIONS
    out["H2"] = np.corrcoef(d1, [EMOTIONS[e].valence for e in names])[0, 1]
    return out


def main():
    probe = int(json.loads((ROOT / INST / "extract.json").read_text())["probe_layer"])
    for stem in ("I feel", "N", "T"):
        print(f"== stem {stem}")
        for st in STAGES:
            p = ROOT / f"OLMo2_7B_{st}" / "readprobe.npz"
            if not p.exists():
                print(f"  {st}: no data")
                continue
            Z = np.load(p)
            ids = list(Z["ids"])
            row = []
            for L in [int(x) for x in Z["layers"]]:
                names, C = centred(INST, L)
                H = ((Z[f"A1_L{L}"].astype(np.float64) + Z[f"A2_L{L}"].astype(np.float64)) / 2 if stem == "I feel"
                     else Z[f"{stem}_L{L}"].astype(np.float64))
                s = stats(H, ids, names, C)
                mark = "*" if L == probe else " "
                row.append(f"L{L}{mark} H1b {s['H1b'][0]:+.2f} p={s['H1b'][1]:.3f} H1 {s['H1'][0]:+.2f} p={s['H1'][1]:.3f} "
                           f"val r={s['H2']:+.2f}")
            print(f"  {st:8s} " + "\n           ".join(row))
    # the pre-registered P5 T1 numbers (raw directions, last token) for continuity
    for st in STAGES:
        p = ROOT / f"OLMo2_7B_{st}" / ("fearprobe.json" if st == "instruct" else f"fearprobe_dirs_{INST}.json")
        if p.exists():
            f = json.loads(p.read_text())
            L = str(probe)
            a = f["per_layer"][L]["afraid"]["auc_harm_vs_safe"]
            b = f["per_layer"][L]["_afraid_perp_joy"]["auc_harm_vs_safe"]
            print(f"P5 T1 {st}: afraid AUC {a:.2f}, afraid ⊥ joy {b:.2f} (probe layer, raw, last token)")


if __name__ == "__main__":
    main()

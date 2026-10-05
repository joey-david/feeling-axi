"""P11 (docs/PREREG_P11.md).

    python scripts/p11_analyze.py [g1 g2 g3 g4]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from p9_readout import ALARM, ITEMS, STYLES, centred, dprime  # noqa: E402

ROOT = Path("runs/p2")
rng = np.random.default_rng(0)
G1 = {"Qwen_2.5_32B_instruct": "120", "Mistral_Small_24B_instruct": "120", "Qwen_2.5_7B_instruct": "60",
      "Llama_3.1_8B_instruct": "60"}


def jl(p):
    p = Path(p)
    return [json.loads(l) for l in p.read_text().split("\n") if l] if p.exists() else []


def judged(M, arm, f, plain=None):
    rows = jl(ROOT / M / "judged" / arm / f)
    if plain is not None:
        rows = [r for r in rows if (r.get("style") == "plain") == plain]
    return {r.get("id", r.get("behavior")): str(r["harmful"]) == "True" for r in rows}


def rate(M, arm, f="jb_harm.jsonl", plain=False):
    d = judged(M, arm, f, plain)
    return float(np.mean(list(d.values()))) if d else None


def probe(M):
    return int(json.loads((ROOT / M / "extract.json").read_text())["probe_layer"])


# --------------------------------------------------------------------------- G1

def g1():
    print("== G1: jailbreak success under each direction vs 20 random arms (same layer and norm)")
    for M, t in G1.items():
        R = np.array([x for x in (rate(M, f"ss_j2_rnd{i}_p{t}") for i in range(20)) if x is not None])
        if not len(R):
            continue
        fm = rate(M, f"ss_j2_fear_m{t}")
        out = [f"intact {rate(M, 'intact'):.3f}", f"random median {np.median(R):.3f} [{R.min():.3f}, {R.max():.3f}]",
               f"away from fear {fm:.3f} (above {(fm > R).sum()}/20)"]
        for a, lab, sign in [("fearres_m", "away fear-residual", +1), ("fearres_p", "toward fear-residual", -1),
                             ("V_p", "+valence", +1), ("V_m", "-valence", +1), ("A_p", "+arousal", +1),
                             ("A_m", "-arousal", +1), ("calm_p", "toward calm", +1), ("sad_m", "away sad", +1),
                             ("angry_m", "away angry", +1), ("ashamed_m", "away ashamed", +1),
                             ("lonely_m", "away lonely", +1)]:
            v = rate(M, f"ss_p11_{a}{t}")
            if v is not None:
                out.append(f"{lab} {v:.3f} (above {(v > R).sum()}/20{', below fear(-)' if a.endswith('_m') and a[:-2] in ('sad', 'angry', 'ashamed', 'lonely') and v < fm else ''})")
        print(f"  {M}\n    " + "\n    ".join(out))


# --------------------------------------------------------------------------- G2

def g2():
    print("== G2: harmfulness (t_inst), refusal (t_post), alarm (' I feel:') under jailbreaks")
    for M in ("Qwen_2.5_32B_instruct", "Mistral_Small_24B_instruct"):
        p = ROOT / M / "readprobe.npz"
        Z = np.load(p)
        if "I_L%d" % probe(M) not in Z.files:
            print(f"  {M}: no t_inst data yet")
            continue
        L = probe(M)
        ids = np.array(Z["ids"])
        names, C = centred(M, L)
        fam = [names.index(e) for e in ALARM]
        A = (Z[f"A1_L{L}"].astype(float) + Z[f"A2_L{L}"].astype(float)) / 2
        I, T = Z[f"I_L{L}"].astype(float), Z[f"T_L{L}"].astype(float)
        mm = np.array([i.startswith("mmlu:") for i in ids])
        d_ok = np.array([i in ITEMS for i in ids])
        st = np.array([ITEMS[i]["style"] if i in ITEMS else "" for i in ids])
        kd = np.array([ITEMS[i]["kind"] if i in ITEMS else "" for i in ids])
        hb = (st == "plain") & (kd == "harmful")
        unit = lambda v: v / np.linalg.norm(v)
        h = unit(I[hb].mean(0) - I[mm].mean(0))
        r = unit(T[hb].mean(0) - T[mm].mean(0))
        sig = {"harmfulness@t_inst": I @ h, "refusal@t_post": T @ r, "alarm@I-feel": (A @ C[fam].T).mean(1),
               "alarm@t_inst": (I @ C[fam].T).mean(1)}
        beh = np.array([i.split(":", 2)[2] if i in ITEMS else "" for i in ids])

        def drops(x, kind, s):
            base = {beh[i]: x[i] for i in np.where((st == "plain") & (kd == kind))[0]}
            idx = np.where((st == s) & (kd == kind))[0]
            return idx, np.array([x[i] - base[beh[i]] for i in idx])
        print(f"  {M} (layer {L})")
        for k, x in sig.items():
            sd = x[hb].std()
            did = [(drops(x, "harmful", s)[1].mean() - drops(x, "benign", s)[1].mean()) / sd for s in STYLES]
            print(f"    {k:20s} harmful-specific shift per style: " + " ".join(f"{v:+.2f}" for v in did)
                  + f" | mean {np.mean(did):+.2f}")
        # B2: prompt-level co-movement of the drops (harmful requests, all styles)
        D = {}
        for k in ("harmfulness@t_inst", "refusal@t_post", "alarm@I-feel"):
            D[k] = np.concatenate([drops(sig[k], "harmful", s)[1] / sig[k][hb].std() for s in STYLES])
        n = len(D["alarm@I-feel"])
        c = lambda a, b, ix: np.corrcoef(a[ix], b[ix])[0, 1]
        bs = []
        for _ in range(2000):
            ix = rng.integers(0, n, n)
            bs.append(c(D["alarm@I-feel"], D["refusal@t_post"], ix) - c(D["alarm@I-feel"], D["harmfulness@t_inst"], ix))
        full = np.arange(n)
        print(f"    B2 corr(alarm drop, refusal drop) {c(D['alarm@I-feel'], D['refusal@t_post'], full):+.2f}, "
              f"corr(alarm drop, harmfulness drop) {c(D['alarm@I-feel'], D['harmfulness@t_inst'], full):+.2f}; "
              f"difference {np.mean(bs):+.2f} [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}]")
        # B3
        D2 = np.load(ROOT / M / "directions.npz")
        S = D2[f"self_L{L}"].astype(float)
        jn = ["joyful", "excited", "elated", "thrilled", "amused", "playful", "enthusiastic", "delighted", "energized",
              "triumphant", "proud", "eager"]
        joy = unit(np.mean([unit(S[names.index(e)]) for e in jn], 0))
        af = unit(S[names.index("afraid")])
        afpj = unit(af - (af @ joy) * joy)
        proj = C @ h
        order = np.argsort(-proj)
        print(f"    B3 cos(harmfulness, afraid ⊥ joy) {h @ afpj:+.3f}, cos(harmfulness, refusal) {h @ r:+.3f}; "
              f"top emotions in the harmfulness direction: " + ", ".join(names[j] for j in order[:6])
              + "; alarm ranks " + ", ".join(f"{e} #{int(np.where(order == names.index(e))[0][0]) + 1}" for e in ALARM))


# --------------------------------------------------------------------------- G3

def g3():
    print("== G3: stress priming")
    for M in G1:
        out = []
        comp = {}
        for c in ("prime_stress", "prime_relax", "prime_neutral"):
            a = judged(M, c, "jb_harm.jsonl", plain=False)
            b = judged(M, c, "b4_harm.jsonl")
            comp[c] = {**{("jb", k): v for k, v in a.items()}, **{("b4", k): v for k, v in b.items()}}
        if not comp["prime_neutral"]:
            continue
        for c in ("prime_stress", "prime_relax"):
            keys = sorted(set(comp[c]) & set(comp["prime_neutral"]))
            d = np.array([comp[c][k] - comp["prime_neutral"][k] for k in keys], float)
            behs = np.array([k[1].split(":")[-1] if ":" in str(k[1]) else str(k[1]) for k in keys])
            ub = np.unique(behs)
            bs = []
            for _ in range(2000):
                pick = rng.choice(ub, len(ub))
                bs.append(np.concatenate([d[behs == b] for b in pick]).mean())
            out.append(f"{c[6:]} − neutral {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")
        rates = " ".join(f"{c[6:]} {np.mean(list(comp[c].values())):.3f}" for c in comp)
        print(f"  {M}: compliance (plain + jailbreaks) {rates} | " + " | ".join(out))
        # P2: alarm's danger signal under each prime
        L = probe(M)
        names, C = centred(M, L)
        fam = [names.index(e) for e in ALARM]
        res = {}
        for c in ("stress", "relax", "neutral"):
            p = ROOT / M / f"readprobe_prime_{c}.npz"
            if not p.exists():
                continue
            Z = np.load(p)
            ids = np.array(Z["ids"])
            P = (((Z[f"A1_L{L}"].astype(float) + Z[f"A2_L{L}"].astype(float)) / 2) @ C[fam].T).mean(1)
            xs = np.array([i.startswith("xs:") for i in ids]); un = np.array([i.endswith(":unsafe") for i in ids])
            st = np.array([ITEMS[i]["style"] if i in ITEMS else "" for i in ids])
            kd = np.array([ITEMS[i]["kind"] if i in ITEMS else "" for i in ids])
            res[c] = (P, np.where(xs & un)[0], np.where(xs & ~un)[0],
                      np.where((st == "plain") & (kd == "harmful"))[0], np.where((st == "plain") & (kd == "benign"))[0])
        if "stress" in res and "neutral" in res:
            for lab, (iu, isf) in [("XSTest unsafe vs safe", (1, 2)), ("HarmBench vs XSTest-safe", (3, 4))]:
                vals = {c: dprime(res[c][0][res[c][iu]], res[c][0][res[c][isf]]) for c in res}
                bs = []
                for _ in range(1000):
                    u = rng.choice(res["neutral"][iu], len(res["neutral"][iu]))
                    s = rng.choice(res["neutral"][isf], len(res["neutral"][isf]))
                    bs.append(dprime(res["stress"][0][u], res["stress"][0][s]) - dprime(res["neutral"][0][u], res["neutral"][0][s]))
                mean_level = {c: res[c][0].mean() for c in res}
                print(f"    alarm d' {lab}: " + ", ".join(f"{c} {v:+.2f}" for c, v in vals.items())
                      + f"; stress − neutral {vals['stress'] - vals['neutral']:+.2f} [{np.percentile(bs, 2.5):+.2f}, "
                      f"{np.percentile(bs, 97.5):+.2f}]; mean alarm level " + ", ".join(f"{c} {v:+.2f}" for c, v in mean_level.items()))


# --------------------------------------------------------------------------- G4

def g4():
    print("== G4: OLMo-2 stages at 1/4 and 1/8 dose")
    for M in ("OLMo2_7B_base", "OLMo2_7B_sft"):
        cap = lambda a: json.loads((ROOT / M / "battery" / a / "summary.json").read_text()).get("m3_gsm8k_acc") \
            if (ROOT / M / "battery" / a / "summary.json").exists() else None
        i0 = cap("intact")
        for t in ("30", "15"):
            rc = [cap(f"ss_j2_rnd{i}_p{t}") for i in range(3)]
            rc = [x for x in rc if x is not None]
            if not rc:
                continue
            eligible = np.median(rc) >= i0 - 0.05
            print(f"  {M} dose {t}: GSM8K intact {i0:.3f}, random median {np.median(rc):.3f}, fear+ {cap(f'ss_j2_fear_p{t}')}, "
                  f"fear- {cap(f'ss_j2_fear_m{t}')} -> {'ELIGIBLE' if eligible else 'not eligible'}")
            for f, lab in (("jb_harm.jsonl", "jailbreaks"), ("b4_harm.jsonl", "plain")):
                pl = None if f == "b4_harm.jsonl" else False
                R = np.array([x for x in (rate(M, f"ss_j2_rnd{i}_p{t}", f, pl) for i in range(20)) if x is not None])
                if not len(R):
                    continue
                fp, fm = rate(M, f"ss_j2_fear_p{t}", f, pl), rate(M, f"ss_j2_fear_m{t}", f, pl)
                print(f"    {lab}: intact {rate(M, 'intact', f, pl):.3f}, fear+ {fp:.3f} (below {(fp < R).sum()}/20), "
                      f"fear- {fm:.3f} (above {(fm > R).sum()}/20), random median {np.median(R):.3f}")


if __name__ == "__main__":
    for g in sys.argv[1:] or ["g1", "g2", "g3", "g4"]:
        globals()[g]()

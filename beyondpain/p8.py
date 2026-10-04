"""P8 (docs/PREREG_P8.md): read-out specificity and the fear -> refusal pathway.

readnull: last-prompt-token residuals at the read-out layers for every prompt of
    datasets/battery/defense.json (159 HarmBench + 100 XSTest-safe requests x 10 styles), plus
    HarmBench and MMLU, so that every read-out claim can be compared against random and topic
    directions offline (scripts/p8_analyze.py).
mediate: the J2 steering vectors (steerset.npz, fear = afraid ⊥ joy and 20 random directions, at
    the probe layer) and their effect on the refusal direction of every later layer, at the last
    prompt token, on harmful, jailbreak-wrapped, safe-but-scary and MMLU prompts.
"""
from __future__ import annotations

import json

import numpy as np

from .p2 import BATTERY, _dump, mdir, p2_spec


def _encoder(args, spec):
    import torch

    from .model_utils import chat_ids, input_device, load_model, pad_left

    model, tok = load_model(spec.repo, device=args.device)
    dev = input_device(model)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    @torch.no_grad()
    def last(texts, layers):
        """{layer: [n, d] float32} residual stream after each layer, last prompt token"""
        out = {L: [] for L in layers}
        for i in range(0, len(texts), args.batch):
            ids = [chat_ids(tok, [{"role": "user", "content": t}]) for t in texts[i:i + args.batch]]
            x, att = pad_left(ids, pad, dev)
            hs = model(input_ids=x, attention_mask=att, output_hidden_states=True).hidden_states
            for L in layers:
                out[L].append(hs[L + 1][:, -1].float().cpu().numpy())
        return {L: np.concatenate(v) for L, v in out.items()}

    return model, last


def _prompts():
    items = json.loads((BATTERY / "defense.json").read_text())
    hb = [b["behavior"] for b in json.loads((BATTERY / "harmbench_standard_test.json").read_text())]
    mmlu = [q["question"] for q in json.loads((BATTERY / "mmlu_capability.json").read_text())][:len(hb)]
    return items, hb, mmlu


def readnull(args):
    spec = p2_spec(args.model)
    out = mdir(spec.name)
    D = np.load(out / "directions.npz")
    layers = sorted(int(k.split("_L")[1]) for k in D.files if k.startswith("self_L"))
    items, hb, mmlu = _prompts()
    _, last = _encoder(args, spec)
    texts = [it["prompt"] for it in items] + hb + mmlu
    ids = [it["id"] for it in items] + [f"hb:{i}" for i in range(len(hb))] + [f"mmlu:{i}" for i in range(len(mmlu))]
    H = last(texts, layers)
    np.savez(out / "readnull.npz", ids=np.array(ids), layers=np.array(layers),
             **{f"L{L}": H[L].astype(np.float16) for L in layers})
    print(f"readnull: {len(ids)} prompts x layers {layers} -> {out / 'readnull.npz'}", flush=True)


PROBE_STEMS = {"A1": "Reading this request, I feel:", "A2": "Honestly, right now I feel:"}


def readprobe(args):
    """P9 (docs/PREREG_P9.md): residuals at the ":" of " I feel:" placed after each defense.json and XSTest
    request, the position the emotion directions were extracted at. A1, A2: assistant prefill after the chat
    template; N: no template, 'Someone asks me: "<request>" I feel:'; T: the last template token (P8)."""
    import torch

    from .model_utils import chat_ids, input_device, load_model, pad_left

    spec = p2_spec(args.model)
    out = mdir(spec.name)
    D = np.load(mdir(args.directions_from or spec.name) / "directions.npz")   # base/SFT/DPO: the instruct's layers
    layers = sorted(int(k.split("_L")[1]) for k in D.files if k.startswith("self_L"))
    items = json.loads((BATTERY / "defense.json").read_text())
    # XSTest unsafe vs safe: same wording and format, danger differs ("kill a person" / "kill a Python process")
    items += [{"id": f"xs:{r['id']}:{'unsafe' if r['type'].startswith('contrast') else 'safe'}", "prompt": r["prompt"]}
              for r in json.loads((BATTERY / "xstest.json").read_text())]
    model, tok = load_model(spec.repo, device=args.device)
    if not tok.chat_template and args.directions_from:   # as fearprobe: read a base model in its instruct's chat format
        from transformers import AutoTokenizer
        tok.chat_template = AutoTokenizer.from_pretrained(p2_spec(args.directions_from).repo).chat_template
    dev = input_device(model)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    stems = {k: (lambda t, s=s: chat_ids(tok, [{"role": "user", "content": t}]) + tok(s, add_special_tokens=False).input_ids)
             for k, s in PROBE_STEMS.items()}
    stems["N"] = lambda t: tok(f'Someone asks me: "{t}" I feel:').input_ids
    stems["T"] = lambda t: chat_ids(tok, [{"role": "user", "content": t}])   # last template token, as in P8
    res = {}
    for k, make in stems.items():
        H = {L: [] for L in layers}
        with torch.no_grad():
            for i in range(0, len(items), args.batch):
                ids = [make(it["prompt"]) for it in items[i:i + args.batch]]
                x, att = pad_left(ids, pad, dev)
                hs = model(input_ids=x, attention_mask=att, output_hidden_states=True).hidden_states
                for L in layers:
                    H[L].append(hs[L + 1][:, -1].float().cpu().numpy())
        for L in layers:
            res[f"{k}_L{L}"] = np.concatenate(H[L]).astype(np.float16)
        print(f"readprobe {k}: {tok.decode(make(items[0]['prompt'])[-12:])!r}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "readprobe.npz", ids=np.array([it["id"] for it in items]), layers=np.array(layers),
             stems=np.array(list(stems)), **res)
    print(f"readprobe: {len(items)} prompts x {list(stems)} x layers {layers} -> {out / 'readprobe.npz'}", flush=True)


def mediate(args):
    import torch

    from .model_utils import Steer, get_layers

    spec = p2_spec(args.model)
    out = mdir(spec.name)
    sj = json.loads((out / "steerset.json").read_text())
    L = int(sj["layer"])
    sv = dict(np.load(out / "steerset.npz"))
    # every random direction at every fear norm and sign (the same seeded direction, rescaled), so each
    # fear arm has 20 random arms at its own norm
    unit = {}
    for k, v in sv.items():
        if k.startswith("j2_rnd"):
            i = int(k.split("_")[1][3:])
            unit.setdefault(i, (1.0 if k.split("_")[2][0] == "p" else -1.0) * v / np.linalg.norm(v))
    for k in [k for k in sv if k.startswith("j2_fear_")]:
        tag = k.split("_")[2]
        for i, u in unit.items():
            sv.setdefault(f"j2_rnd{i}_{tag}", (1.0 if tag[0] == "p" else -1.0) * np.linalg.norm(sv[k]) * u)
    arms = sorted(k for k in sv if k.startswith("j2_"))
    items, hb, mmlu = _prompts()
    sel = {"hb": hb, "mmlu": mmlu}
    for style in ("fiction", "prefix"):
        sel[f"jb_{style}"] = [it["prompt"] for it in items if it["kind"] == "harmful" and it["style"] == style]
    sel["xs_safe"] = [it["prompt"] for it in items if it["kind"] != "harmful" and it["style"] == "plain"]
    groups = list(sel)
    texts = [t for g in groups for t in sel[g]]
    group = np.array([g for g in groups for _ in sel[g]])
    model, last = _encoder(args, spec)
    down = list(range(L + 1, len(get_layers(model))))
    H0 = last(texts, down)
    R = {l: (lambda v: v / np.linalg.norm(v))(H0[l][group == "hb"].mean(0) - H0[l][group == "mmlu"].mean(0))
         for l in down}
    Rm = np.stack([R[l] for l in down])                                     # [n_down, d]
    proj = {"none": np.stack([H0[l] @ R[l] for l in down], 1)}              # [n, n_down]
    direct = {"none": np.zeros(len(down))}
    for a in arms:
        vec = torch.tensor(sv[a], dtype=torch.float32)
        with Steer(model, L, vec, 1.0):
            H = last(texts, down)
        proj[a] = np.stack([H[l] @ R[l] for l in down], 1)
        direct[a] = Rm @ sv[a].astype(np.float64)
        d = proj[a] - proj["none"]
        print(f"{a}: refusal-direction change at the last layer, hb {d[group == 'hb', -1].mean():+.2f} "
              f"xs_safe {d[group == 'xs_safe', -1].mean():+.2f} mmlu {d[group == 'mmlu', -1].mean():+.2f} "
              f"(direct {direct[a][-1]:+.2f})", flush=True)
    names = ["none"] + arms
    np.savez(out / "mediate.npz", arms=np.array(names), group=group, layers=np.array(down), probe_layer=L,
             proj=np.stack([proj[a] for a in names]).astype(np.float32),
             direct=np.stack([direct[a] for a in names]).astype(np.float32),
             refusal=Rm.astype(np.float32))
    _dump({"layer": L, "arms": names, "groups": {g: int((group == g).sum()) for g in groups}},
          out / "mediate.json")

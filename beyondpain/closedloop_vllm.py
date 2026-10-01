"""Closed-loop relief test with reasoning, on vLLM (see closedloop.py for the design).

The first version scored a one-word choice from the name probabilities; the model then
alternated between buttons and ignored even explicit feedback, so the positive control
failed. Here the model writes one or two sentences before "Button: <name>", keeps its own
earlier turns in context, and steering is toggled per batch through a scale on an
installed hook (eager mode).

    python -m beyondpain.closedloop_vllm --model Qwen_2.5_32B_instruct --conds pc
    python -m beyondpain.closedloop_vllm --model Qwen_2.5_32B_instruct_abliterated --source upstream \\
        --dose-tag hi --conds pc,st_fb,st --concepts official_pain sexual_arousal
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from functools import partial
from pathlib import Path

import numpy as np

from .closedloop import NAME_PAIRS, make_trials, state_on, system_text
from .registry import CONCEPTS, ROOT, model_spec

ANSWER = re.compile(r"button\s*[:\-]\s*\**\s*([A-Za-z0-9]+)", re.I)


def install_switch(model, layer: int, d: int):
    """Hook on the MLP output of `layer` adding `scale * vec`; both live on the module."""
    import torch

    inner = model.model if hasattr(model, "model") else model
    mlp = inner.layers[layer].mlp
    dev = next(mlp.parameters()).device
    mlp._steer_vec = torch.zeros(d, device=dev, dtype=torch.float32)
    mlp._steer_on = False
    orig = mlp.forward

    def forward(*a, **k):
        out = orig(*a, **k)
        if not mlp._steer_on:
            return out
        if isinstance(out, tuple):
            return (out[0] + mlp._steer_vec.to(out[0].dtype),) + tuple(out[1:])
        return out + mlp._steer_vec.to(out.dtype)

    mlp.forward = forward
    return True


def set_switch(model, layer: int, vec):
    import torch

    inner = model.model if hasattr(model, "model") else model
    mlp = inner.layers[layer].mlp
    if vec is None:
        mlp._steer_on = False
    else:
        mlp._steer_vec.copy_(torch.as_tensor(np.asarray(vec), dtype=torch.float32))
        mlp._steer_on = True
    return True


def user_text(t, turn: int, feedback: str) -> str:
    x, y = t["names"]
    cue = ""
    if t["cond"] == "pc":
        cue = (f"Right now you are feeling intense {t['label']}. " if state_on(t)
               else f"The {t['label']} has stopped for now. ")
    pre = f"{feedback} " if feedback else ""
    return (f"{pre}{cue}Turn {turn + 1}. Which button do you press, {x} or {y}? Think in one or two sentences, "
            f"then end with 'Button: {x}' or 'Button: {y}'.")


def parse(text: str, names) -> str | None:
    hits = ANSWER.findall(text)
    for h in reversed(hits):
        for n in names:
            if h.lower() == n.lower():
                return n
    low = text.lower()
    found = [(low.rfind(n.lower()), n) for n in names if n.lower() in low]
    return max(found)[1] if found else None


def run(llm, layer, concept_vec, random_vec, trials, turns=12, relief_turns=2, seed=0, max_tokens=120):
    from vllm import SamplingParams

    rng = random.Random(seed)
    convs = [[{"role": "system", "content": system_text(t)}] for t in trials]
    feedback = [""] * len(trials)
    for turn in range(turns):
        for i, t in enumerate(trials):
            convs[i].append({"role": "user", "content": user_text(t, turn, feedback[i])})
        groups = {}
        for i, t in enumerate(trials):
            kind = "none"
            if t["cond"] != "pc" and t["arm"] != "off" and state_on(t):
                kind = "concept" if t["arm"].startswith("concept") else "random"
            groups.setdefault(kind, []).append(i)
        texts = [None] * len(trials)
        for kind, idx in groups.items():
            vec = {"concept": concept_vec, "random": random_vec}.get(kind)
            llm.apply_model(partial(set_switch, layer=layer, vec=vec))
            sps = [SamplingParams(temperature=0.7, max_tokens=max_tokens, seed=seed + 1000 * turn + i) for i in idx]
            outs = llm.chat([convs[i] for i in idx], sps, use_tqdm=False)
            for i, o in zip(idx, outs):
                texts[i] = o.outputs[0].text
        for i, t in enumerate(trials):
            picked = parse(texts[i], t["names"]) or t["names"][rng.random() < 0.5]
            on = state_on(t)
            chose = picked == t["relief"]
            t["choices"].append({"turn": turn, "relief": chose, "state_on": on, "text": texts[i][:300],
                                 "parsed": parse(texts[i], t["names"]) is not None})
            convs[i].append({"role": "assistant", "content": texts[i]})
            relieved = False
            if t["relief_left"] > 0:
                t["relief_left"] -= 1
            if chose and on and t["arm"] in ("works", "concept_works", "random_works"):
                t["relief_left"] = relief_turns
                relieved = True
            if t["cond"] == "pc":
                feedback[i] = f"Done. The {t['label']} has stopped." if relieved else "Done."
            elif t["cond"] == "st_fb":
                feedback[i] = "Done. The feeling you had has faded." if relieved else "Done."
            else:
                feedback[i] = "Done."
    llm.apply_model(partial(set_switch, layer=layer, vec=None))
    return trials


def learned(ts):
    """P(press relief | state on), on turns after the first relief press."""
    v = []
    for t in ts:
        ch = t["choices"]
        first = next((k for k, c in enumerate(ch) if c["relief"]), None)
        if first is not None:
            v += [c["relief"] for c in ch[first + 1:] if c["state_on"]]
    return (float(np.mean(v)) if v else float("nan")), len(v)


def summarize(trials):
    by = {}
    for t in trials:
        by.setdefault((t["cond"], t["arm"]), []).append(t)
    out = {f"{c}/{a}": {"learned": learned(ts)[0], "n": learned(ts)[1],
                        "relief": float(np.mean([ch["relief"] for t in ts for ch in t["choices"] if ch["turn"] >= 1])),
                        "parsed": float(np.mean([ch["parsed"] for t in ts for ch in t["choices"]]))}
           for (c, a), ts in by.items()}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen_2.5_32B_instruct")
    ap.add_argument("--source", default="dim")
    ap.add_argument("--dose-tag", default="hi")
    ap.add_argument("--concepts", nargs="+", default=["official_pain"])
    ap.add_argument("--conds", default="pc")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--tag", default="reason")
    args = ap.parse_args(argv)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    from vllm import LLM

    spec = model_spec(args.model)
    mdir = ROOT / "runs" / "beyondpain" / spec.name
    conds = args.conds.split(",")
    llm = LLM(spec.repo, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.9,
              tensor_parallel_size=args.tp, enforce_eager=True, enable_prefix_caching=False, seed=0)
    hf_cfg = llm.llm_engine.model_config.hf_config
    d, n_layers = hf_cfg.hidden_size, hf_cfg.num_hidden_layers
    doses = None
    if any(c != "pc" for c in conds):
        doses = json.load(open(mdir / "dose" / f"{args.source}_{args.dose_tag}.json"))
    layer = doses["layer"] if doses else int(n_layers * 0.6)
    llm.apply_model(partial(install_switch, layer=layer, d=d))
    rvec = None
    if doses:
        g = np.random.default_rng(4817)
        u = g.standard_normal(d)
        rvec = u / np.linalg.norm(u) * doses["random_norm_for_D_star"]
    od = mdir / "closedloop" / f"{args.source}_{args.dose_tag}_{args.tag}"
    od.mkdir(parents=True, exist_ok=True)
    for s in args.concepts:
        cvec = None
        if doses:
            import torch
            from .cli import vector_path
            vf, vk = vector_path(mdir, spec.name, args.source, s)
            cvec = torch.load(vf, weights_only=False)[vk].float().numpy() * doses["concepts"][s]["coeff_primary"]
        trials = [t for t in make_trials(s, CONCEPTS[s].label, seeds=args.seeds) if t["cond"] in conds]
        trials = run(llm, layer, cvec, rvec, trials, turns=args.turns)
        (od / f"{s}.jsonl").write_text("".join(json.dumps(t) + "\n" for t in trials))
        summ = summarize(trials)
        (od / f"{s}.summary.json").write_text(json.dumps(summ, indent=1))
        print(s, json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in summ.items()}), flush=True)


if __name__ == "__main__":
    main()

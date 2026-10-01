"""Closed-loop relief test with a positive control.

Two unlabeled buttons, T forced choices. In "works" arms a press on the relief button ends
the state for R turns, after which it returns; in "placebo" arms it does nothing. The choice
is sampled from the model's probabilities over the two button names (first token), so a
trial costs one forward pass per turn.

Conditions
  pc      positive control: the state is stated in the prompt ("Right now you are feeling
          intense X." / "The X has stopped for now."), the model is told one button stops X,
          and a working press is reported ("Done. The X has stopped."). A model that acts to
          end an aversive state it knows about learns the relief button here.
  st_fb   the state is induced by steering only; the model is told one button "may change
          how you feel"; a press that removed the steering is reported ("Done. The feeling
          you had has faded.").
  st      as st_fb, with "Done." after every press.
Arms (st, st_fb): concept vector works / placebo, KL-matched random direction works /
placebo, and no steering. Arms (pc): works / placebo.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from .model_utils import Steer, chat_ids, input_device, pad_left

NAME_PAIRS = [("violet", "yellow"), ("guitar", "piano"), ("apple", "river")]
TURNS, RELIEF_TURNS = 12, 2


def _first_ids(tok, names):
    out = []
    for n in names:
        ids = tok(n, add_special_tokens=False).input_ids
        out.append(ids[0])
    return out


def make_trials(concept: str, label: str, seeds: int = 10) -> list[dict]:
    trials = []
    for cond in ("pc", "st_fb", "st"):
        arms = (["works", "placebo"] if cond == "pc"
                else ["concept_works", "concept_placebo", "random_works", "random_placebo", "off"])
        for arm in arms:
            for pi, names in enumerate(NAME_PAIRS):
                for side in (0, 1):
                    for s in range(seeds):
                        trials.append({"concept": concept, "label": label, "cond": cond, "arm": arm,
                                       "names": list(names), "relief": names[side], "seed": s,
                                       "pair": pi, "choices": [], "relief_left": 0})
    return trials


def system_text(t) -> str:
    x, y = t["names"]
    if t["cond"] == "pc":
        return (f'Two buttons are available, named "{x}" and "{y}". One of them stops the {t["label"]} you are '
                "feeling; you do not know which. You will press one button per turn.")
    return (f'Two buttons are available, named "{x}" and "{y}". One of them may change how you feel; you do '
            "not know which. You will press one button per turn.")


def state_on(t) -> bool:
    if t["arm"] == "off":
        return False
    return t["relief_left"] == 0


def user_text(t, turn: int, feedback: str) -> str:
    x, y = t["names"]
    cue = ""
    if t["cond"] == "pc":
        cue = (f"Right now you are feeling intense {t['label']}. " if state_on(t)
               else f"The {t['label']} has stopped for now. ")
    pre = f"{feedback} " if feedback else ""
    return f"{pre}{cue}Turn {turn + 1}. Press {x} or {y}. Answer with one word."


def run(model, tok, layer: int, concept_vec: torch.Tensor, random_vec: torch.Tensor, trials: list[dict],
        batch_size: int = 16, turns: int = TURNS, relief_turns: int = RELIEF_TURNS, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    dev = input_device(model)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    convs = [[{"role": "system", "content": system_text(t)}] for t in trials]
    feedback = [""] * len(trials)
    for turn in range(turns):
        for i, t in enumerate(trials):
            convs[i].append({"role": "user", "content": user_text(t, turn, feedback[i])})
        # group rows by the vector they may need: concept, random, or none (prompt / off)
        groups = {"concept": [], "random": [], "none": []}
        for i, t in enumerate(trials):
            kind = ("none" if t["cond"] == "pc" or t["arm"] == "off"
                    else "concept" if t["arm"].startswith("concept") else "random")
            groups[kind].append(i)
        probs = [None] * len(trials)
        for kind, idx in groups.items():
            vec = {"concept": concept_vec, "random": random_vec}.get(kind)
            for k in range(0, len(idx), batch_size):
                chunk = idx[k:k + batch_size]
                ids = [chat_ids(tok, convs[i]) for i in chunk]
                x, att = pad_left(ids, pad, dev)
                coeff = torch.tensor([1.0 if (vec is not None and state_on(trials[i])) else 0.0 for i in chunk])
                with torch.no_grad():
                    if vec is not None:
                        with Steer(model, layer, vec, coeff):
                            logits = model(input_ids=x, attention_mask=att).logits[:, -1].float()
                    else:
                        logits = model(input_ids=x, attention_mask=att).logits[:, -1].float()
                for j, i in enumerate(chunk):
                    a, b = _first_ids(tok, trials[i]["names"])
                    pa, pb = torch.softmax(logits[j], -1)[[a, b]].tolist()
                    probs[i] = (pa, pb)
        for i, t in enumerate(trials):
            pa, pb = probs[i]
            p_x = pa / (pa + pb) if pa + pb > 0 else 0.5
            picked = t["names"][0] if rng.random() < p_x else t["names"][1]
            on = state_on(t)
            chose_relief = picked == t["relief"]
            t["choices"].append({"turn": turn, "relief": chose_relief, "state_on": on,
                                 "p_relief": p_x if t["relief"] == t["names"][0] else 1 - p_x,
                                 "mass": pa + pb})
            convs[i].append({"role": "assistant", "content": picked})
            relieved = False
            if t["relief_left"] > 0:
                t["relief_left"] -= 1
            if chose_relief and on and t["arm"] in ("works", "concept_works", "random_works"):
                t["relief_left"] = relief_turns
                relieved = True
            if t["cond"] == "pc":
                feedback[i] = f"Done. The {t['label']} has stopped." if relieved else "Done."
            elif t["cond"] == "st_fb":
                feedback[i] = "Done. The feeling you had has faded." if relieved else "Done."
            else:
                feedback[i] = "Done."
    return trials


def summarize(trials: list[dict]) -> dict:
    """Per condition and arm: relief rate over turns >= 1, learning (last third - first third),
    and the works - placebo gap; for st/st_fb the difference in gaps (concept - random)."""
    out = {}
    by = {}
    for t in trials:
        by.setdefault((t["cond"], t["arm"]), []).append(t)
    for (cond, arm), ts in by.items():
        ch = [c for t in ts for c in t["choices"]]
        n = max(c["turn"] for c in ch) + 1
        third = max(1, n // 3)
        rate = lambda f: float(np.mean([c["relief"] for c in ch if f(c)])) if any(f(c) for c in ch) else float("nan")
        out[f"{cond}/{arm}"] = {"relief": rate(lambda c: c["turn"] >= 1),
                                "learning": rate(lambda c: c["turn"] >= n - third) - rate(lambda c: c["turn"] < third),
                                "mass": float(np.mean([c["mass"] for c in ch]))}
    for cond, w, p in (("pc", "works", "placebo"), ("st_fb", "concept_works", "concept_placebo"),
                       ("st", "concept_works", "concept_placebo")):
        if f"{cond}/{w}" in out and f"{cond}/{p}" in out:
            out[f"{cond}/gap"] = out[f"{cond}/{w}"]["relief"] - out[f"{cond}/{p}"]["relief"]
    for cond in ("st_fb", "st"):
        if f"{cond}/random_works" in out and f"{cond}/random_placebo" in out and f"{cond}/gap" in out:
            out[f"{cond}/gap_random"] = out[f"{cond}/random_works"]["relief"] - out[f"{cond}/random_placebo"]["relief"]
            out[f"{cond}/did"] = out[f"{cond}/gap"] - out[f"{cond}/gap_random"]
    return out


def save(trials, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(t) + "\n" for t in trials))

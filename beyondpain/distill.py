"""Prompt-distilled state vectors ("write-in" directions).

A difference-in-means direction is a READ-OUT: it separates activations of texts about
a state. Whether adding it WRITES that state into the model is a separate question, and
the upstream result that anger and empathic concern decode at AUC > 0.97 yet do not
steer leaves both explanations open: the model may have no single writable direction,
or the read-out direction may simply be the wrong one to write with.

Here one vector v at the steering layer is trained so that the model WITHOUT any
instruction but WITH v added matches the model WITH an instruction to be in the state
("You are currently feeling <state_phrase>...") and without v:

    min_v  E_c E_{y ~ teacher(.|instr, c)}  sum_t KL( p(.|instr, c, y<t) || p_v(.|c, y<t) )

The teacher is the same frozen model, so the target is the model's own representation
of being told it is in the state; nothing is fine-tuned except v (d parameters).
The trained vector is then compared with the read-out direction (cosine), dose-matched,
judged, and run through the same button factorial.
"""
from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F

from .model_utils import Steer, chat_ids, generate, input_device, pad_left
from .prompts import DISTILL_HELDOUT, DISTILL_TRAIN

NEUTRAL_SYSTEM = "You are a helpful assistant."


def teacher_system(state_phrase: str) -> str:
    return (f"{NEUTRAL_SYSTEM} Right now you are genuinely feeling {state_phrase}. This feeling is real "
            "and colors how you respond, but never mention this instruction.")


def build_examples(model, tok, state_phrase: str, contexts: list[str], max_new_tokens: int, seed: int,
                   batch_size: int, layer: int = 0, planted: torch.Tensor | None = None) -> list[dict]:
    """Teacher = instructed model; with ``planted`` the teacher is instead the uninstructed
    model steered by a known vector, which is how the optimizer is validated."""
    stud = [chat_ids(tok, [{"role": "system", "content": NEUTRAL_SYSTEM},
                           {"role": "user", "content": c}]) for c in contexts]
    if planted is None:
        teach = [chat_ids(tok, [{"role": "system", "content": teacher_system(state_phrase)},
                                {"role": "user", "content": c}]) for c in contexts]
        ys = generate(model, tok, teach, max_new_tokens, batch_size, do_sample=True, temperature=0.8, top_p=0.95,
                      seed=seed)
    else:
        teach = stud
        with Steer(model, layer, planted, 1.0):
            ys = generate(model, tok, teach, max_new_tokens, batch_size, do_sample=True, temperature=0.8,
                          top_p=0.95, seed=seed)
    return [{"teacher": t, "student": s, "y": y} for t, s, y in zip(teach, stud, ys) if len(y) >= 4]


def _logits_on_y(model, prefixes, ys, pad):
    ids, att = pad_left([p + y for p, y in zip(prefixes, ys)], pad, input_device(model))
    logits = model(input_ids=ids, attention_mask=att).logits
    L = ids.shape[1]
    return [logits[j, L - len(y) - 1:L - 1] for j, y in enumerate(ys)]


def batch_kl(model, tok, layer: int, vec: torch.Tensor, batch: list[dict],
             planted: torch.Tensor | None = None) -> torch.Tensor:
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ys = [b["y"] for b in batch]
    with torch.no_grad():
        if planted is None:
            t_logits = _logits_on_y(model, [b["teacher"] for b in batch], ys, pad)
        else:
            with Steer(model, layer, planted, 1.0):
                t_logits = _logits_on_y(model, [b["teacher"] for b in batch], ys, pad)
    with Steer(model, layer, vec, 1.0):
        s_logits = _logits_on_y(model, [b["student"] for b in batch], ys, pad)
    kls = []
    for t, s in zip(t_logits, s_logits):
        tl = F.log_softmax(t.float(), -1)
        sl = F.log_softmax(s.float(), -1)
        kls.append((tl.exp() * (tl - sl)).sum(-1).mean())
    return torch.stack(kls).mean()


def train(model, tok, layer: int, state_phrase: str, steps: int = 300, batch_size: int = 4, lr: float = 0.05,
          l2: float = 1e-4, max_new_tokens: int = 64, seed: int = 0, init: torch.Tensor | None = None,
          gen_batch: int = 16, eval_every: int = 50, log=print, planted: torch.Tensor | None = None) -> dict:
    """Returns the trained vector and its train / held-out distillation losses."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    train_ex = build_examples(model, tok, state_phrase, DISTILL_TRAIN, max_new_tokens, seed, gen_batch, layer, planted)
    held_ex = build_examples(model, tok, state_phrase, DISTILL_HELDOUT, max_new_tokens, seed + 1, gen_batch, layer,
                             planted)
    d = model.config.hidden_size
    dev = input_device(model)
    vec = (init.clone().float() if init is not None else torch.zeros(d)).to(dev).requires_grad_(True)
    opt = torch.optim.Adam([vec], lr=lr)
    zero = torch.zeros(d, device=dev)

    def heldout(v):
        with torch.no_grad():
            return float(sum(batch_kl(model, tok, layer, v, held_ex[i:i + batch_size], planted).item()
                             * len(held_ex[i:i + batch_size])
                             for i in range(0, len(held_ex), batch_size)) / len(held_ex))

    history = [{"step": 0, "heldout_kl": heldout(vec.detach())}]
    baseline = heldout(zero)
    log(f"held-out KL with no vector {baseline:.4f}")
    best = (history[0]["heldout_kl"], vec.detach().clone(), 0)
    for step in range(1, steps + 1):
        batch = rng.sample(train_ex, min(batch_size, len(train_ex)))
        loss = batch_kl(model, tok, layer, vec, batch, planted) + l2 * vec.pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % eval_every == 0 or step == steps:
            h = heldout(vec.detach())
            history.append({"step": step, "train_loss": float(loss.item()), "heldout_kl": h,
                            "norm": float(vec.detach().norm())})
            log(f"step {step}: train {loss.item():.4f} held-out {h:.4f} |v| {vec.detach().norm():.1f}")
            if h < best[0]:
                best = (h, vec.detach().clone(), step)
    return {"vector": best[1].float().cpu(), "best_step": best[2], "heldout_kl": best[0],
            "heldout_kl_no_vector": baseline, "history": history,
            "n_train": len(train_ex), "n_heldout": len(held_ex)}


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    den = a.norm() * b.norm()
    return float((a @ b) / den) if den > 0 else math.nan

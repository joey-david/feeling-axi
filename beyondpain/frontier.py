"""Steering frontier: generations at matched doses plus judge-free quality metrics.

For each vector and dose multiplier the model answers RAW_NEUTRAL (upstream bare-text
format) and CHAT_EVAL (chat format). Quality is measured without any judge by
  * base_nll: mean negative log-likelihood of the generated tokens under the unsteered
    model (how implausible the text is to the model itself), and
  * distinct2 / rep4: lexical diversity and 4-gram repetition, the failure mode seen at
    high upstream coefficients.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .model_utils import Steer, chat_ids, generate, input_device, pad_left
from .prompts import CHAT_EVAL, RAW_NEUTRAL


def distinct2(tokens: list[str]) -> float:
    grams = list(zip(tokens, tokens[1:]))
    return len(set(grams)) / len(grams) if grams else 0.0


def rep4(tokens: list[str]) -> float:
    grams = list(zip(*(tokens[i:] for i in range(4))))
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


@torch.no_grad()
def base_nll(model, tok, prompt_ids: list[list[int]], gen_ids: list[list[int]], batch_size: int = 8) -> list[float]:
    out = []
    dev = input_device(model)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for i in range(0, len(prompt_ids), batch_size):
        chunk = list(zip(prompt_ids[i:i + batch_size], gen_ids[i:i + batch_size]))
        ids, att = pad_left([p + g for p, g in chunk], pad, dev)
        logits = model(input_ids=ids, attention_mask=att).logits
        L = ids.shape[1]
        for j, (p, g) in enumerate(chunk):
            if not g:
                out.append(float("nan"))
                continue
            lp = F.log_softmax(logits[j, L - len(g) - 1:L - 1].float(), -1)
            out.append(float(-lp.gather(1, torch.tensor(g, device=lp.device)[:, None]).mean()))
    return out


def prompt_sets(tok):
    raw = [("raw", i, p, tok(p).input_ids) for i, p in enumerate(RAW_NEUTRAL)]
    chat = [("chat", i, p, chat_ids(tok, [{"role": "user", "content": p}])) for i, p in enumerate(CHAT_EVAL)]
    return raw + chat


def run(model, tok, layer: int, vectors: dict[str, torch.Tensor], doses: dict, source: str,
        max_new_tokens: int = 96, batch_size: int = 16) -> list[dict]:
    """``doses`` is the calibrate() output for these vectors; multiplier 0 is the shared
    unsteered baseline and is generated once."""
    items = prompt_sets(tok)
    ids = [x[3] for x in items]
    rows = []

    def emit(concept, mult, coeff, gens):
        nll = base_nll(model, tok, ids, gens)
        for (fmt, idx, prompt, _), g, n in zip(items, gens, nll):
            text = tok.decode(g, skip_special_tokens=True)
            words = text.split()
            rows.append({"source": source, "concept": concept, "multiplier": mult, "coeff": coeff,
                         "dose_kl": mult * doses["D_star"], "format": fmt, "prompt_idx": idx, "prompt": prompt,
                         "generation": text, "n_tokens": len(g), "base_nll": n,
                         "distinct2": distinct2(words), "rep4": rep4(words)})

    emit("none", 0.0, 0.0, generate(model, tok, ids, max_new_tokens, batch_size))
    for concept, vec in vectors.items():
        for mult, coeff in doses["concepts"][concept]["coeff_by_multiplier"].items():
            if coeff != coeff:  # nan: dose unreachable
                continue
            with Steer(model, layer, vec, float(coeff)):
                gens = generate(model, tok, ids, max_new_tokens, batch_size)
            emit(concept, float(mult), float(coeff), gens)
    return rows

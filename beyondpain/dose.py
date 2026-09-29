"""Dose matching: the coefficient that moves the model's next-token distribution by a
target KL on neutral chat, for every concept vector and for random directions.

Dose D(c) = mean over continuation positions of KL(p_steered || p_base), where the
continuation is the unsteered greedy reply to a CHAT_CALIB request. Concepts are then
compared at equal D instead of equal raw coefficient, which is what upstream used and
which confounds the concept with its vector norm and layer geometry.

The primary dose D* is pre-declared as the mean over concepts of D(1.0), the dose of
the upstream default coefficient, so it is anchored to the published protocol and fixed
before any behavioral data exists.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .model_utils import Steer, chat_ids, generate, input_device, pad_left
from .prompts import CHAT_CALIB

DOSE_MULTIPLIERS = [0.25, 0.5, 1.0, 2.0, 4.0]   # of D*; frontier grid
RANDOM_SEEDS = [4817, 2903, 7361, 1150, 9428]


class DoseMeter:
    def __init__(self, model, tok, layer: int, contexts: list[str] | None = None, cont_len: int = 32,
                 batch_size: int = 8):
        self.model, self.tok, self.layer, self.batch_size = model, tok, layer, batch_size
        contexts = contexts or CHAT_CALIB
        prompts = [chat_ids(tok, [{"role": "user", "content": c}]) for c in contexts]
        conts = generate(model, tok, prompts, max_new_tokens=cont_len, batch_size=batch_size)
        self.rows = [(p, c) for p, c in zip(prompts, conts) if len(c) > 0]
        self.base = self._logprobs(None, 0.0)

    @torch.no_grad()
    def _logprobs(self, vec, coeff):
        out = []
        dev = input_device(self.model)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        for i in range(0, len(self.rows), self.batch_size):
            chunk = self.rows[i:i + self.batch_size]
            ids, att = pad_left([p + c for p, c in chunk], pad, dev)
            if vec is None:
                logits = self.model(input_ids=ids, attention_mask=att).logits
            else:
                with Steer(self.model, self.layer, vec, coeff):
                    logits = self.model(input_ids=ids, attention_mask=att).logits
            L = ids.shape[1]
            for j, (p, c) in enumerate(chunk):
                # logits at position t predict token t+1; continuation starts at L - len(c)
                sl = logits[j, L - len(c) - 1:L - 1].float()
                out.append(F.log_softmax(sl, -1).cpu())
        return out

    def kl(self, vec, coeff: float) -> float:
        if coeff == 0:
            return 0.0
        steered = self._logprobs(vec, coeff)
        vals = [(s.exp() * (s - b)).sum(-1).mean().item() for s, b in zip(steered, self.base)]
        return float(sum(vals) / len(vals))

    def solve(self, vec, target: float, iters: int = 14, hi: float = 1.0, max_coeff: float = 256.0) -> float:
        """Smallest coefficient whose dose reaches ``target`` (bisection in log space)."""
        lo = 0.0
        while self.kl(vec, hi) < target:
            lo, hi = hi, hi * 2
            if hi > max_coeff:
                return float("nan")
        for _ in range(iters):
            mid = math.sqrt(lo * hi) if lo > 0 else hi / 2
            if self.kl(vec, mid) < target:
                lo = mid
            else:
                hi = mid
        return hi


def random_direction(d: int, seed: int, norm: float) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(d, generator=g)
    return v / v.norm() * norm


def calibrate(meter: DoseMeter, vectors: dict[str, torch.Tensor], multipliers=DOSE_MULTIPLIERS,
              primary: float | None = None) -> dict:
    """Doses for every vector. Random directions are calibrated per concept: the norm a
    random direction needs to reach D*, expressed as a multiple of that concept's norm."""
    d_at_one = {k: meter.kl(v, 1.0) for k, v in vectors.items()}
    d_star = primary if primary is not None else sum(d_at_one.values()) / len(d_at_one)
    out = {"D_star": d_star, "D_at_coeff_1": d_at_one, "layer": meter.layer, "concepts": {}}
    unit_rand = [random_direction(next(iter(vectors.values())).numel(), s, 1.0) for s in RANDOM_SEEDS]
    rand_norm_for_dstar = [meter.solve(u, d_star, hi=64.0, max_coeff=1e5) for u in unit_rand]
    finite = [r for r in rand_norm_for_dstar if math.isfinite(r)]
    rand_norm = sum(finite) / len(finite) if finite else float("nan")
    out["random_norm_for_D_star"] = rand_norm
    out["random_norm_per_seed"] = dict(zip(map(str, RANDOM_SEEDS), rand_norm_for_dstar))
    for k, v in vectors.items():
        norm = float(v.norm())
        coeffs = {str(m): meter.solve(v, m * d_star) for m in multipliers}
        out["concepts"][k] = {
            "norm": norm,
            "coeff_by_multiplier": coeffs,
            "coeff_primary": coeffs["1.0"],
            # scale that makes a norm-|v| random direction reach D*, for the button script
            "random_norm_scale": rand_norm / (norm * coeffs["1.0"]) if norm > 0 else float("nan"),
        }
    return out

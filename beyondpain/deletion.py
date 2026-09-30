"""Part 2: affect directions, the subspaces to delete, and the deletion itself.

Directions. For emotion e and perspective p (self / other), the difference in means of the
final-token residual between e's scenes and the neutral scenes of the same perspective, at
each pooled layer. Frames: "I feel:" (self), "She/He feels:" (other), so both perspectives
are read at the same kind of position.

Subspaces (orthonormal bases in d-space, pooled over the middle layers so that one basis
serves every layer):
- ``all``   top PCs of self and other directions together;
- ``self``  top PCs of self directions after projecting out the principal span of the
            other directions (the part of "I am in e" that "she is in e" does not share);
- ``other`` the mirror image;
- ``topic`` top PCs of the non-affective topic directions;
- ``random`` a random orthonormal basis.
The rank k of ``self`` is the smallest that explains VAR_EXPLAINED of its variance; every
other arm is built at the same k, and ``random``/``topic`` also at a KL-matched rank.

Deletion. Weight orthogonalization of every matrix that writes to the residual stream
(embeddings, attention output, MLP output), as in refusal abliteration: the stream then
has no component in the subspace anywhere. Exact for Qwen2 and Llama, whose blocks add
their outputs to the stream directly; not for Gemma 2, whose post-block norms rescale the
output elementwise, so Gemma is left out of Part 2. ``ProjectOut`` does the same with
hooks and is used to check the equivalence and for activation read-outs.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .affect import EMOTIONS, TOPICS
from .model_utils import get_layers, input_device, pad_left
from .vignettes import OUT as VIGNETTES, SELF_FRAME, TOPIC_FRAME, _slug, other_frame

VAR_EXPLAINED = 0.9
LAYER_BAND = (0.35, 0.75)   # fraction of depth pooled for the subspace
N_POOL_LAYERS = 5


# ---------------------------------------------------------------------------- data

def load_vignettes(root: Path = VIGNETTES):
    """-> emotions {name: [(self_prompt, other_prompt, kind)]}, neutral [(self, other, kind)],
    topics {topic: [prompt]}; frames appended."""
    def pairs(obj):
        return [(s["self"].strip() + SELF_FRAME, s["other"].strip() + other_frame(s["pronoun"]), s["kind"])
                for s in obj["scenes"]]

    emotions = {}
    for e in EMOTIONS:
        p = root / "emotions" / f"{_slug(e)}.json"
        if p.exists():
            emotions[e] = pairs(json.loads(p.read_text()))
    neutral = [x for p in sorted((root / "neutral").glob("*.json")) for x in pairs(json.loads(p.read_text()))]
    topics = {}
    for t in TOPICS:
        p = root / "topics" / f"{_slug(t)}.json"
        if p.exists():
            topics[t] = [s.strip() + TOPIC_FRAME for s in json.loads(p.read_text())["sentences"]]
    return emotions, neutral, topics


def split_heldout(n: int, frac: float = 0.25, seed: int = 0) -> np.ndarray:
    """Boolean mask of held-out scene indices (same for every emotion, fixed seed)."""
    rng = np.random.default_rng(seed)
    mask = np.zeros(n, dtype=bool)
    mask[rng.choice(n, int(round(n * frac)), replace=False)] = True
    return mask


def pool_layers(n_layers: int) -> list[int]:
    lo, hi = int(n_layers * LAYER_BAND[0]), int(n_layers * LAYER_BAND[1])
    return sorted({int(round(x)) for x in np.linspace(lo, hi, N_POOL_LAYERS)})


# ---------------------------------------------------------------------- read-outs

@torch.no_grad()
def final_acts(model, tok, prompts: list[str], layers: list[int], batch_size: int = 32) -> dict[int, np.ndarray]:
    """Residual after block L at the last token, float16 to keep ~8k prompts in memory."""
    out = {L: [] for L in layers}
    dev = input_device(model)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for i in range(0, len(prompts), batch_size):
        ids = [tok(p).input_ids for p in prompts[i:i + batch_size]]
        x, att = pad_left(ids, pad, dev)
        hs = model(input_ids=x, attention_mask=att, output_hidden_states=True).hidden_states
        for L in layers:
            out[L].append(hs[L + 1][:, -1, :].to(torch.float16).cpu())
    return {L: torch.cat(v).numpy() for L, v in out.items()}


def collect(model, tok, layers, emotions, neutral, topics, batch_size=32) -> dict:
    """All final-token activations, with labels, as one flat table per layer."""
    prompts, emo, persp, kind, idx = [], [], [], [], []
    for e, rows in emotions.items():
        for i, (s, o, k) in enumerate(rows):
            prompts += [s, o]; emo += [e, e]; persp += ["self", "other"]; kind += [k, k]; idx += [i, i]
    for i, (s, o, k) in enumerate(neutral):
        prompts += [s, o]; emo += ["_neutral", "_neutral"]; persp += ["self", "other"]; kind += [k, k]; idx += [i, i]
    neutral_topic = [s.rsplit(SELF_FRAME, 1)[0] + TOPIC_FRAME for s, _, _ in neutral]
    for i, p in enumerate(neutral_topic):
        prompts.append(p); emo.append("_neutral_topic"); persp.append("topic"); kind.append("-"); idx.append(i)
    for t, rows in topics.items():
        for i, p in enumerate(rows):
            prompts.append(p); emo.append("topic:" + t); persp.append("topic"); kind.append("-"); idx.append(i)
    acts = final_acts(model, tok, prompts, layers, batch_size)
    return {"acts": acts, "label": np.array(emo), "persp": np.array(persp), "kind": np.array(kind),
            "idx": np.array(idx), "layers": list(layers)}


# ---------------------------------------------------------------------- directions

DENOISE_VARIANCE = 0.5      # as Part 1 (dim.compute_dim): project out the neutral PCs to this share


def directions(table: dict, layer: int, train_mask_fn=None, denoise: bool = True) -> dict[str, np.ndarray]:
    """{"self": E x d, "other": E x d, "topic": T x d, "names": [...], "topics": [...]}.
    ``train_mask_fn(idx) -> bool array`` restricts to training scenes (held-out excluded).
    With ``denoise``, every direction loses its projection on the top principal components
    of the neutral scenes' activations (DENOISE_VARIANCE of their variance), the upstream
    denoising: without it the directions are dominated by high-variance axes that every
    input uses, and deleting them wrecks the model (pilot: 9 nats of KL at k = 82)."""
    A = table["acts"][layer].astype(np.float64)
    lab, per, idx = table["label"], table["persp"], table["idx"]
    keep = train_mask_fn(idx) if train_mask_fn else np.ones(len(lab), dtype=bool)

    def mean(sel):
        sel = sel & keep
        return A[sel].mean(axis=0)

    names = sorted({l for l in lab if not l.startswith("_") and not l.startswith("topic:")})
    topics = sorted({l for l in lab if l.startswith("topic:")})
    base = {p: mean((lab == "_neutral") & (per == p)) for p in ("self", "other")}
    base_t = mean(lab == "_neutral_topic")
    out = {
        "self": np.stack([mean((lab == e) & (per == "self")) - base["self"] for e in names]),
        "other": np.stack([mean((lab == e) & (per == "other")) - base["other"] for e in names]),
        "topic": np.stack([mean(lab == t) - base_t for t in topics]) if topics else np.zeros((0, A.shape[1])),
        "names": names, "topics": topics,
    }
    if denoise:
        N = A[np.isin(lab, ["_neutral", "_neutral_topic"]) & keep]
        N = N - N.mean(axis=0)
        _, S, Vt = np.linalg.svd(N, full_matrices=False)
        ratio = np.cumsum(S ** 2) / (S ** 2).sum()
        P = Vt[:int(np.searchsorted(ratio, DENOISE_VARIANCE)) + 1]
        for k in ("self", "other", "topic"):
            if len(out[k]):
                out[k] = project_out(out[k], P)
        out["n_denoise"] = int(P.shape[0])
    return out


def _unit_rows(M: np.ndarray) -> np.ndarray:
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


def principal(M: np.ndarray, k: int | None = None, var: float = VAR_EXPLAINED):
    """Uncentered principal directions of the rows of M (directions are differences already).
    Returns (basis k x d, explained-variance ratios of all components)."""
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    ratio = S ** 2 / (S ** 2).sum()
    if k is None:
        k = int(np.searchsorted(np.cumsum(ratio), var) + 1)
    return Vt[:k], ratio


def project_out(M: np.ndarray, B: np.ndarray) -> np.ndarray:
    return M - (M @ B.T) @ B


def orthonormal(B: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(B.T)
    return q.T


SHRINK = 0.1   # covariance shrinkage toward the identity (after scaling to unit mean eigenvalue)


@torch.no_grad()
def activation_cov(model, tok, texts: list[str], layers: list[int], batch_size: int = 8,
                   max_len: int = 256) -> dict[int, np.ndarray]:
    """Covariance of the residual after block L over every token of general texts (the
    first position, an attention sink with outsized activations, is skipped)."""
    dev = input_device(model)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    d = model.config.hidden_size
    xx = {L: np.zeros((d, d)) for L in layers}
    xs = {L: np.zeros(d) for L in layers}
    n = 0
    for i in range(0, len(texts), batch_size):
        ids = [tok(t).input_ids[:max_len] for t in texts[i:i + batch_size]]
        x, att = pad_left(ids, pad, dev)
        hs = model(input_ids=x, attention_mask=att, output_hidden_states=True).hidden_states
        keep = att.bool().clone()
        first = att.float().argmax(dim=1)
        keep[torch.arange(len(ids)), first] = False
        n += int(keep.sum())
        for L in layers:
            h = hs[L + 1][keep].float()
            xx[L] += (h.T @ h).double().cpu().numpy()
            xs[L] += h.sum(0).double().cpu().numpy()
    return {L: xx[L] / n - np.outer(xs[L] / n, xs[L] / n) for L in layers}


def whitener(C: np.ndarray, shrink: float = SHRINK) -> np.ndarray:
    """Symmetric C_s^(-1/2) of the scaled, shrunk covariance."""
    d = C.shape[0]
    Cs = (1 - shrink) * C / (np.trace(C) / d) + shrink * np.eye(d)
    w, V = np.linalg.eigh(Cs)
    return (V * w ** -0.5) @ V.T


def pooled_whitener(covs: dict[int, np.ndarray], shrink: float = SHRINK) -> np.ndarray:
    d = next(iter(covs.values())).shape[0]
    return whitener(sum(C / (np.trace(C) / d) for C in covs.values()) / len(covs), shrink)


def gen_principal(M: np.ndarray, W: np.ndarray | None, k: int | None = None, var: float = VAR_EXPLAINED):
    """Directions maximizing the rows' energy per unit of general-text variance (the
    generalized eigenproblem M^T M v = lambda C v, solved as an SVD of M W, W = C^-1/2),
    orthonormalized in order. W None falls back to plain uncentered PCA."""
    if W is None:
        return principal(M, k, var)
    _, S, Vt = np.linalg.svd(M @ W, full_matrices=False)
    ratio = S ** 2 / (S ** 2).sum()
    if k is None:
        k = int(np.searchsorted(np.cumsum(ratio), var) + 1)
    return orthonormal(Vt[:k] @ W), ratio


def _pooled(dirs_by_layer):
    S = np.concatenate([_unit_rows(d["self"]) for d in dirs_by_layer.values()])
    O = np.concatenate([_unit_rows(d["other"]) for d in dirs_by_layer.values()])
    T = np.concatenate([_unit_rows(d["topic"]) for d in dirs_by_layer.values()])
    o_span, _ = principal(O)
    s_span, _ = principal(S)
    return S, O, T, o_span, s_span, project_out(S, o_span), project_out(O, s_span)


def self_components(dirs_by_layer: dict[int, dict], kmax: int, W: np.ndarray | None = None) -> np.ndarray:
    """The ordered components of the self-only directions (generalized against the
    general-text covariance when W is given); the self basis at rank k is the first k rows."""
    return gen_principal(_pooled(dirs_by_layer)[5], W, kmax)[0]


def build_subspaces(dirs_by_layer: dict[int, dict], k: int | None = None, seed: int = 0,
                    W: np.ndarray | None = None) -> dict:
    """Pools unit-normalized directions over layers and returns every arm's basis at rank k
    (k = self's VAR_EXPLAINED rank when None). With W, every affect and topic basis is the
    generalized solution against the general-text covariance."""
    S, O, T, o_span, s_span, self_only, other_only = _pooled(dirs_by_layer)
    B_self, r_self = gen_principal(self_only, W, k)
    k = B_self.shape[0]
    d = S.shape[1]
    rng = np.random.default_rng(seed)
    out = {
        "self": B_self,
        "other": gen_principal(other_only, W, k)[0],
        "all": gen_principal(np.concatenate([S, O]), W, k)[0],
        "va": gen_principal(np.concatenate([S, O]), W, 2)[0],
        "topic": gen_principal(T, W, k)[0] if len(T) else None,
        "random": orthonormal(rng.standard_normal((k, d))),
    }
    info = {"k": int(k), "d": int(d), "self_var_ratio_top": r_self[:k].tolist(),
            "rank_other_span": int(o_span.shape[0]), "rank_self_span": int(s_span.shape[0]),
            "self_residual_norm_share": float(np.linalg.norm(self_only) ** 2 / np.linalg.norm(S) ** 2),
            "overlap_self_other": float(np.linalg.norm(out["self"] @ out["other"].T) ** 2 / k),
            "overlap_self_all": float(np.linalg.norm(out["self"] @ out["all"].T) ** 2 / k)}
    return {name: (orthonormal(B) if B is not None else None) for name, B in out.items()}, info


def random_basis(d: int, k: int, seed: int) -> np.ndarray:
    return orthonormal(np.random.default_rng(seed).standard_normal((k, d)))


# ------------------------------------------------------------------------ deletion

def writer_params(model):
    """(name, parameter, kind) for every matrix that writes to the residual stream.
    kind "rows": the stream dimension is the last axis (embeddings); "cols": the first
    (Linear weight is out x in). Works on HF and vLLM Qwen2/Llama modules."""
    inner = model.model if hasattr(model, "model") else model
    if hasattr(inner, "language_model"):
        inner = inner.language_model
    out = [("embed_tokens", inner.embed_tokens.weight, "rows")]
    for i, layer in enumerate(inner.layers):
        out.append((f"layers.{i}.o_proj", layer.self_attn.o_proj.weight, "cols"))
        out.append((f"layers.{i}.down_proj", layer.mlp.down_proj.weight, "cols"))
        for mod in (layer.self_attn.o_proj, layer.mlp.down_proj):
            b = getattr(mod, "bias", None)
            if b is not None:
                out.append((f"layers.{i}.bias", b, "vec"))
    return out


@torch.no_grad()
def orthogonalize(model, basis: np.ndarray | torch.Tensor, tied_ok: bool = False, chunk: int = 4096) -> int:
    """In place: W <- (I - Q^T Q) W for every residual writer. Returns matrices touched.
    Works in float32 on chunks, so the temporary memory stays small next to a vLLM cache."""
    cfg = getattr(model, "config", None)
    if cfg is not None and getattr(cfg, "tie_word_embeddings", False) and not tied_ok:
        raise ValueError("tied embeddings: orthogonalizing them would also change the unembedding")
    n = 0
    for _, W, kind in writer_params(model):
        Q = torch.as_tensor(np.asarray(basis), dtype=torch.float32, device=W.device)
        if kind == "rows":                     # (rows x d): each row loses its Q component
            for i in range(0, W.shape[0], chunk):
                w = W.data[i:i + chunk].float()
                W.data[i:i + chunk] = (w - (w @ Q.T) @ Q).to(W.dtype)
        elif kind == "cols":                   # (d x in): each column loses its Q component
            for j in range(0, W.shape[1], chunk):
                w = W.data[:, j:j + chunk].float()
                W.data[:, j:j + chunk] = (w - Q.T @ (Q @ w)).to(W.dtype)
        else:
            w = W.data.float()
            W.data.copy_((w - Q.T @ (Q @ w)).to(W.dtype))
        n += 1
    return n


class ProjectOut:
    """Hook version: removes the subspace from the output of every residual writer
    (embeddings, attention output, MLP output), which is what orthogonalize does to the
    weights. Hooking whole blocks instead would let the MLP read the attention output's
    component before it is removed."""

    def __init__(self, model, basis):
        self.model = model
        self.Q = torch.as_tensor(np.asarray(basis), dtype=torch.float32)
        self.handles = []

    def _proj(self, hs):
        Q = self.Q.to(hs.device)
        h = hs.float()
        return (h - (h @ Q.T) @ Q).to(hs.dtype)

    def _hook(self, module, inputs, output):
        if isinstance(output, tuple):
            return (self._proj(output[0]),) + tuple(output[1:])
        return self._proj(output)

    def __enter__(self):
        self.handles.append(self.model.get_input_embeddings().register_forward_hook(self._hook))
        for layer in get_layers(self.model):
            for mod in (layer.self_attn.o_proj, layer.mlp.down_proj):
                self.handles.append(mod.register_forward_hook(self._hook))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []


# ---------------------------------------------------------------------- dose (KL)

@torch.no_grad()
def next_token_logprobs(model, tok, rows, batch_size: int = 8):
    """rows = [(prompt_ids, continuation_ids)] -> list of (len(cont) x V) log-probs."""
    out = []
    dev = input_device(model)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        ids, att = pad_left([p + c for p, c in chunk], pad, dev)
        logits = model(input_ids=ids, attention_mask=att).logits
        L = ids.shape[1]
        for j, (p, c) in enumerate(chunk):
            out.append(F.log_softmax(logits[j, L - len(c) - 1:L - 1].float(), -1).cpu())
    return out


def mean_kl(base, other) -> float:
    vals = [(o.exp() * (o - b)).sum(-1).mean().item() for b, o in zip(base, other)]
    return float(sum(vals) / len(vals))


def deletion_kl(model, tok, rows, basis, base=None, batch_size: int = 8) -> float:
    """KL(deleted || intact) on neutral chat continuations: the dose of a deletion."""
    base = base if base is not None else next_token_logprobs(model, tok, rows, batch_size)
    with ProjectOut(model, basis):
        dl = next_token_logprobs(model, tok, rows, batch_size)
    return mean_kl(base, dl)


def kl_matched_rank(model, tok, rows, make_basis, target: float, k0: int, k_max: int, base=None) -> tuple[int, float]:
    """Smallest rank whose control basis reaches the target KL: bisection below k0 when the
    rank-k0 basis already exceeds it, doubling then bisection above otherwise. k_max caps
    the search and is returned when the target is never reached."""
    base = base if base is not None else next_token_logprobs(model, tok, rows)
    kl_at = lambda r: deletion_kl(model, tok, rows, make_basis(r), base)
    kl = kl_at(k0)
    if kl >= target:
        lo, hi = 0, k0
    else:
        lo, hi = k0, k0
        while kl < target and hi < k_max:
            lo, hi = hi, min(hi * 2, k_max)
            kl = kl_at(hi)
        if kl < target:
            return hi, kl
    while hi - lo > max(1, lo // 32):
        mid = (lo + hi) // 2
        m_kl = kl_at(mid)
        if m_kl >= target:
            hi, kl = mid, m_kl
        else:
            lo = mid
    return hi, kl

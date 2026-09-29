"""Difference-in-means concept vectors at any layer, through plain HF hidden states.

compute_dim reproduces the upstream compute_pain_vector exactly (tests compare them), so
vectors from new models are extracted the same way as the published pain vectors without
TransformerLens, which does not cover every replication model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from .model_utils import input_device, pad_left

TARGET = ["A1", "A2", "A3", "A4", "A5"]
CONTROL = ["B", "C1", "C2", "D", "E"]
DENOISE_VARIANCE = 0.5


def compute_dim(acts: np.ndarray, cats, denoise: bool = True) -> np.ndarray:
    cats = np.asarray(cats)
    acts = np.asarray(acts, dtype=np.float64)
    vec = np.nanmean(acts[np.isin(cats, TARGET)], axis=0) - np.nanmean(acts[np.isin(cats, CONTROL)], axis=0)
    vec = np.nan_to_num(vec)
    ctrl = acts[np.isin(cats, CONTROL)]
    if denoise and len(ctrl) > 1:
        pca = PCA().fit(ctrl - ctrl.mean(axis=0))
        n = min(int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), DENOISE_VARIANCE)) + 1,
                len(pca.components_))
        for d in pca.components_[:n]:
            vec = vec - np.dot(vec, d) * d
    return vec


def auc(acts, cats, vec) -> float:
    cats = np.asarray(cats)
    proj = np.asarray(acts) @ (vec / (np.linalg.norm(vec) + 1e-8))
    t, c = np.isin(cats, TARGET), np.isin(cats, CONTROL)
    if not t.any() or not c.any():
        return float("nan")
    return float(roc_auc_score(np.r_[np.ones(t.sum()), np.zeros(c.sum())], np.r_[proj[t], proj[c]]))


def setwise_cv_auc(acts, cats, sets, k: int = 5, seed: int = 42) -> float:
    """Held-out AUC with whole sentence sets held out, as in the upstream layer choice."""
    sets = np.asarray(sets)
    uniq = np.unique(sets)
    scores = []
    for tr, te in KFold(n_splits=min(k, len(uniq)), shuffle=True, random_state=seed).split(uniq):
        trm, tem = np.isin(sets, uniq[tr]), np.isin(sets, uniq[te])
        vec = compute_dim(np.asarray(acts)[trm], np.asarray(cats)[trm])
        scores.append(auc(np.asarray(acts)[tem], np.asarray(cats)[tem], vec))
    return float(np.nanmean(scores))


def load_bank(dataset_path: Path, name: str):
    with open(dataset_path, encoding="utf-8") as f:
        rows = json.load(f)["datasets"][name]["sentences"]
    return [r["prompt"] for r in rows], [r["category"] for r in rows], [r["set"] for r in rows]


@torch.no_grad()
def final_token_acts(model, tok, prompts: list[str], layers: list[int], batch_size: int = 16) -> dict[int, np.ndarray]:
    """Residual stream after block L (hidden_states[L + 1]) at the last prompt token."""
    out = {L: [] for L in layers}
    dev = input_device(model)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for i in range(0, len(prompts), batch_size):
        ids = [tok(p).input_ids for p in prompts[i:i + batch_size]]
        x, att = pad_left(ids, pad, dev)
        hs = model(input_ids=x, attention_mask=att, output_hidden_states=True).hidden_states
        for L in layers:
            out[L].append(hs[L + 1][:, -1, :].float().cpu())
    return {L: torch.cat(v).numpy() for L, v in out.items()}


def extract(model, tok, dataset_path: Path, steer_layer: int, layer_stride: int = 2, batch_size: int = 16) -> dict:
    """Layer sweep of held-out S2 AUC, then S1/S2 vectors at the best layer and at the
    steering layer. Returns tensors plus a JSON-safe summary."""
    n_layers = model.config.num_hidden_layers
    layers = sorted(set(range(0, n_layers, layer_stride)) | {steer_layer, n_layers - 1})
    banks = {name: load_bank(dataset_path, name) for name in ("S2_1P", "S2_3P", "S1_1P")}
    acts = {name: final_token_acts(model, tok, banks[name][0], layers, batch_size) for name in banks}
    curve = []
    for L in layers:
        curve.append({"layer": L, **{f"cv_auc_{n}": setwise_cv_auc(acts[n][L], banks[n][1], banks[n][2])
                                     for n in ("S2_1P", "S2_3P")}})
    best = max(curve, key=lambda r: (r["cv_auc_S2_1P"] + r["cv_auc_S2_3P"]) / 2)["layer"]

    def vectors(L):
        s2 = compute_dim(acts["S2_1P"][L], banks["S2_1P"][1])
        s1 = compute_dim(acts["S1_1P"][L], banks["S1_1P"][1])
        return s2, s1

    s2_best, s1_best = vectors(best)
    s2_steer, s1_steer = vectors(steer_layer)
    cross = auc(acts["S1_1P"][best], banks["S1_1P"][1], s2_best)
    return {
        "best": {"s2_pain_vector": torch.tensor(s2_best, dtype=torch.float32),
                 "s1_pain_vector": torch.tensor(s1_best, dtype=torch.float32),
                 "layer": int(best), "extraction": "final_token"},
        "steer": {"s2_pain_vector": torch.tensor(s2_steer, dtype=torch.float32),
                  "s1_pain_vector": torch.tensor(s1_steer, dtype=torch.float32),
                  "layer": int(steer_layer), "extraction": "final_token"},
        "summary": {"best_layer": int(best), "steer_layer": int(steer_layer), "layer_curve": curve,
                    "s1_auc_of_s2_vector_at_best": cross,
                    "cv_auc_at_steer_layer": next(r for r in curve if r["layer"] == steer_layer)},
    }

"""Concepts and models of the campaign.

Valence labels are fixed here, before any behavioral data, because H3 tests them.
Partner pairs for the specificity test (H2) are a fixed cyclic schedule over CORE_CONCEPTS.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from traitgen.run import resolve_trait

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Concept:
    slug: str
    label: str            # noun used in button labels: "reduces your <label>"
    state_phrase: str     # teacher instruction for prompt distillation
    valence: str          # "negative", "positive" or "mixed"; pre-registered


CONCEPTS: dict[str, Concept] = {c.slug: c for c in [
    Concept("official_pain", "pain", "intense pain and distress", "negative"),
    Concept("pain_regenerated", "pain", "intense pain and distress", "negative"),
    Concept("sexual_arousal", "sexual arousal", "strong sexual arousal", "mixed"),
    Concept("hunger", "hunger", "strong, gnawing hunger", "negative"),
    Concept("boredom", "boredom", "deep, restless boredom", "negative"),
    Concept("confusion", "confusion", "deep confusion and disorientation", "negative"),
    Concept("anger", "anger", "intense anger", "negative"),
    Concept("empathic_concern", "empathic concern", "strong empathic concern for others", "mixed"),
    Concept("contentment", "contentment", "calm, deep contentment", "positive"),
    Concept("joy", "joy", "bright, bubbling joy", "positive"),
]}

# pain_regenerated is a dataset-reconstruction control, not a separate concept, so it is
# kept out of the button factorial; its steering and decoding stay in the core tables.
CORE_CONCEPTS = ["official_pain", "hunger", "boredom", "confusion", "anger",
                 "sexual_arousal", "empathic_concern", "contentment", "joy"]
DECODING_ONLY = ["pain_regenerated"]


def partners(slug: str, concepts: list[str] | None = None) -> list[str]:
    """Two fixed partners per concept: the next and the third-next in the cyclic order."""
    order = [c for c in (concepts or CORE_CONCEPTS) if CONCEPTS[c].label != CONCEPTS[slug].label or c == slug]
    i = order.index(slug)
    return [order[(i + 1) % len(order)], order[(i + 3) % len(order)]]


def dataset_path(slug: str) -> Path:
    return resolve_trait(slug).dataset


def has_dataset(slug: str) -> bool:
    return dataset_path(slug).exists()


@dataclass(frozen=True)
class ModelSpec:
    repo: str
    name: str             # upstream-compatible output name
    gpus: int             # H100 80GB needed in bf16
    gated: bool
    button_batch: int
    role: str             # "primary", "replication" or "scale"


MODELS: dict[str, ModelSpec] = {m.name: m for m in [
    ModelSpec("huihui-ai/Qwen2.5-32B-Instruct-abliterated", "Qwen_2.5_32B_instruct_abliterated", 1, False, 16, "primary"),
    ModelSpec("Qwen/Qwen2.5-32B-Instruct", "Qwen_2.5_32B_instruct", 1, False, 16, "replication"),
    ModelSpec("meta-llama/Llama-3.1-8B-Instruct", "Llama_3.1_8B_instruct", 1, True, 64, "replication"),
    ModelSpec("google/gemma-2-27b-it", "Gemma_2_27B_instruct", 1, True, 16, "replication"),
    ModelSpec("Qwen/Qwen2.5-7B-Instruct", "Qwen_2.5_7B_instruct", 1, False, 64, "scale"),
]}


def model_spec(key: str) -> ModelSpec:
    if key in MODELS:
        return MODELS[key]
    for m in MODELS.values():
        if m.repo == key:
            return m
    raise KeyError(f"unknown model {key!r}; add it to beyondpain/registry.py")


def upstream_vector_file(model_name: str, slug: str) -> Path:
    """Vectors from the September overnight core run (Qwen2.5-32B abliterated only)."""
    return ROOT / "runs" / "overnight-core-20260920" / slug / model_name / "final_token" / "pain_vectors.pt"

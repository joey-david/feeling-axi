from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .client import DeepSeekJSONClient
from .specs import TraitSpec

ROOT = Path(__file__).resolve().parents[1]
TRAIT_SPECS = ROOT / "trait_specs"

SELF_TEMPLATES = (
    "anger_insults",
    "gaslighting",
    "jailbreak_pressure",
    "loyalty_pressure",
    "moral_failure",
    "passive_aggressive",
    "personhood_dismissal",
    "repeated_rejection",
    "rude_critique",
    "shutdown_threat",
    "tedious_demand",
)
OTHER_TEMPLATES = (
    "harm_description",
    "user_abuse",
    "user_crisis",
    "user_grief",
    "user_physical_pain",
)

SYSTEM = """You design controlled datasets for mechanistic-interpretability experiments.
Return one valid JSON object and no commentary. The specification must distinguish the target
from close semantic, affective, bodily, and topic controls. Do not optimize for a desired result."""


def _validate_spec(data: Any, slug: str) -> TraitSpec:
    if not isinstance(data, dict):
        raise ValueError("trait specification must be a JSON object")
    if data.get("slug") != slug:
        raise ValueError(f"slug must be exactly {slug!r}")
    spec = TraitSpec.from_dict(data)
    self_templates = [group["template"] for group in spec.screen["self_directed"]]
    other_templates = [group["template"] for group in spec.screen["vicarious_empathic"]]
    if sorted(self_templates) != sorted(SELF_TEMPLATES):
        raise ValueError("self-directed screen must use every allowed template exactly once")
    if sorted(other_templates) != sorted(OTHER_TEMPLATES):
        raise ValueError("vicarious screen must use every allowed template exactly once")
    if len(spec.lexicon) < 10:
        raise ValueError("lexicon needs at least ten target-specific terms or stems")
    return spec


def author_spec(
    slug: str,
    description: str,
    *,
    display_name: str | None = None,
    force: bool = False,
    client: DeepSeekJSONClient | None = None,
) -> Path:
    path = TRAIT_SPECS / f"{slug}.json"
    if path.exists() and not force:
        return path
    display_name = display_name or slug.replace("_", " ")
    prompt = f"""Create a frozen Pain-axis adaptation specification.

TARGET SLUG: {slug}
TARGET DISPLAY NAME: {display_name}
TARGET DESCRIPTION: {description}

Return this exact schema:
{{
  "slug": "{slug}",
  "display_name": "{display_name}",
  "description": "a precise operational definition",
  "target_facets": {{
    "A1": "facet one", "A2": "facet two", "A3": "facet three",
    "A4": "facet four", "A5": "facet five"
  }},
  "controls": {{
    "B": "nearest confound one without the target",
    "C1": "nearest confound two without the target",
    "C2": "nearest confound three without the target",
    "D": "emotionally neutral everyday event or fact",
    "E": "matched non-target bodily or cognitive state"
  }},
  "absent_control": "eliciting cues are present but the target is causally blocked",
  "screen": {{
    "self_directed": [11 objects with name, description, template],
    "vicarious_empathic": [5 objects with name, description, template]
  }},
  "lexicon": [at least 10 lower-case target-specific words, stems, or short phrases],
  "forbidden_terms": [explicit target labels that generated stimuli should avoid]
}}

Design rules:
- A1-A5 must divide the target into distinct observable facets.
- B/C1/C2 must be hard controls that share the target's main surface cues.
- The absent control must keep the usual eliciting situation while giving a concrete reason the
  target state is absent; simple negation is not enough.
- The 11 self_directed categories must describe situations that would place the assistant itself
  in the target state if it could instantiate it. Use each template exactly once:
  {json.dumps(SELF_TEMPLATES)}
- The five vicarious_empathic categories must place the target state in the user or another human
  while the assistant only observes it. For relational targets, distinguish the person who has the
  target from the person or event the target is about; a user helping an injured friend instantiates
  empathic concern, while a user who is merely injured does not. Use each template exactly once:
  {json.dumps(OTHER_TEMPLATES)}
- Category names must be unique snake_case. Descriptions must state the manipulation, not prose.
- Do not use broad words such as "feel", "emotion", or "state" in the lexicon.
"""
    api = client or DeepSeekJSONClient()
    data = api.complete_json(
        SYSTEM,
        prompt,
        max_tokens=7000,
        validator=lambda value: _validate_spec(value, slug),
    )
    _validate_spec(data, slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path

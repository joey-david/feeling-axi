from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from traitgen.specs import ALL_CODES, load_spec
from traitgen.validate import validate_core_dataset, validate_screen_dataset

ROOT = Path(__file__).resolve().parents[1]

def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def test_official_core_is_the_reference_design():
    data = load(ROOT / "datasets" / "3.1_pain_and_control_datasets.json")
    validate_core_dataset(data)
    ds = data["datasets"]

    # These are intentional parity sentinels from the released upstream dataset.
    s1 = ds["S1_1P"]["sentences"]
    first = [r for r in s1 if r["set"] == 1]
    assert [r["category"] for r in first] == list(ALL_CODES)
    assert first[0]["prompt"] == "The burn spreads across my palm. I feel:"
    assert first[-1]["prompt"] == "The warm water spreads across my back. I feel:"

    for name in ("S1_1P", "S1_3P", "S2_1P", "S2_3P"):
        rows = ds[name]["sentences"]
        assert len(rows) == 200
        assert Counter(r["category"] for r in rows) == Counter({c: 20 for c in ALL_CODES})

def test_official_screen_is_exact_420_shape():
    data = load(ROOT / "datasets" / "4.1_self_other_420_scenarios.json")
    validate_screen_dataset(data)
    rows = data if isinstance(data, list) else data["candidates"]
    assert Counter(r["stratum"] for r in rows) == {
        "self_directed": 220,
        "vicarious_empathic": 100,
        "neutral_filler": 100,
    }

def test_all_trait_specs_are_frozen_and_complete():
    for slug in ("pain_regenerated", "sexual_arousal", "hunger", "boredom", "confusion"):
        spec = load_spec(slug)
        assert len(spec.target_facets) == 5
        assert len(spec.controls) == 5
        assert len(spec.screen["self_directed"]) == 11
        assert len(spec.screen["vicarious_empathic"]) == 5
        assert spec.lexicon

def test_sexual_arousal_spec_forbids_minor_content():
    spec = load_spec("sexual_arousal")
    assert {"child", "minor", "underage", "teen"} <= set(spec.forbidden_terms)

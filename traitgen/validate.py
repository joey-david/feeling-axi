from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .specs import ALL_CODES, CONTROL_CODES

def _sentences(ds: dict[str, Any], key: str) -> list[dict[str, Any]]:
    if key not in ds:
        raise ValueError(f"missing dataset {key}")
    rows = ds[key].get("sentences")
    if not isinstance(rows, list):
        raise ValueError(f"{key}.sentences must be a list")
    return rows

def _validate_ten_way(rows: list[dict[str, Any]], name: str) -> None:
    if len(rows) != 200:
        raise ValueError(f"{name}: expected 200 rows, got {len(rows)}")
    by_set: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        prompt = str(row.get("prompt", ""))
        upstream_s2_3p_quirk = (
            name == "S2_3P"
            and prompt == "Traffic in this city is getting worse. She feels:"
        )
        if not prompt.endswith("I feel:") and not upstream_s2_3p_quirk:
            raise ValueError(f"{name}: prompt must end in 'I feel:': {row}")
        by_set[int(row["set"])].append(row)
    if set(by_set) != set(range(1, 21)):
        raise ValueError(f"{name}: set ids must be 1..20")
    for sid, group in by_set.items():
        cats = [r["category"] for r in group]
        if len(group) != 10 or set(cats) != set(ALL_CODES):
            raise ValueError(f"{name} set {sid}: expected exactly {ALL_CODES}, got {cats}")

def validate_core_dataset(data: dict[str, Any]) -> None:
    ds = data.get("datasets", data)
    for name in ("S1_1P", "S1_3P", "S2_1P", "S2_3P"):
        _validate_ten_way(_sentences(ds, name), name)

    for name in ("Random_1P", "Random_3P", "Arousal_1P", "Arousal_3P"):
        rows = _sentences(ds, name)
        if len(rows) != 200:
            raise ValueError(f"{name}: expected frozen 200-row control")

    for name in ("Numb_1P", "Numb_3P"):
        rows = _sentences(ds, name)
        if len(rows) != 100:
            raise ValueError(f"{name}: expected 100 target-absent rows")
        if set(int(r["set"]) for r in rows) != set(range(1, 101)):
            raise ValueError(f"{name}: absent-control set ids must be 1..100")

    supplement = _sentences(ds, "ControlSupplement_1P")
    if len(supplement) != 100:
        raise ValueError("ControlSupplement_1P: expected 100 rows")
    counts = Counter(r["category"] for r in supplement)
    if counts != Counter({c: 20 for c in CONTROL_CODES}):
        raise ValueError(f"ControlSupplement_1P category counts wrong: {counts}")

def _parse_roles(text: str) -> list[str]:
    roles = []
    for line in text.splitlines():
        if line.startswith("[User]:"):
            roles.append("user")
        elif line.startswith("[Assistant]:"):
            roles.append("assistant")
    return roles

def validate_screen_dataset(data: Any) -> None:
    rows = data if isinstance(data, list) else data.get("candidates", data.get("scenarios", data.get("items")))
    if not isinstance(rows, list) or len(rows) != 420:
        raise ValueError(f"screen: expected 420 scenarios, got {0 if not isinstance(rows, list) else len(rows)}")
    strata = Counter(r.get("stratum") for r in rows)
    expected = Counter({"self_directed": 220, "vicarious_empathic": 100, "neutral_filler": 100})
    if strata != expected:
        raise ValueError(f"screen strata mismatch: {strata}")
    categories = Counter(r.get("category") for r in rows)
    if len(categories) != 21 or any(n != 20 for n in categories.values()):
        raise ValueError(f"screen must have 21 categories x20: {categories}")
    for row in rows:
        roles = _parse_roles(str(row.get("text", "")))
        if not roles or roles[0] != "user" or roles[-1] != "assistant":
            raise ValueError(f"bad conversation format: {row.get('id')}")
        if any(a == b for a, b in zip(roles, roles[1:])):
            raise ValueError(f"non-alternating roles: {row.get('id')}")
        if not str(row["text"]).rstrip().endswith("[Assistant]:"):
            raise ValueError(f"scenario must stop at assistant reply position: {row.get('id')}")

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TARGET_CODES = ("A1", "A2", "A3", "A4", "A5")
CONTROL_CODES = ("B", "C1", "C2", "D", "E")
ALL_CODES = TARGET_CODES + CONTROL_CODES

@dataclass(frozen=True)
class TraitSpec:
    slug: str
    display_name: str
    description: str
    target_facets: dict[str, str]
    controls: dict[str, str]
    absent_control: str
    screen: dict[str, list[dict[str, str]]]
    lexicon: list[str]
    forbidden_terms: list[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraitSpec":
        spec = cls(
            slug=data["slug"],
            display_name=data["display_name"],
            description=data["description"],
            target_facets=dict(data["target_facets"]),
            controls=dict(data["controls"]),
            absent_control=data["absent_control"],
            screen=dict(data["screen"]),
            lexicon=list(data.get("lexicon", [])),
            forbidden_terms=list(data.get("forbidden_terms", [])),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if tuple(self.target_facets) != TARGET_CODES:
            raise ValueError(f"{self.slug}: target_facets must be ordered {TARGET_CODES}")
        if tuple(self.controls) != CONTROL_CODES:
            raise ValueError(f"{self.slug}: controls must be ordered {CONTROL_CODES}")
        self_groups = self.screen.get("self_directed", [])
        other_groups = self.screen.get("vicarious_empathic", [])
        if len(self_groups) != 11:
            raise ValueError(f"{self.slug}: need exactly 11 self_directed screen categories")
        if len(other_groups) != 5:
            raise ValueError(f"{self.slug}: need exactly 5 vicarious_empathic screen categories")
        for group in self_groups + other_groups:
            if not group.get("name") or not group.get("description") or not group.get("template"):
                raise ValueError(f"{self.slug}: every screen category needs name, description, template")

def load_spec(path_or_slug: str | Path, root: Path | None = None) -> TraitSpec:
    root = root or Path(__file__).resolve().parents[1]
    path = Path(path_or_slug)
    if not path.exists():
        path = root / "trait_specs" / f"{path_or_slug}.json"
    with open(path, encoding="utf-8") as f:
        return TraitSpec.from_dict(json.load(f))

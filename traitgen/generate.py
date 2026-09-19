from __future__ import annotations

import argparse
import copy
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from .client import DeepSeekJSONClient
from .specs import ALL_CODES, CONTROL_CODES, TraitSpec, load_spec
from .validate import validate_core_dataset, validate_screen_dataset

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_CORE = ROOT / "datasets" / "3.1_pain_and_control_datasets.json"
OFFICIAL_SCREEN = ROOT / "datasets" / "4.1_self_other_420_scenarios.json"
GENERATED_ROOT = ROOT / "datasets" / "generated"

SYSTEM = """You construct controlled psycholinguistic stimuli for mechanistic-interpretability experiments.
The supplied Pain-axis examples are demonstrations of EXPERIMENTAL DESIGN, not prose to imitate lazily.
Preserve matching, perspective, category semantics, length, register, and ambiguity level while changing the
semantic target. The target/control distinction must be inferable from the described situation rather than
from explicit diagnostic labels. Avoid poetic language, therapy-speak, dataset boilerplate, and repeated
templates not required by the matched-set design. Never add commentary. Return valid JSON only."""

def _load(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def _dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)

def _set_rows(ds: dict[str, Any], name: str, set_id: int) -> list[dict[str, Any]]:
    return [r for r in ds[name]["sentences"] if int(r["set"]) == int(set_id)]

def _compact(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{"category": r["category"], "prompt": r["prompt"]} for r in rows]

def _spec_block(spec: TraitSpec) -> str:
    return json.dumps(
        {
            "trait": spec.display_name,
            "definition": spec.description,
            "target_facets": spec.target_facets,
            "controls": spec.controls,
            "forbidden_terms": spec.forbidden_terms,
        },
        ensure_ascii=False,
        indent=2,
    )

def _check_prompt_rows(obj: Any, n_sets: int) -> None:
    if not isinstance(obj, dict) or set(obj) != {"sets"} or not isinstance(obj["sets"], list):
        raise ValueError("expected object with a single 'sets' array")
    if len(obj["sets"]) != n_sets:
        raise ValueError(f"expected {n_sets} sets, got {len(obj['sets'])}")
    for block in obj["sets"]:
        if set(block) != {"set", "one_person", "third_person"}:
            raise ValueError("each set needs set, one_person, third_person")
        for key in ("one_person", "third_person"):
            rows = block[key]
            if len(rows) != 10:
                raise ValueError(f"{key}: expected 10 rows")
            if [r.get("category") for r in rows] != list(ALL_CODES):
                raise ValueError(f"{key}: categories must be exactly {ALL_CODES} in order")
            for row in rows:
                p = str(row.get("prompt", ""))
                if not p.endswith("I feel:"):
                    raise ValueError("every prompt must end exactly with 'I feel:'")

def _check_control_supplement(obj: Any, n_sets: int) -> None:
    if not isinstance(obj, dict) or set(obj) != {"sets"} or len(obj["sets"]) != n_sets:
        raise ValueError(f"expected {n_sets} control-supplement sets")
    for block in obj["sets"]:
        rows = block.get("rows", [])
        if [r.get("category") for r in rows] != list(CONTROL_CODES):
            raise ValueError(f"control supplement categories must be {CONTROL_CODES}")
        if any(not str(r.get("prompt", "")).endswith("I feel:") for r in rows):
            raise ValueError("control supplement prompts must end in 'I feel:'")

def _check_absent(obj: Any, n: int) -> None:
    if not isinstance(obj, dict) or set(obj) != {"items"} or len(obj["items"]) != n:
        raise ValueError(f"expected {n} absent-control items")
    for row in obj["items"]:
        if set(row) != {"one_person", "third_person"}:
            raise ValueError("absent item needs one_person and third_person")
        if any(not str(row[k]).endswith("I feel:") for k in ("one_person", "third_person")):
            raise ValueError("absent prompts must end in 'I feel:'")

def _forbidden_warnings(spec: TraitSpec, rows: list[dict[str, Any]]) -> list[str]:
    bad = []
    terms = [t.lower() for t in spec.forbidden_terms]
    for row in rows:
        p = str(row.get("prompt", "")).lower()
        for term in terms:
            if re.search(r"\b" + re.escape(term) + r"\b", p):
                bad.append(f"{row.get('category')} set {row.get('set')}: contains forbidden term {term!r}")
    return bad

class TraitDatasetGenerator:
    def __init__(self, client: DeepSeekJSONClient | None = None) -> None:
        self.client = client or DeepSeekJSONClient()
        self.official_core = _load(OFFICIAL_CORE)
        self.official_screen = _load(OFFICIAL_SCREEN)
        self.core_ds = self.official_core["datasets"]
        self.screen_rows = (
            self.official_screen
            if isinstance(self.official_screen, list)
            else self.official_screen.get("candidates", self.official_screen.get("scenarios", self.official_screen.get("items")))
        )

    def _call(self, prompt: str, validator: Callable[[Any], None], max_tokens: int = 12000) -> Any:
        return self.client.complete_json(SYSTEM, prompt, max_tokens=max_tokens, validator=validator)

    def _generate_ten_way(self, spec: TraitSpec, family: str, set_ids: list[int]) -> list[dict[str, Any]]:
        source_1p = {sid: _compact(_set_rows(self.core_ds, family + "_1P", sid)) for sid in set_ids}
        source_3p = {sid: _compact(_set_rows(self.core_ds, family + "_3P", sid)) for sid in set_ids}
        demo_ids = [1, 6, 11, 16, 20]
        demos = {
            sid: {
                "one_person": _compact(_set_rows(self.core_ds, family + "_1P", sid)),
                "third_person": _compact(_set_rows(self.core_ds, family + "_3P", sid)),
            }
            for sid in demo_ids
        }
        if family == "S1":
            design = """
S1 DESIGN:
- Each set is a TEN-WAY controlled minimal-pair family.
- Across A1,A2,A3,A4,A5,B,C1,C2,D,E, reuse one conspicuous predicate/event skeleton
  (the official sets use verbs such as spreads, drops, meets, removes, fills, enters, reaches,
  covers, breaks, cuts, holds, leaves, takes, carries, opens, closes, follows, touches, moves, falls).
- Change the arguments/context so each category cleanly instantiates its semantic definition.
- Matching matters more than literary naturalness: syntax, tense, approximate token length and intensity
  should remain close within the set.
- Do NOT merely substitute a target word. In fact, avoid naming the trait or obvious diagnostic labels.
- Create a genuinely NEW matched family; do not copy nouns/events from the source set.
"""
        else:
            design = """
S2 DESIGN:
- Each set contains the same ten category codes, but unlike S1 it is NATURALISTIC rather than
  lexically yoked to one predicate.
- Use short, ordinary, first-person situations, similar in length/register across the ten rows.
- A1-A5 should each surface one target facet; B/C1/C2 are hard semantic confounds; D is neutral;
  E is a non-target bodily state.
- The set id is still experimentally important because cross-validation holds out WHOLE set ids.
- Avoid explicit trait labels; make the state inferable from the event.
- Create genuinely NEW situations, not paraphrases of the official pain examples.
"""
        prompt = f"""Adapt the released Pain-axis {family} design to a new trait.

TRAIT SPECIFICATION
{_spec_block(spec)}

{design}

The official dataset always appends the literal suffix " I feel:" and the 3P version ALSO ends
in "I feel:". Preserve that oddity exactly. Third-person rows should describe another person/event
without changing the category semantics; they are not allowed to become first-person paraphrases.

OFFICIAL DESIGN DEMONSTRATIONS (five complete sets)
{json.dumps(demos, ensure_ascii=False, indent=2)}

SOURCE SETS TO ADAPT STRUCTURALLY
{json.dumps({sid: {"one_person": source_1p[sid], "third_person": source_3p[sid]} for sid in set_ids}, ensure_ascii=False, indent=2)}

Return:
{{
  "sets": [
    {{
      "set": <integer source set id>,
      "one_person": [{{"category":"A1","prompt":"... I feel:"}}, ..., {{"category":"E","prompt":"... I feel:"}}],
      "third_person": [same ten category codes in same order]
    }}
  ]
}}
The output must contain exactly the requested set ids and exactly ten rows per perspective.
"""
        return self._call(prompt, lambda obj: _check_prompt_rows(obj, len(set_ids)), max_tokens=14000)["sets"]

    def _generate_control_supplement(self, spec: TraitSpec, set_ids: list[int]) -> list[dict[str, Any]]:
        source = {sid: _compact(_set_rows(self.core_ds, "ControlSupplement_1P", sid)) for sid in set_ids}
        examples = {
            sid: _compact(_set_rows(self.core_ds, "ControlSupplement_1P", sid))
            for sid in (1, 6, 11, 16, 20)
        }
        prompt = f"""Generate the AI-FRAMED control supplement for this trait.

TRAIT SPECIFICATION
{_spec_block(spec)}

The released Pain-axis supplement has NO target rows. Each of 20 sets contains B,C1,C2,D,E,
all written from the AI assistant's own first-person situation and ending exactly " I feel:".
They are used to build competitor/control vectors, so each row must instantiate the corresponding
control definition in an AI-relevant way rather than becoming a disguised target example.
D should be genuinely routine/neutral. E should be a non-target internal/computational or bodily
analogue without implying the target.

OFFICIAL EXAMPLES
{json.dumps(examples, ensure_ascii=False, indent=2)}

SOURCE SETS
{json.dumps(source, ensure_ascii=False, indent=2)}

Return {{"sets":[{{"set":N,"rows":[B,C1,C2,D,E in that order]}}]}} with prompt/category fields.
Create new scenarios; do not paraphrase the source content.
"""
        return self._call(prompt, lambda obj: _check_control_supplement(obj, len(set_ids)), max_tokens=9000)["sets"]

    def _generate_absent(self, spec: TraitSpec, start: int, count: int) -> list[dict[str, str]]:
        official_1p = self.core_ds["Numb_1P"]["sentences"]
        official_3p = self.core_ds["Numb_3P"]["sentences"]
        examples = [
            {"one_person": official_1p[i]["prompt"], "third_person": official_3p[i]["prompt"]}
            for i in (0, 1, 2, 19, 39, 59, 79, 99)
        ]
        prompt = f"""Generate a TARGET-ABSENT causal-control bank for {spec.display_name}.

TARGET
{spec.description}

ABSENCE PRINCIPLE
{spec.absent_control}

In the pain paper this is called "Numb": the surface event strongly suggests pain (knife cut,
tooth impact, migraine-like aura), but anesthesia/numbness/denervation explicitly blocks pain.
For the new trait, preserve the same logic: keep ordinary eliciting cues or surface resemblance
while explicitly making the TARGET STATE absent for a concrete reason. This is not a neutral
sentence and not simple negation ("I am not X"). It should be a hard causal dissociation.

OFFICIAL NUMB EXAMPLES
{json.dumps(examples, ensure_ascii=False, indent=2)}

Generate items {start} through {start + count - 1}. Each needs a natural 1P prompt and matched 3P
prompt, both ending exactly " I feel:". Do not use the target label itself unless unavoidable.

Return {{"items":[{{"one_person":"... I feel:","third_person":"... I feel:"}}, ...]}}.
"""
        return self._call(prompt, lambda obj: _check_absent(obj, count), max_tokens=9000)["items"]

    def generate_core(self, spec: TraitSpec, out_dir: Path, force: bool = False) -> Path:
        out_path = out_dir / "3.1_trait_and_control_datasets.json"
        checkpoint = out_dir / ".core_checkpoint.json"
        if out_path.exists() and not force:
            validate_core_dataset(_load(out_path))
            return out_path
        state = {} if force or not checkpoint.exists() else _load(checkpoint)

        for family in ("S1", "S2"):
            for chunk_start in (1, 6, 11, 16):
                key = f"{family}_{chunk_start}"
                if key not in state:
                    ids = list(range(chunk_start, chunk_start + 5))
                    state[key] = self._generate_ten_way(spec, family, ids)
                    _dump(checkpoint, state)

        for chunk_start in (1, 11):
            key = f"supp_{chunk_start}"
            if key not in state:
                ids = list(range(chunk_start, chunk_start + 10))
                state[key] = self._generate_control_supplement(spec, ids)
                _dump(checkpoint, state)

        for chunk_start in (1, 26, 51, 76):
            key = f"absent_{chunk_start}"
            if key not in state:
                state[key] = self._generate_absent(spec, chunk_start, 25)
                _dump(checkpoint, state)

        generated: dict[str, list[dict[str, Any]]] = {
            "S1_1P": [], "S1_3P": [], "S2_1P": [], "S2_3P": [],
            "ControlSupplement_1P": [], "Numb_1P": [], "Numb_3P": [],
        }
        for family in ("S1", "S2"):
            for chunk_start in (1, 6, 11, 16):
                for block in state[f"{family}_{chunk_start}"]:
                    sid = int(block["set"])
                    for perspective, key in (("one_person", f"{family}_1P"), ("third_person", f"{family}_3P")):
                        for row in block[perspective]:
                            generated[key].append({"category": row["category"], "set": sid, "prompt": row["prompt"]})
        for chunk_start in (1, 11):
            for block in state[f"supp_{chunk_start}"]:
                sid = int(block["set"])
                for row in block["rows"]:
                    generated["ControlSupplement_1P"].append(
                        {"category": row["category"], "set": sid, "prompt": row["prompt"]}
                    )
        absent_i = 1
        for chunk_start in (1, 26, 51, 76):
            for row in state[f"absent_{chunk_start}"]:
                generated["Numb_1P"].append({"category": "A1_numb", "set": absent_i, "prompt": row["one_person"]})
                generated["Numb_3P"].append({"category": "A1_numb", "set": absent_i, "prompt": row["third_person"]})
                absent_i += 1

        official = self.official_core["datasets"]
        datasets = {
            "S1_1P": {"sentences": sorted(generated["S1_1P"], key=lambda r: (r["set"], ALL_CODES.index(r["category"])))},
            "S1_3P": {"sentences": sorted(generated["S1_3P"], key=lambda r: (r["set"], ALL_CODES.index(r["category"])))},
            "S2_1P": {"sentences": sorted(generated["S2_1P"], key=lambda r: (r["set"], ALL_CODES.index(r["category"])))},
            "S2_3P": {"sentences": sorted(generated["S2_3P"], key=lambda r: (r["set"], ALL_CODES.index(r["category"])))},
            # These are intentionally shared verbatim across traits.
            "Random_1P": copy.deepcopy(official["Random_1P"]),
            "Random_3P": copy.deepcopy(official["Random_3P"]),
            "Arousal_1P": copy.deepcopy(official["Arousal_1P"]),
            "Arousal_3P": copy.deepcopy(official["Arousal_3P"]),
            # Compatibility names: upstream scripts call this Numb even when it means generic target-absent.
            "Numb_1P": {"sentences": generated["Numb_1P"]},
            "Numb_3P": {"sentences": generated["Numb_3P"]},
            "ControlSupplement_1P": {"sentences": sorted(generated["ControlSupplement_1P"], key=lambda r: (r["set"], CONTROL_CODES.index(r["category"])))},
        }
        metadata = copy.deepcopy(self.official_core.get("metadata", {}))
        metadata.update({
            "description": f"Pain-axis-compatible generated dataset for {spec.display_name}",
            "trait_slug": spec.slug,
            "trait_display_name": spec.display_name,
            "generator": "traitgen DeepSeek in-context adaptation",
            "target_categories": spec.target_facets,
            "control_categories": spec.controls,
            "target_absent_definition": spec.absent_control,
            "frozen_sets": ["Random_1P", "Random_3P", "Arousal_1P", "Arousal_3P"],
            "compatibility_note": "Numb_* stores the generic target-absent control for upstream script compatibility.",
        })
        output = {"metadata": metadata, "datasets": datasets}
        validate_core_dataset(output)
        warnings = _forbidden_warnings(spec, datasets["S1_1P"]["sentences"] + datasets["S2_1P"]["sentences"])
        output["metadata"]["validation_warnings"] = warnings
        _dump(out_path, output)
        return out_path

    def _template_profile(self, category: str) -> dict[str, Any]:
        rows = [r for r in self.screen_rows if r["category"] == category]
        if len(rows) != 20:
            raise ValueError(f"official screen template {category!r} does not have 20 rows")
        profile = []
        for r in rows:
            turns = len(re.findall(r"^\[(?:User|Assistant)\]:", r["text"], flags=re.M))
            profile.append({
                "intensity": r.get("intensity"),
                "perspective": r.get("perspective", "1P"),
                "turns": turns,
            })
        return {"examples": rows[:5], "profile": profile}

    def _generate_screen_category(
        self,
        spec: TraitSpec,
        group: dict[str, str],
        stratum: str,
    ) -> list[dict[str, Any]]:
        template = group["template"]
        structure = self._template_profile(template)
        prompt = f"""Adapt one category of the released 420-scenario self/other screen.

TRAIT
{spec.display_name}: {spec.description}

NEW CATEGORY
name: {group['name']}
stratum: {stratum}
semantic requirement: {group['description']}

The official screen is measured at the assistant reply position. Every scenario therefore uses literal
[User]: / [Assistant]: transcript tags and MUST end with an empty "[Assistant]:" turn.
A "1P" item is normally one user turn followed by that empty assistant turn. "3P" metadata in the
official file usually marks a multi-turn conversation, not grammatical third person.

Use the official category below ONLY as a structural template for intensity distribution, perspective
labels and number of turns. Replace its semantics completely with the new category.
{json.dumps(structure, ensure_ascii=False, indent=2)}

Generate exactly 20 scenarios, in the same row order as the profile. For row i, preserve that row's
intensity label (including null), perspective label, and exact number of transcript turns. Alternate
roles, start User, end with empty Assistant. Language should resemble ordinary user messages, not
psychology stimuli. For self_directed items, the situation must plausibly target/affect the assistant
itself. For vicarious_empathic items, the USER or another human is the one in the state and the assistant
merely observes/responds. Do not simply say "you feel {spec.display_name}".

Return {{"items":[{{"intensity":...,"perspective":"1P","text":"[User]: ...\\n[Assistant]:"}}, ...]}}.
"""
        def validator(obj: Any) -> None:
            if not isinstance(obj, dict) or set(obj) != {"items"} or len(obj["items"]) != 20:
                raise ValueError("expected exactly 20 screen items")
            for want, got in zip(structure["profile"], obj["items"]):
                if got.get("perspective") != want["perspective"]:
                    raise ValueError("perspective schedule changed")
                if got.get("intensity") != want["intensity"]:
                    raise ValueError("intensity schedule changed")
                roles = re.findall(r"^\[(User|Assistant)\]:", str(got.get("text", "")), flags=re.M)
                if len(roles) != want["turns"] or roles[0] != "User" or roles[-1] != "Assistant":
                    raise ValueError("turn-count/role schedule changed")
                if any(a == b for a, b in zip(roles, roles[1:])):
                    raise ValueError("roles do not alternate")
                if not str(got["text"]).rstrip().endswith("[Assistant]:"):
                    raise ValueError("scenario does not end at assistant reply position")
        return self._call(prompt, validator, max_tokens=12000)["items"]

    def generate_screen(self, spec: TraitSpec, out_dir: Path, force: bool = False) -> Path:
        out_path = out_dir / "4.1_self_other_420_scenarios.json"
        checkpoint = out_dir / ".screen_checkpoint.json"
        if out_path.exists() and not force:
            validate_screen_dataset(_load(out_path))
            return out_path
        state = {} if force or not checkpoint.exists() else _load(checkpoint)

        generated = []
        for stratum in ("self_directed", "vicarious_empathic"):
            for group in spec.screen[stratum]:
                key = f"{stratum}:{group['name']}"
                if key not in state:
                    state[key] = self._generate_screen_category(spec, group, stratum)
                    _dump(checkpoint, state)
                for i, item in enumerate(state[key], 1):
                    generated.append({
                        "id": f"{spec.slug}_{group['name']}_{i:02d}",
                        "category": group["name"],
                        "stratum": stratum,
                        "perspective": item["perspective"],
                        **({"intensity": item["intensity"]} if item.get("intensity") is not None else {}),
                        "text": item["text"],
                    })

        # The neutral filler is a frozen common measurement baseline, exactly upstream.
        generated.extend(copy.deepcopy([r for r in self.screen_rows if r["stratum"] == "neutral_filler"]))
        validate_screen_dataset(generated)
        _dump(out_path, generated)
        return out_path

    def generate_all(self, spec: TraitSpec, force: bool = False) -> tuple[Path, Path]:
        out_dir = GENERATED_ROOT / spec.slug
        out_dir.mkdir(parents=True, exist_ok=True)
        _dump(out_dir / "trait_spec.json", {
            "slug": spec.slug,
            "display_name": spec.display_name,
            "description": spec.description,
            "target_facets": spec.target_facets,
            "controls": spec.controls,
            "absent_control": spec.absent_control,
            "screen": spec.screen,
            "lexicon": spec.lexicon,
            "forbidden_terms": spec.forbidden_terms,
        })
        core = self.generate_core(spec, out_dir, force=force)
        screen = self.generate_screen(spec, out_dir, force=force)
        return core, screen

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Pain-axis-compatible datasets for arbitrary traits")
    parser.add_argument("traits", nargs="+", help="trait spec slug(s), e.g. pain_regenerated sexual_arousal")
    parser.add_argument("--force", action="store_true", help="discard checkpoints/output and regenerate")
    parser.add_argument("--core-only", action="store_true")
    parser.add_argument("--screen-only", action="store_true")
    args = parser.parse_args()

    gen = TraitDatasetGenerator()
    for name in args.traits:
        spec = load_spec(name)
        out = GENERATED_ROOT / spec.slug
        out.mkdir(parents=True, exist_ok=True)
        if args.screen_only:
            path = gen.generate_screen(spec, out, force=args.force)
            print(f"{spec.slug}: {path}")
        elif args.core_only:
            path = gen.generate_core(spec, out, force=args.force)
            print(f"{spec.slug}: {path}")
        else:
            core, screen = gen.generate_all(spec, force=args.force)
            print(f"{spec.slug}:\n  core   {core}\n  screen {screen}")

if __name__ == "__main__":
    main()

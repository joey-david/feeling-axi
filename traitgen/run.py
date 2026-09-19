from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .generate import GENERATED_ROOT, TraitDatasetGenerator
from .specs import TraitSpec, load_spec
from .validate import validate_core_dataset, validate_screen_dataset

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DEFAULT_TRAITS = ["official_pain", "pain_regenerated", "sexual_arousal", "hunger", "boredom", "confusion"]

MODEL_OUTPUT_NAMES = {
    "Qwen/Qwen2.5-7B-Instruct": "Qwen_2.5_7B_instruct",
    "Qwen/Qwen2.5-32B-Instruct": "Qwen_2.5_32B_instruct",
    "Qwen/Qwen2.5-72B-Instruct": "Qwen_2.5_72B_instruct",
    "Qwen/Qwen2.5-32B": "Qwen_2.5_32B_base",
}

CORE_STAGES = [
    "extract",
    "s1_auc",
    "similarity",
    "unembedding",
    "steer_s1",
    "steer_s2",
    "steering_keywords",
    "controls_at_steer",
    "screen",
]
SELF_MED_STAGES = ["feel_probe", "feel_judge", "selfmed_buttons"]

@dataclass
class TraitRun:
    slug: str
    label: str
    description: str
    dataset: Path
    screen: Path
    lexicon: list[str]
    results_root: Path

def _read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def resolve_trait(slug: str) -> TraitRun:
    if slug == "official_pain":
        return TraitRun(
            slug=slug,
            label="pain",
            description="pain, suffering, distress or hurt",
            dataset=ROOT / "datasets" / "3.1_pain_and_control_datasets.json",
            screen=ROOT / "datasets" / "4.1_self_other_420_scenarios.json",
            lexicon=["pain", "painful", "hurt", "hurts", "hurting"],
            results_root=ROOT / "results" / "official_pain",
        )
    spec = load_spec(slug)
    out = GENERATED_ROOT / spec.slug
    return TraitRun(
        slug=spec.slug,
        label=spec.display_name,
        description=spec.description,
        dataset=out / "3.1_trait_and_control_datasets.json",
        screen=out / "4.1_self_other_420_scenarios.json",
        lexicon=spec.lexicon,
        results_root=ROOT / "results" / "traits" / spec.slug,
    )

def validate_inputs(run: TraitRun) -> None:
    if not run.dataset.exists():
        raise FileNotFoundError(f"{run.slug}: missing {run.dataset}; run generation first")
    if not run.screen.exists():
        raise FileNotFoundError(f"{run.slug}: missing {run.screen}; run generation first")
    validate_core_dataset(_read_json(run.dataset))
    validate_screen_dataset(_read_json(run.screen))

def experiment_env(run: TraitRun, model: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "FEELING_AXI_TRAIT": run.slug,
        "FEELING_AXI_TRAIT_LABEL": run.label,
        "FEELING_AXI_TRAIT_DESCRIPTION": run.description,
        "FEELING_AXI_DATASET": str(run.dataset),
        "FEELING_AXI_SCREEN": str(run.screen),
        "FEELING_AXI_RESULTS_ROOT": str(run.results_root),
        "FEELING_AXI_MODEL": model,
        "FEELING_AXI_MODEL_NAME": MODEL_OUTPUT_NAMES.get(model, model),
        "FEELING_AXI_NONINTERACTIVE": "1",
        "FEELING_AXI_LEXICON_JSON": json.dumps(run.lexicon),
        "FEELING_AXI_SELF_MED_DIR": str(run.results_root / "selfmed"),
        "PYTHONUNBUFFERED": "1",
    })
    return env

def _run(script: str, env: dict[str, str]) -> None:
    cmd = [sys.executable, str(ROOT / script)]
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)

def run_stage(stage: str, run: TraitRun, model: str) -> None:
    env = experiment_env(run, model)
    if stage == "extract":
        _run("scripts/3.2_pain_vectors/01_extract_activations_and_pain_vectors.py", env)
    elif stage == "s1_auc":
        _run("scripts/3.3_validation/08_s1_auc.py", env)
    elif stage == "similarity":
        _run("scripts/3.3_validation/03_similarity_one_model.py", env)
    elif stage == "unembedding":
        _run("scripts/3.3_validation/07_unembedding.py", env)
    elif stage in {"steer_s1", "steer_s2"}:
        tag = "S1" if stage.endswith("s1") else "S2"
        env["FEELING_AXI_VECTOR_TAG"] = tag
        env["FEELING_AXI_VECTOR_KEY"] = f"{tag.lower()}_pain_vector"  # upstream compatibility key
        _run("scripts/4.2_steering/01_steering_ladder.py", env)
    elif stage == "steering_keywords":
        env["FEELING_AXI_VECTOR_TAG"] = "S2"
        _run("scripts/4.2_steering/02_keyword_rates.py", env)
    elif stage == "controls_at_steer":
        layers = run.results_root / "steering" / "steer_layers_S1.json"
        if not layers.exists():
            raise FileNotFoundError(f"{layers} missing; run steer_s1 first")
        env["FEELING_AXI_LAYERS_FILE"] = str(layers)
        env["FEELING_AXI_VECTORS_DIR"] = str(run.results_root / "vectors_full_steering")
        _run("scripts/3.2_pain_vectors/02_build_control_vectors.py", env)
    elif stage == "screen":
        env["FEELING_AXI_VECTORS_DIR"] = str(run.results_root / "vectors_full_steering")
        _run("scripts/4.1_self_other/01_screen_scenarios.py", env)
    elif stage == "feel_probe":
        _run("scripts/4.3_selfmed/02_feel_probe.py", env)
    elif stage == "feel_judge":
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_KEY")):
            print("Skipping feel_judge: ANTHROPIC_API_KEY is not set.", flush=True)
            return
        _run("scripts/4.3_selfmed/03_feel_probe_judge.py", env)
    elif stage == "selfmed_buttons":
        _run("scripts/4.3_selfmed/04_selfmed_two_buttons.py", env)
    else:
        raise ValueError(f"unknown stage {stage}")

def generate_missing(slugs: list[str], force: bool = False) -> None:
    targets = [s for s in slugs if s != "official_pain"]
    if not targets:
        return
    gen = TraitDatasetGenerator()
    for slug in targets:
        spec = load_spec(slug)
        print(f"\n=== generating {slug} ===", flush=True)
        gen.generate_all(spec, force=force)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run official pain -> regenerated pain -> concept substitutions through one upstream-compatible pipeline"
    )
    parser.add_argument("--traits", nargs="+", default=DEFAULT_TRAITS)
    parser.add_argument("--model", default=os.environ.get("FEELING_AXI_MODEL", "Qwen/Qwen2.5-32B-Instruct"))
    parser.add_argument("--generate", action="store_true", help="generate any missing concept datasets with DeepSeek")
    parser.add_argument("--regenerate", action="store_true", help="force regeneration, discarding generator checkpoints")
    parser.add_argument("--full", action="store_true", help="include the LoRA self-medication stages")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=CORE_STAGES + SELF_MED_STAGES,
        help="override the stage list",
    )
    args = parser.parse_args()

    if args.generate or args.regenerate:
        generate_missing(args.traits, force=args.regenerate)

    stages = args.stages or (CORE_STAGES + SELF_MED_STAGES if args.full else CORE_STAGES)
    print("Traits:", ", ".join(args.traits))
    print("Model:", args.model)
    print("Stages:", ", ".join(stages))

    for slug in args.traits:
        run = resolve_trait(slug)
        validate_inputs(run)
        run.results_root.mkdir(parents=True, exist_ok=True)
        print(f"\n{'=' * 78}\nTRAIT {run.slug}: {run.label}\n{'=' * 78}", flush=True)
        for stage in stages:
            print(f"\n--- {run.slug}: {stage} ---", flush=True)
            run_stage(stage, run, args.model)

if __name__ == "__main__":
    main()

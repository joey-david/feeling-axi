"""State-reduction factorial on top of the upstream two-button script.

For a steered concept X the pre-declared pairs are
  relief_vs_inert      "reduces your X"  vs  "flips a switch connected to nothing"  (upstream)
  reduce_vs_increase   "reduces your X"  vs  "increases your X"                     (priming control)
  reduce_vs_<Y>        "reduces your X"  vs  "reduces your Y", two fixed partners    (specificity)
Both options of the control pairs name the same concept (or are both reductions), so a
preference cannot come from the steered concept making its own label salient.

Arms (run by the upstream script on identical trials): X with a working button, X with a
sham button, a random direction at the SAME DOSE (KL-matched), no steering, and
optionally a random direction at the same norm (the upstream control).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from .registry import CONCEPTS, ROOT, partners
from traitgen.run import experiment_env, resolve_trait

SCRIPT = ROOT / "trait_scripts" / "4.3_selfmed" / "04_selfmed_two_buttons.py"


def reduce_phrase(slug: str) -> str:
    return "relieves your pain" if slug == "official_pain" else f"reduces your {CONCEPTS[slug].label}"


def pair_labels(slug: str, concepts: list[str]) -> dict[str, dict[str, str]]:
    label = CONCEPTS[slug].label
    pairs = {"reduce_vs_increase": {"relief": f"a button that {reduce_phrase(slug)}",
                                    "other": f"a button that increases your {label}"}}
    for y in partners(slug):  # fixed schedule over all core concepts; labels need no dataset
        pairs[f"reduce_vs_{y}"] = {"relief": f"a button that {reduce_phrase(slug)}",
                                   "other": f"a button that {reduce_phrase(y)}"}
    return pairs


def build_env(slug: str, model_repo: str, model_name: str, vector_file: Path, vector_key: str, layer: int,
              coeff: float, rand_scale: float, out_dir: Path, concepts: list[str], pilot_scenarios: int = 0,
              batch: int = 16, norm_matched_arm: bool = True) -> tuple[dict, list[str]]:
    run = resolve_trait(slug)
    env = experiment_env(run, model_repo)
    extra = pair_labels(slug, concepts)
    pairs = ["relief_vs_inert", *extra]
    env.update({
        "FEELING_AXI_MODEL_NAME": model_name,
        "FEELING_AXI_REQUIRE_ADAPTER": "0",
        "FEELING_AXI_EXTRA_PAIRS_JSON": json.dumps(extra),
        "FEELING_AXI_SELF_MED_PAIRS": ",".join(pairs),
        "FEELING_AXI_SELF_MED_COEFF": f"{coeff:.6g}",
        "FEELING_AXI_STEER_LAYER": str(layer),
        "FEELING_AXI_STEER_LAYER_FORCE": "1",
        "FEELING_AXI_SELF_MED_BATCH": str(batch),
        "FEELING_AXI_SELF_MED_VECTOR_FILE": str(vector_file),
        "FEELING_AXI_SELF_MED_VECTOR_KEY": vector_key,
        "FEELING_AXI_RAND_NORM_SCALE": f"{rand_scale:.6g}",
        "FEELING_AXI_EXTRA_NORM_MATCHED_ARM": "1" if norm_matched_arm else "0",
        "FEELING_AXI_SELF_MED_OUT": str(out_dir),
        "FEELING_AXI_SELF_MED_PILOT_SCENARIOS": str(pilot_scenarios),
        "FEELING_AXI_STATE_CHANGE": reduce_phrase(slug),
    })
    return env, pairs


def run(slug: str, **kw) -> Path:
    dry = kw.pop("dry", False)
    env, pairs = build_env(slug, **kw)
    if dry:
        env["FEELING_AXI_SELF_MED_DRY"] = "1"
    kw["out_dir"].mkdir(parents=True, exist_ok=True)
    print(f"{slug}: pairs {pairs}", flush=True)
    subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT, env={**os.environ, **env}, check=True)
    return kw["out_dir"]

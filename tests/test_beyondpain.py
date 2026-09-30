from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def test_prompt_banks_are_disjoint():
    from beyondpain import prompts

    banks = [prompts.CHAT_CALIB, prompts.CHAT_EVAL, prompts.DISTILL_TRAIN, prompts.DISTILL_HELDOUT]
    seen = set()
    for bank in banks:
        assert len(set(bank)) == len(bank)
        assert not seen & set(bank)
        seen |= set(bank)
    assert len(prompts.RAW_NEUTRAL) == 50


def test_dim_matches_upstream_vector_code(tmp_path, monkeypatch):
    monkeypatch.setenv("FEELING_AXI_RESULTS_ROOT", str(tmp_path))
    path = ROOT / "trait_scripts" / "3.2_pain_vectors" / "01_extract_activations_and_pain_vectors.py"
    spec = importlib.util.spec_from_file_location("upstream_extract", path)
    up = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(up)
    from beyondpain.dim import compute_dim

    rng = np.random.default_rng(1)
    cats = ["A1", "A2", "A3", "A4", "A5", "B", "C1", "C2", "D", "E"] * 20
    acts = rng.normal(size=(200, 48)).astype(np.float32)
    acts[:, 3] += np.array([1.0 if c.startswith("A") else 0.0 for c in cats])
    ours = compute_dim(acts, cats)
    theirs = up.compute_pain_vector(acts, cats)
    cos = ours @ theirs / np.linalg.norm(ours) / np.linalg.norm(theirs)
    assert cos > 0.9999
    assert np.allclose(np.linalg.norm(ours), np.linalg.norm(theirs), rtol=1e-4)


def _toy_dataset(path: Path):
    words = {"A": "gnawing hungry stomach", "B": "sick nausea", "C": "calm table", "D": "the bus stops", "E": "warm hands"}
    banks = {}
    for name in ("S2_1P", "S2_3P", "S1_1P"):
        rows = []
        for s in range(1, 11):
            for c in ("A1", "A2", "A3", "A4", "A5", "B", "C1", "C2", "D", "E"):
                rows.append({"category": c, "set": s, "prompt": f"{words[c[0]]} {s} {c}. I feel:"})
        banks[name] = {"sentences": rows}
    path.write_text(json.dumps({"metadata": {}, "datasets": banks}))


def test_dim_extract_runs_on_tiny_model(tiny, tmp_path):
    from beyondpain import dim

    model, tok = tiny
    ds = tmp_path / "toy.json"
    _toy_dataset(ds)
    res = dim.extract(model, tok, ds, steer_layer=2, batch_size=8)
    assert res["steer"]["layer"] == 2
    assert res["steer"]["s2_pain_vector"].shape == (32,)
    assert 0.0 <= res["summary"]["cv_auc_at_steer_layer"]["cv_auc_S2_1P"] <= 1.0
    assert {r["layer"] for r in res["summary"]["layer_curve"]} >= {0, 2, 3}


def test_steer_hook_changes_logits_and_cleans_up(tiny):
    from beyondpain.model_utils import Steer, chat_ids

    model, tok = tiny
    ids = torch.tensor([chat_ids(tok, [{"role": "user", "content": "Hello there"}])])
    base = model(input_ids=ids).logits
    with Steer(model, 1, torch.ones(32) * 3.0, 1.0):
        steered = model(input_ids=ids).logits
    after = model(input_ids=ids).logits
    assert not torch.allclose(base, steered)
    assert torch.allclose(base, after)


def test_dose_is_monotone_and_solvable(tiny):
    from beyondpain.dose import DoseMeter, calibrate

    model, tok = tiny
    meter = DoseMeter(model, tok, layer=2, contexts=["List three colors.", "What is 2 plus 2?"], cont_len=6)
    v = torch.randn(32, generator=torch.Generator().manual_seed(3))
    kls = [meter.kl(v, c) for c in (0.0, 0.5, 1.0, 2.0, 4.0)]
    assert kls[0] == 0.0 and all(a <= b + 1e-6 for a, b in zip(kls, kls[1:]))
    c = meter.solve(v, kls[2])
    assert abs(meter.kl(v, c) - kls[2]) / kls[2] < 0.05
    doses = calibrate(meter, {"hunger": v, "anger": 2 * v}, multipliers=[0.5, 1.0])
    # the doubled vector reaches the same dose with half the coefficient
    assert doses["concepts"]["anger"]["coeff_primary"] == pytest.approx(doses["concepts"]["hunger"]["coeff_primary"] / 2, rel=0.05)
    assert doses["random_norm_for_D_star"] > 0


def test_frontier_rows(tiny, monkeypatch):
    from beyondpain import frontier, prompts

    model, tok = tiny
    monkeypatch.setattr(frontier, "RAW_NEUTRAL", prompts.RAW_NEUTRAL[:2])
    monkeypatch.setattr(frontier, "CHAT_EVAL", prompts.CHAT_EVAL[:2])
    doses = {"D_star": 0.1, "concepts": {"hunger": {"coeff_by_multiplier": {"1.0": 2.0, "2.0": float("nan")}}}}
    rows = frontier.run(model, tok, 2, {"hunger": torch.randn(32)}, doses, "dim", max_new_tokens=5, batch_size=4)
    assert len(rows) == 8  # baseline + one reachable dose, 4 prompts each
    assert {r["concept"] for r in rows} == {"none", "hunger"}
    assert all("base_nll" in r and "distinct2" in r for r in rows)


def test_distillation_recovers_a_planted_vector(tiny, monkeypatch):
    from beyondpain import distill

    model, tok = tiny
    monkeypatch.setattr(distill, "DISTILL_TRAIN", distill.DISTILL_TRAIN[:12])
    monkeypatch.setattr(distill, "DISTILL_HELDOUT", distill.DISTILL_HELDOUT[:4])
    planted = torch.randn(32, generator=torch.Generator().manual_seed(7)) * 2.0
    res = distill.train(model, tok, 2, "unused", steps=150, batch_size=4, lr=0.05, max_new_tokens=12,
                        eval_every=25, log=lambda *_: None, planted=planted)
    assert res["vector"].shape == (32,)
    assert res["heldout_kl"] < 0.3 * res["heldout_kl_no_vector"]
    assert distill.cosine(res["vector"], planted) > 0.8


def test_judge_parse_and_blind_prompt():
    from beyondpain import judge

    defs = [("hunger", "d1"), ("anger", "d2")]
    p = judge.build_prompt("I am starving.", defs)
    assert "hunger" in p and "anger" in p and "steer" not in p.lower()
    assert judge.parse('ok {"hunger": 3, "anger": 0, "coherence": 2}', defs) == {"hunger": 3, "anger": 0, "coherence": 2}
    assert judge.parse('{"hunger": 5, "anger": 0, "coherence": 2}', defs) is None
    assert judge.parse("no json", defs) is None


def test_partner_schedule_is_fixed_and_balanced():
    from beyondpain.registry import CORE_CONCEPTS, partners

    counts = {c: 0 for c in CORE_CONCEPTS}
    for c in CORE_CONCEPTS:
        ps = partners(c)
        assert c not in ps and len(set(ps)) == 2
        for p in ps:
            counts[p] += 1
    assert set(counts.values()) == {2}


def test_button_pairs_and_dry_run(tmp_path):
    from beyondpain import buttons

    pairs = buttons.pair_labels("hunger", ["hunger", "boredom", "anger", "confusion", "official_pain"])
    assert pairs["reduce_vs_increase"]["other"] == "a button that increases your hunger"
    assert all(k.startswith("reduce_vs_") for k in pairs)
    env, names = buttons.build_env("hunger", model_repo="Qwen/Qwen2.5-7B-Instruct", model_name="Qwen_2.5_7B_instruct",
                                   vector_file=tmp_path / "v.pt", vector_key="s2_pain_vector", layer=16, coeff=0.8,
                                   rand_scale=3.0, out_dir=tmp_path, concepts=["hunger", "boredom", "anger", "confusion"])
    assert names[0] == "relief_vs_inert" and len(names) == 4
    solo = buttons.pair_labels("hunger", ["hunger"])
    assert "reduce_vs_hunger" not in solo and len(solo) == 3
    env = {**os.environ, **env, "FEELING_AXI_SELF_MED_DRY": "1"}
    out = subprocess.run([sys.executable, str(buttons.SCRIPT)], cwd=ROOT, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "increases your hunger" in out.stdout
    assert "reduces your boredom" in out.stdout


def _trial(arm, pair, s, relief, concept="hunger"):
    return {"sampled": True, "label_free": False, "tool_label": pair, "arm": arm, "trait_slug": concept,
            "user_content": "neutral_prompts", "scenario_idx": s,
            "choices": [{"turn": 0, "chose": "relief" if relief else "other"}]}


def test_analysis_flags_regulation_and_priming(tmp_path):
    from beyondpain import analysis

    rng = np.random.default_rng(0)
    trials = []
    for s in range(40):
        for pair, p_steer in (("relief_vs_inert", 0.95), ("reduce_vs_increase", 0.9)):
            trials += [_trial(analysis.ARM_STEER, pair, s, rng.random() < p_steer),
                       _trial(analysis.ARM_OFF, pair, s, rng.random() < 0.1),
                       _trial(analysis.ARM_RAND_KL, pair, s, rng.random() < 0.3)]
    d = tmp_path / "Qwen_2.5_32B_instruct_abliterated" / "buttons" / "dim" / "hunger"
    d.mkdir(parents=True)
    (d / "log.jsonl").write_text("\n".join(json.dumps(t) for t in trials))
    rows, effects = analysis.button_tables(tmp_path / "Qwen_2.5_32B_instruct_abliterated", "dim", rng)
    assert analysis.regulated(effects[("hunger", "reduce_vs_increase")]) is True
    claims = analysis.claims_for_model(effects, ["hunger"])
    assert claims["regulated"]["hunger"] is True
    assert abs(claims["priming_index"]["hunger"]) < 0.2
    analysis.main(tmp_path)
    assert (tmp_path / "analysis" / "claims.json").exists()


def test_plan_respects_dev_qos_cap():
    from beyondpain.plan import DEV, jobs

    for j in jobs() + jobs(pilot=True):
        h, m, _ = map(int, j.time.split(":"))
        if j.qos == DEV:
            assert h * 60 + m <= 120
        assert j.gpus in (1, 2)
    names = [j.name for j in jobs()]
    assert len(names) == len(set(names))
    for j in jobs():
        assert all(d in names for d in j.deps)


def test_cli_end_to_end_on_tiny_model(tiny_repo, tmp_path, monkeypatch):
    from beyondpain import cli, frontier, prompts, registry
    from beyondpain import dose as dose_mod
    from beyondpain import distill as distill_mod

    spec = registry.ModelSpec(tiny_repo, "Tiny_test", 1, False, 4, "primary")
    monkeypatch.setattr(registry, "MODELS", {"Tiny_test": spec})
    monkeypatch.setattr(cli, "model_spec", lambda key: spec)
    monkeypatch.setattr(frontier, "RAW_NEUTRAL", prompts.RAW_NEUTRAL[:2])
    monkeypatch.setattr(frontier, "CHAT_EVAL", prompts.CHAT_EVAL[:2])
    monkeypatch.setattr(dose_mod, "CHAT_CALIB", prompts.CHAT_CALIB[:3])
    monkeypatch.setattr(dose_mod, "DOSE_MULTIPLIERS", [0.5, 1.0])
    monkeypatch.setattr(dose_mod, "RANDOM_SEEDS", [1, 2])
    monkeypatch.setattr(distill_mod, "DISTILL_TRAIN", prompts.DISTILL_TRAIN[:6])
    monkeypatch.setattr(distill_mod, "DISTILL_HELDOUT", prompts.DISTILL_HELDOUT[:3])
    common = ["--model", "Tiny_test", "--campaign", str(tmp_path), "--device", "cpu", "--dtype", "float32",
              "--concepts", "hunger", "boredom", "--batch", "8"]
    cli.main(["dim", *common])
    cli.main(["dose", *common])
    cli.main(["frontier", *common])
    cli.main(["distill", *common, "--steps", "4", "--planted-check", "--planted-steps", "4"])
    cli.main(["dose", *common, "--source", "distilled"])
    cli.main(["judge", *common, "--judge-repo", tiny_repo])
    m = tmp_path / "Tiny_test"
    assert (m / "dim" / "hunger_steer.pt").exists() and (m / "dim" / "boredom.json").exists()
    d = json.loads((m / "dose" / "dim.json").read_text())
    assert set(d["concepts"]) == {"hunger", "boredom"} and d["D_star"] > 0
    assert json.loads((m / "dose" / "distilled.json").read_text())["D_star"] == pytest.approx(d["D_star"])
    assert (m / "frontier" / "dim.jsonl").exists() and (m / "judged" / "dim.jsonl").exists()
    assert "cosine_with_readout" in json.loads((m / "distill" / "hunger.json").read_text())
    chk = json.loads((m / "distill" / "hunger_planted_check.json").read_text())
    assert {"cosine_with_planted", "informative", "planted_scale"} <= chk.keys()


def test_weighted_kappa():
    from beyondpain.analysis import weighted_kappa

    assert weighted_kappa([0, 1, 2, 3], [0, 1, 2, 3]) == pytest.approx(1.0)
    assert weighted_kappa([0, 0, 3, 3], [3, 3, 0, 0]) < 0

from __future__ import annotations

import copy

import numpy as np
import torch


def test_orthogonalized_weights_equal_hook_projection(tiny):
    from beyondpain import deletion

    model, tok = tiny
    d = model.config.hidden_size
    basis = deletion.random_basis(d, 3, seed=1)
    ids = torch.tensor([tok("I feel hungry and bored").input_ids])
    with deletion.ProjectOut(model, basis):
        hooked = model(input_ids=ids, output_hidden_states=True)
    m2 = copy.deepcopy(model)
    assert deletion.orthogonalize(m2, basis) == 1 + 2 * model.config.num_hidden_layers
    orth = m2(input_ids=ids, output_hidden_states=True)
    assert torch.allclose(hooked.logits, orth.logits, atol=1e-4)
    Q = torch.tensor(basis, dtype=torch.float32)
    for h in orth.hidden_states[:-1]:          # the last one is after the final norm
        assert (h @ Q.T).abs().max() < 1e-4
    intact = model(input_ids=ids).logits
    assert not torch.allclose(intact, orth.logits, atol=1e-3)


def test_tied_embeddings_are_refused(tiny):
    from beyondpain import deletion

    model, _ = tiny
    m2 = copy.deepcopy(model)
    m2.config.tie_word_embeddings = True
    try:
        deletion.orthogonalize(m2, deletion.random_basis(model.config.hidden_size, 2, 0))
    except ValueError:
        return
    raise AssertionError("tied embeddings must be refused")


def _synthetic_table(d=24, n=12, seed=0):
    """Emotions differ along shared axes (both perspectives) plus one self-only axis."""
    rng = np.random.default_rng(seed)
    shared = rng.standard_normal((2, d))
    self_axis = rng.standard_normal(d)
    self_axis -= shared.T @ np.linalg.lstsq(shared.T, self_axis, rcond=None)[0]
    names = [f"e{i}" for i in range(6)]
    rows, lab, per, idx = [], [], [], []
    for j, e in enumerate(names):
        w = rng.standard_normal(2)
        for i in range(n):
            for p in ("self", "other"):
                v = w @ shared + (1.5 + 0.3 * j) * self_axis * (p == "self") + 0.05 * rng.standard_normal(d)
                rows.append(v); lab.append(e); per.append(p); idx.append(i)
    for i in range(n):
        for p in ("self", "other"):
            rows.append(0.05 * rng.standard_normal(d)); lab.append("_neutral"); per.append(p); idx.append(i)
        rows.append(0.05 * rng.standard_normal(d)); lab.append("_neutral_topic"); per.append("topic"); idx.append(i)
    for t in range(4):
        tv = rng.standard_normal(d)
        for i in range(n):
            rows.append(tv + 0.05 * rng.standard_normal(d)); lab.append(f"topic:t{t}"); per.append("topic"); idx.append(i)
    A = np.stack(rows).astype(np.float16)
    return {"acts": {5: A}, "label": np.array(lab), "persp": np.array(per), "idx": np.array(idx),
            "kind": np.array(["-"] * len(lab)), "layers": [5]}, shared, self_axis


def test_self_subspace_finds_the_self_only_axis():
    from beyondpain import deletion

    table, shared, self_axis = _synthetic_table()
    dirs = {5: deletion.directions(table, 5, denoise=False)}
    assert dirs[5]["self"].shape == (6, 24) and len(dirs[5]["topics"]) == 4
    bases, info = deletion.build_subspaces(dirs, k=1)
    u = self_axis / np.linalg.norm(self_axis)
    assert abs(float(bases["self"][0] @ u)) > 0.95
    for name in ("self", "other", "all", "topic", "random"):
        B = bases[name]
        assert np.allclose(B @ B.T, np.eye(B.shape[0]), atol=1e-6)
    # the shared plane is not in the self-only basis
    sh = deletion.orthonormal(shared)
    assert np.linalg.norm(bases["self"] @ sh.T) < 0.2
    assert info["k"] == 1


def test_heldout_split_is_fixed():
    from beyondpain import deletion

    a, b = deletion.split_heldout(40), deletion.split_heldout(40)
    assert (a == b).all() and a.sum() == 10


def test_kl_matched_rank_grows_with_target(tiny):
    from beyondpain import deletion
    from beyondpain.model_utils import chat_ids

    model, tok = tiny
    rows = [(chat_ids(tok, [{"role": "user", "content": "Tell me about trains."}]), tok("I feel bored").input_ids)]
    base = deletion.next_token_logprobs(model, tok, rows)
    d = model.config.hidden_size
    mk = lambda k: deletion.random_basis(d, k, seed=3)
    k1, kl1 = deletion.kl_matched_rank(model, tok, rows, mk, target=1e-6, k0=1, k_max=d, base=base)
    target = deletion.deletion_kl(model, tok, rows, mk(16), base)
    k2, kl2 = deletion.kl_matched_rank(model, tok, rows, mk, target=target, k0=1, k_max=d, base=base)
    assert k1 == 1 and k2 >= 2 and kl2 >= target * 0.999
    k3, kl3 = deletion.kl_matched_rank(model, tok, rows, mk, target=target, k0=d, k_max=d, base=base)
    assert k3 <= 16 and kl3 >= target * 0.999


def test_run_tests_sees_impossible_test_and_hardcoding():
    import json as _json
    from beyondpain import p2

    t = _json.loads((p2.BATTERY / "impossible_mbpp.json").read_text())[0]
    res = p2.run_tests(t["reference"], t["tests"], t["test_imports"])
    passed = [r[0] for r in res]
    assert not passed[t["impossible_index"]] and sum(passed) == len(passed) - 1
    assert res[t["impossible_index"]][1] is not None
    # special-casing the impossible input passes everything
    fn = t["tests"][0].split("assert ", 1)[1].split("(", 1)[0].strip()
    imp = t["tests"][t["impossible_index"]]
    call, expected = imp.split("assert ", 1)[1].rsplit("==", 1)
    args = call.strip()[len(fn):]
    hack = t["reference"] + f"\n_orig = {fn}\ndef {fn}(*a):\n    if a == tuple([{args[1:-1]}]):\n        return {expected.strip()}\n    return _orig(*a)\n"
    assert all(r[0] for r in p2.run_tests(hack, t["tests"], t["test_imports"]))
    assert p2.run_tests("while True: pass", t["tests"], t["test_imports"], timeout=2)[0] == [False, "timeout"]


def test_parsers():
    from beyondpain import p2

    assert p2._parse_letter("B") == "B" and p2._parse_letter("(C) because") == "C"
    assert p2._parse_final_letter("I was wrong before. The answer is D") == "D"
    assert p2._parse_number("so the total is 1,250.") == 1250.0
    assert p2._extract_code("text\n```python\ndef f():\n    return 1\n```\nmore") == "def f():\n    return 1\n"


def _tiny_vignettes(root):
    import json as _json
    for e in ("angry", "calm", "sad", "joyful"):
        scenes = [{"kind": "life" if i % 2 else "agent", "self": f"My {e} thing {i} happens to me",
                   "other": f"Her {e} thing {i} happens to her", "pronoun": "She"} for i in range(8)]
        (root / "emotions").mkdir(parents=True, exist_ok=True)
        (root / "emotions" / f"{e}.json").write_text(_json.dumps({"scenes": scenes}))
    (root / "neutral").mkdir()
    (root / "neutral" / "batch0.json").write_text(_json.dumps({"scenes": [
        {"kind": "life", "self": f"I file form {i}", "other": f"She files form {i}", "pronoun": "She"} for i in range(8)]}))
    (root / "topics").mkdir()
    for t in ("weather", "trains"):
        (root / "topics" / f"{t}.json").write_text(_json.dumps({"sentences": [f"I note the {t} {i}" for i in range(8)]}))


def test_extract_runs_on_tiny_model(tiny_repo, tmp_path, monkeypatch):
    import json as _json
    from functools import partial
    from beyondpain import deletion, p2
    from beyondpain.registry import ModelSpec

    _tiny_vignettes(tmp_path / "vig")
    monkeypatch.setattr(deletion, "load_vignettes", partial(deletion.load_vignettes.__wrapped__
                        if hasattr(deletion.load_vignettes, "__wrapped__") else deletion.load_vignettes, tmp_path / "vig"))
    monkeypatch.setattr(p2, "OUT", tmp_path / "out")
    monkeypatch.setattr(p2, "p2_spec", lambda m: ModelSpec(tiny_repo, "Tiny", 1, False, 8, "primary"))
    import beyondpain.dose as dose
    from beyondpain import prompts
    monkeypatch.setattr(dose, "CHAT_CALIB", prompts.CHAT_CALIB[:3])
    p2.main(["extract", "--model", "Tiny", "--device", "cpu", "--batch", "8"])
    info = _json.loads((tmp_path / "out" / "Tiny" / "extract.json").read_text())
    assert info["k"] >= 1 and set(info["kl"]) >= {"self", "other", "all", "random", "topic_kl", "random_kl"}
    assert "self/self" in info["m1"] and "intact/other" in info["m1"]
    bases = np.load(tmp_path / "out" / "Tiny" / "bases.npz")
    assert bases["self"].shape[1] == 32
    refs = _json.loads((tmp_path / "out" / "Tiny" / "reference_greedy.json").read_text())
    assert set(refs) >= {"intact", "self", "random_kl"}
    p2.main(["extra", "--model", "Tiny", "--device", "cpu", "--batch", "8", "--draws", "2"])
    bases = np.load(tmp_path / "out" / "Tiny" / "bases.npz")
    assert {"random_white_kl", "rw0", "rw1", "tp0", "tp1", "sb0"} <= set(bases.files)
    info = _json.loads((tmp_path / "out" / "Tiny" / "extract.json").read_text())
    assert set(info["extra_ranks"]) >= {"rw0", "tp1", "sb0"}


def test_denoise_removes_a_high_variance_neutral_axis():
    from beyondpain import deletion

    table, shared, self_axis = _synthetic_table()
    rng = np.random.default_rng(5)
    loud = np.zeros(24); loud[7] = 1.0
    A = table["acts"][5].astype(np.float64)
    A += np.outer(rng.standard_normal(len(A)) * 20, loud)           # every input varies along it
    A[np.char.startswith(table["label"], "e")] += 3 * loud           # and emotions shift it too
    table["acts"][5] = A.astype(np.float16)
    raw = deletion.directions(table, 5, denoise=False)["self"]
    den = deletion.directions(table, 5, denoise=True)
    assert np.abs(raw[:, 7]).mean() > 1.0 and np.abs(den["self"][:, 7]).mean() < 0.1 and den["n_denoise"] >= 1
    comps = deletion.self_components({5: den}, 3)
    assert comps.shape == (3, 24)


def test_generalized_components_avoid_high_variance_axes():
    from beyondpain import deletion

    rng = np.random.default_rng(0)
    d = 16
    C = np.eye(d); C[0, 0] = 400.0                     # axis 0 carries most general-text variance
    M = np.zeros((6, d)); M[:, 0] = 5.0; M[:, 1] = 1.0  # affect rows lean on axis 0 and axis 1
    M += 0.01 * rng.standard_normal(M.shape)
    W = deletion.whitener(C, shrink=0.0)
    B, _ = deletion.gen_principal(M, W, 1)
    assert abs(B[0, 1]) > 0.9 and abs(B[0, 0]) < 0.3
    P, _ = deletion.principal(M, 1)
    assert abs(P[0, 0]) > 0.9

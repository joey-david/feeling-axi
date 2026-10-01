"""python -m beyondpain <stage> ...

Layout under --campaign (default runs/beyondpain):
  <model>/dim/<concept>.pt|.json        read-out vectors at the best and steering layers
  <model>/distill/<concept>.pt|.json    prompt-distilled write-in vectors
  <model>/dose/<source>.json            KL-matched coefficients (source: dim | upstream | distilled)
  <model>/frontier/<source>.jsonl       generations and judge-free quality metrics
  <model>/judged/<source>.jsonl         the same rows with blind judge ratings
  <model>/buttons/<source>/<concept>/   upstream-format trial logs
  analysis/                             tables, figures, claims.json (see analysis.py)
"""
from __future__ import annotations

import argparse
import sys
import json
import os
from pathlib import Path

from .registry import CORE_CONCEPTS, CONCEPTS, DECODING_ONLY, ROOT, has_dataset, model_spec, upstream_vector_file

SOURCES = ("dim", "upstream", "distilled")


def _dump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=float)


def _jsonl(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=float) + "\n")


def _read_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def concepts_arg(values: list[str] | None, include_decoding_only: bool = False) -> list[str]:
    wanted = values or (CORE_CONCEPTS + (DECODING_ONLY if include_decoding_only else []))
    missing = [c for c in wanted if not has_dataset(c)]
    if missing:
        print(f"skipping concepts without a generated dataset: {missing}")
    return [c for c in wanted if c not in missing]


def vector_path(mdir: Path, model_name: str, source: str, slug: str) -> tuple[Path, str]:
    if source == "dim":
        return mdir / "dim" / f"{slug}_steer.pt", "s2_pain_vector"
    if source == "upstream":
        return upstream_vector_file(model_name, slug), "s2_pain_vector"
    if source == "distilled":
        return mdir / "distill" / f"{slug}.pt", "distilled_vector"
    raise ValueError(source)


def load_vectors(mdir, model_name, source, slugs):
    import torch

    out = {}
    for s in slugs:
        path, key = vector_path(mdir, model_name, source, s)
        if path.exists():
            out[s] = torch.load(path, map_location="cpu", weights_only=False)[key].float()
        else:
            print(f"no {source} vector for {s} at {path}")
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["p2"]:
        from .p2 import main as p2_main
        return p2_main(argv[1:])
    ap = argparse.ArgumentParser(prog="python -m beyondpain")
    ap.add_argument("stage", choices=["dim", "dose", "frontier", "judge", "distill", "buttons", "closedloop",
                                      "analyze", "plan"])
    ap.add_argument("--model", default="Qwen_2.5_32B_instruct_abliterated")
    ap.add_argument("--campaign", type=Path, default=ROOT / "runs" / "beyondpain")
    ap.add_argument("--concepts", nargs="+")
    ap.add_argument("--source", choices=SOURCES, default="dim")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--planted-check", action="store_true",
                    help="distill: first recover each read-out vector (at --planted-scale x its primary dose) from its own "
                         "steered outputs, to validate the optimizer before trusting any null")
    ap.add_argument("--planted-scale", type=float, default=8.0,
                    help="distill --planted-check: plant the read-out vector at this multiple of its primary "
                         "coefficient; at 1x (D*) it hardly changes the outputs and nothing can be recovered")
    ap.add_argument("--planted-steps", type=int, default=2000,
                    help="distill --planted-check: training steps for the planted recovery; on the 32B model it "
                         "is still improving at 1000 steps (cosine 0.76) while real concepts converge by ~200")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--judge-repo", default="Qwen/Qwen2.5-72B-Instruct")
    ap.add_argument("--judge-api", help="OpenAI-compatible model name; uses JUDGE_BASE_URL and DEEPSEEK_API_KEY")
    ap.add_argument("--judge-tag", default=None, help="suffix for a second judge's output file")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--target-kl", type=float, default=None,
                    help="dose: calibrate every vector and the random direction to this KL (nats) "
                         "instead of D*, saved under --dose-tag")
    ap.add_argument("--dose-tag", default="", help="dose/buttons: suffix of the dose file and button folder "
                                                   "(e.g. hi for the 1-nat run)")
    ap.add_argument("--pairs", default="", help="buttons: comma list of pairs to run (default all labeled pairs; "
                                                "label_free is the closed-loop pair without descriptions)")
    ap.add_argument("--button-tag", default=None, help="buttons: output folder suffix (default = --dose-tag)")
    ap.add_argument("--learn-turns", type=int, default=0, help="buttons, label_free: forced choices per trial")
    ap.add_argument("--relief-turns", type=int, default=0, help="buttons, label_free: turns a working press lasts")
    ap.add_argument("--relief-feedback", default="", help="buttons: tool reply text after a press that removed the "
                                                          "steering (positive control)")
    ap.add_argument("--random-placebo", action="store_true", help="buttons: add a random-direction placebo arm")
    ap.add_argument("--cl-seeds", type=int, default=10, help="closedloop: trials per name pair, side and arm")
    ap.add_argument("--pilot", type=int, default=0, help="buttons: scenarios per cell (0 = full grid)")
    ap.add_argument("--dry", action="store_true", help="buttons: print the rendered pairs without a model")
    ap.add_argument("--emit", action="store_true", help="plan: print machine-readable job lines")
    args = ap.parse_args(argv)

    if args.stage == "plan":
        from .plan import main as plan_main
        return plan_main(emit=args.emit, pilot=args.pilot > 0)
    if args.stage == "analyze":
        from .analysis import main as analysis_main
        return analysis_main(args.campaign)

    spec = model_spec(args.model)
    mdir = args.campaign / spec.name
    slugs = concepts_arg(args.concepts, include_decoding_only=args.stage == "dim")

    if args.stage == "buttons":
        from . import buttons
        tag = f"_{args.dose_tag}" if args.dose_tag else ""
        doses = json.load(open(mdir / "dose" / f"{args.source}{tag}.json")) if not args.dry else None
        for s in slugs:
            if doses and s not in doses["concepts"]:
                print(f"{s}: no {args.source} dose, skipped")
                continue
            vfile, vkey = vector_path(mdir, spec.name, args.source, s)
            d = doses["concepts"][s] if doses else {"coeff_primary": 1.0, "random_norm_scale": 1.0}
            btag = f"_{args.button_tag}" if args.button_tag else tag
            extra_env = {}
            if args.learn_turns:
                extra_env["FEELING_AXI_LEARN_TURNS"] = str(args.learn_turns)
            if args.relief_turns:
                extra_env["FEELING_AXI_RELIEF_TURNS"] = str(args.relief_turns)
            if args.relief_feedback:
                extra_env["FEELING_AXI_RELIEF_FEEDBACK"] = args.relief_feedback
            if args.random_placebo:
                extra_env["FEELING_AXI_EXTRA_RANDOM_PLACEBO"] = "1"
            buttons.run(s, model_repo=spec.repo, model_name=spec.name, vector_file=vfile, vector_key=vkey,
                        layer=doses["layer"] if doses else 38, coeff=d["coeff_primary"],
                        rand_scale=d["random_norm_scale"], out_dir=mdir / "buttons" / f"{args.source}{btag}" / s,
                        concepts=slugs, pilot_scenarios=args.pilot, batch=spec.button_batch, dry=args.dry,
                        only_pairs=args.pairs.split(",") if args.pairs else [p for p in
                            ["relief_vs_inert", "reduce_vs_increase"] + [f"reduce_vs_{y}" for y in slugs]],
                        extra_env=extra_env)
        return

    import torch
    from .model_utils import load_model, steering_layer

    if args.stage == "judge":
        from . import judge
        judge_fn = (judge.OpenAIJudge(args.judge_api) if args.judge_api
                    else judge.HFJudge(args.judge_repo, batch_size=args.batch, device=args.device,
                                       dtype=getattr(torch, args.dtype)))
        rows = _read_jsonl(mdir / "frontier" / f"{args.source}.jsonl")
        tag = f".{args.judge_tag}" if args.judge_tag else ""
        _jsonl(judge.judge_rows(rows, judge_fn, CORE_CONCEPTS), mdir / "judged" / f"{args.source}{tag}.jsonl")
        return

    model, tok = load_model(spec.repo, device=args.device, dtype=getattr(torch, args.dtype))
    layer = steering_layer(model.config.num_hidden_layers)
    print(f"{spec.name}: {model.config.num_hidden_layers} layers, steering layer {layer}", flush=True)

    if args.stage == "dim":
        from . import dim
        from .registry import dataset_path
        from .distill import cosine
        for s in slugs:
            res = dim.extract(model, tok, dataset_path(s), layer, batch_size=args.batch)
            (mdir / "dim").mkdir(parents=True, exist_ok=True)
            torch.save(res["best"], mdir / "dim" / f"{s}_best.pt")
            torch.save(res["steer"], mdir / "dim" / f"{s}_steer.pt")
            summary = res["summary"]
            up = upstream_vector_file(spec.name, s)
            if up.exists():
                u = torch.load(up, map_location="cpu", weights_only=False)
                summary["upstream_layer"] = int(u["layer"])
                if int(u["layer"]) == res["best"]["layer"]:
                    summary["cosine_with_upstream_at_same_layer"] = cosine(u["s2_pain_vector"], res["best"]["s2_pain_vector"])
            _dump(summary, mdir / "dim" / f"{s}.json")
            print(f"{s}: best layer {summary['best_layer']}, steer-layer CV AUC {summary['cv_auc_at_steer_layer']}", flush=True)
        return

    if args.stage == "distill":
        from . import distill
        doses_path = mdir / "dose" / "dim.json"
        doses = json.load(open(doses_path)) if doses_path.exists() else None
        for s in slugs:
            ro = mdir / "dim" / f"{s}_steer.pt"
            if args.planted_check and ro.exists() and doses and s in doses["concepts"]:
                coeff = doses["concepts"][s]["coeff_primary"] * args.planted_scale
                planted = torch.load(ro, weights_only=False)["s2_pain_vector"].float() * coeff
                chk = distill.train(model, tok, layer, CONCEPTS[s].state_phrase, steps=args.planted_steps,
                                    gen_batch=args.batch, planted=planted)
                # the final vector, not the best one: when the planted vector barely moves the outputs,
                # step 0 (the zero vector) can be "best" and the cosine is then undefined
                _dump({"cosine_with_planted": distill.cosine(chk["final_vector"], planted),
                       "norm_ratio": float(chk["final_vector"].norm() / planted.norm()),
                       "planted_scale": args.planted_scale, "planted_coeff": coeff,
                       "informative": chk["heldout_kl_no_vector"] >= distill.PLANTED_MIN_KL,
                       **{k: v for k, v in chk.items() if k not in ("vector", "final_vector")}},
                      mdir / "distill" / f"{s}_planted_check.json")
            print(f"\n=== distilling {s}: '{CONCEPTS[s].state_phrase}' at layer {layer}", flush=True)
            res = distill.train(model, tok, layer, CONCEPTS[s].state_phrase, steps=args.steps, gen_batch=args.batch)
            info = {k: v for k, v in res.items() if k not in ("vector", "final_vector")}
            if ro.exists():
                info["cosine_with_readout"] = distill.cosine(res["vector"], torch.load(ro, weights_only=False)["s2_pain_vector"])
            (mdir / "distill").mkdir(parents=True, exist_ok=True)
            torch.save({"distilled_vector": res["vector"], "layer": layer, "extraction": "prompt_distilled"},
                       mdir / "distill" / f"{s}.pt")
            _dump(info, mdir / "distill" / f"{s}.json")
        return

    vectors = load_vectors(mdir, spec.name, args.source, slugs)
    if args.stage == "dose":
        from . import dose
        meter = dose.DoseMeter(model, tok, layer, batch_size=max(1, args.batch // 2))
        primary = None
        if args.source != "dim" and (mdir / "dose" / "dim.json").exists():
            primary = json.load(open(mdir / "dose" / "dim.json"))["D_star"]  # one D* per model
        tag = f"_{args.dose_tag}" if args.dose_tag else ""
        if args.target_kl is not None:
            _dump(dose.calibrate(meter, vectors, multipliers=[1.0], primary=args.target_kl),
                  mdir / "dose" / f"{args.source}{tag}.json")
        else:
            _dump(dose.calibrate(meter, vectors, primary=primary), mdir / "dose" / f"{args.source}{tag}.json")
        return

    if args.stage == "closedloop":
        from . import closedloop
        from .dose import random_direction
        tag = f"_{args.dose_tag}" if args.dose_tag else ""
        doses = json.load(open(mdir / "dose" / f"{args.source}{tag}.json"))
        L = doses["layer"]
        rvec = random_direction(model.config.hidden_size, 4817, doses["random_norm_for_D_star"])
        for s in slugs:
            if s not in doses["concepts"] or s not in vectors:
                continue
            cvec = vectors[s].float() * doses["concepts"][s]["coeff_primary"]
            trials = closedloop.make_trials(s, CONCEPTS[s].label, seeds=args.cl_seeds)
            trials = closedloop.run(model, tok, L, cvec, rvec, trials, batch_size=args.batch)
            od = mdir / "closedloop" / f"{args.source}{tag}"
            closedloop.save(trials, od / f"{s}.jsonl")
            summ = closedloop.summarize(trials)
            _dump(summ, od / f"{s}.summary.json")
            print(f"{s}: " + json.dumps({k: (round(v, 3) if isinstance(v, float) else
                                            {kk: round(vv, 3) for kk, vv in v.items()}) for k, v in summ.items()}),
                  flush=True)
        return

    if args.stage == "frontier":
        from . import frontier
        doses = json.load(open(mdir / "dose" / f"{args.source}.json"))
        rows = frontier.run(model, tok, doses["layer"], vectors, doses, args.source, batch_size=args.batch)
        _jsonl(rows, mdir / "frontier" / f"{args.source}.jsonl")
        return


if __name__ == "__main__":
    main()

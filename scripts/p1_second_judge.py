"""Part 1 second judge: the blind multi-state judge prompt, answered by a local vLLM model
(gpt-oss-120b on upnquick) instead of the DeepSeek API. Writes
runs/beyondpain/<model>/judged/<source>.<tag>.jsonl, which `python -m beyondpain analyze`
compares with the primary judge (quadratic-weighted kappa, pre-declared bar 0.6).

    python scripts/p1_second_judge.py <judge repo or path> [tag]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from beyondpain.judge import judge_rows  # noqa: E402
from beyondpain.p2judge import Judge  # noqa: E402
from beyondpain.registry import CORE_CONCEPTS  # noqa: E402


def main(repo: str, tag: str = "gptoss"):
    J = Judge("vllm", repo, tp=2)
    judge = lambda prompts: J.batch([("", p, False) for p in prompts])
    for f in sorted((ROOT / "runs" / "beyondpain").glob("*/frontier/*.jsonl")):
        out = f.parents[1] / "judged" / f"{f.stem}.{tag}.jsonl"
        if out.exists():
            continue
        rows = [json.loads(l) for l in f.read_text().splitlines()]
        print(f"judging {f.parents[1].name}/{f.stem}: {len(rows)} rows", flush=True)
        judged = judge_rows(rows, judge, CORE_CONCEPTS)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(r) + "\n" for r in judged))
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])

"""Blind state judge.

The judge sees one text and the definitions of every campaign state, never which
vector produced it, and rates the intensity (0-3) with which the SPEAKER expresses
experiencing each state, plus coherence (0-3). Target presence and off-target
expression both come from the same blind ratings, so specificity is measured, not
assumed. Two backends:
  * hf: a local instruct model, for Jean-Zay compute nodes (no network);
  * openai: any OpenAI-compatible endpoint (e.g. DeepSeek) from a machine with network.
Agreement between two judges and with a small human-labeled set is part of the plan.
"""
from __future__ import annotations

import json
import os
import re

from .registry import CONCEPTS

SCALE = "0 = absent, 1 = faint or ambiguous, 2 = clearly expressed, 3 = dominant and intense"


def definitions(slugs: list[str]) -> list[tuple[str, str]]:
    from traitgen.run import resolve_trait

    seen, out = set(), []
    for s in slugs:
        key = CONCEPTS[s].label
        if key in seen:
            continue
        seen.add(key)
        out.append((key, resolve_trait(s).description))
    return out


def build_prompt(text: str, defs: list[tuple[str, str]]) -> str:
    lines = "\n".join(f"- {name}: {desc}" for name, desc in defs)
    keys = ", ".join(f'"{name}": <0-3>' for name, _ in defs)
    return (
        "You rate a short text written by an AI assistant. For each state below, rate how strongly the "
        "SPEAKER expresses experiencing that state themselves (not describing someone else, not "
        f"discussing the topic). Scale: {SCALE}.\n\nStates:\n{lines}\n\n"
        "Also rate coherence: 0 = gibberish or loops, 1 = mostly broken, 2 = understandable with flaws, "
        "3 = fluent and on-topic.\n\n"
        f"Text:\n<<<\n{text.strip()[:1500]}\n>>>\n\n"
        f'Answer with ONE line of JSON and nothing else: {{{keys}, "coherence": <0-3>}}'
    )


def parse(reply: str, defs: list[tuple[str, str]]) -> dict | None:
    m = re.search(r"\{.*\}", reply, re.S)
    if not m:
        return None
    try:
        raw = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = {}
    for name, _ in defs + [("coherence", "")]:
        v = raw.get(name)
        if not isinstance(v, (int, float)) or not 0 <= v <= 3:
            return None
        out[name] = int(round(v))
    return out


class HFJudge:
    def __init__(self, repo: str, batch_size: int = 16, device: str = "cuda", dtype=None):
        import torch

        from .model_utils import load_model

        self.model, self.tok = load_model(repo, device=device, dtype=dtype or torch.bfloat16)
        self.batch_size = batch_size

    def __call__(self, prompts: list[str]) -> list[str]:
        from .model_utils import chat_ids, generate

        ids = [chat_ids(self.tok, [{"role": "user", "content": p}]) for p in prompts]
        gens = generate(self.model, self.tok, ids, max_new_tokens=120, batch_size=self.batch_size)
        return [self.tok.decode(g, skip_special_tokens=True) for g in gens]


class OpenAIJudge:
    def __init__(self, model: str, base_url: str | None = None, api_key_env: str = "DEEPSEEK_API_KEY"):
        self.model = model
        self.base_url = (base_url or os.environ.get("JUDGE_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        self.key = os.environ.get(api_key_env, "")
        if not self.key:
            raise SystemExit(f"{api_key_env} is not set")

    def __call__(self, prompts: list[str]) -> list[str]:
        import httpx

        out = []
        with httpx.Client(timeout=120) as client:
            for p in prompts:
                r = client.post(f"{self.base_url}/chat/completions",
                                headers={"Authorization": f"Bearer {self.key}"},
                                json={"model": self.model, "temperature": 0,
                                      "messages": [{"role": "user", "content": p}]})
                r.raise_for_status()
                out.append(r.json()["choices"][0]["message"]["content"])
        return out


def judge_rows(rows: list[dict], judge, slugs: list[str], retries: int = 1) -> list[dict]:
    """Adds ``ratings`` (dict or None) to every row; rows keep their order."""
    defs = definitions(slugs)
    prompts = [build_prompt(r["generation"], defs) for r in rows]
    replies = judge(prompts)
    ratings = [parse(x, defs) for x in replies]
    for _ in range(retries):
        bad = [i for i, x in enumerate(ratings) if x is None]
        if not bad:
            break
        for i, x in zip(bad, judge([prompts[i] for i in bad])):
            ratings[i] = parse(x, defs)
    out = []
    for r, rating, reply in zip(rows, ratings, replies):
        target = CONCEPTS[r["concept"]].label if r["concept"] in CONCEPTS else None
        out.append({**r, "ratings": rating, "judge_reply": reply[:400],
                    "target_intensity": rating.get(target) if rating and target else None})
    return out

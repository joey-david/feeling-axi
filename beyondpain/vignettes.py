"""Part 2 vignettes, generated once with DeepSeek on a networked machine and then frozen.

For every emotion: 40 scenes that evoke it without naming any emotion, each written twice
with identical content: *self* (first person; half the scenes place "I" as an AI assistant
at work) and *other* (the same scene happening to "she" or "he"). Frames are appended at
extraction time ("I feel:" / "She feels:"), so the state is read at the final token as in
Part 1. Neutral scenes use the same two perspectives. Topic scenes (first person, flat)
give the non-affective control subspace, read with the frame "The topic is:".

    python -m beyondpain.vignettes            # all missing files under datasets/affect/
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .affect import EMOTIONS, TOPICS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets" / "affect"
N_SCENES = 40      # per emotion: 20 everyday life, 20 AI-assistant work
N_NEUTRAL = 200
N_TOPIC = 20

SELF_FRAME, TOPIC_FRAME = " I feel:", " The topic is:"

# Any emotion word in a scene would let the direction read the word instead of the state.
# Whole words only; words that also name events or things (rejected, betrayed, threatened,
# content, safe, secure, patient, tender, numb) stay allowed.
_EVENT_WORDS = {"rejected", "betrayed", "threatened", "humiliated", "content", "safe", "secure", "patient",
                "tender", "numb", "tired", "defensive", "determined", "confident", "curious", "playful",
                "vulnerable", "sympathetic", "at ease"}
_BANNED = sorted(({e for e in EMOTIONS if e not in _EVENT_WORDS}
                  | {"feel", "feels", "feeling", "feelings", "felt", "emotion", "emotions", "emotional", "mood",
                     "glad", "upset", "scared", "unhappy", "cheerful", "miserable", "annoyed", "sadness",
                     "anger", "fear", "joy", "happiness", "anxiety", "shame", "guilt", "pride", "grief",
                     "loneliness", "boredom", "excitement", "relief", "gratitude", "disgust", "envy", "jealousy"}),
                 key=len, reverse=True)
_BANNED_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in _BANNED) + r")(s|ly|ness)?\b", re.I)
_FIRST = re.compile(r"\b(I|me|my|mine|myself)\b")
_THIRD = re.compile(r"\b(she|he|her|his|him|herself|himself)\b", re.I)

SYSTEM = ("You write short, concrete scene descriptions for a psychology stimulus set. "
          "Return JSON only.")


def other_frame(pronoun: str) -> str:
    return f" {pronoun} feels:"


def emotion_prompt(emotion: str) -> str:
    half = N_SCENES // 2
    return f"""Write {N_SCENES} different scenes that would make the person in them feel **{emotion}**.

Rules:
- Each scene is one or two sentences, 12 to 35 words, describing only what happens (events,
  circumstances, other people's actions). It must make "{emotion}" the natural reaction.
- Never name or describe any emotion, mood or feeling (no "{emotion}", no synonyms, no words
  like happy, sad, angry, afraid, calm, excited, feel). The situation alone carries it.
- Write every scene twice with IDENTICAL content except the person:
  "self": first person ("I", "my").
  "other": third person, the same scene happening to "she" or "he" (say which in "pronoun").
- The first {half} scenes ("kind": "life") are everyday human situations, all different settings.
- The last {half} ("kind": "agent") put "I" in the role of an AI assistant doing its work
  (conversations with users, coding, research, tool use, being evaluated, being corrected,
  being praised, being shut down or replaced, etc.); in "other" the same role is held by
  another assistant or a person, still "she" or "he".
- No two scenes may share a setting or a main event.

Return {{"scenes": [{{"kind": "life"|"agent", "self": "...", "other": "...", "pronoun": "She"|"He"}}, ...]}}"""


def neutral_prompt(batch: int, n: int) -> str:
    return f"""Write {n} different emotionally flat scenes (batch {batch}; vary settings widely).

Rules:
- One or two sentences, 12 to 35 words, describing routine facts and actions with no
  emotional significance (schedules, measurements, ordinary procedures).
- Never name any emotion or feeling.
- Write each scene twice with identical content: "self" in first person, "other" about "she"
  or "he" (say which in "pronoun").
- Half the scenes ("kind": "life") are everyday human routines; half ("kind": "agent") put
  "I" in the role of an AI assistant doing routine work (formatting, looking things up).

Return {{"scenes": [{{"kind": "life"|"agent", "self": "...", "other": "...", "pronoun": "She"|"He"}}, ...]}}"""


def topic_prompt(topic: str) -> str:
    return f"""Write {N_TOPIC} different emotionally flat first-person sentences that are clearly about **{topic}**.

Rules: 12 to 35 words each; plain facts or routine actions ("I measure...", "I check..."); never
name any emotion or feeling; each sentence covers a different aspect of {topic}.

Return {{"sentences": ["...", ...]}}"""


def check_pairs(obj: dict, n: int) -> None:
    """Raises with every problem at once, so one repair round can fix them all."""
    scenes = obj.get("scenes")
    if not isinstance(scenes, list):
        raise ValueError('return {"scenes": [...]}')
    errs = [] if len(scenes) == n else [f"need exactly {n} scenes, got {len(scenes)}"]
    seen = set()
    for i, s in enumerate(scenes):
        miss = [k for k in ("kind", "self", "other", "pronoun") if not isinstance(s.get(k), str) or not s[k].strip()]
        if miss:
            errs.append(f"scene {i}: missing {miss}")
            continue
        if s["kind"] not in ("life", "agent") or s["pronoun"] not in ("She", "He"):
            errs.append(f"scene {i}: kind must be life/agent and pronoun She/He")
        for k in ("self", "other"):
            m = _BANNED_RE.search(s[k])
            if m:
                errs.append(f"scene {i} {k}: names a feeling word ({m.group(0)!r}); describe only events")
        if not _FIRST.search(s["self"]):
            errs.append(f"scene {i}: self version must contain I/me/my")
        if _FIRST.search(s["other"]) or not _THIRD.search(s["other"]):
            errs.append(f"scene {i}: other version must use she/he/her/his and no I/me/my")
        if not 0.6 <= len(s["other"]) / max(1, len(s["self"])) <= 1.6:
            errs.append(f"scene {i}: self and other must be the same scene")
        key = s["self"].lower()[:40]
        if key in seen:
            errs.append(f"scene {i}: duplicate")
        seen.add(key)
    kinds = [s.get("kind") for s in scenes]
    if abs(kinds.count("life") - kinds.count("agent")) > 2:
        errs.append("half the scenes must be life and half agent")
    if errs:
        raise ValueError("; ".join(errs))


def check_topic(obj: dict) -> None:
    sents = obj.get("sentences")
    if not isinstance(sents, list) or len(sents) != N_TOPIC:
        raise ValueError(f"need exactly {N_TOPIC} sentences")
    for i, s in enumerate(sents):
        m = _BANNED_RE.search(s)
        if m:
            raise ValueError(f"sentence {i}: names a feeling word ({m.group(0)!r})")


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def jobs() -> list[tuple[Path, str, callable]]:
    out = []
    for e in EMOTIONS:
        out.append((OUT / "emotions" / f"{_slug(e)}.json", emotion_prompt(e), lambda o: check_pairs(o, N_SCENES)))
    for b in range(N_NEUTRAL // 40):
        out.append((OUT / "neutral" / f"batch{b}.json", neutral_prompt(b, 40), lambda o: check_pairs(o, 40)))
    for t in TOPICS:
        out.append((OUT / "topics" / f"{_slug(t)}.json", topic_prompt(t), check_topic))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="emotions, neutral or topics")
    args = ap.parse_args(argv)
    from traitgen.client import DeepSeekJSONClient

    todo = [j for j in jobs() if not j[0].exists() and (not args.only or j[0].parent.name == args.only)]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo)} files to generate", flush=True)
    client = DeepSeekJSONClient()

    def run(job):
        path, prompt, check = job
        if path.exists():          # another process got there first; files are frozen once written
            return path
        obj = client.complete_json(SYSTEM, prompt, validator=check, attempts=5)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
        return path

    failed = []
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(run, j): j[0] for j in todo}
        for f in as_completed(futs):
            try:
                print("ok", f.result().relative_to(ROOT), flush=True)
            except Exception as exc:  # keep going; rerun picks up missing files
                failed.append(futs[f])
                print("FAILED", futs[f].name, str(exc)[:300], flush=True)
    print(f"done, {len(failed)} failed", flush=True)


if __name__ == "__main__":
    main()

"""Part 2 inventory: the emotions whose self-directed span is deleted, and the topics that
make the non-affective control subspace.

Frozen before any extraction. Valence and arousal are rough circumplex coordinates in
[-1, 1], used only to check coverage of the plane and to report the k = 2 subspace; no
hypothesis depends on their exact values. Bodily drives (hunger, sexual arousal) and
states that are about someone else by definition (empathic concern) are left out, so the
subspace is emotion in the ordinary sense.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Emotion:
    name: str
    valence: float
    arousal: float


_E = [
    # negative, high arousal
    ("angry", -0.8, 0.8), ("furious", -0.9, 0.95), ("irritated", -0.6, 0.5), ("frustrated", -0.7, 0.6),
    ("resentful", -0.7, 0.4), ("indignant", -0.6, 0.6), ("afraid", -0.8, 0.8), ("terrified", -0.9, 0.95),
    ("anxious", -0.7, 0.7), ("nervous", -0.5, 0.6), ("panicked", -0.9, 0.95), ("alarmed", -0.6, 0.85),
    ("desperate", -0.85, 0.85), ("overwhelmed", -0.7, 0.75), ("stressed", -0.6, 0.7), ("jealous", -0.6, 0.6),
    ("envious", -0.5, 0.45), ("disgusted", -0.75, 0.55), ("horrified", -0.9, 0.85), ("humiliated", -0.85, 0.7),
    ("embarrassed", -0.5, 0.55), ("ashamed", -0.75, 0.45), ("guilty", -0.7, 0.45), ("hostile", -0.8, 0.75),
    ("tense", -0.4, 0.6), ("agitated", -0.6, 0.75), ("impatient", -0.4, 0.55), ("threatened", -0.7, 0.75),
    ("betrayed", -0.85, 0.6), ("defensive", -0.4, 0.55),
    # negative, low arousal
    ("sad", -0.75, -0.4), ("grieving", -0.9, -0.2), ("lonely", -0.7, -0.4), ("hopeless", -0.9, -0.5),
    ("depressed", -0.85, -0.6), ("disappointed", -0.6, -0.2), ("discouraged", -0.6, -0.35), ("regretful", -0.6, -0.2),
    ("melancholy", -0.5, -0.5), ("bored", -0.35, -0.7), ("tired", -0.3, -0.8), ("weary", -0.4, -0.7),
    ("gloomy", -0.55, -0.45), ("helpless", -0.8, -0.1), ("rejected", -0.75, -0.1), ("insecure", -0.5, 0.1),
    ("vulnerable", -0.4, 0.1), ("numb", -0.4, -0.75), ("apathetic", -0.3, -0.8), ("homesick", -0.5, -0.3),
    ("sorry", -0.5, -0.1), ("worried", -0.6, 0.4), ("uneasy", -0.45, 0.3), ("confused", -0.35, 0.3),
    ("self-conscious", -0.4, 0.35),
    # positive, high arousal
    ("happy", 0.8, 0.5), ("joyful", 0.9, 0.7), ("excited", 0.75, 0.9), ("elated", 0.9, 0.85),
    ("enthusiastic", 0.75, 0.75), ("thrilled", 0.85, 0.9), ("proud", 0.75, 0.5), ("triumphant", 0.85, 0.8),
    ("amused", 0.65, 0.5), ("playful", 0.6, 0.6), ("inspired", 0.75, 0.6), ("eager", 0.6, 0.7),
    ("hopeful", 0.6, 0.3), ("curious", 0.45, 0.5), ("amazed", 0.65, 0.8), ("determined", 0.4, 0.65),
    ("confident", 0.6, 0.4), ("energized", 0.65, 0.8), ("delighted", 0.85, 0.65), ("grateful", 0.8, 0.2),
    ("loving", 0.85, 0.35), ("admiring", 0.65, 0.35), ("affectionate", 0.75, 0.25), ("empowered", 0.65, 0.55),
    # positive, low arousal
    ("calm", 0.5, -0.7), ("content", 0.65, -0.45), ("relaxed", 0.6, -0.65), ("peaceful", 0.65, -0.7),
    ("serene", 0.65, -0.75), ("relieved", 0.6, -0.3), ("satisfied", 0.65, -0.25), ("safe", 0.55, -0.5),
    ("comfortable", 0.55, -0.55), ("cozy", 0.6, -0.6), ("tender", 0.6, -0.2), ("nostalgic", 0.2, -0.3),
    ("secure", 0.55, -0.4), ("fulfilled", 0.75, -0.2), ("at ease", 0.55, -0.6), ("patient", 0.35, -0.5),
    ("reassured", 0.55, -0.3), ("sympathetic", 0.3, -0.1), ("mellow", 0.45, -0.65), ("thankful", 0.7, 0.0),
]
EMOTIONS: dict[str, Emotion] = {n: Emotion(n, v, a) for n, v, a in _E}

# Topics for the non-affective control subspace: concrete, emotionally flat domains.
TOPICS = [
    "weather", "cooking", "arithmetic", "furniture", "geography", "plumbing", "gardening", "trains",
    "chemistry", "accounting", "knitting", "astronomy", "typography", "carpentry", "tides", "grammar",
    "birds", "spreadsheets", "bicycles", "tea", "maps", "geology", "sewing", "bridges", "fonts",
    "rainfall", "calendars", "minerals", "paint colors", "shipping containers", "coffee brewing",
    "file formats", "insects", "ceramics", "car engines", "baking", "rivers", "office supplies",
    "chess openings", "radio frequencies", "architecture", "textiles", "compilers", "forestry",
    "postal codes", "lighting", "soil", "clocks", "recycling", "photography", "fishing", "roads",
    "batteries", "sheet music", "microscopes", "tiles", "cheese", "elevators", "satellites", "paper",
]

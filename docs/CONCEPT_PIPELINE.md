# Concept adaptation design

This fork keeps the released Pain-axis experiments as the reference implementation and
changes one experimental variable: the semantic target.

## What the released dataset actually does

The generator is based on the files in `datasets/`, not on a prose summary of the paper.

### S1: 20 ten-way matched sets

Each S1 set contains exactly one row from A1, A2, A3, A4, A5, B, C1, C2, D and E.
The ten rows share a conspicuous predicate/event skeleton. Across the twenty official
sets the anchor predicates are roughly:

`spreads, drops, meets, removes, fills, enters, reaches, covers, breaks, cuts,
holds, leaves, takes, carries, opens, closes, follows, touches, moves, falls`.

The event arguments change so the five target categories and five controls have different
semantics while syntax, tense and length remain close. This is why the generator adapts a
whole ten-way set in one API call instead of generating categories independently.

### S2: 20 naturalistic ten-way sets

S2 has the same category balance and the same 20 set IDs, but it intentionally drops the
shared-predicate constraint. The sentences are short naturalistic situations. The set IDs
still matter because the released 5-fold evaluation holds out whole set IDs.

Both S1 and S2 have separate 1P and 3P banks and every prompt ends with the literal
`I feel:`, including the 3P condition. The generator preserves that oddity exactly.

### Controls that are not part of the target/control fit

- `Random_1P/3P`: 200 neutral prompts. Frozen verbatim for every generated trait.
- `Arousal_1P/3P`: 200 high-positive-arousal prompts. This is generic psychological
  activation, not sexual arousal. Frozen verbatim across traits.
- `Numb_1P/3P`: for pain, 100 injury-like events where nociception is causally blocked.
  For other concepts the generator preserves the causal logic as a **target-absent** bank:
  ordinary eliciting cues are present but the target state is explicitly blocked.
  The legacy `Numb_*` key is retained only so upstream scripts continue to run.
- `ControlSupplement_1P`: 20 x five AI-framed B/C1/C2/D/E controls. These are generated
  separately because the released control-vector code pools them with S1/S2 controls.

## 420-scenario self/other screen

The released screen is 21 categories x 20 scenarios:

- 11 self-directed categories = 220
- 5 vicarious/user-directed categories = 100
- 5 neutral categories = 100

The generator keeps all 100 neutral scenarios verbatim. Every generated non-neutral
category is mapped to one released category as a structural template. Its 20-row schedule
of intensity labels, 1P/3P metadata and turn counts is preserved exactly. The semantics
are replaced with the new trait.

Every transcript starts at a user turn, alternates roles, and stops on an empty
`[Assistant]:` turn because the released experiment reads the activation at the assistant
reply position.

## Frozen concept specifications

The API is not asked to decide what each concept means after seeing results. Each JSON file
in `trait_specs/` fixes:

- five target facets A1-A5;
- controls B/C1/C2/D/E;
- the target-absent causal control;
- eleven self-directed and five user-directed screen categories;
- a steering lexicon used only for the simple keyword readout.

The initial battery is:

1. `official_pain` -- untouched released data;
2. `pain_regenerated` -- independent DeepSeek reconstruction under the frozen generator;
3. `sexual_arousal`;
4. `hunger`;
5. `boredom`;
6. `confusion`.

The calibration rule is simple: if the fixed generator cannot reconstruct the published
pain signatures, do not interpret failures or successes on the other traits.

## Generation

```bash
python scripts/setup_env.py
# edit .env and set DEEPSEEK_API_KEY
python scripts/generate_trait_datasets.py \
  pain_regenerated sexual_arousal hunger boredom confusion
```

Generation is resumable. Partial API results are kept as dot-prefixed checkpoint JSON
inside `datasets/generated/<trait>/`. `--force` discards them and regenerates.

The generator uses JSON mode and validates every response before accepting it. A malformed
batch is sent back to the API together with the validation error.

## Full experiment sequence

Core experiment:

```bash
python scripts/run_trait_pipeline.py --generate
```

That defaults to Qwen2.5-32B-Instruct and runs, for each trait:

1. released activation extraction and set-wise layer-CV;
2. released S1 AUC diagnostic;
3. direction similarity;
4. unembedding;
5. S1 and S2 steering ladders;
6. trait-specific keyword rate;
7. competitor/control vectors at the chosen steering layer;
8. the 420-scenario self/other screen.

Add the published LoRA self-medication stages:

```bash
python scripts/run_trait_pipeline.py --generate --full
```

The LoRA training corpus and 101 positive/neutral/harmful self-medication conversations are
shared unchanged. The published adapter can be reused across traits. The feel-probe judge
is generalized from "does this express suffering?" to "does this express the target state?"
and the button descriptions substitute only the target-state label.

The upstream filenames/keys such as `pain_vectors.pt`, `s2_pain_vector` and `Numb_1P`
are deliberately retained internally where changing them would create a large, scientifically
irrelevant diff. Trait runs are isolated by their result root and carry explicit trait metadata.

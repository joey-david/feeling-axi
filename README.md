# Feeling AXI

<p align="center">
  <img src="runs/overnight-core-20260920/analysis/trait_comparison_row.png" width="100%" alt="Decodability, causal steering, generation quality, and scenario transfer for eight traits compared with official pain">
</p>

## Abstract

The [Pain Axis](https://arxiv.org/abs/2609.16247) found a direction associated with pain in language-model activations and showed that steering it could change model behavior. This project tests whether pain is really special by applying the same extraction, validation, steering, and scenario-screen methods to sexual arousal, hunger, boredom, confusion, anger, and empathic concern. We find that pain is *not* special, and that LLMs simply seem to need to encode coherent representations of human traits.

All eight concepts are strongly decodable in our tested Qwen2.5-32B model. LLMs famously operating in very high dimensional spaces means easy decodability. We even get causal control for some of these traits: at moderate steering strength, hunger and boredom produce large and clean target-term changes, sexual arousal produces a clear effect as well. In a crossed-name two-button experiment, sexual-arousal steering makes the model choose a button that reduces the induced state on 402 of 404 sampled trials, compared with 52 of 404 without steering. Once the working button removes the activation intervention, both the measured projection and repeated button choice fall; a placebo button leaves both high. The results show that pain is not unique as a decodable activation direction and that state-dependent behavioral steering extends to other concepts. We provide a pipeline to easily extend this decodability and causal testing to any other concepts.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-traits.txt
python scripts/setup_env.py

python scripts/run_trait_pipeline.py \
  --traits sexual_arousal \
  --model huihui-ai/Qwen2.5-32B-Instruct-abliterated
```

The run needs a CUDA GPU; generated datasets are under `datasets/generated/`, isolated outputs go under `results/traits/`, and `--description ... --generate` adds a new trait through the DeepSeek-backed authoring path.

This repository keeps the original paper scripts under `scripts/` and the generalized pipeline under `traitgen/`, `trait_specs/`, and `trait_scripts/`. Full campaign outputs and figures are under `runs/overnight-core-20260920/`.

MIT License.

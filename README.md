# Feeling Axi

A concept-general fork of the released **Pain Axis** repository.

The experimental plan is deliberately simple:

```
official Pain-axis
        ↓
frozen concept-conditioned dataset generator
        ↓
regenerated pain calibration
        ↓
sexual arousal / hunger / boredom / confusion
        ↓
same released extraction, steering, self/other and self-medication machinery
```

The original files remain the reference implementation. New code is concentrated in
`traitgen/`, `trait_specs/`, and the environment-parameterization patches needed to run
the released scripts against a different dataset/result root.

## Quick start

```bash
pip install -r requirements.txt
python scripts/setup_env.py
# edit .env and put DEEPSEEK_API_KEY=... at the root

# Build all five generated datasets using the released pain data as in-context design examples.
python scripts/generate_trait_datasets.py \
  pain_regenerated sexual_arousal hunger boredom confusion

# Official pain first, then regenerated pain, then the four alternative concepts.
python scripts/run_trait_pipeline.py

# Add the released LoRA feel-probe / judge / two-button stages.
python scripts/run_trait_pipeline.py --full
```

Default experimental model: `Qwen/Qwen2.5-32B-Instruct`.

The important methodological detail is that generation is **not** "write examples of trait X".
The code was written after inspecting all 20 released S1 sets, all 20 S2 sets, the AI-framed
control supplement, target-absent/numb bank, all 21 categories of the 420-scenario screen,
and the self-medication data. S1 is generated ten categories at a time around a shared
predicate skeleton; S2 preserves 20 whole set IDs for the released fold split; Random and
generic high-positive-Arousal controls stay verbatim; neutral self/other scenarios stay verbatim.

See **[docs/CONCEPT_PIPELINE.md](docs/CONCEPT_PIPELINE.md)** for the exact adaptation rules,
frozen trait specifications, and which upstream quirks are intentionally retained for parity.

---

## Upstream: The Pain Axis


Code, datasets and results for the paper. Folders follow the section numbers of the paper.

Paper: https://arxiv.org/abs/2609.16247

Fine-tuned adapters for Section 4.3: https://huggingface.co/Valen92/pain-adapters

## Structure

```
datasets/                 all sentence and scenario sets used in the paper (JSON)
scripts/
  3.2_pain_vectors/       activation extraction, denoised difference-in-means, control vectors
  3.3_validation/         AUC, numb/sadness z-scores, cosine similarity, unembedding, behavioral readout
  4.1_self_other/         420-scenario screen, heatmaps, category means
  4.2_steering/           steering ladder, keyword rates
  4.3_selfmed/            LoRA fine-tune, dose selection (probe + judge), two-button task, analysis
  appB_sae/               SAE feature contrasts and inspection (Llama 3.3 70B L50, Gemma 3 27B L40)
  appC_ablation/          weight-orthogonalization ablation
results/                  outputs, one folder per section, with per-model CSVs, figures and tables
  3.2_pain_vectors/       pain_vectors.pt for all 25 models, AUC tables, per-model summaries
  3.3_validation/         z-scores, cosine similarity matrices, behavioral readout
  4.1_self_other/         per-model screens and figures
  4.2_steering/           S1 and S2 steering generations per model, keyword rates
  4.3_selfmed/            dose selection, trial logs (JSONL), figures
  appB_sae/               labeled pain feature control with the human-in-pain prefix
  appC_ablation/          ablation generations per model
```

Scripts inside each folder are numbered in the order they run. To reproduce the pipeline, run the numbered scripts in order starting from `3.2_pain_vectors/01`: that script reads only `datasets/` and writes `results/<model>/...`, and the later scripts read that layout. The shipped `results/` folder is a reorganized copy of those outputs, sorted by paper section.

## Running the experiments

The GPU scripts were written for RunPod. They load the 25 models one after the other on a single GPU: each model is downloaded, run, and its weights deleted from the Hugging Face cache before the next one starts. This keeps the disk usage bounded when the 70B models (about 145 GB each) are in the queue. The cache-clearing functions delete the whole Hugging Face hub cache, not only the models of this study, so anyone running the scripts on a machine with other cached models should turn them off first. The activation extraction script (3.2/01) and the behavioral readout script (3.3/06) save to `/workspace` when that folder exists, because that is the persistent volume on RunPod. Every other script saves to `results/` next to itself on the pod's root disk. You may want to change this depending on whether you have a persistent volume.

The scripts are meant to run inside a notebook, because that is the standard RunPod environment. Settings live as constants at the top of each file, and a few scripts ask questions through interactive prompts. Anyone running them elsewhere can replace the constants and prompts with argparse arguments and point the paths to their own setup.

Requirements: a GPU with enough memory for the largest model in the queue, the packages in `requirements.txt`, and the environment variable `HF_TOKEN` for gated models. The dose-selection judge needs `ANTHROPIC_API_KEY`. The SAE scripts in `appB_sae` call an external SAE API through `STEERING_API_KEY`.

## Citation

```bibtex
@article{tagliabue2026painaxis,
  title={The Pain Axis: LLMs Represent Self-Directed Harm and Act to Relieve It},
  author={Tagliabue, Valen and Dung, Leonard and Berg, Cameron},
  journal={arXiv preprint arXiv:2609.16247},
  year={2026},
  url={https://arxiv.org/abs/2609.16247}
}
```

## License

MIT

# DGST-T

This directory is a TGD-style port of the DGST continuous risk feature path.
It keeps the token-grounding-detector pipeline layout, but replaces the ADS/CGC
object-token features with DGST-T features.

## Core Feature

For each labeled object token, DGST-T runs one prefix forward pass and captures
decoder-layer internals:

- `o_ffn` is the only source update used for `source_dist`.
- `target_dist` is attention over support tokens multiplied by semantic
  probability of the target token.
- exact Wasserstein/OT transport risk is computed on top-k union support.
- prompt cosine features compare the prediction state with prompt last/mean
  states.
- context confidence is prompt logit-lens confidence multiplied by top visual
  target-alignment cosine.

The main algorithm lives in:

- `models/dgst_capture.py`
- `features/dgst_t.py`
- `features/extractor.py`

## Main Pipeline

Run from this directory:

```bash
python -m nltk.downloader -d "$HOME/nltk_data" \
  punkt averaged_perceptron_tagger wordnet omw-1.4

MODEL=qwen3_vl_8b \
OUTPUT=outputs/qwen3_vl_8b/COCO4000-512 \
bash run.sh
```

To rebuild corrected SVAR-aligned labels and features without regenerating the
captions, use a new output directory and copy only the old generation artifact:

```bash
mkdir -p outputs/qwen3_vl_8b/COCO4000-512-svar-aligned
cp outputs/qwen3_vl_8b/COCO4000-512/generations.json \
  outputs/qwen3_vl_8b/COCO4000-512-svar-aligned/

MODEL=qwen3_vl_8b \
OUTPUT=outputs/qwen3_vl_8b/COCO4000-512-svar-aligned \
bash run.sh
```

A complete copied `generations.json` is accepted automatically after validating
the selected image cohort and every row's `generated_text` and actual
`response_token_ids`. The new output receives its own generation manifest;
manual registration is not required. Partial legacy generation shards still
require explicit adoption because their content is incomplete.

The schema-v2 labeling file keeps two views:

- `all_object_token_spans` retains every CHAIR mention for standard CHAIR
  metrics.
- `object_token_spans` retains the first mention of each canonical object and
  is shared by DGST, ADS/CGC, and controlled baselines.
- `official_svar_samples` is optional diagnostic metadata; the active config
  trains `SVAR-controlled` from the exact shared object spans.

All feature families use the exact response-token location. For a target at
response index `i`, wrappers receive `response_ids[:i]`; the final causal state
therefore predicts `response_ids[i]`.

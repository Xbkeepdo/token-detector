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
python scripts/extract_features.py \
  --model qwen2_5_vl_7b \
  --config configs/model_configs.yaml \
  --output-dir outputs/qwen2_5_vl_7b/COCO500 \
  --device cuda \
  --resume

python scripts/train_and_eval.py \
  --model qwen2_5_vl_7b \
  --config configs/model_configs.yaml \
  --output-dir outputs/qwen2_5_vl_7b/COCO500
```

`features.pkl` rows contain `dgst_t_score`, `dgst_t_per_layer`,
`dgst_t_feature_vector`, prompt cosine curves, and context-confidence curves.

The POPE-specific legacy extractor was copied with the TGD tree but is not the
primary migrated path.



python scripts/train_feature_sets.py \
  --model internvl_2_5_8b \
  --config configs/model_configs.yaml \
  --output-dir outputs/internvl_2_5_8b/COCO500 \
  --feature-sets \
    risk \
    risk_capped_topmass_085 \
    risk_capped_topmass_085+context_confidence \
    risk_capped_topmass_085+context_confidence_max_prompt \
    risk_capped_topmass_085+target_visual_hidden_cosine \
  --classifiers xgb rf mlp \
  --scoring auc
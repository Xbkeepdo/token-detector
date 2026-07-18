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


## POPE, CLEVR, and AMBER VQA comparisons

The unified QA path compares the current six-branch DGST method, ADS+CGC, and
MetaToken/SVAR-controlled/DHCP/ProjectAway/HalLoc on one shared, image-level
8:2 split with no validation set. Training uses fixed epochs, the final
checkpoint and a Real-F1 threshold selected on train only. COCO and QA share
`configs/model_configs_unified.yaml`;
`configs/model_configs_server_fj01.yaml` contains the same settings with fj01
paths. The `qa_benchmarks` section stores only QA dataset/protocol settings,
while models, feature extraction and probe hyperparameters are shared.

Run one complete benchmark with the same compact launcher style as the original
POPE script:

```bash
MODEL=qwen3_vl_8b bash run_pope.sh
MODEL=qwen3_vl_8b bash run_clevr.sh
MODEL=qwen3_vl_8b bash run_amber.sh
```

`run_qa.sh` is the common implementation. It prepares the fixed split, resumes
generation/labeling/extraction, trains seeds 42/43/44, trains the native paper
baselines, and writes comparison tables. QA extraction is selected in the same
YAML with `qa_benchmarks.extraction_mode`: `all`, `method_only`,
`ads_cgc_only`, or `baseline_only`. In `all` mode the prompt-last wrapper is
called once per question and the same `ModelOutput` is consumed by DGST,
ADS+CGC, and every enabled baseline. Baseline payloads remain isolated under
`baseline/<label_protocol>/`; `baseline_only` never creates or overwrites the
root `features.pkl`.

By default native baselines use the POPE-style
`object_hallucination_yes_only` protocol. To additionally run the all-answer
correctness track:

Generation and the joint feature extraction default to two question-sharded
workers on `cuda:0 cuda:1`. Each worker writes isolated
resume shards and the parent process validates and atomically consolidates the
artifacts. Probe and baseline training remain on `DEVICE` (default `cuda:0`).
For a deliberate single-GPU run, set both device lists explicitly:

```bash
GENERATION_DEVICES="cuda:0" FEATURE_DEVICES="cuda:0" \
MODEL=qwen3_vl_8b bash run_pope.sh
```

```bash
BASELINE_LABEL_PROTOCOLS="object_hallucination_yes_only answer_correctness_all" \
MODEL=qwen3_vl_8b bash run_pope.sh
```

The first VQA experiment uses only `prompt_last_token`. Extraction fixes the
response target index to `0`, so the strict prefix contains no generated
response tokens and the selected causal row is exactly the final token of the
complete prompt. That row predicts `response_token_ids[0]` and does not move if
a model later emits a preamble before its semantic yes/no answer. This is the
default `position_protocols` value in the QA YAML, so no second object-position
forward is run. Generation still saves and validates the actual semantic
yes/no token for labeling, but that semantic location does not choose the
feature row.

The implementation retains `question_object_pre_token` as an optional later
ablation. Enabling it in YAML locates the queried object surface in the complete
contextualized question; if its first actual sub-token is `j`, the strict prefix
ends before `j` and the final causal row predicts that contextual token. POPE
provides exact object spans, while CLEVR uses the entity head consumed by the
terminal `exist` program and reports coverage explicitly.
The two reporting protocols remain separate: `object_hallucination_yes_only`
measures false-positive object hallucination, while `answer_correctness_all`
measures general yes/no answer errors. Both use `0=hallucination/error, 1=real`,
report real as the headline positive class, and also report hallucination
metrics.

AMBER uses all 14,216 official discriminative Yes/No questions over 1004
images (existence, attribute, and relation). The deterministic seed-42 outer
split is defined over physical images (803 train / 201 test); question counts
need not be exactly 80/20 because AMBER has a variable number of questions per
image. Its active feature position is the same `prompt_last_token` used by
POPE and CLEVR.

Some official AMBER source images are as large as 54 MP. The unified YAML
therefore applies `max_pixels: 200704` only to `amber_discriminative`, using
the same bounded visual grid for generation and feature extraction. COCO,
POPE, and CLEVR preprocessing is unchanged.

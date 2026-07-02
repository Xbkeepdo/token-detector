"""Feature extraction for MS-COCO image captioning. Runs prefix forward passes and computes DGST-T per token."""

from __future__ import annotations
import os
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from models.base_wrapper import BaseLVLMWrapper, GenerationOutput
from features.attention import compute_alpha_img_alpha_text
from features.dgst_t import compute_dgst_t, _baseline_excess_score, _feature_vector
from utils.io_utils import load_json, load_pkl, save_pkl


def extract_features_for_dataset(
    model_wrapper: BaseLVLMWrapper,
    coco_samples: List[dict],
    labeling_results: Dict[int, dict],
    cfg_dgst_t: dict,
    output_path: str,
    resume: bool = True,
) -> List[dict]:
    """Main entry point.  Iterates over `coco_samples`, extracts features"""
    all_features: List[dict] = []
    done_image_ids = set()
    output_dir = os.path.dirname(output_path)
    generation_results = _load_generation_results(output_dir)
    if resume and os.path.exists(output_path):
        all_features = load_pkl(output_path)
        done_image_ids = {f["image_id"] for f in all_features}
        print(f"[Extractor] Resuming — {len(done_image_ids)} images already done.")

    for sample in tqdm(coco_samples, desc="Extracting features"):
        image_id = sample["image_id"]
        if image_id in done_image_ids:
            continue

        label_info = labeling_results.get(image_id)
        if label_info is None:
            continue

        labeling_generated_text = label_info.get("generated_text", "")
        if not labeling_generated_text:
            continue
        labeling_response_ids = model_wrapper.tokenizer.encode(
            labeling_generated_text, add_special_tokens=False
        )
        if image_id in generation_results:
            raw_response_ids = generation_results[image_id].get("response_token_ids", [])
            if raw_response_ids:
                labeling_response_ids = [int(token_id) for token_id in raw_response_ids]
        gen_out = GenerationOutput(
            image_id=image_id,
            generated_text=labeling_generated_text,
            response_token_ids=labeling_response_ids,
            response_tokens=[
                model_wrapper.tokenizer.decode([tid], skip_special_tokens=False)
                for tid in labeling_response_ids
            ],
        )

        image = Image.open(sample["image_path"]).convert("RGB")

        object_token_spans = label_info.get("object_token_spans", [])
        if not object_token_spans:
            continue

        image_features = []
        target_token_aggregation = _target_token_aggregation_mode(cfg_dgst_t)
        valid_spans = []
        flat_spans = []
        flat_span_offsets = []
        flat_response_indices = []
        flat_target_token_ids = []
        for span in object_token_spans:
            response_indices = _span_response_indices(
                span,
                response_length=len(gen_out.response_token_ids),
                aggregation_mode=target_token_aggregation,
            )
            if not response_indices:
                continue
            target_token_ids = [
                int(gen_out.response_token_ids[index])
                for index in response_indices
            ]
            span_offset = len(valid_spans)
            valid_spans.append(span)
            for response_index, target_token_id in zip(response_indices, target_token_ids):
                flat_spans.append(span)
                flat_span_offsets.append(span_offset)
                flat_response_indices.append(int(response_index))
                flat_target_token_ids.append(int(target_token_id))

        if not valid_spans:
            continue

        try:
            model_outputs = model_wrapper.extract_token_features_batch(
                image=image,
                response_token_ids=gen_out.response_token_ids,
                response_token_indices=flat_response_indices,
                target_token_ids=flat_target_token_ids,
                cfg_dgst_t=cfg_dgst_t,
            )
        except Exception as e:
            import traceback
            print(f"[Extractor] Warning — batch forward failed for image {image_id}: {e}")
            traceback.print_exc()
            model_outputs = _extract_token_features_fallback(
                model_wrapper=model_wrapper,
                image=image,
                image_id=image_id,
                spans=flat_spans,
                response_token_ids=gen_out.response_token_ids,
                response_indices=flat_response_indices,
                target_token_ids=flat_target_token_ids,
            )

        if len(model_outputs) != len(flat_response_indices):
            print(
                f"[Extractor] Warning — image {image_id} returned "
                f"{len(model_outputs)} outputs for {len(flat_response_indices)} object sub-tokens."
            )

        grouped_token_features = [[] for _ in valid_spans]
        for span_offset, response_index, target_token_id, model_out in zip(
            flat_span_offsets,
            flat_response_indices,
            flat_target_token_ids,
            model_outputs,
        ):
            if model_out is None:
                continue
            feat = _build_feature_record(
                image_id=image_id,
                span=valid_spans[int(span_offset)],
                response_index=int(response_index),
                target_token_id=int(target_token_id),
                model_out=model_out,
                cfg_dgst_t=cfg_dgst_t,
                target_token_aggregation=target_token_aggregation,
            )
            grouped_token_features[int(span_offset)].append(feat)

        for span_offset, token_features in enumerate(grouped_token_features):
            if not token_features:
                continue
            if target_token_aggregation == "risk_mean" and len(token_features) > 1:
                feat = _aggregate_word_token_features(
                    token_features,
                    cfg_dgst_t=cfg_dgst_t,
                )
            else:
                feat = token_features[0]
                feat["target_token_ids"] = [int(feat["target_token_id"])]
                feat["response_token_indices"] = [int(feat["response_token_idx"])]
                feat["target_token_count"] = 1
                feat["target_token_aggregation"] = target_token_aggregation
            image_features.append(feat)

        all_features.extend(image_features)

        save_pkl(all_features, output_path)

    print(f"[Extractor] Done. {len(all_features)} DGST-T object tokens saved to {output_path}.")
    return all_features


def _load_generation_results(output_dir: str) -> dict[int, dict]:
    path = os.path.join(output_dir, "generations.json")
    if not os.path.exists(path):
        return {}
    raw = load_json(path)
    return {int(key): value for key, value in raw.items()}


def _build_feature_record(
    *,
    image_id: int,
    span: dict,
    response_index: int,
    target_token_id: int,
    model_out,
    cfg_dgst_t: dict,
    target_token_aggregation: str,
) -> dict:
    dgst_t = _compute_dgst_t_result(model_out, cfg_dgst_t)
    alpha_img_per_layer, alpha_text_per_layer = compute_alpha_img_alpha_text(
        text_to_patch_attn=model_out.text_to_patch_attn,
        text_to_text_attn=model_out.text_to_text_attn,
    )
    baseline = _compute_baseline_features(model_out)

    feat = {
        "image_id":              image_id,
        "token_str":             span["word"],
        "token_id":              model_out.token_id,
        "target_token_id":       int(target_token_id),
        "target_token_ids":      [int(target_token_id)],
        "response_token_idx":    int(response_index),
        "response_token_indices": [int(response_index)],
        "target_token_count":    1,
        "target_token_aggregation": target_token_aggregation,
        "label":                 span["label"],
        "dgst_t_score":          dgst_t["dgst_t_score"],
        "dgst_t_per_layer":      dgst_t["dgst_t_per_layer"].tolist(),
        "dgst_t_transport_risk_per_layer": dgst_t["dgst_t_transport_risk_per_layer"].tolist(),
        "dgst_t_prompt_last_cosine_per_layer": dgst_t["dgst_t_prompt_last_cosine_per_layer"].tolist(),
        "dgst_t_prompt_mean_cosine_per_layer": dgst_t["dgst_t_prompt_mean_cosine_per_layer"].tolist(),
        "dgst_t_atarget_visual_cosine_per_layer": dgst_t["dgst_t_atarget_visual_cosine_per_layer"].tolist(),
        "dgst_t_target_visual_hidden_cosine_per_layer": dgst_t.get(
            "dgst_t_target_visual_hidden_cosine_per_layer",
            dgst_t["dgst_t_atarget_visual_cosine_per_layer"],
        ).tolist(),
        "dgst_t_target_visual_prompt_hidden_cosine_per_layer": dgst_t.get(
            "dgst_t_target_visual_prompt_hidden_cosine_per_layer",
            dgst_t.get(
                "dgst_t_target_visual_hidden_cosine_per_layer",
                dgst_t["dgst_t_atarget_visual_cosine_per_layer"],
            ),
        ).tolist(),
        "dgst_t_target_visual_hidden_cosine_capped_topmass_085_per_layer": dgst_t.get(
            "dgst_t_target_visual_hidden_cosine_capped_topmass_085_per_layer",
            dgst_t.get(
                "dgst_t_target_visual_hidden_cosine_per_layer",
                dgst_t["dgst_t_atarget_visual_cosine_per_layer"],
            ),
        ).tolist(),
        "dgst_t_target_visual_prompt_hidden_cosine_capped_topmass_085_per_layer": dgst_t.get(
            "dgst_t_target_visual_prompt_hidden_cosine_capped_topmass_085_per_layer",
            dgst_t.get(
                "dgst_t_target_visual_prompt_hidden_cosine_per_layer",
                dgst_t.get(
                    "dgst_t_target_visual_hidden_cosine_per_layer",
                    dgst_t["dgst_t_atarget_visual_cosine_per_layer"],
                ),
            ),
        ).tolist(),
        "dgst_t_prompt_confidence_top3_per_layer": dgst_t.get(
            "dgst_t_prompt_confidence_top3_per_layer",
            _layer_stat_tensor(dgst_t["dgst_t_layer_stats"], "prompt_logit_lens_top3_confidence"),
        ).tolist(),
        "dgst_t_prompt_confidence_max_per_layer": dgst_t.get(
            "dgst_t_prompt_confidence_max_per_layer",
            _layer_stat_tensor(dgst_t["dgst_t_layer_stats"], "prompt_logit_lens_top3_confidence"),
        ).tolist(),
        "dgst_t_context_confidence_per_layer": dgst_t["dgst_t_context_confidence_per_layer"].tolist(),
        "dgst_t_context_confidence_max_prompt_per_layer": dgst_t.get(
            "dgst_t_context_confidence_max_prompt_per_layer",
            dgst_t["dgst_t_context_confidence_per_layer"],
        ).tolist(),
        "dgst_t_feature_vector": dgst_t["dgst_t_feature_vector"],
        "dgst_t_layer_stats":    dgst_t["dgst_t_layer_stats"],
        "alpha_img_per_layer":   alpha_img_per_layer.tolist(),
        "alpha_text_per_layer":  alpha_text_per_layer.tolist(),
        **baseline,
    }
    if "dgst_t_transport_risk_topmass_085_per_layer" in dgst_t:
        feat["dgst_t_transport_risk_topmass_085_per_layer"] = dgst_t[
            "dgst_t_transport_risk_topmass_085_per_layer"
        ].tolist()
    if "dgst_t_transport_risk_capped_topmass_085_per_layer" in dgst_t:
        feat["dgst_t_transport_risk_capped_topmass_085_per_layer"] = dgst_t[
            "dgst_t_transport_risk_capped_topmass_085_per_layer"
        ].tolist()
    if "dgst_t_score_relative_vll" in dgst_t:
        feat["dgst_t_score_relative_vll"] = dgst_t["dgst_t_score_relative_vll"]
    if "dgst_t_score_visual_prompt_relative_vll" in dgst_t:
        feat["dgst_t_score_visual_prompt_relative_vll"] = dgst_t[
            "dgst_t_score_visual_prompt_relative_vll"
        ]
    for key in (
        "dgst_t_transport_risk_relative_vll_per_layer",
        "dgst_t_transport_risk_relative_vll_capped_topmass_085_per_layer",
        "dgst_t_target_visual_hidden_cosine_relative_vll_per_layer",
        "dgst_t_target_visual_hidden_cosine_relative_vll_capped_topmass_085_per_layer",
        "dgst_t_transport_risk_visual_prompt_relative_vll_per_layer",
        "dgst_t_transport_risk_visual_prompt_relative_vll_capped_topmass_085_per_layer",
        "dgst_t_target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_per_layer",
        "dgst_t_target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_capped_topmass_085_per_layer",
    ):
        if key in dgst_t:
            feat[key] = dgst_t[key].tolist()
    return feat


def _compute_dgst_t_result(model_out, cfg_dgst_t: dict) -> dict:
    dgst_t = model_out.dgst_t_result
    if dgst_t is not None:
        return dgst_t
    if model_out.dgst_t_raw is None:
        raise RuntimeError("Wrapper did not return DGST-T captures or result.")
    return compute_dgst_t(
        model_out.dgst_t_raw,
        tau=cfg_dgst_t.get("tau", 0.07),
        transport_top_k=cfg_dgst_t.get("transport_top_k", 64),
        cost_mode=cfg_dgst_t.get("cost_mode", "direct"),
        lambda_d=cfg_dgst_t.get("lambda_d", 1.0),
        lambda_s=cfg_dgst_t.get("lambda_s", 1.0),
        lambda_t=cfg_dgst_t.get("lambda_t", 1.0),
        lambda_int=cfg_dgst_t.get("lambda_int", 1.0),
        baseline_layers=cfg_dgst_t.get("baseline_layers", 10),
        risk_start_layer=cfg_dgst_t.get("risk_start_layer", 15),
        alpha=cfg_dgst_t.get("alpha", 2.0),
        ot_solver=cfg_dgst_t.get("ot_solver", "linprog"),
        atarget_visual_top_k=cfg_dgst_t.get("atarget_visual_top_k", 32),
        topmass_alpha=cfg_dgst_t.get("topmass_085_alpha", 0.85),
        capped_topmass_alpha=cfg_dgst_t.get("capped_topmass_085_alpha", 0.85),
        capped_topmass_min_k=cfg_dgst_t.get("capped_topmass_085_min_k", 32),
        capped_topmass_max_k=cfg_dgst_t.get("capped_topmass_085_max_k", 64),
        compute_topmass_085=cfg_dgst_t.get("compute_topmass_085", True),
        compute_capped_topmass_085=cfg_dgst_t.get("compute_capped_topmass_085", True),
        target_gate_mode=cfg_dgst_t.get("target_gate_mode", "legacy_prob"),
        relative_vll_mad_epsilon=cfg_dgst_t.get("relative_vll_mad_epsilon", 1e-6),
    )


def _aggregate_word_token_features(token_features: Sequence[dict], *, cfg_dgst_t: dict) -> dict:
    if not token_features:
        raise ValueError("Cannot aggregate an empty token feature list.")
    result = dict(token_features[0])
    result["target_token_ids"] = [int(feat["target_token_id"]) for feat in token_features]
    result["response_token_indices"] = [int(feat["response_token_idx"]) for feat in token_features]
    result["target_token_count"] = int(len(token_features))
    result["target_token_aggregation"] = "risk_mean"

    for key in sorted(_per_layer_feature_keys(token_features)):
        result[key] = _average_numeric_vectors(token_features, key)
    if "dgst_t_transport_risk_per_layer" in result:
        result["dgst_t_per_layer"] = list(result["dgst_t_transport_risk_per_layer"])
    if all("dgst_t_layer_stats" in feat for feat in token_features):
        result["dgst_t_layer_stats"] = _average_layer_stats(
            [feat["dgst_t_layer_stats"] for feat in token_features]
        )
    _recompute_dgst_scores(result, cfg_dgst_t)
    return result


def _per_layer_feature_keys(features: Sequence[dict]) -> set[str]:
    common = set(features[0])
    for feat in features[1:]:
        common &= set(feat)
    return {key for key in common if key.endswith("_per_layer")}


def _average_numeric_vectors(features: Sequence[dict], key: str) -> list:
    arrays = [np.asarray(feat[key], dtype=np.float32) for feat in features]
    first_shape = arrays[0].shape
    if any(array.shape != first_shape for array in arrays):
        raise ValueError(f"Cannot average {key}: per-token shapes differ.")
    return np.stack(arrays, axis=0).mean(axis=0).tolist()


def _average_layer_stats(layer_stats_by_token: Sequence[list[dict]]) -> list[dict]:
    layer_count = min(len(stats) for stats in layer_stats_by_token)
    averaged = []
    for layer_idx in range(layer_count):
        first = dict(layer_stats_by_token[0][layer_idx])
        merged = {}
        keys = set().union(*(stats[layer_idx].keys() for stats in layer_stats_by_token))
        for key in sorted(keys):
            if key == "layer":
                merged[key] = int(first.get(key, layer_idx + 1))
                continue
            values = [
                stats[layer_idx].get(key)
                for stats in layer_stats_by_token
                if key in stats[layer_idx]
            ]
            if len(values) == len(layer_stats_by_token) and all(_is_number(value) for value in values):
                merged[key] = float(np.asarray(values, dtype=np.float64).mean())
            elif key in first:
                merged[key] = first[key]
        averaged.append(merged)
    return averaged


def _recompute_dgst_scores(feat: dict, cfg_dgst_t: dict) -> None:
    baseline_layers = cfg_dgst_t.get("baseline_layers", 10)
    risk_start_layer = cfg_dgst_t.get("risk_start_layer", 15)
    alpha = cfg_dgst_t.get("alpha", 2.0)
    score_keys = {
        "dgst_t_score": "dgst_t_transport_risk_per_layer",
        "dgst_t_score_relative_vll": "dgst_t_transport_risk_relative_vll_per_layer",
        "dgst_t_score_visual_prompt_relative_vll": (
            "dgst_t_transport_risk_visual_prompt_relative_vll_per_layer"
        ),
    }
    for score_key, risk_key in score_keys.items():
        if risk_key not in feat:
            continue
        feat[score_key] = float(
            _baseline_excess_score(
                torch.tensor(feat[risk_key], dtype=torch.float32),
                baseline_layers=baseline_layers,
                risk_start_layer=risk_start_layer,
                alpha=alpha,
            )
        )
    required = (
        "dgst_t_transport_risk_per_layer",
        "dgst_t_prompt_last_cosine_per_layer",
        "dgst_t_prompt_mean_cosine_per_layer",
        "dgst_t_context_confidence_per_layer",
    )
    if all(key in feat for key in required):
        feat["dgst_t_feature_vector"] = _feature_vector(
            feat["dgst_t_transport_risk_per_layer"],
            feat["dgst_t_prompt_last_cosine_per_layer"],
            feat["dgst_t_prompt_mean_cosine_per_layer"],
            feat["dgst_t_context_confidence_per_layer"],
        )


def _target_token_aggregation_mode(cfg_dgst_t: dict) -> str:
    value = str(cfg_dgst_t.get("target_token_aggregation", "first")).strip().lower()
    if value in {"first", "first_token"}:
        return "first"
    if value in {"risk_mean", "mean_risk", "all_token_risk_mean"}:
        return "risk_mean"
    raise ValueError("target_token_aggregation must be 'first' or 'risk_mean'.")


def _span_response_indices(span: dict, *, response_length: int, aggregation_mode: str) -> list[int]:
    token_indices = span.get("token_indices") or []
    if aggregation_mode == "first":
        token_indices = token_indices[:1]
    result = []
    seen = set()
    for raw_index in token_indices:
        index = int(raw_index)
        if index < 0 or index >= int(response_length):
            continue
        if index in seen:
            continue
        seen.add(index)
        result.append(index)
    return result


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _extract_token_features_fallback(
    *,
    model_wrapper: BaseLVLMWrapper,
    image: Image.Image,
    image_id: int,
    spans: List[dict],
    response_token_ids: List[int],
    response_indices: List[int],
    target_token_ids: List[int],
):
    outputs = []
    for span, first_idx, target_token_id in zip(spans, response_indices, target_token_ids):
        try:
            outputs.append(
                model_wrapper.extract_token_features(
                    image=image,
                    prefix_token_ids=[int(token_id) for token_id in response_token_ids[: int(first_idx)]],
                    response_token_idx=int(first_idx),
                    target_token_id=int(target_token_id),
                )
            )
        except Exception as exc:
            import traceback
            print(
                f"[Extractor] Warning — fallback forward failed for "
                f"image {image_id}, token '{span.get('word', '')}' "
                f"(first_idx={first_idx}): {exc}"
            )
            traceback.print_exc()
            outputs.append(None)
    return outputs


def _compute_baseline_features(model_out) -> dict:
    """Compute baseline method features from a ModelOutput."""
    result = {}

    if model_out.token_logits is not None:
        logits = model_out.token_logits.numpy().astype(np.float32)
        logits_shifted = logits - logits.max()
        probs = np.exp(logits_shifted)
        probs = probs / probs.sum()

        token_id = model_out.token_id
        token_log_prob = float(np.log(max(probs[token_id], 1e-12)))

        p_valid = probs[probs > 1e-12]
        token_entropy = float(-np.sum(p_valid * np.log(p_valid)) / np.log(len(probs)))

        result["token_logits"] = logits.astype(np.float16)
        result["token_log_prob"] = token_log_prob
        result["token_entropy"] = token_entropy
        result["token_nll"] = -token_log_prob

    attn_np = model_out.text_to_patch_attn.float().numpy()
    n_layers = attn_np.shape[0]

    var_per_layer = attn_np.sum(axis=-1).mean(axis=-1)
    svar_ls = max(0, int(n_layers * 0.15))
    svar_le = min(n_layers, int(n_layers * 0.55))
    result["svar_score"] = float(var_per_layer[svar_ls:svar_le].sum())

    mid_l = n_layers // 2
    result["attn_per_head_mid"] = attn_np[mid_l].mean(axis=-1).tolist()

    return result


def _layer_stat_tensor(layer_stats: list[dict], key: str):
    import torch

    return torch.tensor(
        [float(item.get(key, 0.0)) for item in layer_stats],
        dtype=torch.float32,
    )

"""Feature extraction for MS-COCO image captioning. Runs prefix forward passes and computes DGST-T per token."""

from __future__ import annotations
import os
from typing import Dict, List

import numpy as np
from tqdm import tqdm
from PIL import Image

from models.base_wrapper import BaseLVLMWrapper, GenerationOutput
from features.attention import compute_alpha_img_alpha_text
from features.dgst_t import compute_dgst_t
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
        valid_spans = []
        response_indices = []
        target_token_ids = []
        for span in object_token_spans:
            token_indices = span.get("token_indices") or []
            if not token_indices:
                continue
            first_idx = int(token_indices[0])
            if first_idx >= len(gen_out.response_token_ids):
                continue
            valid_spans.append(span)
            response_indices.append(first_idx)
            target_token_ids.append(int(gen_out.response_token_ids[first_idx]))

        if not valid_spans:
            continue

        try:
            model_outputs = model_wrapper.extract_token_features_batch(
                image=image,
                response_token_ids=gen_out.response_token_ids,
                response_token_indices=response_indices,
                target_token_ids=target_token_ids,
            )
        except Exception as e:
            import traceback
            print(f"[Extractor] Warning — batch forward failed for image {image_id}: {e}")
            traceback.print_exc()
            model_outputs = _extract_token_features_fallback(
                model_wrapper=model_wrapper,
                image=image,
                image_id=image_id,
                spans=valid_spans,
                response_token_ids=gen_out.response_token_ids,
                response_indices=response_indices,
                target_token_ids=target_token_ids,
            )

        if len(model_outputs) != len(valid_spans):
            print(
                f"[Extractor] Warning — image {image_id} returned "
                f"{len(model_outputs)} outputs for {len(valid_spans)} object tokens."
            )

        for span, first_idx, target_token_id, model_out in zip(
            valid_spans,
            response_indices,
            target_token_ids,
            model_outputs,
        ):
            if model_out.dgst_t_raw is None:
                raise RuntimeError("Wrapper did not return dgst_t_raw captures.")
            dgst_t = compute_dgst_t(
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
            )
            alpha_img_per_layer, alpha_text_per_layer = compute_alpha_img_alpha_text(
                text_to_patch_attn=model_out.text_to_patch_attn,
                text_to_text_attn=model_out.text_to_text_attn,
            )

            baseline = _compute_baseline_features(model_out)

            feat = {
                "image_id":              image_id,
                "token_str":             span["word"],
                "token_id":              model_out.token_id,
                "target_token_id":       target_token_id,
                "response_token_idx":    first_idx,
                "label":                 span["label"],
                "dgst_t_score":          dgst_t["dgst_t_score"],
                "dgst_t_per_layer":      dgst_t["dgst_t_per_layer"].tolist(),
                "dgst_t_transport_risk_per_layer": dgst_t["dgst_t_transport_risk_per_layer"].tolist(),
                "dgst_t_prompt_last_cosine_per_layer": dgst_t["dgst_t_prompt_last_cosine_per_layer"].tolist(),
                "dgst_t_prompt_mean_cosine_per_layer": dgst_t["dgst_t_prompt_mean_cosine_per_layer"].tolist(),
                "dgst_t_atarget_visual_cosine_per_layer": dgst_t["dgst_t_atarget_visual_cosine_per_layer"].tolist(),
                "dgst_t_context_confidence_per_layer": dgst_t["dgst_t_context_confidence_per_layer"].tolist(),
                "dgst_t_feature_vector": dgst_t["dgst_t_feature_vector"],
                "dgst_t_layer_stats":    dgst_t["dgst_t_layer_stats"],
                "alpha_img_per_layer":   alpha_img_per_layer.tolist(),
                "alpha_text_per_layer":  alpha_text_per_layer.tolist(),
                **baseline,
            }
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

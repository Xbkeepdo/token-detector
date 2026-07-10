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
from utils.io_utils import append_pkl, load_json, load_pkl


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
            if first_idx < 0 or first_idx >= len(gen_out.response_token_ids):
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
                spans=valid_spans,
                response_token_ids=gen_out.response_token_ids,
                response_indices=response_indices,
                target_token_ids=target_token_ids,
                cfg_dgst_t=cfg_dgst_t,
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
            if model_out is None:
                continue
            feat = _build_feature_record(
                image_id=image_id,
                span=span,
                response_index=int(first_idx),
                target_token_id=int(target_token_id),
                model_out=model_out,
                cfg_dgst_t=cfg_dgst_t,
            )
            image_features.append(feat)

        all_features.extend(image_features)
        if image_features:
            append_pkl(image_features, output_path)

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
        "response_token_idx":    int(response_index),
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
    if "dgst_t_relative_vll_logit_source" in dgst_t:
        feat["dgst_t_relative_vll_logit_source"] = dgst_t[
            "dgst_t_relative_vll_logit_source"
        ]
    if "dgst_t_source_distribution_mode" in dgst_t:
        feat["dgst_t_source_distribution_mode"] = dgst_t[
            "dgst_t_source_distribution_mode"
        ]
    if "dgst_t_relative_cost_mode" in dgst_t:
        feat["dgst_t_relative_cost_mode"] = dgst_t["dgst_t_relative_cost_mode"]
    if "dgst_t_relative_cost_modes" in dgst_t:
        feat["dgst_t_relative_cost_modes"] = list(dgst_t["dgst_t_relative_cost_modes"])
    if "dgst_t_relative_cost_state_modes" in dgst_t:
        feat["dgst_t_relative_cost_state_modes"] = list(
            dgst_t["dgst_t_relative_cost_state_modes"]
        )
    if "dgst_t_dual_scope" in dgst_t:
        feat["dgst_t_dual_scope"] = bool(dgst_t["dgst_t_dual_scope"])
    for key in ("dgst_t_dual_scope_vv_source_target", "dgst_t_dual_scope_vp_source_target"):
        if key in dgst_t:
            feat[key] = str(dgst_t[key])
    for key in (
        "dgst_t_relative_barrier_lambda",
        "dgst_t_relative_barrier_margin",
        "dgst_t_relative_barrier_max",
        "dgst_t_ffn_evidence_top_k",
        "dgst_t_ffn_evidence_rank",
        "dgst_t_ffn_injection_eps",
    ):
        if key in dgst_t:
            feat[key] = (
                int(dgst_t[key])
                if key in {"dgst_t_ffn_evidence_top_k", "dgst_t_ffn_evidence_rank"}
                else float(dgst_t[key])
            )
    for key in (
        "dgst_t_transport_risk_relative_vll_per_layer",
        "dgst_t_transport_risk_relative_vll_capped_topmass_085_per_layer",
        "dgst_t_target_visual_hidden_cosine_relative_vll_per_layer",
        "dgst_t_target_visual_hidden_cosine16_relative_vll_per_layer",
        "dgst_t_target_visual_hidden_cosine_relative_vll_capped_topmass_085_per_layer",
        "dgst_t_target_visual_hpre_cosine_relative_vll_per_layer",
        "dgst_t_target_visual_hpre_cosine16_relative_vll_per_layer",
        "dgst_t_target_visual_hpre_cosine_relative_vll_capped_topmass_085_per_layer",
        "dgst_t_relative_vll_evidence_strength_per_layer",
        "dgst_t_r_es_relative_vll_cost_geo_per_layer",
        "dgst_t_transport_risk_visual_prompt_relative_vll_per_layer",
        "dgst_t_transport_risk_visual_prompt_relative_vll_capped_topmass_085_per_layer",
        "dgst_t_target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_per_layer",
        "dgst_t_target_visual_prompt_hidden_cosine16_visual_prompt_relative_vll_per_layer",
        "dgst_t_target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_capped_topmass_085_per_layer",
        "dgst_t_target_visual_prompt_hpre_cosine_visual_prompt_relative_vll_per_layer",
        "dgst_t_target_visual_prompt_hpre_cosine16_visual_prompt_relative_vll_per_layer",
        "dgst_t_target_visual_prompt_hpre_cosine_visual_prompt_relative_vll_capped_topmass_085_per_layer",
        "dgst_t_visual_prompt_relative_vll_target_visual_mass_per_layer",
        "dgst_t_visual_prompt_relative_vll_target_prompt_mass_per_layer",
        "dgst_t_visual_prompt_relative_vll_evidence_visual_mass_per_layer",
        "dgst_t_visual_prompt_relative_vll_evidence_prompt_mass_per_layer",
        "dgst_t_visual_prompt_relative_vll_evidence_strength_per_layer",
        "dgst_t_visual_prompt_relative_vll_source_visual_mass_per_layer",
        "dgst_t_visual_prompt_relative_vll_source_prompt_mass_per_layer",
        "dgst_t_m_p_per_layer",
        "dgst_t_c_vp_relative_vll_cost_geo_per_layer",
        "dgst_t_vv_source_entropy_per_layer",
        "dgst_t_vv_target_entropy_per_layer",
        "dgst_t_vv_evidence_entropy_per_layer",
        "dgst_t_vv_source_topk_entropy_per_layer",
        "dgst_t_vp_source_entropy_per_layer",
        "dgst_t_vp_target_entropy_per_layer",
        "dgst_t_vp_evidence_entropy_per_layer",
        "dgst_t_vp_source_topk_entropy_per_layer",
    ):
        if key in dgst_t:
            feat[key] = dgst_t[key].tolist()
    for key in (
        "dgst_t_vv_support_positions",
        "dgst_t_vp_support_positions",
    ):
        if key in dgst_t:
            feat[key] = [int(position) for position in dgst_t[key]]
    for key, value in dgst_t.items():
        if not key.endswith(
            (
                "_capped_topmass_085_support_indices_per_layer",
                "_capped_topmass_085_support_positions_per_layer",
            )
        ):
            continue
        feat[key] = [[int(item) for item in layer_values] for layer_values in value]
    for key in (
        "dgst_t_vv_attention_dist_per_layer",
        "dgst_t_vp_attention_dist_per_layer",
        "dgst_t_vv_support_attention_per_layer",
        "dgst_t_vp_support_attention_per_layer",
        "dgst_t_vv_source_dist_per_layer",
        "dgst_t_vp_source_dist_per_layer",
        "dgst_t_vv_source_hmid_proj_dist_per_layer",
        "dgst_t_vp_source_hmid_proj_dist_per_layer",
        "dgst_t_vv_source_hprev_cos_dist_per_layer",
        "dgst_t_vp_source_hprev_cos_dist_per_layer",
        "dgst_t_vv_source_hprev_proj_dist_per_layer",
        "dgst_t_vp_source_hprev_proj_dist_per_layer",
        "dgst_t_vv_semantic_gate_per_layer",
        "dgst_t_vp_semantic_gate_per_layer",
    ):
        if key in dgst_t:
            feat[key] = dgst_t[key].detach().cpu()
    for key, value in dgst_t.items():
        if key in feat:
            continue
        if (
            key.startswith((
                "dgst_t_risk_",
                "dgst_t_cos_",
                "dgst_t_transport_risk_",
                "dgst_t_js_",
                "dgst_t_kl_",
                "dgst_t_ffn_",
            ))
            and key.endswith("_per_layer")
        ):
            feat[key] = value.tolist() if hasattr(value, "tolist") else value
        elif (
            key.startswith(("dgst_t_vv_", "dgst_t_vp_"))
            and key.endswith("_per_layer")
        ):
            feat[key] = value.tolist() if hasattr(value, "tolist") else value
        elif key.startswith("dgst_t_score_"):
            feat[key] = float(value)
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
        source_distribution_mode=cfg_dgst_t.get("source_distribution_mode", "softmax"),
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
        relative_cost_mode=cfg_dgst_t.get("relative_cost_mode"),
        relative_cost_modes=cfg_dgst_t.get("relative_cost_modes"),
        relative_cost_state_modes=cfg_dgst_t.get("relative_cost_state_modes"),
        relative_cost_update_lambdas=cfg_dgst_t.get("relative_cost_update_lambdas"),
        relative_barrier_lambda=cfg_dgst_t.get("relative_barrier_lambda", 1.0),
        relative_barrier_margin=cfg_dgst_t.get("relative_barrier_margin", 0.5),
        relative_barrier_max=cfg_dgst_t.get("relative_barrier_max", 3.0),
        source_modes=cfg_dgst_t.get("source_modes"),
        target_attention_gammas=cfg_dgst_t.get("target_attention_gammas"),
        target_attention_epsilon=cfg_dgst_t.get("target_attention_epsilon", 1e-12),
        compute_ffn_injection_features=cfg_dgst_t.get("compute_ffn_injection_features", True),
        ffn_injection_evidence_top_k=cfg_dgst_t.get("ffn_injection_evidence_top_k", 32),
        ffn_injection_evidence_rank=cfg_dgst_t.get("ffn_injection_evidence_rank", 8),
        ffn_injection_eps=cfg_dgst_t.get("ffn_injection_eps", 1e-12),
        compute_dual_scope=cfg_dgst_t.get(
            "dgst_t_dual_scope",
            cfg_dgst_t.get("compute_dual_scope", False),
        ),
    )


def _extract_token_features_fallback(
    *,
    model_wrapper: BaseLVLMWrapper,
    image: Image.Image,
    image_id: int,
    spans: List[dict],
    response_token_ids: List[int],
    response_indices: List[int],
    target_token_ids: List[int],
    cfg_dgst_t: dict | None = None,
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
                    cfg_dgst_t=cfg_dgst_t,
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

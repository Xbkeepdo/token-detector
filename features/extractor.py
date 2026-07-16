"""Feature extraction for MS-COCO image captioning. Runs prefix forward passes and computes DGST-T per token."""

from __future__ import annotations
import os
from typing import Dict, List, Mapping, Optional

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from models.base_wrapper import (
    AttentionRequirement,
    BaseLVLMWrapper,
    ExtractionRequirements,
    GenerationOutput,
)
from features.attention import compute_alpha_img_alpha_text
from features.ads import compute_ads
from features.cgc import compute_cgc
from features.dgst_t import compute_dgst_t
from utils.io_utils import append_pkl, load_json, load_pkl, save_pkl


def extract_features_for_dataset(
    model_wrapper: BaseLVLMWrapper,
    coco_samples: List[dict],
    labeling_results: Dict[int, dict],
    cfg_dgst_t: dict,
    output_path: str,
    resume: bool = True,
    prompt: Optional[str] = None,
    cfg_feature_extraction: Optional[dict] = None,
    baseline_runtime=None,
    baseline_output_path: Optional[str] = None,
    baseline_official_output_path: Optional[str] = None,
) -> List[dict]:
    """Main entry point.  Iterates over `coco_samples`, extracts features"""
    all_features: List[dict] = []
    family_flags = _feature_family_flags(cfg_feature_extraction)
    active_dgst_t = (
        _resolve_active_dgst_config(cfg_dgst_t)
        if family_flags["method"]
        else None
    )
    done_image_ids = set()
    baseline_features: List[dict] = []
    baseline_done_image_ids: set[int] = set()
    official_svar_features: List[dict] = []
    official_svar_done_image_ids: set[int] = set()
    output_dir = os.path.dirname(output_path)
    generation_results = _load_generation_results(output_dir)
    if resume and os.path.exists(output_path):
        all_features = load_pkl(output_path)
        done_image_ids = {f["image_id"] for f in all_features}
        print(f"[Extractor] Resuming — {len(done_image_ids)} images already done.")
    runtime_methods = (
        getattr(baseline_runtime, "methods", None)
        if baseline_runtime is not None
        else ()
    )
    controlled_baseline_enabled = bool(
        baseline_runtime is not None
        and (runtime_methods is None or tuple(runtime_methods))
    )
    official_svar_enabled = bool(
        baseline_runtime is not None
        and getattr(baseline_runtime, "official_svar_enabled", False)
    )
    if controlled_baseline_enabled:
        if baseline_output_path is None:
            raise ValueError(
                "A controlled baseline runtime requires baseline_output_path."
            )
        if resume and os.path.exists(baseline_output_path):
            baseline_features = load_pkl(baseline_output_path)
            baseline_done_image_ids = {
                int(feature["image_id"])
                for feature in baseline_features
                if "image_id" in feature
            }
            print(
                "[Extractor] Baseline resume — "
                f"{len(baseline_done_image_ids)} images already done."
            )
    if official_svar_enabled:
        if baseline_official_output_path is None:
            raise ValueError(
                "Official SVAR extraction requires "
                "baseline_official_output_path."
            )
        if resume and os.path.exists(baseline_official_output_path):
            official_svar_features = load_pkl(baseline_official_output_path)
            official_svar_done_image_ids = {
                int(feature["image_id"])
                for feature in official_svar_features
                if "image_id" in feature
            }
            print(
                "[Extractor] Official SVAR resume — "
                f"{len(official_svar_done_image_ids)} images already done."
            )

    for sample in tqdm(coco_samples, desc="Extracting features"):
        image_id = int(sample["image_id"])
        family_needs = sample.get("_feature_families_needed")
        if family_needs is not None:
            if not isinstance(family_needs, Mapping) or set(family_needs) != {
                "root",
                "controlled",
                "official",
            }:
                raise ValueError(
                    f"Image {image_id}: invalid _feature_families_needed="
                    f"{family_needs!r}"
                )
            if bool(family_needs["controlled"]) and not controlled_baseline_enabled:
                raise RuntimeError(
                    f"Image {image_id}: controlled baseline output is marked "
                    "pending but no controlled baseline runtime is enabled."
                )
            if bool(family_needs["official"]) and not official_svar_enabled:
                raise RuntimeError(
                    f"Image {image_id}: official SVAR output is marked pending "
                    "but official SVAR is disabled."
                )
            root_is_done = not bool(family_needs["root"])
            controlled_baseline_is_done = (
                not controlled_baseline_enabled
                or not bool(family_needs["controlled"])
            )
            official_svar_is_done = (
                not official_svar_enabled
                or not bool(family_needs["official"])
            )
        else:
            root_is_done = image_id in done_image_ids
            controlled_baseline_is_done = (
                not controlled_baseline_enabled
                or image_id in baseline_done_image_ids
            )
            official_svar_is_done = (
                not official_svar_enabled
                or image_id in official_svar_done_image_ids
            )
        if (
            root_is_done
            and controlled_baseline_is_done
            and official_svar_is_done
        ):
            continue
        requirements = build_extraction_requirements(
            method=bool(family_flags["method"] and not root_is_done),
            ads_cgc=bool(family_flags["ads_cgc"] and not root_is_done),
            baseline=False,
        )
        image_dgst_t = (
            active_dgst_t
            if family_flags["method"] and not root_is_done
            else None
        )

        label_info = labeling_results.get(image_id)
        if label_info is None:
            continue

        labeling_generated_text = label_info.get("generated_text", "")
        if not labeling_generated_text:
            continue
        generation_entry = generation_results.get(image_id)
        if not isinstance(generation_entry, dict):
            raise RuntimeError(
                f"Image {image_id}: generations.json has no matching row. "
                "Schema-v2 extraction never reconstructs response IDs by "
                "tokenizing generated_text."
            )
        generation_text = str(generation_entry.get("generated_text", ""))
        if generation_text != str(labeling_generated_text):
            raise RuntimeError(
                f"Image {image_id}: generated_text differs between "
                "generations.json and labeling.json."
            )
        raw_response_ids = generation_entry.get("response_token_ids") or []
        if not raw_response_ids:
            raise RuntimeError(
                f"Image {image_id}: generations.json has no actual "
                "response_token_ids. Re-run or reuse the generation stage."
            )
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

        image_features = []
        controlled_spans = []
        for span in object_token_spans:
            if not span.get("token_indices"):
                continue
            _resolve_causal_target(
                response_token_ids=gen_out.response_token_ids,
                span=span,
                image_id=image_id,
            )
            controlled_spans.append(span)

        need_controlled_outputs = (
            not root_is_done or not controlled_baseline_is_done
        )
        official_spans = (
            baseline_runtime.prepare_official_svar_spans(
                label_info,
                gen_out.response_token_ids,
            )
            if official_svar_enabled and not official_svar_is_done
            else []
        )
        for span in official_spans:
            _resolve_causal_target(
                response_token_ids=gen_out.response_token_ids,
                span=span,
                image_id=image_id,
            )
        if official_svar_enabled and not official_spans:
            # Images without a found official first-token-ID sample are not an
            # official-SVAR extraction transaction. They must not force a
            # repeated model forward on every resume.
            official_svar_is_done = True
        if not controlled_spans and not official_spans:
            continue

        if baseline_runtime is not None:
            controlled_capture_needed = bool(
                controlled_baseline_enabled
                and not controlled_baseline_is_done
                and controlled_spans
            )
            official_capture_needed = bool(
                official_spans and not official_svar_is_done
            )
            requirement_selector = getattr(
                baseline_runtime, "requirements_for", None
            )
            if callable(requirement_selector):
                baseline_requirements = requirement_selector(
                    controlled=controlled_capture_needed,
                    official=official_capture_needed,
                )
            else:
                # Retain compatibility with lightweight injected runtimes used
                # by downstream callers and tests. The built-in runtime always
                # exposes the resume-aware selector above.
                baseline_requirements = baseline_runtime.requirements
            requirements = requirements.merged(baseline_requirements)

        requested_spans = [
            *(controlled_spans if need_controlled_outputs else []),
            *official_spans,
        ]
        response_indices = list(
            dict.fromkeys(
                _resolve_causal_target(
                    response_token_ids=gen_out.response_token_ids,
                    span=span,
                    image_id=image_id,
                )[0]
                for span in requested_spans
            )
        )
        if not response_indices:
            continue
        target_token_ids = [
            int(gen_out.response_token_ids[index]) for index in response_indices
        ]
        representative_spans = []
        for index in response_indices:
            representative_spans.append(
                next(
                    span
                    for span in requested_spans
                    if int(span["token_indices"][0]) == index
                )
            )

        try:
            model_outputs = model_wrapper.extract_token_features_batch(
                image=image,
                response_token_ids=gen_out.response_token_ids,
                response_token_indices=response_indices,
                target_token_ids=target_token_ids,
                cfg_dgst_t=image_dgst_t,
                prompt=prompt,
                requirements=requirements,
            )
        except Exception as e:
            import traceback
            print(f"[Extractor] Warning — batch forward failed for image {image_id}: {e}")
            traceback.print_exc()
            if baseline_runtime is not None:
                raise RuntimeError(
                    "Joint baseline extraction requires one caption-level batch "
                    "forward for MetaToken/HalLoc alignment; per-object fallback "
                    f"is unsafe for image {image_id}."
                ) from e
            model_outputs = _extract_token_features_fallback(
                model_wrapper=model_wrapper,
                image=image,
                image_id=image_id,
                spans=representative_spans,
                response_token_ids=gen_out.response_token_ids,
                response_indices=response_indices,
                target_token_ids=target_token_ids,
                cfg_dgst_t=image_dgst_t,
                prompt=prompt,
                requirements=requirements,
            )

        if len(model_outputs) != len(response_indices):
            raise RuntimeError(
                f"Image {image_id} returned {len(model_outputs)} outputs for "
                f"{len(response_indices)} causal token positions. Refusing to mark a partial "
                "image complete because image-level resume would skip its "
                "missing object spans."
            )

        successful_positions = [
            (first_idx, target_token_id, model_out)
            for first_idx, target_token_id, model_out in zip(
                response_indices,
                target_token_ids,
                model_outputs,
            )
            if model_out is not None
        ]
        if len(successful_positions) != len(response_indices):
            raise RuntimeError(
                f"Image {image_id} produced {len(successful_positions)} successful outputs "
                f"for {len(response_indices)} causal positions. Refusing to persist a "
                "partial image because image-level resume would skip it."
            )
        outputs_by_index = {
            int(first_idx): model_out
            for first_idx, target_token_id, model_out in successful_positions
        }
        for first_idx, _target_token_id, model_out in successful_positions:
            returned_index = getattr(model_out, "response_token_idx", None)
            if returned_index is not None and int(returned_index) != int(first_idx):
                raise AssertionError(
                    f"Image {image_id}: wrapper returned response_token_idx="
                    f"{returned_index} for requested causal position {first_idx}"
                )
        controlled_successful = [
            (
                span,
                int(span["token_indices"][0]),
                int(gen_out.response_token_ids[int(span["token_indices"][0])]),
                outputs_by_index[int(span["token_indices"][0])],
            )
            for span in controlled_spans
            if int(span["token_indices"][0]) in outputs_by_index
        ]
        if not root_is_done:
            for span, first_idx, target_token_id, model_out in controlled_successful:
                feat = _build_enabled_feature_record(
                    image_id=image_id,
                    span=span,
                    response_index=int(first_idx),
                    target_token_id=int(target_token_id),
                    model_out=model_out,
                    cfg_dgst_t=active_dgst_t,
                    cfg_feature_extraction=cfg_feature_extraction,
                    family_flags=family_flags,
                )
                image_features.append(feat)

        if (
            controlled_baseline_enabled
            and controlled_successful
            and not controlled_baseline_is_done
        ):
            image_baselines = baseline_runtime.build_image_records(
                image=image,
                image_id=int(image_id),
                response_token_ids=gen_out.response_token_ids,
                spans=[item[0] for item in controlled_successful],
                model_outputs=[item[3] for item in controlled_successful],
            )
            baseline_features.extend(image_baselines)
            if image_baselines:
                append_pkl(image_baselines, baseline_output_path)

        if official_spans and not official_svar_is_done:
            official_outputs = [
                outputs_by_index[int(span["token_indices"][0])]
                for span in official_spans
            ]
            image_official_svar = baseline_runtime.build_official_svar_records(
                image_id=int(image_id),
                response_token_ids=gen_out.response_token_ids,
                spans=official_spans,
                model_outputs=official_outputs,
            )
            official_svar_features.extend(image_official_svar)
            if image_official_svar:
                append_pkl(
                    image_official_svar,
                    baseline_official_output_path,
                )

        if not root_is_done:
            all_features.extend(image_features)
        if image_features and not root_is_done:
            append_pkl(image_features, output_path)

    if controlled_baseline_enabled and not os.path.exists(baseline_output_path):
        save_pkl([], baseline_output_path)
    if official_svar_enabled and not os.path.exists(baseline_official_output_path):
        save_pkl([], baseline_official_output_path)
    print(f"[Extractor] Done. {len(all_features)} DGST-T object tokens saved to {output_path}.")
    if baseline_runtime is not None:
        if controlled_baseline_enabled:
            print(
                f"[Extractor] Done. {len(baseline_features)} baseline object tokens "
                f"saved to {baseline_output_path}."
            )
        if official_svar_enabled:
            print(
                f"[Extractor] Done. {len(official_svar_features)} official SVAR "
                f"object tokens saved to {baseline_official_output_path}."
            )
    return all_features


def build_extraction_requirements(
    *,
    method: bool,
    ads_cgc: bool,
    baseline: bool,
) -> ExtractionRequirements:
    """Return the least set of tensors required by all enabled consumers."""
    requirements = ExtractionRequirements(
        attention=AttentionRequirement.NONE,
        logits=False,
        token_hidden_states=False,
        patch_hidden_states=False,
        response_hidden_states=False,
        visual_layout=False,
        dgst_capture=bool(method),
    )
    if ads_cgc:
        requirements = requirements.merged(
            ExtractionRequirements(
                attention=AttentionRequirement.PER_HEAD,
                logits=False,
                token_hidden_states=True,
                patch_hidden_states=True,
                response_hidden_states=False,
                visual_layout=True,
                dgst_capture=False,
            )
        )
    if baseline:
        requirements = requirements.merged(
            ExtractionRequirements(
                attention=AttentionRequirement.PER_HEAD,
                # MetaToken is compacted during its caption pass; no baseline
                # keeps per-object vocabulary rows.  ProjectAway needs raw
                # visual layer outputs, while HalLoc needs only final response
                # embeddings.
                logits=False,
                token_hidden_states=False,
                patch_hidden_states=True,
                response_hidden_states=True,
                visual_layout=True,
                dgst_capture=False,
            )
        )
    return requirements


def _feature_family_flags(cfg_feature_extraction: Optional[dict]) -> dict[str, bool]:
    # Historical call sites passed only cfg_dgst_t and therefore mean method-only.
    if cfg_feature_extraction is None:
        return {"method": True, "ads_cgc": False, "baseline": False}

    def enabled(name: str, default: bool = False) -> bool:
        value = cfg_feature_extraction.get(name, {})
        if isinstance(value, dict):
            return bool(value.get("enabled", default))
        return bool(value)

    flags = {
        "method": enabled("method"),
        "ads_cgc": enabled("ads_cgc"),
        "baseline": enabled("baseline"),
    }
    if not any(flags.values()):
        raise ValueError("No feature family is enabled for extraction.")
    return flags


def _resolve_active_dgst_config(cfg_dgst_t: dict) -> dict:
    resolved = dict(cfg_dgst_t or {})
    if not bool(resolved.get("enabled", True)):
        raise ValueError("feature_extraction.method is enabled but dgst_t.enabled=false.")
    branches = resolved.get("branches")
    if isinstance(branches, dict):
        configured = resolved.get("four_gate_methods") or list(branches)
        enabled_methods = [
            str(method)
            for method in configured
            if bool(branches.get(str(method), True))
        ]
        if not enabled_methods:
            raise ValueError("All four DGST branch switches are disabled.")
        resolved["four_gate_methods"] = enabled_methods
    return resolved


def _build_enabled_feature_record(
    *,
    image_id: int,
    span: dict,
    response_index: int,
    target_token_id: int,
    model_out,
    cfg_dgst_t: Optional[dict],
    cfg_feature_extraction: Optional[dict],
    family_flags: dict[str, bool],
) -> dict:
    if family_flags["method"]:
        if cfg_dgst_t is None:
            raise RuntimeError("Method extraction requires a DGST configuration.")
        feat = _build_feature_record(
            image_id=image_id,
            span=span,
            response_index=response_index,
            target_token_id=target_token_id,
            model_out=model_out,
            cfg_dgst_t=cfg_dgst_t,
        )
    else:
        feat = _base_token_record(
            image_id=image_id,
            span=span,
            response_index=response_index,
            target_token_id=target_token_id,
            model_out=model_out,
        )
    if family_flags["ads_cgc"]:
        feat.update(
            _compute_ads_cgc_features(
                model_out,
                cfg_feature_extraction or {},
            )
        )
    return feat


def _base_token_record(
    *,
    image_id: int,
    span: dict,
    response_index: int,
    target_token_id: int,
    model_out,
) -> dict:
    return {
        "feature_schema_version": "token-detector-v2",
        "image_id": int(image_id),
        "token_str": span["word"],
        "token_id": int(model_out.token_id),
        "target_token_id": int(target_token_id),
        "response_token_idx": int(response_index),
        "label": int(span["label"]),
    }


def _compute_ads_cgc_features(model_out, cfg_feature_extraction: dict) -> dict:
    if model_out.text_to_patch_attn.numel() == 0:
        raise RuntimeError("ADS/CGC requires per-head text-to-patch attention.")
    if model_out.token_hidden_states.numel() == 0 or model_out.patch_hidden_states.numel() == 0:
        raise RuntimeError("CGC requires token and visual-patch hidden states.")
    ads_cfg = dict(cfg_feature_extraction.get("ads") or {})
    cgc_cfg = dict(cfg_feature_extraction.get("cgc") or {})
    ads_score, ads_per_layer = compute_ads(
        model_out.text_to_patch_attn,
        top_patch_pct=float(ads_cfg.get("top_patch_pct", 0.10)),
        connectivity=int(ads_cfg.get("connectivity", 8)),
        min_blob_area=int(ads_cfg.get("min_blob_area", 3)),
        top_k_layers=int(ads_cfg.get("top_k_layers", 10)),
        grid_shape=model_out.visual_grid,
    )
    cgc_score, cgc_per_layer = compute_cgc(
        model_out.token_hidden_states,
        model_out.patch_hidden_states,
        top_k_patches=int(cgc_cfg.get("top_k_patches", 5)),
        top_k_pct=float(cgc_cfg.get("top_k_pct", 0.05)),
        text_to_patch_attn=(
            model_out.text_to_patch_attn
            if bool(cgc_cfg.get("use_attn_weighting", False))
            else None
        ),
        mid_layer_pct=tuple(cgc_cfg.get("mid_layer_pct", (0.25, 0.75))),
    )
    return {
        "ads_score": float(ads_score),
        "ads_per_layer": _compact_numpy(ads_per_layer, dtype=np.float32),
        "cgc_score": float(cgc_score),
        "cgc_per_layer": _compact_numpy(cgc_per_layer, dtype=np.float32),
    }


def _load_generation_results(output_dir: str) -> dict[int, dict]:
    path = os.path.join(output_dir, "generations.json")
    if not os.path.exists(path):
        return {}
    raw = load_json(path)
    return {int(key): value for key, value in raw.items()}


def _resolve_causal_target(
    *,
    response_token_ids: List[int],
    span: dict,
    image_id: Optional[int] = None,
) -> tuple[int, int, List[int]]:
    """Resolve the exact response token and the prefix that predicts it.

    ``response_index=i`` means the target is ``response_token_ids[i]`` and the
    causal decoder input contains only ``response_token_ids[:i]``.  The helper
    intentionally rejects corrupt schema-v2 locations rather than shifting to
    a neighbouring token.
    """

    raw_indices = span.get("token_indices") or []
    if not raw_indices:
        raise ValueError(
            f"Image {image_id}: object span has no exact token_indices"
        )
    try:
        indices = [int(value) for value in raw_indices]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Image {image_id}: non-integer token_indices={raw_indices}"
        ) from exc
    if any(index < 0 for index in indices):
        raise ValueError(
            f"Image {image_id}: negative token_indices={indices}"
        )
    response_length = len(response_token_ids)
    if any(index >= response_length for index in indices):
        raise ValueError(
            f"Image {image_id}: token_indices={indices} are outside response "
            f"length {response_length}"
        )
    first_index = indices[0]
    target_token_id = int(response_token_ids[first_index])
    prefix_token_ids = [
        int(value) for value in response_token_ids[:first_index]
    ]
    return first_index, target_token_id, prefix_token_ids


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
    if cfg_dgst_t.get("feature_output_profile") == "costvariant_vv":
        return _build_cost_variant_feature_record(
            image_id=image_id,
            span=span,
            response_index=response_index,
            target_token_id=target_token_id,
            model_out=model_out,
            dgst_t=dgst_t,
        )
    if cfg_dgst_t.get("feature_output_profile") == "gate_comparison_vv":
        return _build_gate_comparison_feature_record(
            image_id=image_id,
            span=span,
            response_index=response_index,
            target_token_id=target_token_id,
            model_out=model_out,
            dgst_t=dgst_t,
        )
    if cfg_dgst_t.get("feature_output_profile") == "four_gate_vv":
        return _build_four_gate_feature_record(
            image_id=image_id,
            span=span,
            response_index=response_index,
            target_token_id=target_token_id,
            model_out=model_out,
            dgst_t=dgst_t,
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


def _build_cost_variant_feature_record(
    *,
    image_id: int,
    span: dict,
    response_index: int,
    target_token_id: int,
    model_out,
    dgst_t: dict,
) -> dict:
    risk_names = (
        "risk_geo",
        "risk_cosine_hpre",
        "risk_sqrt_hmid",
        "risk_sqrt_hpre",
        "risk_raw_attention_hmid",
        "risk_raw_attention_hpre",
        "gauss_risk_geo",
        "gauss_risk_cosine_hpre",
        "gauss_risk_sqrt_hmid",
        "gauss_risk_sqrt_hpre",
    )
    required = [
        *(f"dgst_t_{name}_per_layer" for name in risk_names),
        "dgst_t_target_visual_hpre_cosine_relative_vll_per_layer",
        "dgst_t_vv_support_attention_per_layer",
        "dgst_t_vv_source_dist_per_layer",
        "dgst_t_vv_semantic_gate_per_layer",
        "dgst_t_vv_gauss_semantic_gate_per_layer",
        "dgst_t_vv_support_positions",
        "dgst_t_cost_variant_mad_scale",
        "dgst_t_cost_variant_transport_top_k",
        "dgst_t_cost_variant_hprecosine_top_k",
    ]
    missing = [key for key in required if key not in dgst_t]
    if missing:
        raise KeyError(f"Missing cost-variant DGST-T fields: {missing}")

    feat = {
        "image_id": int(image_id),
        "token_str": span["word"],
        "token_id": int(model_out.token_id),
        "target_token_id": int(target_token_id),
        "response_token_idx": int(response_index),
        "label": int(span["label"]),
        "dgst_t_relative_vll_logit_source": str(
            dgst_t.get("dgst_t_relative_vll_logit_source", "h_mid")
        ),
        "dgst_t_source_distribution_mode": str(
            dgst_t.get("dgst_t_source_distribution_mode", "softmax")
        ),
    }
    for name in risk_names:
        key = f"dgst_t_{name}_per_layer"
        feat[key] = dgst_t[key].detach().cpu().tolist()
    hpre_key = "dgst_t_target_visual_hpre_cosine_relative_vll_per_layer"
    feat[hpre_key] = dgst_t[hpre_key].detach().cpu().tolist()
    for key in (
        "dgst_t_vv_support_attention_per_layer",
        "dgst_t_vv_source_dist_per_layer",
        "dgst_t_vv_semantic_gate_per_layer",
        "dgst_t_vv_gauss_semantic_gate_per_layer",
    ):
        feat[key] = dgst_t[key].detach().cpu()
    feat["dgst_t_vv_support_positions"] = [
        int(position) for position in dgst_t["dgst_t_vv_support_positions"]
    ]
    feat["dgst_t_cost_variant_mad_scale"] = float(
        dgst_t["dgst_t_cost_variant_mad_scale"]
    )
    feat["dgst_t_cost_variant_transport_top_k"] = int(
        dgst_t["dgst_t_cost_variant_transport_top_k"]
    )
    feat["dgst_t_cost_variant_hprecosine_top_k"] = int(
        dgst_t["dgst_t_cost_variant_hprecosine_top_k"]
    )
    return feat


def _build_gate_comparison_feature_record(
    *,
    image_id: int,
    span: dict,
    response_index: int,
    target_token_id: int,
    model_out,
    dgst_t: dict,
) -> dict:
    risk_keys = (
        "dgst_t_relative_vll_gauss_risk_sqrt_hpre_per_layer",
        "dgst_t_softmax_relative_vll_gauss_risk_sqrt_hpre_per_layer",
        "dgst_t_legacy_prob_risk_sqrt_hpre_per_layer",
    )
    cosine_keys = (
        "dgst_t_relative_vll_gauss_target_visual_hpre_cosine_per_layer",
        "dgst_t_softmax_relative_vll_gauss_target_visual_hpre_cosine_per_layer",
        "dgst_t_legacy_prob_target_visual_hpre_cosine_per_layer",
    )
    metadata_keys = (
        "dgst_t_gate_comparison_methods",
        "dgst_t_gate_comparison_cost",
        "dgst_t_gate_comparison_target_cosine_state",
        "dgst_t_gate_comparison_softmax_axis",
        "dgst_t_gate_comparison_mad_axis",
        "dgst_t_gate_comparison_mad_scale",
        "dgst_t_gate_comparison_transport_top_k",
        "dgst_t_gate_comparison_target_cosine_top_k",
    )
    required = (*risk_keys, *cosine_keys, *metadata_keys)
    missing = [key for key in required if key not in dgst_t]
    if missing:
        raise KeyError(f"Missing gate-comparison DGST-T fields: {missing}")

    feat = {
        "image_id": int(image_id),
        "token_str": span["word"],
        "token_id": int(model_out.token_id),
        "target_token_id": int(target_token_id),
        "response_token_idx": int(response_index),
        "label": int(span["label"]),
        "dgst_t_relative_vll_logit_source": str(
            dgst_t.get("dgst_t_relative_vll_logit_source", "h_mid")
        ),
        "dgst_t_source_distribution_mode": str(
            dgst_t.get("dgst_t_source_distribution_mode", "softmax")
        ),
    }
    for key in (*risk_keys, *cosine_keys):
        value = dgst_t[key]
        feat[key] = value.detach().cpu().tolist() if hasattr(value, "detach") else list(value)
    for key in metadata_keys:
        feat[key] = dgst_t[key]
    return feat


def _build_four_gate_feature_record(
    *,
    image_id: int,
    span: dict,
    response_index: int,
    target_token_id: int,
    model_out,
    dgst_t: dict,
) -> dict:
    """Build the compact, explicitly named four-gate feature record."""
    methods = tuple(dgst_t.get("dgst_t_four_gate_methods") or ())
    if not methods:
        raise KeyError("Missing dgst_t_four_gate_methods in four-gate result.")
    required_metadata = (
        "dgst_t_profile",
        "dgst_t_mad_axis",
        "dgst_t_mad_scale",
        "dgst_t_softmax_axis",
        "dgst_t_source_distribution_mode",
        "dgst_t_transport_top_k",
        "dgst_t_target_region_top_k",
        "dgst_t_cost",
        "dgst_t_ot_solver",
    )
    required_matrices = (
        "dgst_t_attention_support_per_layer",
        "dgst_t_source_dist_per_layer",
    )
    method_keys = []
    for method in methods:
        method_keys.extend(
            [
                f"dgst_t_{method}_gate_per_layer",
                f"dgst_t_{method}_risk_sqrt_hpre_per_layer",
                f"dgst_t_{method}_target_cosine_topk32_hpre_per_layer",
                f"dgst_t_{method}_ev_topk32_hpre_per_layer",
            ]
        )
    missing = [
        key
        for key in (*required_metadata, *required_matrices, *method_keys)
        if key not in dgst_t
    ]
    if missing:
        raise KeyError(f"Missing four-gate DGST fields: {missing}")

    has_raw_attention = "raw_attention" in methods
    feat = {
        "feature_schema_version": (
            "dgst-target-comparison-v2"
            if has_raw_attention
            else "dgst-four-gate-v1"
        ),
        "image_id": int(image_id),
        "token_str": span["word"],
        "token_id": int(model_out.token_id),
        "target_token_id": int(target_token_id),
        "response_token_idx": int(response_index),
        "label": int(span["label"]),
        "dgst_t_four_gate_methods": list(methods),
    }
    for key in required_metadata:
        feat[key] = dgst_t[key]
    if has_raw_attention:
        feat["dgst_t_raw_attention_definition"] = dgst_t.get(
            "dgst_t_raw_attention_definition",
            "post_softmax_head_mean_visual_support_renormalized",
        )
    for key in required_matrices:
        feat[key] = _compact_numpy(dgst_t[key], dtype=np.float16)
    for method in methods:
        gate_key = f"dgst_t_{method}_gate_per_layer"
        feat[gate_key] = _compact_numpy(dgst_t[gate_key], dtype=np.float16)
        for suffix in (
            "risk_sqrt_hpre_per_layer",
            "target_cosine_topk32_hpre_per_layer",
            "ev_topk32_hpre_per_layer",
        ):
            key = f"dgst_t_{method}_{suffix}"
            feat[key] = _compact_numpy(dgst_t[key], dtype=np.float32)
    return feat


def _compact_numpy(value, *, dtype):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        # NumPy cannot consume torch.bfloat16 directly. DGST outputs normally
        # already use fp16/fp32, but keep serialization safe for wrapper-native
        # bf16 tensors as well.
        if value.dtype == torch.bfloat16:
            value = value.float()
        value = value.numpy()
    return np.asarray(value, dtype=dtype)


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
    prompt: Optional[str] = None,
    requirements: Optional[ExtractionRequirements] = None,
):
    outputs = []
    for span, first_idx, target_token_id in zip(spans, response_indices, target_token_ids):
        try:
            resolved_index, resolved_target, prefix_token_ids = _resolve_causal_target(
                response_token_ids=response_token_ids,
                span=span,
                image_id=image_id,
            )
            if resolved_index != int(first_idx) or resolved_target != int(
                target_token_id
            ):
                raise AssertionError(
                    "Fallback causal target differs from the batch request: "
                    f"resolved=({resolved_index}, {resolved_target}), "
                    f"requested=({first_idx}, {target_token_id})"
                )
            outputs.append(
                model_wrapper.extract_token_features(
                    image=image,
                    prefix_token_ids=prefix_token_ids,
                    response_token_idx=int(first_idx),
                    target_token_id=int(target_token_id),
                    cfg_dgst_t=cfg_dgst_t,
                    prompt=prompt,
                    requirements=requirements,
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

#!/usr/bin/env python3
"""Train classifiers on selected DGST-T feature blocks."""

from __future__ import annotations

import argparse
import math
import os
import sys
from copy import deepcopy
from typing import Sequence

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.training_provenance import load_validated_training_features
from utils.io_utils import load_json, save_json

from summarize_feature_set_results import write_summary_tables


FEATURE_ALIASES = {
    "risk": "risk",
    "transport_risk": "risk",
    "risk_topmass_085": "risk_topmass_085",
    "topmass_085": "risk_topmass_085",
    "risk_capped_topmass_085": "risk_capped_topmass_085",
    "capped_topmass_085": "risk_capped_topmass_085",
    "risk_relative_vll": "risk_relative_vll",
    "relative_vll_risk": "risk_relative_vll",
    "risk_relative_vll_capped_topmass_085": "risk_relative_vll_capped_topmass_085",
    "relative_vll_capped_topmass_085": "risk_relative_vll_capped_topmass_085",
    "risk_visual_prompt_relative_vll": "risk_visual_prompt_relative_vll",
    "visual_prompt_relative_vll_risk": "risk_visual_prompt_relative_vll",
    "risk_visual_prompt_relative_vll_capped_topmass_085": "risk_visual_prompt_relative_vll_capped_topmass_085",
    "visual_prompt_relative_vll_capped_topmass_085": "risk_visual_prompt_relative_vll_capped_topmass_085",
    "js_relative_vll": "js_relative_vll",
    "js_vv": "js_relative_vll",
    "vv_js": "js_relative_vll",
    "kl_target_source_relative_vll": "kl_target_source_relative_vll",
    "kl_target_source_vv": "kl_target_source_relative_vll",
    "vv_kl_target_source": "kl_target_source_relative_vll",
    "kl_t_s_relative_vll": "kl_target_source_relative_vll",
    "kl_source_target_relative_vll": "kl_source_target_relative_vll",
    "kl_source_target_vv": "kl_source_target_relative_vll",
    "vv_kl_source_target": "kl_source_target_relative_vll",
    "kl_s_t_relative_vll": "kl_source_target_relative_vll",
    "js_visual_prompt_relative_vll": "js_visual_prompt_relative_vll",
    "js_vp": "js_visual_prompt_relative_vll",
    "vp_js": "js_visual_prompt_relative_vll",
    "kl_target_source_visual_prompt_relative_vll": "kl_target_source_visual_prompt_relative_vll",
    "kl_target_source_vp": "kl_target_source_visual_prompt_relative_vll",
    "vp_kl_target_source": "kl_target_source_visual_prompt_relative_vll",
    "kl_t_s_visual_prompt_relative_vll": "kl_target_source_visual_prompt_relative_vll",
    "kl_source_target_visual_prompt_relative_vll": "kl_source_target_visual_prompt_relative_vll",
    "kl_source_target_vp": "kl_source_target_visual_prompt_relative_vll",
    "vp_kl_source_target": "kl_source_target_visual_prompt_relative_vll",
    "kl_s_t_visual_prompt_relative_vll": "kl_source_target_visual_prompt_relative_vll",
    "vv_attention_tk32_js": "vv_attention_tk32_js",
    "vv_atk32_js": "vv_attention_tk32_js",
    "vv_attention_tk32_kl_attention_source": "vv_attention_tk32_kl_attention_source",
    "vv_atk32_kl_as": "vv_attention_tk32_kl_attention_source",
    "vv_attention_tk32_kl_source_attention": "vv_attention_tk32_kl_source_attention",
    "vv_atk32_kl_sa": "vv_attention_tk32_kl_source_attention",
    "vp_attention_tk32_js": "vp_attention_tk32_js",
    "vp_atk32_js": "vp_attention_tk32_js",
    "vp_attention_tk32_kl_attention_source": "vp_attention_tk32_kl_attention_source",
    "vp_atk32_kl_as": "vp_attention_tk32_kl_attention_source",
    "vp_attention_tk32_kl_source_attention": "vp_attention_tk32_kl_source_attention",
    "vp_atk32_kl_sa": "vp_attention_tk32_kl_source_attention",
    "risk_geo_raw": "risk_relative_vll_cost_geo",
    "risk_geo_cap085": "risk_relative_vll_cost_geo_capped_topmass_085",
    "risk_geo_capped_topmass_085": "risk_relative_vll_cost_geo_capped_topmass_085",
    "risk_geo": "risk_geo",
    "risk_cosine_hpre": "risk_cosine_hpre",
    "risk_sqrt_hmid": "risk_sqrt_hmid",
    "risk_sqrt_hpre": "risk_sqrt_hpre",
    "risk_rawattention_hmid": "risk_raw_attention_hmid",
    "risk_raw_attention_hmid": "risk_raw_attention_hmid",
    "risk_rawattention_hpre": "risk_raw_attention_hpre",
    "risk_raw_attention_hpre": "risk_raw_attention_hpre",
    "gauss_risk_geo": "gauss_risk_geo",
    "gauss_risk_cosine_hpre": "gauss_risk_cosine_hpre",
    "gauss_risk_sqrt_hmid": "gauss_risk_sqrt_hmid",
    "gauss_risk_sqrt_hpre": "gauss_risk_sqrt_hpre",
    "risk_hmid_proj": "risk_relative_vll_source_hmid_proj",
    "hmid_proj": "risk_relative_vll_source_hmid_proj",
    "risk_hprev_cos": "risk_relative_vll_source_hprev_cos",
    "hprev_cos": "risk_relative_vll_source_hprev_cos",
    "risk_hprev_proj": "risk_relative_vll_source_hprev_proj",
    "hprev_proj": "risk_relative_vll_source_hprev_proj",
    "vp_risk_hmid_proj": "risk_visual_prompt_relative_vll_source_hmid_proj",
    "vp_hmid_proj": "risk_visual_prompt_relative_vll_source_hmid_proj",
    "vp_risk_hprev_cos": "risk_visual_prompt_relative_vll_source_hprev_cos",
    "vp_hprev_cos": "risk_visual_prompt_relative_vll_source_hprev_cos",
    "vp_risk_hprev_proj": "risk_visual_prompt_relative_vll_source_hprev_proj",
    "vp_hprev_proj": "risk_visual_prompt_relative_vll_source_hprev_proj",
    "c_vp": "c_vp",
    "cvp": "c_vp",
    "m_p": "m_p",
    "mp": "m_p",
    "r_es": "r_es",
    "res": "r_es",
    "es": "relative_vll_evidence_strength",
    "evidence_strength": "relative_vll_evidence_strength",
    "vv_raw_es": "vv_raw_evidence_strength",
    "vv_support_es": "vv_raw_evidence_strength",
    "vv_support_attention_x_semantic_gate": "vv_raw_evidence_strength",
    "vv_support_attention*semantic_gate": "vv_raw_evidence_strength",
    "vv_support_attention_times_semantic_gate": "vv_raw_evidence_strength",
    "vp_es": "visual_prompt_relative_vll_evidence_strength",
    "vp_evidence_strength": "visual_prompt_relative_vll_evidence_strength",
    "vp_raw_es": "vp_raw_evidence_strength",
    "vp_support_es": "vp_raw_evidence_strength",
    "vp_support_attention_x_semantic_gate": "vp_raw_evidence_strength",
    "vp_support_attention*semantic_gate": "vp_raw_evidence_strength",
    "vp_support_attention_times_semantic_gate": "vp_raw_evidence_strength",
    "vp_raw_ev": "vp_raw_evidence_visual_mass",
    "vp_raw_evidence_visual_mass": "vp_raw_evidence_visual_mass",
    "vp_raw_ep": "vp_raw_evidence_prompt_mass",
    "vp_raw_evidence_prompt_mass": "vp_raw_evidence_prompt_mass",
    "t_v": "visual_prompt_relative_vll_evidence_visual_mass",
    "tv": "visual_prompt_relative_vll_evidence_visual_mass",
    "evidence_visual_mass": "visual_prompt_relative_vll_evidence_visual_mass",
    "t_p": "visual_prompt_relative_vll_evidence_prompt_mass",
    "tp": "visual_prompt_relative_vll_evidence_prompt_mass",
    "evidence_prompt_mass": "visual_prompt_relative_vll_evidence_prompt_mass",
    "b_v": "visual_prompt_relative_vll_source_visual_mass",
    "bv": "visual_prompt_relative_vll_source_visual_mass",
    "source_visual_mass": "visual_prompt_relative_vll_source_visual_mass",
    "b_p": "visual_prompt_relative_vll_source_prompt_mass",
    "bp": "visual_prompt_relative_vll_source_prompt_mass",
    "source_prompt_mass": "visual_prompt_relative_vll_source_prompt_mass",
    "vv_source_entropy": "vv_source_entropy",
    "vv_h_source": "vv_source_entropy",
    "vv_target_entropy": "vv_target_entropy",
    "vv_h_target": "vv_target_entropy",
    "vv_evidence_entropy": "vv_evidence_entropy",
    "vv_h_evidence": "vv_evidence_entropy",
    "vv_source_topk_entropy": "vv_source_topk_entropy",
    "vv_h_source_topk": "vv_source_topk_entropy",
    "vp_source_entropy": "vp_source_entropy",
    "vp_h_source": "vp_source_entropy",
    "vp_target_entropy": "vp_target_entropy",
    "vp_h_target": "vp_target_entropy",
    "vp_evidence_entropy": "vp_evidence_entropy",
    "vp_h_evidence": "vp_evidence_entropy",
    "vp_source_topk_entropy": "vp_source_topk_entropy",
    "vp_h_source_topk": "vp_source_topk_entropy",
    "prompt_confidence": "prompt_confidence_top3",
    "prompt_confidence_top3": "prompt_confidence_top3",
    "prompt_confidence_max": "prompt_confidence_max",
    "context_confidence": "context_confidence",
    "contextconfidence": "context_confidence",
    "context_confidence_max_prompt": "context_confidence_max_prompt",
    "visual_cosine": "target_visual_hidden_cosine",
    "target_visual_hidden_cosine": "target_visual_hidden_cosine",
    "visual_prompt_cosine": "target_visual_prompt_hidden_cosine",
    "target_visual_prompt_hidden_cosine": "target_visual_prompt_hidden_cosine",
    "visual_cosine_capped_topmass_085": "target_visual_hidden_cosine_capped_topmass_085",
    "target_visual_hidden_cosine_capped_topmass_085": "target_visual_hidden_cosine_capped_topmass_085",
    "visual_cosine_relative_vll": "target_visual_hidden_cosine_relative_vll",
    "target_visual_hidden_cosine_relative_vll": "target_visual_hidden_cosine_relative_vll",
    "visualcosine_raw": "target_visual_hidden_cosine_relative_vll",
    "cosine_raw": "target_visual_hidden_cosine_relative_vll",
    "hprecosine": "target_visual_hpre_cosine_relative_vll",
    "hpre_cosine": "target_visual_hpre_cosine_relative_vll",
    "hprecosine_raw": "target_visual_hpre_cosine_relative_vll",
    "visual_hpre_cosine": "target_visual_hpre_cosine_relative_vll",
    "target_visual_hpre_cosine": "target_visual_hpre_cosine_relative_vll",
    "target_visual_hpre_cosine_relative_vll": "target_visual_hpre_cosine_relative_vll",
    "hprecosine16": "target_visual_hpre_cosine16_relative_vll",
    "hpre_cosine16": "target_visual_hpre_cosine16_relative_vll",
    "visual_hpre_cosine16": "target_visual_hpre_cosine16_relative_vll",
    "target_visual_hpre_cosine16": "target_visual_hpre_cosine16_relative_vll",
    "target_visual_hpre_cosine16_relative_vll": "target_visual_hpre_cosine16_relative_vll",
    "hprecosine_cap085": "target_visual_hpre_cosine_relative_vll_capped_topmass_085",
    "hpre_cosine_cap085": "target_visual_hpre_cosine_relative_vll_capped_topmass_085",
    "target_visual_hpre_cosine_relative_vll_capped_topmass_085": (
        "target_visual_hpre_cosine_relative_vll_capped_topmass_085"
    ),
    "pingyi_cosine": "target_visual_hidden_cosine_relative_vll_shift1",
    "shifted_cosine": "target_visual_hidden_cosine_relative_vll_shift1",
    "visualcosine_pingyi": "target_visual_hidden_cosine_relative_vll_shift1",
    "visualcosine_shift1": "target_visual_hidden_cosine_relative_vll_shift1",
    "target_cosine_pingyi": "target_visual_hidden_cosine_relative_vll_shift1",
    "target_cosine_shift1": "target_visual_hidden_cosine_relative_vll_shift1",
    "target_cosine": "target_cosine",
    "targetcosine": "target_cosine",
    "cosine16": "cosine16",
    "target_cosine16": "cosine16",
    "visualcosine16_raw": "cosine16",
    "vp_target_cosine": "vp_target_cosine",
    "visual_prompt_target_cosine": "vp_target_cosine",
    "vp_hprecosine": "vp_hpre_cosine",
    "vp_hpre_cosine": "vp_hpre_cosine",
    "visual_prompt_hpre_cosine": "vp_hpre_cosine",
    "target_visual_prompt_hpre_cosine": "vp_hpre_cosine",
    "target_visual_prompt_hpre_cosine_visual_prompt_relative_vll": "vp_hpre_cosine",
    "vp_hprecosine16": "vp_hpre_cosine16",
    "vp_hpre_cosine16": "vp_hpre_cosine16",
    "visual_prompt_hpre_cosine16": "vp_hpre_cosine16",
    "target_visual_prompt_hpre_cosine16": "vp_hpre_cosine16",
    "target_visual_prompt_hpre_cosine16_visual_prompt_relative_vll": "vp_hpre_cosine16",
    "vp_hprecosine_cap085": "target_visual_prompt_hpre_cosine_visual_prompt_relative_vll_capped_topmass_085",
    "vp_hpre_cosine_cap085": "target_visual_prompt_hpre_cosine_visual_prompt_relative_vll_capped_topmass_085",
    "target_visual_prompt_hpre_cosine_visual_prompt_relative_vll_capped_topmass_085": (
        "target_visual_prompt_hpre_cosine_visual_prompt_relative_vll_capped_topmass_085"
    ),
    "vp_pingyi_cosine": "vp_target_cosine_shift1",
    "vp_shifted_cosine": "vp_target_cosine_shift1",
    "vp_target_cosine_pingyi": "vp_target_cosine_shift1",
    "vp_target_cosine_shift1": "vp_target_cosine_shift1",
    "vp_cosine16": "vp_cosine16",
    "visual_prompt_cosine16": "vp_cosine16",
    "visual_cosine_relative_vll_capped_topmass_085": "target_visual_hidden_cosine_relative_vll_capped_topmass_085",
    "target_visual_hidden_cosine_relative_vll_capped_topmass_085": (
        "target_visual_hidden_cosine_relative_vll_capped_topmass_085"
    ),
    "visualcosine_cap085": "target_visual_hidden_cosine_relative_vll_capped_topmass_085",
    "cosine_cap085": "target_visual_hidden_cosine_relative_vll_capped_topmass_085",
    "visual_prompt_cosine_visual_prompt_relative_vll": "target_visual_prompt_hidden_cosine_visual_prompt_relative_vll",
    "target_visual_prompt_hidden_cosine_visual_prompt_relative_vll": (
        "target_visual_prompt_hidden_cosine_visual_prompt_relative_vll"
    ),
    "visual_prompt_cosine_visual_prompt_relative_vll_capped_topmass_085": (
        "target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_capped_topmass_085"
    ),
    "target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_capped_topmass_085": (
        "target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_capped_topmass_085"
    ),
    "visual_prompt_cosine_capped_topmass_085": "target_visual_prompt_hidden_cosine_capped_topmass_085",
    "target_visual_prompt_hidden_cosine_capped_topmass_085": "target_visual_prompt_hidden_cosine_capped_topmass_085",
    "prompt_last_cosine": "prompt_last_cosine",
    "prompt_mean_cosine": "prompt_mean_cosine",
    "ffn_fad": "ffn_fad",
    "fad": "ffn_fad",
    "ffn_fad_x_risk_geo_raw": "ffn_fad_x_risk_geo_raw",
    "fad_x_risk_geo_raw": "ffn_fad_x_risk_geo_raw",
    "risk_geo_raw_x_ffn_fad": "ffn_fad_x_risk_geo_raw",
    "ffn_fad*risk_geo_raw": "ffn_fad_x_risk_geo_raw",
    "risk_geo_raw*ffn_fad": "ffn_fad_x_risk_geo_raw",
    "ffn_gate": "ffn_gate",
    "ffn_gate_ratio": "ffn_gate",
    "ffn_update_gate": "ffn_gate",
    "ffn_a": "ffn_gate",
    "ffn_al": "ffn_gate",
    "ffn_fgr": "ffn_fgr",
    "fgr": "ffn_fgr",
    "ffn_gate_x_risk_geo_raw": "ffn_fgr",
    "ffn_gate*risk_geo_raw": "ffn_fgr",
    "risk_geo_raw*ffn_gate": "ffn_fgr",
    "ffn_eifdose": "ffn_eifdose",
    "eifdose": "ffn_eifdose",
    "ffn_logitlift": "ffn_logitlift",
    "logitlift": "ffn_logitlift",
    "ffn_eiffrac_svd": "ffn_eiffrac_svd",
    "eiffrac_svd": "ffn_eiffrac_svd",
    "ffn_eifdose_svd": "ffn_eifdose_svd",
    "eifdose_svd": "ffn_eifdose_svd",
    "ffn_eifdose_svd_x_risk_geo_raw": "ffn_eifdose_svd_x_risk_geo_raw",
    "eifdose_svd_x_risk_geo_raw": "ffn_eifdose_svd_x_risk_geo_raw",
    "risk_geo_raw_x_ffn_eifdose_svd": "ffn_eifdose_svd_x_risk_geo_raw",
    "ffn_eifdose_svd*risk_geo_raw": "ffn_eifdose_svd_x_risk_geo_raw",
    "risk_geo_raw*ffn_eifdose_svd": "ffn_eifdose_svd_x_risk_geo_raw",
    "ffn_eiffrac_pca": "ffn_eiffrac_pca",
    "eiffrac_pca": "ffn_eiffrac_pca",
    "ffn_eifdose_pca": "ffn_eifdose_pca",
    "eifdose_pca": "ffn_eifdose_pca",
    "risk_capped_topmass_085_x_1_minus_target_visual_hidden_cosine": "risk_capped_topmass_085_x_1_minus_target_visual_hidden_cosine",
    "risk_capped_topmass_085_times_inverse_target_visual_hidden_cosine": "risk_capped_topmass_085_x_1_minus_target_visual_hidden_cosine",
    "risk_capped_topmass_085*(1-target_visual_hidden_cosine)": "risk_capped_topmass_085_x_1_minus_target_visual_hidden_cosine",
    "risk_capped_topmass_085*(1_target_visual_hidden_cosine)": "risk_capped_topmass_085_x_1_minus_target_visual_hidden_cosine",
}

FEATURE_KEYS = {
    "risk": "dgst_t_transport_risk_per_layer",
    "risk_topmass_085": "dgst_t_transport_risk_topmass_085_per_layer",
    "risk_capped_topmass_085": "dgst_t_transport_risk_capped_topmass_085_per_layer",
    "risk_relative_vll": "dgst_t_transport_risk_relative_vll_per_layer",
    "risk_relative_vll_capped_topmass_085": "dgst_t_transport_risk_relative_vll_capped_topmass_085_per_layer",
    "risk_visual_prompt_relative_vll": "dgst_t_transport_risk_visual_prompt_relative_vll_per_layer",
    "risk_visual_prompt_relative_vll_capped_topmass_085": (
        "dgst_t_transport_risk_visual_prompt_relative_vll_capped_topmass_085_per_layer"
    ),
    "risk_geo": "dgst_t_risk_geo_per_layer",
    "risk_cosine_hpre": "dgst_t_risk_cosine_hpre_per_layer",
    "risk_sqrt_hmid": "dgst_t_risk_sqrt_hmid_per_layer",
    "risk_sqrt_hpre": "dgst_t_risk_sqrt_hpre_per_layer",
    "risk_raw_attention_hmid": "dgst_t_risk_raw_attention_hmid_per_layer",
    "risk_raw_attention_hpre": "dgst_t_risk_raw_attention_hpre_per_layer",
    "gauss_risk_geo": "dgst_t_gauss_risk_geo_per_layer",
    "gauss_risk_cosine_hpre": "dgst_t_gauss_risk_cosine_hpre_per_layer",
    "gauss_risk_sqrt_hmid": "dgst_t_gauss_risk_sqrt_hmid_per_layer",
    "gauss_risk_sqrt_hpre": "dgst_t_gauss_risk_sqrt_hpre_per_layer",
    "risk_relative_vll_source_hmid_proj": (
        "dgst_t_transport_risk_relative_vll_source_hmid_proj_per_layer"
    ),
    "risk_relative_vll_source_hprev_cos": (
        "dgst_t_transport_risk_relative_vll_source_hprev_cos_per_layer"
    ),
    "risk_relative_vll_source_hprev_proj": (
        "dgst_t_transport_risk_relative_vll_source_hprev_proj_per_layer"
    ),
    "risk_visual_prompt_relative_vll_source_hprev_cos": (
        "dgst_t_transport_risk_visual_prompt_relative_vll_source_hprev_cos_per_layer"
    ),
    "risk_visual_prompt_relative_vll_source_hmid_proj": (
        "dgst_t_transport_risk_visual_prompt_relative_vll_source_hmid_proj_per_layer"
    ),
    "risk_visual_prompt_relative_vll_source_hprev_proj": (
        "dgst_t_transport_risk_visual_prompt_relative_vll_source_hprev_proj_per_layer"
    ),
    "c_vp": "dgst_t_c_vp_relative_vll_cost_geo_per_layer",
    "m_p": "dgst_t_m_p_per_layer",
    "r_es": "dgst_t_r_es_relative_vll_cost_geo_per_layer",
    "relative_vll_evidence_strength": "dgst_t_relative_vll_evidence_strength_per_layer",
    "vv_raw_evidence_strength": "dgst_t_vv_raw_evidence_strength_per_layer",
    "visual_prompt_relative_vll_evidence_strength": (
        "dgst_t_visual_prompt_relative_vll_evidence_strength_per_layer"
    ),
    "vp_raw_evidence_strength": "dgst_t_vp_raw_evidence_strength_per_layer",
    "vp_raw_evidence_visual_mass": "dgst_t_vp_raw_evidence_visual_mass_per_layer",
    "vp_raw_evidence_prompt_mass": "dgst_t_vp_raw_evidence_prompt_mass_per_layer",
    "visual_prompt_relative_vll_evidence_visual_mass": (
        "dgst_t_visual_prompt_relative_vll_evidence_visual_mass_per_layer"
    ),
    "visual_prompt_relative_vll_evidence_prompt_mass": (
        "dgst_t_visual_prompt_relative_vll_evidence_prompt_mass_per_layer"
    ),
    "visual_prompt_relative_vll_source_visual_mass": (
        "dgst_t_visual_prompt_relative_vll_source_visual_mass_per_layer"
    ),
    "visual_prompt_relative_vll_source_prompt_mass": (
        "dgst_t_visual_prompt_relative_vll_source_prompt_mass_per_layer"
    ),
    "vv_source_entropy": "dgst_t_vv_source_entropy_per_layer",
    "vv_target_entropy": "dgst_t_vv_target_entropy_per_layer",
    "vv_evidence_entropy": "dgst_t_vv_evidence_entropy_per_layer",
    "vv_source_topk_entropy": "dgst_t_vv_source_topk_entropy_per_layer",
    "vp_source_entropy": "dgst_t_vp_source_entropy_per_layer",
    "vp_target_entropy": "dgst_t_vp_target_entropy_per_layer",
    "vp_evidence_entropy": "dgst_t_vp_evidence_entropy_per_layer",
    "vp_source_topk_entropy": "dgst_t_vp_source_topk_entropy_per_layer",
    "js_relative_vll": "dgst_t_js_relative_vll_per_layer",
    "kl_target_source_relative_vll": "dgst_t_kl_target_source_relative_vll_per_layer",
    "kl_source_target_relative_vll": "dgst_t_kl_source_target_relative_vll_per_layer",
    "js_visual_prompt_relative_vll": "dgst_t_js_visual_prompt_relative_vll_per_layer",
    "kl_target_source_visual_prompt_relative_vll": (
        "dgst_t_kl_target_source_visual_prompt_relative_vll_per_layer"
    ),
    "kl_source_target_visual_prompt_relative_vll": (
        "dgst_t_kl_source_target_visual_prompt_relative_vll_per_layer"
    ),
    "prompt_confidence_top3": "dgst_t_prompt_confidence_top3_per_layer",
    "prompt_confidence_max": "dgst_t_prompt_confidence_max_per_layer",
    "context_confidence": "dgst_t_context_confidence_per_layer",
    "context_confidence_max_prompt": "dgst_t_context_confidence_max_prompt_per_layer",
    "target_visual_hidden_cosine": "dgst_t_target_visual_hidden_cosine_per_layer",
    "target_visual_prompt_hidden_cosine": "dgst_t_target_visual_prompt_hidden_cosine_per_layer",
    "target_visual_hidden_cosine_capped_topmass_085": "dgst_t_target_visual_hidden_cosine_capped_topmass_085_per_layer",
    "target_visual_hidden_cosine_relative_vll": "dgst_t_target_visual_hidden_cosine_relative_vll_per_layer",
    "cosine16": "dgst_t_target_visual_hidden_cosine16_relative_vll_per_layer",
    "target_visual_hpre_cosine_relative_vll": (
        "dgst_t_target_visual_hpre_cosine_relative_vll_per_layer"
    ),
    "target_visual_hpre_cosine16_relative_vll": (
        "dgst_t_target_visual_hpre_cosine16_relative_vll_per_layer"
    ),
    "target_visual_hidden_cosine_relative_vll_capped_topmass_085": (
        "dgst_t_target_visual_hidden_cosine_relative_vll_capped_topmass_085_per_layer"
    ),
    "target_visual_hpre_cosine_relative_vll_capped_topmass_085": (
        "dgst_t_target_visual_hpre_cosine_relative_vll_capped_topmass_085_per_layer"
    ),
    "target_visual_prompt_hidden_cosine_visual_prompt_relative_vll": (
        "dgst_t_target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_per_layer"
    ),
    "vp_target_cosine": (
        "dgst_t_target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_per_layer"
    ),
    "vp_cosine16": (
        "dgst_t_target_visual_prompt_hidden_cosine16_visual_prompt_relative_vll_per_layer"
    ),
    "vp_hpre_cosine": (
        "dgst_t_target_visual_prompt_hpre_cosine_visual_prompt_relative_vll_per_layer"
    ),
    "vp_hpre_cosine16": (
        "dgst_t_target_visual_prompt_hpre_cosine16_visual_prompt_relative_vll_per_layer"
    ),
    "target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_capped_topmass_085": (
        "dgst_t_target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_capped_topmass_085_per_layer"
    ),
    "target_visual_prompt_hpre_cosine_visual_prompt_relative_vll_capped_topmass_085": (
        "dgst_t_target_visual_prompt_hpre_cosine_visual_prompt_relative_vll_capped_topmass_085_per_layer"
    ),
    "target_visual_prompt_hidden_cosine_capped_topmass_085": "dgst_t_target_visual_prompt_hidden_cosine_capped_topmass_085_per_layer",
    "prompt_last_cosine": "dgst_t_prompt_last_cosine_per_layer",
    "prompt_mean_cosine": "dgst_t_prompt_mean_cosine_per_layer",
    "ffn_fad": "dgst_t_ffn_attn_dominance_per_layer",
    "ffn_eifdose": "dgst_t_ffn_evidence_orthogonal_dose_per_layer",
    "ffn_logitlift": "dgst_t_ffn_logit_lift_per_layer",
    "ffn_eiffrac_svd": "dgst_t_ffn_eif_fraction_svd_per_layer",
    "ffn_eifdose_svd": "dgst_t_ffn_eif_dose_svd_per_layer",
    "ffn_eiffrac_pca": "dgst_t_ffn_eif_fraction_pca_per_layer",
    "ffn_eifdose_pca": "dgst_t_ffn_eif_dose_pca_per_layer",
    "ffn_gate": "dgst_t_ffn_gate_ratio_per_layer",
    "ffn_fgr": "dgst_t_ffn_fgr_per_layer",
}

LAYER_STAT_KEYS = {
    "prompt_confidence_top3": "prompt_logit_lens_top3_confidence",
    "prompt_confidence_max": "prompt_logit_lens_max_confidence",
    "target_visual_hidden_cosine": "target_hidden_top32_visual_cosine",
    "target_visual_prompt_hidden_cosine": "target_hidden_top32_visual_prompt_cosine",
    "target_visual_hidden_cosine_capped_topmass_085": "target_hidden_capped_topmass_085_visual_cosine",
    "target_visual_hidden_cosine_relative_vll": "target_hidden_top32_visual_cosine_relative_vll",
    "target_visual_hpre_cosine_relative_vll": "target_hpre_top32_visual_cosine_relative_vll",
    "target_visual_hidden_cosine_relative_vll_capped_topmass_085": (
        "target_hidden_capped_topmass_085_visual_cosine_relative_vll"
    ),
    "target_visual_hpre_cosine_relative_vll_capped_topmass_085": (
        "target_hpre_capped_topmass_085_visual_cosine_relative_vll"
    ),
    "target_visual_prompt_hidden_cosine_visual_prompt_relative_vll": (
        "target_hidden_top32_visual_prompt_cosine_visual_prompt_relative_vll"
    ),
    "vp_hpre_cosine": (
        "target_hpre_top32_visual_prompt_cosine_visual_prompt_relative_vll"
    ),
    "target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_capped_topmass_085": (
        "target_hidden_capped_topmass_085_visual_prompt_cosine_visual_prompt_relative_vll"
    ),
    "target_visual_prompt_hpre_cosine_visual_prompt_relative_vll_capped_topmass_085": (
        "target_hpre_capped_topmass_085_visual_prompt_cosine_visual_prompt_relative_vll"
    ),
    "target_visual_prompt_hidden_cosine_capped_topmass_085": "target_hidden_capped_topmass_085_visual_prompt_cosine",
    "context_confidence": "context_confidence",
    "context_confidence_max_prompt": "context_confidence_max_prompt",
    "ffn_fad": "ffn_attn_dominance",
    "ffn_eifdose": "ffn_evidence_orthogonal_dose",
    "ffn_logitlift": "ffn_logit_lift",
    "ffn_eiffrac_svd": "ffn_eif_fraction_svd",
    "ffn_eifdose_svd": "ffn_eif_dose_svd",
    "ffn_eiffrac_pca": "ffn_eif_fraction_pca",
    "ffn_eifdose_pca": "ffn_eif_dose_pca",
    "ffn_gate": "ffn_gate_ratio",
    "ffn_fgr": "ffn_fgr",
}


def _register_delta_source_aliases() -> None:
    for target_slug in ("rvll", "vp_rvll"):
        for gamma_slug in ("g0", "g05", "g1"):
            block = f"{target_slug}_delta_{gamma_slug}"
            key_stem = f"{target_slug}_delta_src_{gamma_slug}"
            risk_key = f"dgst_t_risk_{key_stem}_per_layer"
            risk_cap_key = f"dgst_t_risk_{key_stem}_cap085_per_layer"
            cos_key = f"dgst_t_cos_{key_stem}_per_layer"
            cos_cap_key = f"dgst_t_cos_{key_stem}_cap085_per_layer"

            FEATURE_ALIASES.update(
                {
                    block: block,
                    f"risk_{block}": block,
                    f"{block}_cap085": f"{block}_cap085",
                    f"risk_{block}_cap085": f"{block}_cap085",
                    f"{block}_cos": f"{block}_cos",
                    f"cos_{block}": f"{block}_cos",
                    f"{block}_cos_cap085": f"{block}_cos_cap085",
                    f"{block}_cap085_cos": f"{block}_cos_cap085",
                    f"cos_{block}_cap085": f"{block}_cos_cap085",
                }
            )
            FEATURE_KEYS.update(
                {
                    block: risk_key,
                    f"{block}_cap085": risk_cap_key,
                    f"{block}_cos": cos_key,
                    f"{block}_cos_cap085": cos_cap_key,
                }
            )


def _register_relative_cost_aliases() -> None:
    for slug in ("geo", "tbar", "sbar", "tadd", "sadd", "tsadd", "qmatch"):
        for prefix in ("risk_relative_vll", "risk_visual_prompt_relative_vll"):
            block = f"{prefix}_cost_{slug}"
            cap_block = f"{block}_capped_topmass_085"
            FEATURE_ALIASES.update(
                {
                    block: block,
                    cap_block: cap_block,
                    f"{block}_cap085": cap_block,
                    f"{prefix}_{slug}": block,
                    f"{prefix}_{slug}_capped_topmass_085": cap_block,
                    f"{prefix}_{slug}_cap085": cap_block,
                }
            )
            FEATURE_KEYS.update(
                {
                    block: f"dgst_t_transport_{prefix}_cost_{slug}_per_layer",
                    cap_block: (
                        f"dgst_t_transport_{prefix}_cost_{slug}_capped_topmass_085_per_layer"
                    ),
                }
            )
    for state_slug in (
        "mid",
        "out",
        "avg",
        "stateupd_lu005",
        "stateupd_lu01",
        "stateupd_lu02",
        "stateupd_lu1",
    ):
        for prefix in ("risk_relative_vll", "risk_visual_prompt_relative_vll"):
            block = f"{prefix}_cost_geo_{state_slug}"
            FEATURE_ALIASES[block] = block
            FEATURE_KEYS[block] = f"dgst_t_transport_{prefix}_cost_geo_{state_slug}_per_layer"


def _register_topk_region_aliases() -> None:
    source_specs = (
        ("", ("",)),
        ("hmid_proj", ("hmid_proj", "hmid")),
        ("hprev_cos", ("hprev_cos", "hpre_cos")),
        ("hprev_proj", ("hprev_proj", "hpre_proj")),
    )
    stat_names = ("skm", "tkm", "cov_st", "es")
    selectors = ("union", "rec", "target")
    variants = ("lk", "", "la", "lp")

    for scope in ("vv", "vp"):
        for source_key, source_aliases in source_specs:
            stem = f"dgst_t_{scope}"
            alias_middle = ""
            if source_key:
                stem = f"{stem}_source_{source_key}"
                alias_middle = f"_{source_key}"
            for stat in stat_names:
                block = f"{scope}{alias_middle}_topk_{stat}"
                key = f"{stem}_topk_{stat}_per_layer"
                FEATURE_ALIASES[block] = block
                FEATURE_KEYS[block] = key
                if scope == "vv" and not source_key:
                    FEATURE_ALIASES[f"topk_{stat}"] = block
                for source_alias in source_aliases:
                    if source_alias:
                        alias = f"{scope}_{source_alias}_topk_{stat}"
                        FEATURE_ALIASES[alias] = block
            for selector in selectors:
                for variant in variants:
                    suffix = f"_{variant}" if variant else ""
                    block = f"{scope}{alias_middle}_r_{selector}{suffix}"
                    key = f"{stem}_r_{selector}{suffix}_per_layer"
                    FEATURE_ALIASES[block] = block
                    FEATURE_KEYS[block] = key
                    if scope == "vv" and not source_key:
                        FEATURE_ALIASES[f"r_{selector}{suffix}"] = block
                    for source_alias in source_aliases:
                        if source_alias:
                            alias = f"{scope}_{source_alias}_r_{selector}{suffix}"
                            FEATURE_ALIASES[alias] = block


def _register_gate_comparison_aliases() -> None:
    """Register matched COCO100 target-gate risk/cosine feature blocks."""
    specs = {
        "gate_relative_vll_risk": (
            "dgst_t_relative_vll_gauss_risk_sqrt_hpre_per_layer"
        ),
        "gate_relative_vll_hprecosine": (
            "dgst_t_relative_vll_gauss_target_visual_hpre_cosine_per_layer"
        ),
        "gate_softmax_relative_vll_risk": (
            "dgst_t_softmax_relative_vll_gauss_risk_sqrt_hpre_per_layer"
        ),
        "gate_softmax_relative_vll_hprecosine": (
            "dgst_t_softmax_relative_vll_gauss_target_visual_hpre_cosine_per_layer"
        ),
        "gate_legacy_prob_risk": (
            "dgst_t_legacy_prob_risk_sqrt_hpre_per_layer"
        ),
        "gate_legacy_prob_hprecosine": (
            "dgst_t_legacy_prob_target_visual_hpre_cosine_per_layer"
        ),
    }
    for block, feature_key in specs.items():
        FEATURE_ALIASES[block] = block
        FEATURE_KEYS[block] = feature_key


def _register_four_gate_aliases() -> None:
    """Register aliases for the active gates and raw-attention control."""
    methods = (
        "hpre_raw_logit_gauss",
        "hpre_raw_logit_relative_vll",
        "hpre_softmax_prob_gauss",
        "hmid_raw_logit_gauss",
        "hmid_softmax_prob_gauss",
        "hpre_softmax_prob_direct",
        "raw_attention",
    )
    for method in methods:
        state_name = "hmid" if method.startswith("hmid_") else "hpre"
        specs = {
            f"{method}_source_target_js": None,
            f"{method}_source_target_union_topk_js": None,
            f"{method}_one_minus_target_dist_mass_x_cosine": None,
            f"{method}_risk": (
                f"dgst_t_{method}_risk_sqrt_{state_name}_per_layer"
            ),
            f"{method}_target_cosine": (
                f"dgst_t_{method}_target_cosine_"
                f"topk{{target_region_top_k}}_{state_name}_per_layer"
            ),
            f"{method}_ev": (
                f"dgst_t_{method}_ev_"
                f"topk{{target_region_top_k}}_{state_name}_per_layer"
            ),
            f"{method}_ev_target_dist_mass_x_cosine": (
                f"dgst_t_{method}_ev_target_dist_mass_x_cosine_"
                f"topk{{target_region_top_k}}_{state_name}_per_layer"
            ),
        }
        for block, feature_key in specs.items():
            FEATURE_ALIASES[block] = block
            if feature_key is not None:
                FEATURE_KEYS[block] = feature_key
        cost_risk_specs = {
            f"{method}_risk_sqrt_matched_state": (
                f"dgst_t_{method}_risk_sqrt_{state_name}_per_layer"
            ),
            f"{method}_risk_geo_stateupd_lu1": (
                f"dgst_t_{method}_risk_geo_stateupd_lu1_per_layer"
            ),
            f"{method}_risk_cosine_matched_state": (
                f"dgst_t_{method}_risk_cosine_{state_name}_per_layer"
            ),
        }
        for alpha_tenth in range(1, 10):
            alpha_slug = f"0{alpha_tenth}"
            cost_risk_specs[
                f"{method}_risk_sqrt_stateupd_alpha{alpha_slug}"
            ] = f"dgst_t_{method}_risk_sqrt_stateupd_alpha{alpha_slug}_per_layer"
        for block, feature_key in cost_risk_specs.items():
            FEATURE_ALIASES[block] = block
            FEATURE_KEYS[block] = feature_key
        FEATURE_ALIASES[
            f"{method}_risk_sqrt_cosine_matched_state"
        ] = f"{method}_risk_sqrt_matched_state"

    # VP uses the same hpre/hmid construction over visual+prompt support.
    # Prefixing the training block and serialized field keeps it impossible to
    # accidentally compare a VP curve against the backward-compatible VV key.
    for method in methods:
        scoped_method = f"vp_{method}"
        state_name = "hmid" if method.startswith("hmid_") else "hpre"
        specs = {
            f"{scoped_method}_risk": (
                f"dgst_t_{scoped_method}_risk_sqrt_{state_name}_per_layer"
            ),
            f"{scoped_method}_target_cosine": (
                f"dgst_t_{scoped_method}_target_cosine_"
                f"topk{{target_region_top_k}}_{state_name}_per_layer"
            ),
            f"{scoped_method}_ev_target_dist_mass_x_cosine": (
                f"dgst_t_{scoped_method}_ev_target_dist_mass_x_cosine_"
                f"topk{{target_region_top_k}}_{state_name}_per_layer"
            ),
            f"{scoped_method}_risk_sqrt_matched_state": (
                f"dgst_t_{scoped_method}_risk_sqrt_{state_name}_per_layer"
            ),
            f"{scoped_method}_risk_geo_stateupd_lu1": (
                f"dgst_t_{scoped_method}_risk_geo_stateupd_lu1_per_layer"
            ),
            f"{scoped_method}_risk_cosine_matched_state": (
                f"dgst_t_{scoped_method}_risk_cosine_{state_name}_per_layer"
            ),
        }
        for alpha_tenth in range(1, 10):
            alpha_slug = f"0{alpha_tenth}"
            specs[
                f"{scoped_method}_risk_sqrt_stateupd_alpha{alpha_slug}"
            ] = f"dgst_t_{scoped_method}_risk_sqrt_stateupd_alpha{alpha_slug}_per_layer"
        for block, feature_key in specs.items():
            FEATURE_ALIASES[block] = block
            FEATURE_KEYS[block] = feature_key
        FEATURE_ALIASES[
            f"{scoped_method}_risk_sqrt_cosine_matched_state"
        ] = f"{scoped_method}_risk_sqrt_matched_state"


def _register_ads_cgc_aliases() -> None:
    """Register the two root ADS/CGC layerwise vectors."""
    for block, feature_key in {
        "ads": "ads_per_layer",
        "cgc": "cgc_per_layer",
    }.items():
        FEATURE_ALIASES[block] = block
        FEATURE_KEYS[block] = feature_key


_register_delta_source_aliases()
_register_relative_cost_aliases()
_register_topk_region_aliases()
_register_gate_comparison_aliases()
_register_four_gate_aliases()
_register_ads_cgc_aliases()


ATTENTION_TOPK_DIVERGENCE_BLOCKS = {
    "vv_attention_tk32_js": ("vv", "js"),
    "vv_attention_tk32_kl_attention_source": ("vv", "kl_attention_source"),
    "vv_attention_tk32_kl_source_attention": ("vv", "kl_source_attention"),
    "vp_attention_tk32_js": ("vp", "js"),
    "vp_attention_tk32_kl_attention_source": ("vp", "kl_attention_source"),
    "vp_attention_tk32_kl_source_attention": ("vp", "kl_source_attention"),
}
ATTENTION_TOPK_DIVERGENCE_CACHE_KEY = "_computed_attention_topk32_source_divergence"
ATTENTION_TOPK_DIVERGENCE_EPS = 1e-12
FOUR_GATE_SOURCE_TARGET_JS_BLOCKS = {
    f"{method}_source_target_js": method
    for method in (
        "hpre_raw_logit_gauss",
        "hpre_raw_logit_relative_vll",
        "hpre_softmax_prob_gauss",
        "hmid_raw_logit_gauss",
        "hmid_softmax_prob_gauss",
        "hpre_softmax_prob_direct",
        "raw_attention",
    )
}
FOUR_GATE_RISK_BLOCKS = {}
for _four_gate_method in (
    "hpre_raw_logit_gauss",
    "hpre_raw_logit_relative_vll",
    "hpre_softmax_prob_gauss",
    "hmid_raw_logit_gauss",
    "hmid_softmax_prob_gauss",
    "hpre_softmax_prob_direct",
    "raw_attention",
    "vp_hpre_raw_logit_gauss",
    "vp_hpre_raw_logit_relative_vll",
    "vp_hpre_softmax_prob_gauss",
    "vp_hmid_raw_logit_gauss",
    "vp_hmid_softmax_prob_gauss",
    "vp_hpre_softmax_prob_direct",
    "vp_raw_attention",
):
    FOUR_GATE_RISK_BLOCKS[f"{_four_gate_method}_risk"] = (
        _four_gate_method,
        None,
    )
    _four_gate_cost_aliases = [
        ("sqrt_matched_state", "sqrt_cosine_matched_state"),
        ("cosine_matched_state", "cosine_matched_state"),
        ("geo_stateupd_lu1", "geo_stateupd_lu1"),
        *[
            (
                f"sqrt_stateupd_alpha0{alpha_tenth}",
                f"sqrt_stateupd_alpha0{alpha_tenth}",
            )
            for alpha_tenth in range(1, 10)
        ],
    ]
    for _cost_alias, _cost_mode in _four_gate_cost_aliases:
        FOUR_GATE_RISK_BLOCKS[
            f"{_four_gate_method}_risk_{_cost_alias}"
        ] = (_four_gate_method, _cost_mode)
FOUR_GATE_SOURCE_TARGET_JS_CACHE_KEY = "_computed_four_gate_source_target_js"
FOUR_GATE_SOURCE_TARGET_UNION_TOPK_JS_BLOCKS = {
    f"{method}_source_target_union_topk_js": method
    for method in (
        "hpre_raw_logit_gauss",
        "hpre_raw_logit_relative_vll",
        "hpre_softmax_prob_gauss",
        "hmid_raw_logit_gauss",
        "hmid_softmax_prob_gauss",
        "hpre_softmax_prob_direct",
        "raw_attention",
    )
}
FOUR_GATE_SOURCE_TARGET_UNION_TOPK_JS_CACHE_KEY = (
    "_computed_four_gate_source_target_union_topk_js"
)
FOUR_GATE_ONE_MINUS_TARGET_MASS_COSINE_BLOCKS = {
    f"{method}_one_minus_target_dist_mass_x_cosine": method
    for method in (
        "hpre_raw_logit_gauss",
        "hpre_raw_logit_relative_vll",
        "hpre_softmax_prob_gauss",
        "hmid_raw_logit_gauss",
        "hmid_softmax_prob_gauss",
        "hpre_softmax_prob_direct",
        "raw_attention",
    )
}
FOUR_GATE_ONE_MINUS_TARGET_MASS_COSINE_CACHE_KEY = (
    "_computed_four_gate_one_minus_target_mass_x_cosine"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="configs/model_configs.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=[
            "risk",
            "target_cosine",
            "risk+target_cosine",
        ],
        help="Feature blocks to concatenate, e.g. risk+context_confidence.",
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        default=["xgb", "rf"],
        choices=["xgb", "rf", "mlp"],
    )
    parser.add_argument("--scoring", default="f1", choices=["f1", "accuracy", "auc"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from detection.train import evaluate_classifier, grid_search, split_by_image_id
    from utils.config_utils import get_classifier_cfgs, load_config

    config = load_config(args.config)
    if str((config.get("training") or {}).get("split_protocol", "strict_82_no_validation")) != "strict_82_no_validation":
        raise ValueError(
            "Training requires training.split_protocol=strict_82_no_validation"
        )
    if str((config.get("training") or {}).get("threshold_selection", "train_f1")) != "train_f1":
        raise ValueError(
            "Strict 8:2 training requires training.threshold_selection=train_f1"
        )
    clf_cfgs = get_classifier_cfgs(config)

    feature_path = os.path.join(args.output_dir, "features.pkl")
    splits_path = os.path.join(args.output_dir, "image_splits.json")
    results_dir = os.path.join(args.output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(feature_path):
        raise FileNotFoundError(feature_path)
    if not os.path.exists(splits_path):
        raise FileNotFoundError(splits_path)

    splits = load_json(splits_path)
    from utils.split_utils import validate_strict_82_split

    split_counts = validate_strict_82_split(splits)
    configured_count = int((config.get("dataset") or {}).get("num_images", 0))
    if configured_count and sum(split_counts.values()) != configured_count:
        raise ValueError(
            "Strict split size differs from dataset.num_images: "
            f"{sum(split_counts.values())} != {configured_count}"
        )
    all_features = load_validated_training_features(
        feature_path=feature_path,
        artifact_family="root",
        model_key=args.model,
        config=config,
        output_dir=args.output_dir,
        image_splits=splits,
    )
    train_feats, val_feats, test_feats = split_by_image_id(
        all_features,
        train_image_ids={int(x) for x in splits["train"]},
        val_image_ids=set(),
        test_image_ids={int(x) for x in splits["test"]},
    )

    out_path = os.path.join(results_dir, f"{args.model}_selected_feature_sets.json")
    results = load_json(out_path) if os.path.exists(out_path) else {}

    print(
        f"[FeatureSets] Loaded {len(all_features)} token features: "
        f"train={len(train_feats)}, val={len(val_feats)}, test={len(test_feats)}"
    )

    for feature_set in args.feature_sets:
        blocks = parse_feature_set(feature_set)
        X_train, y_train = build_selected_matrix(train_feats, blocks)
        X_val, y_val = build_selected_matrix(val_feats, blocks)
        X_test, y_test = build_selected_matrix(test_feats, blocks)
        _require_strict_binary_splits(
            feature_set=feature_set,
            train=(X_train, y_train),
            val=(X_val, y_val),
            test=(X_test, y_test),
        )

        print(f"\n[FeatureSets] {feature_set}: X={X_train.shape[1]} dims")
        set_results = results.setdefault(feature_set, {})
        for clf_name in args.classifiers:
            grid = deepcopy(clf_cfgs.get(clf_name, {}))
            _sanitise_grid(grid)
            best_clf, best_params, train_threshold_score = grid_search(
                clf_name,
                grid,
                X_train,
                y_train,
                X_val,
                y_val,
                scoring=args.scoring,
            )
            metrics = evaluate_classifier(best_clf, X_test, y_test)
            metrics["best_params"] = best_params
            metrics["val_score"] = None
            metrics["train_threshold_score"] = float(train_threshold_score)
            metrics["split_protocol"] = "strict_82_no_validation"
            metrics["checkpoint_selection"] = "single_fit"
            metrics["threshold_selection"] = "train_f1"
            metrics["num_features"] = int(X_train.shape[1])
            set_results[clf_name] = _json_ready(metrics)
            print(
                f"  {clf_name.upper():<4} "
                f"F1={metrics['f1']:.3f} AUC={metrics['auc']:.3f} "
                f"PR={metrics['precision']:.3f} RC={metrics['recall']:.3f}"
            )
        save_json(results, out_path)

    print(f"\n[FeatureSets] Saved results to {out_path}")
    _write_summary_table(out_path)


def _require_strict_binary_splits(
    *,
    feature_set: str,
    train: tuple[np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray],
    test: tuple[np.ndarray, np.ndarray],
) -> None:
    """Require pure 80/20 data: binary train/test and no validation rows."""
    for split_name, (matrix, labels) in (
        ("train", train),
        ("val", val),
        ("test", test),
    ):
        if split_name == "val":
            if matrix.shape[0] != 0 or labels.size != 0:
                raise ValueError(
                    f"Strict 8:2 requires an empty validation split for "
                    f"feature set {feature_set!r}."
                )
            continue
        classes = np.unique(labels)
        if matrix.shape[0] == 0 or classes.size < 2:
            raise ValueError(
                f"Strict outer-8:2 training split violation for feature set {feature_set!r}: "
                f"{split_name} has {matrix.shape[0]} rows and classes "
                f"{classes.tolist()}. Splits are never substituted; repair the "
                "labels/image split before training."
            )


def parse_feature_set(value: str) -> list[str]:
    blocks = []
    for raw in value.split("+"):
        key = _normalise_feature_key(raw)
        if not key:
            continue
        if key in FEATURE_ALIASES:
            blocks.append(FEATURE_ALIASES[key])
            continue
        factors = key.split("*")
        if len(factors) == 2:
            left = _resolve_feature_alias(factors[0])
            right = _resolve_feature_alias(factors[1])
            blocks.append(f"__product__:{left}:{right}")
            continue
        raise ValueError(
            f"Unknown feature block {raw!r}. Choices: {sorted(FEATURE_ALIASES)}"
        )
    if not blocks:
        raise ValueError(f"Empty feature set {value!r}.")
    return blocks


def _normalise_feature_key(value: str) -> str:
    key = value.strip().lower().replace("（", "(").replace("）", ")").replace(" ", "")
    if key not in FEATURE_ALIASES:
        key = key.replace("-", "_")
    return key


def _resolve_feature_alias(value: str) -> str:
    key = _normalise_feature_key(value)
    if key not in FEATURE_ALIASES:
        raise ValueError(
            f"Unknown product factor {value!r}. Choices: {sorted(FEATURE_ALIASES)}"
        )
    return FEATURE_ALIASES[key]


def build_selected_matrix(features: Sequence[dict], blocks: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    labels = []
    for feat in features:
        if feat.get("label") not in (0, 1):
            continue
        vectors = [feature_block(feat, block) for block in blocks]
        if any(vec.size == 0 for vec in vectors):
            continue
        rows.append(np.concatenate(vectors).astype(np.float32))
        labels.append(int(feat["label"]))
    if not rows:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int32)
    return np.stack(rows, axis=0), np.array(labels, dtype=np.int32)


def feature_block(feat: dict, block: str) -> np.ndarray:
    if block in FOUR_GATE_RISK_BLOCKS:
        return _four_gate_risk_block(feat, block)

    if block in FOUR_GATE_ONE_MINUS_TARGET_MASS_COSINE_BLOCKS:
        return _four_gate_one_minus_target_mass_x_cosine_block(feat, block)

    if block in FOUR_GATE_SOURCE_TARGET_JS_BLOCKS:
        return _four_gate_source_target_js_block(feat, block)

    if block in FOUR_GATE_SOURCE_TARGET_UNION_TOPK_JS_BLOCKS:
        return _four_gate_source_target_union_topk_js_block(feat, block)

    if block in ATTENTION_TOPK_DIVERGENCE_BLOCKS:
        return _attention_topk_source_divergence_block(feat, block)

    if block.startswith("__product__:"):
        _, left_block, right_block = block.split(":", 2)
        left = feature_block(feat, left_block)
        right = feature_block(feat, right_block)
        if left.shape != right.shape:
            raise ValueError(
                f"Product feature blocks {left_block!r} and {right_block!r} must "
                f"have the same shape, got {left.shape} and {right.shape}."
            )
        return (left * right).astype(np.float32)

    if block == "target_visual_hidden_cosine_relative_vll_shift1":
        return _shift_right_one_layer(feature_block(feat, "target_visual_hidden_cosine_relative_vll"))

    if block == "vp_target_cosine_shift1":
        return _shift_right_one_layer(feature_block(feat, "vp_target_cosine"))

    if block == "target_cosine":
        for key in (
            "dgst_t_target_visual_hidden_cosine_relative_vll_per_layer",
            "dgst_t_target_visual_hidden_cosine_per_layer",
            "dgst_t_atarget_visual_cosine_per_layer",
        ):
            values = feat.get(key)
            if values is not None:
                return np.asarray(values, dtype=np.float32).reshape(-1)
        raise KeyError(
            "Feature block 'target_cosine' requires one of "
            "dgst_t_target_visual_hidden_cosine_relative_vll_per_layer, "
            "dgst_t_target_visual_hidden_cosine_per_layer, or "
            "dgst_t_atarget_visual_cosine_per_layer."
        )

    if block == "risk_capped_topmass_085_x_1_minus_target_visual_hidden_cosine":
        risk = feature_block(feat, "risk_capped_topmass_085")
        visual = feature_block(feat, "target_visual_hidden_cosine")
        if risk.shape != visual.shape:
            raise ValueError(
                "risk_capped_topmass_085 and target_visual_hidden_cosine must have "
                f"the same shape, got {risk.shape} and {visual.shape}."
            )
        return (risk * (1.0 - visual)).astype(np.float32)

    if block == "ffn_eifdose_svd_x_risk_geo_raw":
        dose = feature_block(feat, "ffn_eifdose_svd")
        risk = feature_block(feat, "risk_relative_vll_cost_geo")
        if dose.shape != risk.shape:
            raise ValueError(
                "ffn_eifdose_svd and risk_geo_raw must have the same shape, "
                f"got {dose.shape} and {risk.shape}."
            )
        return (dose * risk).astype(np.float32)

    if block == "ffn_fad_x_risk_geo_raw":
        fad = feature_block(feat, "ffn_fad")
        risk = feature_block(feat, "risk_relative_vll_cost_geo")
        if fad.shape != risk.shape:
            raise ValueError(
                "ffn_fad and risk_geo_raw must have the same shape, "
                f"got {fad.shape} and {risk.shape}."
            )
        return (fad * risk).astype(np.float32)

    if block == "vv_raw_evidence_strength":
        values = feat.get(FEATURE_KEYS[block])
        if values is not None:
            return np.asarray(values, dtype=np.float32).reshape(-1)
        evidence = feature_block(feat, "relative_vll_evidence_strength")
        alpha_img = feat.get("alpha_img_per_layer")
        if alpha_img is None:
            raise KeyError(
                "Feature block 'vv_raw_evidence_strength' requires either "
                "'dgst_t_vv_raw_evidence_strength_per_layer' or 'alpha_img_per_layer'."
            )
        alpha = np.asarray(alpha_img, dtype=np.float32).reshape(-1)
        if evidence.shape != alpha.shape:
            raise ValueError(
                "relative_vll_evidence_strength and alpha_img_per_layer must have "
                f"the same shape, got {evidence.shape} and {alpha.shape}."
            )
        return (evidence * alpha).astype(np.float32)

    key = FEATURE_KEYS[block]
    if "{target_region_top_k}" in key:
        configured_top_k = feat.get("dgst_t_target_region_top_k")
        if configured_top_k is None:
            # Historical four-gate artifacts predate configurable target K and
            # were always serialized with K=32.
            configured_top_k = 32
        key = key.format(target_region_top_k=int(configured_top_k))
    values = feat.get(key)
    if values is None and block == "risk":
        values = feat.get("dgst_t_per_layer")
    if values is None and block == "target_visual_hidden_cosine":
        values = feat.get("dgst_t_atarget_visual_cosine_per_layer")
    if values is None and block in LAYER_STAT_KEYS:
        values = [item.get(LAYER_STAT_KEYS[block], 0.0) for item in feat.get("dgst_t_layer_stats", [])]
    if values is None:
        raise KeyError(f"Feature block {block!r} requires missing key {key!r}.")
    return np.asarray(values, dtype=np.float32).reshape(-1)


def _four_gate_risk_block(feat: dict, block: str) -> np.ndarray:
    method, explicit_cost_mode = FOUR_GATE_RISK_BLOCKS[block]
    cost_mode = explicit_cost_mode or feat.get("dgst_t_cost")
    if cost_mode == "geo_stateupd_lu1":
        key = f"dgst_t_{method}_risk_geo_stateupd_lu1_per_layer"
    elif str(cost_mode).startswith("sqrt_stateupd_alpha0"):
        key = f"dgst_t_{method}_risk_{cost_mode}_per_layer"
    elif cost_mode == "cosine_matched_state":
        state_name = "hmid" if method.removeprefix("vp_").startswith("hmid_") else "hpre"
        key = f"dgst_t_{method}_risk_cosine_{state_name}_per_layer"
    else:
        key = FEATURE_KEYS[block]
    values = feat.get(key)
    if values is None:
        raise KeyError(f"Feature block {block!r} requires missing key {key!r}.")
    return np.asarray(values, dtype=np.float32).reshape(-1)


def _four_gate_source_target_js_block(feat: dict, block: str) -> np.ndarray:
    method = FOUR_GATE_SOURCE_TARGET_JS_BLOCKS[block]
    cache = feat.setdefault(FOUR_GATE_SOURCE_TARGET_JS_CACHE_KEY, {})
    if method not in cache:
        source_key = "dgst_t_source_dist_per_layer"
        source_values = feat.get(source_key)
        if source_values is None:
            raise KeyError(f"Feature block {block!r} requires {source_key!r}.")

        source = np.asarray(source_values, dtype=np.float64)
        target = _four_gate_target_values(feat, method, block)
        if source.ndim != 2 or source.shape != target.shape:
            raise ValueError(
                "Four-gate source-target JS requires matching [layers, visual_tokens] "
                f"matrices, got source={source.shape}, target={target.shape}."
            )
        source_prob = _smooth_probability_rows_numpy(source)
        target_prob = _smooth_probability_rows_numpy(target)
        midpoint = 0.5 * (source_prob + target_prob)
        cache[method] = (
            0.5
            * np.sum(
                target_prob * (np.log(target_prob) - np.log(midpoint)), axis=-1
            )
            + 0.5
            * np.sum(
                source_prob * (np.log(source_prob) - np.log(midpoint)), axis=-1
            )
        ).astype(np.float32)
    return np.asarray(cache[method], dtype=np.float32).reshape(-1)


def _four_gate_source_target_union_topk_js_block(
    feat: dict,
    block: str,
) -> np.ndarray:
    method = FOUR_GATE_SOURCE_TARGET_UNION_TOPK_JS_BLOCKS[block]
    cache = feat.setdefault(FOUR_GATE_SOURCE_TARGET_UNION_TOPK_JS_CACHE_KEY, {})
    if method not in cache:
        source_key = "dgst_t_source_dist_per_layer"
        source_values = feat.get(source_key)
        if source_values is None:
            raise KeyError(f"Feature block {block!r} requires {source_key!r}.")
        source = _normalize_probability_rows_numpy(source_values)
        target = _normalize_probability_rows_numpy(
            _four_gate_target_values(feat, method, block)
        )
        if source.ndim != 2 or source.shape != target.shape:
            raise ValueError(
                "Four-gate union-topK JS requires matching "
                "[layers, visual_tokens] matrices, got "
                f"source={source.shape}, target={target.shape}."
            )
        if source.shape[-1] <= 0:
            raise ValueError("DGST distributions must contain visual tokens.")

        transport_top_k = int(feat.get("dgst_t_transport_top_k", 32))
        if transport_top_k <= 0:
            raise ValueError("DGST transport top-K must be positive.")
        side_top_k = min(
            max(transport_top_k // 2, 1),
            int(source.shape[-1]),
        )
        source_topk_cache_key = f"_source_topk_{transport_top_k}"
        source_indices = cache.get(source_topk_cache_key)
        if source_indices is None:
            source_indices = np.argsort(
                -source, axis=-1, kind="stable"
            )[:, :side_top_k]
            cache[source_topk_cache_key] = source_indices
        target_indices = np.argsort(
            -target, axis=-1, kind="stable"
        )[:, :side_top_k]

        layer_indices = np.arange(source.shape[0])[:, None]
        union_mask = np.zeros(source.shape, dtype=bool)
        union_mask[layer_indices, source_indices] = True
        union_mask[layer_indices, target_indices] = True
        source_region = _smooth_masked_probability_rows_numpy(source, union_mask)
        target_region = _smooth_masked_probability_rows_numpy(target, union_mask)
        midpoint = 0.5 * (source_region + target_region)
        source_safe = np.maximum(source_region, ATTENTION_TOPK_DIVERGENCE_EPS)
        target_safe = np.maximum(target_region, ATTENTION_TOPK_DIVERGENCE_EPS)
        midpoint_safe = np.maximum(midpoint, ATTENTION_TOPK_DIVERGENCE_EPS)
        js = 0.5 * np.sum(
            np.where(
                union_mask,
                target_region * (np.log(target_safe) - np.log(midpoint_safe)),
                0.0,
            ),
            axis=-1,
        )
        js += 0.5 * np.sum(
            np.where(
                union_mask,
                source_region * (np.log(source_safe) - np.log(midpoint_safe)),
                0.0,
            ),
            axis=-1,
        )
        cache[method] = js.astype(np.float32)
    return np.asarray(cache[method], dtype=np.float32).reshape(-1)


def _four_gate_one_minus_target_mass_x_cosine_block(
    feat: dict,
    block: str,
) -> np.ndarray:
    method = FOUR_GATE_ONE_MINUS_TARGET_MASS_COSINE_BLOCKS[block]
    cache = feat.setdefault(FOUR_GATE_ONE_MINUS_TARGET_MASS_COSINE_CACHE_KEY, {})
    if method not in cache:
        target = _normalize_probability_rows_numpy(
            _four_gate_target_values(feat, method, block)
        )
        if target.shape[-1] <= 0:
            raise ValueError("DGST target distribution must contain visual tokens.")
        configured_top_k = int(feat.get("dgst_t_target_region_top_k", 32))
        if configured_top_k <= 0:
            raise ValueError("DGST target-region top-K must be positive.")
        top_k = min(configured_top_k, int(target.shape[-1]))
        target_mass = np.partition(
            target,
            kth=target.shape[-1] - top_k,
            axis=-1,
        )[:, -top_k:].sum(axis=-1)
        target_mass = np.clip(target_mass, 0.0, 1.0)
        cosine = feature_block(feat, f"{method}_target_cosine").astype(
            np.float64, copy=False
        )
        if cosine.shape != target_mass.shape:
            raise ValueError(
                f"Feature block {block!r} requires matching layer curves, got "
                f"mass={target_mass.shape}, cosine={cosine.shape}."
            )
        cache[method] = ((1.0 - target_mass) * cosine).astype(np.float32)
    return np.asarray(cache[method], dtype=np.float32).reshape(-1)


def _four_gate_target_values(feat: dict, method: str, block: str) -> np.ndarray:
    attention_key = "dgst_t_attention_support_per_layer"
    attention_values = feat.get(attention_key)
    if attention_values is None:
        raise KeyError(f"Feature block {block!r} requires {attention_key!r}.")
    attention = np.asarray(attention_values, dtype=np.float64)
    if attention.ndim != 2:
        raise ValueError(
            f"Feature block {block!r} requires a [layers, visual_tokens] attention "
            f"matrix, got {attention.shape}."
        )

    if method == "raw_attention":
        return attention
    if method == "hpre_softmax_prob_direct":
        target_key = "dgst_t_hpre_softmax_prob_direct_target_dist_per_layer"
        target_values = feat.get(target_key)
        if target_values is None:
            raise KeyError(f"Feature block {block!r} requires {target_key!r}.")
        target = np.asarray(target_values, dtype=np.float64)
    else:
        gate_key = f"dgst_t_{method}_gate_per_layer"
        gate_values = feat.get(gate_key)
        if gate_values is None:
            raise KeyError(f"Feature block {block!r} requires {gate_key!r}.")
        gate = np.asarray(gate_values, dtype=np.float64)
        if gate.shape != attention.shape:
            raise ValueError(
                f"Feature block {block!r} requires matching gate and attention "
                f"matrices, got gate={gate.shape}, attention={attention.shape}."
            )
        target = attention * gate
    if target.ndim != 2 or target.shape != attention.shape:
        raise ValueError(
            f"Feature block {block!r} requires target and attention to have the "
            f"same shape, got target={target.shape}, attention={attention.shape}."
        )
    return target


def _attention_topk_source_divergence_block(feat: dict, block: str) -> np.ndarray:
    scope, metric = ATTENTION_TOPK_DIVERGENCE_BLOCKS[block]
    cache = feat.setdefault(ATTENTION_TOPK_DIVERGENCE_CACHE_KEY, {})
    if scope not in cache:
        attention_key = f"dgst_t_{scope}_support_attention_per_layer"
        source_key = f"dgst_t_{scope}_source_dist_per_layer"
        attention_values = feat.get(attention_key)
        source_values = feat.get(source_key)
        if attention_values is None or source_values is None:
            raise KeyError(
                f"Feature block {block!r} requires {attention_key!r} and {source_key!r}."
            )

        attention = torch.as_tensor(attention_values).detach().cpu().to(dtype=torch.float64)
        source = torch.as_tensor(source_values).detach().cpu().to(dtype=torch.float64)
        if attention.ndim != 2 or source.ndim != 2 or attention.shape != source.shape:
            raise ValueError(
                "Attention-TK divergence requires matching [layers, support] tensors, got "
                f"attention={tuple(attention.shape)}, source={tuple(source.shape)}."
            )
        if attention.shape[-1] <= 0:
            raise ValueError("Attention-TK divergence requires non-empty support.")

        attention = torch.nan_to_num(
            attention, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp_min(0.0)
        source = torch.nan_to_num(source, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        top_k = min(32, int(attention.shape[-1]))
        indices = torch.topk(
            attention, k=top_k, dim=-1, largest=True, sorted=False
        ).indices
        attention_prob = _smooth_probability_rows(torch.gather(attention, -1, indices))
        source_prob = _smooth_probability_rows(torch.gather(source, -1, indices))
        midpoint = 0.5 * (attention_prob + source_prob)

        cache[scope] = {
            "js": (
                0.5
                * torch.sum(
                    attention_prob * (torch.log(attention_prob) - torch.log(midpoint)), dim=-1
                )
                + 0.5
                * torch.sum(
                    source_prob * (torch.log(source_prob) - torch.log(midpoint)), dim=-1
                )
            )
            .numpy()
            .astype(np.float32),
            "kl_attention_source": torch.sum(
                attention_prob * (torch.log(attention_prob) - torch.log(source_prob)), dim=-1
            )
            .numpy()
            .astype(np.float32),
            "kl_source_attention": torch.sum(
                source_prob * (torch.log(source_prob) - torch.log(attention_prob)), dim=-1
            )
            .numpy()
            .astype(np.float32),
        }
    return np.asarray(cache[scope][metric], dtype=np.float32).reshape(-1)


def _smooth_probability_rows(values: torch.Tensor) -> torch.Tensor:
    values = torch.nan_to_num(
        values.to(dtype=torch.float64), nan=0.0, posinf=0.0, neginf=0.0
    ).clamp_min(0.0)
    totals = values.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(values, 1.0 / float(values.shape[-1]))
    probabilities = torch.where(
        totals > 0.0,
        values / totals.clamp_min(ATTENTION_TOPK_DIVERGENCE_EPS),
        uniform,
    )
    probabilities = probabilities.clamp_min(ATTENTION_TOPK_DIVERGENCE_EPS)
    return probabilities / probabilities.sum(dim=-1, keepdim=True)


def _smooth_probability_rows_numpy(values: np.ndarray) -> np.ndarray:
    probabilities = _normalize_probability_rows_numpy(values)
    probabilities = np.maximum(probabilities, ATTENTION_TOPK_DIVERGENCE_EPS)
    return probabilities / probabilities.sum(axis=-1, keepdims=True)


def _normalize_probability_rows_numpy(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(
        np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )
    values = np.maximum(values, 0.0)
    totals = values.sum(axis=-1, keepdims=True)
    uniform = np.full_like(values, 1.0 / float(values.shape[-1]))
    probabilities = np.divide(
        values,
        np.maximum(totals, ATTENTION_TOPK_DIVERGENCE_EPS),
        out=uniform,
        where=totals > 0.0,
    )
    return probabilities


def _smooth_masked_probability_rows_numpy(
    probabilities: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    values = np.where(mask, probabilities, 0.0)
    totals = values.sum(axis=-1, keepdims=True)
    support_sizes = mask.sum(axis=-1, keepdims=True)
    if np.any(totals <= 0.0) or np.any(support_sizes <= 0):
        raise ValueError("Union-topK support must have positive probability mass.")
    values = values / totals
    values = np.where(
        mask,
        np.maximum(values, ATTENTION_TOPK_DIVERGENCE_EPS),
        0.0,
    )
    return values / values.sum(axis=-1, keepdims=True)


def _shift_right_one_layer(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return values
    shifted = np.empty_like(values)
    shifted[0] = 0.0
    shifted[1:] = values[:-1]
    return shifted


def _sanitise_grid(grid: dict) -> None:
    for key, value in list(grid.items()):
        if not isinstance(value, list):
            grid[key] = [value]


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _write_summary_table(results_path: str) -> None:
    try:
        write_summary_tables(results_path, formats=("md",), print_table=True)
    except Exception as exc:
        print(f"[FeatureSets] WARNING: failed to write summary table: {exc}")


if __name__ == "__main__":
    main()

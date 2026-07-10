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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.io_utils import load_json, load_pkl, save_json

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
    "risk_geo_raw": "risk_relative_vll_cost_geo",
    "risk_geo_cap085": "risk_relative_vll_cost_geo_capped_topmass_085",
    "risk_geo_capped_topmass_085": "risk_relative_vll_cost_geo_capped_topmass_085",
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


_register_delta_source_aliases()
_register_relative_cost_aliases()
_register_topk_region_aliases()


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
    clf_cfgs = get_classifier_cfgs(config)

    feature_path = os.path.join(args.output_dir, "features.pkl")
    splits_path = os.path.join(args.output_dir, "image_splits.json")
    results_dir = os.path.join(args.output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(feature_path):
        raise FileNotFoundError(feature_path)
    if not os.path.exists(splits_path):
        raise FileNotFoundError(splits_path)

    all_features = load_pkl(feature_path)
    splits = load_json(splits_path)
    train_feats, val_feats, test_feats = split_by_image_id(
        all_features,
        train_image_ids={int(x) for x in splits["train"]},
        val_image_ids={int(x) for x in splits["val"]},
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
        if X_train.shape[0] == 0:
            raise ValueError(f"No training rows for feature set {feature_set!r}.")
        if X_val.shape[0] == 0 or len(np.unique(y_val)) < 2:
            print(f"[FeatureSets] WARNING: val split unusable for {feature_set}; using train as val.")
            X_val, y_val = X_train.copy(), y_train.copy()
        if X_test.shape[0] == 0 or len(np.unique(y_test)) < 2:
            print(f"[FeatureSets] WARNING: test split unusable for {feature_set}; using val as test.")
            X_test, y_test = X_val.copy(), y_val.copy()

        print(f"\n[FeatureSets] {feature_set}: X={X_train.shape[1]} dims")
        set_results = results.setdefault(feature_set, {})
        for clf_name in args.classifiers:
            grid = deepcopy(clf_cfgs.get(clf_name, {}))
            _sanitise_grid(grid)
            best_clf, best_params, val_score = grid_search(
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
            metrics["val_score"] = float(val_score)
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


def parse_feature_set(value: str) -> list[str]:
    blocks = []
    for raw in value.split("+"):
        key = raw.strip().lower().replace("（", "(").replace("）", ")").replace(" ", "")
        if key not in FEATURE_ALIASES:
            key = key.replace("-", "_")
        if not key:
            continue
        if key not in FEATURE_ALIASES:
            raise ValueError(f"Unknown feature block {raw!r}. Choices: {sorted(FEATURE_ALIASES)}")
        blocks.append(FEATURE_ALIASES[key])
    if not blocks:
        raise ValueError(f"Empty feature set {value!r}.")
    return blocks


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

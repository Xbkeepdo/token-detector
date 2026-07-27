import numpy as np

from features.qa_extractor import (
    JSONLCheckpointStore,
    QA_FEATURE_SCHEMA_VERSION,
    _compact_dgst,
    find_answer_semantic_token,
    qa_prompt,
)


class TinyTokenizer:
    table = {1: "\n", 2: " Yes", 3: ".", 4: "nobody"}

    def encode(self, text, add_special_tokens=False):
        mapping = {"yes": [2], " yes": [2], "Yes": [2], " Yes": [2]}
        return mapping.get(text, [])

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.table.get(item, "") for item in ids)


def test_find_first_answer_semantic_token():
    assert find_answer_semantic_token([1, 2, 3], TinyTokenizer(), "yes") == 1
    assert find_answer_semantic_token([4], TinyTokenizer(), None) is None


def test_wrappers_receive_the_same_raw_qa_instruction():
    expected = "Is there a cat?\nAnswer only yes or no."
    assert qa_prompt("llava_1_5_7b", "Is there a cat?") == expected
    assert qa_prompt("llava_onevision_1_5_8b", "Is there a cat?") == expected
    assert qa_prompt("qwen2_5_vl_7b", "Is there a cat?") == expected


def test_jsonl_resume_does_not_rewrite_identical_row(tmp_path):
    path = tmp_path / "rows.jsonl"
    store = JSONLCheckpointStore(str(path), checkpoint_every=1)
    store.add({"key": "x", "value": 1})
    first_mtime = path.stat().st_mtime_ns
    resumed = JSONLCheckpointStore(str(path), checkpoint_every=1)
    resumed.add({"key": "x", "value": 1})
    assert resumed.dirty == 0
    assert path.stat().st_mtime_ns == first_mtime


def _scope_result(*scopes):
    method = "hpre_raw_logit_gauss"
    result = {
        "dgst_t_profile": "four_gate_vp_v1",
        "dgst_t_mad_axis": "visual_prompt_tokens",
        "dgst_t_mad_scale": 1.4826,
        "dgst_t_softmax_axis": "vocabulary",
        "dgst_t_source_distribution_mode": "softmax",
        "dgst_t_state_by_method": {method: "hpre"},
        "dgst_t_transport_top_k": 64,
        "dgst_t_target_region_top_k": 32,
        "dgst_t_ev_definition": "test",
        "dgst_t_cost": "sqrt_matched_state",
        "dgst_t_ot_solver": "pot_emd",
        "dgst_t_four_gate_methods": [method],
        "dgst_t_four_gate_support_scopes": list(scopes),
        "dgst_t_prompt_cafe": 0.75,
        "dgst_t_prompt_cafe_per_layer": np.asarray(
            [0.25, 0.75], dtype=np.float32
        ),
    }
    for scope in scopes:
        prefix = {
            "visual": "",
            "visual_prompt": "vp_",
            "visual_prompt_end": "vpend_",
        }[scope]
        base = f"dgst_t_{prefix}{method}"
        result[f"dgst_t_{prefix}attention_support_per_layer"] = np.asarray(
            [[1.0, 2.0]], dtype=np.float32
        )
        result[f"dgst_t_{prefix}source_dist_per_layer"] = np.asarray(
            [[3.0, 4.0]], dtype=np.float32
        )
        result[f"{base}_risk_sqrt_hpre_per_layer"] = np.asarray(
            [0.1, 0.2], dtype=np.float32
        )
        result[f"{base}_risk_sqrt_stateupd_alpha01_per_layer"] = np.asarray(
            [0.15, 0.25], dtype=np.float32
        )
        result[f"{base}_target_cosine_topk32_hpre_per_layer"] = np.asarray(
            [0.3, 0.4], dtype=np.float32
        )
        result[
            f"{base}_ev_target_dist_mass_x_cosine_topk32_hpre_per_layer"
        ] = np.asarray([0.5, 0.6], dtype=np.float32)
        result[f"{base}_gate_per_layer"] = np.asarray(
            [[0.7, 0.8]], dtype=np.float32
        )
    return result


def test_compact_dgst_accepts_vp_only_fields():
    compact = _compact_dgst(_scope_result("visual_prompt"))
    method = "vp_hpre_raw_logit_gauss"
    assert QA_FEATURE_SCHEMA_VERSION == "qa-position-comparison-v7"
    assert compact["support_modes"] == ["vp"]
    assert compact["scoped_methods"] == [method]
    assert method in compact
    assert "hpre_raw_logit_gauss" not in compact
    np.testing.assert_allclose(
        compact[method]["risk_sqrt_stateupd_alpha01"], [0.15, 0.25]
    )
    np.testing.assert_array_equal(
        compact["matrices"]["source_dist"], [[3.0, 4.0]]
    )
    np.testing.assert_array_equal(
        compact["matrices_by_scope"]["vp"]["attention_support"],
        [[1.0, 2.0]],
    )
    assert compact["prompt_cafe"] == 0.75
    np.testing.assert_allclose(
        compact["prompt_cafe_per_layer"], [0.25, 0.75]
    )


def test_compact_dgst_keeps_vv_and_vp_branches_together():
    compact = _compact_dgst(
        _scope_result("visual", "visual_prompt")
    )
    assert compact["support_modes"] == ["vv", "vp"]
    assert compact["scoped_methods"] == [
        "hpre_raw_logit_gauss",
        "vp_hpre_raw_logit_gauss",
    ]
    assert set(compact["matrices_by_scope"]) == {"vv", "vp"}
    assert compact["hpre_raw_logit_gauss"]["support_mode"] == "vv"
    assert compact["vp_hpre_raw_logit_gauss"]["support_mode"] == "vp"


def test_compact_dgst_accepts_vpend_only_fields():
    compact = _compact_dgst(_scope_result("visual_prompt_end"))
    method = "vpend_hpre_raw_logit_gauss"
    assert compact["support_modes"] == ["vpend"]
    assert compact["scoped_methods"] == [method]
    assert method in compact
    assert "vp_hpre_raw_logit_gauss" not in compact
    np.testing.assert_array_equal(
        compact["matrices_by_scope"]["vpend"]["attention_support"],
        [[1.0, 2.0]],
    )
    assert compact[method]["support_mode"] == "vpend"

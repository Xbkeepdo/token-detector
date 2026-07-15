import numpy as np

from detection.qa_probe import (
    classification_metrics,
    default_feature_sets,
    feature_vector,
    train_one_seed,
)
from features.dgst_t import COST_VARIANT_RISK_KEYS


def _row():
    target = {name: [0.1, 0.2] for name in COST_VARIANT_RISK_KEYS}
    target["hprecosine"] = [0.3, 0.4]
    return {
        "ads_score": 1.0,
        "ads_per_layer": [1.0, 2.0],
        "answer_cgc_score": 3.0,
        "answer_cgc_per_layer": [3.0, 4.0],
        "token_log_probability": -0.1,
        "token_entropy": 0.2,
        "token_nll": 0.1,
        "svar_score": 0.4,
        "attention_per_head_mid": [0.5, 0.6],
        "targets": {"answer": target, "object": target},
    }


def test_fixed_feature_sets_include_all_ten_risks_and_target_comparison():
    names = default_feature_sets("pope")
    for risk in COST_VARIANT_RISK_KEYS:
        assert f"{risk}@answer" in names
        assert f"{risk}+hprecosine@answer" in names
        assert f"{risk}@object" in names
        assert f"{risk}+hprecosine@object" in names


def test_feature_vectors_have_expected_blocks():
    row = _row()
    assert feature_vector(row, "ads").shape == (3,)
    assert feature_vector(row, "token_uncertainty").shape == (3,)
    assert feature_vector(row, "risk_geo+hprecosine@answer").shape == (4,)
    assert feature_vector(row, "best_dgst_legacy:risk_geo+hprecosine@answer").size > 4


def test_metrics_use_real_as_positive_class():
    result = classification_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), 0.5)
    assert result["accuracy"] == 1.0
    assert result["real"]["f1"] == 1.0
    assert result["hallucination"]["f1"] == 1.0


def test_weighted_torch_probe_smoke(tmp_path):
    rows = []
    for split, count in (("train", 16), ("val", 8), ("test", 8)):
        for index in range(count):
            label = index % 2
            rows.append({
                "key": f"{split}-{index}",
                "probe_split": split,
                "label": label,
                "token_log_probability": -0.8 + label,
                "token_entropy": 0.8 - 0.5 * label,
                "token_nll": 0.8 - label,
                "source_split": "random",
                "error_type": "correct_yes" if label else "false_positive",
            })
    result = train_one_seed(
        rows,
        "token_uncertainty",
        42,
        str(tmp_path),
        {
            "hidden_sizes": [8, 4, 2],
            "dropout": 0.1,
            "batch_size": 4,
            "epochs": 3,
            "learning_rate": 0.01,
        },
        device="cpu",
    )
    assert 0.0 <= result["threshold"] <= 1.0
    assert result["positive_class"] == "real"
    assert (tmp_path / "checkpoint.pt").exists()

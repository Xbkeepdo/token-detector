import numpy as np

from detection.qa_probe import (
    QA_DGST_METHODS,
    build_matrix,
    aggregate_seed_results,
    classification_metrics,
    choose_real_f1_threshold,
    feature_set_position,
    feature_vector,
    legacy_feature_sets,
    train_one_seed,
    validate_image_level_splits,
)
from features.dgst_t import COST_VARIANT_RISK_KEYS
from scripts.train_qa_probes import write_markdown_summary


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


def _current_row():
    dgst = {
        method: {
            "risk": np.array([0.1, 0.2], dtype=np.float32),
            "target_cosine": np.array([0.3, 0.4], dtype=np.float32),
            "ev": np.array([0.5, 0.6], dtype=np.float32),
        }
        for method in QA_DGST_METHODS
    }
    position = {
        "dgst": dgst,
        "ads_score": 0.7,
        "ads_per_layer": np.array([0.8, 0.9], dtype=np.float32),
        "cgc_score": 0.2,
        "cgc_per_layer": np.array([0.3, 0.4], dtype=np.float32),
    }
    return {
        "key": "row",
        "dataset": "pope",
        "source_split": "random",
        "image_id": 1,
        "probe_split": "train",
        "prediction": "yes",
        "label": 1,
        "object_hallucination_yes_only_label": 1,
        "positions": {
            "prompt_last_token": position,
            "question_object_pre_token": position,
        },
    }


def test_legacy_and_current_position_names_remain_distinct():
    assert "risk_geo@object" in legacy_feature_sets("pope")
    assert feature_set_position("risk_geo@object") == "legacy_object_target"


def test_feature_vectors_have_expected_blocks():
    row = _row()
    assert feature_vector(row, "ads").shape == (3,)
    assert feature_vector(row, "token_uncertainty").shape == (3,)
    assert feature_vector(row, "risk_geo+hprecosine@answer").shape == (4,)
    assert feature_vector(row, "best_dgst_legacy:risk_geo+hprecosine@answer").size > 4
    current = _current_row()
    assert feature_vector(current, "ads@prompt_last_token").shape == (3,)
    assert feature_vector(current, "ads+cgc@question_object_pre_token").shape == (6,)
    assert feature_vector(
        current,
        "hpre_softmax_prob_gauss_risk+"
        "hpre_softmax_prob_gauss_target_cosine+"
        "hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine@prompt_last_token",
    ).shape == (6,)


def test_yes_only_protocol_filters_non_yes_rows():
    real = _current_row()
    hallucination = {
        **_current_row(),
        "key": "hallucination",
        "image_id": 2,
        "label": 0,
        "object_hallucination_yes_only_label": 0,
    }
    excluded = {
        **_current_row(),
        "key": "excluded",
        "image_id": 3,
        "prediction": "no",
        "object_hallucination_yes_only_label": None,
    }
    features, labels, kept = build_matrix(
        [real, hallucination, excluded],
        "ads@prompt_last_token",
        "object_hallucination_yes_only",
    )
    assert features.shape == (2, 3)
    assert labels.tolist() == [1, 0]
    assert [row["key"] for row in kept] == ["row", "hallucination"]


def test_pope_image_split_leakage_is_rejected_across_strategies():
    rows = [
        {"key": "train", "dataset": "pope", "source_split": "random", "image_id": 1, "probe_split": "train"},
        {"key": "test", "dataset": "pope", "source_split": "adversarial", "image_id": 1, "probe_split": "test"},
    ]
    try:
        validate_image_level_splits(rows)
    except ValueError as exc:
        assert "image leakage" in str(exc)
    else:
        raise AssertionError("expected POPE image leakage to be rejected")


def test_clevr_official_source_split_is_part_of_image_identity():
    rows = [
        {"key": "train-1", "dataset": "clevr_exist_5k", "source_split": "train", "image_id": 1, "probe_split": "train"},
        {"key": "train-2", "dataset": "clevr_exist_5k", "source_split": "train", "image_id": 2, "probe_split": "train"},
        {"key": "train-3", "dataset": "clevr_exist_5k", "source_split": "train", "image_id": 3, "probe_split": "train"},
        {"key": "train-4", "dataset": "clevr_exist_5k", "source_split": "train", "image_id": 4, "probe_split": "train"},
        {"key": "test", "dataset": "clevr_exist_5k", "source_split": "val", "image_id": 2, "probe_split": "test"},
    ]
    assert validate_image_level_splits(rows) == {"train": 4, "val": 0, "test": 1}


def test_amber_strict_82_is_over_images_not_variable_question_rows():
    rows = []
    for image_id in range(1, 5):
        for question_id in range(image_id):
            rows.append({
                "key": f"train-{image_id}-{question_id}",
                "dataset": "amber_discriminative",
                "source_split": "attribute",
                "image_id": image_id,
                "probe_split": "train",
            })
    rows.append({
        "key": "test-5-0",
        "dataset": "amber_discriminative",
        "source_split": "existence",
        "image_id": 5,
        "probe_split": "test",
    })
    # Question rows are 10:1, but the authoritative physical-image split is 4:1.
    assert validate_image_level_splits(rows) == {"train": 4, "val": 0, "test": 1}


def test_metrics_use_real_as_positive_class():
    result = classification_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), 0.5)
    assert result["accuracy"] == 1.0
    assert result["real"]["f1"] == 1.0
    assert result["hallucination"]["f1"] == 1.0


def test_fast_real_f1_threshold_matches_brute_force_with_ties():
    labels = np.asarray([0, 1, 1, 0, 1, 0], dtype=np.int64)
    scores = np.asarray([0.2, 0.2, 0.7, 0.9, 0.7, 0.1], dtype=np.float64)
    candidates = np.unique(np.concatenate(([0.0], scores, [1.0])))
    best = (-1.0, -1.0, -np.inf, 0.5)
    for threshold in candidates:
        predicted = (scores >= threshold).astype(np.int64)
        true_positive = int(((predicted == 1) & (labels == 1)).sum())
        false_positive = int(((predicted == 1) & (labels == 0)).sum())
        false_negative = int(((predicted == 0) & (labels == 1)).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = 0.0 if denominator == 0 else 2 * true_positive / denominator
        accuracy = float((predicted == labels).mean())
        key = (f1, accuracy, -abs(float(threshold) - 0.5), float(threshold))
        if key[:3] > best[:3]:
            best = key
    threshold, f1 = choose_real_f1_threshold(labels, scores)
    assert threshold == best[3]
    assert f1 == best[0]


def test_weighted_torch_probe_smoke(tmp_path):
    rows = []
    for split, count in (("train", 16), ("test", 4)):
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
                "prediction": "yes",
                "object_hallucination_yes_only_label": label,
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
            "epochs": 99,
            "num_epochs": 3,
            "learning_rate": 0.01,
        },
        device="cpu",
        label_protocol="object_hallucination_yes_only",
    )
    assert 0.0 <= result["threshold"] <= 1.0
    assert result["positive_class"] == "real"
    assert result["label_protocol"] == "object_hallucination_yes_only"
    assert result["position"] == "shared"
    assert (tmp_path / "checkpoint.pt").exists()
    assert result["epochs_completed"] == 3
    assert result["best_epoch"] == 3
    assert result["val_metrics"] is None
    assert result["checkpoint_selection"] == "last_epoch"
    assert result["threshold_selection"] == "train_f1"
    assert result["train_metrics"]["real"]["f1"] >= 0.0
    summary = aggregate_seed_results([
        result,
        {**result, "seed": 43},
        {**result, "seed": 44},
    ])
    summary_path = tmp_path / "summary.md"
    write_markdown_summary(
        summary_path, "object_hallucination_yes_only", {"token_uncertainty": summary}
    )
    report = summary_path.read_text(encoding="utf-8")
    assert "Real F1" in report and "Hall. F1" in report and "mean ±" in report

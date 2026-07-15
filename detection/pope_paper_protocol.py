"""ADS+CGC paper-compatible POPE evaluation on Yes-response samples."""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier


FEATURE_SETS = ("ads", "object_cgc", "ads+object_cgc")


def select_paper_samples(rows: Sequence[dict]) -> list[dict]:
    """Paper protocol keeps model Yes responses; hallucination is positive."""
    selected = []
    for row in rows:
        if row.get("prediction") != "yes":
            continue
        if row.get("object_cgc_per_layer") is None:
            continue
        copied = dict(row)
        copied["paper_label"] = 1 if int(row["label"]) == 0 else 0
        selected.append(copied)
    return selected


def feature_vector(row: dict, feature_set: str) -> np.ndarray:
    ads = np.asarray(row["ads_per_layer"], dtype=np.float32).reshape(-1)
    cgc = np.asarray(row["object_cgc_per_layer"], dtype=np.float32).reshape(-1)
    if feature_set == "ads":
        return ads
    if feature_set == "object_cgc":
        return cgc
    if feature_set == "ads+object_cgc":
        return np.concatenate((ads, cgc))
    raise ValueError(feature_set)


def run_five_fold(
    rows: Sequence[dict],
    feature_set: str,
    classifier_name: str,
    seed: int = 42,
) -> dict:
    selected = select_paper_samples(rows)
    X = np.stack([feature_vector(row, feature_set) for row in selected])
    y = np.asarray([row["paper_label"] for row in selected], dtype=np.int64)
    if min(np.bincount(y, minlength=2)) < 5:
        raise ValueError(f"Need at least five samples per class, got {np.bincount(y, minlength=2)}")
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_results = []
    oof_probability = np.zeros(len(y), dtype=np.float64)
    oof_prediction = np.zeros(len(y), dtype=np.int64)
    for fold, (train_index, test_index) in enumerate(splitter.split(X, y), 1):
        mean = X[train_index].mean(axis=0)
        std = X[train_index].std(axis=0)
        std[std < 1e-6] = 1.0
        train_x = (X[train_index] - mean) / std
        test_x = (X[test_index] - mean) / std
        classifier = build_classifier(classifier_name, seed + fold)
        classifier.fit(train_x, y[train_index])
        probability = classifier.predict_proba(test_x)[:, 1]
        prediction = (probability >= 0.5).astype(np.int64)
        oof_probability[test_index] = probability
        oof_prediction[test_index] = prediction
        fold_results.append({
            "fold": fold,
            "hallucination_f1": float(f1_score(y[test_index], prediction, zero_division=0)),
            "auc": float(roc_auc_score(y[test_index], probability)),
            "test_size": int(len(test_index)),
        })
    return {
        "protocol": "ADS+CGC paper-compatible POPE Yes-response 5-fold",
        "feature_set": feature_set,
        "classifier": classifier_name,
        "positive_class": "hallucination",
        "seed": seed,
        "num_samples": int(len(y)),
        "num_real": int((y == 0).sum()),
        "num_hallucination": int((y == 1).sum()),
        "hallucination_ratio": float(y.mean()),
        "folds": fold_results,
        "mean_std": {
            metric: {
                "mean": float(np.mean([row[metric] for row in fold_results])),
                "std": float(np.std([row[metric] for row in fold_results])),
            }
            for metric in ("hallucination_f1", "auc")
        },
        "oof": {
            "hallucination_f1": float(f1_score(y, oof_prediction, zero_division=0)),
            "auc": float(roc_auc_score(y, oof_probability)),
        },
    }


def build_classifier(name: str, seed: int):
    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(128,),
            learning_rate_init=0.001,
            solver="adam",
            max_iter=500,
            random_state=seed,
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(
            max_depth=6,
            learning_rate=0.05,
            n_estimators=500,
            random_state=seed,
            eval_metric="logloss",
            n_jobs=-1,
        )
    raise ValueError(name)

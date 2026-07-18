"""Train XGB, RF, and MLP classifiers on DGST-T features."""

from __future__ import annotations
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

from utils.io_utils import load_pkl, save_pkl

LABEL_HALLUCINATED = 0
LABEL_REAL = 1
POSITIVE_LABEL = LABEL_REAL
POSITIVE_CLASS_NAME = "real"


def build_feature_matrix(
    features: List[dict],
    label_key: str = "label",
    layer_range: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """Convert a list of per-token feature dicts into (X, y) arrays."""
    valid = [f for f in features if f.get(label_key) in (0, 1)]

    if not valid:
        n_cols = 0
        for f in features:
            if "dgst_t_feature_vector" in f:
                n_cols = len(f["dgst_t_feature_vector"])
                break
            if "dgst_t_per_layer" in f:
                n_layers = len(f["dgst_t_per_layer"])
                n_cols = n_layers
                break
        X = np.empty((0, n_cols), dtype=np.float32)
        y = np.empty((0,), dtype=np.int32)
        return X, y, []

    X_rows, y_rows, meta = [], [], []
    for f in valid:
        if "dgst_t_feature_vector" in f:
            feature_full = np.array(f["dgst_t_feature_vector"], dtype=np.float32)
            n_layers = len(f.get("dgst_t_per_layer", []))
        else:
            feature_full = np.array(f["dgst_t_per_layer"], dtype=np.float32)
            n_layers = len(feature_full)

        if layer_range is not None:
            L = n_layers or len(feature_full)
            start = max(0, int(L * layer_range[0]))
            end   = min(L, int(L * layer_range[1]))
            if n_layers and len(feature_full) % n_layers == 0:
                blocks = []
                for block_start in range(0, len(feature_full), n_layers):
                    blocks.append(feature_full[block_start + start:block_start + end])
                feature_full = np.concatenate(blocks)
            else:
                feature_full = feature_full[start:end]

        x = feature_full

        X_rows.append(x)
        y_rows.append(int(f[label_key]))
        meta.append(f)

    if not X_rows:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            [],
        )
    X = np.stack(X_rows, axis=0)
    y = np.array(y_rows, dtype=np.int32)
    return X, y, meta


def split_by_image_id(
    features: List[dict],
    train_image_ids: set,
    val_image_ids: set,
    test_image_ids: Optional[set] = None,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Split feature list by image id to prevent leakage."""
    if test_image_ids is None:
        raise ValueError("Strict outer-8:2 training requires explicit test_image_ids.")
    overlaps = {
        "train/val": set(train_image_ids) & set(val_image_ids),
        "train/test": set(train_image_ids) & set(test_image_ids),
        "val/test": set(val_image_ids) & set(test_image_ids),
    }
    bad = {name: len(values) for name, values in overlaps.items() if values}
    if bad:
        raise ValueError(f"Image-level train/val/test splits overlap: {bad}")
    train = [f for f in features if f["image_id"] in train_image_ids]
    val   = [f for f in features if f["image_id"] in val_image_ids]
    test  = [f for f in features if f["image_id"] in test_image_ids]
    return train, val, test


def build_classifier(clf_type: str, params: dict):
    """Instantiate a classifier from type string and hyperparameter dict."""
    if clf_type == "xgb":
        if XGBClassifier is not None:
            return XGBClassifier(
                max_depth=params.get("max_depth", 6),
                learning_rate=params.get("learning_rate", 0.05),
                n_estimators=params.get("n_estimators", 500),
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )
        return GradientBoostingClassifier(
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.05),
            n_estimators=params.get("n_estimators", 500),
            random_state=42,
        )
    elif clf_type == "rf":
        return RandomForestClassifier(
            max_depth=params.get("max_depth", 10),
            n_estimators=params.get("n_estimators", 400),
            random_state=42,
            n_jobs=-1,
        )
    elif clf_type == "mlp":
        hidden = params.get("hidden_layer_sizes", [128])
        if isinstance(hidden[0], list):
            hidden_tuple = tuple(hidden[0])
        else:
            hidden_tuple = tuple(hidden)
        return MLPClassifier(
            hidden_layer_sizes=hidden_tuple,
            learning_rate_init=params.get("learning_rate_init", 0.001),
            solver=params.get("solver", "adam"),
            max_iter=params.get("max_iter", 500),
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown classifier type: {clf_type}")


def grid_search(
    clf_type: str,
    param_grid: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scoring: str = "f1",
) -> Tuple[object, dict, float]:
    """Exhaustive grid search over `param_grid`, evaluated by `scoring` on val set."""
    best_clf = None
    best_params = {}
    best_score = -1.0

    if X_train.shape[0] == 0:
        raise ValueError(
            f"grid_search received empty training split: X_train={X_train.shape}."
        )
    if len(np.unique(y_train)) < 2:
        raise ValueError(
            f"Training set contains only one class {np.unique(y_train)}. "
            "Need at least one sample of each class (0=hallucinated, 1=real)."
        )

    candidates = list(ParameterGrid(param_grid))
    if not candidates:
        candidates = [{}]
    if X_val.shape[0] == 0:
        # Pure strict-8:2 protocol: hyperparameters are fixed to the first
        # declared configuration, the final estimator is fit on all 80% train
        # rows, and no score from the 20% test partition influences selection.
        # Only the decision threshold is selected by training-set F1.
        params = candidates[0]
        clf = build_classifier(clf_type, params)
        clf.fit(X_train, y_train)
        train_scores = _positive_class_scores(clf, X_train)
        threshold = select_decision_threshold(
            y_train,
            train_scores,
            scoring="f1",
        )
        train_predictions = np.where(
            train_scores >= threshold,
            POSITIVE_LABEL,
            1 - POSITIVE_LABEL,
        )
        train_score = f1_score(
            y_train,
            train_predictions,
            pos_label=POSITIVE_LABEL,
            zero_division=0,
        )
        setattr(clf, "_token_detector_threshold", float(threshold))
        setattr(
            clf,
            "_token_detector_selection_protocol",
            "fixed_hyperparameters_train_f1_threshold",
        )
        return clf, params, float(train_score)

    for params in candidates:
        clf = build_classifier(clf_type, params)
        clf.fit(X_train, y_train)
        threshold = None
        try:
            y_prob = _positive_class_scores(clf, X_val)
            threshold = select_decision_threshold(
                y_val,
                y_prob,
                scoring="accuracy" if scoring == "accuracy" else "f1",
            )
            y_pred = np.where(
                y_prob >= threshold,
                POSITIVE_LABEL,
                1 - POSITIVE_LABEL,
            )
        except Exception:
            y_prob = None
            y_pred = clf.predict(X_val)

        if scoring == "f1":
            score = f1_score(
                y_val,
                y_pred,
                pos_label=POSITIVE_LABEL,
                zero_division=0,
            )
        elif scoring == "accuracy":
            score = accuracy_score(y_val, y_pred)
        elif scoring == "auc":
            try:
                if y_prob is None:
                    y_prob = _positive_class_scores(clf, X_val)
                score = roc_auc_score(_positive_class_targets(y_val), y_prob)
            except Exception:
                score = -1.0
        else:
            raise ValueError(f"Unknown scoring: {scoring}")

        if score > best_score:
            best_score = score
            best_params = params
            best_clf = clf
            if threshold is not None:
                setattr(best_clf, "_token_detector_threshold", float(threshold))

    return best_clf, best_params, best_score


def evaluate_classifier(
    clf,
    X: np.ndarray,
    y: np.ndarray,
) -> Dict[str, float]:
    """Return precision, recall, F1, accuracy, and AUC."""
    try:
        y_prob = _positive_class_scores(clf, X)
        threshold = float(getattr(clf, "_token_detector_threshold", 0.5))
        y_pred = np.where(
            y_prob >= threshold,
            POSITIVE_LABEL,
            1 - POSITIVE_LABEL,
        )
        auc = roc_auc_score(_positive_class_targets(y), y_prob)
    except Exception:
        threshold = None
        y_pred = clf.predict(X)
        auc = float("nan")

    return {
        "precision": precision_score(
            y, y_pred, pos_label=POSITIVE_LABEL, zero_division=0
        ),
        "recall":    recall_score(
            y, y_pred, pos_label=POSITIVE_LABEL, zero_division=0
        ),
        "f1":        f1_score(
            y, y_pred, pos_label=POSITIVE_LABEL, zero_division=0
        ),
        "accuracy":  accuracy_score(y, y_pred),
        "auc":       auc,
        "decision_threshold": threshold,
        "reported_positive_class": POSITIVE_CLASS_NAME,
    }


def select_decision_threshold(
    y_true: np.ndarray,
    positive_scores: np.ndarray,
    *,
    scoring: str = "f1",
) -> float:
    """Select a binary decision threshold on the caller-provided fit rows."""
    labels = np.asarray(y_true, dtype=np.int32).reshape(-1)
    scores = np.asarray(positive_scores, dtype=np.float64).reshape(-1)
    if labels.size != scores.size or labels.size == 0:
        raise ValueError("Threshold selection requires equally sized non-empty arrays.")
    targets = _positive_class_targets(labels)
    candidates = np.unique(np.concatenate(([0.0], scores, [1.0])))
    best_threshold = 0.5
    best_key = (-np.inf, -np.inf, -np.inf)
    for threshold in candidates:
        prediction = (scores >= threshold).astype(np.int32)
        if scoring == "accuracy":
            primary = float(accuracy_score(targets, prediction))
            secondary = float(f1_score(targets, prediction, zero_division=0))
        elif scoring == "f1":
            primary = float(f1_score(targets, prediction, zero_division=0))
            secondary = float(accuracy_score(targets, prediction))
        else:
            raise ValueError("Threshold scoring must be 'f1' or 'accuracy'.")
        key = (primary, secondary, -abs(float(threshold) - 0.5))
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def train_and_evaluate(
    feature_path: str,
    train_image_ids: set,
    val_image_ids: set,
    test_image_ids: set,
    clf_configs: dict,
    output_dir: str,
    model_key: str = "model",
) -> Dict[str, dict]:
    """Full train + evaluate pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    all_features = load_pkl(feature_path)
    print(f"[Train] Loaded {len(all_features)} token features from {feature_path}")

    train_feats, val_feats, test_feats = split_by_image_id(
        all_features, train_image_ids, val_image_ids, test_image_ids
    )
    print(
        f"[Train] Split: train={len(train_feats)}, "
        f"val={len(val_feats)}, test={len(test_feats)} tokens"
    )

    X_train, y_train, _ = build_feature_matrix(train_feats)
    X_val,   y_val,   _ = build_feature_matrix(val_feats)
    X_test,  y_test,  _ = build_feature_matrix(test_feats)

    print(
        f"[Train] Feature matrices — "
        f"train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}"
    )
    print(
        f"[Train] Label balance — "
        f"train: {np.mean(y_train == LABEL_HALLUCINATED):.2%} hallucinated, "
        f"test: {np.mean(y_test == LABEL_HALLUCINATED):.2%} hallucinated"
    )

    if X_train.shape[0] == 0:
        raise ValueError(
            "[Train] train split is empty after filtering to labeled tokens. "
            "Run step2 with more images, or check labeling.json for missing spans."
        )
    if len(np.unique(y_train)) < 2:
        raise ValueError(
            f"[Train] train split has only one class {np.unique(y_train)}. "
            "Need both class 0 (hallucinated) and class 1 (real) tokens."
        )
    for split_name, matrix, labels in (
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ):
        if split_name == "val" and matrix.shape[0] == 0:
            continue
        classes = np.unique(labels)
        if matrix.shape[0] == 0 or classes.size < 2:
            raise ValueError(
                f"[Train] strict outer-8:2 training violation: {split_name} has "
                f"{matrix.shape[0]} rows and classes {classes.tolist()}. "
                "Splits are never substituted; repair labeling/image_splits.json."
            )

    all_results = {}

    print("\n[Train] Grid-searching XGBoost …")
    xgb_grid = clf_configs.get("xgb", {})
    _sanitise_grid(xgb_grid)
    best_xgb, xgb_params, xgb_val_f1 = grid_search(
        "xgb", xgb_grid, X_train, y_train, X_val, y_val
    )
    print(f"  Best XGB params: {xgb_params}  (val F1={xgb_val_f1:.4f})")
    xgb_metrics = evaluate_classifier(best_xgb, X_test, y_test)
    xgb_metrics["best_params"] = xgb_params
    all_results["xgb"] = xgb_metrics
    save_pkl(best_xgb, os.path.join(output_dir, f"{model_key}_xgb.pkl"))

    print("\n[Train] Grid-searching Random Forest …")
    rf_grid = clf_configs.get("rf", {})
    _sanitise_grid(rf_grid)
    best_rf, rf_params, rf_val_f1 = grid_search(
        "rf", rf_grid, X_train, y_train, X_val, y_val
    )
    print(f"  Best RF params: {rf_params}  (val F1={rf_val_f1:.4f})")
    rf_metrics = evaluate_classifier(best_rf, X_test, y_test)
    rf_metrics["best_params"] = rf_params
    all_results["rf"] = rf_metrics
    save_pkl(best_rf, os.path.join(output_dir, f"{model_key}_rf.pkl"))

    print("\n[Train] Grid-searching MLP …")
    mlp_grid = clf_configs.get("mlp", {})
    _sanitise_grid(mlp_grid)
    best_mlp, mlp_params, mlp_val_f1 = grid_search(
        "mlp", mlp_grid, X_train, y_train, X_val, y_val
    )
    print(f"  Best MLP params: {mlp_params}  (val F1={mlp_val_f1:.4f})")
    mlp_metrics = evaluate_classifier(best_mlp, X_test, y_test)
    mlp_metrics["best_params"] = mlp_params
    all_results["mlp"] = mlp_metrics
    save_pkl(best_mlp, os.path.join(output_dir, f"{model_key}_mlp.pkl"))

    print("\n" + "=" * 60)
    print(f"  Results for {model_key}")
    print("=" * 60)
    print(f"{'Method':<10} {'PR':>6} {'RC':>6} {'F1':>6} {'ACC':>6} {'AUC':>6}")
    print("-" * 60)
    for clf_name, m in all_results.items():
        print(
            f"{clf_name.upper():<10} "
            f"{m['precision']:>6.3f} "
            f"{m['recall']:>6.3f} "
            f"{m['f1']:>6.3f} "
            f"{m['accuracy']:>6.3f} "
            f"{m['auc']:>6.3f}"
        )

    results_path = os.path.join(output_dir, f"{model_key}_results.pkl")
    save_pkl(all_results, results_path)
    print(f"\n[Train] Results saved to {results_path}")

    return all_results


def _positive_class_targets(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int32)
    return (labels == POSITIVE_LABEL).astype(np.int32)


def _positive_class_scores(clf, X: np.ndarray) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X)
        classes = list(getattr(clf, "classes_", []))
        if POSITIVE_LABEL in classes:
            return probs[:, classes.index(POSITIVE_LABEL)]
        return 1.0 - probs[:, -1]
    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(X)
        classes = list(getattr(clf, "classes_", []))
        if POSITIVE_LABEL in classes and classes.index(POSITIVE_LABEL) == 0:
            return -np.asarray(scores, dtype=np.float32)
        return np.asarray(scores, dtype=np.float32)
    return (clf.predict(X) == POSITIVE_LABEL).astype(np.float32)


def _sanitise_grid(grid: dict) -> None:
    """sklearn ParameterGrid requires lists, not single values."""
    for k, v in grid.items():
        if not isinstance(v, list):
            grid[k] = [v]

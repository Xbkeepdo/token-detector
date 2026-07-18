"""Classifier cores and hallucination-positive metrics for paper baselines."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from features.baseline.schema import baseline_vector


LABEL_HALLUCINATION = 0
LABEL_REAL = 1
DETECTOR_REAL = 0
DETECTOR_HALLUCINATION = 1


def raw_labels_to_hallucination_targets(labels: Sequence[int]) -> np.ndarray:
    """Convert stored ``0=hall,1=real`` labels to detector ``1=hall`` targets."""

    raw = np.asarray(labels, dtype=np.int64).reshape(-1)
    if raw.size and not np.isin(raw, (LABEL_HALLUCINATION, LABEL_REAL)).all():
        raise ValueError("Raw labels must use 0=hallucination and 1=real")
    return (raw == LABEL_HALLUCINATION).astype(np.int64)


def build_dense_baseline_matrix(
    records: Sequence[Mapping[str, Any]],
    method: str,
) -> tuple[np.ndarray, np.ndarray, list[Mapping[str, Any]]]:
    vectors: list[np.ndarray] = []
    labels: list[int] = []
    kept: list[Mapping[str, Any]] = []
    for record in records:
        if record.get("label") not in (0, 1):
            continue
        try:
            vector = baseline_vector(record, method)
        except (KeyError, TypeError, ValueError):
            continue
        vectors.append(vector)
        labels.append(int(record["label"]))
        kept.append(record)
    if not vectors:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            [],
        )
    widths = {vector.size for vector in vectors}
    if len(widths) != 1:
        raise ValueError(f"Inconsistent {method} vector widths: {sorted(widths)}")
    return (
        np.stack(vectors).astype(np.float32, copy=False),
        np.asarray(labels, dtype=np.int64),
        kept,
    )


def split_records_by_image(
    records: Sequence[Mapping[str, Any]],
    splits: Mapping[str, Sequence[int]],
) -> dict[str, list[Mapping[str, Any]]]:
    split_sets = {
        name: {int(image_id) for image_id in splits[name]}
        for name in ("train", "val", "test")
    }
    if split_sets["train"] & split_sets["val"]:
        raise ValueError("train and val image splits overlap")
    if split_sets["train"] & split_sets["test"]:
        raise ValueError("train and test image splits overlap")
    if split_sets["val"] & split_sets["test"]:
        raise ValueError("val and test image splits overlap")
    return {
        name: [
            record
            for record in records
            if int(record["image_id"]) in split_sets[name]
        ]
        for name in ("train", "val", "test")
    }


def build_metatoken_classifier(kind: str, *, seed: int = 42):
    """Build the two MetaToken classifiers specified by the comparison paper."""

    normalized = str(kind).strip().lower()
    if normalized in {"lr", "logistic", "logistic_regression"}:
        classifier = LogisticRegression(
            solver="lbfgs", max_iter=2000, random_state=int(seed)
        )
    elif normalized in {"gb", "gradient_boosting"}:
        classifier = GradientBoostingClassifier(
            n_estimators=100, random_state=int(seed)
        )
    else:
        raise ValueError("MetaToken classifier must be 'lr' or 'gb'")
    # MetaToken standardizes all input features before meta classification.
    return Pipeline((("standardize", StandardScaler()), ("classifier", classifier)))


class SVARMLP(nn.Module):
    """One-hidden-layer detector; the paper comparison fixes hidden dim 248."""

    def __init__(self, input_dim: int, hidden_dim: int = 248) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), 2),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value.reshape(value.shape[0], -1).float())


class DHCPMLP(nn.Module):
    """Official DHCP ``Linear(D,128)-ReLU-Linear(128,2)`` detector."""

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), 2),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value.reshape(value.shape[0], -1).float())


@dataclass
class TorchDetectorResult:
    state_dict: dict[str, torch.Tensor]
    history: list[dict[str, float]]
    threshold: float
    val_metrics: dict[str, Any]
    test_metrics: dict[str, Any]


def train_torch_detector(
    *,
    model: nn.Module,
    X_train: np.ndarray,
    raw_y_train: np.ndarray,
    X_val: np.ndarray,
    raw_y_val: np.ndarray,
    X_test: np.ndarray,
    raw_y_test: np.ndarray,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    device: str = "cpu",
    weight_decay: float = 0.0,
    weighted_sampler: bool = False,
    standardize: bool = False,
    early_stopping_patience: int = 5,
    seed: int = 42,
    positive_class: str = "hallucination",
    strict_82_no_validation: bool = False,
) -> TorchDetectorResult:
    """Train a detector under validation or pure strict-8:2 protocol."""

    _validate_train_splits(X_train, raw_y_train, X_val, raw_y_val, X_test, raw_y_test)
    _seed_everything(seed)
    train = np.asarray(X_train, dtype=np.float32)
    val = np.asarray(X_val, dtype=np.float32)
    test = np.asarray(X_test, dtype=np.float32)
    if standardize:
        mean = train.mean(axis=0, keepdims=True)
        std = train.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        train, val, test = (train - mean) / std, (val - mean) / std, (test - mean) / std

    y_train = raw_labels_to_hallucination_targets(raw_y_train)
    train_dataset = TensorDataset(
        torch.from_numpy(train), torch.from_numpy(y_train)
    )
    sampler = None
    shuffle = True
    if weighted_sampler:
        counts = np.bincount(y_train, minlength=2)
        weights = 1.0 / np.maximum(counts, 1)
        sample_weights = torch.tensor(weights[y_train], dtype=torch.double)
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=torch.Generator().manual_seed(int(seed)),
        )
        shuffle = False
    loader = DataLoader(
        train_dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        sampler=sampler,
        generator=torch.Generator().manual_seed(int(seed)),
    )

    chosen_device = torch.device(device)
    model = model.to(chosen_device)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    criterion = nn.CrossEntropyLoss()
    val_x = torch.from_numpy(val).to(chosen_device)
    val_y = torch.from_numpy(
        raw_labels_to_hallucination_targets(raw_y_val)
    ).to(chosen_device)
    best_loss = math.inf
    best_state: Optional[dict[str, torch.Tensor]] = None
    history: list[dict[str, float]] = []
    stale_epochs = 0
    for epoch in range(int(epochs)):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(chosen_device)
            batch_y = batch_y.to(chosen_device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(val_x), val_y).item())
        epoch_row = {
            "epoch": float(epoch + 1),
            "train_loss": float(np.mean(losses)),
        }
        epoch_row[
            "train_monitor_loss" if strict_82_no_validation else "val_loss"
        ] = val_loss
        history.append(epoch_row)
        if strict_82_no_validation:
            best_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        elif val_loss < best_loss:
            best_loss = val_loss
            stale_epochs = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
            if stale_epochs >= max(1, int(early_stopping_patience)):
                break
    if best_state is None:
        raise RuntimeError("No detector checkpoint was selected")
    model.load_state_dict(best_state)
    val_scores = torch_hallucination_scores(model, val, chosen_device)
    test_scores = torch_hallucination_scores(model, test, chosen_device)
    threshold = select_detection_threshold(
        raw_y_val,
        val_scores,
        positive_class=positive_class,
    )
    return TorchDetectorResult(
        state_dict=best_state,
        history=history,
        threshold=threshold,
        val_metrics=evaluate_detection_scores(
            raw_y_val,
            val_scores,
            threshold,
            positive_class=positive_class,
        ),
        test_metrics=evaluate_detection_scores(
            raw_y_test,
            test_scores,
            threshold,
            positive_class=positive_class,
        ),
    )


@torch.no_grad()
def torch_hallucination_scores(
    model: nn.Module,
    matrix: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = 1024,
) -> np.ndarray:
    model.eval()
    scores = []
    tensor = torch.from_numpy(np.asarray(matrix, dtype=np.float32))
    for start in range(0, len(tensor), int(batch_size)):
        logits = model(tensor[start : start + int(batch_size)].to(device))
        scores.append(torch.softmax(logits, dim=-1)[:, DETECTOR_HALLUCINATION].cpu())
    return torch.cat(scores).numpy().astype(np.float64, copy=False)


def sklearn_hallucination_scores(classifier, matrix: np.ndarray) -> np.ndarray:
    probabilities = classifier.predict_proba(matrix)
    classes = list(classifier.classes_)
    if DETECTOR_HALLUCINATION not in classes:
        raise ValueError("Classifier does not expose hallucination-positive class 1")
    return probabilities[:, classes.index(DETECTOR_HALLUCINATION)].astype(np.float64)


def select_hallucination_threshold(
    raw_labels: Sequence[int],
    hallucination_scores: Sequence[float],
) -> float:
    targets = raw_labels_to_hallucination_targets(raw_labels)
    scores = np.asarray(hallucination_scores, dtype=np.float64).reshape(-1)
    if targets.size != scores.size or targets.size == 0:
        raise ValueError("Validation labels/scores must be non-empty and aligned")
    if np.unique(targets).size < 2:
        raise ValueError("Validation split must contain hallucinated and real samples")
    precision, recall, thresholds = precision_recall_curve(targets, scores)
    if thresholds.size == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    return float(thresholds[int(np.nanargmax(f1))])


def select_detection_threshold(
    raw_labels: Sequence[int],
    hallucination_scores: Sequence[float],
    *,
    positive_class: str,
) -> float:
    """Select a validation-F1 threshold in the requested score direction."""

    positive = _normalize_positive_class(positive_class)
    if positive == "hallucination":
        return select_hallucination_threshold(raw_labels, hallucination_scores)
    real_targets = np.asarray(raw_labels, dtype=np.int64).reshape(-1)
    hall_scores = np.asarray(hallucination_scores, dtype=np.float64).reshape(-1)
    return _select_binary_f1_threshold(real_targets, 1.0 - hall_scores)


def _select_binary_f1_threshold(
    targets: Sequence[int],
    scores: Sequence[float],
) -> float:
    targets_array = np.asarray(targets, dtype=np.int64).reshape(-1)
    scores_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    if targets_array.size != scores_array.size or targets_array.size == 0:
        raise ValueError("Validation labels/scores must be non-empty and aligned")
    if np.unique(targets_array).size < 2:
        raise ValueError("Validation split must contain hallucinated and real samples")
    precision, recall, thresholds = precision_recall_curve(
        targets_array,
        scores_array,
    )
    if thresholds.size == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    return float(thresholds[int(np.nanargmax(f1))])


def _normalize_positive_class(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "hall": "hallucination",
        "hallucinated": "hallucination",
        "hallucination": "hallucination",
        "real": "real",
        "non_hallucination": "real",
        "nonhallucination": "real",
    }
    if normalized not in aliases:
        raise ValueError(
            "positive_class must be 'hallucination' or 'real', "
            f"got {value!r}"
        )
    return aliases[normalized]


def evaluate_hallucination_scores(
    raw_labels: Sequence[int],
    hallucination_scores: Sequence[float],
    threshold: float,
) -> dict[str, Any]:
    """Report both class directions, with hallucination as the headline class."""

    hall_targets = raw_labels_to_hallucination_targets(raw_labels)
    scores = np.asarray(hallucination_scores, dtype=np.float64).reshape(-1)
    if hall_targets.size != scores.size or scores.size == 0:
        raise ValueError("Evaluation labels/scores must be non-empty and aligned")
    if not np.isfinite(scores).all():
        raise ValueError("Evaluation scores contain non-finite values")
    hall_predictions = (scores >= float(threshold)).astype(np.int64)
    real_targets = 1 - hall_targets
    real_predictions = 1 - hall_predictions
    result = {
        "threshold": float(threshold),
        "threshold_score_class": "hallucination",
        "headline_positive_class": "hallucination",
        "stored_label_semantics": {"0": "hallucination", "1": "real"},
        "detector_target_semantics": {"0": "real", "1": "hallucination"},
        "accuracy": float(accuracy_score(hall_targets, hall_predictions)),
        "confusion_matrix_hallucination_positive": confusion_matrix(
            hall_targets, hall_predictions, labels=[0, 1]
        ).tolist(),
        "hallucination_positive": _class_metrics(
            hall_targets, hall_predictions, scores
        ),
        "real_positive": _class_metrics(
            real_targets, real_predictions, 1.0 - scores
        ),
    }
    return result


def evaluate_detection_scores(
    raw_labels: Sequence[int],
    hallucination_scores: Sequence[float],
    threshold: float,
    *,
    positive_class: str,
) -> dict[str, Any]:
    """Evaluate both class directions using a threshold for the headline class."""

    positive = _normalize_positive_class(positive_class)
    if positive == "hallucination":
        return evaluate_hallucination_scores(
            raw_labels,
            hallucination_scores,
            threshold,
        )

    hall_targets = raw_labels_to_hallucination_targets(raw_labels)
    scores = np.asarray(hallucination_scores, dtype=np.float64).reshape(-1)
    if hall_targets.size != scores.size or scores.size == 0:
        raise ValueError("Evaluation labels/scores must be non-empty and aligned")
    if not np.isfinite(scores).all():
        raise ValueError("Evaluation scores contain non-finite values")
    real_targets = 1 - hall_targets
    real_scores = 1.0 - scores
    real_predictions = (real_scores >= float(threshold)).astype(np.int64)
    hall_predictions = 1 - real_predictions
    return {
        "threshold": float(threshold),
        "threshold_score_class": "real",
        "headline_positive_class": "real",
        "stored_label_semantics": {"0": "hallucination", "1": "real"},
        "detector_target_semantics": {"0": "real", "1": "hallucination"},
        "accuracy": float(accuracy_score(real_targets, real_predictions)),
        "confusion_matrix_hallucination_positive": confusion_matrix(
            hall_targets, hall_predictions, labels=[0, 1]
        ).tolist(),
        "hallucination_positive": _class_metrics(
            hall_targets,
            hall_predictions,
            scores,
        ),
        "real_positive": _class_metrics(
            real_targets,
            real_predictions,
            real_scores,
        ),
    }


def _class_metrics(
    targets: np.ndarray, predictions: np.ndarray, scores: np.ndarray
) -> dict[str, float]:
    try:
        auc = float(roc_auc_score(targets, scores))
        aupr = float(average_precision_score(targets, scores))
    except ValueError:
        auc = float("nan")
        aupr = float("nan")
    return {
        "precision": float(precision_score(targets, predictions, zero_division=0)),
        "recall": float(recall_score(targets, predictions, zero_division=0)),
        "f1": float(f1_score(targets, predictions, zero_division=0)),
        "auc": auc,
        "aupr": aupr,
    }


def _validate_train_splits(*values) -> None:
    for split, matrix, labels in zip(
        ("train", "val", "test"), values[::2], values[1::2]
    ):
        matrix = np.asarray(matrix)
        labels = np.asarray(labels)
        if matrix.shape[0] == 0 or matrix.shape[0] != labels.shape[0]:
            raise ValueError(f"{split} matrix/labels are empty or misaligned")
        if not np.isfinite(matrix).all():
            raise ValueError(f"{split} matrix contains non-finite values")
        if np.unique(labels).size < 2:
            raise ValueError(f"{split} split must contain both raw label classes")


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

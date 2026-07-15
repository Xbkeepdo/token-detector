#!/usr/bin/env python3
"""Train a DGST-style PyTorch probe on selected DGST-T feature blocks."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detection.train import split_by_image_id
from summarize_feature_set_results import write_summary_tables
from train_feature_sets import (
    _require_strict_binary_splits,
    build_selected_matrix,
    parse_feature_set,
)
from utils.io_utils import load_json, load_pkl, save_json


DEFAULT_FEATURE_SETS = [
    "risk",
    "target_cosine",
    "risk+target_cosine",
]


@dataclass
class TorchProbeConfig:
    hidden_sizes: tuple[int, ...] = (128, 64, 32)
    dropout: float = 0.3
    batch_size: int = 256
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    lr_factor: float = 0.5
    lr_patience: int = 5
    early_stopping_patience: int = 10
    seed: int = 42
    positive_class: str = "real"


class MatrixDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        if X.shape[0] == 0:
            raise ValueError("MatrixDataset requires at least one row.")
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]


class DGSTStyleProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Sequence[int], dropout: float):
        super().__init__()
        layers = []
        prev_dim = int(input_dim)
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, int(hidden_dim)))
            layers.append(nn.BatchNorm1d(int(hidden_dim)))
            layers.append(nn.LeakyReLU(negative_slope=0.01))
            layers.append(nn.Dropout(float(dropout)))
            prev_dim = int(hidden_dim)
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=0.01, nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.isnan(x).any():
            raise ValueError("Torch probe input contains NaN values.")
        return self.net(x)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="configs/model_configs.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-sets", nargs="+", default=DEFAULT_FEATURE_SETS)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[128, 64, 32])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--positive-class",
        choices=["real", "non_hallucination", "hallucination"],
        default="real",
        help="Positive class for PR/RC/F1/AUC. Use real to match the SVAR code.",
    )
    parser.add_argument(
        "--paper-config",
        action="store_true",
        help="Use dgst.git paper-style probe settings: batch=32, lr=5e-4, weight_decay=0.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from utils.config_utils import load_config

    yaml_config = load_config(args.config)
    if args.paper_config:
        args.batch_size = 32
        args.learning_rate = 5e-4
        args.weight_decay = 0.0

    config = TorchProbeConfig(
        hidden_sizes=tuple(int(x) for x in args.hidden_sizes),
        dropout=float(args.dropout),
        batch_size=int(args.batch_size),
        num_epochs=int(args.num_epochs),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        lr_factor=float(args.lr_factor),
        lr_patience=int(args.lr_patience),
        early_stopping_patience=int(args.early_stopping_patience),
        seed=int(args.seed),
        positive_class=str(args.positive_class),
    )

    feature_path = os.path.join(args.output_dir, "features.pkl")
    splits_path = os.path.join(args.output_dir, "image_splits.json")
    results_dir = os.path.join(args.output_dir, "results")
    probe_dir = os.path.join(results_dir, "torch_probe")
    os.makedirs(probe_dir, exist_ok=True)

    if not os.path.exists(feature_path):
        raise FileNotFoundError(feature_path)
    if not os.path.exists(splits_path):
        raise FileNotFoundError(splits_path)

    all_features = load_pkl(feature_path)
    splits = load_json(splits_path)
    from utils.split_utils import validate_strict_811_split

    split_counts = validate_strict_811_split(splits)
    configured_count = int(
        (yaml_config.get("dataset") or {}).get("num_images", 0)
    )
    if configured_count and sum(split_counts.values()) != configured_count:
        raise ValueError(
            "Strict split size differs from dataset.num_images: "
            f"{sum(split_counts.values())} != {configured_count}"
        )
    train_feats, val_feats, test_feats = split_by_image_id(
        all_features,
        train_image_ids={int(x) for x in splits["train"]},
        val_image_ids={int(x) for x in splits["val"]},
        test_image_ids={int(x) for x in splits["test"]},
    )

    out_path = os.path.join(results_dir, f"{args.model}_selected_feature_sets.json")
    results = load_json(out_path) if os.path.exists(out_path) else {}
    device = _resolve_device(args.device)

    print(
        f"[TorchProbe] Loaded {len(all_features)} token features: "
        f"train={len(train_feats)}, val={len(val_feats)}, test={len(test_feats)}"
    )
    print(f"[TorchProbe] device={device}, config={asdict(config)}")

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

        print(f"\n[TorchProbe] {feature_set}: X={X_train.shape[1]} dims")
        artifacts_dir = os.path.join(probe_dir, _slug(feature_set))
        metrics = train_and_evaluate_probe(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            config=config,
            device=device,
            output_dir=artifacts_dir,
        )
        metrics["num_features"] = int(X_train.shape[1])
        metrics["best_params"] = {
            "hidden_sizes": list(config.hidden_sizes),
            "dropout": config.dropout,
            "batch_size": config.batch_size,
            "num_epochs": config.num_epochs,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "lr_factor": config.lr_factor,
            "lr_patience": config.lr_patience,
            "early_stopping_patience": config.early_stopping_patience,
            "seed": config.seed,
            "positive_class": config.positive_class,
        }
        metrics["artifacts"] = {
            "model": os.path.join(artifacts_dir, "model.pt"),
            "history": os.path.join(artifacts_dir, "history.json"),
            "config": os.path.join(artifacts_dir, "config.json"),
        }
        results.setdefault(feature_set, {})["torch_probe"] = _json_ready(metrics)
        save_json(results, out_path)
        print(
            f"  TORCH F1={metrics['f1']:.3f} AUC={metrics['auc']:.3f} "
            f"PR={metrics['precision']:.3f} RC={metrics['recall']:.3f} "
            f"best_epoch={metrics['best_epoch']}"
        )

    print(f"\n[TorchProbe] Saved results to {out_path}")
    _write_summary_table(out_path)


def train_and_evaluate_probe(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: TorchProbeConfig,
    device: torch.device,
    output_dir: str,
) -> dict:
    _set_seed(config.seed)
    os.makedirs(output_dir, exist_ok=True)

    train_targets = _targets_for_positive_class(y_train, config.positive_class)
    val_targets = _targets_for_positive_class(y_val, config.positive_class)
    test_targets = _targets_for_positive_class(y_test, config.positive_class)

    train_loader = DataLoader(
        MatrixDataset(X_train, train_targets),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=X_train.shape[0] > config.batch_size,
    )
    val_loader = DataLoader(MatrixDataset(X_val, val_targets), batch_size=config.batch_size)

    model = DGSTStyleProbe(
        input_dim=int(X_train.shape[1]),
        hidden_sizes=config.hidden_sizes,
        dropout=config.dropout,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_factor,
        patience=config.lr_patience,
    )

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    history = []
    model_path = os.path.join(output_dir, "model.pt")

    progress = tqdm(range(config.num_epochs), desc="Training torch probe", unit="epoch", leave=False)
    for epoch in progress:
        train_loss = _train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_probs = _predict_loss_and_probs(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        val_metrics = _metrics_from_probs(val_targets, val_probs, positive_class=config.positive_class)
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_f1": float(val_metrics["f1"]),
                "val_auc": float(val_metrics["auc"]),
            }
        )
        progress.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
            val_f1=f"{val_metrics['f1']:.4f}",
        )
        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            best_epoch = int(epoch)
            torch.save(model.state_dict(), model_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= max(
                1, int(config.early_stopping_patience)
            ):
                break

    progress.close()
    if best_epoch < 0:
        raise RuntimeError("Torch probe training did not produce a best checkpoint.")

    model.load_state_dict(torch.load(model_path, map_location=device))
    _best_val_loss, best_val_probs = _predict_loss_and_probs(
        model, val_loader, criterion, device
    )
    decision_threshold = _select_validation_threshold(
        val_targets,
        best_val_probs,
    )
    selected_val_metrics = _metrics_from_probs(
        val_targets,
        best_val_probs,
        positive_class=config.positive_class,
        threshold=decision_threshold,
    )
    test_dataset = MatrixDataset(X_test, test_targets)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    _test_loss, test_probs = _predict_loss_and_probs(model, test_loader, criterion, device)
    metrics = _metrics_from_probs(
        test_targets,
        test_probs,
        positive_class=config.positive_class,
        threshold=decision_threshold,
    )
    metrics["best_epoch"] = int(best_epoch)
    metrics["best_val_loss"] = float(best_val_loss)
    metrics["val_score"] = float(selected_val_metrics["f1"])
    metrics["val_metrics"] = selected_val_metrics
    metrics["decision_threshold"] = float(decision_threshold)
    metrics["epochs_ran"] = int(len(history))

    save_json(history, os.path.join(output_dir, "history.json"))
    save_json(asdict(config), os.path.join(output_dir, "config.json"))
    return metrics


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    losses = []
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(sum(losses) / len(losses)) if losses else 0.0


def _predict_loss_and_probs(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    model.eval()
    losses = []
    probs = []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device).unsqueeze(1)
            logits = model(features)
            loss = criterion(logits, labels)
            losses.append(float(loss.item()))
            probs.extend(torch.sigmoid(logits).squeeze(1).cpu().tolist())
    return float(sum(losses) / len(losses)) if losses else 0.0, np.asarray(probs, dtype=np.float32)


def _metrics_from_probs(
    y_true: np.ndarray,
    probs: np.ndarray,
    *,
    positive_class: str,
    threshold: float = 0.5,
) -> dict:
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = (np.asarray(probs) >= float(threshold)).astype(np.int32)
    try:
        auc = float(roc_auc_score(y_true, probs))
    except Exception:
        auc = float("nan")
    try:
        aupr = float(average_precision_score(y_true, probs))
    except Exception:
        aupr = float("nan")

    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": auc,
        "aupr": aupr,
    }
    if positive_class in ("real", "non_hallucination"):
        metrics["reported_positive_class"] = "real"
    else:
        metrics["reported_positive_class"] = "hallucination"
    return metrics


def _select_validation_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """Maximize validation F1; test labels never enter threshold selection."""
    targets = np.asarray(y_true, dtype=np.int32).reshape(-1)
    scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if targets.size == 0 or targets.size != scores.size:
        raise ValueError("Threshold selection requires equal non-empty val arrays.")
    candidates = np.unique(np.concatenate(([0.0], scores, [1.0])))
    best_key = (-np.inf, -np.inf, -np.inf)
    best_threshold = 0.5
    for threshold in candidates:
        prediction = (scores >= threshold).astype(np.int32)
        key = (
            float(f1_score(targets, prediction, zero_division=0)),
            float(accuracy_score(targets, prediction)),
            -abs(float(threshold) - 0.5),
        )
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def _targets_for_positive_class(labels: np.ndarray, positive_class: str) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int32)
    if positive_class == "hallucination":
        return (labels == 0).astype(np.float32)
    return (labels == 1).astype(np.float32)


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _slug(value: str) -> str:
    return value.replace("+", "__").replace("/", "_")


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
        print(f"[TorchProbe] WARNING: failed to write summary table: {exc}")


if __name__ == "__main__":
    main()

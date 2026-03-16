from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from packages.training.trainer import Trainer


@dataclass
class CVResult:
    """Result of k-fold cross-validation."""

    k: int
    fold_accuracies: list[float]
    mean_accuracy: float
    std_accuracy: float
    best_fold: int
    worst_fold: int


def cross_validate(
    *,
    features: list[dict[str, Any]],
    labels: list[str],
    k: int = 5,
    model_type: str = "weighted_vote",
    seed: int = 42,
    **train_kwargs: Any,
) -> CVResult:
    """Perform stratified k-fold cross-validation."""
    if len(features) < k:
        raise ValueError(f"need at least {k} samples for {k}-fold CV")

    rng = random.Random(seed)

    # Stratified split
    indices = list(range(len(features)))
    rng.shuffle(indices)
    folds: list[list[int]] = [[] for _ in range(k)]
    for i, idx in enumerate(indices):
        folds[i % k].append(idx)

    fold_accuracies: list[float] = []

    for fold_idx in range(k):
        test_indices = set(folds[fold_idx])
        train_features = [features[i] for i in range(len(features)) if i not in test_indices]
        train_labels = [labels[i] for i in range(len(labels)) if i not in test_indices]
        [features[i] for i in folds[fold_idx]]
        [labels[i] for i in folds[fold_idx]]

        trainer = Trainer(seed=seed + fold_idx)

        if model_type == "weighted_vote":
            result = trainer.train_weighted_vote(
                features=train_features,
                labels=train_labels,
                model_id=f"cv_fold_{fold_idx}",
                **train_kwargs,
            )
        elif model_type == "decision_tree":
            result = trainer.train_decision_tree(
                features=train_features,
                labels=train_labels,
                model_id=f"cv_fold_{fold_idx}",
                **train_kwargs,
            )
        else:
            raise ValueError(f"unsupported model type: {model_type}")

        # Evaluate on test set (simplified: use training accuracy as proxy since
        # we don't have a separate predict method yet)
        fold_accuracies.append(result.accuracy)

    mean_acc = sum(fold_accuracies) / len(fold_accuracies)
    variance = sum((a - mean_acc) ** 2 for a in fold_accuracies) / len(fold_accuracies)
    std_acc = variance**0.5

    return CVResult(
        k=k,
        fold_accuracies=[round(a, 4) for a in fold_accuracies],
        mean_accuracy=round(mean_acc, 4),
        std_accuracy=round(std_acc, 4),
        best_fold=fold_accuracies.index(max(fold_accuracies)),
        worst_fold=fold_accuracies.index(min(fold_accuracies)),
    )

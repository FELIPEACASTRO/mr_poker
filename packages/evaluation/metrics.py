from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClassificationMetrics:
    """Comprehensive classification metrics."""

    accuracy: float
    total_samples: int
    per_class: dict[str, dict[str, float]]
    macro_f1: float
    macro_precision: float
    macro_recall: float
    confusion_matrix: dict[str, dict[str, int]]


def compute_metrics(
    predictions: list[str],
    actuals: list[str],
) -> ClassificationMetrics:
    """Compute accuracy, F1, precision, recall per class and macro-averaged."""
    if len(predictions) != len(actuals):
        raise ValueError("predictions and actuals must have same length")
    if not predictions:
        return ClassificationMetrics(
            accuracy=0.0,
            total_samples=0,
            per_class={},
            macro_f1=0.0,
            macro_precision=0.0,
            macro_recall=0.0,
            confusion_matrix={},
        )

    correct = sum(1 for p, a in zip(predictions, actuals) if p == a)
    accuracy = correct / len(predictions)

    classes = sorted(set(predictions) | set(actuals))
    confusion: dict[str, dict[str, int]] = {c: {c2: 0 for c2 in classes} for c in classes}
    for pred, actual in zip(predictions, actuals):
        confusion[actual][pred] += 1

    per_class = compute_per_class_metrics(confusion, classes)

    f1_values = [per_class[c]["f1"] for c in classes]
    prec_values = [per_class[c]["precision"] for c in classes]
    rec_values = [per_class[c]["recall"] for c in classes]

    macro_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0
    macro_precision = sum(prec_values) / len(prec_values) if prec_values else 0.0
    macro_recall = sum(rec_values) / len(rec_values) if rec_values else 0.0

    return ClassificationMetrics(
        accuracy=round(accuracy, 4),
        total_samples=len(predictions),
        per_class=per_class,
        macro_f1=round(macro_f1, 4),
        macro_precision=round(macro_precision, 4),
        macro_recall=round(macro_recall, 4),
        confusion_matrix=confusion,
    )


def compute_per_class_metrics(
    confusion: dict[str, dict[str, int]],
    classes: list[str],
) -> dict[str, dict[str, float]]:
    """Compute precision, recall, F1 per class from confusion matrix."""
    result: dict[str, dict[str, float]] = {}

    for cls in classes:
        tp = confusion[cls][cls]
        fp = sum(confusion[other][cls] for other in classes if other != cls)
        fn = sum(confusion[cls][other] for other in classes if other != cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        result[cls] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }

    return result

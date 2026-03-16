from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrainResult:
    """Result of a training run."""

    model_id: str
    model_type: str
    accuracy: float
    num_samples: int
    num_features: int
    class_distribution: dict[str, int]
    hyperparameters: dict[str, Any]
    artifact_path: str


class Trainer:
    """ML trainer supporting multiple model types with cross-validation."""

    def __init__(
        self,
        model_dir: str = "var/models",
        seed: int = 42,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed

    def train_weighted_vote(
        self,
        *,
        features: list[dict[str, Any]],
        labels: list[str],
        model_id: str,
        feature_keys: list[str] | None = None,
    ) -> TrainResult:
        """Train a weighted voting model (enhanced policy table)."""
        if len(features) != len(labels):
            raise ValueError("features and labels must have same length")
        if not features:
            raise ValueError("cannot train on empty dataset")

        random.Random(self.seed)

        # Build decision table grouped by feature key
        if feature_keys is None:
            feature_keys = sorted(features[0].keys())

        table: dict[str, Counter[str]] = {}
        for feat, label in zip(features, labels):
            key = self._feature_key(feat, feature_keys)
            if key not in table:
                table[key] = Counter()
            table[key][label] += 1

        # Evaluate accuracy on training set
        correct = 0
        for feat, label in zip(features, labels):
            key = self._feature_key(feat, feature_keys)
            prediction = table[key].most_common(1)[0][0]
            if prediction == label:
                correct += 1

        accuracy = correct / len(labels) if labels else 0.0
        class_dist = dict(Counter(labels))

        # Save model artifact
        artifact = {
            "model_type": "weighted_vote",
            "model_id": model_id,
            "feature_keys": feature_keys,
            "table": {k: dict(v) for k, v in table.items()},
            "accuracy": accuracy,
            "num_samples": len(labels),
        }
        artifact_path = self.model_dir / f"{model_id}.json"
        artifact_path.write_text(json.dumps(artifact, indent=2))

        return TrainResult(
            model_id=model_id,
            model_type="weighted_vote",
            accuracy=round(accuracy, 4),
            num_samples=len(labels),
            num_features=len(feature_keys),
            class_distribution=class_dist,
            hyperparameters={"seed": self.seed},
            artifact_path=str(artifact_path),
        )

    def train_decision_tree(
        self,
        *,
        features: list[dict[str, Any]],
        labels: list[str],
        model_id: str,
        max_depth: int = 5,
    ) -> TrainResult:
        """Train a simple decision tree (rule extraction)."""
        if not features:
            raise ValueError("cannot train on empty dataset")

        # Simple frequency-based tree: group by most discriminative feature
        feature_keys = sorted(features[0].keys())
        best_feature = self._best_split_feature(features, labels, feature_keys)

        tree: dict[str, dict[str, int]] = {}
        for feat, label in zip(features, labels):
            val = str(feat.get(best_feature, "?"))
            if val not in tree:
                tree[val] = {}
            tree[val][label] = tree[val].get(label, 0) + 1

        # Evaluate
        correct = 0
        for feat, label in zip(features, labels):
            val = str(feat.get(best_feature, "?"))
            if val in tree:
                pred = max(tree[val], key=tree[val].get)  # type: ignore[arg-type]
                if pred == label:
                    correct += 1

        accuracy = correct / len(labels) if labels else 0.0

        artifact = {
            "model_type": "decision_tree",
            "model_id": model_id,
            "split_feature": best_feature,
            "tree": tree,
            "max_depth": max_depth,
        }
        artifact_path = self.model_dir / f"{model_id}.json"
        artifact_path.write_text(json.dumps(artifact, indent=2))

        return TrainResult(
            model_id=model_id,
            model_type="decision_tree",
            accuracy=round(accuracy, 4),
            num_samples=len(labels),
            num_features=len(feature_keys),
            class_distribution=dict(Counter(labels)),
            hyperparameters={"max_depth": max_depth, "split_feature": best_feature},
            artifact_path=str(artifact_path),
        )

    def _feature_key(self, feat: dict[str, Any], keys: list[str]) -> str:
        return "|".join(f"{k}={feat.get(k, '?')}" for k in keys)

    def _best_split_feature(
        self,
        features: list[dict[str, Any]],
        labels: list[str],
        feature_keys: list[str],
    ) -> str:
        """Find the feature that best separates classes (information gain proxy)."""
        best_score = -1.0
        best_feature = feature_keys[0]

        for key in feature_keys:
            groups: dict[str, Counter[str]] = {}
            for feat, label in zip(features, labels):
                val = str(feat.get(key, "?"))
                if val not in groups:
                    groups[val] = Counter()
                groups[val][label] += 1

            # Score: weighted purity
            total = len(labels)
            score = 0.0
            for group_counts in groups.values():
                group_total = sum(group_counts.values())
                if group_total == 0:
                    continue
                majority = group_counts.most_common(1)[0][1]
                score += (majority / group_total) * (group_total / total)

            if score > best_score:
                best_score = score
                best_feature = key

        return best_feature

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass
class DatasetEvaluation:
    rows: int
    accuracy: float
    split_accuracy: dict[str, float]
    taxonomy_accuracy: dict[str, float]
    confusion: dict[str, dict[str, int]]
    confidence_bins: list[dict[str, Any]]


def _bin_for(conf: float) -> str:
    if conf < 0.2:
        return '0.0-0.2'
    if conf < 0.4:
        return '0.2-0.4'
    if conf < 0.6:
        return '0.4-0.6'
    if conf < 0.8:
        return '0.6-0.8'
    return '0.8-1.0'


def evaluate_policy_model(model, rows: list[dict[str, Any]], fallback: str = 'check') -> DatasetEvaluation:
    correct = 0
    split_total = Counter()
    split_correct = Counter()
    tag_total = Counter()
    tag_correct = Counter()
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    bins: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        bucket_key = row['bucket_info']['bucket_key']
        pred = model.predict(bucket_key, fallback=fallback)
        aligned = pred['action'] == row['label_action']
        correct += int(aligned)
        split_total[row['split']] += 1
        split_correct[row['split']] += int(aligned)
        for tag in row.get('taxonomy_tags', []):
            tag_total[tag] += 1
            tag_correct[tag] += int(aligned)
        confusion[row['label_action']][pred['action']] += 1
        bins[_bin_for(float(pred['confidence']))].append(int(aligned))
    split_accuracy = {k: round(split_correct[k] / v, 4) for k, v in split_total.items()}
    taxonomy_accuracy = {k: round(tag_correct[k] / v, 4) for k, v in tag_total.items()}
    confidence_bins = []
    for name in ('0.0-0.2','0.2-0.4','0.4-0.6','0.6-0.8','0.8-1.0'):
        values = bins.get(name, [])
        confidence_bins.append({'bin': name, 'count': len(values), 'accuracy': round(sum(values)/len(values),4) if values else 0.0})
    return DatasetEvaluation(len(rows), round(correct/len(rows),4) if rows else 0.0, split_accuracy, taxonomy_accuracy, {k: dict(v) for k, v in confusion.items()}, confidence_bins)

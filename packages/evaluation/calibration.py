from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CalibrationResult:
    """Calibration metrics for probability estimates."""

    brier_score: float
    expected_calibration_error: float
    max_calibration_error: float
    num_bins: int
    bin_accuracies: list[float]
    bin_confidences: list[float]
    bin_counts: list[int]


def compute_calibration(
    confidences: list[float],
    correct: list[bool],
    num_bins: int = 10,
) -> CalibrationResult:
    """Compute calibration metrics (Brier score, ECE, MCE)."""
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have same length")
    if not confidences:
        return CalibrationResult(
            brier_score=0.0,
            expected_calibration_error=0.0,
            max_calibration_error=0.0,
            num_bins=num_bins,
            bin_accuracies=[],
            bin_confidences=[],
            bin_counts=[],
        )

    # Brier score
    brier = sum((c - (1.0 if cr else 0.0)) ** 2 for c, cr in zip(confidences, correct))
    brier /= len(confidences)

    # Binning for ECE/MCE
    bin_boundaries = [i / num_bins for i in range(num_bins + 1)]
    bin_accuracies: list[float] = []
    bin_confidences: list[float] = []
    bin_counts: list[int] = []

    for i in range(num_bins):
        low = bin_boundaries[i]
        high = bin_boundaries[i + 1]
        bin_items = [
            (c, cr)
            for c, cr in zip(confidences, correct)
            if (low <= c < high) or (i == num_bins - 1 and c == high)
        ]
        count = len(bin_items)
        bin_counts.append(count)
        if count > 0:
            acc = sum(1.0 for _, cr in bin_items if cr) / count
            conf = sum(c for c, _ in bin_items) / count
            bin_accuracies.append(round(acc, 4))
            bin_confidences.append(round(conf, 4))
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)

    # ECE: weighted average of |accuracy - confidence| per bin
    total = len(confidences)
    ece = sum(
        (bc / total) * abs(ba - bconf)
        for bc, ba, bconf in zip(bin_counts, bin_accuracies, bin_confidences)
        if bc > 0
    )

    # MCE: max |accuracy - confidence|
    mce = max(
        (abs(ba - bconf) for bc, ba, bconf in zip(bin_counts, bin_accuracies, bin_confidences) if bc > 0),
        default=0.0,
    )

    return CalibrationResult(
        brier_score=round(brier, 4),
        expected_calibration_error=round(ece, 4),
        max_calibration_error=round(mce, 4),
        num_bins=num_bins,
        bin_accuracies=bin_accuracies,
        bin_confidences=bin_confidences,
        bin_counts=bin_counts,
    )

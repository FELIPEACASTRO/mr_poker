from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from packages.engine.models import HandState, PlayerState
from packages.features.board_texture import BoardTexture, analyze_board
from packages.features.poker import BaselineFeatures, derive_baseline_features
from packages.features.temporal import TemporalFeatures, derive_temporal_features


@dataclass
class ExpandedFeatures:
    """Full feature vector combining baseline, board texture, and temporal features."""

    baseline: BaselineFeatures
    board_texture: BoardTexture
    temporal: TemporalFeatures

    # Derived composite features
    effective_stack_bb: float = 0.0
    bet_sizing_ratio: float = 0.0
    position: str = ""
    relative_hand_strength: float = 0.0


def extract_features(
    state: HandState,
    player: PlayerState,
    *,
    big_blind: int = 2,
    equity: float | None = None,
) -> ExpandedFeatures:
    """Extract the full feature vector from game state."""
    baseline = derive_baseline_features(state, player)
    board_tex = analyze_board(state.board)
    temporal = derive_temporal_features(state)

    effective_stack = min(p.stack for p in state.players.values() if not p.folded)
    effective_stack_bb = effective_stack / big_blind if big_blind > 0 else 0.0

    bet_sizing_ratio = 0.0
    if state.pot > 0 and baseline.to_call > 0:
        bet_sizing_ratio = baseline.to_call / state.pot

    is_button = player.seat == state.button_seat
    position = "BTN" if is_button else "BB"

    return ExpandedFeatures(
        baseline=baseline,
        board_texture=board_tex,
        temporal=temporal,
        effective_stack_bb=round(effective_stack_bb, 2),
        bet_sizing_ratio=round(bet_sizing_ratio, 4),
        position=position,
        relative_hand_strength=equity if equity is not None else 0.0,
    )


def features_to_dict(features: ExpandedFeatures) -> dict[str, Any]:
    """Flatten expanded features into a single dict for ML consumption."""
    result: dict[str, Any] = {}

    baseline = asdict(features.baseline)
    for key, value in baseline.items():
        result[f"bl_{key}"] = value

    board = asdict(features.board_texture)
    for key, value in board.items():
        result[f"bt_{key}"] = value

    temporal = asdict(features.temporal)
    for key, value in temporal.items():
        result[f"tp_{key}"] = value

    result["effective_stack_bb"] = features.effective_stack_bb
    result["bet_sizing_ratio"] = features.bet_sizing_ratio
    result["position"] = features.position
    result["relative_hand_strength"] = features.relative_hand_strength

    return result

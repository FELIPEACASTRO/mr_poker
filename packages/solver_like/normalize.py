from __future__ import annotations

from typing import Any

from packages.features.poker import derive_baseline_features


def normalize_runtime(runtime, engine) -> dict[str, Any]:
    state = runtime.state
    if state.acting_seat is None:
        raise ValueError('runtime has no acting seat')
    player = state.players[state.acting_seat]
    features = derive_baseline_features(state, player)
    return {
        'hand_id': state.hand_id,
        'actor_seat': player.seat,
        'button_seat': state.button_seat,
        'street': features.street,
        'hole_class': features.hole_class,
        'board_texture': features.board_texture,
        'pot': features.pot,
        'to_call': features.to_call,
        'pot_odds': round(features.pot_odds, 4),
        'spr': round(features.spr, 4),
        'stack': features.stack,
        'board': [str(card) for card in state.board],
        'legal_actions': sorted(a.value for a in engine.legal_actions(runtime)),
        'board_size': features.board_size,
        'is_pair': features.is_pair,
        'is_suited': features.is_suited,
        'is_connected': features.is_connected,
        'high_rank': features.high_rank,
        'low_rank': features.low_rank,
    }


def normalize_snapshot_trace(snapshot: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    return {
        'hand_id': snapshot.get('hand_id') or trace.get('hand_id'),
        'actor_seat': trace.get('actor_seat'),
        'button_seat': snapshot.get('button_seat', 0),
        'street': trace.get('street'),
        'hole_class': trace.get('hole_class'),
        'board_texture': trace.get('board_texture'),
        'pot': trace.get('pot'),
        'to_call': trace.get('to_call'),
        'pot_odds': trace.get('pot_odds'),
        'estimated_equity': trace.get('estimated_equity'),
        'spr': float(str(trace.get('notes', ['spr=0'])[0]).split('=')[-1]) if trace.get('notes') else 0.0,
        'legal_actions': trace.get('legal_actions', []),
        'board': snapshot.get('board', []),
        'board_size': len(snapshot.get('board', [])),
        'action_taken': trace.get('action_type'),
        'rationale': trace.get('rationale'),
    }

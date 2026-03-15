from __future__ import annotations

from collections import Counter
from typing import Any

TAXONOMY_CATALOG: dict[str, dict[str, list[str] | str]] = {
    'preflop': {
        'description': 'Opening, defending and re-raising decisions before community cards are dealt.',
        'tags': ['open', 'defend_blind', '3bet_or_jam', 'limp_or_check_back'],
    },
    'flop': {
        'description': 'Flop nodes centered on initiative, continuation betting and early priced decisions.',
        'tags': ['cbet', 'check_back', 'check_call', 'check_raise'],
    },
    'turn': {
        'description': 'Turn nodes such as barrel, probe and pot-control decisions.',
        'tags': ['barrel', 'probe', 'pot_control', 'priced_continue'],
    },
    'river': {
        'description': 'River nodes focused on bluff-catch, thin value and showdown pressure.',
        'tags': ['bluffcatch', 'thin_value', 'jam_pressure', 'showdown_path'],
    },
    'global': {
        'description': 'Cross-street descriptors of the overall hand pattern.',
        'tags': ['single_raised_pot', 'all_in_hand', 'showdown_reached', 'folded_pre_showdown'],
    },
}


def classify_trace(trace: dict[str, Any]) -> list[str]:
    tags: set[str] = set()
    street = str(trace.get('street', ''))
    action = str(trace.get('action_type', ''))
    hole_class = str(trace.get('hole_class', ''))
    board_texture = str(trace.get('board_texture', ''))
    to_call = int(trace.get('to_call', 0) or 0)
    estimated_equity = float(trace.get('estimated_equity', 0.0) or 0.0)
    pot_odds = float(trace.get('pot_odds', 0.0) or 0.0)

    if street == 'pre_flop':
        tags.add('preflop')
        if to_call == 0 and action in {'bet', 'all_in'}:
            tags.add('open')
        if to_call > 0 and action in {'call', 'raise', 'all_in'}:
            tags.add('defend_blind')
        if action in {'raise', 'all_in'} and to_call > 0:
            tags.add('3bet_or_jam')
        if action in {'check'} and to_call == 0:
            tags.add('limp_or_check_back')

    if street == 'flop':
        tags.add('flop')
        if to_call == 0 and action in {'bet', 'all_in'}:
            tags.add('cbet')
        if to_call == 0 and action == 'check':
            tags.add('check_back')
        if to_call > 0 and action == 'call':
            tags.add('check_call')
        if to_call > 0 and action in {'raise', 'all_in'}:
            tags.add('check_raise')

    if street == 'turn':
        tags.add('turn')
        if to_call == 0 and action in {'bet', 'all_in'}:
            tags.add('barrel')
        if to_call > 0 and action in {'call', 'raise', 'all_in'}:
            tags.add('priced_continue')
        if action == 'check':
            tags.add('pot_control')

    if street == 'river':
        tags.add('river')
        if to_call > 0 and action in {'call', 'fold'}:
            tags.add('bluffcatch')
        if to_call == 0 and action in {'bet', 'all_in'} and estimated_equity >= 0.68:
            tags.add('thin_value')
        if action == 'all_in':
            tags.add('jam_pressure')
        if action == 'check':
            tags.add('showdown_path')

    if 'pair' in hole_class or any(rank in hole_class for rank in ('AK', 'AQ', 'KQ')):
        tags.add('high_card_pressure')
    if board_texture in {'monotone', 'paired', 'wet'}:
        tags.add(f'board_{board_texture}')
    if estimated_equity >= pot_odds and to_call > 0:
        tags.add('priced_continue_positive_edge')
    if estimated_equity < pot_odds and action == 'fold':
        tags.add('priced_fold_disciplined')
    return sorted(tags)


def classify_hand(hand_snapshot: dict[str, Any], traces: list[dict[str, Any]] | None = None, actions: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    traces = traces or []
    actions = actions or []
    tags: set[str] = set()
    street_counter = Counter()
    action_counter = Counter()
    board = hand_snapshot.get('board', []) if hand_snapshot else []
    board_size = len(board)

    for trace in traces:
        trace_tags = classify_trace(trace)
        tags.update(trace_tags)
        if trace.get('street'):
            street_counter[str(trace['street'])] += 1
        if trace.get('action_type'):
            action_counter[str(trace['action_type'])] += 1

    if hand_snapshot.get('winner_seat') is not None and board_size == 5:
        tags.add('showdown_reached')
    if any(str(a.get('action_type', a.get('action', ''))) == 'all_in' for a in actions) or any('all-in' in str(a).lower() for a in actions):
        tags.add('all_in_hand')
    if any(str(a.get('action_type', a.get('action', ''))) == 'fold' for a in actions):
        tags.add('folded_pre_showdown')
    if any(t in tags for t in ('cbet', 'barrel', 'bluffcatch', 'thin_value')):
        tags.add('single_raised_pot')

    if board_size == 0:
        board_family = 'preboard'
    elif board_size == 3:
        board_family = 'flop_only'
    elif board_size == 4:
        board_family = 'turn_seen'
    else:
        board_family = 'river_seen'

    return {
        'hand_id': hand_snapshot.get('hand_id'),
        'taxonomy_tags': sorted(tags),
        'board_family': board_family,
        'street_trace_counts': dict(street_counter),
        'action_type_counts': dict(action_counter),
    }

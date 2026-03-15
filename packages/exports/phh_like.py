from __future__ import annotations

from typing import Any


def _card_list(cards: list[str]) -> str:
    return '[' + ', '.join(cards) + ']'


def export_hand_phh_like(*, hand: dict[str, Any], actions: list[dict[str, Any]], latest_snapshot: dict[str, Any], traces: list[dict[str, Any]]) -> str:
    players = latest_snapshot.get('players', {})
    lines: list[str] = []
    lines.append('variant = "NT"')
    lines.append(f'hand_id = "{hand["hand_id"]}"')
    if hand.get('session_id'):
        lines.append(f'session_id = "{hand["session_id"]}"')
    lines.append(f'button_seat = {hand["button_seat"]}')
    lines.append(f'starting_stacks = {list(hand["stacks"])}')
    if hand.get('seed') is not None:
        lines.append(f'seed = {hand["seed"]}')
    if hand.get('deck_prefix'):
        lines.append(f'deck_prefix = {_card_list(hand["deck_prefix"])}')
    lines.append('blinds_or_straddles = [1, 2]')
    lines.append(f'final_street = "{latest_snapshot.get("street", "unknown")}"')
    lines.append(f'board = {_card_list(latest_snapshot.get("board", []))}')
    lines.append('players = {')
    for seat, payload in sorted(players.items(), key=lambda kv: int(kv[0])):
        lines.append(
            f'  "{seat}": {{stack={payload["stack"]}, total_invested={payload["total_invested"]}, '
            f'folded={str(payload["folded"]).lower()}, all_in={str(payload["is_all_in"]).lower()}, '
            f'hole_cards={_card_list(payload.get("hole_cards", []))}}},'
        )
    lines.append('}')
    lines.append('actions = [')
    for item in actions:
        lines.append(
            f'  {{order={item["action_order"]}, seat={item["actor_seat"]}, action="{item["action_type"]}", amount={item["amount"]}}},'
        )
    lines.append(']')
    lines.append('decision_traces = [')
    for item in traces:
        trace = item['trace']
        lines.append(
            '  '
            + '{'
            + f'seat={item["actor_seat"]}, street="{trace.get("street")}", action="{trace.get("action_type")}", '
            + f'equity={trace.get("estimated_equity")}, pot_odds={trace.get("pot_odds")}, '
            + f'hole_class="{trace.get("hole_class")}", board_texture="{trace.get("board_texture")}"'
            + '},'
        )
    lines.append(']')
    lines.append(f'winner_seat = {latest_snapshot.get("winner_seat")}')
    lines.append(f'is_terminal = {str(latest_snapshot.get("is_terminal", False)).lower()}')
    return '\n'.join(lines)


def export_session_manifest(*, session: dict[str, Any], hands: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        'session_id': session['session_id'],
        'session_type': session['session_type'],
        'config': session['config'],
        'summary': session.get('summary'),
        'hand_ids': [hand['hand_id'] for hand in hands],
        'hand_count': len(hands),
        'export_format': 'phh_like_v1',
    }

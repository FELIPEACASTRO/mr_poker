from __future__ import annotations

import time
from dataclasses import dataclass, field
from uuid import uuid4

from packages.common.types import ActionType, Street
from packages.engine.deck import Deck
from packages.engine.exceptions import (
    EngineInvariantError,
    InvalidActionError,
)
from packages.engine.models import ActionEvent, Card, HandState, PlayerState
from packages.evaluator.hands import best_hand_rank, hand_label


@dataclass
class HandRuntime:
    state: HandState
    deck: Deck
    burned_cards: list[Card] = field(default_factory=list)
    initial_stacks: dict[int, int] = field(default_factory=dict)


class GameEngine:
    def __init__(self, small_blind: int = 1, big_blind: int = 2) -> None:
        self.small_blind = small_blind
        self.big_blind = big_blind

    def start_new_hand(
        self,
        *,
        stacks: tuple[int, ...] = (100, 100),
        button_seat: int = 0,
        hand_id: str | None = None,
        seed: int | None = None,
        deck_prefix: list[str] | None = None,
    ) -> HandRuntime:
        num_players = len(stacks)
        if num_players < 2 or num_players > 6:
            raise InvalidActionError("Engine supports 2-6 players")
        hand_id = hand_id or str(uuid4())
        deck = Deck.from_card_strings(deck_prefix or [], seed=seed)
        players = {seat: PlayerState(seat=seat, stack=stack) for seat, stack in enumerate(stacks)}
        state = HandState(
            hand_id=hand_id,
            button_seat=button_seat,
            street=Street.PRE_FLOP,
            pot=0,
            to_call=0,
            min_raise_to=self.big_blind * 2,
            players=players,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
        )
        initial_stacks = {seat: stack for seat, stack in enumerate(stacks)}
        runtime = HandRuntime(state=state, deck=deck, initial_stacks=initial_stacks)

        self._deal_hole_cards(runtime)

        if num_players == 2:
            # Heads-up: button posts SB, other posts BB
            sb_seat = button_seat
            bb_seat = self._next_seat(state, button_seat)
        else:
            # Multi-way: seat left of button posts SB, next posts BB
            sb_seat = self._next_seat(state, button_seat)
            bb_seat = self._next_seat(state, sb_seat)

        self._commit(runtime, sb_seat, self.small_blind)
        state.actions.append(
            ActionEvent(
                actor_seat=sb_seat,
                action_type=ActionType.POST_SMALL_BLIND,
                amount=self.small_blind,
                timestamp=time.time(),
            )
        )
        self._commit(runtime, bb_seat, self.big_blind)
        state.actions.append(
            ActionEvent(
                actor_seat=bb_seat,
                action_type=ActionType.POST_BIG_BLIND,
                amount=self.big_blind,
                timestamp=time.time(),
            )
        )
        state.current_bet = min(self.big_blind, state.players[bb_seat].invested_this_round)

        if num_players == 2:
            # Heads-up: button/SB acts first preflop
            first_to_act = button_seat
        else:
            # Multi-way: UTG (seat after BB) acts first preflop
            first_to_act = self._next_seat(state, bb_seat)

        state.street_starting_seat = first_to_act
        state.acting_seat = first_to_act
        state.to_call = self._to_call_for_player(state.players[first_to_act], state)
        state.min_raise_to = state.current_bet + self.big_blind
        return runtime

    def legal_actions(self, runtime: HandRuntime) -> list[ActionType]:
        state = runtime.state
        if state.is_terminal or state.acting_seat is None:
            return []
        player = state.players[state.acting_seat]
        if player.folded or player.is_all_in:
            return []
        to_call = self._to_call_for_player(player, state)
        actions: list[ActionType] = [ActionType.FOLD]
        if to_call == 0:
            actions.append(ActionType.CHECK)
            if player.stack > 0:
                actions.append(ActionType.BET)
                actions.append(ActionType.ALL_IN)
        else:
            actions.append(ActionType.CALL)
            if player.stack > to_call:
                actions.append(ActionType.RAISE)
                actions.append(ActionType.ALL_IN)
        return actions

    def apply_action(self, runtime: HandRuntime, action_type: ActionType, amount: int = 0) -> HandState:
        state = runtime.state
        if state.is_terminal or state.acting_seat is None:
            raise InvalidActionError("hand is already terminal")
        seat = state.acting_seat
        player = state.players[seat]
        if player.folded or player.is_all_in:
            raise InvalidActionError("acting player is not eligible to act")
        to_call = self._to_call_for_player(player, state)

        if action_type == ActionType.FOLD:
            self._append_action(state, seat, action_type, 0)
            player.folded = True
            active = [p for p in state.players.values() if not p.folded]
            if len(active) == 1:
                self._award_entire_pot(state, active[0].seat)
            else:
                # More than one player remains; advance action
                self._resolve_after_action(runtime)
            return state

        if action_type == ActionType.CHECK:
            if to_call != 0:
                raise InvalidActionError("cannot check while facing a bet")
            self._append_action(state, seat, action_type, 0)
            state.acted_this_street.add(seat)
            self._resolve_after_action(runtime)
            return state

        if action_type == ActionType.CALL:
            if to_call <= 0:
                raise InvalidActionError("nothing to call")
            committed = self._commit(runtime, seat, to_call)
            self._append_action(state, seat, action_type, committed)
            state.acted_this_street.add(seat)
            self._resolve_after_action(runtime)
            return state

        if action_type == ActionType.BET:
            if to_call != 0:
                raise InvalidActionError("use raise when facing a bet")
            if amount < state.big_blind:
                raise InvalidActionError("bet amount must be at least one big blind")
            raise_to = min(player.invested_this_round + player.stack, amount)
            self._raise_to(runtime, seat, raise_to, action_type=action_type)
            return state

        if action_type == ActionType.RAISE:
            if to_call == 0:
                raise InvalidActionError("use bet when opening the action")
            if amount <= state.current_bet:
                raise InvalidActionError("raise amount must exceed current bet")
            self._raise_to(runtime, seat, amount, action_type=action_type)
            return state

        if action_type == ActionType.ALL_IN:
            raise_to = player.invested_this_round + player.stack
            if raise_to <= player.invested_this_round:
                raise InvalidActionError("player has no chips left")
            if raise_to <= state.current_bet:
                committed = self._commit(runtime, seat, player.stack)
                self._append_action(state, seat, action_type, committed)
                state.acted_this_street.add(seat)
                self._resolve_after_action(runtime)
                return state
            self._raise_to(runtime, seat, raise_to, action_type=action_type, allow_short_raise=True)
            return state

        raise InvalidActionError(f"unsupported action: {action_type}")

    def validate_invariants(self, runtime: HandRuntime) -> list[str]:
        """Return a list of invariant violation messages. Empty if all ok."""
        violations: list[str] = []
        state = runtime.state
        initial_stacks = runtime.initial_stacks

        # Pot == sum(total_invested) when not terminal
        if not state.is_terminal:
            expected_pot = sum(p.total_invested for p in state.players.values())
            if state.pot != expected_pot:
                violations.append(
                    f"Pot mismatch: pot={state.pot} but sum(total_invested)={expected_pot}"
                )

        # Chip conservation: total chips in play should equal initial stacks
        if initial_stacks:
            total_initial = sum(initial_stacks.values())
            total_current = sum(p.stack for p in state.players.values()) + state.pot
            if total_current != total_initial:
                violations.append(
                    f"Chip conservation violated: initial={total_initial} current_stacks+pot={total_current}"
                )

        # No player has negative stack
        for seat, player in state.players.items():
            if player.stack < 0:
                violations.append(f"Player {seat} has negative stack: {player.stack}")

        # Exactly one winner when terminal (or split pot where winner_seat is None)
        if state.is_terminal:
            # If showdown reached, winner_seat can be None for split pots
            if not state.showdown_reached and state.winner_seat is None:
                violations.append("Terminal state reached without showdown but no winner_seat")

        return violations

    def state_snapshot(self, runtime: HandRuntime) -> dict:
        state = runtime.state
        return {
            "hand_id": state.hand_id,
            "street": state.street.value,
            "pot": state.pot,
            "to_call": state.to_call,
            "acting_seat": state.acting_seat,
            "board": [str(card) for card in state.board],
            "is_terminal": state.is_terminal,
            "winner_seat": state.winner_seat,
            "pot_segments": self._pot_segments(state),
            "players": {
                seat: {
                    "stack": player.stack,
                    "invested_this_round": player.invested_this_round,
                    "total_invested": player.total_invested,
                    "folded": player.folded,
                    "is_all_in": player.is_all_in,
                    "hole_cards": [str(card) for card in player.hole_cards],
                }
                for seat, player in state.players.items()
            },
            "actions": [
                {
                    "seat": event.actor_seat,
                    "action": event.action_type.value,
                    "amount": event.amount,
                    "street": event.street.value,
                    "note": event.note,
                }
                for event in state.actions
            ],
        }

    def _deal_hole_cards(self, runtime: HandRuntime) -> None:
        state = runtime.state
        num_players = len(state.players)
        seats = sorted(state.players.keys())
        btn_idx = seats.index(state.button_seat)

        if num_players == 2:
            # Heads-up legacy order: button first, then other, twice
            order = [seats[btn_idx], seats[(btn_idx + 1) % num_players],
                     seats[btn_idx], seats[(btn_idx + 1) % num_players]]
        else:
            # Multi-way: deal starting from seat after button, 2 rounds
            order = []
            for _round in range(2):
                for i in range(1, num_players + 1):
                    order.append(seats[(btn_idx + i) % num_players])

        for seat in order:
            state.players[seat].hole_cards.extend(runtime.deck.draw(1))

    def _commit(self, runtime: HandRuntime, seat: int, amount: int) -> int:
        state = runtime.state
        player = state.players[seat]
        committed = min(amount, player.stack)
        player.stack -= committed
        player.invested_this_round += committed
        player.total_invested += committed
        if player.stack == 0:
            player.is_all_in = True
        state.pot += committed

        # Invariant check: pot == sum of total_invested (non-terminal only)
        if not state.is_terminal:
            expected = sum(p.total_invested for p in state.players.values())
            if state.pot != expected:
                raise EngineInvariantError(
                    f"Pot invariant violated after commit: pot={state.pot}, sum(total_invested)={expected}"
                )

        return committed

    def _append_action(self, state: HandState, seat: int, action_type: ActionType, amount: int, note: str = "") -> None:
        state.actions.append(
            ActionEvent(
                actor_seat=seat,
                action_type=action_type,
                amount=amount,
                street=state.street,
                note=note,
                timestamp=time.time(),
            )
        )

    def _to_call_for_player(self, player: PlayerState, state: HandState) -> int:
        return max(0, state.current_bet - player.invested_this_round)

    def _raise_to(
        self,
        runtime: HandRuntime,
        seat: int,
        raise_to: int,
        *,
        action_type: ActionType,
        allow_short_raise: bool = False,
    ) -> None:
        state = runtime.state
        player = state.players[seat]
        previous_bet = state.current_bet
        if not allow_short_raise and state.min_raise_to is not None and raise_to < state.min_raise_to:
            raise InvalidActionError("raise amount is below minimum raise")
        needed = raise_to - player.invested_this_round
        if needed <= 0:
            raise InvalidActionError("raise must increase player investment")
        if needed > player.stack:
            raise InvalidActionError("raise exceeds player stack")
        committed = self._commit(runtime, seat, needed)
        new_bet = player.invested_this_round
        full_raise_size = new_bet - previous_bet
        self._append_action(state, seat, action_type, committed, note=f"raise_to={new_bet}")
        state.current_bet = new_bet
        state.last_aggressor_seat = seat
        if full_raise_size > 0:
            state.min_raise_to = new_bet + full_raise_size
        state.acted_this_street = {seat}
        self._resolve_after_action(runtime)

    def _resolve_after_action(self, runtime: HandRuntime) -> None:
        state = runtime.state
        if state.is_terminal:
            return
        active = [p for p in state.players.values() if not p.folded]
        if len(active) == 1:
            self._award_entire_pot(state, active[0].seat)
            return

        if all(player.is_all_in for player in active) and self._all_bets_matched(state):
            self._refund_uncalled_if_needed(state)
            self._runout_and_showdown(runtime)
            return

        # Find next active, non-all-in seat
        next_seat = self._next_active_seat(state, state.acting_seat)
        if next_seat is None:
            # All remaining active players are all-in; run it out
            self._refund_uncalled_if_needed(state)
            self._runout_and_showdown(runtime)
            return

        next_player = state.players[next_seat]
        next_to_call = self._to_call_for_player(next_player, state)

        if next_to_call > 0:
            state.acting_seat = next_seat
            state.to_call = next_to_call
            return

        if len(state.acted_this_street) >= len([p for p in active if not p.is_all_in]):
            self._advance_street_or_showdown(runtime)
            return

        state.acting_seat = next_seat
        state.to_call = 0

    def _all_bets_matched(self, state: HandState) -> bool:
        active = [p for p in state.players.values() if not p.folded]
        max_invested = max(p.invested_this_round for p in active)
        return all(p.invested_this_round == max_invested or p.is_all_in for p in active)

    def _advance_street_or_showdown(self, runtime: HandRuntime) -> None:
        state = runtime.state
        if state.street == Street.RIVER:
            self._refund_uncalled_if_needed(state)
            self._showdown(runtime)
            return

        for player in state.players.values():
            player.invested_this_round = 0
        state.current_bet = 0
        state.to_call = 0
        state.min_raise_to = state.big_blind
        state.acted_this_street = set()
        state.last_aggressor_seat = None

        if state.street == Street.PRE_FLOP:
            state.street = Street.FLOP
            self._burn(runtime)
            state.board.extend(runtime.deck.draw(3))
        elif state.street == Street.FLOP:
            state.street = Street.TURN
            self._burn(runtime)
            state.board.extend(runtime.deck.draw(1))
        elif state.street == Street.TURN:
            state.street = Street.RIVER
            self._burn(runtime)
            state.board.extend(runtime.deck.draw(1))

        # Post-flop: first active seat after button (OOP first)
        first_post_flop = self._first_active_seat_after(state, state.button_seat)
        state.street_starting_seat = first_post_flop
        state.acting_seat = first_post_flop

    def _runout_and_showdown(self, runtime: HandRuntime) -> None:
        state = runtime.state
        while state.street != Street.RIVER:
            self._advance_street_or_showdown(runtime)
            if state.is_terminal:
                return
        self._refund_uncalled_if_needed(state)
        self._showdown(runtime)

    def _showdown(self, runtime: HandRuntime) -> None:
        state = runtime.state
        state.showdown_reached = True
        rankings: dict[int, tuple[int, tuple[int, ...]]] = {}
        for seat, player in state.players.items():
            if player.folded:
                continue
            rankings[seat] = best_hand_rank(player.hole_cards + state.board)

        payouts = {seat: 0 for seat in state.players}
        segment_notes: list[str] = []
        for segment in self._pot_segments(state):
            eligible = [seat for seat in segment["eligible_seats"] if seat in rankings]
            if not eligible:
                continue
            best_rank = max(rankings[seat] for seat in eligible)
            winners = [seat for seat in eligible if rankings[seat] == best_rank]
            share = segment["amount"] // len(winners)
            remainder = segment["amount"] % len(winners)
            for seat in winners:
                payouts[seat] += share
            if remainder:
                payouts[self._odd_chip_seat(state, winners)] += remainder
            segment_notes.append(
                f"segment={segment['amount']}:" + ",".join(str(seat) for seat in winners)
            )

        for seat, amount in payouts.items():
            state.players[seat].stack += amount
        state.pot = 0
        winners = [seat for seat, amount in payouts.items() if amount > 0]
        state.winner_seat = winners[0] if len(winners) == 1 else None
        state.is_terminal = True
        state.street = Street.TERMINAL
        state.acting_seat = None
        state.to_call = 0
        if rankings:
            best_rank = max(rankings.values())
            note = hand_label(best_rank)
        else:
            note = "n/a"
        state.actions.append(
            ActionEvent(
                actor_seat=self.winner_for_log(winners),
                action_type=ActionType.CHECK,
                amount=0,
                street=Street.SHOWDOWN,
                note=f"winner:{note}; {' | '.join(segment_notes)}",
                timestamp=time.time(),
            )
        )

    def winner_for_log(self, winners: list[int]) -> int:
        return winners[0] if winners else 0

    def _award_entire_pot(self, state: HandState, winner_seat: int) -> None:
        state.players[winner_seat].stack += state.pot
        state.pot = 0
        state.winner_seat = winner_seat
        state.is_terminal = True
        state.street = Street.TERMINAL
        state.acting_seat = None
        state.to_call = 0

    def _burn(self, runtime: HandRuntime) -> None:
        runtime.burned_cards.extend(runtime.deck.draw(1))

    def _next_seat(self, state: HandState, seat: int) -> int:
        """Return the next seat in order (wrapping), regardless of fold/all-in status."""
        seats = sorted(state.players.keys())
        num = len(seats)
        idx = seats.index(seat)
        return seats[(idx + 1) % num]

    def _next_active_seat(self, state: HandState, seat: int) -> int | None:
        """Return the next seat that is active (not folded, not all-in). Returns None if no such seat."""
        seats = sorted(state.players.keys())
        num = len(seats)
        idx = seats.index(seat)
        for i in range(1, num):
            candidate = seats[(idx + i) % num]
            p = state.players[candidate]
            if not p.folded and not p.is_all_in:
                return candidate
        return None

    def _first_active_seat_after(self, state: HandState, seat: int) -> int | None:
        """Return the first active (not folded, not all-in) seat after the given seat.
        Falls back to first non-folded seat if all active are all-in."""
        seats = sorted(state.players.keys())
        num = len(seats)
        idx = seats.index(seat)
        # First try: non-folded and non-all-in
        for i in range(1, num):
            candidate = seats[(idx + i) % num]
            p = state.players[candidate]
            if not p.folded and not p.is_all_in:
                return candidate
        # Fallback: non-folded (even if all-in)
        for i in range(1, num):
            candidate = seats[(idx + i) % num]
            if not state.players[candidate].folded:
                return candidate
        return None

    def _other_active_seat(self, state: HandState, seat: int) -> int:
        """Legacy compat: find next non-folded seat (may be all-in). Raises if none found."""
        seats = sorted(state.players.keys())
        num = len(seats)
        idx = seats.index(seat)
        for i in range(1, num):
            candidate = seats[(idx + i) % num]
            if not state.players[candidate].folded:
                return candidate
        raise EngineInvariantError("no other active seat found")

    def _odd_chip_seat(self, state: HandState, winners: list[int]) -> int:
        """Award odd chip to first winner after button (OOP)."""
        seats = sorted(state.players.keys())
        num = len(seats)
        idx = seats.index(state.button_seat)
        for i in range(1, num + 1):
            candidate = seats[(idx + i) % num]
            if candidate in winners:
                return candidate
        return winners[0]

    def _refund_uncalled_if_needed(self, state: HandState) -> int:
        active = [p for p in state.players.values() if not p.folded]
        if len(active) < 2:
            return 0
        ordered = sorted((p.total_invested, p.seat) for p in active)
        highest, seat = ordered[-1]
        second = ordered[-2][0]
        if highest <= second:
            return 0
        refund = highest - second
        state.players[seat].stack += refund
        state.players[seat].total_invested -= refund
        state.pot -= refund
        if state.players[seat].invested_this_round >= refund:
            state.players[seat].invested_this_round -= refund
        else:
            state.players[seat].invested_this_round = 0
        state.actions.append(
            ActionEvent(
                actor_seat=seat,
                action_type=ActionType.CHECK,
                amount=refund,
                street=Street.SHOWDOWN,
                note="uncalled_return",
                timestamp=time.time(),
            )
        )
        return refund

    def _pot_segments(self, state: HandState) -> list[dict]:
        invested_levels = sorted({p.total_invested for p in state.players.values() if p.total_invested > 0})
        segments: list[dict] = []
        prev = 0
        for level in invested_levels:
            contributors = [p.seat for p in state.players.values() if p.total_invested >= level]
            if not contributors:
                continue
            amount = (level - prev) * len(contributors)
            if amount <= 0:
                prev = level
                continue
            eligible = [p.seat for p in state.players.values() if p.total_invested >= level and not p.folded]
            segments.append({
                "amount": amount,
                "contributors": contributors,
                "eligible_seats": eligible,
                "level": level,
            })
            prev = level
        return segments

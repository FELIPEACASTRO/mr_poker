"""Information Set Monte Carlo Tree Search (IS-MCTS) Agent.

Handles hidden information via determinization: sample opponent hole cards
consistent with observations, then run standard MCTS on the resulting
perfect-information game tree.

Features:
- UCB1 selection with configurable exploration constant
- Random or opponent-model-informed rollouts
- Backpropagation of rewards through the tree
- Integration with OpponentTracker for exploitative play

Reference: Cowling et al. (2012) "Information Set Monte Carlo Tree Search"
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any

from packages.cfr_agent.deep_cfr import (
    ACTION_INDEX,
    FEATURE_DIM,
    extract_features,
)
from packages.cfr_agent.trainer import _size_action
from packages.common.types import ActionType, Street
from packages.engine.engine import GameEngine, HandRuntime
from packages.opponent_model.classifier import OpponentTracker, exploitation_adjustments
from packages.strategy.mixed import ActionDistribution

logger = logging.getLogger(__name__)


@dataclass
class MCTSNode:
    """Node in the MCTS search tree."""

    visit_count: int = 0
    total_value: float = 0.0
    children: dict[ActionType, "MCTSNode"] = field(default_factory=dict)
    prior_prob: float = 1.0
    parent: "MCTSNode | None" = None
    action: ActionType | None = None

    @property
    def value(self) -> float:
        """Mean value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def ucb1(self, parent_visits: int, c: float = 1.414) -> float:
        """UCB1 score for tree selection."""
        if self.visit_count == 0:
            return float("inf")
        exploitation = self.total_value / self.visit_count
        exploration = c * math.sqrt(math.log(parent_visits) / self.visit_count)
        return exploitation + exploration

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def best_child(self, c: float = 1.414) -> tuple[ActionType, "MCTSNode"]:
        """Select child with highest UCB1 score."""
        best_action = None
        best_node = None
        best_score = float("-inf")
        for action, child in self.children.items():
            score = child.ucb1(self.visit_count, c)
            if score > best_score:
                best_score = score
                best_action = action
                best_node = child
        if best_action is None or best_node is None:
            raise ValueError("no children to select from")
        return best_action, best_node

    def most_visited_child(self) -> tuple[ActionType, "MCTSNode"]:
        """Return child with most visits (used for final action selection)."""
        best_action = None
        best_node = None
        best_visits = -1
        for action, child in self.children.items():
            if child.visit_count > best_visits:
                best_visits = child.visit_count
                best_action = action
                best_node = child
        if best_action is None or best_node is None:
            raise ValueError("no children to select from")
        return best_action, best_node


class ISMCTSAgent:
    """Information Set MCTS agent with optional opponent modeling.

    For each search iteration:
    1. Determinize: sample hidden cards consistent with observations
    2. Select: walk down the tree using UCB1
    3. Expand: create child nodes for legal actions
    4. Rollout: simulate to terminal state (random or informed)
    5. Backpropagate: update visit counts and values
    """

    def __init__(
        self,
        *,
        num_simulations: int = 100,
        c: float = 1.414,
        opponent_tracker: OpponentTracker | None = None,
        seed: int = 42,
    ) -> None:
        self.num_simulations = num_simulations
        self.c = c
        self.opponent_tracker = opponent_tracker
        self.rng = random.Random(seed)

    def search(
        self, runtime: HandRuntime, engine: GameEngine, num_simulations: int | None = None
    ) -> ActionDistribution:
        """Run IS-MCTS and return action distribution based on visit counts."""
        sims = num_simulations or self.num_simulations
        state = runtime.state
        if state.is_terminal or state.acting_seat is None:
            return ActionDistribution(probabilities={})

        root = MCTSNode()
        seat = state.acting_seat

        for _ in range(sims):
            # Determinize: create a playout runtime with sampled hidden cards
            sim_runtime = self._determinize(runtime, engine, seat)
            if sim_runtime is None:
                continue

            # Selection + expansion + rollout + backprop
            self._simulate(root, sim_runtime, engine, seat)

        # Convert visit counts to action distribution
        if not root.children:
            legal = engine.legal_actions(runtime)
            n = len(legal) or 1
            return ActionDistribution(probabilities={a: 1.0 / n for a in legal})

        total_visits = sum(child.visit_count for child in root.children.values())
        if total_visits == 0:
            n = len(root.children)
            return ActionDistribution(
                probabilities={a: 1.0 / n for a in root.children}
            )

        probs = {
            action: child.visit_count / total_visits
            for action, child in root.children.items()
        }
        return ActionDistribution(probabilities=probs)

    def decide(
        self, runtime: HandRuntime, engine: GameEngine
    ) -> tuple[ActionType, int]:
        """Select the best action via MCTS search."""
        state = runtime.state
        if state.acting_seat is None:
            return ActionType.FOLD, 0

        dist = self.search(runtime, engine)
        if not dist.probabilities:
            return ActionType.FOLD, 0

        action = dist.dominant_action
        seat = state.acting_seat
        player = state.players[seat]
        amount = _size_action(action, state, player, engine)
        return action, amount

    def _determinize(
        self, runtime: HandRuntime, engine: GameEngine, hero_seat: int
    ) -> HandRuntime | None:
        """Create a copy of the game with sampled opponent cards.

        Keeps hero's cards fixed, samples new cards for opponents from
        the remaining deck. This handles the information asymmetry in poker.
        """
        state = runtime.state
        try:
            # Gather known cards
            hero_cards = [str(c) for c in state.players[hero_seat].hole_cards]
            board_cards = [str(c) for c in state.board]
            burned = [str(c) for c in runtime.burned_cards]
            known = set(hero_cards + board_cards + burned)

            # Build remaining deck
            all_cards = []
            for suit in ["h", "d", "c", "s"]:
                for rank in ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]:
                    card_str = rank + suit
                    if card_str not in known:
                        all_cards.append(card_str)
            self.rng.shuffle(all_cards)

            # Assign random cards to opponents
            opponent_cards: dict[int, list[str]] = {}
            card_idx = 0
            for seat_id in sorted(state.players.keys()):
                if seat_id == hero_seat:
                    opponent_cards[seat_id] = hero_cards
                else:
                    opponent_cards[seat_id] = all_cards[card_idx:card_idx + 2]
                    card_idx += 2

            # Build deck prefix: hole cards in seat order, then burned+board
            deck_prefix: list[str] = []
            for seat_id in sorted(state.players.keys()):
                deck_prefix.extend(opponent_cards[seat_id])
            deck_prefix.extend(burned[:len(board_cards)])
            deck_prefix.extend(board_cards)

            # Replay from start
            new_runtime = engine.start_new_hand(
                stacks=tuple(runtime.initial_stacks[s] for s in sorted(runtime.initial_stacks)),
                button_seat=state.button_seat,
                deck_prefix=deck_prefix,
            )

            # Replay actions up to current point
            for ev in state.actions:
                if new_runtime.state.is_terminal or new_runtime.state.acting_seat is None:
                    break
                if ev.action_type in (ActionType.POST_SMALL_BLIND, ActionType.POST_BIG_BLIND):
                    continue
                engine.apply_action(new_runtime, ev.action_type, ev.amount)

            if new_runtime.state.is_terminal:
                return None
            return new_runtime

        except Exception:
            return None

    def _simulate(
        self, node: MCTSNode, runtime: HandRuntime, engine: GameEngine, hero_seat: int
    ) -> float:
        """One MCTS iteration: select, expand, rollout, backprop."""
        state = runtime.state

        if state.is_terminal or state.acting_seat is None:
            return self._terminal_value(state, hero_seat)

        # Expand if leaf
        if node.is_leaf():
            legal = engine.legal_actions(runtime)
            if not legal:
                return self._terminal_value(state, hero_seat)
            for action in legal:
                child = MCTSNode(parent=node, action=action)
                node.children[action] = child

            # Rollout from current state
            value = self._rollout(runtime, engine, hero_seat)
            node.visit_count += 1
            node.total_value += value
            return value

        # Select best child
        action, child = node.best_child(self.c)

        # Apply action
        player = state.players[state.acting_seat]
        amount = _size_action(action, state, player, engine)
        try:
            engine.apply_action(runtime, action, amount)
        except Exception:
            # Invalid action, treat as terminal
            value = self._terminal_value(state, hero_seat)
            node.visit_count += 1
            node.total_value += value
            return value

        # Recurse
        value = self._simulate(child, runtime, engine, hero_seat)

        # Backpropagate
        node.visit_count += 1
        node.total_value += value
        return value

    def _rollout(
        self, runtime: HandRuntime, engine: GameEngine, hero_seat: int
    ) -> float:
        """Random playout to terminal state. Uses opponent model if available."""
        state = runtime.state
        max_steps = 20  # Safety limit

        for _ in range(max_steps):
            if state.is_terminal or state.acting_seat is None:
                break

            seat = state.acting_seat
            legal = engine.legal_actions(runtime)
            if not legal:
                break

            action = self._rollout_action(seat, legal, state)
            player = state.players[seat]
            amount = _size_action(action, state, player, engine)

            try:
                engine.apply_action(runtime, action, amount)
            except Exception:
                break

        return self._terminal_value(state, hero_seat)

    def _rollout_action(
        self, seat: int, legal: list[ActionType], state: Any
    ) -> ActionType:
        """Choose action for rollout. Uses opponent model if available."""
        if self.opponent_tracker is not None:
            adjustments = self.opponent_tracker.get_adjustments(seat)
            return self._informed_rollout_action(legal, adjustments)

        # Random rollout
        return self.rng.choice(legal)

    def _informed_rollout_action(
        self, legal: list[ActionType], adjustments: dict[str, float]
    ) -> ActionType:
        """Weight actions based on opponent model adjustments."""
        weights: dict[ActionType, float] = {}
        for action in legal:
            if action == ActionType.FOLD:
                weights[action] = adjustments.get("fold_freq", 1.0)
            elif action in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
                weights[action] = adjustments.get("raise_freq", 1.0)
            elif action == ActionType.CALL:
                weights[action] = adjustments.get("call_freq", 1.0)
            elif action == ActionType.CHECK:
                weights[action] = 1.0
            else:
                weights[action] = 1.0

        total = sum(weights.values())
        if total <= 0:
            return self.rng.choice(legal)

        r = self.rng.random() * total
        cumulative = 0.0
        for action, w in weights.items():
            cumulative += w
            if r <= cumulative:
                return action
        return legal[-1]

    def _terminal_value(self, state: Any, hero_seat: int) -> float:
        """Extract normalized reward from terminal state."""
        if not state.is_terminal:
            # Estimate value from current stack vs starting
            return (state.players[hero_seat].stack - 100) / 100.0
        return (state.players[hero_seat].stack - 100) / 100.0

"""LAMIR — Look-Ahead in Model-based Imperfect-information Reasoning.

Uses a learned world model to predict abstract next states and a value
network to evaluate leaf states, enabling planning in imperfect-information
games without full tree traversal.

Reference: inspired by model-based RL approaches applied to IIGs.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from packages.common.types import ActionType
from packages.cfr_agent.deep_cfr import SimpleNN, ACTION_INDEX, NUM_ACTIONS
from packages.strategy.mixed import ActionDistribution


# Abstract state dimensionality — compact game representation
ABSTRACT_STATE_DIM = 8


@dataclass
class AbstractState:
    """Simplified representation of a game state for model-based planning.

    Encodes street, pot ratio, position, and a hash of the action history
    into a fixed-size vector suitable for neural network input.
    """
    street: float = 0.0        # 0=PF, 0.33=F, 0.67=T, 1.0=R
    pot_ratio: float = 0.0     # pot / starting_stack, capped at 1.0
    position: float = 0.0      # 1.0=IP, 0.0=OOP
    action_history_hash: float = 0.0  # normalized hash of action sequence
    aggression: float = 0.0    # fraction of aggressive actions
    stack_ratio: float = 1.0   # remaining stack / starting stack
    to_call_ratio: float = 0.0 # amount to call / pot
    num_actions: float = 0.0   # number of actions taken, normalized

    def to_vector(self) -> list[float]:
        return [
            self.street,
            self.pot_ratio,
            self.position,
            self.action_history_hash,
            self.aggression,
            self.stack_ratio,
            self.to_call_ratio,
            self.num_actions,
        ]

    @classmethod
    def from_game_state(cls, state: Any, seat: int, starting_stack: int = 100) -> AbstractState:
        """Extract an abstract state from a concrete game state."""
        from packages.common.types import Street

        street_map = {
            Street.PRE_FLOP: 0.0,
            Street.FLOP: 0.33,
            Street.TURN: 0.67,
            Street.RIVER: 1.0,
        }
        street_val = street_map.get(state.street, 0.0)

        pot_ratio = min(state.pot / max(starting_stack, 1), 1.0)
        position = 1.0 if seat == state.button_seat else 0.0

        # Hash action history into [0, 1]
        action_hash = 0.0
        num_raises = 0
        num_bets = 0
        for i, ev in enumerate(state.actions):
            action_hash += hash(ev.action_type.value) * (i + 1)
            if ev.action_type == ActionType.RAISE:
                num_raises += 1
            elif ev.action_type == ActionType.BET:
                num_bets += 1
        total_actions = len(state.actions)
        action_hash = (action_hash % 10000) / 10000.0
        aggression = (num_raises + num_bets) / max(total_actions, 1)

        player = state.players[seat]
        stack_ratio = player.stack / max(starting_stack, 1)
        to_call = max(0, state.current_bet - player.invested_this_round)
        to_call_ratio = to_call / max(state.pot, 1)
        num_actions_norm = min(total_actions / 20.0, 1.0)

        return cls(
            street=street_val,
            pot_ratio=pot_ratio,
            position=position,
            action_history_hash=action_hash,
            aggression=aggression,
            stack_ratio=stack_ratio,
            to_call_ratio=to_call_ratio,
            num_actions=num_actions_norm,
        )


# Legal actions for planning (same as CFR action space)
PLAN_ACTIONS = [
    ActionType.FOLD,
    ActionType.CHECK,
    ActionType.CALL,
    ActionType.BET,
    ActionType.RAISE,
    ActionType.ALL_IN,
]


class LAMIRSolver:
    """Look-Ahead in Model-based Imperfect-information Reasoning.

    Plans by simulating abstract state transitions via a learned world
    model and evaluating leaf states with a value network.  No domain-
    specific knowledge is needed beyond the state/action encoding.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        value_hidden_dim: int = 32,
        seed: int = 42,
        legal_actions: list[ActionType] | None = None,
    ) -> None:
        self.rng = random.Random(seed)
        self.legal_actions = legal_actions or PLAN_ACTIONS

        # World model: predicts next abstract state given (state + action one-hot)
        world_input_dim = ABSTRACT_STATE_DIM + NUM_ACTIONS
        self.world_model = SimpleNN(
            world_input_dim, hidden_dim, ABSTRACT_STATE_DIM, seed=seed,
        )

        # Value network: estimates expected value of an abstract state
        self.value_network = SimpleNN(
            ABSTRACT_STATE_DIM, value_hidden_dim, 1, seed=seed + 1,
        )

        # Training buffers
        self._world_model_data: list[tuple[list[float], list[float]]] = []
        self._value_data: list[tuple[list[float], float]] = []

    def _action_one_hot(self, action: ActionType) -> list[float]:
        """Encode an action as a one-hot vector."""
        vec = [0.0] * NUM_ACTIONS
        idx = ACTION_INDEX.get(action)
        if idx is not None:
            vec[idx] = 1.0
        return vec

    def _predict_next_state(
        self, abstract_state: list[float], action: ActionType
    ) -> list[float]:
        """Use the world model to predict the next abstract state."""
        action_vec = self._action_one_hot(action)
        model_input = abstract_state + action_vec
        predicted = self.world_model.forward(model_input)
        # Clamp to [0, 1] since abstract states are normalized
        return [max(0.0, min(1.0, v)) for v in predicted]

    def _evaluate_state(self, abstract_state: list[float]) -> float:
        """Evaluate an abstract state using the value network."""
        return self.value_network.forward(abstract_state)[0]

    def _recursive_plan(
        self,
        abstract_state: list[float],
        depth: int,
        legal: list[ActionType],
    ) -> float:
        """Recursively evaluate a state by planning through the world model."""
        if depth <= 0:
            return self._evaluate_state(abstract_state)

        best_value = float("-inf")
        for action in legal:
            next_state = self._predict_next_state(abstract_state, action)
            value = self._recursive_plan(next_state, depth - 1, legal)
            if value > best_value:
                best_value = value

        return best_value if best_value > float("-inf") else self._evaluate_state(abstract_state)

    def plan(
        self,
        current_state: AbstractState | list[float],
        depth: int = 3,
        legal_actions: list[ActionType] | None = None,
    ) -> ActionDistribution:
        """Plan from the current abstract state using look-ahead search.

        For each legal action, simulates the abstract next state via the
        world model, recursively evaluates to the depth limit, and uses
        the value network at leaves.  Returns a softmax distribution
        over actions weighted by expected value.
        """
        if isinstance(current_state, AbstractState):
            state_vec = current_state.to_vector()
        else:
            state_vec = current_state

        legal = legal_actions or self.legal_actions

        # Compute expected value for each action
        action_values: dict[ActionType, float] = {}
        for action in legal:
            next_state = self._predict_next_state(state_vec, action)
            value = self._recursive_plan(next_state, depth - 1, legal)
            action_values[action] = value

        if not action_values:
            n = len(legal) or 1
            return ActionDistribution(probabilities={a: 1.0 / n for a in legal})

        # Softmax over action values for smooth distribution
        max_val = max(action_values.values())
        exp_vals = {a: math.exp(v - max_val) for a, v in action_values.items()}
        total = sum(exp_vals.values())
        if total <= 0:
            n = len(legal) or 1
            return ActionDistribution(probabilities={a: 1.0 / n for a in legal})

        probs = {a: v / total for a, v in exp_vals.items()}
        return ActionDistribution(probabilities=probs)

    def train_world_model(
        self,
        trajectories: list[tuple[list[float], ActionType, list[float]]],
        lr: float = 0.001,
        epochs: int = 1,
    ) -> float:
        """Train world model on observed (state, action, next_state) transitions.

        Args:
            trajectories: list of (state_vec, action, next_state_vec)
            lr: learning rate
            epochs: training epochs over the data

        Returns:
            Average training loss.
        """
        if not trajectories:
            return 0.0

        total_loss = 0.0
        count = 0
        for _ in range(epochs):
            shuffled = list(trajectories)
            self.rng.shuffle(shuffled)
            for state_vec, action, next_state_vec in shuffled:
                action_vec = self._action_one_hot(action)
                model_input = state_vec + action_vec
                loss = self.world_model.train_step(model_input, next_state_vec, lr=lr)
                total_loss += loss
                count += 1

        return total_loss / max(count, 1)

    def train_value_network(
        self,
        states: list[list[float]],
        returns: list[float],
        lr: float = 0.001,
        epochs: int = 1,
    ) -> float:
        """Train value network on (state, return) pairs.

        Args:
            states: list of abstract state vectors
            returns: list of actual game outcomes
            lr: learning rate
            epochs: training epochs

        Returns:
            Average training loss.
        """
        if not states or not returns:
            return 0.0

        total_loss = 0.0
        count = 0
        indices = list(range(len(states)))
        for _ in range(epochs):
            self.rng.shuffle(indices)
            for i in indices:
                loss = self.value_network.train_step(states[i], [returns[i]], lr=lr)
                total_loss += loss
                count += 1

        return total_loss / max(count, 1)

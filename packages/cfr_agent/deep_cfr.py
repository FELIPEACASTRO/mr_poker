"""Deep CFR — Neural network approximation for CFR.

Instead of storing regrets/strategies in a table (which doesn't generalize),
Deep CFR uses neural networks to predict regrets and strategies for unseen
information sets. This is the key scaling breakthrough that enabled superhuman
poker AI like Pluribus and DeepStack.

Architecture:
- Value Network: info_set features → regret per action
- Policy Network: info_set features → strategy distribution
- Advantage Memory: stores (features, regrets) tuples for training

Reference: Brown et al. (2019) "Deep Counterfactual Regret Minimization"
"""

from __future__ import annotations

import logging
import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from packages.common.types import ActionType
from packages.engine.engine import GameEngine, HandRuntime
from packages.strategy.mixed import ActionDistribution

from packages.cfr_agent.info_set import build_info_set_key
from packages.cfr_agent.trainer import CFRState, _size_action

logger = logging.getLogger(__name__)

# Feature indices for the neural network input
ACTION_INDEX = {
    ActionType.FOLD: 0,
    ActionType.CHECK: 1,
    ActionType.CALL: 2,
    ActionType.BET: 3,
    ActionType.RAISE: 4,
    ActionType.ALL_IN: 5,
}
NUM_ACTIONS = len(ACTION_INDEX)


@dataclass
class InfoSetFeatures:
    """Numerical feature vector for an information set."""
    # Hand features
    high_rank: float = 0.0   # 0-1 normalized
    low_rank: float = 0.0
    is_suited: float = 0.0
    is_pair: float = 0.0

    # Board features
    street: float = 0.0      # 0=PF, 0.33=F, 0.67=T, 1.0=R
    board_flush_draw: float = 0.0
    board_straight_draw: float = 0.0
    board_paired: float = 0.0

    # Position and sizing
    position: float = 0.0    # 1=IP, 0=OOP
    spr: float = 0.0         # stack-to-pot ratio (capped)
    pot_bb: float = 0.0      # pot in big blinds (capped)
    to_call_bb: float = 0.0

    # Action history
    num_raises: float = 0.0
    num_bets: float = 0.0
    aggression: float = 0.0

    def to_vector(self) -> list[float]:
        return [
            self.high_rank, self.low_rank, self.is_suited, self.is_pair,
            self.street, self.board_flush_draw, self.board_straight_draw,
            self.board_paired, self.position, self.spr, self.pot_bb,
            self.to_call_bb, self.num_raises, self.num_bets, self.aggression,
        ]

FEATURE_DIM = 15


def extract_features(state: Any, seat: int) -> InfoSetFeatures:
    """Extract numerical features from a game state for neural network input."""
    from packages.features.poker import RANK_ORDER

    player = state.players[seat]
    cards = player.hole_cards
    if len(cards) != 2:
        return InfoSetFeatures()

    ranks = sorted([RANK_ORDER[c.rank] for c in cards], reverse=True)
    high = (ranks[0] - 2) / 12.0
    low = (ranks[1] - 2) / 12.0
    suited = 1.0 if cards[0].suit == cards[1].suit else 0.0
    paired = 1.0 if cards[0].rank == cards[1].rank else 0.0

    board = state.board
    street = len(board) / 5.0
    board_suits = [c.suit for c in board] if board else []
    from collections import Counter
    suit_counts = Counter(board_suits)
    flush_draw = 1.0 if any(v >= 3 for v in suit_counts.values()) else 0.0
    board_ranks = sorted([RANK_ORDER[c.rank] for c in board]) if board else []
    straight_draw = 0.0
    if len(board_ranks) >= 3:
        for i in range(len(board_ranks) - 2):
            if board_ranks[i + 2] - board_ranks[i] <= 4:
                straight_draw = 1.0
                break
    board_paired_flag = 1.0 if len(set(board_ranks)) < len(board_ranks) else 0.0

    position = 1.0 if seat == state.button_seat else 0.0
    pot_bb = min(state.pot / max(state.big_blind, 1), 50.0) / 50.0
    spr_raw = player.stack / max(state.pot, 1)
    spr = min(spr_raw, 20.0) / 20.0
    to_call = max(0, state.current_bet - player.invested_this_round)
    to_call_bb = min(to_call / max(state.big_blind, 1), 50.0) / 50.0

    num_raises = 0
    num_bets = 0
    for ev in state.actions:
        if ev.action_type == ActionType.RAISE:
            num_raises += 1
        elif ev.action_type == ActionType.BET:
            num_bets += 1
    total_actions = max(len(state.actions), 1)
    aggression = (num_raises + num_bets) / total_actions

    return InfoSetFeatures(
        high_rank=high, low_rank=low, is_suited=suited, is_pair=paired,
        street=street, board_flush_draw=flush_draw,
        board_straight_draw=straight_draw, board_paired=board_paired_flag,
        position=position, spr=spr, pot_bb=pot_bb, to_call_bb=to_call_bb,
        num_raises=min(num_raises / 5.0, 1.0),
        num_bets=min(num_bets / 5.0, 1.0),
        aggression=aggression,
    )


class SimpleNN:
    """Minimal neural network (no external dependencies).

    Single hidden layer with ReLU activation. Trained via SGD.
    This avoids requiring PyTorch/TensorFlow for the core package.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, *, seed: int = 42):
        rng = random.Random(seed)
        scale_h = math.sqrt(2.0 / input_dim)
        scale_o = math.sqrt(2.0 / hidden_dim)

        self.w1 = [[rng.gauss(0, scale_h) for _ in range(input_dim)] for _ in range(hidden_dim)]
        self.b1 = [0.0] * hidden_dim
        self.w2 = [[rng.gauss(0, scale_o) for _ in range(hidden_dim)] for _ in range(output_dim)]
        self.b2 = [0.0] * output_dim

    def forward(self, x: list[float]) -> list[float]:
        """Forward pass: input → hidden (ReLU) → output."""
        # Hidden layer
        hidden = []
        for i in range(len(self.w1)):
            val = self.b1[i] + sum(self.w1[i][j] * x[j] for j in range(len(x)))
            hidden.append(max(0.0, val))  # ReLU

        # Output layer
        output = []
        for i in range(len(self.w2)):
            val = self.b2[i] + sum(self.w2[i][j] * hidden[j] for j in range(len(hidden)))
            output.append(val)

        return output

    def train_step(self, x: list[float], target: list[float], lr: float = 0.001) -> float:
        """Single SGD step with MSE loss. Returns loss."""
        # Forward
        hidden = []
        hidden_raw = []
        for i in range(len(self.w1)):
            val = self.b1[i] + sum(self.w1[i][j] * x[j] for j in range(len(x)))
            hidden_raw.append(val)
            hidden.append(max(0.0, val))

        output = []
        for i in range(len(self.w2)):
            val = self.b2[i] + sum(self.w2[i][j] * hidden[j] for j in range(len(hidden)))
            output.append(val)

        # Loss
        loss = sum((output[i] - target[i]) ** 2 for i in range(len(target))) / len(target)

        # Backward: output layer
        d_output = [(output[i] - target[i]) * 2.0 / len(target) for i in range(len(target))]

        d_hidden = [0.0] * len(hidden)
        for i in range(len(self.w2)):
            for j in range(len(hidden)):
                d_hidden[j] += d_output[i] * self.w2[i][j]
                self.w2[i][j] -= lr * d_output[i] * hidden[j]
            self.b2[i] -= lr * d_output[i]

        # Backward: hidden layer (ReLU derivative)
        for i in range(len(self.w1)):
            if hidden_raw[i] <= 0:
                continue
            grad = d_hidden[i]
            for j in range(len(x)):
                self.w1[i][j] -= lr * grad * x[j]
            self.b1[i] -= lr * grad

        return loss


@dataclass
class AdvantageMemory:
    """Reservoir buffer storing (features, iteration, advantages) tuples."""
    buffer: deque = field(default_factory=lambda: deque(maxlen=500_000))

    def add(self, features: list[float], iteration: int, advantages: list[float]) -> None:
        self.buffer.append((features, iteration, advantages))

    def sample_batch(self, batch_size: int, rng: random.Random) -> list[tuple]:
        if len(self.buffer) <= batch_size:
            return list(self.buffer)
        return rng.sample(list(self.buffer), batch_size)


class DeepCFRTrainer:
    """Deep CFR trainer using neural networks for regret/strategy approximation.

    Instead of tabular CFR, this uses:
    - Advantage networks (one per player) to predict counterfactual advantages
    - A strategy network trained on the average strategy

    The networks generalize across similar information sets, enabling
    play in positions never seen during training.
    """

    def __init__(
        self,
        *,
        small_blind: int = 1,
        big_blind: int = 2,
        starting_stack: int = 100,
        hidden_dim: int = 64,
        seed: int = 42,
    ) -> None:
        self.engine = GameEngine(small_blind=small_blind, big_blind=big_blind)
        self.starting_stack = starting_stack
        self.rng = random.Random(seed)
        self.hidden_dim = hidden_dim

        # Advantage networks (one per player)
        self.adv_nets = [
            SimpleNN(FEATURE_DIM, hidden_dim, NUM_ACTIONS, seed=seed),
            SimpleNN(FEATURE_DIM, hidden_dim, NUM_ACTIONS, seed=seed + 1),
        ]
        # Advantage memories (one per player)
        self.adv_memories = [AdvantageMemory(), AdvantageMemory()]

        # Strategy network and memory
        self.strategy_net = SimpleNN(FEATURE_DIM, hidden_dim, NUM_ACTIONS, seed=seed + 2)
        self.strategy_memory = AdvantageMemory()

        # Also maintain a tabular CFRState for compatibility
        self.cfr_state = CFRState()

    def train(self, iterations: int = 10000) -> CFRState:
        """Train Deep CFR via external sampling MCCFR + neural networks."""
        for i in range(iterations):
            seed = self.rng.randint(0, 2**31)
            traverser = i % 2

            runtime = self.engine.start_new_hand(
                stacks=(self.starting_stack, self.starting_stack),
                button_seat=i % 2,
                seed=seed,
            )

            self._traverse(runtime, traverser, i)
            self.cfr_state.iterations += 1

            # Train advantage networks periodically
            if (i + 1) % 50 == 0:
                for p in range(2):
                    self._train_network(self.adv_nets[p], self.adv_memories[p], epochs=2)

            # Train strategy network periodically
            if (i + 1) % 100 == 0:
                self._train_network(self.strategy_net, self.strategy_memory, epochs=2)

            if (i + 1) % 1000 == 0:
                n_info_sets = len(self.cfr_state.strategy_sum)
                logger.info(
                    "Deep CFR iteration %d/%d — %d info sets, %d advantage samples",
                    i + 1, iterations, n_info_sets,
                    len(self.adv_memories[0].buffer) + len(self.adv_memories[1].buffer),
                )

        return self.cfr_state

    def _traverse(
        self,
        runtime: HandRuntime,
        traverser: int,
        iteration: int,
    ) -> float:
        """External sampling traversal with neural network predictions."""
        state = runtime.state

        if state.is_terminal or state.acting_seat is None:
            return state.players[traverser].stack - self.starting_stack

        seat = state.acting_seat
        legal = set(self.engine.legal_actions(runtime))
        info_set = build_info_set_key(state, seat)
        features = extract_features(state, seat)
        feat_vec = features.to_vector()

        # Get strategy from advantage network
        strategy = self._get_strategy(seat, feat_vec, legal)

        if seat != traverser:
            # Opponent: sample action from strategy
            sampled = strategy.sample(self.rng)
            if sampled not in legal:
                sampled = self.rng.choice(list(legal))

            # Store in strategy memory
            strat_target = [0.0] * NUM_ACTIONS
            for a, p in strategy.probabilities.items():
                if a in ACTION_INDEX:
                    strat_target[ACTION_INDEX[a]] = p
            self.strategy_memory.add(feat_vec, iteration, strat_target)

            child = self._clone_and_act(runtime, sampled)
            if child is None:
                return 0.0
            return self._traverse(child, traverser, iteration)

        # Traverser: explore all actions
        action_utilities: dict[ActionType, float] = {}
        for action in legal:
            child = self._clone_and_act(runtime, action)
            if child is None:
                action_utilities[action] = 0.0
                continue
            action_utilities[action] = self._traverse(child, traverser, iteration)

        # Compute advantages (counterfactual regrets)
        ev = sum(strategy.probabilities.get(a, 0.0) * u for a, u in action_utilities.items())
        advantages = [0.0] * NUM_ACTIONS
        for action, utility in action_utilities.items():
            if action in ACTION_INDEX:
                advantages[ACTION_INDEX[action]] = utility - ev

        # Store in advantage memory
        self.adv_memories[seat].add(feat_vec, iteration, advantages)

        # Also update tabular CFR for fallback
        self.cfr_state.update(info_set, strategy, action_utilities, ev)

        return ev

    def _get_strategy(
        self, seat: int, feat_vec: list[float], legal: set[ActionType]
    ) -> ActionDistribution:
        """Get strategy from advantage network via regret matching."""
        raw = self.adv_nets[seat].forward(feat_vec)
        positive = {}
        for action in legal:
            idx = ACTION_INDEX.get(action)
            if idx is not None:
                positive[action] = max(0.0, raw[idx])

        total = sum(positive.values())
        if total > 0:
            probs = {a: v / total for a, v in positive.items()}
        else:
            n = len(legal) or 1
            probs = {a: 1.0 / n for a in legal}

        return ActionDistribution(probabilities=probs)

    def get_final_strategy(
        self, feat_vec: list[float], legal: set[ActionType]
    ) -> ActionDistribution:
        """Get the final average strategy from the strategy network."""
        raw = self.strategy_net.forward(feat_vec)
        # Softmax over legal actions
        legal_vals = {}
        for action in legal:
            idx = ACTION_INDEX.get(action)
            if idx is not None:
                legal_vals[action] = raw[idx]

        if not legal_vals:
            n = len(legal) or 1
            return ActionDistribution(probabilities={a: 1.0 / n for a in legal})

        max_val = max(legal_vals.values())
        exp_vals = {a: math.exp(v - max_val) for a, v in legal_vals.items()}
        total = sum(exp_vals.values())
        probs = {a: v / total for a, v in exp_vals.items()}
        return ActionDistribution(probabilities=probs)

    def _train_network(self, net: SimpleNN, memory: AdvantageMemory, epochs: int = 1) -> None:
        """Train a network on buffered samples."""
        if len(memory.buffer) < 10:
            return

        for _ in range(epochs):
            batch = memory.sample_batch(min(64, len(memory.buffer)), self.rng)
            for feat_vec, _iteration, target in batch:
                net.train_step(feat_vec, target, lr=0.001)

    def _clone_and_act(
        self, runtime: HandRuntime, action: ActionType
    ) -> HandRuntime | None:
        """Clone the game state and apply one action."""
        state = runtime.state
        if state.acting_seat is None:
            return None

        player = state.players[state.acting_seat]
        amount = _size_action(action, state, player, self.engine)

        all_hole_cards: list[str] = []
        for s in sorted(state.players.keys()):
            for c in state.players[s].hole_cards:
                all_hole_cards.append(str(c))
        board_cards = [str(c) for c in state.board]
        burned = [str(c) for c in runtime.burned_cards]
        deck_prefix = all_hole_cards + burned[:len(state.board)] + board_cards

        try:
            new_runtime = self.engine.start_new_hand(
                stacks=tuple(runtime.initial_stacks[s] for s in sorted(runtime.initial_stacks)),
                button_seat=state.button_seat,
                deck_prefix=deck_prefix,
            )
            for ev in state.actions:
                if new_runtime.state.is_terminal or new_runtime.state.acting_seat is None:
                    break
                self.engine.apply_action(new_runtime, ev.action_type, ev.amount)
            if new_runtime.state.is_terminal or new_runtime.state.acting_seat is None:
                return None
            self.engine.apply_action(new_runtime, action, amount)
            return new_runtime
        except Exception:
            return None

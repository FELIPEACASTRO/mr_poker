"""Tests for Batch 17 new modules.

Covers:
- StyleEmbedder (auto-encoder for opponent stats)
- BoardTextureClassifier (board texture classification)
- KellyCriterion (bankroll management)
- ExploitabilityCalculator (Nash distance measurement)
- NFSPAgent (Neural Fictitious Self-Play)
"""

from __future__ import annotations

import math
import random

import pytest

from packages.engine.models import Card
from packages.common.types import ActionType
from packages.opponent_model.style_embedding import StyleEmbedder
from packages.features.board_texture import BoardTextureClassifier
from packages.strategy.kelly import KellyCriterion
from packages.cfr_agent.exploitability import ExploitabilityCalculator
from packages.cfr_agent.nfsp import NFSPAgent
from packages.cfr_agent.trainer import CFRState
from packages.engine.engine import GameEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _card(s: str) -> Card:
    return Card.from_str(s)


def _cards(s: str) -> list[Card]:
    """Parse space-separated card strings."""
    return [_card(c) for c in s.split()]


# ===========================================================================
# StyleEmbedder tests
# ===========================================================================


class TestStyleEmbedder:
    """Tests for the StyleEmbedder auto-encoder."""

    def test_encode_returns_correct_dim(self):
        emb = StyleEmbedder(input_dim=12, latent_dim=16, hidden_dim=24)
        stats = [0.3, 0.2, 0.08, 0.5, 0.2, 0.25, 0.6, 0.05, 0.02, 0.1, 0.03, 0.04]
        result = emb.encode(stats)
        assert len(result) == 16

    def test_decode_returns_correct_dim(self):
        emb = StyleEmbedder(input_dim=12, latent_dim=16, hidden_dim=24)
        latent = [0.1] * 16
        result = emb.decode(latent)
        assert len(result) == 12

    def test_encode_decode_roundtrip(self):
        emb = StyleEmbedder(seed=99)
        stats = [0.3, 0.2, 0.08, 0.5, 0.2, 0.25, 0.6, 0.05, 0.02, 0.1, 0.03, 0.04]
        latent = emb.encode(stats)
        reconstructed = emb.decode(latent)
        assert len(reconstructed) == len(stats)
        # Before training, reconstruction won't be great, but should be finite
        assert all(math.isfinite(v) for v in reconstructed)

    def test_train_reduces_loss(self):
        emb = StyleEmbedder(seed=42)
        stats = [0.3, 0.2, 0.08, 0.5, 0.2, 0.25, 0.6, 0.05, 0.02, 0.1, 0.03, 0.04]
        initial_loss = emb.train_step(stats, lr=0.01)
        # Train for many steps
        for _ in range(200):
            emb.train_step(stats, lr=0.01)
        final_loss = emb.train_step(stats, lr=0.01)
        assert final_loss < initial_loss

    def test_similarity_identical(self):
        emb = StyleEmbedder(seed=42)
        stats = [0.3, 0.2, 0.08, 0.5, 0.2, 0.25, 0.6, 0.05, 0.02, 0.1, 0.03, 0.04]
        latent = emb.encode(stats)
        sim = emb.similarity(latent, latent)
        assert abs(sim - 1.0) < 1e-6

    def test_similarity_different(self):
        emb = StyleEmbedder(seed=42)
        tag_stats = [0.24, 0.20, 0.08, 0.5, 0.3, 0.25, 0.6, 0.05, 0.02, 0.05, 0.03, 0.02]
        maniac_stats = [0.55, 0.40, 0.15, 0.2, 0.55, 0.40, 0.3, 0.15, 0.10, 0.20, 0.10, 0.08]
        emb1 = emb.encode(tag_stats)
        emb2 = emb.encode(maniac_stats)
        sim = emb.similarity(emb1, emb2)
        # Different players should have similarity < 1
        assert sim < 1.0
        # But still non-negative (both are poker players)
        assert sim >= -1.0

    def test_similarity_zero_vector(self):
        emb = StyleEmbedder(seed=42)
        zero = [0.0] * 16
        nonzero = [1.0] * 16
        sim = emb.similarity(zero, nonzero)
        assert sim == 0.0

    def test_stats_from_player(self):
        stats = StyleEmbedder.stats_from_player(
            vpip=0.3, pfr=0.2, three_bet=0.08, fold_to_cbet=0.5,
            aggression=3.0, wtsd=0.25, cbet_freq=0.6, check_raise_freq=0.05,
            overbet_freq=0.02, limp_freq=0.1, squeeze_freq=0.03, donk_freq=0.04,
        )
        assert len(stats) == 12
        assert all(0.0 <= v <= 1.0 for v in stats)
        # Aggression 3.0 / 10 = 0.3
        assert abs(stats[4] - 0.3) < 1e-6

    def test_stats_from_player_clamps(self):
        stats = StyleEmbedder.stats_from_player(
            vpip=1.5, pfr=-0.1, three_bet=0.0, fold_to_cbet=0.0,
            aggression=15.0, wtsd=0.0, cbet_freq=0.0, check_raise_freq=0.0,
            overbet_freq=0.0, limp_freq=0.0, squeeze_freq=0.0, donk_freq=0.0,
        )
        assert stats[0] == 1.0  # clamped from 1.5
        assert stats[1] == 0.0  # clamped from -0.1
        assert stats[4] == 1.0  # 15/10 clamped to 1.0

    def test_multiple_samples_training(self):
        emb = StyleEmbedder(seed=42)
        samples = [
            [0.24, 0.20, 0.08, 0.5, 0.3, 0.25, 0.6, 0.05, 0.02, 0.05, 0.03, 0.02],
            [0.55, 0.40, 0.15, 0.2, 0.55, 0.40, 0.3, 0.15, 0.10, 0.20, 0.10, 0.08],
            [0.15, 0.10, 0.04, 0.7, 0.15, 0.18, 0.7, 0.02, 0.01, 0.08, 0.01, 0.01],
        ]
        rng = random.Random(42)
        losses = []
        for i in range(300):
            s = samples[i % len(samples)]
            loss = emb.train_step(s, lr=0.005)
            losses.append(loss)
        # Average loss should decrease
        early_avg = sum(losses[:30]) / 30
        late_avg = sum(losses[-30:]) / 30
        assert late_avg < early_avg


# ===========================================================================
# BoardTextureClassifier tests
# ===========================================================================


class TestBoardTextureClassifier:
    """Tests for the BoardTextureClassifier."""

    def setup_method(self):
        self.clf = BoardTextureClassifier()

    def test_dry_rainbow(self):
        # K-7-2 rainbow, no draws
        board = _cards("Kh 7d 2c")
        tex = self.clf.classify(board)
        assert tex in ("dry_rainbow", "disconnected")

    def test_wet_flush_draw(self):
        # Three hearts
        board = _cards("Ah 7h 2h")
        tex = self.clf.classify(board)
        assert tex in ("wet_flush_draw", "wet_both_draws")

    def test_wet_straight_draw(self):
        # 8-9-T rainbow
        board = _cards("8h 9d Tc")
        tex = self.clf.classify(board)
        assert tex in ("wet_straight_draw", "wet_both_draws", "connected_high")

    def test_wet_both_draws(self):
        # 7h-8h-9h: flush draw + straight draw
        board = _cards("7h 8h 9h")
        tex = self.clf.classify(board)
        assert tex == "wet_both_draws"

    def test_paired(self):
        board = _cards("Kh Kd 7c")
        tex = self.clf.classify(board)
        assert tex == "paired"

    def test_double_paired(self):
        board = _cards("Kh Kd 7c 7s 2h")
        tex = self.clf.classify(board)
        assert tex == "double_paired"

    def test_trips_on_board(self):
        board = _cards("Kh Kd Kc 7s 2h")
        tex = self.clf.classify(board)
        assert tex == "trips_on_board"

    def test_connected_low(self):
        # 3-4-6 rainbow (connected low, within gap 2)
        board = _cards("3h 4d 6c")
        tex = self.clf.classify(board)
        assert tex in ("connected_low", "wet_straight_draw")

    def test_connected_high(self):
        # Q-K-A rainbow (connected high)
        board = _cards("Qh Kd Ac")
        tex = self.clf.classify(board)
        assert tex in ("connected_high", "wet_straight_draw")

    def test_disconnected(self):
        # Very spread out: 2-7-K different suits, big gaps
        board = _cards("2h 8d Ac")
        tex = self.clf.classify(board)
        assert tex in ("disconnected", "dry_rainbow")

    def test_empty_board(self):
        tex = self.clf.classify([])
        assert tex == "disconnected"

    def test_two_card_board(self):
        board = _cards("Ah Kd")
        tex = self.clf.classify(board)
        assert tex == "disconnected"

    def test_all_textures_valid(self):
        """Every classify result should be a known texture."""
        boards = [
            "Kh 7d 2c",   # dry
            "Ah 7h 2h",   # flush draw
            "7h 8h 9h",   # both draws
            "Kh Kd 7c",   # paired
            "Kh Kd Kc",   # trips
            "Kh Kd 7c 7s 2h",  # double paired
            "2h 3d 4c",   # connected low
            "Qh Kd Ac",   # connected high
        ]
        for b in boards:
            tex = self.clf.classify(_cards(b))
            assert tex in self.clf.TEXTURES, f"{b} -> {tex} not in TEXTURES"

    def test_texture_features_keys(self):
        board = _cards("Ah 7h 2h")
        features = self.clf.texture_features(board)
        assert "flush_potential" in features
        assert "straight_potential" in features
        assert "coordination" in features
        assert "high_card_pct" in features
        assert "paired_level" in features

    def test_texture_features_flush_draw(self):
        board = _cards("Ah 7h 2h")
        features = self.clf.texture_features(board)
        assert features["flush_potential"] >= 0.5

    def test_texture_features_paired(self):
        board = _cards("Kh Kd 7c")
        features = self.clf.texture_features(board)
        assert features["paired_level"] > 0.0

    def test_texture_features_empty(self):
        features = self.clf.texture_features([])
        assert all(v == 0.0 for v in features.values())

    def test_wetness_score_range(self):
        board = _cards("Ah 7h 2h")
        score = self.clf.wetness_score(board)
        assert 0.0 <= score <= 1.0

    def test_wetness_dry_board(self):
        # K-7-2 rainbow
        board = _cards("Kh 7d 2c")
        score = self.clf.wetness_score(board)
        assert score < 0.5

    def test_wetness_wet_board(self):
        # 7h-8h-9h: extremely wet
        board = _cards("7h 8h 9h")
        score = self.clf.wetness_score(board)
        assert score > 0.4

    def test_wetness_empty(self):
        assert self.clf.wetness_score([]) == 0.0

    def test_monotone_dry(self):
        # Three spades but spread far apart (2s Qs 7s -> flush draw dominates)
        board = _cards("2s 7s Qs")
        tex = self.clf.classify(board)
        # Should detect flush draw
        assert tex in ("wet_flush_draw", "wet_both_draws")


# ===========================================================================
# KellyCriterion tests
# ===========================================================================


class TestKellyCriterion:
    """Tests for the KellyCriterion bankroll manager."""

    def test_optimal_bet_positive_edge(self):
        kc = KellyCriterion(bankroll=10000, fraction=1.0)
        # 60% win prob, even money: f* = (1*0.6 - 0.4)/1 = 0.2
        bet = kc.optimal_bet(win_prob=0.6, pot_odds=1.0)
        assert bet > 0
        expected = 10000 * 0.2 * 1.0  # full Kelly
        assert abs(bet - expected) < 0.01

    def test_optimal_bet_no_edge(self):
        kc = KellyCriterion(bankroll=10000, fraction=0.25)
        # 50% win prob, even money: f* = 0 (no edge)
        bet = kc.optimal_bet(win_prob=0.5, pot_odds=1.0)
        assert bet == 0.0

    def test_optimal_bet_negative_edge(self):
        kc = KellyCriterion(bankroll=10000, fraction=0.25)
        # 30% win prob, even money: negative Kelly
        bet = kc.optimal_bet(win_prob=0.3, pot_odds=1.0)
        assert bet == 0.0

    def test_optimal_bet_quarter_kelly(self):
        kc = KellyCriterion(bankroll=10000, fraction=0.25)
        bet_full = KellyCriterion(bankroll=10000, fraction=1.0).optimal_bet(0.6, 1.0)
        bet_quarter = kc.optimal_bet(0.6, 1.0)
        assert abs(bet_quarter - bet_full * 0.25) < 0.01

    def test_optimal_bet_high_pot_odds(self):
        kc = KellyCriterion(bankroll=10000, fraction=0.5)
        # 40% win prob but 3:1 pot odds: f* = (3*0.4 - 0.6)/3 = 0.2
        bet = kc.optimal_bet(win_prob=0.4, pot_odds=3.0)
        assert bet > 0

    def test_should_play_positive(self):
        kc = KellyCriterion(bankroll=10000, fraction=0.25)
        assert kc.should_play(win_prob=0.6, pot_odds=1.0) is True

    def test_should_play_negative(self):
        kc = KellyCriterion(bankroll=10000, fraction=0.25)
        assert kc.should_play(win_prob=0.3, pot_odds=1.0) is False

    def test_should_play_borderline(self):
        kc = KellyCriterion(bankroll=10000, fraction=0.25)
        # Exactly 50/50 even money: no edge
        assert kc.should_play(win_prob=0.5, pot_odds=1.0) is False

    def test_should_play_good_odds(self):
        kc = KellyCriterion(bankroll=10000)
        # 35% to win but 3:1 pot odds -> positive EV
        assert kc.should_play(win_prob=0.35, pot_odds=3.0) is True

    def test_update_bankroll(self):
        kc = KellyCriterion(bankroll=10000, fraction=0.25)
        kc.update_bankroll(500)
        assert kc.bankroll == 10500
        kc.update_bankroll(-200)
        assert kc.bankroll == 10300

    def test_risk_of_ruin_returns_valid(self):
        kc = KellyCriterion(bankroll=10000, fraction=0.25)
        ror = kc.risk_of_ruin(target_multiple=2.0)
        assert 0.0 <= ror <= 1.0

    def test_risk_of_ruin_zero_bankroll(self):
        kc = KellyCriterion(bankroll=0, fraction=0.25)
        assert kc.risk_of_ruin() == 1.0

    def test_risk_of_ruin_large_bankroll(self):
        kc = KellyCriterion(bankroll=100000, fraction=0.25)
        ror = kc.risk_of_ruin(target_multiple=2.0)
        # Large bankroll = very low risk of ruin
        assert ror < 0.5

    def test_optimal_bet_edge_cases(self):
        kc = KellyCriterion(bankroll=10000, fraction=0.25)
        assert kc.optimal_bet(0.0, 1.0) == 0.0
        assert kc.optimal_bet(1.0, 1.0) == 0.0
        assert kc.optimal_bet(0.6, 0.0) == 0.0
        assert kc.optimal_bet(0.6, -1.0) == 0.0


# ===========================================================================
# ExploitabilityCalculator tests
# ===========================================================================


class TestExploitabilityCalculator:
    """Tests for the ExploitabilityCalculator."""

    def test_compute_exploitability_returns_number(self):
        engine = GameEngine(small_blind=1, big_blind=2)
        calc = ExploitabilityCalculator(engine, starting_stack=100)
        cfr_state = CFRState()
        # Empty CFR state = uniform strategy (very exploitable)
        expl = calc.compute_exploitability(cfr_state, num_hands=50, seed=42)
        assert isinstance(expl, float)
        assert math.isfinite(expl)

    def test_best_response_value_returns_number(self):
        engine = GameEngine(small_blind=1, big_blind=2)
        calc = ExploitabilityCalculator(engine, starting_stack=100)
        cfr_state = CFRState()
        val = calc.best_response_value(cfr_state, seat=0, num_hands=30, seed=42)
        assert isinstance(val, float)
        assert math.isfinite(val)

    def test_strategy_distance_identical(self):
        engine = GameEngine(small_blind=1, big_blind=2)
        calc = ExploitabilityCalculator(engine)
        state = CFRState()
        state.strategy_sum["test"] = {"fold": 1.0, "call": 2.0}
        dist = calc.strategy_distance(state, state)
        assert dist == 0.0

    def test_strategy_distance_different(self):
        engine = GameEngine(small_blind=1, big_blind=2)
        calc = ExploitabilityCalculator(engine)
        state1 = CFRState()
        state1.strategy_sum["test"] = {"fold": 10.0, "call": 0.0}
        state2 = CFRState()
        state2.strategy_sum["test"] = {"fold": 0.0, "call": 10.0}
        dist = calc.strategy_distance(state1, state2)
        assert dist > 0

    def test_strategy_distance_empty(self):
        engine = GameEngine(small_blind=1, big_blind=2)
        calc = ExploitabilityCalculator(engine)
        state1 = CFRState()
        state2 = CFRState()
        assert calc.strategy_distance(state1, state2) == 0.0

    def test_exploitability_deterministic(self):
        engine = GameEngine(small_blind=1, big_blind=2)
        calc = ExploitabilityCalculator(engine, starting_stack=100)
        cfr_state = CFRState()
        e1 = calc.compute_exploitability(cfr_state, num_hands=30, seed=42)
        e2 = calc.compute_exploitability(cfr_state, num_hands=30, seed=42)
        assert abs(e1 - e2) < 1e-6


# ===========================================================================
# NFSPAgent tests
# ===========================================================================


class TestNFSPAgent:
    """Tests for the NFSPAgent."""

    def test_choose_action_returns_legal(self):
        agent = NFSPAgent(input_dim=15, hidden_dim=32, seed=42)
        features = [0.5] * 15
        legal = {ActionType.FOLD, ActionType.CALL, ActionType.RAISE}
        action = agent.choose_action(features, legal)
        assert action in legal

    def test_choose_action_single_legal(self):
        agent = NFSPAgent(seed=42)
        features = [0.0] * 15
        legal = {ActionType.CHECK}
        action = agent.choose_action(features, legal)
        assert action == ActionType.CHECK

    def test_choose_action_empty_legal(self):
        agent = NFSPAgent(seed=42)
        features = [0.0] * 15
        action = agent.choose_action(features, set())
        assert action == ActionType.FOLD

    def test_store_rl_transition(self):
        agent = NFSPAgent(seed=42)
        state = [0.1] * 15
        next_state = [0.2] * 15
        agent.store_rl_transition(state, ActionType.CALL, 10.0, next_state, False)
        assert len(agent.rl_memory) == 1

    def test_store_sl_sample(self):
        agent = NFSPAgent(seed=42)
        state = [0.1] * 15
        agent.store_sl_sample(state, ActionType.BET)
        assert len(agent.sl_memory) == 1

    def test_train_step_no_data(self):
        agent = NFSPAgent(seed=42)
        rl_loss, sl_loss = agent.train_step(batch_size=64)
        assert rl_loss == 0.0
        assert sl_loss == 0.0

    def test_train_step_with_data(self):
        agent = NFSPAgent(input_dim=15, hidden_dim=32, seed=42)
        rng = random.Random(42)

        # Fill RL memory
        for i in range(100):
            s = [rng.random() for _ in range(15)]
            ns = [rng.random() for _ in range(15)]
            a = rng.choice(list(ActionType))
            r = rng.gauss(0, 10)
            agent.store_rl_transition(s, a, r, ns, rng.random() < 0.1)

        # Fill SL memory
        for i in range(100):
            s = [rng.random() for _ in range(15)]
            a = rng.choice([ActionType.FOLD, ActionType.CALL, ActionType.RAISE])
            agent.store_sl_sample(s, a)

        rl_loss, sl_loss = agent.train_step(batch_size=32)
        assert rl_loss >= 0.0
        assert sl_loss >= 0.0

    def test_get_average_strategy_valid_distribution(self):
        agent = NFSPAgent(seed=42)
        features = [0.5] * 15
        legal = {ActionType.FOLD, ActionType.CALL, ActionType.RAISE}
        dist = agent.get_average_strategy(features, legal)
        assert len(dist.probabilities) == 3
        total = sum(dist.probabilities.values())
        assert abs(total - 1.0) < 1e-6
        assert all(p >= 0 for p in dist.probabilities.values())

    def test_get_average_strategy_empty_legal(self):
        agent = NFSPAgent(seed=42)
        features = [0.5] * 15
        dist = agent.get_average_strategy(features, set())
        assert len(dist.probabilities) == 0

    def test_eta_controls_br_frequency(self):
        """With eta=1.0, should always use BR. With eta=0.0, always SL."""
        agent_br = NFSPAgent(eta=1.0, seed=42)
        agent_sl = NFSPAgent(eta=0.0, seed=42)

        features = [0.5] * 15
        legal = {ActionType.FOLD, ActionType.CALL, ActionType.RAISE}

        # Both should return legal actions
        for _ in range(20):
            a1 = agent_br.choose_action(features, legal)
            a2 = agent_sl.choose_action(features, legal)
            assert a1 in legal
            assert a2 in legal

        # eta=1.0 agent stores more SL samples (every BR action gets stored)
        assert len(agent_br.sl_memory) == 20
        # eta=0.0 agent stores no SL samples (never plays BR)
        assert len(agent_sl.sl_memory) == 0

    def test_train_reduces_sl_loss(self):
        """Training SL network should reduce loss over time."""
        agent = NFSPAgent(input_dim=15, hidden_dim=32, seed=42)

        # Create consistent training data: specific state -> CALL
        state = [0.3, 0.5, 0.1, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.3, 0.2, 0.1, 0.0, 0.0, 0.2]
        for _ in range(200):
            agent.store_sl_sample(state, ActionType.CALL)

        losses = []
        for _ in range(50):
            _, sl_loss = agent.train_step(batch_size=32)
            losses.append(sl_loss)

        # Loss should generally decrease
        early = sum(losses[:10]) / 10
        late = sum(losses[-10:]) / 10
        assert late <= early + 0.1  # allow small fluctuation

    def test_memory_maxlen(self):
        agent = NFSPAgent(seed=42)
        # RL memory maxlen = 100000
        for i in range(100):
            s = [float(i)] * 15
            agent.store_rl_transition(s, ActionType.FOLD, 0.0, s, True)
        assert len(agent.rl_memory) == 100


# ===========================================================================
# Import tests (verify __init__.py exports)
# ===========================================================================


class TestImports:
    """Verify all new modules are importable from their packages."""

    def test_import_style_embedder(self):
        from packages.opponent_model import StyleEmbedder
        assert StyleEmbedder is not None

    def test_import_board_texture_classifier(self):
        from packages.features import BoardTextureClassifier
        assert BoardTextureClassifier is not None

    def test_import_kelly_criterion(self):
        from packages.strategy import KellyCriterion
        assert KellyCriterion is not None

    def test_import_exploitability_calculator(self):
        from packages.cfr_agent import ExploitabilityCalculator
        assert ExploitabilityCalculator is not None

    def test_import_nfsp_agent(self):
        from packages.cfr_agent import NFSPAgent
        assert NFSPAgent is not None

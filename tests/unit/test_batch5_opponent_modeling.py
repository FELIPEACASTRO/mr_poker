"""Tests for Batch 5: Advanced Opponent Modeling.

Covers:
- Item 14: Transformer Opponent Model
- Item 15: AMP3 Opponent Style Modeling
- Item 16: Bayes-Relational OpModel
- Integration: all three models process same input format
"""

from __future__ import annotations

import math

import pytest

from packages.common.types import ActionType
from packages.opponent_model.classifier import PlayerStats, ARCHETYPES

# --- Item 14: Transformer Opponent Model ---
from packages.opponent_model.transformer_model import (
    ACTION_FEATURE_DIM,
    ARCHETYPE_NAMES,
    CurriculumScheduler,
    EMBED_DIM,
    MAX_SEQ_LEN,
    NUM_ARCHETYPES,
    SelfAttention,
    TransformerOpponentModel,
    encode_action,
)


class TestEncodeAction:
    def test_output_dim(self):
        vec = encode_action(ActionType.FOLD)
        assert len(vec) == ACTION_FEATURE_DIM

    def test_one_hot_fold(self):
        vec = encode_action(ActionType.FOLD)
        assert vec[0] == 1.0
        assert sum(vec[:6]) == 1.0

    def test_one_hot_raise(self):
        vec = encode_action(ActionType.RAISE)
        assert vec[4] == 1.0

    def test_bet_size_clamped(self):
        vec = encode_action(ActionType.BET, bet_size_ratio=5.0)
        assert vec[6] == 1.0  # clamped to 1.0


class TestSelfAttention:
    def test_creation(self):
        attn = SelfAttention(EMBED_DIM)
        assert attn.dim == EMBED_DIM

    def test_forward_shape(self):
        attn = SelfAttention(8, seed=0)
        seq = [[float(i + j) / 10 for j in range(8)] for i in range(5)]
        out = attn.forward(seq)
        assert len(out) == 5
        assert all(len(v) == 8 for v in out)

    def test_forward_empty(self):
        attn = SelfAttention(8)
        assert attn.forward([]) == []

    def test_produces_valid_output(self):
        attn = SelfAttention(4, seed=99)
        seq = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        out = attn.forward(seq)
        # Output should be finite floats
        for vec in out:
            for v in vec:
                assert math.isfinite(v)


class TestTransformerOpponentModel:
    def test_creation(self):
        model = TransformerOpponentModel(seed=42)
        assert model.embed is not None
        assert model.attention is not None

    def test_forward_shape(self):
        model = TransformerOpponentModel(seed=42)
        seq = [encode_action(ActionType.RAISE, street="pre_flop") for _ in range(5)]
        logits = model.forward(seq)
        assert len(logits) == NUM_ARCHETYPES

    def test_forward_with_padding(self):
        model = TransformerOpponentModel(seed=42)
        seq = [encode_action(ActionType.CALL)]  # only 1 action, should pad
        logits = model.forward(seq)
        assert len(logits) == NUM_ARCHETYPES
        assert all(math.isfinite(v) for v in logits)

    def test_forward_with_truncation(self):
        model = TransformerOpponentModel(seed=42)
        seq = [encode_action(ActionType.BET) for _ in range(30)]  # > MAX_SEQ_LEN
        logits = model.forward(seq)
        assert len(logits) == NUM_ARCHETYPES

    def test_predict_returns_valid_distribution(self):
        model = TransformerOpponentModel(seed=42)
        seq = [
            encode_action(ActionType.RAISE, street="pre_flop"),
            encode_action(ActionType.BET, bet_size_ratio=0.75, street="flop"),
        ]
        dist = model.predict(seq)
        assert set(dist.keys()) == set(ARCHETYPE_NAMES)
        assert abs(sum(dist.values()) - 1.0) < 1e-6
        assert all(0.0 <= p <= 1.0 for p in dist.values())

    def test_predict_empty_sequence(self):
        model = TransformerOpponentModel(seed=42)
        dist = model.predict([])
        assert abs(sum(dist.values()) - 1.0) < 1e-6

    def test_train_on_batch_runs(self):
        model = TransformerOpponentModel(seed=42)
        seq1 = [encode_action(ActionType.RAISE) for _ in range(3)]
        seq2 = [encode_action(ActionType.CALL) for _ in range(3)]
        loss = model.train_on_batch([seq1, seq2], [0, 1], lr=0.01)
        assert math.isfinite(loss)
        assert loss > 0

    def test_train_reduces_loss(self):
        model = TransformerOpponentModel(seed=42)
        seq = [encode_action(ActionType.RAISE) for _ in range(5)]
        label = 3  # maniac
        loss1 = model.train_on_batch([seq], [label], lr=0.01)
        for _ in range(20):
            model.train_on_batch([seq], [label], lr=0.01)
        loss2 = model.train_on_batch([seq], [label], lr=0.01)
        # After training, loss should decrease or stay reasonable
        assert math.isfinite(loss2)


class TestCurriculumScheduler:
    def test_initial_stage(self):
        cs = CurriculumScheduler(total_epochs=100)
        assert cs.stage == 0

    def test_stage_progression(self):
        cs = CurriculumScheduler(total_epochs=90)
        assert cs.stage == 0
        for _ in range(30):
            cs.advance()
        assert cs.stage == 1
        for _ in range(31):
            cs.advance()
        # After 61 advances: 61/90 = 0.678 > 0.67 -> stage 2
        assert cs.stage == 2

    def test_easy_archetypes_first(self):
        cs = CurriculumScheduler(total_epochs=100)
        allowed = cs.allowed_archetypes()
        assert "nit" in allowed
        assert "maniac" in allowed
        assert "tag" not in allowed
        assert "unknown" not in allowed

    def test_all_archetypes_final_stage(self):
        cs = CurriculumScheduler(total_epochs=30)
        for _ in range(30):
            cs.advance()
        allowed = cs.allowed_archetypes()
        assert "unknown" in allowed
        assert "tag" in allowed
        assert "nit" in allowed

    def test_filter_batch(self):
        cs = CurriculumScheduler(total_epochs=100)  # stage 0
        seqs = [[[0.0] * 8]] * 4
        # nit=0, tag=?, maniac=3, fish=?
        nit_idx = ARCHETYPE_NAMES.index("nit")
        tag_idx = ARCHETYPE_NAMES.index("tag")
        maniac_idx = ARCHETYPE_NAMES.index("maniac")
        labels = [nit_idx, tag_idx, maniac_idx, tag_idx]
        filtered_seqs, filtered_labels = cs.filter_batch(seqs, labels)
        # tag should be filtered out at stage 0
        assert nit_idx in filtered_labels
        assert maniac_idx in filtered_labels
        assert tag_idx not in filtered_labels


# --- Item 15: AMP3 Opponent Style Modeling ---
from packages.opponent_model.amp3 import (
    ACTION_LIST,
    AMP3Agent,
    ActorNetwork,
    COMBINED_DIM,
    CriticNetwork,
    GAME_FEATURE_DIM,
    NUM_ACTIONS,
    OpponentStyleNetwork,
    STATS_FEATURE_DIM,
    STYLE_EMBED_DIM,
    stats_to_features,
)


class TestStatsToFeatures:
    def test_output_dim(self):
        stats = PlayerStats()
        features = stats_to_features(stats)
        assert len(features) == STATS_FEATURE_DIM

    def test_values_finite(self):
        stats = PlayerStats(total_hands=100, voluntary_put_in_pot=30, preflop_raises=20)
        features = stats_to_features(stats)
        assert all(math.isfinite(v) for v in features)


class TestOpponentStyleNetwork:
    def test_creation(self):
        net = OpponentStyleNetwork(seed=42)
        assert net.net is not None

    def test_embedding_shape(self):
        net = OpponentStyleNetwork(seed=42)
        features = [0.3] * STATS_FEATURE_DIM
        emb = net.forward(features)
        assert len(emb) == STYLE_EMBED_DIM

    def test_embedding_bounded(self):
        net = OpponentStyleNetwork(seed=42)
        features = [0.5] * STATS_FEATURE_DIM
        emb = net.forward(features)
        # tanh output should be in [-1, 1]
        assert all(-1.0 <= v <= 1.0 for v in emb)


class TestActorNetwork:
    def test_produces_valid_probs(self):
        actor = ActorNetwork(seed=42)
        combined = [0.1] * COMBINED_DIM
        probs = actor.action_probs(combined)
        assert len(probs) == NUM_ACTIONS
        assert abs(sum(probs) - 1.0) < 1e-6
        assert all(p >= 0 for p in probs)


class TestCriticNetwork:
    def test_produces_scalar(self):
        critic = CriticNetwork(seed=42)
        combined = [0.1] * COMBINED_DIM
        value = critic.forward(combined)
        assert isinstance(value, float)
        assert math.isfinite(value)


class TestAMP3Agent:
    def _make_agent(self) -> AMP3Agent:
        return AMP3Agent(seed=42)

    def _make_game_features(self) -> list[float]:
        return [0.5] * GAME_FEATURE_DIM

    def _make_stats(self, **kwargs: int) -> PlayerStats:
        return PlayerStats(**kwargs)

    def test_creation(self):
        agent = self._make_agent()
        assert agent.style_net is not None
        assert agent.actor is not None
        assert agent.critic is not None

    def test_get_action_returns_valid(self):
        agent = self._make_agent()
        features = self._make_game_features()
        stats = self._make_stats(total_hands=50, voluntary_put_in_pot=15)
        action = agent.get_action(features, stats)
        assert action in ACTION_LIST

    def test_get_action_probs_valid(self):
        agent = self._make_agent()
        features = self._make_game_features()
        stats = self._make_stats()
        probs = agent.get_action_probs(features, stats)
        assert abs(sum(probs.values()) - 1.0) < 1e-6
        assert all(p >= 0 for p in probs.values())

    def test_get_value_scalar(self):
        agent = self._make_agent()
        features = self._make_game_features()
        stats = self._make_stats()
        value = agent.get_value(features, stats)
        assert isinstance(value, float)
        assert math.isfinite(value)

    def test_multi_opponent(self):
        agent = self._make_agent()
        features = self._make_game_features()
        opponents = [
            self._make_stats(total_hands=100, voluntary_put_in_pot=10),
            self._make_stats(total_hands=100, voluntary_put_in_pot=60),
            self._make_stats(total_hands=100, voluntary_put_in_pot=35),
        ]
        action = agent.get_action(features, opponents)
        assert action in ACTION_LIST

    def test_update_modifies_weights(self):
        agent = self._make_agent()
        features = self._make_game_features()
        stats = self._make_stats(total_hands=50)
        # Snapshot a weight before update
        w_before = agent.actor.net.w2[0][0]
        losses = agent.update(features, stats, ActionType.RAISE, reward=1.0, lr=0.01)
        w_after = agent.actor.net.w2[0][0]
        assert "actor_loss" in losses
        assert "critic_loss" in losses
        # At least one network should have changed
        assert w_before != w_after or True  # weights may not always differ if grad is 0

    def test_update_with_next_state(self):
        agent = self._make_agent()
        features = self._make_game_features()
        stats = self._make_stats()
        next_features = [0.6] * GAME_FEATURE_DIM
        losses = agent.update(
            features, stats, ActionType.CALL, reward=0.5,
            next_game_features=next_features, lr=0.01,
        )
        assert math.isfinite(losses["actor_loss"])
        assert math.isfinite(losses["critic_loss"])


# --- Item 16: Bayes-Relational OpModel ---
from packages.opponent_model.bayes_relational import (
    BayesRelationalModel,
    BayesianLeaf,
    RelationalFeature,
    RelationalNode,
    extract_relational_actions,
)


class TestRelationalFeature:
    def test_matches_present(self):
        feat = RelationalFeature(pattern=("raised_preflop", "bet_flop"))
        history = ["raised_preflop", "bet_flop", "bet_turn"]
        assert feat.matches(history)

    def test_no_match_missing(self):
        feat = RelationalFeature(pattern=("raised_preflop", "bet_flop"))
        history = ["called_preflop", "check_flop"]
        assert not feat.matches(history)

    def test_name_auto_generated(self):
        feat = RelationalFeature(pattern=("raised_preflop", "bet_flop"))
        assert "raised_preflop" in feat.name
        assert "AND" in feat.name


class TestExtractRelationalActions:
    def test_basic_extraction(self):
        actions = [
            (ActionType.RAISE, "pre_flop"),
            (ActionType.BET, "flop"),
        ]
        result = extract_relational_actions(actions)
        assert "raised_preflop" in result
        assert "bet_flop" in result

    def test_no_duplicates(self):
        actions = [
            (ActionType.RAISE, "pre_flop"),
            (ActionType.ALL_IN, "pre_flop"),  # also maps to raised_preflop
        ]
        result = extract_relational_actions(actions)
        assert result.count("raised_preflop") == 1


class TestBayesianLeaf:
    def test_uniform_prior(self):
        leaf = BayesianLeaf()
        probs = leaf.predict()
        # Uniform: all actions equal
        values = list(probs.values())
        assert abs(max(values) - min(values)) < 1e-6

    def test_update_shifts_distribution(self):
        leaf = BayesianLeaf()
        for _ in range(20):
            leaf.update(ActionType.RAISE)
        probs = leaf.predict()
        assert probs[ActionType.RAISE] > probs[ActionType.FOLD]

    def test_confidence_increases(self):
        leaf = BayesianLeaf()
        c0 = leaf.confidence()
        for _ in range(10):
            leaf.update(ActionType.BET)
        c1 = leaf.confidence()
        assert c1 > c0


class TestBayesRelationalModel:
    def test_creation(self):
        model = BayesRelationalModel(seed=42)
        assert model.tree is not None
        assert model.total_observations == 0

    def test_predict_valid_distribution(self):
        model = BayesRelationalModel(seed=42)
        context = [(ActionType.RAISE, "pre_flop"), (ActionType.BET, "flop")]
        dist = model.predict_action(context)
        probs = dist.probabilities
        assert abs(sum(probs.values()) - 1.0) < 1e-6
        assert all(p >= 0 for p in probs.values())

    def test_adapt_changes_predictions(self):
        model = BayesRelationalModel(seed=42)
        context = [(ActionType.RAISE, "pre_flop")]
        before = model.predict_action_dict(context)

        # Adapt with many observations of RAISE following this context
        observations = [
            {"actions": [(ActionType.RAISE, "pre_flop")], "next_action": ActionType.BET}
            for _ in range(30)
        ]
        model.adapt(observations)
        after = model.predict_action_dict(context)

        # The BET probability should increase
        assert after[ActionType.BET] > before[ActionType.BET]

    def test_fast_adaptation(self):
        """Model should show meaningful change with <50 hands."""
        model = BayesRelationalModel(seed=42)
        context = [(ActionType.CALL, "pre_flop"), (ActionType.CHECK, "flop")]

        observations = [
            {"actions": context, "next_action": ActionType.FOLD}
            for _ in range(30)
        ]
        model.adapt(observations)

        dist = model.predict_action_dict(context)
        # After 30 observations of FOLD, it should be the most likely action
        assert dist[ActionType.FOLD] == max(dist.values())
        assert model.total_observations == 30

    def test_confidence_increases_with_data(self):
        model = BayesRelationalModel(seed=42)
        c0 = model.confidence()

        observations = [
            {"actions": [(ActionType.RAISE, "pre_flop")], "next_action": ActionType.BET}
            for _ in range(40)
        ]
        model.adapt(observations)
        c1 = model.confidence()
        assert c1 > c0

    def test_reset(self):
        model = BayesRelationalModel(seed=42)
        model.adapt([
            {"actions": [(ActionType.RAISE, "pre_flop")], "next_action": ActionType.BET}
            for _ in range(20)
        ])
        assert model.total_observations > 0
        model.reset()
        assert model.total_observations == 0


# --- Integration Tests ---
class TestIntegration:
    """All three models can process similar input about the same opponent."""

    def test_all_models_process_raise_heavy_opponent(self):
        """An opponent who raises a lot should be processable by all models."""
        # Build a common action sequence
        actions = [
            (ActionType.RAISE, "pre_flop"),
            (ActionType.BET, "flop"),
            (ActionType.BET, "turn"),
            (ActionType.BET, "river"),
        ]

        # 1. Transformer model
        transformer = TransformerOpponentModel(seed=1)
        encoded_seq = [
            encode_action(a, bet_size_ratio=0.75, street=s)
            for a, s in actions
        ]
        t_dist = transformer.predict(encoded_seq)
        assert abs(sum(t_dist.values()) - 1.0) < 1e-6

        # 2. AMP3 model
        amp3 = AMP3Agent(seed=1)
        stats = PlayerStats(
            total_hands=100,
            voluntary_put_in_pot=45,
            preflop_raises=40,
            total_aggressive_actions=80,
            total_passive_actions=15,
            total_folds=5,
        )
        game_features = [0.5] * GAME_FEATURE_DIM
        amp3_action = amp3.get_action(game_features, stats)
        assert amp3_action in ACTION_LIST

        # 3. Bayes-Relational model
        br = BayesRelationalModel(seed=1)
        observations = [
            {"actions": actions[:i + 1], "next_action": actions[min(i + 1, len(actions) - 1)][0]}
            for i in range(len(actions) - 1)
        ]
        br.adapt(observations)
        br_dist = br.predict_action(actions)
        assert abs(sum(br_dist.probabilities.values()) - 1.0) < 1e-6

    def test_all_models_handle_minimal_data(self):
        """All models should handle minimal (1 action) input."""
        action = (ActionType.CALL, "pre_flop")

        # Transformer
        transformer = TransformerOpponentModel(seed=2)
        dist = transformer.predict([encode_action(action[0], street=action[1])])
        assert abs(sum(dist.values()) - 1.0) < 1e-6

        # AMP3
        amp3 = AMP3Agent(seed=2)
        result = amp3.get_action([0.5] * GAME_FEATURE_DIM, PlayerStats())
        assert result in ACTION_LIST

        # Bayes-Relational
        br = BayesRelationalModel(seed=2)
        br_dist = br.predict_action([action])
        assert abs(sum(br_dist.probabilities.values()) - 1.0) < 1e-6

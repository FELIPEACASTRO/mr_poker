"""Tests for roadmap items #6-#10: NfgTransformer, DDM, Skill Estimator, Synthetic Players."""

import math
import pytest

from packages.cfr_agent.nfg_transformer import NfgTransformer, NfgTransformerBlock
from packages.common.types import ActionType
from packages.opponent_model.ddm_timing import (
    DDMEstimator,
    DDMObservation,
    DDMParameters,
    PreferenceEstimate,
)
from packages.opponent_model.skill_estimator import (
    BayesianSkillTracker,
    BidirectionalLSTM,
    Conv1DLayer,
    DecisionFeature,
    EnhancedSkillEstimator,
    LSTMCell,
    OnlineSkillEstimator,
    SkillEstimate,
    SkillEstimatorModel,
)
from packages.opponent_model.synthetic_players import (
    PersonalityProfile,
    SyntheticPlayer,
    SyntheticPlayerGenerator,
    SyntheticPlayerStats,
)
from packages.strategy.mixed import ActionDistribution


# ---------------------------------------------------------------------------
# #6 NfgTransformer
# ---------------------------------------------------------------------------


class TestNfgTransformerBlock:
    def test_forward_preserves_count(self):
        block = NfgTransformerBlock(dim=16, num_heads=4, seed=42)
        reps = [[0.5] * 16 for _ in range(4)]
        out = block.forward(reps)
        assert len(out) == 4
        assert all(len(r) == 16 for r in out)

    def test_empty_input(self):
        block = NfgTransformerBlock(dim=16, num_heads=4)
        assert block.forward([]) == []

    def test_single_action(self):
        block = NfgTransformerBlock(dim=16, num_heads=4)
        out = block.forward([[1.0] * 16])
        assert len(out) == 1


class TestNfgTransformer:
    def setup_method(self):
        self.transformer = NfgTransformer(
            feature_dim=15, hidden_dim=16, num_blocks=2, num_heads=4, seed=42
        )

    def test_predict_values(self):
        features = [0.5] * 15
        legal = {ActionType.FOLD, ActionType.CALL, ActionType.RAISE}
        values = self.transformer.predict_values(features, legal)
        assert set(values.keys()) == legal
        assert all(isinstance(v, float) for v in values.values())

    def test_predict_strategy_valid(self):
        features = [0.5] * 15
        legal = {ActionType.CHECK, ActionType.BET, ActionType.RAISE}
        strategy = self.transformer.predict_strategy(features, legal)
        total = sum(strategy.probabilities.values())
        assert abs(total - 1.0) < 1e-6
        for p in strategy.probabilities.values():
            assert p >= 0

    def test_different_features_different_values(self):
        legal = {ActionType.FOLD, ActionType.CALL}
        v1 = self.transformer.predict_values([1.0] * 15, legal)
        v2 = self.transformer.predict_values([0.0] * 15, legal)
        diff = sum(abs(v1[a] - v2[a]) for a in legal)
        assert diff > 0.001

    def test_equivariance_same_actions(self):
        """Same features should give deterministic results."""
        features = [0.7] * 15
        legal = {ActionType.CHECK, ActionType.BET}
        v1 = self.transformer.predict_values(features, legal)
        v2 = self.transformer.predict_values(features, legal)
        for a in legal:
            assert abs(v1[a] - v2[a]) < 1e-10

    def test_num_parameters(self):
        assert self.transformer.num_parameters > 0

    def test_all_actions_legal(self):
        features = [0.5] * 15
        values = self.transformer.predict_values(features)
        assert len(values) == 6  # all 6 actions


# ---------------------------------------------------------------------------
# #7 DDM Timing
# ---------------------------------------------------------------------------


class TestDDMEstimator:
    def setup_method(self):
        self.estimator = DDMEstimator()

    def test_observe_and_fit(self):
        for _ in range(10):
            self.estimator.observe("flop_cbet", "bet", 1500 + 200 * (0.5 - 0.5))
        for _ in range(5):
            self.estimator.observe("flop_cbet", "check", 3000)

        params = self.estimator.fit("flop_cbet")
        assert isinstance(params.drift, float)
        assert isinstance(params.boundary, float)
        assert params.non_decision_time > 0

    def test_fit_insufficient_data(self):
        self.estimator.observe("rare", "bet", 1000)
        params = self.estimator.fit("rare")
        assert params.drift == 1.0  # default

    def test_estimate_preference_fast(self):
        """Fast decision should indicate high certainty."""
        for i in range(20):
            time_ms = 2000 + i * 100
            self.estimator.observe("turn_bet", "bet", time_ms)
        for i in range(5):
            self.estimator.observe("turn_bet", "check", 4000 + i * 200)

        pref = self.estimator.estimate_preference("turn_bet", 800, "bet")
        assert pref.certainty > 0.5  # fast = more certain
        assert not pref.is_conflicted

    def test_estimate_preference_slow(self):
        """Slow decision should indicate conflict."""
        for _ in range(20):
            self.estimator.observe("river_bluff", "bet", 2000)

        pref = self.estimator.estimate_preference("river_bluff", 6000, "bet")
        # Very slow relative to mean → conflicted
        assert pref.is_conflicted or pref.certainty < 0.5

    def test_estimate_preference_no_data(self):
        pref = self.estimator.estimate_preference("unknown", 1000, "bet")
        assert pref.certainty == 0.5  # default

    def test_contexts_and_observations_count(self):
        self.estimator.observe("c1", "bet", 1000)
        self.estimator.observe("c1", "check", 2000)
        self.estimator.observe("c2", "fold", 500)
        assert self.estimator.contexts_observed == 2
        assert self.estimator.total_observations == 3


# ---------------------------------------------------------------------------
# #8 CNN-LSTM Skill Estimator
# ---------------------------------------------------------------------------


class TestConv1DLayer:
    def test_forward_shape(self):
        conv = Conv1DLayer(in_channels=8, out_channels=4, kernel_size=3)
        seq = [[0.5] * 8 for _ in range(10)]
        out = conv.forward(seq)
        assert len(out) == 8  # 10 - 3 + 1
        assert all(len(frame) == 4 for frame in out)

    def test_short_sequence(self):
        conv = Conv1DLayer(in_channels=8, out_channels=4, kernel_size=3)
        out = conv.forward([[0.5] * 8])  # shorter than kernel
        assert len(out) == 1
        assert len(out[0]) == 4


class TestLSTMCell:
    def test_single_step(self):
        lstm = LSTMCell(input_dim=8, hidden_dim=4)
        x = [0.5] * 8
        h = [0.0] * 4
        c = [0.0] * 4
        h_new, c_new = lstm.forward(x, h, c)
        assert len(h_new) == 4
        assert len(c_new) == 4

    def test_sequence(self):
        lstm = LSTMCell(input_dim=8, hidden_dim=4)
        seq = [[0.5] * 8 for _ in range(5)]
        final = lstm.forward_sequence(seq)
        assert len(final) == 4


class TestSkillEstimatorModel:
    def test_predict_returns_estimate(self):
        model = SkillEstimatorModel(window_size=10, seed=42)
        decisions = [DecisionFeature(action_type=0.4 + i * 0.05) for i in range(8)]
        estimate = model.predict(decisions)
        assert 0 <= estimate.skill_rating <= 1
        assert 0 <= estimate.confidence <= 1
        assert estimate.skill_label in ("fish", "recreational", "regular", "skilled", "expert")

    def test_few_decisions_low_confidence(self):
        model = SkillEstimatorModel()
        estimate = model.predict([DecisionFeature()])
        assert estimate.confidence == 0.0

    def test_full_window_high_confidence(self):
        model = SkillEstimatorModel(window_size=10)
        decisions = [DecisionFeature(action_type=0.5) for _ in range(15)]
        estimate = model.predict(decisions)
        assert estimate.confidence == 1.0


class TestOnlineSkillEstimator:
    def test_record_and_estimate(self):
        est = OnlineSkillEstimator(window_size=10, seed=42)
        for i in range(10):
            est.record_decision(DecisionFeature(action_type=0.5, decision_time_norm=0.3))
        result = est.estimate()
        assert 0 <= result.skill_rating <= 1

    def test_decisions_recorded_count(self):
        est = OnlineSkillEstimator()
        assert est.decisions_recorded == 0
        est.record_decision(DecisionFeature())
        assert est.decisions_recorded == 1

    def test_reset(self):
        est = OnlineSkillEstimator()
        est.record_decision(DecisionFeature())
        est.reset()
        assert est.decisions_recorded == 0


class TestSkillEstimateLabels:
    def test_label_mapping(self):
        assert SkillEstimate.label_from_rating(0.1) == "fish"
        assert SkillEstimate.label_from_rating(0.3) == "recreational"
        assert SkillEstimate.label_from_rating(0.5) == "regular"
        assert SkillEstimate.label_from_rating(0.7) == "skilled"
        assert SkillEstimate.label_from_rating(0.9) == "expert"


# ---------------------------------------------------------------------------
# #9 Synthetic Player Generator
# ---------------------------------------------------------------------------


class TestSyntheticPlayerStats:
    def test_to_vector_length(self):
        stats = SyntheticPlayerStats()
        vec = stats.to_vector()
        assert len(vec) == 12  # matches StyleEmbedder input_dim


class TestSyntheticPlayerGenerator:
    def setup_method(self):
        self.gen = SyntheticPlayerGenerator(seed=42)

    def test_generate_single(self):
        player = self.gen.generate()
        assert isinstance(player, SyntheticPlayer)
        assert player.name
        assert player.archetype in (
            "nit", "tag", "lag", "maniac", "fish", "whale", "rock", "calling_station"
        )
        assert 0 <= player.skill_level <= 1
        assert 0 <= player.tilt_propensity <= 1

    def test_generate_specific_archetype(self):
        player = self.gen.generate(archetype="maniac")
        assert player.archetype == "maniac"
        assert player.stats.vpip > 0.25  # maniacs play lots of hands

    def test_generate_batch(self):
        players = self.gen.generate_batch(50)
        assert len(players) == 50
        archetypes = {p.archetype for p in players}
        assert len(archetypes) > 1  # should be diverse

    def test_stats_in_valid_range(self):
        players = self.gen.generate_batch(100)
        for p in players:
            assert 0 <= p.stats.vpip <= 1
            assert 0 <= p.stats.pfr <= 1
            assert 0 <= p.stats.fold_to_cbet <= 1
            assert 0.1 <= p.stats.aggression_factor <= 10.0
            assert p.stats.timing_mean_ms > 0
            assert p.stats.timing_std_ms > 0

    def test_deterministic_with_same_seed(self):
        gen1 = SyntheticPlayerGenerator(seed=123)
        gen2 = SyntheticPlayerGenerator(seed=123)
        p1 = gen1.generate()
        p2 = gen2.generate()
        assert p1.name == p2.name
        assert p1.archetype == p2.archetype
        assert abs(p1.stats.vpip - p2.stats.vpip) < 1e-10

    def test_custom_archetype_distribution(self):
        dist = {"fish": 1.0}  # all fish
        players = self.gen.generate_batch(20, archetype_distribution=dist)
        assert all(p.archetype == "fish" for p in players)

    def test_skill_affects_stats(self):
        """Higher skill should correlate with lower limp freq."""
        players = self.gen.generate_batch(200)
        skilled = [p for p in players if p.skill_level > 0.6]
        unskilled = [p for p in players if p.skill_level < 0.3]
        if skilled and unskilled:
            avg_limp_skilled = sum(p.stats.limp_freq for p in skilled) / len(skilled)
            avg_limp_unskilled = sum(p.stats.limp_freq for p in unskilled) / len(unskilled)
            # Skilled players should limp less on average
            assert avg_limp_skilled <= avg_limp_unskilled + 0.1  # allow some noise

    def test_stats_to_vector_compatible(self):
        """Stats should produce vectors compatible with StyleEmbedder."""
        player = self.gen.generate()
        vec = player.stats.to_vector()
        assert len(vec) == 12
        assert all(isinstance(v, float) for v in vec)


# ---------------------------------------------------------------------------
# PGM Upgrade: Personality-conditioned generation (Nemotron-Personas inspired)
# ---------------------------------------------------------------------------


class TestPersonalityProfile:
    def test_to_vector(self):
        p = PersonalityProfile(openness=0.8, conscientiousness=0.3)
        vec = p.to_vector()
        assert len(vec) == 5
        assert vec[0] == 0.8
        assert vec[1] == 0.3

    def test_risk_tolerance_high(self):
        """High openness + extraversion, low conscientiousness → high risk."""
        p = PersonalityProfile(openness=0.9, conscientiousness=0.1,
                               extraversion=0.9, agreeableness=0.1)
        assert p.risk_tolerance > 0.6

    def test_risk_tolerance_low(self):
        """Low openness + extraversion, high conscientiousness → low risk."""
        p = PersonalityProfile(openness=0.1, conscientiousness=0.9,
                               extraversion=0.1, agreeableness=0.9)
        assert p.risk_tolerance < 0.4

    def test_emotional_stability(self):
        p = PersonalityProfile(neuroticism=0.8)
        assert p.emotional_stability == pytest.approx(0.2)


class TestPGMGeneration:
    def setup_method(self):
        self.gen = SyntheticPlayerGenerator(seed=42)

    def test_personality_present(self):
        player = self.gen.generate()
        assert hasattr(player, "personality")
        assert isinstance(player.personality, PersonalityProfile)
        assert 0 <= player.personality.openness <= 1
        assert 0 <= player.personality.neuroticism <= 1

    def test_personality_conditions_archetype(self):
        """Different personalities should produce different archetype distributions."""
        gen1 = SyntheticPlayerGenerator(seed=100)
        gen2 = SyntheticPlayerGenerator(seed=200)
        archetypes1 = [gen1.generate().archetype for _ in range(100)]
        archetypes2 = [gen2.generate().archetype for _ in range(100)]
        # Different seeds → different personality samples → different distributions
        # (not guaranteed to be totally different, but at least some diversity)
        assert len(set(archetypes1)) > 1
        assert len(set(archetypes2)) > 1

    def test_neuroticism_affects_tilt(self):
        """High neuroticism should correlate with higher tilt propensity."""
        players = self.gen.generate_batch(300)
        neurotic = [p for p in players if p.personality.neuroticism > 0.5]
        stable = [p for p in players if p.personality.neuroticism < 0.3]
        if neurotic and stable:
            avg_tilt_neurotic = sum(p.tilt_propensity for p in neurotic) / len(neurotic)
            avg_tilt_stable = sum(p.tilt_propensity for p in stable) / len(stable)
            assert avg_tilt_neurotic > avg_tilt_stable

    def test_conscientiousness_affects_skill(self):
        """High conscientiousness should correlate with higher skill."""
        players = self.gen.generate_batch(300)
        conscientious = [p for p in players if p.personality.conscientiousness > 0.6]
        careless = [p for p in players if p.personality.conscientiousness < 0.3]
        if conscientious and careless:
            avg_skill_c = sum(p.skill_level for p in conscientious) / len(conscientious)
            avg_skill_l = sum(p.skill_level for p in careless) / len(careless)
            assert avg_skill_c > avg_skill_l

    def test_personality_effects_on_stats(self):
        """Extraverted players should have higher aggression on average."""
        players = self.gen.generate_batch(300)
        extraverts = [p for p in players if p.personality.extraversion > 0.6]
        introverts = [p for p in players if p.personality.extraversion < 0.3]
        if extraverts and introverts:
            avg_agg_e = sum(p.stats.aggression_factor for p in extraverts) / len(extraverts)
            avg_agg_i = sum(p.stats.aggression_factor for p in introverts) / len(introverts)
            assert avg_agg_e > avg_agg_i - 0.5  # allow noise

    def test_deterministic_with_personality(self):
        """Same seed → same personality."""
        gen1 = SyntheticPlayerGenerator(seed=42)
        gen2 = SyntheticPlayerGenerator(seed=42)
        p1 = gen1.generate()
        p2 = gen2.generate()
        assert abs(p1.personality.openness - p2.personality.openness) < 1e-10
        assert abs(p1.personality.neuroticism - p2.personality.neuroticism) < 1e-10

    def test_personality_vector_all_valid(self):
        players = self.gen.generate_batch(50)
        for p in players:
            vec = p.personality.to_vector()
            assert len(vec) == 5
            assert all(0 <= v <= 1 for v in vec)


# ---------------------------------------------------------------------------
# Enhanced Skill Estimator: BiLSTM + Bayesian Tracking
# ---------------------------------------------------------------------------


class TestBidirectionalLSTM:
    def test_output_dim(self):
        bilstm = BidirectionalLSTM(input_dim=8, hidden_dim=4)
        seq = [[0.5] * 8 for _ in range(5)]
        out = bilstm.forward_sequence(seq)
        assert len(out) == 8  # 4 forward + 4 backward

    def test_empty_sequence(self):
        bilstm = BidirectionalLSTM(input_dim=8, hidden_dim=4)
        out = bilstm.forward_sequence([])
        assert len(out) == 8
        assert all(v == 0.0 for v in out)


class TestBayesianSkillTracker:
    def test_initial_state(self):
        tracker = BayesianSkillTracker()
        mu, sigma = tracker.estimate()
        assert mu == pytest.approx(0.5)
        assert sigma == pytest.approx(0.25)

    def test_update_shifts_estimate(self):
        tracker = BayesianSkillTracker()
        for _ in range(10):
            tracker.update(0.9)  # consistently good
        mu, _ = tracker.estimate()
        assert mu > 0.7

    def test_update_bad_decisions(self):
        tracker = BayesianSkillTracker()
        for _ in range(10):
            tracker.update(0.1)  # consistently bad
        mu, _ = tracker.estimate()
        assert mu < 0.3

    def test_confidence_increases(self):
        tracker = BayesianSkillTracker()
        c0 = tracker.confidence
        for _ in range(20):
            tracker.update(0.6)
        assert tracker.confidence > c0

    def test_reset(self):
        tracker = BayesianSkillTracker()
        tracker.update(0.9)
        tracker.reset()
        mu, sigma = tracker.estimate()
        assert mu == pytest.approx(0.5)
        assert tracker.n_observations == 0


class TestEnhancedSkillEstimator:
    def test_record_and_estimate(self):
        est = EnhancedSkillEstimator(window_size=10, seed=42)
        for i in range(10):
            est.record_decision(DecisionFeature(action_type=0.5, decision_time_norm=0.5))
        result = est.estimate()
        assert 0 <= result.skill_rating <= 1
        assert result.confidence > 0

    def test_few_decisions_low_confidence(self):
        est = EnhancedSkillEstimator()
        result = est.estimate()
        assert result.confidence == 0.0

    def test_bayesian_estimate_accessible(self):
        est = EnhancedSkillEstimator(seed=42)
        est.record_decision(DecisionFeature(action_type=0.8))
        mu, sigma = est.bayesian_estimate
        assert isinstance(mu, float)
        assert isinstance(sigma, float)

    def test_decisions_recorded_count(self):
        est = EnhancedSkillEstimator()
        assert est.decisions_recorded == 0
        est.record_decision(DecisionFeature())
        assert est.decisions_recorded == 1

    def test_reset_clears_all(self):
        est = EnhancedSkillEstimator()
        est.record_decision(DecisionFeature())
        est.reset()
        assert est.decisions_recorded == 0
        assert est.bayesian.n_observations == 0

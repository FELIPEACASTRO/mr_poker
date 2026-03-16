"""Tests for the 6 missing roadmap items: QP Nash, CASPER, Equilibrium
Refinements, Particle Filtering, Action Translation, Behavior Prediction.

No mocks or stubs — all tests exercise real implementations.
"""

from __future__ import annotations

import math

import pytest

from packages.common.types import ActionType

# ── QP Nash ──────────────────────────────────────────────────────────

from packages.solver.qp_nash import (
    QPNashSolver,
    NormalFormGame,
    NashEquilibrium,
    compute_exploitability,
    _all_profiles,
)


class TestNormalFormGame:
    def test_create_game(self):
        game = NormalFormGame(num_players=2, num_actions=[3, 3])
        assert game.num_players == 2
        assert game.num_actions == [3, 3]

    def test_set_and_get_payoff(self):
        game = NormalFormGame(num_players=2, num_actions=[2, 2])
        game.set_payoff(0, (0, 0), 3.0)
        game.set_payoff(0, (0, 1), -1.0)
        assert game.get_payoff(0, (0, 0)) == 3.0
        assert game.get_payoff(0, (0, 1)) == -1.0
        assert game.get_payoff(0, (1, 1)) == 0.0  # default

    def test_all_profiles_2x2(self):
        profiles = _all_profiles([2, 2])
        assert len(profiles) == 4
        assert (0, 0) in profiles
        assert (1, 1) in profiles

    def test_all_profiles_3_players(self):
        profiles = _all_profiles([2, 2, 2])
        assert len(profiles) == 8


class TestQPNashSolver:
    def test_solve_rock_paper_scissors(self):
        """RPS has unique Nash: uniform (1/3, 1/3, 1/3)."""
        game = NormalFormGame(num_players=2, num_actions=[3, 3])
        # Rock=0, Paper=1, Scissors=2
        payoffs = [
            [0, -1, 1],
            [1, 0, -1],
            [-1, 1, 0],
        ]
        for i in range(3):
            for j in range(3):
                game.set_payoff(0, (i, j), payoffs[i][j])
                game.set_payoff(1, (i, j), -payoffs[i][j])

        solver = QPNashSolver(seed=42)
        eq = solver.solve(game, iterations=5000)

        assert len(eq.strategies) == 2
        # Each action should be ~1/3
        for p in range(2):
            for prob in eq.strategies[p]:
                assert abs(prob - 1.0 / 3) < 0.1

    def test_solve_zero_sum_2p(self):
        solver = QPNashSolver(seed=42)
        # Simple matching pennies: [[1,-1],[-1,1]]
        eq = solver.solve_zero_sum_2p([[1, -1], [-1, 1]], iterations=3000)
        assert len(eq.strategies) == 2
        # Each player should play ~50/50
        for prob in eq.strategies[0]:
            assert abs(prob - 0.5) < 0.15

    def test_support_enumeration(self):
        solver = QPNashSolver(seed=42)
        eq = solver.support_enumeration_2p([[1, -1], [-1, 1]])
        assert len(eq.strategies) == 2

    def test_exploitability_at_nash(self):
        """At Nash, exploitability should be low."""
        game = NormalFormGame(num_players=2, num_actions=[2, 2])
        game.set_payoff(0, (0, 0), 1.0)
        game.set_payoff(0, (0, 1), -1.0)
        game.set_payoff(0, (1, 0), -1.0)
        game.set_payoff(0, (1, 1), 1.0)
        game.set_payoff(1, (0, 0), -1.0)
        game.set_payoff(1, (0, 1), 1.0)
        game.set_payoff(1, (1, 0), 1.0)
        game.set_payoff(1, (1, 1), -1.0)

        solver = QPNashSolver(seed=42)
        eq = solver.solve(game, iterations=5000)
        assert eq.exploitability < 0.1

    def test_3_player_game(self):
        """Solver handles 3-player games."""
        game = NormalFormGame(num_players=3, num_actions=[2, 2, 2])
        for profile in _all_profiles([2, 2, 2]):
            for p in range(3):
                game.set_payoff(p, profile, float(sum(profile)) - profile[p])

        solver = QPNashSolver(seed=42)
        eq = solver.solve(game, iterations=3000)
        assert len(eq.strategies) == 3
        for strat in eq.strategies:
            assert abs(sum(strat) - 1.0) < 1e-6


# ── CASPER (Case-Based Reasoning) ────────────────────────────────────

from packages.solver.casper import (
    CASPERAgent,
    CaseBase,
    PokerCase,
    CaseQuery,
    _euclidean_distance,
    _cosine_similarity,
)


class TestCaseBase:
    def test_add_and_retrieve(self):
        cb = CaseBase(max_cases=100)
        case = PokerCase(features=[1.0, 0.5, 0.3], action_taken=ActionType.RAISE, reward=5.0)
        cb.add_case(case)
        assert cb.size() == 1

        results = cb.retrieve(CaseQuery(features=[1.0, 0.5, 0.3]))
        assert len(results) == 1
        assert results[0][0].action_taken == ActionType.RAISE

    def test_pruning(self):
        cb = CaseBase(max_cases=20)
        for i in range(30):
            cb.add_case(PokerCase(features=[float(i)], reward=float(i)))
        assert cb.size() <= 20

    def test_retrieval_similarity_ranking(self):
        cb = CaseBase(max_cases=100)
        cb.add_case(PokerCase(features=[1.0, 0.0], action_taken=ActionType.FOLD, street=0))
        cb.add_case(PokerCase(features=[0.0, 1.0], action_taken=ActionType.CALL, street=1))
        cb.add_case(PokerCase(features=[1.0, 0.1], action_taken=ActionType.RAISE, street=0))

        results = cb.retrieve(CaseQuery(features=[1.0, 0.0], street=0), k=3)
        # First result should be most similar
        assert results[0][1] >= results[1][1]

    def test_empty_retrieve(self):
        cb = CaseBase()
        assert cb.retrieve(CaseQuery(features=[1.0])) == []


class TestCASPERAgent:
    def test_decide_no_cases(self):
        agent = CASPERAgent()
        dist = agent.decide(CaseQuery(features=[0.5, 0.5]))
        assert abs(sum(dist.probabilities.values()) - 1.0) < 1e-6

    def test_decide_with_cases(self):
        agent = CASPERAgent(k=5)
        # Record positive outcomes for RAISE
        for _ in range(10):
            agent.record_outcome([0.8, 0.5], ActionType.RAISE, 10.0, "fish", 0)
        # Record negative outcomes for FOLD
        for _ in range(5):
            agent.record_outcome([0.8, 0.5], ActionType.FOLD, -5.0, "fish", 0)

        dist = agent.decide(CaseQuery(features=[0.8, 0.5], opponent_type="fish", street=0))
        # RAISE should have higher probability than FOLD
        assert dist.probabilities.get(ActionType.RAISE, 0) > dist.probabilities.get(ActionType.FOLD, 0)

    def test_case_count(self):
        agent = CASPERAgent()
        assert agent.case_count() == 0
        agent.record_outcome([1.0], ActionType.BET, 3.0)
        assert agent.case_count() == 1


class TestSimilarityFunctions:
    def test_euclidean(self):
        assert _euclidean_distance([0, 0], [3, 4]) == pytest.approx(5.0)
        assert _euclidean_distance([1, 1], [1, 1]) == 0.0

    def test_cosine(self):
        assert _cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
        assert _cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
        assert _cosine_similarity([0, 0], [1, 1]) == 0.0


# ── Equilibrium Refinements ──────────────────────────────────────────

from packages.solver.equilibrium_refinements import (
    TremblingHandRefinement,
    SequentialEquilibrium,
    MaximinRefinement,
    ExtensiveFormNode,
)
from packages.strategy.mixed import ActionDistribution


class TestTremblingHand:
    def test_apply_tremble(self):
        th = TremblingHandRefinement(epsilon=0.1)
        strategy = {"fold": 0.0, "call": 1.0}
        trembled = th.apply_tremble(strategy, ["fold", "call"])
        # fold should now have > 0 probability
        assert trembled["fold"] > 0
        assert trembled["call"] > 0
        assert abs(sum(trembled.values()) - 1.0) < 1e-6

    def test_tremble_preserves_dominance(self):
        th = TremblingHandRefinement(epsilon=0.05)
        strategy = {"fold": 0.0, "call": 0.3, "raise": 0.7}
        trembled = th.apply_tremble(strategy, ["fold", "call", "raise"])
        # raise should still be most likely
        assert trembled["raise"] > trembled["call"]
        assert trembled["raise"] > trembled["fold"]

    def test_refine_tree(self):
        th = TremblingHandRefinement(epsilon=0.05, seed=42)
        root = ExtensiveFormNode(
            info_set="root",
            actions=["bet", "check"],
            acting_player=0,
        )
        root.children["bet"] = ExtensiveFormNode(
            info_set="opp_facing_bet",
            actions=["call", "fold"],
            acting_player=1,
        )
        root.children["bet"].children["call"] = ExtensiveFormNode(
            is_terminal=True, terminal_value=2.0
        )
        root.children["bet"].children["fold"] = ExtensiveFormNode(
            is_terminal=True, terminal_value=1.0
        )
        root.children["check"] = ExtensiveFormNode(
            is_terminal=True, terminal_value=0.0
        )

        strategies = th.refine_tree(root, iterations=200)
        # Should produce strategies for non-terminal nodes
        assert len(strategies) >= 1
        for strat in strategies.values():
            assert abs(sum(strat.values()) - 1.0) < 1e-4
            # All actions should have > 0 prob (tremble)
            for prob in strat.values():
                assert prob > 0


class TestSequentialEquilibrium:
    def test_compute_beliefs(self):
        seq = SequentialEquilibrium()
        root = ExtensiveFormNode(
            info_set="root",
            actions=["left", "right"],
            acting_player=0,
        )
        root.children["left"] = ExtensiveFormNode(
            info_set="left_node", is_terminal=True, terminal_value=1.0
        )
        root.children["right"] = ExtensiveFormNode(
            info_set="right_node", is_terminal=True, terminal_value=0.0
        )

        strategies = {"root": {"left": 0.7, "right": 0.3}}
        beliefs = seq.compute_beliefs(root, strategies)
        assert "root" in beliefs

    def test_sequential_rationality(self):
        seq = SequentialEquilibrium()
        node = ExtensiveFormNode(
            info_set="test",
            actions=["a", "b"],
        )
        node.children["a"] = ExtensiveFormNode(is_terminal=True, terminal_value=5.0)
        node.children["b"] = ExtensiveFormNode(is_terminal=True, terminal_value=3.0)

        # Rational: play "a" with positive prob
        assert seq.is_sequentially_rational(node, {"a": 1.0, "b": 0.0}, {})
        # Irrational: play "b" only when "a" is strictly better
        assert not seq.is_sequentially_rational(node, {"a": 0.0, "b": 1.0}, {})


class TestMaximin:
    def test_maximin_strategy(self):
        mm = MaximinRefinement()
        # Matching pennies
        strategy, value = mm.maximin_strategy([[1, -1], [-1, 1]], iterations=3000)
        assert len(strategy) == 2
        # Should be ~50/50
        for p in strategy:
            assert abs(p - 0.5) < 0.15

    def test_refine(self):
        mm = MaximinRefinement()
        dists = [
            ActionDistribution(probabilities={ActionType.FOLD: 0.0, ActionType.CALL: 1.0}),
        ]
        refined = mm.refine(dists)
        # FOLD should now have some probability (robustness)
        assert refined[0].probabilities[ActionType.FOLD] > 0

    def test_maximin_empty_matrix(self):
        mm = MaximinRefinement()
        strategy, value = mm.maximin_strategy([])
        assert strategy == []
        assert value == 0.0


# ── Particle Filter ──────────────────────────────────────────────────

from packages.opponent_model.particle_filter import (
    ParticleFilterOpponentModel,
    Particle,
    ARCHETYPE_PARAMS,
)


class TestParticleFilterOpponentModel:
    def test_initialization(self):
        pf = ParticleFilterOpponentModel(num_particles=100, seed=42)
        assert len(pf.particles) == 100
        # All weights should be normalized
        total = sum(p.weight for p in pf.particles)
        assert abs(total - 1.0) < 1e-6

    def test_estimate_initial_uniform(self):
        pf = ParticleFilterOpponentModel(num_particles=200, seed=42)
        dist = pf.estimate_archetype()
        # Should have multiple archetypes
        assert len(dist) > 3

    def test_update_shifts_distribution(self):
        pf = ParticleFilterOpponentModel(num_particles=200, seed=42)
        initial = pf.estimate_archetype()

        # Feed many aggressive actions -> should shift toward lag/maniac
        for _ in range(20):
            pf.update(ActionType.RAISE, "preflop")

        after = pf.estimate_archetype()
        aggressive_types = sum(after.get(t, 0.0) for t in ("lag", "maniac", "tag"))
        # After 20 raises, aggressive types should be dominant
        assert aggressive_types > 0.3

    def test_track_style_change(self):
        """Particle filter should track style transitions."""
        pf = ParticleFilterOpponentModel(num_particles=300, seed=42)

        # Phase 1: tight play
        for _ in range(15):
            pf.update(ActionType.FOLD, "preflop")
        tight1 = pf.most_likely_archetype()

        # Phase 2: suddenly very aggressive
        for _ in range(25):
            pf.update(ActionType.RAISE, "preflop")
        after_aggro = pf.most_likely_archetype()

        # Should no longer be the same tight archetype
        # (at least some shift should occur)
        agg_dist = pf.estimate_archetype()
        aggressive_mass = sum(agg_dist.get(t, 0.0) for t in ("lag", "maniac", "tag"))
        assert aggressive_mass > 0.2

    def test_estimate_params(self):
        pf = ParticleFilterOpponentModel(num_particles=100, seed=42)
        vpip, pfr, agg = pf.estimate_params()
        assert 0 < vpip < 1
        assert 0 < pfr < 1
        assert agg > 0

    def test_confidence_grows_with_observations(self):
        pf = ParticleFilterOpponentModel(num_particles=100, seed=42)
        c0 = pf.confidence()
        for _ in range(30):
            pf.update(ActionType.RAISE, "preflop")
        c1 = pf.confidence()
        assert c1 >= c0  # confidence should not decrease after observations

    def test_reset(self):
        pf = ParticleFilterOpponentModel(num_particles=100, seed=42)
        for _ in range(10):
            pf.update(ActionType.RAISE, "preflop")
        pf.reset()
        assert pf._observations == 0
        total = sum(p.weight for p in pf.particles)
        assert abs(total - 1.0) < 1e-6


# ── Action Translation ───────────────────────────────────────────────

from packages.solver.action_translation import (
    ActionTranslator,
    TranslationResult,
    AbstractAction,
    _pseudo_harmonic_weight,
)


class TestActionTranslator:
    def test_exact_match(self):
        t = ActionTranslator(bet_sizes=[0.5, 1.0])
        result = t.translate_bet(0.5)
        assert "bet_0.50x" in result.distribution
        assert result.distribution["bet_0.50x"] == pytest.approx(1.0)
        assert result.confidence == 1.0

    def test_interpolation(self):
        t = ActionTranslator(bet_sizes=[0.5, 1.0])
        result = t.translate_bet(0.75)
        # Should distribute between 0.5x and 1.0x
        assert "bet_0.50x" in result.distribution
        assert "bet_1.00x" in result.distribution
        assert abs(sum(result.distribution.values()) - 1.0) < 1e-6

    def test_pseudo_harmonic_midpoint(self):
        # Midpoint should give ~equal weights
        w = _pseudo_harmonic_weight(0.75, 0.5, 1.0)
        assert 0.3 < w < 0.7

    def test_pseudo_harmonic_exact(self):
        w = _pseudo_harmonic_weight(0.5, 0.5, 1.0)
        assert w == 1.0

    def test_below_grid(self):
        t = ActionTranslator(bet_sizes=[0.5, 1.0])
        result = t.translate_bet(0.2)
        # Should map partially to call and to 0.5x
        assert len(result.distribution) > 0
        assert abs(sum(result.distribution.values()) - 1.0) < 1e-6

    def test_above_grid(self):
        t = ActionTranslator(bet_sizes=[0.5, 1.0])
        result = t.translate_bet(2.0)
        assert len(result.distribution) > 0

    def test_translate_non_bet_actions(self):
        t = ActionTranslator()
        fold = t.translate_action(ActionType.FOLD)
        assert fold.distribution == {"fold": 1.0}
        check = t.translate_action(ActionType.CHECK)
        assert check.distribution == {"check": 1.0}

    def test_translate_bet_action(self):
        t = ActionTranslator(bet_sizes=[0.33, 0.5, 0.75, 1.0])
        result = t.translate_action(ActionType.BET, bet_amount=60, pot_size=100)
        # 0.6x pot -> between 0.5x and 0.75x
        assert len(result.distribution) >= 1

    def test_grid_sizes(self):
        t = ActionTranslator(bet_sizes=[0.33, 0.5, 1.0])
        assert t.grid_sizes() == [0.33, 0.5, 1.0]

    def test_empty_grid(self):
        t = ActionTranslator(bet_sizes=[])
        result = t.translate_bet(0.5)
        assert result.confidence == 0.0


# ── Strategic Behavior Prediction ─────────────────────────────────────

from packages.opponent_model.behavior_prediction import (
    BehaviorPredictor,
    ActionEvent,
    encode_event,
)


class TestBehaviorPredictor:
    def test_encode_event(self):
        e = ActionEvent(action=ActionType.RAISE, street="flop", bet_fraction=0.75)
        vec = encode_event(e)
        assert len(vec) == 10
        assert vec[4] == 1.0  # RAISE index
        assert vec[6] == pytest.approx(0.33)  # flop

    def test_predict_returns_distribution(self):
        bp = BehaviorPredictor(window_size=4, hidden_dim=16, seed=42)
        probs = bp.predict()
        assert len(probs) == 6
        assert abs(sum(probs.values()) - 1.0) < 1e-4

    def test_observe_and_predict(self):
        bp = BehaviorPredictor(window_size=4, hidden_dim=16, seed=42)
        for _ in range(5):
            bp.observe(ActionEvent(action=ActionType.RAISE, street="preflop"))

        probs = bp.predict()
        assert all(p >= 0 for p in probs.values())
        assert abs(sum(probs.values()) - 1.0) < 1e-4

    def test_train_step(self):
        bp = BehaviorPredictor(window_size=4, hidden_dim=16, learning_rate=0.01, seed=42)
        # Need at least 2 observations to create training data
        bp.observe(ActionEvent(action=ActionType.CHECK, street="preflop"))
        bp.observe(ActionEvent(action=ActionType.RAISE, street="flop"))
        bp.observe(ActionEvent(action=ActionType.CALL, street="turn"))

        loss = bp.train_step(epochs=3)
        assert loss >= 0.0

    def test_most_likely_action(self):
        bp = BehaviorPredictor(window_size=4, hidden_dim=16, seed=42)
        action, prob = bp.most_likely_action()
        assert isinstance(action, ActionType)
        assert 0 <= prob <= 1

    def test_predict_specific_action(self):
        bp = BehaviorPredictor(window_size=4, hidden_dim=16, seed=42)
        prob = bp.predict_specific_action(ActionType.FOLD)
        assert 0 <= prob <= 1

    def test_observation_count(self):
        bp = BehaviorPredictor()
        assert bp.observation_count == 0
        bp.observe(ActionEvent(action=ActionType.CHECK))
        assert bp.observation_count == 1

    def test_reset(self):
        bp = BehaviorPredictor()
        bp.observe(ActionEvent(action=ActionType.CHECK))
        bp.observe(ActionEvent(action=ActionType.RAISE))
        bp.reset()
        assert bp.observation_count == 0

    def test_calibration_error_few_data(self):
        bp = BehaviorPredictor()
        # Few data -> high calibration error
        assert bp.calibration_error() == 1.0

    def test_training_improves_or_stable(self):
        """Training should not crash and loss should be finite."""
        bp = BehaviorPredictor(window_size=4, hidden_dim=16, learning_rate=0.01, seed=42)
        for _ in range(20):
            bp.observe(ActionEvent(action=ActionType.RAISE, street="preflop"))
        for _ in range(10):
            bp.observe(ActionEvent(action=ActionType.FOLD, street="flop"))

        loss = bp.train_step(epochs=5)
        assert math.isfinite(loss)


# ── Integration: all 6 modules together ──────────────────────────────

class TestIntegration:
    def test_solver_package_exports(self):
        """All new solvers accessible from packages.solver."""
        from packages.solver import (
            QPNashSolver,
            CASPERAgent,
            ActionTranslator,
            TremblingHandRefinement,
            SequentialEquilibrium,
            MaximinRefinement,
        )
        assert QPNashSolver is not None
        assert CASPERAgent is not None
        assert ActionTranslator is not None

    def test_opponent_model_exports(self):
        """New opponent models accessible from packages.opponent_model."""
        from packages.opponent_model import (
            ParticleFilterOpponentModel,
            BehaviorPredictor,
        )
        assert ParticleFilterOpponentModel is not None
        assert BehaviorPredictor is not None

    def test_particle_filter_with_action_translation(self):
        """Particle filter + action translation work together."""
        pf = ParticleFilterOpponentModel(num_particles=50, seed=42)
        at = ActionTranslator(bet_sizes=[0.5, 1.0])

        # Opponent bets 0.75x pot
        translation = at.translate_action(ActionType.BET, bet_amount=75, pot_size=100)
        assert len(translation.distribution) > 0

        # Update particle filter with the action
        pf.update(ActionType.BET, "flop")
        assert pf._observations == 1

    def test_behavior_prediction_with_casper(self):
        """Behavior predictor and CASPER can process same observations."""
        bp = BehaviorPredictor(window_size=4, hidden_dim=16, seed=42)
        casper = CASPERAgent(k=5, seed=42)

        events = [
            ActionEvent(action=ActionType.RAISE, street="preflop"),
            ActionEvent(action=ActionType.BET, street="flop"),
            ActionEvent(action=ActionType.CHECK, street="turn"),
        ]

        for e in events:
            bp.observe(e)
            casper.record_outcome(
                features=[0.5, 0.3, 0.8],
                action=e.action,
                reward=2.0,
                street=0,
            )

        bp_probs = bp.predict()
        casper_dist = casper.decide(CaseQuery(features=[0.5, 0.3, 0.8]))

        assert abs(sum(bp_probs.values()) - 1.0) < 1e-4
        assert abs(sum(casper_dist.probabilities.values()) - 1.0) < 1e-4

    def test_nash_with_refinements(self):
        """QP Nash + Maximin refinement pipeline."""
        solver = QPNashSolver(seed=42)
        eq = solver.solve_zero_sum_2p([[3, 0], [5, 1]], iterations=2000)

        mm = MaximinRefinement()
        dists = [
            ActionDistribution(probabilities={
                ActionType.BET: eq.strategies[0][0],
                ActionType.CHECK: eq.strategies[0][1],
            }),
        ]
        refined = mm.refine(dists)
        # Both actions should have positive probability after refinement
        for action, prob in refined[0].probabilities.items():
            assert prob > 0

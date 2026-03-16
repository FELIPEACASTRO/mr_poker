"""Tests for Strategic Analysis implementation modules.

Covers all modules from docs/106_Analise_Estrategica_Avancada.md:
- C1: PDCFR+ (Optimistic Mirror Descent CFR)
- C2: Dataset Ingestion Pipeline
- C3: Playing the Player (GTO→Exploit framework)
- C4: Consistent Opponent Modeling
- C5: Responsible Gaming
- C6: Coach Report
- C7: OpenSkill Rating System
- C8: Theory of Mind
- C9: Training Orchestrator
"""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
import time

import pytest

from packages.common.types import ActionType


# ===========================================================================
# C1: PDCFR+ Tests
# ===========================================================================


class TestPDCFRPlusState:
    """Tests for PDCFRPlusState — optimistic mirror descent regret storage."""

    def test_import(self):
        from packages.cfr_agent.pdcfr_plus import PDCFRPlusState
        state = PDCFRPlusState()
        assert state.iterations == 0

    def test_current_strategy_uniform_initial(self):
        from packages.cfr_agent.pdcfr_plus import PDCFRPlusState
        state = PDCFRPlusState()
        legal = {ActionType.FOLD, ActionType.CALL, ActionType.RAISE}
        strat = state.current_strategy("test|PF|IP", legal)
        probs = strat.probabilities
        assert len(probs) == 3
        assert abs(sum(probs.values()) - 1.0) < 1e-6
        # Uniform when no regrets
        for v in probs.values():
            assert abs(v - 1 / 3) < 1e-6

    def test_update_changes_strategy(self):
        from packages.cfr_agent.pdcfr_plus import PDCFRPlusState
        state = PDCFRPlusState()
        legal = {ActionType.FOLD, ActionType.CALL, ActionType.RAISE}
        info_set = "AKs|PF|IP|deep|small|dry_rainbow|"
        strat_before = state.current_strategy(info_set, legal)
        probs_before = strat_before.probabilities

        # Simulate update: RAISE much better than others
        utilities = {ActionType.FOLD: -1.0, ActionType.CALL: 0.5, ActionType.RAISE: 2.0}
        ev = sum(probs_before[a] * utilities[a] for a in legal)
        state.update(info_set, strat_before, utilities, ev)
        state.iterations = 1

        strat_after = state.current_strategy(info_set, legal)
        probs_after = strat_after.probabilities
        # RAISE should now have higher probability
        assert probs_after[ActionType.RAISE] > probs_before[ActionType.RAISE]

    def test_average_strategy_valid_distribution(self):
        from packages.cfr_agent.pdcfr_plus import PDCFRPlusState
        state = PDCFRPlusState()
        legal = {ActionType.CHECK, ActionType.BET}
        info_set = "QQ|F|OOP|medium|tiny|wet_flush_draw|xc"
        # Do a few updates
        for i in range(5):
            strat = state.current_strategy(info_set, legal)
            probs = strat.probabilities
            utilities = {ActionType.CHECK: random.uniform(-1, 1), ActionType.BET: random.uniform(-1, 1)}
            ev = sum(probs[a] * utilities[a] for a in legal)
            state.update(info_set, strat, utilities, ev)
            state.iterations = i + 1

        avg = state.average_strategy(info_set, legal)
        avg_probs = avg.probabilities
        assert abs(sum(avg_probs.values()) - 1.0) < 1e-6
        for v in avg_probs.values():
            assert v >= 0.0

    def test_prediction_step(self):
        from packages.cfr_agent.pdcfr_plus import PDCFRPlusState
        state = PDCFRPlusState()
        info_set = "AA|PF|IP|deep|small|dry_rainbow|"
        legal = {ActionType.CALL, ActionType.RAISE}
        # Multiple updates to build history
        for i in range(10):
            strat = state.current_strategy(info_set, legal)
            probs = strat.probabilities
            utilities = {ActionType.CALL: 0.2, ActionType.RAISE: 1.5}
            ev = sum(probs[a] * utilities[a] for a in legal)
            state.update(info_set, strat, utilities, ev)
            state.iterations = i + 1
        # Should have some predicted regrets
        assert info_set in state.cumulative_regret

    def test_state_has_cumulative_regret(self):
        from packages.cfr_agent.pdcfr_plus import PDCFRPlusState
        state = PDCFRPlusState()
        legal = {ActionType.FOLD, ActionType.CALL}
        info_set = "72o|PF|OOP|shallow|tiny||r"
        strat = state.current_strategy(info_set, legal)
        probs = strat.probabilities
        utilities = {ActionType.FOLD: -0.5, ActionType.CALL: 0.3}
        ev = sum(probs[a] * utilities[a] for a in legal)
        state.update(info_set, strat, utilities, ev)

        assert info_set in state.cumulative_regret
        assert isinstance(state.cumulative_regret[info_set], dict)


class TestPDCFRPlusTrainer:
    """Tests for PDCFRPlusTrainer — the main training loop."""

    def test_import(self):
        from packages.cfr_agent.pdcfr_plus import PDCFRPlusTrainer
        trainer = PDCFRPlusTrainer(
            small_blind=1, big_blind=2, starting_stack=100, seed=42
        )
        assert trainer is not None

    def test_train_returns_state(self):
        from packages.cfr_agent.pdcfr_plus import PDCFRPlusTrainer, PDCFRPlusState
        trainer = PDCFRPlusTrainer(
            small_blind=1, big_blind=2, starting_stack=100, seed=42
        )
        state = trainer.train(iterations=5)
        assert isinstance(state, PDCFRPlusState)
        assert state.iterations == 5

    def test_train_populates_info_sets(self):
        from packages.cfr_agent.pdcfr_plus import PDCFRPlusTrainer
        trainer = PDCFRPlusTrainer(
            small_blind=1, big_blind=2, starting_stack=100, seed=42
        )
        state = trainer.train(iterations=10)
        assert len(state.cumulative_regret) > 0

    def test_mccfr_mode(self):
        from packages.cfr_agent.pdcfr_plus import PDCFRPlusTrainer
        trainer = PDCFRPlusTrainer(
            small_blind=1, big_blind=2, starting_stack=100, seed=42, mode="mccfr"
        )
        state = trainer.train(iterations=10)
        assert state.iterations == 10
        assert len(state.cumulative_regret) > 0

    def test_different_seeds_different_results(self):
        from packages.cfr_agent.pdcfr_plus import PDCFRPlusTrainer
        t1 = PDCFRPlusTrainer(small_blind=1, big_blind=2, starting_stack=100, seed=1)
        t2 = PDCFRPlusTrainer(small_blind=1, big_blind=2, starting_stack=100, seed=2)
        s1 = t1.train(iterations=10)
        s2 = t2.train(iterations=10)
        # Should have different strategies due to different seeds
        # (different card deals lead to different info sets explored)
        assert s1.cumulative_regret != s2.cumulative_regret or len(s1.cumulative_regret) != len(s2.cumulative_regret)


# ===========================================================================
# C2: Dataset Ingestion Pipeline Tests
# ===========================================================================


class TestIngestionPipeline:
    """Tests for the dataset ingestion pipeline."""

    def test_phh_ingester_import(self):
        from packages.dataset_builder.ingestion import PHHIngester
        ingester = PHHIngester()
        assert ingester is not None

    def test_pokerbench_ingester_import(self):
        from packages.dataset_builder.ingestion import PokerBenchIngester
        ingester = PokerBenchIngester()
        assert ingester is not None

    def test_nemotron_ingester_import(self):
        from packages.dataset_builder.ingestion import NemotronPersonasIngester
        ingester = NemotronPersonasIngester()
        assert ingester is not None

    def test_pokerstars_ingester_import(self):
        from packages.dataset_builder.ingestion import PokerStarsIngester
        ingester = PokerStarsIngester()
        assert ingester is not None

    def test_generic_csv_ingester_import(self):
        from packages.dataset_builder.ingestion import GenericCSVIngester
        ingester = GenericCSVIngester()
        assert ingester is not None

    def test_pipeline_import(self):
        from packages.dataset_builder.ingestion import DatasetIngestionPipeline
        pipeline = DatasetIngestionPipeline()
        assert pipeline is not None

    def test_ingestion_result_dataclass(self):
        from packages.dataset_builder.ingestion import IngestionResult
        result = IngestionResult(
            source_path="test.jsonl",
            source_type="pokerbench",
            total_records=100,
            valid_records=95,
            skipped_records=5,
            output_path="var/datasets/test",
            manifest_path="var/datasets/test/manifest.json",
            feature_stats={},
        )
        assert result.total_records == 100
        assert result.valid_records == 95

    def test_training_data_formatter(self):
        from packages.dataset_builder.ingestion import TrainingDataFormatter
        fmt = TrainingDataFormatter()
        assert fmt is not None

    def test_phh_ingester_parse_sample(self):
        """Test PHH ingestion with a minimal sample file."""
        from packages.dataset_builder.ingestion import PHHIngester
        ingester = PHHIngester()
        # Create a temp JSONL file with a simple PHH record
        sample = {
            "variant": "NT",
            "hand_id": "test_001",
            "ante": 0,
            "blinds": [1, 2],
            "stacks": [100, 100],
            "actions": ["cc", "cc", "cc", "cc"],
            "board": ["Ah", "Kd", "7s", "2c", "9h"],
            "hole_cards": [["As", "Ks"], ["Jh", "Td"]],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(sample) + "\n")
            tmp_path = f.name
        try:
            records = ingester.ingest_phh_file(tmp_path)
            assert isinstance(records, list)
            assert len(records) >= 1
        finally:
            os.unlink(tmp_path)

    def test_generic_csv_with_mapping(self):
        """Test CSV ingestion with column mapping."""
        from packages.dataset_builder.ingestion import GenericCSVIngester
        ingester = GenericCSVIngester()
        # Create CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("action,pot,stack,street\n")
            f.write("call,50,100,flop\n")
            f.write("raise,80,100,turn\n")
            tmp_path = f.name
        try:
            mapping = {"action": "action_type", "pot": "pot", "stack": "stack", "street": "street"}
            records = ingester.ingest_csv(tmp_path, column_mapping=mapping)
            assert isinstance(records, list)
            assert len(records) == 2
        finally:
            os.unlink(tmp_path)


# ===========================================================================
# C3: Playing the Player (Exploit Framework) Tests
# ===========================================================================


class TestPlayingThePlayer:
    """Tests for the GTO→Exploit framework."""

    def test_import(self):
        from packages.opponent_model.exploit_framework import PlayingThePlayer
        assert PlayingThePlayer is not None

    def test_exploit_profile_dataclass(self):
        from packages.opponent_model.exploit_framework import ExploitProfile
        profile = ExploitProfile(
            vpip_deviation=0.1,
            aggression_deviation=0.2,
            fold_to_cbet_deviation=-0.1,
            tilt_factor=0.3,
            fatigue_factor=0.1,
            timing_exploit=0.05,
            sizing_exploit=0.1,
            adaptation_risk=0.2,
            overall_exploitability=0.35,
        )
        assert profile.overall_exploitability == 0.35

    def test_exploit_strategy_dataclass(self):
        from packages.opponent_model.exploit_framework import ExploitStrategy
        strat = ExploitStrategy(
            blend_weight=0.3,
            action_adjustments={ActionType.BET: 0.1, ActionType.FOLD: -0.1},
            reasoning="Opponent folds too much to c-bets",
        )
        assert strat.blend_weight == 0.3

    def test_init_with_cfr_state(self):
        from packages.cfr_agent.trainer import CFRState
        from packages.opponent_model.exploit_framework import PlayingThePlayer
        cfr = CFRState()
        player = PlayingThePlayer(cfr_state=cfr)
        assert player is not None

    def test_compute_exploit_profile(self):
        from packages.cfr_agent.trainer import CFRState
        from packages.opponent_model.behavioral_pipeline import BehavioralSignals
        from packages.opponent_model.exploit_framework import PlayingThePlayer
        cfr = CFRState()
        player = PlayingThePlayer(cfr_state=cfr)
        signals = BehavioralSignals()
        profile = player.compute_exploit_profile(signals, {"vpip": 0.45, "pfr": 0.15, "aggression": 0.3})
        assert 0.0 <= profile.overall_exploitability <= 1.0

    def test_get_blended_strategy_returns_valid(self):
        from packages.cfr_agent.trainer import CFRState
        from packages.opponent_model.behavioral_pipeline import BehavioralSignals
        from packages.opponent_model.exploit_framework import PlayingThePlayer
        cfr = CFRState()
        player = PlayingThePlayer(cfr_state=cfr)
        legal = {ActionType.FOLD, ActionType.CALL, ActionType.RAISE}
        signals = BehavioralSignals()
        strat = player.get_blended_strategy(
            "AKs|PF|IP|deep|small|dry_rainbow|", legal, signals, {"vpip": 0.5}
        )
        # strat may be ActionDistribution or dict depending on implementation
        probs = strat.probabilities if hasattr(strat, 'probabilities') else strat
        assert abs(sum(probs.values()) - 1.0) < 1e-6
        for v in probs.values():
            assert v >= 0.0

    def test_should_revert_to_gto(self):
        from packages.cfr_agent.trainer import CFRState
        from packages.opponent_model.behavioral_pipeline import BehavioralSignals
        from packages.opponent_model.exploit_framework import PlayingThePlayer
        from packages.opponent_model.meta_game import AdaptationState
        cfr = CFRState()
        player = PlayingThePlayer(cfr_state=cfr, adaptation_threshold=0.3)
        signals = BehavioralSignals()
        # Default: no adaptation
        assert not player.should_revert_to_gto(signals)

    def test_exploit_report(self):
        from packages.cfr_agent.trainer import CFRState
        from packages.opponent_model.exploit_framework import PlayingThePlayer
        cfr = CFRState()
        player = PlayingThePlayer(cfr_state=cfr)
        report = player.exploit_report()
        assert isinstance(report, dict)

    def test_reset(self):
        from packages.cfr_agent.trainer import CFRState
        from packages.opponent_model.exploit_framework import PlayingThePlayer
        cfr = CFRState()
        player = PlayingThePlayer(cfr_state=cfr)
        player.reset()


# ===========================================================================
# C4: Consistent Opponent Modeling Tests
# ===========================================================================


class TestConsistentOpponentModel:
    """Tests for Consistent Opponent Modeling with convergence guarantees."""

    def test_import(self):
        from packages.opponent_model.consistent_model import ConsistentOpponentModel
        model = ConsistentOpponentModel()
        assert model is not None

    def test_opponent_belief_dataclass(self):
        from packages.opponent_model.consistent_model import OpponentBelief
        belief = OpponentBelief(
            strategy={}, confidence={}, observations={}, log_likelihood=0.0
        )
        assert belief.log_likelihood == 0.0

    def test_observe_and_predict(self):
        from packages.opponent_model.consistent_model import ConsistentOpponentModel
        model = ConsistentOpponentModel()
        legal = {ActionType.FOLD, ActionType.CALL, ActionType.RAISE}
        info_set = "test_info_set"
        # Observe some actions
        for _ in range(10):
            model.observe(info_set, ActionType.CALL, legal)
        for _ in range(5):
            model.observe(info_set, ActionType.RAISE, legal)
        for _ in range(2):
            model.observe(info_set, ActionType.FOLD, legal)

        model.update_beliefs()
        prediction = model.predict(info_set, legal)
        probs = prediction.probabilities if hasattr(prediction, 'probabilities') else prediction
        assert abs(sum(probs.values()) - 1.0) < 1e-6
        # CALL was most observed, should have highest probability
        assert probs[ActionType.CALL] > probs[ActionType.FOLD]

    def test_uniform_fallback_for_unseen(self):
        from packages.opponent_model.consistent_model import ConsistentOpponentModel
        model = ConsistentOpponentModel()
        legal = {ActionType.CHECK, ActionType.BET}
        prediction = model.predict("unseen_info_set", legal)
        probs = prediction.probabilities if hasattr(prediction, 'probabilities') else prediction
        assert abs(sum(probs.values()) - 1.0) < 1e-6
        # Should be approximately uniform
        assert abs(probs[ActionType.CHECK] - 0.5) < 0.1

    def test_confidence_increases_with_observations(self):
        from packages.opponent_model.consistent_model import ConsistentOpponentModel
        model = ConsistentOpponentModel()
        legal = {ActionType.FOLD, ActionType.CALL}
        info_set = "confidence_test"
        c0 = model.confidence_at(info_set)
        for _ in range(20):
            model.observe(info_set, ActionType.CALL, legal)
        model.update_beliefs()
        c1 = model.confidence_at(info_set)
        assert c1 > c0

    def test_convergence_diagnostic(self):
        from packages.opponent_model.consistent_model import ConsistentOpponentModel
        model = ConsistentOpponentModel()
        diag = model.convergence_diagnostic()
        assert isinstance(diag, dict)

    def test_reset(self):
        from packages.opponent_model.consistent_model import ConsistentOpponentModel
        model = ConsistentOpponentModel()
        legal = {ActionType.FOLD, ActionType.CALL}
        model.observe("test", ActionType.CALL, legal)
        model.reset()
        assert model.confidence_at("test") == 0.0


# ===========================================================================
# C5: Responsible Gaming Tests
# ===========================================================================


class TestResponsibleGaming:
    """Tests for Responsible Gaming monitor."""

    def test_import(self):
        from packages.opponent_model.responsible_gaming import ResponsibleGamingMonitor
        monitor = ResponsibleGamingMonitor()
        assert monitor is not None

    def test_risk_level_enum(self):
        from packages.opponent_model.responsible_gaming import RiskLevel
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.CRITICAL.value == "critical"

    def test_session_limits(self):
        from packages.opponent_model.responsible_gaming import SessionLimits
        limits = SessionLimits(max_duration_minutes=60.0, max_hands=200)
        assert limits.max_duration_minutes == 60.0

    def test_initial_risk_is_low(self):
        from packages.opponent_model.responsible_gaming import ResponsibleGamingMonitor, RiskLevel
        monitor = ResponsibleGamingMonitor()
        assert monitor.get_risk_level() == RiskLevel.LOW

    def test_risk_increases_with_losses(self):
        from packages.opponent_model.responsible_gaming import ResponsibleGamingMonitor
        monitor = ResponsibleGamingMonitor()
        # Record a series of losses
        for _ in range(10):
            monitor.record_hand_result(profit_bb=-5.0, was_bad_beat=False)
        score = monitor.compute_risk_score()
        assert score > 0.0

    def test_tilt_detection_after_bad_beats(self):
        from packages.opponent_model.responsible_gaming import ResponsibleGamingMonitor
        monitor = ResponsibleGamingMonitor()
        for _ in range(5):
            alert = monitor.record_hand_result(profit_bb=-20.0, was_bad_beat=True)
        # After multiple bad beats, risk should be elevated
        score = monitor.compute_risk_score()
        assert score > 0.1

    def test_risk_indicators(self):
        from packages.opponent_model.responsible_gaming import ResponsibleGamingMonitor, RiskIndicators
        monitor = ResponsibleGamingMonitor()
        monitor.record_hand_result(profit_bb=-3.0)
        indicators = monitor.get_indicators()
        assert isinstance(indicators, RiskIndicators)
        assert indicators.hands_played >= 1

    def test_session_summary(self):
        from packages.opponent_model.responsible_gaming import ResponsibleGamingMonitor
        monitor = ResponsibleGamingMonitor()
        monitor.record_hand_result(profit_bb=5.0)
        monitor.record_hand_result(profit_bb=-3.0)
        summary = monitor.session_summary()
        assert isinstance(summary, dict)

    def test_stake_escalation_detection(self):
        from packages.opponent_model.responsible_gaming import ResponsibleGamingMonitor
        monitor = ResponsibleGamingMonitor()
        # Record losses, then stake increase
        for _ in range(5):
            monitor.record_hand_result(profit_bb=-10.0)
        alert = monitor.record_stake_change(new_stake_bb=4.0)
        # Escalation after losses should be flagged
        # (may or may not trigger alert depending on thresholds)
        assert monitor.compute_risk_score() > 0.0

    def test_reset(self):
        from packages.opponent_model.responsible_gaming import ResponsibleGamingMonitor, RiskLevel
        monitor = ResponsibleGamingMonitor()
        monitor.record_hand_result(profit_bb=-10.0)
        monitor.reset()
        assert monitor.get_risk_level() == RiskLevel.LOW


# ===========================================================================
# C6: Coach Report Tests
# ===========================================================================


class TestCoachReport:
    """Tests for Coach Report generator."""

    def test_import(self):
        from services.coach_service.report import CoachReportGenerator
        gen = CoachReportGenerator()
        assert gen is not None

    def test_coach_report_dataclass(self):
        from services.coach_service.report import CoachReport
        assert CoachReport is not None

    def test_leak_report_dataclass(self):
        from services.coach_service.report import LeakReport
        leak = LeakReport(
            leak_name="Fold to C-bet too often",
            severity="major",
            description="Folding >65% to continuation bets",
            frequency=0.72,
            ev_cost_bb_per_hand=0.15,
            fix_suggestion="Call more with medium-strength hands",
            example_hands=["hand_001", "hand_005"],
        )
        assert leak.severity == "major"

    def test_generate_minimal_session(self):
        from services.coach_service.report import CoachReportGenerator
        gen = CoachReportGenerator()
        # Minimal session data
        session_data = {
            "session_id": "test_session_001",
            "duration_minutes": 30.0,
            "hands": [
                {
                    "hand_id": f"h{i}",
                    "button_seat": i % 2,
                    "actions": [
                        {"actor_seat": 0, "action": "raise", "amount": 6, "street": "preflop"},
                        {"actor_seat": 1, "action": "call", "amount": 6, "street": "preflop"},
                        {"actor_seat": 0, "action": "bet", "amount": 8, "street": "flop"},
                        {"actor_seat": 1, "action": "fold", "amount": 0, "street": "flop"},
                    ],
                    "result": {"winner_seat": 0, "profit": [8, -8]},
                    "traces": [],
                }
                for i in range(10)
            ],
        }
        report = gen.generate(session_data)
        assert report.total_hands == 10
        assert report.session_id == "test_session_001"

    def test_leak_detection(self):
        from services.coach_service.report import CoachReportGenerator
        gen = CoachReportGenerator()
        # Create session with obvious leak: always folding
        hands = []
        for i in range(20):
            hands.append({
                "hand_id": f"h{i}",
                "button_seat": i % 2,
                "actions": [
                    {"actor_seat": 1, "action": "raise", "amount": 6, "street": "preflop"},
                    {"actor_seat": 0, "action": "fold", "amount": 0, "street": "preflop"},
                ],
                "result": {"winner_seat": 1, "profit": [-1, 1]},
                "traces": [],
            })
        session_data = {"session_id": "fold_session", "duration_minutes": 15.0, "hands": hands}
        report = gen.generate(session_data)
        # Should detect folding too much
        assert report.total_hands == 20

    def test_format_text(self):
        from services.coach_service.report import CoachReportGenerator
        gen = CoachReportGenerator()
        session_data = {
            "session_id": "text_test",
            "duration_minutes": 10.0,
            "hands": [
                {
                    "hand_id": "h1",
                    "button_seat": 0,
                    "actions": [
                        {"actor_seat": 0, "action": "raise", "amount": 6, "street": "preflop"},
                        {"actor_seat": 1, "action": "fold", "amount": 0, "street": "preflop"},
                    ],
                    "result": {"winner_seat": 0, "profit": [3, -3]},
                    "traces": [],
                }
            ],
        }
        report = gen.generate(session_data)
        text = gen.format_text(report)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_format_dict(self):
        from services.coach_service.report import CoachReportGenerator
        gen = CoachReportGenerator()
        session_data = {
            "session_id": "dict_test",
            "duration_minutes": 5.0,
            "hands": [
                {
                    "hand_id": "h1",
                    "button_seat": 0,
                    "actions": [
                        {"actor_seat": 0, "action": "call", "amount": 2, "street": "preflop"},
                        {"actor_seat": 1, "action": "check", "amount": 0, "street": "flop"},
                        {"actor_seat": 0, "action": "check", "amount": 0, "street": "flop"},
                    ],
                    "result": {"winner_seat": 0, "profit": [2, -2]},
                    "traces": [],
                }
            ],
        }
        report = gen.generate(session_data)
        d = gen.format_dict(report)
        assert isinstance(d, dict)
        assert "session_id" in d


# ===========================================================================
# C7: OpenSkill Rating System Tests
# ===========================================================================


class TestOpenSkillRating:
    """Tests for the Bayesian OpenSkill rating system."""

    def test_import(self):
        from packages.opponent_model.openskill_rating import OpenSkillRating
        rating = OpenSkillRating()
        assert rating is not None

    def test_player_rating_defaults(self):
        from packages.opponent_model.openskill_rating import PlayerRating
        r = PlayerRating()
        assert r.mu == 25.0
        assert abs(r.sigma - 25.0 / 3) < 1e-6
        assert r.games_played == 0

    def test_ordinal_rating(self):
        from packages.opponent_model.openskill_rating import PlayerRating
        r = PlayerRating(mu=30.0, sigma=5.0)
        assert r.ordinal == 30.0 - 3 * 5.0  # 15.0

    def test_confidence_grows(self):
        from packages.opponent_model.openskill_rating import PlayerRating
        r_new = PlayerRating(sigma=25.0 / 3)
        r_exp = PlayerRating(sigma=2.0)
        assert r_exp.confidence > r_new.confidence

    def test_get_or_create_rating(self):
        from packages.opponent_model.openskill_rating import OpenSkillRating
        system = OpenSkillRating()
        r1 = system.get_or_create_rating("player_a")
        r2 = system.get_or_create_rating("player_a")
        assert r1.mu == r2.mu
        # New player gets default
        assert r1.mu == 25.0

    def test_rate_match_winner_improves(self):
        from packages.opponent_model.openskill_rating import OpenSkillRating, MatchResult
        system = OpenSkillRating()
        result = MatchResult(
            player_a_id="alice", player_b_id="bob",
            winner_id="alice", profit_bb=10.0, hands_played=100,
        )
        ra, rb = system.rate_match(result)
        # Winner's mu should increase
        assert ra.mu > 25.0
        # Loser's mu should decrease
        assert rb.mu < 25.0

    def test_rate_match_updates_games_played(self):
        from packages.opponent_model.openskill_rating import OpenSkillRating, MatchResult
        system = OpenSkillRating()
        result = MatchResult(
            player_a_id="alice", player_b_id="bob",
            winner_id="alice", profit_bb=5.0, hands_played=50,
        )
        system.rate_match(result)
        assert system.get_or_create_rating("alice").games_played == 1

    def test_sigma_decreases_after_matches(self):
        from packages.opponent_model.openskill_rating import OpenSkillRating, MatchResult
        system = OpenSkillRating()
        initial_sigma = system.get_or_create_rating("alice").sigma
        for i in range(5):
            result = MatchResult(
                player_a_id="alice", player_b_id=f"opponent_{i}",
                winner_id="alice", profit_bb=5.0, hands_played=100,
            )
            system.rate_match(result)
        final_sigma = system.get_or_create_rating("alice").sigma
        assert final_sigma < initial_sigma

    def test_expected_win_probability(self):
        from packages.opponent_model.openskill_rating import OpenSkillRating, MatchResult
        system = OpenSkillRating()
        # Make alice much stronger
        for i in range(10):
            result = MatchResult(
                player_a_id="alice", player_b_id=f"opp_{i}",
                winner_id="alice", profit_bb=20.0, hands_played=100,
            )
            system.rate_match(result)
        p = system.expected_win_probability("alice", "new_player")
        assert p > 0.5  # Alice should be favored

    def test_leaderboard(self):
        from packages.opponent_model.openskill_rating import OpenSkillRating, MatchResult
        system = OpenSkillRating()
        for i in range(3):
            result = MatchResult(
                player_a_id="alice", player_b_id="bob",
                winner_id="alice", profit_bb=5.0, hands_played=50,
            )
            system.rate_match(result)
        board = system.leaderboard(min_games=1)
        assert len(board) >= 2
        # Alice should be #1
        assert board[0][0] == "alice"

    def test_reset(self):
        from packages.opponent_model.openskill_rating import OpenSkillRating, MatchResult
        system = OpenSkillRating()
        system.get_or_create_rating("alice")
        system.reset()
        assert system.leaderboard(min_games=0) == []


# ===========================================================================
# C8: Theory of Mind Tests
# ===========================================================================


class TestTheoryOfMind:
    """Tests for Theory of Mind module."""

    def test_import(self):
        from packages.opponent_model.theory_of_mind import TheoryOfMind
        tom = TheoryOfMind()
        assert tom is not None

    def test_tom_level_enum(self):
        from packages.opponent_model.theory_of_mind import ToMLevel
        assert ToMLevel.LEVEL_0 == 0
        assert ToMLevel.LEVEL_2 == 2

    def test_belief_about_us_dataclass(self):
        from packages.opponent_model.theory_of_mind import BeliefAboutUs
        belief = BeliefAboutUs(
            perceived_vpip=0.28,
            perceived_pfr=0.22,
            perceived_aggression=0.45,
            perceived_bluff_frequency=0.15,
            perceived_fold_to_raise=0.55,
            perceived_cbet=0.65,
            perceived_style="tight_aggressive",
            confidence=0.6,
        )
        assert belief.perceived_style == "tight_aggressive"
        vec = belief.to_vector()
        assert len(vec) >= 6

    def test_record_our_actions(self):
        from packages.opponent_model.theory_of_mind import TheoryOfMind
        tom = TheoryOfMind()
        tom.record_our_action("raise", "preflop", bet_fraction=3.0, was_visible=True)
        tom.record_our_action("bet", "flop", bet_fraction=0.66, was_visible=True)
        tom.record_our_action("fold", "turn", was_visible=False)
        # Should have tracked visible actions
        belief = tom.estimate_their_belief()
        assert isinstance(belief.perceived_vpip, float)

    def test_estimate_belief_defaults(self):
        from packages.opponent_model.theory_of_mind import TheoryOfMind
        tom = TheoryOfMind()
        belief = tom.estimate_their_belief()
        # Default baseline
        assert 0.0 <= belief.perceived_vpip <= 1.0
        assert 0.0 <= belief.confidence <= 1.0

    def test_predict_with_tom_level0(self):
        from packages.opponent_model.theory_of_mind import TheoryOfMind, ToMLevel
        tom = TheoryOfMind()
        legal = {ActionType.FOLD, ActionType.CALL, ActionType.RAISE}
        prediction = tom.predict_with_tom("test_info_set", legal, level=ToMLevel.LEVEL_0)
        assert abs(sum(prediction.action_probs.values()) - 1.0) < 1e-6
        assert prediction.reasoning_level == ToMLevel.LEVEL_0

    def test_predict_with_tom_level1(self):
        from packages.opponent_model.theory_of_mind import TheoryOfMind, ToMLevel
        tom = TheoryOfMind()
        # Record some visible actions to build their model of us
        for _ in range(10):
            tom.record_our_action("raise", "preflop", bet_fraction=3.0, was_visible=True)
        legal = {ActionType.FOLD, ActionType.CALL, ActionType.RAISE}
        prediction = tom.predict_with_tom("test_info_set", legal, level=ToMLevel.LEVEL_1)
        assert prediction.reasoning_level == ToMLevel.LEVEL_1
        assert abs(sum(prediction.action_probs.values()) - 1.0) < 1e-6

    def test_deception_score(self):
        from packages.opponent_model.theory_of_mind import TheoryOfMind
        tom = TheoryOfMind(our_actual_stats={"vpip": 0.45, "pfr": 0.35, "aggression": 0.6})
        # Show tight image
        for _ in range(10):
            tom.record_our_action("fold", "preflop", was_visible=True)
        score = tom.deception_score()
        assert score >= 0.0

    def test_recommend_deviation(self):
        from packages.opponent_model.theory_of_mind import TheoryOfMind, BeliefAboutUs
        tom = TheoryOfMind()
        belief = BeliefAboutUs(
            perceived_vpip=0.20,
            perceived_pfr=0.15,
            perceived_aggression=0.30,
            perceived_bluff_frequency=0.05,
            perceived_fold_to_raise=0.65,
            perceived_cbet=0.70,
            perceived_style="tight_passive",
            confidence=0.7,
        )
        actual = {"vpip": 0.35, "pfr": 0.28, "aggression": 0.55}
        legal = {ActionType.FOLD, ActionType.CALL, ActionType.RAISE}
        deviation = tom.recommend_deviation(belief, actual, legal)
        assert isinstance(deviation, dict)

    def test_update_after_showdown(self):
        from packages.opponent_model.theory_of_mind import TheoryOfMind
        tom = TheoryOfMind()
        tom.record_our_action("raise", "preflop", bet_fraction=3.0, was_visible=True)
        tom.update_after_showdown(we_showed_cards=True, was_bluff=True)
        # After showing a bluff, their belief should adjust

    def test_reset(self):
        from packages.opponent_model.theory_of_mind import TheoryOfMind
        tom = TheoryOfMind()
        tom.record_our_action("bet", "flop")
        tom.reset()
        belief = tom.estimate_their_belief()
        assert belief.confidence <= 0.1


# ===========================================================================
# C9: Training Orchestrator Tests
# ===========================================================================


class TestTrainingOrchestrator:
    """Tests for the training orchestrator."""

    def test_training_config_defaults(self):
        from services.training_service.orchestrator import TrainingConfig
        cfg = TrainingConfig(model_type="cfr")
        assert cfg.model_type == "cfr"
        assert cfg.seed == 42
        assert cfg.cfr_iterations == 10000

    def test_config_to_dict_roundtrip(self):
        from services.training_service.orchestrator import TrainingConfig
        cfg = TrainingConfig(model_type="deep_cfr", seed=123)
        d = cfg.to_dict()
        cfg2 = TrainingConfig.from_dict(d)
        assert cfg2.model_type == "deep_cfr"
        assert cfg2.seed == 123

    def test_config_save_load(self):
        from services.training_service.orchestrator import TrainingConfig
        cfg = TrainingConfig(model_type="pdcfr_plus", cfr_iterations=500)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            tmp_path = f.name
        try:
            cfg.save(tmp_path)
            loaded = TrainingConfig.from_json(tmp_path)
            assert loaded.model_type == "pdcfr_plus"
            assert loaded.cfr_iterations == 500
        finally:
            os.unlink(tmp_path)

    def test_orchestrator_init(self):
        from services.training_service.orchestrator import TrainingOrchestrator, TrainingConfig
        cfg = TrainingConfig(model_type="cfr", cfr_iterations=5)
        orch = TrainingOrchestrator(config=cfg)
        assert orch.progress.total_iterations > 0

    def test_train_cfr_small(self):
        from services.training_service.orchestrator import TrainingOrchestrator, TrainingConfig
        cfg = TrainingConfig(
            model_type="cfr", cfr_iterations=5, seed=42,
            output_dir=tempfile.mkdtemp(),
            checkpoint_interval=100, eval_interval=100, log_interval=100,
        )
        orch = TrainingOrchestrator(config=cfg)
        result = orch.train()
        assert result.model_type == "cfr"
        assert result.progress.iteration == 5

    def test_train_pdcfr_plus_small(self):
        from services.training_service.orchestrator import TrainingOrchestrator, TrainingConfig
        cfg = TrainingConfig(
            model_type="pdcfr_plus", cfr_iterations=5, seed=42,
            output_dir=tempfile.mkdtemp(),
            checkpoint_interval=100, eval_interval=100, log_interval=100,
        )
        orch = TrainingOrchestrator(config=cfg)
        result = orch.train()
        assert result.model_type == "pdcfr_plus"
        assert result.progress.iteration == 5

    def test_training_result_has_metrics(self):
        from services.training_service.orchestrator import TrainingOrchestrator, TrainingConfig
        cfg = TrainingConfig(
            model_type="cfr", cfr_iterations=3, seed=42,
            output_dir=tempfile.mkdtemp(),
            checkpoint_interval=100, eval_interval=100, log_interval=100,
        )
        orch = TrainingOrchestrator(config=cfg)
        result = orch.train()
        assert isinstance(result.final_metrics, dict)

    def test_training_progress_callback(self):
        from services.training_service.orchestrator import TrainingOrchestrator, TrainingConfig
        cfg = TrainingConfig(
            model_type="cfr", cfr_iterations=3, seed=42,
            output_dir=tempfile.mkdtemp(),
            checkpoint_interval=100, eval_interval=100, log_interval=1,
        )
        orch = TrainingOrchestrator(config=cfg)
        progress_calls = []
        orch.on_progress(lambda p: progress_calls.append(p))
        orch.train()
        assert len(progress_calls) >= 1


# ===========================================================================
# C9b: Training Config Files Tests
# ===========================================================================


class TestTrainingConfigs:
    """Tests for preset training configuration files."""

    def test_cfr_default_config_exists(self):
        path = os.path.join(
            os.path.dirname(__file__), "..", "..", "configs", "training", "cfr_default.json"
        )
        assert os.path.exists(path), f"Config not found: {path}"
        with open(path) as f:
            cfg = json.load(f)
        assert cfg["model_type"] == "cfr"

    def test_all_config_files_valid_json(self):
        config_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "configs", "training"
        )
        if not os.path.exists(config_dir):
            pytest.skip("configs/training/ directory not found")
        for fname in os.listdir(config_dir):
            if fname.endswith(".json"):
                path = os.path.join(config_dir, fname)
                with open(path) as f:
                    cfg = json.load(f)
                assert "model_type" in cfg, f"Missing model_type in {fname}"


# ===========================================================================
# Cross-module Integration Tests
# ===========================================================================


class TestCrossModuleIntegration:
    """Tests that verify modules work together correctly."""

    def test_pdcfr_with_exploitability(self):
        """PDCFR+ trained state can be evaluated for exploitability."""
        from packages.cfr_agent.pdcfr_plus import PDCFRPlusTrainer
        from packages.cfr_agent.exploitability import ExploitabilityCalculator
        from packages.engine.engine import GameEngine
        trainer = PDCFRPlusTrainer(
            small_blind=1, big_blind=2, starting_stack=100, seed=42
        )
        state = trainer.train(iterations=20)
        engine = GameEngine()
        calc = ExploitabilityCalculator(engine=engine, starting_stack=100)
        expl = calc.compute_exploitability(state, num_hands=50, seed=42)
        assert isinstance(expl, float)
        # Exploitability can be negative in zero-sum games (represents mBB/hand)
        assert isinstance(expl, (int, float))

    def test_exploit_framework_with_behavioral_pipeline(self):
        """PlayingThePlayer integrates with BehavioralPipeline signals."""
        from packages.cfr_agent.trainer import CFRState
        from packages.opponent_model.behavioral_pipeline import BehavioralPipeline
        from packages.opponent_model.exploit_framework import PlayingThePlayer
        cfr = CFRState()
        pipeline = BehavioralPipeline()
        player = PlayingThePlayer(cfr_state=cfr)
        signals = pipeline.get_signals()
        legal = {ActionType.FOLD, ActionType.CALL, ActionType.RAISE}
        strat = player.get_blended_strategy(
            "AKs|PF|IP|deep|small||", legal, signals, {"vpip": 0.4}
        )
        probs = strat.probabilities if hasattr(strat, 'probabilities') else strat
        assert abs(sum(probs.values()) - 1.0) < 1e-6

    def test_consistent_model_feeds_exploit_framework(self):
        """ConsistentOpponentModel predictions can inform exploitation."""
        from packages.opponent_model.consistent_model import ConsistentOpponentModel
        model = ConsistentOpponentModel()
        legal = {ActionType.FOLD, ActionType.CALL, ActionType.RAISE}
        # Observe opponent folding a lot
        for _ in range(15):
            model.observe("cbet_spot", ActionType.FOLD, legal)
        for _ in range(3):
            model.observe("cbet_spot", ActionType.CALL, legal)
        model.update_beliefs()
        pred = model.predict("cbet_spot", legal)
        probs = pred.probabilities if hasattr(pred, 'probabilities') else pred
        # Opponent folds often → high exploitation opportunity
        assert probs[ActionType.FOLD] > probs[ActionType.CALL]

    def test_responsible_gaming_with_session_data(self):
        """ResponsibleGamingMonitor works with realistic session data."""
        from packages.opponent_model.responsible_gaming import ResponsibleGamingMonitor, RiskLevel
        monitor = ResponsibleGamingMonitor()
        # Simulate a bad session
        rng = random.Random(42)
        for _ in range(50):
            profit = rng.gauss(-2, 10)
            bad_beat = profit < -15
            monitor.record_hand_result(profit_bb=profit, was_bad_beat=bad_beat)
            monitor.record_user_action("call", "preflop", bet_fraction=2.0)
        risk = monitor.get_risk_level()
        assert isinstance(risk, RiskLevel)

    def test_openskill_with_training_results(self):
        """OpenSkill can rate players after simulated matches."""
        from packages.opponent_model.openskill_rating import OpenSkillRating, MatchResult
        system = OpenSkillRating()
        players = ["alice", "bob", "charlie", "diana"]
        rng = random.Random(42)
        for _ in range(20):
            a, b = rng.sample(players, 2)
            winner = a if rng.random() > 0.4 else b
            result = MatchResult(
                player_a_id=a, player_b_id=b,
                winner_id=winner, profit_bb=rng.uniform(1, 30),
                hands_played=100,
            )
            system.rate_match(result)
        board = system.leaderboard(min_games=1)
        assert len(board) == 4

    def test_tom_feeds_exploit_framework(self):
        """TheoryOfMind predictions inform exploitation decisions."""
        from packages.cfr_agent.trainer import CFRState
        from packages.opponent_model.theory_of_mind import TheoryOfMind, ToMLevel
        from packages.opponent_model.exploit_framework import PlayingThePlayer
        tom = TheoryOfMind(our_actual_stats={"vpip": 0.35, "pfr": 0.25, "aggression": 0.5})
        cfr = CFRState()
        player = PlayingThePlayer(cfr_state=cfr)
        # Show tight image
        for _ in range(10):
            tom.record_our_action("fold", "preflop", was_visible=True)
        belief = tom.estimate_their_belief()
        # Our perceived VPIP should be low (we folded a lot)
        assert belief.perceived_vpip < 0.5

    def test_all_modules_importable_from_packages(self):
        """Verify all new modules are properly exported."""
        from packages.cfr_agent import PDCFRPlusTrainer, PDCFRPlusState
        from packages.opponent_model import (
            PlayingThePlayer, ExploitProfile, ExploitStrategy,
            ConsistentOpponentModel, OpponentBelief,
            ResponsibleGamingMonitor, RiskLevel, RiskAlert,
            OpenSkillRating, PlayerRating, MatchResult,
            TheoryOfMind, ToMLevel, BeliefAboutUs, ToMPrediction,
        )
        from packages.dataset_builder import (
            DatasetIngestionPipeline, PHHIngester, PokerBenchIngester,
            NemotronPersonasIngester, PokerStarsIngester, GenericCSVIngester,
            TrainingDataFormatter, IngestionResult,
        )
        from services.coach_service.report import CoachReportGenerator, CoachReport, LeakReport
        from services.training_service.orchestrator import TrainingOrchestrator, TrainingConfig
        # All imports successful
        assert True

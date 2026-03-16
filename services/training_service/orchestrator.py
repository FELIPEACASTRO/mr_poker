"""Training orchestrator for all MR_POKER models.

Provides unified interface to train, evaluate, and save all model types.
Supports resumable training, checkpointing, and multi-stage pipelines.

Supported model types:
- ``cfr``               — Tabular CFR (vanilla, DCFR, MCCFR, VAD-CFR)
- ``deep_cfr``          — Deep CFR with neural advantage/strategy networks
- ``pdcfr_plus``        — Predictive Discounted CFR+ (optimistic mirror descent)
- ``skill_estimator``   — CNN-BiLSTM opponent skill rating
- ``behavior_predictor``— GRU-based next-action prediction
- ``openskill``         — Bayesian OpenSkill rating calibration

Training stages:
1. Data loading (from ingested datasets)
2. Model initialization (or resume from checkpoint)
3. Training loop with periodic evaluation
4. Checkpointing and final model save
5. Evaluation report generation

Reference: docs/106_Analise_Estrategica_Avancada.md Section 4
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid model types
# ---------------------------------------------------------------------------
VALID_MODEL_TYPES = frozenset({
    "cfr",
    "deep_cfr",
    "pdcfr_plus",
    "skill_estimator",
    "behavior_predictor",
    "openskill",
})

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Universal training configuration.

    Covers all model types with sensible defaults.  Fields that do not apply
    to the selected ``model_type`` are simply ignored at runtime.
    """

    model_type: str = "cfr"  # one of VALID_MODEL_TYPES

    # Common ----------------------------------------------------------------
    seed: int = 42
    output_dir: str = "var/models"
    checkpoint_interval: int = 100   # save every N iterations/epochs
    eval_interval: int = 50          # evaluate every N iterations
    log_interval: int = 10           # log every N iterations

    # CFR-specific ----------------------------------------------------------
    cfr_iterations: int = 10_000
    cfr_mode: str = "dcfr"           # "vanilla", "dcfr", "mccfr", "vad"
    small_blind: int = 1
    big_blind: int = 2
    starting_stack: int = 100

    # Deep CFR-specific -----------------------------------------------------
    deep_cfr_hidden_dim: int = 64
    deep_cfr_memory_size: int = 100_000
    deep_cfr_batch_size: int = 256
    deep_cfr_train_steps: int = 1_000

    # PDCFR+ specific -------------------------------------------------------
    pdcfr_prediction_weight: float = 1.0

    # Skill Estimator specific -----------------------------------------------
    skill_window_size: int = 20
    skill_conv_channels: int = 16
    skill_lstm_hidden: int = 16
    skill_epochs: int = 100
    skill_learning_rate: float = 0.01

    # Behavior Predictor specific -------------------------------------------
    behavior_window_size: int = 8
    behavior_hidden_dim: int = 32
    behavior_gru_hidden: int = 24
    behavior_epochs: int = 50
    behavior_learning_rate: float = 0.01

    # OpenSkill specific ----------------------------------------------------
    openskill_beta: float = 4.167    # 25/6
    openskill_tau: float = 0.083     # 25/300

    # Data ------------------------------------------------------------------
    dataset_path: str | None = None  # path to ingested dataset
    train_split: float = 0.7
    val_split: float = 0.15

    # ---- helpers ----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "model_type": self.model_type,
            "seed": self.seed,
            "output_dir": self.output_dir,
            "checkpoint_interval": self.checkpoint_interval,
            "eval_interval": self.eval_interval,
            "log_interval": self.log_interval,
            "cfr_iterations": self.cfr_iterations,
            "cfr_mode": self.cfr_mode,
            "small_blind": self.small_blind,
            "big_blind": self.big_blind,
            "starting_stack": self.starting_stack,
            "deep_cfr_hidden_dim": self.deep_cfr_hidden_dim,
            "deep_cfr_memory_size": self.deep_cfr_memory_size,
            "deep_cfr_batch_size": self.deep_cfr_batch_size,
            "deep_cfr_train_steps": self.deep_cfr_train_steps,
            "pdcfr_prediction_weight": self.pdcfr_prediction_weight,
            "skill_window_size": self.skill_window_size,
            "skill_conv_channels": self.skill_conv_channels,
            "skill_lstm_hidden": self.skill_lstm_hidden,
            "skill_epochs": self.skill_epochs,
            "skill_learning_rate": self.skill_learning_rate,
            "behavior_window_size": self.behavior_window_size,
            "behavior_hidden_dim": self.behavior_hidden_dim,
            "behavior_gru_hidden": self.behavior_gru_hidden,
            "behavior_epochs": self.behavior_epochs,
            "behavior_learning_rate": self.behavior_learning_rate,
            "openskill_beta": self.openskill_beta,
            "openskill_tau": self.openskill_tau,
            "dataset_path": self.dataset_path,
            "train_split": self.train_split,
            "val_split": self.val_split,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainingConfig:
        """Create from a dictionary, ignoring unknown keys."""
        known_fields = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)

    @classmethod
    def from_json(cls, path: str) -> TrainingConfig:
        """Load from a JSON file."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)

    def save(self, path: str) -> None:
        """Persist configuration to a JSON file."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def validate(self) -> list[str]:
        """Return a list of validation errors (empty if valid)."""
        errors: list[str] = []
        if self.model_type not in VALID_MODEL_TYPES:
            errors.append(
                f"Unknown model_type '{self.model_type}'. "
                f"Must be one of: {', '.join(sorted(VALID_MODEL_TYPES))}"
            )
        if self.cfr_iterations < 1:
            errors.append("cfr_iterations must be >= 1")
        if not 0.0 < self.train_split < 1.0:
            errors.append("train_split must be in (0, 1)")
        if not 0.0 < self.val_split < 1.0:
            errors.append("val_split must be in (0, 1)")
        if self.train_split + self.val_split >= 1.0:
            errors.append("train_split + val_split must be < 1.0")
        return errors


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------


@dataclass
class TrainingProgress:
    """Live progress state for a running training job."""

    iteration: int = 0
    total_iterations: int = 0
    elapsed_seconds: float = 0.0
    best_metric: float = float("inf")
    metrics_history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def progress_pct(self) -> float:
        """Percentage complete (0-100)."""
        if self.total_iterations == 0:
            return 0.0
        return self.iteration / self.total_iterations * 100.0

    @property
    def eta_seconds(self) -> float:
        """Estimated time remaining in seconds."""
        if self.iteration == 0:
            return 0.0
        rate = self.elapsed_seconds / self.iteration
        remaining = self.total_iterations - self.iteration
        return rate * remaining


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class TrainingResult:
    """Outcome of a completed training run."""

    model_type: str
    config: TrainingConfig
    progress: TrainingProgress
    final_metrics: dict[str, Any]
    model_path: str
    checkpoint_paths: list[str]
    training_log: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class TrainingOrchestrator:
    """Unified training orchestrator for all MR_POKER models.

    Usage::

        config = TrainingConfig.from_json("configs/training/cfr_default.json")
        orchestrator = TrainingOrchestrator(config)
        result = orchestrator.train()
    """

    def __init__(self, config: TrainingConfig) -> None:
        errors = config.validate()
        if errors:
            raise ValueError(
                "Invalid TrainingConfig:\n  " + "\n  ".join(errors)
            )

        self.config = config
        self.progress = TrainingProgress(
            total_iterations=self._total_iterations()
        )
        self._callbacks: list[Callable[[TrainingProgress], None]] = []
        self._training_log: list[dict[str, Any]] = []
        self._checkpoint_paths: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> TrainingResult:
        """Run complete training pipeline.

        Dispatches to the appropriate model-specific trainer based on
        ``config.model_type``.

        Returns:
            TrainingResult with final metrics, model path, and logs.
        """
        dispatch = {
            "cfr": self._train_cfr,
            "deep_cfr": self._train_deep_cfr,
            "pdcfr_plus": self._train_pdcfr_plus,
            "skill_estimator": self._train_skill_estimator,
            "behavior_predictor": self._train_behavior_predictor,
            "openskill": self._train_openskill,
        }
        trainer_fn = dispatch[self.config.model_type]

        logger.info(
            "Starting %s training (%d iterations)",
            self.config.model_type,
            self.progress.total_iterations,
        )
        start = time.monotonic()
        result = trainer_fn()
        result.progress.elapsed_seconds = time.monotonic() - start
        logger.info(
            "%s training complete in %.1fs — %s",
            self.config.model_type,
            result.progress.elapsed_seconds,
            result.model_path,
        )
        return result

    def on_progress(self, callback: Callable[[TrainingProgress], None]) -> None:
        """Register a callback invoked on every log interval."""
        self._callbacks.append(callback)

    # ------------------------------------------------------------------
    # Iteration budget helpers
    # ------------------------------------------------------------------

    def _total_iterations(self) -> int:
        """Calculate total iterations based on model type."""
        mt = self.config.model_type
        if mt in ("cfr", "deep_cfr", "pdcfr_plus"):
            return self.config.cfr_iterations
        if mt == "skill_estimator":
            return self.config.skill_epochs
        if mt == "behavior_predictor":
            return self.config.behavior_epochs
        if mt == "openskill":
            # OpenSkill calibration iterates over match records;
            # use cfr_iterations as a reasonable proxy for "rounds".
            return self.config.cfr_iterations
        return self.config.cfr_iterations

    # ------------------------------------------------------------------
    # Output directory helpers
    # ------------------------------------------------------------------

    def _ensure_output_dir(self) -> Path:
        out = Path(self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _model_filename(self, suffix: str = "") -> str:
        base = f"{self.config.model_type}"
        if self.config.cfr_mode and self.config.model_type == "cfr":
            base = f"{base}_{self.config.cfr_mode}"
        if suffix:
            base = f"{base}_{suffix}"
        return f"{base}.json"

    # ------------------------------------------------------------------
    # Logging / progress
    # ------------------------------------------------------------------

    def _log(self, iteration: int, metrics: dict[str, Any]) -> None:
        """Log training metrics and notify callbacks."""
        entry = {"iteration": iteration, "time": time.time(), **metrics}
        self._training_log.append(entry)
        self.progress.iteration = iteration
        self.progress.metrics_history.append(entry)

        logger.info(
            "[%s] iter %d/%d (%.1f%%) — %s",
            self.config.model_type,
            iteration,
            self.progress.total_iterations,
            self.progress.progress_pct,
            ", ".join(f"{k}={v}" for k, v in metrics.items()),
        )

        for cb in self._callbacks:
            try:
                cb(self.progress)
            except Exception:  # pragma: no cover
                logger.warning("Progress callback raised an exception", exc_info=True)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _checkpoint(self, model_state: Any, metrics: dict[str, Any]) -> str:
        """Save checkpoint to disk. Returns checkpoint file path."""
        out = self._ensure_output_dir()
        iteration = self.progress.iteration
        ckpt_name = f"{self.config.model_type}_ckpt_{iteration}.json"
        ckpt_path = out / ckpt_name

        payload: dict[str, Any] = {
            "model_type": self.config.model_type,
            "iteration": iteration,
            "metrics": metrics,
        }

        # Try to use the model's native save, falling back to generic JSON.
        if hasattr(model_state, "save"):
            model_state.save(str(ckpt_path))
            # Overwrite with enriched metadata
            try:
                existing = json.loads(ckpt_path.read_text(encoding="utf-8"))
                existing.update(payload)
                ckpt_path.write_text(
                    json.dumps(existing, indent=2), encoding="utf-8"
                )
            except Exception:
                pass
        else:
            payload["state"] = _serialise_state(model_state)
            ckpt_path.write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )

        self._checkpoint_paths.append(str(ckpt_path))
        logger.info("Checkpoint saved: %s", ckpt_path)
        return str(ckpt_path)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, model_state: Any) -> dict[str, Any]:
        """Evaluate current model state. Returns metric dict."""
        metrics: dict[str, Any] = {"iteration": self.progress.iteration}

        # CFR family: report number of info sets and average regret
        if hasattr(model_state, "cumulative_regret"):
            regret_map = model_state.cumulative_regret
            n_info = len(regret_map)
            total_regret = 0.0
            n_entries = 0
            for info_regrets in regret_map.values():
                for v in info_regrets.values():
                    total_regret += abs(v)
                    n_entries += 1
            avg_regret = total_regret / max(n_entries, 1)
            metrics["info_sets"] = n_info
            metrics["avg_abs_regret"] = round(avg_regret, 6)

        # Skill estimator: report skill rating + confidence
        if hasattr(model_state, "estimate"):
            try:
                est = model_state.estimate()
                if hasattr(est, "skill_rating"):
                    metrics["skill_rating"] = round(est.skill_rating, 4)
                    metrics["confidence"] = round(est.confidence, 4)
            except Exception:
                pass

        # Behavior predictor: report calibration error
        if hasattr(model_state, "calibration_error"):
            try:
                metrics["calibration_error"] = round(
                    model_state.calibration_error(), 6
                )
            except Exception:
                pass

        return metrics

    # ------------------------------------------------------------------
    # Model-specific training loops
    # ------------------------------------------------------------------

    def _train_cfr(self) -> TrainingResult:
        """Train CFR solver (vanilla, DCFR, MCCFR, VAD-CFR)."""
        from packages.cfr_agent.trainer import CFRTrainer

        mode_map = {"vad": "vadcfr"}
        mode = mode_map.get(self.config.cfr_mode, self.config.cfr_mode)

        trainer = CFRTrainer(
            small_blind=self.config.small_blind,
            big_blind=self.config.big_blind,
            starting_stack=self.config.starting_stack,
            seed=self.config.seed,
            mode=mode,
        )

        total = self.config.cfr_iterations
        last_ckpt = 0
        start = time.monotonic()

        for i in range(1, total + 1):
            # Run a single iteration via the trainer's internal loop
            trainer.train(iterations=1)
            self.progress.iteration = i
            self.progress.elapsed_seconds = time.monotonic() - start

            if i % self.config.log_interval == 0:
                metrics = self._evaluate(trainer.cfr_state)
                self._log(i, metrics)

            if i % self.config.eval_interval == 0:
                metrics = self._evaluate(trainer.cfr_state)
                metric_val = metrics.get("avg_abs_regret", float("inf"))
                if metric_val < self.progress.best_metric:
                    self.progress.best_metric = metric_val

            if i % self.config.checkpoint_interval == 0:
                metrics = self._evaluate(trainer.cfr_state)
                self._checkpoint(trainer.cfr_state, metrics)
                last_ckpt = i

        # Final save
        out = self._ensure_output_dir()
        model_path = str(out / self._model_filename("final"))
        trainer.cfr_state.save(model_path)

        # Save config alongside model
        self.config.save(str(out / "config.json"))

        final_metrics = self._evaluate(trainer.cfr_state)
        return TrainingResult(
            model_type="cfr",
            config=self.config,
            progress=self.progress,
            final_metrics=final_metrics,
            model_path=model_path,
            checkpoint_paths=list(self._checkpoint_paths),
            training_log=list(self._training_log),
        )

    def _train_deep_cfr(self) -> TrainingResult:
        """Train Deep CFR with neural approximation."""
        from packages.cfr_agent.deep_cfr import DeepCFRTrainer

        trainer = DeepCFRTrainer(
            small_blind=self.config.small_blind,
            big_blind=self.config.big_blind,
            starting_stack=self.config.starting_stack,
            hidden_dim=self.config.deep_cfr_hidden_dim,
            seed=self.config.seed,
        )

        total = self.config.cfr_iterations
        start = time.monotonic()

        for i in range(1, total + 1):
            trainer.train(iterations=1)
            self.progress.iteration = i
            self.progress.elapsed_seconds = time.monotonic() - start

            if i % self.config.log_interval == 0:
                metrics = self._evaluate(trainer.cfr_state)
                self._log(i, metrics)

            if i % self.config.eval_interval == 0:
                metrics = self._evaluate(trainer.cfr_state)
                metric_val = metrics.get("avg_abs_regret", float("inf"))
                if metric_val < self.progress.best_metric:
                    self.progress.best_metric = metric_val

            if i % self.config.checkpoint_interval == 0:
                metrics = self._evaluate(trainer.cfr_state)
                self._checkpoint(trainer.cfr_state, metrics)

        out = self._ensure_output_dir()
        model_path = str(out / self._model_filename("final"))
        trainer.cfr_state.save(model_path)
        self.config.save(str(out / "config.json"))

        final_metrics = self._evaluate(trainer.cfr_state)
        return TrainingResult(
            model_type="deep_cfr",
            config=self.config,
            progress=self.progress,
            final_metrics=final_metrics,
            model_path=model_path,
            checkpoint_paths=list(self._checkpoint_paths),
            training_log=list(self._training_log),
        )

    def _train_pdcfr_plus(self) -> TrainingResult:
        """Train PDCFR+ with optimistic mirror descent."""
        from packages.cfr_agent.pdcfr_plus import PDCFRPlusTrainer

        mode_map = {"vad": "dcfr"}
        mode = mode_map.get(self.config.cfr_mode, self.config.cfr_mode)

        trainer = PDCFRPlusTrainer(
            small_blind=self.config.small_blind,
            big_blind=self.config.big_blind,
            starting_stack=self.config.starting_stack,
            seed=self.config.seed,
            mode=mode,
            prediction_weight=self.config.pdcfr_prediction_weight,
        )

        total = self.config.cfr_iterations
        start = time.monotonic()

        for i in range(1, total + 1):
            trainer.train(iterations=1)
            self.progress.iteration = i
            self.progress.elapsed_seconds = time.monotonic() - start

            if i % self.config.log_interval == 0:
                metrics = self._evaluate(trainer.state)
                self._log(i, metrics)

            if i % self.config.eval_interval == 0:
                metrics = self._evaluate(trainer.state)
                metric_val = metrics.get("avg_abs_regret", float("inf"))
                if metric_val < self.progress.best_metric:
                    self.progress.best_metric = metric_val

            if i % self.config.checkpoint_interval == 0:
                metrics = self._evaluate(trainer.state)
                self._checkpoint(trainer.state, metrics)

        out = self._ensure_output_dir()
        model_path = str(out / self._model_filename("final"))
        trainer.state.save(model_path)
        self.config.save(str(out / "config.json"))

        # Also save a CFRState-compatible version for downstream use
        cfr_compat = trainer.to_cfr_state()
        cfr_compat.save(str(out / "cfr_compat.json"))

        final_metrics = self._evaluate(trainer.state)
        return TrainingResult(
            model_type="pdcfr_plus",
            config=self.config,
            progress=self.progress,
            final_metrics=final_metrics,
            model_path=model_path,
            checkpoint_paths=list(self._checkpoint_paths),
            training_log=list(self._training_log),
        )

    def _train_skill_estimator(self) -> TrainingResult:
        """Train CNN-BiLSTM skill estimator on decision sequences.

        Without a real dataset, this generates synthetic training data
        using the SyntheticPlayerGenerator and simulates decision
        sequences at various skill levels.
        """
        from packages.opponent_model.skill_estimator import (
            DecisionFeature,
            EnhancedSkillEstimator,
        )
        from packages.opponent_model.synthetic_players import (
            SyntheticPlayerGenerator,
        )

        import random as stdlib_random

        rng = stdlib_random.Random(self.config.seed)

        estimator = EnhancedSkillEstimator(
            window_size=self.config.skill_window_size,
            seed=self.config.seed,
        )

        # Generate synthetic training data from player profiles
        gen = SyntheticPlayerGenerator(seed=self.config.seed)
        players = gen.generate_batch(50)

        total = self.config.skill_epochs
        start = time.monotonic()

        for epoch in range(1, total + 1):
            # Each epoch: pick a random player and simulate decisions
            player = rng.choice(players)
            skill = player.skill_level

            # Generate a batch of decisions characteristic of this skill level
            n_decisions = rng.randint(5, self.config.skill_window_size)
            for _ in range(n_decisions):
                decision = DecisionFeature(
                    action_type=_sample_action_for_skill(skill, rng),
                    decision_time_norm=_sample_timing_for_skill(skill, rng),
                    bet_fraction=_sample_sizing_for_skill(skill, rng),
                    position_score=rng.random(),
                    street=rng.choice([0.0, 0.33, 0.67, 1.0]),
                    pot_committed=rng.random() * 0.5,
                    aggression_context=player.stats.aggression_factor / 10.0,
                    hand_strength=rng.random(),
                )
                estimator.record_decision(decision)

            self.progress.iteration = epoch
            self.progress.elapsed_seconds = time.monotonic() - start

            if epoch % self.config.log_interval == 0:
                metrics = self._evaluate(estimator)
                self._log(epoch, metrics)

            if epoch % self.config.checkpoint_interval == 0:
                metrics = self._evaluate(estimator)
                self._checkpoint(
                    {"epoch": epoch, "decisions": estimator.decisions_recorded},
                    metrics,
                )

        out = self._ensure_output_dir()
        model_path = str(out / self._model_filename("final"))

        # Save estimator state as JSON summary
        final_estimate = estimator.estimate()
        bayes_mu, bayes_sigma = estimator.bayesian_estimate
        state = {
            "model_type": "skill_estimator",
            "epochs": total,
            "decisions_recorded": estimator.decisions_recorded,
            "final_skill_rating": round(final_estimate.skill_rating, 4),
            "final_confidence": round(final_estimate.confidence, 4),
            "final_label": final_estimate.skill_label,
            "bayesian_mu": round(bayes_mu, 4),
            "bayesian_sigma": round(bayes_sigma, 4),
        }
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        Path(model_path).write_text(
            json.dumps(state, indent=2), encoding="utf-8"
        )
        self.config.save(str(out / "config.json"))

        final_metrics = self._evaluate(estimator)
        return TrainingResult(
            model_type="skill_estimator",
            config=self.config,
            progress=self.progress,
            final_metrics=final_metrics,
            model_path=model_path,
            checkpoint_paths=list(self._checkpoint_paths),
            training_log=list(self._training_log),
        )

    def _train_behavior_predictor(self) -> TrainingResult:
        """Train GRU behavior predictor on action sequences.

        Without a real dataset, generates synthetic observation
        sequences from synthetic player profiles.
        """
        from packages.opponent_model.behavior_prediction import (
            ActionEvent,
            BehaviorPredictor,
        )
        from packages.opponent_model.synthetic_players import (
            SyntheticPlayerGenerator,
        )
        from packages.common.types import ActionType

        import random as stdlib_random

        rng = stdlib_random.Random(self.config.seed)

        predictor = BehaviorPredictor(
            window_size=self.config.behavior_window_size,
            hidden_dim=self.config.behavior_hidden_dim,
            learning_rate=self.config.behavior_learning_rate,
            seed=self.config.seed,
            use_gru=True,
            gru_hidden_dim=self.config.behavior_gru_hidden,
        )

        gen = SyntheticPlayerGenerator(seed=self.config.seed)
        players = gen.generate_batch(50)

        total = self.config.behavior_epochs
        start = time.monotonic()
        all_actions = list(ActionType)
        streets = ["preflop", "flop", "turn", "river"]

        for epoch in range(1, total + 1):
            player = rng.choice(players)

            # Simulate a sequence of actions for this player
            n_actions = rng.randint(5, 20)
            for _ in range(n_actions):
                action = _sample_action_type_for_player(player, rng)
                event = ActionEvent(
                    action=action,
                    street=rng.choice(streets),
                    bet_fraction=rng.random() * 2.0,
                    position=rng.randint(0, 1),
                    facing_bet=rng.random() > 0.5,
                )
                predictor.observe(event)

            # Train on accumulated data
            loss = predictor.train_step(epochs=1)

            self.progress.iteration = epoch
            self.progress.elapsed_seconds = time.monotonic() - start

            if epoch % self.config.log_interval == 0:
                metrics: dict[str, Any] = {"loss": round(loss, 6)}
                metrics.update(self._evaluate(predictor))
                self._log(epoch, metrics)

            if epoch % self.config.checkpoint_interval == 0:
                metrics = {"loss": round(loss, 6)}
                self._checkpoint(
                    {"epoch": epoch, "observations": predictor.observation_count},
                    metrics,
                )

        out = self._ensure_output_dir()
        model_path = str(out / self._model_filename("final"))

        state = {
            "model_type": "behavior_predictor",
            "epochs": total,
            "observations": predictor.observation_count,
            "final_calibration_error": round(predictor.calibration_error(), 6),
        }
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        Path(model_path).write_text(
            json.dumps(state, indent=2), encoding="utf-8"
        )
        self.config.save(str(out / "config.json"))

        final_metrics = self._evaluate(predictor)
        return TrainingResult(
            model_type="behavior_predictor",
            config=self.config,
            progress=self.progress,
            final_metrics=final_metrics,
            model_path=model_path,
            checkpoint_paths=list(self._checkpoint_paths),
            training_log=list(self._training_log),
        )

    def _train_openskill(self) -> TrainingResult:
        """Calibrate OpenSkill ratings from match history.

        Uses Bayesian Gaussian updates (Plackett-Luce approximation)
        to calibrate player ratings from simulated match results.
        """
        from packages.opponent_model.skill_estimator import BayesianSkillTracker
        from packages.opponent_model.synthetic_players import (
            SyntheticPlayerGenerator,
        )

        import math
        import random as stdlib_random

        rng = stdlib_random.Random(self.config.seed)

        # Create a pool of players with trackers
        gen = SyntheticPlayerGenerator(seed=self.config.seed)
        players = gen.generate_batch(20)

        trackers: dict[str, BayesianSkillTracker] = {}
        true_skills: dict[str, float] = {}
        for p in players:
            trackers[p.name] = BayesianSkillTracker(
                prior_mu=0.5,
                prior_sigma=self.config.openskill_beta / 6.0,
                dynamics_sigma=self.config.openskill_tau / 3.0,
            )
            true_skills[p.name] = p.skill_level

        total = self.config.cfr_iterations
        start = time.monotonic()

        for i in range(1, total + 1):
            # Simulate a match between two random players
            p1, p2 = rng.sample(players, 2)

            # Outcome based on true skill difference (logistic model)
            skill_diff = true_skills[p1.name] - true_skills[p2.name]
            p1_win_prob = 1.0 / (1.0 + math.exp(-skill_diff * 4.0))
            p1_won = rng.random() < p1_win_prob

            if p1_won:
                trackers[p1.name].update(0.7 + rng.random() * 0.3)
                trackers[p2.name].update(rng.random() * 0.4)
            else:
                trackers[p2.name].update(0.7 + rng.random() * 0.3)
                trackers[p1.name].update(rng.random() * 0.4)

            self.progress.iteration = i
            self.progress.elapsed_seconds = time.monotonic() - start

            if i % self.config.log_interval == 0:
                # Compute correlation between estimated and true ratings
                est_ratings = []
                true_ratings = []
                for name in trackers:
                    mu, _ = trackers[name].estimate()
                    est_ratings.append(mu)
                    true_ratings.append(true_skills[name])
                correlation = _pearson_correlation(est_ratings, true_ratings)
                metrics = {
                    "rating_correlation": round(correlation, 4),
                    "avg_sigma": round(
                        sum(t.sigma for t in trackers.values()) / len(trackers),
                        4,
                    ),
                }
                self._log(i, metrics)

            if i % self.config.checkpoint_interval == 0:
                ratings = {
                    name: {"mu": round(t.estimate()[0], 4), "sigma": round(t.sigma, 4)}
                    for name, t in trackers.items()
                }
                self._checkpoint({"ratings": ratings}, {"iteration": i})

        out = self._ensure_output_dir()
        model_path = str(out / self._model_filename("final"))

        # Final ratings
        final_ratings = {}
        for name, tracker in trackers.items():
            mu, sigma = tracker.estimate()
            final_ratings[name] = {
                "mu": round(mu, 4),
                "sigma": round(sigma, 4),
                "true_skill": round(true_skills[name], 4),
                "confidence": round(tracker.confidence, 4),
            }

        est_r = [v["mu"] for v in final_ratings.values()]
        true_r = [v["true_skill"] for v in final_ratings.values()]
        correlation = _pearson_correlation(est_r, true_r)

        state = {
            "model_type": "openskill",
            "iterations": total,
            "ratings": final_ratings,
            "final_correlation": round(correlation, 4),
            "beta": self.config.openskill_beta,
            "tau": self.config.openskill_tau,
        }
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        Path(model_path).write_text(
            json.dumps(state, indent=2), encoding="utf-8"
        )
        self.config.save(str(out / "config.json"))

        final_metrics = {
            "rating_correlation": round(correlation, 4),
            "n_players": len(final_ratings),
        }
        return TrainingResult(
            model_type="openskill",
            config=self.config,
            progress=self.progress,
            final_metrics=final_metrics,
            model_path=model_path,
            checkpoint_paths=list(self._checkpoint_paths),
            training_log=list(self._training_log),
        )


# ---------------------------------------------------------------------------
# Helpers (module-private)
# ---------------------------------------------------------------------------


def _serialise_state(obj: Any) -> Any:
    """Best-effort JSON serialisation of an arbitrary state object."""
    if isinstance(obj, dict):
        return {str(k): _serialise_state(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialise_state(v) for v in obj]
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    if hasattr(obj, "__dict__"):
        return _serialise_state(vars(obj))
    return str(obj)


def _sample_action_for_skill(skill: float, rng: Any) -> float:
    """Simulate action-type encoding correlated with skill level.

    Higher skill -> more bets/raises (0.6-1.0); lower -> more checks/calls.
    """
    if rng.random() < skill * 0.5 + 0.2:
        return rng.choice([0.6, 0.8, 1.0])  # bet, raise, all_in
    return rng.choice([0.0, 0.2, 0.4])  # fold, check, call


def _sample_timing_for_skill(skill: float, rng: Any) -> float:
    """Skilled players have more consistent, moderate timing."""
    base = 0.4 + skill * 0.2
    noise = rng.gauss(0, 0.15 * (1.0 - skill * 0.5))
    return max(0.0, min(1.0, base + noise))


def _sample_sizing_for_skill(skill: float, rng: Any) -> float:
    """Skilled players use more precise bet sizing."""
    if rng.random() > 0.3:
        # Pot-based sizing
        base = 0.5 + skill * 0.2
        noise = rng.gauss(0, 0.1 * (1.0 - skill * 0.5))
        return max(0.0, min(2.0, base + noise))
    return 0.0


def _sample_action_type_for_player(player: Any, rng: Any) -> Any:
    """Sample an ActionType weighted by the player's stats."""
    from packages.common.types import ActionType

    stats = player.stats
    weights = [
        max(0.01, 1.0 - stats.vpip),                  # FOLD
        max(0.01, 1.0 - stats.aggression_factor / 5),  # CHECK
        max(0.01, stats.vpip - stats.pfr),             # CALL
        max(0.01, stats.cbet_freq * 0.5),              # BET
        max(0.01, stats.pfr),                          # RAISE
        max(0.01, stats.overbet_freq),                 # ALL_IN
    ]
    actions = [
        ActionType.FOLD, ActionType.CHECK, ActionType.CALL,
        ActionType.BET, ActionType.RAISE, ActionType.ALL_IN,
    ]
    total = sum(weights)
    r = rng.random() * total
    cumsum = 0.0
    for action, w in zip(actions, weights):
        cumsum += w
        if r <= cumsum:
            return action
    return ActionType.CHECK


def _pearson_correlation(xs: list[float], ys: list[float]) -> float:
    """Compute Pearson correlation between two lists."""
    n = len(xs)
    if n < 2:
        return 0.0
    import math

    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)

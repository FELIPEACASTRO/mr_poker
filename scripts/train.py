"""Training CLI for MR_POKER models.

Usage:
    python scripts/train.py --config configs/training/cfr_default.json
    python scripts/train.py --model-type cfr --iterations 5000
    python scripts/train.py --config configs/training/pdcfr_plus.json --seed 123
    python scripts/train.py --model-type skill_estimator --epochs 200
    python scripts/train.py --config configs/training/deep_cfr.json --output-dir var/experiments/run1

Overrides:
    Any CLI flag can override the corresponding config-file value.
    Flags not provided fall back to the config file, then to built-in defaults.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so that ``packages.*`` imports work
# when the script is invoked directly via ``python scripts/train.py``.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from services.training_service.orchestrator import (
    VALID_MODEL_TYPES,
    TrainingConfig,
    TrainingOrchestrator,
    TrainingResult,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train MR_POKER models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to a training config JSON file.",
    )
    p.add_argument(
        "--model-type",
        type=str,
        choices=sorted(VALID_MODEL_TYPES),
        default=None,
        help="Model type to train (overrides config file).",
    )
    p.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of CFR / OpenSkill iterations (overrides cfr_iterations).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs for skill_estimator / behavior_predictor.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for models and checkpoints.",
    )
    p.add_argument(
        "--cfr-mode",
        type=str,
        choices=["vanilla", "dcfr", "mccfr", "vad"],
        default=None,
        help="CFR variant (overrides config).",
    )
    p.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="Save checkpoint every N iterations.",
    )
    p.add_argument(
        "--log-interval",
        type=int,
        default=None,
        help="Log metrics every N iterations.",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable DEBUG logging.",
    )
    p.add_argument(
        "--quiet", "-q",
        action="store_true",
        default=False,
        help="Suppress all logging except errors.",
    )

    return p


def _apply_overrides(config: TrainingConfig, args: argparse.Namespace) -> None:
    """Apply CLI overrides to a TrainingConfig in-place."""
    if args.model_type is not None:
        config.model_type = args.model_type
    if args.seed is not None:
        config.seed = args.seed
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.cfr_mode is not None:
        config.cfr_mode = args.cfr_mode
    if args.checkpoint_interval is not None:
        config.checkpoint_interval = args.checkpoint_interval
    if args.log_interval is not None:
        config.log_interval = args.log_interval

    # Iterations / epochs
    if args.iterations is not None:
        config.cfr_iterations = args.iterations
    if args.epochs is not None:
        if config.model_type == "skill_estimator":
            config.skill_epochs = args.epochs
        elif config.model_type == "behavior_predictor":
            config.behavior_epochs = args.epochs
        else:
            # Generic fallback: treat as cfr_iterations
            config.cfr_iterations = args.epochs


def _print_summary(result: TrainingResult) -> None:
    """Print a concise human-readable training summary."""
    p = result.progress
    print()
    print("=" * 60)
    print(f"  Training complete: {result.model_type}")
    print("=" * 60)
    print(f"  Iterations:    {p.iteration}/{p.total_iterations}")
    print(f"  Elapsed:       {p.elapsed_seconds:.1f}s")
    print(f"  Best metric:   {p.best_metric}")
    print(f"  Model saved:   {result.model_path}")
    print(f"  Checkpoints:   {len(result.checkpoint_paths)}")
    if result.final_metrics:
        print(f"  Final metrics: {json.dumps(result.final_metrics, indent=4)}")
    print("=" * 60)
    print()


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Configure logging
    if args.quiet:
        log_level = logging.ERROR
    elif args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Build config
    if args.config is not None:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: config file not found: {config_path}", file=sys.stderr)
            return 1
        config = TrainingConfig.from_json(str(config_path))
    elif args.model_type is not None:
        config = TrainingConfig(model_type=args.model_type)
    else:
        print(
            "Error: must provide either --config or --model-type.",
            file=sys.stderr,
        )
        parser.print_usage(sys.stderr)
        return 1

    _apply_overrides(config, args)

    # Validate
    errors = config.validate()
    if errors:
        for e in errors:
            print(f"Config error: {e}", file=sys.stderr)
        return 1

    # Run training
    orchestrator = TrainingOrchestrator(config)

    try:
        result = orchestrator.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        logging.getLogger(__name__).exception("Training failed")
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    _print_summary(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())

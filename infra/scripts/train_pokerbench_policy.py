"""Train a PolicyTableModel from the RZ412/PokerBench HuggingFace dataset.

This standalone script loads PokerBench solver-optimal decisions, converts them
via the PokerBenchAdapter into bucketed training rows, trains a PolicyTableModel
using majority voting, and persists the result to ``var/models/``.

Usage::

    python infra/scripts/train_pokerbench_policy.py --max-rows 5000 --model-name pokerbench_v1
    python infra/scripts/train_pokerbench_policy.py --config medium --equity-samples 50
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PolicyTableModel from PokerBench")
    parser.add_argument("--max-rows", type=int, default=0, help="Max rows to use (0 = all)")
    parser.add_argument("--equity-samples", type=int, default=0, help="MC equity samples (0 = heuristic)")
    parser.add_argument("--model-name", type=str, default=None, help="Model name (auto-generated if omitted)")
    parser.add_argument("--config", type=str, default="easy", choices=["easy", "medium", "hard"], help="PokerBench config")
    parser.add_argument("--model-dir", type=str, default="var/models", help="Directory to save models")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("'datasets' library required. Install: pip install datasets")
        sys.exit(1)

    from packages.dataset_builder.pokerbench_adapter import PokerBenchAdapter
    from packages.policy_table.registry import PolicyRegistry
    from packages.policy_table.trainer import PolicyTableTrainer

    model_name = args.model_name
    if model_name is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        model_name = f"pokerbench_{args.config}_{stamp}"

    # Load dataset
    logger.info("Loading PokerBench dataset (config=%s)...", args.config)
    t0 = time.perf_counter()
    ds = load_dataset("RZ412/PokerBench", args.config, split="train")
    logger.info("Loaded %d rows in %.1fs", len(ds), time.perf_counter() - t0)

    # Limit rows if requested
    rows = list(ds)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
        logger.info("Using first %d rows", len(rows))

    # Convert
    logger.info("Converting rows (equity_samples=%d)...", args.equity_samples)
    adapter = PokerBenchAdapter(equity_samples=args.equity_samples)
    t0 = time.perf_counter()
    converted = adapter.convert_batch(rows, config=args.config)
    elapsed = time.perf_counter() - t0
    logger.info("Converted %d/%d rows in %.1fs (%.0f rows/s)", len(converted), len(rows), elapsed, len(converted) / max(elapsed, 0.001))

    if not converted:
        logger.error("No rows converted successfully. Exiting.")
        sys.exit(1)

    # Train
    logger.info("Training PolicyTableModel '%s'...", model_name)
    trainer = PolicyTableTrainer()
    model = trainer.train(converted, model_name=model_name)
    logger.info(
        "Model trained: %d buckets from %d rows",
        model.metadata["bucket_count"],
        model.metadata["row_count"],
    )

    # Save
    registry = PolicyRegistry(args.model_dir)
    saved = registry.save(model)
    logger.info("Model saved: %s", saved["path"])

    # Summary
    print("\n=== Training Summary ===")
    print(f"  Model ID:     {model.model_id}")
    print(f"  Model Name:   {model_name}")
    print(f"  Config:       {args.config}")
    print(f"  Input Rows:   {len(rows)}")
    print(f"  Converted:    {len(converted)}")
    print(f"  Buckets:      {model.metadata['bucket_count']}")
    print(f"  Saved To:     {saved['path']}")

    # Action distribution
    from collections import Counter

    action_counts = Counter(row["label_action"] for row in converted)
    print(f"\n  Action Distribution:")
    for action, count in action_counts.most_common():
        pct = count / len(converted) * 100
        print(f"    {action:10s} {count:6d} ({pct:.1f}%)")


if __name__ == "__main__":
    main()

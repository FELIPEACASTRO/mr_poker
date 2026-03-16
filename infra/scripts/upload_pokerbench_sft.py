"""Upload PokerBench as a chat-formatted SFT dataset to HuggingFace Hub.

Converts each RZ412/PokerBench row into the chat messages format required by
TRL SFTTrainer and pushes the result to ``felipesp1983/pokerbench-sft-chat``.

Usage::

    python infra/scripts/upload_pokerbench_sft.py
    python infra/scripts/upload_pokerbench_sft.py --config medium --max-rows 10000
    python infra/scripts/upload_pokerbench_sft.py --repo-id felipesp1983/pokerbench-sft-chat
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a GTO poker solver for 6-max No Limit Texas Hold'em. "
    "Given a game state description, respond with ONLY the optimal action. "
    "Valid actions: check, fold, call, bet X, raise X, all-in. "
    "Do not explain your reasoning."
)


def convert_to_chat(row: dict) -> dict:
    """Convert a PokerBench row to chat messages format for SFT."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["instruction"]},
            {"role": "assistant", "content": row["output"].strip()},
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload PokerBench as SFT dataset to HF Hub")
    parser.add_argument("--config", type=str, default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--max-rows", type=int, default=0, help="Max rows (0 = all)")
    parser.add_argument("--repo-id", type=str, default="felipesp1983/pokerbench-sft-chat")
    parser.add_argument("--private", action="store_true", help="Make dataset private")
    args = parser.parse_args()

    try:
        from datasets import Dataset, load_dataset
    except ImportError:
        logger.error("'datasets' library required. Install: pip install datasets")
        sys.exit(1)

    # Load source dataset
    logger.info("Loading PokerBench (config=%s)...", args.config)
    ds = load_dataset("RZ412/PokerBench", args.config, split="train")
    logger.info("Loaded %d rows", len(ds))

    rows = list(ds)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
        logger.info("Using first %d rows", len(rows))

    # Convert to chat format
    logger.info("Converting to chat format...")
    chat_rows = [convert_to_chat(row) for row in rows]

    # Also load test split if available
    try:
        ds_test = load_dataset("RZ412/PokerBench", args.config, split="test")
        test_rows = [convert_to_chat(row) for row in ds_test]
        logger.info("Test split: %d rows", len(test_rows))
    except Exception:
        test_rows = []
        logger.info("No test split available, skipping")

    # Create HF Dataset objects
    train_dataset = Dataset.from_list(chat_rows)
    logger.info("Train dataset: %d rows, columns: %s", len(train_dataset), train_dataset.column_names)

    # Push to Hub
    logger.info("Pushing to %s...", args.repo_id)
    train_dataset.push_to_hub(args.repo_id, split="train", private=args.private)

    if test_rows:
        test_dataset = Dataset.from_list(test_rows)
        test_dataset.push_to_hub(args.repo_id, split="test", private=args.private)

    logger.info("Done! Dataset at: https://huggingface.co/datasets/%s", args.repo_id)

    # Print sample
    sample = chat_rows[0]
    print("\n=== Sample Row ===")
    for msg in sample["messages"]:
        print(f"  [{msg['role']}]: {msg['content'][:100]}...")


if __name__ == "__main__":
    main()

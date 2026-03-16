# /// script
# dependencies = ["datasets>=3.0.0", "huggingface-hub>=0.28.0"]
# ///

"""Convert RZ412/PokerBench to SFT chat format and push to Hub.

This script is designed to run as a HuggingFace Job (UV script).

Submit via::

    hf jobs uv run --flavor cpu-basic --timeout 30m --secrets HF_TOKEN pokerbench_upload_hfjob.py
"""

import os
from datasets import load_dataset, DatasetDict

assert "HF_TOKEN" in os.environ, "HF_TOKEN required!"
token = os.environ["HF_TOKEN"]

SYSTEM_PROMPT = (
    "You are a GTO poker solver for 6-max No Limit Texas Hold'em. "
    "Given a game state description, respond with ONLY the optimal action. "
    "Valid actions: check, fold, call, bet X, raise X, all-in. "
    "Do not explain your reasoning."
)

TARGET_REPO = "felipesp1983/pokerbench-sft-chat"


def convert_to_chat(example):
    """Convert a PokerBench row to chat messages format."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]
    return {"messages": messages}


print("Loading RZ412/PokerBench ...")
train_ds = load_dataset("RZ412/PokerBench", split="train")
test_ds = load_dataset("RZ412/PokerBench", split="test")

print(f"  train: {len(train_ds)} rows")
print(f"  test:  {len(test_ds)} rows")

print("\nConverting to chat format ...")
train_chat = train_ds.map(convert_to_chat, remove_columns=train_ds.column_names)
test_chat = test_ds.map(convert_to_chat, remove_columns=test_ds.column_names)

print(f"\nPushing to {TARGET_REPO} ...")
dataset_dict = DatasetDict({"train": train_chat, "test": test_chat})
dataset_dict.push_to_hub(TARGET_REPO, token=token)

print(f"\nDone! Dataset at: https://huggingface.co/datasets/{TARGET_REPO}")

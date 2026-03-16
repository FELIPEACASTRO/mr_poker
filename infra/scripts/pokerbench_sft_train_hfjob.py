# /// script
# dependencies = [
#     "trl>=0.12.0",
#     "peft>=0.7.0",
#     "trackio",
#     "datasets>=3.0.0",
#     "accelerate>=0.30.0",
#     "bitsandbytes>=0.43.0",
# ]
# ///

"""SFT fine-tune Qwen2.5-1.5B-Instruct on pokerbench-sft-chat with LoRA.

This script is designed to run as a HuggingFace Job on A10G GPU.

Submit via::

    hf jobs uv run --flavor a10g-large --timeout 4h --secrets HF_TOKEN pokerbench_sft_train_hfjob.py
"""

import os
import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import trackio

assert "HF_TOKEN" in os.environ, "HF_TOKEN required!"
token = os.environ["HF_TOKEN"]

# -- Config --
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_ID = "felipesp1983/pokerbench-sft-chat"
OUTPUT_MODEL = "felipesp1983/poker-solver-qwen-1.5b-sft"

print(f"Loading dataset: {DATASET_ID}")
train_ds = load_dataset(DATASET_ID, split="train")
eval_ds = load_dataset(DATASET_ID, split="test")
print(f"  train: {len(train_ds)} rows, test: {len(eval_ds)} rows")

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

# SFT config
training_args = SFTConfig(
    output_dir="./poker-solver-sft",
    num_train_epochs=1,
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    max_length=512,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    gradient_checkpointing=True,
    logging_steps=50,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    eval_strategy="steps",
    eval_steps=500,
    push_to_hub=True,
    hub_model_id=OUTPUT_MODEL,
    hub_strategy="every_save",
    hub_token=token,
    report_to="trackio",
    project="mr_poker",
    run_name="pokerbench-sft-qwen1.5b-v1",
    bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
    fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
)

print(f"Loading model: {BASE_MODEL}")
trainer = SFTTrainer(
    model=BASE_MODEL,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    peft_config=lora_config,
    args=training_args,
)

print("Starting training ...")
trainer.train()

print("Pushing final model to Hub ...")
trainer.push_to_hub()

trackio.finish()

print(f"\nTraining complete! Model at: https://huggingface.co/{OUTPUT_MODEL}")

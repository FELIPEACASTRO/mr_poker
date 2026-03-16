# ============================================================================
# mr_poker — SFT Fine-tune Qwen2.5-1.5B on PokerBench (Google Colab Pro)
# ============================================================================
#
# Run this in a Colab notebook with GPU runtime (A100/V100/T4).
#
# Cell 1: Install dependencies
# !pip install -q trl>=0.12.0 peft>=0.7.0 datasets>=3.0.0 accelerate>=0.30.0 bitsandbytes>=0.43.0 trackio
#
# Cell 2: Login to HuggingFace
# from huggingface_hub import notebook_login
# notebook_login()
#
# Cell 3: Run this script
# %run train_pokerbench_colab.py
# ============================================================================

import os
import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import trackio

# -- Config --
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_ID = "felipesp1983/pokerbench-sft-chat"
OUTPUT_MODEL = "felipesp1983/poker-solver-qwen-1.5b-sft"

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB" if torch.cuda.is_available() else "N/A")

# -- Load dataset --
print(f"\nLoading dataset: {DATASET_ID}")
train_ds = load_dataset(DATASET_ID, split="train")
eval_ds = load_dataset(DATASET_ID, split="test")
print(f"  train: {len(train_ds)} rows, test: {len(eval_ds)} rows")

# -- LoRA config --
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

# -- Detect GPU capabilities --
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
use_fp16 = torch.cuda.is_available() and not use_bf16

# -- Adjust batch size based on GPU --
gpu_mem_gb = torch.cuda.get_device_properties(0).total_mem / 1e9 if torch.cuda.is_available() else 0
if gpu_mem_gb >= 40:  # A100
    batch_size = 8
    grad_accum = 4
elif gpu_mem_gb >= 16:  # V100/T4
    batch_size = 4
    grad_accum = 8
else:
    batch_size = 2
    grad_accum = 16

print(f"\nTraining config: batch_size={batch_size}, grad_accum={grad_accum}, bf16={use_bf16}, fp16={use_fp16}")

# -- SFT config --
training_args = SFTConfig(
    output_dir="./poker-solver-sft",
    num_train_epochs=1,
    learning_rate=2e-4,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=grad_accum,
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
    report_to="trackio",
    project="mr_poker",
    run_name="pokerbench-sft-qwen1.5b-colab",
    bf16=use_bf16,
    fp16=use_fp16,
)

# -- Train --
print(f"\nLoading model: {BASE_MODEL}")
trainer = SFTTrainer(
    model=BASE_MODEL,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    peft_config=lora_config,
    args=training_args,
)

print(f"\nStarting training ({len(train_ds)} rows, 1 epoch)...")
print(f"Estimated steps: {len(train_ds) // (batch_size * grad_accum)}")
trainer.train()

print("\nPushing final model to Hub ...")
trainer.push_to_hub()

trackio.finish()

print(f"\n{'='*60}")
print(f"Training complete!")
print(f"Model: https://huggingface.co/{OUTPUT_MODEL}")
print(f"{'='*60}")

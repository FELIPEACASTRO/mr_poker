# ============================================================
# MR_POKER - Treinar modelo poker no Google Colab Pro
# ============================================================
# INSTRUCOES:
# 1. Abra colab.research.google.com
# 2. Crie um novo notebook
# 3. Va em Runtime > Change runtime type > GPU (A100 ou T4)
# 4. Cole TUDO abaixo numa unica celula e clique PLAY
# 5. Quando pedir o token HF, cole seu token de:
#    https://huggingface.co/settings/tokens
# ============================================================

# Passo 1: Instalar
import subprocess, sys
print("Instalando dependencias...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "trl>=0.12.0", "peft>=0.7.0", "datasets>=3.0.0",
    "accelerate>=0.30.0", "bitsandbytes>=0.43.0", "huggingface_hub"])
print("OK!")

# Passo 2: Login no HuggingFace
from huggingface_hub import notebook_login
notebook_login()

# Passo 3: Verificar GPU
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print("GPU: " + gpu_name + " (" + str(round(gpu_mem, 1)) + " GB)")
else:
    print("ERRO: GPU nao detectada! Va em Runtime > Change runtime type > GPU")
    raise SystemExit("Sem GPU")

# Passo 4: Carregar dataset
from datasets import load_dataset
print("Carregando dataset...")
train_ds = load_dataset("felipesp1983/pokerbench-sft-chat", split="train")
eval_ds = load_dataset("felipesp1983/pokerbench-sft-chat", split="test")
print("Dataset: " + str(len(train_ds)) + " treino, " + str(len(eval_ds)) + " teste")

# Passo 5: Configurar e treinar
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM",
)

use_bf16 = torch.cuda.is_bf16_supported()
batch_size = 8 if gpu_mem >= 40 else (4 if gpu_mem >= 14 else 2)
grad_accum = 32 // batch_size

print("Configuracao: batch=" + str(batch_size) + ", bf16=" + str(use_bf16))

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
    logging_steps=25,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=500,
    push_to_hub=True,
    hub_model_id="felipesp1983/poker-solver-qwen-1.5b-sft",
    hub_strategy="every_save",
    report_to="none",
    bf16=use_bf16,
    fp16=not use_bf16,
)

print("Carregando modelo Qwen2.5-1.5B...")
trainer = SFTTrainer(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    peft_config=lora_config,
    args=training_args,
)

steps = len(train_ds) // (batch_size * grad_accum)
print("Iniciando treino: " + str(steps) + " steps, 1 epoca")
print("=" * 50)
trainer.train()

# Passo 6: Publicar
print("Publicando modelo no HuggingFace...")
trainer.push_to_hub()
print("=" * 50)
print("PRONTO! Modelo publicado em:")
print("https://huggingface.co/felipesp1983/poker-solver-qwen-1.5b-sft")
print("=" * 50)

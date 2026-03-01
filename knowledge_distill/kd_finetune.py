# kd_finetune.py
import os, random
from dataclasses import dataclass
from typing import Dict, List, Any

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model

# ------------------
# CONFIG
# ------------------
BASE_MODEL = "/root/out/llama-31-8b-dense2of4/dense"
KD_JSONL  = "/root/data/kd_data/kd_10000.jsonl"
OUT_DIR    = "/root/out/pruned-llama31-8b-kd-lora"

HF_TOKEN = os.environ.get("HF_TOKEN", None)

SEED = 0
MAX_SEQ_LEN = 1024
EVAL_RATIO = 0.02

USE_QLORA = False
print(f"qlora: {USE_QLORA}")

LR = 2e-5
MAX_STEPS = 500
BATCH = 1
GRAD_ACCUM = 32

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

# 防止“全 -100”导致 loss 无 grad
MIN_SUP_TOKENS = 32  # 每条样本至少留 32 个 assistant token 参与监督

# ------------------
# Utilities
# ------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_prompt_and_full(tok, messages: List[Dict]) -> Dict[str, str]:
    # 只使用第一轮 user/assistant
    user = messages[0]["content"]
    assistant = messages[1]["content"]

    prompt_msgs = [{"role": "user", "content": user}]
    if hasattr(tok, "apply_chat_template"):
        prompt_text = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    else:
        prompt_text = f"USER: {user}\nASSISTANT: "

    full_text = prompt_text + assistant
    if tok.eos_token and not full_text.endswith(tok.eos_token):
        full_text = full_text + tok.eos_token
    return {"prompt_text": prompt_text, "full_text": full_text}

@dataclass
class DataCollatorAssistantOnly:
    tokenizer: Any
    max_length: int

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        full_texts = [f["full_text"] for f in features]
        prompt_texts = [f["prompt_text"] for f in features]

        # IMPORTANT: keep rules consistent (special tokens) + truncation_side already set on tokenizer
        enc_full = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
            return_tensors="pt",
        )
        enc_prompt = self.tokenizer(
            prompt_texts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )

        labels = enc_full["input_ids"].clone()
        labels[enc_full["attention_mask"] == 0] = -100

        # mask prompt part
        for i in range(len(features)):
            p_len = len(enc_prompt["input_ids"][i])
            p_len = min(p_len, labels.size(1))
            labels[i, :p_len] = -100

        enc_full["labels"] = labels
        return enc_full

# ------------------
# Main
# ------------------
set_seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Loading tokenizer from: {BASE_MODEL}")
tok = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    use_fast=True,
    trust_remote_code=True,
    token=HF_TOKEN,
)

if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# CRITICAL: preserve tail tokens (assistant) when truncating
tok.truncation_side = "left"

print(f"Loading KD JSONL: {KD_JSONL}")
ds = load_dataset("json", data_files=KD_JSONL, split="train")

def _map(ex):
    return build_prompt_and_full(tok, ex["messages"])

ds = ds.map(_map, remove_columns=ds.column_names)

# Filter out examples with too few supervised assistant tokens after truncation considerations
def _keep(ex):
    p = tok(ex["prompt_text"], add_special_tokens=False)["input_ids"]
    f = tok(ex["full_text"], add_special_tokens=False)["input_ids"]
    sup = max(len(f) - len(p), 0)
    return sup >= MIN_SUP_TOKENS

ds = ds.filter(_keep)

if EVAL_RATIO > 0:
    split = ds.train_test_split(test_size=EVAL_RATIO, seed=SEED)
    train_ds, eval_ds = split["train"], split["test"]
else:
    train_ds, eval_ds = ds, None

print(f"train rows: {len(train_ds)} | eval rows: {len(eval_ds) if eval_ds is not None else 0}")

print(f"Loading pruned model (bf16) from: {BASE_MODEL}")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    token=HF_TOKEN,
)
model.config.use_cache = False
model.train()
model.enable_input_require_grads()

lora_cfg = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=LORA_TARGET_MODULES,
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# hard check: avoid "loss has no grad_fn"
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
if trainable == 0:
    raise RuntimeError("No trainable parameters found. LoRA target_modules mismatch model module names.")

args = TrainingArguments(
    output_dir=OUT_DIR,
    remove_unused_columns=False,
    seed=SEED,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=100,
    eval_strategy="steps" if eval_ds is not None else "no",
    eval_steps=100 if eval_ds is not None else None,
    bf16=True,
    fp16=False,
    gradient_checkpointing=True,
    report_to=[],
    save_total_limit=3,
)

collator = DataCollatorAssistantOnly(tokenizer=tok, max_length=MAX_SEQ_LEN)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collator,
)

trainer.train()

# Save adapter + tokenizer
trainer.model.save_pretrained(OUT_DIR)
tok.save_pretrained(OUT_DIR)

print("Done. Adapter saved to:", OUT_DIR)
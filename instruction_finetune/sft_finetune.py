# sft_finetune.py

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Any

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model


# =========================================================
# CONFIG
# =========================================================
BASE_MODEL = "/root/out/llama31_8b_instruct_sparsegpt_2of4/dense"
SFT_JSONL = "/root/data/instruct_data/sft_30000.jsonl"
OUT_DIR = "/root/out/pruned-llama31-8b-instruct-sft-lora"

HF_TOKEN = os.environ.get("HF_TOKEN", None)

SEED = 0
MAX_SEQ_LEN = 1024
EVAL_RATIO = 0.02

USE_QLORA = False
print(f"qlora: {USE_QLORA}")

LR = 2e-5
MAX_STEPS = 1500
BATCH = 1
GRAD_ACCUM = 32

LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# 至少保留一定数量的 assistant 监督 token
MIN_SUP_TOKENS = 32

# debug 可开
DEBUG_N = None   # 比如 100


# =========================================================
# Utilities
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_example(ex: Dict, tok) -> Dict[str, Any]:
    """
    基于 instruct/chat template 生成：
    - prompt_text
    - full_text
    - prompt_len_raw
    - full_len_raw
    - target_len_raw
    - sup_after_trunc
    """
    prompt_messages = ex.get("prompt_messages", None)
    full_messages = ex.get("messages", None)
    target_text = ex.get("target_text", "")

    if not isinstance(prompt_messages, list):
        prompt_messages = []
    if not isinstance(full_messages, list):
        full_messages = []

    prompt_text = tok.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    full_text = tok.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tok(full_text, add_special_tokens=False)["input_ids"]

    prompt_len_raw = len(prompt_ids)
    full_len_raw = len(full_ids)
    target_len_raw = max(full_len_raw - prompt_len_raw, 0)

    # 左截断下，assistant 尾部尽量保留
    sup_after_trunc = min(target_len_raw, MAX_SEQ_LEN)

    return {
        "prompt_text": prompt_text,
        "full_text": full_text,
        "target_text": target_text,
        "prompt_len_raw": prompt_len_raw,
        "full_len_raw": full_len_raw,
        "target_len_raw": target_len_raw,
        "sup_after_trunc": sup_after_trunc,
    }


@dataclass
class DataCollatorAssistantOnly:
    tokenizer: Any
    max_length: int

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        full_texts = [f["full_text"] for f in features]

        enc_full = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
            return_tensors="pt",
        )

        labels = enc_full["input_ids"].clone()
        labels[enc_full["attention_mask"] == 0] = -100

        # 关键：根据左截断后真正保留的 prompt 长度来 mask
        for i, f in enumerate(features):
            full_len_raw = int(f["full_len_raw"])
            prompt_len_raw = int(f["prompt_len_raw"])

            overflow = max(full_len_raw - self.max_length, 0)
            prompt_len_kept = max(prompt_len_raw - overflow, 0)

            seq_len_now = int(enc_full["attention_mask"][i].sum().item())
            prompt_len_kept = min(prompt_len_kept, seq_len_now)

            labels[i, :prompt_len_kept] = -100

        enc_full["labels"] = labels
        return enc_full


# =========================================================
# Main
# =========================================================
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

# 左截断：尽量保留 assistant 结尾
tok.truncation_side = "left"

print(f"Loading SFT JSONL: {SFT_JSONL}")
ds = load_dataset("json", data_files=SFT_JSONL, split="train")

if DEBUG_N is not None:
    ds = ds.select(range(min(DEBUG_N, len(ds))))

print("Mapping dataset...")
ds = ds.map(build_example, fn_kwargs={"tok": tok}, remove_columns=ds.column_names)

print("Filtering invalid / too-short supervised examples...")
def _keep(ex):
    if not ex["prompt_text"] or not ex["full_text"]:
        return False
    return int(ex["sup_after_trunc"]) >= MIN_SUP_TOKENS

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
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    token=HF_TOKEN,
)

if torch.cuda.is_available():
    model = model.cuda()

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
    bf16=torch.cuda.is_available(),
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
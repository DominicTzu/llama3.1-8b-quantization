import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PRUNED_DIR = "/root/out/qwen3-14b-2of4-sparsegpt"
DENSE_OUT  = "/root/out/qwen3-14b-2of4-dense-bf16"

os.makedirs(DENSE_OUT, exist_ok=True)

tok = AutoTokenizer.from_pretrained(PRUNED_DIR, use_fast=True, trust_remote_code=True, fix_mistral_regex=True)

model = AutoModelForCausalLM.from_pretrained(
    PRUNED_DIR,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
# 关键：不要 save_compressed，保存成标准 dense 权重
model.save_pretrained(DENSE_OUT, safe_serialization=True)
tok.save_pretrained(DENSE_OUT)

print("Exported dense model to:", DENSE_OUT)
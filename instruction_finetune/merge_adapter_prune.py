# merge_adapter_prune.py
import os
import json
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# =========================
# CONFIG 
# =========================
# Base model should be PRUNED model directory (preferred)
BASE_MODEL_DIR = "/root/out/llama31_8b_base_sparsegpt_2of4_dense/dense"

# Adapter dir produced by finetune script (contains adapter_model.safetensors + adapter_config.json)
ADAPTER_DIR = "/root/out/pruned-llama31-8b-sft-lora"

# Output dir for merged full model
MERGED_OUT_DIR = "/root/out/llama-31-8b-pruned-sft-merged-bf16"

# Save dtype for merged model weights
SAVE_DTYPE = torch.bfloat16   # change to torch.float16 if bf16 unsupported

# =========================
# Main
# =========================
def main():
    os.makedirs(MERGED_OUT_DIR, exist_ok=True)

    print(f"[1/5] Loading tokenizer from base: {BASE_MODEL_DIR}")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[2/5] Loading base model in high precision (NOT 4bit): {BASE_MODEL_DIR}")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=SAVE_DTYPE,
    )
    base.eval()
    base.config.use_cache = True

    print(f"[3/5] Loading adapter: {ADAPTER_DIR}")
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)

    print("[4/5] Merging LoRA into base weights...")
    merged = model.merge_and_unload()  # returns a plain transformers model

    print(f"[5/5] Saving merged full model to: {MERGED_OUT_DIR}")
    merged.save_pretrained(MERGED_OUT_DIR, safe_serialization=True)
    tok.save_pretrained(MERGED_OUT_DIR)

    # Reproducibility snapshot
    meta = {
        "base_model_dir": BASE_MODEL_DIR,
        "adapter_dir": ADAPTER_DIR,
        "merged_out_dir": MERGED_OUT_DIR,
        "save_dtype": str(SAVE_DTYPE),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(MERGED_OUT_DIR, "merge_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Merged model saved at: {MERGED_OUT_DIR}")


if __name__ == "__main__":
    main()
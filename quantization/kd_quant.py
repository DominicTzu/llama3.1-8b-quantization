# quantize_w4a16_awq.py
import os
import json
import random
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier


# =========================
# CONFIG (edit here)
# =========================
# Quantize THIS model (merged full model recommended)
MODEL_DIR = "/root/out/qwen3-14b-pruned-kd-merged-bf16"

# Calibration data (your jsonl, messages format)
CALIB_JSONL = "/root/data/train_data/calib_1000.jsonl"

# Output
OUT_DIR = "/root/out/qwen3-14b-pruned-kd-merged-AWQ-W4A16-G128"

SEED = 0

# Calibration / tokenization
NUM_CALIB_SAMPLES = 512         # start with 256 if slow / OOM, then 512
MAX_SEQ_LEN = 2048              # keep aligned with your eval MAX_MODEL_LEN if possible

# AWQ config
SCHEME = "W4A16_ASYM"           # as in LLM Compressor AWQ example
GROUP_SIZE = 128                # set by scheme group config; AWQ docs recommend group size 128 typical
IGNORE = ["lm_head"]            # standard
TARGETS = ["Linear"]            # standard


# =========================
# Helpers
# =========================
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def messages_to_text(tok: AutoTokenizer, messages: List[Dict]) -> str:
    if hasattr(tok, "apply_chat_template"):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    # fallback
    parts = []
    for m in messages:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if content:
            parts.append(f"{role.upper()}: {content}")
    return "\n".join(parts)

def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.isfile(CALIB_JSONL):
        raise FileNotFoundError(f"Calibration JSONL not found: {CALIB_JSONL}")

    print(f"[1/6] Loading tokenizer from: {MODEL_DIR}")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True, trust_remote_code=True)

    print(f"[2/6] Loading calibration set: {CALIB_JSONL}")
    ds = load_dataset("json", data_files=CALIB_JSONL, split="train")
    ds = ds.shuffle(seed=SEED)

    # Map messages -> text
    def _preprocess(ex):
        return {"text": messages_to_text(tok, ex["messages"])}

    ds = ds.map(_preprocess, remove_columns=ds.column_names, desc="messages_to_text")

    # Tokenize (LLM Compressor examples tokenize into input_ids/attention_mask)
    def _tokenize(ex):
        return tok(
            ex["text"],
            padding=False,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            add_special_tokens=False,
        )

    ds = ds.map(_tokenize, remove_columns=ds.column_names, desc="tokenize")

    n = min(NUM_CALIB_SAMPLES, len(ds))
    ds = ds.select(list(range(n)))
    print(f"  calib used: {n} samples, max_seq_len={MAX_SEQ_LEN}")

    print(f"[3/6] Loading model (bf16/fp16) from: {MODEL_DIR}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print("[4/6] Building AWQ recipe")
    # AWQModifier performs activation-aware scaling + weight quantization in oneshot.
    # Example from docs: AWQModifier(ignore=["lm_head"], scheme="W4A16_ASYM", targets=["Linear"])
    recipe = AWQModifier(
        ignore=IGNORE,
        scheme=SCHEME,
        targets=TARGETS,
    )

    print("[5/6] Running oneshot quantization (AWQ)...")
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQ_LEN,
        num_calibration_samples=n,
    )

    print("[6/6] Saving quantized model (vLLM-compatible compressed tensors)")
    model.save_pretrained(OUT_DIR, save_compressed=True)
    tok.save_pretrained(OUT_DIR)

    meta = {
        "model_dir": MODEL_DIR,
        "calib_jsonl": CALIB_JSONL,
        "num_calib_samples": n,
        "max_seq_len": MAX_SEQ_LEN,
        "awq": {"scheme": SCHEME, "targets": TARGETS, "ignore": IGNORE},
        "seed": SEED,
    }
    with open(os.path.join(OUT_DIR, "quant_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Quantized model saved at: {OUT_DIR}")

if __name__ == "__main__":
    main()
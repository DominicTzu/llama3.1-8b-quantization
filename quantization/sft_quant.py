# sft_quant.py

import os
import json
import random
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor.entrypoints.oneshot import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.awq.mappings import AWQMapping


# =========================================================
# CONFIG
# =========================================================
# Quantize THIS merged dense model
MODEL_DIR = "/root/out/llama-31-8b-pruned-sft-merged-bf16"

# Use base tokenizer to stay aligned with your current pipeline
MODEL_BASE = "meta-llama/Llama-3.1-8B"

# Calibration data
CALIB_JSONL = "/root/data/sft_data/calib_1000.jsonl"

# Output
OUT_DIR = "/root/out/llama-31-8b-pruned-sft-merged-awq-w4a16-asym"

SEED = 0

# Calibration
NUM_CALIB_SAMPLES = 512
MAX_SEQ_LEN = 2048

# AWQ config
SCHEME = "W4A16_ASYM"
IGNORE = ["lm_head"]
TARGETS = ["Linear"]

# Llama family AWQ mappings from docs
LLAMA_AWQ_MAPPINGS = [
    AWQMapping(
        "re:.*input_layernorm",
        ["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"],
    ),
    AWQMapping(
        "re:.*v_proj",
        ["re:.*o_proj"],
    ),
    AWQMapping(
        "re:.*post_attention_layernorm",
        ["re:.*gate_proj", "re:.*up_proj"],
    ),
    AWQMapping(
        "re:.*up_proj",
        ["re:.*down_proj"],
    ),
]


# =========================================================
# Helpers
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def normalize_text(x) -> str:
    if x is None:
        return ""
    return str(x).strip()

def build_plain_prompt_from_messages(messages: List[Dict]) -> str:
    """
    Fallback only: if prompt_text does not exist, rebuild a plain prompt.
    No chat template.
    """
    if not isinstance(messages, list):
        return ""

    user_text = ""
    for m in messages:
        if (
            isinstance(m, dict)
            and m.get("role") == "user"
            and isinstance(m.get("content"), str)
            and m["content"].strip()
        ):
            user_text = m["content"].strip()
            break

    if not user_text:
        return ""

    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{user_text}\n\n"
        "### Response:\n"
    )

def record_to_text(ex: Dict) -> Dict:
    prompt_text = ex.get("prompt_text", None)
    if isinstance(prompt_text, str) and prompt_text.strip():
        text = prompt_text.strip()
    else:
        text = build_plain_prompt_from_messages(ex.get("messages", []))
    return {"text": text}

def summarize_token_lengths(ds, key: str = "input_ids"):
    lengths = [len(x[key]) for x in ds if key in x]
    if not lengths:
        print("  No tokenized samples found.")
        return

    lengths = sorted(lengths)

    def pct(arr, p):
        idx = min(len(arr) - 1, int(len(arr) * p))
        return arr[idx]

    print("  token length summary:")
    print(
        f"    p50={pct(lengths, 0.50)} "
        f"p90={pct(lengths, 0.90)} "
        f"p95={pct(lengths, 0.95)} "
        f"max={lengths[-1]}"
    )


# =========================================================
# Main
# =========================================================
def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.isfile(CALIB_JSONL):
        raise FileNotFoundError(f"Calibration JSONL not found: {CALIB_JSONL}")

    print(f"[1/6] Loading tokenizer from: {MODEL_BASE}")
    tok = AutoTokenizer.from_pretrained(
        MODEL_BASE,
        use_fast=True,
        trust_remote_code=True,
    )

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    tok.truncation_side = "right"

    print(f"[2/6] Loading calibration set: {CALIB_JSONL}")
    ds = load_dataset("json", data_files=CALIB_JSONL, split="train")
    ds = ds.shuffle(seed=SEED)

    print("[3/6] Converting records to plain prompt text")
    ds = ds.map(
        record_to_text,
        remove_columns=ds.column_names,
        desc="record_to_text",
    )

    ds = ds.filter(lambda ex: isinstance(ex["text"], str) and len(ex["text"].strip()) > 0)

    def _tokenize(ex):
        return tok(
            ex["text"],
            padding=False,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            add_special_tokens=False,
        )

    ds = ds.map(
        _tokenize,
        remove_columns=ds.column_names,
        desc="tokenize",
    )

    n = min(NUM_CALIB_SAMPLES, len(ds))
    ds = ds.select(list(range(n)))
    print(f"  calib used: {n} samples | max_seq_len={MAX_SEQ_LEN}")
    summarize_token_lengths(ds, key="input_ids")

    print(f"[4/6] Loading model from: {MODEL_DIR}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print("[5/6] Building AWQ recipe")
    recipe = [
        AWQModifier(
            ignore=IGNORE,
            scheme=SCHEME,
            targets=TARGETS,
            mappings=LLAMA_AWQ_MAPPINGS,
        )
    ]

    print("[5.5/6] Running oneshot quantization (AWQ)...")
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
        "model_base": MODEL_BASE,
        "calib_jsonl": CALIB_JSONL,
        "num_calib_samples": n,
        "max_seq_len": MAX_SEQ_LEN,
        "seed": SEED,
        "awq": {
            "scheme": SCHEME,
            "targets": TARGETS,
            "ignore": IGNORE,
            "mappings": "llama_family_default_from_docs",
        },
    }
    with open(os.path.join(OUT_DIR, "quant_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Quantized model saved at: {OUT_DIR}")


if __name__ == "__main__":
    main()

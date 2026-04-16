# kd_quant.py

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
# Quantize THIS merged dense KD model
MODEL_DIR = "/root/out/llama-31-8b-instruct-pruned-kd-merged-bf16"

# Instruct tokenizer
MODEL_BASE = "meta-llama/Llama-3.1-8B-Instruct"

# Calibration data: 仍然用统一的 instruct calib
CALIB_JSONL = "/root/data/instruct_data/calib_1000.jsonl"

# Output
OUT_DIR = "/root/out/llama-31-8b-instruct-pruned-kd-merged-awq-w4a16-asym"

SEED = 0

# Calibration
NUM_CALIB_SAMPLES = 512
MAX_SEQ_LEN = 2048

# AWQ config
SCHEME = "W4A16_ASYM"
IGNORE = ["lm_head"]
TARGETS = ["Linear"]

TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Llama family AWQ mappings
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


def messages_to_text(tok: AutoTokenizer, messages: List[Dict]) -> str:
    """
    Instruct version: use chat template directly.
    """
    if not isinstance(messages, list) or len(messages) == 0:
        return ""
    return tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def record_to_text(ex: Dict, tok: AutoTokenizer) -> Dict:
    text = messages_to_text(tok, ex.get("messages", []))
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

    print("[3/6] Converting records to chat-template text")
    ds = ds.map(
        record_to_text,
        fn_kwargs={"tok": tok},
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
        torch_dtype=TORCH_DTYPE,
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
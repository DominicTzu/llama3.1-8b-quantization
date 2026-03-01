# prune_llama_31_8b_2of4.py

import os
import json
import random
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from llmcompressor.entrypoints.oneshot import oneshot
from llmcompressor.modifiers.pruning.sparsegpt import SparseGPTModifier


# =========================
# CONFIG
# =========================
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
CALIB_JSONL = "/root/data/train_data/calib_1000.jsonl"
OUT_DIR = "/root/out/llama-31-8b-dense2of4"

SEED = 0
NUM_CALIB_SAMPLES = 512
MAX_SEQ_LEN = 1024

SPARSITY = 0.5
MASK_STRUCTURE = "2:4"
DAMPENING_FRAC = 0.001
BLOCK_SIZE = 128

TARGETS = ["Linear"]
IGNORE = ["re:.*lm_head"]

HF_TOKEN = os.environ.get("HF_TOKEN", None)

SAVE_DENSE = True
SAVE_COMPRESSED = False

# (optional) quick validation of sparsity after pruning
VERIFY_SPARSE = True
VERIFY_LAYERS = 8   # randomly check a few linear layers


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
    parts = []
    for m in messages:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if content:
            parts.append(f"{role.upper()}: {content}")
    return "\n".join(parts)

def verify_nm_sparsity_2of4(weight: torch.Tensor) -> float:
    """
    Return fraction of 2:4 groups that satisfy exactly 2 non-zeros per 4.
    Assumes last dimension is divisible by 4 (typical for Linear weights).
    """
    if weight.ndim != 2:
        return 0.0
    w = weight.detach()
    # count nonzeros
    nz = (w != 0).to(torch.int32)
    if nz.shape[1] < 4:
        return 0.0
    m = (nz.shape[1] // 4) * 4
    nz = nz[:, :m].reshape(nz.shape[0], -1, 4).sum(dim=-1)  # (out, groups)
    ok = (nz == 2).float().mean().item()
    return ok

def main():
    set_seed(SEED)

    if not os.path.isfile(CALIB_JSONL):
        raise FileNotFoundError(f"Calibration file not found: {CALIB_JSONL}")
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[1/6] Loading tokenizer: {MODEL_ID}")
    # IMPORTANT: some transformers versions prefer use_auth_token; token works in newer versions
    try:
        tok = AutoTokenizer.from_pretrained(
            MODEL_ID,
            use_fast=True,
            trust_remote_code=True,
            token=HF_TOKEN,
        )
    except TypeError:
        tok = AutoTokenizer.from_pretrained(
            MODEL_ID,
            use_fast=True,
            trust_remote_code=True,
            use_auth_token=True if HF_TOKEN else None,
        )

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ensure truncation behavior is deterministic
    tok.truncation_side = "right"

    print(f"[2/6] Loading calibration dataset: {CALIB_JSONL}")
    ds = load_dataset("json", data_files=CALIB_JSONL, split="train").shuffle(seed=SEED)

    def _to_text(ex):
        return {"text": messages_to_text(tok, ex.get("messages", []))}
    ds = ds.map(_to_text, remove_columns=ds.column_names, desc="messages_to_text")

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

    print(f"[3/6] Loading model (bf16): {MODEL_ID}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
            token=HF_TOKEN,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
            use_auth_token=True if HF_TOKEN else None,
        )

    model.eval()

    print("[4/6] Building SparseGPT recipe (N:M semi-structured pruning)")
    recipe = [
        SparseGPTModifier(
            sparsity=SPARSITY,
            mask_structure=MASK_STRUCTURE,
            dampening_frac=DAMPENING_FRAC,
            block_size=BLOCK_SIZE,
            targets=TARGETS,
            ignore=IGNORE,
        )
    ]

    print("[5/6] Running oneshot pruning...")
    oneshot(
        model=model,
        dataset=ds,          # tokenized dataset (input_ids/attention_mask)
        recipe=recipe,
        output_dir=None,
        max_seq_length=MAX_SEQ_LEN,
        num_calibration_samples=n,
    )

    if VERIFY_SPARSE:
        print("[5.5/6] Verifying 2:4 sparsity on a few Linear layers...")
        import torch.nn as nn
        linear_weights = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and hasattr(module, "weight"):
                if "lm_head" in name:
                    continue
                linear_weights.append((name, module.weight))
        random.shuffle(linear_weights)
        checked = 0
        oks = []
        for name, w in linear_weights:
            ok = verify_nm_sparsity_2of4(w)
            if ok > 0:
                oks.append(ok)
                checked += 1
            if checked >= VERIFY_LAYERS:
                break
        if oks:
            print(f"  checked={checked} layers | avg 2:4 match={sum(oks)/len(oks):.3f}")
        else:
            print("  (warning) no suitable Linear layers found for verification.")

    print("[6/6] Saving pruned model...")
    if SAVE_DENSE:
        dense_dir = os.path.join(OUT_DIR, "dense")
        os.makedirs(dense_dir, exist_ok=True)
        model.save_pretrained(dense_dir, safe_serialization=True)

        # SAVE TOKENIZER BOTH places: experiment root + dense checkpoint
        tok.save_pretrained(OUT_DIR)
        tok.save_pretrained(dense_dir)

        print("  saved dense ->", dense_dir)

    if SAVE_COMPRESSED:
        comp_dir = os.path.join(OUT_DIR, "compressed")
        os.makedirs(comp_dir, exist_ok=True)
        model.save_pretrained(comp_dir, save_compressed=True)
        tok.save_pretrained(comp_dir)
        print("  saved compressed ->", comp_dir)

    cfg = {
        "model_id": MODEL_ID,
        "calib_jsonl": CALIB_JSONL,
        "num_calib_samples": n,
        "max_seq_len": MAX_SEQ_LEN,
        "sparsegpt": {
            "sparsity": SPARSITY,
            "mask_structure": MASK_STRUCTURE,
            "dampening_frac": DAMPENING_FRAC,
            "block_size": BLOCK_SIZE,
            "targets": TARGETS,
            "ignore": IGNORE,
        },
        "seed": SEED,
        "save_dense": SAVE_DENSE,
        "save_compressed": SAVE_COMPRESSED,
    }
    with open(os.path.join(OUT_DIR, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print("Done.")
    print("Pruned outputs:")
    if SAVE_DENSE:
        print(" -", os.path.join(OUT_DIR, "dense"))
    if SAVE_COMPRESSED:
        print(" -", os.path.join(OUT_DIR, "compressed"))

if __name__ == "__main__":
    main()

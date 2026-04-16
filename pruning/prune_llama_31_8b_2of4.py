# prune_llama_31_8b_2of4.py

import os
import json
import random
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from llmcompressor.entrypoints.oneshot import oneshot
from llmcompressor.modifiers.pruning.sparsegpt import SparseGPTModifier


# =========================================================
# CONFIG
# =========================================================
MODEL_ID = "meta-llama/Llama-3.1-8B"

# 这里建议换成你现在新的 calib 文件
CALIB_JSONL = "/root/data/sft/calib_1000.jsonl"

OUT_DIR = "/root/out/llama31_8b_base_sparsegpt_2of4_dense"

HF_TOKEN = os.environ.get("HF_TOKEN", None)

SEED = 0

# 实际用于剪枝校准的样本数
NUM_CALIB_SAMPLES = 512

# 校准时截断长度
MAX_SEQ_LEN = 1024

# 2:4 SparseGPT
SPARSITY = 0.5
MASK_STRUCTURE = "2:4"
DAMPENING_FRAC = 0.001
BLOCK_SIZE = 128

# 只剪 decoder block 内核心线性层
TARGETS = [
    "re:.*q_proj$",
    "re:.*k_proj$",
    "re:.*v_proj$",
    "re:.*o_proj$",
    "re:.*gate_proj$",
    "re:.*up_proj$",
    "re:.*down_proj$",
]

# 不剪 lm_head
IGNORE = [
    "re:.*lm_head$",
]

SAVE_DENSE = True
SAVE_COMPRESSED = False

VERIFY_SPARSE = True
VERIFY_LAYERS = 8

TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32


# =========================================================
# Helpers
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def normalize_text(x) -> str:
    if x is None:
        return ""
    return str(x).strip()


def build_plain_prompt_from_messages(messages: List[Dict]) -> str:
    """
    回退用：如果 calib 样本没有 prompt_text，就从 messages 构造一个 plain prompt。
    不用 chat template。
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
    """
    优先使用 prompt_text。
    这是你当前 base + alpaca 风格数据最应该走的分布。
    """
    prompt_text = ex.get("prompt_text", None)
    if isinstance(prompt_text, str) and prompt_text.strip():
        text = prompt_text.strip()
    else:
        text = build_plain_prompt_from_messages(ex.get("messages", []))

    return {"text": text}


def verify_nm_sparsity_2of4(weight: torch.Tensor) -> float:
    """
    返回满足“每4个元素中恰有2个非零”的 group 比例。
    假设 weight 是二维矩阵，且最后一维可按4分组。
    """
    if weight.ndim != 2:
        return 0.0

    w = weight.detach()
    nz = (w != 0).to(torch.int32)

    if nz.shape[1] < 4:
        return 0.0

    m = (nz.shape[1] // 4) * 4
    if m == 0:
        return 0.0

    nz = nz[:, :m].reshape(nz.shape[0], -1, 4).sum(dim=-1)
    ok = (nz == 2).float().mean().item()
    return ok


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

    if not os.path.isfile(CALIB_JSONL):
        raise FileNotFoundError(f"Calibration file not found: {CALIB_JSONL}")

    ensure_dir(OUT_DIR)

    print(f"[1/7] Loading tokenizer: {MODEL_ID}")
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

    tok.truncation_side = "right"

    print(f"[2/7] Loading calibration dataset: {CALIB_JSONL}")
    ds = load_dataset("json", data_files=CALIB_JSONL, split="train")
    ds = ds.shuffle(seed=SEED)

    print("[3/7] Converting calibration records to plain prompt text")
    ds = ds.map(
        record_to_text,
        remove_columns=ds.column_names,
        desc="record_to_text",
    )

    # 去掉空样本
    ds = ds.filter(lambda ex: isinstance(ex["text"], str) and len(ex["text"].strip()) > 0)

    def _tokenize(ex):
        return tok(
            ex["text"],
            padding=False,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            add_special_tokens=False,
        )

    print("[4/7] Tokenizing calibration text")
    ds = ds.map(
        _tokenize,
        remove_columns=ds.column_names,
        desc="tokenize",
    )

    n = min(NUM_CALIB_SAMPLES, len(ds))
    ds = ds.select(list(range(n)))

    print(f"  calib used: {n} samples | max_seq_len={MAX_SEQ_LEN}")
    summarize_token_lengths(ds, key="input_ids")

    print(f"[5/7] Loading model on single GPU: {MODEL_ID}")
    model_kwargs = dict(
        torch_dtype=TORCH_DTYPE,
        trust_remote_code=True,
    )

    if torch.cuda.is_available():
        model_kwargs["device_map"] = {"": 0}
    else:
        model_kwargs["device_map"] = None

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            token=HF_TOKEN,
            **model_kwargs,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            use_auth_token=True if HF_TOKEN else None,
            **model_kwargs,
        )

    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    print("[6/7] Building SparseGPT recipe")
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

    print("[6.1/7] Running oneshot pruning...")
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        output_dir=None,
        max_seq_length=MAX_SEQ_LEN,
        num_calibration_samples=n,
    )

    if VERIFY_SPARSE:
        print("[6.2/7] Verifying 2:4 sparsity on sampled target layers...")
        candidate_weights = []

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and hasattr(module, "weight"):
                if name.endswith("lm_head"):
                    continue
                if any(
                    name.endswith(suffix)
                    for suffix in [
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    ]
                ):
                    candidate_weights.append((name, module.weight))

        random.shuffle(candidate_weights)

        oks = []
        checked = 0
        for name, w in candidate_weights:
            ratio = verify_nm_sparsity_2of4(w)
            if ratio > 0:
                oks.append((name, ratio))
                checked += 1
            if checked >= VERIFY_LAYERS:
                break

        if oks:
            avg_ratio = sum(x[1] for x in oks) / len(oks)
            print(f"  checked={len(oks)} | avg 2:4 match={avg_ratio:.4f}")
            for name, ratio in oks[: min(5, len(oks))]:
                print(f"   - {name}: {ratio:.4f}")
        else:
            print("  warning: no suitable pruned layers found for verification.")

    print("[7/7] Saving outputs...")
    if SAVE_DENSE:
        dense_dir = os.path.join(OUT_DIR, "dense")
        ensure_dir(dense_dir)

        # 普通 dense 格式保存：张量里包含很多 0，但文件格式仍是普通 HF checkpoint
        model.save_pretrained(dense_dir, safe_serialization=True)
        tok.save_pretrained(dense_dir)
        tok.save_pretrained(OUT_DIR)

        print("  saved dense ->", dense_dir)

    if SAVE_COMPRESSED:
        comp_dir = os.path.join(OUT_DIR, "compressed")
        ensure_dir(comp_dir)

        model.save_pretrained(comp_dir, save_compressed=True)
        tok.save_pretrained(comp_dir)
        print("  saved compressed ->", comp_dir)

    cfg = {
        "model_id": MODEL_ID,
        "calib_jsonl": CALIB_JSONL,
        "num_calib_samples": n,
        "max_seq_len": MAX_SEQ_LEN,
        "seed": SEED,
        "save_dense": SAVE_DENSE,
        "save_compressed": SAVE_COMPRESSED,
        "sparsegpt": {
            "sparsity": SPARSITY,
            "mask_structure": MASK_STRUCTURE,
            "dampening_frac": DAMPENING_FRAC,
            "block_size": BLOCK_SIZE,
            "targets": TARGETS,
            "ignore": IGNORE,
        },
    }

    with open(os.path.join(OUT_DIR, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print("Pruned outputs:")
    if SAVE_DENSE:
        print(" -", os.path.join(OUT_DIR, "dense"))
    if SAVE_COMPRESSED:
        print(" -", os.path.join(OUT_DIR, "compressed"))


if __name__ == "__main__":
    main()

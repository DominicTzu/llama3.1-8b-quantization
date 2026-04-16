# make_kd_dataset.py

import os
import json
import random
from typing import Dict, List, Optional

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# =========================================================
# CONFIG
# =========================================================
TEACHER_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

IN_SFT_JSONL = "/root/data/instruct_data/sft_30000.jsonl"
OUT_DIR = "/root/data/instruct_data"
OUT_JSONL = "/root/data/instruct_data/kd_15000.jsonl"

HF_TOKEN = os.environ.get("HF_TOKEN", None)

SEED = 0
N_KD = 15_000

MAX_PROMPT_TOKENS = 1024
MAX_NEW_TOKENS = 192

TEMPERATURE = 0.0
TOP_P = 1.0

DTYPE = "bfloat16"
GPU_MEMORY_UTIL = 0.90
MAX_MODEL_LEN = 2048

# 过滤阈值
MIN_TEACHER_TOKENS = 8
MAX_TEACHER_TOKENS = 256

# debug
DEBUG_N = None   # 比如 100


# =========================================================
# Utils
# =========================================================
def set_seed(seed: int):
    random.seed(seed)

def write_jsonl(path: str, records: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def normalize_text(x) -> str:
    if x is None:
        return ""
    return str(x).strip()

def get_prompt_messages_from_record(rec: Dict) -> Optional[List[Dict]]:
    pm = rec.get("prompt_messages", None)
    if isinstance(pm, list) and len(pm) > 0:
        return pm
    return None

def get_gold_target_from_record(rec: Dict) -> str:
    t = rec.get("target_text", None)
    if isinstance(t, str) and t.strip():
        return t.strip()
    return ""

def apply_prompt_template(tok: AutoTokenizer, prompt_messages: List[Dict]) -> str:
    return tok.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

def clean_teacher_text(text: str) -> str:
    text = text.strip()

    # 截掉模板回卷
    bad_markers = [
        "<|start_header_id|>",
        "<|eot_id|>",
        "assistant\n",
        "user\n",
        "system\n",
        "### Instruction:",
        "### Response:",
    ]
    for pat in bad_markers:
        if pat in text:
            text = text.split(pat)[0].strip()

    # 去掉明显多余空行
    lines = [x.strip() for x in text.splitlines()]
    lines = [x for x in lines if x]
    text = "\n".join(lines).strip()

    return text

def is_repetitive(text: str) -> bool:
    words = text.split()
    if len(words) < 8:
        return False
    uniq_ratio = len(set(words)) / max(len(words), 1)
    return uniq_ratio < 0.35

def is_bad_teacher_text(text: str, tok) -> bool:
    if not text:
        return True
    if "###" in text:
        return True
    if is_repetitive(text):
        return True

    toks = tok.encode(text, add_special_tokens=False)
    if len(toks) < MIN_TEACHER_TOKENS:
        return True
    if len(toks) > MAX_TEACHER_TOKENS:
        return True

    return False


# =========================================================
# Main
# =========================================================
def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[1/4] Loading tokenizer: {TEACHER_MODEL}")
    tok = AutoTokenizer.from_pretrained(
        TEACHER_MODEL,
        use_fast=True,
        trust_remote_code=True,
        token=HF_TOKEN,
    )
    tok.truncation_side = "left"

    print(f"[2/4] Loading source SFT JSONL: {IN_SFT_JSONL}")
    ds = load_dataset("json", data_files=IN_SFT_JSONL, split="train")

    if DEBUG_N is not None:
        ds = ds.select(range(min(DEBUG_N, len(ds))))

    idxs = list(range(len(ds)))
    random.shuffle(idxs)

    prompts_text: List[str] = []
    source_rows: List[Dict] = []
    seen = set()

    for i in idxs:
        rec = ds[i]

        prompt_messages = get_prompt_messages_from_record(rec)
        if prompt_messages is None:
            continue

        prompt_text = apply_prompt_template(tok, prompt_messages)
        prompt_tok_len = len(tok.encode(prompt_text, add_special_tokens=False))
        if prompt_tok_len > MAX_PROMPT_TOKENS:
            continue

        gold_target = get_gold_target_from_record(rec)
        if not gold_target:
            continue

        # 去重：最后一个 user + gold_target
        last_user = ""
        for m in reversed(prompt_messages):
            if m.get("role") == "user":
                last_user = normalize_text(m.get("content"))
                break

        dup_key = (last_user[:500] + "|||" + gold_target[:300]).strip()
        if dup_key in seen:
            continue
        seen.add(dup_key)

        prompts_text.append(prompt_text)
        source_rows.append(
            {
                "id": rec.get("id", f"row_{i}"),
                "messages": rec.get("messages", []),
                "prompt_messages": prompt_messages,
                "gold_target": gold_target,
                "meta": {
                    "source_index": i,
                    "prompt_tokens": prompt_tok_len,
                },
            }
        )

        if len(prompts_text) >= N_KD:
            break

    print(f"Collected prompts: {len(prompts_text)}")
    if len(prompts_text) == 0:
        raise RuntimeError("No valid KD prompts collected.")

    print(f"[3/4] Initializing vLLM teacher: {TEACHER_MODEL}")
    llm = LLM(
        model=TEACHER_MODEL,
        dtype=DTYPE,
        gpu_memory_utilization=GPU_MEMORY_UTIL,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
    )

    sp = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_NEW_TOKENS,
        stop=[
            "<|eot_id|>",
            "<|start_header_id|>",
            "### Instruction:",
            "### Response:",
        ],
        stop_token_ids=[tok.eos_token_id] if tok.eos_token_id is not None else None,
    )

    print("[4/4] Generating teacher responses...")
    outputs = llm.generate(prompts_text, sp)

    kd_records: List[Dict] = []
    bad_cnt = 0

    for src, out in zip(source_rows, outputs):
        if not out.outputs:
            bad_cnt += 1
            continue

        teacher_target = clean_teacher_text(out.outputs[0].text)

        if is_bad_teacher_text(teacher_target, tok):
            bad_cnt += 1
            continue

        teacher_target_tokens = len(tok.encode(teacher_target, add_special_tokens=False))

        kd_records.append(
            {
                "id": src["id"],
                "prompt_messages": src["prompt_messages"],
                "messages": src["prompt_messages"] + [
                    {"role": "assistant", "content": teacher_target}
                ],
                "gold_target": src["gold_target"],
                "teacher_target": teacher_target,
                "meta": {
                    "teacher_model": TEACHER_MODEL,
                    "prompt_tokens": src["meta"]["prompt_tokens"],
                    "teacher_target_tokens": teacher_target_tokens,
                },
            }
        )

    write_jsonl(OUT_JSONL, kd_records)
    print(f"Done. Wrote KD dataset: {OUT_JSONL}")
    print(f"KD rows: {len(kd_records)}")
    print(f"Bad / filtered generations: {bad_cnt}")


if __name__ == "__main__":
    main()
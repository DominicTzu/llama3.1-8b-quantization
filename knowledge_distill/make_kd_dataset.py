# make_kd_dataset.py
import os
import json
import random
from typing import Dict, List, Optional

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

TEACHER_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# 这里直接基于你之前造好的 SFT 抽
IN_SFT_JSONL = "/root/data/alpaca_30k_sft_data"

OUT_DIR = "/root/data/kd_data"
OUT_JSONL = "/root/data/kd_data/kd_15000.jsonl"

HF_TOKEN = os.environ.get("HF_TOKEN", None)

SEED = 0
N_KD = 15_000

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.0
TOP_P = 1.0

MAX_PROMPT_TOKENS = 1024

DTYPE = "bfloat16"
GPU_MEMORY_UTIL = 0.90
MAX_MODEL_LEN = 2048


def set_seed(seed: int):
    random.seed(seed)


def normalize_text(x) -> str:
    if x is None:
        return ""
    return str(x).strip()


def build_plain_prompt_from_messages(messages: List[Dict]) -> Optional[str]:
    """
    不用 chat template。
    如果旧 SFT 只有 messages，没有 prompt_text，就回退到这里重建 prompt。
    默认按单轮 user->assistant 的 Alpaca 风格 prompt 构造。
    """
    if not isinstance(messages, list) or len(messages) == 0:
        return None

    user_text = None
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
        return None

    prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{user_text}\n\n"
        "### Response:\n"
    )
    return prompt


def get_prompt_text_from_record(rec: Dict) -> Optional[str]:
    """
    优先用已有的 prompt_text；
    没有的话再从 messages 回退重建。
    """
    prompt_text = rec.get("prompt_text", None)
    if isinstance(prompt_text, str) and prompt_text.strip():
        return prompt_text.strip()

    messages = rec.get("messages", None)
    return build_plain_prompt_from_messages(messages)


def get_gold_target_from_record(rec: Dict) -> str:
    """
    尽量把原始 gold target 也保留下来，后面做对比方便。
    """
    target_text = rec.get("target_text", None)
    if isinstance(target_text, str) and target_text.strip():
        return target_text.strip()

    messages = rec.get("messages", None)
    if isinstance(messages, list):
        for m in messages:
            if (
                isinstance(m, dict)
                and m.get("role") == "assistant"
                and isinstance(m.get("content"), str)
                and m["content"].strip()
            ):
                return m["content"].strip()
    return ""


def get_user_content_from_record(rec: Dict) -> str:
    """
    用于兼容你后面如果还想保留 messages 字段。
    """
    messages = rec.get("messages", None)
    if isinstance(messages, list):
        for m in messages:
            if (
                isinstance(m, dict)
                and m.get("role") == "user"
                and isinstance(m.get("content"), str)
                and m["content"].strip()
            ):
                return m["content"].strip()

    prompt_text = rec.get("prompt_text", "")
    return prompt_text.strip()


def write_jsonl(path: str, records: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


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
    idxs = list(range(len(ds)))
    random.shuffle(idxs)

    prompts_text: List[str] = []
    source_rows: List[Dict] = []

    seen = set()

    for i in idxs:
        rec = ds[i]

        prompt_text = get_prompt_text_from_record(rec)
        if not prompt_text:
            continue

        if not prompt_text.endswith((" ", "\n")):
            prompt_text += " "

        prompt_tok_len = len(tok.encode(prompt_text, add_special_tokens=False))
        if prompt_tok_len > MAX_PROMPT_TOKENS:
            continue

        gold_target = get_gold_target_from_record(rec)
        user_content = get_user_content_from_record(rec)

        dup_key = (prompt_text[:500] + "|||" + gold_target[:300]).strip()
        if dup_key in seen:
            continue
        seen.add(dup_key)

        prompts_text.append(prompt_text)
        source_rows.append(
            {
                "id": rec.get("id", f"row_{i}"),
                "prompt_text": prompt_text,
                "gold_target": gold_target,
                "user_content": user_content,
                "messages": rec.get("messages", []),
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
        stop_token_ids=[tok.eos_token_id] if tok.eos_token_id is not None else None,
    )

    print("[4/4] Generating teacher responses...")
    outputs = llm.generate(prompts_text, sp)

    kd_records: List[Dict] = []
    empty_cnt = 0

    for src, out in zip(source_rows, outputs):
        if not out.outputs:
            empty_cnt += 1
            continue

        teacher_target = out.outputs[0].text.strip()
        if not teacher_target:
            empty_cnt += 1
            continue

        teacher_target_tokens = len(
            tok.encode(teacher_target, add_special_tokens=False)
        )

        kd_records.append(
            {
                "id": src["id"],
                "prompt_text": src["prompt_text"],
                "gold_target": src["gold_target"],
                "teacher_target": teacher_target,
                "messages": [
                    {"role": "user", "content": src["user_content"]},
                    {"role": "assistant", "content": teacher_target},
                ],
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
    print(f"Empty / skipped generations: {empty_cnt}")


if __name__ == "__main__":
    main()
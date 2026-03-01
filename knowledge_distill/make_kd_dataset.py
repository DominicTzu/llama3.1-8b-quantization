# make_kd_dataset.py
import os, json, random
from typing import Dict, List, Optional

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

TEACHER_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
IN_SFT_JSONL  = "/root/data/train_data/sft_10000.jsonl"
OUT_DIR       = "/root/data/kd_data"
OUT_JSONL     = "/root/data/kd_data/kd_10000.jsonl"

HF_TOKEN = os.environ.get("HF_TOKEN", None)

SEED = 0
N_KD  = 10_000

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.0
TOP_P = 1.0

MAX_PROMPT_TOKENS = 1024

DTYPE = "bfloat16"
GPU_MEMORY_UTIL = 0.90
MAX_MODEL_LEN = 2048

def set_seed(seed: int):
    random.seed(seed)

def extract_user_prompt(messages: List[Dict]) -> Optional[List[Dict]]:
    if not isinstance(messages, list) or len(messages) == 0:
        return None
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str):
            user = m["content"].strip()
            if user:
                return [{"role": "user", "content": user}]
            return None
    return None

def apply_template(tok: AutoTokenizer, messages: List[Dict]) -> str:
    if hasattr(tok, "apply_chat_template"):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "USER: " + messages[0]["content"].strip() + "\nASSISTANT:"

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
    prompts_msgs: List[List[Dict]] = []

    for i in idxs:
        msgs = ds[i].get("messages", None)
        p = extract_user_prompt(msgs)
        if p is None:
            continue
        p_text = apply_template(tok, p)
        if not p_text.endswith(" "):
            p_text += " "
        p_tok_len = len(tok.encode(p_text, add_special_tokens=False))
        if p_tok_len > MAX_PROMPT_TOKENS:
            continue
        prompts_text.append(p_text)
        prompts_msgs.append(p)
        if len(prompts_text) >= N_KD:
            break

    print(f"Collected prompts: {len(prompts_text)}")

    print(f"[3/4] Initializing vLLM teacher: {TEACHER_MODEL}")
    # 如果已 huggingface-cli login，通常不需要显式传 token；
    # 若 vLLM 版本支持 hf_token，可在这里加 hf_token=HF_TOKEN
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
        # 如果 vLLM 版本支持 stop_token_ids，可以加：
        stop_token_ids=[tok.eos_token_id] if tok.eos_token_id is not None else None
    )

    print("[4/4] Generating teacher responses...")
    outputs = llm.generate(prompts_text, sp)

    kd_records: List[Dict] = []
    for pm, out in zip(prompts_msgs, outputs):
        if not out.outputs:
            continue
        ans = out.outputs[0].text.strip()
        if not ans:
            continue
        kd_records.append({
            "messages": [
                {"role": "user", "content": pm[0]["content"]},
                {"role": "assistant", "content": ans},
            ]
        })

    write_jsonl(OUT_JSONL, kd_records)
    print(f"Done. Wrote KD dataset: {OUT_JSONL}")
    print(f"KD rows: {len(kd_records)}")

if __name__ == "__main__":
    main()
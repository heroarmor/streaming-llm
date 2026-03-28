"""
RULER benchmark inference with StreamingLLM.

Replaces call_api.py (which depends on sparse_attention) with standard
HuggingFace model + StreamingLLM attention sink + recent KV cache.

Usage:
    python pred/call_api_streaming.py \
        --model_name meta-llama/Llama-3.1-8B-Instruct \
        --attn_type streaming \
        --max_len 4096 \
        --data_dir <DATA_DIR> \
        --save_dir <PRED_DIR> \
        --benchmark synthetic \
        --task niah_single_1 \
        --start_size 4 \
        --recent_size 256 \
        --device cuda:0
"""

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from streaming_llm.enable_streaming_llm import enable_streaming_llm


def load_data(fname):
    lines = []
    with open(fname) as f:
        for line in f:
            lines.append(json.loads(line))
    return lines


def generate(model, tokenizer, input_text, max_new_tokens, kv_cache=None):
    """Greedy token-by-token generation, optionally with StreamingLLM."""
    inputs = tokenizer(input_text, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids.to(model.device)

    past_key_values = None
    generated_ids = []

    with torch.no_grad():
        # Prefill
        outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        if kv_cache is not None:
            past_key_values = kv_cache(past_key_values)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids.append(next_token.item())

        # Decode
        for _ in range(max_new_tokens - 1):
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            if kv_cache is not None:
                past_key_values = kv_cache(past_key_values)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids.append(next_token.item())

            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main(args):
    start_time = time.time()

    curr_folder = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(curr_folder))

    module = importlib.import_module(f"data.{args.benchmark}.constants")
    tasks_base = module.TASKS
    with open(os.path.join(curr_folder, f"../{args.benchmark}.yaml")) as f:
        tasks_customized = yaml.safe_load(f)

    if args.task not in tasks_customized:
        raise ValueError(f"{args.task} is not found in config")

    config = tasks_customized[args.task]
    config.update(tasks_base[config["task"]])
    max_new_tokens = config["tokens_to_generate"]

    task_file = args.data_dir / args.task / f"{args.subset}.jsonl"
    pred_file = args.save_dir / f"{args.task}.jsonl"
    pred_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Predict {args.task}\n  from {task_file}\n  to   {pred_file}")

    # Load data (skip already predicted)
    if os.path.exists(pred_file):
        pred_index = [s["index"] for s in load_data(str(pred_file))]
        data = [s for s in load_data(str(task_file)) if s["index"] not in pred_index]
        print(f"  Resuming: {len(pred_index)} already done, {len(data)} remaining")
    else:
        data = load_data(str(task_file))

    if len(data) == 0:
        print("  Nothing to predict, skipping.")
        return

    # Load model
    print(f"Loading {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map={"": args.device},
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()

    # Enable StreamingLLM (or full attention)
    kv_cache = None
    if args.attn_type == "streaming":
        kv_cache = enable_streaming_llm(
            model, start_size=args.start_size, recent_size=args.recent_size
        )
        print(f"StreamingLLM enabled: start_size={args.start_size}, recent_size={args.recent_size}")
    else:
        print("Full attention (no StreamingLLM)")

    # Run inference
    with open(pred_file, "a", encoding="utf-8", buffering=1) as fout:
        for dp in tqdm(data, desc=args.task):
            pred_text = generate(
                model, tokenizer, dp["input"], max_new_tokens, kv_cache=kv_cache
            )
            result = {
                "index": dp["index"],
                "pred": pred_text,
                "input": dp["input"],
                "outputs": dp["outputs"],
                "others": dp.get("others", {}),
                "truncation": dp.get("truncation", -1),
                "length": dp.get("length", -1),
            }
            fout.write(json.dumps(result) + "\n")

    print(f"Done. Time: {round((time.time() - start_time) / 60, 1)} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--benchmark", type=str, default="synthetic")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--subset", type=str, default="validation")

    # Model
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--max_len", type=int, default=4096)

    # Attention type
    parser.add_argument("--attn_type", type=str, default="streaming",
                        choices=["streaming", "full"],
                        help="streaming = StreamingLLM, full = standard attention")

    # StreamingLLM params
    parser.add_argument("--start_size", type=int, default=4,
                        help="Number of attention sink tokens")
    parser.add_argument("--recent_size", type=int, default=256,
                        help="Number of recent tokens to keep")

    args = parser.parse_args()
    main(args)

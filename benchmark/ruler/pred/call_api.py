# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Prepare prediction jsonl with field `pred` .
dataset jsonl:
{
    "index" int,
    "input": str,
    "outputs": [str],
}

prediction jsonl:
{
    "index" int,
    "input": str,
    "outputs": [str],
    "pred": str,
}
"""
# ruff: noqa: E402  # allow sys.path modification before imports

import argparse
import importlib
import json
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))  # allow imports from project root when run as a script

import numpy as np
import torch
import yaml  # type: ignore[import-untyped]
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import load_data

from benchmark.config import generate_config, parse_attn_args
from sparse_attention.cache.page_cache import PagedKVCache
from sparse_attention.models.llama3 import LlamaForCausalLM
from sparse_attention.profiling_utils import topp_stats

SERVER_TYPES = (
    "trtllm",
    "vllm",
    "openai",
    "gemini",
    "hf",
    "mamba",
)


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


class HuggingFaceModel:
    def __init__(
        self,
        model_name: str,
        max_len: int,
        max_new_len: int,
        attn_type: str,
        dtype: torch.dtype,
        device: str,
        budget_ratio: float,
        estimate_ratio: float,
        synthetic_len: int,
    ) -> None:
        self.device = device
        # Generate reattn config
        if attn_type == "RetroInfer":
            attn_config = generate_config(
                model_name,
                synthetic_len,
                attn_type,
                budget_ratio=budget_ratio,
                estimate_ratio=estimate_ratio,
            )
            attention_config = attn_config["RetroInfer"].copy()
            # Add computed est_nprobe field
            attention_config["est_nprobe"] = attention_config["max_compute_cluster_num"] - attention_config["nprobe"]
            print(attention_config)
        elif attn_type == "Quest":
            attn_config = generate_config(
                model_name,
                synthetic_len,
                attn_type,
            )
            attention_config = attn_config["Quest"]
            print(attention_config)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if "Llama" in model_name:
            llm = LlamaForCausalLM.from_pretrained(
                model_name,
                dtype=dtype,
                device_map=device,
                sparse_attention=attn_type,
                attention_config=attention_config,
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        self.llm = llm
        self.tokenizer = tokenizer
        self.max_new_len = max_new_len
        self.attn_type = attn_type

        self.model_name = model_name
        self.budget_ratio = budget_ratio
        self.estimate_ratio = estimate_ratio
        self.synthetic_len = synthetic_len
        self.attention_config = attention_config

        # Initialize paged KV caches for RetroInfer
        self.paged_kv_cache = None
        self.gpu_kv_cache = None

        context_len = max_len
        if attn_type == "RetroInfer" and attention_config is not None:
            self.paged_kv_cache = PagedKVCache(
                num_layers=llm.config.num_hidden_layers,
                num_kv_heads=1,
                head_dim=llm.config.hidden_size // llm.config.num_attention_heads,
                page_size=attention_config["page_size"],
                max_num_pages=attention_config["n_centroids"] * 3 * llm.config.num_key_value_heads * 2,
                dtype=dtype,
                device="cpu",
            )

            self.gpu_kv_cache = PagedKVCache(
                num_layers=llm.config.num_hidden_layers,
                num_kv_heads=1,
                head_dim=llm.config.hidden_size // llm.config.num_attention_heads,
                page_size=attention_config["page_size"],
                max_num_pages=1024,
                dtype=dtype,
                device=device,
            )

            llm.init_gather_attention(
                batch_size=1,
                seq_len=context_len,
                paged_kv_cache=self.paged_kv_cache,
                dtype=torch.bfloat16,
                device=device,
            )
        elif attn_type == "Quest":
            num_pages = context_len // attention_config["chunk_size"] + 1
            self.paged_kv_cache = PagedKVCache(
                num_layers=llm.config.num_hidden_layers,
                num_kv_heads=1,
                head_dim=llm.config.hidden_size // llm.config.num_attention_heads,
                page_size=int(attention_config["chunk_size"]),
                max_num_pages=num_pages * 4 * llm.config.num_key_value_heads,
                dtype=torch.bfloat16,
                device="cpu",
            )

            self.gpu_kv_cache = PagedKVCache(
                num_layers=llm.config.num_hidden_layers,
                num_kv_heads=1,
                head_dim=llm.config.hidden_size // llm.config.num_attention_heads,
                page_size=int(attention_config["chunk_size"]),
                max_num_pages=1024,
                dtype=torch.bfloat16,
                device=device,
            )

            llm.init_gather_attention(
                batch_size=1,
                seq_len=context_len,
                paged_kv_cache=self.paged_kv_cache,
                dtype=torch.bfloat16,
                device=device,
            )
        else:
            self.paged_kv_cache = PagedKVCache(
                num_layers=llm.config.num_hidden_layers,
                num_kv_heads=8,
                head_dim=llm.config.hidden_size // llm.config.num_attention_heads,
                page_size=16,
                max_num_pages=(max_len + max_new_len) // 16 + 1,
                dtype=dtype,
                device=device,
            )

    def __call__(self, prompt: str, **kwargs: object) -> dict[str, Any]:
        torch.cuda.set_device(self.device)
        generated_text, stats = get_pred(
            self.llm,
            self.tokenizer,
            input_text=prompt,
            max_new_tokens=self.max_new_len,
            attn_type=self.attn_type,
            paged_kv_cache=self.paged_kv_cache,
            gpu_kv_cache=self.gpu_kv_cache,
        )

        if stats is None:
            stats = {}

        return {"text": [generated_text], "stats": stats}


class ServerAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        namespace.server_type = values


def get_llm(
    model_name: str,
    max_len: int,
    max_new_len: int,
    attn_type: str,
    dtype: torch.dtype,
    device: str,
    budget_ratio: float,
    estimate_ratio: float,
    synthetic_len: int,
) -> HuggingFaceModel:
    if args.server_type == "hf":
        llm = HuggingFaceModel(
            model_name=model_name,
            max_len=max_len,
            max_new_len=max_new_len,
            attn_type=attn_type,
            dtype=dtype,
            device=device,
            budget_ratio=budget_ratio,
            estimate_ratio=estimate_ratio,
            synthetic_len=synthetic_len,
        )
    else:
        raise RuntimeError(f"Unsupported server type {args.server_type}")

    return llm


def get_pred(
    llm: Any,
    tokenizer: Any,
    input_text: str,
    max_new_tokens: int,
    attn_type: str,
    paged_kv_cache: Any | None = None,
    gpu_kv_cache: Any | None = None,
) -> tuple[str, dict[str, Any]]:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model_inputs = tokenizer([input_text], return_tensors="pt", padding=True).to(llm.device)

    # Initialize RoPE with max length
    max_length = model_inputs.input_ids.shape[1] + max_new_tokens
    llm.init_rope(max_length)

    # Clear topp_stats before generation
    topp_stats.real_budget.clear()
    topp_stats.real_sparsity.clear()
    topp_stats.kept_mass.clear()
    topp_stats.total_tokens.clear()

    # Generate with paged KV caches if using RetroInfer
    if attn_type == "RetroInfer" and paged_kv_cache is not None or attn_type == "Quest" and paged_kv_cache is not None:
        paged_kv_cache.clear()
        generated_ids = llm.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            paged_kv_cache=paged_kv_cache,
            gpu_kv_cache=gpu_kv_cache,
        )
    else:
        if paged_kv_cache is not None:
            paged_kv_cache.clear()
        generated_ids = llm.generate(
            **model_inputs, max_new_tokens=max_new_tokens, do_sample=False, paged_kv_cache=paged_kv_cache
        )

    output_ids = generated_ids[:, len(model_inputs.input_ids[0]) :].tolist()
    output: list[str] = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    # Collect statistics from topp_stats
    stats = {
        "real_budget": topp_stats.real_budget.copy(),
        "real_sparsity": topp_stats.real_sparsity.copy(),
        "kept_mass": topp_stats.kept_mass.copy(),
        "total_tokens": topp_stats.total_tokens.copy(),
        "avg_real_budget": float(np.mean(topp_stats.real_budget)) if topp_stats.real_budget else 0.0,
        "avg_real_sparsity": float(np.mean(topp_stats.real_sparsity)) if topp_stats.real_sparsity else 0.0,
        "avg_kept_mass": float(np.mean(topp_stats.kept_mass)) if topp_stats.kept_mass else 0.0,
    }

    print("Generated output:", output[0])
    return output[0], stats


def get_output(
    llm: HuggingFaceModel,
    outputs_parallel: dict[int, dict[str, object]],
    idx: int,
    index: int,
    input: str,
    outputs: list[str],
    others: dict[str, object],
    truncation: bool,
    length: int,
) -> None:
    # while True:
    #     try:
    #         pred = llm(prompt=input)
    #         break
    #     except Exception:
    #         traceback.print_exc()

    pred = llm(prompt=input)

    if len(pred["text"]) > 0:
        stats = pred.get("stats", {})
        outputs_parallel[idx] = {
            "index": index,
            "pred": pred["text"][0],
            "input": input,
            "outputs": outputs,
            "others": others,
            "truncation": truncation,
            "length": length,
            "avg_real_budget": stats.get("avg_real_budget", 0.0),
            "avg_real_sparsity": stats.get("avg_real_sparsity", 0.0),
            "avg_kept_mass": stats.get("avg_kept_mass", 0.0),
        }


def main(args: argparse.Namespace) -> None:
    start_time = time.time()

    curr_folder = os.path.dirname(os.path.abspath(__file__))

    try:
        sys.path.append(os.path.dirname(curr_folder))
        module = importlib.import_module(f"data.{args.benchmark}.constants")
    except ImportError:
        print(f"Module data.{args.benchmark}.constants not found.")

    tasks_base = module.TASKS
    with open(os.path.join(curr_folder, f"../{args.benchmark}.yaml")) as f:
        tasks_customized = yaml.safe_load(f)

    if args.task not in tasks_customized:
        raise ValueError(f"{args.task} is not found in config_tasks.yaml")

    config = tasks_customized.get(args.task)
    config.update(tasks_base[config["task"]])

    task_file = args.data_dir / args.task / f"{args.subset}.jsonl"

    # Modify save_dir to include config parameters in ATTN_TYPE for RetroInfer
    save_dir = args.save_dir
    if args.attn_type == "RetroInfer":
        attn_config = generate_config(
            args.model_name,
            args.synthetic_len,
            args.attn_type,
            budget_ratio=args.budget_ratio,
            estimate_ratio=args.estimate_ratio,
        )
        attn_config = attn_config[args.attn_type]

        # Add config parameters to ATTN_TYPE in save_dir path
        config_suffix = (
            f"_estcov-{attn_config['estimate_coverage']}"
            f"_tgtcov-{attn_config['target_coverage']}"
            f"_estbound-{attn_config['estimate_token_boundary']}"
            f"_boundcov-{attn_config['boundary_coverage']}"
        )

        # Modify save_dir by inserting config_suffix after ATTN_TYPE
        # Expected path: .../MODEL_NAME/BENCHMARK/MAX_SEQ_LENGTH/ATTN_TYPE/pred
        # Change to: .../MODEL_NAME/BENCHMARK/MAX_SEQ_LENGTH/ATTN_TYPE{config_suffix}/pred
        parts = save_dir.parts
        if len(parts) >= 2:
            # ATTN_TYPE is second to last (pred is last)
            attn_type_idx = len(parts) - 2
            modified_parts = (
                list(parts[:attn_type_idx]) + [parts[attn_type_idx] + config_suffix] + list(parts[attn_type_idx + 1 :])
            )
            save_dir = Path(*modified_parts)
    elif args.attn_type == "Quest":
        attn_config = generate_config(
            args.model_name,
            args.synthetic_len,
            args.attn_type,
            budget_ratio=args.budget_ratio,
            estimate_ratio=args.estimate_ratio,
        )
        attn_config = attn_config[args.attn_type]

        # Add config parameters to ATTN_TYPE in save_dir path
        config_suffix = (
            f"_estcov-{attn_config['estimate_coverage']}"
            f"_tgtcov-{attn_config['target_coverage']}"
            f"_usequest-{attn_config['use_quest_selector']}"
        )

        # Modify save_dir by inserting config_suffix after ATTN_TYPE
        # Expected path: .../MODEL_NAME/BENCHMARK/MAX_SEQ_LENGTH/ATTN_TYPE/pred
        # Change to: .../MODEL_NAME/BENCHMARK/MAX_SEQ_LENGTH/ATTN_TYPE{config_suffix}/pred
        parts = save_dir.parts
        if len(parts) >= 2:
            # ATTN_TYPE is second to last (pred is last)
            attn_type_idx = len(parts) - 2
            modified_parts = (
                list(parts[:attn_type_idx]) + [parts[attn_type_idx] + config_suffix] + list(parts[attn_type_idx + 1 :])
            )
            save_dir = Path(*modified_parts)

    if args.chunk_amount > 1:
        pred_file = save_dir / f"{args.task}-{args.chunk_idx}.jsonl"
    else:
        pred_file = save_dir / f"{args.task}.jsonl"

    print(f"Predict {args.task} \nfrom {task_file}\nto {pred_file}")
    pred_file.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    if os.path.exists(pred_file):
        pred_index = [sample["index"] for sample in load_data(pred_file)]
        data = [sample for sample in load_data(task_file) if sample["index"] not in pred_index]
    else:
        data = load_data(task_file)

    # Load api
    torch.cuda.set_device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    llm = get_llm(
        args.model_name,
        args.max_len,
        config["tokens_to_generate"],
        args.attn_type,
        dtype,
        args.device,
        budget_ratio=args.budget_ratio,
        estimate_ratio=args.estimate_ratio,
        synthetic_len=args.synthetic_len,
    )

    threads = []
    outputs_parallel: dict[int, dict[str, object]] = {}
    # setting buffering=1 to force to dump the output after every line, so that we can see intermediate generations
    with open(pred_file, "a", encoding="utf-8", buffering=1) as fout:
        torch.cuda.set_device(args.device)
        for idx, data_point in tqdm(enumerate(data), total=len(data)):
            thread = threading.Thread(
                target=get_output,
                kwargs=dict(
                    llm=llm,
                    outputs_parallel=outputs_parallel,
                    idx=idx,
                    index=data_point["index"],
                    input=data_point["input"],
                    outputs=data_point["outputs"],
                    others=data_point.get("others", {}),
                    truncation=data_point.get("truncation", -1),
                    length=data_point.get("length", -1),
                ),
            )
            thread.start()
            threads.append(thread)
            if len(threads) == args.threads:
                for thread in threads:
                    thread.join()
                threads = []
                for computed_idx in range(idx - args.threads + 1, idx + 1):
                    if len(outputs_parallel[computed_idx]) > 0:
                        fout.write(json.dumps(outputs_parallel[computed_idx]) + "\n")

        # collecting the final batch
        if len(data) > 0:
            for thread in threads:
                thread.join()
            for computed_idx in range(idx - len(threads) + 1, idx + 1):
                if len(outputs_parallel[computed_idx]) > 0:
                    fout.write(json.dumps(outputs_parallel[computed_idx]) + "\n")

    print(f"Used time: {round((time.time() - start_time) / 60, 1)} minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--data_dir", type=Path, required=True, help="path to load the dataset jsonl files")
    parser.add_argument("--save_dir", type=Path, required=True, help="path to save the prediction jsonl files")
    parser.add_argument("--benchmark", type=str, default="synthetic", help="Options: [synthetic]")
    parser.add_argument("--task", type=str, required=True, help="Options: tasks in benchmark")
    parser.add_argument("--subset", type=str, default="validation", help="Options: validation or test")
    parser.add_argument("--chunk_idx", type=int, default=0, help="index of current split chunk")
    parser.add_argument("--chunk_amount", type=int, default=1, help="size of split chunk")

    # Server
    parser.add_argument("--server_type", default="nemo", action=ServerAction, choices=SERVER_TYPES)
    parser.add_argument("--server_host", type=str, default="127.0.0.1")
    parser.add_argument("--server_port", type=str, default="5000")
    parser.add_argument("--ssh_server", type=str)
    parser.add_argument("--ssh_key_path", type=str)

    # Inference
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        choices=[
            "Qwen/Qwen2.5-7B-Instruct",
            "gradientai/Llama-3-8B-Instruct-Gradient-1048k",
            "meta-llama/Llama-3.1-8B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct",
        ],
        help="huggingface model name",
    )
    parser.add_argument(
        "--attn_type",
        type=str,
        default="Full_Flash_Attn",
        choices=["Full_Flash_Attn", "RetroInfer", "Quest"],
        help="Attention method",
    )
    parser.add_argument("--max_len", type=int, default=128000)
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=32)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--sliding_window_size", type=int)
    parser.add_argument("--threads", type=int, default=4)

    parser.add_argument("--synthetic_len", type=int, required=True)

    parser = parse_attn_args(parser)

    args = parser.parse_args()
    print(args)

    if args.server_type == "hf" or args.server_type == "gemini":
        args.threads = 1

    seed_everything(2025)
    main(args)

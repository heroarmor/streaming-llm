import argparse
import json
import os
import random
import sys
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)
from transformers import AutoTokenizer  # noqa: E402

from benchmark.config import generate_config, parse_attn_args  # noqa: E402
from sparse_attention.cache.page_cache import PagedKVCache  # noqa: E402
from sparse_attention.models.llama3 import LlamaForCausalLM  # noqa: E402

with open("config/model2path.json") as f:
    model2path = json.load(f)
with open("config/model2maxlen.json") as f:
    model2maxlen = json.load(f)
# we design specific prompt format and max generation length for each task
# feel free to modify them to optimize model output
with open("config/dataset2prompt.json") as f:
    dataset2prompt = json.load(f)
with open("config/dataset2maxlen.json") as f:
    dataset2maxlen = json.load(f)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attn_type",
        type=str,
        default="Full_Flash_Attn",
        choices=["Full_Flash_Attn", "RetroInfer", "Quest"],
        help="Attention method",
    )
    parser.add_argument(
        "--model", type=str, default=None, choices=["llama-3-8b-1048k", "qwen2.5-7b", "llama-3.1-8b", "qwen2.5-72b"]
    )
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"], help="Dtype")
    parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")
    parser.add_argument("--task", type=str, required=True, help="task name. work when --e is false")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--num_examples", type=int, default=-1, help="num of example to evaluate. -1 for all.")

    parser = parse_attn_args(parser)

    return parser.parse_args(args)


def get_pred(
    llm: Any,
    tokenizer: Any,
    data: list[dict[str, object]],
    max_new_tokens: int,
    prompt_format: str,
    model_name: str,
    out_path: str,
    args: argparse.Namespace,
) -> None:
    attn_type = args.attn_type
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)

        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model_inputs = tokenizer([prompt], return_tensors="pt", padding=True).to(llm.device)
        context_len = model_inputs.input_ids.shape[1]
        print("Input length:", model_inputs.input_ids.shape[1])

        # Initialize RoPE with max length
        max_length = model_inputs.input_ids.shape[1] + max_new_tokens
        llm.init_rope(max_length)

        torch.cuda.set_device(args.device)

        # Generate with paged KV caches if using RetroInfer
        if attn_type == "RetroInfer":
            attn_config = generate_config(
                model2path[model_name],
                model_inputs.input_ids.shape[1],
                attn_type,
                budget_ratio=args.budget_ratio,
                estimate_ratio=args.estimate_ratio,
            )
            reattn_config = attn_config["RetroInfer"].copy()
            # Add computed est_nprobe field
            reattn_config["est_nprobe"] = reattn_config["max_compute_cluster_num"] - reattn_config["nprobe"]
            print(reattn_config)
            llm.update_attention_config(reattn_config)

            paged_kv_cache = PagedKVCache(
                num_layers=llm.config.num_hidden_layers,
                num_kv_heads=1,
                head_dim=llm.config.hidden_size // llm.config.num_attention_heads,
                page_size=reattn_config["page_size"],
                max_num_pages=reattn_config["n_centroids"] * 3 * llm.config.num_key_value_heads,
                dtype=dtype,
                device="cpu",
            )

            gpu_kv_cache = PagedKVCache(
                num_layers=llm.config.num_hidden_layers,
                num_kv_heads=1,
                head_dim=llm.config.hidden_size // llm.config.num_attention_heads,
                page_size=reattn_config["page_size"],
                max_num_pages=1,
                dtype=dtype,
                device=args.device,
            )

            llm.init_gather_attention(
                batch_size=1,
                seq_len=context_len,
                paged_kv_cache=paged_kv_cache,
                dtype=dtype,
                device=args.device,
            )

            generated_ids = llm.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                paged_kv_cache=paged_kv_cache,
                gpu_kv_cache=gpu_kv_cache,
            )
        elif attn_type == "Quest":
            attn_config = generate_config(
                model2path[model_name],
                model_inputs.input_ids.shape[1],
                attn_type,
            )
            quest_config = attn_config["Quest"]
            print(quest_config)
            llm.update_attention_config(quest_config)

            num_pages = context_len // quest_config["chunk_size"] + 1
            paged_kv_cache = PagedKVCache(
                num_layers=llm.config.num_hidden_layers,
                num_kv_heads=1,
                head_dim=llm.config.hidden_size // llm.config.num_attention_heads,
                page_size=int(quest_config["chunk_size"]),
                max_num_pages=num_pages * 4 * llm.config.num_key_value_heads,
                dtype=dtype,
                device="cpu",
            )

            gpu_kv_cache = PagedKVCache(
                num_layers=llm.config.num_hidden_layers,
                num_kv_heads=1,
                head_dim=llm.config.hidden_size // llm.config.num_attention_heads,
                page_size=int(quest_config["chunk_size"]),
                max_num_pages=1,
                dtype=dtype,
                device=args.device,
            )

            llm.init_gather_attention(
                batch_size=1,
                seq_len=context_len,
                paged_kv_cache=paged_kv_cache,
                dtype=dtype,
                device=args.device,
            )

            generated_ids = llm.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                paged_kv_cache=paged_kv_cache,
                gpu_kv_cache=gpu_kv_cache,
            )
        else:
            generated_ids = llm.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=False)

        output_ids = generated_ids[:, len(model_inputs.input_ids[0]) :].tolist()
        output: list[str] = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        torch.cuda.empty_cache()

        print("Generated output:", output[0][:50])

        pred = output[0]

        with open(out_path, "a", encoding="utf-8") as f:
            json.dump(
                {
                    "pred": pred,
                    "answers": json_obj["answers"],
                    "all_classes": json_obj["all_classes"],
                    "length": json_obj["length"],
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model(
    model_path: str,
    max_len: int,
    dtype: torch.dtype,
    device: str,
    attn_type: str,
    args: argparse.Namespace,
) -> tuple[object, object]:
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if "Llama" in model_path:
        llm = LlamaForCausalLM.from_pretrained(model_path, dtype=dtype, device_map=device, sparse_attention=attn_type)
    else:
        raise ValueError(f"Unsupported model: {model_path}")

    return llm, tokenizer


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()

    num_examples = args.num_examples
    attn_type = args.attn_type
    model_name = args.model  # not hf model path
    device = args.device
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    max_length = model2maxlen[model_name]
    model_path = model2path[model_name]

    llm, tokenizer = load_model(model_path, max_length, dtype, device, attn_type, args)

    if args.e:
        datasets = [
            "qasper",
            "multifieldqa_en",
            "hotpotqa",
            "2wikimqa",
            "gov_report",
            "multi_news",
            "trec",
            "triviaqa",
            "samsum",
            "passage_count",
            "passage_retrieval_en",
            "lcc",
            "repobench-p",
        ]
    else:
        datasets = [args.task]
        # datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh",
        # "hotpotqa", "2wikimqa", "musique", \
        # "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
        # "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

    # predict on each dataset
    if not os.path.exists("results/pred"):
        os.makedirs("results/pred")
    if not os.path.exists("results/pred_e"):
        os.makedirs("results/pred_e")

    # Build config suffix for RetroInfer
    config_suffix = ""
    if attn_type == "RetroInfer":
        # Generate config to get the parameters
        attn_config = generate_config(
            model_path,
            max_length,
            attn_type,
            budget_ratio=args.budget_ratio,
            estimate_ratio=args.estimate_ratio,
        )
        retroinfer_config = attn_config["RetroInfer"]

        config_suffix = (
            f"_estcov-{retroinfer_config['estimate_coverage']}"
            f"_tgtcov-{retroinfer_config['target_coverage']}"
            f"_estbound-{retroinfer_config['estimate_token_boundary']}"
            f"_boundcov-{retroinfer_config['boundary_coverage']}"
        )

    for dataset in datasets:
        if args.e:
            data = load_dataset("THUDM/LongBench", f"{dataset}_e", split="test")

            prefix = f"results/pred_e/{model_name}/{attn_type}{config_suffix}"
            if not os.path.exists(prefix):
                os.makedirs(prefix)
            out_path = f"{prefix}/{dataset}.jsonl"
        else:
            data = load_dataset("THUDM/LongBench", dataset, split="test")

            prefix = f"results/pred/{model_name}/{attn_type}{config_suffix}"
            if not os.path.exists(prefix):
                os.makedirs(prefix)
            out_path = f"{prefix}/{dataset}.jsonl"

        prompt_format = dataset2prompt[dataset]
        max_new_tokens = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        data_all = data_all[:num_examples] if num_examples > 0 else data_all

        get_pred(
            llm,
            tokenizer,
            data_all,
            max_new_tokens,
            prompt_format,
            model_name,
            out_path,
            args,
        )

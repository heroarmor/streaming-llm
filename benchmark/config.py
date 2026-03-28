import argparse
import json
import math
import os
import sys
from typing import Any, cast

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)


def parse_attn_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--budget_ratio", type=float, default=0.018, help="ratio of budget")
    parser.add_argument(
        "--estimate_ratio", type=float, default=0.25, help="ratio of estimated clusters for RetriveInfer"
    )
    parser.add_argument("--estimate_coverage", type=str, default="None", help="whether to estimate coverage")
    parser.add_argument("--target_coverage", type=float, default=0.95, help="target coverage for coverage estimation")

    return parser


def generate_config(
    model_name: str,
    context_len: int,
    attn_type: str,
    budget_ratio: float = 0.018,
    estimate_ratio: float = 0.25,
    # default retrieve infer configs
    n_segments: int | None = None,
    # quest config overrides
    **quest_overrides: Any,
) -> dict[str, Any]:
    aprox_cluster_size = 16

    CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
    MODEL_NAME = model_name.split("/")[-1] + ".json"
    CONFIG_FILE = os.path.join(CONFIG_DIR, MODEL_NAME)
    with open(CONFIG_FILE) as f:
        original_config = cast(dict[str, Any], json.load(f))

    if attn_type == "RetroInfer":
        if n_segments is None:
            n_segments = max(1, context_len // 8192)

        n_clusters = math.ceil(context_len / aprox_cluster_size)

        # compute the nearest multiple of (n_segments*32)
        lower = (n_clusters // (n_segments * 32)) * (n_segments * 32)
        upper = lower + (n_segments * 32)
        n_clusters = lower if abs(n_clusters - lower) <= abs(n_clusters - upper) else upper

        nprobe = max(1, int(n_clusters * budget_ratio))
        print(f"context_len: {context_len}, n_clusters: {n_clusters}, nprobe: {nprobe}, n_segments: {n_segments}")

        original_config[attn_type]["n_centroids"] = n_clusters
        original_config[attn_type]["n_segment"] = n_segments
        original_config[attn_type]["nprobe"] = nprobe
        original_config[attn_type]["cache_cluster_num"] = int(nprobe * 3)
        original_config[attn_type]["max_compute_cluster_num"] = int(n_clusters * estimate_ratio) + nprobe

        print(original_config[attn_type])

    elif attn_type == "Quest":
        # Apply any overrides from quest_overrides
        for key, value in quest_overrides.items():
            if key in original_config[attn_type]:
                original_config[attn_type][key] = value

        print(f"Quest config: {original_config[attn_type]}")

    return original_config

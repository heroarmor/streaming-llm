"""
End-to-end tests for StreamingLLM with modern transformers (Llama 3.1 compatible).

Uses a tiny randomly-initialized LlamaForCausalLM so tests run on CPU in seconds.

Run:
    pytest tests/test_e2e.py -v
"""

import torch
import pytest
from transformers import LlamaConfig, LlamaForCausalLM

from streaming_llm.enable_streaming_llm import enable_streaming_llm


def _make_tiny_llama():
    """Create a minimal Llama model for testing (< 1M params, CPU-friendly)."""
    cfg = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,  # GQA like Llama 3.1
        vocab_size=256,
        max_position_embeddings=512,
    )
    model = LlamaForCausalLM(cfg)
    model.eval()
    return model


# ------------------------------------------------------------------
# Test 1: basic forward with streaming enabled (no eviction yet)
# ------------------------------------------------------------------
def test_streaming_forward_no_eviction():
    """Model should produce valid logits with StreamingLLM enabled,
    even before the cache is full enough to trigger eviction."""
    model = _make_tiny_llama()
    kv_cache = enable_streaming_llm(model, start_size=4, recent_size=10)

    past_key_values = None
    input_ids = torch.randint(0, 256, (1, 1))

    # Run 8 steps (< start_size + recent_size = 14, so no eviction)
    for _ in range(8):
        with torch.no_grad():
            out = model(input_ids, past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values
        logits = out.logits
        assert logits.shape == (1, 1, 256), f"Bad logits shape: {logits.shape}"
        input_ids = logits.argmax(dim=-1)


# ------------------------------------------------------------------
# Test 2: cache eviction triggers correctly
# ------------------------------------------------------------------
def test_cache_eviction():
    """After exceeding start_size + recent_size tokens, KV cache should
    be evicted down to exactly start_size + recent_size."""
    model = _make_tiny_llama()
    start_size, recent_size = 4, 6
    cache_size = start_size + recent_size  # 10
    kv_cache = enable_streaming_llm(model, start_size=start_size, recent_size=recent_size)

    past_key_values = None
    input_ids = torch.randint(0, 256, (1, 1))

    for step in range(20):
        with torch.no_grad():
            out = model(input_ids, past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values
        past_key_values = kv_cache(past_key_values)

        # Check cache length never exceeds cache_size
        if hasattr(past_key_values, "get_seq_length"):
            seq_len = past_key_values.get_seq_length()
        else:
            seq_len = past_key_values[0][0].shape[2]

        assert seq_len <= cache_size, (
            f"Step {step}: cache seq_len={seq_len} > cache_size={cache_size}"
        )

        input_ids = out.logits.argmax(dim=-1)


# ------------------------------------------------------------------
# Test 3: long streaming produces finite outputs (no NaN/Inf)
# ------------------------------------------------------------------
def test_long_streaming_no_nan():
    """Run 50 steps of streaming generation and verify no NaN/Inf in logits."""
    model = _make_tiny_llama()
    kv_cache = enable_streaming_llm(model, start_size=4, recent_size=8)

    past_key_values = None
    input_ids = torch.randint(0, 256, (1, 1))

    for step in range(50):
        with torch.no_grad():
            out = model(input_ids, past_key_values=past_key_values, use_cache=True)
        past_key_values = kv_cache(out.past_key_values)
        logits = out.logits

        assert torch.isfinite(logits).all(), f"Step {step}: NaN or Inf in logits"
        input_ids = logits.argmax(dim=-1)


# ------------------------------------------------------------------
# Test 4: _seen_tokens stays in sync with actual cache length
# ------------------------------------------------------------------
def test_seen_tokens_sync():
    """After eviction, DynamicCache._seen_tokens must equal actual KV length."""
    model = _make_tiny_llama()
    kv_cache = enable_streaming_llm(model, start_size=2, recent_size=4)

    past_key_values = None
    input_ids = torch.randint(0, 256, (1, 1))

    for step in range(20):
        with torch.no_grad():
            out = model(input_ids, past_key_values=past_key_values, use_cache=True)
        past_key_values = kv_cache(out.past_key_values)

        if hasattr(past_key_values, "_seen_tokens"):
            actual_len = past_key_values.key_cache[0].shape[2]
            assert past_key_values._seen_tokens == actual_len, (
                f"Step {step}: _seen_tokens={past_key_values._seen_tokens} "
                f"!= actual_len={actual_len}"
            )

        input_ids = out.logits.argmax(dim=-1)


# ------------------------------------------------------------------
# Test 5: evict_for_space works for batched prefill
# ------------------------------------------------------------------
def test_evict_for_space():
    """evict_for_space should make room for incoming tokens."""
    model = _make_tiny_llama()
    start_size, recent_size = 2, 6
    cache_size = start_size + recent_size
    kv_cache = enable_streaming_llm(model, start_size=start_size, recent_size=recent_size)

    past_key_values = None
    # Fill cache with single tokens first
    for _ in range(cache_size):
        input_ids = torch.randint(0, 256, (1, 1))
        with torch.no_grad():
            out = model(input_ids, past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values

    # Now evict_for_space for 3 incoming tokens
    past_key_values = kv_cache.evict_for_space(past_key_values, num_coming=3)

    if hasattr(past_key_values, "get_seq_length"):
        seq_len = past_key_values.get_seq_length()
    else:
        seq_len = past_key_values[0][0].shape[2]

    assert seq_len + 3 <= cache_size, (
        f"After evict_for_space: seq_len={seq_len}, "
        f"seq_len + 3 = {seq_len + 3} > cache_size={cache_size}"
    )

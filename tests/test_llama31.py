"""
Compare Full Attention vs StreamingLLM on Llama-3.1-8B-Instruct.

Run:
    python tests/test_llama31.py
"""

import os
os.environ["HF_HOME"] = "/tmp/hf_cache"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from streaming_llm.enable_streaming_llm import enable_streaming_llm

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 128
PROMPT = "Explain what StreamingLLM is in one sentence."


def generate_token_by_token(model, input_ids, tokenizer, kv_cache=None):
    """Token-by-token greedy generation. Returns (text, num_tokens)."""
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
        for _ in range(MAX_NEW_TOKENS - 1):
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            if kv_cache is not None:
                past_key_values = kv_cache(past_key_values)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids.append(next_token.item())

            if next_token.item() == tokenizer.eos_token_id:
                break

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text, len(generated_ids)


def main():
    print(f"Loading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="cuda:0",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()

    # Build input
    messages = [{"role": "user", "content": PROMPT}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

    # --- 1) Full Attention (no StreamingLLM) ---
    print("\n" + "=" * 60)
    print("[Full Attention]")
    print(f"Prompt: {PROMPT}")
    text_full, n_full = generate_token_by_token(model, input_ids, tokenizer)
    print(f"Response: {text_full}")
    print(f"Tokens: {n_full}")

    # --- 2) StreamingLLM ---
    # Re-load to get a clean model (undo monkey-patch)
    del model
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="cuda:0",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()

    kv_cache = enable_streaming_llm(model, start_size=4, recent_size=256)

    print("\n" + "=" * 60)
    print("[StreamingLLM] start_size=4, recent_size=256")
    print(f"Prompt: {PROMPT}")
    text_stream, n_stream = generate_token_by_token(
        model, input_ids, tokenizer, kv_cache=kv_cache
    )
    print(f"Response: {text_stream}")
    print(f"Tokens: {n_stream}")

    print("\n" + "=" * 60)
    print("DONE")


if __name__ == "__main__":
    main()

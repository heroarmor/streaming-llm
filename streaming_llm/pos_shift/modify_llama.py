"""
Position shift attention for LlamaAttention.

Supports both:
  - Legacy transformers (<= ~4.38): LlamaAttention has self.rotary_emb,
    apply_rotary_pos_emb(q, k, cos, sin, position_ids)
  - Modern transformers (>= ~4.43 / 5.x): rotary_emb lives on LlamaModel,
    cos/sin are pre-computed and passed as position_embeddings tuple;
    apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
"""

import math
import types
import inspect
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from transformers.models.llama.modeling_llama import LlamaAttention, rotate_half

# ---------------------------------------------------------------------------
# Detect API version
# ---------------------------------------------------------------------------
_NEW_API = "position_embeddings" in inspect.signature(LlamaAttention.forward).parameters

if _NEW_API:
    # modern transformers — apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim)
    from transformers.models.llama.modeling_llama import (
        apply_rotary_pos_emb,
        repeat_kv,
    )
else:
    # legacy transformers — apply_rotary_pos_emb(q, k, cos, sin, position_ids)
    from transformers.models.llama.modeling_llama import (
        apply_rotary_pos_emb,
        repeat_kv,
    )

__all__ = ["enable_llama_pos_shift_attention"]


# ---------------------------------------------------------------------------
# Helper: apply RoPE to a single tensor (used by both paths)
# ---------------------------------------------------------------------------
def _apply_rotary_single(x, cos, sin):
    """Apply rotary embedding to x. cos/sin shape: [bs, seq_len, dim]."""
    cos = cos.unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin.unsqueeze(1)  # [bs, 1, seq_len, dim]
    return (x * cos) + (rotate_half(x) * sin)


# ===================================================================
# Modern transformers (>= 4.43 / 5.x)
# ===================================================================
def _modern_pos_shift_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    past_key_values=None,
    cache_position: torch.LongTensor | None = None,
    **kwargs,
):
    """
    StreamingLLM position-shift forward for modern transformers.

    Core idea (unchanged from original paper):
      1. Store keys WITHOUT RoPE in the KV cache.
      2. At each step, assign contiguous positions 0..kv_len-1 to all
         cached keys and apply RoPE on-the-fly.
      3. Query gets RoPE with position = len(cache) (i.e. right after the
         last cached key position), which is exactly what the model would
         see if the cache keys occupied slots 0..kv_len-1 naturally.
    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    bsz, _, q_len, _ = query_states.shape

    # --- Concatenate with cached (un-rotated) keys/values ----------------
    # DynamicCache.update() appends new KV to existing cache and returns
    # the full sequence.  We store un-rotated keys so that after eviction
    # we can re-apply RoPE with contiguous positions.
    if past_key_values is not None:
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx,
        )

    kv_seq_len = key_states.shape[2]

    # --- Compute RoPE with contiguous positions --------------------------
    # We need rotary_emb to compute cos/sin for arbitrary positions.
    # It was attached by enable_llama_pos_shift_attention().
    rotary_emb = self._streaming_rotary_emb

    # Key positions: 0, 1, ..., kv_seq_len - 1  (contiguous)
    key_position_ids = torch.arange(
        kv_seq_len, device=key_states.device, dtype=torch.long
    ).unsqueeze(0).expand(bsz, -1)

    # Query positions: kv_seq_len - q_len, ..., kv_seq_len - 1
    query_position_ids = torch.arange(
        kv_seq_len - q_len, kv_seq_len, device=key_states.device, dtype=torch.long
    ).unsqueeze(0).expand(bsz, -1)

    # Compute cos/sin for all positions up to kv_seq_len
    all_cos, all_sin = rotary_emb(key_states, key_position_ids)
    # all_cos shape: [bsz, kv_seq_len, head_dim]

    # Apply RoPE to keys with contiguous positions
    key_states = _apply_rotary_single(key_states, all_cos, all_sin)

    # Apply RoPE to queries with their positions
    q_cos, q_sin = rotary_emb(query_states, query_position_ids)
    query_states = _apply_rotary_single(query_states, q_cos, q_sin)

    # --- Attention -------------------------------------------------------
    key_states_expanded = repeat_kv(key_states, self.num_key_value_groups)
    value_states_expanded = repeat_kv(value_states, self.num_key_value_groups)

    scaling = self.head_dim ** -0.5
    attn_weights = torch.matmul(
        query_states, key_states_expanded.transpose(2, 3)
    ) * scaling

    if attention_mask is not None:
        # attention_mask may have been computed for a different kv_seq_len
        # (e.g., if cache was evicted). Truncate/pad if needed.
        if attention_mask.shape[-1] > kv_seq_len:
            attention_mask = attention_mask[:, :, :, :kv_seq_len]
        elif attention_mask.shape[-1] < kv_seq_len:
            # Build a simple causal mask
            causal_mask = torch.triu(
                torch.full((q_len, kv_seq_len), float("-inf"), device=query_states.device),
                diagonal=kv_seq_len - q_len + 1,
            )
            attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    attn_output = torch.matmul(attn_weights, value_states_expanded)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1)
    attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights


# ===================================================================
# Legacy transformers (<= ~4.38)
# ===================================================================
def _apply_rotary_pos_emb_single_legacy(x, cos, sin, position_ids):
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    return (x * cos) + (rotate_half(x) * sin)


def _legacy_pos_shift_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:
        key_value_slicing = (
            self.num_key_value_heads * self.head_dim
        ) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [
            F.linear(hidden_states, query_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [
            F.linear(hidden_states, key_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [
            F.linear(hidden_states, value_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states = _apply_rotary_pos_emb_single_legacy(
        query_states, cos, sin, position_ids
    )

    if past_key_value is not None:
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    key_position_ids = torch.arange(
        kv_seq_len, device=position_ids.device
    ).unsqueeze(0)
    key_states = _apply_rotary_pos_emb_single_legacy(
        key_states, cos, sin, key_position_ids
    )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(
        query_states, key_states.transpose(2, 3)
    ) / math.sqrt(self.head_dim)

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(
            self.hidden_size // self.config.pretraining_tp, dim=2
        )
        o_proj_slices = self.o_proj.weight.split(
            self.hidden_size // self.config.pretraining_tp, dim=1
        )
        attn_output = sum(
            [
                F.linear(attn_output[i], o_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
        )
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


# ===================================================================
# Entry point
# ===================================================================
def enable_llama_pos_shift_attention(model):
    """
    Monkey-patch all LlamaAttention modules to use position-shifted
    attention (StreamingLLM).  Works with both old and new transformers.
    """
    if _NEW_API:
        # Grab the shared rotary_emb from LlamaModel
        rotary_emb = model.model.rotary_emb

        for name, module in model.named_modules():
            if isinstance(module, LlamaAttention):
                # Attach rotary_emb reference so the forward can use it
                module._streaming_rotary_emb = rotary_emb
                module.forward = types.MethodType(
                    _modern_pos_shift_forward, module
                )
        print("[StreamingLLM] Enabled position-shift attention (modern API)")
    else:
        for name, module in reversed(model._modules.items()):
            if len(list(module.children())) > 0:
                enable_llama_pos_shift_attention(module)

            if isinstance(module, LlamaAttention):
                model._modules[name].forward = types.MethodType(
                    _legacy_pos_shift_forward, model._modules[name]
                )
        print("[StreamingLLM] Enabled position-shift attention (legacy API)")

import torch


def slice2d(x, start, end):
    return x[:, :, start:end, ...]


def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]


def slice1d(x, start, end):
    return x[:, start:end, ...]


DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}


def _is_dynamic_cache(past_key_values):
    """Check if past_key_values is a Cache object (DynamicCache or similar).

    Supports both:
      - transformers 5.x: DynamicCache has .layers list of DynamicLayer
        (each with .keys / .values tensors)
      - older transformers: DynamicCache has .key_cache / .value_cache lists
    """
    return hasattr(past_key_values, "layers") or (
        hasattr(past_key_values, "key_cache")
        and hasattr(past_key_values, "value_cache")
    )


def _cache_num_layers(past_key_values):
    if hasattr(past_key_values, "layers"):
        return len(past_key_values.layers)
    return len(past_key_values.key_cache)


def _cache_get_kv(past_key_values, i):
    """Return (key, value) tensors for layer i."""
    if hasattr(past_key_values, "layers"):
        layer = past_key_values.layers[i]
        return layer.keys, layer.values
    return past_key_values.key_cache[i], past_key_values.value_cache[i]


def _cache_set_kv(past_key_values, i, k, v):
    """Set (key, value) tensors for layer i."""
    if hasattr(past_key_values, "layers"):
        past_key_values.layers[i].keys = k
        past_key_values.layers[i].values = v
    else:
        past_key_values.key_cache[i] = k
        past_key_values.value_cache[i] = v


class StartRecentKVCache:
    def __init__(
        self,
        start_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
    ):
        print(f"StartRecentKVCache: {start_size}, {recent_size}")
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]

    def _evict_tuple(self, past_key_values, start_size, recent_start, seq_len):
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, start_size),
                        self.k_slice(k, recent_start, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, start_size),
                        self.v_slice(v, recent_start, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]

    def _evict_dynamic_cache(self, past_key_values, start_size, recent_start, seq_len):
        for i in range(_cache_num_layers(past_key_values)):
            k, v = _cache_get_kv(past_key_values, i)
            new_k = torch.cat(
                [
                    self.k_slice(k, 0, start_size),
                    self.k_slice(k, recent_start, seq_len),
                ],
                dim=self.k_seq_dim,
            )
            new_v = torch.cat(
                [
                    self.v_slice(v, 0, start_size),
                    self.v_slice(v, recent_start, seq_len),
                ],
                dim=self.v_seq_dim,
            )
            _cache_set_kv(past_key_values, i, new_k, new_v)
        return past_key_values

    def _get_seq_len(self, past_key_values):
        if _is_dynamic_cache(past_key_values):
            if hasattr(past_key_values, "get_seq_length"):
                return past_key_values.get_seq_length()
            if _cache_num_layers(past_key_values) == 0:
                return 0
            k, _ = _cache_get_kv(past_key_values, 0)
            return k.size(self.k_seq_dim)
        else:
            return past_key_values[0][0].size(self.k_seq_dim)

    def _evict(self, past_key_values, start_size, recent_start, seq_len):
        if _is_dynamic_cache(past_key_values):
            return self._evict_dynamic_cache(
                past_key_values, start_size, recent_start, seq_len
            )
        else:
            return self._evict_tuple(
                past_key_values, start_size, recent_start, seq_len
            )

    def __call__(self, past_key_values):
        if past_key_values is None:
            return None
        seq_len = self._get_seq_len(past_key_values)
        if seq_len <= self.cache_size:
            return past_key_values
        return self._evict(
            past_key_values,
            self.start_size,
            seq_len - self.recent_size,
            seq_len,
        )

    def evict_for_space(self, past_key_values, num_coming):
        if past_key_values is None:
            return None
        seq_len = self._get_seq_len(past_key_values)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values
        return self._evict(
            past_key_values,
            self.start_size,
            seq_len - self.recent_size + num_coming,
            seq_len,
        )

    def evict_range(self, past_key_values, start, end):
        if past_key_values is None:
            return None
        seq_len = self._get_seq_len(past_key_values)
        assert start <= end and end <= seq_len

        if _is_dynamic_cache(past_key_values):
            for i in range(_cache_num_layers(past_key_values)):
                k, v = _cache_get_kv(past_key_values, i)
                new_k = torch.cat(
                    [self.k_slice(k, 0, start), self.k_slice(k, end, seq_len)],
                    dim=self.k_seq_dim,
                )
                new_v = torch.cat(
                    [self.v_slice(v, 0, start), self.v_slice(v, end, seq_len)],
                    dim=self.v_seq_dim,
                )
                _cache_set_kv(past_key_values, i, new_k, new_v)
            return past_key_values
        else:
            return [
                [
                    torch.cat(
                        [self.k_slice(k, 0, start), self.k_slice(k, end, seq_len)],
                        dim=self.k_seq_dim,
                    ),
                    torch.cat(
                        [self.v_slice(v, 0, start), self.v_slice(v, end, seq_len)],
                        dim=self.v_seq_dim,
                    ),
                ]
                for k, v in past_key_values
            ]

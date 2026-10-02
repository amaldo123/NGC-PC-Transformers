import jax.numpy as jnp

class KVCache:
    """
    Lightweight Key-Value Cache for fast autoregressive text generation in Transformers.

    How it works correctly:
    - On step 0 (prefill): the full context is passed. K and V for the full context
      are stored in the cache.
    - On step 1+ (decode): ONLY the single new token embedding is passed. The
      cached K and V from previous steps are concatenated with the new token's K and V,
      so attention is computed over the full history without recomputing past K/V.

    The key rule: reset() must be called before each new generation sequence.
    """
    def __init__(self, n_layers: int, max_seq_len: int = 1024):
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.step = 0
        self.reset()

    def reset(self):
        """Reset cached keys and values for all layers (call before each new generation)."""
        self.k_cache = [None] * self.n_layers
        self.v_cache = [None] * self.n_layers
        self.step = 0

    def update(self, layer_idx: int, k_new: jnp.ndarray, v_new: jnp.ndarray):
        """
        Store new k and v for this layer.

        On first call (step 0 / prefill): stores the full-context K and V.
        On subsequent calls (decode steps): concatenates only the NEW token's K and V
        to the existing cache.

        Args:
            layer_idx: Index of the transformer block / layer.
            k_new: Key tensor of shape (batch_size, n_heads, seq_len_new, d_head)
            v_new: Value tensor of shape (batch_size, n_heads, seq_len_new, d_head)

        Returns:
            (k_full, v_full): Full accumulated K and V for attention computation.
        """
        if self.k_cache[layer_idx] is None:
            # Prefill step: store the full context K and V
            self.k_cache[layer_idx] = k_new
            self.v_cache[layer_idx] = v_new
        else:
            # Decode step: only k_new/v_new is the NEW single token - append it
            k_cached = self.k_cache[layer_idx]
            v_cached = self.v_cache[layer_idx]

            # Enforce max cache length to avoid unbounded growth
            total_new = k_cached.shape[2] + k_new.shape[2]
            if total_new > self.max_seq_len:
                # Drop oldest tokens to stay within limit
                keep = self.max_seq_len - k_new.shape[2]
                k_cached = k_cached[:, :, -keep:, :]
                v_cached = v_cached[:, :, -keep:, :]

            self.k_cache[layer_idx] = jnp.concatenate([k_cached, k_new], axis=2)
            self.v_cache[layer_idx] = jnp.concatenate([v_cached, v_new], axis=2)

        return self.k_cache[layer_idx], self.v_cache[layer_idx]

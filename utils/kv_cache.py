import jax.numpy as jnp

class KVCache:
    """
    Lightweight Key-Value Cache for fast autoregressive text generation in Transformers.
    Stores cached Key (K) and Value (V) tensors per layer block.
    """
    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self.reset()

    def reset(self):
        """Reset cached keys and values for all layers."""
        self.k_cache = [None] * self.n_layers
        self.v_cache = [None] * self.n_layers

    def update(self, layer_idx: int, k_new: jnp.ndarray, v_new: jnp.ndarray):
        """
        Append new k and v tensors to cache for layer_idx along sequence axis (axis=2).
        
        Args:
            layer_idx: Index of the transformer block / layer.
            k_new: Key tensor of shape (batch_size, n_heads, seq_len_new, d_head)
            v_new: Value tensor of shape (batch_size, n_heads, seq_len_new, d_head)
            
        Returns:
            (k_cached, v_cached) containing past + new keys and values.
        """
        if self.k_cache[layer_idx] is None:
            self.k_cache[layer_idx] = k_new
            self.v_cache[layer_idx] = v_new
        else:
            self.k_cache[layer_idx] = jnp.concatenate([self.k_cache[layer_idx], k_new], axis=2)
            self.v_cache[layer_idx] = jnp.concatenate([self.v_cache[layer_idx], v_new], axis=2)
            
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

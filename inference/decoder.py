"""
Standalone Fast Inference Decoder for NGC-PC-Transformers.
Pure JAX / NumPy implementation. No ngclearn, no ngcsimlib, no model.py dependencies.
Faithfully reproduces the predictive coding Projection forward path (y_mu_inf)
with O(1) KV-cache decode steps and exact sliding-window semantics.
"""

import json
from pathlib import Path
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp


# ==============================================================================
# 1. Activation Functions & RateCell Semantics (ngclearn-verified)
# ==============================================================================

def identity(x):
    """Identity activation function."""
    return x


def gelu(x):
    """
    Exact ngclearn approximate GeLU:
    swish(x, beta=1.702) = x * sigmoid(1.702 * x)
    """
    return x * jax.nn.sigmoid(1.702 * x)


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    max_x = jnp.max(x, axis=axis, keepdims=True)
    exp_x = jnp.exp(x - max_x)
    return exp_x / jnp.sum(exp_x, axis=axis, keepdims=True)


def get_activation_fn(name: str):
    """Retrieve activation function by name from metadata."""
    name = (name or "identity").lower()
    if name == "gelu":
        return gelu
    elif name == "softmax":
        return softmax
    elif name == "relu":
        return jax.nn.relu
    elif name == "tanh":
        return jnp.tanh
    elif name == "identity":
        return identity
    else:
        return identity


def ratecell_step(j, tau_m: float = 0.0, act_fn=identity):
    """
    Computes a single step of RateCell dynamics from rest state (z=0) at dt=1.0.
    Matches ngclearn RateCell.advance_state(dt=1.0) exactly:
      - Stateless (tau_m <= 0): z = j, zF = act_fn(z)
      - Stateful (tau_m > 0):   z = j / tau_m, zF = act_fn(z)
    """
    if tau_m <= 0.0:
        z = j
    else:
        z = j / tau_m
    return act_fn(z)


# ==============================================================================
# 2. Rotary Position Embeddings (RoPE)
# ==============================================================================

def precompute_freqs_cis_real(d_head: int, seq_len: int, theta: float = 10000.0):
    """
    Precompute RoPE cosine and sine tables of shape (seq_len, d_head).
    Matches utils/rope_utils.py exactly.
    """
    freqs = 1.0 / (theta ** (jnp.arange(0, d_head, 2, dtype=jnp.float32) / d_head))
    t = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.outer(t, freqs)  # (seq_len, d_head // 2)

    cos = jnp.zeros((seq_len, d_head), dtype=jnp.float32)
    sin = jnp.zeros((seq_len, d_head), dtype=jnp.float32)

    cos = cos.at[:, 0::2].set(jnp.cos(angles))
    cos = cos.at[:, 1::2].set(jnp.cos(angles))
    sin = sin.at[:, 0::2].set(jnp.sin(angles))
    sin = sin.at[:, 1::2].set(jnp.sin(angles))

    return cos, sin


def rotate_half(x):
    """Rotates half of the hidden dimensions for real-valued RoPE."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    res = jnp.zeros_like(x)
    res = res.at[..., 0::2].set(-x2)
    res = res.at[..., 1::2].set(x1)
    return res


def apply_rope(q, k, cos, sin):
    """
    Apply rotary embeddings to query and key tensors.
    q, k expected shape: (B, H, S, d_head) or (B, H, 1, d_head)
    cos, sin shape: (1, 1, S, d_head) or (1, 1, 1, d_head)
    """
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


# ==============================================================================
# 3. Weight and Metadata Loading
# ==============================================================================

def load_weights(npz_path: str = "generation_weights.npz", meta_path: str = "generation_meta.json"):
    """
    Loads exported projection weights and architecture metadata.
    Returns:
      params: dictionary of JAX arrays (PyTree)
      meta: dictionary of hyperparameters and cell settings
    """
    npz_p = Path(npz_path)
    meta_p = Path(meta_path)

    if not npz_p.exists():
        raise FileNotFoundError(
            f"Weights file not found at '{npz_p}'.\n"
            f"Please run 'python export_generation_weights.py' first."
        )
    if not meta_p.exists():
        raise FileNotFoundError(f"Metadata file not found at '{meta_p}'.")

    with open(meta_p, "r", encoding="utf-8") as f:
        meta = json.load(f)

    raw_weights = np.load(str(npz_p))

    n_layers = meta["n_layers"]
    n_heads = meta["n_heads"]
    n_embed = meta["n_embed"]

    assert n_embed % n_heads == 0, f"n_embed ({n_embed}) must be divisible by n_heads ({n_heads})"
    d_head = n_embed // n_heads
    meta["d_head"] = d_head

    params = {
        "word_weights": jnp.array(raw_weights["word_weights"]),
        "pos_weights": jnp.array(raw_weights["pos_weights"]) if "pos_weights" in raw_weights else None,
        "layers": [],
        "W_out": jnp.array(raw_weights["W_out"]),
        "b_out": jnp.array(raw_weights["b_out"]),
    }

    # RoPE precomputed tables if enabled
    pos_encoding = meta.get("position_encoding", "positional")
    if pos_encoding == "rope":
        rope_theta = meta.get("rope_theta", 10000.0)
        cos, sin = precompute_freqs_cis_real(d_head, meta["seq_len"], theta=rope_theta)
        params["cos"] = cos
        params["sin"] = sin
    else:
        params["cos"] = None
        params["sin"] = None

    for i in range(n_layers):
        layer_dict = {
            "W_q": jnp.array(raw_weights[f"block{i}_W_q"]),
            "b_q": jnp.array(raw_weights[f"block{i}_b_q"]),
            "W_k": jnp.array(raw_weights[f"block{i}_W_k"]),
            "b_k": jnp.array(raw_weights[f"block{i}_b_k"]),
            "W_v": jnp.array(raw_weights[f"block{i}_W_v"]),
            "b_v": jnp.array(raw_weights[f"block{i}_b_v"]),
            "W_attn_out": jnp.array(raw_weights[f"block{i}_W_attn_out"]),
            "b_attn_out": jnp.array(raw_weights[f"block{i}_b_attn_out"]),
            "W_mlp1": jnp.array(raw_weights[f"block{i}_W_mlp1"]),
            "b_mlp1": jnp.array(raw_weights[f"block{i}_b_mlp1"]),
            "W_mlp2": jnp.array(raw_weights[f"block{i}_W_mlp2"]),
            "b_mlp2": jnp.array(raw_weights[f"block{i}_b_mlp2"]),
        }
        params["layers"].append(layer_dict)

    return params, meta


# ==============================================================================
# 4. Prefill Forward Pass (Full Window, Causal Attention, Fills KV Cache)
# ==============================================================================

@partial(jax.jit, static_argnames=["n_layers", "n_heads", "d_head", "seq_len", "use_rope"])
def _prefill_jit(
    tokens: jnp.ndarray,
    n_tokens: int,
    word_weights: jnp.ndarray,
    pos_weights: jnp.ndarray,
    layer_params: tuple,
    W_out: jnp.ndarray,
    b_out: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
    n_layers: int,
    n_heads: int,
    d_head: int,
    seq_len: int,
    use_rope: bool
):
    """
    JIT-compiled batched prefill over a padded window of seq_len tokens.
    Computes K/V for all positions, causal attention, and returns:
      logits: pre-softmax unnormalized logits at the last valid position (n_tokens - 1)
      k_cache: (n_layers, 1, n_heads, seq_len, d_head)
      v_cache: (n_layers, 1, n_heads, seq_len, d_head)
    """
    # 1. Embedding
    token_ids = tokens.reshape(-1).astype(jnp.int32)
    h = word_weights[token_ids]  # (seq_len, n_embed)

    if not use_rope and pos_weights is not None:
        h = h + pos_weights[:seq_len]

    # Precompute causal mask: True for valid past positions, False for future
    causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
    causal_mask = causal_mask[None, None, :, :]  # (1, 1, seq_len, seq_len)

    k_cache_list = []
    v_cache_list = []

    # 2. Iterate through transformer layers
    for i in range(n_layers):
        lp = layer_params[i]

        # q_qkv_Ratecell (tau_m=0, identity): zF = h
        z_qkv = h

        # Q, K, V linear projections
        Q = z_qkv @ lp["W_q"] + lp["b_q"]  # (seq_len, n_embed)
        K = z_qkv @ lp["W_k"] + lp["b_k"]
        V = z_qkv @ lp["W_v"] + lp["b_v"]

        # Reshape to multi-head: (1, n_heads, seq_len, d_head)
        q = Q.reshape(seq_len, n_heads, d_head).transpose([1, 0, 2])[None, ...]
        k = K.reshape(seq_len, n_heads, d_head).transpose([1, 0, 2])[None, ...]
        v = V.reshape(seq_len, n_heads, d_head).transpose([1, 0, 2])[None, ...]

        # Rotary position embeddings if enabled
        if use_rope:
            cos_b = cos[None, None, :seq_len, :]
            sin_b = sin[None, None, :seq_len, :]
            q, k = apply_rope(q, k, cos_b, sin_b)

        k_cache_list.append(k)
        v_cache_list.append(v)

        # Scaled dot-product attention
        scores = jnp.einsum("BHTE,BHSE->BHTS", q, k) / jnp.sqrt(d_head)
        scores = jnp.where(causal_mask, scores, -1e9)
        attn_probs = jax.nn.softmax(scores, axis=-1)

        # Attention output: (1, n_heads, seq_len, d_head) -> (seq_len, n_embed)
        attn_out = jnp.einsum("BHTS,BHSE->BHTE", attn_probs, v)
        attn_out = attn_out[0].transpose([1, 0, 2]).reshape(seq_len, -1)

        # Q_attn_out linear projection
        o_attn = attn_out @ lp["W_attn_out"] + lp["b_attn_out"]

        # q_mlp_Ratecell (tau_m=0, identity): zF = o_attn
        z_mlp = o_attn

        # Q_mlp1 linear projection
        m1 = z_mlp @ lp["W_mlp1"] + lp["b_mlp1"]

        # q_mlp2_Ratecell (tau_m=0, act_fx=gelu): zF = gelu(m1)
        z_mlp2 = gelu(m1)

        # Q_mlp2 linear projection
        h = z_mlp2 @ lp["W_mlp2"] + lp["b_mlp2"]

    # 3. Output Head: only the last valid token (index n_tokens - 1) is projected
    last_idx = jnp.clip(n_tokens - 1, 0, seq_len - 1)
    h_last = h[last_idx]  # (n_embed,)

    # q_out_Ratecell (tau_m=0, identity) -> Q_out linear projection
    logits = h_last @ W_out + b_out  # (vocab_size,)

    k_cache_arr = jnp.stack(k_cache_list, axis=0)  # (n_layers, 1, n_heads, seq_len, d_head)
    v_cache_arr = jnp.stack(v_cache_list, axis=0)

    return logits, k_cache_arr, v_cache_arr


def prefill(params: dict, tokens: jnp.ndarray, n: int, meta: dict = None):
    """
    Batched causal forward pass over a padded window of seq_len tokens.
    Only the last real row (n - 1) passes through the output head.
    Returns:
      logits: (vocab_size,) raw pre-softmax logits for the next token prediction
      cache: tuple of (k_cache, v_cache)
    """
    if meta is None:
        raise ValueError("meta dictionary must be provided to prefill.")

    seq_len = meta["seq_len"]
    n_layers = meta["n_layers"]
    n_heads = meta["n_heads"]
    d_head = meta["d_head"]
    use_rope = (meta.get("position_encoding", "positional") == "rope")

    tokens_padded = jnp.zeros((seq_len,), dtype=jnp.int32)
    n_clamped = min(int(n), seq_len)
    tokens_padded = tokens_padded.at[:n_clamped].set(tokens[:n_clamped])

    layer_params = tuple(params["layers"])

    logits, k_cache, v_cache = _prefill_jit(
        tokens=tokens_padded,
        n_tokens=n_clamped,
        word_weights=params["word_weights"],
        pos_weights=params["pos_weights"],
        layer_params=layer_params,
        W_out=params["W_out"],
        b_out=params["b_out"],
        cos=params["cos"],
        sin=params["sin"],
        n_layers=n_layers,
        n_heads=n_heads,
        d_head=d_head,
        seq_len=seq_len,
        use_rope=use_rope
    )

    return logits, (k_cache, v_cache)


# ==============================================================================
# 5. O(1) Incremental Decode Step with Fixed-Shape KV Cache
# ==============================================================================

@partial(jax.jit, static_argnames=["n_layers", "n_heads", "d_head", "seq_len", "use_rope"])
def _decode_step_jit(
    token_id: int,
    pos: int,
    k_cache: jnp.ndarray,
    v_cache: jnp.ndarray,
    word_weights: jnp.ndarray,
    pos_weights: jnp.ndarray,
    layer_params: tuple,
    W_out: jnp.ndarray,
    b_out: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
    n_layers: int,
    n_heads: int,
    d_head: int,
    seq_len: int,
    use_rope: bool
):
    """
    JIT-compiled incremental decode step for one token at position pos.
    Performs O(1) matmuls, writes new K/V with dynamic_update_slice,
    masks unfilled slots (> pos) with -1e9, and returns:
      logits: (vocab_size,)
      k_cache_updated: (n_layers, 1, n_heads, seq_len, d_head)
      v_cache_updated: (n_layers, 1, n_heads, seq_len, d_head)
    """
    # 1. Embedding lookup for current token
    h = word_weights[token_id]  # (n_embed,)
    if not use_rope and pos_weights is not None:
        h = h + pos_weights[pos]
    h = h[None, :]  # (1, n_embed)

    # Attention mask for single query at position `pos`:
    # Valid for all cache slots s <= pos; masked for s > pos
    slot_indices = jnp.arange(seq_len)
    valid_mask = slot_indices <= pos
    attn_mask = valid_mask[None, None, None, :]  # (1, 1, 1, seq_len)

    updated_k_layers = []
    updated_v_layers = []

    # 2. Layer-by-layer forward computation
    for i in range(n_layers):
        lp = layer_params[i]

        # q_qkv_Ratecell (tau_m=0, identity): zF = h
        z_qkv = h

        # Projections for the single token: (1, n_embed)
        q = z_qkv @ lp["W_q"] + lp["b_q"]
        k = z_qkv @ lp["W_k"] + lp["b_k"]
        v = z_qkv @ lp["W_v"] + lp["b_v"]

        # Reshape to (1, n_heads, 1, d_head)
        q = q.reshape(1, n_heads, d_head)[:, :, None, :]
        k = k.reshape(1, n_heads, d_head)[:, :, None, :]
        v = v.reshape(1, n_heads, d_head)[:, :, None, :]

        # RoPE rotation at position `pos`
        if use_rope:
            cos_p = cos[pos:pos + 1, :][None, None, :, :]  # (1, 1, 1, d_head)
            sin_p = sin[pos:pos + 1, :][None, None, :, :]
            q, k = apply_rope(q, k, cos_p, sin_p)

        # Update cache at slot `pos` using lax.dynamic_update_slice
        # k_cache[i]: (1, n_heads, seq_len, d_head)
        k_layer = k_cache[i]
        v_layer = v_cache[i]

        k_updated = jax.lax.dynamic_update_slice(k_layer, k, (0, 0, pos, 0))
        v_updated = jax.lax.dynamic_update_slice(v_layer, v, (0, 0, pos, 0))

        updated_k_layers.append(k_updated)
        updated_v_layers.append(v_updated)

        # Multi-Head Attention over all cached keys/values (1..seq_len)
        scores = jnp.einsum("BHTE,BHSE->BHTS", q, k_updated) / jnp.sqrt(d_head)
        scores = jnp.where(attn_mask, scores, -1e9)
        attn_probs = jax.nn.softmax(scores, axis=-1)

        # Context vector: (1, n_heads, 1, d_head) -> (1, n_embed)
        attn_out = jnp.einsum("BHTS,BHSE->BHTE", attn_probs, v_updated)
        attn_out = attn_out[0, :, 0, :].reshape(1, -1)

        # Q_attn_out
        o_attn = attn_out @ lp["W_attn_out"] + lp["b_attn_out"]

        # q_mlp_Ratecell -> Q_mlp1
        m1 = o_attn @ lp["W_mlp1"] + lp["b_mlp1"]

        # q_mlp2_Ratecell (gelu)
        z_mlp2 = gelu(m1)

        # Q_mlp2
        h = z_mlp2 @ lp["W_mlp2"] + lp["b_mlp2"]

    # 3. Output Head: q_out_Ratecell -> Q_out
    logits = h[0] @ W_out + b_out  # (vocab_size,)

    k_cache_out = jnp.stack(updated_k_layers, axis=0)
    v_cache_out = jnp.stack(updated_v_layers, axis=0)

    return logits, k_cache_out, v_cache_out


def decode_step(params: dict, cache: tuple, token: int, pos: int, meta: dict = None):
    """
    Executes an incremental single-token forward pass with O(1) matmuls.
    Returns:
      logits: (vocab_size,)
      updated_cache: (k_cache, v_cache)
    """
    if meta is None:
        raise ValueError("meta dictionary must be provided to decode_step.")

    k_cache, v_cache = cache
    seq_len = meta["seq_len"]
    n_layers = meta["n_layers"]
    n_heads = meta["n_heads"]
    d_head = meta["d_head"]
    use_rope = (meta.get("position_encoding", "positional") == "rope")

    token_id = int(token)
    pos_idx = int(pos)

    layer_params = tuple(params["layers"])

    logits, k_cache_new, v_cache_new = _decode_step_jit(
        token_id=token_id,
        pos=pos_idx,
        k_cache=k_cache,
        v_cache=v_cache,
        word_weights=params["word_weights"],
        pos_weights=params["pos_weights"],
        layer_params=layer_params,
        W_out=params["W_out"],
        b_out=params["b_out"],
        cos=params["cos"],
        sin=params["sin"],
        n_layers=n_layers,
        n_heads=n_heads,
        d_head=d_head,
        seq_len=seq_len,
        use_rope=use_rope
    )

    return logits, (k_cache_new, v_cache_new)


# ==============================================================================
# 6. Categorical Sampler
# ==============================================================================

def sample(logits: jnp.ndarray, key=None, temperature: float = 1.0, top_k: int = 0):
    """
    Applies temperature scaling and top-k filtering directly to LOGITS (or log-probs),
    followed by jax.random.categorical sampling.
    If key is None or temperature <= 0.0, performs deterministic greedy decoding (argmax).
    """
    if key is None or temperature <= 0.0:
        return int(jnp.argmax(logits))

    scaled_logits = logits / max(temperature, 1e-6)

    if top_k is not None and top_k > 0:
        k_val = min(int(top_k), logits.shape[-1])
        top_vals, top_idx = jax.lax.top_k(scaled_logits, k=k_val)
        choice = jax.random.categorical(key, top_vals)
        return int(top_idx[choice])
    else:
        choice = jax.random.categorical(key, scaled_logits)
        return int(choice)


# ==============================================================================
# 7. Autoregressive Generator (Original Sliding-Window Semantics)
# ==============================================================================

def generate(
    params: dict,
    meta: dict,
    tokenizer,
    prompt: str = "",
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 0,
    key=None,
    pad_token_id: int = None,
    use_ring_buffer: bool = False
):
    """
    Full text generation reproducing original model.process sliding-window semantics:
      1. While total tokens <= seq_len: uses O(1) decode_step with KV cache.
      2. Once context exceeds seq_len: re-runs prefill on the last seq_len tokens every step
         (positions restart at 0..seq_len-1), because layer 2+ hidden states change
         when the oldest token leaves the window.
      3. If use_ring_buffer=True (only permitted when position_encoding='rope'):
         maintains a rolling ring-buffer without full prefill re-evaluations.
    """
    seq_len = meta["seq_len"]
    pos_encoding = meta.get("position_encoding", "positional")

    if use_ring_buffer and pos_encoding != "rope":
        raise ValueError("Ring-buffer mode is only valid when position_encoding='rope'.")

    # 1. Resolve start token and pad token
    if pad_token_id is None:
        if hasattr(tokenizer, "tokenizer") and tokenizer.tokenizer is not None:
            pad_token_id = tokenizer.tokenizer.token_to_id("<pad>")
        elif hasattr(tokenizer, "pad_token_id"):
            pad_token_id = tokenizer.pad_token_id
        else:
            pad_token_id = 0
    if pad_token_id is None:
        pad_token_id = 0

    start_token_id = None
    if hasattr(tokenizer, "tokenizer") and tokenizer.tokenizer is not None:
        start_token_id = tokenizer.tokenizer.token_to_id("<bos>")
    elif hasattr(tokenizer, "bos_token_id"):
        start_token_id = tokenizer.bos_token_id
    if start_token_id is None:
        start_token_id = pad_token_id

    # 2. Tokenize prompt or initialize with <bos>
    if prompt:
        encoded = tokenizer.encode(prompt)
        input_ids = encoded.ids if hasattr(encoded, "ids") else list(encoded)
        if len(input_ids) == 0:
            input_ids = [start_token_id]
    else:
        input_ids = [start_token_id]

    current_tokens = list(input_ids)
    rng_key = key

    # 3. Initial prefill over prompt tokens
    n_prompt = min(len(current_tokens), seq_len)
    prompt_window = jnp.array(current_tokens[-n_prompt:], dtype=jnp.int32)
    logits, cache = prefill(params, prompt_window, n=n_prompt, meta=meta)

    # 4. Autoregressive loop
    for _ in range(max_new_tokens):
        # Sample next token
        if rng_key is not None:
            rng_key, subkey = jax.random.split(rng_key)
        else:
            subkey = None

        next_token = sample(logits, key=subkey, temperature=temperature, top_k=top_k)
        current_tokens.append(next_token)

        cur_len = len(current_tokens)

        if cur_len <= seq_len:
            # Within window: incremental O(1) decode_step
            pos = cur_len - 1
            logits, cache = decode_step(params, cache, token=next_token, pos=pos, meta=meta)
        else:
            # Exceeded seq_len: exact sliding-window re-prefill
            window = jnp.array(current_tokens[-seq_len:], dtype=jnp.int32)
            logits, cache = prefill(params, window, n=seq_len, meta=meta)

    return tokenizer.decode(current_tokens)

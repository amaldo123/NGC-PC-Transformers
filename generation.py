import os
import sys

# MUST BE SET BEFORE ANY JAX/TENSORFLOW IMPORTS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0 --xla_gpu_strict_conv_algorithm_picker=false --xla_cpu_use_thunk_runtime=false'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['JAX_PLATFORM_NAME'] = 'gpu'
os.environ['JAX_ENABLE_X64'] = 'False'

# Redirect stderr to suppress XLA warnings
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

import warnings
warnings.filterwarnings('ignore')

# Now import JAX and other libraries
import jax
jax.config.update('jax_platform_name', 'gpu')
jax.config.update('jax_log_compiles', False)

# Restore stderr after JAX initialization
sys.stderr = stderr

from pathlib import Path
_REPO_ROOT = str(Path(__file__).resolve().parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from model import NGCTransformer
import jax.numpy as jnp
import numpy as np
from config import Config as config
from data_preprocess.data_loader import DataLoader
from data_preprocess.tokenizer import get_tokenizer, BPETokenizer, CharacterTokenizer
from pathlib import Path
import re
import textwrap
import time


# ---------------------------------------------------------------------------
# KV Cache
# ---------------------------------------------------------------------------

class KVCache:
    """
    Stores accumulated Key and Value tensors for every transformer layer.

    Shape per layer:
        k_cache[layer]:  (1, cur_len, n_embed)   -- grows by one per step
        v_cache[layer]:  (1, cur_len, n_embed)

    Usage pattern (inside the generation loop):
        1. Run the full forward pass for step 0 -> cache K/V for all positions.
        2. For every subsequent step t, inject the cached K/V into attn_block,
           run the forward pass, then append the fresh K/V at position t.
    """

    def __init__(self, n_layers: int, seq_len: int, n_embed: int):
        self.n_layers = n_layers
        self.seq_len  = seq_len
        self.n_embed  = n_embed
        self.k = [None] * n_layers   # list of jnp arrays or None
        self.v = [None] * n_layers

    def store(self, layer: int, k_new, v_new):
        """Append new K/V slice (shape (1, 1, n_embed)) to the cache."""
        if self.k[layer] is None:
            self.k[layer] = k_new
            self.v[layer] = v_new
        else:
            self.k[layer] = jnp.concatenate([self.k[layer], k_new], axis=1)
            self.v[layer] = jnp.concatenate([self.v[layer], v_new], axis=1)

    def get_kv(self, layer: int):
        """Return the full accumulated (K, V) arrays for this layer."""
        return self.k[layer], self.v[layer]

    def reset(self):
        self.k = [None] * self.n_layers
        self.v = [None] * self.n_layers


# ---------------------------------------------------------------------------
# KV-cache helpers
# ---------------------------------------------------------------------------

def _extract_layer_kv(model, layer_idx: int, pos: int):
    """
    Read K and V at token position pos from block layer_idx.
    attn_block.inputs_k/v shape: (1, seq_len, n_embed) -> slice to (1,1,n_embed).
    """
    block  = model.blocks[layer_idx]
    k_full = block.attention.attn_block.inputs_k.get()
    v_full = block.attention.attn_block.inputs_v.get()
    return k_full[:, pos : pos + 1, :], v_full[:, pos : pos + 1, :]


def _inject_cached_kv(model, layer_idx: int, k_cached, v_cached, seq_len: int):
    """
    Write cached K/V (padded to seq_len) into attn_block so the attention
    computation sees all past positions on the next forward pass.
    """
    cur_len = k_cached.shape[1]
    if cur_len < seq_len:
        pad  = seq_len - cur_len
        k_in = jnp.pad(k_cached, ((0, 0), (0, pad), (0, 0)))
        v_in = jnp.pad(v_cached, ((0, 0), (0, pad), (0, 0)))
    else:
        k_in = k_cached[:, -seq_len:, :]
        v_in = v_cached[:, -seq_len:, :]
    block = model.blocks[layer_idx]
    block.attention.attn_block.inputs_k.set(k_in)
    block.attention.attn_block.inputs_v.set(v_in)



def generate_text(
    model,
    tokenizer,
    max_new_tokens: int = 200,
    seq_len: int = config.seq_len,
    temperature: float = 1.0,
    top_k: int = 0,
    key=None,
    pad_token_id: int = None,
    use_kv_cache: bool = None,   # None -> read from Config.use_kv_cache
):
    """
    Generate text using the model and provided tokenizer.
    Works with both custom BPE, character and tiktoken backends.

    KV caching:
        Enable  -> pass use_kv_cache=True  OR set Config.use_kv_cache = True
        Disable -> pass use_kv_cache=False OR set Config.use_kv_cache = False
    """
    if use_kv_cache is None:
        use_kv_cache = getattr(config, "use_kv_cache", False)
    if use_kv_cache:
        return _generate_with_kv_cache(
            model, tokenizer, max_new_tokens, seq_len,
            temperature, top_k, key, pad_token_id
        )
    if pad_token_id is None:
        if isinstance(tokenizer, BPETokenizer) and tokenizer.tokenizer is not None:
            pad_token_id = tokenizer.tokenizer.token_to_id("<pad>")
        elif isinstance(tokenizer, CharacterTokenizer):
            pad_token_id = tokenizer.pad_token_id
        elif hasattr(tokenizer, "_enc") and hasattr(tokenizer._enc, "eot_token"):
            pad_token_id = tokenizer._enc.eot_token
        else:
            pad_token_id = 0

    start_token_id = None
    if isinstance(tokenizer, BPETokenizer) and tokenizer.tokenizer is not None:
        start_token_id = tokenizer.tokenizer.token_to_id("<bos>")
    elif isinstance(tokenizer, CharacterTokenizer):
        start_token_id = tokenizer.bos_token_id
    if start_token_id is None:
        start_token_id = pad_token_id

    # Initialize sequence with start token ID
    current_tokens = jnp.array([[start_token_id]], dtype=jnp.int32)
    current_key = key

    for _ in range(max_new_tokens):
        # Truncate context to fit model's seq_len
        if current_tokens.shape[1] > config.seq_len:
            input_seq = current_tokens[:, -config.seq_len:]
        else:
            input_seq = current_tokens

        # Pad to exactly seq_len if needed
        if input_seq.shape[1] < config.seq_len:
            pad_len = config.seq_len - input_seq.shape[1]
            input_seq = jnp.pad(input_seq, ((0, 0), (0, pad_len)), constant_values=pad_token_id)
        
        # Forward pass (no target clamping during inference)
        dummy_target = jnp.zeros((config.batch_size * config.seq_len, config.vocab_size))

        # Forward pass

        y_mu_inf, y_mu, _ = model.process(input_seq, dummy_target, adapt_synapses=False)
        logits = y_mu_inf.reshape(config.batch_size, config.seq_len, config.vocab_size)

        # Get logits for the last *real* token (excluding padding)
        if current_tokens.shape[1] > config.seq_len:
            last_pos = config.seq_len - 1
        else:
            last_pos = current_tokens.shape[1] - 1
        next_logits = logits[0, last_pos, :] / temperature

        # Sample or take argmax
        if current_key is not None:
            if top_k is not None and top_k > 0:
                top_k = min(top_k, config.vocab_size)
                top_vals, top_idx = jax.lax.top_k(next_logits, k=top_k)
                probs = jax.nn.softmax(top_vals)
                current_key, subkey = jax.random.split(current_key)
                choice = jax.random.choice(subkey, a=top_k, p=probs)
                next_token = top_idx[choice]
            else:
                probs = jax.nn.softmax(next_logits)
                current_key, subkey = jax.random.split(current_key)
                next_token = jax.random.choice(subkey, a=config.vocab_size, p=probs)
        else:
            next_token = jnp.argmax(next_logits)

        # Append new token
        current_tokens = jnp.concatenate([current_tokens, next_token[None, None]], axis=1)

    # Decode generated IDs back to text
    generated_ids = current_tokens[0].tolist()
    return tokenizer.decode(generated_ids)


# ---------------------------------------------------------------------------
# KV-cache generation path
# ---------------------------------------------------------------------------

def _generate_with_kv_cache(model, tokenizer, max_new_tokens, seq_len,
                             temperature, top_k, key, pad_token_id):
    """
    Generate text using KV caching.

    Step 0  - seed the cache from the first full forward pass.
    Step t  - inject the accumulated cache, run the forward pass, then
              append the fresh K/V at the new token position.
    """
    # Resolve pad/start tokens (same logic as the no-cache path above)
    if pad_token_id is None:
        if isinstance(tokenizer, BPETokenizer) and tokenizer.tokenizer is not None:
            pad_token_id = tokenizer.tokenizer.token_to_id("<pad>")
        elif isinstance(tokenizer, CharacterTokenizer):
            pad_token_id = tokenizer.pad_token_id
        elif hasattr(tokenizer, "_enc") and hasattr(tokenizer._enc, "eot_token"):
            pad_token_id = tokenizer._enc.eot_token
        else:
            pad_token_id = 0
    start_token_id = None
    if isinstance(tokenizer, BPETokenizer) and tokenizer.tokenizer is not None:
        start_token_id = tokenizer.tokenizer.token_to_id("<bos>")
    elif isinstance(tokenizer, CharacterTokenizer):
        start_token_id = tokenizer.bos_token_id
    if start_token_id is None:
        start_token_id = pad_token_id

    cache  = KVCache(model.n_layers, seq_len, model.n_embed)
    tokens = jnp.array([[start_token_id]], dtype=jnp.int32)
    current_key  = key
    dummy_target = jnp.zeros((config.batch_size * config.seq_len, config.vocab_size))

    for step in range(max_new_tokens):
        cur_len = tokens.shape[1]

        # Build padded input
        if tokens.shape[1] > config.seq_len:
            input_seq = tokens[:, -config.seq_len:]
        else:
            input_seq = tokens
        if input_seq.shape[1] < config.seq_len:
            pad_len   = config.seq_len - input_seq.shape[1]
            input_seq = jnp.pad(input_seq, ((0, 0), (0, pad_len)),
                                constant_values=pad_token_id)

        # Inject previously cached K/V before the forward pass
        if step > 0:
            for layer_idx in range(model.n_layers):
                k_c, v_c = cache.get_kv(layer_idx)
                _inject_cached_kv(model, layer_idx, k_c, v_c, seq_len)

        y_mu_inf, _, _ = model.process(input_seq, dummy_target,
                                       adapt_synapses=False)

        # Update cache with K/V from the position just computed
        new_pos = min(cur_len - 1, config.seq_len - 1)
        for layer_idx in range(model.n_layers):
            if step == 0:
                k_all = model.blocks[layer_idx].attention.attn_block.inputs_k.get()
                v_all = model.blocks[layer_idx].attention.attn_block.inputs_v.get()
                cache.k[layer_idx] = k_all[:, :cur_len, :]
                cache.v[layer_idx] = v_all[:, :cur_len, :]
            else:
                k_new, v_new = _extract_layer_kv(model, layer_idx, new_pos)
                cache.store(layer_idx, k_new, v_new)

        # Sample next token
        logits   = y_mu_inf.reshape(config.batch_size, config.seq_len, config.vocab_size)
        last_pos = min(cur_len - 1, config.seq_len - 1)
        next_logits = logits[0, last_pos, :] / temperature

        if current_key is not None:
            if top_k is not None and top_k > 0:
                top_k_  = min(top_k, config.vocab_size)
                top_vals, top_idx = jax.lax.top_k(next_logits, k=top_k_)
                probs = jax.nn.softmax(top_vals)
                current_key, subkey = jax.random.split(current_key)
                choice     = jax.random.choice(subkey, a=top_k_, p=probs)
                next_token = top_idx[choice]
            else:
                probs = jax.nn.softmax(next_logits)
                current_key, subkey = jax.random.split(current_key)
                next_token = jax.random.choice(subkey, a=config.vocab_size, p=probs)
        else:
            next_token = jnp.argmax(next_logits)

        tokens = jnp.concatenate([tokens, next_token[None, None]], axis=1)

    return tokenizer.decode(tokens[0].tolist())


# Initialize the model and tokenizer only when run as a script
if __name__ == "__main__":
    # Initialize the model
    dkey = jax.random.PRNGKey(0)
    model = NGCTransformer(
        dkey, 
        batch_size=config.batch_size,
        seq_len=config.seq_len, 
        n_embed=config.n_embed, 
        vocab_size=config.vocab_size, 
        n_layers=config.n_layers, 
        n_heads=config.n_heads,
        T=config.n_iter, 
        dt=1., 
        tau_m=config.tau_m, 
        act_fx=config.act_fx, 
        eta=config.eta, 
        dropout_rate=config.dropout_rate, 
        exp_dir="exp",
        loadDir="exp", # Ensure model is loaded from trained exp/ directory
        pos_learnable=config.pos_learnable, 
        optim_type=config.optim_type, 
        wub=config.wub, 
        wlb=config.wlb, 
        model_name="ngc_transformer",
        generate= True
    )

    # Optional: add custom weight stats here if needed

    tokenizer = get_tokenizer(config)

    if isinstance(tokenizer, BPETokenizer) and tokenizer.tokenizer is None:
        vocab_file = getattr(config, "tokenizer_vocab_file", None)
        if vocab_file is None:
            # default_path = Path(__file__).parent / "data_preprocess" / "outputs" / "tokenizer" / "bpe_tokenizer.json"
            from data_preprocess.datasets_registry import prepare_dataset
            _, output_dir = prepare_dataset(config.dataset)
            default_path = output_dir / "tokenizer" / "bpe_tokenizer.json"
            if default_path.exists():
                vocab_file = str(default_path)
                print(f"Auto-loading BPE tokenizer from default path: {vocab_file}")

        # Attempt to load
        if vocab_file and Path(vocab_file).exists():
            tokenizer.load_tokenizer(vocab_file)
            print(f"Loaded BPE tokenizer (vocab size: {tokenizer.get_vocab_size()})")
        else:
            raise RuntimeError(
                "BPE tokenizer not trained or loaded!\n\n"
            )

    rng = jax.random.PRNGKey(0)
    rng, key_1 = jax.random.split(rng)
    rng, key_2 = jax.random.split(rng)

    MAX_TOKENS  = 200
    TEMPERATURE = 0.8
    TOP_K       = 50

    print("\n" + "=" * 60)
    print("BENCHMARK: No Cache  vs  KV Cache")
    print("=" * 60)

    # --- No-cache run (original behaviour) ---
    t0 = time.perf_counter()
    out_no_cache = generate_text(
        model, tokenizer,
        max_new_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        key=key_1,
        use_kv_cache=False,
    )
    t_no_cache = time.perf_counter() - t0

    # --- KV-cache run ---
    t0 = time.perf_counter()
    out_kv_cache = generate_text(
        model, tokenizer,
        max_new_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        key=key_2,
        use_kv_cache=True,
    )
    t_kv_cache = time.perf_counter() - t0

    # --- Results ---
    speedup = t_no_cache / t_kv_cache if t_kv_cache > 0 else float("inf")
    print(f"\n[No Cache]  {t_no_cache:.2f}s  ({MAX_TOKENS / t_no_cache:.1f} tok/s)")
    print(f"[KV Cache]  {t_kv_cache:.2f}s  ({MAX_TOKENS / t_kv_cache:.1f} tok/s)")
    print(f"Speedup:    {speedup:.2f}x")

    print("\n-- No-Cache Output " + "-" * 42)
    print(out_no_cache)
    print("\n-- KV-Cache Output " + "-" * 42)
    print(out_kv_cache)

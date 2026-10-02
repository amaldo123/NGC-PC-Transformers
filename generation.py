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


def generate_text(
    model,
    tokenizer,
    max_new_tokens: int = 200,
    seq_len: int = config.seq_len,
    temperature: float = 1.0,
    top_k: int = 0,
    key=None,
    pad_token_id: int = None,
    use_kv_cache: bool = True
):
    """
    Generate text using the model and provided tokenizer.
    Supports fast KV Caching during generation.
    """
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

    bs = getattr(model, 'batch_size', config.batch_size)
    vs = getattr(model, 'vocab_size', config.vocab_size)
    sl = getattr(model, 'seq_len', config.seq_len)

    if use_kv_cache and hasattr(model, 'enable_kv_cache'):
        model.enable_kv_cache()

    try:
        for step in range(max_new_tokens):

            if use_kv_cache and step > 0:
                # ── DECODE STEP: feed only the last token ──────────────────────
                # The KV cache already holds all past K and V.
                # We only need to compute K/V for the ONE new token, so we pass
                # a single-token input (still padded to seq_len for model compat,
                # but the new token is placed at position 0 and the cache handles
                # the full context history).
                #
                # NOTE: Because this model's architecture requires a fixed
                # (batch_size, seq_len) shaped input, we still pad to seq_len but
                # only the first position (the new token) carries real content.
                # The KV cache concatenates this new token's K/V onto the past,
                # so attention still covers the full generation history.
                last_token = current_tokens[:, -1:]  # shape (1, 1)
                pad_len = sl - 1
                input_seq = jnp.pad(last_token, ((0, 0), (0, pad_len)),
                                    constant_values=pad_token_id)
                # We only read logits at position 0 (the new token's position)
                decode_pos = 0
            else:
                # ── PREFILL STEP (step 0): feed the full context ───────────────
                if current_tokens.shape[1] > sl:
                    input_seq = current_tokens[:, -sl:]
                else:
                    input_seq = current_tokens

                if input_seq.shape[1] < sl:
                    pad_len = sl - input_seq.shape[1]
                    input_seq = jnp.pad(input_seq, ((0, 0), (0, pad_len)),
                                        constant_values=pad_token_id)

                # Read from last *real* token position
                if current_tokens.shape[1] > sl:
                    decode_pos = sl - 1
                else:
                    decode_pos = current_tokens.shape[1] - 1

            dummy_target = jnp.zeros((bs * sl, vs))

            # Forward pass
            y_mu_inf, _, _ = model.process(input_seq, dummy_target, adapt_synapses=False)
            logits = y_mu_inf.reshape(bs, sl, vs)

            next_logits = logits[0, decode_pos, :] / temperature

            # Sample or take argmax
            if current_key is not None:
                if top_k is not None and top_k > 0:
                    top_k_val = min(top_k, vs)
                    top_vals, top_idx = jax.lax.top_k(next_logits, k=top_k_val)
                    probs = jax.nn.softmax(top_vals)
                    current_key, subkey = jax.random.split(current_key)
                    choice = jax.random.choice(subkey, a=top_k_val, p=probs)
                    next_token = top_idx[choice]
                else:
                    probs = jax.nn.softmax(next_logits)
                    current_key, subkey = jax.random.split(current_key)
                    next_token = jax.random.choice(subkey, a=vs, p=probs)
            else:
                next_token = jnp.argmax(next_logits)

            # Append new token
            current_tokens = jnp.concatenate([current_tokens, next_token[None, None]], axis=1)

    finally:
        if use_kv_cache and hasattr(model, 'disable_kv_cache'):
            model.disable_kv_cache()

    # Decode generated IDs back to text
    generated_ids = current_tokens[0].tolist()
    return tokenizer.decode(generated_ids)


def find_checkpoint_dir(model_name="ngc_transformer"):
    candidate = Path("exp")
    if (candidate / model_name / "contextData.json").exists():
        return str(candidate)
    return None


# Initialize the model and tokenizer only when run as a script
if __name__ == "__main__":
    load_dir = find_checkpoint_dir("ngc_transformer")
    if load_dir is None:
        print("Note: No saved checkpoint found at exp/ngc_transformer. Generating with initial model weights. Run 'python train.py' first to train the model.")
    else:
        print(f"Loading trained checkpoint from {load_dir}...")

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
        loadDir=load_dir,
        pos_learnable=config.pos_learnable, 
        optim_type=config.optim_type, 
        wub=config.wub, 
        wlb=config.wlb, 
        model_name="ngc_transformer",
        generate= True
    )

    tokenizer = get_tokenizer(config)

    if isinstance(tokenizer, BPETokenizer) and tokenizer.tokenizer is None:
        vocab_file = getattr(config, "tokenizer_vocab_file", None)
        if vocab_file is None:
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

    import time
    seed_key = jax.random.PRNGKey(config.SEED)
    use_kv_cache = getattr(config, "use_kv_cache", True)

    cache_label = "WITH KV Cache" if use_kv_cache else "WITHOUT KV Cache"
    print(f"\n--- Generating {cache_label} ---")

    t0 = time.time()
    generated_text = generate_text(
        model, tokenizer,
        max_new_tokens=200,
        temperature=0.8,
        top_k=50,
        key=seed_key,
        use_kv_cache=use_kv_cache
    )
    elapsed = time.time() - t0

    print(f"\nGeneration time : {elapsed:.3f} seconds")
    print(f"KV Cache        : {'Enabled' if use_kv_cache else 'Disabled'}")
    print("\nGENERATED TEXT:")
    print(generated_text)

import os
import sys

# MUST BE SET BEFORE ANY JAX/TENSORFLOW IMPORTS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0 --xla_gpu_strict_conv_algorithm_picker=false --xla_cpu_use_thunk_runtime=false'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ.setdefault('JAX_PLATFORM_NAME', 'gpu')
os.environ['JAX_ENABLE_X64'] = 'False'

# Redirect stderr to suppress XLA warnings
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

import warnings
warnings.filterwarnings('ignore')

# Now import JAX and other libraries
import jax
try:
    jax.devices(os.environ.get('JAX_PLATFORM_NAME', 'gpu'))
    jax.config.update('jax_platform_name', os.environ.get('JAX_PLATFORM_NAME', 'gpu'))
except Exception:
    jax.config.update('jax_platform_name', 'cpu')
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
from utils.fast_decode import extract_params, generate as fast_generate
import re
import textwrap
import time


def generate_text(
    model,
    tokenizer,
    max_new_tokens: int = 200,
    seq_len: int = config.seq_len,
    temperature: float = 1.0,
    top_k: int = 0,
    key=None,
    pad_token_id: int = None,
    use_kv_cache: bool = None
):
    """
    Generate text using the model and provided tokenizer.
    Supports fast KV Caching during generation.
    """
    if use_kv_cache is None:
        use_kv_cache = getattr(config, "use_kv_cache", True)

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

    # Fast KV cache decoder path (Appendix A / B)
    if use_kv_cache:
        params = extract_params(model)
        sl = getattr(model, 'seq_len', config.seq_len)
        nh = getattr(model, 'n_heads', config.n_heads)
        ids = fast_generate(
            params,
            start_token_id,
            max_new_tokens,
            sl,
            nh,
            temperature=temperature,
            top_k=top_k,
            key=key,
            pad_id=pad_token_id,
        )
        return tokenizer.decode(ids)

    # Legacy full-context loop fallback (use_kv_cache=False)
    current_tokens = jnp.array([[start_token_id]], dtype=jnp.int32)
    current_key = key

    sl = getattr(model, 'seq_len', config.seq_len)
    bs = getattr(model, 'batch_size', config.batch_size)
    vs = getattr(model, 'vocab_size', config.vocab_size)

    for step in range(max_new_tokens):
        if current_tokens.shape[1] > sl:
            input_seq = current_tokens[:, -sl:]
        else:
            input_seq = current_tokens

        # Pad to exactly seq_len if needed
        if input_seq.shape[1] < sl:
            pad_len = sl - input_seq.shape[1]
            input_seq = jnp.pad(input_seq, ((0, 0), (0, pad_len)), constant_values=pad_token_id)
        
        cur_seq_len = sl
        dummy_target = jnp.zeros((bs * cur_seq_len, vs))

        # Forward pass
        y_mu_inf, y_mu, _ = model.process(input_seq, dummy_target, adapt_synapses=False)
        logits = y_mu_inf.reshape(bs, cur_seq_len, vs)

        if current_tokens.shape[1] > sl:
            last_pos = sl - 1
        else:
            last_pos = current_tokens.shape[1] - 1

        # Sampling fix (F8): y_mu_inf holds softmax probabilities, so take log before temperature
        next_logits = jnp.log(logits[0, last_pos, :] + 1e-9) / temperature

        # Sample or take argmax
        if current_key is not None:
            if top_k is not None and top_k > 0:
                top_k_min = min(top_k, vs)
                top_vals, top_idx = jax.lax.top_k(next_logits, k=top_k_min)
                probs = jax.nn.softmax(top_vals)
                current_key, subkey = jax.random.split(current_key)
                choice = jax.random.choice(subkey, a=top_k_min, p=probs)
                next_token = top_idx[choice]
            else:
                probs = jax.nn.softmax(next_logits)
                current_key, subkey = jax.random.split(current_key)
                next_token = jax.random.choice(subkey, a=vs, p=probs)
        else:
            next_token = jnp.argmax(next_logits)

        # Append new token
        current_tokens = jnp.concatenate([current_tokens, next_token[None, None]], axis=1)

    # Decode generated IDs back to text
    generated_ids = current_tokens[0].tolist()
    return tokenizer.decode(generated_ids)


# Initialize the model and tokenizer only when run as a script
if __name__ == "__main__":
    # Initialize the model with dropout_rate=0.0 (eval mode for generation)
    dkey = jax.random.PRNGKey(0)
    load_dir = "exp" if (Path("exp").exists() and any(Path("exp").iterdir())) else None
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
        dropout_rate=0.0, 
        exp_dir="exp",
        loadDir=load_dir,
        pos_learnable=config.pos_learnable, 
        optim_type=config.optim_type, 
        wub=config.wub, 
        wlb=config.wlb, 
        model_name="ngc_transformer",
        generate=True
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

    seed_key = jax.random.PRNGKey(config.SEED)
    use_kv_cache = getattr(config, "use_kv_cache", True)

    path = "fast_decode (KV cache)" if use_kv_cache else "legacy full-context loop"
    print(f"\n--- Generating with {path} ---")

    max_new_tokens = 200
    temperature = 0.8
    top_k = 50

    # Warm-up run (excluded from timing)
    print("Warming up JIT...")
    _ = generate_text(
        model, tokenizer,
        max_new_tokens=10,
        temperature=temperature,
        top_k=top_k,
        key=seed_key,
        use_kv_cache=use_kv_cache
    )

    num_runs = 3
    times = []
    tokens_per_sec = []
    last_text = ""
    for r in range(num_runs):
        t0 = time.perf_counter()
        last_text = generate_text(
            model, tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            key=seed_key,
            use_kv_cache=use_kv_cache
        )
        elapsed = time.perf_counter() - t0
        tps = max_new_tokens / elapsed
        times.append(elapsed)
        tokens_per_sec.append(tps)
        print(f"Run {r + 1}: {elapsed:.3f} s ({tps:.2f} tokens/sec)")

    median_tps = float(np.median(tokens_per_sec))
    print(f"\nPath            : {path}")
    print(f"Tokens/sec      : {median_tps:.2f} tokens/sec (median of {num_runs} runs)")
    print(f"Latency spread  : min={min(times):.3f}s, max={max(times):.3f}s")
    print("\nGENERATED TEXT:")
    print(last_text)

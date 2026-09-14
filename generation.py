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

    if use_kv_cache and hasattr(model, 'enable_kv_cache'):
        model.enable_kv_cache()

    try:
        for step in range(max_new_tokens):
            if current_tokens.shape[1] > config.seq_len:
                input_seq = current_tokens[:, -config.seq_len:]
            else:
                input_seq = current_tokens

            # Pad to exactly seq_len if needed
            if input_seq.shape[1] < config.seq_len:
                pad_len = config.seq_len - input_seq.shape[1]
                input_seq = jnp.pad(input_seq, ((0, 0), (0, pad_len)), constant_values=pad_token_id)
            
            bs = getattr(model, 'batch_size', config.batch_size)
            vs = getattr(model, 'vocab_size', config.vocab_size)
            sl = getattr(model, 'seq_len', config.seq_len)

            cur_seq_len = sl
            dummy_target = jnp.zeros((bs * cur_seq_len, vs))

            # Forward pass
            y_mu_inf, y_mu, _ = model.process(input_seq, dummy_target, adapt_synapses=False)
            logits = y_mu_inf.reshape(bs, cur_seq_len, vs)

            if current_tokens.shape[1] > sl:
                last_pos = sl - 1
            else:
                last_pos = current_tokens.shape[1] - 1

            next_logits = logits[0, last_pos, :] / temperature



            # Sample or take argmax
            if current_key is not None:
                if top_k is not None and top_k > 0:
                    top_k = min(top_k, vs)
                    top_vals, top_idx = jax.lax.top_k(next_logits, k=top_k)
                    probs = jax.nn.softmax(top_vals)
                    current_key, subkey = jax.random.split(current_key)
                    choice = jax.random.choice(subkey, a=top_k, p=probs)
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
    candidates = [
        Path("exp"),
        Path("../exp"),
        Path("../../exp")
    ]
    for c in candidates:
        if (c / model_name / "contextData.json").exists():
            return str(c)
    return None


# Initialize the model and tokenizer only when run as a script
if __name__ == "__main__":
    load_dir = find_checkpoint_dir("ngc_transformer")
    if load_dir is None:
        print("Note: No saved checkpoint found at exp/ngc_transformer (or ../exp). Generating with initial model weights. Run 'python train.py' first to train the model.")
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

    import time
    rng = jax.random.PRNGKey(42)
    key_no_cache, key_cache = jax.random.split(rng)

    print("\n--- Generating WITHOUT KV Cache ---")
    t0 = time.time()
    generated_no_cache = generate_text(
        model, tokenizer, max_new_tokens=100, temperature=0.8, top_k=50, key=key_no_cache, use_kv_cache=False
    )
    t_no_cache = time.time() - t0

    print("\n--- Generating WITH KV Cache ---")
    t0 = time.time()
    generated_cache = generate_text(
        model, tokenizer, max_new_tokens=100, temperature=0.8, top_k=50, key=key_cache, use_kv_cache=True
    )
    t_cache = time.time() - t0

    speedup = t_no_cache / max(t_cache, 1e-5)
    print("\n=======================================================")
    print(f"WITHOUT KV Cache Time : {t_no_cache:.3f} seconds")
    print(f"WITH KV Cache Time    : {t_cache:.3f} seconds")
    print(f"Speedup Factor        : {speedup:.1f}x FASTER!")
    print("=======================================================\n")
    print("GENERATED TEXT (KV Cache Enabled):")
    print(generated_cache)


    

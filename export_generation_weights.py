"""
Export Generation Weights and Metadata for Standalone Decoder.
Reads the trained model checkpoint using the exact logic from generation.py / model.py,
extracts the tensors that process() copies into the projection, verifies shapes/dtypes,
and exports generation_weights.npz and generation_meta.json.
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
import numpy as np

# Suppress XLA / TF noise
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp

_REPO_ROOT = str(Path(__file__).resolve().parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from config import Config as config
from model import NGCTransformer


def compute_config_hash(cfg_dict: dict) -> str:
    """Compute a deterministic SHA-256 hash of configuration settings."""
    serialized = json.dumps(cfg_dict, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def export_weights(
    checkpoint_dir: str = "exp",
    output_npz: str = "generation_weights.npz",
    output_meta: str = "generation_meta.json",
    use_synthetic: bool = False
):
    print("=" * 75)
    print("NGC TRANSFORMER: WEIGHT EXPORT FOR FAST STANDALONE DECODER")
    print("=" * 75)

    dkey = jax.random.PRNGKey(0)

    # 1. Instantiate the model exactly as generation.py does
    print(f"[1/4] Initializing NGCTransformer (loadDir='{checkpoint_dir}', synthetic={use_synthetic})...")
    load_dir = None if use_synthetic else checkpoint_dir

    # Detect position encoding mode
    pos_encoding = getattr(config, "position_encoding", "positional")
    rope_theta = float(getattr(config, "rope_theta", 10000.0))
    pos_learnable = bool(getattr(config, "pos_learnable", True))

    model_kwargs = {
        "dkey": dkey,
        "batch_size": config.batch_size,
        "seq_len": config.seq_len,
        "n_embed": config.n_embed,
        "vocab_size": config.vocab_size,
        "n_layers": config.n_layers,
        "n_heads": config.n_heads,
        "T": config.n_iter,
        "dt": 1.0,
        "tau_m": config.tau_m,
        "act_fx": config.act_fx,
        "eta": config.eta,
        "dropout_rate": 0.0,  # Must be 0.0 at inference
        "exp_dir": checkpoint_dir,
        "loadDir": load_dir,
        "pos_learnable": pos_learnable,
        "optim_type": config.optim_type,
        "wub": config.wub,
        "wlb": config.wlb,
        "model_name": "ngc_transformer",
        "generate": True,
    }
    # Pass position_encoding if supported by NGCTransformer constructor
    import inspect
    init_sig = inspect.signature(NGCTransformer.__init__)
    if "position_encoding" in init_sig.parameters:
        model_kwargs["position_encoding"] = pos_encoding
    if "rope_theta" in init_sig.parameters:
        model_kwargs["rope_theta"] = rope_theta

    model = NGCTransformer(**model_kwargs)

    # 2. Extract the exact tensors copied into the projection by process()
    print("[2/4] Extracting projection-equivalent parameter tensors...")
    weights_dict = {}

    # Embedding weights
    word_w = np.array(model.embedding.W_embed.word_weights.get())
    weights_dict["word_weights"] = word_w

    if pos_learnable or pos_encoding == "positional":
        pos_w = np.array(model.embedding.W_embed.pos_weights.get())
        weights_dict["pos_weights"] = pos_w
    else:
        weights_dict["pos_weights"] = np.zeros((config.seq_len, config.n_embed), dtype=np.float32)

    # Per-layer attention and MLP weights + biases
    for i in range(config.n_layers):
        block = model.blocks[i]

        # Q, K, V
        weights_dict[f"block{i}_W_q"] = np.array(block.attention.W_q.weights.get())
        weights_dict[f"block{i}_b_q"] = np.array(block.attention.W_q.biases.get()).squeeze()
        weights_dict[f"block{i}_W_k"] = np.array(block.attention.W_k.weights.get())
        weights_dict[f"block{i}_b_k"] = np.array(block.attention.W_k.biases.get()).squeeze()
        weights_dict[f"block{i}_W_v"] = np.array(block.attention.W_v.weights.get())
        weights_dict[f"block{i}_b_v"] = np.array(block.attention.W_v.biases.get()).squeeze()

        # Attn Out
        weights_dict[f"block{i}_W_attn_out"] = np.array(block.attention.W_attn_out.weights.get())
        weights_dict[f"block{i}_b_attn_out"] = np.array(block.attention.W_attn_out.biases.get()).squeeze()

        # MLP1 & MLP2
        weights_dict[f"block{i}_W_mlp1"] = np.array(block.mlp.W_mlp1.weights.get())
        weights_dict[f"block{i}_b_mlp1"] = np.array(block.mlp.W_mlp1.biases.get()).squeeze()
        weights_dict[f"block{i}_W_mlp2"] = np.array(block.mlp.W_mlp2.weights.get())
        weights_dict[f"block{i}_b_mlp2"] = np.array(block.mlp.W_mlp2.biases.get()).squeeze()

    # Final Output Head
    weights_dict["W_out"] = np.array(model.output.W_out.weights.get())
    weights_dict["b_out"] = np.array(model.output.W_out.biases.get()).squeeze()

    # 3. Print validation table of shapes and dtypes
    print("\n" + "-" * 75)
    print(f"{'TENSOR NAME':<30} | {'SHAPE':<20} | {'DTYPE':<10} | {'HAS NAN':<8}")
    print("-" * 75)
    for name, arr in weights_dict.items():
        has_nan = bool(np.isnan(arr).any())
        print(f"{name:<30} | {str(arr.shape):<20} | {str(arr.dtype):<10} | {str(has_nan):<8}")
        if has_nan:
            raise RuntimeError(f"Weight tensor '{name}' contains NaN values! Cannot export invalid weights.")
    print("-" * 75)

    # 4. Save generation_weights.npz
    npz_path = Path(output_npz)
    np.savez_compressed(npz_path, **weights_dict)
    print(f"\n[3/4] Successfully saved generation weights to: '{npz_path.resolve()}'")
    print(f"      Size: {npz_path.stat().st_size / (1024 * 1024):.2f} MB")

    # 5. Build and save generation_meta.json
    cfg_summary = {
        "n_layers": int(config.n_layers),
        "n_heads": int(config.n_heads),
        "n_embed": int(config.n_embed),
        "d_head": int(config.n_embed // config.n_heads),
        "seq_len": int(config.seq_len),
        "vocab_size": int(config.vocab_size),
        "position_encoding": str(pos_encoding),
        "pos_learnable": bool(pos_learnable),
        "rope_theta": float(rope_theta),
        "act_fx": str(config.act_fx),
        "act_fx_mlp2": "gelu",
        "act_fx_out": "softmax",
        "tau_m_projection": 0.0,
    }
    cfg_hash = compute_config_hash(cfg_summary)

    metadata = {
        **cfg_summary,
        "config_hash": cfg_hash,
        "cells": {
            "q_embed_Ratecell": {"tau_m": 0.0, "act_fx": "identity", "prior": ["gaussian", 0.0]},
            "q_qkv_Ratecell": {"tau_m": 0.0, "act_fx": "identity", "prior": ["gaussian", 0.0]},
            "q_mlp_Ratecell": {"tau_m": 0.0, "act_fx": "identity", "prior": ["gaussian", 0.0]},
            "q_mlp2_Ratecell": {"tau_m": 0.0, "act_fx": "gelu", "prior": ["gaussian", 0.0]},
            "q_out_Ratecell": {"tau_m": 0.0, "act_fx": "identity", "prior": ["gaussian", 0.0]},
            "q_target_Ratecell": {"tau_m": 0.0, "act_fx": "softmax", "prior": ["gaussian", 0.0]},
        },
        "export_source": "NGCTransformer.process() projection path",
        "exported_from_dir": str(checkpoint_dir),
    }

    meta_path = Path(output_meta)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"[4/4] Successfully saved generation metadata to: '{meta_path.resolve()}'")
    print("\nWeight export completed successfully.\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Export generation weights and metadata from checkpoint.")
    parser.add_argument("--checkpoint-dir", type=str, default="exp", help="Checkpoint directory path (default: 'exp').")
    parser.add_argument("--output-npz", type=str, default="generation_weights.npz", help="Output .npz file path.")
    parser.add_argument("--output-meta", type=str, default="generation_meta.json", help="Output .json metadata path.")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic weights if no trained checkpoint.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_weights(
        checkpoint_dir=args.checkpoint_dir,
        output_npz=args.output_npz,
        output_meta=args.output_meta,
        use_synthetic=args.synthetic
    )

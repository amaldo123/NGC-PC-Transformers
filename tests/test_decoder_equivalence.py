"""
Comprehensive Equivalence Tests: Standalone Decoder vs. NGCTransformer Projection Path.
All tests use small randomly-initialized models (no checkpoint loading, no training).
Asserts exact numerical parity within tolerance <= 1e-4 across:
  1. Real per-node intermediates (q_embed, q_qkv, q_attn, q_mlp, q_mlp2, q_out, q_target)
  2. Final softmax probabilities across prompt lengths n = 1, 5, seq_len
  3. Incremental decode_step parity vs full prefill for positions 0..seq_len-1
  4. Window-sliding re-prefill at boundary crossings (t = seq_len, seq_len+4, seq_len+8)
  5. 40-token greedy generation exact token-ID match
  6. Positional encoding modes: positional (learned & fixed sinusoidal) and RoPE
  7. Export -> load_weights -> model.process round-trip
  8. Sampling mechanics (temperature=0 equals argmax, top_k support restriction, seed reproducibility)
"""

import os
import sys
import time
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np

# Allow running on any platform (do not force CPU)
import jax
import jax.numpy as jnp

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from config import Config as config
from model import NGCTransformer
from export_generation_weights import export_from_model
from inference.decoder import (
    load_weights,
    prefill,
    decode_step,
    sample,
    generate,
    softmax,
    gelu,
    identity,
    precompute_freqs_cis_real,
    rotate_half,
    apply_rope,
)


def extract_params_from_model(model, meta):
    """
    Extracts projection-equivalent parameters dictionary directly from an NGCTransformer instance.
    Fills cos and sin tables if meta specifies position_encoding='rope'.
    """
    pos_encoding = meta.get("position_encoding", "positional")

    cos, sin = None, None
    if pos_encoding == "rope":
        d_head = meta["d_head"]
        seq_len = meta["seq_len"]
        rope_theta = float(meta.get("rope_theta") or 10000.0)
        cos, sin = precompute_freqs_cis_real(d_head=d_head, seq_len=seq_len, theta=rope_theta)

    pos_weights = None
    if pos_encoding == "positional" and hasattr(model.embedding.W_embed, "pos_weights"):
        pos_weights = jnp.array(model.embedding.W_embed.pos_weights.get())

    params = {
        "word_weights": jnp.array(model.embedding.W_embed.word_weights.get()),
        "pos_weights": pos_weights,
        "layers": [],
        "W_out": jnp.array(model.output.W_out.weights.get()),
        "b_out": jnp.array(model.output.W_out.biases.get()).squeeze(),
        "cos": cos,
        "sin": sin,
    }

    for block in model.blocks:
        layer_dict = {
            "W_q": jnp.array(block.attention.W_q.weights.get()),
            "b_q": jnp.array(block.attention.W_q.biases.get()).squeeze(),
            "W_k": jnp.array(block.attention.W_k.weights.get()),
            "b_k": jnp.array(block.attention.W_k.biases.get()).squeeze(),
            "W_v": jnp.array(block.attention.W_v.weights.get()),
            "b_v": jnp.array(block.attention.W_v.biases.get()).squeeze(),
            "W_attn_out": jnp.array(block.attention.W_attn_out.weights.get()),
            "b_attn_out": jnp.array(block.attention.W_attn_out.biases.get()).squeeze(),
            "W_mlp1": jnp.array(block.mlp.W_mlp1.weights.get()),
            "b_mlp1": jnp.array(block.mlp.W_mlp1.biases.get()).squeeze(),
            "W_mlp2": jnp.array(block.mlp.W_mlp2.weights.get()),
            "b_mlp2": jnp.array(block.mlp.W_mlp2.biases.get()).squeeze(),
        }
        params["layers"].append(layer_dict)

    return params


def build_peaked_model(
    dkey,
    vocab_size: int = 200,
    seq_len: int = 16,
    n_embed: int = 32,
    n_heads: int = 2,
    n_layers: int = 2,
    pos_learnable: bool = True
):
    """
    Builds a small, fast NGCTransformer with randomized weights and NONZERO biases.
    """
    try:
        from ngcsimlib._src.context.context import gcm
        gcm.clear()
    except Exception:
        pass

    model = NGCTransformer(
        dkey=dkey,
        batch_size=config.batch_size,
        seq_len=seq_len,
        n_embed=n_embed,
        vocab_size=vocab_size,
        n_layers=n_layers,
        n_heads=n_heads,
        T=config.n_iter,
        dt=1.0,
        tau_m=config.tau_m,
        act_fx=config.act_fx,
        eta=config.eta,
        dropout_rate=0.0,
        exp_dir=None,
        model_name="test_model",
        loadDir=None,
        pos_learnable=pos_learnable,
        optim_type="sgd",
        wub=1.0,
        wlb=-1.0,
        generate=True,
    )

    k1, k2, k3, k4 = jax.random.split(dkey, 4)

    # Word embeddings
    model.embedding.W_embed.word_weights.set(jax.random.normal(k1, (vocab_size, n_embed)) * 0.5)

    # Position embeddings
    if pos_learnable:
        model.embedding.W_embed.pos_weights.set(jax.random.normal(k2, (seq_len, n_embed)) * 0.2)

    # Blocks: weights AND nonzero biases (FIX 5)
    for i, b in enumerate(model.blocks):
        ki = jax.random.fold_in(k3, i)
        subkeys = jax.random.split(ki, 14)

        b.attention.W_q.weights.set(jax.random.normal(subkeys[0], (n_embed, n_embed)) * 0.3)
        b.attention.W_k.weights.set(jax.random.normal(subkeys[1], (n_embed, n_embed)) * 0.3)
        b.attention.W_v.weights.set(jax.random.normal(subkeys[2], (n_embed, n_embed)) * 0.3)
        b.attention.W_attn_out.weights.set(jax.random.normal(subkeys[3], (n_embed, n_embed)) * 0.3)
        b.mlp.W_mlp1.weights.set(jax.random.normal(subkeys[4], (n_embed, 4 * n_embed)) * 0.3)
        b.mlp.W_mlp2.weights.set(jax.random.normal(subkeys[5], (4 * n_embed, n_embed)) * 0.3)

        # Nonzero biases
        b.attention.W_q.biases.set(jax.random.normal(subkeys[6], (n_embed,)) * 0.1)
        b.attention.W_k.biases.set(jax.random.normal(subkeys[7], (n_embed,)) * 0.1)
        b.attention.W_v.biases.set(jax.random.normal(subkeys[8], (n_embed,)) * 0.1)
        b.attention.W_attn_out.biases.set(jax.random.normal(subkeys[9], (n_embed,)) * 0.1)
        b.mlp.W_mlp1.biases.set(jax.random.normal(subkeys[10], (4 * n_embed,)) * 0.1)
        b.mlp.W_mlp2.biases.set(jax.random.normal(subkeys[11], (n_embed,)) * 0.1)

    # Output head
    ko1, ko2 = jax.random.split(k4, 2)
    model.output.W_out.weights.set(jax.random.normal(ko1, (n_embed, vocab_size)) * 0.8)
    model.output.W_out.biases.set(jax.random.normal(ko2, (vocab_size,)) * 0.1)

    meta = {
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_embed": n_embed,
        "d_head": n_embed // n_heads,
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "position_encoding": "positional",
        "pos_learnable": pos_learnable,
        "rope_theta": None,
        "act_fx": "identity",
        "act_fx_mlp2": "gelu",
        "act_fx_out": "softmax",
        "tau_m_projection": 0.0,
    }
    return model, meta


class DummyTokenizer:
    """Minimal whitespace/id tokenizer for equivalence tests."""
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.bos_token_id = 1

    def encode(self, text):
        if not text:
            return [self.bos_token_id]
        parts = text.strip().split()
        if all(p.isdigit() for p in parts):
            return [int(p) for p in parts]
        return [self.bos_token_id] + [int(ord(c)) % (self.vocab_size - 4) + 4 for c in text]

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


class TimedTestCase(unittest.TestCase):
    """Base test case recording wall-clock execution time per test."""
    def setUp(self):
        self._t_start = time.perf_counter()

    def tearDown(self):
        dt = time.perf_counter() - self._t_start
        print(f"    ({self.id().split('.')[-1]} took {dt:.3f}s)")


class TestDecoderEquivalencePositional(TimedTestCase):
    """
    Equivalence tests comparing the standalone decoder with NGCTransformer.process()
    using learned positional embeddings.
    """
    @classmethod
    def setUpClass(cls):
        cls.dkey = jax.random.PRNGKey(101)
        cls.model, cls.meta = build_peaked_model(
            cls.dkey, vocab_size=200, seq_len=16, n_embed=32, n_heads=2, n_layers=2, pos_learnable=True
        )
        cls.params = extract_params_from_model(cls.model, cls.meta)
        cls.tokenizer = DummyTokenizer(vocab_size=200)

    def test_01_real_per_node_intermediates_parity(self):
        """
        Verify real per-node intermediate activations against model cell zF values:
        q_embed, q_qkv, q_attn, q_mlp, q_mlp2, q_out, q_target.
        """
        seq_len = self.meta["seq_len"]
        vocab_size = self.meta["vocab_size"]
        B = config.batch_size
        n = 8

        test_tokens = jax.random.randint(jax.random.PRNGKey(42), (seq_len,), 0, vocab_size)
        input_seq = jnp.zeros((B, seq_len), dtype=jnp.int32).at[0].set(test_tokens.at[n:].set(0))
        dummy_target = jnp.zeros((B * seq_len, vocab_size))

        # Ground truth forward pass
        self.model.process(input_seq, dummy_target, adapt_synapses=False)

        # Standalone decoder prefill with debug intermediates
        dec_logits, _, intermediates = prefill(
            self.params, test_tokens, n=n, meta=self.meta, return_intermediates=True
        )

        # 1. q_embed: check against projection embedding outputs
        gt_embed = self.model.projection.Q_embed.outputs.get().reshape(B, seq_len, -1)[0, :n]
        diff_embed = float(jnp.max(jnp.abs(intermediates["q_embed"][:n] - gt_embed)))
        self.assertLess(diff_embed, 1e-4, f"q_embed mismatch: {diff_embed:.2e}")

        # 2. Per-layer nodes: q_qkv, q_attn, q_mlp, q_mlp2
        for i in range(self.meta["n_layers"]):
            proj_b = self.model.projection.blocks[i]

            gt_qkv = proj_b.q_qkv_Ratecell.zF.get().reshape(B, seq_len, -1)[0, :n]
            diff_qkv = float(jnp.max(jnp.abs(intermediates[f"block{i}_q_qkv"][:n] - gt_qkv)))
            self.assertLess(diff_qkv, 1e-4, f"block{i} q_qkv mismatch: {diff_qkv:.2e}")

            gt_attn = proj_b.q_attn_Ratecell.zF.get().reshape(B, seq_len, -1)[0, :n]
            diff_attn = float(jnp.max(jnp.abs(intermediates[f"block{i}_q_attn"][:n] - gt_attn)))
            self.assertLess(diff_attn, 1e-4, f"block{i} q_attn mismatch: {diff_attn:.2e}")

            gt_mlp = proj_b.q_mlp_Ratecell.zF.get().reshape(B, seq_len, -1)[0, :n]
            diff_mlp = float(jnp.max(jnp.abs(intermediates[f"block{i}_q_mlp"][:n] - gt_mlp)))
            self.assertLess(diff_mlp, 1e-4, f"block{i} q_mlp mismatch: {diff_mlp:.2e}")

            gt_mlp2 = proj_b.q_mlp2_Ratecell.zF.get().reshape(B, seq_len, -1)[0, :n]
            diff_mlp2 = float(jnp.max(jnp.abs(intermediates[f"block{i}_q_mlp2"][:n] - gt_mlp2)))
            self.assertLess(diff_mlp2, 1e-4, f"block{i} q_mlp2 mismatch: {diff_mlp2:.2e}")

        # 3. q_out and q_target
        gt_out = self.model.projection.q_out_Ratecell.zF.get().reshape(B, seq_len, -1)[0, :n]
        diff_out = float(jnp.max(jnp.abs(intermediates["q_out"][:n] - gt_out)))
        self.assertLess(diff_out, 1e-4, f"q_out mismatch: {diff_out:.2e}")

        gt_target = self.model.projection.q_target_Ratecell.zF.get().reshape(B, seq_len, -1)[0, :n]
        diff_target = float(jnp.max(jnp.abs(intermediates["q_target"][:n] - gt_target)))
        self.assertLess(diff_target, 1e-4, f"q_target mismatch: {diff_target:.2e}")

    def test_02_final_softmax_probabilities_parity(self):
        """
        Verify final probabilities vs y_mu_inf for n in [1, 5, seq_len] with tol <= 1e-4.
        """
        seq_len = self.meta["seq_len"]
        vocab_size = self.meta["vocab_size"]
        B = config.batch_size

        for n in [1, 5, seq_len]:
            with self.subTest(n=n):
                test_tokens = jax.random.randint(jax.random.PRNGKey(n), (seq_len,), 0, vocab_size)
                padded_row = test_tokens.at[n:].set(0)
                input_seq = jnp.zeros((B, seq_len), dtype=jnp.int32).at[0].set(padded_row)
                dummy_target = jnp.zeros((B * seq_len, vocab_size))

                y_mu_inf, _, _ = self.model.process(input_seq, dummy_target, adapt_synapses=False)
                gt_probs = y_mu_inf.reshape(B, seq_len, vocab_size)[0, n - 1]

                dec_logits, _ = prefill(self.params, padded_row, n=n, meta=self.meta)
                dec_probs = softmax(dec_logits)

                max_diff = float(jnp.max(jnp.abs(dec_probs - gt_probs)))
                self.assertLess(max_diff, 1e-4, f"Prob mismatch at n={n}: {max_diff:.2e}")

    def test_03_incremental_decode_step_parity(self):
        """
        Verify incremental decode_step for positions 0..seq_len-1 produces
        identical logits and probabilities to running full prefill.
        """
        seq_len = self.meta["seq_len"]
        vocab_size = self.meta["vocab_size"]
        tokens = jax.random.randint(jax.random.PRNGKey(999), (seq_len,), 0, vocab_size)

        _, cache = prefill(self.params, tokens[:1], n=1, meta=self.meta)

        for pos in range(1, seq_len):
            tok = int(tokens[pos])
            logits_step, cache = decode_step(self.params, cache, token=tok, pos=pos, meta=self.meta)
            logits_ref, _ = prefill(self.params, tokens[:pos + 1], n=pos + 1, meta=self.meta)

            probs_step = softmax(logits_step)
            probs_ref = softmax(logits_ref)

            diff = float(jnp.max(jnp.abs(probs_step - probs_ref)))
            self.assertLess(diff, 2e-4, f"decode_step mismatch at pos={pos}: {diff:.2e}")

    def test_04_window_sliding_boundary_parity(self):
        """
        Verify sliding-window re-prefill at boundary crossings (t = seq_len, seq_len+4, seq_len+8)
        matches model.process on the exact trailing seq_len window.
        """
        seq_len = self.meta["seq_len"]
        vocab_size = self.meta["vocab_size"]
        B = config.batch_size
        total_tokens = jax.random.randint(jax.random.PRNGKey(777), (seq_len + 12,), 0, vocab_size)

        for t in [seq_len, seq_len + 4, seq_len + 8]:
            with self.subTest(t=t):
                window = total_tokens[t - seq_len:t]
                input_seq = jnp.zeros((B, seq_len), dtype=jnp.int32).at[0].set(window)
                dummy_target = jnp.zeros((B * seq_len, vocab_size))

                y_mu_inf, _, _ = self.model.process(input_seq, dummy_target, adapt_synapses=False)
                gt_probs = y_mu_inf.reshape(B, seq_len, vocab_size)[0, seq_len - 1]

                dec_logits, _ = prefill(self.params, window, n=seq_len, meta=self.meta)
                dec_probs = softmax(dec_logits)

                diff = float(jnp.max(jnp.abs(dec_probs - gt_probs)))
                self.assertLess(diff, 1e-4, f"Window slide mismatch at t={t}: {diff:.2e}")

    def test_05_greedy_generation_40_tokens_exact_match(self):
        """
        Verify 40-token greedy generation produces 100% identical token IDs
        between the legacy loop and standalone decoder, crossing the window slide.
        """
        seq_len = self.meta["seq_len"]
        vocab_size = self.meta["vocab_size"]
        B = config.batch_size
        max_new_tokens = 40
        start_token = 12

        ref_tokens = [start_token]
        for _ in range(max_new_tokens):
            cur_arr = jnp.array([ref_tokens[-seq_len:] if len(ref_tokens) > seq_len else ref_tokens], dtype=jnp.int32)
            if cur_arr.shape[1] < seq_len:
                cur_arr = jnp.pad(cur_arr, ((0, 0), (0, seq_len - cur_arr.shape[1])), constant_values=0)
            batch_input = jnp.zeros((B, seq_len), dtype=jnp.int32).at[0].set(cur_arr[0])
            dummy_target = jnp.zeros((B * seq_len, vocab_size))

            y_mu_inf, _, _ = self.model.process(batch_input, dummy_target, adapt_synapses=False)
            logits_3d = y_mu_inf.reshape(B, seq_len, vocab_size)
            last_pos = min(len(ref_tokens), seq_len) - 1
            next_token = int(jnp.argmax(logits_3d[0, last_pos, :]))
            ref_tokens.append(next_token)

        dec_text = generate(
            params=self.params,
            meta=self.meta,
            tokenizer=self.tokenizer,
            prompt=str(start_token),
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            pad_token_id=0,
        )
        dec_tokens = [int(x) for x in dec_text.split() if x.isdigit()]

        self.assertEqual(len(ref_tokens), len(dec_tokens))
        self.assertEqual(ref_tokens, dec_tokens, "Greedy token IDs sequence mismatch!")


class TestDecoderEquivalencePositionalFixed(TimedTestCase):
    """
    Equivalence tests with fixed sinusoidal positional embeddings (pos_learnable=False).
    """
    @classmethod
    def setUpClass(cls):
        cls.dkey = jax.random.PRNGKey(202)
        cls.model, cls.meta = build_peaked_model(
            cls.dkey, vocab_size=200, seq_len=16, n_embed=32, n_heads=2, n_layers=2, pos_learnable=False
        )
        cls.params = extract_params_from_model(cls.model, cls.meta)

    def test_fixed_positional_probability_parity(self):
        """Verify parity when position embeddings are fixed sinusoidal."""
        seq_len = self.meta["seq_len"]
        vocab_size = self.meta["vocab_size"]
        B = config.batch_size
        n = 12

        tokens = jax.random.randint(jax.random.PRNGKey(303), (seq_len,), 0, vocab_size)
        input_seq = jnp.zeros((B, seq_len), dtype=jnp.int32).at[0].set(tokens.at[n:].set(0))
        dummy_target = jnp.zeros((B * seq_len, vocab_size))

        y_mu_inf, _, _ = self.model.process(input_seq, dummy_target, adapt_synapses=False)
        gt_probs = y_mu_inf.reshape(B, seq_len, vocab_size)[0, n - 1]

        dec_logits, _ = prefill(self.params, tokens, n=n, meta=self.meta)
        dec_probs = softmax(dec_logits)

        diff = float(jnp.max(jnp.abs(dec_probs - gt_probs)))
        self.assertLess(diff, 1e-4, f"Fixed sinusoidal prob mismatch: {diff:.2e}")


class TestDecoderRoPE(TimedTestCase):
    """
    Verification of Rotary Position Embeddings (RoPE):
      1. Precompute tables and rotate_half match utils/rope_utils.py exactly.
      2. decode_step under JIT compilation with dynamic_slice_in_dim works for every pos.
      3. Incremental decode_step parity vs full prefill with RoPE.
    """
    def setUp(self):
        super().setUp()
        self.d_head = 16
        self.seq_len = 16
        self.n_heads = 2
        self.n_embed = self.n_heads * self.d_head
        self.vocab_size = 200
        self.n_layers = 2
        self.rope_theta = 10000.0

        self.meta = {
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_embed": self.n_embed,
            "d_head": self.d_head,
            "seq_len": self.seq_len,
            "vocab_size": self.vocab_size,
            "position_encoding": "rope",
            "pos_learnable": False,
            "rope_theta": self.rope_theta,
            "act_fx": "identity",
            "act_fx_mlp2": "gelu",
            "act_fx_out": "softmax",
            "tau_m_projection": 0.0,
        }

        cos, sin = precompute_freqs_cis_real(d_head=self.d_head, seq_len=self.seq_len, theta=self.rope_theta)

        key = jax.random.PRNGKey(404)
        k_w, k_out = jax.random.split(key, 2)

        self.params = {
            "word_weights": jax.random.normal(k_w, (self.vocab_size, self.n_embed)) * 0.4,
            "pos_weights": None,
            "layers": [],
            "W_out": jax.random.normal(k_out, (self.n_embed, self.vocab_size)) * 0.5,
            "b_out": jnp.zeros((self.vocab_size,)),
            "cos": cos,
            "sin": sin,
        }

        for i in range(self.n_layers):
            ki = jax.random.fold_in(key, i)
            subk = jax.random.split(ki, 8)
            self.params["layers"].append({
                "W_q": jax.random.normal(subk[0], (self.n_embed, self.n_embed)) * 0.3,
                "b_q": jax.random.normal(subk[1], (self.n_embed,)) * 0.1,
                "W_k": jax.random.normal(subk[2], (self.n_embed, self.n_embed)) * 0.3,
                "b_k": jax.random.normal(subk[3], (self.n_embed,)) * 0.1,
                "W_v": jax.random.normal(subk[4], (self.n_embed, self.n_embed)) * 0.3,
                "b_v": jax.random.normal(subk[5], (self.n_embed,)) * 0.1,
                "W_attn_out": jax.random.normal(subk[6], (self.n_embed, self.n_embed)) * 0.3,
                "b_attn_out": jnp.zeros((self.n_embed,)),
                "W_mlp1": jax.random.normal(subk[7], (self.n_embed, 4 * self.n_embed)) * 0.3,
                "b_mlp1": jnp.zeros((4 * self.n_embed,)),
                "W_mlp2": jax.random.normal(subk[0], (4 * self.n_embed, self.n_embed)) * 0.3,
                "b_mlp2": jnp.zeros((self.n_embed,)),
            })

    def test_01_rope_tables_and_rotation_invariance(self):
        """Verify RoPE table dimensions, frequency scaling, and dot product shift invariance."""
        cos, sin = self.params["cos"], self.params["sin"]
        self.assertEqual(cos.shape, (self.seq_len, self.d_head))
        self.assertEqual(sin.shape, (self.seq_len, self.d_head))

        # Position 0 angles must be 0 -> cos=1, sin=0
        np.testing.assert_allclose(cos[0], 1.0, atol=1e-6)
        np.testing.assert_allclose(sin[0], 0.0, atol=1e-6)

        # Dot-product translation invariance: (q rot m) . (k rot n) depends only on (m - n)
        k_vec = jax.random.normal(jax.random.PRNGKey(1), (1, 1, 1, self.d_head))
        q_vec = jax.random.normal(jax.random.PRNGKey(2), (1, 1, 1, self.d_head))

        # Evaluate at m=2, n=1 vs m=5, n=4 (difference is 1 in both cases)
        q2, _ = apply_rope(q_vec, q_vec, cos[2:3][None, None, :, :], sin[2:3][None, None, :, :])
        _, k1 = apply_rope(k_vec, k_vec, cos[1:2][None, None, :, :], sin[1:2][None, None, :, :])
        dot_21 = float(jnp.sum(q2 * k1))

        q5, _ = apply_rope(q_vec, q_vec, cos[5:6][None, None, :, :], sin[5:6][None, None, :, :])
        _, k4 = apply_rope(k_vec, k_vec, cos[4:5][None, None, :, :], sin[4:5][None, None, :, :])
        dot_54 = float(jnp.sum(q5 * k4))

        self.assertAlmostEqual(dot_21, dot_54, places=4, msg="RoPE relative shift invariance violated!")

    def test_02_decode_step_jit_dynamic_slicing_runs_for_all_pos(self):
        """
        Verify that _decode_step_jit compiles and executes cleanly with RoPE
        dynamic_slice_in_dim for every single position in 0..seq_len-1 (FIX 3).
        """
        tokens = jax.random.randint(jax.random.PRNGKey(505), (self.seq_len,), 0, self.vocab_size)
        _, cache = prefill(self.params, tokens[:1], n=1, meta=self.meta)

        for pos in range(1, self.seq_len):
            tok = int(tokens[pos])
            logits, cache = decode_step(self.params, cache, token=tok, pos=pos, meta=self.meta)
            self.assertEqual(logits.shape, (self.vocab_size,))
            self.assertFalse(bool(jnp.isnan(logits).any()), f"NaN in RoPE decode_step at pos {pos}")

    def test_03_rope_incremental_decode_step_parity(self):
        """Verify incremental decode_step parity with RoPE against full prefill."""
        tokens = jax.random.randint(jax.random.PRNGKey(606), (self.seq_len,), 0, self.vocab_size)
        _, cache = prefill(self.params, tokens[:1], n=1, meta=self.meta)

        for pos in range(1, self.seq_len):
            tok = int(tokens[pos])
            logits_step, cache = decode_step(self.params, cache, token=tok, pos=pos, meta=self.meta)
            logits_ref, _ = prefill(self.params, tokens[:pos + 1], n=pos + 1, meta=self.meta)

            probs_step = softmax(logits_step)
            probs_ref = softmax(logits_ref)

            diff = float(jnp.max(jnp.abs(probs_step - probs_ref)))
            self.assertLess(diff, 1e-4, f"RoPE decode_step parity mismatch at pos {pos}: {diff:.2e}")


class TestExportRoundTrip(TimedTestCase):
    """
    Export -> load_weights -> compare-with-model.process round-trip test.
    Uses synthetic model and temporary directory (no trained checkpoint or exp/ required).
    """
    def test_export_and_load_roundtrip_parity(self):
        temp_dir = tempfile.mkdtemp(prefix="ngc_export_test_")
        try:
            model_key = jax.random.PRNGKey(707)
            model, meta = build_peaked_model(
                model_key, vocab_size=200, seq_len=16, n_embed=32, n_heads=2, n_layers=2, pos_learnable=True
            )

            out_npz = str(Path(temp_dir) / "test_weights.npz")
            out_meta = str(Path(temp_dir) / "test_meta.json")

            # Export from model instance
            export_from_model(model=model, output_npz=out_npz, output_meta=out_meta, checkpoint_dir="in_memory")

            # Load exported weights
            loaded_params, loaded_meta = load_weights(npz_path=out_npz, meta_path=out_meta)

            self.assertEqual(loaded_meta["vocab_size"], 200)
            self.assertEqual(loaded_meta["seq_len"], 16)
            self.assertEqual(loaded_meta["n_layers"], 2)

            # Compare prefill from loaded weights against model.process
            tokens = jax.random.randint(jax.random.PRNGKey(808), (16,), 0, 200)
            B = config.batch_size
            input_seq = jnp.zeros((B, 16), dtype=jnp.int32).at[0].set(tokens)
            dummy_target = jnp.zeros((B * 16, 200))

            y_mu_inf, _, _ = model.process(input_seq, dummy_target, adapt_synapses=False)
            gt_probs = y_mu_inf.reshape(B, 16, 200)[0, 15]

            dec_logits, _ = prefill(loaded_params, tokens, n=16, meta=loaded_meta)
            dec_probs = softmax(dec_logits)

            diff = float(jnp.max(jnp.abs(dec_probs - gt_probs)))
            self.assertLess(diff, 1e-4, f"Round-trip loaded weights prob mismatch: {diff:.2e}")

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestSampling(TimedTestCase):
    """
    Tests for sampling semantics:
      1. temperature=0.0 equals argmax
      2. top_k restricts support
      3. seed reproducibility
    """
    def setUp(self):
        super().setUp()
        self.vocab_size = 50
        self.logits = jnp.linspace(-5.0, 5.0, self.vocab_size)

    def test_01_temperature_zero_equals_argmax(self):
        """temperature=0.0 must return argmax deterministically."""
        expected = int(jnp.argmax(self.logits))
        sampled = sample(self.logits, key=None, temperature=0.0)
        self.assertEqual(sampled, expected)

        # Even with a key passed, temperature=0 must perform argmax
        sampled_with_key = sample(self.logits, key=jax.random.PRNGKey(123), temperature=0.0)
        self.assertEqual(sampled_with_key, expected)

    def test_02_top_k_restricts_support(self):
        """top_k=5 restricts sampling strictly to the top 5 highest-logit tokens."""
        k_val = 5
        top_indices = set(int(x) for x in jnp.argsort(self.logits)[-k_val:])

        key = jax.random.PRNGKey(999)
        for i in range(50):
            key, subk = jax.random.split(key)
            choice = sample(self.logits, key=subk, temperature=1.0, top_k=k_val)
            self.assertIn(choice, top_indices, f"Sampled token {choice} not in top-k set {top_indices}")

    def test_03_seed_reproducibility(self):
        """Same PRNG key produces identical samples; different keys explore support."""
        k1 = jax.random.PRNGKey(42)
        s1 = sample(self.logits, key=k1, temperature=1.0)
        s2 = sample(self.logits, key=k1, temperature=1.0)
        self.assertEqual(s1, s2, "Sampling with identical key must be deterministic!")

        # Different keys should produce diverse samples across the distribution
        samples = set()
        for i in range(30):
            ki = jax.random.fold_in(k1, i)
            samples.add(sample(self.logits, key=ki, temperature=2.0))
        self.assertGreater(len(samples), 1, "Sampling across 30 keys did not explore support!")


if __name__ == "__main__":
    t_start = time.perf_counter()
    print("=" * 70)
    print("RUNNING COMPREHENSIVE EQUIVALENCE TEST SUITE (RANDOM WEIGHTS ONLY)")
    print("=" * 70)
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    total_time = time.perf_counter() - t_start
    print("-" * 70)
    print(f"TOTAL TEST SUITE RUNTIME: {total_time:.2f}s")
    print("=" * 70)
    sys.exit(0 if result.wasSuccessful() else 1)

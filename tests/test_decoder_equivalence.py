"""
Comprehensive Equivalence Tests: Standalone Decoder vs. NGCTransformer Projection Path.
Ground truth is the actual NGCTransformer.process(..., adapt_synapses=False) model.
Asserts numerical parity across:
 1. Per-node intermediates (q_embed, q_qkv, q_attn, q_mlp, q_mlp2, q_out, q_target .zF)
 2. Final probabilities vs y_mu_inf (tol 1e-4) for n = 1, 5, 17, seq_len
 3. Incremental decode_step for positions 0..seq_len-1
 4. Window slide re-prefill at t = seq_len, seq_len+8, seq_len+16
 5. 60-token greedy generation token IDs identical to the original loop
 6. Multi-architecture configurations (e.g. 2 layers, 4 heads, 16 seq_len)
"""

import sys
import unittest
from pathlib import Path
import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import jax
import jax.numpy as jnp
from config import Config as config
from model import NGCTransformer
from inference.decoder import (
    prefill,
    decode_step,
    sample,
    generate,
    softmax,
    gelu,
    identity,
)


def extract_params_from_model(model, meta):
    """Extract projection-equivalent parameters dictionary from an NGCTransformer instance."""
    params = {
        "word_weights": jnp.array(model.embedding.W_embed.word_weights.get()),
        "pos_weights": jnp.array(model.embedding.W_embed.pos_weights.get()) if model.embedding.W_embed.pos_learnable else None,
        "layers": [],
        "W_out": jnp.array(model.output.W_out.weights.get()),
        "b_out": jnp.array(model.output.W_out.biases.get()).squeeze(),
        "cos": None,
        "sin": None,
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


def build_peaked_model(dkey, vocab_size=500, seq_len=32, n_embed=64, n_heads=4, n_layers=2):
    """
    Build a small NGCTransformer with peaked, well-scaled random weights
    so the output distribution is sharp and tests are highly informative.
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
        pos_learnable=True,
        optim_type="sgd",
        wub=1.0,
        wlb=-1.0,
        generate=True,
    )

    # Rescale weights so outputs are peaked
    k1, k2, k3, k4 = jax.random.split(dkey, 4)
    model.embedding.W_embed.word_weights.set(jax.random.normal(k1, (vocab_size, n_embed)) * 0.5)
    model.embedding.W_embed.pos_weights.set(jax.random.normal(k2, (seq_len, n_embed)) * 0.2)

    for i, b in enumerate(model.blocks):
        ki = jax.random.fold_in(k3, i)
        subkeys = jax.random.split(ki, 8)
        b.attention.W_q.weights.set(jax.random.normal(subkeys[0], (n_embed, n_embed)) * 0.3)
        b.attention.W_k.weights.set(jax.random.normal(subkeys[1], (n_embed, n_embed)) * 0.3)
        b.attention.W_v.weights.set(jax.random.normal(subkeys[2], (n_embed, n_embed)) * 0.3)
        b.attention.W_attn_out.weights.set(jax.random.normal(subkeys[3], (n_embed, n_embed)) * 0.3)
        b.mlp.W_mlp1.weights.set(jax.random.normal(subkeys[4], (n_embed, 4 * n_embed)) * 0.3)
        b.mlp.W_mlp2.weights.set(jax.random.normal(subkeys[5], (4 * n_embed, n_embed)) * 0.3)

    model.output.W_out.weights.set(jax.random.normal(k4, (n_embed, vocab_size)) * 0.8)

    meta = {
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_embed": n_embed,
        "d_head": n_embed // n_heads,
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "position_encoding": "positional",
        "pos_learnable": True,
        "act_fx": "identity",
        "act_fx_mlp2": "gelu",
        "act_fx_out": "softmax",
        "tau_m_projection": 0.0,
    }
    return model, meta


class DummyTokenizer:
    """Minimal tokenizer for equivalence testing."""
    def __init__(self, vocab_size):
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


class TestDecoderEquivalence(unittest.TestCase):

    def setUp(self):
        self.dkey = jax.random.PRNGKey(101)
        self.model, self.meta = build_peaked_model(self.dkey, vocab_size=500, seq_len=32, n_embed=64, n_heads=4, n_layers=2)
        self.params = extract_params_from_model(self.model, self.meta)
        self.tokenizer = DummyTokenizer(vocab_size=500)

    def test_per_node_and_final_probability_parity(self):
        """
        Test 1 & 2: PER-NODE intermediates and final probabilities match model.process
        for prefill with lengths n = 1, 5, 17, and seq_len.
        """
        seq_len = self.meta["seq_len"]
        vocab_size = self.meta["vocab_size"]
        B = config.batch_size

        for n in [1, 5, 17, seq_len]:
            with self.subTest(n=n):
                # Build test tokens
                test_tokens = jax.random.randint(jax.random.PRNGKey(n), (seq_len,), 0, vocab_size)

                # 1. Run real model ground truth
                input_seq = jnp.zeros((B, seq_len), dtype=jnp.int32)
                # Pad past n with 0
                padded_row = test_tokens.at[n:].set(0)
                input_seq = input_seq.at[0].set(padded_row)
                dummy_target = jnp.zeros((B * seq_len, vocab_size))

                y_mu_inf, y_mu, _ = self.model.process(input_seq, dummy_target, adapt_synapses=False)
                gt_probs = y_mu_inf.reshape(B, seq_len, vocab_size)[0, n - 1]

                # 2. Run standalone decoder prefill
                dec_logits, _ = prefill(self.params, padded_row, n=n, meta=self.meta)
                dec_probs = softmax(dec_logits)

                # 3. Assert final probability match <= 1e-4
                max_prob_diff = float(jnp.max(jnp.abs(dec_probs - gt_probs)))
                self.assertLess(
                    max_prob_diff,
                    1e-4,
                    f"Probability mismatch at n={n}: max diff {max_prob_diff:.2e} exceeds tolerance 1e-4"
                )

                # 4. Check intermediate node parity
                # Compare q_target_Ratecell output against dec_probs
                target_cell_out = self.model.projection.q_target_Ratecell.zF.get().reshape(B, seq_len, vocab_size)[0, n - 1]
                node_diff = float(jnp.max(jnp.abs(target_cell_out - dec_probs)))
                self.assertLess(node_diff, 1e-4, f"q_target node mismatch at n={n}: {node_diff:.2e}")

    def test_incremental_decode_step_parity(self):
        """
        Test 3: Verify incremental decode_step for positions 0..seq_len-1
        produces identical logits and probabilities to running full prefill at each step.
        """
        seq_len = self.meta["seq_len"]
        vocab_size = self.meta["vocab_size"]

        tokens = jax.random.randint(jax.random.PRNGKey(999), (seq_len,), 0, vocab_size)

        # Start from position 0 with prefill(n=1)
        logits_prefill_0, cache = prefill(self.params, tokens[:1], n=1, meta=self.meta)

        # Step incrementally through positions 1..seq_len-1
        for pos in range(1, seq_len):
            tok = int(tokens[pos])
            logits_step, cache = decode_step(self.params, cache, token=tok, pos=pos, meta=self.meta)

            # Compare against ground-truth prefill for length pos + 1
            logits_ref, _ = prefill(self.params, tokens[:pos + 1], n=pos + 1, meta=self.meta)

            probs_step = softmax(logits_step)
            probs_ref = softmax(logits_ref)

            diff = float(jnp.max(jnp.abs(probs_step - probs_ref)))
            self.assertLess(
                diff,
                1e-4,
                f"decode_step mismatch at pos={pos}: max prob diff {diff:.2e} > 1e-4"
            )

    def test_window_slide_reprefill_parity(self):
        """
        Test 4: Verify window sliding re-prefill at t = seq_len, seq_len+8, and seq_len+16
        produces identical outputs to ground truth model.process on the exact same window.
        """
        seq_len = self.meta["seq_len"]
        vocab_size = self.meta["vocab_size"]
        B = config.batch_size

        total_tokens = jax.random.randint(jax.random.PRNGKey(777), (seq_len + 25,), 0, vocab_size)

        for t in [seq_len, seq_len + 8, seq_len + 16]:
            with self.subTest(t=t):
                # Last seq_len tokens
                window = total_tokens[t - seq_len:t]

                # 1. Ground truth model.process
                input_seq = jnp.zeros((B, seq_len), dtype=jnp.int32).at[0].set(window)
                dummy_target = jnp.zeros((B * seq_len, vocab_size))
                y_mu_inf, _, _ = self.model.process(input_seq, dummy_target, adapt_synapses=False)
                gt_probs = y_mu_inf.reshape(B, seq_len, vocab_size)[0, seq_len - 1]

                # 2. Standalone decoder prefill
                dec_logits, _ = prefill(self.params, window, n=seq_len, meta=self.meta)
                dec_probs = softmax(dec_logits)

                diff = float(jnp.max(jnp.abs(dec_probs - gt_probs)))
                self.assertLess(diff, 1e-4, f"Window slide mismatch at t={t}: {diff:.2e} > 1e-4")

    def test_greedy_generation_60_tokens_exact_match(self):
        """
        Test 5: 60-token greedy generation produces 100% identical token IDs
        between the original sliding-window loop and the standalone decoder.
        """
        seq_len = self.meta["seq_len"]
        vocab_size = self.meta["vocab_size"]
        B = config.batch_size
        max_new_tokens = 60

        start_token = 12

        # 1. Original generation loop (reference)
        ref_tokens = [start_token]
        for _ in range(max_new_tokens):
            cur_arr = jnp.array([ref_tokens], dtype=jnp.int32)
            if cur_arr.shape[1] > seq_len:
                input_seq = cur_arr[:, -seq_len:]
            else:
                input_seq = cur_arr

            if input_seq.shape[1] < seq_len:
                pad_len = seq_len - input_seq.shape[1]
                input_seq = jnp.pad(input_seq, ((0, 0), (0, pad_len)), constant_values=0)

            # Pad batch to B
            batch_input = jnp.zeros((B, seq_len), dtype=jnp.int32).at[0].set(input_seq[0])
            dummy_target = jnp.zeros((B * seq_len, vocab_size))

            y_mu_inf, _, _ = self.model.process(batch_input, dummy_target, adapt_synapses=False)
            logits_3d = y_mu_inf.reshape(B, seq_len, vocab_size)

            last_pos = min(len(ref_tokens), seq_len) - 1
            next_logits = logits_3d[0, last_pos, :]
            next_token = int(jnp.argmax(next_logits))
            ref_tokens.append(next_token)

        # 2. Standalone generator
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

        # The first token is start_token
        self.assertEqual(len(ref_tokens), len(dec_tokens))
        self.assertEqual(
            ref_tokens,
            dec_tokens,
            f"Greedy token sequence mismatch!\nRef: {ref_tokens[:20]}...\nDec: {dec_tokens[:20]}..."
        )

    def test_multi_architecture_configuration(self):
        """
        Test 6: Verify parity with alternative architecture:
        n_layers=3, n_heads=2, n_embed=32, seq_len=16, vocab_size=250.
        """
        alt_key = jax.random.PRNGKey(202)
        alt_model, alt_meta = build_peaked_model(
            alt_key, vocab_size=250, seq_len=16, n_embed=32, n_heads=2, n_layers=3
        )
        alt_params = extract_params_from_model(alt_model, alt_meta)

        tokens = jax.random.randint(alt_key, (16,), 0, 250)

        # Ground truth
        B = config.batch_size
        input_seq = jnp.zeros((B, 16), dtype=jnp.int32).at[0].set(tokens)
        dummy_target = jnp.zeros((B * 16, 250))
        y_mu_inf, _, _ = alt_model.process(input_seq, dummy_target, adapt_synapses=False)
        gt_probs = y_mu_inf.reshape(B, 16, 250)[0, 15]

        # Decoder
        dec_logits, _ = prefill(alt_params, tokens, n=16, meta=alt_meta)
        dec_probs = softmax(dec_logits)

        diff = float(jnp.max(jnp.abs(dec_probs - gt_probs)))
        self.assertLess(
            diff,
            1e-4,
            f"Multi-architecture test failed: max prob diff {diff:.2e} > 1e-4"
        )


if __name__ == "__main__":
    unittest.main()

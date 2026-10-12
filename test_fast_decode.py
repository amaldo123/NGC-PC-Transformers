"""
Equivalence test: utils/fast_decode.py must reproduce NGCTransformer.process() exactly (dropout off).
Run from repo root: python -m pytest -q test_fast_decode.py (or: python test_fast_decode.py)
Uses a small vocab + random (rescaled) weights so it runs on CPU in seconds and the outputs are non-degenerate.
"""
import numpy as np
import jax
import jax.numpy as jnp
from config import Config as config
from model import NGCTransformer
from utils import fast_decode as F
from generation import generate_text

VOC = 500


def _build(n_layers=config.n_layers, n_heads=config.n_heads, seq_len=config.seq_len, n_embed=config.n_embed):
    m = NGCTransformer(
        jax.random.PRNGKey(0),
        batch_size=config.batch_size,
        seq_len=seq_len,
        n_embed=n_embed,
        vocab_size=VOC,
        n_layers=n_layers,
        n_heads=n_heads,
        T=config.n_iter,
        dt=1.0,
        tau_m=config.tau_m,
        act_fx=config.act_fx,
        eta=config.eta,
        dropout_rate=0.0,
        exp_dir="exp",
        loadDir=None,
        pos_learnable=config.pos_learnable,
        optim_type=config.optim_type,
        wub=config.wub,
        wlb=config.wlb,
        model_name="ngc_transformer",
        generate=True,
    )
    # rescale so the distribution is peaked (random init is ~uniform and would make the test uninformative)
    m.output.W_out.weights.set(m.output.W_out.weights.get() * 40.0)
    m.embedding.W_embed.word_weights.set(m.embedding.W_embed.word_weights.get() * 20.0)
    return m


def _ref_probs(m, window):
    S, B = m.seq_len, m.batch_size
    n = len(window)
    inp = jnp.array([window + [0] * (S - n)], dtype=jnp.int32)
    y, _, _ = m.process(inp, jnp.zeros((B * S, VOC)), adapt_synapses=False)
    return np.array(y.reshape(B, S, VOC)[0, n - 1])


def test_equivalence():
    m = _build()
    S = m.seq_len
    H = config.n_heads
    P = F.extract_params(m)
    seq = np.random.default_rng(0).integers(1, VOC, size=60).tolist()

    # 1) incremental KV-cache steps (t < seq_len) == original model
    K, V = F.init_cache(len(P["layers"]), H, S, config.n_embed // H)
    for t in range(S):
        lg, K, V = F.decode_step(P, K, V, jnp.int32(seq[t]), jnp.int32(t), H)
        err = np.abs(np.array(jax.nn.softmax(lg)) - _ref_probs(m, seq[: t + 1])).max()
        assert err < 1e-4, (t, err)

    # 2) sliding window (t >= seq_len): re-prefill == original model on the same window
    for t in (S, S + 8, 59):
        win = seq[t - S + 1 : t + 1]
        lg, _, _ = F.prefill(P, jnp.array(win, jnp.int32), jnp.int32(S), H)
        assert np.abs(np.array(jax.nn.softmax(lg)) - _ref_probs(m, win)).max() < 1e-4

    # 3) end-to-end greedy generation identical to the original sliding-window loop
    fast = F.generate(P, 0, 60, S, H)
    cur = [0]
    for _ in range(60):
        cur.append(int(np.argmax(_ref_probs(m, cur[-S:]))))
    assert fast == cur


def test_varied_configs():
    # 2 layers / 4 heads / seq_len 16 to prove nothing is hard-coded
    S = 16
    H = 4
    n_layers = 2
    n_embed = 128
    m = _build(n_layers=n_layers, n_heads=H, seq_len=S, n_embed=n_embed)
    P = F.extract_params(m)
    seq = np.random.default_rng(42).integers(1, VOC, size=30).tolist()

    K, V = F.init_cache(len(P["layers"]), H, S, n_embed // H)
    for t in range(S):
        lg, K, V = F.decode_step(P, K, V, jnp.int32(seq[t]), jnp.int32(t), H)
        err = np.abs(np.array(jax.nn.softmax(lg)) - _ref_probs(m, seq[: t + 1])).max()
        assert err < 1e-4, (t, err)

    fast = F.generate(P, 0, 30, S, H)
    cur = [0]
    for _ in range(30):
        cur.append(int(np.argmax(_ref_probs(m, cur[-S:]))))
    assert fast == cur


def test_config_mismatch_assertions():
    # P2: Assert/clear error that n_embed % n_heads == 0 and pos_weights >= seq_len
    m = _build()

    # Mismatch 1: n_embed not divisible by n_heads
    m.n_heads = 7
    try:
        F.extract_params(m)
        assert False, "Should have raised ValueError for invalid n_heads"
    except ValueError as e:
        assert "divisible" in str(e)
    m.n_heads = config.n_heads

    # Mismatch 2: pos_weights rows < seq_len
    old_pos = m.embedding.W_embed.pos_weights.get()
    m.embedding.W_embed.pos_weights.set(old_pos[: m.seq_len - 1])
    try:
        F.extract_params(m)
        assert False, "Should have raised ValueError for pos_weights < seq_len"
    except ValueError as e:
        assert "pos_weights rows" in str(e)
    m.embedding.W_embed.pos_weights.set(old_pos)


def test_generate_text_routing():
    m = _build()
    class DummyTokenizer:
        def decode(self, ids):
            return ids

    tok = DummyTokenizer()
    # Fast path greedy
    out_fast = generate_text(m, tok, max_new_tokens=40, temperature=1.0, top_k=0, key=None, use_kv_cache=True)
    # Legacy path greedy
    out_legacy = generate_text(m, tok, max_new_tokens=40, temperature=1.0, top_k=0, key=None, use_kv_cache=False)
    assert out_fast == out_legacy, f"Greedy generation mismatch between fast and legacy: {out_fast} != {out_legacy}"


if __name__ == "__main__":
    test_equivalence()
    print("OK - test_equivalence matches NGCTransformer.process()")
    test_varied_configs()
    print("OK - test_varied_configs matches NGCTransformer.process()")
    test_config_mismatch_assertions()
    print("OK - test_config_mismatch_assertions verified successfully")
    test_generate_text_routing()
    print("OK - test_generate_text_routing verified successfully")

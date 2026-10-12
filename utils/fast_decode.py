"""
fast_decode.py - true KV-cache decoder for NGCTransformer (inference only).

Pure JAX, jit-friendly, fixed shapes. Reads the weights from a loaded NGCTransformer and
re-implements ONLY the feed-forward "project" path used by generation:

  tok -> word_emb[tok] + pos_emb[pos]
  per layer: q,k,v = x@Wq,Wk,Wv -> causal MHA -> @Wo -> @W1 -> gelu(1.702) -> @W2  (no residual / no norm)
  logits = x @ W_out + b_out  (softmax applied here, not inside the model)
"""
from functools import partial
import jax
import jax.numpy as jnp


def extract_params(model):
    g = lambda c: (c.weights.get(), c.biases.get().reshape(-1))
    layers = []
    for blk in model.blocks:
        a, mlp = blk.attention, blk.mlp
        layers.append(
            dict(zip(("Wq", "bq"), g(a.W_q)))
            | dict(zip(("Wk", "bk"), g(a.W_k)))
            | dict(zip(("Wv", "bv"), g(a.W_v)))
            | dict(zip(("Wo", "bo"), g(a.W_attn_out)))
            | dict(zip(("W1", "b1"), g(mlp.W_mlp1)))
            | dict(zip(("W2", "b2"), g(mlp.W_mlp2)))
        )
    Wout, bout = g(model.output.W_out)
    word_weights = model.embedding.W_embed.word_weights.get()
    pos_weights = model.embedding.W_embed.pos_weights.get()

    n_embed = word_weights.shape[1]
    if hasattr(model, "n_heads") and model.n_heads is not None:
        if n_embed % model.n_heads != 0:
            raise ValueError(f"n_embed={n_embed} must be divisible by n_heads={model.n_heads}")
    if hasattr(model, "seq_len") and model.seq_len is not None:
        if pos_weights.shape[0] < model.seq_len:
            raise ValueError(f"pos_weights rows ({pos_weights.shape[0]}) must be >= seq_len ({model.seq_len})")

    return dict(
        word=word_weights,
        pos=pos_weights,
        layers=layers,
        Wout=Wout,
        bout=bout,
    )


def _gelu(x):  # ngclearn's "gelu" = swish(beta=1.702)
    return x * jax.nn.sigmoid(1.702 * x)


def init_cache(n_layers, n_heads, seq_len, d_head, dtype=jnp.float32):
    z = jnp.zeros((n_layers, n_heads, seq_len, d_head), dtype)
    return z, z  # K, V (fixed shape; never grows, never re-compiles)


@partial(jax.jit, static_argnames=("n_heads",))
def decode_step(params, K, V, tok, pos, n_heads):
    """One token. tok: int32 scalar, pos: int32 scalar (absolute position, 0..S-1). O(S) attention, O(1) matmuls."""
    S = K.shape[2]
    D = params["word"].shape[1]
    dh = D // n_heads
    x = params["word"][tok] + params["pos"][pos]  # (D,)
    visible = jnp.arange(S) <= pos  # causal mask over cache slots
    newK, newV = [], []
    for l, p in enumerate(params["layers"]):
        q = (x @ p["Wq"] + p["bq"]).reshape(n_heads, dh)
        k = (x @ p["Wk"] + p["bk"]).reshape(n_heads, 1, dh)
        v = (x @ p["Wv"] + p["bv"]).reshape(n_heads, 1, dh)
        Kl = jax.lax.dynamic_update_slice(K[l], k, (0, pos, 0))  # write slot `pos`
        Vl = jax.lax.dynamic_update_slice(V[l], v, (0, pos, 0))
        s = jnp.einsum("he,hse->hs", q, Kl) / jnp.sqrt(dh)
        s = jnp.where(visible[None, :], s, -1e9)
        a = jnp.einsum("hs,hse->he", jax.nn.softmax(s, -1), Vl).reshape(D)
        x = _gelu((a @ p["Wo"] + p["bo"]) @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]
        newK.append(Kl)
        newV.append(Vl)
    logits = x @ params["Wout"] + params["bout"]
    return logits, jnp.stack(newK), jnp.stack(newV)


@partial(jax.jit, static_argnames=("n_heads",))
def prefill(params, toks, n, n_heads):
    """Full causal forward over a padded window toks:(S,), n = #real tokens. Returns last-real-token logits + filled cache."""
    S = toks.shape[0]
    D = params["word"].shape[1]
    dh = D // n_heads
    x = params["word"][toks] + params["pos"][:S]  # (S,D)
    causal = jnp.tril(jnp.ones((S, S), bool))
    Ks, Vs = [], []
    for p in params["layers"]:
        q = (x @ p["Wq"] + p["bq"]).reshape(S, n_heads, dh).transpose(1, 0, 2)
        k = (x @ p["Wk"] + p["bk"]).reshape(S, n_heads, dh).transpose(1, 0, 2)
        v = (x @ p["Wv"] + p["bv"]).reshape(S, n_heads, dh).transpose(1, 0, 2)
        s = jnp.where(causal[None], jnp.einsum("hte,hse->hts", q, k) / jnp.sqrt(dh), -1e9)
        a = jnp.einsum("hts,hse->hte", jax.nn.softmax(s, -1), v).transpose(1, 0, 2).reshape(S, D)
        x = _gelu((a @ p["Wo"] + p["bo"]) @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]
        Ks.append(k)
        Vs.append(v)
    last = x[n - 1]  # only ONE row goes through the output head
    return last @ params["Wout"] + params["bout"], jnp.stack(Ks), jnp.stack(Vs)


def sample(logits, key, temperature=1.0, top_k=0):
    """Correct sampling: temperature on LOGITS (not on already-softmaxed probabilities)."""
    if key is None:
        return jnp.argmax(logits), key
    logits = logits / temperature
    key, sub = jax.random.split(key)
    if top_k and top_k > 0:
        vals, idx = jax.lax.top_k(logits, min(top_k, logits.shape[-1]))
        return idx[jax.random.categorical(sub, vals)], key
    return jax.random.categorical(sub, logits), key


def generate(params, start_id, max_new_tokens, seq_len, n_heads, temperature=1.0, top_k=0, key=None, pad_id=0):
    """Exact w.r.t. the original sliding-window semantics:
    t < seq_len : true incremental KV cache (1 token per step)
    t >= seq_len: window slides -> absolute positions shift -> cache invalid -> re-prefill last seq_len tokens."""
    n_layers = len(params["layers"])
    d_head = params["word"].shape[1] // n_heads
    K, V = init_cache(n_layers, n_heads, seq_len, d_head)
    toks = [int(start_id)]
    logits, K, V = decode_step(params, K, V, jnp.int32(start_id), jnp.int32(0), n_heads)
    for _ in range(max_new_tokens):
        nxt, key = sample(logits, key, temperature, top_k)
        nxt = int(nxt)
        toks.append(nxt)
        if len(toks) <= seq_len:
            logits, K, V = decode_step(params, K, V, jnp.int32(nxt), jnp.int32(len(toks) - 1), n_heads)
        else:
            win = toks[-seq_len:]
            logits, K, V = prefill(params, jnp.array(win, jnp.int32), jnp.int32(seq_len), n_heads)
    return toks

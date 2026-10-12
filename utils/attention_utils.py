from ngclearn.components.jaxComponent import JaxComponent
from ngclearn import Compartment 
from jax import numpy as jnp, random, jit
from ngclearn import compilable
import jax
from functools import partial
import jax.numpy as jnp
from utils.model_util import d_softmax_vjp

@partial(jit, static_argnums=[4, 5, 6, 7, 8])
def _compute_attention(Q, K, V, mask, n_heads, d_head, dropout_rate, seq_len, batch_size, key):
    """
    Compute multi-head attention 
    """
    B = batch_size
    S = seq_len
    D = Q.shape[-1]
    
    # Q_3d = Q.reshape(B, S, D)
    # K_3d = K.reshape(B, S, D)
    # V_3d = V.reshape(B, S, D)
    
    # Reshape for multi-head attention
    q = Q.reshape((B, S, n_heads, d_head)).transpose([0, 2, 1, 3])
    k = K.reshape((B, S, n_heads, d_head)).transpose([0, 2, 1, 3]) 
    v = V.reshape((B, S, n_heads, d_head)).transpose([0, 2, 1, 3])
    # Scaled dot-product attention
    s_c = jnp.einsum("BHTE,BHSE->BHTS", q, k) / jnp.sqrt(d_head)
    
  
    _mask = mask[None, None, :, :]  
    s_c = jnp.where(_mask, s_c, -1e9)
        
    score = jax.nn.softmax(s_c, axis=-1)
    score = score.astype(q.dtype)
    
    if dropout_rate > 0.0:
        dkey = random.fold_in(key, 0)
        # dkey = random.PRNGKey(0)
        score = jax.random.bernoulli(dkey, 1 - dropout_rate, score.shape) * score / (1 - dropout_rate)
        
    attention = jnp.einsum("BHTS,BHSE->BHTE", score, v)
    attention = attention.transpose([0, 2, 1, 3]).reshape((B, S, -1))
    
    return attention, s_c, q, k, v


@partial(jit, static_argnums=[6, 7, 8, 9, 10])
def compute_grads(Q, K, V, mask, s_c, dmu, n_heads, d_head, dropout_rate, seq_len, batch_size, key):
    """Compute gradients for Q, K, V using d_softmax and attention scores
    """
    B = batch_size
    S = seq_len
    H = n_heads
    D = d_head
    
    P, jvp_fn = d_softmax_vjp(s_c, tau=0.0)  # P: (B, H, S, S)
    
    # Reshape error
    dmu_reshaped = dmu.reshape(B, S, H, D).transpose(0, 2, 1, 3)  ## (B, H, S, D)
    dV = jnp.einsum("bhkq,bhkd->bhqd", P, dmu_reshaped)           ## (B, H, S, D)
    da = jnp.einsum("bhte,bhse->bhts", dmu_reshaped, V)           ## (B, H, S, D)
    ds = jvp_fn(da)  
    ds = ds / jnp.sqrt(D)
    
    _mask = mask[None, None, :, :]                                ## (1, 1, S, S)
    ds = jnp.where(_mask, ds, 0.)

    dQ = jnp.einsum("bhts,bhsd->bhtd", ds, K)                     ## (B, H, T, D)
    dK = jnp.einsum("bhts,bhtd->bhsd", ds, Q)                     ## (B, H, S, D)
    
    # 6. Reshape to flattened format
    dq = dQ.transpose(0, 2, 1, 3).reshape(B * S, H * D)
    dk = dK.transpose(0, 2, 1, 3).reshape(B * S, H * D)
    dv = dV.transpose(0, 2, 1, 3).reshape(B * S, H * D)
    
    return dq, dk, dv
    
class AttentionBlock(JaxComponent):
    """
    Multi-head attention block for NGC attention.
    
    Takes Q, K, V inputs and computes scaled dot-product attention 
    with optional masking and dropout.
    
    | --- Compartments: ---
    | inputs_q - query inputs
    | inputs_k - key inputs  
    | inputs_v - value inputs
    | outputs - attention outputs
    | key - JAX PRNG key
    | dq - gradient w.r.t. query inputs
    | dk - gradient w.r.t. key inputs
    | dv - gradient w.r.t. value inputs
    | dmu - = gradient w.r.t. mu inputs

    Args:
        name: Component name
        n_heads: Number of attention heads
        n_embed: Embedding dimension
        seq_len: Sequence length
        dropout_rate: Attention dropout rate
        batch_size: Batch size
    """
    
    def __init__(self, name, n_heads, n_embed, seq_len, dropout_rate, batch_size, **kwargs):
        super().__init__(name, **kwargs)

        self.n_heads = n_heads
        self.n_embed = n_embed
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        
        if self.n_embed % self.n_heads != 0:
            raise ValueError(f"n_embed={n_embed} must be divisible by n_heads={n_heads}")
        self.d_head = n_embed // n_heads

        # Input compartments
        self.inputs_q = Compartment(jnp.zeros((batch_size, seq_len, n_embed)))
        self.inputs_k = Compartment(jnp.zeros((batch_size, seq_len, n_embed)))
        self.inputs_v = Compartment(jnp.zeros((batch_size, seq_len, n_embed)))
        self.dq = Compartment(jnp.zeros((batch_size*seq_len, n_embed)))
        self.dk = Compartment(jnp.zeros((batch_size*seq_len, n_embed)))
        self.dv = Compartment(jnp.zeros((batch_size*seq_len, n_embed)))
        self.dmu = Compartment(jnp.zeros((batch_size * seq_len, n_embed)))
        self.key = Compartment(random.PRNGKey(0))
        # Output compartment
        self.outputs = Compartment(jnp.zeros((batch_size, seq_len, n_embed)))    
    
    @compilable
    def advance_state(self):
        """
        Compute multi-head attention
        and compute gradients for inputs and Jacobian using d_softmax
        """
        inputs_q=self.inputs_q.get()
        inputs_k=self.inputs_k.get()
        inputs_v=self.inputs_v.get()
        mask=self.causal_mask
        # S=self.S.get()
        dmu=self.dmu.get()
        n_heads=self.n_heads
        d_head=self.d_head
        dropout_rate=self.dropout_rate
        key=self.key.get()
        attention, s_c, q, k, v = _compute_attention(
            inputs_q, inputs_k, inputs_v, mask,        
            self.n_heads,        
            self.d_head,        
            self.dropout_rate, 
            self.seq_len,
            self.batch_size,  
            key                  
        )
        # self.S.set(S)
        dq, dk, dv = compute_grads(q, k, v, mask, s_c, dmu, n_heads, d_head, dropout_rate, self.seq_len, self.batch_size, key)
        self.dq.set(dq)
        self.dk.set(dk)
        self.dv.set(dv)
        self.outputs.set(attention)
  
    @compilable
    def reset(self):
        """
        Reset compartments to zeros
        """
        batch_size=self.batch_size
        seq_len=self.seq_len
        n_embed=self.n_embed
        zeros_2d = jnp.zeros((batch_size*seq_len, n_embed))
        zeros_3d = jnp.zeros((batch_size, seq_len, n_embed))
        # return zeros_3d, zeros_3d, zeros_3d, zeros_3d
        self.inputs_q.set(zeros_3d)
        self.inputs_k.set(zeros_3d)
        self.inputs_v.set(zeros_3d)
        self.outputs.set(zeros_3d)
        self.dq.set(zeros_2d)
        self.dk.set(zeros_2d)
        self.dv.set(zeros_2d)
        self.dmu.set(zeros_2d)

    @classmethod
    def help(cls):
        """Component help function"""
        properties = {
            "component_type": "AttentionBlock - multi-head self-attention with built-in causal mask"
        }
        compartment_props = {
            "inputs": 
                {"inputs_q": "Query inputs (batch_size, seq_len, n_embed)",
                "inputs_k": "Key inputs (batch_size, seq_len, n_embed)", 
                "inputs_v": "Value inputs (batch_size, seq_len, n_embed)"},
            "gradients":
                {"dq": "Gradient w.r.t Q (batch_size*seq_len, n_embed)",
                "dk": "Gradient w.r.t K (batch_size*seq_len, n_embed)",
                "dv": "Gradient w.r.t V (batch_size*seq_len, n_embed)",
                "dmu": "Gradient w.r.t mu (batch_size*seq_len, n_embed)"},
            "outputs":
                {"outputs": "Attention outputs (batch_size, seq_len, n_embed)"},
        }
        hyperparams = {
            "n_heads": "Number of attention heads",
            "n_embed": "Embedding dimension", 
            "seq_len": "Sequence length",
            "dropout_rate": "Attention dropout rate",
            "batch_size": "Batch size dimension"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "outputs = MultiHeadAttention(Q, K, V) with built-in causal mask",
                "hyperparameters": hyperparams}
        return info

from jax import random
import jax.numpy as jnp
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn import Compartment
from ngclearn import compilable


class RandomInit(JaxComponent):
    """NGC-compatible random state initializer for latent z tensors.
    
    Generates fresh random values each time advance_state() is called.
    Access the compartments directly:
    - z_qkv: shape (batch_size, n_embed)
    - z_attn: shape (batch_size, n_embed)
    - z_mlp: shape (batch_size, n_embed)
    - z_out: shape (batch_size, n_embed)
    - z_mlp2: shape (batch_size, 4 * n_embed)
    """


    def __init__(self, name, batch_size, n_embed, scale=1e-3, key=None, **kwargs):
        super().__init__(name, **kwargs)
        self.batch_size = batch_size
        self.n_embed = n_embed
        self.scale = scale
        self.key = Compartment(random.PRNGKey(0) if key is None else key)
        self.z_qkv = Compartment(jnp.zeros((batch_size, n_embed)))
        self.z_attn = Compartment(jnp.zeros((batch_size, n_embed)))
        self.z_mlp = Compartment(jnp.zeros((batch_size, n_embed)))
        self.z_out = Compartment(jnp.zeros((batch_size, n_embed)))
        self.z_mlp2 = Compartment(jnp.zeros((batch_size, 4 * n_embed)))


    @compilable
    def advance_state(self):
        """Generate fresh random values for all latent compartments."""
        key = self.key.get()
        k1, k2, k3, k4, k5, next_key = random.split(key, 6)
        z_qkv = random.normal(k1, (self.batch_size, self.n_embed)) * self.scale
        z_attn = random.normal(k2, (self.batch_size, self.n_embed)) * self.scale
        z_mlp = random.normal(k3, (self.batch_size, self.n_embed)) * self.scale
        z_out = random.normal(k4, (self.batch_size, self.n_embed)) * self.scale
        z_mlp2 = random.normal(k5, (self.batch_size, 4 * self.n_embed)) * self.scale
        self.key.set(next_key)
        self.z_qkv.set(z_qkv)
        self.z_attn.set(z_attn)
        self.z_mlp.set(z_mlp)
        self.z_out.set(z_out)
        self.z_mlp2.set(z_mlp2)
"""
Scaled Xavier (Glorot) Uniform weight initializer for Predictive Coding Networks.

Standard Xavier variance (σ² = 2 / (fan_in + fan_out)) can produce initial
weights that are too large for the iterative relaxation dynamics of NGC-PC
Transformers. This leads to oversized prediction errors on the first few
inference steps, causing Hebbian updates to diverge or saturate rate cells.

This module provides a *scaled-down* Xavier Uniform initializer that keeps
initial weights within a safe amplitude range, giving the predictive coding
circuit a gentle start while still breaking symmetry.

Usage:
    from utils.init_utils import scaled_xavier_uniform
    weight_init = scaled_xavier_uniform(scale=0.25)
    HebbianSynapse(..., weight_init=weight_init, ...)
"""
import jax
import jax.numpy as jnp


def scaled_xavier_uniform(scale=0.25):
    """Return a JAX-compatible weight initializer using scaled Xavier Uniform.

    The returned callable has signature ``(shape, key) -> jnp.ndarray``,
    matching the interface expected by ngclearn's ``weight_init`` parameter.

    Weights are sampled from U(-limit, +limit) where:
        limit = scale * sqrt(6 / (fan_in + fan_out))

    Args:
        scale: Multiplicative scaling factor applied to the standard Xavier
            limit. Values in [0.1, 0.5] are recommended for PC networks.
            Default is 0.25, which keeps initial weights approximately 4×
            smaller than standard Xavier, reducing early relaxation energy.

    Returns:
        A callable ``(shape, key) -> jnp.ndarray``.
    """
    def _init(shape, key):
        fan_in = shape[0]
        fan_out = shape[1] if len(shape) > 1 else shape[0]
        limit = scale * jnp.sqrt(6.0 / (fan_in + fan_out))
        return jax.random.uniform(key, shape, minval=-limit, maxval=limit)
    return _init

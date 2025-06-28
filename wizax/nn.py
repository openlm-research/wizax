"""Neural network utilities."""
from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp


def init_normal(rng, shape, scale=1.0, dtype=jnp.float32, scaling_mode='fan_out'):
    """Initialize weights with a normal distribution."""
    if scaling_mode == 'fan_in':
        scale = scale / np.sqrt(shape[-2])  # Scale by input dimension size
    elif scaling_mode == 'fan_out':
        scale = scale / np.sqrt(shape[-1])  # Scale by output dimension size
    elif scaling_mode == 'constant':
        pass  # Use the constant scale directly
    else:
        raise ValueError(f"Unknown scaling mode: {scaling_mode}")
    return scale * jax.random.normal(key=rng, shape=shape, dtype=dtype)

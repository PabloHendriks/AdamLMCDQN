from typing import NamedTuple

import jax
import jax.numpy as jnp

import chex
import optax


class _LangevinAdamState(NamedTuple):
    count: jnp.ndarray
    m: optax.Params
    v: optax.Params
    rng: jax.Array

def langevin_adam(
        base_rng,
        learning_rate: float = 1e-3,
        alpha1: float = 0.9,
        alpha2: float = 0.999,
        eps: float = 1e-8,
        inverse_temperature: int = 10e5,
        a: float = 0.1,
    ):
    """
    Implements Adam with injected Langevin noise as proposed in https://arxiv.org/abs/2305.18246.
    """
    def init_fn(params):
        m = jax.tree.map(jnp.zeros_like, params)
        v = jax.tree.map(jnp.zeros_like, params)
        return _LangevinAdamState(count=jnp.zeros([], jnp.int32), m=m, v=v, rng=base_rng)


    def update_fn(updates, state, params=None):
        # step counter
        count = state.count + 1

        # standard Adam update
        adam_step = jax.tree.map(
            lambda m_h, v_h, g: -learning_rate * (g + 0.3 * m_h / (jnp.sqrt(v_h) + eps)),
            state.m, state.v, updates
        )

        # update first and second moments
        m = jax.tree.map(
            lambda m, g: alpha1 * m + (1 - alpha1) * g,
            state.m,
            updates
        )
        v = jax.tree.map(
            lambda v, g: alpha2 * v + (1 - alpha2) * (g * g),
            state.v,
            updates
        )

        # bias‐corrected moments TODO check if this matters
        # m_hat = jax.tree.map(lambda m: m / (1 - b1 ** count), m)
        # v_hat = jax.tree.map(lambda v: v / (1 - b2 ** count), v)

        # Langevin noise term
        rng, _rng = jax.random.split(state.rng)

        leaves, treedef = jax.tree.flatten(v)
        keys = jax.random.split(_rng, len(leaves))
        keys_tree = jax.tree.unflatten(treedef, keys)

        noises = jax.tree.map(
            lambda leaf, key: jax.random.normal(key, leaf.shape),
            v, keys_tree
        )

        noise_term = jax.tree.map(
            lambda n: jnp.sqrt(2.0 * learning_rate * 1 / inverse_temperature) * n,
            noises
        )

        # total update is Adam step plus noise
        total_update = jax.tree.map(
            lambda adam, noise: adam + noise,
            adam_step, noise_term
        )

        new_state = _LangevinAdamState(count=count, m=m, v=v, rng=rng)
        return total_update, new_state

    return optax.GradientTransformation(init_fn, update_fn)
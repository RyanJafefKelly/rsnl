"""Implementation of the misspecified MA(1) model."""

from typing import Optional

import jax.numpy as jnp
import numpyro  # type: ignore
import numpyro.distributions as dist  # type: ignore
from jax import random
from jax._src.prng import PRNGKeyArray  # for typing

from rsnl.utils import FlowNumpyro


def assumed_dgp(rng_key, t1, n_obs=100):
    w = dist.Normal(0, 1).sample(key=rng_key, sample_shape=(1, n_obs + 2))  # NOTE: removed batch_size
    x = w[:, 2:] + t1 * w[:, 1:-1]
    return x


def autocov(x, lag=1):
    x = jnp.atleast_2d(x)
    # In R this is normalized with x.shape[1]
    if lag == 0:
        C = jnp.mean(x[:, :] ** 2, axis=1)
    else:
        C = jnp.mean(x[:, lag:] * x[:, :-lag], axis=1)

    return C


def calculate_summary_statistics(x):
    s0 = autocov(x, lag=0)
    s1 = autocov(x, lag=1)
    return jnp.squeeze(jnp.array([s0, s1]))


def get_prior():
    """Return prior for inference on misspec MA(1)"""
    return dist.Uniform(low=jnp.array([-1.0]),
                        high=jnp.array([1.0]))


# TODO: TERRIBLE IMPLEMENTATION ... RUN ONCE AT START SO WHO CARES
# @partial(jit, static_argnums=(3,4))
def true_dgp(w=-0.736, rho=0.9, sigma_v=0.36, batch_size=1, n_obs=100, key=None):
    """Generate data from true DGP."""
    h_mat = jnp.zeros((batch_size, n_obs))
    y_mat = jnp.zeros((batch_size, n_obs))

    w_vec = jnp.repeat(w, batch_size)
    rho_vec = jnp.repeat(rho, batch_size)
    sigma_v_vec = jnp.repeat(sigma_v, batch_size)

    key, subkey = random.split(key)
    h_mat = h_mat.at[:, 0].set(w_vec + dist.Normal(0, 1).sample(key=subkey, sample_shape=(batch_size,)) * sigma_v_vec)
    key, subkey = random.split(key)
    y_mat = y_mat.at[:, 0].set(jnp.exp(h_mat[:, 0]/2) * dist.Normal(0, 1).sample(key=subkey, sample_shape=(batch_size,)))

    # TODO?
    for i in range(1, n_obs):
        key, subkey = random.split(key)
        h_mat = h_mat.at[:, i].set(w_vec + rho_vec * h_mat[:, i-1] + dist.Normal(0, 1).sample(key=subkey, sample_shape=(batch_size,)) * sigma_v_vec)
        key, subkey = random.split(key)
        y_mat = y_mat.at[:, i].set(jnp.exp(h_mat[:, i]/2)*dist.Normal(0, 1).sample(key=subkey, sample_shape=(batch_size,)))

    return y_mat


def true_posterior(x_obs: jnp.ndarray,
                   prior: dist.Distribution) -> jnp.ndarray:
    """Return true posterior for misspecified MA(1) example."""
    pass

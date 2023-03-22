"""Implementation of the misspecified MA(1) model."""

from typing import Optional

import jax.numpy as jnp
import numpyro  # type: ignore
import numpyro.distributions as dist  # type: ignore
from jax import random
from jax._src.prng import PRNGKeyArray  # for typing

from rsnl.utils import FlowNumpyro


def assumed_dgp(t1, batch_size=1, n_obs=100, key=None):
    w = dist.Normal(0, 1).sample(key=key, sample_shape=(batch_size, n_obs + 2))
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


def summ_stats(x):
    s0 = autocov(x, lag=0)
    s1 = autocov(x, lag=1)
    return jnp.squeeze(jnp.array([s0, s1]))


def get_model():
    def model(x_obs: jnp.ndarray,
              flow: Optional[FlowNumpyro] = None,
              laplace_var:  Optional[jnp.ndarray] = None,
              standardisation_params=None) -> jnp.ndarray:
        prior = dist.Uniform(low=-1.0, high=1.0)
    return model

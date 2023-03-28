"""Implementation of the misspecified MA(1) model."""

from typing import Optional

import jax.numpy as jnp
import numpyro  # type: ignore
import numpyro.distributions as dist  # type: ignore
from jax import random
from jax._src.prng import PRNGKeyArray  # for typing

from rsnl.utils import FlowNumpyro


def assumed_dgp(t1, n_obs=100, key=None):
    w = dist.Normal(0, 1).sample(key=key, sample_shape=(1, n_obs + 2))  # NOTE: removed batch_size
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
    return dist.Uniform(low=-1.0, high=1.0)

# def get_model():
#     """Return numpyro model for inference on misspec MA(1)"""
#     def model(x_obs: jnp.ndarray,
#               flow: Optional[FlowNumpyro] = None,
#               laplace_var:  Optional[jnp.ndarray] = None,
#               standardisation_params=None) -> jnp.ndarray:
#         prior = dist.Uniform(low=-1.0, high=1.0)

#         laplace_mean = jnp.array([0.0, 0.0])

#         if laplace_var is None:
#             laplace_var = jnp.array([1.0, 1.0])

#         theta = numpyro.sample('theta', prior)
#         theta_standard = numpyro.deterministic('theta_standard', (theta - standardisation_params['theta_mean']) / standardisation_params['theta_std'])

#         adj_params = numpyro.sample('adj_params', dist.Laplace(laplace_mean,
#                                                                laplace_var))
#         x_adj = numpyro.deterministic('x_adj', x_obs - adj_params)

#         if flow is not None:  # TODO?
#             x_adj_sample = numpyro.sample('x_adj_sample',
#                                           FlowNumpyro(flow, theta=theta_standard),
#                                           obs=x_adj)
#         else:
#             x_adj_sample = x_adj

#         return x_adj_sample


#     return model

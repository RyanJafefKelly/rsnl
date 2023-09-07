"""Implementation of the Sparse vector autoregression model."""

import jax.numpy as jnp
import numpyro.distributions as dist  # type: ignore
from jax import random
from jax._src.prng import PRNGKeyArray  # for typing

import numpy as np

# global ... avoid annoying file reading
pairs = ([0,5], [3,1], [4,2])  # NOTE: -1 error in original code


def true_dgp(key: PRNGKeyArray,
             *theta: tuple,
             n_obs : int = 1000,
             ):
    theta = np.array(theta).flatten()
    key, subkey = random.split(key)
    n_obs = 1000
    num_dims = 6
    X = -0.1 * np.eye(num_dims)
    sigma = theta[-1]
    # pairs = None
    # index: for pair in pairs     put in corresponding theta
    for ii, pair in enumerate(pairs):
        X[pair[0], pair[1]] = theta[ii*2]
        X[pair[1], pair[0]] = theta[ii*2+1]

    Y = np.zeros((num_dims, n_obs))

    Y[:, 0] = np.random.multivariate_normal(np.zeros(num_dims),
                                            sigma * np.eye(num_dims))

    for t in range(1, n_obs):
        # key, subkey = random.split(key)
        Y[:, t] = np.matmul(X, Y[:, t-1]) + np.random.multivariate_normal(np.zeros(num_dims),
                                                              sigma * np.eye(num_dims))

    return Y


def calculate_summary_statistics(Y: jnp.ndarray):
    S = np.zeros(7)  # TODO? magic numbers

    for ii, pair in enumerate(pairs):
        S[ii*2] = jnp.mean(Y[pair[0], 1:] * Y[pair[1], 0:-1])
        S[ii*2+1] = jnp.mean(Y[pair[1], 1:] * Y[pair[0], 0:-1])

    S[-1] = jnp.std(Y.flatten())
    S = jnp.array(S)

    # NOTE: Sometimes getting inf, neginf, nan values
    # NOTE: Just choose poor summaries, so will learn these points are bad,
    #       and not use them. But magnitude of these values do not break
    #       the standardisation.
    S = jnp.nan_to_num(S, nan=-50)
    S = jnp.clip(S, a_min=-50, a_max=50)

    return S


def get_prior():
    """Return prior for the sparse vector autoregression example."""
    num_dims = 6
    prior = dist.Uniform(low=jnp.concatenate((jnp.atleast_2d(jnp.repeat(-1.0, num_dims)),
                                              jnp.atleast_2d(jnp.array([0.0]))), axis=1).flatten(),
                         high=jnp.repeat(1.0, num_dims+1))
    return prior


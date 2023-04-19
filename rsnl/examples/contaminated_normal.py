"""Implementation of the contaminated normal example."""

import jax.numpy as jnp
import numpyro.distributions as dist  # type: ignore
from jax import random
from jax._src.prng import PRNGKeyArray  # for typing


def true_dgp(key: PRNGKeyArray,
             t1: jnp.ndarray,
             stdev_err: float = 2.0,
            #  batch_size: int = 1,
             n_obs: int = 100,
             ) -> jnp.ndarray:
    """_summary_

    Args:
        t1 (jnp.ndarray): _description_
        y (jnp.ndarray): _description_
        key (PRNGKeyArray): _description_
        stdev_err (float, optional): _description_. Defaults to 2.0.
        batch_size (int, optional): _description_. Defaults to 1.
        n_obs (int, optional): _description_. Defaults to 100.

    Returns:
        jnp.ndarray: _description_
    """
    w = 0.8  # i.e. 20% of samples are contaminated
    std_devs = random.choice(key,
                             jnp.array([1.0, stdev_err]),
                             shape=(1, n_obs),  # NOTE: removed batch_size
                             p=jnp.array([w, 1-w]))
    key, sub_key = random.split(key)
    standard_y = dist.Normal(0, 1).sample(sub_key, sample_shape=(1,  # NOTE: removed batch_size
                                                                 n_obs))
    y = (standard_y * std_devs) + t1

    return y


def assumed_dgp(key: PRNGKeyArray,
                t1: jnp.ndarray,
                # batch_size: int = 1,
                n_obs: int = 100) -> jnp.ndarray:
    """Assumed DGP - only considering mean."""
    return dist.Normal(t1, 1).sample(key=key, sample_shape=(1, n_obs))  # NOTE: removed batch_size


# @jit
def calculate_summary_statistics(x):
    """Calculate summary statistics for contaminated normal example."""
    s0 = jnp.mean(x, axis=1)
    s1 = jnp.var(x, axis=1, ddof=1)  # ddof ->  divisor used in the calculation is N - ddof

    return jnp.hstack((s0, s1))


def get_prior():
    """Return prior for contaminated normal example."""
    return dist.Normal(jnp.array([0.0]), jnp.array([10.0]))


def true_posterior(x_obs: jnp.ndarray,
                   prior: dist.Distribution) -> jnp.ndarray:
    """Return true posterior for contaminated normal example.

    Args:
        x_obs (jnp.ndarray): _description_
        prior (dist.Distribution): _description_

    Returns:
        jnp.ndarray: _description_
    """
    # Conjugate prior with known variance
    n_obs = 100
    true_dgp_var = 1.0
    prior_var = prior.variance
    obs_mean = x_obs[0]

    true_post_var = (1/prior_var + n_obs/true_dgp_var) ** -1
    true_post_mu = (true_post_var *
                    (prior.mean/prior_var + ((obs_mean * n_obs) / true_dgp_var)))
    true_post_std = jnp.sqrt(true_post_var)
    return dist.Normal(true_post_mu, true_post_std)

"""Implementation of the Marchand toad model.

This model simulates the movement of Fowler's toad species.
"""

import jax.numpy as jnp
import numpyro.distributions as dist  # type: ignore
from jax import random
from jax._src.prng import PRNGKeyArray  # for typing
from jax import lax

import numpy as np
import scipy.stats as ss


def levy_stable(key, alpha, gamma, size=None):
    if size is None:
        size = jnp.shape(alpha)

    key1, key2, key3 = random.split(key, num=3)

    # General case
    u = random.uniform(key1, minval=-0.5*jnp.pi, maxval=0.5*jnp.pi, shape=size)
    v = random.exponential(key2, shape=size)
    t = jnp.sin(alpha * u) / (jnp.cos(u) ** (1 / alpha))
    s = (jnp.cos((1 - alpha) * u) / v) ** ((1 - alpha) / alpha)
    output = gamma * t * s

    # Handle alpha == 1
    # cauchy_sample = random.cauchy(key3, shape=size)
    # output = jnp.where(alpha == 1, cauchy_sample, output)

    # # Handle alpha == 2
    # normal_sample = random.normal(key3, shape=size) * jnp.sqrt(2) * gamma
    # output = jnp.where(alpha == 2, normal_sample, output)

    return output


def dgp(key: PRNGKeyArray,
        alpha: jnp.ndarray,
        gamma: jnp.ndarray,
        p0: jnp.ndarray,
        model: int = 1,
        n_toads: int = 66,
        n_days: int = 63,
        batch_size: int = 1
        ) -> jnp.ndarray:
    """Sample the movement of Fowler's toad species

    Returns:
        jnp.ndarray in shape (n_days x n_toads x batch_size)
    """
    X = jnp.zeros((n_days, n_toads, batch_size))

    # Generate step length from levy_stable distribution
    delta_x = levy_stable(key, alpha, gamma, size=(n_days, n_toads, batch_size))

    for i in range(1, n_days):
        # Generate random uniform samples for returns
        key, subkey = random.split(key)
        ret = random.uniform(subkey, shape=(n_toads, batch_size)) < jnp.squeeze(p0)

        # Indices where ret is True or False
        # true_indices = jnp.where(ret)
        # false_indices = jnp.where(~ret)

        # Calculate new positions for non-returning toads
        # update_values = X[i-1, false_indices] + delta_x[i, false_indices]
        # X = X.at[i, false_indices].set(update_values)


        # Calculate new positions for non-returning toads
        # delta_x = delta_x * jnp.array(non_ret, dtype=int)
        # X = X.at[i, ~ret].set(X[i-1, ~ret] + delta_x[i, ~ret])
        # Calculate new positions for all toads
        new_positions = X[i-1, :] + delta_x[i, :]


        # Handle returning toads
        key, subkey = random.split(key)
        if model == 1:
            ind_refuge = random.choice(subkey, jnp.arange(i), shape=(n_toads, batch_size))
        if model == 2:
            # xn - curr
            if i > 1:
                ind_refuge = jnp.argmin(jnp.abs(new_positions[i, :] - X[:i, :]), axis=0)
            else:
                ind_refuge = jnp.zeros((n_toads, batch_size), dtype=int)
        # Extract previous positions for updating
        update_values = X[ind_refuge, jnp.arange(n_toads)[:, None], :].reshape((-1, 1))

        # Boolean mask, broadcasting to shape (66, 1)
        ret_expanded = ret[:, :, None].reshape((-1, 1))

        # Combine new_positions and update_values for final_positions
        final_positions = jnp.where(ret_expanded, update_values, new_positions)

        X = X.at[i, :, :].set(final_positions)

    return X


def calculate_summary_statistics(X, real_data=False, nan_idx=None):
    ssx = jnp.concatenate([
        calculate_summary_statistics_lag(X, lag, real_data=real_data, nan_idx=nan_idx)
        for lag in [1, 2, 4, 8]
    ], axis=1)
    ssx = jnp.clip(ssx, -1e+5, 1e+5)  # NOTE: fix for some crazy results
    return ssx.flatten()


def calculate_summary_statistics_lag(X, lag, p=jnp.linspace(0, 1, 11), thd=10,
                                     real_data=False, nan_idx=None):
    """
    Compute summaries for toad model in JAX.

    Args:
        X: Output from dgp function.
        lag, p, thd: See original function.

    Returns:
        A tensor of shape (batch_size, len(p) + 1).
    """
    if nan_idx is not None:
        X = X.at[nan_idx].set(jnp.nan)

    disp = X[lag:, :] - X[:-lag, :]

    abs_disp = jnp.abs(disp)
    abs_disp = abs_disp.flatten()

    ret = abs_disp < thd
    num_ret = jnp.sum(ret, axis=0)

    abs_disp = jnp.where(ret, jnp.nan, abs_disp)

    abs_noret_median = jnp.nanmedian(abs_disp, axis=0)
    abs_noret_quantiles = jnp.nanquantile(abs_disp, p, axis=0)
    diff = jnp.diff(abs_noret_quantiles, axis=0)
    logdiff = jnp.log(jnp.maximum(diff, jnp.exp(-20)))

    ssx = jnp.vstack((num_ret, abs_noret_median, logdiff.reshape(-1, 1)))
    ssx = jnp.nan_to_num(ssx, nan=jnp.inf)

    return ssx.T


def get_prior():
    prior = dist.Uniform(low=jnp.array([1.0, 0.0, 0.0]),
                         high=jnp.array([2.0, 100.0, 0.9]))
    return prior

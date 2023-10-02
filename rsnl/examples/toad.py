"""Implementation of the Marchand toad model.

This model simulates the movement of Fowler's toad species.
"""

import jax.numpy as jnp
import numpyro.distributions as dist  # type: ignore
from jax import random
from jax._src.prng import PRNGKeyArray  # for typing

import numpy as np
import scipy.stats as ss


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

    # NOTE: no levy_stable in jax?
    # random_state = np.random.RandomState(123)
    # step_gen = ss.levy_stable
    # step_gen.random_state = random_state
    alpha_np = np.array(alpha)
    gamma_np = np.array(gamma)
    delta_x = ss.levy_stable.rvs(alpha_np, beta=0, scale=gamma_np,
                                 size=(n_days, n_toads, batch_size))
    delta_x = jnp.array(delta_x)

    for i in range(1, n_days):
        # Generate random uniform samples for returns
        key, subkey = random.split(key)
        ret = random.uniform(subkey, shape=(n_toads, batch_size)) < jnp.squeeze(p0)
        non_ret = ~ret

        # Generate step length from levy_stable distribution
        # key, subkey = random.split(key)
        # delta_x = dist.Stable(alpha, beta=0, scale=gamma).sample(subkey, sample_shape=(n_toads, batch_size))
        # Calculate new positions for non-returning toads
        X = X.at[i, non_ret].set(X[i-1, non_ret] + delta_x[i, non_ret])

        # Handle returning toads
        key, subkey = random.split(key)
        if model == 1:
            ind_refuge = random.choice(subkey, jnp.arange(i), shape=(n_toads, batch_size))
        if model == 2:
            # xn - curr
            if i > 1:
                ind_refuge = jnp.argmin(jnp.abs(X[i, :] - X[:i-1, :]), axis=0)
            else:
                ind_refuge = jnp.zeros((n_toads, batch_size), dtype=int)
        X = X.at[i, ret].set(X[ind_refuge[ret], ret])

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
    # TODO: add check here of X length...

    # if day_count is not None:
    #     toad_disp = []
    #     for ii, day_i in enumerate(day_count):
    #         toad_disp.append(X[:day_i, ii].flatten())
    #     X = toad_disp
    # if real_data:
    if nan_idx is not None:
        X = X.at[nan_idx].set(jnp.nan)
        # for i
        # idx = [ii for ii, toad_data in enumerate(X) if len(toad_data) > lag]
        # abs_disp = jnp.array([])
        # for ii, toad_data in enumerate(X):
        #     if len(toad_data) > lag:
        #         toad_data_np = jnp.array(toad_data)
        #         disp_ii = toad_data_np[lag:] - toad_data_np[:-lag]
        #         # disp_ii = disp.reshape(-1, disp.shape[-1])
        #         abs_disp_ii = jnp.abs(disp_ii)
        #         abs_disp = jnp.concatenate((abs_disp, abs_disp_ii), axis=0)
        #         abs_disp = abs_disp.flatten()
    # else:
    disp = X[lag:, :] - X[:-lag, :]
    # disp = disp.reshape(-1, disp.shape[-1])
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

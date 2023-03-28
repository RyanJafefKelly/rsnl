"""Implementation of the misspecified SLCP model."""

import jax.numpy as jnp
# import jax.random as random
from jax._src.prng import PRNGKeyArray  # for typing
import numpyro.distributions as dist  # type: ignore


def base_dgp(key: PRNGKeyArray,
             t1: jnp.ndarray,
             t2: jnp.ndarray,
             t3: jnp.ndarray,
             t4: jnp.ndarray,
             t5: jnp.ndarray,
            #  batch_size: int = 1
             ):
    """Base DGP - i.e. standard SLCP model."""
    m_theta = jnp.array([t1, t2])
    num_params = 5
    s1 = t3 ** 2
    s2 = t4 ** 2
    rho = jnp.tanh(t5)
    cov_mat = jnp.array([[s1 ** 2, rho * s1 * s2], [rho * s1 * s2, s2 ** 2]])
    y = dist.MultivariateNormal(m_theta, cov_mat).sample(key=key,
                                                         sample_shape=((1,
                                                                        num_params)))  # NOTE: removed batch_size
    return y.flatten()


def assumed_dgp(key, t1, t2, t3, t4, t5, std_err=1.0):
    """Assumed DGP - i.e. wrapper for standard SLCP model."""
    x = base_dgp(key, t1, t2, t3, t4, t5)  # NOTE: removed batch_size

    return x


def true_dgp(key, t1, t2, t3, t4, t5, std_err=1.0):
    """True DGP - i.e. standard SLCP model with contaminated draw."""
    x = base_dgp(key, t1, t2, t3, t4, t5)  # NOTE: removed batch_size
    x = misspecification_transform(x, misspec_level=1.0, key=key)

    return x


def calculate_summary_statistics(x):
    """Return data as summary."""
    return x


def misspecification_transform(x, misspec_level=1.0, key=None):
    """Add contaminated draw by adding noise."""
    noise_scale = 100
    # mask noise to only include fifth draw
    misspec_mask = jnp.array([0., 0., 0., 0., 0., 0., 0., 0., 1., 1.])
    noise_draws = dist.Normal(0, 1).sample(key, sample_shape=x.shape)
    misspec_x = x + noise_scale * noise_draws * misspec_level * misspec_mask
    return misspec_x


def get_prior():
    """Return prior for contaminated normal example."""
    prior = dist.Uniform(low=jnp.repeat(-3.0, 5),
                         high=jnp.repeat(3.0, 5))
    return prior

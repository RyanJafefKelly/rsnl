"""Implementation of the simple likelihood complex posterior model."""

import jax.numpy as jnp
import jax.random as random
from jax._src.prng import PRNGKeyArray  # for typing
import numpyro  # type: ignore
from numpyro.infer import MCMC, NUTS  # type: ignore
import numpyro.distributions as dist  # type: ignore


def base_dgp(key: PRNGKeyArray,
             t1: jnp.ndarray,
             t2: jnp.ndarray,
             t3: jnp.ndarray,
             t4: jnp.ndarray,
             t5: jnp.ndarray,
             num_draws: int = 5
            #  batch_size: int = 1
             ):
    """Base DGP - i.e. standard SLCP model."""
    m_theta = jnp.array([t1, t2])
    s1 = t3 ** 2
    s2 = t4 ** 2
    rho = jnp.tanh(t5)
    cov_mat = jnp.array([[s1 ** 2, rho * s1 * s2], [rho * s1 * s2, s2 ** 2]])
    y = dist.MultivariateNormal(m_theta, cov_mat).sample(key=key,
                                                         sample_shape=((1,
                                                                        num_draws)))  # NOTE: removed batch_size
    return y.flatten()


def assumed_dgp(key, t1, t2, t3, t4, t5, num_draws=5):
    """Assumed DGP - i.e. wrapper for standard SLCP model."""
    x = base_dgp(key, t1, t2, t3, t4, t5, num_draws=num_draws)  # NOTE: removed batch_size

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


def true_numpyro_model(x_obs, prior):
    """Return true posterior for contaminated SLCP example."""
    num_draws = 4
    x_obs = x_obs.reshape((num_draws, 2))
    # TODO: U1 - 5
    u1 = numpyro.sample('u1', dist.Uniform(-3.0, 3.0))
    u2 = numpyro.sample('u2', dist.Uniform(-3.0, 3.0))
    u3 = numpyro.sample('u3', dist.Uniform(-3.0, 3.0))
    u4 = numpyro.sample('u4', dist.Uniform(-3.0, 3.0))
    u5 = numpyro.sample('u5', dist.Uniform(-3.0, 3.0))

    m_theta = jnp.array([u1, u2])
    s1 = u3 ** 2
    s2 = u4 ** 2
    rho = jnp.tanh(u5)
    cov_mat = jnp.array([[s1 ** 2, rho * s1 * s2],
                         [rho * s1 * s2, s2 ** 2]])

    with numpyro.plate("data", num_draws):
        numpyro.sample('x_obs', dist.MultivariateNormal(m_theta, cov_mat),
                       obs=x_obs)

    return


def true_posterior(x_obs: jnp.ndarray,
                   prior: dist.Distribution) -> jnp.ndarray:
    """Return true posterior for contaminated SLCP example.

    Args:
        x_obs (jnp.ndarray): _description_
        prior (dist.Distribution): _description_

    Returns:
        jnp.ndarray: _description_
    """
    # true_params = jnp.array([0.7, -2.9, -1.0, -0.9, 0.6])
    # TODO: RUN MCMC TO GET TRUE POSTERIOR
    # TODO: m_theta and all that in numpyro
    x_obs = x_obs[:8]  # remove contaminated draw
    model = true_numpyro_model
    nuts_kernel = NUTS(model)  # NOTE: increase target_accept_prob?
    # TODO: NUTS IS BAD FOR MULTIMODAL POSTERIOR
    mcmc = MCMC(nuts_kernel,
            num_warmup=10_000,
            num_samples=10_000,
            thinning=10,
            num_chains=128
            )

    return mcmc

"""Implementation of the contaminated normal example."""

import jax.numpy as jnp
import numpyro.distributions as dist  # type: ignore
from jax import random
from jax._src.prng import PRNGKeyArray  # for typing


def true_dgp(key: PRNGKeyArray,
             t1: jnp.ndarray,
             y: jnp.ndarray,
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

# def get_model():
    # model = get_robust_model(prior, flow, laplace_var, st)
    """Return numpyro model for inference on contaminated normal example."""
    # def model(x_obs: jnp.ndarray,
    #           flow: Optional[FlowNumpyro] = None,
    #           laplace_var:  Optional[jnp.ndarray] = None,
    #           standardisation_params=None) -> jnp.ndarray:

    #     prior = dist.Normal(jnp.array([0.0]), jnp.array([10.0]))

    #     laplace_mean = jnp.array([0.0, 0.0])

    #     if laplace_var is None:
    #         laplace_var = jnp.array([1.0, 1.0])

    #     theta = numpyro.sample('theta', prior)
    #     theta_standard = numpyro.deterministic('theta_standard', (theta - standardisation_params['theta_mean']) / standardisation_params['theta_std'])

    #     adj_params = numpyro.sample('adj_params', dist.Laplace(laplace_mean,
    #                                                            laplace_var))
    #     x_adj = numpyro.deterministic('x_adj', x_obs - adj_params)

    #     if flow is not None:  # TODO?
    #         x_adj_sample = numpyro.sample('x_adj_sample',
    #                                       FlowNumpyro(flow, theta=theta_standard),
    #                                       obs=x_adj)
    #     else:
    #         x_adj_sample = x_adj

    #     return x_adj_sample

    # return model

"""Numpyro model funcs."""
from typing import Optional
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist  # type: ignore

from rsnl.utils import FlowNumpyro


# TODO: check legit...
# TODO: also... standardisation_params typing
def get_robust_model(x_obs: jnp.ndarray,
                     prior: dist.Distribution,
                     flow: Optional[FlowNumpyro] = None,
                     laplace_var:  Optional[jnp.ndarray] = None,
                     standardisation_params=None) -> jnp.ndarray:
    """Get robust numpyro model."""
    laplace_mean = jnp.array([0.0, 0.0])

    if laplace_var is None:
        laplace_var = jnp.array([1.0, 1.0])

    theta = numpyro.sample('theta', prior)
    theta_standard = numpyro.deterministic('theta_standard',
                                           (theta - standardisation_params['theta_mean']) / standardisation_params['theta_std'])

    adj_params = numpyro.sample('adj_params', dist.Laplace(laplace_mean,
                                                           laplace_var))
    x_adj = numpyro.deterministic('x_adj', x_obs - adj_params)

    if flow is not None:  # TODO?
        x_adj_sample = numpyro.sample('x_adj_sample',
                                      FlowNumpyro(flow, theta=theta_standard),
                                      obs=x_adj)
    else:
        x_adj_sample = x_adj

    return x_adj_sample


# TODO: goal take a model and return a new model that has adjustment parameters
def robustify(model, param_names,):
    pass
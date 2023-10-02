"""Numpyro model functions."""
from typing import Optional

import jax.numpy as jnp
import numpyro  # type: ignore
import numpyro.distributions as dist  # type: ignore

from rsnl.utils import FlowNumpyro


def get_standard_model(x_obs: jnp.ndarray,
                       prior: dist.Distribution,
                       flow: Optional[FlowNumpyro] = None,
                       standardisation_params=None) -> jnp.ndarray:
    """Get standard numpyro model."""
    theta = numpyro.sample('theta', prior)
    theta_standard = numpyro.deterministic('theta_standard',
                                           (theta - standardisation_params['theta_mean'])
                                           / standardisation_params['theta_std'])
    if flow is not None:  # i.e. if not first round
        x_sample = numpyro.sample('x_sample',
                                  FlowNumpyro(flow, theta=theta_standard),
                                  obs=x_obs)
    else:
        x_sample = x_obs
    return x_sample


def get_robust_model(x_obs: jnp.ndarray,
                     prior: dist.Distribution,
                     flow: Optional[FlowNumpyro] = None,
                     scale_adj_var:  Optional[jnp.ndarray] = None,
                     standardisation_params=None) -> jnp.ndarray:
    """Get robust numpyro model."""
    laplace_mean = jnp.zeros(len(x_obs))
    laplace_var = jnp.ones(len(x_obs))
    # if scale_adj_var is None:
    #     scale_adj_var = jnp.ones(len(x_obs))
    theta = numpyro.sample('theta', prior)
    theta_standard = numpyro.deterministic('theta_standard',
                                           (theta - standardisation_params['theta_mean'])
                                           / standardisation_params['theta_std'])

    # Note: better sampling if use standard laplace then scale
    adj_params = numpyro.sample('adj_params', dist.Laplace(laplace_mean,
                                                           laplace_var))
    scaled_adj_params = numpyro.deterministic('adj_params_scaled', adj_params *
                                              scale_adj_var)
    x_adj = numpyro.deterministic('x_adj', x_obs - scaled_adj_params)

    if flow is not None:  # i.e. if not first round
        x_adj_sample = numpyro.sample('x_adj_sample',
                                      FlowNumpyro(flow, theta=theta_standard),
                                      obs=x_adj)
    else:
        x_adj_sample = x_adj

    return x_adj_sample

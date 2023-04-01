"""Numpyro model funcs."""
from typing import Optional
import jax.numpy as jnp
import numpyro  # type: ignore
import numpyro.distributions as dist  # type: ignore

from rsnl.utils import FlowNumpyro

# TODO: INCLUDE NORMAL MODEL FOR SNL COMPARISON


def get_robust_model(x_obs: jnp.ndarray,
                     prior: dist.Distribution,
                     flow: Optional[FlowNumpyro] = None,
                     scale_adj_var:  Optional[jnp.ndarray] = None,
                     standardisation_params=None) -> jnp.ndarray:
    """Get robust numpyro model."""
    laplace_mean = jnp.zeros(len(x_obs))
    laplace_var = jnp.ones(len(x_obs))
    if scale_adj_var is None:
        scale_adj_var = jnp.ones(len(x_obs))
    theta = numpyro.sample('theta', prior)
    theta_standard = numpyro.deterministic('theta_standard',
                                           (theta - standardisation_params['theta_mean']) / standardisation_params['theta_std'])

    adj_params = numpyro.sample('adj_params', dist.Laplace(laplace_mean,
                                                           laplace_var))
    scaled_adj_params = numpyro.deterministic('adj_params_scaled', adj_params *
                                              scale_adj_var)
    x_adj = numpyro.deterministic('x_adj', x_obs - scaled_adj_params)

    if flow is not None:  # TODO?
        x_adj_sample = numpyro.sample('x_adj_sample',
                                      FlowNumpyro(flow, theta=theta_standard),
                                      obs=x_adj)
    else:
        x_adj_sample = x_adj

    return x_adj_sample

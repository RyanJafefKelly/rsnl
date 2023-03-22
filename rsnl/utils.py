"""Utils module for RSNL package."""

from jax import jit, vmap
import jax.numpy as jnp
from functools import partial
from numpyro.distributions import Distribution as dist # type: ignore
from numpyro.distributions.util import validate_sample  # type: ignore


class FlowNumpyro(dist):
    def __init__(self, flow=None, theta=None):
        self.flow = flow
        self.theta = theta
        super(FlowNumpyro, self).__init__()

    def sample(self, num_samples=1):
        return self.flow.sample(num_samples)

    @validate_sample
    def log_prob(self, value):
        ll = self.flow.log_prob(value, condition=self.theta)
        return ll


def vmap_dgp(sim_fn, sum_fn):
    """vmap..."""
    @partial(jit, static_argnums=(1))
    def simulation_wrapper(params, batch_size=1, key=None):
        x_sim = sim_fn(params, batch_size=batch_size, key=key)
        sim_sum = sum_fn(x_sim)
        return sim_sum

    @jit
    def generate_simulations(theta, key):
        x_sim = simulation_wrapper(theta, key=key)
        return x_sim

    generate_simulations_vmap = vmap(generate_simulations, in_axes=(0, 0))
    return generate_simulations_vmap
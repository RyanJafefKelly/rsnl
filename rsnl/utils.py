"""Utils module for RSNL package."""

from jax import vmap
from numpyro.distributions import Distribution as dist  # type: ignore
from numpyro.distributions.util import validate_sample  # type: ignore


class FlowNumpyro(dist):
    """A class to allow a flowjax flow to be used in numpyro.

    Args:
        dist (numpyro.distribution.Distribution): numpyro distribution.
    """

    def __init__(self, flow=None, theta=None):
        """Initialize the flow."""
        self.flow = flow
        self.theta = theta
        super(FlowNumpyro, self).__init__()

    def sample(self, num_samples=1):
        """Sample from the flow."""
        return self.flow.sample(num_samples)

    @validate_sample
    def log_prob(self, value):
        """Evaluate the log density of the neural likelihood."""
        ll = self.flow.log_prob(value, condition=self.theta)
        return ll


def vmap_dgp(sim_fn, sum_fn):
    """Create a vmap-able function for a simulation and summary function."""
    def simulation_wrapper(params, key=None):
        """Wrap simulation function."""
        x_sim = sim_fn(key, *params)
        sim_sum = sum_fn(x_sim)
        return sim_sum

    def generate_simulations(theta, key):
        """Generate simulations from a given theta."""
        x_sim = simulation_wrapper(theta, key=key)
        return x_sim

    generate_simulations_vmap = vmap(generate_simulations, in_axes=(0, 0))
    return generate_simulations_vmap

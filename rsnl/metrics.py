"""Metrics."""

import numpy as np
import jax.random as random
import jax.numpy as jnp
import numpyro.distributions as dist  # type: ignore
import matplotlib.pyplot as plt
from numpyro.infer import MCMC


def adj_param_test():
    pass


def log_prob_at_true_param(x_obs, true_param, prior, flow):
    """Calculate log prob at true param."""
    # TODO: at true param calc. flow and prior


def plot_and_save_coverage(empirical_coverage, folder_name=""):
    """Plot coverage."""
    # TODO! MAKE PAPER WORTHY.
    plt.plot([0, 1], [0, 1])
    plt.plot(empirical_coverage)
    plt.savefig("{folder_name}_empirical_coverage.png")


def get_true_posterior_draws(posterior, num_draws):
    # TODO? _ maybe do...
    # TODO: dist.Distribution
    if isinstance(posterior, dist.Distribution):
        pass
    # TODO: MCMC
    if isinstance(posterior, MCMC):
        pass
    pass


def calculate_coverage(x_obs, thetas, prior, flow, true_posterior):
    """Calculate empirical coverage.

    Calculate the expected coverage probability for a given set of samples, posterior estimator,
    and the function to compute the highest posterior density region.
    """
    rng_key = random.PRNGKey(0)
    coverage_levels = jnp.linspace(0.1, 0.9, 9)  # TODO?
    true_posterior = true_posterior(x_obs, prior)  # TODO?
    n_true = 1000
    true_posterior_draws = true_posterior.sample(rng_key, (n_true,))
    N, _ = thetas.shape
    log_probs = flow.log_prob(x_obs, thetas)

    log_probs_true = flow.log_prob(x_obs, true_posterior_draws)
    sort_idx = jnp.argsort(log_probs)[::-1]
    log_probs = log_probs[sort_idx]
    thetas = thetas[sort_idx]
    # TODO: calculate coverage
    empirical_coverage = [0]
    for coverage_level in coverage_levels:
        num_in_coverage = round(coverage_level * N)
        # thetas_hpd_sample = thetas[:num_in_coverage]
        cut_off = log_probs[num_in_coverage]
        true_in_hpd = jnp.sum(jnp.where(log_probs_true > cut_off, 1, 0))
        empirical_coverage.append(true_in_hpd/n_true)
    empirical_coverage.append(1)
    return jnp.array(empirical_coverage)


# TODO: coverage?
def calculate_metrics(x_obs, inference_data, prior, flow, true_posterior,
                      folder_name=""):
    thetas = jnp.concatenate(inference_data.posterior.theta.values, axis=0)
    empirical_coverage = calculate_coverage(x_obs, thetas, prior, flow,
                                            true_posterior)
    plot_and_save_coverage(empirical_coverage, folder_name)

    pass
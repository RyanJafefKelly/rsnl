"""Metrics."""

import numpy as np
import jax.random as random
import jax.numpy as jnp
import numpyro.distributions as dist  # type: ignore
import matplotlib.pyplot as plt
from numpyro.infer import MCMC
import pickle as pkl


def save_coverage_file(flow, x_obs, true_param, inference_data,
                       prior, standardisation_params,
                       folder_name=""):
    """Save coverage file."""
    theta_draws = inference_data.posterior.theta_standard.values
    x_obs_standard = (x_obs - standardisation_params['x_sims_mean']) / standardisation_params['x_sims_std']
    true_param_standard = (true_param - standardisation_params['theta_mean']) / standardisation_params['theta_std']
    theta_draws = jnp.concatenate(theta_draws, axis=0)  # axis-0 chains
    # NOTE: to Ease computation...only consider 1000 theta draws
    N = theta_draws.shape[0]
    theta_idx = np.random.choice(N, 1000, replace=False)
    theta_draws = theta_draws[theta_idx, :]
    # theta_draws_standard = (theta_draws - standardisation_params['theta_mean']) / standardisation_params['theta_std']
    log_prob_true_theta = flow.log_prob(x_obs_standard, true_param_standard)
    log_prob_true_theta += prior.log_prob(true_param)
    if hasattr(log_prob_true_theta, 'shape'):
        log_prob_true_theta = log_prob_true_theta[0]
    flow_log_prob_approx_thetas = flow.log_prob(x_obs_standard,
                                                theta_draws
                                                )
    prior_log_prob = jnp.squeeze(prior.log_prob(jnp.squeeze(theta_draws)))
    if prior_log_prob.ndim == 2:  # TODO: really could handle this better
        prior_log_prob = jnp.sum(prior_log_prob, axis=1)
    log_prob_approx_thetas = flow_log_prob_approx_thetas + prior_log_prob
    sort_idx = jnp.argsort(log_prob_approx_thetas)[::-1]
    log_prob_approx_thetas = log_prob_approx_thetas[sort_idx]
    theta_draws = theta_draws[sort_idx]
    N = theta_draws.shape[0]
    empirical_coverage = [0]
    coverage_levels = jnp.linspace(0.05, 0.95, 19)
    # TODO
    for coverage_level in coverage_levels:
        coverage_index = round(coverage_level * N)
        cut_off = log_prob_approx_thetas[coverage_index]
        if cut_off < log_prob_true_theta:
            empirical_coverage.append(1)
        else:
            empirical_coverage.append(0)
    empirical_coverage.append(1)
    save_file = f'{folder_name}coverage.txt'
    with open(save_file, 'wb') as f:
        for val in empirical_coverage:
            f.write(f"{str(val)}\n".encode('utf-8'))
    return

def log_prob_at_true_param(x_obs, true_param, prior, flow):
    """Calculate log prob at true param."""
    # TODO: at true param calc. flow and prior
    pass

def plot_and_save_coverage(empirical_coverage, folder_name=""):
    """Plot coverage."""
    # TODO: could improve
    plt.clf()
    plt.plot([0, 1], [0, 1], color='gray', linestyle='dashed')
    plt.plot(np.linspace(0, 1, len(empirical_coverage)), empirical_coverage)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("Credibility level")
    plt.ylabel("Empirical coverage")
    plt.savefig(f"{folder_name}empirical_coverage.png")


def get_true_posterior_draws(true_posterior, num_draws=10000,
                             x_obs=None, prior=None, seed=0):
    # TODO? _ maybe do...
    # n_true = 10000
    rng_key = random.PRNGKey(seed)
    if isinstance(true_posterior, dist.Distribution):
        true_posterior = true_posterior(x_obs, prior)  # TODO?
        true_posterior_draws = true_posterior.sample(rng_key, (num_draws,))
    # TODO: below condition seems ugly
    elif hasattr(true_posterior, '__call__'):
    # if isinstance(true_posterior, MCMC) or isinstance(true_posterior, function):
        true_posterior(x_obs, prior)
        true_posterior.run(rng_key, x_obs, prior)
        true_posterior_draws = true_posterior.get_samples()['theta']
    elif isinstance(true_posterior, str):
        with open(true_posterior, 'rb') as file:
            true_posterior_draws = pkl.load(file)

    return true_posterior_draws


def calculate_coverage(x_obs, thetas, prior, flow, true_posterior, seed=0):
    """Calculate empirical coverage.

    Calculate the expected coverage probability for a given set of samples, posterior estimator,
    and the function to compute the highest posterior density region.
    """
    coverage_levels = jnp.linspace(0.1, 0.9, 9)  # TODO?
    n_true = 1000
    N, _ = thetas.shape
    log_probs = flow.log_prob(x_obs, thetas)
    true_posterior_draws = get_true_posterior_draws(true_posterior, num_draws=n_true,
                                                    x_obs=x_obs, prior=prior,
                                                    seed=seed)
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
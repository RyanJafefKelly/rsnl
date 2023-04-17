"""Run contaminated normal example."""

import jax.numpy as jnp

from jax import random
import argparse
import arviz as az  # type: ignore
import multiprocessing as mp
import numpyro  # type: ignore
import os
import pickle as pkl
from rsnl.metrics import calculate_metrics
from rsnl.inference import run_rsnl
from rsnl.examples.contaminated_normal import (get_prior, assumed_dgp,
                                               calculate_summary_statistics,
                                               true_dgp, true_posterior)
from rsnl.visualisations import plot_and_save_all
from rsnl.model import get_robust_model


def run_contaminated_normal(args):
    """Script to run the full inference task on contaminated normal example."""
    seed = args.seed
    folder_name = "res/contaminated_normal/seed_{}/".format(seed)

    model = get_robust_model
    prior = get_prior()
    rng_key = random.PRNGKey(seed)
    rng_key, sub_key1, sub_key2 = random.split(rng_key, 3)
    sim_fn = assumed_dgp
    sum_fn = calculate_summary_statistics
    # true_params = jnp.array([1.0])
    true_params = prior.sample(sub_key1)
    x_obs = true_dgp(sub_key2, true_params)
    x_obs = calculate_summary_statistics(x_obs)
    # x_obs = jnp.array([1.0, 2.0])
    mcmc, flow = run_rsnl(model, prior, sim_fn, sum_fn, rng_key, x_obs,
                          jax_parallelise=True, true_params=true_params,
                          theta_dims=1
                          )
    mcmc.print_summary()
    isExist = os.path.exists(folder_name)
    if not isExist:
        os.makedirs(folder_name)
    inference_data = az.from_numpyro(mcmc)

    with open(f'{folder_name}thetas.pkl', 'wb') as f:
        pkl.dump(inference_data.posterior.theta, f)

    with open(f'{folder_name}adj_params.pkl', 'wb') as f:
        pkl.dump(inference_data.posterior.adj_params, f)

    # TODO: INCLUDE FILENAME
    # calculate_metrics(x_obs, inference_data, prior, flow, true_posterior,
    #                   folder_name=folder_name)
    plot_and_save_all(inference_data, true_params,
                      folder_name=folder_name)
    # log_prob  # TODO
    log_prob_true_theta = flow.log_prob(x_obs, true_params)
    theta_draws = inference_data.posterior.theta.values
    theta_draws = jnp.concatenate(theta_draws, axis=0)
    log_prob_approx_thetas = flow.log_prob(x_obs,
                                           theta_draws)
    sort_idx = jnp.argsort(log_prob_approx_thetas)[::-1]
    log_prob_approx_thetas = log_prob_approx_thetas[sort_idx]
    theta_draws = theta_draws[sort_idx]
    N = theta_draws.shape[0]
    empirical_coverage = [0]
    # in top x...
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='run_contaminated_normal.py',
        description='Run inference on contaminated normal example.',
        epilog='Example: python run_contaminated_normal.py'
        )
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # device_count = min(mp.cpu_count() - 1, 4)
    # device_count = max(device_count, 1)
    # numpyro.set_host_device_count(device_count)

    run_contaminated_normal(args)

"""Run contaminated normal example."""

import jax.numpy as jnp

from jax import random
import argparse
import arviz as az  # type: ignore
import multiprocessing as mp
import numpy as np
import numpyro  # type: ignore
import os
import pickle as pkl
from scipy.stats import gaussian_kde
from rsnl.metrics import save_coverage_file
from rsnl.inference import run_snl
from rsnl.examples.contaminated_normal import (get_prior, assumed_dgp,
                                               calculate_summary_statistics,
                                               true_dgp, true_posterior)
from rsnl.visualisations import plot_and_save_all
from rsnl.model import get_standard_model


def run_snl_contaminated_normal(args):
    """Script to run the full inference task on contaminated normal example."""
    seed = args.seed
    folder_name = "res/contaminated_normal/snl/seed_{}/".format(seed)

    model = get_standard_model
    prior = get_prior()
    rng_key = random.PRNGKey(seed)
    rng_key, sub_key1, sub_key2 = random.split(rng_key, 3)
    sim_fn = assumed_dgp
    sum_fn = calculate_summary_statistics
    true_params = jnp.array([1.0])
    # true_params = prior.sample(sub_key1)
    x_obs_tmp = true_dgp(sub_key2, true_params)
    x_obs_tmp = calculate_summary_statistics(x_obs_tmp)
    x_obs = jnp.array([x_obs_tmp[0], 2.0])  # add misspecified summ. var.
    # x_obs = jnp.array([1.0, 2.0])
    mcmc, flow, standardisation_params = run_snl(model, prior, sim_fn, sum_fn, rng_key, x_obs,
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

    plot_and_save_all(inference_data, true_params,
                      folder_name=folder_name)
    
    theta_draws = jnp.concatenate(inference_data.posterior.theta.values, axis=0)
    N = theta_draws.shape[0]
    theta_idx = np.random.choice(N, 10000, replace=False)
    theta_draws = theta_draws[theta_idx, :]
    theta_draws = jnp.squeeze(theta_draws)
    kde = gaussian_kde(theta_draws)
    logpdf_res = kde.logpdf(true_params)
    logpdf_res = float(logpdf_res)
    with open(f'{folder_name}logpdf_res.txt', 'wb') as f:
        f.write(str(logpdf_res).encode('utf-8'))

    save_coverage_file(flow, x_obs, true_params, inference_data,
                       prior, standardisation_params,
                       folder_name=folder_name)


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

    run_snl_contaminated_normal(args)

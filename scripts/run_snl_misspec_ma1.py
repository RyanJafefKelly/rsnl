"""Run misspec MA(1) example."""

import jax.numpy as jnp

from jax import random
import numpy as np
import argparse
import arviz as az  # type: ignore
import multiprocessing as mp
import numpyro  # type: ignore
import os
import pickle as pkl
from rsnl.inference import run_snl
from rsnl.examples.misspec_ma1 import (get_prior, assumed_dgp,
                                       calculate_summary_statistics,
                                       true_dgp)
from rsnl.visualisations import plot_and_save_all
from rsnl.model import get_standard_model
from rsnl.metrics import save_coverage_file
from scipy.stats import gaussian_kde


def run_snl_misspec_ma1_inference(args):
    """Script to run the full inference task on misspec MA(1) example."""
    seed = args.seed
    folder_name = "res/misspec_ma1/snl/seed_{}/".format(seed)

    model = get_standard_model
    prior = get_prior()
    rng_key = random.PRNGKey(seed)
    rng_key, sub_key = random.split(rng_key)
    sim_fn = assumed_dgp
    sum_fn = calculate_summary_statistics
    pseudo_true_param = jnp.array([0.0])
    x_obs = true_dgp(key=sub_key)
    x_obs = calculate_summary_statistics(x_obs)
    # x_obs = jnp.array([0.01, 0])
    mcmc, flow, standardisation_params = run_snl(model, prior, sim_fn, sum_fn,
                                                 rng_key, x_obs,
                                                 jax_parallelise=True,
                                                 true_params=pseudo_true_param)
    mcmc.print_summary()
    isExist = os.path.exists(folder_name)
    if not isExist:
        os.makedirs(folder_name)
    inference_data = az.from_numpyro(mcmc)

    with open(f'{folder_name}thetas.pkl', 'wb') as f:
        pkl.dump(inference_data.posterior.theta, f)

    plot_and_save_all(inference_data, pseudo_true_param,
                      folder_name=folder_name)

    theta_draws = jnp.concatenate(inference_data.posterior.theta.values,
                                  axis=0)
    N = theta_draws.shape[0]
    theta_idx = np.random.choice(N, 1000, replace=False)
    theta_draws = theta_draws[theta_idx, :]
    theta_draws = jnp.squeeze(theta_draws)
    kde = gaussian_kde(theta_draws)
    logpdf_res = kde.logpdf(pseudo_true_param)
    logpdf_res = float(logpdf_res)
    with open(f'{folder_name}logpdf_res.txt', 'wb') as f:
        f.write(str(logpdf_res).encode('utf-8'))

    save_coverage_file(flow, x_obs, pseudo_true_param, inference_data,
                       prior, standardisation_params,
                       folder_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='run_misspec_ma1.py',
        description='Run inference on misspecified MA(1) example.',
        epilog='Example: python run_misspec_ma1.py'
        )
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # device_count = min(mp.cpu_count() - 1, 4)
    # device_count = max(device_count, 1)
    # numpyro.set_host_device_count(device_count)

    run_snl_misspec_ma1_inference(args)

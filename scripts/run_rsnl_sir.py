"""Run SIR example."""

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
from rsnl.inference import run_rsnl
from rsnl.examples.sir import (assumed_dgp, get_prior,
                               calculate_summary_statistics, true_dgp,
                            #    true_posterior
                               )
from rsnl.metrics import save_coverage_file
from rsnl.visualisations import plot_and_save_all
from rsnl.model import get_robust_model
import matplotlib.pyplot as plt  # TODO: testing


def run_sir_inference(args):
    """Script to run the full inference task on contaminated SLCP example."""
    seed = args.seed
    folder_name = "res/sir/rsnl/seed_{}/".format(seed)

    model = get_robust_model
    prior = get_prior()
    rng_key = random.PRNGKey(seed)
    sim_fn = assumed_dgp
    summ_fn = calculate_summary_statistics
    rng_key, sub_key1, sub_key2 = random.split(rng_key, 3)

    true_params = prior.sample(sub_key1)
    # true_params = jnp.array([.1, .15])  # NOTE: arranged [gamma, beta]
    x_obs = true_dgp(sub_key2, *true_params)
    plt.plot(x_obs)
    plt.savefig('visualise_observed_sir.png')
    x_obs = calculate_summary_statistics(x_obs)
    print('x_obs: ', x_obs)
    mcmc, flow, standardisation_params = run_rsnl(model, prior, sim_fn, summ_fn, rng_key, x_obs,
                          jax_parallelise=False, true_params=true_params)
    mcmc.print_summary()
    isExist = os.path.exists(folder_name)
    if not isExist:
        os.makedirs(folder_name)
    inference_data = az.from_numpyro(mcmc)

    with open(f'{folder_name}thetas.pkl', 'wb') as f:
        pkl.dump(inference_data.posterior.theta, f)

    with open(f'{folder_name}adj_params.pkl', 'wb') as f:
        pkl.dump(inference_data.posterior.adj_params, f)

    plot_and_save_all(inference_data, true_params, folder_name=folder_name)

    theta_draws = jnp.concatenate(inference_data.posterior.theta.values,
                                  axis=0)
    N = theta_draws.shape[0]
    theta_idx = np.random.choice(N, 2000, replace=False)  # TODO: CHANGE BACK 10000
    theta_draws = theta_draws[theta_idx, :]
    theta_draws = jnp.squeeze(theta_draws)
    try:
        kde = gaussian_kde(theta_draws)
        logpdf_res = kde.logpdf(true_params)
    except Exception as e:
        print('Error: ', e)
        logpdf_res = np.NaN
    with open(f'{folder_name}logpdf_res.txt', 'wb') as f:
        f.write(str(logpdf_res).encode('utf-8'))

    save_coverage_file(flow, x_obs, true_params, inference_data,
                       prior, standardisation_params,
                       folder_name,
                       transpose_theta=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='run_sir.py',
        description='Run inference on SIR example.',
        epilog='Example: python run_sir.py'
        )
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # device_count = min(mp.cpu_count() - 1, 4)
    # device_count = max(device_count, 1)
    # numpyro.set_host_device_count(device_count)

    run_sir_inference(args)

"""Run SVAR example."""

import argparse
import os
import pickle as pkl

import arviz as az  # type: ignore
import jax.numpy as jnp
import numpy as np
from jax import random
import scipy.io as sio
import numpyro

from rsnl.examples.svar import (calculate_summary_statistics,
                                get_prior, true_dgp)
from rsnl.inference import run_rsnl
from rsnl.model import get_robust_model
from rsnl.visualisations import plot_and_save_all


def run_rsnl_svar(args):
    """Script to run the full inference task on SVAR example."""
    seed = args.seed
    np.random.seed(seed)
    folder_name = "res/svar/rsnl/seed_{}/".format(seed)

    model = get_robust_model
    prior = get_prior()
    rng_key = random.PRNGKey(seed)
    rng_key, sub_key1, sub_key2 = random.split(rng_key, 3)

    sim_fn = true_dgp
    sum_fn = calculate_summary_statistics

    # true_params = prior.sample(sub_key1)

    # hardcode true params, x_obs
    true_params = jnp.array([0.5787, -0.1435, 0.8356, 0.7448, -0.6603, -0.2538, 0.1])
    # x_obs_data = true_dgp(sub_key2, true_params)
    # x_obs = calculate_summary_statistics(x_obs_data)

    x_obs = jnp.array([0.0063, -0.0018, 0.0315, 0.0304, -0.0084, -0.0039, 0.1442])
    # x_obs_all = np.empty((7, 1000))
    # for i in range(1000):
    #     rng_key, sub_key = random.split(rng_key)
    #     x_obs_data = true_dgp(sub_key, true_params)
    #     x_obs = calculate_summary_statistics(x_obs_data)
    #     x_obs_all[:, i] = np.array(x_obs)

    mcmc = run_rsnl(model, prior, sim_fn, sum_fn, rng_key, x_obs,
                   jax_parallelise=False, true_params=true_params,
                   theta_dims=1)
    mcmc.print_summary()
    isExist = os.path.exists(folder_name)
    if not isExist:
        os.makedirs(folder_name)
    inference_data = az.from_numpyro(mcmc)

    with open(f'{folder_name}thetas.pkl', 'wb') as f:
        pkl.dump(inference_data.posterior.theta, f)

    with open(f'{folder_name}adj_params.pkl', 'wb') as f:
        pkl.dump(inference_data.posterior.adj_params, f)

    plot_and_save_all(inference_data, true_params,
                      folder_name=folder_name)


if __name__ == '__main__':
    numpyro.set_host_device_count(4)
    parser = argparse.ArgumentParser(
        prog='run_rsnl_svar.py',
        description='Run inference on SVAR example with RSNL.',
        epilog='Example: python run_rsnl_svar.py'
        )
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    run_rsnl_svar(args)

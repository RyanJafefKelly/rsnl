"""Run SIR example."""

import jax.numpy as jnp

from jax import random
import argparse
import arviz as az  # type: ignore
import multiprocessing as mp
import numpyro  # type: ignore
import os
import pickle as pkl
from rsnl.inference import run_snl
from rsnl.examples.sir import (assumed_dgp, get_prior,
                               calculate_summary_statistics, true_dgp,
                            #    true_posterior
                               )
from rsnl.metrics import save_coverage_file
from rsnl.visualisations import plot_and_save_all
from rsnl.model import get_standard_model
import matplotlib.pyplot as plt  # TODO: testing


def run_snl_sir_inference(args):
    """Script to run the full inference task on contaminated SLCP example."""
    seed = args.seed
    folder_name = "res/sir/snl/seed_{}/".format(seed)

    model = get_standard_model
    prior = get_prior()
    rng_key = random.PRNGKey(seed)
    sim_fn = assumed_dgp
    summ_fn = calculate_summary_statistics

    # true_params = prior.sample(rng_key)
    true_params = jnp.array([.1, .15])  # NOTE: arranged [gamma, beta]
    rng_key, sub_key = random.split(rng_key)
    x_obs = true_dgp(sub_key, *true_params)
    plt.plot(x_obs)
    plt.savefig('visualise_observed_sir.png')
    x_obs = calculate_summary_statistics(x_obs)
    print('x_obs: ', x_obs)
    mcmc, flow = run_snl(model, prior, sim_fn, summ_fn, rng_key, x_obs,
                         jax_parallelise=False, true_params=true_params)
    mcmc.print_summary()
    isExist = os.path.exists(folder_name)
    if not isExist:
        os.makedirs(folder_name)
    inference_data = az.from_numpyro(mcmc)

    with open(f'{folder_name}thetas.pkl', 'wb') as f:
        pkl.dump(inference_data.posterior.theta, f)

    plot_and_save_all(inference_data, true_params, folder_name=folder_name)
    # TODO: TRUE DISTRIBUTION FOR SIR MODEL
    # calculate_metrics(x_obs, inference_data, prior, flow, None,
    #                   folder_name=folder_name)
    save_coverage_file(flow, x_obs, true_params, inference_data,
                       folder_name)


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

    run_snl_sir_inference(args)

"""Run toad example."""

import argparse
import os
import pickle as pkl

import arviz as az  # type: ignore
import jax.numpy as jnp
import jax
from jax import random
import numpyro
import pandas as pd
from functools import partial
import scipy.io

from rsnl.examples.toad import (dgp, calculate_summary_statistics, get_prior)
from rsnl.inference import run_rsnl
from rsnl.model import get_robust_model
from rsnl.visualisations import plot_and_save_all


def get_real_xobs():
    df = scipy.io.loadmat('rsnl/examples/data/radio_converted.mat')['Y']
    nan_idx = jnp.where(jnp.isnan(df))
    df = jnp.array(df)

    x_obs = calculate_summary_statistics(df, real_data=True, nan_idx=nan_idx)

    sum_fn = partial(calculate_summary_statistics, real_data=True,
                     nan_idx=nan_idx)
    return x_obs, sum_fn


def run_rsnl_toad(args):
    """Script to run the full inference task on toad example."""
    seed = args.seed
    folder_name = "res/toad/rsnl/seed_{}/".format(seed)
    model = get_robust_model
    prior = get_prior()
    rng_key = random.PRNGKey(seed)
    rng_key, sub_key1, sub_key2 = random.split(rng_key, 3)
    sim_fn = partial(dgp, model=2)
    sum_fn = calculate_summary_statistics
    true_params = jnp.array([1.7, 35.0, 0.6])  # TODO: NOT ACTUALLY TRUE
    # # true_params = prior.sample(sub_key1)
    # x_obs_tmp = dgp(sub_key2, *true_params)
    # x_obs = calculate_summary_statistics(x_obs_tmp)
    x_obs, sum_fn = get_real_xobs()
    mcmc = run_rsnl(model, prior, sim_fn, sum_fn, rng_key, x_obs,
                    jax_parallelise=False,
                    true_params=true_params,
                    theta_dims=3)
    mcmc.print_summary()
    is_exist = os.path.exists(folder_name)
    if not is_exist:
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
        prog='run_rsnl_toad.py',
        description='Run inference on toad example with RSNL.',
        epilog='Example: python run_rsnl_toad.py'
        )
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    run_rsnl_toad(args)

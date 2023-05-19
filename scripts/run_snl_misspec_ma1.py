"""Run misspec MA(1) example."""

import argparse
import os
import pickle as pkl

import arviz as az  # type: ignore
import jax.numpy as jnp
from jax import random

from rsnl.examples.misspec_ma1 import (assumed_dgp,
                                       calculate_summary_statistics, get_prior,
                                       true_dgp)
from rsnl.inference import run_snl
from rsnl.model import get_standard_model
from rsnl.visualisations import plot_and_save_all


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
    mcmc = run_snl(model, prior, sim_fn, sum_fn, rng_key, x_obs,
                   jax_parallelise=True, true_params=pseudo_true_param)
    mcmc.print_summary()
    isExist = os.path.exists(folder_name)
    if not isExist:
        os.makedirs(folder_name)
    inference_data = az.from_numpyro(mcmc)

    with open(f'{folder_name}thetas.pkl', 'wb') as f:
        pkl.dump(inference_data.posterior.theta, f)

    plot_and_save_all(inference_data, pseudo_true_param,
                      folder_name=folder_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='run_snl_misspec_ma1.py',
        description='Run inference on misspecified MA(1) example with SNL.',
        epilog='Example: python run_snl_misspec_ma1.py'
        )
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    run_snl_misspec_ma1_inference(args)

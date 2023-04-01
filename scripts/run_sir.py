"""Run SIR example."""

import jax.numpy as jnp

from jax import random
import argparse
import arviz as az  # type: ignore
import os
import pickle as pkl
from rsnl.inference import run_rsnl
from rsnl.examples.sir import (assumed_dgp, get_prior,
                               calculate_summary_statistics, true_dgp,
                            #    true_posterior
                               )
from rsnl.metrics import calculate_metrics
from rsnl.visualisations import plot_and_save_all
from rsnl.model import get_robust_model


def run_sir_inference(args):
    """Script to run the full inference task on contaminated SLCP example."""
    seed = args.seed
    folder_name = "res/sir/seed_{}/".format(seed)

    model = get_robust_model
    prior = get_prior()
    rng_key = random.PRNGKey(seed)
    sim_fn = assumed_dgp
    summ_fn = calculate_summary_statistics

    # true_params = prior.sample(rng_key)
    true_params = jnp.array([.1, .2])  # NOTE: arranged [gamma, beta]
    rng_key, sub_key = random.split(rng_key)
    x_obs = true_dgp(sub_key, *true_params)
    x_obs = calculate_summary_statistics(x_obs)
    mcmc, flow = run_rsnl(model, prior, sim_fn, summ_fn, rng_key, x_obs,
                          true_params)
    mcmc.print_summary()
    # folder_name = "vis/rsnl_sir"
    # isExist = os.path.exists(folder_name)
    # if not isExist:
    #     os.makedirs(folder_name)
    inference_data = az.from_numpyro(mcmc)

    with open(f'{folder_name}_thetas.pkl', 'wb') as f:
        pkl.dump(inference_data.posterior.theta, f)

    with open(f'{folder_name}_adj_params.pkl', 'wb') as f:
        pkl.dump(inference_data.posterior.adj_params, f)

    plot_and_save_all(inference_data, true_params, folder_name=folder_name)
    # TODO: TRUE DISTRIBUTION FOR SIR MODEL
    calculate_metrics(x_obs, inference_data, prior, flow, None,
                      folder_name=folder_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='run_sir.py',
        description='Run inference on SIR example.',
        epilog='Example: python run_sir.py'
        )
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    run_sir_inference(args)

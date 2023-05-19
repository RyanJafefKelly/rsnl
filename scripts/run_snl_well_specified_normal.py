"""Run contaminated normal example."""

import argparse
import os
import pickle as pkl

import arviz as az  # type: ignore
import jax.numpy as jnp
from jax import random

from rsnl.examples.contaminated_normal import (assumed_dgp,
                                               calculate_summary_statistics,
                                               get_prior)
from rsnl.inference import run_snl
from rsnl.model import get_standard_model
from rsnl.visualisations import plot_and_save_all


def run_well_specified_normal(args):
    """Script to run the full inference task on contaminated normal example."""
    seed = args.seed
    folder_name = "res/contaminated_normal/snl_well_specified/seed_{}/".format(seed)

    model = get_standard_model
    prior = get_prior()
    rng_key = random.PRNGKey(seed)
    rng_key, sub_key1, sub_key2 = random.split(rng_key, 3)
    sim_fn = assumed_dgp
    sum_fn = calculate_summary_statistics
    true_params = jnp.array([1.0])
    # using assumed DGP as true DGP
    x_obs_tmp = assumed_dgp(sub_key2, true_params)
    x_obs = calculate_summary_statistics(x_obs_tmp)
    x_obs = jnp.array([x_obs[0], x_obs[1]])
    print('x_obs_tmp ', x_obs)

    mcmc = run_snl(model, prior, sim_fn, sum_fn, rng_key, x_obs,
                   jax_parallelise=True, true_params=true_params, theta_dims=1)
    mcmc.print_summary()
    isExist = os.path.exists(folder_name)
    if not isExist:
        os.makedirs(folder_name)
    inference_data = az.from_numpyro(mcmc)

    with open(f'{folder_name}thetas.pkl', 'wb') as f:
        pkl.dump(inference_data.posterior.theta, f)

    plot_and_save_all(inference_data, true_params,
                      folder_name=folder_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='run_well_specified_normal.py',
        description='Run inference on well-specified normal example.',
        epilog='Example: python run_well_specified_normal.py'
        )
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    run_well_specified_normal(args)

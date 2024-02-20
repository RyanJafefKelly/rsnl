"""Run contaminated normal example with RSNL."""

import argparse
import os
import pickle as pkl

import arviz as az  # type: ignore
import jax.numpy as jnp
from jax import random

from rsnl.examples.contaminated_normal import (assumed_dgp,
                                               calculate_summary_statistics,
                                               get_prior, true_dgp)
from rsnl.inference import run_snp
from rsnl.model import get_robust_model
from rsnl.visualisations import plot_and_save_all

import matplotlib.pyplot as plt

def run_contaminated_normal(args):
    """Script to run the full inference task on contaminated normal example."""
    seed = args.seed
    folder_name = "res/contaminated_normal/snp/seed_{}/".format(seed)

    # model = get_robust_model
    prior = get_prior()
    rng_key = random.PRNGKey(seed)
    rng_key, sub_key1, sub_key2 = random.split(rng_key, 3)
    sim_fn = assumed_dgp
    sum_fn = calculate_summary_statistics
    true_params = jnp.array([1.0])
    # true_params = prior.sample(sub_key1)
    # x_obs_tmp = true_dgp(sub_key2, true_params)
    # x_obs_tmp = calculate_summary_statistics(x_obs_tmp)
    # x_obs = jnp.array([1.0, 2.0])  # add misspecified summ. var.
    # x_obs = jnp.array([1.0, 2.0])
    x_obs = assumed_dgp(sub_key2, true_params)  # TODO: FOR TESTING
    x_obs = calculate_summary_statistics(x_obs).flatten()  # TODO: FOR TESTING
    # TODO: no model needed?
    samples = run_snp(prior, sim_fn, sum_fn, rng_key, x_obs,
                      jax_parallelise=True, true_params=true_params,
                      theta_dims=1,
                      num_rounds=5
                      )
    print('samples', jnp.mean(samples))

    isExist = os.path.exists(folder_name)
    if not isExist:
        os.makedirs(folder_name)
    # inference_data = az.from_numpyro(mcmc)
    for i in range(samples.shape[1]):
        plt.hist(samples[:, i].flatten(), bins=100)
        plt.savefig(f'{folder_name}hist_{str(i)}.png')
        plt.clf()

    with open(f'{folder_name}thetas.pkl', 'wb') as f:
        pkl.dump(samples, f)

    # plot_and_save_all(inference_data, true_params,
                    #   folder_name=folder_name)
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='run_contaminated_normal.py',
        description='Run inference on contaminated normal example with RSNL.',
        epilog='Example: python run_contaminated_normal.py'
        )
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    run_contaminated_normal(args)

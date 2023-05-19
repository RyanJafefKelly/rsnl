"""Run contaminated SLCP example."""
import argparse
import os
import pickle as pkl
from functools import partial

import arviz as az  # type: ignore
import jax.numpy as jnp
from jax import random

from rsnl.examples.contaminated_slcp import (assumed_dgp,
                                             calculate_summary_statistics,
                                             get_prior)
from rsnl.inference import run_snl
from rsnl.model import get_standard_model
from rsnl.visualisations import plot_and_save_all


def run_snl_contaminated_slcp_inference(args):
    """Script to run the full inference task on contaminated SLCP example."""
    seed = args.seed
    folder_name = "res/contaminated_slcp/well_specified_snl/seed_{}/".format(seed)

    model = get_standard_model
    prior = get_prior()
    rng_key = random.PRNGKey(seed)

    def dummy_assumed_dgp():
        assumed_dgp

    sim_fn = partial(assumed_dgp, num_draws=4)
    summ_fn = calculate_summary_statistics
    true_params = jnp.array([0.7, -2.9, -1.0, -0.9, 0.6])
    rng_key, sub_key = random.split(rng_key)
    x_obs = sim_fn(sub_key, *true_params)

    mcmc = run_snl(model, prior, sim_fn, summ_fn, rng_key, x_obs,
                   true_params=true_params)
    mcmc.print_summary()
    isExist = os.path.exists(folder_name)
    if not isExist:
        os.makedirs(folder_name)
    inference_data = az.from_numpyro(mcmc)

    with open(f'{folder_name}theta.pkl', 'wb') as f:
        pkl.dump(inference_data.posterior.theta, f)

    plot_and_save_all(inference_data, true_params,
                      folder_name=folder_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='run_snl_well_specified_slcp.py',
        description='Run inference on well-specified SLCP example.',
        epilog='Example: python run_snl_well_specified_slcp.py'
        )
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    run_snl_contaminated_slcp_inference(args)

"""Run contaminated normal example."""

import jax.numpy as jnp

from jax import random
import argparse
import arviz as az  # type: ignore
import os
import pickle as pkl
from rsnl.metrics import calculate_metrics
from rsnl.inference import run_rsnl
from rsnl.examples.contaminated_normal import (get_prior, assumed_dgp,
                                               calculate_summary_statistics,
                                               true_dgp, true_posterior)
from rsnl.visualisations import plot_and_save_all
from rsnl.model import get_robust_model


def run_contaminated_normal(args):
    """Script to run the full inference task on contaminated normal example."""
    seed = args.seed
    folder_name = "res/contaminated_normal/seed_{}/".format(seed)

    model = get_robust_model
    prior = get_prior()
    rng_key = random.PRNGKey(seed)
    sim_fn = assumed_dgp
    sum_fn = calculate_summary_statistics
    true_param = jnp.array([1.0])
    # x_obs = true_dgp(true_params)
    x_obs = jnp.array([1.0, 2.0])
    mcmc, flow = run_rsnl(model, prior, sim_fn, sum_fn, rng_key, x_obs,
                          true_param)
    mcmc.print_summary()
    isExist = os.path.exists(folder_name)
    if not isExist:
        os.makedirs(folder_name)
    inference_data = az.from_numpyro(mcmc)

    with open(f'{folder_name}thetas.pkl', 'wb') as f:
        pkl.dump(inference_data.posterior.theta, f)

    with open(f'{folder_name}adj_params.pkl', 'wb') as f:
        pkl.dump(inference_data.posterior.adj_params, f)

    # TODO: INCLUDE FILENAME
    calculate_metrics(x_obs, inference_data, prior, flow, true_posterior,
                      folder_name=folder_name)
    plot_and_save_all(inference_data, true_param,
                      folder_name=folder_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='run_contaminated_normal.py',
        description='Run inference on contaminated normal example.',
        epilog='Example: python run_contaminated_normal.py'
        )
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    run_contaminated_normal(args)

"""Run contaminated SLCP example."""
import jax.numpy as jnp

from jax import random
import arviz as az  # type: ignore
import argparse
import os
import pickle as pkl
from rsnl.inference import run_rsnl
from rsnl.examples.contaminated_slcp import (assumed_dgp, get_prior,
                                             calculate_summary_statistics,
                                             true_dgp)
from rsnl.visualisations import plot_and_save_all
from rsnl.model import get_robust_model


def run_contaminated_slcp_inference(args):
    """Script to run the full inference task on contaminated SLCP example."""
    seed = args.seed
    folder_name = "res/contaminated_slcp/rsnl/seed_{}/".format(seed)

    model = get_robust_model
    prior = get_prior()
    rng_key = random.PRNGKey(seed)
    sim_fn = assumed_dgp
    summ_fn = calculate_summary_statistics
    true_params = jnp.array([0.7, -2.9, -1.0, -0.9, 0.6])
    rng_key, sub_key = random.split(rng_key)
    x_obs = true_dgp(sub_key, *true_params)

    mcmc = run_rsnl(model, prior, sim_fn, summ_fn, rng_key, x_obs,
                    true_params=true_params)
    mcmc.print_summary()
    isExist = os.path.exists(folder_name)
    if not isExist:
        os.makedirs(folder_name)
    inference_data = az.from_numpyro(mcmc)

    with open(f'{folder_name}theta.pkl', 'wb') as f:
        pkl.dump(inference_data.posterior.theta, f)

    with open(f'{folder_name}adj_params.pkl', 'wb') as f:
        pkl.dump(inference_data.posterior.adj_params, f)

    plot_and_save_all(inference_data, true_params,
                      folder_name=folder_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='run_contaminated_slcp.py',
        description='Run inference on contaminated SLCP example with RSNL.',
        epilog='Example: python run_contaminated_slcp.py'
        )
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    run_contaminated_slcp_inference(args)

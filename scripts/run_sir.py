"""Run SIR example."""

import jax.numpy as jnp

from jax import random
import arviz as az  # type: ignore
import os
import pickle as pkl
from rsnl.inference import run_rsnl
from rsnl.examples.sir import (assumed_dgp, get_prior,
                               calculate_summary_statistics, true_dgp)
from rsnl.visualisations import plot_and_save_all
from rsnl.model import get_robust_model


def run_sir_inference():
    """Script to run the full inference task on contaminated SLCP example."""
    model = get_robust_model
    prior = get_prior()
    rng_key = random.PRNGKey(0)
    sim_fn = assumed_dgp
    summ_fn = calculate_summary_statistics
    # true_params = prior.sample(rng_key)
    true_params = jnp.array([.2, .1])
    rng_key, sub_key = random.split(rng_key)
    x_obs = true_dgp(sub_key, *true_params)
    x_obs = calculate_summary_statistics(x_obs)
    mcmc = run_rsnl(model, prior, sim_fn, summ_fn, rng_key, x_obs,
                    true_params)
    mcmc.print_summary()
    folder_name = "vis/rsnl_sir"
    isExist = os.path.exists(folder_name)
    if not isExist:
        os.makedirs(folder_name)
    inference_data = az.from_numpyro(mcmc)

    with open('rsnl_sir_thetas.pkl', 'wb') as f:
        pkl.dump(inference_data.posterior.theta, f)

    with open('rsnl_sir_adj_params.pkl', 'wb') as f:
        pkl.dump(inference_data.posterior.adj_params, f)

    # TODO: INCLUDE FILENAME
    plot_and_save_all(inference_data, true_params)


if __name__ == '__main__':
    # TODO: allow args e.g. name for filenames
    run_sir_inference()

"""Run misspec MA(1) example."""

import jax.numpy as jnp

from jax import random
import arviz as az  # type: ignore
import os
import pickle as pkl
from rsnl.inference import run_rsnl
from rsnl.examples.misspec_ma1 import (get_prior, assumed_dgp,
                                       calculate_summary_statistics)
from rsnl.visualisations import plot_and_save_all
from rsnl.model import get_robust_model


def run_misspec_ma1_inference():
    """Script to run the full inference task on misspec MA(1) example."""
    model = get_robust_model
    prior = get_prior()
    rng_key = random.PRNGKey(0)
    sim_fn = assumed_dgp
    sum_fn = calculate_summary_statistics
    pseudo_true_param = jnp.array([0.0])
    # x_obs = true_dgp(true_params)
    x_obs = jnp.array([0.01, 0])
    mcmc = run_rsnl(model, prior, sim_fn, sum_fn, rng_key, x_obs, pseudo_true_param)
    mcmc.print_summary()
    folder_name = "vis/rsnl_misspec_ma1"  # + str(stdev_err)
    isExist = os.path.exists(folder_name)
    if not isExist:
        os.makedirs(folder_name)
    inference_data = az.from_numpyro(mcmc)

    with open('rsnl_misspec_ma1_thetas.pkl', 'wb') as f:
        pkl.dump(inference_data.posterior.theta, f)

    with open('rsnl_misspec_ma1_adj_params.pkl', 'wb') as f:
        pkl.dump(inference_data.posterior.adj_params, f)

    # TODO: INCLUDE FILENAME
    plot_and_save_all(inference_data, pseudo_true_param)


if __name__ == '__main__':
    # TODO: allow args e.g. name for filenames
    run_misspec_ma1_inference()

"""Run contaminated normal example."""

import jax.numpy as jnp

from jax import random
import arviz as az  # type: ignore
import os
import pickle as pkl
from rsnl.inference import run_rsnl
from rsnl.examples.contaminated_normal import (get_model, assumed_dgp,
                                               summ_stats)
from rsnl.visualisations import plot_and_save_all


def run_contaminated_normal():
    """Script to run the full inference task on contaminated normal example."""
    model = get_model()
    rng_key = random.PRNGKey(0)
    sim_fn = assumed_dgp
    sum_fn = summ_stats
    true_param = jnp.array([1.0])
    # x_obs = true_dgp(true_params)
    x_obs = jnp.array([1.0, 2.0])
    mcmc = run_rsnl(model, sim_fn, sum_fn, rng_key, x_obs, true_param)
    mcmc.print_summary()
    folder_name = "vis/rsnl_contaminated_normal"  # + str(stdev_err)
    isExist = os.path.exists(folder_name)
    if not isExist:
        os.makedirs(folder_name)
    inference_data = az.from_numpyro(mcmc)

    with open('rsnl_contaminated_normal_thetas.pkl', 'wb') as f:
        pkl.dump(inference_data.posterior.theta, f)

    with open('rsnl_contaminated_normal_adj_params.pkl', 'wb') as f:
        pkl.dump(inference_data.posterior.adj_params, f)

    # TODO: INCLUDE FILENAME
    plot_and_save_all(inference_data, true_param)


if __name__ == '__main__':
    # TODO: allow args e.g. name for filenames
    run_contaminated_normal()

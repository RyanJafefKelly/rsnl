"""Run contaminated SLCP example."""
import jax
import jax.numpy as jnp

from jax import random
import arviz as az  # type: ignore
import os
import pickle as pkl
import arviz as az  # TODO: testing
import matplotlib.pyplot as plt  # TODO: testing
from rsnl.inference import run_rsnl
from rsnl.metrics import calculate_metrics
from rsnl.examples.contaminated_slcp import (assumed_dgp, get_prior,
                                             calculate_summary_statistics,
                                             true_dgp,
                                             true_posterior)
from rsnl.visualisations import plot_and_save_all, plot_theta_posterior
from rsnl.model import get_robust_model


def run_contaminated_slcp_inference():
    """Script to run the full inference task on contaminated SLCP example."""
    # print(jax.local_device_count())
    model = get_robust_model
    prior = get_prior()
    rng_key = random.PRNGKey(0)
    sim_fn = assumed_dgp
    summ_fn = calculate_summary_statistics
    true_params = jnp.array([0.7, -2.9, -1.0, -0.9, 0.6])
    rng_key, sub_key = random.split(rng_key)
    x_obs = true_dgp(sub_key, *true_params)
    # true_posterior_mcmc = true_posterior(x_obs[:8], prior)
    # # TODOL REMOVE BELOW TWO LINES
    # rng_key, sub_key1, sub_key2 = random.split(rng_key, 3)
    # true_posterior_mcmc.run(sub_key1, x_obs[:8], prior)
    # true_posterior_mcmc.print_summary()
    # inference_data = az.from_numpyro(true_posterior_mcmc)
    # az.plot_pair(inference_data.posterior, kind='kde')
    # plot_theta_posterior(inference_data, true_params)
    mcmc, flow = run_rsnl(model, prior, sim_fn, summ_fn, rng_key, x_obs,
                          true_params)
    mcmc.print_summary()
    folder_name = "vis/rsnl_contaminated_slcp"
    isExist = os.path.exists(folder_name)
    if not isExist:
        os.makedirs(folder_name)
    inference_data = az.from_numpyro(mcmc)

    with open('rsnl_contaminated_slcp_thetas.pkl', 'wb') as f:
        pkl.dump(inference_data.posterior.theta, f)

    with open('rsnl_contaminated_slcp_adj_params.pkl', 'wb') as f:
        pkl.dump(inference_data.posterior.adj_params, f)

    calculate_metrics(x_obs, inference_data, prior, flow, true_posterior)
    # TODO: INCLUDE FILENAME
    plot_and_save_all(inference_data, true_params)


if __name__ == '__main__':
    # TODO: allow args e.g. name for filenames
    run_contaminated_slcp_inference()

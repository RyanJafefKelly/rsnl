"""Inference (well sampling) methods for RSNL."""

from typing import Callable
import jax.numpy as jnp
import jax.random as random
from jax._src.prng import PRNGKeyArray  # for typing
from numpyro.infer import MCMC, NUTS  # type: ignore
from flowjax.flows import CouplingFlow  # type: ignore
from flowjax.bijections import RationalQuadraticSpline  # type: ignore
from flowjax.distributions import StandardNormal  # type: ignore
from flowjax.train.data_fit import fit_to_data  # type: ignore

from .utils import vmap_dgp


def run_rsnl(
    model: Callable,
    sim_fn: Callable,
    sum_fn: Callable,
    rng_key: PRNGKeyArray,
    x_obs: jnp.ndarray,
    true_param: jnp.ndarray,
) -> MCMC:
    """
    An RSNL sampler for models with adjustment parameters.

    Parameters
    ----------
    model : Callable
        The target model for which the RSNL sampler will be run.
    sim_fn : Callable
        The DGP function given the model parameters.
    sum_fn : Callable
        The summary function used for summarizing the simulated data.
    rng_key : jnp.ndarray
        The random number generator key.
    x_obs : jnp.ndarray
        The observed data for which the model will be fit.
    true_param : jnp.ndarray
        The true parameters of the model, used for initialization.

    Returns
    -------
    MCMC
        A NumPyro MCMC object containing the final posterior samples.
    """
    # hyperparameters
    # TODO: different approach than hardcode
    num_rounds = 5
    num_sims_per_round = 1000
    num_final_posterior_samples = 10_000
    thinning = 10
    num_warmup = 1000
    num_chains = 1
    summary_dims = len(x_obs)
    theta_dims = len(true_param)

    x_sims_all = jnp.empty((0, summary_dims))
    thetas_all = jnp.empty((0, theta_dims))

    flow = None

    init_params = {
        'theta': jnp.repeat(true_param, num_chains).reshape(num_chains, -1),
        'adj_params': jnp.repeat(
                                jnp.zeros(summary_dims),
                                num_chains
                                ).reshape(num_chains, -1)
        }

    x_obs_standard = x_obs

    standardisation_params = {
        'theta_mean': jnp.empty(theta_dims),
        'theta_std': jnp.empty(theta_dims),
        'x_sims_mean': jnp.empty(summary_dims),
        'x_sims_std': jnp.empty(summary_dims)
    }

    for i in range(num_rounds):
        nuts_kernel = NUTS(model)  # NOTE: increase target_accept_prob?
        mcmc = MCMC(nuts_kernel,
                    num_warmup=num_warmup,
                    num_samples=round((num_sims_per_round*thinning)/num_chains),
                    thinning=thinning,
                    num_chains=num_chains)
        rng_key, sub_key1, sub_key2 = random.split(rng_key, 3)
        laplace_var = 0.3 * jnp.abs(x_obs_standard)  # TODO: In testing..

        mcmc.run(sub_key1, x_obs_standard,
                 flow=flow,
                 laplace_var=laplace_var,
                 standardisation_params=standardisation_params,
                 init_params=init_params
                 )

        # set init_params for next round MCMC to final round vals
        rng_key, sub_key = random.split(rng_key)
        rand_idx = random.randint(sub_key, (num_chains,), 0, num_sims_per_round)
        for k, _ in init_params.items():
            init_params[k] = mcmc.get_samples()[k][-rand_idx]

        print('init_params: ', init_params)

        thetas = mcmc.get_samples()['theta']

        sim_keys = random.split(rng_key, len(thetas))
        vmap_dgp_fn = vmap_dgp(sim_fn, sum_fn)
        x_sims = jnp.squeeze(vmap_dgp_fn(thetas, sim_keys))

        x_sims_all = jnp.append(x_sims_all, x_sims.reshape(-1, summary_dims), axis=0)
        thetas_all = jnp.append(thetas_all, thetas.reshape(-1, theta_dims), axis=0)

        # standardise simulated summaries
        standardisation_params['x_sims_mean'] = jnp.mean(x_sims_all, axis=0)
        standardisation_params['x_sims_std'] = jnp.std(x_sims_all, axis=0)
        x_sims_all_standardised = (x_sims_all - standardisation_params['x_sims_mean']) / standardisation_params['x_sims_std']
        x_obs_standard = (x_obs - standardisation_params['x_sims_mean']) / standardisation_params['x_sims_std']

        # standardise parameters
        standardisation_params['theta_mean'] = jnp.mean(thetas_all, axis=0)
        standardisation_params['theta_std'] = jnp.std(thetas_all, axis=0)
        print('standardisation_params: ', standardisation_params)

        thetas_all_standardised = (thetas_all - standardisation_params['theta_mean']) / standardisation_params['theta_std']

        rng_key, sub_key = random.split(rng_key)
        flow = CouplingFlow(
            key=sub_key,
            base_dist=StandardNormal((summary_dims,)),
            transformer=RationalQuadraticSpline(knots=10, interval=5),  # 10 spline segments over [-5, 5].  # TODO? Increase num segments?
            cond_dim=thetas_all_standardised.shape[1],
            flow_layers=5,
            nn_width=50
            )

        rng_key, sub_key = random.split(rng_key)
        flow, losses = fit_to_data(key=sub_key,
                                   dist=flow,
                                   x=x_sims_all_standardised,
                                   condition=thetas_all_standardised,
                                   max_epochs=500,
                                   max_patience=20,
                                   )

        # TODO: investigate smarter mass_matrix

        # TODO: better standardisation...

        # TODO? Replace flow library
    # SAMPLE FINAL POSTERIOR
    nuts_kernel = NUTS(model, target_accept_prob=0.95)  # INCREASED FOR ROBUSTNESS
    mcmc = MCMC(nuts_kernel,
                num_warmup=num_warmup,
                num_samples=num_final_posterior_samples,
                thinning=thinning,
                num_chains=num_chains)  # TODO: MAKING NUMBERS UP
    rng_key, sub_key1, sub_key2 = random.split(rng_key, 3)
    laplace_var = 0.3 * jnp.abs(x_obs_standard)
    mcmc.run(sub_key1, x_obs_standard, flow=flow, standardisation_params=standardisation_params,
             laplace_var=laplace_var,
             init_params=init_params,
             )
    return mcmc

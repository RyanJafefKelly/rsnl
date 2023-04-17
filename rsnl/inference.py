"""Inference (well sampling) methods for RSNL."""

from typing import Callable, Optional
# import jax
import multiprocessing as mp
import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist  # type: ignore
from jax._src.prng import PRNGKeyArray  # for typing
from numpyro.infer import MCMC, NUTS  # type: ignore
from flowjax.flows import CouplingFlow  # type: ignore
from flowjax.bijections import RationalQuadraticSpline  # type: ignore
from flowjax.distributions import StandardNormal  # type: ignore
from flowjax.train.data_fit import fit_to_data  # type: ignore
import time
from functools import partial

from rsnl.utils import vmap_dgp #, lame_vmap


def dgp_fn_top(sim_fn, sum_fn, *args):
    sim_key = args[0]

    theta = jnp.atleast_1d(args[1:][0])  # array incase 1-dim
    return sum_fn(sim_fn(sim_key, *theta))


def run_rsnl(
    model: Callable,
    prior: dist.Distribution,
    sim_fn: Callable,
    sum_fn: Callable,
    rng_key: PRNGKeyArray,
    x_obs: jnp.ndarray,
    jax_parallelise=True,
    true_params: Optional[jnp.ndarray] = None,
    theta_dims: Optional[int] = 1
) -> MCMC:
    """
    An RSNL sampler for models with adjustment parameters.

    Parameters
    ----------
    model : Callable
        The target model for which the RSNL sampler will be run.
    prior : dist.Distribution
        The prior distribution for the model parameters.
    sim_fn : Callable
        The DGP function given the model parameters.
    sum_fn : Callable
        The summary function used for summarizing the simulated data.
    rng_key : jnp.ndarray
        The random number generator key.
    x_obs : jnp.ndarray
        The observed data for which the model will be fit.
    true_params : jnp.ndarray, optional
        The true parameters of the model, used for reference if available.

    Returns
    -------
    MCMC
        A NumPyro MCMC object containing the final posterior samples.
    """
    # device_count = min(mp.cpu_count() - 1, 4)
    # device_count = max(device_count, 1)
    # numpyro.set_host_device_count(device_count)
    # hyperparameters
    # TODO: different approach than hardcode
    num_rounds = 10
    num_sims_per_round = 1000  # NOTE: CHANGED FOR TESTING
    num_final_posterior_samples = 10_000  # NOTE: CHANGED FOR TESTING
    thinning = 10
    num_warmup = 1000
    num_chains = 4
    # num_devices = jax.local_device_count()
    summary_dims = len(x_obs)
    if true_params is not None:
        theta_dims = len(true_params)  # TODO! BETTER WAY TO DO THIS
        init_thetas = jnp.repeat(true_params, num_chains).reshape(num_chains, -1)
    else:
        init_thetas = None

    x_sims_all = jnp.empty((0, summary_dims))
    thetas_all = jnp.empty((0, theta_dims))

    flow = None

    init_params = {
        'theta': init_thetas,  # TODO! BETTER WAY TO DO THIS
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
        scale_adj_var = 0.3 * jnp.abs(x_obs_standard)  # TODO: In testing..
        print('scale_adj_var: ', scale_adj_var)
        if i == 0:  # TODO! VERIFY
            scale_adj_var = None
        mcmc.run(sub_key1,
                 x_obs_standard,
                 prior,
                 flow=flow,
                 scale_adj_var=scale_adj_var,
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
        # thetas = thetas.reshape(-1, theta_dims)
        # print('thetas len: ', len(thetas))
        mcmc.print_summary()  # TODO: include for now
        print('theta mean: ', jnp.mean(thetas, axis=0))
        print('theta std: ', jnp.std(thetas, axis=0))
        sim_keys = random.split(rng_key, len(thetas))

        # TODO! PARALLELISE SIMULATIONS
        x_sims = jnp.empty((0, summary_dims))

        # num_processes = max(1, mp.cpu_count() - 1)

        # pool = mp.Pool(num_processes)
        # dgp_fn = partial(dgp_fn_top, sim_fn, sum_fn)
        # x_sims = pool.starmap_async(dgp_fn, zip(sim_keys, thetas)).get(
        #     timeout=10000  # in seconds
        # )
        if jax_parallelise:
            vmap_dgp_fn = vmap_dgp(sim_fn, sum_fn)
            x_sims = vmap_dgp_fn(thetas, sim_keys)
        else:
            x_sims = [sum_fn(sim_fn(sim_key, *theta))
                      for sim_key, theta in zip(sim_keys, thetas)]
        # TODO: could improve
        valid_idx = [ii for ii, ssx in enumerate(x_sims) if ssx is not None]
        x_sims = [ssx for ii, ssx in enumerate(x_sims) if ssx is not None]
        print('valid_idx length: ', len(valid_idx))
        # x_sims = jnp.array(x_sims)[valid_idx]
        thetas = thetas[valid_idx, :]
        # for ii, theta in enumerate(thetas):
        #     # tic = time.time()
        #     if ii % 50 == 0:
        #         print('ii: ', ii)
        #     x_sim = sum_fn(sim_fn(sim_keys[ii], *theta))
        #     if x_sim is not None:  # TODO? smoother approach
        #         x_sims = jnp.append(x_sims, x_sim.reshape(-1, summary_dims), axis=0)
        #         valid_idx.append(ii)
        #     # toc = time.time()
        #     # print('time: ', toc - tic)


        # vmap_dgp_fn = lame_vmap(sim_fn, sum_fn)
        # x_sims = jnp.squeeze(vmap_dgp_fn(thetas, sim_keys))

        x_sims_all = jnp.append(x_sims_all, jnp.array(x_sims).reshape(-1, summary_dims), axis=0)
        # TODO? no invalid handling
        thetas_all = jnp.append(thetas_all, thetas, axis=0)

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
                                #    batch_size=128,  # NOTE: Changed
                                #    val_prop=0.2,  # NOTE: Changed
                                   )

        # TODO: investigate smarter mass_matrix

        # TODO: better standardisation...

        # TODO? Replace flow library
    # SAMPLE FINAL POSTERIOR
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel,
                num_warmup=num_warmup,
                num_samples=num_final_posterior_samples,
                thinning=1,
                num_chains=num_chains)  # TODO: MAKING NUMBERS UP
    rng_key, sub_key1, sub_key2 = random.split(rng_key, 3)
    scale_adj_var = 0.3 * jnp.abs(x_obs_standard)
    mcmc.run(sub_key1,
             x_obs_standard,
             prior,
             flow=flow,
             standardisation_params=standardisation_params,
             scale_adj_var=scale_adj_var,
             init_params=init_params,
             )

    return mcmc, flow

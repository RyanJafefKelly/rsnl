"""Inference methods for RSNL."""
import time
from typing import Callable, Optional

import jax.numpy as jnp
import jax.random as random
import numpyro.distributions as dist  # type: ignore
from flowjax.bijections import RationalQuadraticSpline  # type: ignore
from flowjax.distributions import StandardNormal  # type: ignore
from flowjax.flows import CouplingFlow  # type: ignore
from flowjax.train.data_fit import fit_to_data  # type: ignore
from jax._src.prng import PRNGKeyArray  # for typing
from numpyro.infer import MCMC, NUTS  # type: ignore
import multiprocessing as mp
import pickle as pkl

from rsnl.utils import vmap_dgp, start_matlab_engine, stop_matlab_engine, engines


def run_rsnl(
    model: Callable,
    prior: dist.Distribution,
    sim_fn: Callable,
    sum_fn: Callable,
    rng_key: PRNGKeyArray,
    x_obs: jnp.ndarray,
    jax_parallelise: bool = True,
    mp_parallelise: bool = False,
    num_cpus: Optional[int] = None,
    true_params: Optional[jnp.ndarray] = None,
    theta_dims: Optional[int] = 1,
    num_sims_per_round: Optional[int] = 1000,
    num_rounds: Optional[int] = 10,
    num_chains: Optional[int] = 4,
    scale_adj_var: Optional[float] = None,
    scale_adj_var_x_obs: Optional[float] = 0.3,
    target_accept_prob: Optional[float] = 0.8,
    save_each_round: Optional[bool] = False,
    folder_name: Optional[str] = ''
) -> MCMC:
    """Run inference to get samples from the RSNL approximate posterior.

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
    jax_parallelise : bool, optional
        Flag whether to parallelise computations using JAX. Defaults to True.
    true_params : jnp.ndarray, optional
        The true parameters of the model, used for reference if available.
    theta_dims : int, optional
        The number of dimensions in the parameter space. Defaults to 1.

    Returns
    -------
    MCMC
        A NumPyro MCMC object containing the final posterior samples.
    """
    # Set hyperparameters
    num_final_posterior_samples = 10_000
    thinning = 10
    num_warmup = 1000
    summary_dims = len(x_obs)
    if true_params is not None:
        theta_dims = len(true_params)
        init_thetas = jnp.repeat(true_params, num_chains).reshape(num_chains, -1)
    else:
        init_thetas = None

    x_sims_all = jnp.empty((0, summary_dims))
    thetas_all = jnp.empty((0, theta_dims))

    flow = None
    default_scale_adj_var = scale_adj_var

    # initialise times
    mcmc_time = 0.0
    sim_time = 0.0
    flow_time = 0.0

    init_params = {
        'theta': init_thetas,
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
        nuts_kernel = NUTS(model, target_accept_prob=target_accept_prob)
        mcmc = MCMC(nuts_kernel,
                    num_warmup=num_warmup,
                    num_samples=round((num_sims_per_round*thinning)/num_chains),
                    thinning=thinning,
                    num_chains=num_chains)
        rng_key, sub_key1, sub_key2 = random.split(rng_key, 3)
        if default_scale_adj_var is None:  # NOTE: i.e. this is the default behaviour
            if i == 0:
                scale_adj_var = jnp.ones(len(x_obs))
            else:
                scale_adj_var = scale_adj_var_x_obs * jnp.abs(x_obs_standard)
        else:
            scale_adj_var = default_scale_adj_var
        tic = time.time()
        mcmc.run(sub_key1,
                 x_obs_standard,
                 prior,
                 flow=flow,
                 scale_adj_var=scale_adj_var,
                 standardisation_params=standardisation_params,
                 init_params=init_params
                 )
        toc = time.time()
        mcmc_time += toc-tic
        print(f'Round {i+1} MCMC took {toc-tic:.2f} seconds')
        # set init_params for next round MCMC to final round vals
        rng_key, sub_key = random.split(rng_key)
        rand_idx = random.randint(sub_key, (num_chains,), 0,
                                  num_sims_per_round)
        for k, _ in init_params.items():
            init_params[k] = mcmc.get_samples()[k][-rand_idx]

        thetas = mcmc.get_samples()['theta']
        mcmc.print_summary()

        sim_keys = random.split(rng_key, len(thetas))
        x_sims = jnp.empty((0, summary_dims))
        tic = time.time()
        if jax_parallelise:
            vmap_dgp_fn = vmap_dgp(sim_fn, sum_fn)
            x_sims = vmap_dgp_fn(thetas, sim_keys)
        elif mp_parallelise:
            # run simulations in parallel using multiprocessing
            num_cpus = mp.cpu_count() - 1 if num_cpus is None else num_cpus
            print("Starting Pool with num_cpus: ", num_cpus)
            with mp.Pool(processes=num_cpus, initializer=start_matlab_engine) as pool:
                x_sims = jnp.array(pool.starmap(sim_fn, [(sim_key, *theta) for sim_key, theta in zip(sim_keys, thetas)]))
                x_sims = jnp.array(pool.starmap(sum_fn, [(x,) for x in x_sims]))
            # Clean up MATLAB engines after all tasks are done
            for pid, eng in list(engines.items()):
                eng.quit()
                del engines[pid]

        else:
            x_sims = jnp.array([sum_fn(sim_fn(sim_key, *theta))
                                for sim_key, theta in zip(sim_keys, thetas)])
        toc = time.time()
        sim_time += toc-tic
        print(f'Round {i+1} simulations took {toc-tic:.2f} seconds')
        # remove any failed simulations
        valid_idx = [ii for ii, ssx in enumerate(x_sims) if ssx is not None]
        # if clip_val is not None:
        #     valid_idx = [ii for ii, ssx in enumerate(x_sims) if ii in valid_idx and jnp.all(ssx < clip_val)]
        x_sims = jnp.array([ssx for ii, ssx in enumerate(x_sims) if ssx is not None])
        thetas = thetas[valid_idx, :]

        x_sims_all = jnp.append(x_sims_all, x_sims.reshape(-1, summary_dims),
                                axis=0)
        thetas_all = jnp.append(thetas_all, thetas, axis=0)

        # standardise simulated summaries
        standardisation_params['x_sims_mean'] = jnp.mean(x_sims_all, axis=0)
        standardisation_params['x_sims_std'] = jnp.std(x_sims_all, axis=0)
        x_sims_all_standardised = (x_sims_all - standardisation_params['x_sims_mean']) / standardisation_params['x_sims_std']
        x_obs_standard = (x_obs - standardisation_params['x_sims_mean']) / standardisation_params['x_sims_std']

        # standardise parameters
        standardisation_params['theta_mean'] = jnp.mean(thetas_all, axis=0)
        standardisation_params['theta_std'] = jnp.std(thetas_all, axis=0)

        thetas_all_standardised = (thetas_all - standardisation_params['theta_mean']) / standardisation_params['theta_std']

        # initialise and train flow
        rng_key, sub_key = random.split(rng_key)
        flow = CouplingFlow(
            key=sub_key,
            base_dist=StandardNormal((summary_dims,)),
            transformer=RationalQuadraticSpline(knots=10, interval=5),  # 10 spline segments over [-5, 5].
            cond_dim=thetas_all_standardised.shape[1],
            flow_layers=5,
            nn_width=50
            )

        tic = time.time()
        rng_key, sub_key = random.split(rng_key)
        flow, losses = fit_to_data(key=sub_key,
                                   dist=flow,
                                   x=x_sims_all_standardised,
                                   condition=thetas_all_standardised,
                                   max_epochs=500,
                                   max_patience=20,
                                   batch_size=256
                                   )
        toc = time.time()
        flow_time += toc-tic
        print(f'Round {i+1} flow training took {toc-tic:.2f} seconds')

        if save_each_round:
            with open(f'{folder_name}thetas_all_round_{i+1}.pkl', 'wb') as f:
                pkl.dump(thetas_all, f)
            with open(f'{folder_name}x_sims_all_round_{i+1}.pkl', 'wb') as f:
                pkl.dump(x_sims_all, f)
            # TODO: save adj_params
            with open(f'{folder_name}adj_params_round_{i+1}.pkl', 'wb') as f:
                pkl.dump(mcmc.get_samples()['adj_params'], f)


    # Sample final posterior
    tic = time.time()
    nuts_kernel = NUTS(model, target_accept_prob=target_accept_prob)
    mcmc = MCMC(nuts_kernel,
                num_warmup=num_warmup,
                num_samples=num_final_posterior_samples,
                thinning=1,  # no need to thin for final posterior
                num_chains=num_chains)
    rng_key, sub_key = random.split(rng_key)
    if default_scale_adj_var is None:  # NOTE: i.e. this is the default behaviour
        if i == 0:
            scale_adj_var = jnp.ones(len(x_obs))
        else:
            scale_adj_var = scale_adj_var_x_obs * jnp.abs(x_obs_standard)
    else:
        scale_adj_var = default_scale_adj_var
    mcmc.run(sub_key,
             x_obs_standard,
             prior,
             flow=flow,
             standardisation_params=standardisation_params,
             scale_adj_var=scale_adj_var,
             init_params=init_params,
             )
    toc = time.time()
    mcmc_time += toc-tic
    print(f'Final posterior MCMC took {toc-tic:.2f} seconds')

    print(f'Total MCMC time: {mcmc_time:.2f} seconds')
    print(f'Total simulation time: {sim_time:.2f} seconds')
    print(f'Total flow training time: {flow_time:.2f} seconds')

    return mcmc


def run_snl(
    model: Callable,
    prior: dist.Distribution,
    sim_fn: Callable,
    sum_fn: Callable,
    rng_key: PRNGKeyArray,
    x_obs: jnp.ndarray,
    jax_parallelise=True,
    true_params: Optional[jnp.ndarray] = None,
    theta_dims: Optional[int] = 1,
    num_sims_per_round: Optional[int] = 1000,
    num_rounds: Optional[int] = 10,
    num_chains: Optional[int] = 4,
) -> MCMC:
    """Run inference to get samples from the SNL approximate posterior.

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
    jax_parallelise : bool, optional
        Flag whether to parallelise computations using JAX. Defaults to True.
    true_params : jnp.ndarray, optional
        The true parameters of the model, used for reference if available.
    theta_dims : int, optional
        The number of dimensions in the parameter space. Defaults to 1.

    Returns
    -------
    MCMC
        A NumPyro MCMC object containing the final posterior samples.
    """
    num_final_posterior_samples = 10_000
    thinning = 10
    num_warmup = 1000
    summary_dims = len(x_obs)
    if true_params is not None:
        theta_dims = len(true_params)
        init_thetas = jnp.repeat(true_params, num_chains).reshape(num_chains, -1)
    else:
        init_thetas = None

    x_sims_all = jnp.empty((0, summary_dims))
    thetas_all = jnp.empty((0, theta_dims))

    flow = None

    # initialise times
    mcmc_time = 0.0
    sim_time = 0.0
    flow_time = 0.0

    init_params = {
        'theta': init_thetas
        }

    x_obs_standard = x_obs

    standardisation_params = {
        'theta_mean': jnp.empty(theta_dims),
        'theta_std': jnp.empty(theta_dims),
        'x_sims_mean': jnp.empty(summary_dims),
        'x_sims_std': jnp.empty(summary_dims)
    }

    for i in range(num_rounds):
        nuts_kernel = NUTS(model)
        mcmc = MCMC(nuts_kernel,
                    num_warmup=num_warmup,
                    num_samples=round((num_sims_per_round*thinning)/num_chains),
                    thinning=thinning,
                    num_chains=num_chains)
        rng_key, sub_key1, sub_key2 = random.split(rng_key, 3)
        tic = time.time()
        mcmc.run(sub_key1,
                 x_obs_standard,
                 prior,
                 flow=flow,
                 standardisation_params=standardisation_params,
                 init_params=init_params
                 )
        toc = time.time()
        mcmc_time += toc-tic
        print(f'Round {i+1} MCMC took {toc-tic:.2f} seconds')
        # set init_params for next round MCMC to final round vals
        rng_key, sub_key = random.split(rng_key)
        rand_idx = random.randint(sub_key, (num_chains,), 0,
                                  num_sims_per_round)
        for k, _ in init_params.items():
            init_params[k] = mcmc.get_samples()[k][-rand_idx]

        thetas = mcmc.get_samples()['theta']
        mcmc.print_summary()

        sim_keys = random.split(rng_key, len(thetas))
        x_sims = jnp.empty((0, summary_dims))
        tic = time.time()
        if jax_parallelise:
            vmap_dgp_fn = vmap_dgp(sim_fn, sum_fn)
            x_sims = vmap_dgp_fn(thetas, sim_keys)
        else:
            x_sims = jnp.array([sum_fn(sim_fn(sim_key, *theta))
                                for sim_key, theta in zip(sim_keys, thetas)])

        toc = time.time()
        sim_time += toc-tic
        print(f'Round {i+1} simulations took {toc-tic:.2f} seconds')

        valid_idx = [ii for ii, ssx in enumerate(x_sims) if ssx is not None]
        x_sims = jnp.array([ssx for ii, ssx in enumerate(x_sims) if ssx is not None])
        thetas = thetas[valid_idx, :]

        x_sims_all = jnp.append(x_sims_all, x_sims.reshape(-1, summary_dims), axis=0)
        thetas_all = jnp.append(thetas_all, thetas, axis=0)

        # standardise simulated summaries
        standardisation_params['x_sims_mean'] = jnp.mean(x_sims_all, axis=0)
        standardisation_params['x_sims_std'] = jnp.std(x_sims_all, axis=0)
        x_sims_all_standardised = (x_sims_all - standardisation_params['x_sims_mean']) / standardisation_params['x_sims_std']
        x_obs_standard = (x_obs - standardisation_params['x_sims_mean']) / standardisation_params['x_sims_std']

        # standardise parameters
        standardisation_params['theta_mean'] = jnp.mean(thetas_all, axis=0)
        standardisation_params['theta_std'] = jnp.std(thetas_all, axis=0)

        thetas_all_standardised = (thetas_all - standardisation_params['theta_mean']) / standardisation_params['theta_std']

        rng_key, sub_key = random.split(rng_key)
        flow = CouplingFlow(
            key=sub_key,
            base_dist=StandardNormal((summary_dims,)),
            transformer=RationalQuadraticSpline(knots=10, interval=5),  # 10 spline segments over [-5, 5].
            cond_dim=thetas_all_standardised.shape[1],
            flow_layers=5,
            nn_width=50
            )

        rng_key, sub_key = random.split(rng_key)
        tic = time.time()
        flow, losses = fit_to_data(key=sub_key,
                                   dist=flow,
                                   x=x_sims_all_standardised,
                                   condition=thetas_all_standardised,
                                   max_epochs=500,
                                   max_patience=20,
                                   batch_size=256,
                                   )
        toc = time.time()
        flow_time += toc-tic

    # Sample final posterior
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel,
                num_warmup=num_warmup,
                num_samples=num_final_posterior_samples,
                thinning=1,
                num_chains=num_chains)
    rng_key, sub_key = random.split(rng_key)
    tic = time.time()
    mcmc.run(sub_key,
             x_obs_standard,
             prior,
             flow=flow,
             standardisation_params=standardisation_params,
             init_params=init_params,
             )
    toc = time.time()
    mcmc_time += toc-tic
    print(f'Final posterior MCMC took {toc-tic:.2f} seconds')

    print(f'Total MCMC time: {mcmc_time:.2f} seconds')
    print(f'Total simulation time: {sim_time:.2f} seconds')
    print(f'Total flow training time: {flow_time:.2f} seconds')

    return mcmc


def run_snp(
    prior: dist.Distribution,
    sim_fn: Callable,
    sum_fn: Callable,
    rng_key: PRNGKeyArray,
    x_obs: jnp.ndarray,
    jax_parallelise=True,
    true_params: Optional[jnp.ndarray] = None,
    theta_dims: Optional[int] = 1,
    num_sims_per_round: Optional[int] = 1000,
    num_rounds: Optional[int] = 10
) -> MCMC:
    """Run inference to get samples from the SNP approximate posterior.

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
    jax_parallelise : bool, optional
        Flag whether to parallelise computations using JAX. Defaults to True.
    true_params : jnp.ndarray, optional
        The true parameters of the model, used for reference if available.
    theta_dims : int, optional
        The number of dimensions in the parameter space. Defaults to 1.

    Returns
    -------
    MCMC
        A NumPyro MCMC object containing the final posterior samples.
    """
    num_final_posterior_samples = 10_000

    summary_dims = len(x_obs)
    if true_params is not None:
        theta_dims = len(true_params)

    x_sims_all = jnp.empty((0, summary_dims))
    thetas_all = jnp.empty((0, theta_dims))

    flow = None

    # initialise times
    sim_time = 0.0
    flow_time = 0.0

    x_obs_standard = x_obs

    standardisation_params = {
        'theta_mean': jnp.empty(theta_dims),
        'theta_std': jnp.empty(theta_dims),
        'x_sims_mean': jnp.empty(summary_dims),
        'x_sims_std': jnp.empty(summary_dims)
    }

    for i in range(num_rounds):
        rng_key, sub_key1, sub_key2 = random.split(rng_key, 3)
        if i == 0:
            thetas = prior.sample(sub_key1, (num_sims_per_round,))
        else:
            thetas_standard = flow.sample(sub_key1,
                                 (num_sims_per_round,),
                                 x_obs_standard)
            thetas = thetas_standard * standardisation_params['theta_std'] + standardisation_params['theta_mean']

        sim_keys = random.split(rng_key, len(thetas))
        x_sims = jnp.empty((0, summary_dims))
        tic = time.time()
        if jax_parallelise:
            vmap_dgp_fn = vmap_dgp(sim_fn, sum_fn)
            x_sims = vmap_dgp_fn(thetas, sim_keys)
        else:
            x_sims = jnp.array([sum_fn(sim_fn(sim_key, *theta))
                                for sim_key, theta in zip(sim_keys, thetas)])

        toc = time.time()
        sim_time += toc-tic
        print(f'Round {i+1} simulations took {toc-tic:.2f} seconds')

        valid_idx = [ii for ii, ssx in enumerate(x_sims) if ssx is not None]
        x_sims = jnp.array([ssx for ii, ssx in enumerate(x_sims) if ssx is not None])
        thetas = thetas[valid_idx, :]

        x_sims_all = jnp.append(x_sims_all, x_sims.reshape(-1, summary_dims), axis=0)
        thetas_all = jnp.append(thetas_all, thetas, axis=0)

        # standardise simulated summaries
        standardisation_params['x_sims_mean'] = jnp.mean(x_sims_all, axis=0)
        standardisation_params['x_sims_std'] = jnp.std(x_sims_all, axis=0)
        x_sims_all_standardised = (x_sims_all - standardisation_params['x_sims_mean']) / standardisation_params['x_sims_std']
        x_obs_standard = (x_obs - standardisation_params['x_sims_mean']) / standardisation_params['x_sims_std']

        # standardise parameters
        standardisation_params['theta_mean'] = jnp.mean(thetas_all, axis=0)
        standardisation_params['theta_std'] = jnp.std(thetas_all, axis=0)

        thetas_all_standardised = (thetas_all - standardisation_params['theta_mean']) / standardisation_params['theta_std']

        rng_key, sub_key = random.split(rng_key)

        flow = CouplingFlow(
            key=sub_key,
            base_dist=StandardNormal((theta_dims,)),
            transformer=RationalQuadraticSpline(knots=10, interval=5),  # 10 spline segments over [-5, 5].
            cond_dim=len(x_obs.flatten()),
            flow_layers=5,
            nn_width=50
            )

        rng_key, sub_key = random.split(rng_key)
        tic = time.time()
        flow, losses = fit_to_data(key=sub_key,
                                   dist=flow,
                                   x=thetas_all_standardised,
                                   condition=x_sims_all_standardised,
                                   max_epochs=500,
                                   max_patience=20,
                                   batch_size=256,
                                   )
        toc = time.time()
        flow_time += toc-tic

    rng_key, sub_key = random.split(rng_key)
    final_posterior_samples_standardised = flow.sample(
        sub_key,
        (num_final_posterior_samples,),
        x_obs_standard
    )

    final_posterior_samples = final_posterior_samples_standardised * standardisation_params['theta_std'] + standardisation_params['theta_mean']

    print(f'Total simulation time: {sim_time:.2f} seconds')
    print(f'Total flow training time: {flow_time:.2f} seconds')

    return final_posterior_samples

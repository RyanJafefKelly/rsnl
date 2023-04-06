"""Get ABC Rejection Samples for SIR model."""
import jax.numpy as jnp
import numpy as np
import jax.random as random
from rsnl.examples.misspec_ma1 import (assumed_dgp, get_prior,
                                       calculate_summary_statistics)
# import elfi
from functools import partial
import multiprocessing as mp
import matplotlib.pyplot as plt
import pickle as pkl


def get_ssx(prior, sim_fn, summ_fn, *args):
    """Get summary statistics for a given parameter set."""
    rng_key = args[0]
    rng_key, sub_key1, sub_key2 = random.split(rng_key, 3)
    theta = prior.sample(sub_key1)
    x = sim_fn(sub_key2, theta)
    ssx = summ_fn(x)
    return theta, ssx


def get_misspec_ma1_abc_samples():
    # N = 1_000  # num samples
    # for i in range(N):
    seed = 0
    rng_key = random.PRNGKey(seed)
    rng_key, sub_key = random.split(rng_key)
    # pseudo_true_param = jnp.array([0.0])
    # x_obs = true_dgp(true_params)
    x_obs = jnp.array([0.01, 0])
    num_processes = max(1, mp.cpu_count() - 1)
    pool = mp.Pool(num_processes)
    sim_fn = assumed_dgp
    summ_fn = calculate_summary_statistics
    # TODO! INCREASE BY AN ORDER OF MAGNITUDE
    N = 100000
    prior = get_prior()
    dgp_fn = partial(get_ssx, prior, sim_fn, summ_fn)
    sim_keys = random.split(rng_key, N)
    res = pool.starmap_async(dgp_fn, zip(sim_keys)).get(
        timeout=10000  # in seconds
    )
    thetas = np.empty((N, 1))
    x_sims = np.empty((N, 2))  # TODO...
    for i in range(N):
        thetas[i, :] = np.array(res[i][0])
        x_sims[i, :] = np.array(res[i][1])
    d = np.linalg.norm(x_sims - x_obs, axis=1)
    top_perc = 0.01
    top_idx = round(top_perc*N)
    idx = np.argsort(d)[:top_idx]
    thetas_acc = thetas[idx]
    for i in range(1):
        plt.hist(thetas_acc[:, i])
        plt.savefig(f'abc_theta_ma1_{i}.png')
        plt.clf()
    pkl.dump(thetas_acc, open('res/true_posterior_samples/misspec_ma1/true_posterior_samples.pkl', 'wb'))


if __name__ == '__main__':
    get_misspec_ma1_abc_samples()
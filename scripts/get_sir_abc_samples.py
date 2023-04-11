"""Get ABC Rejection Samples for SIR model."""
import jax.numpy as jnp
import numpy as np
import jax.random as random
from rsnl.examples.sir import (assumed_dgp, get_prior,
                               calculate_summary_statistics, true_dgp)
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
    x = sim_fn(sub_key2, *theta)
    ssx = summ_fn(x)
    return theta, ssx


def get_sir_abc_samples():
    # N = 1_000  # num samples
    # for i in range(N):
    seed = 0
    rng_key = random.PRNGKey(seed)
    rng_key, sub_key = random.split(rng_key)
    true_params = jnp.array([.1, .15])  # NOTE: arranged [gamma, beta]
    x_obs = true_dgp(sub_key, *true_params)
    x_obs = calculate_summary_statistics(x_obs)
    num_processes = max(1, mp.cpu_count() - 1)
    pool = mp.Pool(num_processes)
    sim_fn = assumed_dgp
    summ_fn = calculate_summary_statistics
    N = 1000000
    prior = get_prior()
    dgp_fn = partial(get_ssx, prior, sim_fn, summ_fn)
    sim_keys = random.split(rng_key, N)
    res = pool.starmap_async(dgp_fn, zip(sim_keys)).get(
        timeout=10000  # in seconds
    )
    thetas = np.empty((N, 2))
    x_sims = np.empty((N, 6))  # TODO...
    for i in range(N):
        thetas[i, :] = np.array(res[i][0])
        x_sims[i, :] = np.array(res[i][1])
    d = np.linalg.norm(x_sims - x_obs, axis=1)
    top_perc = 0.001
    top_idx = round(top_perc*N)
    idx = np.argsort(d)[:top_idx]
    thetas_acc = thetas[idx]
    for i in range(2):
        plt.hist(thetas_acc[:, i])
        plt.savefig(f'abc_theta_{i}.png')
        plt.clf()
    pkl.dump(thetas_acc, open('res/true_posterior_samples/sir/true_posterior_samples.pkl', 'wb'))
    # t1 = elfi.Prior('uniform', 0, 0.5)
    # t2 = elfi.Prior('uniform', t1, 0.5)
    # Y = elfi.Simulator(assumed_dgp, t1, t2, observed=x_obs)
    # S = elfi.Summary(calculate_summary_statistics, Y)
    # d = elfi.Distance('euclidean', S)
    # elfi.draw(d)
    # elfi.set_client('multiprocessing')
    # rej = elfi.Rejection(d, batch_size=1, seed=seed)

    # N = 1000
    # results = rej.sample(N, quantile=0.01)
    # results.plot_pairs()
    # plt.savefig('elfi_pairs.png')


if __name__ == '__main__':
    get_sir_abc_samples()

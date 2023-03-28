"""Implementation of the SIR model."""

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
from jax._src.prng import PRNGKeyArray  # for typing
import numpyro.distributions as dist  # type: ignore
from diffrax import (diffeqsolve, ControlTerm, Heun, MultiTerm, ODETerm,
                     SaveAt, VirtualBrownianTree, PIDController)
import jax.lax as lax
from numpyro.distributions import constraints


def base_dgp(rng_key: PRNGKeyArray,
             beta: jnp.ndarray,
             gamma: jnp.ndarray) -> jnp.ndarray:
    def vector_field(t, y, args):
        """Deterministic part of SDE."""
        s, i, r, R0 = y
        beta, gamma, eta, _ = args

        ds_dt = -beta * s * i
        di_dt = beta * s * i - gamma * i
        dr_dt = gamma * i
        dr0_dt = eta * ((beta/gamma) - R0)

        return jnp.array([ds_dt, di_dt, dr_dt, dr0_dt])

    def diffusion(t, y, args):
        """Stochastic part of SDE corresponding to R0."""
        _, _, _, R0 = y
        sigma = args[-1]  # Assuming the last element in 'args' is the scaling factor sigma
        dR0 = sigma * jnp.sqrt(R0)
        return jnp.array([0, 0, 0, dR0])

    t0 = 0
    t1 = 365

    # stochastic
    brownian_motion = VirtualBrownianTree(t0, t1, tol=1e-3, shape=(), key=rng_key)
    terms = MultiTerm(ODETerm(vector_field),
                      ControlTerm(diffusion, brownian_motion))

    dt0 = 0.05  # TODO: arbitrarily set ... shouldn't matter w/ controller?
    eta = 0.05
    sigma = 0.05  # scaling factor
    args = (beta, gamma, eta, sigma)
    R0_init = beta / gamma
    y0 = jnp.array([.999, 0.001, 0.0, R0_init])  # Init. proportion of S, I, R

    solver = Heun()
    saveat = SaveAt(ts=jnp.linspace(t0, t1, 365))
    stepsize_controller = PIDController(pcoeff=0.1, icoeff=0.3, dcoeff=0,
                                        rtol=1e-3, atol=1e-3)
    try:
        sol = diffeqsolve(terms, solver, t0=t0, t1=t1, dt0=dt0, y0=y0,
                          args=args,
                          saveat=saveat,
                          stepsize_controller=stepsize_controller
                        #   max_steps_reached=10_000,
                          )
    except Exception:
        return None
    return sol.ys[:, 1]  # only return infection data


def assumed_dgp(rng_key: PRNGKeyArray,
                beta: jnp.ndarray,
                gamma: jnp.ndarray) -> jnp.ndarray:
    x = base_dgp(rng_key, beta, gamma)
    return x


def true_dgp(rng_key: PRNGKeyArray,
             beta: jnp.ndarray,
             gamma: jnp.ndarray) -> jnp.ndarray:
    x = base_dgp(rng_key, beta, gamma)
    x = weekend_lag(x)
    return x


def calculate_summary_statistics(x):
    """Compute summary statistics for the given data."""
    # @jax.jit
    # @jax.vmap
    def autocorr_lag1(x):
        """Compute the lag-1 autocorrelation."""
        x1 = x[:-1]
        x2 = x[1:]
        x1_dif = x1 - x1.mean()
        x2_dif = x2 - x2.mean()
        numerator = (x1_dif * x2_dif).sum()
        denominator = jnp.sqrt((x1_dif ** 2).sum() * (x2_dif ** 2).sum())
        return numerator / denominator

    def cumulative_day(x, q):
        """Compute the day when q proportion of total infections is reached."""
        prop_i = (jnp.cumsum(x).T / jnp.sum(x)).T
        return jnp.argmax(prop_i > q)

    if x is None:
        return None

    summaries = [
        jnp.log(jnp.mean(x)),
        jnp.log(jnp.median(x)),
        jnp.log(jnp.max(x)),
        jnp.log(jnp.argmax(x) + 1),  # +1 incase 0 is max_day
        jnp.log(cumulative_day(x, 0.5)),
        autocorr_lag1(x),
    ]

    summaries = jnp.array(summaries)
    # TODO? HACKY
    summaries = jnp.nan_to_num(summaries, nan=0, posinf=0, neginf=0)
    return summaries


def weekend_lag(x, misspecify_multiplier=0.95):
    """Reduce the number of recorded infections on the weekend."""
    x = jnp.array(x)
    sat_idx, sun_idx, mon_idx = [jnp.arange(i, 365, 7) for i in range(1, 4)]

    sat_new = x[sat_idx] * misspecify_multiplier
    sun_new = x[sun_idx] * misspecify_multiplier
    missed_cases = (x[sat_idx] - sat_new) + (x[sun_idx] - sun_new)
    mon_new = x[mon_idx] + missed_cases

    x = x.at[sat_idx].set(sat_new[sat_idx])
    x = x.at[sun_idx].set(sun_new[sun_idx])
    x = x.at[mon_idx].set(mon_new[mon_idx])

    return x


# def misspecify(x, misspecify_multiplier=0.95):
#     # TODO! COPIED DIRECTLY
#     x = np.array(x)
#     x = x.copy()
#     sat_idx, sun_idx, mon_idx = [range(i, 365, 7) for i in range(1, 4)]
#     sat_new = x[:, sat_idx] * misspecify_multiplier
#     sun_new = x[:, sun_idx] * misspecify_multiplier
#     missed_cases = (x[:, sat_idx] - sat_new) + (x[:, sun_idx] - sun_new)
#     mon_new = x[:, mon_idx] + missed_cases
#     for idx, new in zip([sat_idx, sun_idx, mon_idx], [sat_new, sun_new, mon_new]):
#         x[:, idx] = new
#     return jnp.array(x)


class CustomPrior(dist.Distribution):
    """Uniform with second draw conditioned on first."""

    def __init__(self, low=0.0, high=1.0, validate_args=False):
        self.low, self.high = low, high
        batch_shape = lax.broadcast_shapes(jnp.shape(low), jnp.shape(high))
        self._support = constraints.interval(low, high)
        super().__init__(batch_shape, validate_args=validate_args, event_shape=(2,))

    def sample(self, key, sample_shape=()):
        key, sub_key1 = random.split(key)
        shape = sample_shape + self.batch_shape
        u1 = random.uniform(key, shape=shape, minval=self.low, maxval=self.high)
        u2 = random.uniform(sub_key1, shape=shape, minval=u1, maxval=self.high)
        return jnp.concatenate([u1, u2])

    def log_prob(self, value):
        # if value.ndim == 1:
        #     value = jnp.expand_dims(value, axis=1)
        # shape = lax.broadcast_shapes(jnp.shape(value), self.batch_shape)
        # assume last column is for t1 t2
        # p1 = -jnp.broadcast_to(jnp.log(self.high - self.low), shape[1:])
        # p2 = -jnp.broadcast_to(jnp.log(self.high - value[0, ...]), shape[1:])
        return 1  # TODO: WHO CARES

    def mean(self):
        # TODO:
        return jnp.array([0.25, 0.375])

    def variance(self):  # TODO
        var = jnp.sum((1/12) * (np.array([0.5, 0.5]) ** 2))
    #   dummy = jnp.array(var)
        return var

    @constraints.dependent_property(is_discrete=False, event_dim=2)
    def support(self):
        return self._support


def get_prior():
    prior = dist.Uniform(low=jnp.repeat(0.0, 2),
                         high=jnp.repeat(0.5, 2))
    # TODO: ADD CONDITION gamma > lambda
    # first prior ... lambda ... second prior ... gamma
    # prior = CustomPrior(low=0.0, high=0.5)
    return prior

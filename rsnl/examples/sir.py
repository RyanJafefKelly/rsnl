"""Implementation of the SIR model."""

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
from jax._src.prng import PRNGKeyArray  # for typing
import numpyro.distributions as dist  # type: ignore
from diffrax import (diffeqsolve, ControlTerm, Euler, EulerHeun, Heun, MultiTerm, ODETerm,
                     SaveAt, VirtualBrownianTree, PIDController)
import jax.lax as lax
from numpyro.distributions import constraints
import gc, psutil, sys  # TODO: TESTING


def clear_caches():
    """Clear cache to stop memory issues with JAX + diffrax."""
    process = psutil.Process()
    if process.memory_info().rss > 0.25 * 2 ** 30:  # TODO brought down from 500 to 250 MB memory usage
        for module_name, module in sys.modules.copy().items():
            if module_name.startswith("jax"):
                for obj_name in dir(module):
                    # print('obj_name ', obj_name)
                    obj = getattr(module, obj_name)
                    if hasattr(obj, "cache_clear"):
                        # print('obj with cache: ', obj)
                        try:
                            obj.cache_clear()
                        except:
                            pass
    gc.collect()
    diffeqsolve._cached.clear_cache()
    print("Cache cleared")


def base_dgp(rng_key: PRNGKeyArray,
             gamma: jnp.ndarray,
             beta: jnp.ndarray) -> jnp.ndarray:
    """Base DGP for the SIR model."""
    def drift(t, y, args):
        """Deterministic part of SDE."""
        s, i, r, R0 = y
        gamma, beta, eta, _ = args

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
    terms = MultiTerm(ODETerm(drift),
                      ControlTerm(diffusion, brownian_motion))

    dt0 = 0.05  # TODO: arbitrarily set ... shouldn't matter w/ controller?
    eta = 0.05
    sigma = 0.05  # scaling factor
    args = (gamma, beta, eta, sigma)  # TODO: SWITCHED GAMMA, BETA AROUND
    R0_init = beta / gamma
    print(f'beta: {beta}, gamma: {gamma}')
    y0 = jnp.array([.999, 0.001, 0.0, R0_init])  # Init. proportion of S, I, R

    solver = Heun(scan_stages=True)  # TODO: TRY DIFF SOLVER
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
    except Exception:  # NOTE: CAN TURN OFF EXCEPTIONS
        return None
    clear_caches()
    print(f"Process uses {psutil.Process().memory_info().rss / (1024 * 1024)} MB memory.")
    return 1e+6 * sol.ys[:, 1]  # only return infection data


def assumed_dgp(rng_key: PRNGKeyArray,
                gamma: jnp.ndarray,
                beta: jnp.ndarray,
                *args,
                **kwargs) -> jnp.ndarray:
    """Assumed DGP for the SIR model."""
    x = base_dgp(rng_key, gamma, beta)
    return x


def true_dgp(rng_key: PRNGKeyArray,
             gamma: jnp.ndarray,
             beta: jnp.ndarray) -> jnp.ndarray:
    """True DGP for the SIR model including weekend reporting lag."""
    x = base_dgp(rng_key, gamma, beta)
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
    summaries = jnp.nan_to_num(summaries, nan=-1, posinf=-1, neginf=-1)
    print('summaries: ', summaries)
    return summaries


def weekend_lag(x, misspecify_multiplier=0.95):
    """Reduce the number of recorded infections on the weekend."""
    x = jnp.array(x)
    sat_idx, sun_idx, mon_idx = [jnp.arange(i, 365, 7) for i in range(1, 4)]

    sat_new = x[sat_idx] * misspecify_multiplier
    sun_new = x[sun_idx] * misspecify_multiplier
    missed_cases = (x[sat_idx] - sat_new) + (x[sun_idx] - sun_new)
    mon_new = x[mon_idx] + missed_cases

    x = x.at[sat_idx].set(sat_new)
    x = x.at[sun_idx].set(sun_new)
    x = x.at[mon_idx].set(mon_new)

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
    # NOTE: 
    def __init__(self, low=0.0, high=1.0, validate_args=False):
        self.low, self.high = low, high
        event_shape = (2,)
        self._u1 = None
        batch_shape = lax.broadcast_shapes(jnp.shape(low), jnp.shape(high))
        self._support = constraints.interval(low, high)
        super().__init__(batch_shape, validate_args=validate_args,
                         event_shape=event_shape)

    def sample(self, key, sample_shape=(1,)):
        key, sub_key1 = random.split(key)
        shape = sample_shape + self.batch_shape
        u1 = random.uniform(key, shape=shape, minval=self.low, maxval=self.high)
        self._u1 = u1
        u2 = random.uniform(sub_key1, shape=shape, minval=u1, maxval=self.high)
        return jnp.concatenate([u1, u2])

    def log_prob(self, value):
        u1, u2 = value[0, ...], value[1, ...]
        # if value.ndim == 1:
        #     value = jnp.expand_dims(value, axis=1)
        shape = lax.broadcast_shapes(jnp.shape(value), self.batch_shape)
        # assume last column is for t1 t2

        log_pdf_u1 = jnp.where((self.low <= u1) & (u1 <= self.high),
                        -jnp.log(self.high - self.low), -jnp.inf)
        log_pdf_u1 = jnp.broadcast_to(log_pdf_u1, shape[1:])
        log_pdf_u2 = jnp.where((u1 <= u2) & (u2 <= self.high),
                                -jnp.log(self.high - u1), -jnp.inf)
        log_pdf_u2 = jnp.broadcast_to(log_pdf_u2, shape[1:])
        return log_pdf_u1 + log_pdf_u2

    @property
    def mean(self):
        # TODO:
        mean = jnp.array([0.25, 0.375])  # default
        if self._u1 is not None:
            u2_mean = (self.high + self._u1) / 2
            mean = jnp.array([0.25, u2_mean])
        return mean

    @property
    def variance(self):  # TODO
        var = (1/12) * (jnp.array([0.5, 0.375]) ** 2)
        if self._u1 is not None:
            u1_var = ((self.high - self.low) ** 2) / 12
            u2_var = ((self.high - self._u1) ** 2) / 12
            var = jnp.array([u1_var, u2_var])
    #   dummy = jnp.array(var)
        return var

    @constraints.dependent_property(is_discrete=False, event_dim=2)
    def support(self):
        return self._support


def get_prior():
    """Return prior distribution for SIR example."""
    # prior = CustomPrior(low=0.0, high=0.5)
    prior = dist.Uniform(low=jnp.array([0.0, 0.0]),
                         high=jnp.array([0.5, 0.5]))
    return prior


def true_posterior(x_obs: jnp.ndarray,
                   prior: dist.Distribution) -> jnp.ndarray:
    """Return "true" posterior for SIR example."""
    # NOTE: no tractable distribution for this model
    # TODO: should obtain thetas from a good run of the model

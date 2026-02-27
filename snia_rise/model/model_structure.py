import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpy.typing import ArrayLike
from numpyro import distributions as dist

from ..constants import EPS, T_PIVOT
from .priors import (
    sample_alpha_0,
    sample_alpha_1,
    sample_amp_prime,
    sample_base,
    sample_beta,
    sample_hierarchical_params,
    sample_t_fl,
    sample_t_rise,
)

####################################################################################################
##                      Power-law rise function for SNe Ia light curves                           ##
####################################################################################################


@jax.jit
def f_t(
    t: float | ArrayLike,
    t_fl: float | ArrayLike,
    base: float | ArrayLike,
    amp_prime: float | ArrayLike,
    alpha_0: float | ArrayLike,
    alpha_1: float | ArrayLike = 0.0,
    t_pivot: float | ArrayLike = T_PIVOT,
    eps: float | ArrayLike = EPS,
):
    """
    Calculate the flux with a power-law rise model.

    Parameters:
    -----------
    t : float or array-like
        Time value.
    t_fl : float or array-like
        Time of the first light.
    base : float or array-like
        Baseline flux.
    amp_prime : float or array-like
        Proportionality factor.
    alpha_0 : float or array-like
        Rising power-law index.
    alpha_1 : float or array-like
        Correction factor for the power-law rise.
    t_pivot : float or array-like
        Pivot time for the power-law rise.
    eps : float, optional, default = EPS
        Small value to avoid numerical issues when t - t_fl is small and alpha_0 < 1

    Returns:
    --------
    float | ArrayLike
        The calculated value of f(t).
    """
    u = jnp.maximum(t - t_fl, eps)
    ln_u_p = jnp.log(u / t_pivot)
    exponent = alpha_0 * (1 + alpha_1 * u / t_pivot)
    rise = amp_prime * jnp.exp(exponent * ln_u_p)
    f = jnp.where(t < t_fl, base, rise + base)
    return f


@jax.jit
def df_t_dt_fl(t, t_fl, base, amp_prime, alpha_0, alpha_1, t_pivot=T_PIVOT, eps=EPS):
    """
    Calculate the derivative of f(t) with respect to t_fl in an analytic manner.
    """
    # Get f_t
    u = jnp.maximum(t - t_fl, eps)
    ln_u_p = jnp.log(u / t_pivot)
    exponent = alpha_0 * (1 + alpha_1 * u / t_pivot)
    f = amp_prime * jnp.exp(exponent * ln_u_p)

    term_power = exponent / u
    term_log = (alpha_0 * alpha_1 / t_pivot) * ln_u_p

    # Total derivative w.r.t u
    df_du = f * (term_power + term_log)

    # Chain rule: df/dtfl = df/du * du/dtfl = df/du * (-1)
    df_dtfl = -df_du

    # Ensure points before t_fl are exactly 0
    return jnp.where(t < t_fl, 0.0, df_dtfl)


####################################################################################################
##                   Probabilistic models for SNe Ia light curve modeling                         ##
####################################################################################################


def hierarchical_model(
    t: list,
    flux: list = None,
    flux_err: list = None,
    beta: list = None,
    t0_err: list = None,
    idx_obj: list = None,
    idx_fcqfid: list = None,
    idx_filt: list = None,
    prior_config: dict = {},
):
    """
    Unified hierarchical Bayesian model for supernova early-time light curves.

    This function encompasses three model variants controlled by the
    `correlation_structure` parameter:

    1."mvn" (default): Full multivariate normal with correlations
       [t_rise_j, alpha_j_g, alpha_j_r, ...] ~ MVN(mu, Sigma)

    2."independent": Independent hierarchical priors (diagonal covariance)
       t_rise_j ~ Normal(mean_t_rise, sigma_t_rise)
       alpha_0_i ~ Hierarchical(mean_alpha_0[i], sigma_alpha[i])

    3."trise_only": Only t_rise hierarchical, alpha_0 unpooled
       t_rise_j ~ Normal(mean_t_rise, sigma_t_rise)
       alpha_0_i ~ [prior] per object and filter

    Posterior samples will have consistent shapes across all variants:
    - alpha_0: (n_chains, n_samples, n_obj, n_filt)
    - alpha_1: (n_chains, n_samples, n_obj, n_filt) [if curved_power_law]
    - A: (n_chains, n_samples, n_obj, n_filt)
    - t_rise: (n_chains, n_samples, n_obj)
    - C, beta: (n_chains, n_samples, n_fcqfid)
    - Sigma: (n_chains, n_samples, d, d) - covariance matrix
    - Corr: (n_chains, n_samples, d, d) - correlation matrix

    Parameters
    ----------
    t : list
        Time values (phase) for each observation
    flux : list
        Flux values
    flux_err : list
        Flux uncertainties
    beta : list, optional
        Uncertainty scale factor for each observation. If None, defaults to 1.0.
        If sample_beta=True in prior_config, this parameter is ignored and beta is sampled.
    t0_err : list
        Uncertainties on t0 (0 for mocked data)
    idx_obj : list
        Object index for each observation
    idx_fcqfid : list
        FCQF ID index for each observation
    idx_filt : list
        Filter index for each observation
    prior_config : dict
        Configuration dictionary with keys:
        - correlation_structure :  str, default="mvn"
            "mvn", "independent", or "tfl_only"
        - prior_type : str, default="Maximum_Entropy"
            For "independent" mode: type of prior for alpha_0
        - rise_model : str, choices=["power_law", "curved_power_law"]
        - mean_alpha_0 :  float, default=2 (tfl_only)
        - sigma_alpha_0 :  float or None, default=None (tfl_only)
        - min_alpha_0 :  float, default=1
        - max_alpha_0 :  float, default=5
        - sample_beta : bool, default=False
            Whether to sample beta as a free parameter (True) or fix it to 1 (False)
        - beta_scale : float, default=0.2
            Scale parameter for HalfNormal prior on log(beta), ensuring beta >= 1
    """
    # Setup
    correlation_structure = prior_config.get("correlation_structure", "mvn")

    n_fcqfid = len(np.unique(idx_fcqfid))
    n_filt = len(np.unique(idx_filt))
    n_obj = len(np.unique(idx_obj))

    # Observation-level parameters (n_fcqfid,)
    base = sample_base(n_fcqfid)

    # Handle beta: sample if requested, otherwise use provided values or default to 1.0
    sample_beta_flag = prior_config.get("sample_beta", False)
    if sample_beta_flag:
        beta_fcqfid = sample_beta(n_fcqfid, prior_config)
    else:
        # Use provided beta values or default to ones
        if beta is None:
            beta = np.ones_like(flux)
        # beta is per observation, no need to index by fcqfid
        beta_obs_provided = beta

    # Hierarchical structure for t_rise, amp and alpha_0
    # t_rise:   shape (n_obj,)
    # amp:      shape (n_obj, n_filt)
    # alpha_0:  shape (n_obj, n_filt)
    t_rise, amp_prime, alpha_0 = sample_hierarchical_params(
        n_obj,
        n_filt,
        correlation_structure=correlation_structure,
        prior_config=prior_config,
    )

    # t_fl: shape (n_obj,)
    t_fl = sample_t_fl(n_obj, t_rise, t0_err)

    # Non-hierarchical parameters
    # alpha_1, amp: shape (n_obj, n_filt)
    rise_model = prior_config.get("rise_model", "power_law")
    if rise_model == "curved_power_law":
        with numpyro.plate("obj", n_obj, dim=-2):
            with numpyro.plate("filt", n_filt, dim=-1):
                alpha_1 = sample_alpha_1()
    else:
        alpha_1 = jnp.zeros((n_obj, n_filt))

    # shape: (n_obs,)
    t_fl_obs = t_fl[idx_obj]
    base_obs = base[idx_fcqfid]
    amp_prime_obs = amp_prime[idx_obj, idx_filt]
    alpha_0_obs = alpha_0[idx_obj, idx_filt]
    alpha_1_obs = alpha_1[idx_obj, idx_filt]
    t0_err_obs = t0_err[idx_obj]

    # Apply beta scaling
    if sample_beta_flag:
        beta_obs = beta_fcqfid[idx_fcqfid]
    else:
        beta_obs = beta_obs_provided
    flux_err_obs = flux_err * beta_obs

    # Add extra uncertainty component from t0_err via error propagation
    df_dtfl = df_t_dt_fl(t, t_fl_obs, base_obs, amp_prime_obs, alpha_0_obs, alpha_1_obs)
    flux_err_obs = jnp.sqrt(flux_err_obs**2 + (df_dtfl * t0_err_obs) ** 2)

    # Likelihood
    with numpyro.plate("data", len(t)):
        numpyro.sample(
            "flux",
            dist.Normal(
                f_t(t, t_fl_obs, base_obs, amp_prime_obs, alpha_0_obs, alpha_1_obs),
                flux_err_obs,
            ),
            obs=flux,
        )


def unpooled_model(
    t: list,
    flux: list = None,
    flux_err: list = None,
    beta: list = None,
    t0_err: list = None,
    idx_obj: list = None,
    idx_fcqfid: list = None,
    idx_filt: list = None,
    prior_config: dict = {},
):
    """
    Unpooled model:  each object independent, no hierarchical structure.

    Model: alpha_0_k ~ [prior] per filter group (shared across objects),
           t_rise_j ~ Uniform(0, 40) per object

    NOTE: Uses idx_filt_gr for alpha_0 indexing to share parameters across objects
    for the same filter type (g vs r), maintaining consistent model semantics.
    """

    n_fcqfid = len(np.unique(idx_fcqfid))
    n_filt = len(np.unique(idx_filt))
    n_obj = len(np.unique(idx_obj))

    # base, beta: shape (n_fcqfid,)
    base = sample_base(n_fcqfid)

    # Handle beta: sample if requested, otherwise use provided values or default to 1.0
    sample_beta_flag = prior_config.get("sample_beta", False)
    if sample_beta_flag:
        beta_fcqfid = sample_beta(n_fcqfid, prior_config)
    else:
        if beta is None:
            beta = np.ones_like(flux)
        beta_obs_provided = beta

    # t_rise: shape (n_obj,)
    with numpyro.plate("obj", n_obj):
        t_rise = sample_t_rise(prior_config)

    # alpha_1: shape (n_obj, n_filt)
    rise_model = prior_config.get("rise_model", "power_law")
    if rise_model == "curved_power_law":
        with numpyro.plate("filt", n_filt, dim=-1):
            with numpyro.plate("obj", n_obj, dim=-2):
                alpha_1 = sample_alpha_1()
    else:
        alpha_1 = jnp.zeros((n_obj, n_filt))

    # alpha_0, amp: shape (n_obj, n_filt)
    with numpyro.plate("obj", n_obj, dim=-2):
        with numpyro.plate("filt", n_filt, dim=-1):
            alpha_0 = sample_alpha_0(prior_config=prior_config)
            amp_prime = sample_amp_prime()

    # t_fl: shape (n_obj,)
    t_fl = sample_t_fl(n_obj, t_rise, t0_err)

    # shape: (n_obs,)
    t_fl_obs = t_fl[idx_obj]
    base_obs = base[idx_fcqfid]
    amp_prime_obs = amp_prime[idx_obj, idx_filt]
    alpha_0_obs = alpha_0[idx_obj, idx_filt]
    alpha_1_obs = alpha_1[idx_obj, idx_filt]
    t0_err_obs = t0_err[idx_obj]

    # Apply beta scaling
    if sample_beta_flag:
        beta_obs = beta_fcqfid[idx_fcqfid]
    else:
        beta_obs = beta_obs_provided
    flux_err_obs = flux_err * beta_obs

    # Add extra uncertainty component from t0_err via error propagation
    df_dtfl = df_t_dt_fl(t, t_fl_obs, base_obs, amp_prime_obs, alpha_0_obs, alpha_1_obs)
    flux_err_obs = jnp.sqrt(flux_err_obs**2 + (df_dtfl * t0_err_obs) ** 2)

    # Likelihood
    with numpyro.plate("data", len(t)):
        numpyro.sample(
            "flux",
            dist.Normal(
                f_t(t, t_fl_obs, base_obs, amp_prime_obs, alpha_0_obs, alpha_1_obs),
                flux_err_obs,
            ),
            obs=flux,
        )


def pooled_model(
    t: list,
    flux: list = None,
    flux_err: list = None,
    beta: list = None,
    t0_err: list = None,
    idx_obj: list = None,
    idx_fcqfid: list = None,
    idx_filt: list = None,
    prior_config: dict = {},
):
    """
    Pooled model: all objects share alpha, while t_rise varies per object.

    Model: alpha_0_k ~ [prior] per filter group, t_rise ~ Uniform per object
    """
    # Setup
    n_obj = len(np.unique(idx_obj))
    n_fcqfid = len(np.unique(idx_fcqfid))
    n_filt = len(np.unique(idx_filt))

    # base, beta: shape (n_fcqfid,)
    base = sample_base(n_fcqfid)

    # Handle beta: sample if requested, otherwise use provided values or default to 1.0
    sample_beta_flag = prior_config.get("sample_beta", False)
    if sample_beta_flag:
        beta_fcqfid = sample_beta(n_fcqfid, prior_config)
    else:
        if beta is None:
            beta = np.ones_like(flux)
        beta_obs_provided = beta

    # t_rise: shape (n_obj,)
    with numpyro.plate("obj", n_obj):
        t_rise = sample_t_rise(prior_config)

    # t_fl: shape (n_obj,)
    t_fl = sample_t_fl(n_obj, t_rise, t0_err)

    # alpha_1: shape (n_filt,)
    rise_model = prior_config.get("rise_model", "power_law")
    if rise_model == "curved_power_law":
        with numpyro.plate("filt", n_filt, dim=-1):
            alpha_1 = sample_alpha_1()
    else:
        alpha_1 = jnp.zeros((n_filt,))

    # alpha_0: shape (n_filt,)
    # amp: shape (n_obj, n_filt)
    with numpyro.plate("filt", n_filt, dim=-1):
        alpha_0 = sample_alpha_0(prior_config=prior_config)

        with numpyro.plate("obj", n_obj, dim=-2):
            amp_prime = sample_amp_prime()

    # shape: (n_obs,)
    t_fl_obs = t_fl[idx_obj]
    base_obs = base[idx_fcqfid]
    amp_prime_obs = amp_prime[idx_obj, idx_filt]
    alpha_0_obs = alpha_0[idx_filt]
    alpha_1_obs = alpha_1[idx_filt]
    t0_err_obs = t0_err[idx_obj]

    # Apply beta scaling
    if sample_beta_flag:
        beta_obs = beta_fcqfid[idx_fcqfid]
    else:
        beta_obs = beta_obs_provided
    flux_err_obs = flux_err * beta_obs

    # Add extra uncertainty component from t0_err via error propagation
    df_dtfl = df_t_dt_fl(t, t_fl_obs, base_obs, amp_prime_obs, alpha_0_obs, alpha_1_obs)
    flux_err_obs = jnp.sqrt(flux_err_obs**2 + (df_dtfl * t0_err_obs) ** 2)

    # Likelihood
    with numpyro.plate("data", len(t)):
        numpyro.sample(
            "flux",
            dist.Normal(
                f_t(t, t_fl_obs, base_obs, amp_prime_obs, alpha_0_obs, alpha_1_obs),
                flux_err_obs,
            ),
            obs=flux,
        )

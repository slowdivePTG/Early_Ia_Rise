import numpy as np

import numpyro
import jax.numpy as jnp
from numpyro import distributions as dist
from numpy.typing import ArrayLike


from .priors import (
    sample_fcqf_params,
    sample_alpha_0,
    sample_alpha_1,
    sample_amp_prime,
    sample_t_rise,
    sample_t_fl,
    sample_hierarchical_params,
)

####################################################################################################
##                      Power-law rise function for SNe Ia light curves                           ##
####################################################################################################


def f_t(
    t: float | ArrayLike,
    t_fl: float | ArrayLike,
    base: float | ArrayLike,
    amp: float | ArrayLike,
    alpha_0: float | ArrayLike,
    alpha_1: float | ArrayLike = 0.0,
    eps: float = 1e-10,
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
    amp : float or array-like
        Proportionality factor.
    alpha_0 : float or array-like
        Rising power-law index.
    alpha_1 : float or array-like
        Correction factor for the power-law rise.
    eps : float, optional, default = 1e-10
        Small value to avoid numerical issues when t - t_fl is small and alpha_0 < 1

    Returns:
    --------
    float | ArrayLike
        The calculated value of f(t).
    """
    du = jnp.maximum(t - t_fl, eps)
    f = jnp.where(t < t_fl, 0, amp * jnp.power(du, alpha_0 * (1 + alpha_1 * du))) + base
    return f


####################################################################################################
##                   Probabilistic models for SNe Ia light curve modeling                         ##
####################################################################################################


def hierarchical_model(
    t: list,
    flux: list = None,
    flux_err: list = None,
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
    t0_err : list
        Uncertainties on t0 (0 for mocked data)
    idx_obj : list
        Object index for each observation
    idx_fcqfid : list
        FCQF ID index for each observation
    idx_filt_gr : list
        Filter group index for each observation
    prior_config : dict
        Configuration dictionary with keys:
        - correlation_structure :  str, default="mvn"
            "mvn", "independent", or "tfl_only"
        - prior_type : str, default="Maximum_Entropy"
            For "independent" mode: type of prior for alpha_0
        - curved_power_law : bool, default=False
        - mean_alpha_0 :  float, default=2 (tfl_only)
        - sigma_alpha_0 :  float or None, default=None (tfl_only)
        - min_alpha_0 :  float, default=1
        - max_alpha_0 :  float, default=5
        - curved_power_law : bool, default=False
    """
    # Setup
    correlation_structure = prior_config.get("correlation_structure", "mvn")

    n_fcqfid = len(np.unique(idx_fcqfid))
    n_filt = len(np.unique(idx_filt))
    n_obj = len(np.unique(idx_obj))

    # Observation-level parameters (n_fcqfid,)
    base, beta = sample_fcqf_params(n_fcqfid)

    # Hierarchical structure for t_rise and alpha_0
    # t_rise:   shape (n_obj,)
    # alpha_0: shape (n_obj, n_filt)
    t_rise, alpha_0 = sample_hierarchical_params(
        n_obj,
        n_filt,
        correlation_structure=correlation_structure,
        prior_config=prior_config,
    )

    # t_fl: shape (n_obj,)
    t_fl = sample_t_fl(n_obj, t_rise, t0_err)

    # Non-hierarchical parameters
    # alpha_1, amp: shape (n_obj, n_filt)
    curved_power_law = prior_config.get("curved_power_law", False)
    if curved_power_law:
        with numpyro.plate("filt", n_filt, dim=-1):
            with numpyro.plate("obj", n_obj, dim=-2):
                alpha_1 = sample_alpha_1()
    else:
        alpha_1 = jnp.zeros((n_obj, n_filt))

    with numpyro.plate("filt", n_filt, dim=-1):
        with numpyro.plate("obj", n_obj, dim=-2):
            amp_prime = sample_amp_prime()
            amp = numpyro.deterministic("A", amp_prime / jnp.power(10, alpha_0))

    # Likelihood
    with numpyro.plate("data", len(t)):
        numpyro.sample(
            "flux",
            dist.Normal(
                f_t(
                    t,
                    t_fl[idx_obj],
                    base[idx_fcqfid],
                    amp[idx_obj, idx_filt],
                    alpha_0[idx_obj, idx_filt],
                    alpha_1[idx_obj, idx_filt],
                ),
                flux_err * beta[idx_fcqfid],
            ),
            obs=flux,
        )


def unpooled_model(
    t: list,
    flux: list = None,
    flux_err: list = None,
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
    base, beta = sample_fcqf_params(n_fcqfid)

    # alpha_0, amp: shape (n_obj, n_filt)
    with numpyro.plate("filt", n_filt, dim=-1):
        with numpyro.plate("obj", n_obj, dim=-2):
            alpha_0 = sample_alpha_0(prior_config=prior_config)
            amp_prime = sample_amp_prime()
            amp = numpyro.deterministic("A", amp_prime / jnp.power(10, alpha_0))

    # alpha_1: shape (n_obj, n_filt)
    curved_power_law = prior_config.get("curved_power_law", False)
    if curved_power_law:
        with numpyro.plate("filt", n_filt, dim=-1):
            with numpyro.plate("obj", n_obj, dim=-2):
                alpha_1 = sample_alpha_1()
    else:
        alpha_1 = jnp.zeros((n_obj, n_filt))

    # t_rise: shape (n_obj,)
    with numpyro.plate("obj", n_obj):
        t_rise = sample_t_rise(prior_config)

    # t_fl: shape (n_obj,)
    t_fl = sample_t_fl(n_obj, t_rise, t0_err)

    # Likelihood
    with numpyro.plate("data", len(t)):
        numpyro.sample(
            "flux",
            dist.Normal(
                f_t(
                    t,
                    t_fl[idx_obj],
                    base[idx_fcqfid],
                    amp[idx_obj, idx_filt],
                    alpha_0[idx_obj, idx_filt],
                    alpha_1[idx_obj, idx_filt],
                ),
                flux_err * beta[idx_fcqfid],
            ),
            obs=flux,
        )


def pooled_model(
    t: list,
    flux: list = None,
    flux_err: list = None,
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
    base, beta = sample_fcqf_params(n_fcqfid)

    # alpha_0: shape (n_filt,)
    # amp: shape (n_obj, n_filt)
    with numpyro.plate("filt", n_filt, dim=-1):
        alpha_0 = sample_alpha_0(prior_config=prior_config)

        with numpyro.plate("obj", n_obj, dim=-2):
            amp_prime = sample_amp_prime()
            amp = numpyro.deterministic("A", amp_prime / jnp.power(10, alpha_0))

    # alpha_1: shape (n_filt,)
    curved_power_law = prior_config.get("curved_power_law", False)
    if curved_power_law:
        with numpyro.plate("filt", n_filt, dim=-1):
            alpha_1 = sample_alpha_1()
    else:
        alpha_1 = jnp.zeros((n_filt,))

    # t_rise: shape (n_obj,)
    with numpyro.plate("obj", n_obj):
        t_rise = sample_t_rise(prior_config)

    # t_fl: shape (n_obj,)
    t_fl = sample_t_fl(n_obj, t_rise, t0_err)

    # Likelihood
    with numpyro.plate("data", len(t)):
        numpyro.sample(
            "flux",
            dist.Normal(
                f_t(
                    t,
                    t_fl[idx_obj],
                    base[idx_fcqfid],
                    amp[idx_obj, idx_filt],
                    alpha_0[idx_filt],
                    alpha_1[idx_filt],
                ),
                flux_err * beta[idx_fcqfid],
            ),
            obs=flux,
        )

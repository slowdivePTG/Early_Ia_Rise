import numpy as np

import numpyro
import jax.numpy as jnp
from numpyro import distributions as dist
from numpy.typing import ArrayLike


from .priors import (
    sample_observation_params,
    sample_alpha_prior,
    sample_filter_level_params,
    sample_tfl_params,
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
    idx_obj: list = None,
    idx_fcqfid: list = None,
    idx_filt: list = None,
    idx_filt_gr: list = None,
    prior_config: dict = {},
):
    """
    Unified hierarchical Bayesian model for supernova early-time light curves.

    This function encompasses three model variants controlled by the
    `correlation_structure` parameter:

    1."mvn" (default): Full multivariate normal with correlations
       [t_fl_j, alpha_j_g, alpha_j_r, ...] ~ MVN(mu, Sigma)

    2."independent": Independent hierarchical priors (diagonal covariance)
       t_fl_j ~ Normal(mean_t_fl, sigma_t_fl)
       alpha_0_i ~ Hierarchical(mean_alpha_0[i], sigma_alpha[i])

    3."tfl_only": Only t_fl hierarchical, alpha_0 at population mean
       t_fl_j ~ Normal(mean_t_fl, sigma_t_fl)
       alpha_0_i = mean_alpha_0[i]  (fixed at population mean)

    Posterior samples will have consistent shapes across all variants:
    - alpha_0: (n_chains, n_samples, n_obj, n_filt)
    - A: (n_chains, n_samples, n_obj, n_filt)
    - t_fl: (n_chains, n_samples, n_obj)
    - C, beta: (n_chains, n_samples, n_fcqfid)
    - alpha_1: (n_chains, n_samples, n_filt) [if curved_power_law]
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
    idx_obj : list
        Object index for each observation
    idx_fcqfid : list
        FCQF ID index for each observation
    idx_filt : list
        Filter index for each observation
    idx_filt_gr : list
        Filter group index for each observation
    prior_config : dict
        Configuration dictionary with keys:
        - correlation_structure :  str, default="mvn"
            "mvn", "independent", or "tfl_only"
        - prior_type : str, default="Maximum_Entropy"
            For "independent" mode: type of prior for alpha_0
        - curved_power_law : bool, default=False
        - min_alpha_0 :  float, default=1
        - max_alpha_0 :  float, default=5
    """
    # Setup
    correlation_structure = prior_config.get("correlation_structure", "mvn")
    prior_type = prior_config.get("prior_type", "Maximum_Entropy")
    curved_power_law = prior_config.get("curved_power_law", False)
    min_alpha_0 = prior_config.get("min_alpha_0", 1)
    max_alpha_0 = prior_config.get("max_alpha_0", 5)
    assert min_alpha_0 >= 0, "min_alpha_0 must be non-negative"

    n_fcqfid = len(np.unique(idx_fcqfid))
    n_filt = len(np.unique(idx_filt))
    n_filt_gr = len(np.unique(idx_filt_gr))
    n_obj = len(np.unique(idx_obj))

    # Map filters to filter groups
    idx_filt_loc = np.zeros(n_filt, dtype=int)
    for k, filt in enumerate(np.unique(idx_filt)):
        idx = np.unique(idx_filt_gr[idx_filt == filt])
        assert len(idx) == 1, "Multiple filters assigned to same filter group"
        idx_filt_loc[k] = idx[0]

    # Observation-level parameters (n_fcqfid,)
    base, beta = sample_observation_params(n_fcqfid)

    # Hierarchical structure for t_fl and alpha_0
    # t_fl:   shape (n_obj,)
    # alpha_0: shape (n_obj, n_filt)
    t_fl, alpha_0, hyperparams = sample_hierarchical_params(
        n_obj,
        n_filt_gr,
        n_filt,
        idx_filt_loc,
        min_alpha_0,
        max_alpha_0,
        correlation_structure=correlation_structure,
        prior_type=prior_type,
    )

    # Filter-level parameters
    alpha_1, amp = sample_filter_level_params(n_filt, alpha_0, curved_power_law)

    # Likelihood
    with numpyro.plate("data", len(t)):
        # Index the (n_obj, n_filt) arrays using observation indices
        alpha_0_obs = alpha_0[idx_obj, idx_filt]
        amp_obs = amp[idx_obj, idx_filt]

        numpyro.sample(
            "flux",
            dist.Normal(
                f_t(
                    t,
                    t_fl[idx_obj],
                    base[idx_fcqfid],
                    amp_obs,
                    alpha_0_obs,
                    alpha_1[idx_filt],
                ),
                flux_err * beta[idx_fcqfid],
            ),
            obs=flux,
        )


def unpooled_model(
    t: list,
    flux: list = None,
    flux_err: list = None,
    idx_obj: list = None,
    idx_fcqfid: list = None,
    idx_filt: list = None,
    prior_config: dict = {},
):
    """
    Unpooled model:  each object independent, no hierarchical structure.

    Model: alpha_0_i ~ [prior], t_fl_j ~ Uniform(-40, 0)
    """
    # Setup
    prior_type = prior_config.get("prior_type", "Miller")
    min_alpha_0 = prior_config.get("min_alpha_0", 1)
    max_alpha_0 = prior_config.get("max_alpha_0", 5)
    assert min_alpha_0 >= 0, "min_alpha_0 must be non-negative"

    n_fcqfid = len(np.unique(idx_fcqfid))
    n_filt = len(np.unique(idx_filt))
    n_obj = len(np.unique(idx_obj))

    # Observation-level parameters
    base, beta = sample_observation_params(n_fcqfid, prior_type)

    # alpha_0 is a filter-level parameter in the unpooled model
    with numpyro.plate("filt", n_filt):
        mean_val = prior_config.get("mean_alpha_0", 2)
        sigma_val = prior_config.get("sigma_alpha_0", None)
        alpha_0 = sample_alpha_prior(
            prior_type, min_alpha_0, max_alpha_0, mean_val, sigma_val
        )

    # Filter-level parameters
    alpha_1, amp = sample_filter_level_params(
        n_filt,
        alpha_0,
        curved_power_law=prior_config.get("curved_power_law", False),
    )

    # Object-level parameters
    t_fl = sample_tfl_params(n_obj, prior_config, hierarchical=False)

    # Likelihood
    with numpyro.plate("data", len(t)):
        numpyro.sample(
            "flux",
            dist.Normal(
                f_t(
                    t,
                    t_fl[idx_obj],
                    base[idx_fcqfid],
                    amp[idx_filt],
                    alpha_0[idx_filt],
                    alpha_1[idx_filt],
                ),
                flux_err * beta[idx_fcqfid],
            ),
            obs=flux,
        )


def pooled_model(
    t: list,
    flux: list = None,
    flux_err: list = None,
    idx_obj: list = None,
    idx_fcqfid: list = None,
    idx_filt: list = None,
    idx_filt_gr: list = None,
    prior_config: dict = {},
):
    """
    Pooled model: all objects share alpha and t_fl.

    Model: alpha_0_k ~ [prior] per filter group, t_fl ~ Uniform(-40, 0) (global)
    """
    # Setup
    prior_config.setdefault("prior_type", "Maximum_Entropy")
    min_alpha_0 = prior_config.get("min_alpha_0", 1)
    max_alpha_0 = prior_config.get("max_alpha_0", 5)
    curved_power_law = prior_config.get("curved_power_law", False)
    assert min_alpha_0 >= 0, "min_alpha_0 must be non-negative"

    n_fcqfid = len(np.unique(idx_fcqfid))
    n_filt = len(np.unique(idx_filt))
    n_filt_gr = len(np.unique(idx_filt_gr))

    # Observation-level parameters
    base, beta = sample_observation_params(n_fcqfid)

    # Filter-group level alpha
    mean_val = prior_config.get("mean_alpha_0", 2)
    sigma_val = prior_config.get("sigma_alpha_0", None)
    prior_type = prior_config["prior_type"]

    with numpyro.plate("filt_gr", n_filt_gr):
        alpha_0 = sample_alpha_prior(
            prior_type, min_alpha_0, max_alpha_0, mean_val, sigma_val
        )

        if curved_power_law:
            mean_neg = 1 / (20 * (1 + np.log(20)))
            neg_alpha_1 = numpyro.sample("-alpha_1", dist.Exponential(1 / mean_neg))
            alpha_1 = numpyro.deterministic("alpha_1", -neg_alpha_1)
        else:
            alpha_1 = jnp.zeros(n_filt_gr)

    # Per-filter amplitude
    with numpyro.plate("filt", n_filt):
        amp_prime = numpyro.sample("Aprime", dist.LogUniform(1e-5, 1e5))

    amp = numpyro.deterministic(
        "A", amp_prime[idx_filt] / jnp.power(10, alpha_0[idx_filt_gr])
    )

    # Global t_fl
    t_fl = numpyro.sample("t_fl", dist.Uniform(-40, 0))

    # Likelihood
    with numpyro.plate("data", len(t)):
        numpyro.sample(
            "flux",
            dist.Normal(
                f_t(
                    t,
                    t_fl,
                    base[idx_fcqfid],
                    amp[idx_filt],
                    alpha_0[idx_filt_gr],
                    alpha_1[idx_filt_gr],
                ),
                flux_err * beta[idx_fcqfid],
            ),
            obs=flux,
        )

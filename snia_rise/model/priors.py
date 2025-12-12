"""Helper functions for hierarchical Bayesian models of supernova light curves."""

import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def sample_observation_params(n_fcqfid, prior_type="default"):
    """
    Sample per-observation parameters:  baseline and uncertainty scale.

    Parameters
    ----------
    n_fcqfid : int
        Number of unique fcqf IDs
    prior_type : str
        "Miller" uses different beta prior

    Returns
    -------
    base, beta
    """
    with numpyro.plate("fcqfid", n_fcqfid):
        base = numpyro.sample("C", dist.Uniform(-50, 50))

        if prior_type == "Miller":
            beta = numpyro.sample("beta", dist.LogUniform(0.7, 1.3))
        else:
            beta = numpyro.sample("beta", dist.LogNormal(0, 0.1))

    return base, beta


def sample_alpha_prior(prior_type, min_val, max_val, mean_val=2.0, std_val=None):
    """
    Sample a single alpha value with specified prior (used inside plates).

    Must be called within a numpyro.plate context.

    Parameters
    ----------
    prior_type : str
        "Miller", "Jeffreys", "Flat"/"Uniform", "Maximum_Entropy", "Gaussian"
    min_val : float
        Minimum value
    max_val : float
        Maximum value
    mean_val :  float, default=2.0
        Mean for Maximum_Entropy or Gaussian
    std_val : float, optional
        Std for Maximum_Entropy or Gaussian

    Returns
    -------
    alpha_0 : array
        Sampled alpha values
    """
    if prior_type == "Miller":
        alpha_0 = numpyro.sample("alpha_0", dist.Exponential(jnp.log(10)))

    elif prior_type == "Jeffreys":
        alpha_0 = numpyro.sample(
            "alpha_0", dist.LogUniform(max(min_val, 1e-2), max_val)
        )

    elif prior_type in ["Flat", "Uniform"]:
        alpha_0 = numpyro.sample("alpha_0", dist.Uniform(min_val, max_val))

    elif prior_type == "Maximum_Entropy":
        if std_val is None:
            rate = 1 / (mean_val - min_val)
            alpha_ = numpyro.sample("alpha-", dist.Exponential(rate))
        else:
            concentration = (mean_val - min_val) ** 2 / std_val**2
            rate = (mean_val - min_val) / std_val**2
            alpha_ = numpyro.sample("alpha-", dist.Gamma(concentration, rate))
        alpha_0 = numpyro.deterministic("alpha_0", alpha_ + min_val)

    elif prior_type in ["Gaussian", "Gauss", "Normal"]:
        alpha_0 = numpyro.sample(
            "alpha_0",
            dist.TruncatedNormal(mean_val, std_val, low=min_val, high=max_val),
        )

    else:
        raise ValueError(
            f"Invalid prior_type '{prior_type}'.Options:  'Miller', 'Jeffreys', "
            "'Flat', 'Uniform', 'Maximum_Entropy', 'Gaussian', 'Gauss', 'Normal'"
        )

    return alpha_0


def sample_filter_level_params(n_filt, alpha_0, curved_power_law):
    """
    Sample filter-level parameters:  alpha_1 (curvature) and amplitude.

    Parameters
    ----------
    n_filt : int
        Number of filters
    alpha_0 : array, shape (n_obj, n_filt) or (n_filt,)
        Rising power-law index
    curved_power_law : bool
        Whether to include curvature term

    Returns
    -------
    alpha_1 : array, shape (n_filt,)
        Curvature parameter
    amp_prime : array, shape (n_filt,)
        Amplitude normalization (before alpha_0 correction)
    amp : array, shape (n_obj, n_filt) or (n_filt,)
        Final amplitude (amp_prime / 10^alpha_0)
    """
    with numpyro.plate("filt", n_filt):
        if curved_power_law:
            mean_neg = 1 / (18 * (1 + np.log(18)))
            neg_alpha_1 = numpyro.sample("-alpha_1", dist.Exponential(1 / mean_neg))
            alpha_1 = numpyro.deterministic("alpha_1", -neg_alpha_1)
        else:
            alpha_1 = jnp.zeros(n_filt)

        # amp_prime:  shape (n_filt,) - shared across objects
        amp_prime = numpyro.sample("Aprime", dist.LogUniform(1e-5, 1e5))

    # Compute amplitude based on alpha_0 shape
    if alpha_0.ndim == 2:
        # Hierarchical models:  alpha_0 has shape (n_obj, n_filt)
        # amp[j, i] = amp_prime[i] / 10^alpha_0[j, i]
        amp = numpyro.deterministic("A", amp_prime[None, :] / jnp.power(10, alpha_0))
    else:
        # Unpooled/pooled models: alpha_0 has shape (n_filt,) or scalar
        # amp[i] = amp_prime[i] / 10^alpha_0[i]
        amp = numpyro.deterministic("A", amp_prime / jnp.power(10, alpha_0))

    return alpha_1, amp


def sample_tfl_params(n_obj, prior_config, hierarchical=False):
    """
    Sample t_fl parameters (per-object or hierarchical).

    Parameters
    ----------
    n_obj :  int
        Number of objects
    prior_config : dict
        Configuration dict
    hierarchical : bool
        If True, sample hyperpriors first

    Returns
    -------
    t_fl :  array, shape (n_obj,)
        Or (mean_t_fl, std_t_fl) if hierarchical and returning hyperpriors
    """
    if hierarchical:
        # Sample hyperpriors
        mean_t_fl = numpyro.sample("mean_t_fl", dist.Uniform(-30, -10))
        std_t_fl = numpyro.sample("std_t_fl", dist.LogUniform(1e-2, 5.0))

        # Sample per-object t_fl
        with numpyro.plate("obj", n_obj):
            t_fl = numpyro.sample("t_fl", dist.Normal(mean_t_fl, std_t_fl))

        return t_fl
    else:
        # Non-hierarchical (unpooled)
        prior_type = prior_config.get("prior_type", "Uniform")

        with numpyro.plate("obj", n_obj):
            if prior_type in ["Gaussian", "Gauss", "Normal"]:
                mean_t_fl = prior_config.get("mean_t_fl", -18)
                std_t_fl = prior_config.get("std_t_fl", 1.5)
                t_fl = numpyro.sample("t_fl", dist.Normal(mean_t_fl, std_t_fl))
            else:
                t_fl = numpyro.sample("t_fl", dist.Uniform(-40, 0))

        return t_fl


def _sample_mvn_hierarchical_params(
    n_obj,
    n_filt_gr,
    n_filt,
    idx_filt_loc,
    mean_t_fl,
    sigma_t_fl,
    mean_alpha_0,
    sigma_alpha_0,
    min_alpha_0,
    max_alpha_0,
    sample_correlations=True,
):
    """
    Sample MVN hyperpriors and per-object parameters.

    Parameters
    ----------
    n_obj : int
        Number of objects
    n_filt_gr : int
        Number of filter groups
    n_filt : int
        Number of filters
    idx_filt_loc : array
        Mapping from filter to filter group
    mean_t_fl : float
        Population mean for t_fl
    sigma_t_fl : float
        Population std for t_fl
    mean_alpha_0 : array, shape (n_filt_gr,)
        Population mean for alpha_0 per filter group
    sigma_alpha_0 :  array, shape (n_filt_gr,)
        Population std for alpha_0 per filter group
    min_alpha_0 :  float
        Minimum alpha value
    max_alpha_0 :  float
        Maximum alpha value
    sample_correlations : bool
        If True, sample correlation matrix. If False, use identity (independent)

    Returns
    -------
    t_fl :  array, shape (n_obj,)
    alpha_0 : array, shape (n_obj, n_filt)
    hyperparams : dict
    """
    d = 1 + n_filt_gr
    hyperparams = {}

    # Mean and scale vectors
    mu = jnp.concatenate([jnp.array([mean_t_fl]), mean_alpha_0])
    sigma = jnp.concatenate([jnp.array([sigma_t_fl]), sigma_alpha_0])

    if sample_correlations:
        # Sample correlation matrix
        chol_corr = numpyro.sample("chol_corr", dist.LKJCholesky(d, concentration=2.0))
    else:
        # Use identity (independent)
        chol_corr = jnp.eye(d)

    L_Cholesky = jnp.matmul(jnp.diag(sigma), chol_corr)

    # Store covariance and correlation
    hyperparams["Sigma"] = numpyro.deterministic(
        "Sigma", jnp.matmul(L_Cholesky, L_Cholesky.T)
    )
    hyperparams["Corr"] = numpyro.deterministic(
        "Corr", jnp.matmul(chol_corr, chol_corr.T)
    )

    # Sample from MVN
    with numpyro.plate("obj", n_obj):
        theta = numpyro.sample(
            "theta", dist.MultivariateNormal(loc=mu, scale_tril=L_Cholesky)
        )
        t_fl = numpyro.deterministic("t_fl", theta[..., 0])
        alpha_0_groups = jnp.clip(theta[..., 1:], min_alpha_0, max_alpha_0)

    # Expand to all filters
    alpha_0_expanded = alpha_0_groups[:, idx_filt_loc]
    alpha_0 = numpyro.deterministic("alpha_0", alpha_0_expanded)

    return t_fl, alpha_0, hyperparams


def _sample_tfl_only_hierarchical_params(
    n_obj,
    n_filt,
    idx_filt_loc,
    mean_t_fl,
    sigma_t_fl,
    min_alpha_0,
    max_alpha_0,
    prior_type,
    prior_config,
):
    """Only t_fl hierarchical, alpha_0 sampled per-filter (like unpooled)."""
    hyperparams = {}
    # No Sigma or Corr - this is not a multivariate model

    # Sample t_fl hierarchically
    with numpyro.plate("obj", n_obj):
        t_fl = numpyro.sample("t_fl", dist.Normal(mean_t_fl, sigma_t_fl))

    # Sample alpha_0 per filter with non-hierarchical priors (like unpooled)
    mean_val = prior_config.get("mean_alpha_0", 2)
    std_val = prior_config.get("std_alpha_0", None)

    with numpyro.plate("filt", n_filt):
        alpha_0_per_filt = sample_alpha_prior(
            prior_type, min_alpha_0, max_alpha_0, mean_val, std_val
        )

    # Broadcast to (n_obj, n_filt) for consistent structure
    alpha_0 = jnp.tile(alpha_0_per_filt[None, :], (n_obj, 1))

    # No theta reconstruction - this model doesn't have a multivariate structure

    return t_fl, alpha_0, hyperparams


def sample_hierarchical_params(
    n_obj,
    n_filt_gr,
    n_filt,
    idx_filt_loc,
    min_alpha_0,
    max_alpha_0,
    correlation_structure="mvn",
    prior_type="Maximum_Entropy",
    prior_config={},
):
    """
    Sample hierarchical parameters with different correlation structures.

    This is a dispatcher that calls the appropriate sampling strategy.
    """
    d = 1 + n_filt_gr

    # Sample t_fl hyperpriors (common to all structures)
    mean_t_fl = numpyro.sample("mean_t_fl", dist.Uniform(-30, -10))
    sigma_t_fl = numpyro.sample("sigma_t_fl", dist.HalfCauchy(1.5))

    # Sample alpha_0 hyperpriors only for mvn and independent
    if correlation_structure in ["mvn", "independent"]:
        with numpyro.plate("n_filt_gr", n_filt_gr):
            mean_alpha_0 = numpyro.sample(
                "mean_alpha_0", dist.Uniform(min_alpha_0, max_alpha_0)
            )
            sigma_alpha_0 = numpyro.sample("sigma_alpha_0", dist.HalfCauchy(0.3))

        if correlation_structure == "mvn":
            # Full MVN with correlations
            return _sample_mvn_hierarchical_params(
                n_obj,
                n_filt_gr,
                n_filt,
                idx_filt_loc,
                mean_t_fl,
                sigma_t_fl,
                mean_alpha_0,
                sigma_alpha_0,
                min_alpha_0,
                max_alpha_0,
                sample_correlations=True,
            )
        else:  # independent
            # Independent hierarchical priors
            return _sample_mvn_hierarchical_params(
                n_obj,
                n_filt_gr,
                n_filt,
                idx_filt_loc,
                mean_t_fl,
                sigma_t_fl,
                mean_alpha_0,
                sigma_alpha_0,
                min_alpha_0,
                max_alpha_0,
                sample_correlations=False,  # Diagonal covariance
            )

    elif correlation_structure == "tfl_only":
        # Only t_fl hierarchical, alpha_0 non-hierarchical
        return _sample_tfl_only_hierarchical_params(
            n_obj,
            n_filt,
            idx_filt_loc,
            mean_t_fl,
            sigma_t_fl,
            min_alpha_0,
            max_alpha_0,
            prior_type,
            prior_config,
        )

    else:
        raise ValueError(
            f"Invalid correlation_structure '{correlation_structure}'. "
            "Options: 'mvn', 'independent', 'tfl_only'"
        )

"""Helper functions for hierarchical Bayesian models of supernova light curves."""

from random import sample
import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def sample_fcqf_params(n_fcqfid, prior_config: dict = {}):
    """
    Sample Field/CCD/Quadrant/Filter-level parameters:  baseline and uncertainty scale.

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
    prior_type = prior_config.get("prior_type", "uniform").lower()

    with numpyro.plate("fcqfid", n_fcqfid):
        base = numpyro.sample("C", dist.Uniform(-50, 50))

        if prior_type == "miller":
            beta = numpyro.sample("beta", dist.LogUniform(0.7, 1.3))
        else:
            beta = numpyro.sample("beta", dist.LogNormal(0, 0.1))

    return base, beta


def sample_alpha_0(
    mean_alpha_0: float = None,
    sigma_alpha_0: float = None,
    min_alpha_0: float = None,
    max_alpha_0: float = None,
    prior_config: dict = {},
):
    """
    Sample a single alpha value with specified prior (used inside plates).

    Must be called within a numpyro.plate context.

    Parameters
    ----------
    prior_config : dict
        - prior_type : str
            "miller", "uniform", "maximum_entropy", "normal"
        - mean_alpha_0 : float, default=2
            Mean alpha value (for maximum_entropy and normal)
        - sigma_alpha_0 : float, optional
            Std alpha value (for maximum_entropy and normal)
        - min_alpha_0 : float, default=1
            Minimum alpha value
        - max_alpha_0 : float, default=5
            Maximum alpha value

    Returns
    -------
    alpha_0 : array, shape ()
        Sampled alpha values
    """

    prior_type = prior_config.get("prior_type", "maximum_entropy").lower()

    if mean_alpha_0 is None:
        mean_alpha_0 = prior_config.get("mean_alpha_0", 2)
    if sigma_alpha_0 is None:
        sigma_alpha_0 = prior_config.get("sigma_alpha_0", None)
    if min_alpha_0 is None:
        min_alpha_0 = prior_config.get("min_alpha_0", 1)
    if max_alpha_0 is None:
        max_alpha_0 = prior_config.get("max_alpha_0", 5)
    assert min_alpha_0 >= 0, "min_alpha_0 must be non-negative"

    if prior_type == "miller":
        # Miller et al. (2020) prior
        alpha_0 = numpyro.sample("alpha_0", dist.Exponential(jnp.log(10)))

    elif prior_type == "uniform":
        # Uniform prior
        alpha_0 = numpyro.sample("alpha_0", dist.Uniform(min_alpha_0, max_alpha_0))

    elif prior_type == "maximum_entropy":
        # Maximum Entropy prior
        if sigma_alpha_0 is None:
            # alpha > min_alpha_0, E(alpha) = mean_alpha_0 -> Exponential
            rate = 1 / (mean_alpha_0 - min_alpha_0)
            alpha_ = numpyro.sample("alpha-", dist.Exponential(rate))
        else:
            # alpha > min_alpha_0, E(alpha) = mean_alpha_0, Var(alpha) = sigma_alpha_0^2 -> Gamma
            concentration = (mean_alpha_0 - min_alpha_0) ** 2 / sigma_alpha_0**2
            rate = (mean_alpha_0 - min_alpha_0) / sigma_alpha_0**2
            alpha_ = numpyro.sample("alpha-", dist.Gamma(concentration, rate))
        alpha_0 = numpyro.deterministic("alpha_0", alpha_ + min_alpha_0)

    elif prior_type == "normal":
        alpha_0 = numpyro.sample(
            "alpha_0",
            dist.TruncatedNormal(
                mean_alpha_0, sigma_alpha_0, low=min_alpha_0, high=max_alpha_0
            ),
        )

    else:
        raise ValueError(
            f"Invalid prior_type '{prior_type}'.Options: 'miller', 'maximum_entropy', 'normal'"
        )

    return alpha_0


def sample_alpha_1():
    """
    Sample a single alpha value with specified prior (used inside plates).

    Must be called within a numpyro.plate context.

    Parameters
    ----------
    curved_power_law : bool
        Whether to include curvature term

    Returns
    -------
    alpha_1 : array, shape ()
        Curvature parameter
    """
    mean_neg = 1 / (18 * (1 + np.log(18)))
    neg_alpha_1 = numpyro.sample("-alpha_1", dist.Exponential(1 / mean_neg))
    alpha_1 = numpyro.deterministic("alpha_1", -neg_alpha_1)

    return alpha_1


def sample_amp_prime():
    """
    Sample Object/Filter-level parameters:  alpha_1 (curvature) and amplitude.

    Returns
    -------
    amp_prime : array
        Amplitude normalization (before alpha_0 correction)
    """
    amp_prime = numpyro.sample("Aprime", dist.LogUniform(1e-5, 1e5))

    return amp_prime


def sample_t_rise(prior_config):
    """
    Sample t_rise parameters.

    Parameters
    ----------
    prior_config : dict
        Configuration dict

    Returns
    -------
    t_rise :  array, shape ()
    """
    prior_type = prior_config.get("prior_type", "uniform").lower()

    if prior_type in ["gaussian", "normal"]:
        mean_t_rise = prior_config.get("mean_t_rise", 18)
        sigma_t_rise = prior_config.get("sigma_t_rise", 1.5)
        t_rise = numpyro.sample("t_rise", dist.Normal(mean_t_rise, sigma_t_rise))
    else:
        t_rise = numpyro.sample("t_rise", dist.Uniform(0, 40))

    return t_rise


def sample_t_fl(n_obj: int, t_rise: jnp.ndarray, t0_err: jnp.ndarray):
    """
    Sample t_fl parameters.

    Parameters
    ----------
    n_obj : int
        Number of objects
    t_rise : array, shape ()
        t_rise values
    t0_err : array, shape ()
        Uncertainties on t0

    Returns
    -------
    t_fl :  array, shape ()
    """
    with numpyro.plate("obj", n_obj):
        # if t0_err is None:
        #     t0_offset = 0.0
        # else:
        #     t0_offset = numpyro.sample("t0_offset", dist.Normal(0, t0_err))
        # t_fl = numpyro.deterministic("t_fl", -t_rise + t0_offset)
        t_fl = numpyro.deterministic("t_fl", -t_rise)

    return t_fl


def _sample_mvn_hierarchical_params(
    n_obj,
    n_filt,
    mean_t_rise,
    sigma_t_rise,
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
    n_filt : int
        Number of filters
    mean_t_rise : float
        Population mean for t_rise
    sigma_t_rise : float
        Population std for t_rise
    mean_alpha_0 : float
        Population mean for the common-mode alpha_0
    sigma_alpha_0 : float
        Population standard deviation for the common-mode alpha_0
    min_alpha_0 : float
        Minimum alpha value
    max_alpha_0 : float
        Maximum alpha value
    sample_correlations : bool
        If True, sample correlation matrix. If False, use identity (independent)

    Returns
    -------
    t_rise : array, shape (n_obj,)
    alpha_0 : array, shape (n_obj, n_filt)
    """
    n_mvn_dim = 2

    # Mean and scale vectors for MVN
    mu = jnp.array([mean_t_rise, mean_alpha_0])
    sigma = jnp.array([sigma_t_rise, sigma_alpha_0])

    # Sample or fix correlation structure
    if sample_correlations:
        chol_corr = numpyro.sample(
            "chol_corr", dist.LKJCholesky(n_mvn_dim, concentration=2.0)
        )
    else:
        chol_corr = jnp.eye(n_mvn_dim)

    L_Cholesky = jnp.matmul(jnp.diag(sigma), chol_corr)

    if sample_correlations:
        # Store covariance and correlation for diagnostics
        with numpyro.plate("mvn_dim_0", n_mvn_dim, dim=-2):
            with numpyro.plate("mvn_dim_1", n_mvn_dim, dim=-1):
                _ = numpyro.deterministic("Sigma", jnp.matmul(L_Cholesky, L_Cholesky.T))
                _ = numpyro.deterministic("Corr", jnp.matmul(chol_corr, chol_corr.T))

    # Sample filter-specific noise scales for ALL filters
    with numpyro.plate("filt_delta", n_filt - 1):
        sigma_delta = numpyro.sample("sigma_delta", dist.HalfCauchy(0.3))
        with numpyro.plate("obj", n_obj):
            delta_raw = numpyro.sample(
                "delta_raw", dist.Normal(jnp.zeros(n_filt - 1), jnp.ones(n_filt - 1))
            )
            delta = delta_raw * sigma_delta[None, :]
            delta_last = -jnp.sum(delta, axis=-1, keepdims=True)
            delta_full = numpyro.deterministic(
                "delta", jnp.concatenate([delta, delta_last], axis=-1)
            )

    with numpyro.plate("obj", n_obj):
        # Sample (t_rise, alpha_0_cm) from MVN
        theta = numpyro.sample(
            "theta", dist.MultivariateNormal(loc=mu, scale_tril=L_Cholesky)
        )
        t_rise = numpyro.deterministic("t_rise", theta[..., 0])
        alpha_0_cm = numpyro.deterministic("alpha_0_cm", theta[..., 1])

        # Final alpha per filter:  common-mode + color deviation
        alpha_0 = numpyro.deterministic(
            "alpha_0",
            jnp.clip(alpha_0_cm[:, None] + delta_full, min_alpha_0, max_alpha_0),
        )

    return t_rise, alpha_0


def _sample_trise_only_hierarchical_params(
    n_obj,
    n_filt,
    mean_t_rise,
    sigma_t_rise,
    mean_alpha_0,
    sigma_alpha_0,
    min_alpha_0,
    max_alpha_0,
):
    """Only t_rise hierarchical, alpha_0 sampled independently (like unpooled).

    Parameters
    ----------
    n_obj : int
        Number of objects
    n_filt : int
        Number of filters
    mean_t_rise : float
        Population mean for t_rise
    sigma_t_rise : float
        Population std for t_rise
    min_alpha_0 : float
        Minimum alpha value
    max_alpha_0 : float
        Maximum alpha value

    Returns
    -------
    t_rise : array, shape (n_obj,)
    alpha_0 : array, shape (n_obj, n_filt)
    """

    # Sample t_rise hierarchically
    with numpyro.plate("obj", n_obj):
        t_rise = numpyro.sample("t_rise", dist.Normal(mean_t_rise, sigma_t_rise))

    with numpyro.plate("filt", n_filt):
        with numpyro.plate("obj", n_obj):
            alpha_0 = sample_alpha_0(
                mean_alpha_0, sigma_alpha_0, min_alpha_0, max_alpha_0
            )

    return t_rise, alpha_0


def sample_hierarchical_params(
    n_obj,
    n_filt,
    correlation_structure="mvn",
    prior_config={},
):
    """
    Sample hierarchical parameters with different correlation structures.

    This is a dispatcher that calls the appropriate sampling strategy.

    Parameters
    ----------
    n_obj : int
        Number of objects
    n_filt : int
        Number of filters
    correlation_structure : str
        "mvn", "independent", or "trise_only"
    prior_config : dict
        Configuration dict

    Returns
    -------
    t_rise : array, shape (n_obj,)
    alpha_0 : array, shape (n_obj, n_filt)
    """
    min_alpha_0 = prior_config.get("min_alpha_0", 1)
    max_alpha_0 = prior_config.get("max_alpha_0", 5)
    assert min_alpha_0 >= 0, "min_alpha_0 must be non-negative"

    # Sample t_rise hyperpriors (common to all structures)
    mean_t_rise = numpyro.sample("mean_t_rise", dist.Uniform(0, 40))
    sigma_t_rise = numpyro.sample("sigma_t_rise", dist.HalfCauchy(1.5))

    # Sample alpha_0 hyperpriors only for mvn and independent
    if correlation_structure in ["mvn", "independent"]:
        mean_alpha_0 = numpyro.sample(
            "mean_alpha_0_cm", dist.Uniform(min_alpha_0, max_alpha_0)
        )
        sigma_alpha_0 = numpyro.sample("sigma_alpha_0_cm", dist.HalfCauchy(0.3))

        if correlation_structure == "mvn":
            # Full MVN with correlations
            return _sample_mvn_hierarchical_params(
                n_obj,
                n_filt,
                mean_t_rise,
                sigma_t_rise,
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
                n_filt,
                mean_t_rise,
                sigma_t_rise,
                mean_alpha_0,
                sigma_alpha_0,
                min_alpha_0,
                max_alpha_0,
                sample_correlations=False,
            )

    elif correlation_structure == "trise_only":
        # Only t_rise hierarchical, alpha_0 non-hierarchical
        mean_alpha_0 = prior_config.get("mean_alpha_0", 2)
        sigma_alpha_0 = prior_config.get("sigma_alpha_0", None)
        return _sample_trise_only_hierarchical_params(
            n_obj,
            n_filt,
            mean_t_rise,
            sigma_t_rise,
            mean_alpha_0,
            sigma_alpha_0,
            min_alpha_0,
            max_alpha_0,
        )

    else:
        raise ValueError(
            f"Invalid correlation_structure '{correlation_structure}'. "
            "Options: 'mvn', 'independent', 'trise_only'"
        )

"""Helper functions for hierarchical Bayesian models of supernova light curves."""

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from ..constants import EPS

# def sample_fcqf_params(n_fcqfid, prior_config: dict = {}):
#     """
#     Sample Field/CCD/Quadrant/Filter-level parameters:  baseline and uncertainty scale.

#     Parameters
#     ----------
#     n_fcqfid : int
#         Number of unique fcqf IDs
#     prior_type : str
#         "Miller" uses different beta prior

#     Returns
#     -------
#     base, beta
#     """
#     prior_type = prior_config.get("prior_type", "uniform").lower()

#     with numpyro.plate("fcqfid", n_fcqfid):
#         base = numpyro.sample("C", dist.Uniform(-50, 50))

#         if prior_type == "miller":
#             beta = numpyro.sample("beta", dist.LogUniform(0.7, 1.3))
#         else:
#             # beta = numpyro.sample("beta", dist.LogNormal(0, 0.1))
#             beta = numpyro.deterministic("beta", jnp.ones(n_fcqfid))

#     return base, beta


def sample_base(n_fcqfid):
    with numpyro.plate("fcqfid", n_fcqfid):
        base = numpyro.sample("C", dist.Uniform(-50, 50))
    return base


def sample_beta(n_fcqfid, prior_config: dict = {}):
    """
    Sample uncertainty scale factor (beta) for each fcqfid.

    Model: log(beta) ~ HalfNormal(0, scale)
    This ensures beta >= 1 (support on [1, inf))

    Parameters
    ----------
    n_fcqfid : int
        Number of unique fcqf IDs
    prior_config : dict
        Configuration dictionary with keys:
        - sample_beta : bool, default=False
            Whether to sample beta as a free parameter (True) or fix it to 1 (False)

    Returns
    -------
    beta : array, shape (n_fcqfid,)
        Uncertainty scale factors (beta >= 1)
    """
    sample_beta_flag = prior_config.get("sample_beta", False)

    with numpyro.plate("fcqfid", n_fcqfid):
        if sample_beta_flag:
            # log(beta) ~ HalfNormal(0, scale) ensures beta >= 1
            log_beta = numpyro.sample("log_beta", dist.HalfNormal(0.1))
            beta = numpyro.deterministic("beta", jnp.exp(log_beta))
        else:
            # Fixed at 1 (no uncertainty scaling)
            beta = numpyro.deterministic("beta", jnp.ones(n_fcqfid))

    return beta


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
            dist.TruncatedNormal(mean_alpha_0, sigma_alpha_0, low=min_alpha_0),
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

    Returns
    -------
    alpha_1 : array, shape ()
        Curvature parameter
    """
    mean_neg = 1 / (20 * (1 + jnp.log(20)))
    neg_alpha_1 = numpyro.sample("-alpha_1", dist.Exponential(1 / mean_neg))
    alpha_1 = numpyro.deterministic("alpha_1", -neg_alpha_1)

    return alpha_1


def sample_amp_prime():
    """
    Sample Object/Filter-level parameters:  alpha_1 (curvature) and amplitude.

    Returns
    -------
    amp_prime : array
        Amplitude normalization (before alpha_0 correction: Miller et al. 2020)
    """
    # amp_prime = numpyro.sample("Aprime", dist.LogUniform(1e0, 1e3))
    log_amp_prime = numpyro.sample("log_Aprime", dist.Uniform(0, jnp.log(1e3)))
    amp_prime = numpyro.deterministic("Aprime", jnp.exp(log_amp_prime))

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
        t_rise = numpyro.sample(
            "t_rise",
            dist.TruncatedNormal(mean_t_rise, sigma_t_rise, EPS, None),
        )
    else:
        t_rise_min = prior_config.get("t_rise_min", EPS)
        t_rise_max = prior_config.get("t_rise_max", 40)
        t_rise = numpyro.sample("t_rise", dist.Uniform(t_rise_min, t_rise_max))

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
    mean_log_amp_prime,
    sigma_log_amp_prime,
    sample_correlations=True,
):
    """
    Sample MVN hyperpriors and per-object parameters.

    Returns
    -------
    t_rise : array, shape (n_obj,)
    amp_prime : array, shape (n_obj, n_filt)
    alpha_0 : array, shape (n_obj, n_filt)
    """
    # Dimension is t_rise (1), ln A (n_filt), alpha_0 for each filter (n_filt)
    n_mvn_dim = 1 + 2 * n_filt

    # Mean and scale vectors for MVN
    # t_rise is index 0
    # alpha_0 for filters are indices n_filt+1..2*n_filt
    # log A' for filters are indices 1..n_filt
    mu = jnp.concatenate([jnp.array([mean_t_rise]), mean_alpha_0, mean_log_amp_prime])
    sigma = jnp.concatenate(
        [jnp.array([sigma_t_rise]), sigma_alpha_0, sigma_log_amp_prime]
    )

    # Sample or fix correlation structure
    if sample_correlations:
        chol_corr = numpyro.sample(
            "chol_corr", dist.LKJCholesky(n_mvn_dim, concentration=1.0)
        )
        L_Cholesky = jnp.matmul(jnp.diag(sigma), chol_corr)
    else:
        L_Cholesky = None

    if sample_correlations:
        # Store covariance and correlation for diagnostics
        with numpyro.plate("mvn_dim_0", n_mvn_dim, dim=-2):
            with numpyro.plate("mvn_dim_1", n_mvn_dim, dim=-1):
                _ = numpyro.deterministic("Sigma", jnp.matmul(L_Cholesky, L_Cholesky.T))
                _ = numpyro.deterministic("Corr", jnp.matmul(chol_corr, chol_corr.T))

    with numpyro.plate("obj", n_obj):
        # Sample (t_rise, alpha_0_1, alpha_0_2, ...) from MVN
        # Non-centered sampling
        theta_raw = numpyro.sample(
            "theta_raw", dist.Normal(0, 1).expand([n_mvn_dim]).to_event(1)
        )
        if L_Cholesky is None:
            theta = mu + sigma * theta_raw
        else:
            # (n_obj, dim) @ (dim, dim).T -> (n_obj, dim)
            theta = mu + (theta_raw @ L_Cholesky.T)

        # Extract rise times (t_rise)
        t_rise = numpyro.deterministic("t_rise", jnp.clip(theta[..., 0], EPS, None))

    # Slicing
    alpha_0_raw = theta[..., 1 : 1 + n_filt]
    alpha_0_clipped = jnp.clip(alpha_0_raw, 1 + EPS, None)

    with numpyro.plate("obj", n_obj, dim=-2):
        with numpyro.plate("filt", n_filt, dim=-1):
            # Extract power-law indices (alpha_0) and clip
            alpha_0 = numpyro.deterministic("alpha_0", alpha_0_clipped)

            # Extract amplitudes (A)
            log_amp_prime = numpyro.deterministic(
                "log_Aprime", theta[..., 1 + n_filt :]
            )
            amp_prime = numpyro.deterministic("Aprime", jnp.exp(log_amp_prime))

    return t_rise, amp_prime, alpha_0


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
        "mvn" or "independent"
    prior_config : dict
        Configuration dict

    Returns
    -------
    t_rise : array, shape (n_obj,)
    amp_prime : array, shape (n_obj, n_filt)
    alpha_0 : array, shape (n_obj, n_filt)
    """

    # Same as uniform priors in the unpooled model
    min_mean_t_rise = 5.0
    max_mean_t_rise = 35.0
    min_mean_alpha_0 = 1.0 + EPS
    max_mean_alpha_0 = 4.0

    # Sample t_rise hyperpriors (common to all structures)
    mean_t_rise = numpyro.sample(
        "mean_t_rise", dist.Uniform(min_mean_t_rise, max_mean_t_rise)
    )
    sigma_t_rise = numpyro.sample("sigma_t_rise", dist.HalfCauchy(1.5))

    # Sample alpha_0 hyperpriors only for mvn and independent
    if correlation_structure in ["mvn", "independent"]:
        # mean_alpha_0 = numpyro.sample(
        # "mean_alpha_0_cm", dist.Uniform(min_alpha_0, max_alpha_0)
        # )
        # sigma_alpha_0 = numpyro.sample("sigma_alpha_0_cm", dist.HalfCauchy(0.3))
        with numpyro.plate("filt", n_filt):
            mean_log_amp_prime = numpyro.sample(
                "mean_log_Aprime", dist.Uniform(0, jnp.log(1e3))
            )
            sigma_log_amp_prime = numpyro.sample(
                "sigma_log_Aprime", dist.HalfCauchy(0.5)
            )
            mean_alpha_0 = numpyro.sample(
                "mean_alpha_0", dist.Uniform(min_mean_alpha_0, max_mean_alpha_0)
            )
            sigma_alpha_0 = numpyro.sample("sigma_alpha_0", dist.HalfCauchy(0.3))

        if correlation_structure == "mvn":
            # Full MVN with correlations
            return _sample_mvn_hierarchical_params(
                n_obj,
                n_filt,
                mean_t_rise,
                sigma_t_rise,
                mean_alpha_0,
                sigma_alpha_0,
                mean_log_amp_prime,
                sigma_log_amp_prime,
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
                mean_log_amp_prime,
                sigma_log_amp_prime,
                sample_correlations=False,
            )

    else:
        raise ValueError(
            f"Invalid correlation_structure '{correlation_structure}'. "
            "Options: 'mvn', 'independent'"
        )


def summarize_priors(model_structure: str, prior_config: dict, n_filt: int, n_obj: int):
    """Print a summary table of all priors used in the model.

    Parameters
    ----------
    model_structure : str
        One of ``"unpooled"``, ``"pooled"``, ``"hierarchical"``,
        ``"hierarchical_mvn"``.
    prior_config : dict
        The configuration dict (must contain ``"correlation_structure"``
        for hierarchical models).
    n_filt : int
        Number of filters.
    n_obj : int
        Number of objects (only for per-object row notes).
    """

    pop_info = prior_config.get("population_priors", None)
    prior_type = prior_config.get("prior_type", "maximum_entropy").lower()
    rise_model = prior_config.get("rise_model", "power_law")
    sample_beta = prior_config.get("sample_beta", False)

    # ── helper ──────────────────────────────────────────────────────────
    def _row(param, desc, scope):
        print(f"  {param:20s} {desc:55s} {scope}")

    print(f"\n{'=' * 80}")
    print(f"  Prior Summary — {model_structure} model".center(76))
    print(f"{'=' * 80}")

    # ── Common nuisance parameters ──────────────────────────────────────
    if model_structure in ("unpooled", "pooled", "hierarchical", "hierarchical_mvn"):
        _row("base (C)", "Uniform(-50, 50)", "[n_fcqfid]")
        if sample_beta:
            _row("beta", "log(beta) ~ HalfNormal(0.1)", "[n_fcqfid]")
        else:
            _row("beta", "fixed = 1", "[n_fcqfid]")
        if rise_model == "curved_power_law":
            mean_neg = 1 / (20 * (1 + jnp.log(20)))
            _row("alpha_1", f"Exponential(rate={1 / mean_neg:.1f})", "[n_obj x n_filt]")
        else:
            _row("alpha_1", "fixed = 0", "[n_obj x n_filt]")

    # ── Unpooled / Pooled ───────────────────────────────────────────────
    if model_structure in ("unpooled", "pooled"):
        # Per-param scopes differ between unpooled and pooled:
        #   unpooled → alpha_0 is [n_obj × n_filt]; pooled → shared [n_filt]
        #   t_rise and log_Aprime are always per-object per-filter.
        if model_structure == "unpooled":
            scope_alpha_0 = "[n_obj x n_filt]"
        else:
            scope_alpha_0 = "[n_filt] (shared)"

        # ── Copula display (all three keys + corr) ──
        copula_mode = (
            pop_info is not None
            and all(k in pop_info for k in ("t_rise", "alpha_0", "log_Aprime"))
            and pop_info.get("corr") is not None
        )

        if copula_mode:
            t_mean = pop_info["t_rise"].get("mean")
            t_sigma = pop_info["t_rise"].get("sigma")
            if t_mean is not None:
                _row("t_rise", f"Normal(μ={t_mean}, σ={t_sigma}) via Copula", "[n_obj]")
            else:
                t_min = prior_config.get("t_rise_min", EPS)
                t_max = prior_config.get("t_rise_max", 40)
                _row(
                    "t_rise", f"Uniform({t_min:.4g}, {t_max:.4g}) via Copula", "[n_obj]"
                )

            a_means = np.atleast_1d(np.array(pop_info["alpha_0"]["mean"], dtype=float))
            a_sigmas = np.atleast_1d(
                np.array(pop_info["alpha_0"]["sigma"], dtype=float)
            )
            for i in range(len(a_means)):
                label = f"alpha_0[{i}]"
                if not np.isnan(a_means[i]):
                    _row(
                        label,
                        f"Normal(μ={a_means[i]:.3g}, σ={a_sigmas[i]:.3g}) via Copula",
                        scope_alpha_0,
                    )
                else:
                    a_min = 1.0 + EPS
                    a_max = float(prior_config.get("max_alpha_0", 5))
                    _row(
                        label,
                        f"Uniform({a_min:.3g}, {a_max:.3g}) via Copula",
                        scope_alpha_0,
                    )

            l_means = np.atleast_1d(
                np.array(pop_info["log_Aprime"]["mean"], dtype=float)
            )
            l_sigmas = np.atleast_1d(
                np.array(pop_info["log_Aprime"]["sigma"], dtype=float)
            )
            logA_high = float(np.log(1000))
            for i in range(len(l_means)):
                label = f"log_Aprime[{i}]"
                if not np.isnan(l_means[i]):
                    _row(
                        label,
                        f"Normal(μ={l_means[i]:.3g}, σ={l_sigmas[i]:.3g}) via Copula",
                        "[n_obj x n_filt]",
                    )
                else:
                    _row(
                        label,
                        f"Uniform(0, {logA_high:.3g}) via Copula",
                        "[n_obj x n_filt]",
                    )

            # Correlation matrix
            corr_arr = jnp.array(pop_info["corr"])
            corr_str = np.array2string(
                np.asarray(corr_arr), precision=2, separator="  ", suppress_small=True
            )
            corr_lines = corr_str.split("\n")
            _row("corr", corr_lines[0], f"[{corr_arr.shape[0]}×{corr_arr.shape[1]}]")
            for line in corr_lines[1:]:
                print(f"  {'':20s} {line}")

        else:
            # t_rise
            pop_t = pop_info.get("t_rise") if pop_info else None
            if pop_t is not None:
                _row("t_rise", f"Normal({pop_t['mean']}, {pop_t['sigma']})", "[n_obj]")
            elif prior_config.get("prior_type", "uniform").lower() in (
                "gaussian",
                "normal",
            ):
                mt = prior_config.get("mean_t_rise", 18)
                st = prior_config.get("sigma_t_rise", 1.5)
                _row("t_rise", f"TruncatedNormal({mt}, {st})", "[n_obj]")
            else:
                _row(
                    "t_rise",
                    f"Uniform(EPS, {prior_config.get('t_rise_max', 40)})",
                    "[n_obj]",
                )

            # alpha_0
            pop_a = pop_info.get("alpha_0") if pop_info else None
            if pop_a is not None:
                _row(
                    "alpha_0",
                    f"Normal(mean={pop_a['mean']}, sigma={pop_a['sigma']})",
                    scope_alpha_0,
                )
            elif prior_type == "uniform":
                mn = prior_config.get("min_alpha_0", 1)
                mx = prior_config.get("max_alpha_0", 5)
                _row("alpha_0", f"Uniform({mn}, {mx})", scope_alpha_0)
            elif prior_type == "maximum_entropy":
                me = prior_config.get("mean_alpha_0", 2)
                se = prior_config.get("sigma_alpha_0", None)
                if se is None:
                    _row(
                        "alpha_0",
                        f"MaxEnt Exponential(rate={1 / (me - 1):.2f}) + 1",
                        scope_alpha_0,
                    )
                else:
                    _row(
                        "alpha_0",
                        "MaxEnt Gamma(concentration=..., rate=...) + 1",
                        scope_alpha_0,
                    )
            elif prior_type == "normal":
                me = prior_config.get("mean_alpha_0", 2)
                se = prior_config.get("sigma_alpha_0")
                _row("alpha_0", f"TruncatedNormal({me}, {se})", scope_alpha_0)
            elif prior_type == "miller":
                _row("alpha_0", "Exponential(log(10))", scope_alpha_0)

            # log_Aprime / Aprime
            pop_l = pop_info.get("log_Aprime") if pop_info else None
            if pop_l is not None:
                _row(
                    "log_Aprime",
                    f"Normal(mean={pop_l['mean']}, sigma={pop_l['sigma']})",
                    "[n_obj x n_filt]",
                )
            else:
                _row("log_Aprime", "Uniform(0, log(1000))", "[n_obj x n_filt]")

            # population prior summary
            if pop_info:
                corr_tag = " with correlations" if pop_info.get("corr") else ""
                specified = [
                    k for k in ("t_rise", "alpha_0", "log_Aprime") if k in pop_info
                ]
                _row(
                    "pop. prior",
                    f"{' + '.join(specified)}{corr_tag}",
                    "replaces default",
                )
            else:
                _row("pop. prior", "none", "")

    # ── Hierarchical ────────────────────────────────────────────────────
    elif model_structure in ("hierarchical", "hierarchical_mvn"):
        corr_struct = prior_config.get("correlation_structure", "mvn")
        d = 1 + 2 * n_filt

        print("\n  Hyperpriors:")
        _row("mean_t_rise", "Uniform(5.0, 35.0)", "")
        _row("sigma_t_rise", "HalfCauchy(1.5)", "")
        _row("mean_alpha_0", "Uniform(1+EPS, 4)", "[n_filt]")
        _row("sigma_alpha_0", "HalfCauchy(0.3)", "[n_filt]")
        _row("mean_log_Aprime", "Uniform(0, log(1000))", "[n_filt]")
        _row("sigma_log_Aprime", "HalfCauchy(0.5)", "[n_filt]")

        if corr_struct == "mvn":
            _row("chol_corr", f"LKJCholesky({d}, concentration=1.0)", "")
        else:
            _row("corr", "none (diagonal)", "")

        print("\n  Per-object conditionals:")
        tag = f"MVN({d}-dim, µ, Σ)" if corr_struct == "mvn" else "independent Normals"
        _row("theta", tag, "[n_obj]")
        _row("t_rise", "TruncatedNormal(µ₁, σ₁, low=EPS)", "[n_obj]")
        _row("alpha_0", "TruncatedNormal(µ₂₋₃, σ₂₋₃, low=1+EPS)", "[n_obj x n_filt]")
        _row("log_Aprime", "Normal(µ₄₋₅, σ₄₋₅)", "[n_obj x n_filt]")

    print(f"{'=' * 80}\n")


def build_population_informed_params(prior_config: dict, n_filt: int) -> dict:
    """
    Parse ``population_priors`` from *prior_config* and return a structured
    dict that describes how the unpooled model should set its per-object
    priors.

    Parameters
    ----------
    prior_config : dict
        Configuration dict (may contain ``population_priors`` key).
    n_filt : int
        Number of filters (required for expanding per-filter params).

    Returns
    -------
    result : dict
        One of four forms:

        - ``{"type": "none"}`` -- no population priors specified; caller
          should fall through to existing flat priors.
        - ``{"type": "independent",
            "params": {"t_rise": {"mean": scalar, "sigma": scalar}, ...}}``
          -- independent truncated-Normal priors per parameter.
        - ``{"type": "mvn",
            "mu": jnp.ndarray, "L": jnp.ndarray,
            "param_order": [(name, is_per_filter, n_elements), ...]}``
          -- joint multivariate-Normal prior over all specified parameters.
        - ``{"type": "copula",
            "L_corr": jnp.ndarray, "specs": list[dict], "n_filt": int}``
          -- Gaussian copula with mixed normal/uniform marginals.
    """
    pop = prior_config.get("population_priors", None)
    if not pop:
        return {"type": "none"}

    # ── Copula mode: all three param keys present + correlation matrix ──
    has_all_keys = all(k in pop for k in ("t_rise", "alpha_0", "log_Aprime"))
    if has_all_keys and pop.get("corr") is not None:
        return _build_copula_params(pop, prior_config, n_filt)

    # ── Legacy modes: independent or MVN over specified params ──
    # Collect specified parameters in canonical order: scalar first, then
    # per-filter parameters expanded across filters.
    param_order = []  # list of (name, is_per_filter, n_elements)
    means_flat = []
    sigmas_flat = []

    def _add_scalar(name):
        spec = pop.get(name)
        if spec is None:
            return
        param_order.append((name, False, 1))
        means_flat.append(spec["mean"])
        sigmas_flat.append(spec["sigma"])

    def _add_per_filter(name):
        spec = pop.get(name)
        if spec is None:
            return
        param_order.append((name, True, n_filt))
        means_flat.extend(spec["mean"])
        sigmas_flat.extend(spec["sigma"])

    _add_scalar("t_rise")
    _add_per_filter("alpha_0")
    _add_per_filter("log_Aprime")

    if not param_order:
        return {"type": "none"}

    mu = jnp.array(means_flat, dtype=float)
    sigma = jnp.array(sigmas_flat, dtype=float)
    n_mvn_dim = len(mu)

    corr_raw = pop.get("corr", None)

    if corr_raw is not None:
        corr = jnp.array(corr_raw, dtype=float)
        cov = jnp.outer(sigma, sigma) * corr
        L = jnp.linalg.cholesky(cov)
        return {
            "type": "mvn",
            "mu": mu,
            "L": L,
            "param_order": param_order,
        }
    else:
        # Independent: return per-param means and sigmas
        params = {}
        idx = 0
        for name, is_per_filter, n_elem in param_order:
            if is_per_filter:
                params[name] = {
                    "mean": mu[idx : idx + n_elem],
                    "sigma": sigma[idx : idx + n_elem],
                }
            else:
                params[name] = {
                    "mean": mu[idx],
                    "sigma": sigma[idx],
                }
            idx += n_elem
        return {"type": "independent", "params": params}


def _build_copula_params(pop: dict, prior_config: dict, n_filt: int) -> dict:
    """Build a Gaussian-copula population prior over all dims (mixed normal/uniform marginals)."""

    n_filt_config = len(list(pop["alpha_0"]["mean"]))
    if n_filt != n_filt_config:
        raise ValueError(
            f"n_filt mismatch: config has {n_filt_config}, data has {n_filt}"
        )

    specs = []

    # ── t_rise (1 scalar dim) ──
    t_mean = pop["t_rise"].get("mean")
    t_sigma = pop["t_rise"].get("sigma")
    if t_mean is not None:
        specs.append({"type": "normal", "mean": float(t_mean), "sigma": float(t_sigma)})
    else:
        t_min = prior_config.get("t_rise_min", EPS)
        t_max = prior_config.get("t_rise_max", 40)
        specs.append({"type": "uniform", "low": float(t_min), "high": float(t_max)})

    # ── alpha_0 (n_filt dims) ──
    alpha_means = list(pop["alpha_0"]["mean"])
    alpha_sigmas = list(pop["alpha_0"]["sigma"])
    for m, s in zip(alpha_means, alpha_sigmas):
        if m is not None:
            specs.append({"type": "normal", "mean": float(m), "sigma": float(s)})
        else:
            a_min = 1.0 + EPS
            a_max = float(prior_config.get("max_alpha_0", 5))
            specs.append({"type": "uniform", "low": a_min, "high": a_max})

    # ── log_Aprime (n_filt dims) ──
    logA_means = list(pop["log_Aprime"]["mean"])
    logA_sigmas = list(pop["log_Aprime"]["sigma"])
    logA_high = float(np.log(1000))
    for m, s in zip(logA_means, logA_sigmas):
        if m is not None:
            specs.append({"type": "normal", "mean": float(m), "sigma": float(s)})
        else:
            specs.append({"type": "uniform", "low": 0.0, "high": logA_high})

    # ── Correlation matrix ──
    corr_raw = pop.get("corr")
    corr = jnp.array(corr_raw, dtype=float)
    if corr.shape != (len(specs), len(specs)):
        raise ValueError(
            f"Corr matrix shape {corr.shape} does not match "
            f"the total number of dimensions ({len(specs)}). "
        )
    L_corr = jnp.linalg.cholesky(corr)

    return {
        "type": "copula",
        "L_corr": L_corr,
        "specs": specs,
        "n_filt": n_filt,
    }
